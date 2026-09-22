"""Fresh, setting-specific GFE and Conventional endpoint charts.

B1/B2 track two observed GFE references selected by the farthest root pair.
Six independent cold starts at every target supply the observed reference bank;
nearest-neighbor projector assignment tracks two displayed references without
asserting two endpoint classes. Conventional retains two fixed cold seeds and
never contributes to reference selection or the frozen MDS fit. No archived
phases, target continuation, or prescribed phase banks enter the experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from hat_revision_pipeline import pipeline as p
from hat_revision_pipeline import main_feg as mf
from hat_revision_pipeline import final_figures as ff
from hat_revision_pipeline.branch_figures import (
    ComponentBank, pairwise_projective_distance, projective_similarity,
    target_demodulated_states,
)
from hat_revision_pipeline.cache import digest_array
from hat_revision_pipeline.sota import solve_corrected_gorkov_fe

ITERATION_CAP = 10_000
CHECKPOINTS = (100, 1_000, 10_000)
FAMILIES = ("B1", "B2")
CONTRACT = "fresh-generality-cold-seed-chart-v1"
REFERENCE_COLLAPSE_DISTANCE = 1.0e-3


def _states(ctx, phases, table):
    return target_demodulated_states(phases, table,
        positions_m=ctx.positions_m, frequency_hz=ctx.config.frequency_hz)


@dataclass
class FrozenFrame:
    """Positive MDS axes padded only for display if the observed rank is low."""
    landmark_table: pd.DataFrame
    landmark_states: np.ndarray
    coordinates: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    column_mean: np.ndarray
    grand_mean: float
    rotation: np.ndarray
    stress: float

    def project(self, states):
        squared = 1.0 - projective_similarity(states, self.landmark_states) ** 2
        cross = -0.5 * (squared - squared.mean(axis=1, keepdims=True)
                        - self.column_mean[None, :] + self.grand_mean)
        raw = np.zeros((len(squared), 3))
        if len(self.eigenvalues):
            raw[:, :len(self.eigenvalues)] = (cross @ self.eigenvectors
                / np.sqrt(self.eigenvalues)[None, :])
        return raw @ self.rotation


def _fit_frame(bank):
    # Fit every observed GFE endpoint, including states outside displayed meshes.
    states = bank.endpoint_states
    table = bank.endpoint_table
    distance = pairwise_projective_distance(states)
    squared = distance ** 2
    center = np.eye(len(states)) - np.ones((len(states), len(states))) / len(states)
    gram = -0.5 * center @ squared @ center
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    order = np.argsort(values)[::-1]
    threshold = max(float(values[order[0]]) * 1e-12, 1e-14)
    selected = [int(index) for index in order[:3] if values[index] > threshold]
    eigenvalues, eigenvectors = values[selected], vectors[:, selected]
    raw = np.zeros((len(states), 3))
    raw[:, :len(selected)] = eigenvectors * np.sqrt(eigenvalues)[None, :]
    reference = np.column_stack([
        table.xi, table.eta, table.display_reference_axis])
    reference -= reference.mean(axis=0)
    reference *= np.linalg.norm(raw) / max(np.linalg.norm(reference), 1e-30)
    rotation, _ = orthogonal_procrustes(raw, reference)
    coordinates = raw @ rotation
    reconstructed = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    stress = math.sqrt(float(np.sum((distance - reconstructed) ** 2))
                       / max(float(np.sum(distance ** 2)), 1e-30))
    return FrozenFrame(table.copy(), states.copy(), coordinates,
        eigenvectors, eigenvalues, squared.mean(axis=0), float(squared.mean()), rotation, stress)


def _solve(ctx, method, target, seed):
    objective = (mf.feg_single_objective(ctx, target) if method == "GFE"
                 else p._objective_at(ctx, "Conventional", target))
    initial = np.random.default_rng(int(seed)).uniform(-np.pi, np.pi, objective.n_transducers)
    payload = dict(contract=CONTRACT, method=method,
        objective_method=asdict(objective.method),
        force_target_n=np.asarray(objective.force_target_n).tolist(),
        target_m=np.asarray(target).tolist(), seed=int(seed),
        initial_phase_content_id=digest_array(initial),
        positions_content_id=digest_array(ctx.positions_m), normals_content_id=digest_array(ctx.normals),
        frequency_hz=float(ctx.config.frequency_hz),
        maxiter=ITERATION_CAP, gtol=float(ctx.config.gtol))
    result, status, _ = ctx.cache.get_or_compute("generality_cold_branch_run", payload,
        lambda: solve_corrected_gorkov_fe(objective, initial,
            maxiter=ITERATION_CAP, gtol=ctx.config.gtol), recompute=ctx.recompute)
    return result, status


def _row(ctx, method, target, ix, iy, seed, run, status):
    return dict(method=method, objective_method=method, target_id=f"xy_{ix}_{iy}",
        target_x_m=float(target[0]), target_y_m=float(target[1]), target_z_m=float(target[2]),
        ix=int(ix), iy=int(iy), xi=float(ix - 1), eta=float(iy - 1), seed=int(seed),
        iterations=int(run.iterations), evaluations=int(run.evaluations),
        objective=float(run.objective), gradient_norm=float(run.gradient_norm),
        wall_time_s=float(run.history_wall_s[-1]), optimizer_success=bool(run.success),
        optimizer_status=int(run.status), optimizer_message=str(run.message),
        iteration_cap=ITERATION_CAP, phase_content_id=digest_array(run.phase_rad),
        cache_status=status, frequency_hz=float(ctx.config.frequency_hz),
        initialization="independent cold random phase; same seed across targets",
        component_semantics="observed root reference or associated fixed Conventional cold seed")


def _snapshot(run, budget):
    index = min(int(budget), int(run.iterations))
    history = np.asarray(run.history_phase_rad)
    # History index zero is the initial command; callback i is accepted step i.
    phase = np.asarray(run.phase_rad) if budget >= run.iterations else history[index]
    return phase, index


def _track_references(observed_table, observed_states, tracked):
    """Assign two distinct observed samples per target along the physical grid.

All candidates remain in the exported bank. Distinct samples may represent the
same endpoint; their measured separation records reference-pair collapse.
The central cross is matched to the root and each corner to its two previously
assigned neighbors. Conventional states never influence these assignments.
"""
    selected = {}
    diagnostics, candidates = [], []
    nodes = sorted(((ix, iy) for iy in range(3) for ix in range(3)),
                   key=lambda node: (abs(node[0] - 1) + abs(node[1] - 1), node[1], node[0]))
    for ix, iy in nodes:
        local = observed_table.index[observed_table.ix.eq(ix) & observed_table.iy.eq(iy)].to_numpy(int)
        neighbors = [(jx, jy) for jx, jy in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1))
                     if (jx, jy) in selected]
        if (ix, iy) == (1, 1):
            pair = tuple(int(observed_table.index[observed_table.ix.eq(ix)
                & observed_table.iy.eq(iy) & observed_table.seed.eq(tracked[family])][0])
                for family in FAMILIES)
            costs = np.zeros((2, len(local)))
            joint_cost, margin = 0.0, np.nan
        else:
            costs = np.zeros((2, len(local)))
            for family_index in range(2):
                references = np.stack([observed_states[selected[node][family_index]] for node in neighbors])
                distance = np.sqrt(np.maximum(0., 1. - projective_similarity(observed_states[local], references) ** 2))
                costs[family_index] = distance.mean(axis=1)
            ordered = sorted((float(costs[0, a] + costs[1, b]), a, b)
                for a in range(len(local)) for b in range(len(local)) if a != b)
            joint_cost, a, b = ordered[0]
            pair = (int(local[a]), int(local[b]))
            margin = float(ordered[1][0] - joint_cost)
            for cost, a_index, b_index in ordered:
                candidates.append(dict(ix=ix, iy=iy,
                    observed_index_B1=int(local[a_index]), observed_index_B2=int(local[b_index]),
                    seed_B1=int(observed_table.loc[local[a_index], "seed"]),
                    seed_B2=int(observed_table.loc[local[b_index], "seed"]),
                    assignment_cost=cost, selected=bool((int(local[a_index]), int(local[b_index])) == pair)))
        selected[(ix, iy)] = pair
        separation = float(pairwise_projective_distance(observed_states[list(pair)])[0, 1])
        for family_index, family in enumerate(FAMILIES):
            position = int(np.flatnonzero(local == pair[family_index])[0])
            diagnostics.append(dict(ix=ix, iy=iy, component=family,
                observed_state_index=pair[family_index],
                selected_seed=int(observed_table.loc[pair[family_index], "seed"]),
                root_reference_seed=int(tracked[family]), neighbor_count=len(neighbors),
                mean_projector_distance_to_assigned_neighbors=float(costs[family_index, position]),
                joint_assignment_cost=joint_cost, assignment_cost_margin=margin,
                reference_pair_projector_distance=separation,
                reference_pair_collapsed=bool(separation < REFERENCE_COLLAPSE_DISTANCE),
                collapse_distance_threshold=REFERENCE_COLLAPSE_DISTANCE))
    return selected, pd.DataFrame(diagnostics), pd.DataFrame(candidates)


def fixed_feg_branch_pack(ctx):
    key = "generality_fresh_branch_pack_v2"
    if key in ctx.memo:
        return ctx.memo[key]
    root = np.asarray(p.MAIN_TARGET_M, dtype=float)
    seeds = [int(ctx.config.random_seed) + index for index in range(6)]
    solved = {}
    root_rows = []
    for method in ("GFE", "Conventional"):
        for seed in seeds:
            run, status = _solve(ctx, method, root, seed)
            solved[(method, 1, 1, seed)] = (run, status)
            root_rows.append(_row(ctx, method, root, 1, 1, seed, run, status))
        print(f"Branch root: {method}, six cold starts complete", flush=True)
    root_table = pd.DataFrame(root_rows)
    root_gfe_table = root_table[root_table.method.eq("GFE")].reset_index(drop=True)
    root_gfe_phases = np.stack([solved[("GFE", 1, 1, seed)][0].phase_rad for seed in seeds])
    root_gfe_states = _states(ctx, root_gfe_phases, root_gfe_table)
    root_distances = pairwise_projective_distance(root_gfe_states)
    upper = np.triu(root_distances, k=1)
    first, second = np.unravel_index(np.argmax(upper), upper.shape)
    if first == second:
        first, second = 0, 1
    tracked = dict(zip(FAMILIES, [seeds[int(first)], seeds[int(second)]]))
    observed_rows, observed_results = [], []
    trajectory_rows, trajectory_results = [], []
    # The root audit retains every observed start, including untracked families.
    for seed in seeds:
        run, status = solved[("Conventional", 1, 1, seed)]
        row = _row(ctx, "Conventional", root, 1, 1, seed, run, status)
        row.update(trajectory_index=len(trajectory_rows),
                   component=next((family for family, value in tracked.items() if value == seed), "untracked"))
        trajectory_rows.append(row)
        trajectory_results.append(run)
    trajectory_lookup = {(1, 1, int(row["seed"])): index for index, row in enumerate(trajectory_rows)}
    for iy, dy in enumerate(np.linspace(-0.001, 0.001, 3)):
        for ix, dx in enumerate(np.linspace(-0.001, 0.001, 3)):
            target = root + np.array([dx, dy, 0.0])
            for seed in seeds:
                identity = ("GFE", ix, iy, seed)
                if identity not in solved:
                    solved[identity] = _solve(ctx, "GFE", target, seed)
                run, status = solved[identity]
                row = _row(ctx, "GFE", target, ix, iy, seed, run, status)
                row.update(state_index=len(observed_rows), observed_state_index=len(observed_rows),
                    source_endpoint_position=len(observed_rows), component="unselected")
                observed_rows.append(row)
                observed_results.append(run)
            for component, seed in tracked.items():
                identity = ("Conventional", ix, iy, seed)
                if identity not in solved:
                    solved[identity] = _solve(ctx, "Conventional", target, seed)
                run, status = solved[identity]
                if (ix, iy, seed) not in trajectory_lookup:
                    row = _row(ctx, "Conventional", target, ix, iy, seed, run, status)
                    row.update(component=component, trajectory_index=len(trajectory_rows))
                    trajectory_lookup[(ix, iy, seed)] = len(trajectory_rows)
                    trajectory_rows.append(row)
                    trajectory_results.append(run)
            print(f"Branch chart target {ix},{iy}: six GFE cold starts and two C trajectories complete", flush=True)
    observed_table = pd.DataFrame(observed_rows)
    observed_phases = np.stack([run.phase_rad for run in observed_results])
    observed_states = _states(ctx, observed_phases, observed_table)
    selected, assignment_table, assignment_candidates = _track_references(observed_table, observed_states, tracked)
    anchor_rows, anchor_results = [], []
    for iy in range(3):
        for ix in range(3):
            for family_index, family in enumerate(FAMILIES):
                observed_index = selected[(ix, iy)][family_index]
                row = observed_table.iloc[observed_index].to_dict()
                row.update(component=family, root_reference_seed=tracked[family],
                    reference_selection="nearest-neighbor one-to-one projector assignment")
                observed_table.loc[observed_index, "component"] = family
                anchor_rows.append(row)
                anchor_results.append(observed_results[observed_index])
    anchor_table = pd.DataFrame(anchor_rows)
    anchor_phases = np.stack([run.phase_rad for run in anchor_results])
    anchor_states = _states(ctx, anchor_phases, anchor_table)
    edges = pd.DataFrame([dict(a_ix=ix, a_iy=iy, b_ix=jx, b_iy=jy)
        for iy in range(3) for ix in range(3)
        for jx, jy in ((ix + 1, iy), (ix, iy + 1)) if jx < 3 and jy < 3])
    root_reference_states = np.stack([observed_states[index] for index in selected[(1, 1)]])
    root_overlap = projective_similarity(observed_states, root_reference_states)
    observed_table["display_reference_axis"] = root_overlap[:, 1] - root_overlap[:, 0]
    bank = ComponentBank(observed_table.copy(), observed_states.copy(), anchor_table.copy(),
        anchor_states.copy(), edges, (1, 1))
    frame = _fit_frame(bank)
    anchors = anchor_table.copy()
    anchors[["mds1", "mds2", "mds3"]] = frame.project(anchor_states)
    observed_coordinates = observed_table.copy()
    observed_coordinates[["mds1", "mds2", "mds3"]] = frame.coordinates
    trajectories = pd.DataFrame(trajectory_rows)
    snapshots = np.stack([np.stack([_snapshot(run, budget)[0] for budget in CHECKPOINTS])
                          for run in trajectory_results])
    repeated = trajectories.loc[trajectories.index.repeat(len(CHECKPOINTS))].reset_index(drop=True)
    trajectory_states = _states(ctx, snapshots.reshape(-1, len(ctx.positions_m)), repeated).reshape(snapshots.shape)
    evolution_rows = []
    for anchor in anchor_table.itertuples(index=False):
        index = trajectory_lookup[(int(anchor.ix), int(anchor.iy), int(tracked[anchor.component]))]
        states = trajectory_states[index]
        coordinates = frame.project(states)
        local_indices = observed_table.index[observed_table.ix.eq(anchor.ix)
            & observed_table.iy.eq(anchor.iy)].to_numpy(int)
        terminal_similarities = projective_similarity(states[-1:], observed_states[local_indices]).ravel()
        matched_index = int(local_indices[int(np.argmax(terminal_similarities))])
        reference = observed_states[matched_index]
        similarity = projective_similarity(states, reference).ravel()
        displayed = bank.target_medoid(int(anchor.ix), int(anchor.iy), anchor.component)
        displayed_similarity = projective_similarity(states, displayed).ravel()
        for checkpoint_index, budget in enumerate(CHECKPOINTS):
            sim = float(similarity[checkpoint_index])
            row = trajectories.iloc[index].to_dict()
            row.update(component=anchor.component, iteration=budget,
                accepted_iteration=min(budget, int(trajectory_results[index].iterations)),
                stopped_before_budget=bool(trajectory_results[index].iterations < budget),
                selected_trajectory_index=index, checkpoint_index=checkpoint_index,
                mds1=float(coordinates[checkpoint_index, 0]),
                mds2=float(coordinates[checkpoint_index, 1]),
                mds3=float(coordinates[checkpoint_index, 2]),
                similarity_to_matched_fe=sim, similarity_to_matched_feg=sim,
                distance_to_matched_fe=math.sqrt(max(0., 1. - sim ** 2)),
                distance_to_matched_feg=math.sqrt(max(0., 1. - sim ** 2)),
                similarity_to_displayed_reference=float(displayed_similarity[checkpoint_index]),
                distance_to_displayed_reference=math.sqrt(max(0., 1. - float(displayed_similarity[checkpoint_index]) ** 2)),
                matched_gfe_observed_state_index=matched_index,
                matched_gfe_seed=int(observed_table.loc[matched_index, "seed"]),
                matched_gfe_reference_family=str(observed_table.loc[matched_index, "component"]),
                matched_gfe_outside_two_references=bool(observed_table.loc[matched_index, "component"] == "unselected"),
                reference_method="GFE", correspondence_rule="nearest of six local GFE endpoints at C termination; fixed over whole trajectory")
            evolution_rows.append(row)
    evolution = pd.DataFrame(evolution_rows)
    root_c_phases = np.stack([solved[("Conventional", 1, 1, seed)][0].phase_rad for seed in seeds])
    root_c_states = _states(ctx, root_c_phases, root_gfe_table)
    root_cross = projective_similarity(root_c_states, root_gfe_states)
    correlation_rows = []
    for index, seed in enumerate(seeds):
        matched_index = int(np.argmax(root_cross[index]))
        sim = float(root_cross[index, matched_index])
        correlation_rows.append(dict(trajectory_index=index, seed=seed, target_id="xy_1_1",
            component=next((family for family, value in tracked.items() if value == seed), "untracked"),
            objective=float(solved[("Conventional", 1, 1, seed)][0].objective),
            similarity_to_matched_fe=sim, similarity_to_matched_feg=sim,
            distance_to_matched_fe=math.sqrt(max(0., 1. - sim ** 2)),
            distance_to_matched_feg=math.sqrt(max(0., 1. - sim ** 2)),
            same_seed_gfe_similarity=float(root_cross[index, index]),
            matched_gfe_seed=int(seeds[matched_index]),
            matched_gfe_reference_family=next((family for family, value in tracked.items() if value == seeds[matched_index]), "unselected"),
            maximum_similarity_to_six_observed_gfe_endpoints=float(root_cross[index].max()),
            reference_method="GFE", correspondence_rule="nearest of six observed GFE root endpoints"))
    correlation = pd.DataFrame(correlation_rows)
    protocol = pd.DataFrame([dict(contract=CONTRACT, analysis_contract="six-local-endpoints-neighbor-tracking-v2", root_start_count=6,
        tracked_seed_B1=tracked["B1"], tracked_seed_B2=tracked["B2"], chart_points=9,
        observed_root_farthest_projector_distance=float(root_distances[first, second]),
        tracked_family_count=2, endpoint_class_count="not inferred",
        family_selection="farthest root pair; nearest-neighbor one-to-one projector assignment across targets",
        gfe_starts_per_target=6, observed_gfe_endpoint_count=len(observed_table),
        conventional_chart_seeds_per_target=2, conventional_run_count=len(trajectory_results),
        reference_pair_collapse_threshold=REFERENCE_COLLAPSE_DISTANCE,
        reference_pair_collapsed_target_count=int(assignment_table.drop_duplicates(["ix", "iy"]).reference_pair_collapsed.sum()),
        conventional_matching="nearest of six local GFE endpoints at terminal state; same match at all checkpoints",
        target_half_width_m=.001, target_z_m=float(root[2]), iteration_cap=ITERATION_CAP,
        checkpoints=",".join(map(str, CHECKPOINTS)), warm_start=False,
        archived_phase_reuse=False, conventional_in_frame_fit=False,
        frame="all 54 observed GFE endpoints; fixed projective MDS", frame_rank=len(frame.eigenvalues),
        frame_stress=frame.stress, reference_method="GFE")])
    ctx.branch_bank = bank
    ctx.conventional_evolution = evolution
    ctx.artifact = SimpleNamespace(anchor_table=observed_table.assign(method="Force-Equilibrium"),
        anchor_phases=observed_phases, trajectory_table=trajectories,
        checkpoints=np.asarray(CHECKPOINTS), trajectory_snapshots=snapshots,
        trajectory_states=lambda: trajectory_states)
    table_items = {"fig2_fixed_single_feg_branch_anchors": anchors,
        "fig2_fixed_feg_conventional_evolution": evolution,
        "fig2_fixed_feg_continuation_protocol": protocol,
        "fig2_six_root_start_observations": root_table,
        "fig2_root_endpoint_correspondence": correlation,
        "fig2_all_observed_gfe_endpoints": observed_coordinates,
        "fig2_reference_assignments": assignment_table,
        "fig2_reference_assignment_candidates": assignment_candidates,
        "fig2_conventional_trajectory_inventory": trajectories}
    for name, table in table_items.items():
        ctx.tables[name] = table.copy()
    destination = Path(ctx.output_root) / "branch_data"
    destination.mkdir(parents=True, exist_ok=True)
    obsolete_name = "fig2_root_same_seed_correspondence"
    ctx.tables.pop(obsolete_name, None)
    for folder in (destination, Path(ctx.output_root) / "tables"):
        (folder / f"{obsolete_name}.csv").unlink(missing_ok=True)
    for name, table in table_items.items():
        table.to_csv(destination / f"{name}.csv", index=False)
    np.savez_compressed(destination / "fresh_branch_arrays.npz",
        positions_m=ctx.positions_m, normals=ctx.normals,
        frequency_hz=np.asarray(ctx.config.frequency_hz), root_seeds=np.asarray(seeds),
        root_initial_phases=np.stack([np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ctx.positions_m)) for seed in seeds]),
        root_gfe_phases=root_gfe_phases, root_conventional_phases=root_c_phases,
        root_gfe_projective_distances=root_distances, root_c_gfe_similarities=root_cross,
        anchor_phases=anchor_phases, anchor_states=anchor_states,
        anchor_coordinates=anchors[["mds1", "mds2", "mds3"]].to_numpy(float),
        all_gfe_phases=observed_phases, all_gfe_states=observed_states,
        all_gfe_coordinates=frame.coordinates,
        conventional_snapshots=snapshots, checkpoints=np.asarray(CHECKPOINTS),
        mds_eigenvalues=frame.eigenvalues, mds_eigenvectors=frame.eigenvectors,
        mds_rotation=frame.rotation, mds_column_mean=frame.column_mean,
        mds_grand_mean=np.asarray(frame.grand_mean))
    (destination / "protocol.json").write_text(json.dumps(protocol.iloc[0].to_dict(), indent=2))
    value = dict(bank=bank, frame=frame, anchors=anchors, anchor_phases=anchor_phases,
        anchor_results=tuple(anchor_results), evolution=evolution, correlation=correlation,
        protocol=protocol, root_observations=root_table,
        all_observed_gfe_endpoints=observed_coordinates,
        reference_assignments=assignment_table,
        trajectory_results=tuple(trajectory_results), trajectory_table=trajectories,
        tracked_seeds=tracked, root_conventional_runs={seed: solved[("Conventional", 1, 1, seed)][0] for seed in seeds})
    ctx.memo[key] = value
    return value


def _conventional_decision_run(ctx, branch):
    return branch["root_conventional_runs"][branch["tracked_seeds"]["B1"]]


def install(ctx):
    """Install lazy adapters; constructing a context performs no branch solves."""
    mf.fixed_feg_branch_pack = fixed_feg_branch_pack
    mf._conventional_decision_run = _conventional_decision_run
    return ctx


def figure_2(ctx):
    """Retain the original small-marker webs for three accepted-step budgets."""
    stem = "Figure_2_two_branch_evolution"
    if stem in ctx.figures:
        return ctx.figures[stem]
    pack = fixed_feg_branch_pack(ctx)
    anchors, evolution, correlation = pack["anchors"], pack["evolution"], pack["correlation"]
    observed = pack["all_observed_gfe_endpoints"]
    fig = plt.figure(figsize=(13.4, 10.2))
    grid = fig.add_gridspec(3, 3, height_ratios=(1., 1., .65), hspace=.34, wspace=.19)
    panel_specs = []
    for row, family in enumerate(FAMILIES):
        fixed = anchors[anchors.component.eq(family)]
        moving = evolution[evolution.component.eq(family)]
        matches = observed.loc[moving.matched_gfe_observed_state_index.unique()]
        visible = pd.concat([fixed, moving, matches], ignore_index=True)
        limits = []
        for column in ("mds1", "mds2", "mds3"):
            low, high = float(visible[column].min()), float(visible[column].max())
            pad = .065 * max(high - low, 1e-5)
            limits.append((low - pad, high + pad))
        for col, budget in enumerate(CHECKPOINTS):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            panel_specs.append(grid[row, col])
            faint = observed[~observed.component.eq(family)]
            points = ax.scatter(faint.mds1, faint.mds2, faint.mds3,
                s=8., color="#007C78", alpha=.17, linewidths=0., depthshade=False)
            points._manuscript_preserve_marker_area = True
            ff._draw_target_web_3d(ax, moving[moving.iteration.eq(budget)], pack["bank"].edge_table,
                color=ff.COLORS["Conventional"], marker="o", line_width=2.1,
                line_alpha=.24, marker_size=16.)
            extra_matches = matches[~matches.component.eq(family)]
            points = ax.scatter(extra_matches.mds1, extra_matches.mds2, extra_matches.mds3,
                s=12., marker="s", facecolors="none", edgecolors="#007C78",
                alpha=.65, linewidths=.7, depthshade=False, zorder=4)
            points._manuscript_preserve_marker_area = True
            ff._draw_target_web_3d(ax, fixed, pack["bank"].edge_table,
                color=ff.COLORS.get("GFE", "#007C78"), marker="s", line_width=1.6,
                line_alpha=.28, marker_size=12.)
            ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
            ax.view_init(*( (25,165) if row == 0 else (10,-165) ))
            ff._style_branch_axis_3d(ax)
            ax.set_xlabel(r"$q_1$", labelpad=-1, fontsize=9)
            ax.set_ylabel(r"$q_2$", labelpad=-1, fontsize=9)
            ax.set_zlabel(r"$q_3$", labelpad=-2, fontsize=9)
            if row == 0:
                ax.set_title(f"{budget:,} iterations", pad=2, fontsize=10)
            if col == 0:
                ax.text2D(-.13,.50,f"Root reference {row+1}",transform=ax.transAxes,
                    fontsize=9,rotation=90,ha="center",va="center")
    axis = fig.add_subplot(grid[2, :])
    panel_specs.append(grid[2, :])
    axis.scatter(correlation.similarity_to_matched_feg, correlation.objective,
        color="#334C67", s=35, edgecolors="white", linewidth=.65)
    axis.set_xlabel(r"Nearest sampled GFE similarity $|u_{\rm C}^{H}u_{\rm GFE}|$",fontsize=10)
    axis.set_ylabel(r"$J_{\mathrm{C}}$")
    axis.margins(x=.07, y=.09)
    fig.legend(handles=[mpl.lines.Line2D([0],[0],color=ff.COLORS["Conventional"],marker="o",
        markerfacecolor="white",lw=2,label="Conventional"),
        mpl.lines.Line2D([0],[0],color=ff.COLORS.get("GFE", "#007C78"),marker="s",
        markerfacecolor="white",lw=1.6,label="GFE")],
        frameon=False,loc="lower center",bbox_to_anchor=(.5,.010),ncol=2)
    fig.subplots_adjust(left=.080,right=.925,top=.935,bottom=.155)
    ff.outside_grid_panel_labels(fig, tuple(panel_specs), tuple("abcdefg"))
    for artist in fig.texts:
        if artist.get_text() in {f"({letter})" for letter in "abcdefg"}:
            artist.set_fontsize(12)
    ctx.tables["fig2_full_dimensional_correlation"] = correlation.copy()
    ctx.tables["fig2_representative_target_correlation"] = correlation.copy()
    rho = (float(correlation.objective.rank().corr(correlation.similarity_to_matched_feg.rank()))
           if correlation.objective.nunique() > 1 and correlation.similarity_to_matched_feg.nunique() > 1 else np.nan)
    ctx.tables["fig2_full_dimensional_correlation_summary"] = pd.DataFrame([dict(
        scope="six shared cold starts at root", n=len(correlation),
        spearman_rho_objective_vs_similarity=rho,
        correspondence_rule="nearest of six GFE root endpoints; no endpoint class count inferred")])
    ctx.tables["fig2_full_dimensional_distance_summary"] = evolution.groupby(["component", "iteration"]).agg(
        n=("distance_to_matched_feg", "size"),
        median_projector_distance=("distance_to_matched_feg", "median"),
        q25_projector_distance=("distance_to_matched_feg", lambda x: x.quantile(.25)),
        q75_projector_distance=("distance_to_matched_feg", lambda x: x.quantile(.75))).reset_index()
    # This review plate has three 3-D columns. Keep its declared width and
    # typography; the inherited 180-mm font floor otherwise overlaps labels.
    fig._manuscript_layout_applied = True
    return ff._save(ctx, fig, stem)


__all__ = ["install", "fixed_feg_branch_pack", "figure_2", "ITERATION_CAP", "CHECKPOINTS"]
