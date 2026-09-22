"""Continue the two deposited FE branches to the fixed Single alpha=10 objective.

The alpha=9 archive is retained as source provenance only.  Its two
target-local medoid phase commands provide branch-identified initial states;
each is independently reoptimized with the current correct-sign FE objective,
alpha=10, and a 10,000-iteration cap.  Conventional trajectories are then
placed out of sample in one FE-only three-dimensional projective-MDS frame.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import math
import os
from typing import Any

import numpy as np
import pandas as pd

from . import pipeline as p
from .branch_figures import (
    COMPONENTS,
    ComponentBank,
    fit_fixed_3d_frame,
    pairwise_projective_distance,
    projective_similarity,
    target_demodulated_states,
)
from .cache import digest_array
from .sota import solve_corrected_gorkov_fe


FIXED_ALPHA_PER_M = 10.0
ITERATION_CAP = 10_000


def _continue_one(
    ctx: p.PipelineContext,
    source_row: pd.Series,
    source_phase: np.ndarray,
) -> tuple[dict[str, Any], Any]:
    target = source_row[["target_x_m", "target_y_m", "target_z_m"]].to_numpy(dtype=float)
    objective = p._objective_at(ctx, "Force-Equilibrium", target)
    if not np.isclose(objective.method.alpha_per_m, FIXED_ALPHA_PER_M, rtol=0.0, atol=0.0):
        raise RuntimeError("The production Single FE objective is not fixed at alpha=10")
    payload = {
        "contract": "alpha10-two-branch-medoid-continuation-v1",
        "source_phase": digest_array(source_phase),
        "source_component": str(source_row["component"]),
        "source_target_id": str(source_row["target_id"]),
        "target_m": target.tolist(),
        "objective_method": asdict(objective.method),
        "positions": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "frequency_hz": float(ctx.config.frequency_hz),
        "maxiter": ITERATION_CAP,
        "gtol": float(ctx.config.gtol),
    }
    result, cache_status, _ = ctx.cache.get_or_compute(
        "alpha10_branch_medoid_continuation",
        payload,
        lambda: solve_corrected_gorkov_fe(
            objective,
            np.asarray(source_phase, dtype=float),
            maxiter=ITERATION_CAP,
            gtol=ctx.config.gtol,
        ),
        recompute=ctx.recompute,
    )
    row = source_row.to_dict()
    row.update(
        {
            "method": "Force-Equilibrium",
            "method_short": "FE",
            "alpha_per_m": FIXED_ALPHA_PER_M,
            "iteration_cap": ITERATION_CAP,
            "iterations": int(result.iterations),
            "evaluations": int(result.evaluations),
            "wall_time_s": float(result.history_wall_s[-1]),
            "objective": float(result.objective),
            "gradient_norm": float(result.gradient_norm),
            "phase_fingerprint": digest_array(result.phase_rad),
            "source_alpha_per_m": 9.0,
            "source_role": "branch-identified initialization only",
            "cache_status": str(cache_status),
            "correct_sign_gorkov": True,
        }
    )
    return row, result


def _medoid_index(states: np.ndarray) -> int:
    distance = pairwise_projective_distance(states)
    return int(np.argmin(distance.sum(axis=1)))


def fixed_single_branch_pack(ctx: p.PipelineContext) -> dict[str, Any]:
    """Return alpha=10 FE anchors, projected C webs, and root correlations."""

    memo_key = "alpha10_branch_pack"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]

    old_medoids = ctx.branch_bank.medoid_table.copy().reset_index(drop=True)
    fe_selector = ctx.artifact.anchor_table["method"].eq("Force-Equilibrium").to_numpy()
    old_fe_phases = np.asarray(ctx.artifact.anchor_phases[fe_selector], dtype=float)
    source_phases = np.stack(
        [old_fe_phases[int(row.source_endpoint_position)] for row in old_medoids.itertuples(index=False)]
    )
    jobs = [
        (index, old_medoids.iloc[index].copy(), source_phases[index].copy())
        for index in range(len(old_medoids))
    ]
    workers = max(
        1,
        min(len(jobs), int(os.environ.get("HAT_BRANCH_ALPHA1000_WORKERS", "4"))),
    )
    ordered: list[tuple[dict[str, Any], Any] | None] = [None] * len(jobs)
    if workers == 1:
        for index, row, phase in jobs:
            ordered[index] = _continue_one(ctx, row, phase)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(_continue_one, ctx, row, phase): index
                for index, row, phase in jobs
            }
            for future in as_completed(future_to_index):
                ordered[future_to_index[future]] = future.result()
    completed = [item for item in ordered if item is not None]
    anchor_table = pd.DataFrame([item[0] for item in completed]).reset_index(drop=True)
    anchor_results = tuple(item[1] for item in completed)
    anchor_phases = np.stack([np.asarray(item[1].phase_rad, dtype=float) for item in completed])
    anchor_states = target_demodulated_states(
        anchor_phases,
        anchor_table,
        positions_m=ctx.positions_m,
        frequency_hz=ctx.config.frequency_hz,
    )
    bank = ComponentBank(
        endpoint_table=anchor_table.copy(),
        endpoint_states=anchor_states.copy(),
        medoid_table=anchor_table.copy(),
        medoid_states=anchor_states.copy(),
        edge_table=ctx.branch_bank.edge_table.copy(),
        root_target=ctx.branch_bank.root_target,
    )
    frame = fit_fixed_3d_frame(bank)
    anchors = anchor_table.copy()
    anchors[["mds1", "mds2", "mds3"]] = frame.coordinates

    trajectory_table = ctx.artifact.trajectory_table.reset_index(drop=True)
    trajectory_states = ctx.artifact.trajectory_states()
    terminal_states = trajectory_states[:, -1, :]
    selected: list[tuple[int, str]] = []
    assigned = np.empty(len(trajectory_table), dtype=object)
    for (ix, iy), group in trajectory_table.groupby(["ix", "iy"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        target_anchors = np.stack(
            [bank.target_medoid(int(ix), int(iy), component) for component in COMPONENTS]
        )
        similarity = projective_similarity(terminal_states[positions], target_anchors)
        assigned[positions] = np.asarray(COMPONENTS, dtype=object)[np.argmax(similarity, axis=1)]
        for component in COMPONENTS:
            members = positions[assigned[positions] == component]
            if not len(members):
                raise RuntimeError(
                    f"No Conventional terminal trajectory maps to {component} at {(int(ix), int(iy))}"
                )
            selected.append((int(members[_medoid_index(terminal_states[members])]), component))

    evolution_rows: list[dict[str, Any]] = []
    for trajectory_index, component in selected:
        source = trajectory_table.iloc[trajectory_index]
        ix, iy = int(source["ix"]), int(source["iy"])
        matched = bank.target_medoid(ix, iy, component)
        states = trajectory_states[trajectory_index]
        coordinates = frame.project(states)
        similarity = projective_similarity(states, matched[None, :]).reshape(-1)
        for checkpoint_index, iteration in enumerate(ctx.artifact.checkpoints):
            evolution_rows.append(
                source.to_dict()
                | {
                    "component": component,
                    "iteration": int(iteration),
                    "selected_trajectory_index": int(trajectory_index),
                    "mds1": float(coordinates[checkpoint_index, 0]),
                    "mds2": float(coordinates[checkpoint_index, 1]),
                    "mds3": float(coordinates[checkpoint_index, 2]),
                    "similarity_to_matched_fe": float(similarity[checkpoint_index]),
                    "distance_to_matched_fe": float(math.sqrt(max(0.0, 1.0 - similarity[checkpoint_index] ** 2))),
                }
            )
    evolution = pd.DataFrame(evolution_rows)

    root_ix, root_iy = bank.root_target
    root_positions = trajectory_table.index[
        trajectory_table["ix"].eq(root_ix) & trajectory_table["iy"].eq(root_iy)
    ].to_numpy(dtype=int)
    root_anchors = np.stack(
        [bank.target_medoid(root_ix, root_iy, component) for component in COMPONENTS]
    )
    root_similarity = projective_similarity(terminal_states[root_positions], root_anchors)
    root_component_index = np.argmax(root_similarity, axis=1)
    correlation_rows: list[dict[str, Any]] = []
    for local_index, trajectory_index in enumerate(root_positions):
        component_index = int(root_component_index[local_index])
        source = trajectory_table.iloc[int(trajectory_index)]
        correlation_rows.append(
            {
                "trajectory_index": int(trajectory_index),
                "target_id": str(source["target_id"]),
                "component": COMPONENTS[component_index],
                "objective": float(source["objective"]),
                "similarity_to_matched_fe": float(root_similarity[local_index, component_index]),
                "distance_to_matched_fe": float(
                    math.sqrt(max(0.0, 1.0 - root_similarity[local_index, component_index] ** 2))
                ),
            }
        )
    correlation = pd.DataFrame(correlation_rows)
    protocol = pd.DataFrame(
        [
            {
                "source_archive_alpha_per_m": 9.0,
                "source_archive_role": "branch-identified initialization only",
                "production_alpha_per_m": FIXED_ALPHA_PER_M,
                "production_iteration_cap": ITERATION_CAP,
                "continued_anchor_count": len(anchor_table),
                "component_count": len(COMPONENTS),
                "conventional_in_frame_fit": False,
                "rh_in_frame_fit": False,
                "frame": "FE-only projective-distance MDS, fixed before Conventional projection",
                "worker_count": workers,
            }
        ]
    )
    ctx.tables["fig2_fixed_single_fe_branch_anchors"] = anchor_table.copy()
    ctx.tables["fig2_fixed_single_conventional_evolution"] = evolution.copy()
    ctx.tables["fig2_fixed_single_continuation_protocol"] = protocol
    value = {
        "bank": bank,
        "frame": frame,
        "anchors": anchors,
        "anchor_phases": anchor_phases,
        "anchor_results": anchor_results,
        "evolution": evolution,
        "correlation": correlation,
        "protocol": protocol,
    }
    ctx.memo[memo_key] = value
    return value


__all__ = ["FIXED_ALPHA_PER_M", "ITERATION_CAP", "fixed_single_branch_pack"]
