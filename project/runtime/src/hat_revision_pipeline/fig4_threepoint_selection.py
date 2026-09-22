"""Bounded, predeclared selection audit for the Figure 4 Triple chart.

This module changes only the three prescribed target coordinates used by the
connected Triple demonstration.  The array, pressure model, correct-sign
Gor'kov implementation, Conventional and FE objectives, optimizer, initial
phase command, connected-domain deformation, and fixed-frame 3-D display
metric remain unchanged.

The candidate set and selection rule are constants so that the reported
layout cannot be selected by inspecting the eventual 10,000-iteration web.
Three centroid-preserving contractions of the already accepted scalene
triangle are screened on the same five-node connected cross at 1,000 accepted
iterations.  The candidate with the smallest median paired projective
distance to its matched FE command is selected; the 75th percentile is the
predeclared tie-breaker.  Only the selected candidate is then evaluated on the
3 x 3 connected chart at 10,000 Conventional iterations.

The projective distance is the same existing full-dimensional quantity used
by Figure 4,

    d_P(phi_C, phi_FE) = sqrt(1 - |u_C^H u_FE|^2),

where u = exp(i phi) / sqrt(N).  It is an audit/selection quantity, not a new
scientific metric and not an embedding-space distance.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import pipeline as p
from .cache import digest_array
from .gorkov_core import formula_self_test
from .multitrap import (
    MultiTrapProblem,
    MultitrapObjectiveConfig,
    SPUComponents,
    SphereInFluid,
    TargetStencil,
    evaluate_multitrap_objective,
    run_multitrap,
)


ACCEPTED_BASE_TARGETS_M = np.asarray(
    ((0.0, 0.013, 0.030), (-0.011, -0.009, 0.030), (0.011, -0.009, 0.030)),
    dtype=float,
)

# Frozen before execution.  Only one geometric factor is changed.
CANDIDATE_CONTRACTIONS: tuple[tuple[str, float], ...] = (
    ("accepted_1p00", 1.00),
    ("contracted_0p80", 0.80),
    ("contracted_0p65", 0.65),
)
SCREEN_NODES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
)
FINAL_VALUES = np.asarray((-1.0, 0.0, 1.0), dtype=float)
DOMAIN_DELTA_M = 0.001
SCREEN_CAP = 1_000
FINAL_COMMON_CAP = 1_000
FINAL_EXTENDED_CAP = 10_000
SEED_OFFSET = 41
SELECTION_RULE = (
    "minimum median paired full-dimensional projector distance over the frozen "
    "five-node cross at 1,000 iterations; q75 is the tie-breaker"
)


def contracted_targets(scale: float) -> np.ndarray:
    """Contract the accepted triangle about its unchanged centroid."""

    centroid = ACCEPTED_BASE_TARGETS_M.mean(axis=0, keepdims=True)
    return centroid + float(scale) * (ACCEPTED_BASE_TARGETS_M - centroid)


def domain_targets(base_targets_m: np.ndarray, u: float, v: float) -> np.ndarray:
    """Apply the unchanged centroid-preserving connected-chart deformation."""

    du = np.asarray(((1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)))
    dv = np.asarray(((0.0, -0.5, 0.0), (0.0, 1.0, 0.0), (0.0, -0.5, 0.0)))
    return np.asarray(base_targets_m, dtype=float) + DOMAIN_DELTA_M * (
        float(u) * du + float(v) * dv
    )


def make_problem(
    ctx: p.PipelineContext,
    base_targets_m: np.ndarray,
    u: float,
    v: float,
    *,
    case_prefix: str,
) -> MultiTrapProblem:
    targets = domain_targets(base_targets_m, u, v)
    stencils = tuple(
        TargetStencil(
            target_id=f"T{index + 1}",
            target_m=tuple(target),
            transfer=p._stencil_transfer(
                ctx.positions_m, target, ctx.config.frequency_hz
            ),
            spacing_m=ctx.config.stencil_m,
            curvature_weight=np.eye(3),
        )
        for index, target in enumerate(targets)
    )
    return MultiTrapProblem(
        case_id=f"{case_prefix}-u{u:+.3f}-v{v:+.3f}",
        stencils=stencils,
        material=SphereInFluid(frequency_hz=ctx.config.frequency_hz),
        geometry_id="square-16x16",
    )


def configs(
    ctx: p.PipelineContext,
    conventional_cap: int,
    fe_cap: int,
) -> tuple[MultitrapObjectiveConfig, MultitrapObjectiveConfig]:
    """Return the unchanged Figure 4 Conventional and pure-FE configs."""

    conventional = MultitrapObjectiveConfig(
        method="Conventional",
        components=SPUComponents(False, True, True),
        alpha_force=0.0,
        beta_pressure=1.0,
        gamma_uniformity=1.0,
        curvature_weight_override=(
            (1000.0, 0.0, 0.0),
            (0.0, 1000.0, 0.0),
            (0.0, 0.0, 10.0),
        ),
        pressure_epsilon_rel=0.0,
        smooth_stage_factors=(0.01,),
        smooth_stage_maxiters=(int(conventional_cap),),
        gtol=ctx.config.gtol,
        report_gradient_tol=1.0e-3,
    )
    fe = MultitrapObjectiveConfig(
        method="Regularized-FE",  # implementation enum; displayed as FE
        components=SPUComponents(False, False, True),
        alpha_force=1000.0,
        beta_pressure=0.0,
        gamma_uniformity=1.0,
        curvature_weight_override=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        pressure_epsilon_rel=0.0,
        smooth_stage_factors=(0.01,),
        smooth_stage_maxiters=(int(fe_cap),),
        gtol=ctx.config.gtol,
        report_gradient_tol=1.0e-3,
    )
    return conventional, fe


def projector_distance(left_phase: np.ndarray, right_phase: np.ndarray) -> float:
    left = np.exp(1j * np.asarray(left_phase, dtype=float))
    right = np.exp(1j * np.asarray(right_phase, dtype=float))
    similarity = float(
        np.clip(abs(np.vdot(left, right)) / math.sqrt(left.size * right.size), 0.0, 1.0)
    )
    return math.sqrt(max(0.0, 1.0 - similarity**2))


def _phase_at(result: Any, accepted_iteration: int) -> np.ndarray:
    history = np.asarray(result.history_phase_rad, dtype=float)
    return history[min(int(accepted_iteration), len(history) - 1)]


def _native_directions(
    problem: MultiTrapProblem,
    phase: np.ndarray,
    history: np.ndarray,
    config: MultitrapObjectiveConfig,
    *,
    force_epsilon: float,
    uniformity_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the existing terminal-gradient/history direction definition."""

    evaluation = evaluate_multitrap_objective(
        problem,
        phase,
        config,
        force_epsilon=float(force_epsilon),
        uniformity_epsilon=float(uniformity_epsilon),
    )
    gradient = np.asarray(evaluation.gradient_full, dtype=float)
    u = -gradient / max(float(np.linalg.norm(gradient)), 1.0e-30)
    current = np.asarray(phase, dtype=float)
    v_direction: np.ndarray | None = None
    for previous in np.asarray(history, dtype=float)[::-1]:
        gauge = np.angle(np.vdot(np.exp(1j * previous), np.exp(1j * current)))
        step = np.angle(
            np.exp(1j * current) * np.exp(-1j * gauge) * np.exp(-1j * previous)
        )
        step -= u * float(np.dot(step, u))
        norm = float(np.linalg.norm(step))
        if norm > 1.0e-14:
            v_direction = step / norm
            break
    if v_direction is None:
        raise RuntimeError("Selected Triple history has no transverse accepted step")
    return u, v_direction


def _cached_run(
    ctx: p.PipelineContext,
    problem: MultiTrapProblem,
    config: MultitrapObjectiveConfig,
    *,
    seed: int,
    initial: np.ndarray,
    display: str,
    candidate_id: str,
    u: float,
    v: float,
) -> Any:
    payload = {
        "contract": "fig4-frozen-threepoint-selection-v1",
        "problem": problem.fingerprint,
        "config": config.to_payload(),
        "seed": int(seed),
        "initial": digest_array(initial),
        "display": display,
        "candidate_id": candidate_id,
        "u": float(u),
        "v": float(v),
    }
    result, _, _ = ctx.cache.get_or_compute(
        "fig4_threepoint_selection_run",
        payload,
        lambda: run_multitrap(
            problem,
            config,
            seed=seed,
            initial_phases=initial,
            record_history=True,
            metadata={
                "display_method": display,
                "candidate_id": candidate_id,
                "u": float(u),
                "v": float(v),
            },
        ),
        recompute=ctx.recompute,
    )
    return result


def run_candidate_screen(
    ctx: p.PipelineContext,
    *,
    candidate_subset: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run the frozen candidate screen and return raw and summary tables.

    ``candidate_subset`` only shards the predeclared candidate list across
    independent processes; it cannot introduce another candidate or change
    the five frozen screen nodes.  The unsharded call remains the authoritative
    selector after every shard has populated the content-addressed cache.
    """

    guard = formula_self_test()
    if guard.get("passed") is not True:
        raise RuntimeError(guard)
    seed = int(ctx.config.random_seed + SEED_OFFSET)
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ctx.positions_m))
    rows: list[dict[str, Any]] = []
    c_config, fe_config = configs(ctx, SCREEN_CAP, SCREEN_CAP)
    candidate_ids = {name for name, _ in CANDIDATE_CONTRACTIONS}
    if candidate_subset is None:
        selected_candidates = list(CANDIDATE_CONTRACTIONS)
    else:
        requested = set(candidate_subset)
        invalid = sorted(requested.difference(candidate_ids))
        if invalid:
            raise ValueError(f"Unknown screen candidates: {invalid}")
        selected_candidates = [
            pair for pair in CANDIDATE_CONTRACTIONS if pair[0] in requested
        ]
    for candidate_id, scale in selected_candidates:
        base = contracted_targets(scale)
        pairwise = np.linalg.norm(base[:, None, :] - base[None, :, :], axis=2)
        nonzero = pairwise[pairwise > 0.0]
        for u, v in SCREEN_NODES:
            problem = make_problem(
                ctx, base, u, v, case_prefix=f"fig4-screen-{candidate_id}"
            )
            conventional = _cached_run(
                ctx, problem, c_config, seed=seed, initial=initial,
                display="Conventional", candidate_id=candidate_id, u=u, v=v,
            )
            fe = _cached_run(
                ctx, problem, fe_config, seed=seed, initial=initial,
                display="FE", candidate_id=candidate_id, u=u, v=v,
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "contraction": float(scale),
                    "u": float(u),
                    "v": float(v),
                    "screen_cap": SCREEN_CAP,
                    "paired_projector_distance": projector_distance(
                        conventional.phases, fe.phases
                    ),
                    "conventional_iterations": int(conventional.iterations),
                    "fe_iterations": int(fe.iterations),
                    "conventional_gradient_norm": float(
                        conventional.terminal_gradient_norm
                    ),
                    "fe_gradient_norm": float(fe.terminal_gradient_norm),
                    "minimum_pair_separation_m_at_center": float(nonzero.min()),
                    "base_targets_json": json.dumps(base.tolist()),
                    "seed": seed,
                    "initial_phase_content_id": digest_array(initial),
                }
            )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["candidate_id", "contraction"], as_index=False)
        .agg(
            node_count=("paired_projector_distance", "size"),
            median_paired_projector_distance=("paired_projector_distance", "median"),
            q75_paired_projector_distance=(
                "paired_projector_distance", lambda x: float(np.quantile(x, 0.75))
            ),
            minimum_pair_separation_m=("minimum_pair_separation_m_at_center", "first"),
        )
        .sort_values(
            ["median_paired_projector_distance", "q75_paired_projector_distance", "candidate_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    summary["selection_rank"] = np.arange(1, len(summary) + 1)
    summary["selected"] = summary["selection_rank"].eq(1)
    summary["selection_rule"] = SELECTION_RULE
    selected = str(summary.iloc[0]["candidate_id"])
    return raw, summary, selected


def run_selected_domain(
    ctx: p.PipelineContext,
    selected_candidate_id: str,
    *,
    node_subset: tuple[tuple[int, int], ...] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the selected 3 x 3 chart at 1,000/10,000 iterations.

    ``node_subset`` exists only to permit independent process shards.  Each
    shard writes the same content-addressed run cache; a final unsharded call
    assembles the complete ordered bank without rerunning completed nodes.
    """

    candidate_map = dict(CANDIDATE_CONTRACTIONS)
    if selected_candidate_id not in candidate_map:
        raise KeyError(selected_candidate_id)
    scale = float(candidate_map[selected_candidate_id])
    base = contracted_targets(scale)
    seed = int(ctx.config.random_seed + SEED_OFFSET)
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ctx.positions_m))
    c_config, fe_config = configs(ctx, FINAL_EXTENDED_CAP, FINAL_COMMON_CAP)
    all_nodes = [
        (ix, iy)
        for iy in range(len(FINAL_VALUES))
        for ix in range(len(FINAL_VALUES))
    ]
    if node_subset is None:
        nodes = all_nodes
    else:
        requested = [tuple(map(int, node)) for node in node_subset]
        invalid = sorted(set(requested).difference(all_nodes))
        if invalid:
            raise ValueError(f"Unknown selected-domain nodes: {invalid}")
        nodes = [node for node in all_nodes if node in set(requested)]
    objects: dict[tuple[int, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for ix, iy in nodes:
        u, v = float(FINAL_VALUES[ix]), float(FINAL_VALUES[iy])
        problem = make_problem(
            ctx, base, u, v,
            case_prefix=f"fig4-selected-{selected_candidate_id}",
        )
        conventional = _cached_run(
            ctx, problem, c_config, seed=seed, initial=initial,
            display="Conventional", candidate_id=selected_candidate_id, u=u, v=v,
        )
        fe = _cached_run(
            ctx, problem, fe_config, seed=seed, initial=initial,
            display="FE", candidate_id=selected_candidate_id, u=u, v=v,
        )
        c_common = _phase_at(conventional, FINAL_COMMON_CAP)
        c_extended = np.asarray(conventional.phases, dtype=float)
        fe_phase = np.asarray(fe.phases, dtype=float)
        distance_common = projector_distance(c_common, fe_phase)
        distance_extended = projector_distance(c_extended, fe_phase)
        objects[(ix, iy)] = {
            "problem": problem,
            "c_config": c_config,
            "fe_config": fe_config,
            "Conventional": conventional,
            "FE": fe,
            "Conventional_common_phase": c_common,
            "Conventional_extended_phase": c_extended,
            "FE_common_phase": fe_phase,
            "FE_extended_phase": fe_phase,
        }
        rows.append(
            {
                "candidate_id": selected_candidate_id,
                "contraction": scale,
                "ix": ix,
                "iy": iy,
                "u": u,
                "v": v,
                "common_cap": FINAL_COMMON_CAP,
                "extended_cap": FINAL_EXTENDED_CAP,
                "conventional_iterations": int(conventional.iterations),
                "fe_iterations": int(fe.iterations),
                "distance_to_FE_common": distance_common,
                "distance_to_FE_extended": distance_extended,
                "distance_change": distance_extended - distance_common,
                "conventional_gradient_norm": float(
                    conventional.terminal_gradient_norm
                ),
                "fe_gradient_norm": float(fe.terminal_gradient_norm),
                "targets_json": json.dumps(
                    domain_targets(base, u, v).tolist()
                ),
                "seed": seed,
                "initial_phase_content_id": digest_array(initial),
            }
        )
    table = pd.DataFrame(rows)
    bank = {
        "values": FINAL_VALUES.copy(),
        "nodes": nodes,
        "objects": objects,
        "table": table,
        "common_cap": FINAL_COMMON_CAP,
        "extended_cap": FINAL_EXTENDED_CAP,
        "seed": seed,
        "base_targets_m": base,
        "selected_candidate_id": selected_candidate_id,
    }
    return bank, table


def export_selected_artifact(
    destination: str | Path,
    *,
    raw_screen: pd.DataFrame,
    screen_summary: pd.DataFrame,
    selected_bank: Mapping[str, Any],
    selected_table: pd.DataFrame,
    development_destination: str | Path | None = None,
) -> Path:
    """Write the selected runtime artifact and keep screening data separate."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    if development_destination is not None:
        development = Path(development_destination)
        development.mkdir(parents=True, exist_ok=True)
        raw_screen.to_csv(development / "candidate_screen_raw.csv", index=False)
        screen_summary.to_csv(development / "candidate_screen_summary.csv", index=False)
    selected_table.to_csv(root / "selected_domain_runs.csv", index=False)
    nodes = list(selected_bank["nodes"])
    objects = selected_bank["objects"]
    np.savez_compressed(
        root / "selected_phase_bank.npz",
        values=np.asarray(selected_bank["values"], dtype=float),
        nodes=np.asarray(nodes, dtype=int),
        fe=np.stack([objects[node]["FE_common_phase"] for node in nodes]),
        conventional_common=np.stack(
            [objects[node]["Conventional_common_phase"] for node in nodes]
        ),
        conventional_extended=np.stack(
            [objects[node]["Conventional_extended_phase"] for node in nodes]
        ),
    )

    values = np.asarray(selected_bank["values"], dtype=float)
    center_candidates = np.flatnonzero(np.isclose(values, 0.0, atol=1.0e-15, rtol=0.0))
    if len(center_candidates) != 1:
        raise RuntimeError("Selected Triple chart must contain one zero-valued center")
    center = int(center_candidates[0])
    central = objects[(center, center)]
    central_directions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    central_parameters: dict[str, dict[str, Any]] = {}
    for display, result_key, config_key, phase_key in (
        ("Conventional", "Conventional", "c_config", "Conventional_common_phase"),
        ("FE", "FE", "fe_config", "FE_common_phase"),
    ):
        result = central[result_key]
        config = central[config_key]
        phase = np.asarray(central[phase_key], dtype=float)
        history = np.asarray(result.history_phase_rad, dtype=float)
        if display == "Conventional":
            history = history[: min(len(history), FINAL_COMMON_CAP + 1)]
        force_epsilon = float(result.stages[-1].force_epsilon)
        uniformity_epsilon = float(result.stages[-1].uniformity_epsilon)
        directions = _native_directions(
            central["problem"],
            phase,
            history,
            config,
            force_epsilon=force_epsilon,
            uniformity_epsilon=uniformity_epsilon,
        )
        central_directions[display] = directions
        central_parameters[display] = {
            "phase_content_id": digest_array(phase),
            "direction_u_content_id": digest_array(directions[0]),
            "direction_v_content_id": digest_array(directions[1]),
            "force_epsilon": force_epsilon,
            "uniformity_epsilon": uniformity_epsilon,
            "direction_u": "terminal negative gradient",
            "direction_v": "orthogonalized last nonzero accepted step",
        }
    np.savez_compressed(
        root / "central_native_directions.npz",
        conventional_u=central_directions["Conventional"][0],
        conventional_v=central_directions["Conventional"][1],
        fe_u=central_directions["FE"][0],
        fe_v=central_directions["FE"][1],
    )
    metadata = {
        "contract": "fig4-frozen-threepoint-selection-v1",
        "selected_candidate_id": selected_bank["selected_candidate_id"],
        "selected_geometry": "0.80 x accepted triangle about fixed centroid",
        "selected_base_targets_m": np.asarray(
            selected_bank["base_targets_m"], dtype=float
        ).tolist(),
        "domain_delta_m": DOMAIN_DELTA_M,
        "domain_values": np.asarray(selected_bank["values"], dtype=float).tolist(),
        "common_cap": FINAL_COMMON_CAP,
        "extended_cap": FINAL_EXTENDED_CAP,
        "seed": int(selected_bank["seed"]),
        "array_and_objective_changes": "none",
        "correct_sign_gorkov": True,
        "central_problem_fingerprint": central["problem"].fingerprint,
        "central_decision_definition": central_parameters,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def load_selected_display_bank(source: str | Path) -> dict[str, Any]:
    """Load the compact artifact in the shape expected by the 3-D frame helper."""

    root = Path(source)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "selected_phase_bank.npz") as archive:
        values = np.asarray(archive["values"], dtype=float)
        nodes = [tuple(map(int, row)) for row in np.asarray(archive["nodes"], dtype=int)]
        fe = np.asarray(archive["fe"], dtype=float)
        common = np.asarray(archive["conventional_common"], dtype=float)
        extended = np.asarray(archive["conventional_extended"], dtype=float)
    objects = {
        node: {
            "FE_common_phase": fe[index],
            "Conventional_common_phase": common[index],
            "Conventional_extended_phase": extended[index],
        }
        for index, node in enumerate(nodes)
    }
    return {
        "values": values,
        "nodes": nodes,
        "objects": objects,
        "table": pd.read_csv(root / "selected_domain_runs.csv"),
        "common_cap": int(metadata["common_cap"]),
        "extended_cap": int(metadata["extended_cap"]),
        "seed": int(metadata["seed"]),
        "base_targets_m": np.asarray(metadata["selected_base_targets_m"], dtype=float),
        "selected_candidate_id": metadata["selected_candidate_id"],
    }


def selected_figure4_bank(
    ctx: p.PipelineContext,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Load the frozen selected bank and attach only main-text evidence tables.

    Candidate-screen tables intentionally remain development artifacts on
    disk.  They are not attached to ``ctx.tables`` and therefore cannot leak
    into the notebook body as an additional parameter study.
    """

    root = Path(source) if source is not None else (
        Path(ctx.data_root) / "fig4_selected_triple_10k"
    )
    bank = load_selected_display_bank(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    table = pd.read_csv(root / "selected_domain_runs.csv")
    ctx.tables["fig4_selected_triple_domain_runs"] = table
    ctx.tables["fig4_selected_triple_domain_definition"] = pd.DataFrame(
        [
            {
                "base_targets_m": json.dumps(metadata["selected_base_targets_m"]),
                "delta_m": float(metadata["domain_delta_m"]),
                "u_values": json.dumps(metadata["domain_values"]),
                "v_values": json.dumps(metadata["domain_values"]),
                "q1_rule": "(+1,-1/2,-1/2) x displacement",
                "q2_rule": "(-1/2,+1,-1/2) y displacement",
                "centroid_preserved": True,
                "common_cap": int(metadata["common_cap"]),
                "extended_conventional_cap": int(metadata["extended_cap"]),
                "selected_geometry": "0.80 x accepted triangle about fixed centroid",
                "minimum_center_pair_separation_m": float(
                    np.min(
                        np.linalg.norm(
                            bank["base_targets_m"][:, None, :]
                            - bank["base_targets_m"][None, :, :],
                            axis=2,
                        )
                        + np.eye(3) * 1.0e9
                    )
                ),
            }
        ]
    )
    return bank


def selected_domain_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return only the 1k/10k paired-distance evidence used in body prose."""

    required = {
        "common_cap", "extended_cap", "distance_to_FE_common",
        "distance_to_FE_extended", "distance_change",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Selected Figure 4 table is missing {missing}")
    improved = table["distance_to_FE_extended"] < table["distance_to_FE_common"]
    return pd.DataFrame(
        [
            {
                "nodes": int(len(table)),
                "common_cap": int(table["common_cap"].iloc[0]),
                "extended_cap": int(table["extended_cap"].iloc[0]),
                "median_full_projective_distance_to_FE_common": float(
                    table["distance_to_FE_common"].median()
                ),
                "median_full_projective_distance_to_FE_extended": float(
                    table["distance_to_FE_extended"].median()
                ),
                "nodes_with_lower_distance_after_extension": int(improved.sum()),
                "fraction_with_lower_distance_after_extension": float(improved.mean()),
                "median_paired_distance_change": float(
                    table["distance_change"].median()
                ),
            }
        ]
    )


def selected_figure5_node(
    ctx: p.PipelineContext,
    bank: Mapping[str, Any] | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Reconstruct the selected central node without rerunning optimization."""

    root = Path(source) if source is not None else (
        Path(ctx.data_root) / "fig4_selected_triple_10k"
    )
    selected_bank = dict(bank) if bank is not None else load_selected_display_bank(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    values = np.asarray(selected_bank["values"], dtype=float)
    zero = np.flatnonzero(np.isclose(values, 0.0, atol=1.0e-15, rtol=0.0))
    if len(zero) != 1:
        raise RuntimeError("Selected Triple chart must contain one center node")
    center = int(zero[0])
    phase_node = selected_bank["objects"][(center, center)]
    base = np.asarray(selected_bank["base_targets_m"], dtype=float)
    problem = make_problem(
        ctx,
        base,
        0.0,
        0.0,
        case_prefix=f"fig4-selected-{selected_bank['selected_candidate_id']}",
    )
    if problem.fingerprint != metadata["central_problem_fingerprint"]:
        raise RuntimeError("Selected central Triple problem fingerprint changed")
    c_config, fe_config = configs(ctx, FINAL_EXTENDED_CAP, FINAL_COMMON_CAP)
    with np.load(root / "central_native_directions.npz") as archive:
        directions = {
            "Conventional": (
                np.asarray(archive["conventional_u"], dtype=float),
                np.asarray(archive["conventional_v"], dtype=float),
            ),
            "FE": (
                np.asarray(archive["fe_u"], dtype=float),
                np.asarray(archive["fe_v"], dtype=float),
            ),
        }
    output: dict[str, Any] = {
        "problem": problem,
        "c_config": c_config,
        "fe_config": fe_config,
        "Conventional_common_phase": np.asarray(
            phase_node["Conventional_common_phase"], dtype=float
        ),
        "FE_common_phase": np.asarray(phase_node["FE_common_phase"], dtype=float),
        "directions": directions,
        "parameters": metadata["central_decision_definition"],
        "targets_m": base,
    }
    for display, phase_key in (
        ("Conventional", "Conventional_common_phase"),
        ("FE", "FE_common_phase"),
    ):
        parameters = output["parameters"][display]
        u, v = directions[display]
        if digest_array(output[phase_key]) != parameters["phase_content_id"]:
            raise RuntimeError(f"Selected {display} phase hash changed")
        if digest_array(u) != parameters["direction_u_content_id"]:
            raise RuntimeError(f"Selected {display} gradient direction hash changed")
        if digest_array(v) != parameters["direction_v_content_id"]:
            raise RuntimeError(f"Selected {display} transverse direction hash changed")
    return output


def selected_central_decision_plane(
    ctx: p.PipelineContext,
    node: Mapping[str, Any],
    display: str,
    *,
    points: int | None = None,
    half_range_rad: float = 7.50,
) -> dict[str, np.ndarray]:
    """Evaluate the unchanged native Figure 5 section on the selected node."""

    if display not in {"Conventional", "FE"}:
        raise KeyError(display)
    config = node["c_config"] if display == "Conventional" else node["fe_config"]
    phase_key = (
        "Conventional_common_phase" if display == "Conventional" else "FE_common_phase"
    )
    phase = np.asarray(node[phase_key], dtype=float)
    u, v = node["directions"][display]
    parameters = node["parameters"][display]
    force_epsilon = float(parameters["force_epsilon"])
    uniformity_epsilon = float(parameters["uniformity_epsilon"])
    count = int(points if points is not None else (101 if ctx.config.full else 61))
    if count < 3:
        raise ValueError("Decision-section points must be at least three")
    q = np.linspace(-float(half_range_rad), float(half_range_rad), count)
    payload = {
        "contract": "fig5-wide-native-selected-triple-section-v1",
        "problem": node["problem"].fingerprint,
        "config": config.to_payload(),
        "phase": digest_array(phase),
        "direction_u": digest_array(u),
        "direction_v": digest_array(v),
        "half_range_rad": float(half_range_rad),
        "points": count,
        "force_epsilon": force_epsilon,
        "uniformity_epsilon": uniformity_epsilon,
    }

    def compute() -> dict[str, np.ndarray]:
        plane = np.empty((count, count), dtype=float)
        for j, b in enumerate(q):
            for i, a in enumerate(q):
                plane[j, i] = evaluate_multitrap_objective(
                    node["problem"],
                    phase + a * u + b * v,
                    config,
                    force_epsilon=force_epsilon,
                    uniformity_epsilon=uniformity_epsilon,
                ).value
        return {"q": q, "plane": plane - np.nanmin(plane)}

    value, _, _ = ctx.cache.get_or_compute(
        "final_selected_tri3_wide_section",
        payload,
        compute,
        recompute=ctx.recompute,
    )
    return value


__all__ = [
    "CANDIDATE_CONTRACTIONS",
    "DOMAIN_DELTA_M",
    "FINAL_COMMON_CAP",
    "FINAL_EXTENDED_CAP",
    "SCREEN_CAP",
    "SCREEN_NODES",
    "SELECTION_RULE",
    "contracted_targets",
    "domain_targets",
    "export_selected_artifact",
    "load_selected_display_bank",
    "run_candidate_screen",
    "run_selected_domain",
    "selected_domain_summary",
    "selected_central_decision_plane",
    "selected_figure4_bank",
    "selected_figure5_node",
]
