"""Accepted center-node BFGS implementation and cache compatibility types.

The retired Triple long-branch grid and plotting experiment have been removed.
Only the fixed central endpoint is requested by main_triple_endpoint.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import pipeline as p
from .cache import digest_array, digest_payload
from .fig34_3d_patch import (
    FE_KEY,
    draw_triple_fixed_web_3d,
    fit_triple_fixed_frame_3d,
    triple_fixed_frame_summary,
    triple_method_legend,
)
from .fig4_threepoint_selection import (
    contracted_targets,
    make_problem,
    projector_distance,
)
from .fixed_fe_endpoints import TRIPLE_FE_ENDPOINT, fixed_triple_fe_config
from .figure_presentation import outside_grid_panel_labels
from .gorkov_core import formula_self_test
from .multitrap import (
    MultitrapObjectiveConfig,
    MultiTrapProblem,
    SPUComponents,
    estimate_smooth_scales,
    evaluate_multitrap_objective,
    run_multitrap,
)


SCHEMA_VERSION = 2
OBSERVATION_STEM = "Observation_Triple_3x3_long_branch"
TABLE_PREFIX = "triple_long_iteration"
DATA_STEM = "triple_long_iteration_phase_bank"

DOMAIN_VALUES = np.asarray((-1.0, 0.0, 1.0), dtype=float)
FIXED_BASE_TARGETS_M = contracted_targets(0.80)
DOMAIN_DELTA_M = 0.001
SEED_OFFSET = 41
FE_ALPHA = float(TRIPLE_FE_ENDPOINT.alpha)
MAIN_TRIPLE_ITERATION_CAP = 30_000
FE_CAP = MAIN_TRIPLE_ITERATION_CAP
FE_CURVATURE_WX = float(TRIPLE_FE_ENDPOINT.curvature_weight_override[0][0])
FE_CURVATURE_WY = float(TRIPLE_FE_ENDPOINT.curvature_weight_override[1][1])
FE_CURVATURE_WZ = float(TRIPLE_FE_ENDPOINT.curvature_weight_override[2][2])
SMOKE_CONVENTIONAL_CAP = MAIN_TRIPLE_ITERATION_CAP
FULL_CONVENTIONAL_CAP = MAIN_TRIPLE_ITERATION_CAP
CHECKPOINTS = (10_000, 20_000, 30_000)
DEFAULT_WORKERS = 3
RESTART_POLICY = (
    "restart-BFGS-from-identical-terminal-phase-until-cumulative-cap-v1"
)


def _wrap_phase(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _gauge_reduce(full_phase: np.ndarray) -> np.ndarray:
    phase = _wrap_phase(np.asarray(full_phase, dtype=float))
    phase = _wrap_phase(phase - phase[0])
    return phase[1:]


def _gauge_expand(reduced_phase: np.ndarray) -> np.ndarray:
    reduced = np.asarray(reduced_phase, dtype=float)
    return _wrap_phase(np.concatenate((np.zeros(1, dtype=float), reduced)))


def _ordered_nodes() -> tuple[tuple[int, int], ...]:
    """Row-major chart order used by computation, tables and plotting."""

    return tuple(
        (ix, iy)
        for iy in range(len(DOMAIN_VALUES))
        for ix in range(len(DOMAIN_VALUES))
    )


def _worker_count() -> int:
    raw = os.environ.get("HAT_LONG_TRIPLE_WORKERS", str(DEFAULT_WORKERS))
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError("HAT_LONG_TRIPLE_WORKERS must be an integer") from error
    if workers < 1:
        raise ValueError("HAT_LONG_TRIPLE_WORKERS must be positive")
    return min(workers, len(_ordered_nodes()))


def _caps(ctx: p.PipelineContext) -> tuple[int, tuple[int, ...]]:
    cap = FULL_CONVENTIONAL_CAP if ctx.config.full else SMOKE_CONVENTIONAL_CAP
    checkpoints = tuple(value for value in CHECKPOINTS if value <= cap)
    if checkpoints[-1] != cap:
        raise RuntimeError("The cumulative cap must be one of the frozen checkpoints")
    return cap, checkpoints


def _configs(
    ctx: p.PipelineContext,
    conventional_cap: int,
) -> tuple[MultitrapObjectiveConfig, MultitrapObjectiveConfig]:
    """Construct the fixed Conventional and frozen Triple FE objectives."""

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
    fixed_fe = fixed_triple_fe_config(gtol=ctx.config.gtol)
    # The original constructor remains frozen for Appendix C reference tests.
    fixed_fe = replace(fixed_fe, smooth_stage_maxiters=(250, FE_CAP - 250))
    return conventional, fixed_fe


@dataclass(frozen=True)
class CumulativeBFGSResult:
    """Compact result retaining only the predeclared accepted checkpoints."""

    problem_fingerprint: str
    config_fingerprint: str
    seed: int
    initial_phase_hash: str
    terminal_phase_rad: np.ndarray
    checkpoint_phases_rad: Mapping[int, np.ndarray]
    checkpoint_records: tuple[Mapping[str, Any], ...]
    segment_records: tuple[Mapping[str, Any], ...]
    accepted_iterations: int
    nfev: int
    njev: int
    terminal_objective: float
    terminal_gradient_norm: float
    force_epsilon: float
    uniformity_epsilon: float
    solve_sec: float
    termination_reason: str
    restart_policy: str = RESTART_POLICY

    @property
    def segment_count(self) -> int:
        return len(self.segment_records)

    @property
    def restart_count(self) -> int:
        return max(0, self.segment_count - 1)


def _run_cumulative_conventional(
    problem: MultiTrapProblem,
    config: MultitrapObjectiveConfig,
    *,
    seed: int,
    initial_phases: np.ndarray,
    cumulative_cap: int,
    checkpoints: Sequence[int],
    record_stalled_terminal: bool = False,
) -> CumulativeBFGSResult:
    """Run fixed-objective BFGS segments until the accepted-iteration cap.

    A segment termination below the cap is treated as a numerical BFGS stall,
    not as permission to change the phase or objective.  The next invocation
    receives ``result.x`` byte-for-byte as its start and consequently resets
    only the optimizer's inverse-Hessian state.  A zero-accepted-step restart
    is an explicit failure because progressing would require an unauthorized
    phase perturbation or algorithm change.
    """

    if config.method != "Conventional":
        raise ValueError("The cumulative restart runner accepts Conventional only")
    requested = tuple(sorted({int(value) for value in checkpoints}))
    if not requested or requested[-1] != int(cumulative_cap):
        raise ValueError("checkpoints must include the cumulative cap")
    if requested[0] <= 0 or any(value > cumulative_cap for value in requested):
        raise ValueError("checkpoints must lie in (0, cumulative_cap]")

    initial_full = _gauge_expand(_gauge_reduce(initial_phases))
    scales = estimate_smooth_scales(problem, initial_full, config)
    stage_factor = float(config.smooth_stage_factors[0])
    force_epsilon = (
        stage_factor * scales.force_scale
        if config.components.force_smoothing
        else 0.0
    )
    uniformity_epsilon = (
        stage_factor * scales.loss_std_scale
        if config.components.uniformity
        else 0.0
    )

    x_current = _gauge_reduce(initial_full)
    accepted = 0
    total_nfev = 0
    total_njev = 0
    checkpoint_phases: dict[int, np.ndarray] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    solve_start = time.perf_counter()
    stalled_terminal = False

    def evaluate_reduced(x: np.ndarray) -> Any:
        return evaluate_multitrap_objective(
            problem,
            _gauge_expand(x),
            config,
            force_epsilon=force_epsilon,
            uniformity_epsilon=uniformity_epsilon,
        )

    while accepted < cumulative_cap:
        segment_index = len(segment_rows)
        remaining = int(cumulative_cap - accepted)
        segment_start_x = np.asarray(x_current, dtype=float).copy()
        segment_start_phase = _gauge_expand(segment_start_x)
        evaluation_cache: dict[str, Any] = {}

        def cached_evaluate(x: np.ndarray) -> Any:
            key = np.asarray(x, dtype=float).tobytes()
            if evaluation_cache.get("key") != key:
                evaluation_cache["key"] = key
                evaluation_cache["value"] = evaluate_reduced(x)
            return evaluation_cache["value"]

        segment_accepted = 0

        def callback(x: np.ndarray) -> None:
            nonlocal accepted, segment_accepted
            segment_accepted += 1
            accepted += 1
            if accepted in requested:
                phase = _gauge_expand(np.asarray(x, dtype=float))
                evaluation = cached_evaluate(x)
                checkpoint_phases[accepted] = phase.copy()
                checkpoint_rows.append(
                    {
                        "accepted_iteration": int(accepted),
                        "segment_index": int(segment_index),
                        "segment_iteration": int(segment_accepted),
                        "objective": float(evaluation.value),
                        "gradient_norm": float(
                            np.linalg.norm(evaluation.gradient_reduced)
                        ),
                        "elapsed_sec": float(time.perf_counter() - solve_start),
                        "phase_content_id": digest_array(phase),
                    }
                )

        segment_start = time.perf_counter()
        result = minimize(
            lambda x: cached_evaluate(x).value,
            segment_start_x,
            jac=lambda x: cached_evaluate(x).gradient_reduced,
            method="BFGS",
            callback=callback,
            options={
                "maxiter": remaining,
                "gtol": float(config.gtol),
                "disp": False,
            },
        )
        result_x = np.asarray(result.x, dtype=float)
        result_phase = _gauge_expand(result_x)
        result_evaluation = cached_evaluate(result_x)
        total_nfev += int(getattr(result, "nfev", 0) or 0)
        total_njev += int(getattr(result, "njev", 0) or 0)
        reported_nit = int(getattr(result, "nit", segment_accepted) or 0)
        if reported_nit != segment_accepted:
            raise RuntimeError(
                "SciPy callback count and reported BFGS iterations disagree: "
                f"{segment_accepted} != {reported_nit}"
            )
        segment_rows.append(
            {
                "segment_index": int(segment_index),
                "cumulative_start_iteration": int(accepted - segment_accepted),
                "accepted_iterations": int(segment_accepted),
                "cumulative_end_iteration": int(accepted),
                "requested_remaining_iterations": int(remaining),
                "start_phase_content_id": digest_array(segment_start_phase),
                "terminal_phase_content_id": digest_array(result_phase),
                "objective": float(result_evaluation.value),
                "gradient_norm": float(
                    np.linalg.norm(result_evaluation.gradient_reduced)
                ),
                "success": bool(getattr(result, "success", False)),
                "status": int(getattr(result, "status", -999)),
                "message": str(getattr(result, "message", "")),
                "nfev": int(getattr(result, "nfev", 0) or 0),
                "njev": int(getattr(result, "njev", 0) or 0),
                "runtime_sec": float(time.perf_counter() - segment_start),
                "restart_from_identical_terminal_phase": bool(segment_index > 0),
            }
        )
        x_current = result_x
        if accepted < cumulative_cap and segment_accepted == 0:
            if record_stalled_terminal:
                stalled_terminal = True
                break
            raise RuntimeError(
                "Conventional BFGS restart accepted zero iterations at cumulative "
                f"iteration {accepted}; the fixed protocol forbids perturbing the "
                "terminal phase or changing the optimizer."
            )

    missing = sorted(set(requested).difference(checkpoint_phases))
    terminal_phase = _gauge_expand(x_current)
    terminal_evaluation = evaluate_reduced(x_current)
    if missing and not stalled_terminal:
        raise RuntimeError(f"Cumulative BFGS missed checkpoints {missing}")
    if stalled_terminal:
        # Retain the actual unsuccessful terminal state at later requested
        # budgets; never label those copies as additional accepted iterations.
        for budget in missing:
            checkpoint_phases[budget] = terminal_phase.copy()
            checkpoint_rows.append(dict(requested_budget=budget, accepted_iteration=int(accepted),
                actual_terminal_iteration=int(accepted), is_early_terminal=True,
                optimizer_success=False, terminal_message=str(segment_rows[-1]['message']),
                objective=float(terminal_evaluation.value),
                gradient_norm=float(np.linalg.norm(terminal_evaluation.gradient_reduced)),
                phase_content_id=digest_array(terminal_phase)))
    return CumulativeBFGSResult(
        problem_fingerprint=problem.fingerprint,
        config_fingerprint=digest_payload(config.to_payload()),
        seed=int(seed),
        initial_phase_hash=digest_array(initial_full),
        terminal_phase_rad=terminal_phase,
        checkpoint_phases_rad={
            key: checkpoint_phases[key] for key in requested
        },
        checkpoint_records=tuple(checkpoint_rows),
        segment_records=tuple(segment_rows),
        accepted_iterations=int(accepted),
        nfev=int(total_nfev),
        njev=int(total_njev),
        terminal_objective=float(terminal_evaluation.value),
        terminal_gradient_norm=float(
            np.linalg.norm(terminal_evaluation.gradient_reduced)
        ),
        force_epsilon=float(force_epsilon),
        uniformity_epsilon=float(uniformity_epsilon),
        solve_sec=float(time.perf_counter() - solve_start),
        termination_reason=("zero-accepted-restart-stall: " + str(segment_rows[-1]["message"])
                            if stalled_terminal else "cumulative-accepted-iteration-cap"),
    )


def _cached_node(
    ctx: p.PipelineContext,
    node: tuple[int, int],
    *,
    initial: np.ndarray,
    seed: int,
    conventional_config: MultitrapObjectiveConfig,
    fe_config: MultitrapObjectiveConfig,
    conventional_cap: int,
    checkpoints: tuple[int, ...],
) -> tuple[tuple[int, int], dict[str, Any]]:
    ix, iy = node
    u, v = float(DOMAIN_VALUES[ix]), float(DOMAIN_VALUES[iy])
    problem = make_problem(
        ctx,
        FIXED_BASE_TARGETS_M,
        u,
        v,
        case_prefix="fixed-long-triple-observation",
    )
    shared = {
        "schema_version": SCHEMA_VERSION,
        "problem": problem.fingerprint,
        "seed": int(seed),
        "initial_phase_content_id": digest_array(initial),
        "u": u,
        "v": v,
        "fixed_geometry": "accepted 0.80-contracted Triple chart",
    }
    conventional_payload = {
        **shared,
        "method": "Conventional",
        "config": conventional_config.to_payload(),
        "cumulative_cap": int(conventional_cap),
        "checkpoints": list(checkpoints),
        "stalled_endpoint_policy": RESTART_POLICY,
    }
    conventional, _, _ = ctx.cache.get_or_compute(
        "triple_long_conventional_cumulative",
        conventional_payload,
        lambda: _run_cumulative_conventional(
            problem,
            conventional_config,
            seed=seed,
            initial_phases=initial,
            cumulative_cap=conventional_cap,
            checkpoints=checkpoints,
        ),
        recompute=ctx.recompute,
    )
    fe_payload = {
        **shared,
        "contract": (
            "main-triple-fe-endpoint-v4-alpha3000-axial10-uniform-30000-cap"
        ),
        "method": "FE",
        "config": fe_config.to_payload(),
        "fixed_alpha_force": FE_ALPHA,
        "fixed_fe_curvature_wx": FE_CURVATURE_WX,
        "fixed_fe_curvature_wy": FE_CURVATURE_WY,
        "fixed_fe_curvature_wz": FE_CURVATURE_WZ,
        "iteration_cap": FE_CAP,
    }
    fe, _, _ = ctx.cache.get_or_compute(
        "triple_long_fixed_fe",
        fe_payload,
        lambda: run_multitrap(
            problem,
            fe_config,
            seed=seed,
            initial_phases=initial,
            record_history=True,
            metadata={
                "display_method": "FE",
                "fixed_alpha_force": FE_ALPHA,
                "fixed_fe_curvature_wx": FE_CURVATURE_WX,
                "fixed_fe_curvature_wy": FE_CURVATURE_WY,
                "fixed_fe_curvature_wz": FE_CURVATURE_WZ,
                "u": u,
                "v": v,
            },
        ),
        recompute=ctx.recompute,
    )
    return node, {
        "problem": problem,
        "conventional_config": conventional_config,
        "fe_config": fe_config,
        "Conventional": conventional,
        "FE": fe,
        "FE_phase": np.asarray(fe.phases, dtype=float),
        "Conventional_checkpoint_phases": {
            int(key): np.asarray(value, dtype=float)
            for key, value in conventional.checkpoint_phases_rad.items()
        },
        "targets_m": np.asarray(
            [stencil.target_m for stencil in problem.stencils], dtype=float
        ),
    }

