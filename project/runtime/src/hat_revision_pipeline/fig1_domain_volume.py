"""Cartesian Single-target convergence audit for Figure 1 and its appendix.

The deposited branch chart is intentionally not reused here: its first chart
coordinate changes x and z together.  This module generates a true Cartesian
XZ section and a separate sparse XYZ volume.  Conventional and FE always see
the same target-local cold start and the same 10,000-iteration cap.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import math
import os
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import pipeline as p
from .cache import digest_array
from .figure_presentation import add_adjacent_colorbar, outside_grid_panel_labels
from .sota import solve_corrected_gorkov_fe
from .style import COLORS
from .sweeps import _matched_seed


METHODS = ("Conventional", "FE")
METHOD_OBJECTIVES = {"Conventional": "Conventional", "FE": "Force-Equilibrium"}
ITERATION_CAP = 10_000
X_BOUNDS_M = (-0.012, 0.012)
Y_BOUNDS_M = (-0.012, 0.012)
Z_BOUNDS_M = (0.040, 0.060)


def _coordinate_id(target_m: Sequence[float]) -> str:
    x, y, z = (float(value) for value in target_m)
    return f"x{x:+.6f}_y{y:+.6f}_z{z:+.6f}"


def _run_one(
    ctx: p.PipelineContext,
    target_m: np.ndarray,
    restart: int,
    method: str,
) -> tuple[dict[str, Any], Any]:
    target_id = _coordinate_id(target_m)
    seed = _matched_seed(ctx.config.random_seed + 401, target_id, int(restart))
    objective_name = METHOD_OBJECTIVES[method]
    objective = p._objective_at(ctx, objective_name, target_m)
    initial = np.random.default_rng(seed).uniform(
        -np.pi, np.pi, objective.n_transducers
    )
    payload = {
        "contract": "fig1-cartesian-domain-alpha10-uniform10000-serial-timing-v2",
        "method": method,
        "objective_method": asdict(objective.method),
        "target_m": np.asarray(target_m, dtype=float).tolist(),
        "restart": int(restart),
        "seed": int(seed),
        "initial_phase": digest_array(initial),
        "positions": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "frequency_hz": float(ctx.config.frequency_hz),
        "maxiter": ITERATION_CAP,
        "gtol": float(ctx.config.gtol),
    }
    result, cache_status, _ = ctx.cache.get_or_compute(
        "fig1_cartesian_domain_run",
        payload,
        lambda: solve_corrected_gorkov_fe(
            objective,
            initial,
            maxiter=ITERATION_CAP,
            gtol=ctx.config.gtol,
        ),
        recompute=ctx.recompute,
    )
    reduced_dimension = max(objective.n_transducers - 1, 1)
    row = {
        "method": method,
        "objective_method": objective_name,
        "alpha_per_m": float(objective.method.alpha_per_m),
        "target_id": target_id,
        "target_x_m": float(target_m[0]),
        "target_y_m": float(target_m[1]),
        "target_z_m": float(target_m[2]),
        "restart": int(restart),
        "seed": int(seed),
        "iteration_cap": ITERATION_CAP,
        "iterations": int(result.iterations),
        "evaluations": int(result.evaluations),
        "wall_time_s": float(result.history_wall_s[-1]),
        "objective": float(result.objective),
        "terminal_gradient_norm": float(result.gradient_norm),
        "terminal_gradient_rms": float(result.gradient_norm / math.sqrt(reduced_dimension)),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "phase_content_id": digest_array(result.phase_rad),
        "cache_status": str(cache_status),
        "correct_sign_gorkov": True,
        "run_mode": ctx.config.mode,
    }
    return row, result


def cartesian_domain_bank(
    ctx: p.PipelineContext,
    *,
    x_values_m: Iterable[float],
    y_values_m: Iterable[float],
    z_values_m: Iterable[float],
    restarts: int,
    table_prefix: str,
) -> dict[str, Any]:
    """Run one explicitly declared Cartesian target bank."""

    x_values = np.asarray(tuple(x_values_m), dtype=float)
    y_values = np.asarray(tuple(y_values_m), dtype=float)
    z_values = np.asarray(tuple(z_values_m), dtype=float)
    if not (len(x_values) and len(y_values) and len(z_values)):
        raise ValueError("Cartesian target axes must be nonempty")
    if int(restarts) < 1:
        raise ValueError("restarts must be positive")
    memo_key = (
        "fig1_cartesian_domain",
        tuple(x_values), tuple(y_values), tuple(z_values), int(restarts),
    )
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]

    targets = [
        np.asarray((x, y, z), dtype=float)
        for y in y_values
        for z in z_values
        for x in x_values
    ]
    jobs = [
        (target, restart, method)
        for target in targets
        for restart in range(int(restarts))
        for method in METHODS
    ]
    worker_default = 1
    workers = max(
        1,
        min(len(jobs), int(os.environ.get("HAT_FIG1_DOMAIN_WORKERS", worker_default))),
    )
    ordered: list[tuple[dict[str, Any], Any] | None] = [None] * len(jobs)
    if workers == 1:
        for index, (target, restart, method) in enumerate(jobs):
            ordered[index] = _run_one(ctx, target, restart, method)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(_run_one, ctx, target, restart, method): index
                for index, (target, restart, method) in enumerate(jobs)
            }
            for future in as_completed(future_to_index):
                ordered[future_to_index[future]] = future.result()
    completed = [item for item in ordered if item is not None]
    raw = pd.DataFrame([item[0] for item in completed])
    summary = (
        raw.groupby(
            ["method", "alpha_per_m", "target_id", "target_x_m", "target_y_m", "target_z_m"],
            sort=True,
            dropna=False,
        )
        .agg(
            run_count=("restart", "size"),
            wall_time_median_s=("wall_time_s", "median"),
            wall_time_q25_s=("wall_time_s", lambda values: values.quantile(0.25)),
            wall_time_q75_s=("wall_time_s", lambda values: values.quantile(0.75)),
            terminal_gradient_rms_median=("terminal_gradient_rms", "median"),
            terminal_gradient_rms_q25=("terminal_gradient_rms", lambda values: values.quantile(0.25)),
            terminal_gradient_rms_q75=("terminal_gradient_rms", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    definition = pd.DataFrame(
        [
            {
                "table_prefix": table_prefix,
                "x_values_m": ",".join(f"{value:.6f}" for value in x_values),
                "y_values_m": ",".join(f"{value:.6f}" for value in y_values),
                "z_values_m": ",".join(f"{value:.6f}" for value in z_values),
                "target_count": len(targets),
                "matched_restarts_per_target": int(restarts),
                "method_count": len(METHODS),
                "iteration_cap_per_run": ITERATION_CAP,
                "worker_count": workers,
                "shared_initialization_within_method_pair": True,
                "correct_sign_gorkov": True,
            }
        ]
    )
    ctx.tables[f"{table_prefix}_runs"] = raw.copy()
    ctx.tables[f"{table_prefix}_summary"] = summary.copy()
    ctx.tables[f"{table_prefix}_definition"] = definition
    value = {"raw": raw, "summary": summary, "runs": completed, "definition": definition}
    ctx.memo[memo_key] = value
    return value


def figure_1_xz_domain_bank(ctx: p.PipelineContext) -> dict[str, Any]:
    x_points = 7 if ctx.config.full else int(os.environ.get("HAT_QUICK_FIG1_XZ_X_POINTS", "5"))
    z_points = 7 if ctx.config.full else int(os.environ.get("HAT_QUICK_FIG1_XZ_Z_POINTS", "3"))
    restarts = 3 if ctx.config.full else int(os.environ.get("HAT_QUICK_FIG1_DOMAIN_RESTARTS", "1"))
    scale = ctx.memo.get("study_scale", {})
    if scale.get("profile") == "supersmoke":
        x_points = int(scale.get("first_result_domain_x_points", 3))
        z_points = int(scale.get("first_result_domain_z_points", 3))
        restarts = int(scale.get("first_result_domain_restarts", 1))
    return cartesian_domain_bank(
        ctx,
        x_values_m=np.linspace(*X_BOUNDS_M, x_points),
        y_values_m=(0.0,),
        z_values_m=np.linspace(*Z_BOUNDS_M, z_points),
        restarts=restarts,
        table_prefix="fig1_xz_domain",
    )


def appendix_3d_domain_bank(ctx: p.PipelineContext) -> dict[str, Any]:
    points_xy = 5 if ctx.config.full else 3
    points_z = 5 if ctx.config.full else 3
    restarts = 3 if ctx.config.full else int(os.environ.get("HAT_QUICK_FIG1_DOMAIN_RESTARTS", "1"))
    return cartesian_domain_bank(
        ctx,
        x_values_m=np.linspace(*X_BOUNDS_M, points_xy),
        y_values_m=np.linspace(*Y_BOUNDS_M, points_xy),
        z_values_m=np.linspace(*Z_BOUNDS_M, points_z),
        restarts=restarts,
        table_prefix="appendix_fig1_xyz_domain",
    )


def _shared_norm(summary: pd.DataFrame, column: str) -> mpl.colors.Normalize:
    values = summary[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"No finite values for {column}")
    low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + max(abs(low), 1.0) * np.finfo(float).eps
    return mpl.colors.Normalize(vmin=low, vmax=high)


def _volume_panel(
    ax: mpl.axes.Axes,
    table: pd.DataFrame,
    method: str,
    column: str,
    norm: mpl.colors.Normalize,
    *,
    cmap: str,
) -> mpl.collections.PathCollection:
    selected = table[table["method"].eq(method)]
    return ax.scatter(
        1.0e3 * selected["target_x_m"],
        1.0e3 * selected["target_y_m"],
        1.0e3 * selected["target_z_m"],
        c=selected[column], cmap=cmap, norm=norm,
        s=62, edgecolor="white", linewidth=0.45, depthshade=False,
    )


def appendix_figure_1_xyz_convergence_audit(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Separate 3-D evidence that the Figure 1 comparison persists in volume."""

    stem = "Appendix_Figure_1_xyz_convergence_audit"
    if stem in ctx.figures:
        return ctx.figures[stem]
    summary = appendix_3d_domain_bank(ctx)["summary"]
    metrics = (
        ("wall_time_median_s", "viridis", r"$t_{\mathrm{solve}}$ (s)"),
        ("terminal_gradient_rms_median", "magma", r"terminal $g_{\mathrm{RMS}}$"),
    )
    fig = plt.figure(figsize=(10.8, 9.5))
    grid = fig.add_gridspec(2, 2, hspace=0.19, wspace=0.17)
    specs: list[mpl.gridspec.SubplotSpec] = []
    for row, (column, cmap, label) in enumerate(metrics):
        norm = _shared_norm(summary, column)
        for col, method in enumerate(METHODS):
            spec = grid[row, col]
            specs.append(spec)
            ax = fig.add_subplot(spec, projection="3d")
            mappable = _volume_panel(ax, summary, method, column, norm, cmap=cmap)
            ax.set(xlabel=r"$x^*$ (mm)", ylabel=r"$y^*$ (mm)", zlabel=r"$z^*$ (mm)")
            ax.set_xlim(1e3 * X_BOUNDS_M[0], 1e3 * X_BOUNDS_M[1])
            ax.set_ylim(1e3 * Y_BOUNDS_M[0], 1e3 * Y_BOUNDS_M[1])
            ax.set_zlim(1e3 * Z_BOUNDS_M[0], 1e3 * Z_BOUNDS_M[1])
            ax.view_init(elev=25, azim=-58)
            ax.set_title(method, fontsize=10.5, pad=2.0)
            colorbar = fig.colorbar(mappable, ax=ax, fraction=0.036, pad=0.045, shrink=0.74)
            colorbar.set_label(label, fontsize=8.5)
            colorbar.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.94, bottom=0.055)
    outside_grid_panel_labels(fig, specs, tuple("abcd"), dx=0.014, dy=0.006)
    return p._save_figure(ctx, fig, stem, tight=False)


__all__ = [
    "ITERATION_CAP",
    "appendix_3d_domain_bank",
    "appendix_figure_1_xyz_convergence_audit",
    "cartesian_domain_bank",
    "figure_1_xz_domain_bank",
]
