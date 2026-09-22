"""Final eight-figure composition for the RinEng revision.
This module only composes the corrected-sign producers already deposited with
the revision runtime.  The one new numerical producer is the explicitly
requested connected Triple domain; its deformation, caps, and representative
node are constants recorded in the exported tables rather than plot-time
choices.

All current and recovered runs are SMOKE, including runs at expanded budgets.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.signal import find_peaks
from skimage import measure

from . import pipeline as p
from .alpha1000_branch import fixed_single_branch_pack
from .branch_figures import COMPONENTS, target_demodulated_states
from .cache import digest_array
from .gorkov_core import (
    Condition,
    MethodSpec,
    SingleTargetObjective,
    formula_self_test,
)
from .fig67_layout import (
    compact_force_concentration_labels,
)
from .fig1_domain_volume import figure_1_xz_domain_bank
from .fig7_xy_patch import compose_figure_7_xy
from .fig8_methods_only import figure_8_methods_only as _figure_8_methods_only
from .figure_presentation import (
    add_adjacent_colorbar,
    outside_grid_panel_labels,
    terse_math_label,
)
from .multitrap import (
    MultiTrapProblem,
    MultitrapObjectiveConfig,
    SPUComponents,
    SphereInFluid,
    TargetStencil,
    effective_weight_force_target_n,
    evaluate_multitrap_objective,
    run_multitrap,
)
from .sota import (
    benchmark_synthesis_method_factory,
    iterative_back_projection,
    phase_only_signature_projection_audit,
    solve_corrected_gorkov_fe,
)
from .diff_pat import DIFF_PAT_CONFIG, diff_pat_phase_only
from .fixed_fe_endpoints import (
    SINGLE_FE_ENDPOINT,
    TRIPLE_FE_ENDPOINT,
    fixed_single_fe_method_spec,
)
from .style import (
    COLORS,
    TARGET_COLOR,
    field_panel,
    set_panel_title,
    shared_power_norm,
    target_marker,
)
from .main_triple_endpoint import ensure_main_triple_endpoint


REPORT_GRADIENT_NORM = 1.0e-3
TRI3_BASE_TARGETS_M = np.asarray(
    ((0.0, 0.013, 0.030), (-0.011, -0.009, 0.030), (0.011, -0.009, 0.030)),
    dtype=float,
)
TRI3_DELTA_M = 0.001
ALPHA_VALUES = (1.0e3, 1.0e6, 1.0e7)


def _plot_points(ctx: p.PipelineContext, name: str, original: int) -> int:
    """Reduce computed sampling grids only for the explicit plot preview."""
    scale = ctx.memo.get("study_scale", {})
    if scale.get("profile") == "supersmoke":
        return int(scale.get(name, original))
    return int(original)


def _mode_note(ctx: p.PipelineContext) -> str:
    if ctx.memo.get("study_scale", {}).get("profile") == "supersmoke":
        return "SUPERSMOKE — minimal plot preview"
    if ctx.config.full:
        if ctx.memo.get("study_scale", {}).get("use_smoke_validation"):
            return "FULL — expanded populations; smoke validation numerics"
        return "FULL — expanded populations with refined validation numerics"
    if os.environ.get("HAT_PROTOTYPE_LABEL", "0").strip().lower() in {"1", "true", "yes"}:
        return "PROTOTYPE LABEL REQUESTED"
    return "PAPER-SCALE SMOKE — accepted manuscript protocol"


def _guard() -> None:
    report = formula_self_test()
    if report.get("passed") is not True:
        raise RuntimeError(f"Correct-sign Gor'kov guard failed: {report}")


def _save(ctx: p.PipelineContext, fig: mpl.figure.Figure, stem: str) -> mpl.figure.Figure:
    if os.environ.get("HAT_HIDE_MODE_NOTE", "0").strip().lower() not in {
        "1", "true", "yes"
    }:
        fig.text(
            0.995, 0.004, _mode_note(ctx), ha="right", va="bottom",
            fontsize=6.7, color="0.38",
        )
    from .publication_layout import polish_main_figure
    polish_main_figure(fig)
    return p._save_figure(ctx, fig, stem, tight=False)


def _central_seed_rows(ctx: p.PipelineContext) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    root_ix, root_iy = ctx.branch_bank.root_target
    fe = ctx.artifact.anchor_table[
        ctx.artifact.anchor_table["method"].eq("Force-Equilibrium")
        & ctx.artifact.anchor_table["ix"].eq(root_ix)
        & ctx.artifact.anchor_table["iy"].eq(root_iy)
    ].copy()
    conventional = ctx.artifact.trajectory_table[
        ctx.artifact.trajectory_table["method"].eq("Conventional")
        & ctx.artifact.trajectory_table["ix"].eq(root_ix)
        & ctx.artifact.trajectory_table["iy"].eq(root_iy)
    ].copy()
    seeds = sorted(set(fe["seed"].astype(int)) & set(conventional["seed"].astype(int)))
    if not seeds:
        raise RuntimeError("No paired central-target seed was found")
    return fe, conventional, seeds


def _single_vortex_bank(ctx: p.PipelineContext) -> dict[str, Any]:
    """Run the paired correct-sign single-target bank and preserve histories."""

    if "final_single_vortex_bank" in ctx.memo:
        return ctx.memo["final_single_vortex_bank"]
    _guard()
    requested = 100 if ctx.config.full else int(os.environ.get("HAT_QUICK_SINGLE_PAIRS", "4"))
    requested = _plot_points(ctx, "first_result_seed_count", requested)
    conventional_cap = 10_000
    seeds = [ctx.config.random_seed + index for index in range(requested)]
    rows: list[dict[str, Any]] = []
    runs: dict[tuple[str, int], Any] = {}
    for pair_index, seed in enumerate(seeds):
        initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ctx.positions_m))
        for display, method in (("Conventional", "Conventional"), ("FE", "Force-Equilibrium")):
            method_cap = (
                conventional_cap
                if display == "Conventional"
                else int(SINGLE_FE_ENDPOINT.maxiter)
            )
            objective = p._objective_at(ctx, method)
            payload = {
                "contract": "final-fig1-paired-correct-sign-v5-alpha10-uniform-10000-cap",
                "method": method,
                "objective_method": asdict(objective.method),
                "target_m": p.MAIN_TARGET_M.tolist(),
                "seed": int(seed),
                "initial_phase": digest_array(initial),
                "maxiter": int(method_cap),
                "gtol": float(ctx.config.gtol),
                "report_gradient_norm": REPORT_GRADIENT_NORM,
            }
            result, _, _ = ctx.cache.get_or_compute(
                "final_single_vortex_run",
                payload,
                lambda objective=objective, initial=initial, method_cap=method_cap: solve_corrected_gorkov_fe(
                    objective, initial, maxiter=method_cap, gtol=ctx.config.gtol
                ),
                recompute=ctx.recompute,
            )
            runs[(display, pair_index)] = result
            rows.append(
                {
                    "method": display,
                    "optimizer": result.optimizer,
                    "evaluator_backend": "native" if display == "Conventional" else "compact-single-float64-v1",
                    "alpha_per_m": float(objective.method.alpha_per_m),
                    "pair_index": pair_index,
                    "seed": int(seed),
                    "target_x_m": p.MAIN_TARGET_M[0],
                    "target_y_m": p.MAIN_TARGET_M[1],
                    "target_z_m": p.MAIN_TARGET_M[2],
                    "iteration_cap": int(method_cap),
                    "iterations": int(result.iterations),
                    "evaluations": int(result.evaluations),
                    "wall_time_s": float(result.history_wall_s[-1]),
                    "objective": float(result.objective),
                    "gradient_norm": float(result.gradient_norm),
                    "near_stationary": bool(result.gradient_norm <= REPORT_GRADIENT_NORM),
                    "optimizer_success": bool(result.success),
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "phase_content_id": digest_array(result.phase_rad),
                    "correct_sign_gorkov": True,
                    "run_mode": ctx.config.mode,
                }
            )
    table = pd.DataFrame(rows)
    ctx.tables["fig1_single_vortex_runs"] = table
    ctx.tables["fig1_single_vortex_iterations_for_table_only"] = table[
        ["method", "pair_index", "seed", "iterations", "evaluations", "iteration_cap"]
    ].copy()
    value = {
        "runs": runs,
        "table": table,
        "caps": {
            "Conventional": int(conventional_cap),
            "FE": int(SINGLE_FE_ENDPOINT.maxiter),
        },
        "seeds": seeds,
    }
    ctx.memo["final_single_vortex_bank"] = value
    return value


def _normalized_remaining(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    start, terminal = float(values[0]), float(values[-1])
    scale = max(abs(start - terminal), np.finfo(float).tiny)
    return np.clip((values - terminal) / scale, -0.03, 1.03)


def _accepted_iteration_axis(result: Any) -> np.ndarray:
    """Map the stored objective history back to accepted BFGS iterations."""

    count = len(np.asarray(result.history_objective))
    terminal = max(int(result.iterations), 0)
    if count == terminal + 1:
        return np.arange(count, dtype=float)
    if count == terminal + 2:
        return np.r_[np.arange(terminal + 1, dtype=float), float(terminal)]
    if count <= 1:
        return np.zeros(count, dtype=float)
    return np.linspace(0.0, float(terminal), count)


def _termination_mode(result: Any, iteration_cap: int) -> str:
    """Classify the actual solver exit without collapsing all failures together."""

    if float(result.gradient_norm) <= REPORT_GRADIENT_NORM:
        return "Near-stationary"
    message = str(result.message).lower()
    if int(result.iterations) >= int(iteration_cap) or any(
        token in message for token in ("maximum number of iterations", "maxiter", "iteration limit")
    ):
        return "Max iteration"
    if any(
        token in message
        for token in (
            "precision loss",
            "desired error not necessarily achieved",
            "line search",
            "safeguard",
            "stall",
        )
    ) or not bool(result.success):
        return "Nonzero-gradient stall"
    return "Other"


def _field_montage(
    fig: mpl.figure.Figure,
    spec: mpl.gridspec.SubplotSpec,
    sections: Mapping[str, Any],
    norm: Normalize,
    method: str,
) -> tuple[list[plt.Axes], mpl.image.AxesImage]:
    inner = spec.subgridspec(1, 2, wspace=0.48)
    axes = [fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])]
    offset = 1e3 * np.asarray(sections["xy_axis_m"])
    zoffset = 1e3 * np.asarray(sections["z_axis_m"])
    xaxis = offset
    yaxis = offset
    zaxis = zoffset
    image_xy = field_panel(
        axes[0], sections["xy"], (xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]), norm=norm,
        xlabel=r"$x-x^*$ (mm)", ylabel=r"$y-y^*$ (mm)",
        target_mm=(0.0, 0.0),
    )
    image_xz = field_panel(
        axes[1], sections["xz"], (xaxis[0], xaxis[-1], zaxis[0], zaxis[-1]), norm=norm,
        xlabel=r"$x-x^*$ (mm)", ylabel=r"$z-z^*$ (mm)",
        target_mm=(0.0, 0.0),
    )
    axes[0].text(
        0.02, 1.055, method, transform=axes[0].transAxes,
        ha="left", va="bottom", fontsize=10.5, fontweight="semibold", clip_on=False,
    )
    for axis, image in zip(axes, (image_xy, image_xz), strict=True):
        axis.set_box_aspect(1)
        add_adjacent_colorbar(
            image, axis, size="3.4%", pad="2.4%", label_key="pressure",
            labelsize=7.2, ticksize=6.4,
        )
    axes[1].set_xlim(float(zaxis[0]), float(zaxis[-1]))
    return axes, image_xy


def _fig1_morphology_volume(
    ctx: p.PipelineContext,
    phase: np.ndarray,
    target_m: np.ndarray,
    *,
    samples: int = 51,
    half_width_m: float = 0.018,
) -> dict[str, np.ndarray]:
    """Sample the pre-revision target-centred volume for transparent shells."""

    samples = _plot_points(ctx, "pressure_points", samples)
    payload = {
        "contract": "fig1-pre-revision-transparent-volume-v1",
        "phase": digest_array(np.asarray(phase, dtype=float)),
        "target_m": np.asarray(target_m, dtype=float).tolist(),
        "samples": int(samples),
        "half_width_m": float(half_width_m),
    }

    def producer() -> dict[str, np.ndarray]:
        axis = np.linspace(-half_width_m, half_width_m, int(samples))
        xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
        relative = np.stack((xx, yy, zz), axis=-1)
        values = p._pressure_abs_chunked(
            p._array_field(ctx, phase),
            np.asarray(target_m, dtype=float) + relative.reshape(-1, 3),
        ).reshape(relative.shape[:-1])
        return {"axis_m": axis, "pressure_pa": values}

    value, _, _ = ctx.cache.get_or_compute(
        "fig1_morphology_volume",
        payload,
        producer,
        recompute=ctx.recompute,
    )
    return value


def _fig1_shared_iso_spec(
    volumes: Sequence[np.ndarray],
) -> tuple[np.ndarray, mpl.colors.Normalize]:
    finite = np.concatenate(
        [np.asarray(volume, dtype=float)[np.isfinite(volume)] for volume in volumes]
    )
    if not len(finite):
        raise ValueError("Figure 1 morphology volumes contain no finite pressure")
    quantiles = np.asarray((84.0, 90.0, 94.0, 97.0, 98.8, 99.3, 99.7))
    levels = np.unique(np.percentile(finite, quantiles))
    pmin = float(np.percentile(finite, 84.0))
    pmax = float(np.percentile(finite, 99.7))
    if pmax <= pmin:
        pmax = pmin + max(abs(pmin), 1.0) * np.finfo(float).eps
    return levels, mpl.colors.PowerNorm(
        gamma=0.60, vmin=pmin, vmax=pmax, clip=True
    )


def _fig1_isosurface_panel(
    ax: mpl.axes.Axes,
    volume: Mapping[str, np.ndarray],
    levels: np.ndarray,
    norm: mpl.colors.Normalize,
) -> None:
    """Render the supplied pre-revision marching-cubes visual grammar."""

    pressure = np.asarray(volume["pressure_pa"], dtype=float)
    axis_mm = 1.0e3 * np.asarray(volume["axis_m"], dtype=float)
    step = float(axis_mm[1] - axis_mm[0])
    cmap = mpl.colormaps["viridis"]
    high_rgb = np.asarray(cmap(1.0))[:3]
    denom = max(float(norm.vmax - norm.vmin), 1.0e-30)
    for level in levels:
        if not (float(np.nanmin(pressure)) < float(level) < float(np.nanmax(pressure))):
            continue
        try:
            vertices, faces, _, _ = measure.marching_cubes(
                pressure, level=float(level), spacing=(step, step, step)
            )
        except (RuntimeError, ValueError):
            continue
        vertices += axis_mm[0]
        face_vertices = vertices[faces]
        level_fraction = float(
            np.clip((float(level) - float(norm.vmin)) / denom, 0.0, 1.0)
        )
        rgba = np.asarray(cmap(norm(float(level))), dtype=float)
        fade = float(np.clip((1.0 - level_fraction) ** 1.3, 0.0, 1.0))
        highlight = float(np.clip((level_fraction - 0.72) / 0.28, 0.0, 1.0))
        rgba[:3] = rgba[:3] * (1.0 - 0.82 * fade) + 0.82 * fade
        rgba[:3] = np.clip(
            rgba[:3] * (0.88 + 0.12 * level_fraction)
            + high_rgb * (0.18 * highlight),
            0.0,
            1.0,
        )
        rgba[3] = float(np.clip(0.015 + 0.06 * level_fraction + 0.92 * highlight**2.2, 0.01, 0.96))
        mesh = Poly3DCollection(
            face_vertices,
            facecolors=np.repeat(rgba[None, :], len(face_vertices), axis=0),
            edgecolor="none",
            linewidths=0.0,
            antialiased=True,
        )
        ax.add_collection3d(mesh)
    limit = float(max(abs(axis_mm[0]), abs(axis_mm[-1])))
    ax.scatter([0.0], [0.0], [0.0], c="crimson", s=28, marker="x", linewidth=1.5)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=26, azim=-58)
    ax.set_xlabel(r"$x-x^*$ (mm)", fontsize=7.4, labelpad=-2)
    ax.set_ylabel(r"$y-y^*$ (mm)", fontsize=7.4, labelpad=-2)
    ax.set_zlabel(r"$z-z^*$ (mm)", fontsize=7.4, labelpad=-2)
    ax.tick_params(labelsize=6.5, pad=-1)
    ax.grid(alpha=0.14)


def _fig1_xz_grid_values(
    summary: pd.DataFrame,
    method: str,
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = summary[
        summary["method"].eq(method) & np.isclose(summary["target_y_m"], 0.0)
    ]
    x = np.sort(selected["target_x_m"].unique().astype(float))
    z = np.sort(selected["target_z_m"].unique().astype(float))
    values = (
        selected.pivot(index="target_z_m", columns="target_x_m", values=column)
        .reindex(index=z, columns=x)
        .to_numpy(dtype=float)
    )
    if values.shape != (len(z), len(x)) or not np.isfinite(values).all():
        raise RuntimeError(f"Incomplete Figure 1 XZ grid for {method}: {column}")
    return x, z, values


def _fig1_domain_pair_panel(
    fig: mpl.figure.Figure,
    spec: mpl.gridspec.SubplotSpec,
    summary: pd.DataFrame,
    column: str,
    *,
    cmap: str,
    colorbar_label: str,
) -> tuple[plt.Axes, plt.Axes]:
    all_values = summary[column].to_numpy(dtype=float)
    finite = all_values[np.isfinite(all_values)]
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + max(abs(low), 1.0) * np.finfo(float).eps
    shared_norm = Normalize(vmin=low, vmax=high)
    inner = spec.subgridspec(1, 3, width_ratios=(1.0, 1.0, 0.07), wspace=0.20)
    axes = (fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1]))
    cax = fig.add_subplot(inner[0, 2])
    image: mpl.image.AxesImage | None = None
    for index, (ax, method) in enumerate(zip(axes, ("Conventional", "FE"), strict=True)):
        x, z, values = _fig1_xz_grid_values(summary, method, column)
        image = ax.imshow(
            values,
            origin="lower",
            extent=(1e3 * x[0], 1e3 * x[-1], 1e3 * z[0], 1e3 * z[-1]),
            interpolation="nearest",
            aspect="auto",
            cmap=cmap,
            norm=shared_norm,
        )
        ax.set_xticks([-10.0, 0.0, 10.0])
        ax.set_yticks([40.0, 50.0, 60.0])
        ax.set_box_aspect(1)
        ax.set_xlabel(r"$x^*$ (mm)", fontsize=8.0)
        ax.set_ylabel(r"$z^*$ (mm)" if index == 0 else "", fontsize=8.0)
        if index:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=7.0)
        ax.set_title(method, fontsize=8.6, pad=2.0)
        ax.grid(False)
    if image is None:
        raise AssertionError("Figure 1 domain pair did not produce an image")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(colorbar_label, fontsize=8.0, labelpad=3.0)
    colorbar.ax.tick_params(labelsize=7.0)
    if column == "terminal_gradient_rms_median":
        colorbar.formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        colorbar.formatter.set_powerlimits((-2, 2))
        colorbar.update_ticks()
    return axes


def _fig1_pressure_panel(
    ax: plt.Axes,
    values: np.ndarray,
    extent: tuple[float, float, float, float],
    norm: Normalize,
    *,
    plane: str,
) -> None:
    image = field_panel(
        ax,
        values,
        extent,
        norm=norm,
        xlabel=r"$x-x^*$ (mm)",
        ylabel=(r"$y-y^*$ (mm)" if plane == "xy" else r"$z-z^*$ (mm)"),
    )
    import matplotlib.patheffects as pe
    marker = ax.scatter(0.0, 0.0, marker="x", color=TARGET_COLOR, s=62,
                        linewidth=1.8, zorder=20)
    marker.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
    if plane == "xz":
        ax.set_aspect("auto")
    ax.set_box_aspect(1)
    add_adjacent_colorbar(
        image,
        ax,
        size="3.2%",
        pad="2.0%",
        label=r"$|p|$ (Pa)",
        labelsize=7.2,
        ticksize=6.4,
    )


def figure_1_single_vortex_convergence(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Introduce the convergence contrast across targets and matched Vortex fields."""

    stem = "Figure_1_single_vortex_convergence"
    if stem in ctx.figures:
        return ctx.figures[stem]
    bank = _single_vortex_bank(ctx)
    display_seed = ctx.memo.get("production_single_display_seed")
    display_index = (bank["seeds"].index(display_seed)
                     if ctx.memo.get("production_main") and display_seed in bank["seeds"] else 0)
    c_run = bank["runs"][("Conventional", display_index)]
    fe_run = bank["runs"][("FE", display_index)]
    target = p.MAIN_TARGET_M

    run_table = bank["table"].copy()
    run_table["termination_mode"] = [
        _termination_mode(
            bank["runs"][(row.method, int(row.pair_index))],
            int(row.iteration_cap),
        )
        for row in run_table.itertuples(index=False)
    ]
    ctx.tables["fig1_single_vortex_runs"] = run_table
    ctx.tables["fig1_single_vortex_outcome_taxonomy"] = (
        run_table.groupby(["method", "termination_mode"], sort=False)
        .size()
        .rename("count")
        .reset_index()
        .merge(
            run_table.groupby("method", sort=False).size().rename("total").reset_index(),
            on="method",
            how="left",
            validate="many_to_one",
        )
        .assign(fraction=lambda table: table["count"] / table["total"])
    )

    domain_summary = figure_1_xz_domain_bank(ctx)["summary"]
    pressure_samples = _plot_points(ctx, "pressure_points", 181 if ctx.memo.get("production_main") else 161)
    def sections_for(run, zoom):
        payload = dict(contract="fig1-pressure-sections-v1", phase=digest_array(run.phase_rad),
            array=digest_array(ctx.positions_m), normals=digest_array(ctx.normals),
            config=asdict(ctx.config), target=np.asarray(target).tolist(),
            half_xy_m=0.075 / zoom, half_z_m=0.050 / zoom, samples=pressure_samples)
        value, _, _ = ctx.cache.get_or_compute("fig1_pressure_sections", payload,
            lambda: p._pressure_sections(p._array_field(ctx, run.phase_rad), target,
                half_xy_m=payload["half_xy_m"], half_z_m=payload["half_z_m"], samples=pressure_samples),
            recompute=ctx.recompute)
        return value
    broad_c, broad_fe = sections_for(c_run, 1.0), sections_for(fe_run, 1.0)
    c_sections, fe_sections = sections_for(c_run, 3.0), sections_for(fe_run, 3.0)
    pressure_norm = shared_power_norm(
        [broad_c["xy"], broad_c["xz"], broad_fe["xy"], broad_fe["xz"]],
        lower=0.0,
        upper=99.5,
    )
    ctx.tables["fig1_pressure_display"] = pd.DataFrame([dict(
        linear_magnification=3.0, xy_half_span_mm=25.0, z_half_span_mm=50.0/3.0,
        samples_per_axis=pressure_samples, shared_norm_vmin_pa=pressure_norm.vmin,
        shared_norm_vmax_pa=pressure_norm.vmax, shared_norm_gamma=pressure_norm.gamma,
        normalization_source="unchanged broad Conventional/FE xy/xz fields; pooled 99.5th percentile")])
    fig = plt.figure(figsize=(10.8, 13.8))
    grid = fig.add_gridspec(3, 2, height_ratios=(0.82, 1.0, 1.0),
                           hspace=0.36, wspace=0.52)
    ax_curve = fig.add_subplot(grid[0, 0])
    for label, run in (("Conventional", c_run), ("FE", fe_run)):
        if ctx.memo.get("production_main"):
            iteration_axis = np.arange(10001, dtype=float)
            population = np.stack([
                np.interp(iteration_axis, _accepted_iteration_axis(bank["runs"][(label, i)]),
                          _normalized_remaining(bank["runs"][(label, i)].history_objective))
                for i in range(len(bank["seeds"]))])
            lower, median, upper = np.quantile(population, [.25, .5, .75], axis=0)
            ax_curve.fill_between(iteration_axis, lower, upper, color=COLORS[label], alpha=.18, linewidth=0)
            ax_curve.plot(iteration_axis, median, lw=2.2, color=COLORS[label], label=label)
            ctx.tables[f"production_fig1_{label}_convergence"] = pd.DataFrame(dict(
                accepted_iteration=iteration_axis, q25=lower, median=median, q75=upper,
                independent_starts=len(bank["seeds"])))
        else:
            ax_curve.plot(_accepted_iteration_axis(run), _normalized_remaining(run.history_objective),
                          lw=2.2, color=COLORS[label], label=label)
    ax_curve.set_xlabel(terse_math_label("accepted_iteration"))
    ax_curve.set_ylabel(r"Normalized $\widetilde{J}$")
    ax_curve.set_ylim(-0.02, 1.05)
    ax_curve.legend(frameon=False, fontsize=8.5)
    _fig1_domain_pair_panel(fig, grid[0, 1], domain_summary, "wall_time_median_s",
                           cmap="viridis", colorbar_label=r"$t_{\mathrm{solve}}$ (s)")
    for row, (method, sections) in enumerate((("Conventional", c_sections), ("FE", fe_sections)), start=1):
        xy_axis = 1.0e3 * np.asarray(sections["xy_axis_m"], dtype=float)
        z_axis = 1.0e3 * np.asarray(sections["z_axis_m"], dtype=float)
        for column, plane in enumerate(("xy", "xz")):
            axis = fig.add_subplot(grid[row, column])
            extent = ((xy_axis[0], xy_axis[-1], xy_axis[0], xy_axis[-1]) if plane == "xy"
                      else (xy_axis[0], xy_axis[-1], z_axis[0], z_axis[-1]))
            _fig1_pressure_panel(axis, sections[plane], extent, pressure_norm, plane=plane)
            set_panel_title(axis, method)
    ctx.tables["fig1_panel_mapping"] = pd.DataFrame([
        dict(panel=letter, previous_panel=old, content=content)
        for letter, old, content in zip("abcdef", "abefhi", (
            "convergence", "XZ command time", "Conventional xy", "Conventional xz", "FE xy", "FE xz"))])
    fig.subplots_adjust(left=0.115, right=0.935, top=0.945, bottom=0.060)
    outside_grid_panel_labels(fig, tuple(grid[row,col] for row in range(3) for col in range(2)),
                              tuple("abcdef"), dx=0.016, dy=0.008)
    return _save(ctx, fig, stem)


def _load_authoritative_3d_branch_coordinates(
    ctx: p.PipelineContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the exact fixed-frame coordinates deposited by the supplied 3-D script."""

    root = ctx.project_root / "data" / "corrected_branch_evolution"
    anchor_path = root / "fixed_anchor_coordinates.csv"
    evolution_path = root / "conventional_iteration_coordinates_split.csv"
    if not anchor_path.exists() or not evolution_path.exists():
        raise FileNotFoundError(
            "The supplied fixed 3-D branch coordinate bundle is incomplete: "
            f"{anchor_path} / {evolution_path}"
        )
    mapping_table = ctx.branch_bank.medoid_table[
        ["component", "posthoc_legacy_winding_label"]
    ].drop_duplicates()
    if len(mapping_table) != len(COMPONENTS):
        raise RuntimeError("The supplied 3-D sheets cannot be mapped uniquely to Branch 1/2")
    component_map = dict(
        zip(
            mapping_table["posthoc_legacy_winding_label"].astype(str),
            mapping_table["component"].astype(str),
        )
    )
    anchors = pd.read_csv(anchor_path)
    evolution = pd.read_csv(evolution_path)
    anchors["component"] = anchors["branch"].map(component_map)
    evolution["component"] = evolution["branch"].map(component_map)
    if anchors["component"].isna().any() or evolution["component"].isna().any():
        raise RuntimeError("The supplied 3-D sheets do not match the discovered Branch 1/2 labels")
    anchors = anchors[anchors["method"].eq("Force-Equilibrium")].copy()
    return anchors, evolution


def _draw_target_web_3d(
    ax: plt.Axes,
    table: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    color: str,
    marker: str,
    line_width: float,
    line_alpha: float,
    marker_size: float,
) -> None:
    lookup = {
        (int(row.ix), int(row.iy)): np.asarray([row.mds1, row.mds2, row.mds3], dtype=float)
        for row in table.itertuples(index=False)
    }
    for edge in edge_table.itertuples(index=False):
        a = lookup[(int(edge.a_ix), int(edge.a_iy))]
        b = lookup[(int(edge.b_ix), int(edge.b_iy))]
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            color=color, lw=line_width, alpha=line_alpha,
            solid_capstyle="round", zorder=2,
        )
    values = table[["mds1", "mds2", "mds3"]].to_numpy(float)
    markers = ax.scatter(
        values[:, 0], values[:, 1], values[:, 2],
        s=marker_size, marker=marker, facecolors="white", edgecolors=color,
        linewidths=0.85, alpha=1.0, depthshade=False, zorder=4,
    )
    # Dense branch webs retain the original marker areas at manuscript scale.
    # The general white-panel readability floor obscures nearby endpoints.
    markers._manuscript_preserve_marker_area = True


def _style_branch_axis_3d(ax: plt.Axes) -> None:
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.08, 1.0, 0.94))
    ax.tick_params(labelsize=7.4, pad=-1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(mpl.ticker.LinearLocator(3))
        axis._axinfo["grid"].update(color=(0.58, 0.58, 0.58, 0.12), linewidth=0.45)
        axis.set_pane_color((0.992, 0.992, 0.992, 0.50))
    # q1/q2 are display-only coordinates; suppress their numeric tick labels
    # to prevent the oblique 3-D axes from stacking text at the lower corner.
    ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))


def figure_2_two_branch_evolution(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Two FE-defined branches and the full-dimensional Conventional-to-FE audit."""

    stem = "Figure_2_two_branch_evolution"
    if stem in ctx.figures:
        return ctx.figures[stem]
    checkpoints = (500, 10_000)
    branch = fixed_single_branch_pack(ctx)
    anchors_3d = branch["anchors"].copy()
    evolution_3d = branch["evolution"].copy()
    representative = branch["correlation"].copy()
    if representative.empty:
        raise RuntimeError("The representative branch target is unavailable")
    root_target_id = str(representative.iloc[0]["target_id"])
    correlation_rows: list[dict[str, Any]] = []
    for component, group in representative.groupby("component", sort=True):
        rho = float(group["objective"].rank().corr(group["similarity_to_matched_fe"].rank()))
        correlation_rows.append(
            {
                "scope": "representative root target",
                "target_id": root_target_id,
                "component": component,
                "n": len(group),
                "spearman_rho_objective_vs_similarity": rho,
            }
        )
    ctx.tables["fig2_full_dimensional_correlation"] = representative[
        ["trajectory_index", "target_id", "component", "objective", "distance_to_matched_fe", "similarity_to_matched_fe"]
    ].copy()
    ctx.tables["fig2_representative_target_correlation"] = representative[
        ["trajectory_index", "target_id", "component", "objective", "distance_to_matched_fe", "similarity_to_matched_fe"]
    ].copy()
    ctx.tables["fig2_full_dimensional_correlation_summary"] = pd.DataFrame(correlation_rows)
    distance_rows: list[dict[str, Any]] = []
    for (component, iteration), group in evolution_3d.groupby(
        ["component", "iteration"], sort=True
    ):
        if int(iteration) not in checkpoints:
            continue
        values = group["distance_to_matched_fe"].to_numpy(float)
        distance_rows.append(
            {
                "component": component,
                "iteration_budget": int(iteration),
                "n": len(values),
                "median_projector_distance": float(np.median(values)),
                "q25_projector_distance": float(np.quantile(values, 0.25)),
                "q75_projector_distance": float(np.quantile(values, 0.75)),
            }
        )
    ctx.tables["fig2_full_dimensional_distance_summary"] = pd.DataFrame(distance_rows)

    fig = plt.figure(figsize=(10.8, 11.2))
    grid = fig.add_gridspec(
        3, 2,
        height_ratios=(1.0, 1.0, 0.68),
        hspace=0.23,
        wspace=0.14,
    )
    for row, component in enumerate(COMPONENTS):
        component_anchors = anchors_3d[anchors_3d["component"].eq(component)]
        component_evolution = evolution_3d[
            evolution_3d["component"].eq(component)
            & evolution_3d["iteration"].isin(checkpoints)
        ]
        visible = pd.concat([component_anchors, component_evolution], ignore_index=True)
        limits: list[tuple[float, float]] = []
        for column in ("mds1", "mds2", "mds3"):
            low, high = float(visible[column].min()), float(visible[column].max())
            pad = 0.055 * max(high - low, 1.0e-12)
            limits.append((low - pad, high + pad))
        camera = (25, 165) if component == "B1" else (10, -165)
        for col, checkpoint in enumerate(checkpoints):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            conventional = component_evolution[
                component_evolution["iteration"].eq(int(checkpoint))
            ].copy().sort_values(["iy", "ix"])
            _draw_target_web_3d(
                ax, conventional, branch["bank"].edge_table,
                color=COLORS["Conventional"], marker="o", line_width=2.3,
                line_alpha=0.20, marker_size=27.0,
            )
            _draw_target_web_3d(
                ax, component_anchors, branch["bank"].edge_table,
                color=COLORS["FE"], marker="s", line_width=1.8,
                line_alpha=0.20, marker_size=20.0,
            )
            ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
            ax.view_init(*camera)
            _style_branch_axis_3d(ax)
            ax.set_xlabel(r"$q_1$", labelpad=-1, fontsize=9.0)
            ax.set_ylabel(r"$q_2$", labelpad=-1, fontsize=9.0)
            ax.set_zlabel(r"$q_3$", labelpad=-2, fontsize=9.0)
            if row == 0:
                ax.set_title(f"{checkpoint:,} iterations", pad=1.5, fontsize=10.5, fontweight="semibold")
            if col == 0:
                ax.text2D(
                    -0.25, 1.055, f"Branch {row + 1}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9.5, fontweight="semibold",
                    clip_on=False,
                )

    ax_corr = fig.add_subplot(grid[2, :])
    for component in COMPONENTS:
        group = representative[representative["component"].eq(component)].copy()
        color = COLORS[component]
        rho = next(
            row["spearman_rho_objective_vs_similarity"]
            for row in correlation_rows if row["component"] == component
        )
        ax_corr.scatter(
            group["similarity_to_matched_fe"], group["objective"],
            s=42, color=color, alpha=1.0, edgecolor="white", linewidth=0.7,
            marker="o" if component == "B1" else "D", zorder=4,
            label=rf"{component.replace('B', 'Branch ')} ($\rho={rho:.2f}$)", rasterized=True,
        )
        x_values = group["similarity_to_matched_fe"].to_numpy(float)
        y_values = group["objective"].to_numpy(float)
        if len(group) >= 2 and float(np.ptp(x_values)) > 0.0:
            fit = np.polyfit(x_values, y_values, 1)
            line_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
            ax_corr.plot(line_x, np.polyval(fit, line_x), color=color, lw=1.8,
                         ls="-" if component == "B1" else "--", zorder=2)
    ax_corr.set_xlabel(terse_math_label("similarity_fe"))
    ax_corr.set_ylabel(r"$J_{\mathrm{C}}$")
    ax_corr.margins(x=0.07, y=0.09)
    ax_corr.legend(frameon=False, loc="lower left")
    fig.legend(
        handles=[
            mpl.lines.Line2D(
                [0], [0], color=COLORS["Conventional"], marker="o", markerfacecolor="white",
                lw=2.5, label="Conventional",
            ),
            mpl.lines.Line2D(
                [0], [0], color=COLORS["FE"], marker="s", markerfacecolor="white",
                lw=1.6, label="FE",
            ),
        ],
        frameon=False, loc="lower center", bbox_to_anchor=(0.50, 0.018), ncol=2,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.935, bottom=0.120)
    outside_grid_panel_labels(
        fig,
        (grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1], grid[2, :]),
        tuple("abcde"),
    )
    return _save(ctx, fig, stem)


def _decision_pack(ctx: p.PipelineContext) -> dict[str, Any]:
    if "final_decision_pack" in ctx.memo:
        return ctx.memo["final_decision_pack"]
    objectives = {
        "Conventional": p._objective_at(ctx, "Conventional"),
        "FE": p._objective_at(ctx, "Force-Equilibrium"),
    }
    runs = {
        "Conventional": p._decision_endpoint_run(ctx, "Conventional"),
        "FE": p._decision_endpoint_run(ctx, "Force-Equilibrium"),
    }
    sections = {}
    for method in ("Conventional", "FE"):
        directions = p._native_local_directions(objectives[method], runs[method])
        sections[method] = p._native_relative_sections(
            objectives[method], runs[method], directions,
            line_points=_plot_points(ctx, "objective_line_points", 101),
            plane_points=_plot_points(ctx, "objective_plane_points", 51),
        )
    value = {"objectives": objectives, "runs": runs, "sections": sections}
    ctx.memo["final_decision_pack"] = value
    return value


def figure_3_local_decision_geometry(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Two-dimensional local sections first, followed by their one-dimensional cuts."""

    stem = "Figure_3_local_decision_geometry"
    if stem in ctx.figures:
        return ctx.figures[stem]
    pack = _decision_pack(ctx)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), gridspec_kw={"height_ratios": (1.05, 0.72)})
    summary: list[dict[str, Any]] = []
    for column, method in enumerate(("Conventional", "FE")):
        data = pack["sections"][method]
        color = COLORS[method]
        finite = data["plane"][np.isfinite(data["plane"])]
        vmax = max(float(np.quantile(finite, 0.985)), np.finfo(float).tiny)
        image = axes[0, column].imshow(
            np.clip(data["plane"], 0.0, vmax), origin="lower",
            extent=(data["q"][0], data["q"][-1], data["q"][0], data["q"][-1]),
            cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto", interpolation="bilinear",
        )
        axes[0, column].grid(False)
        target_marker(axes[0, column], 0.0, 0.0, s=48, linewidth=1.5)
        axes[0, column].set_xlabel(terse_math_label("descent_coordinate"))
        axes[0, column].set_ylabel(terse_math_label("transverse_coordinate"))
        set_panel_title(axes[0, column], method)
        add_adjacent_colorbar(
            image, axes[0, column], size="3.5%", pad="2.5%",
            label=r"$\Delta J$", labelsize=8.0, ticksize=7.0,
        )

        axes[1, column].plot(data["t"], data["line"], color=color, lw=2.2)
        center = len(data["line"]) // 2
        target_marker(axes[1, column], [0.0], [data["line"][center]], s=45)
        axes[1, column].set_xlabel(r"$t$ (rad)")
        axes[1, column].set_ylabel(r"$\Delta J$")
        axes[1, column].set_title("")
        summary.append(
            {
                "method": method,
                "phase_content_id": digest_array(pack["runs"][method].phase_rad),
                "terminal_gradient_norm": float(pack["runs"][method].gradient_norm),
                "range_rad": 0.30,
                "plane_points": len(data["q"]),
                "line_points": len(data["t"]),
                "direction_definition": "terminal negative gradient and orthogonalized last accepted step",
                "center_left_slope": float(
                    (data["line"][len(data["line"]) // 2] - data["line"][len(data["line"]) // 2 - 1])
                    / (data["t"][1] - data["t"][0])
                ),
                "center_right_slope": float(
                    (data["line"][len(data["line"]) // 2 + 1] - data["line"][len(data["line"]) // 2])
                    / (data["t"][1] - data["t"][0])
                ),
                "quadratic_fit_relative_residual": float(
                    np.linalg.norm(
                        data["line"] - np.polyval(np.polyfit(data["t"], data["line"], 2), data["t"])
                    ) / max(float(np.linalg.norm(data["line"] - np.mean(data["line"]))), 1.0e-30)
                ),
            }
        )
    ctx.tables["fig3_local_decision_sections"] = pd.DataFrame(summary)
    fig.subplots_adjust(left=0.10, right=0.965, top=0.90, bottom=0.135, hspace=0.36, wspace=0.46)
    outside_grid_panel_labels(
        fig,
        tuple(axis.get_subplotspec() for axis in axes.ravel()),
        tuple("abcd"),
    )
    return _save(ctx, fig, stem)

def _tri3_xy_pressure(ctx: p.PipelineContext, phase: np.ndarray, *, samples: int) -> tuple[np.ndarray, np.ndarray]:
    samples = _plot_points(ctx, "pressure_points", samples)
    axis = np.linspace(-0.026, 0.026, int(samples))
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.stack([xx, yy, np.full_like(xx, 0.030)], axis=-1)
    amplitude = np.abs(p._array_field(ctx, phase).pressure(points.reshape(-1, 3))).reshape(xx.shape)
    return axis, amplitude


def _native_multitrap_directions(
    problem: MultiTrapProblem,
    phase: np.ndarray,
    history: np.ndarray,
    config: MultitrapObjectiveConfig,
    *,
    force_epsilon: float,
    uniformity_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    evaluation = evaluate_multitrap_objective(
        problem, phase, config,
        force_epsilon=float(force_epsilon),
        uniformity_epsilon=float(uniformity_epsilon),
    )
    gradient = np.asarray(evaluation.gradient_full, dtype=float)
    u = -gradient / max(float(np.linalg.norm(gradient)), 1.0e-30)
    current = np.asarray(phase, dtype=float)
    v_direction = None
    for previous in np.asarray(history, dtype=float)[::-1]:
        gauge = np.angle(np.vdot(np.exp(1j * previous), np.exp(1j * current)))
        step = np.angle(np.exp(1j * current) * np.exp(-1j * gauge) * np.exp(-1j * previous))
        step -= u * float(np.dot(step, u))
        norm = float(np.linalg.norm(step))
        if norm > 1.0e-14:
            v_direction = step / norm
            break
    if v_direction is None:
        raise RuntimeError("Triple history does not contain a transverse accepted step")
    return u, v_direction


def _tri3_plane(
    ctx: p.PipelineContext,
    node: Mapping[str, Any],
    display: str,
    phase: np.ndarray,
    *,
    half_range_rad: float,
) -> dict[str, np.ndarray]:
    """Evaluate the current main endpoint on the archived, fixed native axes."""

    if display == "Conventional":
        result = node["Conventional"]
        config = node["conventional_config"]
        if "Conventional_history_phase_rad" in node:
            history = np.asarray(node["Conventional_history_phase_rad"], dtype=float)
        else:
            checkpoints = sorted(node["Conventional_checkpoint_phases"])
            history = np.stack(
                [
                    node["Conventional_checkpoint_phases"][checkpoint]
                    for checkpoint in checkpoints
                ]
            )
        force_epsilon = float(result.force_epsilon)
        uniformity_epsilon = float(result.uniformity_epsilon)
    elif display == "FE":
        result = node["FE"]
        config = node["fe_config"]
        history = np.asarray(result.history_phase_rad, dtype=float)
        force_epsilon = float(result.stages[-1].force_epsilon)
        uniformity_epsilon = float(result.stages[-1].uniformity_epsilon)
    else:
        raise KeyError(display)
    # Freeze the original accepted chart axes while evaluating the current
    # endpoint and objective. A terminal gradient near zero is a poor moving
    # coordinate reference; re-estimating it changed the visible basin.
    with np.load(ctx.data_root / "fig4_selected_triple_10k" / "central_native_directions.npz") as directions:
        prefix = "conventional" if display == "Conventional" else "fe"
        u = np.asarray(directions[f"{prefix}_u"], dtype=float)
        v = np.asarray(directions[f"{prefix}_v"], dtype=float)
    points = _plot_points(ctx, "objective_plane_points", 101 if ctx.config.full else 61)
    half_range = float(half_range_rad)
    if half_range <= 0.0:
        raise ValueError("half_range_rad must be positive")
    q = np.linspace(-half_range, half_range, points)
    payload = {
        "contract": "fig4-current-endpoint-frozen-original-axes-gradient-grid-v3",
        "problem": node["problem"].fingerprint,
        "config": config.to_payload(),
        "phase": digest_array(phase),
        "direction_u": digest_array(u),
        "direction_v": digest_array(v),
        "half_range_rad": half_range,
        "points": points,
        "force_epsilon": force_epsilon,
        "uniformity_epsilon": uniformity_epsilon,
    }

    def compute() -> dict[str, np.ndarray]:
        plane = np.empty((points, points), dtype=float)
        gradient_s1 = np.empty_like(plane)
        gradient_s2 = np.empty_like(plane)
        for j, b in enumerate(q):
            for i, a in enumerate(q):
                evaluation = evaluate_multitrap_objective(
                    node["problem"], phase + a * u + b * v, config,
                    force_epsilon=force_epsilon,
                    uniformity_epsilon=uniformity_epsilon,
                )
                plane[j, i] = evaluation.value
                gradient_s1[j, i] = np.dot(evaluation.gradient_full, u)
                gradient_s2[j, i] = np.dot(evaluation.gradient_full, v)
        return {
            "q": q,
            "plane": plane - np.nanmin(plane),
            "objective_raw": plane,
            "gradient_s1": gradient_s1,
            "gradient_s2": gradient_s2,
            "direction_u": u,
            "direction_v": v,
        }

    value, _, _ = ctx.cache.get_or_compute(
        "final_tri3_wide_section", payload, compute, recompute=ctx.recompute,
    )
    evaluation = evaluate_multitrap_objective(
        node["problem"], phase, config, force_epsilon=force_epsilon,
        uniformity_epsilon=uniformity_epsilon)
    gradient = np.asarray(evaluation.gradient_full, dtype=float)
    value = dict(value)
    value["projected_gradient"] = np.asarray([np.dot(gradient, u), np.dot(gradient, v)])
    value["full_gradient_norm"] = float(np.linalg.norm(gradient))
    value["optimizer_gradient_norm"] = float(np.linalg.norm(evaluation.gradient_reduced))
    value["optimizer_gradient_reduced"] = evaluation.gradient_reduced
    return value


def _fixed_triple_conventional_history(
    ctx: p.PipelineContext,
    bank: Mapping[str, Any],
    node: Mapping[str, Any],
) -> np.ndarray:
    """Archived 10k history recovery; not used by the current main figures."""

    terminal_phase = np.asarray(
        node["Conventional_checkpoint_phases"][10_000], dtype=float
    )
    initial_phase = np.asarray(bank["initial_phase"], dtype=float)
    config = node["conventional_config"]
    payload = {
        "contract": "fixed-triple-conventional-central-history-10000-v1",
        "problem": node["problem"].fingerprint,
        "config": config.to_payload(),
        "seed": int(bank["seed"]),
        "initial_phase": digest_array(initial_phase),
        "expected_terminal_phase": digest_array(terminal_phase),
    }

    def producer() -> Any:
        return run_multitrap(
            node["problem"],
            config,
            seed=int(bank["seed"]),
            initial_phases=initial_phase,
            record_history=True,
            metadata={
                "display_method": "Conventional",
                "fixed_iteration_cap": 10_000,
                "history_role": "Figure 4 transverse-axis recovery",
            },
        )

    result, _, _ = ctx.cache.get_or_compute(
        "fixed_triple_conventional_central_history_10000",
        payload,
        producer,
        recompute=ctx.recompute,
    )
    if digest_array(np.asarray(result.phases, dtype=float)) != digest_array(
        terminal_phase
    ):
        raise RuntimeError(
            "Recovered Conventional history does not terminate at the fixed 10k phase"
        )
    history = np.asarray(result.history_phase_rad, dtype=float)
    if history.ndim != 2 or len(history) < 2:
        raise RuntimeError("Recovered Conventional 10k history is incomplete")
    return history


def figure_4_triple_fields_and_decision(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Matched Triple fields and local sections at the current fixed main endpoint."""

    stem = "Figure_4_triple_fields_and_decision"
    if stem in ctx.figures:
        return ctx.figures[stem]
    bank = ensure_main_triple_endpoint(ctx)
    node = dict(bank["objects"][(1, 1)])
    terminal_checkpoint = int(bank["conventional_cap"])
    phases = {
        "Conventional": np.asarray(
            node["Conventional_checkpoint_phases"][terminal_checkpoint], dtype=float
        ),
        "FE": np.asarray(node["FE_phase"], dtype=float),
    }
    fields = {
        method: _tri3_xy_pressure(
            ctx, phase, samples=181 if ctx.memo.get("production_main") else 121 if ctx.config.full else 81
        )
        for method, phase in phases.items()
    }
    norm = shared_power_norm([fields["Conventional"][1], fields["FE"][1]], lower=0.0, upper=99.5)
    planes = {
        method: _tri3_plane(
            ctx,
            node,
            method,
            phases[method],
            half_range_rad=3.675,
        )
        for method in phases
    }
    local_rows = []
    for method, data in planes.items():
        section_path = ctx.output_root / "data" / f"fig4_{method.lower()}_objective_gradient_grid.npz"
        section_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(section_path, **data, phase_rad=phases[method])
        ctx.writer.record_file(section_path, "objective-gradient-section")
        aa, bb = np.meshgrid(data["q"], data["q"], indexing="xy")
        design = np.column_stack(
            [np.ones(aa.size), aa.ravel(), bb.ravel(), aa.ravel() ** 2, aa.ravel() * bb.ravel(), bb.ravel() ** 2]
        )
        values = data["plane"].ravel()
        fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
        center_index = len(data["q"]) // 2
        gradient_span = float(np.ptp(data["plane"][center_index, :]))
        transverse_span = float(np.ptp(data["plane"][:, center_index]))
        local_rows.append(
            {
                "method": method,
                "phase_content_id": digest_array(phases[method]),
                "quadratic_surface_relative_residual": float(
                    np.linalg.norm(values - fitted)
                    / max(float(np.linalg.norm(values - np.mean(values))), 1.0e-30)
                ),
                "transverse_to_gradient_dynamic_range": float(
                    transverse_span / max(gradient_span, np.finfo(float).tiny)
                ),
                "half_range_rad": float(max(abs(data["q"][0]), abs(data["q"][-1]))),
                "plane_points_per_axis": len(data["q"]),
                "endpoint_iteration": (
                    terminal_checkpoint
                    if method == "Conventional"
                    else int(node["FE"].iterations)
                ),
                "fixed_fe_alpha_force": (
                    float(TRIPLE_FE_ENDPOINT.alpha)
                    if method == "FE"
                    else np.nan
                ),
                "fixed_fe_curvature_wx": (
                    float(TRIPLE_FE_ENDPOINT.curvature_weight_override[0][0])
                    if method == "FE"
                    else np.nan
                ),
                "fixed_fe_curvature_wy": (
                    float(TRIPLE_FE_ENDPOINT.curvature_weight_override[1][1])
                    if method == "FE"
                    else np.nan
                ),
                "fixed_fe_curvature_wz": (
                    float(TRIPLE_FE_ENDPOINT.curvature_weight_override[2][2])
                    if method == "FE"
                    else np.nan
                ),
                "phase_source": (
                    f"fixed Triple {terminal_checkpoint:,}-iteration central endpoint"
                    if method == "Conventional"
                    else "fixed Triple FE endpoint"
                ),
                "coordinate_source": "archived native axes from accepted Triple chart, held fixed",
                "gradient_s1_n_per_m_rad": float(data["projected_gradient"][0]),
                "gradient_s2_n_per_m_rad": float(data["projected_gradient"][1]),
                "projected_gradient_norm_n_per_m_rad": float(np.linalg.norm(data["projected_gradient"])),
                "full_phase_gradient_norm_n_per_m_rad": float(data["full_gradient_norm"]),
                "optimizer_gradient_norm_n_per_m_rad": float(data["optimizer_gradient_norm"]),
                "optimizer_gradient_definition": "L2 norm of analytic gradient_full[1:], first phase fixed; passed to optimizer",
                "gradient_flow": "sparse local negative-gradient directions at cached grid nodes; equal-length glyphs, no trajectory integration",
            }
        )
    ctx.tables["fig4_triple_local_geometry_summary"] = pd.DataFrame(local_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0))
    targets = np.asarray(node["targets_m"], dtype=float)
    for column, method in enumerate(("Conventional", "FE")):
        axis, amplitude = fields[method]
        pressure_image = field_panel(
            axes[0, column], amplitude,
            (1e3 * axis[0], 1e3 * axis[-1], 1e3 * axis[0], 1e3 * axis[-1]),
            norm=norm, xlabel=r"$x$ (mm)", ylabel=r"$y$ (mm)",
        )
        import matplotlib.patheffects as pe
        target_markers = axes[0, column].scatter(1e3 * targets[:, 0], 1e3 * targets[:, 1],
            marker="x", color=TARGET_COLOR, s=62, lw=1.8, zorder=20)
        target_markers.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
        set_panel_title(axes[0, column], method)
        add_adjacent_colorbar(
            pressure_image, axes[0, column], size="3.5%", pad="2.5%",
            label_key="pressure", labelsize=8.0, ticksize=7.0,
        )
        data = planes[method]
        finite = data["plane"][np.isfinite(data["plane"])]
        vmax = max(float(np.quantile(finite, 0.985)), np.finfo(float).tiny)
        normalized_plane = np.clip(data["plane"] / vmax, 0.0, 1.0)
        image = axes[1, column].imshow(
            normalized_plane, origin="lower",
            extent=(data["q"][0], data["q"][-1], data["q"][0], data["q"][-1]),
            cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="bilinear",
        )
        axes[1, column].contour(
            data["q"], data["q"], normalized_plane,
            levels=np.linspace(0.15, 0.9, 6), colors="#151D29",
            linewidths=0.55, alpha=0.25,
        )
        ax = axes[1, column]
        # Show instantaneous directions, not paths integrated through a coarse
        # interpolation of the sharp valley. Apply identical sampling to C/FE.
        margin = min(5, max(0, (len(data["q"]) - 1) // 4))
        sample_indices = np.unique(np.rint(np.linspace(
            margin, len(data["q"]) - 1 - margin, min(7, len(data["q"]))
        )).astype(int))
        xx, yy = np.meshgrid(data["q"][sample_indices], data["q"][sample_indices])
        gx = -data["gradient_s1"][np.ix_(sample_indices, sample_indices)]
        gy = -data["gradient_s2"][np.ix_(sample_indices, sample_indices)]
        magnitude = np.hypot(gx, gy)
        valid = (magnitude > 1e-30) & (np.hypot(xx, yy) > 1e-12)
        glyph_length = 0.62  # rad, direction-only display length for both methods
        ax.quiver(xx[valid], yy[valid], glyph_length * gx[valid] / magnitude[valid],
            glyph_length * gy[valid] / magnitude[valid], angles="xy", scale_units="xy",
            scale=1, pivot="middle", color="#F8FAFC", edgecolor="#35404D",
            linewidth=0.20, width=0.0040, headwidth=3.2, headlength=4.0,
            headaxislength=3.6, zorder=8)
        marker = ax.scatter(0.0, 0.0, marker="x", color=TARGET_COLOR, s=80, linewidth=2.0, zorder=20)
        marker.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
        ax.set_aspect("equal")
        ax.set_xlim(data["q"][0], data["q"][-1])
        ax.set_ylim(data["q"][0], data["q"][-1])
        ax.grid(False)
        axes[1, column].set_xlabel(r"$s_1$ (rad)")
        axes[1, column].set_ylabel(r"$s_2$ (rad)")
        axes[1, column].set_title("")
        add_adjacent_colorbar(
            image, axes[1, column], size="3.5%", pad="2.5%",
            label=r"$\Delta J/Q_{98.5}$", labelsize=8.0, ticksize=7.0,
            ticks=(0.0, 0.5, 1.0),
        )
    fig.subplots_adjust(left=0.095, right=0.965, top=0.90, bottom=0.125, hspace=0.30, wspace=0.34)
    outside_grid_panel_labels(
        fig,
        tuple(axis.get_subplotspec() for axis in axes.ravel()),
        tuple("abcd"),
    )
    return _save(ctx, fig, stem)


def _field_concentration(ctx: p.PipelineContext, phase: np.ndarray, target: np.ndarray) -> dict[str, float]:
    half_width = 0.012
    roi_radius = 0.003
    samples = _plot_points(ctx, "pressure_points", 121 if ctx.config.full else 81)
    axis = np.linspace(-half_width, half_width, samples)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.stack([target[0] + xx, target[1] + yy, np.full_like(xx, target[2])], axis=-1)
    pressure = np.abs(p._array_field(ctx, phase).pressure(points.reshape(-1, 3))).reshape(xx.shape)
    energy = pressure**2
    roi = xx**2 + yy**2 <= roi_radius**2
    concentration = float(np.sum(energy[roi]) / max(float(np.sum(energy)), 1.0e-30))
    return {"field_concentration_fraction": concentration, "plane_half_width_m": half_width, "roi_radius_m": roi_radius}


def _single_ad_command(ctx: p.PipelineContext) -> Any:
    """Build the locked eight-control-point Diff-PAT Single command once."""

    memo_key = "single_diff_pat_command"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    payload = {
        "contract": "single-diff-pat-amplitude-only-v1",
        "target_m": p.MAIN_TARGET_M.tolist(),
        "positions": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "frequency_hz": float(ctx.config.frequency_hz),
        "control_spec": asdict(p.IB_VORTEX_SPEC),
        "diff_pat": asdict(DIFF_PAT_CONFIG),
        "random_seed": int(ctx.config.random_seed),
    }

    def producer() -> Any:
        return benchmark_synthesis_method_factory(
            diff_pat_phase_only,
            p._vortex_transfer_factory(
                ctx, p.MAIN_TARGET_M, spec=p.IB_VORTEX_SPEC
            ),
            target_m=p.MAIN_TARGET_M,
            method_kwargs={
                "task": "Single",
                "random_seed": int(ctx.config.random_seed),
            },
        )

    run, _, _ = ctx.cache.get_or_compute(
        "single_diff_pat_command", payload, producer, recompute=ctx.recompute
    )
    ctx.memo[memo_key] = run
    return run


def _single_fixed_endpoint_specs(ctx: p.PipelineContext) -> list[dict[str, Any]]:
    """Create paired C endpoints and fixed alpha=10, 10,000-cap FE endpoints."""

    root_ix, root_iy = ctx.branch_bank.root_target
    rows = ctx.artifact.trajectory_table[
        ctx.artifact.trajectory_table["method"].eq("Conventional")
        & ctx.artifact.trajectory_table["ix"].eq(root_ix)
        & ctx.artifact.trajectory_table["iy"].eq(root_iy)
    ].sort_values("seed")
    if rows["seed"].nunique() != 10:
        raise RuntimeError(
            f"Expected 10 central Conventional seeds, found {rows['seed'].nunique()}"
        )
    rows = rows if ctx.config.full else rows.iloc[:1]
    fe_method = fixed_single_fe_method_spec()
    condition = Condition(
        label="fixed-single-fe-alpha10",
        positions=ctx.positions_m,
        target_m=tuple(float(value) for value in p.MAIN_TARGET_M),
        frequency_hz=ctx.config.frequency_hz,
        curvature_weight=fe_method.curvature_weight,
        normals=ctx.normals,
    )
    objective = SingleTargetObjective(condition, fe_method)
    upward_target_n = effective_weight_force_target_n(SphereInFluid(frequency_hz=ctx.config.frequency_hz))
    compensated_objective = SingleTargetObjective(condition, fe_method, force_target_n=upward_target_n)
    specs: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        seed = int(row.seed)
        trajectory_index = int(row.trajectory_index)
        initial = np.random.default_rng(seed).uniform(
            -np.pi, np.pi, objective.n_transducers
        )
        payload = {
            "contract": "fixed-single-fe-endpoint-v2-uniform-10000-cap",
            "alpha_per_m": float(SINGLE_FE_ENDPOINT.alpha),
            "iteration_cap": int(SINGLE_FE_ENDPOINT.maxiter),
            "target_m": p.MAIN_TARGET_M.tolist(),
            "seed": seed,
            "initial_phase": digest_array(initial),
            "gtol": float(ctx.config.gtol),
            "positions": digest_array(ctx.positions_m),
            "normals": digest_array(ctx.normals),
        }
        fe_result, _, _ = ctx.cache.get_or_compute(
            "fixed_single_fe_endpoint",
            payload,
            lambda initial=initial: solve_corrected_gorkov_fe(
                objective,
                initial,
                maxiter=int(SINGLE_FE_ENDPOINT.maxiter),
                gtol=ctx.config.gtol,
            ),
            recompute=ctx.recompute,
        )
        pair_id = f"seed-{seed}"
        specs.extend(
            [
                {
                    "method": "Conventional",
                    "pair_id": pair_id,
                    "seed": seed,
                    "phase": np.asarray(
                        ctx.artifact.trajectory_snapshots[trajectory_index, -1],
                        dtype=float,
                    ),
                    "target_m": p.MAIN_TARGET_M.copy(),
                    "phase_source": "deposited Conventional final-budget state",
                    "terminal_iteration": int(row.final_budget_state_iteration),
                    "source_index": trajectory_index,
                    "command_time_s": float(row.runtime_s),
                },
                {
                    "method": "FE",
                    "pair_id": pair_id,
                    "seed": seed,
                    "phase": np.asarray(fe_result.phase_rad, dtype=float),
                    "target_m": p.MAIN_TARGET_M.copy(),
                    "phase_source": (
                        "fixed Single FE endpoint; alpha=10; maxiter=10000"
                    ),
                    "terminal_iteration": int(fe_result.iterations),
                    "source_index": -1,
                    "command_time_s": float(fe_result.history_wall_s[-1]),
                },
            ]
        )
        compensated_payload = dict(payload, contract="fixed-single-fe-gravity-endpoint-v1",
            force_target_n=upward_target_n.tolist())
        compensated_result, _, _ = ctx.cache.get_or_compute(
            "fixed_single_fe_gravity_endpoint", compensated_payload,
            lambda initial=initial: solve_corrected_gorkov_fe(compensated_objective, initial,
                maxiter=int(SINGLE_FE_ENDPOINT.maxiter), gtol=ctx.config.gtol),
            recompute=ctx.recompute)
        specs.append(dict(method="FE+g", pair_id=pair_id, seed=seed,
            phase=np.asarray(compensated_result.phase_rad), target_m=p.MAIN_TARGET_M.copy(),
            phase_source="Single FE with effective-weight residual only; same seed, start, alpha=10 and 10000 cap",
            terminal_iteration=int(compensated_result.iterations), source_index=-1,
            command_time_s=float(compensated_result.history_wall_s[-1])))
    baselines = _single_matched_baseline_commands(ctx)
    for method, run in (("IB", baselines["ib"]), ("GS", baselines["audit"])):
        specs.append(
            {
                "method": method,
                "pair_id": "",
                "seed": math.nan,
                "phase": np.asarray(run.phase_rad, dtype=float),
                "target_m": p.MAIN_TARGET_M.copy(),
                "phase_source": "matched eight-point, R=0.45 lambda prescription",
                "terminal_iteration": int(run.solver_result.iterations),
                "source_index": -1,
                "command_run": run,
            }
        )
    ad_run = _single_ad_command(ctx)
    specs.append(
        {
            "method": "AD",
            "pair_id": "",
            "seed": int(ctx.config.random_seed),
            "phase": np.asarray(ad_run.phase_rad, dtype=float),
            "target_m": p.MAIN_TARGET_M.copy(),
            "phase_source": (
                "Acoustic hologram optimisation using automatic "
                "differentiation; eight pressure-amplitude controls"
            ),
            "terminal_iteration": int(ad_run.solver_result.iterations),
            "source_index": -1,
            "command_run": ad_run,
        }
    )
    return specs


def _single_matched_baseline_commands(ctx: p.PipelineContext) -> dict[str, Any]:
    """Build only the two matched Single prescriptions used in the main figures."""

    memo_key = "final_single_matched_baseline_commands"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    target = p.MAIN_TARGET_M.copy()
    ib_factory = p._vortex_transfer_factory(ctx, target, spec=p.IB_VORTEX_SPEC)
    gs_factory = p._vortex_transfer_factory(
        ctx, target, spec=p.SINGLE_SIDED_GS_VORTEX_SPEC
    )
    payload = {
        "contract": "final-single-matched-baselines-only-v1",
        "array": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "target_m": target.tolist(),
        "frequency_hz": float(ctx.config.frequency_hz),
        "ib_spec": asdict(p.IB_VORTEX_SPEC),
        "gs_spec": asdict(p.SINGLE_SIDED_GS_VORTEX_SPEC),
        "ib_maxiter": 200,
        "ib_tolerance_rad": 0.01,
        "gs_iterations": 100,
    }

    def producer() -> dict[str, Any]:
        return {
            "ib": benchmark_synthesis_method_factory(
                iterative_back_projection,
                ib_factory,
                target_m=target,
                method_kwargs={"maxiter": 200, "tolerance_rad": 0.01},
            ),
            "audit": benchmark_synthesis_method_factory(
                phase_only_signature_projection_audit,
                gs_factory,
                target_m=target,
                method_kwargs={"iterations": 100},
            ),
        }

    commands, _, _ = ctx.cache.get_or_compute(
        "final_single_matched_baseline_commands",
        payload,
        producer,
        recompute=ctx.recompute,
    )
    ctx.memo[memo_key] = commands
    return commands


def _single_exact_benchmark(ctx: p.PipelineContext) -> dict[str, Any]:
    """Validate the fixed Single endpoint set with the common exact model."""

    memo_key = "final_single_fixed_exact_benchmark"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    specs = _single_fixed_endpoint_specs(ctx)

    def evaluate(spec: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        phase = np.asarray(spec["phase"], dtype=float)
        target = np.asarray(spec["target_m"], dtype=float)
        result = p._finite_ka_validation(
            ctx,
            phase,
            target,
            key=(
                f"fig5-single-fixed-{spec['method']}-"
                f"{spec['pair_id'] or digest_array(phase)[:12]}"
            ),
        )
        row = p._static_validation_row(str(spec["method"]), result)
        viscous_result = p._viscous_elastic_validation(
            ctx,
            phase,
            target,
            key=(
                f"fig5-single-viscous-{spec['method']}-"
                f"{spec['pair_id'] or digest_array(phase)[:12]}"
            ),
        )
        row.update(
            p._viscous_static_validation_row(
                str(spec["method"]), viscous_result
            )
        )
        field = p._array_field(ctx, phase)
        equilibrium = np.asarray(result.equilibrium.equilibrium_m, dtype=float)
        row["pressure_abs_at_intended_target_pa"] = float(
            np.abs(field.pressure(target[None, :]))[0]
        )
        row["pressure_abs_at_exact_equilibrium_pa"] = (
            float(np.abs(field.pressure(equilibrium[None, :]))[0])
            if bool(result.equilibrium.numerical_root_found)
            else math.nan
        )
        row.update(
            {
                "pair_id": str(spec["pair_id"]),
                "seed": spec["seed"],
                "phase_content_id": digest_array(phase),
                "phase_source": str(spec["phase_source"]),
                "terminal_iteration": int(spec["terminal_iteration"]),
                "source_index": int(spec["source_index"]),
                "command_time_s": _spec_measured_command_time(spec),
                "run_mode": ctx.config.mode,
                "manuscript_numerics": bool(ctx.config.full),
            }
        )
        return row, result

    workers = max(
        1,
        min(len(specs), int(os.environ.get("HAT_EXACT_WORKERS", "1"))),
    )
    ordered: list[tuple[dict[str, Any], Any] | None] = [None] * len(specs)
    if workers == 1:
        for index, spec in enumerate(specs):
            ordered[index] = evaluate(spec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate, spec): index
                for index, spec in enumerate(specs)
            }
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
    completed = [item for item in ordered if item is not None]
    raw = pd.DataFrame([item[0] for item in completed])
    p._register_viscous_elastic_model_tables(ctx)
    viscous_columns = [
        column for column in raw.columns if column.startswith("viscous_")
    ]
    viscous_single = raw[["method", "phase_content_id", *viscous_columns]].copy()
    viscous_single.insert(1, "task", "Single")
    viscous_single.insert(2, "target_id", "Single")
    ctx.tables["viscous_elastic_single_endpoint_sensitivity"] = viscous_single
    value = {
        "raw": raw,
        "specs": specs,
        "results": [item[1] for item in completed],
        "exact_worker_count": workers,
    }
    ctx.tables["finite_ka_equilibrium_benchmark_raw"] = raw
    ctx.tables["single_fixed_endpoint_contract"] = pd.DataFrame(
        [
            {
                "task": "Single",
                "fe_alpha_per_m": float(SINGLE_FE_ENDPOINT.alpha),
                "fe_iteration_cap": int(SINGLE_FE_ENDPOINT.maxiter),
                "correct_sign_gorkov": True,
                "exact_worker_count": workers,
            }
        ]
    )
    ctx.memo[memo_key] = value
    return value


def _exact_benchmark_with_concentration(ctx: p.PipelineContext) -> dict[str, Any]:
    if "final_exact_force_concentration" in ctx.memo:
        return ctx.memo["final_exact_force_concentration"]
    benchmark = _single_exact_benchmark(ctx)
    rows = benchmark["raw"].copy().reset_index(drop=True)
    specs = benchmark["specs"]
    required_methods = {"Conventional", "FE", "FE+g", "IB", "GS", "AD"}
    missing = required_methods.difference(rows["method"].astype(str))
    if missing:
        raise RuntimeError(
            "Single exact benchmark is missing locked methods: "
            f"{sorted(missing)}"
        )
    concentrations = [
        _field_concentration(ctx, np.asarray(spec["phase"], dtype=float), np.asarray(spec["target_m"], dtype=float))
        for spec in specs
    ]
    for key in concentrations[0]:
        rows[key] = [item[key] for item in concentrations]
    rows["display_method"] = rows["method"]
    rows["task"] = "Single"
    rows["target_index"] = 0
    rows["target_id"] = "Single"
    rows["method_family"] = np.select(
        [
            rows["method"].isin(["IB", "GS"]),
            rows["method"].eq("AD"),
        ],
        [
            "prescribed phase-only",
            "automatic-differentiation holography",
        ],
        default="optimization",
    )
    rows["preset_role"] = np.select(
        [rows["method"].isin(["IB", "GS"]), rows["method"].eq("AD")],
        ["matched", "matched-amplitude"],
        default="not_applicable",
    )
    rows["preset_radius_lambda"] = np.where(
        rows["method"].isin(["IB", "GS", "AD"]), 0.45, np.nan
    )
    rows["command_time_s"] = [
        _spec_measured_command_time(spec) for spec in specs
    ]
    value = dict(benchmark) | {"concentration_table": rows}
    ctx.memo["final_exact_force_concentration"] = value
    return value


def _spec_measured_command_time(spec: Mapping[str, Any]) -> float:
    """Read a measured command time retained on an endpoint specification."""

    for key in (
        "command_run", "benchmark_run", "synthesis_run", "ad_run", "run", "command"
    ):
        value = spec.get(key)
        timing = getattr(value, "timing", None)
        if timing is not None:
            elapsed = float(getattr(timing, "total_s"))
            if np.isfinite(elapsed) and elapsed >= 0.0:
                return elapsed
    for key in (
        "command_time_s", "measured_command_time_s", "runtime_s", "wall_time_s"
    ):
        if key in spec:
            elapsed = float(spec[key])
            if np.isfinite(elapsed) and elapsed >= 0.0:
                return elapsed
    return math.nan


def _triple_ad_command(
    ctx: p.PipelineContext,
    targets_m: np.ndarray,
) -> Any:
    """Build the locked 24-control-point Diff-PAT Triple command once."""

    targets = np.asarray(targets_m, dtype=float)
    memo_key = f"triple_diff_pat:{digest_array(targets)}"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    payload = {
        "contract": "triple-diff-pat-amplitude-only-v2-main-30000-timing",
        "timing_campaign": "main-triple-30000-matched-session-v1",
        "targets": digest_array(targets),
        "target_coordinates_m": targets.tolist(),
        "positions": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "frequency_hz": float(ctx.config.frequency_hz),
        "control_spec": asdict(p.IB_VORTEX_SPEC),
        "diff_pat": asdict(DIFF_PAT_CONFIG),
        "random_seed": int(ctx.config.random_seed),
    }

    def producer() -> Any:
        return benchmark_synthesis_method_factory(
            diff_pat_phase_only,
            p._multivortex_transfer_factory(
                ctx, targets, spec=p.IB_VORTEX_SPEC
            ),
            target_m=np.mean(targets, axis=0),
            method_kwargs={
                "task": "Triple",
                "random_seed": int(ctx.config.random_seed),
            },
        )

    run, _, _ = ctx.cache.get_or_compute(
        "triple_diff_pat_command", payload, producer, recompute=ctx.recompute
    )
    ctx.memo[memo_key] = run
    return run


def _triple_prescribed_commands(
    ctx: p.PipelineContext,
    targets_m: np.ndarray,
) -> dict[str, Any]:
    """Build the four frozen Triple prescription commands.

    Each eight-point ring retains its own free complex phase.  This is the
    direct multi-target extension of the same matched or offset prescription;
    no ring gauge is optimized after synthesis.
    """

    targets = np.asarray(targets_m, dtype=float)
    digest = digest_array(targets)
    memo_key = f"final_triple_matched_baseline_commands:{digest}"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    factory = p._multivortex_transfer_factory(
        ctx, targets, spec=p.IB_VORTEX_SPEC
    )
    representative_target = np.mean(targets, axis=0)
    payload = {
        "contract": "final-triple-matched-baselines-only-v2-main-30000-timing",
        "timing_campaign": "main-triple-30000-matched-session-v1",
        "array": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "targets": digest,
        "target_coordinates_m": targets.tolist(),
        "frequency_hz": float(ctx.config.frequency_hz),
        "control_spec": asdict(p.IB_VORTEX_SPEC),
        "control_group_size": 8,
        "ib_maxiter": 200,
        "ib_tolerance_rad": 0.01,
        "gs_iterations": 100,
    }

    def producer() -> dict[str, Any]:
        return {
            "IB_matched": benchmark_synthesis_method_factory(
                iterative_back_projection,
                factory,
                target_m=representative_target,
                method_kwargs={
                    "maxiter": 200,
                    "tolerance_rad": 0.01,
                    "control_group_size": 8,
                },
            ),
            "GS_matched": benchmark_synthesis_method_factory(
                phase_only_signature_projection_audit,
                factory,
                target_m=representative_target,
                method_kwargs={"iterations": 100, "control_group_size": 8},
            ),
        }

    commands, _, _ = ctx.cache.get_or_compute(
        "final_triple_matched_baseline_commands",
        payload,
        producer,
        recompute=ctx.recompute,
    )
    ctx.memo[memo_key] = commands
    return commands


def _triple_exact_benchmark_with_concentration(ctx: p.PipelineContext) -> dict[str, Any]:
    """Apply the unchanged exact validator independently at each Triple target."""

    if "final_triple_exact_force_concentration" in ctx.memo:
        return ctx.memo["final_triple_exact_force_concentration"]
    bank = ensure_main_triple_endpoint(ctx)
    node = bank["objects"][(1, 1)]
    targets = np.asarray(node["targets_m"], dtype=float)
    prescribed = _triple_prescribed_commands(ctx, targets)
    ad_run = _triple_ad_command(ctx, targets)
    terminal_checkpoint = int(bank["conventional_cap"])
    method_specs = (
        {
            "method": "Conventional",
            "phase": np.asarray(
                node["Conventional_checkpoint_phases"][terminal_checkpoint],
                dtype=float,
            ),
            "method_family": "optimization",
            "preset_role": "not_applicable",
            "preset_radius_lambda": math.nan,
            "phase_source": f"fixed central Triple {terminal_checkpoint:,}-iteration Conventional endpoint",
            "terminal_iteration": terminal_checkpoint,
            "command_time_s": float(node["Conventional"].solve_sec),
        },
        {
            "method": "FE",
            "phase": np.asarray(node["FE_phase"], dtype=float),
            "method_family": "optimization",
            "preset_role": "not_applicable",
            "preset_radius_lambda": math.nan,
            "phase_source": (
                "fixed Triple FE endpoint; alpha_force=3000; "
                f"curvature=diag(1,1,10); maxiter={sum(node['fe_config'].smooth_stage_maxiters)}"
            ),
            "terminal_iteration": int(node["FE"].iterations),
            "command_time_s": float(node["FE"].end_to_end_sec),
        },
        *(
            {
                "method": method,
                "phase": np.asarray(prescribed[f"{method}_matched"].phase_rad, dtype=float),
                "method_family": "prescribed phase-only",
                "preset_role": "matched",
                "preset_radius_lambda": 0.45,
                "phase_source": (
                    "three eight-point vortex rings; R=0.45 lambda; "
                    "one native free phase per ring"
                ),
                "terminal_iteration": int(
                    prescribed[f"{method}_matched"].solver_result.iterations
                ),
                "command_run": prescribed[f"{method}_matched"],
            }
            for method in ("IB", "GS")
        ),
        {
            "method": "AD",
            "phase": np.asarray(ad_run.phase_rad, dtype=float),
            "method_family": "automatic-differentiation holography",
            "preset_role": "matched-amplitude",
            "preset_radius_lambda": 0.45,
            "phase_source": (
                "Acoustic hologram optimisation using automatic "
                "differentiation; 24 pressure-amplitude controls"
            ),
            "terminal_iteration": int(ad_run.solver_result.iterations),
            "command_run": ad_run,
        },
    )
    jobs = [
        (spec, target_index, target)
        for spec in method_specs
        for target_index, target in enumerate(targets)
    ]

    def evaluate(
        job: tuple[Mapping[str, Any], int, np.ndarray]
    ) -> tuple[dict[str, Any], Any, pd.DataFrame | None, dict[str, Any]]:
        spec, target_index, target = job
        method = str(spec["method"])
        phase = np.asarray(spec["phase"], dtype=float)
        pair_id = f"Triple-T{target_index + 1}"
        validation_key = (
            f"fig5-triple-{method}-{spec['preset_role']}-"
            f"target-{target_index + 1}"
        )
        result = p._finite_ka_validation(
            ctx,
            phase,
            target,
            key=validation_key,
        )
        row = p._static_validation_row(method, result)
        viscous_result = p._viscous_elastic_validation(
            ctx,
            phase,
            target,
            key=f"{validation_key}-viscous",
        )
        row.update(p._viscous_static_validation_row(method, viscous_result))
        field = p._array_field(ctx, phase)
        equilibrium = np.asarray(result.equilibrium.equilibrium_m, dtype=float)
        row["pressure_abs_at_intended_target_pa"] = float(
            np.abs(field.pressure(target[None, :]))[0]
        )
        row["pressure_abs_at_exact_equilibrium_pa"] = (
            float(np.abs(field.pressure(equilibrium[None, :]))[0])
            if bool(result.equilibrium.numerical_root_found)
            else math.nan
        )
        audit: pd.DataFrame | None = None
        if method in {"Conventional", "FE"}:
            audit_spec = {"method": method, "pair_id": pair_id, "phase": phase}
            audit = p._finite_ka_jacobian_step_audit(ctx, audit_spec, result)
            evaluated = (
                audit[audit["status"].eq("evaluated")]
                if "status" in audit
                else audit.iloc[0:0]
            )
            values = (
                evaluated["stiffness_min_n_m"].to_numpy(float)
                if len(evaluated)
                else np.asarray([], dtype=float)
            )
            if bool(row["finite_ka_root_found"]):
                if len(values) != 3 or not np.all(np.isfinite(values)):
                    root_class = "resolved-stiffness-indeterminate"
                elif np.all(values > 0.0):
                    root_class = "resolved-restoring"
                elif np.all(values < 0.0):
                    root_class = "resolved-nonrestoring"
                else:
                    root_class = "resolved-stiffness-indeterminate"
                row["finite_ka_root_class"] = root_class
                row["finite_ka_locally_restoring"] = root_class == "resolved-restoring"
                row["finite_ka_stiffness_step_robust"] = root_class in {
                    "resolved-restoring", "resolved-nonrestoring",
                }
                row["finite_ka_stiffness_min_over_steps_n_m"] = (
                    float(np.nanmin(values)) if len(values) else math.nan
                )
                row["finite_ka_stiffness_max_over_steps_n_m"] = (
                    float(np.nanmax(values)) if len(values) else math.nan
                )
            else:
                row["finite_ka_stiffness_step_robust"] = False
                row["finite_ka_stiffness_min_over_steps_n_m"] = math.nan
                row["finite_ka_stiffness_max_over_steps_n_m"] = math.nan
            row["jacobian_step_audit_performed"] = True
        else:
            row["finite_ka_stiffness_step_robust"] = math.nan
            row["finite_ka_stiffness_min_over_steps_n_m"] = math.nan
            row["finite_ka_stiffness_max_over_steps_n_m"] = math.nan
            row["jacobian_step_audit_performed"] = False
        row.update(_field_concentration(ctx, phase, target))
        row.update(
            {
                "display_method": method,
                "task": "Triple",
                "target_index": int(target_index),
                "target_id": f"T{target_index + 1}",
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "phase_content_id": digest_array(phase),
                "phase_source": str(spec["phase_source"]),
                "terminal_iteration": int(spec["terminal_iteration"]),
                "method_family": str(spec["method_family"]),
                "preset_role": str(spec["preset_role"]),
                "preset_radius_lambda": float(spec["preset_radius_lambda"]),
                "command_time_s": _spec_measured_command_time(spec),
                "fixed_fe_alpha_force": (
                    float(TRIPLE_FE_ENDPOINT.alpha)
                    if method == "FE"
                    else math.nan
                ),
                "fixed_fe_curvature_wx": (
                    float(TRIPLE_FE_ENDPOINT.curvature_weight_override[0][0])
                    if method == "FE"
                    else math.nan
                ),
                "fixed_fe_curvature_wy": (
                    float(TRIPLE_FE_ENDPOINT.curvature_weight_override[1][1])
                    if method == "FE"
                    else math.nan
                ),
                "fixed_fe_curvature_wz": (
                    float(TRIPLE_FE_ENDPOINT.curvature_weight_override[2][2])
                    if method == "FE"
                    else math.nan
                ),
                "run_mode": ctx.config.mode,
                "manuscript_numerics": bool(ctx.config.full),
                "validation_scope": "one target at a time in the shared field; no particle-particle multiple scattering",
            }
        )
        if audit is not None:
            audit = audit.copy()
            audit["task"] = "Triple"
            audit["target_index"] = int(target_index)
            audit["target_id"] = f"T{target_index + 1}"
        return row, result, audit, dict(spec)

    workers = max(
        1,
        min(len(jobs), int(os.environ.get("HAT_EXACT_WORKERS", "1"))),
    )
    ordered: list[
        tuple[dict[str, Any], Any, pd.DataFrame | None, dict[str, Any]] | None
    ] = [None] * len(jobs)
    if workers == 1:
        for index, job in enumerate(jobs):
            ordered[index] = evaluate(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(evaluate, job): index for index, job in enumerate(jobs)
            }
            for future in as_completed(future_to_index):
                ordered[future_to_index[future]] = future.result()
    completed = [item for item in ordered if item is not None]
    table = pd.DataFrame([item[0] for item in completed]).reset_index(drop=True)
    p._register_viscous_elastic_model_tables(ctx)
    viscous_columns = [
        column for column in table.columns if column.startswith("viscous_")
    ]
    ctx.tables["viscous_elastic_triple_endpoint_sensitivity"] = table[
        [
            "method",
            "task",
            "target_index",
            "target_id",
            "target_x_m",
            "target_y_m",
            "target_z_m",
            "phase_content_id",
            *viscous_columns,
        ]
    ].copy()
    audit_parts = [item[2] for item in completed if item[2] is not None]
    audits = (
        pd.concat(audit_parts, ignore_index=True)
        if audit_parts
        else pd.DataFrame()
    )
    ctx.tables["fig5_triple_exact_jacobian_step_audit"] = audits
    ctx.tables["fig5_triple_command_timing"] = pd.DataFrame(
        [
            {
                "method": str(spec["method"]),
                "command_time_s": _spec_measured_command_time(spec),
                "terminal_iteration": int(spec["terminal_iteration"]),
                "phase_source": str(spec["phase_source"]),
            }
            for spec in method_specs
        ]
    )
    evaluations = [
        {"row": item[0], "result": item[1], "spec": item[3]}
        for item in completed
    ]
    value = {
        "table": table,
        "all_table": table,
        "results": [item[1] for item in completed],
        "evaluations": evaluations,
        "method_specs": method_specs,
        "node": node,
        "targets_m": targets,
        "prescribed_commands": prescribed,
        "ad_command": ad_run,
        "long_bank": bank,
    }
    ctx.memo["final_triple_exact_force_concentration"] = value
    compensation = _figure6_vertical_compensated_fe(ctx, value)
    for item in compensation["evaluations"]:
        item["row"].update(_field_concentration(ctx, item["spec"]["phase"], item["result"].equilibrium.target_m))
        item["row"]["validation_scope"] = "one target at a time in the shared field; no particle-particle multiple scattering"
        item["spec"].update(command_time_s=float(compensation["run"].end_to_end_sec),
            terminal_iteration=int(compensation["run"].iterations))
    extra = pd.DataFrame([item["row"] for item in compensation["evaluations"]])
    value["table"] = value["all_table"] = pd.concat([table, extra], ignore_index=True, sort=False)
    value["evaluations"] += compensation["evaluations"]
    value["results"] += [item["result"] for item in compensation["evaluations"]]
    value["method_specs"] = (*method_specs, dict(compensation["evaluations"][0]["spec"]))
    ctx.tables["fig5_triple_command_timing"] = pd.DataFrame([
        dict(method=spec["method"], command_time_s=_spec_measured_command_time(spec),
             terminal_iteration=spec["terminal_iteration"], phase_source=spec["phase_source"])
        for spec in value["method_specs"]])
    return value


FIG5_METHOD_ORDER = ("Conventional", "FE", "FE+g", "IB", "GS", "AD")
FIG5_MARKERS = {
    "FE+g": "P",
    "Conventional": "o",
    "FE": "s",
    "IB": "^",
    "GS": "v",
    "AD": "D",
}
FIG5_MARKER_SIZES = {
    "FE+g": 135,
    "Conventional": 92,
    "FE": 92,
    "IB": 160,
    "GS": 135,
    "AD": 68,
}
FIG5_HOLLOW_METHODS = {"IB", "GS"}


def _aggregate_force_concentration(
    table: pd.DataFrame,
    *,
    task: str,
) -> pd.DataFrame:
    """Reduce each locked method to one readable marker for one task."""

    rows: list[dict[str, Any]] = []
    for method in FIG5_METHOD_ORDER:
        group = table[table["method"].eq(method)].copy()
        if group.empty:
            raise RuntimeError(f"{task} exact benchmark is missing {method}")
        resolved = group[group["finite_ka_root_found"].astype(bool)].copy()
        finite_stiffness = resolved[
            np.isfinite(resolved["finite_ka_stiffness_min_n_m"].to_numpy(float))
        ]
        rows.append(
            {
                "task": task,
                "method": method,
                "display_method": method,
                "field_concentration_fraction": float(
                    group["field_concentration_fraction"].median()
                ),
                "finite_ka_displacement_a": (
                    float(resolved["finite_ka_displacement_a"].median())
                    if len(resolved)
                    else math.nan
                ),
                "finite_ka_stiffness_min_n_m": (
                    float(finite_stiffness["finite_ka_stiffness_min_n_m"].median())
                    if len(finite_stiffness)
                    else math.nan
                ),
                "resolved_count": int(len(resolved)),
                "endpoint_count": int(len(group)),
                "root_found": bool(len(resolved)),
                "aggregation": "median within the displayed task and method",
            }
        )
    return pd.DataFrame(rows)


def _independent_force_concentration_limits(
    table: pd.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    x = table["field_concentration_fraction"].to_numpy(float)
    x_span = max(float(np.ptp(x)), 0.01)
    xlim = (float(np.min(x) - 0.10 * x_span), float(np.max(x) + 0.10 * x_span))
    y = 1.0e3 * table["finite_ka_stiffness_min_n_m"].to_numpy(float)
    finite_y = y[np.isfinite(y)]
    if len(finite_y):
        low = min(0.0, float(np.min(finite_y)))
        high = max(0.0, float(np.max(finite_y)))
        span = max(high - low, 0.20 * max(abs(low), abs(high), 1.0e-6))
    else:
        low, high, span = 0.0, 1.0, 1.0
    unresolved_y = low - 0.13 * span
    ylim = (unresolved_y - 0.07 * span, high + 0.14 * span)
    return xlim, ylim, unresolved_y


def _plot_exact_force_concentration_panel(
    ax: mpl.axes.Axes,
    table: pd.DataFrame,
    *,
    title: str,
) -> None:
    xlim, ylim, unresolved_y = _independent_force_concentration_limits(table)
    for row in table.itertuples(index=False):
        resolved = bool(row.root_found) and np.isfinite(
            float(row.finite_ka_stiffness_min_n_m)
        )
        color = COLORS[str(row.method)]
        hollow = str(row.method) in FIG5_HOLLOW_METHODS
        ax.scatter(
            float(row.field_concentration_fraction),
            1.0e3 * float(row.finite_ka_stiffness_min_n_m)
            if resolved
            else unresolved_y,
            marker=FIG5_MARKERS[str(row.method)],
            s=FIG5_MARKER_SIZES[str(row.method)],
            facecolor=color if resolved and not hollow else "none",
            edgecolor=color if hollow or not resolved else "white",
            linewidth=1.9 if hollow or not resolved else 0.9,
            zorder=5 if str(row.method) == "AD" else 4,
        )
    if title == "Single":
        fe_row = table[table.method.eq("FE")].iloc[0]
        ax.annotate("FE, FE+g", (float(fe_row.field_concentration_fraction),
                    1e3 * float(fe_row.finite_ka_stiffness_min_n_m)),
                    xytext=(-7, -13), textcoords="offset points", ha="right", va="top",
                    fontsize=9, color="0.3", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_box_aspect(1)
    compact_force_concentration_labels(ax)
    set_panel_title(ax, title)


def figure_5_single_triple_exact_force_concentration(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Matched exact force–field concentration analysis for Single and Triple."""

    stem = "Figure_5_single_triple_exact_force_concentration"
    if stem in ctx.figures:
        return ctx.figures[stem]
    single = _exact_benchmark_with_concentration(ctx)["concentration_table"].copy()
    triple = _triple_exact_benchmark_with_concentration(ctx)["table"].copy()
    combined = pd.concat([single, triple], ignore_index=True, sort=False)
    ctx.tables["fig5_exact_force_field_concentration"] = combined
    aggregates = pd.concat(
        [
            _aggregate_force_concentration(single, task="Single"),
            _aggregate_force_concentration(triple, task="Triple"),
        ],
        ignore_index=True,
    )
    ctx.tables["fig5_exact_force_field_concentration_aggregates"] = aggregates

    fig = plt.figure(figsize=(9.2, 5.6))
    grid = fig.add_gridspec(1, 2, wspace=0.40)
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    for axis, task in zip(axes, ("Single", "Triple"), strict=True):
        _plot_exact_force_concentration_panel(
            axis,
            aggregates[aggregates["task"].eq(task)],
            title=task,
        )
    handles = [
        mpl.lines.Line2D(
            [0], [0], marker=FIG5_MARKERS[method], linestyle="none",
            markerfacecolor=(
                "none" if method in FIG5_HOLLOW_METHODS else COLORS[method]
            ),
            markeredgecolor=(
                COLORS[method] if method in FIG5_HOLLOW_METHODS else "white"
            ),
            markersize=9.5, label=method,
            markeredgewidth=1.8 if method in FIG5_HOLLOW_METHODS else 1.1,
        )
        for method in FIG5_METHOD_ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=3, frameon=False, columnspacing=1.8, handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.105, right=0.97, top=0.77, bottom=0.15)
    outside_grid_panel_labels(fig, (grid[0, 0], grid[0, 1]), tuple("ab"))
    return _save(ctx, fig, stem)


def _figure6_vertical_compensated_fe(
    ctx: p.PipelineContext,
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the one requested Triple FE vertical-load compensation experiment.

    The production Triple problem, alpha, curvature weight, smoothing schedule,
    matched main iteration cap, seed and cold start are unchanged. The only changed
    objective term is the opt-in fixed upward radiation-force target equal to
    the buoyancy-corrected bead weight.
    """

    memo_key = "final_fig6_vertical_compensated_fe"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    bank = benchmark["long_bank"]
    node = bank["objects"][(1, 1)]
    problem = node["problem"]
    base_config = node["fe_config"]
    compensated_config = replace(
        base_config,
        compensate_effective_gravity=True,
    )
    base_fields = asdict(base_config)
    compensated_fields = asdict(compensated_config)
    changed_fields = {
        key
        for key in base_fields
        if base_fields[key] != compensated_fields[key]
    }
    if changed_fields != {"compensate_effective_gravity"}:
        raise RuntimeError(
            "Figure 6 compensation changed fields other than the opt-in flag: "
            f"{sorted(changed_fields)}"
        )
    iteration_cap = sum(int(value) for value in compensated_config.smooth_stage_maxiters)
    if iteration_cap != int(bank["conventional_cap"]):
        raise RuntimeError("Figure 6 compensated FE must retain the matched main Triple iteration cap")
    initial_phase = np.asarray(bank["initial_phase"], dtype=float)
    seed = int(bank["seed"])
    upward_target_n = effective_weight_force_target_n(problem.material)
    payload = {
        "contract": "fig6-fixed-effective-gravity-compensated-triple-fe-v1",
        "problem": problem.fingerprint,
        "base_config": base_config.to_payload(),
        "compensated_config": compensated_config.to_payload(),
        "seed": seed,
        "initial_phase_content_id": digest_array(initial_phase),
        "fixed_iteration_cap": iteration_cap,
        "fixed_upward_force_target_n": upward_target_n.tolist(),
    }

    compensated_run, _, _ = ctx.cache.get_or_compute(
        "final_fig6_vertical_compensated_fe",
        payload,
        lambda: run_multitrap(
            problem,
            compensated_config,
            seed=seed,
            initial_phases=initial_phase,
            record_history=True,
            metadata={
                "display_method": "FE+g",
                "experiment_role": "Figure 6 vertical-load compensation only",
                "fixed_iteration_cap": iteration_cap,
                "fixed_upward_force_target_n": upward_target_n.tolist(),
            },
        ),
        recompute=ctx.recompute,
    )
    compensated_phase = np.asarray(compensated_run.phases, dtype=float)
    targets = np.asarray(benchmark["targets_m"], dtype=float)

    def validate_target(target_index: int) -> dict[str, Any]:
        target = targets[target_index]
        result = p._finite_ka_validation(
            ctx,
            compensated_phase,
            target,
            key=f"fig6-triple-FE-gz-target-{target_index + 1}",
        )
        row = p._static_validation_row("FE+g", result)
        field = p._array_field(ctx, compensated_phase)
        equilibrium = np.asarray(result.equilibrium.equilibrium_m, dtype=float)
        row["pressure_abs_at_intended_target_pa"] = float(
            np.abs(field.pressure(target[None, :]))[0]
        )
        row["pressure_abs_at_exact_equilibrium_pa"] = (
            float(np.abs(field.pressure(equilibrium[None, :]))[0])
            if bool(result.equilibrium.numerical_root_found)
            else math.nan
        )
        row.update(
            {
                "display_method": "FE+g",
                "task": "Triple",
                "target_index": int(target_index),
                "target_id": f"T{target_index + 1}",
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "phase_content_id": digest_array(compensated_phase),
                "phase_source": (
                    "fixed Triple FE endpoint with the buoyancy-corrected "
                    f"effective-weight residual; maxiter={iteration_cap}; all other settings unchanged"
                ),
                "terminal_iteration": int(compensated_run.iterations),
                "method_family": "optimization",
                "preset_role": "not_applicable",
                "preset_radius_lambda": math.nan,
                "command_time_s": float(compensated_run.end_to_end_sec),
                "fixed_fe_alpha_force": float(TRIPLE_FE_ENDPOINT.alpha),
                "fixed_fe_curvature_wx": float(
                    TRIPLE_FE_ENDPOINT.curvature_weight_override[0][0]
                ),
                "fixed_fe_curvature_wy": float(
                    TRIPLE_FE_ENDPOINT.curvature_weight_override[1][1]
                ),
                "fixed_fe_curvature_wz": float(
                    TRIPLE_FE_ENDPOINT.curvature_weight_override[2][2]
                ),
                "run_mode": ctx.config.mode,
                "manuscript_numerics": bool(ctx.config.full),
            }
        )
        return {
            "row": row,
            "result": result,
            "spec": {
                "method": "FE+g",
                "phase": compensated_phase,
                "phase_source": row["phase_source"],
            },
        }

    workers = max(
        1,
        min(len(targets), int(os.environ.get("HAT_EXACT_WORKERS", "1"))),
    )
    if workers == 1:
        evaluations = [validate_target(index) for index in range(len(targets))]
    else:
        ordered: list[dict[str, Any] | None] = [None] * len(targets)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(validate_target, index): index
                for index in range(len(targets))
            }
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        evaluations = [item for item in ordered if item is not None]
    if len(evaluations) != len(targets):
        raise RuntimeError("Figure 6 compensated FE validation is incomplete")

    standard_run = node["FE"]
    standard_phase = np.asarray(node["FE_phase"], dtype=float)
    method_evaluations = (
        (
            "FE",
            standard_phase,
            base_config,
            standard_run,
            [
                item
                for item in benchmark["evaluations"]
                if item["row"]["method"] == "FE"
            ],
        ),
        (
            "FE+g",
            compensated_phase,
            compensated_config,
            compensated_run,
            evaluations,
        ),
    )
    comparison_rows: list[dict[str, Any]] = []
    for method, phase, config, run, exact_evaluations in method_evaluations:
        exact_by_target = {
            int(item["row"]["target_index"]): item["row"]
            for item in exact_evaluations
        }
        objective_evaluation = evaluate_multitrap_objective(
            problem,
            phase,
            config,
            force_epsilon=float(run.stages[-1].force_epsilon),
            uniformity_epsilon=float(run.stages[-1].uniformity_epsilon),
        )
        prescribed_force_n = (
            upward_target_n
            if config.compensate_effective_gravity
            else np.zeros(3, dtype=float)
        )
        for target_index, local in enumerate(objective_evaluation.locals):
            exact_row = exact_by_target[target_index]
            residual_n = np.asarray(local.force_vector_n, dtype=float) - prescribed_force_n
            comparison_rows.append(
                {
                    "method": method,
                    "target_index": int(target_index),
                    "target_id": f"T{target_index + 1}",
                    "compensation_enabled": bool(
                        config.compensate_effective_gravity
                    ),
                    "effective_weight_n": float(upward_target_n[2]),
                    "prescribed_radiation_force_x_n": float(prescribed_force_n[0]),
                    "prescribed_radiation_force_y_n": float(prescribed_force_n[1]),
                    "prescribed_radiation_force_z_n": float(prescribed_force_n[2]),
                    "surrogate_radiation_force_x_n": float(local.force_vector_n[0]),
                    "surrogate_radiation_force_y_n": float(local.force_vector_n[1]),
                    "surrogate_radiation_force_z_n": float(local.force_vector_n[2]),
                    "surrogate_force_residual_x_n": float(residual_n[0]),
                    "surrogate_force_residual_y_n": float(residual_n[1]),
                    "surrogate_force_residual_z_n": float(residual_n[2]),
                    "surrogate_force_residual_norm_n": float(
                        np.linalg.norm(residual_n)
                    ),
                    "exact_root_found": bool(exact_row["finite_ka_root_found"]),
                    "exact_displacement_x_m": float(
                        exact_row["finite_ka_displacement_x_m"]
                    ),
                    "exact_displacement_y_m": float(
                        exact_row["finite_ka_displacement_y_m"]
                    ),
                    "exact_displacement_z_m": float(
                        exact_row["finite_ka_displacement_z_m"]
                    ),
                    "exact_displacement_a": float(
                        exact_row["finite_ka_displacement_a"]
                    ),
                    "pressure_abs_at_exact_equilibrium_pa": float(
                        exact_row["pressure_abs_at_exact_equilibrium_pa"]
                    ),
                    "phase_content_id": digest_array(phase),
                    "terminal_iteration": int(run.iterations),
                    "iteration_cap": iteration_cap,
                    "seed": seed,
                    "initial_phase_content_id": digest_array(initial_phase),
                }
            )
    compensation_table = pd.DataFrame(comparison_rows)
    ctx.tables["fig6_vertical_compensation_all_targets"] = compensation_table
    value = {
        "run": compensated_run,
        "config": compensated_config,
        "phase": compensated_phase,
        "evaluations": evaluations,
        "compensation_table": compensation_table,
        "fixed_upward_force_target_n": upward_target_n,
    }
    ctx.memo[memo_key] = value
    return value


def figure_6_force_field_comparison(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Two-column, three-row Triple context and force cards, including GS."""

    stem = "Figure_6_force_field_comparison"
    if stem in ctx.figures:
        return ctx.figures[stem]
    benchmark = _triple_exact_benchmark_with_concentration(ctx)
    compensation = _figure6_vertical_compensated_fe(ctx, benchmark)
    from .thermoelastic_sensitivity import run_bead_sensitivity
    run_bead_sensitivity(ctx, benchmark, compensation)
    method_order = ("Conventional", "FE", "FE+g", "GS", "AD")
    selected: list[dict[str, Any]] = []
    for method in method_order:
        source = (
            compensation["evaluations"]
            if method == "FE+g"
            else benchmark["evaluations"]
        )
        matches = [
            item
            for item in source
            if item["row"]["method"] == method
            and int(item["row"]["target_index"]) == 1
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one matched Triple-T2 evaluation for {method}, found {len(matches)}"
            )
        selected.append(matches[0])

    dense_sections = {
        method: p._pressure_sections(
            p._array_field(ctx, np.asarray(item["spec"]["phase"], dtype=float)),
            np.asarray(benchmark["targets_m"][1], dtype=float),
            half_xy_m=0.006,
            half_z_m=0.006,
            samples=_plot_points(ctx, "pressure_points", 91),
        )
        for method, item in zip(method_order, selected, strict=True)
    }
    force_keys = {
        method: f"final-fig6-triple-T2-{method}-xz"
        for method in method_order
    }

    def precompute_force(item: dict[str, Any], method: str) -> Any:
        return p._force_section_cached(
            ctx,
            np.asarray(item["spec"]["phase"], dtype=float),
            np.asarray(benchmark["targets_m"][1], dtype=float),
            "xz",
            key=force_keys[method],
        )

    force_sections: dict[str, Any] = {}
    workers = max(
        1,
        min(len(selected), int(os.environ.get("HAT_EXACT_WORKERS", "1"))),
    )
    if workers == 1:
        for method, item in zip(method_order, selected, strict=True):
            force_sections[method] = precompute_force(item, method)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(precompute_force, item, method): method
                for method, item in zip(method_order, selected, strict=True)
            }
            for future in as_completed(futures):
                force_sections[futures[future]] = future.result()
    if set(force_sections) != set(method_order):
        raise RuntimeError("Figure 6 force-section precomputation is incomplete")

    fe_item = next(item for item in selected if item["row"]["method"] == "FE")
    fe_phase = np.asarray(fe_item["spec"]["phase"], dtype=float)
    context_axis, context_amplitude = _tri3_xy_pressure(
        ctx, fe_phase, samples=141 if ctx.config.full else 101
    )
    pressure_norm = shared_power_norm(
        [context_amplitude]
        + [dense_sections[method]["xz"] for method in method_order],
        lower=0.0,
        upper=99.5,
    )
    pressure_tick_step = 1.0e3
    pressure_ticks = np.arange(
        0.0,
        pressure_tick_step
        * (math.floor(float(pressure_norm.vmax) / pressure_tick_step) + 1),
        pressure_tick_step,
    )
    pressure_ticks = np.unique(np.r_[pressure_ticks[::2], pressure_ticks[-1]])
    fig = plt.figure(figsize=(10.8, 13.8))
    grid = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.40)
    summary_rows: list[dict[str, Any]] = []
    target = np.asarray(benchmark["targets_m"][1], dtype=float)

    context_ax = fig.add_subplot(grid[0, 0])
    context_image = field_panel(
        context_ax,
        context_amplitude,
        (
            1.0e3 * context_axis[0],
            1.0e3 * context_axis[-1],
            1.0e3 * context_axis[0],
            1.0e3 * context_axis[-1],
        ),
        norm=pressure_norm,
        xlabel=r"$x$ (mm)",
        ylabel=r"$y$ (mm)",
    )
    for target_index, point in enumerate(
        np.asarray(benchmark["targets_m"], dtype=float), start=1
    ):
        target_marker(
            context_ax,
            1.0e3 * point[0], 1.0e3 * point[1], marker="x",
            s=54, linewidth=1.7, zorder=20,
        )
        context_ax.annotate(
            rf"$T_{target_index}$",
            (1.0e3 * point[0], 1.0e3 * point[1]),
            xytext=(4, 5), textcoords="offset points", color="white",
            fontsize=8.0, fontweight="semibold", zorder=21,
        )
    context_ax.scatter(
        1.0e3 * target[0], 1.0e3 * target[1], s=110,
        facecolor="none", edgecolor="white", linewidth=1.4, zorder=19,
    )
    set_panel_title(context_ax, r"FE — Triple $xy$")
    context_ax.set_box_aspect(1)
    add_adjacent_colorbar(
        context_image, context_ax, size="3.4%", pad="2.4%",
        label=r"$|p|$ (Pa)", labelsize=8.0, ticksize=7.0,
        ticks=pressure_ticks,
    )

    for index, (method, item) in enumerate(zip(method_order, selected, strict=True)):
        result = item["result"]
        phase = np.asarray(item["spec"]["phase"], dtype=float)
        equilibrium = np.asarray(result.equilibrium.equilibrium_m, dtype=float)
        resolved = bool(result.equilibrium.numerical_root_found)
        displacement_a = float(item["row"]["finite_ka_displacement_a"])
        pressure_at_equilibrium_pa = float(
            item["row"]["pressure_abs_at_exact_equilibrium_pa"]
        )
        display_method = r"FE+$g$" if method == "FE+g" else method
        card_title = (
            display_method
            + "\n"
            + rf"$d_{{\rm eq}}/a={displacement_a:.2f},\ "
            + rf"|p_{{\rm eq}}|={pressure_at_equilibrium_pa:.0f}\ \mathrm{{Pa}}$"
        )
        field_ax = fig.add_subplot(grid[(index + 1) // 2, (index + 1) % 2])
        p._plot_force_card(
            ctx,
            field_ax,
            phase,
            target,
            equilibrium,
            equilibrium_resolved=resolved,
            plane="xz",
            key=force_keys[method],
            title=card_title,
            dense=dense_sections[method],
            pressure_norm=pressure_norm,
        )
        for artist in list(field_ax.texts):
            if artist.get_text() == "unresolved candidate":
                artist.remove()
        if resolved:
            displacement_mm = 1.0e3 * (equilibrium - target)
            field_ax.plot(
                [0.0, float(displacement_mm[0])],
                [0.0, float(displacement_mm[2])],
                color="white",
                lw=1.15,
                alpha=0.95,
                zorder=19,
            )
        add_adjacent_colorbar(
            field_ax.images[0], field_ax, size="3.4%", pad="2.4%",
            label=r"$|p|$ (Pa)", labelsize=8.0, ticksize=7.0,
            ticks=pressure_ticks,
        )
        field_ax.set_box_aspect(1)
        field_ax.set_xlim(-6.0, 6.0)
        field_ax.set_ylim(-6.0, 6.0)
        summary = dict(item["row"])
        summary.update(
            {
                "plane": "xz",
                "force_section_half_width_m": 0.006,
                "pressure_section_half_width_m": 0.006,
                "pressure_norm_vmin_pa": float(pressure_norm.vmin),
                "pressure_norm_vmax_pa": float(pressure_norm.vmax),
                "force_section_precomputed": method in force_sections,
                "context_method": "FE",
                "context_plane": "xy",
                "analyzed_target_id": "T2",
                "figure6_vertical_compensation": method == "FE+g",
                "exact_worker_count": workers,
            }
        )
        summary_rows.append(summary)
    ctx.tables["fig6_triple_force_field_summary"] = pd.DataFrame(summary_rows)
    fig.subplots_adjust(left=0.105, right=0.950, top=0.945, bottom=0.060)
    outside_grid_panel_labels(fig, tuple(grid[row,col] for row in range(3) for col in range(2)),
                              tuple("abcdef"), dx=0.016, dy=0.008)
    return _save(ctx, fig, stem)


def appendix_single_sided_gs_prescription(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Expose the array-specific radius engineering used by the GS baseline."""

    stem = "Appendix_A_single_sided_GS_prescription"
    if stem in ctx.figures:
        return ctx.figures[stem]

    wavelength = p.AcousticMedium().sound_speed_m_s / ctx.config.frequency_hz
    reference_spec = p.NOMINAL_REFERENCE_VORTEX_SPEC
    engineered_spec = p.SINGLE_SIDED_GS_VORTEX_SPEC
    cases = (
        ("reference-current", p.MAIN_TARGET_M.copy(), reference_spec),
        ("reference-distant", np.array([0.0, 0.0, 0.130]), reference_spec),
        ("single-sided-engineered", p.MAIN_TARGET_M.copy(), engineered_spec),
    )
    half_width = 0.015
    samples = 181 if ctx.config.full else 141
    axis = np.linspace(-half_width, half_width, samples)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    theta = np.linspace(0.0, 2.0 * np.pi, 180 if ctx.config.full else 96, endpoint=False)
    radial_axis = np.linspace(0.0, 0.012, 241 if ctx.config.full else 161)
    rr, tt = np.meshgrid(radial_axis, theta, indexing="ij")
    records: list[dict[str, Any]] = []
    fields: list[np.ndarray] = []
    profiles: dict[str, np.ndarray] = {}

    radial_offsets = np.stack(
        (rr * np.cos(tt), rr * np.sin(tt), np.zeros_like(rr)), axis=-1
    )

    def radial_profile(field: Any, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = target[None, None, :] + radial_offsets
        amplitude = np.abs(field.pressure(points.reshape(-1, 3))).reshape(rr.shape)
        return np.mean(amplitude, axis=1), amplitude

    def first_peak_index(profile: np.ndarray) -> int:
        peaks, _ = find_peaks(
            profile,
            prominence=max(float(np.max(profile)) * 0.01, 1.0e-30),
            distance=4,
        )
        peaks = peaks[radial_axis[peaks] >= 0.00015]
        return int(peaks[0]) if len(peaks) else int(np.argmax(profile))

    fe_phase = next(
        spec["phase"] for spec in p._central_exact_endpoint_specs(ctx)
        if spec["method"] == "FE"
    )
    fe_profile, _ = radial_profile(p._array_field(ctx, np.asarray(fe_phase)), p.MAIN_TARGET_M)
    fe_peak_index = first_peak_index(fe_profile)
    fe_profile_normalized = fe_profile / max(float(fe_profile[fe_peak_index]), 1.0e-30)
    profiles["FE"] = fe_profile_normalized

    for case_id, target, spec in cases:
        transfer, signature = p._vortex_transfer_factory(ctx, target, spec=spec)()
        result = phase_only_signature_projection_audit(transfer, signature, iterations=100)
        field = p._array_field(ctx, result.phase_rad)
        plane_points = np.stack(
            [target[0] + xx, target[1] + yy, np.full_like(xx, target[2])], axis=-1
        )
        amplitude = np.abs(field.pressure(plane_points.reshape(-1, 3))).reshape(xx.shape)
        profile, polar_amplitude = radial_profile(field, target)
        peak_index = first_peak_index(profile)
        ring_amplitude = polar_amplitude[peak_index]
        center_amplitude = float(np.abs(field.pressure(target[None, :]))[0])
        mean_ring = float(profile[peak_index])
        spacing_lambda = 2.0 * spec.radius_wavelengths * math.sin(math.pi / spec.point_count)
        normalized_profile = profile / max(mean_ring, 1.0e-30)
        local_rmse = (
            float(np.sqrt(np.mean((normalized_profile - fe_profile_normalized) ** 2)))
            if math.isclose(float(target[2]), float(p.MAIN_TARGET_M[2]), abs_tol=1.0e-12)
            else math.nan
        )
        records.append(
            {
                "case": case_id,
                "control_points": int(spec.point_count),
                "radius_lambda": float(spec.radius_wavelengths),
                "target_z_mm": float(1.0e3 * target[2]),
                "adjacent_control_spacing_lambda": float(spacing_lambda),
                "realized_first_maximum_mm": float(1.0e3 * radial_axis[peak_index]),
                "realized_first_maximum_lambda": float(radial_axis[peak_index] / wavelength),
                "first_maximum_pressure_pa": mean_ring,
                "first_maximum_angular_cv": float(np.std(ring_amplitude) / mean_ring),
                "first_maximum_min_to_max": float(np.min(ring_amplitude) / np.max(ring_amplitude)),
                "center_to_first_maximum": float(center_amplitude / mean_ring),
                "radial_profile_rmse_to_fe_0_12mm": local_rmse,
                "solver": "phase-only GS",
                "topological_charge": int(spec.topological_charge),
            }
        )
        fields.append(amplitude / max(float(np.max(amplitude)), 1.0e-30))
        profiles[case_id] = normalized_profile

    ctx.tables["appendix_single_sided_gs_prescription"] = pd.DataFrame(records)
    radius_records: list[dict[str, Any]] = []
    for radius_lambda in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.90, 1.10, 1.40):
        spec = p.VortexControlSpec(
            point_count=8,
            radius_wavelengths=radius_lambda,
            topological_charge=1,
        )
        transfer, signature = p._vortex_transfer_factory(
            ctx, p.MAIN_TARGET_M, spec=spec
        )()
        result = phase_only_signature_projection_audit(transfer, signature, iterations=100)
        field = p._array_field(ctx, result.phase_rad)
        profile, polar_amplitude = radial_profile(field, p.MAIN_TARGET_M)
        peak_index = first_peak_index(profile)
        ring_amplitude = polar_amplitude[peak_index]
        normalized_profile = profile / max(float(profile[peak_index]), 1.0e-30)
        radius_records.append(
            {
                "control_points": 8,
                "radius_lambda": radius_lambda,
                "adjacent_control_spacing_lambda": float(
                    2.0 * radius_lambda * math.sin(math.pi / 8.0)
                ),
                "realized_first_maximum_mm": float(1.0e3 * radial_axis[peak_index]),
                "realized_first_maximum_lambda": float(radial_axis[peak_index] / wavelength),
                "first_maximum_angular_cv": float(np.std(ring_amplitude) / np.mean(ring_amplitude)),
                "first_maximum_min_to_max": float(np.min(ring_amplitude) / np.max(ring_amplitude)),
                "radial_profile_rmse_to_fe_0_12mm": float(
                    np.sqrt(np.mean((normalized_profile - fe_profile_normalized) ** 2))
                ),
            }
        )
    ctx.tables["appendix_single_sided_gs_radius_sweep"] = pd.DataFrame(
        radius_records
    )

    fig = plt.figure(figsize=(12.2, 9.3))
    grid = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.34)
    field_titles = (
        r"$R=1.4\lambda,\ z^*=50$ mm",
        r"$R=1.4\lambda,\ z^*=130$ mm",
        r"$R=0.45\lambda,\ z^*=50$ mm",
    )
    extent = 1.0e3 * np.array([axis[0], axis[-1], axis[0], axis[-1]])
    for index, (amplitude, title) in enumerate(zip(fields, field_titles, strict=True)):
        row, column = divmod(index, 2)
        ax = fig.add_subplot(grid[row, column])
        image = ax.imshow(
            amplitude,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        ax.set_xlabel(r"$x-x^*$ (mm)")
        ax.set_ylabel(r"$y-y^*$ (mm)")
        ax.set_box_aspect(1)
        ax.set_title(title, fontsize=10.5, pad=7)
        add_adjacent_colorbar(
            image, ax, label=r"$|p|/|p|_{\max}$", labelsize=8.0, ticksize=7.0
        )

    profile_ax = fig.add_subplot(grid[1, 1])
    profile_ax.plot(1.0e3 * radial_axis, profiles["FE"], lw=1.9, color=COLORS["FE"], label="FE")
    profile_ax.plot(
        1.0e3 * radial_axis, profiles["reference-current"], lw=1.7,
        color=COLORS["IB"], label=r"GS, $R=1.4\lambda$",
    )
    profile_ax.plot(
        1.0e3 * radial_axis, profiles["single-sided-engineered"], lw=1.7,
        color=COLORS["GS-PO audit"], label=r"GS, $R=0.45\lambda$",
    )
    profile_ax.axvline(
        1.0e3 * radial_axis[fe_peak_index], color="0.35", lw=0.9, ls=":"
    )
    profile_ax.set_xlim(0.0, 12.0)
    profile_ax.set_xlabel(r"radius from $\mathbf{r}^*$ (mm)")
    profile_ax.set_ylabel(r"azimuthal mean / first maximum")
    profile_ax.grid(alpha=0.20)
    profile_ax.legend(frameon=False, fontsize=8.0, loc="best")
    profile_ax.set_box_aspect(1)

    fig.subplots_adjust(left=0.075, right=0.965, top=0.91, bottom=0.08)
    outside_grid_panel_labels(
        fig,
        tuple(grid[row, column] for row in range(2) for column in range(2)),
        tuple("abcd"),
    )
    return _save(ctx, fig, stem)


def _alpha_bank(ctx: p.PipelineContext) -> dict[str, Any]:
    if "final_alpha_bank" in ctx.memo:
        return ctx.memo["final_alpha_bank"]
    single = _single_vortex_bank(ctx)
    seed = int(single["seeds"][0])
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ctx.positions_m))
    maxiter = 10_000 if ctx.config.full else int(
        os.environ.get("HAT_QUICK_ALPHA_CAP", "10000")
    )
    runs: list[Any] = []
    rows: list[dict[str, Any]] = []
    phases: list[np.ndarray] = []
    target_hessian_eigenvalues: list[np.ndarray] = []
    production_bank = fixed_single_branch_pack(ctx)["bank"]
    root_ix, root_iy = production_bank.root_target
    source_row = production_bank.medoid_table[
        production_bank.medoid_table["ix"].eq(root_ix)
        & production_bank.medoid_table["iy"].eq(root_iy)
    ].iloc[0]
    for alpha in ALPHA_VALUES:
        spec = MethodSpec(
            name="Force-Equilibrium", alpha_per_m=float(alpha), beta_curvature_per_pa=0.0,
            pressure_mode="abs", curvature_weight=np.eye(3),
        )
        condition = Condition(
            label=f"alpha-{alpha:g}", positions=ctx.positions_m, target_m=tuple(p.MAIN_TARGET_M),
            frequency_hz=ctx.config.frequency_hz, curvature_weight=np.eye(3), normals=ctx.normals,
        )
        objective = SingleTargetObjective(condition, spec)
        payload = {
            "contract": "final-alpha-transition-v1", "alpha_per_m": float(alpha),
            "seed": seed, "initial": digest_array(initial), "maxiter": maxiter, "gtol": ctx.config.gtol,
        }
        run, _, _ = ctx.cache.get_or_compute(
            "final_alpha_run", payload,
            lambda objective=objective: solve_corrected_gorkov_fe(
                objective, initial, maxiter=maxiter, gtol=ctx.config.gtol
            ),
            recompute=ctx.recompute,
        )
        surrogate = objective._raw_metrics(run.phase_rad, need_phase_gradient=False)
        target_hessian = np.asarray(surrogate["potential_hessian_n_m"], dtype=float)
        target_hessian_eigenvalues.append(
            np.linalg.eigvalsh(0.5 * (target_hessian + target_hessian.T))
        )
        runs.append(run); phases.append(np.asarray(run.phase_rad, dtype=float))
    phase_array = np.stack(phases)
    repeated = pd.DataFrame([source_row.to_dict() for _ in ALPHA_VALUES])
    states = target_demodulated_states(phase_array, repeated)
    references = np.stack(
        [
            production_bank.target_medoid(root_ix, root_iy, component)
            for component in COMPONENTS
        ]
    )
    distance = p._projective_distance_between(states, references)
    assignment = np.argmin(distance, axis=1)
    for index, (alpha, run) in enumerate(zip(ALPHA_VALUES, runs)):
        row = {
            "alpha_per_m": float(alpha), "seed": seed, "maxiter": maxiter,
            "iterations": int(run.iterations), "wall_time_s": float(run.history_wall_s[-1]),
            "objective": float(run.objective), "gradient_norm": float(run.gradient_norm),
            "component": COMPONENTS[int(assignment[index])],
            "distance_to_B1": float(distance[index, 0]), "distance_to_B2": float(distance[index, 1]),
            "component_margin": float(abs(distance[index, 0] - distance[index, 1])),
            "phase_content_id": digest_array(run.phase_rad), "run_mode": ctx.config.mode,
            "target_gorkov_hessian_min_n_m": float(target_hessian_eigenvalues[index][0]),
            "target_gorkov_hessian_mid_n_m": float(target_hessian_eigenvalues[index][1]),
            "target_gorkov_hessian_max_n_m": float(target_hessian_eigenvalues[index][2]),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    ctx.tables["fig7_alpha_transition"] = table
    value = {"runs": runs, "phases": phase_array, "table": table, "seed": seed}
    ctx.memo["final_alpha_bank"] = value
    return value


def figure_7_alpha_transition(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Three predeclared broad horizontal pressure fields."""

    stem = "Figure_7_alpha_transition"
    if stem in ctx.figures:
        return ctx.figures[stem]
    bank = _alpha_bank(ctx)
    fig = compose_figure_7_xy(ctx, bank, ALPHA_VALUES, p.MAIN_TARGET_M)
    return _save(ctx, fig, stem)


def _command_time(ctx: p.PipelineContext, spec: Mapping[str, Any]) -> tuple[float, str]:
    method = str(spec["method"])
    retained = _spec_measured_command_time(spec)
    if np.isfinite(retained):
        return float(retained), "measured command time retained on endpoint specification"
    if method == "Conventional":
        row = ctx.artifact.trajectory_table.iloc[int(spec["source_index"])]
        return float(row["runtime_s"]), "runtime stored with the deposited Conventional endpoint"
    if method == "FE":
        row = ctx.artifact.anchor_table.iloc[int(spec["source_index"])]
        return float(row["runtime_s"]), "runtime stored with the deposited FE endpoint"
    baselines = p._ensure_field_baseline_commands(ctx)
    if method not in {"IB", "GS"}:
        raise RuntimeError(f"No measured command time retained for {method}")
    run = baselines["ib"] if method == "IB" else baselines["audit"]
    return float(run.timing.total_s), "timing stored with the generated baseline command"


def _multitrap_time_at_iteration(result: Any, iteration: int) -> float:
    records = [record for record in result.history if int(record.global_iteration) <= int(iteration)]
    if not records:
        raise RuntimeError(f"No multitrap timing record at or before iteration {iteration}")
    record = max(records, key=lambda item: int(item.global_iteration))
    if int(record.global_iteration) != int(iteration):
        raise RuntimeError(
            f"Multitrap history ended at {record.global_iteration}, not requested iteration {iteration}"
        )
    return float(result.setup_sec + record.elapsed_sec)


def _nondominated_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    best = math.inf
    chosen: list[int] = []
    for index in order:
        if y[index] < best:
            chosen.append(int(index)); best = float(y[index])
    return np.asarray(chosen, dtype=int)


def figure_8_performance_time_tradeoff(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Standalone performance–time tradeoff and Discussion bridge."""

    return _figure_8_methods_only(ctx)


def build_story_tables(ctx: p.PipelineContext) -> dict[str, pd.DataFrame]:
    """Return only the numeric summaries required to write the eight-figure story."""

    required = {
        "fig1_single_vortex_runs",
        "fig1_xz_domain_summary",
        "fig2_full_dimensional_distance_summary",
        "fig2_full_dimensional_correlation_summary",
        "fig3_local_decision_sections",
        "fig4_triple_local_geometry_summary",
        "fig5_exact_force_field_concentration",
        "fig5_triple_exact_jacobian_step_audit",
        "fig6_triple_force_field_summary",
        "fig6_vertical_compensation_all_targets",
        "fig7_alpha_transition",
        "fig8_performance_time_tradeoff",
    }
    missing = required - set(ctx.tables)
    if missing:
        raise RuntimeError(f"Build all main figures before story tables; missing {sorted(missing)}")

    fig1 = ctx.tables["fig1_single_vortex_runs"]
    fig1_summary = (
        fig1.groupby("method", sort=False)
        .agg(
            runs=("pair_index", "count"),
            near_stationary=("near_stationary", "sum"),
            wall_time_median_s=("wall_time_s", "median"),
            wall_time_q25_s=("wall_time_s", lambda values: values.quantile(0.25)),
            wall_time_q75_s=("wall_time_s", lambda values: values.quantile(0.75)),
            iterations_median=("iterations", "median"),
            iterations_q25=("iterations", lambda values: values.quantile(0.25)),
            iterations_q75=("iterations", lambda values: values.quantile(0.75)),
            terminal_gradient_median=("gradient_norm", "median"),
        )
        .reset_index()
    )

    exact = ctx.tables["fig5_exact_force_field_concentration"]
    exact_summary = exact[
        [
            "task",
            "target_id",
            "display_method",
            "method_family",
            "preset_role",
            "preset_radius_lambda",
            "field_concentration_fraction",
            "finite_ka_root_found",
            "finite_ka_displacement_a",
            "finite_ka_target_radiation_force_norm_n",
            "finite_ka_target_force_norm_n",
            "finite_ka_target_force_z_n",
            "finite_ka_effective_weight_n",
            "finite_ka_stiffness_min_n_m",
            "pressure_abs_at_intended_target_pa",
            "pressure_abs_at_exact_equilibrium_pa",
        ]
    ].copy()

    alpha = ctx.tables["fig7_alpha_transition"]
    alpha_summary = alpha[
        [
            "alpha_per_m",
            "component",
            "distance_to_B1",
            "distance_to_B2",
            "component_margin",
            "iterations",
            "maxiter",
            "gradient_norm",
            "target_gorkov_hessian_min_n_m",
        ]
    ].copy()

    story_chain = pd.DataFrame(
        [
            {
                "figure": 1,
                "visible_claim": "Across the sampled XZ trapping domain, Conventional and FE retain corresponding Vortex pressure morphologies but differ consistently in solve time and terminal gradient.",
                "body_or_table_evidence": "one linear accepted-iteration history, shared-scale XZ solve-time and terminal-gradient maps, the solver-exit taxonomy, and the separate matched XYZ volume audit",
                "handoff": "Is the field match a one-target coincidence?",
            },
            {
                "figure": 2,
                "visible_claim": "Across two FE-defined target-connected branches, later Conventional states lie closer to FE.",
                "body_or_table_evidence": "256-D projector-distance medians and representative-target raw objective/similarity correlations",
                "handoff": "Why does Conventional approach the same class so slowly?",
            },
            {
                "figure": 3,
                "visible_claim": "At matched endpoints, the same local decision coordinates expose sharply different objective-variation and terminal-gradient scales.",
                "body_or_table_evidence": "two-dimensional sections followed by one-dimensional cuts, with one-sided center slopes, terminal gradients and quadratic-fit residuals retained in the table",
                "handoff": "Does this local scale separation persist in a practical Triple task?",
            },
            {
                "figure": 4,
                "visible_claim": "At the selected Triple node, Conventional and FE produce corresponding fields but different local decision geometry.",
                "body_or_table_evidence": "matched pressure fields; ±3.675-rad sections (70% of the previous ±5.25-rad span), sparse local analytic-gradient directions; optimizer norms retained in tables in the archived native coordinates, evaluated at the current fixed endpoints",
                "handoff": "Does that difference survive an independent exact-force audit?",
            },
            {
                "figure": 5,
                "visible_claim": "The same exact force–field concentration coordinates compare Conventional, FE, FE+g, IB, GS and AD for both Single and Triple without changing the validator.",
                "body_or_table_evidence": "one method aggregate per task; per-endpoint concentration, exact root class, displacement, weakest restoring stiffness, pressure suppression at the resolved equilibrium, and matched-preset metadata remain in the table",
                "handoff": "What does the Triple force-field contrast look like spatially at one frozen off-axis target?",
            },
            {
                "figure": 6,
                "visible_claim": "At the representative off-axis Triple target T2, the fixed vertical-load-compensated FE experiment directly tests whether the predominantly downward FE offset is attributable to uncompensated effective weight.",
                "body_or_table_evidence": "one broad FE Triple context and five XZ force-field cards at T2 (Conventional, FE, FE+g, GS and AD); the separate Figure 6 table validates standard and compensated FE at all three targets under the same 30,000-iteration cap and start; IB and GS remain in Figures 5 and 8",
                "handoff": "What changes when equilibrium forcing is pushed further?",
            },
            {
                "figure": 7,
                "visible_claim": "The three predeclared alpha endpoints select different broad horizontal pressure morphologies, including a twin-like local-minimum regime at stronger force weighting.",
                "body_or_table_evidence": "three broad horizontal pressure fields only; fixed-reference branch distances, convergence status and target Hessian remain table evidence",
                "handoff": "What is the complete engineering tradeoff?",
            },
            {
                "figure": 8,
                "visible_claim": "The six methods occupy distinct time–precision regimes for both Single and Triple, while Triple pressure suppression is assessed separately at the same resolved exact equilibria.",
                "body_or_table_evidence": "four panels on one pooled broken-linear command-time mapping with Conventional, FE, FE+g, IB, GS and AD: Single exact displacement in (a), Single |p(r_eq)| in (b), Triple median exact displacement in (c), and Triple median |p(r_eq)| in (d); Triple bars show the T1–T3 min–max range; no Pareto line",
                "handoff": "Discussion",
            },
        ]
    )

    tables = {
        "figure_1_body_summary": fig1_summary,
        "figure_1_xz_domain_summary": ctx.tables["fig1_xz_domain_summary"],
        "figure_2_distance_summary": ctx.tables["fig2_full_dimensional_distance_summary"],
        "figure_2_correlation_summary": ctx.tables["fig2_full_dimensional_correlation_summary"],
        "figure_3_local_geometry_summary": ctx.tables["fig3_local_decision_sections"],
        "figure_4_triple_local_geometry_summary": ctx.tables["fig4_triple_local_geometry_summary"],
        "figure_5_exact_force_concentration_summary": exact_summary,
        "figure_5_triple_jacobian_step_audit": ctx.tables["fig5_triple_exact_jacobian_step_audit"],
        "figure_6_triple_force_field_summary": ctx.tables["fig6_triple_force_field_summary"],
        "figure_6_vertical_compensation_all_targets": ctx.tables[
            "fig6_vertical_compensation_all_targets"
        ],
        "figure_7_alpha_branch_summary": alpha_summary,
        "figure_8_tradeoff_rows": ctx.tables["fig8_performance_time_tradeoff"],
        "main_text_story_chain": story_chain,
    }
    for name, table in tables.items():
        ctx.tables[name] = table
    return tables


FINAL_MAIN_FIGURES = (
    figure_1_single_vortex_convergence,
    figure_2_two_branch_evolution,
    figure_3_local_decision_geometry,
    figure_4_triple_fields_and_decision,
    figure_5_single_triple_exact_force_concentration,
    figure_6_force_field_comparison,
    figure_7_alpha_transition,
    figure_8_performance_time_tradeoff,
)


def build_all_main_figures(ctx: p.PipelineContext) -> list[mpl.figure.Figure]:
    return [producer(ctx) for producer in FINAL_MAIN_FIGURES]


__all__ = [producer.__name__ for producer in FINAL_MAIN_FIGURES] + [
    "appendix_single_sided_gs_prescription",
    "build_all_main_figures", "build_story_tables", "FINAL_MAIN_FIGURES"
]
