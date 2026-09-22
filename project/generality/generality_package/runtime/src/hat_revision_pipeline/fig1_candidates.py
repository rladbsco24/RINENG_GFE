"""Alternative Figure 1 compositions from the deposited optimization domain.

The four candidates in this module are presentation alternatives only.  They
reuse the authoritative 7 x 7 Single-target optimization archive and the
representative central Conventional/FE endpoints already loaded by
``prepare_pipeline``.  No optimization, endpoint selection, or additional
parameter experiment is performed here.

Every candidate has the same two-column grammar (Conventional, FE), no subplot
titles, square axes, outside panel identifiers, and adjacent colorbars.  The
pressure cards use the requested broad +/-75 mm horizontal section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import numpy as np
import pandas as pd

from . import pipeline as p
from .figure_presentation import add_adjacent_colorbar, outside_grid_panel_labels
from .style import TARGET_COLOR, field_panel, shared_power_norm


_METHODS = ("Conventional", "FE")
_EXIT_ORDER = (
    "Near-stationary",
    "Nonzero-gradient stall",
    "Max iteration",
    "Other",
)
_EXIT_COLORS = {
    "Near-stationary": "#54A24B",
    "Nonzero-gradient stall": "#E45756",
    "Max iteration": "#B279A2",
    "Other": "#9D9D9D",
}
_EXIT_TICK_LABELS = ("Near stat.", r"Nonzero-$\nabla J$ stall", "Max iter.", "Other")
_STATIONARITY_TOLERANCE = 1.0e-3


def _archive_iteration_caps(ctx: p.PipelineContext) -> dict[str, int]:
    path = Path(ctx.data_root) / "corrected_branch_evolution" / "anchor_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    return {
        "Conventional": int(config["conventional_maxiter"]),
        "FE": int(config["anchor_maxiter"]),
    }


def _classify_exit(row: pd.Series) -> str:
    residual = float(row["gradient_rms"])
    if np.isfinite(residual) and residual <= _STATIONARITY_TOLERANCE:
        return "Near-stationary"
    message = str(row["optimizer_message"]).lower()
    if int(row["iterations"]) >= int(row["iteration_cap"]) or any(
        token in message
        for token in ("maximum number of iterations", "maxiter", "iteration limit")
    ):
        return "Max iteration"
    if (not bool(row["optimizer_success"])) or any(
        token in message
        for token in (
            "precision loss",
            "desired error not necessarily achieved",
            "line search",
            "safeguard",
            "stall",
        )
    ):
        return "Nonzero-gradient stall"
    return "Other"


def _domain_run_records(ctx: p.PipelineContext) -> pd.DataFrame:
    """Return a common-column view of the deposited FE and C runs."""

    memo_key = "fig1_candidate_domain_run_records"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key].copy()
    caps = _archive_iteration_caps(ctx)
    required = (
        "ix", "iy", "xi", "eta", "target_id", "target_x_m", "target_y_m",
        "target_z_m", "phase_gradient_rms", "nit", "nfev", "runtime_s",
        "scipy_success", "status", "message", "seed", "objective",
    )
    blocks: list[pd.DataFrame] = []
    for display, source, selector in (
        (
            "Conventional",
            ctx.artifact.trajectory_table,
            ctx.artifact.trajectory_table["method"].eq("Conventional"),
        ),
        (
            "FE",
            ctx.artifact.anchor_table,
            ctx.artifact.anchor_table["method"].eq("Force-Equilibrium"),
        ),
    ):
        missing = set(required) - set(source.columns)
        if missing:
            raise KeyError(f"Deposited {display} archive is missing {sorted(missing)}")
        block = source.loc[selector, list(required)].copy()
        block = block.rename(
            columns={
                "phase_gradient_rms": "gradient_rms",
                "nit": "iterations",
                "nfev": "evaluations",
                "runtime_s": "wall_time_s",
                "scipy_success": "optimizer_success",
                "status": "optimizer_status",
                "message": "optimizer_message",
            }
        )
        block.insert(0, "method", display)
        block["iteration_cap"] = int(caps[display])
        blocks.append(block)
    records = pd.concat(blocks, ignore_index=True)
    records["termination_mode"] = records.apply(_classify_exit, axis=1)
    records["archive_role"] = "deposited 7x7 Single-target optimization domain"
    records["correct_sign_gorkov"] = True
    ctx.tables["fig1_candidate_domain_run_records"] = records.copy()
    ctx.memo[memo_key] = records.copy()
    return records


def _dominant_exit(values: pd.Series) -> str:
    counts = values.value_counts()
    return min(
        _EXIT_ORDER,
        key=lambda mode: (-int(counts.get(mode, 0)), _EXIT_ORDER.index(mode)),
    )


def _domain_summary(ctx: p.PipelineContext) -> pd.DataFrame:
    memo_key = "fig1_candidate_domain_summary"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key].copy()
    records = _domain_run_records(ctx)
    keys = [
        "method", "ix", "iy", "xi", "eta", "target_id", "target_x_m",
        "target_y_m", "target_z_m",
    ]
    summary = (
        records.groupby(keys, sort=True, dropna=False)
        .agg(
            run_count=("seed", "size"),
            gradient_rms_median=("gradient_rms", "median"),
            gradient_rms_q25=("gradient_rms", lambda values: values.quantile(0.25)),
            gradient_rms_q75=("gradient_rms", lambda values: values.quantile(0.75)),
            wall_time_median_s=("wall_time_s", "median"),
            wall_time_q25_s=("wall_time_s", lambda values: values.quantile(0.25)),
            wall_time_q75_s=("wall_time_s", lambda values: values.quantile(0.75)),
            dominant_termination_mode=("termination_mode", _dominant_exit),
        )
        .reset_index()
    )
    for mode in _EXIT_ORDER:
        fractions = (
            records.assign(_is_mode=records["termination_mode"].eq(mode).astype(float))
            .groupby(keys, sort=True, dropna=False)["_is_mode"]
            .mean()
            .rename(f"fraction_{mode.lower().replace('-', '_').replace(' ', '_')}")
            .reset_index()
        )
        summary = summary.merge(fractions, on=keys, how="left", validate="one_to_one")
    summary["exit_code"] = summary["dominant_termination_mode"].map(
        {mode: index for index, mode in enumerate(_EXIT_ORDER)}
    ).astype(int)
    ctx.tables["fig1_candidate_domain_summary"] = summary.copy()
    ctx.tables["fig1_candidate_exit_taxonomy"] = (
        records.groupby(["method", "ix", "iy", "termination_mode"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .merge(
            records.groupby(["method", "ix", "iy"], sort=True)
            .size()
            .rename("total")
            .reset_index(),
            on=["method", "ix", "iy"],
            how="left",
            validate="many_to_one",
        )
        .assign(fraction=lambda table: table["count"] / table["total"])
    )
    ctx.memo[memo_key] = summary.copy()
    return summary


def _representative_fields(ctx: p.PipelineContext) -> dict[str, Any]:
    memo_key = "fig1_candidate_representative_fields"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    target = np.asarray(p.MAIN_TARGET_M, dtype=float)
    phases = {
        "Conventional": p._conventional_raw_phase(ctx, "B1"),
        "FE": p._fe_raw_phase(ctx, "B1"),
    }
    sections = {
        method: p._pressure_sections(
            p._array_field(ctx, phase),
            target,
            half_xy_m=0.075,
            half_z_m=0.050,
            samples=161,
        )
        for method, phase in phases.items()
    }
    norm = shared_power_norm(
        [sections[method]["xy"] for method in _METHODS],
        lower=0.0,
        upper=99.5,
    )
    table = pd.DataFrame(
        [
            {
                "method": method,
                "component": "B1",
                "target_x_m": target[0],
                "target_y_m": target[1],
                "target_z_m": target[2],
                "plane": "xy",
                "half_width_m": 0.075,
                "samples_per_axis": 161,
                "pressure_norm_vmin_pa": float(norm.vmin),
                "pressure_norm_vmax_pa": float(norm.vmax),
            }
            for method in _METHODS
        ]
    )
    ctx.tables["fig1_candidate_representative_field_definition"] = table
    value = {"sections": sections, "norm": norm}
    ctx.memo[memo_key] = value
    return value


def _grid_values(
    summary: pd.DataFrame,
    method: str,
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = summary[summary["method"].eq(method)].copy()
    if selected.empty:
        raise RuntimeError(f"No domain records for {method}")
    x = np.sort(selected["xi"].unique().astype(float))
    y = np.sort(selected["eta"].unique().astype(float))
    values = (
        selected.pivot(index="eta", columns="xi", values=column)
        .reindex(index=y, columns=x)
        .to_numpy()
    )
    if values.shape != (len(y), len(x)) or not np.isfinite(values).all():
        raise RuntimeError(f"Incomplete {method} domain grid for {column}")
    return x, y, values


def _cell_extent(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0] - 0.5), float(values[0] + 0.5)
    step = float(np.median(np.diff(values)))
    return float(values[0] - 0.5 * step), float(values[-1] + 0.5 * step)


def _finite_linear_norm(values: np.ndarray) -> Normalize:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values are available for a linear color scale")
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + max(abs(low), 1.0) * np.finfo(float).eps
    return Normalize(vmin=low, vmax=high)


def _domain_metric_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    method: str,
    column: str,
    *,
    cmap: str,
    colorbar_label: str,
) -> mpl.image.AxesImage:
    x, y, values = _grid_values(summary, method, column)
    x0, x1 = _cell_extent(x)
    y0, y1 = _cell_extent(y)
    image = ax.imshow(
        values,
        origin="lower",
        extent=(x0, x1, y0, y1),
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=_finite_linear_norm(values),
    )
    ax.scatter(0.0, 0.0, marker="x", s=54, linewidth=1.7, color=TARGET_COLOR, zorder=20)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\eta$")
    ax.set_box_aspect(1)
    ax.grid(False)
    colorbar = add_adjacent_colorbar(
        image,
        ax,
        size="4.2%",
        pad="3.0%",
        label=colorbar_label,
        labelsize=8.5,
        ticksize=7.5,
    )
    if column == "gradient_rms_median":
        colorbar.formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        colorbar.formatter.set_powerlimits((-2, 2))
        colorbar.update_ticks()
    return image


def _exit_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    method: str,
) -> mpl.image.AxesImage:
    x, y, values = _grid_values(summary, method, "exit_code")
    x0, x1 = _cell_extent(x)
    y0, y1 = _cell_extent(y)
    cmap = ListedColormap([_EXIT_COLORS[mode] for mode in _EXIT_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(_EXIT_ORDER) + 0.5), cmap.N)
    image = ax.imshow(
        values,
        origin="lower",
        extent=(x0, x1, y0, y1),
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=norm,
    )
    ax.scatter(0.0, 0.0, marker="x", s=54, linewidth=1.7, color=TARGET_COLOR, zorder=20)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\eta$")
    ax.set_box_aspect(1)
    ax.grid(False)
    colorbar = add_adjacent_colorbar(
        image,
        ax,
        size="4.2%",
        pad="3.0%",
        labelsize=8.0,
        ticksize=7.1,
    )
    colorbar.set_ticks(np.arange(len(_EXIT_ORDER)))
    colorbar.set_ticklabels(_EXIT_TICK_LABELS)
    return image


def _field_panel(
    ax: plt.Axes,
    field_bank: Mapping[str, Any],
    method: str,
) -> mpl.image.AxesImage:
    section = field_bank["sections"][method]
    axis_mm = 1e3 * np.asarray(section["xy_axis_m"], dtype=float)
    image = field_panel(
        ax,
        section["xy"],
        (axis_mm[0], axis_mm[-1], axis_mm[0], axis_mm[-1]),
        norm=field_bank["norm"],
        xlabel=r"$x-x^*$ (mm)",
        ylabel=r"$y-y^*$ (mm)",
        target_mm=(0.0, 0.0),
    )
    ax.set_xlim(-75.0, 75.0)
    ax.set_ylim(-75.0, 75.0)
    ax.set_box_aspect(1)
    add_adjacent_colorbar(
        image,
        ax,
        size="4.2%",
        pad="3.0%",
        label=r"$|p|$ (Pa)",
        labelsize=8.5,
        ticksize=7.5,
    )
    return image


def _column_headers(
    fig: mpl.figure.Figure,
    specs: Sequence[mpl.gridspec.SubplotSpec],
) -> None:
    for spec, method in zip(specs[:2], _METHODS, strict=True):
        bounds = spec.get_position(fig)
        fig.text(
            0.5 * (bounds.x0 + bounds.x1),
            bounds.y1 + 0.042,
            method,
            ha="center",
            va="bottom",
            fontsize=12.0,
            fontweight="semibold",
        )


def _record_panel_registry(ctx: p.PipelineContext) -> None:
    if "fig1_candidate_panel_registry" in ctx.tables:
        return
    ctx.tables["fig1_candidate_panel_registry"] = pd.DataFrame(
        [
            {"candidate": "A", "row": 1, "content": "median terminal phase-gradient RMS over connected Single-target domain"},
            {"candidate": "A", "row": 2, "content": "median measured solve time over the same domain"},
            {"candidate": "B", "row": 1, "content": "median measured solve time over connected Single-target domain"},
            {"candidate": "B", "row": 2, "content": "representative broad +/-75 mm horizontal pressure field"},
            {"candidate": "C", "row": 1, "content": "median terminal phase-gradient RMS over connected Single-target domain"},
            {"candidate": "C", "row": 2, "content": "representative broad +/-75 mm horizontal pressure field"},
            {"candidate": "D", "row": 1, "content": "dominant deposited solver-exit class over connected Single-target domain"},
            {"candidate": "D", "row": 2, "content": "median measured solve time over the same domain"},
        ]
    )


def _new_figure() -> tuple[
    mpl.figure.Figure,
    mpl.gridspec.GridSpec,
    list[mpl.gridspec.SubplotSpec],
    list[plt.Axes],
]:
    fig = plt.figure(figsize=(11.6, 10.4))
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.34)
    specs = [grid[row, column] for row in range(2) for column in range(2)]
    axes = [fig.add_subplot(spec) for spec in specs]
    return fig, grid, specs, axes


def _finish(
    ctx: p.PipelineContext,
    fig: mpl.figure.Figure,
    specs: Sequence[mpl.gridspec.SubplotSpec],
    stem: str,
) -> mpl.figure.Figure:
    fig.subplots_adjust(left=0.105, right=0.945, top=0.900, bottom=0.080)
    _column_headers(fig, specs)
    outside_grid_panel_labels(fig, specs, tuple("abcd"), dx=0.014, dy=0.010)
    note = (
        "FULL manuscript numerics"
        if ctx.config.full
        else "PROTOTYPE SMOKE — values are not manuscript results"
    )
    fig.text(0.995, 0.004, note, ha="right", va="bottom", fontsize=6.7, color="0.38")
    return p._save_figure(ctx, fig, stem, tight=False)


def figure_1_candidate_a_domain_residual_time(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Candidate A: residual and time across the deposited 2-D domain."""

    stem = "Figure_1_candidate_A_domain_residual_time"
    if stem in ctx.figures:
        return ctx.figures[stem]
    summary = _domain_summary(ctx)
    _record_panel_registry(ctx)
    fig, _, specs, axes = _new_figure()
    for column, method in enumerate(_METHODS):
        _domain_metric_panel(
            axes[column], summary, method, "gradient_rms_median",
            cmap="magma", colorbar_label=r"$g_{\mathrm{RMS}}$",
        )
        _domain_metric_panel(
            axes[2 + column], summary, method, "wall_time_median_s",
            cmap="viridis", colorbar_label=r"$t$ (s)",
        )
    return _finish(ctx, fig, specs, stem)


def figure_1_candidate_b_domain_time_fields(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Candidate B: domain timing and representative broad pressure fields."""

    stem = "Figure_1_candidate_B_domain_time_fields"
    if stem in ctx.figures:
        return ctx.figures[stem]
    summary = _domain_summary(ctx)
    fields = _representative_fields(ctx)
    _record_panel_registry(ctx)
    fig, _, specs, axes = _new_figure()
    for column, method in enumerate(_METHODS):
        _domain_metric_panel(
            axes[column], summary, method, "wall_time_median_s",
            cmap="viridis", colorbar_label=r"$t$ (s)",
        )
        _field_panel(axes[2 + column], fields, method)
    return _finish(ctx, fig, specs, stem)


def figure_1_candidate_c_domain_residual_fields(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Candidate C: domain residual and representative broad pressure fields."""

    stem = "Figure_1_candidate_C_domain_residual_fields"
    if stem in ctx.figures:
        return ctx.figures[stem]
    summary = _domain_summary(ctx)
    fields = _representative_fields(ctx)
    _record_panel_registry(ctx)
    fig, _, specs, axes = _new_figure()
    for column, method in enumerate(_METHODS):
        _domain_metric_panel(
            axes[column], summary, method, "gradient_rms_median",
            cmap="magma", colorbar_label=r"$g_{\mathrm{RMS}}$",
        )
        _field_panel(axes[2 + column], fields, method)
    return _finish(ctx, fig, specs, stem)


def figure_1_candidate_d_domain_exit_time(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Candidate D: solver-exit taxonomy and time across the same domain."""

    stem = "Figure_1_candidate_D_domain_exit_time"
    if stem in ctx.figures:
        return ctx.figures[stem]
    summary = _domain_summary(ctx)
    _record_panel_registry(ctx)
    fig, _, specs, axes = _new_figure()
    for column, method in enumerate(_METHODS):
        _exit_panel(axes[column], summary, method)
        _domain_metric_panel(
            axes[2 + column], summary, method, "wall_time_median_s",
            cmap="viridis", colorbar_label=r"$t$ (s)",
        )
    return _finish(ctx, fig, specs, stem)


def figure_1_candidates(ctx: p.PipelineContext) -> dict[str, mpl.figure.Figure]:
    """Render all four independent Figure 1 alternatives."""

    return {
        "A": figure_1_candidate_a_domain_residual_time(ctx),
        "B": figure_1_candidate_b_domain_time_fields(ctx),
        "C": figure_1_candidate_c_domain_residual_fields(ctx),
        "D": figure_1_candidate_d_domain_exit_time(ctx),
    }


__all__ = [
    "figure_1_candidate_a_domain_residual_time",
    "figure_1_candidate_b_domain_time_fields",
    "figure_1_candidate_c_domain_residual_fields",
    "figure_1_candidate_d_domain_exit_time",
    "figure_1_candidates",
]
