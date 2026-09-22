"""Figure 8: Single/Triple displacement and equilibrium pressure against time.

The figure retains one aggregate point per method while keeping the two task
classes explicit. Single displacement and pressure occupy separate panels. Triple placement
and pressure suppression use medians with min--max bars over the same three
targets. All four panels use one pooled broken-linear command-time mapping.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import pipeline as p
from .figure_presentation import outside_grid_panel_labels, terse_math_label
from .style import COLORS


FIGURE_STEM = "Figure_8_performance_time_tradeoff"
METHODS = ("Conventional", "FE", "FE+g", "IB", "GS", "AD")
FAST_TIME_CUTOFF_S = 1.0
METHOD_MARKERS = {
    "FE+g": "P",
    "Conventional": "o",
    "FE": "s",
    "IB": "^",
    "GS": "v",
    "AD": "D",
}
METHOD_MARKER_SIZES = {
    "FE+g": 135,
    "Conventional": 92,
    "FE": 92,
    "IB": 160,
    "GS": 135,
    "AD": 68,
}
HOLLOW_METHODS = {"IB", "GS"}


def _timing_total_s(value: Any) -> float | None:
    """Return a measured ``BenchmarkRun`` time without inferring one."""

    if value is None:
        return None
    timing = value.get("timing") if isinstance(value, Mapping) else getattr(value, "timing", None)
    if timing is None:
        return None
    total = timing.get("total_s") if isinstance(timing, Mapping) else getattr(timing, "total_s", None)
    if total is None:
        return None
    elapsed = float(total)
    return elapsed if np.isfinite(elapsed) and elapsed >= 0.0 else None


def _measured_command_time(
    ctx: p.PipelineContext,
    spec: Mapping[str, Any],
    command_time: Callable[
        [p.PipelineContext, Mapping[str, Any]], tuple[float, str]
    ],
) -> tuple[float, str]:
    """Read the authoritative measured command time for one endpoint spec."""

    method = str(spec["method"])
    for key in (
        "command_run",
        "benchmark_run",
        "synthesis_run",
        "ad_run",
        "run",
        "command",
    ):
        elapsed = _timing_total_s(spec.get(key))
        if elapsed is not None:
            return elapsed, f"{key}.timing.total_s"
    for key in (
        "command_time_s",
        "measured_command_time_s",
        "runtime_s",
        "wall_time_s",
    ):
        if key in spec:
            elapsed = float(spec[key])
            if np.isfinite(elapsed) and elapsed >= 0.0:
                return elapsed, key
    if method == "AD":
        raise RuntimeError(
            "AD benchmark spec must retain its BenchmarkRun so Figure 8 can use "
            "BenchmarkRun.timing.total_s (including JIT and device sync)"
        )
    elapsed, source = command_time(ctx, spec)
    elapsed = float(elapsed)
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError(f"Invalid measured command time for {method}: {elapsed}")
    return elapsed, str(source)


def methods_only_tradeoff_table(
    ctx: p.PipelineContext,
    benchmark: Mapping[str, Any],
    triple_benchmark: Mapping[str, Any],
    command_time: Callable[
        [p.PipelineContext, Mapping[str, Any]], tuple[float, str]
    ],
) -> pd.DataFrame:
    """Aggregate like-for-like Single and Triple exact-evaluation records."""

    concentration = benchmark.get("concentration_table")
    if concentration is None:
        raise RuntimeError("Figure 8 requires the existing concentration benchmark table")
    concentration = concentration.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for spec, result, concentration_row in zip(
        benchmark["specs"], benchmark["results"], concentration.itertuples(index=False), strict=True
    ):
        method = str(spec["method"])
        if method not in METHODS:
            continue
        if str(concentration_row.method) != method:
            raise RuntimeError("Figure 8 benchmark/spec ordering is inconsistent")
        elapsed_s, time_source = _measured_command_time(
            ctx, spec, command_time
        )
        root_found = bool(result.equilibrium.numerical_root_found)
        eigenvalues = np.asarray(
            result.symmetric_stiffness_eigenvalues_n_m, dtype=float
        )
        stiffness_min_n_m = (
            float(np.min(eigenvalues))
            if root_found
            and eigenvalues.shape == (3,)
            and np.all(np.isfinite(eigenvalues))
            else np.nan
        )
        rows.append(
            {
                "method": method,
                "time_s": elapsed_s,
                "root_found": root_found,
                "displacement_a": (
                    float(result.equilibrium.displacement_norm_a)
                    if root_found
                    else np.nan
                ),
                "stiffness_min_n_m": stiffness_min_n_m,
                "pressure_abs_at_exact_equilibrium_pa": float(
                    concentration_row.pressure_abs_at_exact_equilibrium_pa
                ),
                "time_source": time_source,
                "phase_source": str(spec.get("phase_source", "")),
                "terminal_iteration": int(spec.get("terminal_iteration", 0)),
                "phase_fingerprint": str(getattr(concentration_row, "phase_fingerprint", "")),
                "pair_id": str(spec.get("pair_id", "")),
                "seed": spec.get("seed", np.nan),
            }
        )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("The matched exact-force benchmark produced no Figure 8 rows")
    # Retain every population row for writing; use the first prescribed seed
    # as the fixed displayed representative in both smoke and expanded runs.
    # Selection precedes and is independent of mechanical outcomes.
    ctx.tables["fig8_single_endpoint_population"] = raw.copy()

    present = set(raw["method"])
    missing = [method for method in METHODS if method not in present]
    if missing:
        raise RuntimeError(f"Figure 8 is missing locked methods: {missing}")

    aggregates: list[dict[str, Any]] = []
    for method in METHODS:
        group = raw[raw["method"].eq(method)].copy()
        population_count = len(group)
        group = group.sort_values("seed", kind="stable", na_position="last").iloc[:1]
        resolved = group[group["root_found"]]
        if len(resolved) != 1:
            raise RuntimeError(
                f"Figure 8 Single benchmark requires a resolved root for {method}"
            )
        aggregates.append(
            {
                "method": method,
                "single_time_s": float(group["time_s"].median()),
                "single_displacement_a": (
                    float(resolved["displacement_a"].median())
                    if len(resolved)
                    else np.nan
                ),
                "single_stiffness_min_n_m": (
                    float(resolved["stiffness_min_n_m"].median())
                    if len(resolved["stiffness_min_n_m"].dropna())
                    else np.nan
                ),
                "single_pressure_abs_at_exact_equilibrium_pa": (
                    float(resolved["pressure_abs_at_exact_equilibrium_pa"].median())
                    if len(resolved["pressure_abs_at_exact_equilibrium_pa"].dropna())
                    else np.nan
                ),
                "single_resolved_count": int(group["root_found"].sum()),
                "single_endpoint_count": int(len(group)),
                "single_population_count": int(population_count),
                "single_phase_fingerprint": str(group.iloc[0]["phase_fingerprint"]),
                "single_representative_selection": "smallest prescribed seed; independent of result",
                "single_root_found": bool(len(resolved)),
                "single_time_source": " | ".join(sorted(set(group["time_source"]))),
                "single_phase_source": " | ".join(
                    source
                    for source in sorted(set(group["phase_source"]))
                    if source
                ),
            }
        )
    triple = triple_benchmark.get("table")
    if triple is None:
        raise RuntimeError("Figure 8 requires the existing Triple exact benchmark table")
    triple = triple.reset_index(drop=True)
    triple_records: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        group = triple[triple["method"].eq(method)].copy()
        if group.empty:
            raise RuntimeError(f"Figure 8 Triple benchmark is missing {method}")
        target_indices = set(group["target_index"].astype(int))
        if len(group) != 3 or target_indices != {0, 1, 2}:
            raise RuntimeError(
                f"Figure 8 Triple benchmark requires T1-T3 once for {method}; "
                f"found rows={len(group)}, target_indices={sorted(target_indices)}"
            )
        resolved = group[
            group["finite_ka_root_found"].astype(bool)
            & group["finite_ka_locally_restoring"].astype(bool)
        ].copy()
        if len(resolved) != 3:
            raise RuntimeError(
                f"Figure 8 Triple summary requires 3/3 resolved, locally "
                f"restoring roots for {method}; "
                f"found {len(resolved)}/3"
            )
        displacements = resolved["finite_ka_displacement_a"].dropna()
        pressures = resolved["pressure_abs_at_exact_equilibrium_pa"].dropna()
        stiffness = resolved["finite_ka_stiffness_min_n_m"].dropna()
        triple_records[method] = {
            "triple_time_s": float(group["command_time_s"].median()),
            "triple_displacement_a": (
                float(displacements.median()) if len(displacements) else np.nan
            ),
            "triple_displacement_min_a": (
                float(displacements.min()) if len(displacements) else np.nan
            ),
            "triple_displacement_max_a": (
                float(displacements.max()) if len(displacements) else np.nan
            ),
            "triple_stiffness_min_n_m": (
                float(stiffness.median()) if len(stiffness) else np.nan
            ),
            "triple_pressure_abs_at_exact_equilibrium_pa": (
                float(pressures.median()) if len(pressures) else np.nan
            ),
            "triple_pressure_min_pa": (
                float(pressures.min()) if len(pressures) else np.nan
            ),
            "triple_pressure_max_pa": (
                float(pressures.max()) if len(pressures) else np.nan
            ),
            "triple_resolved_count": int(len(resolved)),
            "triple_endpoint_count": int(len(group)),
            "triple_time_source": "command_time_s",
            "triple_phase_source": " | ".join(
                source
                for source in sorted(set(group["phase_source"].astype(str)))
                if source
            ),
        }
    for record in aggregates:
        record.update(triple_records[str(record["method"])])
        record["aggregation"] = (
            "Single endpoint; Triple median and min-max over three targets"
        )
    table = pd.DataFrame(aggregates)
    if table["method"].tolist() != list(METHODS) or len(table) != len(METHODS):
        raise AssertionError("Figure 8 must contain one aggregate row per locked method")
    return table


def _format_time_tick(seconds: float) -> str:
    if seconds < 1.0:
        return f"{1.0e3 * seconds:.1f}\nms"
    return f"{seconds:.1f}\ns"


def _broken_linear_time_coordinates(
    table: pd.DataFrame,
    *,
    time_column: str = "time_s",
) -> tuple[dict[str, float], np.ndarray, list[str], float | None]:
    """Map two linear time regimes onto one readable, non-logarithmic axis."""

    fast = table[table[time_column] < FAST_TIME_CUTOFF_S]
    slow = table[table[time_column] >= FAST_TIME_CUTOFF_S]
    coordinates: dict[str, float] = {}
    tick_pairs: list[tuple[float, float]] = []

    if fast.empty or slow.empty:
        values = table[time_column].to_numpy(float)
        low, high = float(values.min()), float(values.max())
        span = max(high - low, 1.0e-12)
        for row in table.itertuples(index=False):
            value = float(getattr(row, time_column))
            coordinates[row.method] = 0.08 + 0.84 * (value - low) / span
        tick_values = np.linspace(low, high, min(4, len(values)))
        for value in tick_values:
            tick_pairs.append((0.08 + 0.84 * (float(value) - low) / span, float(value)))
        break_x: float | None = None
    else:
        fast_low = float(fast[time_column].min())
        fast_high = float(fast[time_column].max())
        slow_low = float(slow[time_column].min())
        slow_high = float(slow[time_column].max())

        def scale(value: float, low: float, high: float, left: float, right: float) -> float:
            if np.isclose(high, low):
                return 0.5 * (left + right)
            return left + (right - left) * (value - low) / (high - low)

        for row in fast.itertuples(index=False):
            value = float(getattr(row, time_column))
            coordinates[row.method] = scale(
                value, fast_low, fast_high, 0.07, 0.32
            )
        for row in slow.itertuples(index=False):
            value = float(getattr(row, time_column))
            coordinates[row.method] = scale(
                value, slow_low, slow_high, 0.53, 0.94
            )
        tick_pairs.extend(
            (scale(value, fast_low, fast_high, 0.07, 0.32), value)
            for value in sorted(set((fast_low, fast_high)))
        )
        tick_pairs.extend(
            (scale(value, slow_low, slow_high, 0.53, 0.94), value)
            for value in sorted(set((slow_low, slow_high)))
        )
        break_x = 0.425

    tick_positions = np.asarray([pair[0] for pair in tick_pairs], dtype=float)
    tick_labels = [_format_time_tick(pair[1]) for pair in tick_pairs]
    return coordinates, tick_positions, tick_labels, break_x


def _draw_axis_break(ax: mpl.axes.Axes, x_position: float) -> None:
    for offset in (-0.014, 0.014):
        ax.plot(
            [x_position + offset - 0.010, x_position + offset + 0.010],
            [-0.010, 0.010],
            transform=ax.get_xaxis_transform(),
            color="0.20",
            lw=1.05,
            clip_on=False,
        )


def _metric_limits(values: np.ndarray, *, include_zero: bool) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return (0.0, 1.0)
    low, high = float(finite.min()), float(finite.max())
    if include_zero:
        low = min(0.0, low)
        high = max(0.0, high)
    span = max(high - low, 0.12 * max(abs(low), abs(high), 1.0e-12))
    return low - 0.08 * span, high + 0.16 * span


def render_methods_only_tradeoff(
    ctx: p.PipelineContext,
    table: pd.DataFrame,
    *,
    save_figure: Callable[
        [p.PipelineContext, mpl.figure.Figure, str], mpl.figure.Figure
    ],
) -> mpl.figure.Figure:
    """Render Single and Triple exact metrics without conflating task classes."""

    if table["method"].tolist() != list(METHODS) or len(table) != len(METHODS):
        raise ValueError("Figure 8 rendering requires exactly six method aggregates")
    fig = plt.figure(figsize=(10.8, 10.2))
    grid = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.48)
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    )
    colors = {
        "Conventional": COLORS["Conventional"],
        "FE": COLORS["FE"],
        "FE+g": COLORS["FE+g"],
        "IB": COLORS["IB"],
        "GS": COLORS["GS-PO audit"],
        "AD": COLORS["AD"],
    }
    metrics = (
        (
            "single_displacement_a",
            "Single",
            r"Single $\|\Delta\mathbf{r}\|/a$",
            None,
        ),
        (
            "single_pressure_abs_at_exact_equilibrium_pa",
            "Single",
            r"Single $|p(\mathbf{r}_{\mathrm{eq}})|$ (Pa)",
            None,
        ),
        (
            "triple_displacement_a",
            "Triple",
            r"Triple $\|\Delta\mathbf{r}\|/a$",
            ("triple_displacement_min_a", "triple_displacement_max_a"),
        ),
        (
            "triple_pressure_abs_at_exact_equilibrium_pa",
            "Triple",
            r"Triple $|p(\mathbf{r}_{\mathrm{eq}})|$ (Pa)",
            ("triple_pressure_min_pa", "triple_pressure_max_pa"),
        ),
    )

    pooled_single = table[["method", "single_time_s"]].rename(
        columns={"single_time_s": "pooled_time_s"}
    )
    pooled_single["method"] = "Single::" + pooled_single["method"].astype(str)
    pooled_triple = table[["method", "triple_time_s"]].rename(
        columns={"triple_time_s": "pooled_time_s"}
    )
    pooled_triple["method"] = "Triple::" + pooled_triple["method"].astype(str)
    pooled_time = pd.concat((pooled_single, pooled_triple), ignore_index=True)
    x_by_endpoint, ticks, tick_labels, break_x = _broken_linear_time_coordinates(
        pooled_time, time_column="pooled_time_s"
    )
    for ax, (column, task, ylabel, range_columns) in zip(
        axes, metrics, strict=True
    ):
        raw_values = table[column].to_numpy(float)
        range_low = (
            table[range_columns[0]].to_numpy(float)
            if range_columns is not None else raw_values
        )
        range_high = (
            table[range_columns[1]].to_numpy(float)
            if range_columns is not None else raw_values
        )
        values = raw_values
        lower, upper = _metric_limits(
            np.r_[raw_values, range_low, range_high], include_zero=True
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(lower, upper)
        ax.set_xticks(ticks, tick_labels, rotation=0)
        ax.set_xlabel(r"$t_{\mathrm{cmd}}$ (broken linear axis)")
        ax.set_ylabel(ylabel)
        ax.set_title("")
        ax.grid(True, axis="y", alpha=0.24)
        ax.grid(False, axis="x")
        ax.axhline(0.0, color="0.45", lw=0.7, zorder=1)
        if break_x is not None:
            _draw_axis_break(ax, break_x)

        unresolved_y = upper - 0.07 * (upper - lower)
        for index, (row, value) in enumerate(
            zip(table.itertuples(index=False), values, strict=True)
        ):
            x_value = x_by_endpoint[f"{task}::{row.method}"]
            finite = np.isfinite(raw_values[index])
            hollow = row.method in HOLLOW_METHODS
            if finite and range_columns is not None:
                low_error = max(0.0, float(raw_values[index] - range_low[index]))
                high_error = max(0.0, float(range_high[index] - raw_values[index]))
                ax.errorbar(
                    [x_value], [float(value)],
                    yerr=np.asarray([[low_error], [high_error]]),
                    fmt="none", ecolor=colors[row.method], elinewidth=1.55,
                    capsize=3.2, capthick=1.4, alpha=1.0, zorder=4,
                )
            ax.scatter(
                x_value,
                float(value) if finite else unresolved_y,
                s=METHOD_MARKER_SIZES[row.method],
                marker=METHOD_MARKERS[row.method] if finite else "^",
                facecolor=(
                    colors[row.method] if finite and not hollow else "none"
                ),
                edgecolor=(
                    colors[row.method] if hollow or not finite else "white"
                ),
                linewidth=2.0 if hollow or not finite else 1.1,
                zorder=6 if row.method == "AD" else 5,
            )

    fe_single = table[table.method.eq("FE")].iloc[0]
    for axis, value in ((axes[0], fe_single.single_displacement_a),
                        (axes[1], fe_single.single_pressure_abs_at_exact_equilibrium_pa)):
        axis.annotate("FE, FE+g", (x_by_endpoint["Single::FE"], value),
            xytext=(10, 12), textcoords="offset points", ha="left", va="bottom",
            fontsize=10.5, color="#444444", arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8))
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            linestyle="none",
            markerfacecolor=("none" if method in HOLLOW_METHODS else colors[method]),
            markeredgecolor=(colors[method] if method in HOLLOW_METHODS else "white"),
            markersize=9.5,
            markeredgewidth=1.8 if method in HOLLOW_METHODS else 1.1,
            label=method,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        labels=list(METHODS),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.12, right=0.975, top=0.85, bottom=0.14)
    outside_grid_panel_labels(
        fig,
        (grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]),
        ("a", "b", "c", "d"),
        dx=0.016,
        dy=0.008,
    )
    return save_figure(ctx, fig, FIGURE_STEM)


def figure_8_methods_only(ctx: p.PipelineContext) -> mpl.figure.Figure:
    """Build and render the locked methods-only Figure 8."""

    if FIGURE_STEM in ctx.figures:
        return ctx.figures[FIGURE_STEM]
    from .final_figures import (
        _command_time,
        _exact_benchmark_with_concentration,
        _triple_exact_benchmark_with_concentration,
        _save,
    )

    benchmark = _exact_benchmark_with_concentration(ctx)
    triple_benchmark = _triple_exact_benchmark_with_concentration(ctx)
    table = methods_only_tradeoff_table(
        ctx, benchmark, triple_benchmark, _command_time
    )
    ctx.tables["fig8_performance_time_tradeoff"] = table
    return render_methods_only_tradeoff(ctx, table, save_figure=_save)


__all__ = [
    "FIGURE_STEM",
    "FAST_TIME_CUTOFF_S",
    "METHODS",
    "figure_8_methods_only",
    "methods_only_tradeoff_table",
    "render_methods_only_tradeoff",
]
