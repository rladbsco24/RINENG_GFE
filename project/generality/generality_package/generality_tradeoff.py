"""Preserve unresolved mechanical outcomes in the existing Figure 8 comparison.

This reporting adapter does not run solvers or change physical validation. It
keeps the prescribed Single representative and all three Triple targets. The
native figure is unchanged when every required metric is available; otherwise
unavailable outcomes are shown outside the numerical ordinate with counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.text import Annotation

from hat_revision_pipeline import fig8_methods_only as native


_NATIVE_RENDER = native.render_methods_only_tradeoff
METRIC_COLUMNS = (
    "single_displacement_a",
    "single_pressure_abs_at_exact_equilibrium_pa",
    "triple_displacement_a",
    "triple_pressure_abs_at_exact_equilibrium_pa",
)


def _finite_values(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def _median(series: pd.Series) -> float:
    values = _finite_values(series)
    return float(values.median()) if len(values) else np.nan


def _range_record(series: pd.Series, median: str, low: str, high: str) -> dict:
    values = _finite_values(series)
    return {
        median: float(values.median()) if len(values) else np.nan,
        low: float(values.min()) if len(values) else np.nan,
        high: float(values.max()) if len(values) else np.nan,
    }


def _sources(series: pd.Series) -> str:
    return " | ".join(sorted({str(value) for value in series.dropna() if str(value)}))


def methods_only_tradeoff_table(ctx, benchmark, triple_benchmark, command_time):
    """Aggregate available endpoints without discarding failed denominators."""
    concentration = benchmark.get("concentration_table")
    if concentration is None:
        raise RuntimeError("Figure 8 requires the existing concentration benchmark table")
    rows = []
    for spec, result, concentration_row in zip(
        benchmark["specs"], benchmark["results"],
        concentration.reset_index(drop=True).itertuples(index=False), strict=True,
    ):
        method = str(spec["method"])
        if method not in native.METHODS:
            continue
        if str(concentration_row.method) != method:
            raise RuntimeError("Figure 8 benchmark/spec ordering is inconsistent")
        elapsed, source = native._measured_command_time(ctx, spec, command_time)
        found = bool(result.equilibrium.numerical_root_found)
        eigenvalues = np.asarray(result.symmetric_stiffness_eigenvalues_n_m, dtype=float)
        stiffness = (
            float(eigenvalues.min()) if found and eigenvalues.shape == (3,)
            and np.isfinite(eigenvalues).all() else np.nan
        )
        rows.append(dict(
            method=method, time_s=elapsed, root_found=found,
            displacement_a=float(result.equilibrium.displacement_norm_a) if found else np.nan,
            stiffness_min_n_m=stiffness,
            pressure_abs_at_exact_equilibrium_pa=(
                float(concentration_row.pressure_abs_at_exact_equilibrium_pa)
                if found else np.nan
            ),
            time_source=source, phase_source=str(spec.get("phase_source", "")),
            terminal_iteration=int(spec.get("terminal_iteration", 0)),
            phase_content_id=str(getattr(concentration_row, "phase_content_id", "")),
            pair_id=str(spec.get("pair_id", "")), seed=spec.get("seed", np.nan),
        ))
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("The matched exact-force benchmark produced no Figure 8 rows")
    ctx.tables["fig8_single_endpoint_population"] = raw.copy()
    missing = [method for method in native.METHODS if method not in set(raw.method)]
    if missing:
        raise RuntimeError(f"Figure 8 is missing locked methods: {missing}")
    triple = triple_benchmark.get("table")
    if triple is None:
        raise RuntimeError("Figure 8 requires the existing Triple exact benchmark table")
    triple = triple.reset_index(drop=True)
    aggregates = []
    for method in native.METHODS:
        population = raw[raw.method.eq(method)]
        single = population.sort_values("seed", kind="stable", na_position="last").iloc[:1]
        resolved_single = single[single.root_found]
        group = triple[triple.method.eq(method)].copy()
        indices = set(group["target_index"].astype(int))
        if len(group) != 3 or indices != {0, 1, 2}:
            raise RuntimeError(
                f"Figure 8 Triple benchmark requires T1-T3 once for {method}; "
                f"found rows={len(group)}, target_indices={sorted(indices)}"
            )
        root_mask = group.finite_ka_root_found.fillna(False).astype(bool)
        restoring_mask = root_mask & group.finite_ka_locally_restoring.fillna(False).astype(bool)
        resolved_triple = group[restoring_mask]
        record = dict(
            method=method,
            single_time_s=float(single.time_s.median()),
            single_displacement_a=_median(resolved_single.displacement_a),
            single_stiffness_min_n_m=_median(resolved_single.stiffness_min_n_m),
            single_pressure_abs_at_exact_equilibrium_pa=_median(
                resolved_single.pressure_abs_at_exact_equilibrium_pa),
            single_resolved_count=int(len(resolved_single)),
            single_root_found_count=int(len(resolved_single)),
            single_endpoint_count=1,
            single_population_count=int(len(population)),
            single_unresolved_count=1-int(len(resolved_single)),
            single_phase_content_id=str(single.iloc[0].phase_content_id),
            single_representative_selection="smallest prescribed seed; independent of result",
            single_root_found=bool(len(resolved_single)),
            single_time_source=_sources(single.time_source),
            single_phase_source=_sources(single.phase_source),
            single_displacement_valid_count=len(_finite_values(resolved_single.displacement_a)),
            single_pressure_valid_count=len(_finite_values(resolved_single.pressure_abs_at_exact_equilibrium_pa)),
            single_stiffness_valid_count=len(_finite_values(resolved_single.stiffness_min_n_m)),
            triple_time_s=float(group.command_time_s.median()),
            triple_stiffness_min_n_m=_median(resolved_triple.finite_ka_stiffness_min_n_m),
            triple_resolved_count=int(restoring_mask.sum()),
            triple_endpoint_count=3,
            triple_root_found_count=int(root_mask.sum()),
            triple_nonrestoring_root_count=int((root_mask & ~restoring_mask).sum()),
            triple_unresolved_count=3-int(root_mask.sum()),
            triple_unavailable_count=3-int(restoring_mask.sum()),
            triple_time_source="command_time_s",
            triple_phase_source=_sources(group.phase_source),
            triple_displacement_valid_count=len(_finite_values(resolved_triple.finite_ka_displacement_a)),
            triple_pressure_valid_count=len(_finite_values(resolved_triple.pressure_abs_at_exact_equilibrium_pa)),
            triple_stiffness_valid_count=len(_finite_values(resolved_triple.finite_ka_stiffness_min_n_m)),
            aggregation=(
                "Single fixed prescribed endpoint; Triple median/min-max over available "
                "locally restoring roots, with resolved and expected target counts retained"
            ),
            single_resolution_rule="numerical root found",
            triple_resolution_rule="numerical root found and locally restoring",
        )
        record.update(_range_record(
            resolved_triple.finite_ka_displacement_a,
            "triple_displacement_a", "triple_displacement_min_a", "triple_displacement_max_a"))
        record.update(_range_record(
            resolved_triple.pressure_abs_at_exact_equilibrium_pa,
            "triple_pressure_abs_at_exact_equilibrium_pa", "triple_pressure_min_pa", "triple_pressure_max_pa"))
        aggregates.append(record)
    table = pd.DataFrame(aggregates)
    if table.method.tolist() != list(native.METHODS):
        raise AssertionError("Figure 8 must retain every native method exactly once")
    return table


def _count_line(table, task: str) -> str:
    qualifier = "roots" if task == "single" else "locally restoring roots"
    values = "  |  ".join(
        f"{row.method} {getattr(row, task + '_resolved_count')}/{getattr(row, task + '_endpoint_count')}"
        for row in table.itertuples(index=False)
    )
    return f"{task.capitalize()} {qualifier}: {values}"


def render_methods_only_tradeoff(ctx, table, *, save_figure):
    """Move unavailable points off the numerical ordinate and show denominators."""
    complete = (
        np.isfinite(table[list(METRIC_COLUMNS)].to_numpy(float)).all()
        and (table.single_resolved_count == table.single_endpoint_count).all()
        and (table.triple_resolved_count == table.triple_endpoint_count).all()
    )
    if complete:
        return _NATIVE_RENDER(ctx, table, save_figure=save_figure)
    fig = _NATIVE_RENDER(ctx, table, save_figure=lambda _ctx, figure, _stem: figure)
    # Keep the existing four panels and command-time coordinates. A marker
    # above an axis has an axes-relative y coordinate, never a fabricated metric.
    for axis, column in zip(fig.axes, METRIC_COLUMNS, strict=True):
        scatters = [artist for artist in axis.collections if isinstance(artist, PathCollection)]
        if len(scatters) != len(table):
            raise RuntimeError("Native Figure 8 marker inventory changed")
        for scatter, value in zip(scatters, table[column].to_numpy(float), strict=True):
            if not np.isfinite(value):
                x = float(scatter.get_offsets()[0, 0])
                scatter.set_offsets(np.array([[x, 1.045]]))
                scatter.set_offset_transform(axis.get_xaxis_transform())
                scatter.set_clip_on(False)
        for artist in list(axis.texts):
            if isinstance(artist, Annotation) and artist.get_text() == "FE, FE+g":
                artist.remove()
        if not np.isfinite(table[column].to_numpy(float)).any():
            axis.set_yticks([])
            axis.text(.5, .5, "No available equilibria", transform=axis.transAxes,
                      ha="center", va="center", color="0.35", fontsize=11)
    fig.set_size_inches(10.8, 11.2, forward=True)
    fig.subplots_adjust(bottom=.20)
    fig.text(.5, .077, _count_line(table, "single"), ha="center", fontsize=8.8)
    fig.text(.5, .054, _count_line(table, "triple"), ha="center", fontsize=8.8)
    fig.text(.5, .026,
             "Markers above axes: unavailable. Triple medians and ranges use the reported available roots.",
             ha="center", fontsize=8.6, color="0.35")
    return save_figure(ctx, fig, native.FIGURE_STEM)


def install(ctx=None) -> None:
    """Install reporting replacements; the mechanical and timing code is intact."""
    native.methods_only_tradeoff_table = methods_only_tradeoff_table
    native.render_methods_only_tradeoff = render_methods_only_tradeoff
