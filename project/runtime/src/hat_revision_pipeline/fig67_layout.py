"""Bounded Figure 6/7 layout helpers.

This module does not change either figure's scientific objects.  It only
provides the compact Figure 6 labels and the wider Figure 7 XZ rendering
domain requested for the revision.  The wide domain is copied from the
authoritative HAT reproduction notebook: x/y in [-0.075, 0.075] m and z in
[0, 0.10] m.  For the canonical target z*=0.050 m, the corresponding
target-centred XZ limits are [-75, 75] x [-50, 50] mm.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib as mpl
import numpy as np

from . import pipeline as p
from .cache import digest_array
from .exact_validator import ForceSection, sample_force_section
from .style import field_panel, force_streamlines, shared_power_norm


# Authoritative render extent in HAT_main_figures_reproduction(4).ipynb:
# XY_EXTENT = 0.075 m and Z_EXTENT = 0.10 m.
BASELINE_XY_HALF_WIDTH_M = 0.075
BASELINE_Z_HALF_WIDTH_M = 0.050
BASELINE_XZ_EXTENT_MM = (-75.0, 75.0, -50.0, 50.0)
BASELINE_PRESSURE_SAMPLES = 161


def compact_force_concentration_labels(ax: mpl.axes.Axes) -> None:
    """Apply only the requested abbreviated Figure 6 axis labels."""

    ax.set_xlabel(r"$C_{\mathrm{ROI}}$")
    ax.set_ylabel(r"Exact $\kappa_{\min}$ (mN m$^{-1}$)")
    ax.set_title("")


def add_dedicated_vertical_colorbar(
    fig: mpl.figure.Figure,
    mappable: mpl.cm.ScalarMappable,
    cax: mpl.axes.Axes,
    *,
    title: str,
) -> mpl.colorbar.Colorbar:
    """Put a colorbar in its own GridSpec column with a short top label.

    A caller must allocate ``cax``; this prevents the colorbar from being
    inserted between a data axis and that axis's label.
    """

    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.ax.set_title(title, fontsize=8.2, pad=4.0)
    colorbar.ax.tick_params(labelsize=7.5)
    return colorbar


def baseline_wide_pressure_sections(
    ctx: p.PipelineContext,
    phase: np.ndarray,
    target_m: Sequence[float],
    *,
    samples: int = BASELINE_PRESSURE_SAMPLES,
) -> Mapping[str, Any]:
    """Return pressure sections on the exact pre-revision render domain."""

    return p._pressure_sections(
        p._array_field(ctx, np.asarray(phase, dtype=float)),
        np.asarray(target_m, dtype=float),
        half_xy_m=BASELINE_XY_HALF_WIDTH_M,
        half_z_m=BASELINE_Z_HALF_WIDTH_M,
        samples=int(samples),
    )


def baseline_wide_force_section_cached(
    ctx: p.PipelineContext,
    phase: np.ndarray,
    target_m: Sequence[float],
    *,
    key: str,
    points: int | None = None,
) -> ForceSection:
    """Sample the existing exact finite-ka force model on the same wide XZ ROI.

    ``points`` defaults to the already frozen run-mode force-grid size.  The
    helper deliberately does not invent a new smoke/full resolution.
    """

    phase = np.asarray(phase, dtype=float)
    target = np.asarray(target_m, dtype=float)
    count = int(ctx.config.force_slice_points if points is None else points)
    if count < 2:
        raise ValueError("force-section grid must contain at least two points per axis")
    payload = {
        "phase": digest_array(phase),
        "target": target.tolist(),
        "plane": "xz",
        "shape": (count, count),
        "half_width_m": (BASELINE_XY_HALF_WIDTH_M, BASELINE_Z_HALF_WIDTH_M),
        "exact_validator_protocol": p._finite_ka_protocol(ctx),
        "key": str(key),
    }
    value, _, _ = ctx.cache.get_or_compute(
        "finite_ka_force_section_wide_baseline",
        payload,
        lambda: sample_force_section(
            p._exact_evaluator_for_phase(ctx, phase),
            target,
            axes=(0, 2),
            half_width_m=(BASELINE_XY_HALF_WIDTH_M, BASELINE_Z_HALF_WIDTH_M),
            shape=(count, count),
        ),
        recompute=ctx.recompute,
    )
    return value


def plot_baseline_wide_force_card(
    ctx: p.PipelineContext,
    ax: mpl.axes.Axes,
    phase: np.ndarray,
    target_m: Sequence[float],
    equilibrium_m: Sequence[float],
    *,
    equilibrium_resolved: bool,
    key: str,
    dense: Mapping[str, Any] | None = None,
    pressure_norm: mpl.colors.Normalize | None = None,
    force_points: int | None = None,
    title: str = "",
) -> mpl.image.AxesImage:
    """Draw one exact-force XZ card on the authoritative wider domain.

    The open triangle retains the existing unresolved-candidate convention;
    no in-panel prose is added.  The returned pressure image can be attached
    to a caller-owned shared colorbar.
    """

    phase = np.asarray(phase, dtype=float)
    target = np.asarray(target_m, dtype=float)
    equilibrium = np.asarray(equilibrium_m, dtype=float)
    if dense is None:
        dense = baseline_wide_pressure_sections(ctx, phase, target)
    force = baseline_wide_force_section_cached(
        ctx, phase, target, key=key, points=force_points
    )
    amplitude = np.asarray(dense["xz"], dtype=float)
    norm = pressure_norm or shared_power_norm([amplitude], lower=0.0, upper=99.5)
    delta_mm = 1.0e3 * (equilibrium - target)
    equilibrium_xz = (float(delta_mm[0]), float(delta_mm[2]))
    image = field_panel(
        ax,
        amplitude,
        BASELINE_XZ_EXTENT_MM,
        norm=norm,
        xlabel=r"$x-x^*$ (mm)",
        ylabel=r"$z-z^*$ (mm)",
        target_mm=(0.0, 0.0),
        equilibrium_mm=equilibrium_xz if equilibrium_resolved else None,
    )
    if not equilibrium_resolved:
        ax.scatter(
            *equilibrium_xz,
            marker="^",
            facecolor="none",
            edgecolor="white",
            s=45,
            linewidth=1.1,
            zorder=25,
        )
    u_mm = 1.0e3 * force.coordinates_u_m[0]
    v_mm = 1.0e3 * force.coordinates_v_m[:, 0]
    vector = force.in_plane_total_force_n
    force_streamlines(
        ax, u_mm, v_mm, vector[..., 0], vector[..., 1], density=0.8
    )
    ax.set_xlim(BASELINE_XZ_EXTENT_MM[0], BASELINE_XZ_EXTENT_MM[1])
    ax.set_ylim(BASELINE_XZ_EXTENT_MM[2], BASELINE_XZ_EXTENT_MM[3])
    if title:
        ax.set_title(title, pad=5.0, fontweight="normal")
    else:
        ax.set_title("")
    return image


def alpha_table_facts(table: Any) -> dict[str, Any]:
    """Extract only directly measured Figure 7 facts from its exported table."""

    required = {
        "alpha_per_m",
        "iterations",
        "maxiter",
        "gradient_norm",
        "component",
        "component_margin",
        "finite_ka_root_found",
        "finite_ka_displacement_a",
        "target_gorkov_hessian_min_n_m",
        "root_exact_stiffness_min_n_m",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"alpha table is missing required columns: {missing}")
    resolved = table[table["finite_ka_root_found"].astype(bool)].copy()
    best = resolved.loc[resolved["finite_ka_displacement_a"].astype(float).idxmin()]
    capped = table[table["iterations"].astype(int) >= table["maxiter"].astype(int)]
    return {
        "best_resolved_alpha_per_m": float(best["alpha_per_m"]),
        "best_resolved_displacement_a": float(best["finite_ka_displacement_a"]),
        "cap_limited_alpha_per_m": tuple(capped["alpha_per_m"].astype(float)),
        "component_sequence": tuple(table.sort_values("alpha_per_m")["component"].astype(str)),
        "highest_alpha_root_found": bool(
            table.sort_values("alpha_per_m").iloc[-1]["finite_ka_root_found"]
        ),
        "highest_alpha_target_hessian_min_n_m": float(
            table.sort_values("alpha_per_m").iloc[-1]["target_gorkov_hessian_min_n_m"]
        ),
        "highest_alpha_root_stiffness_min_n_m": float(
            table.sort_values("alpha_per_m").iloc[-1]["root_exact_stiffness_min_n_m"]
        ),
    }


__all__ = [
    "BASELINE_PRESSURE_SAMPLES",
    "BASELINE_XY_HALF_WIDTH_M",
    "BASELINE_XZ_EXTENT_MM",
    "BASELINE_Z_HALF_WIDTH_M",
    "add_dedicated_vertical_colorbar",
    "alpha_table_facts",
    "baseline_wide_force_section_cached",
    "baseline_wide_pressure_sections",
    "compact_force_concentration_labels",
    "plot_baseline_wide_force_card",
]
