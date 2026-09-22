"""Figure 7: three broad horizontal pressure sections.

The fixed ``alpha = 10^3, 10^6, 10^7 m^-1`` scientific endpoints are rendered as
broad target-centred XY ``|p|`` fields and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import pipeline as p
from .cache import digest_array
from .fig67_layout import BASELINE_PRESSURE_SAMPLES, BASELINE_XY_HALF_WIDTH_M
from .figure_presentation import add_adjacent_colorbar, outside_grid_panel_labels


ALPHA_XY_DISPLAY: tuple[float, ...] = (1.0e3, 1.0e6, 1.0e7)
XY_EXTENT_MM: tuple[float, float, float, float] = tuple(
    1.0e3 * value
    for value in (
        -BASELINE_XY_HALF_WIDTH_M,
        BASELINE_XY_HALF_WIDTH_M,
        -BASELINE_XY_HALF_WIDTH_M,
        BASELINE_XY_HALF_WIDTH_M,
    )
)


@dataclass(frozen=True)
class AlphaXYSection:
    """One target-centred horizontal pressure section."""

    alpha_per_m: float
    axis_m: np.ndarray
    pressure_pa: np.ndarray


def _display_indices(
    alpha_values: Sequence[float],
    display: Sequence[float] = ALPHA_XY_DISPLAY,
) -> tuple[int, ...]:
    """Locate the three locked display endpoints in the scientific schedule."""

    values = np.asarray(alpha_values, dtype=float)
    indices: list[int] = []
    for requested in display:
        matches = np.flatnonzero(
            np.isclose(values, float(requested), rtol=0.0, atol=0.0)
        )
        if len(matches) != 1:
            raise ValueError(
                f"alpha={requested:g} must occur exactly once in the fixed schedule; "
                f"found {len(matches)}"
            )
        indices.append(int(matches[0]))
    return tuple(indices)


def _xy_pressure_section(
    ctx: p.PipelineContext,
    phase: np.ndarray,
    target_m: Sequence[float],
    *,
    alpha_per_m: float,
    samples: int = BASELINE_PRESSURE_SAMPLES,
) -> AlphaXYSection:
    """Evaluate only the requested broad XY pressure plane at ``z=z*``."""

    phase = np.asarray(phase, dtype=float)
    target = np.asarray(target_m, dtype=float)
    count = int(samples)
    if count < 2:
        raise ValueError("XY pressure section requires at least two samples per axis")
    payload = {
        "contract": "fig7-three-broad-xy-pressure-v2",
        "phase": digest_array(phase),
        "target_m": target.tolist(),
        "alpha_per_m": float(alpha_per_m),
        "half_width_m": float(BASELINE_XY_HALF_WIDTH_M),
        "samples": count,
    }

    def compute() -> AlphaXYSection:
        axis = np.linspace(
            -BASELINE_XY_HALF_WIDTH_M,
            BASELINE_XY_HALF_WIDTH_M,
            count,
        )
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        points = np.stack(
            (target[0] + xx, target[1] + yy, np.full_like(xx, target[2])),
            axis=-1,
        )
        pressure = p._pressure_abs_chunked(
            p._array_field(ctx, phase), points.reshape(-1, 3)
        ).reshape(xx.shape)
        return AlphaXYSection(
            alpha_per_m=float(alpha_per_m),
            axis_m=axis,
            pressure_pa=np.asarray(pressure, dtype=float),
        )

    value, _, _ = ctx.cache.get_or_compute(
        "fig7_alpha_xy_pressure_section",
        payload,
        compute,
        recompute=ctx.recompute,
    )
    return value


def alpha_xy_sections(
    ctx: p.PipelineContext,
    phases: np.ndarray,
    alpha_values: Sequence[float],
    target_m: Sequence[float],
    *,
    display: Sequence[float] = ALPHA_XY_DISPLAY,
    samples: int = BASELINE_PRESSURE_SAMPLES,
) -> tuple[AlphaXYSection, ...]:
    """Return exactly the three locked broad XY pressure sections."""

    phase_array = np.asarray(phases, dtype=float)
    indices = _display_indices(alpha_values, display)
    if len(phase_array) != len(alpha_values):
        raise ValueError("phase and alpha arrays must have the same leading length")
    if len(indices) != 3:
        raise ValueError("Figure 7 requires exactly three alpha endpoints")
    return tuple(
        _xy_pressure_section(
            ctx,
            phase_array[index],
            target_m,
            alpha_per_m=float(alpha_values[index]),
            samples=samples,
        )
        for index in indices
    )


def _alpha_math(alpha: float) -> str:
    if np.isclose(float(alpha), 9.0):
        return "9"
    exponent = int(round(np.log10(float(alpha))))
    return rf"10^{{{exponent}}}"


def _plot_xy_field(
    ax: mpl.axes.Axes,
    section: AlphaXYSection,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> mpl.image.AxesImage:
    pressure_kpa = np.asarray(section.pressure_pa, dtype=float) * 1.0e-3
    vmax = float(np.nanpercentile(pressure_kpa, 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        raise ValueError("Figure 7 pressure section has no positive finite scale")
    norm = mpl.colors.PowerNorm(gamma=0.70, vmin=0.0, vmax=vmax, clip=True)
    extent_mm = (
        1.0e3 * float(section.axis_m[0]),
        1.0e3 * float(section.axis_m[-1]),
        1.0e3 * float(section.axis_m[0]),
        1.0e3 * float(section.axis_m[-1]),
    )
    image = ax.imshow(
        pressure_kpa,
        origin="lower",
        extent=extent_mm,
        cmap="viridis",
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax.grid(False)
    ax.set_xlim(extent_mm[:2])
    ax.set_ylim(extent_mm[2:])
    ax.set_xlabel(r"$x-x^*$ (mm)" if show_xlabel else "")
    ax.set_ylabel(r"$y-y^*$ (mm)" if show_ylabel else "")
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    ax.set_title(
        rf"$\alpha={_alpha_math(section.alpha_per_m)}\ \mathrm{{m}}^{{-1}}$",
        fontsize=11.0,
        pad=3.0,
    )
    ticks = np.linspace(0.0, vmax, 3)
    colorbar = add_adjacent_colorbar(
        image,
        ax,
        size="3.0%",
        pad="2.0%",
        label=r"$|p|$ (kPa)",
        labelsize=8.0,
        ticksize=7.5,
        ticks=ticks,
    )
    colorbar.formatter = mpl.ticker.FuncFormatter(
        lambda value, _position: (
            "0" if np.isclose(value, 0.0) else f"{value:.2f}".rstrip("0").rstrip(".")
        )
    )
    colorbar.update_ticks()
    return image


def compose_figure_7_xy(
    ctx: p.PipelineContext,
    bank: Mapping[str, Any],
    alpha_values: Sequence[float],
    target_m: Sequence[float],
    *,
    samples: int = BASELINE_PRESSURE_SAMPLES,
) -> mpl.figure.Figure:
    """Compose three square XY fields with at most two panels across."""

    if "phases" not in bank:
        raise KeyError("Figure 7 bank must contain phases")
    sections = alpha_xy_sections(
        ctx,
        np.asarray(bank["phases"], dtype=float),
        alpha_values,
        target_m,
        samples=samples,
    )
    fig = plt.figure(figsize=(8.2, 8.0))
    grid = fig.add_gridspec(2, 4, wspace=0.80, hspace=0.52)
    field_specs = (grid[0, :2], grid[0, 2:], grid[1, 1:3])
    field_axes = tuple(fig.add_subplot(spec) for spec in field_specs)
    for index, (ax, section) in enumerate(zip(field_axes, sections, strict=True)):
        _plot_xy_field(
            ax,
            section,
            show_xlabel=True,
            show_ylabel=index != 1,
        )
        ax.set_box_aspect(1)

    fig.subplots_adjust(left=0.100, right=0.900, top=0.930, bottom=0.090)
    outside_grid_panel_labels(
        fig,
        field_specs,
        tuple("abc"),
        dx=0.012,
        dy=0.010,
    )
    return fig


__all__ = [
    "ALPHA_XY_DISPLAY",
    "AlphaXYSection",
    "XY_EXTENT_MM",
    "alpha_xy_sections",
    "compose_figure_7_xy",
]
