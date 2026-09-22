"""Visual-only helpers for the final RinEng figure pass.

The helpers in this module deliberately do not select data, alter numerical
normalization, or add scientific panels.  They only provide the presentation
operations requested for the final figures:

* large panel identifiers outside the upper-left corner of each axis;
* a compact colorbar adjacent to the axis that owns its mappable; and
* short, reusable mathematical axis labels.

Keeping these operations in one module makes the typography consistent while
leaving the scientific composition in :mod:`hat_revision_pipeline.final_figures`.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


# The previous in-panel labels used 12.5 pt.  The requested outside labels are
# intentionally more than twice that size.
OUTSIDE_PANEL_LABEL_SIZE = 27.0


TERSE_MATH_LABELS: Mapping[str, str] = MappingProxyType(
    {
        # Convergence and outcome panels.
        "iteration": r"Iteration, $k$",
        "accepted_iteration": r"Accepted iteration, $k$",
        "normalized_objective": r"Normalized objective, $\widetilde{J}$",
        "objective": r"Objective, $J$",
        "objective_rise": r"$\Delta J/\Delta J_{\max}$",
        "run_fraction": r"Run fraction, $N/N_{\mathrm{run}}$",
        "gradient_norm": r"$\|\nabla J\|$",
        # Field and phase panels.
        "pressure": r"$|p|$ (Pa)",
        "normalized_pressure": r"$|p|/|p|_{\max}$",
        "phase": r"$\phi$ (rad)",
        "x_offset": r"$x-x^*$ (mm)",
        "y_offset": r"$y-y^*$ (mm)",
        "z_offset": r"$z-z^*$ (mm)",
        # Branch and decision-space panels.
        "similarity_fe": r"$S_{\mathrm{FE}}$",
        "branch_1": r"$\xi_1$",
        "branch_2": r"$\xi_2$",
        "branch_3": r"$\xi_3$",
        "descent_coordinate": r"$s_{\nabla}$ (rad)",
        "transverse_coordinate": r"$s_{\perp}$ (rad)",
        "decision_coordinate_1": r"$s_1$ (rad)",
        "decision_coordinate_2": r"$s_2$ (rad)",
        # Exact-force and engineering-comparison panels.
        "concentration": r"$C_{\mathrm{ROI}}$",
        "exact_stiffness": r"$\kappa_{\min}^{\mathrm{exact}}$ (mN m$^{-1}$)",
        "equilibrium_displacement": r"$\|\Delta\mathbf{r}\|/a$",
        "solve_time": r"$t_{\mathrm{solve}}$ (s)",
        "alpha": r"$\alpha$",
    }
)


def terse_math_label(key: str) -> str:
    """Return the frozen short label for ``key``.

    A missing key is an error rather than an invitation to silently invent new
    plot terminology.  Callers can use a literal label when a quantity is not
    represented in :data:`TERSE_MATH_LABELS`.
    """

    try:
        return TERSE_MATH_LABELS[str(key)]
    except KeyError as error:
        choices = ", ".join(sorted(TERSE_MATH_LABELS))
        raise KeyError(f"Unknown terse label {key!r}; available keys: {choices}") from error


def set_terse_axis_labels(
    ax: mpl.axes.Axes,
    *,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    labelpad: float | None = None,
) -> None:
    """Apply selected terse labels without changing limits, scales, or data."""

    kwargs = {} if labelpad is None else {"labelpad": float(labelpad)}
    if x is not None:
        ax.set_xlabel(terse_math_label(x), **kwargs)
    if y is not None:
        ax.set_ylabel(terse_math_label(y), **kwargs)
    if z is not None:
        if not hasattr(ax, "set_zlabel"):
            raise TypeError("A z label was requested for an axis without set_zlabel")
        ax.set_zlabel(terse_math_label(z), **kwargs)


def _panel_identifier(label: str, parenthesize: bool) -> str:
    value = str(label).strip()
    if parenthesize and len(value) == 1 and value.isalpha():
        return f"({value})"
    return value


def outside_panel_label(
    ax: mpl.axes.Axes,
    label: str,
    *,
    x: float = -0.105,
    y: float = 1.075,
    fontsize: float = OUTSIDE_PANEL_LABEL_SIZE,
    parenthesize: bool = True,
    color: str = "black",
) -> mpl.text.Text:
    """Place a large panel identifier diagonally above-left of ``ax``.

    ``clip_on=False`` keeps the identifier out of the data rectangle.  The
    returned artist remains in the layout calculation, which lets tight-bbox
    exports retain it.  The caller controls figure margins when using a fixed
    (non-tight) export.
    """

    if float(fontsize) <= 25.0:
        raise ValueError("Outside panel labels must remain larger than twice 12.5 pt")
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    artist = text_method(
        float(x),
        float(y),
        _panel_identifier(label, parenthesize),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=float(fontsize),
        fontweight="bold",
        color=color,
        clip_on=False,
        zorder=100,
    )
    artist.set_in_layout(True)
    return artist


def outside_panel_labels(
    axes: Iterable[mpl.axes.Axes],
    labels: Sequence[str],
    **kwargs: object,
) -> list[mpl.text.Text]:
    """Apply :func:`outside_panel_label` to equally sized axis/label lists."""

    axis_list = list(axes)
    label_list = list(labels)
    if len(axis_list) != len(label_list):
        raise ValueError(
            f"axes and labels must have equal length, got {len(axis_list)} and {len(label_list)}"
        )
    return [
        outside_panel_label(axis, label, **kwargs)
        for axis, label in zip(axis_list, label_list, strict=True)
    ]


def outside_grid_panel_labels(
    fig: mpl.figure.Figure,
    specs: Sequence[object],
    labels: Sequence[str],
    *,
    dx: float = 0.016,
    dy: float = 0.010,
    fontsize: float = OUTSIDE_PANEL_LABEL_SIZE,
    parenthesize: bool = True,
    color: str = "black",
) -> list[mpl.text.Text]:
    """Place aligned identifiers from logical GridSpec-cell boundaries.

    Adjacent colorbars and fixed image aspect ratios can move an Axes inside
    its allotted cell.  GridSpec coordinates keep identifiers in exact rows
    and columns regardless of those presentation-only adjustments.  Call this
    helper after the final ``subplots_adjust`` operation.
    """

    spec_list = list(specs)
    label_list = list(labels)
    if len(spec_list) != len(label_list):
        raise ValueError("specs and labels must have the same length")
    if float(fontsize) <= 25.0:
        raise ValueError("Outside panel labels must remain larger than twice 12.5 pt")
    artists: list[mpl.text.Text] = []
    for spec, label in zip(spec_list, label_list, strict=True):
        if not hasattr(spec, "get_position"):
            raise TypeError("Each logical panel must provide SubplotSpec.get_position")
        bounds = spec.get_position(fig)
        artist = fig.text(
            float(bounds.x0 - dx),
            float(bounds.y1 + dy),
            _panel_identifier(label, parenthesize),
            ha="right",
            va="bottom",
            fontsize=float(fontsize),
            fontweight="bold",
            color=color,
            clip_on=False,
            zorder=100,
        )
        artist.set_in_layout(True)
        artists.append(artist)
    return artists


def add_adjacent_colorbar(
    mappable: mpl.cm.ScalarMappable,
    ax: mpl.axes.Axes,
    *,
    side: str = "right",
    size: str | float = "3.5%",
    pad: str | float = "2.5%",
    label: str | None = None,
    label_key: str | None = None,
    labelsize: float = 8.5,
    ticksize: float = 7.5,
    ticks: Sequence[float] | None = None,
    extend: str = "neither",
) -> mpl.colorbar.Colorbar:
    """Attach a compact colorbar directly beside its owning subplot.

    The helper never creates a figure-wide bottom colorbar.  ``side`` may be
    ``"right"``, ``"left"``, ``"top"``, or ``"bottom"``.  A short label can
    be provided literally or through one frozen ``label_key``; providing both
    is rejected to prevent duplicate text.
    """

    side = str(side).lower()
    if side not in {"right", "left", "top", "bottom"}:
        raise ValueError("side must be one of: right, left, top, bottom")
    if label is not None and label_key is not None:
        raise ValueError("Provide either label or label_key, not both")
    if label_key is not None:
        label = terse_math_label(label_key)

    orientation = "vertical" if side in {"right", "left"} else "horizontal"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        side,
        size=size,
        pad=pad,
        axes_class=mpl.axes.Axes,
    )
    colorbar = ax.figure.colorbar(
        mappable,
        cax=cax,
        orientation=orientation,
        ticks=ticks,
        extend=extend,
    )

    if side == "left":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    elif side == "top":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

    if label:
        colorbar.set_label(label, fontsize=float(labelsize), labelpad=3.0)
    colorbar.ax.tick_params(labelsize=float(ticksize), pad=1.5, length=2.5)
    colorbar.ax.grid(False)
    return colorbar


__all__ = [
    "OUTSIDE_PANEL_LABEL_SIZE",
    "TERSE_MATH_LABELS",
    "add_adjacent_colorbar",
    "outside_grid_panel_labels",
    "outside_panel_label",
    "outside_panel_labels",
    "set_terse_axis_labels",
    "terse_math_label",
]
