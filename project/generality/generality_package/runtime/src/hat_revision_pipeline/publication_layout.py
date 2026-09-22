"""Reproducible typography and stroke sizing at a declared manuscript width."""
from __future__ import annotations
import re
import numpy as np
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.ticker import AutoLocator, MaxNLocator

WIDTH_MM = 180.0


def _white_data_axis(ax):
    """Identify white scatter/line panels without restyling sampled fields."""
    if ax is None or hasattr(ax, "_colorbar") or ax.images:
        return False
    if hasattr(ax, "zaxis"):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        return not any(isinstance(item, Poly3DCollection) for item in ax.collections)
    return True


def polish_main_figure(fig):
    if getattr(fig, "_manuscript_layout_applied", False):
        return
    width, height = fig.get_size_inches()
    scale = WIDTH_MM / 25.4 / width
    fig.set_size_inches(WIDTH_MM / 25.4, height * scale, forward=True)
    for artist in fig.findobj(Text):
        content = artist.get_text()
        white_panel = _white_data_axis(artist.axes)
        size = max(9.5 if white_panel else 8.0,
                   min(11.0 if white_panel else 9.5, artist.get_fontsize() * scale))
        if re.fullmatch(r"\([a-z]\)", content):
            size = 12.0
            artist.set_fontweight("bold")
        elif "SMOKE" in content:
            size = 6.5
        elif white_panel:
            artist.set_fontweight("semibold")
        artist.set_fontsize(size)
    for ax in fig.axes:
        is_colorbar = hasattr(ax, "_colorbar")
        white_panel = _white_data_axis(ax)
        ax.tick_params(axis="both", which="major",
                       labelsize=9.5 if white_panel else 8.0,
                       width=1.2 if white_panel else 0.8,
                       length=3.5 if white_panel else 3.0, pad=3)
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_fontsize(10.5 if white_panel else 8.5)
            axis.label.set_fontweight("semibold")
            axis.labelpad = 4.5 if white_panel else 3.0
            if white_panel:
                for label in axis.get_ticklabels():
                    label.set_fontweight("semibold")
            if isinstance(axis.get_major_locator(), (AutoLocator, MaxNLocator)):
                axis.set_major_locator(MaxNLocator(nbins=3 if is_colorbar else 4))
        if hasattr(ax, "zaxis"):
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            for collection in ax.collections:
                if isinstance(collection, Poly3DCollection):
                    collection.set_rasterized(True)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.label.set_fontsize(9.0 if white_panel else 8.0)
                axis.label.set_fontweight("semibold")
                axis.labelpad = 2.0 if white_panel else 0.0
                axis.line.set_linewidth(1.2 if white_panel else 0.8)
                for label in axis.get_ticklabels():
                    label.set_fontweight("semibold" if white_panel else "normal")
            ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis="both", labelsize=8.5 if white_panel else 8.0, pad=1)
            ax.tick_params(axis="z", labelsize=8.5 if white_panel else 8.0, pad=1)
        ax.title.set_fontsize(10.5 if white_panel else 9.0)
        ax.title.set_fontweight("semibold")
        for spine in ax.spines.values():
            spine.set_linewidth(1.2 if white_panel else 0.8)
        for grid in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            grid.set_linewidth(0.4)
            grid.set_alpha(0.16)
    for line in fig.findobj(Line2D):
        white_panel = _white_data_axis(line.axes)
        line.set_linewidth(max(1.2 if white_panel else 0.65,
                              min(2.0, line.get_linewidth() * scale)))
        if line.get_marker() not in (None, "None", "", " "):
            line.set_markersize(max(5.5 if white_panel else 4.5,
                                   line.get_markersize() * scale))
            line.set_markeredgewidth(max(1.2 if white_panel else 0.9,
                                         line.get_markeredgewidth() * scale))
    for collection in fig.findobj(PathCollection):
        white_panel = _white_data_axis(collection.axes)
        if len(collection.get_sizes()):
            sizes = collection.get_sizes() * scale**2
            if not getattr(collection, "_manuscript_preserve_marker_area", False):
                sizes = np.maximum(40.0 if white_panel else 18.0, sizes)
            # A figure may request smaller markers after the readability floor.
            # Scatter sizes are areas, so a linear factor must be squared.
            marker_scale = getattr(collection, "_manuscript_marker_linear_scale", 1.0)
            collection.set_sizes(sizes * marker_scale**2)
        collection.set_linewidth(np.maximum(1.3 if white_panel else 0.8,
                                            np.asarray(collection.get_linewidths()) * scale))
    for collection in fig.findobj(LineCollection):
        minimum = 1.2 if _white_data_axis(collection.axes) else 0.5
        collection.set_linewidth(np.maximum(minimum, np.asarray(collection.get_linewidths()) * scale))
    for legend in fig.legends:
        for label in legend.get_texts():
            label.set_fontsize(9.5)
            label.set_fontweight("semibold")
    fig._manuscript_layout_applied = True
