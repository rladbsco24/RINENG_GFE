"""Compact source-layout thumbnails drawn from the plotted command geometry."""
from __future__ import annotations

import fingerprintlib
from pathlib import Path

import numpy as np


def load_source_geometry(data_root, setting_id, task="Single", expected_count=None):
    """Read saved coordinates; never substitute a newly configured geometry."""
    data_root = Path(data_root)
    folder = "triple_pressure" if task == "Triple" else "pressure"
    candidates = [data_root / folder / f"{setting_id}_pressure.npz"]
    if task == "Single":
        candidates.append(data_root.parent / "campaign" / setting_id / "array_geometry.npz")
    for path in candidates:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            if "positions_m" not in saved or "normals" not in saved:
                continue
            positions = np.asarray(saved["positions_m"], dtype=float)
            normals = np.asarray(saved["normals"], dtype=float)
        if positions.shape != normals.shape or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Invalid source coordinates: {path}")
        if not np.isfinite(positions).all() or not np.isfinite(normals).all():
            raise ValueError(f"Nonfinite source coordinates: {path}")
        if expected_count is not None and len(positions) != int(expected_count):
            raise ValueError(f"{setting_id}: plotted command has {expected_count} phases, geometry has {len(positions)} sources")
        return positions, normals
    raise FileNotFoundError(f"No saved {task} source geometry for {setting_id}")


def reserve_header(ax, height_in=.57):
    """Reserve white space above a plot without placing geometry over its data."""
    box = ax.get_position()
    inset_height = height_in / ax.figure.get_figheight()
    ax.set_position([box.x0, box.y0, box.width, box.height - inset_height])


def add_array_inset(fig, data_box, positions, normals, width_in=.78,
                    height_in=.51, bottom_gap_in=.025):
    """Show every source under the same orthographic view, with physical aspect."""
    positions, normals = np.asarray(positions, float), np.asarray(normals, float)
    fw, fh = fig.get_size_inches()
    width, height = width_in / fw, height_in / fh
    ax = fig.add_axes([data_box.x1 - width, data_box.y1 + bottom_gap_in / fh, width, height])
    azimuth, elevation = np.deg2rad([-62., 25.])
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    vertical = np.array([-np.sin(elevation)*np.cos(azimuth),
                         -np.sin(elevation)*np.sin(azimuth), np.cos(elevation)])
    depth_axis = np.cross(horizontal, vertical)
    centered = positions - .5 * (positions.min(axis=0) + positions.max(axis=0))
    projected = np.column_stack((centered @ horizontal, centered @ vertical))
    depth = centered @ depth_axis
    order = np.argsort(depth)
    # Normal indicators use a fixed fraction of the physical aperture. They
    # identify inward-facing opposed panels and nonplanar source orientations.
    aperture = max(float(np.ptp(positions, axis=0).max()), 1e-12)
    normal_xy = np.column_stack((normals @ horizontal, normals @ vertical))
    selected = np.linspace(0, len(positions)-1, min(12, len(positions)), dtype=int)
    from matplotlib.collections import LineCollection
    segments = np.stack((projected[selected], projected[selected] + .055*aperture*normal_xy[selected]), axis=1)
    ax.add_collection(LineCollection(segments, colors="#A2B4C7", linewidths=.40, zorder=1))
    shade = .50 + .42*(depth-depth.min()) / max(float(np.ptp(depth)), 1e-12)
    base = np.array([.19, .38, .56])
    colors = 1. - shade[:, None] * (1. - base)
    ax.scatter(projected[order, 0], projected[order, 1], s=1.25,
               c=colors[order], linewidths=0, zorder=2, rasterized=False)
    center = .5*(projected.min(axis=0)+projected.max(axis=0))
    halfwidth = .57*max(float(np.ptp(projected[:, 0])),
                        float(np.ptp(projected[:, 1])) * width_in/(height_in*.81))
    halfheight = halfwidth * height_in/width_in
    ax.set_xlim(center[0]-halfwidth, center[0]+halfwidth)
    ax.set_ylim(center[1]-1.34*halfheight, center[1]+.90*halfheight)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.text(.5, -.015, f"N = {len(positions)}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, color="#536274")
    return dict(source_count=len(positions),
                positions_fingerprint=fingerprintlib.fingerprint(np.ascontiguousarray(positions, dtype="<f8").tobytes()).hexdigest(),
                normals_fingerprint=fingerprintlib.fingerprint(np.ascontiguousarray(normals, dtype="<f8").tobytes()).hexdigest(),
                view_azimuth_deg=-62, view_elevation_deg=25,
                projection="Orthographic; equal physical coordinate scaling within each inset",
                location="Reserved header outside the data axes")
