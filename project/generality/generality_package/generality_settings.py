"""Fixed array and carrier settings for the eight-setting main-figure study.

All source coordinates and bead/target dimensions remain in physical metres.
The alternate carriers describe ideal retuned sources with the reference source
amplitude and directivity law; they do not assume a 40 kHz device is broadband.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np


SETTINGS = [
    dict(id="S01", label="Rectangular · 40 kHz",
         geometry="rectangular", frequency_hz=40000.0),
    dict(id="S02", label="Opposed · 40 kHz",
         geometry="opposed", frequency_hz=40000.0),
    dict(id="S03", label="Spherical cap · 40 kHz",
         geometry="spherical_cap", frequency_hz=40000.0),
    dict(id="S04", label="Fermat disk · 40 kHz",
         geometry="fermat_disk", frequency_hz=40000.0),
    dict(id="S05", label="Arbitrary 3D · 40 kHz",
         geometry="arbitrary", frequency_hz=40000.0),
    dict(id="S06", label="Square · 25 kHz",
         geometry="square", frequency_hz=25000.0),
    dict(id="S07", label="Square · 58 kHz",
         geometry="square", frequency_hz=58000.0),
    dict(id="S08", label="Square · 100 kHz",
         geometry="square", frequency_hz=100000.0),
]

BEAD_RADIUS_M = 0.00065
SINGLE_TARGET_M = np.array([0.0, 0.0, 0.050])
_BASE_TRIPLE_M = np.array([[0.0, 0.013, 0.030],
                           [-0.011, -0.009, 0.030],
                           [0.011, -0.009, 0.030]])
TRIPLE_TARGETS_M = (_BASE_TRIPLE_M.mean(axis=0)
                    + 0.8 * (_BASE_TRIPLE_M - _BASE_TRIPLE_M.mean(axis=0)))
ARBITRARY_SEED = 260906
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))

FREQUENCY_REFERENCES = {
    25000: dict(author_year="Andrade et al. (2016)",
                title="Acoustic levitation of a large solid sphere",
                url="https://pubs.aip.org/aip/apl/article/109/4/044101/892084/Acoustic-levitation-of-a-large-solid-sphere"),
    58000: dict(author_year="Park et al. (2023)",
                url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9988397/"),
    100000: dict(author_year="Milsom et al. (2024)",
                 url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11138859/"),
}

GEOMETRY_DESCRIPTIONS = {
    "rectangular": "32 × 8 planar grid, 10 mm pitch, source normals +z.",
    "opposed": "Two 16 × 8 grids, 10 mm pitch, at z = 0 and 100 mm; inward normals.",
    "spherical_cap": "256 Fermat directions on a 64° spherical cap centered at (0, 0, 70) mm; radius at least 110 mm and scaled only as needed for 10.05 mm source spacing; inward normals.",
    "fermat_disk": "256 Fermat points in a planar disk of radius at least 90 mm, scaled only as needed for 10.05 mm source spacing; normals +z.",
    "arbitrary": "256 deterministically sampled sources in a 120 × 105 mm semiaxis ellipse, nonplanar height and randomized normals; minimum 10.1 mm center spacing; seed 260906.",
    "square": "16 × 16 planar grid, 10 mm pitch, source normals +z; original physical dimensions retained.",
}


def _minimum_spacing(positions):
    delta = positions[:, None, :] - positions[None, :, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(squared, np.inf)
    return float(np.sqrt(np.min(squared)))


def _normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _grid(nx, ny, z=0.0):
    x = (np.arange(nx) - (nx - 1) / 2.0) * 0.010
    y = (np.arange(ny) - (ny - 1) / 2.0) * 0.010
    return np.array([(xx, yy, z) for xx in x for yy in y], dtype=float)


def _upward(count):
    return np.tile([0.0, 0.0, 1.0], (count, 1))


def _fermat_disk():
    index = np.arange(256)
    radius = np.sqrt((index + 0.5) / 256)
    phi = index * GOLDEN_ANGLE
    positions = np.column_stack((radius * np.cos(phi), radius * np.sin(phi),
                                 np.zeros(256)))
    scale = max(0.090, 0.01005 / _minimum_spacing(positions))
    return scale * positions, _upward(256)


def _spherical_cap():
    index = np.arange(256)
    cosine = 1.0 - (index + 0.5) / 256 * (1.0 - np.cos(np.deg2rad(64.0)))
    sine = np.sqrt(1.0 - cosine**2)
    phi = index * GOLDEN_ANGLE
    direction = np.column_stack((sine * np.cos(phi), sine * np.sin(phi), -cosine))
    radius = max(0.110, 0.01005 / _minimum_spacing(direction))
    positions = np.array([0.0, 0.0, 0.070]) + radius * direction
    return positions, -direction


def _arbitrary():
    rng = np.random.default_rng(ARBITRARY_SEED)
    accepted = []
    # Source packing is deterministic and fixed before any optimization.
    for _ in range(200000):
        x = rng.uniform(-0.120, 0.120)
        y = rng.uniform(-0.105, 0.105)
        if (x / 0.120)**2 + (y / 0.105)**2 > 1.0:
            continue
        z = -0.020 + 0.014 * np.sin(24.0 * x + 1.2) * np.cos(31.0 * y)
        z += rng.uniform(-0.011, 0.011)
        point = np.array([x, y, z])
        if accepted and np.any(np.linalg.norm(np.asarray(accepted) - point, axis=1) < 0.0101):
            continue
        accepted.append(point)
        if len(accepted) == 256:
            break
    if len(accepted) != 256:
        raise RuntimeError("Deterministic source packing did not reach 256 sources")
    positions = np.asarray(accepted)
    aim = _normalize(np.array([0.0, 0.0, 0.045]) - positions)
    reference = np.tile([0.0, 0.0, 1.0], (256, 1))
    reference[np.abs(aim[:, 2]) > 0.90] = [1.0, 0.0, 0.0]
    tangent_a = _normalize(np.cross(aim, reference))
    tangent_b = np.cross(aim, tangent_a)
    angle = np.arccos(rng.uniform(np.cos(np.deg2rad(32.0)), 1.0, 256))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, 256)
    tangent = (np.cos(azimuth)[:, None] * tangent_a
               + np.sin(azimuth)[:, None] * tangent_b)
    normals = np.cos(angle)[:, None] * aim + np.sin(angle)[:, None] * tangent
    return positions, _normalize(normals)


def geometry(setting):
    """Return the 256 physical source positions and their unit normals."""
    if isinstance(setting, str):
        matches = [entry for entry in SETTINGS if entry["id"] == setting]
        kind = matches[0]["geometry"] if matches else setting
    else:
        kind = setting["geometry"]
    if kind == "rectangular":
        positions, normals = _grid(32, 8), _upward(256)
    elif kind == "opposed":
        positions = np.vstack((_grid(16, 8, 0.0), _grid(16, 8, 0.100)))
        normals = np.vstack((_upward(128), -_upward(128)))
    elif kind == "spherical_cap":
        positions, normals = _spherical_cap()
    elif kind == "fermat_disk":
        positions, normals = _fermat_disk()
    elif kind == "arbitrary":
        positions, normals = _arbitrary()
    elif kind == "square":
        positions, normals = _grid(16, 16), _upward(256)
    else:
        raise ValueError(f"Unknown geometry: {kind}")
    assert positions.shape == normals.shape == (256, 3)
    assert np.isfinite(positions).all() and np.isfinite(normals).all()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
    assert _minimum_spacing(positions) >= 0.010 - 1e-12
    return positions, normals


def geometry_metadata(setting):
    positions, normals = geometry(setting)
    record = dict(setting)
    record["label"] = setting["label"].split(" · ")[0]
    record.update(
        description=GEOMETRY_DESCRIPTIONS[setting["geometry"]],
        source_count=len(positions),
        minimum_source_spacing_mm=1000.0 * _minimum_spacing(positions),
        source_extent_mm=(1000.0 * np.ptp(positions, axis=0)).tolist(),
        source_bounds_mm=(1000.0 * np.array([positions.min(axis=0), positions.max(axis=0)])).tolist(),
        bead_radius_mm=1000.0 * BEAD_RADIUS_M,
        single_target_m=SINGLE_TARGET_M.tolist(),
        triple_targets_m=TRIPLE_TARGETS_M.tolist(),
        wavelength_air_mm=1000.0 * 343.0 / setting["frequency_hz"],
        bead_ka=2.0 * np.pi * setting["frequency_hz"] * BEAD_RADIUS_M / 343.0,
        geometry_file=f"geometries/{setting['id']}.npz",
        frequency_reference=FREQUENCY_REFERENCES.get(int(setting["frequency_hz"])),
    )
    if setting["geometry"] == "spherical_cap":
        record["sphere_radius_mm"] = float(1000.0 * np.linalg.norm(positions[0] - [0, 0, 0.070]))
    if setting["geometry"] == "fermat_disk":
        record["disk_outer_source_radius_mm"] = float(1000.0 * np.linalg.norm(positions[:, :2], axis=1).max())
    if setting["geometry"] == "arbitrary":
        aim = _normalize(np.array([0.0, 0.0, 0.045]) - positions)
        record["normal_max_tilt_from_aim_deg"] = float(np.rad2deg(np.arccos(np.clip(np.sum(aim * normals, axis=1), -1, 1))).max())
        record["geometry_seed"] = ARBITRARY_SEED
    return record


def preview(campaign):
    """Save a compact source-layout plate with common limits and view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(15.2, 8.1), facecolor="white")
    for index, setting in enumerate(SETTINGS):
        positions, normals = geometry(setting)
        positions_mm = 1000.0 * positions
        ax = figure.add_subplot(2, 4, index + 1, projection="3d")
        color = "#1864ab" if index < 5 else "#7950a3"
        ax.scatter(*positions_mm.T, s=5.5, color=color, alpha=0.9, depthshade=False)
        selection = np.arange(0, 256, 16)
        ax.quiver(*positions_mm[selection].T, *normals[selection].T,
                  length=13.0, color=color, linewidth=0.65, alpha=0.8,
                  arrow_length_ratio=0.25)
        ax.scatter(*(1000.0 * SINGLE_TARGET_M), color="#c92a2a", marker="x", s=42, linewidth=1.5)
        ax.scatter(*(1000.0 * TRIPLE_TARGETS_M).T, color="#e67700", marker="+", s=34, linewidth=1.5)
        ax.set(xlim=(-165, 165), ylim=(-165, 165), zlim=(-65, 115))
        ax.set_box_aspect((330, 330, 180))
        ax.view_init(elev=25, azim=-62)
        ax.set_xticks([-150, 0, 150])
        ax.set_yticks([-150, 0, 150])
        ax.set_zticks([-50, 0, 50, 100])
        ax.tick_params(labelsize=7, pad=0)
        ax.set_xlabel("x (mm)", fontsize=8, labelpad=-3)
        ax.set_ylabel("y (mm)", fontsize=8, labelpad=-3)
        ax.set_zlabel("z (mm)", fontsize=8, labelpad=-3)
        ax.set_title(f"{index + 1:02d}  {setting['label'].split(' · ')[0]} · {setting['frequency_hz'] / 1000:g} kHz", fontsize=11, pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
        ax.grid(True, alpha=0.2)
    handles = [Line2D([], [], marker="o", color="#1864ab", linestyle="none", markersize=4, label="256 sources per setting"),
               Line2D([], [], marker="x", color="#c92a2a", linestyle="none", markersize=7, label="Single target"),
               Line2D([], [], marker="+", color="#e67700", linestyle="none", markersize=7, label="Triple targets")]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.018))
    figure.subplots_adjust(left=0.025, right=0.945, top=0.94, bottom=0.11, wspace=0.06, hspace=0.12)
    path = campaign / "geometry_overview.png"
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return path


def export(campaign, mode="smoke"):
    """Export exact source coordinates, setting metadata, and geometry preview."""
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "geometries").mkdir(exist_ok=True)
    records = []
    for setting in SETTINGS:
        positions, normals = geometry(setting)
        record = geometry_metadata(setting)
        np.savez_compressed(campaign / record["geometry_file"], positions_m=positions,
                            normals=normals, frequency_hz=setting["frequency_hz"])
        records.append(record)
    (campaign / "settings.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    metadata = dict(
        title="Main figures across eight physical settings",
        settings=records,
        setting_count=8,
        main_figure_count_per_setting=8,
        total_requested_main_figures=64,
        selection="Five array alternatives at 40 kHz plus three literature-supported carrier choices on the original square array; this is an eight-setting study, not a 5 × 3 Cartesian sweep.",
        controls="256 sources, fixed physical bead radius 0.65 mm, original physical targets, fixed source amplitude and reference directivity law; no wavelength rescaling of geometry or target coordinates.",
        source_assumption="Alternate carriers are ideal retuned sources with the Murata reference radiation pattern. The carrier references establish acoustical use of the selected frequencies and do not validate this precise array hardware at those frequencies.",
        geometry_overview="geometry_overview.png",
    )
    metadata["notes"] = [metadata["controls"], metadata["source_assumption"],
        ("Full run: 30 paired Single starts; finite-ka elastic validation uses lmax=8, 19 root starts and up to 70 solver evaluations per start." if mode == "full" else "Review run: four paired Single starts; finite-ka elastic validation uses lmax=6, seven root starts and up to 22 solver evaluations per start. The notebook also offers a larger full mode."),
        "Read the numerical colorbars when comparing settings: each setting retains the original within-figure normalization. Single optimizers retain a 10,000-iteration ceiling and matched Triple optimizers retain 30,000.",
        "Figure 2 uses six independently cold-started GFE endpoints at each target. Two observed reference sheets are tracked by full-dimensional projector matching, and the remaining GFE endpoints are retained. Each Conventional trajectory is matched once from its terminal state. The displayed sheets do not assert exactly two endpoint classes."]
    metadata["figure_titles"] = {"1":"Convergence and target domain", "2":"Endpoint correspondence and branch evolution", "3":"Local objective geometry", "4":"Triple fields and local geometry", "5":"Field concentration and elastic stiffness", "6":"Triple fields and equilibria", "7":"Force-weight response", "8":"Synthesis time and mechanical precision"}
    metadata["sources"] = [dict(
        title=f"{frequency / 1000:g} kHz: {reference['author_year']}",
        url=reference["url"],
        note=reference.get("title", "Literature support for the carrier choice"))
        for frequency, reference in FREQUENCY_REFERENCES.items()]
    (campaign / "atlas_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    preview(campaign)
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    export(arguments.output)
