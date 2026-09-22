"""Explicit 9-array and 6-carrier appendix settings (14 unique cases).

Four mathematical arrays and two carrier settings extend the original experiment.
The opposed layout now uses two full 16 × 16 faces; its original 256-source
results remain a separate budget comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE_ROOT = ROOT.parent / "generality_package"
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))
import generality_settings as original

NEW_SETTINGS = [
    dict(id="A06", label="Square · 40 kHz", geometry="square", frequency_hz=40000.0),
    dict(id="A07", label="Annular · 40 kHz", geometry="annular", frequency_hz=40000.0),
    dict(id="A08", label="Four clusters · 40 kHz", geometry="four_clusters", frequency_hz=40000.0),
    dict(id="A09", label="Tilted plane · 40 kHz", geometry="tilted_plane", frequency_hz=40000.0),
    dict(id="F20", label="Square · 20 kHz", geometry="square", frequency_hz=20000.0),
    dict(id="F28", label="Square · 28 kHz", geometry="square", frequency_hz=28000.0),
]
SETTINGS = [dict(s) for s in [*original.SETTINGS, *NEW_SETTINGS]]
for setting in SETTINGS:
    if setting['id'] == 'S02':
        setting.update(source_count=512, face_grid=[16, 16], budget_version='opposed_512')
    elif setting['id'] == 'A07':
        setting.update(source_count=512, ring_counts=[32,38,45,51,58,64,70,76,78], budget_version='annular_512')
ARRAY_IDS = ["A06", "S01", "S02", "S03", "S04", "S05", "A07", "A08", "A09"]
FREQUENCY_IDS = ["F20", "S06", "F28", "A06", "S07", "S08"]
DESCRIPTIONS = {
    **original.GEOMETRY_DESCRIPTIONS,
    "opposed": "Two 16 × 16 grids, 10 mm pitch, at z = 0 and 100 mm; inward normals; 512 sources total. Expanded from the earlier two 16 × 8 faces, retaining pitch and separation.",
    "annular": "Nine concentric planar rings with 32,38,45,51,58,64,70,76,78 sources; radii52.0 to133.6mm in10.2mm steps; normals+z; minimum source spacing10.048mm;512sources.",
    "four_clusters": "Four 8 × 8 planar grids at 10 mm pitch, centered at (±65, ±65, 0) mm; source normals +z.",
    "tilted_plane": "Original 16 × 16 square grid rotated 25 degrees about y, then translated to z = −20 mm; normals rotated with the plane.",
}
FREQUENCY_REFERENCES = dict(original.FREQUENCY_REFERENCES)
FREQUENCY_REFERENCES.update({
    20000: dict(author_year="Rueckner et al. (2023)", title="Particle size effects on stable levitation positions in acoustic standing waves",
        url="https://pubs.aip.org/asa/jasa/article/154/2/1339/2908529/Particle-size-effects-on-stable-levitation"),
    28000: dict(author_year="Jackson and Chang (2021)", title="Acoustic levitation and the acoustic radiation force",
        url="https://pubs.aip.org/aapt/ajp/article/89/4/383/1057857/Acoustic-levitation-and-the-acoustic-radiation"),
    40000: dict(author_year="Ochiai et al. (2014)", title="Three-Dimensional Mid-Air Acoustic Manipulation by Ultrasonic Phased Arrays",
        url="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0097590"),
})


def geometry(setting):
    if isinstance(setting, str):
        setting = next((s for s in SETTINGS if s["id"] == setting), {"geometry": setting})
    kind = setting["geometry"]
    if kind == "opposed":
        positions = np.vstack((original._grid(16, 16, 0.0), original._grid(16, 16, .100)))
        normals = np.vstack((original._upward(256), -original._upward(256)))
    elif kind == "annular":
        from annular_resources import geometry as annular_geometry
        positions, normals = annular_geometry()
    elif kind == "four_clusters":
        local = original._grid(8, 8)
        positions = np.vstack([local + [x, y, 0] for x in (-.065, .065) for y in (-.065, .065)])
        normals = original._upward(256)
    elif kind == "tilted_plane":
        angle = np.deg2rad(25.)
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        positions = original._grid(16, 16) @ rotation.T + [0., 0., -.020]
        normals = original._upward(256) @ rotation.T
    else:
        return original.geometry(setting)
    count = 512 if kind in ("opposed", "annular") else 256
    assert positions.shape == normals.shape == (count, 3)
    assert np.isfinite(positions).all() and np.isfinite(normals).all()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-12)
    assert original._minimum_spacing(positions) >= .010 - 1e-12
    return positions, normals


def install():
    """Extend only the imported adapter; the deposited source remains untouched."""
    import run_generality
    run_generality.SETTINGS = SETTINGS
    run_generality.geometry = geometry
    return run_generality


def export(campaign):
    campaign = Path(campaign)
    (campaign / "geometries").mkdir(parents=True, exist_ok=True)
    records = []
    for setting in SETTINGS:
        positions, normals = geometry(setting)
        record = dict(setting, description=DESCRIPTIONS[setting["geometry"]],
            source_count=len(positions), minimum_source_spacing_mm=1000 * original._minimum_spacing(positions),
            source_bounds_mm=(1000 * np.array([positions.min(axis=0), positions.max(axis=0)])).tolist(),
            bead_radius_mm=1000 * original.BEAD_RADIUS_M,
            single_target_m=original.SINGLE_TARGET_M.tolist(),
            wavelength_air_mm=343000 / setting["frequency_hz"],
            bead_ka=2 * np.pi * setting["frequency_hz"] * original.BEAD_RADIUS_M / 343,
            frequency_reference=FREQUENCY_REFERENCES.get(int(setting["frequency_hz"])),
            geometry_file=f"geometries/{setting['id']}.npz")
        np.savez_compressed(campaign / record["geometry_file"], positions_m=positions, normals=normals, frequency_hz=setting["frequency_hz"])
        records.append(record)
    (campaign / "settings.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (campaign / "appendix_design.json").write_text(json.dumps(dict(
        array_ids=ARRAY_IDS, frequency_ids=FREQUENCY_IDS, unique_case_count=14,
        shared_case="A06", array_grid=[3, 3], frequency_grid=[2, 3],
        source_assumption="Fixed reference source amplitude and Murata angular pattern; alternate carriers denote ideal retuned sources.",
        original_results_root=str(BASE_ROOT / "outputs")), indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    export(ROOT / "campaign")
