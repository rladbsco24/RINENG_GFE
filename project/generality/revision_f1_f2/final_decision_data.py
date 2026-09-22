"""Sample wider native sections at the exact frozen Single commands and bases.

This optional preparation script evaluates the unchanged analytic objective.
It never optimizes, changes a phase command, constructs a new basis, or calls
the finite-ka force/equilibrium model. Notebook replay reads its completed cache.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from appendix_io import save_npz, write_bytes
from appendix_settings import SETTINGS
from decision_data import METHODS, objectives
from hat_revision_pipeline.cache import digest_array

ROOT = Path(__file__).resolve().parent
HALF_RANGE = 3.0
GRID_SIZE = 61


def _section(objective, phase, directions, grid_size=None):
    grid_size = GRID_SIZE if grid_size is None else int(grid_size)
    q = np.linspace(-HALF_RANGE, HALF_RANGE, grid_size)
    u, v = directions
    values = np.empty((grid_size, grid_size))
    for j, y in enumerate(q):
        for i, x in enumerate(q):
            values[j, i] = objective.full_value(phase + x*u + y*v)
    return q, values - values.min()


def _arrows(objective, phase, directions, q):
    indices = np.rint(np.linspace(6, len(q)-7, 7)).astype(int)
    arrow_q = q[indices]
    xx, yy = np.meshgrid(arrow_q, arrow_q)
    du, dv = np.empty_like(xx), np.empty_like(xx)
    u, v = directions
    for index in np.ndindex(xx.shape):
        _, gradient = objective.full_fun_grad(phase + xx[index]*u + yy[index]*v)
        du[index], dv[index] = -gradient@u, -gradient@v
    return arrow_q, du, dv


def build_case(sid):
    data = ROOT / "data"
    output = data / "final_decision"
    output.mkdir(parents=True, exist_ok=True)
    source = data / "decision" / f"{sid}_decision.npz"
    destination = output / source.name
    metadata_path = output / f"{sid}_decision_metadata.json"
    with np.load(source, allow_pickle=False) as z:
        original = {key: z[key].copy() for key in z.files}
    if destination.exists() and metadata_path.exists():
        with np.load(destination, allow_pickle=False) as saved:
            for key in ("methods", "endpoint_phase_rad", "directions", "target_m", "frequency_hz"):
                assert np.array_equal(saved[key], original[key]), (sid, key)
            assert np.array_equal(saved["q"], np.linspace(-HALF_RANGE, HALF_RANGE, GRID_SIZE))
        return json.loads(metadata_path.read_text())
    native, phases, target, frequency = objectives(data / "solution_correspondence/campaign" / sid)
    assert np.array_equal(phases, original["endpoint_phase_rad"]), sid
    assert np.array_equal(target, original["target_m"]), sid
    assert frequency == float(original["frequency_hz"]), sid
    started = time.perf_counter()
    planes, arrows_u, arrows_v, records = [], [], [], []
    for index, (method, objective, phase) in enumerate(zip(METHODS, native, phases)):
        directions = original["directions"][index]
        candidate = data / "opposed_visual_window" / f"{sid}_{method}_range3.npz"
        reused = False
        if candidate.exists():
            with np.load(candidate, allow_pickle=False) as saved:
                reused = (np.array_equal(saved["phase_rad"], phase)
                    and np.array_equal(saved["directions"], directions)
                    and np.array_equal(saved["q"], np.linspace(-HALF_RANGE, HALF_RANGE, GRID_SIZE)))
                if reused:
                    q, plane = saved["q"].copy(), saved["plane_delta_j"].copy()
                    arrow_q = saved["arrow_q"].copy()
                    du, dv = saved["descent_u"].copy(), saved["descent_v"].copy()
        if not reused:
            q, plane = _section(objective, phase, directions)
            arrow_q, du, dv = _arrows(objective, phase, directions, q)
        planes.append(plane); arrows_u.append(du); arrows_v.append(dv)
        center = len(q)//2
        minimum = np.unravel_index(int(np.argmin(plane)), plane.shape)
        scale = float(np.quantile(plane, .985))
        horizontal = float(np.ptp(plane[center, :]))
        vertical = float(np.ptp(plane[:, center]))
        records.append(dict(method=method, phase_fingerprint=digest_array(phase),
            directions_fingerprint=digest_array(directions), command_and_basis_identical_to_original=True,
            reused_exact_cache=str(candidate.relative_to(ROOT)) if reused else None,
            color_scale=scale, center_above_sampled_min=float(plane[center, center]),
            sampled_minimum_coordinates_rad=[float(q[minimum[1]]), float(q[minimum[0]])],
            horizontal_centerline_span=horizontal, vertical_centerline_span=vertical,
            centerline_span_ratio=horizontal/vertical if vertical else None,
            alpha_per_m=float(objective.method.alpha_per_m),
            curvature_weight=np.asarray(objective.method.curvature_weight).tolist()))
        print(f"{sid} {method}: {'reused exact cache' if reused else 'sampled frozen section'}", flush=True)
    arrays = dict(original)
    arrays.update(q=q, plane_delta_j=np.stack(planes), arrow_q=arrow_q,
        descent_u=np.stack(arrows_u), descent_v=np.stack(arrows_v))
    assert all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind != "U")
    save_npz(destination, **arrays)
    metadata = dict(setting_id=sid, methods=list(METHODS), grid_size=GRID_SIZE,
        half_range_l2_rad=HALF_RANGE, source=str(source.relative_to(ROOT)),
        endpoint_and_native_basis="Exact unchanged current Single commands and deposited original native directions",
        direction_definition="Native terminal gradient and orthogonalized last accepted step; read from original cache without recomputation",
        objective="Unchanged native dimensional objective, alpha, force target and directional weights",
        ordinate="J minus the sampled section minimum",
        arrow_definition="Negative analytic full gradient projected onto the same frozen native basis; equal-length glyphs",
        color_scale="Linear per-section Q98.5 normalization; equal axis scales",
        optimizer_calls=0, basis_recomputations=0, finite_ka_calls=0,
        elapsed_s=time.perf_counter()-started, endpoints=records)
    write_bytes(metadata_path, (json.dumps(metadata, indent=2)+"\n").encode())
    return metadata


def resolution_check():
    """Check the final frozen Opposed GFE section, with a nested 121-square grid."""
    from scipy.interpolate import RegularGridInterpolator

    output = ROOT / "data/final_decision"
    destination = output / "S02_GFE_resolution121.npz"
    metadata_path = output / "resolution_check.json"
    with np.load(output / "S02_decision.npz", allow_pickle=False) as saved:
        index = list(saved["methods"]).index("GFE")
        phase, directions = saved["endpoint_phase_rad"][index], saved["directions"][index]
        coarse_q, coarse = saved["q"], saved["plane_delta_j"][index]
    if destination.exists() and metadata_path.exists():
        with np.load(destination, allow_pickle=False) as saved:
            assert np.array_equal(saved["phase_rad"], phase)
            assert np.array_equal(saved["directions"], directions)
        return json.loads(metadata_path.read_text())
    objective = objectives(ROOT / "data/solution_correspondence/campaign/S02")[0][2]
    q, fine = _section(objective, phase, directions, grid_size=121)
    scale = float(np.quantile(fine, .985))
    coarse_scale = float(np.quantile(coarse, .985))
    xx, yy = np.meshgrid(q, q)
    interpolated = RegularGridInterpolator((coarse_q, coarse_q), coarse/coarse_scale)(
        np.c_[yy.ravel(), xx.ravel()]).reshape(fine.shape)
    minimum = np.unravel_index(int(np.argmin(fine)), fine.shape)
    record = dict(setting_id="S02", method="GFE", grid_points=121, coarse_grid_points=61,
        half_range_l2_rad=HALF_RANGE, phase_fingerprint=digest_array(phase),
        directions_fingerprint=digest_array(directions),
        shared_node_max_absolute_delta_j_error=float(np.max(np.abs(fine[::2, ::2]-coarse))),
        coarse_vs_fine_color_scale_relative_difference=coarse_scale/scale-1,
        normalized_interpolation_rmse=float(np.sqrt(np.mean((interpolated-fine/scale)**2))),
        sampled_minimum_coordinates_rad=[float(q[minimum[1]]), float(q[minimum[0]])],
        centerline_span_ratio=float(np.ptp(fine[60, :])/np.ptp(fine[:, 60])),
        optimizer_calls=0, basis_recomputations=0, finite_ka_calls=0)
    save_npz(destination, q=q, plane_delta_j=fine, phase_rad=phase, directions=directions)
    write_bytes(metadata_path, (json.dumps(record, indent=2)+"\n").encode())
    return record


def main(workers=4):
    ids = [setting["id"] for setting in SETTINGS]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(build_case, ids))
    else:
        records = [build_case(sid) for sid in ids]
    resolution = resolution_check()
    protocol = dict(setting_ids=ids, settings_count=len(ids), section_count=3*len(ids),
        half_range_l2_rad=HALF_RANGE, grid_size=GRID_SIZE,
        exact_cache_reuse_count=sum(item["reused_exact_cache"] is not None for row in records for item in row["endpoints"]),
        scope="One common wider range for every setting and method; exact frozen Single command and native basis identities preserved",
        new_optimization=False, basis_recomputation=False, finite_ka_calls=0,
        interpretation="A wider view of existing objective sections, not an optimization-induced change in the landscape",
        resolution_check=resolution)
    write_bytes(ROOT / "data/final_decision/protocol.json", (json.dumps(protocol, indent=2)+"\n").encode())
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    main(parser.parse_args().workers)
