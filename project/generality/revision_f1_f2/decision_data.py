"""Reproduce the main Figure 3 endpoint-local decision sections.

The main source supplies both direction construction and scalar-grid evaluation.
Only projected analytic descent arrows are added to its two-dimensional display.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / "generality_package"
sys.path.insert(0, str(PACKAGE / "runtime/src"))
from hat_revision_pipeline.gorkov_core import Condition, SingleTargetObjective
from hat_revision_pipeline.fixed_fe_endpoints import fixed_single_fe_method_spec
from hat_revision_pipeline.multitrap import SphereInFluid, effective_weight_force_target_n
from hat_revision_pipeline.pipeline import _native_local_directions, _native_relative_sections
from hat_revision_pipeline.cache import digest_array
from appendix_io import save_npz, write_bytes
from appendix_settings import SETTINGS

METHODS = ("Conventional", "FE", "GFE")
COMMAND_FILES = ("Single_Conventional.npz", "Single_FE.npz", "Single_FEplusg.npz")


def objectives(directory):
    """Use the same dimensional objectives and frozen commands as the main study."""
    with np.load(directory / "array_geometry.npz") as geometry:
        positions = geometry["positions_m"]
        normals = geometry["normals"]
        frequency = float(geometry["frequency_hz"])
    phases, targets, alphas = [], [], []
    for filename in COMMAND_FILES:
        with np.load(directory / "commands" / filename) as command:
            phases.append(command["phase_rad"])
            targets.append(command["target_m"])
            alphas.append(float(command['alpha_per_m']) if 'alpha_per_m' in command else 10.)
    if not np.allclose(targets, targets[0], rtol=0, atol=1e-14):
        raise ValueError("Paired endpoints have different targets")
    conventional = SingleTargetObjective(Condition(
        label="appendix-native-decision-section", positions=positions,
        normals=normals, target_m=tuple(targets[0]), frequency_hz=frequency,
    ), "Conventional")
    objective_list = [conventional]
    for name, alpha in zip(METHODS[1:], alphas[1:]):
        method = replace(fixed_single_fe_method_spec(), alpha_per_m=alpha)
        condition = Condition(label=f"Single-{name}-alpha{alpha:g}", positions=positions,
            target_m=tuple(targets[0]), frequency_hz=frequency,
            curvature_weight=method.curvature_weight, normals=normals)
        # Keep the native zero force target for FE and effective weight for GFE.
        force_target = (effective_weight_force_target_n(SphereInFluid(frequency_hz=frequency))
                        if name == "GFE" else np.zeros(3))
        objective_list.append(SingleTargetObjective(condition, method, force_target_n=force_target))
    return tuple(objective_list), np.stack(phases), np.asarray(targets[0]), frequency


def load_run(directory, method, endpoint, history_root=None):
    """Require recorded accepted phases; never invent an optimizer step."""
    candidates = [directory / "paired_single" / f"{method}_pair0.npz"]
    if history_root is not None:
        candidates = [Path(history_root) / f"{directory.name}_{method}_seed260828.npz",
                      Path(history_root) / directory.name / f"{method}_pair0.npz",
                      Path(history_root) / directory.name / "paired_single" / f"{method}_pair0.npz",
                      *candidates]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        compact = ROOT / "data/decision" / f"{directory.name}_decision.npz"
        key = f"history_{method}_tail_rad"
        if not compact.exists():
            raise FileNotFoundError(f"Recorded accepted history required: {directory.name} {method}")
        with np.load(compact, allow_pickle=False) as pack:
            if key not in pack:
                raise FileNotFoundError(f"Recorded accepted history required: {directory.name} {method}")
            index = list(pack["methods"]).index(method)
            phase = pack["endpoint_phase_rad"][index]
            history = pack[key]
            iterations = int(pack["iterations"][index])
        path = compact
    else:
        with np.load(path, allow_pickle=False) as pack:
            phase = pack["phase_rad"]
            history_key = "history_phase_rad" if "history_phase_rad" in pack else "history_phase_tail_rad"
            history = pack[history_key]
            iterations = int(pack["iterations"])
    if not np.allclose(np.exp(1j*phase), np.exp(1j*endpoint), atol=1e-10, rtol=0):
        raise ValueError(f"History endpoint differs from displayed command: {path}")
    if history.ndim != 2 or history.shape[0] < 2:
        raise ValueError(f"Missing accepted history states: {path}")
    if not np.allclose(np.exp(1j*history[-1]), np.exp(1j*phase), atol=1e-10, rtol=0):
        raise ValueError(f"History does not end at phase_rad: {path}")
    return SimpleNamespace(phase_rad=phase, history_phase_rad=history,
                           iterations=iterations), path


def analyze(directory, destination, grid_size=51, history_root=None):
    directory, destination = Path(directory), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    setting_id = directory.name
    objective_list, endpoints, target, frequency = objectives(directory)
    q = np.linspace(-.30, .30, grid_size)
    sample_indices = np.rint(np.linspace(5, grid_size-6, 7)).astype(int)
    arrow_q = q[sample_indices]
    xx, yy = np.meshgrid(arrow_q, arrow_q, indexing="xy")
    planes, directions, descent_u, descent_v = [], [], [], []
    metadata_rows = []
    tails = {}
    for method, objective, endpoint in zip(METHODS, objective_list, endpoints):
        run, history_path = load_run(directory, method, endpoint, history_root)
        basis = _native_local_directions(objective, run)
        # Retain only the accepted suffix needed to reproduce the native axes.
        count = 2
        while True:
            tail = run.history_phase_rad[-count:]
            short_run = SimpleNamespace(phase_rad=run.phase_rad, history_phase_rad=tail)
            try:
                recovered = _native_local_directions(objective, short_run)
                matched = np.allclose(recovered, basis, rtol=0, atol=1e-12)
            except RuntimeError:
                matched = False
            if matched:
                tails[f"history_{method}_tail_rad"] = tail
                break
            if count >= len(run.history_phase_rad):
                raise AssertionError("Full accepted history failed native direction recovery")
            count = min(2*count, len(run.history_phase_rad))
        data = _native_relative_sections(objective, run, basis,
                                        line_points=101, plane_points=grid_size)
        d1, d2 = basis
        u, v = np.empty_like(xx), np.empty_like(yy)
        for index in np.ndindex(xx.shape):
            point = endpoint + xx[index]*d1 + yy[index]*d2
            _, gradient = objective.full_fun_grad(point)
            u[index] = -float(np.dot(gradient, d1))
            v[index] = -float(np.dot(gradient, d2))
        _, gradient = objective.full_fun_grad(endpoint)
        planes.append(data["plane"])
        directions.append(np.stack(basis))
        descent_u.append(u); descent_v.append(v)
        metadata_rows.append(dict(method=method, phase_fingerprint=digest_array(endpoint),
            accepted_history_file=str(history_path), iterations=run.iterations,
            compact_history_rows=int(len(tail)),
            terminal_gradient_norm=float(np.linalg.norm(gradient)),
            direction_gram=np.stack(basis) @ np.stack(basis).T,
            plane_min=float(data["plane"].min()), plane_max=float(data["plane"].max())))
        if method in ('FE', 'GFE'):
            filename = 'Single_FEplusg.npz' if method == 'GFE' else 'Single_FE.npz'
            with np.load(directory / 'commands' / filename) as command:
                metadata_rows[-1]['alpha_per_m'] = float(command['alpha_per_m']) if 'alpha_per_m' in command else 10.
    arrays = dict(methods=np.array(METHODS), q=data["q"], plane_delta_j=np.stack(planes),
        arrow_q=arrow_q, descent_u=np.stack(descent_u), descent_v=np.stack(descent_v),
        endpoint_phase_rad=endpoints, directions=np.stack(directions), target_m=target,
        frequency_hz=np.array(frequency),
        terminal_gradient_norm=np.array([x["terminal_gradient_norm"] for x in metadata_rows]),
        iterations=np.array([x["iterations"] for x in metadata_rows]), **tails)
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind != "U"):
        raise ValueError(f"Nonfinite native decision data: {setting_id}")
    save_npz(destination / f"{setting_id}_decision.npz", **arrays)
    for row in metadata_rows:
        row["direction_gram"] = row["direction_gram"].tolist()
    metadata = dict(setting_id=setting_id, methods=list(METHODS), target_m=target.tolist(),
        algorithm_source="hat_revision_pipeline.pipeline._native_local_directions and _native_relative_sections",
        direction_definition="u=-terminal full gradient/its L2 norm; v=gauge-aligned last accepted step orthogonalized to u and L2 normalized",
        half_range_l2_rad=.30, grid_size=grid_size,
        ordinate="J(phi_endpoint+s_gradient*u+s_perp*v) minus the sampled section minimum",
        arrow_definition="Negative analytic full gradient projected onto the displayed u,v plane; glyphs have equal length",
        scale="Linear, with each panel's original-main 98.5th percentile upper color limit",
        endpoints=metadata_rows)
    write_bytes(destination / f"{setting_id}_decision_metadata.json",
                (json.dumps(metadata, indent=2)+"\n").encode())
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "campaign")
    parser.add_argument("--history-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/decision")
    parser.add_argument("--settings", default=",".join(s["id"] for s in SETTINGS))
    parser.add_argument("--grid-size", type=int, default=51)
    args = parser.parse_args()
    for setting in args.settings.split(","):
        print(f"{setting}: native decision sections", flush=True)
        analyze(args.input_root / setting, args.output, args.grid_size, args.history_root)
        print(f"{setting}: saved", flush=True)
