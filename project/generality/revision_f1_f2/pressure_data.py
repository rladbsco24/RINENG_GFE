"""Current Single pressure fields for the appendix; no optimisation is run."""
from pathlib import Path
import argparse
import json
import os
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'generality_package' / 'runtime' / 'src'))
from hat_revision_pipeline.gorkov_core import ArrayGeometry, transfer_matrix, SOURCE_SCALE_PA_M_PER_MURATA_UNIT
from hat_revision_pipeline.cache import digest_array
from appendix_settings import SETTINGS
from appendix_io import save_npz, write_bytes

METHODS = ('Conventional', 'FE', 'GFE')

def source_dir(sid):
    current = ROOT / 'campaign' / sid
    if (current / 'commands' / 'Single_FEplusg.npz').exists():
        return current
    old = ROOT.parent / 'generality_package' / 'outputs' / sid
    return old if (old / 'array_geometry.npz').exists() else current

def extract(sid, destination, points=201, input_root=None):
    src = Path(input_root) / sid if input_root is not None else source_dir(sid)
    setting = next(s for s in SETTINGS if s['id'] == sid)
    with np.load(src / 'array_geometry.npz') as z:
        positions, normals = z['positions_m'], z['normals']
    phases, targets, alphas = {}, {}, {}
    for method in METHODS:
        filename = 'Single_FEplusg.npz' if method == 'GFE' else f'Single_{method}.npz'
        with np.load(src / 'commands' / filename) as z:
            phases[method], targets[method] = z['phase_rad'].copy(), z['target_m'].copy()
            alphas[method] = float(z['alpha_per_m']) if 'alpha_per_m' in z else (0. if method == 'Conventional' else 10.)
    target = targets['GFE']
    assert all(np.array_equal(t, target) for t in targets.values())
    wavelength = 343. / setting['frequency_hz']
    q = np.linspace(-.5, .5, points)
    xx, zz = np.meshgrid(q, q)
    xyz = target + wavelength * np.c_[xx.ravel(), np.zeros(xx.size), zz.ravel()]
    commands = np.exp(1j * np.array([phases[m] for m in METHODS])).T
    field = np.empty((len(xyz), len(METHODS)), dtype=complex)
    geom = ArrayGeometry(positions, normals)
    for start in range(0, len(xyz), 2048):
        h = transfer_matrix(xyz[start:start+2048], geom, setting['frequency_hz'],
                            source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT)
        field[start:start+len(h)] = h @ commands
    amplitude = np.abs(field).T.reshape(len(METHODS), points, points)
    # A single pooled scale per setting is shared by all three fields/isobars.
    pstar = float(np.max(amplitude))
    rows = []
    for i, method in enumerate(METHODS):
        a, ref = amplitude[i].ravel(), amplitude[2].ravel()
        rows.append(dict(setting_id=sid, method=method, reference_method='GFE',
            pressure_scale_pa=pstar, field_amplitude_cosine=float(a@ref/(np.linalg.norm(a)*np.linalg.norm(ref))),
            field_amplitude_relative_l2=float(np.linalg.norm(a-ref)/np.linalg.norm(ref)),
            pressure_at_target_pa=float(amplitude[i,points//2,points//2]),
            phase_fingerprint=digest_array(phases[method]), frequency_hz=setting['frequency_hz'], alpha_per_m=alphas[method],
            reference_phase_fingerprint=digest_array(phases['GFE']),
            field_plane='XZ at intended target y; target-centered coordinates / wavelength',
            field_half_width_lambda=.5, grid_points=points, source_directory=str(src)))
    destination.mkdir(parents=True, exist_ok=True)
    save_npz(destination / f'{sid}_pressure.npz',
        coordinate_lambda=q, amplitude_pa=amplitude, complex_pressure=field.T.reshape(amplitude.shape),
        pressure_scale_pa=pstar, target_m=target, wavelength_m=wavelength,
        methods=np.array(METHODS), phase_rad=np.array([phases[m] for m in METHODS]),
        positions_m=positions, normals=normals, frequency_hz=setting['frequency_hz'],
        alpha_per_m=np.array([alphas[m] for m in METHODS]))
    write_bytes(destination / f'{sid}_pressure_metrics.csv', pd.DataFrame(rows).to_csv(index=False).encode())
    print(f'{sid}: pressure fields exported, p*={pstar:.3f} Pa', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default=','.join(s['id'] for s in SETTINGS))
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'pressure')
    parser.add_argument('--input-root', type=Path)
    args = parser.parse_args()
    for sid in args.settings.split(','):
        extract(sid, args.output, input_root=args.input_root)
