"""Freeze one FE/GFE display identity per setting from equilibrium displacement.

This presentation choice uses the already tuned seed-260828 endpoints. It does
not retune alpha, alter any command, or change the FE/GFE ACS reference bank.
"""
from __future__ import annotations

import argparse
import fingerprintlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from appendix_io import write_bytes
from appendix_settings import SETTINGS

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "generality_package/runtime/src"))
from hat_revision_pipeline.cache import digest_array

PRESSURE_COLUMN = "pressure_abs_at_exact_equilibrium_pa"
DISPLACEMENT_COLUMN = "displacement_um"
CRITERION = "Strictly lower finite-ka equilibrium displacement among frozen tuned FE/GFE commands at seed 260828."


def selection_table(data_root=None):
    """Return one auditable display choice for every unique setting."""
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    correspondence = data_root / 'solution_correspondence/selection.csv'
    if correspondence.exists():
        return pd.read_csv(correspondence)
    source = data_root / "equilibrium_pressure/outcomes.csv"
    outcomes = pd.read_csv(source)
    rows = []
    for setting in SETTINGS:
        sid = setting["id"]
        pair = outcomes[outcomes.setting_id.eq(sid) & outcomes.seed.eq(260828)
                        & outcomes.method.isin(("FE", "GFE"))]
        if len(pair) != 2 or set(pair.method) != {"FE", "GFE"}:
            raise ValueError(f"{sid}: exactly one frozen FE and GFE endpoint required")
        pair = pair.set_index("method")
        if not pair.finite_ka_root_found.astype(str).str.lower().eq("true").all():
            raise ValueError(f"{sid}: both recorded equilibria must be resolved")
        pressure = pair[PRESSURE_COLUMN].astype(float)
        if not np.isfinite(pressure).all() or (pressure < 0).any():
            raise ValueError(f"{sid}: invalid equilibrium pressure")
        displacement = pair[DISPLACEMENT_COLUMN].astype(float)
        if not np.isfinite(displacement).all() or (displacement < 0).any():
            raise ValueError(f"{sid}: invalid equilibrium displacement")
        if displacement.FE == displacement.GFE:
            raise ValueError(f"{sid}: exact displacement tie requires an explicit display rule")
        method = "FE" if displacement.FE < displacement.GFE else "GFE"
        chosen = pair.loc[method]
        rows.append(dict(setting_id=sid, method=method, seed=260828,
            FE_equilibrium_displacement_um=float(displacement.FE),
            GFE_equilibrium_displacement_um=float(displacement.GFE),
            equilibrium_displacement_um=float(displacement[method]),
            FE_equilibrium_pressure_pa=float(pressure.FE),
            GFE_equilibrium_pressure_pa=float(pressure.GFE),
            equilibrium_pressure_pa=float(pressure[method]),
            alpha_per_m=float(chosen.alpha_per_m),
            phase_fingerprint=str(chosen.phase_fingerprint), criterion=CRITERION))
    return pd.DataFrame(rows)


def load_selection(data_root=None):
    """Map setting ID to its one displacement-based FE/GFE display identity."""
    table = selection_table(data_root)
    return dict(zip(table.setting_id, table.method))


def validate_alignment(data_root=None, setting_ids=None, include_all_methods=False):
    """Check that field, isobar, decision and equilibrium use identical commands."""
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    if (data_root / 'solution_correspondence/selection.csv').exists():
        from single_correspondence import validate
        return validate(data_root, setting_ids)
    choices = selection_table(data_root).set_index("setting_id")
    outcomes = pd.read_csv(data_root / "equilibrium_pressure/outcomes.csv")
    setting_ids = list(choices.index) if setting_ids is None else list(setting_ids)
    records = []
    for sid in setting_ids:
        representative = str(choices.loc[sid, "method"])
        methods = ("Conventional", "FE", "GFE") if include_all_methods else ("Conventional", representative)
        pressure_path = data_root / "pressure" / f"{sid}_pressure.npz"
        decision_path = data_root / "decision" / f"{sid}_decision.npz"
        with np.load(pressure_path, allow_pickle=False) as pressure, np.load(
                decision_path, allow_pickle=False) as decision:
            if "phase_rad" not in pressure:
                raise ValueError(f"{sid}: pressure command phases are required for alignment")
            if not np.array_equal(pressure["target_m"], decision["target_m"]):
                raise ValueError(f"{sid}: pressure and decision target mismatch")
            if float(pressure["frequency_hz"]) != float(decision["frequency_hz"]):
                raise ValueError(f"{sid}: pressure and decision frequency mismatch")
            for method in methods:
                pi = list(pressure["methods"]).index(method)
                di = list(decision["methods"]).index(method)
                pressure_hash = digest_array(pressure["phase_rad"][pi])
                decision_hash = digest_array(decision["endpoint_phase_rad"][di])
                row = outcomes[outcomes.setting_id.eq(sid) & outcomes.method.eq(method)
                               & outcomes.seed.eq(260828)]
                if len(row) != 1:
                    raise ValueError(f"{sid} {method}: one benchmark endpoint is required")
                row = row.iloc[0]
                benchmark_hash = str(row.phase_fingerprint)
                if len({pressure_hash, decision_hash, benchmark_hash}) != 1:
                    raise ValueError(f"{sid} {method}: field/decision/equilibrium command mismatch")
                if method == representative and pressure_hash != choices.loc[sid, "phase_fingerprint"]:
                    raise ValueError(f"{sid}: representative command identity mismatch")
                target = np.array([row.target_x_m, row.target_y_m, row.target_z_m], dtype=float)
                if not np.array_equal(pressure["target_m"], target):
                    raise ValueError(f"{sid} {method}: field and benchmark target mismatch")
                if float(pressure["frequency_hz"]) != float(row.frequency_hz):
                    raise ValueError(f"{sid} {method}: field and benchmark frequency mismatch")
                records.append(dict(setting_id=sid, method=method,
                    representative_method=representative, phase_fingerprint=pressure_hash,
                    pressure_decision_equilibrium_hash_match=True,
                    target_and_frequency_match=True,
                    pressure_amplitude_fingerprint=digest_array(pressure["amplitude_pa"][pi]),
                    decision_plane_fingerprint=digest_array(decision["plane_delta_j"][di]),
                    decision_descent_u_fingerprint=digest_array(decision["descent_u"][di]),
                    decision_descent_v_fingerprint=digest_array(decision["descent_v"][di])))
    return dict(passed=True, criterion=CRITERION, settings=setting_ids,
        audited_methods="Conventional, FE and GFE" if include_all_methods else "Conventional and the named FE/GFE representative",
        field_and_isobars="The same amplitude_pa array is used for the named FE/GFE background and its isobars.",
        records=records)


def write_alignment(data_root=None):
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    result = validate_alignment(data_root, include_all_methods=True)
    path = data_root / "presentation_selection_alignment.json"
    write_bytes(path, (json.dumps(result, indent=2)+"\n").encode())
    return path


def write_selection(data_root=None):
    data_root = Path(data_root) if data_root is not None else ROOT / "data"
    table = selection_table(data_root)
    correspondence = data_root / 'solution_correspondence/selection.csv'
    if correspondence.exists():
        csv_path, json_path = data_root/'presentation_selection.csv', data_root/'presentation_selection.json'
        write_bytes(csv_path, table.to_csv(index=False).encode())
        payload = dict(criterion=str(table.iloc[0].criterion), seed=260828,
            source='solution_correspondence/selection.csv',
            source_fingerprint=fingerprintlib.fingerprint(correspondence.read_bytes()).hexdigest(),
            scope='Single fields and native decision planes use the terminal Conventional command and nearest FE and GFE members of the frozen bank; ACS headings name the closer of these references.',
            benchmark_scope='Triple retains all five methods independently.',
            ACS_reference='Unchanged fixed same-setting set of four FE plus four GFE endpoints.',
            choices=table.to_dict(orient='records'))
        write_bytes(json_path,(json.dumps(payload,indent=2)+'\n').encode())
        return csv_path, json_path
    source = data_root / "equilibrium_pressure/outcomes.csv"
    csv_path = data_root / "presentation_selection.csv"
    json_path = data_root / "presentation_selection.json"
    write_bytes(csv_path, table.to_csv(index=False).encode())
    payload = dict(criterion=CRITERION, seed=260828,
        source="equilibrium_pressure/outcomes.csv",
        source_fingerprint=fingerprintlib.fingerprint(source.read_bytes()).hexdigest(),
        alpha="Previously frozen per-setting, per-method tuning remains unchanged.",
        scope="One displacement-based identity used consistently in pressure, decision and equilibrium displacement/pressure displays.",
        ACS_reference="Unchanged fixed same-setting set of four FE plus four GFE endpoints.",
        choices=table.to_dict(orient="records"))
    write_bytes(json_path, (json.dumps(payload, indent=2)+"\n").encode())
    return csv_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--audit-alignment", action="store_true")
    args = parser.parse_args()
    for path in write_selection(args.data_root):
        print(path)
    if args.audit_alignment:
        print(write_alignment(args.data_root))
