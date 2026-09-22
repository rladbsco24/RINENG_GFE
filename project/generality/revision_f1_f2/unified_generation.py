"""Run the complete nine-figure generality appendix in an isolated process.

Smoke/resume validates recorded numerical caches. Fresh smoke and full runs
synthesize commands using frozen coefficients and recorded proposal inputs,
validate the Triple commands with the independent finite-ka elastic model,
and regenerate fields, decision sections, ACS and figures. Each mode retains
its own source identity, cache, generated data and completion receipt.
"""
from __future__ import annotations

import argparse
import fingerprintlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCIENTIFIC_PROTOCOL_VERSION = 'rineng-generality-20260909-v1'
sys.path.insert(0, str(ROOT.parent / "generality_package/runtime/src"))
from appendix_settings import SETTINGS, geometry
from appendix_io import save_npz, write_bytes
from hat_revision_pipeline.cache import digest_array


def _json(path, value):
    write_bytes(Path(path), (json.dumps(value, indent=2, default=str) + "\n").encode())


def generation_audit(project_root=ROOT):
    """Describe the compact, fresh-generation closure without running it."""
    root = Path(project_root).resolve()
    manifest = json.loads((root / "data/recorded_proposals/manifest.json").read_text())
    with np.load(root / "data/recorded_proposals/initial_phase_bank.npz", allow_pickle=False) as bank:
        bank_rows = sum(len(bank[key]) for key in bank.files)
    recipes = pd.read_csv(root / "data/selected_triple_recipes.csv")
    weights = pd.read_csv(root.parent / "tables/selected_weights.csv")
    return dict(schema_version=2, settings=len(SETTINGS), proposal_trials=len(manifest["trials"]),
        proposal_bank_rows=bank_rows, single_weight_rows=len(weights), triple_recipe_rows=len(recipes),
        fresh_generation=True, historical_result_caches=False)


def _materialize_frozen_commands(root):
    """Recover command and geometry files exactly from deposited NPZ inputs."""
    out = root / "data/solution_correspondence/campaign"
    for setting in SETTINGS:
        sid = setting["id"]
        with np.load(root / "data/pressure" / f"{sid}_pressure.npz", allow_pickle=False) as data:
            destination = out / sid
            save_npz(destination / "array_geometry.npz",
                positions_m=data["positions_m"], normals=data["normals"],
                frequency_hz=data["frequency_hz"])
            for index, name in enumerate(data["methods"]):
                suffix = "FEplusg" if name == "GFE" else str(name)
                save_npz(destination / "commands" / f"Single_{suffix}.npz",
                    phase_rad=data["phase_rad"][index], target_m=data["target_m"],
                    alpha_per_m=data["alpha_per_m"][index])
    return out


def regenerate_acs_table(project_root=ROOT):
    """Aggregate all retained scalar traces; never manufacture phase iterates."""
    root = Path(project_root).resolve()
    folder = root / "data/compact_acs"
    protocol = json.loads((folder / "protocol.json").read_text())
    budgets = np.arange(int(protocol["displayed_iteration_max"]) + 1)
    rows = []
    for setting in SETTINGS:
        sid = setting["id"]
        samples = []
        count = None
        for path in sorted((folder / "traces").glob(f"{sid}_Conventional_seed*.npz")):
            with np.load(path, allow_pickle=False) as data:
                x = data["accepted_iteration"]
                y = data["cosine_similarity"]
                if not np.array_equal(x, np.arange(len(x))):
                    raise ValueError(f"Nonconsecutive accepted iteration grid: {path}")
                samples.append(y[np.minimum(budgets, len(y)-1)])
                count = len(data["terminal_phase_rad"])
        if len(samples) != int(protocol["paired_starts"]):
            raise ValueError(f"Missing retained trajectories for {sid}")
        mean = np.mean(samples, axis=0)
        rows.append(pd.DataFrame(dict(iteration_budget=budgets,
            setting_id=sid, frequency_hz=setting["frequency_hz"],
            trajectory_method="Conventional", budget_mean=mean,
            n_pairs=float(len(samples)), source_count=float(count))))
    result = pd.concat(rows, ignore_index=True)
    write_bytes(folder / "plotgrid.csv", result.to_csv(index=False).encode())
    return result


def prepare_cached_data(project_root=ROOT, *, regenerate_fields=False):
    """Prepare all accepted plot inputs from their deposited numerical caches.

    This function is intentionally mode-neutral: it is historical-cache
    preparation, never a population expansion. A caller must identify these
    figures as cached smoke results even inside a larger full-mode notebook.
    """
    root = Path(project_root).resolve()
    from single_correspondence import validate
    from final_benchmark_data import export_final_benchmark
    validation = validate(root / "data")
    command_root = _materialize_frozen_commands(root)
    started = time.perf_counter()
    table = regenerate_acs_table(root)
    if regenerate_fields:
        from pressure_data import extract
        for setting in SETTINGS:
            extract(setting["id"], root / "data/pressure", points=201,
                    input_root=command_root)
    benchmark = export_final_benchmark(root=root)
    record = dict(schema_version=1, evidence_mode="recorded_smoke_cache",
        optimizer_calls=0, finite_ka_calls=0, pressure_fields_regenerated=regenerate_fields,
        scalar_acs_aggregation_regenerated=True, acs_rows=len(table),
        command_correspondence_passed=validation["passed"],
        elapsed_s=time.perf_counter()-started, generation_audit=generation_audit(root),
        scope="Accepted commands, search interventions, equilibrium results and scalar trajectories retained.")
    _json(root / "data/unified_cache_preparation.json", record)
    return record


def run_campaign(output_root, *, mode="smoke", cache_policy="resume", workers=1, project_root=ROOT):
    """Launch a mode-isolated appendix process and return its active project.

    Smoke/resume validates and reuses the accepted numerical cache. Smoke/fresh
    and full mode generate data before figures using fixed selected recipes.
    Full mode never promotes archived smoke root outcomes into full evidence.
    """
    import shutil
    import subprocess
    if mode not in {"smoke", "full"} or cache_policy not in {"resume", "fresh"}:
        raise ValueError("Invalid mode or cache policy")
    source = Path(project_root).resolve()
    base = Path(output_root).resolve() / mode
    active = base / "revision_f1_f2"
    if active == source:
        raise ValueError("The active mode directory must be separate from bundled source")
    for folder in ("generality_package", "revision_f1_f2", "tables"):
        original = source.parent / folder
        destination = base / folder
        for path in original.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(original)
            target = destination / relative
            # Existing generated numerical files belong to the active campaign.
            # Copy code changes, but never overwrite them with bundled old fields.
            update_source = path.suffix == ".py" or path.name in {"generation_protocol.json", "selected_triple_recipes.csv", "selected_weights.csv", "manifest.json"}
            if not target.exists() or (update_source and target.read_bytes() != path.read_bytes()):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    command = [sys.executable, "-u", str(active / "unified_generation.py"),
               "--active-worker", "--mode", mode, "--cache-policy", cache_policy,
               "--workers", str(int(workers))]
    subprocess.run(command, cwd=base, check=True)
    receipt = json.loads((base / "generation_status.json").read_text())
    if (receipt["state"] != "completed" or receipt["mode"] != mode or
            receipt.get("scientific_protocol_version") != SCIENTIFIC_PROTOCOL_VERSION):
        raise RuntimeError("Generality campaign did not finish in the requested mode")
    return receipt


def _active_worker(mode, cache_policy, workers):
    import subprocess
    status = ROOT.parent / "generation_status.json"
    acs_path = ROOT / "data/compact_acs/protocol.json"
    acs = json.loads(acs_path.read_text(encoding="utf-8"))
    acs.update(matching_rule="fixed_same_setting_endpoint_set", reference_method="FE_or_GFE")
    _json(acs_path, acs)
    from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
    progress = dict(fe_solver_policy=FE_SOLVER_REVISION,
        mode=mode, cache_policy=cache_policy, workers=int(workers),
        scientific_protocol_version=SCIENTIFIC_PROTOCOL_VERSION,
        active_project=str(ROOT), state="running", started_at=time.time())
    if status.exists() and cache_policy == "resume":
        old = json.loads(status.read_text())
        if (old.get("state") == "completed" and old.get("mode") == mode and
                old.get("scientific_protocol_version") == SCIENTIFIC_PROTOCOL_VERSION and
                old.get("fe_solver_policy") == FE_SOLVER_REVISION):
            # Re-rendering is inexpensive and reflects the current numerical inputs.
            subprocess.run([sys.executable, str(ROOT / "render_all.py")], check=True)
            return old
    _json(status, progress)
    try:
        from unified_single_generation import generate_single_data
        from unified_triple_generation import generate_triple_data, activate_generated_triple
        progress["stage"] = "Single synthesis from compact explicit inputs"
        _json(status, progress)
        progress["single"] = generate_single_data(ROOT, mode=mode, cache_policy=cache_policy)
        progress["stage"] = "Triple synthesis and finite-ka validation"
        _json(status, progress)
        triple = generate_triple_data(ROOT / "data/generated_campaign", project_root=ROOT,
                                      mode=mode, cache_policy=cache_policy)
        activate_generated_triple(triple["output_root"], ROOT)
        progress["triple"] = triple
        progress["evidence_mode"] = "generated_" + mode
        progress["stage"] = "figure rendering"
        _json(status, progress)
        subprocess.run([sys.executable, str(ROOT / "render_all.py")], check=True)
        progress.update(state="completed", stage="completed", completed_at=time.time(), figure_count=9)
        _json(status, progress)
        print(json.dumps(progress, indent=2), flush=True)
        return progress
    except Exception as exc:
        progress.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        _json(status, progress)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--regenerate-fields", action="store_true")
    parser.add_argument("--active-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cache-policy", choices=("resume", "fresh"), default="resume")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(generation_audit(), indent=2))
    elif args.prepare_cache:
        print(json.dumps(prepare_cached_data(regenerate_fields=args.regenerate_fields), indent=2))
    elif args.active_worker:
        _active_worker(args.mode, args.cache_policy, args.workers)
    elif args.output_root:
        print(json.dumps(run_campaign(args.output_root, mode=args.mode,
              cache_policy=args.cache_policy, workers=args.workers), indent=2))
    else:
        parser.error("Provide --output-root or --audit")
