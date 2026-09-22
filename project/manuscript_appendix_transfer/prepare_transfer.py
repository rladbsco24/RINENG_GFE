"""Validate the transfer package and export existing C3 rows; never solve.

Run from any directory: python /path/to/manuscript_appendix_transfer/prepare_transfer.py
This script uses only the Python standard library and performs no pickle loads,
objective evaluations, optimizer steps, network calls or figure regeneration.
"""
from pathlib import Path
import csv

from rineng_content_id import content_identity
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    identity = json.loads((HERE / "source_identity.json").read_text())
    pdf = ROOT / identity["source_in_package"]
    digest = content_identity(pdf.read_bytes()).hexdigest()
    if digest != identity["content_id"]:
        raise ValueError("Received manuscript PDF does not match the audited source")
    plan = json.loads((HERE / "transfer_experiments.json").read_text())
    src = ROOT / "appendix_outputs/C_GFE/C3_warm_cold_results.csv"
    with src.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        available = set(reader.fieldnames or ())
    cols = ["initialization", "alpha_per_m", "seed", "phase_content_id", "initial_phase_content_id",
            "iterations", "evaluations", "maxiter", "objective", "terminal_gradient_l2",
            "optimizer_success", "optimizer_status", "optimizer_message", "command_time_s",
            "source", "cache_status", "evidence_status", "command_cosine_to_fixed_GFE",
            "field_cosine_to_fixed_GFE", "command_projector_distance", "reference_phase_content_id",
            "reference_seed", "reference_alpha_per_m", "field_point_count", "force_target_z_n",
            "finite_ka_displacement_a", "finite_ka_stiffness_min_n_m", "finite_ka_root_class",
            "validation_source", "previous_alpha_per_m"]
    missing = sorted(set(cols) - available)
    if missing:
        raise ValueError(f"Current C3 export is missing required columns: {missing}")
    out = HERE / "current_GFE_cold_warm_transfer.csv"
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    large = [r for r in rows if r["initialization"] in ("Cold", "Warm")]
    result = {
        "operation": "read existing manuscript and tabular results; export only",
        "new_numerical_solves": 0,
        "source_pdf_content_id": digest,
        "current_C3_source_content_id": content_identity(src.read_bytes()).hexdigest(),
        "current_C3_rows": len(rows),
        "current_C3_large_alpha_rows": len(large),
        "large_alpha_precision_loss_stops": sum("precision loss" in r["optimizer_message"].lower() for r in large),
        "experiments": [{"id": e["id"], "status": e["status"]} for e in plan["experiments"]],
        "cache_namespace_file_counts": {
            name: len(list((ROOT / name).glob("*.pkl")))
            for e in plan["experiments"] for name in e.get("current_cache_namespaces", [])
        },
        "scope": "Current C3 is an initialization adaptation; original D optimizer runs and E1 derivative-spacing grid remain unexecuted in GFE."
    }
    (HERE / "transfer_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
