"""Explicit, source-controlled production design; importing it runs no experiments."""
from __future__ import annotations

import json
from pathlib import Path

SINGLE30 = (172817509, 425727192, 469986874, 519693606, 681979972,
            693201818, 713854823, 731661603, 764705224, 991245467,
            *range(260828, 260848))
TRIPLE30 = tuple(range(260869, 260899))
SENS_SINGLE10 = SINGLE30[:10]
TRIPLE10 = TRIPLE30[:10]
B10 = tuple(range(260905, 260915))
PROTOCOL_VERSION = "gfe-production-20260908-v2"

DEFAULTS = {
    "protocol_version": PROTOCOL_VERSION,
    "campaign_id": "gfe-final-production-v2",
    "benchmark_campaign_id": "gfe-compact-v3-task-specific-20260920",
    "device": "cpu",
    "workers": 3,
    "blas_threads": 1,
    "single_seeds": list(SINGLE30),
    "triple_seeds": list(TRIPLE30),
    "sensitivity_single_seeds": list(SENS_SINGLE10),
    "sensitivity_triple_seeds": list(TRIPLE10),
    "formulation_seeds": list(B10),
    "single_maxiter": 10000,
    "triple_maxiter": 30000,
    "gtol": 1e-8,
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20260906,
    "pressure_points": 181,
    "objective_plane_points": 101,
    "xz_x_points": 7,
    "xz_z_points": 7,
    "xz_restarts": 3,
    "exact_lmax": 8,
    "exact_fit_shape": [3, 12, 24],
    "exact_surface_shape": [20, 40],
    "root_start_offsets_a": [0.5, 1.0, 1.5],
    "root_max_nfev": 70,
    "root_residual_tolerance": 0.001,
    "deterministic_timing_repeats": 30,
    "discretization_timing_repeats": 30,
    "generality": "ACTIVE: Appendix G1-G9",
    "expand_rhg": False,
    "expand_d3_population": False,
}


def load_settings(root: str | Path, overrides: dict | None = None) -> dict:
    """Load the declared design, rejecting accidental edits to scientific controls."""
    result = dict(DEFAULTS)
    path = Path(root) / "production_config.json"
    if path.exists():
        result.update(json.loads(path.read_text()))
    if overrides:
        result.update(overrides)
    unknown = set(result) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown production settings: {sorted(unknown)}")
    adjustable = {"device", "workers", "blas_threads", "campaign_id", "benchmark_campaign_id"}
    changed = [key for key in DEFAULTS if key not in adjustable and result[key] != DEFAULTS[key]]
    if changed:
        raise ValueError("The frozen production design was changed: " + ", ".join(changed))
    if result["device"] not in {"cpu", "gpu", "auto"}:
        raise ValueError("device must be cpu, gpu, or auto")
    for key in ("workers", "blas_threads"):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(result["campaign_id"], str) or not result["campaign_id"].strip():
        raise ValueError("campaign_id must be a nonempty string")
    if not isinstance(result["benchmark_campaign_id"], str) or not result["benchmark_campaign_id"].strip():
        raise ValueError("benchmark_campaign_id must be a nonempty string")
    return result
