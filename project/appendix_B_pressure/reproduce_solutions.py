"""Reproduce missing Appendix B commands, reusing every compatible cached run."""
from dataclasses import replace
from pathlib import Path
from rineng_content_id import content_identity
import json
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime/src"))
from hat_revision_pipeline.gorkov_core import (
    ArrayGeometry, Condition, MethodSpec, SingleTargetObjective, gauge_full,
)
from hat_revision_pipeline.sota import solve_corrected_gorkov_fe


def nominal_objective(config):
    geometry = ArrayGeometry.square(config["side"], config["pitch_m"])
    weights = np.diag(config["curvature_weights"])
    spec = MethodSpec(config["method"], config["alpha_per_m"],
                      config["beta_curvature_per_pa"], config["pressure_mode"], weights)
    objective = SingleTargetObjective(
        Condition("Appendix B", geometry.positions_m, tuple(config["target_m"]),
                  config["frequency_hz"], weights), spec,
        stencil_spacing_m=config["stencil_spacing_m"],
        smooth_pressure_relative=config["smooth_pressure_relative"])
    if config["sign"] == "historical_plus":
        objective.coefficients = replace(objective.coefficients,
            gradient_j_m2_pa2=-objective.coefficients.gradient_j_m2_pa2)
    elif config["sign"] != "corrected_minus":
        raise ValueError(config["sign"])
    return objective


def compute_nominal_missing(root=ROOT):
    root = Path(root).resolve()
    cases = json.loads((root / "data/nominal_cases.json").read_text())
    manifest = []
    for case in cases:
        case = dict(case)
        if case["method"] != "Conventional":
            from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
            case.update(solver="L-BFGS-B", fe_solver_policy=FE_SOLVER_REVISION)
            case["cache_key"] = case["cache_key"] + "_" + FE_SOLVER_REVISION
        key = case["cache_key"]
        path = root / "data/cache" / key
        if path.with_suffix(".json").is_file() and path.with_suffix(".npz").is_file():
            record = json.loads(path.with_suffix(".json").read_text())
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
                digest = content_identity(archive["terminal_phase_rad"].tobytes()).hexdigest()
            if digest != record["phase_content_id"]:
                raise ValueError(f"Invalid cached phase: {key}")
        else:
            with threadpool_limits(limits=1):
                objective = nominal_objective(case)
                initial = gauge_full(np.random.default_rng(case["seed"]).uniform(
                    -np.pi, np.pi, objective.n_transducers - 1))
                start = time.perf_counter()
                result = solve_corrected_gorkov_fe(objective, initial,
                    maxiter=case["maxiter"], gtol=case["gtol"])
                elapsed = time.perf_counter() - start
                raw = objective._raw_metrics(result.phase_rad, False)
                record = dict(case, iterations=int(result.iterations),
                    gradient_l2=float(result.gradient_norm), success=bool(result.success),
                    status_code=int(result.status), message=str(result.message),
                    objective=float(result.objective), pressure_pa=float(raw["pressure_abs_pa"]),
                    gorkov_force_n=float(raw["force_norm_n"]), command_time_s=elapsed,
                    phase_content_id=content_identity(result.phase_rad.tobytes()).hexdigest(),
                    initial_content_id=content_identity(initial.tobytes()).hexdigest())
                budgets = np.array([0, 100, 500, 1000, 10000])
                accepted = np.minimum(budgets, result.iterations)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path.with_suffix(".npz"), initial_phase_rad=initial,
                    terminal_phase_rad=result.phase_rad, checkpoint_budgets=budgets,
                    checkpoint_accepted_iterations=accepted,
                    checkpoint_phase_rad=np.stack([result.history_phase_rad[k] for k in accepted]),
                    history_objective=result.history_objective,
                    history_gradient_l2=result.history_gradient_norm,
                    history_wall_s=result.history_wall_s)
                path.with_suffix(".json").write_text(json.dumps(record, indent=2))
        manifest.append({"requested": case["requested"], "sign": case["sign"],
                         "method": case["method"], "cache_key": key,
                         "phase_content_id": record["phase_content_id"]})
    (root / "data/solution_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    compute_nominal_missing()
    import sensitivity_b
    sensitivity_b.compute_missing(ROOT)
