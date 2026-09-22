#!/usr/bin/env python3
"""Benchmark mechanics-aware global-offset rounding on 15 FE target positions.

The script is intentionally separate from the production notebook generator.
It regenerates or reuses the same deterministic continuous FE endpoints used by
``FE_discretization_methods_rapid_implementation``, stores their phases for
reuse, and compares:

* fixed gauge nearest rounding;
* FE-objective best-global-offset rounding; and
* mechanics-aware best-global-offset rounding.

Validated metrics are equilibrium shift from the continuous equilibrium,
stable-root survival, weakest-stiffness retention, and transition runtime.
"""

from __future__ import annotations

import argparse
import gc

from rineng_content_id import content_identity
import json
import os
import platform
import sys
import time
from pathlib import Path

for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares, root

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from inputs.hat_geometry_branch_benchmark import (
    BRANCH_FORCE_SQUARED_TERM_WEIGHT,
    BRANCH_PRESSURE_SQUARED_TERM_WEIGHT,
    BRANCH_SOFTMIN_TAU,
    BRANCH_STIFFNESS_WEIGHT,
    FORCE_SQUARED_TERM_WEIGHT,
    FORCE_TERM_WEIGHT,
    FREQUENCY,
    Geometry,
    build_quadratic_model,
    optimize,
    square_points,
)
from prototypes.discrete_solver import (
    ObjectiveSpec,
    TransitionProblem,
    best_global_offset_rounding,
    fixed_gauge_nearest,
    gauge_fix_phase,
)
from prototypes.mechanics_aware_offset import (
    build_local_mechanics_cache,
    mechanics_aware_best_offset_rounding,
    reference_from_continuous_command,
)


LEVELS = (4, 8, 10, 16, 32, 128, 2048)
FD_H_M = 0.0005
ROOT_H_M = 0.0005
ROOT_WINDOW_M = 0.5 * (343.0 / FREQUENCY)
BASE_SEED = 20260902
CALIBRATION_SEED = 20260825
CALIBRATION_SAMPLES = 48
MAXITER = 500
INITIALIZATIONS = 3
SOURCE_ENDPOINT_PATH = (
    WORKSPACE_ROOT
    / "FE_discretization_methods_rapid_implementation"
    / "continuous_FE_endpoints.csv"
)
ROOT_START_OFFSETS_M = np.vstack(
    [np.zeros(3), 0.00075 * np.eye(3), -0.00075 * np.eye(3)]
)


def fe_spec() -> ObjectiveSpec:
    return ObjectiveSpec(
        method="force_equilibrium",
        pressure_weight=25.0,
        force_weight=FORCE_TERM_WEIGHT,
        force_squared_weight=FORCE_SQUARED_TERM_WEIGHT,
        branch_pressure_squared_weight=BRANCH_PRESSURE_SQUARED_TERM_WEIGHT,
        branch_force_squared_weight=BRANCH_FORCE_SQUARED_TERM_WEIGHT,
        branch_stiffness_weight=BRANCH_STIFFNESS_WEIGHT,
        branch_softmin_tau=BRANCH_SOFTMIN_TAU,
    )


def target_specs() -> list[tuple[str, np.ndarray]]:
    lateral = ((0.0, 0.0), (0.010, 0.0), (-0.010, 0.0), (0.0, 0.010), (0.0, -0.010))
    output: list[tuple[str, np.ndarray]] = []
    for z_m in (0.045, 0.050, 0.055):
        for x_m, y_m in lateral:
            case_id = (
                f"X{int(round(1e3 * x_m)):+d}_"
                f"Y{int(round(1e3 * y_m)):+d}_"
                f"Z{int(round(1e3 * z_m)):02d}"
            )
            output.append((case_id, np.asarray([x_m, y_m, z_m], dtype=float)))
    return output


def source_selected_initializations() -> dict[str, int]:
    """Reuse the exact branch choices made by the executed companion notebook."""

    if not SOURCE_ENDPOINT_PATH.exists():
        return {}
    frame = pd.read_csv(SOURCE_ENDPOINT_PATH)
    required = {"case_id", "selected_initialization"}
    if not required.issubset(frame.columns):
        return {}
    return {
        str(row.case_id): int(row.selected_initialization)
        for row in frame.itertuples()
    }


def direct_u_many(geometry: Geometry, phase: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    from inputs.hat_geometry_branch_benchmark import K1, K2, transfer_matrix

    points = np.atleast_2d(np.asarray(points_m, dtype=float))
    blocks = [points]
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = ROOT_H_M
        blocks.extend([points + shift, points - shift])
    pressure = transfer_matrix(np.vstack(blocks), geometry) @ np.exp(1j * phase)
    count = len(points)
    center = pressure[:count]
    gradient = np.empty((count, 3), dtype=np.complex128)
    cursor = count
    for axis in range(3):
        gradient[:, axis] = (
            pressure[cursor : cursor + count]
            - pressure[cursor + count : cursor + 2 * count]
        ) / (2.0 * ROOT_H_M)
        cursor += 2 * count
    return K1 * np.abs(center) ** 2 - K2 * np.sum(np.abs(gradient) ** 2, axis=1)


def direct_force(geometry: Geometry, phase: np.ndarray, point_m: np.ndarray) -> np.ndarray:
    point = np.asarray(point_m, dtype=float)
    probes = []
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = ROOT_H_M
        probes.extend([point + shift, point - shift])
    potential = direct_u_many(geometry, phase, np.asarray(probes))
    return -np.asarray(
        [
            (potential[2 * axis] - potential[2 * axis + 1]) / (2.0 * ROOT_H_M)
            for axis in range(3)
        ]
    )


def validate_root(model: object, phase: np.ndarray) -> dict[str, object]:
    geometry = model.geometry
    force_scale = model.scales["force"]
    scaled_force = lambda position: direct_force(geometry, phase, position) / force_scale
    candidates: list[tuple[np.ndarray, bool, float, str]] = []
    for offset in ROOT_START_OFFSETS_M:
        solution = root(
            scaled_force,
            geometry.target + offset,
            method="hybr",
            options={"xtol": 1e-9, "maxfev": 90},
        )
        if np.all(np.isfinite(solution.x)) and (
            np.linalg.norm(solution.x - geometry.target) <= ROOT_WINDOW_M
        ):
            candidates.append(
                (
                    np.asarray(solution.x),
                    bool(solution.success),
                    float(np.linalg.norm(scaled_force(solution.x))),
                    "root",
                )
            )
    good = [candidate for candidate in candidates if candidate[1] and candidate[2] <= 1e-5]
    if not good:
        lower = geometry.target - ROOT_WINDOW_M
        upper = geometry.target + ROOT_WINDOW_M
        solution = least_squares(
            scaled_force,
            geometry.target,
            bounds=(lower, upper),
            x_scale=1e-3,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=150,
        )
        if np.all(np.isfinite(solution.x)):
            candidates.append(
                (
                    np.asarray(solution.x),
                    bool(solution.success),
                    float(np.linalg.norm(scaled_force(solution.x))),
                    "least_squares",
                )
            )
    good = [candidate for candidate in candidates if candidate[1] and candidate[2] <= 1e-5]
    pool = good or candidates
    if not pool:
        return {
            "root_position_m": np.full(3, np.nan),
            "root_converged": False,
            "root_stable": False,
            "root_residual_scaled": np.inf,
            "root_stiffness_eigenvalues_n_m": np.full(3, np.nan),
            "root_solver": "none",
        }
    equilibrium, solver_ok, residual, solver_name = min(
        pool, key=lambda candidate: np.linalg.norm(candidate[0] - geometry.target)
    )
    jacobian = np.empty((3, 3), dtype=float)
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = ROOT_H_M
        jacobian[:, axis] = (
            direct_force(geometry, phase, equilibrium + shift)
            - direct_force(geometry, phase, equilibrium - shift)
        ) / (2.0 * ROOT_H_M)
    stiffness = -0.5 * (jacobian + jacobian.T)
    eigenvalues = np.linalg.eigvalsh(stiffness)
    converged = bool(solver_ok and residual <= 1e-5)
    return {
        "root_position_m": np.asarray(equilibrium),
        "root_converged": converged,
        "root_stable": bool(converged and eigenvalues[0] > 0.0),
        "root_residual_scaled": float(residual),
        "root_stiffness_eigenvalues_n_m": eigenvalues,
        "root_solver": solver_name,
    }


def fingerprint_payload() -> dict[str, object]:
    source_selection = source_selected_initializations()
    return {
        "schema": 2,
        "target_specs": [(case_id, target.tolist()) for case_id, target in target_specs()],
        "array": {"side": 16, "pitch_m": 0.010},
        "fd_h_m": FD_H_M,
        "root_h_m": ROOT_H_M,
        "base_seed": BASE_SEED,
        "calibration_seed": CALIBRATION_SEED,
        "calibration_samples": CALIBRATION_SAMPLES,
        "maxiter": MAXITER,
        "initializations": INITIALIZATIONS,
        "source_selected_initializations": source_selection,
        "objective": "force_equilibrium",
        "force_weight": FORCE_TERM_WEIGHT,
    }


def fingerprint() -> str:
    from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
    serialized = json.dumps(dict(fingerprint_payload(), fe_solver_policy=FE_SOLVER_REVISION, evaluator_backend="compact-piston-stencil-v1"), sort_keys=True).encode("utf-8")
    return content_identity(serialized).hexdigest()


def build_models_and_phases(
    output_dir: Path,
    force_rebuild: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray], pd.DataFrame]:
    phase_path = output_dir / "continuous_FE_phases.npz"
    metadata_path = output_dir / "continuous_FE_phase_metadata.json"
    endpoint_path = output_dir / "continuous_FE_endpoints.csv"
    expected_fingerprint = fingerprint()
    cached = False
    phases: dict[str, np.ndarray] = {}
    if phase_path.exists() and metadata_path.exists() and endpoint_path.exists() and not force_rebuild:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == expected_fingerprint:
            with np.load(phase_path) as archive:
                phases = {key: np.asarray(archive[key], dtype=float) for key in archive.files}
            cached = len(phases) == len(target_specs())

    positions = square_points(16, pitch=0.010)
    normals = np.tile([0.0, 0.0, 1.0], (len(positions), 1))
    models: dict[str, object] = {}
    endpoint_rows: list[dict[str, object]] = []
    previous_endpoints = pd.read_csv(endpoint_path) if cached else None
    source_selection = source_selected_initializations()
    for case_index, (case_id, target) in enumerate(target_specs()):
        geometry = Geometry(
            case_id,
            positions,
            normals,
            target,
            "single-sided planar lattice",
            "16x16, 10 mm pitch",
        )
        model = build_quadratic_model(
            geometry,
            h=FD_H_M,
            calibration_samples=CALIBRATION_SAMPLES,
            seed=CALIBRATION_SEED + case_index,
        )
        models[case_id] = model
        if cached:
            phase = phases[case_id]
            row = previous_endpoints.loc[previous_endpoints.case_id == case_id].iloc[0].to_dict()
            endpoint_rows.append(row)
            continue

        candidates = []
        initializations = (
            (source_selection[case_id],)
            if case_id in source_selection
            else range(INITIALIZATIONS)
        )
        for initialization in initializations:
            rng = np.random.default_rng(BASE_SEED + 1000 * case_index + initialization)
            phase0 = gauge_fix_phase(rng.uniform(-np.pi, np.pi, geometry.n))
            solve_row, phase, _ = optimize(
                model,
                "force_equilibrium",
                phase0,
                MAXITER,
                keep_history=False,
            )
            candidates.append((solve_row, phase, initialization))
        converged = [candidate for candidate in candidates if candidate[0]["stationary"]]
        if not converged:
            raise RuntimeError(f"No stationary FE endpoint for {case_id}")
        solve_row, phase, initialization = min(
            converged, key=lambda candidate: candidate[0]["objective"]
        )
        root_metrics = validate_root(model, phase)
        if not root_metrics["root_converged"]:
            raise RuntimeError(f"Continuous root failed for {case_id}")
        phases[case_id] = phase.copy()
        equilibrium = np.asarray(root_metrics["root_position_m"])
        eigenvalues = np.asarray(root_metrics["root_stiffness_eigenvalues_n_m"])
        endpoint_rows.append(
            {
                "case_id": case_id,
                "target_x_mm": 1e3 * target[0],
                "target_y_mm": 1e3 * target[1],
                "target_z_mm": 1e3 * target[2],
                "selected_initialization": initialization,
                **solve_row,
                "root_x_m": equilibrium[0],
                "root_y_m": equilibrium[1],
                "root_z_m": equilibrium[2],
                "root_target_mm": 1e3 * np.linalg.norm(equilibrium - target),
                "root_converged": root_metrics["root_converged"],
                "root_stable": root_metrics["root_stable"],
                "root_stiffness_min": eigenvalues[0],
                "root_stiffness_mid": eigenvalues[1],
                "root_stiffness_max": eigenvalues[2],
            }
        )
        print("continuous", case_id, "init", initialization, "objective", solve_row["objective"])

    endpoints = pd.DataFrame(endpoint_rows).sort_values("case_id").reset_index(drop=True)
    if not cached:
        np.savez_compressed(phase_path, **phases)
        endpoints.to_csv(endpoint_path, index=False)
        metadata = {
            "fingerprint": expected_fingerprint,
            "payload": fingerprint_payload(),
            "model_regeneration": (
                "Models are deterministically regenerated from target, array, finite-difference, "
                "calibration-sample, and calibration-seed metadata. Dense matrices are omitted "
                "to avoid a roughly 0.17-GB duplicate archive."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return models, phases, endpoints


def timed(function, repeats: int) -> tuple[object, dict[str, float | int]]:
    function()
    gc.collect()
    samples = []
    last = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        last = function()
        samples.append((time.perf_counter_ns() - started) * 1e-6)
    values = np.asarray(samples, dtype=float)
    return last, {
        "transition_time_median_ms": float(np.median(values)),
        "transition_time_q25_ms": float(np.quantile(values, 0.25)),
        "transition_time_q75_ms": float(np.quantile(values, 0.75)),
        "timing_repeats": int(repeats),
    }


def run(output_dir: Path, timing_repeats: int, force_rebuild: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models, phases, endpoints = build_models_and_phases(output_dir, force_rebuild)
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    mechanics_commands: dict[str, np.ndarray] = {}
    command_manifest_rows: list[dict[str, object]] = []
    for case_id, _ in target_specs():
        model = models[case_id]
        phase = phases[case_id]
        problem = TransitionProblem.from_model(model, fe_spec())
        endpoint = endpoints.loc[endpoints.case_id == case_id].iloc[0]
        continuous_root = np.asarray(
            [endpoint.root_x_m, endpoint.root_y_m, endpoint.root_z_m], dtype=float
        )
        continuous_stiffness_min = float(endpoint.root_stiffness_min)

        setup_started = time.perf_counter_ns()
        mechanics_cache = build_local_mechanics_cache(model.geometry, continuous_root, FD_H_M)
        reference = reference_from_continuous_command(mechanics_cache, phase)
        mechanics_setup_ms = (time.perf_counter_ns() - setup_started) * 1e-6
        reference_force = reference.force_n

        for levels in LEVELS:
            methods = []
            nearest, nearest_timing = timed(
                lambda levels=levels: fixed_gauge_nearest(problem, phase, levels),
                timing_repeats,
            )
            methods.append(("fixed_nearest", nearest, nearest_timing, {}))
            objective_offset, offset_timing = timed(
                lambda levels=levels: best_global_offset_rounding(problem, phase, levels),
                timing_repeats,
            )
            methods.append(("best_offset", objective_offset, offset_timing, {}))
            mechanics, mechanics_timing = timed(
                lambda levels=levels: mechanics_aware_best_offset_rounding(
                    problem,
                    phase,
                    levels,
                    mechanics_cache,
                    reference=reference,
                ),
                timing_repeats,
            )
            methods.append(
                (
                    "mechanics_offset",
                    mechanics.quantization,
                    mechanics_timing,
                    {
                        "predicted_shift_mm": 1e3 * mechanics.predicted_shift_m,
                        "predicted_local_stable": mechanics.locally_stable,
                        "predicted_local_stiffness_min": mechanics.local_stiffness_eigenvalues_n_m[0],
                        "stable_offset_candidates": mechanics.stable_candidate_count,
                    },
                )
            )

            command_key = (
                case_id.replace("+", "p").replace("-", "m")
                + f"__Q{levels:04d}"
            )
            mechanics_commands[command_key] = mechanics.quantization.indices.astype(
                np.uint16, copy=True
            )
            command_manifest_rows.append(
                {
                    "command_key": command_key,
                    "case_id": case_id,
                    "levels": int(levels),
                    "phase_step_deg": 360.0 / levels,
                    "channels": int(mechanics.quantization.indices.size),
                    "gauge_channel": 0,
                    "gauge_index": int(mechanics.quantization.indices[0]),
                    "encoding": "phase_rad = 2*pi*uint16_index/levels",
                }
            )

            for method_name, result, timing, extra in methods:
                actual = validate_root(model, result.phase)
                actual_root = np.asarray(actual["root_position_m"], dtype=float)
                eigenvalues = np.asarray(actual["root_stiffness_eigenvalues_n_m"], dtype=float)
                actual_shift_mm = (
                    float(1e3 * np.linalg.norm(actual_root - continuous_root))
                    if np.all(np.isfinite(actual_root))
                    else np.nan
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "method": method_name,
                        "levels": levels,
                        "phase_step_deg": 360.0 / levels,
                        "objective": result.objective,
                        "actual_equilibrium_shift_mm": actual_shift_mm,
                        "actual_root_converged": actual["root_converged"],
                        "actual_root_stable": actual["root_stable"],
                        "actual_stiffness_min_n_m": eigenvalues[0],
                        "stiffness_retention_pct": (
                            100.0 * eigenvalues[0] / continuous_stiffness_min
                            if np.isfinite(eigenvalues[0])
                            else np.nan
                        ),
                        "candidate_evaluations": result.evaluations,
                        "command_key": command_key if method_name == "mechanics_offset" else "",
                        "mechanics_setup_ms": mechanics_setup_ms,
                        "reference_force_norm_n": float(np.linalg.norm(reference_force)),
                        **timing,
                        **extra,
                    }
                )
                if method_name == "mechanics_offset":
                    prediction_rows.append(
                        {
                            "case_id": case_id,
                            "levels": levels,
                            "predicted_shift_mm": extra["predicted_shift_mm"],
                            "actual_shift_mm": actual_shift_mm,
                            "absolute_prediction_error_mm": abs(
                                float(extra["predicted_shift_mm"]) - actual_shift_mm
                            ),
                            "reference_condition_number": reference.condition_number,
                        }
                    )
        print("discrete", case_id)

    results = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = (
        results.groupby(["method", "levels"], as_index=False)
        .agg(
            cases=("case_id", "nunique"),
            stable_fraction=("actual_root_stable", "mean"),
            converged_fraction=("actual_root_converged", "mean"),
            shift_median_mm=("actual_equilibrium_shift_mm", "median"),
            shift_p90_mm=("actual_equilibrium_shift_mm", lambda values: values.quantile(0.9)),
            stiffness_retention_median_pct=("stiffness_retention_pct", "median"),
            transition_time_median_ms=("transition_time_median_ms", "median"),
            mechanics_setup_median_ms=("mechanics_setup_ms", "median"),
        )
        .sort_values(["levels", "method"])
    )
    paired = results.pivot(
        index=["case_id", "levels"], columns="method", values="actual_equilibrium_shift_mm"
    )
    paired_summary = {
        "mechanics_better_than_fixed_count": int(
            (paired.mechanics_offset < paired.fixed_nearest - 1e-12).sum()
        ),
        "mechanics_tied_fixed_count": int(
            np.isclose(paired.mechanics_offset, paired.fixed_nearest, atol=1e-12).sum()
        ),
        "mechanics_better_than_best_offset_count": int(
            (paired.mechanics_offset < paired.best_offset - 1e-12).sum()
        ),
        "mechanics_tied_best_offset_count": int(
            np.isclose(paired.mechanics_offset, paired.best_offset, atol=1e-12).sum()
        ),
        "pairs": int(len(paired)),
    }
    results.to_csv(output_dir / "mechanics_offset_results.csv", index=False)
    summary.to_csv(output_dir / "mechanics_offset_summary.csv", index=False)
    predictions.to_csv(output_dir / "mechanics_offset_prediction_audit.csv", index=False)
    np.savez_compressed(
        output_dir / "mechanics_offset_phase_indices.npz", **mechanics_commands
    )
    pd.DataFrame(command_manifest_rows).to_csv(
        output_dir / "mechanics_offset_command_manifest.csv", index=False
    )
    manifest = {
        "configuration": {
            "levels": LEVELS,
            "targets": len(target_specs()),
            "timing_repeats": timing_repeats,
            "fd_h_m": FD_H_M,
            "root_h_m": ROOT_H_M,
        },
        "paired_shift_comparison": paired_summary,
        "prediction": {
            "spearman_rho": float(
                scipy.stats.spearmanr(
                    predictions.predicted_shift_mm,
                    predictions.actual_shift_mm,
                ).statistic
            ),
            "median_absolute_error_mm": float(
                predictions.absolute_prediction_error_mm.median()
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "scope": (
            "Same-Rayleigh-model mechanics; 16x16 single-sided array; 15 target positions. "
            "No finite-ka, calibration-error, or experimental validation claim."
        ),
    }
    (output_dir / "mechanics_offset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mechanics_offset_prototype_results"),
    )
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.timing_repeats, args.force_rebuild)


if __name__ == "__main__":
    main()
