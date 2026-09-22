"""Reproduce the Conventional, FE, GFE, IB and GS Single commands.

Four paired native cold starts use alpha10 FE/GFE and the unchanged 10,000-step
BFGS ceiling. The first seed supplies the pressure/decision representatives and
three independently validated finite-ka elastic-bead equilibria. The matched
native eight-point IB and GS prescriptions supply two additional commands and
independently validated equilibria. Full mode increases only the native
mechanical validation resolution and root budget.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from dataclasses import asdict, fields, replace
import fingerprintlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from appendix_settings import ROOT, SETTINGS, geometry
from appendix_io import save_npz, write_bytes
sys.path.insert(0, str(ROOT.parent / "generality_package/runtime/src"))
from hat_revision_pipeline.cache import CacheStore, digest_array, digest_payload, dependency_snapshot
from hat_revision_pipeline.config import RunConfig
from hat_revision_pipeline.gorkov_core import (
    ArrayGeometry, Condition, SingleTargetObjective, method_spec, formula_self_test,
    transfer_matrix, SOURCE_SCALE_PA_M_PER_MURATA_UNIT, REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
)
from hat_revision_pipeline.fixed_fe_endpoints import fixed_single_fe_method_spec
from hat_revision_pipeline.multitrap import SphereInFluid, effective_weight_force_target_n
from hat_revision_pipeline.sota import FESolveResult, solve_corrected_gorkov_fe
from hat_revision_pipeline.exact_validator import (
    ArbitraryArrayPressureField, PartialWaveForceEvaluator, PartialWaveNumerics,
    Medium, ElasticSphere, common_cartesian_root_starts, validate_static_trap,
)

METHODS = ("Conventional", "FE", "GFE")
LINEAR_METHODS = ("IB", "GS")
BENCHMARK_METHODS = METHODS + LINEAR_METHODS
TARGET_M = np.array([0., 0., .05])
PAIR_COUNT = 4


def atomic_json(path, value):
    write_bytes(Path(path), (json.dumps(value, indent=2, default=str) + "\n").encode())


def native_objectives(positions, normals, frequency_hz):
    """Exact native Single objectives; field and curvature weights are explicit."""
    result = {}
    for method in METHODS:
        spec = method_spec("Conventional") if method == "Conventional" else fixed_single_fe_method_spec()
        condition = Condition(label=f"appendix-Single-{method}", positions=positions,
            normals=normals, target_m=tuple(TARGET_M), frequency_hz=frequency_hz,
            curvature_weight=spec.curvature_weight)
        target_force = (effective_weight_force_target_n(SphereInFluid(frequency_hz=frequency_hz))
                        if method == "GFE" else np.zeros(3))
        result[method] = SingleTargetObjective(condition, spec, force_target_n=target_force)
    return result


def native_linear_baselines(positions, normals, config, cache):
    """Use the main-figure IB/GS prescriptions and complete synthesis timing."""
    from hat_revision_pipeline import pipeline as p
    from hat_revision_pipeline.final_figures import _single_matched_baseline_commands

    if not np.array_equal(TARGET_M, p.MAIN_TARGET_M):
        raise ValueError("The appendix target must match the native Single target.")
    context = SimpleNamespace(positions_m=positions, normals=normals,
        config=config, cache=cache, memo={}, recompute=False)
    runs = _single_matched_baseline_commands(context)
    return {"IB": runs["ib"], "GS": runs["audit"]}


def endpoint_population(mechanics):
    """Export the native main-Figure-8 Single endpoint table schema."""
    return pd.DataFrame([dict(
        method=row["method"], time_s=row["command_time_s"],
        root_found=row["finite_ka_root_found"],
        displacement_a=row["finite_ka_displacement_a"],
        stiffness_min_n_m=row["finite_ka_stiffness_min_n_m"],
        pressure_abs_at_exact_equilibrium_pa=row["pressure_abs_at_exact_equilibrium_pa"],
        time_source=("command_run.timing.total_s" if row["method"] in LINEAR_METHODS
                     else "command_time_s"),
        phase_source=row["phase_source"], terminal_iteration=row["terminal_iteration"],
        phase_fingerprint=row["phase_fingerprint"],
        pair_id="" if row["method"] in LINEAR_METHODS else f"seed-{row['seed']}",
        seed=row["seed"],
    ) for row in mechanics])


def finite_ka_protocol(mode, setting_id):
    """Same native generality validation settings, without a plotting import."""
    full = mode == "full"
    numerics = PartialWaveNumerics(lmax=8 if full else 6,
        fit_shell_wavelengths=(.10, .15, .20), fit_n_mu=12, fit_n_phi=24,
        surface_n_mu=20, surface_n_phi=40)
    starts = common_cartesian_root_starts() if full else common_cartesian_root_starts((1.,))
    return dict(numerics=asdict(numerics), root_starts_a=starts.tolist(),
        search_half_width_a=4., max_nfev_per_start=70 if full else 22,
        numerical_root_tolerance=1e-3 if full else 2.5e-3,
        least_squares=dict(algorithm="scipy.optimize.least_squares with bounded default method",
                           xtol=1e-10, ftol=1e-10, gtol=1e-10),
        root_acceptance="optimizer success and interior point and scaled total-force residual <= numerical_root_tolerance",
        boundary_fraction_epsilon=1e-7,
        resolved_root_selection="minimum displacement norm, then minimum scaled residual",
        unresolved_candidate_selection="interior, then optimizer success, then scaled residual, then displacement norm",
        jacobian_step_a=.10, medium=asdict(Medium()), sphere=asdict(ElasticSphere()),
        include_effective_gravity=True, source_strength_pa_m_peak=REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
        directivity="Fixed reference angular pattern; ideal retuned sources", front_only=True,
        model=PartialWaveForceEvaluator.model_name,
        model_role="independent finite-ka elastic-bead force reference",
        model_scope="homogeneous isotropic elastic solid sphere with internal longitudinal and shear modes; inviscid host momentum-flux integration",
        dependencies=dependency_snapshot(), setting_id=setting_id)


def make_evaluator(positions, normals, frequency_hz, phase, protocol):
    """Construct the independent native evaluator from the saved protocol."""
    from generality_mechanics_speed import install as accelerate
    accelerate()
    field = ArbitraryArrayPressureField(frequency_hz, positions, np.exp(1j * phase),
        normals=normals, source_strength_pa_m=protocol["source_strength_pa_m_peak"],
        front_only=protocol["front_only"])
    numerics = dict(protocol["numerics"])
    numerics["fit_shell_wavelengths"] = tuple(numerics["fit_shell_wavelengths"])
    return PartialWaveForceEvaluator(frequency_hz, field, Medium(**protocol["medium"]),
        ElasticSphere(**protocol["sphere"]), PartialWaveNumerics(**numerics),
        include_effective_gravity=protocol["include_effective_gravity"])


def mechanics_row(label: str, result: Any) -> dict[str, Any]:
    """Preserve native raw equilibrium mechanics without a stability classification."""

    equilibrium = result.equilibrium
    resolved = bool(equilibrium.numerical_root_found)
    eigenvalues = np.asarray(result.symmetric_stiffness_eigenvalues_n_m, dtype=float)
    minimum = float(np.min(eigenvalues)) if resolved and np.all(np.isfinite(eigenvalues)) else math.nan
    target_metadata = dict(result.metadata.get("target_force_evaluation", {}))
    equilibrium_metadata = dict(result.metadata.get("equilibrium_force_evaluation", {}))
    displacement = np.asarray(equilibrium.displacement_m, dtype=float)
    target_force = np.asarray(result.target_force_n, dtype=float)
    equilibrium_force = np.asarray(result.equilibrium_force_n, dtype=float)
    medium_metadata = dict(result.metadata.get("medium", {}))
    sphere_metadata = dict(result.metadata.get("sphere", {}))
    required_medium = {"density_kg_m3", "gravity_m_s2"}
    required_sphere = {"density_kg_m3", "volume_m3"}
    missing_medium = sorted(required_medium.difference(medium_metadata))
    missing_sphere = sorted(required_sphere.difference(sphere_metadata))
    if missing_medium or missing_sphere:
        raise ValueError(
            "StaticValidationResult lacks the validator provenance required for "
            "force decomposition: "
            f"missing medium={missing_medium}, sphere={missing_sphere}"
        )
    include_effective_gravity = bool(
        result.metadata.get("include_effective_gravity", False)
    )
    effective_gravity = np.zeros(3, dtype=float)
    if include_effective_gravity:
        effective_gravity[2] = -(
            float(sphere_metadata["density_kg_m3"])
            - float(medium_metadata["density_kg_m3"])
        ) * float(sphere_metadata["volume_m3"]) * float(
            medium_metadata["gravity_m_s2"]
        )
    target_radiation_force = target_force - effective_gravity
    equilibrium_radiation_force = equilibrium_force - effective_gravity
    stiffness = np.asarray(result.symmetric_stiffness_n_m, dtype=float)
    jacobian = np.asarray(result.force_jacobian_n_m, dtype=float)
    antisymmetric = np.asarray(result.antisymmetric_force_jacobian_n_m, dtype=float)
    candidate_point = np.asarray(equilibrium.equilibrium_m, dtype=float)
    selected_candidate = min(
        equilibrium.candidates,
        key=lambda candidate: float(
            np.linalg.norm(candidate.equilibrium_m - candidate_point)
            + np.linalg.norm(candidate.initial_offset_a - equilibrium.selected_start_offset_a)
        ),
    )
    finite_eigenvalues = bool(np.all(np.isfinite(eigenvalues)))
    resolved_array_json = lambda values: (
        json.dumps(np.asarray(values, dtype=float).tolist(), allow_nan=False) if resolved else "null"
    )
    return {
        "method": str(label),
        "finite_ka_root_found": resolved,
        "finite_ka_equilibrium_x_m": float(candidate_point[0]) if resolved else math.nan,
        "finite_ka_equilibrium_y_m": float(candidate_point[1]) if resolved else math.nan,
        "finite_ka_equilibrium_z_m": float(candidate_point[2]) if resolved else math.nan,
        "finite_ka_displacement_x_m": float(displacement[0]) if resolved else math.nan,
        "finite_ka_displacement_y_m": float(displacement[1]) if resolved else math.nan,
        "finite_ka_displacement_z_m": float(displacement[2]) if resolved else math.nan,
        "finite_ka_displacement_m": float(equilibrium.displacement_norm_m) if resolved else math.nan,
        "finite_ka_displacement_um": 1.0e6 * float(equilibrium.displacement_norm_m) if resolved else math.nan,
        "finite_ka_displacement_a": float(equilibrium.displacement_norm_a) if resolved else math.nan,
        "finite_ka_candidate_x_m": float(candidate_point[0]),
        "finite_ka_candidate_y_m": float(candidate_point[1]),
        "finite_ka_candidate_z_m": float(candidate_point[2]),
        "finite_ka_candidate_displacement_x_m": float(displacement[0]),
        "finite_ka_candidate_displacement_y_m": float(displacement[1]),
        "finite_ka_candidate_displacement_z_m": float(displacement[2]),
        "finite_ka_candidate_displacement_m": float(equilibrium.displacement_norm_m),
        "finite_ka_candidate_displacement_a": float(equilibrium.displacement_norm_a),
        "finite_ka_root_residual_scaled": float(equilibrium.residual_scaled_norm) if resolved else math.nan,
        "finite_ka_candidate_residual_scaled": float(equilibrium.residual_scaled_norm),
        "finite_ka_root_on_search_boundary": bool(equilibrium.on_search_boundary) if resolved else math.nan,
        "finite_ka_candidate_on_search_boundary": bool(equilibrium.on_search_boundary),
        "finite_ka_root_nfev": int(equilibrium.nfev) if resolved else math.nan,
        "finite_ka_candidate_nfev": int(equilibrium.nfev),
        "finite_ka_selected_optimizer_success": bool(selected_candidate.optimizer_success),
        "finite_ka_selected_optimizer_message": str(selected_candidate.message),
        "finite_ka_selected_start_a_json": json.dumps(
            np.asarray(equilibrium.selected_start_offset_a, dtype=float).tolist()
        ),
        "finite_ka_target_force_x_n": float(target_force[0]),
        "finite_ka_target_force_y_n": float(target_force[1]),
        "finite_ka_target_force_z_n": float(target_force[2]),
        "finite_ka_target_force_norm_n": float(np.linalg.norm(target_force)),
        "finite_ka_target_radiation_force_x_n": float(target_radiation_force[0]),
        "finite_ka_target_radiation_force_y_n": float(target_radiation_force[1]),
        "finite_ka_target_radiation_force_z_n": float(target_radiation_force[2]),
        "finite_ka_target_radiation_force_norm_n": float(
            np.linalg.norm(target_radiation_force)
        ),
        "finite_ka_effective_gravity_x_n": float(effective_gravity[0]),
        "finite_ka_effective_gravity_y_n": float(effective_gravity[1]),
        "finite_ka_effective_gravity_z_n": float(effective_gravity[2]),
        "finite_ka_effective_weight_n": float(np.linalg.norm(effective_gravity)),
        "finite_ka_equilibrium_force_x_n": float(equilibrium_force[0]) if resolved else math.nan,
        "finite_ka_equilibrium_force_y_n": float(equilibrium_force[1]) if resolved else math.nan,
        "finite_ka_equilibrium_force_z_n": float(equilibrium_force[2]) if resolved else math.nan,
        "finite_ka_equilibrium_radiation_force_x_n": float(equilibrium_radiation_force[0]) if resolved else math.nan,
        "finite_ka_equilibrium_radiation_force_y_n": float(equilibrium_radiation_force[1]) if resolved else math.nan,
        "finite_ka_equilibrium_radiation_force_z_n": float(equilibrium_radiation_force[2]) if resolved else math.nan,
        "finite_ka_candidate_force_x_n": float(equilibrium_force[0]),
        "finite_ka_candidate_force_y_n": float(equilibrium_force[1]),
        "finite_ka_candidate_force_z_n": float(equilibrium_force[2]),
        "finite_ka_stiffness_min_n_m": minimum,
        "finite_ka_stiffness_mid_n_m": float(np.median(eigenvalues)) if resolved and finite_eigenvalues else math.nan,
        "finite_ka_stiffness_max_n_m": float(np.max(eigenvalues)) if resolved and finite_eigenvalues else math.nan,
        "finite_ka_stiffness_eigenvalues_n_m_json": resolved_array_json(eigenvalues),
        "finite_ka_stiffness_matrix_n_m_json": resolved_array_json(stiffness),
        "finite_ka_force_jacobian_n_m_json": resolved_array_json(jacobian),
        "finite_ka_antisymmetric_jacobian_fro_n_m": float(np.linalg.norm(antisymmetric)) if resolved else math.nan,
        "finite_ka_candidate_stiffness_eigenvalues_n_m_json": json.dumps(eigenvalues.tolist(), allow_nan=False),
        "finite_ka_candidate_stiffness_matrix_n_m_json": json.dumps(stiffness.tolist(), allow_nan=False),
        "finite_ka_candidate_force_jacobian_n_m_json": json.dumps(jacobian.tolist(), allow_nan=False),
        "finite_ka_candidate_antisymmetric_jacobian_fro_n_m": float(np.linalg.norm(antisymmetric)),
        "finite_ka_projection_residual_target": float(target_metadata.get("projection_residual", math.nan)),
        "finite_ka_projection_residual_equilibrium": float(equilibrium_metadata.get("projection_residual", math.nan)) if resolved else math.nan,
        "finite_ka_projection_residual_candidate": float(equilibrium_metadata.get("projection_residual", math.nan)),
        "finite_ka_projection_condition": float(equilibrium_metadata.get("projection_condition", math.nan)),
        "finite_ka_model_name": str(result.model_name),
        "finite_ka_model_metadata_json": json.dumps(result.metadata, sort_keys=True, allow_nan=False),
    }


def verify_outputs(campaign, setting_id):
    """Verify five-method Single exports and retained legacy source tables."""
    root = Path(campaign) / setting_id
    with np.load(root / "array_geometry.npz", allow_pickle=False) as saved_geometry:
        source_count = len(saved_geometry["positions_m"])
    rows = pd.read_csv(root / "tables/fig5_exact_force_field_concentration.csv")
    legacy_methods = {"Conventional", "FE", "GFE", "IB", "GS", "AD"}
    if len(rows) == 5:
        expected_methods = set(BENCHMARK_METHODS)
        assert rows.task.eq("Single").all() and set(rows.method) == expected_methods
        scope = "requested-five-method-Single"
    elif len(rows) == 3:
        expected_methods = set(METHODS)
        assert rows.task.eq("Single").all() and set(rows.method) == expected_methods
        scope = "legacy-three-method-Single"
    elif len(rows) in (6, 24):
        expected_methods = legacy_methods
        assert set(rows.method) == expected_methods
        assert set(rows.task) == ({"Single"} if len(rows) == 6 else {"Single", "Triple"})
        scope = "legacy-read-only"
    else:
        raise AssertionError(f"Unexpected mechanics row count: {len(rows)}")
    commands = []
    for (task, method), group in rows.groupby(["task", "method"], sort=True):
        assert len(group) == (1 if task == "Single" else 3)
        suffix = "FEplusg" if method == "GFE" and task == "Single" else method
        path = root / "commands" / f"{task}_{suffix}.npz"
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, path
        with np.load(path, allow_pickle=False) as data:
            phase = data["phase_rad"]
            assert phase.shape == (source_count,) and np.isfinite(phase).all()
            phase_hash = digest_array(phase)
            assert group.phase_fingerprint.eq(phase_hash).all(), (setting_id, task, method)
        commands.append(dict(task=task, method=method, filename=path.name, phase_fingerprint=phase_hash))
    if len(rows) == 5:
        endpoints = pd.read_csv(root / "tables/fig8_single_endpoint_population.csv")
        assert len(endpoints) == 5 and set(endpoints.method) == expected_methods
        for method in BENCHMARK_METHODS:
            source = rows.loc[rows.method.eq(method)].iloc[0]
            endpoint = endpoints.loc[endpoints.method.eq(method)].iloc[0]
            assert endpoint.phase_fingerprint == source.phase_fingerprint
            assert np.isclose(endpoint.time_s, source.command_time_s, rtol=1e-14, atol=0.)
            assert np.isclose(endpoint.pressure_abs_at_exact_equilibrium_pa,
                source.pressure_abs_at_exact_equilibrium_pa, rtol=1e-14, atol=0., equal_nan=True)
            assert np.isfinite(endpoint.time_s) and endpoint.time_s > 0.
    return dict(setting_id=setting_id, scope=scope, fig5_rows=len(rows),
        command_npz_count=len(commands), command_crcs_passed=True,
        command_table_phase_hashes_passed=True, commands=commands)


def compute(campaign, setting_id, mode="smoke"):
    setting = next(s for s in SETTINGS if s["id"] == setting_id)
    positions, normals = geometry(setting)
    output = Path(campaign).resolve() / setting_id
    for name in ("data", "tables", "commands", "paired_single"):
        (output / name).mkdir(parents=True, exist_ok=True)
    config = replace(RunConfig.for_mode("full" if mode == "full" else "quick"),
        frequency_hz=setting["frequency_hz"], stencil_m=.0005 * 40000 / setting["frequency_hz"],
        exact_lmax=8 if mode == "full" else 6)
    protocol = finite_ka_protocol(mode, setting_id)
    objectives = native_objectives(positions, normals, config.frequency_hz)
    guard = formula_self_test()
    assert guard["passed"], guard
    initial = np.random.default_rng(config.random_seed).uniform(-np.pi, np.pi, len(positions))
    points = np.array([[0., 0., .05], [.01, -.006, .03], [-.008, .005, .06]])
    evaluator = make_evaluator(positions, normals, config.frequency_hz, initial, protocol)
    direct = transfer_matrix(points, ArrayGeometry(positions, normals), config.frequency_hz,
        source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT) @ np.exp(1j * initial)
    independent = evaluator.field.pressure(points)
    field_error = float(np.linalg.norm(direct - independent) / np.linalg.norm(direct))
    assert field_error < 1e-10, field_error
    physics_files = ["gorkov_core.py", "sota.py", "multitrap.py", "exact_validator.py", "fixed_fe_endpoints.py",
                     "final_figures.py", "pipeline.py"]
    source_dir = ROOT.parent / "generality_package/runtime/src/hat_revision_pipeline"
    hashes = {name: fingerprintlib.fingerprint((source_dir / name).read_bytes()).hexdigest() for name in physics_files}
    identity = dict(setting=setting, positions=digest_array(positions), normals=digest_array(normals),
        config=asdict(config), source_hashes=hashes, mode=mode)
    cache = CacheStore(output / "cache", namespace="appendix-single-native-v2-" + digest_payload(identity))
    manifest = dict(identity, dependencies=dependency_snapshot(), finite_ka_protocol=protocol,
        paired_single_starts=PAIR_COUNT, root_starts=len(protocol["root_starts_a"]),
        field_checks=dict(formula_guard=guard, field_relative_error=field_error),
        source_directivity_note="The reference 40-kHz angular pattern and source strength are held fixed at all carriers.",
        computation_scope="Four paired Conventional/FE/GFE Single starts; first seed and native matched IB/GS supply mechanics",
        linear_baseline_source="hat_revision_pipeline.final_figures._single_matched_baseline_commands",
        linear_baseline_protocol=dict(control_points=8, radius_wavelengths=.45,
            ib_maxiter=200, ib_tolerance_rad=.01, gs_iterations=100,
            command_time_source="BenchmarkRun.timing.total_s, including transfer construction"),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_json(output / "execution_settings.json", manifest)
    save_npz(output / "array_geometry.npz", positions_m=positions, normals=normals, frequency_hz=config.frequency_hz)
    rows, first_runs = [], {}
    started = time.monotonic()
    def progress(stage, **extra):
        record = dict(setting_id=setting_id, stage=stage, elapsed_s=time.monotonic()-started,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra)
        atomic_json(output / "appendix_status.json", record)
        print(f"{setting_id}: {stage} ({record['elapsed_s']:.1f}s)", flush=True)
    progress("paired_single_commands")
    for pair_index in range(PAIR_COUNT):
        seed = config.random_seed + pair_index
        initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(positions))
        for method in METHODS:
            objective = objectives[method]
            payload = dict(method="Force-Equilibrium" if method == "FE" else method,
                objective_method=asdict(objective.method), target_m=TARGET_M.tolist(), seed=int(seed),
                initial_phase=digest_array(initial), maxiter=10000, gtol=config.gtol,
                force_target_n=objective.force_target_n.tolist(),
                role="Figure1 paired central target" if method == "GFE" else "Single paired central target")
            kind = "main_feg_single_run" if method == "GFE" else "final_single_vortex_run"
            run, _, _ = cache.get_or_compute(kind, payload,
                lambda objective=objective, initial=initial: solve_corrected_gorkov_fe(
                    objective, initial, maxiter=10000, gtol=config.gtol))
            values = {f.name: np.asarray(getattr(run, f.name)) for f in fields(FESolveResult)}
            save_npz(output / "paired_single" / f"{method}_pair{pair_index}.npz",
                **values, seed=seed, pair_index=pair_index, target_m=TARGET_M)
            rows.append(dict(method=method, alpha_per_m=objective.method.alpha_per_m,
                pair_index=pair_index, seed=seed, iteration_cap=10000, iterations=run.iterations,
                evaluations=run.evaluations, wall_time_s=float(run.history_wall_s[-1]),
                objective=run.objective, gradient_norm=run.gradient_norm,
                optimizer_success=run.success, optimizer_status=run.status,
                optimizer_message=run.message, phase_fingerprint=digest_array(run.phase_rad)))
            if pair_index == 0:
                first_runs[method] = run
                suffix = "FEplusg" if method == "GFE" else method
                save_npz(output / "commands" / f"Single_{suffix}.npz",
                    phase_rad=run.phase_rad, target_m=TARGET_M)
            progress("paired_single_commands", completed=len(rows), total=PAIR_COUNT * len(METHODS))
    write_bytes(output / "tables/appendix_paired_single_runs.csv", pd.DataFrame(rows).to_csv(index=False).encode())
    progress("matched_linear_commands")
    linear_runs = native_linear_baselines(positions, normals, config, cache)
    for method, run in linear_runs.items():
        save_npz(output / "commands" / f"Single_{method}.npz",
            phase_rad=run.phase_rad, target_m=TARGET_M,
            command_time_s=run.timing.total_s,
            timing_json=json.dumps(asdict(run.timing), sort_keys=True),
            terminal_iteration=run.solver_result.iterations)
    progress("single_exact_mechanics")
    mechanics = []
    for method in BENCHMARK_METHODS:
        linear = method in LINEAR_METHODS
        run = linear_runs[method] if linear else first_runs[method]
        evaluator = make_evaluator(positions, normals, config.frequency_hz, run.phase_rad, protocol)
        payload = dict(phase_fingerprint=digest_array(run.phase_rad), target_m=TARGET_M.tolist(), protocol=protocol)
        result, _, _ = cache.get_or_compute("appendix_single_elastic_validation", payload,
            lambda evaluator=evaluator: validate_static_trap(evaluator, TARGET_M,
                search_half_width_a=protocol["search_half_width_a"],
                initial_offsets_a=np.asarray(protocol["root_starts_a"]),
                max_nfev=protocol["max_nfev_per_start"],
                numerical_root_tolerance=protocol["numerical_root_tolerance"],
                jacobian_step_a=protocol["jacobian_step_a"]))
        record = mechanics_row(method, result)
        resolved = result.equilibrium.numerical_root_found
        record.update(task="Single", target_id="T1", target_index=0, seed=config.random_seed,
            phase_fingerprint=digest_array(run.phase_rad),
            terminal_iteration=run.solver_result.iterations if linear else run.iterations,
            command_time_s=float(run.timing.total_s if linear else run.history_wall_s[-1]),
            alpha_per_m=math.nan if linear else objectives[method].method.alpha_per_m,
            pressure_abs_at_intended_target_pa=float(abs(evaluator.field.pressure(TARGET_M))),
            pressure_abs_at_exact_equilibrium_pa=float(abs(evaluator.field.pressure(result.equilibrium.equilibrium_m))) if resolved else math.nan,
            phase_source=("Native matched eight-point, R=0.45 lambda prescription" if linear
                          else "Native paired cold Single solve; first prescribed seed"),
            target_x_m=TARGET_M[0], target_y_m=TARGET_M[1], target_z_m=TARGET_M[2])
        if linear:
            record["command_timing_json"] = json.dumps(asdict(run.timing), sort_keys=True)
        mechanics.append(record)
        progress("single_exact_mechanics", completed=len(mechanics), total=len(BENCHMARK_METHODS))
    write_bytes(output / "tables/fig5_exact_force_field_concentration.csv", pd.DataFrame(mechanics).to_csv(index=False).encode())
    population = endpoint_population(mechanics)
    write_bytes(output / "tables/fig8_single_endpoint_population.csv", population.to_csv(index=False).encode())
    atomic_json(output / "cache_access.json", cache.access_records)
    atomic_json(output / "output_integrity_checks.json", verify_outputs(campaign, setting_id))
    progress("complete", exit_code=0, single_count=len(BENCHMARK_METHODS), paired_single_count=PAIR_COUNT * len(METHODS))
    return SimpleNamespace(output_root=output, config=config, cache=cache,
        tables={"appendix_paired_single_runs": pd.DataFrame(rows),
                "fig5_exact_force_field_concentration": pd.DataFrame(mechanics),
                "fig8_single_endpoint_population": population})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "campaign")
    parser.add_argument("--setting", required=True, choices=[s["id"] for s in SETTINGS])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    if args.verify_only:
        print(json.dumps(verify_outputs(args.campaign, args.setting), indent=2))
    else:
        compute(args.campaign, args.setting, args.mode)
