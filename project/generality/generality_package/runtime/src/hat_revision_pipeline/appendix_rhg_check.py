"""One paired RH+gravity check against the current nominal C5 RH Bottle.

Only the constant effective-weight force target is changed. The shared initial
command, alternative-sign Gor'kov coefficients, weights, smoothing and native
compact L-BFGS-B protocol are fixed. This is one check, with no tuning or restarts.
"""
from __future__ import annotations

from dataclasses import asdict, replace

from rineng_content_id import content_identity
import io
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize._lbfgsb_py import _minimize_lbfgsb
from threadpoolctl import threadpool_limits

from . import exact_validator as ev, pipeline as p, sota
from .appendix_b3_mechanics import transverse_descriptor
from .cache import digest_array, digest_file, digest_payload
from .gorkov_core import ArrayGeometry, Condition, MethodSpec, SingleTargetObjective, gauge_full
from .multitrap import SphereInFluid, effective_weight_force_target_n

SCHEMA = "single-paired-RH-effective-gravity-check-v1"



def _json(path, value):
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _npz(path, **values):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    temporary = path.with_suffix(".writing.npz")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def _objective(config, target_force):
    geometry = ArrayGeometry.square(config["side"], config["pitch_m"])
    weight = np.diag(config["curvature_weights"])
    spec = MethodSpec("RH", config["alpha_per_m"], config["beta_curvature_per_pa"],
                      config["pressure_mode"], weight)
    objective = SingleTargetObjective(
        Condition("Paired C5 RH gravity check", geometry.positions_m, tuple(config["target_m"]),
                  config["frequency_hz"], weight), spec,
        stencil_spacing_m=config["stencil_spacing_m"],
        smooth_pressure_relative=config["smooth_pressure_relative"],
        force_target_n=target_force)
    assert config["sign"] == "historical_plus"
    objective.coefficients = replace(objective.coefficients,
        gradient_j_m2_pa2=-objective.coefficients.gradient_j_m2_pa2)
    return objective


def run(root, ctx=None):
    root = Path(root).resolve()
    output = root / "appendix_outputs/C_GFE/RHg_check"
    output.mkdir(parents=True, exist_ok=True)
    ctx = ctx or p.prepare_pipeline(run_mode="quick", recompute=False, output_root=root / "smoke_outputs")
    import importlib.util
    spec = importlib.util.spec_from_file_location("rhg_sensitivity", root / "appendix_B_pressure/sensitivity_b.py")
    sensitivity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sensitivity)
    nominal_config = sensitivity.configuration("Bottle", "historical_plus", "RH", beta_factor=1.)
    source = root / "appendix_B_pressure/data/sensitivity_cache" / (sensitivity.cache_key(nominal_config) + ".npz")
    if not source.exists() or not source.with_suffix(".json").exists():
        sensitivity.solve_one(nominal_config, root / "appendix_B_pressure")
    config, saved = sensitivity.load_run(nominal_config, root / "appendix_B_pressure")
    initial = saved["initial_phase_rad"].copy()
    baseline = saved["terminal_phase_rad"].copy()
    np.testing.assert_array_equal(initial, gauge_full(np.random.default_rng(config["seed"]).uniform(-np.pi, np.pi, 255)))
    upward = effective_weight_force_target_n(SphereInFluid(frequency_hz=config["frequency_hz"]))
    np.testing.assert_allclose(upward, -ev._effective_gravity_n(ev.Medium(), ev.ElasticSphere()), rtol=1e-13, atol=0.)
    rh = _objective(config, np.zeros(3))
    rhg = _objective(config, upward)
    baseline_value, baseline_gradient = rh.fun_grad(baseline[1:])
    np.testing.assert_allclose(baseline_value, config["objective"], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(baseline_gradient), config["gradient_l2"], rtol=1e-5, atol=1e-11)
    from .fe_solver_policy import FE_SOLVER_REVISION
    identity = dict(schema=SCHEMA, fe_solver_policy=FE_SOLVER_REVISION, evaluator_backend="compact-single-float64-v1", source_phase_content_id=digest_array(baseline),
        initial_phase_content_id=digest_array(initial), method=asdict(rhg.method),
        force_target_n=upward.tolist(), sign="alternative-sign Gor'kov formulation",
        coefficients=asdict(rhg.coefficients), stencil_spacing_m=rhg.spacing_m,
        half_stencil=rhg.half_stencil, pressure_smoothing_relative=rhg.smooth_pressure_relative,
        maxiter=int(config["maxiter"]), gtol=float(config["gtol"]), solver="L-BFGS-B",
        initial_seed=int(config["seed"]), phase_dimension=256, reduced_dimension=255,
        source_hashes={name: digest_file(Path(__file__).with_name(name)) for name in ("gorkov_core.py", "sota.py")})
    # ndarray-valued dataclass fields are converted before building a durable key.
    identity["method"]["curvature_weight"] = identity["method"]["curvature_weight"].tolist()
    key = digest_payload(identity)
    path = output / ("RHg_" + key + ".npz")
    metadata_path = path.with_suffix(".json")
    with threadpool_limits(limits=1):
        if path.exists() and metadata_path.exists():
            result_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert result_meta["identity"] == identity
            with np.load(path, allow_pickle=False) as saved:
                phase = saved["terminal_phase_rad"].copy()
            assert digest_array(phase) == result_meta["phase_content_id"]
            solved_now = False
        else:
            start = time.perf_counter()
            result = sota.solve_corrected_gorkov_fe(rhg, initial, maxiter=config["maxiter"], gtol=config["gtol"])
            elapsed = time.perf_counter() - start
            phase = np.asarray(result.phase_rad)
            _npz(path, initial_phase_rad=initial, terminal_phase_rad=phase,
                 history_phase_rad=result.history_phase_rad, history_objective=result.history_objective,
                 history_gradient_l2=result.history_gradient_norm, history_wall_s=result.history_wall_s,
                 force_target_n=upward)
            result_meta = dict(identity=identity, phase_content_id=digest_array(phase),
                iterations=int(result.iterations), evaluations=int(result.evaluations),
                optimizer_success=bool(result.success), optimizer_status=int(result.status),
                optimizer_message=str(result.message), objective=float(result.objective),
                terminal_gradient_l2=float(result.gradient_norm), command_wall_s=float(elapsed))
            _json(metadata_path, result_meta)
            solved_now = True
            print(f"RH+g: {result.iterations} iterations; {result.message}", flush=True)
        baseline_meta = dict(iterations=int(config["iterations"]), optimizer_success=bool(config["success"]),
            optimizer_status=int(config["status_code"]), optimizer_message=config["message"],
            objective=float(config["objective"]), terminal_gradient_l2=float(config["gradient_l2"]),
            command_wall_s=float(config["command_time_s"]))
        fields, records = {}, []
        axis = np.linspace(-.007, .007, 121)
        xx, yy = np.meshgrid(axis, axis)
        for method, command, objective, meta in (("RH", baseline, rh, baseline_meta), ("RH+g", phase, rhg, result_meta)):
            result = p._finite_ka_validation(ctx, command, p.MAIN_TARGET_M,
                        key="B3_Bottle" if method == "RH" else "C5_RHg_one_check")
            row = p._static_validation_row(method, result)
            row.update({key: value for key, value in meta.items() if key != "identity"})
            row.update(seed=int(config["seed"]), maxiter=int(config["maxiter"]),
                       phase_content_id=digest_array(command), force_target_z_n=float(objective.force_target_n[2]),
                       mechanical_success=bool(row["finite_ka_locally_restoring"]),
                       position_error_of_resolved_root_um=float(1e6 * result.equilibrium.displacement_norm_m),
                       minimum_symmetric_stiffness_mN_per_m=float(1000 * min(result.symmetric_stiffness_eigenvalues_n_m)))
            raw = objective._raw_metrics(command, False)
            row.update(surrogate_force_residual_n=float(raw["force_residual_norm_n"]),
                       target_pressure_pa=float(raw["pressure_abs_pa"]))
            field = p._array_field(ctx, command)
            descriptor, ring = transverse_descriptor(field, p.MAIN_TARGET_M, objective.condition.wavelength_m)
            row.update(descriptor)
            label = "RH" if method == "RH" else "RHg"
            fields[label + "_surrogate_acoustic_force_n"] = np.asarray(raw["gorkov_force_n"])
            fields[label + "_surrogate_potential_hessian_n_m"] = np.asarray(raw["potential_hessian_n_m"])
            fields[label + "_force_target_n"] = np.asarray(objective.force_target_n)
            for plane, dims in (("XY", (0, 1)), ("XZ", (0, 2))):
                points = np.tile(p.MAIN_TARGET_M, (xx.size, 1))
                points[:, dims[0]] += xx.ravel(); points[:, dims[1]] += yy.ravel()
                fields[label + "_" + plane + "_pressure_pa"] = np.concatenate(
                    [field.pressure(points[i:i+1024]) for i in range(0, len(points), 1024)]).reshape(xx.shape)
            _npz(output / (label + "_mechanics.npz"), phase_rad=command, target_m=p.MAIN_TARGET_M,
                 equilibrium_m=result.equilibrium.equilibrium_m, residual_force_n=result.equilibrium.residual_force_n,
                 force_jacobian_n_m=result.force_jacobian_n_m,
                 symmetric_stiffness_n_m=result.symmetric_stiffness_n_m,
                 stiffness_eigenvalues_n_m=result.symmetric_stiffness_eigenvalues_n_m,
                 root_found=np.asarray(result.equilibrium.numerical_root_found),
                 protocol_json=np.asarray(json.dumps(p._finite_ka_protocol(ctx), default=str)))
            _npz(output / (label + "_ring_descriptor.npz"), **ring)
            records.append(row)
    fields.update(axis_offset_m=axis, initial_phase_rad=initial, RH_phase_rad=baseline, RHg_phase_rad=phase)
    _npz(output / "paired_pressure_fields.npz", **fields)
    comparison = pd.DataFrame(records)
    amplitude_cosine = float(np.dot(np.abs(fields["RH_XZ_pressure_pa"]).ravel(), np.abs(fields["RHg_XZ_pressure_pa"]).ravel()) /
        (np.linalg.norm(fields["RH_XZ_pressure_pa"]) * np.linalg.norm(fields["RHg_XZ_pressure_pa"])))
    comparison["paired_xz_amplitude_cosine"] = amplitude_cosine
    comparison["paired_phase_cosine"] = float(abs(np.mean(np.exp(1j * (phase - baseline)))))
    comparison.to_csv(output / "RH_RHg_paired_results.csv", index=False)
    defaults = {name: str(value.default) for name, value in inspect.signature(_minimize_lbfgsb).parameters.items()
                if value.default is not inspect.Parameter.empty}
    _json(output / "protocol.json", dict(**identity, numerical_scope="One matched initial seed; one new RH+g solve; zero additional RH solves.",
        baseline_source=str(source.relative_to(root)), baseline_solver_record=config,
        baseline_objective_replay=dict(value=float(baseline_value), gradient_l2=float(np.linalg.norm(baseline_gradient))),
        validator_protocol=p._finite_ka_protocol(ctx), validator_content_id=digest_file(ev.__file__),
        source_commands=str(path.name), solver_defaults=defaults,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        pressure_section_points_per_axis=121, pressure_section_half_width_m=.007,
        changed_parameter="Constant upward acoustic force target, equal to the buoyancy-corrected bead weight.",
        unchanged="Initial phase; alpha; beta=nominal beta0; curvature weights; sign; smoothing; stencil; compact L-BFGS-B cap/tolerance; elastic validator.",
        optimization_cache_reused=not solved_now))
    first, second = comparison.iloc[0], comparison.iloc[1]
    ratio = second.position_error_of_resolved_root_um / first.position_error_of_resolved_root_um
    report = ("# One paired RH+g check\n\n"
        "The nominal alternative-sign Bottle RH case from C5 was repeated once after adding the same buoyancy-corrected "
        "gravity force target used by GFE. The original random initial phase was retained; the existing RH endpoint was "
        "not used as a warm start. All objective weights and numerical controls were held fixed.\n\n"
        f"Compact L-BFGS-B cap: {config['maxiter']:,}; gtol: {config['gtol']:g}; alpha: {config['alpha_per_m']:g} m^-1; "
        f"beta=beta0={config['beta_curvature_per_pa']:.8g}; upward force target: {upward[2]:.8g} N.\n\n"
        "| Method | Iterations | Root position error (µm) | Minimum stiffness (mN/m) | Restoring root |\n"
        "|---|---:|---:|---:|---|\n" + "".join(
            f"| {r.method} | {int(r.iterations)} | {r.position_error_of_resolved_root_um:.3f} | "
            f"{r.minimum_symmetric_stiffness_mN_per_m:.4f} | {bool(r.mechanical_success)} |\n" for r in comparison.itertuples()) +
        f"\nRH+g solver stop: {second.optimizer_message} Terminal gradient L2: {second.terminal_gradient_l2:.8g}. "
        f"The resolved-root position error ratio RH+g/RH is {ratio:.6f}. XZ amplitude cosine is {amplitude_cosine:.8f}.\n\n"
        + ("Both resolved equilibria are non-restoring under the independent elastic model. This paired check "
           "does not improve mechanical precision: the gravity-compensated variant retains the transverse saddle "
           "and slightly increases its root-position error.\n" if not first.mechanical_success and not second.mechanical_success else
           "Root position error and restoring stiffness must be considered together when interpreting mechanical precision.\n") +
        "\nThe alternative-sign surrogate is used only in synthesis; the elastic radiation-force model is unchanged. "
        "This paired check does not tune RH weights, add restarts, or establish the frequency of either outcome.\n")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(comparison[["method", "iterations", "position_error_of_resolved_root_um", "minimum_symmetric_stiffness_mN_per_m", "mechanical_success"]].to_string(index=False), flush=True)
    return dict(table=comparison, output=str(output), new_optimization_count=int(solved_now), figures=[])
