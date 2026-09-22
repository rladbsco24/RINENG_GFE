"""Appendix C: matched Triple ablations and an explicit numerical protocol.

The accepted Single and Triple FE parameters remain unchanged. Numerical caches
are matched by problem, full configuration, seed, and initial phase. Imported
timings retain their source scope; no replay time is treated as command time.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import fingerprintlib
import io
import json
import pickle
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import pipeline as p
from .cache import CacheStore, canonical_json, dependency_snapshot, digest_array, digest_payload
from .config import RunConfig, square_normals, square_positions
from .exact_validator import ElasticSphere
from .fig4_threepoint_selection import contracted_targets, make_problem
from .fixed_fe_endpoints import fixed_single_fe_method_spec, fixed_triple_fe_config
from .gorkov_core import formula_self_test
from .multitrap import evaluate_multitrap_objective, run_multitrap

SEEDS = (260869, 260870, 260871)
LABELS = ("Full", "Minus force smoothing", "Minus pressure", "Minus uniformity")
SHORT_LABELS = ("Full", "No force\nsmoothing", "No pressure\npenalty", "No loss\nuniformity")
SCHEMA = "appendix-C-matched-triple-3seeds-v2"


def _write_table(directory, name, rows):
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame["evidence_status"] = "SMOKE"
    frame.to_csv(directory / f"{name}.csv", index=False)
    return frame


def _recover_rendered_tables(root, output):
    """Preserve available notebook measurements without inventing missing precision."""
    deposited = Path(__file__).resolve().parents[2] / "data" / "recovered_notebook_tables"
    deposited_files = sorted(deposited.glob("C_recovered_*")) + sorted(deposited.glob("C_rendered_recovery_provenance.csv"))
    if deposited_files:
        for source_file in deposited_files:
            shutil.copy2(source_file,output/source_file.name)
        return
    candidates = list((root.parent / "upload").glob("RINENG_main8_elastic_smoke*.ipynb"))
    if not candidates:
        return
    source = sorted(candidates)[0]
    notebook = json.loads(source.read_text())
    selected = {
        (25,1):"C_recovered_alpha_transfer",
        (27,7):"C_recovered_main_benchmark",
        (27,9):"C_recovered_main_stopping",
        (27,10):"C_recovered_main_parameters",
        (27,11):"C_recovered_main_timing",
        (27,12):"C_recovered_single_seed_statistics",
        (27,14):"C_recovered_physical_parameters",
        (27,15):"C_recovered_field_parameters",
    }
    provenance = []
    for (cell_index,output_index),name in selected.items():
        try:
            html = "".join(notebook["cells"][cell_index]["outputs"][output_index]["data"]["text/html"])
        except (KeyError,IndexError):
            continue
        frame = pd.read_html(io.StringIO(html))[0]
        frame = frame.loc[:,~frame.columns.str.startswith("Unnamed:")]
        frame["recovery_precision"] = "source rendered precision; raw replicate arrays not reconstructed"
        _write_table(output,name,frame)
        (output / f"{name}_source.html").write_text(html)
        provenance.append(dict(table=name,source_notebook=source.name,
            source_fingerprint=fingerprintlib.fingerprint(source.read_bytes()).hexdigest(),cell_index=cell_index,output_index=output_index,
            precision="Rendered table values are rounded; long text may contain original ellipses. Rounded seed displays are not exact seed identities."))
    _write_table(output,"C_rendered_recovery_provenance",provenance)


def _context(root, output):
    return SimpleNamespace(config=RunConfig(), positions_m=square_positions(), normals=square_normals(),
        output_root=output, recompute=False, cache=CacheStore(output / "cache", namespace=p.PIPELINE_IMPLEMENTATION),
        data_root=root / "runtime" / "data")


def _compatible_run(root, problem, config, seed, initial):
    expected = dict(contract="one-seed-smoke-10000-cap-v1", problem=problem.fingerprint,
        config=config.to_payload(), seed=seed, initial=digest_array(initial))
    for path in sorted((root / "smoke_outputs" / "cache" / "revision_triple_ablation").glob("*.pkl")):
        with path.open("rb") as stream:
            record = pickle.load(stream)
        if (record.get("namespace") == p.PIPELINE_IMPLEMENTATION
                and digest_payload(record.get("payload")) == digest_payload(expected)):
            value = record["value"]
            if value.problem_fingerprint != problem.fingerprint or value.seed != seed:
                raise ValueError("Recovered ablation metadata does not match its cache identity")
            return value, str(path.relative_to(root)), record.get("dependencies", {})
    return None, "", {}


def _compatible_validations(root, ctx):
    """Index the independently validated frozen commands, excluding version labels."""
    available = {}
    required = dict(p._finite_ka_protocol(ctx))
    required.pop("dependencies", None)
    for path in sorted((root / "smoke_outputs" / "cache" / "finite_ka_static_validation").glob("*.pkl")):
        with path.open("rb") as stream:
            record = pickle.load(stream)
        payload = record.get("payload", {})
        protocol = dict(payload.get("protocol", {}))
        protocol.pop("dependencies", None)
        if (record.get("namespace") == p.PIPELINE_IMPLEMENTATION
                and canonical_json(protocol) == canonical_json(required)
                and payload.get("positions") == digest_array(ctx.positions_m)
                and payload.get("normals") == digest_array(ctx.normals)
                and payload.get("frequency_hz") == ctx.config.frequency_hz):
            key = (payload.get("phase"), tuple(payload.get("target", [])))
            available[key] = (record["value"], str(path.relative_to(root)), record.get("dependencies", {}))
    return available


def _term_rows(problem, config, run, label, seed):
    rows = []
    for state, phase, stage in (("initial", run.initial_phases, run.stages[0]),
            ("terminal", run.phases, run.stages[-1])):
        ev = evaluate_multitrap_objective(problem, phase, config,
            force_epsilon=stage.force_epsilon, uniformity_epsilon=stage.uniformity_epsilon)
        curvature = -np.mean([local.d_curvature for local in ev.locals], axis=0)
        force_parts, pressure_parts = [], []
        for index, local in enumerate(ev.locals):
            denominator = np.hypot(local.force_norm_n, stage.force_epsilon) if config.components.force_smoothing else local.force_norm_n
            force_parts.append(config.alpha_force * local.d_force_norm * local.force_norm_n / max(denominator, 1e-30))
            pressure_parts.append(config.beta_pressure * local.d_pressure_penalty if config.components.pressure_retention
                else np.zeros_like(local.d_pressure_penalty))
            rows.append(dict(configuration=label, seed=seed, state=state, target_id=local.target_id,
                pressure_abs_pa=local.pressure_abs, force_norm_n=local.force_norm_n,
                curvature_term_n_m=-local.curvature_n_m,
                force_term_n_m=config.alpha_force * ev.force_penalties[index],
                pressure_term_n_m=config.beta_pressure * local.pressure_penalty if config.components.pressure_retention else 0.,
                local_loss_n_m=ev.local_losses[index],
                weighted_uniformity_n_m=config.gamma_uniformity * ev.uniformity_penalty,
                force_epsilon_n=stage.force_epsilon, uniformity_epsilon_n_m=stage.uniformity_epsilon,
                phase_fingerprint=digest_array(np.asarray(phase))))
        parts = {"curvature": curvature, "force": np.mean(force_parts, axis=0),
            "pressure": np.mean(pressure_parts, axis=0),
            "uniformity": ev.gradient_full - np.mean(ev.local_gradients, axis=0)}
        reconstructed = sum(parts.values())
        rows.append(dict(configuration=label, seed=seed, state=state, target_id="pooled objective",
            objective_n_m=ev.value, gradient_reconstruction_error_l2=float(np.linalg.norm(reconstructed - ev.gradient_full)),
            **{name + "_weighted_gradient_l2_n_m_rad": float(np.linalg.norm(value[1:])) for name, value in parts.items()},
            total_gradient_l2_n_m_rad=float(np.linalg.norm(ev.gradient_reduced))))
    return rows


def _derivative_checks(problem, configs, jobs):
    rows = []
    direction = np.random.default_rng(20260905).normal(size=problem.n_actuators)
    direction[0] = 0.
    direction /= np.linalg.norm(direction)
    for label in ("Full", "Minus force smoothing"):
        run = jobs[(SEEDS[0], label)]["run"]
        config = configs[label]
        for state, phase, stage in (("initial", run.initial_phases, run.stages[0]),
                ("terminal", run.phases, run.stages[-1])):
            def evaluate(x):
                return evaluate_multitrap_objective(problem, x, config, force_epsilon=stage.force_epsilon,
                    uniformity_epsilon=stage.uniformity_epsilon)
            analytic = float(evaluate(phase).gradient_full @ direction)
            steps = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8) if label == "Minus force smoothing" and state == "terminal" else (1e-4, 1e-5, 1e-6)
            for step in steps:
                finite = (evaluate(phase + step * direction).value - evaluate(phase - step * direction).value) / (2 * step)
                rows.append(dict(configuration=label, seed=SEEDS[0], state=state, step_rad=step,
                    analytic_n_m_rad=analytic, finite_difference_n_m_rad=finite,
                    absolute_error_n_m_rad=abs(analytic - finite),
                    relative_error=abs(analytic - finite) / max(abs(analytic), abs(finite), 1e-14),
                    interpretation="absolute error governs near stationary gradients; unsmoothed norm may have a cusp"))
    return rows


def _protocol_tables(root, output, ctx, problem, configs):
    params = []
    for label, config in configs.items():
        for name, value, units in (
            ("alpha", config.alpha_force, "m^-1"), ("beta", config.beta_pressure if config.components.pressure_retention else 0., "N m^-1 Pa^-1"),
            ("gamma", config.gamma_uniformity if config.components.uniformity else 0., "1"),
            ("curvature W", config.curvature_weight_override, "1"), ("pressure epsilon relative", config.pressure_epsilon_rel, "1"),
            ("force smoothing", config.components.force_smoothing, "boolean"),
            ("stage factors", config.smooth_stage_factors, "1"), ("stage caps", config.smooth_stage_maxiters, "accepted iterations"),
            ("gtol", config.gtol, "N m^-1 rad^-1; reduced gradient infinity norm"),
            ("report threshold", config.report_gradient_tol, "N m^-1 rad^-1; reduced gradient L2")):
            params.append(dict(experiment="Triple ablation", configuration=label, parameter=name, value=str(value), units=units))
    single = fixed_single_fe_method_spec()
    params.extend([dict(experiment="Accepted main Single", configuration="FE", parameter="alpha", value=single.alpha_per_m, units="m^-1"),
        dict(experiment="Accepted main Single", configuration="FE", parameter="beta", value=single.beta_curvature_per_pa, units="N m^-1 Pa^-1"),
        dict(experiment="Accepted main Single", configuration="FE", parameter="curvature W", value="diag(1,1,1)", units="1")])
    _write_table(output, "C_parameters", params)
    notation = [
        ("p", "peak complex pressure phasor; physical pressure is its real time-harmonic field", "Pa", "global Murata-table directivity; exp(+ikr)"),
        ("U_G", "Kp |p|^2 - Kg |grad_r p|^2", "J", "corrected-sign Gor'kov surrogate; Appendix B audits sign controls"),
        ("F_G", "-grad_r U_G", "N", "acoustic radiation force surrogate"),
        ("W:Hess_r(U_G)", "sum_ij W_ij partial_i partial_j U_G", "N m^-1", "weighted curvature; not the Laplacian unless W=I"),
        ("Laplacian(U_G)", "trace Hess_r(U_G)", "N m^-1", "unweighted spatial Laplacian"),
        ("ell_j", "-W:Hess(U_j) + alpha rho_epsilon(F_j) + beta P_j", "N m^-1", "per-target loss; FE+g substitutes residual to upward effective weight"),
        ("rho_epsilon(F)", "sqrt(||F||^2+epsilon_F^2)-epsilon_F", "N", "epsilon_F from initial median force scale, separately at each fixed stage"),
        ("P_j", "sqrt(|p_j|^2 + (epsilon_relative local_RMS(p))^2)", "Pa", "pressure penalty; RMS evaluated over stencil, with its analytic phase derivative"),
        ("L", "mean_j ell_j + gamma (sqrt(var_j ell_j+epsilon_U^2)-epsilon_U)", "N m^-1", "uniformity penalizes per-target loss dispersion, not pressure variance directly"),
        ("grad_phi L", "analytic phase derivative of the discretized objective, with first phase fixed", "N m^-1 rad^-1", "255 reduced phases; reported L2 and native infinity norms distinct"),
        ("J_F", "partial_r F_total at the elastic total-force root", "N m^-1", "3D central difference; stiffness is -sym(J_F)"),
        ("a", "elastic bead radius", "0.65 mm", "distinct from control radius 2a and root-search half-width 4a"),
    ]
    _write_table(output, "C_notation", [dict(symbol=a, definition=b, units=c, interpretation=d) for a,b,c,d in notation])
    _write_table(output, "C_spatial_operators", [
        dict(quantity="Triple pressure stencil", implementation="5 x 5 x 5 global field evaluations, spacing 0.5 mm", boundary="stencil centered at each target"),
        dict(quantity="Triple grad(p), grad(U), Hess(U)", implementation="successive numpy.gradient calls; edge_order=1; Hessian symmetrized", boundary="center values reported; boundary samples use one-sided differences"),
        dict(quantity="Analytic phase derivative", implementation="differentiate exp(i phi), then apply the identical spatial stencil operators", boundary="gauge phase index zero fixed; no finite-difference phase gradient used inside optimizer"),
        dict(quantity="Elastic radiation force", implementation="partial-wave scattering and momentum-flux surface integration", boundary="independent model; exact quadrature settings recorded in C_elastic_protocol.json"),
        dict(quantity="Elastic force Jacobian", implementation="central force differences at equilibrium; step 0.1a", boundary="root acceptance requires interior root in the target-centered 4a search cube"),
    ])
    target_rows = [dict(experiment="Main Single", target_id="T1", x_m=0., y_m=0., z_m=.05)]
    target_rows.extend(dict(experiment="Accepted Triple / Appendix C", target_id=s.target_id,
        x_m=s.target_m[0], y_m=s.target_m[1], z_m=s.target_m[2]) for s in problem.stencils)
    for row in target_rows:
        row.update(stencil_half_width_m=.001, root_search_half_width_m=4 * ElasticSphere().radius_m,
            array_x_min_m=-.075, array_x_max_m=.075, array_y_min_m=-.075, array_y_max_m=.075,
            target_in_front_of_array=row["z_m"] > 0., root_search_above_array=row["z_m"] - .0026 > 0.)
    _write_table(output, "C_target_domains", target_rows)
    audit = [
        ("E4/E8", "Matched component ablations", "Three shared random starts; 12 configurations, grouped by seed. Alpha=3000 and W=diag(1,1,10) retained."),
        ("E4", "Alpha transfer", "No shared candidate among 10/100/1000 preserved Single and Triple; main alpha=10/3000 retained. C_recovered_alpha_transfer preserves the original notebook's rendered numeric table at display precision; its raw replicate cache is unavailable."),
        ("R2.3", "Pressure regularization", "The requested formulation-specific beta study is Appendix B; Triple component deletion does not substitute for it."),
        ("E5", "Native stopping", "SciPy BFGS gtol=1e-8 on reduced infinity norm; stages cap at 250 and 9750 accepted iterations. The separately reported L2 threshold=1e-3 is not the native solver stop."),
        ("E5", "Histories", "Initial stage rows repeat cumulative iteration values at stage boundaries. Counts are summed accepted nit, not history-array length; line-search calls are nfev/njev."),
        ("E5", "Archived Conventional", "Some archived branch Conventional runs used gtol=0. Main central Triple Conventional has a cumulative same-phase BFGS restart policy. Neither is rewritten as identical to staged FE stopping."),
        ("E5", "Smoothing", "Force and uniformity epsilons are fixed within each stage using initial scales. The native uniformity scale depends on initial per-target losses, so removing pressure also changes that derived scale; realized epsilons are exported."),
        ("E6", "Timing", "Fresh C rows measure CPU setup inside run_multitrap, staged solve, final evaluation, and wrapper end-to-end separately. Shared field-transfer setup is separately measured. No JIT compilation or device transfer exists in this NumPy/SciPy path."),
        ("E6", "Timing comparability", "Recovered and fresh solves were measured in separate sessions. Concurrent appendix work may affect fresh wall time. Timings are descriptive; no cross-session cost ratio or common-production timing claim is made."),
        ("E10/E12", "Mechanical numerics", "Current elastic inviscid-host partial-wave model and common SMOKE root protocol are used. Appendix A supplies convergence and surrogate validation; no fluid-sphere endpoint is promoted to this table."),
        ("E13", "Operators", "Analytic phase derivatives are checked against directional finite differences. Stencil spatial derivatives and exact-force Jacobians are distinct operators."),
        ("E14", "Domain", "Active Single/Triple coordinates, 5x5x5 stencils and 4a root-search bounds are exported. Appendix B owns its pressure volume/domain rows."),
        ("E15/E16", "Statistics", "Three paired seed-level commands; dependent Triple targets reduced to worst displacement for intervals. Bootstrap is exploratory at n=3, with all individual pairs displayed."),
        ("E15/E16", "Main benchmark", "One command per method/case in the accepted six-method benchmark cannot provide independent-start confidence intervals. IB/GS native reconstruction residuals are not FE objective gradients."),
        ("E11", "Reproduction", "This module executes from the integrated runtime and replays exact compatible caches. It is not an isolated clean-install production run; the main reproduction ZIP is missing."),
        ("E7/E9", "Generality", "Array/frequency generality is reported separately in active Appendix G1-G9; Appendix F is optimizer robustness."),
    ]
    _write_table(output, "C_review_scope", [dict(review=a, topic=b, status_and_scope=c) for a,b,c in audit])
    (output / "C_elastic_protocol.json").write_text(json.dumps(p._finite_ka_protocol(ctx), indent=2, default=str))
    (output / "C_configurations.json").write_text(json.dumps({label:config.to_payload() for label,config in configs.items()}, indent=2))


def _statistics(output, command_rows):
    frame = pd.DataFrame(command_rows)
    rows = []
    metrics = ("terminal_gradient_l2", "terminal_gradient_inf", "iterations", "worst_displacement_a", "minimum_stiffness_n_m")
    for label in LABELS:
        group = frame[frame.configuration.eq(label)]
        for metric in metrics:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            rows.append(dict(configuration=label, metric=metric, n_seed_commands=len(group), n_resolved_metric=len(values),
                median=np.median(values) if len(values) else np.nan,
                q25=np.quantile(values,.25) if len(values) else np.nan,
                q75=np.quantile(values,.75) if len(values) else np.nan,
                minimum=np.min(values) if len(values) else np.nan,
                maximum=np.max(values) if len(values) else np.nan))
    _write_table(output, "C_summary_statistics", rows)
    pairs = []
    rng = np.random.default_rng(20260905)
    for metric in ("worst_displacement_a", "minimum_stiffness_n_m"):
        wide = frame.pivot(index="seed", columns="configuration", values=metric)
        for label in LABELS[1:]:
            delta = (wide[label] - wide["Full"]).dropna().to_numpy(float)
            if len(delta):
                samples = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(axis=1)
                pairs.append(dict(configuration=label, reference="Full", metric=metric, n_paired_seeds=len(delta),
                    mean_paired_change=delta.mean(), bootstrap_95_low=np.quantile(samples,.025), bootstrap_95_high=np.quantile(samples,.975),
                    bootstrap_seed=20260905, bootstrap_resamples=5000,
                    interpretation="exploratory 3-seed interval; targets within a command are dependent"))
    _write_table(output, "C_paired_seed_bootstrap", pairs)
    return frame


def _plot(output, command_rows, jobs):
    frame = pd.DataFrame(command_rows)
    colors = ("#0072B2", "#D55E00", "#009E73")
    styles = {"font.size":14, "axes.labelsize":16, "axes.labelweight":"bold", "axes.linewidth":1.6,
        "xtick.labelsize":12, "ytick.labelsize":13, "xtick.major.width":1.5, "ytick.major.width":1.5,
        "xtick.major.size":6, "ytick.major.size":6, "legend.fontsize":12, "svg.fonttype":"none"}
    with plt.rc_context(styles):
        figure, axes = plt.subplots(2, 1, figsize=(8.4,9.8), layout="constrained")
        for ax, metric, ylabel in zip(axes, ("terminal_gradient_l2", "worst_displacement_a"),
                ("Terminal gradient L2\n(N m$^{-1}$ rad$^{-1}$)", "Worst displacement, $d/a$")):
            for index, seed in enumerate(SEEDS):
                group = frame[frame.seed.eq(seed)].set_index("configuration").reindex(LABELS)
                ax.plot(np.arange(4) + (index-1)*.035, group[metric], marker="o", markersize=8.5,
                    markeredgecolor="white", markeredgewidth=1.3, linewidth=1.7,
                    color=colors[index], label=f"Seed {index+1}")
            ax.set_xticks(np.arange(4), SHORT_LABELS)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=.22, linewidth=.8)
            ax.spines[["top", "right"]].set_visible(False)
            for tick in ax.get_xticklabels() + ax.get_yticklabels(): tick.set_fontweight("semibold")
        axes[0].set_yscale("log")
        axes[0].axhline(1e-3, color=".35", linestyle="--", linewidth=1.3)
        axes[0].legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.,1.17))
        axes[1].set_ylim(0.,max(1.05,1.08*frame.worst_displacement_a.max()))
        for letter, ax in zip("ab",axes): ax.text(-.11,1.03,f"({letter})",transform=ax.transAxes,fontweight="bold",fontsize=18)
        paths = []
        for ext in ("png", "pdf", "svg"):
            path = output / f"Figure_C1_matched_component_ablations.{ext}"
            figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            paths.append(str(path))
        plt.close(figure)
    return paths


def run(root):
    """Run or replay the fixed Appendix C study; return notebook-ready exports."""
    root = Path(root).resolve()
    output = root / "appendix_outputs" / "C"
    output.mkdir(parents=True, exist_ok=True)
    ctx = _context(root, output)
    guard = formula_self_test()
    if not guard.get("passed"):
        raise RuntimeError(f"Corrected Gor'kov formula check failed: {guard}")
    setup_started = time.perf_counter()
    problem = make_problem(ctx, contracted_targets(.8), 0., 0., case_prefix="fixed-long-triple-observation")
    transfer_setup_sec = time.perf_counter() - setup_started
    base = fixed_triple_fe_config(gtol=ctx.config.gtol)
    configs = {"Full":base}
    for label, component in zip(LABELS[1:], ("force_smoothing", "pressure_retention", "uniformity")):
        configs[label] = replace(base, components=replace(base.components, **{component:False}))
    _recover_rendered_tables(root, output)
    _protocol_tables(root, output, ctx, problem, configs)
    jobs, rows, stages, terms, histories = {}, [], [], [], []
    for seed in SEEDS:
        initial = np.random.default_rng(seed).uniform(-np.pi,np.pi,problem.n_actuators)
        for label, config in configs.items():
            recovered, source, source_dependencies = _compatible_run(root, problem, config, seed, initial)
            payload = dict(contract=SCHEMA, problem=problem.fingerprint, config=config.to_payload(),
                seed=seed, initial_phase=digest_array(initial))
            if recovered is not None:
                result, status = recovered, "recovered compatible source"
            else:
                result, status, path = ctx.cache.get_or_compute("C_triple_ablation", payload,
                    lambda:run_multitrap(problem, config, seed=seed, initial_phases=initial, record_history=True,
                        metadata={"evidence_status":"SMOKE", "role":"Appendix C matched component ablation"}))
                source = str(path.relative_to(root)); source_dependencies = dependency_snapshot()
            stage = result.stages[-1]
            ev = evaluate_multitrap_objective(problem,result.phases,config,
                force_epsilon=stage.force_epsilon,uniformity_epsilon=stage.uniformity_epsilon)
            row = dict(configuration=label,seed=seed,run_id=result.run_id,
                initial_phase_fingerprint=digest_array(initial), phase_fingerprint=digest_array(np.asarray(result.phases)),
                iterations=result.iterations, cap=10000, nfev=result.nfev, njev=result.njev,
                terminal_gradient_l2=float(np.linalg.norm(ev.gradient_reduced)),
                terminal_gradient_inf=float(np.linalg.norm(ev.gradient_reduced,np.inf)),
                native_success=stage.success,native_status=stage.status,native_message=stage.message,
                reporting_termination=result.termination_reason,report_threshold_l2=1e-3,native_gtol_inf=1e-8,
                setup_sec=result.setup_sec,solve_sec=result.solve_sec,final_evaluation_sec=result.final_evaluation_sec,
                wrapper_end_to_end_sec=result.end_to_end_sec,
                timing_scope="run_multitrap: shared transfer setup excluded; optimization and internal history included",
                timing_origin="archived source session" if recovered is not None else "C execution session; concurrent task load uncontrolled",
                cache_status=status,cache_source=source,source_dependencies_json=json.dumps(source_dependencies),
                pressure_at_target_cv=float(np.std([x.pressure_abs for x in ev.locals])/max(np.mean([x.pressure_abs for x in ev.locals]),1e-30)),
                local_loss_std_n_m=float(np.std(ev.local_losses)))
            jobs[(seed,label)] = dict(run=result, row=row, config=config)
            rows.append(row)
            for stage_row in result.stages:
                stages.append(dict(configuration=label,seed=seed,**asdict(stage_row)))
            terms.extend(_term_rows(problem,config,result,label,seed))
            for history in result.history:
                histories.append(dict(configuration=label,seed=seed,**asdict(history)))
            safe_label=label.lower().replace(" ","_")
            np.savez_compressed(output / f"C_command_{seed}_{safe_label}.npz", phase_rad=result.phases,
                initial_phase_rad=initial, history_phase_rad=result.history_phase_rad,
                positions_m=ctx.positions_m,normals=ctx.normals,targets_m=np.array([s.target_m for s in problem.stencils]),
                frequency_hz=ctx.config.frequency_hz,config_json=json.dumps(config.to_payload()))
            _write_table(output,"C_command_runs",rows)
            print(f"Appendix C: seed {seed}, {label}: {status}; {result.iterations} iterations, gradient {row['terminal_gradient_l2']:.3g}", flush=True)
    _write_table(output,"C_stage_records",stages)
    _write_table(output,"C_objective_contributions",terms)
    _write_table(output,"C_histories",histories)
    _write_table(output,"C_directional_derivative_checks",_derivative_checks(problem,configs,jobs))
    _write_table(output,"C_setup_timing",[dict(shared_transfer_setup_sec=transfer_setup_sec,formula_guard_passed=guard["passed"],
        compilation="not applicable: NumPy/SciPy CPU",device_transfer="not applicable: NumPy/SciPy CPU",
        synchronization="synchronous CPU calls",cache_replay_note="fresh setup time is not added to recovered solve times")])
    # Validate only after all timed C solves are complete.
    available = _compatible_validations(root,ctx)
    target_jobs = [(seed,label,index,np.asarray(stencil.target_m)) for seed,label in jobs
        for index,stencil in enumerate(problem.stencils)]
    def validate(item):
        seed,label,index,target=item
        result=jobs[(seed,label)]["run"]
        phase=np.asarray(result.phases);content_id=digest_array(phase)
        saved=available.get((content_id,tuple(target)))
        started=time.perf_counter()
        if saved is not None:
            mechanical,source,_=saved;status="recovered compatible frozen-phase validation";validation_wall=np.nan
        else:
            mechanical=p._finite_ka_validation(ctx,phase,target,key=f"appendix-C-{seed}-{label}-T{index+1}")
            source="C/cache/finite_ka_static_validation";status="C exact-physics cache or new validation"
            validation_wall=time.perf_counter()-started
        row=p._static_validation_row(label,mechanical)
        row.update(configuration=label,seed=seed,target_id=f"T{index+1}",phase_fingerprint=content_id,
            validation_source=source,validation_cache_status=status,
            validation_request_wall_sec=validation_wall,
            validation_timing_scope="request wall time includes cache lookup; never synthesis time")
        np.savez_compressed(output / f"C_mechanics_{seed}_{label.lower().replace(' ','_')}_T{index+1}.npz",
            target_m=target,equilibrium_m=mechanical.equilibrium.equilibrium_m,
            displacement_m=mechanical.equilibrium.displacement_m,
            force_jacobian_n_m=mechanical.force_jacobian_n_m,
            symmetric_stiffness_n_m=mechanical.symmetric_stiffness_n_m,
            stiffness_eigenvalues_n_m=mechanical.symmetric_stiffness_eigenvalues_n_m,
            phase_rad=phase,protocol_json=json.dumps(p._finite_ka_protocol(ctx),default=str))
        print(f"Appendix C mechanics: seed {seed}, {label}, T{index+1}: {status}; d/a={row['finite_ka_displacement_a']:.3g}",flush=True)
        return row
    with ThreadPoolExecutor(max_workers=3) as executor:
        target_rows=[]
        for row in executor.map(validate,target_jobs):
            target_rows.append(row)
            _write_table(output,"C_target_mechanics",target_rows)
    target_frame=pd.DataFrame(target_rows)
    for row in rows:
        group=target_frame[target_frame.seed.eq(row["seed"]) & target_frame.configuration.eq(row["configuration"])]
        all_roots=bool(group.finite_ka_root_found.all())
        row.update(target_count=len(group),all_roots=all_roots,all_locally_restoring=bool(group.finite_ka_locally_restoring.all()),
            worst_displacement_a=group.finite_ka_displacement_a.max() if all_roots else np.nan,
            minimum_stiffness_n_m=group.finite_ka_stiffness_min_n_m.min() if all_roots else np.nan)
    commands=_write_table(output,"C_command_runs",rows)
    _statistics(output,rows)
    compact=[]
    for label in LABELS:
        group=commands[commands.configuration.eq(label)]
        compact.append({"Configuration":label,
            "Gradient L2 median (N/m/rad)":f"{group.terminal_gradient_l2.median():.3g}",
            "Worst d/a median":f"{group.worst_displacement_a.median():.3g}",
            "All roots restoring":f"{int(group.all_locally_restoring.sum())}/{len(group)}"})
    _write_table(output,"C_display_summary",compact)
    figures=_plot(output,rows,jobs)
    output_sentences=[]
    full_group=commands[commands.configuration.eq("Full")]
    unsmoothed_group=commands[commands.configuration.eq("Minus force smoothing")]
    output_sentences.append(f"At the matched 10,000-iteration cap, the Full configuration reached the reduced-gradient L2 reporting threshold 1e-3 in {int((full_group.terminal_gradient_l2<=1e-3).sum())}/{len(full_group)} starts; removing force smoothing reached it in {int((unsmoothed_group.terminal_gradient_l2<=1e-3).sum())}/{len(unsmoothed_group)} starts. The latter used the same two declared stage budgets, with {int((unsmoothed_group.iterations>=10000).sum())} cap-limited runs; other native terminations are retained in C_command_runs.csv.")
    for label in LABELS:
        group=commands[commands.configuration.eq(label)]
        output_sentences.append(f"{label}: {len(group)} paired-seed commands; terminal reduced-gradient L2 median {group.terminal_gradient_l2.median():.6g}, range {group.terminal_gradient_l2.min():.6g}–{group.terminal_gradient_l2.max():.6g} N m^-1 rad^-1; {int(group.all_roots.sum())}/{len(group)} commands resolved all three elastic total-force roots; {int(group.all_locally_restoring.sum())}/{len(group)} retained positive symmetric restoring eigenvalues at all three roots. Median worst-target displacement d/a={group.worst_displacement_a.median():.6g}.")
    alpha_sentence = ("Single alpha=10 m^-1 and Triple alpha=3000 m^-1 are retained. The earlier alpha-transfer table is recovered from the attached notebook's rendered output at display precision, separately from raw numerical caches. In that original table, Single FE displacement d/a is 0.04486 at alpha=10 versus 0.1895/0.2313 at 100/1000; Triple FE+g worst displacement d/a is 0.458 at alpha=3000 versus 1.757/1.701/0.6725 at 10/100/1000."
        if (output/"C_recovered_alpha_transfer.csv").exists() else
        "Single alpha=10 m^-1 and Triple alpha=3000 m^-1 are retained. The original rendered alpha-transfer table is not present in this execution directory; no missing numerical values are reconstructed.")
    output_sentences.extend(["All numbers above are SMOKE. The three targets of each command are dependent; the three random initializations are the paired sampling unit.",
        alpha_sentence,
        "The Bottle pressure-regularization study is reported in Appendix B. Array/frequency generality is reported separately in Appendix G; Appendix F is optimizer robustness.",
        "Archived and fresh command times retain different recording sessions. No common production timing claim or isolated clean-install claim is made."])
    (output/"measured_results_smoke.txt").write_text("\n\n".join(output_sentences)+"\n")
    (output/"mock_results_sentences.txt").write_text("PRODUCTION TEMPLATES — placeholders, not measurements\n\nAcross [N] paired initializations, removing [component] changed worst-target elastic-equilibrium displacement by [estimate] a (95% paired interval [lower, upper]) and terminal phase-gradient L2 from [full] to [ablated] N m^-1 rad^-1. All configurations used alpha=3000 m^-1, W=diag(1,1,10), the declared pressure/uniformity coefficients, and the same 10,000 accepted-iteration budget.\n\nUnder the common production timing protocol, command construction required [setup] s for shared transfer setup and [solve] s for optimization; independent mechanical validation required [validation] s.\n")
    (output/"C_figure_caption.txt").write_text("Figure C1. Matched Triple component ablations at alpha=3000 m^-1 and W=diag(1,1,10). (a) Terminal reduced-phase gradient L2; dashed line denotes the reporting threshold 10^-3 N m^-1 rad^-1, distinct from the BFGS infinity-norm stopping tolerance 10^-8. (b) Worst target displacement from the prescribed coordinate to the independently resolved finite-ka elastic total-force root, normalized by bead radius a=0.65 mm. Lines connect identical initial seeds; three targets are summarized within each command and are not treated as independent replicates. All results are SMOKE.\n")
    (output/"C_manifest.json").write_text(json.dumps(dict(schema=SCHEMA,seeds=SEEDS,configurations=LABELS,
        problem_fingerprint=problem.fingerprint,formula_guard=guard,dependencies=dependency_snapshot(),
        model=p.PartialWaveForceEvaluator.model_name,all_results_smoke=True,
        source_fingerprint=fingerprintlib.fingerprint(Path(__file__).read_bytes()).hexdigest(),
        cache_access=ctx.cache.access_records),indent=2,default=str))
    return dict(appendix="C",output_dir=str(output),figures=[path for path in figures if path.endswith(".png")],figure_exports=figures,
        tables={name:str(output/f"{name}.csv") for name in ("C_display_summary","C_command_runs","C_target_mechanics","C_summary_statistics","C_paired_seed_bootstrap","C_parameters","C_notation","C_review_scope","C_recovered_alpha_transfer","C_recovered_single_seed_statistics") if (output/f"{name}.csv").exists()},
        measured_results=str(output/"measured_results_smoke.txt"),evidence_status="SMOKE")


if __name__ == "__main__":
    import sys
    print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else Path.cwd()),indent=2))
