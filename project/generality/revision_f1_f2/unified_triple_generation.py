"""Generate the selected native Triple benchmark from geometry and seeds.

The coefficients and source-resource choices are frozen by the accepted
appendix. Cold generation repeats those selected recipes rather than searching
for new alpha values. Its numerical outputs are isolated from the historical
smoke cache. The archived smoke results remain the exact historical replay.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fingerprintlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ("Conventional", "FE", "GFE", "IB", "GS")
RESOURCE_IDS = ("A07", "A08", "F20", "S06", "F28")
SCHEMA = "selected-native-triple-generation-v1"


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=lambda x:
        x.tolist() if isinstance(x, np.ndarray) else
        x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    temporary.replace(path)


def _csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def selected_recipes(project_root=ROOT):
    """Decode the explicit four-column selected-recipe table."""
    root = Path(project_root)
    compact = pd.read_csv(root / "data/selected_triple_recipes.csv")
    required = ["setting_id", "seed_index", "selected_key", "recipe_kind"]
    if compact.columns.tolist() != required or len(compact) != 70:
        raise ValueError("Expected the 70-row compact selected-recipe table")
    rows = []
    for item in compact.itertuples(index=False):
        method, candidate, alpha_text = str(item.selected_key).split(":", 2)
        if int(item.seed_index) != METHODS.index(method):
            raise ValueError(f"Method-slot mismatch for {item.setting_id}: {item.selected_key}")
        rows.append(dict(setting_id=item.setting_id, method=method, candidate_id=candidate,
            alpha_force_per_m=float(alpha_text), recipe_kind=str(item.recipe_kind)))
    return pd.DataFrame(rows).sort_values(["setting_id", "method"]).reset_index(drop=True)


def resource_geometry(setting, native_geometry):
    """The accepted resource intervention, copied from the deposited drivers."""
    sid = setting["id"]
    if sid == "A07":
        counts = (32, 38, 45, 51, 58, 64, 70, 76, 78, 90, 96, 103, 109, 114)
        rings = []
        for ring, count in enumerate(counts):
            radius = .052 + ring * .0102
            angle = (np.arange(count) + .5 * (ring % 2)) * 2 * np.pi / count
            rings.append(np.c_[radius * np.cos(angle), radius * np.sin(angle), np.zeros(count)])
        positions = np.vstack(rings)
    elif sid == "A08":
        axis = (np.arange(12) - 5.5) * .010
        x, y = np.meshgrid(axis, axis, indexing="ij")
        local = np.column_stack((x.ravel(), y.ravel(), np.zeros(144)))
        positions = np.vstack([local + [cx, cy, 0.] for cx in (-.065, .065) for cy in (-.065, .065)])
    elif sid in ("F20", "S06"):
        axis = (np.arange(24) - 11.5) * .010
        x, y = np.meshgrid(axis, axis, indexing="ij")
        positions = np.column_stack((x.ravel(), y.ravel(), np.zeros(576)))
    else:
        return native_geometry(setting)
    return positions, np.tile([0., 0., 1.], (len(positions), 1))


def _problem(setting, positions, normals, seed):
    from hat_revision_pipeline.config import RunConfig
    from hat_revision_pipeline.gorkov_core import ArrayGeometry, transfer_matrix, SOURCE_SCALE_PA_M_PER_MURATA_UNIT
    from hat_revision_pipeline.multitrap import MultiTrapProblem, TargetStencil, SphereInFluid
    from hat_revision_pipeline.triple_long_iteration import FIXED_BASE_TARGETS_M, _configs
    config = replace(RunConfig.for_mode("quick"), frequency_hz=setting["frequency_hz"],
        stencil_m=.0005 * 40000 / setting["frequency_hz"], multitrap_maxiter=30000, exact_lmax=6)
    targets = np.asarray(FIXED_BASE_TARGETS_M)
    stencils = []
    axis = np.arange(-2, 3) * config.stencil_m
    for index, target in enumerate(targets):
        points = target + np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        transfer = transfer_matrix(points, ArrayGeometry(positions, normals), config.frequency_hz,
            source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT)
        stencils.append(TargetStencil(target_id=f"T{index+1}", target_m=tuple(target),
            transfer=transfer, spacing_m=config.stencil_m, curvature_weight=np.eye(3)))
    problem = MultiTrapProblem(case_id=f'appendix-Triple-{setting["id"]}', stencils=tuple(stencils),
        material=SphereInFluid(frequency_hz=config.frequency_hz), geometry_id=setting["geometry"])
    conventional, fe = _configs(SimpleNamespace(config=config), 30000)
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(positions))
    return problem, conventional, fe, initial, targets, config


@contextmanager
def _frozen_scales(module, scales):
    original = module.estimate_smooth_scales
    module.estimate_smooth_scales = lambda *args, **kwargs: scales
    try:
        yield
    finally:
        module.estimate_smooth_scales = original


def _run_fixed(problem, config, initial, epsilon_f, epsilon_u, *, cap=4000):
    """The same fixed objective and intervention with the global FE solver."""
    from scipy.optimize import minimize
    from hat_revision_pipeline.multitrap import evaluate_multitrap_objective
    from hat_revision_pipeline.triple_long_iteration import _gauge_reduce, _gauge_expand
    from hat_revision_pipeline.compact_multitrap import CompactMultitrapEvaluator
    from hat_revision_pipeline.multitrap import _multitrap_solver_options
    use_fe = config.method == "Regularized-FE"
    compact = CompactMultitrapEvaluator(problem, config) if use_fe else None
    from hat_revision_pipeline.fe_solver_policy import require_fe_solver
    optimizer = require_fe_solver(task="Triple")[0] if use_fe else "BFGS"
    cache = {}
    values, gradients = [], []
    def evaluate(x):
        key = np.asarray(x).tobytes()
        if cache.get("key") != key:
            cache["key"] = key
            cache["value"] = (compact.evaluate(_gauge_expand(x),
                force_epsilon=epsilon_f, uniformity_epsilon=epsilon_u) if compact is not None else
                evaluate_multitrap_objective(problem, _gauge_expand(x), config,
                    force_epsilon=epsilon_f, uniformity_epsilon=epsilon_u))
        return cache["value"]
    def callback(x):
        value = evaluate(x)
        values.append(value.value)
        gradients.append(float(np.linalg.norm(value.gradient_reduced)))
    started = time.perf_counter()
    before = float(evaluate(_gauge_reduce(initial)).value)
    result = minimize(lambda x: evaluate(x).value, _gauge_reduce(initial),
        jac=lambda x: evaluate(x).gradient_reduced, method=optimizer, callback=callback,
        options=_multitrap_solver_options(optimizer, None, config.gtol, cap))
    terminal = evaluate(result.x)
    return _gauge_expand(result.x), dict(iterations=int(result.nit), optimizer=optimizer,
        evaluator_backend="compact-multitrap-discrete-v1" if use_fe else "native",
        terminal_gradient_norm=float(np.linalg.norm(terminal.gradient_reduced)),
        final_objective=float(terminal.value), initial_objective=before,
        force_epsilon=epsilon_f, uniformity_epsilon=epsilon_u,
        config=config.to_payload(), iteration_cap=cap, termination_reason=str(result.message),
        elapsed_s=time.perf_counter()-started), dict(history_objective=np.asarray(values),
            history_gradient=np.asarray(gradients))


def _objective_config(method, alpha, conventional, regularized_fe):
    """Return an objective only for methods that use the multitrap solver."""
    if method == "Conventional":
        return conventional
    if method in ("FE", "GFE"):
        return replace(
            regularized_fe,
            alpha_force=float(alpha),
            compensate_effective_gravity=method == "GFE",
        )
    if method in ("IB", "GS"):
        return None
    raise ValueError(method)


def _commands(setting, recipes, folder, seed):
    """Solve only selected recipes and the explicit warm-start prerequisites."""
    from appendix_settings import geometry
    from hat_revision_pipeline.cache import CacheStore, digest_array
    from hat_revision_pipeline import multitrap as mt, triple_long_iteration as native
    from hat_revision_pipeline import pipeline as p
    from hat_revision_pipeline.final_figures import _triple_prescribed_commands
    from hat_revision_pipeline.sota import phase_only_signature_projection_audit
    positions, normals = resource_geometry(setting, geometry)
    problem, conven_config, fe_config, cold, targets, run_config = _problem(setting, positions, normals, seed)
    _npz(folder / "geometry.npz", positions_m=positions, normals=normals,
        targets_m=targets, frequency_hz=setting["frequency_hz"])
    solved = {}
    def command(method, alpha=0., kind="cold"):
        key = f"{method}_{alpha:g}_{kind}"
        directory = folder / "commands" / key
        command_path, info_path = directory / "command.npz", directory / "solver.json"
        if key in solved:
            return solved[key]
        if command_path.exists() and info_path.exists():
            with np.load(command_path, allow_pickle=False) as saved:
                phase = saved["phase_rad"].copy()
            info = json.loads(info_path.read_text())
            if info["phase_fingerprint"] != digest_array(phase):
                raise ValueError(f"Changed command cache: {command_path}")
            from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
            dependency_valid = (kind != "warm" or info.get("initial_fe_solver_policy") == FE_SOLVER_REVISION)
            if dependency_valid and (method not in ("FE", "GFE") or info.get("fe_solver_policy") == FE_SOLVER_REVISION):
                solved[key] = phase, info, command_path
                return solved[key]
        initial = cold.copy()
        config = _objective_config(method, alpha, conven_config, fe_config)
        sid = setting["id"]
        arrays = {}
        started = time.perf_counter()
        if kind in ("continuation", "perturbed_restart"):
            reference, source_info, source_path = command(method, alpha, "cold")
            initial = reference.copy()
            proposal_seed = seed + 401 + (0 if method == "FE" else 1000)
            if kind == "perturbed_restart":
                initial += np.random.default_rng(proposal_seed).normal(0., .35, len(initial))
            phase, info, arrays = _run_fixed(problem, config, initial,
                source_info["force_epsilon"], source_info["uniformity_epsilon"])
            reference_value = float(mt.evaluate_multitrap_objective(problem, reference, config,
                force_epsilon=source_info["force_epsilon"],
                uniformity_epsilon=source_info["uniformity_epsilon"]).value)
            gain = reference_value - float(info["final_objective"])
            info.update(reference_objective=reference_value, objective_improvement=gain,
                accepted_intervention=bool(gain > 0.))
            if gain <= 0.:
                _npz(directory / "rejected_command.npz", phase_rad=phase, targets_m=targets, **arrays)
                _json(directory / "rejected_solver.json", info)
                phase = reference.copy()
                initial = reference.copy()
                arrays = {}
                info = dict(source_info, rejected_intervention=kind,
                    accepted_intervention=False, intervention_objective_improvement=gain)
            info.update(initial_phase_source=str(source_path.relative_to(folder)),
                proposal_seed=proposal_seed if kind == "perturbed_restart" else None,
                kick_std_rad=.35 if kind == "perturbed_restart" else 0.)
        elif kind == "warm":
            source_method = "FE" if sid in ("A08", "F20") else "GFE"
            source_alpha = 300. if sid in ("A08", "F20") else 3000. if sid == "A07" else 30000.
            initial, _, source_path = command(source_method, source_alpha, "cold")
            if method == "FE" and sid == "S06":
                _, cold_info, _ = command("FE", alpha, "cold")
                phase, info, arrays = _run_fixed(problem, config, initial,
                    cold_info["force_epsilon"], cold_info["uniformity_epsilon"])
            elif sid == "A07":
                config = replace(config, smooth_stage_factors=(.01,), smooth_stage_maxiters=(4000,))
                module = native if method == "Conventional" else mt
                scales = module.estimate_smooth_scales(problem, native._gauge_expand(native._gauge_reduce(cold)), config)
                with _frozen_scales(module, scales):
                    if method == "Conventional":
                        run = native._run_cumulative_conventional(problem, config, seed=seed,
                            initial_phases=initial, cumulative_cap=4000, checkpoints=(1000, 4000),
                            record_stalled_terminal=True)
                        phase = np.asarray(run.terminal_phase_rad)
                        info = dict(iterations=run.accepted_iterations,
                            terminal_gradient_norm=run.terminal_gradient_norm,
                            force_epsilon=run.force_epsilon, uniformity_epsilon=run.uniformity_epsilon,
                            final_objective=run.terminal_objective, termination_reason=run.termination_reason,
                            config=config.to_payload(), segments=list(run.segment_records), iteration_cap=4000)
                    else:
                        run = mt.run_multitrap(problem, config, seed=seed, initial_phases=initial, record_history=True)
                        phase = np.asarray(run.phases)
                        info = dict(run.summary_row(), config=config.to_payload(),
                            force_epsilon=run.stages[-1].force_epsilon,
                            uniformity_epsilon=run.stages[-1].uniformity_epsilon,
                            iteration_cap=4000)
            else:
                scales = native.estimate_smooth_scales(problem, native._gauge_expand(native._gauge_reduce(cold)), config)
                phase, info, arrays = _run_fixed(problem, config, initial, 0.,
                    config.smooth_stage_factors[0] * scales.loss_std_scale)
            info["initial_phase_source"] = str(source_path.relative_to(folder))
        elif method in ("FE", "GFE"):
            run = mt.run_multitrap(problem, config, seed=seed, initial_phases=cold, record_history=True,
                metadata=dict(display_method=method, selected_recipe=True, alpha_tuning=False))
            phase = np.asarray(run.phases)
            info = dict(run.summary_row(), config=config.to_payload(),
                stages=[asdict(stage) for stage in run.stages],
                force_epsilon=run.stages[-1].force_epsilon,
                uniformity_epsilon=run.stages[-1].uniformity_epsilon, iteration_cap=30000)
            arrays = dict(history_objective=np.asarray([x.objective for x in run.history]),
                history_gradient=np.asarray([x.gradient_norm for x in run.history]))
        elif method == "Conventional":
            run = native._run_cumulative_conventional(problem, config, seed=seed, initial_phases=cold,
                cumulative_cap=30000, checkpoints=(10000, 20000, 30000), record_stalled_terminal=True)
            phase = np.asarray(run.terminal_phase_rad)
            keys = sorted(run.checkpoint_phases_rad)
            arrays = dict(checkpoint_iterations=np.asarray(keys),
                checkpoint_phase_rad=np.stack([run.checkpoint_phases_rad[k] for k in keys]))
            info = dict(iterations=run.accepted_iterations, terminal_gradient_norm=run.terminal_gradient_norm,
                force_epsilon=run.force_epsilon, uniformity_epsilon=run.uniformity_epsilon,
                final_objective=run.terminal_objective, config=config.to_payload(),
                termination_reason=run.termination_reason, segments=list(run.segment_records), iteration_cap=30000)
        elif method in ("IB", "GS"):
            context = SimpleNamespace(positions_m=positions, normals=normals, config=run_config,
                cache=CacheStore(folder / "linear_cache", namespace=SCHEMA), memo={}, recompute=False)
            if method == "GS" and sid in ("F20", "S06"):
                transfer, signature = p._multivortex_transfer_factory(context, targets, spec=p.IB_VORTEX_SPEC)()
                run = phase_only_signature_projection_audit(transfer, signature,
                    iterations=20000 if sid == "F20" else 1000, control_group_size=8)
                phase = np.asarray(run.phase_rad)
                info = dict(iterations=run.iterations, native_solver_metadata=dict(run.metadata))
            else:
                run = _triple_prescribed_commands(context, targets)[f"{method}_matched"]
                phase = np.asarray(run.phase_rad)
                info = dict(iterations=run.solver_result.iterations,
                    native_solver_metadata=dict(run.solver_result.metadata))
            info.update(terminal_gradient_norm=np.nan, initialization="Native deterministic prescription")
        else:
            raise ValueError(method)
        info.update(seed=int(seed), elapsed_s=time.perf_counter()-started,
            phase_fingerprint=digest_array(phase), initial_phase_fingerprint=digest_array(initial),
            generation_schema=SCHEMA, method=method, alpha_force_per_m=alpha, recipe_kind=kind)
        if kind == "warm":
            from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
            info["initial_fe_solver_policy"] = FE_SOLVER_REVISION
        if method in ("FE", "GFE"):
            from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
            info['fe_solver_policy'] = FE_SOLVER_REVISION
            info['optimizer'] = 'BFGS'
            info['evaluator_backend'] = 'compact-multitrap-discrete-v1'
        _npz(command_path, phase_rad=phase, targets_m=targets, initial_phase_rad=initial, **arrays)
        _json(info_path, info)
        solved[key] = phase, info, command_path
        return solved[key]
    outputs = []
    for row in recipes.itertuples(index=False):
        kind = str(row.recipe_kind)
        if kind not in {"cold", "warm", "continuation", "perturbed_restart"}:
            raise ValueError(f"Unknown recipe kind: {kind}")
        phase, info, path = command(row.method, float(row.alpha_force_per_m), kind)
        outputs.append((row, phase, info, path))
    return positions, normals, targets, outputs


def _validate(setting, positions, normals, targets, row, phase, info, path, folder, mode):
    from compute_appendix_case import finite_ka_protocol, make_evaluator, mechanics_row
    from hat_revision_pipeline.exact_validator import validate_static_trap
    sid = setting["id"]
    protocol = finite_ka_protocol(mode, sid)
    if sid == "A07":
        protocol["max_nfev_per_start"] = max(protocol["max_nfev_per_start"], 100)
    if sid in ("F20", "S06", "F28"):
        protocol["max_nfev_per_start"] = max(protocol["max_nfev_per_start"], 80)
    evaluator = None
    result_rows = []
    for index, target in enumerate(targets):
        destination = folder / "mechanics" / row.method / f"T{index+1}.json"
        if destination.exists():
            record = json.loads(destination.read_text())
            if record["phase_fingerprint"] == info["phase_fingerprint"]:
                result_rows.append(record)
                continue
            # The changed command needs new mechanics; other commands reuse theirs.
        if evaluator is None:
            evaluator = make_evaluator(positions, normals, setting["frequency_hz"], phase, protocol)
        started = time.perf_counter()
        def trial(width, starts, label):
            result = validate_static_trap(evaluator, target, search_half_width_a=width,
                initial_offsets_a=np.asarray(starts), max_nfev=protocol["max_nfev_per_start"],
                numerical_root_tolerance=protocol["numerical_root_tolerance"], jacobian_step_a=protocol["jacobian_step_a"])
            item = mechanics_row(row.method, result)
            _json(destination.with_name(f"T{index+1}_{label}.json"), item)
            return item
        record = trial(4., protocol["root_starts_a"], "native_width")
        native_found = bool(record["finite_ka_root_found"])
        width, search_method = 4., "native_cartesian_starts"
        if not native_found and sid in ("F20", "S06", "F28"):
            candidate = np.asarray([record[f"finite_ka_candidate_{axis}_m"] for axis in "xyz"])
            starts = np.clip((candidate-target)/protocol["sphere"]["radius_m"], -7.999, 7.999)[None, :]
            record = trial(8., starts, "expanded_continuation")
            width, search_method = 8., "expanded_from_native_candidate"
            if not record["finite_ka_root_found"]:
                starts = np.vstack((np.zeros((1, 3)), np.eye(3), -np.eye(3), 4*np.eye(3), -4*np.eye(3)))
                record = trial(8., starts, "expanded_axis_starts")
                search_method = "expanded_thirteen_axis_starts"
        found = bool(record["finite_ka_root_found"])
        equilibrium = np.asarray([record[f"finite_ka_equilibrium_{axis}_m"] for axis in "xyz"])
        displacement = equilibrium-target
        record.update(setting_id=sid, method=row.method, candidate_id=row.candidate_id,
            task="Triple", target_id=f"T{index+1}", target_index=index,
            alpha_force_per_m=float(row.alpha_force_per_m), seed=info["seed"],
            source_count=len(positions), frequency_hz=setting["frequency_hz"],
            phase_fingerprint=info["phase_fingerprint"], command_path=str(path.relative_to(folder)),
            root_found=found, native_width_root_found=native_found,
            root_search_half_width_a=width, search_method=search_method,
            root_max_nfev_per_start=protocol["max_nfev_per_start"],
            pressure_abs_at_exact_equilibrium_pa=float(abs(evaluator.field.pressure(equilibrium[None, :])[0])) if found else np.nan,
            pressure_abs_at_intended_target_pa=float(abs(evaluator.field.pressure(target[None, :])[0])),
            displacement_m=float(np.linalg.norm(displacement)) if found else np.nan,
            displacement_um=float(np.linalg.norm(displacement)*1e6) if found else np.nan,
            terminal_iteration=info["iterations"], terminal_gradient_norm=info["terminal_gradient_norm"],
            validation_elapsed_s=time.perf_counter()-started, evidence_mode=f"generated_{mode}",
            **{f"target_{axis}_m":float(target[j]) for j, axis in enumerate("xyz")},
            **{f"displacement_{axis}_m":float(displacement[j]) if found else np.nan for j, axis in enumerate("xyz")},
            **{f"displacement_{axis}_um":float(displacement[j]*1e6) if found else np.nan for j, axis in enumerate("xyz")})
        _json(destination, record)
        result_rows.append(record)
    _json(folder / "mechanics" / row.method / "protocol.json", protocol)
    return result_rows


def generate_triple_data(output_root, *, project_root=ROOT, mode="smoke", cache_policy="resume",
                         setting_ids=None, seeds=None):
    """Generate all selected Triple commands and independent physical roots.

    Smoke uses seed 260869; full uses ten consecutive paired seeds. Native
    objective iteration caps are unchanged. Full upgrades the existing native
    validator from lmax=6 to 8 and its declared root-start/tolerance settings.
    ``fresh`` creates a new directory; ``resume`` reuses only this generator's
    matching command and mechanics caches. No archived outcomes are copied.
    """
    if mode not in ("smoke", "full") or cache_policy not in ("resume", "fresh"):
        raise ValueError("Expected smoke/full and resume/fresh")
    root = Path(project_root).resolve()
    for location in (root, root.parent / "generality_package", root.parent / "generality_package/runtime/src"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    from appendix_settings import SETTINGS
    from hat_revision_pipeline.cache import digest_array
    settings = [dict(s) for s in SETTINGS if setting_ids is None or s["id"] in setting_ids]
    if setting_ids is not None and {s["id"] for s in settings} != set(setting_ids):
        raise ValueError("Unknown setting ID")
    recipes = selected_recipes(root)
    recipes = recipes.loc[recipes.setting_id.isin([s["id"] for s in settings])]
    parameters = json.loads((root / "data/generation_protocol.json").read_text())
    default_seeds = parameters["modes"][mode]["triple_seeds"]
    seeds = list(map(int, seeds)) if seeds is not None else list(map(int, default_seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("At least one unique seed is required")
    output = Path(output_root).resolve() / mode / "triple_generation"
    if cache_policy == "fresh" and output.exists():
        output = output.with_name(output.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    plan = dict(schema=SCHEMA, mode=mode, seeds=seeds, settings=settings,
        selected_recipes=json.loads(recipes.to_json(orient="records")),
        cold_iteration_cap=int(parameters["triple"]["cold_iteration_cap"]),
        intervention_iteration_cap=int(parameters["triple"]["intervention_iteration_cap"]),
        smoke_evidence="Regenerated selected recipes, not the archived campaign",
        selection="Frozen coefficients, source resources and intervention types; no alpha search")
    plan_path = output / "generation_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("The generation cache has a different plan; use cache_policy='fresh'.")
    _json(plan_path, plan)
    all_rows = []
    for setting in settings:
        selected = recipes[recipes.setting_id.eq(setting["id"])]
        if set(selected.method) != set(METHODS) or len(selected) != len(METHODS):
            raise ValueError(f'Missing selected recipes for {setting["id"]}')
        for seed in seeds:
            folder = output / setting["id"] / f"seed{seed}"
            positions, normals, targets, commands = _commands(setting, selected, folder, seed)
            for row, phase, info, path in commands:
                records = _validate(setting, positions, normals, targets, row, phase, info, path, folder, mode)
                for record in records:
                    record["command_path"] = str(path.relative_to(output))
                    record["geometry_path"] = str((folder / "geometry.npz").relative_to(output))
                all_rows.extend(records)
                _csv(output / "outcomes.csv", pd.DataFrame(all_rows))
            print(f'Triple {setting["id"]}, seed {seed}: generated five methods and three target records each.', flush=True)
    outcomes = pd.DataFrame(all_rows)
    summaries = []
    for (sid, method), frame in outcomes.groupby(["setting_id", "method"], sort=False):
        resolved = frame[frame.root_found.astype(bool)]
        summary = dict(setting_id=sid, method=method, expected_targets=len(frame),
            root_found_count=len(resolved), paired_starts=len(seeds),
            alpha_force_per_m=float(frame.iloc[0].alpha_force_per_m),
            source_count=int(frame.iloc[0].source_count), evidence_mode=f"generated_{mode}",
            aggregation="All seeds and all three targets; unresolved roots retained as NaN")
        for column, stem in (("displacement_m", "displacement"), ("pressure_abs_at_exact_equilibrium_pa", "pressure")):
            unit = "m" if stem == "displacement" else "pa"
            for statistic in ("median", "min", "max"):
                summary[f"{stem}_{statistic}_{unit}"] = float(getattr(resolved[column], statistic)()) if len(resolved) else np.nan
        summaries.append(summary)
    _csv(output / "summary.csv", pd.DataFrame(summaries))
    receipt = dict(output_root=str(output), evidence_mode=f"generated_{mode}",
        settings=len(settings), paired_seeds=len(seeds), selected_commands=len(settings)*5*len(seeds),
        target_rows=len(outcomes), alpha_search_performed=False,
        cached_historical_outcomes_used=False, objectives="Native Gor'kov synthesis; independent finite-ka elastic validation")
    _json(output / "generation_receipt.json", receipt)
    return receipt


def activate_generated_triple(generated_output, active_root):
    """Point an isolated active appendix at its freshly generated Triple data.

    This changes only an explicit selector file in ``active_root``. If needed,
    the newly generated inputs are copied beneath that root for portability.
    The archived baseline and resource-release inputs are left unchanged.
    """
    import shutil
    generated = Path(generated_output).resolve()
    active = Path(active_root).resolve()
    plan = json.loads((generated / "generation_plan.json").read_text())
    receipt = json.loads((generated / "generation_receipt.json").read_text())
    if receipt["evidence_mode"] != f'generated_{plan["mode"]}':
        raise ValueError("Generated Triple mode identity differs")
    if not generated.is_relative_to(active):
        destination = active / "data/generated_triple"
        if destination.exists():
            existing = json.loads((destination / "generation_plan.json").read_text())
            if existing != plan:
                raise ValueError("Active generated Triple inputs use a different plan")
        shutil.copytree(generated, destination, dirs_exist_ok=True)
        generated = destination
    frame = pd.read_csv(generated / "outcomes.csv")
    command_columns = ["setting_id", "method", "seed", "candidate_id", "command_path", "phase_fingerprint", "geometry_path"]
    _csv(generated / "command_index.csv", frame[command_columns].drop_duplicates())
    marker = dict(schema=SCHEMA, mode=plan["mode"], evidence_mode=receipt["evidence_mode"],
        relative_data_root=str(generated.relative_to(active)),
        outcome_rows=len(frame), command_rows=len(frame[command_columns].drop_duplicates()))
    _json(active / "data/generated_triple_source.json", marker)
    return marker
