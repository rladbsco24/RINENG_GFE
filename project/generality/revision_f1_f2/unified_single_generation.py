"""Generate the Single appendix from fixed coefficients and recorded proposals.

Four cold paired starts and the eight FE/GFE references are retained in both
modes. The archived retained Conventional proposal inputs are explicit data;
they are rerun by the deposited native optimizer, with objective acceptance
checked again. Full mode refines field and decision sampling, not the chosen
coefficients or the completed experiment's proposal selection.
"""
from __future__ import annotations
from dataclasses import fields, replace
import fingerprintlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "generality_package/runtime/src"))
from appendix_settings import SETTINGS, geometry
from appendix_io import save_npz, write_bytes
from hat_revision_pipeline.cache import digest_array, digest_payload
from hat_revision_pipeline.gorkov_core import Condition, SingleTargetObjective, method_spec
from hat_revision_pipeline.fixed_fe_endpoints import fixed_single_fe_method_spec
from hat_revision_pipeline.sota import FESolveResult, solve_corrected_gorkov_fe


def _json(path, value):
    write_bytes(Path(path), (json.dumps(value, indent=2, default=str) + "\n").encode())


def _run(objective, initial, output, *, maxiter, gtol, identity, reuse=True):
    protocol = dict(identity, initial_phase_fingerprint=digest_array(initial), maxiter=maxiter, gtol=gtol)
    from hat_revision_pipeline.fe_solver_policy import is_fe_objective, FE_SOLVER_REVISION
    if is_fe_objective(objective):
        protocol['fe_solver_policy'] = FE_SOLVER_REVISION
    key = digest_payload(protocol)
    if reuse and output.exists():
        with np.load(output, allow_pickle=False) as data:
            if str(data["generation_key"]) == key:
                return {name: data[name].copy() for name in data.files}
            # Regenerate only this changed command, retaining all other files.
    run = solve_corrected_gorkov_fe(objective, initial, maxiter=maxiter, gtol=gtol)
    arrays = {field.name: np.asarray(json.dumps(getattr(run, field.name), sort_keys=True)
        if isinstance(getattr(run, field.name), dict) else getattr(run, field.name))
        for field in fields(FESolveResult)}
    arrays.update(initial_phase_rad=np.asarray(initial), generation_key=np.asarray(key))
    save_npz(output, **arrays)
    _json(output.with_suffix(".json"), protocol)
    return arrays


def _accepted_history(run):
    """Keep the initial state and each native accepted step once."""
    count = int(run["iterations"])
    phase = run["history_phase_rad"][:count+1]
    objective = run["history_objective"][:count+1]
    if not np.allclose(np.exp(1j*phase[-1]), np.exp(1j*run["phase_rad"]), atol=1e-10, rtol=0):
        raise ValueError("Native accepted history does not end at its reported phase")
    return phase, objective


def generate_single_data(project_root=ROOT, *, mode="smoke", cache_policy="resume", setting_ids=None):
    if mode not in {"smoke", "full"} or cache_policy not in {"resume", "fresh"}:
        raise ValueError("Unknown execution mode or cache policy")
    root = Path(project_root).resolve()
    data_root = root / "data"
    proposal_root = data_root / "recorded_proposals"
    manifest = json.loads((proposal_root / "manifest.json").read_text())
    proposal_by_file = {row["input_file"]: row for row in manifest["trials"]}
    cases = {(row["setting_id"], row["seed"]): row for row in manifest["cases"]}
    with np.load(proposal_root / manifest["bank_file"], allow_pickle=False) as packed:
        proposal_bank = {key: packed[key].copy() for key in packed.files}
    parameters = json.loads((data_root / "generation_protocol.json").read_text())
    single_protocol = parameters["single"]
    mode_protocol = parameters["modes"][mode]
    paired_seeds = tuple(map(int, single_protocol["paired_seeds"]))
    ids = [setting["id"] for setting in SETTINGS] if setting_ids is None else list(setting_ids)
    settings = {setting["id"]: setting for setting in SETTINGS}
    weights = pd.read_csv(root.parent / "tables/selected_weights.csv")
    weights = weights.set_index(["setting_id", "method"])
    run_root = data_root / "generated_single/runs"
    rows, endpoint_rows, selections, provenance = [], [], [], []
    setting_time = time.perf_counter()
    for sid in ids:
        setting = settings[sid]
        positions, normals = geometry(setting)
        objectives = {}
        for method in ("Conventional", "FE", "GFE"):
            row = None if method == "Conventional" else weights.loc[sid, method]
            spec = method_spec("Conventional") if row is None else replace(
                fixed_single_fe_method_spec(), alpha_per_m=float(row.alpha_per_m),
                curvature_weight=tuple(tuple(v) for v in np.diag([
                    row.curvature_weight_x, row.curvature_weight_y, row.curvature_weight_z])))
            force = np.zeros(3) if row is None else np.asarray([
                row.force_target_x_n, row.force_target_y_n, row.force_target_z_n])
            objectives[method] = SingleTargetObjective(Condition(label=f"appendix-{sid}-{method}",
                positions=positions, normals=normals, target_m=(0., 0., .05),
                frequency_hz=setting["frequency_hz"], curvature_weight=spec.curvature_weight),
                spec, force_target_n=force)
        runs = {}
        for seed in paired_seeds:
            initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(positions))
            for method in ("FE", "GFE", "Conventional"):
                objective = objectives[method]
                identity = dict(setting=setting, positions=digest_array(positions), normals=digest_array(normals),
                    method=method, seed=seed, alpha_per_m=float(objective.method.alpha_per_m),
                    force_target_n=objective.force_target_n.tolist(),
                    role="paired cold command", mode=mode)
                path = run_root / sid / f"{method}_seed{seed}_cold.npz"
                run = _run(objective, initial, path,
                    maxiter=int(single_protocol["cold_maxiter"]), gtol=float(single_protocol["cold_gtol"]),
                           identity=identity, reuse=cache_policy == "resume")
                runs[method, seed] = (run, path)
                rows.append(dict(setting_id=sid, method=method, seed=seed, stage="cold",
                    iterations=int(run["iterations"]), gradient_norm=float(run["gradient_norm"]),
                    objective=float(run["objective"]), phase_fingerprint=digest_array(run["phase_rad"]),
                    source=str(path.relative_to(root))))
        reference_keys = [(method, seed) for method in ("FE", "GFE") for seed in paired_seeds]
        references = np.stack([runs[key][0]["phase_rad"] for key in reference_keys])
        reference_ids = np.array([f"{method}_seed{seed}" for method, seed in reference_keys])
        reference_methods = np.array([method for method, _ in reference_keys])
        reference_seeds = np.array([seed for _, seed in reference_keys])
        reference_alphas = np.array([objectives[method].method.alpha_per_m for method, _ in reference_keys])
        reference_hashes = np.array([digest_array(phase) for phase in references])
        bank_hash = digest_payload(dict(ids=reference_ids.tolist(), phase_hashes=reference_hashes.tolist(),
                                        alphas=reference_alphas.tolist()))
        save_npz(data_root / "solution_set_acs/references" / f"{sid}_bank.npz",
            setting_id=sid, reference_phase_rad=references, reference_ids=reference_ids,
            reference_methods=reference_methods, reference_seeds=reference_seeds,
            reference_alpha_per_m=reference_alphas, reference_phase_fingerprint=reference_hashes,
            reference_source_paths=np.array([str(runs[key][1].relative_to(root)) for key in reference_keys]),
            bank_fingerprint=bank_hash)
        for seed in paired_seeds:
            objective = objectives["Conventional"]
            cold, _ = runs["Conventional", seed]
            initial_history, initial_values = _accepted_history(cold)
            segments, values = [initial_history], [initial_values]
            incumbent = cold
            accepted_boundaries = [int(cold["iterations"])]
            accepted = []
            for filename in cases[sid, seed]["stage_input_files"]:
                proposal = proposal_by_file[filename]
                initial = proposal_bank[proposal["bank_key"]][int(proposal["row_index"])].copy()
                if initial.shape != (len(positions),) or not np.isfinite(initial).all():
                    raise ValueError(f"Invalid compact proposal input: {filename}")
                path = run_root / sid / filename
                candidate = _run(objective, initial, path,
                    maxiter=int(proposal["maxiter"]), gtol=float(proposal["gtol"]),
                    identity=dict(setting=setting, mode=mode, method="Conventional", seed=seed,
                        input_id=filename, role="recorded proposal replay", stage=proposal["stage"]),
                    reuse=cache_policy == "resume")
                improved = float(candidate["objective"]) < float(incumbent["objective"])
                if improved:
                    history, objective_values = _accepted_history(candidate)
                    # The proposal's initial state is an explicit intervention.
                    # It is not counted as an accepted optimizer iteration.
                    segments.append(history[1:]); values.append(objective_values[1:])
                    incumbent = candidate
                    accepted.append(int(proposal["stage"]))
                    accepted_boundaries.append(accepted_boundaries[-1] + int(candidate["iterations"]))
                rows.append(dict(setting_id=sid, method="Conventional", seed=seed,
                    stage=int(proposal["stage"]), accepted=improved,
                    iterations=int(candidate["iterations"]), objective=float(candidate["objective"]),
                    gradient_norm=float(candidate["gradient_norm"]),
                    phase_fingerprint=digest_array(candidate["phase_rad"]),
                    source=str(path.relative_to(root))))
            history = np.concatenate(segments)
            objective_values = np.concatenate(values)
            similarity = np.empty(len(history))
            for start in range(0, len(history), 256):
                block = history[start:start+256]
                similarity[start:start+len(block)] = np.max(
                    np.abs(np.exp(1j*block) @ np.exp(-1j*references).T) / len(positions), axis=1)
            total = len(history)-1
            assert total == accepted_boundaries[-1]
            terminal = incumbent["phase_rad"]
            save_npz(data_root / "compact_acs/traces" / f"{sid}_Conventional_seed{seed}.npz",
                accepted_iteration=np.arange(len(history)), cosine_similarity=similarity,
                reference_ids=reference_ids, bank_fingerprint=bank_hash,
                cold_iterations=int(cold["iterations"]), total_iterations=total,
                completed_search_boundaries=np.asarray(accepted_boundaries),
                terminal_phase_rad=terminal, objective=objective_values)
            scores = np.abs(np.exp(1j*(terminal-references)).mean(axis=1))
            winner = int(np.argmax(scores))
            endpoint_rows.append(dict(setting_id=sid, seed=seed, trajectory_method="Conventional",
                original_iterations=int(cold["iterations"]), total_iterations=total,
                cold_cosine_similarity=float(similarity[int(cold["iterations"])]),
                terminal_cosine_similarity=float(similarity[-1]),
                cold_objective=float(cold["objective"]), terminal_objective=float(incumbent["objective"]),
                accepted_stages=len(accepted), terminal_nearest_reference_id=reference_ids[winner],
                bank_fingerprint=bank_hash, source_count=len(positions)))
            if seed == 260828:
                campaign = data_root / "solution_correspondence/campaign" / sid
                save_npz(campaign / "array_geometry.npz", positions_m=positions, normals=normals,
                         frequency_hz=setting["frequency_hz"])
                selected = {method: int(np.flatnonzero(reference_methods == method)[
                    np.argmax(scores[reference_methods == method])]) for method in ("FE", "GFE")}
                selection = dict(setting_id=sid, method=str(reference_methods[winner]), seed=seed,
                    reference_seed=int(reference_seeds[winner]), reference_id=str(reference_ids[winner]),
                    phase_fingerprint=str(reference_hashes[winner]), conventional_phase_fingerprint=digest_array(terminal),
                    bank_fingerprint=bank_hash, alpha_per_m=float(reference_alphas[winner]),
                    terminal_phase_cosine=float(scores[winner]),
                    criterion="Nearest member of the fixed same-setting eight-endpoint FE/GFE bank")
                command_provenance = []
                for method in ("Conventional", "FE", "GFE"):
                    run = incumbent if method == "Conventional" else runs[reference_keys[selected[method]]][0]
                    phase = run["phase_rad"]
                    suffix = "FEplusg" if method == "GFE" else method
                    save_npz(campaign / "commands" / f"Single_{suffix}.npz", phase_rad=phase,
                        target_m=np.array([0., 0., .05]), alpha_per_m=objectives[method].method.alpha_per_m)
                    save_npz(campaign / "paired_single" / f"{method}_pair0.npz",
                        phase_rad=phase, history_phase_tail_rad=run["history_phase_rad"][-20:],
                        iterations=run["iterations"])
                    command_provenance.append(dict(method=method, phase_fingerprint=digest_array(phase),
                        source="Generated native cold solve or retained recorded proposal replay"))
                _json(data_root / "solution_correspondence/records" / f"{sid}.json",
                      dict(selection=selection, commands=command_provenance))
                selections.append(selection)
                from pressure_data import extract
                from decision_data import analyze
                extract(sid, data_root / "pressure", points=int(mode_protocol["pressure_grid_points_per_axis"]),
                        input_root=data_root / "solution_correspondence/campaign")
                analyze(campaign, data_root / "decision", grid_size=51)
                # The final wider view uses the same generated endpoints and
                # native directions, as in the accepted appendix.
                import final_decision_data as fd
                fd.ROOT = root
                fd.HALF_RANGE = 3.
                fd.GRID_SIZE = int(mode_protocol["decision_grid_points_per_axis"])
                for path in (data_root / "final_decision" / f"{sid}_decision.npz",
                             data_root / "final_decision" / f"{sid}_decision_metadata.json"):
                    path.unlink(missing_ok=True)
                fd.build_case(sid)
                selection_file = data_root / "decision_reveal/selection.json"
                if selection_file.exists():
                    window = json.loads(selection_file.read_text())
                    half_range = float(window["half_ranges_l2_rad"][sid])
                    if half_range != 3.:
                        import decision_reveal as reveal
                        reveal.ROOT = root
                        folder = data_root / f"decision_reveal/range{half_range:g}"
                        for path in (folder / f"{sid}_decision.npz", folder / f"{sid}_decision_metadata.json"):
                            path.unlink(missing_ok=True)
                        reveal.sample((sid, half_range))
        write_bytes(data_root / "generated_single/solver_records.csv", pd.DataFrame(rows).to_csv(index=False).encode())
        provenance.append(dict(setting_id=sid, bank_fingerprint=bank_hash, paired_starts=4,
                               reference_endpoints=8, source_count=len(positions)))
        print(f"{sid}: Single commands and figure inputs generated", flush=True)
    write_bytes(data_root / "compact_acs/endpoints.csv", pd.DataFrame(endpoint_rows).to_csv(index=False).encode())
    write_bytes(data_root / "solution_correspondence/selection.csv", pd.DataFrame(selections).to_csv(index=False).encode())
    old_protocol = json.loads((data_root / "compact_acs/protocol.json").read_text())
    maximum = max(row["total_iterations"] for row in endpoint_rows)
    old_protocol.update(evidence_mode=mode, generation="native recorded-proposal replay",
        matching_rule="fixed_same_setting_endpoint_set", reference_method="FE_or_GFE",
        paired_starts=4, references_per_setting=8,
        displayed_iteration_max=max(10000, int(np.ceil(maximum/1000)*1000)),
        max_actual_accepted_iterations=maximum,
        bank_fingerprint={row["setting_id"]: row["bank_fingerprint"] for row in provenance})
    _json(data_root / "compact_acs/protocol.json", old_protocol)
    from unified_generation import regenerate_acs_table
    regenerate_acs_table(root)
    protocol = dict(mode=mode, paired_starts=4, references_per_setting=8, settings=provenance,
        cold_iteration_cap=int(single_protocol["cold_maxiter"]), cold_gtol=float(single_protocol["cold_gtol"]),
        pressure_grid=int(mode_protocol["pressure_grid_points_per_axis"]),
        decision_grid=int(mode_protocol["decision_grid_points_per_axis"]),
        decision_half_width_l2_rad="Frozen per-setting viewing choices:3 or20, same for compared methods",
        proposal_scope=manifest["selection_scope"], alpha_search=False,
        elapsed_s=time.perf_counter()-setting_time)
    _json(data_root / "generated_single/protocol.json", protocol)
    return protocol
