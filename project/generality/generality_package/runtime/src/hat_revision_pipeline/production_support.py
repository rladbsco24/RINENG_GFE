"""Production D/E orchestration; importing this module executes no experiments.

Historical phase commands are immutable. Production mechanical records and
machine-specific transition timings use separate namespaces. D3 only revalidates
the retained offset selections; it never grows the seed bank or reranks them.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import fingerprintlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from . import appendix_de as de, appendix_support_gfe as support, appendix_d3_fields as spatial
from . import pipeline as p
from .cache import digest_array

SCHEMA = "production-frozen-support-D-E-v1"
D1_REPEATS = 30


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _content_id(path):
    return fingerprintlib.fingerprint(Path(path).read_bytes()).hexdigest()


def _csv(path, frame):
    path=Path(path)
    temporary=path.with_name(path.name+".writing")
    frame.to_csv(temporary,index=False)
    temporary.replace(path)


def _complete_timing_block(path,identity_hash):
    try:
        frame=pd.read_csv(path)
        return (len(frame)==D1_REPEATS and frame.environment_hash.eq(identity_hash).all()
                and sorted(frame.repetition.tolist())==list(range(D1_REPEATS)))
    except (OSError,ValueError,AttributeError,KeyError,pd.errors.ParserError,pd.errors.EmptyDataError):
        return False


def _machine(campaign_id):
    return dict(campaign_id=campaign_id, python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
        processor=platform.processor(), cpu_count=os.cpu_count(),
        threadpools=threadpool_info(), requested_blas_threads=1)


def describe(root):
    """Read-only finite-design inventory; no synthesis, force or timing work."""
    root = Path(root).resolve()
    source = root / de.D_SOURCE
    old = pd.read_csv(source / "mechanics_offset_results.csv")
    d2 = pd.read_csv(source / "mechanics_offset_command_manifest.csv")
    source_d3 = root / "appendix_outputs/D"
    d3 = pd.read_csv(source_d3 / "D3_GFE_quantization_target_results.csv")
    candidates = pd.read_csv(root / "appendix_outputs/E/candidate_command_manifest.csv")
    with np.load(source / "continuous_FE_phases.npz", allow_pickle=False) as saved:
        historical_commands = len(saved.files)
    expected = dict(D1_historical_commands=15, D1_outcomes=315, D1_timing_repeats=30,
                    D1_timing_observations=9450, D2_records=120, D3_records=76,
                    D3_continuous_commands=2, D3_quantized_commands=36,
                    E1_candidates=290, E2_candidates=20)
    actual = dict(D1_historical_commands=historical_commands, D1_outcomes=len(old),
                  D1_timing_repeats=D1_REPEATS, D1_timing_observations=len(old)*D1_REPEATS,
                  D2_records=historical_commands+len(d2), D3_records=len(d3),
                  D3_continuous_commands=d3[d3.levels.eq(0)].task.nunique(),
                  D3_quantized_commands=len(d3[d3.levels.gt(0)][["task","levels","method"]].drop_duplicates()),
                  E1_candidates=int(candidates.geometry.eq("opposed").sum()),
                  E2_candidates=int(candidates.geometry.eq("single").sum()))
    if actual != expected:
        raise ValueError(f"Frozen support design changed: {actual}; expected {expected}")
    transfer = json.loads((root / "manuscript_appendix_transfer/transfer_experiments.json").read_text())
    return dict(schema=SCHEMA, counts=actual, production_execution="NOT RUN by this inventory",
                D3_phase_levels=list(support.LEVELS), D3_display_phase_levels=32,
                original_CDE_transfer_scope=transfer["scope"],
                original_CDE_campaigns="Prepared only; never invoked by production_support.run",
                source_hashes={str(path.relative_to(root)):_content_id(path) for path in (
                    source / "continuous_FE_phases.npz", source / "mechanics_offset_phase_indices.npz",
                    source_d3 / "D3_GFE_source_commands.npz", source_d3 / "D3_GFE_quantization_target_results.csv",
                    root / "appendix_outputs/E/all_candidate_commands_profiles.npz")})


def _d1_worker(root, output, campaign_id):
    """Fresh native transition timings in an isolated source-runtime process."""
    root, output = Path(root).resolve(), Path(output).resolve()
    source = root / de.D_SOURCE
    # Only original deterministic model/transition kernels are imported. The
    # source benchmark's run/build_models_and_phases entry points are never called.
    sys.path.insert(0, str(source / "code"))
    benchmark = importlib.import_module("prototypes.benchmark_mechanics_aware_offset")
    ds = importlib.import_module("prototypes.discrete_solver")
    mechanics = importlib.import_module("prototypes.mechanics_aware_offset")
    with threadpool_limits(limits=1):
        environment = _machine(campaign_id)
        identity = dict(schema=SCHEMA+"-D1-timing", environment=environment,
                        repetitions=D1_REPEATS, warmups_per_outcome=1,
                        source_phase_fingerprint=_content_id(source / "continuous_FE_phases.npz"),
                        source_hashes={name:_content_id(source / "code/prototypes" / name) for name in (
                            "benchmark_mechanics_aware_offset.py", "discrete_solver.py", "mechanics_aware_offset.py")},
                        source_backend_fingerprint=_content_id(source / "code/inputs/hat_geometry_branch_benchmark.py"))
        identity_hash = fingerprintlib.fingerprint(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = output / "D1_timing" / identity_hash
        directory.mkdir(parents=True, exist_ok=True)
        _json(directory / "protocol.json", dict(**identity,
            scope="Native transition call including discrete objective/candidate ranking; model construction and mechanics-reference setup outside timer; root validation and plotting excluded.",
            clock="time.perf_counter_ns", order="15 deposited target IDs; seven Q values; nearest, objective offset, mechanics offset; 30 serial repetitions after one unmeasured warm-up per outcome."))
        historical = pd.read_csv(source / "mechanics_offset_results.csv")
        endpoints = pd.read_csv(source / "continuous_FE_endpoints.csv")
        with np.load(source / "continuous_FE_phases.npz", allow_pickle=False) as saved:
            phases = {key:saved[key].copy() for key in saved.files}
        with np.load(source / "mechanics_offset_phase_indices.npz", allow_pickle=False) as saved:
            hardware = {key:saved[key].copy() for key in saved.files}
        setup_path = directory / "setup.csv"
        setups = pd.read_csv(setup_path).to_dict("records") if setup_path.exists() else []
        positions = benchmark.square_points(16, pitch=.010)
        normals = np.tile([0.,0.,1.], (len(positions),1))
        all_rows = []
        for case_index, (case, target) in enumerate(benchmark.target_specs()):
            pending = [(q,method) for q in de.LEVELS for method in ("fixed_nearest","best_offset","mechanics_offset")
                       if not _complete_timing_block(directory / f"{case}_Q{q}_{method}.csv",identity_hash)]
            if pending:
                started = time.perf_counter_ns()
                geometry = benchmark.Geometry(case, positions, normals, target,
                    "single-sided planar lattice", "16x16, 10 mm pitch")
                model = benchmark.build_quadratic_model(geometry, h=benchmark.FD_H_M,
                    calibration_samples=benchmark.CALIBRATION_SAMPLES, seed=benchmark.CALIBRATION_SEED+case_index)
                problem = ds.TransitionProblem.from_model(model, benchmark.fe_spec())
                model_setup_ms = (time.perf_counter_ns()-started)*1e-6
                endpoint = endpoints[endpoints.case_id.eq(case)].iloc[0]
                continuous_root = np.asarray([endpoint.root_x_m,endpoint.root_y_m,endpoint.root_z_m])
                started = time.perf_counter_ns()
                local = mechanics.build_local_mechanics_cache(geometry,continuous_root,benchmark.FD_H_M)
                reference = mechanics.reference_from_continuous_command(local,phases[case])
                mechanics_setup_ms = (time.perf_counter_ns()-started)*1e-6
                setups.append(dict(case_id=case, model_setup_ms=model_setup_ms,
                                   mechanics_reference_setup_ms=mechanics_setup_ms,
                                   environment_hash=identity_hash))
                _csv(setup_path,pd.DataFrame(setups))
                for q, method in pending:
                    if method == "fixed_nearest":
                        operation = lambda q=q: ds.fixed_gauge_nearest(problem,phases[case],q)
                    elif method == "best_offset":
                        operation = lambda q=q: ds.best_global_offset_rounding(problem,phases[case],q)
                    else:
                        operation = lambda q=q: mechanics.mechanics_aware_best_offset_rounding(problem,phases[case],q,local,reference=reference)
                    warmup = operation()
                    quantized = warmup.quantization if method == "mechanics_offset" else warmup
                    original = historical[historical.case_id.eq(case)&historical.levels.eq(q)&historical.method.eq(method)].iloc[0]
                    np.testing.assert_allclose(quantized.objective,original.objective,rtol=1e-8,atol=1e-9)
                    if method == "mechanics_offset":
                        np.testing.assert_array_equal(quantized.indices,hardware[original.command_key])
                    stable_indices = quantized.indices.copy()
                    gc.collect()
                    block=[]
                    for repetition in range(D1_REPEATS):
                        started=time.perf_counter_ns()
                        result=operation()
                        elapsed=(time.perf_counter_ns()-started)*1e-6
                        quantized=result.quantization if method == "mechanics_offset" else result
                        np.testing.assert_array_equal(quantized.indices,stable_indices)
                        block.append(dict(case_id=case,levels=q,method=method,repetition=repetition,
                            transition_ms=elapsed,phase_indices_fingerprint=digest_array(stable_indices),
                            objective=float(quantized.objective),environment_hash=identity_hash))
                    _csv(directory / f"{case}_Q{q}_{method}.csv",pd.DataFrame(block))
                    print(f"D1 production timing {case} Q={q} {method}: 30 uncached calls",flush=True)
            for q in de.LEVELS:
                for method in ("fixed_nearest","best_offset","mechanics_offset"):
                    block=pd.read_csv(directory / f"{case}_Q{q}_{method}.csv")
                    if len(block)!=D1_REPEATS or block.environment_hash.nunique()!=1 or block.environment_hash.iloc[0]!=identity_hash:
                        raise ValueError("Incomplete or foreign D1 timing block")
                    all_rows.extend(block.to_dict("records"))
        raw=pd.DataFrame(all_rows)
        assert len(raw)==9450
        _csv(output / "D1_production_timing_observations.csv",raw)
        summary=raw.groupby(["case_id","levels","method"]).transition_ms.agg(
            transition_time_median_ms="median",transition_time_q25_ms=lambda x:x.quantile(.25),
            transition_time_q75_ms=lambda x:x.quantile(.75),timing_repeats="count").reset_index()
        _csv(output / "D1_production_timing_summary.csv",summary)
        _csv(output / "D1_production_setup_observations.csv",pd.DataFrame(setups))
        _json(output / "D1_production_timing_manifest.json",dict(**identity,identity_hash=identity_hash,
            records=len(raw),outcomes=len(summary),cache_namespace=str(directory.relative_to(root)),
            evidence_status="PRODUCTION",fresh_machine_transition_timings=True))


def _timings(root, output, settings):
    env=os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    env["PYTHONPATH"]=str(root / "runtime/src")+os.pathsep+env.get("PYTHONPATH","")
    with (output / "D1_production_timing.log").open("a") as stream:
        subprocess.run([sys.executable,"-m","hat_revision_pipeline.production_support","--d1-worker",
                        str(root),str(output),settings["campaign_id"]],check=True,cwd=root,env=env,
                       stdout=stream,stderr=subprocess.STDOUT)
    return pd.read_csv(output / "D1_production_timing_summary.csv")


def _d3(root,ctx,output):
    source=root / "appendix_outputs/D"
    source_parameters=json.loads((source / "D3_GFE_source_parameters.json").read_text())
    source_records={row["task"]:row for row in source_parameters["commands"]}
    with np.load(source / "D3_GFE_source_commands.npz",allow_pickle=False) as saved:
        arrays={key:saved[key].copy() for key in saved.files}
    np.testing.assert_array_equal(arrays["positions_m"],ctx.positions_m)
    np.testing.assert_array_equal(arrays["normals"],ctx.normals)
    records=[];commands=[]
    for task in ("Single","Triple"):
        phase=arrays[task+"_phase_rad"]
        assert digest_array(phase)==source_records[task]["phase_fingerprint"]
        command=dict(task=task,phase=phase,targets=arrays[task+"_targets_m"],results=[],
                     seed=int(source_records[task]["seed"]),source=source_records[task]["source"],
                     solver_iteration_cap=int(source_records[task]["solver_iteration_cap"]))
        for index,target in enumerate(command["targets"]):
            result=p._finite_ka_validation(ctx,phase,target,key=f"production-D3-{task}-continuous-T{index+1}")
            command["results"].append(result)
            row=support._validation_row(command,result,result,0,"Continuous GFE",index)
            row.update(evidence_status="PRODUCTION",reference_root_found=bool(result.equilibrium.numerical_root_found))
            records.append(row)
            destination=output / "gfe_validated_commands"
            destination.mkdir(exist_ok=True)
            support._npz(destination / f"{task}_continuous_T{index+1}.npz",phase_rad=phase,target_m=target,
                root_m=result.equilibrium.equilibrium_m,root_found=np.asarray(result.equilibrium.numerical_root_found),
                residual_force_n=result.equilibrium.residual_force_n,force_jacobian_n_m=result.force_jacobian_n_m,
                stiffness_n_m=result.symmetric_stiffness_n_m)
        commands.append(command)
        for q in support.LEVELS:
            for method,label in (("Nearest","nearest"),("Elastic mechanics offset","offset")):
                for index,(target,reference) in enumerate(zip(command["targets"],command["results"])):
                    frozen=source / "gfe_validated_commands" / f"{task}_Q{q}_{label}_T{index+1}.npz"
                    with np.load(frozen,allow_pickle=False) as saved:
                        discrete=saved["phase_rad"].copy()
                    result=p._finite_ka_validation(ctx,discrete,target,key=f"production-D3-{task}-Q{q}-{label}-T{index+1}")
                    row=support._validation_row(command,result,reference,q,method,index)
                    row.update(evidence_status="PRODUCTION",phase_fingerprint=digest_array(discrete),
                        source_command_file=str(frozen.relative_to(root)),
                        reference_root_found=bool(reference.equilibrium.numerical_root_found))
                    if not reference.equilibrium.numerical_root_found:
                        row["additional_equilibrium_shift_um"]=np.nan
                    records.append(row)
                    support._npz(destination / frozen.name,phase_rad=discrete,target_m=target,
                        root_m=result.equilibrium.equilibrium_m,root_found=np.asarray(result.equilibrium.numerical_root_found),
                        residual_force_n=result.equilibrium.residual_force_n,force_jacobian_n_m=result.force_jacobian_n_m,
                        stiffness_n_m=result.symmetric_stiffness_n_m,
                        candidate_roots_m=np.asarray([v.equilibrium_m for v in result.equilibrium.candidates]),
                        candidate_residuals_scaled=np.asarray([v.residual_scaled_norm for v in result.equilibrium.candidates]))
                    print(f"D3 production {task} Q={q} {label} T{index+1}: root={row['root_found']}",flush=True)
    table=pd.DataFrame(records)
    assert len(table)==76 and int(table.levels.gt(0).sum())==72
    table.to_csv(output / "D3_GFE_quantization_target_results.csv",index=False)
    summaries=[]
    for (task,q,method),part in table.groupby(["task","levels","method"],sort=False):
        valid=part.root_found & part.reference_root_found
        summaries.append(dict(task=task,levels=int(q),method=method,targets=len(part),
            roots_found=int(part.root_found.sum()),locally_restoring=int(part.locally_restoring.sum()),
            worst_added_shift_um=float(part.additional_equilibrium_shift_um.max()) if valid.all() else np.nan,
            worst_absolute_target_error_um=float(part.absolute_target_error_um.max()) if part.root_found.all() else np.nan))
    pd.DataFrame(summaries).to_csv(output / "D3_GFE_quantization_summary.csv",index=False)
    selection_scope=("The two continuous commands and 36 discrete commands are frozen from the deposited finite deployment study. "
        "Mechanics-aware offset selection used its archived quick elastic protocol; this figure independently revalidates those "
        "same selections using the full production elastic protocol. Offset candidates are not reranked and no synthesis is repeated.")
    figures,caption,coordinates=spatial.plot(output,ctx,commands,evidence_status="PRODUCTION",
                                           selection_scope=selection_scope,allow_unresolved=True)
    (output / "D3_figure_caption.txt").write_text(caption)
    source_archive=output / "deposited_D3_selection"
    source_archive.mkdir(exist_ok=True)
    for name in ("D3_GFE_source_commands.npz","D3_GFE_source_parameters.json","D3_selected_offsets.csv",
                 "D3_offset_candidates.csv","D3_provenance.json","D3_elastic_factorization_audit.csv"):
        shutil.copy2(source / name,source_archive / name)
    support._write_phase_resolution_provenance(output)
    _json(output / "D3_production_provenance.json",dict(schema=SCHEMA,evidence_status="PRODUCTION",
        source_command_scope=selection_scope,phase_levels=list(support.LEVELS),display_levels=32,
        continuous_commands=2,quantized_commands=36,target_records=76,production_protocol=p._finite_ka_protocol(ctx),
        validation_namespace=str(ctx.output_root.relative_to(root)),
        production_source_fingerprint=_content_id(__file__),source_phase_archive_fingerprint=_content_id(source / "D3_GFE_source_commands.npz"),
        new_continuous_optimizations=0,new_offset_selection_calls=0))
    return dict(figures=figures,table=table,coordinates=coordinates)


def _e(root,output):
    source=root / "appendix_outputs/E"
    output.mkdir(parents=True,exist_ok=True)
    for name in ("all_candidate_commands_profiles.npz","candidate_command_manifest.csv","execution_provenance.json"):
        shutil.copy2(source / name,output / name)
    shutil.copytree(source / "tables",output / "tables",dirs_exist_ok=True)
    provenance=json.loads((source / "execution_provenance.json").read_text())
    with support._reading_titles():
        result=de._render_e(root,provenance,output=output)
    text=(output / "measured_results_smoke.txt").read_text()
    text=text.replace("SMOKE measured results — Appendix E","Finite-design source replay — Appendix E")
    text=text.replace("The exact source numerical kernels were reexecuted with unchanged morphology definitions, radius candidates and historical FE reference.",
        "The complete deposited deterministic command/profile bank is reused with unchanged morphology definitions, radius candidates and historical FE reference; no numerical candidate synthesis is repeated in this production replay.")
    text += "\nAll source synthesis_wall_s values are historical provenance; they are excluded from current-machine production timing comparisons.\n"
    (output / "measured_results.txt").write_text(text)
    (output / "measured_results_smoke.txt").unlink()
    result["results"]=str(output / "measured_results.txt")
    manifest=pd.read_csv(output / "candidate_command_manifest.csv")
    manifest["timing_origin"]="historical source execution; not production-machine timing"
    manifest.to_csv(output / "candidate_command_manifest.csv",index=False)
    timing=pd.read_csv(output / "timing_scope_summary.csv")
    timing["timing_origin"]="historical source execution; excluded from production timing comparisons"
    timing.to_csv(output / "timing_scope_summary.csv",index=False)
    _json(output / "production_replay_provenance.json",dict(schema=SCHEMA,
        evidence_scope="Complete finite deterministic source design; numerical arrays reused unchanged",
        E1_candidates=290,E2_candidates=20,new_candidate_syntheses=0,
        phase_profile_archive_fingerprint=_content_id(source / "all_candidate_commands_profiles.npz"),
        source_execution_provenance=provenance,timing_origin="Historical only"))
    return result


def run(root,ctx):
    """Execute ready production support after the user launches Run All."""
    root=Path(root).resolve()
    if not ctx.config.full or ctx.output_root.resolve()!=root / "production_outputs":
        raise ValueError("Production support requires full ctx rooted at study/production_outputs")
    settings=ctx.memo.get("production_settings")
    if settings is None or int(settings["discretization_timing_repeats"])!=30:
        raise ValueError("Production support requires the declared 30-repeat production settings")
    protocol=p._finite_ka_protocol(ctx)
    if protocol["numerics"]["lmax"]!=8 or len(protocol["root_starts_a"])!=19 or protocol["max_nfev_per_start"]!=70:
        raise ValueError("Production support mechanics must use lmax8 and 19×70 root search")
    inventory=describe(root)
    base=ctx.output_root / "appendices"
    d_output,e_output=base / "D",base / "E"
    d_output.mkdir(parents=True,exist_ok=True)
    for name in ("source_unit_amplitude_probe.csv","field_mapping_audit.csv"):
        shutil.copy2(root / "appendix_outputs/D" / name,d_output / name)
    with threadpool_limits(limits=1):
        timing=_timings(root,d_output,settings)
        elastic,contract=de._d_elastic(root,ctx=ctx,output=d_output)
        if len(elastic)!=120:
            raise ValueError("D2 production must retain all120 finite-design outcomes")
        with support._reading_titles(allow_unresolved=True):
            d=de._render_d(root,elastic,contract,output=d_output,timing_results=timing,evidence_status="PRODUCTION")
        d3=_d3(root,ctx,d_output)
        e=_e(root,e_output)
    # Existing presentation crosswalks operate on an appendix root. Here the
    # captions are finalized directly to keep source and production trees apart.
    d_caption=(d_output / "figure_captions.txt").read_text()
    reference=support._d2_continuous_reference(d_output,strict=False)
    if reference["resolved_reference_targets"]:
        d_caption=d_caption.replace("(b) Absolute distance from requested target.",
            "(b) Absolute distance from requested target. The dashed line is the production continuous-FE median "
            f"({reference['absolute_target_error_um']:.3f} µm across {reference['resolved_reference_targets']} resolved targets).")
    else:
        d_caption += " No continuous reference root resolved under this production protocol; the baseline line is absent.\n"
    (d_output / "figure_captions.txt").write_text(d_caption)
    e_caption=(e_output / "figure_captions.txt").read_text().replace(
        "These are pressure-morphology comparisons with 256 paired phase commands shared by two panels.",
        "The two opposed arrays share 256 independent phase coordinates. The complete finite design contains 290 commands: two methods, five separations and 29 radii.")
    (e_output / "figure_captions.txt").write_text(e_caption)
    measured=(d_output / "measured_results_smoke.txt").read_text().replace(
        "FE+g quantization was not tested. Source historical timings are not a new cross-method common-platform timing study.",
        "D1 transition timings were freshly measured on the recorded production machine; historical Rayleigh mechanics remain source evidence. D3 separately validates the frozen current-GFE deployment commands with production elastic numerics.")
    (d_output / "measured_results.txt").write_text(measured)
    (d_output / "measured_results_smoke.txt").unlink()
    d["results"]=str(d_output / "measured_results.txt")
    figures=d["figures"]+d3["figures"]+e["figures"]
    result=dict(figures=figures,D=d,E=e,D3=dict(figures=d3["figures"],records=len(d3["table"])),
                inventory=inventory,output=str(base),original_CDE_transfer="Prepared only; no experiments invoked")
    _json(base / "support_production_manifest.json",result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--d1-worker",nargs=3,metavar=("ROOT","OUTPUT","CAMPAIGN"))
    arguments=parser.parse_args()
    if arguments.d1_worker:
        _d1_worker(*arguments.d1_worker)
