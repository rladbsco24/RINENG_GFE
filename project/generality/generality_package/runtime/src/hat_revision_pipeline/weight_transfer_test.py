"""Paired FE/FE+g weight-transfer smoke test; main figure producers are untouched."""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from . import pipeline as p
from .cache import digest_array
from .gorkov_core import Condition,SingleTargetObjective,reduce_gauge
from .fixed_fe_endpoints import fixed_single_fe_method_spec
from .main_triple_endpoint import ensure_main_triple_endpoint
from .multitrap import SphereInFluid,effective_weight_force_target_n,run_multitrap,evaluate_multitrap_objective
from .sota import solve_corrected_gorkov_fe

CANDIDATES=(10.,100.,1000.)

def context(root):
    ctx=p.prepare_pipeline(run_mode='quick',recompute=False,output_root=Path(root)/'smoke_outputs')
    bank=ensure_main_triple_endpoint(ctx)
    ix,iy=ctx.branch_bank.root_target
    rows=ctx.artifact.trajectory_table
    seed=int(rows[rows.method.eq('Conventional') & rows.ix.eq(ix) & rows.iy.eq(iy)].sort_values('seed').iloc[0].seed)
    return ctx,bank,seed

def run_case(ctx,bank,single_seed,task,method,alpha,wzz):
    started=time.perf_counter();compensated=method=='FE+g'
    run_id=f'{task}_{method.replace("+","plus")}_a{alpha:g}_wz{wzz:g}'
    target_force=effective_weight_force_target_n(SphereInFluid(frequency_hz=ctx.config.frequency_hz))
    if task=='Single':
        seed=single_seed
        spec=replace(fixed_single_fe_method_spec(),alpha_per_m=alpha,curvature_weight=np.diag([1.,1.,wzz]))
        condition=Condition(label='fixed-single-fe-alpha10',positions=ctx.positions_m,
            target_m=tuple(p.MAIN_TARGET_M),frequency_hz=ctx.config.frequency_hz,
            curvature_weight=spec.curvature_weight,normals=ctx.normals)
        objective=SingleTargetObjective(condition,spec,force_target_n=target_force if compensated else None)
        initial=np.random.default_rng(seed).uniform(-np.pi,np.pi,objective.n_transducers)
        payload=dict(contract='fixed-single-fe-gravity-endpoint-v1' if compensated else 'fixed-single-fe-endpoint-v2-uniform-10000-cap',
            alpha_per_m=alpha,iteration_cap=10000,target_m=p.MAIN_TARGET_M.tolist(),seed=seed,
            initial_phase=digest_array(initial),gtol=ctx.config.gtol,positions=digest_array(ctx.positions_m),normals=digest_array(ctx.normals))
        if compensated:payload['force_target_n']=target_force.tolist()
        if wzz!=1:payload.update(contract='weight-transfer-single-v1',curvature_weight=spec.curvature_weight.tolist())
        run,status,_=ctx.cache.get_or_compute('fixed_single_fe_gravity_endpoint' if compensated else 'fixed_single_fe_endpoint',payload,
            lambda:solve_corrected_gorkov_fe(objective,initial,maxiter=10000,gtol=ctx.config.gtol))
        phase=np.asarray(run.phase_rad);targets=np.asarray([p.MAIN_TARGET_M]);solve_sec=float(run.history_wall_s[-1])
        grad=run.gradient_norm;termination=run.message
        local_metrics=[objective.metrics(reduce_gauge(phase,0))]
        config=dict(alpha_per_m=alpha,curvature_weight=spec.curvature_weight.tolist(),force_target_n=(target_force if compensated else np.zeros(3)).tolist(),gtol=ctx.config.gtol)
        stages=[]
    else:
        node=bank['objects'][(1,1)];base=node['fe_config'];seed=bank['seed'];initial=bank['initial_phase']
        config_obj=replace(base,alpha_force=alpha,curvature_weight_override=((1.,0.,0.),(0.,1.,0.),(0.,0.,wzz)),compensate_effective_gravity=compensated)
        problem=node['problem'];targets=node['targets_m'];config=config_obj.to_payload()
        if alpha==3000 and wzz==10 and not compensated:
            run=node['FE'];status='cached'
        else:
            payload=dict(contract='weight-transfer-triple-v1',problem=problem.fingerprint,config=config,seed=seed,initial_phase=digest_array(initial))
            kind='fe_g_weight_transfer_triple'
            if alpha==3000 and wzz==10 and compensated:
                payload=dict(contract='fig6-fixed-effective-gravity-compensated-triple-fe-v1',problem=problem.fingerprint,
                    base_config=base.to_payload(),compensated_config=config,seed=seed,initial_phase_fingerprint=digest_array(initial),
                    fixed_iteration_cap=10000,fixed_upward_force_target_n=target_force.tolist())
                kind='final_fig6_vertical_compensated_fe'
            run,status,_=ctx.cache.get_or_compute(kind,payload,
                lambda:run_multitrap(problem,config_obj,seed=seed,initial_phases=initial,record_history=True,
                    metadata={'display_method':method,'experiment_role':'paired weight-transfer SMOKE'}))
        phase=np.asarray(run.phases);solve_sec=float(run.end_to_end_sec);grad=run.terminal_gradient_norm;termination=run.termination_reason
        evaluation=evaluate_multitrap_objective(problem,phase,config_obj,force_epsilon=run.stages[-1].force_epsilon,uniformity_epsilon=run.stages[-1].uniformity_epsilon)
        local_metrics=[dict(gorkov_force_n=x.force_vector_n,weighted_curvature_n_m=x.curvature_n_m) for x in evaluation.locals]
        stages=[asdict(s) for s in run.stages]
    print(f'{run_id}: optimizer {status}; {run.iterations} iterations; measured command {solve_sec:.3f} s',flush=True)
    records=[]
    # Validation follows each solve; parallel validation never overlaps a timed solve.
    from concurrent.futures import ThreadPoolExecutor
    def validate(index):
        target=targets[index]
        key=f'weight-transfer-{run_id}-seed-{seed}-T{index+1}'
        if task=='Single' and alpha==10 and wzz==1:key=f'fig5-single-fixed-{method}-seed-{seed}'
        if task=='Triple' and alpha==3000 and wzz==10:
            key=f'fig6-triple-FE-gz-target-{index+1}' if compensated else f'fig5-triple-FE-not_applicable-target-{index+1}'
        result=p._finite_ka_validation(ctx,phase,target,key=key)
        row=p._static_validation_row(method,result)
        acoustic=np.asarray(local_metrics[index]['gorkov_force_n']);residual=acoustic-(target_force if compensated else np.zeros(3))
        field=p._array_field(ctx,phase);eq=np.asarray(result.equilibrium.equilibrium_m)
        row.update(run_id=run_id,task=task,alpha_per_m=alpha,wzz=wzz,seed=seed,target_id=f'T{index+1}',target_index=index,
            phase_fingerprint=digest_array(phase),initial_phase_fingerprint=digest_array(initial),iterations=run.iterations,iteration_cap=10000,
            command_time_s=solve_sec,command_timing_scope='solver wall time' if task=='Single' else 'run_multitrap end-to-end wall time',
            command_cache_status=status,terminal_gradient_norm=grad,termination=termination,
            surrogate_residual_norm_n=float(np.linalg.norm(residual)),surrogate_radiation_force_z_n=float(acoustic[2]),
            weighted_curvature_n_m=float(local_metrics[index]['weighted_curvature_n_m']),
            pressure_abs_at_exact_equilibrium_pa=float(abs(field.pressure(eq[None,:]))[0]) if row['finite_ka_root_found'] else np.nan,
            config_json=json.dumps(config),stages_json=json.dumps(stages),evidence_status='SMOKE',manuscript_numerics=False)
        return row
    with ThreadPoolExecutor(max_workers=min(3,len(targets))) as pool:records=list(pool.map(validate,range(len(targets))))
    directory=ctx.output_root/'weight_test';directory.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(directory/(run_id+'.csv'),index=False)
    np.savez_compressed(directory/(run_id+'.npz'),phase_rad=phase,initial_phase_rad=initial,targets_m=targets)
    print(f'{run_id}: exact displacement a = '+', '.join(f"{r["finite_ka_displacement_a"]:.5g}" for r in records)+f'; elapsed {time.perf_counter()-started:.1f} s',flush=True)
    return records

def summarize(records):
    raw=pd.DataFrame(records);rows=[]
    for run_id,g in raw.groupby('run_id',sort=False):
        f=g.iloc[0]
        rows.append(dict(run_id=run_id,task=f.task,method=f.method,alpha_per_m=f.alpha_per_m,wzz=f.wzz,
            median_displacement_a=g.finite_ka_displacement_a.median(),worst_displacement_a=g.finite_ka_displacement_a.max(),
            median_dz_um=g.finite_ka_displacement_z_m.median()*1e6,min_stiffness_mn_m=g.finite_ka_stiffness_min_n_m.min()*1e3,
            all_roots=bool(g.finite_ka_root_found.all()),all_restoring=bool(g.finite_ka_locally_restoring.all()),
            target_count=len(g),command_time_s=f.command_time_s,iterations=f.iterations,terminal_gradient_norm=f.terminal_gradient_norm,
            median_pressure_pa=g.pressure_abs_at_exact_equilibrium_pa.median(),evidence_status='SMOKE'))
    return raw,pd.DataFrame(rows)

def run_primary(root):
    ctx,bank,seed=context(root);records=[]
    jobs=[('Single',m,a,1.) for a in CANDIDATES for m in ('FE','FE+g')]
    jobs += [('Triple',m,a,10.) for a in (3000.,*CANDIDATES) for m in ('FE','FE+g')]
    for task,method,alpha,wzz in jobs:
        records.extend(run_case(ctx,bank,seed,task,method,alpha,wzz))
        raw,summary=summarize(records)
        raw.to_csv(ctx.output_root/'weight_test/weight_test_raw.csv',index=False)
        summary.to_csv(ctx.output_root/'weight_test/weight_test_summary.csv',index=False)
        (ctx.output_root/'weight_test/cache_access.json').write_text(json.dumps(ctx.cache.access_records,indent=2))
    return ctx,bank,seed,raw,summary
