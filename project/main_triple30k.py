"""Execute and export the matched main Triple 30,000-iteration comparison.

This campaign changes only the main optimization ceilings. Appendix C retains
its frozen 10,000-iteration references. Cached replays preserve measured times.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from importlib.metadata import version
from threadpoolctl import threadpool_info
import os
import platform
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime/src'))
from hat_revision_pipeline.cache import digest_array, dependency_snapshot

CAMPAIGN = 'main-triple-30000-matched-session-v1'


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def command_similarity(first, second):
    first, second = np.exp(1j*np.asarray(first)), np.exp(1j*np.asarray(second))
    return float(np.clip(abs(np.vdot(first, second))/(np.linalg.norm(first)*np.linalg.norm(second)),0,1))


def export(ctx, bank, gfe, prescribed, ad, benchmark=None):
    from hat_revision_pipeline.triple_long_iteration import RESTART_POLICY
    node = bank['objects'][(1,1)]
    conventional, fe = node['Conventional'], node['FE']
    output = ctx.output_root / 'tables' / 'main_triple30k'
    output.mkdir(parents=True, exist_ok=True)
    rows, phases, stages = [], {}, []
    from hat_revision_pipeline import pipeline as pipeline
    transfer, signature = pipeline._multivortex_transfer_factory(ctx, node['targets_m'], spec=pipeline.IB_VORTEX_SPEC)()
    capacity=np.sum(np.abs(transfer),axis=1)
    normalized_transfer=transfer/capacity[:,None]
    ad_command=np.exp(1j*np.asarray(ad.phase_rad))
    ad_pressure=normalized_transfer@ad_command
    ad_target=np.abs(signature)/max(float(np.sqrt(np.mean(np.abs(signature)**2))),1e-8)
    # Exact analytic derivative of the unchanged normalized Diff-PAT amplitude loss.
    phase_derivative=np.real((np.conj(ad_pressure)/np.maximum(np.abs(ad_pressure),np.finfo(float).tiny))[:,None]*(1j*normalized_transfer*ad_command[None,:]))
    ad_gradient=(2/len(signature))*((np.abs(ad_pressure)-ad_target)@phase_derivative)
    for method, run, config in [('Conventional', conventional, node['conventional_config']), ('FE',fe,node['fe_config']), ('GFE',gfe['run'],gfe['config'])]:
        is_conventional = method == 'Conventional'
        phase = run.terminal_phase_rad if is_conventional else run.phases
        phases[method] = np.asarray(phase)
        records = list(run.segment_records) if is_conventional else [asdict(stage) for stage in run.stages]
        for row in records: stages.append(dict(method=method, **row))
        last = records[-1]
        rows.append(dict(method=method, task='Triple', iteration_ceiling=30000,
            actual_iterations=run.accepted_iterations if is_conventional else run.iterations,
            stage_iteration_ceilings=str(tuple(config.smooth_stage_maxiters)),
            terminal_gradient_norm=run.terminal_gradient_norm,
            gradient_definition='L2 norm of the analytic objective gradient over reduced phases (first actuator phase fixed)',
            terminal_objective=run.terminal_objective if is_conventional else run.final_objective,
            termination_reason=run.termination_reason,
            final_solver_status=last['status'] if is_conventional else last.get('optimizer_status',last.get('status')),
            final_solver_message=last['message'] if is_conventional else last.get('optimizer_message',last.get('message')),
            nfev=run.nfev,njev=run.njev,seed=bank['seed'],
            initial_phase_content_id=digest_array(bank['initial_phase']), phase_content_id=digest_array(phase),
            reported_command_time_s=run.solve_sec if is_conventional else run.end_to_end_sec,
            setup_s=np.nan if is_conventional else run.setup_sec,
            solve_s=run.solve_sec,
            final_evaluation_s=np.nan if is_conventional else run.final_evaluation_sec,
            timing_scope=('native cumulative BFGS interval, including checkpoint callbacks; excludes transfer construction, initial phase generation and initial scale estimation' if is_conventional else 'native run_multitrap end-to-end: internal phase handling and scale estimation, compact staged BFGS, terminal evaluation and result construction; excludes transfer construction and externally supplied initial phase generation'),
            stopping_rule='same terminal phase BFGS restarts until accepted-iteration ceiling' if is_conventional else 'compact two-stage BFGS; early stopping is retained',
            gtol=config.gtol, campaign=CAMPAIGN))
    for method, run in [('IB',prescribed['IB_matched']),('GS',prescribed['GS_matched']),('AD',ad)]:
        solver = run.solver_result
        metadata = dict(solver.metadata)
        phases[method]=np.asarray(run.phase_rad)
        rows.append(dict(method=method, task='Triple',iteration_ceiling={'IB':200,'GS':100,'AD':150}[method],
            actual_iterations=solver.iterations,
            terminal_gradient_norm=float(np.linalg.norm(ad_gradient)) if method=='AD' else np.nan,
            gradient_definition='L2 norm of the analytic normalized pressure-amplitude loss gradient over all actuator phases' if method=='AD' else 'not applicable: native phase projection with no scalar minimization objective',
            terminal_projective_change_rad=solver.terminal_projective_change_rad,
            terminal_objective=metadata.get('final_loss',np.nan),
            termination_reason=('phase-change tolerance' if solver.converged else 'native fixed-step or capped synthesis'),
            final_solver_status='converged' if solver.converged else 'fixed-step/capped',
            final_solver_message='',nfev=np.nan,njev=np.nan,
            seed=metadata.get('effective_seed',metadata.get('seed',np.nan)),phase_content_id=digest_array(run.phase_rad),
            reported_command_time_s=run.timing.total_s,setup_s=run.timing.initialization_s,
            solve_s=run.timing.solve_s,
            timing_scope='native factory total: pressure-transfer/signature construction and solver; AD solver interval contains JAX transfer, compilation and explicit synchronization',
            stopping_rule={'IB':'projective phase change <=0.01 rad or 200 steps','GS':'100 fixed iterations','AD':'150 Adam steps'}[method],campaign=CAMPAIGN))
    cache_kinds={'Conventional':'triple_long_conventional_cumulative','FE':'triple_long_fixed_fe','GFE':'final_fig6_vertical_compensated_fe','IB':'final_triple_matched_baseline_commands','GS':'final_triple_matched_baseline_commands','AD':'triple_diff_pat_command'}
    for row in rows:
        accesses=[item for item in ctx.cache.access_records if item['kind']==cache_kinds[row['method']]]
        row['timing_source']='measured in this process' if any(item['status']=='new' for item in accesses) else 'retained command-cache measurement'
    table=pd.DataFrame(rows)
    table.to_csv(output/'main_triple30k_commands.csv', index=False)
    pd.DataFrame(stages).to_csv(output/'main_triple30k_solver_stages.csv',index=False)
    checkpoints=[]
    for row in conventional.checkpoint_records:
        row=dict(row);iteration=int(row['accepted_iteration']);phase=conventional.checkpoint_phases_rad[iteration]
        row.update(command_cosine_to_GFE=command_similarity(phase,gfe['phase']),
            command_cosine_to_30000=command_similarity(phase,conventional.terminal_phase_rad),
            cumulative_iteration_ceiling=30000,checkpoint_forces_restart=False)
        checkpoints.append(row);phases[f'Conventional_{iteration}']=phase
    pd.DataFrame(checkpoints).to_csv(output/'main_triple30k_conventional_checkpoints.csv',index=False)
    phases['initial_phase']=bank['initial_phase'];phases['targets_m']=node['targets_m']
    np.savez_compressed(output/'main_triple30k_commands_and_checkpoints.npz',**phases)
    prior_protocol_path=output/'main_triple30k_protocol.json'
    prior_protocol=json.loads(prior_protocol_path.read_text()) if prior_protocol_path.exists() else {}
    protocol=dict(campaign=CAMPAIGN,report_generated_utc=datetime.now(timezone.utc).isoformat(),
        command_measurement_session_started_utc=(ctx.memo.get('_main_triple30k_session_started_utc') if any(row['timing_source']=='measured in this process' for row in rows) else prior_protocol.get('command_measurement_session_started_utc')),
        original_command_report_generated_utc=prior_protocol.get('original_command_report_generated_utc',prior_protocol.get('report_generated_utc',datetime.now(timezone.utc).isoformat())),
        command_timing_source={row['method']:row['timing_source'] for row in rows},
        evidence_status='SMOKE: one fixed Triple target configuration, one paired optimization seed',
        main_optimization_cap=30000,appendix_C_frozen_reference_cap=10000,
        seed=bank['seed'],initial_phase_content_id=digest_array(bank['initial_phase']),
        conventional_restart_policy=RESTART_POLICY,
        checkpoints_are_observations_not_restarts=True,
        configs={'Conventional':node['conventional_config'].to_payload(),'FE':node['fe_config'].to_payload(),'GFE':gfe['config'].to_payload()},
        dependencies={**dependency_snapshot(),**{name:version(name) for name in ['jax','jaxlib','scikit-image','scikit-learn','threadpoolctl']}},
        threadpool_info=threadpool_info(),platform=platform.platform(),processor=platform.processor(),
        environment={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','HAT_EXACT_WORKERS']},
        prescription_timing={method:asdict(run.timing) for method,run in [('IB',prescribed['IB_matched']),('GS',prescribed['GS_matched']),('AD',ad)]},
        prescription_metadata={method:dict(run.solver_result.metadata) for method,run in [('IB',prescribed['IB_matched']),('GS',prescribed['GS_matched']),('AD',ad)]},
        native_timing_scope_note='All newly generated main Triple commands are measured in this one serial campaign. Native timer scopes differ and are disclosed per method; this is not a new unified end-to-end timer.',
        cache_access_records=list(ctx.cache.access_records))
    (output/'main_triple30k_protocol.json').write_text(json.dumps(protocol,indent=2,default=_plain))
    audit=[]
    for path in (ctx.cache.root/'triple_long_conventional_cumulative').glob('*.pkl'):
        with path.open('rb') as stream: record=pickle.load(stream)
        if record['payload'].get('cumulative_cap')!=10000: continue
        old=record['value']
        if old.problem_fingerprint!=conventional.problem_fingerprint or old.seed!=conventional.seed: continue
        checkpoint=next(item for item in checkpoints if item['accepted_iteration']==10000)
        audit.append(dict(method='Conventional',archived_cap=10000,current_cap=30000,
            archived_cache=str(path.relative_to(ctx.output_root)),
            archived_10000_solve_s=old.solve_sec,current_10000_checkpoint_s=checkpoint['elapsed_sec'],
            archived_phase_content_id=digest_array(old.terminal_phase_rad),
            current_10000_phase_content_id=checkpoint['phase_content_id'],
            exact_phase_match=digest_array(old.terminal_phase_rad)==checkpoint['phase_content_id'],
            archived_gradient_norm=old.terminal_gradient_norm,current_10000_gradient_norm=checkpoint['gradient_norm']))
    (output/'main_triple30k_budget_audit.json').write_text(json.dumps(audit,indent=2,default=_plain))
    if benchmark is not None:
        validation=benchmark['table'].copy();validation['method']=validation['method'].replace({'FE+g':'GFE'})
        validation.to_csv(output/'main_triple30k_independent_elastic_validation.csv',index=False)
        (output/'main_triple30k_measured_results.txt').write_text(
            'Main Triple matched 30,000-iteration campaign; SMOKE, one paired seed.\n'
            +table[['method','iteration_ceiling','actual_iterations','terminal_gradient_norm','reported_command_time_s','termination_reason']].to_string(index=False)
            +'\n\nConventional checkpoints (no forced restarts):\n'+pd.DataFrame(checkpoints).to_string(index=False)
            +'\n\nIndependent elastic displacement by method (three target roots; d/a):\n'
            +validation.groupby('method')['finite_ka_displacement_a'].agg(['min','median','max']).to_string()+'\n')
    for name,path in [('commands',output/'main_triple30k_commands.csv'),('checkpoints',output/'main_triple30k_conventional_checkpoints.csv')]:
        ctx.tables['main_triple30k_'+name]=pd.read_csv(path)
    return output


def run(root=ROOT):
    import run_study
    from hat_revision_pipeline.main_triple_endpoint import ensure_main_triple_endpoint
    from hat_revision_pipeline.main_feg import feg_triple_run
    from hat_revision_pipeline import final_figures as ff
    started_utc=datetime.now(timezone.utc).isoformat()
    ctx=run_study.prepare(root)
    ctx.memo['_main_triple30k_session_started_utc']=started_utc
    print('Starting matched main Triple Conventional and FE commands',flush=True)
    bank=ensure_main_triple_endpoint(ctx)
    print('Conventional and FE commands ready',flush=True)
    gfe=feg_triple_run(ctx)
    print('GFE command ready',flush=True)
    targets=bank['objects'][(1,1)]['targets_m']
    prescribed=ff._triple_prescribed_commands(ctx,targets)
    ad=ff._triple_ad_command(ctx,targets)
    export(ctx,bank,gfe,prescribed,ad)
    print('All six commands exported; starting independent elastic validation',flush=True)
    benchmark=ff._triple_exact_benchmark_with_concentration(ctx)
    output=export(ctx,bank,gfe,prescribed,ad,benchmark)
    print((output/'main_triple30k_measured_results.txt').read_text(),flush=True)
    print('Campaign complete: '+str(output),flush=True)
    return ctx

if __name__=='__main__': run()
