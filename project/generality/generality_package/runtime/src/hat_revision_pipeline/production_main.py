"""Explicit production sampling for the eight established GFE main figures.

No computation is performed on import. All synthesis uses the deposited solver
implementations and native stopping rules. Timed records belong to the current
machine campaign; archived phase-only references remain a separate cache source.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace, is_dataclass, fields
from pathlib import Path
import fingerprintlib
import json
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import pipeline as p, final_figures as ff, main_feg as mf
from . import main_feg_branch as branch_source, fig1_domain_volume as domain
from .cache import CacheStore, CacheMissError, digest_array, digest_payload, dependency_snapshot
from .config import RunConfig
from .gorkov_core import Condition, MethodSpec, SingleTargetObjective
from .sota import solve_corrected_gorkov_fe, benchmark_synthesis_method_factory
from .multitrap import run_multitrap, evaluate_multitrap_objective
from .triple_long_iteration import _configs, _run_cumulative_conventional, CHECKPOINTS, FIXED_BASE_TARGETS_M
from .fig4_threepoint_selection import make_problem
from .branch_figures import projective_similarity, target_demodulated_states, COMPONENTS

SINGLE30 = (172817509, 425727192, 469986874, 519693606, 681979972,
            693201818, 713854823, 731661603, 764705224, 991245467,
            *range(260828, 260848))
TRIPLE30 = tuple(range(260869, 260899))
ALPHAS = (1e3, 1e6, 1e7)
METHODS = ('Conventional', 'FE', 'GFE', 'IB', 'GS', 'AD')
SCHEMA = 'gfe-production-main-v1'
_ORIGINAL_FIXED_BRANCH = branch_source.fixed_feg_branch_pack
_ORIGINAL_SINGLE_BENCHMARK = ff._exact_benchmark_with_concentration
_ORIGINAL_TRIPLE_BENCHMARK = ff._triple_exact_benchmark_with_concentration


class ProductionCache(CacheStore):
    """Reuse exact immutable non-timing inputs without importing smoke times."""
    IMMUTABLE_KINDS = frozenset({'alpha10_feg_branch_medoid_continuation',
                                'decision_endpoint_run'})

    def __init__(self, root, study_root, *, read_only=False):
        source_root=Path(__file__).parent
        self.source_hashes={name:fingerprintlib.fingerprint((source_root/name).read_bytes()).hexdigest()
                            for name in ('gorkov_core.py','multitrap.py','sota.py','diff_pat.py','exact_validator.py','triple_long_iteration.py')}
        super().__init__(root, namespace=SCHEMA+'-'+digest_payload(self.source_hashes)[:16], read_only=read_only)
        self.immutable = CacheStore(Path(study_root) / 'smoke_outputs/cache',
                                    namespace=p.PIPELINE_IMPLEMENTATION, read_only=True)

    def get_or_compute(self, kind, payload, producer, *, recompute=False):
        def compute():
            if kind in self.IMMUTABLE_KINDS and not recompute:
                try:
                    value, _, source = self.immutable.get_or_compute(kind, payload, producer)
                    self.access_records.append(dict(kind=kind, status='immutable phase reference', path=str(source)))
                    return value
                except CacheMissError:
                    pass
            return producer()
        result = super().get_or_compute(kind, payload, compute, recompute=recompute)
        value = result[0]
        mapping = value if isinstance(value, dict) else {f.name:getattr(value,f.name) for f in fields(value)} if is_dataclass(value) else {}
        arrays = {str(key): item for key, item in mapping.items()
                  if isinstance(item, np.ndarray) and item.dtype.kind != 'O'}
        if isinstance(value, (tuple, list)):
            arrays.update({f'array_{index}':item for index,item in enumerate(value)
                           if isinstance(item,np.ndarray) and item.dtype.kind != 'O'})
        if arrays and not self.read_only:
            output = self.root.parent / 'numerical_arrays' / kind
            output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output / (result[2].stem+'.npz'), **arrays,
                                payload_json=json.dumps(payload,default=str))
        return result


def _machine_identity(settings):
    """Fingerprint the actual backend without running a synthesis command."""
    host = dict(dependency_snapshot(), machine=platform.machine(), processor=platform.processor(), threads=settings['blas_threads'])
    from importlib.metadata import version, PackageNotFoundError, distributions
    for package in ('jax','jaxlib'):
        try: host[package] = version(package)
        except PackageNotFoundError: host[package] = 'not installed'
    host['requested_device'] = settings['device']
    host['jax_platform_request'] = os.environ.get('JAX_PLATFORMS', '')
    # The requested "auto" mode is insufficient to identify a timing machine.
    # Query the initialized backend before the campaign/cache key is constructed.
    import jax
    host['jax_backend'] = str(jax.default_backend())
    host['jax_enable_x64'] = bool(jax.config.jax_enable_x64)
    host['jax_devices'] = [dict(platform=str(device.platform),
        device_kind=str(device.device_kind), id=int(device.id),
        process_index=int(device.process_index),
        platform_version=str(getattr(device.client, 'platform_version', 'unreported')))
        for device in jax.devices()]
    host['accelerator_package_versions'] = {
        str(dist.metadata['Name']): str(dist.version) for dist in distributions()
        if str(dist.metadata.get('Name', '')).lower().startswith(('jax-cuda', 'nvidia-'))}
    return host


def prepare(root, settings=None):
    """Create a full numerical context; optimization starts only in producers."""
    root = Path(root).resolve()
    from production_settings import load_settings
    settings = load_settings(root, settings)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = str(settings['blas_threads'])
    os.environ['HAT_FIG1_DOMAIN_WORKERS'] = '1'
    os.environ['HAT_HIDE_MODE_NOTE'] = '1'
    os.environ['HAT_PROTOTYPE_LABEL'] = '0'
    output = root / 'production_outputs'
    ctx = p.prepare_pipeline(run_mode='full', recompute=False, output_root=output)
    ctx.config = replace(ctx.config, multitrap_maxiter=30000)
    ctx.cache = ProductionCache(output / 'cache', root)
    host = _machine_identity(settings)
    campaign = settings['campaign_id'] + '-' + digest_payload(host)[:20]
    ctx.memo.update(production_root=root, production_settings=settings,
                    production_timing_campaign=str(campaign), production_machine=host,
                    production_single_seeds=tuple(SINGLE30), production_triple_seeds=tuple(TRIPLE30),
                    production_main=True, production_single_display_seed=260828)
    manifest = dict(schema=SCHEMA, status='configured; execution starts in figure cells',
        single_seeds=list(SINGLE30), triple_seeds=list(TRIPLE30), single_cap=10000,
        triple_cap=30000, triple_checkpoints=list(CHECKPOINTS),
        field_samples=181, local_plane_samples=101, figure1_xz_shape=[7,7],
        figure1_xz_restarts=3, nominal_command_count=244,
        nominal_target_validation_count=488, deterministic_timing_repeats=30,
        machine=host, timing_campaign=campaign, source_hashes=ctx.cache.source_hashes,
        exact_protocol=p._finite_ka_protocol(ctx),
        generality='ACTIVE: Appendix G1-G9', optional_experiments=False)
    (output / 'production_main_settings.json').write_text(json.dumps(manifest, indent=2, default=str))
    return ctx


def _timed_payload(ctx, payload):
    return dict(payload, timing_campaign=ctx.memo['production_timing_campaign'],
                machine=ctx.memo['production_machine'], schema=SCHEMA)


def _save_command(ctx, task, method, seed, value, *, suffix='nominal'):
    directory = ctx.output_root / 'commands' / task
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'{method}_seed{seed}_{suffix}'
    row = value['row']
    np.savez_compressed(directory / (stem+'.npz'), phase_rad=value['phase'],
        targets_m=value.get('targets', np.asarray([p.MAIN_TARGET_M])),
        metadata_json=json.dumps(row, default=str))
    (directory / (stem+'.json')).write_text(json.dumps(row, indent=2, default=str))


def ensure_single_run(ctx, method, seed, *, target=None, alpha=None, role='nominal'):
    """Return {run, phase, row}; FE/GFE/Conventional share the same cold phase."""
    if method not in ('Conventional','FE','GFE'):
        raise ValueError(method)
    target = np.asarray(p.MAIN_TARGET_M if target is None else target, float)
    if method == 'GFE':
        objective = mf.feg_single_objective(ctx, target)
    elif method == 'Conventional':
        objective = p._objective_at(ctx, 'Conventional', target)
    else:
        from .fixed_fe_endpoints import fixed_single_fe_method_spec
        spec = fixed_single_fe_method_spec()
        condition = Condition(label='fixed-single-fe-alpha10',positions=ctx.positions_m,
            target_m=tuple(target),frequency_hz=ctx.config.frequency_hz,
            curvature_weight=spec.curvature_weight,normals=ctx.normals)
        objective = SingleTargetObjective(condition,spec)
    if alpha is not None:
        spec = MethodSpec(name='Force-Equilibrium', alpha_per_m=float(alpha),
                          beta_curvature_per_pa=0., pressure_mode='abs', curvature_weight=np.eye(3))
        condition = Condition(label=f'feg-alpha-{alpha:g}', positions=ctx.positions_m,
            target_m=tuple(target), frequency_hz=ctx.config.frequency_hz,
            curvature_weight=np.eye(3), normals=ctx.normals)
        objective = SingleTargetObjective(condition, spec, force_target_n=objective.force_target_n)
    initial = np.random.default_rng(int(seed)).uniform(-np.pi,np.pi,len(ctx.positions_m))
    payload = _timed_payload(ctx, dict(method=method, objective=asdict(objective.method),
        target_m=target.tolist(), force_target_n=objective.force_target_n.tolist(),
        seed=int(seed), initial_phase_fingerprint=digest_array(initial),
        positions=digest_array(ctx.positions_m), normals=digest_array(ctx.normals),
        frequency_hz=ctx.config.frequency_hz, maxiter=10000, gtol=ctx.config.gtol))
    key = ('production_single', digest_payload(payload))
    if key in ctx.memo:
        return ctx.memo[key]
    run, status, path = ctx.cache.get_or_compute('production_single_run', payload,
        lambda: solve_corrected_gorkov_fe(objective, initial, maxiter=10000, gtol=ctx.config.gtol))
    row = dict(method=method, task='Single', seed=int(seed), alpha_per_m=float(objective.method.alpha_per_m),
        iteration_cap=10000, maxiter=10000, cap=10000, iterations=int(run.iterations),
        evaluations=int(run.evaluations), objective=float(run.objective),
        gradient_norm=float(run.gradient_norm), terminal_gradient_norm=float(run.gradient_norm),
        terminal_gradient_l2=float(run.gradient_norm), terminal_gradient_rms=float(run.gradient_norm/np.sqrt(len(initial)-1)),
        optimizer_success=bool(run.success), optimizer_status=int(run.status), optimizer_message=str(run.message),
        native_success=bool(run.success), native_status=int(run.status), native_message=str(run.message),
        wall_time_s=float(run.history_wall_s[-1]), command_time_s=float(run.history_wall_s[-1]),
        phase_fingerprint=digest_array(run.phase_rad), initial_phase_fingerprint=digest_array(initial),
        actual_force_target_z_n=float(objective.force_target_n[2]), force_target_z_n=float(objective.force_target_n[2]),
        cache_status=status, cache_path=str(path), cache_source=str(path),
        timing_campaign=ctx.memo['production_timing_campaign'],
        timing_scope='native BFGS solver history wall time; objective/transfer construction excluded',
        role=role, correct_sign_gorkov=True, run_mode='full', evidence_status='PRODUCTION', manuscript_numerics=True,
        near_stationary=bool(run.gradient_norm <= ff.REPORT_GRADIENT_NORM),
        **{f'target_{axis}_m':float(target[i]) for i,axis in enumerate('xyz')})
    value=dict(run=run,phase=np.asarray(run.phase_rad),row=row,objective=objective,
               initial_phase=initial,targets=np.asarray([target]))
    ctx.memo[key]=value
    _save_command(ctx,'Single',method,seed,value,suffix=f'{role}_{digest_array(target)[:10]}_a{objective.method.alpha_per_m:g}')
    return value


def triple_problem(ctx):
    if 'production_triple_problem' not in ctx.memo:
        ctx.memo['production_triple_problem'] = make_problem(ctx,FIXED_BASE_TARGETS_M,0.,0.,case_prefix='fixed-long-triple-observation')
    return ctx.memo['production_triple_problem']


def ensure_triple_run(ctx, method, seed):
    """Return {run, phase, row, config, problem, targets}; native restart policy."""
    if method not in ('Conventional','FE','GFE'):
        raise ValueError(method)
    key=('production_triple',method,int(seed))
    if key in ctx.memo:return ctx.memo[key]
    problem=triple_problem(ctx)
    conventional,fe=_configs(ctx,30000)
    config=conventional if method=='Conventional' else replace(fe,compensate_effective_gravity=method=='GFE')
    initial=np.random.default_rng(int(seed)).uniform(-np.pi,np.pi,len(ctx.positions_m))
    payload=_timed_payload(ctx,dict(method=method,seed=int(seed),problem=problem.fingerprint,
        config=config.to_payload(),initial_phase_fingerprint=digest_array(initial),maxiter=30000,
        checkpoints=list(CHECKPOINTS) if method=='Conventional' else []))
    def compute():
        if method=='Conventional':
            return _run_cumulative_conventional(problem,config,seed=int(seed),initial_phases=initial,
                                                cumulative_cap=30000,checkpoints=CHECKPOINTS,record_stalled_terminal=True)
        return run_multitrap(problem,config,seed=int(seed),initial_phases=initial,record_history=True,
                             metadata=dict(display_method=method,evidence_status='PRODUCTION'))
    run,status,path=ctx.cache.get_or_compute('production_triple_run',payload,compute)
    if method=='Conventional':
        phase=np.asarray(run.terminal_phase_rad);iterations=run.accepted_iterations;evals=run.nfev
        grad=run.terminal_gradient_norm;loss=run.terminal_objective;elapsed=run.solve_sec
        last=run.segment_records[-1];success=bool(last.get('success',False)) and not run.termination_reason.startswith('zero-accepted');code=last.get('status',1)
        message=run.termination_reason;scope='native cumulative Conventional BFGS solve time'
    else:
        phase=np.asarray(run.phases);iterations=run.iterations;evals=run.nfev;last=run.stages[-1]
        evaluation=evaluate_multitrap_objective(problem,phase,config,force_epsilon=last.force_epsilon,
                                               uniformity_epsilon=last.uniformity_epsilon)
        grad=float(np.linalg.norm(evaluation.gradient_reduced));loss=float(evaluation.value)
        elapsed=run.end_to_end_sec;success=last.success;code=last.status;message=last.message
        scope='native run_multitrap end-to-end time; external problem construction excluded'
    row=dict(method=method,task='Triple',seed=int(seed),iterations=int(iterations),evaluations=int(evals),
        gradient_norm=float(grad),terminal_gradient_norm=float(grad),terminal_gradient_l2=float(grad),
        objective=float(loss),wall_time_s=float(elapsed),command_time_s=float(elapsed),
        iteration_cap=30000,maxiter=30000,cap=30000,optimizer_success=bool(success),optimizer_status=int(code),
        optimizer_message=str(message),native_success=bool(success),native_status=int(code),native_message=str(message),
        phase_fingerprint=digest_array(phase),initial_phase_fingerprint=digest_array(initial),cache_status=status,
        cache_path=str(path),cache_source=str(path),timing_campaign=ctx.memo['production_timing_campaign'],
        timing_scope=scope,config_json=json.dumps(config.to_payload()),
        native_stage_records_json=json.dumps(list(run.segment_records) if method=='Conventional' else [asdict(stage) for stage in run.stages],default=str),
        evidence_status='PRODUCTION',manuscript_numerics=True)
    value=dict(run=run,phase=phase,row=row,config=config,problem=problem,
               initial_phase=initial,targets=np.array([s.target_m for s in problem.stencils]))
    ctx.memo[key]=value;_save_command(ctx,'Triple',method,seed,value)
    return value


def ensure_alpha_bank(ctx, seed):
    """The three cold commands shared exactly with production Appendix C3."""
    values=[ensure_single_run(ctx,'GFE',int(seed),alpha=alpha,role='cold_alpha') for alpha in ALPHAS]
    table=pd.DataFrame([dict(v['row'],prescribed_radiation_force_z_n=v['row']['force_target_z_n']) for v in values])
    return dict(runs=[v['run'] for v in values],phases=np.stack([v['phase'] for v in values]),table=table,seed=int(seed))


def _holography_run(ctx, task, method, seed=None, repeat=0):
    target=p.MAIN_TARGET_M if task=='Single' else np.mean(FIXED_BASE_TARGETS_M,axis=0)
    spec=p.SINGLE_SIDED_GS_VORTEX_SPEC if task=='Single' and method=='GS' else p.IB_VORTEX_SPEC
    factory=(p._vortex_transfer_factory(ctx,target,spec=spec) if task=='Single'
             else p._multivortex_transfer_factory(ctx,FIXED_BASE_TARGETS_M,spec=spec))
    if method=='AD':fn=ff.diff_pat_phase_only;kwargs=dict(task=task,random_seed=int(seed))
    elif method=='IB':fn=ff.iterative_back_projection;kwargs=dict(maxiter=200,tolerance_rad=.01)
    elif method=='GS':fn=ff.phase_only_signature_projection_audit;kwargs=dict(iterations=100)
    else:raise ValueError(method)
    if task=='Triple' and method!='AD':kwargs['control_group_size']=8
    payload=_timed_payload(ctx,dict(task=task,method=method,seed=seed,repeat=int(repeat),
        settings=kwargs,control_spec=asdict(spec),positions=digest_array(ctx.positions_m),
        normals=digest_array(ctx.normals),frequency_hz=ctx.config.frequency_hz,
        targets_m=np.asarray([target] if task=='Single' else FIXED_BASE_TARGETS_M).tolist(),
        diff_pat=asdict(ff.DIFF_PAT_CONFIG) if method=='AD' else None))
    run,status,path=ctx.cache.get_or_compute('production_holography_run',payload,
        lambda:benchmark_synthesis_method_factory(fn,factory,target_m=target,method_kwargs=kwargs))
    solver=run.solver_result
    row=dict(task=task,method=method,seed=seed,repeat=int(repeat),iterations=int(solver.iterations),
        command_time_s=float(run.timing.total_s),phase_fingerprint=digest_array(run.phase_rad),
        cache_path=str(path),cache_status=status,timing_campaign=ctx.memo['production_timing_campaign'],
        timing_scope='BenchmarkRun.timing.total_s including transfer construction and native compilation/synchronization',
        timing_json=json.dumps(asdict(run.timing),default=str),
        solver_metadata_json=json.dumps(dict(solver.metadata),default=str),
        native_success=solver.converged,terminal_projective_change_rad=solver.terminal_projective_change_rad,
        evidence_status='PRODUCTION')
    return dict(run=run,phase=np.asarray(run.phase_rad),row=row,
                targets=np.asarray([p.MAIN_TARGET_M] if task=='Single' else FIXED_BASE_TARGETS_M))


def ensure_command_bank(ctx):
    """Synthesize serially: 244 command slots; deterministic timings are repeats."""
    if 'production_commands' in ctx.memo:return ctx.memo['production_commands']
    bank={};timing_rows=[]
    for task,seeds in [('Single',SINGLE30),('Triple',TRIPLE30)]:
        for seed in seeds:
            for method in ('Conventional','FE','GFE','AD'):
                value=(_holography_run(ctx,task,method,seed) if method=='AD' else
                       ensure_single_run(ctx,method,seed) if task=='Single' else ensure_triple_run(ctx,method,seed))
                bank[(task,method,int(seed))]=value
                timing_rows.append(dict(value['row'],repeat=0))
                _save_command(ctx,task,method,seed,value)
        for method in ('IB','GS'):
            first=None
            for repeat in range(30):
                value=_holography_run(ctx,task,method,repeat=repeat)
                if first is None:first=value
                elif not np.allclose(np.exp(1j*value['phase']),np.exp(1j*first['phase']),atol=1e-10,rtol=0.):
                    raise RuntimeError(f'{task} {method} repeat changed deterministic phase')
                timing_rows.append(value['row'])
            first['row']=dict(first['row'],command_time_s=float(np.median([r['command_time_s'] for r in timing_rows if r['task']==task and r['method']==method])))
            bank[(task,method,None)]=first;_save_command(ctx,task,method,'deterministic',first)
    if len(bank)!=244:raise AssertionError(f'Expected244 command slots, got{len(bank)}')
    ctx.memo['production_commands']=bank
    ctx.tables['production_command_runs']=pd.DataFrame([v['row'] for v in bank.values()])
    ctx.tables['production_timing_observations']=pd.DataFrame(timing_rows)
    p.export_all_tables(ctx)
    return bank


def pressure_descriptor(ctx, phase):
    """The established 80-radius × 256-azimuth morphology diagnostic."""
    from .appendix_b3_mechanics import transverse_descriptor
    payload=dict(phase=digest_array(phase),positions=digest_array(ctx.positions_m),
        normals=digest_array(ctx.normals),frequency_hz=ctx.config.frequency_hz,
        target_m=p.MAIN_TARGET_M.tolist(),radial_points=80,azimuth_points=256,
        descriptor_source=fingerprintlib.fingerprint(Path(__file__).with_name('appendix_b3_mechanics.py').read_bytes()).hexdigest())
    value,_,_=ctx.cache.get_or_compute('production_pressure_descriptor',payload,
        lambda:transverse_descriptor(p._array_field(ctx,phase),p.MAIN_TARGET_M,343./ctx.config.frequency_hz))
    return dict(value[0])


def _single_plot_bank(ctx):
    runs={};rows=[]
    for index,seed in enumerate(SINGLE30):
        for method in ('Conventional','GFE'):
            value=ensure_single_run(ctx,method,seed)
            label='FE' if method=='GFE' else method
            runs[(label,index)]=value['run'];rows.append(dict(value['row'],method=label,pair_index=index))
    table=pd.DataFrame(rows)
    ctx.tables['main_feg_single_runs']=table.assign(method=table.method.replace({'FE':'GFE'}))
    ctx.tables['fig1_single_vortex_iterations_for_table_only']=table[['method','pair_index','seed','iterations','evaluations','iteration_cap']]
    return dict(runs=runs,table=table,seeds=list(SINGLE30),caps={'Conventional':10000,'FE':10000})


def _domain_plot_bank(ctx):
    key='production_xz_bank'
    if key in ctx.memo:return ctx.memo[key]
    rows=[]
    for z in np.linspace(*domain.Z_BOUNDS_M,7):
        for x in np.linspace(*domain.X_BOUNDS_M,7):
            target=np.array([x,0.,z]);target_id=domain._coordinate_id(target)
            for restart in range(3):
                seed=domain._matched_seed(ctx.config.random_seed+401,target_id,restart)
                for method in ('Conventional','FE','GFE'):
                    value=ensure_single_run(ctx,method,seed,target=target,role='xz_domain')
                    rows.append(dict(value['row'],target_id=target_id,restart=restart))
    raw=pd.DataFrame(rows)
    ctx.tables['main_fe_control_xz_domain']=raw[raw.method.eq('FE')].copy()
    raw=raw[~raw.method.eq('FE')].copy()
    summary=raw.groupby(['method','alpha_per_m','target_id','target_x_m','target_y_m','target_z_m'],sort=True).agg(
        run_count=('restart','size'),wall_time_median_s=('wall_time_s','median'),
        wall_time_q25_s=('wall_time_s',lambda a:a.quantile(.25)),wall_time_q75_s=('wall_time_s',lambda a:a.quantile(.75)),
        terminal_gradient_rms_median=('terminal_gradient_rms','median')).reset_index()
    if len(raw)!=294:raise AssertionError('XZ production map must have294 displayed-method runs')
    ctx.tables['fig1_xz_domain_runs']=raw;ctx.tables['fig1_xz_domain_summary']=summary
    definition=pd.DataFrame([dict(x_points=7,z_points=7,restarts=3,target_count=49,methods='Conventional;GFE',
        x_min_m=domain.X_BOUNDS_M[0],x_max_m=domain.X_BOUNDS_M[1],z_min_m=domain.Z_BOUNDS_M[0],z_max_m=domain.Z_BOUNDS_M[1])])
    ctx.tables['fig1_xz_domain_definition']=definition
    value=dict(raw=raw,summary=summary.assign(method=summary.method.replace({'GFE':'FE'})),definition=definition)
    ctx.memo[key]=value;return value


def _branch_pack(ctx):
    key='production_fixed_branch'
    if key in ctx.memo:return ctx.memo[key]
    base=_ORIGINAL_FIXED_BRANCH(ctx)
    if len(base['anchors'])!=98 or base['anchors'].target_id.nunique()!=49:
        raise AssertionError('Production must preserve the98-anchor/49-target GFE frame')
    bank=base['bank'];ix,iy=bank.root_target
    root_table=base['anchors'].query('ix == @ix and iy == @iy')
    anchors=np.stack([bank.target_medoid(ix,iy,component) for component in COMPONENTS])
    phases=np.stack([ensure_single_run(ctx,'Conventional',seed)['phase'] for seed in SINGLE30])
    rows=pd.DataFrame([dict(target_x_m=p.MAIN_TARGET_M[0],target_y_m=p.MAIN_TARGET_M[1],target_z_m=p.MAIN_TARGET_M[2]) for _ in SINGLE30])
    states=target_demodulated_states(phases,rows,positions_m=ctx.positions_m,frequency_hz=ctx.config.frequency_hz)
    similarity=projective_similarity(states,anchors);records=[]
    for index,seed in enumerate(SINGLE30):
        component_index=int(np.argmax(similarity[index]));s=float(similarity[index,component_index])
        records.append(dict(trajectory_index=index,seed=seed,target_id=str(root_table.iloc[0].target_id),
            component=COMPONENTS[component_index],objective=ensure_single_run(ctx,'Conventional',seed)['row']['objective'],
            similarity_to_matched_fe=s,distance_to_matched_fe=float(np.sqrt(max(0.,1-s*s))),
            reference_method='GFE',similarity_to_matched_feg=s,distance_to_matched_feg=float(np.sqrt(max(0.,1-s*s)))))
    value=dict(base,correlation=pd.DataFrame(records));ctx.memo[key]=value
    ctx.tables['production_fig2_central_correspondence']=value['correlation'].copy()
    return value


def _representative_triple_bank(ctx, compensated=False):
    c=ensure_triple_run(ctx,'Conventional',260869);fe=ensure_triple_run(ctx,'GFE' if compensated else 'FE',260869)
    node=dict(problem=c['problem'],conventional_config=c['config'],fe_config=fe['config'],
        Conventional=c['run'],FE=fe['run'],FE_phase=fe['phase'],
        Conventional_checkpoint_phases=c['run'].checkpoint_phases_rad,targets_m=c['targets'])
    return dict(objects={(1,1):node},initial_phase=c['initial_phase'],seed=260869,
        conventional_cap=30000,fe_cap=30000,checkpoints=CHECKPOINTS,nodes=((1,1),))


def _representative_gfe_triple(ctx):
    value=ensure_triple_run(ctx,'GFE',260869)
    return dict(value,bank=_representative_triple_bank(ctx),cache_status=value['row']['cache_status'])


def _spec(task, method, seed, value):
    label='FE+g' if method=='GFE' else method
    return dict(method=label,display_method=label,task=task,seed=seed,pair_id=f'{task}-seed-{seed}',
        phase=value['phase'],target_m=value['targets'][0],source_index=-1,
        phase_source='current production command with fixed declared objective',
        terminal_iteration=int(value['row']['iterations']),command_time_s=float(value['row']['command_time_s']),
        timing_scope=value['row']['timing_scope'],command_record=value,
        method_family='prescribed phase-only' if method in ('IB','GS') else 'automatic-differentiation holography' if method=='AD' else 'optimization',
        preset_role='matched' if method in ('IB','GS') else 'matched-amplitude' if method=='AD' else 'not_applicable',
        preset_radius_lambda=.45 if method in ('IB','GS','AD') else np.nan)


def _validate_spec(ctx, spec, target_index, target):
    phase=spec['phase'];label=spec['method'];key=f"production-{spec['task']}-{label}-{spec['seed']}-T{target_index+1}"
    result=p._finite_ka_validation(ctx,phase,target,key=key)
    row=p._static_validation_row(label,result)
    viscous=p._viscous_elastic_validation(ctx,phase,target,key=key+'-viscous')
    row.update(p._viscous_static_validation_row(label,viscous))
    field=p._array_field(ctx,phase);resolved=bool(result.equilibrium.numerical_root_found)
    equilibrium=np.asarray(result.equilibrium.equilibrium_m)
    row.update(pressure_abs_at_intended_target_pa=float(abs(field.pressure(np.asarray(target)[None,:])[0])),
        pressure_abs_at_exact_equilibrium_pa=float(abs(field.pressure(equilibrium[None,:])[0])) if resolved else np.nan,
        task=spec['task'],method=label,display_method=label,seed=spec['seed'],pair_id=spec['pair_id'],
        target_index=target_index,target_id='Single' if spec['task']=='Single' else f'T{target_index+1}',
        phase_fingerprint=digest_array(phase),phase_source=spec['phase_source'],terminal_iteration=spec['terminal_iteration'],
        command_time_s=spec['command_time_s'],timing_scope=spec['timing_scope'],
        method_family=spec['method_family'],preset_role=spec['preset_role'],preset_radius_lambda=spec['preset_radius_lambda'],
        **{f'target_{axis}_m':float(target[i]) for i,axis in enumerate('xyz')},
        evidence_status='PRODUCTION',manuscript_numerics=True)
    row.update(ff._field_concentration(ctx,phase,target))
    directory=ctx.output_root/'mechanics';directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/(key+'.npz'),phase_rad=phase,target_m=target,equilibrium_m=equilibrium,
        displacement_m=result.equilibrium.displacement_m,force_jacobian_n_m=result.force_jacobian_n_m,
        stiffness_eigenvalues_n_m=result.symmetric_stiffness_eigenvalues_n_m,
        metadata_json=json.dumps(row,default=str),protocol_json=json.dumps(p._finite_ka_protocol(ctx),default=str))
    return dict(row=row,result=result,spec=dict(spec))


def ensure_benchmarks(ctx):
    """Validate every production command; all488 target records reach plots/tables."""
    if 'production_benchmarks' in ctx.memo:return ctx.memo['production_benchmarks']
    bank=ensure_command_bank(ctx);output={}
    for task in ('Single','Triple'):
        specs=[];evaluations=[]
        for (bank_task,method,seed),value in bank.items():
            if bank_task!=task:continue
            spec=_spec(task,method,seed,value);specs.append(spec)
            for index,target in enumerate(value['targets']):
                evaluations.append(_validate_spec(ctx,spec,index,target))
        table=pd.DataFrame([item['row'] for item in evaluations])
        value=dict(raw=table,table=table,all_table=table,concentration_table=table,
            specs=specs,method_specs=tuple(specs),results=[item['result'] for item in evaluations],
            evaluations=evaluations,targets_m=np.asarray([p.MAIN_TARGET_M] if task=='Single' else FIXED_BASE_TARGETS_M))
        if task=='Triple':value.update(node=_representative_triple_bank(ctx)['objects'][(1,1)],long_bank=_representative_triple_bank(ctx))
        output[task]=value;ctx.tables[f'production_{task.lower()}_target_mechanics']=table
    if sum(len(v['table']) for v in output.values())!=488:raise AssertionError('Expected488 target-level records')
    ctx.memo['production_benchmarks']=output
    ctx.tables['fig5_exact_force_field_concentration']=pd.concat([v['table'] for v in output.values()],ignore_index=True)
    _population_summaries(ctx,ctx.tables['fig5_exact_force_field_concentration'])
    p.export_all_tables(ctx);return output


def _command_metrics(table):
    records=[]
    for (task,method,phase,seed),group in table.groupby(['task','method','phase_fingerprint','seed'],sort=False,dropna=False):
        all_roots=bool(group.finite_ka_root_found.all())
        resolved=group[group.finite_ka_root_found.astype(bool)]
        records.append(dict(task=task,method=method,phase_fingerprint=phase,seed=group.iloc[0].seed,
            command_time_s=float(group.command_time_s.iloc[0]),target_count=len(group),
            root_found=all_roots,all_restoring=bool(all_roots and group.finite_ka_locally_restoring.all()),
            field_concentration_fraction=float(group.field_concentration_fraction.median()),
            finite_ka_displacement_a=float(resolved.finite_ka_displacement_a.median()) if all_roots else np.nan,
            finite_ka_stiffness_min_n_m=float(resolved.finite_ka_stiffness_min_n_m.min()) if all_roots else np.nan,
            pressure_abs_at_exact_equilibrium_pa=float(resolved.pressure_abs_at_exact_equilibrium_pa.median()) if all_roots else np.nan))
    return pd.DataFrame(records)


def _population_summaries(ctx, table):
    commands=_command_metrics(table);summary=[]
    rng=np.random.default_rng(20260906)
    metrics=('field_concentration_fraction','finite_ka_displacement_a','finite_ka_stiffness_min_n_m','pressure_abs_at_exact_equilibrium_pa','command_time_s')
    for (task,method),group in commands.groupby(['task','method'],sort=False):
        timing_values=_timing_values(ctx,task,method)
        row=dict(task=task,method=method,endpoint_count=len(group),resolved_count=int(group.root_found.sum()),
                 timing_observation_count=len(timing_values),
                 root_found=bool(group.root_found.any()),aggregation='within-command target metric, then median/IQR across paired seeds')
        for metric in metrics:
            a=(timing_values if metric=='command_time_s' else group[metric].dropna()).to_numpy(float)
            row[metric]=float(np.median(a)) if len(a) else np.nan
            for q,name in ((.25,'q25'),(.75,'q75')):row[f'{metric}_{name}']=float(np.quantile(a,q)) if len(a) else np.nan
            boot=np.median(a[rng.integers(0,len(a),(10000,len(a)))],axis=1) if len(a) else np.array([np.nan])
            row[f'{metric}_ci_low'],row[f'{metric}_ci_high']=np.quantile(boot,[.025,.975])
        summary.append(row)
    ctx.tables['production_command_metrics']=commands
    ctx.tables['production_method_summary']=pd.DataFrame(summary)
    paired=[]
    for task in ('Single','Triple'):
        selected=commands[commands.task.eq(task)]
        reference=selected[selected.method.eq('FE+g')].set_index('seed')
        for method in ('Conventional','FE','AD'):
            other=selected[selected.method.eq(method)].set_index('seed')
            common=reference.index.intersection(other.index)
            for metric in metrics:
                delta=(reference.loc[common,metric]-other.loc[common,metric]).dropna().to_numpy(float)
                boot=np.median(delta[rng.integers(0,len(delta),(10000,len(delta)))],axis=1) if len(delta) else np.array([np.nan])
                low,high=np.quantile(boot,[.025,.975])
                paired.append(dict(task=task,comparison=f'GFE minus {method}',metric=metric,
                    planned_pairs=len(common),valid_pairs=len(delta),median_difference=float(np.median(delta)) if len(delta) else np.nan,
                    ci_low=float(low),ci_high=float(high),bootstrap_resamples=10000,bootstrap_seed=20260906))
    ctx.tables['production_paired_GFE_differences']=pd.DataFrame(paired)
    return ctx.tables['production_method_summary']


def _force_aggregates(table, *, task):
    commands=_command_metrics(table.assign(task=task));rows=[]
    for method in ff.FIG5_METHOD_ORDER:
        group=commands[commands.method.eq(method)];resolved=group[group.root_found]
        row=dict(task=task,method=method,display_method=method,endpoint_count=len(group),resolved_count=len(resolved),
                 root_found=bool(len(resolved)),aggregation='per-command targets then median/IQR across seeds')
        for metric in ('field_concentration_fraction','finite_ka_displacement_a','finite_ka_stiffness_min_n_m'):
            values=group[metric].dropna();row[metric]=float(values.median()) if len(values) else np.nan
            row[metric+'_q25']=float(values.quantile(.25)) if len(values) else np.nan
            row[metric+'_q75']=float(values.quantile(.75)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _timing_values(ctx, task, method):
    """Keep repeated timing observations separate from deterministic endpoints."""
    observations=ctx.tables.get('production_timing_observations')
    if observations is None:
        raise RuntimeError('Production timing observations must precede population summaries')
    public_method='GFE' if method=='FE+g' else method
    selected=observations[observations.task.eq(task)&observations.method.eq(public_method)]
    if selected.empty:
        raise RuntimeError(f'Missing production timing observations for {task} {method}')
    return selected.command_time_s.dropna()


def _tradeoff_table(ctx, benchmark, triple_benchmark, command_time):
    records=[]
    for method in ('Conventional','FE','FE+g','IB','GS','AD'):
        record=dict(method=method,aggregation='median and IQR across independent commands; Triple metrics first reduced within each command')
        for task,source in [('Single',benchmark['concentration_table']),('Triple',triple_benchmark['table'])]:
            frame=_command_metrics(source);group=frame[frame.method.eq(method)];prefix=task.lower()
            valid=group[group.root_found]
            timing_values=_timing_values(ctx,task,method)
            values={'time_s':timing_values,'displacement_a':valid.finite_ka_displacement_a,
                    'stiffness_min_n_m':valid.finite_ka_stiffness_min_n_m,
                    'pressure_abs_at_exact_equilibrium_pa':valid.pressure_abs_at_exact_equilibrium_pa}
            for name,v in values.items():
                record[f'{prefix}_{name}']=float(v.median()) if len(v) else np.nan
                record[f'{prefix}_{name}_q25']=float(v.quantile(.25)) if len(v) else np.nan
                record[f'{prefix}_{name}_q75']=float(v.quantile(.75)) if len(v) else np.nan
            record.update({f'{prefix}_resolved_count':len(valid),f'{prefix}_endpoint_count':len(group),
                           f'{prefix}_timing_observation_count':len(timing_values),
                           f'{prefix}_population_count':len(group),f'{prefix}_root_found':bool(len(valid))})
            record[f'{prefix}_displacement_min_a']=record[f'{prefix}_displacement_a_q25']
            record[f'{prefix}_displacement_max_a']=record[f'{prefix}_displacement_a_q75']
            record[f'{prefix}_pressure_min_pa']=record[f'{prefix}_pressure_abs_at_exact_equilibrium_pa_q25']
            record[f'{prefix}_pressure_max_pa']=record[f'{prefix}_pressure_abs_at_exact_equilibrium_pa_q75']
        records.append(record)
    ctx.tables['fig8_single_endpoint_population']=_command_metrics(benchmark['concentration_table'])
    return pd.DataFrame(records)


def _representative_triple_benchmark(ctx):
    full=ensure_benchmarks(ctx)['Triple'];items=[]
    for item in full['evaluations']:
        seed=item['spec']['seed'];method=item['spec']['method']
        if (method in ('IB','GS') or (seed==260869 and method!='AD')):
            items.append(item)
    # The fixed published AD illustration has seed260828, outside TRIPLE30.
    # Retain it separately; it is not a31st sample in the statistical bank.
    ad=_holography_run(ctx,'Triple','AD',260828)
    spec=_spec('Triple','AD',260828,ad)
    for index,target in enumerate(FIXED_BASE_TARGETS_M):items.append(_validate_spec(ctx,spec,index,target))
    table=pd.DataFrame([item['row'] for item in items])
    ctx.tables['production_fixed_triple_illustration_mechanics']=table
    return dict(full,table=table,all_table=table,evaluations=items,results=[i['result'] for i in items])


def _compensation(ctx, benchmark):
    value=ensure_triple_run(ctx,'GFE',260869)
    items=[item for item in benchmark['evaluations'] if item['spec']['method']=='FE+g' and item['spec']['seed']==260869]
    return dict(value,evaluations=items,cache_status=value['row']['cache_status'])


def _alpha_display_bank(ctx):
    rows=[]
    for seed in SINGLE30:
        value=ensure_alpha_bank(ctx,seed)
        for run,row in zip(value['runs'],value['table'].to_dict('records')):
            rows.append(dict(row,**{'morphology_'+k:v for k,v in pressure_descriptor(ctx,run.phase_rad).items()}))
    table=pd.DataFrame(rows)
    ctx.tables['fig7_feg_alpha_examples']=table
    ctx.tables['fig7_alpha_transition']=table
    descriptor=[c for c in table if c.startswith('morphology_') and pd.api.types.is_numeric_dtype(table[c])]
    summary=table.groupby('alpha_per_m')[descriptor].agg(['median','min','max']).reset_index()
    summary.columns=['_'.join(str(part) for part in key if str(part)) if isinstance(key,tuple) else key for key in summary.columns]
    ctx.tables['production_fig7_morphology_summary']=summary
    return ensure_alpha_bank(ctx,260828)


@contextmanager
def _hooks(ctx):
    """Scope adapters to serial figure composition; restore all legacy producers."""
    import main_feg_benchmarks as benchmarks
    from . import fig8_methods_only as tradeoff
    updates=[(mf,'_single_plot_adapter',_single_plot_bank),
             (mf,'_domain_plot_adapter',_domain_plot_bank),
             (mf,'fixed_feg_branch_pack',_branch_pack),
             (mf,'feg_triple_run',_representative_gfe_triple),
             (mf,'_triple_plot_adapter',lambda c:_representative_triple_bank(c,True)),
             (ff,'_exact_benchmark_with_concentration',lambda c:ensure_benchmarks(c)['Single']),
             (ff,'_triple_exact_benchmark_with_concentration',lambda c:ensure_benchmarks(c)['Triple']),
             (ff,'_aggregate_force_concentration',_force_aggregates),
             (ff,'_figure6_vertical_compensated_fe',_compensation),
             (benchmarks,'alpha_bank',_alpha_display_bank),
             (tradeoff,'methods_only_tradeoff_table',_tradeoff_table)]
    old=[(obj,name,getattr(obj,name)) for obj,name,_ in updates]
    try:
        for obj,name,value in updates:setattr(obj,name,value)
        yield benchmarks
    finally:
        for obj,name,value in reversed(old):setattr(obj,name,value)


def _expand_interval_limits(ax, *, xvalues=(), yvalues=()):
    """Retain existing framing while making every finite quartile visible."""
    for values,getter,setter in ((xvalues,ax.get_xlim,ax.set_xlim),(yvalues,ax.get_ylim,ax.set_ylim)):
        values=np.asarray(values,dtype=float)
        values=values[np.isfinite(values)]
        if not len(values):continue
        original=getter(); low,high=sorted(original)
        span=max(high-low,float(np.ptp(values)),np.finfo(float).eps)
        low=min(low,float(values.min())-.035*span)
        high=max(high,float(values.max())+.035*span)
        setter((low,high) if original[0]<=original[1] else (high,low))


def _add_population_intervals(ctx, figure, number):
    if number==5:
        tables=ctx.tables['fig5_exact_force_field_concentration_aggregates']
        for ax,task in zip(figure.axes,('Single','Triple')):
            xvalues=[];yvalues=[]
            for row in tables[tables.task.eq(task)].itertuples(index=False):
                x=float(row.field_concentration_fraction);y=1e3*float(row.finite_ka_stiffness_min_n_m)
                if not np.isfinite(y):continue
                xerr=np.array([[max(0.,x-row.field_concentration_fraction_q25)],[max(0.,row.field_concentration_fraction_q75-x)]])
                yerr=1e3*np.array([[max(0.,row.finite_ka_stiffness_min_n_m-row.finite_ka_stiffness_min_n_m_q25)],
                                 [max(0.,row.finite_ka_stiffness_min_n_m_q75-row.finite_ka_stiffness_min_n_m)]])
                ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='none',ecolor=ff.COLORS[row.method],capsize=3,elinewidth=1.5,zorder=4)
                xvalues.extend((row.field_concentration_fraction_q25,row.field_concentration_fraction_q75))
                yvalues.extend((1e3*row.finite_ka_stiffness_min_n_m_q25,1e3*row.finite_ka_stiffness_min_n_m_q75))
            _expand_interval_limits(ax,xvalues=xvalues,yvalues=yvalues)
    elif number==8:
        from . import fig8_methods_only as tradeoff
        table=ctx.tables['fig8_performance_time_tradeoff']
        pooled=[]
        for task in ('Single','Triple'):
            for row in table.itertuples(index=False):pooled.append(dict(method=f'{task}::{row.method}',pooled_time_s=getattr(row,task.lower()+'_time_s')))
        xmap,_,_,_=tradeoff._broken_linear_time_coordinates(pd.DataFrame(pooled),time_column='pooled_time_s')
        for ax,column in zip(figure.axes[:2],('single_displacement_a','single_pressure_abs_at_exact_equilibrium_pa')):
            yvalues=[]
            for row in table.to_dict('records'):
                y=row[column]
                if not np.isfinite(y):continue
                ax.errorbar(xmap['Single::'+row['method']],y,
                    yerr=np.array([[max(0.,y-row[column+'_q25'])],[max(0.,row[column+'_q75']-y)]]),
                    fmt='none',ecolor=ff.COLORS[row['method']],capsize=3,elinewidth=1.5,zorder=4)
                yvalues.extend((row[column+'_q25'],row[column+'_q75']))
            _expand_interval_limits(ax,yvalues=yvalues)
        # Expanded results need not retain the smoke FE/GFE overlap annotation.
        for axis in figure.axes:
            for text in list(axis.texts):
                if text.get_text() in ('FE, FE+g','FE, GFE'):text.remove()


def main_figure(ctx, number):
    """Render an established main figure with its complete production population."""
    number=int(number)
    if number not in range(1,9):raise ValueError(number)
    if number == 5:
        raise ValueError(
            "The obsolete concentration-versus-stiffness graphic is disabled; "
            "its numerical benchmark records remain available."
        )
    with _hooks(ctx) as benchmarks:
        if number<=4:figure=mf.MAIN_FEG_FIRST_FOUR[number-1](ctx)
        elif number==6:
            original=ff._triple_exact_benchmark_with_concentration
            ff._triple_exact_benchmark_with_concentration=_representative_triple_benchmark
            try:figure=benchmarks.figure_6(ctx)
            finally:ff._triple_exact_benchmark_with_concentration=original
        else:figure={7:benchmarks.figure_7,8:benchmarks.figure_8}[number](ctx)
        _add_population_intervals(ctx,figure,number)
        stem=list(ctx.figures)[-1]
        # Reserialize after adding IQRs so notebook and standalone files agree.
        from .publication_layout import polish_main_figure
        polish_main_figure(figure)
        for extension in ('png','pdf','svg'):
            figure.savefig(ctx.output_root/'figures'/f'{stem}.{extension}',dpi=240,bbox_inches='tight',facecolor='white')
        if number == 6:
            benchmarks.finalize_figure_6_visual(ctx)
    p.export_all_tables(ctx)
    (ctx.output_root/'main_cache_access.json').write_text(json.dumps(ctx.cache.access_records,indent=2))
    path=ctx.output_root/'figures'/f'{stem}.png';plt.close(figure);return path


def run(root, ctx=None):
    """Run the current figures only; root orchestration controls appendices/export."""
    ctx=prepare(root) if ctx is None else ctx
    figures=[str(main_figure(ctx,number)) for number in (1,2,3,4,6,7,8)]
    return dict(figures=figures,output_dir=str(ctx.output_root),context=ctx)
