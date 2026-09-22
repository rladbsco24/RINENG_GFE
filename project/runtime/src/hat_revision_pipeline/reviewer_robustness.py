"""Current-objective optimizer, initialization and hyperparameter appendix.

Importing this module performs no experiment. The established main objectives
and figures are not edited. Run each section from the notebook's separate cells.
Every comparison retains native stops, a complete scalar trace and frozen phases.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from rineng_content_id import content_identity
import inspect
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits, threadpool_info
from . import pipeline as p, multitrap as mt
from .gorkov_core import Condition, SingleTargetObjective, reduce_gauge, gauge_full, method_spec
from .fixed_fe_endpoints import fixed_single_fe_method_spec, fixed_triple_fe_config
from .fig4_threepoint_selection import make_problem, contracted_targets
from .cache import CacheStore, digest_array, digest_file, canonical_json

SCHEMA = 'reviewer-robustness-20260908-v1'
SEED_BOOTSTRAP = 20260908
N_BOOTSTRAP = 10000
OPTIMIZERS = ('BFGS', 'L-BFGS-B', 'Adam', 'AdamW')
METHODS = ('Conventional', 'FE', 'GFE')
SOLVER = dict(optimizer='BFGS', gtol=1e-8, c1=1e-4, c2=.9,
              maxcor=10, maxls=40, ftol=0., maxfun=1000000,
              learning_rate=.03, beta1=.9, beta2=.999, adam_epsilon=1e-12,
              learning_rate_end_ratio=1e-4, schedule='cosine', weight_decay=0.)
ALPHAS_LARGE = (10., 100., 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
JUMP_RAD = (.05, .15, .5, 1., 2., np.pi)
FACTOR7 = (.1, .3, .5, 1., 2., 3., 10.)
FACTOR9 = (0., .01, .03, .1, .3, 1., 3., 10., 30.)

# Values are declared before results are inspected. These are OAT families;
# the two small crossed experiments below test the specifically coupled terms.
GRIDS = [
    ('alpha_factor', (.01,.03,.1,.3,1.,3.,10.,30.,100.), ('Single','Triple'), ('GFE','FE'), '1'),
    ('lambda_pressure_factor', FACTOR9, ('Single','Triple'), ('Conventional',), '1'),
    ('beta_factor', FACTOR9, ('Triple',), ('GFE',), '1'),
    ('gamma_uniformity', (0.,.03,.1,.3,1.,3.,10.), ('Triple',), ('GFE',), '1'),
    ('epsilon_pressure_rel', (0.,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2), ('Triple',), ('GFE',), '1'),
    ('epsilon_force_factor', FACTOR9, ('Triple',), ('GFE',), '1'),
    ('epsilon_std_factor', FACTOR9, ('Triple',), ('GFE',), '1'),
    ('axial_ratio_factor', FACTOR7, ('Single','Triple'), ('GFE',), '1; fixed trace'),
    ('transverse_ratio_factor', FACTOR7, ('Single','Triple'), ('GFE',), '1; fixed trace'),
    ('stencil_factor', (.25,.5,.75,1.,1.5,2.), ('Single','Triple'), ('GFE',), '1; h0=0.5 mm'),
    ('first_stage_iterations', (1,50,100,250,1000,3000), ('Triple',), ('GFE',), 'accepted iterations'),
    ('first_stage_factor', (.01,.02,.05,.1,.2,.5), ('Triple',), ('GFE',), 'initial scale multiplier'),
    ('gtol', (1e-6,1e-7,1e-8,1e-9,1e-10), ('Single','Triple'), ('GFE',), 'N/m/rad'),
]
SOLVER_GRIDS = [
    ('learning_rate', (.003,.01,.03,.1,.3), 'Adam'),
    ('beta1', (0.,.5,.8,.9,.95), 'Adam'),
    ('beta2', (.9,.99,.999,.9999,.99999), 'Adam'),
    ('adam_epsilon', (1e-16,1e-14,1e-12,1e-10,1e-8), 'Adam'),
    ('learning_rate_end_ratio', (1e-5,1e-4,1e-3,.01,1.), 'Adam'),
    ('weight_decay', (0.,1e-6,1e-4,.01,.1), 'AdamW'),
    ('maxcor', (3,5,10,20,40), 'L-BFGS-B'),
    ('maxls', (10,20,40,80,160), 'L-BFGS-B'),
    ('ftol', (0.,1e-15,1e-12,1e-9,1e-6), 'L-BFGS-B'),
    ('c1', (1e-6,1e-5,1e-4,1e-3,1e-2), 'BFGS'),
    ('c2', (.1,.3,.5,.7,.9), 'BFGS'),
]


def _json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
    temporary.replace(path)


def context(root, shared=None, mode='smoke'):
    from .config import RunConfig, square_positions, square_normals
    root=Path(root).resolve()
    if shared is not None: mode='full' if shared.config.full else 'smoke'
    base=root/('production_outputs/appendices/C_GFE' if mode=='full' else 'appendix_outputs/C_GFE')/'reviewer_robustness'
    base.mkdir(parents=True,exist_ok=True)
    cfg=shared.config if shared is not None else RunConfig.for_mode('full' if mode=='full' else 'quick')
    ctx=SimpleNamespace(config=cfg,positions_m=shared.positions_m if shared is not None else square_positions(),
        normals=shared.normals if shared is not None else square_normals(), data_root=root/'runtime/data',
        output_root=base,recompute=False,memo={},tables={},cache=CacheStore(base/'mechanical_cache',namespace=p.PIPELINE_IMPLEMENTATION,read_only=cfg.cache_only),
        root=root,mode=mode)
    # Commands have identical solver budgets in both modes and are shared. Only
    # their independent mechanical validation uses the selected numerical mode.
    ctx.command_cache=root/'reviewer_robustness_cache';ctx.command_cache.mkdir(exist_ok=True)
    ctx.source_hashes={name:digest_file(Path(__file__).with_name(name)) for name in
        ('gorkov_core.py','multitrap.py','fixed_fe_endpoints.py')}
    ctx.source_hashes['reviewer_solver']=content_identity('\n'.join(inspect.getsource(f) for f in
        (_weight,_build,_fg,_optimize,solve)).encode()).hexdigest()
    ctx.environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),blas=threadpool_info(),threads=1,dtype='float64/complex128')
    _json(base/'environment.json',ctx.environment)
    _json(base/'protocol.json',dict(schema=SCHEMA,mode=mode,solver_defaults=SOLVER,
        optimizer_seeds=seeds(mode,'optimizer'),sensitivity_seeds=seeds(mode,'sensitivity'),
        large_alpha_seeds=seeds(mode,'large_alpha'),bootstrap_seed=SEED_BOOTSTRAP,bootstrap_resamples=N_BOOTSTRAP,
        large_alpha_values=ALPHAS_LARGE,jump_rms_rad=JUMP_RAD,source_hashes=ctx.source_hashes,
        single_cap=10000,triple_cap=30000,first_stage_nominal_cap=250,
        raw_gtol_norm='reduced phase infinity norm',historical_report_threshold=1e-3,
        reporting_threshold_is_not_native_stop=True,objective_scaling=1.,
        AdamW_scope='Nonzero decay is a separate coordinate-dependent phase-prior control; only zero-decay equals Adam.',
        timing_scope='Serial CPU solve, including scalar callback bookkeeping; excludes initialization, objective/transfer construction, mechanics, field sampling, cache reads, rendering.',
        gpu_claim='No GPU speedup is asserted. Adam updates are first order; this implementation uses the existing float64 CPU forward model.',
        physical_validation=p._finite_ka_protocol(ctx)))
    return ctx


def seeds(mode, section, task='Single'):
    start=260869 if task=='Triple' else 260828
    count=(30 if section in ('optimizer','large_alpha') else 10) if mode=='full' else (3 if section in ('optimizer','large_alpha') else 1)
    return list(range(start,start+count))


def _spec(task='Single', method='GFE', seed=260828, **changes):
    s=dict(task=task,method=method,seed=int(seed),alpha_factor=1.,lambda_pressure_factor=1.,beta_factor=1.,
        gamma_uniformity=1.,epsilon_pressure_rel=.001,epsilon_force_factor=1.,epsilon_std_factor=1.,
        axial_ratio_factor=1.,transverse_ratio_factor=1.,stencil_factor=1.,
        first_stage_iterations=250,first_stage_factor=.05,cap=10000 if task=='Single' else 30000,**SOLVER)
    if method in ('FE', 'GFE'):
        s['optimizer'] = 'L-BFGS-B' if task == 'Single' else 'BFGS'
    s.update(changes)
    if method in ('FE', 'GFE'):
        from .fe_solver_policy import require_fe_solver
        s['optimizer'], _ = require_fe_solver(s['optimizer'], task=task)
    return s


def _weight(task, method, s):
    v=np.array([1000.,1000.,10.] if method=='Conventional' else [1.,1.,1. if task=='Single' else 10.])
    trace=v.sum();v[1]*=s['transverse_ratio_factor'];v[2]*=s['axial_ratio_factor']
    return np.diag(v*trace/v.sum())


def _build(ctx, s, initial):
    """The original scalar objective and analytic phase derivative, unscaled."""
    method=s['method'];task=s['task'];h=ctx.config.stencil_m*s['stencil_factor']
    alpha=(10. if task=='Single' else 3000.)*s['alpha_factor'] if method!='Conventional' else 0.
    target_force=mt.effective_weight_force_target_n(mt.SphereInFluid(frequency_hz=ctx.config.frequency_hz)) if method=='GFE' else np.zeros(3)
    weight=_weight(task,method,s)
    if task=='Single':
        base=method_spec('Conventional') if method=='Conventional' else fixed_single_fe_method_spec()
        method_value=replace(base,alpha_per_m=alpha,curvature_weight=weight,
            beta_curvature_per_pa=s['lambda_pressure_factor'] if method=='Conventional' else 0.)
        cond=Condition(label='reviewer-current-Single',positions=ctx.positions_m,target_m=tuple(p.MAIN_TARGET_M),
            frequency_hz=ctx.config.frequency_hz,curvature_weight=weight,normals=ctx.normals)
        obj=SingleTargetObjective(cond,method_value,stencil_spacing_m=h,force_target_n=target_force)
        if method != 'Conventional':
            from .compact_single import CompactSingleObjective
            obj = CompactSingleObjective(obj)
        return obj,[dict(cap=s['cap'],force_epsilon=0.,uniformity_epsilon=0.)],np.array([p.MAIN_TARGET_M]),dict(
            alpha_per_m=alpha,beta_n_per_m_pa=method_value.beta,curvature_weight=weight.tolist(),h_m=h,force_target_n=target_force.tolist())
    key=float(h)
    if key not in ctx.memo:
        # Main Triple uses the accepted 0.8 contraction. Rebuild transfer values
        # as well as derivative denominators when varying the stencil spacing.
        targets=contracted_targets(.8)
        stencils=tuple(mt.TargetStencil(target_id=f'T{i+1}',target_m=tuple(target),
            transfer=p._stencil_transfer(ctx.positions_m,target,ctx.config.frequency_hz,spacing_m=h),
            spacing_m=h,curvature_weight=np.eye(3)) for i,target in enumerate(targets))
        ctx.memo[key]=mt.MultiTrapProblem(case_id='reviewer-current-Triple',stencils=stencils,
            material=mt.SphereInFluid(frequency_hz=ctx.config.frequency_hz),geometry_id='square-16x16')
    problem=ctx.memo[key]
    config=replace(fixed_triple_fe_config(gtol=s['gtol']),alpha_force=alpha if method!='Conventional' else 3000.,
        beta_pressure=5e-5*s['beta_factor'],gamma_uniformity=s['gamma_uniformity'],
        curvature_weight_override=tuple(map(tuple,weight)),pressure_epsilon_rel=s['epsilon_pressure_rel'],
        smooth_stage_factors=(s['first_stage_factor'],.01),
        smooth_stage_maxiters=(min(s['first_stage_iterations'],s['cap']-1),s['cap']-min(s['first_stage_iterations'],s['cap']-1)),
        compensate_effective_gravity=method=='GFE')
    if method=='Conventional':
        config=replace(config,method='Conventional',alpha_force=0.,beta_pressure=s['lambda_pressure_factor'],
            components=mt.SPUComponents(False,True,True),pressure_epsilon_rel=0.,
            smooth_stage_factors=(.01,),smooth_stage_maxiters=(s['cap'],))
    scales=mt.estimate_smooth_scales(problem,initial,config)
    stages=[dict(cap=cap,force_epsilon=factor*scales.force_scale*s['epsilon_force_factor'] if config.components.force_smoothing else 0.,
        uniformity_epsilon=factor*scales.loss_std_scale*s['epsilon_std_factor'] if config.components.uniformity else 0.)
        for cap,factor in zip(config.smooth_stage_maxiters,config.smooth_stage_factors)]
    from .compact_multitrap import CompactMultitrapEvaluator
    obj=SimpleNamespace(problem=problem,config=config,
        compact=CompactMultitrapEvaluator(problem, config) if method != 'Conventional' else None)
    return obj,stages,np.array([t.target_m for t in problem.stencils]),dict(config.to_payload(),h_m=h,
        initial_force_scale_n=scales.force_scale,initial_loss_std_scale_n_m=scales.loss_std_scale,
        epsilon_force_multiplier=s['epsilon_force_factor'],epsilon_std_multiplier=s['epsilon_std_factor'],
        smooth_scale_floor_activated=bool(scales.force_scale==config.smooth_scale_floor or scales.loss_std_scale==config.smooth_scale_floor))


def _fg(obj, stage):
    if hasattr(obj, "fun_grad"):return obj.fun_grad
    def evaluate(q):
        if obj.compact is not None:
            ev=obj.compact.evaluate(gauge_full(q), force_epsilon=stage['force_epsilon'],
                uniformity_epsilon=stage['uniformity_epsilon'])
        else:
            ev=mt.evaluate_multitrap_objective(obj.problem,gauge_full(q),obj.config,
                force_epsilon=stage['force_epsilon'],uniformity_epsilon=stage['uniformity_epsilon'])
        return ev.value,ev.gradient_reduced
    return evaluate


def _optimize(fg, q0, settings, cap):
    """No smoothing, scaling, polishing, phase wrapping or hidden restarts."""
    if settings.get("method") in ("FE", "GFE"):
        from .fe_solver_policy import require_fe_solver
        require_fe_solver(settings.get("optimizer"), task=settings.get("task", "Single"))
    q=np.asarray(q0,float).copy();memo={};evaluations=0;hist=[]
    def f(q):
        nonlocal evaluations
        key=q.tobytes()
        if memo.get('key')!=key:
            v,g=fg(q);memo.update(key=key,value=float(v),gradient=np.asarray(g,float));evaluations+=1
        return memo['value'],memo['gradient']
    start=time.perf_counter();v0,g0=f(q)
    def record(q,iteration):
        v,g=f(q);hist.append([iteration,evaluations,time.perf_counter()-start,v,np.linalg.norm(g),np.linalg.norm(g,np.inf)])
    record(q,0)
    checkpoints={0:q.copy()};nit=0
    def callback(q):
        nonlocal nit
        nit+=1;record(q,nit)
        if nit in (500,1000,3000,10000,30000):checkpoints[nit]=np.asarray(q).copy()
    optimizer=settings['optimizer']
    if optimizer in ('BFGS','L-BFGS-B'):
        options=dict(maxiter=int(cap),gtol=settings['gtol'])
        if optimizer=='BFGS':options.update(norm=np.inf,c1=settings['c1'],c2=settings['c2'],xrtol=0.)
        else: options.update(maxcor=int(settings['maxcor']),maxls=int(settings['maxls']),ftol=settings['ftol'],maxfun=int(settings['maxfun']))
        result=minimize(lambda x:f(x)[0],q,jac=lambda x:f(x)[1],method=optimizer,callback=callback,options=options)
        q=np.asarray(result.x);nit=int(result.nit);success=bool(result.success);status=int(result.status);message=str(result.message)
    else:
        m=np.zeros_like(q);v=np.zeros_like(q);b1=settings['beta1'];b2=settings['beta2']
        success=False;status=1;message='Accepted update cap reached'
        for k in range(1,int(cap)+1):
            value,g=f(q)
            if not np.isfinite(value) or not np.all(np.isfinite(g)):
                status=3;message='Non-finite objective or gradient';break
            if np.linalg.norm(g,np.inf)<=settings['gtol']:
                success=True;status=0;message='Raw reduced-phase infinity-gradient tolerance reached';break
            m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g
            frac=(k-1)/max(1,cap-1)
            lr=settings['learning_rate']*(settings['learning_rate_end_ratio']+(1-settings['learning_rate_end_ratio'])*.5*(1+np.cos(np.pi*frac)))
            if optimizer=='AdamW':q=q*(1-lr*settings['weight_decay'])
            q=q-lr*(m/(1-b1**k))/(np.sqrt(v/(1-b2**k))+settings['adam_epsilon'])
            nit=k;record(q,nit)
            if nit in (500,1000,3000,10000,30000):checkpoints[nit]=q.copy()
        value,g=f(q)
        if np.linalg.norm(g,np.inf)<=settings['gtol']:
            success=True;status=0;message='Raw reduced-phase infinity-gradient tolerance reached'
    value,g=f(q)
    if hist[-1][0]!=nit or not np.isclose(hist[-1][3],value,rtol=0,atol=0):record(q,nit)
    checkpoints[nit]=q.copy()
    return dict(q=q,value=value,gradient=g,iterations=nit,evaluations=evaluations,success=success,status=status,message=message,
        wall_s=time.perf_counter()-start,history=np.asarray(hist),checkpoints=checkpoints,initial_value=v0,initial_gradient_l2=float(np.linalg.norm(g0)))


def solve(ctx,s,initial=None):
    if s['method'] in ('FE', 'GFE'):
        from .fe_solver_policy import require_fe_solver
        require_fe_solver(s.get('optimizer'), task=s.get('task', 'Single'))
    t=time.perf_counter()
    phase0=np.random.default_rng(s['seed']).uniform(-np.pi,np.pi,len(ctx.positions_m)) if initial is None else np.asarray(initial).copy()
    init_s=time.perf_counter()-t
    numeric={k:v for k,v in s.items() if k not in ('section','parameter','parameter_value','initialization','replicate','strategy','attempt')}
    from .fe_solver_policy import compatible_source_ids, FE_SOLVER_REVISION
    sources = compatible_source_ids(ctx.source_hashes)
    if s['method'] != 'Conventional':
        sources = dict(sources, fe_solver_policy=FE_SOLVER_REVISION)
    payload=dict(schema=SCHEMA,settings=numeric,initial_phase_content_id=digest_array(phase0),sources=sources,
        numpy=np.__version__,scipy=scipy.__version__,positions=digest_array(ctx.positions_m),normals=digest_array(ctx.normals),frequency_hz=ctx.config.frequency_hz,
        nominal_stencil_m=ctx.config.stencil_m)
    key=content_identity(canonical_json(payload).encode()).hexdigest()[:28]
    prefix=ctx.command_cache/key
    if prefix.with_suffix('.json').exists() and prefix.with_suffix('.npz').exists():
        row=json.loads(prefix.with_suffix('.json').read_text())
        with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        if digest_array(arrays['phase_rad'])!=row['phase_content_id']:raise ValueError('Command cache hash mismatch')
        row.update(s,cache_status='reused');return dict(row=row,arrays=arrays,phase=arrays['phase_rad'],targets=arrays['targets_m'])
    if ctx.config.cache_only:raise FileNotFoundError(f'Missing command cache {key}')
    t=time.perf_counter();obj,stages,targets,actual=_build(ctx,s,phase0);build_s=time.perf_counter()-t
    q=reduce_gauge(phase0,0);traces=[];stage_rows=[];checkpoints={};offset=0;eval_offset=0;wall_offset=0.
    with threadpool_limits(limits=1):
        for i,stage in enumerate(stages):
            r=_optimize(_fg(obj,stage),q,s,stage['cap']);q=r['q']
            trace=r['history'].copy();trace[:,0]+=offset;trace[:,1]+=eval_offset;trace[:,2]+=wall_offset
            traces.append(np.c_[np.full(len(trace),i),trace]);checkpoints.update({offset+k:v for k,v in r['checkpoints'].items()})
            offset+=r['iterations'];eval_offset+=r['evaluations'];wall_offset+=r['wall_s']
            stage_rows.append(dict(stage=i,**stage,iterations=r['iterations'],evaluations=r['evaluations'],success=r['success'],status=r['status'],message=r['message'],
                terminal_value=r['value'],terminal_gradient_l2=float(np.linalg.norm(r['gradient'])),terminal_gradient_inf=float(np.linalg.norm(r['gradient'],np.inf))))
    phase=np.angle(np.exp(1j*gauge_full(q)))
    terminal_value,terminal_grad=_fg(obj,stages[-1])(q)
    initial_value,initial_grad=_fg(obj,stages[-1])(reduce_gauge(phase0,0))
    row=dict(s,run_id=key,phase_content_id=digest_array(phase),initial_phase_content_id=digest_array(phase0),
        objective_n_m=float(terminal_value),initial_objective_final_stage_n_m=float(initial_value),
        terminal_gradient_l2=float(np.linalg.norm(terminal_grad)),terminal_gradient_inf=float(np.linalg.norm(terminal_grad,np.inf)),
        relative_gradient_l2=float(np.linalg.norm(terminal_grad)/max(np.linalg.norm(initial_grad),1e-30)),
        initial_gradient_l2=float(np.linalg.norm(initial_grad)),native_success=r['success'],native_status=r['status'],native_message=r['message'],
        raw_gtol_met=bool(np.linalg.norm(terminal_grad,np.inf)<=s['gtol']),
        legacy_reporting_threshold_met=bool(np.linalg.norm(terminal_grad)<=1e-3),
        iterations=offset,evaluations=eval_offset,solve_wall_s=wall_offset,initialization_s=init_s,objective_setup_s=build_s,
        actual_config_json=json.dumps(actual),stages_json=json.dumps(stage_rows),cache_status='computed',
        stationarity_scope='original loss, reduced gauge, raw SI units',
        same_objective_optimizer_control=not(s['optimizer']=='AdamW' and s['weight_decay']!=0.))
    if s["task"] == "Single":
        raw=obj._raw_metrics(phase,False)
        row.update(surrogate_residual_n=float(raw['force_residual_norm_n']),weighted_curvature_n_m=float(raw['weighted_curvature_n_m']),target_pressure_pa=float(raw['pressure_abs_pa']))
    else:
        ev=mt.evaluate_multitrap_objective(obj.problem,phase,obj.config,force_epsilon=stages[-1]['force_epsilon'],uniformity_epsilon=stages[-1]['uniformity_epsilon'])
        nominal=replace(fixed_triple_fe_config(gtol=1e-8),compensate_effective_gravity=True)
        ns=mt.estimate_smooth_scales(obj.problem,phase0,nominal)
        common=mt.evaluate_multitrap_objective(obj.problem,phase,nominal,force_epsilon=.01*ns.force_scale,uniformity_epsilon=.01*ns.loss_std_scale)
        row.update(common_nominal_loss_std_n_m=float(np.std(common.local_losses)),local_loss_std_n_m=float(np.std(ev.local_losses)),
            surrogate_residual_n=float(max(np.linalg.norm(local.force_vector_n-(mt.effective_weight_force_target_n(obj.problem.material) if s['method']=='GFE' else 0.)) for local in ev.locals)))
    arrays=dict(phase_rad=phase,initial_phase_rad=phase0,targets_m=targets,history=np.concatenate(traces),
        checkpoint_iterations=np.array(sorted(checkpoints)),checkpoint_phases_rad=np.array([gauge_full(checkpoints[k]) for k in sorted(checkpoints)]),
        history_columns=np.array(['stage','accepted_iteration','evaluations','wall_s','loss_n_m','gradient_l2','gradient_inf']),payload_json=np.array(json.dumps(payload)))
    np.savez_compressed(prefix.with_suffix('.npz'),**arrays);_json(prefix.with_suffix('.json'),row)
    print(f"{s.get('section','reference')} {s['task']} {s['method']} {s['optimizer']} {s.get('parameter','nominal')}={s.get('parameter_value',1)} seed{s['seed']}: {offset} it, g={row['terminal_gradient_inf']:.3g}, {r['message']}",flush=True)
    return dict(row=row,arrays=arrays,phase=phase,targets=targets)


def _field(ctx,phase,points):
    field=p._array_field(ctx,phase)
    return np.concatenate([np.atleast_1d(field.pressure(points[i:i+1024])).ravel() for i in range(0,len(points),1024)])


def correspondence(ctx,job,reference,n=9,width=.75):
    lam=343./ctx.config.frequency_hz
    offsets=np.array(np.meshgrid(*([np.linspace(-width*lam,width*lam,n)]*3),indexing='ij')).reshape(3,-1).T
    points=np.concatenate([target+offsets for target in job['targets']])
    refkey=('field',reference['row']['phase_content_id'],n,width)
    if refkey not in ctx.memo:ctx.memo[refkey]=_field(ctx,reference['phase'],points)
    pr=ctx.memo[refkey];pc=_field(ctx,job['phase'],points)
    cu=float(abs(np.mean(np.exp(1j*(job['phase']-reference['phase'])))))
    cp=float(np.clip(abs(np.vdot(pr,pc))/max(np.linalg.norm(pr)*np.linalg.norm(pc),1e-30),0,1))
    job['row'].update(command_cosine_to_fixed_GFE=cu,field_cosine_to_fixed_GFE=cp,
        fixed_reference_run_id=reference['row']['run_id'],fixed_reference_phase_content_id=reference['row']['phase_content_id'])
    job['arrays'].update(points_m=points,pressure_complex_pa=pc,reference_pressure_complex_pa=pr,reference_phase_rad=reference['phase'])


def validate(ctx,job):
    rows=[]
    for index,target in enumerate(job['targets']):
        v=p._finite_ka_validation(ctx,job['phase'],target,key='reviewer-'+job['row']['run_id']+f'-T{index+1}')
        row=p._static_validation_row(job['row']['method'],v)
        row.update(run_id=job['row']['run_id'],seed=job['row']['seed'],task=job['row']['task'],target_id=index+1,phase_content_id=job['row']['phase_content_id'])
        rows.append(row)
    resolved=all(r['finite_ka_root_found'] for r in rows)
    job['row'].update(mechanics_evaluated=True,all_roots=resolved,all_locally_restoring=all(r['finite_ka_locally_restoring'] for r in rows),
        worst_displacement_a=max(r['finite_ka_displacement_a'] for r in rows) if resolved else np.nan,
        minimum_stiffness_n_m=min(r['finite_ka_stiffness_min_n_m'] for r in rows) if resolved else np.nan)
    return rows


def nominal(ctx,task,seed):
    key=('reference',task,seed)
    if key not in ctx.memo:ctx.memo[key]=solve(ctx,_spec(task=task,seed=seed,section='reference'))
    return ctx.memo[key]


def _export(ctx,section,jobs,mechanics):
    out=ctx.output_root/section;out.mkdir(exist_ok=True)
    commands=out/'commands';commands.mkdir(exist_ok=True)
    for j in jobs:
        r=j['row'];r.setdefault('mechanics_evaluated',False);r['evidence_mode']=ctx.mode
        # A solver result may be reused in several logical sweeps. Each logical
        # row is retained, while raw commands share their physical run ID.
        np.savez_compressed(commands/(r['run_id']+'.npz'),**j['arrays'])
    references={}
    for j in jobs:
        r=j['row'];ref=nominal(ctx,r['task'],r['seed'])
        if mechanics and not ref['row'].get('mechanics_evaluated',False):mechanics.extend(validate(ctx,ref))
        ref['row'].update(command_cosine_to_fixed_GFE=1.,field_cosine_to_fixed_GFE=1.)
        references[(r['task'],r['seed'])]=dict(ref['row'])
    pd.DataFrame(references.values()).to_csv(out/'frozen_GFE_reference_audit.csv',index=False)
    frame=pd.DataFrame([j['row'] for j in jobs]);frame.to_csv(out/'command_runs.csv',index=False)
    if mechanics:pd.DataFrame(mechanics).drop_duplicates(['run_id','target_id']).to_csv(out/'target_mechanics.csv',index=False)
    metrics=[m for m in ('solve_wall_s','iterations','evaluations','terminal_gradient_l2','terminal_gradient_inf','relative_gradient_l2',
        'objective_n_m','surrogate_residual_n','command_cosine_to_fixed_GFE','field_cosine_to_fixed_GFE','worst_displacement_a','minimum_stiffness_n_m','common_nominal_loss_std_n_m') if m in frame]
    groups=[x for x in ('task','method','optimizer','parameter','parameter_value','initialization','strategy','attempt') if x in frame and not frame[x].isna().all()]
    summaries=[]
    for key,g in frame.groupby(groups,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        tag=dict(zip(groups,key));rng=np.random.default_rng(SEED_BOOTSTRAP)
        for metric in metrics:
            vals=pd.to_numeric(g[metric],errors='coerce').to_numpy();vals=vals[np.isfinite(vals)]
            lo=hi=np.nan
            if len(vals)>=2:
                boot=np.median(vals[rng.integers(0,len(vals),(N_BOOTSTRAP,len(vals)))],axis=1);lo,hi=np.quantile(boot,[.025,.975])
            summaries.append(dict(**tag,metric=metric,n_pairs=g.seed.nunique(),n_valid=len(vals),n_missing=len(g)-len(vals),
                median=np.median(vals) if len(vals) else np.nan,q25=np.quantile(vals,.25) if len(vals) else np.nan,q75=np.quantile(vals,.75) if len(vals) else np.nan,
                minimum=vals.min() if len(vals) else np.nan,maximum=vals.max() if len(vals) else np.nan,ci95_lo=lo,ci95_hi=hi))
    pd.DataFrame(summaries).to_csv(out/'summary_median_IQR_range_CI.csv',index=False)
    native=frame.groupby(groups,dropna=False).agg(n=('seed','size'),native_success=('native_success','sum'),raw_gtol_met=('raw_gtol_met','sum'))
    native.to_csv(out/'native_stopping_counts.csv')
    # Paired changes versus the same-seed, same-task nominal GFE; never treat
    # three targets, optimizer iterates or jump attempts as independent seeds.
    paired=[]
    for j in jobs:
        r=j['row'];ref=references[(r['task'],r['seed'])];pair=dict(run_id=r['run_id'],seed=r['seed'],**{key:r.get(key,np.nan) for key in groups})
        for metric in metrics:
            if metric in r and metric in ref:pair['delta_'+metric]=r[metric]-ref[metric]
        paired.append(pair)
    pairs=pd.DataFrame(paired);pairs.to_csv(out/'paired_changes_to_fixed_GFE.csv',index=False)
    paired_summary=[]
    for key,g in pairs.groupby(groups,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        tags=dict(zip(groups,key));rng=np.random.default_rng(SEED_BOOTSTRAP)
        for col in [c for c in pairs if c.startswith('delta_')]:
            vals=pd.to_numeric(g[col],errors='coerce').dropna().to_numpy();lo=hi=np.nan
            if len(vals)>=2:
                boot=np.median(vals[rng.integers(0,len(vals),(N_BOOTSTRAP,len(vals)))],axis=1);lo,hi=np.quantile(boot,[.025,.975])
            paired_summary.append(dict(**tags,metric=col,n_valid_pairs=len(vals),median_paired_difference=np.median(vals) if len(vals) else np.nan,
                ci95_lo=lo,ci95_hi=hi,bootstrap_seed=SEED_BOOTSTRAP,bootstrap_resamples=N_BOOTSTRAP,sampling_unit='paired seed; targets nested inside command'))
    pd.DataFrame(paired_summary).to_csv(out/'paired_bootstrap_to_fixed_GFE.csv',index=False)
    if section=='large_alpha':
        contrasts=[]
        for alpha,g in frame.groupby('parameter_value'):
            for label in ('Twin','Random'):
                a=g[g.initialization.eq(label)].set_index('seed');b=g[g.initialization.eq('Vortex')].set_index('seed')
                for metric in ('twofold_amplitude_modulation','field_cosine_to_fixed_GFE','surrogate_residual_n','worst_displacement_a','minimum_stiffness_n_m'):
                    if metric not in a:continue
                    delta=(a[metric]-b[metric]).dropna().to_numpy();rng=np.random.default_rng(SEED_BOOTSTRAP);lo=hi=np.nan
                    if len(delta)>=2:lo,hi=np.quantile(np.median(delta[rng.integers(0,len(delta),(N_BOOTSTRAP,len(delta)))],axis=1),[.025,.975])
                    contrasts.append(dict(alpha_per_m=alpha,contrast=label+' minus Vortex',metric=metric,n_pairs=len(delta),
                        median_paired_change=np.median(delta) if len(delta) else np.nan,ci95_lo=lo,ci95_hi=hi))
        pd.DataFrame(contrasts).to_csv(out/'initialization_paired_contrasts.csv',index=False)
    if 'worst_displacement_a' in frame:
        outcomes=[]
        for key,g in frame.groupby(groups,dropna=False):
            if not isinstance(key,tuple):key=(key,)
            evaluated=g[g.mechanics_evaluated.astype(bool)]
            for limit in (.1,.25,.5,1.):
                good=evaluated.all_locally_restoring.fillna(False).astype(bool)&(evaluated.worst_displacement_a<=limit)
                outcomes.append(dict(**dict(zip(groups,key)),displacement_limit_a=limit,n_total=len(g),n_mechanically_evaluated=len(evaluated),n_resolved_restoring_localized=int(good.sum())))
        pd.DataFrame(outcomes).to_csv(out/'mechanical_localization_thresholds.csv',index=False)
    return frame


def optimizer_robustness(ctx, mechanics=True, optimizers=None):
    jobs=[];mech=[]
    optimizers=OPTIMIZERS if optimizers is None else tuple(optimizers)
    if not optimizers or not set(optimizers).issubset(OPTIMIZERS):
        raise ValueError('Unknown optimizer selection')
    # Serial solves are intentional: these timings share one environment and
    # cannot be contaminated by concurrent sweep/validator work in this cell.
    for seed in seeds(ctx.mode,'optimizer'):
        ref=nominal(ctx,'Single',seed)
        for method in METHODS:
            method_optimizers = ('L-BFGS-B',) if method in ('FE', 'GFE') else optimizers
            for optimizer in method_optimizers:
                s=_spec(method=method,seed=seed,optimizer=optimizer,
                    weight_decay=.01 if optimizer=='AdamW' else 0.,section='optimizer',parameter='optimizer',parameter_value=optimizer)
                j=solve(ctx,s);jobs.append(j)
    for j in jobs:correspondence(ctx,j,nominal(ctx,'Single',j['row']['seed']))
    if mechanics:
        for j in jobs:mech.extend(validate(ctx,j))
    frame=_export(ctx,'optimizer',jobs,mech)
    return dict(section='optimizer',rows=len(frame),output=str(ctx.output_root/'optimizer'))


def sensitivity_design(mode):
    specs=[]
    for parameter,values,tasks,methods,units in GRIDS:
        for task in tasks:
            for seed in seeds(mode,'sensitivity',task):
                for method in methods:
                    for value in values:
                        specs.append(_spec(task,method,seed,section='continuous',parameter=parameter,parameter_value=value,**{parameter:value}))
    # Three-by-three crossed grids for the two coupled regularization choices.
    for seed in seeds(mode,'sensitivity','Triple'):
        for a in (.1,1.,10.):
            for e in (.1,1.,10.):
                specs.append(_spec('Triple','GFE',seed,section='interactions',parameter='alpha_x_epsilon_force',parameter_value=f'{a:g},{e:g}',alpha_factor=a,epsilon_force_factor=e))
        for gamma in (.1,1.,10.):
            for e in (.1,1.,10.):
                specs.append(_spec('Triple','GFE',seed,section='interactions',parameter='gamma_x_epsilon_std',parameter_value=f'{gamma:g},{e:g}',gamma_uniformity=gamma,epsilon_std_factor=e))
    return specs


def _continuous_one(ctx,group,mechanics):
    local=SimpleNamespace(**dict(vars(ctx),memo=dict(ctx.memo)))
    s=group[0];j=solve(local,s);correspondence(local,j,nominal(local,s['task'],s['seed']))
    outcomes=validate(local,j) if mechanics else []
    j['row']['timing_comparison_eligible']=False
    j['row']['timing_note']='Co-scheduled sensitivity workers; use C7 serial timings for optimizer cost comparisons.'
    aliases=[dict(j,row=dict(j['row'],**alias),arrays=dict(j['arrays'])) for alias in group]
    return aliases,outcomes


def continuous_sensitivity(ctx, mechanics=True, selected_parameters=None, workers=4, _worker_process=False):
    if workers>1 and not _worker_process:
        # Spawn a guarded module process, so this also works from a notebook's
        # interactive __main__ after JAX initialization. No shell is involved.
        import pickle,subprocess,sys,tempfile
        with tempfile.TemporaryDirectory(prefix='worker_context_',dir=ctx.output_root) as temporary:
            input_path=Path(temporary)/'trusted_context.pkl';output_path=Path(temporary)/'result.json'
            with input_path.open('wb') as stream:pickle.dump(dict(ctx=ctx,mechanics=mechanics,selected_parameters=selected_parameters,workers=workers),stream)
            command=[sys.executable,str(Path(__file__).with_name('reviewer_worker.py')),str(input_path),str(output_path)]
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:print(line,end='',flush=True)
            code=proc.wait()
            if code:raise RuntimeError(f'Sensitivity worker exited with status {code}; completed caches are retained.')
            return json.loads(output_path.read_text())
    from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    specs=sensitivity_design(ctx.mode)
    if selected_parameters is not None:specs=[s for s in specs if s['parameter'] in selected_parameters]
    jobs=[];mech=[];groups={}
    # Freeze references serially and deduplicate the many nominal OAT slots.
    for task,seed in sorted({(s['task'],s['seed']) for s in specs}):
        ref=nominal(ctx,task,seed)
        if mechanics:mech.extend(validate(ctx,ref))
    for s in specs:
        key=canonical_json({k:v for k,v in s.items() if k not in ('section','parameter','parameter_value')})
        groups.setdefault(key,[]).append(s)
    # Spawn avoids inheriting an already initialized JAX/BLAS thread runtime
    # after the main notebook figures. Workers are importable module functions.
    pool_type=ProcessPoolExecutor if workers>1 else ThreadPoolExecutor
    kw=dict(max_workers=max(1,int(workers)))
    if pool_type is ProcessPoolExecutor:kw['mp_context']=mp.get_context('spawn')
    with pool_type(**kw) as pool:
        for future in as_completed([pool.submit(_continuous_one,ctx,group,mechanics) for group in groups.values()]):
            completed,outcomes=future.result();jobs.extend(completed);mech.extend(outcomes)
            temporary=ctx.output_root/'continuous_progress.writing.csv'
            pd.DataFrame([q['row'] for q in jobs]).to_csv(temporary,index=False)
            temporary.replace(ctx.output_root/'continuous_progress.csv')
    jobs.sort(key=lambda j:(j['row']['parameter'],j['row']['task'],j['row']['method'],j['row']['seed'],str(j['row']['parameter_value'])))
    return dict(section='continuous',rows=len(_export(ctx,'continuous',jobs,mech)),output=str(ctx.output_root/'continuous'))


def solver_sensitivity(ctx,mechanics=True,selected_parameters=None):
    jobs=[];mech=[]
    for seed in seeds(ctx.mode,'sensitivity'):
        for parameter,values,optimizer in SOLVER_GRIDS:
            if selected_parameters is not None and parameter not in selected_parameters:
                continue
            # Preserve existing Conventional controls; FE/GFE controls vary
            # only the configured compact L-BFGS-B method.
            methods = ('GFE',) if optimizer == 'L-BFGS-B' else (
                ('Conventional',) if parameter == 'learning_rate' else ())
            for method in methods:
                for value in values:
                    s=_spec(method=method,seed=seed,optimizer=optimizer,section='solver_controls',parameter=parameter,parameter_value=value,**{parameter:value})
                    j=solve(ctx,s);correspondence(ctx,j,nominal(ctx,'Single',seed));jobs.append(j)
    if mechanics:
        for j in jobs:mech.extend(validate(ctx,j))
    return dict(section='solver_controls',rows=len(_export(ctx,'solver_controls',jobs,mech)),output=str(ctx.output_root/'solver_controls'))


def _rms_perturb(rng,n,rms):
    v=rng.normal(size=n);v-=v.mean();return v*rms/np.sqrt(np.mean(v*v))


def large_alpha(ctx,mechanics=True,alphas=ALPHAS_LARGE):
    from .appendix_b3_mechanics import _load_commands,transverse_descriptor
    sources={q['name']:q for q in _load_commands(ctx.root)}
    jobs=[];mech=[]
    for seed in seeds(ctx.mode,'large_alpha'):
        rng=np.random.default_rng(seed+11000000)
        dither=_rms_perturb(rng,len(ctx.positions_m),.05)
        initializations=dict(Random=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(ctx.positions_m)),
            Vortex=sources['Vortex']['phase']+dither,Twin=sources['Twin']['phase']+dither)
        for alpha in alphas:
            for init,phase0 in initializations.items():
                s=_spec(seed=seed,alpha_factor=alpha/10,section='large_alpha',parameter='alpha_per_m',parameter_value=alpha,initialization=init)
                j=solve(ctx,s,phase0);correspondence(ctx,j,nominal(ctx,'Single',seed))
                desc,_=transverse_descriptor(p._array_field(ctx,j['phase']),p.MAIN_TARGET_M,343/ctx.config.frequency_hz)
                j['row'].update(desc,initialization_dither_rms_rad=.05 if init!='Random' else 0.,source_morphology_phase_content_id=sources[init]['source_phase_content_id'] if init!='Random' else '')
                if mechanics:mech.extend(validate(ctx,j))
                jobs.append(j)
    return dict(section='large_alpha',rows=len(_export(ctx,'large_alpha',jobs,mech)),output=str(ctx.output_root/'large_alpha'))


def jumps(ctx,mechanics=True,alphas=(1e4,1e6,1e8),origin='cold'):
    if origin not in ('cold','nominal_warm'):raise ValueError('origin must be cold or nominal_warm')
    jobs=[];mech=[]
    for seed in seeds(ctx.mode,'sensitivity'):
        ref=nominal(ctx,'Single',seed)
        for alpha in alphas:
            base=_spec(seed=seed,alpha_factor=alpha/10,section='jumps',parameter='alpha_per_m',parameter_value=alpha)
            initial=(np.random.default_rng(seed).uniform(-np.pi,np.pi,len(ctx.positions_m)) if origin=='cold' else ref['phase'])
            common=solve(ctx,dict(base,initialization=origin),initial)
            for strategy in ('Phase jump','Random restart','Local restart'):
                rng=np.random.default_rng(seed+22000000);best=common;calls=0;iters=0;totalwall=0.
                for attempt in range(7):
                    if attempt==0:j=dict(common,row=dict(common['row']),arrays=dict(common['arrays']))
                    else:
                        if strategy=='Phase jump':initial=best['phase']+_rms_perturb(rng,len(best['phase']),JUMP_RAD[attempt-1])
                        elif strategy=='Random restart':initial=rng.uniform(-np.pi,np.pi,len(best['phase']))
                        else:initial=best['phase'].copy()
                        j=solve(ctx,base,initial)
                    calls+=j['row']['evaluations'];iters+=j['row']['iterations'];totalwall+=j['row']['solve_wall_s']
                    improved=j['row']['objective_n_m']<best['row']['objective_n_m']
                    if improved:best=j
                    j['row'].update(strategy=strategy,attempt=attempt,accepted_improvement=improved,
                        common_origin=origin,
                        cumulative_evaluations=calls,cumulative_iterations=iters,cumulative_solve_s=totalwall,
                        best_objective_n_m=best['row']['objective_n_m'],best_phase_content_id=best['row']['phase_content_id'],
                        candidate_jump_rms_rad=JUMP_RAD[attempt-1] if attempt and strategy=='Phase jump' else 0.,
                        best_improvement_from_common_n_m=common['row']['objective_n_m']-best['row']['objective_n_m'])
                    j['arrays']['best_phase_rad']=best['phase'].copy()
                    correspondence(ctx,j,ref)
                    if mechanics and attempt==6:
                        selected=dict(row=dict(j['row'],run_id=j['row']['run_id']+'-best',phase_content_id=best['row']['phase_content_id']),phase=best['phase'],targets=j['targets'])
                        mech.extend(validate(ctx,selected))
                        j['row'].update({k:v for k,v in selected['row'].items() if k in ('mechanics_evaluated','all_roots','all_locally_restoring','worst_displacement_a','minimum_stiffness_n_m')})
                        j['row']['mechanics_phase_role']='best-so-far, not necessarily last proposal'
                    jobs.append(dict(j,row=dict(j['row']),arrays=dict(j['arrays'])))
    section='jumps' if origin=='cold' else 'jumps_from_nominal_warm'
    # One candidate can recur in several strategies. Preserve each logical
    # incumbent separately; a run-ID command archive alone cannot encode it.
    destination=ctx.output_root/section;destination.mkdir(exist_ok=True)
    incumbents={};manifest=[]
    for index,j in enumerate(jobs):
        key=f'incumbent_{index:04d}';incumbents[key]=j['arrays']['best_phase_rad']
        manifest.append(dict(array_key=key,seed=j['row']['seed'],alpha_per_m=j['row']['parameter_value'],
            strategy=j['row']['strategy'],attempt=j['row']['attempt'],candidate_run_id=j['row']['run_id'],
            incumbent_phase_content_id=digest_array(incumbents[key]),origin=origin))
    np.savez_compressed(destination/'logical_incumbent_phases.npz',**incumbents)
    pd.DataFrame(manifest).to_csv(destination/'logical_incumbent_manifest.csv',index=False)
    return dict(section=section,rows=len(_export(ctx,section,jobs,mech)),output=str(ctx.output_root/section))


def design_table(ctx):
    rows=[]
    for parameter,values,tasks,methods,units in GRIDS:
        rows.append(dict(parameter=parameter,values_json=json.dumps(values),distinct_values=len(values),tasks='/'.join(tasks),methods='/'.join(methods),units=units,
            seeds_per_task_smoke=1,seeds_per_task_full=10,section='continuous',status='Executable; completion comes from command_runs.csv'))
    for parameter,values,optimizer in SOLVER_GRIDS:
        rows.append(dict(parameter=parameter,values_json=json.dumps(values),distinct_values=len(values),tasks='Single',methods='GFE; learning rate also FE/Conventional',units='see protocol',
            seeds_per_task_smoke=1,seeds_per_task_full=10,section='solver_controls',status='Executable; completion comes from command_runs.csv'))
    pd.DataFrame(rows).to_csv(ctx.output_root/'sensitivity_design.csv',index=False)
    specs=sensitivity_design(ctx.mode)
    _json(ctx.output_root/'continuous_design.json',specs)
    return rows
