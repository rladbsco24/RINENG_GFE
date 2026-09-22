"""GFE fixed-reference sensitivity and matched Triple component ablations.

Frozen references are the nominal main Single/Triple GFE commands, independently
validated with the current elastic total-force model. FE is the no-gravity
control. Raw optimizer messages and all target-level outcomes are retained.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import os
from types import SimpleNamespace

from rineng_content_id import content_identity
import json
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import pipeline as p
from .branch_figures import projective_similarity
from .cache import CacheStore, canonical_json, digest_array, dependency_snapshot
from .config import RunConfig, square_positions, square_normals
from .fixed_fe_endpoints import fixed_single_fe_method_spec, fixed_triple_fe_config
from .fig4_threepoint_selection import contracted_targets, make_problem
from .gorkov_core import Condition, SingleTargetObjective, reduce_gauge
from .multitrap import (SphereInFluid, effective_weight_force_target_n,
                       run_multitrap, evaluate_multitrap_objective)
from .sota import solve_corrected_gorkov_fe

def _appendix_c_io_path(path):
    path = Path(path)
    text = str(path)
    prefix = chr(92) * 2 + '?' + chr(92)
    if os.name == 'nt' and path.is_absolute() and not text.startswith(prefix):
        return Path(prefix + text)
    return path


SCHEMA = 'appendix-C-GFE-fixed-reference-v1'
ALPHA_FACTORS = (.3, 1., 3.)
SINGLE_SEED = 172817509
TRIPLE_SEEDS = (260869, 260870, 260871)
ABLATIONS = ('GFE', 'No force smoothing', 'No pressure', 'No uniformity', 'FE')
SHORT_LABELS = ('GFE', 'No force\nsmoothing', 'No pressure\npenalty', 'No loss\nuniformity', 'FE')
COLORS = {'GFE':'#16659A', 'FE':'#D08021'}


def _hash_sources():
    base=Path(__file__).parent
    return {name:content_identity((base/name).read_bytes()).hexdigest()
            for name in ('gorkov_core.py','multitrap.py','sota.py','exact_validator.py')}


def _context(root, output, cache_only):
    return SimpleNamespace(config=RunConfig(), positions_m=square_positions(), normals=square_normals(),
        output_root=output, recompute=False,
        cache=CacheStore(output/'cache',namespace=p.PIPELINE_IMPLEMENTATION,read_only=cache_only),
        data_root=root/'runtime'/'data')


def _canonical_config(config):
    value=dict(config)
    value.setdefault('compensate_effective_gravity',False)
    return canonical_json(value)


def _source_commands(root):
    """Index compatible scientific configurations, without changing source caches."""
    indexed={}
    folders=[root/'smoke_outputs/cache/revision_triple_ablation',
             root/'appendix_outputs/C/cache/C_triple_ablation',
             root/'smoke_outputs/cache/final_fig6_vertical_compensated_fe']
    for folder in folders:
        for path in sorted(folder.glob('*.pkl')):
            with _appendix_c_io_path(path).open('rb') as stream: record=pickle.load(stream)
            if record.get('namespace') != p.PIPELINE_IMPLEMENTATION: continue
            q=record['payload']
            from .fe_solver_policy import FE_SOLVER_REVISION
            if q.get('fe_solver_policy') != FE_SOLVER_REVISION:
                continue
            config=q.get('compensated_config',q.get('config'))
            if config is None:continue
            key=(q.get('problem'),_canonical_config(config),q.get('seed'),
                 q.get('initial_phase_content_id',q.get('initial_phase',q.get('initial'))))
            indexed[key]=(record['value'],str(path.relative_to(root)))
    return indexed


def _source_validations(root,ctx):
    indexed={};required=dict(p._finite_ka_protocol(ctx));required.pop('dependencies',None)
    allowed_namespaces=set(getattr(ctx,'validation_source_namespaces',(p.PIPELINE_IMPLEMENTATION,)))
    for folder in (root/'smoke_outputs/cache/finite_ka_static_validation',
                   root/'appendix_outputs/C/cache/finite_ka_static_validation',
                   ctx.output_root/'cache/finite_ka_static_validation',
                   *[Path(folder)/'finite_ka_static_validation' for folder in getattr(ctx,'validation_source_roots',())]):
        for path in sorted(folder.glob('*.pkl')):
            with _appendix_c_io_path(path).open('rb') as stream: record=pickle.load(stream)
            q=record.get('payload',{});protocol=dict(q.get('protocol',{}));protocol.pop('dependencies',None)
            if (record.get('namespace') in allowed_namespaces
                and canonical_json(protocol)==canonical_json(required)
                and q.get('positions')==digest_array(ctx.positions_m)
                and q.get('normals')==digest_array(ctx.normals)
                and q.get('frequency_hz')==ctx.config.frequency_hz):
                indexed[(q.get('phase'),tuple(q.get('target',())))]=(record['value'],str(path.relative_to(root)))
    return indexed


def specifications(ctx):
    specs=[]
    for task,seed,nominal in (('Single',SINGLE_SEED,10.),('Triple',TRIPLE_SEEDS[0],3000.)):
        for method in ('GFE','FE'):
            for factor in ALPHA_FACTORS:
                specs.append(dict(task=task,seed=seed,method=method,alpha=nominal*factor,
                                  alpha_factor=factor,ablation='GFE' if method=='GFE' else 'FE',sensitivity=True))
    for seed in TRIPLE_SEEDS:
        for label in ABLATIONS:
            if seed==TRIPLE_SEEDS[0] and label in ('GFE','FE'):continue
            specs.append(dict(task='Triple',seed=seed,method='FE' if label=='FE' else 'GFE',alpha=3000.,
                              alpha_factor=1.,ablation=label,sensitivity=False))
    for spec in specs:
        spec['run_id']=f"{spec['task']}_{spec['ablation'].replace(' ','_')}_alpha{spec['alpha']:g}_seed{spec['seed']}"
    return specs


def _solve_one(root,ctx,problem,source_index,spec,provided=None):
    seed=spec['seed'];alpha=spec['alpha'];compensated=spec['method']=='GFE'
    initial=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(ctx.positions_m))
    target_force=effective_weight_force_target_n(SphereInFluid(frequency_hz=ctx.config.frequency_hz))
    source='';status='';cap=int(spec.get('maxiter',10000));evidence=getattr(ctx,'evidence_status','SMOKE')
    if spec['task']=='Single':
        method=replace(fixed_single_fe_method_spec(),alpha_per_m=alpha)
        condition=Condition(label='fixed-single-fe-alpha10',positions=ctx.positions_m,target_m=tuple(p.MAIN_TARGET_M),
                            frequency_hz=ctx.config.frequency_hz,curvature_weight=method.curvature_weight,normals=ctx.normals)
        objective=SingleTargetObjective(condition,method,force_target_n=target_force if compensated else None)
        payload=dict(contract='fixed-single-fe-gravity-endpoint-v1' if compensated else 'fixed-single-fe-endpoint-v2-uniform-10000-cap',
                     alpha_per_m=alpha,iteration_cap=10000,target_m=p.MAIN_TARGET_M.tolist(),seed=seed,
                     initial_phase=digest_array(initial),gtol=ctx.config.gtol,
                     positions=digest_array(ctx.positions_m),normals=digest_array(ctx.normals))
        if compensated:payload['force_target_n']=target_force.tolist()
        kind='fixed_single_fe_gravity_endpoint' if compensated else 'fixed_single_fe_endpoint'
        original=CacheStore(root/'smoke_outputs/cache',namespace=p.PIPELINE_IMPLEMENTATION,read_only=True)
        old_path=original.path_for(kind,payload,create_parent=False)
        if provided is not None:
            result=provided['run'];source=str(provided['row'].get('cache_path',provided['row'].get('source','shared production nominal bank')));status='shared production nominal command'
        elif old_path.exists():
            result,_,_=original.get_or_compute(kind,payload,lambda:None)
            source=str(old_path.relative_to(root));status='compatible main cache'
        else:
            result,status,path=ctx.cache.get_or_compute('C_GFE_single',dict(payload,source_hashes=_hash_sources()),
                lambda:solve_corrected_gorkov_fe(objective,initial,maxiter=10000,gtol=ctx.config.gtol))
            source=str(path.relative_to(root))
        phase=np.asarray(result.phase_rad);targets=np.asarray([p.MAIN_TARGET_M])
        config=dict(alpha=alpha,curvature_weight=method.curvature_weight.tolist(),
                    force_target_n=(target_force if compensated else np.zeros(3)).tolist(),gtol=ctx.config.gtol)
        _,gradient=objective.fun_grad(reduce_gauge(phase,0))
        native=dict(native_success=result.success,native_status=result.status,native_message=result.message,
                    terminal_gradient_l2=float(np.linalg.norm(gradient)),terminal_gradient_inf=float(np.linalg.norm(gradient,np.inf)),
                    iterations=result.iterations,evaluations=result.evaluations,command_time_s=float(result.history_wall_s[-1]),
                    command_timing_scope='solver wall time; source session retained')
        stages=[]
    else:
        config=replace(fixed_triple_fe_config(gtol=ctx.config.gtol),alpha_force=alpha,
                       compensate_effective_gravity=compensated)
        if cap!=10000:config=replace(config,smooth_stage_maxiters=(250,cap-250))
        deleted={'No force smoothing':'force_smoothing','No pressure':'pressure_retention','No uniformity':'uniformity'}.get(spec['ablation'])
        if deleted:config=replace(config,components=replace(config.components,**{deleted:False}))
        key=(problem.fingerprint,_canonical_config(config.to_payload()),seed,digest_array(initial))
        if provided is not None:
            if 'config' in provided and _canonical_config(provided['config'].to_payload())!=_canonical_config(config.to_payload()):
                raise ValueError('Shared nominal Triple configuration does not match C1')
            if 'problem' in provided and provided['problem'].fingerprint!=problem.fingerprint:
                raise ValueError('Shared nominal Triple problem does not match C1')
            result=provided['run'];source=str(provided['row'].get('cache_path',provided['row'].get('source','shared production nominal bank')));status='shared production nominal command'
        elif key in source_index:
            result,source=source_index[key];status='compatible main or prior matched cache'
        else:
            payload=dict(contract=SCHEMA,problem=problem.fingerprint,config=config.to_payload(),seed=seed,
                         initial_phase=digest_array(initial),source_hashes=_hash_sources())
            result,status,path=ctx.cache.get_or_compute('C_GFE_triple',payload,
                lambda:run_multitrap(problem,config,seed=seed,initial_phases=initial,record_history=True,
                    metadata=dict(display_method=spec['method'],experiment_role='matched GFE Appendix C',evidence_status=evidence)))
            source=str(path.relative_to(root))
        phase=np.asarray(result.phases);targets=np.array([s.target_m for s in problem.stencils]);stage=result.stages[-1]
        ev=evaluate_multitrap_objective(problem,phase,config,force_epsilon=stage.force_epsilon,uniformity_epsilon=stage.uniformity_epsilon)
        native=dict(native_success=stage.success,native_status=stage.status,native_message=stage.message,
                    terminal_gradient_l2=float(np.linalg.norm(ev.gradient_reduced)),terminal_gradient_inf=float(np.linalg.norm(ev.gradient_reduced,np.inf)),
                    iterations=result.iterations,evaluations=result.nfev,command_time_s=result.end_to_end_sec,
                    command_timing_scope='run_multitrap end-to-end; source session retained')
        stages=[asdict(s) for s in result.stages];config=config.to_payload()
    row=dict(spec,**native,phase_content_id=digest_array(phase),initial_phase_content_id=digest_array(initial),
             cache_status=status,cache_source=source,cap=cap,gtol=ctx.config.gtol,actual_force_target_z_n=float(target_force[2]) if compensated else 0.,config_json=json.dumps(config),
             stages_json=json.dumps(stages),evidence_status=evidence)
    (ctx.output_root/'commands'/f"{spec['run_id']}.json").write_text(json.dumps(row,indent=2))
    np.savez_compressed(ctx.output_root/'commands'/f"{spec['run_id']}.npz",phase_rad=phase,initial_phase_rad=initial,
                        targets_m=targets,positions_m=ctx.positions_m,normals=ctx.normals,config_json=json.dumps(config))
    print(f"C {spec['run_id']}: {status}; {row['iterations']} iterations; {row['native_message']}",flush=True)
    return dict(row=row,phase=phase,targets=targets)


def _correspondence(ctx,jobs):
    for task in ('Single','Triple'):
        selected=[j for j in jobs if j['row']['task']==task]
        for seed in sorted({j['row']['seed'] for j in selected}):
            paired=[j for j in selected if j['row']['seed']==seed]
            reference=next(j for j in paired if j['row']['ablation']=='GFE' and j['row']['alpha_factor']==1.)
            # Identical prescribed targets and identical surrounding volume for every candidate.
            lam=343./ctx.config.frequency_hz
            offsets=np.array(np.meshgrid(*([np.linspace(-.75*lam,.75*lam,9)]*3),indexing='ij')).reshape(3,-1).T
            points=np.concatenate([target+offsets for target in reference['targets']])
            reference_field=p._array_field(ctx,reference['phase']).pressure(points)
            field_directory=ctx.output_root/'fields';field_directory.mkdir(exist_ok=True)
            field_ref=np.asarray(reference_field).ravel()
            for job in paired:
                pressure=np.asarray(p._array_field(ctx,job['phase']).pressure(points)).ravel()
                similarity=float(projective_similarity(np.exp(1j*job['phase'])[None,:],np.exp(1j*reference['phase'])[None,:])[0,0])
                field_similarity=float(np.clip(abs(np.vdot(field_ref,pressure))/(np.linalg.norm(field_ref)*np.linalg.norm(pressure)),0.,1.))
                amp_similarity=float(np.clip(np.dot(abs(field_ref),abs(pressure))/(np.linalg.norm(field_ref)*np.linalg.norm(pressure)),0.,1.))
                np.savez_compressed(field_directory/f"{job['row']['run_id']}.npz", points_m=points, pressure_complex_pa=pressure, reference_pressure_complex_pa=field_ref, reference_phase_rad=reference['phase'], candidate_phase_rad=job['phase'])
                job['row'].update(reference_run_id=reference['row']['run_id'],reference_phase_content_id=reference['row']['phase_content_id'],
                                  command_cosine_to_fixed_GFE=similarity,field_cosine_to_fixed_GFE=field_similarity,
                                  pressure_amplitude_cosine_to_fixed_GFE=amp_similarity,command_projective_distance_to_fixed_GFE=float(np.sqrt(max(0.,1.-similarity**2))),field_point_count=len(points))


def _validate(root,ctx,jobs,workers):
    available=_source_validations(root,ctx)
    requests=[(job,index,target) for job in jobs for index,target in enumerate(job['targets'])]
    def one(request):
        job,index,target=request;row=job['row'];content_id=row['phase_content_id'];key=(content_id,tuple(target))
        saved=available.get(key)
        if saved:
            validation,source=saved;status='compatible frozen-command elastic cache'
        else:
            validation=p._finite_ka_validation(ctx,job['phase'],target,key=f"C-GFE-{row['run_id']}-T{index+1}")
            source='appendix_outputs/C_GFE/cache/finite_ka_static_validation';status='current elastic cache or new'
        mechanical_directory=ctx.output_root/'mechanics';mechanical_directory.mkdir(exist_ok=True)
        np.savez_compressed(mechanical_directory/f"{row['run_id']}_T{index+1}.npz", target_m=target, equilibrium_m=validation.equilibrium.equilibrium_m, displacement_m=validation.equilibrium.displacement_m, force_jacobian_n_m=validation.force_jacobian_n_m, symmetric_stiffness_n_m=validation.symmetric_stiffness_n_m, stiffness_eigenvalues_n_m=validation.symmetric_stiffness_eigenvalues_n_m, phase_rad=job['phase'], protocol_json=json.dumps(p._finite_ka_protocol(ctx),default=str))
        out=p._static_validation_row(row['method'],validation)
        out.update(run_id=row['run_id'],task=row['task'],seed=row['seed'],ablation=row['ablation'],alpha=row['alpha'],
                   alpha_factor=row['alpha_factor'],target_id=f'T{index+1}',target_x_m=target[0],target_y_m=target[1],target_z_m=target[2],
                   phase_content_id=content_id,validation_source=source,validation_cache_status=status,evidence_status=getattr(ctx,'evidence_status','SMOKE'))
        print(f"C elastic {row['run_id']} T{index+1}: d/a={out['finite_ka_displacement_a']:.5g}; root={out['finite_ka_root_found']}",flush=True)
        return out
    target_rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one,requests):
            target_rows.append(row);pd.DataFrame(target_rows).to_csv(ctx.output_root/'C_target_mechanics.csv',index=False)
    mechanics=pd.DataFrame(target_rows)
    for job in jobs:
        group=mechanics[mechanics.run_id.eq(job['row']['run_id'])]
        all_roots=bool(group.finite_ka_root_found.all())
        job['row'].update(all_roots=all_roots,all_locally_restoring=bool(group.finite_ka_locally_restoring.all()),
                          target_count=len(group),worst_displacement_a=float(group.finite_ka_displacement_a.max()) if all_roots else np.nan,
                          median_target_displacement_a=float(group.finite_ka_displacement_a.median()) if all_roots else np.nan,
                          minimum_stiffness_n_m=float(group.finite_ka_stiffness_min_n_m.min()) if all_roots else np.nan)
    return mechanics


def _style(ax):
    for spine in ax.spines.values():spine.set_linewidth(1.7)
    ax.tick_params(width=1.6,length=6,labelsize=14)
    ax.grid(alpha=.22,linewidth=.8)
    ax.set_axisbelow(True)


def _save(figure,output,name):
    paths=[]
    for ext in ('png','pdf','svg'):
        path=output/f'{name}.{ext}';figure.savefig(path,dpi=220,bbox_inches='tight',facecolor='white');paths.append(str(path))
    plt.close(figure);return paths


def render(output):
    output=Path(output);frame=pd.read_csv(output/'C_command_runs.csv')
    sensitivity=frame[frame.sensitivity.astype(bool)]
    settings={'font.size':15,'axes.labelsize':16,'axes.labelweight':'bold','axes.titleweight':'bold','axes.titlesize':17,
              'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':13,'lines.linewidth':2.8,
              'lines.markersize':8,'lines.markeredgewidth':1.8,'svg.fonttype':'none','pdf.fonttype':42}
    paths=[]
    coordinates=[]
    for row in sensitivity.itertuples():
        for metric in ('command_cosine_to_fixed_GFE','field_cosine_to_fixed_GFE','worst_displacement_a'):
            coordinates.append(dict(figure='C1',task=row.task,method=row.method,seed=row.seed,metric=metric,x=row.alpha_factor,y=getattr(row,metric)))
    pd.DataFrame(coordinates).to_csv(output/'C_plot_coordinates.csv',index=False)
    with plt.rc_context(settings):
        fig,axes=plt.subplots(2,2,figsize=(13.8,10.8));fig.subplots_adjust(left=.10,right=.98,top=.89,bottom=.13,hspace=.42,wspace=.31)
        fig.suptitle('C1  Force weight: field similarity and target error',fontweight='bold',fontsize=20,y=.975)
        for i,task in enumerate(('Single','Triple')):
            data=sensitivity[sensitivity.task.eq(task)]
            for method in ('GFE','FE'):
                group=data[data.method.eq(method)].sort_values('alpha_factor');color=COLORS[method]
                axes[i,0].plot(group.alpha_factor,group.command_cosine_to_fixed_GFE,color=color,marker='o',linestyle='-',lw=4.3 if method=='GFE' else 2.0,ms=10 if method=='GFE' else 6.5,mfc=color if method=='GFE' else 'white')
                axes[i,0].plot(group.alpha_factor,group.field_cosine_to_fixed_GFE,color=color,marker='s',markerfacecolor='white',linestyle='--',lw=3.2 if method=='GFE' else 1.8,ms=9 if method=='GFE' else 6.3)
                axes[i,1].plot(group.alpha_factor,group.worst_displacement_a,color=color,marker='o',linestyle='-' if method=='GFE' else '--',lw=4 if method=='GFE' else 2.1,ms=10 if method=='GFE' else 7,mfc=color if method=='GFE' else 'white')
            axes[i,0].set_title(f'({chr(97+2*i)}) {task}: correspondence',pad=12)
            axes[i,1].set_title(f'({chr(98+2*i)}) {task}: elastic equilibrium',pad=12)
            axes[i,0].set_ylabel('Similarity to fixed GFE')
            axes[i,1].set_ylabel(r'Displacement $d/a$' if task=='Single' else r'Worst target $d/a$')
            values=data[['command_cosine_to_fixed_GFE','field_cosine_to_fixed_GFE']].values
            axes[i,0].set_ylim(max(0,float(np.nanmin(values))-.045),1.008)
            axes[i,0].set_yticks([tick for tick in axes[i,0].get_yticks() if axes[i,0].get_ylim()[0]<=tick<=1.000001])
            for ax in axes[i]:
                _style(ax);ax.set_xscale('log');ax.set_xticks(ALPHA_FACTORS,['0.3','1','3']);ax.minorticks_off()
                ax.set_xlabel(r'$\alpha/\alpha_0$');ax.axvline(1,color='.65',lw=1.2,zorder=0)
        handles=[Line2D([],[],color=COLORS[m],lw=3,label=m) for m in ('GFE','FE')]
        handles += [Line2D([],[],color='.2',marker='o',lw=2.8,label='Command cosine'),
                    Line2D([],[],color='.2',marker='s',mfc='white',ls='--',lw=2.8,label='Field cosine')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.018),ncol=4,frameon=False)
        paths+=_save(fig,output,'Figure_C1_GFE_fixed_reference_sensitivity')

    return paths


def _write_text(output,commands,mechanics):
    sensitivity=commands[commands.sensitivity]
    reference=commands[commands.ablation.eq('GFE') & commands.alpha_factor.eq(1)]
    refs=[]
    for row in reference.itertuples():
        refs.append(dict(task=row.task,seed=row.seed,run_id=row.run_id,phase_content_id=row.phase_content_id,
            all_roots=row.all_roots,all_locally_restoring=row.all_locally_restoring,worst_displacement_a=row.worst_displacement_a,
            native_success=row.native_success,native_message=row.native_message))
    pd.DataFrame(refs).to_csv(output/'C_frozen_reference_validation.csv',index=False)
    summary=[]
    ablations=commands[commands.task.eq('Triple') & commands.alpha_factor.eq(1)]
    for label in ABLATIONS:
        group=ablations[ablations.ablation.eq(label)]
        summary.append(dict(configuration=label,paired_start_count=len(group),
            median_gradient_l2=group.terminal_gradient_l2.median(),
            median_worst_displacement_a=group.worst_displacement_a.median(),
            gradient_l2_q25=group.terminal_gradient_l2.quantile(.25),gradient_l2_q75=group.terminal_gradient_l2.quantile(.75),
            worst_displacement_a_q25=group.worst_displacement_a.quantile(.25),worst_displacement_a_q75=group.worst_displacement_a.quantile(.75),
            minimum_worst_displacement_a=group.worst_displacement_a.min(),maximum_worst_displacement_a=group.worst_displacement_a.max(),
            resolved_commands=int(group.all_roots.sum()),locally_restoring_commands=int(group.all_locally_restoring.sum()),
            median_command_cosine_to_fixed_GFE=group.command_cosine_to_fixed_GFE.median(),
            median_field_cosine_to_fixed_GFE=group.field_cosine_to_fixed_GFE.median(),
            evidence_status='SMOKE'))
    summary=pd.DataFrame(summary);summary.to_csv(output/'C_ablation_summary_historical_10000.csv',index=False)
    # Bootstrap paired command-level changes, never individual dependent targets.
    bootstrap=[];random=np.random.default_rng(20260905)
    for metric in ('terminal_gradient_l2','worst_displacement_a'):
        pivot=ablations.pivot(index='seed',columns='ablation',values=metric).reindex(TRIPLE_SEEDS)
        for label in ABLATIONS:
            delta=(pivot[label]-pivot['GFE']).dropna().to_numpy(float)
            if len(delta):
                resamples=np.median(delta[random.integers(0,len(delta),size=(10000,len(delta)))],axis=1)
                lo,hi=np.quantile(resamples,[.025,.975]);estimate=float(np.median(delta))
            else:lo=hi=estimate=np.nan
            bootstrap.append(dict(configuration=label,reference='GFE',metric=metric,paired_start_count=len(delta),
                median_paired_change=estimate,paired_percentile_95_lower=lo,paired_percentile_95_upper=hi,
                bootstrap_resamples=10000,bootstrap_seed=20260905,
                interval_scope='Exploratory paired start bootstrap; n=3; targets summarized within each command',evidence_status='SMOKE'))
    pd.DataFrame(bootstrap).to_csv(output/'C_paired_seed_bootstrap_historical_10000.csv',index=False)
    captions={
       'Figure_C1_GFE_fixed_reference_sensitivity':
       'Fixed-reference sensitivity of GFE and its uncompensated FE control. The nominal GFE command is frozen separately for Single (alpha0=10 m^-1, seed 172817509) and Triple (alpha0=3000 m^-1, seed 260869). Both methods are independently optimized from the same original random command at alpha/alpha0=0.3,1,3; no nearest endpoint is substituted after the sweep. Left: the main-text gauge-invariant command cosine |mean exp(i(delta phi))| and normalized complex-pressure cosine on a fixed 9x9x9 volume spanning +/-0.75 wavelength around each prescribed target. Right: displacement to the current finite-ka elastic total-force equilibrium normalized by bead radius a=0.65 mm, with the worst target reported for Triple. The reference is a validated nominal main command, not a claim of global optimality. Its root and restoring-status records are exported. Standard-sign Gor\'kov synthesis, fixed prescribed targets and a 10000-iteration cap are retained. One paired start per task; SMOKE.',
       'Figure_C2_GFE_multitrap_regularization_ablations':
       'Matched Triple GFE ablations. GFE uses the upward buoyancy-corrected weight as its surrogate radiation-force target. Individual columns remove force smoothing, pressure penalty or loss uniformity; FE is the same complete regularized formulation with zero force target. Alpha=3000 m^-1, beta=5e-5 N m^-1 Pa^-1, gamma=1 and W=diag(1,1,10) remain fixed. Each curve connects one identical random initialization (seeds 260869, 260870, 260871); dark diamonds show the median of three starts. (a) Actual terminal reduced-phase gradient L2. (b) Worst of the three target displacements to the independently resolved elastic total-force equilibrium, in bead radii. The native BFGS infinity-norm tolerance is 1e-8; the staged caps are 250 and 9750. Removing one component retains the original smoothing-scale rules, so realized stage epsilons are recorded. All target-level results and native stopping messages are exported. SMOKE.'}
    captions.pop('Figure_C2_GFE_multitrap_regularization_ablations',None)
    existing=json.loads((output/'figure_captions.json').read_text()) if (output/'figure_captions.json').exists() else {}
    existing.update(captions)
    (output/'figure_captions.json').write_text(json.dumps(existing,indent=2))
    lines=['Appendix C: objective sensitivity, regularization and initialization dependence.',
           'C1/C2 link weight changes and component removals to fixed-reference GFE correspondence and independent elastic displacement. C3 adds cold/warm initialization. C4/C5 complete the same sensitivity section with Twin/Bottle directional weights and the RH pressure term, using their separate frozen FE formulation references.',
           'GFE changes the force residual to the fixed upward effective-weight target. FE is the zero-target control. No prescribed-coordinate shift is used.']
    for task in ('Single','Triple'):
        for method in ('GFE','FE'):
            q=sensitivity[sensitivity.task.eq(task)&sensitivity.method.eq(method)]
            lines.append(f"{task} {method}: command cosine to frozen GFE spans {q.command_cosine_to_fixed_GFE.min():.6f} to {q.command_cosine_to_fixed_GFE.max():.6f}; complex-field cosine spans {q.field_cosine_to_fixed_GFE.min():.6f} to {q.field_cosine_to_fixed_GFE.max():.6f}; elastic d/a spans {q.worst_displacement_a.min():.5g} to {q.worst_displacement_a.max():.5g}.")
    lines.append('Current C2 component ablations and C6 STD sensitivity use the independent 30000-cap campaign in std_sensitivity/. Its C_STD_command_runs.csv, C_STD_summary.csv and measured_results_smoke.txt are authoritative. Earlier 10000-cap component rows are retained only as historical data in C_ablation_summary_historical_10000.csv and C_paired_seed_bootstrap_historical_10000.csv; they do not describe current Figure C2.')
    lines.append('These C1 and archived 10k results are SMOKE. Correspondence is measured against a fixed same-condition reference. Archived component bootstrap intervals resample the three paired starts, not their dependent targets. Current C2/C6 numerical summaries reside in std_sensitivity/. Native optimizer stops are retained, and cap-limited or precision-loss endpoints are not relabeled converged.')
    (output/'measured_results_smoke.txt').write_text('\n\n'.join(lines)+'\n')
    (output/'C_protocol.json').write_text(json.dumps(dict(schema=SCHEMA,alpha_factors=ALPHA_FACTORS,
        nominal_alpha={'Single':10.,'Triple':3000.},ablation_seeds=TRIPLE_SEEDS,ablation_labels=ABLATIONS,
        single_sensitivity_seed=SINGLE_SEED,triple_sensitivity_seed=TRIPLE_SEEDS[0],
        reference_rule='Frozen same-seed nominal GFE command; mechanically validated and not reselected during sweep',
        command_correspondence='main branch_figures.projective_similarity on exp(i phi)',
        complex_field_correspondence='abs(vdot(p_ref,p))/(norm(p_ref)*norm(p)); common 9x9x9 grid per target',
        field_volume_halfwidth_lambda=.75,targets_fixed=True,force_target_n=effective_weight_force_target_n(SphereInFluid()).tolist(),
        physics_source_content_id=_hash_sources(),dependencies=dependency_snapshot(),evidence_status='SMOKE'),indent=2))


def run(root,cache_only=False,workers=3):
    """Replay the C1 alpha study; keep previous 10k component rows archival.

    Current C2 and C6 are generated separately by appendix_c_std_gfe.run.
    """
    root=Path(root).resolve();output=root/'appendix_outputs/C_GFE';output.mkdir(parents=True,exist_ok=True)
    (output/'commands').mkdir(exist_ok=True)
    ctx=_context(root,output,cache_only)
    problem=make_problem(ctx,contracted_targets(.8),0.,0.,case_prefix='fixed-long-triple-observation')
    specs=specifications(ctx);source_index=_source_commands(root);jobs=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_solve_one,root,ctx,problem,source_index,spec) for spec in specs]
        for future in as_completed(futures):
            jobs.append(future.result());pd.DataFrame([j['row'] for j in jobs]).to_csv(output/'C_command_runs_partial.csv',index=False)
    jobs.sort(key=lambda j:next(i for i,spec in enumerate(specs) if spec['run_id']==j['row']['run_id']))
    _correspondence(ctx,jobs)
    mechanics=_validate(root,ctx,jobs,workers)
    for job in jobs:
        (output/'commands'/f"{job['row']['run_id']}.json").write_text(json.dumps(job['row'],indent=2))
    commands=pd.DataFrame([job['row'] for job in jobs]);commands['record_scope']=np.where(commands.sensitivity,'Current C1 alpha sensitivity','Historical C2 10000-cap component study');commands.to_csv(output/'C_command_runs.csv',index=False)
    commands[commands.sensitivity].to_csv(output/'C_fixed_reference_sensitivity.csv',index=False)
    _write_text(output,commands,mechanics)
    paths=render(output)
    (output/'C_elastic_protocol.json').write_text(json.dumps(p._finite_ka_protocol(ctx),indent=2))
    (output/'C_cache_access.json').write_text(json.dumps(ctx.cache.access_records,indent=2))
    return dict(appendix='C',output_dir=str(output),figures=[x for x in paths if x.endswith('.png')],figure_exports=paths,
        tables={name:str(output/(name+'.csv')) for name in ('C_command_runs','C_fixed_reference_sensitivity','C_ablation_summary_historical_10000','C_target_mechanics','C_frozen_reference_validation','C_plot_coordinates','C_paired_seed_bootstrap_historical_10000')},
        measured_results=str(output/'measured_results_smoke.txt'),evidence_status='SMOKE')

if __name__=='__main__':
    import sys
    print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else Path.cwd(),cache_only='--cache-only' in sys.argv),indent=2))
