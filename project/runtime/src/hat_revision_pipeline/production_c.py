"""Declared production C1/C2/C3/C6 populations; no work runs at import time.

C1 has 120 slots. C2/C6 have 100 shared Triple slots, including 20 nominal
FE/GFE slots shared with C1 and the main bank. C3 has 210 records, including
30 main nominal commands and 90 main Figure 7 cold commands. Every mechanical
record uses the passed full context, independently of command-cache provenance.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import pipeline as p
from . import appendix_c_feg as c
from . import appendix_c_std_gfe as std
from . import appendix_c_warm_gfe as warm
from .cache import CacheStore, canonical_json, digest_array, digest_file
from .gorkov_core import SingleTargetObjective
from .sota import solve_corrected_gorkov_fe

SCHEMA='gfe-production-C-populations-v1'
DISPLAY_SEED=260828
COMMAND_KINDS={'C_GFE_single','C_GFE_triple','C_STD_commands','C3_gfe_warm_run'}


class CommandReadThroughCache(CacheStore):
    """Reuse only exact command payloads; never import quick mechanics."""
    def __init__(self,root,legacy_roots,*,read_only=False):
        super().__init__(root,namespace=p.PIPELINE_IMPLEMENTATION,read_only=read_only)
        self.legacy=[CacheStore(path,namespace=p.PIPELINE_IMPLEMENTATION,read_only=True) for path in legacy_roots]
    def get_or_compute(self,kind,payload,producer,*,recompute=False):
        destination=self.path_for(kind,payload,create_parent=False)
        if not recompute and not destination.exists() and kind in COMMAND_KINDS:
            for source in self.legacy:
                path=source.path_for(kind,payload,create_parent=False)
                if path.exists():
                    value,_,_=source.get_or_compute(kind,payload,lambda:None)
                    self.access_records.append(dict(kind=kind,status='exact-compatible-command-reuse',path=str(path)))
                    return value,'exact-compatible-command-reuse',path
        return super().get_or_compute(kind,payload,producer,recompute=recompute)


def _settings(root,ctx):
    from production_settings import load_settings
    settings=load_settings(root,overrides=getattr(ctx,'memo',{}).get('production_settings',{}))
    if not ctx.config.full or ctx.config.exact_lmax!=8:
        raise ValueError('Production C requires the shared full-numerics context, lmax=8')
    if ctx.config.gtol!=settings['gtol']:
        raise ValueError('Production C solver tolerance differs from the frozen design')
    return settings


def design(settings):
    """Enumerate the exact population without synthesis or mechanics."""
    single=tuple(settings['sensitivity_single_seeds']);triple=tuple(settings['sensitivity_triple_seeds'])
    specs=[]
    for task,seeds,alpha0,cap in (('Single',single,10.,10000),('Triple',triple,3000.,30000)):
        for seed in seeds:
            for method in ('GFE','FE'):
                for factor in c.ALPHA_FACTORS:
                    spec=dict(task=task,seed=int(seed),method=method,alpha=alpha0*factor,alpha_factor=factor,
                        ablation=method,sensitivity=True,maxiter=cap)
                    spec['run_id']=f"{task}_{method}_alpha{spec['alpha']:g}_seed{seed}"
                    specs.append(spec)
    std_specs=std.specifications(triple)
    origins=tuple(settings['single_seeds'])
    if len(specs)!=120 or len(std_specs)!=100 or len(origins)!=30:
        raise ValueError('Production C design must contain 120 C1 slots, 100 C2/C6 slots and 30 C3 origins')
    return dict(c1=specs,std=std_specs,c3_seeds=origins,c1_slots=120,std_slots=100,
        shared_c1_std_slots=20,unique_c1_c2_c6_slots=200,c3_records=210,c3_new_warm_slots=90)


def _context(root,shared,section):
    root=Path(root).resolve();base=root/'production_outputs/appendices/C_GFE';output=base/section
    output.mkdir(parents=True,exist_ok=True);(output/'commands').mkdir(exist_ok=True)
    legacy=[root/'appendix_outputs/C_GFE/cache',root/'appendix_outputs/C_GFE/std_sensitivity/cache',root/'smoke_outputs/cache']
    cache=CommandReadThroughCache(output/'cache',legacy,read_only=shared.config.cache_only)
    validation_roots=[shared.cache.root,*[base/name/'cache' for name in ('alpha_sensitivity','std_sensitivity','warm_continuation')]]
    return SimpleNamespace(config=shared.config,positions_m=shared.positions_m,normals=shared.normals,
        output_root=output,data_root=shared.data_root,recompute=False,cache=cache,evidence_status='PRODUCTION',
        validation_source_roots=validation_roots,
        validation_source_namespaces=(p.PIPELINE_IMPLEMENTATION,shared.cache.namespace)),base


def _map_jobs(function,specs,output,workers):
    jobs=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(function,spec) for spec in specs]
        for future in as_completed(futures):
            jobs.append(future.result())
            pd.DataFrame([j['row'] for j in jobs]).to_csv(output/'command_progress.csv',index=False)
    return jobs


def _finish(root,ctx,jobs,workers,filename):
    mechanics=c._validate(root,ctx,jobs,workers)
    # The generic legacy helper has a display-only source label; make the full
    # protocol and actual production cache namespace explicit for every row.
    mechanics['evidence_status']='PRODUCTION'
    mechanics.loc[mechanics.validation_source.eq('appendix_outputs/C_GFE/cache/finite_ka_static_validation'),'validation_source']=str(ctx.cache.root/'finite_ka_static_validation')
    mechanics['production_protocol_json']=json.dumps(p._finite_ka_protocol(ctx),default=str)
    mechanics.to_csv(ctx.output_root/'C_target_mechanics.csv',index=False)
    frame=pd.DataFrame([j['row'] for j in jobs])
    frame['evidence_status']='PRODUCTION';frame['validation_protocol']='full elastic lmax8, 19 starts, 70 evaluations/start'
    frame.to_csv(ctx.output_root/filename,index=False)
    stages=[]
    for job in jobs:
        (ctx.output_root/'commands'/f"{job['row']['run_id']}.json").write_text(json.dumps(job['row'],indent=2))
        for stage in json.loads(job['row'].get('stages_json','[]')):
            stages.append(dict(run_id=job['row']['run_id'],seed=job['row']['seed'],**stage))
    if stages:pd.DataFrame(stages).to_csv(ctx.output_root/'solver_stages.csv',index=False)
    (ctx.output_root/'cache_access.json').write_text(json.dumps(ctx.cache.access_records,indent=2))
    return frame,mechanics


def _statistics(frame,groups,metrics,reference,output,stem,settings,reference_keys=('seed',)):
    """Summarize paired seeds, retaining missing/root-failure denominators."""
    random=np.random.default_rng(settings['bootstrap_seed']);summary=[];paired=[]
    for key,group in frame.groupby(groups,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        tags=dict(zip(groups,key));n_total=int(group.seed.nunique())
        for metric in metrics:
            values=pd.to_numeric(group[metric],errors='coerce').to_numpy(float)
            valid=values[np.isfinite(values)]
            summary.append(dict(**tags,metric=metric,n_origins=n_total,n_valid=len(valid),n_missing=len(values)-len(valid),
                median=float(np.median(valid)) if len(valid) else np.nan,
                q25=float(np.quantile(valid,.25)) if len(valid) else np.nan,
                q75=float(np.quantile(valid,.75)) if len(valid) else np.nan,
                minimum=float(np.min(valid)) if len(valid) else np.nan,maximum=float(np.max(valid)) if len(valid) else np.nan))
            keys=list(reference_keys)
            joined=group[keys+[metric]].merge(reference[keys+[metric]],on=keys,suffixes=('_candidate','_reference'),validate='many_to_one')
            delta=(joined[metric+'_candidate']-joined[metric+'_reference']).to_numpy(float)
            delta=delta[np.isfinite(delta)]
            if len(delta):
                boot=np.median(delta[random.integers(0,len(delta),(settings['bootstrap_resamples'],len(delta)))],axis=1)
                lo,hi=np.quantile(boot,[.025,.975]);estimate=float(np.median(delta))
            else:lo=hi=estimate=np.nan
            paired.append(dict(**tags,metric=metric,n_origins=n_total,n_valid_pairs=len(delta),median_paired_change=estimate,
                ci95_lower=lo,ci95_upper=hi,bootstrap_seed=settings['bootstrap_seed'],bootstrap_resamples=settings['bootstrap_resamples'],
                sampling_unit='paired initialization; target outcomes summarized within a Triple command'))
    stats=pd.DataFrame(summary);stats.to_csv(output/(stem+'_population_summary.csv'),index=False)
    pd.DataFrame(paired).to_csv(output/(stem+'_paired_bootstrap.csv'),index=False)
    return stats


def _native_counts(frame,groups,output,name):
    success='native_success' if 'native_success' in frame else 'optimizer_success'
    root='all_roots' if 'all_roots' in frame else 'finite_ka_root_found'
    restoring='all_locally_restoring' if 'all_locally_restoring' in frame else 'finite_ka_locally_restoring'
    result=frame.groupby(groups,dropna=False).agg(origins=('seed','nunique'),records=('seed','size'),
        native_success=(success,'sum'),roots_resolved=(root,'sum'),locally_restoring=(restoring,'sum')).reset_index()
    result.to_csv(output/name,index=False)
    return result


def _caption(base,name,text):
    path=base/'figure_captions.json';value=json.loads(path.read_text()) if path.exists() else {}
    value[name]=text;path.write_text(json.dumps(value,indent=2))


def _render_c1(base,frame):
    coordinates=[]
    with plt.rc_context({'font.size':13,'axes.labelsize':14,'axes.labelweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,2,figsize=(13.8,10.7));fig.subplots_adjust(left=.11,right=.98,top=.94,bottom=.14,hspace=.40,wspace=.30)
        for i,task in enumerate(('Single','Triple')):
            data=frame[frame.task.eq(task)]
            for method in ('GFE','FE'):
                q=data[data.method.eq(method)];color=c.COLORS[method]
                for ax,metric,marker,ls in ((axes[i,0],'command_cosine_to_fixed_GFE','o','-'),
                    (axes[i,0],'field_cosine_to_fixed_GFE','s','--'),(axes[i,1],'worst_displacement_a','o','-')):
                    grouped=q.groupby('alpha_factor')[metric]
                    for seed,small in q.groupby('seed'):
                        vals=small.set_index('alpha_factor')[metric].reindex(c.ALPHA_FACTORS)
                        ax.plot(range(3),vals,color=color,lw=.8,alpha=.13)
                    med=grouped.median().reindex(c.ALPHA_FACTORS)
                    ax.fill_between(range(3),grouped.quantile(.25).reindex(c.ALPHA_FACTORS).to_numpy(),grouped.quantile(.75).reindex(c.ALPHA_FACTORS).to_numpy(),color=color,alpha=.12)
                    ax.plot(range(3),med,color=color,marker=marker,ls=ls,ms=6.5,lw=2.4,mfc='white' if marker=='s' else color,mew=1.4)
                    for row in q.itertuples():coordinates.append(dict(task=task,method=method,seed=row.seed,metric=metric,x=row.alpha_factor,y=getattr(row,metric)))
            for j,ax in enumerate(axes[i]):
                c._style(ax);ax.set_xticks(range(3),['0.3','1','3']);ax.set_xlabel(r'$\alpha/\alpha_0$')
                ax.set_title(f'({chr(97+2*i+j)}) {task}',loc='left',pad=12)
            axes[i,0].set_ylabel('Similarity to fixed GFE');axes[i,0].set_ylim(0,1.035)
            axes[i,1].set_ylabel(r'Displacement $d/a$' if task=='Single' else r'Worst target $d/a$')
        handles=[Line2D([],[],color=c.COLORS[name],lw=2.5,label=name) for name in ('GFE','FE')]
        handles += [Line2D([],[],color='.25',marker='o',label='Command'),Line2D([],[],color='.25',marker='s',mfc='white',ls='--',label='Complex field')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.025),ncol=4,frameon=False)
        paths=c._save(fig,base,'Figure_C1_GFE_fixed_reference_sensitivity')
    pd.DataFrame(coordinates).to_csv(base/'alpha_sensitivity/C1_plot_coordinates.csv',index=False)
    return paths


def run_c1(root,ctx,workers=None):
    from . import production_main as main
    root=Path(root).resolve();settings=_settings(root,ctx);plan=design(settings)
    local,base=_context(root,ctx,'alpha_sensitivity');workers=workers or settings['workers']
    problem=main.triple_problem(ctx);sources=c._source_commands(root)
    # Shared nominal bank calls are serial; benchmark timing is owned by main.
    provided={}
    for task,seeds in (('Single',settings['sensitivity_single_seeds']),('Triple',settings['sensitivity_triple_seeds'])):
        for seed in seeds:
            for method in ('GFE','FE'):
                provided[(task,seed,method)]=(main.ensure_single_run(ctx,method,seed) if task=='Single' else main.ensure_triple_run(ctx,method,seed))
    def one(spec):
        shared=provided.get((spec['task'],spec['seed'],spec['method'])) if spec['alpha_factor']==1 else None
        if shared is not None and spec['task']=='Single':
            expected=replace(c.fixed_single_fe_method_spec(),alpha_per_m=spec['alpha'])
            if canonical_json(asdict(shared['objective'].method))!=canonical_json(asdict(expected)):
                raise ValueError('Main Single nominal objective does not match the C1 reference')
        return c._solve_one(root,local,problem,sources,spec,provided=shared)
    jobs=_map_jobs(one,plan['c1'],local.output_root,workers)
    c._correspondence(local,jobs)
    frame,mechanics=_finish(root,local,jobs,workers,'C1_command_runs.csv')
    if len(frame)!=120 or len(mechanics)!=240:raise RuntimeError('Incomplete production C1 population')
    reference=frame[frame.method.eq('GFE')&frame.alpha_factor.eq(1)]
    reference.to_csv(local.output_root/'C1_frozen_references.csv',index=False)
    metrics=['command_cosine_to_fixed_GFE','field_cosine_to_fixed_GFE','pressure_amplitude_cosine_to_fixed_GFE','worst_displacement_a','terminal_gradient_l2']
    _statistics(frame,['task','method','alpha_factor'],metrics,reference,local.output_root,'C1',settings,('task','seed'))
    _native_counts(frame,['task','method','alpha_factor'],local.output_root,'C1_outcome_counts.csv')
    paths=_render_c1(base,frame)
    caption=('C1. Ten paired initializations per task, two methods and alpha/alpha0=0.3,1,3 (120 command records; 240 target-level validations). '
        'Each seed uses its own frozen nominal GFE reference; alpha0=10 m^-1 for Single and 3000 m^-1 for Triple. '
        'Lines show medians, bands IQRs and faint curves individual paired starts. Correspondence uses gauge-invariant command cosine and complex-pressure cosine on 729 points per target. '
        'Right panels show full elastic equilibrium displacement; Triple uses the worst of its three dependent targets. '
        'Single retains 10000 iterations; all Triple solves have 30000 ceilings (250+29750 stages), with native earlier stops retained. '
        'Nominal FE/GFE commands are shared with the main bank and C2/C6. Missing mechanical outcomes stay in outcome denominators and are excluded only from numeric quantiles. '
        'All raw commands, target mechanics, median/IQR and 10000 paired-bootstrap resamples are exported; full lmax=8 validation with 19 root starts is independent of command-cache provenance.')
    _caption(base,'Figure_C1_GFE_fixed_reference_sensitivity',caption)
    (local.output_root/'protocol.json').write_text(json.dumps(dict(schema=SCHEMA,section='C1',settings=settings,command_count=120,target_records=240,
        reference_rule='same-seed nominal GFE fixed before alpha variation',mechanical_protocol=p._finite_ka_protocol(local)),indent=2))
    return dict(figures=paths,table=str(local.output_root/'C1_command_runs.csv'),output_dir=str(local.output_root))


def run_std(root,ctx,workers=None,render_figures=True):
    from . import production_main as main
    root=Path(root).resolve();settings=_settings(root,ctx);plan=design(settings)
    local,base=_context(root,ctx,'std_sensitivity');workers=workers or settings['workers'];problem=main.triple_problem(ctx)
    provided={(seed,method):main.ensure_triple_run(ctx,method,seed) for seed in settings['sensitivity_triple_seeds'] for method in ('GFE','FE')}
    def one(spec):
        shared=provided.get((spec['seed'],'FE' if spec['ablation']=='FE' else 'GFE')) if spec['nominal'] or spec['ablation']=='FE' else None
        return std._solve(root,local,problem,spec,provided=shared)
    jobs=_map_jobs(one,plan['std'],local.output_root,workers)
    std._reference_correspondence(local,jobs)
    frame,mechanics=_finish(root,local,jobs,workers,'C_STD_command_runs.csv')
    if len(frame)!=100 or len(mechanics)!=300:raise RuntimeError('Incomplete production C2/C6 population')
    reference=frame[frame.nominal];reference.to_csv(local.output_root/'C_STD_frozen_references.csv',index=False)
    metrics=['command_cosine_to_fixed_GFE','field_cosine_to_fixed_GFE','pressure_amplitude_cosine_to_fixed_GFE','worst_displacement_a','common_nominal_loss_std_n_m','minimum_stiffness_n_m','terminal_gradient_l2']
    _statistics(frame,['sweep','ablation','gamma_uniformity','std_scale_multiplier'],metrics,reference,local.output_root,'C_STD',settings)
    _native_counts(frame,['sweep','ablation','gamma_uniformity','std_scale_multiplier'],local.output_root,'C_STD_outcome_counts.csv')
    ablations=frame[frame.nominal|frame.sweep.eq('ablation')];ablations.to_csv(local.output_root/'C2_command_runs.csv',index=False)
    sweep=frame[frame.ablation.eq('GFE')];sweep.to_csv(local.output_root/'C6_command_runs.csv',index=False)
    if len(ablations)!=50 or len(sweep)!=60:raise RuntimeError('Incomplete C2/C6 cases')
    for seed,group in sweep.groupby('seed'):
        if group.terminal_force_epsilon_n.nunique()!=1:
            raise RuntimeError(f'STD sensitivity changed force smoothing for seed {seed}')
    losses=[]
    for job in jobs:
        with np.load(local.output_root/'commands'/f"{job['row']['run_id']}.npz",allow_pickle=False) as arrays:
            for i,(native,common) in enumerate(zip(arrays['local_losses_n_m'],arrays['common_nominal_local_losses_n_m'])):
                losses.append(dict(run_id=job['row']['run_id'],seed=job['row']['seed'],target_id=f'T{i+1}',native_local_loss_n_m=native,common_nominal_local_loss_n_m=common))
    pd.DataFrame(losses).to_csv(local.output_root/'C_STD_target_losses.csv',index=False)
    paths=std.render(root,output_dir=base) if render_figures else []
    common=('Ten paired Triple seeds; 100 unique commands and 300 dependent target records across C2/C6. '
        'The same-seed nominal GFE reference is frozen before every variation. Native stops at the 30000 ceiling or earlier are retained. '
        'All mechanical results use full elastic numerics; compatible 30k command caches may be reused without reusing quick mechanics. '
        'Thin curves retain individual starts, diamonds show medians and bands show IQRs. Complete paired-bootstrap, outcome and source tables are exported. ')
    _caption(base,'Figure_C2_GFE_multitrap_regularization_ablations','C2. '+common+
        'Five ablations × 10 seeds give 50 records. Panels show complex-field cosine, worst-target elastic displacement, local-loss STD evaluated under one fixed nominal loss, and the native objective gradient. '
        'The STD term balances local objective losses rather than displacement. The dotted 1e-3 line is the L2 reporting threshold, distinct from L-BFGS-B infinity-norm gtol=1e-8. '
        'The original initialization-based smoothing-scale rule is retained for component removals; all actual stage epsilons are exported.')
    _caption(base,'Figure_C6_GFE_STD_weight_smoothing_sensitivity','C6. '+common+
        'Six settings × 10 seeds give 60 records. gamma_u=0.3,1,3 varies the uniformity weight; epsilon_STD multipliers 0,0.1,1,10 vary only STD smoothing. '
        'Zero retains an unsmoothed STD penalty and differs from removing the term. Left panels show fixed-reference complex-field cosine; upper right shows common-loss STD and lower right the native terminal gradient. '
        'J=mean(ell_i)+gamma_u*(sqrt(var_pop(ell_i)+epsilon_STD^2)-epsilon_STD). The force epsilon stays fixed across these sweeps. '
        'Nominal epsilon_STD stages are (0.05,0.01)*max(initial loss STD,1e-20); the floor and actual values are exported. Exact displacement and stiffness remain in the tables.')
    protocol=dict(schema=SCHEMA,section='C2/C6',settings=settings,command_count=100,target_records=300,shared_c1_slots=20,
        gamma_values=std.GAMMA_VALUES,std_multipliers=std.STD_FACTORS,stage_caps=std.CAPS,variance_ddof=0,
        stage_factors=[.05,.01],initial_scale_floor=1e-20,
        reference_rule='frozen same-seed complete GFE command',mechanical_protocol=p._finite_ka_protocol(local))
    (local.output_root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (base/'C2_active_manifest.json').write_text(json.dumps(dict(iteration_cap=30000,paired_seeds=10,command_table='std_sensitivity/C2_command_runs.csv',mechanics_table='std_sensitivity/C_target_mechanics.csv'),indent=2))
    return dict(figures=paths,table=str(local.output_root/'C_STD_command_runs.csv'),output_dir=str(local.output_root))


def _c3_jobs(root,shared,local,seed):
    from . import production_main as main
    nominal=main.ensure_single_run(shared,'GFE',seed);cold=main.ensure_alpha_bank(shared,seed)
    initial=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(shared.positions_m))
    objective=nominal['objective'];reference=np.asarray(nominal['phase']).copy()
    jobs=[dict(run=nominal['run'],initial=initial,row=warm._row(nominal['run'],'Reference',10.,seed,initial,nominal['row']['cache_path'],'shared main nominal'))]
    for i,alpha in enumerate(warm.ALPHAS):
        row=cold['table'].iloc[i]
        if row.initial_phase_content_id!=digest_array(initial):raise ValueError('Main7 cold origin differs from C3')
        jobs.append(dict(run=cold['runs'][i],initial=initial,row=warm._row(cold['runs'][i],'Cold',alpha,seed,initial,row.cache_path,'shared Main7 cold command')))
    hashes={name:digest_file(Path(c.__file__).parent/name) for name in ('gorkov_core.py','sota.py','exact_validator.py')}
    previous=10.;phase0=reference.copy()
    for alpha in warm.ALPHAS:
        method=replace(objective.method,alpha_per_m=alpha)
        current=SingleTargetObjective(objective.condition,method,force_target_n=objective.force_target_n)
        payload=dict(contract=warm.SCHEMA,method=asdict(method),force_target_n=current.force_target_n.tolist(),
            target_m=p.MAIN_TARGET_M.tolist(),positions=digest_array(shared.positions_m),normals=digest_array(shared.normals),
            frequency_hz=shared.config.frequency_hz,seed=seed,original_initial_content_id=digest_array(initial),
            initial_phase_content_id=digest_array(phase0),previous_alpha=previous,reference_phase_content_id=digest_array(reference),
            maxiter=10000,gtol=shared.config.gtol,source_content_id=hashes)
        start=phase0.copy()
        result,status,path=local.cache.get_or_compute('C3_gfe_warm_run',payload,
            lambda current=current,start=start:solve_corrected_gorkov_fe(current,start,maxiter=10000,gtol=shared.config.gtol))
        row=warm._row(result,'Warm',alpha,seed,start,str(path),status);row['previous_alpha_per_m']=previous
        jobs.append(dict(run=result,initial=start,row=row));phase0=np.asarray(result.phase_rad).copy();previous=alpha
    lam=343./shared.config.frequency_hz
    offsets=np.array(np.meshgrid(*([np.linspace(-.75*lam,.75*lam,9)]*3),indexing='ij')).reshape(3,-1).T
    points=p.MAIN_TARGET_M+offsets;reference_pressure=p._array_field(shared,reference).pressure(points)
    for job in jobs:
        phase=np.asarray(job['run'].phase_rad);pressure=p._array_field(shared,phase).pressure(points)
        row=job['row'];row.update(evidence_status='PRODUCTION',reference_phase_content_id=digest_array(reference),reference_seed=seed,
            reference_alpha_per_m=10.,field_point_count=729,
            command_cosine_to_fixed_GFE=float(np.clip(abs(np.mean(np.exp(1j*(phase-reference)))),0,1)),
            field_cosine_to_fixed_GFE=float(np.clip(abs(np.vdot(reference_pressure,pressure))/(np.linalg.norm(reference_pressure)*np.linalg.norm(pressure)),0,1)))
        row.update(main.pressure_descriptor(shared,phase))
        job['array_name']=warm._save_arrays(local.output_root,job,reference,points,pressure,reference_pressure,include_seed=True)
    return jobs


def _render_c3(base,frame,fields):
    # Preserve the accepted 2x2 layout and fixed display command; statistics above
    # it now aggregate the complete 30-origin population.
    with plt.rc_context({'font.size':12,'axes.labelsize':14,'axes.labelweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,2,figsize=(11.4,10.6));colors={'Cold':'#3B6FA1','Warm':'#007C78'}
        reference=frame[frame.initialization.eq('Reference')]
        for mode in ('Cold','Warm'):
            q=frame[frame.initialization.eq(mode)];color=colors[mode]
            for j,metric,marker,ls in ((0,'command_cosine_to_fixed_GFE','o','-'),(0,'field_cosine_to_fixed_GFE','s','--'),(1,'finite_ka_displacement_a','o','-')):
                grouped=q.groupby('alpha_per_m')[metric]
                ref=reference[metric].median();med=np.r_[ref,grouped.median().reindex(warm.ALPHAS).to_numpy()]
                lower=np.r_[reference[metric].quantile(.25),grouped.quantile(.25).reindex(warm.ALPHAS).to_numpy()]
                upper=np.r_[reference[metric].quantile(.75),grouped.quantile(.75).reindex(warm.ALPHAS).to_numpy()]
                axes[0,j].fill_between(range(4),lower,upper,color=color,alpha=.15)
                axes[0,j].plot(range(4),med,color=color,marker=marker,ls=ls,lw=2.3,ms=6,mfc='white' if marker=='s' else color,mew=1.5)
        for j,ax in enumerate(axes[0]):
            c._style(ax);ax.set_xticks(range(4),['10',r'$10^3$',r'$10^6$',r'$10^7$'])
            ax.set_xlabel(r'Force weight $\alpha$ (m$^{-1}$)');ax.set_title(f'({chr(97+j)})',loc='left',pad=12)
        axes[0,0].set_ylim(0,1.035);axes[0,0].set_ylabel('Similarity to nominal GFE');axes[0,1].set_ylabel(r'Elastic displacement / $a$')
        nominal=fields[('Reference',10.)];axis=nominal['axis_m']*1000
        scale=max(float(abs(fields[k]['pressure_complex_pa']).max()) for k in [('Reference',10.),('Cold',1e7),('Warm',1e7)])
        for j,mode in enumerate(('Cold','Warm')):
            ax=axes[1,j];image=abs(fields[(mode,1e7)]['pressure_complex_pa'])/scale
            im=ax.imshow(image,origin='lower',extent=[axis[0],axis[-1],axis[0],axis[-1]],cmap='magma',vmin=0,vmax=1)
            ax.contour(axis,axis,abs(nominal['pressure_complex_pa'])/scale,levels=[.25,.5,.75],colors='#71DFF4',linestyles='--',linewidths=1.3)
            ax.plot(0,0,'+',color='white',ms=8,mew=1.6);ax.set_aspect('equal')
            ax.set_xticks([-6,0,6]);ax.set_yticks([-6,0,6]);ax.set_xlabel(r'$x-x_0$ (mm)');ax.set_ylabel(r'$y-y_0$ (mm)')
            ax.set_title(f'({chr(99+j)}) {mode}',loc='left',pad=10)
        handles=[Line2D([],[],color=colors[mode],lw=2.4,label=mode) for mode in ('Cold','Warm')]
        handles += [Line2D([],[],color='.25',marker='o',label='Command'),Line2D([],[],color='.25',marker='s',mfc='white',ls='--',label='Complex field')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.565),ncol=4,frameon=False,fontsize=10)
        fig.subplots_adjust(left=.11,right=.98,top=.95,bottom=.18,hspace=.57,wspace=.35)
        cb=fig.add_axes([.32,.045,.4,.014]);fig.colorbar(im,cax=cb,orientation='horizontal',ticks=[0,.5,1]).set_label('Pressure / shared peak')
        fig.legend(handles=[Line2D([],[],color='#169AB7',ls='--',lw=1.4,label='Nominal GFE contours')],loc='lower center',bbox_to_anchor=(.55,.084),frameon=False)
        return c._save(fig,base,'Figure_C3_GFE_warm_continuation')


def run_c3(root,ctx,workers=None):
    from . import production_main as main
    root=Path(root).resolve();settings=_settings(root,ctx);plan=design(settings)
    local,base=_context(root,ctx,'warm_continuation');workers=workers or settings['workers']
    # Ensure shared nominal/cold runs serially before parallel warm continuation.
    for seed in plan['c3_seeds']:
        main.ensure_single_run(ctx,'GFE',seed);main.ensure_alpha_bank(ctx,seed)
    jobs=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_c3_jobs,root,ctx,local,seed) for seed in plan['c3_seeds']]
        for future in as_completed(futures):
            jobs.extend(future.result());pd.DataFrame([j['row'] for j in jobs]).to_csv(local.output_root/'C3_command_progress.csv',index=False)
    if len(jobs)!=210:raise RuntimeError('C3 production requires 210 records')
    existing=c._source_validations(root,local)
    def validate(job):
        phase=np.asarray(job['run'].phase_rad);key=(digest_array(phase),tuple(p.MAIN_TARGET_M))
        if key in existing:result,source=existing[key]
        else:
            result=p._finite_ka_validation(local,phase,p.MAIN_TARGET_M,key=job['array_name']);source=str(local.cache.root)
        row=p._static_validation_row('GFE',result);row.update(job['row'],validation_source=source)
        np.savez_compressed(local.output_root/(job['array_name']+'_mechanics.npz'),phase_rad=phase,target_m=p.MAIN_TARGET_M,
            equilibrium_m=result.equilibrium.equilibrium_m,force_jacobian_n_m=result.force_jacobian_n_m,
            stiffness_eigenvalues_n_m=result.symmetric_stiffness_eigenvalues_n_m,protocol_json=json.dumps(p._finite_ka_protocol(local)))
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(validate,jobs))
    frame=pd.DataFrame(rows).sort_values(['seed','initialization','alpha_per_m']);frame.to_csv(local.output_root/'C3_warm_cold_results.csv',index=False)
    reference=frame[frame.initialization.eq('Reference')];reference.to_csv(local.output_root/'C3_fixed_references.csv',index=False)
    metrics=['command_cosine_to_fixed_GFE','field_cosine_to_fixed_GFE','finite_ka_displacement_a','terminal_gradient_l2','twofold_amplitude_modulation','phase_winding','ring_radius_mm']
    _statistics(frame,['initialization','alpha_per_m'],metrics,reference,local.output_root,'C3',settings)
    counts=_native_counts(frame,['initialization','alpha_per_m'],local.output_root,'C3_outcome_counts.csv')
    ecdf=[]
    for (mode,alpha),q in frame.groupby(['initialization','alpha_per_m']):
        for metric in ('twofold_amplitude_modulation','phase_winding','field_cosine_to_fixed_GFE'):
            values=np.sort(q[metric].dropna().to_numpy(float))
            for threshold in np.unique(values):ecdf.append(dict(initialization=mode,alpha_per_m=alpha,metric=metric,threshold=threshold,count_at_or_below=int((values<=threshold).sum()),valid_origins=len(values),total_origins=len(q),fraction_at_or_below=float(np.mean(values<=threshold))))
    pd.DataFrame(ecdf).to_csv(local.output_root/'C3_morphology_empirical_distributions.csv',index=False)
    selected=[j for j in jobs if j['row']['seed']==DISPLAY_SEED]
    if len(selected)!=7:raise RuntimeError('Fixed C3 illustration seed is absent')
    fields,morphology=warm._display_fields(local,local.output_root,selected,cache_only=local.config.cache_only)
    paths=_render_c3(base,frame,fields)
    frame[['seed','initialization','alpha_per_m',*metrics]].to_csv(local.output_root/'C3_plot_source_data.csv',index=False)
    caption=('C3. Thirty matched origins and 210 records: 30 nominal, 90 cold and 90 sequential warm commands. Nominal commands reuse the main bank and cold commands reuse Figure 7. '
        'Warm paths follow 10→1000→1e6→1e7 m^-1 from each preceding terminal phase; cold commands always use the original same-seed random phase. '
        'Top panels aggregate all 30 origins with medians/IQRs, against independently frozen same-seed nominal GFE references. '
        'Bottom panels preserve the deposited display seed 260828 and alpha=1e7 cold/warm fields, with nominal pressure contours and one shared scale. '
        'All solves retain the 10000 ceiling and actual native stops; high-weight precision loss is not recast as converged local minima. '
        'Independent full elastic mechanics uses lmax=8 and 19 root starts. Missing roots remain in exported outcome denominators. '
        'Twofold amplitude modulation, phase winding, reference-field correspondence and paired changes are exported for all 210 records; empirical distributions remain continuous without imposing a binary Twin classifier. '
        'The morphology and mechanics of the fixed illustrations are distinct from the paired population summaries.')
    _caption(base,'Figure_C3_GFE_warm_continuation',caption)
    (local.output_root/'C3_figure_caption.txt').write_text(caption+'\n')
    (local.output_root/'protocol.json').write_text(json.dumps(dict(schema=SCHEMA,section='C3',settings=settings,origins=30,records=210,
        shared_nominal=30,shared_cold=90,warm_new_slots=90,alpha_sequence=[10,*warm.ALPHAS],display_seed=DISPLAY_SEED,
        reference_rule='freeze each nominal before cold/warm comparison',maxiter=10000,field_points_per_command=729,
        morphology_grid=[80,256],display_grid=[151,151],mechanical_protocol=p._finite_ka_protocol(local)),indent=2))
    (local.output_root/'cache_access.json').write_text(json.dumps(local.cache.access_records,indent=2))
    return dict(figures=paths,table=str(local.output_root/'C3_warm_cold_results.csv'),output_dir=str(local.output_root))


def run(root,ctx,workers=None):
    """Explicit opt-in execution of the declared C populations."""
    results=[run_c1(root,ctx,workers),run_std(root,ctx,workers),run_c3(root,ctx,workers)]
    return dict(figures=[path for result in results for path in result['figures']],sections=results)
