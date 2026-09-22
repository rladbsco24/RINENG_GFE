"""Executable production B/C4/C5 campaign with shared frozen references.

This producer reuses compatible deposited commands and solves only missing
scientific configurations when explicitly run. Import and specification building
do not execute optimizers or mechanical validators.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from dataclasses import replace

from rineng_content_id import content_identity
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from . import pipeline as p
from .cache import digest_array,digest_file

SCHEMA='production-BC45-shared-250-commands-v1'
B10=tuple(range(260905,260915))
C4_CASES=(("corrected_minus","Twin"),("historical_plus","Bottle"))
METHODS=('FE','RH','Conventional')
ETAS=(.001,.01,.1)
BETAS=(0.,.1,1.,10.)


def _modules(root):
    modules=[]
    for name in ('sensitivity_b','render_b12'):
        path=root/'appendix_B_pressure'/(name+'.py')
        spec=importlib.util.spec_from_file_location('production_'+name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    return modules


def _configuration(module,shape,sign,method,eta=.01,beta=1.,*,seed):
    config=module.configuration(shape,sign,method,eta,beta,seed=seed)
    return dict(config,version=SCHEMA,status='PRODUCTION')


def specifications(root):
    """250 unique commands; C4/C5 share200, B1 adds20 and B2 adds30."""
    s,_=_modules(Path(root));configs={};uses=[]
    def add(section,seed,shape,sign,method,eta=.01,beta=1.):
        config=_configuration(s,shape,sign,method,eta,beta,seed=seed)
        key=s.cache_key(config);configs[key]=config
        uses.append(dict(section=section,cache_key=key,seed=seed,requested=shape,sign=sign,method=method,eta=eta,beta_factor=beta))
    for seed in B10:
        for sign,shape in C4_CASES:
            for method in METHODS:
                for eta in ETAS:add('C4',seed,shape,sign,method,eta)
        for beta in BETAS:
            add('C5',seed,'Bottle','historical_plus','FE' if beta==0 else 'RH',beta=beta)
        for method in ('FE','Conventional'):add('B1',seed,'Bottle','corrected_minus',method)
        for shape in ('Twin','Bottle'):
            for method in METHODS:add('B2',seed,shape,'historical_plus',method)
    # beta is irrelevant for FE: zero and default values must use one key.
    assert len(configs)==250
    table=pd.DataFrame(uses)
    assert table.groupby('section').size().to_dict()=={'B1':20,'B2':60,'C4':180,'C5':40}
    assert table[table.section.isin(('C4','C5'))].cache_key.nunique()==200
    return configs,table


def _import_or_solve(config,legacy,output,module,cache_only):
    key=module.cache_key(config);base=output/'data/sensitivity_cache'/key
    if base.with_suffix('.json').exists() and base.with_suffix('.npz').exists():
        row,arrays=module.load_run(config,output)
        return key,row,arrays
    for directory in (legacy/'data/sensitivity_cache',legacy/'data/cache'):
        for source in sorted(directory.glob('*.json')):
            try:row=json.loads(source.read_text())
            except (ValueError,OSError):continue
            if not source.with_suffix('.npz').exists() or not module.compatible(config,row):continue
            with np.load(source.with_suffix('.npz'),allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
            if content_identity(arrays['terminal_phase_rad'].tobytes()).hexdigest()!=row['phase_content_id']:raise ValueError('Invalid source phase')
            row=module.save_run(config,{**row,'cache_origin':'compatible immutable deposited command','source_cache_path':str(source.relative_to(legacy.parent))},arrays,output)
            return key,*module.load_run(config,output)
    if cache_only:raise FileNotFoundError(f'Production command missing: {key}')
    module.solve_one(config,output)
    return key,*module.load_run(config,output)


def _pressure(config,arrays,output,cache_only):
    phase=arrays['terminal_phase_rad'];axis=np.linspace(-1.25*343./40000.,1.25*343./40000.,181)
    stamp=dict(schema=SCHEMA,phase_content_id=digest_array(phase),pressure_points=181,half_width_m=float(axis[-1]),
        target_m=config['target_m'],frequency_hz=config['frequency_hz'],source_hash=config['source_content_id']['gorkov_core.py'])
    key=content_identity(json.dumps(stamp,sort_keys=True).encode()).hexdigest()
    path=output/'data/pressure_cache'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
    if cache_only:raise FileNotFoundError(f'Production pressure cache missing: {path}')
    from .gorkov_core import ArrayGeometry,pressure_field
    geometry=ArrayGeometry.square(16,.01);u,v=np.meshgrid(axis,axis,indexing='xy')
    result=dict(axis_m=axis,phase_rad=phase,metadata_json=np.asarray(json.dumps(stamp)))
    for plane in ('XY','XZ'):
        points=np.stack([u,v,np.full_like(u,.05)],axis=-1) if plane=='XY' else np.stack([u,np.zeros_like(u),v+.05],axis=-1)
        flat=points.reshape(-1,3)
        result[plane+'_points_m']=points
        result[plane]=np.concatenate([pressure_field(phase,flat[i:i+1024],geometry,40000.) for i in range(0,len(flat),1024)]).reshape(u.shape)
    np.savez_compressed(path,**result)
    return result


def _summary(frame,groups,metrics):
    random=np.random.default_rng(20260906);out=[]
    for identity,subset in frame.groupby(groups,dropna=False):
        if not isinstance(identity,tuple):identity=(identity,)
        for metric in metrics:
            vals=subset[metric].dropna().to_numpy(float)
            if len(vals):
                median=float(np.median(vals));minimum=float(np.min(vals));maximum=float(np.max(vals))
            else:
                median=minimum=maximum=np.nan
            if len(vals)>=2:
                low,high=np.quantile(vals,[.25,.75])
                boot=np.median(vals[random.integers(0,len(vals),size=(10000,len(vals)))],axis=1);ci=np.quantile(boot,[.025,.975])
                status='estimated';performed=10000
            else:
                low=high=np.nan;ci=[np.nan,np.nan]
                status='insufficient_independent_replicates' if len(vals) else 'no_finite_values';performed=0
            out.append(dict(zip(groups,identity))|dict(metric=metric,n_valid=len(vals),n_total=len(subset),median=median,q25=low,q75=high,
                iqr=float(high-low) if np.isfinite(low) and np.isfinite(high) else np.nan,
                minimum=minimum,maximum=maximum,
                range=float(maximum-minimum) if np.isfinite(minimum) and np.isfinite(maximum) else np.nan,
                ci95_low=ci[0],ci95_high=ci[1],interval_status=status,
                bootstrap_seed=20260906,bootstrap_resamples=10000,
                bootstrap_resamples_performed=performed,
                sampling_unit='paired initialization; no pixel resampling'))
    return pd.DataFrame(out)


def _render_c45(s,c4,c5,output):
    output.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(s.STYLE):
        fig,axes=plt.subplots(1,2,figsize=(10.0,4.9),sharey=True)
        fig.subplots_adjust(left=.10,right=.98,bottom=.25,top=.88,wspace=.24)
        for panel,(ax,(sign,shape)) in enumerate(zip(axes,C4_CASES)):
            for method in METHODS:
                q=c4[(c4.sign==sign)&(c4.requested==shape)&(c4.method==method)]
                for _,g in q.groupby('seed'):
                    g=g.sort_values('eta');ax.plot(g.eta,g.xz_amplitude_cosine_to_fixed_fe,color=s.COLORS[method],alpha=.17,lw=.9)
                grouped=q.groupby('eta').xz_amplitude_cosine_to_fixed_fe
                median=grouped.median().reindex(ETAS);lo=grouped.quantile(.25).reindex(ETAS);hi=grouped.quantile(.75).reindex(ETAS)
                ax.plot(ETAS,median,color=s.COLORS[method],marker=s.MARKERS[method],mfc='white',mew=1.5,ms=6,lw=2.2,label=method)
                ax.fill_between(ETAS,lo,hi,color=s.COLORS[method],alpha=.12)
            ax.set_xscale('log');ax.set_xticks(ETAS,['0.001','0.01','0.1']);ax.minorticks_off();s.tidy(ax)
            ax.set_title(('Standard sign' if sign=='corrected_minus' else 'Alternative sign')+' · '+shape,fontsize=13)
            ax.set_xlabel(r'Directional ratio $\eta$');ax.text(-.09,1.06,f'({chr(97+panel)})',transform=ax.transAxes,fontweight='bold')
        axes[0].set_ylabel('XZ field similarity to fixed FE');axes[0].set_ylim(max(0,c4.xz_amplitude_cosine_to_fixed_fe.min()-.03),1.015)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.025),ncol=3,frameon=False)
        s.export(fig,'Figure_C4_Twin_Bottle_directional_sensitivity',output)
        fig,axes=plt.subplots(1,2,figsize=(10,4.9));fig.subplots_adjust(left=.10,right=.98,bottom=.25,top=.90,wspace=.36)
        for panel,(ax,metric,ylabel) in enumerate(zip(axes,['target_pressure_over_shared_sweep_peak','xz_amplitude_cosine_to_nominal_conventional'],[r'Target $|p|/p_{max}$','XZ similarity to Conventional'])):
            for _,q in c5.groupby('seed'):
                q=q.sort_values('beta_factor');ax.plot(range(4),q[metric],color=s.COLORS['RH'],lw=.9,alpha=.22)
            group=c5.groupby('beta_factor')[metric];median=group.median().reindex(BETAS)
            lo=group.quantile(.25).reindex(BETAS);hi=group.quantile(.75).reindex(BETAS)
            ax.errorbar(range(4),median,yerr=[median-lo,hi-median],fmt='^-',color=s.COLORS['RH'],mfc='white',mew=1.6,ms=6,lw=2,capsize=4)
            ax.set_xticks(range(4),['0\n(FE)','0.1','1','10']);ax.set_xlabel(r'RH pressure weight $\beta/\beta_0$');ax.set_ylabel(ylabel);s.tidy(ax)
            ax.text(-.10,1.05,f'({chr(97+panel)})',transform=ax.transAxes,fontweight='bold')
        fig.legend(handles=[Line2D([],[],color=s.COLORS['RH'],marker='^',mfc='white',lw=2,label='Median / IQR; paired starts in pale lines')],loc='lower center',bbox_to_anchor=(.54,.02),frameon=False,fontsize=11)
        s.export(fig,'Figure_C5_RH_pressure_sensitivity',output)


def run(root,*,ctx=None,cache_only=False,workers=3,render_weight_figures=True):
    root=Path(root).resolve();base=Path(ctx.output_root) if ctx is not None else root/'production_outputs'
    output=base/'appendices/B_pressure';coutput=base/'appendices/C_GFE'
    output.mkdir(parents=True,exist_ok=True);coutput.mkdir(parents=True,exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    s,renderer=_modules(root);configs,uses=specifications(root)
    uses.to_csv(output/'command_section_map.csv',index=False)
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            solved=list(pool.map(lambda config:_import_or_solve(config,root/'appendix_B_pressure',output,s,cache_only),configs.values()))
        rows={key:row for key,row,_ in solved};arrays={key:data for key,_,data in solved};fields={}
        for key,config in configs.items():fields[key]=_pressure(config,arrays[key],output,cache_only)
    def lookup(seed,shape,sign,method,eta=.01,beta=1.):return s.cache_key(_configuration(s,shape,sign,method,eta,beta,seed=seed))
    def base_row(use):
        key=use.cache_key
        row=dict(rows[key]);row.update(section=use.section,seed=int(use.seed),requested=use.requested,sign=use.sign,method=use.method,eta=float(use.eta),beta_factor=float(use.beta_factor),evidence_status='PRODUCTION')
        row['cache_key']=key;return row
    c4=[];c5=[];b1=[];b2=[]
    for use in uses.itertuples():
        key=use.cache_key;row=base_row(use);field=fields[key]['XZ']
        if use.section in ('C4','C5'):
            ref=lookup(use.seed,use.requested,use.sign,'FE')
            similarity=s.command_similarity(arrays[key]['terminal_phase_rad'],arrays[ref]['terminal_phase_rad'])
            row.update(reference_seed=int(use.seed),reference_method='FE',reference_eta=.01,fixed_fe_reference_cache_key=ref,
                fixed_fe_reference_phase_content_id=rows[ref]['phase_content_id'],phase_command_similarity_to_fixed_fe=similarity,
                phase_projector_distance_to_fixed_fe=float(np.sqrt(max(0,1-similarity**2))),xz_amplitude_cosine_to_fixed_fe=s.cosine(field,fields[ref]['XZ']))
        if use.section=='C4':c4.append(row)
        elif use.section=='C5':
            keys=[lookup(use.seed,'Bottle','historical_plus','FE' if beta==0 else 'RH',beta=beta) for beta in BETAS]
            peak=max(float(abs(fields[k]['XZ']).max()) for k in keys);ref=lookup(use.seed,'Bottle','historical_plus','Conventional')
            row.update(matched_FE=use.beta_factor==0,shared_sweep_peak_pa=peak,target_pressure_over_shared_sweep_peak=float(abs(field[90,90])/peak),
                xz_amplitude_cosine_to_nominal_conventional=s.cosine(field,fields[ref]['XZ']),conventional_reference_phase_content_id=rows[ref]['phase_content_id'])
            c5.append(row)
        elif use.section=='B1':
            peak=max(float(abs(fields[key][plane]).max()) for plane in ('XY','XZ'))
            row.update(shared_XY_XZ_peak_pa=peak,target_pressure_over_shared_peak=float(abs(field[90,90])/peak));b1.append(row)
        elif use.section=='B2' and use.method!='Conventional':
            ref=lookup(use.seed,use.requested,use.sign,'Conventional');peaks=[float(abs(fields[lookup(use.seed,use.requested,use.sign,m)]['XZ']).max()) for m in METHODS]
            peak=max(peaks);a=abs(field).ravel();b=abs(fields[ref]['XZ']).ravel()
            row.update(reference_method=use.method,xz_amplitude_cosine_similarity=s.cosine(field,fields[ref]['XZ']),xz_amplitude_pearson_correlation=float(np.corrcoef(a,b)[0,1]),
                shared_peak_pa=peak,reference_target_relative_pressure=float(abs(field[90,90])/peak),conventional_target_relative_pressure=float(abs(fields[ref]['XZ'][90,90])/peak),
                command_cosine_to_conventional=s.command_similarity(arrays[key]['terminal_phase_rad'],arrays[ref]['terminal_phase_rad']),conventional_phase_content_id=rows[ref]['phase_content_id'])
            b2.append(row)
    c4,c5,b1,b2=map(pd.DataFrame,(c4,c5,b1,b2))
    for frame,path in [(c4,coutput/'C4_Twin_Bottle_directional_sensitivity.csv'),(c5,coutput/'C5_RH_pressure_sensitivity.csv'),(b1,output/'B1_pressure_summary.csv'),(b2,output/'B2_field_summary_all_seeds.csv')]:frame.to_csv(path,index=False)
    representatives={'axis_m':next(iter(fields.values()))['axis_m']}
    for shape,sign,method in renderer.REQUIRED:
        key=lookup(260905,shape,sign,method)
        for plane in ('XY','XZ'):representatives[renderer.field_name(shape,sign,method,plane)]=fields[key][plane]
    with plt.rc_context(renderer.STYLE):renderer.render_b1(representatives,output);renderer.render_b2(representatives,output)
    np.savez_compressed(output/'B1_B2_representative_pressure_fields.npz',**representatives)
    if render_weight_figures:
        _render_c45(s,c4,c5,coutput)
    allrows=pd.DataFrame(rows.values());allrows['evidence_status']='PRODUCTION';allrows.to_csv(output/'production_command_runs.csv',index=False)
    section_commands=pd.DataFrame([base_row(use) for use in uses.itertuples()])
    gradient_groups=['section','requested','sign','method','eta','beta_factor']
    gradient_summary=_summary(section_commands,gradient_groups,['gradient_l2'])
    gradient_summary['gradient_units']='N m^-1 rad^-1'
    gradient_summary['gradient_definition']='L2 norm of the terminal native phase-objective gradient'
    gradient_summary['cross_objective_comparison']='not valid; report within each declared objective'
    gradient_summary[gradient_summary.section.isin(('B1','B2'))].to_csv(
        output/'B_terminal_phase_gradient_summary.csv',index=False)
    gradient_summary[gradient_summary.section.isin(('C4','C5'))].to_csv(
        coutput/'C4_C5_terminal_phase_gradient_summary.csv',index=False)
    _summary(c4,['sign','requested','method','eta'],['xz_amplitude_cosine_to_fixed_fe','phase_command_similarity_to_fixed_fe']).to_csv(coutput/'C4_summary.csv',index=False)
    _summary(c5,['beta_factor'],['target_pressure_over_shared_sweep_peak','xz_amplitude_cosine_to_nominal_conventional']).to_csv(coutput/'C5_summary.csv',index=False)
    _summary(b2,['requested','reference_method'],['xz_amplitude_cosine_similarity','command_cosine_to_conventional']).to_csv(output/'B2_correspondence_summary.csv',index=False)
    protocol=dict(schema=SCHEMA,status='PRODUCTION',paired_seeds=B10,unique_commands=250,C45_unique_commands=200,B1_commands=20,B2_commands=60,
        figures_preserve_representative_seed=260905,pressure_grid_per_plane=181,source_hashes=s.SOURCE_HASHES,
        objective_signs={'standard':'Kp|p|^2-Kv|v|^2','alternative':'Kp|p|^2+Kv|v|^2'},solver_by_method={'FE':'compact L-BFGS-B','RH':'compact L-BFGS-B','Conventional':'BFGS'},maxiter=10000,gtol=1e-8,
        initial_phase_rule='gauge_full(default_rng(seed).uniform(-pi,pi,255)); shared within seed; cold independent solves',
        C4_reference='Fixed same-seed FE at eta=.01; retained for all methods and ratios',C5_reference='Fixed same-seed nominal Conventional for correspondence; same-seed FE for separate reference metrics',
        statistical_unit='Paired initialization; never pressure pixels',bootstrap_seed=20260906,bootstrap_resamples=10000,
        optional_RH_plus_g='Existing one-pair cache only; no extra production RH+g solve')
    (output/'production_protocol.json').write_text(json.dumps(protocol,indent=2))
    captions={
        'B1':'Standard-sign Bottle-formulation FE and Conventional pressure examples retain the original seed260905, with matched XY/XZ scaling. Ten paired initializations are exported,20commands. Fields show pressure geometry; no stable bottle conclusion is assumed.',
        'B2':'Alternative-sign Twin and Bottle correspondence at the original seed260905; the same three pressure comparisons are retained, with no branch embedding. Ten matched starts give60commands and40reference-versus-Conventional comparisons. Native terminal states and both command/pressure correspondence distributions are exported.',
        'C4':'Standard-sign Twin and alternative-sign Bottle directional-weight sensitivity at ten paired starts. Each seed has one frozen FE eta=.01 reference. Curves show median field cosine, shading IQR, pale lines individual starts;180command slots. The reference is a formulation reference, not a mechanically selected global optimum.',
        'C5':'Alternative-sign Bottle RH beta/beta0=0,.1,1,10 at ten paired starts. beta0=4.242640687e-6 N m^-1 Pa^-1. beta=0 reuses the identical FE objective and beta=1 the nominal C4 RH command; only two further commands per start are synthesized. Left:target pressure/shared within-seed sweep peak;right:XZ amplitude cosine to fixed same-seed Conventional. Median/IQR and all paired starts are shown;40slots,20additional commands.'}
    (output/'figure_captions.json').write_text(json.dumps({k:v for k,v in captions.items() if k.startswith('B')},indent=2))
    (coutput/'C4_C5_figure_captions.json').write_text(json.dumps({k:v for k,v in captions.items() if k.startswith('C')},indent=2))
    measured=f"PRODUCTION. {len(allrows)} actual command records; {int(allrows.success.sum())} native optimizer successes. All other native stops retained. B2/C4/C5 interval tables resample paired initializations, not pressure pixels. No RH+g expansion.\n"
    (output/'measured_results_production.txt').write_text(measured)
    figures=[str(path) for folder in (output/'figures',coutput) for path in sorted(folder.glob('Figure_*.png')) if path.stem.startswith(('Figure_B1','Figure_B2','Figure_C4','Figure_C5'))]
    return dict(appendix='B_C45',output_dir=str(output),figures=figures,unique_commands=len(allrows),command_map=str(output/'command_section_map.csv'))


def run_b3(root,*,ctx=None,cache_only=False):
    from . import appendix_b3_mechanics as b3
    root=Path(root).resolve()
    if ctx is None:raise ValueError('Production B3 requires the shared full-numerics context')
    return b3.run(root,ctx=ctx,output_root=Path(ctx.output_root)/'appendices/B_pressure/data/B3',production=True,cache_only=cache_only)
