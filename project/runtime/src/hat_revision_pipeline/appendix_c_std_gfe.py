"""Independent loss-STD weight/smoothing sensitivity and 30k GFE ablations.

The authoritative objective and staged optimizer are retained. An explicit
optional solver argument scales only STD epsilon and leaves force epsilon fixed;
its default preserves all established calculations.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from rineng_content_id import content_identity
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import appendix_c_feg as c
from . import multitrap as m
from .cache import digest_array, dependency_snapshot

SCHEMA = 'GFE-STD-independent-sensitivity-30k-v1'
SEEDS = (260869, 260870, 260871)
GAMMA_VALUES = (.3, 1., 3.)
STD_FACTORS = (0., .1, 1., 10.)
CAPS = (250, 29750)
ABLATIONS = ('GFE', 'No force smoothing', 'No pressure', 'No uniformity', 'FE')
LABELS = ('GFE', 'No force\nsmoothing', 'No pressure\npenalty', 'No STD\npenalty', 'FE')


def specifications(seeds=SEEDS):
    specs=[]
    for seed in seeds:
        for gamma in GAMMA_VALUES:
            specs.append(dict(seed=seed, gamma_uniformity=gamma, std_scale_multiplier=1., ablation='GFE', sweep='gamma'))
        for factor in STD_FACTORS:
            if factor != 1.:
                specs.append(dict(seed=seed,gamma_uniformity=1.,std_scale_multiplier=factor,ablation='GFE',sweep='epsilon_std'))
        for label in ABLATIONS[1:]:
            specs.append(dict(seed=seed,gamma_uniformity=1.,std_scale_multiplier=1.,ablation=label,sweep='ablation'))
    for s in specs:
        s['run_id']=f"STD_{s['ablation'].replace(' ','_')}_gamma{s['gamma_uniformity']:g}_eps{s['std_scale_multiplier']:g}_seed{s['seed']}"
        s['nominal']=s['ablation']=='GFE' and s['gamma_uniformity']==1 and s['std_scale_multiplier']==1
    return specs


def _solve(root,ctx,problem,spec,provided=None):
    config=replace(c.fixed_triple_fe_config(gtol=ctx.config.gtol),
        smooth_stage_maxiters=CAPS, compensate_effective_gravity=spec['ablation']!='FE',
        gamma_uniformity=spec['gamma_uniformity'])
    removed={'No force smoothing':'force_smoothing','No pressure':'pressure_retention','No uniformity':'uniformity'}.get(spec['ablation'])
    if removed: config=replace(config,components=replace(config.components,**{removed:False}))
    phase0=np.random.default_rng(spec['seed']).uniform(-np.pi,np.pi,problem.n_actuators)
    payload=dict(schema=SCHEMA,problem=problem.fingerprint,seed=spec['seed'],initial_phase_content_id=digest_array(phase0),
        config=config.to_payload(),std_scale_multiplier=spec['std_scale_multiplier'],physics_source_content_id=c._hash_sources(),
        adapter_rule='explicit uniformity_epsilon_multiplier; force epsilon unchanged')
    evidence=getattr(ctx,'evidence_status','SMOKE')
    if provided is not None:
        if spec['std_scale_multiplier']!=1 or c._canonical_config(provided['config'].to_payload())!=c._canonical_config(config.to_payload()):
            raise ValueError('Shared Triple command configuration does not match STD case')
        if provided['problem'].fingerprint!=problem.fingerprint:
            raise ValueError('Shared Triple problem does not match STD case')
        result=provided['run'];status='shared production nominal command'
        path=Path(provided['row'].get('cache_path',provided['row'].get('source',ctx.cache.root)))
        if not path.is_absolute():path=root/path
    else:
        result,status,path=ctx.cache.get_or_compute('C_STD_commands',payload,
            lambda:m.run_multitrap(problem,config,seed=spec['seed'],initial_phases=phase0,record_history=True,
                uniformity_epsilon_multiplier=spec['std_scale_multiplier'],
                metadata=dict(experiment=SCHEMA,std_scale_multiplier=spec['std_scale_multiplier'],evidence_status=evidence)))
    stage=result.stages[-1]
    ev=m.evaluate_multitrap_objective(problem,result.phases,config,force_epsilon=stage.force_epsilon,uniformity_epsilon=stage.uniformity_epsilon)
    original_scales=m.estimate_smooth_scales(problem,result.initial_phases,config)
    for st in result.stages:
        expected_force=st.stage_factor*original_scales.force_scale if config.components.force_smoothing else 0.
        expected_std=st.stage_factor*original_scales.loss_std_scale*spec['std_scale_multiplier'] if config.components.uniformity else 0.
        assert np.isclose(st.force_epsilon,expected_force,rtol=1e-14,atol=0)
        assert np.isclose(st.uniformity_epsilon,expected_std,rtol=1e-14,atol=0)
    # Compare physical loss dispersion using one common nominal loss definition.
    nominal=replace(c.fixed_triple_fe_config(gtol=ctx.config.gtol),compensate_effective_gravity=True,smooth_stage_maxiters=CAPS)
    reference_scales=m.estimate_smooth_scales(problem,result.initial_phases,nominal)
    fixed_ev=m.evaluate_multitrap_objective(problem,result.phases,nominal,
        force_epsilon=nominal.smooth_stage_factors[-1]*reference_scales.force_scale,
        uniformity_epsilon=nominal.smooth_stage_factors[-1]*reference_scales.loss_std_scale)
    row=dict(spec,task='Triple',method='FE' if spec['ablation']=='FE' else 'GFE',alpha=3000.,alpha_factor=1.,
        phase_content_id=digest_array(result.phases),initial_phase_content_id=digest_array(phase0),
        native_success=stage.success,native_status=stage.status,native_message=stage.message,
        iterations=result.iterations,evaluations=result.nfev,cap=sum(CAPS),gtol=config.gtol,
        terminal_gradient_l2=float(np.linalg.norm(ev.gradient_reduced)),terminal_gradient_inf=float(np.linalg.norm(ev.gradient_reduced,np.inf)),
        command_time_s=result.end_to_end_sec,command_timing_scope='native run_multitrap end-to-end; no pooled benchmark claim',
        local_loss_std_n_m=float(np.std(ev.local_losses,ddof=0)),
        common_nominal_loss_std_n_m=float(np.std(fixed_ev.local_losses,ddof=0)),
        common_nominal_loss_mean_n_m=float(np.mean(fixed_ev.local_losses)),
        terminal_uniformity_penalty_n_m=ev.uniformity_penalty,
        terminal_force_epsilon_n=stage.force_epsilon,terminal_uniformity_epsilon_n_m=stage.uniformity_epsilon,
        initial_force_scale_n=original_scales.force_scale,initial_loss_std_scale_n_m=original_scales.loss_std_scale,
        config_json=json.dumps(config.to_payload()),stages_json=json.dumps([asdict(st) for st in result.stages]),
        cache_status=status,cache_source=str(path.relative_to(root)),evidence_status=evidence)
    targets=np.array([s.target_m for s in problem.stencils])
    np.savez_compressed(ctx.output_root/'commands'/f"{spec['run_id']}.npz",phase_rad=result.phases,initial_phase_rad=phase0,
        targets_m=targets,local_losses_n_m=ev.local_losses,common_nominal_local_losses_n_m=fixed_ev.local_losses,
        positions_m=ctx.positions_m,normals=ctx.normals,config_json=json.dumps(config.to_payload()),
        std_scale_multiplier=spec['std_scale_multiplier'])
    (ctx.output_root/'commands'/f"{spec['run_id']}.json").write_text(json.dumps(row,indent=2))
    print(f"STD {spec['run_id']}: {status}; {row['iterations']} iters; {stage.message}",flush=True)
    return dict(row=row,phase=result.phases,targets=targets)


def _reference_correspondence(ctx,jobs):
    # Existing helper selects a GFE alpha=1 row. Make the nominal reference the
    # first row for each seed, then every other candidate retains that reference.
    jobs.sort(key=lambda j:(j['row']['seed'],not j['row']['nominal'],j['row']['run_id']))
    c._correspondence(ctx,jobs)
    for job in jobs:
        ref=next(j for j in jobs if j['row']['seed']==job['row']['seed'] and j['row']['nominal'])
        assert job['row']['reference_phase_content_id']==ref['row']['phase_content_id']


def _style(ax):
    c._style(ax)
    ax.tick_params(labelsize=13)


def _paired(ax,data,xcolumn,metric,values):
    colors=('#94B6CA','#B6BDC5','#95B6AB')
    seeds=tuple(sorted(data.seed.unique()))
    for index,seed in enumerate(seeds):
        color=colors[index%len(colors)]
        q=data[data.seed.eq(seed)].set_index(xcolumn).reindex(values)
        ax.plot(range(len(values)),q[metric],'-o',color=color,lw=1.8,ms=6.5,mfc='white',mew=1.6,zorder=2)
    if len(seeds)>3:
        grouped=data.groupby(xcolumn)[metric]
        ax.fill_between(range(len(values)),grouped.quantile(.25).reindex(values).to_numpy(),grouped.quantile(.75).reindex(values).to_numpy(),color='#16659A',alpha=.12,zorder=1)
    med=data.groupby(xcolumn)[metric].median().reindex(values)
    ax.plot(range(len(values)),med,'D',color='#153E56',ms=8,mew=1.5,zorder=4)
    ax.set_xlim(-.2,len(values)-.8)
    _style(ax)


def render(root,output_dir=None):
    output=Path(output_dir) if output_dir is not None else Path(root)/'appendix_outputs/C_GFE'; data=output/'std_sensitivity'
    frame=pd.read_csv(data/'C_STD_command_runs.csv');paths=[];coords=[]
    settings={'font.size':14,'axes.labelsize':15,'axes.labelweight':'bold','axes.titleweight':'bold','axes.titlesize':16,
        'legend.fontsize':13,'svg.fonttype':'none','pdf.fonttype':42}
    ablations=frame[frame.nominal|frame.sweep.eq('ablation')]
    with plt.rc_context(settings):
        fig,axes=plt.subplots(2,2,figsize=(14.2,10.7));fig.subplots_adjust(left=.105,right=.985,top=.93,bottom=.13,wspace=.29,hspace=.48)
        panels=(('field_cosine_to_fixed_GFE','(a) Pressure-field preservation','Similarity to fixed GFE'),
                ('worst_displacement_a','(b) Elastic equilibrium',r'Worst target $d/a$'),
                ('common_nominal_loss_std_n_m','(c) Target-loss balance',r'Loss STD (N m$^{-1}$)'),
                ('terminal_gradient_l2','(d) Terminal stationarity',r'Gradient norm (N m$^{-1}$ rad$^{-1}$)'))
        for ax,(metric,title,ylabel) in zip(axes.flat,panels):
            _paired(ax,ablations,'ablation',metric,ABLATIONS)
            ax.set_xticks(range(5),LABELS,fontsize=12);ax.set_title(title[:3],loc='left',y=1.095,pad=0);ax.set_ylabel(ylabel)
            for row in ablations.itertuples():coords.append(dict(figure='C2',run_id=row.run_id,metric=metric,x=ABLATIONS.index(row.ablation),y=getattr(row,metric)))
        axes[0,0].set_ylim(min(.95,ablations.field_cosine_to_fixed_GFE.min()-.03),1.035)
        axes[0,0].set_yticks([v for v in axes[0,0].get_yticks() if 0<=v<=1])
        axes[1,1].set_ylim(bottom=0)
        axes[1,1].axhline(1e-3,color='.4',lw=1.7,ls=':',zorder=1)
        handles=[Line2D([],[],color='#9BB3BE',marker='o',mfc='white',lw=1.8,label='Paired start'),
            Line2D([],[],color='#153E56',marker='D',ls='None',ms=8,label=f'Median of {frame.seed.nunique()} starts'),
            Line2D([],[],color='.4',lw=1.7,ls=':',label=r'Gradient: $10^{-3}$')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.015),ncol=3,frameon=False)
        paths+=c._save(fig,output,'Figure_C2_GFE_multitrap_regularization_ablations')
        fig,axes=plt.subplots(2,2,figsize=(13.7,10.5));fig.subplots_adjust(left=.105,right=.98,top=.93,bottom=.13,wspace=.29,hspace=.43)
        for i,(sweep,column,values,xlabel) in enumerate((('gamma','gamma_uniformity',GAMMA_VALUES,r'STD weight $\gamma_u/\gamma_{u,0}$'),
                ('epsilon_std','std_scale_multiplier',STD_FACTORS,r'STD smoothing $\epsilon_{\mathrm{STD}}/\epsilon_{\mathrm{STD},0}$'))):
            q=frame[(frame.sweep.eq(sweep))|frame.nominal]
            for j,(metric,title,ylabel) in enumerate((('field_cosine_to_fixed_GFE','Pressure-field preservation','Similarity to fixed GFE'),
                    ('common_nominal_loss_std_n_m' if i==0 else 'terminal_gradient_l2',
                     'Target-loss balance' if i==0 else 'Terminal stationarity',
                     r'Loss STD (N m$^{-1}$)' if i==0 else r'Gradient norm (N m$^{-1}$ rad$^{-1}$)'))):
                ax=axes[i,j];_paired(ax,q,column,metric,values)
                ax.set_xticks(range(len(values)),[f'{v:g}' for v in values]);ax.set_xlabel(xlabel)
                ax.set_title(f'({chr(97+2*i+j)})',loc='left',y=1.095,pad=0);ax.set_ylabel(ylabel)
                ax.axvline(list(values).index(1.),color='.75',lw=1.2,zorder=0)
                if i==1 and j==1:
                    ax.set_ylim(bottom=0)
                    ax.axhline(1e-3,color='.4',lw=1.7,ls=':',zorder=1)
                if j==0:
                    ax.set_ylim(min(.95,q[metric].min()-.03),1.035)
                    ax.set_yticks([v for v in ax.get_yticks() if 0<=v<=1])
                for row in q.itertuples():coords.append(dict(figure='C6',run_id=row.run_id,metric=metric,x=getattr(row,column),y=getattr(row,metric)))
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.018),ncol=3,frameon=False)
        paths+=c._save(fig,output,'Figure_C6_GFE_STD_weight_smoothing_sensitivity')
    pd.DataFrame(coords).to_csv(data/'C_STD_plot_coordinates.csv',index=False)
    return paths


def _write_text(root,ctx,commands,mechanics):
    output=ctx.output_root
    stage_rows=[];loss_rows=[]
    for row in commands.itertuples():
        for stage in json.loads(row.stages_json):
            stage_rows.append(dict(run_id=row.run_id,seed=row.seed,ablation=row.ablation,gamma_uniformity=row.gamma_uniformity,
                std_scale_multiplier=row.std_scale_multiplier,**stage))
        with np.load(output/'commands'/f'{row.run_id}.npz',allow_pickle=False) as packed:
            for index,(native,common) in enumerate(zip(packed['local_losses_n_m'],packed['common_nominal_local_losses_n_m'])):
                loss_rows.append(dict(run_id=row.run_id,seed=row.seed,target_id=f'T{index+1}',
                    native_local_loss_n_m=native,common_nominal_local_loss_n_m=common))
    pd.DataFrame(stage_rows).to_csv(output/'C_STD_solver_stages.csv',index=False)
    pd.DataFrame(loss_rows).to_csv(output/'C_STD_target_losses.csv',index=False)
    assert bool((commands.initial_force_scale_n > 1e-20).all())
    assert bool((commands.initial_loss_std_scale_n_m > 1e-20).all())
    for seed,q in commands[commands.ablation.eq('GFE')].groupby('seed'):
        assert len(set(q.terminal_force_epsilon_n)) == 1, 'STD sweep changed force smoothing'
    summary=commands.groupby(['sweep','ablation','gamma_uniformity','std_scale_multiplier']).agg(
        paired_seeds=('seed','nunique'),median_field_cosine=('field_cosine_to_fixed_GFE','median'),
        minimum_field_cosine=('field_cosine_to_fixed_GFE','min'),median_worst_displacement_a=('worst_displacement_a','median'),
        field_cosine_q25=('field_cosine_to_fixed_GFE',lambda x:x.quantile(.25)),field_cosine_q75=('field_cosine_to_fixed_GFE',lambda x:x.quantile(.75)),
        worst_displacement_a_q25=('worst_displacement_a',lambda x:x.quantile(.25)),worst_displacement_a_q75=('worst_displacement_a',lambda x:x.quantile(.75)),
        median_terminal_gradient_l2=('terminal_gradient_l2','median'),
        terminal_gradient_l2_q25=('terminal_gradient_l2',lambda x:x.quantile(.25)),terminal_gradient_l2_q75=('terminal_gradient_l2',lambda x:x.quantile(.75)),
        loss_std_q25=('common_nominal_loss_std_n_m',lambda x:x.quantile(.25)),loss_std_q75=('common_nominal_loss_std_n_m',lambda x:x.quantile(.75)),
        median_loss_std_n_m=('common_nominal_loss_std_n_m','median'),median_minimum_stiffness_n_m=('minimum_stiffness_n_m','median'),
        native_success_count=('native_success','sum'),all_roots_count=('all_roots','sum'),all_restoring_count=('all_locally_restoring','sum')).reset_index()
    summary.to_csv(output/'C_STD_summary.csv',index=False)
    current_c2=commands[commands.nominal|commands.sweep.eq('ablation')]
    current_c2.to_csv(output.parent/'C2_current_30000_command_runs.csv',index=False)
    summary[summary.sweep.eq('ablation')|(summary.gamma_uniformity.eq(1)&summary.std_scale_multiplier.eq(1)&summary.ablation.eq('GFE'))].to_csv(output.parent/'C_ablation_summary.csv',index=False)
    (output.parent/'C2_active_manifest.json').write_text(json.dumps(dict(figure='Figure_C2_GFE_multitrap_regularization_ablations',iteration_cap=30000,command_table='std_sensitivity/C_STD_command_runs.csv',mechanics_table='std_sensitivity/C_target_mechanics.csv',summary='std_sensitivity/C_STD_summary.csv',measured_results='std_sensitivity/measured_results_smoke.txt',historical_status='C_command_runs.csv contains the unchanged C1 alpha study and labeled historical 10000-cap component rows; those rows are not used in current C2.'),indent=2))
    random=np.random.default_rng(20260906);paired=[]
    for keys,q in commands.groupby(['sweep','ablation','gamma_uniformity','std_scale_multiplier']):
        for metric in ('field_cosine_to_fixed_GFE','worst_displacement_a','common_nominal_loss_std_n_m','minimum_stiffness_n_m','terminal_gradient_l2'):
            ref=commands[commands.nominal].set_index('seed')[metric]
            diff=q.set_index('seed')[metric]-ref
            diff=diff.dropna().to_numpy()
            if len(diff):
                draw=np.median(diff[random.integers(0,len(diff),size=(10000,len(diff)))],axis=1)
                lo,hi=np.quantile(draw,[.025,.975]);estimate=float(np.median(diff))
            else:
                lo=hi=estimate=float('nan')
            paired.append(dict(sweep=keys[0],ablation=keys[1],gamma_uniformity=keys[2],std_scale_multiplier=keys[3],metric=metric,
                n_paired_seeds=len(diff),median_paired_change=estimate,ci95_lower=lo,ci95_upper=hi,
                bootstrap_seed=20260906,bootstrap_resamples=10000,interval_scope='Exploratory paired initialization bootstrap; targets are not independent'))
    pd.DataFrame(paired).to_csv(output/'C_STD_paired_bootstrap.csv',index=False)
    cap2=('Matched 30,000-cap Triple GFE component ablations, with three shared random initializations (260869, 260870, 260871). '
        'Each candidate is compared to the frozen same-seed complete GFE solution. (a) Complex-pressure cosine on fixed 9x9x9 volumes around all three prescribed targets. '
        '(b) Worst target displacement to the independent finite-ka elastic total-force equilibrium, in bead radii a=0.65 mm. '
        '(c) Population STD of local target losses, always reevaluated using the same nominal GFE loss and its fixed same-seed terminal smoothing; this keeps component-removal comparisons commensurate. '
        'Uniformity penalizes dispersion of objective losses, not STD of displacement or stiffness. (d) Terminal reduced-phase gradient L2 for each native objective. The linear axis begins at zero; the dotted 1e-3 L2 line is the established reporting criterion, distinct from native BFGS infinity-norm gtol=1e-8. This gradient is a solver diagnostic, while the other panels use fixed reference or common-loss metrics. '
        'Curves pair identical starts and diamonds show medians. FE retains all regularizers with zero gravity target. Alpha=3000 m^-1, beta=5e-5 N m^-1 Pa^-1, gamma_u=1, W=diag(1,1,10). '
        'For component removals, the established initialization-based smoothing-scale rule is retained, so the realized epsilon values are exported rather than assumed identical between different local losses. '
        'Stage budgets are 250 and 29750; native stops and gradient norms remain in the exports. This replaces the earlier 10k C2. Data: std_sensitivity/C_STD_command_runs.csv and C_target_mechanics.csv. SMOKE.')
    cap6=('Independent sensitivity of the loss-uniformity weight gamma_u and STD smoothing epsilon_STD. '
        'The objective is mean(ell_i)+gamma_u*(sqrt(var_pop(ell_i)+epsilon_STD^2)-epsilon_STD). '
        'Top row: gamma_u=0.3,1,3 with both smoothing schedules fixed. Bottom row: epsilon_STD alone is multiplied by 0,0.1,1,10, with gamma_u=1 and the force smoothing schedule unchanged. '
        'Zero means unsmoothed STD, not removal of the STD penalty. Nominal stage values are (0.05,0.01)*max(STD of initial unsmoothed local losses,1e-20 N m^-1); all actual epsilons are exported. '
        'Each sweep uses three shared random starts and a frozen same-seed nominal GFE reference, independent of the sweep outcome. '
        'Left: complex-pressure cosine to that reference. Upper right: population STD of local losses reevaluated under the same fixed nominal GFE loss for all gamma values. Lower right: terminal reduced-phase gradient L2 for each native objective at its actual epsilon. A linear zero-based axis and the established 1e-3 L2 reporting threshold show residual stationarity; the native BFGS tolerance is a separate infinity-norm criterion. The two right panels show the distinct balance and smoothing roles of the constants. Independent elastic equilibrium displacement and stiffness for every candidate remain in the exports; C2(b) shows the mechanical comparison for component removals. '
        'At most 30000 iterations per command, with the same BFGS native stopping criteria and no forced restart. Three targets from one command are dependent. SMOKE.')
    captions={'Figure_C2_GFE_multitrap_regularization_ablations':cap2,'Figure_C6_GFE_STD_weight_smoothing_sensitivity':cap6}
    (output/'figure_captions.json').write_text(json.dumps(captions,indent=2))
    shared=output.parent/'figure_captions.json'
    old=json.loads(shared.read_text()) if shared.exists() else {};old.update(captions);shared.write_text(json.dumps(old,indent=2))
    protocol=dict(schema=SCHEMA,paired_seeds=SEEDS,commands=len(commands),target_count=len(mechanics),
        stage_maxiters=CAPS,gamma_values=GAMMA_VALUES,std_multipliers=STD_FACTORS,
        uniformity_expression='gamma_u * (sqrt(mean((ell_i-mean(ell))**2)+epsilon_STD**2)-epsilon_STD)',
        variance_ddof=0,nominal_epsilon_STD='stage_factor * max(std(initial unsmoothed local losses, ddof=0), 1e-20)',
        smooth_stage_factors=[.05,.01],smooth_scale_floor=1e-20,
        observed_initial_force_scale_range_n=[float(commands.initial_force_scale_n.min()),float(commands.initial_force_scale_n.max())],
        observed_initial_STD_scale_range_n_m=[float(commands.initial_loss_std_scale_n_m.min()),float(commands.initial_loss_std_scale_n_m.max())],
        force_smoothing_held_fixed_in_STD_sweep=True,pressure_epsilon_rel=.001,
        floor_audit='Floor is not an active experimental hyperparameter here: actual initial force and STD scales are exported and far exceed the floor.',
        reference_rule='Freeze independently solved nominal GFE for each shared seed; never select nearest swept solution',
        adapter='Explicit run_multitrap uniformity_epsilon_multiplier argument; default=1 preserves established nominal behavior.',
        physics_source_content_id=c._hash_sources(),dependencies=dependency_snapshot(),evidence_status='SMOKE')
    (output/'C_STD_protocol.json').write_text(json.dumps(protocol,indent=2))
    lines=['STD regularization: weight sensitivity, smoothing sensitivity and matched component removals.',
        'Previously the uniformity penalty had only been removed as an ablation; gamma_u and epsilon_STD had not been independently swept.',
        'New data use 30 commands = 10 unique configurations x 3 paired initializations and 90 target validations. Nominal gamma_u=1. Epsilon_STD and force epsilon are varied independently.',
        'The uniformity penalty balances per-target objective losses. It does not mathematically equalize exact displacement or stiffness. The common nominal-loss STD in C2 is recomputed under one fixed loss definition for every ablation.',
        f"Native optimizer successes: {int(commands.native_success.sum())}/{len(commands)}; all-target resolved commands: {int(commands.all_roots.sum())}/{len(commands)}; all-target restoring commands: {int(commands.all_locally_restoring.sum())}/{len(commands)}.",
        '\n'.join(f'{r.ablation}, gamma_u={r.gamma_uniformity:g}, epsilon_STD multiplier={r.std_scale_multiplier:g}: median field cosine {r.median_field_cosine:.6f}; common-loss STD {r.median_loss_std_n_m:.6g} N/m; terminal gradient {r.median_terminal_gradient_l2:.6g} N/m/rad; worst target displacement {r.median_worst_displacement_a:.6g} a; native successes {r.native_success_count}/{r.paired_seeds}.' for r in summary.itertuples()),
        'Stage factor 0.05 -> 0.01 is a shared continuation schedule. The independent STD multiplier changes only epsilon_STD. Pressure smoothing (epsilon_p), force smoothing (epsilon_g), pressure weight beta and curvature weights are separate controls, not constants of the STD penalty. The 1e-20 scale floor is inactive for these measured initial scales.',
        'Three paired starts are a smoke-level sensitivity assessment, not full production uncertainty. Terminal messages and seed-paired bootstrap summaries are exported without relabeling precision loss as convergence.']
    (output/'measured_results_smoke.txt').write_text('\n\n'.join(lines)+'\n')


def run(root,cache_only=False,workers=3,render_figures=True):
    root=Path(root).resolve();output=root/'appendix_outputs/C_GFE/std_sensitivity';output.mkdir(parents=True,exist_ok=True)
    (output/'commands').mkdir(exist_ok=True)
    ctx=c._context(root,output,cache_only)
    problem=c.make_problem(ctx,c.contracted_targets(.8),0.,0.,case_prefix='fixed-long-triple-observation')
    jobs=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for job in pool.map(lambda spec:_solve(root,ctx,problem,spec),specifications()):
            jobs.append(job);pd.DataFrame([j['row'] for j in jobs]).to_csv(output/'C_STD_command_runs_partial.csv',index=False)
    _reference_correspondence(ctx,jobs)
    mechanics=c._validate(root,ctx,jobs,workers)
    mechanics['validation_source']=mechanics.validation_source.str.replace('appendix_outputs/C_GFE/cache/finite_ka_static_validation',
        'appendix_outputs/C_GFE/std_sensitivity/cache/finite_ka_static_validation',regex=False)
    mechanics.to_csv(output/'C_target_mechanics.csv',index=False)
    commands=pd.DataFrame([j['row'] for j in jobs]);commands.to_csv(output/'C_STD_command_runs.csv',index=False)
    for job in jobs:(output/'commands'/f"{job['row']['run_id']}.json").write_text(json.dumps(job['row'],indent=2))
    _write_text(root,ctx,commands,mechanics)
    paths=render(root) if render_figures else []
    (output/'C_STD_cache_access.json').write_text(json.dumps(ctx.cache.access_records,indent=2))
    return dict(appendix='C_STD',output_dir=str(output),figures=[x for x in paths if x.endswith('.png')],figure_exports=paths,
        tables={path.stem:str(path) for path in sorted(output.glob('*.csv'))},measured_results=str(output/'measured_results_smoke.txt'),evidence_status='SMOKE')

if __name__=='__main__':
    import sys
    print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else Path.cwd(),cache_only='--cache-only' in sys.argv),indent=2))
