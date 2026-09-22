"""Sensitivity Appendix C and optimizer Appendix F: 22-figure sequence.

This is a reporting revision. Native objectives, solver arithmetic, fixed
references and mechanical validators are unchanged. Archived exploratory
results remain data; they are never discovered as active figures by a glob.
"""
from pathlib import Path
from io import BytesIO

from rineng_content_id import content_identity
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

SCHEMA = 'focused-editor-response-20260908-v2-si-visual'
STEMS = ('Figure_C1_objective_weight_sensitivity',
         'Figure_C2_multitrap_regularization',
         'Figure_C3_RH_beta_choice',
         'Figure_F1_optimizer_robustness')
COLORS = {'GFE':'#087F72', 'FE':'#C78725', 'Conventional':'#427AA1',
          'beta':'#87569B', 'pressure':'#87569B', 'force':'#087F72', 'STD':'#C78725'}
STYLE = {'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12.5,
         'axes.labelweight':'semibold','axes.linewidth':1.4,'xtick.labelsize':10.5,
         'ytick.labelsize':11,'legend.fontsize':10,'pdf.fonttype':42,'svg.fonttype':'none'}


def base(ctx):
    root=Path(ctx.memo['notebook_root'])
    return Path(ctx.output_root)/'appendices/C_GFE' if ctx.config.full else root/'appendix_outputs/C_GFE'


def output(ctx):
    path=base(ctx)/'reviewer_focused';path.mkdir(parents=True,exist_ok=True)
    return path


def optimizer_output(ctx):
    root=Path(ctx.memo['notebook_root'])
    path=Path(ctx.output_root)/'appendices/F_optimizer' if ctx.config.full else root/'appendix_outputs/F_optimizer'
    path.mkdir(parents=True,exist_ok=True)
    return path


def _content_id(path):
    return content_identity(Path(path).read_bytes()).hexdigest()


def _json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str))


def _axes(ax,letter):
    ax.text(-.14,1.045,f'({letter})',transform=ax.transAxes,fontweight='bold',fontsize=13)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18,lw=.7);ax.set_axisbelow(True)
    ax.tick_params(length=4,width=1.25)


def _save(fig,ctx,stem,caption):
    from pypdf import PdfReader
    out=(optimizer_output(ctx) if stem.startswith('Figure_F') else output(ctx))/'figures';out.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):
        buffer=BytesIO();fig.savefig(buffer,format=ext,dpi=260,bbox_inches='tight',facecolor='white')
        content=buffer.getvalue()
        if ext=='pdf':assert len(PdfReader(BytesIO(content)).pages)==1
        path=out/(stem+'.'+ext);tmp=path.with_suffix('.'+ext+'.tmp');tmp.write_bytes(content);tmp.replace(path)
    plt.close(fig)
    cp=out/'figure_captions.json';caps=json.loads(cp.read_text()) if cp.exists() else {}
    caps[stem]=caption;_json(cp,caps)
    return out/(stem+'.png')


def _curve(ax,frame,x,metric,color,marker='o',ls='-',label=None,failures=True):
    frame=frame.copy();frame[x]=pd.to_numeric(frame[x]);frame=frame.sort_values(x)
    g=frame.groupby(x)[metric];med=g.median();lo=g.quantile(.25);hi=g.quantile(.75)
    ax.plot(med.index,med,color=color,marker=marker,mfc='white',mew=1.2,ms=4.4,lw=1.7,ls=ls,label=label)
    if frame.seed.nunique()>1:ax.fill_between(med.index,lo,hi,color=color,alpha=.10)
    if failures and 'raw_gtol_met' in frame:
        f=frame[~frame.raw_gtol_met.astype(bool)]
        ax.scatter(f[x],f[metric],color=color,marker='x',s=22,lw=1.05,zorder=5)


def _meter_columns(frame):
    """Add SI displacement columns without changing the frozen normalized data."""
    from .exact_validator import ElasticSphere
    result = frame.copy()
    radius_m = float(ElasticSphere().radius_m)
    for column in frame:
        if column.endswith('displacement_a'):
            result[column[:-2] + '_m'] = pd.to_numeric(frame[column]) * radius_m
    return result


def _meter_axis(ax):
    # SI values remain in the plotted data; only their tick representation is compact.
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.get_offset_text().set_fontsize(10)


def weights(ctx,frame):
    frame = _meter_columns(frame)
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,2,figsize=(11.5,8.6))
        fig.subplots_adjust(left=.12,right=.985,bottom=.18,top=.945,hspace=.37,wspace=.32)
        for i,task in enumerate(('Single','Triple')):
            cases=[('alpha_factor','GFE',COLORS['GFE'],'o','-',r'GFE: $\alpha$'),
                   ('alpha_factor','FE',COLORS['FE'],'s','-',r'FE: $\alpha$'),
                   ('lambda_pressure_factor','Conventional',COLORS['Conventional'],'^','-',r'Conventional: $\lambda_p$')]
            if task=='Triple':cases.append(('beta_factor','GFE',COLORS['beta'],'D','--',r'GFE: $\beta$'))
            for parameter,method,color,marker,ls,label in cases:
                q=frame[frame.task.eq(task)&frame.method.eq(method)&frame.parameter.eq(parameter)]
                for j,metric in enumerate(('field_cosine_to_fixed_GFE','worst_displacement_m')):
                    _curve(axs[i,j],q,'parameter_value',metric,color,marker,ls,label)
                    if metric=='worst_displacement_m':
                        missing=q[q[metric].isna()]
                        if len(missing):axs[i,j].scatter(pd.to_numeric(missing.parameter_value),[.96]*len(missing),transform=axs[i,j].get_xaxis_transform(),marker='x',color=color,s=30,zorder=7)
            for j,ax in enumerate(axs[i]):
                _axes(ax,chr(97+2*i+j));ax.set_xscale('symlog',linthresh=.01,linscale=.35)
                ax.set_xticks([0,.01,.1,1,10,100],['0','0.01','0.1','1','10','100'])
                ax.axvline(1,color='.65',ls=':',lw=1.1)
                ax.set_xlabel('Coefficient / nominal')
                ax.set_ylabel('Field cosine' if j==0 else 'Worst displacement (m)')
                if j==0:ax.set_ylim(0,1.035)
                else:
                    ax.set_ylim(bottom=0)
                    _meter_axis(ax)
            axs[i,0].text(-.24,.5,task,transform=axs[i,0].transAxes,rotation=90,va='center',fontweight='semibold')
        handles=axs[1,0].get_legend_handles_labels()[0]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.025),ncol=2,frameon=False)
        return _save(fig,ctx,STEMS[0],
            'C1. One coefficient is varied at a time, holding a same-seed nominal GFE reference fixed. '
            'Rows: Single and Triple; columns: complex-pressure cosine on the fixed 9-cubed local volumes and worst-target independently validated elastic equilibrium displacement in metres (m), d = max_j ||r_eq,j - r_target,j||. '
            'Alpha curves use FE/GFE; lambda_p is the Conventional pressure coefficient; beta is the Triple GFE pressure coefficient. The abscissa divides each varied coefficient by its own dimensional nominal value. '
            'Vertical dotted lines denote nominal coefficients. Crosses on curves retain endpoints that miss their native infinity-gradient tolerance; crosses at the upper edge of displacement panels denote unresolved roots. Field similarity is not a substitute for mechanical precision. '
            f"Independent seeds per task: {frame.groupby('task').seed.nunique().to_dict()}; lines are medians and, when replicated, bands are IQR. "
            'Single/Triple native ceilings are 10000/30000. All other weight, smoothing and curvature-ratio sweeps are in the sensitivity table; grid points are not independent replications.')


def regularization(ctx,continuous,std):
    continuous, std = _meter_columns(continuous), _meter_columns(std)
    ab=std[std.nominal.astype(bool)|std.sweep.eq('ablation')].copy()
    labels=['GFE','No force\nsmoothing','No pressure\npenalty','No STD\npenalty','FE']
    keys=['GFE','No force smoothing','No pressure','No uniformity','FE']
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,2,figsize=(11.5,8.8))
        fig.subplots_adjust(left=.12,right=.98,bottom=.105,top=.945,hspace=.50,wspace=.34)
        for j,metric in enumerate(('field_cosine_to_fixed_GFE','worst_displacement_m')):
            ax=axs[0,j]
            for seed,q in ab.groupby('seed'):
                q=q.set_index('ablation').reindex(keys)
                ax.plot(range(5),q[metric],color='#9AADB0',marker='o',mfc='white',mew=1,ms=3.8,lw=.9,alpha=.65)
            g=ab.groupby('ablation')[metric];med=g.median().reindex(keys)
            ax.errorbar(range(5),med,yerr=[med-g.quantile(.25).reindex(keys),g.quantile(.75).reindex(keys)-med],
                        fmt='D',color=COLORS['GFE'],ms=5,capsize=3,lw=1.5)
            if metric=='worst_displacement_m':
                for index,key in enumerate(keys):
                    count=int(ab[ab.ablation.eq(key)][metric].isna().sum())
                    if count:ax.scatter(index+np.linspace(-.045,.045,count),[.94]*count,transform=ax.get_xaxis_transform(),marker='x',color='#596B73',s=30,zorder=6)
            ax.set_xticks(range(5),labels,fontsize=10)
            ax.set_ylabel('Field cosine' if j==0 else 'Worst displacement (m)')
            ax.set_ylim((0,1.035) if j==0 else (0,None))
            if j == 1: _meter_axis(ax)
        g=continuous[continuous.parameter.eq('gamma_uniformity')&continuous.task.eq('Triple')]
        _curve(axs[1,0],g,'parameter_value','common_nominal_loss_std_n_m',COLORS['GFE'])
        axs[1,0].set_xscale('symlog',linthresh=.03,linscale=.4)
        axs[1,0].set_xticks([0,.03,.1,.3,1,3,10],['0','.03','.1','.3','1','3','10'])
        axs[1,0].set_yscale('log');axs[1,0].set_xlabel(r'STD weight $\gamma_u$')
        axs[1,0].set_ylabel('Loss STD (N m$^{-1}$)')
        for parameter,key,label in [('epsilon_pressure_rel','pressure',r'$\epsilon_p$'),
                                     ('epsilon_force_factor','force',r'$\epsilon_g$'),
                                     ('epsilon_std_factor','STD',r'$\epsilon_{\mathrm{STD}}$')]:
            q=continuous[continuous.parameter.eq(parameter)&continuous.task.eq('Triple')].copy()
            q['multiplier']=pd.to_numeric(q.parameter_value)/(.001 if key=='pressure' else 1.)
            _curve(axs[1,1],q,'multiplier','iterations',COLORS[key],label=label)
        axs[1,1].set_xscale('symlog',linthresh=.01,linscale=.35)
        axs[1,1].set_xticks([0,.01,.1,1,10,30],['0','.01','.1','1','10','30'])
        axs[1,1].set_xlabel('Smoothing / nominal');axs[1,1].set_ylabel('Terminal iteration')
        axs[1,1].set_ylim(0,31500);axs[1,1].axhline(30000,color='.6',ls=':',lw=1)
        axs[1,1].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.12),handlelength=1.4,columnspacing=1)
        for k,ax in enumerate(axs.flat):_axes(ax,chr(97+k))
        return _save(fig,ctx,STEMS[1],
            f'C2. Triple component ablations use {ab.seed.nunique()} paired starts at the unchanged 30000-iteration cap. '
            '(a,b) Complete GFE, removal of force smoothing, pressure penalty or STD penalty, and FE as the zero-gravity-target control; displacement is reported in metres. Pale points are starts; diamonds and bars are median and IQR. Upper-edge crosses in (b) denote unresolved roots, not measured displacements; finite-value medians retain separate failure counts in the table. '
            '(c) Gamma changes the population STD of target losses, reevaluated under one common nominal loss definition. This is not the STD of equilibrium displacement. '
            '(d) Independently changing pressure, force or STD smoothing shows the actual terminal iteration; crosses are failed native stops and the dotted line is the cap. '
            f"Panels (c,d) use {g.seed.nunique()} seed(s); the pressure epsilon is divided by its nominal relative value 0.001. "
            'Zero epsilon_STD retains an unsmoothed STD penalty, whereas removal of the STD term in (a,b) removes its weight. '
            'Complex-pressure cosine is sensitive to local phase/circulation as well as amplitude; a low value alone does not mean the trap disappeared. Every smoothing candidate also has exported field correspondence, equilibrium displacement, all three stiffness eigenvalues and native stop status. '
            'The initialization-based dimensional scale rule is retained for changed losses; epsilon_g and epsilon_STD are varied independently in their dedicated sweeps.')


def beta_choice(ctx,table):
    from . import reviewer_beta as b
    # Reuse the reviewed four-panel design without generating a legacy C5 file.
    factors=sorted(table.beta_factor.unique());pos=np.arange(len(factors))
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,2,figsize=(11.5,8.5));fig.subplots_adjust(left=.11,right=.985,bottom=.105,top=.945,hspace=.39,wspace=.31)
        def curve(ax,metric,color,marker,label=None):
            q=table.copy();q['position']=q.beta_factor.map(dict(zip(factors,pos)))
            _curve(ax,q,'position',metric,color,marker,label=label,failures=False)
        curve(axs[0,0],'iterations',COLORS['beta'],'o')
        bad=table[~table.optimizer_success.astype(bool)]
        axs[0,0].scatter([factors.index(v) for v in bad.beta_factor],bad.iterations,marker='x',s=32,color='#B33B35',zorder=7,label='Native stop unmet')
        axs[0,0].set_ylabel('Terminal iteration');axs[0,0].set_ylim(bottom=0)
        for ax,prefix,label in [(axs[0,1],'command_similarity','Command cosine'),(axs[1,0],'xz_field_similarity','XZ amplitude cosine')]:
            for ref,color,marker in [('fe',COLORS['FE'],'s'),('rh',COLORS['beta'],'^')]:
                curve(ax,prefix+'_to_fixed_'+ref,color,marker,'FE reference' if ref=='fe' else 'RH reference')
            ax.set_ylabel(label);ax.set_ylim(0,1.035)
        axs[0,1].legend(frameon=False,loc='lower left')
        curve(axs[1,1],'target_pressure_over_shared_peak',COLORS['beta'],'o')
        axs[1,1].set_ylabel(r'$|p_t|/p_{\max}$');axs[1,1].set_ylim(bottom=0)
        for k,ax in enumerate(axs.flat):
            _axes(ax,chr(97+k))
            major = [index for index, value in enumerate(factors) if value in (0., .01, .1, 1., 10., 100.)]
            ax.set_xticks(major, [f'{factors[index]:g}' for index in major])
            ax.set_xticks(pos, minor=True)
            ax.set_xlabel(r'$\beta/\beta_0$')
            ax.axvline(factors.index(1.),color='.6',ls=':',lw=1)
        return _save(fig,ctx,STEMS[2],
            f'C3. RH pressure-weight choice in the alternative-sign Bottle formulation, {table.seed.nunique()} paired start(s) and ten beta values. '
            '(a) Native terminal iterations and unmet stops. (b,c) Full-dimensional command and XZ pressure-amplitude cosines to independently frozen same-seed FE at beta=0 and RH at beta0. '
            '(d) Target pressure amplitude |p_t| divided by p_max, the fixed shared peak over beta/beta0=0,0.1,1,10. The ten plotted beta factors are 0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 and 100; minor ticks retain intermediate settings. References and normalization do not move during the sweep. '
            'Beta0=4.242640687119285e-6 N m^-1 Pa^-1, alpha=9 m^-1, eta=0.01; native cap=10000, gtol=1e-8. '
            'A numerical beta/alpha ratio is dimensionful: report the SI coefficients, beta/beta0 and exported weighted loss/gradient contributions rather than a unit-free six-order rule. '
            'Curves are medians; replicated runs include IQR. This is an RH Bottle control, not a claim about arbitrary trap shapes or all GFE settings.')


def optimizers(ctx,frame):
    opts=['BFGS','L-BFGS-B','Adam'];frame=_meter_columns(frame[frame.optimizer.isin(opts)])
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(2,2,figsize=(11.5,8.6));fig.subplots_adjust(left=.115,right=.985,bottom=.15,top=.945,hspace=.40,wspace=.32)
        for i,method in enumerate(('Conventional','FE','GFE')):
            color=COLORS[method];offset=(i-1)*.23
            for idx,opt in enumerate(opts):
                q=frame[frame.method.eq(method)&frame.optimizer.eq(opt)];x=idx+offset;n=len(q)
                if not n:
                    continue
                success=int(q.raw_gtol_met.astype(bool).sum());axs[0,0].scatter(x,success/n,s=34,color=color,marker='o')
                for ax,metric in [(axs[0,1],'solve_wall_s'),(axs[1,0],'field_cosine_to_fixed_GFE'),(axs[1,1],'worst_displacement_m')]:
                    vals=q[metric].to_numpy(float);valid=vals[np.isfinite(vals)]
                    if len(valid):
                        ax.scatter(x+np.linspace(-.045,.045,n),vals,s=15,color=color,alpha=.55)
                        lo,med,hi=np.quantile(valid,[.25,.5,.75]);ax.errorbar(x,med,yerr=[[med-lo],[hi-med]],color=color,fmt='o',ms=4.7,lw=1.5,capsize=3)
                    if metric=='solve_wall_s':
                        bad=q[~q.raw_gtol_met.astype(bool)];ax.scatter([x]*len(bad),bad[metric],color=color,marker='x',s=28,zorder=5)
                    if metric=='worst_displacement_m':
                        missing=int(q[metric].isna().sum())
                        if missing:ax.scatter(x+np.linspace(-.045,.045,missing),[.94]*missing,transform=ax.get_xaxis_transform(),marker='x',color=color,s=30)
        labels=['Stationarity fraction','Solver return (s)','Field cosine','Displacement (m)']
        for k,ax in enumerate(axs.flat):_axes(ax,chr(97+k));ax.set_xticks(range(3),opts);ax.set_ylabel(labels[k])
        axs[0,0].set_ylim(-.08,1.12);axs[0,0].set_yticks([0,.5,1]);axs[0,1].set_yscale('log')
        axs[1,0].set_ylim(0,1.04);axs[1,1].set_ylim(bottom=0);_meter_axis(axs[1,1])
        handles=[Line2D([],[],color=COLORS[m],marker='o',lw=0,label=m) for m in ('Conventional','FE','GFE')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.062),ncol=3,frameon=False)
        failure_key = Line2D([], [], color='.3', marker='x', ls='none',
                             label='Stop unmet (b); unresolved root (d)')
        fig.legend(handles=[failure_key], loc='lower center', bbox_to_anchor=(.52,.012),
                   frameon=False, fontsize=9.5, handlelength=1.0)
        return _save(fig,ctx,STEMS[3],
            f'F1. Single-target solver control, {frame.seed.nunique()} paired independent initializations. '
            'FE/GFE use the compact evaluator with L-BFGS-B throughout. Conventional is evaluated with BFGS, L-BFGS-B and Adam; its objective and paired starts are unchanged. All methods use the native 10000-iteration ceiling. Adam has no quasi-Newton polish. '
            '(a) Fraction meeting each objective\'s native reduced-phase infinity-gradient tolerance 1e-8. (b) Serial CPU time to solver return, excluding setup and mechanics; crosses are unsuccessful returns, not time to successful convergence. '
            '(c) Fixed-reference complex-pressure cosine. (d) Independent elastic total-force equilibrium displacement including effective gravity, in metres (m); crosses at the upper edge denote unresolved roots, not displacement values. '
            'Small dots are starts and bars are median/IQR. Physical root/restoring counts, all stiffness eigenvalues, evaluations and raw gradient norms remain in tables. '
            'The FE/GFE results describe the configured compact L-BFGS-B method; the Conventional controls assess its dependence on optimizer choice in this Single configuration.')


def _read_inputs(ctx):
    b=base(ctx)
    return {key:_meter_columns(pd.read_csv(b/name)) for key,name in {
        'continuous':'reviewer_robustness/continuous/command_runs.csv',
        'optimizer':'reviewer_robustness/optimizer/command_runs.csv',
        'std':'std_sensitivity/C_STD_command_runs.csv',
        'beta':'beta_review_data/C5_beta_results.csv'}.items() if (b/name).exists()}


def run_sensitivity(ctx,workers=4):
    from . import reviewer_robustness as r, reviewer_beta as b, reviewer_formulation_weights as dw
    root=Path(ctx.memo['notebook_root']);inputs=None
    if not ctx.config.full and not ctx.memo.get('regenerate_smoke_analysis', False):
        # The complete immutable input manifest is embedded in the notebook;
        # a changed objective invalidates the report-only smoke shortcut.
        stamp=root/'focused_smoke_input_manifest.json'
        if stamp.exists():
            record=json.loads(stamp.read_text())
            for name,content_id in record['source_content_id'].items():
                if _content_id(root/name)!=content_id:raise RuntimeError('Changed scientific source; rebuild numerical evidence before using saved smoke tables: '+name)
            if all((root/name).is_file() for name in record['input_content_id']):
                for name,content_id in record['input_content_id'].items():
                    if _content_id(root/name)!=content_id:raise RuntimeError('Changed frozen smoke input: '+name)
                inputs=_read_inputs(ctx)
    saved_smoke_replay=inputs is not None
    if inputs is None:
        rc=r.context(root,shared=ctx)
        r.continuous_sensitivity(rc,workers=workers)
        if ctx.config.full:
            from . import production_c
            production_c.run_std(root,ctx,workers=workers,render_figures=False)
        else:
            from . import appendix_c_std_gfe
            appendix_c_std_gfe.run(root,cache_only=ctx.config.cache_only,workers=workers,render_figures=False)
        b.run_beta_sensitivity(ctx,render_figures=False)
        dw.run(ctx,render_figures=False)
        inputs=_read_inputs(ctx)
    paths=[weights(ctx,inputs['continuous']),regularization(ctx,inputs['continuous'],inputs['std']),
           beta_choice(ctx,inputs['beta'])]
    _json(output(ctx)/'active_figures.json',dict(schema=SCHEMA,mode='full' if ctx.config.full else 'smoke',
        figure_stems=STEMS[:3],figures=[str(p.relative_to(root)) for p in paths],
        reference='frozen same-seed nominal GFE; RH panel uses frozen FE and RH',
        focused_core_completed=True,report_only_smoke_replay=saved_smoke_replay,
        archived_exploration_executed=False))
    export_tables(ctx)
    return paths


def run_optimizer(ctx):
    from . import reviewer_robustness as r
    root=Path(ctx.memo['notebook_root'])
    stamp=root/'focused_smoke_input_manifest.json'
    saved_complete=False
    if stamp.exists():
        saved_record=json.loads(stamp.read_text())
        saved_complete=all((root/name).is_file() for name in saved_record['input_content_id'])
    if ctx.config.full or ctx.memo.get('regenerate_smoke_analysis', False) or not saved_complete:
        rc=r.context(root,shared=ctx)
        r.optimizer_robustness(rc,optimizers=('BFGS','L-BFGS-B','Adam'))
    else:
        record=json.loads((root/'focused_smoke_input_manifest.json').read_text())
        for name,content_id in {**record['source_content_id'],**record['input_content_id']}.items():
            if _content_id(root/name)!=content_id:raise RuntimeError('Changed frozen smoke source/input: '+name)
    path=optimizers(ctx,_read_inputs(ctx)['optimizer'])
    _json(optimizer_output(ctx)/'active_figures.json',dict(schema=SCHEMA,section='F',figure='F1',
        mode='full' if ctx.config.full else 'smoke',figure_path=str(path.relative_to(root))))
    export_tables(ctx)
    return path


def run_core(ctx,workers=4):
    # Compatibility helper; notebook uses separate C and F analysis cells.
    return [*run_sensitivity(ctx,workers),run_optimizer(ctx)]


def export_tables(ctx):
    frames=_read_inputs(ctx);out=output(ctx)/'tables';out.mkdir(exist_ok=True)
    f=frames['continuous'];rows=[]
    for (parameter,task,method),q in f.groupby(['parameter','task','method'],sort=False):
        rows.append(dict(parameter=parameter,task=task,method=method,values='; '.join(map(str,sorted(q.parameter_value.unique(),key=str))),
            n_seeds=q.seed.nunique(),n_records=len(q),native_stops_met=int(q.raw_gtol_met.astype(bool).sum()),
            locally_restoring=int(q.all_locally_restoring.fillna(False).astype(bool).sum()),unresolved=int(q.worst_displacement_a.isna().sum()),
            field_cosine_min=q.field_cosine_to_fixed_GFE.min(),field_cosine_max=q.field_cosine_to_fixed_GFE.max(),
            displacement_a_min=q.worst_displacement_a.min(),displacement_a_max=q.worst_displacement_a.max(),
            displacement_m_min=q.worst_displacement_m.min(),displacement_m_max=q.worst_displacement_m.max(),
            terminal_iterations_min=q.iterations.min(),terminal_iterations_max=q.iterations.max()))
    pd.DataFrame(rows).to_csv(out/'C_all_parameter_sensitivity.csv',index=False)
    if 'optimizer' in frames:
        fout=optimizer_output(ctx)/'tables';fout.mkdir(exist_ok=True)
        op=frames['optimizer'];op=op[op.optimizer.isin(('BFGS','L-BFGS-B','Adam'))]
        op.to_csv(fout/'F1_optimizer_raw.csv',index=False)
        summary=op.groupby(['method','optimizer']).agg(n=('seed','size'),native_stops_met=('raw_gtol_met','sum'),
            locally_restoring=('all_locally_restoring','sum'),time_to_return_s=('solve_wall_s','median'),
            evaluations=('evaluations','median'),iterations=('iterations','median'),
            field_cosine=('field_cosine_to_fixed_GFE','median'),displacement_a=('worst_displacement_a','median'),
            displacement_m=('worst_displacement_m','median'),
            gradient_inf=('terminal_gradient_inf','median')).reset_index()
        summary.to_csv(fout/'F1_optimizer_summary.csv',index=False)
    for key,name in [('continuous','C1_C2_continuous_raw'),('std','C2_STD_and_ablations_raw'),('beta','C3_RH_beta_raw')]:
        frames[key].to_csv(out/(name+'.csv'),index=False)
    directional=base(ctx)/'directional_review_data/C4_expanded_directional_weights.csv'
    if directional.exists():shutil.copy2(directional,out/'C_directional_weights_raw.csv')
    sources=[]
    for folder in (base(ctx)/'reviewer_robustness',base(ctx)/'std_sensitivity',base(ctx)/'beta_review_data',base(ctx)/'directional_review_data'):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.csv','.npz','.json'} and not any(p in {'pressure_cache','mechanical_cache','sensitivity_cache'} for p in path.parts):
                sources.append(dict(path=str(path.relative_to(Path(ctx.memo['notebook_root']))),bytes=path.stat().st_size,content_id=_content_id(path),
                    section='F' if any(x in path.parts for x in ('optimizer','solver_controls')) else 'C',
                    scope='current figure/table evidence' if folder.name!='reviewer_robustness' or any(x in path.parts for x in ('continuous','optimizer')) else 'retained exploration; inspect protocol and mode'))
    index=pd.DataFrame(sources)
    index[index.section.eq('C')].to_csv(out/'C_exportable_data_index.csv',index=False)
    fdir=optimizer_output(ctx)/'tables';fdir.mkdir(exist_ok=True)
    index[index.section.eq('F')].to_csv(fdir/'F_exportable_data_index.csv',index=False)
    _json(output(ctx)/'scope.json',dict(schema=SCHEMA,sensitivity_figures=3,optimizer_figures=1,main_figures=8,total_scientific_figures=22,
        displacement_display_unit='m',displacement_conversion='d_m = (d/a) * ElasticSphere().radius_m; original normalized columns retained',
        numerical_grid_and_window='Tabulated; original standalone equilibrium section retained',
        large_alpha_jumps_solver_knobs='Archived data only; not rerun in the focused default',
        generality='ACTIVE: Appendix G1-G9 (integrated separately by unified_study)'))
    return str(out)


def ordered_figures(ctx):
    from . import reviewer_focused
    import run_notebook
    root=Path(ctx.memo['notebook_root']);main=Path(ctx.output_root)/'figures'
    if ctx.config.full:
        import production_reporting
        return [(row['figure'],Path(ctx.output_root)/row['pdf']) for row in production_reporting.figure_inventory(ctx.output_root)]
    paths=[]
    for number in range(1,9):
        matches=list(main.glob(f'Figure_{number}_*.pdf'))
        if len(matches)!=1:raise RuntimeError(f'Expected current main Figure {number}, found {matches}')
        paths.append((str(number),matches[0]))
    for identifier in ('A1','A2','B1','B2','B3'):
        paths.append((identifier,run_notebook.appendix_figure_paths(ctx,identifier)[0].with_suffix('.pdf')))
    paths.extend((f'C{i+1}',output(ctx)/'figures'/(stem+'.pdf')) for i,stem in enumerate(STEMS[:3]))
    for identifier in ('D1','D2','D3','E1','E2'):
        paths.append((identifier,run_notebook.appendix_figure_paths(ctx,identifier)[0].with_suffix('.pdf')))
    paths.append(('F1',optimizer_output(ctx)/'figures'/(STEMS[3]+'.pdf')))
    return paths


def figures_pdf(ctx,destination=None):
    from pypdf import PdfReader,PdfWriter
    root=Path(ctx.memo['notebook_root']);paths=ordered_figures(ctx);writer=PdfWriter();index=[]
    if len(paths)!=22:raise RuntimeError('Expected exactly 22 scientific figures')
    for n,(identifier,path) in enumerate(paths,1):
        if len(PdfReader(path).pages)!=1:raise RuntimeError('Not a single figure PDF: '+str(path))
        writer.append(path,import_outline=False);writer.add_outline_item('Figure '+identifier,n-1)
        index.append(dict(page=n,figure=identifier,pdf=str(path.relative_to(root)),content_id=_content_id(path)))
    writer.add_metadata({'/Title':'GFE - main figures and focused appendices','/Subject':'22 figures in notebook order; '+('full' if ctx.config.full else 'cached smoke evidence')})
    path=Path(destination) if destination else root/'RINENG_GFE_figures_in_order.pdf'
    b=BytesIO();writer.write(b);path.write_bytes(b.getvalue())
    _json(output(ctx)/'figure_sequence.json',index)
    return str(path)


def finalize(ctx):
    """Complete exports, with scientific figures and parameters in separate PDFs."""
    root=Path(ctx.memo['notebook_root']);export_tables(ctx)
    if ctx.config.full:
        import production_reporting as pr
        params=pr.parameters(root,ctx);data=pr.export_data(ctx.output_root)
    else:
        from . import parameter_appendix
        import run_study
        params=parameter_appendix.render(root);data=run_study.data_exports(root)
    return dict(mode='full' if ctx.config.full else 'smoke',pdf=figures_pdf(ctx),
        summary_pdf=summary_pdf(ctx),parameters=params,exported_files=len(data),
        scientific_figures=22,sensitivity_figures=3,optimizer_figures=1,
        generality='ACTIVE: Appendix G1-G9 (integrated separately by unified_study)')


def summary_pdf(ctx,destination=None):
    from .reviewer_focused_report import write_report
    return write_report(ctx,destination)
