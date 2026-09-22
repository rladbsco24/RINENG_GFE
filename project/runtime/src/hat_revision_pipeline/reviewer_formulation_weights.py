"""Expanded C4 using the unchanged deposited Twin/Bottle formulations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import reviewer_beta as b

ETAS=(.0001,.001,.003,.01,.03,.1,1.)


def summarize(frame,data):
    rows=[]
    for (shape,method,eta),group in frame.groupby(['requested','method','eta']):
        for metric in ('iterations','gradient_l2','gradient_inf','command_cosine_fixed_FE','xz_amplitude_cosine_fixed_FE'):
            vals=group[metric].to_numpy(float);rng=np.random.default_rng(20260908)
            ci=[np.nan,np.nan]
            if len(vals)>1:ci=np.quantile(np.median(vals[rng.integers(0,len(vals),(10000,len(vals)))],axis=1),[.025,.975])
            rows.append(dict(requested=shape,method=method,eta=eta,metric=metric,n_pairs=len(vals),
                median=np.median(vals),q25=np.quantile(vals,.25),q75=np.quantile(vals,.75),minimum=vals.min(),maximum=vals.max(),
                ci95_lo=ci[0],ci95_hi=ci[1],bootstrap_seed=20260908,bootstrap_resamples=10000,
                sampling_unit='independent paired initial seed; fixed same-seed FE reference'))
    pd.DataFrame(rows).to_csv(Path(data)/'C4_summary_median_IQR_range_CI.csv',index=False)


def run(ctx,root=None,output=None,render_figures=True):
    root=Path(root or ctx.memo['notebook_root']);full=ctx.config.full
    module=b._beta_module(root)
    out=Path(output) if output is not None else (Path(ctx.output_root)/'appendices/C_GFE' if full else root/'appendix_outputs/C_GFE')
    data=out/'directional_review_data';data.mkdir(parents=True,exist_ok=True)
    roots=[data,root/'appendix_B_pressure',root/'production_outputs/appendices/B_pressure']
    seedlist=list(range(260905,260915)) if full else [260905]
    rows=[];added=0
    for seed in seedlist:
        for sign,shape in (('corrected_minus','Twin'),('historical_plus','Bottle')):
            config=module.configuration(shape,sign,'FE',.01,seed=seed)
            ref,arrays,source,new=b._beta_load(config,module,roots,data,not ctx.config.cache_only);added+=int(new)
            pref,axis=b._beta_field(config,ref,arrays,module,roots,data);phiref=arrays['terminal_phase_rad']
            for method in ('FE','RH','Conventional'):
                for eta in ETAS:
                    cfg=module.configuration(shape,sign,method,eta,seed=seed)
                    row,ar,source,new=b._beta_load(cfg,module,roots,data,not ctx.config.cache_only);added+=int(new)
                    pc,axis=b._beta_field(cfg,row,ar,module,roots,data);phase=ar['terminal_phase_rad']
                    terms=b._beta_terms(module,cfg,phase)
                    result=dict(row,requested=shape,eta=eta,method=method,seed=seed,
                        command_cosine_fixed_FE=float(abs(np.mean(np.exp(1j*(phase-phiref))))),
                        xz_amplitude_cosine_fixed_FE=float(np.vdot(abs(pc).ravel(),abs(pref).ravel()).real/(np.linalg.norm(pc)*np.linalg.norm(pref))),
                        reference_phase_content_id=ref['phase_content_id'],source=source,**terms)
                    rows.append(result)
                    key=module.cache_key(cfg)
                    np.savez_compressed(data/(key+'_export.npz'),phase_rad=phase,initial_phase_rad=ar['initial_phase_rad'],
                        pressure_complex_pa=pc,reference_pressure_complex_pa=pref,reference_phase_rad=phiref,axis_m=axis,
                        history_objective=ar.get('history_objective',np.array([])),history_gradient_norm=ar.get('history_gradient_norm',np.array([])))
    frame=pd.DataFrame(rows);frame.to_csv(data/'C4_expanded_directional_weights.csv',index=False);summarize(frame,data)
    name='Figure_C4_Twin_Bottle_directional_sensitivity'
    if render_figures:
        with plt.rc_context(module.STYLE):
            fig,axes=plt.subplots(2,2,figsize=(11,8.8));fig.subplots_adjust(hspace=.42,wspace=.32,bottom=.18)
            for i,shape in enumerate(('Twin','Bottle')):
                for method in ('FE','RH','Conventional'):
                    q=frame[frame.requested.eq(shape)&frame.method.eq(method)]
                    for j,metric in enumerate(('command_cosine_fixed_FE','xz_amplitude_cosine_fixed_FE')):
                        group=q.groupby('eta')[metric];ax=axes[i,j]
                        ax.plot(ETAS,group.median().reindex(ETAS),color=module.COLORS[method],marker=module.MARKERS[method],ms=4.5,lw=1.7,label=method)
                        ax.fill_between(ETAS,group.quantile(.25).reindex(ETAS).to_numpy(),group.quantile(.75).reindex(ETAS).to_numpy(),color=module.COLORS[method],alpha=.1)
                for j,ax in enumerate(axes[i]):
                    ax.text(-.14,1.045,f'({chr(97+2*i+j)})',transform=ax.transAxes,fontweight='bold')
                    ax.set_xscale('log');ax.set_xlabel(r'$\eta$');ax.set_ylim(0,1.035);ax.grid(alpha=.18)
                    ax.set_ylabel(('Command cosine' if j==0 else 'XZ amplitude cosine')+' to fixed FE')
                axes[i,0].text(.03,.07,shape,transform=axes[i,0].transAxes)
            handles=[Line2D([],[],color=module.COLORS[m],marker=module.MARKERS[m],ms=4.5,lw=1.7,label=m) for m in ('FE','RH','Conventional')]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.025),ncol=3,frameon=False)
            for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=240,bbox_inches='tight',facecolor='white')
            plt.close(fig)
    caption=(f"Seven trace-preserving directional ratios eta={','.join(f'{x:g}' for x in ETAS)}, with {len(seedlist)} paired starts. "
        "Rows retain only standard-sign Twin and alternative-sign Bottle. The same-seed nominal FE command at eta=0.01 is frozen before the sweep; "
        "the method-specific alpha, pressure weight and curvature trace are unchanged from B. Command and XZ amplitude correspondence do not require a new embedding or clustering. "
        "Native stopping iterations, actual SI-weighted objective/gradient contributions, reference hashes and complex fields accompany the figure. "
        "The reference is a formulation control, not a claim of globally optimal Bottle mechanics. "+('FULL.' if full else 'SMOKE.'))
    (data/'caption.txt').write_text(caption)
    (data/'protocol.json').write_text(json.dumps(dict(eta=ETAS,seeds=seedlist,new_solves=added,reference='same-seed nominal FE, eta0.01',mode='full' if full else 'smoke'),indent=2))
    path=out/'C4_C5_figure_captions.json';captions=json.loads(path.read_text()) if path.exists() else {}
    captions[name]=caption;path.write_text(json.dumps(captions,indent=2))
    return dict(rows=len(frame),new_solves=added,figure=str(out/(name+'.png')) if render_figures else None,data=str(data))
