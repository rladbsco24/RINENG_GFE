"""Replacement C5: convergence and correspondence to frozen RH/FE endpoints.

The four deposited settings are reused. Both modes cover ten pressure weights;
full mode uses ten paired starts with the unchanged objective and native solver.
"""
from pathlib import Path
import importlib.util

from rineng_content_id import content_identity
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BETA_SETTINGS = {
    "smoke_factors": [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    "full_factors": [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    "smoke_seeds": [260905],
    "full_seeds": list(range(260905, 260915)),
    "bootstrap_seed": 20260907,
    "bootstrap_resamples": 10000,
}


def _beta_module(root):
    path = Path(root)/"appendix_B_pressure/sensitivity_b.py"
    spec = importlib.util.spec_from_file_location("reviewer_sensitivity_b",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _beta_load(config, module, roots, destination, allow_new_solves):
    """Strictly match physics, source hashes, initialization and native solver."""
    key = module.cache_key(config)
    for directory in roots:
        base = directory/"data/sensitivity_cache"/key
        if base.with_suffix(".json").exists() and base.with_suffix(".npz").exists():
            row, arrays = module.load_run(config,directory)
            return row,arrays,str(base),False
    for directory in roots:
        for source in sorted((directory/"data/sensitivity_cache").glob("*.json")):
            row = json.loads(source.read_text())
            if not module.compatible(config,row) or not source.with_suffix(".npz").exists():
                continue
            with np.load(source.with_suffix(".npz"),allow_pickle=False) as z:
                arrays = {k:z[k].copy() for k in z.files}
            if content_identity(arrays["terminal_phase_rad"].tobytes()).hexdigest()!=row["phase_content_id"]:
                raise ValueError(f"Invalid source command: {source}")
            expected = module.gauge_full(np.random.default_rng(config["seed"]).uniform(-np.pi,np.pi,255))
            if not np.array_equal(expected,arrays["initial_phase_rad"]):
                raise ValueError("Imported beta run has a different initial command")
            return row,arrays,str(source.with_suffix("")),False
    if not allow_new_solves:
        raise FileNotFoundError(f"Missing beta/initialization run: {key}. No optimizer was started.")
    module.solve_one(config,destination)
    row,arrays = module.load_run(config,destination)
    return row,arrays,str(destination/"data/sensitivity_cache"/key),True


def _beta_field(config,row,arrays,module,roots,destination):
    """Reuse deposited complex XZ fields by phase hash; otherwise sample once."""
    axis = np.linspace(-1.25*343/40000,1.25*343/40000,181)
    wanted = row["phase_content_id"]
    for directory in roots:
        archive = directory/"data/sensitivity_pressure_slices.npz"
        provenance = directory/"data/sensitivity_pressure_provenance.json"
        if archive.exists() and provenance.exists():
            meta = json.loads(provenance.read_text())
            keys = [key for key,value in meta.items() if isinstance(value,dict)
                    and value.get("phase_content_id")==wanted
                    and value.get("model_content_id")==config["source_content_id"]["gorkov_core.py"]
                    and value.get("target_m")==config["target_m"]
                    and value.get("frequency_hz")==config["frequency_hz"]]
            if keys:
                with np.load(archive,allow_pickle=False) as z:
                    if np.array_equal(z["axis_m"],axis) and keys[0] in z.files:
                        return z[keys[0]].copy(),axis
    stamp = dict(phase=wanted,source=config["source_content_id"]["gorkov_core.py"],
                 target=config["target_m"],frequency=config["frequency_hz"],axis=axis.tolist())
    key = content_identity(json.dumps(stamp,sort_keys=True).encode()).hexdigest()
    path = destination/"pressure_cache"/(key+".npz")
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            return z["pressure"].copy(),z["axis_m"].copy()
    geometry = module.ArrayGeometry.square(config["side"],config["pitch_m"])
    xx,zz = np.meshgrid(axis,axis,indexing="xy")
    points = np.stack([xx,np.zeros_like(xx),zz+.05],axis=-1).reshape(-1,3)
    field = np.concatenate([module.pressure_field(arrays["terminal_phase_rad"],points[i:i+1024],geometry,config["frequency_hz"])
                            for i in range(0,len(points),1024)]).reshape(xx.shape)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,pressure=field,axis_m=axis,metadata_json=json.dumps(stamp))
    return field,axis


def _beta_terms(module,config,phase):
    obj = module.objective(config)
    raw = obj._raw_metrics(phase,True)
    terms = {"curvature_term_n_m":-float(raw["weighted_curvature_n_m"]),
             "force_term_n_m":config["alpha_per_m"]*float(raw["force_residual_norm_n"]),
             "pressure_term_n_m":config["beta_curvature_per_pa"]*float(raw["pressure_penalty_pa"])}
    gradients = (-np.asarray(raw["d_weighted_curvature_d_phase"]),
        config["alpha_per_m"]*np.asarray(raw["d_force_norm_d_phase"]),
        config["beta_curvature_per_pa"]*np.asarray(raw["d_pressure_penalty_d_phase"]))
    for tag,gradient in zip(("curvature","force","pressure"),gradients):
        terms[tag+"_gradient_l2"] = float(np.linalg.norm(np.delete(gradient,obj.gauge_index)))
    reduced = np.delete(sum(gradients),obj.gauge_index)
    terms.update(gradient_inf=float(np.linalg.norm(reduced,np.inf)),gradient_l2=float(np.linalg.norm(reduced)),
                 objective_n_m=float(sum(terms[k] for k in ("curvature_term_n_m","force_term_n_m","pressure_term_n_m"))))
    return terms


def _beta_plot(table,out):
    factors = sorted(table.beta_factor.unique())
    positions = np.arange(len(factors))
    labels = ["0\n(FE)" if f==0 else f"{f:g}" for f in factors]
    colors = {"RH":"#87569B","FE":"#C78D25"}
    style = {"font.family":"DejaVu Sans","font.size":12,"axes.labelsize":12,
             "axes.labelweight":"semibold","axes.linewidth":1.5,"xtick.labelsize":11,
             "ytick.labelsize":11,"pdf.fonttype":42,"svg.fonttype":"none"}
    with plt.rc_context(style):
        fig,axs = plt.subplots(2,2,figsize=(9.4,7.7))
        fig.subplots_adjust(left=.105,right=.98,bottom=.11,top=.955,wspace=.31,hspace=.43)
        def curve(ax,metric,color,marker,label=None):
            for _,q in table.groupby("seed"):
                q = q.set_index("beta_factor").reindex(factors)
                if table.seed.nunique()>1:
                    ax.plot(positions,q[metric],color=color,lw=.75,alpha=.18)
            g = table.groupby("beta_factor")[metric]
            median=g.median().reindex(factors);lo=g.quantile(.25).reindex(factors);hi=g.quantile(.75).reindex(factors)
            ax.errorbar(positions,median,yerr=[median-lo,hi-median],fmt=marker+"-",color=color,
                        mfc="white",mew=1.4,ms=5.2,lw=1.8,capsize=3,label=label)
        curve(axs[0,0],"iterations",colors["RH"],"o")
        for reason,marker in [("precision loss","x"),("iteration cap","v"),("other stop","D")]:
            q = table[table.stop_reason.eq(reason)]
            if len(q):
                axs[0,0].scatter([factors.index(v) for v in q.beta_factor],q.iterations,
                                 color="#B33B35",marker=marker,s=32,zorder=6,label=reason.capitalize())
        if table.stop_reason.ne("native convergence").any():
            axs[0,0].legend(frameon=False,fontsize=9)
        axs[0,0].set_ylabel("Terminal iteration")
        axs[0,0].set_ylim(bottom=0)
        for ax,metric,label in [(axs[0,1],"command_similarity","Command similarity"),
                               (axs[1,0],"xz_field_similarity","XZ field similarity")]:
            for ref,marker in [("FE","s"),("RH","^")]:
                curve(ax,metric+"_to_fixed_"+ref.lower(),colors[ref],marker,
                      r"Fixed FE ($\beta=0$)" if ref=="FE" else r"Fixed RH ($\beta_0$)")
            ax.set_ylabel(label)
            columns=[metric+"_to_fixed_"+ref for ref in ("fe","rh")]
            low = max(0.,np.floor((table[columns].min().min()-.03)*10)/10)
            ax.set_ylim(low,1.015)
        axs[0,1].legend(frameon=False,fontsize=10,loc="lower left",handlelength=2)
        curve(axs[1,1],"target_pressure_over_shared_peak",colors["RH"],"o")
        axs[1,1].set_ylabel(r"Target $|p|/p_{\max}$")
        axs[1,1].set_ylim(bottom=-.01)
        for i,ax in enumerate(axs.flat):
            ax.set_xticks(positions,labels)
            ax.set_xlabel(r"Pressure weight $\beta/\beta_0$")
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(length=4.5,width=1.3)
            ax.grid(axis="y",color=".88",lw=.7)
            ax.set_axisbelow(True)
            ax.text(-.12,1.04,f"({chr(97+i)})",transform=ax.transAxes,fontweight="bold",fontsize=13)
        # Keep the C5 filename so the existing complete-figure exporter picks it up.
        paths=[]
        for ext in ("png","pdf","svg"):
            path=out/("Figure_C5_RH_pressure_sensitivity."+ext)
            fig.savefig(path,dpi=300,facecolor="white")
            paths.append(path)
        plt.close(fig)
    return paths


def run_beta_sensitivity(ctx,root=None,settings=None,allow_new_solves=None,output=None,render_figures=True):
    settings=dict(BETA_SETTINGS,**(settings or {}))
    root=Path(root or ctx.memo["notebook_root"]).resolve()
    full=bool(ctx.config.full)
    factors=settings["full_factors" if full else "smoke_factors"]
    seeds=settings["full_seeds" if full else "smoke_seeds"]
    if not {0.,.1,1.,10.}.issubset(factors):
        raise ValueError("Keep 0, 0.1, 1 and 10 to preserve references and the frozen pressure normalization")
    if allow_new_solves is None:
        allow_new_solves=not ctx.config.cache_only
    module=_beta_module(root)
    out=Path(output) if output else (Path(ctx.output_root)/"appendices/C_GFE" if full else root/"appendix_outputs/C_GFE")
    out.mkdir(parents=True,exist_ok=True)
    data=out/"beta_review_data"
    data.mkdir(exist_ok=True)
    production_base=Path(ctx.output_root)/"appendices/B_pressure"
    roots=[data,production_base,root/"appendix_B_pressure"]
    rows=[];history=[];terms=[];arrays_export={};added=0
    for seed in seeds:
        runs={}
        for factor in factors:
            config=module.configuration("Bottle","historical_plus","FE" if factor==0 else "RH",.01,factor,seed=seed)
            if full:
                from hat_revision_pipeline.production_ab import SCHEMA
                config=dict(config,version=SCHEMA,status="PRODUCTION")
            row,arrays,source,new=_beta_load(config,module,roots,data,allow_new_solves)
            field,axis=_beta_field(config,row,arrays,module,roots,data)
            runs[factor]=(config,row,arrays,field,source)
            added+=int(new)
        initial=runs[1.][2]["initial_phase_rad"]
        if not all(np.array_equal(initial,run[2]["initial_phase_rad"]) for run in runs.values()):
            raise ValueError("The beta comparison is not paired on the initial command")
        # Freeze normalization to the original four-point sweep when adding 0.3/3.
        peak=max(float(np.abs(runs[f][3]).max()) for f in (0.,.1,1.,10.))
        for factor,(config,source_row,arrays,field,source) in runs.items():
            stop=("native convergence" if source_row["success"] else "iteration cap" if source_row["iterations"]>=config["maxiter"]
                  else "precision loss" if source_row["status_code"]==2 else "other stop")
            row=dict(seed=seed,method="FE" if factor==0 else "RH",beta_factor=factor,beta_curvature_per_pa=config["beta_curvature_per_pa"],
                alpha_per_m=config["alpha_per_m"],iterations=int(source_row["iterations"]),
                optimizer_success=bool(source_row["success"]),stop_reason=stop,message=source_row["message"],
                native_status_code=int(source_row["status_code"]),gtol=config["gtol"],maxiter=config["maxiter"],
                gradient_l2=float(source_row["gradient_l2"]),command_time_s=float(source_row["command_time_s"]),
                target_pressure_pa=float(source_row["pressure_pa"]),shared_peak_pa=peak,
                target_pressure_over_shared_peak=float(source_row["pressure_pa"])/peak,
                phase_content_id=source_row["phase_content_id"],initial_content_id=source_row["initial_content_id"],
                source_cache=str(Path(source).relative_to(root)) if Path(source).is_relative_to(root) else source,
                sign="alternative",requested="Bottle",mode="full" if full else "smoke")
            for ref,ref_factor in [("fe",0.),("rh",1.)]:
                reference=runs[ref_factor]
                row["command_similarity_to_fixed_"+ref]=module.command_similarity(arrays["terminal_phase_rad"],reference[2]["terminal_phase_rad"])
                row["xz_field_similarity_to_fixed_"+ref]=module.cosine(field,reference[3])
                row["reference_"+ref+"_content_id"]=reference[1]["phase_content_id"]
            row["command_similarity_margin_rh_minus_fe"]=(row["command_similarity_to_fixed_rh"]-row["command_similarity_to_fixed_fe"])
            row["nearer_command_reference"]="RH" if row["command_similarity_margin_rh_minus_fe"]>0 else "FE"
            row["iterations_relative_to_nominal_rh"]=row["iterations"]/runs[1.][1]["iterations"]
            for point,phase in [("initial",arrays["initial_phase_rad"]),("terminal",arrays["terminal_phase_rad"])]:
                t=_beta_terms(module,config,phase)
                terms.append(dict(seed=seed,beta_factor=factor,evaluation=point,**t))
                if point=="terminal":
                    row["terminal_gradient_inf"]=t["gradient_inf"]
            rows.append(row)
            length=len(arrays["history_objective"])
            for k in range(length):
                history.append(dict(seed=seed,beta_factor=factor,iteration=k,
                    objective=float(arrays["history_objective"][k]),gradient_l2=float(arrays["history_gradient_l2"][k]),
                    elapsed_s=float(arrays["history_wall_s"][k])))
            tag=f"seed{seed}_beta{factor:g}"
            arrays_export[tag+"_phase_rad"]=arrays["terminal_phase_rad"]
            arrays_export[tag+"_xz_pressure_pa"]=field
        arrays_export[f"seed{seed}_initial_phase_rad"]=initial
        print(f"C5 seed {seed}: {len(factors)} independent beta settings, paired initialization",flush=True)
    table=pd.DataFrame(rows)
    table.to_csv(data/"C5_beta_results.csv",index=False)
    # Keep the reader-facing C5 table aligned with the replacement figure.
    current_table=out/"C5_RH_pressure_sensitivity.csv"
    old_table=data/"C5_previous_four_weight_table.csv"
    if current_table.exists() and not old_table.exists():
        old_table.write_bytes(current_table.read_bytes())
    table.to_csv(current_table,index=False)
    pd.DataFrame(history).to_csv(data/"C5_convergence_histories.csv",index=False)
    pd.DataFrame(terms).to_csv(data/"C5_objective_and_gradient_terms.csv",index=False)
    np.savez_compressed(data/"C5_exportable_fields_and_commands.npz",axis_m=axis,**arrays_export)
    rng=np.random.default_rng(settings["bootstrap_seed"])
    summary=[]
    metrics=["iterations","iterations_relative_to_nominal_rh","terminal_gradient_inf","command_similarity_to_fixed_rh",
             "command_similarity_to_fixed_fe","xz_field_similarity_to_fixed_rh","xz_field_similarity_to_fixed_fe",
             "target_pressure_over_shared_peak"]
    for factor,q in table.groupby("beta_factor"):
        indices=rng.integers(0,len(q),size=(settings["bootstrap_resamples"],len(q))) if len(q)>1 else None
        for metric in metrics:
            values=q[metric].to_numpy(float)
            low,median,high=np.quantile(values,[.25,.5,.75])
            ci=np.quantile(np.median(values[indices],axis=1),[.025,.975]) if indices is not None else [np.nan,np.nan]
            summary.append(dict(beta_factor=factor,metric=metric,n=len(q),median=median,q25=low,q75=high,
                ci95_low=ci[0],ci95_high=ci[1],native_convergence_count=int(q.optimizer_success.sum()),
                nearer_rh_reference_count=int(q.nearer_command_reference.eq("RH").sum())))
    pd.DataFrame(summary).to_csv(data/"C5_beta_summary.csv",index=False)
    paths=_beta_plot(table,out) if render_figures else []
    ctx.tables["reviewer_C5_beta_results"]=table
    ctx.tables["reviewer_C5_beta_summary"]=pd.DataFrame(summary)
    nominal=module.NOMINAL_BETA
    caption=("RH pressure-weight sensitivity for the alternative-sign Bottle formulation. "
        f"Only beta is varied: beta/beta0={','.join(f'{f:g}' for f in factors)}; beta0={nominal:.10g} N m^-1 Pa^-1, alpha=9 m^-1, "
        "eta=0.01, trace(W)=3, pressure smoothing=0.001, compact L-BFGS-B cap=10000 and native gtol=1e-8 for FE and RH. "
        "(a) Actual terminal iterations; circles show the aggregate and any nonconverged native stops are marked separately. "
        "(b) Gauge-invariant full-dimensional command cosine to fixed same-seed FE (beta=0) and RH (beta=beta0) endpoints. "
        "(c) XZ pressure-amplitude cosine to those same fixed references. (d) Central pressure normalized by the fixed shared peak over the original beta/beta0=0,0.1,1,10 sweep. "
        "The references are held fixed across beta. These continuous correspondences quantify endpoint selection without refitting an embedding or imposing a new clustering threshold. "
        f"All {len(seeds)} initializations are paired across beta, with independent cold starts and no continuation. "
        + ("Lines/markers show medians, error bars IQR and pale curves individual starts. " if len(seeds)>1 else "One initialization; no population interval is implied. ")
        + "Full gradient histories, native stopping reasons, paired iteration ratios, reference hashes and dimensional objective/gradient contributions are exported. "
        + ("FULL." if full else "SMOKE."))
    (data/"C5_caption.txt").write_text(caption+"\n")
    captions_path=out/"C4_C5_figure_captions.json"
    captions=json.loads(captions_path.read_text()) if captions_path.exists() else {}
    captions["Figure_C5_RH_pressure_sensitivity"]=caption
    captions_path.write_text(json.dumps(captions,indent=2))
    (out/"C4_C5_captions.txt").write_text("\n\n".join(f"{k}. {v}" for k,v in captions.items())+"\n")
    (data/"parameters.json").write_text(json.dumps(dict(settings=settings,nominal_beta=nominal,alpha_per_m=9.,
        sign="alternative Bottle",new_synthesis_runs=added,mode="full" if full else "smoke",configuration=module.configuration("Bottle","historical_plus","RH"),
        reference_policy="fixed within paired seed; never refit",pressure_normalization="original four beta settings per seed"),indent=2))
    lines=["## Pressure weight: convergence and endpoint selection", "",
        "The preceding C5 compared central pressure and similarity to Conventional. The replacement directly addresses convergence and variation relative to the nominal RH command, while retaining FE as the zero-pressure-weight control.","",
        "The dimensional objective is L = -W:H_U + alpha ||F_U|| + beta P_epsilon, where U is the alternative-sign synthesis potential in this Bottle test. Alpha has units m^-1; beta has units N m^-1 Pa^-1. Therefore the numerical statement 'six orders smaller' depends on the unit convention. Specify beta0 and the tested dimensionless multiplier beta/beta0, with the remaining dimensional coefficients fixed. The table of weighted objective and gradient contributions shows the actual relative role of the pressure term.","",
        "| beta/beta0 | n | Native convergence | Median iterations | Command cosine to fixed RH | Field cosine to fixed RH | Target pressure/shared peak |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for factor,q in table.groupby("beta_factor"):
        lines.append(f"| {factor:g} | {len(q)} | {int(q.optimizer_success.sum())}/{len(q)} | {q.iterations.median():.0f} | {q.command_similarity_to_fixed_rh.median():.6f} | {q.xz_field_similarity_to_fixed_rh.median():.6f} | {q.target_pressure_over_shared_peak.median():.6g} |")
    lines += ["",f"{len(seeds)} paired start(s); {added} new optimization runs in this execution. Nominal beta0={nominal:.10g} N m^-1 Pa^-1. Existing command times retain their original provenance; cache-loading time is not synthesis time.","",
        "Use the convergence panel together with the two reference-correspondence panels: weak beta may converge quickly to an FE-like pressure field; larger beta may preserve the nominal RH endpoint while taking more iterations. The executed values above determine the useful range. The expanded grid contains ten distinct settings from zero through 100 beta0, including values on both sides of the nominal choice. Full mode repeats all ten settings at ten paired starts. No continuous nonzero-beta GFE claim follows from this Bottle-specific RH test."]
    (data/"C5_measured_results.md").write_text("\n".join(lines)+"\n")
    (out/"C5_beta_measured_results.txt").write_text("\n".join(lines)+"\n")
    return dict(table=table,paths=paths,data_directory=data,new_synthesis_runs=added,caption=caption)

