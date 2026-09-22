"""One matched-origin warm-continuation control for the Figure 7 GFE examples.

The nominal GFE endpoint is frozen before the sweep. Cold runs reuse Figure 7;
warm runs sequentially start from the preceding accepted terminal command.
Both originate from the same nominal random seed. No nearest-reference fitting
or alternative optimizer is introduced.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import pipeline as p
from .cache import CacheStore, digest_array, digest_file
from .gorkov_core import SingleTargetObjective
from .main_feg import feg_single_objective, _solve as solve_nominal_gfe
from .sota import solve_corrected_gorkov_fe
from .appendix_c_feg import _source_validations
from .appendix_b3_mechanics import transverse_descriptor
from .cache import digest_payload

ALPHAS=(1.e3,1.e6,1.e7)
SCHEMA="GFE-matched-origin-warm-continuation-v1"
DISPLAY_SCHEMA="C3-warm-cold-visible-morphology-v1"
DISPLAY_POINTS=151


def _figure7(root):
    spec=importlib.util.spec_from_file_location("gfe_c3_figure7_source",root/"main_feg_benchmarks.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _row(run,mode,alpha,seed,initial,source,status):
    return dict(method="GFE",initialization=mode,alpha_per_m=float(alpha),seed=int(seed),
        phase_content_id=digest_array(run.phase_rad),initial_phase_content_id=digest_array(initial),
        iterations=int(run.iterations),evaluations=int(run.evaluations),maxiter=10000,
        objective=float(run.objective),terminal_gradient_l2=float(run.gradient_norm),
        optimizer_success=bool(run.success),optimizer_status=int(run.status),optimizer_message=str(run.message),
        command_time_s=float(run.history_wall_s[-1]),source=source,cache_status=status,evidence_status="SMOKE")


def _optimizer_stop_sentence(frame):
    """Report actual high-weight stops separately from mechanical root status."""
    selected=frame[frame.initialization.isin(("Cold","Warm"))]
    count=len(selected)
    precision_loss=int(selected.optimizer_message.astype(str).str.contains("precision loss",case=False,regex=False).sum())
    if count and precision_loss==count:
        return f"All {count} large-alpha solves ended with precision loss. "
    return f"Of {count} large-alpha solves, {precision_loss} ended with precision loss; individual stopping statuses are exported. "


def _save_arrays(output,job,reference_phase,points,pressure,reference_pressure,include_seed=False):
    row=job["row"];run=job["run"]
    name=f"C3_{row['initialization'].lower()}_alpha{row['alpha_per_m']:g}"
    if include_seed:name+=f"_seed{row['seed']}"
    np.savez_compressed(output/(name+".npz"),phase_rad=run.phase_rad,
        initial_phase_rad=job["initial"],history_phase_rad=run.history_phase_rad,
        history_objective=run.history_objective,history_gradient_norm=run.history_gradient_norm,
        history_wall_s=run.history_wall_s,points_m=points,pressure_complex_pa=pressure,
        reference_pressure_complex_pa=reference_pressure,reference_phase_rad=reference_phase,
        metadata_json=json.dumps(row,default=str))
    return name


def _display_fields(ctx,output,jobs,cache_only=False):
    """Sample unchanged commands to make pressure-shape retention visible."""
    rows=[];fields={}
    for job in jobs:
        row=job["row"];phase=np.asarray(job["run"].phase_rad)
        identity=dict(schema=DISPLAY_SCHEMA,phase=digest_array(phase),positions=digest_array(ctx.positions_m),
            target=p.MAIN_TARGET_M.tolist(),frequency_hz=ctx.config.frequency_hz,points=DISPLAY_POINTS,
            half_width_m=.007,model_content_id=digest_file(Path(p.__file__).with_name("exact_validator.py")))
        path=output/"cache"/("C3_display_"+digest_payload(identity)+".npz")
        if path.exists():
            with np.load(path,allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
        else:
            if cache_only:raise FileNotFoundError(f'C3 pressure display cache missing: {path}')
            field=p._array_field(ctx,phase);axis=np.linspace(-.007,.007,DISPLAY_POINTS)
            x,y=np.meshgrid(axis,axis,indexing="xy");points=np.broadcast_to(p.MAIN_TARGET_M,x.shape+(3,)).copy()
            points[...,0]+=x;points[...,1]+=y;flat=points.reshape(-1,3)
            pressure=np.concatenate([field.pressure(flat[i:i+2048]) for i in range(0,len(flat),2048)]).reshape(x.shape)
            descriptor,ring=transverse_descriptor(field,p.MAIN_TARGET_M,343./ctx.config.frequency_hz)
            values=dict(axis_m=axis,points_m=points,pressure_complex_pa=pressure,phase_rad=phase,
                **{"morphology_"+k:np.asarray(v) for k,v in descriptor.items()},
                **{"ring_"+k:v for k,v in ring.items()},identity_json=np.asarray(json.dumps(identity)))
            np.savez_compressed(path,**values)
        name=job["array_name"]
        np.savez_compressed(output/(name+"_pressure_xy.npz"),**values)
        descriptor={k.removeprefix("morphology_"):float(v) for k,v in values.items() if k.startswith("morphology_")}
        rows.append(dict(initialization=row["initialization"],alpha_per_m=row["alpha_per_m"],
            phase_content_id=digest_array(phase),**descriptor))
        fields[(row["initialization"],row["alpha_per_m"])]=values
    morphology=pd.DataFrame(rows);morphology.to_csv(output/"C3_pressure_morphology.csv",index=False)
    return fields,morphology


def _render(output,frame,fields):
    with plt.rc_context({"font.size":13,"axes.labelsize":14,"axes.labelweight":"semibold",
                         "axes.titleweight":"semibold","xtick.labelsize":12,"ytick.labelsize":12}):
        fig,grid=plt.subplots(2,2,figsize=(10.7,10.2))
        axes=grid[0]
        colors={"Cold":"#3B6FA1","Warm":"#007C78"}
        reference=frame[frame.initialization.eq("Reference")].iloc[0]
        for mode in ("Cold","Warm"):
            selected=frame[frame.initialization.eq(mode)].sort_values("alpha_per_m")
            alpha=np.r_[reference.alpha_per_m,selected.alpha_per_m.to_numpy()]
            for column,marker,style in (("command_cosine_to_fixed_GFE","o","-"),("field_cosine_to_fixed_GFE","s","--")):
                y=np.r_[1.,selected[column].to_numpy()]
                axes[0].plot(alpha,y,color=colors[mode],marker=marker,ls=style,lw=2.4,ms=7,
                             mfc="white" if marker=="s" else colors[mode],mew=1.5)
            displacement=np.r_[reference.finite_ka_displacement_a,selected.finite_ka_displacement_a.to_numpy()]
            axes[1].plot(alpha,displacement,color=colors[mode],marker="o",lw=2.4,ms=7,label=mode)
            failed=selected[~selected.finite_ka_root_found.astype(bool)]
            if len(failed):
                axes[1].plot(failed.alpha_per_m,np.full(len(failed),.95),"x",color=colors[mode],ms=10,mew=2,
                             transform=axes[1].get_xaxis_transform())
        axes[0].set_ylabel("Similarity to nominal GFE")
        low=float(frame[["command_cosine_to_fixed_GFE","field_cosine_to_fixed_GFE"]].min().min())
        axes[0].set_ylim(max(0.,low-.05),1.025)
        axes[1].set_ylabel(r"Elastic displacement / $a$")
        for letter,ax in zip("ab",axes):
            ax.set_xscale("log");ax.set_xlabel(r"Force weight $\alpha$ (m$^{-1}$)")
            ax.set_xticks([10,1e3,1e6,1e7]);ax.minorticks_off()
            ax.grid(alpha=.2,linewidth=.7);ax.set_axisbelow(True)
            ax.tick_params(width=1.4,length=5)
            for spine in ax.spines.values():spine.set_linewidth(1.4)
            ax.text(-.15,1.025,f"({letter})",transform=ax.transAxes,fontweight="bold",fontsize=16)
        handles=[Line2D([],[],color=colors[mode],lw=2.5,label=mode) for mode in ("Cold","Warm")]
        handles += [Line2D([],[],color=".25",marker="o",lw=2,ls="-",label="Command"),
                    Line2D([],[],color=".25",marker="s",mfc="white",lw=2,ls="--",label="Complex field")]
        if not frame.finite_ka_root_found.all():handles.append(Line2D([],[],color=".3",marker="x",ls="none",label="Unresolved root"))
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.552),ncol=4,frameon=False,fontsize=11)
        nominal=fields[("Reference",10.)];axis=nominal["axis_m"]*1000
        peak=max(float(abs(fields[key]["pressure_complex_pa"]).max()) for key in [("Reference",10.),("Cold",1e7),("Warm",1e7)])
        for letter,mode,ax in zip("cd",("Cold","Warm"),grid[1]):
            values=fields[(mode,1e7)];amplitude=abs(values["pressure_complex_pa"])/peak
            im=ax.imshow(amplitude,origin="lower",extent=[axis[0],axis[-1],axis[0],axis[-1]],cmap="magma",vmin=0,vmax=1)
            ax.contour(axis,axis,abs(nominal["pressure_complex_pa"])/peak,levels=[.25,.5,.75],colors="#71DFF4",linestyles="--",linewidths=1.25)
            ax.plot(0,0,"+",color="white",ms=9,mew=1.7)
            ax.set_title(mode+r" · $\alpha=10^7$ m$^{-1}$",fontsize=14,fontweight="semibold",pad=9)
            ax.set_xlabel(r"$x-x_0$ (mm)");ax.set_ylabel(r"$y-y_0$ (mm)")
            ax.set_xticks([-6,0,6]);ax.set_yticks([-6,0,6]);ax.set_aspect("equal")
            ax.text(-.15,1.025,f"({letter})",transform=ax.transAxes,fontweight="bold",fontsize=16)
        fig.subplots_adjust(left=.10,right=.98,top=.97,bottom=.18,wspace=.35,hspace=.60)
        cbax=fig.add_axes([.30,.047,.43,.015]);bar=fig.colorbar(im,cax=cbax,orientation="horizontal",ticks=[0,.5,1])
        bar.set_label("Pressure / shared peak",fontsize=11)
        fig.legend(handles=[Line2D([],[],color="#169AB7",lw=1.5,ls="--",label="Nominal GFE pressure contours")],
            loc="lower center",bbox_to_anchor=(.55,.083),frameon=False,fontsize=11)
        paths=[]
        for ext in ("png","pdf","svg"):
            path=output/f"Figure_C3_GFE_warm_continuation.{ext}";fig.savefig(path,dpi=230,facecolor="white",bbox_inches="tight");paths.append(str(path))
        plt.close(fig)
    return paths


def run(root,ctx=None):
    root=Path(root).resolve();output=root/"appendix_outputs/C_GFE";output.mkdir(parents=True,exist_ok=True)
    if ctx is None:ctx=p.prepare_pipeline(run_mode="quick",recompute=False,output_root=root/"smoke_outputs")
    cold=_figure7(root).alpha_bank(ctx);seed=int(cold["seed"])
    if tuple(cold["table"].alpha_per_m)!=ALPHAS:raise RuntimeError("Figure 7 alpha cases differ from this control")
    nominal_objective=feg_single_objective(ctx)
    original_initial=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(ctx.positions_m))
    nominal,nominal_status=solve_nominal_gfe(ctx,nominal_objective,original_initial,seed,"Figure1 paired central target")
    reference_phase=np.asarray(nominal.phase_rad).copy()
    c3=SimpleNamespace(config=ctx.config,positions_m=ctx.positions_m,normals=ctx.normals,recompute=False,
        output_root=output,data_root=ctx.data_root,
        cache=CacheStore(output/"cache",namespace=p.PIPELINE_IMPLEMENTATION,read_only=ctx.config.cache_only))
    source=Path(__file__).parent
    hashes={name:digest_file(source/name) for name in ("gorkov_core.py","sota.py","exact_validator.py")}
    jobs=[dict(run=nominal,initial=original_initial,row=_row(nominal,"Reference",10.,seed,original_initial,
        "Figure 1 same-seed nominal GFE",nominal_status))]
    for index,alpha in enumerate(ALPHAS):
        run=cold["runs"][index];source_row=cold["table"].iloc[index]
        if source_row.initial_phase_content_id!=digest_array(original_initial):
            raise RuntimeError("Cold Figure 7 initial phases do not match the nominal reference origin")
        jobs.append(dict(run=run,initial=original_initial,row=_row(run,"Cold",alpha,seed,original_initial,
            str(source_row.cache_path),"reused Figure 7 command")))
    initial=reference_phase.copy();previous_alpha=10.
    for alpha in ALPHAS:
        method=replace(nominal_objective.method,alpha_per_m=alpha)
        objective=SingleTargetObjective(nominal_objective.condition,method,force_target_n=nominal_objective.force_target_n)
        payload=dict(contract=SCHEMA,method=asdict(method),force_target_n=objective.force_target_n.tolist(),
            target_m=p.MAIN_TARGET_M.tolist(),positions=digest_array(ctx.positions_m),normals=digest_array(ctx.normals),
            frequency_hz=ctx.config.frequency_hz,seed=seed,original_initial_content_id=digest_array(original_initial),
            initial_phase_content_id=digest_array(initial),previous_alpha=previous_alpha,
            reference_phase_content_id=digest_array(reference_phase),maxiter=10000,gtol=ctx.config.gtol,source_content_id=hashes)
        current_initial=initial.copy()
        run,status,path=c3.cache.get_or_compute("C3_gfe_warm_run",payload,
            lambda objective=objective,initial=current_initial:solve_corrected_gorkov_fe(objective,initial,maxiter=10000,gtol=ctx.config.gtol))
        row=_row(run,"Warm",alpha,seed,current_initial,str(path.relative_to(root)),status)
        row["previous_alpha_per_m"]=previous_alpha
        jobs.append(dict(run=run,initial=current_initial,row=row))
        initial=np.asarray(run.phase_rad).copy();previous_alpha=alpha
        print(f"C3 warm alpha={alpha:g}: {status}, {run.iterations} iterations, gradient={run.gradient_norm:.4g}",flush=True)
    lam=343./ctx.config.frequency_hz
    offsets=np.array(np.meshgrid(*([np.linspace(-.75*lam,.75*lam,9)]*3),indexing="ij")).reshape(3,-1).T
    points=p.MAIN_TARGET_M+offsets
    reference_pressure=p._array_field(ctx,reference_phase).pressure(points)
    for job in jobs:
        phase=np.asarray(job["run"].phase_rad)
        payload=dict(contract=SCHEMA+"-field",phase=digest_array(phase),points=digest_array(points),
            positions=digest_array(ctx.positions_m),normals=digest_array(ctx.normals),frequency_hz=ctx.config.frequency_hz,
            source_content_id=hashes)
        pressure,_,_=c3.cache.get_or_compute("C3_pressure_volume",payload,
            lambda phase=phase:p._array_field(ctx,phase).pressure(points))
        command_cosine=float(np.clip(abs(np.mean(np.exp(1j*(phase-reference_phase)))),0.,1.))
        field_cosine=float(np.clip(abs(np.vdot(reference_pressure,pressure))/(np.linalg.norm(reference_pressure)*np.linalg.norm(pressure)),0.,1.))
        job["row"].update(command_cosine_to_fixed_GFE=command_cosine,field_cosine_to_fixed_GFE=field_cosine,
            command_projector_distance=float(np.sqrt(max(0.,1.-command_cosine**2))),
            reference_phase_content_id=digest_array(reference_phase),reference_seed=seed,
            reference_alpha_per_m=10.,field_point_count=len(points),force_target_z_n=float(nominal_objective.force_target_n[2]))
        job["array_name"]=_save_arrays(output,job,reference_phase,points,pressure,reference_pressure)
    existing=_source_validations(root,c3)
    def validate(job):
        phase=np.asarray(job["run"].phase_rad);key=(digest_array(phase),tuple(p.MAIN_TARGET_M))
        if key in existing:validation,provenance=existing[key]
        else:
            validation=p._finite_ka_validation(c3,phase,p.MAIN_TARGET_M,key=job["array_name"])
            provenance="current same-protocol C3 elastic cache"
        row=p._static_validation_row("GFE",validation)
        row.update(job["row"],validation_source=provenance)
        np.savez_compressed(output/(job["array_name"]+"_mechanics.npz"),phase_rad=phase,
            target_m=p.MAIN_TARGET_M,equilibrium_m=validation.equilibrium.equilibrium_m,
            force_jacobian_n_m=validation.force_jacobian_n_m,
            symmetric_stiffness_eigenvalues_n_m=validation.symmetric_stiffness_eigenvalues_n_m,
            protocol_json=json.dumps(p._finite_ka_protocol(c3),default=str))
        print(f"C3 mechanics {job['array_name']}: root={row['finite_ka_root_found']}, d/a={row['finite_ka_displacement_a']:.5g}",flush=True)
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(validate,jobs))
    frame=pd.DataFrame(rows);frame.to_csv(output/"C3_warm_cold_results.csv",index=False)
    frame[frame.initialization.eq("Reference")].to_csv(output/"C3_fixed_reference.csv",index=False)
    fields,morphology=_display_fields(ctx,output,jobs)
    paths=_render(output,frame,fields)
    stop_sentence=_optimizer_stop_sentence(frame)
    caption=("Figure C3. Initialization dependence of large-weight GFE solutions. Cold commands reuse the three Figure 7 cases; "
        "warm continuation starts from the frozen same-seed nominal alpha=10 m^-1 GFE endpoint and sequentially solves alpha=10^3,10^6,10^7 m^-1. "
        f"Both originate from seed {seed}; each solve has a 10,000-iteration cap and retains its actual stopping status. "
        +stop_sentence+
        "(a) Gauge-invariant command cosine and normalized complex-pressure cosine relative to the same fixed nominal GFE command. "
        "The complex field is sampled on a 9 x 9 x 9 volume spanning +/-0.75 wavelength about the prescribed Single target. "
        "(b) Displacement to the independent finite-ka elastic total-force equilibrium, normalized by bead radius a=0.65 mm. "
        "(c,d) Terminal cold and warm XY pressure at alpha=10^7 m^-1, with the frozen nominal GFE amplitude contours at 0.25, 0.50 and 0.75 of one shared pressure scale. "
        "Warm continuation retains the nearby Vortex pressure pattern; the cold high-weight endpoints develop twofold, Twin-like amplitude lobes. "
        "Twin-like describes the twofold amplitude shape; the pressure descriptor alone does not establish membership of the same terminal-command class as a separately synthesized Twin. "
        "Near-reference field retention is consistent with remaining near the original basin, but precision-loss stops do not certify converged local minima. "
        "The retained field shape does not preserve nominal mechanical precision: the independent elastic equilibrium shifts with the increased weight. "
        "One matched initial seed supplies an example, not an estimate of how often a cold start changes shape; SMOKE.")
    (output/"C3_figure_caption.txt").write_text(caption+"\n")
    protocol=dict(schema=SCHEMA,seed=seed,alpha_sequence=[10.,*ALPHAS],nominal_reference_phase_content_id=digest_array(reference_phase),
        nominal_source="same-seed Figure 1 alpha10 GFE command",cold_source="Figure 7 cold GFE cached commands",
        warm_initialization="previous terminal phase, unchanged; nominal then alpha1e3 then alpha1e6",
        reference_frozen_before_sweep=True,paired_origin_initial_content_id=digest_array(original_initial),
        force_target_n=nominal_objective.force_target_n.tolist(),maxiter_per_solve=10000,gtol=ctx.config.gtol,
        source_content_id=hashes,positions_content_id=digest_array(ctx.positions_m),frequency_hz=ctx.config.frequency_hz,
        curvature_weight=np.eye(3).tolist(),target_m=p.MAIN_TARGET_M.tolist(),
        mechanical_protocol=p._finite_ka_protocol(c3),evidence_status="SMOKE",
        pressure_display=dict(schema=DISPLAY_SCHEMA,plane="XY at target z",points_per_axis=DISPLAY_POINTS,half_width_m=.007,
            nominal_contour_levels=[.25,.5,.75],shared_pressure_scale=True),
        morphology=dict(radii_wavelength=[.04,.8],radius_count=80,azimuth_count=256,
            ring_selection="First radial maximum of mean pressure amplitude",twofold_measure="Absolute normalized azimuthal Fourier coefficient m=2",
            phase_winding="Sum wrapped adjacent phase increments around selected ring / (2 pi)",statistical_unit="One matched initial seed"))
    (output/"C3_protocol.json").write_text(json.dumps(protocol,indent=2,default=str))
    (output/"C3_cache_access.json").write_text(json.dumps(c3.cache.access_records,indent=2))
    small=frame[["initialization","alpha_per_m","command_cosine_to_fixed_GFE","field_cosine_to_fixed_GFE","finite_ka_displacement_a","terminal_gradient_l2","optimizer_message"]]
    reference_row=frame[frame.initialization.eq("Reference")].iloc[0]
    warm_terminal=frame[frame.initialization.eq("Warm")].sort_values("alpha_per_m").iloc[-1]
    cold_terminal=frame[frame.initialization.eq("Cold")].sort_values("alpha_per_m").iloc[-1]
    summary=(f"At alpha={warm_terminal.alpha_per_m:g}, warm command/complex-field cosines were "
        f"{warm_terminal.command_cosine_to_fixed_GFE:.6f}/{warm_terminal.field_cosine_to_fixed_GFE:.6f}; "
        f"cold values were {cold_terminal.command_cosine_to_fixed_GFE:.6f}/{cold_terminal.field_cosine_to_fixed_GFE:.6f}. "
        f"The warm elastic displacement was {warm_terminal.finite_ka_displacement_a:.6f} a, "
        f"compared with {reference_row.finite_ka_displacement_a:.6f} a for the frozen nominal command. "
        "Field correspondence and coordinate precision therefore need to be read separately. "
        +stop_sentence+"Native termination messages are listed below.\n")
    morphology_summary=("Warm fields retain near-zero twofold amplitude modulation, whereas cold alpha=10^6 and 10^7 fields have stronger twofold lobes. "
        "Cold high-weight examples retain winding -1, while nominal/warm examples have winding +1; winding and twofold modulation are separately exported pressure descriptors, not categorical endpoint-class tests. "
        "All high-weight endpoints are precision-loss stops; local-minimum convergence and across-seed occurrence rates are not established.\n")
    (output/"C3_measured_results.txt").write_text("SMOKE. Same original seed; fixed nominal GFE reference.\n\n"+summary+"\n"+small.to_string(index=False)+"\n\n"+morphology_summary+"\n"+morphology.to_string(index=False)+"\n")
    return dict(figures=paths,table=str(output/"C3_warm_cold_results.csv"),caption=str(output/"C3_figure_caption.txt"),protocol=str(output/"C3_protocol.json"))
