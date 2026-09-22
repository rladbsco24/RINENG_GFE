"""Appendix B: paired Twin/Bottle sign and weight SMOKE study.

Only this module permits the explicitly named historical-sign diagnostic.
The common elastic-sphere force validator is never sign-flipped.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import fingerprintlib
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.ndimage import label, map_coordinates
from threadpoolctl import threadpool_limits

from .gorkov_core import (ArrayGeometry, Condition, MethodSpec, SingleTargetObjective,
    SOURCE_SCALE_PA_M_PER_MURATA_UNIT, gauge_full, gorkov_coefficients, pressure_field)
from .exact_validator import (ArbitraryArrayPressureField, PartialWaveForceEvaluator,
    PartialWaveNumerics, ElasticSphere, Medium, common_cartesian_root_starts,
    validate_static_trap)

VERSION = "appendix-B-smoke-v1"
TARGET = (0.0, 0.0, 0.050)
FREQUENCY = 40000.0
SEEDS = (260905, 260906, 260907)
BETA_NOMINAL = 5.0e-5 * SOURCE_SCALE_PA_M_PER_MURATA_UNIT
SOLVER = dict(maxiter=400, gtol=1e-8, ftol=0.0, maxls=30, maxcor=10)
VOLUME_N = 49
HALF_WIDTH_M = 1.25 * 343.0 / FREQUENCY
CAVITY_THRESHOLD = 0.35
NUMERICS = PartialWaveNumerics(lmax=3, fit_shell_wavelengths=(.10,.16,.22),
    fit_n_mu=6, fit_n_phi=12, surface_n_mu=8, surface_n_phi=16)
ROOT_PROTOCOL = dict(search_half_width_a=4., max_nfev=14,
    numerical_root_tolerance=2.5e-3, jacobian_step_a=.10)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2), encoding="utf-8")


def _configuration(requested, method, sign, seed, eta=.01, beta_factor=1., stage="nominal"):
    trace = {"FE":1.02, "Conventional":2010., "RH":3.}[method]
    direction = np.array([1.,eta,eta] if requested == "Twin" else [eta,eta,1.])
    weights = trace * direction / direction.sum()
    config = dict(version=VERSION, requested=requested, method=method, sign=sign,
        seed=int(seed), eta=float(eta), beta_factor=float(beta_factor),
        alpha_per_m={"FE":1000., "Conventional":0., "RH":9.}[method],
        beta_curvature_per_pa={"FE":0., "Conventional":1., "RH":BETA_NOMINAL*beta_factor}[method],
        pressure_mode="smooth_abs" if method=="RH" else "abs",
        curvature_weights=weights.tolist(), smooth_pressure_relative=.001,
        target_m=list(TARGET), array_side=16, pitch_m=.010, frequency_hz=FREQUENCY,
        stencil_spacing_m=.0005, source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
        solver=SOLVER, stage=stage, status="SMOKE")
    if method != "Conventional":
        from .fe_solver_policy import FE_SOLVER_REVISION
        config.update(fe_solver_policy=FE_SOLVER_REVISION, evaluator_backend="compact-single-float64-v1")
    return config


def configurations():
    rows = [_configuration(shape, method, sign, seed)
        for shape in ("Twin","Bottle") for method in ("FE","Conventional","RH")
        for sign in ("corrected_minus","historical_plus") for seed in SEEDS]
    rows += [_configuration(shape, method, "corrected_minus", seed, eta=eta, stage="anisotropy")
        for shape in ("Twin","Bottle") for method in ("FE","Conventional","RH")
        for eta in (.001,.1) for seed in SEEDS]
    rows += [_configuration("Bottle", "RH", "corrected_minus", seed,
        beta_factor=bf, stage="beta") for bf in (0.,.1,10.) for seed in SEEDS]
    return rows


def _key(config):
    return fingerprintlib.fingerprint(json.dumps(config, sort_keys=True).encode()).hexdigest()[:18]


def _objective(config):
    geometry=ArrayGeometry.square(16, .010)
    weight=np.diag(config["curvature_weights"])
    spec=MethodSpec(config["method"], config["alpha_per_m"],
        config["beta_curvature_per_pa"], config["pressure_mode"], weight)
    obj=SingleTargetObjective(Condition("Appendix B", geometry.positions_m, TARGET,
        FREQUENCY, weight), spec, stencil_spacing_m=.0005,
        smooth_pressure_relative=config["smooth_pressure_relative"])
    if config["sign"] == "historical_plus":
        # This instance alone changes U and its analytic phase derivative together.
        obj.coefficients=replace(obj.coefficients,
            gradient_j_m2_pa2=-obj.coefficients.gradient_j_m2_pa2)
    elif config["sign"] != "corrected_minus":
        raise ValueError("Sign must be an explicit Appendix B convention")
    return obj


def _gradient_audit():
    records=[]
    with threadpool_limits(limits=1):
        for method in ("FE","Conventional","RH"):
            for sign in ("corrected_minus","historical_plus"):
                config=_configuration("Bottle",method,sign,SEEDS[0])
                obj=_objective(config)
                rng=np.random.default_rng(SEEDS[0]); x=rng.uniform(-np.pi,np.pi,255)
                value, grad=obj.fun_grad(x)
                for k in range(3):
                    d=rng.normal(size=255);d/=np.linalg.norm(d);h=1e-5
                    fd=(obj.full_value(gauge_full(x+h*d))-obj.full_value(gauge_full(x-h*d)))/(2*h)
                    analytic=float(grad@d)
                    error=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-10)
                    records.append(dict(method=method,sign=sign,direction=k,
                        analytic=analytic,finite_difference=fd,relative_error=error,
                        full_value_difference=abs(value-obj.full_value(gauge_full(x)))))
    if max(r["relative_error"] for r in records)>2e-5:
        raise RuntimeError("Appendix B sign/phase-gradient audit failed")
    return records


def _terms(obj, phase):
    raw=obj._raw_metrics(phase,True)
    a=obj.method.alpha_per_m;b=obj.method.beta_curvature_per_pa
    gradients=np.stack([-np.asarray(raw["d_weighted_curvature_d_phase"]),
        a*np.asarray(raw["d_force_norm_d_phase"]),
        b*np.asarray(raw["d_pressure_penalty_d_phase"])])
    values=np.array([-raw["weighted_curvature_n_m"],
        a*raw["force_residual_norm_n"], b*raw["pressure_penalty_pa"]])
    return raw,values,gradients


def _cavity(amplitude, threshold_fraction):
    threshold=threshold_fraction*float(amplitude.max())
    mask=amplitude<threshold
    labels,_=label(mask,structure=np.ones((3,3,3)))
    n=amplitude.shape[0];mid=n//2; target_label=int(labels[mid,mid,mid])
    if target_label==0:
        return dict(cavity_closed=False,cavity_status="target_above_threshold",
            boundary_contact=False,cavity_voxels=0,cavity_dx_mm=0.,cavity_dy_mm=0.,cavity_dz_mm=0.)
    region=labels==target_label
    boundary=bool(any(np.any(np.take(region,i,axis=ax)) for ax in range(3) for i in (0,-1)))
    inds=np.argwhere(region);dims=(inds.max(0)-inds.min(0))*2*HALF_WIDTH_M/(n-1)*1000
    resolved=bool(np.all(dims >= 4*HALF_WIDTH_M/(n-1)*1000))
    closed=not boundary and resolved
    return dict(cavity_closed=closed,
        cavity_status="closed_sampled_cavity" if closed else ("open_or_clipped" if boundary else "underresolved"),
        boundary_contact=boundary,cavity_voxels=int(region.sum()),
        cavity_dx_mm=float(dims[0]),cavity_dy_mm=float(dims[1]),cavity_dz_mm=float(dims[2]))


def _morphology(pressure, axis):
    amplitude=np.abs(pressure);n=len(axis);m=n//2
    xy=amplitude[:,:,m]
    # Assess twofold transverse modulation on a ring at the dominant x-lobe radius.
    line=xy[:,m];left=int(np.argmax(line[:m]));right=m+1+int(np.argmax(line[m+1:]))
    radius=.5*(abs(axis[left])+abs(axis[right]));radius=max(radius,axis[1]-axis[0])
    theta=np.linspace(0.,2*np.pi,144,endpoint=False)
    xi=(radius*np.cos(theta)-axis[0])/(axis[1]-axis[0])
    yi=(radius*np.sin(theta)-axis[0])/(axis[1]-axis[0])
    ring=map_coordinates(xy,np.vstack([xi,yi]),order=1,mode="nearest")
    m2=complex(np.mean(ring*np.exp(-2j*theta)))/max(float(ring.mean()),1e-30)
    twofold=float(abs(m2)); angle=float((-.5*np.angle(m2))*180/np.pi)
    lobe_pressure=float(.5*(line[left]+line[right]))
    central_ratio=float(amplitude[m,m,m]/max(lobe_pressure,1e-30))
    imbalance=float(abs(line[left]-line[right])/max(line[left]+line[right],1e-30))
    twin=bool(twofold>=.25 and central_ratio<.35 and abs(angle)<=30 and imbalance<.5)
    cavity=_cavity(amplitude,CAVITY_THRESHOLD)
    observed="Bottle-like enclosure" if cavity["cavity_closed"] else (
        "Twin-like lobes" if twin else ("open ring-like" if central_ratio<.35 and twofold<.15 else "open/other"))
    values=dict(cavity,observed_morphology=observed,twin_detected=twin,
        central_to_lobe_pressure=central_ratio,twin_twofold_contrast=twofold,
        twin_angle_deg=angle,twin_separation_mm=float((axis[right]-axis[left])*1000),
        twin_lobe_imbalance=imbalance,pressure_max_pa=float(amplitude.max()),
        target_pressure_pa=float(amplitude[m,m,m]),cavity_threshold_fraction=CAVITY_THRESHOLD)
    for fraction in (.25,.45):
        control=_cavity(amplitude,fraction)
        values[f"cavity_closed_at_{fraction:.2f}"]=control["cavity_closed"]
    return values,ring,theta


def _solve_one(task):
    config,output=task;output=Path(output);key=_key(config)
    record_path=output/"cache"/(key+".json");npz_path=output/"cache"/(key+".npz")
    if record_path.exists() and npz_path.exists() and zipfile.is_zipfile(npz_path):
        return json.loads(record_path.read_text())
    with threadpool_limits(limits=1):
        start=time.perf_counter();obj=_objective(config);setup_s=time.perf_counter()-start
        rng=np.random.default_rng(config["seed"]);x0=rng.uniform(-np.pi,np.pi,255)
        from .compact_single import CompactSingleObjective
        optimizer_objective = CompactSingleObjective(obj) if config["method"] != "Conventional" else obj
        history=[];phase_history=[]
        def callback(x):
            value,gradient=optimizer_objective.fun_grad(x)
            history.append([len(history)+1,value,np.linalg.norm(gradient),np.linalg.norm(gradient,np.inf)])
            phase_history.append(x.copy())
        start=time.perf_counter()
        result=minimize(optimizer_objective.fun_grad,x0,jac=True,method="L-BFGS-B",callback=callback,options=SOLVER)
        solve_s=time.perf_counter()-start;phase=gauge_full(result.x)
        raw,loss_terms,gradient_terms=_terms(obj,phase)
        axis=np.linspace(-HALF_WIDTH_M,HALF_WIDTH_M,VOLUME_N)
        xx,yy,zz=np.meshgrid(axis,axis,axis,indexing="ij")
        points=np.stack([xx,yy,zz],axis=-1)+np.asarray(TARGET)
        start=time.perf_counter();flat=points.reshape(-1,3)
        field=ArbitraryArrayPressureField(FREQUENCY,obj.condition.positions,np.exp(1j*phase))
        pressure=np.concatenate([field.pressure(flat[i:i+2048]) for i in range(0,len(flat),2048)]).reshape(points.shape[:-1])
        morphology,ring,theta=_morphology(pressure,axis)
        volume_s=time.perf_counter()-start
        start=time.perf_counter()
        evaluator=PartialWaveForceEvaluator(FREQUENCY,field,numerics=NUMERICS,include_effective_gravity=True)
        validation=validate_static_trap(evaluator,TARGET,
            initial_offsets_a=common_cartesian_root_starts((1.,)),**ROOT_PROTOCOL)
        validation_s=time.perf_counter()-start;eq=validation.equilibrium
        eig=validation.symmetric_stiffness_eigenvalues_n_m
        row=dict(config,cache_key=key,solver_success=bool(result.success),solver_status=int(result.status),
            solver_message=str(result.message),iterations=int(result.nit),function_evaluations=int(result.nfev),
            terminal_gradient_l2=float(np.linalg.norm(result.jac)),terminal_gradient_inf=float(np.linalg.norm(result.jac,np.inf)),
            objective_n_m=float(result.fun),setup_time_s=setup_s,command_time_s=solve_s,
            command_timing_scope="CPU L-BFGS-B including stored per-iteration callback diagnostics; setup/volume/elastic validation excluded",
            pressure_volume_time_s=volume_s,elastic_validation_time_s=validation_s,
            root_found=bool(eq.numerical_root_found),root_boundary=bool(eq.on_search_boundary),
            residual_scaled=float(eq.residual_scaled_norm),residual_force_n=float(np.linalg.norm(eq.residual_force_n)),
            displacement_um=float(eq.displacement_norm_m*1e6),displacement_a=float(eq.displacement_norm_a),
            dx_um=float(eq.displacement_m[0]*1e6),dy_um=float(eq.displacement_m[1]*1e6),dz_um=float(eq.displacement_m[2]*1e6),
            lambda1_mN_m=float(eig[0]*1000),lambda2_mN_m=float(eig[1]*1000),lambda3_mN_m=float(eig[2]*1000),
            locally_restoring=bool(eq.numerical_root_found and np.min(eig)>0),
            validator_model=validation.model_name,**morphology)
        for j,name in enumerate(("curvature","force","pressure")):
            row[name+"_weighted_loss_n_m"]=float(loss_terms[j])
            row[name+"_gradient_l2"]=float(np.linalg.norm(gradient_terms[j,1:]))
        temporary_npz=npz_path.with_name(npz_path.stem+".writing.npz")
        np.savez_compressed(temporary_npz,phase_rad=phase,initial_reduced_phase_rad=x0,
            reduced_phase_history=np.asarray(phase_history),history=np.asarray(history),
            weighted_loss_terms=loss_terms,weighted_phase_gradient_terms=gradient_terms,
            pressure_volume_pa_peak=pressure,volume_axis_m=axis,ring_pressure_pa=ring,ring_angle_rad=theta,
            equilibrium_m=eq.equilibrium_m,force_jacobian_n_m=validation.force_jacobian_n_m,
            symmetric_stiffness_n_m=validation.symmetric_stiffness_n_m,
            symmetric_stiffness_eigenvalues_n_m=eig,root_residual_n=eq.residual_force_n,
            objective_potential_j=raw["potential_j"],objective_pressure_pa=raw["pressure"])
        os.replace(temporary_npz,npz_path)
        row["validation_details"]=_jsonable(asdict(validation))
        _write_json(record_path,row)
        return row


def _plot(df, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"FE":"#C76A12","Conventional":"#3B6FA1","RH":"#80568D"}
    plt.rcParams.update({"font.size":11,"axes.labelsize":11,"xtick.labelsize":10,
        "ytick.labelsize":10,"legend.fontsize":10,"axes.linewidth":1.5,
        "xtick.major.width":1.5,"ytick.major.width":1.5,"lines.linewidth":2.,
        "font.family":"DejaVu Sans","font.weight":"semibold","axes.labelweight":"semibold",
        "svg.fonttype":"none","pdf.fonttype":42})
    paths=[]
    selected=df[(df.stage=="nominal")&(df.requested=="Bottle")&(df.seed==SEEDS[0])&df.method.isin(["FE","RH"])].copy()
    selected=selected.sort_values(["method","sign"])
    _write_json(output/"representatives.json",dict(rule="First predeclared paired seed; nominal Bottle FE and RH, both signs. Conventional remains in full tables/volumes.",
        seed=SEEDS[0],cache_keys=selected.cache_key.tolist()))
    volumes=[np.load(output/"cache"/(row.cache_key+".npz")) for _,row in selected.iterrows()]
    pmax=max(float(np.abs(v["pressure_volume_pa_peak"]).max()) for v in volumes)
    fig,axes=plt.subplots(4,2,figsize=(7.0866,13.6),layout="constrained")
    for i,((_,row),data) in enumerate(zip(selected.iterrows(),volumes)):
        p=np.abs(data["pressure_volume_pa_peak"])/pmax;m=VOLUME_N//2
        axis=data["volume_axis_m"]*1000;root=(data["equilibrium_m"]-TARGET)*1000
        for j,(section,vertical_index) in enumerate(((p[:,:,m],1),(p[:,m,:],2))):
            ax=axes[i,j];im=ax.imshow(section.T,origin="lower",extent=[axis[0],axis[-1],axis[0],axis[-1]],vmin=0,vmax=1,cmap="magma",interpolation="bilinear")
            ax.plot(0,0,"+",color="white",ms=10,mew=2.)
            if row.root_found:
                ax.plot(root[0],root[vertical_index],"o",mfc="none",mec="#49D9F0",ms=7,mew=1.6)
            ax.set_xlabel("x (mm)");ax.set_ylabel("y (mm)" if j==0 else "z − z* (mm)")
            ax.set_xticks([-10,0,10]);ax.set_yticks([-10,0,10]);ax.set_box_aspect(1)
            label_text=f"({chr(97+2*i+j)}) {row.method}  U{'−' if row.sign=='corrected_minus' else '+'}"
            ax.text(.03,.96,label_text,transform=ax.transAxes,ha="left",va="top",color="white",weight="bold",fontsize=11)
    cbar=fig.colorbar(im,ax=axes.ravel().tolist(),location="bottom",shrink=.68,pad=.02,aspect=35)
    cbar.set_label("|p| / shared maximum")
    for extension in ("png","pdf","svg"):
        path=output/("B1_nominal_bottle_pressure."+extension);fig.savefig(path,dpi=220,bbox_inches="tight");paths.append(str(path))
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.0866,6.8),layout="constrained")
    corrected=df[(df.sign=="corrected_minus")&df.stage.isin(["nominal","anisotropy"])]
    for method in ("FE","Conventional","RH"):
        for j,shape in enumerate(("Twin","Bottle")):
            part=corrected[(corrected.method==method)&(corrected.requested==shape)]
            metric="twin_twofold_contrast" if shape=="Twin" else "cavity_closed_any_declared_threshold"
            group=part.groupby("eta")[metric].agg(["mean","min","max"])
            ax=axes[0,j]
            x=group.index.to_numpy()*({"FE":.86,"Conventional":1.,"RH":1.16}[method] if j==1 else 1.)
            ax.errorbar(x,group["mean"].to_numpy(),
                yerr=np.vstack([(group["mean"]-group["min"]).to_numpy(),(group["max"]-group["mean"]).to_numpy()]),
                color=colors[method],fmt="o-",mec="#222222",mew=1.0,ms=7,capsize=3,label=method)
    for ax in axes[0]:
        ax.set_xscale("log");ax.set_xlabel("Curvature ratio η")
        ax.set_xticks([.001,.01,.1],["0.001","0.01","0.1"]);ax.set_xlim(.0007,.15)
    axes[0,0].set_ylabel("Twin: twofold contrast");axes[0,0].axhline(.25,color=".55",ls=":",lw=1.3)
    axes[0,1].set_ylabel("Bottle: closed fraction");axes[0,1].set_ylim(-.06,1.06);axes[0,1].set_yticks([0,.5,1])
    axes[0,0].legend(frameon=False,loc="best")
    beta=df[(df.method=="RH")&(df.requested=="Bottle")&(df.sign=="corrected_minus")&df.stage.isin(["nominal","beta"])]
    factors=[0.,.1,1.,10.]
    for j,metric in enumerate(("cavity_closed_any_declared_threshold","lambda1_mN_m")):
        ax=axes[1,j]
        for ix,factor in enumerate(factors):
            part=beta[beta.beta_factor==factor].sort_values("seed")
            if metric == "lambda1_mN_m":
                part=part[part.root_found]
            values=part[metric].astype(float).to_numpy()
            if not len(values):
                continue
            ax.scatter(ix+np.linspace(-.1,.1,len(values)),values,s=43,color=colors["RH"],edgecolors="#222222",linewidths=1.1,zorder=3)
            ax.plot([ix-.19,ix+.19],[np.median(values)]*2,color="black",lw=2.)
        ax.set_xticks(range(4),["0","0.1","1","10"]);ax.set_xlabel("RH β / nominal β")
    axes[1,0].set_ylabel("Bottle enclosure (0 / 1)");axes[1,0].set_ylim(-.1,1.1);axes[1,0].set_yticks([0,1])
    axes[1,1].set_ylabel("Weakest stiffness (mN/m)");axes[1,1].axhline(0,color=".55",ls=":",lw=1.3)
    for ax,letter in zip(axes.ravel(),"abcd"):
        ax.text(-.02,1.02,"("+letter+")",transform=ax.transAxes,ha="left",va="bottom",weight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.2,lw=.8)
    for extension in ("png","pdf","svg"):
        path=output/("B2_weight_response."+extension);fig.savefig(path,dpi=220,bbox_inches="tight");paths.append(str(path))
    plt.close(fig)
    return paths


def _write_audit(output):
    coefficients=asdict(gorkov_coefficients(FREQUENCY))
    _write_json(output/"physics_and_numerical_protocol.json",dict(version=VERSION,
        coefficients=coefficients,corrected_potential="U_minus=Kp*abs(p)^2-Kg*sum(abs(grad(p))^2)",
        diagnostic_potential="U_plus=Kp*abs(p)^2+Kg*sum(abs(grad(p))^2)",
        force="F=-grad(U)",phase_convention="peak phasor; exp(+ikr), time exp(-i omega t)",
        pressure_term="Conventional uses abs(p0); RH uses sqrt(abs(p0)^2+epsilon^2); FE beta=0",
        smoothing="epsilon=0.001*spatial RMS amplitude on 5x5x5 stencil; analytic epsilon derivative retained",
        rh_amplitude_ratio=SOURCE_SCALE_PA_M_PER_MURATA_UNIT,rh_beta_nominal=BETA_NOMINAL,
        beta_scaling="For p_new=s*p_old, quadratic force/curvature multiply by s^2 and P by s; beta_new=s*beta_old.",
        rh_scope="Nominal appendix re-instantiation: old RH alpha=9,beta=5e-5,W=I. Directional W replaces I at fixed trace=3; legacy smoothing equality not certified.",
        fe_scope="Requested legacy Twin/Bottle FE alpha=1000, trace(W)=1.02; main Single FE alpha=10 remains unchanged.",
        conventional_scope="alpha=0,beta=1, trace(W)=2010 retained; requested directional weights replace main diagonal(1000,1000,10).",
        solver_method="scipy.optimize.minimize(method='L-BFGS-B')",
        solver_scope="Appendix B bounded SMOKE uses identical L-BFGS-B across all sign/weight controls; main compact L-BFGS-B is not reproduced and absent enclosure is not a converged exclusion.",
        solver=SOLVER,seeds=SEEDS,volume_shape=[VOLUME_N]*3,volume_half_width_m=HALF_WIDTH_M,
        target_m=TARGET,root_protocol=ROOT_PROTOCOL,root_starts_a=common_cartesian_root_starts((1.,)),
        elastic_numerics=asdict(NUMERICS),elastic_sphere=asdict(ElasticSphere()),medium=asdict(Medium()),
        cavity_rule="26-connected component of |p|<0.35*volume_max at requested target. Closed only if no boundary contact and at least 3 voxels extent per axis; 0.25/0.45 controls exported. Sampled closure is resolution/threshold dependent.",
        twin_rule="Pressure ring m=2 coefficient / mean >=0.25, target/lobe pressure<0.35, orientation within 30 degrees of x and lobe imbalance<0.5.",
        status="SMOKE; fixed 400-iteration cap is not a claim of convergence"))
    formulations=[]
    for shape in ("Twin","Bottle"):
        for method in ("FE","Conventional","RH"):
            config=_configuration(shape,method,"corrected_minus",SEEDS[0])
            formulations.append(dict(requested=shape,method=method,
                alpha_per_m=config["alpha_per_m"],beta_N_per_m_Pa=config["beta_curvature_per_pa"],
                W_xx=config["curvature_weights"][0],W_yy=config["curvature_weights"][1],
                W_zz=config["curvature_weights"][2],pressure_mode=config["pressure_mode"],
                eta=.01,paired_seeds=3,maxiter=400))
    pd.DataFrame(formulations).to_csv(output/"B_nominal_formulations.csv",index=False)
    audit="""# Appendix B equation and implementation audit

The current peak-pressure surrogate is U_minus = Kp |p|² − Kg |grad p|², with positive Kg for the declared denser bead and F = −grad U. The appendix diagnostic U_plus changes only this term's sign, including its analytic phase derivative. All final mechanical values use the unchanged elastic-solid-sphere evaluator with effective gravity.

RH denotes Regularized-Hybrid: the appendix-local pressure-regularized objective −W:Hess(U) + alpha ||grad(U)|| + beta P_smooth. Its force term, pressure term and directional curvature coefficients are specified in B_nominal_formulations.csv. RH is not added to the main method factory or main comparisons, and its requested Bottle label does not establish its observed morphology.

The stencil is 5³ points at 0.5 mm spacing. The same central spatial differences and globally evaluated directivity enter both sign controls. Conventional uses |p|; RH uses sqrt(|p|² + epsilon²), epsilon = 0.001 times the stencil RMS amplitude. This is not |p|². Loss and analytic phase-gradient contributions are separately exported for pressure, force and curvature.

Appendix B uses scipy.optimize.minimize(method='L-BFGS-B') with a fixed 400-iteration cap, gtol=1e−8 and ftol=0 for all paired sign/weight controls. This bounded SMOKE solver differs from the main compact L-BFGS-B protocol; it is not a reproduction of main convergence timings. Failure to form an enclosure within this declared optimizer/initialization budget is not a converged nonexistence result.

Calibrating the historical directivity table multiplies pressure by s = 0.0848528137423857. Relative to curvature/force, a linear pressure penalty therefore requires beta_new = s beta_old = 4.24264068711929e−6 N m−1 Pa−1 for the archived beta_old=5e−5. The present RH is explicitly a nominal appendix re-instantiation: historical smoothing equivalence is not certified, and directional weights replace its archived isotropic W while keeping trace(W)=3. FE uses the requested legacy morphology alpha=1000 m−1, not the accepted main Single alpha=10. The main methods are unchanged.

The primary source is [Marzo et al., 2015](https://www.nature.com/articles/ncomms9661), with an [institutional published-version deposit](https://sussex.figshare.com/articles/journal_contribution/Holographic_acoustic_elements_for_manipulation_of_levitated_objects/23423144). The source reports Bottle-shaped fields and experimental trapping under its conditions. The text-extraction sign concern in the supplied handoff is not sufficient to certify the printed equation, and is never evidence of the authors' executable implementation. A separate source-status file records whether a printed PDF was actually obtained and visually checked. This local controlled study cannot explain or invalidate those published experiments by itself.
"""
    (output/"equation_and_implementation_audit.md").write_text(audit,encoding="utf-8")


def run(root, recompute=False, workers=2):
    """Return figures/tables; replay cached commands unless explicitly recomputed."""
    root=Path(root).resolve();output=root/"appendix_outputs"/"B"
    output.mkdir(parents=True,exist_ok=True);(output/"cache").mkdir(exist_ok=True)
    _write_audit(output)
    pd.DataFrame(_gradient_audit()).to_csv(output/"sign_phase_gradient_audit.csv",index=False)
    configs=configurations();_write_json(output/"run_matrix.json",configs)
    if recompute:
        raise ValueError("Use a new VERSION/output identity for deliberate fresh scientific runs; do not overwrite frozen SMOKE commands")
    rows=[]
    pending=[]
    for config in configs:
        path=output/"cache"/(_key(config)+".json")
        if path.exists() and path.with_suffix(".npz").exists():
            if not zipfile.is_zipfile(path.with_suffix(".npz")):
                raise RuntimeError(f"Damaged existing B cache {path.stem}; recover frozen arrays before replay rather than silently rerunning")
            rows.append(json.loads(path.read_text()))
        else:pending.append((config,str(output)))
    if pending:
        print(f"Appendix B: {len(pending)} new paired SMOKE solves, cap={SOLVER['maxiter']}; {len(rows)} reused",flush=True)
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures=[pool.submit(_solve_one,task) for task in pending]
            for future in as_completed(futures):
                row=future.result();rows.append(row)
                print(f"B {len(rows)}/{len(configs)} {row['requested']} {row['method']} {row['sign']} seed={row['seed']} nit={row['iterations']} shape={row['observed_morphology']} root={row['root_found']}",flush=True)
    df=pd.DataFrame([{k:v for k,v in row.items() if k!="validation_details"} for row in rows]).sort_values(["stage","requested","method","sign","eta","beta_factor","seed"])
    # Retain candidate diagnostics, while identifying physically resolved roots explicitly.
    df["resolved_displacement_a"]=df.displacement_a.where(df.root_found)
    df["resolved_displacement_um"]=df.displacement_um.where(df.root_found)
    df["resolved_lambda1_mN_m"]=df.lambda1_mN_m.where(df.root_found)
    df["solver_method"]="L-BFGS-B"
    df["stationarity_reached"]=df.terminal_gradient_inf <= SOLVER["gtol"]
    df["termination_category"]=["gradient_tolerance" if stationary else (
        "iteration_cap" if nit>=SOLVER["maxiter"] else (
        "objective_stagnation" if "RELATIVE REDUCTION" in message else "line_search_or_other_stop"))
        for stationary,nit,message in zip(df.stationarity_reached,df.iterations,df.solver_message)]
    df["terminal_pressure_smoothing_epsilon_pa"]=[
        float(np.sqrt(np.mean(np.abs(np.load(output/"cache"/(key+".npz"))["objective_pressure_pa"])**2))*.001)
        if method=="RH" else 0. for key,method in zip(df.cache_key,df.method)]
    # Preserve the primary threshold, and report the two predeclared controls explicitly.
    threshold_rows=[]
    for _,row in df.iterrows():
        amplitude=np.abs(np.load(output/"cache"/(row.cache_key+".npz"))["pressure_volume_pa_peak"])
        for fraction in (.25,.35,.45):
            threshold_rows.append(dict(cache_key=row.cache_key,stage=row.stage,
                requested=row.requested,method=row.method,sign=row.sign,seed=int(row.seed),
                eta=row.eta,beta_factor=row.beta_factor,threshold_fraction=fraction,
                **_cavity(amplitude,fraction)))
    threshold_table=pd.DataFrame(threshold_rows)
    threshold_table.to_csv(output/"B_cavity_threshold_controls.csv",index=False)
    df["cavity_closed_any_declared_threshold"]=df.cavity_closed|df["cavity_closed_at_0.25"]|df["cavity_closed_at_0.45"]
    df["restoring_enclosure"]=df.cavity_closed_any_declared_threshold & df.locally_restoring
    df["observed_morphology_at_primary_threshold"]=df.observed_morphology
    threshold_dependent=df.cavity_closed_any_declared_threshold & ~df.cavity_closed
    df.loc[threshold_dependent,"observed_morphology"]="threshold-dependent Bottle-like enclosure"
    df.to_csv(output/"B_all_runs_smoke.csv",index=False)
    nominal=df[df.stage=="nominal"]
    summary=nominal.groupby(["requested","method","sign"]).agg(n=("seed","size"),
        closed_cavities=("cavity_closed","sum"),twins=("twin_detected","sum"),
        closed_cavities_at_025=("cavity_closed_at_0.25","sum"),
        closed_cavities_at_045=("cavity_closed_at_0.45","sum"),
        resolved_roots=("root_found","sum"),restoring_roots=("locally_restoring","sum"),
        restoring_enclosures=("restoring_enclosure","sum"),
        stationary=("stationarity_reached","sum"),
        median_displacement_a=("resolved_displacement_a","median"),median_terminal_gradient=("terminal_gradient_l2","median"),
        median_iterations=("iterations","median")).reset_index()
    summary.to_csv(output/"B_nominal_summary.csv",index=False)
    stats=df.groupby(["stage","requested","method","sign","eta","beta_factor"]).terminal_gradient_l2.agg(
        n="size",median="median",minimum="min",maximum="max",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index()
    stats.to_csv(output/"B_terminal_gradient_statistics.csv",index=False)
    lines=["APPENDIX B — MEASURED SMOKE RESULTS",f"{len(df)} new paired-seed commands: 36 nominal, 36 extra anisotropy and 9 extra beta controls. Three fixed seeds; L-BFGS-B cap 400; gtol=1e-8, ftol=0. All numbers below are SMOKE.",
        "Requested morphology is separate from observed pressure enclosure/twofold-lobe classification. Cavity counts refer to the declared 49³ grid and the predeclared 0.25/0.35/0.45 amplitude thresholds; they are not mechanical success counts. The primary threshold is 0.35; existence at a lower threshold is reported as threshold-dependent enclosure rather than hidden."]
    for _,r in summary.iterrows():
        lines.append(f"Requested {r.requested}, {r.method}, {r.sign}: closed sampled cavities at thresholds 0.25/0.35/0.45 = {int(r.closed_cavities_at_025)}/{int(r.n)}, {int(r.closed_cavities)}/{int(r.n)}, {int(r.closed_cavities_at_045)}/{int(r.n)}; Twin-like criterion {int(r.twins)}/{int(r.n)}; resolved elastic roots {int(r.resolved_roots)}/{int(r.n)}; locally restoring roots {int(r.restoring_roots)}/{int(r.n)}; median terminal phase-gradient L2 {r.median_terminal_gradient:.4g} N/m; median iterations {r.median_iterations:g}.")
    bottle=df[(df.requested=="Bottle")&(df.sign=="corrected_minus")]
    lines.append(f"Across all tested corrected-sign requested Bottle controls: {int(bottle.cavity_closed.sum())}/{len(bottle)} sampled closed cavities at the primary 0.35 threshold and {int(bottle.cavity_closed_any_declared_threshold.sum())}/{len(bottle)} at any of the three declared thresholds. This bounded study does not establish nonexistence outside its weights, starts, thresholds and solver budget.")
    for _,r in bottle[bottle.cavity_closed_any_declared_threshold].iterrows():
        lines.append(f"The corrected-sign enclosure occurred for {r.method}, eta={r.eta:g}, beta/nominal={r.beta_factor:g}, seed={int(r.seed)}: displacement {r.displacement_a:.4g} a, weakest elastic symmetric stiffness {r.lambda1_mN_m:.4g} mN/m, locally restoring={bool(r.locally_restoring)}, terminal gradient L2={r.terminal_gradient_l2:.4g} N/m.")
    lines.append(f"Corrected requested Bottle commands with both a sampled enclosure at a declared threshold and a locally restoring elastic root: {int(bottle.restoring_enclosure.sum())}/{len(bottle)}.")
    lines.append(f"Termination: {int(df.solver_success.sum())}/{len(df)} solver success flags; {int((df.iterations>=400).sum())}/{len(df)} reached the cap. Full messages and terminal norms are in B_all_runs_smoke.csv.")
    lines.append(f"Only {int(df.stationarity_reached.sum())}/{len(df)} met the declared terminal infinity-norm tolerance. A SciPy relative-reduction success flag can occur with a large gradient at the nonsmooth pressure null and is classified as objective stagnation, not phase stationarity.")
    lines.append("Pressure closure and local restoration remain separate outcomes; positive symmetric stiffness does not establish capture or full dynamic stability.")
    (output/"measured_results_smoke.txt").write_text("\n\n".join(lines)+"\n",encoding="utf-8")
    (output/"mock_results_sentences.txt").write_text("TEMPLATE ONLY — no fabricated measurements.\nWith [specified weights], [method/sign] produced a sampled Bottle enclosure in [n/N] starts, of which [m/N] retained locally restoring elastic-bead equilibria.\nAt the declared production budget, [outcome] changed from [value] to [value] across [weight bracket].\n",encoding="utf-8")
    paths=_plot(df,output)
    captions="""B1. Nominal requested Bottle fields, using the first predeclared paired seed (260905), FE and RH, corrected U_minus and diagnostic U_plus. Each row shows XY and XZ sections through the target, with one shared pressure maximum across all eight panels. White crosses mark the request; cyan open circles mark resolved total-force roots projected into the plane. Panels do not alone certify enclosure: the cached 3D pressure volumes supply the threshold/connectivity check. The historical-sign examples are closed at 0.25 of their own volume maximum, but open at 0.35; corrected-sign examples are open at both. The full table reports all three predeclared thresholds, component dimensions and boundary contacts. Conventional controls and every seed remain in the full table and cached volumes. All results are SMOKE.

B2. Corrected-sign weight response. (a) Twin twofold transverse contrast, with dotted classification threshold; (b) sampled closed-cavity fraction for requested Bottle at any of the three predeclared amplitude thresholds (0.25, 0.35, 0.45); symbols/lines show three-seed means and bars show min–max. Method symbols in (b) are offset slightly along eta for visibility. (c,d) Matched RH beta controls for requested Bottle: individual seeds and median bars, using the same enclosure criterion. The full table preserves separate threshold outcomes. The sole corrected enclosure in the beta bracket occurs at beta=0 and threshold 0.25; it has negative weakest stiffness. Stiffness in (d) is shown only for resolved total-force roots. The dotted stiffness reference is zero. RH nominal beta is 4.24264e−6 N m−1 Pa−1; eta varies W direction at fixed method-specific trace. No additional solve is performed for the nominal row reused in both brackets. Counts are descriptive for three paired seeds, not production uncertainty estimates.
"""
    (output/"figure_captions.txt").write_text(captions,encoding="utf-8")
    result=dict(appendix="B",status="SMOKE",n_runs=len(df),figures=paths,
        tables=[str(output/name) for name in ("B_nominal_formulations.csv","B_nominal_summary.csv","B_all_runs_smoke.csv","B_cavity_threshold_controls.csv","B_terminal_gradient_statistics.csv","sign_phase_gradient_audit.csv")],
        measured_results=str(output/"measured_results_smoke.txt"),summary=summary.to_dict("records"))
    _write_json(output/"integration_summary.json",result)
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("root");parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args();run(args.root,workers=args.workers)
