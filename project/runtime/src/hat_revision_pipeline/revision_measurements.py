"""Supplementary measurement export for the unchanged eight main figures.

Every current/recovered result is SMOKE. This collector never promotes a cache
or a run-mode flag to manuscript evidence, and never changes plotted phases.
All raw target rows are retained. Missing production studies remain explicit.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import csv
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from . import pipeline as p
from .cache import digest_array
from .exact_validator import (
    ElasticSphere, Medium, PartialWaveForceEvaluator,
    ViscousElasticRayleighEvaluator, force_jacobian, validate_static_trap,
    standard_gorkov_contrast_factors,
)

SMOKE = "SMOKE — not manuscript results"
BOOTSTRAP_SEED = 20260905


def _table(ctx, name, rows):
    frame = pd.DataFrame(rows).copy()
    frame["evidence_status"] = "SMOKE"
    frame["manuscript_numerics"] = False
    ctx.tables[name] = frame
    # Checkpoint tables independently of later, potentially costly sweeps.
    path = ctx.writer.table_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    ctx.writer.record_file(path, "revision-measurement")
    return frame


def _stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, median=np.nan, q25=np.nan, q75=np.nan, minimum=np.nan, maximum=np.nan)
    return dict(n=len(x), median=float(np.median(x)), q25=float(np.quantile(x,.25)),
                q75=float(np.quantile(x,.75)), minimum=float(x.min()), maximum=float(x.max()))


def _convergence(ctx):
    from .final_figures import _accepted_iteration_axis
    bank = ctx.memo["final_single_vortex_bank"]
    rows = []
    derivatives = []
    for record in bank["table"].to_dict("records"):
        run = bank["runs"][(record["method"],record["pair_index"])]
        obj = p._objective_at(ctx, "Conventional" if record["method"] == "Conventional" else "Force-Equilibrium")
        _, full = obj.full_fun_grad(run.phase_rad)
        reduced = np.delete(full,obj.gauge_index)
        crossed = np.flatnonzero(np.asarray(run.history_gradient_norm) <= 1e-3)
        first = int(crossed[0]) if len(crossed) else None
        iterations = _accepted_iteration_axis(run)
        row = dict(record, gradient_l2=float(np.linalg.norm(reduced)),
            gradient_inf=float(np.linalg.norm(reduced,np.inf)), gradient_rms=float(np.sqrt(np.mean(reduced**2))),
            full_gradient_l2=float(np.linalg.norm(full)), n_free_phases=len(reduced), gradient_units="N/m/rad",
            reporting_threshold_l2=1e-3, native_gtol=float(ctx.config.gtol), native_solver_norm="infinity",
            threshold_ever_reached=first is not None,
            first_threshold_iteration=float(iterations[first]) if first is not None else np.nan,
            first_threshold_time_s=float(run.history_wall_s[first]) if first is not None else np.nan,
            reporting_outcome="threshold reached" if first is not None else "iteration cap" if run.iterations>=record["iteration_cap"] else "nonzero-gradient stall",
            timing_scope="solver interval; source field setup excluded", source_table="fig1_single_vortex_runs")
        rows.append(row)
        # One fixed direction at initial/terminal of the first paired run.
        if record["pair_index"] == 0:
            direction=np.random.default_rng(BOOTSTRAP_SEED).normal(size=len(full));direction-=direction.mean();direction/=np.linalg.norm(direction)
            for state,phase in (("initial",run.history_phase_rad[0]),("terminal",run.phase_rad)):
                _,gradient=obj.full_fun_grad(phase)
                for step in (1e-4,1e-5,1e-6):
                    fd=(obj.full_value(phase+step*direction)-obj.full_value(phase-step*direction))/(2*step)
                    analytic=float(np.dot(gradient,direction))
                    derivatives.append(dict(method=record["method"],state=state,step_rad=step,
                        analytic=analytic,finite_difference=fd,absolute_error=abs(fd-analytic),
                        relative_error=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-16),
                        phase_content_id=digest_array(np.asarray(phase)),
                        caveat="central difference at a norm cusp need not equal a unique derivative"))
    raw=_table(ctx,"revision_T1_solver_runs",rows)
    summary=[]
    for method,group in raw.groupby("method",sort=False):
        for metric in ("gradient_l2","gradient_inf","gradient_rms","iterations","wall_time_s","first_threshold_time_s"):
            summary.append(dict(method=method,metric=metric,**_stats(group[metric]),
                n_threshold=int(group.threshold_ever_reached.sum()),n_runs=len(group)))
    _table(ctx,"revision_T1_solver_summary",summary)
    pairs=raw.pivot(index="pair_index",columns="method",values=["wall_time_s","threshold_ever_reached"])
    n=len(pairs);rng=np.random.default_rng(BOOTSTRAP_SEED);indices=rng.integers(0,n,(5000,n))
    ratio=np.asarray(pairs["wall_time_s"]["Conventional"],float)/np.asarray(pairs["wall_time_s"]["FE"],float)
    delta=np.asarray(pairs["threshold_ever_reached"]["FE"],float)-np.asarray(pairs["threshold_ever_reached"]["Conventional"],float)
    _table(ctx,"revision_T1_paired_bootstrap",[
        dict(metric="median Conventional/FE solver-time ratio",estimate=float(np.median(ratio)),
             ci_low=float(np.quantile(np.median(ratio[indices],axis=1),.025)),ci_high=float(np.quantile(np.median(ratio[indices],axis=1),.975)),n_pairs=n,bootstrap_seed=BOOTSTRAP_SEED,n_resamples=5000),
        dict(metric="FE minus Conventional threshold attainment fraction",estimate=float(delta.mean()),
             ci_low=float(np.quantile(delta[indices].mean(axis=1),.025)),ci_high=float(np.quantile(delta[indices].mean(axis=1),.975)),n_pairs=n,bootstrap_seed=BOOTSTRAP_SEED,n_resamples=5000)])
    _table(ctx,"revision_T8_phase_gradient_check",derivatives)
    # Retain the original spatial-domain populations without inventing histories.
    for source in ("fig1_xz_domain_runs","appendix_fig1_xyz_domain_runs"):
        if source in ctx.tables:
            _table(ctx,"revision_T1_"+source,ctx.tables[source].assign(source_table=source))


def _silhouette(distance, labels):
    scores=[]
    for i,label in enumerate(labels):
        members=np.flatnonzero(labels==label);members=members[members!=i]
        if not len(members):scores.append(0.);continue
        a=float(distance[i,members].mean())
        b=min(float(distance[i,labels==other].mean()) for other in np.unique(labels) if other!=label)
        scores.append((b-a)/max(a,b,1e-30))
    return float(np.mean(scores))


def _branch_and_local(ctx):
    from .alpha1000_branch import fixed_single_branch_pack
    from .multitrap import evaluate_multitrap_objective
    pack=fixed_single_branch_pack(ctx)
    evolution=pack["evolution"]
    _table(ctx,"revision_T2_branch_raw",evolution)
    summaries=[]
    for (component,budget),group in evolution.groupby(["component","iteration"]):
        summaries.append(dict(component=component,iteration_budget=int(budget),**_stats(group.distance_to_matched_fe)))
    _table(ctx,"revision_T2_branch_distance",summaries)
    wide=evolution.pivot(index=["selected_trajectory_index","component"],columns="iteration",values="distance_to_matched_fe")
    changes=[]
    for key,row in wide.iterrows():
        for stop in (2000,10000):
            if 500 in row and stop in row:
                changes.append(dict(trajectory_index=key[0],component=key[1],from_budget=500,to_budget=stop,
                    distance_change=float(row[stop]-row[500]),distance_decreased=bool(row[stop]<row[500])))
    _table(ctx,"revision_T2_branch_paired_changes",changes)
    z=np.exp(1j*np.asarray(pack["anchor_phases"]));z/=np.linalg.norm(z,axis=1,keepdims=True)
    similarity=np.clip(abs(z.conj()@z.T),0,1)
    distance=np.sqrt(np.maximum(0,1-similarity**2));np.fill_diagonal(distance,0)
    shifted=z*np.exp(1j*np.random.default_rng(BOOTSTRAP_SEED).normal(size=len(z)))[:,None]
    ds=np.sqrt(np.maximum(0,1-np.clip(abs(shifted.conj()@shifted.T),0,1)**2));np.fill_diagonal(ds,0)
    cluster=[]
    for mode in ("average","complete"):
        tree=linkage(squareform(distance,checks=False),method=mode)
        for k in (2,3,4):
            labels=fcluster(tree,k,criterion="maxclust")
            cluster.append(dict(linkage=mode,k=k,silhouette=_silhouette(distance,labels),
                n_endpoints=len(z),population="continued chart medoids; not independent starts",
                maximum_global_phase_distance_error=float(np.max(abs(ds-distance)))))
    _table(ctx,"revision_T2_descriptive_cluster_checks",cluster)
    local=[];cuts=[]
    single=ctx.memo["final_decision_pack"]
    cases=[]
    for method in ("Conventional","FE"):
        obj=single["objectives"][method];run=single["runs"][method]
        u,v=p._native_local_directions(obj,run)
        cases.append(("Single",method,np.asarray(run.phase_rad),u,v,obj.full_value,"Figure 3 native directions"))
    node=ctx.memo["final_triple_exact_force_concentration"]["node"]
    archive=np.load(ctx.data_root/"fig4_selected_triple_10k"/"central_native_directions.npz")
    for method in ("Conventional","FE"):
        config=node["conventional_config" if method=="Conventional" else "fe_config"]
        run=node[method]
        if method=="Conventional":
            phase=node["Conventional_checkpoint_phases"][10000];ef=run.force_epsilon;eu=run.uniformity_epsilon;prefix="conventional"
        else:
            phase=node["FE_phase"];ef=run.stages[-1].force_epsilon;eu=run.stages[-1].uniformity_epsilon;prefix="fe"
        def objective(x,config=config,ef=ef,eu=eu):
            return evaluate_multitrap_objective(node["problem"],x,config,force_epsilon=ef,uniformity_epsilon=eu).value
        cases.append(("Triple",method,np.asarray(phase),archive[prefix+"_u"],archive[prefix+"_v"],objective,"Figure 4 archived directions"))
    for task,method,phase,u,v,fun,source in cases:
        value=fun(phase)
        common=dict(task=task,method=method,phase_content_id=digest_array(phase),direction_source=source,
            direction_u_content_id=digest_array(np.asarray(u)),direction_v_content_id=digest_array(np.asarray(v)),
            direction_u_l2=float(np.linalg.norm(u)),direction_v_l2=float(np.linalg.norm(v)))
        for h in (.001,.003,.01):
            left=(value-fun(phase-h*u))/h;right=(fun(phase+h*u)-value)/h
            cuts.append(dict(common,step_rad=h,left_slope=left,right_slope=right,slope_jump=right-left))
        q=np.linspace(-.03,.03,21);xy=np.array([(x,y) for y in q for x in q]);x,y=xy.T
        values=np.array([fun(phase+xx*u+yy*v)-value for xx,yy in xy])
        design=np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y]);coef=np.linalg.lstsq(design,values,rcond=None)[0]
        hessian=np.array([[2*coef[3],coef[4]],[coef[4],2*coef[5]]]);eig=np.linalg.eigvalsh(hessian)
        matrix=values.reshape(21,21);primary=np.ptp(matrix[10]);transverse=np.ptp(matrix[:,10])
        local.append(dict(common,half_range_rad=.03,samples_per_axis=21,
            quadratic_relative_residual=float(np.linalg.norm(values-design@coef)/max(np.linalg.norm(values-values.mean()),1e-30)),
            section_hessian_min=float(eig[0]),section_hessian_max=float(eig[1]),
            transverse_primary_variation_ratio=float(transverse/max(primary,1e-30)),
            section_hessian_units="N/m/rad^2"))
    _table(ctx,"revision_T2_local_slopes",cuts);_table(ctx,"revision_T2_local_quadratic",local)


def _items(ctx):
    bank=ctx.memo["final_exact_force_concentration"]
    items=[]
    for spec,result,row in zip(bank["specs"],bank["results"],bank["concentration_table"].to_dict("records"),strict=True):
        items.append(dict(spec=spec,result=result,row=row))
    items.extend(ctx.memo["final_triple_exact_force_concentration"]["evaluations"])
    return items


def _root_kwargs(ctx):
    protocol=p._finite_ka_protocol(ctx)
    return dict(search_half_width_a=protocol["search_half_width_a"],initial_offsets_a=np.asarray(protocol["root_starts_a"]),
                max_nfev=protocol["max_nfev_per_start"],numerical_root_tolerance=protocol["numerical_root_tolerance"],
                jacobian_step_a=protocol["jacobian_step_a"])


def _evaluator(ctx,phase,*,sphere=None,gravity=True,numerics=None):
    return PartialWaveForceEvaluator(ctx.config.frequency_hz,p._array_field(ctx,phase),
        sphere=sphere or ElasticSphere(),numerics=numerics or p._partial_wave_numerics(ctx),include_effective_gravity=gravity)


def _mechanics(ctx,items):
    rows=[];probes=[];surrogate=[];gravity=[];commands=[];sizes=[];timings=[]
    sphere=ElasticSphere();medium=Medium();a=sphere.radius_m;mass=sphere.density_kg_m3*sphere.volume_m3
    drag=6*math.pi*medium.dynamic_viscosity_pa_s*a
    output=ctx.output_root/"data"/"revision_frozen_commands";output.mkdir(parents=True,exist_ok=True)
    for item in items:
        spec=item["spec"];result=item["result"];row=dict(item["row"])
        phase=np.asarray(spec["phase"]);target=result.equilibrium.target_m;eq=result.equilibrium.equilibrium_m
        if row["method"] == "FE+g" and row["task"] == "Triple":
            row["command_time_s"] = ctx.memo["final_fig6_vertical_compensated_fe"]["run"].end_to_end_sec
        root=bool(result.equilibrium.numerical_root_found);content_id=digest_array(phase)
        common=dict(task=row["task"],method=row["method"],target_id=row["target_id"],phase_content_id=content_id)
        command_path=output/(content_id+".npz")
        if not command_path.exists():
            np.savez_compressed(command_path,phase_rad=phase,positions_m=ctx.positions_m,frequency_hz=ctx.config.frequency_hz)
            ctx.writer.record_file(command_path,"frozen-phase-command")
        commands.append(dict(common,command_file=str(command_path.relative_to(ctx.output_root)),
            target_x_m=target[0],target_y_m=target[1],target_z_m=target[2]))
        command_run=spec.get("command_run")
        timing=getattr(command_run,"timing",None)
        timings.append(dict(common,command_time_s=row.get("command_time_s",np.nan),
            measured_timing_fields_json=json.dumps(asdict(timing),default=str) if timing is not None else "null",
            timing_scope="native measured wrapper" if timing is not None else "recorded solver/source interval; setup breakdown unavailable",
            phase_source=spec.get("phase_source",""),native_update_count=spec.get("terminal_iteration",np.nan)))
        evaluator=_evaluator(ctx,phase);j=result.force_jacobian_n_m
        eigen=np.linalg.eigvals(np.block([[np.zeros((3,3)),np.eye(3)],[j/mass,-drag/mass*np.eye(3)]]))
        row.update(common,dx_um=1e6*result.equilibrium.displacement_m[0] if root else np.nan,
            dy_um=1e6*result.equilibrium.displacement_m[1] if root else np.nan,
            dz_um=1e6*result.equilibrium.displacement_m[2] if root else np.nan,
            lateral_displacement_um=1e6*np.linalg.norm(result.equilibrium.displacement_m[:2]) if root else np.nan,
            axial_displacement_fraction=abs(result.equilibrium.displacement_m[2])/max(result.equilibrium.displacement_norm_m,1e-30) if root else np.nan,
            dynamic_max_real_eigenvalue_s_inv=float(eigen.real.max()) if root else np.nan,
            linearly_stable=bool(root and eigen.real.max()<0),drag_coefficient_kg_s=drag,particle_mass_kg=mass,
            dynamics_scope="linearized finite-ka force plus isolated-sphere steady Stokes drag; no streaming or walls",
            actuator_count=len(phase),drive_sum_abs_weights_squared=float(len(phase)),
            drive_scope="unit-amplitude phase commands; drive sum is not measured electrical/acoustic power",
            support_ratio_at_target=float((result.target_force_n[2]+evaluator.effective_weight_n)/evaluator.effective_weight_n))
        for i,axis in enumerate("xyz"):
            row["K"+axis+axis+"_n_m"]=result.symmetric_stiffness_n_m[i,i] if root else np.nan
            for k,axis2 in enumerate("xyz"):
                row["J"+axis+axis2+"_n_m"]=j[i,k] if root else np.nan
        jt=force_jacobian(evaluator,target,.1*a)
        try:prediction=-np.linalg.solve(jt,result.target_force_n)
        except np.linalg.LinAlgError:prediction=np.full(3,np.nan)
        row["linear_prediction_error_um"]=float(1e6*np.linalg.norm(prediction-result.equilibrium.displacement_m)) if root else np.nan
        for i,axis in enumerate("xyz"):row["target_linear_prediction_"+axis+"_um"]=1e6*prediction[i]
        if root:
            for i,axis in enumerate("xyz"):
                for sign in (-1,1):
                    offset=np.eye(3)[i]*sign*.1*a;force=evaluator.force(eq+offset).total_n
                    probes.append(dict(common,axis=axis,sign=sign,offset_a=.1,fx_n=force[0],fy_n=force[1],fz_n=force[2],
                        inward_radial_force_n=-float(force@offset)/np.linalg.norm(offset)))
        rows.append(row)
        # Same-field solid Gor'kov limit, with the SAME regular-wave derivatives.
        # This avoids silently comparing a directivity-knot FD operator to a projection.
        inviscid=ViscousElasticRayleighEvaluator(ctx.config.frequency_hz,p._array_field(ctx,phase),
            numerics=p._viscous_rayleigh_numerics(ctx),incident_projection_numerics=p._partial_wave_numerics(ctx))
        inviscid.monopole_contrast,inviscid.dipole_contrast=standard_gorkov_contrast_factors(medium,sphere)
        payload=dict(contract="revision-solid-gorkov-v1",phase=content_id,target=target.tolist(),protocol=p._finite_ka_protocol(ctx),array=digest_array(ctx.positions_m))
        other,_,_=ctx.cache.get_or_compute("revision_solid_gorkov",payload,
            lambda:validate_static_trap(inviscid,target,**_root_kwargs(ctx)),recompute=ctx.recompute)
        ok=root and other.equilibrium.numerical_root_found
        surrogate.append(dict(common,gorkov_root_found=other.equilibrium.numerical_root_found,
            elastic_root_found=root,equilibrium_difference_um=1e6*np.linalg.norm(eq-other.equilibrium.equilibrium_m) if ok else np.nan,
            gorkov_displacement_a=other.equilibrium.displacement_norm_a if other.equilibrium.numerical_root_found else np.nan,
            gorkov_stiffness_min_n_m=other.symmetric_stiffness_eigenvalues_n_m.min() if other.equilibrium.numerical_root_found else np.nan,
            same_position_jacobian_difference_fro_n_m=np.linalg.norm(force_jacobian(inviscid,eq,.1*a)-j) if root else np.nan,
            model_scope="solid Gor'kov limit with regular-wave pressure derivatives; not fluid-sphere scattering"))
        if row["task"]=="Single" and row["method"] in ("Conventional","FE"):
            for radius_factor in (1.,.5):
                small=replace(sphere,radius_m=a*radius_factor)
                if radius_factor==1.:
                    size_exact=result;size_gorkov=other
                else:
                    se=_evaluator(ctx,phase,sphere=small)
                    sg=ViscousElasticRayleighEvaluator(ctx.config.frequency_hz,p._array_field(ctx,phase),sphere=small,
                        numerics=p._viscous_rayleigh_numerics(ctx),incident_projection_numerics=p._partial_wave_numerics(ctx))
                    sg.monopole_contrast,sg.dipole_contrast=standard_gorkov_contrast_factors(medium,small)
                    size_payload=dict(payload,radius_m=small.radius_m)
                    size_exact,_,_=ctx.cache.get_or_compute("revision_size_elastic",size_payload,
                        lambda:validate_static_trap(se,target,**_root_kwargs(ctx)),recompute=ctx.recompute)
                    size_gorkov,_,_=ctx.cache.get_or_compute("revision_size_gorkov",size_payload,
                        lambda:validate_static_trap(sg,target,**_root_kwargs(ctx)),recompute=ctx.recompute)
                size_ok=size_exact.equilibrium.numerical_root_found and size_gorkov.equilibrium.numerical_root_found
                sizes.append(dict(common,radius_m=small.radius_m,ka=2*np.pi*ctx.config.frequency_hz*small.radius_m/medium.sound_speed_m_s,
                    both_roots_found=size_ok,equilibrium_difference_um=1e6*np.linalg.norm(size_exact.equilibrium.equilibrium_m-size_gorkov.equilibrium.equilibrium_m) if size_ok else np.nan,
                    elastic_displacement_a=size_exact.equilibrium.displacement_norm_a if size_exact.equilibrium.numerical_root_found else np.nan))
        if row["task"]=="Triple":
            off=_evaluator(ctx,phase,gravity=False)
            free,_,_=ctx.cache.get_or_compute("revision_gravity_off",dict(payload,gravity=False),
                lambda:validate_static_trap(off,target,**_root_kwargs(ctx)),recompute=ctx.recompute)
            both=root and free.equilibrium.numerical_root_found
            shift=free.equilibrium.equilibrium_m-eq
            gravity.append(dict(common,gravity_off_root_found=free.equilibrium.numerical_root_found,
                gravity_on_displacement_a=result.equilibrium.displacement_norm_a if root else np.nan,
                gravity_off_displacement_a=free.equilibrium.displacement_norm_a if free.equilibrium.numerical_root_found else np.nan,
                **{"gravity_off_minus_on_"+axis+"_um":float(1e6*shift[i]) if both else np.nan for i,axis in enumerate("xyz")}))
    mechanics=_table(ctx,"revision_T3_mechanics_per_target",rows)
    _table(ctx,"revision_T3_force_probes",probes);_table(ctx,"revision_T3_surrogate_comparison",surrogate)
    _table(ctx,"revision_T3_gravity_counterfactual",gravity);_table(ctx,"revision_command_index",commands)
    _table(ctx,"revision_T3_particle_size_check",sizes);_table(ctx,"revision_T5_native_timing_records",timings)
    summary=[]
    for (task,method,content_id),group in mechanics.groupby(["task","method","phase_content_id"],sort=False):
        resolved=group[group.finite_ka_root_found.astype(bool)]
        positions=resolved[["finite_ka_equilibrium_x_m","finite_ka_equilibrium_y_m","finite_ka_equilibrium_z_m"]].to_numpy(float)
        distinct=[]
        for point in positions:
            if all(np.linalg.norm(point-prior)>.01*a for prior in distinct):distinct.append(point)
        summary.append(dict(task=task,method=method,phase_content_id=content_id,n_targets=len(group),n_resolved=len(resolved),n_distinct_roots=len(distinct),
            root_distinctness_tolerance_a=.01,n_restoring=int(group.finite_ka_locally_restoring.sum()),
            n_linearly_stable=int(group.linearly_stable.sum()),median_displacement_a=resolved.finite_ka_displacement_a.median(),
            maximum_displacement_a=resolved.finite_ka_displacement_a.max(),weakest_target_stiffness_n_m=resolved.finite_ka_stiffness_min_n_m.min(),
            median_equilibrium_pressure_pa=resolved.pressure_abs_at_exact_equilibrium_pa.median(),
            command_time_s=group.command_time_s.median() if "command_time_s" in group else np.nan,
            all_targets_within_one_radius=bool(len(distinct)==len(group) and group.linearly_stable.all() and group.finite_ka_locally_restoring.all() and (group.finite_ka_displacement_a<=1).all()),
            sampling_unit="one shared command; targets are not independent optimizer repeats"))
    _table(ctx,"revision_T3_command_summary",summary)
    _table(ctx,"revision_T5_accuracy_cost",pd.DataFrame(summary).assign(timing_caveat="original measured intervals retained; cold/warm replication remains pending"))
    return mechanics


def _quantization(ctx,items):
    jobs=[]
    for item in items:
        for levels in (8,16,32,64):jobs.append((item,levels))
    def evaluate(job):
        item,levels=job;phase=np.asarray(item["spec"]["phase"]);original=item["result"];row=item["row"]
        # One fixed gauge before quantization, independent of outcome.
        canonical=np.angle(np.exp(1j*(phase-phase[0])))
        start=time.perf_counter();quantized=np.angle(np.exp(1j*(2*np.pi/levels*np.rint(canonical*levels/(2*np.pi)))))
        elapsed=time.perf_counter()-start;error=np.angle(np.exp(1j*(quantized-canonical)))
        target=original.equilibrium.target_m
        result=p._finite_ka_validation(ctx,quantized,target,key=f"revision-Q{levels}-{row['task']}-{row['method']}-{row['target_id']}")
        root=result.equilibrium.numerical_root_found;both=root and original.equilibrium.numerical_root_found
        return dict(task=row["task"],method=row["method"],target_id=row["target_id"],levels=levels,
            continuous_phase_content_id=digest_array(phase),quantized_phase_content_id=digest_array(quantized),
            gauge="subtract actuator 0 phase; nearest-level np.rint; wrap to [-pi,pi]",
            phase_step_rad=2*np.pi/levels,phase_rms_error_rad=np.sqrt(np.mean(error**2)),phase_max_error_rad=max(abs(error)),
            rounding_time_s=elapsed,root_found=root,locally_restoring=bool(root and result.symmetric_stiffness_eigenvalues_n_m.min()>0),
            target_radiation_force_change_norm_n=float(np.linalg.norm(result.target_force_n-original.target_force_n)),
            **{"target_radiation_force_change_"+axis+"_n":float(result.target_force_n[i]-original.target_force_n[i]) for i,axis in enumerate("xyz")},
            continuous_displacement_a=original.equilibrium.displacement_norm_a if original.equilibrium.numerical_root_found else np.nan,
            quantized_displacement_a=result.equilibrium.displacement_norm_a if root else np.nan,
            equilibrium_shift_um=1e6*np.linalg.norm(result.equilibrium.equilibrium_m-original.equilibrium.equilibrium_m) if both else np.nan,
            continuous_stiffness_min_n_m=original.symmetric_stiffness_eigenvalues_n_m.min() if original.equilibrium.numerical_root_found else np.nan,
            quantized_stiffness_min_n_m=result.symmetric_stiffness_eigenvalues_n_m.min() if root else np.nan)
    with ThreadPoolExecutor(max_workers=int(os.environ.get("HAT_EXACT_WORKERS","3"))) as executor:
        rows=list(executor.map(evaluate,jobs))
    _table(ctx,"revision_T6_discrete_phases",rows)


def _ablations(ctx):
    from .multitrap import run_multitrap,evaluate_multitrap_objective
    benchmark=ctx.memo["final_triple_exact_force_concentration"];node=benchmark["node"];bank=benchmark["long_bank"]
    base=node["fe_config"];problem=node["problem"];initial=np.asarray(bank["initial_phase"]);seed=int(bank["seed"])
    configurations=[("Full",base)]
    for label,component in (("Minus force smoothing","force_smoothing"),("Minus pressure","pressure_retention"),("Minus uniformity","uniformity")):
        configurations.append((label,replace(base,components=replace(base.components,**{component:False}))))
    rows=[];stages=[];terms=[]
    for label,config in configurations:
        if label=="Full":run=node["FE"]
        else:
            run,_,_=ctx.cache.get_or_compute("revision_triple_ablation",dict(contract="one-seed-smoke-10000-cap-v1",problem=problem.fingerprint,
                config=config.to_payload(),seed=seed,initial=digest_array(initial)),
                lambda:run_multitrap(problem,config,seed=seed,initial_phases=initial,record_history=True),recompute=ctx.recompute)
        command_path=ctx.writer.data_dir/"revision_frozen_commands"/(digest_array(np.asarray(run.phases))+".npz")
        command_path.parent.mkdir(parents=True,exist_ok=True)
        if not command_path.exists():
            np.savez_compressed(command_path,phase_rad=run.phases,initial_phase_rad=initial,
                positions_m=ctx.positions_m,frequency_hz=ctx.config.frequency_hz)
        ctx.writer.record_file(command_path,"ablation-phase-command")
        for index,stage in enumerate(run.stages):stages.append(dict(configuration=label,stage=index,**asdict(stage)))
        for state,phase in (("initial",initial),("terminal",run.phases)):
            stage=run.stages[0 if state=="initial" else -1]
            ev=evaluate_multitrap_objective(problem,phase,config,force_epsilon=stage.force_epsilon,uniformity_epsilon=stage.uniformity_epsilon)
            for i,local in enumerate(ev.locals):
                force_denominator = math.hypot(local.force_norm_n, stage.force_epsilon) if config.components.force_smoothing else local.force_norm_n
                force_gradient = config.alpha_force * local.d_force_norm * local.force_norm_n / max(force_denominator,1e-30)
                uniformity_gradient = ev.gradient_full - np.mean(ev.local_gradients,axis=0)
                terms.append(dict(configuration=label,state=state,target_id=local.target_id,
                    curvature_term_n_m=-local.curvature_n_m,force_term_n_m=config.alpha_force*ev.force_penalties[i],
                    pressure_term_n_m=config.beta_pressure*local.pressure_penalty if config.components.pressure_retention else 0,
                    pressure_abs_at_target_pa=local.pressure_abs,
                    realized_pressure_epsilon_pa=math.sqrt(max(local.pressure_penalty**2-local.pressure_abs**2,0.0)),
                    local_loss_n_m=ev.local_losses[i],total_objective_n_m=ev.value,uniformity_penalty=ev.uniformity_penalty,
                    curvature_gradient_l2=float(np.linalg.norm(local.d_curvature)),
                    pressure_gradient_l2=float(config.beta_pressure*np.linalg.norm(local.d_pressure_penalty)) if config.components.pressure_retention else 0,
                    force_gradient_l2=float(np.linalg.norm(force_gradient)),
                    weighted_uniformity_term_n_m=config.gamma_uniformity*ev.uniformity_penalty,
                    weighted_uniformity_gradient_l2=float(np.linalg.norm(uniformity_gradient)),
                    total_gradient_l2=float(np.linalg.norm(ev.gradient_reduced)),phase_content_id=digest_array(np.asarray(phase))))
        for target_index,target in enumerate(benchmark["targets_m"]):
            result=p._finite_ka_validation(ctx,np.asarray(run.phases),target,key=f"revision-ablation-{label}-T{target_index+1}")
            row=p._static_validation_row(label,result)
            row.update(configuration=label,target_id=f"T{target_index+1}",seed=seed,phase_content_id=digest_array(np.asarray(run.phases)),
                initial_phase_content_id=digest_array(initial),iterations=run.iterations,iteration_cap=10000,
                command_file=str(command_path.relative_to(ctx.output_root)),
                terminal_gradient_l2=run.terminal_gradient_norm,command_time_s=run.end_to_end_sec,
                config_json=json.dumps(config.to_payload()),scope="one shared initialization; SMOKE integration check")
            rows.append(row)
    _table(ctx,"revision_T4_component_ablations",rows);_table(ctx,"revision_T4_stage_records",stages);_table(ctx,"revision_T4_objective_terms",terms)
    planned=[("Single alpha","5,10,20","1/m"),("Triple alpha","1500,3000,6000","1/m"),
        ("Triple beta","0,5e-6,5e-5,5e-4","N/m/Pa"),("Triple gamma","0,.5,1,2","1"),
        ("Triple Wzz","5,10,20","1"),("smoothing schedule multipliers",".5,1,2","1")]
    _table(ctx,"revision_T4_sensitivity_registry",[dict(parameter=k,values=v,units=u,status="PENDING independent-seed sensitivity study",planned_common_seeds=10) for k,v,u in planned])


def _numerics(ctx,items):
    sphere=ElasticSphere();medium=Medium();a=sphere.radius_m;n=p._partial_wave_numerics(ctx)
    _table(ctx,"revision_T8_parameters",[dict(frequency_hz=ctx.config.frequency_hz,wavelength_m=medium.sound_speed_m_s/ctx.config.frequency_hz,
        particle_radius_m=a,particle_diameter_m=2*a,ka=2*np.pi*ctx.config.frequency_hz*a/medium.sound_speed_m_s,
        bead_density_kg_m3=sphere.density_kg_m3,c_longitudinal_m_s=sphere.longitudinal_sound_speed_m_s,
        c_shear_m_s=sphere.shear_sound_speed_m_s,bulk_modulus_pa=sphere.bulk_modulus_pa,shear_modulus_pa=sphere.shear_modulus_pa,
        host_viscosity_pa_s=medium.dynamic_viscosity_pa_s,medium_json=json.dumps(asdict(medium)),numerics_json=json.dumps(asdict(n)),
        bead_parameter_status="specified elastic simulation bead; not experimentally calibrated foam",
        radiation_reference="elastic-solid partial waves in inviscid host",
        viscosity_models="solid Rayleigh viscous dipole and separate thermoelastic thin-layer sensitivity",
        phasor="peak pressure, exp(-i omega t)")])
    checks=[];window=[];domain=[]
    representatives=[item for item in items if item["row"]["task"]=="Single" and item["row"]["method"] in ("Conventional","FE")]
    for method in ("Conventional","FE","IB","GS","AD","FE+g"):
        candidates=[item for item in items if item["row"]["task"]=="Triple" and item["row"]["method"]==method]
        if candidates:representatives.append(min(candidates,key=lambda x:float(x["result"].symmetric_stiffness_eigenvalues_n_m.min())))
    for item in representatives:
        row=item["row"];phase=np.asarray(item["spec"]["phase"]);result=item["result"];point=result.equilibrium.equilibrium_m
        common=dict(task=row["task"],method=row["method"],target_id=row["target_id"],phase_content_id=digest_array(phase))
        base=_evaluator(ctx,phase);f=base.force(point).radiation_n
        for label,number in (("L+2",replace(n,lmax=n.lmax+2)),
            ("double angular nodes",replace(n,fit_n_mu=2*n.fit_n_mu,fit_n_phi=2*n.fit_n_phi,surface_n_mu=2*n.surface_n_mu,surface_n_phi=2*n.surface_n_phi)),
            *((f"R/a={radius}",replace(n,control_radius_a=radius)) for radius in (1.5,2.,3.))):
            evaluator=_evaluator(ctx,phase,numerics=number)
            force=evaluator.force(point).radiation_n
            jac=force_jacobian(evaluator,point,.1*a)
            checks.append(dict(common,check=label,force_difference_n=float(np.linalg.norm(force-f)),
                stiffness_min_at_fixed_point_n_m=float(np.linalg.eigvalsh(-(jac+jac.T)/2).min()),
                evaluation_position="baseline candidate/equilibrium; no re-rooting in this check",baseline_root_found=result.equilibrium.numerical_root_found,
                numerics_json=json.dumps(asdict(number))))
        for h in (.05,.1,.2):
            jac=force_jacobian(base,point,h*a)
            checks.append(dict(common,check=f"Jacobian h/a={h}",jacobian_step_a=h,
                stiffness_min_at_fixed_point_n_m=float(np.linalg.eigvalsh(-(jac+jac.T)/2).min()),baseline_root_found=result.equilibrium.numerical_root_found))
        field=p._array_field(ctx,phase);target=result.equilibrium.target_m
        for half,count in ((.009,81),(.012,81),(.015,81),(.012,161)):
            axis=np.linspace(-half,half,count);xx,yy=np.meshgrid(axis,axis);mask=xx**2+yy**2<=.003**2
            points=target+np.stack([xx,yy,np.zeros_like(xx)],axis=-1)
            values=p._pressure_abs_chunked(field,points.reshape(-1,3)).reshape(xx.shape)**2
            window.append(dict(common,half_width_m=half,samples_per_axis=count,grid_spacing_m=2*half/(count-1),roi_radius_m=.003,
                roi_points=int(mask.sum()),total_points=mask.size,roi_pressure_squared_sum=float(values[mask].sum()),
                total_pressure_squared_sum=float(values.sum()),concentration_fraction=float(values[mask].sum()/values.sum()),
                boundary_rule="equal-weight grid points; include center if r<=ROI radius; no partial cells"))
    for item in items:
        target=np.asarray(item["result"].equilibrium.target_m);distance=np.linalg.norm(ctx.positions_m-target,axis=1).min()
        domain.append(dict(task=item["row"]["task"],method=item["row"]["method"],target_id=item["row"]["target_id"],
            nearest_source_distance_m=distance,largest_projection_shell_m=max(n.fit_shell_wavelengths)*medium.sound_speed_m_s/ctx.config.frequency_hz,
            stress_surface_radius_m=n.control_radius_a*a,root_search_half_width_m=4*a,
            minimum_search_cube_z_m=target[2]-4*a,target_in_front_of_array=bool(target[2]>0),
            coordinate_scope="global analytic field; sampling ROI is not a field validity boundary"))
    _table(ctx,"revision_T8_numerical_checks",checks);_table(ctx,"revision_T8_concentration_window",window);_table(ctx,"revision_T8_domain_records",domain)


def export_results_sentences(table_directory,output_directory,mock_source=None):
    """Regenerate honest TXT audit sentences from CSV alone (no plot context)."""
    tables=Path(table_directory);out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    sentences=[];index=[]
    def add(identifier,table,row_number,message,phase=""):
        sentences.append(f"[{SMOKE}] [{identifier}] {message} [source={table}.csv; row={row_number}; phase={phase or 'not applicable'}]")
        index.append(dict(sentence_id=identifier,source_table=table,source_row_zero_based=row_number,phase_content_id=phase,evidence_status="SMOKE"))
    def load(name):return pd.read_csv(tables/(name+".csv"))
    name="revision_T1_solver_summary"
    for i,row in load(name).iterrows():
        add(f"M1.{i+1:03d}",name,i,f"{row.method}: {row.metric} median {row['median']:.6g}, IQR {row.q25:.6g}–{row.q75:.6g}, range {row.minimum:.6g}–{row.maximum:.6g}, n={row.n}; threshold attained in {row.n_threshold}/{row.n_runs} runs.")
    name="revision_T1_paired_bootstrap"
    for i,row in load(name).iterrows():
        caveat=" A degenerate observed-sample interval is not a population success guarantee." if row.ci_low==row.ci_high else ""
        add(f"M1.CI{i+1}",name,i,f"{row.metric}: {row.estimate:.6g}, paired bootstrap 95% CI [{row.ci_low:.6g}, {row.ci_high:.6g}], n={row.n_pairs}; solver interval only.{caveat}")
    name="revision_T2_branch_distance"
    for i,row in load(name).iterrows():add(f"M2.{i+1:03d}",name,i,f"{row.component} at budget {row.iteration_budget}: projector distance median {row['median']:.6g}, IQR {row.q25:.6g}–{row.q75:.6g} over {row.n} selected trajectories.")
    name="revision_T2_local_quadratic"
    for i,row in load(name).iterrows():add(f"M3.{i+1:03d}",name,i,f"{row.task} {row.method}, ±0.03 rad: quadratic-fit residual {row.quadratic_relative_residual:.6g}; section Hessian eigenvalues {row.section_hessian_min:.6g}, {row.section_hessian_max:.6g} N/m/rad².",row.phase_content_id)
    name="revision_T3_mechanics_per_target"
    for i,row in load(name).iterrows():
        if bool(row.finite_ka_root_found):
            msg=f"{row.task} {row.method} {row.target_id}: elastic equilibrium displacement ({row.dx_um:.6g}, {row.dy_um:.6g}, {row.dz_um:.6g}) µm, total {row.finite_ka_displacement_a:.6g}a; weakest stiffness {row.finite_ka_stiffness_min_n_m:.6g} N/m; target upward support/weight {row.support_ratio_at_target:.6g}; largest real linear-dynamics eigenvalue {row.dynamic_max_real_eigenvalue_s_inv:.6g} s⁻¹ under the stated Stokes-drag model."
        else:msg=f"{row.task} {row.method} {row.target_id}: no numerical root resolved; no equilibrium displacement is reported."
        add(f"M4.{i+1:03d}",name,i,msg,row.phase_content_id)
    name="revision_T3_command_summary"
    for i,row in load(name).iterrows():add(f"M4.C{i+1:03d}",name,i,f"{row.task} {row.method}: {row.n_distinct_roots}/{row.n_targets} distinct roots, {row.n_restoring} locally restoring, {row.n_linearly_stable} linearly stable; displacement median/max {row.median_displacement_a:.6g}/{row.maximum_displacement_a:.6g}a; weakest target stiffness {row.weakest_target_stiffness_n_m:.6g} N/m.",row.phase_content_id)
    name="revision_T3_surrogate_comparison"
    for i,row in load(name).iterrows():add(f"M5.{i+1:03d}",name,i,f"{row.task} {row.method} {row.target_id}: solid Gor'kov/elastic equilibrium separation {row.equilibrium_difference_um:.6g} µm; same-position Jacobian difference {row.same_position_jacobian_difference_fro_n_m:.6g} N/m.",row.phase_content_id)
    name="revision_T4_component_ablations"
    for i,row in load(name).iterrows():add(f"M6.{i+1:03d}",name,i,f"{row.configuration}, {row.target_id}, one smoke seed {row.seed}: displacement {row.finite_ka_displacement_a:.6g}a; weakest stiffness {row.finite_ka_stiffness_min_n_m:.6g} N/m; {row.iterations} iterations.",row.phase_content_id)
    name="revision_T6_discrete_phases"
    for i,row in load(name).iterrows():add(f"M8.{i+1:03d}",name,i,f"{row.task} {row.method} {row.target_id}, Q={row.levels}: wrapped RMS phase error {row.phase_rms_error_rad:.6g} rad; equilibrium displacement {row.quantized_displacement_a:.6g}a, equilibrium shift {row.equilibrium_shift_um:.6g} µm; root_found={row.root_found}.",row.quantized_phase_content_id)
    coverage=load("revision_measurement_coverage")
    for i,row in coverage.iterrows():
        if row.status!="COLLECTED SMOKE":add(f"PENDING.{row.measurement}","revision_measurement_coverage",i,f"{row.measurement}: {row.remaining}")
    header="RINENG REVISION RESULTS AUDIT\nALL CURRENT AND RECOVERED RUNS ARE SMOKE. NOT MANUSCRIPT RESULTS.\nNumeric sentences below are measured smoke outputs, not production findings.\nNaN means unavailable/unresolved, not zero. No production TXT is generated.\n\n"
    path=out/"revision_results_smoke.txt";path.write_text(header+"\n".join(sentences)+"\n",encoding="utf-8")
    pd.DataFrame(index).to_csv(out/"revision_sentence_index.csv",index=False)
    if mock_source:
        shutil.copyfile(mock_source,out/"revision_results_mock.txt")
    return path


def collect_revision_measurements(ctx):
    """Run the fixed supplementary SMOKE measurements after Figures 1–8."""
    started=time.perf_counter();items=_items(ctx)
    figure_records=[]
    for stem,figure in ctx.figures.items():
        if stem.startswith("Figure_"):
            for axis_index,axis in enumerate(figure.axes):
                figure_records.append(dict(figure=stem,axis_index=axis_index,
                    x_label=axis.get_xlabel(),y_label=axis.get_ylabel(),title=axis.get_title(),
                    x_min=axis.get_xlim()[0],x_max=axis.get_xlim()[1],
                    y_min=axis.get_ylim()[0],y_max=axis.get_ylim()[1]))
    _table(ctx,"revision_figure_axes_record",figure_records)
    stages=[("M1 convergence",lambda:_convergence(ctx)),("M2/M3 branch and local geometry",lambda:_branch_and_local(ctx)),
        ("M4/M5 mechanics and gravity",lambda:_mechanics(ctx,items)),("M6 four component configurations",lambda:_ablations(ctx)),
        ("M8 quantized commands",lambda:_quantization(ctx,items)),("M10 numerical and parameter records",lambda:_numerics(ctx,items))]
    for label,function in stages:
        tick=time.perf_counter();print("SMOKE measurements:",label,flush=True);function()
        (ctx.output_root/"measurement_progress.json").write_text(json.dumps(dict(last_completed=label,stage_seconds=time.perf_counter()-tick,total_seconds=time.perf_counter()-started,evidence_status="SMOKE"),indent=2))
    coverage=[
        ("M1","PARTIAL SMOKE","Native IB/GS update residuals and expanded independent-seed production populations remain pending."),
        ("M2","PARTIAL SMOKE","Independent cold-start, held-out class repeatability and adjacency agreement remain pending; continued-medoid clustering is descriptive."),
        ("M3","COLLECTED SMOKE","Production repetition remains required."),
        ("M4","COLLECTED SMOKE","Representative frozen commands only; independent-command population expansion remains required."),
        ("M5","COLLECTED SMOKE","Same-field solid surrogate, half-radius and gravity comparisons collected for the specified representative commands."),
        ("M6","PARTIAL SMOKE","Four component configurations run with one common seed; ten-seed weight/smoothing sweeps and retained Hybrid appendix beta test remain pending."),
        ("M7","PARTIAL SMOKE","Matched phase/timing sources retained. Ten cold/warm trials and comparable full setup/compile intervals remain pending."),
        ("M8","COLLECTED SMOKE","Q=8,16,32,64, all current benchmark commands; no independent discrete optimizer added."),
        ("M9","INTEGRATED SEPARATELY","Appendix G1-G9 supplies the active array/frequency study in the unified workflow; this legacy Main/A-F collector does not duplicate it."),
        ("M10","PARTIAL SMOKE","L/quadrature/surface/Jacobian fixed-point audits collected; refined-numerics equilibrium re-solving remains pending.")]
    _table(ctx,"revision_measurement_coverage",[dict(measurement=m,status=s,remaining=r) for m,s,r in coverage])
    _table(ctx,"revision_T7_generality_import_status",[dict(status="ACTIVE_IN_UNIFIED_APPENDIX_G",source="unified_study.py + generality/revision_f1_f2",scope="Appendix G1-G9 is integrated separately; the baseline target-position grid is not relabeled as that study")])
    # Make every exported table explicit, including inherited main-figure rows.
    for name,frame in list(ctx.tables.items()):
        frame=pd.DataFrame(frame).copy();frame["evidence_status"]="SMOKE";frame["manuscript_numerics"]=False;ctx.tables[name]=frame
    p.export_all_tables(ctx)
    # A few inherited producers write CSV directly rather than via ctx.tables.
    # Append the same smoke status while preserving their numeric text exactly.
    for path in sorted(ctx.writer.table_dir.glob("*.csv")):
        if path.stem in ctx.tables:
            continue
        with path.open(newline="",encoding="utf-8") as stream:
            reader=csv.DictReader(stream);fields=list(reader.fieldnames or []);records=list(reader)
        if not fields:
            raise ValueError(f"Missing CSV header: {path.name}")
        for column in ("evidence_status","manuscript_numerics"):
            if column not in fields:fields.append(column)
        for record in records:
            record["evidence_status"]="SMOKE";record["manuscript_numerics"]="False"
        with path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
        ctx.writer.record_file(path,"smoke-table")
    out=ctx.output_root/"revision_measurements";out.mkdir(exist_ok=True)
    result=export_results_sentences(ctx.writer.table_dir,out,ctx.data_root/"revision_results_mock.txt")
    for path in out.iterdir():ctx.writer.record_file(path,"revision-result-sentences")
    print(f"SMOKE measurement export complete in {time.perf_counter()-started:.1f} s: {result}",flush=True)
    return result
