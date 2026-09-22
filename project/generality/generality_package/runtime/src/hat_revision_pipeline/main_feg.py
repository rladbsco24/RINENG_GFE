"""GFE-primary main Figures 1--4, with authoritative FE controls preserved.

Numerical producers explicitly construct a nonzero force target and distinct
cache payloads. The established figure functions use an internal ``FE`` slot;
only during composition that slot receives the already-solved GFE command.
The save adapter labels every displayed object and exported main table GFE.
It never changes the objective or solver through a plotting-time label.
Run these figure producers serially, as the original composition API is global.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, replace
import math
import re
import numpy as np
import pandas as pd
from matplotlib.text import Text

from . import pipeline as p
from . import final_figures as ff
from . import fig1_domain_volume as domain
from .cache import digest_array
from .gorkov_core import Condition, SingleTargetObjective
from .fixed_fe_endpoints import fixed_single_fe_method_spec
from .multitrap import SphereInFluid, effective_weight_force_target_n, run_multitrap
from .sota import solve_corrected_gorkov_fe
from .main_feg_branch import fixed_feg_branch_pack

_ORIGINAL_SINGLE_BANK = ff._single_vortex_bank
_ORIGINAL_DOMAIN_BANK = domain.figure_1_xz_domain_bank
_ORIGINAL_TRIPLE_BANK = ff.ensure_main_triple_endpoint
_ORIGINAL_SAVE = ff._save


def feg_single_objective(ctx, target=None):
    """Keep the accepted Single formulation and prescribe effective weight."""
    target = p.MAIN_TARGET_M if target is None else np.asarray(target, dtype=float)
    method = fixed_single_fe_method_spec()
    condition = Condition(label="Single-GFE-alpha10", positions=ctx.positions_m,
        target_m=tuple(target), frequency_hz=ctx.config.frequency_hz,
        curvature_weight=method.curvature_weight, normals=ctx.normals)
    force_target = effective_weight_force_target_n(
        SphereInFluid(frequency_hz=ctx.config.frequency_hz))
    return SingleTargetObjective(condition, method, force_target_n=force_target)


def _solve(ctx, objective, initial, seed, role):
    payload = dict(contract="main-feg-same-start-v1", role=role,
        objective_method=asdict(objective.method), force_target_n=objective.force_target_n.tolist(),
        target_m=list(objective.condition.target_m), seed=int(seed),
        initial_phase_fingerprint=digest_array(initial), positions=digest_array(ctx.positions_m),
        normals=digest_array(ctx.normals), frequency_hz=float(ctx.config.frequency_hz),
        maxiter=10000, gtol=float(ctx.config.gtol))
    run, status, _ = ctx.cache.get_or_compute("main_feg_single_run", payload,
        lambda: solve_corrected_gorkov_fe(objective, initial, maxiter=10000, gtol=ctx.config.gtol),
        recompute=ctx.recompute)
    return run, status


def _updated_row(row, run, objective, status):
    row = dict(row)
    row.update(method="GFE", objective_method="GFE", alpha_per_m=10.,
        optimizer=run.optimizer, evaluator_backend="compact-single-float64-v1",
        iterations=int(run.iterations), evaluations=int(run.evaluations),
        wall_time_s=float(run.history_wall_s[-1]), objective=float(run.objective),
        gradient_norm=float(run.gradient_norm), terminal_gradient_norm=float(run.gradient_norm),
        terminal_gradient_rms=float(run.gradient_norm/math.sqrt(objective.n_transducers-1)),
        optimizer_success=bool(run.success), optimizer_status=int(run.status),
        optimizer_message=str(run.message), phase_fingerprint=digest_array(run.phase_rad),
        cache_status=status, near_stationary=bool(run.gradient_norm<=ff.REPORT_GRADIENT_NORM),
        force_target_z_n=float(objective.force_target_n[2]),
        force_residual_definition="F_Gorkov - effective_weight * e_z")
    return row


def feg_single_bank(ctx):
    key = "main_feg_single_bank"
    if key in ctx.memo:
        return ctx.memo[key]
    base = _ORIGINAL_SINGLE_BANK(ctx)
    ctx.tables["main_fe_control_single_runs"] = base["table"].copy()
    rows, runs = [], {}
    for row in base["table"].to_dict("records"):
        index = int(row["pair_index"])
        if row["method"] == "Conventional":
            rows.append(row); runs[("Conventional", index)] = base["runs"][("Conventional",index)]
        else:
            seed=int(row["seed"]); objective=feg_single_objective(ctx)
            initial=np.random.default_rng(seed).uniform(-np.pi,np.pi,objective.n_transducers)
            run,status=_solve(ctx,objective,initial,seed,"Figure1 paired central target")
            rows.append(_updated_row(row,run,objective,status)); runs[("GFE",index)]=run
    table=pd.DataFrame(rows)
    ctx.tables["main_feg_single_runs"] = table.copy()
    result=dict(runs=runs,table=table,seeds=base["seeds"],caps={"Conventional":10000,"GFE":10000})
    ctx.memo[key]=result
    return result


def _single_plot_adapter(ctx):
    bank=feg_single_bank(ctx)
    table=bank["table"].copy(); table["method"]=table.method.replace({"GFE":"FE"})
    ctx.tables["fig1_single_vortex_iterations_for_table_only"] = table[["method", "pair_index", "seed", "iterations", "evaluations", "iteration_cap"]].copy()
    runs={("FE" if m=="GFE" else m,i):run for (m,i),run in bank["runs"].items()}
    return dict(bank,table=table,runs=runs,caps={"Conventional":10000,"FE":10000})


def feg_xz_domain_bank(ctx):
    key="main_feg_xz_domain"
    if key in ctx.memo: return ctx.memo[key]
    base=_ORIGINAL_DOMAIN_BANK(ctx)
    ctx.tables["main_fe_control_xz_domain"] = base["raw"].copy()
    rows=[]
    for old in base["raw"].to_dict("records"):
        if old["method"]=="Conventional": rows.append(old);continue
        target=np.asarray([old["target_x_m"],old["target_y_m"],old["target_z_m"]])
        objective=feg_single_objective(ctx,target);seed=int(old["seed"])
        initial=np.random.default_rng(seed).uniform(-np.pi,np.pi,objective.n_transducers)
        run,status=_solve(ctx,objective,initial,seed,"Figure1 Cartesian XZ target")
        rows.append(_updated_row(old,run,objective,status))
    raw=pd.DataFrame(rows)
    summary=raw.groupby(["method","alpha_per_m","target_id","target_x_m","target_y_m","target_z_m"],sort=True,dropna=False).agg(
        run_count=("restart","size"),wall_time_median_s=("wall_time_s","median"),
        wall_time_q25_s=("wall_time_s",lambda x:x.quantile(.25)),wall_time_q75_s=("wall_time_s",lambda x:x.quantile(.75)),
        terminal_gradient_rms_median=("terminal_gradient_rms","median"),
        terminal_gradient_rms_q25=("terminal_gradient_rms",lambda x:x.quantile(.25)),
        terminal_gradient_rms_q75=("terminal_gradient_rms",lambda x:x.quantile(.75))).reset_index()
    definition=base["definition"].copy();definition["primary_method"]="GFE"
    ctx.tables["fig1_xz_domain_runs"]=raw.copy();ctx.tables["fig1_xz_domain_summary"]=summary.copy()
    ctx.tables["fig1_xz_domain_definition"]=definition
    value=dict(raw=raw,summary=summary,definition=definition)
    ctx.memo[key]=value
    return value


def _domain_plot_adapter(ctx):
    value=feg_xz_domain_bank(ctx);summary=value["summary"].copy()
    summary["method"]=summary.method.replace({"GFE":"FE"})
    return dict(value,summary=summary)


def _conventional_decision_run(ctx,branch):
    rx,ry=branch["bank"].root_target
    selected=branch["evolution"].query("component == 'B1' and ix == @rx and iy == @ry")
    index=int(selected.iloc[0]["selected_trajectory_index"])
    seed=int(ctx.artifact.trajectory_table.iloc[index]["seed"])
    objective=p._objective_at(ctx,"Conventional")
    initial=np.random.default_rng(seed).uniform(-np.pi,np.pi,objective.n_transducers)
    payload=dict(contract="native-terminal-gradient-and-last-accepted-step-v2-alpha-keyed",
        method="Conventional",objective_method=asdict(objective.method),target_m=p.MAIN_TARGET_M.tolist(),
        seed=seed,initial_phase=digest_array(initial),maxiter=int(ctx.config.conventional_maxiter),
        gtol=ctx.config.gtol,array=digest_array(ctx.positions_m))
    run,_,_=ctx.cache.get_or_compute("decision_endpoint_run",payload,
        lambda:solve_corrected_gorkov_fe(objective,initial,maxiter=ctx.config.conventional_maxiter,gtol=ctx.config.gtol),
        recompute=ctx.recompute)
    return run


def _decision_plot_adapter(ctx):
    key="main_feg_decision_pack"
    if key in ctx.memo:return ctx.memo[key]
    branch=fixed_feg_branch_pack(ctx);rx,ry=branch["bank"].root_target
    anchors=branch["anchors"]
    match=anchors.query("component == 'B1' and ix == @rx and iy == @ry")
    if len(match)!=1:raise RuntimeError("One frozen GFE B1 root landmark is required")
    runs={"Conventional":_conventional_decision_run(ctx,branch),"FE":branch["anchor_results"][int(match.index[0])]}
    objectives={"Conventional":p._objective_at(ctx,"Conventional"),"FE":feg_single_objective(ctx)}
    sections={}
    for label in runs:
        directions=p._native_local_directions(objectives[label],runs[label])
        sections[label]=p._native_relative_sections(objectives[label],runs[label],directions,line_points=101,plane_points=101 if ctx.memo.get("production_main") else 51)
    value=dict(runs=runs,objectives=objectives,sections=sections)
    ctx.memo[key]=value
    return value


def feg_triple_run(ctx):
    """Use the same Triple formulation with the current matched main budget."""
    key="main_feg_triple_run"
    if key in ctx.memo:return ctx.memo[key]
    bank=_ORIGINAL_TRIPLE_BANK(ctx);node=bank["objects"][(1,1)]
    problem=node["problem"];base_config=node["fe_config"]
    config=replace(base_config,compensate_effective_gravity=True)
    initial=np.asarray(bank["initial_phase"],dtype=float);seed=int(bank["seed"])
    target=effective_weight_force_target_n(problem.material)
    cap=sum(config.smooth_stage_maxiters)
    payload=dict(contract="fig6-fixed-effective-gravity-compensated-triple-fe-v1",problem=problem.fingerprint,
        base_config=base_config.to_payload(),compensated_config=config.to_payload(),seed=seed,
        initial_phase_fingerprint=digest_array(initial),fixed_iteration_cap=cap,
        fixed_upward_force_target_n=target.tolist())
    run,status,_=ctx.cache.get_or_compute("final_fig6_vertical_compensated_fe",payload,
        lambda:run_multitrap(problem,config,seed=seed,initial_phases=initial,record_history=True,
          metadata=dict(display_method="GFE",experiment_role="GFE-primary Triple main reference",
            fixed_iteration_cap=cap,fixed_upward_force_target_n=target.tolist())),recompute=ctx.recompute)
    result=dict(run=run,config=config,phase=np.asarray(run.phases),bank=bank,cache_status=status)
    ctx.memo[key]=result
    return result


def _triple_plot_adapter(ctx):
    result=feg_triple_run(ctx);bank=result["bank"];node=dict(bank["objects"][(1,1)])
    ctx.tables["main_fe_control_triple_command"]=pd.DataFrame([dict(method="FE",
        phase_fingerprint=digest_array(node["FE_phase"]),iterations=node["FE"].iterations,
        alpha_force=node["fe_config"].alpha_force,compensate_effective_gravity=False)])
    node.update(FE=result["run"],FE_phase=result["phase"],fe_config=result["config"])
    return dict(bank,objects={(1,1):node})


def _replace_fe_text(value):
    return re.sub(r"(?<![A-Za-z])FE(?![A-Za-z+])","GFE",value)


@contextmanager
def _compose(ctx,number,title,**overrides):
    previous={name:getattr(ff,name) for name in overrides}
    previous_color=ff.COLORS["FE"]
    prefixes=(f"fig{number}_",f"figure_{number}_")
    def save(context,figure,stem):
        for artist in figure.findobj(Text):
            artist.set_text(_replace_fe_text(artist.get_text()))
        for name,table in list(context.tables.items()):
            if not name.startswith(prefixes):continue
            frame=pd.DataFrame(table).copy()
            for column in frame.select_dtypes(include="object"):
                frame[column]=frame[column].map(lambda v:_replace_fe_text(v) if isinstance(v,str) else v)
            frame["primary_method"]="GFE"
            if number==2:
                frame["reference_method"]="GFE"
                for metric in ("similarity","distance"):
                    column=f"{metric}_to_matched_fe"
                    if column in frame:frame[f"{metric}_to_matched_feg"]=frame[column]
            context.tables[name]=frame
        return _ORIGINAL_SAVE(context,figure,stem)
    try:
        for name,value in overrides.items():setattr(ff,name,value)
        ff._save=save;ff.COLORS["FE"]=ff.COLORS.get("GFE", "#007C78")
        yield
    finally:
        for name,value in previous.items():setattr(ff,name,value)
        ff._save=_ORIGINAL_SAVE;ff.COLORS["FE"]=previous_color


def figure_1_feg(ctx):
    """Matched Single convergence and pressure fields, GFE primary."""
    with _compose(ctx,1,"Figure 1. GFE convergence and Single pressure fields",
        _single_vortex_bank=_single_plot_adapter,figure_1_xz_domain_bank=_domain_plot_adapter):
        return ff.figure_1_single_vortex_convergence(ctx)


def figure_2_feg(ctx):
    """Freeze independently solved GFE landmarks; project Conventional OOS."""
    with _compose(ctx,2,"Figure 2. Conventional correspondence to fixed GFE endpoints",
        fixed_single_branch_pack=fixed_feg_branch_pack):
        return ff.figure_2_two_branch_evolution(ctx)


def figure_3_feg(ctx):
    """Actual GFE objective sections at its solved root endpoint."""
    with _compose(ctx,3,"Figure 3. Local decision geometry: Conventional and GFE",
        _decision_pack=_decision_plot_adapter):
        return ff.figure_3_local_decision_geometry(ctx)


def figure_4_feg(ctx):
    """Compensated Triple fields and objective on unchanged archived axes."""
    with _compose(ctx,4,"Figure 4. Triple fields and local decision geometry",
        ensure_main_triple_endpoint=_triple_plot_adapter):
        return ff.figure_4_triple_fields_and_decision(ctx)


MAIN_FEG_FIRST_FOUR=(figure_1_feg,figure_2_feg,figure_3_feg,figure_4_feg)
