"""End-to-end engineering timing with the global FE/GFE solver policy.

Each mode uses its declared command population. FE/GFE uses compact evaluation
with L-BFGS-B for Single and BFGS for Triple. Conventional and prescribed/AD methods
keep their native solvers.
The objective, iteration ceilings, physics, and independent validators are not
changed here.  No computation takes place on import.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import pipeline as p, production_main as production
from .cache import digest_payload
from .triple_long_iteration import SEED_OFFSET


def prepare_smoke_timing(ctx):
    """Attach the same timing boundary as full without changing smoke settings."""
    if ctx.config.full:
        return
    if "engineering_timing_campaign" in ctx.memo:
        return
    settings = dict(blas_threads=1, device=ctx.memo.get("notebook_device", "cpu"))
    host = production._machine_identity(settings)
    if not host["blas_thread_contract_satisfied"]:
        raise RuntimeError(
            "Engineering timing requires one BLAS thread; restart through "
            "run_notebook.prepare before importing scientific libraries."
        )
    if settings["device"] in ("cpu", "gpu") and host["jax_backend"] != settings["device"]:
        raise RuntimeError("The requested engineering timing device is unavailable")
    campaign = "gfe-smoke-compact-v3-task-specific-20260920-" + digest_payload(host)[:20]
    ctx.memo.update(
        engineering_timing_campaign=campaign,
        production_timing_campaign=campaign,
        benchmark_timing_campaign=campaign,
        production_machine=host,
    )


def _single_seed(ctx):
    """Use the established first central Conventional seed, before outcomes."""
    ix, iy = ctx.branch_bank.root_target
    rows = ctx.artifact.trajectory_table
    selected = rows[
        rows["method"].eq("Conventional") & rows["ix"].eq(ix) & rows["iy"].eq(iy)
    ].sort_values("seed")
    if selected["seed"].nunique() != 10:
        raise RuntimeError("Expected the unchanged ten-seed central reference bank")
    return int(selected.iloc[0]["seed"])


def ensure_engineering_commands(ctx):
    """Retain one fresh-or-exact-cache command per smoke method/task."""
    if ctx.config.full:
        return production.ensure_command_bank(ctx)
    key = "engineering_commands"
    if key in ctx.memo:
        return ctx.memo[key]
    prepare_smoke_timing(ctx)
    seeds = dict(Single=_single_seed(ctx), Triple=int(ctx.config.random_seed + SEED_OFFSET))
    bank = {}
    observations = []
    for task in ("Single", "Triple"):
        for method in production.METHODS:
            seed = seeds[task] if method in ("Conventional", "FE", "GFE") else (
                int(ctx.config.random_seed) if method == "AD" else None
            )
            if method in ("Conventional", "FE", "GFE"):
                solve = production.ensure_single_run if task == "Single" else production.ensure_triple_run
                value = solve(ctx, method, seed, role="engineering")
            else:
                value = production._holography_run(ctx, task, method, seed, repeat=0)
            value = dict(value, row=dict(value["row"],
                role="engineering", run_mode="smoke", evidence_status="PAPER-SCALE SMOKE",
                manuscript_numerics=False, timing_observation_count=1))
            if not bool(value["row"]["timing_comparable"]):
                raise RuntimeError(
                    f"{task} {method} has non-comparable timing: "
                    f"{value['row']['timing_exclusion_reason']}"
                )
            bank[(task, method, seed)] = value
            observations.append(dict(value["row"], timing_observation_index=0))
            production._save_command(ctx, task, method,
                seed if seed is not None else "deterministic", value, suffix="engineering")
    ctx.tables["engineering_command_runs"] = pd.DataFrame([value["row"] for value in bank.values()])
    ctx.tables["engineering_timing_observations"] = pd.DataFrame(observations)
    ctx.tables["engineering_timing_protocol"] = pd.DataFrame([dict(
        role="engineering", mode="smoke", selected_single_fe_optimizer="L-BFGS-B",
        selected_triple_fe_optimizer="BFGS",
        conventional_optimizer="BFGS", observations_per_method_task=1,
        single_seed=seeds["Single"], triple_seed=seeds["Triple"],
        single_iteration_cap=10000, triple_iteration_cap=30000,
        timing_protocol_id=production.TIMING_PROTOCOL_ID, timing_scope=production.TIMING_SCOPE,
        machine_record_json=json.dumps(ctx.memo["production_machine"], default=str),
    )])
    ctx.memo[key] = bank
    return bank


def ensure_engineering_benchmarks(ctx):
    """Independently validate the current commands and retain matching timings."""
    if ctx.config.full:
        return production.ensure_benchmarks(ctx)
    key = "engineering_benchmarks"
    if key in ctx.memo:
        return ctx.memo[key]
    bank = ensure_engineering_commands(ctx)
    output = {}
    for task in ("Single", "Triple"):
        specs, evaluations = [], []
        for (bank_task, method, seed), value in bank.items():
            if bank_task != task:
                continue
            spec = production._spec(task, method, seed, value)
            spec.update(role="engineering", optimizer=value["row"].get("optimizer", "native"),
                phase_source="current command under the task-specific compact FE/GFE solver policy")
            specs.append(spec)
            for index, target in enumerate(value["targets"]):
                item = production._validate_spec(ctx, spec, index, target)
                item["row"].update(role="engineering", run_mode="smoke",
                    evidence_status="PAPER-SCALE SMOKE", manuscript_numerics=False,
                    optimizer=spec["optimizer"])
                evaluations.append(item)
        table = pd.DataFrame([item["row"] for item in evaluations])
        output[task] = dict(raw=table, table=table, all_table=table,
            concentration_table=table, specs=specs, method_specs=tuple(specs),
            results=[item["result"] for item in evaluations], evaluations=evaluations,
            targets_m=np.asarray([p.MAIN_TARGET_M] if task == "Single" else production.FIXED_BASE_TARGETS_M))
        ctx.tables[f"engineering_{task.lower()}_target_mechanics"] = table
    ctx.memo[key] = output
    p.export_all_tables(ctx)
    return output
