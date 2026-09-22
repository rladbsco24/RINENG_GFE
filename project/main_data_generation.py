"""Generate current main-study data before rendering the accepted figures.

The deposited historical phase banks are immutable scientific inputs. They
define the accepted branch chart and the historical Appendix A/B/D/E controls.
The current GFE/FE/Conventional commands, mechanical validation and tables are
produced by the same numerical functions used by the figure cells. No image or
PDF is a numerical input to this entry point.
"""
from __future__ import annotations

from datetime import datetime, timezone

from rineng_content_id import content_identity
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


SCHEMA = "unified-data-generation-20260908-v1"
MAIN_STAGES = ("single", "domain", "branch", "triple", "benchmarks", "alpha")
REFERENCE_FILES = (
    "anchor_config.json", "anchor_endpoints.csv", "anchor_phases.npz",
    "conventional_trajectories.csv", "conventional_snapshots.npz",
    "evolution_metadata.json",
)


def _content_id(path):
    digest = content_identity()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def reference_inventory(root):
    """Verify the required deposited inputs and expose their exact identities."""
    root = Path(root).resolve()
    directory = root / "runtime/data/corrected_branch_evolution"
    records = []
    for name in REFERENCE_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Required deposited branch input is missing: {path}. "
                "Restore it from the notebook payload; its historical builder "
                "is not part of the current numerical campaign."
            )
        records.append({"path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size, "content_id": _content_id(path)})
    return {
        "schema": SCHEMA,
        "branch_reference_files": records,
        "reference_role": "Historical phases identify fixed target-local branch initializations; current GFE landmarks are reoptimized at alpha=10.",
        "historical_bank_cold_rebuild": "Not implemented; audited deposited raw inputs are required and their original provenance guards remain active.",
        "other_deposited_controls": {
            "A": "Paired historical FE and Conventional commands; force samples and elastic/Gor'kov equilibria are regenerated or resumed.",
            "B": "Deposited matched Twin/Bottle commands; pressure slices and independent elastic mechanics are regenerated or resumed.",
            "D": "Historical FE commands and retained current GFE quantization selections; validation and full-mode transition timings use their own producers.",
            "E": "Deposited prescription candidate bank; fields, comparisons and full-mode validation use their own producers.",
        },
    }


def _fresh_workspace(root):
    """Copy source and reference inputs without current main/C/F replay caches.

    A fresh kernel is required so imported modules cannot keep the source path
    of a previous workspace. Historical command inputs are deliberately kept.
    """
    if "hat_revision_pipeline.pipeline" in sys.modules or "run_notebook" in sys.modules:
        raise RuntimeError("Select cache_policy='fresh' in a fresh kernel, before importing the study runtime.")
    destination = Path(tempfile.mkdtemp(prefix=root.name + "_fresh_", dir=root.parent))
    excluded = (
        "unified_outputs", "full_runs", "smoke_runs",
        "smoke_outputs", "supersmoke_outputs", "production_outputs", "reviewer_robustness_cache",
        "appendix_outputs", "appendix_B_pressure/figures",
        "appendix_B_pressure/data/cache", "appendix_B_pressure/data/B3",
        "appendix_B_pressure/data/solution_manifest.json",
        "appendix_B_pressure/data/solver_results.csv",
        "appendix_B_pressure/data/pressure_slices.npz",
        "appendix_B_pressure/data/pressure_slice_provenance.json",
        "appendix_B_pressure/B2_field_summary.csv",
        "appendix_B_pressure/figure_provenance.json",
        "focused_smoke_input_manifest.json",
    )

    def ignore(directory, names):
        relative = Path(directory).relative_to(root)
        dropped = []
        for name in names:
            path = (relative / name).as_posix()
            if name == "__pycache__" or any(path == prefix or path.startswith(prefix + "/") for prefix in excluded):
                dropped.append(name)
        return dropped

    shutil.copytree(root, destination, dirs_exist_ok=True, ignore=ignore)
    _json(destination / "fresh_workspace.json", {
        "schema": SCHEMA, "source_workspace": str(root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_current_results": excluded,
        "immutable_reference_commands_retained": True,
    })
    return destination


def prepare(root, mode="smoke", device="cpu", cache_policy="resume", study_scale=None, cache_source_root=None):
    """Prepare one supersmoke/smoke/full context with explicit cache policy.

    resume reuses exact numerical cache keys and computes missing records.
    fresh creates a separate workspace and omits current main/C/F results;
    it retains the deposited historical reference banks. Neither policy uses
    existing figure files to generate data.
    """
    if cache_policy not in {"resume", "fresh"}:
        raise ValueError("cache_policy must be 'resume' or 'fresh'")
    root = Path(root).resolve()
    if cache_policy == "fresh":
        root = _fresh_workspace(root)
    sys.path.insert(0, str(root))
    from cache_reuse import stage_existing_caches
    cache_reuse = stage_existing_caches(
        root, cache_source_root if cache_policy == "resume" else None
    )
    inventory = reference_inventory(root)
    import run_notebook
    ctx = run_notebook.prepare(root, mode=mode, device=device, study_scale=study_scale)
    ctx.memo.update(data_generation_schema=SCHEMA, cache_policy=cache_policy,
                    regenerate_smoke_analysis=True)
    inventory.update(mode=mode, cache_policy=cache_policy, cache_reuse=cache_reuse,
                     output_root=str(ctx.output_root),
                     main_stages=("lazy main-figure dependencies" if mode == "supersmoke"
                                  else MAIN_STAGES),
                     single_iteration_cap=10000, triple_iteration_cap=30000,
                     analysis_sections={"C": "Sensitivity", "F": "Optimizer robustness"})
    _json(ctx.output_root / "data_generation_inputs.json", inventory)
    return ctx


def _stage_producer(ctx, stage):
    from hat_revision_pipeline import main_feg as mf, final_figures as ff
    import main_feg_benchmarks as benchmarks
    if ctx.config.full:
        from hat_revision_pipeline import production_main as production
        if stage in {"single", "triple"}:
            return production.ensure_command_bank(ctx)
        if stage == "domain":
            return production._domain_plot_bank(ctx)
        if stage == "branch":
            return production._branch_pack(ctx)
        if stage == "benchmarks":
            return production.ensure_benchmarks(ctx)
        if stage == "alpha":
            return production._alpha_display_bank(ctx)
    else:
        if stage == "single":
            return mf.feg_single_bank(ctx)
        if stage == "domain":
            return mf.feg_xz_domain_bank(ctx)
        if stage == "branch":
            from hat_revision_pipeline.main_feg_branch import fixed_feg_branch_pack
            return fixed_feg_branch_pack(ctx)
        if stage == "triple":
            from hat_revision_pipeline.main_triple_endpoint import ensure_main_triple_endpoint
            return {"uncompensated": ensure_main_triple_endpoint(ctx),
                    "GFE": mf.feg_triple_run(ctx)}
        if stage == "benchmarks":
            from hat_revision_pipeline.engineering_benchmarks import ensure_engineering_benchmarks
            # All banks use the global FE/GFE solver policy. Keep the existing
            # figure populations and their respective timing scopes.
            controlled = {"Single": ff._exact_benchmark_with_concentration(ctx),
                          "Triple": ff._triple_exact_benchmark_with_concentration(ctx)}
            if ctx.memo.get("final_benchmark_owned_by_expanded_context"):
                return {"field_validation": controlled}
            return {"field_validation": controlled,
                    "benchmark": ensure_engineering_benchmarks(ctx)}
        if stage == "alpha":
            return benchmarks.alpha_bank(ctx)
    raise ValueError(f"Unknown main data stage: {stage}")


def prepare_baseline_context(ctx):
    """Prepare unchanged-scale outputs inside one full run; execute no producers."""
    if not ctx.config.full:
        return ctx
    if "baseline_context" in ctx.memo:
        return ctx.memo["baseline_context"]
    import run_notebook
    root = Path(ctx.memo["notebook_root"]).resolve()
    environment = dict(os.environ)
    try:
        baseline = run_notebook.prepare(
            root, mode="smoke", device=ctx.memo.get("notebook_device", "cpu")
        )
        baseline.memo.update(
            cache_policy=ctx.memo.get("cache_policy", "resume"),
            regenerate_smoke_analysis=True,
            final_benchmark_owned_by_expanded_context=True,
        )
    finally:
        for name in set(os.environ) - set(environment):
            del os.environ[name]
        os.environ.update(environment)
    ctx.memo["baseline_context"] = baseline
    return baseline


def generate(ctx, stages=MAIN_STAGES):
    """Generate or resume numerical data before the main figure cells.

    Figure-specific pressure/decision meshes are generated in their own figure
    cells, using these same commands and cache identities. Appendix C and F
    retain separate numerical/figure cells; prepare disables their report-only
    smoke shortcut, so they also execute their numerical cache-aware producers.
    """
    if ctx.memo.get("notebook_mode") == "supersmoke":
        progress = {"schema": SCHEMA, "mode": "supersmoke",
                    "cache_policy": ctx.memo.get("cache_policy", "resume"),
                    "status": "deferred to main-figure producers", "stages": {},
                    "scope": "Only dependencies requested by the eight main plots; no appendices or unplotted audits."}
        _json(ctx.output_root / "data_generation_progress.json", progress)
        return progress
    from hat_revision_pipeline import pipeline
    import main_feg_benchmarks as benchmarks
    stages = tuple(stages)
    if any(stage not in MAIN_STAGES for stage in stages):
        raise ValueError(f"stages must be selected from {MAIN_STAGES}")
    progress_path = ctx.output_root / "data_generation_progress.json"
    progress = {"schema": SCHEMA, "mode": "full" if ctx.config.full else "smoke",
                "cache_policy": ctx.memo.get("cache_policy", "resume"), "stages": {}}
    for stage in stages:
        print(f"Generating or resuming main data: {stage}", flush=True)
        start = time.perf_counter()
        before = len(getattr(ctx.cache, "access_records", []))
        record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
        progress["stages"][stage] = record
        _json(progress_path, progress)
        try:
            _stage_producer(ctx, stage)
            if ctx.config.full:
                # The legacy public-caption adapter contains measured smoke
                # values; do not write those values into a new full campaign.
                import pandas as pd
                for name, table in list(ctx.tables.items()):
                    if isinstance(table, pd.DataFrame):
                        ctx.tables[name] = benchmarks.with_si_displacements(table)
            else:
                benchmarks.public_tables(ctx)
            pipeline.export_all_tables(ctx)
        except BaseException as error:
            record.update(status="failed", error=type(error).__name__, message=str(error))
            _json(progress_path, progress)
            raise
        access = getattr(ctx.cache, "access_records", [])[before:]
        record.update(status="completed", elapsed_s=time.perf_counter() - start,
                      cache_accesses=len(access),
                      cache_status_counts={value: sum(row["status"] == value for row in access)
                                           for value in sorted({row["status"] for row in access})},
                      exported_tables=len(ctx.tables))
        _json(progress_path, progress)
        print(f"Completed main data: {stage}", flush=True)
    return progress
