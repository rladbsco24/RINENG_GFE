"""Full-run orchestration. Importing this module never starts a calculation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parent


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def configure(root=ROOT):
    """Set device/thread choices before NumPy, SciPy or JAX are imported."""
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "runtime/src"))
    from production_settings import load_settings
    requested = os.environ.get("GFE_JAX_PLATFORM")
    overrides = None
    if requested:
        overrides = {"device": {"cuda": "gpu", "gpu": "gpu", "cpu": "cpu",
                                "auto": "auto"}[requested]}
    settings = load_settings(root, overrides)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(settings["blas_threads"])
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["JAX_ENABLE_X64"] = "true"
    if settings["device"] == "auto":
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["JAX_PLATFORMS"] = "cuda" if settings["device"] == "gpu" else "cpu"
    os.environ["HAT_HIDE_MODE_NOTE"] = "1"
    os.environ["HAT_PROTOTYPE_LABEL"] = "0"
    return settings


def prepare(root=ROOT):
    root = Path(root).resolve()
    settings = configure(root)
    from hat_revision_pipeline import production_main
    # Reuse the established atomic scientific-figure export implementation.
    import run_study
    run_study._install_atomic_figure_exports()
    ctx = production_main.prepare(root, settings=settings)
    ctx.memo["production_root"] = root
    ctx.memo["production_settings"] = settings
    if not ctx.config.full or ctx.config.multitrap_maxiter != 30000:
        raise RuntimeError("The full numerical protocol and Triple 30000 ceiling are required")
    atomic_json(ctx.output_root / "resolved_production_settings.json", settings)
    return ctx


@contextmanager
def stage(ctx, name):
    """Checkpoint orchestration status; numerical producers own their exact caches."""
    path = ctx.output_root / "execution_progress.json"
    state = json.loads(path.read_text()) if path.exists() else {"stages": {}}
    state.update(status="running", protocol=ctx.memo["production_settings"]["protocol_version"])
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    state["stages"][name] = record
    atomic_json(path, state)
    print(f"Starting {name}", flush=True)
    try:
        yield
    except BaseException as error:
        record.update(status="failed", error=type(error).__name__, message=str(error),
                      traceback=traceback.format_exc())
        state["status"] = "failed"
        raise
    else:
        record.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        print(f"Completed {name}", flush=True)
    finally:
        atomic_json(path, state)


def main_figure(ctx, number):
    if int(number) == 5:
        raise ValueError(
            "The obsolete concentration-versus-stiffness graphic is disabled; "
            "its numerical benchmark records remain available."
        )
    from hat_revision_pipeline import production_main
    with stage(ctx, f"Main {number}"):
        return production_main.main_figure(ctx, number)


def appendix_a(ctx):
    from hat_revision_pipeline import appendix_a_production
    with stage(ctx, "Appendix A"):
        return appendix_a_production.run_production(ctx.memo["production_root"], ctx=ctx,
            workers=ctx.memo["production_settings"]["workers"])


def appendix_b_and_weights(ctx):
    from hat_revision_pipeline import production_ab
    with stage(ctx, "Appendix B and shared sensitivity tables"):
        return production_ab.run(ctx.memo["production_root"], ctx=ctx,
            workers=ctx.memo["production_settings"]["workers"],render_weight_figures=False)


def appendix_b3(ctx):
    from hat_revision_pipeline import production_ab
    with stage(ctx, "Appendix B3"):
        return production_ab.run_b3(ctx.memo["production_root"], ctx=ctx)


def appendix_c(ctx):
    from hat_revision_pipeline import production_c
    with stage(ctx, "Appendix C1-C3 and C6"):
        return production_c.run(ctx.memo["production_root"], ctx=ctx)


def appendix_support(ctx):
    from hat_revision_pipeline import production_support
    with stage(ctx, "Appendices D and E"):
        return production_support.run(ctx.memo["production_root"], ctx=ctx)


def retained_rhg_report(ctx):
    """Retain the completed one-pair C5 follow-up without authorizing more solves."""
    source = Path(ctx.memo["production_root"]) / "appendix_outputs/C_GFE/RHg_check"
    destination = ctx.output_root / "appendices/C_GFE/RHg_check_archived"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    atomic_json(destination / "production_inclusion.json", {
        "status": "retained previously executed single-pair check",
        "new_optimizations": 0,
        "validation_protocol": "Archived C5/B3 protocol; not relabeled as production numerics",
        "report": "REPORT.md",
    })
    return destination / "REPORT.md"


def finalize(ctx):
    from production_reporting import finalize as report
    with stage(ctx, "Parameters, data exports, and complete PDF"):
        result = report(ctx.memo["production_root"], ctx)
    path = ctx.output_root / "execution_progress.json"
    state = json.loads(path.read_text())
    state.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(path, state)
    return result


def run(root=ROOT):
    ctx = prepare(root)
    # Produce the removed concentration/stiffness panel's numerical
    # records without exporting that obsolete visual.
    from hat_revision_pipeline import production_main
    production_main.ensure_benchmarks(ctx)
    for number in (1, 2, 3, 4, 6, 7, 8):
        main_figure(ctx, number)
    appendix_a(ctx)
    appendix_b_and_weights(ctx)
    appendix_b3(ctx)
    appendix_c(ctx)
    appendix_support(ctx)
    return finalize(ctx)


if __name__ == "__main__":
    run()
