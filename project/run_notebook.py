"""Choose a plot preview, the accepted smoke study, or focused full sampling."""
from pathlib import Path
import json
import os
import sys


def prepare(root, mode="smoke", device="cpu", study_scale=None):
    root = Path(root).resolve()
    if mode not in {"supersmoke", "smoke", "full"}:
        raise ValueError("MODE must be 'supersmoke', 'smoke', or 'full'")
    if device not in {"cpu", "gpu", "auto"}:
        raise ValueError("DEVICE must be 'cpu', 'gpu', or 'auto'")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(MPLBACKEND="Agg", JAX_ENABLE_X64="true",
                      HAT_HIDE_MODE_NOTE="1", HAT_PROTOTYPE_LABEL="0",
                      GFE_JAX_PLATFORM=device)
    if device == "auto":
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "runtime/src"))
    if mode == "full":
        import run_production
        ctx = run_production.prepare(root)
    else:
        import run_study
        ctx = (run_study.prepare(root, output_root=root / "supersmoke_outputs")
               if mode == "supersmoke" else run_study.prepare(root))
    ctx.memo.update(notebook_mode=mode, notebook_root=root, notebook_device=device)
    from run_profiles import apply
    apply(ctx, mode, overrides=study_scale)
    record = {"mode": mode, "device": device, "primary_objective": "GFE",
              "entry_point": "run_notebook.py",
              "generality": ("SKIPPED: main-figure supersmoke preview" if mode == "supersmoke"
                             else "ACTIVE: Appendix G1-G9"),
              "outputs": str(ctx.output_root), "recompute": False}
    (ctx.output_root / "notebook_mode.json").write_text(json.dumps(record, indent=2))
    return ctx


def _root(ctx):
    return Path(ctx.memo["notebook_root"])


def _full(ctx):
    return ctx.memo["notebook_mode"] == "full"


def main_figure(ctx, number):
    if int(number) == 5:
        raise ValueError(
            "The obsolete concentration-versus-stiffness graphic is not an active "
            "paper figure; its numerical records are produced by the benchmark stage."
        )
    if _full(ctx):
        import run_production
        return run_production.main_figure(ctx, number)
    import run_study
    path = run_study.main_figure(ctx, number)
    if int(number) == 8 and ctx.memo["notebook_mode"] != "supersmoke":
        run_study.main_budget_report(ctx)
    return path


def appendix_a(ctx):
    if _full(ctx):
        import run_production
        return run_production.appendix_a(ctx)
    from hat_revision_pipeline import appendix_a_gfe
    return appendix_a_gfe.run(_root(ctx), ctx=ctx)


def appendix_b(ctx):
    if _full(ctx):
        import run_production
        return run_production.appendix_b_and_weights(ctx)
    import run_study
    return run_study.appendix_b(_root(ctx))


def appendix_b3(ctx):
    if _full(ctx):
        import run_production
        return run_production.appendix_b3(ctx)
    # The deposited smoke Appendix B entry already renders B1, B2 and B3.
    return {"status": "Rendered by the smoke Appendix B entry"}


def appendix_c_archived(ctx):
    if _full(ctx):
        import run_production
        return run_production.appendix_c(ctx)
    from hat_revision_pipeline import appendix_c_feg, appendix_c_std_gfe, appendix_c_warm_gfe
    return {"C1": appendix_c_feg.run(_root(ctx)),
            "C2_C6": appendix_c_std_gfe.run(_root(ctx)),
            "C3": appendix_c_warm_gfe.run(_root(ctx), ctx=ctx)}


def appendix_c(ctx, workers=4):
    """Run three sensitivity figures; optimizer robustness is a separate appendix."""
    from hat_revision_pipeline import reviewer_focused
    return reviewer_focused.run_sensitivity(ctx, workers=workers)


def appendix_f(ctx):
    """Run the separate same-objective optimizer comparison."""
    from hat_revision_pipeline import reviewer_focused
    return reviewer_focused.run_optimizer(ctx)


def appendix_c45(ctx):
    if _full(ctx):
        return {"status": "Rendered by the shared full Appendix B/C4/C5 bank"}
    import run_study
    return run_study.appendix_sensitivity(_root(ctx))


def retained_rhg_report(ctx):
    if _full(ctx):
        import run_production
        return run_production.retained_rhg_report(ctx)
    return _root(ctx) / "appendix_outputs/C_GFE/RHg_check/REPORT.md"


def appendix_support(ctx):
    if _full(ctx):
        import run_production
        return run_production.appendix_support(ctx)
    from hat_revision_pipeline import appendix_support_gfe
    return appendix_support_gfe.run(_root(ctx), ctx=ctx, include_historical_a=False)


def appendix_figure_paths(ctx, *identifiers):
    root = _root(ctx)
    bases = [ctx.output_root / "appendices"] if _full(ctx) else [
        root / "appendix_outputs/A_GFE", root / "appendix_B_pressure/figures",
        root / "appendix_outputs/C_GFE", root / "appendix_outputs/D",
        root / "appendix_outputs/E", root / "appendix_outputs/F_optimizer"]
    paths = []
    for identifier in identifiers:
        from production_reporting import is_active_figure
        found = [p for base in bases for p in base.rglob(f"Figure_{identifier}_*.png")
                 if is_active_figure(p, root)]
        if len(found) != 1:
            raise RuntimeError(f"Expected one current Figure {identifier}, found {found}")
        paths.append(found[0])
    return paths


def finalize_archived(ctx):
    if _full(ctx):
        import run_production
        return run_production.finalize(ctx)
    import run_study
    from manuscript_appendix_transfer.prepare_transfer import main as prepare_transfer
    from hat_revision_pipeline.parameter_appendix import render as render_parameters
    prepare_transfer()
    parameters = render_parameters(_root(ctx))
    data = run_study.data_exports(_root(ctx))
    pdf = run_study.review_pdf(_root(ctx))
    return {"mode": "smoke", "pdf": str(pdf), "parameters": parameters,
            "exported_files": len(data), "generality": "ACTIVE: Appendix G1-G9"}


def finalize(ctx):
    """Export the focused, explicitly ordered 22-figure release."""
    from hat_revision_pipeline import reviewer_focused
    return reviewer_focused.finalize(ctx)
