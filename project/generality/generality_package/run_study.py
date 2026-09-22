"""Serial figure orchestration for the GFE manuscript and active appendices."""
from pathlib import Path

from rineng_content_id import content_identity
import json
import os
import sys
import tempfile
from io import BytesIO

ROOT = Path(__file__).resolve().parent
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT / "runtime/src"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from hat_revision_pipeline import pipeline as pipeline


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _install_atomic_figure_exports():
    """Serialize complete figures before replacing an existing export."""
    from matplotlib.figure import Figure
    if getattr(Figure.savefig, "_gfe_atomic", False):
        return
    original = Figure.savefig
    def save(figure, filename, *args, **kwargs):
        from hat_revision_pipeline.final_export_style import remove_review_text
        remove_review_text(figure)
        # Keep fixed locator ticks within the actual color scale for each new setting.
        import numpy as np
        for axis in figure.axes:
            colorbar = getattr(axis, "_colorbar", None)
            if colorbar is None:
                continue
            low, high = float(colorbar.norm.vmin), float(colorbar.norm.vmax)
            ticks = np.asarray(colorbar.get_ticks(), dtype=float)
            inside = ticks[(ticks >= low - 1e-10) & (ticks <= high + 1e-10)]
            if len(inside) != len(ticks):
                if len(inside) < 2:
                    inside = np.linspace(low, high, 4)
                colorbar.set_ticks(inside)
                if colorbar.orientation == "vertical":
                    colorbar.ax.set_ylim((high, low) if colorbar.ax.yaxis_inverted() else (low, high))
                else:
                    colorbar.ax.set_xlim((high, low) if colorbar.ax.xaxis_inverted() else (low, high))
        if not isinstance(filename, (str, os.PathLike)):
            return original(figure, filename, *args, **kwargs)
        path = Path(filename)
        extension = kwargs.get("format", path.suffix.lstrip(".")).lower()
        if extension not in {"pdf", "png", "svg"}:
            return original(figure, filename, *args, **kwargs)
        buffer = BytesIO()
        kwargs["format"] = extension
        original(figure, buffer, *args, **kwargs)
        content = buffer.getvalue()
        if extension == "pdf":
            from pypdf import PdfReader
            assert len(PdfReader(BytesIO(content)).pages) == 1
        _atomic_bytes(path, content)
    save._gfe_atomic = True
    Figure.savefig = save


def prepare(root=ROOT):
    root = Path(root).resolve()
    _install_atomic_figure_exports()
    settings = {
        "HAT_HIDE_MODE_NOTE": "1", "HAT_PROTOTYPE_LABEL": "0",
        "HAT_QUICK_SINGLE_PAIRS": "16", "HAT_QUICK_ALPHA_CAP": "10000",
        "HAT_QUICK_FIG1_XZ_X_POINTS": "5", "HAT_QUICK_FIG1_XZ_Z_POINTS": "3",
        "HAT_QUICK_FIG1_DOMAIN_RESTARTS": "1", "HAT_FIG1_DOMAIN_WORKERS": "3",
        "HAT_BRANCH_ALPHA1000_WORKERS": "3", "HAT_EXACT_WORKERS": "3",
    }
    os.environ.update(settings)
    ctx = pipeline.prepare_pipeline(run_mode="quick", recompute=False,
                                    output_root=root / "smoke_outputs")
    # ``quick`` is the historical internal cache namespace.  In this package
    # it is the accepted paper-scale smoke protocol, not a smaller preview.
    protocol = pipeline._finite_ka_protocol(ctx)
    expected_numerics = {
        "lmax": 3, "fit_shell_wavelengths": (0.10, 0.16, 0.22),
        "fit_n_mu": 6, "fit_n_phi": 12,
        "surface_n_mu": 8, "surface_n_phi": 16,
    }
    observed_numerics = dict(protocol["numerics"])
    observed_numerics["fit_shell_wavelengths"] = tuple(
        observed_numerics["fit_shell_wavelengths"]
    )
    expected_controls = {
        "search_half_width_a": 4.0,
        "max_nfev_per_start": 14,
        "numerical_root_tolerance": 2.5e-3,
        "jacobian_step_a": 0.10,
    }
    if (any(observed_numerics[key] != value
            for key, value in expected_numerics.items()) or
            any(protocol[key] != value for key, value in expected_controls.items()) or
            len(protocol["root_starts_a"]) != 7 or
            ctx.config.conventional_maxiter != 10_000 or
            ctx.config.fe_maxiter != 10_000):
        raise RuntimeError("The smoke context no longer matches the accepted paper-scale protocol")
    ctx.memo["paper_scale_smoke_contract"] = {
        "validator": expected_numerics | expected_controls,
        "root_start_count": 7,
        "single_iteration_cap": 10_000,
        "triple_iteration_cap": 30_000,
    }
    (root / "resolved_execution_settings.json").write_text(json.dumps({
        "evidence_status": "PAPER-SCALE SMOKE", "primary_objective": "GFE",
        "control_objective": "FE", "environment": settings,
        "main_triple_iteration_cap": 30000,
        "main_triple_checkpoints": [10000, 20000, 30000],
        "appendix_C_fixed_protocol_iteration_cap": 10000,
        "appendix_C2_C6_iteration_cap": 30000,
        "default_recompute": False, "generality": "ACTIVE: Appendix G1-G9",
        "run_scope": "Current paper figures and all active appendix figures",
        "paper_scale_smoke_contract": ctx.memo["paper_scale_smoke_contract"],
    }, indent=2))
    return ctx


def main_figure(ctx, number):
    if int(number) == 5:
        raise ValueError(
            "The obsolete concentration-versus-stiffness graphic is disabled; "
            "its numerical benchmark records remain available."
        )
    from hat_revision_pipeline.main_feg import MAIN_FEG_FIRST_FOUR
    import main_feg_benchmarks as benchmarks
    producers = MAIN_FEG_FIRST_FOUR + (benchmarks.figure_5, benchmarks.figure_6,
                                      benchmarks.figure_7, benchmarks.figure_8)
    figure = producers[number - 1](ctx)
    benchmarks.public_tables(ctx)
    pipeline.export_all_tables(ctx)
    stem = list(ctx.figures)[-1]
    path = ctx.output_root / "figures" / (stem + ".png")
    plt.close(figure)
    return path


def appendix_b(root=ROOT):
    """Keep the alternative-sign formulation local to Appendix B and reuse its phase caches."""
    root = Path(root).resolve()
    directory = root / "appendix_B_pressure"
    sys.path.insert(0, str(directory))
    try:
        import reproduce_solutions
        manifest = reproduce_solutions.compute_nominal_missing(directory)
        if len(manifest) != 8:
            raise RuntimeError(f"Appendix B requires eight solutions, got {len(manifest)}")
        import render_b12
        nominal = render_b12.render(directory)
    finally:
        sys.path.remove(str(directory))
    from hat_revision_pipeline import appendix_b3_mechanics
    appendix_b3_mechanics.run(root)
    return {"appendix": "B", "output_dir": str(directory),
            "figures": [str(p) for p in sorted((directory / "figures").glob("Figure_B[123]_*.png"))],
            "nominal_rows": len(nominal)}


def appendix_sensitivity(root=ROOT):
    """Place the formulation-specific weight sweeps in the common sensitivity section."""
    root = Path(root).resolve()
    directory = root / "appendix_B_pressure"
    sys.path.insert(0, str(directory))
    try:
        import sensitivity_b
        tables = sensitivity_b.render(directory, output=root / "appendix_outputs/C_GFE")
    finally:
        sys.path.remove(str(directory))
    return {"figures": [str(path) for path in sorted(
                (root / "appendix_outputs/C_GFE").glob("Figure_C[45]_*.png"))],
            "rows": [len(table) for table in tables]}


def main_budget_report(ctx):
    """Export actual stops and matched-session checkpoint provenance using existing caches."""
    import main_triple30k
    from hat_revision_pipeline.main_triple_endpoint import ensure_main_triple_endpoint
    from hat_revision_pipeline import main_feg, final_figures as ff
    bank = ensure_main_triple_endpoint(ctx)
    targets = bank["objects"][(1, 1)]["targets_m"]
    output = main_triple30k.export(ctx, bank, main_feg.feg_triple_run(ctx),
        ff._triple_prescribed_commands(ctx, targets), ff._triple_ad_command(ctx, targets),
        ff._triple_exact_benchmark_with_concentration(ctx))
    from hat_revision_pipeline.main_results import export_main_results
    export_main_results(ctx)
    pipeline.export_all_tables(ctx)
    return output


def data_exports(root=ROOT):
    """Index portable numerical arrays and tables for every active section."""
    root = Path(root).resolve()
    output = root / "data_exports"
    output.mkdir(exist_ok=True)
    sensitivity = root / "SENSITIVITY_VARIABLES.csv"
    if sensitivity.exists():
        _atomic_bytes(output / sensitivity.name, sensitivity.read_bytes())
    export_measured_summary(root, output)
    folders = {
        "Main": root / "smoke_outputs/tables",
        "A": root / "appendix_outputs/A_GFE",
        "B": root / "appendix_B_pressure",
        "C": root / "appendix_outputs/C_GFE",
        "D": root / "appendix_outputs/D",
        "E": root / "appendix_outputs/E",
        "Parameters": root / "appendix_outputs/P_parameters",
        "Manuscript transfer": root / "manuscript_appendix_transfer",
        "Summary": output,
    }
    records = []
    for section, directory in folders.items():
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in {".csv", ".tsv", ".npz", ".json", ".txt", ".tex"}:
                continue
            if path.name.startswith("exportable_data_index"):
                continue
            if any(x in path.parts for x in ("__pycache__", "runtime", "tools", "qa", "tmp")):
                continue
            assigned_section = ("C" if section == "B" and
                ("sensitivity" in path.name.lower()
                 or "sensitivity_cache" in path.parts) else section)
            records.append({"section": assigned_section, "path": str(path.relative_to(root)),
                            "format": path.suffix[1:], "bytes": path.stat().st_size,
                            "content_id": content_identity(path.read_bytes()).hexdigest()})
    frame = pd.DataFrame(records)
    frame.to_csv(output / "exportable_data_index.csv", index=False)
    frame.to_json(output / "exportable_data_index.json", orient="records", indent=2)
    (output / "README.md").write_text(
        "# Exportable numerical data\n\n"
        "The index links every active section to portable CSV/TSV tables, NPZ arrays, "
        "resolved parameters, captions, and measured results. NPZ files are opened with "
        "numpy.load(path, allow_pickle=False). Solver replay caches are additionally "
        "retained in their original cache folders.\n\n"
        "B contains Twin/Bottle formulation references. C groups all sensitivity "
        "studies: frozen nominal GFE correspondence, compensated Triple ablations, "
        "initialization, isolated STD constants, and Twin/Bottle formulation-specific weights. "
        "B3 supplies independent elastic mechanics for Vortex, Twin and Bottle commands. Legacy B3/B4 "
        "cache filenames are retained for reproducibility. "
        "D distinguishes the historical FE discretization study from current GFE "
        "discretization. E retains its historical FE morphology reference. "
        "Array/frequency generality is integrated as Appendix G1-G9 by unified_study.py.\n")
    return frame


def export_measured_summary(root, output):
    """Produce readable values from current tables, with no legacy relabeling."""
    table = pd.read_csv(root / "smoke_outputs/tables/fig8_performance_time_tradeoff.csv")
    table["method"] = table["method"].replace({"FE+g": "GFE"})
    columns = ["method", "single_time_s", "single_displacement_a",
               "single_stiffness_min_n_m", "single_pressure_abs_at_exact_equilibrium_pa",
               "triple_time_s", "triple_displacement_a", "triple_displacement_min_a",
               "triple_displacement_max_a", "triple_stiffness_min_n_m",
               "triple_pressure_abs_at_exact_equilibrium_pa"]
    compact = table[columns].copy()
    compact.to_csv(output / "main_benchmark_values.csv", index=False)
    compact.to_csv(output / "main_benchmark_values.tsv", index=False, sep="\t")
    (output / "main_benchmark_values.txt").write_text(compact.to_string(index=False) + "\n")
    # Keep the portable LaTeX export independent of pandas' optional Styler/Jinja2 stack.
    def latex_value(value):
        if isinstance(value, (float, int)):
            return "--" if pd.isna(value) else f"{value:.5g}"
        return str(value).replace("_", r"\_").replace("%", r"\%")
    latex_rows = [" & ".join(latex_value(value) for value in row) + r" \\"
                  for row in compact.itertuples(index=False, name=None)]
    latex = [r"\begin{tabular}{l" + "r" * (len(columns) - 1) + "}", r"\hline",
             " & ".join(latex_value(name) for name in columns) + r" \\", r"\hline",
             *latex_rows, r"\hline", r"\end{tabular}"]
    (output / "main_benchmark_values.tex").write_text("\n".join(latex) + "\n")
    by_method = table.set_index("method")
    lines = [
        "GFE STUDY - MEASURED RESULTS (SMOKE)",
        "GFE is the gravity-compensated objective; FE is the uncompensated control. "
        "Archived cache names are retained for provenance only.",
    ]
    for task in ("single", "triple"):
        g, fe, conventional = (by_method.loc[m] for m in ("GFE", "FE", "Conventional"))
        lines.append(f"{task.title()}: elastic-equilibrium displacement d/a is "
                     f"{g[task+'_displacement_a']:.6g} for GFE, "
                     f"{fe[task+'_displacement_a']:.6g} for FE, and "
                     f"{conventional[task+'_displacement_a']:.6g} for Conventional.")
    lines += [
        "Triple values above are medians over the three targets of one command. "
        "They are not independent replicate estimates.",
        "Command times retain their original measured scope. Full-notebook replay "
        "duration is not used as a synthesis benchmark.",
    ]
    for section, directory in (("A", root / "appendix_outputs/A_GFE"),
                               ("B", root / "appendix_B_pressure"),
                               ("C", root / "appendix_outputs/C_GFE"),
                               ("D", root / "appendix_outputs/D"),
                               ("E", root / "appendix_outputs/E")):
        sources = sorted(directory.glob("*measured*.txt"))
        if section == "B":
            sources = [directory / "appendix_text.txt", directory / "data/B3/B3_measured_results.txt"]
        if section == "C":
            sources.append(directory / "std_sensitivity/measured_results_smoke.txt")
            sources.append(directory / "RHg_check/REPORT.md")
        for path in sources:
            if path.is_file():
                lines.append(f"APPENDIX {section} - {path.name}\n{path.read_text()}")
    (output / "all_measured_results.txt").write_text("\n\n".join(lines) + "\n")


def figure_paths(root=ROOT):
    root = Path(root).resolve()
    def numbered(path):
        import re
        return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.name)]
    main = sorted((root / "smoke_outputs/figures").glob("Figure_*.pdf"), key=numbered)
    if len(main) != 8:
        raise ValueError(f"Expected 8 main figures, found {len(main)}")
    paths = [("Main", p) for p in main]
    for section, directory in (
        ("A", root / "appendix_outputs/A_GFE"),
        ("B", root / "appendix_B_pressure/figures"),
        ("C", root / "appendix_outputs/C_GFE"),
        ("D", root / "appendix_outputs/D"),
        ("E", root / "appendix_outputs/E"),
    ):
        figures = sorted((p for p in directory.rglob("*.pdf")
                          if p.with_suffix(".png").is_file()
                          and (section != "B" or p.name.startswith(("Figure_B1_", "Figure_B2_alternative_", "Figure_B3_")))
                          and not any(s in p.parts for s in ("cache", "qa", "tmp", "sources"))), key=numbered)
        if not figures:
            raise ValueError(f"No figure for Appendix {section}")
        paths += [(section, p) for p in figures]
    return paths


def review_pdf(root=ROOT):
    """Preserve vector plots and append the detailed reproducibility tables."""
    from pypdf import PdfReader, PdfWriter
    root = Path(root).resolve()
    writer = PdfWriter()
    index = []
    paths = figure_paths(root)
    parameter_pdfs = sorted((root / "appendix_outputs/P_parameters").glob("*.pdf"))
    paths += [("Parameters", p) for p in parameter_pdfs]
    for section, path in paths:
        reader = PdfReader(path)
        first = len(writer.pages)
        writer.append(reader, import_outline=False)
        writer.add_outline_item(f"{section}: {path.stem.replace('_', ' ')}", first)
        index.append({"section": section, "source": str(path.relative_to(root)),
                      "first_page": first + 1, "pages": len(reader.pages)})
    writer.add_metadata({"/Title": "GFE: Eight main figures and active appendices",
                         "/Subject": "Figures, sensitivity, ablations and reproducibility parameters; SMOKE"})
    output = root / "RINENG_GFE_all_figures_and_parameters.pdf"
    buffer = BytesIO()
    writer.write(buffer)
    content = buffer.getvalue()
    assert len(PdfReader(BytesIO(content)).pages) == sum(row["pages"] for row in index)
    _atomic_bytes(output, content)
    (root / "figure_review_index.json").write_text(json.dumps(index, indent=2))
    return output


def run(root=ROOT):
    root = Path(root).resolve()
    ctx = prepare(root)
    for number in (1, 2, 3, 4, 6, 7, 8):
        print(main_figure(ctx, number), flush=True)
    main_budget_report(ctx)
    from hat_revision_pipeline import appendix_a_gfe
    appendix_a_gfe.run(root, ctx=ctx)
    from hat_revision_pipeline import appendix_support_gfe
    appendix_support_gfe.run(root, ctx=ctx, include_historical_a=False)
    appendix_b(root)
    from hat_revision_pipeline.appendix_c_feg import run as run_c
    run_c(root)
    from hat_revision_pipeline.appendix_c_std_gfe import run as run_std
    run_std(root)
    from hat_revision_pipeline.appendix_c_warm_gfe import run as run_warm
    run_warm(root, ctx=ctx)
    appendix_sensitivity(root)
    from hat_revision_pipeline.appendix_rhg_check import run as run_rhg
    run_rhg(root, ctx=ctx)
    from manuscript_appendix_transfer.prepare_transfer import main as prepare_manuscript_transfer
    prepare_manuscript_transfer()
    from hat_revision_pipeline.parameter_appendix import render as parameter_tables
    parameter_tables(root)
    data_exports(root)
    print(review_pdf(root), flush=True)


if __name__ == "__main__":
    run()
