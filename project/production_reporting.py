"""Export the completed production campaign without mixing archived smoke evidence."""
from __future__ import annotations

from dataclasses import asdict

from rineng_content_id import content_identity
from importlib import metadata
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import sys
import xml.sax.saxutils as xml


PAPER_MAIN_STEMS = (
    ("1", "Paper_Figure_1_target_grid"),
    ("2", "Figure_1_single_vortex_convergence"),
    ("3", "Figure_2_two_branch_evolution"),
    ("4", "Figure_3_local_decision_geometry"),
    ("5", "Figure_4_triple_fields_and_decision"),
    ("6", "Figure_6_force_field_comparison"),
    ("7", "Figure_7_alpha_transition"),
    ("8", "Figure_8_performance_time_tradeoff"),
)
APPENDIX_IDS = tuple(
    f"{section}{index}"
    for section, count in (("A", 2), ("B", 3), ("C", 3), ("D", 3), ("E", 2), ("F", 1))
    for index in range(1, count + 1)
)
EXPECTED_IDS = tuple(identifier for identifier, _ in PAPER_MAIN_STEMS) + APPENDIX_IDS
OBSOLETE_MAIN_STEM = "Figure_5_single_triple_exact_force_concentration"


FOCUSED_C_STEMS = {
    "C1": "Figure_C1_objective_weight_sensitivity",
    "C2": "Figure_C2_multitrap_regularization",
    "C3": "Figure_C3_RH_beta_choice",
}


OPTIMIZER_F_STEMS = {"F1": "Figure_F1_optimizer_robustness"}

def is_active_figure(path, base):
    """Whitelist current identifiers; legacy C images never enter the report."""
    path, base = Path(path), Path(base)
    parts = path.relative_to(base).parts
    if any(part in {"cache", "sources", "historical", "qa", "RHg_check_archived",
                    "archive", "archived", "reviewer_robustness"} for part in parts):
        return False
    if path.stem == OBSOLETE_MAIN_STEM:
        return False
    if path.stem in {stem for _, stem in PAPER_MAIN_STEMS}:
        return path.parent.name == "figures"
    match = re.match(r"Figure_([A-F]\d+)(?:_|\.)", path.name)
    if not match or match.group(1) not in APPENDIX_IDS:
        return False
    key = match.group(1)
    if key.startswith("C"):
        return ("reviewer_focused" in parts and path.parent.name == "figures"
                and path.stem == FOCUSED_C_STEMS[key])
    if key.startswith("F"):
        return ("F_optimizer" in parts and path.parent.name == "figures"
                and path.stem == OPTIMIZER_F_STEMS[key])
    return True


def content_id(path):
    h = content_identity()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def figure_inventory(output):
    """Require the current paper sequence and active appendix figures."""
    from pypdf import PdfReader
    output = Path(output)
    candidates = {}
    for identifier, stem in PAPER_MAIN_STEMS:
        path = output / "figures" / f"{stem}.pdf"
        if not path.is_file():
            raise RuntimeError(f"Missing current paper Figure {identifier}: {path}")
        candidates[identifier] = path
    for path in sorted(output.rglob("Figure_[A-F]*.pdf")):
        if not is_active_figure(path, output):
            continue
        match = re.match(r"Figure_([A-F]\d+)(?:_|\.)", path.name)
        if not match or match.group(1) not in APPENDIX_IDS:
            continue
        key = match.group(1)
        if key in candidates:
            raise RuntimeError(f"Duplicate active Figure {key}: {candidates[key]} and {path}")
        candidates[key] = path
    missing = set(EXPECTED_IDS) - set(candidates)
    if missing:
        raise RuntimeError(f"Missing active production figures: {sorted(missing)}")
    for key, path in candidates.items():
        if not path.with_suffix(".png").is_file():
            raise RuntimeError(f"Missing PNG companion for Figure {key}: {path}")
        pdf = PdfReader(path)
        if len(pdf.pages) != 1:
            raise RuntimeError(f"Expected one figure per PDF: {path}")
        text = pdf.pages[0].extract_text() or ""
        if "PROTOTYPE" in text or "SMOKE" in text:
            raise RuntimeError(f"Temporary figure text remains in {path}")
    result=[{"figure": key, "section": "Main" if key.isdigit() else key[0],
             "pdf": str(candidates[key].relative_to(output)),
             "png": str(candidates[key].with_suffix(".png").relative_to(output))}
            for key in EXPECTED_IDS]
    return result


def export_data(output):
    import numpy as np
    import pandas as pd
    output = Path(output)
    destination = output / "data_exports"
    destination.mkdir(exist_ok=True)
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.suffix not in {".csv", ".tsv", ".npz", ".json", ".txt", ".md", ".tex"}:
            continue
        relative = path.relative_to(output)
        if path.name in {"execution_progress.json", "production_release_verification.json", "figure_review_index.json"}:
            continue
        if relative.parts[0] == "data_exports" or any(p in {"cache", "qa", "__pycache__"} for p in relative.parts):
            continue
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                for key in arrays.files:
                    arrays[key]
        rows.append({"path": str(relative), "format": path.suffix[1:],
                     "bytes": path.stat().st_size, "content_id": content_id(path),
                     "evidence": "retained prior check" if "RHg_check_archived" in relative.parts else "production output; consult source protocol"})
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "exportable_data_index.csv", index=False)
    (destination / "exportable_data_index.json").write_text(frame.to_json(orient="records", indent=2))
    (destination / "README.md").write_text(
        "# Production data\n\nCSV/TSV tables and non-pickled NPZ arrays accompany all figures. "
        "The content_id index identifies the exact exported files. Full numerical replay caches remain "
        "under the production cache folders. Command seeds are the replication unit; Triple targets "
        "are nested in each command and Appendix A retains nine target clusters. "
        "RHg_check_archived is the retained earlier one-pair check, with its original numerical protocol.\n")
    return frame


def parameters(root, ctx):
    """Read active settings and source declarations; do not infer historical timings."""
    import pandas as pd
    import jax
    from threadpoolctl import threadpool_info
    from hat_revision_pipeline import parameter_appendix as old
    from hat_revision_pipeline import pipeline
    root = Path(root)
    out = ctx.output_root / "parameters"
    out.mkdir(exist_ok=True)
    rows = []
    def add(group, key, value, units="1", source="production_config.json", note="Recorded production control"):
        rows.append({"group": group, "parameter": key, "value": old._fmt(value), "units": units,
                     "applicability": note, "source_file": source, "source_function": "", "source_line": "", "note": ""})
    for key, value in ctx.memo["production_settings"].items():
        add("Production design", key, value, old._unit(key))
    for key, value in asdict(ctx.config).items():
        add("Resolved runtime configuration", key, value, old._unit(key), "production_main.prepare", "Resolved context; individual producer overrides are recorded with each command")
    for key, value in pipeline._finite_ka_protocol(ctx).items():
        add("Independent elastic mechanics", key, value, old._unit(key), "pipeline._finite_ka_protocol")
    groups = {"Physical model", "Array and objective", "Triple objective", "Main Triple objective",
              "Main Triple Conventional", "Publication layout", "Phase resolution and deployment",
              "Current reviewer optimizer controls", "Expanded sensitivity design", "Conventional notation"}
    for row in old._readable_parameters(root):
        if row["group"] in groups:
            rows.append(row)
    add("Statistics", "paired bootstrap", "10000 resamples; seed 20260906; command/seed-level paired median differences; 2.5/97.5 percentiles", "95% interval")
    add("Statistics", "sampling unit", "Command/seed; Triple targets nested; Appendix A target clusters retained; IB/GS deterministic endpoints have no endpoint-population CI, while their 30 timing repetitions remain valid timing observations")
    add("Statistics", "terminal gradient summaries", "Median, IQR and range within each method's declared objective; objective-specific gradient norms are not pooled across formulations; unavailable for non-gradient IB/GS prescriptions")
    add("Timing", "Figure 8 protocol", "main-fig8-comparable-end-to-end-v2", "status",
        "production_main.py", "Only complete common-boundary observations enter Figure 8")
    add("Timing", "common boundary", "Problem/objective initialization through terminal host phase; acoustic transfer, applicable device transfer, compilation/warmup, synchronization, native solve and result extraction included", "s",
        "MAIN_FIG8_TIMING_PROTOCOL.md", "Independent mechanics, cache lookup, export and rendering excluded")
    add("Timing", "sampling", "30 complete observations per method/task; IB/GS retain one deterministic endpoint with 30 timing repetitions", "observations",
        "production_timing_observations.csv", "Missing stages are N/A, never zero; incomplete observations are excluded")
    add("Scope", "Generality", "ACTIVE: Appendix G1-G9", "status")
    add("Scope", "Original manuscript optimizer/stencil studies", "Active C1-C3 contain objective, regularization and RH-beta sensitivity. Appendix F1 separately tests optimizer robustness. Wider source modules and archived tables do not imply additional default experiments or figures.", "status")
    add("Scope", "RH+g", "Retained one-pair C5 check; no expansion", "status")
    parameter_frame = old._write(out, "P_parameters", rows)
    numeric, sources = old._source_inventory(root)
    old._write(out, "P_source_numeric_inventory", numeric)
    old._write(out, "P_source_hashes", sources)
    seeds, inventory, protocols = [], [], []
    for path in sorted(ctx.output_root.rglob("*.csv")):
        if out in path.parents or any(p in {"cache", "data_exports"} for p in path.relative_to(ctx.output_root).parts):
            continue
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(ctx.output_root))
        inventory.append({"source_file": rel, "rows": len(frame), "columns": json.dumps(list(frame.columns)), "content_id": content_id(path)})
        for col in frame.columns:
            if old._is_seed_identity(col):
                values = pd.to_numeric(frame[col], errors="coerce").dropna()
                for value, count in values.value_counts().sort_index().items():
                    if float(value).is_integer():
                        seeds.append({"source_file": rel, "seed_column": col, "seed": int(value), "record_count": int(count)})
    for path in sorted(ctx.output_root.rglob("*.json")):
        if out in path.parents or path.stat().st_size > 10_000_000:
            continue
        if any(token in path.name.lower() for token in ("protocol", "setting", "manifest", "environment", "verification")):
            protocols.append({"source_file": str(path.relative_to(ctx.output_root)), "content_id": content_id(path), "record_json": path.read_text()})
    seed_frame = old._write(out, "P_actual_seeds", seeds)
    old._write(out, "P_exported_tables", inventory)
    old._write(out, "P_runtime_protocols", protocols)
    software = []
    for name in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "scikit-image", "jax", "jaxlib", "IPython", "nbformat", "nbclient", "ipykernel", "pypdf", "reportlab", "Pillow", "threadpoolctl"):
        software.append({"package": name, "version": metadata.version(name)})
    software.extend([{"package": "Python", "version": platform.python_version()}, {"package": "platform", "version": platform.platform()}])
    old._write(out, "P_software_versions", software)
    hardware = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(), "executable": sys.executable,
                "jax_devices": [str(d) for d in jax.devices()], "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "threadpools": threadpool_info(), "thread_environment": {k: os.environ.get(k) for k in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "JAX_PLATFORMS", "JAX_ENABLE_X64")}}
    (out / "P_current_hardware_and_threads.json").write_text(json.dumps(hardware, indent=2, default=str))
    # Preserve every editor/reviewer request verbatim. Earlier coverage is explicitly archival.
    crosswalk = pd.DataFrame(old._editor_crosswalk(root))
    crosswalk = crosswalk.rename(columns={"status": "archived_coverage_status", "response_scope": "archived_response_scope", "evidence_paths_or_patterns": "archived_evidence_paths"})
    crosswalk["production_evidence"] = "Active data index, runtime protocol inventory and figure-specific production manifests; manuscript prose remains separate"
    old._write(out, "P_editor_reviewer_crosswalk", crosswalk)
    pdf = parameter_pdf(out, parameter_frame, pd.DataFrame(software), hardware, crosswalk)
    return {"output": str(out), "pdf": str(pdf), "parameters": len(parameter_frame), "actual_seed_records": len(seed_frame), "source_numeric_rows": len(numeric)}


def parameter_pdf(out, parameters, software, hardware, crosswalk):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, LongTable, TableStyle, Spacer, PageBreak, CondPageBreak
    from matplotlib.font_manager import findfont, FontProperties
    for name, weight in (("GFEReport", "normal"), ("GFEReportBold", "bold")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, findfont(FontProperties(family="DejaVu Sans", weight=weight))))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "GFEReportBold" if style.name in {"Title", "Heading1", "Heading2"} else "GFEReport"
    styles.add(ParagraphStyle(name="GFECell", fontName="GFEReport", fontSize=8.5, leading=11, splitLongWords=True))
    styles["Title"].fontSize = 19
    styles["Title"].leading = 23
    def para(value, style="GFECell"):
        return Paragraph(xml.escape(str(value)).replace("\n", "<br/>"), styles[style])
    def table(data, widths):
        result = LongTable([[para(v) for v in row] for row in data], colWidths=[x * mm for x in widths], repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf4")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return result
    story = [para("Production parameters and reproducibility", "Title"),
             para("Resolved production controls and observed software/device settings. Full source numeric declarations, per-command protocols, actual seed identities and original editor requests accompany these tables. Historical command selection and the retained RH+g check keep their recorded protocols.", "BodyText"), Spacer(1, 5 * mm)]
    for group, frame in parameters.groupby("group", sort=False):
        story += [CondPageBreak(50 * mm), para(group, "Heading1")]
        data = [["Parameter / units", "Value and scope", "Source / interpretation"]]
        for row in frame.itertuples(index=False):
            data.append([f"{row.parameter}\n[{row.units}]", f"{row.value}\n{row.applicability}", f"{row.source_file}\n{row.source_function}:{row.source_line}\n{row.note}"])
        story.append(table(data, [62, 111, 98]))
    story += [PageBreak(), para("Software and devices", "Heading1"), table([list(software.columns), *software.values.tolist()], [71, 200])]
    story += [Spacer(1, 4 * mm), table([["Observed setting", "Value"], *[[k, json.dumps(v, default=str)] for k, v in hardware.items()]], [71, 200])]
    story += [PageBreak(), para("Editor and reviewer request mapping", "Heading1"),
              para("The requested changes below are retained verbatim. Earlier coverage is identified as archival; current numerical evidence is indexed separately. Manuscript-only requests are not marked complete by successful code execution.", "BodyText")]
    data = [["Request", "Exact requested change", "Earlier coverage and current evidence"]]
    for row in crosswalk.to_dict("records"):
        data.append([row["request_id"], row["exact_request"], "\n".join(str(row.get(k, "")) for k in ("archived_coverage_status", "archived_response_scope", "production_evidence"))])
    story.append(table(data, [18, 115, 138]))
    width, height = landscape(A4)
    def footer(canvas, doc):
        canvas.setFont("GFEReport", 8)
        canvas.drawString(13 * mm, 9 * mm, "RINENG | Production parameters")
        canvas.drawRightString(width - 13 * mm, 9 * mm, str(doc.page))
    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=(width, height), leftMargin=13 * mm, rightMargin=13 * mm,
                      topMargin=13 * mm, bottomMargin=17 * mm).build(story, onFirstPage=footer, onLaterPages=footer)
    path = Path(out) / "Parameter_and_reproducibility_appendix.pdf"
    path.write_bytes(buffer.getvalue())
    return path


def finalize(root, ctx):
    from pypdf import PdfReader, PdfWriter
    output = ctx.output_root
    figures = figure_inventory(output)
    parameter_info = parameters(root, ctx)
    exports = export_data(output)
    writer = PdfWriter()
    index = []
    for row in figures:
        path = output / row["pdf"]
        index.append(dict(row, first_page=len(writer.pages) + 1, pages=1))
        writer.append(PdfReader(path), import_outline=False)
        writer.add_outline_item(f"Figure {row['figure']}", len(writer.pages) - 1)
    path = Path(parameter_info["pdf"])
    first = len(writer.pages)
    reader = PdfReader(path)
    writer.append(reader, import_outline=False)
    writer.add_outline_item("Parameters and reproducibility", first)
    index.append({"section": "Parameters", "pdf": str(path.relative_to(output)), "first_page": first + 1, "pages": len(reader.pages)})
    writer.add_metadata({"/Title": "GFE production: main figures and active appendices"})
    pdf_path = output / "RINENG_GFE_PRODUCTION_all_figures_and_parameters.pdf"
    buffer = BytesIO()
    writer.write(buffer)
    pdf_path.write_bytes(buffer.getvalue())
    (output / "figure_review_index.json").write_text(json.dumps(index, indent=2))
    result = {"status": "completed", "figures": len(figures), "main_figures": 8, "appendix_figures": len(figures)-8,
              "pdf_pages": len(writer.pages), "exported_files": len(exports), "parameters": parameter_info,
              "pdf": str(pdf_path), "generality": "ACTIVE: Appendix G1-G9", "source_numeric_declarations_are_execution_evidence": False}
    (output / "production_release_verification.json").write_text(json.dumps(result, indent=2))
    return result
