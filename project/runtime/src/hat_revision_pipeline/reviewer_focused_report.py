"""Compact, data-derived interpretation of Appendix C sensitivity and Appendix F optimizer evidence.

This module only reports recorded tables. It does not run an optimizer or infer
completion of the standalone force/Hessian cell from the selected notebook mode.
"""
from pathlib import Path
from io import BytesIO
import json
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


def _truth(series):
    return series.map(lambda value: str(value).lower() in {"true", "1", "1.0"})


def _num(value, digits=4):
    return "unresolved" if pd.isna(value) else f"{float(value):.{digits}g}"


def _values(frame, column="parameter_value"):
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    return ", ".join(f"{v:g}" for v in sorted(values))


def _seeds(frame):
    return int(frame.seed.nunique()) if len(frame) else 0


def _fraction(frame, column):
    return f"{int(_truth(frame[column]).sum())}/{len(frame)}"


def write_report(ctx, destination=None):
    """Write a five-page, cache-derived results guide; return its PDF path."""
    from . import reviewer_focused as focused
    from matplotlib.font_manager import findfont, FontProperties
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    )

    data = focused._read_inputs(ctx)
    continuous, std, beta = data["continuous"], data["std"], data["beta"]
    optimizer = data["optimizer"]
    optimizer = optimizer[optimizer.optimizer.isin(("BFGS", "L-BFGS-B", "Adam"))].copy()
    ablation = std[_truth(std.nominal) | std.sweep.eq("ablation")].copy()
    root = Path(ctx.memo["notebook_root"])
    out = focused.output(ctx)
    full = bool(ctx.config.full)
    mode = "FULL" if full else "SMOKE"
    path = Path(destination) if destination else root / "RINENG_GFE_focused_results_guide.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)

    for family, weight in (("FocusedReport", "normal"), ("FocusedReportBold", "bold")):
        if family not in pdfmetrics.getRegisteredFontNames():
            font = findfont(FontProperties(family="DejaVu Sans", weight=weight))
            pdfmetrics.registerFont(TTFont(family, font))
    pdfmetrics.registerFontFamily("FocusedReport", normal="FocusedReport", bold="FocusedReportBold")
    ink, teal = colors.HexColor("#233941"), colors.HexColor("#087F72")
    body = ParagraphStyle("FocusedBody", fontName="FocusedReport", fontSize=9.1,
                          leading=12.5, textColor=ink, spaceAfter=7)
    small = ParagraphStyle("FocusedSmall", parent=body, fontSize=8.0, leading=10.4, spaceAfter=0)
    cell = ParagraphStyle("FocusedCell", parent=body, fontSize=7.75, leading=10.0, spaceAfter=0)
    cell_head = ParagraphStyle("FocusedCellHead", parent=cell, fontName="FocusedReportBold")
    title = ParagraphStyle("FocusedTitle", parent=body, fontName="FocusedReportBold",
                           fontSize=21, leading=26, spaceAfter=13)
    heading = ParagraphStyle("FocusedHeading", parent=body, fontName="FocusedReportBold",
                             fontSize=15, leading=19, spaceAfter=10)
    sub = ParagraphStyle("FocusedSub", parent=body, fontName="FocusedReportBold",
                         fontSize=10.5, leading=14, spaceBefore=9, spaceAfter=5)
    formula = ParagraphStyle("FocusedFormula", parent=body, fontSize=10.1, leading=15,
                             leftIndent=10, borderPadding=8, backColor=colors.HexColor("#EFF6F5"),
                             spaceBefore=4, spaceAfter=10)
    story = []
    width = A4[0] - 92

    def p(text, style=body):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)

    def rich(text, style=body):
        return Paragraph(text, style)

    def table(headers, rows, widths, font=cell):
        items = [[p(v, cell_head) for v in headers]]
        items += [[p(v, font) for v in row] for row in rows]
        t = Table(items, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9F1F1")),
            ("LINEBELOW", (0, 0), (-1, 0), .8, teal),
            ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#D6E0E2")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    def sweep(name, task=None, method=None):
        q = continuous[continuous.parameter.eq(name)]
        if task is not None:
            q = q[q.task.eq(task)]
        if method is not None:
            q = q[q.method.eq(method)]
        return q

    def subset(frame, value, column="parameter_value"):
        return frame[np.isclose(pd.to_numeric(frame[column], errors="coerce"), value)]

    def med(frame, metric):
        values = frame[metric].dropna() if len(frame) else pd.Series(dtype=float)
        return values.median() if len(values) else np.nan

    def evidence_line():
        return (f"Optimizer: {_seeds(optimizer)} paired Single seeds; component ablations: "
                f"{_seeds(ablation)} paired Triple seeds; continuous sweeps: "
                + ", ".join(f"{k} {v}" for k, v in continuous.groupby("task").seed.nunique().items())
                + f" seed(s); RH beta: {_seeds(beta)} seed(s).")

    # Page 1: explicit publication sequence and scope of this reporting delivery.
    story += [p("GFE: focused results guide", title),
              p("Appendix C: sensitivity. Appendix F: optimizer robustness.", sub)]
    if full:
        intro = ("This guide summarizes the recorded full-mode tables available to the notebook. "
                 "Its counts and statistics are read from those tables. The scientific PDF follows "
                 "the explicit 22-figure manifest; exploratory atlases are excluded. Full Run All "
                 "and standalone-equilibrium completion must be read from their execution records, "
                 "not inferred from the selected mode.")
    else:
        intro = ("This delivery presents the 22 accepted figures in notebook order, with SI "
                 "displacement axes and a final visual cleanup. It is a "
                 "reporting revision using measured smoke evidence; a complete notebook Run All "
                 "was not executed for this delivery. The numerical commands, native budgets and "
                 "fixed references were preserved.")
    story += [p(intro), p(evidence_line()),
              table(["PDF pages", "Figures", "Role in the argument"], [
                  ["1-8", "Main 1-8", "Retain the accepted main narrative and current Figure 2 presentation."],
                  ["9-10", "A1-A2", "Surrogate force direction/magnitude and independent elastic mechanics."],
                  ["11-13", "B1-B3", "Formulation-dependent pressure morphology and exact mechanics of representative traps."],
                  ["14", "C1", "Which objective weights change the field or mechanical precision?"],
                  ["15", "C2", "What do the multitrap components and smoothing constants contribute?"],
                  ["16", "C3", "How does RH beta change the selected field and iteration cost?"],
                  ["17-19", "D1-D3", "Discrete phases, mechanics-aware offsets, and equilibrium shifts."],
                  ["20-21", "E1-E2", "Prescription radius and reference-field reproduction."],
                  ["22", "F1", "How does compact L-BFGS-B FE/GFE compare with Conventional solver controls?"],
              ], [64, 68, width-132]), Spacer(1, 11),
              p("What has been removed from the default figure sequence", sub),
              p("The large-alpha initialization and jump/restart studies, individual solver-knob "
                "pages and duplicate directional/STD plots remain archived as data. Named objective, "
                "curvature and numerical controls are summarized in tables, so removing repetitive "
                "figures does not remove the measured sensitivity evidence."),
              p("Editorial focus", sub),
              p("E04 asks for parameter values, units and sensitivity; E08 asks for multitrap "
                "component ablations; R2.3 asks whether the RH pressure weight controls convergence "
                "and branch selection. Appendix F reports compact L-BFGS-B FE/GFE and Conventional solver controls. This guide "
                "does not declare every manuscript/editorial task complete.")]

    # Page 2: definitions and the directly interpretable regularization evidence.
    story += [PageBreak(), p("C1-C2: physical effects and regularization", heading),
              p("Read field similarity and displacement together", sub),
              rich("C<sub>p</sub> = |p<sub>ref</sub><super>H</super>p| / "
                   "(‖p<sub>ref</sub>‖<sub>2</sub> ‖p‖<sub>2</sub>), "
                   "&nbsp; d = max<sub>j</sub> ‖r<sub>eq,j</sub> − r<sub>target,j</sub>‖ (m)", formula),
              p("C1 and C2 use complex pressure samples on the same local 9 × 9 × 9 volumes. "
                "The reference is the frozen, same-seed nominal GFE command. Cp is dimensionless, "
                "invariant to one global phase, and not a physical stability criterion. The separate "
                "elastic-force model determines equilibrium displacement in metres. Cached d/a "
                "values are multiplied by the declared bead radius a = 6.5 × 10^-4 m; the "
                "original normalized columns remain in the exported data."),
              p("C1 varies alpha for FE/GFE, lambda_p for Conventional and beta for Triple GFE. "
                "Each coefficient is divided by its own nominal value. High correspondence can "
                "coexist with meaningful displacement changes. Crosses keep native-stopping failures "
                "visible; the curves do not retune the main-paper parameters."),
              rich("J = mean<sub>j</sub>(ℓ<sub>j</sub>) + γ<sub>u</sub> "
                   "[√(Var<sub>pop</sub>(ℓ<sub>j</sub>) + ε<sub>STD</sub><super>2</super>) − ε<sub>STD</sub>]", formula),
              p("The local losses ℓj and epsilon_STD have units N m^-1; gamma_u is dimensionless. "
                "The regularizer balances local objective losses, not pressure, stiffness or displacement "
                "directly. Every ablated endpoint is reevaluated with the same nominal local-loss "
                "definition when reporting STD.")]
    rows = []
    for name in ("GFE", "No force smoothing", "No pressure", "No uniformity", "FE"):
        q = ablation[ablation.ablation.eq(name)]
        rows.append([name, _fraction(q, "native_success"), _num(med(q, "field_cosine_to_fixed_GFE"), 5),
                     _num(med(q, "worst_displacement_m"), 5),
                     _num(med(q, "common_nominal_loss_std_n_m"), 4)])
    story += [table(["Ablation", "Native stop", "Median Cp", "Median d (m)", "Common STD\n(N m^-1)"],
                    rows, [131, 73, 85, 82, width-371]), Spacer(1, 8)]
    qg = ablation[ablation.ablation.eq("GFE")]
    qn = ablation[ablation.ablation.eq("No uniformity")]
    qp = ablation[ablation.ablation.eq("No pressure")]
    ratios = [med(q, "common_nominal_loss_std_n_m") / med(qg, "common_nominal_loss_std_n_m")
              for q in (qn, qp)]
    story.append(p(f"Compared with complete GFE, removing the STD and pressure penalties raises "
                   f"the median common-loss STD by {_num(ratios[0], 3)}× and {_num(ratios[1], 3)}×, "
                   "respectively. These are loss-balancing effects, reported alongside each method's "
                   "independent equilibrium error."))
    ep0 = subset(sweep("epsilon_pressure_rel", "Triple"), 0)
    es0 = subset(sweep("epsilon_std_factor", "Triple"), 0)
    ep1 = subset(sweep("epsilon_pressure_rel", "Triple"), .001)
    story.append(p(f"C2(d) separates optimization from morphology: at zero pressure smoothing and "
                   f"zero STD smoothing, native stops are {_fraction(ep0, 'raw_gtol_met')} and "
                   f"{_fraction(es0, 'raw_gtol_met')}, with median terminal iterations "
                   f"{_num(med(ep0, 'iterations'), 6)} and {_num(med(es0, 'iterations'), 6)}. "
                   f"Nominal smoothing uses {_num(med(ep1, 'iterations'), 6)} iterations. "
                   "Zero epsilon_STD retains an unsmoothed penalty; deleting the penalty sets its "
                   "weight to zero. Their effects must not be conflated."))

    # Page 3: direct reviewer-beta answer and the small optimizer control.
    story += [PageBreak(), p("RH sensitivity and optimizer robustness", heading),
              p("C3: RH beta affects both morphology and numerical cost", sub)]
    b1, b30, bmax = (subset(beta, v, "beta_factor") for v in (1, 30, float(beta.beta_factor.max())))
    story.append(p(f"The RH Bottle test uses {_seeds(beta)} paired seed(s) and "
                   f"{beta.beta_factor.nunique()} beta settings. Beta0 = "
                   f"{_num(med(b1, 'beta_curvature_per_pa'), 10)} N m^-1 Pa^-1. "
                   "Fixed FE at beta = 0 and fixed RH at beta0 remain unchanged throughout the sweep. "
                   "Command cosine and XZ amplitude cosine are reported separately; these are not "
                   "the complex-volume Cp used in C1, C2 and F1."))
    story.append(p(f"At beta0 and 30 beta0, median iterations are {_num(med(b1, 'iterations'), 6)} "
                   f"and {_num(med(b30, 'iterations'), 6)}, while median XZ amplitude cosine to "
                   f"fixed RH is {_num(med(b1, 'xz_field_similarity_to_fixed_rh'), 6)} and "
                   f"{_num(med(b30, 'xz_field_similarity_to_fixed_rh'), 6)}. At the largest setting "
                   f"({beta.beta_factor.max():g} beta0), {_fraction(bmax, 'optimizer_success')} "
                   "meet native stopping. Thus a similar pressure pattern can require substantially "
                   "more iterations. Report dimensional coefficients and normalized beta/beta0, "
                   "rather than a unit-free claim that beta is six orders smaller than alpha. The pressure pattern does not establish three-dimensional stability; B3 reports its mechanics."))
    story += [p("Appendix F / F1: configured solvers and Conventional controls", sub),
              p("All methods receive the same paired initial phases and each objective keeps "
                "its own unchanged loss. FE/GFE use compact L-BFGS-B throughout. Conventional Adam receives no quasi-Newton polish. Time is serial CPU "
                "time to solver return, excluding setup and independent mechanics. Small dots show "
                "starts; bars show median/IQR. Crosses mark unmet native stops in F1(b), and "
                "unresolved roots at the upper edge of F1(d); these are not measured displacements.")]
    rows = []
    for method in ("GFE", "FE", "Conventional"):
        for opt in ("BFGS", "L-BFGS-B", "Adam"):
            q = optimizer[optimizer.method.eq(method) & optimizer.optimizer.eq(opt)]
            if q.empty:
                continue
            rows.append([method, opt, _fraction(q, "raw_gtol_met"),
                         _fraction(q, "all_locally_restoring"),
                         _num(med(q, "solve_wall_s"), 4), _num(med(q, "worst_displacement_m"), 5)])
    story += [table(["Objective", "Optimizer", "Native stop", "Restoring\ncommand", "Median\nreturn (s)", "Median d (m)"],
                    rows, [80, 85, 75, 85, 82, width-407]), Spacer(1, 9)]
    good = optimizer[optimizer.method.isin(("FE", "GFE"))]
    complete = bool(len(good) and _truth(good.raw_gtol_met).all() and _truth(good.all_locally_restoring).all())
    if complete:
        conclusion = (f"All {len(good)} FE/GFE compact L-BFGS-B starts meet native stationarity "
                      "and have resolved, locally restoring elastic equilibria.")
    else:
        conclusion = (f"Across FE/GFE compact L-BFGS-B starts, {_fraction(good, 'raw_gtol_met')} meet native stationarity "
                      f"and {_fraction(good, 'all_locally_restoring')} resolve restoring equilibria.")
    story += [p(conclusion),
              p("The FE/GFE rows evaluate the configured compact L-BFGS-B method. "
                "Conventional controls vary its optimizer while keeping the objective fixed. "
                "A solver return without the declared raw-gradient criterion is not counted as "
                "successful convergence. Raw gradients have objective-dependent scales; "
                "physical validation and within-objective stopping provide complementary evidence.")]

    # Page 4: every editor-named coefficient, its units, actual grid and replication.
    story += [PageBreak(), p("Parameter coverage without a plot atlas", heading),
              p("Values below are read from the active cached sweep tables. A factor multiplies "
                "the listed nominal coefficient; actual dimensional stage values and all endpoints "
                "remain in the exported CSV/NPZ data. S/T denote Single/Triple.")]
    definitions = [
        ("alpha_factor", "α", "S 10; T 3000 m^-1", "S/T, FE+GFE"),
        ("lambda_pressure_factor", "λp", "1 N m^-1 Pa^-1", "S/T, Conventional"),
        ("beta_factor", "β (GFE)", "T 5e-5 N m^-1 Pa^-1; S 0", "T, GFE"),
        ("gamma_uniformity", "γu", "1; dimensionless", "T, GFE"),
        ("epsilon_pressure_rel", "εp", ".001 × local pressure RMS; Pa", "T, GFE; relative values"),
        ("epsilon_force_factor", "εg", ".05 then .01 × initial median residual; N", "T, GFE"),
        ("epsilon_std_factor", "εSTD", ".05 then .01 × initial loss STD; N m^-1", "T, GFE"),
        ("axial_ratio_factor", "W: axial ratio", "Trace fixed; dimensionless", "S/T, GFE"),
        ("transverse_ratio_factor", "W: transverse ratio", "Trace fixed; dimensionless", "S/T, GFE"),
    ]
    rows = []
    for parameter, symbol, nominal, scope in definitions:
        q = sweep(parameter)
        seeds = "/".join(f"{k[0]}:{v}" for k, v in q.groupby("task").seed.nunique().items())
        unit_label = "values" if parameter in ("gamma_uniformity", "epsilon_pressure_rel") else "factors"
        rows.append([symbol, nominal, _values(q) + f"\n{unit_label}; {scope}", seeds])
    rows.append(["β (RH)", f"{_num(med(b1, 'beta_curvature_per_pa'), 10)} N m^-1 Pa^-1",
                 _values(beta, "beta_factor") + "\nfactors; alternative Bottle", str(_seeds(beta))])
    directional = focused.base(ctx) / "directional_review_data/C4_expanded_directional_weights.csv"
    if directional.exists():
        d = pd.read_csv(directional)
        rows.append(["η", ".01; trace-fixed directional weights", _values(d, "eta") +
                     "\nvalues; standard Twin / alternative Bottle", str(_seeds(d))])
    story += [table(["Parameter", "Nominal / units", "Tested grid and scope", "Seeds"],
                    rows, [69, 141, width-252, 42]), Spacer(1, 8),
              p("Curvature and smoothing definitions", sub),
              p("Nominal diagonal curvature weights are W = (1,1,1) for Single FE/GFE, "
                "(1,1,10) for Triple, and (1000,1000,10) for Conventional. Both independent "
                "trace-fixed anisotropy ratios are swept. The initialization-based dimensional "
                "smoothing-scale rule is retained when the local loss changes; dedicated εg and "
                "εSTD sweeps change their multipliers independently."),
              p("The acoustic wavelength λac = c/f = 8.575 mm at 40 kHz is distinct from λp. "
                "Frequency and array generality are reported separately in Appendix G. These symbols are separated in "
                "the exported notation and parameter tables.")]

    # Page 5: compact numerical controls and exact evidence/reviewer boundaries.
    story += [PageBreak(), p("Numerical controls and response scope", heading)]
    numerical = [
        ("stencil_factor", "Physical derivative spacing", "h0 = 0.5 mm; physical transfer samples and denominators rebuilt"),
        ("first_stage_iterations", "First-stage cap", "Total Triple cap preserved; second stage receives remaining updates"),
        ("first_stage_factor", "First smoothing factor", "Second-stage factor .01; initialization-based dimensional scales"),
        ("gtol", "Native infinity-gradient tolerance", "N m^-1 rad^-1; unscaled reduced-phase gradient"),
        ("alpha_x_epsilon_force", "α × εg interaction", "3 × 3 crossed factors; Triple GFE"),
        ("gamma_x_epsilon_std", "γu × εSTD interaction", "3 × 3 crossed factors; Triple GFE"),
    ]
    rows = []
    for parameter, label, rule in numerical:
        q = sweep(parameter)
        if parameter in ("alpha_x_epsilon_force", "gamma_x_epsilon_std"):
            columns = ("alpha_factor", "epsilon_force_factor") if parameter.startswith("alpha") else ("gamma_uniformity", "epsilon_std_factor")
            grid = " × ".join("[" + _values(q, col) + "]" for col in columns)
        else:
            grid = _values(q)
        seeds = "/".join(f"{k[0]}:{v}" for k, v in q.groupby("task").seed.nunique().items())
        rows.append([label, grid, rule + f"; seeds {seeds}"])
    story += [table(["Control", "Recorded values", "Interpretation"], rows,
                    [133, 150, width-283]), Spacer(1, 9)]
    caps = ", ".join(f"{task}: " + _values(q, "cap") for task, q in continuous.groupby("task"))
    story += [p("Stopping and statistical units", sub),
              p(f"Active iteration ceilings: {caps}. The nominal raw infinity-gradient tolerance "
                "is 1e-8; it is distinct from the historical 1e-3 L2 reporting line and historical "
                "2,000/gtol=0 clustering-source bank. Seeds are the independent units. Targets "
                "sharing a phase command, sweep settings, pressure pixels and optimizer iterations "
                "are not independent replications."),
              p(f"The continuous tables contain {len(continuous)} logical records, with "
                f"{_fraction(continuous, 'raw_gtol_met')} meeting their requested native criterion. "
                "Repeated nominal cases reuse commands. Replicated plots show medians and IQR; "
                "one-seed curves describe the tested start without a population confidence interval."),
              p("Defaults and retained numerical audits", sub)]
    first = optimizer[optimizer.method.eq("GFE")].iloc[0]
    config = (f"FE/GFE compact L-BFGS-B: memory={int(first.maxcor)}, "
              f"maxls={int(first.maxls)}, ftol={first.ftol:g}.")
    story += [p(config + " Conventional BFGS and Adam controls are recorded separately. "
                "Window/grid, gauge/embedding, stopping-prefix and phase-perturbation "
                "checks remain exported tables."),
              p("The local analysis window is a measurement volume, not a finite propagation "
                "boundary. Its nominal half-width is 0.75 λac (6.43125 mm), enclosing the local "
                "root-search region and pressure structure. Window sensitivity is tabulated; "
                "invariance across all field patterns is not presumed."),
              p("Separate mechanics section", sub)]
    if not full:
        mechanics = ("The standalone main-equilibrium cell is preserved and remains pending for "
                     "this delivery. It prepares three-dimensional total forces, force Jacobians, "
                     "symmetric-stiffness eigenvalues, the separately defined Gor'kov-potential "
                     "Hessian, perturbations and static load/displacement checks.")
    else:
        mechanics = ("The standalone main-equilibrium cell remains separate. Use its own completion "
                     "record and force/Jacobian/Hessian/perturbation exports to determine execution "
                     "status; this reporting module does not execute or certify that section.")
    story += [p(mechanics + " Local restoring stiffness is not relabeled a conservative potential "
                "Hessian, and static displacement checks are not dynamic trajectory simulations."),
              p("Evidence map: E04 → C1/C2 + parameter tables; E08 → C2 + per-target mechanics; "
                "R2.3 → C3 + weighted-term exports; configured solver controls → F1. E05/E06 are supported "
                "by exact stopping and timing records. Generality and manuscript-level editing "
                "remain separate tasks.", small)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6E0E2"))
        canvas.line(46, 41, A4[0]-46, 41)
        canvas.setFont("FocusedReport", 7.5)
        canvas.setFillColor(colors.HexColor("#5E747C"))
        canvas.drawString(46, 28, "GFE | Focused sensitivity and optimizer evidence | " + mode)
        canvas.drawRightString(A4[0]-46, 28, str(doc.page))
        canvas.restoreState()

    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=46, rightMargin=46,
                      topMargin=43, bottomMargin=53,
                      title="GFE: focused sensitivity and optimizer results guide",
                      author="GFE study").build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader
    content = buffer.getvalue()
    pages = len(PdfReader(BytesIO(content)).pages)
    if pages != 5:
        raise RuntimeError(f"Focused results guide layout expanded to {pages} pages; inspect before release")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    (out / "results_guide_metadata.json").write_text(json.dumps({
        "mode": mode, "pages": pages, "scientific_figures": 22,
        "optimizer_seeds": _seeds(optimizer), "ablation_seeds": _seeds(ablation),
        "continuous_seeds_by_task": continuous.groupby("task").seed.nunique().to_dict(),
        "rh_beta_seeds": _seeds(beta),
        "standalone_equilibrium_completion_certified_by_report": False,
    }, indent=2))
    return str(path)
