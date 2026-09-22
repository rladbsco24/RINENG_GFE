"""GFE presentation and genuine compensated commands for main Figures 5-8.

Existing elastic-bead validation and original command timings are reused.
Figure 7 changes the objective through its force target and has a new cache
identity; uncompensated FE phases are never relabelled as compensated phases.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime" / "src"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hat_revision_pipeline import pipeline as p
from hat_revision_pipeline import final_figures as original
from hat_revision_pipeline import fig8_methods_only as tradeoff
from hat_revision_pipeline.cache import digest_array, digest_file
from hat_revision_pipeline.fig7_xy_patch import compose_figure_7_xy
from hat_revision_pipeline.figure_presentation import add_adjacent_colorbar, outside_grid_panel_labels
from hat_revision_pipeline.gorkov_core import (
    Condition, MethodSpec, SingleTargetObjective, gauge_full, reduce_gauge,
)
from hat_revision_pipeline.multitrap import SphereInFluid, effective_weight_force_target_n
from hat_revision_pipeline.publication_layout import polish_main_figure
from hat_revision_pipeline.sota import FESolveResult
from hat_revision_pipeline.style import COLORS, field_panel, shared_power_norm, target_marker

METHOD_ORDER = ("FE+g", "FE", "Conventional", "IB", "GS", "AD")
ALPHA_VALUES = (1.0e3, 1.0e6, 1.0e7)
TITLES = {
    5: "Figure 5 · Field concentration and elastic-bead stiffness",
    6: "Figure 6 · GFE Triple field and equilibrium comparison",
    7: "Figure 7 · GFE pressure fields at three force weights",
    8: "Figure 8 · Command time, displacement and equilibrium pressure",
}


def _save(ctx, fig, number, stem):
    """Apply publication typography without figure-review titles."""
    for text in fig.findobj(Text):
        text.set_text(text.get_text().replace("FE+$g$", "GFE").replace("FE+g", "GFE"))
    polish_main_figure(fig)
    for legend in fig.legends:
        for label in legend.get_texts():
            if label.get_text() == "GFE":
                label.set_color(COLORS["FE+g"])
                label.set_fontweight("bold")
    return p._save_figure(ctx, fig, stem, tight=False)


def _legend(fig, *, y=0.92):
    handles = []
    for method in METHOD_ORDER:
        hollow = method in {"FE", "IB", "GS"}
        handles.append(mpl.lines.Line2D(
            [], [], marker=original.FIG5_MARKERS[method], linestyle="none",
            markerfacecolor="none" if hollow else COLORS[method],
            markeredgecolor=COLORS[method] if hollow else "white",
            markersize=11 if method == "FE+g" else 8.5,
            markeredgewidth=1.8, label=method,
        ))
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
                      ncol=3, frameon=False, columnspacing=2.2, handletextpad=0.6)


def figure_5(ctx):
    """Retain the measured Single/Triple comparison, with FE as control."""
    single = original._exact_benchmark_with_concentration(ctx)["concentration_table"].copy()
    triple = original._triple_exact_benchmark_with_concentration(ctx)["table"].copy()
    ctx.tables["fig5_exact_force_field_concentration"] = pd.concat([single, triple], ignore_index=True)
    aggregates = pd.concat([
        original._aggregate_force_concentration(single, task="Single"),
        original._aggregate_force_concentration(triple, task="Triple"),
    ], ignore_index=True)
    ctx.tables["fig5_exact_force_field_concentration_aggregates"] = aggregates
    return render_figure_5(ctx, aggregates)


def render_figure_5(ctx, aggregates):
    """Render measured aggregates without repeating the numerical validation."""
    fig = plt.figure(figsize=(9.8, 6.2))
    grid = fig.add_gridspec(1, 2, wspace=0.45)
    for i, task in enumerate(("Single", "Triple")):
        ax = fig.add_subplot(grid[0, i])
        table = aggregates[aggregates.task.eq(task)].set_index("method")
        xlim, ylim, unresolved_y = original._independent_force_concentration_limits(table)
        for method in reversed(METHOD_ORDER):
            row = table.loc[method]
            resolved = bool(row.root_found) and np.isfinite(row.finite_ka_stiffness_min_n_m)
            hollow = method in {"FE", "IB", "GS"}
            ax.scatter(row.field_concentration_fraction,
                       1e3 * row.finite_ka_stiffness_min_n_m if resolved else unresolved_y,
                       marker=original.FIG5_MARKERS[method],
                       s=180 if method == "FE+g" else original.FIG5_MARKER_SIZES[method],
                       facecolor=COLORS[method] if resolved and not hollow else "none",
                       edgecolor=COLORS[method] if hollow or not resolved else "white",
                       linewidth=2.0, zorder=8 if method == "FE+g" else 5)
        ax.axhline(0, lw=1, color="0.4")
        ax.set(xlim=xlim, ylim=ylim, xlabel=r"$C_{\mathrm{ROI}}$",
               ylabel=r"Exact $\kappa_{\min}$ (mN m$^{-1}$)", title=task)
        ax.set_box_aspect(1)
        if task == "Single" and not ctx.memo.get("production_main"):
            row = table.loc["FE+g"]
            ax.annotate("FE+g / FE", (row.field_concentration_fraction, 1e3*row.finite_ka_stiffness_min_n_m),
                        xytext=(-8, -15), textcoords="offset points", ha="right", va="top",
                        fontsize=10, arrowprops=dict(arrowstyle="-", color="0.4"))
    _legend(fig, y=0.925)
    fig.subplots_adjust(left=0.115, right=0.97, top=0.75, bottom=0.14)
    outside_grid_panel_labels(fig, (grid[0, 0], grid[0, 1]), ("a", "b"))
    return _save(ctx, fig, 5, "Figure_5_single_triple_exact_force_concentration")


def finalize_figure_6_visual(ctx):
    """Apply the accepted final visual revision without changing Figure 6 data."""
    import fitz
    from lxml import etree

    stem = "Figure_6_force_field_comparison"
    directory = Path(ctx.output_root) / "figures"
    pdf_path = directory / f"{stem}.pdf"
    png_path = directory / f"{stem}.png"
    svg_path = directory / f"{stem}.svg"
    for path in (pdf_path, png_path, svg_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    document = fitz.open(pdf_path)
    page = document[0]
    removed = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if "deq/a" not in text:
                continue
            rectangle = fitz.Rect(line["bbox"])
            rectangle.y0 += 0.55
            page.add_redact_annot(rectangle, fill=(1, 1, 1), cross_out=False)
            removed.append({"text": text, "rect": list(rectangle)})
    if len(removed) != 5:
        document.close()
        raise RuntimeError(f"Expected five Figure 6 numerical subtitles, found {removed}")
    page.apply_redactions(images=0, graphics=0, text=0)
    remaining_text = page.get_text()
    if "deq/a" in remaining_text:
        document.close()
        raise RuntimeError("Figure 6 numerical subtitle survived finalization")
    for method in ("GFE", "FE", "Conventional", "GS", "AD"):
        if method not in remaining_text:
            document.close()
            raise RuntimeError(f"Figure 6 method title lost: {method}")
    pdf_temporary = pdf_path.with_name(f".{stem}.final.pdf")
    pdf_temporary.unlink(missing_ok=True)
    document.save(pdf_temporary, garbage=4, deflate=True)
    document.close()
    pdf_temporary.replace(pdf_path)

    raster_document = fitz.open(pdf_path)
    pixmap = raster_document[0].get_pixmap(dpi=300, alpha=False)
    png_temporary = png_path.with_name(f".{stem}.final.png")
    png_temporary.unlink(missing_ok=True)
    pixmap.save(png_temporary)
    raster_document.close()
    png_temporary.replace(png_path)

    tree = etree.parse(str(svg_path))
    removed_svg = 0
    for group in tree.xpath(
        '//*[local-name()="g" and starts-with(@id,"text_")]'
    ):
        comments = group.xpath("./comment()")
        if not any("d_{" in str(comment) and "/a=" in str(comment)
                   for comment in comments):
            continue
        for child in list(group):
            if child.tag == etree.Comment or (
                isinstance(child.tag, str) and child.tag.endswith("}g")
            ):
                group.remove(child)
        removed_svg += 1
    if removed_svg != 5:
        raise RuntimeError(
            f"Expected five Figure 6 SVG numerical subtitles, found {removed_svg}"
        )
    svg_temporary = svg_path.with_name(f".{stem}.final.svg")
    svg_temporary.unlink(missing_ok=True)
    tree.write(str(svg_temporary), xml_declaration=True, encoding="utf-8")
    svg_temporary.replace(svg_path)

    ctx.writer.records[:] = [
        record for record in ctx.writer.records
        if Path(str(record.get("path", ""))).stem != stem
    ]
    for path in (png_path, pdf_path, svg_path):
        ctx.writer.record_file(path, "figure")
    return {"removed_subtitles": removed, "pdf": pdf_path,
            "png": png_path, "svg": svg_path}


def _supersmoke_figure6_inputs(ctx):
    """Build only the five displayed T2 cards with unchanged controlled commands."""
    key = "supersmoke_figure6_inputs"
    if key in ctx.memo:
        return ctx.memo[key]
    from hat_revision_pipeline import production_main as production

    # Figure 4 uses the same declared objectives and global FE/GFE solver policy.
    # Figure 8 records end-to-end command timings for its paired case population.
    commands = {
        method: production.ensure_triple_run(ctx, method, 260869, role="controlled")
        for method in ("Conventional", "FE", "GFE")
    }
    commands["GS"] = production._holography_run(ctx, "Triple", "GS")
    commands["AD"] = production._holography_run(
        ctx, "Triple", "AD", int(ctx.config.random_seed)
    )
    targets = np.asarray(commands["GFE"]["targets"], dtype=float)
    target = targets[1]
    evaluations = []
    for method in ("GFE", "FE", "Conventional", "GS", "AD"):
        label = "FE+g" if method == "GFE" else method
        phase = np.asarray(commands[method]["phase"], dtype=float)
        result = p._finite_ka_validation(
            ctx, phase, target, key=f"supersmoke-figure6-{label}-T2"
        )
        row = p._static_validation_row(label, result)
        field = p._array_field(ctx, phase)
        equilibrium = np.asarray(result.equilibrium.equilibrium_m, dtype=float)
        row.update(
            method=label, display_method=label, task="Triple", target_index=1,
            target_id="T2", phase_content_id=digest_array(phase),
            pressure_abs_at_intended_target_pa=float(abs(field.pressure(target[None, :])[0])),
            pressure_abs_at_exact_equilibrium_pa=(
                float(abs(field.pressure(equilibrium[None, :])[0]))
                if result.equilibrium.numerical_root_found else np.nan
            ),
            execution_profile="supersmoke", evidence_status="SUPERSMOKE PREVIEW",
            manuscript_numerics=False,
            validation_scope="displayed T2 card only; unchanged independent finite-ka validator",
            **{f"target_{axis}_m": float(target[index]) for index, axis in enumerate("xyz")},
        )
        evaluations.append(dict(row=row, result=result, spec=dict(method=label, phase=phase)))
    benchmark = dict(targets_m=targets, evaluations=evaluations)
    compensation = dict(phase=np.asarray(commands["GFE"]["phase"], dtype=float))
    ctx.tables["fig6_supersmoke_target_mechanics"] = pd.DataFrame(
        [item["row"] for item in evaluations]
    )
    ctx.memo[key] = (benchmark, compensation)
    return benchmark, compensation


def figure_6(ctx):
    """Show the true compensated broad field and the same five T2 force cards."""
    if ctx.memo.get("study_scale", {}).get("profile") == "supersmoke":
        benchmark, compensation = _supersmoke_figure6_inputs(ctx)
    else:
        benchmark = original._triple_exact_benchmark_with_concentration(ctx)
        compensation = original._figure6_vertical_compensated_fe(ctx, benchmark)
    methods = ("FE+g", "FE", "Conventional", "GS", "AD")
    selected = {}
    for method in methods:
        matches = [item for item in benchmark["evaluations"]
                   if item["row"]["method"] == method and int(item["row"]["target_index"]) == 1]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one Triple T2 record for {method}")
        selected[method] = matches[0]
    target = np.asarray(benchmark["targets_m"][1], float)
    phase = np.asarray(compensation["phase"], float)
    context_payload = {"phase": digest_array(phase), "samples": original._plot_points(ctx, "pressure_points", 181 if ctx.memo.get("production_main") else 141 if ctx.config.full else 101),
                       "targets_m": np.asarray(benchmark["targets_m"]).tolist(),
                       "contract": "feg-main-figure6-broad-xy-v1"}
    context, _, _ = ctx.cache.get_or_compute(
        "feg_main_pressure_context", context_payload,
        lambda: original._tri3_xy_pressure(ctx, phase, samples=context_payload["samples"]),
        recompute=ctx.recompute)
    context_axis, context_amplitude = context
    dense = {}
    for method, item in selected.items():
        local_phase = np.asarray(item["spec"]["phase"], float)
        payload = {"phase": digest_array(local_phase), "target": target.tolist(),
                   "half_width_m": 0.006, "samples": original._plot_points(ctx, "pressure_points", 181 if ctx.memo.get("production_main") else 91),
                   "positions": digest_array(ctx.positions_m), "frequency_hz": ctx.config.frequency_hz}
        dense[method], _, _ = ctx.cache.get_or_compute(
            "feg_main_pressure_card", payload,
            lambda local_phase=local_phase: p._pressure_sections(p._array_field(ctx, local_phase), target,
                     half_xy_m=0.006, half_z_m=0.006,
                     samples=original._plot_points(ctx, "pressure_points", 181 if ctx.memo.get("production_main") else 91)), recompute=ctx.recompute)
    norm = shared_power_norm([context_amplitude, *[dense[m]["xz"] for m in methods]], lower=0, upper=99.5)
    ticks = np.linspace(0, float(norm.vmax), 4)
    fig = plt.figure(figsize=(10.8, 14.0))
    grid = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.42)
    ax = fig.add_subplot(grid[0, 0])
    image = field_panel(ax, context_amplitude,
        (1e3*context_axis[0], 1e3*context_axis[-1], 1e3*context_axis[0], 1e3*context_axis[-1]),
        norm=norm, xlabel=r"$x$ (mm)", ylabel=r"$y$ (mm)")
    for index, point in enumerate(np.asarray(benchmark["targets_m"]), 1):
        target_marker(ax, 1e3*point[0], 1e3*point[1], marker="x", s=54, linewidth=1.7, zorder=20)
        ax.annotate(rf"$T_{index}$", (1e3*point[0], 1e3*point[1]), xytext=(5, 5),
                    textcoords="offset points", color="white", fontweight="bold")
    ax.scatter(1e3*target[0], 1e3*target[1], s=110, facecolor="none", edgecolor="white", linewidth=1.4)
    ax.set_title(r"FE+$g$ · Triple $xy$")
    ax.set_box_aspect(1)
    add_adjacent_colorbar(image, ax, size="3.4%", pad="2.4%", label=r"$|p|$ (Pa)", ticks=ticks)
    rows = []
    for index, method in enumerate(methods, 1):
        item = selected[method]
        result = item["result"]
        phase = np.asarray(item["spec"]["phase"], float)
        equilibrium = np.asarray(result.equilibrium.equilibrium_m, float)
        resolved = bool(result.equilibrium.numerical_root_found)
        ax = fig.add_subplot(grid[index//2, index%2])
        display = r"FE+$g$" if method == "FE+g" else method
        if resolved:
            displacement = float(item["row"]["finite_ka_displacement_a"])
            pressure = float(item["row"]["pressure_abs_at_exact_equilibrium_pa"])
            if np.isfinite(displacement) and np.isfinite(pressure):
                title = (display + "\n" + rf"$d_{{\rm eq}}/a={displacement:.2f},\ "
                         + rf"|p_{{\rm eq}}|={pressure:.0f}\ \mathrm{{Pa}}$")
            else:
                title = display + "\nequilibrium metrics unavailable"
        else:
            title = display + "\nno resolved root"
        p._plot_force_card(ctx, ax, phase, target, equilibrium,
            equilibrium_resolved=resolved, plane="xz", key=f"final-fig6-triple-T2-{method}-xz",
            title=title, dense=dense[method], pressure_norm=norm)
        for text in list(ax.texts):
            if text.get_text() == "unresolved candidate":
                text.remove()
        if resolved:
            delta = 1e3*(equilibrium-target)
            ax.plot([0, delta[0]], [0, delta[2]], color="white", linewidth=1.15, zorder=19)
        ax.set(xlim=(-6, 6), ylim=(-6, 6))
        ax.set_box_aspect(1)
        add_adjacent_colorbar(ax.images[0], ax, size="3.4%", pad="2.4%", label=r"$|p|$ (Pa)", ticks=ticks)
        from hat_revision_pipeline.exact_validator import ElasticSphere
        rows.append(dict(item["row"], finite_ka_displacement_m=float(item['row']['finite_ka_displacement_a'])*ElasticSphere().radius_m,
                         context_method="FE+g", context_phase_content_id=context_payload["phase"],
                         context_plane="xy", plane="xz", analyzed_target_id="T2",
                         pressure_norm_vmin_pa=float(norm.vmin), pressure_norm_vmax_pa=float(norm.vmax)))
    ctx.tables["fig6_triple_force_field_summary"] = pd.DataFrame(rows)
    fig.subplots_adjust(left=0.105, right=0.95, top=0.925, bottom=0.06)
    outside_grid_panel_labels(fig, tuple(grid[r,c] for r in range(3) for c in range(2)), tuple("abcdef"), dx=.016, dy=.008)
    figure = _save(ctx, fig, 6, "Figure_6_force_field_comparison")
    finalize_figure_6_visual(ctx)
    return figure



def _solve_figure7_bfgs(objective, initial_phase_rad, *, maxiter, gtol):
    """Restore the deposited Figure 7 BFGS solve without changing global FE policy."""
    initial = np.asarray(initial_phase_rad, dtype=float)
    if initial.shape != (int(objective.n_transducers),):
        raise ValueError("Figure 7 initial phase has an unexpected shape")
    if int(maxiter) <= 0 or not np.isfinite(gtol) or float(gtol) < 0.0:
        raise ValueError("Figure 7 maxiter and gtol must be valid")
    options = {"maxiter": int(maxiter), "gtol": float(gtol), "disp": False}
    reduced0 = reduce_gauge(initial, objective.gauge_index)
    cache = {}

    def fg(x):
        key = np.asarray(x, dtype=float).tobytes()
        if cache.get("key") != key:
            value, gradient = objective.fun_grad(x)
            cache.update(
                key=key,
                value=float(value),
                gradient=np.asarray(gradient, dtype=float),
            )
        return float(cache["value"]), np.asarray(cache["gradient"], dtype=float)

    start = time.perf_counter()
    initial_value, initial_gradient = fg(reduced0)
    history_wall = [0.0]
    history_objective = [float(initial_value)]
    history_gradient = [float(np.linalg.norm(initial_gradient))]
    history_phase = [gauge_full(reduced0, objective.gauge_index)]

    def callback(xk):
        value, gradient = fg(xk)
        history_wall.append(float(time.perf_counter() - start))
        history_objective.append(float(value))
        history_gradient.append(float(np.linalg.norm(gradient)))
        history_phase.append(
            gauge_full(np.asarray(xk, dtype=float), objective.gauge_index)
        )

    result = minimize(
        lambda x: fg(x)[0],
        reduced0,
        jac=lambda x: fg(x)[1],
        method="BFGS",
        callback=callback,
        options=options,
    )
    value, gradient = fg(np.asarray(result.x, dtype=float))
    elapsed = float(time.perf_counter() - start)
    if not history_wall or elapsed > history_wall[-1] + 1.0e-12:
        history_wall.append(elapsed)
        history_objective.append(float(value))
        history_gradient.append(float(np.linalg.norm(gradient)))
        history_phase.append(
            gauge_full(np.asarray(result.x, dtype=float), objective.gauge_index)
        )
    full_phase = gauge_full(np.asarray(result.x, dtype=float), objective.gauge_index)
    full_phase = np.angle(np.exp(1j * full_phase))
    return FESolveResult(
        phase_rad=full_phase,
        objective=float(value),
        gradient_norm=float(np.linalg.norm(gradient)),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        history_wall_s=np.asarray(history_wall, dtype=float),
        history_objective=np.asarray(history_objective, dtype=float),
        history_gradient_norm=np.asarray(history_gradient, dtype=float),
        history_phase_rad=np.angle(
            np.exp(1j * np.asarray(history_phase, dtype=float))
        ),
        optimizer="BFGS",
        optimizer_options=dict(options),
        gradient_inf_norm=float(np.linalg.norm(gradient, ord=np.inf)),
        stationary_gtol=bool(
            np.linalg.norm(gradient, ord=np.inf) <= float(options["gtol"])
        ),
    )


def alpha_bank(ctx):
    """Three cold-start FE+g solves with the original alpha examples and seed."""
    key = "feg_main_alpha_bank"
    if key in ctx.memo:
        return ctx.memo[key]
    seed = int(ctx.config.random_seed)
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ctx.positions_m))
    upward = effective_weight_force_target_n(SphereInFluid(frequency_hz=ctx.config.frequency_hz))
    source = ROOT / "runtime" / "src" / "hat_revision_pipeline"
    hashes = {name: digest_file(source / name) for name in ("gorkov_core.py", "sota.py")}
    hashes["main_feg_benchmarks.py"] = digest_file(Path(__file__))
    runs, rows = [], []
    for alpha in ALPHA_VALUES:
        method = MethodSpec(name="Force-Equilibrium", alpha_per_m=alpha, beta_curvature_per_pa=0.,
                            pressure_mode="abs", curvature_weight=np.eye(3))
        condition = Condition(label=f"feg-alpha-{alpha:g}", positions=ctx.positions_m,
            target_m=tuple(p.MAIN_TARGET_M), frequency_hz=ctx.config.frequency_hz,
            curvature_weight=np.eye(3), normals=ctx.normals)
        objective = SingleTargetObjective(condition, method, force_target_n=upward)
        payload = {"contract": "feg-main-alpha-transition-figure7-bfgs-v2", "method": asdict(method),
                   "force_target_n": upward.tolist(), "target_m": p.MAIN_TARGET_M.tolist(),
                   "positions": digest_array(ctx.positions_m), "normals": digest_array(ctx.normals),
                   "frequency_hz": ctx.config.frequency_hz, "seed": seed,
                   "initial": digest_array(initial), "maxiter": 10000, "gtol": ctx.config.gtol,
                   "optimizer": "BFGS", "optimizer_scope": "Figure_7_alpha_transition_only",
                   "source_content_id": hashes}
        run, status, path = ctx.cache.get_or_compute("feg_main_alpha_run_figure7_bfgs", payload,
            lambda objective=objective: _solve_figure7_bfgs(objective, initial, maxiter=10000, gtol=ctx.config.gtol),
            recompute=ctx.recompute)
        raw = objective._raw_metrics(run.phase_rad, need_phase_gradient=False)
        eigs = np.linalg.eigvalsh(np.asarray(raw["potential_hessian_n_m"]))
        rows.append(dict(method="GFE", cache_method="FE+g", alpha_per_m=alpha, seed=seed, maxiter=10000,
            optimizer=str(run.optimizer), optimizer_scope="Figure_7_alpha_transition_only",
            iterations=int(run.iterations), gradient_norm=float(run.gradient_norm),
            objective=float(run.objective), wall_time_s=float(run.history_wall_s[-1]),
            optimizer_success=bool(run.success), optimizer_status=int(run.status), optimizer_message=str(run.message),
            phase_content_id=digest_array(run.phase_rad), initial_phase_content_id=digest_array(initial),
            prescribed_radiation_force_z_n=float(upward[2]),
            target_gorkov_hessian_min_n_m=float(eigs[0]), target_gorkov_hessian_mid_n_m=float(eigs[1]),
            target_gorkov_hessian_max_n_m=float(eigs[2]),
            cache_path=str(path.relative_to(ctx.output_root)), result_scope="SMOKE", timing_source="original optimization"))
        runs.append(run)
    table = pd.DataFrame(rows)
    ctx.tables["fig7_alpha_transition"] = table
    ctx.tables["fig7_feg_alpha_examples"] = table.copy()
    bank = dict(runs=runs, phases=np.stack([run.phase_rad for run in runs]), table=table, seed=seed)
    ctx.memo[key] = bank
    return bank


def figure_7(ctx):
    bank = alpha_bank(ctx)
    kwargs = {"samples": 181} if ctx.memo.get("production_main") else {}
    if ctx.memo.get("study_scale", {}).get("profile") == "supersmoke":
        kwargs["samples"] = original._plot_points(ctx, "pressure_points", 181)
    fig = compose_figure_7_xy(ctx, bank, ALPHA_VALUES, p.MAIN_TARGET_M, **kwargs)
    fig.subplots_adjust(top=0.895)
    return _save(ctx, fig, 7, "Figure_7_alpha_transition")


def figure_8(ctx):
    from hat_revision_pipeline.engineering_benchmarks import ensure_engineering_benchmarks
    engineering = ensure_engineering_benchmarks(ctx)
    single = engineering["Single"]
    triple = engineering["Triple"]
    table = tradeoff.methods_only_tradeoff_table(ctx, single, triple, original._command_time)
    ctx.tables["fig8_performance_time_tradeoff"] = table
    return render_figure_8(ctx, table)


def render_figure_8(ctx, table):
    """Render cached or newly measured command metrics with identical styling."""
    fig = tradeoff.render_methods_only_tradeoff(ctx, table, save_figure=lambda _ctx, figure, _stem: figure)
    for legend in list(fig.legends):
        legend.remove()
    _legend(fig, y=.94)
    # Artist changes are purely visual: every point retains the original coordinates.
    for ax in fig.axes:
        for collection in ax.collections:
            if not isinstance(collection, PathCollection) or not len(collection.get_facecolors()):
                continue
            rgb = collection.get_facecolors()[0, :3]
            if np.allclose(rgb, mpl.colors.to_rgb(COLORS["FE+g"])):
                collection.set_sizes(collection.get_sizes()*1.5)
                collection.set_zorder(10)
            elif np.allclose(rgb, mpl.colors.to_rgb(COLORS["FE"])):
                collection.set_facecolor("none")
                collection.set_edgecolor(COLORS["FE"])
                collection.set_linewidth(2)
    fig.subplots_adjust(top=.81)
    return _save(ctx, fig, 8, "Figure_8_performance_time_tradeoff")


def render(ctx):
    """Generate Figures 5–8 using the same context as the complete notebook."""
    figures = []
    for function in (figure_5, figure_6, figure_7, figure_8):
        print(f"Rendering {function.__name__}", flush=True)
        figures.append(function(ctx))
    public_tables(ctx)
    return figures


def public_captions(ctx):
    """Export captions with current numerical and timing scope on every Run All."""
    captions = {'Figure_5_single_triple_exact_force_concentration': 'Field concentration C_ROI and the smallest symmetric elastic-bead stiffness eigenvalue for Single and Triple commands. GFE uses the buoyancy-corrected weight as the prescribed upward Gor’kov radiation-force target; FE is the uncompensated control. Each point is the within-task median; all methods use the same independent finite-ka elastic-bead validator. The GFE/FE Single points nearly overlap. SMOKE numerical scope.', 'Figure_6_force_field_comparison': 'The broad XY context shows the actual GFE Triple command. The five target-centred XZ cards compare GFE, FE, Conventional, GS and AD at T2 using one shared pressure normalization and independent elastic-bead total-force streamlines. Crosses mark intended targets and open circles mark resolved equilibria. Displacement in meters and pressure at equilibrium are provided in the exported table, together with all three GFE and FE targets.', 'Figure_7_alpha_transition': 'Three GFE commands at alpha = 1e3, 1e6 and 1e7 m^-1, keeping the original common seed, array, target, identity curvature weight, beta = 0 and 10000-iteration BFGS cap. Each command explicitly includes the effective-weight radiation-force target and has a new phase-verified cache identity. Broad XY pressure sections retain the deposited +/-75 mm domain and individual pressure scales. Each command retains its actual solver status and terminal gradient; a terminal field alone does not establish convergence.'}
    scale = ctx.memo.get("study_scale", {})
    if scale.get("profile") == "supersmoke":
        captions["Figure_6_force_field_comparison"] = (
            "Supersmoke plot preview: the broad XY context shows the controlled GFE Triple command. "
            "Five T2 XZ cards compare GFE, FE, Conventional, GS and AD with shared pressure normalization "
            "and independent finite-ka elastic-bead total-force streamlines. Only these five displayed "
            "T2 outcomes are validated; unplotted target, viscous, Jacobian-step and concentration audits "
            "are omitted. Objectives are unchanged; FE/GFE uses compact evaluation: L-BFGS-B for Single and BFGS for Triple."
        )
    count = int(scale.get("last_result_seed_count", 1)) if ctx.config.full else 1
    repeats = int(scale.get("deterministic_timing_repeats", 1)) if ctx.config.full else 1
    caption = (
        "Measured end-to-end command time versus independently validated elastic-bead "
        "equilibrium displacement in meters and equilibrium pressure for Single (top) "
        "and Triple (bottom). All FE/GFE commands use compact evaluation: Single uses "
        "L-BFGS-B without phase bounds and Triple uses BFGS; Conventional uses BFGS. Iteration ceilings are 10000 "
        "for Single and 30000 for Triple. Each stochastic method uses "
        f"{count} paired command initializations per task. Markers are arithmetic means "
        "and vertical bars show one sample standard deviation (ddof=1) across command "
        "cases. Triple metrics are first reduced to a median over the three targets of "
        "each command; targets are not counted as independent command cases. IB and GS "
        f"each have one deterministic endpoint and {repeats} complete timing observations; "
        "no mechanical SD is estimated from a singleton. Time coordinates are means "
        "of measured command times; timing SD is exported separately. Unresolved "
        "commands do not enter equilibrium-metric means, and metric-specific valid "
        "counts and total command counts are exported. The common timing boundary "
        "includes problem initialization, acoustic transfer construction, applicable "
        "device transfer and compilation, synchronized solve, and terminal host "
        "extraction; independent mechanical validation, cache retrieval, export, and "
        "rendering are excluded. The pooled time axis is broken linear."
    )
    summary = ctx.tables.get("fig8_performance_time_tradeoff")
    if isinstance(summary, pd.DataFrame) and len(summary):
        counts = []
        for row in summary.itertuples(index=False):
            method = "GFE" if row.method == "FE+g" else str(row.method)
            counts.append(
                f"{method}: {int(row.single_resolved_count)}/{int(row.single_endpoint_count)} "
                f"Single, {int(row.triple_resolved_count)}/{int(row.triple_endpoint_count)} Triple"
            )
        caption += " Resolved/total command cases: " + "; ".join(counts) + "."
    captions["Figure_8_performance_time_tradeoff"] = caption
    (ctx.output_root / "main_gfe_benchmark_captions.json").write_text(
        json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def with_si_displacements(table):
    """Retain normalized audit values and add dimensional display aliases."""
    from hat_revision_pipeline.exact_validator import ElasticSphere
    copy = table.copy()
    for column in table.columns:
        if 'displacement' in column and '_a' in column:
            head, _, tail = column.rpartition('_a')
            if tail in ('', '_q25', '_q75', '_mean', '_median', '_std', '_minimum', '_maximum', '_ci_low', '_ci_high') and pd.api.types.is_numeric_dtype(table[column]):
                copy[head + '_m' + tail] = table[column] * ElasticSphere().radius_m
    return copy


def public_tables(ctx):
    """Name the compensated objective GFE while leaving cache objects intact."""
    public_captions(ctx)
    for name, table in list(ctx.tables.items()):
        if not isinstance(table, pd.DataFrame):
            continue
        copy = with_si_displacements(table)
        for column in copy.select_dtypes(include=["object", "string"]).columns:
            if column == "cache_method":
                continue
            copy[column] = copy[column].map(
                lambda value: value.replace("FE+g", "GFE") if isinstance(value, str) else value)
        ctx.tables[name] = copy


if __name__ == "__main__":
    from hat_revision_pipeline.style import configure_style
    configure_style()
    context = p.prepare_pipeline(run_mode="quick", output_root=ROOT / "smoke_outputs")
    render(context)
    p.export_all_tables(context)
    (ROOT / "smoke_outputs" / "main_feg_benchmarks_cache_access.json").write_text(
        json.dumps(context.cache.access_records, indent=2) + "\n")
