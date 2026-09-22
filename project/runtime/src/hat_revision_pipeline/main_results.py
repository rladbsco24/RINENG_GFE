"""Export writing records from the main figures without additional experiments."""
from __future__ import annotations
import json
import shutil
import numpy as np
import pandas as pd
from . import pipeline as p
from .cache import digest_array
from .revision_measurements import _items


def export_main_results(ctx):
    directory = ctx.output_root / "revision_measurements"
    directory.mkdir(parents=True, exist_ok=True)
    commands_dir = ctx.output_root / "data" / "main_phase_commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    records = []
    sentences = ["ALL CURRENT AND RECOVERED RUNS ARE SMOKE. NOT MANUSCRIPT RESULTS.",
                 "Main figures only. No Triple long-branch experiment was executed.", ""]
    seen = set()
    for item in _items(ctx):
        source_row, result, spec = item["row"], item["result"], item["spec"]
        # Keep cache identities unchanged while applying the public GFE name.
        row = {key: value.replace("FE+g", "GFE") if isinstance(value, str) else value
               for key, value in source_row.items()}
        row["cache_method"] = source_row["method"]
        phase = np.asarray(spec["phase"])
        content_id = digest_array(phase)
        if row.get("phase_content_id") != content_id:
            raise RuntimeError("Main result row does not match its current phase command")
        identity = (row["task"], row["method"], row["target_id"], content_id)
        if identity in seen:
            raise RuntimeError(f"Duplicate main benchmark record: {identity}")
        seen.add(identity)
        path = commands_dir / (content_id + ".npz")
        if not path.exists():
            np.savez_compressed(path, phase_rad=phase, positions_m=ctx.positions_m,
                                frequency_hz=ctx.config.frequency_hz)
        ctx.writer.record_file(path, "main-phase-command")
        records.append(dict(row, command_file=path.relative_to(ctx.output_root).as_posix(),
            force_jacobian_n_m_json=json.dumps(np.asarray(result.force_jacobian_n_m).tolist()),
            symmetric_stiffness_eigenvalues_n_m_json=json.dumps(np.asarray(result.symmetric_stiffness_eigenvalues_n_m).tolist())))
        sentences.append(f"[SMOKE; main_mechanical_results.csv row {len(records)-1}; phase {content_id}] "
            f"For {row['task']} {row['method']} ({row['target_id']}), the elastic-bead equilibrium displacement "
            f"was {row['finite_ka_displacement_a']:.6g} a "
            f"(dx, dy, dz = {row['finite_ka_displacement_x_m']*1e3:.6g}, "
            f"{row['finite_ka_displacement_y_m']*1e3:.6g}, {row['finite_ka_displacement_z_m']*1e3:.6g} mm), "
            f"the minimum symmetric stiffness was {row['finite_ka_stiffness_min_n_m']*1e3:.6g} mN/m, "
            f"and the equilibrium pressure was {row['pressure_abs_at_exact_equilibrium_pa']:.6g} Pa.")
    ctx.tables["main_mechanical_results"] = pd.DataFrame(records)
    # Refresh the legacy writing alias from the current Figure 8 aggregate.
    # Never read a deposited alias CSV to reconstruct current measurements.
    tradeoff = ctx.tables["fig8_performance_time_tradeoff"].copy()
    for column in tradeoff.select_dtypes(include=["object", "string"]):
        tradeoff[column] = tradeoff[column].map(
            lambda value: value.replace("FE+g", "GFE") if isinstance(value, str) else value)
    ctx.tables["figure_8_tradeoff_rows"] = tradeoff
    for row in ctx.tables["fig4_triple_local_geometry_summary"].itertuples(index=False):
        sentences.append(f"[SMOKE; fig4_triple_local_geometry_summary.csv; {row.method}] "
            f"The two-dimensional section spans +/-{row.half_range_rad:g} rad along both frozen axes; "
            f"the projected gradient at the marked endpoint is "
            f"({row.gradient_s1_n_per_m_rad:.6g}, {row.gradient_s2_n_per_m_rad:.6g}) N/m/rad. "
            f"The optimizer's free-phase gradient L2 norm is {row.optimizer_gradient_norm_n_per_m_rad:.6g} N/m/rad. "
            "Sparse glyphs show local negative projected analytic gradient directions; equal glyph length does not encode magnitude or a convergence trajectory.")
    for task, group in ctx.tables["main_mechanical_results"].groupby("task", sort=False):
        fe = group[group.method.eq("FE")]
        fg = group[group.method.eq("GFE")]
        if len(fe) and len(fg):
            before = float(fe.finite_ka_displacement_a.median())
            after = float(fg.finite_ka_displacement_a.median())
            sentences.append(f"[SMOKE; main_mechanical_results.csv; {task}] "
                f"FE and GFE had median elastic-bead equilibrium displacements of "
                f"{before:.6g} a and {after:.6g} a, respectively; "
                f"the signed relative reduction was {100*(before-after)/before:.6g}%.")
    for name, frame in list(ctx.tables.items()):
        frame = pd.DataFrame(frame).copy()
        frame["evidence_status"] = "SMOKE"
        frame["manuscript_numerics"] = False
        ctx.tables[name] = frame
    p.export_all_tables(ctx)
    path = directory / "revision_results_smoke.txt"
    path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    mock = directory / "revision_results_mock.txt"
    mock.write_text("MOCK SENTENCE TEMPLATES. PLACEHOLDERS ARE NOT RESULTS. ALL RUNS ARE SMOKE.\n\n" +
        "\n".join(f"For {task} {method}, the elastic-bead equilibrium displacement was {{displacement_a}} a; "
                   "the signed vertical offset was {dz_mm} mm, the minimum symmetric stiffness was "
                   "{stiffness_mN_m} mN/m, the equilibrium pressure was {pressure_eq_Pa} Pa, "
                   "and the measured command time was {time_s} s."
                   for task in ("Single", "Triple")
                   for method in ("Conventional", "FE", "GFE", "IB", "GS", "AD")) + "\n",
        encoding="utf-8")
    for output in (path, mock):
        ctx.writer.record_file(output, "main-result-sentences")
    return path
