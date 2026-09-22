"""Explicit sampling and plotting profiles; importing this module runs nothing."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path


PROFILES = {
    "supersmoke": {
        "profile": "supersmoke",
        "first_result_seed_count": 1,
        "second_result_seed_count": 1,
        "last_result_seed_count": 1,
        "deterministic_timing_repeats": 1,
        "first_result_domain_x_points": 3,
        "first_result_domain_z_points": 3,
        "first_result_domain_restarts": 1,
        "branch_chart_side": 3,
        "pressure_points": 41,
        "objective_plane_points": 21,
        "objective_line_points": 31,
        "force_points": 5,
        "use_smoke_validation": True,
        "main_plots_only": True,
    },
    "smoke": {
        "profile": "smoke",
        "first_result_seed_count": 16,
        "second_result_seed_count": 16,
        "last_result_seed_count": 1,
        "deterministic_timing_repeats": 1,
        "first_result_domain_x_points": 5,
        "first_result_domain_z_points": 3,
        "first_result_domain_restarts": 1,
        "branch_chart_side": 7,
        "use_smoke_validation": True,
        "main_plots_only": False,
    },
    "full": {
        "profile": "full",
        "first_result_seed_count": 16,
        "second_result_seed_count": 100,
        "last_result_seed_count": 5,
        "deterministic_timing_repeats": 5,
        "first_result_domain_x_points": 5,
        "first_result_domain_z_points": 3,
        "first_result_domain_restarts": 1,
        "branch_chart_side": 7,
        "use_smoke_validation": True,
        "main_plots_only": False,
    },
}

MAIN_PREVIEW_STEMS = (
    "Paper_Figure_1_target_grid",
    "Figure_1_single_vortex_convergence",
    "Figure_2_two_branch_evolution",
    "Figure_3_local_decision_geometry",
    "Figure_4_triple_fields_and_decision",
    "Figure_6_force_field_comparison",
    "Figure_7_alpha_transition",
    "Figure_8_performance_time_tradeoff",
)
_FULL_OVERRIDES = frozenset({
    "second_result_seed_count", "last_result_seed_count", "deterministic_timing_repeats",
})


def _profile(mode, overrides=None):
    if mode not in PROFILES:
        raise ValueError(f"Unknown execution profile: {mode}")
    scale = dict(PROFILES[mode])
    overrides = dict(overrides or {})
    if overrides and (mode != "full" or set(overrides) - _FULL_OVERRIDES):
        raise ValueError("Only the declared full sampling counts may be overridden")
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        if key.endswith("_points") and value < 2:
            raise ValueError(f"{key} must contain at least two points")
        scale[key] = value
    return scale


def _refresh_validation_metadata(ctx):
    """Describe the actual protocol after applying a scoped sampling profile."""
    from hat_revision_pipeline import pipeline
    protocol = pipeline._finite_ka_protocol(ctx)
    table = ctx.tables.get("finite_ka_validation_protocol")
    if table is not None:
        table = table.copy()
        table["root_start_count"] = len(protocol["root_starts_a"])
        table["root_start_offsets_a"] = "target, +/-1a on each Cartesian axis"
        table["partial_wave_lmax"] = protocol["numerics"]["lmax"]
        table["paper_scale_smoke_protocol"] = True
        table["protocol_json"] = json.dumps(protocol, sort_keys=True, allow_nan=False)
        ctx.tables["finite_ka_validation_protocol"] = table
    table = ctx.tables.get("finite_ka_numerical_parameters")
    if table is not None:
        table = table.copy()
        values = dict(protocol["numerics"], **{"root start count": len(protocol["root_starts_a"])})
        for index, row in table.iterrows():
            if row["parameter"] in values:
                table.at[index, "value"] = values[row["parameter"]]
        ctx.tables["finite_ka_numerical_parameters"] = table


def apply(ctx, mode, overrides=None):
    """Attach controls after native preparation; no scientific producer is called."""
    scale = _profile(mode, overrides)
    if mode == "smoke":
        # The accepted smoke producers continue to use their original controls.
        ctx.memo["study_scale"] = scale
    else:
        from hat_revision_pipeline import production_main
        production_main.configure_study_scale(ctx, scale)
        if mode == "supersmoke":
            ctx.config = replace(ctx.config, slice_points=scale["pressure_points"],
                                 force_slice_points=scale["force_points"])
            from hat_revision_pipeline.engineering_benchmarks import prepare_smoke_timing
            prepare_smoke_timing(ctx)
        else:
            _refresh_validation_metadata(ctx)
    ctx.memo["execution_profile"] = mode
    if mode == "supersmoke":
        settings_path = ctx.output_root / "resolved_execution_settings.json"
        if settings_path.is_file():
            settings = json.loads(settings_path.read_text())
            settings.update(evidence_status="SUPERSMOKE PREVIEW", study_scale=scale,
                generality="SKIPPED: main-figure preview only",
                run_scope="Eight main plots with lazy numerical dependencies; no appendices or unplotted audits")
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    record = dict(scale,
        configuration_status="configured; numerical work starts in producer cells",
        first_result="Paper Figure 2 / internal producer 1; unchanged smoke sampling",
        second_result="Paper Figure 3 / internal producer 2; root initialization population",
        last_result="Paper Figure 8 / internal producer 8",
        full_expanded_internal_figures=[2, 8],
        full_other_figures_and_appendices="accepted smoke scale",
        single_iteration_cap=10000, triple_iteration_cap=30000,
        objective_definition="unchanged", mechanical_model="unchanged",
        validation_resolution="accepted smoke protocol",
        evidence_status="SUPERSMOKE PREVIEW" if mode == "supersmoke" else (
            "PAPER-SCALE SMOKE" if mode == "smoke" else "FOCUSED FULL SAMPLING"),
        output_root=str(ctx.output_root))
    (ctx.output_root / "run_scale.json").write_text(json.dumps(record, indent=2) + "\n")
    import pandas as pd
    ctx.tables["run_scale"] = pd.DataFrame([record])
    if mode == "supersmoke":
        label_preview_tables(ctx)
    return ctx


def label_preview_tables(ctx):
    """Keep reduced sample counts visibly separate in exported preview evidence."""
    if ctx.memo.get("notebook_mode") != "supersmoke":
        return
    import pandas as pd
    for name, table in list(ctx.tables.items()):
        if isinstance(table, pd.DataFrame):
            table = table.copy()
            table["execution_profile"] = "supersmoke"
            table["evidence_status"] = "SUPERSMOKE PREVIEW"
            table["manuscript_numerics"] = False
            ctx.tables[name] = table
    captions_path = ctx.output_root / "main_gfe_benchmark_captions.json"
    if captions_path.is_file():
        captions = json.loads(captions_path.read_text(encoding="utf-8"))
        note = ("SUPERSMOKE PREVIEW: reduced sampling and plot meshes; "
                "one engineering command per method/task, unchanged objectives and iteration caps. ")
        captions = {name: text if text.startswith(note) else note + text
                    for name, text in captions.items()}
        captions_path.write_text(json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_main_preview(ctx):
    """Assemble the eight already rendered main plots, without appendix producers."""
    if ctx.memo.get("notebook_mode") != "supersmoke":
        raise ValueError("This export is reserved for the supersmoke preview")
    from pypdf import PdfWriter
    from hat_revision_pipeline import pipeline
    directory = Path(ctx.output_root) / "figures"
    paths = [directory / (stem + ".pdf") for stem in MAIN_PREVIEW_STEMS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Render the main-figure cells before assembly. Missing plots: " + ", ".join(missing))
    output = Path(ctx.output_root) / "Supersmoke_Main_Figures.pdf"
    writer = PdfWriter()
    for path in paths:
        writer.append(str(path))
    with output.open("wb") as stream:
        writer.write(stream)
    writer.close()
    label_preview_tables(ctx)
    pipeline.export_all_tables(ctx)
    result = {"mode": "supersmoke", "evidence_status": "SUPERSMOKE PREVIEW",
              "figures_pdf": str(output), "figure_count": len(paths),
              "figures": [str(path) for path in paths],
              "pngs": [str(directory / (stem + ".png")) for stem in MAIN_PREVIEW_STEMS],
              "appendices": "not requested in this profile"}
    (Path(ctx.output_root) / "supersmoke_preview_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result
