"""Portable parameter exports for the complete 31-figure study; no solves."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from importlib import metadata

from rineng_content_id import content_identity
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def _content_id(path):
    return content_identity(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def _source_record(path, anchor):
    lines = path.read_text().splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if anchor in line]
    if not matches:
        raise ValueError(f"Parameter source anchor is absent: {path}: {anchor}")
    return matches[0]


def _functions(path, names):
    source = path.read_text()
    return [{"source_file": path.parent.name + "/" + path.name,
             "source_content_id": _content_id(path), "function": node.name, "line": node.lineno,
             "source_expression": ast.get_source_segment(source, node)}
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in names]


def write_parameters(root, general_project, output_root, mode):
    """Export current parameters, selected inputs and distinct outcome pointers.

    general_project is revision_f1_f2 (possibly in an isolated run workspace).
    The sibling tables directory supplies the frozen selected coefficient bank.
    All reads are local; generation modules are inspected, never imported.
    """
    import pandas as pd
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    root, project, output = map(lambda p: Path(p).resolve(), (root, general_project, output_root))
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "source_records"
    saved.mkdir(exist_ok=True)
    source_rows = []

    def source(path, role):
        path = Path(path)
        label = ("study/" + path.relative_to(root).as_posix() if path.is_relative_to(root)
                 else "generality/" + path.relative_to(project.parent).as_posix()
                 if path.is_relative_to(project.parent) else str(path))
        row = {"path": label, "content_id": _content_id(path), "bytes": path.stat().st_size,
               "evidence_role": role}
        source_rows.append(row)
        return row

    base = root / "appendix_outputs/P_parameters/P_parameters.csv"
    if mode == "full":
        candidates = list((root / "production_outputs").rglob("P_parameters.csv"))
        if len(candidates) == 1:
            base = candidates[0]
    parameters = pd.read_csv(base).fillna("")
    parameters["evidence_role"] = "deposited main/A-F parameter inventory; exact original source fields retained"
    parameters["section"] = "Main/A-F"
    source(base, "original parameter records")
    shutil.copy2(base, saved / "P_parameters_original.csv")
    single = project / "unified_single_generation.py"
    triple = project / "unified_triple_generation.py"
    validator = project / "compute_appendix_case.py"
    for path in (single, triple, validator, project / "appendix_settings.py",
                 project / "final_decision_data.py", project / "decision_reveal.py",
                 project / "pressure_data.py", project / "decision_data.py"):
        source(path, "active generation or sampling source")
    physics = project.parent / "generality_package/runtime/src/hat_revision_pipeline"
    for name in ("gorkov_core.py", "multitrap.py", "fixed_fe_endpoints.py", "sota.py",
                 "exact_validator.py", "triple_long_iteration.py", "config.py"):
        source(physics / name, "generality numerical source")
    for name in ("unified_study.py", "unified_figure_pdf.py", "unified_export_support.py",
                 "unified_parameters.py", "tools/build_final_notebook.py",
                 "main_data_generation.py", "run_notebook.py",
                 "run_study.py", "production_settings.py", "requirements.txt"):
        path = root / name
        if path.exists(): source(path, "active orchestration or environment specification")

    records = []
    def add(parameter, value, units, path, anchor, *, applicability="both", note="", role="declared generation protocol"):
        records.append(dict(group="Unified generality", section="G", parameter=parameter,
            value=json.dumps(value) if isinstance(value, (list, dict)) else value,
            units=units, applicability=applicability, source_file=str(path.relative_to(project.parent)),
            source_function="", source_line=_source_record(path, anchor), note=note,
            evidence_role=role))

    add("single_seed_values", list(range(260828, 260832)), "seed", single, "for seed in range(260828, 260832)")
    add("single_paired_starts", 4, "starts per setting", single, "paired_starts=4")
    add("single_reference_endpoints", 8, "FE/GFE endpoints per setting", single, "reference_endpoints=8",
        note="Four FE and four GFE solves form one fixed same-setting bank; Conventional is excluded.")
    add("single_cold_maxiter", 10000, "accepted iterations", single, "path, maxiter=10000, gtol=1e-8")
    add("single_cold_gtol", 1e-8, "native reduced-phase gradient tolerance", single, "path, maxiter=10000, gtol=1e-8")
    add("single_pressure_grid", {"smoke": 201, "full": 301}, "points per axis", single, "points=301 if mode == \"full\" else 201")
    add("single_pressure_half_width", .5, "wavelength", project / "pressure_data.py", "field_half_width_lambda=.5")
    add("native_local_decision_grid", 51, "points per axis", single, "grid_size=51")
    add("final_decision_grid", {"smoke": 61, "full": 101}, "points per axis", single, "fd.GRID_SIZE = 101")
    add("final_decision_default_half_range", 3., "L2 phase rad", single, "fd.HALF_RANGE = 3.",
        note="Per-setting wider selections are separately exported from decision_reveal/selection.json.")
    add("triple_seed_values", {"smoke": [260869], "full": list(range(260869, 260879))}, "seed", triple, "260869 + (1 if mode")
    add("triple_cold_maxiter", 30000, "accepted iterations", triple, "cold_iteration_cap=30000")
    add("triple_intervention_maxiter", 4000, "accepted iterations per selected intervention", triple, "intervention_iteration_cap=4000")
    add("triple_stencil_spacing", ".0005 * 40000 / frequency_hz", "m", triple, "stencil_m=.0005 * 40000")
    add("triple_native_checkpoints", [10000, 20000, 30000], "accepted iterations", triple, "checkpoints=(10000, 20000, 30000)")
    add("triple_perturbed_restart_sigma", .35, "phase rad", triple, "normal(0., .35")
    add("triple_proposal_seed", "seed + 401 + (0 if method == FE else 1000)", "seed", triple, "proposal_seed = seed + 401")
    add("validation_lmax", {"smoke": 6, "full": 8}, "partial-wave order", validator, "lmax=8 if full else 6")
    add("validation_root_max_nfev", {"smoke": 22, "full": 70}, "evaluations per start", validator, "max_nfev_per_start=70 if full else 22",
        note="Selected Triple overrides: A07 >=100; F20/S06/F28 >=80; exact override source is exported.")
    add("validation_root_tolerance", {"smoke": .0025, "full": .001}, "scaled total-force residual", validator, "numerical_root_tolerance=1e-3 if full else 2.5e-3")
    add("validation_jacobian_step", .1, "particle radius", validator, "jacobian_step_a=.10")
    add("validation_fit_shells", [.10, .15, .20], "wavelength", validator, "fit_shell_wavelengths=(.10, .15, .20)")
    add("validation_quadrature", {"fit_n_mu": 12, "fit_n_phi": 24, "surface_n_mu": 20, "surface_n_phi": 40}, "nodes", validator, "fit_n_mu=12, fit_n_phi=24")
    add("validation_root_starts", {"smoke": "common_cartesian_root_starts((1.,))", "full": "common_cartesian_root_starts()"}, "radius-normalized offsets", validator, "starts = common_cartesian_root_starts()",
        note="Exact start-builder expression and source are included; normalized internal search settings are not plotted displacements.")
    generation_parameters = project / "data/unified_generation_parameters.json"
    canonical_generation = None
    if generation_parameters.exists():
        canonical_generation = json.loads(generation_parameters.read_text())
        source(generation_parameters, "resolved native per-mode generation protocol")
        shutil.copy2(generation_parameters, output / "G_resolved_generation_parameters.json")
        companion = generation_parameters.with_suffix(".csv")
        if companion.exists():
            shutil.copy2(companion, output / "G_resolved_generation_parameters.csv")
        def flatten(value, prefix=""):
            for key, item in value.items():
                name = prefix + "." + key if prefix else key
                if isinstance(item, dict): yield from flatten(item, name)
                else: yield name, item
        for name, value in flatten({key: value for key, value in canonical_generation.items()
                                    if key != "selected_triple_recipes"}):
            records.append(dict(group="Resolved generality protocol", section="G", parameter=name,
                value=json.dumps(value) if isinstance(value, list) else value,
                units="as declared in parameter name/source", applicability="full" if name.startswith("modes.full.") else "smoke" if name.startswith("modes.smoke.") else "both",
                source_file="revision_f1_f2/data/unified_generation_parameters.json",
                source_function=name, source_line="", note="Exact resolved source record; no inferred default.",
                evidence_role="declared generation protocol; not a run outcome"))
    merged = pd.concat([parameters, pd.DataFrame(records)], ignore_index=True)
    merged.to_csv(output / "Unified_parameters.csv", index=False)
    merged.to_json(output / "Unified_parameters.json", orient="records", indent=2)

    settings_path = project / "campaign/settings.json"
    settings = json.loads(settings_path.read_text())
    source(settings_path, "declared geometry inputs")
    pd.json_normalize(settings).to_csv(output / "G_geometry_settings.csv", index=False)
    _json(output / "G_geometry_settings.json", settings)
    weights_path = project.parent / "tables/selected_weights.csv"
    weights = pd.read_csv(weights_path)
    source(weights_path, "frozen selected coefficients; source file also contains historical selection outcomes")
    shutil.copy2(weights_path, saved / "G_selected_weights_original.csv")
    fields = [name for name in weights.columns if name in {
        "setting_id", "formulation", "geometry", "frequency_hz", "source_count", "method", "task",
        "alpha_unit", "curvature_weight_unit", "beta_pressure_unit", "gamma_uniformity_unit",
        "alpha_per_m", "curvature_weight_x", "curvature_weight_y", "curvature_weight_z",
        "beta_pressure", "gamma_uniformity", "force_target_x_n", "force_target_y_n", "force_target_z_n",
        "residual_direction_weights", "selection_rule", "tuning_seed", "selected_candidate_id", "selection_scope", "weight_source"}]
    coefficients = weights[fields].copy()
    coefficients["evidence_role"] = "fixed selected input; no retuning in unified run"
    coefficients.to_csv(output / "G_selected_coefficients.csv", index=False)
    coefficients.to_json(output / "G_selected_coefficients.json", orient="records", indent=2)

    manifest_path = project / "data/recorded_proposals/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source(manifest_path, "fixed replay inputs plus separately identified archived outcomes")
    shutil.copy2(manifest_path, saved / "G_recorded_proposals_original.json")
    input_names = ("setting_id", "seed", "stage", "input_file", "input_file_content_id", "maxiter", "gtol",
                   "initial_phase_content_id", "proposal_initial_phase_content_id", "arm", "rms_kick_rad", "direction_seed", "replay_cap_policy")
    proposals = pd.DataFrame([{key: row.get(key) for key in input_names} for row in manifest["trials"]])
    proposals["evidence_role"] = "recorded proposal input; acceptance is reevaluated by the native objective"
    proposals.to_csv(output / "G_recorded_proposal_inputs.csv", index=False)
    pd.DataFrame(manifest["cases"])[["setting_id", "seed", "cold_maxiter", "gtol", "cold_initialization", "stage_input_files"]].to_csv(output / "G_single_start_protocol.csv", index=False)

    baseline_path = project / "data/triple_benchmark/summary.csv"
    resource_path = project / "data/equilibrium_resource_check/release/resource_summary.csv"
    baseline, resource = pd.read_csv(baseline_path), pd.read_csv(resource_path)
    source(baseline_path, "selected recipe identifiers read from historical outcomes")
    source(resource_path, "accepted resource recipe identifiers read from historical outcomes")
    parsed = ast.parse(triple.read_text())
    resource_ids = next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "RESOURCE_IDS" for t in n.targets))
    recipes = pd.concat([baseline[~baseline.setting_id.isin(resource_ids)], resource], ignore_index=True)
    recipes = recipes[["setting_id", "method", "candidate_id", "alpha_force_per_m"]].copy()
    if canonical_generation is not None:
        expected = pd.DataFrame(canonical_generation["selected_triple_recipes"])
        keys = ["setting_id", "method"]
        pd.testing.assert_frame_equal(recipes.sort_values(keys).reset_index(drop=True),
                                      expected[recipes.columns].sort_values(keys).reset_index(drop=True),
                                      check_dtype=False)
    recipes["recipe_kind"] = recipes.candidate_id.astype(str).map(lambda s: "perturbed_restart" if "perturbed_restart" in s else "continuation" if "continuation" in s else "warm" if "warm" in s.lower() else "cold")
    recipes["declared_iteration_cap"] = recipes.apply(lambda r: 200 if r.method == "IB" else
        (20000 if r.setting_id == "F20" else 1000 if r.setting_id == "S06" else 100) if r.method == "GS" else
        30000 if r.recipe_kind == "cold" else 4000, axis=1)
    recipes["cap_scope"] = "selected final operation; cold prerequisites use their own 30000 cap; actual stops are outcomes"
    recipes["evidence_role"] = "fixed selected input"
    recipes.to_csv(output / "G_triple_selected_recipes.csv", index=False)
    default_linear_source = physics / "final_figures.py"
    _source_record(default_linear_source, '"ib_maxiter": 200')
    _source_record(default_linear_source, '"gs_iterations": 100')
    source(default_linear_source, "native IB200/GS100 defaults; explicit GS overrides in unified Triple generator")
    expressions = (_functions(single, {"generate_single_data"}) + _functions(triple, {"selected_recipes", "resource_geometry", "_problem", "_commands", "_validate", "generate_triple_data"})
                   + _functions(validator, {"finite_ka_protocol"}))
    _json(output / "G_generation_protocol_source_expressions.json", expressions)
    selection_path = project / "data/decision_reveal/selection.json"
    if selection_path.exists():
        source(selection_path, "fixed selected viewing windows; not optimization outcomes")
        shutil.copy2(selection_path, output / "G_decision_window_selection.json")

    versions = []
    for package in ("numpy", "scipy", "pandas", "matplotlib", "scikit-image", "scikit-learn", "jax", "jaxlib", "pypdf", "reportlab", "threadpoolctl"):
        try: version = metadata.version(package)
        except metadata.PackageNotFoundError: version = "not installed in exporting interpreter"
        versions.append(dict(package=package, version=version, evidence_role="observed export environment"))
    pd.DataFrame(versions).to_csv(output / "Unified_software_versions.csv", index=False)
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        environment={key: os.environ.get(key, "not set") for key in
                     ("PYTHONHASHSEED", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "JAX_ENABLE_X64", "JAX_PLATFORMS", "GFE_JAX_PLATFORM")},
        generality_child_declared={"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        distinction="Observed exporter environment and declared child-process environment are separate records; Python hash seeding takes effect at interpreter startup.")
    _json(output / "Unified_runtime_environment.json", environment)
    pd.DataFrame(source_rows).drop_duplicates("path").to_csv(output / "Unified_parameter_source_hashes.csv", index=False)
    _json(output / "Unified_parameter_source_hashes.json", source_rows)
    (output / "README.md").write_text(
        "# Unified parameters and reproducibility\n\n"
        "31 figures: main 1–8, A1–A2, B1–B3, C1–C3, D1–D3, E1–E2, F1 and G1–G9. "
        "C is sensitivity; F is optimizer robustness; G is array/frequency generality.\n\n"
        "Unified_parameters retains original main/A–F source records and appends explicit G generation settings. "
        "Selected coefficients, start seeds, recorded proposal inputs and Triple recipes are fixed inputs. "
        "The source_records folder preserves original selection files, whose archived outcomes are not new run results. "
        "Generated solver/validation outcomes remain in their own generation manifests and output tables. "
        "Per-operation iteration ceilings are not claims about actual stops or total prerequisite work. "
        "Software and environment records describe this exporter; source hashes identify the exact inspected files.\n")
    result = dict(mode=mode, figure_count=31, generality='ACTIVE: Appendix G1-G9',
        parameter_rows=len(merged), geometry_count=len(settings),
        selected_coefficient_rows=len(coefficients), recorded_proposal_rows=len(proposals),
        selected_triple_recipe_rows=len(recipes), output_root=str(output),
        exported_utc=datetime.now(timezone.utc).isoformat(), scientific_runs_started=0)
    _json(output / "Unified_parameter_manifest.json", result)
    return result
