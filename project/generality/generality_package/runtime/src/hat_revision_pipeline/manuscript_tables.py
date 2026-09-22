"""Fast manuscript table exports from saved main-figure results; no simulations.

All numerical results in this revision are SMOKE. This module reads CSVs and
trusted local command caches only. No optimizer or figure producer is called.
"""
from __future__ import annotations

import fingerprintlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MISSING = "Not recorded"
NA = "N/A"
METHODS = ("Conventional", "FE", "FE+g", "IB", "GS", "AD")
CATEGORIES = {"Yes", "No", "Conditional", "Not evaluated", "Not applicable"}
CACHE_KINDS = (
    "fixed_single_fe_endpoint", "fixed_single_fe_gravity_endpoint",
    "final_single_matched_baseline_commands", "final_triple_matched_baseline_commands",
    "single_diff_pat_command", "triple_diff_pat_command",
    "triple_long_fixed_fe", "triple_long_conventional_cumulative",
    "final_fig6_vertical_compensated_fe", "fe_g_alpha_command",
)


def _content_id(path: Path) -> str:
    h = fingerprintlib.fingerprint()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _phase_content_id(value: Any) -> str:
    a = np.ascontiguousarray(value)
    h = fingerprintlib.fingerprint()
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def _number(value: Any) -> float | None:
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return MISSING
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if v == 0:
            return "0"
        if abs(v) < 0.001 or abs(v) >= 10000:
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(value)


def _finite(values: Any) -> np.ndarray:
    a = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return a[np.isfinite(a)]


def _median(values: Any) -> float | None:
    a = _finite(values)
    return float(np.median(a)) if len(a) else None


def _interval(values: Any, bootstrap: bool = False) -> str:
    a = _finite(values)
    if len(a) < 2:
        return NA
    if bootstrap:
        rng = np.random.default_rng(260905)
        boot = np.median(rng.choice(a, size=(2000, len(a)), replace=True), axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
    else:
        lo, hi = np.quantile(a, [0.25, 0.75])
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def _bool(values: Any) -> pd.Series:
    return pd.Series(values).astype(str).str.lower().isin(["true", "1", "yes"])


def _objects(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _objects(child)
    elif any(hasattr(value, key) for key in ("phase_rad", "phases", "terminal_phase_rad")):
        yield value


def _native_records(cache_root: Path, phase_hashes: set[str], sources: dict) -> dict:
    """Read only existing cache records; bind every object by exact phase hash."""
    found = {}
    directories = [cache_root / kind for kind in CACHE_KINDS]
    # Optional selected-alpha implementations can use another narrowly named kind.
    if cache_root.exists():
        directories.extend(p for p in cache_root.iterdir()
                           if p.is_dir() and "alpha" in p.name and "command" in p.name)
    for directory in dict.fromkeys(directories):
        for path in sorted(directory.glob("*.pkl")):
            with path.open("rb") as stream:
                record = pickle.load(stream)
            for obj in _objects(record.get("value")):
                phase = next((getattr(obj, key) for key in
                              ("phase_rad", "phases", "terminal_phase_rad") if hasattr(obj, key)), None)
                if phase is None:
                    continue
                digest = _phase_content_id(phase)
                if digest in phase_hashes:
                    found[digest] = {"object": obj, "payload": record.get("payload", {}),
                                     "cache_path": str(path), "dependencies": record.get("dependencies", {})}
                    sources[str(path)] = {"fingerprint": _content_id(path), "type": "trusted numerical cache"}
    return found


def _solver_native(record: dict) -> tuple[Any, dict]:
    obj = record.get("object")
    return getattr(obj, "solver_result", obj), record.get("payload", {})


def _grad(record: dict) -> float | None:
    if record.get("archived_gradient_norm") is not None:
        return float(record["archived_gradient_norm"])
    obj, _ = _solver_native(record)
    for name in ("gradient_norm", "terminal_gradient_norm"):
        if hasattr(obj, name):
            return _number(getattr(obj, name))
    return None


def _tex_escape(value: Any) -> str:
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(value))


def _latex(frame: pd.DataFrame, caption: str) -> str:
    # Standard tabular/hline needs no packages and keeps all text escaped.
    lines = ["% SMOKE: all numerical rows are preliminary.", "% " + caption,
             r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\hline",
             " & ".join(_tex_escape(c) for c in frame.columns) + r" \\", r"\hline"]
    lines.extend(" & ".join(_tex_escape(v) for v in row) + r" \\" for row in frame.itertuples(index=False, name=None))
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def export_manuscript_tables(table_root, output_root, runtime_root=None, context=None):
    """Return ``{'tables': DataFrames, 'summary': DataFrame, 'manifest': dict}``.

    ``table_root`` is the authoritative selected result directory. ``context``
    may provide ``cache_root`` and ``source_notebook``. The function never
    computes missing physical quantities. IQRs and bootstrap intervals use
    independent command/seed records, never the three targets of one command.
    """
    table_root, output_root = Path(table_root), Path(output_root)
    runtime_root = Path(runtime_root) if runtime_root else Path(__file__).resolve().parents[2]
    context = dict(context or {})
    cache_root = Path(context.get("cache_root", table_root.parent / "cache"))
    output_root.mkdir(parents=True, exist_ok=True)
    sources, input_frames = {}, {}

    def read(name):
        if name in input_frames:
            return input_frames[name]
        path = table_root / f"{name}.csv"
        if not path.exists():
            input_frames[name] = pd.DataFrame()
        else:
            input_frames[name] = pd.read_csv(path)
            sources[str(path)] = {"fingerprint": _content_id(path), "type": "saved notebook table"}
        return input_frames[name]

    main_name = "main_mechanical_results"
    main = read(main_name)
    if main.empty:
        main_name = "fig5_exact_force_field_concentration"
        main = read(main_name)
    if main.empty:
        raise FileNotFoundError("No saved main mechanical benchmark table was found")
    required = {"task", "method", "phase_fingerprint", "command_time_s", "finite_ka_displacement_a"}
    if not required.issubset(main.columns):
        raise ValueError(f"Missing benchmark columns: {sorted(required - set(main.columns))}")
    hashes = set(main.phase_fingerprint.dropna().astype(str))
    native = _native_records(cache_root, hashes, sources)
    archive = runtime_root / "data" / "corrected_branch_evolution"
    trajectory_path = archive / "conventional_trajectories.csv"
    phase_path = archive / "conventional_snapshots.npz"
    config_path = archive / "anchor_config.json"
    if trajectory_path.exists() and phase_path.exists() and config_path.exists():
        trajectories = pd.read_csv(trajectory_path)
        archive_config = json.loads(config_path.read_text())["config"]
        with np.load(phase_path) as saved:
            snapshots = saved["snapshots"]
        for row in trajectories.itertuples(index=False):
            phase = snapshots[int(row.trajectory_index), -1]
            digest = _phase_content_id(phase)
            if digest in hashes and digest not in native:
                # The supplied core defines RMS = ||reduced gradient|| / sqrt(size).
                native[digest] = {"archived_gradient_norm": float(row.phase_gradient_rms) * np.sqrt(phase.size - 1),
                    "payload": {"gtol": archive_config["gtol"], "seed": int(row.seed),
                                "iteration_cap": archive_config["conventional_maxiter"]},
                    "cache_path": str(trajectory_path)}
                for source in (trajectory_path, phase_path, config_path,
                               runtime_root / "data" / "hat_single_trap_core_reference.py"):
                    sources[str(source)] = {"fingerprint": _content_id(source), "type": "deposited command/solver metadata"}
    for name in ("gorkov_core.py", "sota.py", "diff_pat.py", "multitrap.py", "final_figures.py",
                 "fixed_fe_endpoints.py", "triple_long_iteration.py", "exact_validator.py"):
        path = runtime_root / "src" / "hat_revision_pipeline" / name
        if path.exists():
            sources[str(path)] = {"fingerprint": _content_id(path), "type": "implementation source"}
    if context.get("source_notebook"):
        path = Path(context["source_notebook"])
        if path.exists():
            sources[str(path)] = {"fingerprint": _content_id(path), "type": "source notebook, read only"}

    tables, metadata, raw_audit = {}, {}, []

    def add(name, rows, description, source_type="Notebook-derived", variables=""):
        if not len(rows):
            return
        frame = pd.DataFrame(rows)
        frame = frame.map(_fmt)
        tables[name] = frame
        metadata[name] = {"description": description, "source_type": source_type,
                          "source_variables": variables, "row_count": len(frame)}

    distinctions = []
    for method in METHODS:
        if method not in set(main.method):
            continue
        mechanical = method in ("FE", "FE+g")
        note = {
            "Conventional": "Pressure-null penalty plus Gor'kov curvature; no force residual.",
            "FE": "Force residual and restoring curvature use Gor'kov; finite penalty, not a hard constraint.",
            "FE+g": "FE residual targets upward buoyancy-corrected weight; same independent validator.",
            "IB": "Complex eight-point ring prescription; one free phase per ring in Triple.",
            "GS": "Implemented phase-only GS; not a GS-PAT reproduction.",
            "AD": "Diff-PAT-style AD/Adam amplitude-only loss; no prescribed phase or mechanical loss.",
        }[method]
        distinctions.append({"Method": method, "Phase-only control": "Yes",
            "Prescribed-field synthesis": "Yes" if method in ("IB", "GS", "AD") else "No",
            "Direct mechanical objective": "Yes" if mechanical or method == "Conventional" else "No",
            "Coordinate-equilibrium term": "Yes" if mechanical else "No",
            "Off-target control-point prescription": "Yes" if method in ("IB", "GS", "AD") else "No",
            "Independent mechanical validation": "Yes", "Iterative update": "Yes", "Notes": note})
    add("table_sota_method_distinctions", distinctions,
        "Implemented methods only. Categories describe code behavior, not universal capabilities of a method family.",
        variables="gorkov_core.MethodSpec/SingleTargetObjective; multitrap.evaluate_multitrap_objective; sota synthesis functions; diff_pat.diff_pat_phase_only; saved exact benchmark")
    literature = [
        {"Method family": "Conventional", "Implemented alias": "Conventional", "Claim to verify against citation": "Marzo-style pressure-null and Gor'kov-curvature optimization", "Reference status": "VERIFY_REF"},
        {"Method family": "GS-based reconstruction", "Implemented alias": "GS", "Claim to verify against citation": "GS-PAT supports efficient GS-based field reconstruction; this implementation remains phase-only GS", "Reference status": "VERIFY_REF"},
        {"Method family": "Iterative back-projection", "Implemented alias": "IB", "Claim to verify against citation": "Prescribed acoustic-field construction by iterative back-projection", "Reference status": "VERIFY_REF"},
        {"Method family": "Diff-PAT", "Implemented alias": "AD", "Claim to verify against citation": "AD-based acoustic hologram optimization; FE does not introduce automatic differentiation", "Reference status": "VERIFY_REF"},
        {"Method family": "FE / FE+g", "Implemented alias": "FE; FE+g", "Claim to verify against citation": "Objective formulation distinguishes the study; no first-use-of-force claim", "Reference status": "VERIFY_REF"},
    ]
    add("table_literature_positioning", literature,
        "Citation-check queue. All external method-family attributions require author verification; measured quantities are kept separate.",
        source_type="Literature positioning: VERIFY_REF", variables="Implementation aliases; user-specified citation rationale (not externally verified in this export)")

    benchmark, accuracy, settings, timing, objective_settings = [], [], [], [], []
    for case in ("Single", "Triple"):
        for method in METHODS:
            group = main[(main.task == case) & (main.method == method)].copy()
            if group.empty:
                continue
            commands = group.drop_duplicates("phase_fingerprint")
            times = _finite(commands.command_time_s)
            command_displacements = group.groupby("phase_fingerprint", sort=False).finite_ka_displacement_a.median()
            count = len(commands)
            root_count = int(_bool(group.finite_ka_root_found).sum())
            restoring_count = int(_bool(group.finite_ka_locally_restoring).sum()) if "finite_ka_locally_restoring" in group else None
            displacement = _median(group.finite_ka_displacement_a)
            stiffness = _median(group.finite_ka_stiffness_min_n_m)
            pressure = _median(group.pressure_abs_at_exact_equilibrium_pa)
            records = [native.get(str(s), {}) for s in commands.phase_fingerprint]
            grads = [v for r in records if (v := _grad(r)) is not None]
            terminal_gradient = NA if method in ("GS", "IB") else _median(grads)
            gradient_note = "Not applicable to projection update" if method in ("GS", "IB") else "AD amplitude-loss gradient not saved; no mechanical-gradient units apply" if method == "AD" else "Optimizer reduced phase-gradient L2 norm; objective scales differ"
            first = records[0] if records else {}
            obj, payload = _solver_native(first)
            cfg = payload.get("compensated_config", payload.get("config", {}))
            cap = {"Conventional": 10000, "FE": 10000, "FE+g": 10000, "IB": 200, "GS": 100, "AD": 150}[method]
            if payload.get("iteration_cap") is not None:
                cap = int(payload["iteration_cap"])
            elif cfg.get("smooth_stage_maxiters"):
                cap = sum(cfg["smooth_stage_maxiters"])
            benchmark.append({"Method": method, "Case": case, "Commands (n)": count,
                "Targets (n)": len(group), "Runtime median (s)": _median(times),
                "Runtime IQR (s)": _interval(times), "Roots found": f"{root_count}/{len(group)}",
                "Locally restoring": f"{restoring_count}/{len(group)}" if restoring_count is not None else MISSING,
                "Displacement median (a)": displacement,
                "Displacement 95% CI (a)": _interval(command_displacements, bootstrap=True),
                "Minimum symmetric stiffness median (N/m)": stiffness,
                "Equilibrium pressure median (Pa)": pressure,
                "Terminal gradient L2 (N/(m rad))": terminal_gradient,
                "Iteration cap": cap, "Run status": "SMOKE"})
            tradeoff = {
                "Conventional": "Pressure-null/curvature objective; coordinate force balance is not targeted.",
                "FE": "Gor'kov force residual and restoring curvature; exact displacement measured independently.",
                "FE+g": "Gravity-balanced surrogate residual; exact displacement measured independently.",
                "IB": "Prescribed complex-field reconstruction; coordinate force balance is not targeted.",
                "GS": "Prescribed complex-field reconstruction; coordinate force balance is not targeted.",
                "AD": "Pressure-amplitude optimization; coordinate force balance is not targeted.",
            }[method]
            accuracy.append({"Method": method, "Case": case, "Command time median (s)": _median(times),
                "Exact displacement median (a)": displacement, "Observed trade-off": tradeoff, "Run status": "SMOKE"})
            gtol = cfg.get("gtol", payload.get("gtol"))
            if method == "Conventional" and case == "Single":
                stopping = f"Deposited BFGS final-budget state; gtol={_fmt(gtol)}"
            elif method in ("Conventional", "FE", "FE+g"):
                stopping = f"BFGS gtol={_fmt(gtol)}; cap or precision-loss termination"
                if case == "Triple" and method == "Conventional":
                    stopping += "; same-endpoint BFGS restarts to cumulative cap"
            elif method == "IB":
                stopping = "Projective phase change < 0.01 rad or cap"
            else:
                stopping = "Fixed update count"
            init = "Random phase, paired within objective comparison" if method in ("Conventional", "FE", "FE+g") else "Back-projected prescribed complex field" if method in ("IB", "GS") else "JAX uniform random phase; base seed + 181"
            seed = payload.get("seed", payload.get("random_seed", _number(commands.iloc[0].get("seed"))))
            update = "BFGS, analytic phase gradient" if method in ("Conventional", "FE", "FE+g") else "Iterative back-projection" if method == "IB" else "GS recurrence; final phase-only projection" if method == "GS" else "JAX AD + Adam; lr=0.1, beta=(0.9,0.999), epsilon=1e-8"
            settings.append({"Method": method, "Case": case, "Initialization": init,
                "Seed": seed, "Max iterations": cap, "Terminal iterations": _median(commands.terminal_iteration),
                "Stopping criterion": stopping, "Optimizer/update": update,
                "Precision": "float64 / complex128", "Device": "NumPy CPU" if method != "AD" else MISSING,
                "Compile included?": "Yes, inside solve" if method == "AD" else "Not applicable",
                "Transfer included?": "Inside solve; not timed separately" if method == "AD" else "Not applicable",
                "Synchronization": "terminal_phase and loss_history block_until_ready inside solve" if method == "AD" else "Not applicable (synchronous CPU)",
                "Run status": "SMOKE"})
            alpha = cfg.get("alpha_force", payload.get("alpha_per_m"))
            if alpha is None and method in ("FE", "FE+g"):
                if "fixed_fe_alpha_force" in group:
                    alpha = _median(group.fixed_fe_alpha_force)
                if alpha is None and case == "Single":
                    contract = read("single_fixed_endpoint_contract")
                    if not contract.empty:
                        alpha = _number(contract.iloc[0].get("fe_alpha_per_m"))
            weight = cfg.get("curvature_weight_override")
            weight_text = "; ".join(_fmt(v) for v in np.diag(weight)) if weight is not None else "1; 1; 1" if case == "Single" and method in ("FE", "FE+g") else "1000; 1000; 10" if case == "Single" and method == "Conventional" else MISSING
            objective_settings.append({"Method": method, "Case": case,
                "Alpha (1/m)": alpha if method in ("FE", "FE+g") else NA,
                "Curvature diagonal": weight_text if method in ("Conventional", "FE", "FE+g") else NA,
                "Pressure weight (N/(m Pa))": cfg.get("beta_pressure", NA if method in ("GS", "IB", "AD") else 0.0 if case == "Single" and method in ("FE", "FE+g") else 1.0 if case == "Single" and method == "Conventional" else MISSING),
                "Uniformity weight (1)": cfg.get("gamma_uniformity", NA if case == "Single" or method in ("GS", "IB", "AD") else MISSING),
                "Gradient definition": gradient_note, "Run status": "SMOKE"})
            for command, record in zip(commands.to_dict("records"), records):
                native_obj = record.get("object")
                row = {"Method": method, "Case": case, "Initialization/setup (s)": MISSING,
                       "Compile (s)": NA if method != "AD" else MISSING,
                       "Transfer (s)": NA if method != "AD" else MISSING,
                       "Solve (s)": MISSING, "Final evaluation (s)": MISSING,
                       "Synchronization (s)": NA if method != "AD" else MISSING,
                       "Internal end-to-end (s)": MISSING, "Reported command time (s)": command["command_time_s"],
                       "Included scope": "Recorded command/solver wall time; setup decomposition unavailable",
                       "Run status": "SMOKE"}
                breakdown = getattr(native_obj, "timing", None)
                if breakdown is not None:
                    row.update({"Initialization/setup (s)": breakdown.initialization_s,
                                "Solve (s)": breakdown.solve_s, "Internal end-to-end (s)": breakdown.total_s,
                                "Included scope": "Transfer-matrix construction and synthesis; final mechanics excluded"})
                    if method == "AD":
                        row.update({"Compile (s)": "Included in solve; not isolated", "Transfer (s)": "Included in solve; not isolated",
                                    "Synchronization (s)": "Included in solve; not isolated",
                                    "Included scope": "Transfer-matrix setup + JAX initialization/transfer/JIT/Adam/sync; final mechanics excluded"})
                elif native_obj is not None and hasattr(native_obj, "setup_sec"):
                    row.update({"Initialization/setup (s)": native_obj.setup_sec, "Solve (s)": native_obj.solve_sec,
                                "Final evaluation (s)": native_obj.final_evaluation_sec,
                                "Internal end-to-end (s)": native_obj.end_to_end_sec,
                                "Included scope": "Setup + solve + final evaluation; internal end-to-end command time" if np.isclose(float(command["command_time_s"]), native_obj.end_to_end_sec, rtol=1e-10, atol=1e-10) else "Solve interval; setup and final evaluation recorded separately"})
                elif native_obj is not None and hasattr(native_obj, "history_wall_s"):
                    row.update({"Solve (s)": float(native_obj.history_wall_s[-1]),
                                "Included scope": "Initial/terminal objective evaluation + BFGS/history; objective setup excluded"})
                elif native_obj is not None and hasattr(native_obj, "solve_sec"):
                    row.update({"Solve (s)": native_obj.solve_sec,
                                "Included scope": "Cumulative BFGS/history command wall time; no separate setup record"})
                timing.append(row)
                raw_audit.append({"method": method, "case": case, "phase_fingerprint": command["phase_fingerprint"],
                    "source_table": f"{main_name}.csv", "source_cache": record.get("cache_path", MISSING),
                    "command_time_s": float(command["command_time_s"]), "terminal_gradient_l2": _grad(record),
                    "target_count": int((group.phase_fingerprint == command["phase_fingerprint"]).sum()),
                    "is_independent_command": True, "evidence_status": "SMOKE"})
    note = ("Single has one representative command per method; Triple has one command and three target observations. "
            "Triple target medians are descriptive, not replicate estimates. Roots/restoring counts are mechanical classifications, "
            "not a repeated-trial success probability. N/A CIs/IQRs indicate fewer than two independent commands. "
            "Stiffness is the minimum eigenvalue of the symmetric restoring matrix at the finite-ka equilibrium. "
            "Phase-gradient norms belong to different objectives and are not interchangeable accuracy metrics.")
    add("table_quantitative_benchmark", benchmark, note, variables="main_mechanical_results + exact-phase-matched native solver records")
    add("table_accuracy_cost", accuracy, "Reported command-time scopes differ; consult the timing table before comparing costs.", variables="main_mechanical_results.command_time_s, finite_ka_displacement_a")
    add("table_solver_settings", settings, "Actual algorithm settings are preserved. Cached runs do not retain a hardware-model identifier; AD device is Not recorded.", variables="native command payloads; sota.py; diff_pat.py; fixed_fe_endpoints.py; original Conventional archive")
    add("table_objective_settings", objective_settings, "Weights refer to the implementation's native objective definitions; reported gradients use reduced phase coordinates.", variables="native command config payload; single_fixed_endpoint_contract")
    add("table_timing_protocol", timing, "Only measured components are exported. Wrapper hook zeros do not imply zero JAX compilation/transfer/synchronization; those are inside AD solve.", variables="BenchmarkRun.timing; MultitrapRunResult; FESolveResult.history_wall_s; main_mechanical_results.command_time_s")

    convergence = read("fig1_single_vortex_runs")
    if not convergence.empty:
        rows = []
        for method, group in convergence.groupby("method", sort=False):
            group = group.drop_duplicates(["method", "seed", "target_x_m", "target_y_m", "target_z_m"])
            rows.append({"Method": method, "Independent seeds (n)": len(group),
                "Runtime median (s)": _median(group.wall_time_s), "Runtime IQR (s)": _interval(group.wall_time_s),
                "Runtime median 95% CI (s)": _interval(group.wall_time_s, True),
                "Iterations median": _median(group.iterations), "Iterations IQR": _interval(group.iterations),
                "Terminal gradient median (N/(m rad))": _median(group.gradient_norm),
                "Terminal gradient IQR (N/(m rad))": _interval(group.gradient_norm),
                "Near-stationary count": f"{int(_bool(group.near_stationary).sum())}/{len(group)}",
                "Optimizer success count": f"{int(_bool(group.optimizer_success).sum())}/{len(group)}",
                "Iteration cap": int(group.iteration_cap.max()), "Run status": "SMOKE"})
        add("table_single_convergence_repeats", rows,
            "Separate Figure 1 seed population; not the representative mechanical benchmark. Bootstrap: 2000 resamples, seed 260905, percentile CI for the median. Near-stationarity and optimizer success are separate flags.",
            variables="fig1_single_vortex_runs; unrounded independent seed samples")

    scope = []
    for case in ("Single", "Triple"):
        group = main[main.task == case]
        if len(group):
            scope.append({"Test condition": f"Main {case} fixed target configuration",
                **{f"{m} evaluated": "Yes" if m in set(group.method) else "Not evaluated" for m in METHODS},
                "Measured quantity": "Elastic finite-ka equilibrium displacement, pressure, local stiffness",
                "Result available": "Yes", "Run status": "SMOKE"})
    for name, condition, quantity in [
        ("fig1_xz_domain_runs", "Single targets over xz grid", "Command time; terminal gradient"),
        ("fig2_fixed_single_fe_branch_anchors", "Single FE endpoint set over target grid", "Phase commands; endpoint classes"),
        ("fig7_alpha_transition", "Existing Single FE alpha cases", "Endpoint class; local surrogate Hessian")]:
        data = read(name)
        if data.empty:
            continue
        present = set(data["method"].replace({"Force-Equilibrium": "FE"})) if "method" in data else {"FE"}
        scope.append({"Test condition": condition,
            **{f"{m} evaluated": "Yes" if m in present else "Not evaluated" for m in METHODS},
            "Measured quantity": quantity, "Result available": "Yes", "Run status": "SMOKE"})
    add("table_generality_scope", scope,
        "Index of saved main-notebook tests, not evidence of universal geometry/frequency transfer. No new test is performed by this exporter.", variables="main_mechanical_results; fig1_xz_domain_runs; fig2_fixed_single_fe_branch_anchors; fig7_alpha_transition")
    physical = read("physical_parameters")
    if len(physical):
        add("table_physical_parameters", physical[["scope", "parameter", "value", "unit"]].assign(**{"Run status": "SMOKE"}).rename(columns={"scope": "Model/scope", "parameter": "Parameter", "value": "Value", "unit": "Unit"}),
            "Input parameters from the saved notebook; Gor'kov optimization and elastic finite-ka validation remain distinct.", variables="physical_parameters")

    array = read("array_source_and_field_parameters")
    if len(array):
        selected = array[["model", "parameter", "value", "unit"]].copy()
        selected["unit"] = selected["unit"].fillna(NA)
        selected["Run status"] = "SMOKE"
        add("table_array_and_field_parameters", selected.rename(columns={"model": "Model", "parameter": "Parameter", "value": "Value", "unit": "Unit"}),
            "Saved array and acoustic-field metadata; no geometry or calibration is changed.", variables="array_source_and_field_parameters")

    summary_rows, all_text = [], ["MANUSCRIPT TABLE EXPORTS\nALL NUMERICAL RESULTS: SMOKE\n"]
    for name, frame in tables.items():
        info = metadata[name]
        text = f"{name}\n{info['description']}\nSource type: {info['source_type']}\nSource variables: {info['source_variables']}\n\n{frame.to_string(index=False)}\n"
        files = []
        for extension, content in (("csv", frame.to_csv(index=False)), ("tsv", frame.to_csv(index=False, sep="\t")),
                                   ("txt", text), ("tex", _latex(frame, info["description"]))):
            path = output_root / f"{name}.{extension}"
            path.write_text(content, encoding="utf-8")
            files.append(path.name)
        metadata[name]["files"] = files
        all_text.append(text)
        summary_rows.append({"Table": name, "Rows": len(frame), "Source variables": info["source_variables"],
                             "Files": ", ".join(files), "Source type": info["source_type"]})
    summary = pd.DataFrame(summary_rows)
    all_text.extend(["MANUSCRIPT TABLE EXPORT SUMMARY\n", summary.to_string(index=False)])
    (output_root / "all_table_exports.txt").write_text("\n\n".join(all_text) + "\n", encoding="utf-8")
    manifest = {"status": "SMOKE", "simulation_calls": 0, "figure_calls": 0,
                "source_table_root": str(table_root), "cache_root": str(cache_root),
                "sources": sources, "tables": metadata, "benchmark_row_audit": raw_audit,
                "statistical_unit": "one generated phase command; Triple targets are dependent observations",
                "bootstrap": {"replicates": 2000, "seed": 260905, "kind": "percentile CI of independent-seed median"},
                "external_reference_status": "VERIFY_REF"}
    (output_root / "table_sources_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    for frame in tables.values():
        if frame.isna().any().any():
            raise AssertionError("A manuscript table contains an unformatted missing value")
        if frame.map(lambda x: str(x).strip().lower() in {"nan", "inf", "none", "<na>"}).any().any():
            raise AssertionError("A manuscript table contains an unformatted scalar")
    return {"tables": tables, "summary": summary, "manifest": manifest,
            "all_table_exports": output_root / "all_table_exports.txt"}


def display_manuscript_tables(result):
    """Display tables and the requested concise notebook export manifest."""
    from IPython.display import display
    for name, frame in result["tables"].items():
        print(name)
        display(frame)
    print("MANUSCRIPT TABLE EXPORT SUMMARY")
    display(result["summary"])
    print("All numerical results are SMOKE. External method-family attribution: VERIFY_REF.")
