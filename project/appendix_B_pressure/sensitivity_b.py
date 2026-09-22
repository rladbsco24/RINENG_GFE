"""Appendix C directional-weight and RH pressure-weight sensitivity.

Commands use the current method-specific solver policy. Missing or superseded
FE/RH commands are regenerated before rendering; compatible baseline runs remain reusable.
The legacy Appendix B cache location is retained; current figures are C4/C5.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from rineng_content_id import content_identity
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime/src"))
from hat_revision_pipeline.gorkov_core import (
    ArrayGeometry, Condition, MethodSpec, SingleTargetObjective, gauge_full,
    pressure_field, SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
)
from hat_revision_pipeline.sota import solve_corrected_gorkov_fe

ETA_VALUES = (.001, .01, .1)
BETA_FACTORS = (0., .1, 1., 10.)
SIGNS = ("corrected_minus", "historical_plus")
METHODS = ("FE", "RH", "Conventional")
NOMINAL_BETA = 5e-5 * SOURCE_SCALE_PA_M_PER_MURATA_UNIT
VERSION = "appendix-B-pressure-sensitivity-v1"
ANALYSIS_VERSION = "appendix-B-frozen-FE-reference-v2"
PRESENTATION_VERSION = "appendix-C-relevant-sign-morphology-v3"
C4_CASES = (("corrected_minus", "Twin"), ("historical_plus", "Bottle"))
SOURCE_CONTENT_IDS = {name: content_identity((ROOT / "runtime/src/hat_revision_pipeline" / name).read_bytes()).hexdigest()
                 for name in ("gorkov_core.py", "sota.py")}
STYLE = {"font.family": "DejaVu Sans", "font.size": 13,
         "axes.labelsize": 14, "axes.labelweight": "semibold", "axes.linewidth": 1.7,
         "xtick.labelsize": 12, "ytick.labelsize": 12,
         "xtick.major.width": 1.5, "ytick.major.width": 1.5,
         "legend.fontsize": 12, "pdf.fonttype": 42, "svg.fonttype": "none"}
COLORS = {"FE": "#C78D25", "RH": "#87569B", "Conventional": "#307FA1"}
MARKERS = {"FE": "s", "RH": "^", "Conventional": "o"}


def _source_ids(method):
    from hat_revision_pipeline.fe_solver_policy import compatible_source_ids
    return compatible_source_ids(SOURCE_CONTENT_IDS) if method == 'Conventional' else SOURCE_CONTENT_IDS


def configuration(shape, sign, method, eta=.01, beta_factor=1., *, seed=260905):
    """Hold trace(W) fixed while changing only the directional ratio eta."""
    direction = np.array([1., eta, eta] if shape == "Twin" else [eta, eta, 1.])
    trace = 2010. if method == "Conventional" else (1.02 if shape == "Twin" and method == "FE" else 3.)
    alpha = 0. if method == "Conventional" else (1000. if shape == "Twin" and method == "FE" else 9.)
    beta = (1. if method == "Conventional" else NOMINAL_BETA * beta_factor if method == "RH" else 0.)
    config = dict(version=VERSION, requested=shape, sign=sign, method=method,
                eta=float(eta), beta_factor=float(beta_factor) if method == "RH" else None,
                source_content_id=_source_ids(method), target_m=[0., 0., .05], seed=int(seed),
                frequency_hz=40000., side=16, pitch_m=.01,
                alpha_per_m=alpha, beta_curvature_per_pa=beta,
                curvature_weights=(trace * direction / direction.sum()).tolist(),
                curvature_trace=trace, pressure_mode="abs" if method == "Conventional" or (shape == "Twin" and method == "FE") else "smooth_abs",
                smooth_pressure_relative=.001, stencil_spacing_m=.0005,
                solver="BFGS" if method == "Conventional" else "L-BFGS-B", maxiter=10000, gtol=1e-8,
                initial_phase_protocol=f"Shared reduced random phase vector, seed {seed}; each configuration solved independently; no continuation",
                status="SMOKE")
    if method != "Conventional":
        from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
        config.update(fe_solver_policy=FE_SOLVER_REVISION, evaluator_backend="compact-single-float64-v1")
    return config


def configurations():
    configs = [configuration(shape, sign, method, eta)
               for sign in SIGNS for shape in ("Twin", "Bottle")
               for method in METHODS for eta in ETA_VALUES]
    # beta=0 is the identical Bottle FE objective; beta=1 is already in B3.
    configs += [configuration("Bottle", sign, "RH", .01, factor)
                for sign in SIGNS for factor in (.1, 10.)]
    return configs


def cache_key(config):
    return content_identity(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]


def load_run(config, root=ROOT):
    key = cache_key(config)
    base = Path(root) / "data/sensitivity_cache" / key
    row = json.loads(base.with_suffix(".json").read_text())
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if content_identity(arrays["terminal_phase_rad"].tobytes()).hexdigest() != row["phase_content_id"]:
        raise ValueError(f"Phase content_id mismatch: {key}")
    initial = gauge_full(np.random.default_rng(config["seed"]).uniform(-np.pi, np.pi, 255))
    if not np.array_equal(arrays["initial_phase_rad"], initial):
        raise ValueError(f"Initial command mismatch: {key}")
    return row, arrays


def compatible(config, row):
    """Require the same effective objective, acoustics, initialization and solver."""
    if config["method"] != "Conventional" and (
            row.get("fe_solver_policy") != config.get("fe_solver_policy") or
            row.get("evaluator_backend") != config.get("evaluator_backend")):
        return False
    scalar = ("seed", "frequency_hz", "side", "pitch_m", "alpha_per_m",
              "beta_curvature_per_pa", "smooth_pressure_relative", "stencil_spacing_m", "maxiter", "gtol")
    strings = ("method", "sign", "pressure_mode", "solver")
    try:
        if any(row[k] != config[k] for k in strings):
            return False
        if any(not np.isclose(row[k], config[k], rtol=1e-13, atol=0.) for k in scalar):
            return False
        if not np.allclose(row["target_m"], config["target_m"], rtol=0, atol=1e-12):
            return False
        if not np.allclose(row["curvature_weights"], config["curvature_weights"], rtol=1e-13, atol=0.):
            return False
        sources = row.get("source_content_id")
        if isinstance(sources, dict):
            return sources == config["source_content_id"]
        return sources == config["source_content_id"]["gorkov_core.py"]
    except (KeyError, TypeError, ValueError):
        return False


def save_run(config, record, arrays, root):
    key = cache_key(config)
    directory = Path(root) / "data/sensitivity_cache"
    directory.mkdir(exist_ok=True, parents=True)
    base = directory / key
    record = dict(record, **config, cache_key=key)
    temporary = base.with_suffix(".writing.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(base.with_suffix(".npz"))
    temporary_json = base.with_suffix(".writing.json")
    temporary_json.write_text(json.dumps(record, indent=2))
    temporary_json.replace(base.with_suffix(".json"))
    return record


def reuse_compatible(root=ROOT, source_dirs=()):
    """Import compatible deposited runs without changing their phase commands."""
    root = Path(root)
    candidates = []
    for directory in [root / "data/cache", *map(Path, source_dirs)]:
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                if path.with_suffix(".npz").is_file():
                    try:
                        candidates.append((path, json.loads(path.read_text())))
                    except (ValueError, OSError):
                        continue
    reused = 0
    for config in configurations():
        base = root / "data/sensitivity_cache" / cache_key(config)
        if base.with_suffix(".json").is_file() and base.with_suffix(".npz").is_file():
            load_run(config, root)
            continue
        for path, row in candidates:
            if not compatible(config, row):
                continue
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            if content_identity(arrays["terminal_phase_rad"].tobytes()).hexdigest() != row["phase_content_id"]:
                raise ValueError(f"Invalid imported phase: {path}")
            save_run(config, dict(row, cache_origin="reused_compatible_" + config["solver"],
                     source_cache_key=row["cache_key"], source_cache_filename=path.name,
                     source_cache_version=row.get("version")), arrays, root)
            load_run(config, root)
            reused += 1
            break
    return reused


def objective(config):
    geometry = ArrayGeometry.square(config["side"], config["pitch_m"])
    weights = np.diag(config["curvature_weights"])
    method = MethodSpec(config["method"], config["alpha_per_m"],
                        config["beta_curvature_per_pa"], config["pressure_mode"], weights)
    obj = SingleTargetObjective(Condition("Appendix B pressure sensitivity", geometry.positions_m,
                tuple(config["target_m"]), config["frequency_hz"], weights), method,
                stencil_spacing_m=config["stencil_spacing_m"],
                smooth_pressure_relative=config["smooth_pressure_relative"])
    if config["sign"] == "historical_plus":
        obj.coefficients = replace(obj.coefficients, gradient_j_m2_pa2=-obj.coefficients.gradient_j_m2_pa2)
    return obj


def solve_one(config, root=ROOT):
    with threadpool_limits(limits=1):
        obj = objective(config)
        initial = gauge_full(np.random.default_rng(config["seed"]).uniform(-np.pi, np.pi, obj.n_transducers-1))
        start = time.perf_counter()
        result = solve_corrected_gorkov_fe(obj, initial, maxiter=config["maxiter"], gtol=config["gtol"])
        elapsed = time.perf_counter() - start
        raw = obj._raw_metrics(result.phase_rad, False)
        budgets = np.array([0, 100, 500, 1000, 10000])
        accepted = np.minimum(budgets, result.iterations)
        arrays = dict(initial_phase_rad=initial, terminal_phase_rad=result.phase_rad,
                      checkpoint_budgets=budgets, checkpoint_accepted_iterations=accepted,
                      checkpoint_phase_rad=result.history_phase_rad[accepted],
                      history_objective=result.history_objective,
                      history_gradient_l2=result.history_gradient_norm,
                      history_wall_s=result.history_wall_s)
        row = dict(iterations=int(result.iterations), gradient_l2=float(result.gradient_norm),
                   success=bool(result.success), message=str(result.message), status_code=int(result.status),
                   objective=float(result.objective), pressure_pa=float(raw["pressure_abs_pa"]),
                   gorkov_force_n=float(raw["force_norm_n"]), command_time_s=elapsed,
                   phase_content_id=content_identity(result.phase_rad.tobytes()).hexdigest(),
                   initial_content_id=content_identity(initial.tobytes()).hexdigest(),
                   cache_origin="computed_" + result.optimizer)
        return save_run(config, row, arrays, root)


def compute_missing(root=ROOT, workers=4, source_dirs=()):
    """Regenerate missing or superseded commands without repeating compatible runs."""
    root = Path(root).resolve()
    imported = reuse_compatible(root, source_dirs)
    missing = []
    for config in configurations():
        path = root / "data/sensitivity_cache" / cache_key(config)
        if not path.with_suffix(".json").is_file() or not path.with_suffix(".npz").is_file():
            missing.append(config)
    print(f"Imported {imported} compatible solver caches; {len(missing)} missing solves.", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(solve_one, config, root): config for config in missing}
        for future in as_completed(futures):
            row = future.result()
            print(f"{row['sign']} {row['requested']} {row['method']} eta={row['eta']:g} "
                  f"beta_factor={row['beta_factor']}: {row['iterations']} iterations, "
                  f"status={row['status_code']}", flush=True)
    return pd.DataFrame([load_run(c, root)[0] for c in configurations()])


def fields_from_cache(configs, root):
    axis = np.linspace(-1.25 * 343. / 40000., 1.25 * 343. / 40000., 181)
    xx, zz = np.meshgrid(axis, axis, indexing="xy")
    points = np.stack([xx, np.zeros_like(xx), zz + .05], axis=-1).reshape(-1, 3)
    geometry = ArrayGeometry.square(16, .01)
    cache_path = root / "data/sensitivity_pressure_slices.npz"
    provenance_path = root / "data/sensitivity_pressure_provenance.json"
    fields, previous = {}, {}
    if cache_path.is_file() and provenance_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            fields = {name: archive[name].copy() for name in archive.files}
        previous = json.loads(provenance_path.read_text())
        if "axis_m" not in fields or not np.array_equal(fields["axis_m"], axis):
            fields, previous = {}, {}
    fields["axis_m"] = axis
    metadata, rows = {}, {}
    for config in configs:
        row, arrays = load_run(config, root)
        key = cache_key(config)
        stamp = dict(phase_content_id=row["phase_content_id"], model_content_id=SOURCE_CONTENT_IDS["gorkov_core.py"],
                     plane="XZ", grid_points=181, half_width_m=float(axis[-1]),
                     target_m=[0., 0., .05], frequency_hz=40000.)
        if key not in fields or previous.get(key) != stamp:
            fields[key] = np.concatenate([pressure_field(arrays["terminal_phase_rad"], points[j:j+1024],
                         geometry, 40000.) for j in range(0, len(points), 1024)]).reshape(xx.shape)
        metadata[key], rows[key] = stamp, row
    np.savez_compressed(cache_path, **fields)
    provenance_path.write_text(json.dumps(metadata, indent=2))
    return fields, rows


def cosine(a, b):
    a, b = np.abs(a).ravel(), np.abs(b).ravel()
    return float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1))


def command_similarity(phase, reference_phase):
    """Match main Fig. 2's normalized, gauge-invariant projective overlap.

    The input is the full phase-only transducer command, not an embedding.
    A constant phase shift therefore leaves this metric unchanged.
    """
    command = np.exp(1j * np.asarray(phase, dtype=float).ravel())
    reference = np.exp(1j * np.asarray(reference_phase, dtype=float).ravel())
    if command.shape != reference.shape or command.size == 0:
        raise ValueError("Phase commands must have the same nonzero length")
    return float(np.clip(abs(np.vdot(command, reference)) /
                         (np.linalg.norm(command) * np.linalg.norm(reference)), 0, 1))


def fixed_fe_metrics(config, fields, rows, commands):
    """Use one deposited nominal FE reference for every method in each panel."""
    key = cache_key(config)
    reference_key = cache_key(configuration(config["requested"], config["sign"], "FE", .01))
    overlap = command_similarity(commands[key], commands[reference_key])
    return dict(reference_method="FE", reference_eta=.01,
                fixed_fe_reference_cache_key=reference_key,
                fixed_fe_reference_phase_content_id=rows[reference_key]["phase_content_id"],
                phase_command_similarity_to_fixed_fe=overlap,
                phase_projector_distance_to_fixed_fe=float(np.sqrt(max(0., 1.-overlap**2))),
                xz_amplitude_cosine_to_fixed_fe=cosine(fields[key], fields[reference_key]))


def export(fig, name, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(root / f"{name}.{extension}", dpi=260, facecolor="white")
    plt.close(fig)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=5.5, width=1.5)
    ax.grid(axis="y", color=".88", lw=.8)
    ax.set_axisbelow(True)


def render_b3(table, root):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.11, right=.975, bottom=.25, top=.85, wspace=.20)
    minimum = float(table.xz_amplitude_cosine_to_fixed_fe.min())
    lower = max(0., np.floor((minimum-.015)*20)/20)
    for index, (ax, (sign, shape)) in enumerate(zip(axes, C4_CASES)):
        for method in METHODS:
            group = table[(table.sign == sign) & (table.requested == shape) & (table.method == method)].sort_values("eta")
            ax.plot(group.eta, group.xz_amplitude_cosine_to_fixed_fe, color=COLORS[method],
                    marker=MARKERS[method], mfc="white", mew=1.9, ms=7.0, lw=2.3, label=method)
        ax.set_xscale("log")
        ax.set_xticks(ETA_VALUES, [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
        ax.minorticks_off(); ax.set_ylim(lower, 1.012)
        tidy(ax)
        ax.text(-.08, 1.04, f"({chr(97+index)})", transform=ax.transAxes, fontsize=14, weight="bold")
        sign_label = "Standard sign" if sign == "corrected_minus" else "Alternative sign"
        ax.set_title(f"{sign_label} · {shape}", fontsize=14, weight="semibold", pad=15)
        ax.set_xlabel(r"Directional ratio $\eta$", labelpad=7)
    axes[0].set_ylabel("XZ field similarity to fixed FE", labelpad=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.55, .025),
               ncol=3, frameon=False, handlelength=2.2, columnspacing=1.8)
    export(fig, "Figure_C4_Twin_Bottle_directional_sensitivity", root)


def render_b4(table, root):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.24, top=.82, wspace=.36)
    fig.text(.54, .95, "Alternative sign · Bottle", ha="center", fontsize=13,
             weight="semibold")
    group = table[table.sign.eq("historical_plus") & table.requested.eq("Bottle")].sort_values("beta_factor")
    for ax, column in zip(axes, ("target_pressure_over_shared_sweep_peak", "xz_amplitude_cosine_to_nominal_conventional")):
        ax.plot(range(4), group[column], color=COLORS["RH"], marker=MARKERS["RH"],
                mfc="white", mew=1.9, ms=7.0, lw=2.3)
    for i, ax in enumerate(axes):
        tidy(ax); ax.set_xticks(range(4), ["0\n(FE)", "0.1", "1", "10"])
        ax.set_xlabel(r"RH pressure weight $\beta/\beta_0$", labelpad=7)
        ax.text(-.12, 1.08, f"({chr(97+i)})", transform=ax.transAxes, fontsize=14, weight="bold")
    axes[0].set_ylabel(r"Target $|p|/p_{\max}$", labelpad=8)
    axes[1].set_ylabel("XZ similarity to Conventional", labelpad=8)
    axes[0].set_ylim(-.01, max(.1, float(table.target_pressure_over_shared_sweep_peak.max())*1.08))
    minimum = float(table.xz_amplitude_cosine_to_nominal_conventional.min())
    axes[1].set_ylim(max(0., np.floor((minimum-.015)*20)/20), 1.012)
    export(fig, "Figure_C5_RH_pressure_sensitivity", root)


def _write_current_text(output, directional, pressure):
    """Export current C4/C5 captions and measured interpretation, without solves."""
    captions = {
        "Figure_C4_Twin_Bottle_directional_sensitivity":
        "Directional-weight dependence relative to a fixed nominal FE field. "
        "This tests whether the Twin/Bottle pressure patterns remain close to the same FE reference as the directional curvature ratio changes. "
        "(a) Standard-sign Twin; (b) alternative-sign Bottle. The other sign/morphology combinations are omitted from this figure. "
        "Twin uses W proportional to (1,eta,eta), and Bottle uses (eta,eta,1), at eta=0.001,0.01,0.1 while retaining each method's nominal trace(W). "
        "All methods use one frozen same-morphology/same-sign FE reference at eta=0.01. "
        "Twin correspondence remains high across the bracket; the alternative-sign Bottle comparisons show larger method-dependent departures. "
        "A departure includes the nominal inter-method difference as well as the response to changing eta, so it is not a pure within-method variation. "
        "The tables also give gauge-invariant command cosine and projector distance to that fixed FE command. "
        "These are formulation references, not mechanically selected optimal Bottle fields. One paired start, seed260905; independent solves with a 10000-iteration cap (FE/RH: compact L-BFGS-B; Conventional: BFGS) and native gtol=1e-8. SMOKE.",
        "Figure_C5_RH_pressure_sensitivity":
        "The role of the RH pressure weight in the tested Bottle formulation. "
        "This varies only beta/beta0=0,0.1,1,10 at alpha=9m^-1, trace(W)=3 and eta=0.01; beta0=4.242640687e-6 N m^-1 Pa^-1. "
        "The beta=0 endpoint is the matched FE objective. "
        "Only the alternative-sign Bottle sweep is shown, matching the Bottle-specific role of RH in Appendix B. "
        "(a) Target pressure normalized by one shared xz maximum over the four alternative-sign weight cases. "
        "(b) Xz pressure-amplitude cosine to the fixed alternative-sign nominal Conventional Bottle field. "
        "The nominal RH pressure term removes the appreciable central pressure left by beta=0 and raises correspondence to Conventional. "
        "All settings share the same initial command and retain their actual terminal states; the tables also export fixed-FE command/field metrics. "
        "The alternative sign reverses only the velocity-energy term relative to the standard Gor'kov expression. One paired start, seed260905, 10000-iteration cap (FE/RH: compact L-BFGS-B; Conventional: BFGS), gtol=1e-8. SMOKE."
    }
    (output / "C4_C5_figure_captions.json").write_text(json.dumps(captions, indent=2))
    minima = directional.groupby(["sign", "requested"]).xz_amplitude_cosine_to_fixed_fe.min()
    alternative = pressure[pressure.sign.eq("historical_plus")].set_index("beta_factor")
    lines = [
        "Appendix C4-C5: directional-weight and RH pressure-weight sensitivity.",
        "C4 asks how changing directional curvature ratios moves the Twin/Bottle fields relative to one fixed FE formulation reference. "
        "C5 asks whether the RH pressure term supplies the central-pressure suppression that the alternative-sign FE Bottle formulation lacks.",
        f"C4: minimum xz amplitude cosine over methods and eta is {minima['corrected_minus', 'Twin']:.6f} "
        f"for standard-sign Twin and {minima['historical_plus', 'Bottle']:.6f} for alternative-sign Bottle. "
        "These minima include inter-method differences from the nominal FE reference and are not solely a within-method sensitivity amplitude.",
        f"C5 alternative sign: beta=0 to beta0 changes normalized target pressure from {alternative.loc[0., 'target_pressure_over_shared_sweep_peak']:.6g} "
        f"to {alternative.loc[1., 'target_pressure_over_shared_sweep_peak']:.6g}, and field cosine to Conventional from "
        f"{alternative.loc[0., 'xz_amplitude_cosine_to_nominal_conventional']:.6f} to {alternative.loc[1., 'xz_amplitude_cosine_to_nominal_conventional']:.6f}.",
        "C5 is specifically an alternative-sign Bottle pressure-weight test. No new Twin beta sweep is implied by this figure.",
        "The full historical numerical tables and caches retain their original B3/B4 identifiers. The current reader-facing figures and C4/C5 CSV aliases contain only the displayed cases. No additional optimization is performed during this export."
    ]
    (output / "C4_C5_measured_results.txt").write_text("\n\n".join(lines) + "\n")
    (output / "C4_C5_captions.txt").write_text("\n\n".join(f"{name}. {caption}" for name, caption in captions.items()) + "\n")


def render(root=ROOT, output=None):
    """Render C4/C5 from legacy B caches and return the two summary tables.

    ``root`` identifies the unchanged numerical cache package. Current reader-
    facing figures, captions and CSV aliases are written directly to ``output``.
    """
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root.parent / "appendix_outputs" / "C_GFE"
    output.mkdir(parents=True, exist_ok=True)
    compute_missing(root)
    with threadpool_limits(limits=1), plt.rc_context(STYLE):
        configs = configurations()
        fields, rows = fields_from_cache(configs, root)
        commands = {cache_key(c): load_run(c, root)[1]["terminal_phase_rad"] for c in configs}
        b3 = []
        for config in configs[:36]:
            key = cache_key(config)
            b3.append(dict(rows[key], **fixed_fe_metrics(config, fields, rows, commands),
                           field_peak_pa=float(np.abs(fields[key]).max()),
                           target_pressure_over_field_peak=float(np.abs(fields[key][90, 90])/np.abs(fields[key]).max())))
        b3 = pd.DataFrame(b3)
        b4 = []
        for sign in SIGNS:
            beta_configs = [configuration("Bottle", sign, "FE") if factor == 0
                            else configuration("Bottle", sign, "RH", .01, factor) for factor in BETA_FACTORS]
            peak = max(float(np.abs(fields[cache_key(c)]).max()) for c in beta_configs)
            conventional = fields[cache_key(configuration("Bottle", sign, "Conventional"))]
            for factor, config in zip(BETA_FACTORS, beta_configs):
                key = cache_key(config)
                b4.append(dict(rows[key], beta_factor=factor, sweep="Bottle RH beta", matched_FE=(factor == 0),
                               **fixed_fe_metrics(config, fields, rows, commands),
                               shared_sweep_peak_pa=peak,
                               target_pressure_over_shared_sweep_peak=float(np.abs(fields[key][90, 90])/peak),
                               xz_amplitude_cosine_to_nominal_conventional=cosine(fields[key], conventional)))
        b4 = pd.DataFrame(b4)
        b3.to_csv(root / "B3_directional_sensitivity.csv", index=False)
        b4.to_csv(root / "B4_RH_pressure_sensitivity.csv", index=False)
        c4 = b3[((b3.sign == "corrected_minus") & (b3.requested == "Twin")) |
                ((b3.sign == "historical_plus") & (b3.requested == "Bottle"))].copy()
        c5 = b4[(b4.sign == "historical_plus") & (b4.requested == "Bottle")].copy()
        c4.to_csv(output / "C4_Twin_Bottle_directional_sensitivity.csv", index=False)
        c5.to_csv(output / "C5_RH_pressure_sensitivity.csv", index=False)
        pd.DataFrame(rows.values()).to_csv(root / "data/sensitivity_solver_results.csv", index=False)
        render_b3(c4, output); render_b4(c5, output)
    provenance = dict(status="SMOKE", seed=260905, target_m=[0., 0., .05], frequency_hz=40000.,
                      analysis_version=ANALYSIS_VERSION, presentation_version=PRESENTATION_VERSION,
                      side=16, pitch_m=.01, solver={"FE": "compact L-BFGS-B", "RH": "compact L-BFGS-B", "Conventional": "BFGS"}, maxiter=10000, gtol=1e-8,
                      eta_values=list(ETA_VALUES), beta_factors=list(BETA_FACTORS), nominal_beta=NOMINAL_BETA,
                      xz_grid_points=181, xz_half_width_wavelengths=1.25,
                      beta_zero_reuses_matched_Bottle_FE=True,
                      source_content_ids=SOURCE_CONTENT_IDS, unique_optimization_cases=len(rows),
                      reused_commands=sum(r["cache_origin"].startswith("reused_") for r in rows.values()),
                      computed_commands=sum(r["cache_origin"].startswith("computed_") for r in rows.values()),
                      optimization_rerun_during_render=False,
                      B3_metric="Cosine similarity of XZ pressure magnitude to a fixed FE eta=.01 field for the same requested morphology and sign; the same reference is used for all three methods",
                      command_metric="abs(u^H u_FE)/(norm(u)*norm(u_FE)), u=exp(i*phase); same projective command similarity as main Fig. 2",
                      command_metric_source="hat_revision_pipeline.branch_figures.projective_similarity",
                      projector_distance="sqrt(1 - command_similarity**2)",
                      field_metric="dot(abs(p),abs(p_FE))/(norm(abs(p))*norm(abs(p_FE))) on the common 181x181 XZ grid",
                      frozen_fe_references=[dict(requested=shape, sign=sign,
                          cache_key=cache_key(configuration(shape, sign, "FE")),
                          phase_content_id=rows[cache_key(configuration(shape, sign, "FE"))]["phase_content_id"],
                          eta=.01, selection="Current compact L-BFGS-B nominal FE command, fixed before evaluating sensitivity",
                          independent_mechanical_validation=False)
                          for sign in SIGNS for shape in ("Twin", "Bottle")],
                      B4_pressure_normalization="One shared maximum over the four XZ fields within each sign",
                      B4_similarity="Cosine similarity of XZ pressure magnitude to same-sign nominal Conventional",
                      reference_scope="C4/C5 use nominal FE formulation references, not mechanically validated best solutions. In particular, the alternative-sign Bottle FE reference retains central pressure. C1-C3 use the separate mechanically validated main GFE reference.",
                      interpretation="Directional-weight and pressure-weight sensitivity at one fixed target and initial phase; only FE eta=.01 must have similarity one; terminal states retain actual solver status")
    (root / "sensitivity_provenance.json").write_text(json.dumps(provenance, indent=2))
    provenance.update(current_figures=["C4", "C5"], legacy_figures=["B3", "B4"],
                      legacy_cache_directory=str(root), current_output_directory=str(output),
                      displayed_C4_cases=[dict(sign=sign, requested=shape) for sign, shape in C4_CASES],
                      displayed_C5_cases=[dict(sign="historical_plus", requested="Bottle")],
                      displayed_rows_C4=len(c4), displayed_rows_C5=len(c5),
                      historical_rows_B3=len(b3), historical_rows_B4=len(b4),
                      displayed_case_selection="Standard-sign Twin and alternative-sign Bottle for directional weights; alternative-sign Bottle only for RH pressure weight")
    (output / "C4_C5_sensitivity_provenance.json").write_text(json.dumps(provenance, indent=2))
    _write_current_text(output, c4, c5)
    return c4, c5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-missing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-cache", action="append", default=[])
    args = parser.parse_args()
    if args.compute_missing:
        compute_missing(ROOT, args.workers, args.source_cache)
    tables = render(ROOT)
    print(f"Rendered C4 ({len(tables[0])} cases) and C5 ({len(tables[1])} cases).", flush=True)
