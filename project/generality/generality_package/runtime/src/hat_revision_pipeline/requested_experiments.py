"""Requested exact-force experiments for the RinEng revision appendices.

This module contains only the three experiments explicitly requested after the
main-figure revision:

* matched (0.45 lambda) versus offset (0.60 lambda) IB/GS prescriptions for
  the deposited Single target and the frozen selected Triple target set;
* the exact-force audit along the deposited x=0, z=50 mm lateral target line,
  using the raw FE B1/B2 medoid phase commands and the same two IB/GS
  prescriptions; and
* the exact-force audit of the already frozen 1.00, 0.80, and 0.65 Triple
  contractions, using the existing 1,000-iteration screen contracts.

No success threshold, geometry, objective, solver budget, or physical model is
introduced here.  Every mechanics row is flattened by the existing
correct-sign finite-ka validator so the exported tables retain the complete
force, equilibrium, stiffness, Jacobian, root-search, and model provenance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import pipeline as p
from .cache import digest_array
from .fig4_threepoint_selection import (
    CANDIDATE_CONTRACTIONS,
    SCREEN_CAP,
    SEED_OFFSET,
    _cached_run,
    configs,
    contracted_targets,
    make_problem,
)
from .figure_presentation import outside_panel_labels
from .gorkov_core import (
    AcousticMedium,
    ArrayGeometry,
    SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
    transfer_matrix,
)
from .sota import (
    SynthesisResult,
    VortexControlSpec,
    iterative_back_projection,
    phase_only_signature_projection_audit,
    vortex_control_points,
)
from .style import COLORS, configure_style


MATCHED_RADIUS_LAMBDA = 0.45
OFFSET_RADIUS_LAMBDA = 0.60
CONTROL_POINT_COUNT = 8
LATERAL_Y_M = np.asarray((-0.012, -0.008, -0.004, 0.0, 0.004, 0.008, 0.012))
LATERAL_X_M = 0.0
LATERAL_Z_M = 0.050
CLOSE_TRIPLE_CONTRACTIONS = (1.00, 0.80, 0.65)


def _save(
    ctx: p.PipelineContext,
    fig: mpl.figure.Figure,
    stem: str,
) -> mpl.figure.Figure:
    """Use the existing artifact writer without changing the plotted data."""

    fig.set_constrained_layout(False)
    return p._save_figure(ctx, fig, stem, tight=False)


def _radius_spec(radius_wavelengths: float) -> VortexControlSpec:
    return VortexControlSpec(
        point_count=CONTROL_POINT_COUNT,
        radius_wavelengths=float(radius_wavelengths),
        topological_charge=1,
    )


def _prescribed_transfer_signature(
    ctx: p.PipelineContext,
    targets_m: Sequence[Sequence[float]],
    radius_wavelengths: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack one unchanged eight-control-point vortex prescription per target."""

    targets = np.asarray(targets_m, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_m must have shape (n, 3)")
    spec = _radius_spec(radius_wavelengths)
    wavelength_m = AcousticMedium().sound_speed_m_s / ctx.config.frequency_hz
    geometry = ArrayGeometry(ctx.positions_m, ctx.normals)
    transfers: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    for target in targets:
        points, signature = vortex_control_points(target, wavelength_m, spec=spec)
        transfer = transfer_matrix(
            points,
            geometry,
            ctx.config.frequency_hz,
            source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
        ).reshape(CONTROL_POINT_COUNT, len(ctx.positions_m))
        transfers.append(np.asarray(transfer, dtype=np.complex128))
        signatures.append(np.asarray(signature, dtype=np.complex128))
    return np.vstack(transfers), np.concatenate(signatures)


def _prescribed_command(
    ctx: p.PipelineContext,
    targets_m: Sequence[Sequence[float]],
    method: str,
    radius_wavelengths: float,
) -> SynthesisResult:
    """Return the deterministic phase-only command for one frozen prescription."""

    targets = np.atleast_2d(np.asarray(targets_m, dtype=float))
    method_name = str(method).upper()
    memo_key = (
        "requested_prescribed_command",
        method_name,
        float(radius_wavelengths),
        digest_array(targets),
    )
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    transfer, signature = _prescribed_transfer_signature(
        ctx, targets, radius_wavelengths
    )
    if method_name == "IB":
        result = iterative_back_projection(
            transfer,
            signature,
            maxiter=200,
            tolerance_rad=0.01,
            control_group_size=CONTROL_POINT_COUNT,
        )
    elif method_name == "GS":
        result = phase_only_signature_projection_audit(
            transfer,
            signature,
            iterations=100,
            control_group_size=CONTROL_POINT_COUNT,
        )
    else:
        raise ValueError("method must be IB or GS")
    ctx.memo[memo_key] = result
    return result


def _exact_key(
    method: str,
    phase: np.ndarray,
    target_m: np.ndarray,
    radius_wavelengths: float | None,
) -> str:
    """Make a task-independent key so repeated endpoints reuse exact results."""

    radius = "native" if radius_wavelengths is None else f"r{radius_wavelengths:.2f}"
    return (
        f"requested-{str(method).lower()}-{radius}-"
        f"{digest_array(np.asarray(target_m, dtype=float))[:14]}-"
        f"{digest_array(np.asarray(phase, dtype=float))[:14]}"
    )


def _validate_jobs(
    ctx: p.PipelineContext,
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[Any]]:
    """Run the unchanged exact validator and preserve job order."""

    def evaluate(job: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        phase = np.asarray(job["phase"], dtype=float)
        target = np.asarray(job["target_m"], dtype=float)
        method = str(job["method"])
        radius = job.get("preset_radius_lambda")
        result = p._finite_ka_validation(
            ctx,
            phase,
            target,
            key=_exact_key(method, phase, target, radius),
        )
        row = p._static_validation_row(method, result)
        metadata = {
            key: value
            for key, value in job.items()
            if key not in {"phase", "target_m"}
        }
        row.update(metadata)
        row.update(
            {
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "phase_fingerprint": digest_array(phase),
                "exact_validation_key": _exact_key(method, phase, target, radius),
                "run_mode": ctx.config.mode,
                "manuscript_numerics": bool(ctx.config.full),
                "correct_sign_gorkov": True,
                "exact_validation_model": p.PartialWaveForceEvaluator.model_name,
            }
        )
        return row, result

    count = len(jobs)
    ordered: list[tuple[dict[str, Any], Any] | None] = [None] * count
    workers = max(1, min(count, int(os.environ.get("HAT_EXACT_WORKERS", "4"))))
    if workers == 1:
        for index, job in enumerate(jobs):
            ordered[index] = evaluate(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(evaluate, job): index
                for index, job in enumerate(jobs)
            }
            for future in as_completed(future_to_index):
                ordered[future_to_index[future]] = future.result()
    completed = [item for item in ordered if item is not None]
    if len(completed) != count:
        raise RuntimeError(f"Exact validation produced {len(completed)} of {count} rows")
    return pd.DataFrame([item[0] for item in completed]), [item[1] for item in completed]


def _prescription_jobs(
    ctx: p.PipelineContext,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    cases = (
        ("Single", np.asarray([p.MAIN_TARGET_M], dtype=float)),
        ("Triple", contracted_targets(0.80)),
    )
    for task, targets in cases:
        for radius, role in (
            (MATCHED_RADIUS_LAMBDA, "matched"),
            (OFFSET_RADIUS_LAMBDA, "offset control"),
        ):
            for method in ("IB", "GS"):
                command = _prescribed_command(ctx, targets, method, radius)
                for target_index, target in enumerate(targets):
                    jobs.append(
                        {
                            "method": method,
                            "phase": np.asarray(command.phase_rad, dtype=float),
                            "target_m": np.asarray(target, dtype=float),
                            "task": task,
                            "target_index": int(target_index),
                            "target_id": "Single" if task == "Single" else f"T{target_index + 1}",
                            "preset_role": role,
                            "preset_radius_lambda": float(radius),
                            "preset_point_count_per_target": CONTROL_POINT_COUNT,
                            "preset_target_count": int(len(targets)),
                            "synthesis_iterations": int(command.iterations),
                            "synthesis_converged": command.converged,
                            "synthesis_terminal_projective_change_rad": command.terminal_projective_change_rad,
                            "synthesis_metadata_json": json.dumps(dict(command.metadata), sort_keys=True),
                            "phase_source": (
                                f"{method} phase-only command from one N=8, "
                                f"R={radius:.2f} lambda ring per target with native per-ring gauge"
                            ),
                            "validation_scope": (
                                "one target in the Single field"
                                if task == "Single"
                                else "one target at a time in the shared Triple field; no particle-particle multiple scattering"
                            ),
                        }
                    )
    if len(jobs) != 16:
        raise RuntimeError(f"Prescription sensitivity contract requires 16 endpoints, got {len(jobs)}")
    return jobs


def _prescription_sensitivity_data(ctx: p.PipelineContext) -> pd.DataFrame:
    memo_key = "requested_appendix_b_prescription_sensitivity"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    table, results = _validate_jobs(ctx, _prescription_jobs(ctx))
    table = table.reset_index(drop=True)
    ctx.tables["appendix_prescription_sensitivity"] = table
    ctx.tables["appendix_prescription_sensitivity_exact"] = table
    ctx.tables["appendix_single_triple_prescription_sensitivity"] = table
    ctx.memo[memo_key] = table
    ctx.memo[memo_key + "_results"] = results
    return table


def _unresolved_note(ax: mpl.axes.Axes, table: pd.DataFrame) -> None:
    unresolved = int((~table["finite_ka_root_found"].astype(bool)).sum())
    if unresolved:
        ax.text(
            0.99,
            0.98,
            f"unresolved: {unresolved}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color="0.30",
        )


def appendix_b_prescription_sensitivity(
    ctx: p.PipelineContext,
) -> mpl.figure.Figure:
    """Single and Triple exact displacement for matched and offset prescriptions."""

    configure_style()
    table = _prescription_sensitivity_data(ctx)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    categories = (
        ("IB", MATCHED_RADIUS_LAMBDA),
        ("GS", MATCHED_RADIUS_LAMBDA),
        ("IB", OFFSET_RADIUS_LAMBDA),
        ("GS", OFFSET_RADIUS_LAMBDA),
    )
    labels = tuple(f"{method}\n$R={radius:.2f}\\lambda$" for method, radius in categories)
    method_colors = {"IB": COLORS["IB"], "GS": COLORS["GS-PO audit"]}
    target_markers = {"Single": "o", "T1": "o", "T2": "s", "T3": "^"}
    for ax, task in zip(axes, ("Single", "Triple"), strict=True):
        subset = table[table["task"].eq(task)]
        for x, (method, radius) in enumerate(categories):
            group = subset[
                subset["method"].eq(method)
                & np.isclose(subset["preset_radius_lambda"], radius)
            ]
            valid = group[group["finite_ka_root_found"].astype(bool)]
            for row in valid.itertuples(index=False):
                ax.scatter(
                    x,
                    float(row.finite_ka_displacement_a),
                    marker=target_markers.get(str(row.target_id), "o"),
                    s=58,
                    color=method_colors[method],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
            if len(valid) > 1:
                median = float(valid["finite_ka_displacement_a"].median())
                ax.plot([x - 0.23, x + 0.23], [median, median], color="black", lw=1.25, zorder=5)
        ax.set_xticks(range(len(categories)), labels)
        ax.set_ylabel(r"Exact displacement, $\|\Delta\mathbf{r}\|/a$")
        ax.set_title(task, pad=7.0, fontweight="semibold")
        ax.set_ylim(bottom=0.0)
        ax.set_yscale("linear")
        _unresolved_note(ax, subset)
    outside_panel_labels(axes, ("a", "b"), x=-0.08, y=1.06)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.22, wspace=0.28)
    return _save(ctx, fig, "Appendix_B_prescription_sensitivity")


def _raw_lateral_fe_commands(ctx: p.PipelineContext) -> list[dict[str, Any]]:
    """Resolve the deposited raw FE B1/B2 medoid phases without demodulation."""

    table = ctx.branch_bank.medoid_table.copy()
    selector = (
        np.isclose(table["target_x_m"].to_numpy(float), LATERAL_X_M, rtol=0.0, atol=1.0e-12)
        & np.isclose(table["target_z_m"].to_numpy(float), LATERAL_Z_M, rtol=0.0, atol=1.0e-12)
        & table["component"].isin(("B1", "B2")).to_numpy()
    )
    rows = table.loc[selector].sort_values(["component", "target_y_m"]).reset_index(drop=True)
    if len(rows) != 14:
        raise RuntimeError(f"Deposited lateral FE contract requires 14 medoids, got {len(rows)}")
    expected = np.tile(LATERAL_Y_M, 2)
    if not np.allclose(rows["target_y_m"].to_numpy(float), expected, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Deposited lateral FE target coordinates changed")
    fe_selector = ctx.artifact.anchor_table["method"].eq("Force-Equilibrium").to_numpy()
    raw_phases = np.asarray(ctx.artifact.anchor_phases[fe_selector], dtype=float)
    records: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        source_position = int(row.source_endpoint_position)
        phase = np.asarray(raw_phases[source_position], dtype=float)
        records.append(
            {
                "method": "FE",
                "component": str(row.component),
                "phase": phase,
                "target_m": np.asarray(
                    [row.target_x_m, row.target_y_m, row.target_z_m], dtype=float
                ),
                "source_endpoint_position": source_position,
                "source_state_index": int(row.state_index),
                "source_ix": int(row.ix),
                "source_iy": int(row.iy),
                "source_xi": float(row.xi),
                "source_eta": float(row.eta),
                "source_seed": int(row.seed),
                "source_replicate": int(row.replicate),
                "source_trial_id": str(row.trial_id),
                "source_component_size": int(row.component_size),
                "phase_source": (
                    "deposited raw Force-Equilibrium anchor phase indexed through "
                    "source_endpoint_position in the FE-filtered phase bank"
                ),
            }
        )
    return records


def _lateral_jobs(ctx: p.PipelineContext) -> list[dict[str, Any]]:
    jobs = _raw_lateral_fe_commands(ctx)
    for y_m in LATERAL_Y_M:
        target = np.asarray((LATERAL_X_M, y_m, LATERAL_Z_M), dtype=float)
        for radius, role in (
            (MATCHED_RADIUS_LAMBDA, "matched"),
            (OFFSET_RADIUS_LAMBDA, "offset control"),
        ):
            for method in ("IB", "GS"):
                command = _prescribed_command(ctx, target[None, :], method, radius)
                jobs.append(
                    {
                        "method": method,
                        "component": "prescribed",
                        "phase": np.asarray(command.phase_rad, dtype=float),
                        "target_m": target,
                        "preset_role": role,
                        "preset_radius_lambda": float(radius),
                        "preset_point_count_per_target": CONTROL_POINT_COUNT,
                        "synthesis_iterations": int(command.iterations),
                        "synthesis_converged": command.converged,
                        "synthesis_terminal_projective_change_rad": command.terminal_projective_change_rad,
                        "synthesis_metadata_json": json.dumps(dict(command.metadata), sort_keys=True),
                        "phase_source": (
                            f"{method} phase-only command from N=8, "
                            f"R={radius:.2f} lambda control ring"
                        ),
                    }
                )
    if len(jobs) != 42:
        raise RuntimeError(f"Lateral exact-stress contract requires 42 endpoints, got {len(jobs)}")
    return jobs


def _lateral_target_data(ctx: p.PipelineContext) -> pd.DataFrame:
    memo_key = "requested_appendix_c_lateral_target_stress"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    table, results = _validate_jobs(ctx, _lateral_jobs(ctx))
    table = table.reset_index(drop=True)
    table["target_y_mm"] = 1.0e3 * table["target_y_m"]
    ctx.tables["appendix_lateral_target_exact_stress"] = table
    ctx.memo[memo_key] = table
    ctx.memo[memo_key + "_results"] = results
    return table


def _series_identity(row: pd.Series) -> str:
    if str(row["method"]) == "FE":
        return f"FE {row['component']}"
    return f"{row['method']} {float(row['preset_radius_lambda']):.2f}lambda"


def appendix_c_lateral_target_stress(
    ctx: p.PipelineContext,
) -> mpl.figure.Figure:
    """Exact displacement and target total force along the deposited lateral line."""

    configure_style()
    table = _lateral_target_data(ctx).copy()
    table["series"] = table.apply(_series_identity, axis=1)
    order = (
        "FE B1",
        "FE B2",
        "IB 0.45lambda",
        "GS 0.45lambda",
        "IB 0.60lambda",
        "GS 0.60lambda",
    )
    style = {
        "FE B1": (COLORS["B1"], "o", "-"),
        "FE B2": (COLORS["B2"], "s", "-"),
        "IB 0.45lambda": (COLORS["IB"], "^", "-"),
        "GS 0.45lambda": (COLORS["GS-PO audit"], "v", "-"),
        "IB 0.60lambda": (COLORS["IB"], "^", "--"),
        "GS 0.60lambda": (COLORS["GS-PO audit"], "v", "--"),
    }
    legend_labels = {
        "FE B1": "FE B1",
        "FE B2": "FE B2",
        "IB 0.45lambda": r"IB, $R=0.45\lambda$",
        "GS 0.45lambda": r"GS, $R=0.45\lambda$",
        "IB 0.60lambda": r"IB, $R=0.60\lambda$",
        "GS 0.60lambda": r"GS, $R=0.60\lambda$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.45))
    for series in order:
        group = table[table["series"].eq(series)].sort_values("target_y_mm")
        color, marker, linestyle = style[series]
        resolved = group["finite_ka_root_found"].astype(bool)
        axes[0].plot(
            group.loc[resolved, "target_y_mm"],
            group.loc[resolved, "finite_ka_displacement_a"],
            color=color,
            marker=marker,
            ls=linestyle,
            lw=1.45,
            ms=5.1,
            label=legend_labels[series],
        )
        axes[1].plot(
            group["target_y_mm"],
            1.0e6 * group["finite_ka_target_force_norm_n"],
            color=color,
            marker=marker,
            ls=linestyle,
            lw=1.45,
            ms=5.1,
        )
    axes[0].set_ylabel(r"Exact displacement, $\|\Delta\mathbf{r}\|/a$")
    axes[1].set_ylabel(r"Target total-force norm ($\mu$N)")
    for ax in axes:
        ax.set_xlabel(r"Target coordinate, $y^*$ (mm)")
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlim(float(LATERAL_Y_M[0] * 1.0e3), float(LATERAL_Y_M[-1] * 1.0e3))
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)
    _unresolved_note(axes[0], table)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.015),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    outside_panel_labels(axes, ("a", "b"), x=-0.08, y=1.06)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.87, bottom=0.25, wspace=0.30)
    return _save(ctx, fig, "Appendix_C_lateral_target_stress")


def _triple_prescribed_commands(
    ctx: p.PipelineContext,
    targets: np.ndarray,
) -> dict[str, SynthesisResult]:
    return {
        method: _prescribed_command(
            ctx, targets, method, MATCHED_RADIUS_LAMBDA
        )
        for method in ("IB", "GS")
    }


def _close_triple_jobs(ctx: p.PipelineContext) -> list[dict[str, Any]]:
    candidate_map = {float(scale): candidate_id for candidate_id, scale in CANDIDATE_CONTRACTIONS}
    if set(candidate_map) != set(CLOSE_TRIPLE_CONTRACTIONS):
        raise RuntimeError("Frozen close-Triple contraction set changed")
    seed = int(ctx.config.random_seed + SEED_OFFSET)
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ctx.positions_m))
    conventional_config, fe_config = configs(ctx, SCREEN_CAP, SCREEN_CAP)
    jobs: list[dict[str, Any]] = []
    for contraction in CLOSE_TRIPLE_CONTRACTIONS:
        candidate_id = candidate_map[float(contraction)]
        targets = contracted_targets(contraction)
        problem = make_problem(
            ctx,
            targets,
            0.0,
            0.0,
            case_prefix=f"fig4-screen-{candidate_id}",
        )
        conventional = _cached_run(
            ctx,
            problem,
            conventional_config,
            seed=seed,
            initial=initial,
            display="Conventional",
            candidate_id=candidate_id,
            u=0.0,
            v=0.0,
        )
        fe = _cached_run(
            ctx,
            problem,
            fe_config,
            seed=seed,
            initial=initial,
            display="FE",
            candidate_id=candidate_id,
            u=0.0,
            v=0.0,
        )
        prescribed = _triple_prescribed_commands(ctx, targets)
        phases = {
            "Conventional": np.asarray(conventional.phases, dtype=float),
            "FE": np.asarray(fe.phases, dtype=float),
            "IB": np.asarray(prescribed["IB"].phase_rad, dtype=float),
            "GS": np.asarray(prescribed["GS"].phase_rad, dtype=float),
        }
        pairwise = np.linalg.norm(targets[:, None, :] - targets[None, :, :], axis=2)
        minimum_separation_m = float(pairwise[pairwise > 0.0].min())
        for method, phase in phases.items():
            for target_index, target in enumerate(targets):
                job: dict[str, Any] = {
                    "method": method,
                    "phase": phase,
                    "target_m": np.asarray(target, dtype=float),
                    "target_id": f"T{target_index + 1}",
                    "target_index": int(target_index),
                    "candidate_id": candidate_id,
                    "contraction": float(contraction),
                    "minimum_pair_separation_m": minimum_separation_m,
                    "minimum_pair_separation_mm": 1.0e3 * minimum_separation_m,
                    "screen_iteration_cap": SCREEN_CAP,
                    "screen_seed": seed,
                    "screen_initial_phase_fingerprint": digest_array(initial),
                    "screen_problem_fingerprint": problem.fingerprint,
                    "phase_source": (
                        "existing frozen 1,000-iteration Figure 4 candidate-screen endpoint"
                        if method in {"Conventional", "FE"}
                        else (
                            f"{method} shared Triple phase-only command from one N=8, "
                            f"R={MATCHED_RADIUS_LAMBDA:.2f} lambda ring per target with native per-ring gauge"
                        )
                    ),
                    "validation_scope": "one target at a time in the shared Triple field; no particle-particle multiple scattering",
                }
                if method == "Conventional":
                    job.update(
                        terminal_iterations=int(conventional.iterations),
                        terminal_gradient_norm=float(conventional.terminal_gradient_norm),
                        optimization_config_json=json.dumps(
                            conventional_config.to_payload(), sort_keys=True
                        ),
                    )
                elif method == "FE":
                    job.update(
                        terminal_iterations=int(fe.iterations),
                        terminal_gradient_norm=float(fe.terminal_gradient_norm),
                        optimization_config_json=json.dumps(
                            fe_config.to_payload(), sort_keys=True
                        ),
                    )
                else:
                    command = prescribed[method]
                    job.update(
                        preset_role="matched",
                        preset_radius_lambda=MATCHED_RADIUS_LAMBDA,
                        preset_point_count_per_target=CONTROL_POINT_COUNT,
                        synthesis_iterations=int(command.iterations),
                        synthesis_converged=command.converged,
                        synthesis_terminal_projective_change_rad=command.terminal_projective_change_rad,
                        synthesis_metadata_json=json.dumps(dict(command.metadata), sort_keys=True),
                    )
                jobs.append(job)
    if len(jobs) != 36:
        raise RuntimeError(f"Close-Triple exact contract requires 36 endpoints, got {len(jobs)}")
    return jobs


def _close_triple_data(ctx: p.PipelineContext) -> pd.DataFrame:
    memo_key = "requested_appendix_d_close_triple_stress"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    table, results = _validate_jobs(ctx, _close_triple_jobs(ctx))
    table = table.reset_index(drop=True)
    ctx.tables["appendix_close_triple_exact_scan"] = table
    ctx.memo[memo_key] = table
    ctx.memo[memo_key + "_results"] = results
    return table


def appendix_d_close_triple_stress(
    ctx: p.PipelineContext,
) -> mpl.figure.Figure:
    """Exact Triple mechanics across the three already frozen contractions."""

    configure_style()
    table = _close_triple_data(ctx)
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.45))
    colors = {
        "Conventional": COLORS["Conventional"],
        "FE": COLORS["FE"],
        "IB": COLORS["IB"],
        "GS": COLORS["GS-PO audit"],
    }
    markers = {"Conventional": "o", "FE": "s", "IB": "^", "GS": "v"}
    for method in ("Conventional", "FE", "IB", "GS"):
        group = table[table["method"].eq(method)].copy()
        resolved = group[group["finite_ka_root_found"].astype(bool)]
        axes[0].scatter(
            resolved["minimum_pair_separation_mm"],
            resolved["finite_ka_displacement_a"],
            facecolor="none",
            edgecolor=colors[method],
            marker=markers[method],
            s=45,
            linewidth=1.0,
            alpha=0.72,
        )
        axes[1].scatter(
            resolved["minimum_pair_separation_mm"],
            1.0e3 * resolved["finite_ka_stiffness_min_n_m"],
            facecolor="none",
            edgecolor=colors[method],
            marker=markers[method],
            s=45,
            linewidth=1.0,
            alpha=0.72,
        )
        summary = (
            resolved.groupby("minimum_pair_separation_mm", as_index=False)
            .agg(
                displacement_median=("finite_ka_displacement_a", "median"),
                stiffness_median=("finite_ka_stiffness_min_n_m", "median"),
            )
            .sort_values("minimum_pair_separation_mm")
        )
        axes[0].plot(
            summary["minimum_pair_separation_mm"],
            summary["displacement_median"],
            color=colors[method],
            marker=markers[method],
            lw=1.75,
            ms=5.5,
            label=method,
        )
        axes[1].plot(
            summary["minimum_pair_separation_mm"],
            1.0e3 * summary["stiffness_median"],
            color=colors[method],
            marker=markers[method],
            lw=1.75,
            ms=5.5,
        )
    axes[0].set_ylabel(r"Exact displacement, $\|\Delta\mathbf{r}\|/a$")
    axes[1].set_ylabel(r"Weakest exact stiffness (mN m$^{-1}$)")
    axes[1].axhline(0.0, color="0.45", lw=0.8, zorder=0)
    for ax in axes:
        ax.set_xlabel("Minimum target separation (mm)")
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    axes[0].set_ylim(bottom=0.0)
    _unresolved_note(axes[0], table)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.015),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    outside_panel_labels(axes, ("a", "b"), x=-0.08, y=1.06)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.87, bottom=0.22, wspace=0.30)
    return _save(ctx, fig, "Appendix_D_close_triple_stress")


REQUESTED_APPENDIX_FIGURES = (
    appendix_b_prescription_sensitivity,
    appendix_c_lateral_target_stress,
    appendix_d_close_triple_stress,
)


__all__ = [producer.__name__ for producer in REQUESTED_APPENDIX_FIGURES] + [
    "REQUESTED_APPENDIX_FIGURES",
]
