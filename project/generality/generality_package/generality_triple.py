"""Current-setting native Triple sections for the eight-setting main study.

The main endpoint solvers and figure composition remain the notebook's own.
Only the old archive-dependent axes and obsolete history-recovery hook are
replaced. Each local section is evaluated on its own current objective.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Mapping

import numpy as np
import pandas as pd

from hat_revision_pipeline import final_figures as ff
from hat_revision_pipeline.cache import digest_array, digest_payload
from hat_revision_pipeline.multitrap import evaluate_multitrap_objective
from hat_revision_pipeline.triple_long_iteration import _gauge_expand, _gauge_reduce


def _conventional_history(
    node: Mapping[str, Any], bank: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, str]:
    """Return available phases from this solve, with a verified initial state."""
    result = node["Conventional"]
    if result.problem_fingerprint != node["problem"].fingerprint:
        raise ValueError("Triple history and current problem fingerprints differ")
    if "Conventional_history_phase_rad" in node:
        history = np.asarray(node["Conventional_history_phase_rad"], dtype=float)
        source = "current-setting recorded accepted phases"
    else:
        checkpoints = node.get(
            "Conventional_checkpoint_phases", result.checkpoint_phases_rad
        )
        history = np.stack(
            [np.asarray(checkpoints[key], dtype=float) for key in sorted(checkpoints)]
        )
        source = "current-setting stored cumulative checkpoints"
    if history.ndim != 2 or history.shape[1] != len(result.terminal_phase_rad):
        raise ValueError("Current Triple Conventional phase history has invalid shape")

    initial = (
        np.asarray(bank["initial_phase"], dtype=float)
        if bank is not None
        else np.random.default_rng(int(result.seed)).uniform(
            -np.pi, np.pi, len(result.terminal_phase_rad)
        )
    )
    initial = _gauge_expand(_gauge_reduce(initial))
    if digest_array(initial) != result.initial_phase_hash:
        raise ValueError("Current Triple seed does not reproduce its initial phase hash")
    # Prepending the verified start only supplies a fallback if all stored
    # checkpoints are identical after early termination. The native helper
    # first considers later recorded phases in reverse chronological order.
    if len(history) == 0 or digest_array(history[0]) != digest_array(initial):
        history = np.vstack((initial[None, :], history))
        source += "; verified original initial state prepended"
    return history, source


def _fixed_triple_conventional_history(ctx, bank, node):
    """Return this setting's existing solve data without replaying another run."""
    return _conventional_history(node, bank)[0]


def _tri3_plane(ctx, node, display, phase, *, half_range_rad):
    """Evaluate the current endpoint on native axes computed from this solve."""
    phase = np.asarray(phase, dtype=float)
    if display == "Conventional":
        result = node["Conventional"]
        config = node["conventional_config"]
        history, history_source = _conventional_history(node)
        force_epsilon = float(result.force_epsilon)
        uniformity_epsilon = float(result.uniformity_epsilon)
    elif display == "FE":
        # The established main composition supplies the GFE result in this
        # plotting slot, together with its actual compensated configuration.
        result = node["FE"]
        config = node["fe_config"]
        history = np.asarray(result.history_phase_rad, dtype=float)
        history_source = "current-setting recorded accepted phases"
        force_epsilon = float(result.stages[-1].force_epsilon)
        uniformity_epsilon = float(result.stages[-1].uniformity_epsilon)
    else:
        raise KeyError(display)
    u, v = ff._native_multitrap_directions(
        node["problem"], phase, history, config,
        force_epsilon=force_epsilon, uniformity_epsilon=uniformity_epsilon,
    )
    points = 61 if ctx.config.full else 41
    half_range = float(half_range_rad)
    if half_range <= 0.0:
        raise ValueError("half_range_rad must be positive")
    q = np.linspace(-half_range, half_range, points)
    positions_hash = digest_array(np.asarray(ctx.positions_m, dtype=float))
    normals_hash = digest_array(np.asarray(ctx.normals, dtype=float))
    geometry_hash = digest_payload(dict(positions=positions_hash, normals=normals_hash))
    history_hash = digest_array(history)
    payload = dict(
        contract="generality-fig4-current-setting-native-axes-v1",
        problem=node["problem"].fingerprint,
        config=config.to_payload(),
        phase=digest_array(phase),
        positions=positions_hash, normals=normals_hash,
        geometry=geometry_hash, frequency_hz=float(ctx.config.frequency_hz),
        history=history_hash, history_source=history_source,
        direction_u=digest_array(u), direction_v=digest_array(v),
        half_range_rad=half_range, points=points,
        force_epsilon=force_epsilon, uniformity_epsilon=uniformity_epsilon,
    )

    def compute():
        plane = np.empty((points, points), dtype=float)
        gradient_s1 = np.empty_like(plane)
        gradient_s2 = np.empty_like(plane)
        for j, b in enumerate(q):
            for i, a in enumerate(q):
                evaluation = evaluate_multitrap_objective(
                    node["problem"], phase + a * u + b * v, config,
                    force_epsilon=force_epsilon,
                    uniformity_epsilon=uniformity_epsilon,
                )
                plane[j, i] = evaluation.value
                gradient_s1[j, i] = np.dot(evaluation.gradient_full, u)
                gradient_s2[j, i] = np.dot(evaluation.gradient_full, v)
        return dict(
            q=q, plane=plane - np.nanmin(plane), objective_raw=plane,
            gradient_s1=gradient_s1, gradient_s2=gradient_s2,
            direction_u=u, direction_v=v,
        )

    value, cache_status, cache_path = ctx.cache.get_or_compute(
        "generality_fig4_native_section", payload, compute, recompute=ctx.recompute,
    )
    evaluation = evaluate_multitrap_objective(
        node["problem"], phase, config, force_epsilon=force_epsilon,
        uniformity_epsilon=uniformity_epsilon,
    )
    gradient = np.asarray(evaluation.gradient_full, dtype=float)
    value = dict(value)
    value.update(
        projected_gradient=np.asarray([np.dot(gradient, u), np.dot(gradient, v)]),
        full_gradient_norm=float(np.linalg.norm(gradient)),
        optimizer_gradient_norm=float(np.linalg.norm(evaluation.gradient_reduced)),
        optimizer_gradient_reduced=evaluation.gradient_reduced,
        geometry_content_id=geometry_hash, positions_content_id=positions_hash,
        normals_content_id=normals_hash, frequency_hz=float(ctx.config.frequency_hz),
        history_content_id=history_hash, history_source=history_source,
        direction_u_content_id=digest_array(u), direction_v_content_id=digest_array(v),
        cache_payload_content_id=digest_payload(payload), cache_content_id=cache_path.stem,
        cache_status=cache_status,
    )
    row = dict(
        method=("GFE" if getattr(config, "compensate_effective_gravity", False) else display),
        coordinate_source="fresh native axes from current-setting terminal gradient and available phase history",
        phase_content_id=digest_array(phase), geometry_content_id=geometry_hash,
        positions_content_id=positions_hash, normals_content_id=normals_hash,
        frequency_hz=float(ctx.config.frequency_hz), history_content_id=history_hash,
        history_source=history_source, direction_u_content_id=digest_array(u),
        direction_v_content_id=digest_array(v), cache_payload_content_id=digest_payload(payload),
        cache_content_id=cache_path.stem, cache_status=cache_status,
        plane_points_per_axis=points, half_range_rad=half_range,
    )
    table_name = "fig4_current_setting_native_axes"
    previous = pd.DataFrame(ctx.tables.get(table_name, []))
    if len(previous):
        previous = previous.loc[previous.method != row["method"]]
    ctx.tables[table_name] = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    return value


def install(ctx):
    """Install the per-setting Triple adapter before producing the main figures."""
    ff._tri3_plane = _tri3_plane
    ff._fixed_triple_conventional_history = _fixed_triple_conventional_history
    if not getattr(ff.figure_4_triple_fields_and_decision, "_generality_native_axes", False):
        original = ff.figure_4_triple_fields_and_decision

        @wraps(original)
        def figure_4_current_setting(context):
            figure = original(context)
            name = "fig4_triple_local_geometry_summary"
            if name in context.tables:
                frame = pd.DataFrame(context.tables[name]).copy()
                frame["coordinate_source"] = (
                    "fresh native axes from current-setting terminal gradient and available phase history"
                )
                context.tables[name] = frame
            return figure

        figure_4_current_setting._generality_native_axes = True
        ff.figure_4_triple_fields_and_decision = figure_4_current_setting
    ctx.memo["generality_triple_native_axes"] = True

