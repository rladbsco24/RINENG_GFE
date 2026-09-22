#!/usr/bin/env python3
"""Mechanics-aware exact global-offset phase quantization.

This prototype enumerates the same complete one-dimensional family as
``best_global_offset_rounding``: all projectively distinct commands obtained by
adding one common phase offset and then rounding each channel to a uniform
``Q``-state alphabet.  It changes only the candidate ranking.

Let ``r_star`` be the equilibrium of the continuous command and let

    K_star = -dF/dr (r_star)

be its symmetric restoring stiffness.  For a rounded command ``q``, the
first-order implicit-function prediction is

    delta_r = solve(K_star, F(r_star, q) - F(r_star, u_star)).

The selected candidate first satisfies positive local stiffness at ``r_star``
when such a candidate exists, then minimizes ``norm(delta_r)``.  Candidate
stiffness and force use exactly the same centered finite-difference stencil as
the corrected FE backend.  This is exact enumeration of the offset-rounding
family, not an exact solution of the full ``Q**(N-1)`` discrete problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from inputs.hat_geometry_branch_benchmark import Geometry, K1, K2, transfer_matrix
from prototypes.discrete_solver import (
    QuantizationResult,
    TransitionProblem,
    _breakpoint_groups,
    gauge_fix_indices,
    indices_to_phase,
)


@dataclass(frozen=True)
class LocalMechanicsCache:
    """Target-independent field stencil centered on one equilibrium point."""

    center_m: np.ndarray
    transfer_grid: np.ndarray
    h_m: float


@dataclass(frozen=True)
class MechanicsReference:
    """Continuous-command equilibrium and local restoring stiffness."""

    equilibrium_m: np.ndarray
    force_n: np.ndarray
    stiffness_n_m: np.ndarray
    stiffness_eigenvalues_n_m: np.ndarray
    condition_number: float


@dataclass
class MechanicsOffsetResult:
    """Selected command plus mechanics diagnostics for all offset candidates."""

    quantization: QuantizationResult
    predicted_shift_m: float
    predicted_shift_vector_m: np.ndarray
    local_stiffness_eigenvalues_n_m: np.ndarray
    locally_stable: bool
    stable_candidate_count: int
    reference: MechanicsReference
    candidate_records: list[dict[str, float | int | bool]] = field(default_factory=list)


def build_local_mechanics_cache(
    geometry: Geometry,
    center_m: np.ndarray,
    h_m: float,
) -> LocalMechanicsCache:
    """Build the corrected 5x5x5 pressure stencil about ``center_m``."""

    center = np.asarray(center_m, dtype=float)
    if center.shape != (3,):
        raise ValueError("center_m must have shape (3,)")
    if not np.isfinite(h_m) or h_m <= 0.0:
        raise ValueError("h_m must be finite and positive")
    offsets = np.arange(-2, 3, dtype=float) * float(h_m)
    grid = np.array(
        [[x, y, z] for x in offsets for y in offsets for z in offsets],
        dtype=float,
    )
    transfer = transfer_matrix(center[None, :] + grid, geometry)
    return LocalMechanicsCache(center.copy(), transfer, float(h_m))


def mechanics_qvalues_from_field(
    cache: LocalMechanicsCache,
    field: np.ndarray,
) -> np.ndarray:
    """Return curvature, grad(U), pressure^2, and Hessian(U) at the cache center."""

    p = np.asarray(field, dtype=np.complex128).reshape(5, 5, 5)
    h = cache.h_m
    center = p[1:4, 1:4, 1:4]
    dx = (p[2:5, 1:4, 1:4] - p[0:3, 1:4, 1:4]) / (2.0 * h)
    dy = (p[1:4, 2:5, 1:4] - p[1:4, 0:3, 1:4]) / (2.0 * h)
    dz = (p[1:4, 1:4, 2:5] - p[1:4, 1:4, 0:3]) / (2.0 * h)
    potential = K1 * np.abs(center) ** 2 - K2 * (
        np.abs(dx) ** 2 + np.abs(dy) ** 2 + np.abs(dz) ** 2
    )
    u0 = potential[1, 1, 1]
    ux = (potential[2, 1, 1] - potential[0, 1, 1]) / (2.0 * h)
    uy = (potential[1, 2, 1] - potential[1, 0, 1]) / (2.0 * h)
    uz = (potential[1, 1, 2] - potential[1, 1, 0]) / (2.0 * h)
    uxx = (potential[2, 1, 1] - 2.0 * u0 + potential[0, 1, 1]) / h**2
    uyy = (potential[1, 2, 1] - 2.0 * u0 + potential[1, 0, 1]) / h**2
    uzz = (potential[1, 1, 2] - 2.0 * u0 + potential[1, 1, 0]) / h**2
    uxy = (
        potential[2, 2, 1]
        - potential[2, 0, 1]
        - potential[0, 2, 1]
        + potential[0, 0, 1]
    ) / (4.0 * h**2)
    uxz = (
        potential[2, 1, 2]
        - potential[2, 1, 0]
        - potential[0, 1, 2]
        + potential[0, 1, 0]
    ) / (4.0 * h**2)
    uyz = (
        potential[1, 2, 2]
        - potential[1, 2, 0]
        - potential[1, 0, 2]
        + potential[1, 0, 0]
    ) / (4.0 * h**2)
    p2 = float(np.abs(p[2, 2, 2]) ** 2)
    return np.asarray(
        [uxx + uyy + uzz, ux, uy, uz, p2, uxx, uyy, uzz, uxy, uxz, uyz],
        dtype=float,
    )


def stiffness_from_qvalues(qvalues: np.ndarray) -> np.ndarray:
    """Construct symmetric restoring stiffness ``Hessian(U)``."""

    q = np.asarray(qvalues, dtype=float)
    if q.shape != (11,):
        raise ValueError("qvalues must have shape (11,)")
    hxx, hyy, hzz, hxy, hxz, hyz = q[5:11]
    return np.asarray(
        [[hxx, hxy, hxz], [hxy, hyy, hyz], [hxz, hyz, hzz]],
        dtype=float,
    )


def reference_from_continuous_command(
    cache: LocalMechanicsCache,
    continuous_phase: np.ndarray,
) -> MechanicsReference:
    """Evaluate ``K_star`` at a prevalidated continuous equilibrium."""

    phase = np.asarray(continuous_phase, dtype=float)
    if phase.ndim != 1 or phase.size != cache.transfer_grid.shape[1]:
        raise ValueError("continuous_phase has incompatible shape")
    field = cache.transfer_grid @ np.exp(1j * phase)
    qvalues = mechanics_qvalues_from_field(cache, field)
    force = -qvalues[1:4]
    stiffness = stiffness_from_qvalues(qvalues)
    eigenvalues = np.linalg.eigvalsh(stiffness)
    singular = np.linalg.svd(stiffness, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    return MechanicsReference(
        equilibrium_m=cache.center_m.copy(),
        force_n=force,
        stiffness_n_m=stiffness,
        stiffness_eigenvalues_n_m=eigenvalues,
        condition_number=condition,
    )


def mechanics_aware_best_offset_rounding(
    problem: TransitionProblem,
    continuous_phase: np.ndarray,
    levels: int,
    cache: LocalMechanicsCache,
    *,
    reference: MechanicsReference | None = None,
    gauge: int = 0,
    stability_relative_floor: float = 0.0,
    predicted_shift_tolerance_m: float = 1.0e-12,
    keep_candidate_records: bool = False,
) -> MechanicsOffsetResult:
    """Select the exact offset-rounding candidate with best local mechanics.

    Stable candidates are ranked lexicographically by predicted equilibrium
    shift, then by weakest-stiffness retention, FE objective, and lexicographic
    command.
    If no candidate is locally stable, the least unstable candidate is selected
    first by largest weakest stiffness and then by predicted shift.
    """

    phase = np.asarray(continuous_phase, dtype=float)
    if phase.shape != (problem.n,):
        raise ValueError(f"continuous_phase must have shape ({problem.n},)")
    if cache.transfer_grid.shape != (125, problem.n):
        raise ValueError("cache must be a 5x5x5 field stencil for this problem")
    if stability_relative_floor < 0.0:
        raise ValueError("stability_relative_floor must be nonnegative")
    levels = int(levels)
    if levels < 2:
        raise ValueError("levels must be at least 2")

    if reference is None:
        reference = reference_from_continuous_command(cache, phase)
    if reference.stiffness_n_m.shape != (3, 3):
        raise ValueError("reference stiffness must have shape (3,3)")
    if reference.stiffness_eigenvalues_n_m[0] <= 0.0:
        raise ValueError("continuous reference stiffness must be positive definite")

    step = 2.0 * np.pi / levels
    breakpoints, groups = _breakpoint_groups(phase, step)
    if breakpoints.size == 0:
        raise RuntimeError("quantizer breakpoint enumeration is empty")
    following = breakpoints[1] if len(breakpoints) > 1 else breakpoints[0] + step
    alpha = float(np.mod(breakpoints[0] + 0.5 * (following - breakpoints[0]), step))
    raw_indices = np.mod(
        np.floor((phase + alpha) / step + 0.5).astype(np.int64), levels
    )
    roots = np.exp(2j * np.pi * np.arange(levels) / levels)
    amplitude = roots[raw_indices].copy()
    field = cache.transfer_grid @ amplitude

    stability_floor = float(
        stability_relative_floor * reference.stiffness_eigenvalues_n_m[0]
    )
    candidates: list[dict[str, object]] = []
    seen: set[bytes] = set()

    def consider(candidate_offset: float) -> None:
        fixed = gauge_fix_indices(raw_indices, levels, gauge)
        key = np.ascontiguousarray(fixed).tobytes()
        if key in seen:
            return
        seen.add(key)
        qvalues = mechanics_qvalues_from_field(cache, field)
        force = -qvalues[1:4]
        force_perturbation = force - reference.force_n
        predicted_vector = np.linalg.solve(
            reference.stiffness_n_m, force_perturbation
        )
        predicted_norm = float(np.linalg.norm(predicted_vector))
        local_stiffness = stiffness_from_qvalues(qvalues)
        local_eigenvalues = np.linalg.eigvalsh(local_stiffness)
        locally_stable = bool(local_eigenvalues[0] > stability_floor)
        candidates.append(
            {
                "indices": fixed.copy(),
                "offset_rad": float(candidate_offset),
                "predicted_shift_vector_m": predicted_vector,
                "predicted_shift_m": predicted_norm,
                "local_stiffness_eigenvalues_n_m": local_eigenvalues,
                "locally_stable": locally_stable,
            }
        )

    consider(alpha)
    for group_index in range(1, len(groups)):
        for coordinate in groups[group_index]:
            coordinate = int(coordinate)
            old = amplitude[coordinate]
            raw_indices[coordinate] = (raw_indices[coordinate] + 1) % levels
            amplitude[coordinate] = roots[raw_indices[coordinate]]
            field += (amplitude[coordinate] - old) * cache.transfer_grid[:, coordinate]
        following = (
            breakpoints[group_index + 1]
            if group_index + 1 < len(breakpoints)
            else breakpoints[0] + step
        )
        representative = float(
            np.mod(
                breakpoints[group_index]
                + 0.5 * (following - breakpoints[group_index]),
                step,
            )
        )
        consider(representative)

    stable = [candidate for candidate in candidates if candidate["locally_stable"]]
    if stable:
        minimum_shift = min(float(candidate["predicted_shift_m"]) for candidate in stable)
        near_minimum = [
            candidate
            for candidate in stable
            if float(candidate["predicted_shift_m"])
            <= minimum_shift + predicted_shift_tolerance_m
        ]
        selected = min(
            near_minimum,
            key=lambda candidate: (
                abs(
                    float(candidate["local_stiffness_eigenvalues_n_m"][0])
                    / float(reference.stiffness_eigenvalues_n_m[0])
                    - 1.0
                ),
                problem.score_phase(
                    indices_to_phase(candidate["indices"], levels, gauge)
                ),
                tuple(int(v) for v in candidate["indices"]),
            ),
        )
    else:
        selected = min(
            candidates,
            key=lambda candidate: (
                -float(candidate["local_stiffness_eigenvalues_n_m"][0]),
                float(candidate["predicted_shift_m"]),
                tuple(int(v) for v in candidate["indices"]),
            ),
        )

    indices = np.asarray(selected["indices"], dtype=np.int64)
    selected_phase = indices_to_phase(indices, levels, gauge)
    objective = problem.score_phase(selected_phase)
    quantization = QuantizationResult(
        levels=levels,
        indices=indices,
        phase=selected_phase,
        objective=objective,
        evaluations=len(candidates),
        rule="mechanics_aware_exact_best_global_offset",
        offset_rad=float(selected["offset_rad"]),
        distinct_candidates=len(candidates),
        metadata={
            "breakpoint_groups": int(len(groups)),
            "maximum_candidates": int(problem.n),
            "stable_candidate_count": int(len(stable)),
            "stability_relative_floor": float(stability_relative_floor),
            "reference_condition_number": float(reference.condition_number),
            "reference_force_norm_n": float(np.linalg.norm(reference.force_n)),
        },
    )
    records: list[dict[str, float | int | bool]] = []
    if keep_candidate_records:
        for candidate_id, candidate in enumerate(candidates):
            eigenvalues = np.asarray(candidate["local_stiffness_eigenvalues_n_m"])
            records.append(
                {
                    "candidate": int(candidate_id),
                    "offset_rad": float(candidate["offset_rad"]),
                    "predicted_shift_m": float(candidate["predicted_shift_m"]),
                    "local_stiffness_min_n_m": float(eigenvalues[0]),
                    "local_stiffness_mid_n_m": float(eigenvalues[1]),
                    "local_stiffness_max_n_m": float(eigenvalues[2]),
                    "locally_stable": bool(candidate["locally_stable"]),
                }
            )
    return MechanicsOffsetResult(
        quantization=quantization,
        predicted_shift_m=float(selected["predicted_shift_m"]),
        predicted_shift_vector_m=np.asarray(
            selected["predicted_shift_vector_m"], dtype=float
        ),
        local_stiffness_eigenvalues_n_m=np.asarray(
            selected["local_stiffness_eigenvalues_n_m"], dtype=float
        ),
        locally_stable=bool(selected["locally_stable"]),
        stable_candidate_count=len(stable),
        reference=reference,
        candidate_records=records,
    )


__all__ = [
    "LocalMechanicsCache",
    "MechanicsOffsetResult",
    "MechanicsReference",
    "build_local_mechanics_cache",
    "mechanics_aware_best_offset_rounding",
    "mechanics_qvalues_from_field",
    "reference_from_continuous_command",
    "stiffness_from_qvalues",
]
