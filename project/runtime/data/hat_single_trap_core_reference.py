#!/usr/bin/env python3
"""Exact single-trap C/FE/RH objective and projective phase utilities.

This module contains no continuation machinery.  It implements the supplied
v54 notebook's target-local objective on the fixed 0.5 mm stencil and the
gauge-invariant average-cosine/projective statistics used by the independent
cold-start benchmark.
"""

from __future__ import annotations

from rineng_content_id import content_identity
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from branch_generalization_benchmark import (
    AMP_TABLE,
    FREQ0,
    Condition,
    LocalObjective,
    gauge_full,
    propagation_carrier,
    reduce_gauge,
    solve,
    square_positions,
    wrap_phase,
)


BASE_TARGET = (0.010, 0.010, 0.030)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    alpha: float
    beta: float
    pressure_mode: str
    weight: np.ndarray


def method_spec(
    name: str,
    alpha_override: float | None = None,
    beta_override: float | None = None,
) -> MethodSpec:
    if name == "Conventional":
        spec = MethodSpec(name, 0.0, 1.0, "abs", np.diag([1000.0, 1000.0, 10.0]))
    elif name == "Force-Equilibrium":
        spec = MethodSpec(name, 9.0, 0.0, "abs", np.eye(3))
    elif name == "Regularized-Hybrid":
        spec = MethodSpec(name, 9.0, 5.0e-5, "smooth_abs", np.eye(3))
    else:
        raise KeyError(name)
    return MethodSpec(
        name=spec.name,
        alpha=spec.alpha if alpha_override is None else float(alpha_override),
        beta=spec.beta if beta_override is None else float(beta_override),
        pressure_mode=spec.pressure_mode,
        weight=np.asarray(spec.weight, dtype=float),
    )


def reference_condition(side: int, spec: MethodSpec) -> Condition:
    return Condition(
        label="single-trap-reference",
        path="single-trap",
        s=0.0,
        positions=square_positions(side),
        target=BASE_TARGET,
        frequency=FREQ0,
        weight=np.asarray(spec.weight, dtype=float),
    )


class ExactSingleTrapObjective(LocalObjective):
    """Analytic NumPy implementation numerically identical to v54 at q0."""

    def __init__(
        self,
        condition: Condition,
        spec: MethodSpec,
        smooth_pressure_rel: float = 1.0e-3,
        fixed_stencil_m: float | None = None,
    ):
        super().__init__(
            condition=condition,
            alpha=spec.alpha,
            beta=spec.beta,
            smooth_pressure_rel=smooth_pressure_rel,
            normalize_terms=False,
        )
        self.spec = spec
        self.pressure_mode = spec.pressure_mode
        if fixed_stencil_m is not None and not math.isclose(
            self.h, float(fixed_stencil_m), rel_tol=0.0, abs_tol=1.0e-15
        ):
            self.h = float(fixed_stencil_m)
            self.H = self._build_transfer()
            self.n = self.H.shape[-1]

    def _smooth_pressure_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        phi = gauge_full(x)
        actuator = np.exp(1j * phi)
        pressure = np.einsum("...n,n->...", self.H, actuator, optimize=True)
        phase_jacobian = 1j * self.H * actuator.reshape((1, 1, 1, -1))
        local_rms = math.sqrt(max(float(np.mean(np.abs(pressure) ** 2)), 1.0e-64))
        d_local_rms = (
            np.mean(
                np.real(np.conj(pressure)[..., None] * phase_jacobian),
                axis=(0, 1, 2),
            )
            / max(local_rms, 1.0e-32)
        )
        epsilon = self.smooth_pressure_rel * (local_rms + 1.0e-32)
        d_epsilon = self.smooth_pressure_rel * d_local_rms
        p0 = pressure[self.center]
        q0 = phase_jacobian[self.center]
        value = math.sqrt(max(float(abs(p0) ** 2 + epsilon * epsilon), 1.0e-64))
        gradient = (
            np.real(np.conj(p0) * q0) + epsilon * d_epsilon
        ) / max(value, 1.0e-32)
        return float(value), np.asarray(gradient, dtype=float)

    def pressure_term_and_gradient(self, x: np.ndarray, raw: dict[str, Any]) -> tuple[float, np.ndarray]:
        if self.pressure_mode == "smooth_abs":
            return self._smooth_pressure_and_gradient(x)
        return float(raw["pressure"]), np.asarray(raw["d_pressure"], dtype=float)

    def component_terms(self, x: np.ndarray) -> dict[str, Any]:
        raw = self._raw_metrics(x, need_grad=True)
        pressure, d_pressure = self.pressure_term_and_gradient(x, raw)
        curvature_gradient = -np.asarray(raw["d_curvature"], dtype=float)[1:]
        force_gradient = self.alpha * np.asarray(raw["d_force"], dtype=float)[1:]
        pressure_gradient = self.beta * np.asarray(d_pressure, dtype=float)[1:]
        return {
            "curvature_loss": -float(raw["curvature"]),
            "force_loss": self.alpha * float(raw["force_norm"]),
            "pressure_loss": self.beta * float(pressure),
            "curvature_gradient": curvature_gradient,
            "force_gradient": force_gradient,
            "pressure_gradient": pressure_gradient,
            "raw": raw,
            "pressure_penalty": float(pressure),
        }

    def fun_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        terms = self.component_terms(x)
        value = terms["curvature_loss"] + terms["force_loss"] + terms["pressure_loss"]
        gradient = (
            terms["curvature_gradient"]
            + terms["force_gradient"]
            + terms["pressure_gradient"]
        )
        return float(value), np.asarray(gradient, dtype=float)

    def metrics(self, x: np.ndarray) -> dict[str, Any]:
        terms = self.component_terms(x)
        raw = dict(terms["raw"])
        raw["pressure_penalty"] = terms["pressure_penalty"]
        return raw


def optimize_cold_start(
    objective: ExactSingleTrapObjective,
    seed: int,
    maxiter: int,
    gtol: float,
) -> tuple[np.ndarray, object, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    phase0 = rng.uniform(0.0, 2.0 * np.pi, size=objective.n).astype(float)
    _, initial_gradient = objective.fun_grad(reduce_gauge(phase0))
    initial_gradient_norm = float(np.linalg.norm(initial_gradient))
    result, runtime = solve(objective, reduce_gauge(phase0), maxiter=maxiter, gtol=gtol)
    phase = gauge_full(wrap_phase(np.asarray(result.x, dtype=float)))
    x = reduce_gauge(phase)
    value, gradient = objective.fun_grad(x)
    terms = objective.component_terms(x)
    raw = terms["raw"]
    hessian_u = np.asarray(raw["hess_u"], dtype=float)
    physical_eigenvalues = np.linalg.eigvalsh(hessian_u)
    physical_tolerance = max(float(np.max(np.abs(physical_eigenvalues))) * 1.0e-8, 1.0e-20)
    curvature_gradient_norm = float(np.linalg.norm(terms["curvature_gradient"]))
    force_gradient_norm = float(np.linalg.norm(terms["force_gradient"]))
    pressure_gradient_norm = float(np.linalg.norm(terms["pressure_gradient"]))
    nonforce_gradient_norm = float(
        np.linalg.norm(terms["curvature_gradient"] + terms["pressure_gradient"])
    )
    gradient_dominance = force_gradient_norm / max(nonforce_gradient_norm, 1.0e-30)
    scalar_dominance = abs(float(terms["force_loss"])) / max(
        abs(float(terms["curvature_loss"])) + abs(float(terms["pressure_loss"])),
        1.0e-30,
    )
    row = {
        "objective": float(value),
        "phase_gradient_norm": float(np.linalg.norm(gradient)),
        "phase_gradient_rms": float(np.linalg.norm(gradient) / math.sqrt(gradient.size)),
        "initial_phase_gradient_norm": initial_gradient_norm,
        "initial_phase_gradient_rms": float(initial_gradient_norm / math.sqrt(initial_gradient.size)),
        "terminal_to_initial_gradient_ratio": float(
            np.linalg.norm(gradient) / max(initial_gradient_norm, 1.0e-30)
        ),
        "force_norm": float(raw["force_norm"]),
        "curvature": float(raw["curvature"]),
        "pressure": float(raw["pressure"]),
        "pressure_penalty": float(terms["pressure_penalty"]),
        "curvature_loss": float(terms["curvature_loss"]),
        "force_loss": float(terms["force_loss"]),
        "pressure_loss": float(terms["pressure_loss"]),
        "curvature_gradient_norm": curvature_gradient_norm,
        "force_gradient_norm": force_gradient_norm,
        "pressure_gradient_norm": pressure_gradient_norm,
        "force_gradient_dominance": gradient_dominance,
        "force_scalar_dominance": scalar_dominance,
        "physical_hessian_min": float(physical_eigenvalues[0]),
        "physical_hessian_mid": float(physical_eigenvalues[1]),
        "physical_hessian_max": float(physical_eigenvalues[2]),
        "physical_hessian_inertia": inertia_label(hessian_u),
        "physical_stable_positive_definite": bool(physical_eigenvalues[0] > physical_tolerance),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "runtime_s": float(runtime),
        "scipy_success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
    }
    return phase, result, row


def target_demodulated_state(phase: np.ndarray, condition: Condition) -> np.ndarray:
    residual = wrap_phase(np.asarray(phase, dtype=float) - propagation_carrier(condition))
    return np.exp(1j * residual) / math.sqrt(residual.size)


def aligned_average_cosine_similarity_from_states(u: np.ndarray, v: np.ndarray) -> float:
    """Optimal-gauge mean cosine over all phase coordinates.

    For unit-norm phasor states, max_gamma mean cos(delta_phi-gamma)
    equals rho=|u^H v| exactly.
    """
    rho = float(np.clip(abs(np.vdot(u, v)), 0.0, 1.0))
    return rho


def aligned_average_cosine_similarity(
    phase_a: np.ndarray,
    condition_a: Condition,
    phase_b: np.ndarray,
    condition_b: Condition,
) -> float:
    return aligned_average_cosine_similarity_from_states(
        target_demodulated_state(phase_a, condition_a),
        target_demodulated_state(phase_b, condition_b),
    )


def fubini_study_from_similarity(average_cosine: float) -> float:
    rho = np.clip(float(average_cosine), 0.0, 1.0)
    return float(np.arccos(rho))


def fubini_study_from_states(u: np.ndarray, v: np.ndarray) -> float:
    return fubini_study_from_similarity(aligned_average_cosine_similarity_from_states(u, v))


def projector_chordal_from_states(u: np.ndarray, v: np.ndarray) -> float:
    rho = float(np.clip(abs(np.vdot(u, v)), 0.0, 1.0))
    return float(math.sqrt(max(0.0, 1.0 - rho * rho)))


def complex_field_overlap(
    objective_a: ExactSingleTrapObjective,
    phase_a: np.ndarray,
    objective_b: ExactSingleTrapObjective,
    phase_b: np.ndarray,
) -> float:
    pa = np.asarray(objective_a.metrics(reduce_gauge(phase_a))["p"]).reshape(-1)
    pb = np.asarray(objective_b.metrics(reduce_gauge(phase_b))["p"]).reshape(-1)
    return float(
        np.clip(
            abs(np.vdot(pa, pb))
            / max(float(np.linalg.norm(pa) * np.linalg.norm(pb)), 1.0e-30),
            0.0,
            1.0,
        )
    )


def physical_field_winding(
    phase: np.ndarray,
    condition: Condition,
    radius_wavelengths: float = 0.75,
    samples: int = 96,
) -> int:
    angles = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
    radius_loop = float(radius_wavelengths) * condition.wavelength
    target = np.asarray(condition.target, dtype=float)
    points = np.column_stack(
        [
            target[0] + radius_loop * np.cos(angles),
            target[1] + radius_loop * np.sin(angles),
            np.full_like(angles, target[2]),
        ]
    )
    delta = points[:, None, :] - np.asarray(condition.positions, dtype=float)[None, :, :]
    radius = np.maximum(np.linalg.norm(delta, axis=2), 1.0e-9)
    theta = np.clip(
        np.rad2deg(np.arccos(np.clip(delta[:, :, 2] / radius, -1.0, 1.0))),
        0.0,
        90.0,
    )
    amplitude = np.interp(theta, np.arange(91.0), AMP_TABLE) / radius
    k = 2.0 * np.pi / condition.wavelength
    pressure = np.sum(
        amplitude * np.exp(1j * (np.asarray(phase)[None, :] + k * radius)), axis=1
    )
    closed = np.concatenate([np.angle(pressure), np.angle(pressure[:1])])
    circulation = float(np.sum(wrap_phase(np.diff(closed))))
    return int(np.rint(circulation / (2.0 * np.pi)))


def inertia_label(hessian: np.ndarray, relative_tol: float = 1.0e-8) -> str:
    eigenvalues = np.linalg.eigvalsh(np.asarray(hessian, dtype=float))
    tolerance = max(float(np.max(np.abs(eigenvalues))) * relative_tol, 1.0e-20)
    negative = int(np.sum(eigenvalues < -tolerance))
    zero = int(np.sum(np.abs(eigenvalues) <= tolerance))
    positive = int(np.sum(eigenvalues > tolerance))
    return f"{negative}/{zero}/{positive}"


def dense_phase_hessian(
    objective: ExactSingleTrapObjective,
    phase: np.ndarray,
    epsilon: float = 2.0e-4,
) -> tuple[np.ndarray, float]:
    x = reduce_gauge(phase)
    hessian = np.empty((x.size, x.size), dtype=float)
    started = time.perf_counter()
    for column in range(x.size):
        step = np.zeros_like(x)
        step[column] = epsilon
        _, plus = objective.fun_grad(x + step)
        _, minus = objective.fun_grad(x - step)
        hessian[:, column] = (plus - minus) / (2.0 * epsilon)
    hessian = 0.5 * (hessian + hessian.T)
    return np.linalg.eigvalsh(hessian), time.perf_counter() - started


def file_content_id(path: Path) -> str:
    digest = content_identity()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
