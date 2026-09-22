#!/usr/bin/env python3
"""Branch-informed generalization benchmark for the RINENG HAT revision.

The implementation intentionally stays within the mathematics already present in
the manuscript: a phase-only forward model, target-local Gor'kov derivatives,
gauge-fixed stationary endpoints, and implicit-function predictor-corrector
continuation.  It does not use or name any future surrogate/manifold architecture.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


C0 = 343.0
PITCH0 = 0.010
FREQ0 = 40_000.0
STENCIL_H = 0.0005
HALF_STENCIL = 2
TINY = 1.0e-30

# Murata MA40S4S directivity data used by the manuscript notebook.  The source
# values are tabulated every two degrees; the forward model interpolates them to
# one-degree resolution exactly as in the notebook.
_AMP_RAW = np.array(
    [
        10.000, 11.832, 13.509, 15.414, 17.615, 18.240, 20.172, 21.626,
        22.309, 23.054, 23.820, 23.820, 23.820, 24.650, 24.650, 26.306,
        29.052, 32.140, 35.496, 39.243, 41.952, 46.411, 47.958, 49.598,
        53.009, 56.657, 58.652, 60.581, 62.610, 64.885, 67.007, 69.282,
        71.624, 71.624, 76.551, 76.551, 79.183, 84.676, 84.676, 87.579,
        90.388, 93.488, 93.488, 93.488, 96.695, 100.000, 96.695, 93.488,
        93.488, 93.488, 90.388, 87.579, 84.676, 84.676, 79.183, 76.551,
        76.551, 71.624, 71.624, 69.282, 67.007, 64.885, 62.610, 60.581,
        58.652, 56.657, 53.009, 49.598, 47.958, 46.411, 41.952, 39.243,
        35.496, 32.140, 29.052, 26.306, 24.650, 24.650, 23.820, 23.820,
        23.820, 23.054, 22.309, 21.626, 20.172, 18.240, 17.615, 15.414,
        13.509, 11.832, 10.000,
    ],
    dtype=float,
)
AMP_TABLE = np.interp(np.arange(91.0), np.arange(0.0, 91.0, 2.0), _AMP_RAW[45:])


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def gauge_full(x_reduced: np.ndarray) -> np.ndarray:
    """Use the first transducer as the global-phase gauge."""
    return np.concatenate(([0.0], np.asarray(x_reduced, dtype=float)))


def reduce_gauge(phi: np.ndarray) -> np.ndarray:
    phi = wrap_phase(np.asarray(phi, dtype=float) - float(phi[0]))
    return phi[1:]


def circular_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return wrap_phase(np.asarray(a) - np.asarray(b))


def square_positions(side: int = 16, pitch: float = PITCH0, aspect: float = 1.0) -> np.ndarray:
    """Area-preserving rectangular deformation of a square array."""
    axis = np.arange(side, dtype=float) - (side - 1.0) / 2.0
    sx, sy = math.sqrt(aspect), 1.0 / math.sqrt(aspect)
    return np.array([(sx * pitch * x, sy * pitch * y, 0.0) for x in axis for y in axis])


def rotated_weight(
    axis_azimuth_deg: float,
    axis_tilt_deg: float = 0.0,
    anisotropy: float = 1.0,
) -> np.ndarray:
    """Full SPD curvature tensor; identity at 0 and 4:1:0.25 at 1."""
    az = np.deg2rad(axis_azimuth_deg)
    tilt = np.deg2rad(axis_tilt_deg)
    rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]])
    rotation = rz @ ry
    a = float(anisotropy)
    eig = np.exp(a * np.log(np.array([4.0, 1.0, 0.25])))
    eig *= 3.0 / np.sum(eig)
    return rotation @ np.diag(eig) @ rotation.T


@dataclass(frozen=True)
class Condition:
    label: str
    path: str
    s: float
    positions: np.ndarray
    target: tuple[float, float, float]
    frequency: float
    weight: np.ndarray

    @property
    def wavelength(self) -> float:
        return C0 / self.frequency


def gorkov_constants(frequency: float) -> tuple[float, float]:
    rho0, c0, rho_p, c_p = 1.225, C0, 100.0, 2400.0
    omega = 2.0 * np.pi * frequency
    radius = 1.3e-3 / 2.0
    volume = 4.0 * np.pi * radius**3 / 3.0
    k1 = 0.25 * volume * (1.0 / (c0**2 * rho0) - 1.0 / (c_p**2 * rho_p))
    # Standard Gor'kov density-contrast coefficient.  With peak-amplitude
    # complex phasors, U = k1 |p|^2 - k2 |grad p|^2.  The dipole contrast is
    # 2(rho_p-rho0)/(2rho_p+rho0), so k2 is positive for the denser particle.
    k2 = 0.75 * volume * ((rho_p - rho0) / (omega**2 * rho0 * (rho0 + 2.0 * rho_p)))
    return k1, k2


class LocalObjective:
    """Exact NumPy objective and phase gradient on the manuscript's 5^3 stencil."""

    def __init__(
        self,
        condition: Condition,
        alpha: float = 9.0,
        beta: float = 0.0,
        smooth_pressure_rel: float = 1.0e-3,
        normalize_terms: bool = False,
        reference_phase: np.ndarray | None = None,
    ) -> None:
        self.condition = condition
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth_pressure_rel = float(smooth_pressure_rel)
        self.normalize_terms = bool(normalize_terms)
        # Preserve h/lambda across frequency changes so that the derivative
        # discretization itself is not an uncontrolled generalization axis.
        self.h = STENCIL_H * (condition.wavelength / (C0 / FREQ0))
        self.shape = (2 * HALF_STENCIL + 1,) * 3
        self.center = (HALF_STENCIL,) * 3
        self.H = self._build_transfer()
        # H is stored as (nx, ny, nz, n_actuators).  The spatial stencil is
        # 5 x 5 x 5, so shape[1] would silently report 5 rather than the
        # actuator count.
        self.n = self.H.shape[-1]
        self.scale_curvature = 1.0
        self.scale_force = 1.0
        self.scale_pressure = 1.0
        if self.normalize_terms:
            # Deterministic cross-configuration scales.  P_q is the coherent
            # target amplitude bound and S_U is the corresponding potential
            # scale.  At 40 kHz this normalized objective is a positive scalar
            # multiple of the manuscript objective; elsewhere it holds the
            # dimensionless weighting fixed instead of silently retuning it.
            h0 = self.H[self.center]
            p_q = max(float(np.sum(np.abs(h0))), 1.0e-20)
            k = 2.0 * np.pi / condition.wavelength
            k0 = 2.0 * np.pi / (C0 / FREQ0)
            k1, k2 = gorkov_constants(condition.frequency)
            s_u = p_q**2 * (abs(k1) + abs(k2) * k**2)
            self.scale_curvature = max(k**2 * s_u, 1.0e-30)
            self.scale_force = max(k * s_u, 1.0e-30)
            self.scale_pressure = p_q
            self.alpha = float(alpha) / k0

    def _build_transfer(self) -> np.ndarray:
        c = self.condition
        offsets = np.arange(-HALF_STENCIL, HALF_STENCIL + 1, dtype=float) * self.h
        pts = np.array(
            [
                (c.target[0] + dx, c.target[1] + dy, c.target[2] + dz)
                for dx in offsets
                for dy in offsets
                for dz in offsets
            ],
            dtype=float,
        )
        delta = pts[:, None, :] - np.asarray(c.positions, dtype=float)[None, :, :]
        radius = np.linalg.norm(delta, axis=2)
        radius = np.maximum(radius, 1.0e-9)
        cos_theta = np.clip(delta[:, :, 2] / radius, -1.0, 1.0)
        theta = np.clip(np.rad2deg(np.arccos(cos_theta)), 0.0, 90.0)
        amplitude = np.interp(theta, np.arange(91.0), AMP_TABLE) / radius
        k = 2.0 * np.pi / c.wavelength
        return (amplitude * np.exp(1j * k * radius)).reshape(self.shape + (c.positions.shape[0],))

    def _raw_metrics(self, x: np.ndarray, need_grad: bool) -> dict[str, np.ndarray | float]:
        phi = gauge_full(x)
        actuator = np.exp(1j * phi)
        p = np.einsum("...n,n->...", self.H, actuator, optimize=True)
        q = None
        if need_grad:
            q = 1j * self.H * actuator.reshape((1, 1, 1, -1))

        dp = [np.gradient(p, self.h, axis=ax, edge_order=1) for ax in range(3)]
        dq = None
        if need_grad:
            dq = [np.gradient(q, self.h, axis=ax, edge_order=1) for ax in range(3)]

        k1, k2 = gorkov_constants(self.condition.frequency)
        u = k1 * np.abs(p) ** 2
        for component in dp:
            u -= k2 * np.abs(component) ** 2

        du = None
        if need_grad:
            du = 2.0 * k1 * np.real(np.conj(p)[..., None] * q)
            for component, component_q in zip(dp, dq):
                du -= 2.0 * k2 * np.real(np.conj(component)[..., None] * component_q)

        grad_u = [np.gradient(u, self.h, axis=ax, edge_order=1) for ax in range(3)]
        hess_u = np.empty((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                hess_u[i, j] = np.gradient(grad_u[i], self.h, axis=j, edge_order=1)[self.center]
        hess_u = 0.5 * (hess_u + hess_u.T)
        g = np.array([component[self.center] for component in grad_u], dtype=float)
        force_norm = float(np.linalg.norm(g))
        curvature = float(np.sum(self.condition.weight * hess_u))
        pressure = float(abs(p[self.center]))

        result: dict[str, np.ndarray | float] = {
            "p": p,
            "u": u,
            "grad_u": g,
            "hess_u": hess_u,
            "force_norm": force_norm,
            "curvature": curvature,
            "pressure": pressure,
        }
        if not need_grad:
            return result

        grad_du = [np.gradient(du, self.h, axis=ax, edge_order=1) for ax in range(3)]
        dg = np.stack([component[self.center] for component in grad_du], axis=0)
        d_hess = np.empty((3, 3, self.n), dtype=float)
        for i in range(3):
            for j in range(3):
                d_hess[i, j] = np.gradient(grad_du[i], self.h, axis=j, edge_order=1)[self.center]
        d_hess = 0.5 * (d_hess + np.swapaxes(d_hess, 0, 1))
        d_curvature = np.einsum("ij,ijn->n", self.condition.weight, d_hess, optimize=True)
        d_force = np.einsum("i,in->n", g, dg, optimize=True) / max(force_norm, 1.0e-20)
        p0 = p[self.center]
        q0 = q[self.center]
        d_pressure = np.real(np.conj(p0) * q0) / max(abs(p0), 1.0e-20)
        result.update(
            {
                "d_curvature": d_curvature,
                "d_force": d_force,
                "d_pressure": d_pressure,
            }
        )
        return result

    def fun_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        m = self._raw_metrics(x, need_grad=True)
        c_term = float(m["curvature"]) / self.scale_curvature
        f_term = float(m["force_norm"]) / self.scale_force
        p_term = float(m["pressure"]) / self.scale_pressure
        value = -c_term + self.alpha * f_term + self.beta * p_term
        full_grad = (
            -np.asarray(m["d_curvature"]) / self.scale_curvature
            + self.alpha * np.asarray(m["d_force"]) / self.scale_force
            + self.beta * np.asarray(m["d_pressure"]) / self.scale_pressure
        )
        return float(value), np.asarray(full_grad[1:], dtype=float)

    def metrics(self, x: np.ndarray) -> dict[str, np.ndarray | float]:
        return self._raw_metrics(x, need_grad=False)


def canonical_vortex_phase(condition: Condition) -> np.ndarray:
    positions = np.asarray(condition.positions)
    target = np.asarray(condition.target)
    focus = propagation_carrier(condition)
    azimuth = np.arctan2(positions[:, 1] - target[1], positions[:, 0] - target[0])
    return wrap_phase(focus + azimuth)


def propagation_carrier(condition: Condition) -> np.ndarray:
    """Geometrical propagation phase removed before comparing branches.

    Raw phase vectors are not expected to remain close when a target, frequency,
    or array geometry changes: the focusing carrier itself can rotate by many
    radians.  The branch coordinate is therefore the phase residual after the
    known eikonal carrier -k R_n has been removed.
    """
    positions = np.asarray(condition.positions)
    target = np.asarray(condition.target)
    distance = np.linalg.norm(target[None, :] - positions, axis=1)
    return -(2.0 * np.pi / condition.wavelength) * distance


def align_circular(delta: np.ndarray) -> np.ndarray:
    """Remove the best global phase from a circular phase difference."""
    delta = np.asarray(delta, dtype=float)
    gauge = np.angle(np.mean(np.exp(1j * delta)))
    return wrap_phase(delta - gauge)


def carrier_transport(
    previous_x: np.ndarray,
    previous_condition: Condition,
    new_condition: Condition,
) -> np.ndarray:
    """Transport the trap-specific mode while updating the known carrier."""
    previous_phi = gauge_full(previous_x)
    residual = wrap_phase(previous_phi - propagation_carrier(previous_condition))
    proposal = propagation_carrier(new_condition) + residual
    return reduce_gauge(proposal)


def solve(
    objective: LocalObjective,
    x0: np.ndarray,
    maxiter: int,
    gtol: float,
) -> tuple[object, float]:
    cache: dict[str, object] = {}

    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        key = np.asarray(x, dtype=float).tobytes()
        if cache.get("key") != key:
            value, grad = objective.fun_grad(x)
            cache.update({"key": key, "value": value, "grad": grad})
        return float(cache["value"]), np.asarray(cache["grad"], dtype=float)

    t0 = time.perf_counter()
    result = minimize(
        lambda x: fg(x)[0],
        np.asarray(x0, dtype=float),
        jac=lambda x: fg(x)[1],
        method="BFGS",
        options={"maxiter": int(maxiter), "gtol": float(gtol), "disp": False},
    )
    return result, time.perf_counter() - t0


def inverse_hessian_array(result: object, n: int) -> np.ndarray:
    value = getattr(result, "hess_inv", None)
    if value is None:
        return np.eye(n)
    if hasattr(value, "todense"):
        value = value.todense()
    value = np.asarray(value, dtype=float)
    if value.shape != (n, n) or not np.all(np.isfinite(value)):
        return np.eye(n)
    return 0.5 * (value + value.T)


def branch_predict(previous_x: np.ndarray, previous_hinv: np.ndarray, new_objective: LocalObjective) -> tuple[np.ndarray, float]:
    _, new_grad = new_objective.fun_grad(previous_x)
    step = -previous_hinv @ new_grad
    # Large predictor steps are a branch-risk signal.  Clip only the numerical
    # proposal; retain the unclipped RMS as the reported continuation stress.
    stress = float(np.linalg.norm(step) / math.sqrt(step.size))
    step_limit = 0.75
    rms = max(np.linalg.norm(step) / math.sqrt(step.size), TINY)
    if rms > step_limit:
        step = step * (step_limit / rms)
    return wrap_phase(previous_x + step), stress


def aligned_field_error(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    pv, qv = np.asarray(p).reshape(-1), np.asarray(q).reshape(-1)
    phase = np.angle(np.vdot(qv, pv))
    q_aligned = qv * np.exp(1j * phase)
    rel = float(np.linalg.norm(pv - q_aligned) / max(np.linalg.norm(pv), TINY))
    amp_corr = float(np.corrcoef(np.abs(pv), np.abs(q_aligned))[0, 1])
    return rel, amp_corr


def endpoint_row(
    condition: Condition,
    objective: LocalObjective,
    result: object,
    runtime: float,
    init_kind: str,
    predictor_stress: float,
    transported_x: np.ndarray | None,
    predicted_x: np.ndarray | None,
    previous_p: np.ndarray | None,
) -> dict[str, float | str | bool]:
    x = wrap_phase(np.asarray(result.x, dtype=float))
    value, grad = objective.fun_grad(x)
    metrics = objective.metrics(x)
    phase_step = np.nan
    predictor_error = np.nan
    carrier_residual_distance = np.nan
    carrier_residual_corr = np.nan
    field_error = np.nan
    amp_corr = np.nan
    if transported_x is not None:
        residual_delta = align_circular(circular_delta(x, transported_x))
        phase_step = float(np.percentile(np.abs(residual_delta), 95.0))
        carrier_residual_corr = float(abs(np.mean(np.exp(1j * residual_delta))))
        carrier_residual_distance = float(math.sqrt(max(0.0, 2.0 - 2.0 * carrier_residual_corr)))
    if predicted_x is not None:
        predictor_error = float(np.percentile(np.abs(circular_delta(x, predicted_x)), 95.0))
    if previous_p is not None:
        field_error, amp_corr = aligned_field_error(np.asarray(metrics["p"]), previous_p)

    hess_inv = inverse_hessian_array(result, x.size)
    eig = np.linalg.eigvalsh(hess_inv)
    positive = eig[eig > max(np.max(np.abs(eig)) * 1.0e-12, 1.0e-18)]
    inv_hess_cond = float(np.max(positive) / np.min(positive)) if positive.size > 1 else np.nan
    hess_u = np.asarray(metrics["hess_u"])
    hess_eig = np.linalg.eigvalsh(hess_u)
    sigma_stationarity = (
        condition.wavelength
        * float(metrics["force_norm"])
        / max(abs(float(np.trace(hess_u))), 1.0e-20)
    )
    return {
        "path": condition.path,
        "condition": condition.label,
        "s": condition.s,
        "init": init_kind,
        "target_x_m": condition.target[0],
        "target_y_m": condition.target[1],
        "target_z_m": condition.target[2],
        "frequency_hz": condition.frequency,
        "wavelength_m": condition.wavelength,
        "aperture_aspect": float(
            (np.ptp(condition.positions[:, 0]) + TINY) / (np.ptp(condition.positions[:, 1]) + TINY)
        ),
        "success_scipy": bool(result.success),
        "status": int(result.status),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "runtime_s": runtime,
        "objective": value,
        "grad_rms": float(np.linalg.norm(grad) / math.sqrt(grad.size)),
        "force_norm": float(metrics["force_norm"]),
        "curvature": float(metrics["curvature"]),
        "pressure": float(metrics["pressure"]),
        "stationarity_ratio": float(sigma_stationarity),
        "hess_u_min": float(hess_eig[0]),
        "hess_u_mid": float(hess_eig[1]),
        "hess_u_max": float(hess_eig[2]),
        "predictor_stress_rms_rad": predictor_stress,
        "predictor_error_p95_rad": predictor_error,
        "carrier_corrector_p95_rad": phase_step,
        "carrier_residual_distance": carrier_residual_distance,
        "carrier_residual_corr": carrier_residual_corr,
        "field_complex_rel_l2": field_error,
        "field_amplitude_corr": amp_corr,
        "bfgs_inverse_hessian_cond": inv_hess_cond,
    }


def make_paths(side: int) -> dict[str, list[Condition]]:
    base_positions = square_positions(side=side)
    base_target = (0.010, 0.010, 0.030)
    eye = np.eye(3)

    paths: dict[str, list[Condition]] = {}
    lateral = []
    for i, x_lambda in enumerate([0.0, 0.75, 1.5, 2.25, 3.0]):
        lam = C0 / FREQ0
        target = (base_target[0] + x_lambda * lam, base_target[1] - 0.5 * x_lambda * lam, base_target[2])
        lateral.append(Condition(f"lateral-{i}", "target", float(x_lambda), base_positions, target, FREQ0, eye))
    paths["target"] = lateral

    heights = []
    for z_lambda in [2.5, 3.5, 4.5, 5.5, 7.0]:
        lam = C0 / FREQ0
        heights.append(Condition(f"height-{z_lambda:.2f}", "height", z_lambda, base_positions, (base_target[0], base_target[1], z_lambda * lam), FREQ0, eye))
    paths["height"] = heights

    frequencies = []
    for frequency in [32_000.0, 36_000.0, 40_000.0, 44_000.0, 48_000.0]:
        frequencies.append(Condition(f"frequency-{frequency/1000:.0f}k", "frequency", frequency / 1000.0, base_positions, base_target, frequency, eye))
    paths["frequency"] = frequencies

    geometries = []
    for aspect in [2.0 / 3.0, 0.82, 1.0, 1.22, 1.5]:
        geometries.append(Condition(f"aspect-{aspect:.2f}", "geometry", aspect, square_positions(side, aspect=aspect), base_target, FREQ0, eye))
    paths["geometry"] = geometries

    noncanonical = []
    for t in [0.0, 0.25, 0.50, 0.75, 1.0]:
        weight = rotated_weight(30.0 * t, axis_tilt_deg=25.0 * t, anisotropy=t)
        noncanonical.append(Condition(f"oblique-{t:.2f}", "noncanonical", t, base_positions, base_target, FREQ0, weight))
    paths["noncanonical"] = noncanonical
    return paths


def run_path(
    conditions: list[Condition],
    maxiter: int,
    gtol: float,
    normalize_terms: bool,
    cold_extremes: int,
    seed: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    phases: dict[str, np.ndarray] = {}
    previous_x = None
    previous_condition = None
    previous_hinv = None
    previous_p = None
    rng = np.random.default_rng(seed)

    for index, condition in enumerate(conditions):
        reference = canonical_vortex_phase(condition)
        objective = LocalObjective(condition, normalize_terms=normalize_terms, reference_phase=reference)
        if previous_x is None:
            predicted_x = reduce_gauge(reference)
            transported_x = None
            stress = 0.0
            init_kind = "analytic-base"
        else:
            transported_x = carrier_transport(previous_x, previous_condition, condition)
            predicted_x, stress = branch_predict(transported_x, previous_hinv, objective)
            init_kind = "branch-predictor"
        result, runtime = solve(objective, predicted_x, maxiter=maxiter, gtol=gtol)
        row = endpoint_row(
            condition,
            objective,
            result,
            runtime,
            init_kind,
            stress,
            transported_x,
            predicted_x,
            previous_p,
        )
        rows.append(row)
        x = wrap_phase(np.asarray(result.x, dtype=float))
        phases[f"{condition.path}:{condition.label}:continuation"] = gauge_full(x)
        previous_x = x
        previous_condition = condition
        previous_hinv = inverse_hessian_array(result, x.size)
        previous_p = np.asarray(objective.metrics(x)["p"])
        print(
            f"[{condition.path}] {condition.label:<20s} nit={result.nit:4d} "
            f"grad_rms={row['grad_rms']:.2e} pred_err={row['predictor_error_p95_rad']:.3g} "
            f"t={runtime:.2f}s"
        )

        if index in {0, len(conditions) - 1} and cold_extremes > 0:
            for cold_idx in range(cold_extremes):
                if cold_idx == 0:
                    cold_x = reduce_gauge(reference)
                    cold_name = "analytic-cold"
                else:
                    cold_x = rng.uniform(-np.pi, np.pi, size=objective.n - 1)
                    cold_name = f"random-cold-{cold_idx}"
                cold_result, cold_runtime = solve(objective, cold_x, maxiter=maxiter, gtol=gtol)
                cold_row = endpoint_row(
                    condition,
                    objective,
                    cold_result,
                    cold_runtime,
                    cold_name,
                    np.nan,
                    x,
                    None,
                    previous_p,
                )
                rows.append(cold_row)
                phases[f"{condition.path}:{condition.label}:{cold_name}"] = gauge_full(wrap_phase(np.asarray(cold_result.x)))
    return rows, phases


def plot_results(df: pd.DataFrame, output: Path) -> None:
    cont = df[df["init"].isin(["analytic-base", "branch-predictor"])].copy()
    paths = [p for p in ["target", "height", "frequency", "geometry", "noncanonical"] if p in set(cont["path"])]
    fig, axes = plt.subplots(3, len(paths), figsize=(3.35 * len(paths), 8.2), squeeze=False)
    for col, path in enumerate(paths):
        sub = cont[cont["path"] == path].sort_values("s")
        x = sub["s"].to_numpy()
        axes[0, col].plot(x, sub["predictor_error_p95_rad"], "o-", label="corrector error")
        axes[0, col].plot(x, sub["predictor_stress_rms_rad"], "s--", label="predicted stress")
        axes[0, col].axhline(0.35, color="0.45", lw=1, ls=":")
        axes[0, col].set_title(path)
        axes[0, col].set_ylabel("phase (rad)")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].semilogy(x, np.maximum(sub["grad_rms"], 1e-16), "o-", label="phase grad RMS")
        axes[1, col].semilogy(x, np.maximum(sub["stationarity_ratio"], 1e-16), "s--", label="stationarity ratio")
        axes[1, col].grid(alpha=0.25)
        axes[1, col].set_ylabel("residual")
        axes[2, col].plot(x, sub["carrier_residual_corr"], "o-", label="carrier-removed mode corr")
        axes[2, col].plot(x, sub["carrier_residual_distance"], "s--", label="carrier-removed distance")
        axes[2, col].set_xlabel(path + " parameter")
        axes[2, col].set_ylabel("branch coordinate")
        axes[2, col].grid(alpha=0.25)
    handles_top, labels_top = axes[0, 0].get_legend_handles_labels()
    handles_bottom, labels_bottom = axes[2, 0].get_legend_handles_labels()
    fig.suptitle("Branch-informed predictor-corrector generalization", y=0.992, fontsize=14)
    fig.legend(handles_top + handles_bottom, labels_top + labels_bottom,
               loc="upper center", bbox_to_anchor=(0.5, 0.968), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(output / "branch_generalization_summary.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "branch_generalization_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cont = df[df["init"].isin(["analytic-base", "branch-predictor"])].copy()
    rows = []
    for path, group in cont.groupby("path", sort=False):
        corrected = group[group["init"] == "branch-predictor"]
        rows.append(
            {
                "path": path,
                "conditions": len(group),
                "median_nit": float(group["nit"].median()),
                "max_grad_rms": float(group["grad_rms"].max()),
                "max_predictor_error_p95_rad": float(corrected["predictor_error_p95_rad"].max()),
                "median_predictor_error_p95_rad": float(corrected["predictor_error_p95_rad"].median()),
                "min_carrier_residual_corr": float(corrected["carrier_residual_corr"].min()),
                "max_carrier_residual_distance": float(corrected["carrier_residual_distance"].max()),
                "branch_retention_fraction": float(
                    np.mean(
                        (corrected["carrier_residual_corr"] >= 0.90)
                        & (corrected["grad_rms"] <= 1.0e-5)
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--maxiter", type=int, default=350)
    parser.add_argument("--gtol", type=float, default=1.0e-7)
    parser.add_argument("--cold-extremes", type=int, default=2)
    parser.add_argument("--normalize-terms", action="store_true")
    parser.add_argument("--paths", nargs="*", default=["target", "height", "frequency", "geometry", "noncanonical"])
    parser.add_argument("--output", default="results_branch_generalization")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_paths = make_paths(args.side)
    all_rows: list[dict] = []
    all_phases: dict[str, np.ndarray] = {}
    for path_idx, path in enumerate(args.paths):
        rows, phases = run_path(
            all_paths[path],
            maxiter=args.maxiter,
            gtol=args.gtol,
            normalize_terms=args.normalize_terms,
            cold_extremes=args.cold_extremes,
            seed=42 + path_idx,
        )
        all_rows.extend(rows)
        all_phases.update(phases)

    df = pd.DataFrame(all_rows)
    df.to_csv(output / "branch_generalization_trials.csv", index=False)
    np.savez_compressed(output / "branch_generalization_phases.npz", **all_phases)
    summary = summarize(df)
    summary.to_csv(output / "branch_generalization_summary.csv", index=False)
    plot_results(df, output)
    metadata = {
        "array_side": args.side,
        "n_transducers": args.side**2,
        "maxiter": args.maxiter,
        "gtol": args.gtol,
        "normalize_terms": bool(args.normalize_terms),
        "cold_extremes": args.cold_extremes,
        "paths": args.paths,
        "success_rule": "carrier-removed circular mode correlation >= 0.90 and gradient RMS <= 1e-5",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nPath summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
