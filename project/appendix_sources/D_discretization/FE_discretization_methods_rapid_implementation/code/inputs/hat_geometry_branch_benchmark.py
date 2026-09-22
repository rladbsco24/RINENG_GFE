#!/usr/bin/env python3
"""Geometry-parametric HAT branch, convergence, and quantization benchmark.

This is an independent corrected benchmark, not a byte-for-byte reproduction of
the v45 notebook.  Its deliberate corrections are:

* every element has an explicit position and outward/inward normal;
* the Gor'kov gradient-energy term uses the standard negative sign;
* spatial derivatives use an explicit, uncontaminated central stencil;
* the global phase is removed before optimization;
* all stopping tests use a projected per-variable RMS gradient;
* geometry-dependent matrices are rebuilt and fingerprinted (no cross-array cache).

The acoustic field is linear in a_n = exp(i phi_n).  Every local Gor'kov
quantity is therefore represented by a Hermitian quadratic form, allowing exact
phase gradients and fast discrete phase refinement.
"""

from __future__ import annotations

import argparse
from rineng_content_id import content_identity
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import LogNorm
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import j1, logsumexp


FREQUENCY = 40_000.0
SOUND_SPEED = 343.0
WAVELENGTH = SOUND_SPEED / FREQUENCY
WAVENUMBER = 2.0 * np.pi / WAVELENGTH
ELEMENT_RADIUS = 0.005

RHO_AIR = 1.225
C_AIR = 343.0
RHO_PARTICLE = 100.0
C_PARTICLE = 2400.0
PARTICLE_DIAMETER = 0.0013
PARTICLE_VOLUME = 4.0 * np.pi * (PARTICLE_DIAMETER / 2.0) ** 3 / 3.0
OMEGA = 2.0 * np.pi * FREQUENCY

# Corrected standard-sign coefficients: U = K1 |p|^2 - K2 |grad p|^2.
K1 = 0.25 * PARTICLE_VOLUME * (
    1.0 / (RHO_AIR * C_AIR**2) - 1.0 / (RHO_PARTICLE * C_PARTICLE**2)
)
K2 = (
    0.75
    * PARTICLE_VOLUME
    * (RHO_PARTICLE - RHO_AIR)
    / (OMEGA**2 * RHO_AIR * (RHO_AIR + 2.0 * RHO_PARTICLE))
)

# Frozen dimensionless coefficients after geometry-wise random-phase calibration.
# PRESSURE_TERM_WEIGHT is chosen so the conventional optimum is driven onto the
# silent-center manifold instead of the high-pressure curvature maximum.  The raw
# force norm uses a smaller coefficient so its optimum remains off its own norm
# cusp; FORCE_SQUARED_TERM_WEIGHT is the explicitly smooth equilibrium control.
PRESSURE_TERM_WEIGHT = 25.0
FORCE_TERM_WEIGHT = 0.1
FORCE_SQUARED_TERM_WEIGHT = 1.0
BRANCH_PRESSURE_SQUARED_TERM_WEIGHT = 2.0
BRANCH_FORCE_SQUARED_TERM_WEIGHT = 0.1
BRANCH_STIFFNESS_WEIGHT = 1.0
BRANCH_SOFTMIN_TAU = 0.25


@dataclass(frozen=True)
class Geometry:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    target: np.ndarray
    family: str
    note: str

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class Profile:
    seeds: int
    homotopy_seeds: int
    homotopy_steps: int
    calibration_samples: int
    maxiter: int
    quant_refine_seeds: int
    quant_sweeps: int
    local_directions: int


PROFILES = {
    "smoke": Profile(2, 1, 3, 24, 220, 1, 1, 4),
    "standard": Profile(8, 3, 6, 48, 500, 4, 3, 10),
    "full": Profile(16, 5, 9, 96, 800, 8, 5, 24),
}


def unit_rows(x: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(d <= 0):
        raise ValueError("zero-length normal")
    return x / d


def square_points(side: int, pitch: float = 0.01) -> np.ndarray:
    c = (np.arange(side) - (side - 1) / 2.0) * pitch
    xx, yy = np.meshgrid(c, c, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(side * side)])


def fibonacci_disk(n: int, radius: float = 0.08) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    r = radius * np.sqrt((idx + 0.5) / n)
    theta = idx * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros(n)])


def best_candidate_disk(n: int, radius: float, seed: int, candidates: int = 320) -> np.ndarray:
    """Deterministic blue-noise-like arbitrary points inside a disk."""
    rng = np.random.default_rng(seed)
    pts = [np.array([0.0, 0.0])]
    while len(pts) < n:
        rr = radius * np.sqrt(rng.random(candidates))
        th = 2.0 * np.pi * rng.random(candidates)
        cand = np.column_stack([rr * np.cos(th), rr * np.sin(th)])
        p = np.asarray(pts)
        d2 = ((cand[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
        pts.append(cand[np.argmax(d2.min(axis=1))])
    xy = np.asarray(pts)
    return np.column_stack([xy, np.zeros(n)])


def cap_from_planar(points: np.ndarray, target_z: float = 0.13, radius: float = 0.13) -> tuple[np.ndarray, np.ndarray]:
    r2 = points[:, 0] ** 2 + points[:, 1] ** 2
    if np.max(r2) >= radius**2:
        raise ValueError("cap radius must exceed aperture radius")
    pos = points.copy()
    pos[:, 2] = target_z - np.sqrt(radius**2 - r2)
    target = np.array([0.0, 0.0, target_z])
    normals = unit_rows(target[None, :] - pos)
    return pos, normals


def canonical_geometries() -> list[Geometry]:
    target_single = np.array([0.0, 0.0, 0.13])
    sq = square_points(16)
    fib = fibonacci_disk(191)
    cap, cap_normals = cap_from_planar(fib)
    arb = best_candidate_disk(191, 0.08, seed=20260825)
    p0 = square_points(16)
    p1 = square_points(16)
    p1[:, 2] = 0.23
    opp = np.vstack([p0, p1])
    opp_normals = np.vstack(
        [np.tile([0.0, 0.0, 1.0], (256, 1)), np.tile([0.0, 0.0, -1.0], (256, 1))]
    )
    return [
        Geometry(
            "SQ256",
            sq,
            np.tile([0.0, 0.0, 1.0], (256, 1)),
            target_single,
            "single-sided planar lattice",
            "16x16, 10 mm pitch, 150 mm center-span",
        ),
        Geometry(
            "FIB191",
            fib,
            np.tile([0.0, 0.0, 1.0], (191, 1)),
            target_single,
            "single-sided aperiodic",
            "golden-angle/Fibonacci disk, 160 mm diameter",
        ),
        Geometry(
            "CAP191",
            cap,
            cap_normals,
            target_single,
            "single-sided curved",
            "Fibonacci labels projected to a 130 mm spherical cap",
        ),
        Geometry(
            "OPP512",
            opp,
            opp_normals,
            np.array([0.0, 0.0, 0.115]),
            "opposed planar",
            "two inward-facing 16x16 planes, 230 mm separation",
        ),
        Geometry(
            "ARB191",
            arb,
            np.tile([0.0, 0.0, 1.0], (191, 1)),
            target_single,
            "arbitrary stress test",
            "deterministic best-candidate disk, not a named hardware replica",
        ),
    ]


def geometry_content_id(g: Geometry) -> str:
    h = content_identity()
    for arr in (g.positions, g.normals, g.target):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(f"{FREQUENCY}|{ELEMENT_RADIUS}|standard-gorkov|explicit-fd".encode())
    return h.hexdigest()[:16]


def transfer_matrix(points: np.ndarray, g: Geometry) -> np.ndarray:
    delta = points[:, None, :] - g.positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    if np.any(distance <= 1e-8):
        raise ValueError("field point collides with an element")
    u = delta / distance[:, :, None]
    cos_theta = np.einsum("mni,ni->mn", u, g.normals)
    front = cos_theta > 0.0
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))
    z = WAVENUMBER * ELEMENT_RADIUS * sin_theta
    directivity = np.ones_like(z)
    nz = np.abs(z) > 1e-10
    directivity[nz] = 2.0 * j1(z[nz]) / z[nz]
    directivity[~front] = 0.0
    return directivity * np.exp(1j * WAVENUMBER * distance) / distance


def q_outer(row: np.ndarray) -> np.ndarray:
    return np.outer(np.conj(row), row)


@dataclass
class QuadraticModel:
    geometry: Geometry
    h: float
    q_curv: np.ndarray
    q_force: tuple[np.ndarray, np.ndarray, np.ndarray]
    q_hessian: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    q_pressure: np.ndarray
    q_prms: np.ndarray
    center_row: np.ndarray
    scales: dict[str, float]


def build_quadratic_model(g: Geometry, h: float, calibration_samples: int, seed: int) -> QuadraticModel:
    offsets = np.arange(-2, 3) * h
    grid = np.array(
        [[x, y, z] for x in offsets for y in offsets for z in offsets], dtype=float
    )
    points = g.target[None, :] + grid
    H = transfer_matrix(points, g).reshape(5, 5, 5, g.n)

    # U is needed only on the inner 3x3x3 nodes; pressure gradients there are
    # computed from +/- h samples, never from a boundary one-sided derivative.
    q_u: dict[tuple[int, int, int], np.ndarray] = {}
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                hp = H[i, j, k]
                dx = (H[i + 1, j, k] - H[i - 1, j, k]) / (2.0 * h)
                dy = (H[i, j + 1, k] - H[i, j - 1, k]) / (2.0 * h)
                dz = (H[i, j, k + 1] - H[i, j, k - 1]) / (2.0 * h)
                q_u[(i, j, k)] = K1 * q_outer(hp) - K2 * (
                    q_outer(dx) + q_outer(dy) + q_outer(dz)
                )

    q0 = q_u[(2, 2, 2)]
    qx = (q_u[(3, 2, 2)] - q_u[(1, 2, 2)]) / (2.0 * h)
    qy = (q_u[(2, 3, 2)] - q_u[(2, 1, 2)]) / (2.0 * h)
    qz = (q_u[(2, 2, 3)] - q_u[(2, 2, 1)]) / (2.0 * h)
    qxx = (q_u[(3, 2, 2)] - 2.0 * q0 + q_u[(1, 2, 2)]) / h**2
    qyy = (q_u[(2, 3, 2)] - 2.0 * q0 + q_u[(2, 1, 2)]) / h**2
    qzz = (q_u[(2, 2, 3)] - 2.0 * q0 + q_u[(2, 2, 1)]) / h**2
    qxy = (
        q_u[(3, 3, 2)] - q_u[(3, 1, 2)] - q_u[(1, 3, 2)] + q_u[(1, 1, 2)]
    ) / (4.0 * h**2)
    qxz = (
        q_u[(3, 2, 3)] - q_u[(3, 2, 1)] - q_u[(1, 2, 3)] + q_u[(1, 2, 1)]
    ) / (4.0 * h**2)
    qyz = (
        q_u[(2, 3, 3)] - q_u[(2, 3, 1)] - q_u[(2, 1, 3)] + q_u[(2, 1, 1)]
    ) / (4.0 * h**2)
    # Vortex-family benchmark: equal curvature weights in all three axes.
    q_curv = qxx + qyy + qzz
    center_row = H[2, 2, 2].copy()
    q_pressure = q_outer(center_row)
    flat = H.reshape(-1, g.n)
    q_prms = (flat.conj().T @ flat) / flat.shape[0]

    # Numerical Hermitian cleanup prevents sub-ulp imaginary leakage.
    mats = [q_curv, qx, qy, qz, qxx, qyy, qzz, qxy, qxz, qyz, q_pressure, q_prms]
    mats = [(q + q.conj().T) / 2.0 for q in mats]
    (
        q_curv,
        qx,
        qy,
        qz,
        qxx,
        qyy,
        qzz,
        qxy,
        qxz,
        qyz,
        q_pressure,
        q_prms,
    ) = mats

    rng = np.random.default_rng(seed)
    curv, force, pressure, stiffness = [], [], [], []
    for _ in range(calibration_samples):
        a = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, g.n))
        curv.append(abs(q_value(a, q_curv)))
        force.append(np.linalg.norm([q_value(a, q) for q in (qx, qy, qz)]))
        pressure.append(math.sqrt(max(q_value(a, q_pressure), 0.0)))
        hv = [q_value(a, q) for q in (qxx, qyy, qzz, qxy, qxz, qyz)]
        hh = np.array(
            [[hv[0], hv[3], hv[4]], [hv[3], hv[1], hv[5]], [hv[4], hv[5], hv[2]]]
        )
        stiffness.append(np.max(np.abs(np.linalg.eigvalsh(hh))))
    scales = {
        "curvature": max(float(np.median(curv)), np.finfo(float).tiny),
        "force": max(float(np.median(force)), np.finfo(float).tiny),
        "pressure": max(float(np.median(pressure)), np.finfo(float).tiny),
        "stiffness": max(float(np.median(stiffness)), np.finfo(float).tiny),
    }
    model = QuadraticModel(
        g,
        h,
        q_curv,
        (qx, qy, qz),
        (qxx, qyy, qzz, qxy, qxz, qyz),
        q_pressure,
        q_prms,
        center_row,
        scales,
    )

    # Exact factor representation of the same finite-difference quadratic forms.
    # No eigenvalue truncation or alternative field discretization is introduced.
    nodes = [(i, j, k) for i in range(1, 4) for j in range(1, 4) for k in range(1, 4)]
    rows = []
    for i, j, k in nodes:
        rows.extend((H[i, j, k], (H[i+1, j, k]-H[i-1, j, k])/(2*h),
                     (H[i, j+1, k]-H[i, j-1, k])/(2*h),
                     (H[i, j, k+1]-H[i, j, k-1])/(2*h)))
    model.compact_rows = np.asarray(rows)
    def u(node):
        coefficients = np.zeros(len(nodes))
        coefficients[nodes.index(node)] = 1.
        return coefficients
    center = u((2,2,2))
    first = [(u(tuple(2+int(a==axis) for a in range(3))) -
              u(tuple(2-int(a==axis) for a in range(3))))/(2*h) for axis in range(3)]
    diagonal = [(u(tuple(2+int(a==axis) for a in range(3))) - 2*center +
                 u(tuple(2-int(a==axis) for a in range(3))))/h**2 for axis in range(3)]
    mixed = []
    for axis, other in ((0,1),(0,2),(1,2)):
        coefficient = np.zeros(len(nodes))
        for left, right in ((1,1),(1,-1),(-1,1),(-1,-1)):
            node = [2,2,2]; node[axis] += left; node[other] += right
            coefficient += left*right*u(tuple(node))/(4*h**2)
        mixed.append(coefficient)
    factors = [np.sum(diagonal,axis=0), *first, *diagonal, *mixed]
    matrices = [model.q_curv, *model.q_force, *model.q_hessian]
    model.compact_coefficients = {id(matrix): np.repeat(coefficient,4)*np.tile([K1,-K2,-K2,-K2],len(nodes))
                                  for matrix, coefficient in zip(matrices,factors)}
    return model


def q_value(a: np.ndarray, q: np.ndarray) -> float:
    return float(np.real(np.vdot(a, q @ a)))


def q_gradient(a: np.ndarray, qa: np.ndarray) -> np.ndarray:
    return 2.0 * np.imag(np.conj(a) * qa)


class Objective:
    def __init__(self, model: QuadraticModel, method: str, eps_rel: float = 1e-3):
        self.model = model
        self.method = method
        self.eps_rel = eps_rel
        self.last_x: np.ndarray | None = None
        self.last_fg: tuple[float, np.ndarray] | None = None
        self.evaluations = 0

    def full_phase(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.asarray(x, dtype=float)])

    def evaluate_phase(self, phase: np.ndarray) -> tuple[float, np.ndarray]:
        m = self.model
        a = np.exp(1j * phase)
        use_compact = self.method in {"force_equilibrium", "force_squared", "branch_hybrid"}
        amplitudes = m.compact_rows @ a if use_compact else None
        def apply(matrix):
            if use_compact:
                return m.compact_rows.conj().T @ (m.compact_coefficients[id(matrix)] * amplitudes)
            return matrix @ a
        yc = apply(m.q_curv)
        curv = float(np.real(np.vdot(a, yc)))
        gc = q_gradient(a, yc)
        yp = np.conj(m.center_row) * (m.center_row @ a) if use_compact else m.q_pressure @ a
        p2 = max(float(np.real(np.vdot(a, yp))), 0.0)
        gp2 = q_gradient(a, yp)
        ys = [apply(q) for q in m.q_force]
        forces = np.array([float(np.real(np.vdot(a, y))) for y in ys])
        gforces = np.vstack([q_gradient(a, y) for y in ys])

        f = -curv / m.scales["curvature"]
        grad = -gc / m.scales["curvature"]
        if self.method == "conventional":
            p = math.sqrt(max(p2, 1e-300))
            f += PRESSURE_TERM_WEIGHT * p / m.scales["pressure"]
            grad += PRESSURE_TERM_WEIGHT * 0.5 * gp2 / (p * m.scales["pressure"])
        elif self.method == "regularized":
            floor = self.eps_rel * m.scales["pressure"]
            p = math.sqrt(p2 + floor**2)
            f += PRESSURE_TERM_WEIGHT * p / m.scales["pressure"]
            grad += PRESSURE_TERM_WEIGHT * 0.5 * gp2 / (p * m.scales["pressure"])
        elif self.method == "force_equilibrium":
            fnorm = float(np.linalg.norm(forces))
            safe = max(fnorm, 1e-300)
            f += FORCE_TERM_WEIGHT * fnorm / m.scales["force"]
            grad += FORCE_TERM_WEIGHT * (forces[:, None] * gforces).sum(axis=0) / (safe * m.scales["force"])
        elif self.method == "force_squared":
            scaled = forces / m.scales["force"]
            f += 0.5 * FORCE_SQUARED_TERM_WEIGHT * float(np.dot(scaled, scaled))
            grad += FORCE_SQUARED_TERM_WEIGHT * (forces[:, None] * gforces).sum(axis=0) / m.scales["force"] ** 2
        elif self.method == "branch_hybrid":
            # Replace trace maximization by a differentiable minimum-stiffness
            # objective.  This prevents a large positive trace from hiding one
            # unstable axis on irregular arrays.
            yh = [apply(q) for q in m.q_hessian]
            hv = [float(np.real(np.vdot(a, y))) for y in yh]
            hh = np.array(
                [
                    [hv[0], hv[3], hv[4]],
                    [hv[3], hv[1], hv[5]],
                    [hv[4], hv[5], hv[2]],
                ]
            )
            eig, vec = np.linalg.eigh(hh)
            eig_scaled = eig / m.scales["stiffness"]
            logits = -eig_scaled / BRANCH_SOFTMIN_TAU
            weights = np.exp(logits - logsumexp(logits))
            softmin = -BRANCH_SOFTMIN_TAU * logsumexp(logits)
            grad_softmin = np.zeros_like(grad)
            for ei in range(3):
                v = vec[:, ei]
                yv = (
                    v[0] ** 2 * yh[0]
                    + v[1] ** 2 * yh[1]
                    + v[2] ** 2 * yh[2]
                    + 2.0 * v[0] * v[1] * yh[3]
                    + 2.0 * v[0] * v[2] * yh[4]
                    + 2.0 * v[1] * v[2] * yh[5]
                )
                grad_softmin += weights[ei] * q_gradient(a, yv) / m.scales["stiffness"]
            f += curv / m.scales["curvature"] - BRANCH_STIFFNESS_WEIGHT * softmin
            grad += gc / m.scales["curvature"] - BRANCH_STIFFNESS_WEIGHT * grad_softmin
            f += (
                0.5
                * BRANCH_PRESSURE_SQUARED_TERM_WEIGHT
                * p2
                / m.scales["pressure"] ** 2
            )
            grad += (
                0.5
                * BRANCH_PRESSURE_SQUARED_TERM_WEIGHT
                * gp2
                / m.scales["pressure"] ** 2
            )
            scaled = forces / m.scales["force"]
            f += 0.5 * BRANCH_FORCE_SQUARED_TERM_WEIGHT * float(np.dot(scaled, scaled))
            grad += (
                BRANCH_FORCE_SQUARED_TERM_WEIGHT
                * (forces[:, None] * gforces).sum(axis=0)
                / m.scales["force"] ** 2
            )
        else:
            raise ValueError(f"unknown method: {self.method}")
        return float(f), np.asarray(grad, dtype=float)

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        if self.last_x is not None and np.array_equal(x, self.last_x):
            assert self.last_fg is not None
            return self.last_fg
        phase = self.full_phase(x)
        f, grad = self.evaluate_phase(phase)
        self.last_x = x.copy()
        self.last_fg = (f, grad[1:].copy())
        self.evaluations += 1
        return self.last_fg


def optimize(
    model: QuadraticModel,
    method: str,
    phase0: np.ndarray,
    maxiter: int,
    gtol: float = 1e-7,
    keep_history: bool = False,
) -> tuple[dict, np.ndarray, list[dict]]:
    phase0 = np.angle(np.exp(1j * (phase0 - phase0[0])))
    obj = Objective(model, method)
    history: list[dict] = []
    start = time.perf_counter()

    def callback(xk: np.ndarray) -> None:
        if not keep_history:
            return
        f, grad = obj(xk)
        history.append(
            {
                "iteration": len(history) + 1,
                "objective": f,
                "gradient_rms": float(np.linalg.norm(grad) / np.sqrt(grad.size)),
                "evaluations": obj.evaluations,
                "elapsed_s": time.perf_counter() - start,
            }
        )

    use_fe = method in {"force_equilibrium", "force_squared", "branch_hybrid"}
    solver = "L-BFGS-B" if use_fe else "BFGS"
    options = {"maxiter": maxiter, "gtol": gtol}
    if use_fe:
        options.update(ftol=0., maxls=100, maxcor=10, maxfun=max(15000, 100*maxiter))
    result = minimize(
        obj,
        phase0[1:],
        method=solver,
        jac=True,
        callback=callback,
        options=options,
    )
    elapsed = time.perf_counter() - start
    phase = np.concatenate([[0.0], result.x])
    phase = np.angle(np.exp(1j * phase))
    f, grad = Objective(model, method).evaluate_phase(phase)
    grad_rms = float(np.linalg.norm(grad[1:]) / np.sqrt(model.geometry.n - 1))
    row = {
        "method": method,
        "optimizer": solver,
        "evaluator_backend": "compact-piston-stencil-v1" if use_fe else "dense-quadratic",
        "objective": f,
        "gradient_rms": grad_rms,
        "stationary": bool(np.isfinite(f) and grad_rms <= 1e-6),
        "scipy_success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(obj.evaluations),
        "wall_s": float(elapsed),
    }
    row.update(physical_metrics(model, phase))
    return row, phase, history


def physical_metrics(model: QuadraticModel, phase: np.ndarray) -> dict:
    a = np.exp(1j * phase)
    force = np.array([q_value(a, q) for q in model.q_force])
    qxx, qyy, qzz, qxy, qxz, qyz = model.q_hessian
    vals = [q_value(a, q) for q in (qxx, qyy, qzz, qxy, qxz, qyz)]
    hess = np.array(
        [[vals[0], vals[3], vals[4]], [vals[3], vals[1], vals[5]], [vals[4], vals[5], vals[2]]]
    )
    eig = np.linalg.eigvalsh(hess)
    pressure = abs(model.center_row @ a)
    return {
        "pressure": float(pressure),
        "pressure_scaled": float(pressure / model.scales["pressure"]),
        "force_norm": float(np.linalg.norm(force)),
        "force_scaled": float(np.linalg.norm(force) / model.scales["force"]),
        "stiffness_min": float(eig[0]),
        "stiffness_mid": float(eig[1]),
        "stiffness_max": float(eig[2]),
        "morse_index": int(np.sum(eig < 0.0)),
        "stable_hessian": bool(eig[0] > 0.0),
    }


def phase_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.mean(np.exp(1j * (a - b)))))


def circular_rmse(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.angle(np.mean(np.exp(1j * (a - b))))
    err = np.angle(np.exp(1j * (a - b - delta)))
    return float(np.sqrt(np.mean(err**2)))


def common_field_points(target: np.ndarray) -> np.ndarray:
    xy = np.linspace(-1.5, 1.5, 9) * WAVELENGTH
    zz = np.linspace(-1.0, 1.0, 5) * WAVELENGTH
    offsets = np.array([[x, y, z] for z in zz for x in xy for y in xy])
    # Down-weight neither the core nor annulus; normalization removes gain.
    return target[None, :] + offsets


def normalized_field(g: Geometry, phase: np.ndarray) -> np.ndarray:
    p = transfer_matrix(common_field_points(g.target), g) @ np.exp(1j * phase)
    norm = np.linalg.norm(p)
    return p / max(norm, np.finfo(float).tiny)


def field_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)))


def vortex_winding(g: Geometry, phase: np.ndarray, radius: float = 0.75 * WAVELENGTH) -> int:
    # Opposed arrays are evaluated as a standing-node class; a loop winding at a
    # near-zero nodal plane is numerically ill-conditioned and is not a vortex label.
    if g.family == "opposed planar":
        return 0
    theta = np.linspace(0.0, 2.0 * np.pi, 129)
    points = g.target[None, :] + np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)]
    )
    p = transfer_matrix(points, g) @ np.exp(1j * phase)
    d = np.angle(p[1:] * np.conj(p[:-1]))
    return int(np.rint(d.sum() / (2.0 * np.pi)))


def projector_prototype(fields: list[np.ndarray]) -> np.ndarray:
    if not fields:
        raise ValueError("no fields for prototype")
    stack = np.vstack(fields)
    _, _, vh = np.linalg.svd(stack, full_matrices=False)
    proto = vh[0].conj()
    return proto / np.linalg.norm(proto)


def project_pressure_zero(model: QuadraticModel, phase: np.ndarray, iterations: int = 20) -> np.ndarray:
    x = np.asarray(phase, dtype=float).copy()
    x -= x[0]
    h = model.center_row
    for _ in range(iterations):
        a = np.exp(1j * x)
        p = h @ a
        if abs(p) <= 1e-11 * model.scales["pressure"]:
            break
        dp = 1j * h[1:] * a[1:]
        jac = np.vstack([dp.real, dp.imag])
        gram = jac @ jac.T
        rhs = np.array([p.real, p.imag])
        correction = -jac.T @ np.linalg.solve(gram + 1e-18 * np.eye(2), rhs)
        # Damping makes the projection robust from non-asymptotic endpoints.
        step = min(1.0, 0.5 / max(np.linalg.norm(correction), 1e-12))
        x[1:] += step * correction
    return np.angle(np.exp(1j * x))


def fit_local_exponents(
    model: QuadraticModel, base_phase: np.ndarray, directions: int, seed: int
) -> dict[str, float]:
    base = project_pressure_zero(model, base_phase)
    rng = np.random.default_rng(seed)
    methods = ["conventional", "regularized", "force_equilibrium", "force_squared", "branch_hybrid"]
    alphas: dict[str, list[float]] = {m: [] for m in methods}
    r2s: dict[str, list[float]] = {m: [] for m in methods}
    a = np.exp(1j * base)
    dpvec = 1j * model.center_row[1:] * a[1:]
    for _ in range(directions):
        v = rng.normal(size=base.size - 1)
        v /= np.linalg.norm(v)
        dp_scale = abs(np.dot(dpvec, v)) / model.scales["pressure"]
        if dp_scale < 1e-10:
            continue
        for method in methods:
            # Conventional samples its asymptotic |t| regime.  Smoothed methods
            # sample below their 1e-3 pressure floor.
            target_delta = (
                np.logspace(-8, -4, 9) if method == "conventional" else np.logspace(-9, -5, 9)
            )
            tvals = np.clip(target_delta / dp_scale, 1e-9, 3e-2)
            tvals = np.unique(tvals)
            if tvals.size < 5:
                continue
            obj = Objective(model, method)
            f0, _ = obj.evaluate_phase(base)
            sval, tkeep = [], []
            for t in tvals:
                pp = base.copy()
                pm = base.copy()
                pp[1:] += t * v
                pm[1:] -= t * v
                fp, _ = obj.evaluate_phase(pp)
                fm, _ = obj.evaluate_phase(pm)
                s = 0.5 * (fp + fm) - f0
                if np.isfinite(s) and s > 1e-14:
                    sval.append(s)
                    tkeep.append(t)
            if len(sval) < 5:
                continue
            lx, ly = np.log(tkeep), np.log(sval)
            slope, intercept = np.polyfit(lx, ly, 1)
            pred = slope * lx + intercept
            denom = np.sum((ly - np.mean(ly)) ** 2)
            r2 = 1.0 - np.sum((ly - pred) ** 2) / max(denom, 1e-30)
            alphas[method].append(float(slope))
            r2s[method].append(float(r2))
    out = {"projected_pressure_scaled": abs(model.center_row @ np.exp(1j * base)) / model.scales["pressure"]}
    for method in methods:
        out[f"alpha_{method}"] = float(np.median(alphas[method])) if alphas[method] else np.nan
        out[f"alpha_{method}_r2"] = float(np.median(r2s[method])) if r2s[method] else np.nan
    return out


def objective_value_from_q(method: str, qvals: np.ndarray, scales: dict[str, float]) -> float:
    curv, gx, gy, gz, p2 = qvals[:5]
    if method == "force_equilibrium":
        return float(-curv / scales["curvature"] + FORCE_TERM_WEIGHT * np.linalg.norm([gx, gy, gz]) / scales["force"])
    if method == "force_squared":
        f = np.linalg.norm([gx, gy, gz]) / scales["force"]
        return float(-curv / scales["curvature"] + 0.5 * FORCE_SQUARED_TERM_WEIGHT * f**2)
    if method == "branch_hybrid":
        fs = np.linalg.norm([gx, gy, gz]) / scales["force"]
        ps2 = p2 / scales["pressure"] ** 2
        hxx, hyy, hzz, hxy, hxz, hyz = qvals[5:11]
        hh = np.array(
            [[hxx, hxy, hxz], [hxy, hyy, hyz], [hxz, hyz, hzz]], dtype=float
        )
        eig_scaled = np.linalg.eigvalsh(hh) / scales["stiffness"]
        softmin = -BRANCH_SOFTMIN_TAU * logsumexp(-eig_scaled / BRANCH_SOFTMIN_TAU)
        return float(
            -BRANCH_STIFFNESS_WEIGHT * softmin
            + 0.5 * BRANCH_PRESSURE_SQUARED_TERM_WEIGHT * ps2
            + 0.5 * BRANCH_FORCE_SQUARED_TERM_WEIGHT * fs**2
        )
    raise ValueError(method)


def discrete_refine(
    model: QuadraticModel,
    phase: np.ndarray,
    levels: int,
    sweeps: int,
    seed: int,
    method: str = "branch_hybrid",
) -> tuple[np.ndarray, float, int]:
    step = 2.0 * np.pi / levels
    idx = np.mod(np.rint(phase / step).astype(int), levels)
    idx -= idx[0]
    idx %= levels
    a = np.exp(1j * step * idx)
    mats = [model.q_curv, *model.q_force, model.q_pressure, *model.q_hessian]
    ys = [q @ a for q in mats]
    qvals = np.array([float(np.real(np.vdot(a, y))) for y in ys])
    current = objective_value_from_q(method, qvals, model.scales)
    rng = np.random.default_rng(seed)
    evals = 0
    for _ in range(sweeps):
        improved = False
        for n in rng.permutation(np.arange(1, model.geometry.n)):
            best = current
            best_sign = 0
            best_q = qvals
            old = a[n]
            for sign in (-1, 1):
                new = old * np.exp(1j * sign * step)
                delta = new - old
                candidate_q = np.array(
                    [
                        qvals[j]
                        + 2.0 * np.real(np.conj(delta) * ys[j][n])
                        + abs(delta) ** 2 * np.real(mats[j][n, n])
                        for j in range(len(mats))
                    ]
                )
                candidate = objective_value_from_q(method, candidate_q, model.scales)
                evals += 1
                if candidate < best - 1e-12:
                    best, best_sign, best_q = candidate, sign, candidate_q
            if best_sign:
                new = old * np.exp(1j * best_sign * step)
                delta = new - old
                a[n] = new
                idx[n] = (idx[n] + best_sign) % levels
                for j, q in enumerate(mats):
                    ys[j] += q[:, n] * delta
                qvals = best_q
                current = best
                improved = True
        if not improved:
            break
    return np.angle(a), float(current), evals


def reduced_hessian_ratio(model: QuadraticModel, method: str, phase: np.ndarray, step: float = 2e-5) -> tuple[float, float]:
    x = phase[1:].copy()
    n = x.size
    hess = np.empty((n, n), dtype=float)
    obj = Objective(model, method)
    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += step
        xm[j] -= step
        _, gp = obj(xp)
        _, gm = obj(xm)
        hess[:, j] = (gp - gm) / (2.0 * step)
    hess = (hess + hess.T) / 2.0
    eig = np.linalg.eigvalsh(hess)
    scale = max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
    return float(np.min(np.abs(eig)) / scale), float(np.min(eig))


def match_labels(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cost = ((source[:, None, :2] - target[None, :, :2]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty_like(target)
    out[rows] = target[cols]
    return out


def homotopy_endpoints(n: int = 196) -> dict[str, tuple[Geometry, Geometry]]:
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError("homotopy n must be square")
    target = np.array([0.0, 0.0, 0.13])
    square = square_points(side)
    fib_raw = fibonacci_disk(n, radius=0.075)
    fib = match_labels(square, fib_raw)
    arb_raw = best_candidate_disk(n, 0.075, seed=777)
    arb = match_labels(fib, arb_raw)
    cap, cap_normals = cap_from_planar(fib, target_z=0.13, radius=0.13)
    up = np.tile([0.0, 0.0, 1.0], (n, 1))
    gsq = Geometry("H_SQ196", square, up, target, "homotopy", "14x14 start")
    gfib = Geometry("H_FIB196", fib, up, target, "homotopy", "label-matched Fibonacci")
    gcap = Geometry("H_CAP196", cap, cap_normals, target, "homotopy", "curved endpoint")
    garb = Geometry("H_ARB196", arb, up, target, "homotopy", "label-matched arbitrary endpoint")
    return {
        "square_to_fibonacci": (gsq, gfib),
        "fibonacci_to_cap": (gfib, gcap),
        "fibonacci_to_arbitrary": (gfib, garb),
    }


def interpolate_geometry(a: Geometry, b: Geometry, t: float, name: str) -> Geometry:
    pos = (1.0 - t) * a.positions + t * b.positions
    normals = unit_rows((1.0 - t) * a.normals + t * b.normals)
    target = (1.0 - t) * a.target + t * b.target
    return Geometry(name, pos, normals, target, "controlled homotopy", f"t={t:.6f}")


def run_canonical(
    outdir: Path, profile: Profile, h: float, rng_seed: int
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], np.ndarray], dict[str, QuadraticModel], pd.DataFrame]:
    rows: list[dict] = []
    endpoints: dict[tuple[str, str, int], np.ndarray] = {}
    models: dict[str, QuadraticModel] = {}
    local_rows: list[dict] = []
    methods = ["conventional", "regularized", "force_equilibrium", "force_squared", "branch_hybrid"]
    for gi, g in enumerate(canonical_geometries()):
        print(f"[canonical] build {g.name} N={g.n}", flush=True)
        model = build_quadratic_model(g, h, profile.calibration_samples, rng_seed + 100 * gi)
        models[g.name] = model
        for seed_index in range(profile.seeds):
            rng = np.random.default_rng(rng_seed + 10_000 * gi + seed_index)
            phase0 = rng.uniform(-np.pi, np.pi, g.n)
            for method in methods:
                print(f"[canonical] {g.name} seed={seed_index} {method}", flush=True)
                row, phase, _ = optimize(model, method, phase0, profile.maxiter)
                row.update(
                    {
                        "geometry": g.name,
                        "geometry_family": g.family,
                        "geometry_content_id": geometry_content_id(g),
                        "n_elements": g.n,
                        "seed": seed_index,
                        "winding": vortex_winding(g, phase),
                        "scale_curvature": model.scales["curvature"],
                        "scale_force": model.scales["force"],
                        "scale_pressure": model.scales["pressure"],
                    }
                )
                rows.append(row)
                endpoints[(g.name, method, seed_index)] = phase
            if seed_index == 0:
                local = fit_local_exponents(
                    model,
                    endpoints[(g.name, "conventional", seed_index)],
                    profile.local_directions,
                    rng_seed + 90_000 + gi,
                )
                local.update({"geometry": g.name, "n_elements": g.n})
                local_rows.append(local)
    df = pd.DataFrame(rows)

    # Geometry-specific squared-residual branch prototypes. No forced k clustering.
    fields: dict[tuple[str, str, int], np.ndarray] = {}
    for key, phase in endpoints.items():
        fields[key] = normalized_field(models[key[0]].geometry, phase)
    prototypes: dict[str, np.ndarray] = {}
    branch_fields: dict[str, list[np.ndarray]] = {}
    for gname in models:
        candidates = [
            fields[(gname, "branch_hybrid", s)]
            for s in range(profile.seeds)
            if bool(
                df.loc[
                    (df.geometry == gname)
                    & (df.method == "branch_hybrid")
                    & (df.seed == s),
                    "stationary",
                ].iloc[0]
            )
        ]
        if not candidates:
            candidates = [fields[(gname, "branch_hybrid", s)] for s in range(profile.seeds)]
        branch_fields[gname] = candidates
        prototypes[gname] = projector_prototype(candidates)

    branch_sim, phase_fe_sim, phase_fe_rmse = [], [], []
    for _, row in df.iterrows():
        key = (row.geometry, row.method, int(row.seed))
        branch_sim.append(max(field_similarity(fields[key], f) for f in branch_fields[row.geometry]))
        fephase = endpoints[(row.geometry, "branch_hybrid", int(row.seed))]
        phase_fe_sim.append(phase_similarity(endpoints[key], fephase))
        phase_fe_rmse.append(circular_rmse(endpoints[key], fephase))
    df["field_similarity_to_branch_set"] = branch_sim
    df["phase_similarity_to_paired_branch"] = phase_fe_sim
    df["phase_rmse_to_paired_branch"] = phase_fe_rmse
    df["branch_reached"] = df.field_similarity_to_branch_set >= 0.95
    df["certified"] = df.stationary & df.stable_hessian & df.branch_reached

    # Cross-array recurrence refers to physical branch class, never literal same branch.
    reference = [
        (
            fields[("SQ256", "branch_hybrid", s)],
            int(
                df.loc[
                    (df.geometry == "SQ256")
                    & (df.method == "branch_hybrid")
                    & (df.seed == s),
                    "winding",
                ].iloc[0]
            ),
        )
        for s in range(profile.seeds)
    ]
    cross_rows = []
    for gname in prototypes:
        scores = []
        winding_set = []
        for s in range(profile.seeds):
            w = int(
                df.loc[
                    (df.geometry == gname)
                    & (df.method == "branch_hybrid")
                    & (df.seed == s),
                    "winding",
                ].iloc[0]
            )
            winding_set.append(w)
            same_chirality = [rf for rf, rw in reference if rw == w]
            if same_chirality:
                scores.append(
                    max(
                        field_similarity(fields[(gname, "branch_hybrid", s)], rf)
                        for rf in same_chirality
                    )
                )
        cross_rows.append(
            {
                "geometry": gname,
                "field_similarity_to_SQ256_same_winding_median": (
                    float(np.median(scores)) if scores else np.nan
                ),
                "field_similarity_to_SQ256_same_winding_max": (
                    float(np.max(scores)) if scores else np.nan
                ),
                "observed_winding_set": ",".join(map(str, sorted(set(winding_set)))),
                "same_literal_branch_claim": False,
            }
        )
    return df, endpoints, models, pd.DataFrame(local_rows).merge(pd.DataFrame(cross_rows), on="geometry")


def run_quantization(
    df: pd.DataFrame,
    endpoints: dict[tuple[str, str, int], np.ndarray],
    models: dict[str, QuadraticModel],
    profile: Profile,
    rng_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qrows: list[dict] = []
    rrows: list[dict] = []
    levels_all = [8, 10, 16, 32, 64]
    for _, run in df.iterrows():
        key = (run.geometry, run.method, int(run.seed))
        model = models[run.geometry]
        phase = endpoints[key]
        field0 = normalized_field(model.geometry, phase)
        obj = Objective(model, run.method)
        f0, _ = obj.evaluate_phase(phase)
        for levels in levels_all:
            step = 2.0 * np.pi / levels
            qphase = np.angle(np.exp(1j * step * np.rint(phase / step)))
            fq, _ = obj.evaluate_phase(qphase)
            metrics = physical_metrics(model, qphase)
            qrows.append(
                {
                    "geometry": run.geometry,
                    "method": run.method,
                    "seed": int(run.seed),
                    "levels": levels,
                    "phase_step_deg": 360.0 / levels,
                    "objective_continuous": f0,
                    "objective_quantized": fq,
                    "objective_increase": fq - f0,
                    "phase_similarity": phase_similarity(phase, qphase),
                    "field_similarity": field_similarity(field0, normalized_field(model.geometry, qphase)),
                    "winding_continuous": int(run.winding),
                    "winding_quantized": vortex_winding(model.geometry, qphase),
                    **metrics,
                }
            )

    # Same-budget discrete refinement at the challenging 10-level setting.
    for gi, (gname, model) in enumerate(models.items()):
        for seed_index in range(min(profile.quant_refine_seeds, profile.seeds)):
            fephase = endpoints[(gname, "branch_hybrid", seed_index)]
            continuous_field = normalized_field(model.geometry, fephase)
            for init in ("continuous_branch", "random"):
                if init == "continuous_branch":
                    phase0 = fephase
                else:
                    rng = np.random.default_rng(rng_seed + 700_000 + gi * 100 + seed_index)
                    phase0 = rng.integers(0, 10, model.geometry.n) * (2.0 * np.pi / 10.0)
                start = time.perf_counter()
                refined, value, evals = discrete_refine(
                    model,
                    phase0,
                    levels=10,
                    sweeps=profile.quant_sweeps,
                    seed=rng_seed + 800_000 + gi * 100 + seed_index,
                )
                elapsed = time.perf_counter() - start
                rrows.append(
                    {
                        "geometry": gname,
                        "seed": seed_index,
                        "initialization": init,
                        "levels": 10,
                        "sweeps": profile.quant_sweeps,
                        "evaluations": evals,
                        "wall_s": elapsed,
                        "final_objective": value,
                        "field_similarity_to_continuous_branch": field_similarity(
                            continuous_field, normalized_field(model.geometry, refined)
                        ),
                        "phase_similarity_to_continuous_branch": phase_similarity(fephase, refined),
                        "winding": vortex_winding(model.geometry, refined),
                        **physical_metrics(model, refined),
                    }
                )
    return pd.DataFrame(qrows), pd.DataFrame(rrows)


def run_homotopy(profile: Profile, h: float, rng_seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    tvals = np.linspace(0.0, 1.0, profile.homotopy_steps)
    for pi, (path, (ga, gb)) in enumerate(homotopy_endpoints().items()):
        print(f"[homotopy] {path}", flush=True)
        geometries = [interpolate_geometry(ga, gb, t, f"{path}_t{i:02d}") for i, t in enumerate(tvals)]
        models = [
            build_quadratic_model(g, h, profile.calibration_samples, rng_seed + 500_000 + pi * 100 + i)
            for i, g in enumerate(geometries)
        ]
        for seed_index in range(profile.homotopy_seeds):
            rng = np.random.default_rng(rng_seed + 600_000 + pi * 1000 + seed_index)
            fresh_phase0 = rng.uniform(-np.pi, np.pi, ga.n)
            previous: np.ndarray | None = None
            previous_field: np.ndarray | None = None
            for i, (t, model) in enumerate(zip(tvals, models)):
                prior_phase = previous
                prior_field = previous_field
                transported_phase: np.ndarray | None = None
                transported_field: np.ndarray | None = None
                for init in ("transported", "fresh"):
                    phase0 = prior_phase if init == "transported" and prior_phase is not None else fresh_phase0
                    row, phase, _ = optimize(
                        model,
                        "branch_hybrid",
                        phase0,
                        profile.maxiter,
                    )
                    field = normalized_field(model.geometry, phase)
                    row.update(
                        {
                            "path": path,
                            "path_index": pi,
                            "seed": seed_index,
                            "step_index": i,
                            "t": t,
                            "initialization": init,
                            "geometry_content_id": geometry_content_id(model.geometry),
                            "adjacent_phase_similarity": (
                                phase_similarity(phase, prior_phase) if prior_phase is not None else 1.0
                            ),
                            "adjacent_phase_rmse": (
                                circular_rmse(phase, prior_phase) if prior_phase is not None else 0.0
                            ),
                            "adjacent_field_similarity": (
                                field_similarity(field, prior_field) if prior_field is not None else 1.0
                            ),
                            "paired_phase_similarity": (
                                phase_similarity(phase, transported_phase)
                                if transported_phase is not None
                                else 1.0
                            ),
                            "paired_field_similarity": (
                                field_similarity(field, transported_field)
                                if transported_field is not None
                                else 1.0
                            ),
                            "winding": vortex_winding(model.geometry, phase),
                        }
                    )
                    if seed_index == 0 and init == "transported":
                        ratio, eigmin = reduced_hessian_ratio(model, "branch_hybrid", phase)
                        row["reduced_hessian_ratio"] = ratio
                        row["reduced_hessian_eigmin"] = eigmin
                    else:
                        row["reduced_hessian_ratio"] = np.nan
                        row["reduced_hessian_eigmin"] = np.nan
                    rows.append(row)
                    if init == "transported":
                        transported_phase = phase
                        transported_field = field
                previous = transported_phase
                previous_field = transported_field
    df = pd.DataFrame(rows)
    df["fresh_matches_transported_branch"] = (
        (df.paired_phase_similarity >= 0.95) & (df.paired_field_similarity >= 0.95)
    )
    return df


def summarize(
    df: pd.DataFrame, local: pd.DataFrame, qdf: pd.DataFrame, rdf: pd.DataFrame, hdf: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    method = (
        df.groupby(["geometry", "method"], as_index=False)
        .agg(
            runs=("seed", "count"),
            stationary_rate=("stationary", "mean"),
            certified_rate=("certified", "mean"),
            branch_reach_rate=("branch_reached", "mean"),
            stable_hessian_rate=("stable_hessian", "mean"),
            gradient_rms_median=("gradient_rms", "median"),
            iterations_median=("iterations", "median"),
            evaluations_median=("evaluations", "median"),
            wall_s_median=("wall_s", "median"),
            field_similarity_median=("field_similarity_to_branch_set", "median"),
            paired_phase_similarity_median=("phase_similarity_to_paired_branch", "median"),
            pressure_scaled_median=("pressure_scaled", "median"),
            force_scaled_median=("force_scaled", "median"),
        )
        .merge(local, on="geometry", how="left")
    )
    quant = (
        qdf.groupby(["geometry", "method", "levels"], as_index=False)
        .agg(
            objective_increase_median=("objective_increase", "median"),
            field_similarity_median=("field_similarity", "median"),
            field_similarity_p05=("field_similarity", lambda x: np.quantile(x, 0.05)),
            winding_preservation_rate=(
                "winding_quantized",
                lambda x: np.nan,
            ),
            stable_hessian_rate=("stable_hessian", "mean"),
        )
    )
    # pandas named aggregation cannot compare two columns; fill explicitly.
    preservation = (
        qdf.assign(winding_same=qdf.winding_continuous == qdf.winding_quantized)
        .groupby(["geometry", "method", "levels"]).winding_same.mean()
    )
    quant["winding_preservation_rate"] = [preservation.loc[tuple(v)] for v in quant[["geometry", "method", "levels"]].values]
    refine = (
        rdf.groupby(["geometry", "initialization"], as_index=False)
        .agg(
            final_objective_median=("final_objective", "median"),
            field_similarity_median=("field_similarity_to_continuous_branch", "median"),
            phase_similarity_median=("phase_similarity_to_continuous_branch", "median"),
            wall_s_median=("wall_s", "median"),
            stable_hessian_rate=("stable_hessian", "mean"),
        )
    )
    hom = (
        hdf.groupby(["path", "initialization"], as_index=False)
        .agg(
            stationary_rate=("stationary", "mean"),
            iterations_median=("iterations", "median"),
            evaluations_median=("evaluations", "median"),
            wall_s_median=("wall_s", "median"),
            adjacent_phase_similarity_p05=("adjacent_phase_similarity", lambda x: np.quantile(x, 0.05)),
            adjacent_field_similarity_p05=("adjacent_field_similarity", lambda x: np.quantile(x, 0.05)),
            paired_phase_similarity_median=("paired_phase_similarity", "median"),
            paired_field_similarity_median=("paired_field_similarity", "median"),
            fresh_branch_match_rate=("fresh_matches_transported_branch", "mean"),
            stable_hessian_rate=("stable_hessian", "mean"),
        )
    )
    return {"method_summary": method, "quantization_summary": quant, "refinement_summary": refine, "homotopy_summary": hom}


def plot_geometry_matrix(models: dict[str, QuadraticModel], outdir: Path) -> None:
    fig = plt.figure(figsize=(14, 7.2))
    for i, (name, model) in enumerate(models.items(), start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        p = model.geometry.positions * 1000.0
        n = model.geometry.normals
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=5, alpha=0.75)
        stride = max(1, model.geometry.n // 32)
        ax.quiver(
            p[::stride, 0], p[::stride, 1], p[::stride, 2],
            n[::stride, 0], n[::stride, 1], n[::stride, 2],
            length=12, normalize=True, linewidth=0.4, color="tab:red",
        )
        t = model.geometry.target * 1000.0
        ax.scatter([t[0]], [t[1]], [t[2]], marker="*", s=75, color="black")
        ax.set_title(f"{name} (N={model.geometry.n})")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.view_init(elev=22, azim=-58)
    fig.suptitle("Canonical array matrix: positions, normals, and trap target", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "geometry_matrix.png", dpi=180)
    plt.close(fig)


def plot_convergence(summary: pd.DataFrame, outdir: Path) -> None:
    methods = ["conventional", "regularized", "force_equilibrium", "force_squared", "branch_hybrid"]
    geoms = list(summary.geometry.drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(geoms))
    width = 0.19
    for j, method in enumerate(methods):
        sub = summary[summary.method == method].set_index("geometry").reindex(geoms)
        shift = (j - (len(methods) - 1) / 2.0) * width
        axes[0].bar(x + shift, sub.stationary_rate, width, label=method)
        axes[1].bar(x + shift, sub.iterations_median, width)
        axes[2].bar(x + shift, sub.gradient_rms_median, width)
    axes[0].set_ylabel("Independent stationarity rate")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("Median BFGS iterations")
    axes[2].set_ylabel("Median projected gradient RMS")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xticks(x, geoms, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "convergence_matrix.png", dpi=180)
    plt.close(fig)


def plot_quantization(qsummary: pd.DataFrame, rsummary: pd.DataFrame, outdir: Path) -> None:
    fe = qsummary[qsummary.method == "branch_hybrid"]
    geoms = list(fe.geometry.drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for g in geoms:
        sub = fe[fe.geometry == g].sort_values("levels")
        axes[0].plot(sub.levels, sub.field_similarity_median, marker="o", label=g)
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([8, 10, 16, 32, 64], [8, 10, 16, 32, 64])
    axes[0].set_ylim(0.995, 1.0002)
    axes[0].set_xlabel("Phase levels")
    axes[0].set_ylabel("Median field similarity to continuous branch solution")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    pivot = rsummary.pivot(index="geometry", columns="initialization", values="field_similarity_median")
    pivot.plot(kind="bar", ax=axes[1], color=["tab:blue", "tab:orange"])
    axes[1].set_ylabel("10-level field similarity to continuous branch")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(title="initialization", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "quantization_limits.png", dpi=180)
    plt.close(fig)


def plot_homotopy(hdf: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    transported = hdf[hdf.initialization == "transported"]
    for path, sub in transported.groupby("path"):
        med = sub.groupby("t", as_index=False).median(numeric_only=True)
        axes[0].plot(med.t, med.adjacent_phase_similarity, marker="o", label=path)
        axes[1].plot(med.t, med.adjacent_field_similarity, marker="o", label=path)
    ratio_rows = []
    for (path, seed, step), group in hdf.groupby(["path", "seed", "step_index"]):
        if set(group.initialization) != {"transported", "fresh"}:
            continue
        vals = group.set_index("initialization")
        ratio_rows.append(
            {
                "path": path,
                "t": vals.loc["transported", "t"],
                "iteration_ratio": (vals.loc["fresh", "iterations"] + 1) / (vals.loc["transported", "iterations"] + 1),
            }
        )
    ratios = pd.DataFrame(ratio_rows)
    for path, sub in ratios.groupby("path"):
        med = sub.groupby("t", as_index=False).median(numeric_only=True)
        axes[2].plot(med.t, med.iteration_ratio, marker="o", label=path)
    axes[0].set_ylabel("Adjacent phase similarity")
    axes[1].set_ylabel("Adjacent field similarity")
    axes[2].set_ylabel("Fresh / transported iterations")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("Geometry homotopy t")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axes[1].set_ylim(0.9975, 1.0001)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "branch_homotopy.png", dpi=180)
    plt.close(fig)


def plot_mechanism_and_class(local: pd.DataFrame, outdir: Path) -> None:
    single = local[local.geometry != "OPP512"].copy()
    order = [g for g in ["SQ256", "FIB191", "CAP191", "ARB191"] if g in set(single.geometry)]
    single = single.set_index("geometry").reindex(order)
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].bar(x - 0.18, single.alpha_conventional, 0.36, label="Conventional |p|")
    axes[0].bar(x + 0.18, single.alpha_branch_hybrid, 0.36, label="Squared branch residual")
    axes[0].axhline(1.0, color="tab:blue", linestyle="--", linewidth=0.8)
    axes[0].axhline(2.0, color="tab:orange", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(x, order, rotation=25, ha="right")
    axes[0].set_ylabel("Fitted symmetric local exponent")
    axes[0].set_ylim(0.8, 2.2)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    sims = single.field_similarity_to_SQ256_same_winding_median
    axes[1].bar(x, sims, color="tab:purple")
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x, order, rotation=25, ha="right")
    axes[1].set_ylim(0.95, 1.002)
    axes[1].set_ylabel("Same-winding local-field similarity to SQ256")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "mechanism_and_branch_class.png", dpi=180)
    plt.close(fig)


def write_manifest(outdir: Path, args: argparse.Namespace, profile: Profile) -> None:
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "script": str(Path(__file__).resolve()),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "profile": asdict(profile),
        "physics": {
            "frequency_hz": FREQUENCY,
            "sound_speed_m_s": SOUND_SPEED,
            "wavelength_m": WAVELENGTH,
            "element_radius_m": ELEMENT_RADIUS,
            "particle_diameter_m": PARTICLE_DIAMETER,
            "K1": K1,
            "K2_corrected_positive": K2,
            "potential": "U=K1|p|^2-K2|grad p|^2",
            "dimensionless_coefficients": {
                "pressure_modulus": PRESSURE_TERM_WEIGHT,
                "force_norm": FORCE_TERM_WEIGHT,
                "force_squared": FORCE_SQUARED_TERM_WEIGHT,
                "branch_pressure_squared": BRANCH_PRESSURE_SQUARED_TERM_WEIGHT,
                "branch_force_squared": BRANCH_FORCE_SQUARED_TERM_WEIGHT,
                "branch_softmin_stiffness": BRANCH_STIFFNESS_WEIGHT,
                "branch_softmin_tau": BRANCH_SOFTMIN_TAU,
            },
        },
        "stopping": {
            "optimizer": "SciPy BFGS with analytic phase gradient",
            "global_phase": "phi[0]=0 gauge fixed",
            "gtol": 1e-7,
            "independent_stationarity": "projected gradient RMS <= 1e-6",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--output", type=Path, default=Path("geometry_branch_results"))
    parser.add_argument("--fd-step-mm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--skip-homotopy", action="store_true")
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--homotopy-only", action="store_true")
    parser.add_argument("--homotopy-steps", type=int)
    parser.add_argument("--homotopy-seeds", type=int)
    parser.add_argument("--maxiter", type=int)
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    profile = replace(
        profile,
        homotopy_steps=args.homotopy_steps or profile.homotopy_steps,
        homotopy_seeds=args.homotopy_seeds or profile.homotopy_seeds,
        maxiter=args.maxiter or profile.maxiter,
    )
    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    h = args.fd_step_mm * 1e-3
    write_manifest(outdir, args, profile)

    if args.homotopy_only:
        hdf = run_homotopy(profile, h, args.seed)
        hdf.to_csv(outdir / "homotopy_runs.csv", index=False)
        print(f"[done] {outdir}", flush=True)
        return

    df, endpoints, models, local = run_canonical(outdir, profile, h, args.seed)
    df.to_csv(outdir / "canonical_runs.csv", index=False)
    local.to_csv(outdir / "local_exponents_and_cross_array.csv", index=False)
    np.savez_compressed(
        outdir / "continuous_endpoints.npz",
        **{f"{g}__{m}__seed{s}": p for (g, m, s), p in endpoints.items()},
    )

    if args.skip_quantization:
        qdf = pd.DataFrame()
        rdf = pd.DataFrame()
    else:
        qdf, rdf = run_quantization(df, endpoints, models, profile, args.seed)
        qdf.to_csv(outdir / "quantization_runs.csv", index=False)
        rdf.to_csv(outdir / "quantized_refinement.csv", index=False)
    if args.skip_homotopy:
        hdf = pd.DataFrame()
    else:
        hdf = run_homotopy(profile, h, args.seed)
        hdf.to_csv(outdir / "homotopy_runs.csv", index=False)

    if not qdf.empty and not hdf.empty:
        summaries = summarize(df, local, qdf, rdf, hdf)
        for name, table in summaries.items():
            table.to_csv(outdir / f"{name}.csv", index=False)
        plot_convergence(summaries["method_summary"], outdir)
        plot_quantization(summaries["quantization_summary"], summaries["refinement_summary"], outdir)
        plot_homotopy(hdf, outdir)
        plot_mechanism_and_class(local, outdir)
    plot_geometry_matrix(models, outdir)
    print(f"[done] {outdir}", flush=True)


if __name__ == "__main__":
    main()
