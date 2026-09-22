"""Equivalent compact evaluation of the existing discrete multi-trap objective.

Precompute the exact first-difference transfer samples, including the one-sided
boundary rule. The objective uses 13 potential samples per target. Twelve extra
potential samples preserve the full mixed-Hessian diagnostics, and all pressure
samples preserve the RMS-dependent pressure regularizer. Nothing is dropped
from the discrete objective or its analytic phase gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import multitrap as mt


@dataclass(frozen=True)
class _PreparedStencil:
    transfer: np.ndarray
    potential_transfer: np.ndarray
    pressure_transfer: np.ndarray
    gradient_operator: np.ndarray
    hessian_operator: np.ndarray
    potential_point_count: int
    pressure_point_count: int


def _prepare(stencil: mt.TargetStencil) -> _PreparedStencil:
    center = np.asarray(stencil.center)
    offsets = [(0, 0, 0)]
    for axis in range(3):
        for delta in (-2, -1, 1, 2):
            point = [0, 0, 0]
            point[axis] = delta
            offsets.append(tuple(point))
    for axis in range(3):
        for other in range(axis + 1, 3):
            for delta in (-1, 1):
                for other_delta in (-1, 1):
                    point = [0, 0, 0]
                    point[axis] = delta
                    point[other] = other_delta
                    offsets.append(tuple(point))
    lookup = {offset: index for index, offset in enumerate(offsets)}
    indices = tuple((center + np.asarray(offsets)).T)
    rows = [stencil.transfer[indices]]
    for axis, spacing in enumerate(stencil.spacing_m):
        derivative = np.gradient(
            stencil.transfer, spacing, axis=axis, edge_order=1
        )
        rows.append(derivative[indices])
    potential_transfer = np.ascontiguousarray(np.stack(rows, axis=1))
    pressure_transfer = np.ascontiguousarray(
        stencil.transfer.reshape(-1, stencil.n_actuators)
    )
    transfer = np.ascontiguousarray(np.concatenate(
        (potential_transfer.reshape(-1, stencil.n_actuators), pressure_transfer),
        axis=0,
    ))
    gradient = np.zeros((3, len(offsets)))
    hessian = np.zeros((3, 3, len(offsets)))
    for axis, spacing in enumerate(stencil.spacing_m):
        for delta in (-1, 1):
            offset = [0, 0, 0]
            offset[axis] = delta
            gradient[axis, lookup[tuple(offset)]] = delta / (2.0 * spacing)
            offset[axis] = 2 * delta
            hessian[axis, axis, lookup[tuple(offset)]] = 1.0 / (4.0 * spacing**2)
        hessian[axis, axis, 0] = -1.0 / (2.0 * spacing**2)
        for other in range(axis + 1, 3):
            for delta in (-1, 1):
                for other_delta in (-1, 1):
                    offset = [0, 0, 0]
                    offset[axis] = delta
                    offset[other] = other_delta
                    coefficient = delta * other_delta / (
                        4.0 * spacing * stencil.spacing_m[other]
                    )
                    hessian[axis, other, lookup[tuple(offset)]] = coefficient
                    hessian[other, axis, lookup[tuple(offset)]] = coefficient
    return _PreparedStencil(
        transfer=transfer,
        potential_transfer=potential_transfer,
        pressure_transfer=pressure_transfer,
        gradient_operator=gradient,
        hessian_operator=hessian,
        potential_point_count=len(offsets),
        pressure_point_count=pressure_transfer.shape[0],
    )


class CompactMultitrapEvaluator:
    """Drop-in ObjectiveEvaluation with the unchanged S/P/U aggregation.

    Spatial stencils of odd size >= 5 and symmetric curvature weights are
    supported, including mixed curvature weights. Material coefficients,
    smoothing, gravity and cross-target uniformity use the native definitions.
    """

    def __init__(self, problem: mt.MultiTrapProblem, config: mt.MultitrapObjectiveConfig):
        self.problem = problem
        self.config = config
        self.prepared = {id(stencil): _prepare(stencil) for stencil in problem.stencils}

    def _evaluate_local(
        self,
        stencil: mt.TargetStencil,
        phases: np.ndarray,
        coefficients: mt.GorkovCoefficients,
        pressure_epsilon_rel: float,
        curvature_weight_override,
    ) -> mt._LocalEvaluation:
        prepared = self.prepared[id(stencil)]
        actuator = np.exp(1j * phases)
        all_fields = prepared.transfer @ actuator
        point_count = prepared.potential_point_count
        fields = all_fields[:4 * point_count].reshape(point_count, 4)
        pressure = all_fields[4 * point_count:]
        kp = coefficients.pressure_energy_m3_per_pa
        kv = coefficients.gradient_energy_coefficient
        potential = kp * np.abs(fields[:, 0])**2
        for axis in range(3):
            potential -= kv * np.abs(fields[:, axis + 1])**2
        energy_coefficients = np.asarray((kp, -kv, -kv, -kv))
        potential_adjoint = np.einsum(
            "pc,pcn->pn", fields.conj() * energy_coefficients,
            prepared.potential_transfer, optimize=False,
        )
        d_potential = -2.0 * np.imag(potential_adjoint * actuator)
        force = -(prepared.gradient_operator @ potential)
        d_force = -(prepared.gradient_operator @ d_potential)
        force_norm = float(np.linalg.norm(force))
        d_force_norm = (
            np.einsum("i,in->n", force, d_force) / force_norm
            if force_norm > 1.0e-30 else np.zeros(stencil.n_actuators)
        )
        hessian = np.einsum("ijp,p->ij", prepared.hessian_operator, potential)
        weight = (
            stencil.curvature_weight if curvature_weight_override is None
            else np.asarray(curvature_weight_override, dtype=float)
        )
        curvature_operator = np.einsum("ij,ijp->p", weight, prepared.hessian_operator)
        curvature = float(np.sum(weight * hessian))
        d_curvature = curvature_operator @ d_potential

        local_rms = math.sqrt(max(float(np.mean(np.abs(pressure)**2)), 1.0e-64))
        d_local_rms = -np.imag(
            actuator * (pressure.conj() @ prepared.pressure_transfer)
        ) / (prepared.pressure_point_count * max(local_rms, 1.0e-32))
        p0 = fields[0, 0]
        q0 = 1j * prepared.potential_transfer[0, 0] * actuator
        if float(pressure_epsilon_rel) == 0.0:
            pressure_penalty = float(abs(p0))
            d_pressure = np.real(np.conj(p0) * q0) / max(pressure_penalty, 1.0e-32)
        else:
            epsilon_p = float(pressure_epsilon_rel) * (local_rms + 1.0e-32)
            d_epsilon_p = float(pressure_epsilon_rel) * d_local_rms
            pressure_penalty = math.sqrt(
                max(float(abs(p0)**2 + epsilon_p**2), 1.0e-64)
            )
            d_pressure = (
                np.real(np.conj(p0) * q0) + epsilon_p * d_epsilon_p
            ) / max(pressure_penalty, 1.0e-32)
        return mt._LocalEvaluation(
            target_id=stencil.target_id,
            target_m=stencil.target_m,
            potential_center=float(potential[0]),
            pressure_abs=float(abs(p0)),
            pressure_penalty=float(pressure_penalty),
            force_vector_n=np.asarray(force, dtype=float),
            force_norm_n=force_norm,
            hessian_u_n_m=np.asarray(hessian, dtype=float),
            curvature_n_m=curvature,
            d_force_vector=np.asarray(d_force, dtype=float),
            d_force_norm=np.asarray(d_force_norm, dtype=float),
            d_pressure_penalty=np.asarray(d_pressure, dtype=float),
            d_curvature=np.asarray(d_curvature, dtype=float),
        )

    def evaluate(
        self, phases: np.ndarray, *, force_epsilon: float = 0.0,
        uniformity_epsilon: float = 0.0,
    ) -> mt.ObjectiveEvaluation:
        return mt.evaluate_multitrap_objective(
            self.problem, phases, self.config,
            force_epsilon=force_epsilon,
            uniformity_epsilon=uniformity_epsilon,
            _local_evaluator=self._evaluate_local,
        )

    def fun_grad(
        self, reduced_phase: np.ndarray, *, force_epsilon: float = 0.0,
        uniformity_epsilon: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        evaluation = self.evaluate(
            mt._gauge_expand(reduced_phase), force_epsilon=force_epsilon,
            uniformity_epsilon=uniformity_epsilon,
        )
        return evaluation.value, evaluation.gradient_reduced
