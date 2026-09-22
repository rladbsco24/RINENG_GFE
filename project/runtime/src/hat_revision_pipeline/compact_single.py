"""Exact compact evaluation of the existing Single FE loss.

The 13 potential sites and precomputed pressure derivatives reproduce the
original composed finite differences, including one-sided boundary rules.
No coefficient, stencil spacing, physical model, or optimizer is changed.
"""
from __future__ import annotations

import numpy as np

from .gorkov_core import SingleTargetObjective, gauge_full


class CompactSingleObjective:
    """Forward/adjoint products preserving curvature and pressure terms."""

    backend_id = "compact-single-float64-v1"

    def __init__(self, original: SingleTargetObjective):
        if type(original) is not SingleTargetObjective:
            raise TypeError(
                "CompactSingleObjective requires the canonical SingleTargetObjective; "
                "custom objective classes need an explicit equivalent compact evaluator")
        weight = np.asarray(original.method.curvature_weight, dtype=np.float64)
        self.original = original
        self.n_transducers = original.n_transducers
        self.gauge_index = original.gauge_index
        h = original.spacing_m
        center = np.asarray(original.center)
        points = [tuple(center)]
        for axis in range(3):
            for offset in (-2, -1, 1, 2):
                point = center.copy()
                point[axis] += offset
                points.append(tuple(point))
        mixed_terms = []
        if not np.array_equal(weight, np.diag(np.diag(weight))):
            for axis in range(3):
                for other in range(axis + 1, 3):
                    for delta in (-1, 1):
                        for other_delta in (-1, 1):
                            point = center.copy()
                            point[axis] += delta
                            point[other] += other_delta
                            mixed_terms.append((len(points),
                                (weight[axis, other] + weight[other, axis]) *
                                delta * other_delta / (4 * h * h)))
                            points.append(tuple(point))
        self.point_count = len(points)
        indices = tuple(np.asarray(points).T)
        rows = [original.transfer[indices]]
        for axis in range(3):
            derivative = np.gradient(original.transfer, h, axis=axis, edge_order=1)
            rows.append(derivative[indices])
        self.transfer = np.ascontiguousarray(np.stack(rows, axis=1).reshape(4 * self.point_count, self.n_transducers))
        self.gradient_operator = np.zeros((3, self.point_count), dtype=np.float64)
        self.curvature_operator = np.zeros(self.point_count, dtype=np.float64)
        for axis in range(3):
            start = 1 + 4 * axis
            self.gradient_operator[axis, start + 1] = -1.0 / (2 * h)
            self.gradient_operator[axis, start + 2] = 1.0 / (2 * h)
            self.curvature_operator[start] = weight[axis, axis] / (4 * h * h)
            self.curvature_operator[start + 3] = weight[axis, axis] / (4 * h * h)
            self.curvature_operator[0] -= weight[axis, axis] / (2 * h * h)
        for index, coefficient in mixed_terms:
            self.curvature_operator[index] = coefficient
        self.beta = original.method.beta_curvature_per_pa
        self.pressure_transfer = (np.ascontiguousarray(original.transfer.reshape(-1, self.n_transducers))
            if self.beta and original.method.pressure_mode == "smooth_abs" else None)
        self.energy_coefficients = np.asarray(
            [original.coefficients.pressure_j_pa2] +
            [-original.coefficients.gradient_j_m2_pa2] * 3, dtype=np.float64)
        self.alpha = original.method.alpha_per_m
        self.force_target = original.force_target_n.copy()

    def full_fun_grad(self, phase):
        phase = np.asarray(phase, dtype=np.float64)
        if phase.shape != (self.n_transducers,):
            raise ValueError("Expected one full phase per transducer")
        actuator = np.exp(1j * phase)
        fields = (self.transfer @ actuator).reshape(self.point_count, 4)
        potential = np.abs(fields) ** 2 @ self.energy_coefficients
        residual = self.gradient_operator @ potential + self.force_target
        norm = float(np.linalg.norm(residual))
        value = -self.curvature_operator @ potential + self.alpha * norm
        potential_adjoint = (-self.curvature_operator +
            self.alpha * (residual @ self.gradient_operator) / max(norm, 1e-30))
        field_adjoint = (potential_adjoint[:, None] * self.energy_coefficients * fields.conj()).ravel()
        gradient = -2 * np.imag(actuator * (field_adjoint @ self.transfer))
        if self.beta:
            pressure = fields[0, 0]
            jacobian = 1j * self.transfer[0] * actuator
            if self.pressure_transfer is None:
                penalty = abs(pressure)
                penalty_gradient = np.real(pressure.conjugate() * jacobian) / max(penalty, 1e-32)
            else:
                volume = self.pressure_transfer @ actuator
                rms = np.sqrt(max(float(np.mean(np.abs(volume)**2)), 1e-64))
                rms_gradient = -np.imag(actuator * (volume.conj() @ self.pressure_transfer)) / (
                    len(volume) * max(rms, 1e-32))
                epsilon = self.original.smooth_pressure_relative * (rms + 1e-32)
                epsilon_gradient = self.original.smooth_pressure_relative * rms_gradient
                penalty = np.sqrt(max(float(abs(pressure)**2 + epsilon**2), 1e-64))
                penalty_gradient = (np.real(pressure.conjugate() * jacobian) +
                    epsilon * epsilon_gradient) / max(penalty, 1e-32)
            value += self.beta * penalty
            gradient += self.beta * penalty_gradient
        return float(value), np.asarray(gradient, dtype=np.float64)

    def fun_grad(self, reduced_phase):
        phase = gauge_full(np.asarray(reduced_phase, dtype=np.float64), self.gauge_index)
        value, gradient = self.full_fun_grad(phase)
        return value, np.delete(gradient, self.gauge_index)

    def full_value(self, phase):
        return self.full_fun_grad(phase)[0]

    def __getattr__(self, name):
        # Existing physical diagnostics continue through the original evaluator.
        original = object.__getattribute__(self, 'original')
        return getattr(original, name)
