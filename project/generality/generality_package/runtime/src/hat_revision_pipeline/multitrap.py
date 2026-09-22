"""Double-trap Conventional/regularized-FE optimization and S/P/U ablations.

The module deliberately owns the Gor'kov sign convention used by the
multi-target objective.  A :class:`TargetStencil` contains samples of one
globally defined linear pressure field, but the potential is always evaluated
here as

    U = K_p |p|^2 - K_v |grad p|^2,

where ``K_v`` contains the *standard* density contrast
``(rho_particle - rho_medium)`` and is positive for the denser modeled particle.
No legacy HAT sign switch is available.

``S``, ``P`` and ``U`` denote force-norm smoothization, pressure retention and
target-uniformity regularization, respectively.  The eight S/P/U variants are
run from identical initial phases for each seed.  These are numerical
regularizers of the regularized-FE objective; they are not physical
pass/fail gates.
"""

from __future__ import annotations


from rineng_content_id import content_identity
import itertools
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


MULTITRAP_SCHEMA_VERSION = 3
STANDARD_GORKOV_FORMULATION_ID = "gorkov-peak-phasor-standard-contrast-v1"
DOUBLE_TRAP_TARGET_MODEL_ID = (
    "two-point-gorkov-collocation-independent-spheres-no-interparticle-scattering-v1"
)
MethodName = Literal["Conventional", "Regularized-FE"]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _digest_payload(value: Any) -> str:
    return content_identity(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = content_identity()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _wrap_phase(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _gauge_reduce(full_phase: np.ndarray) -> np.ndarray:
    phase = _wrap_phase(np.asarray(full_phase, dtype=float))
    phase = _wrap_phase(phase - phase[0])
    return phase[1:]


def _gauge_expand(reduced_phase: np.ndarray) -> np.ndarray:
    reduced = np.asarray(reduced_phase, dtype=float)
    return _wrap_phase(np.concatenate((np.zeros(1, dtype=float), reduced)))


@dataclass(frozen=True)
class SphereInFluid:
    """Material parameters for the inviscid small-sphere Gor'kov surrogate."""

    frequency_hz: float = 40_000.0
    medium_density_kg_m3: float = 1.225
    medium_sound_speed_m_s: float = 343.0
    particle_density_kg_m3: float = 100.0
    particle_sound_speed_m_s: float = 2400.0
    particle_radius_m: float = 0.65e-3

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")


@dataclass(frozen=True)
class GorkovCoefficients:
    pressure_energy_m3_per_pa: float
    gradient_energy_coefficient: float
    compressibility_contrast: float
    density_contrast: float
    formulation_id: str = STANDARD_GORKOV_FORMULATION_ID


def standard_gorkov_coefficients(material: SphereInFluid) -> GorkovCoefficients:
    """Return the standard peak-phasor Gor'kov coefficients.

    With ``p(t)=Re[p_hat exp(-i omega t)]`` and inviscid linear momentum,

    ``U = V f1 |p_hat|^2/(4 rho0 c0^2)
          - 3 V f2 |grad p_hat|^2/(8 omega^2 rho0)``.

    Therefore the stored gradient coefficient is positive for a particle
    denser than the medium and is subtracted when constructing ``U``.
    """

    rho0 = float(material.medium_density_kg_m3)
    c0 = float(material.medium_sound_speed_m_s)
    rho_p = float(material.particle_density_kg_m3)
    c_p = float(material.particle_sound_speed_m_s)
    omega = 2.0 * np.pi * float(material.frequency_hz)
    volume = 4.0 * np.pi * float(material.particle_radius_m) ** 3 / 3.0
    f1 = 1.0 - (rho0 * c0**2) / (rho_p * c_p**2)
    f2 = 2.0 * (rho_p - rho0) / (2.0 * rho_p + rho0)
    k_pressure = volume * f1 / (4.0 * rho0 * c0**2)
    k_gradient = 3.0 * volume * f2 / (8.0 * omega**2 * rho0)
    return GorkovCoefficients(
        pressure_energy_m3_per_pa=float(k_pressure),
        gradient_energy_coefficient=float(k_gradient),
        compressibility_contrast=float(f1),
        density_contrast=float(f2),
    )


@dataclass(frozen=True)
class TargetStencil:
    """Linear pressure transfer samples around one requested trap target.

    ``transfer[..., n]`` is the complex pressure at the stencil points from
    actuator ``n`` with unit complex drive.  The transfer must be sampled from
    the same global field implementation used for plotting and finite-ka
    validation; it must not recompute/freeze directivity around each target.
    """

    target_id: str
    target_m: tuple[float, float, float]
    transfer: np.ndarray = field(repr=False)
    spacing_m: tuple[float, float, float] | float = 0.5e-3
    curvature_weight: np.ndarray = field(default_factory=lambda: np.eye(3), repr=False)

    def __post_init__(self) -> None:
        transfer = np.asarray(self.transfer, dtype=np.complex128)
        if transfer.ndim != 4:
            raise ValueError("transfer must have shape (nx, ny, nz, n_actuators)")
        if any(size < 5 or size % 2 == 0 for size in transfer.shape[:3]):
            raise ValueError("each spatial stencil dimension must be odd and at least five")
        if transfer.shape[-1] < 2:
            raise ValueError("at least two actuators are required")
        if not np.all(np.isfinite(transfer.real)) or not np.all(np.isfinite(transfer.imag)):
            raise ValueError("transfer contains non-finite values")
        if np.isscalar(self.spacing_m):
            spacing = (float(self.spacing_m),) * 3
        else:
            spacing = tuple(float(v) for v in self.spacing_m)
        if len(spacing) != 3 or any((not np.isfinite(v)) or v <= 0.0 for v in spacing):
            raise ValueError("spacing_m must contain three positive finite values")
        target = tuple(float(v) for v in self.target_m)
        if len(target) != 3 or not np.all(np.isfinite(target)):
            raise ValueError("target_m must contain three finite coordinates")
        weight = np.asarray(self.curvature_weight, dtype=float)
        if weight.shape != (3, 3) or not np.all(np.isfinite(weight)):
            raise ValueError("curvature_weight must be a finite 3 x 3 matrix")
        if not np.allclose(weight, weight.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("curvature_weight must be symmetric")
        object.__setattr__(self, "transfer", transfer)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "target_m", target)
        object.__setattr__(self, "curvature_weight", weight)

    @property
    def center(self) -> tuple[int, int, int]:
        return tuple(size // 2 for size in self.transfer.shape[:3])

    @property
    def n_actuators(self) -> int:
        return int(self.transfer.shape[-1])

    @property
    def fingerprint(self) -> str:
        return _digest_payload(
            {
                "target_id": self.target_id,
                "target_m": self.target_m,
                "spacing_m": self.spacing_m,
                "transfer_hash": _digest_array(self.transfer),
                "curvature_weight_hash": _digest_array(self.curvature_weight),
            }
        )


@dataclass(frozen=True)
class MultiTrapProblem:
    case_id: str
    stencils: tuple[TargetStencil, ...]
    material: SphereInFluid = field(default_factory=SphereInFluid)
    geometry_id: str = "unspecified-array"

    def __post_init__(self) -> None:
        stencils = tuple(self.stencils)
        if len(stencils) < 2:
            raise ValueError("MultiTrapProblem requires at least two target stencils")
        if len({stencil.n_actuators for stencil in stencils}) != 1:
            raise ValueError("all targets must use the same actuator count")
        if len({stencil.target_id for stencil in stencils}) != len(stencils):
            raise ValueError("target_id values must be distinct")
        object.__setattr__(self, "stencils", stencils)

    @property
    def n_actuators(self) -> int:
        return self.stencils[0].n_actuators

    @property
    def target_model_id(self) -> str:
        return (
            f"{len(self.stencils)}-point-gorkov-collocation-"
            "independent-spheres-no-interparticle-scattering-v1"
        )

    @property
    def fingerprint(self) -> str:
        return _digest_payload(
            {
                "schema_version": MULTITRAP_SCHEMA_VERSION,
                "case_id": self.case_id,
                "geometry_id": self.geometry_id,
                "material": asdict(self.material),
                "stencils": [stencil.fingerprint for stencil in self.stencils],
                "formulation_id": STANDARD_GORKOV_FORMULATION_ID,
                "target_model_id": self.target_model_id,
            }
        )


@dataclass(frozen=True)
class DoubleTrapProblem(MultiTrapProblem):
    """Backward-compatible exact two-target specialization."""

    def __post_init__(self) -> None:
        if len(self.stencils) != 2:
            raise ValueError("DoubleTrapProblem requires exactly two target stencils")
        super().__post_init__()

    @property
    def target_model_id(self) -> str:
        return DOUBLE_TRAP_TARGET_MODEL_ID


@dataclass(frozen=True)
class SPUComponents:
    force_smoothing: bool
    pressure_retention: bool
    uniformity: bool

    @property
    def ablation_id(self) -> str:
        return (
            f"S{int(self.force_smoothing)}"
            f"P{int(self.pressure_retention)}"
            f"U{int(self.uniformity)}"
        )


@dataclass(frozen=True)
class MultitrapObjectiveConfig:
    method: MethodName
    components: SPUComponents
    alpha_force: float
    beta_pressure: float
    gamma_uniformity: float
    curvature_weight_override: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    pressure_epsilon_rel: float = 1.0e-3
    smooth_stage_factors: tuple[float, ...] = (0.05, 0.01)
    smooth_stage_maxiters: tuple[int, ...] = (250, 750)
    smooth_scale_floor: float = 1.0e-20
    gtol: float = 0.0
    report_gradient_tol: float = 1.0e-3
    compensate_effective_gravity: bool = False

    def __post_init__(self) -> None:
        if self.method not in ("Conventional", "Regularized-FE"):
            raise ValueError(f"unsupported method {self.method!r}")
        if len(self.smooth_stage_factors) == 0:
            raise ValueError("at least one smooth stage is required")
        if len(self.smooth_stage_factors) != len(self.smooth_stage_maxiters):
            raise ValueError("smooth stage factors and maxiters must have equal length")
        if any((not np.isfinite(v)) or v < 0.0 for v in self.smooth_stage_factors):
            raise ValueError("smooth stage factors must be finite and non-negative")
        if any(int(v) <= 0 for v in self.smooth_stage_maxiters):
            raise ValueError("each smooth stage must have at least one iteration")
        for name in (
            "alpha_force",
            "beta_pressure",
            "gamma_uniformity",
            "pressure_epsilon_rel",
            "smooth_scale_floor",
            "report_gradient_tol",
        ):
            value = float(getattr(self, name))
            if (not np.isfinite(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.method == "Conventional" and self.alpha_force != 0.0:
            raise ValueError("Conventional must have alpha_force=0")
        if self.method == "Regularized-FE" and self.alpha_force <= 0.0:
            raise ValueError("Regularized-FE requires alpha_force>0")
        if self.curvature_weight_override is not None:
            weight = np.asarray(self.curvature_weight_override, dtype=float)
            if weight.shape != (3, 3) or not np.all(np.isfinite(weight)):
                raise ValueError("curvature_weight_override must be a finite 3 x 3 matrix")
            if not np.allclose(weight, weight.T, atol=1.0e-12, rtol=0.0):
                raise ValueError("curvature_weight_override must be symmetric")
        if not isinstance(self.compensate_effective_gravity, (bool, np.bool_)):
            raise ValueError("compensate_effective_gravity must be boolean")

    @property
    def ablation_id(self) -> str:
        if self.method == "Conventional":
            return "Conventional"
        return self.components.ablation_id

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["components"] = asdict(self.components)
        # Preserve every pre-existing cache identity when compensation is not
        # requested.  The opt-in key is serialized only for the new bounded
        # vertical-load experiment.
        if not self.compensate_effective_gravity:
            payload.pop("compensate_effective_gravity", None)
        return payload


def effective_weight_force_target_n(material: SphereInFluid) -> np.ndarray:
    """Return the fixed upward radiation-force target that balances weight.

    This is not a fitted parameter.  It is the buoyancy-corrected bead weight
    ``(rho_p-rho_0) V g`` for the material already attached to the problem,
    directed along ``+z`` because gravity acts along ``-z``.
    """

    volume_m3 = 4.0 * np.pi * float(material.particle_radius_m) ** 3 / 3.0
    effective_weight_n = (
        (float(material.particle_density_kg_m3) - float(material.medium_density_kg_m3))
        * volume_m3
        * 9.80665
    )
    return np.asarray((0.0, 0.0, effective_weight_n), dtype=float)


def conventional_double_trap_config(
    *,
    stage_maxiters: tuple[int, ...] = (250, 750),
    stage_factors: tuple[float, ...] | None = None,
    gtol: float = 0.0,
    report_gradient_tol: float = 1.0e-3,
) -> MultitrapObjectiveConfig:
    """Pressure/curvature Conventional baseline with target balancing."""

    if stage_factors is None:
        stage_factors = tuple(
            float(v) for v in np.geomspace(0.05, 0.01, num=len(stage_maxiters))
        )
    return MultitrapObjectiveConfig(
        method="Conventional",
        components=SPUComponents(False, True, True),
        alpha_force=0.0,
        beta_pressure=1.0,
        gamma_uniformity=1.0,
        curvature_weight_override=(
            (1000.0, 0.0, 0.0),
            (0.0, 1000.0, 0.0),
            (0.0, 0.0, 10.0),
        ),
        # Preserve the true pressure-null |p| cusp.  Smooth pressure belongs
        # only to the regularized formulation.
        pressure_epsilon_rel=0.0,
        smooth_stage_factors=stage_factors,
        smooth_stage_maxiters=stage_maxiters,
        gtol=gtol,
        report_gradient_tol=report_gradient_tol,
    )


def regularized_fe_double_trap_config(
    *,
    alpha_force: float = 1000.0,
    beta_pressure: float = 5.0e-5,
    gamma_uniformity: float = 1.0,
    stage_maxiters: tuple[int, ...] = (250, 750),
    stage_factors: tuple[float, ...] | None = None,
    gtol: float = 0.0,
    report_gradient_tol: float = 1.0e-3,
) -> MultitrapObjectiveConfig:
    """Full S/P/U regularized force-equilibrium configuration."""

    if stage_factors is None:
        stage_factors = tuple(
            float(v) for v in np.geomspace(0.05, 0.01, num=len(stage_maxiters))
        )
    return MultitrapObjectiveConfig(
        method="Regularized-FE",
        components=SPUComponents(True, True, True),
        alpha_force=float(alpha_force),
        beta_pressure=float(beta_pressure),
        gamma_uniformity=float(gamma_uniformity),
        curvature_weight_override=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        smooth_stage_factors=stage_factors,
        smooth_stage_maxiters=stage_maxiters,
        gtol=gtol,
        report_gradient_tol=report_gradient_tol,
    )


def spu_ablation_configs(
    base: MultitrapObjectiveConfig,
) -> tuple[MultitrapObjectiveConfig, ...]:
    """Return all eight paired regularized-FE S/P/U configurations.

    Full S/P/U is returned first, followed by decreasing component count.  The
    optimization schedule and all scalar weights remain identical.
    """

    if base.method != "Regularized-FE":
        raise ValueError("S/P/U ablations are defined for Regularized-FE")
    components = [
        SPUComponents(bool(s), bool(p), bool(u))
        for s, p, u in itertools.product((False, True), repeat=3)
    ]
    components.sort(
        key=lambda c: (
            -sum((c.force_smoothing, c.pressure_retention, c.uniformity)),
            c.ablation_id,
        )
    )
    return tuple(replace(base, components=value) for value in components)


@dataclass
class _LocalEvaluation:
    target_id: str
    target_m: tuple[float, float, float]
    potential_center: float
    pressure_abs: float
    pressure_penalty: float
    force_vector_n: np.ndarray
    force_norm_n: float
    hessian_u_n_m: np.ndarray
    curvature_n_m: float
    d_force_vector: np.ndarray
    d_force_norm: np.ndarray
    d_pressure_penalty: np.ndarray
    d_curvature: np.ndarray


def _spatial_gradient(value: np.ndarray, spacing: tuple[float, float, float]) -> list[np.ndarray]:
    return [
        np.gradient(value, spacing[axis], axis=axis, edge_order=1)
        for axis in range(3)
    ]


def _evaluate_local(
    stencil: TargetStencil,
    phases: np.ndarray,
    coefficients: GorkovCoefficients,
    pressure_epsilon_rel: float,
    curvature_weight_override: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None,
) -> _LocalEvaluation:
    phases = np.asarray(phases, dtype=float)
    if phases.shape != (stencil.n_actuators,):
        raise ValueError(
            f"phase vector has shape {phases.shape}, expected {(stencil.n_actuators,)}"
        )
    actuator = np.exp(1j * phases)
    transfer = stencil.transfer
    pressure = np.einsum("...n,n->...", transfer, actuator, optimize=True)
    phase_jacobian = 1j * transfer * actuator.reshape((1, 1, 1, -1))
    dp = _spatial_gradient(pressure, stencil.spacing_m)
    dq = _spatial_gradient(phase_jacobian, stencil.spacing_m)

    kp = coefficients.pressure_energy_m3_per_pa
    kv = coefficients.gradient_energy_coefficient
    potential = kp * np.abs(pressure) ** 2
    for component in dp:
        potential -= kv * np.abs(component) ** 2
    d_potential = 2.0 * kp * np.real(np.conj(pressure)[..., None] * phase_jacobian)
    for component, component_q in zip(dp, dq):
        d_potential -= 2.0 * kv * np.real(
            np.conj(component)[..., None] * component_q
        )

    grad_u = _spatial_gradient(potential, stencil.spacing_m)
    d_grad_u = _spatial_gradient(d_potential, stencil.spacing_m)
    center = stencil.center
    hessian = np.empty((3, 3), dtype=float)
    d_hessian = np.empty((3, 3, stencil.n_actuators), dtype=float)
    for i in range(3):
        grad_i = grad_u[i]
        d_grad_i = d_grad_u[i]
        for j in range(3):
            hessian[i, j] = np.gradient(
                grad_i,
                stencil.spacing_m[j],
                axis=j,
                edge_order=1,
            )[center]
            d_hessian[i, j] = np.gradient(
                d_grad_i,
                stencil.spacing_m[j],
                axis=j,
                edge_order=1,
            )[center]
    hessian = 0.5 * (hessian + hessian.T)
    d_hessian = 0.5 * (d_hessian + np.swapaxes(d_hessian, 0, 1))

    grad_center = np.asarray([component[center] for component in grad_u], dtype=float)
    d_grad_center = np.stack([component[center] for component in d_grad_u], axis=0)
    force = -grad_center
    d_force = -d_grad_center
    force_norm = float(np.linalg.norm(force))
    if force_norm > 1.0e-30:
        d_force_norm = np.einsum("i,in->n", force, d_force) / force_norm
    else:
        d_force_norm = np.zeros(stencil.n_actuators, dtype=float)

    p2 = np.abs(pressure) ** 2
    local_rms = math.sqrt(max(float(np.mean(p2)), 1.0e-64))
    d_local_rms = (
        np.mean(
            np.real(np.conj(pressure)[..., None] * phase_jacobian),
            axis=(0, 1, 2),
        )
        / max(local_rms, 1.0e-32)
    )
    p0 = pressure[center]
    q0 = phase_jacobian[center]
    if float(pressure_epsilon_rel) == 0.0:
        pressure_penalty = float(abs(p0))
        d_pressure = np.real(np.conj(p0) * q0) / max(pressure_penalty, 1.0e-32)
    else:
        epsilon_p = float(pressure_epsilon_rel) * (local_rms + 1.0e-32)
        d_epsilon_p = float(pressure_epsilon_rel) * d_local_rms
        pressure_penalty = math.sqrt(
            max(float(abs(p0) ** 2 + epsilon_p**2), 1.0e-64)
        )
        d_pressure = (
            np.real(np.conj(p0) * q0) + epsilon_p * d_epsilon_p
        ) / max(pressure_penalty, 1.0e-32)
    curvature_weight = (
        stencil.curvature_weight
        if curvature_weight_override is None
        else np.asarray(curvature_weight_override, dtype=float)
    )
    curvature = float(np.sum(curvature_weight * hessian))
    d_curvature = np.einsum(
        "ij,ijn->n", curvature_weight, d_hessian, optimize=True
    )
    return _LocalEvaluation(
        target_id=stencil.target_id,
        target_m=stencil.target_m,
        potential_center=float(potential[center]),
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


@dataclass
class ObjectiveEvaluation:
    value: float
    gradient_full: np.ndarray
    local_losses: np.ndarray
    local_gradients: np.ndarray
    locals: tuple[_LocalEvaluation, ...]
    force_penalties: np.ndarray
    uniformity_penalty: float
    force_epsilon: float
    uniformity_epsilon: float

    @property
    def gradient_reduced(self) -> np.ndarray:
        return np.asarray(self.gradient_full[1:], dtype=float)


def evaluate_multitrap_objective(
    problem: MultiTrapProblem,
    phases: np.ndarray,
    config: MultitrapObjectiveConfig,
    *,
    force_epsilon: float = 0.0,
    uniformity_epsilon: float = 0.0,
    _local_evaluator: Callable[..., _LocalEvaluation] | None = None,
) -> ObjectiveEvaluation:
    """Evaluate the multi-target objective and its analytic phase gradient."""

    phases = np.asarray(phases, dtype=float)
    if phases.shape != (problem.n_actuators,):
        raise ValueError(
            f"phases has shape {phases.shape}, expected {(problem.n_actuators,)}"
        )
    coefficients = standard_gorkov_coefficients(problem.material)
    local_evaluator = _evaluate_local if _local_evaluator is None else _local_evaluator
    local_evaluations = tuple(
        local_evaluator(
            stencil,
            phases,
            coefficients,
            pressure_epsilon_rel=config.pressure_epsilon_rel,
            curvature_weight_override=config.curvature_weight_override,
        )
        for stencil in problem.stencils
    )
    losses: list[float] = []
    gradients: list[np.ndarray] = []
    force_penalties: list[float] = []
    compensated_target_n = (
        effective_weight_force_target_n(problem.material)
        if config.compensate_effective_gravity
        else None
    )
    for local in local_evaluations:
        if compensated_target_n is None:
            residual_force_norm = local.force_norm_n
            d_residual_force_norm = local.d_force_norm
        else:
            residual_force = local.force_vector_n - compensated_target_n
            residual_force_norm = float(np.linalg.norm(residual_force))
            if residual_force_norm > 1.0e-30:
                d_residual_force_norm = (
                    np.einsum(
                        "i,in->n", residual_force, local.d_force_vector
                    )
                    / residual_force_norm
                )
            else:
                d_residual_force_norm = np.zeros_like(local.d_force_norm)
        if config.components.force_smoothing:
            denominator = math.sqrt(
                residual_force_norm**2 + float(force_epsilon) ** 2
            )
            force_penalty = denominator - float(force_epsilon)
            if denominator > 1.0e-30:
                d_force_penalty = (
                    residual_force_norm / denominator
                ) * d_residual_force_norm
            else:
                d_force_penalty = np.zeros_like(local.d_force_norm)
        else:
            force_penalty = residual_force_norm
            d_force_penalty = d_residual_force_norm
        pressure_weight = config.beta_pressure if config.components.pressure_retention else 0.0
        local_loss = (
            -local.curvature_n_m
            + config.alpha_force * force_penalty
            + pressure_weight * local.pressure_penalty
        )
        local_gradient = (
            -local.d_curvature
            + config.alpha_force * d_force_penalty
            + pressure_weight * local.d_pressure_penalty
        )
        losses.append(float(local_loss))
        gradients.append(np.asarray(local_gradient, dtype=float))
        force_penalties.append(float(force_penalty))

    losses_array = np.asarray(losses, dtype=float)
    gradients_array = np.stack(gradients, axis=0)
    value = float(np.mean(losses_array))
    gradient = np.mean(gradients_array, axis=0)
    uniformity_penalty = 0.0
    if config.components.uniformity and len(losses_array) > 1:
        centered = losses_array - float(np.mean(losses_array))
        variance = float(np.mean(centered**2))
        denominator = math.sqrt(variance + float(uniformity_epsilon) ** 2)
        uniformity_penalty = denominator - float(uniformity_epsilon)
        if denominator > 1.0e-30:
            d_uniformity = np.mean(centered[:, None] * gradients_array, axis=0) / denominator
        else:
            d_uniformity = np.zeros(problem.n_actuators, dtype=float)
        value += config.gamma_uniformity * uniformity_penalty
        gradient = gradient + config.gamma_uniformity * d_uniformity
    return ObjectiveEvaluation(
        value=float(value),
        gradient_full=np.asarray(gradient, dtype=float),
        local_losses=losses_array,
        local_gradients=gradients_array,
        locals=local_evaluations,
        force_penalties=np.asarray(force_penalties, dtype=float),
        uniformity_penalty=float(uniformity_penalty),
        force_epsilon=float(force_epsilon),
        uniformity_epsilon=float(uniformity_epsilon),
    )


@dataclass(frozen=True)
class SmoothScales:
    force_scale: float
    loss_std_scale: float
    initial_force_norms: tuple[float, ...]
    initial_local_losses: tuple[float, ...]
    scale_floor: float


def estimate_smooth_scales(
    problem: MultiTrapProblem,
    phases: np.ndarray,
    config: MultitrapObjectiveConfig,
) -> SmoothScales:
    evaluation = evaluate_multitrap_objective(problem, phases, config)
    force_values = np.asarray(evaluation.force_penalties, dtype=float)
    finite_force = force_values[np.isfinite(force_values)]
    finite_losses = evaluation.local_losses[np.isfinite(evaluation.local_losses)]
    floor = float(config.smooth_scale_floor)
    force_raw = float(np.median(finite_force)) if finite_force.size else floor
    std_raw = float(np.std(finite_losses, ddof=0)) if finite_losses.size > 1 else 0.0
    return SmoothScales(
        force_scale=max(abs(force_raw), floor),
        loss_std_scale=max(abs(std_raw), floor),
        initial_force_norms=tuple(float(v) for v in force_values),
        initial_local_losses=tuple(float(v) for v in evaluation.local_losses),
        scale_floor=floor,
    )


@dataclass(frozen=True)
class HistoryRecord:
    run_id: str
    stage_index: int
    stage_factor: float
    stage_iteration: int
    global_iteration: int
    elapsed_sec: float
    objective: float
    gradient_norm: float
    mean_force_norm_n: float
    max_force_norm_n: float
    local_loss_std: float
    force_epsilon: float
    uniformity_epsilon: float


@dataclass(frozen=True)
class StageRecord:
    stage_index: int
    stage_factor: float
    maxiter: int
    iterations: int
    nfev: int
    njev: int
    runtime_sec: float
    objective: float
    gradient_norm: float
    success: bool
    status: int
    message: str
    force_epsilon: float
    uniformity_epsilon: float
    gradient_inf_norm: float = float("nan")


@dataclass(frozen=True)
class PerTargetRecord:
    run_id: str
    target_id: str
    target_x_m: float
    target_y_m: float
    target_z_m: float
    local_objective: float
    pressure_abs_pa_arb: float
    gorkov_potential_j_arb: float
    force_x_n_arb: float
    force_y_n_arb: float
    force_z_n_arb: float
    force_norm_n_arb: float
    target_hessian_eig_1_n_m_arb: float
    target_hessian_eig_2_n_m_arb: float
    target_hessian_eig_3_n_m_arb: float


@dataclass
class MultitrapRunResult:
    run_id: str
    case_id: str
    geometry_id: str
    method: MethodName
    ablation_id: str
    seed: int
    formulation_id: str
    target_model_id: str
    problem_fingerprint: str
    config_fingerprint: str
    cache_fingerprint: str
    initial_phases: np.ndarray
    phases: np.ndarray
    final_objective: float
    terminal_gradient_norm: float
    near_stationary: bool
    termination_reason: str
    iterations: int
    nfev: int
    njev: int
    setup_sec: float
    solve_sec: float
    final_evaluation_sec: float
    end_to_end_sec: float
    smooth_scales: SmoothScales
    history: list[HistoryRecord]
    stages: list[StageRecord]
    per_target: list[PerTargetRecord]
    components: SPUComponents
    metadata: dict[str, Any] = field(default_factory=dict)
    history_phase_rad: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float), repr=False
    )

    def summary_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "geometry_id": self.geometry_id,
            "method": self.method,
            "ablation_id": self.ablation_id,
            "seed": self.seed,
            "S": self.components.force_smoothing,
            "P": self.components.pressure_retention,
            "U": self.components.uniformity,
            "formulation_id": self.formulation_id,
            "target_model_id": self.target_model_id,
            "problem_fingerprint": self.problem_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "initial_phase_hash": _digest_array(self.initial_phases),
            "phase_hash": _digest_array(self.phases),
            "final_objective": self.final_objective,
            "terminal_gradient_norm": self.terminal_gradient_norm,
            "near_stationary": self.near_stationary,
            "termination_reason": self.termination_reason,
            "iterations": self.iterations,
            "nfev": self.nfev,
            "njev": self.njev,
            "setup_sec": self.setup_sec,
            "solve_sec": self.solve_sec,
            "final_evaluation_sec": self.final_evaluation_sec,
            "end_to_end_sec": self.end_to_end_sec,
        }

    def history_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.history]

    def per_target_rows(self) -> list[dict[str, Any]]:
        rows = []
        for record in self.per_target:
            row = asdict(record)
            row.update(
                {
                    "method": self.method,
                    "ablation_id": self.ablation_id,
                    "seed": self.seed,
                    "S": self.components.force_smoothing,
                    "P": self.components.pressure_retention,
                    "U": self.components.uniformity,
                    "formulation_id": self.formulation_id,
                    "target_model_id": self.target_model_id,
                }
            )
            rows.append(row)
        return rows


MULTITRAP_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "case_id",
    "geometry_id",
    "method",
    "ablation_id",
    "seed",
    "S",
    "P",
    "U",
    "formulation_id",
    "target_model_id",
    "problem_fingerprint",
    "config_fingerprint",
    "cache_fingerprint",
    "initial_phase_hash",
    "phase_hash",
    "final_objective",
    "terminal_gradient_norm",
    "near_stationary",
    "termination_reason",
    "iterations",
    "nfev",
    "njev",
    "setup_sec",
    "solve_sec",
    "final_evaluation_sec",
    "end_to_end_sec",
)

MULTITRAP_HISTORY_COLUMNS: tuple[str, ...] = tuple(HistoryRecord.__dataclass_fields__)
MULTITRAP_PER_TARGET_COLUMNS: tuple[str, ...] = (
    *tuple(PerTargetRecord.__dataclass_fields__),
    "method",
    "ablation_id",
    "seed",
    "S",
    "P",
    "U",
    "formulation_id",
    "target_model_id",
)


def multitrap_cache_payload(
    problem: MultiTrapProblem,
    config: MultitrapObjectiveConfig,
    seed: int,
    initial_phases: np.ndarray,
    *,
    optimizer: str | None = None,
    optimizer_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .fe_solver_policy import require_fe_solver, FE_SOLVER_REVISION
    if config.method == "Regularized-FE":
        optimizer, _ = require_fe_solver(optimizer, task="Multi")
    elif optimizer is None:
        optimizer = "BFGS"
    payload = {
        "schema_version": MULTITRAP_SCHEMA_VERSION,
        "kind": "multitrap-optimization",
        "formulation_id": STANDARD_GORKOV_FORMULATION_ID,
        "target_model_id": problem.target_model_id,
        "problem_fingerprint": problem.fingerprint,
        "config": config.to_payload(),
        "seed": int(seed),
        "initial_phase_hash": _digest_array(np.asarray(initial_phases, dtype=float)),
        "optimizer": "scipy-BFGS-gauge-fixed",
    }
    if optimizer != "BFGS" or optimizer_options:
        payload["optimizer"] = f"scipy-{optimizer}-gauge-fixed"
        payload["optimizer_stage_options"] = [
            _multitrap_solver_options(optimizer, optimizer_options, config.gtol, cap)
            for cap in config.smooth_stage_maxiters
        ]
    if config.method == "Regularized-FE":
        payload["fe_solver_policy"] = FE_SOLVER_REVISION
        payload["evaluator_backend"] = "compact-multitrap-discrete-v1"
    return payload


def _multitrap_solver_options(
    optimizer: str,
    optimizer_options: Mapping[str, Any] | None,
    gtol: float,
    maxiter: int,
) -> dict[str, Any]:
    if optimizer not in ("BFGS", "L-BFGS-B"):
        raise ValueError("optimizer must be BFGS or L-BFGS-B")
    options: dict[str, Any] = {"maxiter": int(maxiter), "gtol": float(gtol), "disp": False}
    if optimizer == "L-BFGS-B":
        options.update(ftol=0.0, maxcor=10, maxls=40, maxfun=1_000_000)
    options.update(dict(optimizer_options or {}))
    # Existing smoothing-stage budgets remain authoritative for both solvers.
    options["maxiter"] = int(maxiter)
    return options


def _make_history_record(
    *,
    run_id: str,
    stage_index: int,
    stage_factor: float,
    stage_iteration: int,
    global_iteration: int,
    elapsed_sec: float,
    evaluation: ObjectiveEvaluation,
) -> HistoryRecord:
    force_norms = np.asarray([local.force_norm_n for local in evaluation.locals])
    return HistoryRecord(
        run_id=run_id,
        stage_index=int(stage_index),
        stage_factor=float(stage_factor),
        stage_iteration=int(stage_iteration),
        global_iteration=int(global_iteration),
        elapsed_sec=float(elapsed_sec),
        objective=float(evaluation.value),
        gradient_norm=float(np.linalg.norm(evaluation.gradient_reduced)),
        mean_force_norm_n=float(np.mean(force_norms)),
        max_force_norm_n=float(np.max(force_norms)),
        local_loss_std=float(np.std(evaluation.local_losses, ddof=0)),
        force_epsilon=float(evaluation.force_epsilon),
        uniformity_epsilon=float(evaluation.uniformity_epsilon),
    )


def _termination_reason(result: Any, gradient_norm: float, config: MultitrapObjectiveConfig) -> str:
    message = str(getattr(result, "message", "")).lower()
    if np.isfinite(gradient_norm) and gradient_norm <= config.report_gradient_tol:
        return "reported-gradient-tolerance"
    if "precision loss" in message or "desired error not necessarily achieved" in message:
        return "precision-loss/stalled-endpoint"
    if bool(getattr(result, "success", False)):
        return "scipy-success"
    return "iteration-budget-or-other"


def run_multitrap(
    problem: MultiTrapProblem,
    config: MultitrapObjectiveConfig,
    *,
    seed: int,
    initial_phases: np.ndarray | None = None,
    record_history: bool = True,
    metadata: Mapping[str, Any] | None = None,
    uniformity_epsilon_multiplier: float = 1.0,
    optimizer: str | None = None,
    optimizer_options: Mapping[str, Any] | None = None,
    evaluator_backend: str | None = None,
) -> MultitrapRunResult:
    """Optimize one deterministic multi-target case with staged smoothing.

    ``uniformity_epsilon_multiplier`` scales only the STD smoothing constant;
    the default preserves the established solver and force-smoothing schedule.
    Regularized FE/GFE uses compact evaluation and BFGS.
    Conventional retains its native solver. Each smoothing stage starts fresh
    solver memory. Incompatible explicit FE solver/backend settings fail early.
    """
    from .fe_solver_policy import require_fe_solver, FE_SOLVER_REVISION
    if config.method == "Regularized-FE":
        optimizer, evaluator_backend = require_fe_solver(optimizer, evaluator_backend, task="Multi")
    else:
        optimizer = "BFGS" if optimizer is None else optimizer
        evaluator_backend = "native" if evaluator_backend is None else evaluator_backend
    uniformity_epsilon_multiplier = float(uniformity_epsilon_multiplier)
    if evaluator_backend not in ("native", "compact"):
        raise ValueError("evaluator_backend must be native or compact")
    if not np.isfinite(uniformity_epsilon_multiplier) or uniformity_epsilon_multiplier < 0:
        raise ValueError("uniformity_epsilon_multiplier must be finite and non-negative")
    stage_options = [
        _multitrap_solver_options(optimizer, optimizer_options, config.gtol, cap)
        for cap in config.smooth_stage_maxiters
    ]

    end_to_end_start = time.perf_counter()
    setup_start = time.perf_counter()
    if initial_phases is None:
        rng = np.random.default_rng(int(seed))
        initial_full = rng.uniform(-np.pi, np.pi, size=problem.n_actuators)
    else:
        initial_full = np.asarray(initial_phases, dtype=float)
        if initial_full.shape != (problem.n_actuators,):
            raise ValueError(
                f"initial_phases has shape {initial_full.shape}, "
                f"expected {(problem.n_actuators,)}"
            )
    initial_full = _gauge_expand(_gauge_reduce(initial_full))
    cache_payload = multitrap_cache_payload(
        problem, config, seed, initial_full,
        optimizer=optimizer, optimizer_options=optimizer_options,
    )
    if uniformity_epsilon_multiplier != 1.0:
        cache_payload["uniformity_epsilon_multiplier"] = uniformity_epsilon_multiplier
    if evaluator_backend != "native":
        cache_payload["evaluator_backend"] = "compact-multitrap-discrete-v1"
    cache_fingerprint = _digest_payload(cache_payload)
    config_fingerprint = _digest_payload(config.to_payload())
    run_id = (
        f"{problem.case_id}|{config.method}|{config.ablation_id}|"
        f"seed{int(seed)}|{cache_fingerprint[:12]}"
    )
    scales = estimate_smooth_scales(problem, initial_full, config)
    compact_evaluator = None
    if evaluator_backend == "compact":
        from .compact_multitrap import CompactMultitrapEvaluator
        compact_evaluator = CompactMultitrapEvaluator(problem, config)
    setup_sec = time.perf_counter() - setup_start

    x_current = _gauge_reduce(initial_full)
    history: list[HistoryRecord] = []
    history_phase = [np.asarray(initial_full, dtype=float)]
    stages: list[StageRecord] = []
    solve_start = time.perf_counter()
    total_iterations = total_nfev = total_njev = 0
    global_iteration = 0
    final_result: Any | None = None

    for stage_index, (stage_factor, maxiter) in enumerate(
        zip(config.smooth_stage_factors, config.smooth_stage_maxiters)
    ):
        force_epsilon = (
            float(stage_factor) * scales.force_scale
            if config.components.force_smoothing
            else 0.0
        )
        uniformity_epsilon = (
            float(stage_factor) * scales.loss_std_scale * uniformity_epsilon_multiplier
            if config.components.uniformity
            else 0.0
        )
        evaluation_cache: dict[str, Any] = {}

        def evaluate_reduced(x: np.ndarray) -> ObjectiveEvaluation:
            key = np.asarray(x, dtype=float).tobytes()
            if evaluation_cache.get("key") != key:
                evaluation_cache["key"] = key
                if compact_evaluator is None:
                    evaluation_cache["value"] = evaluate_multitrap_objective(
                        problem, _gauge_expand(x), config,
                        force_epsilon=force_epsilon,
                        uniformity_epsilon=uniformity_epsilon,
                    )
                else:
                    evaluation_cache["value"] = compact_evaluator.evaluate(
                        _gauge_expand(x), force_epsilon=force_epsilon,
                        uniformity_epsilon=uniformity_epsilon,
                    )
            return evaluation_cache["value"]

        stage_iteration = 0
        stage_start = time.perf_counter()
        if record_history:
            initial_evaluation = evaluate_reduced(x_current)
            history.append(
                _make_history_record(
                    run_id=run_id,
                    stage_index=stage_index,
                    stage_factor=stage_factor,
                    stage_iteration=0,
                    global_iteration=global_iteration,
                    elapsed_sec=time.perf_counter() - solve_start,
                    evaluation=initial_evaluation,
                )
            )

        def callback(x: np.ndarray) -> None:
            nonlocal stage_iteration, global_iteration
            stage_iteration += 1
            global_iteration += 1
            history_phase.append(_gauge_expand(np.asarray(x, dtype=float)))
            if record_history:
                history.append(
                    _make_history_record(
                        run_id=run_id,
                        stage_index=stage_index,
                        stage_factor=stage_factor,
                        stage_iteration=stage_iteration,
                        global_iteration=global_iteration,
                        elapsed_sec=time.perf_counter() - solve_start,
                        evaluation=evaluate_reduced(x),
                    )
                )

        result = minimize(
            lambda x: evaluate_reduced(x).value,
            x_current,
            jac=lambda x: evaluate_reduced(x).gradient_reduced,
            method=optimizer,
            callback=callback,
            options=stage_options[stage_index],
        )
        x_current = np.asarray(result.x, dtype=float)
        stage_phase = _gauge_expand(x_current)
        if np.max(np.abs(_wrap_phase(stage_phase - history_phase[-1]))) > 1.0e-14:
            history_phase.append(stage_phase)
        stage_evaluation = evaluate_reduced(x_current)
        stage_runtime = time.perf_counter() - stage_start
        stage_grad_norm = float(np.linalg.norm(stage_evaluation.gradient_reduced))
        stage_record = StageRecord(
            stage_index=stage_index,
            stage_factor=float(stage_factor),
            maxiter=int(maxiter),
            iterations=int(getattr(result, "nit", stage_iteration) or 0),
            nfev=int(getattr(result, "nfev", 0) or 0),
            njev=int(getattr(result, "njev", 0) or 0),
            runtime_sec=float(stage_runtime),
            objective=float(stage_evaluation.value),
            gradient_norm=stage_grad_norm,
            success=bool(result.success),
            status=int(getattr(result, "status", -999)),
            message=str(getattr(result, "message", "")),
            force_epsilon=float(force_epsilon),
            uniformity_epsilon=float(uniformity_epsilon),
            gradient_inf_norm=float(np.linalg.norm(stage_evaluation.gradient_reduced, ord=np.inf)),
        )
        stages.append(stage_record)
        total_iterations += stage_record.iterations
        total_nfev += stage_record.nfev
        total_njev += stage_record.njev
        final_result = result

    solve_sec = time.perf_counter() - solve_start
    if final_result is None:
        raise RuntimeError("no optimization stage executed")
    final_eval_start = time.perf_counter()
    final_phases = _gauge_expand(x_current)
    final_force_epsilon = stages[-1].force_epsilon
    final_uniformity_epsilon = stages[-1].uniformity_epsilon
    final_evaluation = evaluate_multitrap_objective(
        problem,
        final_phases,
        config,
        force_epsilon=final_force_epsilon,
        uniformity_epsilon=final_uniformity_epsilon,
    )
    terminal_gradient_norm = float(np.linalg.norm(final_evaluation.gradient_reduced))
    terminal_gradient_inf_norm = float(np.linalg.norm(final_evaluation.gradient_reduced, ord=np.inf))
    solver_metadata = dict(metadata or {})
    solver_metadata.update(
        optimizer=optimizer,
        optimizer_stage_options=stage_options,
        terminal_gradient_inf_norm=terminal_gradient_inf_norm,
        stationary_gtol=bool(terminal_gradient_inf_norm <= float(stage_options[-1]["gtol"])),
        solver_success=bool(final_result.success),
    )
    if evaluator_backend != "native":
        solver_metadata["evaluator_backend"] = "compact-multitrap-discrete-v1"
        solver_metadata["fe_solver_policy"] = FE_SOLVER_REVISION
    if optimizer == "L-BFGS-B" or config.method == "Regularized-FE":
        # Initial and terminal diagnostics use the identical final-stage
        # objective; stage-one smoothing must not masquerade as improvement.
        initial_final_stage = evaluate_multitrap_objective(
            problem, initial_full, config,
            force_epsilon=final_force_epsilon,
            uniformity_epsilon=final_uniformity_epsilon,
        )
        solver_metadata.update(
            initial_final_stage_objective=float(initial_final_stage.value),
            initial_final_stage_gradient_norm=float(np.linalg.norm(initial_final_stage.gradient_reduced)),
            initial_final_stage_gradient_inf_norm=float(np.linalg.norm(initial_final_stage.gradient_reduced, ord=np.inf)),
        )
    final_evaluation_sec = time.perf_counter() - final_eval_start
    reason = _termination_reason(final_result, terminal_gradient_norm, config)
    per_target = []
    for local, local_loss in zip(final_evaluation.locals, final_evaluation.local_losses):
        eigenvalues = np.linalg.eigvalsh(local.hessian_u_n_m)
        per_target.append(
            PerTargetRecord(
                run_id=run_id,
                target_id=local.target_id,
                target_x_m=local.target_m[0],
                target_y_m=local.target_m[1],
                target_z_m=local.target_m[2],
                local_objective=float(local_loss),
                pressure_abs_pa_arb=local.pressure_abs,
                gorkov_potential_j_arb=local.potential_center,
                force_x_n_arb=float(local.force_vector_n[0]),
                force_y_n_arb=float(local.force_vector_n[1]),
                force_z_n_arb=float(local.force_vector_n[2]),
                force_norm_n_arb=local.force_norm_n,
                target_hessian_eig_1_n_m_arb=float(eigenvalues[0]),
                target_hessian_eig_2_n_m_arb=float(eigenvalues[1]),
                target_hessian_eig_3_n_m_arb=float(eigenvalues[2]),
            )
        )
    return MultitrapRunResult(
        run_id=run_id,
        case_id=problem.case_id,
        geometry_id=problem.geometry_id,
        method=config.method,
        ablation_id=config.ablation_id,
        seed=int(seed),
        formulation_id=STANDARD_GORKOV_FORMULATION_ID,
        target_model_id=problem.target_model_id,
        problem_fingerprint=problem.fingerprint,
        config_fingerprint=config_fingerprint,
        cache_fingerprint=cache_fingerprint,
        initial_phases=np.asarray(initial_full, dtype=float),
        phases=np.asarray(final_phases, dtype=float),
        final_objective=float(final_evaluation.value),
        terminal_gradient_norm=terminal_gradient_norm,
        near_stationary=bool(terminal_gradient_norm <= config.report_gradient_tol),
        termination_reason=reason,
        iterations=total_iterations,
        nfev=total_nfev,
        njev=total_njev,
        setup_sec=float(setup_sec),
        solve_sec=float(solve_sec),
        final_evaluation_sec=float(final_evaluation_sec),
        end_to_end_sec=float(time.perf_counter() - end_to_end_start),
        smooth_scales=scales,
        history=history,
        stages=stages,
        per_target=per_target,
        components=config.components,
        metadata=solver_metadata,
        history_phase_rad=_wrap_phase(np.asarray(history_phase, dtype=float)),
    )


def run_double_trap(
    problem: DoubleTrapProblem,
    config: MultitrapObjectiveConfig,
    **kwargs: Any,
) -> MultitrapRunResult:
    """Compatibility wrapper for callers that explicitly require two targets."""

    if not isinstance(problem, DoubleTrapProblem):
        raise TypeError("run_double_trap requires a DoubleTrapProblem")
    return run_multitrap(problem, config, **kwargs)


def run_paired_double_trap_benchmark(
    problem: DoubleTrapProblem,
    *,
    seeds: Iterable[int],
    conventional: MultitrapObjectiveConfig | None = None,
    regularized_fe: MultitrapObjectiveConfig | None = None,
    record_history: bool = True,
) -> list[MultitrapRunResult]:
    """Run Conventional and regularized-FE from the same phase per seed."""

    conventional = conventional or conventional_double_trap_config()
    regularized_fe = regularized_fe or regularized_fe_double_trap_config()
    if conventional.method != "Conventional":
        raise ValueError("conventional config has the wrong method")
    if regularized_fe.method != "Regularized-FE":
        raise ValueError("regularized_fe config has the wrong method")
    results: list[MultitrapRunResult] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        initial = rng.uniform(-np.pi, np.pi, size=problem.n_actuators)
        initial = _gauge_expand(_gauge_reduce(initial))
        for config in (conventional, regularized_fe):
            results.append(
                run_double_trap(
                    problem,
                    config,
                    seed=int(seed),
                    initial_phases=initial,
                    record_history=record_history,
                    metadata={"paired_method_comparison": True},
                )
            )
    return results


def run_paired_spu_ablation(
    problem: MultiTrapProblem,
    base_config: MultitrapObjectiveConfig,
    *,
    seeds: Iterable[int],
    record_history: bool = False,
) -> list[MultitrapRunResult]:
    """Run the complete 2^3 factorial with a shared start for every seed."""

    variants = spu_ablation_configs(base_config)
    results: list[MultitrapRunResult] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        initial = rng.uniform(-np.pi, np.pi, size=problem.n_actuators)
        initial = _gauge_expand(_gauge_reduce(initial))
        for config in variants:
            results.append(
                run_multitrap(
                    problem,
                    config,
                    seed=int(seed),
                    initial_phases=initial,
                    record_history=record_history,
                    metadata={"paired_spu_factorial": True},
                )
            )
    return results


def _result_metadata(result: MultitrapRunResult) -> dict[str, Any]:
    return {
        "schema_version": MULTITRAP_SCHEMA_VERSION,
        "summary": result.summary_row(),
        "smooth_scales": asdict(result.smooth_scales),
        "history": result.history_rows(),
        "stages": [asdict(record) for record in result.stages],
        "per_target": result.per_target_rows(),
        "components": asdict(result.components),
        "metadata": result.metadata,
    }


def save_multitrap_cache(path: str | Path, result: MultitrapRunResult) -> Path:
    """Atomically save one cache item as compressed NPZ plus JSON metadata."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = _canonical_json(_result_metadata(result))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp.npz", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            initial_phases=np.asarray(result.initial_phases, dtype=float),
            phases=np.asarray(result.phases, dtype=float),
            history_phase_rad=np.asarray(result.history_phase_rad, dtype=float),
            metadata_json=np.asarray(metadata_json),
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_multitrap_cache(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> MultitrapRunResult:
    """Load a cache item and reject a mismatched scientific fingerprint."""

    with np.load(Path(path), allow_pickle=False) as archive:
        initial_phases = np.asarray(archive["initial_phases"], dtype=float)
        phases = np.asarray(archive["phases"], dtype=float)
        history_phase_rad = np.asarray(archive["history_phase_rad"], dtype=float)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if int(metadata.get("schema_version", -1)) != MULTITRAP_SCHEMA_VERSION:
        raise ValueError("unsupported multitrap cache schema")
    summary = metadata["summary"]
    if expected_fingerprint is not None and summary["cache_fingerprint"] != expected_fingerprint:
        raise ValueError("cache fingerprint does not match the requested run")
    scales_dict = dict(metadata["smooth_scales"])
    scales_dict["initial_force_norms"] = tuple(scales_dict["initial_force_norms"])
    scales_dict["initial_local_losses"] = tuple(scales_dict["initial_local_losses"])
    components = SPUComponents(**metadata["components"])
    history = [HistoryRecord(**row) for row in metadata["history"]]
    stages = [StageRecord(**row) for row in metadata["stages"]]
    per_target_fields = set(PerTargetRecord.__dataclass_fields__)
    per_target = [
        PerTargetRecord(**{key: value for key, value in row.items() if key in per_target_fields})
        for row in metadata["per_target"]
    ]
    return MultitrapRunResult(
        run_id=summary["run_id"],
        case_id=summary["case_id"],
        geometry_id=summary["geometry_id"],
        method=summary["method"],
        ablation_id=summary["ablation_id"],
        seed=int(summary["seed"]),
        formulation_id=summary["formulation_id"],
        target_model_id=summary["target_model_id"],
        problem_fingerprint=summary["problem_fingerprint"],
        config_fingerprint=summary["config_fingerprint"],
        cache_fingerprint=summary["cache_fingerprint"],
        initial_phases=initial_phases,
        phases=phases,
        final_objective=float(summary["final_objective"]),
        terminal_gradient_norm=float(summary["terminal_gradient_norm"]),
        near_stationary=bool(summary["near_stationary"]),
        termination_reason=summary["termination_reason"],
        iterations=int(summary["iterations"]),
        nfev=int(summary["nfev"]),
        njev=int(summary["njev"]),
        setup_sec=float(summary["setup_sec"]),
        solve_sec=float(summary["solve_sec"]),
        final_evaluation_sec=float(summary["final_evaluation_sec"]),
        end_to_end_sec=float(summary["end_to_end_sec"]),
        smooth_scales=SmoothScales(**scales_dict),
        history=history,
        stages=stages,
        per_target=per_target,
        components=components,
        metadata=dict(metadata.get("metadata", {})),
        history_phase_rad=history_phase_rad,
    )


def rows_from_results(
    results: Sequence[MultitrapRunResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return summary, per-target and history tables without a pandas dependency."""

    summary = [result.summary_row() for result in results]
    per_target = [row for result in results for row in result.per_target_rows()]
    history = [row for result in results for row in result.history_rows()]
    return summary, per_target, history


__all__ = [
    "DoubleTrapProblem",
    "MultiTrapProblem",
    "DOUBLE_TRAP_TARGET_MODEL_ID",
    "GorkovCoefficients",
    "HistoryRecord",
    "MULTITRAP_HISTORY_COLUMNS",
    "MULTITRAP_PER_TARGET_COLUMNS",
    "MULTITRAP_SCHEMA_VERSION",
    "MULTITRAP_SUMMARY_COLUMNS",
    "MultitrapObjectiveConfig",
    "MultitrapRunResult",
    "ObjectiveEvaluation",
    "PerTargetRecord",
    "SPUComponents",
    "STANDARD_GORKOV_FORMULATION_ID",
    "SmoothScales",
    "SphereInFluid",
    "StageRecord",
    "TargetStencil",
    "conventional_double_trap_config",
    "effective_weight_force_target_n",
    "estimate_smooth_scales",
    "evaluate_multitrap_objective",
    "load_multitrap_cache",
    "multitrap_cache_payload",
    "regularized_fe_double_trap_config",
    "rows_from_results",
    "run_double_trap",
    "run_multitrap",
    "run_paired_double_trap_benchmark",
    "run_paired_spu_ablation",
    "save_multitrap_cache",
    "spu_ablation_configs",
    "standard_gorkov_coefficients",
]
