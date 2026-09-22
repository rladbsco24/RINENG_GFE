r"""Standard Gor'kov surrogate and phase-only HAT objectives.

This module is the single source of truth for every Gor'kov quantity used by
the revision notebook.  It deliberately contains *only* the standard
small-sphere, inviscid-host-fluid expression.  For a peak-amplitude pressure
phasor ``p`` (Pa),

.. math::

   U = K_p |p|^2 - K_\nabla |\nabla p|^2,

   K_p = \frac{V}{4}\left(\frac{1}{\rho_0c_0^2}
         -\frac{1}{\rho_pc_p^2}\right),\qquad
   K_\nabla = \frac{3V}{4}\frac{\rho_p-\rho_0}
         {\omega^2\rho_0(2\rho_p+\rho_0)}.

Equivalently, with the conventional contrast factors
``f1 = 1 - kappa_p/kappa_0`` and
``f2 = 2 (rho_p-rho_0)/(2 rho_p+rho_0)``, these coefficients are
``V f1/(4 rho_0 c_0^2)`` and ``3 V f2/(8 omega^2 rho_0)``.
The acoustic radiation force surrogate is ``F_G = -grad(U)``.  Thus the
velocity/pressure-gradient term is subtracted when the particle is denser
than the medium.  There is no legacy reversed-density convention here.

The formula assumes a Rayleigh sphere and an inviscid host.  The notebook's
finite-ka partial-wave calculation remains the independent validation layer;
the word "exact" must not be used for this surrogate.

Field convention
----------------
The forward model uses ``exp(+i k r)`` and peak complex amplitudes.  Replacing
this consistently by ``exp(-i k r)`` complex-conjugates the field and leaves
all Gor'kov quantities below unchanged.  Directivity is evaluated globally at
every requested point; it is never frozen around a differentiation centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np


Array = np.ndarray

SOUND_SPEED_M_S = 343.0
AIR_DENSITY_KG_M3 = 1.225
REFERENCE_FREQUENCY_HZ = 40_000.0
REFERENCE_STENCIL_SPACING_M = 0.5e-3
DEFAULT_PITCH_M = 10.0e-3
DEFAULT_PARTICLE_RADIUS_M = 0.65e-3
REFERENCE_SOURCE_STRENGTH_PA_M_PEAK = math.sqrt(2.0) * 20.0 * 0.30


# Murata MA40S4S amplitude shape used by the submitted HAT implementation.
# The historical notebook stored the on-axis entry as 100 and interpolated the
# table onto the integer 0--90 degree grid.  ``transfer_matrix`` retains that
# representation, then multiplies it by the single calibrated per-table-unit
# factor below.  Its on-axis source strength therefore exactly matches the
# finite-ka field: sqrt(2) * 20 Pa RMS * 0.30 m.
_MURATA_RAW_0_TO_90_EVERY_2_DEG = np.array(
    [
        100.000,
        96.695,
        93.488,
        93.488,
        93.488,
        90.388,
        87.579,
        84.676,
        84.676,
        79.183,
        76.551,
        76.551,
        71.624,
        71.624,
        69.282,
        67.007,
        64.885,
        62.610,
        60.581,
        58.652,
        56.657,
        53.009,
        49.598,
        47.958,
        46.411,
        41.952,
        39.243,
        35.496,
        32.140,
        29.052,
        26.306,
        24.650,
        24.650,
        23.820,
        23.820,
        23.820,
        23.054,
        22.309,
        21.626,
        20.172,
        18.240,
        17.615,
        15.414,
        13.509,
        11.832,
        10.000,
    ],
    dtype=np.float64,
)
MURATA_GAIN_0_TO_90 = np.interp(
    np.arange(91.0),
    np.arange(0.0, 91.0, 2.0),
    _MURATA_RAW_0_TO_90_EVERY_2_DEG,
)
MURATA_ON_AXIS_TABLE_AMPLITUDE = float(MURATA_GAIN_0_TO_90[0])
SOURCE_SCALE_PA_M_PER_MURATA_UNIT = (
    REFERENCE_SOURCE_STRENGTH_PA_M_PEAK / MURATA_ON_AXIS_TABLE_AMPLITUDE
)


def _as_points(value: Array | Sequence[Sequence[float]], *, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim < 2 or result.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3); got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _unit_rows(value: Array | Sequence[Sequence[float]], *, name: str) -> Array:
    result = _as_points(value, name=name)
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} contains a zero vector")
    return result / norms


@dataclass(frozen=True)
class AcousticMedium:
    """Homogeneous inviscid host-medium parameters in SI units."""

    density_kg_m3: float = AIR_DENSITY_KG_M3
    sound_speed_m_s: float = SOUND_SPEED_M_S

    def __post_init__(self) -> None:
        if self.density_kg_m3 <= 0.0 or self.sound_speed_m_s <= 0.0:
            raise ValueError("medium density and sound speed must be positive")


@dataclass(frozen=True)
class CompressibleSphere:
    """Rayleigh-sphere material and radius in SI units."""

    density_kg_m3: float = 100.0
    sound_speed_m_s: float = 2400.0
    radius_m: float = DEFAULT_PARTICLE_RADIUS_M

    def __post_init__(self) -> None:
        if self.density_kg_m3 <= 0.0 or self.sound_speed_m_s <= 0.0:
            raise ValueError("particle density and sound speed must be positive")
        if self.radius_m <= 0.0:
            raise ValueError("particle radius must be positive")

    @property
    def volume_m3(self) -> float:
        return 4.0 * math.pi * self.radius_m**3 / 3.0


@dataclass(frozen=True)
class GorkovCoefficients:
    """Coefficients in ``U = pressure*|p|^2 - gradient*|grad p|^2``.

    ``pressure_j_pa2`` has units J Pa^-2 and
    ``gradient_j_m2_pa2`` has units J m^2 Pa^-2.
    The second coefficient is signed through the physical density contrast.
    """

    pressure_j_pa2: float
    gradient_j_m2_pa2: float
    monopole_contrast_f1: float
    dipole_contrast_f2: float
    angular_frequency_rad_s: float


def gorkov_coefficients(
    frequency_hz: float,
    medium: AcousticMedium = AcousticMedium(),
    particle: CompressibleSphere = CompressibleSphere(),
) -> GorkovCoefficients:
    """Return standard peak-phasor Gor'kov coefficients.

    No sign is hard-coded from a particular particle.  A particle denser than
    the host has ``f2 > 0`` and therefore ``gradient_j_m2_pa2 > 0``; a lighter
    particle has the physically opposite sign.
    """

    frequency_hz = float(frequency_hz)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    rho0 = medium.density_kg_m3
    c0 = medium.sound_speed_m_s
    rho_p = particle.density_kg_m3
    c_p = particle.sound_speed_m_s
    omega = 2.0 * math.pi * frequency_hz
    volume = particle.volume_m3

    kappa0 = 1.0 / (rho0 * c0**2)
    kappa_p = 1.0 / (rho_p * c_p**2)
    f1 = 1.0 - kappa_p / kappa0
    f2 = 2.0 * (rho_p - rho0) / (2.0 * rho_p + rho0)

    pressure = volume * f1 / (4.0 * rho0 * c0**2)
    gradient = 3.0 * volume * f2 / (8.0 * omega**2 * rho0)
    return GorkovCoefficients(
        pressure_j_pa2=float(pressure),
        gradient_j_m2_pa2=float(gradient),
        monopole_contrast_f1=float(f1),
        dipole_contrast_f2=float(f2),
        angular_frequency_rad_s=float(omega),
    )


def gorkov_constants(
    frequency_hz: float,
    medium: AcousticMedium = AcousticMedium(),
    particle: CompressibleSphere = CompressibleSphere(),
) -> tuple[float, float]:
    """Compatibility tuple ``(K_p, K_gradient)`` using the standard signs."""

    coefficients = gorkov_coefficients(frequency_hz, medium, particle)
    return coefficients.pressure_j_pa2, coefficients.gradient_j_m2_pa2


@dataclass(frozen=True)
class ArrayGeometry:
    """Transducer centres, outward normals, and relative source weights."""

    positions_m: Array
    normals: Array | None = None
    output_weights: Array | None = None

    def __post_init__(self) -> None:
        positions = _as_points(self.positions_m, name="positions_m")
        if positions.ndim != 2:
            raise ValueError("positions_m must have shape (n_transducers, 3)")
        normals = (
            np.broadcast_to(np.array([0.0, 0.0, 1.0]), positions.shape).copy()
            if self.normals is None
            else _unit_rows(self.normals, name="normals")
        )
        if normals.shape != positions.shape:
            raise ValueError("normals must match positions_m")
        weights = (
            np.ones(positions.shape[0], dtype=np.float64)
            if self.output_weights is None
            else np.asarray(self.output_weights, dtype=np.float64)
        )
        if weights.shape != (positions.shape[0],):
            raise ValueError("output_weights must have shape (n_transducers,)")
        if not np.all(np.isfinite(weights)):
            raise ValueError("output_weights contains a non-finite value")
        object.__setattr__(self, "positions_m", positions.copy())
        object.__setattr__(self, "normals", normals.copy())
        object.__setattr__(self, "output_weights", weights.copy())

    @property
    def n_transducers(self) -> int:
        return int(self.positions_m.shape[0])

    @classmethod
    def square(
        cls,
        side: int = 16,
        pitch_m: float = DEFAULT_PITCH_M,
        *,
        z_m: float = 0.0,
        aspect: float = 1.0,
        normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> "ArrayGeometry":
        """Area-preserving rectangular deformation of a square array."""

        side = int(side)
        if side <= 0 or pitch_m <= 0.0 or aspect <= 0.0:
            raise ValueError("side, pitch_m, and aspect must be positive")
        axis = np.arange(side, dtype=np.float64) - (side - 1.0) / 2.0
        sx, sy = math.sqrt(aspect), 1.0 / math.sqrt(aspect)
        positions = np.array(
            [(sx * pitch_m * x, sy * pitch_m * y, z_m) for x in axis for y in axis],
            dtype=np.float64,
        )
        normals = np.broadcast_to(np.asarray(normal, dtype=np.float64), positions.shape).copy()
        return cls(positions_m=positions, normals=normals)


@dataclass(frozen=True)
class MurataMA40S4SDirectivity:
    """Linear interpolation of the manuscript's 0--90 degree amplitude table.

    ``rear_policy='zero'`` treats each element as front-radiating and is the
    default.  ``rear_policy='absolute'`` explicitly declares a two-sided
    element and evaluates the table using ``abs(cos(theta))``.  No unlabelled
    clipping of rear angles to the 90-degree table edge is permitted.
    """

    rear_policy: Literal["zero", "absolute"] = "zero"
    table: Array = field(default_factory=lambda: MURATA_GAIN_0_TO_90.copy())

    def __post_init__(self) -> None:
        table = np.asarray(self.table, dtype=np.float64)
        if table.shape != (91,) or not np.all(np.isfinite(table)):
            raise ValueError("directivity table must contain 91 finite values")
        if self.rear_policy not in ("zero", "absolute"):
            raise ValueError("rear_policy must be 'zero' or 'absolute'")
        object.__setattr__(self, "table", table.copy())

    def gain(self, theta_deg: Array) -> Array:
        theta = np.asarray(theta_deg, dtype=np.float64)
        if self.rear_policy == "absolute":
            theta = np.rad2deg(np.arccos(np.abs(np.cos(np.deg2rad(theta)))))
        clipped = np.clip(theta, 0.0, 90.0)
        result = np.interp(clipped, np.arange(91.0), self.table)
        if self.rear_policy == "zero":
            result = np.where(theta <= 90.0, result, 0.0)
        return result


DEFAULT_DIRECTIVITY = MurataMA40S4SDirectivity()


def transfer_matrix(
    points_m: Array,
    geometry: ArrayGeometry,
    frequency_hz: float,
    *,
    medium: AcousticMedium = AcousticMedium(),
    directivity: MurataMA40S4SDirectivity = DEFAULT_DIRECTIVITY,
    source_scale_pa_m: float = SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
    minimum_radius_m: float = 1.0e-9,
) -> Array:
    """Return the global point-to-transducer pressure transfer matrix.

    The result has shape ``points_m.shape[:-1] + (n_transducers,)``.  A single
    call evaluates the directivity at every spatial point, including points in
    derivative stencils.  This prevents the locally frozen-directivity error
    present in older validation prototypes.
    """

    points = _as_points(points_m, name="points_m")
    if frequency_hz <= 0.0 or source_scale_pa_m <= 0.0 or minimum_radius_m <= 0.0:
        raise ValueError("frequency, source scale, and minimum radius must be positive")
    delta = points[..., None, :] - geometry.positions_m
    radius = np.linalg.norm(delta, axis=-1)
    radius_safe = np.maximum(radius, float(minimum_radius_m))
    cos_theta = np.sum(delta * geometry.normals, axis=-1) / radius_safe
    theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    amplitude = (
        float(source_scale_pa_m)
        * directivity.gain(theta_deg)
        * geometry.output_weights
        / radius_safe
    )
    wavenumber = 2.0 * math.pi * float(frequency_hz) / medium.sound_speed_m_s
    return amplitude * np.exp(1j * wavenumber * radius_safe)


def pressure_field(
    phases_rad: Array,
    points_m: Array,
    geometry: ArrayGeometry,
    frequency_hz: float,
    **transfer_kwargs: object,
) -> Array:
    """Evaluate the complex peak-pressure phasor at arbitrary points."""

    phases = np.asarray(phases_rad, dtype=np.float64)
    if phases.shape != (geometry.n_transducers,):
        raise ValueError("phases_rad must have one phase per transducer")
    transfer = transfer_matrix(
        points_m,
        geometry,
        frequency_hz,
        **transfer_kwargs,
    )
    return np.einsum("...n,n->...", transfer, np.exp(1j * phases), optimize=True)


def wrap_phase(phases_rad: Array) -> Array:
    return (np.asarray(phases_rad, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def gauge_full(reduced_phases_rad: Array, gauge_index: int = 0) -> Array:
    """Insert a zero global-phase gauge coordinate."""

    reduced = np.asarray(reduced_phases_rad, dtype=np.float64)
    if reduced.ndim != 1:
        raise ValueError("reduced phases must be one-dimensional")
    if not 0 <= int(gauge_index) <= reduced.size:
        raise ValueError("gauge_index is out of range")
    return np.insert(reduced, int(gauge_index), 0.0)


def reduce_gauge(phases_rad: Array, gauge_index: int = 0) -> Array:
    """Remove global phase by setting ``phases[gauge_index]`` to zero."""

    phases = np.asarray(phases_rad, dtype=np.float64)
    if phases.ndim != 1 or not 0 <= int(gauge_index) < phases.size:
        raise ValueError("invalid phases or gauge_index")
    gauged = wrap_phase(phases - phases[int(gauge_index)])
    return np.delete(gauged, int(gauge_index))


def spatial_gradient(field: Array, spacing_m: float) -> tuple[Array, Array, Array]:
    """Central finite-difference gradient on the first three axes."""

    values = np.asarray(field)
    if values.ndim < 3 or any(size < 3 for size in values.shape[:3]):
        raise ValueError("field must have at least three points on each spatial axis")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    return tuple(
        np.gradient(values, spacing_m, axis=axis, edge_order=1) for axis in range(3)
    )  # type: ignore[return-value]


def scalar_hessian_at(
    scalar_field: Array,
    spacing_m: float,
    center: tuple[int, int, int],
) -> Array:
    """Symmetric 3x3 Hessian of a scalar grid at ``center``."""

    first = spatial_gradient(scalar_field, spacing_m)
    hessian = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            hessian[i, j] = np.gradient(
                first[i], spacing_m, axis=j, edge_order=1
            )[center]
    return 0.5 * (hessian + hessian.T)


def laplacian_at(
    scalar_field: Array,
    spacing_m: float,
    center: tuple[int, int, int],
    axis_weights: Array | Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Weighted Hessian trace, using the same derivative operator as force."""

    weights = np.asarray(axis_weights, dtype=np.float64)
    if weights.shape != (3,):
        raise ValueError("axis_weights must contain three values")
    return float(np.dot(weights, np.diag(scalar_hessian_at(scalar_field, spacing_m, center))))


def gorkov_potential_from_pressure(
    pressure: Array,
    spacing_m: float,
    coefficients: GorkovCoefficients,
) -> Array:
    """Evaluate standard Gor'kov potential on a regular pressure grid."""

    p = np.asarray(pressure, dtype=np.complex128)
    dp = spatial_gradient(p, spacing_m)
    gradient_energy = sum(np.abs(component) ** 2 for component in dp)
    return (
        coefficients.pressure_j_pa2 * np.abs(p) ** 2
        - coefficients.gradient_j_m2_pa2 * gradient_energy
    )


@dataclass(frozen=True)
class Condition:
    """One target-local optimization condition."""

    label: str
    positions: Array
    target_m: tuple[float, float, float]
    frequency_hz: float = REFERENCE_FREQUENCY_HZ
    curvature_weight: Array = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    normals: Array | None = None
    output_weights: Array | None = None
    path: str = "single-target"
    s: float = 0.0

    def __post_init__(self) -> None:
        geometry = ArrayGeometry(self.positions, self.normals, self.output_weights)
        target = np.asarray(self.target_m, dtype=np.float64)
        weight = np.asarray(self.curvature_weight, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_m must contain three finite coordinates")
        if weight.shape != (3, 3) or not np.all(np.isfinite(weight)):
            raise ValueError("curvature_weight must be a finite 3x3 tensor")
        if not np.allclose(weight, weight.T, rtol=0.0, atol=1.0e-13):
            raise ValueError("curvature_weight must be symmetric")
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        object.__setattr__(self, "positions", geometry.positions_m)
        object.__setattr__(self, "normals", geometry.normals)
        object.__setattr__(self, "output_weights", geometry.output_weights)
        object.__setattr__(self, "target_m", tuple(float(v) for v in target))
        object.__setattr__(self, "curvature_weight", weight.copy())

    @property
    def geometry(self) -> ArrayGeometry:
        return ArrayGeometry(self.positions, self.normals, self.output_weights)

    @property
    def target(self) -> tuple[float, float, float]:
        """Compatibility alias used by the corrected branch scripts."""

        return self.target_m

    @property
    def frequency(self) -> float:
        return self.frequency_hz

    @property
    def weight(self) -> Array:
        return self.curvature_weight

    @property
    def wavelength_m(self) -> float:
        return SOUND_SPEED_M_S / self.frequency_hz


@dataclass(frozen=True)
class MethodSpec:
    """Dimensional single-target objective definition.

    The minimized objective is

    ``-W:Hess(U) + alpha_per_m*|grad(U)| + beta*pressure_penalty``.

    ``alpha_per_m`` has units m^-1.  ``beta_curvature_per_pa`` has units
    N m^-1 Pa^-1 so that every term has curvature units N m^-1.
    """

    name: str
    alpha_per_m: float
    beta_curvature_per_pa: float
    pressure_mode: Literal["abs", "smooth_abs"]
    curvature_weight: Array

    def __post_init__(self) -> None:
        weight = np.asarray(self.curvature_weight, dtype=np.float64)
        if weight.shape != (3, 3) or not np.allclose(weight, weight.T):
            raise ValueError("curvature_weight must be symmetric 3x3")
        if self.alpha_per_m < 0.0 or self.beta_curvature_per_pa < 0.0:
            raise ValueError("objective weights must be non-negative")
        object.__setattr__(self, "curvature_weight", weight.copy())

    @property
    def alpha(self) -> float:
        return self.alpha_per_m

    @property
    def beta(self) -> float:
        return self.beta_curvature_per_pa


def method_spec(name: str) -> MethodSpec:
    """Return the two manuscript methods retained in the revised main text."""

    key = str(name).strip().lower().replace("_", "-")
    if key in {"conventional", "c", "conv"}:
        return MethodSpec(
            name="Conventional",
            alpha_per_m=0.0,
            beta_curvature_per_pa=1.0,
            pressure_mode="abs",
            curvature_weight=np.diag([1000.0, 1000.0, 10.0]),
        )
    if key in {"force-equilibrium", "fe"}:
        return MethodSpec(
            name="Force-Equilibrium",
            alpha_per_m=10.0,
            beta_curvature_per_pa=0.0,
            pressure_mode="abs",
            curvature_weight=np.eye(3, dtype=np.float64),
        )
    raise KeyError(f"unknown method {name!r}; expected Conventional or Force-Equilibrium")


def default_stencil_spacing_m(
    frequency_hz: float,
    medium: AcousticMedium = AcousticMedium(),
) -> float:
    """Keep the corrected branch code's fixed ``h/lambda`` across frequency."""

    reference_wavelength = SOUND_SPEED_M_S / REFERENCE_FREQUENCY_HZ
    wavelength = medium.sound_speed_m_s / float(frequency_hz)
    return REFERENCE_STENCIL_SPACING_M * wavelength / reference_wavelength


class SingleTargetObjective:
    """Analytic phase gradient for the standard Gor'kov C/FE objective.

    Spatial derivatives are all obtained from one target-centred ``5^3``
    pressure stencil.  Directivity is recomputed globally for every stencil
    point.  ``fun_grad`` accepts the ``n-1`` phase vector with one global phase
    fixed to zero; ``full_fun_grad`` accepts all ``n`` phases.
    """

    def __init__(
        self,
        condition: Condition,
        method: MethodSpec | str = "Force-Equilibrium",
        *,
        medium: AcousticMedium = AcousticMedium(),
        particle: CompressibleSphere = CompressibleSphere(),
        directivity: MurataMA40S4SDirectivity = DEFAULT_DIRECTIVITY,
        source_scale_pa_m: float = SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
        stencil_spacing_m: float | None = None,
        half_stencil: int = 2,
        smooth_pressure_relative: float = 1.0e-3,
        gauge_index: int = 0,
        force_target_n: Array | None = None,
    ) -> None:
        self.condition = condition
        self.method = method_spec(method) if isinstance(method, str) else method
        self.medium = medium
        self.particle = particle
        self.directivity = directivity
        self.source_scale_pa_m = float(source_scale_pa_m)
        self.half_stencil = int(half_stencil)
        self.smooth_pressure_relative = float(smooth_pressure_relative)
        self.gauge_index = int(gauge_index)
        self.force_target_n = np.zeros(3) if force_target_n is None else np.asarray(force_target_n, dtype=float).copy()
        if self.force_target_n.shape != (3,) or not np.all(np.isfinite(self.force_target_n)):
            raise ValueError("force_target_n must be a finite three-vector")
        if self.half_stencil < 2:
            raise ValueError("half_stencil must be at least 2 for target Hessians")
        if self.smooth_pressure_relative < 0.0:
            raise ValueError("smooth_pressure_relative must be non-negative")
        if not 0 <= self.gauge_index < condition.geometry.n_transducers:
            raise ValueError("gauge_index is out of range")
        self.spacing_m = (
            default_stencil_spacing_m(condition.frequency_hz, medium)
            if stencil_spacing_m is None
            else float(stencil_spacing_m)
        )
        if self.spacing_m <= 0.0:
            raise ValueError("stencil_spacing_m must be positive")
        self.coefficients = gorkov_coefficients(
            condition.frequency_hz, medium, particle
        )
        self.shape = (2 * self.half_stencil + 1,) * 3
        self.center = (self.half_stencil,) * 3
        self.points_m = self._stencil_points()
        self.transfer = transfer_matrix(
            self.points_m,
            condition.geometry,
            condition.frequency_hz,
            medium=medium,
            directivity=directivity,
            source_scale_pa_m=self.source_scale_pa_m,
        )
        self.n_transducers = condition.geometry.n_transducers

    def _stencil_points(self) -> Array:
        offsets = (
            np.arange(-self.half_stencil, self.half_stencil + 1, dtype=np.float64)
            * self.spacing_m
        )
        target = np.asarray(self.condition.target_m, dtype=np.float64)
        x, y, z = np.meshgrid(offsets, offsets, offsets, indexing="ij")
        return target + np.stack((x, y, z), axis=-1)

    def _raw_metrics(self, phases_rad: Array, need_phase_gradient: bool) -> dict[str, Array | float]:
        phases = np.asarray(phases_rad, dtype=np.float64)
        if phases.shape != (self.n_transducers,):
            raise ValueError("phases_rad must have one phase per transducer")
        actuator = np.exp(1j * phases)
        pressure = np.einsum("...n,n->...", self.transfer, actuator, optimize=True)
        pressure_phase_jacobian = None
        if need_phase_gradient:
            pressure_phase_jacobian = 1j * self.transfer * actuator

        dp = spatial_gradient(pressure, self.spacing_m)
        d_dp = None
        if need_phase_gradient:
            d_dp = spatial_gradient(pressure_phase_jacobian, self.spacing_m)

        kp = self.coefficients.pressure_j_pa2
        kg = self.coefficients.gradient_j_m2_pa2
        potential = kp * np.abs(pressure) ** 2
        for component in dp:
            potential -= kg * np.abs(component) ** 2

        potential_phase_jacobian = None
        if need_phase_gradient:
            potential_phase_jacobian = 2.0 * kp * np.real(
                np.conj(pressure)[..., None] * pressure_phase_jacobian
            )
            for component, component_jacobian in zip(dp, d_dp):
                potential_phase_jacobian -= 2.0 * kg * np.real(
                    np.conj(component)[..., None] * component_jacobian
                )

        grad_u_fields = spatial_gradient(potential, self.spacing_m)
        gradient_u = np.array(
            [component[self.center] for component in grad_u_fields], dtype=np.float64
        )
        hessian_u = scalar_hessian_at(potential, self.spacing_m, self.center)
        force_gorkov = -gradient_u
        force_norm = float(np.linalg.norm(force_gorkov))
        residual_gradient_u = gradient_u + self.force_target_n
        force_residual_norm = float(np.linalg.norm(residual_gradient_u))
        curvature = float(np.sum(self.method.curvature_weight * hessian_u))
        pressure_abs = float(abs(pressure[self.center]))
        result: dict[str, Array | float] = {
            "pressure": pressure,
            "potential_j": potential,
            "gorkov_force_n": force_gorkov,
            "force_norm_n": force_norm,
            "force_residual_norm_n": force_residual_norm,
            "potential_hessian_n_m": hessian_u,
            "weighted_curvature_n_m": curvature,
            "pressure_abs_pa": pressure_abs,
        }
        if not need_phase_gradient:
            return result

        gradient_of_du = spatial_gradient(potential_phase_jacobian, self.spacing_m)
        d_gradient_u = np.stack(
            [component[self.center] for component in gradient_of_du], axis=0
        )
        d_hessian_u = np.empty(
            (3, 3, self.n_transducers), dtype=np.float64
        )
        for i in range(3):
            for j in range(3):
                d_hessian_u[i, j] = np.gradient(
                    gradient_of_du[i],
                    self.spacing_m,
                    axis=j,
                    edge_order=1,
                )[self.center]
        d_hessian_u = 0.5 * (
            d_hessian_u + np.swapaxes(d_hessian_u, 0, 1)
        )
        d_curvature = np.einsum(
            "ij,ijn->n", self.method.curvature_weight, d_hessian_u, optimize=True
        )
        # |F_G| == |grad U|, so using gradient_u gives the same derivative.
        d_force_norm = np.einsum(
            "i,in->n", residual_gradient_u, d_gradient_u, optimize=True
        ) / max(force_residual_norm, 1.0e-30)

        p0 = pressure[self.center]
        q0 = pressure_phase_jacobian[self.center]
        if self.method.pressure_mode == "smooth_abs":
            local_rms = math.sqrt(
                max(float(np.mean(np.abs(pressure) ** 2)), 1.0e-64)
            )
            d_local_rms = np.mean(
                np.real(np.conj(pressure)[..., None] * pressure_phase_jacobian),
                axis=(0, 1, 2),
            ) / max(local_rms, 1.0e-32)
            epsilon = self.smooth_pressure_relative * (local_rms + 1.0e-32)
            d_epsilon = self.smooth_pressure_relative * d_local_rms
            pressure_penalty = math.sqrt(
                max(float(abs(p0) ** 2 + epsilon**2), 1.0e-64)
            )
            d_pressure_penalty = (
                np.real(np.conj(p0) * q0) + epsilon * d_epsilon
            ) / max(pressure_penalty, 1.0e-32)
        else:
            pressure_penalty = pressure_abs
            d_pressure_penalty = np.real(np.conj(p0) * q0) / max(
                pressure_abs, 1.0e-32
            )
        result.update(
            {
                "pressure_penalty_pa": float(pressure_penalty),
                "d_weighted_curvature_d_phase": d_curvature,
                "d_force_norm_d_phase": d_force_norm,
                "d_pressure_penalty_d_phase": d_pressure_penalty,
                "d_gorkov_force_d_phase": -d_gradient_u,
            }
        )
        return result

    def full_fun_grad(self, phases_rad: Array) -> tuple[float, Array]:
        """Return dimensional loss and analytic derivative for all phases."""

        raw = self._raw_metrics(phases_rad, need_phase_gradient=True)
        pressure_penalty = float(raw.get("pressure_penalty_pa", raw["pressure_abs_pa"]))
        value = (
            -float(raw["weighted_curvature_n_m"])
            + self.method.alpha_per_m * float(raw["force_residual_norm_n"])
            + self.method.beta_curvature_per_pa * pressure_penalty
        )
        gradient = (
            -np.asarray(raw["d_weighted_curvature_d_phase"], dtype=np.float64)
            + self.method.alpha_per_m
            * np.asarray(raw["d_force_norm_d_phase"], dtype=np.float64)
            + self.method.beta_curvature_per_pa
            * np.asarray(raw["d_pressure_penalty_d_phase"], dtype=np.float64)
        )
        return float(value), gradient

    def full_value(self, phases_rad: Array) -> float:
        """Return the same dimensional loss without building phase Jacobians.

        Local decision-section figures evaluate thousands of nearby phase
        commands.  Their ordinate needs only the scalar objective, so forming
        the ``5 x 5 x 5 x N`` analytic phase Jacobian at every pixel would add
        substantial cost without changing the plotted quantity.
        """

        raw = self._raw_metrics(phases_rad, need_phase_gradient=False)
        pressure_abs = float(raw["pressure_abs_pa"])
        if self.method.pressure_mode == "smooth_abs":
            pressure = np.asarray(raw["pressure"], dtype=np.complex128)
            local_rms = math.sqrt(
                max(float(np.mean(np.abs(pressure) ** 2)), 1.0e-64)
            )
            epsilon = self.smooth_pressure_relative * (local_rms + 1.0e-32)
            pressure_penalty = math.sqrt(
                max(pressure_abs**2 + epsilon**2, 1.0e-64)
            )
        else:
            pressure_penalty = pressure_abs
        return float(
            -float(raw["weighted_curvature_n_m"])
            + self.method.alpha_per_m * float(raw["force_residual_norm_n"])
            + self.method.beta_curvature_per_pa * pressure_penalty
        )

    def fun_grad(self, reduced_phases_rad: Array) -> tuple[float, Array]:
        full = gauge_full(reduced_phases_rad, self.gauge_index)
        value, gradient = self.full_fun_grad(full)
        return value, np.delete(gradient, self.gauge_index)

    def metrics(self, reduced_phases_rad: Array) -> dict[str, Array | float]:
        full = gauge_full(reduced_phases_rad, self.gauge_index)
        return self._raw_metrics(full, need_phase_gradient=False)


# Concise compatibility name for corrected branch scripts; it is not an exact
# force model and is never named as such.
LocalObjective = SingleTargetObjective


def finite_difference_phase_gradient(
    objective: SingleTargetObjective,
    reduced_phases_rad: Array,
    *,
    step_rad: float = 1.0e-6,
) -> Array:
    """Independent centred difference used only for verification/tests."""

    x = np.asarray(reduced_phases_rad, dtype=np.float64)
    gradient = np.empty_like(x)
    for index in range(x.size):
        delta = np.zeros_like(x)
        delta[index] = step_rad
        plus, _ = objective.fun_grad(x + delta)
        minus, _ = objective.fun_grad(x - delta)
        gradient[index] = (plus - minus) / (2.0 * step_rad)
    return gradient


def formula_self_test() -> dict[str, float | bool]:
    """Fast, deterministic checks suitable for an early notebook guard cell."""

    medium = AcousticMedium()
    particle = CompressibleSphere()
    coefficients = gorkov_coefficients(REFERENCE_FREQUENCY_HZ, medium, particle)
    dense_sign_ok = coefficients.gradient_j_m2_pa2 > 0.0

    # A constant complex pressure has neither velocity nor radiation force.
    pressure = np.full((5, 5, 5), 2.0 + 3.0j, dtype=np.complex128)
    potential = gorkov_potential_from_pressure(
        pressure, REFERENCE_STENCIL_SPACING_M, coefficients
    )
    constant_error = float(np.max(np.abs(potential - potential[2, 2, 2])))
    force = np.array(
        [component[2, 2, 2] for component in spatial_gradient(potential, REFERENCE_STENCIL_SPACING_M)]
    )
    constant_force_norm = float(np.linalg.norm(force))

    return {
        "dense_particle_gradient_coefficient_positive": bool(dense_sign_ok),
        "constant_pressure_potential_spread_j": constant_error,
        "constant_pressure_force_norm_n": constant_force_norm,
        "passed": bool(
            dense_sign_ok
            and constant_error <= 1.0e-30
            and constant_force_norm <= 1.0e-30
        ),
    }


__all__ = [
    "AIR_DENSITY_KG_M3",
    "AcousticMedium",
    "ArrayGeometry",
    "CompressibleSphere",
    "Condition",
    "DEFAULT_DIRECTIVITY",
    "DEFAULT_PARTICLE_RADIUS_M",
    "DEFAULT_PITCH_M",
    "GorkovCoefficients",
    "LocalObjective",
    "MURATA_GAIN_0_TO_90",
    "MethodSpec",
    "MurataMA40S4SDirectivity",
    "REFERENCE_FREQUENCY_HZ",
    "REFERENCE_SOURCE_STRENGTH_PA_M_PEAK",
    "REFERENCE_STENCIL_SPACING_M",
    "SOUND_SPEED_M_S",
    "SOURCE_SCALE_PA_M_PER_MURATA_UNIT",
    "SingleTargetObjective",
    "default_stencil_spacing_m",
    "finite_difference_phase_gradient",
    "formula_self_test",
    "gauge_full",
    "gorkov_coefficients",
    "gorkov_constants",
    "gorkov_potential_from_pressure",
    "laplacian_at",
    "method_spec",
    "pressure_field",
    "reduce_gauge",
    "scalar_hessian_at",
    "spatial_gradient",
    "transfer_matrix",
    "wrap_phase",
]
