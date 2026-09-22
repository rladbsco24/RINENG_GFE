"""Common static acoustic-trap validation on one globally defined field.

This module keeps three operations separate:

1. ``ArbitraryArrayPressureField`` defines one pressure phasor ``pressure(x)``
   over the whole computational domain.  Element directivity is evaluated at
   every requested point; it is never frozen around a particle or expansion
   centre.
2. ``StandardGorkovEvaluator`` implements the standard Rayleigh-particle
   Gor'kov potential for peak phasors.  It is an optimisation surrogate and a
   comparator, not the finite-``ka`` reference.
3. ``PartialWaveForceEvaluator`` projects that same incident field onto regular
   spherical waves, applies the homogeneous isotropic elastic-solid-sphere
   scattering coefficients, and integrates the total-minus-incident momentum
   flux.

Only static, directly reported quantities are returned: equilibrium location,
displacement, force residual, the full Cartesian force Jacobian, stiffness
about the *found equilibrium*, and sampled force sections.  There are no pass
criteria, dynamics gates, or phase recentering operations in this module.

Phasor convention
-----------------
``p(x,t) = Re[p_hat(x) exp(-i omega t)]``.  An outgoing source is therefore
proportional to ``exp(+i k r) / r`` and ``v_hat = grad(p_hat)/(i omega rho_0)``.

Model scope
-----------
The partial-wave reference is scattering-inclusive and finite-``ka`` for a
homogeneous isotropic elastic solid sphere in an inviscid host.  Both internal
longitudinal and shear modes are included.  It is not a thermoviscous,
streaming, nearby-wall, particle-particle multiple-scattering, or nonspherical
particle model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
import math
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import j1, sph_harm_y, spherical_jn, spherical_yn

from .gorkov_core import REFERENCE_SOURCE_STRENGTH_PA_M_PEAK


Array = np.ndarray

PEAK_PHASOR_PROVENANCE = {
    "amplitude_convention": "peak pressure phasor",
    "time_dependence": "p(x,t) = Re[p_hat(x) exp(-i omega t)]",
    "outgoing_wave": "exp(+i k r) / r",
}


class GlobalPressureField(Protocol):
    """A single pressure phasor field defined in global Cartesian coordinates."""

    def pressure(self, points_m: Array) -> Array:
        """Return complex peak-pressure phasors [Pa] for points of shape ``(M,3)``."""


@dataclass(frozen=True)
class Medium:
    density_kg_m3: float = 1.225
    sound_speed_m_s: float = 343.0
    gravity_m_s2: float = 9.80665
    dynamic_viscosity_pa_s: float = 1.854e-5


@dataclass(frozen=True)
class ElasticSphere:
    """Declared homogeneous-isotropic elastic bead used by the validator.

    The density and wave speeds are explicit simulation inputs for a solid
    bead.  The defaults retain the previously declared bead
    density and longitudinal speed and add the independently required shear
    speed.  Quantitative experiment matching must replace these declared
    values with bead-specific measurements.
    """

    radius_m: float = 0.65e-3
    density_kg_m3: float = 100.0
    longitudinal_sound_speed_m_s: float = 2400.0
    shear_sound_speed_m_s: float = 1150.0
    material_label: str = "declared homogeneous isotropic elastic bead"
    parameter_provenance: str = (
        "declared simulation defaults; replace density and elastic wave speeds "
        "with measured bead values for quantitative experiment matching"
    )

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("elastic-sphere radius must be positive")
        if self.density_kg_m3 <= 0.0:
            raise ValueError("elastic-sphere density must be positive")
        if self.longitudinal_sound_speed_m_s <= 0.0:
            raise ValueError("longitudinal wave speed must be positive")
        if self.shear_sound_speed_m_s <= 0.0:
            raise ValueError("shear wave speed must be positive")
        if self.bulk_modulus_pa <= 0.0:
            raise ValueError(
                "elastic wave speeds imply a non-positive bulk modulus; "
                "require c_L^2 > 4 c_T^2 / 3"
            )

    @property
    def volume_m3(self) -> float:
        return 4.0 * math.pi * self.radius_m**3 / 3.0

    @property
    def sound_speed_m_s(self) -> float:
        """Compatibility name for the longitudinal elastic-wave speed."""

        return self.longitudinal_sound_speed_m_s

    @property
    def shear_modulus_pa(self) -> float:
        return self.density_kg_m3 * self.shear_sound_speed_m_s**2

    @property
    def lame_first_parameter_pa(self) -> float:
        return self.density_kg_m3 * (
            self.longitudinal_sound_speed_m_s**2
            - 2.0 * self.shear_sound_speed_m_s**2
        )

    @property
    def bulk_modulus_pa(self) -> float:
        return self.lame_first_parameter_pa + 2.0 * self.shear_modulus_pa / 3.0

    @property
    def young_modulus_pa(self) -> float:
        lame = self.lame_first_parameter_pa
        shear = self.shear_modulus_pa
        return shear * (3.0 * lame + 2.0 * shear) / (lame + shear)

    @property
    def poisson_ratio(self) -> float:
        lame = self.lame_first_parameter_pa
        shear = self.shear_modulus_pa
        return lame / (2.0 * (lame + shear))

    @property
    def compressibility_pa_inverse(self) -> float:
        return 1.0 / self.bulk_modulus_pa


@dataclass(frozen=True)
class PartialWaveNumerics:
    """Numerics for spherical-wave fitting and stress integration."""

    lmax: int = 8
    fit_shell_wavelengths: tuple[float, ...] = (0.10, 0.15, 0.20)
    fit_n_mu: int = 12
    fit_n_phi: int = 24
    control_radius_a: float = 2.0
    surface_n_mu: int = 20
    surface_n_phi: int = 40
    velocity_step_a: float = 2.0e-3
    fit_rcond: float = 1.0e-12


@dataclass(frozen=True)
class GorkovNumerics:
    """Central-difference scales relative to the particle radius."""

    pressure_gradient_step_a: float = 0.02
    potential_gradient_step_a: float = 0.05


@dataclass(frozen=True)
class ViscousRayleighNumerics:
    """Central-difference scales for the viscous Rayleigh sensitivity."""

    pressure_gradient_step_a: float = 0.02
    velocity_gradient_step_a: float = 0.05
    regular_wave_derivatives: bool = True


@dataclass(frozen=True)
class ForceEvaluation:
    """Raw static-force decomposition at one point."""

    radiation_n: Array
    effective_gravity_n: Array
    total_n: Array
    metadata: dict = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class EquilibriumCandidate:
    initial_offset_a: Array
    solution_offset_a: Array
    equilibrium_m: Array
    residual_force_n: Array
    residual_scaled_norm: float
    optimizer_success: bool
    on_search_boundary: bool
    nfev: int
    message: str


@dataclass(frozen=True)
class EquilibriumResult:
    """Nearest numerically resolved root selected from common prescribed starts."""

    target_m: Array
    equilibrium_m: Array
    displacement_m: Array
    displacement_norm_m: float
    displacement_norm_a: float
    residual_force_n: Array
    residual_scaled_norm: float
    numerical_root_found: bool
    selected_start_offset_a: Array
    on_search_boundary: bool
    nfev: int
    candidates: tuple[EquilibriumCandidate, ...]


@dataclass(frozen=True)
class StaticValidationResult:
    """Static quantities evaluated at the requested target and found equilibrium."""

    model_name: str
    equilibrium: EquilibriumResult
    target_force_n: Array
    equilibrium_force_n: Array
    force_jacobian_n_m: Array
    symmetric_stiffness_n_m: Array
    symmetric_stiffness_eigenvalues_n_m: Array
    antisymmetric_force_jacobian_n_m: Array
    jacobian_step_m: float
    metadata: dict = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class StaticModelComparison:
    finite_ka: StaticValidationResult
    gorkov: StaticValidationResult
    equilibrium_difference_m: Array
    equilibrium_difference_norm_m: float
    stiffness_difference_n_m: Array


@dataclass(frozen=True)
class ForceSection:
    """Pressure and force samples in one Cartesian plane."""

    center_m: Array
    axes: tuple[int, int]
    coordinates_u_m: Array
    coordinates_v_m: Array
    points_m: Array
    pressure_pa_peak: Array
    radiation_force_n: Array
    total_force_n: Array

    @property
    def in_plane_total_force_n(self) -> Array:
        return self.total_force_n[..., list(self.axes)]


_MURATA_DIRECTIVITY_PERCENT = np.array(
    [
        100.000, 96.695, 93.488, 93.488, 93.488, 90.388, 87.579, 84.676,
        84.676, 79.183, 76.551, 76.551, 71.624, 71.624, 69.282, 67.007,
        64.885, 62.610, 60.581, 58.652, 56.657, 53.009, 49.598, 47.958,
        46.411, 41.952, 39.243, 35.496, 32.140, 29.052, 26.306, 24.650,
        24.650, 23.820, 23.820, 23.820, 23.054, 22.309, 21.626, 20.172,
        18.240, 17.615, 15.414, 13.509, 11.832, 10.000,
    ],
    dtype=float,
)


class ArbitraryArrayPressureField:
    """Global point-source array field with per-element positions and normals.

    ``weights`` are complex drive weights.  ``normals`` set the acoustic front
    direction of each element and make the same implementation usable for
    single-sided, opposed, tilted, and non-planar arrays.  The default source
    strength converts the nominal 120 dB SPL re 20 uPa RMS at 0.30 m into a
    peak-pressure phasor times distance.
    """

    def __init__(
        self,
        frequency_hz: float,
        positions_m: Array,
        weights: Array,
        *,
        normals: Array | None = None,
        sound_speed_m_s: float = 343.0,
        source_strength_pa_m: float = REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
        directivity: str = "murata_table",
        piston_radius_m: float = 5.0e-3,
        front_only: bool = True,
        source_calibration_label: str = (
            "120 dB SPL re 20 uPa RMS at 0.30 m on axis; converted to peak phasor"
        ),
    ) -> None:
        positions = np.asarray(positions_m, dtype=float).reshape(-1, 3)
        drives = np.asarray(weights, dtype=np.complex128).reshape(-1)
        if len(positions) != len(drives):
            raise ValueError("positions_m and weights must contain the same number of elements")
        if normals is None:
            normal_array = np.tile(np.array([0.0, 0.0, 1.0]), (len(positions), 1))
        else:
            normal_array = np.asarray(normals, dtype=float).reshape(-1, 3)
            if len(normal_array) != len(positions):
                raise ValueError("normals must have shape (number_of_elements, 3)")
        normal_norms = np.linalg.norm(normal_array, axis=1)
        if np.any(normal_norms == 0.0):
            raise ValueError("element normals must be nonzero")

        self.frequency_hz = float(frequency_hz)
        # Validation represents a frozen terminal command.  Own immutable
        # copies so later mutation of a caller's phase, weight, geometry, or
        # normal arrays cannot silently change an already-created field.
        self.positions_m = positions.copy()
        self.weights = drives.copy()
        self.normals = (normal_array / normal_norms[:, None]).copy()
        self.positions_m.setflags(write=False)
        self.weights.setflags(write=False)
        self.normals.setflags(write=False)
        self.sound_speed_m_s = float(sound_speed_m_s)
        self.k_rad_m = 2.0 * math.pi * self.frequency_hz / self.sound_speed_m_s
        self.source_strength_pa_m = float(source_strength_pa_m)
        self.directivity_mode = str(directivity)
        self.piston_radius_m = float(piston_radius_m)
        self.front_only = bool(front_only)
        self.source_calibration_label = str(source_calibration_label)

    @staticmethod
    def rectangular_positions(side: int = 16, pitch_m: float = 0.010) -> Array:
        axis = (np.arange(int(side)) - (int(side) - 1.0) / 2.0) * float(pitch_m)
        return np.array([(x, y, 0.0) for x in axis for y in axis], dtype=float)

    def _directivity(self, cos_theta: Array) -> Array:
        cosine = np.asarray(cos_theta, dtype=float)
        front_cosine = np.clip(cosine, 0.0, 1.0)
        sin_theta = np.sqrt(np.maximum(1.0 - front_cosine**2, 0.0))
        if self.directivity_mode == "isotropic":
            value = np.ones_like(cosine)
        elif self.directivity_mode == "piston":
            argument = self.k_rad_m * self.piston_radius_m * sin_theta
            value = np.ones_like(argument)
            mask = np.abs(argument) >= 1.0e-10
            value[mask] = 2.0 * j1(argument[mask]) / argument[mask]
        elif self.directivity_mode == "murata_table":
            angle_deg = np.degrees(np.arccos(front_cosine))
            table_angles = np.arange(0.0, 91.0, 2.0)
            value = np.interp(angle_deg, table_angles, _MURATA_DIRECTIVITY_PERCENT) / 100.0
        else:
            raise ValueError(f"unknown directivity mode: {self.directivity_mode}")
        if self.front_only:
            value = np.where(cosine >= 0.0, value, 0.0)
        return value

    def pressure(self, points_m: Array) -> Array:
        points = np.asarray(points_m, dtype=float)
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        delta = flat[:, None, :] - self.positions_m[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        if np.any(distance == 0.0):
            raise ValueError("pressure is singular at an ideal point-source position")
        directions = delta / distance[:, :, None]
        cos_theta = np.einsum("mni,ni->mn", directions, self.normals)
        directivity = self._directivity(cos_theta)
        propagator = directivity * np.exp(1j * self.k_rad_m * distance) / distance
        pressure = self.source_strength_pa_m * (propagator @ self.weights)
        return pressure.reshape(original_shape)

    def provenance(self) -> dict:
        """JSON-compatible field and calibration metadata."""

        return {
            "field_type": type(self).__name__,
            "frequency_hz": self.frequency_hz,
            "sound_speed_m_s": self.sound_speed_m_s,
            "wavenumber_rad_m": self.k_rad_m,
            "number_of_elements": int(len(self.positions_m)),
            "source_strength_pa_m_peak": self.source_strength_pa_m,
            "source_calibration": self.source_calibration_label,
            "directivity_model": self.directivity_mode,
            "piston_radius_m": self.piston_radius_m,
            "front_only": self.front_only,
            "pressure_phasor": dict(PEAK_PHASOR_PROVENANCE),
        }


class PlaneWavePressureField:
    """Global plane wave used for numerical regression and calibration."""

    def __init__(
        self,
        pressure_pa_peak: float,
        k_rad_m: float,
        direction: Sequence[float] = (0.0, 0.0, 1.0),
        phase_rad: float = 0.0,
    ) -> None:
        direction_array = np.asarray(direction, dtype=float)
        self.direction = direction_array / np.linalg.norm(direction_array)
        self.pressure_pa_peak = float(pressure_pa_peak)
        self.k_rad_m = float(k_rad_m)
        self.phase_rad = float(phase_rad)

    def pressure(self, points_m: Array) -> Array:
        points = np.asarray(points_m, dtype=float)
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        values = self.pressure_pa_peak * np.exp(
            1j * (self.k_rad_m * (flat @ self.direction) + self.phase_rad)
        )
        return values.reshape(original_shape)

    def provenance(self) -> dict:
        return {
            "field_type": type(self).__name__,
            "pressure_pa_peak": self.pressure_pa_peak,
            "wavenumber_rad_m": self.k_rad_m,
            "direction": self.direction.tolist(),
            "phase_rad": self.phase_rad,
            "pressure_phasor": dict(PEAK_PHASOR_PROVENANCE),
        }


def _medium_provenance(medium: Medium) -> dict:
    return {
        "density_kg_m3": float(medium.density_kg_m3),
        "sound_speed_m_s": float(medium.sound_speed_m_s),
        "gravity_m_s2": float(medium.gravity_m_s2),
        "dynamic_viscosity_pa_s": float(medium.dynamic_viscosity_pa_s),
    }


def _sphere_provenance(sphere: ElasticSphere) -> dict:
    return {
        "particle_model": "homogeneous isotropic elastic solid sphere",
        "material_label": str(sphere.material_label),
        "parameter_provenance": str(sphere.parameter_provenance),
        "radius_m": float(sphere.radius_m),
        "density_kg_m3": float(sphere.density_kg_m3),
        "longitudinal_sound_speed_m_s": float(
            sphere.longitudinal_sound_speed_m_s
        ),
        "shear_sound_speed_m_s": float(sphere.shear_sound_speed_m_s),
        "lame_first_parameter_pa": float(sphere.lame_first_parameter_pa),
        "shear_modulus_pa": float(sphere.shear_modulus_pa),
        "bulk_modulus_pa": float(sphere.bulk_modulus_pa),
        "young_modulus_pa": float(sphere.young_modulus_pa),
        "poisson_ratio": float(sphere.poisson_ratio),
        "compressibility_pa_inverse": float(sphere.compressibility_pa_inverse),
        "volume_m3": float(sphere.volume_m3),
    }


def _field_provenance(field: GlobalPressureField) -> dict:
    provider = getattr(field, "provenance", None)
    if callable(provider):
        return dict(provider())
    return {
        "field_type": type(field).__name__,
        "pressure_phasor": dict(PEAK_PHASOR_PROVENANCE),
    }


def standard_gorkov_contrast_factors(
    medium: Medium, sphere: ElasticSphere
) -> tuple[float, float]:
    """Return the standard monopole ``f1`` and dipole ``f2`` factors.

    ``f1 = 1 - kappa_p/kappa_0`` and
    ``f2 = 2 (rho_p-rho_0)/(2 rho_p+rho_0)``.
    """

    # The static bulk compressibility of an isotropic elastic solid is
    # 1 / (lambda + 2 mu / 3), not 1 / (rho c_L^2).  The latter would silently
    # discard the material's shear response and reproduce the removed scalar
    # interior approximation.
    compressibility_ratio = (
        medium.density_kg_m3 * medium.sound_speed_m_s**2 / sphere.bulk_modulus_pa
    )
    f1 = 1.0 - compressibility_ratio
    f2 = 2.0 * (sphere.density_kg_m3 - medium.density_kg_m3) / (
        2.0 * sphere.density_kg_m3 + medium.density_kg_m3
    )
    return float(f1), float(f2)


def standard_gorkov_coefficients(
    medium: Medium, sphere: ElasticSphere, frequency_hz: float
) -> tuple[float, float]:
    """Return ``Kp, Kv`` for ``U = Kp |p|^2 - Kv |grad p|^2``.

    The expression assumes complex *peak* phasors.  For a denser particle,
    ``Kv`` is positive, so the velocity/pressure-gradient energy enters with
    the standard negative sign.
    """

    omega = 2.0 * math.pi * float(frequency_hz)
    f1, f2 = standard_gorkov_contrast_factors(medium, sphere)
    pressure_coefficient = (
        sphere.volume_m3 * f1 / (4.0 * medium.density_kg_m3 * medium.sound_speed_m_s**2)
    )
    gradient_coefficient = (
        3.0
        * sphere.volume_m3
        * f2
        / (8.0 * omega**2 * medium.density_kg_m3)
    )
    return float(pressure_coefficient), float(gradient_coefficient)


def _effective_gravity_n(medium: Medium, sphere: ElasticSphere) -> Array:
    return np.array(
        [
            0.0,
            0.0,
            -(sphere.density_kg_m3 - medium.density_kg_m3)
            * sphere.volume_m3
            * medium.gravity_m_s2,
        ],
        dtype=float,
    )


class StandardGorkovEvaluator:
    """Standard Gor'kov static-force comparator on a global pressure field."""

    model_name = "standard Gor'kov (Rayleigh surrogate)"

    def __init__(
        self,
        frequency_hz: float,
        field: GlobalPressureField,
        medium: Medium = Medium(),
        sphere: ElasticSphere = ElasticSphere(),
        numerics: GorkovNumerics = GorkovNumerics(),
        *,
        include_effective_gravity: bool = True,
    ) -> None:
        self.frequency_hz = float(frequency_hz)
        self.field = field
        self.medium = medium
        self.sphere = sphere
        self.numerics = numerics
        self.omega_rad_s = 2.0 * math.pi * self.frequency_hz
        self.pressure_coefficient, self.gradient_coefficient = standard_gorkov_coefficients(
            medium, sphere, frequency_hz
        )
        self.include_effective_gravity = bool(include_effective_gravity)

    @property
    def effective_weight_n(self) -> float:
        return float(np.linalg.norm(_effective_gravity_n(self.medium, self.sphere)))

    @property
    def force_scale_n(self) -> float:
        return max(self.effective_weight_n, 1.0e-12)

    def pressure_gradient(self, points_m: Array) -> Array:
        points = np.asarray(points_m, dtype=float).reshape(-1, 3)
        step = self.numerics.pressure_gradient_step_a * self.sphere.radius_m
        if step <= 0.0:
            raise ValueError("pressure-gradient step must be positive")
        gradient = np.empty((len(points), 3), dtype=np.complex128)
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            gradient[:, axis] = (
                np.asarray(self.field.pressure(points + shift), dtype=np.complex128)
                - np.asarray(self.field.pressure(points - shift), dtype=np.complex128)
            ) / (2.0 * step)
        return gradient

    def potential(self, points_m: Array) -> Array:
        points = np.asarray(points_m, dtype=float)
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        pressure = np.asarray(self.field.pressure(flat), dtype=np.complex128).reshape(-1)
        gradient = self.pressure_gradient(flat)
        potential = self.pressure_coefficient * np.abs(pressure) ** 2 - self.gradient_coefficient * np.sum(
            np.abs(gradient) ** 2, axis=1
        )
        return potential.reshape(original_shape)

    def radiation_force(self, center_m: Sequence[float]) -> Array:
        center = np.asarray(center_m, dtype=float)
        step = self.numerics.potential_gradient_step_a * self.sphere.radius_m
        if step <= 0.0:
            raise ValueError("potential-gradient step must be positive")
        force = np.empty(3, dtype=float)
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            force[axis] = -float(
                (self.potential(center + shift) - self.potential(center - shift))
                / (2.0 * step)
            )
        return force

    def force(self, center_m: Sequence[float]) -> ForceEvaluation:
        radiation = self.radiation_force(center_m)
        gravity = (
            _effective_gravity_n(self.medium, self.sphere)
            if self.include_effective_gravity
            else np.zeros(3)
        )
        return ForceEvaluation(
            radiation_n=radiation,
            effective_gravity_n=gravity,
            total_n=radiation + gravity,
            metadata={
                "pressure_coefficient_j_pa_minus2": self.pressure_coefficient,
                "gradient_coefficient_j_m2_pa_minus2": self.gradient_coefficient,
                "pressure_gradient_step_a": self.numerics.pressure_gradient_step_a,
                "potential_gradient_step_a": self.numerics.potential_gradient_step_a,
            },
        )

    def provenance(self) -> dict:
        """Complete JSON-compatible parameters for a Gor'kov comparison."""

        return {
            "model_name": self.model_name,
            "model_role": "Rayleigh-particle optimization surrogate/comparator",
            "frequency_hz": self.frequency_hz,
            "angular_frequency_rad_s": self.omega_rad_s,
            "medium": _medium_provenance(self.medium),
            "sphere": _sphere_provenance(self.sphere),
            "field": _field_provenance(self.field),
            "include_effective_gravity": self.include_effective_gravity,
            "gorkov_coefficients": {
                "pressure_j_pa_minus2": self.pressure_coefficient,
                "gradient_j_m2_pa_minus2": self.gradient_coefficient,
            },
            "gorkov_numerics": asdict(self.numerics),
            "pressure_phasor": dict(PEAK_PHASOR_PROVENANCE),
        }


def viscous_elastic_dipole_contrast_factor(
    frequency_hz: float,
    medium: Medium,
    sphere: ElasticSphere,
) -> complex:
    """Return the viscous-boundary-layer dipole coefficient for a solid bead.

    This is the solid-particle result of Settnes and Bruus, Phys. Rev. E 85,
    016327 (2012), and Karlsen and Bruus, Phys. Rev. E 92, 043010 (2015),

    ``f_1 = 2 (rho_tilde - 1) (1 - G) / (2 rho_tilde + 1 - 3 G)``,
    ``G = 3/x_s (1/x_s - i)``, and
    ``x_s = (1+i) a/delta_s``.

    Only the host shear boundary layer enters this coefficient.  The
    zero-viscosity limit is evaluated explicitly to avoid division by zero.
    """

    frequency = float(frequency_hz)
    viscosity = float(medium.dynamic_viscosity_pa_s)
    if frequency <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if viscosity < 0.0:
        raise ValueError("dynamic viscosity must be non-negative")
    density_ratio = sphere.density_kg_m3 / medium.density_kg_m3
    inviscid = 2.0 * (density_ratio - 1.0) / (2.0 * density_ratio + 1.0)
    if viscosity == 0.0:
        return complex(inviscid)
    omega = 2.0 * math.pi * frequency
    delta_s = math.sqrt(2.0 * viscosity / (medium.density_kg_m3 * omega))
    x_s = (1.0 + 1.0j) * sphere.radius_m / delta_s
    g_x = 3.0 / x_s * (1.0 / x_s - 1.0j)
    return complex(
        2.0
        * (density_ratio - 1.0)
        * (1.0 - g_x)
        / (2.0 * density_ratio + 1.0 - 3.0 * g_x)
    )


class ViscousElasticRayleighEvaluator:
    """Viscosity-aware long-wave sensitivity for the elastic-solid bead.

    The evaluator applies the arbitrary-field radiation-force expression of
    Settnes--Bruus and Karlsen--Bruus.  The solid monopole coefficient uses
    the bead's elastic bulk compressibility, while the complex dipole
    coefficient includes the host viscous shear boundary layer.  It is kept
    separate from the finite-``ka`` partial-wave reference: this class assumes
    ``ka << 1`` and does not add an unvalidated viscous correction to the
    finite-``ka`` scattering coefficients.

    No thermal boundary-layer, microstreaming, wall, or particle-particle
    correction is included.  Thus this is a declared sensitivity model, not a
    thermoviscous exact-force replacement.
    """

    model_name = "viscous-boundary-layer elastic-solid Rayleigh sensitivity"

    def __init__(
        self,
        frequency_hz: float,
        field: GlobalPressureField,
        medium: Medium = Medium(),
        sphere: ElasticSphere = ElasticSphere(),
        numerics: ViscousRayleighNumerics = ViscousRayleighNumerics(),
        *,
        include_effective_gravity: bool = True,
        incident_projection_numerics: PartialWaveNumerics | None = None,
    ) -> None:
        self.frequency_hz = float(frequency_hz)
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if medium.dynamic_viscosity_pa_s < 0.0:
            raise ValueError("dynamic viscosity must be non-negative")
        if numerics.pressure_gradient_step_a <= 0.0:
            raise ValueError("pressure-gradient step must be positive")
        if numerics.velocity_gradient_step_a <= 0.0:
            raise ValueError("velocity-gradient step must be positive")
        self.field = field
        self.medium = medium
        self.sphere = sphere
        self.numerics = numerics
        self.include_effective_gravity = bool(include_effective_gravity)
        self.omega_rad_s = 2.0 * math.pi * self.frequency_hz
        self.k_rad_m = self.omega_rad_s / self.medium.sound_speed_m_s
        self.wavelength_m = 2.0 * math.pi / self.k_rad_m
        self.host_compressibility_pa_inverse = 1.0 / (
            self.medium.density_kg_m3 * self.medium.sound_speed_m_s**2
        )
        compressibility_ratio = (
            self.medium.density_kg_m3
            * self.medium.sound_speed_m_s**2
            / self.sphere.bulk_modulus_pa
        )
        self.monopole_contrast = complex(1.0 - compressibility_ratio)
        self.dipole_contrast = viscous_elastic_dipole_contrast_factor(
            self.frequency_hz, self.medium, self.sphere
        )
        self.incident_projection = None
        if numerics.regular_wave_derivatives:
            self.incident_projection = PartialWaveForceEvaluator(
                self.frequency_hz, field, medium, sphere,
                incident_projection_numerics or PartialWaveNumerics(lmax=3),
                include_effective_gravity=False,
            )
            projection = self.incident_projection
            _, angular_weights, _ = spherical_quadrature(
                projection.numerics.fit_n_mu, projection.numerics.fit_n_phi)
            weights = np.sqrt(np.tile(angular_weights,
                len(projection.numerics.fit_shell_wavelengths)))
            weighted_design = weights[:, None] * projection.design
            scales = np.linalg.norm(weighted_design, axis=0)
            projection.design_pinv = (
                np.linalg.pinv(weighted_design / scales[None, :],
                    rcond=projection.numerics.fit_rcond) * weights[None, :]
            ) / scales[:, None]

    @property
    def shear_boundary_layer_thickness_m(self) -> float:
        viscosity = float(self.medium.dynamic_viscosity_pa_s)
        if viscosity == 0.0:
            return 0.0
        return math.sqrt(
            2.0
            * viscosity
            / (self.medium.density_kg_m3 * self.omega_rad_s)
        )

    @property
    def effective_weight_n(self) -> float:
        return float(np.linalg.norm(_effective_gravity_n(self.medium, self.sphere)))

    @property
    def force_scale_n(self) -> float:
        return max(self.effective_weight_n, 1.0e-12)

    @staticmethod
    def _fourth_order_central(
        minus_two: Array,
        minus_one: Array,
        plus_one: Array,
        plus_two: Array,
        step_m: float,
    ) -> Array:
        """Fourth-order centered derivative for real or complex arrays."""

        return (
            np.asarray(minus_two)
            - 8.0 * np.asarray(minus_one)
            + 8.0 * np.asarray(plus_one)
            - np.asarray(plus_two)
        ) / (12.0 * float(step_m))

    def pressure_gradient(self, points_m: Array) -> Array:
        """Return ``grad(p)`` using fourth-order centered differences."""

        points = np.asarray(points_m, dtype=float)
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        step = self.numerics.pressure_gradient_step_a * self.sphere.radius_m
        gradient = np.empty((len(flat), 3), dtype=np.complex128)
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            minus_two = self.field.pressure(flat - 2.0 * shift)
            minus_one = self.field.pressure(flat - shift)
            plus_one = self.field.pressure(flat + shift)
            plus_two = self.field.pressure(flat + 2.0 * shift)
            gradient[:, axis] = np.asarray(
                self._fourth_order_central(
                    minus_two, minus_one, plus_one, plus_two, step
                ),
                dtype=np.complex128,
            ).reshape(-1)
        return gradient.reshape(original_shape + (3,))

    def incident_velocity(self, points_m: Array) -> Array:
        """Return ``v = grad(p)/(i omega rho_0)`` for peak phasors."""

        return self.pressure_gradient(points_m) / (
            1.0j * self.omega_rad_s * self.medium.density_kg_m3
        )

    def velocity_gradient(self, center_m: Sequence[float]) -> Array:
        """Return ``d v_i / d x_j`` using fourth-order centered differences."""

        center = np.asarray(center_m, dtype=float).reshape(3)
        step = self.numerics.velocity_gradient_step_a * self.sphere.radius_m
        jacobian = np.empty((3, 3), dtype=np.complex128)
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            minus_two = self.incident_velocity(center - 2.0 * shift)
            minus_one = self.incident_velocity(center - shift)
            plus_one = self.incident_velocity(center + shift)
            plus_two = self.incident_velocity(center + 2.0 * shift)
            jacobian[:, axis] = self._fourth_order_central(
                minus_two, minus_one, plus_one, plus_two, step
            )
        return jacobian

    def radiation_force(self, center_m: Sequence[float]) -> Array:
        """Evaluate the arbitrary-field long-wave radiation force."""

        center = np.asarray(center_m, dtype=float).reshape(3)
        if self.incident_projection is not None:
            pressure, pressure_gradient, hessian, _ = regular_wave_pressure_derivatives(
                self.incident_projection, center)
            denominator = 1j * self.omega_rad_s * self.medium.density_kg_m3
            velocity = pressure_gradient / denominator
            velocity_gradient = hessian / denominator
        else:
            pressure = complex(np.asarray(self.field.pressure(center)).reshape(()))
            pressure_gradient = self.pressure_gradient(center).reshape(3)
            velocity = self.incident_velocity(center).reshape(3)
            velocity_gradient = self.velocity_gradient(center)
        pressure_contraction = (
            np.conj(self.monopole_contrast)
            * np.conj(pressure)
            * pressure_gradient
        )
        velocity_contraction = np.conj(self.dipole_contrast) * np.einsum(
            "i,ij->j", np.conj(velocity), velocity_gradient
        )
        bracket = (
            2.0
            * self.host_compressibility_pa_inverse
            / 3.0
            * np.real(pressure_contraction)
            - self.medium.density_kg_m3 * np.real(velocity_contraction)
        )
        return np.asarray(-math.pi * self.sphere.radius_m**3 * bracket, dtype=float)

    def force(self, center_m: Sequence[float]) -> ForceEvaluation:
        radiation = self.radiation_force(center_m)
        gravity = (
            _effective_gravity_n(self.medium, self.sphere)
            if self.include_effective_gravity
            else np.zeros(3)
        )
        return ForceEvaluation(
            radiation_n=radiation,
            effective_gravity_n=gravity,
            total_n=radiation + gravity,
            metadata={
                "monopole_contrast_real": float(np.real(self.monopole_contrast)),
                "monopole_contrast_imag": float(np.imag(self.monopole_contrast)),
                "dipole_contrast_real": float(np.real(self.dipole_contrast)),
                "dipole_contrast_imag": float(np.imag(self.dipole_contrast)),
                "shear_boundary_layer_thickness_m": (
                    self.shear_boundary_layer_thickness_m
                ),
                "shear_boundary_layer_to_radius": (
                    self.shear_boundary_layer_thickness_m / self.sphere.radius_m
                ),
                "pressure_gradient_step_a": (
                    self.numerics.pressure_gradient_step_a
                ),
                "velocity_gradient_step_a": (
                    self.numerics.velocity_gradient_step_a
                ),
            },
        )

    def provenance(self) -> dict:
        """Complete JSON-compatible scope and parameters for this sensitivity."""

        delta_s = self.shear_boundary_layer_thickness_m
        return {
            "model_name": self.model_name,
            "model_role": (
                "secondary viscosity sensitivity; not the finite-ka reference"
            ),
            "model_scope": (
                "long-wave elastic solid sphere with host viscous shear boundary "
                "layer in the dipole response; elastic bulk-compressibility "
                "monopole without thermal correction"
            ),
            "validity": "ka << 1 with particle radius and shear layer small versus wavelength",
            "excluded_physics": [
                "thermal boundary-layer correction",
                "microstreaming",
                "nearby-wall correction",
                "particle-particle multiple scattering",
                "finite-ka viscous hybridization",
            ],
            "references": [
                (
                    "Settnes and Bruus, Phys. Rev. E 85, 016327 (2012), "
                    "doi:10.1103/PhysRevE.85.016327"
                ),
                (
                    "Karlsen and Bruus, Phys. Rev. E 92, 043010 (2015), "
                    "doi:10.1103/PhysRevE.92.043010"
                ),
            ],
            "frequency_hz": self.frequency_hz,
            "angular_frequency_rad_s": self.omega_rad_s,
            "wavenumber_rad_m": self.k_rad_m,
            "wavelength_m": self.wavelength_m,
            "ka": self.k_rad_m * self.sphere.radius_m,
            "medium": _medium_provenance(self.medium),
            "sphere": _sphere_provenance(self.sphere),
            "field": _field_provenance(self.field),
            "include_effective_gravity": self.include_effective_gravity,
            "viscous_response": {
                "shear_boundary_layer_thickness_m": delta_s,
                "shear_boundary_layer_to_radius": delta_s / self.sphere.radius_m,
                "monopole_contrast_real": float(np.real(self.monopole_contrast)),
                "monopole_contrast_imag": float(np.imag(self.monopole_contrast)),
                "dipole_contrast_real": float(np.real(self.dipole_contrast)),
                "dipole_contrast_imag": float(np.imag(self.dipole_contrast)),
            },
            "viscous_rayleigh_numerics": asdict(self.numerics),
            "incident_derivatives": (
                {"operator":"analytic center derivatives of least-squares regular spherical-wave projection v1",
                 "projection_numerics":asdict(self.incident_projection.numerics)}
                if self.incident_projection is not None else
                {"operator":"fourth-order finite differences of global field"}
            ),
            "pressure_phasor": dict(PEAK_PHASOR_PROVENANCE),
        }


def regular_wave_pressure_derivatives(projector, center_m):
    """Analytic p, gradient and Hessian at a regular-wave expansion center.

    Only l=0,1,2 contribute at the center. The fit uses the unmodified global
    incident field and the same shells as the elastic reference. This avoids
    taking second derivatives across knots in measured directivity tables.
    It is a declared Helmholtz projection, not a new transducer directivity.
    """
    coefficients, residual = projector.fit_incident_coefficients(center_m)
    c = dict(zip(projector.lm, coefficients, strict=True))
    if projector.numerics.lmax < 2:
        raise ValueError("Pressure Hessian requires lmax >= 2")
    k = projector.k_rad_m
    pressure = c[0, 0] / math.sqrt(4 * math.pi)
    gradient = k / 3 * np.array([
        math.sqrt(3/(8*math.pi)) * (c[1,-1] - c[1,1]),
        -1j * math.sqrt(3/(8*math.pi)) * (c[1,-1] + c[1,1]),
        math.sqrt(3/(4*math.pi)) * c[1,0],
    ])
    hessian = np.eye(3, dtype=complex) * (-k*k*pressure/3)
    factor = k*k/15
    hessian += factor * c[2,0] * math.sqrt(5/(16*math.pi)) * np.diag([-2., -2., 4.])
    for m, sign in ((-1, 1), (1, -1)):
        a = factor * sign * math.sqrt(15/(8*math.pi)) * c[2,m]
        hessian[0,2] += a
        hessian[2,0] += a
        hessian[1,2] += a * 1j * m
        hessian[2,1] += a * 1j * m
    for m in (-2, 2):
        a = factor * math.sqrt(15/(32*math.pi)) * c[2,m]
        hessian[0,0] += 2*a
        hessian[1,1] -= 2*a
        hessian[0,1] += 1j*m*a
        hessian[1,0] += 1j*m*a
    return pressure, gradient, hessian, residual


def _hankel1(order: int, z: Array, derivative: bool = False) -> Array:
    return spherical_jn(order, z, derivative=derivative) + 1j * spherical_yn(
        order, z, derivative=derivative
    )


def _spherical_jn_second_derivative(order: int, z: float) -> float:
    """Second derivative from the spherical-Bessel differential equation."""

    if z == 0.0:
        raise ValueError("spherical-Bessel second derivative requires nonzero argument")
    value = spherical_jn(order, z)
    derivative = spherical_jn(order, z, derivative=True)
    angular = order * (order + 1)
    return float(-2.0 * derivative / z - (1.0 - angular / z**2) * value)


def _column_scaled_solve(matrix: Array, rhs: Array) -> Array:
    """Solve a small boundary matrix after deterministic column scaling."""

    scales = np.linalg.norm(matrix, axis=0)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise np.linalg.LinAlgError("degenerate elastic boundary-condition matrix")
    normalized = matrix / scales[None, :]
    scaled_solution = np.linalg.solve(normalized, rhs)
    return scaled_solution / scales


def elastic_sphere_scattering_coefficients(
    frequency_hz: float,
    medium: Medium,
    sphere: ElasticSphere,
    lmax: int,
) -> Array:
    """Partial-wave coefficients for a homogeneous isotropic elastic sphere.

    For each spherical-harmonic order, the exterior pressure is written as

    ``p_l = j_l(k_0 r) + a_l h_l^(1)(k_0 r)``.

    The solid displacement is the sum of a longitudinal scalar-potential mode
    and a spheroidal shear mode.  The three boundary conditions at ``r=a`` are
    continuity of normal velocity, continuity of normal traction, and zero
    tangential traction in the host fluid.  The monopole has no shear degree of
    freedom and is therefore solved as a 2-by-2 system.  The formulation is the
    standard elastic-sphere construction of Faran (JASA 23, 405--418, 1951,
    doi:10.1121/1.1906780), expressed for the peak-pressure and ``exp(-i wt)``
    convention declared at module level.

    With this convention a passive lossless sphere obeys
    ``abs(1 + 2 a_l) = 1`` and an immovable rigid sphere tends to
    ``a_l = -j_l'(ka)/h_l^(1)'(ka)``.
    """

    omega = 2.0 * math.pi * float(frequency_hz)
    k0 = omega / medium.sound_speed_m_s
    k_longitudinal = omega / sphere.longitudinal_sound_speed_m_s
    k_shear = omega / sphere.shear_sound_speed_m_s
    x0 = k0 * sphere.radius_m
    x_longitudinal = k_longitudinal * sphere.radius_m
    x_shear = k_shear * sphere.radius_m
    lame = sphere.lame_first_parameter_pa
    shear = sphere.shear_modulus_pa
    inertia_scale = medium.density_kg_m3 * omega**2
    radius_squared = sphere.radius_m**2
    coefficients = np.empty(int(lmax) + 1, dtype=np.complex128)

    for ell in range(int(lmax) + 1):
        angular = ell * (ell + 1)
        j_host = spherical_jn(ell, x0)
        dj_host = spherical_jn(ell, x0, derivative=True)
        h_host = _hankel1(ell, x0)
        dh_host = _hankel1(ell, x0, derivative=True)

        j_longitudinal = spherical_jn(ell, x_longitudinal)
        dj_longitudinal = spherical_jn(
            ell, x_longitudinal, derivative=True
        )
        ddj_longitudinal = _spherical_jn_second_derivative(
            ell, x_longitudinal
        )

        # Unknown longitudinal and shear potential amplitudes are scaled by
        # rho_0 omega^2.  Consequently all matrix rows below are in pressure
        # units and no arbitrary dimensional balancing enters the solve.
        radial_longitudinal = x_longitudinal * dj_longitudinal
        normal_longitudinal = (
            k_longitudinal**2
            * (-lame * j_longitudinal + 2.0 * shear * ddj_longitudinal)
            / inertia_scale
        )
        tangential_longitudinal = (
            2.0
            * shear
            * (x_longitudinal * dj_longitudinal - j_longitudinal)
            / (inertia_scale * radius_squared)
        )

        if ell == 0:
            matrix = np.array(
                [
                    [x0 * dh_host, -radial_longitudinal],
                    [h_host, normal_longitudinal],
                ],
                dtype=np.complex128,
            )
            rhs = np.array([-x0 * dj_host, -j_host], dtype=np.complex128)
            coefficients[ell] = _column_scaled_solve(matrix, rhs)[0]
            continue

        j_shear = spherical_jn(ell, x_shear)
        dj_shear = spherical_jn(ell, x_shear, derivative=True)
        radial_shear = angular * j_shear
        normal_shear = (
            2.0
            * shear
            * angular
            * (x_shear * dj_shear - j_shear)
            / (inertia_scale * radius_squared)
        )
        tangential_shear = (
            shear
            * (
                (2.0 * (angular - 1.0) - x_shear**2) * j_shear
                - 2.0 * x_shear * dj_shear
            )
            / (inertia_scale * radius_squared)
        )
        matrix = np.array(
            [
                [
                    x0 * dh_host,
                    -radial_longitudinal,
                    -radial_shear,
                ],
                [h_host, normal_longitudinal, normal_shear],
                [0.0, tangential_longitudinal, tangential_shear],
            ],
            dtype=np.complex128,
        )
        rhs = np.array(
            [-x0 * dj_host, -j_host, 0.0], dtype=np.complex128
        )
        coefficients[ell] = _column_scaled_solve(matrix, rhs)[0]
    return coefficients


def spherical_quadrature(n_mu: int, n_phi: int) -> tuple[Array, Array, Array]:
    """Gauss-Legendre in polar cosine and uniform trapezoidal azimuth."""

    mu, w_mu = np.polynomial.legendre.leggauss(int(n_mu))
    phi = np.arange(int(n_phi), dtype=float) * 2.0 * math.pi / int(n_phi)
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing="ij")
    theta_grid = np.arccos(mu_grid)
    radial = np.sqrt(np.maximum(1.0 - mu_grid**2, 0.0))
    directions = np.stack(
        [radial * np.cos(phi_grid), radial * np.sin(phi_grid), mu_grid], axis=-1
    ).reshape(-1, 3)
    weights = (w_mu[:, None] * np.full((1, int(n_phi)), 2.0 * math.pi / n_phi)).ravel()
    angles = np.column_stack([theta_grid.ravel(), phi_grid.ravel()])
    return directions, weights, angles


class PartialWaveForceEvaluator:
    """Finite-``ka`` static radiation force for the elastic-solid bead model."""

    model_name = "finite-ka elastic-solid-sphere partial-wave reference"

    def __init__(
        self,
        frequency_hz: float,
        field: GlobalPressureField,
        medium: Medium = Medium(),
        sphere: ElasticSphere = ElasticSphere(),
        numerics: PartialWaveNumerics = PartialWaveNumerics(),
        *,
        include_effective_gravity: bool = True,
    ) -> None:
        self.frequency_hz = float(frequency_hz)
        self.field = field
        self.medium = medium
        self.sphere = sphere
        self.numerics = numerics
        self.include_effective_gravity = bool(include_effective_gravity)
        self.omega_rad_s = 2.0 * math.pi * self.frequency_hz
        self.k_rad_m = self.omega_rad_s / self.medium.sound_speed_m_s
        self.wavelength_m = 2.0 * math.pi / self.k_rad_m
        if numerics.control_radius_a <= 1.0:
            raise ValueError("control_radius_a must be greater than the sphere radius")
        self.scatter = elastic_sphere_scattering_coefficients(
            self.frequency_hz, medium, sphere, numerics.lmax
        )
        self.lm = [
            (ell, m)
            for ell in range(numerics.lmax + 1)
            for m in range(-ell, ell + 1)
        ]

        fit_directions, _, fit_angles = spherical_quadrature(
            numerics.fit_n_mu, numerics.fit_n_phi
        )
        fit_offsets: list[Array] = []
        design_blocks: list[Array] = []
        for shell_wavelengths in numerics.fit_shell_wavelengths:
            shell_radius = float(shell_wavelengths) * self.wavelength_m
            fit_offsets.append(shell_radius * fit_directions)
            design_blocks.append(
                np.column_stack(
                    [
                        spherical_jn(ell, self.k_rad_m * shell_radius)
                        * sph_harm_y(ell, m, fit_angles[:, 0], fit_angles[:, 1])
                        for ell, m in self.lm
                    ]
                )
            )
        self.fit_offsets_m = np.vstack(fit_offsets)
        self.design = np.vstack(design_blocks)
        column_scales = np.linalg.norm(self.design, axis=0)
        if np.any(column_scales == 0.0):
            raise ValueError("degenerate spherical-wave projection design")
        normalized_design = self.design / column_scales[None, :]
        self.design_pinv = (
            np.linalg.pinv(normalized_design, rcond=numerics.fit_rcond)
            / column_scales[:, None]
        )
        self.projection_condition = float(np.linalg.cond(normalized_design))

        surface_directions, surface_weights, _ = spherical_quadrature(
            numerics.surface_n_mu, numerics.surface_n_phi
        )
        self.surface_directions = surface_directions
        self.surface_weights = surface_weights
        self.control_radius_m = numerics.control_radius_a * sphere.radius_m
        self.surface_offsets_m = self.control_radius_m * surface_directions
        self.velocity_step_m = numerics.velocity_step_a * sphere.radius_m

    @property
    def effective_weight_n(self) -> float:
        return float(np.linalg.norm(_effective_gravity_n(self.medium, self.sphere)))

    @property
    def force_scale_n(self) -> float:
        return max(self.effective_weight_n, 1.0e-12)

    def fit_incident_coefficients(self, center_m: Sequence[float]) -> tuple[Array, float]:
        center = np.asarray(center_m, dtype=float)
        samples = np.asarray(
            self.field.pressure(center[None, :] + self.fit_offsets_m), dtype=np.complex128
        ).reshape(-1)
        coefficients = self.design_pinv @ samples
        residual = np.linalg.norm(self.design @ coefficients - samples) / max(
            np.linalg.norm(samples), 1.0e-300
        )
        return coefficients, float(residual)

    def _scattered_pressure(self, offsets_m: Array, coefficients: Array) -> Array:
        offsets = np.asarray(offsets_m, dtype=float).reshape(-1, 3)
        radius = np.linalg.norm(offsets, axis=1)
        if np.any(radius <= self.sphere.radius_m):
            raise ValueError("scattered pressure requested on or inside the sphere")
        theta = np.arccos(np.clip(offsets[:, 2] / radius, -1.0, 1.0))
        phi = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * math.pi)
        kr = self.k_rad_m * radius
        result = np.zeros(len(offsets), dtype=np.complex128)
        for index, (ell, m) in enumerate(self.lm):
            result += (
                coefficients[index]
                * self.scatter[ell]
                * _hankel1(ell, kr)
                * sph_harm_y(ell, m, theta, phi)
            )
        return result

    def radiation_force(self, center_m: Sequence[float]) -> tuple[Array, dict]:
        center = np.asarray(center_m, dtype=float)
        coefficients, projection_residual = self.fit_incident_coefficients(center)
        surface_points = center[None, :] + self.surface_offsets_m
        incident_pressure = np.asarray(
            self.field.pressure(surface_points), dtype=np.complex128
        ).reshape(-1)
        scattered_pressure = self._scattered_pressure(self.surface_offsets_m, coefficients)
        total_pressure = incident_pressure + scattered_pressure

        incident_velocity = np.empty((len(surface_points), 3), dtype=np.complex128)
        total_velocity = np.empty_like(incident_velocity)
        step = self.velocity_step_m
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            incident_plus = np.asarray(
                self.field.pressure(surface_points + shift), dtype=np.complex128
            ).reshape(-1)
            incident_minus = np.asarray(
                self.field.pressure(surface_points - shift), dtype=np.complex128
            ).reshape(-1)
            scattered_plus = self._scattered_pressure(
                self.surface_offsets_m + shift, coefficients
            )
            scattered_minus = self._scattered_pressure(
                self.surface_offsets_m - shift, coefficients
            )
            denominator = 2.0 * step * 1j * self.omega_rad_s * self.medium.density_kg_m3
            incident_velocity[:, axis] = (incident_plus - incident_minus) / denominator
            total_velocity[:, axis] = (
                incident_plus + scattered_plus - incident_minus - scattered_minus
            ) / denominator

        def traction(pressure: Array, velocity: Array) -> Array:
            speed_squared = np.sum(np.abs(velocity) ** 2, axis=1)
            isotropic = (
                np.abs(pressure) ** 2
                / (4.0 * self.medium.density_kg_m3 * self.medium.sound_speed_m_s**2)
                - self.medium.density_kg_m3 * speed_squared / 4.0
            )
            velocity_dot_normal_conjugate = np.sum(
                np.conj(velocity) * self.surface_directions, axis=1
            )
            return (
                isotropic[:, None] * self.surface_directions
                + self.medium.density_kg_m3
                * np.real(velocity * velocity_dot_normal_conjugate[:, None])
                / 2.0
            )

        delta_traction = traction(total_pressure, total_velocity) - traction(
            incident_pressure, incident_velocity
        )
        surface_area_weights = self.control_radius_m**2 * self.surface_weights
        force = -np.sum(delta_traction * surface_area_weights[:, None], axis=0)
        return force, {
            "projection_residual": projection_residual,
            "projection_condition": self.projection_condition,
            "lmax": self.numerics.lmax,
            "control_radius_a": self.numerics.control_radius_a,
            "surface_points": len(self.surface_directions),
            "velocity_step_a": self.numerics.velocity_step_a,
        }

    def force(self, center_m: Sequence[float]) -> ForceEvaluation:
        radiation, metadata = self.radiation_force(center_m)
        gravity = (
            _effective_gravity_n(self.medium, self.sphere)
            if self.include_effective_gravity
            else np.zeros(3)
        )
        return ForceEvaluation(
            radiation_n=radiation,
            effective_gravity_n=gravity,
            total_n=radiation + gravity,
            metadata=metadata,
        )

    def provenance(self) -> dict:
        """Complete JSON-compatible parameters for the finite-``ka`` reference."""

        return {
            "model_name": self.model_name,
            "model_role": "independent finite-ka elastic-bead force reference",
            "model_scope": (
                "homogeneous isotropic elastic solid sphere with internal longitudinal "
                "and shear modes; inviscid host momentum-flux integration; no "
                "thermoviscous, streaming, wall, or multiple-scattering correction"
            ),
            "elastic_scattering_reference": (
                "Faran, JASA 23, 405-418 (1951), doi:10.1121/1.1906780"
            ),
            "elastic_boundary_conditions": (
                "normal velocity and normal traction continuous; tangential traction zero"
            ),
            "frequency_hz": self.frequency_hz,
            "angular_frequency_rad_s": self.omega_rad_s,
            "wavenumber_rad_m": self.k_rad_m,
            "wavelength_m": self.wavelength_m,
            "ka": self.k_rad_m * self.sphere.radius_m,
            "medium": _medium_provenance(self.medium),
            "sphere": _sphere_provenance(self.sphere),
            "field": _field_provenance(self.field),
            "include_effective_gravity": self.include_effective_gravity,
            "partial_wave_numerics": asdict(self.numerics),
            "pressure_phasor": dict(PEAK_PHASOR_PROVENANCE),
        }


class StaticForceEvaluator(Protocol):
    model_name: str
    sphere: ElasticSphere
    field: GlobalPressureField

    @property
    def force_scale_n(self) -> float: ...

    def force(self, center_m: Sequence[float]) -> ForceEvaluation: ...


def common_cartesian_root_starts(
    distances_a: Sequence[float] = (0.5, 1.0, 1.5),
) -> Array:
    """The fixed 19-start protocol: target plus Cartesian offsets in radii.

    The default is the target together with ``+/-0.5a``, ``+/-1.0a`` and
    ``+/-1.5a`` along each Cartesian axis.  The returned order is deterministic
    and the same array is passed to every compared force model.
    """

    starts = [np.zeros(3)]
    for distance in distances_a:
        value = float(distance)
        if value <= 0.0:
            raise ValueError("Cartesian start distances must be positive")
        for axis in range(3):
            direction = np.zeros(3)
            direction[axis] = value
            starts.extend((direction.copy(), -direction.copy()))
    return np.asarray(starts, dtype=float)


def find_equilibrium(
    evaluator: StaticForceEvaluator,
    target_m: Sequence[float],
    *,
    search_half_width_a: float = 4.0,
    initial_offsets_a: Array | None = None,
    max_nfev: int = 100,
    numerical_root_tolerance: float = 1.0e-3,
) -> EquilibriumResult:
    """Find the target-nearest numerical force root from common starts.

    ``numerical_root_tolerance`` is a solver/root-identification tolerance on
    ``||F|| / force_scale``.  It is not a physical pass criterion.
    """

    target = np.asarray(target_m, dtype=float).reshape(3)
    radius = float(evaluator.sphere.radius_m)
    half_width = float(search_half_width_a)
    if half_width <= 0.0:
        raise ValueError("search_half_width_a must be positive")
    starts = (
        common_cartesian_root_starts()
        if initial_offsets_a is None
        else np.asarray(initial_offsets_a, dtype=float).reshape(-1, 3)
    )
    if np.any(np.abs(starts) >= half_width):
        raise ValueError("all root starts must lie strictly inside the search box")
    force_scale = max(float(evaluator.force_scale_n), 1.0e-30)

    def scaled_residual(offset_a: Array) -> Array:
        return evaluator.force(target + radius * offset_a).total_n / force_scale

    candidates: list[EquilibriumCandidate] = []
    for start in starts:
        optimization = least_squares(
            scaled_residual,
            start,
            bounds=(-half_width, half_width),
            max_nfev=int(max_nfev),
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
        )
        point = target + radius * optimization.x
        residual_force = evaluator.force(point).total_n
        scaled_norm = float(np.linalg.norm(residual_force) / force_scale)
        on_boundary = bool(np.any(np.abs(optimization.x) >= half_width * (1.0 - 1.0e-7)))
        candidates.append(
            EquilibriumCandidate(
                initial_offset_a=np.asarray(start, dtype=float),
                solution_offset_a=np.asarray(optimization.x, dtype=float),
                equilibrium_m=point,
                residual_force_n=np.asarray(residual_force, dtype=float),
                residual_scaled_norm=scaled_norm,
                optimizer_success=bool(optimization.success),
                on_search_boundary=on_boundary,
                nfev=int(optimization.nfev),
                message=str(optimization.message),
            )
        )

    numerical_roots = [
        candidate
        for candidate in candidates
        if candidate.optimizer_success
        and not candidate.on_search_boundary
        and candidate.residual_scaled_norm <= numerical_root_tolerance
    ]
    if numerical_roots:
        selected = min(
            numerical_roots,
            key=lambda candidate: (
                np.linalg.norm(candidate.solution_offset_a),
                candidate.residual_scaled_norm,
            ),
        )
        numerical_root_found = True
    else:
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate.on_search_boundary,
                not candidate.optimizer_success,
                candidate.residual_scaled_norm,
                np.linalg.norm(candidate.solution_offset_a),
            ),
        )
        numerical_root_found = False

    displacement = selected.equilibrium_m - target
    return EquilibriumResult(
        target_m=target,
        equilibrium_m=selected.equilibrium_m,
        displacement_m=displacement,
        displacement_norm_m=float(np.linalg.norm(displacement)),
        displacement_norm_a=float(np.linalg.norm(displacement) / radius),
        residual_force_n=selected.residual_force_n,
        residual_scaled_norm=selected.residual_scaled_norm,
        numerical_root_found=numerical_root_found,
        selected_start_offset_a=selected.initial_offset_a,
        on_search_boundary=selected.on_search_boundary,
        nfev=selected.nfev,
        candidates=tuple(candidates),
    )


def force_jacobian(
    evaluator: StaticForceEvaluator,
    center_m: Sequence[float],
    step_m: float,
) -> Array:
    """Central-difference full static-force Jacobian ``dF_i/dx_j`` [N/m]."""

    center = np.asarray(center_m, dtype=float).reshape(3)
    step = float(step_m)
    if step <= 0.0:
        raise ValueError("Jacobian step must be positive")
    jacobian = np.empty((3, 3), dtype=float)
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = step
        plus = evaluator.force(center + shift).total_n
        minus = evaluator.force(center - shift).total_n
        jacobian[:, axis] = (plus - minus) / (2.0 * step)
    return jacobian


def validate_static_trap(
    evaluator: StaticForceEvaluator,
    target_m: Sequence[float],
    *,
    search_half_width_a: float = 4.0,
    initial_offsets_a: Array | None = None,
    max_nfev: int = 100,
    numerical_root_tolerance: float = 1.0e-3,
    jacobian_step_a: float = 0.10,
) -> StaticValidationResult:
    """Return raw target/equilibrium forces and equilibrium-centred stiffness."""

    equilibrium = find_equilibrium(
        evaluator,
        target_m,
        search_half_width_a=search_half_width_a,
        initial_offsets_a=initial_offsets_a,
        max_nfev=max_nfev,
        numerical_root_tolerance=numerical_root_tolerance,
    )
    target_evaluation = evaluator.force(equilibrium.target_m)
    equilibrium_evaluation = evaluator.force(equilibrium.equilibrium_m)
    target_force = target_evaluation.total_n
    equilibrium_force = equilibrium_evaluation.total_n
    jacobian_step_m = float(jacobian_step_a) * evaluator.sphere.radius_m
    jacobian = force_jacobian(evaluator, equilibrium.equilibrium_m, jacobian_step_m)
    symmetric_stiffness = -(jacobian + jacobian.T) / 2.0
    antisymmetric_jacobian = (jacobian - jacobian.T) / 2.0
    provenance_provider = getattr(evaluator, "provenance", None)
    provenance = dict(provenance_provider()) if callable(provenance_provider) else {}
    provenance.update(
        {
            "equilibrium_centered": True,
            "equilibrium_root_protocol": {
                "search_half_width_a": float(search_half_width_a),
                "number_of_starts": int(len(equilibrium.candidates)),
                "numerical_root_tolerance_scaled_force": float(
                    numerical_root_tolerance
                ),
                "max_nfev_per_start": int(max_nfev),
                "selection": "target-nearest converged numerical root",
            },
            "jacobian_step_a": float(jacobian_step_a),
            "jacobian_step_m": jacobian_step_m,
            # Keep the numerical projection diagnostics that are otherwise
            # lost when ForceEvaluation is reduced to its three force vectors.
            # These fields make every plotted equilibrium auditable without
            # changing the root-selection rule.
            "target_force_evaluation": dict(target_evaluation.metadata),
            "equilibrium_force_evaluation": dict(equilibrium_evaluation.metadata),
        }
    )
    return StaticValidationResult(
        model_name=str(evaluator.model_name),
        equilibrium=equilibrium,
        target_force_n=np.asarray(target_force, dtype=float),
        equilibrium_force_n=np.asarray(equilibrium_force, dtype=float),
        force_jacobian_n_m=jacobian,
        symmetric_stiffness_n_m=symmetric_stiffness,
        symmetric_stiffness_eigenvalues_n_m=np.linalg.eigvalsh(symmetric_stiffness),
        antisymmetric_force_jacobian_n_m=antisymmetric_jacobian,
        jacobian_step_m=jacobian_step_m,
        metadata=provenance,
    )


def compare_gorkov_and_finite_ka(
    finite_ka_evaluator: PartialWaveForceEvaluator,
    target_m: Sequence[float],
    *,
    gorkov_numerics: GorkovNumerics = GorkovNumerics(),
    search_half_width_a: float = 4.0,
    initial_offsets_a: Array | None = None,
    max_nfev: int = 100,
    numerical_root_tolerance: float = 1.0e-3,
    jacobian_step_a: float = 0.10,
) -> StaticModelComparison:
    """Evaluate both models on the same global field and identical root starts."""

    starts = common_cartesian_root_starts() if initial_offsets_a is None else np.asarray(initial_offsets_a)
    gorkov_evaluator = StandardGorkovEvaluator(
        finite_ka_evaluator.frequency_hz,
        finite_ka_evaluator.field,
        finite_ka_evaluator.medium,
        finite_ka_evaluator.sphere,
        gorkov_numerics,
        include_effective_gravity=finite_ka_evaluator.include_effective_gravity,
    )
    shared_kwargs = dict(
        search_half_width_a=search_half_width_a,
        initial_offsets_a=starts,
        max_nfev=max_nfev,
        numerical_root_tolerance=numerical_root_tolerance,
        jacobian_step_a=jacobian_step_a,
    )
    finite_result = validate_static_trap(finite_ka_evaluator, target_m, **shared_kwargs)
    gorkov_result = validate_static_trap(gorkov_evaluator, target_m, **shared_kwargs)
    equilibrium_difference = (
        finite_result.equilibrium.equilibrium_m - gorkov_result.equilibrium.equilibrium_m
    )
    stiffness_difference = (
        finite_result.symmetric_stiffness_n_m - gorkov_result.symmetric_stiffness_n_m
    )
    return StaticModelComparison(
        finite_ka=finite_result,
        gorkov=gorkov_result,
        equilibrium_difference_m=equilibrium_difference,
        equilibrium_difference_norm_m=float(np.linalg.norm(equilibrium_difference)),
        stiffness_difference_n_m=stiffness_difference,
    )


def sample_force_section(
    evaluator: StaticForceEvaluator,
    center_m: Sequence[float],
    *,
    axes: tuple[int, int] = (0, 1),
    half_width_m: float | tuple[float, float] = 2.0e-3,
    shape: tuple[int, int] = (25, 25),
) -> ForceSection:
    """Sample pressure and raw forces on a horizontal or vertical section.

    ``axes=(0,1)`` gives an ``xy`` section and ``axes=(0,2)`` gives ``xz``.
    The unsampled coordinate is fixed at ``center_m``.
    """

    if len(set(axes)) != 2 or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("axes must contain two distinct Cartesian axes")
    if isinstance(half_width_m, tuple):
        half_u, half_v = map(float, half_width_m)
    else:
        half_u = half_v = float(half_width_m)
    if half_u <= 0.0 or half_v <= 0.0:
        raise ValueError("section half widths must be positive")
    n_u, n_v = map(int, shape)
    if n_u < 2 or n_v < 2:
        raise ValueError("each section dimension must contain at least two points")

    center = np.asarray(center_m, dtype=float).reshape(3)
    coordinate_u = np.linspace(-half_u, half_u, n_u)
    coordinate_v = np.linspace(-half_v, half_v, n_v)
    grid_u, grid_v = np.meshgrid(coordinate_u, coordinate_v, indexing="xy")
    points = np.broadcast_to(center, grid_u.shape + (3,)).copy()
    points[..., axes[0]] += grid_u
    points[..., axes[1]] += grid_v
    flat_points = points.reshape(-1, 3)
    pressure = np.asarray(evaluator.field.pressure(flat_points), dtype=np.complex128).reshape(
        grid_u.shape
    )
    radiation = np.empty((len(flat_points), 3), dtype=float)
    total = np.empty_like(radiation)
    for index, point in enumerate(flat_points):
        evaluation = evaluator.force(point)
        radiation[index] = evaluation.radiation_n
        total[index] = evaluation.total_n
    return ForceSection(
        center_m=center,
        axes=tuple(axes),
        coordinates_u_m=grid_u,
        coordinates_v_m=grid_v,
        points_m=points,
        pressure_pa_peak=pressure,
        radiation_force_n=radiation.reshape(grid_u.shape + (3,)),
        total_force_n=total.reshape(grid_u.shape + (3,)),
    )


def sample_horizontal_vertical_sections(
    evaluator: StaticForceEvaluator,
    center_m: Sequence[float],
    *,
    half_width_m: float | tuple[float, float] = 2.0e-3,
    shape: tuple[int, int] = (25, 25),
) -> dict[str, ForceSection]:
    """Return matched ``xy`` and ``xz`` force sections for manuscript figures."""

    return {
        "horizontal_xy": sample_force_section(
            evaluator, center_m, axes=(0, 1), half_width_m=half_width_m, shape=shape
        ),
        "vertical_xz": sample_force_section(
            evaluator, center_m, axes=(0, 2), half_width_m=half_width_m, shape=shape
        ),
    }


def analytic_plane_wave_radiation_force_z(
    pressure_pa_peak: float,
    frequency_hz: float,
    medium: Medium,
    sphere: ElasticSphere,
    lmax: int = 24,
) -> float:
    """Analytic plane-wave series regression for the elastic partial-wave code."""

    coefficients = elastic_sphere_scattering_coefficients(
        frequency_hz, medium, sphere, lmax + 1
    )
    omega = 2.0 * math.pi * float(frequency_hz)
    series = 0.0
    for ell in range(lmax + 1):
        series += (2 * ell + 1) * np.real(coefficients[ell])
        series += 2 * (ell + 1) * np.real(
            np.conj(coefficients[ell]) * coefficients[ell + 1]
        )
    prefactor = -2.0 * math.pi * pressure_pa_peak**2 / (
        medium.density_kg_m3 * omega**2
    )
    return float(prefactor * series)


def elastic_sphere_scattering_regression(
    frequency_hz: float = 40_000.0,
    medium: Medium = Medium(),
    sphere: ElasticSphere = ElasticSphere(),
    lmax: int = 12,
    *,
    rayleigh_ka: float = 1.0e-2,
) -> dict:
    """Return lossless-unitarity and low-``ka`` coefficient diagnostics.

    These values are numerical regressions, not additional physical success
    criteria.  For the declared ``exp(-i omega t)`` convention, conservation
    of energy requires ``|1 + 2 a_l| = 1`` for every lossless partial wave.  In
    the Rayleigh limit the leading monopole and dipole coefficients are
    ``a_0 = -i f_1 (ka)^3/3`` and ``a_1 = +i f_2 (ka)^3/6``.
    """

    coefficients = elastic_sphere_scattering_coefficients(
        frequency_hz, medium, sphere, lmax
    )
    unitarity = np.abs(np.abs(1.0 + 2.0 * coefficients) - 1.0)

    small_frequency_hz = (
        float(rayleigh_ka)
        * medium.sound_speed_m_s
        / (2.0 * math.pi * sphere.radius_m)
    )
    small = elastic_sphere_scattering_coefficients(
        small_frequency_hz, medium, sphere, 2
    )
    f1, f2 = standard_gorkov_contrast_factors(medium, sphere)
    expected_monopole = -1j * f1 * float(rayleigh_ka) ** 3 / 3.0
    expected_dipole = 1j * f2 * float(rayleigh_ka) ** 3 / 6.0

    def relative_error(value: complex, reference: complex) -> float:
        return float(abs(value - reference) / max(abs(reference), 1.0e-300))

    return {
        "frequency_hz": float(frequency_hz),
        "ka": float(
            2.0
            * math.pi
            * frequency_hz
            * sphere.radius_m
            / medium.sound_speed_m_s
        ),
        "lmax": int(lmax),
        "maximum_lossless_unitarity_residual": float(np.max(unitarity)),
        "rayleigh_test_ka": float(rayleigh_ka),
        "rayleigh_monopole_relative_error": relative_error(
            small[0], expected_monopole
        ),
        "rayleigh_dipole_relative_error": relative_error(
            small[1], expected_dipole
        ),
    }


def viscous_rayleigh_inviscid_regression(
    frequency_hz: float = 4_000.0,
    medium: Medium = Medium(),
    sphere: ElasticSphere = ElasticSphere(),
) -> dict:
    """Compare the viscous formula's ``eta -> 0`` limit with Gor'kov.

    A deterministic three-plane-wave interference field supplies nonzero
    pressure- and velocity-gradient terms at three points.  Gravity is
    disabled in both evaluators because the regression targets the radiation
    formula.  The returned differences are diagnostics, not physical pass
    criteria for the manuscript.
    """

    class _RegressionInterferenceField:
        def __init__(self, k_rad_m: float) -> None:
            directions = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.6, 0.0, 0.8],
                    [0.0, -1.0, 0.0],
                ],
                dtype=float,
            )
            self.directions = directions / np.linalg.norm(
                directions, axis=1, keepdims=True
            )
            self.amplitudes = np.array(
                [
                    900.0,
                    630.0 * np.exp(0.37j),
                    410.0 * np.exp(0.81j),
                ],
                dtype=np.complex128,
            )
            self.k_rad_m = float(k_rad_m)

        def pressure(self, points_m: Array) -> Array:
            points = np.asarray(points_m, dtype=float)
            original_shape = points.shape[:-1]
            flat = points.reshape(-1, 3)
            phase = self.k_rad_m * (flat @ self.directions.T)
            values = np.exp(1.0j * phase) @ self.amplitudes
            return values.reshape(original_shape)

        def provenance(self) -> dict:
            return {
                "field_type": "deterministic three-plane-wave regression field",
                "wavenumber_rad_m": self.k_rad_m,
                "pressure_phasor": dict(PEAK_PHASOR_PROVENANCE),
            }

    frequency = float(frequency_hz)
    if frequency <= 0.0:
        raise ValueError("frequency_hz must be positive")
    near_inviscid_medium = Medium(
        density_kg_m3=medium.density_kg_m3,
        sound_speed_m_s=medium.sound_speed_m_s,
        gravity_m_s2=medium.gravity_m_s2,
        dynamic_viscosity_pa_s=1.0e-30,
    )
    k_rad_m = 2.0 * math.pi * frequency / medium.sound_speed_m_s
    wavelength_m = 2.0 * math.pi / k_rad_m
    field = _RegressionInterferenceField(k_rad_m)
    viscous_numerics = ViscousRayleighNumerics(
        pressure_gradient_step_a=0.005,
        velocity_gradient_step_a=0.01,
    )
    gorkov_numerics = GorkovNumerics(
        pressure_gradient_step_a=0.005,
        potential_gradient_step_a=0.01,
    )
    viscous = ViscousElasticRayleighEvaluator(
        frequency,
        field,
        near_inviscid_medium,
        sphere,
        viscous_numerics,
        include_effective_gravity=False,
    )
    gorkov = StandardGorkovEvaluator(
        frequency,
        field,
        near_inviscid_medium,
        sphere,
        gorkov_numerics,
        include_effective_gravity=False,
    )
    points_m = wavelength_m * np.array(
        [
            [0.031, -0.021, 0.043],
            [-0.027, 0.039, 0.012],
            [0.046, 0.014, -0.033],
        ],
        dtype=float,
    )
    viscous_forces = np.vstack([viscous.radiation_force(point) for point in points_m])
    gorkov_forces = np.vstack([gorkov.radiation_force(point) for point in points_m])
    differences = np.linalg.norm(viscous_forces - gorkov_forces, axis=1)
    reference_norms = np.linalg.norm(gorkov_forces, axis=1)
    pointwise_relative = differences / np.maximum(reference_norms, 1.0e-300)
    inviscid_f2 = standard_gorkov_contrast_factors(
        near_inviscid_medium, sphere
    )[1]
    return {
        "frequency_hz": frequency,
        "ka": k_rad_m * sphere.radius_m,
        "dynamic_viscosity_pa_s": near_inviscid_medium.dynamic_viscosity_pa_s,
        "number_of_points": int(len(points_m)),
        "viscous_dipole_contrast": {
            "real": float(np.real(viscous.dipole_contrast)),
            "imag": float(np.imag(viscous.dipole_contrast)),
        },
        "inviscid_gorkov_dipole_contrast": float(inviscid_f2),
        "dipole_contrast_absolute_error": float(
            abs(viscous.dipole_contrast - inviscid_f2)
        ),
        "pointwise_relative_force_errors": pointwise_relative.tolist(),
        "maximum_relative_force_error": float(np.max(pointwise_relative)),
        "aggregate_relative_force_error": float(
            np.linalg.norm(viscous_forces - gorkov_forces)
            / max(np.linalg.norm(gorkov_forces), 1.0e-300)
        ),
        "viscous_radiation_forces_n": viscous_forces.tolist(),
        "gorkov_radiation_forces_n": gorkov_forces.tolist(),
    }


__all__ = [
    "ArbitraryArrayPressureField",
    "ElasticSphere",
    "EquilibriumCandidate",
    "EquilibriumResult",
    "ForceEvaluation",
    "ForceSection",
    "GlobalPressureField",
    "GorkovNumerics",
    "Medium",
    "PartialWaveForceEvaluator",
    "PartialWaveNumerics",
    "PlaneWavePressureField",
    "PEAK_PHASOR_PROVENANCE",
    "StandardGorkovEvaluator",
    "StaticModelComparison",
    "StaticValidationResult",
    "ViscousElasticRayleighEvaluator",
    "ViscousRayleighNumerics",
    "analytic_plane_wave_radiation_force_z",
    "common_cartesian_root_starts",
    "compare_gorkov_and_finite_ka",
    "find_equilibrium",
    "elastic_sphere_scattering_coefficients",
    "elastic_sphere_scattering_regression",
    "force_jacobian",
    "sample_force_section",
    "sample_horizontal_vertical_sections",
    "spherical_quadrature",
    "standard_gorkov_coefficients",
    "standard_gorkov_contrast_factors",
    "validate_static_trap",
    "viscous_elastic_dipole_contrast_factor",
    "viscous_rayleigh_inviscid_regression",
]
