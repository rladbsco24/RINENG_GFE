"""Solid-bead thermoviscous sensitivity in the thin-boundary-layer limit.

Winckelmann & Bruus, PRE 107, 065103 (2023), Eq. (101c,d), with the
arbitrary-incident-field force formula of Karlsen & Bruus, PRE 92, 043010
(2015), Eq. (5). The corrected solid thermal factor is sqrt(1 + X).

This is a separate long-wave sensitivity; it is neither a fluid-sphere model
nor an additive correction to the finite-ka elastic momentum-flux reference.
No full microstreaming solution or temperature field is inferred here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .exact_validator import (
    ElasticSphere, Medium, ViscousElasticRayleighEvaluator,
    ViscousRayleighNumerics, standard_gorkov_contrast_factors,
    regular_wave_pressure_derivatives,
)


@dataclass(frozen=True)
class ThermalProperties:
    """Explicit thermal inputs; particle values have no implicit defaults."""
    heat_capacity_j_kg_k: float
    conductivity_w_m_k: float
    expansion_per_k: float
    heat_capacity_ratio: float
    provenance: str

    def __post_init__(self):
        if self.heat_capacity_j_kg_k <= 0 or self.conductivity_w_m_k <= 0:
            raise ValueError("Heat capacity and thermal conductivity must be positive")
        if self.heat_capacity_ratio < 1 or not self.provenance:
            raise ValueError("A physical heat-capacity ratio and provenance are required")


# Thermal inputs from Table I of Winckelmann & Bruus (2023), at 300 K.
# The benchmark host rho, c and viscosity remain their existing values.
AIR_THERMAL = ThermalProperties(1007.0, 0.0264, 3.35e-3, 1.40,
    "Winckelmann & Bruus 2023 Table I: air thermal parameters at 300 K")
POLYSTYRENE_THERMAL = ThermalProperties(1220.0, 0.154, 2.09e-4, 1.04,
    "Winckelmann & Bruus 2023 Table I: bulk polystyrene thermal parameters; "
    "borrowed sensitivity inputs, not measurements of the low-density benchmark bead")


def thin_layer_solid_coefficients(frequency_hz, medium, sphere,
                                 host_thermal, solid_thermal):
    """Return complex monopole/dipole factors and all asymptotic scales."""
    if not isinstance(sphere, ElasticSphere):
        raise TypeError("Thermoelastic sensitivity requires an ElasticSphere")
    if frequency_hz <= 0 or medium.dynamic_viscosity_pa_s < 0:
        raise ValueError("Positive frequency and nonnegative viscosity are required")
    omega = 2 * math.pi * frequency_hz
    a = sphere.radius_m
    host_diffusivity = host_thermal.conductivity_w_m_k / (
        medium.density_kg_m3 * host_thermal.heat_capacity_j_kg_k)
    solid_diffusivity = solid_thermal.conductivity_w_m_k / (
        sphere.density_kg_m3 * solid_thermal.heat_capacity_j_kg_k)
    delta_s = math.sqrt(2 * medium.dynamic_viscosity_pa_s / (medium.density_kg_m3 * omega))
    delta_t = math.sqrt(2 * host_diffusivity / omega)
    chi = sphere.bulk_modulus_pa / (sphere.density_kg_m3 * sphere.longitudinal_sound_speed_m_s**2)
    solid_x = (solid_thermal.heat_capacity_ratio - 1) * (1 - chi)
    delta_solid_t = math.sqrt(2 * (1 + solid_x) * solid_diffusivity / omega)
    thermal_ratio = solid_thermal.conductivity_w_m_k / host_thermal.conductivity_w_m_k
    effusivity_denominator = 1 + math.sqrt((1 + solid_x) * solid_diffusivity / host_diffusivity) / thermal_ratio
    expansion_ratio = (solid_thermal.expansion_per_k / (sphere.density_kg_m3 * solid_thermal.heat_capacity_j_kg_k)) / (
        host_thermal.expansion_per_k / (medium.density_kg_m3 * host_thermal.heat_capacity_j_kg_k))
    inviscid_f0, inviscid_f1 = standard_gorkov_contrast_factors(medium, sphere)
    thermal_term = 1.5 * (1 + 1j) * (host_thermal.heat_capacity_ratio - 1) * (
        1 - expansion_ratio)**2 * (delta_t / a) / effusivity_denominator
    density_ratio = sphere.density_kg_m3 / medium.density_kg_m3
    dipole_term = 3 * (1 + 1j) * (density_ratio - 1) / (2 * density_ratio + 1) * delta_s / a
    return complex(inviscid_f0 - thermal_term), complex(inviscid_f1 * (1 + dipole_term)), {
        "ka": omega * a / medium.sound_speed_m_s,
        "internal_longitudinal_ka": omega * a / sphere.longitudinal_sound_speed_m_s,
        "internal_shear_ka": omega * a / sphere.shear_sound_speed_m_s,
        "delta_s_m": delta_s, "delta_t_m": delta_t,
        "delta_solid_t_m": delta_solid_t,
        "delta_s_over_a": delta_s/a, "delta_t_over_a": delta_t/a,
        "delta_solid_t_over_a": delta_solid_t/a,
        "solid_X": solid_x, "thermal_factor_sign": "sqrt(1 + X)",
        "host_thermal": asdict(host_thermal), "solid_thermal": asdict(solid_thermal),
        "equations": "Winckelmann & Bruus 2023 Eq. (101c,d); Karlsen & Bruus 2015 Eq. (5)",
    }


class ThermoelasticThinLayerEvaluator(ViscousElasticRayleighEvaluator):
    """Leading thermal and viscous solid response, with explicit input scope."""
    model_name = "thermoelastic solid thin-boundary-layer Rayleigh sensitivity"

    def __init__(self, pressure_field, frequency_hz, medium=Medium(), sphere=ElasticSphere(),
                 numerics=ViscousRayleighNumerics(), *, solid_thermal,
                 host_thermal=AIR_THERMAL, include_effective_gravity=True,
                 incident_projection_numerics=None):
        super().__init__(frequency_hz, pressure_field, medium, sphere, numerics,
                         include_effective_gravity=include_effective_gravity,
                         incident_projection_numerics=incident_projection_numerics)
        self.monopole_contrast, self.dipole_contrast, self.thermal_scales = thin_layer_solid_coefficients(
            frequency_hz, medium, sphere, host_thermal, solid_thermal)

    def provenance(self):
        result = super().provenance()
        result.update({
            "model_name":self.model_name,
            "model_role":"separate long-wave thermoviscous sensitivity with specified thermal inputs",
            "model_scope":"elastic solid; leading thin thermal and shear boundary layers; no imposed background flow",
            "thermal_scales":self.thermal_scales,
            "validity":"ka << 1 and all thermal/shear boundary layers much thinner than a",
            "excluded_physics":["full microstreaming", "nearby-wall correction",
                                "particle-particle multiple scattering", "finite-ka thermoviscous force"],
            "references":["Winckelmann & Bruus, PRE 107, 065103 (2023), Eq. (101)",
                          "Karlsen & Bruus, PRE 92, 043010 (2015), Eq. (5)"],
            "not_included":["finite-ka thermoviscous force", "full microstreaming solution",
                            "temperature-dependent viscosity field", "measured bead thermal calibration"],
        })
        return result


def run_bead_sensitivity(ctx, benchmark, compensation):
    """Apply the literature sensitivity to the two frozen Triple FE commands.

    Main comparative metrics retain the finite-ka model. Root protocols are
    identical to that model's protocol, including all prescribed starts.
    """
    import pandas as pd
    from . import pipeline as p
    from .cache import digest_array
    from .exact_validator import PartialWaveForceEvaluator, validate_static_trap

    protocol = p._finite_ka_protocol(ctx)
    kwargs = dict(
        search_half_width_a=protocol["search_half_width_a"],
        initial_offsets_a=np.asarray(protocol["root_starts_a"]),
        max_nfev=protocol["max_nfev_per_start"],
        numerical_root_tolerance=protocol["numerical_root_tolerance"],
        jacobian_step_a=protocol["jacobian_step_a"],
    )
    items = [item for item in benchmark["evaluations"] if item["row"]["method"] == "FE"]
    items += list(compensation["evaluations"])
    rows = []
    gravity_rows = []
    for item in items:
        method = item["row"]["method"]
        phase = np.asarray(item["spec"]["phase"])
        target = np.asarray(item["result"].equilibrium.target_m)
        reference = item["result"]
        field = p._array_field(ctx, phase)
        evaluator = ThermoelasticThinLayerEvaluator(field, ctx.config.frequency_hz,
            solid_thermal=POLYSTYRENE_THERMAL, numerics=p._viscous_rayleigh_numerics(ctx),
            incident_projection_numerics=p._partial_wave_numerics(ctx))
        payload = {"phase":digest_array(phase), "target":target.tolist(),
                   "model":evaluator.provenance(), "root_protocol":protocol,
                   "array":digest_array(ctx.positions_m)}
        result, _, _ = ctx.cache.get_or_compute("thermoelastic_thin_layer_v1", payload,
            lambda: validate_static_trap(evaluator, target, **kwargs), recompute=ctx.recompute)
        resolved = bool(result.equilibrium.numerical_root_found)
        minimum_stiffness = float(result.symmetric_stiffness_eigenvalues_n_m.min())
        row = {"method":method, "target_id":item["row"]["target_id"],
               "phase_content_id":digest_array(phase),
               "exact_elastic_displacement_a":reference.equilibrium.displacement_norm_a,
               "thermoelastic_displacement_a":result.equilibrium.displacement_norm_a if resolved else np.nan,
               "thermoelastic_candidate_displacement_a":result.equilibrium.displacement_norm_a,
               "thermoelastic_root_found":resolved,
               "thermoelastic_locally_restoring":resolved and minimum_stiffness > 0,
               "thermoelastic_stiffness_min_n_m":minimum_stiffness if resolved else np.nan,
               "thermoelastic_pressure_at_equilibrium_pa":float(abs(field.pressure(result.equilibrium.equilibrium_m))) if resolved else np.nan,
               "thermal_input_status":"borrowed bulk-polystyrene thermal properties; sensitivity only",
               "model_scope":"long-wave thin-boundary-layer solid response; not finite-ka thermoviscous validation"}
        # Hold the field derivative operator and root protocol fixed while
        # removing dissipation. This isolates boundary-layer changes from
        # the larger finite-ka versus long-wave model discrepancy.
        inviscid = ViscousElasticRayleighEvaluator(ctx.config.frequency_hz, field,
            numerics=p._viscous_rayleigh_numerics(ctx),
            incident_projection_numerics=p._partial_wave_numerics(ctx))
        inviscid.monopole_contrast, inviscid.dipole_contrast = standard_gorkov_contrast_factors(
            inviscid.medium, inviscid.sphere)
        inviscid_payload = dict(payload, coefficients="real solid inviscid contrasts")
        inviscid_result, _, _ = ctx.cache.get_or_compute("projected_inviscid_solid_v1", inviscid_payload,
            lambda: validate_static_trap(inviscid, target, **kwargs), recompute=ctx.recompute)
        inviscid_resolved = bool(inviscid_result.equilibrium.numerical_root_found)
        row["projected_inviscid_displacement_a"] = inviscid_result.equilibrium.displacement_norm_a if inviscid_resolved else np.nan
        row["projected_inviscid_root_found"] = inviscid_resolved
        row["projected_inviscid_stiffness_min_n_m"] = float(inviscid_result.symmetric_stiffness_eigenvalues_n_m.min()) if inviscid_resolved else np.nan
        row["thermal_viscous_equilibrium_shift_a"] = float(np.linalg.norm(
            result.equilibrium.equilibrium_m-inviscid_result.equilibrium.equilibrium_m)/inviscid.sphere.radius_m) if resolved and inviscid_resolved else np.nan
        _, _, _, projection_residual = regular_wave_pressure_derivatives(
            evaluator.incident_projection,result.equilibrium.equilibrium_m)
        row["incident_projection_relative_residual"] = projection_residual
        for i, axis in enumerate("xyz"):
            row[f"thermoelastic_displacement_{axis}_m"] = result.equilibrium.displacement_m[i] if resolved else np.nan
        rows.append(row)
        if method == "FE":
            no_gravity = PartialWaveForceEvaluator(ctx.config.frequency_hz, field,
                numerics=p._partial_wave_numerics(ctx), include_effective_gravity=False)
            gravity_payload = {"phase":digest_array(phase), "target":target.tolist(),
                               "protocol":protocol, "gravity":False}
            free, _, _ = ctx.cache.get_or_compute("elastic_gravity_off_v1", gravity_payload,
                lambda: validate_static_trap(no_gravity, target, **kwargs), recompute=ctx.recompute)
            displacement = reference.equilibrium.displacement_m
            weight = no_gravity.effective_weight_n
            gravity = np.array([0., 0., -weight])
            predicted_shift = -np.linalg.solve(free.force_jacobian_n_m, gravity)
            gravity_rows.append({
                "method":method, "target_id":row["target_id"],
                "gravity_on_displacement_a":reference.equilibrium.displacement_norm_a,
                "gravity_off_displacement_a":free.equilibrium.displacement_norm_a,
                "gravity_off_root_found":free.equilibrium.numerical_root_found,
                "gravity_off_stiffness_min_n_m":float(free.symmetric_stiffness_eigenvalues_n_m.min()),
                "axial_fraction_of_displacement":abs(displacement[2])/np.linalg.norm(displacement),
                "actual_gravity_shift_z_m":reference.equilibrium.equilibrium_m[2]-free.equilibrium.equilibrium_m[2],
                "linear_predicted_gravity_shift_z_m":predicted_shift[2],
                "vertical_stiffness_without_gravity_n_m":-free.force_jacobian_n_m[2,2],
                "effective_weight_n":weight,
                "phase_content_id":digest_array(phase),
            })
    ctx.tables["thermoelastic_fe_endpoint_sensitivity"] = pd.DataFrame(rows)
    ctx.tables["triple_gravity_ablation"] = pd.DataFrame(gravity_rows)
    f0, f1, scales = thin_layer_solid_coefficients(ctx.config.frequency_hz, Medium(), ElasticSphere(), AIR_THERMAL, POLYSTYRENE_THERMAL)
    import json
    ctx.tables["thermoelastic_model_parameters"] = pd.DataFrame([{
        **{k:v for k,v in scales.items() if not isinstance(v,dict)},
        "f0_real":f0.real, "f0_imag":f0.imag, "f1_real":f1.real, "f1_imag":f1.imag,
        "host_thermal_json":json.dumps(scales["host_thermal"]),
        "solid_thermal_json":json.dumps(scales["solid_thermal"]),
    }])
    return ctx.tables["thermoelastic_fe_endpoint_sensitivity"]
