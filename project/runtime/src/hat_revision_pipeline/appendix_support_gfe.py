"""Cached supporting Appendices A, D and E with a current GFE deployment test.

Historical evidence retains its original FE command identities. D3 enumerates
the original offset-rounding family using the current main field and elastic
force evaluator; it does not optimize a new continuous command.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict

from rineng_content_id import content_identity
import importlib.util
import io
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import appendix_a, appendix_de, exact_validator as ev, final_figures as ff, pipeline as p
from .cache import digest_array
from .fixed_fe_endpoints import SINGLE_FE_ENDPOINT
from .triple_long_iteration import MAIN_TRIPLE_ITERATION_CAP
from . import appendix_d3_fields

SCHEMA = "appendix-d3-current-gfe-elastic-offset-v2-literature-grid"
LEVELS = (4, 8, 10, 16, 32, 128, 256, 640, 2048)
QUANTIZED_METHODS = (("Nearest", "nearest"), ("Elastic mechanics offset", "selected"))
PHASE_RESOLUTION_LITERATURE = (
    dict(levels=256, exact_phase_step="2*pi/256", carrier_frequency_hz=40000,
         authors="Shun Suzuki; Seki Inoue; Masahiro Fujiwara; Yasutoshi Makino; Hiroyuki Shinoda",
         title="AUTD3: Scalable Airborne Ultrasound Tactile Display", year=2021,
         venue="IEEE Transactions on Haptics 14(4), 740-749",
         doi="10.1109/TOH.2021.3069976",
         publisher_url="https://ieeexplore.ieee.org/document/9392322/",
         inspected_text_url="https://www.researchgate.net/publication/350535918_AUTD3_Scalable_Airborne_Ultrasound_Tactile_Display",
         location="Accepted manuscript p. 3 Table I; p. 5 Sec. V-A and Fig. 8",
         short_quote="S = 255 corresponds to phi = 2*pi*255/256.",
         hardware="AUTD3; 249 T4010A1 transducers per module; individual 8-bit phase control",
         evidence="Implemented phase parameter S=0,...,255; emitted ultrasound phase measured by a calibrated microphone.",
         interpretation="256 phase levels over one full cycle; separate 8-bit amplitude parameter is not used to infer phase resolution."),
    dict(levels=640, exact_phase_step="2*pi/640", carrier_frequency_hz=40000,
         authors="Shun Suzuki; Masahiro Fujiwara; Yasutoshi Makino; Hiroyuki Shinoda",
         title="Reducing Amplitude Fluctuation by Gradual Phase Shift in Midair Ultrasound Haptics", year=2020,
         venue="IEEE Transactions on Haptics 13(1), 87-93",
         doi="10.1109/TOH.2020.2965946",
         publisher_url="https://ieeexplore.ieee.org/iel7/4543165/9032083/08960301.pdf",
         inspected_text_url="https://www.researchgate.net/publication/338618536_Reducing_Amplitude_Fluctuation_by_Gradual_Phase_Shift_in_Midair_Ultrasound_Haptics",
         location="Accepted manuscript p. 3 Sec. III-A; p. 4 Sec. IV-A, Eq. (17), and Fig. 8",
         short_quote="T into 640 intervals and can specify D and S in 640 levels.",
         hardware="249 T4010A1 transducers; 180 x 140 mm aperture; constant pulse width D and discrete phase parameter S",
         evidence="The experimental phased array uses the explicit 640-step phase command in Eq. (17); its focal field is physically measured.",
         interpretation="640 phase levels over one full cycle; this is a declared implemented count, not an inferred FPGA-clock capability."),
    dict(levels=2048, exact_phase_step="pi/1024", carrier_frequency_hz=40000,
         authors="Sebastian Zehnter; Christoph Ament",
         title="A Modular FPGA-based Phased Array System for Ultrasonic Levitation with MATLAB", year=2019,
         venue="2019 IEEE International Ultrasonics Symposium, 654-658",
         doi="10.1109/ULTSYM.2019.8926194",
         publisher_url="https://ewh.ieee.org/conf/ius/2019/media/files/0201.pdf",
         inspected_text_url="https://ewh.ieee.org/conf/ius/2019/media/files/0201.pdf",
         location="PDF p. 1 abstract; p. 2 Sec. I-B; p. 4 Results",
         short_quote="pi/1024 phase resolution and 10 bit amplitude resolution.",
         hardware="FPGA driver with up to 100 channels; 81.92 MHz clock; 40 kHz emitted ultrasound",
         evidence="Implemented modular driver and physical levitation demonstration; explicit phase increment pi/1024.",
         interpretation="2*pi divided by pi/1024 gives 2048 phase levels; the quoted 10 bits refer to amplitude, not phase."),
)
TITLES = {
    "appendix_A_surrogate_elastic": "Figure A1. Surrogate stiffness and elastic equilibrium offsets",
    "Figure_D1_discretization_comparison": "Figure D1. Phase rounding: accuracy and transition time",
    "Figure_D2_elastic_selected_commands": "Figure D2. Quantization error and the continuous FE baseline",
    "Figure_E1_radius_transfer": "Figure E1. Prescription radius changes with array separation",
    "Figure_E2_single_reference_refinement": "Figure E2. Refining GS against a current compact FE field",
}


def _content_id(path):
    return content_identity(Path(path).read_bytes()).hexdigest()


def _atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_bytes(value)
    temporary.replace(path)


def _text(path, value):
    _atomic_bytes(path, value.encode("utf-8"))


def _json(path, value):
    _text(path, json.dumps(value, indent=2, allow_nan=False))


def _csv(path, frame):
    _text(path, frame.to_csv(index=False))


def _npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_bytes(path, buffer.getvalue())


def _write_phase_resolution_provenance(output):
    """Export literature anchors without claiming to reproduce their hardware."""
    rows = [dict(**record, phase_step_rad=float(2 * np.pi / record["levels"]),
                 phase_step_deg=float(360 / record["levels"]), source_type="primary research paper")
            for record in PHASE_RESOLUTION_LITERATURE]
    _csv(output / "phase_resolution_literature.csv", pd.DataFrame(rows))
    lines = ["# Phase resolutions used in Appendix D", "",
             "The current GFE D3 configuration uses Q = " + ", ".join(map(str, LEVELS)) + ". "
             "The added Q = 256 and 640 values are documented experimental airborne phased-array resolutions. "
             "Q = 2048 retains the explicit high-resolution levitation-driver anchor.", "",
             "D1 and D2 use compact FE commands on their established phase grid, Q = "
             + ", ".join(map(str, appendix_de.LEVELS)) + ". "
             "The same phase grid is retained for these FE comparisons.", "",
             "| Q | Exact phase increment | Degrees | Primary source |", "|---:|---|---:|---|"]
    for record in rows:
        lines.append(f"| {record['levels']} | {record['exact_phase_step']} | {record['phase_step_deg']:.8g} | "
                     f"[{record['authors'].split(';')[0]} et al. ({record['year']})](https://doi.org/{record['doi']}) |")
    for record in rows:
        lines.extend(["", f"## Q = {record['levels']}", "",
                      f"{record['authors']}. **{record['title']}**. {record['venue']} ({record['year']}). "
                      f"[DOI](https://doi.org/{record['doi']}); [publisher source]({record['publisher_url']}); "
                      f"[inspected primary-paper text]({record['inspected_text_url']}).", "",
                      f"Location: {record['location']}.", "",
                      f"Short supporting passage (mathematical symbols normalized to ASCII): “{record['short_quote']}”", "",
                      f"Hardware: {record['hardware']}; carrier frequency {record['carrier_frequency_hz'] / 1000:g} kHz.", "",
                      record['evidence'] + " " + record['interpretation']])
    lines.extend(["", "Q denotes distinct phase levels over 2*pi, with phase step 2*pi/Q. "
                  "It does not denote the number of transducers, the amplitude bit depth, or the packet width. "
                  "No claim of paper-implemented Q = 512 or 1024 is made. "
                  "These sources motivate deployment resolutions; the Appendix D calculations use this study's own field and elastic validator.", ""])
    _text(output / "phase_resolution_literature.md", "\n".join(lines))
    return rows


@contextmanager
def _reading_titles(*, allow_unresolved=False):
    """Apply final readability styling to original vector figure artists."""
    original = Figure.savefig
    def save(figure, filename, *args, **kwargs):
        stem = Path(filename).stem if isinstance(filename, (str, Path)) else ""
        if stem in TITLES and not getattr(figure, "_gfe_support_styled", False):
            if stem == "Figure_D2_elastic_selected_commands":
                reference = _d2_continuous_reference(Path(filename).parents[1],strict=not allow_unresolved)
                if reference["resolved_reference_targets"]:
                    figure.axes[1].axhline(reference["absolute_target_error_um"] * 1e-6, color="#677d62",
                                          linestyle="--", linewidth=2,
                                          label="Continuous FE median")
                    figure.axes[1].legend(loc="best", frameon=False, fontsize=10)
            elif stem == "Figure_E2_single_reference_refinement":
                for axis in figure.axes:
                    if axis.get_xlabel() == "Radial distance (mm)":
                        axis.set_ylabel("Pressure / first ring peak")
            for text in figure.findobj(Text):
                text.set_fontsize(text.get_fontsize() * 1.12)
                if text.get_text():
                    text.set_fontweight("semibold")
            if stem == "appendix_A_surrogate_elastic":
                figure.set_size_inches(7.7, 10.4)
            if figure._suptitle is not None:
                figure._suptitle.remove()
                figure._suptitle = None
            figure._gfe_support_styled = True
        if stem in TITLES:
            # Complete the vector/image serialization before replacing an
            # existing deliverable; interrupted saves never leave an empty PDF.
            path = Path(filename)
            buffer = io.BytesIO()
            options = dict(kwargs)
            options.setdefault("format", path.suffix.lstrip("."))
            original(figure, buffer, *args, **options)
            temporary = path.with_name(path.name + ".rendering")
            temporary.write_bytes(buffer.getvalue())
            temporary.replace(path)
            return None
        return original(figure, filename, *args, **kwargs)
    Figure.savefig = save
    try:
        yield
    finally:
        Figure.savefig = original


def _d2_continuous_reference(output, *, strict=True):
    """Absolute error floor of the frozen FE commands under elastic validation."""
    results = pd.read_csv(output / "elastic_selected_commands.csv")
    continuous = results[results.levels.eq(0) & results.root_found.eq(True)]
    values = continuous.absolute_target_error_um.dropna()
    if not len(values) and strict:
        raise ValueError("D2 requires resolved continuous FE roots for its reference baseline")
    return dict(method="Continuous FE median", levels=0, resolved_reference_targets=len(values),
                absolute_target_error_um=float(values.median()) if len(values) else None,
                absolute_target_error_m=float(values.median()*1e-6) if len(values) else None,
                scope="Frozen current compact FE commands evaluated with the current elastic bead and effective gravity")


def _support_caption_overrides(root):
    """Correct presentation wording while preserving the frozen D/E producer."""
    output = root / "appendix_outputs"
    d_path = output / "D/figure_captions.txt"
    d_text = d_path.read_text(encoding="utf-8")
    reference = _d2_continuous_reference(output / "D")
    marker = "(b) Absolute distance from requested target."
    detail = (" The dashed line denotes the median continuous-FE absolute target error "
              f"({reference['absolute_target_error_um']*1e-6:.3e} m across "
              f"{reference['resolved_reference_targets']} resolved reference targets), "
              "which is present before phase rounding under this elastic validator.")
    if "The dashed line denotes the median continuous-FE" not in d_text:
        _text(d_path, d_text.replace(marker, marker + detail))
    e_path = output / "E/figure_captions.txt"
    e_text = e_path.read_text(encoding="utf-8")
    commands = pd.read_csv(output / "E/candidate_command_manifest.csv")
    opposed = commands[commands.geometry.eq("opposed")]
    dimensions = opposed.independent_phase_coordinates.unique()
    if len(dimensions) != 1:
        raise ValueError("E1 must declare one shared phase-coordinate dimension")
    e_text = e_text.replace(
        "These are pressure-morphology comparisons with 256 paired phase commands shared by two panels.",
        f"The two opposed 16×16 arrays share {int(dimensions[0])} independent phase coordinates. "
        f"These pressure-morphology comparisons evaluate {len(opposed)} candidate commands: "
        f"{opposed.method.nunique()} methods, {opposed.gap_lambda.nunique()} separations, "
        f"and {opposed.radius_lambda.nunique()} radii.")
    _text(e_path, e_text)


def _source_quantizer(root):
    """Load only the model-independent original finite-alphabet utilities."""
    path = appendix_de._de_io_path(root / appendix_de.D_SOURCE / "code/prototypes/discrete_solver.py")
    name = "_gfe_appendix_original_discrete_solver"
    import sys
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name], path


def _frozen_commands(ctx):
    """Use exactly the GFE commands and roots displayed in main mechanics."""
    single = ff._single_exact_benchmark(ctx)
    selected = [(spec, result) for spec, result in zip(single["specs"], single["results"])
                if spec["method"] == "FE+g"]
    if not selected:
        raise ValueError("D3 requires a GFE Single main benchmark command")
    # The smallest paired main seed is fixed before examining quantization.
    spec, result = min(selected, key=lambda pair: int(pair[0]["seed"]))
    triple_bank = ff._triple_exact_benchmark_with_concentration(ctx)
    triple = ff._figure6_vertical_compensated_fe(ctx, triple_bank)
    items = sorted(triple["evaluations"], key=lambda item: item["row"]["target_index"])
    commands = [dict(task="Single", phase=np.asarray(spec["phase"]),
                     targets=np.asarray([spec["target_m"]]), results=[result],
                     seed=int(spec["seed"]),
                     solver_iteration_cap=int(SINGLE_FE_ENDPOINT.maxiter),
                     objective_parameters={"alpha_per_m": 10., "beta_curvature_per_pa": 0.,
                                           "curvature_weight": np.eye(3).tolist(),
                                           "compensate_effective_gravity": True},
                     source="main Single GFE mechanical benchmark; smallest paired seed"),
                dict(task="Triple", phase=np.asarray(triple["phase"]),
                     targets=np.asarray([item["result"].equilibrium.target_m for item in items]),
                     results=[item["result"] for item in items],
                     seed=int(triple_bank["long_bank"]["seed"]),
                     solver_iteration_cap=int(MAIN_TRIPLE_ITERATION_CAP),
                     objective_parameters=triple["config"].to_payload(),
                     source="main Triple GFE mechanical benchmark")]
    if sum(int(value) for value in triple["config"].smooth_stage_maxiters) != MAIN_TRIPLE_ITERATION_CAP:
        raise ValueError("D3 must use the current main Triple GFE command and its configured iteration ceiling")
    for command in commands:
        command["objective_parameters"]["fixed_upward_force_target_n"] = np.asarray(
            triple["fixed_upward_force_target_n"]).tolist()
        if not all(result.equilibrium.numerical_root_found and
                   np.min(result.symmetric_stiffness_eigenvalues_n_m) > 0
                   for result in command["results"]):
            raise ValueError("D3 requires resolved, restoring continuous GFE reference roots")
    return commands


def _transfer(field, points):
    """The exact linear transfer used by ArbitraryArrayPressureField.pressure."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    delta = points[:, None] - field.positions_m[None]
    distance = np.linalg.norm(delta, axis=2)
    cosine = np.einsum("mni,ni->mn", delta / distance[:, :, None], field.normals)
    return (field.source_strength_pa_m * field._directivity(cosine)
            * np.exp(1j * field.k_rad_m * distance) / distance)


def _linear_response(evaluator, center):
    """Factor the unchanged elastic evaluator at a frozen spatial position."""
    center = np.asarray(center)
    field = evaluator.field
    coefficients = evaluator.design_pinv @ _transfer(field, center + evaluator.fit_offsets_m)
    basis = np.eye(len(evaluator.lm), dtype=complex)
    def scattered(offsets):
        matrix = np.column_stack([evaluator._scattered_pressure(offsets, v) for v in basis])
        return matrix @ coefficients
    offsets = evaluator.surface_offsets_m
    incident = _transfer(field, center + offsets)
    total = incident + scattered(offsets)
    incident_velocity, total_velocity = [], []
    for axis in range(3):
        shift = np.eye(3)[axis] * evaluator.velocity_step_m
        plus = _transfer(field, center + offsets + shift)
        minus = _transfer(field, center + offsets - shift)
        denominator = (2 * evaluator.velocity_step_m * 1j * evaluator.omega_rad_s
                       * evaluator.medium.density_kg_m3)
        incident_velocity.append((plus - minus) / denominator)
        total_velocity.append((plus + scattered(offsets + shift)
                               - minus - scattered(offsets - shift)) / denominator)
    return (incident, total, np.stack(incident_velocity, axis=1),
            np.stack(total_velocity, axis=1))


def _forces(evaluator, response, weights):
    """Batch the original momentum-flux contraction without changing physics."""
    incident, total, incident_velocity, total_velocity = response
    def traction(pressure_response, velocity_response):
        pressure = pressure_response @ weights
        velocity = np.einsum("sjn,nc->sjc", velocity_response, weights, optimize=True)
        density = evaluator.medium.density_kg_m3
        isotropic = (np.abs(pressure)**2 / (4 * density * evaluator.medium.sound_speed_m_s**2)
                     - density * np.sum(np.abs(velocity)**2, axis=1) / 4)
        normal_conjugate = np.einsum("sjc,sj->sc", np.conj(velocity), evaluator.surface_directions)
        return (isotropic[:, None] * evaluator.surface_directions[:, :, None]
                + density * np.real(velocity * normal_conjugate[:, None]) / 2)
    delta = traction(total, total_velocity) - traction(incident, incident_velocity)
    area = evaluator.control_radius_m**2 * evaluator.surface_weights
    return -np.einsum("sjc,s->cj", delta, area)


def _candidate_family(phase, levels, quantizer):
    step = 2 * np.pi / levels
    breakpoints, _ = quantizer._breakpoint_groups(phase, step)
    following = np.r_[breakpoints[1:], breakpoints[0] + step]
    offsets = np.mod((breakpoints + following) / 2, step)
    candidates, retained_offsets, seen = [], [], set()
    for offset in offsets:
        indices = np.floor((phase + offset) / step + .5).astype(np.int64) % levels
        indices = quantizer.gauge_fix_indices(indices, levels)
        key = indices.tobytes()
        if key not in seen:
            seen.add(key); candidates.append(indices); retained_offsets.append(offset)
    return np.asarray(candidates), np.asarray(retained_offsets)


def _select_commands(root, ctx, command, output, quantizer, source_path):
    phase = command["phase"]
    identity = dict(schema=SCHEMA, task=command["task"], phase_content_id=digest_array(phase),
                    positions_content_id=digest_array(ctx.positions_m), normals_content_id=digest_array(ctx.normals),
                    frequency_hz=float(ctx.config.frequency_hz), levels=list(LEVELS),
                    protocol=p._finite_ka_protocol(ctx), validator_content_id=_content_id(ev.__file__),
                    source_quantizer_content_id=_content_id(source_path),
                    reference_roots_m=[result.equilibrium.equilibrium_m.tolist() for result in command["results"]],
                    ranking="positive symmetric stiffness at all reference roots, then minimum worst-target predicted shift; stiffness retention and lexicographic ties",
                    prediction="delta_r = K_reference^-1 (F_quantized - F_continuous); K_reference = -sym(J_F)")
    key = content_identity(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = output / "gfe_selection_cache" / key
    cache.mkdir(parents=True, exist_ok=True)
    stamp, data = cache / "selection.json", cache / "selection.npz"
    if stamp.exists() and data.exists():
        saved = json.loads(stamp.read_text(encoding="utf-8"))
        if saved["npz_content_id"] == _content_id(data):
            with np.load(data, allow_pickle=False) as archive:
                return saved, {k: archive[k] for k in archive.files}
    field = p._array_field(ctx, phase)
    evaluator = ev.PartialWaveForceEvaluator(ctx.config.frequency_hz, field,
                                            numerics=p._partial_wave_numerics(ctx))
    h = float(p._finite_ka_protocol(ctx)["jacobian_step_a"]) * evaluator.sphere.radius_m
    began_setup = time.perf_counter()
    responses = []
    audit = []
    for target_index, result in enumerate(command["results"]):
        center = result.equilibrium.equilibrium_m
        centers = [center] + [center + sign * h * np.eye(3)[axis]
                             for axis in range(3) for sign in (1, -1)]
        responses.append([_linear_response(evaluator, point) for point in centers])
        # Verify the factorization against the authoritative evaluator for the
        # continuous command and one fixed quantized command before ranking.
        test_phases = [phase, quantizer.indices_to_phase(quantizer.nearest_indices(phase, 4), 4)]
        for index, test_phase in enumerate(test_phases):
            direct = ev.PartialWaveForceEvaluator(ctx.config.frequency_hz, p._array_field(ctx, test_phase),
                                                  numerics=p._partial_wave_numerics(ctx))
            expected = direct.radiation_force(center)[0]
            actual = _forces(evaluator, responses[-1][0], np.exp(1j * test_phase[:, None]))[0]
            error = float(np.linalg.norm(actual - expected))
            relative = error / max(float(np.linalg.norm(expected)), evaluator.force_scale_n)
            if relative > 1e-9:
                raise ValueError(f"Elastic batch factorization failed: relative discrepancy {relative}")
            audit.append(dict(target_index=target_index, command="continuous" if index == 0 else "Q4 nearest",
                              force_absolute_error_n=error, force_relative_error=relative))
    setup_s = time.perf_counter() - began_setup
    selections, arrays, candidate_rows = [], {}, []
    arrays["continuous_phase_rad"] = phase
    arrays["targets_m"] = command["targets"]
    arrays["reference_roots_m"] = np.asarray(identity["reference_roots_m"])
    for levels in LEVELS:
        began = time.perf_counter()
        indices, offsets = _candidate_family(phase, levels, quantizer)
        weights = np.exp(2j * np.pi * indices.T / levels)
        predicted, eigs = [], []
        for response, result in zip(responses, command["results"]):
            reference_force = _forces(evaluator, response[0], np.exp(1j * phase[:, None]))[0]
            forces = [_forces(evaluator, linear, weights) for linear in response]
            shift = np.linalg.solve(result.symmetric_stiffness_n_m, (forces[0] - reference_force).T).T
            jacobians = np.stack([(forces[1 + 2 * axis] - forces[2 + 2 * axis]) / (2 * h)
                                  for axis in range(3)], axis=2)
            stiffness = -.5 * (jacobians + jacobians.transpose(0, 2, 1))
            predicted.append(np.linalg.norm(shift, axis=1))
            eigs.append(np.linalg.eigvalsh(stiffness))
        predicted = np.asarray(predicted).T
        eigs = np.asarray(eigs).transpose(1, 0, 2)
        worst = predicted.max(axis=1)
        weakest = eigs[:, :, 0].min(axis=1)
        stable = weakest > 0
        if stable.any():
            eligible = np.flatnonzero(stable & (worst <= worst[stable].min() + 1e-12))
            reference_min = np.asarray([r.symmetric_stiffness_eigenvalues_n_m[0] for r in command["results"]])
            retention_error = np.max(np.abs(eigs[:, :, 0] / reference_min - 1), axis=1)
            selected = min(eligible, key=lambda i: (retention_error[i], tuple(indices[i])))
        else:
            selected = min(range(len(indices)), key=lambda i: (-weakest[i], worst[i], tuple(indices[i])))
        transition_s = time.perf_counter() - began
        nearest = quantizer.nearest_indices(phase, levels)
        arrays[f"Q{levels}_candidate_indices"] = indices.astype(np.uint16)
        arrays[f"Q{levels}_offset_rad"] = offsets
        arrays[f"Q{levels}_predicted_shift_m"] = predicted
        arrays[f"Q{levels}_local_eigenvalues_n_m"] = eigs
        for method, command_indices in (("Nearest", nearest), ("Elastic mechanics offset", indices[selected])):
            arrays[f"Q{levels}_{'nearest' if method == 'Nearest' else 'selected'}_phase_rad"] = quantizer.indices_to_phase(command_indices, levels)
        selections.append(dict(levels=levels, selected_candidate=int(selected), candidate_count=len(indices),
                               stable_candidate_count=int(stable.sum()), selected_offset_rad=float(offsets[selected]),
                               predicted_worst_shift_um=float(1e6 * worst[selected]),
                               transition_wall_s=transition_s))
        candidate_rows.extend(dict(task=command["task"], levels=levels, candidate=i,
                                   offset_rad=float(offsets[i]), predicted_worst_shift_um=float(1e6 * worst[i]),
                                   minimum_stiffness_n_m=float(weakest[i]), locally_restoring=bool(stable[i]),
                                   selected=i == int(selected)) for i in range(len(indices)))
        print(f"D3 {command['task']} Q={levels}: {len(indices)} candidates; selected predicted shift {1e6 * worst[selected]:.3g} um", flush=True)
    _npz(data, **arrays)
    saved = dict(identity=identity, npz_content_id=_content_id(data), selections=selections,
                 candidate_rows=candidate_rows, factorization_audit=audit, fixed_root_setup_wall_s=setup_s,
                 timing_scope="Frozen-root elastic response setup is separate; transition times include exact offset-family enumeration and candidate force/stiffness ranking; root validation excluded.")
    _json(stamp, saved)
    return saved, arrays


def _d3(root, ctx):
    output = root / "appendix_outputs/D"
    output.mkdir(parents=True, exist_ok=True)
    literature = _write_phase_resolution_provenance(output)
    quantizer, source_path = _source_quantizer(root)
    commands = _frozen_commands(ctx)
    source_arrays = dict(positions_m=ctx.positions_m, normals=ctx.normals,
                         frequency_hz=np.asarray(ctx.config.frequency_hz))
    source_records = []
    for command in commands:
        for label, values in (("phase_rad", command["phase"]), ("targets_m", command["targets"]),
                              ("elastic_roots_m", np.asarray([r.equilibrium.equilibrium_m for r in command["results"]])),
                              ("elastic_jacobians_n_m", np.asarray([r.force_jacobian_n_m for r in command["results"]]))):
            source_arrays[f"{command['task']}_{label}"] = values
        source_records.append(dict(task=command["task"], method="GFE", seed=command["seed"],
                                   phase_content_id=digest_array(command["phase"]), source=command["source"],
                                   objective_parameters=command["objective_parameters"],
                                   targets_m=command["targets"].tolist(), solver_iteration_cap=command["solver_iteration_cap"],
                                   native_gtol=float(ctx.config.gtol)))
    _npz(output / "D3_GFE_source_commands.npz", **source_arrays)
    _json(output / "D3_GFE_source_parameters.json", dict(commands=source_records,
          field=p._array_field(ctx, commands[0]["phase"]).provenance(),
          sphere=asdict(ev.ElasticSphere()), medium=asdict(ev.Medium()),
          levels=list(LEVELS), historical_d1_d2_levels=list(appendix_de.LEVELS),
          phase_level_definition="Q distinct phases over 2*pi; delta_phi=2*pi/Q",
          literature_sources="phase_resolution_literature.csv", protocol=p._finite_ka_protocol(ctx)))
    rows, candidate_rows, selections, audits = [], [], [], []
    pending = []
    for command in commands:
        saved, arrays = _select_commands(root, ctx, command, output, quantizer, source_path)
        candidate_rows.extend(saved["candidate_rows"])
        selections.extend(dict(task=command["task"], fixed_root_setup_wall_s=saved["fixed_root_setup_wall_s"], **r)
                          for r in saved["selections"])
        audits.extend(dict(task=command["task"], **r) for r in saved["factorization_audit"])
        for index, reference in enumerate(command["results"]):
            rows.append(_validation_row(command, reference, reference, 0, "Continuous GFE", index))
        for levels in LEVELS:
            for method, key in QUANTIZED_METHODS:
                phase = arrays[f"Q{levels}_{key}_phase_rad"]
                for index, reference in enumerate(command["results"]):
                    pending.append((command, phase, reference, levels, method, index))
    def validate(item):
        command, phase, reference, levels, method, index = item
        key = f"D3-GFE-{command['task']}-Q{levels}-{method}-T{index + 1}"
        result = p._finite_ka_validation(ctx, phase, command["targets"][index], key=key)
        row = _validation_row(command, result, reference, levels, method, index)
        row["phase_content_id"] = digest_array(phase)
        dest = output / "gfe_validated_commands"
        dest.mkdir(exist_ok=True)
        _npz(dest / f"{command['task']}_Q{levels}_{'nearest' if method == 'Nearest' else 'offset'}_T{index + 1}.npz",
                            phase_rad=phase, target_m=result.equilibrium.target_m,
                            root_m=result.equilibrium.equilibrium_m,
                            residual_force_n=result.equilibrium.residual_force_n,
                            force_jacobian_n_m=result.force_jacobian_n_m,
                            stiffness_n_m=result.symmetric_stiffness_n_m,
                            candidate_roots_m=np.asarray([c.equilibrium_m for c in result.equilibrium.candidates]),
                            candidate_residuals_scaled=np.asarray([c.residual_scaled_norm for c in result.equilibrium.candidates]),
                            root_found=np.asarray(result.equilibrium.numerical_root_found))
        print(f"D3 validate {command['task']} Q={levels} {method} T{index + 1}: root={row['root_found']}", flush=True)
        return row
    with ThreadPoolExecutor(max_workers=2) as executor:
        for row in executor.map(validate, pending):
            rows.append(row)
            _csv(output / "D3_GFE_quantization_target_results.csv", pd.DataFrame(rows))
    frame = pd.DataFrame(rows)
    _csv(output / "D3_GFE_quantization_target_results.csv", frame)
    _csv(output / "D3_offset_candidates.csv", pd.DataFrame(candidate_rows))
    _csv(output / "D3_selected_offsets.csv", pd.DataFrame(selections))
    _csv(output / "D3_elastic_factorization_audit.csv", pd.DataFrame(audits))
    summary = []
    for (task, levels, method), group in frame.groupby(["task", "levels", "method"], sort=False):
        valid = group.root_found.astype(bool)
        summary.append(dict(task=task, levels=int(levels), method=method, targets=len(group),
                            roots_found=int(valid.sum()), locally_restoring=int(group.locally_restoring.sum()),
                            worst_added_shift_um=float(group.additional_equilibrium_shift_um.max()) if valid.all() else np.nan,
                            worst_absolute_target_error_um=float(group.absolute_target_error_um.max()) if valid.all() else np.nan))
    summary = pd.DataFrame(summary)
    for column in list(summary.columns):
        if column.endswith('_um'):
            summary[column[:-3] + '_m'] = summary[column] * 1e-6
    _csv(output / "D3_GFE_quantization_summary.csv", summary)
    quantized = frame[frame.levels.gt(0)]
    quantized_commands = len(quantized[["task", "levels", "method"]].drop_duplicates())
    quantized_target_validations = len(quantized)
    continuous_reference_roots = int(frame.levels.eq(0).sum())
    figures, spatial_caption, spatial_roots = appendix_d3_fields.plot(output, ctx, commands)
    text = (spatial_caption + "Continuous source solver ceilings: "
        + "; ".join(f"{command['task']} {command['solver_iteration_cap']:,}" for command in commands)
        + " accepted iterations; these ceilings do not override native solver stops. "
        f"{len(commands)} frozen source commands; SMOKE.\n")
    _text(output / "D3_figure_caption.txt", text)
    _text(output / "D3_measured_results_smoke.txt",
        f"SMOKE — current GFE finite-phase deployment\n{int(frame.root_found.sum())}/{len(frame)} target roots resolved; "
        f"{int(frame.locally_restoring.sum())}/{len(frame)} locally restoring, including {continuous_reference_roots} continuous reference roots.\n"
        f"{quantized_commands} quantized commands were validated at {quantized_target_validations} target positions "
        f"across {len(LEVELS)} phase resolutions. Raw targets remain dependent within Triple commands.\n"
        "Same main field, drive, elastic sphere, gravity and root protocol. No continuous optimization was performed in D3.\n\n"
        + summary.to_string(index=False) + "\n")
    _json(output / "D3_provenance.json", dict(schema=SCHEMA, evidence_status="SMOKE", method="GFE",
          commands=[dict(task=c["task"], phase_content_id=digest_array(c["phase"]), targets_m=c["targets"].tolist(),
                         source=c["source"], solver_iteration_cap=c["solver_iteration_cap"]) for c in commands],
          quantized_commands=quantized_commands, quantized_target_validations=quantized_target_validations,
          continuous_reference_roots=continuous_reference_roots,
          levels=list(LEVELS), historical_d1_d2_levels=list(appendix_de.LEVELS),
          literature_phase_levels=[record["levels"] for record in literature],
          literature_sources="phase_resolution_literature.csv",
          active_figure="Figure_D3_GFE_fields_and_equilibria",
          illustration_levels=appendix_d3_fields.ILLUSTRATION_LEVELS,
          spatial_figure_provenance="D3_spatial_figure_provenance.json",
          spatial_producer_content_id=_content_id(appendix_d3_fields.__file__),
          protocol=p._finite_ka_protocol(ctx), implementation_content_id=_content_id(__file__),
          source_quantizer_content_id=_content_id(source_path), validator_content_id=_content_id(ev.__file__),
          timing_scope="Measured candidate-selection work; continuous commands/roots reused; no claim of production synthesis timing",
          generality="ACTIVE: Appendix G1-G9"))
    return dict(figures=figures, table=summary, target_table=frame,
                spatial_table=spatial_roots,
                references=[str(output / "phase_resolution_literature.csv"),
                            str(output / "phase_resolution_literature.md")])


def _validation_row(command, result, reference, levels, method, target_index):
    eq = result.equilibrium
    valid = bool(eq.numerical_root_found)
    return dict(task=command["task"], method=method, levels=int(levels), target_index=int(target_index),
                seed=int(command["seed"]),
                reference_method="GFE", reference_phase_content_id=digest_array(command["phase"]),
                root_found=valid, locally_restoring=bool(valid and min(result.symmetric_stiffness_eigenvalues_n_m) > 0),
                additional_equilibrium_shift_m=float(np.linalg.norm(eq.equilibrium_m - reference.equilibrium.equilibrium_m)) if valid else np.nan,
                absolute_target_error_m=float(eq.displacement_norm_m) if valid else np.nan,
                additional_equilibrium_shift_um=float(1e6 * np.linalg.norm(eq.equilibrium_m - reference.equilibrium.equilibrium_m)) if valid else np.nan,
                absolute_target_error_um=float(1e6 * eq.displacement_norm_m) if valid else np.nan,
                root_residual_scaled=float(eq.residual_scaled_norm),
                stiffness_min_n_m=float(min(result.symmetric_stiffness_eigenvalues_n_m)),
                phase_content_id=digest_array(command["phase"]) if levels == 0 else "",
                evidence_status="SMOKE")


def _export_plot_data(root, *, include_historical_a=True):
    """Expose figure coordinates directly, in addition to complete raw caches."""
    base = root / "appendix_outputs"
    letters = ("A", "D", "E") if include_historical_a else ("D", "E")
    for letter in letters:
        (base / letter / "plot_data").mkdir(exist_ok=True)
    if include_historical_a:
        frame = pd.read_csv(base / "A/elastic_endpoint_results.csv")
        points = []
        for label, eigenvalue in (("Weakest", "min"), ("Middle", "mid"), ("Strongest", "max")):
            for item in frame.to_dict("records"):
                points.append(dict(case_id=item["case_id"], target_id=item["target_id"], seed=item["seed"],
                                   eigenvalue=label, elastic_stiffness_mn_m=1e3 * item[f"elastic_{eigenvalue}_stiffness_n_m"],
                                   gorkov_stiffness_mn_m=1e3 * item[f"gorkov_{eigenvalue}_stiffness_at_elastic_root_n_m"]))
        pd.DataFrame(points).to_csv(base / "A/plot_data/A1a_stiffness_coordinates.csv", index=False)
        targets = pd.read_csv(base / "A/target_coordinates.csv")
        frame.merge(targets[["target_id", "target_index"]], on="target_id")[[
            "case_id", "target_id", "target_index", "selection_role", "seed", "root_displacement_a"]].to_csv(
                base / "A/plot_data/A1b_offset_coordinates.csv", index=False)
    source = pd.read_csv(base / "D/historical_Rayleigh_comparisons.csv")
    plotted = source.groupby(["method", "levels"]).agg(
        added_shift_p90_mm=("actual_equilibrium_shift_mm", lambda values: values.quantile(.9)),
        transition_time_median_ms=("transition_time_median_ms", "median")).reset_index()
    plotted["added_shift_p90_m"] = 1e-3 * plotted.pop("added_shift_p90_mm")
    plotted.to_csv(base / "D/plot_data/D1_rounding_coordinates.csv", index=False)
    elastic_points = pd.read_csv(base / "D/elastic_selected_commands.csv")
    for column in ("additional_equilibrium_shift_um", "absolute_target_error_um"):
        elastic_points[column[:-3] + "_m"] = elastic_points[column] * 1e-6
    elastic_points.to_csv(base / "D/plot_data/D2_elastic_points.csv", index=False)
    pd.DataFrame([_d2_continuous_reference(base / "D")]).to_csv(
        base / "D/plot_data/D2_continuous_reference.csv", index=False)
    pd.read_csv(base / "E/tables/opposed_array_radius_selection_summary.csv").to_csv(
        base / "E/plot_data/E1_radius_transfer_coordinates.csv", index=False)
    with np.load(base / "E/all_candidate_commands_profiles.npz", allow_pickle=False) as archive:
        np.savez_compressed(base / "E/plot_data/E2_pressure_maps.npz",
                            axis_m=archive["map_axis_m"], FE_reference=archive["map_FE"],
                            GS_fixed=archive["map_fixed_GS"], GS_refined=archive["map_refined_GS"])
        rows = []
        for method, key in (("Historical FE", "single_FE_reference__radial_profile"),
                             ("GS R=1.40 lambda", "single_GS_R1p40__radial_profile"),
                             ("GS R=0.45 lambda", "single_GS_R0p45__radial_profile")):
            rows.extend(dict(method=method, radius_mm=float(1e3 * radius), normalized_pressure=float(value))
                        for radius, value in zip(archive["single_radial_axis_m"], archive[key]))
        pd.DataFrame(rows).to_csv(base / "E/plot_data/E2d_radial_coordinates.csv", index=False)
    return {letter: [str(path.relative_to(root)) for path in sorted((base / letter / "plot_data").glob("*"))]
            for letter in letters}


def run(root, ctx=None, *, include_current_gfe=True, include_historical_a=True):
    """Render all planned A/D/E evidence and cache the small GFE deployment extension."""
    root = Path(root).resolve()
    with threadpool_limits(limits=1):
        with _reading_titles():
            a = appendix_a.run(root) if include_historical_a else None
            de = appendix_de.run(root)
        _support_caption_overrides(root)
        result = dict(A=a, D=de["D"], E=de["E"],
                      figures=([str(v) for v in a["figures"]] if a is not None else []) + de["figures"])
        result["plot_data"] = _export_plot_data(root, include_historical_a=include_historical_a)
        if include_current_gfe:
            ctx = ctx or p.prepare_pipeline(run_mode="quick", output_root=root / "smoke_outputs")
            d3 = _d3(root, ctx)
            result["D3"] = d3
            result["figures"] += d3["figures"]
            result["D"]["figures"] = list(result["D"]["figures"]) + d3["figures"]
            result["D"]["tables"] = list(result["D"]["tables"]) + [str(root / "appendix_outputs/D/D3_GFE_quantization_summary.csv")]
            result["D"]["references"] = d3["references"]
            _csv(root / "appendix_outputs/D/plot_data/D3_quantization_coordinates.csv", d3["table"])
            coordinate_path = "appendix_outputs/D/plot_data/D3_quantization_coordinates.csv"
            if coordinate_path not in result["plot_data"]["D"]:
                result["plot_data"]["D"].append(coordinate_path)
            # Spatial D3 creates additional arrays and root coordinates after
            # the historical D/E export pass, including on a clean first run.
            result["plot_data"]["D"] = [str(path.relative_to(root)) for path in
                                        sorted((root / "appendix_outputs/D/plot_data").glob("*"))]
            measured_path = root / "appendix_outputs/D/measured_results_smoke.txt"
            historical = measured_path.read_text(encoding="utf-8").replace(
                "FE+g quantization was not tested.",
                "D1/D2 retain current compact FE commands; current GFE deployment is tested separately in D3.")
            _text(measured_path, historical + "\n\n" + (root / "appendix_outputs/D/D3_measured_results_smoke.txt").read_text(encoding="utf-8"))
        _json(root / "appendix_outputs/support_figure_data_index.json", result["plot_data"])
        return result
