"""End-to-end figure orchestration for the revised HAT manuscript.

The module is intentionally boring at its scientific boundaries: every
optimization-side particle potential comes from :mod:`gorkov_core`, every
finite-``ka`` marker comes from :mod:`exact_validator`, and every solution-web
coordinate is fitted before Conventional iterates or parameter sweeps are
placed in it.  Plotting code only composes those audited producers.

The public functions in ``__all__`` are the stable cells used by the standalone
``HAT_revision_all_figures_standard_Gorkov.ipynb`` notebook.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from rineng_content_id import content_identity as content_id
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from .branch_figures import (
    COMPONENTS,
    ComponentBank,
    CorrectedBranchArtifact,
    FrozenBranchFrame2D,
    MatchedComponentFrames2D,
    component_log_iteration_slopes,
    discover_fe_components,
    fit_fixed_2d_frame,
    fit_fixed_3d_frame,
    fit_matched_component_frames_2d,
    load_corrected_branch_artifact,
    plot_branch_web,
    plot_conventional_evolution,
    plot_full_dimensional_evolution,
    plot_joint_branch_web_3d,
    plot_matched_component_web_pair,
    pairwise_projective_distance,
    project_conventional_evolution,
    projective_similarity,
    summarize_full_dimensional_evolution,
)
from .cache import CacheStore, dependency_snapshot, digest_array, digest_file
from .config import RunConfig, square_normals, square_positions
from .exact_validator import (
    ArbitraryArrayPressureField,
    ElasticSphere,
    GorkovNumerics,
    Medium,
    PartialWaveForceEvaluator,
    PartialWaveNumerics,
    PlaneWavePressureField,
    StandardGorkovEvaluator,
    ViscousElasticRayleighEvaluator,
    ViscousRayleighNumerics,
    analytic_plane_wave_radiation_force_z,
    common_cartesian_root_starts,
    compare_gorkov_and_finite_ka,
    force_jacobian,
    sample_force_section,
    validate_static_trap,
    viscous_elastic_dipole_contrast_factor,
    viscous_rayleigh_inviscid_regression,
)
from .gorkov_core import (
    AcousticMedium,
    ArrayGeometry,
    CompressibleSphere,
    Condition,
    MethodSpec,
    MurataMA40S4SDirectivity,
    REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
    SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
    SingleTargetObjective,
    default_stencil_spacing_m,
    formula_self_test,
    gorkov_coefficients,
    method_spec,
    transfer_matrix,
)
from .multitrap import (
    MultiTrapProblem,
    MultitrapRunResult,
    SphereInFluid,
    TargetStencil,
    conventional_double_trap_config,
    evaluate_multitrap_objective,
    regularized_fe_double_trap_config,
    run_multitrap,
    spu_ablation_configs,
)
from .sota import (
    BenchmarkRun,
    SynthesisResult,
    VortexControlSpec,
    benchmark_fe_adjacent_target_warm_path,
    benchmark_fe_cold_start_factory,
    benchmark_ib_initialized_fe_factory,
    benchmark_run_row,
    benchmark_synthesis_method_factory,
    iterative_back_projection,
    phase_only_signature_projection_audit,
    solve_corrected_gorkov_fe,
    vortex_control_points,
    vortex_ring_metrics,
)
from .style import (
    ArtifactWriter,
    COLORS,
    EQUILIBRIUM_COLOR,
    TARGET_COLOR,
    connect_axes,
    field_panel,
    force_streamlines,
    panel_label,
    phase_grid_panel,
    set_panel_title,
    shared_power_norm,
)
from .sweeps import (
    EndpointBank,
    EndpointRequest,
    EndpointSolution,
    FixedFrameProjection,
    SolverContract,
    TargetPoint,
    default_frequencies_hz,
    default_target_grid,
    compare_array_intrinsic_geometry,
    project_endpoint_bank_fixed_frame,
    run_array_endpoint_bank,
    run_frequency_endpoint_bank,
    split_bank_by_array,
    standard_array_ensemble,
)


MAIN_TARGET_M = np.array([0.0, 0.0, 0.050], dtype=float)
NOMINAL_REFERENCE_VORTEX_SPEC = VortexControlSpec(
    point_count=8,
    radius_wavelengths=1.4,
    topological_charge=1,
)
IB_VORTEX_SPEC = VortexControlSpec(
    point_count=8,
    radius_wavelengths=0.45,
    topological_charge=1,
)
SINGLE_SIDED_GS_VORTEX_SPEC = VortexControlSpec(
    point_count=8,
    radius_wavelengths=0.45,
    topological_charge=1,
)
OFFSET_CONTROL_VORTEX_SPEC = VortexControlSpec(
    point_count=8,
    radius_wavelengths=0.60,
    topological_charge=1,
)
STANDARD_FORMULA_TEXT = (
    "U_G = V[f1|p|^2/(4 rho0 c0^2) - 3 f2 rho0|v|^2/8], "
    "F_G = -grad(U_G), peak-pressure phasors"
)
PIPELINE_IMPLEMENTATION = "hat-revision-all-figures-standard-gorkov-v5-exact-equilibrium-audit"


@dataclass
class PipelineContext:
    config: RunConfig
    project_root: Path
    data_root: Path
    output_root: Path
    writer: ArtifactWriter
    cache: CacheStore
    recompute: bool
    artifact: CorrectedBranchArtifact
    branch_bank: ComponentBank
    joint_frame: FrozenBranchFrame2D
    matched_frames: MatchedComponentFrames2D
    conventional_evolution: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    memo: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, mpl.figure.Figure] = field(default_factory=dict)

    @property
    def positions_m(self) -> np.ndarray:
        return square_positions(16, self.config.pitch_m)

    @property
    def normals(self) -> np.ndarray:
        return square_normals(16)


def _module_project_root() -> Path:
    # Source checkout: <project>/src/hat_revision_pipeline/pipeline.py.
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data").exists() and (candidate / "src").exists():
        return candidate
    # Embedded notebook runtime follows the same layout.  The fallback is only
    # for editable installs that retain the repository parent.
    for parent in Path.cwd().resolve().parents:
        if (parent / "hat_revision_notebook" / "data").exists():
            return parent / "hat_revision_notebook"
    raise FileNotFoundError("Could not locate embedded/source HAT revision data")


def _compatible_branch_archive(data_dir: Path, output_root: Path) -> Path:
    """Create the member layout expected by the recovered-artifact reader.

    The notebook embeds the files flat under ``data/corrected_branch_evolution``
    whereas the historical ZIP had a nested ``data/`` member prefix.  This is
    a packaging adapter only; bytes and provenance hashes are unchanged.
    """

    required = (
        "anchor_config.json",
        "anchor_endpoints.csv",
        "anchor_phases.npz",
        "conventional_trajectories.csv",
        "conventional_snapshots.npz",
        "evolution_metadata.json",
    )
    for name in required:
        if not (data_dir / name).exists():
            raise FileNotFoundError(data_dir / name)
    archive = output_root / "data" / "corrected_branch_evolution_artifact.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    signature = content_id()
    for name in required:
        signature.update((data_dir / name).read_bytes())
    stamp = archive.with_suffix(".content_id")
    current = signature.hexdigest()
    if archive.exists() and stamp.exists() and stamp.read_text().strip() == current:
        return archive
    temporary = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(data_dir.iterdir()):
            if path.is_file():
                handle.write(path, f"data/{path.name}")
    os.replace(temporary, archive)
    stamp.write_text(current + "\n", encoding="utf-8")
    return archive


def prepare_pipeline(
    *,
    run_mode: str = "quick",
    recompute: bool = False,
    output_root: str | Path | None = None,
    cache_source_mode: str | None = None,
) -> PipelineContext:
    """Load audited inputs and freeze all solution-space coordinates once."""

    guard = formula_self_test()
    if guard.get("passed") is not True:
        raise RuntimeError(f"Standard Gor'kov formula guard failed: {guard}")
    config = RunConfig.for_mode(run_mode, cache_source_mode=cache_source_mode)
    if config.cache_only and recompute:
        raise ValueError(
            "plot-only is strictly cache-only and cannot be combined with recompute=True"
        )
    project = _module_project_root()
    output = Path(output_root or project / "outputs").resolve()
    output.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(output)
    cache = CacheStore(
        output / "cache",
        namespace=PIPELINE_IMPLEMENTATION,
        read_only=config.cache_only,
    )
    data_dir = project / "data" / "corrected_branch_evolution"
    artifact_path = _compatible_branch_archive(data_dir, output)
    artifact = load_corrected_branch_artifact(artifact_path)
    bank = discover_fe_components(artifact)
    joint = fit_fixed_2d_frame(bank)
    matched = fit_matched_component_frames_2d(bank)
    evolution = project_conventional_evolution(artifact, bank, joint)
    context = PipelineContext(
        config=config,
        project_root=project,
        data_root=project / "data",
        output_root=output,
        writer=writer,
        cache=cache,
        recompute=bool(recompute),
        artifact=artifact,
        branch_bank=bank,
        joint_frame=joint,
        matched_frames=matched,
        conventional_evolution=evolution,
    )
    context.tables["branch_full_dimensional_evolution"] = summarize_full_dimensional_evolution(evolution)
    context.tables["branch_component_regressions"] = component_log_iteration_slopes(evolution)
    context.tables["branch_assignment_edges"] = bank.edge_table.copy()
    context.tables["finite_ka_validation_protocol"] = pd.DataFrame([
        {
            "model": PartialWaveForceEvaluator.model_name,
            "model_role": "independent finite-ka elastic-bead force reference",
            "run_mode": config.mode,
            "budget_mode": config.budget_mode,
            "cache_only": config.cache_only,
            "root_start_count": 19 if config.full else 7,
            "root_start_offsets_a": "target, +/-0.5a, +/-1a, +/-1.5a on each Cartesian axis" if config.full else "target, +/-1a on each Cartesian axis",
            "partial_wave_lmax": config.exact_lmax if config.full else 3,
            "paper_scale_smoke_protocol": not config.full,
            "protocol_json": json.dumps(
                _finite_ka_protocol(context), sort_keys=True, allow_nan=False
            ),
        }
    ])
    medium = AcousticMedium()
    particle = CompressibleSphere()
    exact_medium = Medium()
    elastic_bead = ElasticSphere()
    coefficients = gorkov_coefficients(config.frequency_hz, medium, particle)
    context.tables["physical_parameters"] = pd.DataFrame([
        {"scope": "host medium", "parameter": "density", "symbol": "rho0", "value": medium.density_kg_m3, "unit": "kg m^-3"},
        {"scope": "host medium", "parameter": "sound speed", "symbol": "c0", "value": medium.sound_speed_m_s, "unit": "m s^-1"},
        {"scope": "host medium", "parameter": "gravity", "symbol": "g", "value": exact_medium.gravity_m_s2, "unit": "m s^-2"},
        {"scope": "Gor'kov optimization surrogate bead", "parameter": "radius", "symbol": "a", "value": particle.radius_m, "unit": "m"},
        {"scope": "Gor'kov optimization surrogate bead", "parameter": "density", "symbol": "rho_p", "value": particle.density_kg_m3, "unit": "kg m^-3"},
        {"scope": "Gor'kov optimization surrogate bead", "parameter": "compressional sound speed", "symbol": "c_p", "value": particle.sound_speed_m_s, "unit": "m s^-1"},
        {"scope": "exact elastic bead", "parameter": "radius", "symbol": "a", "value": elastic_bead.radius_m, "unit": "m"},
        {"scope": "exact elastic bead", "parameter": "density", "symbol": "rho_p", "value": elastic_bead.density_kg_m3, "unit": "kg m^-3"},
        {"scope": "exact elastic bead", "parameter": "longitudinal wave speed", "symbol": "c_L", "value": elastic_bead.longitudinal_sound_speed_m_s, "unit": "m s^-1"},
        {"scope": "exact elastic bead", "parameter": "shear wave speed", "symbol": "c_T", "value": elastic_bead.shear_sound_speed_m_s, "unit": "m s^-1"},
        {"scope": "exact elastic bead", "parameter": "bulk modulus", "symbol": "K", "value": elastic_bead.bulk_modulus_pa, "unit": "Pa"},
        {"scope": "exact elastic bead", "parameter": "shear modulus", "symbol": "mu", "value": elastic_bead.shear_modulus_pa, "unit": "Pa"},
        {"scope": "acoustic drive", "parameter": "frequency", "symbol": "f", "value": config.frequency_hz, "unit": "Hz"},
        {"scope": "Gor'kov", "parameter": "monopole contrast", "symbol": "f1", "value": coefficients.monopole_contrast_f1, "unit": "1"},
        {"scope": "Gor'kov", "parameter": "dipole contrast", "symbol": "f2", "value": coefficients.dipole_contrast_f2, "unit": "1"},
        {"scope": "Gor'kov", "parameter": "pressure coefficient", "symbol": "Kp", "value": coefficients.pressure_j_pa2, "unit": "J Pa^-2"},
        {"scope": "Gor'kov", "parameter": "pressure-gradient coefficient", "symbol": "Kg", "value": coefficients.gradient_j_m2_pa2, "unit": "J m^2 Pa^-2"},
    ])
    context.tables["array_source_and_field_parameters"] = pd.DataFrame([
        {"model": "common array", "parameter": "element count", "value": 256, "unit": "1", "note": "16 x 16"},
        {"model": "common array", "parameter": "pitch", "value": config.pitch_m, "unit": "m", "note": "square array"},
        {"model": "common array", "parameter": "aperture x", "value": float(np.ptp(context.positions_m[:, 0])), "unit": "m", "note": "center-to-center span"},
        {"model": "common field", "parameter": "on-axis source strength", "value": REFERENCE_SOURCE_STRENGTH_PA_M_PEAK, "unit": "Pa m peak", "note": "120 dB SPL re 20 uPa RMS at 0.30 m"},
        {"model": "optimization field", "parameter": "Murata-table unit scale", "value": SOURCE_SCALE_PA_M_PER_MURATA_UNIT, "unit": "Pa m peak per table unit", "note": "100 on-axis table units map to the common source strength"},
        {"model": "both global fields", "parameter": "directivity", "value": "Murata MA40S4S amplitude table", "unit": "", "note": "recomputed at every query point"},
        {"model": "both global fields", "parameter": "rear policy", "value": "zero", "unit": "", "note": "front-radiating elements"},
        {"model": "optimization", "parameter": "stencil spacing", "value": default_stencil_spacing_m(config.frequency_hz), "unit": "m", "note": "5 x 5 x 5 target stencil"},
        {"model": "canonical benchmark", "parameter": "target x", "value": MAIN_TARGET_M[0], "unit": "m", "note": "r*"},
        {"model": "canonical benchmark", "parameter": "target y", "value": MAIN_TARGET_M[1], "unit": "m", "note": "r*"},
        {"model": "canonical benchmark", "parameter": "target z", "value": MAIN_TARGET_M[2], "unit": "m", "note": "r*"},
    ])
    partial = asdict(_partial_wave_numerics(context))
    context.tables["finite_ka_numerical_parameters"] = pd.DataFrame(
        [{"parameter": key, "value": value, "unit": "", "run_mode": config.mode}
         for key, value in partial.items()]
        + [
            {"parameter": "root start count", "value": 19 if config.full else 7, "unit": "1", "run_mode": config.mode},
            {"parameter": "root search half width", "value": 4.0, "unit": "particle radii", "run_mode": config.mode},
            {"parameter": "force Jacobian step", "value": 0.10, "unit": "particle radii", "run_mode": config.mode},
            {"parameter": "Gor'kov pressure-gradient step", "value": GorkovNumerics().pressure_gradient_step_a, "unit": "particle radii", "run_mode": config.mode},
            {"parameter": "Gor'kov potential-gradient step", "value": GorkovNumerics().potential_gradient_step_a, "unit": "particle radii", "run_mode": config.mode},
        ]
    )
    return context


def standard_formula_report(ctx: PipelineContext) -> dict[str, Any]:
    report = formula_self_test()
    report.update(
        {
            "formula": STANDARD_FORMULA_TEXT,
            "array_elements": int(len(ctx.positions_m)),
            "main_target_m": MAIN_TARGET_M.tolist(),
            "run_mode": ctx.config.mode,
            "budget_mode": ctx.config.budget_mode,
            "cache_only": ctx.config.cache_only,
        }
    )
    print("Standard Gor'kov formula guard: PASS")
    print("Canonical array: 16 x 16; canonical r*: (0, 0, 0.050) m")
    return report


def _save_figure(
    ctx: PipelineContext,
    fig: mpl.figure.Figure,
    stem: str,
    *,
    tight: bool = True,
) -> mpl.figure.Figure:
    fig.set_constrained_layout(False)
    if tight:
        fig.tight_layout(pad=0.7)
    ctx.writer.save_figure(fig, stem)
    ctx.figures[stem] = fig
    return fig


def _center_row(ctx: PipelineContext, component: str = "B1") -> pd.Series:
    ix, iy = ctx.branch_bank.root_target
    selector = (
        ctx.branch_bank.medoid_table["ix"].eq(ix)
        & ctx.branch_bank.medoid_table["iy"].eq(iy)
        & ctx.branch_bank.medoid_table["component"].eq(component)
    )
    rows = ctx.branch_bank.medoid_table.loc[selector]
    if len(rows) != 1:
        raise RuntimeError(f"Could not resolve canonical {component} endpoint")
    row = rows.iloc[0]
    target = row[["target_x_m", "target_y_m", "target_z_m"]].to_numpy(dtype=float)
    if not np.allclose(target, MAIN_TARGET_M, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"Recovered root target is {target}, not canonical r*={MAIN_TARGET_M}")
    return row


def _fe_raw_phase(ctx: PipelineContext, component: str = "B1") -> np.ndarray:
    row = _center_row(ctx, component)
    selector = ctx.artifact.anchor_table["method"].eq("Force-Equilibrium").to_numpy()
    phases = ctx.artifact.anchor_phases[selector]
    return np.asarray(phases[int(row["source_endpoint_position"])], dtype=float)


def _conventional_raw_phase(ctx: PipelineContext, component: str = "B1") -> np.ndarray:
    ix, iy = ctx.branch_bank.root_target
    rows = ctx.conventional_evolution[
        ctx.conventional_evolution["ix"].eq(ix)
        & ctx.conventional_evolution["iy"].eq(iy)
        & ctx.conventional_evolution["component"].eq(component)
    ]
    if rows.empty:
        raise RuntimeError("Canonical Conventional trajectory is unavailable")
    trajectory = int(rows.iloc[0]["selected_trajectory_index"])
    return np.asarray(ctx.artifact.trajectory_snapshots[trajectory, -1], dtype=float)


def _projective_distance_between(queries: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Gauge-invariant projector distance used for every branch assignment."""

    similarity = projective_similarity(queries, references)
    return np.sqrt(np.maximum(0.0, 1.0 - np.asarray(similarity, dtype=float) ** 2))


def _state_medoid_index(states: np.ndarray) -> int:
    distance = pairwise_projective_distance(np.asarray(states, dtype=np.complex128))
    return int(np.argmin(distance.sum(axis=1)))


def _rigorous_branch_evolution(ctx: PipelineContext) -> dict[str, Any]:
    """Build the fixed FE+RH landmark frame and the full Conventional web.

    The frame is fitted exactly once from final FE and RH target-local medoids.
    Conventional never enters the fit.  At each target and B1/B2 component, the
    terminal Conventional cloud selects one medoid trajectory; that same
    uninterrupted trajectory is then shown at every recorded checkpoint.
    Component correspondence is decided before embedding in the full 256-D
    projective state.  Plotting edges always follow the physical target grid.
    """

    memo_key = "rigorous_fe_rh_branch_evolution"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]

    fe_table = ctx.branch_bank.medoid_table.copy().reset_index(drop=True)
    fe_states = np.asarray(ctx.branch_bank.medoid_states, dtype=np.complex128)
    fe_table["method"] = "Force-Equilibrium"
    fe_table["method_short"] = "FE"

    rh_endpoints, rh_states = ctx.artifact.anchor_states("Regularized-Hybrid")
    rh_rows: list[dict[str, Any]] = []
    rh_medoid_states: list[np.ndarray] = []
    for (ix, iy), target_group in rh_endpoints.groupby(["ix", "iy"], sort=True):
        target_positions = target_group.index.to_numpy(dtype=int)
        reference_states = np.stack(
            [ctx.branch_bank.target_medoid(int(ix), int(iy), component) for component in COMPONENTS]
        )
        distance = _projective_distance_between(rh_states[target_positions], reference_states)
        labels = np.asarray(COMPONENTS, dtype=object)[np.argmin(distance, axis=1)]
        for component in COMPONENTS:
            members = target_positions[labels == component]
            if len(members) == 0:
                raise RuntimeError(f"RH did not recover {component} at target {(int(ix), int(iy))}")
            chosen = int(members[_state_medoid_index(rh_states[members])])
            source = rh_endpoints.loc[chosen].to_dict()
            source.update(
                component=component,
                method="Regularized-Hybrid",
                method_short="RH",
                source_endpoint_position=chosen,
                component_size=int(len(members)),
            )
            rh_rows.append(source)
            rh_medoid_states.append(rh_states[chosen])

    rh_table = pd.DataFrame(rh_rows)
    combined_table = pd.concat([fe_table, rh_table], ignore_index=True, sort=False)
    combined_states = np.vstack([fe_states, np.stack(rh_medoid_states)])
    landmark_bank = ComponentBank(
        endpoint_table=combined_table.copy(),
        endpoint_states=combined_states.copy(),
        medoid_table=combined_table.copy(),
        medoid_states=combined_states.copy(),
        edge_table=ctx.branch_bank.edge_table.copy(),
        root_target=ctx.branch_bank.root_target,
    )
    frame = fit_fixed_3d_frame(landmark_bank)
    reference = combined_table.copy()
    reference[["coordinate_1", "coordinate_2", "coordinate_3"]] = frame.coordinates

    trajectory_states = ctx.artifact.trajectory_states()
    terminal_states = trajectory_states[:, -1, :]
    trajectory_table = ctx.artifact.trajectory_table.reset_index(drop=True)
    selected: list[tuple[int, str]] = []
    for (ix, iy), group in trajectory_table.groupby(["ix", "iy"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        component_distance = []
        for component in COMPONENTS:
            selector = (
                combined_table["ix"].eq(int(ix))
                & combined_table["iy"].eq(int(iy))
                & combined_table["component"].eq(component)
            ).to_numpy()
            component_distance.append(
                np.min(_projective_distance_between(terminal_states[positions], combined_states[selector]), axis=1)
            )
        labels = np.asarray(COMPONENTS, dtype=object)[np.argmin(np.column_stack(component_distance), axis=1)]
        for component in COMPONENTS:
            members = positions[labels == component]
            if len(members) == 0:
                raise RuntimeError(
                    f"Conventional terminal bank did not recover {component} at {(int(ix), int(iy))}"
                )
            chosen = int(members[_state_medoid_index(terminal_states[members])])
            selected.append((chosen, component))

    evolution_rows: list[dict[str, Any]] = []
    evolution_states: list[np.ndarray] = []
    for trajectory_index, component in selected:
        source = trajectory_table.iloc[trajectory_index]
        ix, iy = int(source["ix"]), int(source["iy"])
        component_selector = (
            combined_table["ix"].eq(ix)
            & combined_table["iy"].eq(iy)
            & combined_table["component"].eq(component)
        ).to_numpy()
        target_references = combined_states[component_selector]
        fe_selector = (
            fe_table["ix"].eq(ix)
            & fe_table["iy"].eq(iy)
            & fe_table["component"].eq(component)
        ).to_numpy()
        target_fe_reference = fe_states[fe_selector]
        if len(target_fe_reference) != 1:
            raise RuntimeError(
                f"FE reference is not unique for {component} at {(ix, iy)}"
            )
        for checkpoint_index, checkpoint in enumerate(ctx.artifact.checkpoints):
            state = trajectory_states[trajectory_index, checkpoint_index]
            evolution_states.append(state)
            evolution_rows.append(
                {
                    **source.to_dict(),
                    "component": component,
                    "iteration": int(checkpoint),
                    "selected_trajectory_index": int(trajectory_index),
                    "distance_to_target_fe_rh": float(
                        np.min(_projective_distance_between(state[None, :], target_references))
                    ),
                    "distance_to_target_fe": float(
                        _projective_distance_between(state[None, :], target_fe_reference)[0, 0]
                    ),
                }
            )
    evolution = pd.DataFrame(evolution_rows)
    evolution[["coordinate_1", "coordinate_2", "coordinate_3"]] = frame.project(
        np.stack(evolution_states)
    )

    # The audited evolution bundle also deposits the exact fixed-frame
    # coordinates used by the original, legible 3-D web.  Its landmark fit is
    # the same declared FE+RH-only fit, and its selected Conventional
    # trajectories must agree with the full-dimensional selection above.  Use
    # those deposited camera coordinates instead of silently refitting an
    # alternative set of equally valid local medoids for presentation.
    coordinate_root = ctx.project_root / "data" / "corrected_branch_evolution"
    reference_path = coordinate_root / "fixed_anchor_coordinates.csv"
    evolution_path = coordinate_root / "conventional_iteration_coordinates.csv"
    if reference_path.exists() and evolution_path.exists():
        component_map_table = fe_table[
            ["component", "posthoc_legacy_winding_label"]
        ].drop_duplicates()
        if len(component_map_table) != len(COMPONENTS):
            raise RuntimeError("could not map deposited sheet identities to B1/B2")
        component_map = dict(
            zip(
                component_map_table["posthoc_legacy_winding_label"],
                component_map_table["component"],
            )
        )
        deposited_reference = pd.read_csv(reference_path)
        deposited_evolution = pd.read_csv(evolution_path)
        deposited_reference["component"] = deposited_reference["branch"].map(component_map)
        deposited_evolution["component"] = deposited_evolution["branch"].map(component_map)
        if deposited_reference["component"].isna().any() or deposited_evolution["component"].isna().any():
            raise RuntimeError("deposited fixed-frame sheets do not match discovered B1/B2 components")
        deposited_reference = deposited_reference.rename(
            columns={
                "mds1": "coordinate_1",
                "mds2": "coordinate_2",
                "mds3": "coordinate_3",
                "source_index": "source_endpoint_position",
            }
        )
        deposited_evolution = deposited_evolution.rename(
            columns={
                "mds1": "coordinate_1",
                "mds2": "coordinate_2",
                "mds3": "coordinate_3",
                "source_index": "selected_trajectory_index",
            }
        )
        quantitative = evolution[
            [
                "ix", "iy", "component", "iteration",
                "selected_trajectory_index", "distance_to_target_fe_rh",
                "distance_to_target_fe",
            ]
        ]
        checked = deposited_evolution.merge(
            quantitative,
            on=["ix", "iy", "component", "iteration"],
            how="left",
            suffixes=("_deposited", "_selected"),
            validate="one_to_one",
        )
        if checked["distance_to_target_fe_rh"].isna().any() or not np.array_equal(
            checked["selected_trajectory_index_deposited"].to_numpy(dtype=int),
            checked["selected_trajectory_index_selected"].to_numpy(dtype=int),
        ):
            raise RuntimeError(
                "deposited web coordinates disagree with the full-dimensional Conventional selection"
            )
        deposited_evolution["distance_to_target_fe_rh"] = checked[
            "distance_to_target_fe_rh"
        ].to_numpy(dtype=float)
        deposited_evolution["distance_to_target_fe"] = checked[
            "distance_to_target_fe"
        ].to_numpy(dtype=float)
        reference = deposited_reference
        evolution = deposited_evolution

    contract = pd.DataFrame(
        [
            {
                "landmark_definition": "final FE+RH full-dimensional medoids",
                "landmark_count": len(reference),
                "conventional_in_landmark_fit": False,
                "stage_refitting": False,
                "trajectory_identity": "terminal Conventional medoid trajectory reused at every checkpoint",
                "target_count": int(reference[["ix", "iy"]].drop_duplicates().shape[0]),
                "component_count": len(COMPONENTS),
                "display_coordinates": "deposited fixed FE+RH landmark frame; Conventional excluded",
            }
        ]
    )
    ctx.tables["branch_evolution_contract"] = contract
    ctx.tables["branch_evolution_full_dimensional"] = evolution[
        [
            "component", "target_id", "ix", "iy", "iteration",
            "selected_trajectory_index", "distance_to_target_fe_rh",
            "distance_to_target_fe",
        ]
    ].copy()
    value = {
        "landmark_bank": landmark_bank,
        "frame": frame,
        "reference": reference,
        "evolution": evolution,
    }
    ctx.memo[memo_key] = value
    return value


def _array_field(
    ctx: PipelineContext,
    phase: np.ndarray,
    *,
    positions_m: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    frequency_hz: float | None = None,
) -> ArbitraryArrayPressureField:
    positions = ctx.positions_m if positions_m is None else np.asarray(positions_m, dtype=float)
    if normals is None:
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(positions), 1))
    return ArbitraryArrayPressureField(
        float(frequency_hz or ctx.config.frequency_hz),
        positions,
        np.exp(1j * np.asarray(phase, dtype=float)),
        normals=normals,
    )


def _pressure_sections(
    field: ArbitraryArrayPressureField,
    target_m: Sequence[float],
    *,
    half_xy_m: float = 0.010,
    half_z_m: float = 0.012,
    samples: int = 101,
) -> dict[str, Any]:
    target = np.asarray(target_m, dtype=float)
    axis_xy = np.linspace(-half_xy_m, half_xy_m, int(samples))
    axis_z = np.linspace(-half_z_m, half_z_m, int(samples))
    xx, yy = np.meshgrid(axis_xy, axis_xy, indexing="xy")
    xz_x, zz = np.meshgrid(axis_xy, axis_z, indexing="xy")
    yz_y, yz_z = np.meshgrid(axis_xy, axis_z, indexing="xy")
    xy_points = np.stack(
        [target[0] + xx, target[1] + yy, np.full_like(xx, target[2])], axis=-1
    )
    xz_points = np.stack(
        [target[0] + xz_x, np.full_like(xz_x, target[1]), target[2] + zz], axis=-1
    )
    yz_points = np.stack(
        [np.full_like(yz_y, target[0]), target[1] + yz_y, target[2] + yz_z], axis=-1
    )
    return {
        "xy_axis_m": axis_xy,
        "z_axis_m": axis_z,
        "xy": np.abs(field.pressure(xy_points.reshape(-1, 3))).reshape(xx.shape),
        "xz": np.abs(field.pressure(xz_points.reshape(-1, 3))).reshape(xz_x.shape),
        "yz": np.abs(field.pressure(yz_points.reshape(-1, 3))).reshape(yz_y.shape),
    }


def _pressure_abs_chunked(
    field: ArbitraryArrayPressureField,
    points_m: np.ndarray,
    *,
    chunk_points: int = 4096,
) -> np.ndarray:
    """Evaluate a large render-only point cloud without a monolithic M x N array."""

    points = np.asarray(points_m, dtype=float).reshape(-1, 3)
    values = np.empty(len(points), dtype=float)
    for start in range(0, len(points), int(chunk_points)):
        stop = min(start + int(chunk_points), len(points))
        values[start:stop] = np.abs(field.pressure(points[start:stop]))
    return values


def _morphology_core_cloud(
    ax: mpl.axes.Axes,
    field: ArbitraryArrayPressureField,
    target_m: Sequence[float],
    *,
    full: bool,
) -> None:
    """Old-notebook-style target-centred 3-D morphology context.

    The original notebook used marching-cubes shells.  To keep the delivered
    notebook dependency-light, this renderer samples the same target-centred
    volume and draws its upper-pressure quantile as a translucent 3-D cloud.
    It is a render-only view; no sampled value enters optimization or tables.
    """

    target = np.asarray(target_m, dtype=float)
    half_width_m = 0.018
    count = 51 if full else 35
    axis = np.linspace(-half_width_m, half_width_m, count)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    relative = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    values = _pressure_abs_chunked(field, target + relative)
    threshold = float(np.quantile(values, 0.985))
    chosen = np.flatnonzero(values >= threshold)
    max_points = 2800
    if len(chosen) > max_points:
        chosen = chosen[np.linspace(0, len(chosen) - 1, max_points, dtype=int)]
    cloud = 1e3 * relative[chosen]
    selected_values = values[chosen]
    norm = mpl.colors.PowerNorm(
        gamma=0.60,
        vmin=float(np.quantile(selected_values, 0.02)),
        vmax=float(np.quantile(selected_values, 0.995)),
    )
    ax.scatter(
        cloud[:, 0], cloud[:, 1], cloud[:, 2],
        c=selected_values, cmap="viridis", norm=norm,
        s=2.2, alpha=0.24, linewidths=0, depthshade=False,
    )
    ax.scatter([0.0], [0.0], [0.0], marker="x", color=TARGET_COLOR, s=28, linewidth=1.4)
    limit = 18.0
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=26, azim=-58)
    ax.set_xlabel("x (mm)", fontsize=7, labelpad=-2)
    ax.set_ylabel("y (mm)", fontsize=7, labelpad=-2)
    ax.set_zlabel("z (mm)", fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=-1)
    ax.grid(alpha=0.16)


def _full_validation_resolution(ctx: PipelineContext) -> bool:
    """Keep statistical full expansions on the accepted smoke validation grid."""
    return ctx.config.full and not ctx.memo.get("study_scale", {}).get("use_smoke_validation", False)


def _partial_wave_numerics(ctx: PipelineContext, lmax: int | None = None) -> PartialWaveNumerics:
    if _full_validation_resolution(ctx):
        return PartialWaveNumerics(
            lmax=int(lmax or ctx.config.exact_lmax),
            fit_shell_wavelengths=(0.10, 0.15, 0.20),
            fit_n_mu=12,
            fit_n_phi=24,
            surface_n_mu=20,
            surface_n_phi=40,
        )
    return PartialWaveNumerics(
        lmax=int(lmax or 3),
        fit_shell_wavelengths=(0.10, 0.16, 0.22),
        fit_n_mu=6,
        fit_n_phi=12,
        surface_n_mu=8,
        surface_n_phi=16,
    )


def _finite_ka_protocol(ctx: PipelineContext) -> dict[str, Any]:
    """Return the complete immutable contract for one equilibrium evaluation."""

    full_resolution = _full_validation_resolution(ctx)
    starts = (
        common_cartesian_root_starts()
        if full_resolution
        else common_cartesian_root_starts((1.0,))
    )
    return {
        "numerics": asdict(_partial_wave_numerics(ctx)),
        "root_starts_a": starts.tolist(),
        "search_half_width_a": 4.0,
        "max_nfev_per_start": 70 if full_resolution else 14,
        "numerical_root_tolerance": 1.0e-3 if full_resolution else 2.5e-3,
        "least_squares": {
            "algorithm": "scipy.optimize.least_squares with bounded default method",
            "xtol": 1.0e-10,
            "ftol": 1.0e-10,
            "gtol": 1.0e-10,
        },
        "root_acceptance": "optimizer success and interior point and scaled total-force residual <= numerical_root_tolerance",
        "boundary_fraction_epsilon": 1.0e-7,
        "resolved_root_selection": "minimum displacement norm, then minimum scaled residual",
        "unresolved_candidate_selection": "interior, then optimizer success, then scaled residual, then displacement norm",
        "jacobian_step_a": 0.10,
        "medium": asdict(Medium()),
        "sphere": asdict(ElasticSphere()),
        "include_effective_gravity": True,
        "source_strength_pa_m_peak": REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
        "directivity": "Murata MA40S4S amplitude table evaluated globally",
        "front_only": True,
        "model": PartialWaveForceEvaluator.model_name,
        "model_role": "independent finite-ka elastic-bead force reference",
        "model_scope": (
            "homogeneous isotropic elastic solid sphere with internal longitudinal "
            "and shear modes; inviscid host momentum-flux integration"
        ),
        "dependencies": dependency_snapshot(),
    }


def _finite_ka_validation(
    ctx: PipelineContext,
    phase: np.ndarray,
    target_m: Sequence[float],
    *,
    key: str,
    positions_m: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    frequency_hz: float | None = None,
) -> Any:
    """Validate one frozen phase command without reusing the design objective.

    Unlike :func:`_exact_comparison`, this path does not also solve a Gor'kov
    equilibrium.  It is the cost-controlled producer used by the paired
    Conventional--FE benchmark; the incident field, finite-ka scattering,
    total-force root and equilibrium-centred stiffness remain identical.
    """

    target = np.asarray(target_m, dtype=float)
    positions = ctx.positions_m if positions_m is None else np.asarray(positions_m, dtype=float)
    normal_values = (
        np.tile(np.array([0.0, 0.0, 1.0]), (len(positions), 1))
        if normals is None
        else np.asarray(normals, dtype=float)
    )
    protocol = _finite_ka_protocol(ctx)
    payload = {
        "key": str(key),
        "phase": digest_array(np.asarray(phase, dtype=float)),
        "positions": digest_array(positions),
        "normals": digest_array(normal_values),
        "target": target.tolist(),
        "frequency_hz": float(frequency_hz or ctx.config.frequency_hz),
        "protocol": protocol,
    }

    def producer() -> Any:
        medium = Medium()
        sphere = ElasticSphere()
        evaluator = PartialWaveForceEvaluator(
            float(frequency_hz or ctx.config.frequency_hz),
            _array_field(
                ctx,
                phase,
                positions_m=positions,
                normals=normal_values,
                frequency_hz=frequency_hz,
            ),
            medium=medium,
            sphere=sphere,
            numerics=_partial_wave_numerics(ctx),
            include_effective_gravity=True,
        )
        return validate_static_trap(
            evaluator,
            target,
            search_half_width_a=float(protocol["search_half_width_a"]),
            initial_offsets_a=np.asarray(protocol["root_starts_a"], dtype=float),
            max_nfev=int(protocol["max_nfev_per_start"]),
            numerical_root_tolerance=float(protocol["numerical_root_tolerance"]),
            jacobian_step_a=float(protocol["jacobian_step_a"]),
        )

    value, _, _ = ctx.cache.get_or_compute(
        "finite_ka_static_validation", payload, producer, recompute=ctx.recompute
    )
    return value


def _viscous_rayleigh_numerics(ctx: PipelineContext) -> ViscousRayleighNumerics:
    """Return the fixed differencing contract for the table-only sensitivity."""

    if ctx.config.full:
        return ViscousRayleighNumerics(
            pressure_gradient_step_a=0.02,
            velocity_gradient_step_a=0.05,
        )
    return ViscousRayleighNumerics(
        pressure_gradient_step_a=0.03,
        velocity_gradient_step_a=0.075,
    )


def _viscous_elastic_protocol(ctx: PipelineContext) -> dict[str, Any]:
    """Complete cache and reporting contract for the secondary sensitivity.

    Root starts, search bounds, residual tolerance and equilibrium-centred
    Jacobian convention are intentionally copied from the finite-``ka``
    protocol.  The force law itself remains a separate long-wave model; it is
    never blended with the finite-``ka`` elastic partial-wave reference.
    """

    finite_protocol = _finite_ka_protocol(ctx)
    medium = Medium()
    sphere = ElasticSphere()
    frequency_hz = float(ctx.config.frequency_hz)
    omega = 2.0 * math.pi * frequency_hz
    wavenumber = omega / medium.sound_speed_m_s
    shear_layer = math.sqrt(
        2.0 * medium.dynamic_viscosity_pa_s / (medium.density_kg_m3 * omega)
    )
    dipole = viscous_elastic_dipole_contrast_factor(
        frequency_hz, medium, sphere
    )
    compressibility_ratio = (
        medium.density_kg_m3
        * medium.sound_speed_m_s**2
        / sphere.bulk_modulus_pa
    )
    return {
        "numerics": asdict(_viscous_rayleigh_numerics(ctx)),
        "incident_projection_numerics":asdict(_partial_wave_numerics(ctx)),
        "incident_derivative_operator":"analytic regular-wave center derivatives; unweighted least-squares fit on spherical quadrature nodes",
        "root_starts_a": list(finite_protocol["root_starts_a"]),
        "search_half_width_a": float(finite_protocol["search_half_width_a"]),
        "max_nfev_per_start": int(finite_protocol["max_nfev_per_start"]),
        "numerical_root_tolerance": float(
            finite_protocol["numerical_root_tolerance"]
        ),
        "root_acceptance": str(finite_protocol["root_acceptance"]),
        "resolved_root_selection": str(
            finite_protocol["resolved_root_selection"]
        ),
        "unresolved_candidate_selection": str(
            finite_protocol["unresolved_candidate_selection"]
        ),
        "jacobian_step_a": float(finite_protocol["jacobian_step_a"]),
        "medium": asdict(medium),
        "sphere": asdict(sphere),
        "include_effective_gravity": True,
        "frequency_hz": frequency_hz,
        "wavenumber_rad_m": wavenumber,
        "ka": wavenumber * sphere.radius_m,
        "shear_boundary_layer_thickness_m": shear_layer,
        "shear_boundary_layer_to_radius": shear_layer / sphere.radius_m,
        "monopole_contrast_real": float(1.0 - compressibility_ratio),
        "monopole_contrast_imag": 0.0,
        "dipole_contrast_real": float(np.real(dipole)),
        "dipole_contrast_imag": float(np.imag(dipole)),
        "source_strength_pa_m_peak": REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
        "directivity": "Murata MA40S4S amplitude table evaluated globally",
        "front_only": True,
        "model": ViscousElasticRayleighEvaluator.model_name,
        "model_role": (
            "secondary table-only viscosity sensitivity; not the finite-ka "
            "elastic reference"
        ),
        "validity": "long-wave sensitivity requiring ka << 1",
        "excluded_physics": [
            "thermal boundary-layer correction",
            "microstreaming",
            "nearby-wall correction",
            "particle-particle multiple scattering",
            "finite-ka viscous hybridization",
        ],
        "references": [
            "Settnes and Bruus, Phys. Rev. E 85, 016327 (2012)",
            "Karlsen and Bruus, Phys. Rev. E 92, 043010 (2015)",
        ],
        "dependencies": dependency_snapshot(),
    }


def _viscous_elastic_validation(
    ctx: PipelineContext,
    phase: np.ndarray,
    target_m: Sequence[float],
    *,
    key: str,
    positions_m: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    frequency_hz: float | None = None,
) -> Any:
    """Validate one frozen command with the table-only viscous sensitivity."""

    target = np.asarray(target_m, dtype=float)
    positions = (
        ctx.positions_m if positions_m is None else np.asarray(positions_m, dtype=float)
    )
    normal_values = (
        np.tile(np.array([0.0, 0.0, 1.0]), (len(positions), 1))
        if normals is None
        else np.asarray(normals, dtype=float)
    )
    protocol = _viscous_elastic_protocol(ctx)
    evaluation_frequency_hz = float(frequency_hz or ctx.config.frequency_hz)
    if not math.isclose(
        evaluation_frequency_hz,
        float(protocol["frequency_hz"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        # A noncanonical frequency still receives a complete material/numerical
        # fingerprint rather than inheriting the default-frequency diagnostics.
        medium = Medium()
        sphere = ElasticSphere()
        omega = 2.0 * math.pi * evaluation_frequency_hz
        shear_layer = math.sqrt(
            2.0
            * medium.dynamic_viscosity_pa_s
            / (medium.density_kg_m3 * omega)
        )
        dipole = viscous_elastic_dipole_contrast_factor(
            evaluation_frequency_hz, medium, sphere
        )
        protocol = dict(protocol)
        protocol.update(
            {
                "frequency_hz": evaluation_frequency_hz,
                "wavenumber_rad_m": omega / medium.sound_speed_m_s,
                "ka": (
                    omega
                    * sphere.radius_m
                    / medium.sound_speed_m_s
                ),
                "shear_boundary_layer_thickness_m": shear_layer,
                "shear_boundary_layer_to_radius": (
                    shear_layer / sphere.radius_m
                ),
                "dipole_contrast_real": float(np.real(dipole)),
                "dipole_contrast_imag": float(np.imag(dipole)),
            }
        )
    payload = {
        "key": str(key),
        "phase": digest_array(np.asarray(phase, dtype=float)),
        "positions": digest_array(positions),
        "normals": digest_array(normal_values),
        "target": target.tolist(),
        "frequency_hz": evaluation_frequency_hz,
        "protocol": protocol,
    }

    def producer() -> Any:
        evaluator = ViscousElasticRayleighEvaluator(
            evaluation_frequency_hz,
            _array_field(
                ctx,
                phase,
                positions_m=positions,
                normals=normal_values,
                frequency_hz=evaluation_frequency_hz,
            ),
            medium=Medium(),
            sphere=ElasticSphere(),
            numerics=_viscous_rayleigh_numerics(ctx),
            incident_projection_numerics=_partial_wave_numerics(ctx),
            include_effective_gravity=True,
        )
        return validate_static_trap(
            evaluator,
            target,
            search_half_width_a=float(protocol["search_half_width_a"]),
            initial_offsets_a=np.asarray(protocol["root_starts_a"], dtype=float),
            max_nfev=int(protocol["max_nfev_per_start"]),
            numerical_root_tolerance=float(protocol["numerical_root_tolerance"]),
            jacobian_step_a=float(protocol["jacobian_step_a"]),
        )

    value, _, _ = ctx.cache.get_or_compute(
        "viscous_elastic_static_validation_v1",
        payload,
        producer,
        recompute=ctx.recompute,
    )
    return value


def _exact_comparison(
    ctx: PipelineContext,
    phase: np.ndarray,
    target_m: Sequence[float],
    *,
    key: str,
    positions_m: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    frequency_hz: float | None = None,
) -> Any:
    target = np.asarray(target_m, dtype=float)
    positions = ctx.positions_m if positions_m is None else np.asarray(positions_m, dtype=float)
    normal_values = (
        np.tile(np.array([0.0, 0.0, 1.0]), (len(positions), 1))
        if normals is None
        else np.asarray(normals, dtype=float)
    )
    protocol = _finite_ka_protocol(ctx)
    starts = np.asarray(protocol["root_starts_a"], dtype=float)
    payload = {
        "key": key,
        "phase": digest_array(np.asarray(phase, dtype=float)),
        "positions": digest_array(positions),
        "normals": digest_array(normal_values),
        "target": target.tolist(),
        "frequency_hz": float(frequency_hz or ctx.config.frequency_hz),
        "protocol": protocol,
    }

    def producer() -> Any:
        medium = Medium()
        sphere = ElasticSphere()
        field = _array_field(
            ctx,
            phase,
            positions_m=positions,
            normals=normal_values,
            frequency_hz=frequency_hz,
        )
        evaluator = PartialWaveForceEvaluator(
            float(frequency_hz or ctx.config.frequency_hz),
            field,
            medium=medium,
            sphere=sphere,
            numerics=_partial_wave_numerics(ctx),
        )
        return compare_gorkov_and_finite_ka(
            evaluator,
            target,
            initial_offsets_a=starts,
            search_half_width_a=float(protocol["search_half_width_a"]),
            max_nfev=int(protocol["max_nfev_per_start"]),
            numerical_root_tolerance=float(protocol["numerical_root_tolerance"]),
            jacobian_step_a=float(protocol["jacobian_step_a"]),
        )

    value, _, _ = ctx.cache.get_or_compute(
        "finite_ka_comparison", payload, producer, recompute=ctx.recompute
    )
    return value


def _restoring_root_class(resolved: bool, eigenvalues: np.ndarray) -> str:
    """Classify only the local symmetric static-restoring test.

    This label is deliberately narrower than dynamical stability.  Numerical
    step sensitivity is exported separately and is not silently promoted to a
    per-endpoint certificate.
    """

    if not resolved:
        return "unresolved"
    values = np.asarray(eigenvalues, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return "resolved-stiffness-indeterminate"
    return "resolved-restoring" if float(np.min(values)) > 0.0 else "resolved-nonrestoring"


def _validation_row(label: str, comparison: Any) -> dict[str, Any]:
    exact = comparison.finite_ka
    surrogate = comparison.gorkov
    exact_resolved = bool(exact.equilibrium.numerical_root_found)
    surrogate_resolved = bool(surrogate.equilibrium.numerical_root_found)
    exact_eigenvalues = np.asarray(exact.symmetric_stiffness_eigenvalues_n_m, dtype=float)
    exact_minimum = float(np.min(exact_eigenvalues)) if exact_resolved and np.all(np.isfinite(exact_eigenvalues)) else math.nan
    exact_class = _restoring_root_class(exact_resolved, exact_eigenvalues)
    return {
        "method": label,
        "finite_ka_root_found": exact.equilibrium.numerical_root_found,
        "finite_ka_root_class": exact_class,
        "finite_ka_locally_restoring": exact_class == "resolved-restoring",
        "finite_ka_displacement_a": exact.equilibrium.displacement_norm_a if exact_resolved else math.nan,
        "finite_ka_displacement_um": 1e6 * exact.equilibrium.displacement_norm_m if exact_resolved else math.nan,
        "finite_ka_candidate_displacement_a": exact.equilibrium.displacement_norm_a,
        "finite_ka_residual_scaled": exact.equilibrium.residual_scaled_norm,
        "finite_ka_stiffness_min_n_m": exact_minimum,
        "finite_ka_stiffness_mid_n_m": float(np.median(exact.symmetric_stiffness_eigenvalues_n_m)) if exact_resolved else math.nan,
        "finite_ka_stiffness_max_n_m": float(np.max(exact.symmetric_stiffness_eigenvalues_n_m)) if exact_resolved else math.nan,
        "gorkov_root_found": surrogate.equilibrium.numerical_root_found,
        "gorkov_displacement_a": surrogate.equilibrium.displacement_norm_a if surrogate_resolved else math.nan,
        "gorkov_candidate_displacement_a": surrogate.equilibrium.displacement_norm_a,
        "gorkov_stiffness_min_n_m": float(np.min(surrogate.symmetric_stiffness_eigenvalues_n_m)) if surrogate_resolved else math.nan,
        "model_equilibrium_difference_um": 1e6 * comparison.equilibrium_difference_norm_m if exact_resolved and surrogate_resolved else math.nan,
    }


def _static_validation_row(label: str, result: Any) -> dict[str, Any]:
    """Flatten every quantity needed to reproduce a benchmark marker."""

    equilibrium = result.equilibrium
    resolved = bool(equilibrium.numerical_root_found)
    eigenvalues = np.asarray(result.symmetric_stiffness_eigenvalues_n_m, dtype=float)
    minimum = float(np.min(eigenvalues)) if resolved and np.all(np.isfinite(eigenvalues)) else math.nan
    root_class = _restoring_root_class(resolved, eigenvalues)
    target_metadata = dict(result.metadata.get("target_force_evaluation", {}))
    equilibrium_metadata = dict(result.metadata.get("equilibrium_force_evaluation", {}))
    displacement = np.asarray(equilibrium.displacement_m, dtype=float)
    target_force = np.asarray(result.target_force_n, dtype=float)
    equilibrium_force = np.asarray(result.equilibrium_force_n, dtype=float)
    medium_metadata = dict(result.metadata.get("medium", {}))
    sphere_metadata = dict(result.metadata.get("sphere", {}))
    required_medium = {"density_kg_m3", "gravity_m_s2"}
    required_sphere = {"density_kg_m3", "volume_m3"}
    missing_medium = sorted(required_medium.difference(medium_metadata))
    missing_sphere = sorted(required_sphere.difference(sphere_metadata))
    if missing_medium or missing_sphere:
        raise ValueError(
            "StaticValidationResult lacks the validator provenance required for "
            "force decomposition: "
            f"missing medium={missing_medium}, sphere={missing_sphere}"
        )
    include_effective_gravity = bool(
        result.metadata.get("include_effective_gravity", False)
    )
    effective_gravity = np.zeros(3, dtype=float)
    if include_effective_gravity:
        effective_gravity[2] = -(
            float(sphere_metadata["density_kg_m3"])
            - float(medium_metadata["density_kg_m3"])
        ) * float(sphere_metadata["volume_m3"]) * float(
            medium_metadata["gravity_m_s2"]
        )
    target_radiation_force = target_force - effective_gravity
    equilibrium_radiation_force = equilibrium_force - effective_gravity
    stiffness = np.asarray(result.symmetric_stiffness_n_m, dtype=float)
    jacobian = np.asarray(result.force_jacobian_n_m, dtype=float)
    antisymmetric = np.asarray(result.antisymmetric_force_jacobian_n_m, dtype=float)
    candidate_point = np.asarray(equilibrium.equilibrium_m, dtype=float)
    selected_candidate = min(
        equilibrium.candidates,
        key=lambda candidate: float(
            np.linalg.norm(candidate.equilibrium_m - candidate_point)
            + np.linalg.norm(candidate.initial_offset_a - equilibrium.selected_start_offset_a)
        ),
    )
    finite_eigenvalues = bool(np.all(np.isfinite(eigenvalues)))
    resolved_array_json = lambda values: (
        json.dumps(np.asarray(values, dtype=float).tolist(), allow_nan=False) if resolved else "null"
    )
    return {
        "method": str(label),
        "finite_ka_root_found": resolved,
        "finite_ka_root_class": root_class,
        "finite_ka_locally_restoring": root_class == "resolved-restoring",
        "finite_ka_equilibrium_x_m": float(candidate_point[0]) if resolved else math.nan,
        "finite_ka_equilibrium_y_m": float(candidate_point[1]) if resolved else math.nan,
        "finite_ka_equilibrium_z_m": float(candidate_point[2]) if resolved else math.nan,
        "finite_ka_displacement_x_m": float(displacement[0]) if resolved else math.nan,
        "finite_ka_displacement_y_m": float(displacement[1]) if resolved else math.nan,
        "finite_ka_displacement_z_m": float(displacement[2]) if resolved else math.nan,
        "finite_ka_displacement_m": float(equilibrium.displacement_norm_m) if resolved else math.nan,
        "finite_ka_displacement_um": 1.0e6 * float(equilibrium.displacement_norm_m) if resolved else math.nan,
        "finite_ka_displacement_a": float(equilibrium.displacement_norm_a) if resolved else math.nan,
        "finite_ka_candidate_x_m": float(candidate_point[0]),
        "finite_ka_candidate_y_m": float(candidate_point[1]),
        "finite_ka_candidate_z_m": float(candidate_point[2]),
        "finite_ka_candidate_displacement_x_m": float(displacement[0]),
        "finite_ka_candidate_displacement_y_m": float(displacement[1]),
        "finite_ka_candidate_displacement_z_m": float(displacement[2]),
        "finite_ka_candidate_displacement_m": float(equilibrium.displacement_norm_m),
        "finite_ka_candidate_displacement_a": float(equilibrium.displacement_norm_a),
        "finite_ka_root_residual_scaled": float(equilibrium.residual_scaled_norm) if resolved else math.nan,
        "finite_ka_candidate_residual_scaled": float(equilibrium.residual_scaled_norm),
        "finite_ka_root_on_search_boundary": bool(equilibrium.on_search_boundary) if resolved else math.nan,
        "finite_ka_candidate_on_search_boundary": bool(equilibrium.on_search_boundary),
        "finite_ka_root_nfev": int(equilibrium.nfev) if resolved else math.nan,
        "finite_ka_candidate_nfev": int(equilibrium.nfev),
        "finite_ka_selected_optimizer_success": bool(selected_candidate.optimizer_success),
        "finite_ka_selected_optimizer_message": str(selected_candidate.message),
        "finite_ka_selected_start_a_json": json.dumps(
            np.asarray(equilibrium.selected_start_offset_a, dtype=float).tolist()
        ),
        "finite_ka_target_force_x_n": float(target_force[0]),
        "finite_ka_target_force_y_n": float(target_force[1]),
        "finite_ka_target_force_z_n": float(target_force[2]),
        "finite_ka_target_force_norm_n": float(np.linalg.norm(target_force)),
        "finite_ka_target_radiation_force_x_n": float(target_radiation_force[0]),
        "finite_ka_target_radiation_force_y_n": float(target_radiation_force[1]),
        "finite_ka_target_radiation_force_z_n": float(target_radiation_force[2]),
        "finite_ka_target_radiation_force_norm_n": float(
            np.linalg.norm(target_radiation_force)
        ),
        "finite_ka_effective_gravity_x_n": float(effective_gravity[0]),
        "finite_ka_effective_gravity_y_n": float(effective_gravity[1]),
        "finite_ka_effective_gravity_z_n": float(effective_gravity[2]),
        "finite_ka_effective_weight_n": float(np.linalg.norm(effective_gravity)),
        "finite_ka_equilibrium_force_x_n": float(equilibrium_force[0]) if resolved else math.nan,
        "finite_ka_equilibrium_force_y_n": float(equilibrium_force[1]) if resolved else math.nan,
        "finite_ka_equilibrium_force_z_n": float(equilibrium_force[2]) if resolved else math.nan,
        "finite_ka_equilibrium_radiation_force_x_n": float(equilibrium_radiation_force[0]) if resolved else math.nan,
        "finite_ka_equilibrium_radiation_force_y_n": float(equilibrium_radiation_force[1]) if resolved else math.nan,
        "finite_ka_equilibrium_radiation_force_z_n": float(equilibrium_radiation_force[2]) if resolved else math.nan,
        "finite_ka_candidate_force_x_n": float(equilibrium_force[0]),
        "finite_ka_candidate_force_y_n": float(equilibrium_force[1]),
        "finite_ka_candidate_force_z_n": float(equilibrium_force[2]),
        "finite_ka_stiffness_min_n_m": minimum,
        "finite_ka_stiffness_mid_n_m": float(np.median(eigenvalues)) if resolved and finite_eigenvalues else math.nan,
        "finite_ka_stiffness_max_n_m": float(np.max(eigenvalues)) if resolved and finite_eigenvalues else math.nan,
        "finite_ka_stiffness_eigenvalues_n_m_json": resolved_array_json(eigenvalues),
        "finite_ka_stiffness_matrix_n_m_json": resolved_array_json(stiffness),
        "finite_ka_force_jacobian_n_m_json": resolved_array_json(jacobian),
        "finite_ka_antisymmetric_jacobian_fro_n_m": float(np.linalg.norm(antisymmetric)) if resolved else math.nan,
        "finite_ka_candidate_stiffness_eigenvalues_n_m_json": json.dumps(eigenvalues.tolist(), allow_nan=False),
        "finite_ka_candidate_stiffness_matrix_n_m_json": json.dumps(stiffness.tolist(), allow_nan=False),
        "finite_ka_candidate_force_jacobian_n_m_json": json.dumps(jacobian.tolist(), allow_nan=False),
        "finite_ka_candidate_antisymmetric_jacobian_fro_n_m": float(np.linalg.norm(antisymmetric)),
        "finite_ka_projection_residual_target": float(target_metadata.get("projection_residual", math.nan)),
        "finite_ka_projection_residual_equilibrium": float(equilibrium_metadata.get("projection_residual", math.nan)) if resolved else math.nan,
        "finite_ka_projection_residual_candidate": float(equilibrium_metadata.get("projection_residual", math.nan)),
        "finite_ka_projection_condition": float(equilibrium_metadata.get("projection_condition", math.nan)),
        "finite_ka_model_name": str(result.model_name),
        "finite_ka_model_metadata_json": json.dumps(result.metadata, sort_keys=True, allow_nan=False),
    }


def _viscous_static_validation_row(label: str, result: Any) -> dict[str, Any]:
    """Flatten a viscous-sensitivity result without changing main-reference keys."""

    finite_named = _static_validation_row(label, result)
    return {
        (
            f"viscous_{name.removeprefix('finite_ka_')}"
            if name.startswith("finite_ka_")
            else name
        ): value
        for name, value in finite_named.items()
    }


def _register_viscous_elastic_model_tables(ctx: PipelineContext) -> None:
    """Register model parameters and deterministic regression diagnostics once."""

    if "viscous_elastic_model_protocol" in ctx.tables:
        return
    protocol = _viscous_elastic_protocol(ctx)
    medium = Medium()
    sphere = ElasticSphere()
    ctx.tables["viscous_elastic_model_protocol"] = pd.DataFrame(
        [
            {
                "model_name": protocol["model"],
                "model_role": protocol["model_role"],
                "validity": protocol["validity"],
                "frequency_hz": protocol["frequency_hz"],
                "ka": protocol["ka"],
                "host_density_kg_m3": medium.density_kg_m3,
                "host_sound_speed_m_s": medium.sound_speed_m_s,
                "host_dynamic_viscosity_pa_s": medium.dynamic_viscosity_pa_s,
                "bead_radius_m": sphere.radius_m,
                "bead_density_kg_m3": sphere.density_kg_m3,
                "bead_longitudinal_sound_speed_m_s": (
                    sphere.longitudinal_sound_speed_m_s
                ),
                "bead_shear_sound_speed_m_s": sphere.shear_sound_speed_m_s,
                "bead_bulk_modulus_pa": sphere.bulk_modulus_pa,
                "shear_boundary_layer_thickness_m": protocol[
                    "shear_boundary_layer_thickness_m"
                ],
                "shear_boundary_layer_to_radius": protocol[
                    "shear_boundary_layer_to_radius"
                ],
                "monopole_contrast_real": protocol["monopole_contrast_real"],
                "monopole_contrast_imag": protocol["monopole_contrast_imag"],
                "dipole_contrast_real": protocol["dipole_contrast_real"],
                "dipole_contrast_imag": protocol["dipole_contrast_imag"],
                "thermal_correction_included": False,
                "microstreaming_included": False,
                "finite_ka_viscous_hybridization": False,
                "excluded_physics_json": json.dumps(
                    protocol["excluded_physics"], sort_keys=True
                ),
                "references_json": json.dumps(protocol["references"]),
                "root_protocol_json": json.dumps(
                    {
                        key: protocol[key]
                        for key in (
                            "root_starts_a",
                            "search_half_width_a",
                            "max_nfev_per_start",
                            "numerical_root_tolerance",
                            "root_acceptance",
                            "resolved_root_selection",
                            "unresolved_candidate_selection",
                            "jacobian_step_a",
                        )
                    },
                    sort_keys=True,
                ),
                "viscous_numerics_json": json.dumps(
                    protocol["numerics"], sort_keys=True
                ),
                "complete_protocol_json": json.dumps(
                    protocol, sort_keys=True, allow_nan=False
                ),
            }
        ]
    )
    regression = viscous_rayleigh_inviscid_regression(
        medium=medium,
        sphere=sphere,
    )
    ctx.tables["viscous_elastic_inviscid_regression"] = pd.DataFrame(
        [
            {
                "regression_role": (
                    "deterministic formula diagnostic; not a manuscript "
                    "acceptance threshold"
                ),
                "frequency_hz": regression["frequency_hz"],
                "ka": regression["ka"],
                "near_inviscid_dynamic_viscosity_pa_s": regression[
                    "dynamic_viscosity_pa_s"
                ],
                "number_of_points": regression["number_of_points"],
                "viscous_dipole_contrast_real": regression[
                    "viscous_dipole_contrast"
                ]["real"],
                "viscous_dipole_contrast_imag": regression[
                    "viscous_dipole_contrast"
                ]["imag"],
                "inviscid_gorkov_dipole_contrast": regression[
                    "inviscid_gorkov_dipole_contrast"
                ],
                "dipole_contrast_absolute_error": regression[
                    "dipole_contrast_absolute_error"
                ],
                "maximum_relative_force_error": regression[
                    "maximum_relative_force_error"
                ],
                "aggregate_relative_force_error": regression[
                    "aggregate_relative_force_error"
                ],
                "pointwise_relative_force_errors_json": json.dumps(
                    regression["pointwise_relative_force_errors"]
                ),
                "viscous_radiation_forces_n_json": json.dumps(
                    regression["viscous_radiation_forces_n"]
                ),
                "gorkov_radiation_forces_n_json": json.dumps(
                    regression["gorkov_radiation_forces_n"]
                ),
            }
        ]
    )


def _root_candidate_rows(
    ctx: PipelineContext,
    endpoint_index: int,
    spec: Mapping[str, Any],
    result: Any,
) -> list[dict[str, Any]]:
    """Export every prescribed-start outcome used by the root selector."""

    equilibrium = result.equilibrium
    protocol = _finite_ka_protocol(ctx)
    tolerance = float(protocol["numerical_root_tolerance"])
    selected_start = np.asarray(equilibrium.selected_start_offset_a, dtype=float)
    selected_point = np.asarray(equilibrium.equilibrium_m, dtype=float)
    rows: list[dict[str, Any]] = []
    for start_index, candidate in enumerate(equilibrium.candidates):
        start = np.asarray(candidate.initial_offset_a, dtype=float)
        solution = np.asarray(candidate.solution_offset_a, dtype=float)
        point = np.asarray(candidate.equilibrium_m, dtype=float)
        residual = np.asarray(candidate.residual_force_n, dtype=float)
        accepted = bool(
            candidate.optimizer_success
            and not candidate.on_search_boundary
            and candidate.residual_scaled_norm <= tolerance
        )
        selected = bool(
            np.array_equal(start, selected_start)
            and np.allclose(point, selected_point, rtol=0.0, atol=1.0e-15)
        )
        rows.append(
            {
                "endpoint_index": int(endpoint_index),
                "method": str(spec["method"]),
                "pair_id": str(spec["pair_id"]),
                "seed": spec["seed"],
                "phase_content_id": digest_array(np.asarray(spec["phase"], dtype=float)),
                "start_index": int(start_index),
                "initial_offset_x_a": float(start[0]),
                "initial_offset_y_a": float(start[1]),
                "initial_offset_z_a": float(start[2]),
                "solution_offset_x_a": float(solution[0]),
                "solution_offset_y_a": float(solution[1]),
                "solution_offset_z_a": float(solution[2]),
                "solution_displacement_a": float(np.linalg.norm(solution)),
                "candidate_x_m": float(point[0]),
                "candidate_y_m": float(point[1]),
                "candidate_z_m": float(point[2]),
                "residual_force_x_n": float(residual[0]),
                "residual_force_y_n": float(residual[1]),
                "residual_force_z_n": float(residual[2]),
                "residual_scaled_norm": float(candidate.residual_scaled_norm),
                "optimizer_success": bool(candidate.optimizer_success),
                "on_search_boundary": bool(candidate.on_search_boundary),
                "accepted_numerical_root": accepted,
                "selected_by_protocol": selected,
                "nfev": int(candidate.nfev),
                "optimizer_message": str(candidate.message),
                "run_mode": ctx.config.mode,
            }
        )
    return rows


def _field_axes(
    axes: Sequence[plt.Axes],
    sections: Mapping[str, Any],
    target: np.ndarray,
    equilibrium: np.ndarray | None,
    *,
    titles: tuple[str, str] = ("Horizontal pressure", "Vertical pressure"),
    equilibrium_resolved: bool = True,
) -> None:
    norm = shared_power_norm([sections["xy"], sections["xz"]], lower=0.0, upper=99.5)
    xy = 1e3 * sections["xy_axis_m"]
    zz = 1e3 * sections["z_axis_m"]
    equilibrium_xy = None
    equilibrium_xz = None
    if equilibrium is not None and equilibrium_resolved:
        delta = 1e3 * (np.asarray(equilibrium) - target)
        equilibrium_xy = (float(delta[0]), float(delta[1]))
        equilibrium_xz = (float(delta[0]), float(delta[2]))
    field_panel(
        axes[0], sections["xy"], (xy[0], xy[-1], xy[0], xy[-1]), norm=norm,
        xlabel=r"$x-x^*$ (mm)", ylabel=r"$y-y^*$ (mm)", target_mm=(0.0, 0.0),
        equilibrium_mm=equilibrium_xy,
    )
    field_panel(
        axes[1], sections["xz"], (xy[0], xy[-1], zz[0], zz[-1]), norm=norm,
        xlabel=r"$x-x^*$ (mm)", ylabel=r"$z-z^*$ (mm)", target_mm=(0.0, 0.0),
        equilibrium_mm=equilibrium_xz,
    )
    set_panel_title(axes[0], titles[0])
    set_panel_title(axes[1], titles[1])
    if equilibrium is not None and not equilibrium_resolved:
        delta = 1e3 * (np.asarray(equilibrium) - target)
        axes[0].scatter(delta[0], delta[1], marker="^", facecolor="none", edgecolor="white", s=48, linewidth=1.2, zorder=25)
        axes[1].scatter(delta[0], delta[2], marker="^", facecolor="none", edgecolor="white", s=48, linewidth=1.2, zorder=25)
        axes[1].text(0.98, 0.02, "unresolved candidate", transform=axes[1].transAxes,
                     ha="right", va="bottom", fontsize=7.5, color="white")


_BRANCH_METHOD_STYLE: dict[str, dict[str, Any]] = {
    "Conventional": {"color": "#3F3F3F", "marker": "o", "short": "C", "lw": 1.00, "z": 3},
    "Force-Equilibrium": {"color": "#2C82C9", "marker": "s", "short": "FE", "lw": 1.55, "z": 5},
    "Regularized-Hybrid": {"color": "#E05A24", "marker": "^", "short": "RH", "lw": 1.15, "z": 4},
}


def _web_coordinate_lookup(table: pd.DataFrame, dimensions: Sequence[str]) -> dict[tuple[int, int], np.ndarray]:
    return {
        (int(row.ix), int(row.iy)): np.asarray([getattr(row, key) for key in dimensions], dtype=float)
        for row in table.itertuples(index=False)
    }


def _draw_physical_grid_3d(ax: Any, reference: pd.DataFrame, edge_table: pd.DataFrame, positions_m: np.ndarray) -> None:
    targets = reference.drop_duplicates(["ix", "iy"]).sort_values(["iy", "ix"])
    ax.scatter(
        1e3 * positions_m[:, 0], 1e3 * positions_m[:, 1], 1e3 * positions_m[:, 2],
        s=3.0, color="0.62", alpha=0.26, depthshade=False,
    )
    lookup = {
        (int(row.ix), int(row.iy)): 1e3 * np.asarray(
            [row.target_x_m, row.target_y_m, row.target_z_m], dtype=float
        )
        for row in targets.itertuples(index=False)
    }
    for edge in edge_table.itertuples(index=False):
        segment = np.stack([lookup[(int(edge.a_ix), int(edge.a_iy))], lookup[(int(edge.b_ix), int(edge.b_iy))]])
        ax.plot(*segment.T, color="#B03A2E", lw=1.55, alpha=0.92)
    points = np.stack(list(lookup.values()))
    ax.scatter(*points.T, s=24, color="#B03A2E", edgecolor="white", linewidth=0.35, depthshade=False)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=28, azim=-120)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.12, 1.0, 0.92))


def _draw_branch_sheet_3d(
    ax: Any,
    reference: pd.DataFrame,
    evolution: pd.DataFrame,
    edge_table: pd.DataFrame,
    component: str,
    *,
    highlight_target: tuple[int, int] | None = None,
    iteration: int | None = None,
) -> None:
    selected_iteration = int(evolution["iteration"].max()) if iteration is None else int(iteration)
    method_tables = {
        "Conventional": evolution[
            evolution["component"].eq(component) & evolution["iteration"].eq(selected_iteration)
        ],
        "Regularized-Hybrid": reference[
            reference["component"].eq(component) & reference["method"].eq("Regularized-Hybrid")
        ],
        "Force-Equilibrium": reference[
            reference["component"].eq(component) & reference["method"].eq("Force-Equilibrium")
        ],
    }
    # Put the nearly constant family-separation coordinate in depth, leaving
    # the two within-family coordinates horizontal and vertical.  This is the
    # camera grammar of the legible deposited web, rather than an edge-on view
    # of the sheet.
    dimensions = ("coordinate_1", "coordinate_3", "coordinate_2")
    for method in ("Conventional", "Regularized-Hybrid", "Force-Equilibrium"):
        table = method_tables[method]
        style = _BRANCH_METHOD_STYLE[method]
        lookup = _web_coordinate_lookup(table, dimensions)
        for edge in edge_table.itertuples(index=False):
            segment = np.stack(
                [lookup[(int(edge.a_ix), int(edge.a_iy))], lookup[(int(edge.b_ix), int(edge.b_iy))]]
            )
            ax.plot(
                *segment.T, color="white", lw=style["lw"] + 1.05, alpha=0.74,
                zorder=style["z"] - 1,
            )
            ax.plot(
                *segment.T, color=style["color"], lw=style["lw"], alpha=0.94,
                zorder=style["z"],
            )
        values = table.loc[:, dimensions].to_numpy(dtype=float)
        ax.scatter(
            *values.T,
            s=25 if method == "Force-Equilibrium" else 22,
            marker=style["marker"], facecolor="white", edgecolor=style["color"],
            linewidth=1.05, depthshade=False, zorder=style["z"] + 1,
        )
    if highlight_target is not None:
        fe_table = method_tables["Force-Equilibrium"]
        selected = fe_table[
            fe_table["ix"].eq(highlight_target[0]) & fe_table["iy"].eq(highlight_target[1])
        ]
        if len(selected) == 1:
            xyz = selected.loc[:, dimensions].to_numpy(dtype=float)[0]
            ax.scatter(*xyz, marker="*", s=145, color=TARGET_COLOR, edgecolor="white", linewidth=0.7, zorder=12)
    combined = pd.concat(list(method_tables.values()), ignore_index=True)
    for coordinate, setter in zip(dimensions, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        low, high = float(combined[coordinate].min()), float(combined[coordinate].max())
        padding = max(0.06 * (high - low), 0.004)
        setter(low - padding, high + padding)
    ax.set_xlabel("Solution 1", labelpad=4, fontsize=8.5)
    ax.set_ylabel("Family", labelpad=4, fontsize=8.5)
    ax.set_zlabel("Solution 2", labelpad=4, fontsize=8.5)
    ax.tick_params(labelsize=7.2, pad=1.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        formatter = axis.get_major_formatter()
        if hasattr(formatter, "set_useOffset"):
            formatter.set_useOffset(False)
        if hasattr(formatter, "set_scientific"):
            formatter.set_scientific(False)
    ax.set_title(component, pad=8, fontweight="semibold")
    ax.view_init(elev=24, azim=-63 if component == "B1" else -117)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.05, 1.0, 0.84))


def _branch_method_legend() -> list[mpl.lines.Line2D]:
    return [
        mpl.lines.Line2D(
            [0], [0], color=style["color"], lw=style["lw"], marker=style["marker"],
            markerfacecolor="white", markeredgecolor=style["color"], label=style["short"],
        )
        for style in _BRANCH_METHOD_STYLE.values()
    ]


def figure_1_integrated_solution_space(ctx: PipelineContext) -> mpl.figure.Figure:
    """Physical target chart -> FE web -> one command -> realized Vortex."""

    stem = "Figure_1_integrated_solution_space"
    if stem in ctx.figures:
        return ctx.figures[stem]
    phase = _fe_raw_phase(ctx, "B1")
    comparison = _exact_comparison(ctx, phase, MAIN_TARGET_M, key="fig1-rstar-B1")
    exact = comparison.finite_ka
    equilibrium = exact.equilibrium.equilibrium_m
    field = _array_field(ctx, phase)
    sections = _pressure_sections(
        field, MAIN_TARGET_M, half_xy_m=0.075, half_z_m=0.050,
        samples=141 if ctx.config.full else 101,
    )
    local_sections = _pressure_sections(field, MAIN_TARGET_M, samples=101)
    branch = _rigorous_branch_evolution(ctx)
    ctx.tables["main_single_trap_finite_ka"] = pd.DataFrame(
        [_validation_row("FE B1 at r*", comparison)]
    )

    fig = plt.figure(figsize=(16.2, 10.0))
    outer = fig.add_gridspec(2, 3, height_ratios=(1.22, 0.90), hspace=0.28, wspace=0.22)
    ax_domain = fig.add_subplot(outer[0, 0], projection="3d")
    ax_b1 = fig.add_subplot(outer[0, 1], projection="3d")
    ax_b2 = fig.add_subplot(outer[0, 2], projection="3d")
    ax_phase = fig.add_subplot(outer[1, 0])
    ax_xy = fig.add_subplot(outer[1, 1])
    ax_xz = fig.add_subplot(outer[1, 2])

    _draw_physical_grid_3d(ax_domain, branch["reference"], ctx.branch_bank.edge_table, ctx.positions_m)
    set_panel_title(ax_domain, "Physical target grid")
    panel_label(ax_domain, "a")
    root_ix, root_iy = ctx.branch_bank.root_target
    _draw_branch_sheet_3d(
        ax_b1, branch["reference"], branch["evolution"], ctx.branch_bank.edge_table,
        "B1", highlight_target=(root_ix, root_iy),
    )
    _draw_branch_sheet_3d(
        ax_b2, branch["reference"], branch["evolution"], ctx.branch_bank.edge_table,
        "B2",
    )
    ax_b2.legend(handles=_branch_method_legend(), loc="upper right", fontsize=8, framealpha=0.94)
    panel_label(ax_b1, "b")
    panel_label(ax_b2, "c")

    phase_image = phase_grid_panel(ax_phase, phase, side=16)
    set_panel_title(ax_phase, "Selected FE command at r*")
    fig.colorbar(phase_image, ax=ax_phase, fraction=0.046, pad=0.03, label="Phase (rad)")
    panel_label(ax_phase, "d")
    _field_axes((ax_xy, ax_xz), sections, MAIN_TARGET_M, equilibrium,
                titles=("Vortex, z = z*", "Vortex, y = y*"),
                equilibrium_resolved=exact.equilibrium.numerical_root_found)
    panel_label(ax_xy, "e")
    panel_label(ax_xz, "f")
    # The full-domain field preserves the old notebook's morphology view; a
    # local inset carries the target-versus-equilibrium placement.
    local_norm = shared_power_norm([local_sections["xy"], local_sections["xz"]], lower=0.0, upper=99.5)
    delta = 1e3 * (np.asarray(equilibrium) - MAIN_TARGET_M)
    for host, key, vertical in ((ax_xy, "xy", False), (ax_xz, "xz", True)):
        inset = host.inset_axes([0.58, 0.055, 0.38, 0.38])
        axis_x = 1e3 * local_sections["xy_axis_m"]
        axis_y = 1e3 * (local_sections["z_axis_m"] if vertical else local_sections["xy_axis_m"])
        field_panel(
            inset, local_sections[key], (axis_x[0], axis_x[-1], axis_y[0], axis_y[-1]),
            norm=local_norm, xlabel="", ylabel="", target_mm=(0.0, 0.0),
            equilibrium_mm=(float(delta[0]), float(delta[2] if vertical else delta[1]))
            if exact.equilibrium.numerical_root_found else None,
        )
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_title("target neighborhood", fontsize=7.5)
    fig.suptitle(
        "Force-equilibrium optimization exposes connected solution families for continuous target placement",
        fontsize=14.2, fontweight="semibold", y=0.985,
    )
    return _save_figure(ctx, fig, stem, tight=False)


def _terminal_equivalence_table(ctx: PipelineContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final = ctx.conventional_evolution[
        ctx.conventional_evolution["iteration"].eq(int(np.max(ctx.artifact.checkpoints)))
    ]
    for row in final.itertuples(index=False):
        rows.append(
            {
                "component": row.component,
                "target_id": row.target_id,
                "projective_similarity_C_to_FE": row.similarity_to_target_fe,
                "projector_distance_C_to_FE": row.distance_to_target_fe,
            }
        )
    return pd.DataFrame(rows)


def _component_view_2d(
    reference: pd.DataFrame,
    evolution: pd.DataFrame,
    component: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return one display-only 2-D view fitted once from FE+RH landmarks.

    Component identity and all quantitative distances have already been
    decided in the full projective state.  This PCA only chooses a readable
    camera through the fixed three-dimensional FE+RH landmark frame; no
    Conventional checkpoint enters the fit.
    """

    dimensions = ["coordinate_1", "coordinate_2", "coordinate_3"]
    reference_component = reference[reference["component"].eq(component)].copy()
    evolution_component = evolution[evolution["component"].eq(component)].copy()
    landmarks = reference_component[dimensions].to_numpy(dtype=float)
    center = np.mean(landmarks, axis=0)
    _, _, right = np.linalg.svd(landmarks - center, full_matrices=False)
    basis = right[:2].T
    reference_component[["view_1", "view_2"]] = (landmarks - center) @ basis
    evolution_component[["view_1", "view_2"]] = (
        evolution_component[dimensions].to_numpy(dtype=float) - center
    ) @ basis
    return reference_component, evolution_component, center, basis


def _draw_web_2d(
    ax: plt.Axes,
    reference: pd.DataFrame,
    evolution: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    iteration: int,
) -> None:
    """Draw one complete target-indexed web at a recorded checkpoint."""

    for method in ("Regularized-Hybrid", "Force-Equilibrium"):
        table = reference[reference["method"].eq(method)]
        style = _BRANCH_METHOD_STYLE[method]
        lookup = _web_coordinate_lookup(table, ("view_1", "view_2"))
        for edge in edge_table.itertuples(index=False):
            segment = np.stack(
                [lookup[(int(edge.a_ix), int(edge.a_iy))], lookup[(int(edge.b_ix), int(edge.b_iy))]]
            )
            ax.plot(
                segment[:, 0], segment[:, 1], color=style["color"],
                lw=0.72, alpha=0.28, zorder=1,
            )
        values = table[["view_1", "view_2"]].to_numpy(dtype=float)
        ax.scatter(
            values[:, 0], values[:, 1], s=10, marker=style["marker"],
            facecolor="white", edgecolor=style["color"], linewidth=0.55,
            alpha=0.62, zorder=2,
        )

    conventional = evolution[evolution["iteration"].eq(int(iteration))]
    lookup = _web_coordinate_lookup(conventional, ("view_1", "view_2"))
    style = _BRANCH_METHOD_STYLE["Conventional"]
    for edge in edge_table.itertuples(index=False):
        segment = np.stack(
            [lookup[(int(edge.a_ix), int(edge.a_iy))], lookup[(int(edge.b_ix), int(edge.b_iy))]]
        )
        ax.plot(
            segment[:, 0], segment[:, 1], color="white", lw=2.15,
            alpha=0.82, zorder=3,
        )
        ax.plot(
            segment[:, 0], segment[:, 1], color=style["color"], lw=1.15,
            alpha=0.94, zorder=4,
        )
    values = conventional[["view_1", "view_2"]].to_numpy(dtype=float)
    ax.scatter(
        values[:, 0], values[:, 1], s=18, marker=style["marker"],
        facecolor="white", edgecolor=style["color"], linewidth=0.85,
        zorder=5,
    )
    # The fixed 2-D camera has a much smaller second coordinate range.  The
    # original manuscript intentionally used the full panel height so that all
    # physical neighbor edges remain readable; numerical axes are not used for
    # metric claims here.
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.82")


def _all_conventional_to_fe_evolution(ctx: PipelineContext) -> dict[str, Any]:
    """Audit every Conventional trajectory against one fixed same-target FE component.

    Component identity is selected once from the final recorded state and then
    held fixed at every earlier checkpoint.  The 491-run audit is independent
    of the 98 terminal-medoid trajectories selected only to keep the web panels
    legible.
    """

    memo_key = "all_conventional_to_fe_evolution"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    states = ctx.artifact.trajectory_states()
    table = ctx.artifact.trajectory_table.reset_index(drop=True)
    checkpoints = np.asarray(ctx.artifact.checkpoints, dtype=int)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for trajectory_index, source in table.iterrows():
        ix, iy = int(source["ix"]), int(source["iy"])
        references = np.stack(
            [ctx.branch_bank.target_medoid(ix, iy, component) for component in COMPONENTS]
        )
        distance = _projective_distance_between(states[trajectory_index], references)
        nearest_index = np.argmin(distance, axis=1)
        final_component_index = int(nearest_index[-1])
        final_component = COMPONENTS[final_component_index]
        matched_distance = distance[:, final_component_index]
        sorted_terminal = np.sort(distance[-1])
        assignment_margin = float(sorted_terminal[1] - sorted_terminal[0])
        branch_switched = bool(np.any(nearest_index != final_component_index))
        final_actual_iteration = int(
            source.get("final_budget_state_iteration", source.get("nit", checkpoints[-1]))
        )
        early_terminal = bool(
            source.get("final_state_is_early_terminal", final_actual_iteration < checkpoints[-1])
        )
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            is_last = checkpoint_index == len(checkpoints) - 1
            d_value = float(matched_distance[checkpoint_index])
            rows.append(
                {
                    **source.to_dict(),
                    "trajectory_index": int(trajectory_index),
                    "component": final_component,
                    "nearest_component_at_checkpoint": COMPONENTS[int(nearest_index[checkpoint_index])],
                    "iteration_budget": int(checkpoint),
                    "actual_state_iteration": final_actual_iteration if is_last else int(checkpoint),
                    "is_final_budget_state": bool(is_last),
                    "is_early_terminal": bool(is_last and early_terminal),
                    "distance_to_matched_fe": d_value,
                    "similarity_to_matched_fe": float(math.sqrt(max(0.0, 1.0 - d_value**2))),
                    "projector_overlap_to_matched_fe": float(max(0.0, 1.0 - d_value**2)),
                    "component_assignment_margin": assignment_margin,
                    "branch_switched": branch_switched,
                }
            )
        audit_rows.append(
            {
                "trajectory_index": int(trajectory_index),
                "target_id": source["target_id"],
                "ix": ix,
                "iy": iy,
                "component": final_component,
                "branch_switched": branch_switched,
                "component_assignment_margin": assignment_margin,
                "final_actual_iteration": final_actual_iteration,
                "is_early_terminal": early_terminal,
            }
        )
    evolution = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    final = evolution[evolution["is_final_budget_state"]].copy()
    grouping = ["ix", "iy", "component"]
    objective_rank = final.groupby(grouping)["objective"].rank(
        method="average", ascending=True
    )
    group_size = final.groupby(grouping)["objective"].transform("size").astype(float)
    final["objective_quality"] = 1.0 - (objective_rank - 1.0) / np.maximum(
        group_size - 1.0, 1.0
    )
    quality_rank = final["objective_quality"].rank(method="average")
    similarity_rank = final["similarity_to_matched_fe"].rank(method="average")
    rank_correlation = float(quality_rank.corr(similarity_rank))
    value = {
        "evolution": evolution,
        "final": final,
        "audit": audit,
        "rank_correlation": rank_correlation,
    }
    ctx.memo[memo_key] = value
    return value


def figure_2_convergence_and_evolution(ctx: PipelineContext) -> mpl.figure.Figure:
    """Place quantitative C-to-FE evidence beside the matching web evolution."""

    stem = "Figure_2_convergence_and_evolution"
    if stem in ctx.figures:
        return ctx.figures[stem]
    fe_rows = ctx.artifact.anchor_table[
        ctx.artifact.anchor_table["method"].eq("Force-Equilibrium")
    ].copy()
    c_rows = ctx.artifact.trajectory_table.copy()
    common_columns = [
        "method", "target_id", "replicate", "runtime_s", "nit", "nfev",
        "phase_gradient_rms", "objective",
    ]
    for frame in (fe_rows, c_rows):
        for column in common_columns:
            if column not in frame:
                frame[column] = np.nan
    anchor = pd.concat([c_rows[common_columns], fe_rows[common_columns]], ignore_index=True)
    branch = _rigorous_branch_evolution(ctx)
    all_run = _all_conventional_to_fe_evolution(ctx)
    evolution_metric = all_run["evolution"]
    final_endpoints = all_run["final"]
    audit = all_run["audit"]
    checkpoints = [int(value) for value in ctx.artifact.checkpoints]
    final_iteration = checkpoints[-1]
    equivalence = branch["evolution"][
        branch["evolution"]["iteration"].eq(final_iteration)
    ][
        [
            "component", "target_id", "ix", "iy", "selected_trajectory_index",
            "distance_to_target_fe_rh", "distance_to_target_fe",
        ]
    ].copy()
    ctx.tables["single_trap_timing_and_stationarity"] = anchor[
        ["method", "target_id", "replicate", "runtime_s", "nit", "nfev", "phase_gradient_rms", "objective"]
    ].copy()
    ctx.tables["terminal_C_to_FE_RH_projective_distance"] = equivalence
    ctx.tables["all_conventional_to_fe_evolution"] = evolution_metric[
        [
            "trajectory_index", "target_id", "ix", "iy", "component",
            "iteration_budget", "actual_state_iteration", "is_early_terminal",
            "distance_to_matched_fe", "similarity_to_matched_fe",
            "nearest_component_at_checkpoint", "branch_switched",
        ]
    ].copy()
    ctx.tables["all_conventional_to_fe_audit"] = audit.copy()
    ctx.tables["terminal_conventional_objective_fe_correlation"] = final_endpoints[
        [
            "trajectory_index", "target_id", "ix", "iy", "component", "objective",
            "objective_quality", "distance_to_matched_fe", "similarity_to_matched_fe",
            "component_assignment_margin",
        ]
    ].copy()

    summary_rows: list[dict[str, Any]] = []
    monotone_rows: list[dict[str, Any]] = []
    for component, component_data in evolution_metric.groupby("component", sort=True):
        pivot = component_data.pivot(
            index="trajectory_index", columns="iteration_budget", values="distance_to_matched_fe"
        ).reindex(columns=checkpoints)
        differences = np.diff(pivot.to_numpy(dtype=float), axis=1)
        monotone_rows.append(
            {
                "component": component,
                "trajectory_count": int(len(pivot)),
                "decreases_at_every_checkpoint_count": int(np.sum(np.all(differences < 0.0, axis=1))),
                "decreases_at_every_checkpoint_fraction": float(np.mean(np.all(differences < 0.0, axis=1))),
            }
        )
        for iteration, stage in component_data.groupby("iteration_budget", sort=True):
            values = stage["distance_to_matched_fe"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "component": component,
                    "iteration_budget": int(iteration),
                    "n": int(len(values)),
                    "q25_distance_to_matched_fe": float(np.quantile(values, 0.25)),
                    "median_distance_to_matched_fe": float(np.median(values)),
                    "q75_distance_to_matched_fe": float(np.quantile(values, 0.75)),
                }
            )
    distance_summary = pd.DataFrame(summary_rows)
    monotone_summary = pd.DataFrame(monotone_rows)
    ctx.tables["all_conventional_to_fe_distance_summary"] = distance_summary
    ctx.tables["all_conventional_to_fe_monotonicity"] = monotone_summary
    ctx.tables["terminal_conventional_objective_fe_correlation_summary"] = pd.DataFrame(
        [
            {
                "trajectory_count": int(len(final_endpoints)),
                "conditioned_objective_quality_vs_raw_fe_similarity_spearman": all_run["rank_correlation"],
                "branch_switch_count": int(audit["branch_switched"].sum()),
                "early_terminal_count": int(audit["is_early_terminal"].sum()),
                "minimum_component_assignment_margin": float(audit["component_assignment_margin"].min()),
                "median_component_assignment_margin": float(audit["component_assignment_margin"].median()),
            }
        ]
    )

    fig = plt.figure(figsize=(16.4, 8.35))
    gs = fig.add_gridspec(
        2, 4, width_ratios=(0.92, 1.0, 1.0, 1.0),
        hspace=0.27, wspace=0.16,
    )
    x_positions = np.arange(len(checkpoints), dtype=float)
    ax_distance = fig.add_subplot(gs[0, 0])
    for component in COMPONENTS:
        component_data = evolution_metric[evolution_metric["component"].eq(component)]
        pivot = component_data.pivot(
            index="trajectory_index", columns="iteration_budget", values="distance_to_matched_fe"
        ).reindex(columns=checkpoints)
        for values in pivot.to_numpy(dtype=float):
            ax_distance.plot(x_positions, values, color=COLORS[component], lw=0.45, alpha=0.035)
        plotted = distance_summary[distance_summary["component"].eq(component)].set_index(
            "iteration_budget"
        ).reindex(checkpoints)
        median = plotted["median_distance_to_matched_fe"].to_numpy(dtype=float)
        lower = plotted["q25_distance_to_matched_fe"].to_numpy(dtype=float)
        upper = plotted["q75_distance_to_matched_fe"].to_numpy(dtype=float)
        ax_distance.fill_between(
            x_positions, lower, upper, color=COLORS[component], alpha=0.16, linewidth=0.0
        )
        ax_distance.plot(
            x_positions, median, "o-", color=COLORS[component], lw=1.7, ms=4.2,
            label=component,
        )
    monotone_total = int(monotone_summary["decreases_at_every_checkpoint_count"].sum())
    trajectory_total = int(monotone_summary["trajectory_count"].sum())
    switch_count = int(audit["branch_switched"].sum())
    ax_distance.set_xticks(x_positions, ["500", "2,000", "final\n≤10,000 budget"])
    ax_distance.set_ylabel("Projector distance\nto matched FE")
    ax_distance.text(
        0.04, 0.96,
        f"{monotone_total}/{trajectory_total} improve at both steps\n{switch_count}/{trajectory_total} component switches",
        transform=ax_distance.transAxes, fontsize=7.4, va="top",
        bbox=dict(facecolor="white", alpha=0.84, edgecolor="0.82", boxstyle="round,pad=0.25"),
    )
    ax_distance.legend(frameon=False, fontsize=7.8, loc="upper right")
    set_panel_title(ax_distance, "All Conventional runs approach FE")
    panel_label(ax_distance, "a")

    ax_correlation = fig.add_subplot(gs[1, 0])
    for component in COMPONENTS:
        selected = final_endpoints[final_endpoints["component"].eq(component)]
        ax_correlation.scatter(
            selected["objective_quality"], selected["similarity_to_matched_fe"],
            s=11, color=COLORS[component], alpha=0.22, edgecolor="none", rasterized=True,
        )
    bin_edges = np.linspace(0.0, 1.0, 6)
    bin_index = pd.cut(
        final_endpoints["objective_quality"], bin_edges, include_lowest=True, labels=False
    )
    binned_x: list[float] = []
    binned_median: list[float] = []
    binned_lower: list[float] = []
    binned_upper: list[float] = []
    for bin_value in range(len(bin_edges) - 1):
        values = final_endpoints.loc[
            bin_index.eq(bin_value), "similarity_to_matched_fe"
        ].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        binned_x.append(float((bin_edges[bin_value] + bin_edges[bin_value + 1]) / 2.0))
        binned_lower.append(float(np.quantile(values, 0.25)))
        binned_median.append(float(np.median(values)))
        binned_upper.append(float(np.quantile(values, 0.75)))
    ax_correlation.fill_between(
        binned_x, binned_lower, binned_upper, color="0.20", alpha=0.12, linewidth=0.0,
    )
    ax_correlation.plot(
        binned_x, binned_median, "o-", color="0.12", lw=1.45, ms=3.8,
        label="binned median",
    )
    ax_correlation.set_xlim(-0.03, 1.03)
    similarity_floor = float(final_endpoints["similarity_to_matched_fe"].min())
    ax_correlation.set_ylim(max(0.0, similarity_floor - 0.008), 1.002)
    ax_correlation.set_xlabel("Conventional objective quality\nwithin target and component")
    ax_correlation.set_ylabel("Terminal phase similarity\nto matched FE")
    ax_correlation.text(
        0.04, 0.96,
        rf"conditioned-quality Spearman $\rho$ = {all_run['rank_correlation']:.2f}",
        transform=ax_correlation.transAxes, fontsize=7.7, va="top",
        bbox=dict(facecolor="white", alpha=0.84, edgecolor="0.82", boxstyle="round,pad=0.25"),
    )
    set_panel_title(ax_correlation, "Better Conventional minima lie closer to FE")
    panel_label(ax_correlation, "b")

    checkpoint_titles = (
        "500 iterations", "2,000 iterations", "Final recorded state\n(≤10,000-iteration budget)"
    )
    panel_names = iter(("c", "d", "e", "f", "g", "h"))
    for row_index, component in enumerate(COMPONENTS):
        reference_view, evolution_view, _, _ = _component_view_2d(
            branch["reference"], branch["evolution"], component
        )
        combined = pd.concat(
            [reference_view[["view_1", "view_2"]], evolution_view[["view_1", "view_2"]]],
            ignore_index=True,
        )
        xlow, xhigh = combined["view_1"].min(), combined["view_1"].max()
        ylow, yhigh = combined["view_2"].min(), combined["view_2"].max()
        xpad = max(0.045 * float(xhigh - xlow), 0.002)
        ypad = max(0.045 * float(yhigh - ylow), 0.002)
        for column_index, checkpoint in enumerate(checkpoints):
            ax = fig.add_subplot(gs[row_index, column_index + 1])
            _draw_web_2d(
                ax, reference_view, evolution_view, ctx.branch_bank.edge_table,
                iteration=checkpoint,
            )
            ax.set_xlim(float(xlow - xpad), float(xhigh + xpad))
            ax.set_ylim(float(ylow - ypad), float(yhigh + ypad))
            if row_index == 0:
                set_panel_title(ax, checkpoint_titles[column_index])
            if column_index == 0:
                ax.text(
                    -0.08, 0.5, component, transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=10.5,
                    fontweight="semibold",
                )
            panel_label(ax, next(panel_names))
            if row_index == 0 and column_index == len(checkpoints) - 1:
                ax.legend(
                    handles=_branch_method_legend(), ncol=3, frameon=True,
                    fontsize=7.5, loc="upper right", borderpad=0.35,
                    handlelength=1.4, columnspacing=0.8,
                )
    fig.suptitle(
        "Conventional approaches FE in full-dimensional phase space and in the fixed FE/RH branch frame",
        fontsize=14.2, fontweight="semibold", y=0.985,
    )
    fig.text(
        0.52, 0.018,
        "Quantitative panels use all 491 Conventional runs; web panels show one terminal-medoid trajectory per target and component. Conventional is excluded from the frame fit.",
        ha="center", va="bottom", fontsize=7.2, color="0.30",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.105)
    return _save_figure(ctx, fig, stem, tight=False)


def _objective_at(ctx: PipelineContext, method: str, target: np.ndarray = MAIN_TARGET_M) -> SingleTargetObjective:
    condition = Condition(
        label=f"canonical-{method}", positions=ctx.positions_m,
        target_m=tuple(float(v) for v in target), frequency_hz=ctx.config.frequency_hz,
        curvature_weight=method_spec(method).curvature_weight,
        normals=ctx.normals,
    )
    return SingleTargetObjective(condition, method)


def _decision_endpoint_run(ctx: PipelineContext, method: str) -> Any:
    """Reproduce one optimizer-local endpoint with its accepted-step history."""

    # Import locally because the fixed-Single branch module imports this module's objective
    # helpers.  The historical alpha=9 branch bank is initialization provenance
    # only; production representative selection uses the continued alpha=10
    # branches.
    from .alpha1000_branch import fixed_single_branch_pack

    branch = fixed_single_branch_pack(ctx)
    root_ix, root_iy = branch["bank"].root_target
    if method == "Force-Equilibrium":
        anchors = branch["anchors"]
        matched = anchors[
            anchors["component"].eq("B1")
            & anchors["ix"].eq(root_ix)
            & anchors["iy"].eq(root_iy)
        ]
        if len(matched) != 1:
            raise RuntimeError(
                "Figure 3 requires one fixed-alpha=10 FE B1 anchor at the root target"
            )
        anchor_index = int(matched.index[0])
        result = branch["anchor_results"][anchor_index]
        objective = _objective_at(ctx, method)
        if not np.isclose(objective.method.alpha_per_m, 10.0, rtol=0.0, atol=0.0):
            raise RuntimeError("Figure 3 FE endpoint is not the fixed alpha=10 objective")
        if not np.isfinite(float(result.gradient_norm)):
            raise RuntimeError("Figure 3 FE branch endpoint has a non-finite terminal gradient")
        return result

    selected = branch["evolution"][
        branch["evolution"]["component"].eq("B1")
        & branch["evolution"]["ix"].eq(root_ix)
        & branch["evolution"]["iy"].eq(root_iy)
    ]
    trajectory_index = int(selected.iloc[0]["selected_trajectory_index"])
    seed = int(ctx.artifact.trajectory_table.iloc[trajectory_index]["seed"])
    objective = _objective_at(ctx, method)
    maxiter = (
        int(ctx.config.conventional_maxiter)
        if method == "Conventional"
        else int(ctx.config.fe_maxiter)
    )
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, objective.n_transducers)
    payload = {
        "contract": "native-terminal-gradient-and-last-accepted-step-v2-alpha-keyed",
        "method": method,
        "objective_method": asdict(objective.method),
        "target_m": MAIN_TARGET_M.tolist(),
        "seed": seed,
        "initial_phase": digest_array(initial),
        "maxiter": maxiter,
        "gtol": ctx.config.gtol,
        "array": digest_array(ctx.positions_m),
    }
    result, _, _ = ctx.cache.get_or_compute(
        "decision_endpoint_run",
        payload,
        lambda: solve_corrected_gorkov_fe(
            objective, initial, maxiter=maxiter, gtol=ctx.config.gtol
        ),
        recompute=ctx.recompute,
    )
    return result


def _native_local_directions(
    objective: SingleTargetObjective,
    result: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Old-notebook contract: -terminal gradient and last accepted step."""

    phase = np.asarray(result.phase_rad, dtype=float)
    _, gradient = objective.full_fun_grad(phase)
    gradient = np.asarray(gradient, dtype=float)
    u = -gradient / max(float(np.linalg.norm(gradient)), 1.0e-30)
    history = np.asarray(result.history_phase_rad, dtype=float)
    if history.ndim != 2 or history.shape[0] < 2:
        raise RuntimeError("decision-space reproduction requires two accepted phase states")
    previous, current = history[-2], history[-1]
    gauge = np.angle(np.vdot(np.exp(1j * previous), np.exp(1j * current)))
    step = np.angle(np.exp(1j * current) * np.exp(-1j * gauge) * np.exp(-1j * previous))
    step -= u * float(np.dot(step, u))
    step_norm = float(np.linalg.norm(step))
    if step_norm <= 1.0e-14:
        # Search backward only within the same uninterrupted accepted history.
        for prior in history[-3::-1]:
            gauge = np.angle(np.vdot(np.exp(1j * prior), np.exp(1j * current)))
            step = np.angle(np.exp(1j * current) * np.exp(-1j * gauge) * np.exp(-1j * prior))
            step -= u * float(np.dot(step, u))
            step_norm = float(np.linalg.norm(step))
            if step_norm > 1.0e-14:
                break
    if step_norm <= 1.0e-14:
        raise RuntimeError("accepted optimizer history does not define a transverse direction")
    v = step / step_norm
    return u, v


def _native_relative_sections(
    objective: SingleTargetObjective,
    result: Any,
    directions: tuple[np.ndarray, np.ndarray],
    *,
    line_points: int,
    plane_points: int,
) -> dict[str, np.ndarray]:
    phase = np.asarray(result.phase_rad, dtype=float)
    d1, d2 = directions
    # This is an L2-normalized direction exactly as in the original notebook.
    # It is not an RMS displacement and it is never a C-to-FE chord.
    t = np.linspace(-0.30, 0.30, line_points)
    line = np.array([objective.full_value(phase + value * d1) for value in t])
    q = np.linspace(-0.30, 0.30, plane_points)
    plane = np.empty((plane_points, plane_points), dtype=float)
    for j, value_2 in enumerate(q):
        for i, value_1 in enumerate(q):
            plane[j, i] = objective.full_value(phase + value_1 * d1 + value_2 * d2)
    line_relative = line - np.nanmin(line)
    plane_relative = plane - np.nanmin(plane)
    return {"t": t, "line": line_relative, "q": q, "plane": plane_relative}


def figure_3_decision_geometry(ctx: PipelineContext) -> mpl.figure.Figure:
    """Native endpoint-local sections explain the speed separation in Figure 2."""

    stem = "Figure_3_decision_geometry"
    if stem in ctx.figures:
        return ctx.figures[stem]
    objectives = {
        "Conventional": _objective_at(ctx, "Conventional"),
        "FE": _objective_at(ctx, "Force-Equilibrium"),
    }
    runs = {
        "Conventional": _decision_endpoint_run(ctx, "Conventional"),
        "FE": _decision_endpoint_run(ctx, "Force-Equilibrium"),
    }
    directions = {
        method: _native_local_directions(objectives[method], runs[method])
        for method in ("Conventional", "FE")
    }
    c_sections = _native_relative_sections(
        objectives["Conventional"], runs["Conventional"], directions["Conventional"],
        line_points=101, plane_points=51,
    )
    fe_sections = _native_relative_sections(
        objectives["FE"], runs["FE"], directions["FE"],
        line_points=101, plane_points=51,
    )
    ctx.memo["decision_sections"] = {"Conventional": c_sections, "FE": fe_sections}
    rows: list[dict[str, Any]] = []
    for method, data in ctx.memo["decision_sections"].items():
        center = len(data["t"]) // 2
        spacing = float(data["t"][1] - data["t"][0])
        left_slope = float((data["line"][center] - data["line"][center - 1]) / spacing)
        right_slope = float((data["line"][center + 1] - data["line"][center]) / spacing)
        rows.append(
            {
                "method": method,
                "endpoint_objective": float(runs[method].objective),
                "terminal_gradient_norm": float(runs[method].gradient_norm),
                "iterations": int(runs[method].iterations),
                "left_one_sided_slope": left_slope,
                "right_one_sided_slope": right_slope,
                "slope_jump": right_slope - left_slope,
                "direction_definition": "u=-terminal gradient; v=last accepted step orthogonalized to u",
                "half_range_l2_rad": 0.30,
            }
        )
    ctx.tables["local_decision_section_summary"] = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0), gridspec_kw={"height_ratios": (0.72, 1.0)})
    for column, (method, data, color) in enumerate(
        (("Conventional", c_sections, COLORS["Conventional"]), ("FE", fe_sections, COLORS["FE"]))
    ):
        axes[0, column].plot(data["t"], data["line"], color=color, lw=2.2)
        axes[0, column].scatter([0.0], [data["line"][len(data["line"]) // 2]], marker="x", color="crimson", s=45, zorder=5)
        axes[0, column].set_xlabel(r"Local coordinate $t$ (rad; $\|u\|_2=1$)")
        axes[0, column].set_ylabel("Objective above local minimum")
        set_panel_title(axes[0, column], f"{method}: native terminal section")
        finite = data["plane"][np.isfinite(data["plane"])]
        vmax = max(float(np.quantile(finite, 0.985)), np.finfo(float).tiny)
        image = axes[1, column].imshow(
            np.clip(data["plane"], 0.0, vmax), origin="lower",
            extent=(data["q"][0], data["q"][-1], data["q"][0], data["q"][-1]),
            cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto",
            interpolation="bilinear", resample=True,
        )
        axes[1, column].scatter(0, 0, marker="x", color="crimson", s=46, linewidth=1.5)
        axes[1, column].set_xlabel(r"$a$ along $u=-\nabla J/\|\nabla J\|_2$ (rad)")
        axes[1, column].set_ylabel(r"$b$ along orthogonalized last step (rad)")
        set_panel_title(axes[1, column], f"{method}: native 2-D section")
        colorbar = fig.colorbar(image, ax=axes[1, column], fraction=0.046, pad=0.035)
        colorbar.set_label("Objective above panel minimum")
        panel_label(axes[0, column], "a" if column == 0 else "b")
        panel_label(axes[1, column], "c" if column == 0 else "d")
    fig.subplots_adjust(left=0.09, right=0.965, top=0.91, bottom=0.09, hspace=0.40, wspace=0.34)
    fig.suptitle(
        "The pressure-null cusp is replaced by a smooth force-equilibrium basin",
        fontsize=14.2, fontweight="semibold", y=0.995,
    )
    return _save_figure(ctx, fig, stem, tight=False)


def _stencil_transfer(
    positions_m: np.ndarray,
    target_m: Sequence[float],
    frequency_hz: float,
    *,
    spacing_m: float = 0.0005,
) -> np.ndarray:
    offsets = np.arange(-2, 3, dtype=float) * float(spacing_m)
    x, y, z = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    points = np.asarray(target_m, dtype=float) + np.stack([x, y, z], axis=-1)
    geometry = ArrayGeometry(positions_m)
    return transfer_matrix(
        points,
        geometry,
        frequency_hz,
        source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
    )


def _tri3_problem(ctx: PipelineContext) -> MultiTrapProblem:
    targets = (
        np.array([0.0, +0.013, 0.030]),
        np.array([-0.011, -0.009, 0.030]),
        np.array([+0.011, -0.009, 0.030]),
    )
    stencils = tuple(
        TargetStencil(
            target_id=f"T{index + 1}", target_m=tuple(target),
            transfer=_stencil_transfer(ctx.positions_m, target, ctx.config.frequency_hz),
            spacing_m=ctx.config.stencil_m, curvature_weight=np.eye(3),
        )
        for index, target in enumerate(targets)
    )
    return MultiTrapProblem(
        case_id="canonical-tri3",
        stencils=stencils,
        material=SphereInFluid(frequency_hz=ctx.config.frequency_hz),
        geometry_id="square-16x16",
    )


def _multitrap_schedule(ctx: PipelineContext) -> tuple[int, int]:
    total = int(ctx.config.multitrap_maxiter)
    first = max(20, int(round(0.25 * total)))
    return first, max(20, total - first)


def _ensure_multitrap(ctx: PipelineContext, *, include_ablation: bool = False) -> dict[str, Any]:
    memo_key = "multitrap_ablation" if include_ablation else "multitrap_main"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]
    problem = _tri3_problem(ctx)
    schedule = _multitrap_schedule(ctx)
    conventional = conventional_double_trap_config(
        stage_maxiters=schedule, gtol=ctx.config.gtol
    )
    regularized = regularized_fe_double_trap_config(
        stage_maxiters=schedule, gtol=ctx.config.gtol
    )
    seed = ctx.config.random_seed + 41
    initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, problem.n_actuators)
    variants = list(spu_ablation_configs(regularized)) if include_ablation else [conventional, regularized]
    results: list[MultitrapRunResult] = []
    for config in variants:
        payload = {
            "problem": problem.fingerprint,
            "config": config.to_payload(),
            "seed": seed,
            "initial_phase": digest_array(initial),
            "history": bool(not include_ablation or config.components.force_smoothing and config.components.pressure_retention and config.components.uniformity),
        }
        result, _, _ = ctx.cache.get_or_compute(
            "multitrap_run", payload,
            lambda config=config: run_multitrap(
                problem, config, seed=seed, initial_phases=initial,
                record_history=not include_ablation or config.ablation_id == "S1P1U1",
            ),
            recompute=ctx.recompute,
        )
        results.append(result)
    output = {"problem": problem, "results": results, "initial": initial}
    ctx.memo[memo_key] = output
    if not include_ablation:
        ctx.tables["tri3_runs"] = pd.DataFrame([result.summary_row() for result in results])
        ctx.tables["tri3_per_target"] = pd.DataFrame(
            [row for result in results for row in result.per_target_rows()]
        )
    else:
        ctx.tables["multitrap_spu_ablation"] = pd.DataFrame([result.summary_row() for result in results])
        ctx.tables["multitrap_spu_per_target"] = pd.DataFrame(
            [row for result in results for row in result.per_target_rows()]
        )
    return output


def _exact_evaluator_for_phase(
    ctx: PipelineContext,
    phase: np.ndarray,
    *,
    positions_m: np.ndarray | None = None,
    frequency_hz: float | None = None,
) -> PartialWaveForceEvaluator:
    medium = Medium()
    sphere = ElasticSphere()
    return PartialWaveForceEvaluator(
        float(frequency_hz or ctx.config.frequency_hz),
        _array_field(ctx, phase, positions_m=positions_m, frequency_hz=frequency_hz),
        medium=medium,
        sphere=sphere,
        numerics=_partial_wave_numerics(ctx),
    )


def _plot_force_neighborhood(
    ax: plt.Axes,
    evaluator: Any,
    target: np.ndarray,
    equilibrium: np.ndarray,
    *,
    equilibrium_resolved: bool,
    half_width_m: float = 0.003,
    shape: int = 7,
    title: str,
) -> None:
    section = sample_force_section(
        evaluator, target, axes=(0, 1), half_width_m=half_width_m, shape=(shape, shape)
    )
    amplitude = np.abs(section.pressure_pa_peak)
    axis_u = 1e3 * section.coordinates_u_m[0]
    axis_v = 1e3 * section.coordinates_v_m[:, 0]
    norm = shared_power_norm([amplitude], lower=0.0, upper=99.5)
    delta = 1e3 * (equilibrium - target)
    field_panel(
        ax, amplitude, (axis_u[0], axis_u[-1], axis_v[0], axis_v[-1]), norm=norm,
        xlabel=r"$x-x^*$ (mm)", ylabel=r"$y-y^*$ (mm)",
        target_mm=(0.0, 0.0),
        equilibrium_mm=(float(delta[0]), float(delta[1])) if equilibrium_resolved else None,
    )
    if not equilibrium_resolved:
        ax.scatter(delta[0], delta[1], marker="^", facecolor="none", edgecolor="white", s=48, linewidth=1.2, zorder=25)
        ax.text(0.98, 0.02, "unresolved candidate", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color="white")
    force = section.in_plane_total_force_n
    force_streamlines(ax, axis_u, axis_v, force[..., 0], force[..., 1], density=0.85)
    set_panel_title(ax, title)


def _multitrap_native_direction(
    problem: MultiTrapProblem,
    result: MultitrapRunResult,
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    evaluation = evaluate_multitrap_objective(
        problem,
        result.phases,
        config,
        force_epsilon=result.stages[-1].force_epsilon,
        uniformity_epsilon=result.stages[-1].uniformity_epsilon,
    )
    gradient = np.asarray(evaluation.gradient_full, dtype=float)
    u = -gradient / max(float(np.linalg.norm(gradient)), 1.0e-30)
    history = np.asarray(result.history_phase_rad, dtype=float)
    if history.ndim != 2 or history.shape[0] < 2:
        raise RuntimeError("multitrap decision section requires accepted phase history")
    current = history[-1]
    v: np.ndarray | None = None
    for previous in history[-2::-1]:
        gauge = np.angle(np.vdot(np.exp(1j * previous), np.exp(1j * current)))
        step = np.angle(np.exp(1j * current) * np.exp(-1j * gauge) * np.exp(-1j * previous))
        step -= u * float(np.dot(step, u))
        norm = float(np.linalg.norm(step))
        if norm > 1.0e-14:
            v = step / norm
            break
    if v is None:
        raise RuntimeError("multitrap history has no nonzero accepted transverse step")
    return u, v


def _multitrap_line_section(
    problem: MultiTrapProblem,
    result: MultitrapRunResult,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = _multitrap_native_direction(problem, result, config)
    t = np.linspace(-0.30, 0.30, 101)
    curves = []
    for direction in (u, v):
        values = np.asarray(
            [
                evaluate_multitrap_objective(
                    problem,
                    result.phases + value * direction,
                    config,
                    force_epsilon=result.stages[-1].force_epsilon,
                    uniformity_epsilon=result.stages[-1].uniformity_epsilon,
                ).value
                for value in t
            ],
            dtype=float,
        )
        curves.append(values - np.nanmin(values))
    return t, curves[0], curves[1]


def figure_4_tri3_multitrap(ctx: PipelineContext) -> mpl.figure.Figure:
    """Repeat the speed/geometry argument on the predeclared tri3 challenge."""

    stem = "Figure_4_tri3_multitrap"
    if stem in ctx.figures:
        return ctx.figures[stem]
    bundle = _ensure_multitrap(ctx)
    problem: MultiTrapProblem = bundle["problem"]
    results: list[MultitrapRunResult] = bundle["results"]
    conventional = next(result for result in results if result.method == "Conventional")
    regularized = next(result for result in results if result.method == "Regularized-FE")
    targets = [np.asarray(stencil.target_m, dtype=float) for stencil in problem.stencils]

    comparisons = [
        _exact_comparison(ctx, regularized.phases, target, key=f"fig4-tri3-regularized-{i}")
        for i, target in enumerate(targets)
    ]
    validation = pd.DataFrame(
        [
            _validation_row(f"regularized FE {problem.stencils[i].target_id}", comparison)
            | {
                "target_x_m": targets[i][0],
                "target_y_m": targets[i][1],
                "target_z_m": targets[i][2],
            }
            for i, comparison in enumerate(comparisons)
        ]
    )
    ctx.tables["tri3_finite_ka"] = validation

    schedule = _multitrap_schedule(ctx)
    configs = {
        "Conventional": conventional_double_trap_config(
            stage_maxiters=schedule, gtol=ctx.config.gtol
        ),
        "regularized FE": regularized_fe_double_trap_config(
            stage_maxiters=schedule, gtol=ctx.config.gtol
        ),
    }
    sections_1d = {
        "Conventional": _multitrap_line_section(
            problem, conventional, configs["Conventional"]
        ),
        "regularized FE": _multitrap_line_section(
            problem, regularized, configs["regularized FE"]
        ),
    }
    slope_rows = []
    for label, (t, values, step_values) in sections_1d.items():
        center_index = len(t) // 2
        dt = float(t[1] - t[0])
        slope_rows.append(
            {
                "method": label,
                "left_one_sided_slope": float((values[center_index] - values[center_index - 1]) / dt),
                "right_one_sided_slope": float((values[center_index + 1] - values[center_index]) / dt),
                "half_range_l2_rad": 0.30,
                "direction_definition": "native negative terminal gradient",
                "transverse_definition": "last accepted step orthogonalized to terminal gradient",
            }
        )
    ctx.tables["tri3_decision_section_summary"] = pd.DataFrame(slope_rows)

    field = _array_field(ctx, regularized.phases)
    field_center = np.array([0.0, 0.0, 0.030])
    sections = _pressure_sections(
        field, field_center, half_xy_m=0.075, half_z_m=0.050,
        samples=161 if ctx.config.full else 121,
    )

    fig = plt.figure(figsize=(14.8, 8.3))
    gs = fig.add_gridspec(
        2, 3, width_ratios=(1.42, 1.0, 1.0),
        hspace=0.40, wspace=0.34,
    )
    ax_field = fig.add_subplot(gs[:, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_eq = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_fe = fig.add_subplot(gs[1, 2])

    xy = 1e3 * sections["xy_axis_m"]
    norm = shared_power_norm([sections["xy"]], lower=5.0, upper=99.0, gamma=0.7)
    image = field_panel(
        ax_field,
        sections["xy"],
        (xy[0], xy[-1], xy[0], xy[-1]),
        norm=norm,
        xlabel="x (mm)",
        ylabel="y (mm)",
    )
    for index, (target, comparison) in enumerate(zip(targets, comparisons), start=1):
        target_xy = 1e3 * target[:2]
        equilibrium = np.asarray(comparison.finite_ka.equilibrium.equilibrium_m, dtype=float)
        equilibrium_xy = 1e3 * equilibrium[:2]
        ax_field.plot(
            [target_xy[0], equilibrium_xy[0]], [target_xy[1], equilibrium_xy[1]],
            color="white", lw=0.9, alpha=0.92, zorder=19,
        )
        ax_field.scatter(*target_xy, marker="x", color=TARGET_COLOR, s=58, linewidth=1.8, zorder=20)
        marker = "o" if comparison.finite_ka.equilibrium.numerical_root_found else "^"
        ax_field.scatter(
            *equilibrium_xy, marker=marker, facecolor="white", edgecolor="black",
            s=34, linewidth=0.9, zorder=21,
        )
        ax_field.text(target_xy[0] + 1.8, target_xy[1] + 1.8, f"T{index}", color="white", fontsize=8.2, zorder=22)
    set_panel_title(ax_field, "Three-target field and partial-wave equilibria")
    panel_label(ax_field, "a")
    colorbar = fig.colorbar(image, ax=ax_field, fraction=0.047, pad=0.025)
    colorbar.set_label(r"$|p|$ (Pa, common source scale)")

    for result, label in ((conventional, "Conventional"), (regularized, "regularized FE")):
        history = pd.DataFrame(result.history_rows())
        if history.empty:
            continue
        objective = history["objective"].to_numpy(dtype=float)
        relative_change = (objective - objective[0]) / max(abs(float(objective[0])), 1.0e-30)
        ax_time.plot(
            history["elapsed_sec"], relative_change,
            color=COLORS[label], lw=2.0, label=label,
        )
    ax_time.axhline(0.0, color="0.55", lw=0.7)
    ax_time.set_xlabel("Wall time (s)")
    ax_time.set_ylabel(r"$(J(t)-J(0))/|J(0)|$")
    ax_time.legend(frameon=False)
    set_panel_title(ax_time, "Wall-time objective progress")
    panel_label(ax_time, "b")

    displacement = []
    for index, comparison in enumerate(comparisons, start=1):
        exact = comparison.finite_ka
        value = exact.equilibrium.displacement_norm_a if exact.equilibrium.numerical_root_found else np.nan
        displacement.append(value)
        ax_eq.scatter(index, value, s=66, color=COLORS["regularized FE"], edgecolor="black", linewidth=0.7, zorder=4)
    ax_eq.axhline(1.0, color="0.45", lw=0.8, ls="--", label="one particle radius")
    ax_eq.set_xticks(range(1, len(targets) + 1), [f"T{i}" for i in range(1, len(targets) + 1)])
    ax_eq.set_ylabel(r"$\|r_{eq}-r^*\|/a$")
    ax_eq.legend(frameon=False, fontsize=7.8)
    set_panel_title(ax_eq, r"Finite-$ka$ partial-wave placement")
    panel_label(ax_eq, "c")

    for ax, label, panel in ((ax_c, "Conventional", "d"), (ax_fe, "regularized FE", "e")):
        t, values, step_values = sections_1d[label]
        ax.plot(t, values, color=COLORS[label], lw=2.15, label=r"$-\nabla J$")
        ax.plot(t, step_values, color=COLORS[label], lw=1.45, ls="--", alpha=0.82, label="last step")
        ax.scatter([0.0], [values[len(values) // 2]], marker="x", color="crimson", s=42, zorder=5)
        ax.set_xlabel(r"Local coordinate $t$ (rad; $\|u\|_2=1$)")
        ax.set_ylabel("Objective above local minimum")
        set_panel_title(ax, f"{label}: native terminal section")
        panel_label(ax, panel)
        inset = ax.inset_axes([0.55, 0.53, 0.40, 0.38])
        selector = np.abs(t) <= 0.03
        inset.plot(t[selector], values[selector], color=COLORS[label], lw=1.35)
        inset.plot(t[selector], step_values[selector], color=COLORS[label], lw=1.0, ls="--", alpha=0.82)
        inset.scatter([0.0], [values[len(values) // 2]], marker="x", color="crimson", s=22, zorder=5)
        inset.tick_params(labelsize=6.5)
        inset.grid(alpha=0.18)
        if label == "Conventional":
            ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    fig.suptitle(
        "The force-equilibrium advantage persists in a three-target inverse-design problem",
        fontsize=14.2, fontweight="semibold", y=0.992,
    )
    fig.subplots_adjust(left=0.06, right=0.975, top=0.92, bottom=0.09)
    return _save_figure(ctx, fig, stem, tight=False)


def _objective_factory(ctx: PipelineContext) -> Callable[[np.ndarray], SingleTargetObjective]:
    def factory(target: np.ndarray) -> SingleTargetObjective:
        condition = Condition(
            label="sota-vortex", positions=ctx.positions_m,
            target_m=tuple(float(v) for v in target), frequency_hz=ctx.config.frequency_hz,
            curvature_weight=np.eye(3), normals=ctx.normals,
        )
        return SingleTargetObjective(condition, "Force-Equilibrium")
    return factory


def _vortex_transfer_factory(
    ctx: PipelineContext,
    target: np.ndarray,
    *,
    spec: VortexControlSpec,
) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    def factory() -> tuple[np.ndarray, np.ndarray]:
        points, signature = vortex_control_points(
            target,
            AcousticMedium().sound_speed_m_s / ctx.config.frequency_hz,
            spec=spec,
        )
        transfer = transfer_matrix(
            points, ArrayGeometry(ctx.positions_m, ctx.normals), ctx.config.frequency_hz,
            source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
        ).reshape(len(points), len(ctx.positions_m))
        return transfer, signature
    return factory


def _multivortex_transfer_factory(
    ctx: PipelineContext,
    targets_m: np.ndarray,
    *,
    spec: VortexControlSpec,
) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    """Build contiguous control-ring blocks for a fixed multitrap target set."""

    targets = np.asarray(targets_m, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3 or targets.shape[0] < 1:
        raise ValueError("targets_m must have shape (target_count, 3)")
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets_m must be finite")
    frozen_targets = targets.copy()

    def factory() -> tuple[np.ndarray, np.ndarray]:
        point_blocks: list[np.ndarray] = []
        signature_blocks: list[np.ndarray] = []
        wavelength_m = AcousticMedium().sound_speed_m_s / ctx.config.frequency_hz
        for target in frozen_targets:
            points, signature = vortex_control_points(
                target,
                wavelength_m,
                spec=spec,
            )
            point_blocks.append(points)
            signature_blocks.append(signature)
        all_points = np.concatenate(point_blocks, axis=0)
        transfer = transfer_matrix(
            all_points,
            ArrayGeometry(ctx.positions_m, ctx.normals),
            ctx.config.frequency_hz,
            source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
        ).reshape(len(all_points), len(ctx.positions_m))
        return transfer, np.concatenate(signature_blocks, axis=0)

    return factory


def _ensure_field_baseline_commands(ctx: PipelineContext) -> dict[str, BenchmarkRun]:
    """Build deterministic matched and offset-control Single prescriptions.

    Existing consumers retain the ``ib`` and ``audit`` keys for the matched
    0.45-wavelength commands.  The 0.60-wavelength commands are explicit
    prescription-sensitivity controls, not alternate solver definitions.
    """

    if "field_baseline_commands" in ctx.memo:
        return ctx.memo["field_baseline_commands"]
    target = MAIN_TARGET_M.copy()
    ib_transfer_factory = _vortex_transfer_factory(
        ctx, target, spec=IB_VORTEX_SPEC
    )
    gs_transfer_factory = _vortex_transfer_factory(
        ctx, target, spec=SINGLE_SIDED_GS_VORTEX_SPEC
    )
    ib_offset_transfer_factory = _vortex_transfer_factory(
        ctx, target, spec=OFFSET_CONTROL_VORTEX_SPEC
    )
    gs_offset_transfer_factory = _vortex_transfer_factory(
        ctx, target, spec=OFFSET_CONTROL_VORTEX_SPEC
    )
    payload = {
        "array": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "target": target.tolist(),
        "frequency_hz": ctx.config.frequency_hz,
        "source_scale_pa_m_per_murata_unit": SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
        "ib": {
            "spec": asdict(IB_VORTEX_SPEC),
            "maxiter": 200,
            "tolerance_rad": 0.01,
        },
        "gs": {
            "spec": asdict(SINGLE_SIDED_GS_VORTEX_SPEC),
            "iterations": 100,
        },
        "ib_offset": {
            "spec": asdict(OFFSET_CONTROL_VORTEX_SPEC),
            "maxiter": 200,
            "tolerance_rad": 0.01,
        },
        "gs_offset": {
            "spec": asdict(OFFSET_CONTROL_VORTEX_SPEC),
            "iterations": 100,
        },
    }

    def producer() -> dict[str, BenchmarkRun]:
        ib = benchmark_synthesis_method_factory(
            iterative_back_projection,
            ib_transfer_factory,
            target_m=target,
            method_kwargs={"maxiter": 200, "tolerance_rad": 0.01},
        )
        gs = benchmark_synthesis_method_factory(
            phase_only_signature_projection_audit,
            gs_transfer_factory,
            target_m=target,
            method_kwargs={"iterations": 100},
        )
        ib_offset = benchmark_synthesis_method_factory(
            iterative_back_projection,
            ib_offset_transfer_factory,
            target_m=target,
            method_kwargs={"maxiter": 200, "tolerance_rad": 0.01},
        )
        gs_offset = benchmark_synthesis_method_factory(
            phase_only_signature_projection_audit,
            gs_offset_transfer_factory,
            target_m=target,
            method_kwargs={"iterations": 100},
        )
        return {
            "ib": replace(
                ib,
                metadata={
                    **dict(ib.metadata),
                    "prescription_role": "matched",
                    "prescription_radius_wavelengths": float(
                        IB_VORTEX_SPEC.radius_wavelengths
                    ),
                },
            ),
            "audit": replace(
                gs,
                metadata={
                    **dict(gs.metadata),
                    "prescription_role": "matched",
                    "prescription_radius_wavelengths": float(
                        SINGLE_SIDED_GS_VORTEX_SPEC.radius_wavelengths
                    ),
                },
            ),
            "ib_offset": replace(
                ib_offset,
                metadata={
                    **dict(ib_offset.metadata),
                    "prescription_role": "offset_control",
                    "prescription_radius_wavelengths": float(
                        OFFSET_CONTROL_VORTEX_SPEC.radius_wavelengths
                    ),
                },
            ),
            "gs_offset": replace(
                gs_offset,
                metadata={
                    **dict(gs_offset.metadata),
                    "prescription_role": "offset_control",
                    "prescription_radius_wavelengths": float(
                        OFFSET_CONTROL_VORTEX_SPEC.radius_wavelengths
                    ),
                },
            ),
        }

    commands, _, _ = ctx.cache.get_or_compute(
        "field_baseline_commands", payload, producer, recompute=ctx.recompute
    )
    ctx.memo["field_baseline_commands"] = commands
    return commands


def _ensure_triple_field_baseline_commands(
    ctx: PipelineContext,
    targets_m: np.ndarray,
) -> dict[str, BenchmarkRun]:
    """Build matched and offset-control IB/GS commands for one Triple set.

    Each eight-point ring retains its own native free phase gauge.  The target
    digest is part of both memo and disk-cache identities so contraction cases
    cannot reuse commands generated for another target set.
    """

    targets = np.asarray(targets_m, dtype=float)
    if targets.ndim != 2 or targets.shape != (3, 3):
        raise ValueError("Triple targets_m must have shape (3, 3)")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Triple targets_m must be finite")
    target_digest = digest_array(targets)
    memo_key = f"triple_field_baseline_commands:{target_digest}"
    if memo_key in ctx.memo:
        return ctx.memo[memo_key]

    matched_factory = _multivortex_transfer_factory(
        ctx,
        targets,
        spec=IB_VORTEX_SPEC,
    )
    offset_factory = _multivortex_transfer_factory(
        ctx,
        targets,
        spec=OFFSET_CONTROL_VORTEX_SPEC,
    )
    representative_target = np.mean(targets, axis=0)
    payload = {
        "array": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "targets": target_digest,
        "target_coordinates_m": targets.tolist(),
        "frequency_hz": ctx.config.frequency_hz,
        "source_scale_pa_m_per_murata_unit": SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
        "control_group_size": 8,
        "matched_spec": asdict(IB_VORTEX_SPEC),
        "offset_control_spec": asdict(OFFSET_CONTROL_VORTEX_SPEC),
        "ib_maxiter": 200,
        "ib_tolerance_rad": 0.01,
        "gs_iterations": 100,
    }

    def annotate(
        run: BenchmarkRun,
        *,
        method: str,
        role: str,
        radius_wavelengths: float,
    ) -> BenchmarkRun:
        return replace(
            run,
            metadata={
                **dict(run.metadata),
                "prescribed_method": method,
                "prescription_role": role,
                "prescription_radius_wavelengths": radius_wavelengths,
                "control_group_size": 8,
                "target_count": 3,
                "targets_content_id": target_digest,
            },
        )

    def producer() -> dict[str, BenchmarkRun]:
        common_ib = {"maxiter": 200, "tolerance_rad": 0.01, "control_group_size": 8}
        common_gs = {"iterations": 100, "control_group_size": 8}
        return {
            "IB_matched": annotate(
                benchmark_synthesis_method_factory(
                    iterative_back_projection,
                    matched_factory,
                    target_m=representative_target,
                    method_kwargs=common_ib,
                ),
                method="IB",
                role="matched",
                radius_wavelengths=float(IB_VORTEX_SPEC.radius_wavelengths),
            ),
            "GS_matched": annotate(
                benchmark_synthesis_method_factory(
                    phase_only_signature_projection_audit,
                    matched_factory,
                    target_m=representative_target,
                    method_kwargs=common_gs,
                ),
                method="GS",
                role="matched",
                radius_wavelengths=float(
                    SINGLE_SIDED_GS_VORTEX_SPEC.radius_wavelengths
                ),
            ),
            "IB_offset": annotate(
                benchmark_synthesis_method_factory(
                    iterative_back_projection,
                    offset_factory,
                    target_m=representative_target,
                    method_kwargs=common_ib,
                ),
                method="IB",
                role="offset_control",
                radius_wavelengths=float(
                    OFFSET_CONTROL_VORTEX_SPEC.radius_wavelengths
                ),
            ),
            "GS_offset": annotate(
                benchmark_synthesis_method_factory(
                    phase_only_signature_projection_audit,
                    offset_factory,
                    target_m=representative_target,
                    method_kwargs=common_gs,
                ),
                method="GS",
                role="offset_control",
                radius_wavelengths=float(
                    OFFSET_CONTROL_VORTEX_SPEC.radius_wavelengths
                ),
            ),
        }

    commands, _, _ = ctx.cache.get_or_compute(
        "triple_field_baseline_commands",
        payload,
        producer,
        recompute=ctx.recompute,
    )
    ctx.memo[memo_key] = commands
    return commands


def _ensure_sota(ctx: PipelineContext) -> dict[str, Any]:
    if "sota" in ctx.memo:
        return ctx.memo["sota"]
    target = MAIN_TARGET_M.copy()
    ib_transfer_factory = _vortex_transfer_factory(
        ctx, target, spec=IB_VORTEX_SPEC
    )
    gs_transfer_factory = _vortex_transfer_factory(
        ctx, target, spec=SINGLE_SIDED_GS_VORTEX_SPEC
    )
    timing_repeat_count = 10 if ctx.config.full else 3
    payload = {
        "mode": ctx.config.budget_mode,
        "array": digest_array(ctx.positions_m),
        "target": target.tolist(),
        "frequency_hz": ctx.config.frequency_hz,
        "fe_maxiter": ctx.config.fe_maxiter,
        "fe_objective_method": asdict(method_spec("Force-Equilibrium")),
        "seed": ctx.config.random_seed + 71,
        "timing_repeat_count": timing_repeat_count,
        "ib_initialized_fe": True,
        "ib_vortex_spec": asdict(IB_VORTEX_SPEC),
        "single_sided_gs_vortex_spec": asdict(SINGLE_SIDED_GS_VORTEX_SPEC),
    }

    def producer() -> dict[str, Any]:
        ib = benchmark_synthesis_method_factory(
            iterative_back_projection, ib_transfer_factory, target_m=target,
            method_kwargs={"maxiter": 200, "tolerance_rad": 0.01},
        )
        audit = benchmark_synthesis_method_factory(
            phase_only_signature_projection_audit, gs_transfer_factory, target_m=target,
            method_kwargs={"iterations": 100},
        )
        fe = benchmark_fe_cold_start_factory(
            _objective_factory(ctx), target, seed=ctx.config.random_seed + 71,
            maxiter=ctx.config.fe_maxiter, gtol=ctx.config.gtol,
        )
        ib_fe = benchmark_ib_initialized_fe_factory(
            _objective_factory(ctx), ib_transfer_factory, target,
            ib_maxiter=200, ib_tolerance_rad=0.01,
            fe_maxiter=ctx.config.fe_maxiter, gtol=ctx.config.gtol,
        )
        warm_targets = [target + np.array([value, 0.0, 0.0]) for value in (-0.0015, 0.0, 0.0015)]
        warm = benchmark_fe_adjacent_target_warm_path(
            _objective_factory(ctx), warm_targets, first_seed=ctx.config.random_seed + 72,
            maxiter=ctx.config.fe_maxiter, gtol=ctx.config.gtol,
        )
        repeats: dict[str, list[BenchmarkRun]] = {
            "ib": [ib], "audit": [audit], "fe": [fe], "ib_fe": [ib_fe]
        }
        for repeat_index in range(1, timing_repeat_count):
            repeats["ib"].append(benchmark_synthesis_method_factory(
                iterative_back_projection, ib_transfer_factory, target_m=target,
                method_kwargs={"maxiter": 200, "tolerance_rad": 0.01},
            ))
            repeats["audit"].append(benchmark_synthesis_method_factory(
                phase_only_signature_projection_audit, gs_transfer_factory, target_m=target,
                method_kwargs={"iterations": 100},
            ))
            repeats["fe"].append(benchmark_fe_cold_start_factory(
                _objective_factory(ctx), target,
                seed=ctx.config.random_seed + 71 + repeat_index,
                maxiter=ctx.config.fe_maxiter, gtol=ctx.config.gtol,
            ))
            repeats["ib_fe"].append(benchmark_ib_initialized_fe_factory(
                _objective_factory(ctx), ib_transfer_factory, target,
                ib_maxiter=200, ib_tolerance_rad=0.01,
                fe_maxiter=ctx.config.fe_maxiter, gtol=ctx.config.gtol,
            ))
        return {
            "ib": ib, "audit": audit, "fe": fe, "ib_fe": ib_fe, "warm": warm,
            "warm_targets": warm_targets, "timing_repeats": repeats,
        }

    bundle, _, _ = ctx.cache.get_or_compute("sota_benchmark", payload, producer, recompute=ctx.recompute)
    # The common reporting ring is the denser single-sided annulus.  IB remains
    # synthesized from its original eight controls; this evaluation grid only
    # measures the resulting annular field without altering its command.
    transfer, signature = gs_transfer_factory()
    bundle["transfer"] = transfer
    bundle["signature"] = signature
    ctx.memo["sota"] = bundle
    timing_rows: list[dict[str, Any]] = []
    for method_key in ("ib", "audit", "fe", "ib_fe"):
        for repeat_index, run in enumerate(bundle["timing_repeats"][method_key]):
            metrics = vortex_ring_metrics(transfer, signature, run.phase_rad)
            timing_rows.append(
                benchmark_run_row(run)
                | {
                    "run_id": f"{method_key}-{repeat_index:02d}",
                    "timing_repeat": repeat_index,
                    **{f"ring_{key}": value for key, value in metrics.items()},
                }
            )
    for path_index, run in enumerate(bundle["warm"]):
        timing_rows.append(
            benchmark_run_row(run)
            | {"run_id": f"warm-path-{path_index:02d}", "timing_repeat": path_index}
        )
    timing = pd.DataFrame(timing_rows)
    ctx.tables["engineering_baseline_timing"] = timing
    ctx.tables["engineering_baseline_scope"] = pd.DataFrame([
        {"method": "IB", "native_objective": "eight-point vortex signature synthesis", "canonical_literature_method": True},
        {"method": "GS", "native_objective": "phase-only synthesis of the single-sided eight-point, 0.45-wavelength annular prescription", "canonical_literature_method": False},
        {"method": "FE", "native_objective": "standard Gor'kov force-equilibrium objective", "canonical_literature_method": False},
        {"method": "IB→FE", "native_objective": "IB initialization followed by the standard Gor'kov force-equilibrium objective", "canonical_literature_method": False},
    ])
    return bundle


def _force_section_cached(
    ctx: PipelineContext,
    phase: np.ndarray,
    target: np.ndarray,
    plane: str,
    *,
    key: str,
) -> Any:
    axes = (0, 1) if plane == "xy" else (0, 2)
    payload = {
        "phase": digest_array(phase), "target": target.tolist(), "plane": plane,
        "shape": ctx.config.force_slice_points,
        "exact_validator_protocol": _finite_ka_protocol(ctx),
        "key": key,
    }
    value, _, _ = ctx.cache.get_or_compute(
        "finite_ka_force_section", payload,
        lambda: sample_force_section(
            _exact_evaluator_for_phase(ctx, phase), target, axes=axes,
            half_width_m=0.006, shape=(ctx.config.force_slice_points, ctx.config.force_slice_points),
        ),
        recompute=ctx.recompute,
    )
    return value


def _plot_force_card(
    ctx: PipelineContext,
    ax: plt.Axes,
    phase: np.ndarray,
    target: np.ndarray,
    equilibrium: np.ndarray,
    *,
    equilibrium_resolved: bool,
    plane: str,
    key: str,
    title: str,
    dense: Mapping[str, Any] | None = None,
    pressure_norm: Normalize | None = None,
) -> None:
    dense = (
        _pressure_sections(
            _array_field(ctx, phase), target,
            half_xy_m=0.006, half_z_m=0.006, samples=91,
        )
        if dense is None else dense
    )
    force = _force_section_cached(ctx, phase, target, plane, key=key)
    axis = 1e3 * dense["xy_axis_m"]
    zaxis = 1e3 * dense["z_axis_m"]
    amplitude = dense["xy"] if plane == "xy" else dense["xz"]
    extent = (axis[0], axis[-1], axis[0], axis[-1]) if plane == "xy" else (axis[0], axis[-1], zaxis[0], zaxis[-1])
    delta = 1e3 * (equilibrium - target)
    equilibrium_uv = (float(delta[0]), float(delta[1] if plane == "xy" else delta[2]))
    field_panel(
        ax, amplitude, extent,
        norm=(
            pressure_norm
            if pressure_norm is not None
            else shared_power_norm([amplitude], lower=0, upper=99.5)
        ),
        xlabel=r"$x-x^*$ (mm)", ylabel=(r"$y-y^*$ (mm)" if plane == "xy" else r"$z-z^*$ (mm)"),
        target_mm=(0, 0), equilibrium_mm=equilibrium_uv if equilibrium_resolved else None,
    )
    if not equilibrium_resolved:
        ax.scatter(*equilibrium_uv, marker="^", facecolor="none", edgecolor="white", s=45, linewidth=1.1, zorder=25)
        ax.text(0.98, 0.02, "unresolved candidate", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.2, color="white")
    u = 1e3 * force.coordinates_u_m[0]
    v = 1e3 * force.coordinates_v_m[:, 0]
    vector = force.in_plane_total_force_n
    force_streamlines(ax, u, v, vector[..., 0], vector[..., 1], density=0.8)
    set_panel_title(ax, title)


def _central_exact_endpoint_specs(ctx: PipelineContext) -> list[dict[str, Any]]:
    """Freeze the paired central-target phase commands before validation.

    Full mode uses all ten shared cold-start seeds deposited by the branch
    benchmark.  Quick mode uses the first seed in sorted order only and is
    explicitly a pipeline check.  IB and GS are deterministic single commands
    using the same single-sided eight-point, 0.45-wavelength annular
    prescription.  Neither is presented as a seed ensemble.  The nominal
    1.4-wavelength reference is reserved for the appendix prescription audit.
    """

    root_ix, root_iy = ctx.branch_bank.root_target
    anchors = ctx.artifact.anchor_table
    trajectories = ctx.artifact.trajectory_table
    fe_rows = anchors[
        anchors["method"].eq("Force-Equilibrium")
        & anchors["ix"].eq(root_ix)
        & anchors["iy"].eq(root_iy)
    ].sort_values("seed")
    conventional_rows = trajectories[
        trajectories["method"].eq("Conventional")
        & trajectories["ix"].eq(root_ix)
        & trajectories["iy"].eq(root_iy)
    ].sort_values("seed")
    fe_by_seed = {int(row.seed): row for row in fe_rows.itertuples(index=False)}
    conventional_by_seed = {
        int(row.seed): row for row in conventional_rows.itertuples(index=False)
    }
    seeds = sorted(set(fe_by_seed) & set(conventional_by_seed))
    if len(seeds) != 10:
        raise RuntimeError(f"Expected 10 paired central-target seeds, found {len(seeds)}")
    selected_seeds = seeds if ctx.config.full else seeds[:1]
    specs: list[dict[str, Any]] = []
    for seed in selected_seeds:
        conventional = conventional_by_seed[seed]
        fe = fe_by_seed[seed]
        conventional_index = int(conventional.trajectory_index)
        fe_index = int(fe.state_index)
        target = np.asarray(
            [conventional.target_x_m, conventional.target_y_m, conventional.target_z_m],
            dtype=float,
        )
        if not np.allclose(target, MAIN_TARGET_M, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("Central paired bank is not at the canonical target")
        pair_id = f"seed-{seed}"
        specs.extend(
            [
                {
                    "method": "Conventional",
                    "pair_id": pair_id,
                    "seed": seed,
                    "phase": np.asarray(
                        ctx.artifact.trajectory_snapshots[conventional_index, -1],
                        dtype=float,
                    ),
                    "target_m": target,
                    "phase_source": "deposited Conventional final budget state",
                    "terminal_iteration": int(conventional.final_budget_state_iteration),
                    "source_index": conventional_index,
                },
                {
                    "method": "FE",
                    "pair_id": pair_id,
                    "seed": seed,
                    "phase": np.asarray(ctx.artifact.anchor_phases[fe_index], dtype=float),
                    "target_m": target,
                    "phase_source": "deposited FE cold-start endpoint",
                    "terminal_iteration": int(fe.nit),
                    "source_index": fe_index,
                },
            ]
        )

    baselines = _ensure_field_baseline_commands(ctx)
    for method, run in (("IB", baselines["ib"]), ("GS", baselines["audit"])):
        specs.append(
            {
                "method": method,
                "pair_id": "",
                "seed": math.nan,
                "phase": np.asarray(run.phase_rad, dtype=float),
                "target_m": MAIN_TARGET_M.copy(),
                "phase_source": (
                    "single-sided eight-point, 0.45-wavelength annular iterative back-projection prescription"
                    if method == "IB"
                    else "single-sided eight-point, 0.45-wavelength annular phase-only GS prescription"
                ),
                "terminal_iteration": int(getattr(run.solver_result, "iterations", 0)),
                "source_index": -1,
            }
        )
    return specs


def _finite_ka_plane_wave_audit(ctx: PipelineContext) -> pd.DataFrame:
    """Regression of stress integration against an analytic partial-wave series."""

    pressure_pa_peak = 1000.0
    medium = Medium()
    sphere = ElasticSphere()
    k_rad_m = 2.0 * math.pi * ctx.config.frequency_hz / medium.sound_speed_m_s
    numerics = _partial_wave_numerics(ctx)
    payload = {
        "pressure_pa_peak": pressure_pa_peak,
        "frequency_hz": ctx.config.frequency_hz,
        "medium": asdict(medium),
        "sphere": asdict(sphere),
        "numerics": asdict(numerics),
        "validator_model": PartialWaveForceEvaluator.model_name,
        "validator_protocol": _finite_ka_protocol(ctx),
        "analytic_lmax": 24,
    }

    def producer() -> dict[str, Any]:
        evaluator = PartialWaveForceEvaluator(
            ctx.config.frequency_hz,
            PlaneWavePressureField(pressure_pa_peak, k_rad_m),
            medium=medium,
            sphere=sphere,
            numerics=numerics,
            include_effective_gravity=False,
        )
        numerical = evaluator.force(np.zeros(3))
        analytic = analytic_plane_wave_radiation_force_z(
            pressure_pa_peak,
            ctx.config.frequency_hz,
            medium,
            sphere,
            lmax=24,
        )
        relative_error = abs(float(numerical.radiation_n[2]) - analytic) / max(
            abs(analytic), 1.0e-30
        )
        return {
            "test": "plane-wave stress integral versus analytic partial-wave series",
            "numerical_force_z_n": float(numerical.radiation_n[2]),
            "analytic_force_z_n": float(analytic),
            "relative_error": float(relative_error),
            "projection_residual": float(numerical.metadata.get("projection_residual", math.nan)),
            "projection_condition": float(numerical.metadata.get("projection_condition", math.nan)),
            "partial_wave_lmax": int(numerics.lmax),
            "run_mode": ctx.config.mode,
        }

    value, _, _ = ctx.cache.get_or_compute(
        "finite_ka_plane_wave_regression", payload, producer, recompute=ctx.recompute
    )
    return pd.DataFrame([value])


def _finite_ka_jacobian_step_audit(
    ctx: PipelineContext,
    spec: Mapping[str, Any],
    result: Any,
) -> pd.DataFrame:
    """Check one endpoint's stiffness sensitivity to the centred difference step.

    The full benchmark stores the complete Jacobian for every endpoint.  This
    audit repeats that derivative at three steps while holding the resolved
    equilibrium fixed.  The caller applies it to every endpoint before assigning
    the plotted restoring/non-restoring class.
    """

    if not bool(result.equilibrium.numerical_root_found):
        return pd.DataFrame(
            [
                {
                    "method": str(spec["method"]),
                    "pair_id": str(spec["pair_id"]),
                    "phase_content_id": digest_array(np.asarray(spec["phase"], dtype=float)),
                    "status": "reference equilibrium unresolved; audit not evaluated",
                    "run_mode": ctx.config.mode,
                }
            ]
        )
    phase = np.asarray(spec["phase"], dtype=float)
    equilibrium = np.asarray(result.equilibrium.equilibrium_m, dtype=float)
    steps_a = (0.05, 0.10, 0.20)
    payload = {
        "method": str(spec["method"]),
        "pair_id": str(spec["pair_id"]),
        "phase": digest_array(phase),
        "positions": digest_array(ctx.positions_m),
        "normals": digest_array(ctx.normals),
        "equilibrium_m": equilibrium.tolist(),
        "steps_a": list(steps_a),
        "frequency_hz": ctx.config.frequency_hz,
        "protocol": _finite_ka_protocol(ctx),
    }

    def producer() -> list[dict[str, Any]]:
        medium = Medium()
        sphere = ElasticSphere()
        evaluator = PartialWaveForceEvaluator(
            ctx.config.frequency_hz,
            _array_field(ctx, phase),
            medium=medium,
            sphere=sphere,
            numerics=_partial_wave_numerics(ctx),
            include_effective_gravity=True,
        )
        jacobians = [
            np.asarray(result.force_jacobian_n_m, dtype=float)
            if math.isclose(float(step_a), 0.10, rel_tol=0.0, abs_tol=1.0e-15)
            else force_jacobian(
                evaluator,
                equilibrium,
                float(step_a) * evaluator.sphere.radius_m,
            )
            for step_a in steps_a
        ]
        reference = jacobians[0]
        reference_norm = max(float(np.linalg.norm(reference)), 1.0e-30)
        rows: list[dict[str, Any]] = []
        for step_a, jacobian in zip(steps_a, jacobians):
            stiffness = -(jacobian + jacobian.T) / 2.0
            eigenvalues = np.linalg.eigvalsh(stiffness)
            antisymmetric = (jacobian - jacobian.T) / 2.0
            rows.append(
                {
                    "method": str(spec["method"]),
                    "pair_id": str(spec["pair_id"]),
                    "phase_content_id": digest_array(phase),
                    "status": "evaluated",
                    "jacobian_step_a": float(step_a),
                    "jacobian_step_m": float(step_a) * evaluator.sphere.radius_m,
                    "jacobian_relative_frobenius_change_from_0p05a": float(
                        np.linalg.norm(jacobian - reference) / reference_norm
                    ),
                    "stiffness_min_n_m": float(eigenvalues[0]),
                    "stiffness_mid_n_m": float(eigenvalues[1]),
                    "stiffness_max_n_m": float(eigenvalues[2]),
                    "antisymmetric_jacobian_fro_n_m": float(np.linalg.norm(antisymmetric)),
                    "equilibrium_x_m": float(equilibrium[0]),
                    "equilibrium_y_m": float(equilibrium[1]),
                    "equilibrium_z_m": float(equilibrium[2]),
                    "run_mode": ctx.config.mode,
                }
            )
        return rows

    rows, _, _ = ctx.cache.get_or_compute(
        "finite_ka_jacobian_step_audit", payload, producer, recompute=ctx.recompute
    )
    return pd.DataFrame(rows)


def _ensure_finite_ka_equilibrium_benchmark(ctx: PipelineContext) -> dict[str, Any]:
    if "finite_ka_equilibrium_benchmark" in ctx.memo:
        return ctx.memo["finite_ka_equilibrium_benchmark"]
    specs = _central_exact_endpoint_specs(ctx)
    default_workers = min(4, len(specs))
    workers = max(1, min(len(specs), int(os.environ.get("HAT_EXACT_WORKERS", default_workers))))

    def evaluate(spec: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        phase = np.asarray(spec["phase"], dtype=float)
        phase_digest = digest_array(phase)
        result = _finite_ka_validation(
            ctx,
            phase,
            np.asarray(spec["target_m"], dtype=float),
            key=f"fig5-{spec['method']}-{spec['pair_id'] or phase_digest[:12]}",
        )
        row = _static_validation_row(str(spec["method"]), result)
        target = np.asarray(spec["target_m"], dtype=float)
        row.update(
            {
                "pair_id": str(spec["pair_id"]),
                "seed": spec["seed"],
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "phase_content_id": phase_digest,
                "phase_source": str(spec["phase_source"]),
                "terminal_iteration": int(spec["terminal_iteration"]),
                "source_index": int(spec["source_index"]),
                "run_mode": ctx.config.mode,
                "manuscript_numerics": bool(ctx.config.full),
            }
        )
        return row, result

    ordered: list[tuple[dict[str, Any], Any] | None] = [None] * len(specs)
    if workers == 1:
        for index, spec in enumerate(specs):
            ordered[index] = evaluate(spec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(evaluate, spec): index for index, spec in enumerate(specs)
            }
            for future in as_completed(future_to_index):
                ordered[future_to_index[future]] = future.result()
    completed = [item for item in ordered if item is not None]
    raw = pd.DataFrame([item[0] for item in completed])
    results = [item[1] for item in completed]
    root_candidates = pd.DataFrame(
        [
            candidate_row
            for index, (spec, result) in enumerate(zip(specs, results))
            for candidate_row in _root_candidate_rows(ctx, index, spec, result)
        ]
    )

    def audit_endpoint(index: int) -> pd.DataFrame:
        table = _finite_ka_jacobian_step_audit(ctx, specs[index], results[index])
        table.insert(0, "endpoint_index", index)
        return table

    audit_tables: list[pd.DataFrame | None] = [None] * len(specs)
    if workers == 1:
        for index in range(len(specs)):
            audit_tables[index] = audit_endpoint(index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(audit_endpoint, index): index for index in range(len(specs))
            }
            for future in as_completed(future_to_index):
                audit_tables[future_to_index[future]] = future.result()
    jacobian_step = pd.concat(
        [table for table in audit_tables if table is not None], ignore_index=True
    )
    raw["finite_ka_root_class_nominal_h0p10a"] = raw["finite_ka_root_class"]
    raw["finite_ka_stiffness_step_robust"] = False
    raw["finite_ka_stiffness_min_over_steps_n_m"] = math.nan
    raw["finite_ka_stiffness_max_over_steps_n_m"] = math.nan
    for index in range(len(raw)):
        if not bool(raw.at[index, "finite_ka_root_found"]):
            continue
        rows = jacobian_step[
            jacobian_step["endpoint_index"].eq(index)
            & jacobian_step["status"].eq("evaluated")
        ]
        values = rows["stiffness_min_n_m"].to_numpy(float)
        if len(values) != 3 or not np.all(np.isfinite(values)):
            root_class = "resolved-stiffness-indeterminate"
        elif np.all(values > 0.0):
            root_class = "resolved-restoring"
        elif np.all(values < 0.0):
            root_class = "resolved-nonrestoring"
        else:
            root_class = "resolved-stiffness-indeterminate"
        raw.at[index, "finite_ka_root_class"] = root_class
        raw.at[index, "finite_ka_locally_restoring"] = root_class == "resolved-restoring"
        raw.at[index, "finite_ka_stiffness_step_robust"] = root_class in {
            "resolved-restoring", "resolved-nonrestoring"
        }
        if len(values) and np.any(np.isfinite(values)):
            raw.at[index, "finite_ka_stiffness_min_over_steps_n_m"] = float(np.nanmin(values))
            raw.at[index, "finite_ka_stiffness_max_over_steps_n_m"] = float(np.nanmax(values))

    summary_rows: list[dict[str, Any]] = []
    method_order = ("Conventional", "FE", "IB", "GS")
    for method in method_order:
        group = raw[raw["method"].eq(method)]
        restoring = group[group["finite_ka_root_class"].eq("resolved-restoring")]
        row: dict[str, Any] = {
            "method": method,
            "n": int(len(group)),
            "summary_population": "resolved-restoring endpoints only; sign checked at h/a=0.05, 0.10, and 0.20",
            "resolved_roots": int(group["finite_ka_root_found"].sum()),
            "resolved_restoring": int(group["finite_ka_root_class"].eq("resolved-restoring").sum()),
            "resolved_nonrestoring": int(group["finite_ka_root_class"].eq("resolved-nonrestoring").sum()),
            "resolved_stiffness_indeterminate": int(group["finite_ka_root_class"].eq("resolved-stiffness-indeterminate").sum()),
            "unresolved": int(group["finite_ka_root_class"].eq("unresolved").sum()),
        }
        for column in ("finite_ka_displacement_a", "finite_ka_stiffness_min_n_m"):
            values = restoring[column].to_numpy(float)
            row[f"{column}_median"] = float(np.median(values)) if len(values) else math.nan
            row[f"{column}_q25"] = float(np.quantile(values, 0.25)) if len(values) else math.nan
            row[f"{column}_q75"] = float(np.quantile(values, 0.75)) if len(values) else math.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    paired_source = raw[raw["pair_id"].astype(str).ne("")]
    paired_numeric = paired_source.pivot(
        index="pair_id",
        columns="method",
        values=["finite_ka_displacement_a", "finite_ka_stiffness_min_n_m"],
    )
    paired_numeric.columns = [
        f"{metric}__{method}" for metric, method in paired_numeric.columns
    ]
    paired_class = paired_source.pivot(
        index="pair_id", columns="method", values="finite_ka_root_class"
    )
    paired_class.columns = [f"finite_ka_root_class__{method}" for method in paired_class.columns]
    paired = paired_numeric.join(paired_class).reset_index()
    for column in paired_numeric.columns:
        paired[column] = pd.to_numeric(paired[column], errors="raise")

    plane_wave = _finite_ka_plane_wave_audit(ctx)
    ctx.tables["finite_ka_equilibrium_benchmark_raw"] = raw
    ctx.tables["finite_ka_equilibrium_benchmark_summary"] = summary
    ctx.tables["finite_ka_equilibrium_benchmark_pairing"] = paired
    ctx.tables["finite_ka_equilibrium_root_candidates"] = root_candidates
    ctx.tables["finite_ka_plane_wave_regression"] = plane_wave
    ctx.tables["finite_ka_jacobian_step_audit"] = jacobian_step
    ctx.tables["finite_ka_equilibrium_benchmark_scope"] = pd.DataFrame(
        [
            {
                "design_layer": "Conventional and FE endpoints",
                "optimization_force_definition": "acoustic Gor'kov radiation force; external gravity-buoyancy load is not included in the phase objective",
                "validation_layer": "independent finite-ka elastic-solid-sphere partial-wave force plus gravity-buoyancy",
                "equilibrium_definition": "numerically resolved total-force root",
                "stiffness_definition": "minimum eigenvalue of -sym(dF/dr) at the resolved root",
                "root_classification": "resolved-restoring means finite positive lambda_min at h/a=0.05, 0.10, and 0.20; sign changes/nonfinite values are stiffness-indeterminate; this is local static restoring, not dynamical stability",
                "model_scope": "homogeneous isotropic elastic solid sphere with internal longitudinal and shear modes in inviscid air; no thermoviscous, streaming, wall, nonspherical, or multiple-scattering correction",
                "quick_mode_boundary": "pipeline verification only" if not ctx.config.full else "not applicable; full manuscript numerics",
                "worker_count": workers,
                "protocol_json": json.dumps(
                    _finite_ka_protocol(ctx), sort_keys=True, allow_nan=False
                ),
            }
        ]
    )
    value = {
        "raw": raw,
        "summary": summary,
        "paired": paired,
        "plane_wave": plane_wave,
        "jacobian_step": jacobian_step,
        "specs": specs,
        "results": results,
    }
    ctx.memo["finite_ka_equilibrium_benchmark"] = value
    return value


def figure_5_matched_holography(ctx: PipelineContext) -> mpl.figure.Figure:
    """Paired objective benchmark under one independent finite-ka evaluator."""

    stem = "Figure_5_matched_holography"
    if stem in ctx.figures:
        return ctx.figures[stem]
    benchmark = _ensure_finite_ka_equilibrium_benchmark(ctx)
    raw = benchmark["raw"].copy()
    summary = benchmark["summary"].copy()
    plane_wave = benchmark["plane_wave"].iloc[0]
    method_order = ["Conventional", "FE", "IB", "GS"]
    display = {
        "Conventional": "Conventional",
        "FE": "FE",
        "IB": "IB",
        "GS": "GS",
    }
    colors = {
        "Conventional": COLORS["Conventional"],
        "FE": COLORS["FE"],
        "IB": COLORS["IB"],
        "GS": COLORS["GS-PO audit"],
    }
    markers = {"Conventional": "o", "FE": "s", "IB": "^", "GS": "v"}

    fig = plt.figure(figsize=(13.8, 6.3))
    grid = fig.add_gridspec(1, 2, width_ratios=(4.25, 1.35), wspace=0.23)
    ax = fig.add_subplot(grid[0, 0])
    ax_status = fig.add_subplot(grid[0, 1])

    paired = raw[raw["pair_id"].astype(str).ne("")]
    for pair_id, group in paired.groupby("pair_id", sort=True):
        endpoints = {row.method: row for row in group.itertuples(index=False)}
        if (
            "Conventional" in endpoints
            and "FE" in endpoints
            and endpoints["Conventional"].finite_ka_root_found
            and endpoints["FE"].finite_ka_root_found
        ):
            conventional = endpoints["Conventional"]
            fe = endpoints["FE"]
            ax.annotate(
                "",
                xy=(fe.finite_ka_displacement_a, 1.0e3 * fe.finite_ka_stiffness_min_n_m),
                xytext=(conventional.finite_ka_displacement_a, 1.0e3 * conventional.finite_ka_stiffness_min_n_m),
                arrowprops=dict(arrowstyle="->", color="0.68", lw=0.8, alpha=0.55),
                zorder=1,
            )

    for method in method_order:
        group = raw[raw["method"].eq(method)]
        resolved = group[group["finite_ka_root_found"]]
        for root_class, class_group in resolved.groupby("finite_ka_root_class"):
            restoring = root_class == "resolved-restoring"
            class_label = {
                "resolved-restoring": display[method],
                "resolved-nonrestoring": f"{display[method]} (resolved, non-restoring)",
                "resolved-stiffness-indeterminate": f"{display[method]} (stiffness indeterminate)",
            }.get(str(root_class), f"{display[method]} ({root_class})")
            ax.scatter(
                class_group["finite_ka_displacement_a"],
                1.0e3 * class_group["finite_ka_stiffness_min_n_m"],
                marker=markers[method],
                s=42 if method in {"Conventional", "FE"} else 68,
                facecolors=colors[method] if restoring else "none",
                edgecolors=colors[method],
                linewidths=1.2,
                alpha=0.58 if method in {"Conventional", "FE"} else 0.92,
                label=class_label,
                zorder=3,
            )
        summary_row = summary[summary["method"].eq(method)].iloc[0]
        if int(summary_row["resolved_restoring"]) >= 2:
            x = float(summary_row["finite_ka_displacement_a_median"])
            xq25 = float(summary_row["finite_ka_displacement_a_q25"])
            xq75 = float(summary_row["finite_ka_displacement_a_q75"])
            y = 1.0e3 * float(summary_row["finite_ka_stiffness_min_n_m_median"])
            yq25 = 1.0e3 * float(summary_row["finite_ka_stiffness_min_n_m_q25"])
            yq75 = 1.0e3 * float(summary_row["finite_ka_stiffness_min_n_m_q75"])
            ax.errorbar(
                x,
                y,
                xerr=np.array([[x - xq25], [xq75 - x]]),
                yerr=np.array([[y - yq25], [yq75 - y]]),
                fmt=markers[method],
                ms=10,
                mfc=colors[method],
                mec="white",
                mew=1.0,
                ecolor=colors[method],
                elinewidth=1.8,
                capsize=4,
                zorder=8,
            )

    positive_x = raw.loc[raw["finite_ka_root_found"], "finite_ka_displacement_a"].to_numpy(float)
    if len(positive_x) and np.all(positive_x > 0.0):
        ax.set_xscale("log")
    ax.axhline(0.0, color="0.28", lw=0.9)
    ax.set_xlabel(r"Equilibrium displacement $\|\mathbf{r}_{eq}-\mathbf{r}^*\|_2/a$")
    ax.set_ylabel(r"Weakest equilibrium-centred restoring stiffness (mN m$^{-1}$)")
    set_panel_title(ax, "Target placement and restoring stiffness are separate observables")
    panel_label(ax, "a")
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="best", fontsize=8.3)
    ax.text(
        0.018,
        0.025,
        "Faint arrows pair identical cold-start seeds (Conventional → FE).\nLarge markers with whiskers: median and IQR (full mode).",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="0.30",
    )

    status = summary.set_index("method").loc[method_order]
    y_positions = np.arange(len(method_order))
    totals = status["n"].to_numpy(float)
    restoring = 100.0 * status["resolved_restoring"].to_numpy(float) / totals
    nonrestoring = 100.0 * status["resolved_nonrestoring"].to_numpy(float) / totals
    indeterminate = 100.0 * status["resolved_stiffness_indeterminate"].to_numpy(float) / totals
    unresolved = 100.0 * status["unresolved"].to_numpy(float) / totals
    ax_status.barh(y_positions, restoring, color="#59A14F", label="restoring")
    ax_status.barh(y_positions, nonrestoring, left=restoring, color="#E15759", label="non-restoring")
    ax_status.barh(
        y_positions,
        indeterminate,
        left=restoring + nonrestoring,
        color="#F1CE63",
        label="stiffness indeterminate",
    )
    ax_status.barh(
        y_positions,
        unresolved,
        left=restoring + nonrestoring + indeterminate,
        facecolor="white",
        edgecolor="0.35",
        hatch="///",
        label="unresolved",
    )
    for index, method in enumerate(method_order):
        total = int(status.loc[method, "n"])
        resolved_count = int(status.loc[method, "resolved_roots"])
        ax_status.text(102.0, index, f"roots {resolved_count}/{total}", va="center", fontsize=8.0)
    ax_status.set_yticks(y_positions, [display[method] for method in method_order])
    ax_status.invert_yaxis()
    ax_status.set_xlabel("Endpoint fraction (%)")
    set_panel_title(ax_status, "Numerically resolved roots")
    panel_label(ax_status, "b")
    ax_status.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        fontsize=6.9,
        columnspacing=0.8,
        handletextpad=0.45,
    )
    ax_status.set_xlim(0.0, 126.0)
    ax_status.set_xticks((0, 25, 50, 75, 100))

    mode_note = (
        "FULL manuscript numerics: 19 starts, lmax=8"
        if ctx.config.full
        else "PAPER-SCALE SMOKE: accepted manuscript configuration; one paired seed, 7 starts, lmax=3"
    )
    fig.suptitle(
        "Independent finite-ka equilibrium evaluation separates target placement from restoring stiffness",
        fontsize=14.0,
        fontweight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        mode_note
        + f". Plane-wave stress regression relative error: {float(plane_wave.relative_error):.2e}. "
        + "Finite-ka elastic-solid bead; total force includes gravity–buoyancy; "
        + r"restoring class requires consistent $\lambda_{min}$ sign at $h/a=0.05,0.10,0.20$.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.17)
    return _save_figure(ctx, fig, stem, tight=False)


def _sweep_solver(request: EndpointRequest) -> EndpointSolution:
    spec = MethodSpec(
        name="Force-Equilibrium",
        alpha_per_m=float(request.alpha),
        beta_curvature_per_pa=0.0,
        pressure_mode="abs",
        curvature_weight=np.eye(3),
    )
    condition = Condition(
        label=request.condition_id,
        positions=np.asarray(request.positions_m, dtype=float),
        target_m=tuple(request.target.position_m),
        frequency_hz=float(request.frequency_hz),
        curvature_weight=np.eye(3),
    )
    objective = SingleTargetObjective(condition, spec)
    rng = np.random.default_rng(int(request.seed))
    phase0 = rng.uniform(-np.pi, np.pi, size=len(request.positions_m))
    maxiter = int(request.solver_options.get("maxiter", 200))
    gtol = float(request.solver_options.get("gtol", 0.0))
    started = time.perf_counter()
    result = solve_corrected_gorkov_fe(objective, phase0, maxiter=maxiter, gtol=gtol)
    elapsed = time.perf_counter() - started
    finite = bool(np.all(np.isfinite(result.phase_rad)) and np.isfinite(result.objective))
    stationarity_tolerance = float(request.solver_options.get("stationarity_tol", 1.0e-5))
    stationary = bool(np.isfinite(result.gradient_norm) and result.gradient_norm <= stationarity_tolerance)
    return EndpointSolution(
        phase_radians=result.phase_rad,
        success=bool(finite and (result.success or stationary)),
        objective_value=result.objective,
        gradient_norm=result.gradient_norm,
        iterations=result.iterations,
        wall_time_s=elapsed,
        diagnostics={
            "optimizer_success": result.success,
            "optimizer_status": result.status,
            "optimizer_message": result.message,
            "finite_terminal": finite,
            "stationarity_tolerance": stationarity_tolerance,
            "stationary_terminal": stationary,
            "standard_gorkov": True,
        },
    )


def _sweep_contract() -> SolverContract:
    return SolverContract(
        solver_id="analytic-BFGS-FE",
        implementation_id=PIPELINE_IMPLEMENTATION + "-sweep-v1",
    )


def _sweep_targets(ctx: PipelineContext) -> tuple[TargetPoint, ...]:
    mode = "full" if ctx.config.full else "quick"
    return default_target_grid(mode)


def _successful_endpoint_bank(
    bank: EndpointBank,
    *,
    smoke_stationarity_tolerance: float | None = None,
) -> EndpointBank:
    records = []
    for record in bank.records:
        accepted = bool(record.solution.success)
        smoke_only = False
        if (
            not accepted
            and smoke_stationarity_tolerance is not None
            and np.isfinite(record.solution.gradient_norm)
            and record.solution.gradient_norm <= smoke_stationarity_tolerance
        ):
            accepted = True
            smoke_only = True
        if not accepted:
            continue
        if smoke_only:
            diagnostics = dict(record.solution.diagnostics) | {
                "quick_smoke_relaxed_stationarity": True,
                "quick_smoke_stationarity_tolerance": float(smoke_stationarity_tolerance),
            }
            records.append(
                replace(
                    record,
                    solution=replace(record.solution, success=True, diagnostics=diagnostics),
                )
            )
        else:
            records.append(record)
    return EndpointBank(
        sweep_name=bank.sweep_name,
        mode=bank.mode,
        contract=bank.contract,
        records=records,
        design=dict(bank.design) | {
            "analysis_includes_only_successful_stationary_endpoints": True,
            "quick_smoke_relaxed_stationarity_tolerance": smoke_stationarity_tolerance,
            "raw_record_count": len(bank.records),
        },
    )


def _empty_projection(bank: EndpointBank, reason: str) -> FixedFrameProjection:
    rows = []
    for record in bank.records:
        rows.append(record.as_row() | {
            "embedding_1": math.nan, "embedding_2": math.nan,
            "component": None, "assigned": False,
            "component_distance": math.nan, "component_margin": math.nan,
            "component_gate_failure": reason,
        })
    coverage = []
    if rows:
        for _, group in pd.DataFrame(rows).groupby(
            ["axis_value", "array_id", "frequency_hz", "alpha", "target_id"], dropna=False
        ):
            row = group.iloc[0]
            coverage.append({
                "axis_value": row["axis_value"], "array_id": row["array_id"],
                "frequency_hz": row["frequency_hz"], "alpha": row["alpha"],
                "target_id": row["target_id"], "grid_i": row["grid_i"], "grid_j": row["grid_j"],
                "chart_u": row["chart_u"], "chart_v": row["chart_v"],
                "n_endpoints": len(group), "n_successful": int(group["success"].astype(bool).sum()),
                "n_solver_failures": int((~group["success"].astype(bool)).sum()),
                "n_assigned": 0, "n_unassigned": len(group), "recovered_components": "",
                "component_coverage_fraction": 0.0, "complete_reference_coverage": False,
            })
    return FixedFrameProjection(rows, coverage, [], [], ())


def _restore_tested_cell_denominator(
    raw_bank: EndpointBank,
    projection: FixedFrameProjection,
) -> FixedFrameProjection:
    """Add failed/missing tested cells back to coverage without assigning them."""

    raw = pd.DataFrame(raw_bank.rows())
    if raw.empty:
        return projection
    keys = ["axis_value", "array_id", "frequency_hz", "alpha", "target_id"]
    existing = {
        tuple(row.get(key) for key in keys): dict(row)
        for row in projection.coverage_rows
    }
    restored: list[dict[str, Any]] = []
    for key, group in raw.groupby(keys, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        template = group.iloc[0]
        row = existing.get(key_tuple, {
            "axis_value": template["axis_value"], "array_id": template["array_id"],
            "frequency_hz": template["frequency_hz"], "alpha": template["alpha"],
            "target_id": template["target_id"], "grid_i": template["grid_i"],
            "grid_j": template["grid_j"], "chart_u": template["chart_u"],
            "chart_v": template["chart_v"], "n_assigned": 0,
            "recovered_components": "", "component_coverage_fraction": 0.0,
            "complete_reference_coverage": False,
        })
        successful = int(group["success"].astype(bool).sum())
        row["n_endpoints"] = int(len(group))
        row["n_successful"] = successful
        row["n_solver_failures"] = int(len(group) - successful)
        row["n_unassigned"] = int(len(group) - int(row.get("n_assigned", 0)))
        if successful == 0:
            row["complete_reference_coverage"] = False
            row["component_coverage_fraction"] = 0.0
            row["recovered_components"] = ""
        restored.append(row)
    return FixedFrameProjection(
        projection.endpoint_rows, restored, projection.node_rows,
        projection.edge_rows, projection.component_labels,
    )


def appendix_a_gorkov_finite_ka(ctx: PipelineContext) -> mpl.figure.Figure:
    """Document the useful—and incomplete—small-particle surrogate relation."""

    stem = "Appendix_A1_gorkov_finite_ka"
    if stem in ctx.figures:
        return ctx.figures[stem]
    phase = _fe_raw_phase(ctx, "B1")
    comparison = _exact_comparison(ctx, phase, MAIN_TARGET_M, key="appendix-a-rstar")
    finite = comparison.finite_ka
    gorkov = comparison.gorkov
    evaluator = _exact_evaluator_for_phase(ctx, phase)
    surrogate = StandardGorkovEvaluator(
        ctx.config.frequency_hz, evaluator.field, evaluator.medium, evaluator.sphere,
        GorkovNumerics(), include_effective_gravity=True,
    )
    radius = evaluator.sphere.radius_m
    offsets_a = np.linspace(-2.0, 2.0, 25 if not ctx.config.full else 61)
    exact_force_x = np.array([
        evaluator.force(finite.equilibrium.equilibrium_m + np.array([value * radius, 0, 0])).total_n[0]
        for value in offsets_a
    ])
    gorkov_force_x = np.array([
        surrogate.force(gorkov.equilibrium.equilibrium_m + np.array([value * radius, 0, 0])).total_n[0]
        for value in offsets_a
    ])
    lvalues = (2, 3, 4) if not ctx.config.full else (2, 4, 6, 8, 10)
    target_forces = []
    for lmax in lvalues:
        lmax_protocol = dict(_finite_ka_protocol(ctx))
        lmax_protocol["numerics"] = asdict(_partial_wave_numerics(ctx, lmax=lmax))
        payload = {
            "phase": digest_array(phase),
            "positions": digest_array(ctx.positions_m),
            "normals": digest_array(ctx.normals),
            "target_m": MAIN_TARGET_M.tolist(),
            "frequency_hz": ctx.config.frequency_hz,
            "lmax": lmax,
            "numerics": asdict(_partial_wave_numerics(ctx, lmax=lmax)),
            "exact_validator_protocol": lmax_protocol,
            "include_effective_gravity": True,
            "source_strength_pa_m_peak": REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
            "kind": "appendix-a-radiation-force-convergence",
        }
        force, _, _ = ctx.cache.get_or_compute(
            "finite_ka_lmax_force", payload,
            lambda lmax=lmax: PartialWaveForceEvaluator(
                ctx.config.frequency_hz, _array_field(ctx, phase),
                medium=Medium(), sphere=ElasticSphere(),
                numerics=_partial_wave_numerics(ctx, lmax=lmax),
            ).force(MAIN_TARGET_M).radiation_n,
            recompute=ctx.recompute,
        )
        target_forces.append(np.asarray(force, dtype=float))
    target_forces_array = np.stack(target_forces)
    reference = target_forces_array[-1]
    absolute_change = np.linalg.norm(target_forces_array - reference, axis=1)
    relative_error = absolute_change / max(np.linalg.norm(reference), 1e-30)
    medium = Medium()
    sphere = ElasticSphere()
    effective_weight_n = abs(
        (sphere.density_kg_m3 - medium.density_kg_m3)
        * sphere.volume_m3
        * medium.gravity_m_s2
    )
    load_scaled_change = absolute_change / max(effective_weight_n, 1e-30)
    ctx.tables["appendix_gorkov_finite_ka"] = pd.DataFrame([_validation_row("FE B1 at r*", comparison)])
    ctx.tables["appendix_partial_wave_order_check"] = pd.DataFrame(
        {
         "lmax": lvalues,
         "relative_target_force_error": relative_error,
         "absolute_radiation_force_change_n": absolute_change,
         "radiation_force_change_over_effective_weight": load_scaled_change,
         "effective_weight_n": effective_weight_n,
         "radiation_force_x_n": target_forces_array[:, 0],
         "radiation_force_y_n": target_forces_array[:, 1],
         "radiation_force_z_n": target_forces_array[:, 2]}
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0))
    points = {
        "requested target": MAIN_TARGET_M,
        ("Gor'kov equilibrium" if gorkov.equilibrium.numerical_root_found else "Gor'kov unresolved candidate"): gorkov.equilibrium.equilibrium_m,
        ("finite-ka equilibrium" if finite.equilibrium.numerical_root_found else "finite-ka unresolved candidate"): finite.equilibrium.equilibrium_m,
    }
    markers = {label: ("x" if label == "requested target" else "s" if label.startswith("Gor'kov") else "o") for label in points}
    colors = {label: (TARGET_COLOR if label == "requested target" else "#4C78A8" if label.startswith("Gor'kov") else "black") for label in points}
    for label, point in points.items():
        delta = 1e6 * (point - MAIN_TARGET_M)
        axes[0, 0].scatter(delta[0], delta[2], marker=markers[label], color=colors[label], s=65, label=label)
    axes[0, 0].axhline(0, color="0.7", lw=0.7); axes[0, 0].axvline(0, color="0.7", lw=0.7)
    axes[0, 0].set_xlabel(r"$x-x^*$ (um)"); axes[0, 0].set_ylabel(r"$z-z^*$ (um)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    set_panel_title(axes[0, 0], "Optimization point and the two equilibria")

    scale = max(np.max(np.abs(exact_force_x)), np.max(np.abs(gorkov_force_x)), 1e-30)
    axes[0, 1].plot(offsets_a, exact_force_x / scale, color="black", lw=2, label="finite-ka")
    axes[0, 1].plot(offsets_a, gorkov_force_x / scale, color="#4C78A8", lw=2, ls="--", label="Gor'kov")
    axes[0, 1].axhline(0, color="0.6", lw=0.7)
    axes[0, 1].set_xlabel("Offset from each resolved root / candidate / a")
    axes[0, 1].set_ylabel("Restoring force, shared normalized scale")
    axes[0, 1].legend(frameon=False)
    set_panel_title(axes[0, 1], "Local restoring-force comparison")

    x = np.arange(3)
    gorkov_stiffness = np.sort(gorkov.symmetric_stiffness_eigenvalues_n_m) if gorkov.equilibrium.numerical_root_found else np.full(3, np.nan)
    finite_stiffness = np.sort(finite.symmetric_stiffness_eigenvalues_n_m) if finite.equilibrium.numerical_root_found else np.full(3, np.nan)
    axes[1, 0].bar(x - 0.18, gorkov_stiffness, width=0.36, color="#4C78A8", label="Gor'kov")
    axes[1, 0].bar(x + 0.18, finite_stiffness, width=0.36, color="black", label="finite-ka")
    if not finite.equilibrium.numerical_root_found or not gorkov.equilibrium.numerical_root_found:
        axes[1, 0].text(0.5, 0.08, "stiffness omitted for unresolved candidate", transform=axes[1, 0].transAxes,
                        ha="center", va="bottom", fontsize=8)
    axes[1, 0].axhline(0, color="0.4", lw=0.8)
    axes[1, 0].set_xticks(x, ("weak", "middle", "strong"))
    axes[1, 0].set_ylabel("Equilibrium-centered stiffness (N/m)")
    axes[1, 0].legend(frameon=False)
    set_panel_title(axes[1, 0], "Recentered stiffness spectra")

    axes[1, 1].semilogy(lvalues, np.maximum(load_scaled_change, 1e-14), marker="o", color="#F28E2B")
    axes[1, 1].set_xlabel("Partial-wave truncation lmax")
    axes[1, 1].set_ylabel(r"$\|\Delta F_{rad}\|$/effective weight")
    set_panel_title(axes[1, 1], "Partial-wave truncation-order check")
    for ax, label in zip(axes.ravel(), "abcd"):
        panel_label(ax, label)
    fig.suptitle(
        "Standard Gor'kov and finite-ka evaluations: local agreement and equilibrium offsets",
        fontsize=14.0, fontweight="semibold", y=0.995,
    )
    return _save_figure(ctx, fig, stem)


def _ensure_morphologies(ctx: PipelineContext) -> dict[str, Any]:
    if "morphologies" in ctx.memo:
        return ctx.memo["morphologies"]
    results: dict[str, Any] = {}
    rows = []
    family_weights = {
        "Vortex": np.diag([1.0, 1.0, 1.0]),
        "Twin": np.diag([1.0, 0.01, 0.01]),
        "Bottle": np.diag([0.01, 0.01, 1.0]),
    }
    run_indices = range(4 if ctx.config.full else 2)
    for name in ("Vortex", "Twin", "Bottle"):
        curvature_weight = family_weights[name]
        spec = MethodSpec(
            name="Force-Equilibrium", alpha_per_m=1000.0,
            beta_curvature_per_pa=0.0, pressure_mode="abs",
            curvature_weight=curvature_weight,
        )
        method_label = "FE"
        condition = Condition(
            label=f"appendix-{name.lower()}", positions=ctx.positions_m,
            target_m=tuple(MAIN_TARGET_M), frequency_hz=ctx.config.frequency_hz,
            curvature_weight=curvature_weight, normals=ctx.normals,
        )
        objective = SingleTargetObjective(condition, spec)
        morphology_maxiter = max(
            ctx.config.fe_maxiter,
            1200 if name == "Bottle" else ctx.config.fe_maxiter,
        )
        candidates = []
        for run_index in run_indices:
            shared_seed = ctx.config.random_seed + 1000 + run_index
            initial = np.random.default_rng(shared_seed).uniform(
                0.0, 2.0 * np.pi, ctx.positions_m.shape[0]
            )
            payload = {
                "name": name,
                "run_index": run_index,
                "initial_phase": digest_array(initial),
                "shared_seed": shared_seed,
                "method": asdict(spec),
                "maxiter": morphology_maxiter,
                "gtol": ctx.config.gtol,
                "target_m": MAIN_TARGET_M.tolist(),
                "frequency_hz": ctx.config.frequency_hz,
                "source_scale_pa_m_per_murata_unit": SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
                "stencil_m": ctx.config.stencil_m,
                "positions": digest_array(ctx.positions_m),
                "normals": digest_array(ctx.normals),
                "scientific_convention": STANDARD_FORMULA_TEXT,
                "field_model": "Murata-MA40S4S-global-front-only-v1",
                "pipeline_implementation": PIPELINE_IMPLEMENTATION,
            }
            solve, _, _ = ctx.cache.get_or_compute(
                "morphology_solve", payload,
                lambda objective=objective, initial=initial: solve_corrected_gorkov_fe(
                    objective, initial, maxiter=morphology_maxiter, gtol=ctx.config.gtol
                ), recompute=ctx.recompute,
            )
            wall_time = float(solve.history_wall_s[-1]) if len(solve.history_wall_s) else math.inf
            near_stationary = bool(solve.success or solve.gradient_norm <= 1.0e-5)
            candidates.append((run_index, shared_seed, solve, wall_time, near_stationary))
        candidates.sort(
            key=lambda item: (
                not item[4],
                item[2].gradient_norm,
                item[2].objective,
                item[0],
            )
        )
        selected_run, selected_seed, solve, selected_wall_time, selected_stationary = candidates[0]
        if ctx.config.full and not selected_stationary:
            raise RuntimeError(
                f"No stationary {name} representative was recovered from the four-start manuscript bank"
            )
        results[name] = {
            "phase": solve.phase_rad,
            "solve": solve,
            "initialization": f"shared cold-start bank; selected run {selected_run}, seed {selected_seed}",
            "method": method_label,
            "selected_stationary": selected_stationary,
        }
        for run_index, shared_seed, candidate, wall_time, near_stationary in candidates:
            rows.append({
                "morphology": name,
                "method": method_label,
                "run_index": run_index,
                "shared_cold_start_seed": shared_seed,
                "selected_representative": run_index == selected_run,
                "selection_rule": "stationarity, gradient norm, objective, run index",
                "objective": candidate.objective,
                "terminal_gradient_norm": candidate.gradient_norm,
                "iterations": candidate.iterations,
                "evaluations": candidate.evaluations,
                "wall_time_s": wall_time,
                "optimizer_success": candidate.success,
                "near_stationary": near_stationary,
                "curvature_weight_x": float(curvature_weight[0, 0]),
                "curvature_weight_y": float(curvature_weight[1, 1]),
                "curvature_weight_z": float(curvature_weight[2, 2]),
                "target_x_m": float(MAIN_TARGET_M[0]),
                "target_y_m": float(MAIN_TARGET_M[1]),
                "target_z_m": float(MAIN_TARGET_M[2]),
            })
    ctx.memo["morphologies"] = results
    ctx.tables["morphology_terminal_metrics"] = pd.DataFrame(rows)
    return results


def appendix_b_morphologies(ctx: PipelineContext) -> mpl.figure.Figure:
    stem = "Appendix_B1_morphologies"
    if stem in ctx.figures:
        return ctx.figures[stem]
    results = _ensure_morphologies(ctx)
    comparisons = {
        name: _exact_comparison(
            ctx, item["phase"], MAIN_TARGET_M, key=f"appendix-b-{name.lower()}"
        )
        for name, item in results.items()
    }
    ctx.tables["morphology_finite_ka_validation"] = pd.DataFrame(
        [_validation_row(f"{name} {results[name]['method']} endpoint", comparison)
         for name, comparison in comparisons.items()]
    )
    fig = plt.figure(figsize=(17.6, 7.0))
    outer = fig.add_gridspec(1, 3, wspace=0.24)
    family_axes: dict[str, dict[str, mpl.axes.Axes]] = {}
    sampled = {
        name: _pressure_sections(
            _array_field(ctx, item["phase"]),
            MAIN_TARGET_M,
            half_xy_m=0.075,
            half_z_m=0.050,
            samples=161 if ctx.config.full else 121,
        )
        for name, item in results.items()
    }
    for column, name in enumerate(("Vortex", "Twin", "Bottle")):
        inner = outer[column].subgridspec(
            2, 3,
            width_ratios=(1.0, 1.0, 0.042),
            height_ratios=(1.0, 1.0),
            wspace=0.28,
            hspace=0.30,
        )
        axes = {
            "core": fig.add_subplot(inner[0, 0], projection="3d"),
            "xy": fig.add_subplot(inner[0, 1]),
            "xz": fig.add_subplot(inner[1, 0]),
            "yz": fig.add_subplot(inner[1, 1]),
            "colorbar": fig.add_subplot(inner[:, 2]),
        }
        family_axes[name] = axes
        section = sampled[name]
        norm = shared_power_norm(
            [section["xy"], section["xz"], section["yz"]],
            lower=0,
            upper=99.0,
        )
        axis = 1e3 * section["xy_axis_m"]; zaxis = 1e3 * section["z_axis_m"]
        equilibrium_delta = 1e3 * (
            comparisons[name].finite_ka.equilibrium.equilibrium_m - MAIN_TARGET_M
        )
        resolved = comparisons[name].finite_ka.equilibrium.numerical_root_found
        field = _array_field(ctx, results[name]["phase"])
        _morphology_core_cloud(axes["core"], field, MAIN_TARGET_M, full=ctx.config.full)
        set_panel_title(axes["core"], "3-D target-centred core")
        image = field_panel(
            axes["xy"], section["xy"], (axis[0], axis[-1], axis[0], axis[-1]), norm=norm,
            xlabel=r"$x-x^*$ (mm)", ylabel=r"$y-y^*$ (mm)", target_mm=(0, 0),
            equilibrium_mm=(float(equilibrium_delta[0]), float(equilibrium_delta[1])) if resolved else None,
        )
        field_panel(
            axes["xz"], section["xz"], (axis[0], axis[-1], zaxis[0], zaxis[-1]), norm=norm,
            xlabel=r"$x-x^*$ (mm)", ylabel=r"$z-z^*$ (mm)", target_mm=(0, 0),
            equilibrium_mm=(float(equilibrium_delta[0]), float(equilibrium_delta[2])) if resolved else None,
        )
        field_panel(
            axes["yz"], section["yz"], (axis[0], axis[-1], zaxis[0], zaxis[-1]), norm=norm,
            xlabel=r"$y-y^*$ (mm)", ylabel=r"$z-z^*$ (mm)", target_mm=(0, 0),
            equilibrium_mm=(float(equilibrium_delta[1]), float(equilibrium_delta[2])) if resolved else None,
        )
        if not resolved:
            axes["xy"].scatter(equilibrium_delta[0], equilibrium_delta[1], marker="^", facecolor="none", edgecolor="white", s=46)
            axes["xz"].scatter(equilibrium_delta[0], equilibrium_delta[2], marker="^", facecolor="none", edgecolor="white", s=46)
            axes["yz"].scatter(equilibrium_delta[1], equilibrium_delta[2], marker="^", facecolor="none", edgecolor="white", s=46)
            axes["xz"].text(0.98, 0.02, "unresolved candidate", transform=axes["xz"].transAxes,
                                 ha="right", va="bottom", fontsize=7.3, color="white")
        terminal_label = "endpoint" if results[name]["selected_stationary"] else "nonstationary smoke candidate"
        set_panel_title(axes["xy"], "XY full field")
        set_panel_title(axes["xz"], "XZ full field")
        set_panel_title(axes["yz"], "YZ full field")
        panel_label(axes["core"], chr(ord("a") + column))
        axes["core"].text2D(
            0.5, 1.14,
            f"{name} {results[name]['method']} {terminal_label}",
            transform=axes["core"].transAxes,
            ha="left", va="bottom", fontsize=11.0, fontweight="semibold",
        )
        colorbar = fig.colorbar(
            image,
            cax=axes["colorbar"],
        )
        colorbar.locator = mpl.ticker.MaxNLocator(nbins=5)
        colorbar.update_ticks()
        colorbar.ax.tick_params(labelsize=7.5)
        colorbar.set_label(r"$|p|$ (Pa)")
    fig.suptitle(
        "Canonical morphology objectives from a shared cold-start bank, independently evaluated",
        fontsize=14.0, fontweight="semibold", y=0.990,
    )
    fig.subplots_adjust(left=0.035, right=0.990, top=0.88, bottom=0.08)
    return _save_figure(ctx, fig, stem, tight=False)


@dataclass(frozen=True)
class _IntrinsicArrayFrame:
    landmark_states: np.ndarray
    coordinates: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    squared_column_mean: np.ndarray
    squared_grand_mean: float
    rotation: np.ndarray

    def project(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.complex128)
        similarity = np.clip(np.abs(values @ self.landmark_states.conj().T), 0.0, 1.0)
        squared = np.maximum(0.0, 1.0 - similarity**2)
        cross = -0.5 * (
            squared - squared.mean(axis=1, keepdims=True)
            - self.squared_column_mean[None, :] + self.squared_grand_mean
        )
        raw = cross @ self.eigenvectors / np.sqrt(self.eigenvalues)[None, :]
        return raw @ self.rotation


@dataclass(frozen=True)
class _IntrinsicWebFrame3D:
    landmark_states: np.ndarray
    coordinates: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    squared_column_mean: np.ndarray
    squared_grand_mean: float
    rotation: np.ndarray
    intrinsic_dimensions: int

    def project(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.complex128)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        similarity = np.clip(np.abs(values @ self.landmark_states.conj().T), 0.0, 1.0)
        squared = np.maximum(0.0, 1.0 - similarity**2)
        cross = -0.5 * (
            squared - squared.mean(axis=1, keepdims=True)
            - self.squared_column_mean[None, :] + self.squared_grand_mean
        )
        raw = cross @ self.eigenvectors / np.sqrt(self.eigenvalues)[None, :]
        padded = np.zeros((len(raw), 3), dtype=float)
        padded[:, : self.intrinsic_dimensions] = raw
        return padded @ self.rotation


def _fit_intrinsic_web_frame_3d(
    nodes: pd.DataFrame,
    record_lookup: Mapping[str, Any],
    *,
    allow_degenerate_smoke_frame: bool = False,
) -> _IntrinsicWebFrame3D:
    if len(nodes) < 2:
        raise ValueError("at least two recovered nodes are needed for a web frame")
    if len(nodes) < 4 and not allow_degenerate_smoke_frame:
        raise ValueError("at least four recovered nodes are needed for a 3-D web frame")
    states = np.stack(
        [np.asarray(record_lookup[str(record_id)].state, dtype=np.complex128) for record_id in nodes["record_id"]]
    )
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    distance = pairwise_projective_distance(states)
    squared = distance**2
    count = len(states)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(values)[::-1]
    positive = values[order] > np.finfo(float).eps
    intrinsic_dimensions = min(3, int(np.count_nonzero(positive)))
    if intrinsic_dimensions < 1:
        raise ValueError("recovered web has no positive MDS axis")
    if intrinsic_dimensions < 3 and not allow_degenerate_smoke_frame:
        raise ValueError("recovered web does not support three positive MDS axes")
    chosen = order[positive][:intrinsic_dimensions]
    eigenvalues = np.asarray(values[chosen], dtype=float)
    eigenvectors = np.asarray(vectors[:, chosen], dtype=float)
    raw = eigenvectors * np.sqrt(eigenvalues)[None, :]
    padded = np.zeros((len(raw), 3), dtype=float)
    padded[:, :intrinsic_dimensions] = raw
    component_names = sorted(str(value) for value in nodes["component"].dropna().unique())
    component_coordinate = {
        name: value for name, value in zip(
            component_names,
            np.linspace(-1.0, 1.0, len(component_names)) if len(component_names) > 1 else [0.0],
        )
    }
    reference = np.column_stack(
        [
            nodes["chart_u"].to_numpy(dtype=float),
            nodes["chart_v"].to_numpy(dtype=float),
            nodes["component"].map(component_coordinate).to_numpy(dtype=float),
        ]
    )
    reference -= reference.mean(axis=0, keepdims=True)
    reference *= np.linalg.norm(padded) / max(float(np.linalg.norm(reference)), 1.0e-30)
    rotation, _ = orthogonal_procrustes(padded, reference)
    return _IntrinsicWebFrame3D(
        landmark_states=states,
        coordinates=padded @ rotation,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        squared_column_mean=squared.mean(axis=0),
        squared_grand_mean=float(squared.mean()),
        rotation=rotation,
        intrinsic_dimensions=intrinsic_dimensions,
    )


def _component_display_color(component: str) -> str:
    if component in COLORS:
        return COLORS[component]
    index = max(0, int(component[1:]) - 1) if component.startswith("B") and component[1:].isdigit() else 0
    return mpl.colors.to_hex(plt.get_cmap("tab10")(index % 10))


def _plot_intrinsic_web_3d(
    ax: Any,
    nodes: pd.DataFrame,
    coordinates: np.ndarray,
    *,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    table = nodes.copy().reset_index(drop=True)
    table[["web_1", "web_2", "web_3"]] = np.asarray(coordinates, dtype=float)
    dimensions = ("web_1", "web_3", "web_2")
    for component, group in table.groupby("component", sort=True):
        color = _component_display_color(str(component))
        lookup = {
            (int(row.grid_i), int(row.grid_j)): np.asarray(
                [getattr(row, name) for name in dimensions], dtype=float
            )
            for row in group.itertuples(index=False)
        }
        for node, start in lookup.items():
            for neighbor in ((node[0] + 1, node[1]), (node[0], node[1] + 1)):
                if neighbor in lookup:
                    segment = np.stack([start, lookup[neighbor]])
                    ax.plot(*segment.T, color="white", lw=2.0, alpha=0.82, zorder=2)
                    ax.plot(*segment.T, color=color, lw=1.25, alpha=0.94, zorder=3)
        values = group.loc[:, dimensions].to_numpy(dtype=float)
        ax.scatter(
            *values.T, s=28, color=color, edgecolor="white", linewidth=0.65,
            depthshade=False, label=str(component), zorder=4,
        )
    if limits is None:
        values = table.loc[:, dimensions].to_numpy(dtype=float)
        limits = tuple(
            (
                float(values[:, index].min() - max(0.06 * np.ptp(values[:, index]), 0.004)),
                float(values[:, index].max() + max(0.06 * np.ptp(values[:, index]), 0.004)),
            )
            for index in range(3)
        )  # type: ignore[assignment]
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.set_xlabel("Solution 1", fontsize=8.0, labelpad=3)
    ax.set_ylabel("Family", fontsize=8.0, labelpad=3)
    ax.set_zlabel("Solution 2", fontsize=8.0, labelpad=3)
    ax.tick_params(labelsize=6.8, pad=1)
    ax.view_init(elev=24, azim=-64)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.05, 0.78, 0.95))


def _fit_intrinsic_array_frame(bank: EndpointBank) -> _IntrinsicArrayFrame:
    states = np.stack([record.state for record in bank.records]).astype(np.complex128)
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    distance = pairwise_projective_distance(states)
    squared = distance**2
    n = len(states)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(values)[::-1]
    eigenvalues = values[order][:2]
    eigenvectors = vectors[:, order][:, :2]
    if np.any(eigenvalues <= np.finfo(float).eps):
        raise ValueError("Array endpoint bank has fewer than two intrinsic projective directions")
    raw = eigenvectors * np.sqrt(eigenvalues)[None, :]
    physical = np.array([
        record.request.target.chart_coordinate for record in bank.records
    ], dtype=float)
    physical -= physical.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(raw, physical)
    return _IntrinsicArrayFrame(
        landmark_states=states, coordinates=raw @ rotation,
        eigenvectors=eigenvectors, eigenvalues=eigenvalues,
        squared_column_mean=squared.mean(axis=0),
        squared_grand_mean=float(squared.mean()), rotation=rotation,
    )


def _ensure_array_sweep(ctx: PipelineContext) -> tuple[EndpointBank, dict[str, FixedFrameProjection]]:
    if "array_sweep" in ctx.memo:
        return ctx.memo["array_sweep"]
    all_layouts = standard_array_ensemble("full", pitch_m=ctx.config.pitch_m)
    if ctx.config.full:
        layouts = all_layouts
        mode = "full"
        restarts = 12
    else:
        keep = ("square", "rectangular", "fermat-disk")
        layouts = {key: all_layouts[key] for key in keep}
        mode = "quick"
        restarts = 4
    cache_dir = None if ctx.recompute else ctx.output_root / "cache" / "endpoint_sweeps"
    bank = run_array_endpoint_bank(
        _sweep_solver, _sweep_contract(), arrays=layouts, targets=_sweep_targets(ctx),
        alpha=1000.0, frequency_hz=ctx.config.frequency_hz, mode=mode,
        restarts=restarts,
        solver_options={"maxiter": ctx.config.sweep_maxiter, "gtol": ctx.config.gtol, "stationarity_tol": 1.0e-5},
        cache_dir=cache_dir,
        cache_only=ctx.config.cache_only,
    )
    banks = split_bank_by_array(bank)
    projections: dict[str, FixedFrameProjection] = {}
    endpoint_tables = []
    coverage_tables = []
    ctx.tables["array_sweep_all_solver_rows"] = pd.DataFrame(bank.rows())
    for array_id, subbank in banks.items():
        analysis_bank = _successful_endpoint_bank(
            subbank,
            smoke_stationarity_tolerance=None if ctx.config.full else 1.0e-4,
        )
        if len(analysis_bank.records) < 2:
            projection = _empty_projection(subbank, "fewer than two successful stationary endpoints")
            projection = _restore_tested_cell_denominator(subbank, projection)
            projections[array_id] = projection
            endpoint_tables.append(pd.DataFrame(projection.endpoint_rows))
            coverage_tables.append(pd.DataFrame(projection.coverage_rows))
            continue
        frame = _fit_intrinsic_array_frame(analysis_bank)
        try:
            projection = project_endpoint_bank_fixed_frame(
                analysis_bank, frame, anchor_selector=lambda record: True,
                component_distance_threshold=0.28,
                minimum_anchor_component_size=max(4, len(_sweep_targets(ctx)) // 2),
                assignment_threshold=0.40, minimum_replicates_per_component=2,
            )
        except ValueError as error:
            coordinates = frame.project(np.stack([record.state for record in analysis_bank.records]))
            endpoint_rows = [
                record.as_row() | {
                    "embedding_1": float(coordinates[index, 0]),
                    "embedding_2": float(coordinates[index, 1]),
                    "component": None, "assigned": False,
                    "component_distance": math.nan, "component_margin": math.nan,
                    "component_gate_failure": str(error),
                }
                for index, record in enumerate(analysis_bank.records)
            ]
            coverage_rows = []
            for target_id, group in pd.DataFrame(endpoint_rows).groupby("target_id"):
                row = group.iloc[0]
                coverage_rows.append({
                    "axis_value": row["axis_value"], "array_id": row["array_id"],
                    "frequency_hz": row["frequency_hz"], "alpha": row["alpha"],
                    "target_id": target_id, "grid_i": row["grid_i"], "grid_j": row["grid_j"],
                    "chart_u": row["chart_u"], "chart_v": row["chart_v"],
                    "n_endpoints": len(group), "n_successful": int(group["success"].sum()),
                    "n_solver_failures": int((~group["success"].astype(bool)).sum()),
                    "n_assigned": 0, "n_unassigned": len(group),
                    "recovered_components": "", "component_coverage_fraction": 0.0,
                    "complete_reference_coverage": False,
                })
            projection = FixedFrameProjection(endpoint_rows, coverage_rows, [], [], ())
        projection = _restore_tested_cell_denominator(subbank, projection)
        projections[array_id] = projection
        endpoint_tables.append(pd.DataFrame(projection.endpoint_rows))
        coverage_tables.append(pd.DataFrame(projection.coverage_rows))
    ctx.memo["array_sweep"] = (bank, projections)
    ctx.tables["array_sweep_endpoints"] = pd.concat(endpoint_tables, ignore_index=True)
    ctx.tables["array_sweep_coverage"] = pd.concat(coverage_tables, ignore_index=True)
    component_counts = {key: len(value.component_labels) for key, value in projections.items()}
    comparable = {
        key for key, count in component_counts.items()
        if count == component_counts.get("square", -1)
    }
    if "square" in comparable and component_counts["square"] > 0:
        ctx.tables["array_intrinsic_geometry_comparison"] = pd.DataFrame(
            compare_array_intrinsic_geometry(
                {key: banks[key] for key in comparable},
                {key: projections[key] for key in comparable},
                reference_array="square",
            )
        )
    else:
        ctx.tables["array_intrinsic_geometry_comparison"] = pd.DataFrame(
            [{"array_id": key, "component_count": count, "comparison_available": False}
             for key, count in component_counts.items()]
        )
    ctx.tables["array_component_gate_protocol"] = pd.DataFrame([{
        "independent_cold_starts_per_cell": restarts,
        "projective_link_threshold": 0.28,
        "assignment_threshold": 0.40,
        "minimum_global_component_support": max(4, len(_sweep_targets(ctx)) // 2),
        "minimum_successful_repeats_per_cell_component": 2,
        "components_may_be_missing": True,
    }])
    return bank, projections


def _plot_projection_category(
    ax: plt.Axes,
    projection: FixedFrameProjection,
    *,
    column: str,
    value: Any,
    title: str,
) -> None:
    edges = pd.DataFrame(projection.edge_rows)
    nodes = pd.DataFrame(projection.node_rows)
    if edges.empty or nodes.empty:
        ax.text(0.5, 0.5, "No recovered endpoints", ha="center", va="center", transform=ax.transAxes)
        set_panel_title(ax, title)
        return
    if column == "array_id":
        edge_mask = edges[column].astype(str).eq(str(value))
        node_mask = nodes[column].astype(str).eq(str(value))
    else:
        edge_mask = np.isclose(pd.to_numeric(edges[column]), float(value))
        node_mask = np.isclose(pd.to_numeric(nodes[column]), float(value))
    for row in edges.loc[edge_mask].itertuples(index=False):
        ax.plot([row.x0, row.x1], [row.y0, row.y1], color=COLORS.get(row.component, "0.5"), lw=0.7, alpha=0.42)
    for component in projection.component_labels:
        group = nodes.loc[node_mask & nodes["component"].eq(component)]
        color = COLORS.get(component, plt.get_cmap("tab10")((int(component[1:]) - 1) % 10))
        ax.scatter(group["embedding_1"], group["embedding_2"], s=23,
                   color=color, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Fixed FE coordinate 1")
    ax.set_ylabel("Fixed FE coordinate 2")
    set_panel_title(ax, title)


def appendix_c_arrays(ctx: PipelineContext) -> mpl.figure.Figure:
    stem = "Appendix_C1_arrays"
    if stem in ctx.figures:
        return ctx.figures[stem]
    bank, projections = _ensure_array_sweep(ctx)
    array_ids = tuple(bank.design["array_ids"])
    columns = min(3, len(array_ids))
    rows = int(math.ceil(len(array_ids) / columns))
    fig = plt.figure(figsize=(5.1 * columns, 4.7 * rows))
    gs = fig.add_gridspec(rows, columns, hspace=0.20, wspace=0.12)
    endpoint_rows = ctx.tables["array_sweep_endpoints"]
    coverage_rows = ctx.tables["array_sweep_coverage"]
    subbanks = split_bank_by_array(bank)
    summary_rows = []
    displayed_components: set[str] = set()
    for index, array_id in enumerate(array_ids):
        ax = fig.add_subplot(gs[index // columns, index % columns], projection="3d")
        runtime = endpoint_rows.loc[endpoint_rows["array_id"].eq(array_id), "wall_time_s"].median()
        selected_coverage = coverage_rows[coverage_rows["array_id"].eq(array_id)]
        complete = float(selected_coverage["complete_reference_coverage"].mean()) if len(selected_coverage) else math.nan
        nodes = pd.DataFrame(projections[array_id].node_rows)
        displayed_components.update(str(value) for value in projections[array_id].component_labels)
        if len(nodes) < 2:
            ax.text2D(
                0.5, 0.5,
                "Insufficient quick-smoke recovery" if not ctx.config.full else "No repeatable web",
                transform=ax.transAxes, ha="center", va="center",
            )
        else:
            records = {record.record_id: record for record in subbanks[array_id].records}
            frame = _fit_intrinsic_web_frame_3d(
                nodes,
                records,
                allow_degenerate_smoke_frame=not ctx.config.full,
            )
            _plot_intrinsic_web_3d(ax, nodes, frame.coordinates)
            if frame.intrinsic_dimensions < 3:
                ax.text2D(
                    0.02, 0.04,
                    f"quick smoke: {frame.intrinsic_dimensions}-D recovered span",
                    transform=ax.transAxes, fontsize=7.0,
                )
        if ctx.config.full:
            panel_title = f"{array_id} | complete {complete:.0%} | {runtime:.2f} s median"
        else:
            panel_title = f"{array_id} | {len(nodes)} web nodes | quick smoke"
        set_panel_title(ax, panel_title)
        panel_label(ax, chr(ord("a") + index))
        summary_rows.append(
            {
                "array_id": array_id,
                "median_wall_time_s": runtime,
                "complete_cell_fraction": complete,
                "recovered_components": ",".join(projections[array_id].component_labels),
            }
        )
    ctx.tables["appendix_array_web_summary"] = pd.DataFrame(summary_rows)
    if displayed_components:
        handles = [
            mpl.lines.Line2D(
                [], [], marker="o", linestyle="none", markersize=5.5,
                markerfacecolor=_component_display_color(component),
                markeredgecolor="white", label=component,
            )
            for component in sorted(displayed_components)
        ]
        fig.legend(handles=handles, frameon=False, fontsize=7.8, loc="upper right")
    fig.suptitle(
        "Array-intrinsic three-dimensional FE webs across matched 256-element geometries",
        fontsize=14.0, fontweight="semibold", y=0.985,
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.03)
    return _save_figure(ctx, fig, stem, tight=False)


def _ensure_frequency_sweep(ctx: PipelineContext) -> tuple[EndpointBank, FixedFrameProjection]:
    if "frequency_sweep" in ctx.memo:
        return ctx.memo["frequency_sweep"]
    mode = "full" if ctx.config.full else "quick"
    cache_dir = None if ctx.recompute else ctx.output_root / "cache" / "endpoint_sweeps"
    bank = run_frequency_endpoint_bank(
        _sweep_solver, _sweep_contract(), positions_m=ctx.positions_m,
        targets=_sweep_targets(ctx), frequencies_hz=default_frequencies_hz(mode),
        alpha=1000.0, array_id="square-16x16", mode=mode,
        restarts=12 if ctx.config.full else 4,
        solver_options={"maxiter": ctx.config.sweep_maxiter, "gtol": ctx.config.gtol, "stationarity_tol": 1.0e-5},
        cache_dir=cache_dir,
        cache_only=ctx.config.cache_only,
    )
    analysis_bank = _successful_endpoint_bank(
        bank,
        smoke_stationarity_tolerance=None if ctx.config.full else 1.0e-4,
    )
    projection = (
        project_endpoint_bank_fixed_frame(
            analysis_bank, ctx.joint_frame, component_bank=ctx.branch_bank,
            assignment_threshold=0.48, minimum_replicates_per_component=2,
        )
        if analysis_bank.records else _empty_projection(bank, "no successful stationary endpoint")
    )
    projection = _restore_tested_cell_denominator(bank, projection)
    ctx.memo["frequency_sweep"] = (bank, projection)
    ctx.tables["frequency_sweep_endpoints"] = pd.DataFrame(projection.endpoint_rows)
    ctx.tables["frequency_sweep_all_solver_rows"] = pd.DataFrame(bank.rows())
    ctx.tables["frequency_sweep_coverage"] = pd.DataFrame(projection.coverage_rows)
    return bank, projection


def appendix_c_frequencies(ctx: PipelineContext) -> mpl.figure.Figure:
    stem = "Appendix_C2_frequencies"
    if stem in ctx.figures:
        return ctx.figures[stem]
    bank, projection = _ensure_frequency_sweep(ctx)
    frequencies = tuple(float(value) for value in bank.design["frequencies_hz"])
    fig = plt.figure(figsize=(5.0 * len(frequencies), 4.7))
    gs = fig.add_gridspec(1, len(frequencies), wspace=0.13)
    endpoint_rows = pd.DataFrame(projection.endpoint_rows)
    coverage_rows = pd.DataFrame(projection.coverage_rows)
    nodes = pd.DataFrame(projection.node_rows)
    record_lookup = {record.record_id: record for record in bank.records}
    nominal = min(frequencies, key=lambda value: abs(value - ctx.config.frequency_hz))
    nominal_nodes = nodes[np.isclose(pd.to_numeric(nodes["frequency_hz"]), nominal)].copy()
    frame_frequency = nominal
    if len(nominal_nodes) < 2 and not ctx.config.full:
        counts = nodes.groupby("frequency_hz").size() if not nodes.empty else pd.Series(dtype=int)
        if not counts.empty and int(counts.max()) >= 2:
            frame_frequency = float(counts.idxmax())
            nominal_nodes = nodes[
                np.isclose(pd.to_numeric(nodes["frequency_hz"]), frame_frequency)
            ].copy()
    frame = _fit_intrinsic_web_frame_3d(
        nominal_nodes,
        record_lookup,
        allow_degenerate_smoke_frame=not ctx.config.full,
    )
    projected_by_frequency: dict[float, tuple[pd.DataFrame, np.ndarray]] = {}
    all_display = []
    for frequency in frequencies:
        selected_nodes = nodes[np.isclose(pd.to_numeric(nodes["frequency_hz"]), frequency)].copy()
        if selected_nodes.empty:
            coordinates = np.empty((0, 3), dtype=float)
        else:
            states = np.stack([record_lookup[str(value)].state for value in selected_nodes["record_id"]])
            coordinates = frame.project(states)
        projected_by_frequency[frequency] = (selected_nodes, coordinates)
        if len(coordinates):
            all_display.append(coordinates[:, [0, 2, 1]])
    display_values = np.vstack(all_display)
    common_limits = tuple(
        (
            float(display_values[:, index].min() - max(0.06 * np.ptp(display_values[:, index]), 0.004)),
            float(display_values[:, index].max() + max(0.06 * np.ptp(display_values[:, index]), 0.004)),
        )
        for index in range(3)
    )
    summary_rows = []
    for index, frequency in enumerate(frequencies):
        ax = fig.add_subplot(gs[0, index], projection="3d")
        runtime = endpoint_rows.loc[np.isclose(endpoint_rows["frequency_hz"], frequency), "wall_time_s"].median()
        selected_coverage = coverage_rows[np.isclose(pd.to_numeric(coverage_rows["frequency_hz"]), frequency)]
        complete = float(selected_coverage["complete_reference_coverage"].mean()) if len(selected_coverage) else math.nan
        selected_nodes, coordinates = projected_by_frequency[frequency]
        _plot_intrinsic_web_3d(ax, selected_nodes, coordinates, limits=common_limits)
        if frame.intrinsic_dimensions < 3:
            ax.text2D(
                0.02, 0.04,
                f"quick smoke: {frame.intrinsic_dimensions}-D recovered span",
                transform=ax.transAxes, fontsize=7.0,
            )
        if ctx.config.full:
            panel_title = f"{frequency / 1000:.0f} kHz | complete {complete:.0%} | {runtime:.2f} s"
        else:
            panel_title = f"{frequency / 1000:.0f} kHz | {len(selected_nodes)} web nodes | quick smoke"
        set_panel_title(ax, panel_title)
        if index == len(frequencies) - 1:
            ax.legend(frameon=False, fontsize=7.4, loc="upper right")
        panel_label(ax, chr(ord("a") + index))
        summary_rows.append(
            {
                "frequency_hz": frequency,
                "median_wall_time_s": runtime,
                "complete_cell_fraction": complete,
                "frame_fit_frequency_hz": frame_frequency,
            }
        )
    ctx.tables["appendix_frequency_web_summary"] = pd.DataFrame(summary_rows)
    fig.suptitle(
        "Frequency-indexed FE webs in one frame fixed at the nominal frequency",
        fontsize=14.0, fontweight="semibold", y=0.985,
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.89, bottom=0.03)
    return _save_figure(ctx, fig, stem, tight=False)


def appendix_d_full_evolution(ctx: PipelineContext) -> mpl.figure.Figure:
    stem = "Appendix_D1_full_evolution"
    if stem in ctx.figures:
        return ctx.figures[stem]
    branch = _rigorous_branch_evolution(ctx)
    checkpoints = [int(value) for value in ctx.artifact.checkpoints]
    fig = plt.figure(figsize=(15.0, 8.0))
    gs = fig.add_gridspec(2, len(checkpoints), hspace=0.06, wspace=0.06)
    panel_index = 0
    for row, component in enumerate(COMPONENTS):
        combined = pd.concat(
            [
                branch["reference"][branch["reference"]["component"].eq(component)],
                branch["evolution"][branch["evolution"]["component"].eq(component)],
            ],
            ignore_index=True,
        )
        display = combined[["coordinate_1", "coordinate_3", "coordinate_2"]].to_numpy(dtype=float)
        limits = tuple(
            (
                float(display[:, index].min() - max(0.055 * np.ptp(display[:, index]), 0.004)),
                float(display[:, index].max() + max(0.055 * np.ptp(display[:, index]), 0.004)),
            )
            for index in range(3)
        )
        for column, checkpoint in enumerate(checkpoints):
            ax = fig.add_subplot(gs[row, column], projection="3d")
            _draw_branch_sheet_3d(
                ax,
                branch["reference"],
                branch["evolution"],
                ctx.branch_bank.edge_table,
                component,
                iteration=checkpoint,
            )
            ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
            if row == 0:
                set_panel_title(ax, f"{checkpoint:,} Conventional iterations")
            else:
                ax.set_title("")
            if row == 0 and column == len(checkpoints) - 1:
                ax.legend(handles=_branch_method_legend(), frameon=False, fontsize=7.3, loc="upper right")
            panel_label(ax, chr(ord("a") + panel_index))
            panel_index += 1

    distance_table = branch["evolution"][
        ["component", "iteration", "distance_to_target_fe_rh"]
    ].copy()
    summary_rows = []
    regression_rows = []
    for component, group in distance_table.groupby("component", sort=True):
        medians = []
        for iteration, stage in group.groupby("iteration", sort=True):
            values = stage["distance_to_target_fe_rh"].to_numpy(dtype=float)
            median = float(np.median(values))
            medians.append((float(iteration), median))
            summary_rows.append(
                {
                    "component": component,
                    "iteration": int(iteration),
                    "median_projector_distance": median,
                    "q25_projector_distance": float(np.quantile(values, 0.25)),
                    "q75_projector_distance": float(np.quantile(values, 0.75)),
                }
            )
        x = np.log([value[0] for value in medians])
        y = np.log([max(value[1], 1.0e-30) for value in medians])
        slope, intercept = np.polyfit(x, y, 1)
        regression_rows.append(
            {
                "component": component,
                "log_log_slope": float(slope),
                "log_log_intercept": float(intercept),
                "fit_points": len(medians),
            }
        )
    ctx.tables["appendix_branch_evolution_distance_summary"] = pd.DataFrame(summary_rows)
    ctx.tables["appendix_branch_evolution_regression"] = pd.DataFrame(regression_rows)
    fig.text(0.014, 0.67, "B1", rotation=90, va="center", ha="center",
             fontsize=11, fontweight="semibold", color=COLORS["B1"])
    fig.text(0.014, 0.25, "B2", rotation=90, va="center", ha="center",
             fontsize=11, fontweight="semibold", color=COLORS["B2"])
    fig.suptitle(
        "The complete Conventional web approaches the same fixed FE/RH sheets without frame refitting",
        fontsize=14.0, fontweight="semibold", y=0.985,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.91, bottom=0.02)
    return _save_figure(ctx, fig, stem, tight=False)


def appendix_f_multitrap_ablation(ctx: PipelineContext) -> mpl.figure.Figure:
    stem = "Appendix_F1_multitrap_ablation"
    if stem in ctx.figures:
        return ctx.figures[stem]
    bundle = _ensure_multitrap(ctx, include_ablation=True)
    problem: MultiTrapProblem = bundle["problem"]
    results: list[MultitrapRunResult] = bundle["results"]
    full_result = next(result for result in results if result.ablation_id == "S1P1U1")
    diagnostic_config = regularized_fe_double_trap_config(
        stage_maxiters=_multitrap_schedule(ctx), gtol=ctx.config.gtol
    )
    diagnostic_force_epsilon = full_result.stages[-1].force_epsilon
    diagnostic_uniformity_epsilon = full_result.stages[-1].uniformity_epsilon
    records = []
    for result in results:
        # Re-evaluate every terminal command with one fixed full S/P/U
        # diagnostic.  Comparing each variant's native local loss would be
        # invalid because toggling S/P/U changes the objective itself.
        diagnostic = evaluate_multitrap_objective(
            problem,
            result.phases,
            diagnostic_config,
            force_epsilon=diagnostic_force_epsilon,
            uniformity_epsilon=diagnostic_uniformity_epsilon,
        )
        force = max(row.force_norm_n_arb for row in result.per_target)
        pressure = min(row.pressure_abs_pa_arb for row in result.per_target)
        records.append({
            "ablation": result.ablation_id, "S": int(result.components.force_smoothing),
            "P": int(result.components.pressure_retention), "U": int(result.components.uniformity),
            "worst_force_norm": force, "weakest_pressure": pressure,
            "wall_time_s": result.end_to_end_sec, "terminal_gradient_norm": result.terminal_gradient_norm,
            "common_full_FE_loss_imbalance": float(np.ptp(diagnostic.local_losses)),
        })
    table = pd.DataFrame(records).sort_values(["S", "P", "U"], ascending=False).reset_index(drop=True)
    ctx.tables["multitrap_spu_compact_metrics"] = table
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(len(table))
    colors = [COLORS["regularized FE"] if label == "S1P1U1" else "0.65" for label in table["ablation"]]
    axes[0].bar(x, table["worst_force_norm"], color=colors)
    axes[0].set_xticks(x, table["ablation"], rotation=45, ha="right")
    axes[0].set_ylabel("Worst-target Gor'kov force residual (N, arb. drive)")
    set_panel_title(axes[0], "Single-seed tri3 S/P/U diagnostic")
    axes[1].scatter(table["common_full_FE_loss_imbalance"], table["weakest_pressure"], c=colors, s=65, edgecolor="white")
    annotation_rows = list(table.itertuples(index=False))
    x_values = np.asarray([float(row.common_full_FE_loss_imbalance) for row in annotation_rows])
    y_values = np.asarray([float(row.weakest_pressure) for row in annotation_rows])
    x_span = max(float(np.ptp(x_values)), 1.0)
    y_span = max(float(np.ptp(y_values)), 1.0)
    clusters: list[list[Any]] = []
    for row in annotation_rows:
        x_value = float(row.common_full_FE_loss_imbalance)
        y_value = float(row.weakest_pressure)
        for cluster in clusters:
            center_x = np.mean([float(item.common_full_FE_loss_imbalance) for item in cluster])
            center_y = np.mean([float(item.weakest_pressure) for item in cluster])
            if abs(x_value - center_x) <= 0.055 * x_span and abs(y_value - center_y) <= 0.055 * y_span:
                cluster.append(row)
                break
        else:
            clusters.append([row])
    for cluster in clusters:
        center_y = np.mean([float(item.weakest_pressure) for item in cluster])
        if center_y < float(y_values.min()) + 0.12 * y_span:
            offsets = 8.0 + 13.0 * np.arange(len(cluster))
        elif center_y > float(y_values.max()) - 0.12 * y_span:
            offsets = -(8.0 + 13.0 * np.arange(len(cluster)))
        else:
            offsets = 13.0 * (np.arange(len(cluster)) - 0.5 * (len(cluster) - 1))
        for row, y_offset in zip(cluster, offsets):
            axes[1].annotate(
                row.ablation,
                (row.common_full_FE_loss_imbalance, row.weakest_pressure),
                xytext=(6, float(y_offset)),
                textcoords="offset points",
                fontsize=7.4,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=0.4),
                annotation_clip=False,
            )
    axes[1].set_xlabel("Between-target loss imbalance under one full-FE diagnostic")
    axes[1].set_ylabel("Weakest-target pressure amplitude (Pa, arb. drive)")
    set_panel_title(axes[1], "Uniformity and pressure retention consequences")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.suptitle(
        "Tri3 regularization ablation under one common post-hoc standard-Gor'kov diagnostic",
        fontsize=14.0, fontweight="semibold", y=1.01,
    )
    return _save_figure(ctx, fig, stem)


def appendix_g_timing_reproduction(ctx: PipelineContext) -> mpl.figure.Figure:
    stem = "Appendix_G1_timing_reproduction"
    if stem in ctx.figures:
        return ctx.figures[stem]
    _ensure_sota(ctx)
    table = ctx.tables["engineering_baseline_timing"].copy()
    ctx.tables["complete_timing_protocol"] = table
    plot_table = (
        table.groupby(["method", "initialization"], as_index=False)
        .median(numeric_only=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": (1.35, 1.0)})
    fields = [
        "initialization_s", "transfer_s", "compilation_s",
        "pre_solve_synchronization_s", "solve_s", "post_solve_synchronization_s",
    ]
    labels = plot_table["method"].astype(str) + "\n" + plot_table["initialization"].astype(str)
    bottom = np.zeros(len(plot_table))
    palette = plt.get_cmap("tab20c")(np.linspace(0.1, 0.9, len(fields)))
    for color, field_name in zip(palette, fields):
        values = plot_table[field_name].fillna(0).to_numpy(float)
        axes[0].barh(np.arange(len(plot_table)), values, left=bottom, color=color, label=field_name.removesuffix("_s"))
        bottom += values
    axes[0].set_yticks(np.arange(len(plot_table)), labels)
    axes[0].set_xlabel("Time (s)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    set_panel_title(axes[0], "Complete timing interval")

    stages = [
        ("1", "formula guard"), ("2", "corrected endpoint data"),
        ("3", "cached computation"), ("4", "figures + tables"), ("5", "hash manifest"),
    ]
    for index, (number, label) in enumerate(stages):
        y = len(stages) - 1 - index
        axes[1].add_patch(mpl.patches.FancyBboxPatch(
            (0.08, y - 0.28), 0.84, 0.56, boxstyle="round,pad=0.02",
            facecolor="#F3F4F6", edgecolor="#9CA3AF",
        ))
        axes[1].text(0.14, y, number, va="center", ha="center", weight="bold", color="#374151")
        axes[1].text(0.24, y, label, va="center", ha="left")
        if index < len(stages) - 1:
            axes[1].annotate("", xy=(0.5, y - 0.68), xytext=(0.5, y - 0.31),
                             arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.0))
    axes[1].set_xlim(0, 1); axes[1].set_ylim(-0.6, len(stages) - 0.4); axes[1].axis("off")
    set_panel_title(axes[1], "Clean reproduction path")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.suptitle(
        "Timing inclusions and clean all-figure reproduction protocol",
        fontsize=14.0, fontweight="semibold", y=1.01,
    )
    return _save_figure(ctx, fig, stem)


def export_all_tables(ctx: PipelineContext) -> dict[str, Path]:
    """Write stable CSV tables; plots never substitute for numeric evidence."""

    outputs: dict[str, Path] = {}
    for name, table in sorted(ctx.tables.items()):
        path = ctx.writer.table_dir / f"{name}.csv"
        pd.DataFrame(table).to_csv(path, index=False)
        ctx.writer.record_file(path, "table")
        outputs[name] = path
    formula_path = ctx.writer.data_dir / "scientific_convention.json"
    formula_path.write_text(
        json.dumps(
            {
                "formula": STANDARD_FORMULA_TEXT,
                "phasor_convention": "peak complex pressure",
                "force_definition": "F_G = -grad(U_G)",
                "formula_self_test": formula_self_test(),
                "canonical_array": "16x16",
                "canonical_target_m": MAIN_TARGET_M.tolist(),
                "run_mode": ctx.config.mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ctx.writer.record_file(formula_path, "scientific-convention")
    outputs["scientific_convention"] = formula_path
    return outputs


def finalize_manifest(ctx: PipelineContext) -> Path:
    medium = AcousticMedium()
    particle = CompressibleSphere()
    elastic_bead = ElasticSphere()
    coefficients = gorkov_coefficients(ctx.config.frequency_hz, medium, particle)
    return ctx.writer.write_manifest(
        {
            "pipeline_schema_version": 2,
            "evidence_status": (
                "FULL — expanded populations with refined validation numerics"
                if ctx.config.full else
                "PAPER-SCALE SMOKE — accepted manuscript protocol"
            ),
            "pipeline_implementation": PIPELINE_IMPLEMENTATION,
            "run_config": asdict(ctx.config),
            "dependencies": dependency_snapshot(),
            "scientific_convention": STANDARD_FORMULA_TEXT,
            "canonical_target_m": MAIN_TARGET_M.tolist(),
            "branch_source": ctx.artifact.source,
            "branch_joint_2d_stress": ctx.joint_frame.stress,
            "finite_ka_protocol": {
                "model": PartialWaveForceEvaluator.model_name,
                "model_role": "independent finite-ka elastic-bead force reference",
                "model_scope": (
                    "homogeneous isotropic elastic solid sphere with internal "
                    "longitudinal and shear modes in an inviscid host"
                ),
                "medium": asdict(Medium()),
                "sphere": asdict(elastic_bead),
                "root_start_count": 19 if ctx.config.full else 7,
                "partial_wave_numerics": asdict(_partial_wave_numerics(ctx)),
                "paper_scale_smoke_protocol": not ctx.config.full,
            },
            "physical_model_parameters": {
                "medium": asdict(medium),
                "gorkov_optimization_surrogate_particle": asdict(particle),
                "exact_validation_elastic_bead": asdict(elastic_bead),
                "gorkov_coefficients": asdict(coefficients),
                "array": {
                    "side": 16,
                    "elements": 256,
                    "pitch_m": ctx.config.pitch_m,
                    "normals": "+z",
                },
                "fields": {
                    "directivity": "Murata MA40S4S amplitude table",
                    "directivity_evaluation": "global, recomputed at every query point",
                    "rear_policy": "zero/front-only",
                    "optimization_source_scale_pa_m_per_murata_unit": SOURCE_SCALE_PA_M_PER_MURATA_UNIT,
                    "common_on_axis_source_strength_pa_m_peak": REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
                },
            },
        }
    )


ALL_FIGURE_PRODUCERS: tuple[Callable[[PipelineContext], mpl.figure.Figure], ...] = (
    figure_1_integrated_solution_space,
    figure_2_convergence_and_evolution,
    figure_3_decision_geometry,
    figure_4_tri3_multitrap,
    figure_5_matched_holography,
    appendix_a_gorkov_finite_ka,
    appendix_b_morphologies,
    appendix_c_arrays,
    appendix_c_frequencies,
    appendix_d_full_evolution,
    appendix_f_multitrap_ablation,
    appendix_g_timing_reproduction,
)


__all__ = [
    "PipelineContext", "prepare_pipeline", "standard_formula_report",
    "figure_1_integrated_solution_space", "figure_2_convergence_and_evolution",
    "figure_3_decision_geometry", "figure_4_tri3_multitrap",
    "figure_5_matched_holography",
    "appendix_a_gorkov_finite_ka", "appendix_b_morphologies",
    "appendix_c_arrays", "appendix_c_frequencies",
    "appendix_d_full_evolution",
    "appendix_f_multitrap_ablation", "appendix_g_timing_reproduction",
    "export_all_tables", "finalize_manifest", "ALL_FIGURE_PRODUCERS",
]
