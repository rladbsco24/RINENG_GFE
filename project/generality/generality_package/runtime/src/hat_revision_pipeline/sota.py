"""Matched single-vortex comparison utilities for the revision notebook.

This module deliberately separates three things that were mixed in the
recovered pilot scripts:

* prescribed-field synthesis on an explicitly declared vortex ring;
* direct force-equilibrium (FE) optimization with the standard Gor'kov
  potential; and
* independent field/force sampling for the presentation cards.

The alternating-projection implementation is the phase-only GS baseline used
in this revision.  Its control-ring prescription is supplied by the calling
pipeline so that target engineering and solver identity remain separate.

All complex pressures are peak-amplitude phasors.  For a spherical particle in
an inviscid host fluid the optimization surrogate is

    U = K_p |p|^2 - K_g |grad p|^2,

where ``K_g`` is positive for a particle denser than the host.  This is the
standard Gor'kov density-contrast sign; no historical HAT sign reversal is
provided anywhere in this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import minimize

from .gorkov_core import (
    AcousticMedium,
    CompressibleSphere,
    GorkovCoefficients,
    SingleTargetObjective as CorrectedGorkovFEObjective,
    gauge_full,
    gorkov_coefficients,
    reduce_gauge,
)


Array = np.ndarray
TINY = 1.0e-30


# ---------------------------------------------------------------------------
# Standard Gor'kov physics used by FE
# ---------------------------------------------------------------------------


def standard_gorkov_coefficients(
    frequency_hz: float,
    *,
    particle_radius_m: float = 0.65e-3,
    medium_density_kg_m3: float = 1.225,
    medium_sound_speed_m_s: float = 343.0,
    particle_density_kg_m3: float = 100.0,
    particle_sound_speed_m_s: float = 2400.0,
) -> GorkovCoefficients:
    """Compatibility wrapper around the notebook's single physics source.

    The returned object is created by :func:`gorkov_core.gorkov_coefficients`;
    SOTA code does not carry a second implementation of the physical formula.
    """

    return gorkov_coefficients(
        float(frequency_hz),
        AcousticMedium(
            density_kg_m3=float(medium_density_kg_m3),
            sound_speed_m_s=float(medium_sound_speed_m_s),
        ),
        CompressibleSphere(
            density_kg_m3=float(particle_density_kg_m3),
            sound_speed_m_s=float(particle_sound_speed_m_s),
            radius_m=float(particle_radius_m),
        ),
    )


def cartesian_stencil_points(
    target_m: Sequence[float],
    *,
    spacing_m: float = 0.5e-3,
    half_width: int = 2,
) -> tuple[Array, tuple[int, int, int]]:
    """Return a target-centred odd Cartesian stencil and its spatial shape."""

    target = np.asarray(target_m, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_m must be a finite length-three coordinate")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    if int(half_width) < 2:
        raise ValueError("half_width must be at least 2 for repeated derivatives")
    offsets = np.arange(-int(half_width), int(half_width) + 1, dtype=float) * float(
        spacing_m
    )
    points = np.array(
        [target + (dx, dy, dz) for dx in offsets for dy in offsets for dz in offsets],
        dtype=float,
    )
    side = offsets.size
    return points, (side, side, side)


@dataclass(frozen=True)
class FESolveResult:
    phase_rad: Array
    objective: float
    gradient_norm: float
    iterations: int
    evaluations: int
    success: bool
    status: int
    message: str
    history_wall_s: Array
    history_objective: Array
    history_gradient_norm: Array
    # Accepted phase commands, including the initial command and the terminal
    # command.  Figure 3 uses the last accepted step together with the native
    # terminal gradient; it must never substitute an endpoint-to-endpoint
    # C--FE chord for this optimizer-local direction.
    history_phase_rad: Array = field(
        default_factory=lambda: np.empty((0, 0), dtype=float),
        repr=False,
    )
    optimizer: str = "BFGS"
    optimizer_options: dict[str, Any] = field(default_factory=dict)
    gradient_inf_norm: float = float("nan")
    stationary_gtol: bool = False


def _fe_actuator_count(objective: CorrectedGorkovFEObjective) -> int:
    return int(objective.n_transducers)


def solve_corrected_gorkov_fe(
    objective: CorrectedGorkovFEObjective,
    initial_phase_rad: Array,
    *,
    maxiter: int = 1000,
    gtol: float = 1.0e-8,
    optimizer: str | None = None,
    optimizer_options: Mapping[str, Any] | None = None,
) -> FESolveResult:
    """Optimize the unchanged FE objective in unbounded reduced phases.

    Single-target FE/GFE/RH use compact evaluation and unbounded L-BFGS-B
    in every figure and appendix. Incompatible explicit FE solver choices fail before
    optimization. Conventional retains its configured solver.
    """

    from .fe_solver_policy import is_fe_objective, require_fe_solver
    from .compact_single import CompactSingleObjective
    use_fe = is_fe_objective(objective)
    if use_fe:
        optimizer, _ = require_fe_solver(optimizer)
    elif optimizer is None:
        optimizer = "BFGS"
    if use_fe and not isinstance(objective, CompactSingleObjective):
        objective = CompactSingleObjective(objective)
    initial = np.asarray(initial_phase_rad, dtype=float)
    actuator_count = _fe_actuator_count(objective)
    if initial.shape != (actuator_count,):
        raise ValueError(
            f"initial_phase_rad must have shape ({actuator_count},)"
        )
    if int(maxiter) <= 0 or not np.isfinite(gtol) or gtol < 0.0:
        raise ValueError("maxiter must be positive and gtol must be non-negative")
    if optimizer not in ("BFGS", "L-BFGS-B"):
        raise ValueError("optimizer must be BFGS or L-BFGS-B")
    options: dict[str, Any] = {"maxiter": int(maxiter), "gtol": float(gtol), "disp": False}
    if optimizer == "L-BFGS-B":
        options.update(ftol=0.0, maxcor=10, maxls=40, maxfun=1_000_000)
    options.update(dict(optimizer_options or {}))
    options["maxiter"] = int(maxiter)
    reduced0 = reduce_gauge(initial, objective.gauge_index)
    cache: dict[str, Any] = {}

    def fg(x: Array) -> tuple[float, Array]:
        key = np.asarray(x, dtype=float).tobytes()
        if cache.get("key") != key:
            value, gradient = objective.fun_grad(x)
            cache.update(key=key, value=float(value), gradient=np.asarray(gradient, dtype=float))
        return float(cache["value"]), np.asarray(cache["gradient"], dtype=float)

    start = time.perf_counter()
    initial_value, initial_gradient = fg(reduced0)
    history_wall = [0.0]
    history_objective = [float(initial_value)]
    history_gradient = [float(np.linalg.norm(initial_gradient))]
    history_phase = [gauge_full(reduced0, objective.gauge_index)]

    def callback(xk: Array) -> None:
        value, gradient = fg(xk)
        history_wall.append(float(time.perf_counter() - start))
        history_objective.append(float(value))
        history_gradient.append(float(np.linalg.norm(gradient)))
        history_phase.append(gauge_full(np.asarray(xk, dtype=float), objective.gauge_index))

    result = minimize(
        lambda x: fg(x)[0],
        reduced0,
        jac=lambda x: fg(x)[1],
        method=optimizer,
        callback=callback,
        options=options,
    )
    value, gradient = fg(np.asarray(result.x, dtype=float))
    if optimizer == "L-BFGS-B":
        # Verify the raw analytic endpoint gradient, independently of SciPy's
        # success flag or projected-gradient termination bookkeeping.
        value, gradient = objective.fun_grad(np.asarray(result.x, dtype=float))
    elapsed = float(time.perf_counter() - start)
    if not history_wall or elapsed > history_wall[-1] + 1.0e-12:
        history_wall.append(elapsed)
        history_objective.append(float(value))
        history_gradient.append(float(np.linalg.norm(gradient)))
        history_phase.append(gauge_full(np.asarray(result.x, dtype=float), objective.gauge_index))
    full_phase = gauge_full(np.asarray(result.x, dtype=float), objective.gauge_index)
    full_phase = np.angle(np.exp(1j * full_phase))
    return FESolveResult(
        phase_rad=full_phase,
        objective=float(value),
        gradient_norm=float(np.linalg.norm(gradient)),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        history_wall_s=np.asarray(history_wall, dtype=float),
        history_objective=np.asarray(history_objective, dtype=float),
        history_gradient_norm=np.asarray(history_gradient, dtype=float),
        history_phase_rad=np.angle(np.exp(1j * np.asarray(history_phase, dtype=float))),
        optimizer=optimizer,
        optimizer_options=dict(options),
        gradient_inf_norm=float(np.linalg.norm(gradient, ord=np.inf)),
        stationary_gtol=bool(np.linalg.norm(gradient, ord=np.inf) <= float(options["gtol"])),
    )


# ---------------------------------------------------------------------------
# Vortex control-ring prescriptions and phase-only synthesis baselines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VortexControlSpec:
    point_count: int = 8
    radius_wavelengths: float = 1.4
    topological_charge: int = 1

    def __post_init__(self) -> None:
        if int(self.point_count) < 3:
            raise ValueError("point_count must be at least three")
        if not np.isfinite(self.radius_wavelengths) or self.radius_wavelengths <= 0:
            raise ValueError("radius_wavelengths must be finite and positive")
        if int(self.topological_charge) == 0:
            raise ValueError("topological_charge must be non-zero")


def vortex_control_points(
    target_m: Sequence[float],
    wavelength_m: float,
    spec: VortexControlSpec = VortexControlSpec(),
) -> tuple[Array, Array]:
    """Return the horizontal ring coordinates and complex vortex signature."""

    target = np.asarray(target_m, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_m must be a finite length-three coordinate")
    if not np.isfinite(wavelength_m) or float(wavelength_m) <= 0.0:
        raise ValueError("wavelength_m must be finite and positive")
    theta = 2.0 * np.pi * np.arange(spec.point_count, dtype=float) / spec.point_count
    radius = float(spec.radius_wavelengths) * float(wavelength_m)
    points = target[None, :] + np.column_stack(
        (radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta))
    )
    signature = np.exp(1j * int(spec.topological_charge) * theta)
    return points, signature


def phase_project(values: Array) -> Array:
    values = np.asarray(values, dtype=np.complex128)
    return np.exp(1j * np.angle(values + 1.0e-300))


def projective_phase_change(previous: Array, current: Array) -> float:
    previous = np.asarray(previous, dtype=np.complex128)
    current = np.asarray(current, dtype=np.complex128)
    if previous.shape != current.shape:
        raise ValueError("phase states must have matching shapes")
    gauge = np.angle(np.vdot(previous, current))
    delta = np.angle(current * np.exp(-1j * gauge) * np.conj(previous))
    return float(np.max(np.abs(delta)))


def _signature_projection(
    field: Array,
    signature: Array,
    *,
    control_group_size: int | None = None,
) -> Array:
    """Project onto prescribed relative phases with native free phase gauges.

    A Single prescription has one global phase gauge.  A stacked multitrap
    prescription instead has one gauge for each contiguous control-point
    group; no gauge is optimized or shared between rings.
    """

    field = np.asarray(field, dtype=np.complex128)
    signature = np.asarray(signature, dtype=np.complex128)
    if field.shape != signature.shape or field.ndim != 1:
        raise ValueError("field and signature must be matching one-dimensional arrays")
    if signature.size == 0:
        raise ValueError("field and signature must not be empty")
    if control_group_size is None:
        return signature * np.exp(1j * np.angle(field[0] + 1.0e-300))
    if isinstance(control_group_size, (bool, np.bool_)):
        raise ValueError("control_group_size must be a positive integer")
    group_size = int(control_group_size)
    if group_size <= 0 or group_size != control_group_size:
        raise ValueError("control_group_size must be a positive integer")
    if signature.size % group_size != 0:
        raise ValueError(
            "control-point count must be divisible by control_group_size"
        )
    projected = np.empty_like(signature)
    for start in range(0, signature.size, group_size):
        stop = start + group_size
        projected[start:stop] = signature[start:stop] * np.exp(
            1j
            * np.angle(
                field[start] * np.conj(signature[start]) + 1.0e-300
            )
        )
    return projected


@dataclass(frozen=True)
class SynthesisResult:
    method_id: str
    display_name: str
    phase_rad: Array
    iterations: int
    converged: bool | None
    terminal_projective_change_rad: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _validate_transfer_signature(transfer: Array, signature: Array) -> tuple[Array, Array]:
    transfer = np.asarray(transfer, dtype=np.complex128)
    signature = np.asarray(signature, dtype=np.complex128)
    if transfer.ndim != 2 or transfer.shape[0] != signature.size:
        raise ValueError("transfer must have shape (control_points, actuators)")
    if transfer.shape[0] < 3 or transfer.shape[1] < 2:
        raise ValueError("transfer matrix is too small for a vortex benchmark")
    if signature.shape != (transfer.shape[0],):
        raise ValueError("signature length must equal the transfer row count")
    if not np.all(np.isfinite(transfer)) or not np.all(np.isfinite(signature)):
        raise ValueError("transfer and signature must be finite")
    return transfer, signature


def iterative_back_projection(
    transfer: Array,
    signature: Array,
    *,
    maxiter: int = 200,
    tolerance_rad: float = 0.01,
    control_group_size: int | None = None,
) -> SynthesisResult:
    """Iterative back-projection with phase-only emitters."""

    transfer, signature = _validate_transfer_signature(transfer, signature)
    if int(maxiter) <= 0 or not np.isfinite(tolerance_rad) or tolerance_rad <= 0:
        raise ValueError("maxiter and tolerance_rad must be positive")
    desired = signature.copy()
    state = phase_project(transfer.conj().T @ desired)
    terminal_change = math.inf
    converged = False
    iteration = 0
    for iteration in range(1, int(maxiter) + 1):
        field_at_controls = transfer @ state
        desired = _signature_projection(
            field_at_controls,
            signature,
            control_group_size=control_group_size,
        )
        next_state = phase_project(transfer.conj().T @ desired)
        terminal_change = projective_phase_change(state, next_state)
        state = next_state
        if terminal_change < float(tolerance_rad):
            converged = True
            break
    return SynthesisResult(
        method_id="ib_eight_point",
        display_name="Eight-point iterative back-projection",
        phase_rad=np.angle(state),
        iterations=int(iteration),
        converged=bool(converged),
        terminal_projective_change_rad=float(terminal_change),
        metadata=(
            {}
            if control_group_size is None
            else {"control_group_size": int(control_group_size)}
        ),
    )


def phase_only_signature_projection_audit(
    transfer: Array,
    signature: Array,
    *,
    iterations: int = 100,
    control_group_size: int | None = None,
) -> SynthesisResult:
    """Phase-only GS recurrence for a supplied complex field prescription."""

    transfer, signature = _validate_transfer_signature(transfer, signature)
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    denominator = np.sum(np.abs(transfer) ** 2, axis=1)
    back_projection = transfer.conj().T / np.maximum(denominator[None, :], TINY)
    recurrence = transfer @ back_projection
    desired = signature.copy()
    reconstructed = np.zeros_like(desired)
    for _ in range(int(iterations)):
        reconstructed = recurrence @ desired
        desired = _signature_projection(
            reconstructed,
            signature,
            control_group_size=control_group_size,
        )
    weighted_desired = desired / np.maximum(np.abs(reconstructed), 1.0e-300)
    native_command = back_projection @ weighted_desired
    clipped_command = np.minimum(np.abs(native_command), 1.0) * np.exp(
        1j * np.angle(native_command)
    )
    phase_only_command = phase_project(clipped_command)
    return SynthesisResult(
        method_id="gs_phase_only",
        display_name="GS",
        phase_rad=np.angle(phase_only_command),
        iterations=int(iterations),
        converged=None,
        terminal_projective_change_rad=None,
        metadata={
            "native_clipped_source_power": float(np.sum(np.abs(clipped_command) ** 2)),
            "phase_only": True,
            **(
                {}
                if control_group_size is None
                else {"control_group_size": int(control_group_size)}
            ),
        },
    )


def vortex_ring_metrics(transfer: Array, signature: Array, phase_rad: Array) -> dict[str, float]:
    """Native prescribed-field metrics on the matched control ring."""

    transfer, signature = _validate_transfer_signature(transfer, signature)
    phase = np.asarray(phase_rad, dtype=float)
    if phase.shape != (transfer.shape[1],):
        raise ValueError("phase length must equal the actuator count")
    field_at_controls = transfer @ np.exp(1j * phase)
    field_norm = float(np.linalg.norm(field_at_controls))
    signature_norm = float(np.linalg.norm(signature))
    overlap = float(
        np.clip(
            abs(np.vdot(signature, field_at_controls))
            / max(field_norm * signature_norm, TINY),
            0.0,
            1.0,
        )
    )
    normalized_error = float(math.sqrt(max(0.0, 2.0 - 2.0 * overlap)))
    amplitudes = np.abs(field_at_controls)
    amplitude_cv = float(np.std(amplitudes) / max(float(np.mean(amplitudes)), TINY))
    return {
        "signature_overlap": overlap,
        "normalized_complex_error": normalized_error,
        "ring_amplitude_cv": amplitude_cv,
    }


# ---------------------------------------------------------------------------
# Complete, explicit timing protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingHooks:
    """Backend hooks used to make timing inclusion explicit.

    NumPy callers can use the defaults.  GPU callers should supply a transfer
    function, a one-time compilation function, and a device synchronize
    function.  Synchronization is called before and after the solve interval.
    """

    transfer_to_compute: Callable[[Any], Any] | None = None
    compile: Callable[[Any], None] | None = None
    synchronize: Callable[[], None] | None = None
    backend: str = "numpy"

    def without_compilation(self) -> "TimingHooks":
        return replace(self, compile=None)


@dataclass(frozen=True)
class TimingBreakdown:
    initialization_s: float
    transfer_s: float
    compilation_s: float
    pre_solve_synchronization_s: float
    solve_s: float
    post_solve_synchronization_s: float
    total_s: float
    backend: str
    transfer_included: bool
    compilation_included: bool
    synchronization_included: bool

    @property
    def accounted_s(self) -> float:
        return float(
            self.initialization_s
            + self.transfer_s
            + self.compilation_s
            + self.pre_solve_synchronization_s
            + self.solve_s
            + self.post_solve_synchronization_s
        )


@dataclass(frozen=True)
class TimedExecution:
    result: Any
    timing: TimingBreakdown


def _elapsed_call(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function()
    return result, float(time.perf_counter() - started)


def execute_with_complete_timing(
    initializer: Callable[[], Any],
    solver: Callable[[Any], Any],
    *,
    hooks: TimingHooks = TimingHooks(),
) -> TimedExecution:
    """Time initialization, transfer, compilation, solve, and synchronization.

    Exact-force validation and plotting are intentionally outside this interval:
    the interval ends when the terminal phase command is available.
    """

    total_started = time.perf_counter()
    payload, initialization_s = _elapsed_call(initializer)
    if hooks.transfer_to_compute is None:
        transferred = payload
        transfer_s = 0.0
    else:
        transferred, transfer_s = _elapsed_call(lambda: hooks.transfer_to_compute(payload))
    if hooks.compile is None:
        compilation_s = 0.0
    else:
        _, compilation_s = _elapsed_call(lambda: hooks.compile(transferred))
    if hooks.synchronize is None:
        pre_sync_s = 0.0
    else:
        _, pre_sync_s = _elapsed_call(hooks.synchronize)
    result, solve_s = _elapsed_call(lambda: solver(transferred))
    if hooks.synchronize is None:
        post_sync_s = 0.0
    else:
        _, post_sync_s = _elapsed_call(hooks.synchronize)
    total_s = float(time.perf_counter() - total_started)
    timing = TimingBreakdown(
        initialization_s=initialization_s,
        transfer_s=transfer_s,
        compilation_s=compilation_s,
        pre_solve_synchronization_s=pre_sync_s,
        solve_s=solve_s,
        post_solve_synchronization_s=post_sync_s,
        total_s=total_s,
        backend=str(hooks.backend),
        transfer_included=hooks.transfer_to_compute is not None,
        compilation_included=hooks.compile is not None,
        synchronization_included=hooks.synchronize is not None,
    )
    return TimedExecution(result=result, timing=timing)


@dataclass(frozen=True)
class BenchmarkRun:
    method_id: str
    display_name: str
    initialization_kind: str
    target_m: Array
    phase_rad: Array
    timing: TimingBreakdown
    solver_result: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


def benchmark_run_row(run: BenchmarkRun) -> dict[str, Any]:
    """Flatten one run into the stable timing-table schema used by the notebook."""

    timing = run.timing
    solver = run.solver_result
    row: dict[str, Any] = {
        "method_id": run.method_id,
        "method": run.display_name,
        "initialization": run.initialization_kind,
        "target_x_m": float(run.target_m[0]),
        "target_y_m": float(run.target_m[1]),
        "target_z_m": float(run.target_m[2]),
        "backend": timing.backend,
        "initialization_s": timing.initialization_s,
        "transfer_s": timing.transfer_s,
        "compilation_s": timing.compilation_s,
        "pre_solve_synchronization_s": timing.pre_solve_synchronization_s,
        "solve_s": timing.solve_s,
        "post_solve_synchronization_s": timing.post_solve_synchronization_s,
        "end_to_end_s": timing.total_s,
        "transfer_included": timing.transfer_included,
        "compilation_included": timing.compilation_included,
        "synchronization_included": timing.synchronization_included,
        "iterations": int(getattr(solver, "iterations", 0)),
        "evaluations": int(getattr(solver, "evaluations", 0)),
    }
    if isinstance(solver, FESolveResult):
        row.update(
            {
                "algorithm_converged": solver.success,
                "terminal_objective": solver.objective,
                "terminal_phase_gradient_norm": solver.gradient_norm,
            }
        )
    elif isinstance(solver, SynthesisResult):
        row.update(
            {
                "algorithm_converged": solver.converged,
                "terminal_projective_change_rad": solver.terminal_projective_change_rad,
            }
        )
    for key, value in run.metadata.items():
        if np.isscalar(value) or value is None:
            row[key] = value
    return row


def benchmark_synthesis_method(
    method: Callable[[Array, Array], SynthesisResult],
    transfer: Array,
    signature: Array,
    *,
    hooks: TimingHooks = TimingHooks(),
    method_kwargs: Mapping[str, Any] | None = None,
) -> BenchmarkRun:
    """Run one prescribed-field method under the complete timing protocol."""

    transfer, signature = _validate_transfer_signature(transfer, signature)
    kwargs = dict(method_kwargs or {})

    def initializer() -> tuple[Array, Array]:
        return transfer.copy(), signature.copy()

    timed = execute_with_complete_timing(
        initializer,
        lambda payload: method(payload[0], payload[1], **kwargs),
        hooks=hooks,
    )
    result: SynthesisResult = timed.result
    metrics = vortex_ring_metrics(transfer, signature, result.phase_rad)
    return BenchmarkRun(
        method_id=result.method_id,
        display_name=result.display_name,
        initialization_kind="deterministic_signature",
        target_m=np.full(3, np.nan),
        phase_rad=np.asarray(result.phase_rad, dtype=float),
        timing=timed.timing,
        solver_result=result,
        metadata={
            **dict(result.metadata),
            **metrics,
            "transfer_construction_included": False,
        },
    )


def benchmark_synthesis_method_factory(
    method: Callable[[Array, Array], SynthesisResult],
    transfer_and_signature_factory: Callable[[], tuple[Array, Array]],
    *,
    target_m: Sequence[float] | None = None,
    hooks: TimingHooks = TimingHooks(),
    method_kwargs: Mapping[str, Any] | None = None,
) -> BenchmarkRun:
    """Benchmark synthesis while including transfer construction in setup.

    This is the preferred manuscript path.  The factory should build the
    control-ring transfer with the common global forward model; that work is
    then included in ``initialization_s``.  The prebuilt-matrix variant above is
    retained for controlled kernel-only timing and labels that fact explicitly.
    """

    kwargs = dict(method_kwargs or {})

    def initializer() -> tuple[Array, Array]:
        transfer, signature = transfer_and_signature_factory()
        return _validate_transfer_signature(transfer, signature)

    timed = execute_with_complete_timing(
        initializer,
        lambda payload: method(payload[0], payload[1], **kwargs),
        hooks=hooks,
    )
    result: SynthesisResult = timed.result
    # Rebuild once outside the timing interval only for common validation.  The
    # phase solve timing above already contains its own constructed transfer.
    transfer, signature = _validate_transfer_signature(*transfer_and_signature_factory())
    metrics = vortex_ring_metrics(transfer, signature, result.phase_rad)
    target = (
        np.full(3, np.nan)
        if target_m is None
        else np.asarray(target_m, dtype=float)
    )
    if target.shape != (3,):
        raise ValueError("target_m must be length three")
    return BenchmarkRun(
        method_id=result.method_id,
        display_name=result.display_name,
        initialization_kind="deterministic_signature",
        target_m=target,
        phase_rad=np.asarray(result.phase_rad, dtype=float),
        timing=timed.timing,
        solver_result=result,
        metadata={
            **dict(result.metadata),
            **metrics,
            "transfer_construction_included": True,
        },
    )


FESolver = Callable[..., FESolveResult]


def benchmark_fe_cold_start(
    objective: CorrectedGorkovFEObjective,
    target_m: Sequence[float],
    *,
    seed: int,
    maxiter: int = 1000,
    gtol: float = 1.0e-8,
    hooks: TimingHooks = TimingHooks(),
    solve_fn: FESolver = solve_corrected_gorkov_fe,
) -> BenchmarkRun:
    """Benchmark a deterministic random cold start of corrected Gor'kov FE."""

    target = np.asarray(target_m, dtype=float)
    if target.shape != (3,):
        raise ValueError("target_m must be length three")

    def initializer() -> Array:
        rng = np.random.default_rng(int(seed))
        return rng.uniform(-np.pi, np.pi, size=_fe_actuator_count(objective))

    timed = execute_with_complete_timing(
        initializer,
        lambda initial: solve_fn(objective, initial, maxiter=maxiter, gtol=gtol),
        hooks=hooks,
    )
    result: FESolveResult = timed.result
    return BenchmarkRun(
        method_id="corrected_gorkov_fe_cold",
        display_name="Force-equilibrium optimization (cold)",
        initialization_kind="cold_random",
        target_m=target,
        phase_rad=np.asarray(result.phase_rad, dtype=float),
        timing=timed.timing,
        solver_result=result,
        metadata={
            "seed": int(seed),
            "standard_gorkov_sign": True,
            "objective_construction_included": False,
        },
    )


def benchmark_fe_cold_start_factory(
    objective_factory: Callable[[Array], CorrectedGorkovFEObjective],
    target_m: Sequence[float],
    *,
    seed: int,
    maxiter: int = 1000,
    gtol: float = 1.0e-8,
    hooks: TimingHooks = TimingHooks(),
    solve_fn: FESolver = solve_corrected_gorkov_fe,
) -> BenchmarkRun:
    """Cold FE benchmark including objective/transfer construction in setup."""

    target = np.asarray(target_m, dtype=float)
    if target.shape != (3,):
        raise ValueError("target_m must be length three")

    def initializer() -> tuple[CorrectedGorkovFEObjective, Array]:
        objective = objective_factory(target.copy())
        rng = np.random.default_rng(int(seed))
        phase = rng.uniform(-np.pi, np.pi, size=_fe_actuator_count(objective))
        return objective, phase

    timed = execute_with_complete_timing(
        initializer,
        lambda payload: solve_fn(
            payload[0], payload[1], maxiter=maxiter, gtol=gtol
        ),
        hooks=hooks,
    )
    result: FESolveResult = timed.result
    return BenchmarkRun(
        method_id="corrected_gorkov_fe_cold",
        display_name="Force-equilibrium optimization (cold)",
        initialization_kind="cold_random",
        target_m=target,
        phase_rad=np.asarray(result.phase_rad, dtype=float),
        timing=timed.timing,
        solver_result=result,
        metadata={
            "seed": int(seed),
            "standard_gorkov_sign": True,
            "objective_construction_included": True,
        },
    )


def benchmark_ib_initialized_fe_factory(
    objective_factory: Callable[[Array], CorrectedGorkovFEObjective],
    transfer_and_signature_factory: Callable[[], tuple[Array, Array]],
    target_m: Sequence[float],
    *,
    ib_maxiter: int = 200,
    ib_tolerance_rad: float = 0.01,
    fe_maxiter: int = 1000,
    gtol: float = 1.0e-8,
    hooks: TimingHooks = TimingHooks(),
    solve_fn: FESolver = solve_corrected_gorkov_fe,
) -> BenchmarkRun:
    """Time an end-to-end IB command followed by FE refinement.

    The objective, control-ring transfer, IB solve, and FE solve are all inside
    the complete timing interval.  This makes the hybrid a deployable method,
    not a warm-start result that receives a free precomputed initialization.
    """

    target = np.asarray(target_m, dtype=float)
    if target.shape != (3,):
        raise ValueError("target_m must be length three")

    def initializer() -> tuple[CorrectedGorkovFEObjective, Array, Array]:
        objective = objective_factory(target.copy())
        transfer, signature = _validate_transfer_signature(
            *transfer_and_signature_factory()
        )
        if transfer.shape[1] != _fe_actuator_count(objective):
            raise ValueError("IB transfer and FE objective must use the same actuators")
        return objective, transfer, signature

    def solve(payload: tuple[CorrectedGorkovFEObjective, Array, Array]) -> tuple[SynthesisResult, FESolveResult]:
        objective, transfer, signature = payload
        ib_result = iterative_back_projection(
            transfer,
            signature,
            maxiter=int(ib_maxiter),
            tolerance_rad=float(ib_tolerance_rad),
        )
        fe_result = solve_fn(
            objective,
            np.asarray(ib_result.phase_rad, dtype=float),
            maxiter=int(fe_maxiter),
            gtol=float(gtol),
        )
        return ib_result, fe_result

    timed = execute_with_complete_timing(initializer, solve, hooks=hooks)
    ib_result, fe_result = timed.result
    transfer, signature = _validate_transfer_signature(
        *transfer_and_signature_factory()
    )
    before = vortex_ring_metrics(transfer, signature, ib_result.phase_rad)
    after = vortex_ring_metrics(transfer, signature, fe_result.phase_rad)
    return BenchmarkRun(
        method_id="ib_initialized_corrected_gorkov_fe",
        display_name="IB-initialized force-equilibrium refinement",
        initialization_kind="IB_vortex_signature",
        target_m=target.copy(),
        phase_rad=np.asarray(fe_result.phase_rad, dtype=float),
        timing=timed.timing,
        solver_result=fe_result,
        metadata={
            "standard_gorkov_sign": True,
            "objective_construction_included": True,
            "ib_transfer_construction_included": True,
            "ib_solve_included": True,
            "ib_iterations": int(ib_result.iterations),
            "ib_converged": bool(ib_result.converged),
            "ib_initial_ring_normalized_complex_error": before["normalized_complex_error"],
            "ring_normalized_complex_error": after["normalized_complex_error"],
        },
    )


def benchmark_fe_adjacent_target_warm_path(
    objective_factory: Callable[[Array], CorrectedGorkovFEObjective],
    targets_m: Sequence[Sequence[float]],
    *,
    first_seed: int,
    maxiter: int = 1000,
    gtol: float = 1.0e-8,
    hooks: TimingHooks = TimingHooks(),
    solve_fn: FESolver = solve_corrected_gorkov_fe,
    reuse_compilation_after_first: bool = True,
) -> list[BenchmarkRun]:
    """Cold-start the first target, then warm-start every adjacent target.

    The returned rows explicitly distinguish the first cold solve and all
    adjacent-target warm solves.  If compilation is reused, later rows record
    zero compilation time and ``compilation_reused=True``.
    """

    targets = [np.asarray(item, dtype=float) for item in targets_m]
    if not targets or any(item.shape != (3,) for item in targets):
        raise ValueError("targets_m must contain one or more length-three coordinates")
    runs: list[BenchmarkRun] = []
    previous_phase: Array | None = None
    expected_actuators: int | None = None
    rng = np.random.default_rng(int(first_seed))

    for index, target in enumerate(targets):
        if previous_phase is None:
            initialization_kind = "cold_random"
        else:
            initialization_kind = "adjacent_target_warm"
        active_hooks = (
            hooks.without_compilation()
            if index > 0 and reuse_compilation_after_first
            else hooks
        )

        def initializer(
            current_target: Array = target.copy(),
            previous: Array | None = None if previous_phase is None else previous_phase.copy(),
        ) -> tuple[CorrectedGorkovFEObjective, Array]:
            nonlocal expected_actuators
            objective = objective_factory(current_target)
            actuator_count = _fe_actuator_count(objective)
            if expected_actuators is None:
                expected_actuators = actuator_count
            elif actuator_count != expected_actuators:
                raise ValueError("all objectives on a warm path must use the same actuators")
            initial = (
                rng.uniform(-np.pi, np.pi, size=actuator_count)
                if previous is None
                else previous.copy()
            )
            return objective, initial

        timed = execute_with_complete_timing(
            initializer,
            lambda payload: solve_fn(
                payload[0], payload[1], maxiter=maxiter, gtol=gtol
            ),
            hooks=active_hooks,
        )
        result: FESolveResult = timed.result
        previous_phase = np.asarray(result.phase_rad, dtype=float).copy()
        runs.append(
            BenchmarkRun(
                method_id=(
                    "corrected_gorkov_fe_cold"
                    if index == 0
                    else "corrected_gorkov_fe_adjacent_warm"
                ),
                display_name=(
                    "Force-equilibrium optimization (cold)"
                    if index == 0
                    else "Force-equilibrium optimization (adjacent warm start)"
                ),
                initialization_kind=initialization_kind,
                target_m=target.copy(),
                phase_rad=previous_phase.copy(),
                timing=timed.timing,
                solver_result=result,
                metadata={
                    "path_index": int(index),
                    "first_seed": int(first_seed),
                    "standard_gorkov_sign": True,
                    "compilation_reused": bool(index > 0 and reuse_compilation_after_first),
                    "objective_construction_included": True,
                },
            )
        )
    return runs


def benchmark_fe_cold_and_warm(
    objective_factory: Callable[[Array], CorrectedGorkovFEObjective],
    targets_m: Sequence[Sequence[float]],
    *,
    seeds: Sequence[int],
    maxiter: int = 1000,
    gtol: float = 1.0e-8,
    hooks: TimingHooks = TimingHooks(),
    solve_fn: FESolver = solve_corrected_gorkov_fe,
) -> dict[str, list[BenchmarkRun]]:
    """Produce paired cold-start rows and one adjacent-target warm path."""

    targets = [np.asarray(item, dtype=float) for item in targets_m]
    if len(seeds) != len(targets):
        raise ValueError("one deterministic cold-start seed is required per target")
    cold = [
        benchmark_fe_cold_start_factory(
            objective_factory,
            target,
            seed=int(seed),
            maxiter=maxiter,
            gtol=gtol,
            hooks=hooks,
            solve_fn=solve_fn,
        )
        for target, seed in zip(targets, seeds)
    ]
    warm = benchmark_fe_adjacent_target_warm_path(
        objective_factory,
        targets,
        first_seed=int(seeds[0]),
        maxiter=maxiter,
        gtol=gtol,
        hooks=hooks,
        solve_fn=solve_fn,
    )
    return {"cold": cold, "adjacent_warm": warm}


# ---------------------------------------------------------------------------
# Data generation for matched pressure/force field cards
# ---------------------------------------------------------------------------


class PressureEvaluator(Protocol):
    def __call__(self, points_m: Array, phase_rad: Array) -> Array: ...


class ForceEvaluator(Protocol):
    def __call__(self, points_m: Array, phase_rad: Array) -> Array: ...


@dataclass(frozen=True)
class FieldPlaneData:
    name: str
    u_m: Array
    v_m: Array
    points_m: Array
    pressure_abs: Array
    pressure_normalized: Array
    force_in_plane_N: Array | None
    force_normalized: Array | None
    target_uv_m: Array
    equilibrium_uv_m: Array


@dataclass(frozen=True)
class MethodFieldCard:
    method_id: str
    horizontal: FieldPlaneData
    vertical: FieldPlaneData


@dataclass(frozen=True)
class MatchedFieldCards:
    cards: Mapping[str, MethodFieldCard]
    shared_pressure_scale: float
    shared_force_scale_N: float | None


def _plane_grid(
    target_m: Array,
    *,
    plane: str,
    u_half_width_m: float,
    v_half_width_m: float,
    samples: int,
) -> tuple[Array, Array, Array]:
    if int(samples) < 3:
        raise ValueError("samples must be at least three")
    u = np.linspace(-float(u_half_width_m), float(u_half_width_m), int(samples))
    v = np.linspace(-float(v_half_width_m), float(v_half_width_m), int(samples))
    uu, vv = np.meshgrid(u, v, indexing="xy")
    if plane == "horizontal":
        points = np.stack(
            (target_m[0] + uu, target_m[1] + vv, np.full_like(uu, target_m[2])),
            axis=-1,
        )
    elif plane == "vertical":
        points = np.stack(
            (target_m[0] + uu, np.full_like(uu, target_m[1]), target_m[2] + vv),
            axis=-1,
        )
    else:
        raise ValueError("plane must be 'horizontal' or 'vertical'")
    return u, v, points


def _equilibrium_uv(target: Array, equilibrium: Array, plane: str) -> Array:
    delta = equilibrium - target
    if plane == "horizontal":
        return np.asarray([delta[0], delta[1]], dtype=float)
    return np.asarray([delta[0], delta[2]], dtype=float)


def generate_matched_field_cards(
    phases_by_method: Mapping[str, Array],
    pressure_evaluator: PressureEvaluator,
    *,
    target_m: Sequence[float],
    equilibrium_by_method_m: Mapping[str, Sequence[float]],
    force_evaluator: ForceEvaluator | None = None,
    horizontal_half_width_m: float = 8.0e-3,
    vertical_x_half_width_m: float = 8.0e-3,
    vertical_z_half_width_m: float = 8.0e-3,
    samples: int = 101,
) -> MatchedFieldCards:
    """Sample matched horizontal/vertical pressure and exact-force cards.

    A single pressure scale and, when supplied, a single force scale are used
    across every method and both planes.  The returned arrays keep raw SI force
    values as well as shared-normalized vectors for plotting.
    """

    if not phases_by_method:
        raise ValueError("phases_by_method cannot be empty")
    if set(phases_by_method) != set(equilibrium_by_method_m):
        raise ValueError("each method needs one independently evaluated equilibrium")
    target = np.asarray(target_m, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_m must be a finite length-three coordinate")
    for value in (horizontal_half_width_m, vertical_x_half_width_m, vertical_z_half_width_m):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("all field-card half widths must be positive")

    grids = {
        "horizontal": _plane_grid(
            target,
            plane="horizontal",
            u_half_width_m=horizontal_half_width_m,
            v_half_width_m=horizontal_half_width_m,
            samples=samples,
        ),
        "vertical": _plane_grid(
            target,
            plane="vertical",
            u_half_width_m=vertical_x_half_width_m,
            v_half_width_m=vertical_z_half_width_m,
            samples=samples,
        ),
    }
    raw: dict[str, dict[str, dict[str, Any]]] = {}
    pressure_scales: list[float] = []
    force_scales: list[float] = []
    for method_id, phase in phases_by_method.items():
        phase_array = np.asarray(phase, dtype=float)
        equilibrium = np.asarray(equilibrium_by_method_m[method_id], dtype=float)
        if equilibrium.shape != (3,) or not np.all(np.isfinite(equilibrium)):
            raise ValueError(f"invalid equilibrium for {method_id!r}")
        raw[method_id] = {}
        for plane_name, (u, v, points) in grids.items():
            flat_points = points.reshape(-1, 3)
            pressure = np.asarray(
                pressure_evaluator(flat_points, phase_array), dtype=np.complex128
            ).reshape(points.shape[:2])
            if not np.all(np.isfinite(pressure)):
                raise ValueError(f"non-finite pressure returned for {method_id!r}")
            pressure_abs = np.abs(pressure)
            pressure_scales.append(float(np.max(pressure_abs)))
            force_in_plane = None
            if force_evaluator is not None:
                force = np.asarray(force_evaluator(flat_points, phase_array), dtype=float)
                if force.shape != (flat_points.shape[0], 3) or not np.all(np.isfinite(force)):
                    raise ValueError("force_evaluator must return a finite (points, 3) array")
                force = force.reshape(points.shape[:2] + (3,))
                if plane_name == "horizontal":
                    force_in_plane = force[..., (0, 1)]
                else:
                    force_in_plane = force[..., (0, 2)]
                force_scales.append(float(np.max(np.linalg.norm(force_in_plane, axis=-1))))
            raw[method_id][plane_name] = {
                "u": u,
                "v": v,
                "points": points,
                "pressure_abs": pressure_abs,
                "force_in_plane": force_in_plane,
                "equilibrium_uv": _equilibrium_uv(target, equilibrium, plane_name),
            }

    pressure_scale = max(max(pressure_scales), TINY)
    force_scale = max(max(force_scales), TINY) if force_scales else None
    cards: dict[str, MethodFieldCard] = {}
    for method_id, method_raw in raw.items():
        planes: dict[str, FieldPlaneData] = {}
        for plane_name, values in method_raw.items():
            force_raw = values["force_in_plane"]
            force_normalized = None if force_raw is None else force_raw / force_scale
            planes[plane_name] = FieldPlaneData(
                name=plane_name,
                u_m=np.asarray(values["u"], dtype=float),
                v_m=np.asarray(values["v"], dtype=float),
                points_m=np.asarray(values["points"], dtype=float),
                pressure_abs=np.asarray(values["pressure_abs"], dtype=float),
                pressure_normalized=np.asarray(values["pressure_abs"], dtype=float)
                / pressure_scale,
                force_in_plane_N=None
                if force_raw is None
                else np.asarray(force_raw, dtype=float),
                force_normalized=None
                if force_normalized is None
                else np.asarray(force_normalized, dtype=float),
                target_uv_m=np.zeros(2, dtype=float),
                equilibrium_uv_m=np.asarray(values["equilibrium_uv"], dtype=float),
            )
        cards[method_id] = MethodFieldCard(
            method_id=method_id,
            horizontal=planes["horizontal"],
            vertical=planes["vertical"],
        )
    return MatchedFieldCards(
        cards=cards,
        shared_pressure_scale=float(pressure_scale),
        shared_force_scale_N=None if force_scale is None else float(force_scale),
    )


def matched_field_cards_npz_payload(cards: MatchedFieldCards) -> dict[str, Array]:
    """Flatten matched field-card data into a deterministic ``np.savez`` payload."""

    payload: dict[str, Array] = {
        "shared_pressure_scale": np.asarray(cards.shared_pressure_scale),
        "shared_force_scale_N": np.asarray(
            np.nan if cards.shared_force_scale_N is None else cards.shared_force_scale_N
        ),
    }
    for method_id in sorted(cards.cards):
        card = cards.cards[method_id]
        for plane_name, plane in (("horizontal", card.horizontal), ("vertical", card.vertical)):
            prefix = f"{method_id}__{plane_name}__"
            payload[prefix + "u_m"] = plane.u_m
            payload[prefix + "v_m"] = plane.v_m
            payload[prefix + "points_m"] = plane.points_m
            payload[prefix + "pressure_abs"] = plane.pressure_abs
            payload[prefix + "pressure_normalized"] = plane.pressure_normalized
            payload[prefix + "target_uv_m"] = plane.target_uv_m
            payload[prefix + "equilibrium_uv_m"] = plane.equilibrium_uv_m
            if plane.force_in_plane_N is not None:
                payload[prefix + "force_in_plane_N"] = plane.force_in_plane_N
                payload[prefix + "force_normalized"] = plane.force_normalized
    return payload


__all__ = [
    "BenchmarkRun",
    "CorrectedGorkovFEObjective",
    "FESolveResult",
    "FieldPlaneData",
    "GorkovCoefficients",
    "MatchedFieldCards",
    "MethodFieldCard",
    "SynthesisResult",
    "TimedExecution",
    "TimingBreakdown",
    "TimingHooks",
    "VortexControlSpec",
    "benchmark_fe_adjacent_target_warm_path",
    "benchmark_fe_cold_and_warm",
    "benchmark_fe_cold_start",
    "benchmark_fe_cold_start_factory",
    "benchmark_ib_initialized_fe_factory",
    "benchmark_run_row",
    "benchmark_synthesis_method",
    "benchmark_synthesis_method_factory",
    "cartesian_stencil_points",
    "execute_with_complete_timing",
    "generate_matched_field_cards",
    "iterative_back_projection",
    "matched_field_cards_npz_payload",
    "phase_only_signature_projection_audit",
    "projective_phase_change",
    "solve_corrected_gorkov_fe",
    "standard_gorkov_coefficients",
    "vortex_control_points",
    "vortex_ring_metrics",
]
