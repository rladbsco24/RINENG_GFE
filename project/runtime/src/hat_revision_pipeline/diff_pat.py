"""Phase-only Diff-PAT command generation for the fixed manuscript tasks.

``AD`` in the revision means *Acoustic hologram optimisation using automatic
differentiation*.  This module implements that method directly: the actuator
amplitudes remain unity, the optimized variables are actuator phases, and the
loss contains only target-plane pressure amplitudes.  It does not use a
Gor'kov, force, equilibrium, or target-phase term.

The two manuscript task shapes are deliberately fixed at eight control points
for Single and twenty-four control points for Triple (three contiguous blocks
of eight).  The caller supplies the common pressure transfer matrix and the
control signature used by the other holography methods.  Only the magnitude of
that signature enters the Diff-PAT loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .sota import SynthesisResult


Array = np.ndarray
DiffPATTask = Literal["Single", "Triple"]


# The transfer matrices in this project are complex128.  Keeping JAX in x64
# avoids silently changing the forward model used by the other methods.
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class DiffPATConfig:
    """Frozen optimization contract used for both manuscript tasks."""

    steps: int = 150
    learning_rate: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    seed_offset: int = 181

    def __post_init__(self) -> None:
        if int(self.steps) != 150:
            raise ValueError("the manuscript Diff-PAT contract uses exactly 150 steps")
        if not np.isclose(float(self.learning_rate), 0.1, rtol=0.0, atol=0.0):
            raise ValueError("the manuscript Diff-PAT learning rate is 0.1")
        if not np.isclose(float(self.beta1), 0.9, rtol=0.0, atol=0.0):
            raise ValueError("the manuscript Diff-PAT beta1 is 0.9")
        if not np.isclose(float(self.beta2), 0.999, rtol=0.0, atol=0.0):
            raise ValueError("the manuscript Diff-PAT beta2 is 0.999")
        if not np.isclose(float(self.epsilon), 1.0e-8, rtol=0.0, atol=0.0):
            raise ValueError("the manuscript Diff-PAT epsilon is 1e-8")
        if int(self.seed_offset) != 181:
            raise ValueError("the manuscript Diff-PAT seed offset is +181")


DIFF_PAT_CONFIG = DiffPATConfig()
DIFF_PAT_CONTROL_POINTS: dict[DiffPATTask, int] = {
    "Single": 8,
    "Triple": 24,
}


def _canonical_task(task: str) -> DiffPATTask:
    key = str(task).strip().lower()
    if key == "single":
        return "Single"
    if key == "triple":
        return "Triple"
    raise ValueError("task must be 'Single' or 'Triple'")


def _validated_inputs(
    transfer: Array,
    signature: Array,
    *,
    task: DiffPATTask,
) -> tuple[Array, Array]:
    matrix = np.asarray(transfer, dtype=np.complex128)
    target = np.abs(np.asarray(signature, dtype=np.complex128))
    expected_points = DIFF_PAT_CONTROL_POINTS[task]
    if matrix.ndim != 2:
        raise ValueError("transfer must be a two-dimensional complex matrix")
    if matrix.shape[0] != expected_points:
        raise ValueError(
            f"{task} Diff-PAT requires exactly {expected_points} control points; "
            f"received {matrix.shape[0]}"
        )
    if matrix.shape[1] < 2:
        raise ValueError("Diff-PAT requires at least two phase-only actuators")
    if target.shape != (expected_points,):
        raise ValueError(
            f"signature must have shape ({expected_points},) for the {task} task"
        )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("transfer and signature must contain only finite values")
    if np.any(target < 0.0) or not np.any(target > 0.0):
        raise ValueError("the target amplitude must contain a positive value")
    row_capacity = np.sum(np.abs(matrix), axis=1)
    if np.any(row_capacity <= 0.0) or not np.all(np.isfinite(row_capacity)):
        raise ValueError("every control point must couple to at least one actuator")
    return matrix, target


def _optimizer_kernel(
    transfer: jax.Array,
    target_amplitude: jax.Array,
    initial_phase: jax.Array,
    row_capacity: jax.Array,
    config: DiffPATConfig,
) -> tuple[jax.Array, jax.Array]:
    """Compile and execute the fixed Adam loop.

    Dividing each pressure amplitude by its phase-aligned row capacity makes
    the supplied target magnitude dimensionless without changing the pressure
    forward model.  No target phase is evaluated.
    """

    target_scale = jnp.sqrt(jnp.mean(jnp.square(target_amplitude)))
    normalized_target = target_amplitude / jnp.maximum(
        target_scale, jnp.asarray(config.epsilon, dtype=jnp.float64)
    )

    def loss(phase: jax.Array) -> jax.Array:
        command = jnp.exp(1j * phase)
        normalized_amplitude = jnp.abs(transfer @ command) / row_capacity
        return jnp.mean(jnp.square(normalized_amplitude - normalized_target))

    value_and_grad = jax.value_and_grad(loss)

    def adam_step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        step_index: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        phase, first_moment, second_moment = carry
        _, gradient = value_and_grad(phase)
        first_moment = (
            config.beta1 * first_moment + (1.0 - config.beta1) * gradient
        )
        second_moment = (
            config.beta2 * second_moment
            + (1.0 - config.beta2) * jnp.square(gradient)
        )
        step_number = step_index.astype(jnp.float64) + 1.0
        corrected_first = first_moment / (1.0 - config.beta1**step_number)
        corrected_second = second_moment / (1.0 - config.beta2**step_number)
        phase = phase - config.learning_rate * corrected_first / (
            jnp.sqrt(corrected_second) + config.epsilon
        )
        phase = jnp.angle(jnp.exp(1j * phase))
        return (phase, first_moment, second_moment), loss(phase)

    initial_loss = loss(initial_phase)
    initial_state = (
        initial_phase,
        jnp.zeros_like(initial_phase),
        jnp.zeros_like(initial_phase),
    )
    (terminal_phase, _, _), post_step_losses = jax.lax.scan(
        adam_step,
        initial_state,
        jnp.arange(config.steps, dtype=jnp.int32),
    )
    return terminal_phase, jnp.concatenate(
        (initial_loss[None], post_step_losses), axis=0
    )


def diff_pat_phase_only(
    transfer: Array,
    signature: Array,
    *,
    task: str,
    random_seed: int,
    config: DiffPATConfig = DIFF_PAT_CONFIG,
) -> SynthesisResult:
    """Generate one deterministic phase-only Diff-PAT command.

    Timing starts before the first invocation of the JIT-compiled optimization
    kernel and stops only after both its command and loss trace are synchronized
    to the host.  Consequently ``metadata['jit_sync_wall_s']`` includes JIT
    compilation, all 150 Adam updates, and device synchronization.
    """

    canonical_task = _canonical_task(task)
    matrix, target_amplitude = _validated_inputs(
        transfer, signature, task=canonical_task
    )
    base_seed = int(random_seed)
    effective_seed = (base_seed + int(config.seed_offset)) % (2**32)
    key = jax.random.PRNGKey(effective_seed)
    initial_phase = jax.random.uniform(
        key,
        shape=(matrix.shape[1],),
        minval=-jnp.pi,
        maxval=jnp.pi,
        dtype=jnp.float64,
    )
    matrix_device = jnp.asarray(matrix, dtype=jnp.complex128)
    target_device = jnp.asarray(target_amplitude, dtype=jnp.float64)
    row_capacity = jnp.sum(jnp.abs(matrix_device), axis=1)
    compiled = jax.jit(
        lambda phase: _optimizer_kernel(
            matrix_device,
            target_device,
            phase,
            row_capacity,
            config,
        )
    )

    started = time.perf_counter()
    terminal_phase, loss_history = compiled(initial_phase)
    terminal_phase.block_until_ready()
    loss_history.block_until_ready()
    jit_sync_wall_s = float(time.perf_counter() - started)

    phase = np.angle(np.exp(1j * np.asarray(terminal_phase, dtype=float)))
    losses = np.asarray(loss_history, dtype=float)
    return SynthesisResult(
        method_id="diff_pat_ad",
        display_name="AD",
        phase_rad=phase,
        iterations=int(config.steps),
        converged=None,
        terminal_projective_change_rad=None,
        metadata={
            "algorithm": "Acoustic hologram optimisation using automatic differentiation",
            "task": canonical_task,
            "control_points": int(matrix.shape[0]),
            "phase_only_commands": True,
            "loss_terms": "target_pressure_amplitude_only",
            "amplitude_normalization": "per-control-point phase-aligned row capacity",
            "base_seed": base_seed,
            "effective_seed": int(effective_seed),
            "seed_offset": int(config.seed_offset),
            "steps": int(config.steps),
            "learning_rate": float(config.learning_rate),
            "beta1": float(config.beta1),
            "beta2": float(config.beta2),
            "epsilon": float(config.epsilon),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "jit_sync_wall_s": jit_sync_wall_s,
            "device_transfer_included": True,
            "jit_included": True,
            "device_synchronization_included": True,
            "loss_history": losses,
        },
    )


def diff_pat_terminal_gradient_l2(
    transfer: Array,
    signature: Array,
    phase_rad: Array,
    *,
    task: str,
    config: DiffPATConfig = DIFF_PAT_CONFIG,
) -> float:
    """Evaluate the terminal L2 phase gradient of the declared Diff-PAT loss.

    This is a reporting diagnostic, not part of the timed synthesis command.
    Its value is dimensionless per radian because the optimized amplitude loss
    is dimensionless.  It must therefore not be compared numerically with the
    N m^-1 rad^-1 gradients of the Gor'kov objectives.
    """

    canonical_task = _canonical_task(task)
    matrix, target_amplitude = _validated_inputs(
        transfer, signature, task=canonical_task
    )
    phase = np.asarray(phase_rad, dtype=float)
    if phase.shape != (matrix.shape[1],):
        raise ValueError(
            f"phase_rad must have shape ({matrix.shape[1]},); received {phase.shape}"
        )

    matrix_device = jnp.asarray(matrix, dtype=jnp.complex128)
    target_device = jnp.asarray(target_amplitude, dtype=jnp.float64)
    row_capacity = jnp.sum(jnp.abs(matrix_device), axis=1)
    target_scale = jnp.sqrt(jnp.mean(jnp.square(target_device)))
    normalized_target = target_device / jnp.maximum(
        target_scale, jnp.asarray(config.epsilon, dtype=jnp.float64)
    )

    def loss(candidate_phase: jax.Array) -> jax.Array:
        command = jnp.exp(1j * candidate_phase)
        normalized_amplitude = jnp.abs(matrix_device @ command) / row_capacity
        return jnp.mean(jnp.square(normalized_amplitude - normalized_target))

    gradient = jax.grad(loss)(jnp.asarray(phase, dtype=jnp.float64))
    gradient.block_until_ready()
    return float(np.linalg.norm(np.asarray(gradient, dtype=float)))


__all__ = [
    "DIFF_PAT_CONFIG",
    "DIFF_PAT_CONTROL_POINTS",
    "DiffPATConfig",
    "DiffPATTask",
    "diff_pat_phase_only",
    "diff_pat_terminal_gradient_l2",
]
