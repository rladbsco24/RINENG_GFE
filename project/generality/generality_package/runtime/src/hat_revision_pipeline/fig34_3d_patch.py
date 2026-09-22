"""Fixed three-dimensional display frame for the Figure 4 Triple webs.

This module is deliberately limited to the two web panels in
``Figure_4_triple_branch_evolution``.  It does not run an optimizer, alter a
phase command, or evaluate Gor'kov quantities.  The FE commands already
produced by ``_tri3_bank`` are fitted once as the landmarks of a classical
projective-distance MDS frame.  The common-cap and extended Conventional
commands are then placed by one unchanged Gower out-of-sample map.

For a phase command ``phi`` the normalized state is

    u(phi) = exp(i phi) / sqrt(N)

and the display dissimilarity is the gauge-invariant projector distance

    d_P(u, v) = sqrt(1 - |u^H v|^2).

The full-dimensional paired projector distances already exported by the
pipeline remain the quantitative evidence.  These three coordinates are only
the fixed display frame requested for the web panels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


FE_KEY = "FE"
CONVENTIONAL_KEYS = ("Conventional common", "Conventional extended")
FE_COLOR = "#2878B5"
CONVENTIONAL_COLOR = "#202124"


def _normalized_states(phases: np.ndarray) -> np.ndarray:
    values = np.asarray(phases, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Triple phase commands must have shape (nodes, transducers)")
    states = np.exp(1j * values) / math.sqrt(values.shape[1])
    return np.asarray(states, dtype=np.complex128)


def _paired_projector_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("Paired state banks must have identical shapes")
    similarity = np.clip(np.abs(np.sum(np.conj(left) * right, axis=1)), 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - similarity**2))


def _pairwise_projector_distance(states: np.ndarray) -> np.ndarray:
    similarity = np.clip(np.abs(states @ states.conj().T), 0.0, 1.0)
    distance = np.sqrt(np.maximum(0.0, 1.0 - similarity**2))
    np.fill_diagonal(distance, 0.0)
    return distance


def _phase_banks(bank: Mapping[str, Any]) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    nodes = [(int(ix), int(iy)) for ix, iy in bank["nodes"]]
    objects = bank["objects"]
    phases = {
        FE_KEY: np.stack([np.asarray(objects[node]["FE_common_phase"], dtype=float) for node in nodes]),
        "Conventional common": np.stack(
            [np.asarray(objects[node]["Conventional_common_phase"], dtype=float) for node in nodes]
        ),
        "Conventional extended": np.stack(
            [np.asarray(objects[node]["Conventional_extended_phase"], dtype=float) for node in nodes]
        ),
    }
    shapes = {value.shape for value in phases.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"Triple phase banks do not share one shape: {sorted(shapes)}")
    return nodes, phases


@dataclass(frozen=True)
class TripleFixedFrame3D:
    """One FE-fitted three-axis MDS frame and its frozen OOS map."""

    nodes: tuple[tuple[int, int], ...]
    landmark_states: np.ndarray
    landmark_coordinates: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    squared_distance_column_mean: np.ndarray
    squared_distance_grand_mean: float
    rotation: np.ndarray
    coordinates: Mapping[str, np.ndarray]
    stress: float

    def project_states(self, states: np.ndarray) -> np.ndarray:
        queries = np.asarray(states, dtype=np.complex128)
        if queries.ndim == 1:
            queries = queries[None, :]
        similarity = np.clip(np.abs(queries @ self.landmark_states.conj().T), 0.0, 1.0)
        squared = np.maximum(0.0, 1.0 - similarity**2)
        cross_gram = -0.5 * (
            squared
            - squared.mean(axis=1, keepdims=True)
            - self.squared_distance_column_mean[None, :]
            + self.squared_distance_grand_mean
        )
        raw = cross_gram @ self.eigenvectors / np.sqrt(self.eigenvalues)[None, :]
        return np.asarray(raw @ self.rotation, dtype=float)


def fit_triple_fixed_frame_3d(bank: Mapping[str, Any]) -> TripleFixedFrame3D:
    """Fit FE once and project both Conventional checkpoints without refitting."""

    nodes, phase_banks = _phase_banks(bank)
    state_banks = {name: _normalized_states(value) for name, value in phase_banks.items()}
    landmarks = state_banks[FE_KEY]
    distance = _pairwise_projector_distance(landmarks)
    squared = distance**2
    count = len(landmarks)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ squared @ centering
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    eigenvalues = np.asarray(values[order][:3], dtype=float)
    eigenvectors = np.asarray(vectors[:, order][:, :3], dtype=float)
    tolerance = np.finfo(float).eps * max(1.0, float(values[order[0]])) * count
    if np.any(eigenvalues <= tolerance):
        raise ValueError("The FE Triple landmarks do not support three positive MDS axes")
    raw = eigenvectors * np.sqrt(eigenvalues)[None, :]

    # Rigidly orient the first two display directions with the prescribed
    # connected target chart.  The zero third reference does not flatten or
    # rescale the MDS geometry; orthogonal Procrustes only rotates/refects it.
    values_uv = np.asarray(bank["values"], dtype=float)
    physical = np.asarray([(values_uv[ix], values_uv[iy], 0.0) for ix, iy in nodes])
    physical -= physical.mean(axis=0, keepdims=True)
    physical *= np.linalg.norm(raw) / max(float(np.linalg.norm(physical)), 1.0e-30)
    rotation, _ = orthogonal_procrustes(raw, physical)
    landmark_coordinates = raw @ rotation

    reconstructed = np.linalg.norm(
        landmark_coordinates[:, None, :] - landmark_coordinates[None, :, :], axis=2
    )
    stress = math.sqrt(
        float(np.sum((distance - reconstructed) ** 2))
        / max(float(np.sum(distance**2)), 1.0e-30)
    )
    provisional = TripleFixedFrame3D(
        nodes=tuple(nodes),
        landmark_states=landmarks,
        landmark_coordinates=landmark_coordinates,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        squared_distance_column_mean=squared.mean(axis=0),
        squared_distance_grand_mean=float(squared.mean()),
        rotation=np.asarray(rotation, dtype=float),
        coordinates={},
        stress=float(stress),
    )
    coordinates = {
        FE_KEY: landmark_coordinates,
        **{
            key: provisional.project_states(state_banks[key])
            for key in CONVENTIONAL_KEYS
        },
    }
    frame = TripleFixedFrame3D(**{**provisional.__dict__, "coordinates": coordinates})
    if not np.allclose(
        frame.project_states(landmarks), landmark_coordinates, atol=2.0e-10, rtol=0.0
    ):
        raise AssertionError("Triple out-of-sample map does not reproduce its FE landmarks")
    return frame


def triple_fixed_frame_summary(
    bank: Mapping[str, Any], frame: TripleFixedFrame3D
) -> pd.DataFrame:
    """Return display diagnostics and the unchanged 256-D paired audit."""

    _, phase_banks = _phase_banks(bank)
    states = {name: _normalized_states(value) for name, value in phase_banks.items()}
    rows: list[dict[str, Any]] = []
    total_eigenvalue = float(np.sum(frame.eigenvalues))
    for key in CONVENTIONAL_KEYS:
        distances = _paired_projector_distance(states[key], states[FE_KEY])
        rows.append(
            {
                "checkpoint": key,
                "node_count": len(distances),
                "median_paired_projector_distance": float(np.median(distances)),
                "q25_paired_projector_distance": float(np.quantile(distances, 0.25)),
                "q75_paired_projector_distance": float(np.quantile(distances, 0.75)),
                "fe_landmark_mds_stress_3d": float(frame.stress),
                "mds_eigenvalue_1_fraction_within_retained_3d": float(
                    frame.eigenvalues[0] / total_eigenvalue
                ),
                "mds_eigenvalue_2_fraction_within_retained_3d": float(
                    frame.eigenvalues[1] / total_eigenvalue
                ),
                "mds_eigenvalue_3_fraction_within_retained_3d": float(
                    frame.eigenvalues[2] / total_eigenvalue
                ),
                "display_metric": "projector distance sqrt(1-|u^H v|^2)",
                "landmark_fit": "FE only; fitted once",
                "out_of_sample_map": "fixed Gower projection; no checkpoint refit",
            }
        )
    return pd.DataFrame(rows)


def shared_triple_limits_3d(
    frame: TripleFixedFrame3D,
    keys: Sequence[str] = (FE_KEY, *CONVENTIONAL_KEYS),
    *,
    padding_fraction: float = 0.055,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """One immutable display box shared by both Figure 4 panels."""

    visible = np.vstack([np.asarray(frame.coordinates[key], dtype=float) for key in keys])
    limits: list[tuple[float, float]] = []
    for column in range(3):
        low, high = float(visible[:, column].min()), float(visible[:, column].max())
        pad = float(padding_fraction) * max(high - low, 1.0e-12)
        limits.append((low - pad, high + pad))
    return tuple(limits)  # type: ignore[return-value]


def _grid_edges(nodes: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    lookup = {node: index for index, node in enumerate(nodes)}
    edges: list[tuple[int, int]] = []
    for node, index in lookup.items():
        ix, iy = node
        for neighbor in ((ix + 1, iy), (ix, iy + 1)):
            if neighbor in lookup:
                edges.append((index, lookup[neighbor]))
    return edges


def _draw_web(
    ax: plt.Axes,
    coordinates: np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    color: str,
    marker: str,
    linewidth: float,
    alpha: float,
    size: float,
) -> None:
    for left, right in edges:
        segment = coordinates[[left, right]]
        ax.plot(
            segment[:, 0], segment[:, 1], segment[:, 2],
            color=color, lw=linewidth, alpha=alpha,
            solid_capstyle="round", zorder=2,
        )
    ax.scatter(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
        s=size, marker=marker, facecolors="white", edgecolors=color,
        linewidths=0.9, alpha=min(1.0, alpha + 0.16),
        depthshade=False, zorder=4,
    )


def style_triple_fixed_axis_3d(ax: plt.Axes) -> None:
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.08, 1.0, 0.94))
    ax.tick_params(labelsize=8.5, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color=(0.58, 0.58, 0.58, 0.12), linewidth=0.45)
        axis.set_pane_color((0.992, 0.992, 0.992, 0.50))


def draw_triple_fixed_web_3d(
    ax: plt.Axes,
    frame: TripleFixedFrame3D,
    conventional_key: str,
    *,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    view: tuple[float, float] = (10.0, -165.0),
    pair_guides: bool = True,
) -> None:
    """Draw one FE/Conventional overlay in the same fixed 3-D frame."""

    if conventional_key not in CONVENTIONAL_KEYS:
        raise KeyError(f"Unknown Triple checkpoint bank: {conventional_key}")
    edges = _grid_edges(frame.nodes)
    conventional = np.asarray(frame.coordinates[conventional_key], dtype=float)
    reference = np.asarray(frame.coordinates[FE_KEY], dtype=float)
    _draw_web(
        ax, conventional, edges, color=CONVENTIONAL_COLOR, marker="o",
        linewidth=2.35, alpha=0.58, size=31.0,
    )
    _draw_web(
        ax, reference, edges, color=FE_COLOR, marker="s",
        linewidth=1.55, alpha=0.88, size=23.0,
    )
    if pair_guides:
        count = int(round(math.sqrt(len(frame.nodes))))
        selected = {
            (0, 0), (0, count - 1), (count - 1, 0),
            (count - 1, count - 1), (count // 2, count // 2),
        }
        lookup = {node: index for index, node in enumerate(frame.nodes)}
        for node in sorted(selected.intersection(lookup)):
            index = lookup[node]
            segment = np.vstack([conventional[index], reference[index]])
            ax.plot(
                segment[:, 0], segment[:, 1], segment[:, 2],
                color="0.32", lw=0.72, ls=":", alpha=0.62, zorder=1,
            )
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.view_init(*view)
    style_triple_fixed_axis_3d(ax)
    ax.set_xlabel(r"$q_1$", labelpad=-1, fontsize=9.0)
    ax.set_ylabel(r"$q_2$", labelpad=-1, fontsize=9.0)
    ax.set_zlabel(r"$q_3$", labelpad=-2, fontsize=9.0)


def triple_method_legend() -> list[mpl.lines.Line2D]:
    return [
        mpl.lines.Line2D(
            [0], [0], color=CONVENTIONAL_COLOR, marker="o", markerfacecolor="white",
            lw=2.35, label="Conventional",
        ),
        mpl.lines.Line2D(
            [0], [0], color=FE_COLOR, marker="s", markerfacecolor="white",
            lw=1.55, label="FE",
        ),
    ]


__all__ = [
    "TripleFixedFrame3D",
    "draw_triple_fixed_web_3d",
    "fit_triple_fixed_frame_3d",
    "shared_triple_limits_3d",
    "style_triple_fixed_axis_3d",
    "triple_fixed_frame_summary",
    "triple_method_legend",
]
