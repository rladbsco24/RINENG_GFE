"""Projective branch extraction and fixed-frame plotting for the revision.

This module deliberately separates three operations that were entangled in the
recovered analysis scripts:

1. load the *corrected-sign* endpoint/trajectory bank;
2. discover two endpoint components from projective phase distances, without
   consulting winding labels; and
3. freeze a joint FE frame before placing Conventional iterates out of sample;
   and
4. make matched B1/B2 within-sheet views for the readable main figure, while
   retaining the joint 3-D geometry for Appendix D.

``B1`` and ``B2`` are therefore arbitrary, reproducible component identifiers.
They do not mean positive/negative winding.  Winding columns recovered from the
old artifact are renamed with a ``posthoc_`` prefix and are never used in
clustering, matching, landmark fitting, or trajectory selection.

The public plotting functions accept Matplotlib axes and do not create or save
figures.  This lets the master notebook retain the nested-GridSpec, shared
colour, panel-label, and export conventions of the central HAT notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from rineng_content_id import content_identity as content_id
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import squareform


Array = np.ndarray

FE_METHOD = "Force-Equilibrium"
CONVENTIONAL_METHOD = "Conventional"
COMPONENTS = ("B1", "B2")

# The semantic implementation marker selects the corrected Gor'kov branch bank.
CORRECTED_IMPLEMENTATION_ID = "fixed-fe-rh-landmarks-c-trajectory-v1"

DEFAULT_COMPONENT_COLORS = {"B1": "#2878B5", "B2": "#D95319"}

_POSTHOC_RENAME = {
    "branch": "posthoc_legacy_winding_label",
    "winding": "posthoc_winding",
    "winding_robust": "posthoc_winding_robust",
    "winding_loop_min_over_median": "posthoc_winding_loop_min_over_median",
    "winding_circulation_residual": "posthoc_winding_circulation_residual",
}


def _normalize_states(states: Array) -> Array:
    values = np.asarray(states, dtype=np.complex128)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("states must have shape (samples, actuators)")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("Projective states must be finite and nonzero")
    return values / norms


def projective_similarity(a: Array, b: Array) -> Array:
    """Return the gauge-invariant overlap ``|u^H v|``."""

    u = _normalize_states(a)
    v = _normalize_states(b)
    return np.clip(np.abs(u.conj() @ v.T), 0.0, 1.0)


def projective_distance(a: Array, b: Array) -> Array:
    """Chordal distance between rank-one projectors represented by states."""

    similarity = projective_similarity(a, b)
    return np.sqrt(np.maximum(0.0, 1.0 - similarity**2))


def pairwise_projective_distance(states: Array) -> Array:
    distance = projective_distance(states, states)
    np.fill_diagonal(distance, 0.0)
    return 0.5 * (distance + distance.T)


def _medoid_local_index(states: Array) -> int:
    distance = pairwise_projective_distance(states)
    return int(np.argmin(distance.sum(axis=1)))


def square_array_positions(side: int = 16, pitch_m: float = 0.010) -> Array:
    """Canonical square-array coordinates used by the corrected artifact."""

    axis = np.arange(int(side), dtype=float) - (float(side) - 1.0) / 2.0
    return np.array(
        [(pitch_m * x, pitch_m * y, 0.0) for x in axis for y in axis],
        dtype=float,
    )


def target_demodulated_states(
    phases: Array,
    table: pd.DataFrame,
    *,
    positions_m: Array | None = None,
    frequency_hz: float = 40_000.0,
    sound_speed_m_s: float = 343.0,
) -> Array:
    """Remove only the known propagation carrier from phase commands.

    This is a presentation coordinate, not a continuation or warm start.  It is
    applied after optimization.  The returned complex states have unit norm and
    are compared only through gauge-invariant projective distances.
    """

    phase = np.asarray(phases, dtype=float)
    if phase.ndim == 1:
        phase = phase[None, :]
    if phase.ndim != 2 or len(phase) != len(table):
        raise ValueError("phases and table rows must have matching sample counts")
    positions = square_array_positions() if positions_m is None else np.asarray(positions_m, dtype=float)
    if positions.shape != (phase.shape[1], 3):
        raise ValueError(
            f"positions must have shape {(phase.shape[1], 3)}, got {positions.shape}"
        )
    target_columns = ["target_x_m", "target_y_m", "target_z_m"]
    missing = [column for column in target_columns if column not in table]
    if missing:
        raise ValueError(f"target coordinate columns are missing: {missing}")
    targets = table[target_columns].to_numpy(dtype=float)
    distance = np.linalg.norm(targets[:, None, :] - positions[None, :, :], axis=2)
    wavelength = float(sound_speed_m_s) / float(frequency_hz)
    carrier = -(2.0 * np.pi / wavelength) * distance
    states = np.exp(1j * (phase - carrier)) / math.sqrt(phase.shape[1])
    return _normalize_states(states)


def _rename_posthoc_columns(table: pd.DataFrame) -> pd.DataFrame:
    rename = {
        source: target
        for source, target in _POSTHOC_RENAME.items()
        if source in table.columns and target not in table.columns
    }
    return table.rename(columns=rename).copy()


class _ArtifactReader:
    """Read a recovered branch artifact from a directory or ZIP in memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zip: zipfile.ZipFile | None = None
        if self.path.is_file() and self.path.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.path)
            self._names = tuple(name for name in self._zip.namelist() if not name.endswith("/"))
        elif self.path.is_dir():
            self._names = tuple(
                str(candidate.relative_to(self.path))
                for candidate in self.path.rglob("*")
                if candidate.is_file()
            )
        else:
            raise FileNotFoundError(self.path)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "_ArtifactReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _match(self, suffix: str) -> str:
        normalized = suffix.replace("\\", "/")
        matches = [name for name in self._names if name.replace("\\", "/").endswith(normalized)]
        if not matches:
            # The master notebook copies the corrected artifact into a directory
            # whose root *is* the data directory.  Accept that layout only when
            # the requested basename is unique, so recursive bundles with two
            # similarly named result sets still fail closed.
            basename = Path(normalized).name
            matches = [name for name in self._names if Path(name).name == basename]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one artifact member ending in {suffix!r}; found {len(matches)}"
            )
        return matches[0]

    def bytes(self, suffix: str) -> bytes:
        name = self._match(suffix)
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def json(self, suffix: str) -> dict[str, Any]:
        return json.loads(self.bytes(suffix).decode("utf-8"))

    def csv(self, suffix: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(self.bytes(suffix).decode("utf-8")))

    def npz(self, suffix: str) -> dict[str, Array]:
        with np.load(BytesIO(self.bytes(suffix)), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}


@dataclass(frozen=True)
class CorrectedBranchArtifact:
    """Raw corrected-sign bank, before component discovery or embedding."""

    anchor_table: pd.DataFrame
    anchor_phases: Array
    trajectory_table: pd.DataFrame
    checkpoints: Array
    trajectory_snapshots: Array
    anchor_config: Mapping[str, Any]
    evolution_metadata: Mapping[str, Any]
    source: str

    def anchor_states(self, method: str = FE_METHOD) -> tuple[pd.DataFrame, Array]:
        selector = self.anchor_table["method"].eq(method).to_numpy()
        table = self.anchor_table.loc[selector].reset_index(drop=True)
        phases = np.asarray(self.anchor_phases)[selector]
        return table, target_demodulated_states(phases, table)

    def trajectory_states(self) -> Array:
        """Return demodulated states with shape ``(runs, checkpoints, N)``."""

        snapshots = np.asarray(self.trajectory_snapshots, dtype=float)
        runs, checkpoints, actuators = snapshots.shape
        repeated = self.trajectory_table.loc[
            self.trajectory_table.index.repeat(checkpoints)
        ].reset_index(drop=True)
        flat = target_demodulated_states(snapshots.reshape(runs * checkpoints, actuators), repeated)
        return flat.reshape(runs, checkpoints, actuators)


def _validate_corrected_provenance(
    source: Path,
    anchor_config: Mapping[str, Any],
    evolution_metadata: Mapping[str, Any],
) -> None:
    source_text = str(source).replace("\\", "/").lower()
    if "core_volume_sheets" in source_text or "core-volume-sheet" in source_text:
        raise ValueError("Stale core-volume branch-sheet outputs are forbidden")

    if anchor_config.get("implementation_id") != CORRECTED_IMPLEMENTATION_ID:
        raise ValueError("Unexpected corrected-branch implementation")
    definition = str(evolution_metadata.get("landmark_definition", ""))
    if "corrected-Gorkov" not in definition:
        raise ValueError("Corrected-Gor'kov landmark provenance marker is missing")
    if bool(evolution_metadata.get("conventional_in_landmark_fit", True)):
        raise ValueError("Conventional states must not enter the fixed landmark fit")
    if bool(evolution_metadata.get("stage_refitting", True)):
        raise ValueError("The fixed landmark frame must not be refit by iteration")


def load_corrected_branch_artifact(
    path: str | Path,
    *,
    strict_provenance: bool = True,
) -> CorrectedBranchArtifact:
    """Load only raw endpoint/trajectory arrays from the corrected artifact.

    The artifact's precomputed winding-labelled medoids and 3-D coordinates are
    intentionally ignored.  This function rebuilds all main-text component
    assignments from the raw corrected-sign endpoint phases.
    """

    source = Path(path)
    with _ArtifactReader(source) as reader:
        anchor_config = reader.json("data/anchor_config.json")
        evolution_metadata = reader.json("data/evolution_metadata.json")
        if strict_provenance:
            _validate_corrected_provenance(source, anchor_config, evolution_metadata)
        anchor_table = _rename_posthoc_columns(reader.csv("data/anchor_endpoints.csv"))
        anchor_phases = reader.npz("data/anchor_phases.npz")["phases"]
        trajectory_table = _rename_posthoc_columns(
            reader.csv("data/conventional_trajectories.csv")
        )
        trajectory_data = reader.npz("data/conventional_snapshots.npz")

    required_anchor = {"state_index", "method", "ix", "iy", "target_id"}
    required_trajectory = {"trajectory_index", "method", "ix", "iy", "target_id"}
    if not required_anchor.issubset(anchor_table.columns):
        raise ValueError(f"Anchor schema is missing {sorted(required_anchor - set(anchor_table))}")
    if not required_trajectory.issubset(trajectory_table.columns):
        raise ValueError(
            f"Trajectory schema is missing {sorted(required_trajectory - set(trajectory_table))}"
        )

    anchor_order = anchor_table["state_index"].to_numpy(dtype=int)
    trajectory_order = trajectory_table["trajectory_index"].to_numpy(dtype=int)
    if set(anchor_order) != set(range(len(anchor_table))):
        raise ValueError("state_index is not a permutation of the phase-bank rows")
    if set(trajectory_order) != set(range(len(trajectory_table))):
        raise ValueError("trajectory_index is not a permutation of the snapshot-bank rows")
    anchor_sort = np.argsort(anchor_order)
    trajectory_sort = np.argsort(trajectory_order)
    anchor_table = anchor_table.iloc[anchor_sort].reset_index(drop=True)
    trajectory_table = trajectory_table.iloc[trajectory_sort].reset_index(drop=True)
    anchor_phases = np.asarray(anchor_phases, dtype=float)[anchor_order[anchor_sort]]
    snapshots = np.asarray(trajectory_data["snapshots"], dtype=float)[trajectory_order[trajectory_sort]]
    checkpoints = np.asarray(trajectory_data["checkpoints"], dtype=int)

    if anchor_phases.shape != (len(anchor_table), 256):
        raise ValueError(f"Unexpected anchor phase shape: {anchor_phases.shape}")
    if snapshots.shape != (len(trajectory_table), len(checkpoints), 256):
        raise ValueError(f"Unexpected Conventional snapshot shape: {snapshots.shape}")
    if not np.all(np.diff(checkpoints) > 0):
        raise ValueError("Conventional checkpoints must be strictly increasing")

    return CorrectedBranchArtifact(
        anchor_table=anchor_table,
        anchor_phases=anchor_phases,
        trajectory_table=trajectory_table,
        checkpoints=checkpoints,
        trajectory_snapshots=snapshots,
        anchor_config=anchor_config,
        evolution_metadata=evolution_metadata,
        source=str(source),
    )


def _local_two_component_labels(states: Array) -> Array:
    values = _normalize_states(states)
    if len(values) < 2:
        raise ValueError("At least two endpoints are needed at every target")
    if len(values) == 2:
        return np.array([0, 1], dtype=int)
    distance = pairwise_projective_distance(values)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    labels = cut_tree(tree, n_clusters=[2]).reshape(-1).astype(int)
    if set(labels) != {0, 1}:
        raise RuntimeError("The target-local endpoint bank did not split into two components")
    return labels


def _root_component_order(medoids: Array) -> tuple[int, int]:
    """Choose reproducible names using a gauge-invariant deterministic probe."""

    values = _normalize_states(medoids)
    index = np.arange(values.shape[1], dtype=float)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    probe = np.exp(2j * np.pi * golden * index) / math.sqrt(values.shape[1])
    scores = np.abs(values.conj() @ probe)

    def canonical_digest(state: Array) -> str:
        pivot = int(np.argmax(np.abs(state)))
        canonical = state * np.exp(-1j * np.angle(state[pivot]))
        rounded = np.round(np.column_stack([canonical.real, canonical.imag]), 12)
        return content_id(np.ascontiguousarray(rounded).tobytes()).hexdigest()

    order = sorted(range(2), key=lambda i: (float(scores[i]), canonical_digest(values[i])))
    return int(order[0]), int(order[1])


def _grid_edges(nodes: Sequence[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    node_set = set(nodes)
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for node in sorted(node_set, key=lambda value: (value[1], value[0])):
        ix, iy = node
        for candidate in ((ix + 1, iy), (ix, iy + 1)):
            if candidate in node_set:
                edges.append((node, candidate))
    return edges


class _DisjointSet:
    def __init__(self, nodes: Iterable[tuple[int, int]]):
        self.parent = {node: node for node in nodes}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        self.parent[root_b] = root_a
        return True


@dataclass(frozen=True)
class ComponentBank:
    """Two target-indexed FE components discovered without winding labels."""

    endpoint_table: pd.DataFrame
    endpoint_states: Array
    medoid_table: pd.DataFrame
    medoid_states: Array
    edge_table: pd.DataFrame
    root_target: tuple[int, int]

    @property
    def labels(self) -> tuple[str, str]:
        return COMPONENTS

    @property
    def anchors(self) -> Mapping[str, Array]:
        """Component anchor states for sweep producers and independent banks."""

        return {component: self.states_for(component) for component in COMPONENTS}

    def states_for(self, component: str) -> Array:
        if component not in COMPONENTS:
            raise KeyError(component)
        selector = self.medoid_table["component"].eq(component).to_numpy()
        return np.asarray(self.medoid_states)[selector]

    def target_medoid(self, ix: int, iy: int, component: str) -> Array:
        selector = (
            self.medoid_table["ix"].eq(int(ix))
            & self.medoid_table["iy"].eq(int(iy))
            & self.medoid_table["component"].eq(component)
        ).to_numpy()
        indices = np.flatnonzero(selector)
        if len(indices) != 1:
            raise KeyError((ix, iy, component))
        return np.asarray(self.medoid_states)[indices[0]]

    def assign(self, states: Array) -> "ComponentAssignment":
        """Assign queries by nearest full-dimensional component anchor.

        This generic assignment is useful for parameter sweeps without a shared
        target grid.  Target-indexed analyses should prefer ``target_medoid``.
        """

        queries = _normalize_states(states)
        per_component = np.column_stack(
            [
                np.min(projective_distance(queries, self.states_for(component)), axis=1)
                for component in COMPONENTS
            ]
        )
        selected = np.argmin(per_component, axis=1)
        labels = np.asarray(COMPONENTS, dtype=object)[selected]
        ordered = np.sort(per_component, axis=1)
        return ComponentAssignment(
            labels=labels,
            distances=per_component,
            margins=ordered[:, 1] - ordered[:, 0],
        )


@dataclass(frozen=True)
class ComponentAssignment:
    labels: Array
    distances: Array
    margins: Array


def discover_fe_components(
    table_or_artifact: pd.DataFrame | CorrectedBranchArtifact,
    states: Array | None = None,
    *,
    method: str = FE_METHOD,
) -> ComponentBank:
    """Recover coherent ``B1/B2`` sheets from target-local endpoint clusters.

    Target-local clusters are obtained by average-linkage clustering of the
    full-dimensional projector distance.  Local identities are then propagated
    over a maximum-confidence spanning tree of the physical target grid.  No
    winding, objective sign, or embedding coordinate participates.
    """

    if isinstance(table_or_artifact, CorrectedBranchArtifact):
        if states is not None:
            raise ValueError("states must be omitted when an artifact is supplied")
        table, values = table_or_artifact.anchor_states(method)
    else:
        table = _rename_posthoc_columns(table_or_artifact)
        if states is None:
            raise ValueError("states are required with an endpoint table")
        values = _normalize_states(states)
        if len(table) != len(values):
            raise ValueError("endpoint table and state array lengths differ")
        if "method" in table:
            selector = table["method"].eq(method).to_numpy()
            table = table.loc[selector].reset_index(drop=True)
            values = values[selector]
        else:
            table = table.reset_index(drop=True)

    required = {"ix", "iy", "xi", "eta", "target_id"}
    if not required.issubset(table.columns):
        raise ValueError(f"Endpoint table is missing {sorted(required - set(table))}")
    if len(table) == 0:
        raise ValueError(f"No endpoint records for {method}")
    values = _normalize_states(values)
    table = table.reset_index(drop=True).copy()
    table["_endpoint_position"] = np.arange(len(table), dtype=int)

    local_labels = np.full(len(table), -1, dtype=int)
    local_medoids: dict[tuple[int, int], Array] = {}
    local_medoid_positions: dict[tuple[int, int], tuple[int, int]] = {}
    target_rows: dict[tuple[int, int], pd.Series] = {}

    for (ix, iy), group in table.groupby(["ix", "iy"], sort=True):
        node = (int(ix), int(iy))
        positions = group["_endpoint_position"].to_numpy(dtype=int)
        labels = _local_two_component_labels(values[positions])
        local_labels[positions] = labels
        medoid_states: list[Array] = []
        medoid_positions: list[int] = []
        for local_component in (0, 1):
            member_positions = positions[labels == local_component]
            chosen = int(member_positions[_medoid_local_index(values[member_positions])])
            medoid_states.append(values[chosen])
            medoid_positions.append(chosen)
        local_medoids[node] = np.stack(medoid_states)
        local_medoid_positions[node] = (medoid_positions[0], medoid_positions[1])
        target_rows[node] = group.iloc[0]

    if np.any(local_labels < 0):
        raise AssertionError("A target-local component label was not assigned")

    nodes = sorted(local_medoids, key=lambda value: (value[1], value[0]))
    edges = _grid_edges(nodes)
    if len(nodes) > 1 and not edges:
        raise ValueError("Target grid has no adjacent cells")

    edge_records: list[dict[str, Any]] = []
    for a, b in edges:
        cost = projective_distance(local_medoids[a], local_medoids[b])
        same = float(cost[0, 0] + cost[1, 1])
        swapped = float(cost[0, 1] + cost[1, 0])
        edge_records.append(
            {
                "a": a,
                "b": b,
                "preferred_swap": bool(swapped < same),
                "preferred_cost": min(same, swapped),
                "alternative_cost": max(same, swapped),
                "margin": abs(same - swapped),
            }
        )

    # Use the most discriminating edges first.  This makes label propagation
    # insensitive to DataFrame ordering and avoids giving a weak boundary edge
    # control over the entire grid.
    disjoint = _DisjointSet(nodes)
    tree_records: list[dict[str, Any]] = []
    for record in sorted(
        edge_records,
        key=lambda row: (-float(row["margin"]), row["a"][1], row["a"][0], row["b"][1], row["b"][0]),
    ):
        if disjoint.union(record["a"], record["b"]):
            tree_records.append(record)
    if len(tree_records) != max(0, len(nodes) - 1):
        raise ValueError("Target graph is disconnected")

    root = min(
        nodes,
        key=lambda node: (
            float(target_rows[node]["xi"]) ** 2 + float(target_rows[node]["eta"]) ** 2,
            node[1],
            node[0],
        ),
    )
    order_by_node: dict[tuple[int, int], tuple[int, int]] = {
        root: _root_component_order(local_medoids[root])
    }
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], bool]]] = {
        node: [] for node in nodes
    }
    for record in tree_records:
        a, b, swap = record["a"], record["b"], bool(record["preferred_swap"])
        adjacency[a].append((b, swap))
        adjacency[b].append((a, swap))
    queue = [root]
    while queue:
        a = queue.pop(0)
        for b, swap in adjacency[a]:
            if b in order_by_node:
                continue
            a_order = order_by_node[a]
            order_by_node[b] = (
                (1 - a_order[0], 1 - a_order[1]) if swap else a_order
            )
            queue.append(b)

    component_values = np.empty(len(table), dtype=object)
    medoid_rows: list[dict[str, Any]] = []
    medoid_states_ordered: list[Array] = []
    for node in nodes:
        order = order_by_node[node]
        inverse = {int(local): COMPONENTS[index] for index, local in enumerate(order)}
        group_positions = table.loc[
            table["ix"].eq(node[0]) & table["iy"].eq(node[1]), "_endpoint_position"
        ].to_numpy(dtype=int)
        for position in group_positions:
            component_values[position] = inverse[int(local_labels[position])]
        for component_index, component in enumerate(COMPONENTS):
            local_component = int(order[component_index])
            source_position = int(local_medoid_positions[node][local_component])
            source = table.iloc[source_position].to_dict()
            source.pop("_endpoint_position", None)
            medoid_rows.append(
                source
                | {
                    "component": component,
                    "local_component": local_component,
                    "component_size": int(
                        np.sum((local_labels == local_component) & table["ix"].eq(node[0]).to_numpy() & table["iy"].eq(node[1]).to_numpy())
                    ),
                    "source_endpoint_position": source_position,
                }
            )
            medoid_states_ordered.append(values[source_position])

    table["component"] = component_values
    table["local_component"] = local_labels
    table = table.drop(columns=["_endpoint_position"])
    medoid_table = pd.DataFrame(medoid_rows)
    medoid_states_array = np.stack(medoid_states_ordered)
    sort_order = np.lexsort(
        (
            medoid_table["ix"].to_numpy(dtype=int),
            medoid_table["iy"].to_numpy(dtype=int),
            medoid_table["component"].map({"B1": 0, "B2": 1}).to_numpy(dtype=int),
        )
    )
    medoid_table = medoid_table.iloc[sort_order].reset_index(drop=True)
    medoid_states_array = medoid_states_array[sort_order]

    tree_keys = {(record["a"], record["b"]) for record in tree_records}
    diagnostic_rows: list[dict[str, Any]] = []
    for record in edge_records:
        a, b = record["a"], record["b"]
        a_order, b_order = order_by_node[a], order_by_node[b]
        realized_swap = bool(b_order[0] != a_order[0])
        diagnostic_rows.append(
            {
                "a_ix": a[0], "a_iy": a[1], "b_ix": b[0], "b_iy": b[1],
                "preferred_swap": bool(record["preferred_swap"]),
                "realized_swap": realized_swap,
                "preferred_consistent": bool(realized_swap == record["preferred_swap"]),
                "preferred_cost": float(record["preferred_cost"]),
                "alternative_cost": float(record["alternative_cost"]),
                "assignment_margin": float(record["margin"]),
                "in_spanning_tree": bool((a, b) in tree_keys),
            }
        )
    edge_table = pd.DataFrame(diagnostic_rows)

    return ComponentBank(
        endpoint_table=table,
        endpoint_states=values,
        medoid_table=medoid_table,
        medoid_states=medoid_states_array,
        edge_table=edge_table,
        root_target=root,
    )


@dataclass(frozen=True)
class FrozenBranchFrame2D:
    """Classical-MDS frame fitted once to FE medoids only."""

    landmark_table: pd.DataFrame
    landmark_states: Array
    coordinates: Array
    eigenvectors: Array
    eigenvalues: Array
    squared_distance_column_mean: Array
    squared_distance_grand_mean: float
    rotation: Array
    stress: float
    display_scale: float = 1.0

    def project(self, states: Array) -> Array:
        """Place new states out of sample; always return shape ``(n, 2)``."""

        queries = _normalize_states(states)
        squared = projective_distance(queries, self.landmark_states) ** 2
        cross_gram = -0.5 * (
            squared
            - squared.mean(axis=1, keepdims=True)
            - self.squared_distance_column_mean[None, :]
            + self.squared_distance_grand_mean
        )
        raw = cross_gram @ self.eigenvectors / np.sqrt(self.eigenvalues)[None, :]
        return np.asarray(raw @ self.rotation * float(self.display_scale), dtype=float)


def _target_principal_coordinate(table: pd.DataFrame) -> Array:
    physical = table[["xi", "eta"]].to_numpy(dtype=float)
    physical -= physical.mean(axis=0, keepdims=True)
    if not np.any(physical):
        return np.arange(len(table), dtype=float)
    _, _, right = np.linalg.svd(physical, full_matrices=False)
    loading = right[0]
    pivot = int(np.argmax(np.abs(loading)))
    if loading[pivot] < 0.0:
        loading = -loading
    return physical @ loading


def fit_fixed_2d_frame(bank: ComponentBank) -> FrozenBranchFrame2D:
    """Fit and deterministically orient a joint 2-D FE projective frame."""

    states = _normalize_states(bank.medoid_states)
    distance = pairwise_projective_distance(states)
    squared = distance**2
    count = len(states)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ squared @ centering
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    tolerance = np.finfo(float).eps * max(1.0, float(values[0])) * count
    if np.sum(values > tolerance) < 2:
        raise ValueError("FE landmark distances do not support a two-dimensional MDS frame")
    eigenvalues = np.asarray(values[:2], dtype=float)
    eigenvectors = np.asarray(vectors[:, :2], dtype=float)
    raw_coordinates = eigenvectors * np.sqrt(eigenvalues)[None, :]

    component_reference = bank.medoid_table["component"].map({"B1": -1.0, "B2": 1.0}).to_numpy()
    target_reference = _target_principal_coordinate(bank.medoid_table)
    reference = np.column_stack([component_reference, target_reference])
    reference -= reference.mean(axis=0, keepdims=True)
    for column in range(2):
        norm = float(np.linalg.norm(reference[:, column]))
        if norm > 0.0:
            reference[:, column] /= norm
    reference *= np.linalg.norm(raw_coordinates) / max(np.linalg.norm(reference), 1.0e-30)
    rotation, _ = orthogonal_procrustes(raw_coordinates, reference)
    coordinates = raw_coordinates @ rotation

    reconstructed = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=2
    )
    denominator = max(float(np.sum(distance**2)), 1.0e-30)
    stress = math.sqrt(float(np.sum((distance - reconstructed) ** 2)) / denominator)

    frame = FrozenBranchFrame2D(
        landmark_table=bank.medoid_table.copy(),
        landmark_states=states,
        coordinates=coordinates,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        squared_distance_column_mean=squared.mean(axis=0),
        squared_distance_grand_mean=float(squared.mean()),
        rotation=rotation,
        stress=float(stress),
        display_scale=1.0,
    )
    # This also guards the Gower out-of-sample centring convention.
    if not np.allclose(frame.project(states), coordinates, rtol=0.0, atol=2.0e-10):
        raise AssertionError("Out-of-sample projection does not reproduce FE landmarks")
    return frame


@dataclass(frozen=True)
class MatchedComponentFrames2D:
    """Matched within-sheet frames used for the readable main-text web.

    B1 and B2 are fitted separately so both target directions remain visible.
    Each raw projective MDS is then mapped to the same dimensionless target-grid
    coordinates by an orthogonal Procrustes transform and an isotropic scale.
    Thus the two axes can be displayed with identical limits without implying
    that B1/B2 are winding polarities or that their cross-component separation
    is represented in these within-sheet panels.
    """

    frames: Mapping[str, FrozenBranchFrame2D]
    common_limits: tuple[tuple[float, float], tuple[float, float]]

    @property
    def anchors(self) -> Mapping[str, Array]:
        return {component: self.frames[component].landmark_states for component in COMPONENTS}

    def assign(self, states: Array) -> ComponentAssignment:
        queries = _normalize_states(states)
        distance = np.column_stack(
            [
                np.min(projective_distance(queries, self.frames[c].landmark_states), axis=1)
                for c in COMPONENTS
            ]
        )
        selected = np.argmin(distance, axis=1)
        ordered = np.sort(distance, axis=1)
        return ComponentAssignment(
            labels=np.asarray(COMPONENTS, dtype=object)[selected],
            distances=distance,
            margins=ordered[:, 1] - ordered[:, 0],
        )

    def project(self, states: Array, components: Sequence[str] | str | None = None) -> Array:
        """Project queries in their component-conditioned within-sheet frame."""

        queries = _normalize_states(states)
        if components is None:
            labels = self.assign(queries).labels
        elif isinstance(components, str):
            labels = np.full(len(queries), components, dtype=object)
        else:
            labels = np.asarray(components, dtype=object)
            if labels.shape != (len(queries),):
                raise ValueError("components must have one label per query")
        output = np.empty((len(queries), 2), dtype=float)
        for component in COMPONENTS:
            selector = labels == component
            if np.any(selector):
                output[selector] = self.frames[component].project(queries[selector])
        unknown = set(str(value) for value in labels) - set(COMPONENTS)
        if unknown:
            raise KeyError(f"Unknown component labels: {sorted(unknown)}")
        return output


def _fit_target_matched_component_frame(
    table: pd.DataFrame,
    states: Array,
) -> FrozenBranchFrame2D:
    values = _normalize_states(states)
    distance = pairwise_projective_distance(values)
    squared = distance**2
    count = len(values)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ squared @ centering
    gram = 0.5 * (gram + gram.T)
    eigenvalues_all, eigenvectors_all = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues_all)[::-1]
    eigenvalues = np.asarray(eigenvalues_all[order][:2], dtype=float)
    eigenvectors = np.asarray(eigenvectors_all[:, order][:, :2], dtype=float)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Within-sheet distances lack two positive MDS axes")
    raw = eigenvectors * np.sqrt(eigenvalues)[None, :]

    reference = table[["xi", "eta"]].to_numpy(dtype=float)
    reference -= reference.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(reference) < 2:
        raise ValueError("Target grid is not two-dimensional")
    rotation, singular_sum = orthogonal_procrustes(raw, reference)
    isotropic_scale = float(singular_sum) / max(float(np.sum(raw**2)), 1.0e-30)
    coordinates = raw @ rotation * isotropic_scale

    reconstructed = np.linalg.norm(raw[:, None, :] - raw[None, :, :], axis=2)
    stress = math.sqrt(
        float(np.sum((distance - reconstructed) ** 2))
        / max(float(np.sum(distance**2)), 1.0e-30)
    )
    frame = FrozenBranchFrame2D(
        landmark_table=table.reset_index(drop=True).copy(),
        landmark_states=values,
        coordinates=coordinates,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        squared_distance_column_mean=squared.mean(axis=0),
        squared_distance_grand_mean=float(squared.mean()),
        rotation=rotation,
        stress=float(stress),
        display_scale=isotropic_scale,
    )
    if not np.allclose(frame.project(values), coordinates, atol=2.0e-10, rtol=0.0):
        raise AssertionError("Within-sheet OOS projection does not reproduce landmarks")
    return frame


def fit_matched_component_frames_2d(bank: ComponentBank) -> MatchedComponentFrames2D:
    """Fit B1/B2 within-sheet frames with common target-grid presentation axes."""

    frames: dict[str, FrozenBranchFrame2D] = {}
    for component in COMPONENTS:
        selector = bank.medoid_table["component"].eq(component).to_numpy()
        table = bank.medoid_table.loc[selector].sort_values(["iy", "ix"]).reset_index(drop=True)
        # Sorting reset the indices, so recover states using a key lookup rather
        # than relying on DataFrame row order.
        state_lookup = {
            (int(row.ix), int(row.iy)): bank.medoid_states[index]
            for index, row in enumerate(bank.medoid_table.itertuples(index=False))
            if str(row.component) == component
        }
        states = np.stack(
            [state_lookup[(int(row.ix), int(row.iy))] for row in table.itertuples(index=False)]
        )
        frames[component] = _fit_target_matched_component_frame(table, states)

    all_coordinates = np.vstack([frames[c].coordinates for c in COMPONENTS])
    padding = 0.06 * np.maximum(np.ptp(all_coordinates, axis=0), 1.0)
    minimum = np.min(all_coordinates, axis=0) - padding
    maximum = np.max(all_coordinates, axis=0) + padding
    return MatchedComponentFrames2D(
        frames=frames,
        common_limits=((float(minimum[0]), float(maximum[0])), (float(minimum[1]), float(maximum[1]))),
    )


@dataclass(frozen=True)
class FrozenBranchFrame3D:
    """Joint FE projective frame retained for the Appendix-D 3-D view."""

    landmark_table: pd.DataFrame
    landmark_states: Array
    coordinates: Array
    eigenvectors: Array
    eigenvalues: Array
    squared_distance_column_mean: Array
    squared_distance_grand_mean: float
    rotation: Array
    stress: float

    def project(self, states: Array) -> Array:
        queries = _normalize_states(states)
        squared = projective_distance(queries, self.landmark_states) ** 2
        cross_gram = -0.5 * (
            squared
            - squared.mean(axis=1, keepdims=True)
            - self.squared_distance_column_mean[None, :]
            + self.squared_distance_grand_mean
        )
        raw = cross_gram @ self.eigenvectors / np.sqrt(self.eigenvalues)[None, :]
        return np.asarray(raw @ self.rotation, dtype=float)


def fit_fixed_3d_frame(bank: ComponentBank) -> FrozenBranchFrame3D:
    """Fit a joint three-axis frame showing target directions and separation."""

    states = _normalize_states(bank.medoid_states)
    distance = pairwise_projective_distance(states)
    squared = distance**2
    count = len(states)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ squared @ centering
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    eigenvalues = np.asarray(values[order][:3], dtype=float)
    eigenvectors = np.asarray(vectors[:, order][:, :3], dtype=float)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("FE landmark distances do not support three positive MDS axes")
    raw = eigenvectors * np.sqrt(eigenvalues)[None, :]

    component_reference = bank.medoid_table["component"].map({"B1": -1.0, "B2": 1.0}).to_numpy()
    reference = np.column_stack(
        [
            bank.medoid_table["xi"].to_numpy(dtype=float),
            bank.medoid_table["eta"].to_numpy(dtype=float),
            component_reference,
        ]
    )
    reference -= reference.mean(axis=0, keepdims=True)
    reference *= np.linalg.norm(raw) / max(np.linalg.norm(reference), 1.0e-30)
    rotation, _ = orthogonal_procrustes(raw, reference)
    coordinates = raw @ rotation

    reconstructed = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=2
    )
    stress = math.sqrt(
        float(np.sum((distance - reconstructed) ** 2))
        / max(float(np.sum(distance**2)), 1.0e-30)
    )
    frame = FrozenBranchFrame3D(
        landmark_table=bank.medoid_table.copy(),
        landmark_states=states,
        coordinates=coordinates,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        squared_distance_column_mean=squared.mean(axis=0),
        squared_distance_grand_mean=float(squared.mean()),
        rotation=rotation,
        stress=float(stress),
    )
    if not np.allclose(frame.project(states), coordinates, atol=2.0e-10, rtol=0.0):
        raise AssertionError("Three-dimensional OOS projection does not reproduce landmarks")
    return frame


def project_out_of_sample(states: Array, frame: FrozenBranchFrame2D) -> Array:
    """Functional alias used by sweep/appendix producers."""

    return frame.project(states)


def project_conventional_evolution(
    artifact: CorrectedBranchArtifact,
    bank: ComponentBank,
    frame: FrozenBranchFrame2D,
) -> pd.DataFrame:
    """Select terminal C medoids and project their uninterrupted trajectories.

    Terminal Conventional endpoints are assigned to the nearest same-target FE
    component in the full 256-D projective metric.  One local medoid per target
    and recovered component is selected, and that *same optimizer trajectory*
    is shown at every checkpoint.  Winding never enters the assignment.
    """

    trajectory_table = artifact.trajectory_table.reset_index(drop=True)
    if not trajectory_table["method"].eq(CONVENTIONAL_METHOD).all():
        raise ValueError("The trajectory bank contains a non-Conventional method")
    states = artifact.trajectory_states()
    final_states = states[:, -1, :]
    checkpoints = np.asarray(artifact.checkpoints, dtype=int)

    lookup: dict[tuple[int, int, str], tuple[int, Array]] = {}
    for index, row in bank.medoid_table.iterrows():
        lookup[(int(row["ix"]), int(row["iy"]), str(row["component"]))] = (
            int(index),
            bank.medoid_states[int(index)],
        )

    assigned = np.empty(len(trajectory_table), dtype=object)
    for (ix, iy), group in trajectory_table.groupby(["ix", "iy"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        target_medoids = np.stack(
            [lookup[(int(ix), int(iy), component)][1] for component in COMPONENTS]
        )
        distance = projective_distance(final_states[positions], target_medoids)
        nearest = np.argmin(distance, axis=1)
        assigned[positions] = np.asarray(COMPONENTS, dtype=object)[nearest]

    selected: list[tuple[int, str]] = []
    for (ix, iy), group in trajectory_table.groupby(["ix", "iy"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        for component in COMPONENTS:
            members = positions[assigned[positions] == component]
            if len(members) == 0:
                raise RuntimeError(
                    f"Conventional terminal bank did not recover {component} at {(int(ix), int(iy))}"
                )
            chosen = int(members[_medoid_local_index(final_states[members])])
            selected.append((chosen, component))

    rows: list[dict[str, Any]] = []
    for trajectory_index, component in selected:
        source = trajectory_table.iloc[trajectory_index]
        ix, iy = int(source["ix"]), int(source["iy"])
        matched_index, matched_state = lookup[(ix, iy, component)]
        component_selector = bank.medoid_table["component"].eq(component).to_numpy()
        component_states = bank.medoid_states[component_selector]
        projected = frame.project(states[trajectory_index])
        target_similarity = projective_similarity(states[trajectory_index], matched_state).reshape(-1)
        web_similarity = np.max(
            projective_similarity(states[trajectory_index], component_states), axis=1
        )
        for checkpoint_index, iteration in enumerate(checkpoints):
            record = source.to_dict()
            record.update(
                {
                    "component": component,
                    "iteration": int(iteration),
                    "checkpoint_index": int(checkpoint_index),
                    "selected_trajectory_index": int(trajectory_index),
                    "matched_fe_landmark_index": int(matched_index),
                    "fixed_mds1": float(projected[checkpoint_index, 0]),
                    "fixed_mds2": float(projected[checkpoint_index, 1]),
                    "similarity_to_target_fe": float(target_similarity[checkpoint_index]),
                    "distance_to_target_fe": float(
                        math.sqrt(max(0.0, 1.0 - target_similarity[checkpoint_index] ** 2))
                    ),
                    "similarity_to_component_web": float(web_similarity[checkpoint_index]),
                    "distance_to_component_web": float(
                        math.sqrt(max(0.0, 1.0 - web_similarity[checkpoint_index] ** 2))
                    ),
                }
            )
            rows.append(record)
    result = pd.DataFrame(rows)
    return result.sort_values(["component", "iy", "ix", "iteration"]).reset_index(drop=True)


def summarize_full_dimensional_evolution(
    evolution: pd.DataFrame,
    *,
    metric: str = "distance_to_target_fe",
) -> pd.DataFrame:
    """Median/IQR summary for text, Appendix D, or a compact table."""

    if metric not in evolution:
        raise KeyError(metric)
    rows: list[dict[str, Any]] = []
    for (component, iteration), group in evolution.groupby(["component", "iteration"], sort=True):
        values = group[metric].to_numpy(dtype=float)
        rows.append(
            {
                "component": component,
                "iteration": int(iteration),
                "n": int(len(values)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def component_log_iteration_slopes(
    evolution: pd.DataFrame,
    *,
    metric: str = "distance_to_target_fe",
) -> pd.DataFrame:
    """Report component-wise slopes without turning them into a main figure."""

    rows: list[dict[str, Any]] = []
    for component, group in evolution.groupby("component", sort=True):
        x = np.log10(group["iteration"].to_numpy(dtype=float))
        y = group[metric].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        correlation = float(np.corrcoef(x, y)[0, 1])
        rows.append(
            {
                "component": component,
                "metric": metric,
                "slope_per_decade": float(slope),
                "intercept": float(intercept),
                "pearson_r": correlation,
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _coordinate_lookup(frame: FrozenBranchFrame2D) -> dict[tuple[str, int, int], Array]:
    return {
        (str(row.component), int(row.ix), int(row.iy)): frame.coordinates[index]
        for index, row in enumerate(frame.landmark_table.itertuples(index=False))
    }


def plot_branch_web(
    ax: plt.Axes,
    bank: ComponentBank,
    frame: FrozenBranchFrame2D,
    *,
    colors: Mapping[str, str] = DEFAULT_COMPONENT_COLORS,
    line_alpha: float = 0.50,
    point_size: float = 21.0,
    label_components: bool = True,
    show_axes: bool = True,
) -> dict[str, Any]:
    """Draw the two data-discovered FE webs in their immutable 2-D frame."""

    coordinates = _coordinate_lookup(frame)
    artists: dict[str, Any] = {"edges": [], "points": []}
    physical_edges = [
        ((int(row.a_ix), int(row.a_iy)), (int(row.b_ix), int(row.b_iy)))
        for row in bank.edge_table.itertuples(index=False)
    ]
    for component in COMPONENTS:
        color = colors[component]
        for a, b in physical_edges:
            line = np.stack(
                [coordinates[(component, *a)], coordinates[(component, *b)]]
            )
            artists["edges"].extend(
                ax.plot(line[:, 0], line[:, 1], color=color, lw=0.75, alpha=line_alpha, zorder=1)
            )
        selector = frame.landmark_table["component"].eq(component).to_numpy()
        point_artist = ax.scatter(
            frame.coordinates[selector, 0],
            frame.coordinates[selector, 1],
            s=point_size,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=component,
            zorder=3,
        )
        artists["points"].append(point_artist)
        if label_components:
            center = np.median(frame.coordinates[selector], axis=0)
            ax.annotate(
                component,
                xy=center,
                xytext=(5, 5),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="semibold",
            )
    if show_axes:
        ax.set_xlabel("Fixed projective coordinate 1")
        ax.set_ylabel("Fixed projective coordinate 2")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(True, color="#D9D9D9", lw=0.45, alpha=0.6)
    return artists


def plot_matched_component_web(
    ax: plt.Axes,
    bank: ComponentBank,
    frames: MatchedComponentFrames2D,
    component: str,
    *,
    colors: Mapping[str, str] = DEFAULT_COMPONENT_COLORS,
    point_size: float = 22.0,
    show_title: bool = True,
) -> dict[str, Any]:
    """Draw one readable within-sheet web with target-matched axes."""

    if component not in COMPONENTS:
        raise KeyError(component)
    frame = frames.frames[component]
    coordinate_lookup = {
        (int(row.ix), int(row.iy)): frame.coordinates[index]
        for index, row in enumerate(frame.landmark_table.itertuples(index=False))
    }
    color = colors[component]
    artists: dict[str, Any] = {"edges": [], "points": None}
    for row in bank.edge_table.itertuples(index=False):
        a = (int(row.a_ix), int(row.a_iy))
        b = (int(row.b_ix), int(row.b_iy))
        line = np.stack([coordinate_lookup[a], coordinate_lookup[b]])
        artists["edges"].extend(
            ax.plot(line[:, 0], line[:, 1], color=color, lw=0.80, alpha=0.56, zorder=1)
        )
    artists["points"] = ax.scatter(
        frame.coordinates[:, 0],
        frame.coordinates[:, 1],
        s=point_size,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    if show_title:
        ax.set_title(component, color=color, fontsize=10, fontweight="semibold")
    ax.set_xlim(*frames.common_limits[0])
    ax.set_ylim(*frames.common_limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target-aligned solution coordinate 1")
    ax.set_ylabel("Target-aligned solution coordinate 2")
    ax.grid(True, color="#D9D9D9", lw=0.45, alpha=0.6)
    return artists


def plot_matched_component_web_pair(
    axes: Sequence[plt.Axes],
    bank: ComponentBank,
    frames: MatchedComponentFrames2D,
    *,
    colors: Mapping[str, str] = DEFAULT_COMPONENT_COLORS,
) -> dict[str, dict[str, Any]]:
    """Draw B1/B2 in matched axes; intended for the integrated Figure 1."""

    if len(axes) != 2:
        raise ValueError("Exactly two axes are required for the B1/B2 web pair")
    output = {
        component: plot_matched_component_web(
            axis, bank, frames, component, colors=colors
        )
        for axis, component in zip(axes, COMPONENTS)
    }
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    return output


def plot_joint_branch_web_3d(
    ax: Any,
    bank: ComponentBank,
    frame: FrozenBranchFrame3D,
    *,
    colors: Mapping[str, str] = DEFAULT_COMPONENT_COLORS,
) -> dict[str, Any]:
    """Appendix-only joint 3-D view with physical grid connectivity."""

    coordinate_lookup = {
        (str(row.component), int(row.ix), int(row.iy)): frame.coordinates[index]
        for index, row in enumerate(frame.landmark_table.itertuples(index=False))
    }
    artists: dict[str, Any] = {"edges": [], "points": []}
    physical_edges = [
        ((int(row.a_ix), int(row.a_iy)), (int(row.b_ix), int(row.b_iy)))
        for row in bank.edge_table.itertuples(index=False)
    ]
    for component in COMPONENTS:
        color = colors[component]
        for a, b in physical_edges:
            line = np.stack(
                [coordinate_lookup[(component, *a)], coordinate_lookup[(component, *b)]]
            )
            artists["edges"].extend(
                ax.plot(
                    line[:, 0], line[:, 1], line[:, 2],
                    color=color, lw=0.72, alpha=0.45, zorder=1,
                )
            )
        selector = frame.landmark_table["component"].eq(component).to_numpy()
        artists["points"].append(
            ax.scatter(
                frame.coordinates[selector, 0],
                frame.coordinates[selector, 1],
                frame.coordinates[selector, 2],
                s=16.0,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=component,
                depthshade=False,
                zorder=3,
            )
        )
    ax.set_xlabel("Joint coordinate 1")
    ax.set_ylabel("Joint coordinate 2")
    ax.set_zlabel("Joint coordinate 3")
    return artists


def plot_conventional_evolution(
    ax: plt.Axes,
    bank: ComponentBank,
    frame: FrozenBranchFrame2D,
    evolution: pd.DataFrame,
    *,
    colors: Mapping[str, str] = DEFAULT_COMPONENT_COLORS,
    target_ids: Sequence[str] | None = None,
    web_alpha: float = 0.16,
) -> dict[str, Any]:
    """Overlay fixed-frame Conventional trajectories on the FE landmark web."""

    # Draw a quiet reference web first; the trajectory is the visual subject.
    plot_branch_web(
        ax,
        bank,
        frame,
        colors=colors,
        line_alpha=web_alpha,
        point_size=9.0,
        label_components=False,
    )
    data = evolution
    if target_ids is not None:
        data = data[data["target_id"].isin(target_ids)]
    artists: dict[str, Any] = {"trajectories": [], "checkpoints": []}
    iterations = sorted(int(value) for value in data["iteration"].unique())
    marker_by_iteration = {
        iteration: marker
        for iteration, marker in zip(iterations, ("o", "s", "D", "^", "v"))
    }
    for (component, target_id), group in data.groupby(["component", "target_id"], sort=True):
        group = group.sort_values("iteration")
        xy = group[["fixed_mds1", "fixed_mds2"]].to_numpy(dtype=float)
        color = colors[str(component)]
        artists["trajectories"].extend(
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=0.65, alpha=0.42, zorder=4)
        )
        for start, stop in zip(xy[:-1], xy[1:]):
            ax.annotate(
                "",
                xy=stop,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 0.55,
                    "alpha": 0.38,
                    "mutation_scale": 6,
                },
                zorder=4,
            )
    for iteration in iterations:
        subset = data[data["iteration"].eq(iteration)]
        point_artist = ax.scatter(
            subset["fixed_mds1"],
            subset["fixed_mds2"],
            s=20.0,
            marker=marker_by_iteration[iteration],
            facecolor="white",
            edgecolor="#202124",
            linewidth=0.55,
            label=f"C: {iteration:,} iterations",
            zorder=5,
        )
        artists["checkpoints"].append(point_artist)
    return artists


def plot_full_dimensional_evolution(
    ax: plt.Axes,
    evolution: pd.DataFrame,
    *,
    metric: str = "distance_to_target_fe",
    colors: Mapping[str, str] = DEFAULT_COMPONENT_COLORS,
) -> pd.DataFrame:
    """Compact Appendix-D median/IQR plot; returns the plotted summary."""

    summary = summarize_full_dimensional_evolution(evolution, metric=metric)
    for component in COMPONENTS:
        group = summary[summary["component"].eq(component)].sort_values("iteration")
        if group.empty:
            continue
        x = group["iteration"].to_numpy(dtype=float)
        median = group["median"].to_numpy(dtype=float)
        lower = group["q25"].to_numpy(dtype=float)
        upper = group["q75"].to_numpy(dtype=float)
        ax.plot(x, median, "o-", color=colors[component], lw=1.25, ms=4.0, label=component)
        ax.fill_between(x, lower, upper, color=colors[component], alpha=0.15, linewidth=0.0)
    ax.set_xscale("log")
    ax.set_xlabel("Conventional iteration")
    ax.set_ylabel("Projector distance to matched FE endpoint")
    ax.grid(True, color="#D9D9D9", lw=0.45, alpha=0.6)
    return summary


__all__ = [
    "COMPONENTS",
    "CONVENTIONAL_METHOD",
    "FE_METHOD",
    "ComponentAssignment",
    "ComponentBank",
    "CorrectedBranchArtifact",
    "FrozenBranchFrame2D",
    "FrozenBranchFrame3D",
    "MatchedComponentFrames2D",
    "component_log_iteration_slopes",
    "discover_fe_components",
    "fit_fixed_2d_frame",
    "fit_fixed_3d_frame",
    "fit_matched_component_frames_2d",
    "load_corrected_branch_artifact",
    "pairwise_projective_distance",
    "plot_branch_web",
    "plot_conventional_evolution",
    "plot_full_dimensional_evolution",
    "plot_joint_branch_web_3d",
    "plot_matched_component_web",
    "plot_matched_component_web_pair",
    "project_conventional_evolution",
    "project_out_of_sample",
    "projective_distance",
    "projective_similarity",
    "square_array_positions",
    "summarize_full_dimensional_evolution",
    "target_demodulated_states",
]
