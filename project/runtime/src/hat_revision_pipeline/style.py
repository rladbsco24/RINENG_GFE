from __future__ import annotations


from rineng_content_id import content_identity
import io
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import Normalize, PowerNorm


COLORS = {
    "FE+g": "#007C78",
    "Conventional": "#3B6FA1",
    "FE": "#C76A12",
    "regularized FE": "#C44956",
    "IB": "#4C7C2F",
    "GS-PO audit": "#80568D",
    "GS": "#80568D",
    "AD": "#C44956",
    "IB→FE": "#D37295",
    "FE warm": "#F6B26B",
    "B1": "#6F4D92",
    "B2": "#008781",
}

TARGET_COLOR = "#B33467"
EQUILIBRIUM_COLOR = "white"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.55,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.formatter.use_mathtext": True,
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str, x: float = 0.012, y: float = 0.988) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", boxstyle="round,pad=0.15"),
        zorder=30,
    )


def set_panel_title(ax: mpl.axes.Axes, title: str) -> None:
    ax.set_title(title, pad=7.0, fontweight="semibold")


def shared_power_norm(
    arrays: Iterable[np.ndarray],
    lower: float = 5.0,
    upper: float = 99.0,
    gamma: float = 0.7,
) -> Normalize:
    finite_parts = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        finite_parts.append(values[np.isfinite(values)])
    values = np.concatenate([part for part in finite_parts if part.size])
    if not values.size:
        raise ValueError("No finite field values")
    vmin, vmax = np.percentile(values, [lower, upper])
    if vmax <= vmin:
        vmax = vmin + np.finfo(float).eps
    return PowerNorm(gamma=gamma, vmin=float(vmin), vmax=float(vmax), clip=True)


def field_panel(
    ax: mpl.axes.Axes,
    amplitude: np.ndarray,
    extent_mm: Sequence[float],
    *,
    norm: Normalize,
    xlabel: str,
    ylabel: str,
    target_mm: tuple[float, float] | None = None,
    equilibrium_mm: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> mpl.image.AxesImage:
    image = ax.imshow(
        np.asarray(amplitude),
        origin="lower",
        extent=extent_mm,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if target_mm is not None:
        target_marker(ax, *target_mm)
    if equilibrium_mm is not None:
        ax.scatter(
            *equilibrium_mm,
            marker="o",
            facecolor=EQUILIBRIUM_COLOR,
            edgecolor="black",
            linewidth=1.0,
            s=34,
            zorder=21,
        )
    return image


def target_marker(ax: mpl.axes.Axes, x, y, **kwargs):
    """Keep prescribed-coordinate crosses visible on light and dark fields."""
    options = dict(marker="x", color=TARGET_COLOR, s=58, linewidth=1.8, zorder=20)
    options.update(kwargs)
    marker = ax.scatter(x, y, **options)
    marker.set_path_effects([pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
    return marker


def phase_grid_panel(ax: mpl.axes.Axes, phase: np.ndarray, side: int = 16) -> mpl.image.AxesImage:
    values = np.mod(np.asarray(phase).reshape(side, side), 2.0 * np.pi)
    image = ax.imshow(values, origin="lower", cmap="hsv", vmin=0.0, vmax=2.0 * np.pi, interpolation="nearest")
    ax.set_xlabel("Element column")
    ax.set_ylabel("Element row")
    ax.grid(False)
    return image


def force_streamlines(
    ax: mpl.axes.Axes,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    *,
    color: str = "white",
    density: float = 1.0,
) -> None:
    magnitude = np.hypot(fx, fy)
    scale = np.nanpercentile(magnitude, 95.0)
    if not np.isfinite(scale) or scale <= 0:
        return
    linewidth = 0.35 + 1.15 * np.clip(magnitude / scale, 0.0, 1.0)
    prior_patches = set(ax.patches)
    stream = ax.streamplot(
        x_mm,
        y_mm,
        fx,
        fy,
        color=color,
        linewidth=linewidth,
        density=density,
        arrowsize=0.75,
        minlength=0.07,
        zorder=12,
    )
    # The white vectors retain their data and density. A narrow dark outline
    # preserves their visibility where the pressure map is bright yellow.
    stream.lines.set_path_effects([
        pe.Stroke(linewidth=1.35, foreground="#26313B", alpha=0.72), pe.Normal()
    ])
    for arrow in set(ax.patches) - prior_patches:
        arrow.set_path_effects([
            pe.Stroke(linewidth=0.65, foreground="#26313B", alpha=0.80), pe.Normal()
        ])


def connect_axes(
    fig: mpl.figure.Figure,
    source_ax: mpl.axes.Axes,
    target_ax: mpl.axes.Axes,
    *,
    color: str = "0.35",
    lw: float = 1.1,
) -> None:
    source = source_ax.get_position()
    target = target_ax.get_position()
    arrow = mpl.patches.FancyArrowPatch(
        (source.x1 + 0.004, 0.5 * (source.y0 + source.y1)),
        (target.x0 - 0.004, 0.5 * (target.y0 + target.y1)),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
        zorder=50,
    )
    fig.add_artist(arrow)


class ArtifactWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.figure_dir = self.root / "figures"
        self.table_dir = self.root / "tables"
        self.data_dir = self.root / "data"
        for directory in (self.figure_dir, self.table_dir, self.data_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, object]] = []

    @staticmethod
    def _content_id(path: Path) -> str:
        digest = content_identity()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def save_figure(self, fig: mpl.figure.Figure, stem: str, dpi: int = 300) -> list[Path]:
        paths = [self.figure_dir / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
        for path in paths:
            temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
            buffer = io.BytesIO()
            fig.savefig(buffer, format=path.suffix[1:], dpi=dpi)
            payload = buffer.getvalue()
            complete = (
                payload.startswith(b"\x89PNG\r\n\x1a\n")
                and payload.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")
                if path.suffix == ".png"
                else (payload.startswith(b"%PDF") and b"%%EOF" in payload[-64:]) if path.suffix == ".pdf"
                else b"</svg>" in payload[-128:]
            )
            if not complete:
                raise OSError(f"Incomplete figure serialization: {path.name}")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            self.records.append(
                {"kind": "figure", "path": str(path.relative_to(self.root)), "bytes": path.stat().st_size, "content_id": self._content_id(path)}
            )
        return paths

    def record_file(self, path: str | Path, kind: str) -> None:
        path = Path(path)
        self.records.append(
            {"kind": kind, "path": str(path.relative_to(self.root)), "bytes": path.stat().st_size, "content_id": self._content_id(path)}
        )

    def write_manifest(self, extra: dict[str, object] | None = None) -> Path:
        path = self.root / "artifact_manifest.json"
        # Tables are checkpointed repeatedly during an all-figure run. Keep
        # one current file record, not obsolete hashes from earlier exports.
        latest = {str(record["path"]): dict(record) for record in self.records}
        for relative, record in latest.items():
            current = self.root / relative
            record["bytes"] = current.stat().st_size
            record["content_id"] = self._content_id(current)
        payload = {"schema_version": 1, "artifacts": [latest[key] for key in sorted(latest)], **(extra or {})}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


configure_style()
