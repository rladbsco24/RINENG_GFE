"""Reproduce Appendix B pressure figures from the deposited terminal commands."""
from __future__ import annotations

from rineng_content_id import content_identity
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as effects
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime" / "src"))
from hat_revision_pipeline.gorkov_core import ArrayGeometry, pressure_field

STYLE = {"font.family": "DejaVu Sans", "font.size": 12,
         "axes.labelsize": 13, "axes.labelweight": "semibold",
         "axes.linewidth": 1.6, "xtick.labelsize": 11, "ytick.labelsize": 11,
         "xtick.major.width": 1.5, "ytick.major.width": 1.5,
         "legend.fontsize": 11, "pdf.fonttype": 42, "svg.fonttype": "none"}
METHOD_STYLE = {"FE": {"color": "white", "ls": "-"},
                "RH": {"color": "#D5F375", "ls": "-."},
                "Conventional": {"color": "#61E4F5", "ls": "--"}}
LEVELS = (.25, .50, .75)
REQUIRED = {("Bottle", "corrected_minus", m) for m in ("FE", "Conventional")}
REQUIRED |= {(shape, "historical_plus", m)
             for shape in ("Twin", "Bottle") for m in ("FE", "RH", "Conventional")}


def load_solutions(root=ROOT):
    """Load the eight specified center-target runs and verify phase provenance."""
    manifest = json.loads((root / "data/solution_manifest.json").read_text())
    records, arrays = {}, {}
    for item in manifest:
        key = item["cache_key"]
        path = root / "data/cache" / key
        row = json.loads(path.with_suffix(".json").read_text())
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        phase = data["terminal_phase_rad"]
        phase_content_id = content_identity(phase.tobytes()).hexdigest()
        if phase_content_id != row["phase_content_id"] or phase_content_id != item["phase_content_id"]:
            raise ValueError(f"Phase content_id mismatch for {key}")
        if phase.shape != (256,) or not np.isfinite(phase).all():
            raise ValueError(f"Invalid terminal phase for {key}")
        if not np.allclose(row["target_m"], [0., 0., .05], rtol=0, atol=1e-12):
            raise ValueError("These figures require the declared center target")
        identity = (item["requested"], row["sign"], row["method"])
        if identity in records:
            raise ValueError(f"Duplicate solution: {identity}")
        records[identity] = dict(row, requested=item["requested"])
        arrays[identity] = data
    if set(records) != REQUIRED:
        raise ValueError(f"Missing or unexpected pressure cases: {set(records) ^ REQUIRED}")
    if len({r["initial_content_id"] for r in records.values()}) != 1:
        raise ValueError("Pressure comparisons must retain the same initial command")
    return records, arrays


def field_name(shape, sign, method, plane):
    return "__".join((shape, sign, method, plane))


def pressure_slices(records, arrays, root=ROOT):
    """Use a slice cache keyed by the exact phase and acoustic model content_ids."""
    path = root / "data/pressure_slices.npz"
    meta_path = root / "data/pressure_slice_provenance.json"
    cached, previous = {}, {}
    if path.is_file() and meta_path.is_file():
        previous = json.loads(meta_path.read_text())
        with np.load(path, allow_pickle=False) as archive:
            cached = {name: archive[name].copy() for name in archive.files}
    axis = np.linspace(-1.25 * 343. / 40000., 1.25 * 343. / 40000., 181)
    if "axis_m" not in cached or not np.array_equal(cached["axis_m"], axis):
        cached, previous = {}, {}
    fields, provenance = {"axis_m": axis}, {}
    aa, bb = np.meshgrid(axis, axis, indexing="xy")
    geometry = ArrayGeometry.square(16, .010)
    model_content_id = content_identity((ROOT / "runtime/src/hat_revision_pipeline/gorkov_core.py").read_bytes()).hexdigest()
    for identity in sorted(records):
        shape, sign, method = identity
        row = records[identity]
        for plane in (("XY", "XZ") if sign == "corrected_minus" else ("XZ",)):
            name = field_name(shape, sign, method, plane)
            metadata = {"requested": shape, "sign": sign, "method": method,
                        "plane": plane, "phase_content_id": row["phase_content_id"],
                        "model_content_id": model_content_id, "cache_key": row["cache_key"],
                        "target_m": [0., 0., .05], "frequency_hz": 40000.,
                        "grid_points": 181, "half_width_m": float(axis[-1])}
            if name in cached and previous.get(name) == metadata:
                fields[name] = cached[name]
            else:
                points = (np.stack([aa, bb, np.full_like(aa, .05)], axis=-1)
                          if plane == "XY" else
                          np.stack([aa, np.zeros_like(aa), bb + .05], axis=-1))
                flat = points.reshape(-1, 3)
                phase = arrays[identity]["terminal_phase_rad"]
                fields[name] = np.concatenate([
                    pressure_field(phase, flat[j:j+1024], geometry, 40000.)
                    for j in range(0, len(flat), 1024)]).reshape(aa.shape)
            provenance[name] = metadata
    np.savez_compressed(path, **fields)
    meta_path.write_text(json.dumps(provenance, indent=2))
    return fields, provenance


def field_axes(ax, plane):
    ax.set_xlabel(r"$x-x_0$ (mm)", labelpad=4)
    ax.set_ylabel((r"$y-y_0$" if plane == "XY" else r"$z-z_0$") + " (mm)", labelpad=3)
    ax.set_xticks([-10, 0, 10]); ax.set_yticks([-10, 0, 10])
    ax.tick_params(length=5, width=1.5)
    ax.set_aspect("equal")


def save_figure(fig, stem, root):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(root / "figures" / f"{stem}.{extension}", dpi=260, facecolor="white")
    plt.close(fig)


def render_b1(fields, root=ROOT):
    axis = fields["axis_m"] * 1000
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.1))
    fig.subplots_adjust(left=.17, right=.96, bottom=.145, top=.88, wspace=.29, hspace=.29)
    normalizations = {}
    for i, method in enumerate(("FE", "Conventional")):
        data = {plane: np.abs(fields[field_name("Bottle", "corrected_minus", method, plane)])
                for plane in ("XY", "XZ")}
        peak = max(float(a.max()) for a in data.values())
        normalizations[method] = peak
        for j, plane in enumerate(("XY", "XZ")):
            ax = axes[i, j]
            im = ax.imshow(data[plane] / peak, origin="lower", extent=extent,
                           cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax.plot(0, 0, "+", color="#62F0EE", ms=10, mew=1.7)
            field_axes(ax, plane)
            ax.text(-.13, 1.04, f"({chr(97 + 2*i + j)})", transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom")
            if i == 0:
                ax.set_title(plane, fontsize=14, fontweight="semibold", pad=12)
        axes[i, 0].text(-.38, .5, method, rotation=90, transform=axes[i, 0].transAxes,
                        ha="center", va="center", fontsize=14, fontweight="semibold")
    cax = fig.add_axes([.29, .065, .55, .018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cb.set_label(r"$|p|/p_{\max}$", labelpad=5)
    save_figure(fig, "Figure_B1_corrected_sign_fields", root)
    return normalizations


def legend_handle(method):
    style = METHOD_STYLE[method]
    line = Line2D([], [], color=style["color"], ls=style["ls"], lw=2,
                  label=method)
    line.set_path_effects([effects.Stroke(linewidth=2.7, foreground=".30"), effects.Normal()])
    return line


def render_b2(fields, root=ROOT):
    """Three pressure panels cover both comparisons in both formulations."""
    panels = [("Twin", ("FE", "RH", "Conventional"), "FE"),
              ("Bottle", ("FE", "Conventional"), "FE"),
              ("Bottle", ("RH", "Conventional"), "RH")]
    axis = fields["axis_m"] * 1000
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    amplitudes = {(shape, method): np.abs(fields[field_name(shape, "historical_plus", method, "XZ")])
                  for shape in ("Twin", "Bottle") for method in ("FE", "RH", "Conventional")}
    peaks = {shape: max(float(amplitudes[shape, m].max()) for m in ("FE", "RH", "Conventional"))
             for shape in ("Twin", "Bottle")}
    fig = plt.figure(figsize=(9.4, 10.5))
    grid = fig.add_gridspec(2, 4, left=.11, right=.96, bottom=.125, top=.84,
                           wspace=.90, hspace=.55)
    slots = (grid[0, :2], grid[0, 2:], grid[1, 1:3])
    for i, ((shape, methods, background), slot) in enumerate(zip(panels, slots)):
        ax = fig.add_subplot(slot)
        peak = peaks[shape]
        im = ax.imshow(amplitudes[shape, background] / peak, origin="lower", extent=extent,
                       cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        for method in methods:
            style = METHOD_STYLE[method]
            contour = ax.contour(axis, axis, amplitudes[shape, method] / peak,
                                 levels=LEVELS, colors=style["color"],
                                 linestyles=style["ls"], linewidths=1.7)
            # The line halo keeps the same method styles legible on both light and dark pixels.
            contour.set_path_effects([effects.Stroke(linewidth=2.2, foreground=(0., 0., 0., .28)), effects.Normal()])
        ax.plot(0, 0, "+", color="#61E4F5", ms=9, mew=1.6)
        field_axes(ax, "XZ")
        ax.text(-.14, 1.19, f"({chr(97+i)})  {shape}", transform=ax.transAxes,
                fontsize=14, fontweight="semibold", va="bottom")
        ax.legend(handles=[legend_handle(m) for m in methods], ncol=len(methods),
                  loc="lower center", bbox_to_anchor=(.5, 1.025), frameon=False,
                  fontsize=10.8, handlelength=1.7, columnspacing=.85, handletextpad=.45)
    cax = fig.add_axes([.30, .055, .48, .015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cb.set_label(r"$|p|/p_{\max}$", labelpad=5)
    save_figure(fig, "Figure_B2_alternative_sign_pressure", root)
    rows = []
    for shape in ("Twin", "Bottle"):
        c = amplitudes[shape, "Conventional"].ravel()
        for method in ("FE", "RH"):
            a = amplitudes[shape, method].ravel()
            rows.append({"requested": shape, "reference_method": method,
                         "xz_amplitude_cosine_similarity": float(a @ c / (np.linalg.norm(a)*np.linalg.norm(c))),
                         "xz_amplitude_pearson_correlation": float(np.corrcoef(a, c)[0, 1]),
                         "shared_peak_pa": peaks[shape],
                         "reference_target_relative_pressure": float(a.reshape(181, 181)[90, 90] / peaks[shape]),
                         "conventional_target_relative_pressure": float(c.reshape(181, 181)[90, 90] / peaks[shape])})
    table = pd.DataFrame(rows)
    table.to_csv(root / "B2_field_summary.csv", index=False)
    return peaks, table


def render(root=ROOT):
    root = Path(root).resolve()
    (root / "figures").mkdir(exist_ok=True)
    with threadpool_limits(limits=1), plt.rc_context(STYLE):
        records, arrays = load_solutions(root)
        fields, provenance = pressure_slices(records, arrays, root)
        b1_peaks = render_b1(fields, root)
        b2_peaks, summary = render_b2(fields, root)
    pd.DataFrame(records.values()).to_csv(root / "data/solver_results.csv", index=False)
    manifest = {"status": "SMOKE", "content": "Pressure fields only", "B1_sign": "corrected_minus",
                "B2_sign": "historical_plus", "B1_pressure_panels": 4, "B2_pressure_panels": 3,
                "B2_comparisons": {"a": "Twin: FE, RH, Conventional", "b": "Bottle: FE, Conventional", "c": "Bottle: RH, Conventional"},
                "B1_shared_XY_XZ_peak_pa_per_method": b1_peaks,
                "B2_shared_peak_pa_per_formulation": b2_peaks,
                "B2_contour_levels": list(LEVELS),
                "B2_backgrounds": {"a": "FE", "b": "FE", "c": "RH"},
                "B2_method_styles": METHOD_STYLE,
                "temporary_descriptive_titles": True,
                "optimization_rerun_during_render": False,
                "pressure_phase_provenance": provenance}
    (root / "figure_provenance.json").write_text(json.dumps(manifest, indent=2))
    return summary


if __name__ == "__main__":
    print(render().round(6).to_string(index=False))
