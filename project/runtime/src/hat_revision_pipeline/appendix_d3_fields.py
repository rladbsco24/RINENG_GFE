"""Spatial deployment illustration from existing GFE quantization commands.

Q=32 is a common coarse illustration with visible perturbations; the complete
nine-resolution numerical study remains in the adjacent exported tables.
Nothing in this producer changes a command or solves a mechanical equilibrium.
"""
from __future__ import annotations


from rineng_content_id import content_identity
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import pipeline as p
from .cache import digest_array


ILLUSTRATION_LEVELS = 32
FIELD_POINTS = 181
FIELD_HALF_WIDTH_M = .004
COLORS = {"Nearest": "#C77835", "Elastic mechanics offset": "#007C78"}


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_bytes(value)
    temporary.replace(path)


def _json(path, value):
    _write(path, json.dumps(value, indent=2, allow_nan=False).encode())


def _pressure_data(output, ctx, commands, *, allow_unresolved=False):
    """Cache six pressure sections, preserving command and field identities."""
    levels = ILLUSTRATION_LEVELS
    destination = output / "plot_data"
    destination.mkdir(exist_ok=True)
    path = destination / f"D3_fields_Q{levels}.npz"
    provenance_path = destination / f"D3_fields_Q{levels}_provenance.json"
    sources, root_rows = {}, []
    field_sources = []
    for command in commands:
        task = command["task"]
        sources[f"{task}_continuous"] = np.asarray(command["phase"])
        for method, label in (("Nearest", "nearest"), ("Elastic mechanics offset", "offset")):
            for index, reference in enumerate(command["results"]):
                source = output / "gfe_validated_commands" / f"{task}_Q{levels}_{label}_T{index + 1}.npz"
                with np.load(source, allow_pickle=False) as saved:
                    phase, root = saved["phase_rad"].copy(), saved["root_m"].copy()
                    root_found = bool(saved["root_found"])
                    if not allow_unresolved:
                        assert root_found
                    sources[f"{task}_{label}"] = phase
                ref = np.asarray(reference.equilibrium.equilibrium_m)
                target = np.asarray(command["targets"][index])
                delta = root - ref
                reference_found = bool(reference.equilibrium.numerical_root_found)
                valid_shift = root_found and reference_found
                candidate = root.copy()
                if not root_found:
                    root = np.full(3, np.nan)
                if not valid_shift:
                    delta = np.full(3, np.nan)
                root_rows.append(dict(task=task, levels=levels, target_index=index + 1,
                                      root_found=root_found, reference_root_found=reference_found,
                                      method=method, phase_content_id=digest_array(phase),
                                      continuous_phase_content_id=digest_array(command["phase"]),
                                      source_command=str(source.relative_to(output)),
                                      **{f"{prefix}_{axis}_{unit}": float(scale * value)
                                         for prefix, vector, scale, unit in
                                         (("target", target, 1., "m"), ("root", root, 1., "m"),
                                          ("continuous_root", ref if reference_found else np.full(3,np.nan), 1., "m"),
                                          ("search_candidate", candidate, 1., "m"), ("added_shift", delta, 1e6, "um"))
                                         for axis, value in zip("xyz", vector)},
                                      added_shift_3d_um=float(1e6 * np.linalg.norm(delta)),
                                      absolute_target_error_3d_um=float(1e6 * np.linalg.norm(root - target))))
        field_sources.append(dict(task=task, target_index=1,
                                  target_m=np.asarray(command["targets"][0]).tolist(),
                                  field=p._array_field(ctx, command["phase"]).provenance()))
    provenance = dict(schema="GFE-D3-spatial-pressure-v1", levels=levels,
                      selection="Illustrative coarse resolution, held identical for Single and Triple; full resolution table retained.",
                      field_points_per_axis=FIELD_POINTS, field_half_width_m=FIELD_HALF_WIDTH_M,
                      plane="XZ at the commanded y coordinate of T1", field_sources=field_sources,
                      command_content_id={key: digest_array(value) for key, value in sources.items()},
                      force_equilibrium="Cached independent elastic-sphere plus gravity roots; no new root solves.",
                      implementation_content_id=content_identity(Path(__file__).read_bytes()).hexdigest())
    # The producer hash includes presentation edits but is excluded from the
    # numerical cache identity: only field, commands and coordinates matter.
    numerical_identity = {key: value for key, value in provenance.items() if key != "implementation_content_id"}
    cache_hit = False
    if path.exists() and provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        if old.get("numerical_identity") == numerical_identity:
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            cache_hit = True
    if not cache_hit:
        axis = np.linspace(-FIELD_HALF_WIDTH_M, FIELD_HALF_WIDTH_M, FIELD_POINTS)
        xx, zz = np.meshgrid(axis, axis)
        arrays = dict(x_offset_m=xx, z_offset_m=zz)
        for command in commands:
            task = command["task"]
            target = command["targets"][0]
            points = target + np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
            for label in ("continuous", "nearest", "offset"):
                field = p._array_field(ctx, sources[f"{task}_{label}"])
                amplitude = np.concatenate([np.abs(field.pressure(points[start:start + 1024]))
                                            for start in range(0, len(points), 1024)])
                arrays[f"{task}_{label}_pressure_pa"] = amplitude.reshape(xx.shape)
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        _write(path, stream.getvalue())
    frame = pd.DataFrame(root_rows)
    _write(destination / "D3_spatial_equilibria.csv", frame.to_csv(index=False).encode())
    _json(provenance_path, dict(**provenance, numerical_identity=numerical_identity,
                               pressure_cache_reused=cache_hit))
    return arrays, frame, provenance


def plot(output, ctx, commands, *, evidence_status="SMOKE", selection_scope=None, allow_unresolved=False):
    output = Path(output)
    arrays, roots, provenance = _pressure_data(output, ctx, commands, allow_unresolved=allow_unresolved)
    for name in list(roots.columns):
        if name.endswith('_um'):
            roots[name[:-3] + '_m'] = roots[name] * 1e-6
    _write(output / "plot_data/D3_spatial_equilibria.csv", roots.to_csv(index=False).encode())
    style = {"font.size": 13, "axes.labelsize": 14, "axes.labelweight": "bold",
             "axes.linewidth": 1.6, "xtick.labelsize": 12, "ytick.labelsize": 12,
             "xtick.major.width": 1.5, "ytick.major.width": 1.5,
             "pdf.fonttype": 42, "svg.fonttype": "none"}
    contour_records = []
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.3), layout="constrained")
        for row, command in enumerate(commands):
            task = command["task"]
            field_ax, shift_ax = axes[row]
            xx, zz = 1e3 * arrays["x_offset_m"], 1e3 * arrays["z_offset_m"]
            peak = max(float(np.max(arrays[f"{task}_{label}_pressure_pa"]))
                       for label in ("continuous", "nearest", "offset"))
            # Identical physical isobars for the two quantized commands. Their
            # normalized fractions are converted to Pa in the exported table.
            contour_levels = peak * np.asarray([.2, .4, .6, .8])
            field_ax.contourf(xx, zz, arrays[f"{task}_continuous_pressure_pa"],
                              levels=np.linspace(0, peak, 33), cmap="Greys", alpha=.12)
            for method, label, linestyle in (("Nearest", "nearest", "--"),
                                              ("Elastic mechanics offset", "offset", "-")):
                field_ax.contour(xx, zz, arrays[f"{task}_{label}_pressure_pa"],
                                 levels=contour_levels, colors=COLORS[method],
                                 linestyles=linestyle, linewidths=1.65, alpha=.92,
                                 zorder=3 if method == "Nearest" else 2)
            for fraction, value in zip((.2, .4, .6, .8), contour_levels):
                contour_records.append(dict(task=task, target_index=1,
                                            fraction_of_common_section_maximum=fraction,
                                            pressure_contour_pa=float(value),
                                            common_section_maximum_pa=peak))
            reference = np.asarray(command["results"][0].equilibrium.equilibrium_m)
            target = np.asarray(command["targets"][0])
            if command["results"][0].equilibrium.numerical_root_found:
                field_ax.plot(1e3 * (reference[0] - target[0]), 1e3 * (reference[2] - target[2]),
                              "+", color="#252525", ms=11, mew=2.2)
            field_ax.set(xlim=(-4, 4), ylim=(-4, 4), xticks=(-4, 0, 4), yticks=(-4, 0, 4),
                         xlabel=r"$x-x_t$ (mm)", ylabel=r"$z-z_t$ (mm)")
            field_ax.set_aspect("equal", adjustable="box")
            field_ax.set_title(f"({'ac'[row]}) {task if task == 'Single' else 'Triple, T1'}, XZ",
                               fontsize=15, weight="bold", pad=12)


            task_roots = roots[roots.task.eq(task)]
            for method, marker in (("Nearest", "s"), ("Elastic mechanics offset", "o")):
                part = task_roots[task_roots.method.eq(method) & task_roots.root_found & task_roots.reference_root_found]
                for record in part.to_dict("records"):
                    x, z = record["added_shift_x_m"], record["added_shift_z_m"]
                    shift_ax.plot([0, x], [0, z], color=COLORS[method], alpha=.65,
                                  lw=1.6, linestyle="--" if method == "Nearest" else "-")
                    shift_ax.plot(x, z, marker=marker, color=COLORS[method], ms=8,
                                  mfc="white", mew=2)
                    if task == "Triple":
                        # Fixed target labels identify dependent roots of the
                        # same phase command; they are not independent samples.
                        label_offsets = ({1: (7, 0), 2: (7, -11), 3: (8, -12)}
                                         if method == "Nearest" else
                                         {1: (8, 10), 2: (-23, -11), 3: (-25, 9)})
                        shift_ax.annotate(f"T{record['target_index']}", (x, z),
                                          xytext=label_offsets[record["target_index"]],
                                          textcoords="offset points", fontsize=11, weight="semibold",
                                          color=COLORS[method])
            shift_ax.plot(0, 0, "+", color="#252525", ms=12, mew=2.2, zorder=8)
            if task == "Single":
                shift_ax.set(xlim=(-90e-6, 90e-6), ylim=(-25e-6, 155e-6), xticks=(-80e-6, 0, 80e-6), yticks=(0, 50e-6, 100e-6, 150e-6))
            else:
                shift_ax.set(xlim=(-100e-6, 100e-6), ylim=(-140e-6, 60e-6), xticks=(-80e-6, 0, 80e-6), yticks=(-120e-6, -80e-6, -40e-6, 0, 40e-6))
            if allow_unresolved:
                finite = task_roots[["added_shift_x_m", "added_shift_z_m"]].dropna().to_numpy()
                if len(finite):
                    finite = np.vstack([finite, np.zeros(2)])
                    lower, upper = finite.min(0), finite.max(0)
                    span = max(float(np.max(upper-lower))*1.30, 180e-6)
                    center = .5*(lower+upper)
                    shift_ax.set(xlim=(center[0]-.5*span,center[0]+.5*span),
                                 ylim=(center[1]-.5*span,center[1]+.5*span))
                    from matplotlib.ticker import MaxNLocator
                    shift_ax.xaxis.set_major_locator(MaxNLocator(4))
                    shift_ax.yaxis.set_major_locator(MaxNLocator(5))
            shift_ax.set_aspect("equal", adjustable="box")
            shift_ax.set(xlabel=r"$\Delta x$ (m)",
                         ylabel=r"$\Delta z$ (m)")
            shift_ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
            shift_ax.set_title(f"({'bd'[row]}) {task}", fontsize=15, weight="bold", pad=12)
            shift_ax.axhline(0, color="#b9b9b9", lw=.8, alpha=.6, zorder=0)
            shift_ax.axvline(0, color="#b9b9b9", lw=.8, alpha=.6, zorder=0)
            for ax in (field_ax, shift_ax):
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontweight("semibold")
        handles = [Line2D([], [], color=COLORS["Nearest"], ls="--", marker="s", mfc="white", mew=1.8, lw=1.8,
                          label="Nearest"),
                   Line2D([], [], color=COLORS["Elastic mechanics offset"], marker="o", mfc="white", mew=1.8, lw=1.8,
                          label="Mechanics offset"),
                   Line2D([], [], color="#252525", marker="+", ms=10, mew=2, ls="none", label="Continuous root")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=11.5, frameon=False)
        paths = []
        for extension in ("png", "pdf", "svg"):
            path = output / "figures" / f"Figure_D3_GFE_fields_and_equilibria.{extension}"
            path.parent.mkdir(exist_ok=True)
            stream = io.BytesIO()
            fig.savefig(stream, format=extension, dpi=240, bbox_inches="tight", facecolor="white")
            _write(path, stream.getvalue())
            if extension == "png":
                paths.append(str(path))
        plt.close(fig)
    _write(output / "plot_data/D3_pressure_contour_levels.csv", pd.DataFrame(contour_records).to_csv(index=False).encode())
    # The previous resolution-sweep plot is intentionally retired. Its complete
    # measurements, candidate tables and caches remain available for export.
    for extension in ("png", "pdf", "svg"):
        (output / "figures" / f"Figure_D3_GFE_quantization.{extension}").unlink(missing_ok=True)
    caption = (
        "Figure D3. Spatial effect of mechanics-aware phase rounding, illustrated at Q = 32 phase levels "
        "for the frozen main GFE Single and Triple commands. (a,c) XZ pressure-amplitude isobars for nearest "
        "rounding (orange dashed) and mechanics-aware common-phase-offset rounding (teal solid); both use "
        "identical physical contour levels within each task. The faint background is the continuous GFE "
        "pressure field. Triple T1 is the fixed first target; the section passes through its commanded y. "
        "(b,d) XZ projections of the independently validated elastic-plus-gravity equilibria, measured relative "
        "to each target's continuous-command equilibrium. All three Triple targets are shown relative to their "
        "own references, which coincide at the displayed origin; this does not imply that the physical traps "
        "coincide. Lines link each reference to its discrete equilibrium and do not show particle trajectories. "
        "The exported table reports full 3D distances, including the unplotted y component, in meters. The common phase offset "
        "changes which elements round to each finite phase level. Candidates are screened for positive symmetric "
        "stiffness at all reference roots and ranked by worst-target first-order displacement; the selected "
        "commands then receive the unchanged exact elastic root validation. A global phase shift alone does "
        "not change the continuous field, and no target-position or gravity-force offset is applied. This "
        "procedure preserves the continuous equilibrium; it need not reduce absolute error from the commanded "
        "target because nearest rounding can accidentally cancel existing continuous-model displacement. "
        "Q = 32 is an illustrative coarse resolution with visible perturbations, held identical for both tasks; "
        "the complete nine-resolution study, including Q = 256, 640 and 2048 literature anchors, remains in "
        "D3_GFE_quantization_target_results.csv and D3_GFE_quantization_summary.csv. No continuous optimization "
        f"or new root solve is performed by this figure producer. Two frozen commands; {evidence_status}.\n")
    if selection_scope:
        caption += selection_scope + "\n"
    if allow_unresolved:
        caption += ("All production validation outcomes remain in the target table. Root-shift coordinates require "
                    "both the continuous and discrete roots to resolve; unresolved search candidates are never plotted as equilibria.\n")
    _json(output / "D3_spatial_figure_provenance.json", dict(**provenance,
          figure="Figure_D3_GFE_fields_and_equilibria", field_sections=6,
          displayed_quantized_commands=4, displayed_quantized_equilibria=8,
          displayed_continuous_reference_equilibria=4,
          evidence_status=evidence_status,
          sample_scaling="Keep the spatial illustration and its two frozen source commands fixed; no additional quantization population is required."))
    return paths, caption, roots
