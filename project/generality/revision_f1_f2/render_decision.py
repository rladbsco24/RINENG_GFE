"""Draw the main decision-space heatmaps and analytic descent arrows by case."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from appendix_io import save_figure, write_bytes
from appendix_settings import SETTINGS, ARRAY_IDS, FREQUENCY_IDS
from representative_selection import load_selection, validate_alignment

ROOT = Path(__file__).resolve().parent
COLORS = {"Conventional": "#3B6FA1", "FE": "#C76A12", "GFE": "#007C78"}


def render(ids, group, data, output):
    data, output = Path(data), Path(output)
    selection = load_selection(data.parent)
    window_path = data.parent / 'decision_reveal/selection.json'
    windows = json.loads(window_path.read_text()) if window_path.exists() else {}
    alignment = validate_alignment(data.parent, ids)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .75,
        "savefig.facecolor": "white", "axes.grid": False})
    rows = len(ids) // 3
    height = 6.30*rows+.90
    fig = plt.figure(figsize=(11.5, height))
    outer = fig.add_gridspec(rows, 3, left=.072, right=.978, bottom=1.10/height,
                            top=1-.80/height, wspace=.29, hspace=.24)
    prefix = "2A" if group == "Arrays" else "2B"
    observations = []
    for index, sid in enumerate(ids):
        setting = next(s for s in SETTINGS if s["id"] == sid)
        label = setting["label"].split(" · ")[0] if group == "Arrays" else f'{setting["frequency_hz"]/1000:g} kHz'
        tile = outer[index//3, index%3]
        inner = tile.subgridspec(2, 1, hspace=.32)
        box = tile.get_position(fig)
        fig.text(box.x0, box.y1+.40/height,
                 f"({chr(97+index)}) {label} · {selection[sid]}", ha="left", va="center",
                 fontsize=11, fontweight="semibold")
        original_path = data / f"{sid}_decision.npz"
        final_path = data.parent / "final_decision" / original_path.name
        source_path = final_path if final_path.exists() else original_path
        if sid in windows.get('selected_cache_paths', {}):
            source_path = data.parent.parent / windows['selected_cache_paths'][sid]
        if source_path != original_path:
            # Wider sections must keep the exact commands and deposited native
            # planes used by the Single pressure and ACS correspondence views.
            with np.load(original_path, allow_pickle=False) as original, np.load(source_path, allow_pickle=False) as final:
                for key in ("methods", "endpoint_phase_rad", "directions", "target_m", "frequency_hz"):
                    if not np.array_equal(original[key], final[key]):
                        raise ValueError(f"{sid}: final decision cache changed {key}")
        with np.load(source_path, allow_pickle=False) as pack:
            methods = ("Conventional", selection[sid])
            indices = [list(pack["methods"]).index(method) for method in methods]
            q = pack["q"]; arrow_q = pack["arrow_q"]
            planes = pack["plane_delta_j"][indices]
            descent_u = pack["descent_u"][indices]; descent_v = pack["descent_v"][indices]
        scales = []
        for m, method in enumerate(methods):
            ax = fig.add_subplot(inner[m, 0])
            vmax = max(float(np.quantile(planes[m][np.isfinite(planes[m])], .985)), np.finfo(float).tiny)
            normalized = np.clip(planes[m]/vmax, 0, 1)
            image = ax.imshow(normalized, origin="lower", extent=(q[0],q[-1],q[0],q[-1]),
                              cmap="viridis", vmin=0, vmax=1, interpolation="bilinear", aspect="equal")
            xx, yy = np.meshgrid(arrow_q, arrow_q)
            gx, gy = descent_u[m], descent_v[m]
            magnitude = np.hypot(gx, gy)
            valid = (magnitude > 1e-30) & (np.hypot(xx, yy) > 1e-12)
            glyph_length = .62 * float(q[-1]) / 3.675
            ax.quiver(xx[valid], yy[valid], glyph_length*gx[valid]/magnitude[valid],
                glyph_length*gy[valid]/magnitude[valid], angles="xy", scale_units="xy",
                scale=1, pivot="middle", color="#F8FAFC", edgecolor="#35404D",
                linewidth=.20, width=.0040, headwidth=3.2, headlength=4.0,
                headaxislength=3.6, zorder=8)
            marker = ax.scatter(0,0,marker="x",color="#B33467",s=42,linewidth=1.6,zorder=20)
            marker.set_path_effects([pe.Stroke(linewidth=2.6,foreground="white"),pe.Normal()])
            ticks = [float(q[0]), 0., float(q[-1])]
            ax.set(xlim=(q[0],q[-1]),ylim=(q[0],q[-1]),xticks=ticks,yticks=ticks)
            if m == 1:
                ax.set_xlabel(r"$s_{\nabla}$ (rad)",labelpad=2)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(r"$s_{\perp}$ (rad)",labelpad=2)
            ax.tick_params(length=3,pad=2)
            ax.set_title(method,fontsize=10.5,color=COLORS[method],pad=5,fontweight="semibold")
            scales.append(vmax)
        observations.append(dict(setting_id=sid,methods=list(methods),
            data_source=str(source_path), half_range_l2_rad=float(q[-1]),
            grid_size=len(q), command_and_native_basis_unchanged=True,
            dimensional_color_limits=dict(zip(methods,scales))))
    colorbar_ax=fig.add_axes([.38,.42/height,.28,.14/height])
    bar=fig.colorbar(image,cax=colorbar_ax,orientation="horizontal",ticks=[0,.5,1])
    bar.set_label(r"$\Delta J/Q_{98.5}$",fontsize=10,labelpad=2)
    bar.ax.tick_params(labelsize=8,length=2,pad=2)
    assert len(fig.axes) == 2*len(ids)+1
    stem=f"Appendix_{prefix}_Decision_{group}"
    paths=[]
    for suffix in ("png","pdf","svg"):
        path=output/f"{stem}.{suffix}"
        save_figure(fig,path,dpi=220)
        paths.append(str(path))
    manifest=dict(stem=stem,setting_ids=ids,case_tile_count=len(ids),data_axes_count=2*len(ids),
        axes_count=len(fig.axes),layout=[rows,3],inner_layout=[2,1],colorbar_axes_count=1,
        array_inset_axes_count=0,
        methods_by_setting={sid:["Conventional",selection[sid]] for sid in ids},
        command_alignment=alignment,
        half_range_l2_rad=(observations[0]['half_range_l2_rad']
            if len({x['half_range_l2_rad'] for x in observations}) == 1 else None),
        half_ranges_l2_rad={x['setting_id']:x['half_range_l2_rad'] for x in observations},
        scales="linear",contours=False,
        viewing_scope="Case-specific wider cached windows; same range for both compared methods, exact original Single endpoint and native basis retained",
        plane_algorithm="main Figure 3 native endpoint gradient / last accepted step",
        arrow_algorithm="main Figure 4 sparse normalized projected analytic negative gradients",
        normalization="main Figure 4: Delta J / per-section 98.5th percentile",
        observations=observations,paths=paths)
    write_bytes(output/f"{stem}.json",(json.dumps(manifest,indent=2)+"\n").encode())
    plt.close(fig)
    return manifest


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"data/decision")
    parser.add_argument("--output",type=Path,default=ROOT/"figures")
    args=parser.parse_args()
    render(ARRAY_IDS,"Arrays",args.data,args.output)
    render(FREQUENCY_IDS,"Frequencies",args.data,args.output)
