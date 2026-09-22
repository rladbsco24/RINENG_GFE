"""Independent elastic mechanics of three current compact pressure-field classes.

The phase commands are fixed-configuration examples, not a paired optimizer ranking.
No sign change is ever applied to the elastic mechanical validator.
"""
from __future__ import annotations

from pathlib import Path
from copy import copy
import io
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from . import pipeline as p
from .cache import CacheStore, digest_array, digest_file, digest_payload

SCHEMA = "B3-frozen-class-elastic-mechanics-v1"
PRESSURE_POINTS = 121
FORCE_POINTS = 15
FIELD_HALF_WIDTH_M = 0.007
STYLE = {"font.size": 12, "axes.labelsize": 13, "axes.labelweight": "semibold",
         "axes.linewidth": 1.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
         "pdf.fonttype": 42, "svg.fonttype": "none"}


def transverse_descriptor(field, target, wavelength):
    """Orientation-invariant twofold amplitude modulation on the first ring peak.

    This is a descriptive pressure statistic, not a proof of endpoint topology.
    Phase winding is exported to distinguish a circulating null from two lobes.
    """
    radii = np.linspace(.04, .8, 80) * wavelength
    theta = np.linspace(0, 2*np.pi, 256, endpoint=False)
    points = np.tile(np.asarray(target), (len(radii), len(theta), 1))
    points[:, :, 0] += radii[:, None]*np.cos(theta)
    points[:, :, 1] += radii[:, None]*np.sin(theta)
    flat = points.reshape(-1, 3)
    pressure = np.concatenate([field.pressure(flat[i:i+2048]) for i in range(0, len(flat), 2048)]).reshape(len(radii), -1)
    means = np.abs(pressure).mean(1)
    peaks = np.flatnonzero((means[1:-1] > means[:-2]) & (means[1:-1] >= means[2:])) + 1
    index = int(peaks[0]) if len(peaks) else int(np.argmax(means))
    ring = pressure[index]
    coefficient = np.mean(np.abs(ring)*np.exp(-2j*theta))/max(means[index], 1e-30)
    winding = float(np.angle(np.roll(ring, -1)*ring.conj()).sum()/(2*np.pi))
    return dict(ring_radius_mm=float(radii[index]*1000),
        twofold_amplitude_modulation=float(abs(coefficient)),
        twofold_orientation_deg=float(-.5*np.angle(coefficient)*180/np.pi),
        phase_winding=float(winding), ring_minimum_relative_amplitude=float(np.abs(ring).min()/means[index]),
        target_to_ring_mean_pressure=float(abs(field.pressure(target))/means[index])), dict(
            radii_m=radii,theta_rad=theta,ring_pressure_complex_pa=pressure,
            selected_radius_index=np.asarray(index))


def _load_commands(root, ctx):
    """Regenerate the three established examples with the current FE policy."""
    import importlib.util
    from .main_feg import feg_single_objective, _solve
    objective = feg_single_objective(ctx)
    initial = np.random.default_rng(260828).uniform(-np.pi, np.pi, objective.n_transducers)
    vortex, _ = _solve(ctx, objective, initial, 260828, "Appendix B3 Vortex")
    out = [dict(name="Vortex", method="GFE", sign="Standard", phase=vortex.phase_rad,
        seed=260828, source_case_id="B3_Vortex", source_array_key="Vortex",
        optimizer_success=bool(vortex.success), optimizer_message=str(vortex.message),
        source_iterations=int(vortex.iterations), source_maxiter=10000)]
    spec = importlib.util.spec_from_file_location("b3_current_sensitivity", root / "appendix_B_pressure/sensitivity_b.py")
    sensitivity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sensitivity)
    for name, method, sign, display_sign in (("Twin", "FE", "corrected_minus", "Standard"),
                                           ("Bottle", "RH", "historical_plus", "Alternative")):
        config = sensitivity.configuration(name, sign, method, seed=260905)
        path = root / "appendix_B_pressure/data/sensitivity_cache" / (sensitivity.cache_key(config) + ".npz")
        if not path.exists() or not path.with_suffix(".json").exists():
            sensitivity.solve_one(config, root / "appendix_B_pressure")
        record, arrays = sensitivity.load_run(config, root / "appendix_B_pressure")
        out.append(dict(name=name, method=method, sign=display_sign,
            phase=arrays["terminal_phase_rad"], seed=260905,
            source_case_id="B3_"+name, source_array_key=name,
            optimizer_success=bool(record["success"]), optimizer_message=str(record["message"]),
            source_iterations=int(record["iterations"]), source_maxiter=int(record["maxiter"])))
    return out


def _mechanics(ctx, command, output):
    validation = p._finite_ka_validation(ctx,command["phase"],p.MAIN_TARGET_M,key="B3_"+command["name"])
    row = p._static_validation_row(command["name"], validation)
    row.update({k:v for k,v in command.items() if k != "phase"})
    jac = validation.force_jacobian_n_m
    symmetric = -.5*(jac+jac.T)
    eig = np.linalg.eigvalsh(symmetric)
    for i,axis in enumerate("xyz"):
        row["k_"+axis+"_mN_per_m"] = float(symmetric[i,i]*1000)
    row["minimum_stiffness_mN_per_m"] = float(eig.min()*1000)
    row["resolved_restoring_root"] = bool(validation.equilibrium.numerical_root_found and eig.min()>0)
    row["stiffness_anisotropy_max_to_min"] = float(eig.max()/eig.min()) if eig.min()>0 else np.nan
    row["transverse_stiffness_ratio"] = float(max(symmetric[0,0],symmetric[1,1])/min(symmetric[0,0],symmetric[1,1])) if min(symmetric[0,0],symmetric[1,1])>0 else np.nan
    np.savez_compressed(output/("B3_"+command["name"]+"_mechanics.npz"),phase_rad=command["phase"],
        target_m=p.MAIN_TARGET_M,equilibrium_m=validation.equilibrium.equilibrium_m,
        force_jacobian_n_m=jac,symmetric_stiffness_n_m=symmetric,stiffness_eigenvalues_n_m=eig,
        protocol_json=json.dumps(p._finite_ka_protocol(ctx),default=str))
    return validation,row


def _fields(ctx,command,output,*,pressure_points=PRESSURE_POINTS,cache_only=False):
    identity = dict(schema=SCHEMA,phase=digest_array(command["phase"]),protocol=p._finite_ka_protocol(ctx),
        positions=digest_array(ctx.positions_m),normals=digest_array(ctx.normals),
        half_width_m=FIELD_HALF_WIDTH_M,pressure_points=pressure_points,force_points=FORCE_POINTS,
        model_content_id=digest_file(Path(p.__file__).with_name("exact_validator.py")))
    path = output/"cache"/(digest_payload(identity)+".npz")
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as data: return {k:data[k].copy() for k in data.files},True
    if cache_only:raise FileNotFoundError(f"Production B3 field cache missing: {path}")
    evaluator = p._exact_evaluator_for_phase(ctx,command["phase"])
    axis = np.linspace(-FIELD_HALF_WIDTH_M,FIELD_HALF_WIDTH_M,pressure_points)
    sparse = np.linspace(-FIELD_HALF_WIDTH_M,FIELD_HALF_WIDTH_M,FORCE_POINTS)
    data = dict(axis_m=axis,force_axis_m=sparse,phase_rad=command["phase"],target_m=p.MAIN_TARGET_M,
        identity_json=np.asarray(json.dumps(identity,default=str)))
    for plane,axes in [("XY",(0,1)),("XZ",(0,2))]:
        for tag,coordinates in [("pressure",axis),("force",sparse)]:
            u,v = np.meshgrid(coordinates,coordinates,indexing="xy")
            points = np.broadcast_to(p.MAIN_TARGET_M,u.shape+(3,)).copy()
            points[...,axes[0]] += u;points[...,axes[1]] += v
            flat = points.reshape(-1,3)
            data[plane+"_"+tag+"_points_m"] = points
            if tag=="pressure":
                data[plane+"_pressure_complex_pa"] = np.concatenate([evaluator.field.pressure(flat[i:i+2048]) for i in range(0,len(flat),2048)]).reshape(u.shape)
            else:
                forces = [evaluator.force(point) for point in flat]
                data[plane+"_total_force_n"] = np.asarray([force.total_n for force in forces]).reshape(u.shape+(3,))
                data[plane+"_radiation_force_n"] = np.asarray([force.radiation_n for force in forces]).reshape(u.shape+(3,))
    descriptor,ring = transverse_descriptor(evaluator.field,p.MAIN_TARGET_M,343./ctx.config.frequency_hz)
    data.update({"morphology_"+k:np.asarray(v) for k,v in descriptor.items()})
    data.update({"morphology_"+k:v for k,v in ring.items()})
    np.savez_compressed(path,**data)
    return data,False


def _render(output,commands,rows,fields):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(3,2,figsize=(9.4,13.7))
        fig.subplots_adjust(left=.14,right=.94,bottom=.18,top=.955,hspace=.61,wspace=.28)
        for i,(command,row,data) in enumerate(zip(commands,rows,fields)):
            peak=max(float(abs(data[plane+"_pressure_complex_pa"]).max()) for plane in ("XY","XZ"))
            for j,(plane,pair) in enumerate([("XY",(0,1)),("XZ",(0,2))]):
                ax=axes[i,j];amplitude=abs(data[plane+"_pressure_complex_pa"])/peak
                axis=data["axis_m"]*1000
                image=ax.imshow(amplitude,origin="lower",extent=[axis[0],axis[-1],axis[0],axis[-1]],cmap="magma",vmin=0,vmax=1)
                ax.contour(axis,axis,amplitude,levels=[.25,.5,.75],colors="white",linewidths=.8,alpha=.5)
                force=data[plane+"_total_force_n"][...,list(pair)]
                norm=np.linalg.norm(force,axis=-1)
                direction=np.divide(force,norm[...,None],out=np.zeros_like(force),where=norm[...,None]>1e-15)
                sparse=data["force_axis_m"]*1000
                ax.quiver(sparse,sparse,direction[...,0],direction[...,1],color="#71DFF4",pivot="mid",
                    angles="xy",scale_units="xy",scale=2.1,width=.005,headwidth=3.3,alpha=.85)
                ax.plot(0,0,"+",color="white",ms=10,mew=2)
                offset=np.array([row["finite_ka_equilibrium_x_m"],row["finite_ka_equilibrium_y_m"],row["finite_ka_equilibrium_z_m"]])-p.MAIN_TARGET_M
                if row["finite_ka_root_found"]:
                    if row["resolved_restoring_root"]:
                        ax.plot(offset[pair[0]]*1000,offset[pair[1]]*1000,"o",mfc="#BFEA66",mec="black",ms=7,mew=1.1)
                    else:
                        ax.plot(offset[pair[0]]*1000,offset[pair[1]]*1000,"D",mfc="#FF827B",mec="black",ms=7,mew=1.1)
                ax.set_xlabel(r"$x-x_0$ (mm)");ax.set_ylabel((r"$y-y_0$" if plane=="XY" else r"$z-z_0$")+" (mm)")
                ax.set_xticks([-6,0,6]);ax.set_yticks([-6,0,6]);ax.set_aspect("equal")
                ax.set_title(f"{command['name']} · {plane}",fontweight="semibold",fontsize=14,pad=10)
                ax.text(-.13,1.035,f"({chr(97+2*i+j)})",transform=ax.transAxes,fontsize=15,fontweight="bold")
            # Numerical force and stiffness details remain in the companion table.
            text=f"{command['method']} · {command['sign'].lower()} sign"
            bottom=min(ax.get_position().y0 for ax in axes[i])
            fig.text(.54,bottom-.050,text,ha="center",fontsize=11.2)
        cax=fig.add_axes([.30,.036,.48,.013]);bar=fig.colorbar(image,cax=cax,orientation="horizontal",ticks=[0,.5,1])
        bar.set_label("Normalized pressure",fontsize=11)
        handles=[Line2D([],[],marker="+",color=".25",ls="none",ms=9,mew=2,label="Target"),
                 Line2D([],[],marker="o",mfc="#BFEA66",mec="black",color="black",ls="none",label="Restoring root"),
                 Line2D([],[],marker="D",mfc="#FF827B",mec="black",color="black",ls="none",label="Non-restoring root")]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.53,.076),ncol=3,frameon=False,fontsize=10.5)
        paths=[]
        for extension in ("png","pdf","svg"):
            path=output.parent.parent/"figures"/("Figure_B3_exact_field_mechanics."+extension)
            path.parent.mkdir(exist_ok=True)
            buffer=io.BytesIO();fig.savefig(buffer,format=extension,dpi=240,facecolor="white")
            payload=buffer.getvalue()
            if extension=="png":
                from PIL import Image
                with Image.open(io.BytesIO(payload)) as image:image.verify()
            temporary=path.with_name(path.name+".partial")
            temporary.write_bytes(payload);os.replace(temporary,path);paths.append(str(path))
        plt.close(fig)
    return paths


def run(root,ctx=None,*,output_root=None,production=False,cache_only=False):
    root=Path(root).resolve();output=Path(output_root) if output_root is not None else root/"appendix_B_pressure/data/B3";output.mkdir(parents=True,exist_ok=True)
    if production and (ctx is None or not ctx.config.full):raise ValueError("Production B3 requires the full shared context")
    pressure_points=181 if production else PRESSURE_POINTS
    if ctx is None:ctx=p.prepare_pipeline(run_mode="quick",recompute=False,output_root=root/"smoke_outputs")
    if cache_only and not ctx.cache.read_only:
        ctx=copy(ctx);ctx.cache=CacheStore(ctx.cache.root,namespace=ctx.cache.namespace,read_only=True)
    commands=_load_commands(root,ctx);rows=[];all_fields=[];force_rows=[];cache_hits=0
    with threadpool_limits(limits=1):
        for command in commands:
            validation,row=_mechanics(ctx,command,output)
            values,cached=_fields(ctx,command,output,pressure_points=pressure_points,cache_only=cache_only);cache_hits+=int(cached)
            row.update({k.removeprefix("morphology_"):float(v) for k,v in values.items() if k.startswith("morphology_") and v.ndim==0})
            all_fields.append(values);rows.append(row)
            np.savez_compressed(output/("B3_"+command["name"]+"_fields.npz"),**values)
            for plane in ("XY","XZ"):
                for point,force,radiation in zip(values[plane+"_force_points_m"].reshape(-1,3),values[plane+"_total_force_n"].reshape(-1,3),values[plane+"_radiation_force_n"].reshape(-1,3)):
                    force_rows.append(dict(field_class=command["name"],plane=plane,
                        **{f"{axis}_m":point[k] for k,axis in enumerate("xyz")},
                        **{f"total_force_{axis}_n":force[k] for k,axis in enumerate("xyz")},
                        **{f"radiation_force_{axis}_n":radiation[k] for k,axis in enumerate("xyz")}))
            print(f"B3 {command['name']}: root={row['finite_ka_root_found']}, d/a={row['finite_ka_displacement_a']:.4f}; fields {'cached' if cached else 'new'}",flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/"B3_mechanical_properties.csv",index=False)
    pd.DataFrame(force_rows).to_csv(output/"B3_force_samples.csv",index=False)
    paths=_render(output,commands,rows,all_fields)
    caption=("Figure B3. Independent finite-ka elastic total-force validation of three current compact commands: nominal standard-sign GFE Vortex; "
        "standard-sign FE Twin (directional ratio 1:0.01:0.01); and alternative-sign RH Bottle (0.01:0.01:1). "
        "Rows show Vortex, Twin, and Bottle; columns show XY and XZ pressure sections through the prescribed target. "
        "Pressure is normalized by one shared XY/XZ peak per command; contours mark 0.25, 0.50 and 0.75. "
        "Equal-length cyan arrows show projected elastic total-force directions, including effective gravity. "
        "Green circles denote restoring roots and coral diamonds non-restoring roots; each marker projects the independently resolved three-dimensional total-force root. Displacement d is its three-dimensional distance from the prescribed target. "
        "Directional stiffness k_i=-J_ii and all eigenvalues of -(J+J^T)/2 are evaluated at that root; the full matrices and raw force vectors are exported. "
        f"Resolved roots: {int(frame.finite_ka_root_found.sum())}/3; restoring roots: {int(frame.resolved_restoring_root.sum())}/3. Full Jacobians and stiffness eigenvalues determine the displayed root classifications independently of pressure shape. "
        "These are one frozen example per field class, not a paired method ranking or a basin-frequency estimate. "
        "The Vortex uses seed 260828; Twin/Bottle use seed 260905. The FE Twin optimization stopped with precision loss; optimizer stopping status is reported separately from independent mechanical root acceptance and restoring status. "
        "The elastic validator is unchanged across surrogate signs. "+("PRODUCTION." if production else "SMOKE."))
    (output/"B3_figure_caption.txt").write_text(caption+"\n")
    columns=["name","method","sign","finite_ka_displacement_m","k_x_mN_per_m","k_y_mN_per_m","k_z_mN_per_m","minimum_stiffness_mN_per_m","twofold_amplitude_modulation","phase_winding"]
    (output/"B3_measured_results.txt").write_text("Three frozen representative fields; common independent elastic model.\n\n"+frame[columns].to_string(index=False)+"\n\nNo optimizer campaign or endpoint-frequency inference is performed.\n")
    protocol=dict(schema=SCHEMA,commands=[{k:v for k,v in command.items() if k!="phase"} for command in commands],
        target_m=p.MAIN_TARGET_M.tolist(),mechanical_protocol=p._finite_ka_protocol(ctx),
        field_half_width_m=FIELD_HALF_WIDTH_M,pressure_points_per_axis=pressure_points,force_points_per_axis=FORCE_POINTS,
        display_plane_count=6,force_sample_count=len(force_rows),field_cache_hits=cache_hits,
        morphology=dict(radii_wavelength=[.04,.8],radius_count=80,azimuth_count=256,rule="First radial maximum of azimuthally averaged pressure amplitude; orientation-invariant second azimuthal Fourier coefficient; phase winding reported separately."),
        representative_selection="Declared objective, seed and morphology examples synthesized before independent mechanical evaluation; no outcome-based selection.",
        physics_source_content_id=digest_file(Path(p.__file__).with_name("exact_validator.py")),evidence_status="PRODUCTION" if production else "SMOKE")
    (output/"B3_protocol.json").write_text(json.dumps(protocol,indent=2,default=str))
    return dict(figures=paths,table=str(output/"B3_mechanical_properties.csv"),caption=str(output/"B3_figure_caption.txt"),protocol=str(output/"B3_protocol.json"))
