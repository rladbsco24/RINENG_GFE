"""Standalone array geometry: source points are emitting-face cylinder centers.

The cylinders are schematic housings, not a change to the acoustic source model.
Every saved source position and normal is retained; no geometry is regenerated.
"""
from pathlib import Path
import argparse
import fingerprintlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from appendix_settings import ROOT, SETTINGS, ARRAY_IDS
from appendix_io import save_figure, write_bytes
from array_insets import load_source_geometry

STEM='Appendix_0_ArrayLayouts'
RADIUS_M=.004
HEIGHT_M=.004
FACETS=16
AZIMUTH_DEG=-62.
ELEVATION_DEG=28.

def digest(value):
    return fingerprintlib.fingerprint(np.ascontiguousarray(value,dtype='<f8').tobytes()).hexdigest()

def view_basis():
    az,el=np.deg2rad([AZIMUTH_DEG,ELEVATION_DEG])
    horizontal=np.array([-np.sin(az),np.cos(az),0.])
    camera=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    vertical=np.cross(camera,horizontal)
    return np.stack([horizontal,vertical,camera])

def cylinder_faces(point,normal):
    """Return true 3D face polygons; the top-face center is exactly point."""
    normal=normal/np.linalg.norm(normal)
    auxiliary=np.eye(3)[np.argmin(np.abs(normal))]
    u=np.cross(normal,auxiliary);u/=np.linalg.norm(u)
    v=np.cross(normal,u)
    theta=np.arange(FACETS)*2*np.pi/FACETS
    ring=RADIUS_M*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
    top=point+ring
    bottom=top-HEIGHT_M*normal
    assert np.allclose(top.mean(axis=0),point,atol=1e-15,rtol=0)
    assert np.allclose(top.mean(axis=0)-bottom.mean(axis=0),HEIGHT_M*normal,atol=1e-15,rtol=0)
    light=np.array([-.35,-.5,.8]);light/=np.linalg.norm(light)
    faces=[(bottom[::-1],np.array([.25,.36,.46])),(top,np.array([.56,.80,.84]))]
    base=np.array([.39,.53,.64])
    for i in range(FACETS):
        j=(i+1)%FACETS
        outward=(ring[i]+ring[j]);outward/=np.linalg.norm(outward)
        brightness=.72+.28*max(float(outward@light),0.)
        faces.append((np.array([bottom[i],bottom[j],top[j],top[i]]),base*brightness))
    # A small disk identifies the precise source point on the emitting face.
    point_disk=point+normal*1e-6+.13*ring
    faces.append((point_disk,np.array([.10,.23,.28])))
    return faces

def render(data_root=ROOT/'data',output=ROOT/'figures'):
    data_root,output=Path(data_root),Path(output)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    basis=view_basis()
    layouts=[]
    for sid in ARRAY_IDS:
        positions,normals=load_source_geometry(data_root,sid)
        # Figure 0 shows the actual final Triple benchmark source layout.
        # Single correspondence retains its separately recorded source budget.
        final_geometry=data_root/'equilibrium_resource_check/release/inputs'/sid/'geometry.npz'
        if sid in ('A07','A08') and final_geometry.exists():
            with np.load(final_geometry,allow_pickle=False) as saved:
                positions,normals=saved['positions_m'],saved['normals']
        unit=normals/np.linalg.norm(normals,axis=1)[:,None]
        assert np.allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-12)
        center=(positions.max(axis=0)+positions.min(axis=0))/2
        projected=(positions-center)@basis.T
        layouts.append((sid,positions,normals,unit,center,projected))
    max_u=max(np.ptp(row[-1][:,0]) for row in layouts)
    max_v=max(np.ptp(row[-1][:,1]) for row in layouts)
    half_u=max_u/2+.019
    half_v=max(max_v/2+.019,half_u*.78)
    fig,axes=plt.subplots(3,3,figsize=(12.8,11.6))
    fig.subplots_adjust(left=.035,right=.98,bottom=.085,top=.945,wspace=.09,hspace=.20)
    records=[]
    for index,((sid,positions,normals,unit,center,projected),ax) in enumerate(zip(layouts,axes.ravel())):
        polygons=[];colors=[];depths=[]
        for point,normal in zip(positions,unit):
            for vertices,color in cylinder_faces(point,normal):
                xyz=(vertices-center)@basis.T
                polygons.append(xyz[:,:2]*1000);depths.append(float(xyz[:,2].mean()));colors.append(color)
        order=np.argsort(depths,kind='stable')
        collection=PolyCollection([polygons[i] for i in order],facecolors=np.asarray(colors)[order],
            edgecolors='none',linewidths=0,antialiaseds=True)
        ax.add_collection(collection)
        label=next(s['label'].split(' · ')[0] for s in SETTINGS if s['id']==sid)
        ax.set_title(f'({chr(97+index)}) {label}   N = {len(positions)}',loc='left',fontsize=12,pad=7)
        ax.set(xlim=(-half_u*1000,half_u*1000),ylim=(-half_v*1000,half_v*1000),aspect='equal')
        ax.axis('off')
        records.append(dict(setting_id=sid,source_count=len(positions),task='Triple',
            positions_fingerprint=digest(positions),normals_fingerprint=digest(normals),
            source_point='Exact top-face center',cylinder_axis='Saved unit source normal',
            housing_extends='From source point in minus-normal direction',
            top_center_max_error_m=float(max(np.linalg.norm(cylinder_faces(p,n)[1][0].mean(0)-p) for p,n in zip(positions,unit))),
            normal_arrow_indices=[]))
    # One common physical scale and one orientation key serve all nine panels.
    handles=[Line2D([],[],marker='o',ls='none',mfc='#8FCCD6',mec='none',ms=8,label='Emitting face'),
             Line2D([],[],marker='o',ls='none',color='#193B47',ms=3,label='Source point')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.49,.017),ncol=2,
        frameon=False,columnspacing=2.8,handlelength=1.8)
    # A scale bar is drawn in physical projected coordinates (mm).
    ax=axes[-1,0]
    y=-half_v*1000*.90;x=-half_u*1000*.80
    ax.plot([x,x+50],[y,y],color='#394B58',lw=1.4)
    ax.text(x+25,y-6,'50 mm',ha='center',va='top',fontsize=9,color='#394B58')
    ax=axes[-1,-1]
    origin=np.array([half_u*1000*.60,-half_v*1000*.65])
    for axis,label in zip(np.eye(3),'xyz'):
        delta=axis@basis[:2].T*23
        ax.annotate('',xy=origin+delta,xytext=origin,
            arrowprops=dict(arrowstyle='->',lw=.9,color='#536674'))
        ax.text(*(origin+delta*1.19),label,fontsize=9,ha='center',va='center',color='#536674')
    output.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf','svg'):
        save_figure(fig,output/f'{STEM}.{extension}',dpi=300,facecolor='white')
    manifest=dict(figure=STEM,task='Geometry',setting_ids=ARRAY_IDS,subplot_count=9,
        cylinder_count=sum(row['source_count'] for row in records),array_inset_axes_count=0,normal_arrow_count=0,
        schematic_cylinder_radius_m=RADIUS_M,schematic_cylinder_height_m=HEIGHT_M,
        model_unchanged=True,projection='Common-scale orthographic',
        view_azimuth_deg=AZIMUTH_DEG,view_elevation_deg=ELEVATION_DEG,
        common_x_limits_mm=[-half_u*1000,half_u*1000],common_y_limits_mm=[-half_v*1000,half_v*1000],
        frequency_scope='Final Triple array cases at 40 kHz; Square 20/25 kHz use 24x24, other Square carriers 16x16. Single Annular/Four clusters retain 512/256 sources.',
        records=records)
    write_bytes(output/f'{STEM}.json',(json.dumps(manifest,indent=2)+'\n').encode())
    plt.close(fig)
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'figures')
    args=parser.parse_args();render(args.data,args.output)
