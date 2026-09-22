"""Render the two pressure-field/isobar appendix plates from exported fields."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from appendix_settings import SETTINGS
from appendix_io import save_figure, write_bytes
from representative_selection import load_selection, validate_alignment

ROOT = Path(__file__).resolve().parent
ARRAY_IDS = ['A06','S01','S02','S03','S04','S05','A07','A08','A09']
FREQUENCY_IDS = ['F20','S06','F28','A06','S07','S08']
COLORS = {'Conventional':'#3B6FA1', 'FE':'#C76A12', 'GFE':'#007C78'}
STYLES = {'Conventional':'--', 'FE':'-', 'GFE':(0,(1.8,1.4))}
CONTOUR_METHODS = ('Conventional', 'FE', 'GFE')

def render(ids, group, data, output, preview=False):
    data, output = Path(data), Path(output)
    selection = load_selection(data.parent)
    alignment = validate_alignment(data.parent, ids, include_all_methods=True) if not preview else None
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
        'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65})
    rows = (len(ids)+2)//3
    frequency_sheet = group=='Frequencies'
    fig, axes = plt.subplots(rows, 3, figsize=(11.6, 8.2 if frequency_sheet else 11.4), squeeze=False)
    fig.subplots_adjust(left=.075,right=.89,top=.90 if frequency_sheet else .925,
        bottom=.085 if frequency_sheet else .065,wspace=.32,hspace=.36)
    cmap = 'viridis'
    levels = [.2,.4,.6,.8]
    images = []
    metrics = []
    for n,(sid,ax) in enumerate(zip(ids,axes.ravel())):
        setting = next(s for s in SETTINGS if s['id']==sid)
        title = setting['label'].split(' · ')[0] if group=='Arrays' else f"{setting['frequency_hz']/1000:g} kHz"
        representative = selection[sid]
        ax.set_title(f'({chr(97+n)}) {title} · {representative}', loc='left', fontsize=10.5, pad=7)
        path = data/f'{sid}_pressure.npz'
        if not path.exists():
            if not preview: raise FileNotFoundError(path)
            ax.text(.5,.5,'Pending computation',ha='center',transform=ax.transAxes)
            continue
        methods = CONTOUR_METHODS
        with np.load(path) as z:
            q=z['coordinate_lambda']
            indices=[list(z['methods']).index(method) for method in methods]
            amp=z['amplitude_pa'][indices]
        pstar=float(np.max(amp))
        background_index = methods.index(representative)
        image=ax.imshow(amp[background_index]/pstar, origin='lower', extent=[q[0],q[-1],q[0],q[-1]],
            cmap=cmap, vmin=0, vmax=1, interpolation='bilinear', aspect='equal', rasterized=True)
        images.append(image)
        for i,method in enumerate(methods):
            contours=ax.contour(q,q,amp[i]/pstar,levels=levels,colors=[COLORS[method]],
                linestyles=[STYLES[method]], linewidths=1.25 if method=='Conventional' else 1.5)
            contours.set_path_effects([pe.Stroke(linewidth=2.2,foreground='white',alpha=.85),pe.Normal()])
        marker,=ax.plot(0,0,'+',ms=10,mew=1.6,color='black')
        marker.set_path_effects([pe.Stroke(linewidth=3,foreground='white'),pe.Normal()])
        ax.set(xlim=(-.5,.5),ylim=(-.5,.5),xticks=[-.5,0,.5],yticks=[-.5,0,.5],
               xlabel=r'$(x-x_t)/\lambda$',ylabel=r'$(z-z_t)/\lambda$')
        reference=amp[background_index].ravel()
        cos=float(amp[0].ravel()@reference/(np.linalg.norm(amp[0])*np.linalg.norm(reference)))
        metrics.append(dict(setting_id=sid, methods=list(methods), contour_methods=list(methods),
            background_method=representative,
            pressure_scale_pa=pstar, conventional_to_representative_amplitude_cosine=cos))
    for ax in axes.ravel()[len(ids):]: ax.remove()
    prefix='1A' if group=='Arrays' else '1B'
    handles=[Line2D([],[],color=COLORS[m],ls=STYLES[m],lw=2,label=m) for m in COLORS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.48,.995),
               ncol=3,frameon=False,fontsize=12,handlelength=2.5,columnspacing=2.8)
    if images:
        cax=fig.add_axes([.93,.19,.017,.61])
        colorbar=fig.colorbar(images[0],cax=cax,ticks=[0,.2,.4,.6,.8,1])
        colorbar.set_label(r'$|p|/p_*$',fontsize=12,labelpad=8)
    assert len(fig.axes) == len(ids) + int(bool(images))
    output.mkdir(parents=True,exist_ok=True)
    stem=f'Appendix_{prefix}_Pressure_{group}'
    for ext in ('png','pdf','svg'):
        save_figure(fig,output/f'{stem}.{ext}',dpi=300,facecolor='white')
    write_bytes(output/f'{stem}.json',json.dumps(dict(figure=prefix, task='Single',subplot_count=len(ids),setting_ids=ids,
        contour_levels_fraction=levels,contour_methods=list(CONTOUR_METHODS),
        background_methods={sid:selection[sid] for sid in ids},
        command_alignment=alignment,
        data_axes_count=len(ids),colorbar_axes_count=int(bool(images)),
        axes_count=len(fig.axes),array_inset_axes_count=0,array_insets=[],
        field_plane='XZ through the Single target',
        normalization='one shared maximum over Conventional, FE and GFE pressure fields per setting',
        metrics=metrics),indent=2).encode())
    plt.close(fig)
    return stem

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data'/'pressure')
    p.add_argument('--output',type=Path,default=ROOT/'figures')
    p.add_argument('--preview',action='store_true')
    a=p.parse_args()
    render(ARRAY_IDS,'Arrays',a.data,a.output,a.preview)
    render(FREQUENCY_IDS,'Frequencies',a.data,a.output,a.preview)
