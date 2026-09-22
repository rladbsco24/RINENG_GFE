"""Render ACS to the fixed same-setting FE/GFE endpoint set on linear axes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd
from appendix_io import save_figure, write_bytes

ROOT=Path(__file__).resolve().parent
ARRAY_IDS=['A06','S01','S02','S03','S04','S05','A07','A08','A09']
FREQUENCY_IDS=['F20','S06','F28','A06','S07','S08']
COLORS={'Conventional':'#3B6FA1'}
LINESTYLES={'Conventional':'-'}
LABELS={'A06':'Square','S01':'Rectangular','S02':'Opposed','S03':'Spherical cap',
        'S04':'Fermat disk','S05':'Arbitrary 3D','A07':'Annular','A08':'Four clusters',
        'A09':'Tilted plane','F20':'Square','F28':'Square','S06':'Square','S07':'Square','S08':'Square'}


def render(data_dir=ROOT/'data',output=ROOT/'figures'):
    from representative_selection import load_selection

    data_dir,output=Path(data_dir),Path(output)
    representatives=load_selection(data_root=data_dir)
    output.mkdir(parents=True,exist_ok=True)
    extended=data_dir/'compact_acs/plotgrid.csv'
    if not extended.exists():
        raise FileNotFoundError('Run compact_acs.py to extract the completed Conventional history before rendering.')
    table=pd.read_csv(extended)
    protocol=json.loads((data_dir/'compact_acs/protocol.json').read_text())
    if protocol['matching_rule']!='fixed_same_setting_endpoint_set' or protocol['reference_method']!='FE_or_GFE':
        raise ValueError('The Conventional curve requires the unchanged eight-endpoint FE/GFE reference set.')
    if set(table.trajectory_method)!={'Conventional'}:
        raise ValueError('Only the Conventional trajectory is displayed in this figure family.')
    endpoints=pd.read_csv(data_dir/'compact_acs/endpoints.csv')
    populations=set(table.n_pairs.astype(int))
    if len(populations)!=1:
        raise ValueError('All settings must use the same paired-start population.')
    n_pairs=populations.pop()
    # Include every actually accepted iteration, then hold each final command
    # to the shared endpoint so that all four starts contribute throughout.
    terminal_max=int(endpoints.total_iterations.max())
    xmax=max(10000,int(np.ceil(terminal_max/1000)*1000))
    if xmax!=protocol['displayed_iteration_max'] or terminal_max!=protocol['max_actual_accepted_iterations']:
        raise ValueError('Displayed iteration limits differ from the completed native histories.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11,
        'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':11,
        'axes.linewidth':.85,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    manifests=[]
    for family,ids,shape,size,stem in (
        ('arrays',ARRAY_IDS,(3,3),(11,10),'Appendix_3A_ACS_Arrays'),
        ('frequencies',FREQUENCY_IDS,(2,3),(11,7),'Appendix_3B_ACS_Frequencies')):
        missing=set(ids)-set(table.setting_id)
        if missing:
            raise ValueError(f'Missing ACS cases: {missing}')
        fig,axes=plt.subplots(*shape,figsize=size,sharex=True,sharey=True)
        fig.subplots_adjust(left=.078,right=.960,bottom=.080 if family=='arrays' else .105,
                            top=.925 if family=='arrays' else .890,hspace=.32,wspace=.22)
        panel_records=[]
        for index,(ax,sid) in enumerate(zip(axes.ravel(),ids)):
            case=table[table.setting_id==sid]
            frequency=float(case.frequency_hz.iloc[0])/1000
            for method in COLORS:
                curve=case[case.trajectory_method==method].sort_values('iteration_budget')
                x=curve.iteration_budget.to_numpy(float)
                mean=curve.budget_mean.to_numpy(float)
                use=x<=xmax
                ax.plot(x[use],mean[use],color=COLORS[method],linestyle=LINESTYLES[method],
                        linewidth=2.2,zorder=2)
            label=LABELS[sid] if family=='arrays' else f'{frequency:g} kHz'
            title=f'({chr(97+index)}) {label} · {representatives[sid]}'
            ax.set_title(title,loc='left',pad=7)
            panel_records.append(dict(setting_id=sid,title=title,
                force_equilibrium_representative=representatives[sid]))
            ax.set_xscale('linear')
            ax.set_yscale('linear')
            ax.set_xlim(0,xmax)
            ax.set_ylim(0,1)
            ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos: f'{int(value):,}'))
            ax.set_yticks([0,.25,.5,.75,1])
            ax.grid(axis='y',color='.91',linewidth=.55)
            ax.tick_params(direction='out',length=3.5,width=.8)
        fig.legend(handles=[Line2D([0],[0],color=COLORS[m],ls=LINESTYLES[m],lw=2.2,label=m) for m in COLORS],
                   loc='upper center',bbox_to_anchor=(.54,.995),ncol=1,frameon=False,handlelength=2.6)
        fig.supxlabel('Accepted iterations',x=.53,y=.018,fontsize=12)
        fig.supylabel('Average cosine similarity',x=.014,y=.50,fontsize=12)
        assert len(fig.axes)==len(ids)
        paths=[]
        for extension in ('pdf','svg','png'):
            path=output/f'{stem}.{extension}'
            save_figure(fig,path,dpi=220 if extension=='png' else None)
            paths.append(str(path))
        plt.close(fig)
        manifests.append(dict(family=family,setting_ids=ids,data_axes_count=len(ids),layout=list(shape),paths=paths,
            axes_count=len(ids),array_inset_axes_count=0,
            reference_method='FE_or_GFE',matching_rule='fixed_same_setting_endpoint_set',
            references_per_setting=8,n_pairs=n_pairs,xscale='linear',yscale='linear',
            displayed_iteration_max=xmax,
            displayed_methods=['Conventional'],
            heading_method='The closest frozen-bank FE/GFE member to the terminal seed260828 Conventional command; the ACS reference bank remains all eight endpoints',
            panels=panel_records,
            trajectory='Every native accepted update in the retained completed Conventional search chain',
            similarity='Mean over four trajectories of the maximum absolute complex phase cosine to the frozen eight-reference set; no running maximum or smoothing',
            alpha_selection='First-seed displacement-selected alpha frozen per method and setting for all paired starts',
            source_data=str(extended)))
    write_bytes(output/'acs_figure_manifest.json',(json.dumps(manifests,indent=2)+'\n').encode())
    return manifests


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'figures')
    args=parser.parse_args()
    render(args.data,args.output)
