"""Compare predefined viewing windows of unchanged native decision sections."""
from pathlib import Path
from io import BytesIO
import json
import zipfile
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from pypdf import PdfReader,PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from appendix_io import write_bytes

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/opposed_visual_window'
W,H=A4
FONT_DIR=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('WindowText',str(FONT_DIR/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('WindowBold',str(FONT_DIR/'DejaVuSans-Bold.ttf')))
COLORS={'Conventional':'#3B6FA1','FE':'#C76A12','GFE':'#007C78'}
RANGES=(.3,3.,10.)

def overlay(title,subtitle,number,paragraphs):
    stream=BytesIO(); c=canvas.Canvas(stream,pagesize=(W,H))
    c.setFont('WindowBold',17);c.drawString(36,H-34,title)
    c.setFont('WindowText',9);c.setFillColorRGB(.30,.34,.38);c.drawString(36,H-52,subtitle)
    for y,paragraph in paragraphs:
        c.setFont('WindowText',9)
        for line in textwrap.wrap(paragraph,105):c.drawString(36,y,line);y-=13
    c.setFont('WindowText',8);c.drawRightString(W-36,17,str(number));c.save()
    return PdfReader(stream).pages[0]

def draw_panel(fig,sid,method,radius,rect,heading=None):
    with np.load(DATA/f'{sid}_{method}_range{radius:g}.npz') as z:
        q=z['q'];plane=z['plane_delta_j'];aq=z['arrow_q'];u=z['descent_u'];v=z['descent_v']
    ax=fig.add_axes([rect[0]/W,rect[1]/H,rect[2]/W,rect[3]/H])
    scale=np.quantile(plane,.985)
    image=ax.imshow(np.clip(plane/scale,0,1),extent=(-radius,radius,-radius,radius),
        origin='lower',cmap='viridis',vmin=0,vmax=1,interpolation='bilinear',aspect='equal')
    xx,yy=np.meshgrid(aq,aq);mag=np.hypot(u,v);valid=(mag>1e-30)&(np.hypot(xx,yy)>1e-12)
    length=.62*radius/3.675
    ax.quiver(xx[valid],yy[valid],length*u[valid]/mag[valid],length*v[valid]/mag[valid],
        angles='xy',scale_units='xy',scale=1,pivot='middle',color='#F8FAFC',edgecolor='#35404D',
        linewidth=.20,width=.004,headwidth=3.2,headlength=4.,headaxislength=3.6)
    marker=ax.scatter(0,0,marker='x',color='#B33467',s=35,linewidth=1.5,zorder=20)
    marker.set_path_effects([pe.Stroke(linewidth=2.5,foreground='white'),pe.Normal()])
    ax.set(xticks=[-radius,0,radius],yticks=[-radius,0,radius],
        xlabel=r'$s_{\nabla}$ (rad)',ylabel=r'$s_{\perp}$ (rad)')
    ax.tick_params(length=2.5,pad=2)
    if heading:ax.set_title(heading,fontsize=10.5,pad=8,fontweight='semibold')
    return image

def add_page(writer,fig,layer):
    stream=BytesIO()
    with plt.rc_context({'savefig.bbox':None}):fig.savefig(stream,format='pdf',facecolor='white',bbox_inches=None)
    plt.close(fig);page=PdfReader(stream).pages[0]
    assert abs(float(page.mediabox.width)-W)<1e-6
    page.merge_page(layer);writer.add_page(page)

def colorbar(fig,image,bottom):
    ax=fig.add_axes([.34,bottom/H,.32,7/H]);bar=fig.colorbar(image,cax=ax,orientation='horizontal',ticks=[0,.5,1])
    bar.ax.tick_params(labelsize=8,length=2,pad=2)
    bar.set_label(r'$\Delta J/Q_{98.5}$',fontsize=9,labelpad=2)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.7,'pdf.fonttype':42})
    writer=PdfWriter();fig=plt.figure(figsize=(W/72,H/72))
    lefts=[62,242,422];bottoms=[580,370,160];size=140
    for row,method in enumerate(('Conventional','FE','GFE')):
        fig.text(12/W,(bottoms[row]+size/2)/H,method,color=COLORS[method],fontweight='semibold',
            rotation=90,ha='center',va='center',fontsize=11)
        for col,radius in enumerate(RANGES):
            image=draw_panel(fig,'S02',method,radius,[lefts[col],bottoms[row],size,size],
                heading=rf'$\pm${radius:g} rad' if row==0 else None)
    colorbar(fig,image,100)
    add_page(writer,fig,overlay('Opposed: the viewing window reveals the basin',
        'Continued endpoints | 512 transducers | 40 kHz | Single',1,
        [(57,'The endpoint and its native two-direction plane are fixed across each row. Only the viewing range changes. '
             'Equal coordinate scales and the original linear color normalization are retained.'),
         (28,'The phase step is L2-normalized: 3 rad along one axis corresponds to 0.133 rad RMS per transducer.')]))

    fig=plt.figure(figsize=(W/72,H/72))
    for col,radius in enumerate(RANGES):
        image=draw_panel(fig,'A09','FE',radius,[lefts[col],560,size,size],heading=rf'$\pm${radius:g} rad')
    colorbar(fig,image,496)
    summary=pd.read_csv(DATA/'window_summary.csv')
    basis=json.loads((DATA/'basis_control/summary.json').read_text())
    layer=overlay('Tilted-plane FE: the same window effect',
        'Current endpoint | 256 transducers | 40 kHz | Single',2,
        [(442,'The fixed Tilted-plane FE section also reveals more transverse curvature as the window expands. '
              'This check changes neither its command nor its objective or directional weights.'),
         (387,'There are two distinct contributions to the earlier Opposed improvement. Continuing FE barely changed '
              'its phase command, but the native last-step construction selected a substantially different second '
              'direction. That exposes a more rounded section of the existing objective.'),
         (316,'The comparisons in this PDF then hold that sampled plane fixed. Their improvement is a viewing-range '
              'effect: transverse curvature becomes visible alongside the sharp normal direction. GFE remains '
              'more elongated than FE, but its basin is easier to see in the larger window.'),
         (244,'For the appendix, +/-3 rad is a useful candidate alongside the current +/-0.30 rad view. The +/-10 rad '
              'column supplies broader context. The range should be stated and kept equal between methods; '
              'these figures use no logarithmic color scale or separate stretching of the two axes.')])
    add_page(writer,fig,layer)
    writer.add_metadata({'/Title':'Decision-space viewing range checks','/Author':'RINENG HAT study'})
    attachment=BytesIO()
    with zipfile.ZipFile(attachment,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('revision_f1_f2/render_decision_window_check.py',Path(__file__).read_bytes())
        for p in DATA.rglob('*'):
            if p.is_file() and p.suffix in {'.py','.json','.csv','.npz','.md'}:
                z.writestr('revision_f1_f2/'+p.relative_to(ROOT).as_posix(),p.read_bytes())
    writer.add_attachment('Decision_Window_Inputs_and_Source.zip',attachment.getvalue())
    output=ROOT/'delivery/RINENG_Decision_Window_Check.pdf'
    stream=BytesIO();writer.write(stream);write_bytes(output,stream.getvalue())
    print(output)

if __name__=='__main__':main()
