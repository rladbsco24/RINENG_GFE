"""Regenerate the array-layout plate and eight comparison plates from saved data."""
from pathlib import Path
from io import BytesIO
import json
import subprocess
import sys
from PIL import Image
from pypdf import PdfReader, PdfWriter
from appendix_io import write_bytes

ROOT=Path(__file__).resolve().parent
COMPARISON_STEMS=[f'Appendix_{n}{letter}_{family}_{group}'
       for n,family in [(1,'Pressure'),(2,'Decision'),(3,'ACS'),(4,'EquilibriumPressure')]
       for letter,group in [('A','Arrays'),('B','Frequencies')]]
ARRAY_STEM='Appendix_0_ArrayLayouts'
STEMS=[ARRAY_STEM,*COMPARISON_STEMS]

def assemble(figures=ROOT/'figures'):
    writer=PdfWriter()
    manifest=[]
    for stem in STEMS:
        png,pdf=figures/(stem+'.png'),figures/(stem+'.pdf')
        with Image.open(png) as im: im.verify()
        reader=PdfReader(pdf)
        assert len(reader.pages)==1,stem
        writer.append(reader,outline_item=stem.replace('_',' '))
        geometry_plate=stem==ARRAY_STEM
        comparison_index=None if geometry_plate else COMPARISON_STEMS.index(stem)
        count=9 if geometry_plate or comparison_index%2==0 else 6
        views=2*count if '_Decision_' in stem else count
        manifest.append(dict(figure=stem,case_panels=count,data_views=views,layout_rows=3 if count==9 else 2,
                             layout_columns=3,
                             task='Geometry' if geometry_plate else 'Triple' if comparison_index in (6,7) else 'Single',
                             figure_kind='array_geometry' if geometry_plate else 'comparison',
                             source_layout_insets=0,
                             png=png.name,pdf=pdf.name,svg=stem+'.svg'))
    comparisons=[row for row in manifest if row['figure_kind']=='comparison']
    assert len(manifest)==9 and len(comparisons)==8
    assert sum(x['case_panels'] for x in comparisons)==60
    assert sum(x['case_panels'] for x in manifest)==69
    assert sum(x['data_views'] for x in comparisons)==75
    assert sum(x['data_views'] for x in manifest)==84
    stream=BytesIO();writer.write(stream)
    path=figures/'RINENG_Appendix_Figures.pdf'
    write_bytes(path,stream.getvalue())
    write_bytes(figures/'appendix_manifest.json',(json.dumps(dict(figure_count=9,comparison_figure_count=8,
        case_panel_count=60,standalone_array_panel_count=9,total_panel_count=69,
        data_view_count=84,comparison_data_view_count=75,source_layout_inset_count=0,
        unique_settings=14,shared_setting='A06: square40kHz',figures=manifest),indent=2)+'\n').encode())
    return path

if __name__=='__main__':
    for arguments in [['render_array_layouts.py'],['render_pressure.py'],['render_decision.py'],['render_acs.py'],
                      ['render_triple_benchmark.py']]:
        subprocess.run([sys.executable,str(ROOT/arguments[0]),*arguments[1:]],check=True)
    print(assemble())
