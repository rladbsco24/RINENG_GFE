"""Assemble all 31 scientific figures with a data-linked discussion per figure.

PDF plates are embedded as vector pages. Reporting never starts an optimizer.
Smoke observations are deliberately excluded when reporting a new full campaign.
"""
from __future__ import annotations

from pathlib import Path
from io import BytesIO
from types import SimpleNamespace
import csv

from rineng_content_id import content_identity
import json
import sys
from xml.sax.saxutils import escape

MAIN_IDS = tuple([str(i) for i in range(1, 9)] +
                 [f'{s}{i}' for s, count in [('A', 2), ('B', 3), ('C', 3),
                                           ('D', 3), ('E', 2), ('F', 1)]
                 for i in range(1, count + 1)])
GENERAL_STEMS = ['Appendix_0_ArrayLayouts'] + [
    f'Appendix_{n}{letter}_{family}_{group}'
    for n, family in [(1, 'Pressure'), (2, 'Decision'), (3, 'ACS'), (4, 'EquilibriumPressure')]
    for letter, group in [('A', 'Arrays'), ('B', 'Frequencies')]]
SECTION_NAMES = {'Main': 'Main results', 'A': 'Surrogate and elastic mechanics',
                 'B': 'Twin, Bottle and Vortex fields', 'C': 'Sensitivity',
                 'D': 'Discrete-phase deployment', 'E': 'Prescription dependence',
                 'F': 'Optimizer robustness', 'G': 'Across arrays and frequencies'}

# These explanations do not assume that a fresh full run repeats smoke outcomes.
FULL_DISCUSSION = {
'1': 'Compare convergence over the same starts and target domain. Field correspondence and numerical stationarity answer different questions: similar pressure patterns can coexist with very different terminal gradients. Compare normalized histories within each formulation and measured command times across methods.',
'2': 'The fixed GFE endpoints and common embedding make the changing Conventional states comparable across budgets. Assess approach using the full-dimensional command distances and the displayed projection together. A fixed reference set avoids alignment being created by refitting the embedding.',
'3': 'The local section distinguishes the shape of the objective near the selected command from the appearance of its acoustic field. Read the central cut, native terminal gradient and quadratic-fit residual together. Objective heights retain their own dimensional scales; spatial mechanics are validated separately.',
'4': 'The Triple task tests whether local objective geometry remains informative when traps compete for shared array resources. Compare the matched pressure fields and fixed decision directions. Sampled descent arrows indicate local directions and are not integrated trajectories or measured basin boundaries.',
'5': 'Pressure concentration and independently validated stiffness are complementary. Read each method together with displacement in Figure 8: maximizing concentration or stiffness alone does not guarantee that the loaded equilibrium lies at the requested coordinate.',
'6': 'GFE prescribes effective-weight balance while FE provides the direct uncompensated control. Compare actual equilibrium markers and total-force structure on the same pressure backgrounds. The finite-ka elastic model remains outside the synthesis optimizer and supplies the common mechanical validation.',
'7': 'Increasing the force coefficient changes the relative priorities of the objective and can change selected endpoint morphology. Check the actual stopping records before interpreting a displayed terminal field as stationary. These examples complement the systematic sensitivity controls in Appendix C.',
'8': 'Read command time, physical equilibrium displacement and equilibrium pressure as separate engineering quantities. GFE and FE isolate gravity compensation, while IB, GS and AD quantify the speed and mechanical precision of the declared pressure-reconstruction baselines. Use the current exported timing scope and population intervals.',
'A1': 'Identical frozen commands isolate the difference between force models. Similar directions support the use of Gor’kov as an inexpensive synthesis surrogate, while root offsets and force magnitude differences motivate independent finite-ka elastic validation. Equal-length arrows encode direction only.',
'A2': 'Assess angular agreement, force magnitude and equilibrium separation together. Good directional agreement need not imply equal force magnitude or exactly matching loaded equilibria. Resolve pressure-minimum subsets and restoring-root status from the current exported tables.',
'B1': 'The standard-sign comparison tests whether the declared Bottle weights produce an enclosed low-pressure region. Read the matched pressure sections and axial channel directly; a low target pressure alone is insufficient to identify a Bottle field.',
'B2': 'Compare the same endpoint morphology across FE or RH and Conventional under the declared alternative sign. Pressure correspondence is checked on fixed field grids and should be read together with the actual terminal iteration and stopping records.',
'B3': 'Compare Vortex, Twin and Bottle using the independent elastic total-force model. Root displacement and three-dimensional stiffness distinguish the loaded mechanical equilibrium from pressure morphology. The same command is used for all panels within each row.',
'C1': 'Sensitivity is assessed against a fixed, validated reference rather than a different reference at every parameter value. Read field correspondence, physical displacement and convergence together; a similar field can persist even when stationarity or placement degrades.',
'C2': 'Separate the physical pressure and force regularization terms from numerical smoothing constants and STD aggregation. The ablations identify which contribution changes trap balance or convergence. Paired starts and fixed references preserve a common comparison across parameter choices.',
'C3': 'The RH beta sweep tests both how pressure weighting changes convergence and whether it changes the selected endpoint class. Read the field correspondence and physical consequences together across the declared beta ratios, including the zero-weight control.',
'D1': 'Discrete deployment introduces an implementation constraint after continuous synthesis. Compare phase-transition rules using achieved objective and command cost at each resolution, with the same continuous source commands and the same declared hardware alphabet.',
'D2': 'Independent elastic validation separates added quantization error from the equilibrium error already present in the continuous reference. Increasing phase resolution can reduce discretization error while leaving the baseline surrogate-to-elastic difference.',
'D3': 'The spatial illustration tests whether mechanics-aware rounding changes the actual loaded equilibrium in the intended direction. Compare matching pressure contours and displacement markers, using the same elastic validation model before and after the correction.',
'E1': 'The prescription comparison asks whether a fixed geometric rule transfers across opposed-array separation. Read the changes in the preferred radius relative to the corresponding optimized reference fields; the recorded search isolates prescription choice from physical validation.',
'E2': 'Refinement quantifies the amount of empirical adjustment needed for a prescribed field to resemble an optimized reference. Compare the field panels and achieved correspondence using the same setting and target. The result explains the calibration burden of a fixed prescription.',
'F1': 'Holding the objective fixed while changing the optimizer isolates whether the main performance contrast depends specifically on BFGS. Compare stationarity, command time, field correspondence and independently validated displacement over paired starts. Sensitivity of objective coefficients is reported separately in Appendix C.',
'G1': 'The geometry plate defines the sources and coordinate conventions for the following comparisons. Distinguish changes in array layout from changes in carrier frequency or available source count using the exported settings.',
'G2': 'Compare matched pressure contours under each array geometry. The fixed field grid and common within-panel normalization make method correspondence interpretable. Command similarity supplies a separate check because similar fields can admit different phase commands.',
'G3': 'Within each carrier, compare methods with the same frequency, array and spatial normalization. Wavelength-scaled field windows describe correspondence without requiring one identical dimensional phase command or one universal objective coefficient across all carriers.',
'G4': 'Compare the paired local sections using identical coordinate ranges within each array case. Wider views reveal surrounding structure while retaining the frozen command and native basis. A finite-distance decision section describes the sampled plane and does not measure a full-dimensional basin volume.',
'G5': 'The frequency comparison evaluates how the sampled objective geometry changes with propagation and the recorded coefficients. Read the central structure together with the displayed finite-distance window and native termination status.',
'G6': 'Conventional trajectories are compared with a fixed FE/GFE endpoint bank from the same formulation. This separates search from evaluation of correspondence. Interpret phase similarity together with field overlap, especially where different commands produce nearly identical fields.',
'G7': 'Each carrier uses its own fixed reference bank. The trajectory comparison tests approach to the sampled endpoint set across frequency, with the shared Square 40 kHz configuration counted only once in pooled summaries.',
'G8': 'The Triple task exposes competition for shared array resources. Compare physical placement and pressure at independently resolved elastic equilibria, preserving target ranges and recording unresolved roots separately. Source budgets and command identities accompany the mechanical comparison.',
'G9': 'Read carrier-dependent precision with the current source counts and root-search outcomes. FE, GFE and pressure-reconstruction methods can trade placement error against equilibrium pressure. Expanded-search roots remain part of the declared validation record rather than being silently excluded.'
}


FULL_READING = {
'2': 'The branch panels compare Conventional states at the plotted budgets against fixed GFE landmarks in one common embedding. Companion diagnostics retain full-dimensional correspondence outside the projection.',
'5': 'The Single and Triple panels compare all six methods using field concentration and independent elastic stiffness. Read the current aggregation and population record alongside the method markers; GFE is primary and FE is its uncompensated control.',
'8': 'The top row reports Single equilibrium displacement and pressure; the bottom row reports Triple results. The pooled command-time mapping is shared. Read target ranges and population intervals using the current exported aggregation record.',
'D2': 'The left panel measures displacement from each continuous command’s elastic equilibrium. The right panel measures absolute distance from the requested target. The dashed reference is the continuous-FE median; distributions retain the current target and command population.',
'E2': 'The pressure panels compare the historical FE reference with GS under the declared baseline and refined radii. The companion panel overlays radial pressure profiles normalized by their first annular maximum; read the radii from the current figure and exports.',
'G6': 'At each accepted iteration, Conventional is compared with the fixed FE/GFE endpoint bank from the same formulation. Similarity is the maximum modulus of the mean relative phasor over that bank, summarized across the recorded starts. The final value is held only after a trajectory ends.',
'G7': 'The same fixed-bank phase similarity is compared across the six Square carriers. Every trajectory is scored against its own carrier-specific FE/GFE bank, which remains fixed throughout the search.',
'G8': 'The array panels compare physical displacement and pressure at independently resolved elastic equilibria for each method. Read the current target and seed aggregation, source budgets and displayed intervals from the exported full-run records.',
'G9': 'The carrier panels repeat the five-method Triple comparison using the current source budgets and validated roots. Target ranges and seed populations are recorded separately in the current exports; recovered roots remain included in the declared aggregation.'
}


def _safe_text(value):
    return str(value).replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', ' - ')


def _content_id(path):
    return content_identity(Path(path).read_bytes()).hexdigest()


def _figure_paths(root, mode, output_root, general_project):
    for entry in (root, root / 'runtime/src'):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    from hat_revision_pipeline.reviewer_focused import ordered_figures
    ctx = SimpleNamespace(config=SimpleNamespace(full=mode == 'full'), output_root=output_root,
                          memo={'notebook_root': root, 'notebook_mode': mode})
    paths = ordered_figures(ctx)
    if tuple(identifier for identifier, _ in paths) != MAIN_IDS:
        raise ValueError('The main/appendix figure sequence does not match the accepted 22-figure order')
    paths += [(f'G{i}', general_project / 'figures' / (stem + '.pdf'))
              for i, stem in enumerate(GENERAL_STEMS, 1)]
    return paths


def _source_path(root, output_root, general_project, source, mode):
    if source.startswith('generality/revision_f1_f2/'):
        relative = source.removeprefix('generality/revision_f1_f2/')
        # G8/G9 are rendered through final_benchmark_data.load_final_benchmark,
        # which follows this selector after a generated smoke/full campaign.
        # Resolve their report evidence through the same selector so a current
        # figure can never be paired with the deposited historical benchmark.
        if relative in ('data/triple_benchmark/summary.csv',
                        'data/triple_benchmark/outcomes.csv'):
            marker_path = general_project / 'data/generated_triple_source.json'
            if marker_path.is_file():
                marker = json.loads(marker_path.read_text())
                if marker.get('schema') != 'selected-native-triple-generation-v1':
                    raise ValueError('Unknown generated Triple selector schema')
                if (marker.get('mode') != mode
                        or marker.get('evidence_mode') != f'generated_{mode}'):
                    raise ValueError('Generated Triple report source does not match report mode')
                selected = (general_project / marker['relative_data_root']).resolve()
                if not selected.is_relative_to(general_project.resolve()):
                    raise ValueError('Generated Triple report source escapes the active project')
                return selected / Path(relative).name
        return general_project / relative
    if source.startswith('generality/'):
        return general_project.parent / source.removeprefix('generality/')
    if mode == 'full':
        if source.startswith('smoke_outputs/'):
            return output_root / source.removeprefix('smoke_outputs/')
        if source.startswith('appendix_outputs/'):
            return output_root / 'appendices' / source.removeprefix('appendix_outputs/')
        if source.startswith('appendix_B_pressure/'):
            relative = source.removeprefix('appendix_B_pressure/')
            production_aliases = {
                'data/solver_results.csv': 'B1_pressure_summary.csv',
                'B2_field_summary.csv': 'B2_field_summary_all_seeds.csv',
            }
            return output_root / 'appendices/B_pressure' / production_aliases.get(relative, relative)
    return root / source


def _current_evidence(entry, root, output_root, general_project, mode="full"):
    """Describe actual current-mode outputs without importing static observations."""
    evidence = []
    available = []
    sources = list(entry['sources'])
    supplemental = []
    generated_marker = general_project / 'data/generated_triple_source.json'
    if entry['id'] in ('G8', 'G9') and generated_marker.is_file():
        # The active selector replaces, rather than supplements, the deposited
        # benchmark/resource-release tables for a newly generated campaign.
        sources = [source for source in sources if source.endswith(
            ('data/triple_benchmark/summary.csv', 'data/triple_benchmark/outcomes.csv'))]
        supplemental = [generated_marker, general_project / 'final_benchmark_data.py']
    for source in sources:
        path = _source_path(root, output_root, general_project, source, mode)
        if not path.exists():
            continue
        available.append(path)
        if path.suffix == '.csv':
            with path.open(newline='', encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            evidence.append(f'{path.name}: {len(rows):,} current-run records.')
        elif path.suffix == '.json' and 'caption' in path.name:
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                matching = [text for key, text in value.items()
                            if f"{entry['id']}_" in key or key == entry['id']]
                evidence += [_safe_text(item) for item in matching if isinstance(item, str)]
    available.extend(path for path in supplemental if path.exists() and path not in available)
    if not evidence:
        evidence = ['Read the current figure together with its exported command, metric and stopping records.']
    return evidence[:3], available


def write_report(root, mode='smoke', main_output_root=None, general_project=None, destination=None,
                 current_run_evidence=False):
    """Write the ordered 31-figure discussion report and return its absolute path.

    ``general_project`` is the active ``revision_f1_f2`` directory. In full mode
    pass the generated full project, never the deposited smoke project.
    Set ``current_run_evidence=True`` after a fresh smoke computation to report
    its actual outputs and omit the deposited smoke observations and counts.
    """
    from matplotlib.font_manager import FontProperties, findfont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, Spacer, Frame
    from pypdf import PdfReader, PdfWriter, Transformation

    if mode not in ('smoke', 'full'):
        raise ValueError('mode must be smoke or full')
    use_saved_observations = mode == 'smoke' and not current_run_evidence
    root = Path(root).resolve()
    output_root = Path(main_output_root).resolve() if main_output_root else root / f'{mode}_outputs'
    general_project = Path(general_project).resolve() if general_project else (
        root / 'generality/revision_f1_f2' if mode == 'smoke' else output_root / 'generality/revision_f1_f2')
    destination = Path(destination).resolve() if destination else root / (
        'RINENG_All_Smokes_with_Discussion.pdf' if mode == 'smoke' else 'RINENG_All_Full_Figures_with_Discussion.pdf')
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / 'unified_figure_discussions.json').read_text())
    entries = data['figures']
    paths = _figure_paths(root, mode, output_root, general_project)
    if len(entries) != 31 or [entry['id'] for entry in entries] != [key for key, _ in paths]:
        raise ValueError('Exactly 31 ordered discussion entries and figures are required')
    for identifier, path in paths:
        if not path.is_file() or len(PdfReader(path).pages) != 1:
            raise ValueError(f'Figure {identifier} is missing or is not a single-page vector PDF: {path}')

    for name, weight in [('UnifiedText', 'normal'), ('UnifiedBold', 'bold')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, findfont(FontProperties(family='DejaVu Sans', weight=weight))))
    pdfmetrics.registerFontFamily('UnifiedText', normal='UnifiedText', bold='UnifiedBold')
    ink, teal, muted = map(colors.HexColor, ['#24343E', '#087F72', '#63737C'])
    body = ParagraphStyle('Body', fontName='UnifiedText', fontSize=10.15, leading=14.5,
                          textColor=ink, spaceAfter=9)
    small = ParagraphStyle('Small', parent=body, fontSize=8.0, leading=11.5, textColor=muted, spaceAfter=6)
    heading = ParagraphStyle('Heading', parent=body, fontName='UnifiedBold', fontSize=19,
                             leading=23, textColor=ink, spaceAfter=12)
    subheading = ParagraphStyle('Subheading', parent=body, fontName='UnifiedBold', fontSize=10.2,
                                leading=14, textColor=teal, spaceBefore=4, spaceAfter=5)
    question = ParagraphStyle('Question', parent=body, fontName='UnifiedBold', fontSize=11.2,
                              leading=15.5, textColor=teal, spaceAfter=10)
    contents_style = ParagraphStyle('Contents', parent=body, fontSize=9.35, leading=12.8, spaceAfter=5)
    writer = PdfWriter()
    manifest = []

    def paragraph(text, style=body):
        return Paragraph(escape(_safe_text(text)).replace('\n', '<br/>'), style)

    def furniture(pdf, width, height, page_number, label, kind='Discussion'):
        pdf.setFillColor(teal)
        pdf.setFont('UnifiedBold', 8.5)
        pdf.drawString(42, height - 30, 'RINENG  |  GFE objective')
        pdf.setFillColor(muted)
        pdf.setFont('UnifiedText', 8.3)
        pdf.drawRightString(width - 42, height - 30, f'{mode.upper()}  |  {kind}')
        pdf.setStrokeColor(colors.HexColor('#CCD8DA'))
        pdf.setLineWidth(.6)
        pdf.line(42, height - 40, width - 42, height - 40)
        pdf.line(42, 37, width - 42, 37)
        pdf.setFont('UnifiedText', 8)
        pdf.drawString(42, 23, label)
        pdf.drawRightString(width - 42, 23, str(page_number))

    def text_page(story, page_number, label):
        stream = BytesIO()
        pdf = canvas.Canvas(stream, pagesize=A4)
        furniture(pdf, *A4, page_number, label)
        frame = Frame(42, 50, A4[0] - 84, A4[1] - 110,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        remaining = list(story)
        frame.addFromList(remaining, pdf)
        if remaining:
            raise RuntimeError(f'Discussion overflows page {page_number}: {label}')
        pdf.showPage()
        pdf.save()
        writer.add_page(PdfReader(BytesIO(stream.getvalue())).pages[0])

    # Two deliberately compact contents pages keep the complete order visible.
    for contents_index, selected in enumerate((entries[:16], entries[16:])):
        page_number = contents_index + 1
        if contents_index == 0:
            story = [paragraph('Complete figures and discussion', heading),
                     paragraph('8 main figures + 23 appendix figures', question),
                     paragraph('The complete figure set is presented in manuscript order. Each figure has a discussion page followed by an uncropped vector plate. Sensitivity (Appendix C) and optimizer robustness (Appendix F) remain separate analyses.'),
                     paragraph('This edition reports the saved smoke campaigns. Numerical observations and sample scopes refer to those exported results; cached replay is not included in solver timing.' if use_saved_observations else
                               f'This edition reports the current {mode} campaign. The reading notes retain the scientific questions, while recorded evidence is drawn from the generated {mode} outputs. Deposited smoke numerical observations are omitted.'),
                     Spacer(1, 5)]
        else:
            story = [paragraph('Contents continued', heading),
                     paragraph('Figures retain their accepted identifiers. The new across-array and across-frequency material is collected as Appendix G.', body), Spacer(1, 10)]
        last_section = None
        for item in selected:
            idx = entries.index(item)
            section = 'Main' if item['id'].isdigit() else item['id'][0]
            if section != last_section:
                story.append(paragraph(SECTION_NAMES[section], subheading))
                last_section = section
            discussion_page, figure_page = 3 + 2 * idx, 4 + 2 * idx
            story.append(paragraph(f"Figure {item['id']}  {item['title']}  |  {discussion_page}-{figure_page}", contents_style))
        text_page(story, page_number, 'Complete figure sequence')
    writer.add_outline_item('Contents', 0)
    section_bookmarks = {}
    for idx, (entry, (identifier, figure_path)) in enumerate(zip(entries, paths)):
        section = 'Main' if identifier.isdigit() else identifier[0]
        discussion_page = 3 + idx * 2
        plate_page = discussion_page + 1
        story = [paragraph(f'Figure {identifier}', subheading), paragraph(entry['title'], heading),
                 paragraph(entry['question'], question), paragraph('Reading the figure', subheading),
                 paragraph(entry['reading'] if use_saved_observations else FULL_READING.get(identifier, entry['reading']))]
        resolved_data_sources = list(entry['sources'])
        if use_saved_observations:
            story += [paragraph('Discussion', subheading)]
            story += [paragraph(part) for part in entry['discussion'].split('\n\n')]
            story += [paragraph('Recorded results', subheading)]
            story += [paragraph('- ' + item) for item in entry['key_results']]
            story += [paragraph('Sample and interpretation', subheading), paragraph(entry['scope'], small)]
        else:
            story += [paragraph('Discussion', subheading), paragraph(FULL_DISCUSSION[identifier])]
            evidence, available = _current_evidence(entry, root, output_root, general_project, mode=mode)
            resolved_data_sources = []
            for path in available:
                if path.is_relative_to(root):
                    resolved_data_sources.append(str(path.relative_to(root)))
                elif path.is_relative_to(general_project):
                    resolved_data_sources.append(str(
                        Path('generality/revision_f1_f2') / path.relative_to(general_project)))
                else:
                    resolved_data_sources.append(str(path))
            story += [paragraph(f'Current {mode}-run evidence', subheading)]
            story += [paragraph('- ' + item) for item in evidence]
        source_labels = [Path(source).name for source in resolved_data_sources[:3]]
        story += [paragraph('Linked data: ' + '; '.join(source_labels) + '.', small)]
        text_page(story, discussion_page, f'Figure {identifier}  |  {SECTION_NAMES[section]}')
        if section not in section_bookmarks:
            section_bookmarks[section] = writer.add_outline_item(SECTION_NAMES[section], discussion_page - 1)
        item_bookmark = writer.add_outline_item(f"Figure {identifier}: {entry['title']}", discussion_page - 1,
                                               parent=section_bookmarks[section])

        # Retain native dimensions for large plates, particularly tall G4/G5.
        source = PdfReader(figure_path).pages[0]
        sw, sh = float(source.mediabox.width), float(source.mediabox.height)
        width = max(A4[0], sw + 48)
        height = max(A4[1], sh + 128)
        scale = min((width - 48) / sw, (height - 126) / sh)
        tx, ty = (width - sw * scale) / 2, 54 + (height - 126 - sh * scale) / 2
        stream = BytesIO()
        pdf = canvas.Canvas(stream, pagesize=(width, height))
        furniture(pdf, width, height, plate_page, f'Figure {identifier}  |  {SECTION_NAMES[section]}', 'Figure plate')
        pdf.setFont('UnifiedBold', 10.7)
        pdf.setFillColor(ink)
        pdf.drawString(42, height - 58, f"Figure {identifier}  |  {SECTION_NAMES[section]}")
        pdf.showPage()
        pdf.save()
        plate = PdfReader(BytesIO(stream.getvalue())).pages[0]
        plate.merge_transformed_page(source, Transformation().scale(scale).translate(tx, ty))
        writer.add_page(plate)
        writer.add_outline_item('Vector figure plate', plate_page - 1, parent=item_bookmark)
        manifest.append({'figure': identifier, 'title': entry['title'], 'discussion_page': discussion_page,
                         'figure_page': plate_page, 'source_pdf': str(figure_path.relative_to(root))
                         if figure_path.is_relative_to(root) else str(Path('generality/revision_f1_f2') / figure_path.relative_to(general_project)), 'source_content_id': _content_id(figure_path),
                         'source_dimensions_pt': [sw, sh], 'page_dimensions_pt': [width, height],
                         'vector_scale': scale, 'data_sources': resolved_data_sources,
                         'evidence_mode': mode, 'smoke_numerical_claims_included': use_saved_observations,
                         'current_run_evidence': not use_saved_observations})
    writer.add_metadata({'/Title': 'RINENG - Complete figures and discussion',
                         '/Author': 'RINENG GFE study',
                         '/Subject': f'31 scientific figures with discussion; {mode} results',
                         '/Keywords': 'GFE, FE, Conventional, Gor’kov, elastic mechanics, sensitivity, optimizer robustness'})
    with destination.open('wb') as stream:
        writer.write(stream)
    if len(PdfReader(destination).pages) != 64:
        raise RuntimeError('The final discussion PDF must contain exactly 64 pages')
    manifest_path = destination.with_suffix('.figure_manifest.json')
    manifest_path.write_text(json.dumps({'schema': 'rineng-unified-report-v1', 'mode': mode,
                                        'scientific_figures': 31, 'pages': 64,
                                        'current_run_evidence': not use_saved_observations,
                                        'separate_diagnostics': ['decision-window grid check', 'equilibrium resource check'],
                                        'figures': manifest}, indent=2) + '\n')
    return str(destination)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--main-output-root', type=Path)
    parser.add_argument('--general-project', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--current-run-evidence', action='store_true',
                        help='Use freshly computed smoke evidence rather than deposited observations')
    args = parser.parse_args()
    print(write_report(args.root, args.mode, args.main_output_root, args.general_project, args.output,
                       current_run_evidence=args.current_run_evidence))
