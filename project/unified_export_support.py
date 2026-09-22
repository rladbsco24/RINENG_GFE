"""Retain the existing generality table, note and diagnostic export scope.

This module only reformats saved numerical records and copies existing files.
The generality runtime is imported in a subprocess so its module names cannot
replace the main study's runtime in a notebook kernel.
"""
from __future__ import annotations

import argparse

from rineng_content_id import content_identity
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ARCHIVED_TABLES = {
    'selected_alpha_table', 'selected_weights', 'alpha_candidate_scores',
    'table_directional_weights', 'table_opposed_stopping',
    'table_decision_windows', 'table_decision_basis_controls',
    'table_equilibrium_resource_baseline_summary',
    'table_equilibrium_resource_resource_summary',
    'table_equilibrium_resource_selected_weights',
}
DIAGNOSTIC_PDFS = (
    'RINENG_Decision_Window_Check.pdf',
    'RINENG_Equilibrium_Resource_Check.pdf',
)
NOTE_NAMES = (
    'RINENG_Appendix_Method_Tuning.md',
    'RINENG_Appendix_Review_Guide.md',
    'RINENG_Decision_Window_Revision.md',
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def _worker(historical_project, active_project, destination):
    """Call the original exporters while preserving historical selection scope."""
    historical_project = Path(historical_project).resolve()
    active_project = Path(active_project).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(historical_project))
    import pandas as pd
    import export_figure_package as exporter

    # Alpha/initialization candidates were selected in a deposited experiment.
    # Reformat those audited selections; do not claim that a fresh run retuned them.
    original_export_weights = exporter.export_weights
    exporter.export_weights = lambda _root, output: original_export_weights(historical_project, output)
    _, table_summary = exporter.export_tables(active_project, destination)
    table_summary['Evidence scope'] = table_summary.Table.map(
        lambda name: 'deposited selection or focused control' if name in ARCHIVED_TABLES
        else 'active figure data')
    correspondence = pd.read_csv(destination / 'tables/table_solution_correspondence.csv')
    endpoints = pd.read_csv(active_project / 'data/compact_acs/endpoints.csv')
    from final_benchmark_data import load_final_benchmark
    benchmark, outcomes, _, _ = load_final_benchmark(active_project)
    sources = {
        'table_solution_correspondence': f'{len(correspondence)} active Single field/command comparisons (G2–G5)',
        'table_convergence': f'{len(endpoints)} active Conventional endpoint records (G6–G7)',
        'table_equilibrium_benchmark': f'{len(benchmark)} active method/setting groups and {len(outcomes)} target records (G8–G9)',
    }
    for name, source in sources.items():
        table_summary.loc[table_summary.Table.eq(name), 'Source'] = source
    table_summary.to_csv(destination / 'tables/table_export_summary.csv', index=False)

    # Existing PDFs are staged at the path expected by the original exporter.
    # With both present, export_diagnostics cannot invoke either renderer.
    with tempfile.TemporaryDirectory(prefix='rineng_export_diagnostics_') as temporary:
        staged = Path(temporary)
        (staged / 'delivery').mkdir()
        (staged / 'data').symlink_to(historical_project / 'data', target_is_directory=True)
        for name in ('render_decision_window_check.py', 'render_equilibrium_resource_check.py'):
            (staged / name).symlink_to(historical_project / name)
        for name in DIAGNOSTIC_PDFS:
            candidates = [historical_project.parent / name, historical_project / 'delivery' / name]
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                raise FileNotFoundError(f'The deposited diagnostic PDF is missing: {name}')
            shutil.copyfile(source, staged / 'delivery' / name)
        diagnostics = exporter.export_diagnostics(staged, destination, regenerate=False)
    if len(diagnostics['pdfs']) != 2:
        raise RuntimeError('Expected the two existing generality diagnostic PDFs')
    diagnostic_manifest = json.loads((destination / 'diagnostic_export_manifest.json').read_text())
    diagnostic_manifest['path_base'] = 'export archive root'
    diagnostic_manifest['pdfs'] = ['diagnostics/G/' + Path(path).name for path in diagnostics['pdfs']]
    diagnostic_manifest['notes'] = [
        'diagnostics/G/' + Path(path).relative_to(destination / 'diagnostics').as_posix()
        for path in diagnostics['notes']]
    diagnostic_manifest['evidence_scope'] = 'deposited focused controls'
    _write_json(destination / 'diagnostic_export_manifest.json', diagnostic_manifest)

    notes_directory = destination / 'notes'
    notes_directory.mkdir(exist_ok=True)
    for name in NOTE_NAMES:
        source = next((path for path in [historical_project.parent / name, historical_project / name]
                       if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f'The deposited explanation is missing: {name}')
        shutil.copyfile(source, notes_directory / name)

    readme = (
        '# Generality export support\n\n'
        'The three figure tables (solution correspondence, convergence, equilibrium benchmark) '
        'use the active G2–G9 numerical data. Displacement tables retain the original micrometer '
        'audit columns and include meter columns. Each of the 13 existing tables is available as '
        'CSV, TSV, TXT and TeX; the table inventory states its evidence scope.\n\n'
        'Selected FE/GFE weights, candidate scores, directional-weight/stopping/window controls, '
        'resource checks, and the two diagnostic PDFs describe the deposited selection campaign '
        'and focused controls. They remain archived evidence in a fresh smoke or full run; '
        'they are not silently repeated or presented as new trials. Current commands, geometry, '
        'seeds and computation/cache status remain in the active numerical exports and parameter tables.\n\n'
        'The retained review/tuning notes use their original local appendix numbering: '
        'layout → G1; pressure arrays/frequencies → G2/G3; decision arrays/frequencies → G4/G5; '
        'ACS arrays/frequencies → G6/G7; equilibrium arrays/frequencies → G8/G9. '
        'The integrated discussion PDF supplies the unified numbering.\n\n'
        'These exports invoke no optimizer, force evaluation or equilibrium search. '
        'The source files under reproduction/G preserve the corresponding exporters and runtime. '
        'The single study notebook contains the complete runnable directory layout and compatible caches.\n'
    )
    (destination / 'README.md').write_text(readme, encoding='utf-8')
    _write_json(destination / 'support_verification.json', {
        'table_count': len(table_summary), 'table_formats': ['csv', 'tsv', 'txt', 'tex'],
        'selected_weight_rows': len(pd.read_csv(destination / 'tables/selected_weights.csv')),
        'candidate_score_rows': len(pd.read_csv(destination / 'tables/alpha_candidate_scores.csv')),
        'diagnostic_pdf_count': len(diagnostics['pdfs']),
        'optimizer_calls': 0, 'force_evaluations': 0, 'equilibrium_searches': 0,
        'active_project_differs_from_deposit': active_project != historical_project,
        'active_correspondence_rows': len(correspondence),
        'active_endpoint_rows': len(endpoints), 'active_target_records': len(outcomes),
    })


def prepare_support(root, general_project, output_root, mode):
    """Return portable file mappings for the existing generality export scope."""
    root = Path(root).resolve()
    project = Path(general_project).resolve()
    output_root = Path(output_root).resolve()
    if mode not in {'smoke', 'full'}:
        raise ValueError('mode must be smoke or full')
    historical = root / 'generality/revision_f1_f2'
    support = output_root / 'support/G'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONHASHSEED='0')
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                    '--historical-project', str(historical), '--active-project', str(project),
                    '--destination', str(support)], check=True, cwd=historical, env=env)
    mappings = []
    # The original combined TXT is identical at its top-level and tables paths.
    # Retain one copy, under tables/, along with every per-table representation.
    for path in sorted(support.rglob('*')):
        if not path.is_file() or path == support / 'all_table_exports.txt':
            continue
        relative = path.relative_to(support)
        archived = (path.stem in ARCHIVED_TABLES or relative.parts[0] in {'notes', 'diagnostics'}
                    or path.name in DIAGNOSTIC_PDFS or path.name.startswith('selected_weights'))
        role = 'archived_control' if archived else 'table_or_export_metadata'
        if path.name == 'all_table_exports.txt':
            role = 'combined_active_and_archived_tables'
        if path.name in DIAGNOSTIC_PDFS:
            archive = 'diagnostics/G/' + path.name
        elif relative.parts[0] == 'diagnostics':
            archive = 'diagnostics/G/' + Path(*relative.parts[1:]).as_posix()
        elif relative.parts[0] == 'notes':
            archive = 'notes/G/' + path.name
        elif relative.parts[0] == 'tables':
            archive = 'tables/G/' + path.name
        else:
            archive = 'tables/G/' + path.name
        mappings.append({'path': str(path), 'archive_path': archive, 'role': role, 'section': 'G'})
    for source_root in (historical, historical.parent / 'generality_package'):
        for path in sorted(source_root.rglob('*.py')):
            relative = path.relative_to(source_root)
            if any(part in {'__pycache__', 'data', 'outputs', 'cache', 'qa'} for part in relative.parts):
                continue
            archive = 'reproduction/G/' + source_root.name + '/' + relative.as_posix()
            mappings.append({'path': str(path), 'archive_path': archive,
                             'role': 'reproduction_source', 'section': 'G'})
    requirements = historical.parent / 'requirements.txt'
    if requirements.is_file():
        mappings.append({'path': str(requirements), 'archive_path': 'reproduction/G/requirements.txt',
                         'role': 'reproduction_environment', 'section': 'G'})
    for record in mappings:
        content = Path(record['path']).read_bytes()
        record.update(bytes=len(content), content_id=content_identity(content).hexdigest())
    if len({record['archive_path'] for record in mappings}) != len(mappings):
        raise RuntimeError('Supplemental export paths must be unique')
    metadata = json.loads((support / 'support_verification.json').read_text())
    metadata.update(mode=mode, file_count=len(mappings), support_directory=str(support))
    return {'files': mappings, 'metadata': metadata}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--historical-project', type=Path, required=True)
    parser.add_argument('--active-project', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    _worker(args.historical_project, args.active_project, args.destination)
