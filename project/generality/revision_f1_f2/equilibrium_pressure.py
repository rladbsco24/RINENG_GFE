"""Main Figure 8 pressure and displacement for the frozen Single commands.

Evaluate the native incident peak-pressure phasor at each method's recorded
finite-ka equilibrium. Alpha choices and equilibrium roots are reused exactly;
this script runs no optimization, root search, or dynamics simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from appendix_io import write_bytes, save_figure
from appendix_settings import ROOT, SETTINGS, ARRAY_IDS, FREQUENCY_IDS

sys.path.insert(0, str(ROOT.parent / 'generality_package/runtime/src'))
from hat_revision_pipeline.fig8_methods_only import (
    _metric_limits, METHOD_MARKERS, METHOD_MARKER_SIZES, HOLLOW_METHODS,
)
from hat_revision_pipeline.style import configure_style, COLORS as NATIVE_COLORS

METHODS = ('Conventional', 'FE', 'GFE')
LINEAR_METHODS = ('IB', 'GS')
MAIN_METHOD = {'Conventional': 'Conventional', 'FE': 'FE', 'GFE': 'FE+g', 'IB': 'IB', 'GS': 'GS'}
COLORS = {method: NATIVE_COLORS[native] for method, native in MAIN_METHOD.items()}
LABELS = {'A06': 'Square', 'S01': 'Rectangular', 'S02': 'Opposed',
          'S03': 'Spherical cap', 'S04': 'Fermat disk', 'S05': 'Arbitrary 3D',
          'A07': 'Annular', 'A08': 'Four clusters', 'A09': 'Tilted plane'}
DATA = ROOT / 'data/equilibrium_pressure'


def json_out(path, value):
    write_bytes(Path(path), (json.dumps(value, indent=2) + '\n').encode())


def figure_style_protocol():
    return dict(
        source='generality_package/runtime/src/hat_revision_pipeline/fig8_methods_only.py',
        panel='Single equilibrium pressure versus displacement, using native main Figure 8 symbols',
        horizontal_metric='displacement_um',
        vertical_metric='pressure_abs_at_exact_equilibrium_pa',
        coordinate_mapping='Direct physical coordinates; a marked pressure-axis break separates isolated outliers',
        metric_limits='_metric_limits(include_zero=True), imported directly and unmodified',
        markers={m: METHOD_MARKERS[MAIN_METHOD[m]] for m in (*METHODS, *LINEAR_METHODS)},
        marker_areas_pt2={m: METHOD_MARKER_SIZES[MAIN_METHOD[m]] for m in (*METHODS, *LINEAR_METHODS)},
        marker_edges='Native white edges for C/FE/GFE; native hollow IB/GS triangles',
        method_alias='GFE uses the existing main FE+g color and filled-plus marker',
        layout='One Single panel per setting; array 3x3 and frequency 3x2; shared method legend',
        scales='Linear displacement; pressure is linear within each explicitly separated range',
        pressure_break_rule='Largest pressure at least 10 times the second largest; no method-specific rule',
        pressure_break_display='Both linear ranges keep physical Pa ticks; no points are omitted')


def pressure_axis(ax, pressure):
    """Keep small pressures legible without dropping a measured outlier."""
    values = np.asarray(pressure, dtype=float)
    ordered = np.sort(values)
    second, largest = float(ordered[-2]), float(ordered[-1])
    if second <= 0. or largest < 10. * second:
        ax.set_ylim(*_metric_limits(values, include_zero=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        return dict(scale='linear', omitted_interval_pa=None)

    locator = MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10])
    lower_top = float(locator.tick_values(0., 1.25 * second)[-1])
    upper_ticks = locator.tick_values(.90 * largest, 1.10 * largest)
    upper_bottom, upper_top = float(upper_ticks[0]), float(upper_ticks[-1])
    if lower_top >= upper_bottom:
        raise ValueError('Pressure ranges for an axis break must be disjoint')
    lower_bottom = -.09 * lower_top
    physical = np.array([lower_bottom, lower_top, upper_bottom, upper_top])
    display = np.array([0., .65, .80, 1.])

    def transform(values, origin, destination):
        values = np.asarray(values, dtype=float)
        result = np.interp(values, origin, destination)
        result = np.where(values < origin[0], destination[0] +
            (values-origin[0]) * (destination[1]-destination[0]) / (origin[1]-origin[0]), result)
        return np.where(values > origin[-1], destination[-1] +
            (values-origin[-1]) * (destination[-1]-destination[-2]) / (origin[-1]-origin[-2]), result)

    ax.set_yscale('function', functions=(
        lambda values: transform(values, physical, display),
        lambda values: transform(values, display, physical)))
    ax.set_ylim(lower_bottom, upper_top)
    ticks = [0., .5 * lower_top, lower_top, upper_bottom, upper_top]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:.3g}'))
    # The open spine and paired slashes mark the interval containing no data.
    ax.spines['left'].set_visible(False)
    for start, stop in [(0., .65), (.80, 1.)]:
        ax.plot([0., 0.], [start, stop], transform=ax.transAxes,
                color='black', lw=.85, clip_on=False, zorder=7)
    for y in [.675, .775]:
        ax.plot([-.017, .017], [y-.010, y+.010], transform=ax.transAxes,
                color='black', lw=1.2, clip_on=False, zorder=8)
    if np.any((values > lower_top) & (values < upper_bottom)):
        raise ValueError('A measured pressure lies inside the omitted interval')
    return dict(scale='broken-linear', omitted_interval_pa=[lower_top, upper_bottom],
        linear_ranges_pa=[[lower_bottom, lower_top], [upper_bottom, upper_top]],
        display_ranges=[[0., .65], [.80, 1.]], ticks_pa=ticks,
        largest_to_second_largest_ratio=largest/second)


def extract(destination=DATA):
    sys.path.insert(0, str(ROOT.parent / 'generality_package/runtime/src'))
    from hat_revision_pipeline.exact_validator import ArbitraryArrayPressureField
    from hat_revision_pipeline.cache import digest_array

    rows, checks = [], []
    for setting in SETTINGS:
        sid = setting['id']
        source = ROOT / 'campaign' / sid
        chosen = json.loads((ROOT / 'data/alpha_tuning' / sid / 'selected.json').read_text())
        baseline = pd.read_csv(source / 'tables/fig5_exact_force_field_concentration.csv')
        with np.load(source / 'array_geometry.npz') as geometry:
            positions, normals = geometry['positions_m'], geometry['normals']
        for method in METHODS:
            if method == 'Conventional':
                entries = baseline[baseline.task.eq('Single') & baseline.method.eq(method)]
                if len(entries) != 1:
                    raise ValueError(f'{sid}: expected one prescribed Conventional endpoint')
                record = entries.iloc[0].to_dict()
                command = source / 'commands/Single_Conventional.npz'
                alpha = np.nan
            else:
                record = chosen['methods'][method]
                command = ROOT / record['command_path']
                alpha = float(record['alpha_per_m'])
            with np.load(command, allow_pickle=False) as saved:
                phase, target = saved['phase_rad'].copy(), saved['target_m'].copy()
                recorded_history_time = (float(saved['history_wall_s'][-1])
                                         if 'history_wall_s' in saved else None)
            phase_hash = digest_array(phase)
            if phase_hash != record['phase_fingerprint']:
                raise ValueError(f'{sid} {method}: equilibrium/command phase mismatch')
            if not record['finite_ka_root_found']:
                raise ValueError(f'{sid} {method}: unresolved equilibrium')
            command_time = float(record['command_time_s'])
            if not np.isfinite(command_time) or command_time < 0:
                raise ValueError(f'{sid} {method}: invalid deposited command time')
            if method != 'Conventional' and command_time != recorded_history_time:
                raise ValueError(f'{sid} {method}: selected command time differs from frozen solver history')
            equilibrium = np.array([record[f'finite_ka_equilibrium_{a}_m'] for a in 'xyz'])
            if not np.isfinite(equilibrium).all():
                raise ValueError(f'{sid} {method}: nonfinite equilibrium')
            if 'equilibrium_m' in record and not np.array_equal(equilibrium, record['equilibrium_m']):
                raise ValueError(f'{sid} {method}: inconsistent recorded equilibrium')
            metadata = json.loads(record['finite_ka_model_metadata_json'])
            field_info = metadata['field']
            if float(field_info['frequency_hz']) != setting['frequency_hz']:
                raise ValueError(f'{sid}: recorded carrier differs from setting')
            field = ArbitraryArrayPressureField(
                setting['frequency_hz'], positions, np.exp(1j * phase), normals=normals,
                sound_speed_m_s=field_info['sound_speed_m_s'],
                source_strength_pa_m=field_info['source_strength_pa_m_peak'],
                directivity=field_info['directivity_model'],
                piston_radius_m=field_info['piston_radius_m'], front_only=field_info['front_only'])
            complex_values = field.pressure(np.vstack([equilibrium, target]))
            p_eq, p_target = np.abs(complex_values).astype(float)
            if not np.isfinite([p_eq, p_target]).all():
                raise ValueError(f'{sid} {method}: nonfinite pressure')
            if method == 'Conventional':
                prior = float(record['pressure_abs_at_exact_equilibrium_pa'])
                if not np.isclose(p_eq, prior, rtol=1e-9, atol=1e-8):
                    raise ValueError(f'{sid}: pressure differs from deposited main-Fig8 metric')
                checks.append(dict(setting_id=sid, recomputed_pa=p_eq, recorded_pa=prior,
                                   absolute_difference_pa=abs(p_eq-prior)))
            rows.append(dict(setting_id=sid, method=method, frequency_hz=setting['frequency_hz'],
                task='Single', seed=int(record['seed']), alpha_per_m=alpha,
                command_time_s=command_time,
                command_time_source=('command_time_s on the hash-matched deposited Single endpoint'
                    if method == 'Conventional' else 'selected native solver history_wall_s[-1]'),
                pressure_abs_at_exact_equilibrium_pa=p_eq,
                pressure_abs_at_intended_target_pa=p_target,
                pressure_real_at_equilibrium_pa=float(complex_values[0].real),
                pressure_imag_at_equilibrium_pa=float(complex_values[0].imag),
                equilibrium_x_m=equilibrium[0], equilibrium_y_m=equilibrium[1], equilibrium_z_m=equilibrium[2],
                target_x_m=target[0], target_y_m=target[1], target_z_m=target[2],
                finite_ka_root_found=True, displacement_um=record['finite_ka_displacement_um'],
                phase_fingerprint=phase_hash, command_path=str(command.relative_to(ROOT)),
                equilibrium_source=(f'campaign/{sid}/tables/fig5_exact_force_field_concentration.csv'
                                    if method == 'Conventional' else f'data/alpha_tuning/{sid}/selected.json'),
                field_provenance_json=json.dumps(field.provenance(), sort_keys=True)))
    table = pd.DataFrame(rows)
    if len(table) != 42 or table[['setting_id', 'method']].duplicated().any():
        raise ValueError('Expected exactly 14 settings and three methods')
    if set(table.seed) != {260828}:
        raise ValueError('Main Figure 8 representatives must use prescribed seed 260828')
    wide = table.pivot(index='setting_id', columns='method', values='pressure_abs_at_exact_equilibrium_pa')
    comparison = wide.rename(columns={m: m+'_pressure_pa' for m in METHODS})
    for method in ('FE', 'GFE'):
        comparison[method+'_below_Conventional'] = wide[method] < wide.Conventional
        comparison['Conventional_over_'+method] = wide.Conventional / wide[method]
    comparison = comparison.reset_index()
    destination = Path(destination)
    write_bytes(destination / 'outcomes.csv', table.to_csv(index=False).encode())
    write_bytes(destination / 'comparisons.csv', comparison.to_csv(index=False).encode())
    json_out(destination / 'protocol.json', dict(
        metric='Absolute incident peak-pressure phasor at each command\'s recorded finite-ka equilibrium',
        units='Pa', amplitude_convention='peak phasor', normalization=None,
        equilibrium='Native independently resolved finite-ka elastic-bead equilibrium with effective gravity',
        pressure_source='ArbitraryArrayPressureField.pressure, with recorded field calibration and geometry',
        authoritative_main_metric='pressure_abs_at_exact_equilibrium_pa',
        representative_seed=260828, task='Single', methods=list(METHODS), unique_settings=14,
        alpha_selection='Existing displacement-minimizing alpha among 10, 100, 1000; unchanged for pressure evaluation',
        root_selection='Reuse existing native resolved root; no new root selection or stiffness filter',
        new_optimization_runs=0, new_root_searches=0,
        Conventional_main_figure_reproduction_checks=checks,
        FE_below_Conventional=int((wide.FE < wide.Conventional).sum()),
        GFE_below_Conventional=int((wide.GFE < wide.Conventional).sum()),
        intended_target_pressure='Retained separately in outcomes.csv; not the plotted quantity',
        main_figure_style=figure_style_protocol()))
    print(f'Equilibrium pressure: {len(table)} frozen commands; '
          f'FE lower in {(wide.FE < wide.Conventional).sum()}/14, '
          f'GFE lower in {(wide.GFE < wide.Conventional).sum()}/14', flush=True)
    return table


def extract_linear(destination=DATA):
    """Restore the deposited IB/GS commands and verify their exact pressure metric."""
    from appendix_settings import geometry as declared_geometry
    from hat_revision_pipeline.cache import digest_array
    from hat_revision_pipeline.exact_validator import ArbitraryArrayPressureField

    rows, checks = [], []
    for setting in SETTINGS:
        sid = setting['id']
        source = ROOT / 'campaign' / sid
        metadata_path = source / 'execution_settings.json'
        execution = json.loads(metadata_path.read_text())
        mechanics_path = source / 'tables/fig5_exact_force_field_concentration.csv'
        mechanics = pd.read_csv(mechanics_path)
        population_path = source / 'tables/fig8_single_endpoint_population.csv'
        population = pd.read_csv(population_path) if population_path.exists() else None
        with np.load(source / 'array_geometry.npz') as saved:
            positions, normals = saved['positions_m'].copy(), saved['normals'].copy()
        expected_positions, expected_normals = declared_geometry(setting)
        if not (np.array_equal(positions, expected_positions) and np.array_equal(normals, expected_normals)):
            raise ValueError(f'{sid}: deposited linear-method geometry differs from the declared setting')
        if digest_array(positions) != execution['positions'] or digest_array(normals) != execution['normals']:
            raise ValueError(f'{sid}: geometry differs from its native execution identity')
        if float(execution['setting']['frequency_hz']) != float(setting['frequency_hz']):
            raise ValueError(f'{sid}: native execution carrier mismatch')
        for method in LINEAR_METHODS:
            entries = mechanics[mechanics.task.eq('Single') & mechanics.method.eq(method)]
            if len(entries) != 1:
                raise ValueError(f'{sid} {method}: expected exactly one deposited Single endpoint')
            record = entries.iloc[0].to_dict()
            command = source / 'commands' / f'Single_{method}.npz'
            with np.load(command, allow_pickle=False) as saved:
                phase, target = saved['phase_rad'].copy(), saved['target_m'].copy()
            phase_hash = digest_array(phase)
            if phase_hash != record['phase_fingerprint']:
                raise ValueError(f'{sid} {method}: command and exact-equilibrium phase mismatch')
            if not np.array_equal(target, np.asarray(execution['config']['single_target_m'])):
                raise ValueError(f'{sid} {method}: command and native target mismatch')
            if not bool(record['finite_ka_root_found']):
                raise ValueError(f'{sid} {method}: unresolved deposited equilibrium')
            equilibrium = np.array([record[f'finite_ka_equilibrium_{axis}_m'] for axis in 'xyz'])
            if not np.isfinite(equilibrium).all():
                raise ValueError(f'{sid} {method}: invalid deposited equilibrium coordinates')
            native_metadata = json.loads(record['finite_ka_model_metadata_json'])
            field_info = native_metadata['field']
            if (float(field_info['frequency_hz']) != float(setting['frequency_hz'])
                    or native_metadata['sphere']['particle_model'] != 'homogeneous isotropic elastic solid sphere'):
                raise ValueError(f'{sid} {method}: native field/model mismatch')
            field = ArbitraryArrayPressureField(
                setting['frequency_hz'], positions, np.exp(1j * phase), normals=normals,
                sound_speed_m_s=field_info['sound_speed_m_s'],
                source_strength_pa_m=field_info['source_strength_pa_m_peak'],
                directivity=field_info['directivity_model'],
                piston_radius_m=field_info['piston_radius_m'], front_only=field_info['front_only'])
            values = field.pressure(np.vstack([equilibrium, target]))
            replay_pressure = float(abs(values[0]))
            pressure = float(record['pressure_abs_at_exact_equilibrium_pa'])
            command_time = float(record['command_time_s'])
            if not np.isfinite([pressure, command_time]).all() or min(pressure, command_time) < 0:
                raise ValueError(f'{sid} {method}: invalid native pressure or time')
            if not np.isclose(replay_pressure, pressure, rtol=1e-9, atol=1e-8):
                raise ValueError(f'{sid} {method}: pressure does not reproduce the native Fig8 metric')
            time_source = str(record.get('command_time_source', 'command_run.timing.total_s'))
            population_checked = False
            if population is not None:
                endpoint = population[population.method.eq(method)]
                if len(endpoint) != 1:
                    raise ValueError(f'{sid} {method}: invalid native Fig8 population')
                endpoint = endpoint.iloc[0]
                if (phase_hash != str(endpoint.phase_fingerprint)
                        or command_time != float(endpoint.time_s)
                        or pressure != float(endpoint.pressure_abs_at_exact_equilibrium_pa)):
                    raise ValueError(f'{sid} {method}: mechanics and native Fig8 records differ')
                time_source = str(endpoint.time_source)
                population_checked = True
            rows.append(dict(setting_id=sid, method=method, frequency_hz=setting['frequency_hz'],
                task='Single', seed=int(record['seed']), alpha_per_m=np.nan,
                command_time_s=command_time, command_time_source=time_source,
                pressure_abs_at_exact_equilibrium_pa=pressure,
                pressure_abs_at_intended_target_pa=float(record['pressure_abs_at_intended_target_pa']),
                pressure_real_at_equilibrium_pa=float(values[0].real),
                pressure_imag_at_equilibrium_pa=float(values[0].imag),
                equilibrium_x_m=equilibrium[0], equilibrium_y_m=equilibrium[1], equilibrium_z_m=equilibrium[2],
                target_x_m=target[0], target_y_m=target[1], target_z_m=target[2],
                finite_ka_root_found=True, displacement_um=float(record['finite_ka_displacement_um']),
                phase_fingerprint=phase_hash, command_path=str(command.relative_to(ROOT)),
                equilibrium_source=str(mechanics_path.relative_to(ROOT)),
                field_provenance_json=json.dumps(field.provenance(), sort_keys=True)))
            checks.append(dict(setting_id=sid, method=method, phase_fingerprint=phase_hash,
                positions_fingerprint=digest_array(positions), normals_fingerprint=digest_array(normals),
                frequency_hz=setting['frequency_hz'], equilibrium_m=equilibrium.tolist(),
                recorded_pressure_pa=pressure, independently_replayed_pressure_pa=replay_pressure,
                pressure_absolute_difference_pa=abs(pressure-replay_pressure),
                native_Fig8_population_checked=population_checked,
                command_time_s=command_time, command_time_source=time_source))
    table = pd.DataFrame(rows)
    if len(table) != 28 or table[['setting_id', 'method']].duplicated().any():
        raise ValueError('Expected exactly 28 distinct IB/GS Single outcomes')
    destination = Path(destination)
    write_bytes(destination / 'linear_outcomes.csv', table.to_csv(index=False).encode())
    json_out(destination / 'linear_protocol.json', dict(
        methods=list(LINEAR_METHODS), task='Single', unique_settings=14, commands=28,
        method_labels='IB and GS, using the native main-Figure-8 implementations',
        native_synthesis='final_figures._single_matched_baseline_commands',
        control_prescription='Eight points on a radius-0.45-wavelength vortex ring',
        IB=dict(implementation='iterative_back_projection', maxiter=200, tolerance_rad=.01),
        GS=dict(implementation='phase_only_signature_projection_audit', iterations=100),
        pressure_metric='Native recorded absolute incident peak pressure at the independently resolved finite-ka elastic-bead equilibrium',
        units='Pa', pressure_scale='linear', displacement_scale='linear',
        timing='Native BenchmarkRun.timing.total_s retained only as command provenance; not plotted',
        new_optimization_runs=0, new_root_searches=0,
        original_Conventional_FE_GFE_outcomes='The existing outcomes.csv remains unchanged by this extraction.',
        checks=checks))
    print(f'Restored {len(table)} native IB/GS Single outcomes; all pressure and command checks passed.', flush=True)
    return table


def render(data_dir=DATA, output=ROOT / 'figures'):
    from representative_selection import selection_table, CRITERION

    data_dir, output = Path(data_dir), Path(output)
    table = pd.read_csv(data_dir / 'outcomes.csv')
    if len(table) != 42 or table[['setting_id', 'method']].duplicated().any():
        raise ValueError('Expected 42 distinct prescribed Single outcomes')
    linear = pd.read_csv(data_dir / 'linear_outcomes.csv')
    if len(linear) != 28 or linear[['setting_id', 'method']].duplicated().any() or set(linear.method) != set(LINEAR_METHODS):
        raise ValueError('Expected 28 distinct native IB/GS Single outcomes')
    selection = selection_table(data_root=data_dir.parent).set_index('setting_id')
    representatives = selection.method.to_dict()
    table = pd.concat([table, linear], ignore_index=True)
    equilibrium = table[[f'equilibrium_{axis}_m' for axis in 'xyz']].to_numpy(float)
    target = table[[f'target_{axis}_m' for axis in 'xyz']].to_numpy(float)
    displacement_recomputed = np.linalg.norm(equilibrium - target, axis=1) * 1e6
    displacement_saved = table.displacement_um.to_numpy(float)
    if (not np.isfinite(displacement_saved).all() or (displacement_saved < 0).any()
            or not np.allclose(displacement_recomputed, displacement_saved, rtol=1e-10, atol=1e-8)):
        raise ValueError('Stored displacement does not reproduce the same-command equilibrium coordinates')
    displacement_audit = table[['setting_id', 'method', 'phase_fingerprint', 'displacement_um']].copy()
    displacement_audit['recomputed_displacement_um'] = displacement_recomputed
    displacement_audit['absolute_difference_um'] = np.abs(displacement_recomputed - displacement_saved)
    write_bytes(data_dir / 'displacement_audit.csv', displacement_audit.to_csv(index=False).encode())
    json_out(data_dir / 'displacement_protocol.json', dict(
        metric='Euclidean distance from the intended target to the same command\'s recorded finite-ka equilibrium',
        formula='norm(equilibrium_m - target_m) * 1e6', units='micrometres',
        authoritative_record='finite_ka_displacement_um', commands_checked=len(table),
        maximum_reproduction_difference_um=float(np.max(np.abs(displacement_recomputed-displacement_saved))),
        representative_selection=CRITERION,
        figure_coordinates='Horizontal displacement in micrometres; vertical equilibrium peak pressure in Pa',
        new_optimization_runs=0, new_root_searches=0))
    configure_style()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'axes.linewidth': .85, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False})
    manifest = []
    for group, ids, shape, size, letter in (
        ('Arrays', ARRAY_IDS, (3, 3), (11, 10.3), 'A'),
        ('Frequencies', FREQUENCY_IDS, (2, 3), (11, 7.3), 'B')):
        fig, axes = plt.subplots(*shape, figsize=size)
        fig.subplots_adjust(left=.094, right=.984, bottom=.092 if group == 'Arrays' else .13,
                            top=.91 if group == 'Arrays' else .87, hspace=.53, wspace=.32)
        axis_records = []
        for index, (sid, ax) in enumerate(zip(ids, axes.ravel(), strict=True)):
            displayed_methods = ('Conventional', representatives[sid], *LINEAR_METHODS)
            case = table[table.setting_id.eq(sid)].set_index('method').loc[list(displayed_methods)]
            chosen = case.loc[representatives[sid]]
            if (chosen.phase_fingerprint != selection.loc[sid, 'phase_fingerprint']
                    or chosen.alpha_per_m != selection.loc[sid, 'alpha_per_m']
                    or int(chosen.seed) != int(selection.loc[sid, 'seed'])):
                raise ValueError(f'{sid}: benchmark representative differs from the common display selection')
            label = LABELS[sid] if group == 'Arrays' else f'{case.frequency_hz.iloc[0]/1000:g} kHz'
            title = f'({chr(97+index)}) {label}'
            ax.set_title(title, loc='left', pad=10)
            displacement = case.displacement_um.to_numpy(float)
            pressure = case.pressure_abs_at_exact_equilibrium_pa.to_numpy(float)
            if (not np.isfinite(displacement).all() or (displacement < 0).any()
                    or not np.isfinite(pressure).all() or (pressure < 0).any()):
                raise ValueError(f'{sid}: invalid pressure or displacement values')
            ax.set_xlim(*_metric_limits(displacement, include_zero=True))
            pressure_display = pressure_axis(ax, pressure)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.tick_params(direction='out', length=3.5, width=.8)
            ax.grid(True, axis='y', alpha=.24)
            ax.grid(False, axis='x')
            ax.axhline(0., color='.45', lw=.7, zorder=1)
            for method, x_value, y_value in zip(displayed_methods, displacement, pressure, strict=True):
                native_method = MAIN_METHOD[method]
                hollow = native_method in HOLLOW_METHODS
                ax.scatter(x_value, y_value,
                    s=METHOD_MARKER_SIZES[native_method], marker=METHOD_MARKERS[native_method],
                    facecolor='none' if hollow else COLORS[method],
                    edgecolor=COLORS[method] if hollow else 'white',
                    linewidth=2.0 if hollow else 1.1, zorder=5)
            axis_records.append(dict(setting_id=sid, title=title,
                displayed_methods=list(displayed_methods),
                force_equilibrium_representative=representatives[sid],
                xaxis='finite-ka equilibrium displacement in micrometres',
                yaxis='absolute peak pressure at the same finite-ka equilibrium in Pa',
                xscale='linear', yscale=pressure_display['scale'],
                pressure_axis=pressure_display,
                x_limits=list(ax.get_xlim()), y_limits=list(ax.get_ylim()),
                points=[dict(method=method,
                    pressure_pa=float(case.loc[method, 'pressure_abs_at_exact_equilibrium_pa']),
                    displacement_um=float(case.loc[method, 'displacement_um']),
                    phase_fingerprint=str(case.loc[method, 'phase_fingerprint']),
                    plot_x=float(case.loc[method, 'displacement_um']),
                    plot_y=float(case.loc[method, 'pressure_abs_at_exact_equilibrium_pa']),
                    marker=METHOD_MARKERS[MAIN_METHOD[method]], marker_area_pt2=METHOD_MARKER_SIZES[MAIN_METHOD[method]])
                    for method in displayed_methods]))
        legend_methods = [m for m in (*METHODS, *LINEAR_METHODS) if m in ('Conventional', *LINEAR_METHODS) or
                          any(representatives[sid] == m for sid in ids)]
        handles = [matplotlib.lines.Line2D([0], [0], marker=METHOD_MARKERS[MAIN_METHOD[m]],
            linestyle='none', markerfacecolor='none' if m in HOLLOW_METHODS else COLORS[m],
            markeredgecolor=COLORS[m] if m in HOLLOW_METHODS else 'white',
            markersize=9.5, markeredgewidth=1.8 if m in HOLLOW_METHODS else 1.1,
            label=m) for m in legend_methods]
        fig.legend(handles=handles, labels=legend_methods, loc='upper center',
            bbox_to_anchor=(.54, .985), ncol=len(legend_methods), frameon=False, columnspacing=1.5,
            handletextpad=.4, fontsize=12)
        fig.supylabel(r'$|p(\mathbf{r}_{\mathrm{eq}})|$ (Pa)', x=.015, y=.51, fontsize=13)
        fig.supxlabel(r'Displacement ($\mu$m)', y=.02, fontsize=13)
        stem = f'Appendix_4{letter}_EquilibriumPressure_{group}'
        for extension in ('png', 'pdf', 'svg'):
            save_figure(fig, output / f'{stem}.{extension}', dpi=240 if extension == 'png' else None)
        plt.close(fig)
        manifest.append(dict(figure=stem, layout=list(shape), setting_ids=ids,
            measurement='pressure_abs_at_exact_equilibrium_pa', pressure_unit='Pa (peak)',
            measurements=['pressure_abs_at_exact_equilibrium_pa', 'displacement_um'], displacement_unit='micrometres',
            case_tile_count=len(ids), data_axes_count=len(ids),
            horizontal_measurement='displacement_um', vertical_measurement='pressure_abs_at_exact_equilibrium_pa',
            rendering_source='native fig8_methods_only symbols; physical coordinates with marked linear pressure ranges',
            presentation_selection=CRITERION,
            raw_outcomes_retained=70, endpoints_per_panel=4,
            task='Single', endpoints_per_method=1, seed=260828, legend_methods=legend_methods,
            panels=axis_records))
    json_out(output / 'equilibrium_pressure_figure_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--linear-only', action='store_true', help='Restore IB/GS records while retaining the existing Conventional/FE/GFE outcomes.')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    arguments = parser.parse_args()
    if not arguments.render_only:
        if not arguments.linear_only:
            extract(arguments.data_dir)
        extract_linear(arguments.data_dir)
    render(arguments.data_dir, arguments.output)
