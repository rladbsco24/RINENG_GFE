"""Render native Triple equilibrium pressure and displacement from saved roots.

The marker coordinates are medians over resolved roots, and the bars show the
minimum and maximum of those same roots. Missing roots remain undefined.
"""
from __future__ import annotations

import argparse
import fingerprintlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

from appendix_io import save_figure, write_bytes
from appendix_settings import ROOT, SETTINGS, ARRAY_IDS, FREQUENCY_IDS
from equilibrium_pressure import (
    COLORS, HOLLOW_METHODS, LABELS, MAIN_METHOD, METHOD_MARKERS,
    METHOD_MARKER_SIZES, _metric_limits, configure_style,
)

METHODS = ('Conventional', 'FE', 'GFE', 'IB', 'GS')
DATA = ROOT / 'data/triple_benchmark'
MEASUREMENTS = {
    'displacement': ('displacement_um', 'um'),
    'pressure': ('pressure_abs_at_exact_equilibrium_pa', 'pa'),
}


def _checked_data(data_dir):
    """Verify every displayed summary against its three saved target records."""
    summary_path, outcomes_path = data_dir / 'summary.csv', data_dir / 'outcomes.csv'
    summary_bytes, outcomes_bytes = summary_path.read_bytes(), outcomes_path.read_bytes()
    from io import BytesIO
    summary = pd.read_csv(BytesIO(summary_bytes))
    outcomes = pd.read_csv(BytesIO(outcomes_bytes))
    if (len(summary) != 70 or len(outcomes) != 210
            or summary[['setting_id', 'method']].duplicated().any()):
        raise ValueError('Triple benchmark requires 70 commands and all 210 target records')
    if set(summary.method) != set(METHODS):
        raise ValueError('Triple benchmark requires Conventional, FE, GFE, IB and GS')
    for row in summary.itertuples(index=False):
        raw = outcomes[outcomes.setting_id.eq(row.setting_id) & outcomes.method.eq(row.method)]
        if len(raw) != 3 or int(row.expected_targets) != 3 or raw.target_index.nunique() != 3:
            raise ValueError(f'{row.setting_id} {row.method}: three distinct targets required')
        if (set(raw.phase_fingerprint.astype(str)) != {str(row.phase_fingerprint)}
                or set(raw.candidate_id.astype(str)) != {str(row.candidate_id)}):
            raise ValueError(f'{row.setting_id} {row.method}: summary command identity mismatch')
        resolved = raw.finite_ka_root_found.astype(str).str.lower().eq('true')
        if int(resolved.sum()) != int(row.root_found_count):
            raise ValueError(f'{row.setting_id} {row.method}: root count mismatch')
        for metric, (raw_column, unit) in MEASUREMENTS.items():
            saved = np.array([getattr(row, f'{metric}_{stat}_{unit}')
                              for stat in ('median', 'min', 'max')], dtype=float)
            if not resolved.any():
                if not np.isnan(saved).all():
                    raise ValueError(f'{row.setting_id} {row.method}: missing roots must remain undefined')
                continue
            values = raw.loc[resolved, raw_column].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'{row.setting_id} {row.method}: invalid resolved {metric}')
            recomputed = np.array([np.median(values), np.min(values), np.max(values)])
            if not np.allclose(saved, recomputed, rtol=1e-10, atol=1e-8):
                raise ValueError(f'{row.setting_id} {row.method}: {metric} aggregate mismatch')
    hashes = dict(summary_fingerprint=fingerprintlib.fingerprint(summary_bytes).hexdigest(),
                  outcomes_fingerprint=fingerprintlib.fingerprint(outcomes_bytes).hexdigest())
    return summary, hashes


def render(data_dir=DATA, output=ROOT / 'figures', *, baseline=False):
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not baseline and data_dir.resolve() == DATA.resolve():
        from final_benchmark_data import load_final_benchmark
        summary, _, _, final_provenance = load_final_benchmark(ROOT)
        source_hashes = final_provenance['source_hashes']
    else:
        summary, source_hashes = _checked_data(data_dir)
        final_provenance = None
    frequencies = {setting['id']: setting['frequency_hz'] for setting in SETTINGS}
    configure_style()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'axes.linewidth': .85, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False})
    manifests = []
    for group, ids, shape, size, letter in (
        ('Arrays', ARRAY_IDS, (3, 3), (11, 10.3), 'A'),
        ('Frequencies', FREQUENCY_IDS, (2, 3), (11, 7.3), 'B')):
        fig, axes = plt.subplots(*shape, figsize=size)
        fig.subplots_adjust(left=.094, right=.984, bottom=.092 if group == 'Arrays' else .13,
                            top=.91 if group == 'Arrays' else .87, hspace=.53, wspace=.32)
        panels = []
        for index, (sid, ax) in enumerate(zip(ids, axes.ravel(), strict=True)):
            case = summary[summary.setting_id.eq(sid)].set_index('method').loc[list(METHODS)]
            label = LABELS[sid] if group == 'Arrays' else f'{frequencies[sid]/1000:g} kHz'
            source_count = int(case.source_count.iloc[0]) if 'source_count' in case else None
            title = f'({chr(97+index)}) {label}'
            if source_count is not None:
                title += f' · N = {source_count}'
            ax.set_title(title, loc='left', pad=7)
            valid = case.root_found_count.gt(0)
            points = []
            if valid.any():
                displacement_bounds = case.loc[valid, ['displacement_min_um', 'displacement_max_um']].to_numpy(float) * 1e-6
                pressure_bounds = case.loc[valid, ['pressure_min_pa', 'pressure_max_pa']].to_numpy(float)
                ax.set_xlim(*_metric_limits(displacement_bounds.ravel(), include_zero=True))
                ax.set_ylim(*_metric_limits(pressure_bounds.ravel(), include_zero=True))
                ax.set_xscale('linear')
                ax.set_yscale('linear')
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                formatter = ScalarFormatter(useMathText=True); formatter.set_powerlimits((0, 0))
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                ax.tick_params(direction='out', length=3.5, width=.8)
                ax.grid(True, axis='y', alpha=.24)
                ax.grid(False, axis='x')
                ax.axhline(0., color='.45', lw=.7, zorder=1)
            else:
                ax.set_axis_off()
                ax.text(.5, .5, 'No resolved equilibria', transform=ax.transAxes,
                        ha='center', va='center', fontsize=11, color='.4')
            for method in METHODS:
                row = case.loc[method]
                n = int(row.root_found_count)
                point = dict(method=method, root_found_count=n, expected_targets=int(row.expected_targets),
                    phase_fingerprint=str(row.phase_fingerprint), candidate_id=str(row.candidate_id),
                    plotted=n > 0, pressure_pa=None, displacement_um=None,
                    pressure_range_pa=None, displacement_range_um=None)
                if final_provenance is not None:
                    point.update(native_width_root_found_count=int(row.native_width_root_found_count),
                        expanded_root_found_count=int(row.expanded_root_found_count),
                        source_count=int(row.source_count), data_scope=str(row.data_scope),
                        command_path=str(row.command_path))
                if n:
                    x, y = 1e-6 * float(row.displacement_median_um), float(row.pressure_median_pa)
                    x_range = [1e-6 * float(row.displacement_min_um), 1e-6 * float(row.displacement_max_um)]
                    y_range = [float(row.pressure_min_pa), float(row.pressure_max_pa)]
                    native_method = MAIN_METHOD[method]
                    hollow = native_method in HOLLOW_METHODS or method == 'FE'
                    ax.errorbar([x], [y],
                        xerr=np.array([[max(0., x-x_range[0])], [max(0., x_range[1]-x)]]),
                        yerr=np.array([[max(0., y-y_range[0])], [max(0., y_range[1]-y)]]),
                        fmt='none', ecolor=COLORS[method], elinewidth=1.55,
                        capsize=3.2, capthick=1.4, alpha=1., zorder=4)
                    ax.scatter(x, y, s=METHOD_MARKER_SIZES[native_method],
                        marker=METHOD_MARKERS[native_method],
                        facecolor='none' if hollow else COLORS[method],
                        edgecolor=COLORS[method] if hollow else 'white',
                        linewidth=2.0 if hollow else 1.1, zorder=5)
                    point.update(pressure_pa=y, displacement_um=1e6*x, displacement_m=x,
                        pressure_range_pa=y_range, displacement_range_um=[1e6*v for v in x_range], displacement_range_m=x_range,
                        marker=METHOD_MARKERS[native_method],
                        marker_area_pt2=METHOD_MARKER_SIZES[native_method])
                points.append(point)
            panels.append(dict(setting_id=sid, title=title, source_count=source_count,
                displayed_methods=list(METHODS),
                root_found_counts={method: int(case.loc[method, 'root_found_count']) for method in METHODS},
                expected_targets_per_method=int(case.expected_targets.iloc[0]), xscale='linear', yscale='linear',
                x_limits=list(ax.get_xlim()) if valid.any() else None,
                y_limits=list(ax.get_ylim()) if valid.any() else None,
                pressure_axis=dict(scale='linear', omitted_interval_pa=None), points=points))
        handles = [Line2D([0], [0], marker=METHOD_MARKERS[MAIN_METHOD[method]],
            linestyle='none', markerfacecolor='none' if MAIN_METHOD[method] in HOLLOW_METHODS or method == 'FE' else COLORS[method],
            markeredgecolor=COLORS[method] if MAIN_METHOD[method] in HOLLOW_METHODS or method == 'FE' else 'white',
            markersize=9.5, markeredgewidth=1.8 if MAIN_METHOD[method] in HOLLOW_METHODS or method == 'FE' else 1.1,
            label=method) for method in METHODS]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .985),
                   ncol=len(METHODS), frameon=False, columnspacing=1.5,
                   handletextpad=.4, fontsize=12)
        fig.supylabel(r'$|p(\mathbf{r}_{\mathrm{eq}})|$ (Pa)', x=.015, y=.51, fontsize=13)
        fig.supxlabel('Displacement (m)', y=.02, fontsize=13)
        assert len(fig.axes) == len(ids)
        stem = f'Appendix_4{letter}_EquilibriumPressure_{group}'
        paths = []
        for extension in ('png', 'pdf', 'svg'):
            path = output / f'{stem}.{extension}'
            save_figure(fig, path, dpi=240 if extension == 'png' else None)
            paths.append(str(path))
        plt.close(fig)
        manifests.append(dict(figure=stem, layout=list(shape), setting_ids=ids,
            task='Triple', case_tile_count=len(ids), data_axes_count=len(ids),
            axes_count=len(ids),array_inset_axes_count=0,
            horizontal_measurement='displacement_m', vertical_measurement='pressure_abs_at_exact_equilibrium_pa',
            measurements=['pressure_abs_at_exact_equilibrium_pa', 'displacement_m'],
            pressure_unit='Pa (peak)', displacement_unit='m',
            marker_statistic='Median over resolved roots',
            error_bars='Minimum and maximum over the same resolved roots; no confidence interval interpretation',
            missing_data='Unresolved target values remain undefined; zero-root commands are not plotted; counts retained for every method',
            aggregation_filter='Numerical finite-ka root found, without a restoring classification filter',
            rendering_source='Native main Figure8 markers, colors and min/max bar style; both physical coordinates linear',
            raw_outcomes_retained=int(final_provenance['target_count']) if final_provenance is not None else 210,
            commands_retained=int(final_provenance['command_count']) if final_provenance is not None else 70, endpoints_per_panel=5,
            legend_methods=list(METHODS), source_hashes=source_hashes,
            active_data_scope=final_provenance.get('evidence_mode', 'final_cached_resource_overlay') if final_provenance is not None else 'frozen_baseline',
            final_root_provenance=({key: final_provenance[key] for key in (
                'resolved_target_count', 'native_width_root_found_count', 'expanded_root_found_count',
                'search_disclosure', 'expanded_targets')} if final_provenance is not None else None),
            paths=paths, panels=panels))
    write_bytes(output / 'equilibrium_pressure_figure_manifest.json',
                (json.dumps(manifests, indent=2, allow_nan=False)+'\n').encode())
    return manifests


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    parser.add_argument('--baseline', action='store_true',
                        help='Render immutable original baseline instead of final cached resource overlay')
    arguments = parser.parse_args()
    render(arguments.data_dir, arguments.output, baseline=arguments.baseline)
