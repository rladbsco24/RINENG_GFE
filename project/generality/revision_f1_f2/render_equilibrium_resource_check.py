"""Plot released Triple equilibrium results before and after array expansion.

Only the two completed release summary tables are read. The renderer performs
no optimization or force evaluation. Main Figure 8 symbols, colors, median
points and min--max bars follow ``render_triple_benchmark.py`` exactly.
"""
from __future__ import annotations

import argparse
from io import BytesIO

from rineng_content_id import content_identity
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from appendix_io import write_bytes


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/equilibrium_resource_check/release'
OUTPUT = ROOT / 'delivery/RINENG_Equilibrium_Resource_Check.pdf'
METHODS = ('Conventional', 'FE', 'GFE', 'IB', 'GS')
SETTING_ORDER = ('A08', 'A07', 'F20', 'S06', 'F28')
LABELS = {
    'A08': 'Four clusters · 40 kHz', 'A07': 'Annular · 40 kHz',
    'F20': 'Square · 20 kHz', 'S06': 'Square · 25 kHz',
    'F28': 'Square · 28 kHz', 'A06': 'Square · 40 kHz',
    'S01': 'Rectangular · 40 kHz', 'S02': 'Opposed · 40 kHz',
    'S03': 'Spherical cap · 40 kHz', 'S04': 'Fermat disk · 40 kHz',
    'S05': 'Arbitrary 3D · 40 kHz', 'A09': 'Tilted plane · 40 kHz',
    'S07': 'Square · 58 kHz', 'S08': 'Square · 100 kHz',
}

# Exact main Figure 8 presentation constants; copied here to keep this reader
# independent of the simulation pipeline imported by fig8_methods_only.py.
COLORS = {'Conventional': '#3B6FA1', 'FE': '#C76A12', 'GFE': '#007C78',
          'IB': '#4C7C2F', 'GS': '#80568D'}
METHOD_MARKERS = {'Conventional': 'o', 'FE': 's', 'GFE': 'P', 'IB': '^', 'GS': 'v'}
METHOD_MARKER_SIZES = {'Conventional': 92, 'FE': 92, 'GFE': 135, 'IB': 160, 'GS': 135}
HOLLOW_METHODS = {'IB', 'GS'}
METRIC_COLUMNS = (
    'displacement_median_um', 'displacement_min_um', 'displacement_max_um',
    'pressure_median_pa', 'pressure_min_pa', 'pressure_max_pa',
)


def _metric_limits(values, *, include_zero=True):
    """Unmodified main Figure 8 physical-coordinate padding."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return (0.0, 1.0)
    low, high = float(finite.min()), float(finite.max())
    if include_zero:
        low = min(0.0, low)
        high = max(0.0, high)
    span = max(high - low, 0.12 * max(abs(low), abs(high), 1.0e-12))
    return low - 0.08 * span, high + 0.16 * span


def _style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 11, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': .85, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True,
        'grid.alpha': .24, 'grid.linewidth': .55,
        'axes.formatter.use_mathtext': True,
    })


def _read_release(path):
    """Reject missing commands or numerical placeholders before rendering."""
    content = path.read_bytes()
    frame = pd.read_csv(BytesIO(content))
    required = {'setting_id', 'method', 'source_count', 'expected_targets',
                'root_found_count', *METRIC_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'{path.name}: missing columns {missing}')
    if frame.empty or frame[['setting_id', 'method']].duplicated().any():
        raise ValueError(f'{path.name}: nonempty unique setting/method rows required')
    for sid, case in frame.groupby('setting_id', sort=False):
        if set(case.method) != set(METHODS) or len(case) != len(METHODS):
            raise ValueError(f'{path.name}: {sid} must include all five methods')
        counts = case.source_count.to_numpy(float)
        if (not np.isfinite(counts).all() or (counts <= 0).any()
                or (counts != np.floor(counts)).any() or len(set(counts)) != 1):
            raise ValueError(f'{path.name}: {sid} needs one positive integer source count')
    for row in frame.itertuples(index=False):
        n = float(row.root_found_count)
        if int(row.expected_targets) != 3 or float(row.expected_targets) != 3:
            raise ValueError(f'{path.name}: {row.setting_id} requires three targets')
        if not np.isfinite(n) or n != int(n) or not 0 <= n <= 3:
            raise ValueError(f'{path.name}: invalid resolved-root count')
        values = np.array([getattr(row, column) for column in METRIC_COLUMNS], dtype=float)
        if n == 0:
            if not np.isnan(values).all():
                raise ValueError(f'{path.name}: unresolved metrics must remain NaN')
            continue
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{path.name}: invalid resolved pressure/displacement')
        for median, lower, upper in values.reshape(2, 3):
            if not lower <= median <= upper:
                raise ValueError(f'{path.name}: inconsistent median/min/max')
    return frame, content_identity(content).hexdigest()


def _plot_case(ax, case, limits):
    """Use the current Triple benchmark's scatter and range-bar calls."""
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.tick_params(direction='out', length=3.5, width=.8)
    ax.grid(True, axis='y', alpha=.24)
    ax.grid(False, axis='x')
    ax.axhline(0., color='.45', lw=.7, zorder=1)
    points = []
    for method in METHODS:
        row = case.loc[method]
        count = int(row.root_found_count)
        point = dict(method=method, root_found_count=count, expected_targets=3,
                     plotted=count > 0, pressure_pa=None, displacement_um=None)
        if count:
            x, y = float(row.displacement_median_um), float(row.pressure_median_pa)
            x_range = [float(row.displacement_min_um), float(row.displacement_max_um)]
            y_range = [float(row.pressure_min_pa), float(row.pressure_max_pa)]
            hollow = method in HOLLOW_METHODS
            ax.errorbar([x], [y],
                xerr=np.array([[max(0., x-x_range[0])], [max(0., x_range[1]-x)]]),
                yerr=np.array([[max(0., y-y_range[0])], [max(0., y_range[1]-y)]]),
                fmt='none', ecolor=COLORS[method], elinewidth=1.55,
                capsize=3.2, capthick=1.4, alpha=1., zorder=4)
            ax.scatter(x, y, s=METHOD_MARKER_SIZES[method],
                marker=METHOD_MARKERS[method],
                facecolor='none' if hollow else COLORS[method],
                edgecolor=COLORS[method] if hollow else 'white',
                linewidth=2.0 if hollow else 1.1, zorder=5)
            point.update(pressure_pa=y, displacement_um=x,
                         pressure_range_pa=y_range, displacement_range_um=x_range)
        points.append(point)
    return points


def _legend(fig):
    handles = [Line2D([0], [0], marker=METHOD_MARKERS[method],
        linestyle='none', markerfacecolor='none' if method in HOLLOW_METHODS else COLORS[method],
        markeredgecolor=COLORS[method] if method in HOLLOW_METHODS else 'white',
        markersize=9.5, markeredgewidth=1.8 if method in HOLLOW_METHODS else 1.1,
        label=method) for method in METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .951),
               ncol=len(METHODS), frameon=False, columnspacing=1.5,
               handletextpad=.4, fontsize=12)


def render(data_dir=DATA, output=OUTPUT):
    """Return a PDF plus a small record of the displayed, released values."""
    data_dir, output = Path(data_dir), Path(output)
    baseline, baseline_hash = _read_release(data_dir / 'baseline_summary.csv')
    resource, resource_hash = _read_release(data_dir / 'resource_summary.csv')
    if set(baseline.setting_id) != set(resource.setting_id):
        raise ValueError('Baseline and expanded release settings must match exactly')
    available = set(resource.setting_id)
    ids = [sid for sid in SETTING_ORDER if sid in available]
    ids += sorted(available - set(ids))
    _style()
    stream = BytesIO()
    pages = []
    with PdfPages(stream, metadata={
        'Title': 'Triple equilibrium pressure and displacement: array resource check',
        'Author': 'RINENG appendix', 'CreationDate': None, 'ModDate': None,
    }) as pdf:
        for offset in range(0, len(ids), 3):
            page_ids = ids[offset:offset+3]
            columns = len(page_ids)
            fig, axes = plt.subplots(2, columns, figsize=(11, 9.7), squeeze=False)
            fig.subplots_adjust(left=.102, right=.982, bottom=.33, top=.863,
                                hspace=.43, wspace=.33)
            fig.text(.5, .983, 'Triple equilibrium pressure and displacement',
                     ha='center', va='top', fontsize=15)
            _legend(fig)
            fig.text(.025, .754, 'Baseline', rotation=90, va='center', fontsize=11)
            fig.text(.025, .451, 'Resource check', rotation=90, va='center', fontsize=11)
            fig.supylabel(r'$|p(\mathbf{r}_{\mathrm{eq}})|$ (Pa)', x=.057, y=.596, fontsize=13)
            fig.supxlabel(r'Displacement ($\mu$m)', y=.271, fontsize=13)
            panels, count_rows = [], []
            for column, sid in enumerate(page_ids):
                paired = [frame.loc[frame.setting_id.eq(sid)].set_index('method').loc[list(METHODS)]
                          for frame in (baseline, resource)]
                pooled = pd.concat(paired)
                valid = pooled.root_found_count.gt(0)
                limits = (
                    _metric_limits(pooled.loc[valid, ['displacement_min_um', 'displacement_max_um']].to_numpy(float).ravel()),
                    _metric_limits(pooled.loc[valid, ['pressure_min_pa', 'pressure_max_pa']].to_numpy(float).ravel()),
                )
                for row_index, (stage, case) in enumerate(zip(('Baseline', 'Resource check'), paired)):
                    ax = axes[row_index, column]
                    n_sources = int(case.source_count.iloc[0])
                    title = f'{LABELS.get(sid, sid)}\n{n_sources} transducers'
                    ax.set_title(title, loc='left', pad=7)
                    points = _plot_case(ax, case, limits)
                    panels.append(dict(setting_id=sid, stage=stage, source_count=n_sources,
                                       xscale='linear', yscale='linear',
                                       x_limits=list(limits[0]), y_limits=list(limits[1]), points=points))
                    count_rows.append([LABELS.get(sid, sid), stage,
                                       *[f'{int(case.loc[m, "root_found_count"])}/3' for m in METHODS]])
            fig.text(.102, .244, 'Resolved equilibria', fontsize=10.5, va='top')
            table_ax = fig.add_axes([.102, .073, .88, .15])
            table_ax.axis('off')
            table = table_ax.table(cellText=count_rows,
                colLabels=['Setting', 'Run', *METHODS],
                colWidths=[.29, .12, .16, .095, .095, .12, .12],
                loc='upper left', cellLoc='center', colLoc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(8.4)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('.84')
                cell.set_linewidth(.35)
                cell.set_facecolor('#F2F4F5' if row == 0 else 'white')
                if row == 0:
                    cell.set_text_props(weight='semibold')
                if col < 2:
                    cell.set_text_props(ha='left')
            fig.text(.102, .052,
                'Points: medians. Bars: minimum–maximum over resolved equilibria. '
                'Linear axes and matched limits within each pair.\n'
                'Unresolved values are not plotted. '
                'Extended-search roots are included where recorded; search regions and displacement vectors are exported.',
                fontsize=8.2, va='top', linespacing=1.35)
            pdf.savefig(fig, facecolor='white', bbox_inches=None)
            plt.close(fig)
            pages.append(dict(page=len(pages)+1, setting_ids=page_ids, panels=panels))
    write_bytes(output, stream.getvalue())
    manifest = dict(
        output=str(output), pages=pages, setting_ids=ids, task='Triple',
        legend_methods=list(METHODS),
        marker_statistic='Median over resolved numerical finite-ka equilibria',
        error_bars='Minimum and maximum over the same resolved roots',
        aggregation_filter='Numerical root found; no restoring classification filter',
        unresolved_values='Undefined and absent from coordinate plots; counts shown below plots',
        source_hashes=dict(baseline_summary_content_id=baseline_hash, resource_summary_content_id=resource_hash),
    )
    write_bytes(output.with_suffix('.json'),
                (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = render(args.data_dir, args.output)
    print(f'{result["output"]}: {len(result["pages"])} pages')
    return result


if __name__ == '__main__':
    main()
