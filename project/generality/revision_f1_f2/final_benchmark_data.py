"""Activate completed Triple resource checks without modifying their baseline.

Only cached tables and command archives are read. The final benchmark contains
the original 45 commands outside the resource check and its 25 released commands.
No optimizer, field evaluation, force model or equilibrium search is called.
"""
from __future__ import annotations

import fingerprintlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from appendix_io import write_bytes


ROOT = Path(__file__).resolve().parent
BASE = Path('data/triple_benchmark')
RELEASE = Path('data/equilibrium_resource_check/release')
OUTPUT = Path('data/final_triple_benchmark')
METHODS = ('Conventional', 'FE', 'GFE', 'IB', 'GS')
RESOURCE_SETTINGS = ('A07', 'A08', 'F20', 'S06', 'F28')
SELECTION_RULE = ('Preserve the completed resource selection: most roots found '
                  'within the native +/-4a search, then most total observed '
                  'roots, then minimum median displacement. No new selection '
                  'or optimization is performed during export.')
METRICS = {'displacement': ('displacement_um', 'um'),
           'pressure': ('pressure_abs_at_exact_equilibrium_pa', 'pa')}


def _digest_array(array):
    """The unchanged phase-identity encoding used by the native cache."""
    array = np.ascontiguousarray(array)
    digest = fingerprintlib.fingerprint()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _truth(values):
    if not values.astype(str).str.lower().isin(('true', 'false')).all():
        raise ValueError('Every root flag must be an explicit true or false')
    return values.astype(str).str.lower().eq('true')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def required_input_paths(root=None):
    """Return all table and command inputs needed by the cached final loader."""
    root = Path(root) if root is not None else ROOT
    paths = [BASE / 'summary.csv', BASE / 'outcomes.csv',
             RELEASE / 'resource_summary.csv',
             RELEASE / 'inputs/resource_outcomes.csv',
             RELEASE / 'selected_weights.csv']
    for relative in (BASE / 'summary.csv', RELEASE / 'resource_summary.csv'):
        paths.extend(Path(value) for value in pd.read_csv(root / relative).command_path)
    return [root / path for path in dict.fromkeys(paths)]


def load_final_benchmark(root=None):
    """Return ``(summary70, outcomes210, resource_weights10, provenance)``.

    The 10 weight rows are the resource release's FE/GFE selections; the unified
    weight exporter combines these with the independently validated baseline
    weights. Baseline source tables and commands remain immutable.
    """
    root = Path(root).resolve() if root is not None else ROOT
    if (root / "data/generated_triple_source.json").is_file():
        return load_generated_benchmark(root)
    source_hashes = {}

    def read(relative):
        content = (root / relative).read_bytes()
        source_hashes[str(relative)] = fingerprintlib.fingerprint(content).hexdigest()
        return pd.read_csv(BytesIO(content))

    baseline = read(BASE / 'summary.csv')
    raw_baseline = read(BASE / 'outcomes.csv')
    resource = read(RELEASE / 'resource_summary.csv')
    raw_resource = read(RELEASE / 'inputs/resource_outcomes.csv')
    weights = read(RELEASE / 'selected_weights.csv')
    _require(len(baseline) == 70 and len(raw_baseline) == 210,
             'The frozen baseline must contain 70 commands and 210 target rows')
    _require(len(resource) == 25 and len(raw_resource) == 75,
             'The resource release must contain 25 commands and 75 target rows')
    _require(set(resource.setting_id) == set(RESOURCE_SETTINGS),
             'The resource release must replace exactly the five approved settings')

    baseline['data_scope'] = 'original_cached_benchmark'
    resource['data_scope'] = 'completed_resource_check'
    raw_baseline['data_scope'] = 'original_cached_benchmark'
    raw_resource['data_scope'] = 'completed_resource_check'
    raw_baseline['native_width_root_found'] = _truth(raw_baseline.finite_ka_root_found)
    _truth(raw_resource.native_width_root_found)
    # Preserve all original benchmark measurement columns, and only the root
    # provenance needed to interpret the final figures. Detailed raw resource
    # records remain in their immutable release table.
    raw_columns = list(raw_baseline.columns)
    for column in ('source_count', 'root_search_half_width_a', 'search_method',
                   'root_search_strategy', 'root_search_max_nfev_per_start',
                   'root_search_start_count'):
        if column not in raw_columns:
            raw_columns.append(column)
    outcomes = pd.concat([
        raw_baseline.loc[~raw_baseline.setting_id.isin(RESOURCE_SETTINGS)].reindex(columns=raw_columns),
        raw_resource.reindex(columns=raw_columns),
    ], ignore_index=True)
    summary = pd.concat([
        baseline.loc[~baseline.setting_id.isin(RESOURCE_SETTINGS)], resource,
    ], ignore_index=True)
    order = {(row.setting_id, row.method): index
             for index, row in enumerate(baseline.itertuples(index=False))}
    summary['_order'] = [order[(sid, method)] for sid, method in zip(summary.setting_id, summary.method)]
    outcomes['_order'] = [order[(sid, method)] for sid, method in zip(outcomes.setting_id, outcomes.method)]
    summary = summary.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    outcomes = outcomes.sort_values(['_order', 'target_index']).drop(columns='_order').reset_index(drop=True)
    _require(len(summary) == 70 and len(outcomes) == 210
             and not summary[['setting_id', 'method']].duplicated().any(),
             'Final benchmark requires exactly 70 unique commands and 210 targets')
    for sid, case in summary.groupby('setting_id', sort=False):
        _require(len(case) == 5 and set(case.method) == set(METHODS),
                 f'{sid}: all five methods must be retained')

    command_audit = []
    for i, row in summary.iterrows():
        key = f'{row.setting_id} {row.method}'
        mask = outcomes.setting_id.eq(row.setting_id) & outcomes.method.eq(row.method)
        frame = outcomes.loc[mask].sort_values('target_index')
        _require(len(frame) == 3 and set(frame.target_index) == {0, 1, 2}
                 and int(row.expected_targets) == 3, f'{key}: three distinct targets required')
        _require(set(frame.phase_fingerprint) == {row.phase_fingerprint}
                 and set(frame.candidate_id) == {row.candidate_id}
                 and set(frame.command_path) == {row.command_path},
                 f'{key}: target records disagree with command identity')
        command_path = root / row.command_path
        with np.load(command_path, allow_pickle=False) as command:
            phase = command['phase_rad']
            targets = command['targets_m']
        _require(phase.ndim == 1 and np.isfinite(phase).all()
                 and _digest_array(phase) == row.phase_fingerprint,
                 f'{key}: terminal phase hash mismatch')
        _require(targets.shape == (3, 3) and np.allclose(
            targets, frame[[f'target_{axis}_m' for axis in 'xyz']].to_numpy(float),
            rtol=0, atol=1e-15), f'{key}: saved target coordinates mismatch')
        source_count = len(phase)
        metadata = [json.loads(value) for value in frame.finite_ka_model_metadata_json]
        _require(all(int(record['field']['number_of_elements']) == source_count for record in metadata),
                 f'{key}: validation source count differs from terminal phase count')
        _require(all('elastic' in str(record['model_name']).lower() for record in metadata),
                 f'{key}: pressure must be validated at elastic-sphere equilibria')
        if row.setting_id in RESOURCE_SETTINGS:
            _require(int(row.source_count) == source_count, f'{key}: resource source budget mismatch')
        summary.at[i, 'source_count'] = source_count
        outcomes.loc[mask, 'source_count'] = source_count
        found = _truth(frame.finite_ka_root_found)
        native = _truth(frame.native_width_root_found)
        _require(not (native & ~found).any(), f'{key}: native roots must also be observed roots')
        _require(int(found.sum()) == int(row.root_found_count), f'{key}: resolved-root count mismatch')
        if row.setting_id in RESOURCE_SETTINGS:
            _require(int(native.sum()) == int(row.native_width_root_found_count)
                     and int((found & ~native).sum()) == int(row.expanded_root_found_count),
                     f'{key}: native/expanded root provenance mismatch')
        summary.at[i, 'native_width_root_found_count'] = int(native.sum())
        summary.at[i, 'expanded_root_found_count'] = int((found & ~native).sum())
        for metric, (column, unit) in METRICS.items():
            values = frame.loc[found, column].to_numpy(float)
            _require(np.isfinite(values).all() and (values >= 0).all()
                     and frame.loc[~found, column].isna().all(),
                     f'{key}: invalid {metric} or unresolved value encoded as a number')
            saved = [float(row[f'{metric}_{stat}_{unit}']) for stat in ('median', 'min', 'max')]
            actual = [float(getattr(np, stat)(values)) if len(values) else np.nan
                      for stat in ('median', 'min', 'max')]
            _require(np.allclose(actual, saved, rtol=1e-10, atol=1e-8, equal_nan=True),
                     f'{key}: {metric} summary does not reproduce target records')
        roots = frame.loc[found, [f'finite_ka_equilibrium_{axis}_m' for axis in 'xyz']].to_numpy(float)
        target_coords = targets[found.to_numpy()]
        _require(np.isfinite(roots).all() and np.allclose(
            np.linalg.norm(roots - target_coords, axis=1) * 1e6,
            frame.loc[found, 'displacement_um'], rtol=1e-10, atol=1e-7),
            f'{key}: displacement does not match saved root/target coordinates')
        separations = [float(np.linalg.norm(a - b) * 1e6)
                       for j, a in enumerate(roots) for b in roots[j + 1:]]
        _require(not separations or min(separations) > 1,
                 f'{key}: multiple targets resolved to the same equilibrium')
        widths = []
        for (_, target), record in zip(frame.iterrows(), metadata, strict=True):
            value = target.get('root_search_half_width_a')
            if pd.isna(value):
                value = record['equilibrium_root_protocol']['search_half_width_a']
            widths.append(float(value))
        outcomes.loc[frame.index, 'root_search_half_width_a'] = widths
        outcomes.loc[frame.index, 'search_scope'] = np.where(native, 'native_width', 'expanded_search')
        outcomes.loc[frame.index[~found], 'search_scope'] = 'unresolved'
        command_audit.append(dict(setting_id=row.setting_id, method=row.method,
            phase_fingerprint=row.phase_fingerprint, command_path=row.command_path,
            command_file_fingerprint=fingerprintlib.fingerprint(command_path.read_bytes()).hexdigest(),
            candidate_id=row.candidate_id, source_count=source_count,
            root_found_count=int(found.sum()), native_width_root_found_count=int(native.sum()),
            expanded_root_found_count=int((found & ~native).sum()),
            minimum_root_separation_um=min(separations) if separations else None,
            data_scope=row.data_scope))

    _require(len(weights) == 10 and not weights[['setting_id', 'method']].duplicated().any(),
             'Resource weights require exactly 10 unique FE/GFE selections')
    for row in weights.itertuples(index=False):
        match = summary[summary.setting_id.eq(row.setting_id) & summary.method.eq(row.method)]
        _require(len(match) == 1 and row.method in ('FE', 'GFE'), 'Invalid resource weight key')
        selected = match.iloc[0]
        _require(str(row.command_fingerprint) == selected.phase_fingerprint
                 and float(row.alpha_force_per_m) == float(selected.alpha_force_per_m)
                 and int(row.source_count) == int(selected.source_count),
                 f'{row.setting_id} {row.method}: selected weights differ from active command')
    for column in ('source_count', 'native_width_root_found_count', 'expanded_root_found_count'):
        summary[column] = summary[column].astype(int)
    outcomes['source_count'] = outcomes.source_count.astype(int)
    unresolved = outcomes.loc[~_truth(outcomes.finite_ka_root_found),
                              ['setting_id', 'method', 'target_id']]
    expanded = outcomes.loc[outcomes.search_scope.eq('expanded_search'),
        ['setting_id', 'method', 'target_id', 'displacement_um',
         'pressure_abs_at_exact_equilibrium_pa', 'root_search_half_width_a']]
    provenance = dict(
        schema_version=1, task='Triple', cache_only=True,
        optimizers_called=0, force_evaluations_called=0, equilibrium_searches_called=0,
        command_count=len(summary), target_count=len(outcomes),
        resolved_target_count=int(_truth(outcomes.finite_ka_root_found).sum()),
        native_width_root_found_count=int(summary.native_width_root_found_count.sum()),
        expanded_root_found_count=int(summary.expanded_root_found_count.sum()),
        untouched_baseline_commands=45, activated_resource_commands=25,
        activated_settings=list(RESOURCE_SETTINGS), selection_rule=SELECTION_RULE,
        aggregation='Median and minimum/maximum over observed finite-ka elastic-sphere roots; no restoring filter.',
        search_disclosure=('The native search is +/-4 bead radii per Cartesian coordinate. '
            'Six selected roots were recovered by expanded searches and are explicitly '
            'identified in the target table. Search provenance is not a stability classification.'),
        weights_scope='Ten resource FE/GFE rows only; the unified selected-weight export contains all final Single/Triple weights.',
        baseline_inputs_immutable=True, source_hashes=source_hashes,
        unresolved_targets=_records(unresolved), expanded_targets=_records(expanded),
        commands=command_audit)
    return summary, outcomes, weights, provenance



def load_generated_benchmark(root):
    """Validate and aggregate explicitly selected generated populations.

    One summary point pools resolved roots across the declared paired seeds.
    The original command/target records and individual command hashes remain
    in the raw table. A pooled summary hash is never a phase-vector hash.
    """
    root = Path(root).resolve()
    marker_path = root / "data/generated_triple_source.json"
    marker = json.loads(marker_path.read_text())
    _require(marker.get("schema") == "selected-native-triple-generation-v1",
             "Unknown generated Triple selector schema")
    directory = (root / marker["relative_data_root"]).resolve()
    _require(directory.is_relative_to(root), "Generated Triple source escapes active project")
    source_hashes = {"selector": str(marker_path.relative_to(root))}
    for filename in ("generation_plan.json", "outcomes.csv", "command_index.csv"):
        path = directory / filename
        _require(path.is_file() and path.stat().st_size > 0,
                 f"Generated Triple {filename} is missing or empty")
        source_hashes[filename] = str(path.relative_to(root))
    plan = json.loads((directory / "generation_plan.json").read_text())
    _require(plan["mode"] == marker["mode"] and marker["evidence_mode"] == f'generated_{plan["mode"]}',
             "Generated Triple evidence mode mismatch")
    outcomes = pd.read_csv(directory / "outcomes.csv")
    settings = [setting["id"] for setting in plan["settings"]]
    seeds = list(map(int, plan["seeds"]))
    _require(set(outcomes.setting_id) == set(settings) and set(outcomes.method) == set(METHODS),
             "Generated Triple setting/method coverage differs from the generation plan")
    _require(len(outcomes) == len(settings)*len(METHODS)*len(seeds)*3 and
             not outcomes[["setting_id", "method", "seed", "target_index"]].duplicated().any(),
             "Generated Triple must retain three target rows for every method and paired seed")
    command_audit = []
    for (sid, method, seed), frame in outcomes.groupby(["setting_id", "method", "seed"], sort=False):
        frame = frame.sort_values("target_index")
        _require(list(frame.target_index.astype(int)) == [0, 1, 2], "Generated command target indices differ")
        _require(int(seed) in seeds and frame.phase_fingerprint.nunique() == frame.command_path.nunique() == 1,
                 "Generated command phase identity differs between target rows")
        path = (directory / frame.iloc[0].command_path).resolve()
        _require(path.is_relative_to(directory), "Generated command escapes generated data root")
        with np.load(path, allow_pickle=False) as saved:
            phase, targets = saved["phase_rad"], saved["targets_m"]
        _require(_digest_array(phase) == frame.iloc[0].phase_fingerprint and targets.shape == (3, 3),
                 "Generated command phase hash or targets differ")
        _require(np.allclose(targets, frame[[f"target_{axis}_m" for axis in "xyz"]], rtol=0, atol=1e-15),
                 "Generated mechanics target coordinates differ from the synthesized command")
        found, native = _truth(frame.finite_ka_root_found), _truth(frame.native_width_root_found)
        _require(not (native & ~found).any(), "A native root cannot be absent from observed roots")
        roots = frame.loc[found, [f"finite_ka_equilibrium_{axis}_m" for axis in "xyz"]].to_numpy(float)
        for column in ("displacement_m", "displacement_um", "pressure_abs_at_exact_equilibrium_pa"):
            values = frame.loc[found, column].to_numpy(float)
            _require(np.isfinite(values).all() and (values >= 0).all() and frame.loc[~found, column].isna().all(),
                     "Generated unresolved or resolved physical measurements are inconsistent")
        _require(np.allclose(np.linalg.norm(roots-targets[found.to_numpy()], axis=1),
            frame.loc[found, "displacement_m"], rtol=1e-10, atol=1e-13),
            "Generated displacement differs from equilibrium minus target")
        metadata = [json.loads(value) for value in frame.finite_ka_model_metadata_json]
        _require(all("elastic" in str(item["model_name"]).lower() and
            int(item["field"]["number_of_elements"]) == len(phase) for item in metadata),
            "Generated mechanics must use the declared elastic model and source count")
        mask = outcomes.index.isin(frame.index)
        outcomes.loc[mask, "command_path"] = str(path.relative_to(root))
        outcomes.loc[mask, "data_scope"] = marker["evidence_mode"]
        outcomes.loc[mask, "search_scope"] = np.where(native, "native_width", "expanded_search")
        outcomes.loc[frame.index[~found], "search_scope"] = "unresolved"
        command_audit.append(dict(setting_id=sid, method=method, seed=int(seed),
            command_path=str(path.relative_to(root)), phase_fingerprint=str(frame.iloc[0].phase_fingerprint),
            command_file_fingerprint=fingerprintlib.fingerprint(path.read_bytes()).hexdigest(), source_count=len(phase),
            root_found_count=int(found.sum()), native_width_root_found_count=int(native.sum()),
            expanded_root_found_count=int((found & ~native).sum()), data_scope=marker["evidence_mode"]))
    summaries = []
    for sid in settings:
        for method in METHODS:
            frame = outcomes[outcomes.setting_id.eq(sid) & outcomes.method.eq(method)]
            found, native = _truth(frame.finite_ka_root_found), _truth(frame.native_width_root_found)
            hashes = frame[["seed", "phase_fingerprint"]].drop_duplicates().sort_values("seed")
            identity = str(hashes.iloc[0].phase_fingerprint) if len(hashes) == 1 else (
                "population:" + fingerprintlib.fingerprint(hashes.to_csv(index=False).encode()).hexdigest())
            summary = dict(setting_id=sid, method=method, candidate_id=str(frame.iloc[0].candidate_id),
                alpha_force_per_m=float(frame.iloc[0].alpha_force_per_m), expected_targets=len(frame),
                root_found_count=int(found.sum()), native_width_root_found_count=int(native.sum()),
                expanded_root_found_count=int((found & ~native).sum()), source_count=int(frame.iloc[0].source_count),
                paired_starts=len(seeds), phase_fingerprint=identity,
                identity_kind="phase_digest" if len(hashes) == 1 else "population_command_index_digest",
                command_path=str(frame.iloc[0].command_path) if len(hashes) == 1 else str((directory / "command_index.csv").relative_to(root)),
                data_scope=marker["evidence_mode"],
                aggregation="Median/min/max over resolved numerical roots across all declared seeds and all three targets")
            for column, stem, unit in (("displacement_um", "displacement", "um"),
                                       ("displacement_m", "displacement", "m"),
                                       ("pressure_abs_at_exact_equilibrium_pa", "pressure", "pa")):
                values = frame.loc[found, column].to_numpy(float)
                for statistic in ("median", "min", "max"):
                    summary[f"{stem}_{statistic}_{unit}"] = float(getattr(np, statistic)(values)) if len(values) else np.nan
            summaries.append(summary)
    summary = pd.DataFrame(summaries)
    weights = summary[summary.setting_id.isin(RESOURCE_SETTINGS) & summary.method.isin(("FE", "GFE"))][
        ["setting_id", "method", "source_count", "alpha_force_per_m", "phase_fingerprint"]].copy()
    weights = weights.rename(columns={"phase_fingerprint":"command_fingerprint"})
    expanded = outcomes.loc[outcomes.search_scope.eq("expanded_search"),
        ["setting_id", "method", "seed", "target_id", "displacement_um", "pressure_abs_at_exact_equilibrium_pa", "root_search_half_width_a"]]
    provenance = dict(schema_version=2, task="Triple", cache_only=False,
        evidence_mode=marker["evidence_mode"], mode=plan["mode"], seeds=seeds,
        command_count=len(command_audit), summary_point_count=len(summary), target_count=len(outcomes),
        resolved_target_count=int(_truth(outcomes.finite_ka_root_found).sum()),
        native_width_root_found_count=int(summary.native_width_root_found_count.sum()),
        expanded_root_found_count=int(summary.expanded_root_found_count.sum()),
        selection_rule=plan["selection"], aggregation="All paired seeds and three targets per command; medians and min/max over found elastic roots; missing roots retained.",
        search_disclosure="Native Cartesian search within +/-4 bead radii; deposited low-frequency interventions allow unresolved searches to expand to +/-8 bead radii. Full mode uses the declared native lmax=8 validation protocol.",
        source_hashes=source_hashes, commands=command_audit,
        expanded_targets=_records(expanded), archived_outcomes_used=False,
        loader_work="Read-only aggregation and identity verification; optimization occurred in the explicit data-generation stage.")
    return summary, outcomes, weights, provenance

def export_final_benchmark(output=None, root=None):
    """Write the auditable final cached tables; do not modify any source table."""
    root = Path(root).resolve() if root is not None else ROOT
    output = Path(output).resolve() if output is not None else root / OUTPUT
    summary, outcomes, weights, provenance = load_final_benchmark(root)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (('summary', summary), ('outcomes', outcomes),
                        ('resource_weights', weights)):
        write_bytes(output / f'{name}.csv', frame.to_csv(index=False).encode())
    write_bytes(output / 'provenance.json',
                (json.dumps(provenance, indent=2, allow_nan=False) + '\n').encode())
    return provenance


if __name__ == '__main__':
    result = export_final_benchmark()
    print(json.dumps({key: result[key] for key in (
        'command_count', 'target_count', 'resolved_target_count',
        'native_width_root_found_count', 'expanded_root_found_count')}, indent=2))
