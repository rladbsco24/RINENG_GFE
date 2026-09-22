"""Paired force-field comparison with compact FE commands.

The nine deposited target/seed selections are preserved. FE commands are
regenerated with compact L-BFGS-B; matched Conventional commands are reused.
Both independent force models include the same effective gravity.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from rineng_content_id import content_identity as content_id
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from . import appendix_a as previous
from . import exact_validator as ev

SCHEMA = 'appendix-a-paired-force-fields-v1'
BOOTSTRAP_SEED = 20260906
BOOTSTRAP_RESAMPLES = 10000
FORCE_FLOOR_WEIGHT_FRACTION = 0.01
NEIGHBOR_RADII_A = (0.5, 1.0, 2.0)
PRESSURE_SHELL_RADII_A = (0.5, 1.0)
PRESSURE_MIN_SEARCH_HALF_WIDTH_A = 2.0
PRESSURE_INTENSITY_REL_TOL = 1e-6
FIELD_HALF_WIDTH_A = 2.5
FIELD_QUIVER_POINTS = 13
FIELD_PRESSURE_POINTS = 81
COLORS = {'FE': '#B57825', 'Conventional': '#266C9C'}


def _directions(all_neighbors=False):
    values = np.asarray([[i, j, k] for i in (-1, 0, 1)
                         for j in (-1, 0, 1) for k in (-1, 0, 1)
                         if (i or j or k) and (all_neighbors or
                         (abs(i) + abs(j) + abs(k) in (1, 3)))], dtype=float)
    return values / np.linalg.norm(values, axis=1)[:, None]


def _write_json(path, value):
    previous._write_json(path, value)


def _key(value):
    return content_id(json.dumps(previous._jsonable(value), sort_keys=True).encode()).hexdigest()


def _load_inputs(root, *, cache_only=False):
    cases, provenance, positions, normals = previous._inputs(root, refresh=False)
    selected = sorted((c for c in cases if c['selection_role'] == 'B1 canonical medoid'),
                      key=lambda c: c['target_id'])
    assert len(selected) == 9 and len({c['target_id'] for c in selected}) == 9
    selected = [previous._refresh_fe_case(root, case, positions, normals, provenance['source'], cache_only=cache_only)
                for case in selected]
    source = root / 'runtime/data/corrected_branch_evolution'
    trajectories = pd.read_csv(source / 'conventional_trajectories.csv')
    with np.load(source / 'conventional_snapshots.npz', allow_pickle=False) as data:
        snapshots = data['snapshots'].copy()
        checkpoint = list(data['checkpoints']).index(10000)
    paired = []
    for case in selected:
        match = trajectories[(trajectories.target_id == case['target_id']) &
                             (trajectories.seed == case['seed'])]
        if len(match) != 1:
            raise ValueError(f"Expected one same-seed Conventional endpoint: {case['case_id']}")
        record = match.iloc[0]
        target = record[['target_x_m', 'target_y_m', 'target_z_m']].to_numpy(dtype=float)
        np.testing.assert_array_equal(target, case['target_m'])
        paired.append({**case, 'method': 'FE', 'source_case_id': case['case_id'],
            'source_path': case['source_path'], 'source_iteration_cap': case['source_iteration_cap']})
        phase = snapshots[int(record.trajectory_index), checkpoint]
        paired.append({**case, 'case_id': case['case_id'] + '_Conventional',
            'source_case_id': str(record.trajectory_id), 'phase_rad': phase,
            'phase_content_id': previous._hash_array(phase), 'method': 'Conventional',
            'new_command_optimization': False, 'optimizer': 'BFGS',
            'evaluator_backend': 'native', 'fe_solver_policy': None,
            'source_path': str((source / 'conventional_snapshots.npz').relative_to(root)),
            'source_state_index': int(record.trajectory_index), 'source_iteration_cap': 10000,
            'source_terminal_iterations': int(record.nit),
            'source_terminal_gradient_rms': float(record.phase_gradient_rms),
            'source_terminal_status': int(record.status),
            'selection_role': 'Same-target same-seed Conventional endpoint'})
    provenance = {**provenance, 'schema': SCHEMA,
        'scope': 'Eighteen commands at nine paired targets; FE uses compact L-BFGS-B and matched Conventional commands are reused.',
        'comparison': 'Total-force direction on common 3D neighborhoods and independently resolved Gor\'kov-plus-gravity versus elastic-plus-gravity root separation. Force magnitudes, pressure minima and per-model restoring flags are separately exported.',
        'sample_selection': 'Nine recorded B1 target/seed selections re-solved with compact L-BFGS-B, paired with same-target/same-seed Conventional 10000 endpoints.',
        'statistical_unit': 'Target; nine paired target clusters. The other 18 selected FE cases are not extra independent replicates for this paired comparison.',
        'generality': ('This Appendix A protocol uses one deposited 40 kHz array/particle/drive regime; '
                       'the separate geometry/frequency study is active in Appendix G1-G9.'),
        'command_scope': 'FE retains appendix alpha=9, beta=0, W=I and uses compact L-BFGS-B; Conventional retains its original specification.',
        'source_files': {str(path.relative_to(root)): previous._hash_file(path) for path in
                         (source / 'conventional_snapshots.npz', source / 'conventional_trajectories.csv')},
        'neighbor_directions': _directions().tolist(), 'neighbor_radii_a': NEIGHBOR_RADII_A,
        'direction_threshold': {'both_total_force_norms_exceed_effective_weight_fraction': FORCE_FLOOR_WEIGHT_FRACTION,
                                'zeros_have_no_direction': True},
        'pressure_minimum': {'search_half_width_a': PRESSURE_MIN_SEARCH_HALF_WIDTH_A,
            'starts_a': ev.common_cartesian_root_starts((1.0,)).tolist(),
            'least_squares': {'method': 'trf', 'jac': '3-point', 'ftol': 1e-10, 'xtol': 1e-10, 'gtol': 1e-10, 'max_nfev': 150},
            'sampled_shell_radii_a': PRESSURE_SHELL_RADII_A,
            'shell_directions': _directions(True).tolist(),
            'intensity_tolerance_fraction_of_shell_mean': PRESSURE_INTENSITY_REL_TOL,
            'criterion': 'Interior candidate with intensity no greater than all 26 shell samples at both radii, within relative tolerance. This is a sampled local minimum; nonisolated pressure-zero minima are allowed.',
            'candidate_selection': 'Among candidates passing the sampled criterion, choose closest to prescribed target; otherwise choose smallest normalized intensity.',
            'root_to_pressure_distance': 'Distance to that selected pressure candidate, not the nearest distance from a model root to a potentially nonisolated pressure-minimum set.',
            'target_contrast': 'I(target) / mean I(shell at 1a)',
            'nearby_minimum_distinct_from_target_minimum': True},
        'field_display': {'half_width_a': FIELD_HALF_WIDTH_A, 'quiver_grid_points_per_axis': FIELD_QUIVER_POINTS,
            'pressure_grid_points_per_axis': FIELD_PRESSURE_POINTS,
            'plane': 'XZ at prescribed target y; 3D roots projected onto this plane',
            'representative_rule': 'Central selected FE target/seed case, then the selected FE case with largest valid model-root separation; ties by case_id.',
            'arrow_encoding': 'Unit XZ-projected total-force direction, fixed length; pressure magnitude uses one shared color scale. 3D angular errors and magnitude ratios are reported separately.'},
        'bootstrap': {'seed': BOOTSTRAP_SEED, 'resamples': BOOTSTRAP_RESAMPLES, 'interval_percentiles': [2.5, 97.5],
            'unit': 'Paired target, never neighboring force samples', 'summary': 'Median of within-target Conventional-minus-FE differences',
            'scope': 'Finite deposited target design; not a population-level generality claim'},
        'root_comparison': 'Distances computed only with both independently resolved roots; restoring flags shown and retained separately. All 18 commands remain in denominators.',
        'gorkov_comparator': 'Existing inviscid elastic Rayleigh limit with regular-wave derivatives; same effective gravity as finite-ka elastic reference.',
    }
    provenance.pop('signed_discrepancy', None)
    provenance.pop('gorkov_numerics', None)
    return paired, cases, provenance, positions, normals


def _pressure_properties(field, target, radius):
    directions = _directions(True)
    target = np.asarray(target)
    intensity = lambda points: np.abs(field.pressure(points)) ** 2
    shell = intensity(target + radius * directions)
    scale = max(float(np.mean(shell)), 1e-30)
    def sampled(point):
        center_i = float(intensity(point))
        details = []
        for factor in PRESSURE_SHELL_RADII_A:
            values = intensity(point + factor * radius * directions)
            details.append(center_i <= float(values.min()) + PRESSURE_INTENSITY_REL_TOL * max(float(values.mean()), 1e-30))
        return all(details)
    def residual(offset):
        value = complex(field.pressure(target + radius * offset)) / np.sqrt(scale)
        return np.array([value.real, value.imag])
    candidates = []
    for start in ev.common_cartesian_root_starts((1.0,)):
        solved = least_squares(residual, start, bounds=(-PRESSURE_MIN_SEARCH_HALF_WIDTH_A, PRESSURE_MIN_SEARCH_HALF_WIDTH_A),
            jac='3-point', ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=150)
        point = target + radius * solved.x
        interior = bool(np.max(np.abs(solved.x)) < PRESSURE_MIN_SEARCH_HALF_WIDTH_A - 1e-5)
        candidates.append(dict(point_m=point, offset_a=float(np.linalg.norm(solved.x)),
            contrast=float(np.sum(residual(solved.x)**2)), sampled_minimum=bool(interior and sampled(point)),
            interior=interior, solver_status=int(solved.status), nfev=int(solved.nfev)))
    good = [c for c in candidates if c['sampled_minimum']]
    chosen = min(good, key=lambda c: (c['offset_a'], c['contrast'])) if good else min(candidates, key=lambda c: c['contrast'])
    row = dict(target_pressure_pa=float(np.sqrt(float(intensity(target)))), target_intensity_shell_ratio=float(intensity(target))/scale,
        target_centered_sampled_pressure_minimum=sampled(target), nearby_sampled_pressure_minimum=chosen['sampled_minimum'],
        pressure_minimum_offset_a=chosen['offset_a'], pressure_minimum_contrast=chosen['contrast'],
        pressure_minimum_x_m=chosen['point_m'][0], pressure_minimum_y_m=chosen['point_m'][1], pressure_minimum_z_m=chosen['point_m'][2])
    return row, candidates


def _mechanics_key(case, provenance):
    return _key(dict(schema=SCHEMA, phase=case['phase_content_id'], target=case['target_m'],
        numerics=asdict(previous.NUMERICS), root_protocol=previous.ROOT_PROTOCOL,
        physics_content_id=previous._hash_file(Path(ev.__file__)), source=provenance['source'],
        pressure_protocol=provenance['pressure_minimum'], source_files=provenance['source_files']))


def _mechanics(case, provenance, positions, normals, output):
    cache_key = _mechanics_key(case, provenance)
    directory = output / 'cache' / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    record_path, arrays_path = directory / 'record.json', directory / 'arrays.npz'
    if record_path.exists() and arrays_path.exists():
        saved = json.loads(record_path.read_text())
        if saved['npz_content_id'] == previous._hash_file(arrays_path):
            with np.load(arrays_path, allow_pickle=False) as data:
                return saved['row'], {k: data[k] for k in data.files}, saved['root_candidates'], True
    started = time.perf_counter()
    elastic, _ = previous._evaluators(case, provenance, positions, normals)
    gorkov = previous._projected_comparator(elastic.field)
    candidates = []
    starts = ev.common_cartesian_root_starts((1.0,))
    result = ev.validate_static_trap(elastic, case['target_m'], initial_offsets_a=starts, **previous.ROOT_PROTOCOL)
    er = result.equilibrium
    gr = ev.find_equilibrium(gorkov, case['target_m'], initial_offsets_a=starts,
        **{k: v for k, v in previous.ROOT_PROTOCOL.items() if k != 'jacobian_step_a'})
    e_root, g_root = er.equilibrium_m, gr.equilibrium_m
    e_found, g_found = er.numerical_root_found, gr.numerical_root_found
    e_eigenvalues = result.symmetric_stiffness_eigenvalues_n_m
    e_residual, g_residual = er.residual_force_n, gr.residual_force_n
    e_boundary = er.on_search_boundary
    e_scaled, g_scaled = er.residual_scaled_norm, gr.residual_scaled_norm
    candidates = [dict(case_id=case['case_id'], model=name, **asdict(c))
        for name, solved in [('elastic', er), ('gorkov', gr)] for c in solved.candidates]
    g_jac = ev.force_jacobian(gorkov, g_root, previous.ROOT_PROTOCOL['jacobian_step_a'] * elastic.sphere.radius_m)
    g_eigenvalues = np.linalg.eigvalsh(-.5*(g_jac + g_jac.T))
    radius = elastic.sphere.radius_m
    pressure, pressure_candidates = _pressure_properties(elastic.field, case['target_m'], radius)
    row = dict(case_id=case['case_id'], target_id=case['target_id'], method=case['method'], seed=case['seed'],
        source_case_id=case['source_case_id'], phase_content_id=case['phase_content_id'], source_path=case['source_path'],
        source_iteration_cap=case['source_iteration_cap'], source_terminal_iterations=case.get('source_terminal_iterations', np.nan),
        target_x_m=case['target_m'][0], target_y_m=case['target_m'][1], target_z_m=case['target_m'][2],
        elastic_root_found=bool(e_found), gorkov_root_found=bool(g_found), elastic_root_on_boundary=bool(e_boundary),
        elastic_root_residual_scaled=float(e_scaled), gorkov_root_residual_scaled=float(g_scaled),
        elastic_restoring=bool(e_found and np.min(e_eigenvalues)>0),
        gorkov_restoring=bool(g_found and np.min(g_eigenvalues)>0),
        elastic_min_stiffness_n_m=float(np.min(e_eigenvalues)), gorkov_min_stiffness_n_m=float(np.min(g_eigenvalues)),
        elastic_target_offset_a=float(np.linalg.norm(e_root-case['target_m'])/radius) if e_found else np.nan,
        gorkov_target_offset_a=float(np.linalg.norm(g_root-case['target_m'])/radius) if g_found else np.nan,
        model_root_separation_a=float(np.linalg.norm(e_root-g_root)/radius) if e_found and g_found else np.nan,
        radius_m=radius, effective_weight_n=elastic.effective_weight_n, validation_wall_s=time.perf_counter()-started,
        new_command_optimization=bool(case.get('new_command_optimization', False)), **pressure)
    arrays = dict(phase_rad=case['phase_rad'], target_m=case['target_m'], elastic_root_m=e_root,
        gorkov_root_m=g_root, elastic_residual_n=e_residual, gorkov_residual_n=g_residual,
        elastic_stiffness_eigenvalues_n_m=e_eigenvalues, gorkov_stiffness_eigenvalues_n_m=g_eigenvalues,
        pressure_minimum_m=np.array([row['pressure_minimum_x_m'],row['pressure_minimum_y_m'],row['pressure_minimum_z_m']]))
    np.savez_compressed(arrays_path, **arrays)
    _write_json(record_path, dict(cache_key=cache_key, row=row, root_candidates=candidates,
        pressure_candidates=pressure_candidates, npz_content_id=previous._hash_file(arrays_path)))
    print(f"A paired mechanics {case['case_id']}: elastic={e_found}, Gorkov={g_found}, root separation/a={row['model_root_separation_a']:.3g}", flush=True)
    return previous._jsonable(row), arrays, previous._jsonable(candidates), False


def _force_samples(case, row, provenance, positions, normals, output, display=False):
    if display:
        q = np.linspace(-FIELD_HALF_WIDTH_A, FIELD_HALF_WIDTH_A, FIELD_QUIVER_POINTS)
        x, z = np.meshgrid(q, q)
        offsets = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
    else:
        offsets = np.vstack([factor * _directions() for factor in NEIGHBOR_RADII_A])
    points = np.asarray(case['target_m']) + row['radius_m'] * offsets
    cache_key = _key(dict(schema=SCHEMA, phase=case['phase_content_id'], points=points,
        numerics=asdict(previous.NUMERICS), source=provenance['source'],
        physics_content_id=previous._hash_file(Path(ev.__file__))))
    path = output / 'cache' / ('force_' + cache_key + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        np.testing.assert_array_equal(arrays['points_m'], points)
        return arrays, True
    elastic, _ = previous._evaluators(case, provenance, positions, normals)
    gorkov = previous._projected_comparator(elastic.field)
    e_forces, g_forces = [], []
    for point in points:
        e_forces.append(elastic.force(point).total_n)
        g_forces.append(gorkov.force(point).total_n)
    arrays = dict(points_m=points, offsets_a=offsets, elastic_total_force_n=np.asarray(e_forces),
        gorkov_total_force_n=np.asarray(g_forces), pressure_pa=elastic.field.pressure(points))
    np.savez_compressed(path, **arrays)
    print(f"A force samples {case['case_id']}: {len(points)} common points", flush=True)
    return arrays, False


def _direction_rows(case, row, values):
    e = values['elastic_total_force_n']; g = values['gorkov_total_force_n']
    en = np.linalg.norm(e, axis=1); gn = np.linalg.norm(g, axis=1)
    threshold = FORCE_FLOOR_WEIGHT_FRACTION * row['effective_weight_n']
    valid = (en > threshold) & (gn > threshold)
    angle = np.full(len(en), np.nan)
    angle[valid] = np.degrees(np.arccos(np.clip(np.sum(e[valid]*g[valid],axis=1)/(en[valid]*gn[valid]),-1,1)))
    ratio = np.divide(gn, en, out=np.full(len(en),np.nan), where=valid)
    data = []
    for i in range(len(en)):
        data.append(dict(case_id=case['case_id'], target_id=case['target_id'], method=case['method'], point_index=i,
            x_m=values['points_m'][i,0], y_m=values['points_m'][i,1], z_m=values['points_m'][i,2],
            radius_a=float(np.linalg.norm(values['offsets_a'][i])), direction_defined=bool(valid[i]),
            force_floor_n=threshold, angular_error_deg=angle[i], magnitude_ratio_gorkov_elastic=ratio[i],
            elastic_force_norm_n=en[i], gorkov_force_norm_n=gn[i],
            **{f'{name}_force_{axis}_n': forces[i,j] for name, forces in [('elastic',e),('gorkov',g)] for j,axis in enumerate('xyz')}))
    return data, dict(direction_points_total=len(en), direction_points_valid=int(sum(valid)),
        angular_error_median_deg=float(np.nanmedian(angle)), angular_error_q25_deg=float(np.nanpercentile(angle,25)),
        angular_error_q75_deg=float(np.nanpercentile(angle,75)), angular_error_p90_deg=float(np.nanpercentile(angle,90)),
        magnitude_ratio_median=float(np.nanmedian(ratio)))


def _statistics(frame):
    random = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for metric in ('angular_error_median_deg', 'model_root_separation_a', 'magnitude_ratio_median', 'elastic_target_offset_a'):
        wide = frame.pivot(index='target_id', columns='method', values=metric).sort_index()
        available = wide.dropna()
        diffs = available.Conventional.to_numpy()-available.FE.to_numpy()
        if not len(diffs): continue
        sampled = random.integers(0, len(diffs), size=(BOOTSTRAP_RESAMPLES, len(diffs)))
        boot = np.median(diffs[sampled], axis=1)
        results.append(dict(metric=metric, statistic='median paired Conventional minus FE', paired_targets=len(diffs),
            total_targets=len(wide), estimate=float(np.median(diffs)), ci_low=float(np.percentile(boot,2.5)),
            ci_high=float(np.percentile(boot,97.5)), conventional_lower_count=int(sum(diffs<0)),
            fe_median=float(available.FE.median()), conventional_median=float(available.Conventional.median()),
            bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            scope='Nine fixed target clusters; all force neighbors belong to their command'))
    return pd.DataFrame(results)


def _save(fig, output, stem):
    paths=[]
    for ext in ('png','pdf','svg'):
        path=output/(stem+'.'+ext);fig.savefig(path,dpi=220,bbox_inches='tight')
        if ext=='png':paths.append(path)
    plt.close(fig)
    return paths[0]


def _figures(cases, frame, arrays, fields, representatives, provenance, positions, normals, output, *, cached_pressure=None):
    style={'font.size':13,'font.weight':'bold','axes.labelweight':'bold','axes.labelsize':14,
        'axes.linewidth':1.7,'xtick.labelsize':12,'ytick.labelsize':12,'xtick.major.width':1.5,
        'ytick.major.width':1.5,'legend.fontsize':11,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'}
    frame = frame.copy()
    frame['model_root_separation_m'] = frame.model_root_separation_a * frame.radius_m
    paths=[];plot_rows=[]
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,2,figsize=(10.8,10.4))
        fig.subplots_adjust(left=.12,right=.865,bottom=.13,top=.91,wspace=.34,hspace=.30)
        scalar=np.linspace(-FIELD_HALF_WIDTH_A,FIELD_HALF_WIDTH_A,FIELD_PRESSURE_POINTS)
        xx,zz=np.meshgrid(scalar,scalar); backgrounds=[]
        for representative_index, index in enumerate(representatives):
            if cached_pressure is not None:
                background = np.asarray(cached_pressure[representative_index])
                if background.shape != xx.shape:
                    raise ValueError('Cached pressure grid does not match the declared display grid')
                backgrounds.append(background)
                continue
            case=cases[index];radius=frame.iloc[index].radius_m
            points=case['target_m']+radius*np.stack([xx,np.zeros_like(xx),zz],axis=-1)
            elastic,_=previous._evaluators(case,provenance,positions,normals)
            backgrounds.append(np.abs(elastic.field.pressure(points)))
        maximum=max(float(p.max()) for p in backgrounds)
        for r,index in enumerate(representatives):
            case=cases[index];row=frame.iloc[index];values=fields[index];mech=arrays[index]
            for c,(name,key) in enumerate([("Gor’kov + g",'gorkov_total_force_n'),('Elastic + g','elastic_total_force_n')]):
                ax=axes[r,c]
                im=ax.pcolormesh(xx*row.radius_m,zz*row.radius_m,backgrounds[r],cmap='magma',vmin=0,vmax=maximum,shading='auto',rasterized=True)
                force=values[key]; norm=np.linalg.norm(force[:,[0,2]],axis=1)
                valid=(np.linalg.norm(force,axis=1)>FORCE_FLOOR_WEIGHT_FRACTION*row.effective_weight_n)&(norm>FORCE_FLOOR_WEIGHT_FRACTION*row.effective_weight_n)
                direction=np.divide(force[:,[0,2]],norm[:,None],out=np.zeros((len(force),2)),where=valid[:,None])
                offsets=values['offsets_a']
                ax.quiver(offsets[valid,0]*row.radius_m,offsets[valid,2]*row.radius_m,direction[valid,0],direction[valid,1],angles='xy',scale_units='xy',scale=4.8/row.radius_m,width=.0045,color='white',pivot='mid',headwidth=3.9,headlength=4.6)
                ax.scatter(0,0,marker='+',s=100,c='#60DBF2',linewidths=2,zorder=6)
                for root_key,found,marker,color in [('gorkov_root_m',row.gorkov_root_found,'o','#75D6FF'),('elastic_root_m',row.elastic_root_found,'s','#9EFF85')]:
                    if found:
                        off=mech[root_key]-case['target_m']
                        ax.scatter(off[0],off[2],marker=marker,s=90,facecolors='none',edgecolors=color,linewidths=2.1,zorder=7)
                half_width_m = FIELD_HALF_WIDTH_A * row.radius_m
                ax.set(xlabel=r'$\Delta x$ (m)',ylabel=r'$\Delta z$ (m)',xlim=(-half_width_m,half_width_m),ylim=(-half_width_m,half_width_m))
                ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
                ax.set_xticks(np.asarray([-1.,0.,1.])*1e-3)
                ax.set_yticks(np.asarray([-1.,0.,1.])*1e-3)
                ax.set_aspect('equal')
                if r == 0: ax.set_title(name,pad=12,fontsize=14)
                ax.text(-.16,1.08,f'({chr(97+r*2+c)})',transform=ax.transAxes,fontsize=16,fontweight='bold')
                for k in range(len(force)):
                    plot_rows.append(dict(panel=chr(97+r*2+c),case_id=case['case_id'],model=name,point_index=k,x_a=offsets[k,0],z_a=offsets[k,2],x_m=offsets[k,0]*row.radius_m,z_m=offsets[k,2]*row.radius_m,
                        direction_displayed=bool(valid[k]),arrow_x=direction[k,0],arrow_z=direction[k,1],force_x_n=force[k,0],force_y_n=force[k,1],force_z_n=force[k,2]))
        cax=fig.add_axes([.905,.26,.022,.48]);cb=fig.colorbar(im,cax=cax);cb.set_label(r'$|p|$ (Pa)',fontsize=14)
        handles=[Line2D([],[],marker='+',color='#159AAE',linestyle='none',markersize=10,markeredgewidth=2,label='Target'),
            Line2D([],[],marker='o',color='#3392C0',markerfacecolor='none',linestyle='none',markersize=8,markeredgewidth=2,label="Gor’kov root"),
            Line2D([],[],marker='s',color='#438E30',markerfacecolor='none',linestyle='none',markersize=8,markeredgewidth=2,label='Elastic root')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.48,.025),ncol=3,frameon=False)
        paths.append(_save(fig,output,'Figure_A1_total_force_fields'))
        pd.DataFrame(plot_rows).to_csv(output/'A1_quiver_plot_data.csv',index=False)
        np.savez_compressed(output/'A1_pressure_plot_data.npz',offset_x_a=xx,offset_z_a=zz,
            offset_x_m=np.stack([xx*frame.iloc[i].radius_m for i in representatives]),
            offset_z_m=np.stack([zz*frame.iloc[i].radius_m for i in representatives]),pressure_pa=np.stack(backgrounds),
            representative_case_ids=np.array([cases[i]['case_id'] for i in representatives]),color_scale_pa=np.array([0.,maximum]))
        fig,axes=plt.subplots(1,2,figsize=(11.1,5.0))
        fig.subplots_adjust(left=.10,right=.975,bottom=.16,top=.88,wspace=.43)
        plotted=[]
        for j,(metric,ylabel,title) in enumerate([
            ('angular_error_median_deg','Median angular error (°)','Local force directions'),
            ('model_root_separation_m','Root separation (m)','Independent equilibria')]):
            ax=axes[j]
            for k,(target,group) in enumerate(frame.groupby('target_id',sort=True)):
                group=group.set_index('method');shift=(k-4)*.017
                vals=[group.loc[m,metric] for m in ('FE','Conventional')]
                if np.all(np.isfinite(vals)):ax.plot(np.array([0,1])+shift,vals,c='.73',lw=1.2,zorder=1)
                for x,method,value in zip((0,1),('FE','Conventional'),vals):
                    row=group.loc[method]
                    marker='o' if row.elastic_restoring and row.gorkov_restoring else 'x'
                    if np.isfinite(value):ax.scatter(x+shift,value,s=65,c=COLORS[method],marker=marker,edgecolors='#222' if marker=='o' else None,linewidths=1.2,zorder=3)
                    plotted.append(dict(panel=chr(97+j),target_id=target,case_id=row.case_id,method=method,metric=metric,x=x+shift,value=value,
                        elastic_root_found=row.elastic_root_found,gorkov_root_found=row.gorkov_root_found,
                        elastic_restoring=row.elastic_restoring,gorkov_restoring=row.gorkov_restoring))
            for x,method in enumerate(('FE','Conventional')):
                values=frame.loc[frame.method==method,metric].dropna().to_numpy()
                if len(values):
                    low,mid,high=np.percentile(values,[25,50,75])
                    ax.errorbar(x+.17,mid,yerr=[[mid-low],[high-mid]],fmt='_',color=COLORS[method],capsize=5,elinewidth=2.5,markersize=16,markeredgewidth=2.8,zorder=5)
            ax.set(xlim=(-.32,1.38),xticks=(0,1),xticklabels=('FE','Conventional'),ylabel=ylabel)
            ax.set_ylim(bottom=0);ax.set_box_aspect(1)
            if metric.endswith('_m'): ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
            ax.text(-.19,1.08,f'({chr(97+j)})',transform=ax.transAxes,fontsize=16,fontweight='bold')
            ax.grid(axis='y',alpha=.18,lw=.8);ax.set_axisbelow(True)
        found=int((frame.elastic_root_found&frame.gorkov_root_found).sum())
        restoring=int((frame.elastic_restoring&frame.gorkov_restoring).sum())
        minima=int(frame.nearby_sampled_pressure_minimum.sum())
        paths.append(_save(fig,output,'Figure_A2_paired_force_and_root_agreement'))
        pd.DataFrame(plotted).to_csv(output/'A2_paired_plot_data.csv',index=False)
    return paths


def run(root, ctx=None, *, workers=2):
    """Replay or compute the paired comparison and export its complete inputs."""
    root=Path(root).resolve();output=root/'appendix_outputs/A_GFE';output.mkdir(parents=True,exist_ok=True)
    cases,archival,provenance,positions,normals=_load_inputs(root, cache_only=bool(ctx is not None and ctx.config.cache_only))
    _write_json(output/'A_protocol.json',provenance)
    inputs=[]
    for case in cases:
        inputs.append({k:case.get(k) for k in ('case_id','target_id','method','seed','source_case_id','source_state_index','source_path','phase_content_id','source_iteration_cap','selection_role')})
    pd.DataFrame(inputs).to_csv(output/'A_paired_command_selection.csv',index=False)
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            evaluated=list(pool.map(lambda c:_mechanics(c,provenance,positions,normals,output),cases))
        rows,arrays,candidates,hits=map(list,zip(*evaluated))
        for row,values in zip(rows,arrays):
            for model in ('elastic','gorkov'):
                row[model+'_root_to_selected_pressure_minimum_a'] = (
                    float(np.linalg.norm(values[model+'_root_m']-values['pressure_minimum_m'])/row['radius_m'])
                    if row[model+'_root_found'] and row['nearby_sampled_pressure_minimum'] else np.nan)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            force=list(pool.map(lambda item:_force_samples(item[0],item[1],provenance,positions,normals,output),zip(cases,rows)))
        force_rows=[]
        for case,row,(values,_) in zip(cases,rows,force):
            samples,summary=_direction_rows(case,row,values);force_rows.extend(samples);row.update(summary)
        frame=pd.DataFrame(rows)
        indices=[i for i,c in enumerate(cases) if c['method']=='FE']
        center=min(indices,key=lambda i:(np.linalg.norm(cases[i]['target_m'][:2]),cases[i]['case_id']))
        valid=[i for i in indices if np.isfinite(frame.iloc[i].model_root_separation_a)]
        worst=min(valid,key=lambda i:(-frame.iloc[i].model_root_separation_a,cases[i]['case_id']))
        representatives=[center,worst]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            displayed=list(pool.map(lambda i:(i,_force_samples(cases[i],rows[i],provenance,positions,normals,output,display=True)),representatives))
    fields={i:result[0] for i,result in displayed}
    for column in list(frame.columns):
        if column.endswith('_a') and any(word in column for word in ('offset', 'separation', 'distance', 'minimum')):
            frame[column[:-2] + '_m'] = frame[column] * frame.radius_m
    frame.to_csv(output/'A_paired_endpoint_results.csv',index=False)
    pd.DataFrame(force_rows).to_csv(output/'A_total_force_samples.csv',index=False)
    stats=_statistics(frame);stats.to_csv(output/'A_paired_bootstrap.csv',index=False)
    np.savez_compressed(output/'A_paired_commands_and_roots.npz',case_ids=np.array([c['case_id'] for c in cases]),
        **{key:np.stack([values[key] for values in arrays]) for key in arrays[0]},positions_m=positions,normals=normals)
    _write_json(output/'A_new_root_candidates.json',[c for group in candidates for c in group])
    figures=_figures(cases,frame,arrays,fields,representatives,provenance,positions,normals,output)
    summary=dict(schema=SCHEMA,evidence_status='SMOKE',paired_targets=9,commands=18,
        selected_fe_commands=9,new_command_optimizations=sum(bool(c.get('new_command_optimization', False)) for c in cases if c['method']=='FE'),mechanics_cache_hits=sum(hits),
        force_grid_cache_hits=sum(x[1] for x in force)+sum(x[1][1] for x in displayed),
        elastic_roots=int(frame.elastic_root_found.sum()),gorkov_roots=int(frame.gorkov_root_found.sum()),
        both_restoring=int((frame.elastic_restoring&frame.gorkov_restoring).sum()),
        nearby_pressure_minima=int(frame.nearby_sampled_pressure_minimum.sum()),
        target_centered_pressure_minima=int(frame.target_centered_sampled_pressure_minimum.sum()),
        representative_case_ids=[cases[i]['case_id'] for i in representatives])
    _write_json(output/'A_summary.json',summary)
    measured=["Nine paired targets, 18 commands; FE uses compact L-BFGS-B and matched Conventional commands are reused.",
        "Each preselected FE target/seed case is re-solved and paired with its same-target, same-seed Conventional 10000 endpoint.",
        f"Independent total-force roots: elastic {summary['elastic_roots']}/18; Gor'kov {summary['gorkov_roots']}/18; both locally restoring {summary['both_restoring']}/18."]
    for row in stats.itertuples(index=False):
        measured.append(f"{row.metric}: FE median {row.fe_median:.6g}; Conventional median {row.conventional_median:.6g}; median paired Conventional-minus-FE {row.estimate:.6g} (95% target-paired bootstrap {row.ci_low:.6g} to {row.ci_high:.6g}; {row.paired_targets}/9 valid pairs).")
    measured.extend([f"Nearby sampled pressure minima: {summary['nearby_pressure_minima']}/18; target-centered sampled minima: {summary['target_centered_pressure_minima']}/18.",
        "A nearby sampled pressure minimum and the target-centered sampled shell criterion are different properties; shell sampling does not certify an exact local minimum at the prescribed target. This paired objective comparison does not isolate the causal effect of a pressure minimum.",
        "When both command families have nearby pressure minima, these data cannot test whether pressure-minimum fields generally agree better than fields without such minima.",
        "Arrows encode direction only; exported force vectors and magnitude ratios provide the separate magnitude comparison. Directions are undefined and excluded if either total-force norm is <=1% of effective weight.",
        "Neighbor samples are dependent within commands. Bootstrap resamples nine paired target clusters, without treating 27 archived FE starts or individual force samples as independent replicates.",
        "All inputs remain historical FE/Conventional commands; current main GFE reference commands are unchanged, and the separate Appendix G formulations are not reused here."])
    (output/'measured_results_smoke.txt').write_text('\n'.join(measured)+'\n',encoding='utf-8')
    captions={
        'A1': "Paired force-model views for the central selected FE command and the selected FE command with largest model-root separation. Each row uses the identical pressure field and scale. White arrows show XZ-projected total-force direction at target y; their length does not encode magnitude. Both models include the same effective gravity. Root markers are projected 3D positions; missing roots would be omitted rather than identified as equilibria. Source phases and representative selection are fixed.",
        'A2': "Model agreement across nine same-target/same-seed FE and Conventional command pairs. Panel (a): median 3D angular discrepancy over 42 common samples at 0.5a, 1a and 2a; samples with either total-force norm <=1% of effective weight are excluded from direction statistics and retained in raw data. Panel (b): distance between independently resolved model roots in meters. Lines pair targets; displaced thick bars show median/IQR. Crosses mark nonrestoring resolved roots; unresolved counts remain in the denominators. Uncertainty resamples paired targets. Pressure-minimum diagnostics describe nearby and target-centered minima separately and do not imply a causal objective-independent effect."
    }
    _write_json(output/'A_figure_captions.json',captions)
    return dict(appendix='A',output_dir=output,figures=figures,table=frame,statistics=stats,summary=summary)


if __name__=='__main__':
    import sys
    result=run(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[4])
    print(json.dumps(result['summary'],indent=2))
