"""Appendices D and E: replay source studies and export current validation.

D retains its historical piston directivity and same-Rayleigh comparisons. New
elastic results never overwrite those records. E executes the deposited
notebook's field and prescription kernels in an isolated interpreter.
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
import time

import numpy as np
import pandas as pd

SCHEMA = 'appendix_de_compact_v3'
LEVELS = [4, 8, 10, 16, 32, 128, 2048]
D_SOURCE = 'appendix_sources/D_discretization/FE_discretization_methods_rapid_implementation'
E_SOURCE = 'appendix_sources/E_prescriptions/final'


def _de_io_path(path):
    path = Path(path)
    text = str(path)
    prefix = chr(92) * 2 + '?' + chr(92)
    if os.name == 'nt' and path.is_absolute() and not text.startswith(prefix):
        return Path(prefix + text)
    return path


def _ensure_d_current(root):
    """Regenerate the existing D design when its FE solver identity changes."""
    import importlib.util
    from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
    source = _de_io_path(Path(root)/D_SOURCE)
    receipt = source/'compact_generation.json'
    products = ('continuous_FE_phases.npz', 'continuous_FE_phase_metadata.json',
                'mechanics_offset_phase_indices.npz', 'mechanics_offset_results.csv')
    if receipt.exists():
        prior = json.loads(receipt.read_text(encoding='utf-8'))
        if prior.get('fe_solver_policy') == FE_SOLVER_REVISION and all(
                (source/name).exists() and prior.get('products', {}).get(name) == _content_id(source/name)
                for name in products):
            return
    code = source/'code'
    sys.path.insert(0, str(code))
    path = code/'prototypes/benchmark_mechanics_aware_offset.py'
    spec = importlib.util.spec_from_file_location('current_discretization_generation', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = json.loads((source/'continuous_FE_phase_metadata.json').read_text(encoding='utf-8'))
    selection = dict(original['payload'].get('source_selected_initializations', {}))
    # Preserve the declared initialization campaign, including an empty mapping
    # meaning all three established initializations, not old outcome selection.
    module.source_selected_initializations = lambda: dict(selection)
    module.run(source, timing_repeats=3, force_rebuild=False)
    _json(receipt, dict(fe_solver_policy=FE_SOLVER_REVISION, evaluator_backend='compact-piston-stencil-v1',
                        products={name: _content_id(source/name) for name in products}))


def _content_id(path):
    return content_identity(_de_io_path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12, 'axes.labelsize':13, 'axes.labelweight':'bold',
                         'axes.linewidth':1.6, 'xtick.labelsize':11, 'ytick.labelsize':11,
                         'xtick.major.width':1.5,'ytick.major.width':1.5,
                         'lines.linewidth':2.2,'lines.markersize':7,
                         'legend.fontsize':10.5,'legend.frameon':False,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':240,'svg.fonttype':'none','pdf.fonttype':42})
    return plt


def _label(ax, letter):
    ax.text(-.14,1.04,f'({letter})',transform=ax.transAxes,weight='bold',fontsize=14)
    for tick in ax.get_xticklabels()+ax.get_yticklabels():
        tick.set_fontweight('medium')


def _save(fig, directory, name):
    directory.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf','svg'):
        fig.savefig(directory/f'{name}.{extension}',bbox_inches='tight',facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(directory/f'{name}.png')


def _d_elastic(root, force=False, *, ctx=None, output=None):
    from dataclasses import asdict
    from .gorkov_core import REFERENCE_SOURCE_STRENGTH_PA_M_PEAK
    from .exact_validator import (ArbitraryArrayPressureField, PartialWaveForceEvaluator,
                                  PartialWaveNumerics, ElasticSphere, Medium,
                                  validate_static_trap, common_cartesian_root_starts)
    _ensure_d_current(root)
    source=_de_io_path(root/D_SOURCE)
    out=Path(output) if output is not None else root/'appendix_outputs/D'
    out.mkdir(parents=True,exist_ok=True)
    phases=np.load(source/'continuous_FE_phases.npz')
    indices=np.load(source/'mechanics_offset_phase_indices.npz')
    metadata=json.loads((source/'continuous_FE_phase_metadata.json').read_text(encoding="utf-8"))
    targets=dict(metadata['payload']['target_specs'])
    manifest=pd.read_csv(source/'mechanics_offset_command_manifest.csv')
    numerics=PartialWaveNumerics(lmax=3,fit_shell_wavelengths=(.10,.16,.22),fit_n_mu=6,
                               fit_n_phi=12,surface_n_mu=8,surface_n_phi=16)
    # Same seven starts and numerical root criterion as the current main SMOKE.
    starts=common_cartesian_root_starts((1.0,))
    root_budget=14
    root_tolerance=.0025
    numerical_scope='SMOKE'
    if ctx is not None:
        from . import pipeline as p
        if not ctx.config.full:
            raise ValueError('Explicit D2 context must carry the full validation protocol')
        protocol=p._finite_ka_protocol(ctx)
        numerics=p._partial_wave_numerics(ctx)
        starts=np.asarray(protocol['root_starts_a'])
        root_budget=int(protocol['max_nfev_per_start'])
        root_tolerance=float(protocol['numerical_root_tolerance'])
        numerical_scope='PRODUCTION'
    contract={'schema':SCHEMA,'physics':'current elastic-solid-sphere partial-wave reference',
              'field':'source-matched 5-mm piston directivity and channel order, front-only; current nominal source amplitude',
              'calibration':'New deployment evaluation scales historical unit piston to current nominal 120 dB SPL at 0.30m (peak pressure phasor); historical source had no measured absolute calibration.',
              'include_effective_gravity':True,'source_strength_pa_m_peak':REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
              'sphere':asdict(ElasticSphere()),'medium':asdict(Medium()),
              'numerics':asdict(numerics),'root_starts_a':starts.tolist(),
              'search_half_width_a':4.,'max_nfev_per_start':root_budget,'root_tolerance_scaled':root_tolerance,
              'jacobian_step_a':.1,
              'source_phases_content_id':_content_id(source/'continuous_FE_phases.npz'),
              'source_indices_content_id':_content_id(source/'mechanics_offset_phase_indices.npz'),
              'validator_content_id':_content_id(Path(__file__).with_name('exact_validator.py')),
              'empirical_scope':'15 current compact FE targets; 105 mechanics-selected commands. FE+g quantization not tested.',
              'numerical_scope':numerical_scope}
    identity=content_identity(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    path=out/'elastic_selected_commands.csv'
    data_path=out/'elastic_selected_commands.npz'
    cp=out/'elastic_contract.json'
    if not force and path.exists() and data_path.exists() and cp.exists() and json.loads(cp.read_text(encoding="utf-8")).get('fingerprint')==identity:
        shutil.rmtree(out/'elastic_checkpoints',ignore_errors=True)
        (out/'elastic_checkpoint.json').unlink(missing_ok=True)
        return pd.read_csv(path),contract
    rows=[];arrays={}
    work=[]
    for case,target in targets.items():
        work.append((case,0,'continuous_FE',np.asarray(phases[case]),np.asarray(target)))
        for record in manifest[manifest.case_id.eq(case)].itertuples():
            phase=2*np.pi*np.asarray(indices[record.command_key],dtype=float)/record.levels
            work.append((case,int(record.levels),record.command_key,phase,np.asarray(target)))
    resume=out/'elastic_checkpoint.json'
    if not force and resume.exists():
        previous=json.loads(resume.read_text(encoding="utf-8"))
        if previous.get('fingerprint')==identity:
            rows=previous['rows']
            for cached in (out/'elastic_checkpoints').glob('*.npz'):
                with np.load(cached) as saved:
                    arrays.update({key:saved[key] for key in saved.files})
    resumed_records=len(rows)
    completed={(row['case_id'],int(row['levels'])) for row in rows}
    (out/'elastic_checkpoints').mkdir(exist_ok=True)
    def evaluate_command(item):
        case,level,key,phase,target=item
        field=ArbitraryArrayPressureField(40000,ArbitraryArrayPressureField.rectangular_positions(),
               np.exp(1j*phase),source_strength_pa_m=REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,directivity='piston',piston_radius_m=.005,
               source_calibration_label=contract['calibration'])
        began=time.perf_counter()
        evaluator=PartialWaveForceEvaluator(40000,field,numerics=numerics)
        validated=validate_static_trap(evaluator,target,initial_offsets_a=starts,max_nfev=root_budget,
                   numerical_root_tolerance=root_tolerance,search_half_width_a=4.,jacobian_step_a=.1)
        eq=validated.equilibrium
        eigen=validated.symmetric_stiffness_eigenvalues_n_m
        row={'case_id':case,'levels':level,'command_key':key,'source_model':'compact corrected-Rayleigh FE',
             'validator_model':'current finite-ka elastic bead + effective gravity',
             'source_strength_pa_m_peak':REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,'validation_workers':2,'root_found':bool(eq.numerical_root_found),
             'locally_restoring':bool(eq.numerical_root_found and eigen[0]>0),
             'absolute_target_error_um':eq.displacement_norm_m*1e6,
             'root_residual_scaled':eq.residual_scaled_norm,
             'root_residual_n':float(np.linalg.norm(eq.residual_force_n)),
             'on_search_boundary':bool(eq.on_search_boundary),
             'stiffness_min_n_m':float(eigen[0]),'validation_wall_s':time.perf_counter()-began}
        for axis,name in enumerate('xyz'):
            row[f'target_{name}_m']=float(target[axis]);row[f'root_{name}_m']=float(eq.equilibrium_m[axis])
            row[f'displacement_{name}_um']=float(eq.displacement_m[axis]*1e6)
            row[f'stiffness_eigenvalue_{axis+1}_n_m']=float(eigen[axis])
        prefix=f'{case}__Q{level:04d}'
        partial={prefix+'__phase_rad':phase,prefix+'__root_m':eq.equilibrium_m,
                 prefix+'__target_m':target,prefix+'__jacobian_n_m':validated.force_jacobian_n_m,
                 prefix+'__stiffness_n_m':validated.symmetric_stiffness_n_m,
                 prefix+'__force_residual_n':eq.residual_force_n,
                 prefix+'__all_root_candidates_m':np.asarray([v.equilibrium_m for v in eq.candidates]),
                 prefix+'__all_root_candidate_residuals':np.asarray([v.residual_scaled_norm for v in eq.candidates])}
        return row,partial,prefix

    from concurrent.futures import ThreadPoolExecutor
    pending=[item for item in work if (item[0],item[1]) not in completed]
    # Independent regenerated compact FE commands use separate evaluator instances.
    with ThreadPoolExecutor(max_workers=2) as workers:
        for row,partial,prefix in workers.map(evaluate_command,pending):
            arrays.update(partial); rows.append(row)
            np.savez_compressed(out/'elastic_checkpoints'/f'{prefix}.npz',**partial)
            _json(resume,{'fingerprint':identity,'rows':rows})
            print(f"D elastic {len(rows)}/{len(work)} {row['case_id']} Q={row['levels']} root={row['root_found']}",flush=True)
    frame=pd.DataFrame(rows)
    frame['additional_equilibrium_shift_um']=np.nan
    frame['stiffness_retention_pct']=np.nan
    for case in targets:
        continuous=frame[frame.case_id.eq(case)&frame.levels.eq(0)].iloc[0]
        base=continuous[[f'root_{a}_m' for a in 'xyz']].to_numpy(dtype=float)
        mask=frame.case_id.eq(case)&frame.root_found&bool(continuous.root_found)
        frame.loc[mask,'additional_equilibrium_shift_um']=np.linalg.norm(
            frame.loc[mask,[f'root_{a}_m' for a in 'xyz']].to_numpy()-base,axis=1)*1e6
        if continuous.locally_restoring:
            frame.loc[mask,'stiffness_retention_pct']=100*frame.loc[mask,'stiffness_min_n_m']/continuous.stiffness_min_n_m
    frame.to_csv(path,index=False)
    np.savez_compressed(data_path,**arrays)
    _json(cp,{'fingerprint':identity,**contract})
    _json(out/'validation_execution.json',{'records_reused_before_execution':resumed_records,
          'records_computed':len(work)-resumed_records,'concurrent_workers_for_new_records':2,
          'scope':f'Evaluator setup, {len(starts)} root starts and force Jacobian; wall times are observational under worker contention, not synthesis or controlled comparison timings.'})
    shutil.rmtree(out/'elastic_checkpoints',ignore_errors=True)
    resume.unlink(missing_ok=True)
    return frame,contract


def _e_worker(root):
    """Run unchanged numerical source cells, adding only capture/export hooks."""
    source=_de_io_path(root/E_SOURCE)
    out=root/'appendix_outputs/E'
    out.mkdir(parents=True,exist_ok=True)
    notebook=json.loads((source/'rule_based_projection_prescription_engineering.ipynb').read_text(encoding="utf-8"))
    ns={'__name__':'appendix_e_source_execution'}
    os.environ['PRESCRIPTION_OUTPUT_ROOT']=str(out)
    began=time.perf_counter()
    reusable_archive = out/'all_candidate_commands_profiles.npz'
    reusable_manifest = out/'candidate_command_manifest.csv'
    previous_phases, previous_rows = {}, {}
    if reusable_archive.exists() and reusable_manifest.exists():
        with np.load(reusable_archive, allow_pickle=False) as saved:
            previous_phases = {key: saved[key].copy() for key in saved.files if key.endswith('__phase_rad') and 'FE_reference' not in key}
        previous_rows = {str(row.archive_key): row._asdict() for row in pd.read_csv(reusable_manifest).itertuples(index=False)}
    for index in (3,5,6,7,9,11):
        code=''.join(notebook['cells'][index]['source'])
        if index==11:
            # Preserve the calculation and retain profiles formerly lost after each loop.
            code=code.replace('SINGLE_COMMANDS = {}','SINGLE_COMMANDS = {}\nSINGLE_PROFILES = {}')
            code=code.replace('        single_rows.append({','        SINGLE_PROFILES[(method, float(radius_lambda))] = profile.copy()\n        single_rows.append({')
        print(f'E executing source cell {index}',flush=True)
        exec(compile(code,f'prescription_source_cell_{index}','exec'),ns)
        if index == 6 and previous_phases:
            original_single = ns['single_command']
            original_paired = ns['paired_command']
            def previous_command(key):
                row = previous_rows.get(key)
                phase = previous_phases.get(key+'__phase_rad')
                if row is None or phase is None:
                    return None
                return dict(phase_rad=phase.copy(), iterations=int(row['iterations']),
                            synthesis_wall_s=float(row['synthesis_wall_s']), cache_reused=True)
            def reused_single(method, radius):
                key = f'single_{method}_R{radius:.2f}'.replace('.','p')
                old = previous_command(key)
                return old if old is not None else original_single(method, radius)
            def reused_paired(method, gap, radius):
                key = f'opposed_D{gap:.2f}_{method}_R{radius:.2f}'.replace('.','p')
                old = previous_command(key)
                if old is not None:
                    old.update(arrays=ns['paired_arrays'](gap), separation_lambda=gap, radius_lambda=radius)
                    return old
                return original_paired(method, gap, radius)
            ns['single_command'] = reused_single
            ns['paired_command'] = reused_paired
    elapsed=time.perf_counter()-began
    arrays={};rows=[]
    for (gap,method,radius),command in ns['OPPOSED_COMMANDS'].items():
        key=f'opposed_D{gap:.2f}_{method}_R{radius:.2f}'.replace('.','p')
        rindex=int(np.where(ns['OPPOSED_RADIUS_CANDIDATES_LAMBDA']==radius)[0][0])
        arrays[key+'__phase_rad']=command['phase_rad']
        arrays[key+'__radial_profile']=ns['OPPOSED_PROFILES'][(gap,method)][:,rindex]
        rows.append({'archive_key':key,'geometry':'opposed','gap_lambda':gap,'method':method,
                     'radius_lambda':radius,'iterations':command['iterations'],
                     'synthesis_wall_s':command['synthesis_wall_s']})
    for (method,radius),command in ns['SINGLE_COMMANDS'].items():
        key=f'single_{method}_R{radius:.2f}'.replace('.','p')
        arrays[key+'__phase_rad']=command['phase_rad']
        arrays[key+'__radial_profile']=ns['SINGLE_PROFILES'][(method,radius)]
        rows.append({'archive_key':key,'geometry':'single','gap_lambda':None,'method':method,
                     'radius_lambda':radius,'iterations':command['iterations'],
                     'synthesis_wall_s':command['synthesis_wall_s']})
    arrays['single_FE_reference__phase_rad']=ns['single_fe_phase']
    arrays['single_FE_reference__radial_profile']=ns['single_fe_profile']
    arrays['single_radial_axis_m']=ns['single_radial_axis_m']
    arrays['opposed_radial_axis_m']=ns['opposed_radial_axis_m']
    for name,phase in [('FE',ns['single_fe_phase']),('fixed_GS',ns['SINGLE_COMMANDS'][('GS',1.4)]['phase_rad']),
                       ('refined_GS',ns['SINGLE_COMMANDS'][('GS',.45)]['phase_rad'])]:
        axis,amplitude=ns['single_map'](phase,samples=151)
        arrays['map_'+name]=amplitude
        arrays['map_axis_m']=axis
    np.savez_compressed(out/'all_candidate_commands_profiles.npz',**arrays)
    pd.DataFrame(rows).to_csv(out/'candidate_command_manifest.csv',index=False)
    from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
    _json(out/'execution_provenance.json',{'schema':SCHEMA,'fe_solver_policy':FE_SOLVER_REVISION,'status':'SMOKE source-study calculation',
          'source_runtime_notebook':f'{E_SOURCE}/rule_based_projection_prescription_engineering.ipynb',
          'executed_source_cell_indices':[3,5,6,7,9,11],
          'source_edits':'Current compact FE reference; unchanged prescription candidates and morphology metrics.',
          'candidate_count':len(rows),'numerical_execution_wall_s':elapsed,
          'timing_scope':'synthesis_wall_s times projection recurrence only after control transfer setup; total numerical execution includes source-kernel import, transfer construction, candidate projection and profile evaluation, and table export. It excludes final maps/rendering. Human hours unmeasured.',
          'single_reference':'Compact L-BFGS-B FE, seed 172817509.',
          'scope':'Pressure morphology transfer/refinement; no new elastic mechanical or levitation claim.',
          'opposed_reference':'same-method D=26.7 lambda, R=1.4 lambda; each profile normalized by global maximum in 0--18 mm',
          'single_reference_normalization':'each radial profile divided by its first annular maximum',
          'field_scope':'Exact historical source runtime with calibrated Murata transfer; isolated from active main runtime.'})
    print(f'E completed: {len(rows)} candidates, {elapsed:.2f} s numerical calculation',flush=True)


def _e_compute(root,force=False):
    out=root/'appendix_outputs/E'
    source=_de_io_path(root/E_SOURCE)
    provenance=out/'execution_provenance.json'
    from hat_revision_pipeline.fe_solver_policy import FE_SOLVER_REVISION
    current = json.loads(provenance.read_text(encoding='utf-8')) if provenance.exists() else {}
    if force or current.get('fe_solver_policy') != FE_SOLVER_REVISION:
        env=os.environ.copy();env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        env['PYTHONPATH']=os.pathsep.join(filter(None,(str(root),str(root/'runtime/src'),env.get('PYTHONPATH'))))
        out.mkdir(parents=True,exist_ok=True)
        with (out/'source_execution.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-e',str(root)],
                           check=True,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=root)
    return json.loads(provenance.read_text(encoding="utf-8"))


def _render_d(root,elastic,contract,*,output=None,timing_results=None,evidence_status='SMOKE'):
    plt=_style();source=_de_io_path(root/D_SOURCE);out=Path(output) if output is not None else root/'appendix_outputs/D'
    old=pd.read_csv(source/'mechanics_offset_results.csv')
    old.to_csv(out/'historical_Rayleigh_comparisons.csv',index=False)
    if timing_results is not None:
        old=old.rename(columns={'mechanics_setup_ms':'historical_mechanics_setup_ms'})
        timing_columns=['transition_time_median_ms','transition_time_q25_ms','transition_time_q75_ms','timing_repeats']
        old=old.drop(columns=[name for name in timing_columns if name in old]).merge(
            timing_results[['case_id','levels','method',*timing_columns]],
            on=['case_id','levels','method'],validate='one_to_one')
        old.to_csv(out/'production_Rayleigh_comparisons.csv',index=False)
    for name in ('mechanics_offset_summary.csv','mechanics_offset_manifest.json','selection_record.json',
                 'continuous_FE_endpoints.csv','mechanics_offset_command_manifest.csv',
                 'continuous_FE_phase_metadata.json'):
        shutil.copy2(source/name,out/f'historical_{name}')
    shutil.copy2(source/'mechanics_offset_phase_indices.npz',out/'hardware_phase_indices.npz')
    plot_directory = out / 'plot_data'
    plot_directory.mkdir(exist_ok=True)
    elastic = elastic.copy()
    for column in ('additional_equilibrium_shift_um', 'absolute_target_error_um'):
        elastic[column[:-3] + '_m'] = elastic[column] * 1e-6
    elastic.to_csv(out/'elastic_selected_commands.csv', index=False)
    elastic.to_csv(plot_directory/'D2_elastic_points.csv', index=False)
    coordinates = old.groupby(['method','levels']).agg(
        added_shift_p90_m=('actual_equilibrium_shift_mm', lambda values: values.quantile(.9)*1e-3),
        transition_time_median_ms=('transition_time_median_ms','median')).reset_index()
    coordinates.to_csv(plot_directory/'D1_rounding_coordinates.csv', index=False)
    labels={'fixed_nearest':'Nearest','best_offset':'Objective offset','mechanics_offset':'Mechanics offset'}
    colors={'fixed_nearest':'#737373','best_offset':'#da8c20','mechanics_offset':'#1665ba'}
    x=np.arange(len(LEVELS));fig,axes=plt.subplots(1,2,figsize=(10.3,4.5),layout='constrained')
    for method in labels:
        part=old[old.method.eq(method)]
        q=part.groupby('levels')['actual_equilibrium_shift_mm'].quantile(.9).reindex(LEVELS)*1e-3
        timing=part.groupby('levels')['transition_time_median_ms'].median().reindex(LEVELS)
        for ax,y in zip(axes,(q,timing)):
            ax.plot(x,y,'o-',color=colors[method],markeredgecolor='white',markeredgewidth=1.2,label=labels[method])
    for i,ax in enumerate(axes):
        ax.set_xticks(x,LEVELS);ax.set_xlabel('Phase levels, Q');ax.set_yscale('log');_label(ax,'ab'[i])
    axes[0].set_ylabel('Added shift, 90th percentile (m)');axes[1].set_ylabel('Median transition time (ms)')
    axes[0].legend(loc='best')
    figures=[_save(fig,out/'figures','Figure_D1_discretization_comparison')]
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.5),layout='constrained')
    selection=elastic[elastic.levels.gt(0)]
    for index,(ax,column,ylabel) in enumerate(zip(axes,('additional_equilibrium_shift_um','absolute_target_error_um'),
                ('Added shift (m)','Target error (m)'))):
        for j,level in enumerate(LEVELS):
            values=selection[selection.levels.eq(level)&selection.root_found][column].dropna().to_numpy()*1e-6
            if not len(values):continue
            jitter=np.linspace(-.18,.18,len(values))
            ax.scatter(j+jitter,values,s=37,color='#58a0cf',alpha=.65,edgecolor='#16456d',linewidth=.8)
            q25,median,q75=np.quantile(values,[.25,.5,.75])
            ax.errorbar(j,median,yerr=[[median-q25],[q75-median]],fmt='o',ms=8,
                        color='#153e69',ecolor='#153e69',elinewidth=2.4,capsize=4,capthick=1.8)
        ax.set_xticks(x,LEVELS);ax.set_xlabel('Phase levels, Q');ax.set_ylabel(ylabel)
        if column=='additional_equilibrium_shift_um' and selection[column].dropna().gt(0).all():ax.set_yscale('log')
        else: ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
        _label(ax,'ab'[index])
    figures.append(_save(fig,out/'figures','Figure_D2_elastic_selected_commands'))
    summaries=[]
    for level,group in selection.groupby('levels',sort=True):
        valid=group[group.root_found]
        row={'levels':int(level),'commands':len(group),'roots_found':int(group.root_found.sum()),
             'locally_restoring_roots':int(group.locally_restoring.sum())}
        for col in ('additional_equilibrium_shift_um','absolute_target_error_um','stiffness_retention_pct'):
            values=valid[col].dropna()
            row[col+'_median']=float(values.median()) if len(values) else None
            row[col+'_p90']=float(values.quantile(.9)) if len(values) else None
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    for column in list(summary.columns):
        if '_um_' in column:
            summary[column.replace('_um_', '_m_')] = summary[column]*1e-6
    summary.to_csv(out/'elastic_resolution_summary.csv',index=False)
    measured=[f'{evidence_status} measured results — Appendix D',
      'Historical source: 15 FE targets; Q=4,8,10,16,32,128,2048; 315 same-Rayleigh command validations.',
      'Mechanics-offset rounding comparisons are recomputed from the current compact FE endpoints.',
      f'Current elastic-bead reference: {int(elastic.root_found.sum())}/{len(elastic)} force roots resolved; '+
      f'{int(elastic.locally_restoring.sum())}/{len(elastic)} locally restoring roots, including continuous commands.',
      'Current validation preserves the source 5 mm piston directivity and regenerated compact FE commands, maps source amplitude to the current nominal 120 dB at 0.30 m, and applies effective gravity. This deployment evaluation is distinct from the source unit-strength Rayleigh comparison.',
      'Added shift is measured from each corresponding continuous elastic equilibrium; absolute target error is reported separately.',
      'FE+g quantization was not tested. Source transition timings are not a new cross-method common-platform timing study.',
      summary.to_string(index=False)]
    (out/'measured_results_smoke.txt').write_text('\n\n'.join(measured)+'\n', encoding="utf-8")
    (out/'figure_captions.txt').write_text(
        'Figure D1. Same-Rayleigh comparison using compact FE commands for 15 targets and seven phase resolutions. (a) 90th percentile of equilibrium shift relative to each continuous equilibrium, in meters. (b) Across-target median transition time, using each source target/method three-repeat median; source setup and root validation are outside this transition timing.\n\n'
        'Figure D2. New elastic-bead deployment evaluation of the 105 source-selected mechanics-offset commands. The source piston directivity and channel order are retained; source amplitude is explicitly mapped from 1 to 8.485 Pa m peak and effective gravity is included. Points denote target cases; symbols and whiskers denote median and interquartile range. (a) Shift relative to each continuous elastic equilibrium, in meters. (b) Absolute distance from requested target. Only resolved roots are included; complete root outcomes are exported.\n', encoding="utf-8")
    if timing_results is not None:
        caption=(out/'figure_captions.txt').read_text(encoding="utf-8").replace(
            'using each source target/method three-repeat median; source setup and root validation are outside this transition timing.',
            'using each target/method 30-repeat median from fresh serial execution on the recorded production environment. One unmeasured warm-up precedes each block; model/reference setup and root validation are outside the transition timer and are recorded separately.')
        (out/'figure_captions.txt').write_text(caption, encoding="utf-8")
    (out/'mock_results_sentences.txt').write_text('TEMPLATE ONLY — do not quote as measured\nFor Q=[Q], mechanics-offset rounding retained [n/N] locally restoring elastic-bead equilibria, with median additional equilibrium shift [x] m and median absolute target error [y] m.\n', encoding="utf-8")
    source_records=[{'path':str(p.relative_to(_de_io_path(root))),'content_id':_content_id(p)} for p in sorted(source.glob('*.csv'))]
    _json(out/'provenance.json',{'status':evidence_status,'schema':SCHEMA,'elastic_contract':contract,
                               'historical_records':source_records,'figures':figures,
                               'original_source_strength_pa_m_peak':1.0,
                               'deployment_source_strength_pa_m_peak':contract['source_strength_pa_m_peak'],
                               'deployment_pressure_scale_factor':contract['source_strength_pa_m_peak']/1.0,
                               'amplitude_probe_table':'source_unit_amplitude_probe.csv',
                               'validation_timing_scope':f"validation_wall_s includes evaluator setup, {len(contract['root_starts_a'])} prescribed root starts and the force Jacobian. See validation_execution.json for worker/resume scope; elapsed values are observational and are not command-synthesis timings or a controlled cross-method cost comparison.",
                               'figure_D1_scope':'current corrected-Rayleigh source mechanics; fresh recorded-machine transition timings' if timing_results is not None else 'current corrected-Rayleigh source comparisons',
                               'figure_D2_scope':'new current elastic validation of source-selected commands; no reselection'})
    return {'figures':figures,'tables':[str(out/'elastic_resolution_summary.csv'),str(out/'historical_Rayleigh_comparisons.csv')],
            'results':str(out/'measured_results_smoke.txt')}


def _render_e(root,provenance,*,output=None):
    plt=_style();out=Path(output) if output is not None else root/'appendix_outputs/E'
    # The standalone payload contains the deposited executable source
    # notebook, not redundant copies of tables that notebook produces.  Audit
    # the reexecuted tables against the source study's declared populations.
    expected_rows={
        'opposed_array_radius_selection_summary.csv':10,
        'opposed_array_radius_transfer_audit.csv':290,
        'single_sided_radius_reference_rmse.csv':20,
    }
    replayed_tables={}
    audit=[]
    for name,expected in expected_rows.items():
        replayed=pd.read_csv(out/'tables'/name)
        numeric=replayed.select_dtypes('number').drop(
            columns=[column for column in replayed.columns if 'wall' in column],
            errors='ignore')
        finite=bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
        audit.append({
            'table':name,'source_runtime_notebook':
            f'{E_SOURCE}/rule_based_projection_prescription_engineering.ipynb',
            'static_source_table_bundled':False,'reexecuted_rows':len(replayed),
            'expected_rows':expected,'row_count_matches':len(replayed)==expected,
            'non_timing_numeric_values_finite':finite,
        })
        replayed_tables[name]=replayed
    selection=replayed_tables['opposed_array_radius_selection_summary.csv']
    transfer=replayed_tables['opposed_array_radius_transfer_audit.csv']
    single=replayed_tables['single_sided_radius_reference_rmse.csv']
    population_ok=(
        set(selection.method)=={'IB','GS'}
        and set(selection.array_separation_lambda)=={26.7,30.,35.,40.,45.}
        and bool(selection.candidate_count.eq(29).all())
        and bool(transfer.groupby(['method','array_separation_lambda']).size().eq(29).all())
        and bool(single.groupby('method').size().eq(10).all())
    )
    pd.DataFrame(audit).to_csv(out/'source_replay_audit.csv',index=False)
    if not population_ok or not all(
            item['row_count_matches'] and item['non_timing_numeric_values_finite']
            for item in audit):
        raise RuntimeError('E source-kernel replay failed its declared-population audit')
    selection=pd.read_csv(out/'tables/opposed_array_radius_selection_summary.csv')
    single=pd.read_csv(out/'tables/single_sided_radius_reference_rmse.csv')
    saved=np.load(out/'all_candidate_commands_profiles.npz')
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.5),layout='constrained')
    for method,color,marker in [('IB','#4C7C2F','o'),('GS','#80568D','s')]:
        part=selection[selection.method.eq(method)]
        axes[0].plot(part.array_separation_lambda,part.selected_radius_lambda,marker+'-',
                     color=color,markersize=9 if method=='IB' else 6.5,
                     markerfacecolor='white' if method=='IB' else color,markeredgecolor=color,markeredgewidth=1.5,label=method)
        axes[1].plot(part.array_separation_lambda,part.fixed_1p4_profile_rmse,marker+'--',color=color,alpha=.9,markersize=9 if method=='IB' else 6.5,markerfacecolor='white' if method=='IB' else color,markeredgewidth=1.5,label=method+' fixed')
        axes[1].plot(part.array_separation_lambda,part.selected_profile_rmse,marker+'-',color=color,markersize=9 if method=='IB' else 6.5,markerfacecolor='white' if method=='IB' else color,markeredgewidth=1.5,label=method+' refined')
    axes[0].axhline(1.4,color='.35',ls=':',lw=1.8)
    for i,ax in enumerate(axes):
        ax.set_xlabel('Array separation (λ)');ax.set_xticks([26.7,30,35,40,45]);_label(ax,'ab'[i])
    axes[0].set_ylabel('Selected prescription radius (λ)');axes[1].set_ylabel('Radial-profile RMSE')
    axes[0].legend();axes[1].legend(fontsize=9.5)
    figures=[_save(fig,out/'figures','Figure_E1_radius_transfer')]
    fig,axes=plt.subplots(2,2,figsize=(9.4,8.6),layout='constrained')
    axis=saved['map_axis_m']*1000
    for ax,name,letter,text in zip(axes.flat[:3],('FE','fixed_GS','refined_GS'),'abc',('FE reference','GS: R = 1.40λ','GS: R = 0.45λ')):
        im=ax.imshow(saved['map_'+name],origin='lower',extent=[axis[0],axis[-1],axis[0],axis[-1]],
                     cmap='magma',vmin=0,vmax=1,interpolation='none')
        ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)');ax.set_box_aspect(1)
        ax.text(.04,.95,text,transform=ax.transAxes,va='top',weight='bold',color='white',fontsize=11)
        _label(ax,letter)
    ax=axes[1,1]
    ax.plot(saved['single_radial_axis_m']*1000,saved['single_FE_reference__radial_profile'],color='#C76A12',label='FE reference')
    for radius,color in [(1.4,'#AA8DB3'),(.45,'#80568D')]:
        key=f'single_GS_R{radius:.2f}'.replace('.','p')
        ax.plot(saved['single_radial_axis_m']*1000,saved[key+'__radial_profile'],color=color,linestyle='--' if radius==1.4 else '-',label=f'GS: R = {radius:.2f}λ')
    ax.set_xlabel('Radial distance (mm)');ax.set_ylabel('Normalized pressure');ax.legend(fontsize=10);ax.set_box_aspect(1);_label(ax,'d')
    colorbar=fig.colorbar(im,ax=axes[0,:],shrink=.85,pad=.025);colorbar.set_label('|p| / max |p|',weight='bold')
    figures.append(_save(fig,out/'figures','Figure_E2_single_reference_refinement'))
    far=selection[np.isclose(selection.array_separation_lambda,45)]
    gs=single[single.method.eq('GS')]
    retained=gs[np.isclose(gs.radius_lambda,.45)].iloc[0]
    fixed=gs[np.isclose(gs.radius_lambda,1.4)].iloc[0]
    best=gs.loc[gs.radial_profile_rmse_to_deposited_fe.idxmin()]
    manifest=pd.read_csv(out/'candidate_command_manifest.csv')
    manifest['independent_phase_coordinates']=256
    manifest['physical_sources']=np.where(manifest.geometry.eq('opposed'),512,256)
    manifest['paired_panels_share_phase']=manifest.geometry.eq('opposed')
    manifest.to_csv(out/'candidate_command_manifest.csv',index=False)
    timing=manifest.groupby('geometry').agg(candidates=('archive_key','count'),
           projection_recurrence_sum_s=('synthesis_wall_s','sum'),projection_recurrence_median_s=('synthesis_wall_s','median')).reset_index()
    timing['scope']='Projection recurrence after control transfer construction; human preparation time unmeasured'
    timing.to_csv(out/'timing_scope_summary.csv',index=False)
    text=['SMOKE measured results — Appendix E',
          'The exact source numerical kernels were reexecuted with unchanged morphology definitions, radius candidates and current compact L-BFGS-B FE reference.',
          'Two facing 16×16 arrays retain the target at their midpoint; array separation spans 26.7–45λ. Each method has 29 candidate radii at each separation.',
          '\n'.join(f'At D=45λ, {r.method}: selected R={r.selected_radius_lambda:.2f}λ; radial-profile RMSE {r.fixed_1p4_profile_rmse:.5f} fixed → {r.selected_profile_rmse:.5f} refined.' for r in far.itertuples()),
          f'One-sided GS: retained R=0.45λ gives FE-reference radial-profile RMSE {retained.radial_profile_rmse_to_deposited_fe:.5f}, versus {fixed.radial_profile_rmse_to_deposited_fe:.5f} at R=1.40λ. Sampled minimum R={best.radius_lambda:.2f}λ gives {best.radial_profile_rmse_to_deposited_fe:.5f}.',
          f'All {len(manifest)} candidate phases and radial profiles plus the deposited FE reference are exported.',
          'The FE reference uses compact L-BFGS-B at seed 172817509. These are morphology comparisons, not independent elastic force benchmarks.',
          provenance['timing_scope'],timing.to_string(index=False)]
    (out/'measured_results_smoke.txt').write_text('\n\n'.join(text)+'\n', encoding="utf-8")
    (out/'figure_captions.txt').write_text(
        'Figure E1. Prescription transfer between opposed-array separations while the trap remains at the midpoint. (a) Radius selected over 29 candidates by same-method radial-profile RMSE against the D=26.7λ, R=1.4λ reference. (b) Fixed-radius versus refined-radius error. These are pressure-morphology comparisons with 256 paired phase commands shared by two panels.\n\n'
        'Figure E2. Single-sided pressure morphology relative to the deposited current compact L-BFGS-B FE reference (seed 172817509, state_index 484). (a) FE reference. (b) GS at R=1.4λ. (c) GS at the retained R=0.45λ. Each pressure map is normalized by its own peak. (d) Azimuthally averaged radial profiles, each normalized by its first annular maximum. The retained radius follows the established prescription; the sampled RMSE minimum is reported in the accompanying table.\n', encoding="utf-8")
    (out/'mock_results_sentences.txt').write_text('TEMPLATE ONLY — do not quote as measured\nAt separation [D]λ, refining the [method] radius from [R0]λ to [R1]λ changed profile RMSE from [e0] to [e1]. Candidate preparation and projection execution are reported under their separately declared timing scopes.\n', encoding="utf-8")
    return {'figures':figures,'tables':[str(out/'tables/opposed_array_radius_selection_summary.csv'),str(out/'tables/single_sided_radius_reference_rmse.csv'),str(out/'timing_scope_summary.csv')],
            'results':str(out/'measured_results_smoke.txt')}



def _d_source_amplitude_probe(root):
    """Retain the unit-source gravity probe that motivated explicit rescaling."""
    from .exact_validator import ArbitraryArrayPressureField, PartialWaveForceEvaluator, PartialWaveNumerics, validate_static_trap
    output=root/'appendix_outputs/D/source_unit_amplitude_probe.csv'
    if output.exists():return
    phase=np.load(_de_io_path(root/D_SOURCE)/'continuous_FE_phases.npz')['X+0_Y+0_Z50']
    field=ArbitraryArrayPressureField(40000,ArbitraryArrayPressureField.rectangular_positions(),
          np.exp(1j*phase),source_strength_pa_m=1.,directivity='piston',piston_radius_m=.005,
          source_calibration_label='Historical unit-source piston; no measured calibration')
    numerics=PartialWaveNumerics(lmax=3,fit_shell_wavelengths=(.10,.16,.22),fit_n_mu=6,fit_n_phi=12,
                               surface_n_mu=8,surface_n_phi=16)
    evaluator=PartialWaveForceEvaluator(40000,field,numerics=numerics)
    result=validate_static_trap(evaluator,[0,0,.05],initial_offsets_a=np.zeros((1,3)),max_nfev=35,
                               numerical_root_tolerance=.0025)
    eq=result.equilibrium
    pd.DataFrame([{'case_id':'X+0_Y+0_Z50','source_strength_pa_m_peak':1.,
       'include_effective_gravity':True,'root_found':eq.numerical_root_found,
       'on_search_boundary':eq.on_search_boundary,'search_half_width_a':4.,
       'starts':1,'max_nfev':35,'root_residual_scaled':eq.residual_scaled_norm,
       'reported_search_candidate_distance_um':eq.displacement_norm_m*1e6,
       'interpretation':'One declared unit-amplitude pilot reached the search boundary without a force root; not an equilibrium displacement measurement.'}]).to_csv(output,index=False)



def _d_field_mapping_audit(root):
    """Compare the source transfer function to the common evaluator field."""
    import importlib.util
    from .exact_validator import ArbitraryArrayPressureField
    from .gorkov_core import REFERENCE_SOURCE_STRENGTH_PA_M_PEAK
    source=_de_io_path(root/D_SOURCE)
    source_file=source/'code/inputs/hat_geometry_branch_benchmark.py'
    spec=importlib.util.spec_from_file_location('_appendix_d_source_backend',source_file)
    backend=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=backend;spec.loader.exec_module(backend)
    points=np.asarray(json.loads((source/'continuous_FE_phase_metadata.json').read_text(encoding="utf-8"))['payload']['target_specs'][0][1])[None,:]
    offsets=np.vstack([np.zeros(3),.001*np.eye(3),-.001*np.eye(3)])
    points=points+offsets
    phase=np.load(source/'continuous_FE_phases.npz')['X+0_Y+0_Z45']
    positions=backend.square_points(16,.01)
    normals=np.tile([0.,0.,1.],(256,1))
    # The transfer operator only uses these geometry attributes.
    from types import SimpleNamespace
    geometry=SimpleNamespace(positions=positions,normals=normals)
    historical=backend.transfer_matrix(points,geometry)@np.exp(1j*phase)
    current=ArbitraryArrayPressureField(40000,positions,np.exp(1j*phase),normals=normals,
                  source_strength_pa_m=REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
                  directivity='piston',piston_radius_m=.005).pressure(points)
    residual=current-REFERENCE_SOURCE_STRENGTH_PA_M_PEAK*historical
    relative=float(np.linalg.norm(residual)/np.linalg.norm(current))
    if relative>1e-12:raise RuntimeError(f'D source/common pressure mapping changed: {relative}')
    pd.DataFrame([{'points':len(points),'phase_case':'X+0_Y+0_Z45',
       'original_source_strength_pa_m_peak':1.,
       'deployment_source_strength_pa_m_peak':REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
       'pressure_scale_factor':REFERENCE_SOURCE_STRENGTH_PA_M_PEAK,
       'relative_complex_pressure_difference':relative,
       'historical_backend_content_id':_content_id(source_file),
       'scope':'same piston directivity, front-only source and channel order; declared uniform amplitude scaling'}]).to_csv(root/'appendix_outputs/D/field_mapping_audit.csv',index=False)


def run(root, *, force=False):
    """Generate/replay D and E independently; return figures, tables, results."""
    root=Path(root).resolve()
    (root/'appendix_outputs/D').mkdir(parents=True,exist_ok=True)
    _ensure_d_current(root)
    _d_source_amplitude_probe(root)
    _d_field_mapping_audit(root)
    elastic,contract=_d_elastic(root,force=force)
    d=_render_d(root,elastic,contract)
    provenance=_e_compute(root,force=force)
    e=_render_e(root,provenance)
    result={'D':d,'E':e,'figures':d['figures']+e['figures'],
            'tables':d['tables']+e['tables'],'results':[d['results'],e['results']]}
    _json(root/'appendix_outputs/de_manifest.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker-e',type=Path)
    args=parser.parse_args()
    if args.worker_e:_e_worker(args.worker_e.resolve())
