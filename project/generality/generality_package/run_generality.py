"""Recompute the deposited eight main figures on each explicit setting."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(MPLBACKEND='Agg',JAX_ENABLE_X64='true',JAX_PLATFORMS='cpu',HAT_HIDE_MODE_NOTE='1',HAT_PROTOTYPE_LABEL='0',HAT_EXACT_WORKERS='1',HAT_BRANCH_ALPHA1000_WORKERS='1',HAT_FIG1_DOMAIN_WORKERS='1')
import argparse, json, sys, time
from rineng_content_id import content_identity
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime/src'));sys.path.insert(0,str(ROOT))
from generality_settings import SETTINGS,geometry,export
from hat_revision_pipeline import pipeline as p,final_figures as ff,main_feg as mf
from hat_revision_pipeline.config import RunConfig
from hat_revision_pipeline.cache import CacheStore,digest_payload,digest_array,dependency_snapshot
from hat_revision_pipeline.gorkov_core import ArrayGeometry,transfer_matrix,SOURCE_SCALE_PA_M_PER_MURATA_UNIT
from hat_revision_pipeline.exact_validator import PartialWaveNumerics

class GeneralityContext(p.PipelineContext):
    @property
    def positions_m(self):return self.memo['setting_positions_m']
    @property
    def normals(self):return self.memo['setting_normals']

def install_physics(ctx):
    """Carry the same source positions, normals and frequency into every path."""
    from generality_mechanics_speed import install
    install()
    for name in ('_array_field','_finite_ka_validation','_viscous_elastic_validation','_exact_comparison'):
        original=getattr(p,name)
        def wrapper(context,*args,_original=original,**kwargs):
            if kwargs.get('normals') is None:
                custom=kwargs.get('positions_m')
                if custom is not None and not np.array_equal(custom,context.positions_m):
                    raise ValueError('Custom source positions require explicit normals')
                kwargs['normals']=context.normals
            return _original(context,*args,**kwargs)
        setattr(p,name,wrapper)
    def stencil(positions_m,target_m,frequency_hz,*,spacing_m=None,normals=None):
        spacing=ctx.config.stencil_m if spacing_m is None else spacing_m
        normal_values=ctx.normals if normals is None else normals
        if not np.array_equal(positions_m,ctx.positions_m) and normals is None:
            raise ValueError('Custom stencil requires explicit source normals')
        axis=np.arange(-2,3)*spacing
        points=np.asarray(target_m)+np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        return transfer_matrix(points,ArrayGeometry(positions_m,normal_values),frequency_hz,source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT)
    p._stencil_transfer=stencil
    def numerics(context,lmax=None):
        return PartialWaveNumerics(lmax=int(lmax or (8 if context.config.full else 6)),fit_shell_wavelengths=(.10,.15,.20),fit_n_mu=12,fit_n_phi=24,surface_n_mu=20,surface_n_phi=40)
    p._partial_wave_numerics=numerics
    original_protocol=p._finite_ka_protocol
    def protocol(context):
        value=original_protocol(context)
        value.update(max_nfev_per_start=70 if context.config.full else 22,directivity='Fixed reference angular pattern; ideal retuned sources',setting_id=context.memo['setting']['id'])
        return value
    p._finite_ka_protocol=protocol

def single_specs(ctx):
    """Fresh matched cold commands, shared with the convergence figure."""
    bank=mf.feg_single_bank(ctx);fe_bank=mf._ORIGINAL_SINGLE_BANK(ctx)
    # Loading the FE control bank also registers legacy Figure 1 table names.
    # Restore the actual GFE-primary population used in the displayed figure.
    public=bank['table'].copy()
    public['termination_mode']=[ff._termination_mode(bank['runs'][(row.method,int(row.pair_index))],int(row.iteration_cap)) for row in public.itertuples(index=False)]
    public['primary_method']='GFE'
    ctx.tables['fig1_single_vortex_runs']=public
    ctx.tables['fig1_single_vortex_iterations_for_table_only']=public[['method','pair_index','seed','iterations','evaluations','iteration_cap']].copy()
    seed=int(bank['seeds'][0]);specs=[]
    for label,run in [('Conventional',bank['runs'][('Conventional',0)]),('FE',fe_bank['runs'][('FE',0)]),('FE+g',bank['runs'][('GFE',0)])]:
        specs.append(dict(method=label,pair_id=f'seed-{seed}',seed=seed,phase=np.asarray(run.phase_rad),target_m=p.MAIN_TARGET_M.copy(),phase_source='Fresh setting-specific matched cold solve',terminal_iteration=int(run.iterations),source_index=-1,command_time_s=float(run.history_wall_s[-1])))
    baselines=ff._single_matched_baseline_commands(ctx)
    for method,run in [('IB',baselines['ib']),('GS',baselines['audit']),('AD',ff._single_ad_command(ctx))]:
        specs.append(dict(method=method,pair_id='',seed=seed,phase=np.asarray(run.phase_rad),target_m=p.MAIN_TARGET_M.copy(),phase_source='Fresh fixed 0.45 lambda eight-point prescription',terminal_iteration=int(run.solver_result.iterations),source_index=-1,command_run=run))
    directory=ctx.output_root/'commands';directory.mkdir(exist_ok=True)
    for spec in specs:
        np.savez_compressed(directory/('Single_'+spec['method'].replace('+','plus')+'.npz'),phase_rad=spec['phase'],target_m=spec['target_m'])
    return specs

def verify_common_field(ctx):
    from hat_revision_pipeline.gorkov_core import formula_self_test
    guard=formula_self_test();assert guard['passed']
    rng=np.random.default_rng(124)
    points=np.array([[0.,0.,.05],[.01,-.006,.03],[-.008,.005,.06]])
    phase=rng.uniform(-np.pi,np.pi,256)
    a=transfer_matrix(points,ArrayGeometry(ctx.positions_m,ctx.normals),ctx.config.frequency_hz,source_scale_pa_m=SOURCE_SCALE_PA_M_PER_MURATA_UNIT)@np.exp(1j*phase)
    b=p._array_field(ctx,phase).pressure(points)
    error=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30));assert error<1e-10,error
    matrix=p._stencil_transfer(ctx.positions_m,p.MAIN_TARGET_M,ctx.config.frequency_hz)
    center=np.asarray(matrix)[2,2,2]@np.exp(1j*phase)
    assert np.isclose(center,b[0],rtol=1e-10,atol=1e-10)
    return dict(formula_guard=guard,field_relative_error=error,stencil_center_error=float(abs(center-b[0])))

def prepare(campaign,setting_id,mode='smoke'):
    setting=next(s for s in SETTINGS if s['id']==setting_id)
    positions,normals=geometry(setting)
    campaign=Path(campaign).resolve();output=campaign/setting_id;output.mkdir(parents=True,exist_ok=True)
    config=replace(RunConfig.for_mode('full' if mode=='full' else 'quick'),frequency_hz=setting['frequency_hz'],stencil_m=.0005*40000/setting['frequency_hz'],multitrap_maxiter=30000,exact_lmax=8 if mode=='full' else 6)
    source_hashes={str(path.relative_to(ROOT)):content_identity(path.read_bytes()).hexdigest() for path in [*sorted(ROOT.glob('*.py')),*sorted((ROOT/'runtime/src/hat_revision_pipeline').glob('*.py'))]}
    identity=dict(setting=setting,positions=digest_array(positions),normals=digest_array(normals),config=asdict(config),source_hashes=source_hashes,mode=mode)
    # Keep physics identity stable across reporting/adapter repairs. Never share settings.
    cache_identity=dict(setting=setting,positions=identity['positions'],normals=identity['normals'],config=asdict(config),numerics='lmax6-review22-roots-v1',physics_hashes={k:v for k,v in source_hashes.items() if Path(k).name in ('gorkov_core.py','multitrap.py','sota.py','exact_validator.py','diff_pat.py')})
    cache=CacheStore(output/'cache',namespace='gfe-generality-v1-'+digest_payload(cache_identity))
    ctx=GeneralityContext(config=config,project_root=ROOT/'runtime',data_root=output/'data',output_root=output,writer=p.ArtifactWriter(output),cache=cache,recompute=False,artifact=None,branch_bank=None,joint_frame=None,matched_frames=None,conventional_evolution=None,memo=dict(setting=setting,setting_positions_m=positions,setting_normals=normals,generality_mode=mode,notebook_mode=mode,notebook_root=ROOT))
    (output/'data').mkdir(exist_ok=True);install_physics(ctx);ff._single_fixed_endpoint_specs=single_specs
    import generality_branch,generality_triple,generality_tradeoff
    generality_branch.install(ctx);generality_triple.install(ctx)
    generality_tradeoff.install(ctx)
    from hat_revision_pipeline.style import configure_style
    configure_style()
    import run_study
    run_study._install_atomic_figure_exports()
    os.environ.update(HAT_QUICK_SINGLE_PAIRS='4',HAT_QUICK_FIG1_XZ_X_POINTS='3',HAT_QUICK_FIG1_XZ_Z_POINTS='3',HAT_QUICK_FIG1_DOMAIN_RESTARTS='1')
    checks=verify_common_field(ctx)
    manifest=dict(identity,started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),dependencies=dependency_snapshot(),field_checks=checks,finite_ka_protocol=p._finite_ka_protocol(ctx),native_timing_note='Fresh native synthesis wall times; one CPU affinity per setting worker when supplied.',paired_single_starts=30 if config.full else 4,root_starts=19 if config.full else 7,source_directivity_note='The reference 40-kHz angular pattern and source strength are held fixed at all carriers.')
    (output/'execution_settings.json').write_text(json.dumps(manifest,indent=2,default=str))
    np.savez_compressed(output/'array_geometry.npz',positions_m=positions,normals=normals,frequency_hz=config.frequency_hz)
    ctx.tables['array_geometry']=pd.DataFrame(np.c_[positions,normals],columns=['x_m','y_m','z_m','nx','ny','nz'])
    return ctx

def run(campaign,setting_id,mode='smoke',numbers=range(1,9)):
    ctx=prepare(campaign,setting_id,mode)
    import matplotlib.pyplot as plt
    import main_feg_benchmarks as b
    import generality_branch
    producers=(mf.figure_1_feg,generality_branch.figure_2,mf.figure_3_feg,mf.figure_4_feg,b.figure_5,b.figure_6,b.figure_7,b.figure_8)
    timings=[]
    progress_path=ctx.output_root/'progress.json'
    try: previous={int(row['figure']):row for row in json.loads(progress_path.read_text())}
    except (FileNotFoundError,json.JSONDecodeError): previous={}
    for number in numbers:
        print(f'{setting_id}: Figure {number} starting',flush=True);start=time.perf_counter()
        figure=producers[number-1](ctx);b.public_tables(ctx)
        # The original caption file includes old results; use current tables instead.
        (ctx.output_root/'main_gfe_benchmark_captions.json').unlink(missing_ok=True)
        p.export_all_tables(ctx)
        paths=sorted((ctx.output_root/'figures').glob(f'Figure_{number}_*.png'))
        if len(paths)!=1:raise RuntimeError(f'Expected one Figure {number}: {paths}')
        elapsed=time.perf_counter()-start;timings.append(dict(figure=number,elapsed_s=elapsed,path=str(paths[0])))
        print(f'{setting_id}: Figure {number} completed in {elapsed:.1f}s',flush=True)
        plt.close(figure)
        previous[number]=timings[-1]
        for available in sorted((ctx.output_root/'figures').glob('Figure_*_*.png')):
            index=int(available.name.split('_')[1])
            previous.setdefault(index,dict(figure=index,elapsed_s=None,path=str(available)))
        progress_path.write_text(json.dumps([previous[index] for index in sorted(previous)],indent=2))
    (ctx.output_root/'cache_access.json').write_text(json.dumps(ctx.cache.access_records,indent=2))
    return timings

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',default=str(ROOT/'outputs'));parser.add_argument('--setting',choices=[s['id'] for s in SETTINGS]);parser.add_argument('--mode',choices=['smoke','full'],default='smoke');parser.add_argument('--figures',default='1,2,3,4,5,6,7,8');parser.add_argument('--cpu',type=int);parser.add_argument('--settings-only',action='store_true');args=parser.parse_args()
    if args.cpu is not None and hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{args.cpu})
    if args.settings_only:export(args.campaign, mode=args.mode)
    elif args.setting:run(args.campaign,args.setting,args.mode,[int(v) for v in args.figures.split(',')])
    else:parser.error('--setting or --settings-only is required')
