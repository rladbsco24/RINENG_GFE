"""Align Single field and native decision views with the displayed ACS endpoints."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from appendix_settings import ROOT, SETTINGS
from appendix_io import save_npz, write_bytes
sys.path.insert(0,str(ROOT.parent/'generality_package/runtime/src'))
from hat_revision_pipeline.cache import digest_array
from pressure_data import extract
from decision_data import analyze

DATA=ROOT/'data'
OUT=DATA/'solution_correspondence'
SEED=260828
METHODS=('Conventional','FE','GFE')

def resolve(source):
    path=Path(source)
    if not path.is_absolute():path=ROOT/path
    if path.exists():return path
    if 'acs_longer/baseline' in path.as_posix():
        alternate=DATA/'endpoint_commands'/('baseline_'+path.name)
        if alternate.exists():return alternate
    raise FileNotFoundError(path)

def history(path,endpoint):
    with np.load(path) as z:
        phase=z['phase_rad']
        if not np.allclose(np.exp(1j*phase),np.exp(1j*endpoint),atol=1e-10,rtol=0):
            return None
        key='history_phase_rad' if 'history_phase_rad' in z else 'history_phase_tail_rad'
        tail=z[key][-20:]
        iterations=int(z['iterations'])
    assert tail.shape[0]>=2 and np.allclose(np.exp(1j*tail[-1]),np.exp(1j*endpoint),atol=1e-10,rtol=0)
    return dict(phase_rad=endpoint,history_phase_tail_rad=tail,iterations=iterations)

def reference_history(sid,method,seed,alpha,source,endpoint):
    recovered=OUT/'recovered_histories'/f'{sid}_{method}_seed{seed}.npz'
    if recovered.exists():
        return history(recovered,endpoint),recovered
    try:
        path=resolve(source)
        run=history(path,endpoint)
        if run is not None:return run,path
    except FileNotFoundError:
        pass
    if seed==SEED:
        path=DATA/'decision'/f'{sid}_decision.npz'
        with np.load(path) as z:
            index=list(z['methods']).index(method)
            if digest_array(z['endpoint_phase_rad'][index])==digest_array(endpoint):
                tail=z[f'history_{method}_tail_rad'].copy()
                iterations=int(z['iterations'][index])
                return dict(phase_rad=endpoint,history_phase_tail_rad=tail,iterations=iterations),path
    # Older archives retain the endpoint but can omit its original history.
    # Reproduce the native solve and require an exact phase hash before use.
    from tune_alpha import objective_for_case
    from hat_revision_pipeline.sota import solve_corrected_gorkov_fe
    from extend_acs import save_run
    objective=objective_for_case(sid,method,alpha)
    initial=np.random.default_rng(seed).uniform(-np.pi,np.pi,objective.n_transducers)
    run=solve_corrected_gorkov_fe(objective,initial,maxiter=10000,gtol=1e-8)
    assert digest_array(run.phase_rad)==digest_array(endpoint),(sid,method,seed,'Native history reproduction differs')
    save_run(recovered,run,seed=seed,alpha_per_m=alpha,reference_phase_hash_verified=True)
    return history(recovered,endpoint),recovered

def conventional_history(sid,endpoint):
    candidates=[]
    folders=[DATA/'jump_conventional/runs']
    if sid=='S02':folders.insert(0,DATA/'opposed_budget/512/single_native512/data/jump_conventional/runs')
    if sid=='A07':folders.insert(0,DATA/'annular_budget/512/single/data/jump_conventional/runs')
    for folder in folders:
        candidates.extend(sorted(folder.glob(f'{sid}_restart_seed{SEED}_stage*.npz'),reverse=True))
    for path in candidates:
        run=history(path,endpoint)
        if run is not None:return run,path
    raise FileNotFoundError(f'{sid}: no recorded accepted history matches the displayed terminal Conventional endpoint')

def build_case(sid):
    with np.load(DATA/'compact_acs/traces'/f'{sid}_Conventional_seed{SEED}.npz') as z:
        conventional=z['terminal_phase_rad'].copy()
    with np.load(DATA/'solution_set_acs/references'/f'{sid}_bank.npz') as z:
        bank={k:z[k] for k in z.files}
    scores=np.abs(np.exp(1j*(conventional-bank['reference_phase_rad'])).mean(axis=1))
    selected={method:int(np.flatnonzero(bank['reference_methods']==method)[np.argmax(scores[bank['reference_methods']==method])]) for method in METHODS[1:]}
    winner=int(np.argmax(scores))
    row=dict(setting_id=sid,method=str(bank['reference_methods'][winner]),seed=SEED,
        reference_seed=int(bank['reference_seeds'][winner]),reference_id=str(bank['reference_ids'][winner]),
        phase_fingerprint=str(bank['reference_phase_fingerprint'][winner]),
        conventional_phase_fingerprint=digest_array(conventional),bank_fingerprint=str(bank['bank_fingerprint']),
        alpha_per_m=float(bank['reference_alpha_per_m'][winner]),terminal_phase_cosine=float(scores[winner]),
        criterion='Nearest member of the frozen eight-endpoint FE/GFE bank to the terminal Conventional command for the fixed seed260828')
    record_path=OUT/'records'/f'{sid}.json'
    if record_path.exists():
        old=json.loads(record_path.read_text())
        assert old['selection']==row
        return row
    case=OUT/'campaign'/sid
    case.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'campaign'/sid/'array_geometry.npz',case/'array_geometry.npz')
    with np.load(DATA/'pressure'/f'{sid}_pressure.npz') as z:target=z['target_m'].copy()
    run,path=conventional_history(sid,conventional)
    save_npz(case/'commands/Single_Conventional.npz',phase_rad=conventional,target_m=target,alpha_per_m=0.)
    save_npz(case/'paired_single/Conventional_pair0.npz',**run)
    provenance=[dict(method='Conventional',phase_fingerprint=digest_array(conventional),source=str(path.relative_to(ROOT)))]
    for method,index in selected.items():
        phase=bank['reference_phase_rad'][index]
        run,path=reference_history(sid,method,int(bank['reference_seeds'][index]),float(bank['reference_alpha_per_m'][index]),str(bank['reference_source_paths'][index]),phase)
        assert run is not None
        name='FEplusg' if method=='GFE' else method
        alpha=float(bank['reference_alpha_per_m'][index])
        save_npz(case/'commands'/f'Single_{name}.npz',phase_rad=phase,target_m=target,alpha_per_m=alpha)
        save_npz(case/'paired_single'/f'{method}_pair0.npz',**run)
        provenance.append(dict(method=method,phase_fingerprint=digest_array(phase),reference_id=str(bank['reference_ids'][index]),
            phase_cosine=float(scores[index]),source=str(path.relative_to(ROOT))))
    extract(sid,OUT/'pressure',input_root=OUT/'campaign')
    analyze(case,OUT/'decision')
    with np.load(OUT/'pressure'/f'{sid}_pressure.npz') as p,np.load(OUT/'decision'/f'{sid}_decision.npz') as d:
        for index,item in enumerate(provenance):
            assert item['phase_fingerprint']==digest_array(p['phase_rad'][index])==digest_array(d['endpoint_phase_rad'][index])
        assert np.array_equal(p['target_m'],d['target_m'])
    write_bytes(record_path,json.dumps(dict(selection=row,commands=provenance),indent=2).encode())
    print(sid,row['method'],row['terminal_phase_cosine'],flush=True)
    return row

def publish():
    rows=[json.loads((OUT/'records'/f"{s['id']}.json").read_text())['selection'] for s in SETTINGS]
    for folder in ('pressure','decision'):
        for source in (OUT/folder).glob('*'):
            target=DATA/folder/source.name
            backup=OUT/'previous_cold_inputs'/folder/source.name
            if target.exists() and not backup.exists():
                backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
            shutil.copy2(source,target)
    write_bytes(OUT/'selection.csv',pd.DataFrame(rows).to_csv(index=False).encode())
    from representative_selection import write_selection,write_alignment
    write_selection(DATA);write_alignment(DATA)

def validate(data_root,setting_ids=None):
    root=Path(data_root)
    selection=pd.read_csv(root/'solution_correspondence/selection.csv').set_index('setting_id')
    ids=list(selection.index) if setting_ids is None else list(setting_ids)
    records=[]
    for sid in ids:
        record=json.loads((root/'solution_correspondence/records'/f'{sid}.json').read_text())
        with np.load(root/'pressure'/f'{sid}_pressure.npz') as p,np.load(root/'decision'/f'{sid}_decision.npz') as d, \
             np.load(root/'solution_set_acs/references'/f'{sid}_bank.npz') as bank:
            assert str(bank['bank_fingerprint'])==record['selection']['bank_fingerprint']
            assert np.array_equal(p['target_m'],d['target_m'])
            for index,item in enumerate(record['commands']):
                actual=digest_array(p['phase_rad'][index])
                assert actual==item['phase_fingerprint']==digest_array(d['endpoint_phase_rad'][index])
                if index==0:assert actual==record['selection']['conventional_phase_fingerprint']
                else:assert actual in bank['reference_phase_fingerprint'].tolist()
                records.append(dict(setting_id=sid,**item,field_decision_hash_match=True))
    return dict(passed=True,task='Single',settings=ids,records=records,
        endpoint_scope='Terminal Conventional for fixed seed260828; nearest FE and GFE members of the unchanged eight-endpoint bank',
        benchmark_scope='The separate Triple benchmark retains all five methods and its own commands')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',default=','.join(s['id'] for s in SETTINGS))
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.workers>1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(build_case,args.settings.split(',')))
    else:
        for sid in args.settings.split(','):build_case(sid)
    if args.publish:publish()
