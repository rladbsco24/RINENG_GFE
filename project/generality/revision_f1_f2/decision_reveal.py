"""Cache wider views of existing native decision sections, without optimization."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from appendix_io import save_npz,write_bytes
from decision_data import objectives,METHODS
from hat_revision_pipeline.cache import digest_array
import final_decision_data as native

ROOT=Path(__file__).resolve().parent

def sample(arguments):
    sid,half_range=arguments
    native.HALF_RANGE=float(half_range)
    destination=ROOT/f'data/decision_reveal/range{half_range:g}'
    path=destination/f'{sid}_decision.npz'
    meta=destination/f'{sid}_decision_metadata.json'
    if path.exists() and meta.exists(): return json.loads(meta.read_text())
    with np.load(ROOT/f'data/decision/{sid}_decision.npz') as original:
        arrays={k:original[k].copy() for k in original.files}
    objectives_,phases,_,_=objectives(ROOT/f'data/solution_correspondence/campaign/{sid}')
    assert np.array_equal(phases,arrays['endpoint_phase_rad'])
    planes,us,vs,rows=[],[],[],[]
    for i,(method,objective,phase) in enumerate(zip(METHODS,objectives_,phases)):
        basis=arrays['directions'][i]
        q,plane=native._section(objective,phase,basis)
        arrow_q,u,v=native._arrows(objective,phase,basis,q)
        planes.append(plane);us.append(u);vs.append(v)
        mid=len(q)//2
        location=np.unravel_index(np.argmin(plane),plane.shape)
        boundary=np.r_[plane[0],plane[-1],plane[:,0],plane[:,-1]]
        scale=float(np.quantile(plane,.985))
        rows.append(dict(method=method,phase_fingerprint=digest_array(phase),directions_fingerprint=digest_array(basis),
            centerline_span_ratio=float(np.ptp(plane[mid,:])/np.ptp(plane[:,mid])),
            center_above_sampled_min=float(plane[mid,mid]),
            sampled_minimum_coordinates_rad=[float(q[location[1]]),float(q[location[0]])],
            minimum_boundary_over_color_scale=float(boundary.min()/scale),color_scale=scale,
            maximum_corner_source_phase_change_rad=float(half_range*np.max(abs(basis[0])+abs(basis[1])))))
    arrays.update(q=q,plane_delta_j=np.stack(planes),arrow_q=arrow_q,descent_u=np.stack(us),descent_v=np.stack(vs))
    save_npz(path,**arrays)
    record=dict(setting_id=sid,half_range_l2_rad=half_range,grid_size=len(q),endpoints=rows,
        phase_and_native_basis_unchanged=True,optimizer_calls=0,finite_ka_calls=0,
        viewing_scope='Same equal-axis range for both compared methods; analytic values with linear Q98.5 colors')
    write_bytes(meta,(json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps(record),flush=True)
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--half-range',type=float,default=10)
    parser.add_argument('--ids',default='S02,A09,F20,S06,F28')
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(sample,[(sid,args.half_range) for sid in args.ids.split(',')]))
