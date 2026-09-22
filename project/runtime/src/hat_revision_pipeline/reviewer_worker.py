"""Guarded process entry for notebook-safe numerical worker pools.

The pickle is produced locally by reviewer_robustness from an already loaded
trusted study context. This is not a reader for arbitrary external pickles.
"""
from pathlib import Path
import json
import pickle
import sys


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    source,destination=map(Path,sys.argv[1:3])
    with source.open('rb') as stream:options=pickle.load(stream)
    from hat_revision_pipeline import reviewer_robustness as r
    if options.pop('probe',False):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        ctx=options['ctx']
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            jobs,_=pool.submit(r._continuous_one,ctx,[r._spec(seed=260828,parameter='worker_probe',parameter_value=1)],False).result(timeout=120)
        result=dict(iterations=jobs[0]['row']['iterations'],phase_content_id=jobs[0]['row']['phase_content_id'],raw_gtol_met=jobs[0]['row']['raw_gtol_met'])
    else:result=r.continuous_sensitivity(**options,_worker_process=True)
    destination.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
