"""Resumable expanded study; original and NW2 methods, unchanged engine files."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
import time
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/korean_mw_ml_numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/korean_mw_ml_matplotlib')
import numpy as np
import pandas as pd
import instrument_processing as IP
import spectra as S
BASELINE_MT=S.mt
import run_catalog as W
import local_magnitude as L
import small_event_nw2_windows as N
S.mt=BASELINE_MT

R=Path(__file__).resolve().parents[1];E=R/'small_events/extension'
SIGNATURE='';NEIGHBORS={}


def process(case):
    eid=case['event_id'];start=time.monotonic();case=dict(case,neighbors=NEIGHBORS[eid])
    base=Path(case['baseline_root']);candidate=Path(case['candidate_root'])
    if base==E:
        S.mt=BASELINE_MT
        if not case['input_available']:
            for folder,sig in [('events',W.CONFIG_SHA),('ml_events',L.CONFIG_SHA)]:
                (E/folder/f'{eid}.json').write_text(json.dumps(dict(event_id=eid,config_sha256=sig,status='missing_input',fits=[],summary={}))+'\n')
            (E/'station_logs'/f'{eid}.csv').write_text('event_id,status,reason\n'+eid+',rejected,missing_input\n')
        else:
            W.run_event(case)
            try:L.run_event(case)
            except Exception as exc:
                (E/'ml_events'/f'{eid}.json').write_text(json.dumps(dict(event_id=eid,config_sha256=L.CONFIG_SHA,summary={},error=type(exc).__name__+':'+str(exc)))+'\n')
    if candidate==E/'nw2_windows':
        S.mt=N.mt;N.P=base
        N.process(case)
    S.mt=BASELINE_MT
    raw=json.loads((base/'events'/f'{eid}.json').read_text())
    alt=json.loads((candidate/'events'/f'{eid}.json').read_text())
    local=json.loads((base/'ml_events'/f'{eid}.json').read_text())
    return dict(event_id=eid,baseline_status=raw.get('status'),NW2_status=alt.get('status'),
        baseline_reused=base!=E,candidate_reused=candidate!=E/'nw2_windows',
        ML=local.get('summary',{}).get('ML_Sheen'),ML_grade=local.get('summary',{}).get('ML_quality'),
        seconds=time.monotonic()-start)


def main():
    global SIGNATURE,NEIGHBORS
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=16);ap.add_argument('--limit',type=int);ap.add_argument('--events',nargs='*')
    args=ap.parse_args();cohort=pd.read_csv(E/'inputs/cohort.csv',dtype={'event_id':str})
    config=json.loads((E/'inputs/config.json').read_text())
    assert hashlib.sha256((E/'inputs/cohort.csv').read_bytes()).hexdigest()==config['cohort_sha256']
    if args.events:cohort=cohort[cohort.event_id.isin(args.events)]
    if args.limit:cohort=cohort.head(args.limit)
    NEIGHBORS=json.loads((E/'inputs/neighbors.json').read_text())
    paths=[Path(__file__),R/'code/run_catalog.py',R/'code/local_magnitude.py',R/'code/instrument_processing.py',
           R/'code/spectra.py',R/'code/source_velocity.py',R/'code/small_event_nw2_windows.py',
           Path(L.RECIPE.__file__),R/'inputs/source_velocity_models.json',R/'inputs/response_sources.json',
           E/'inputs/config.json',E/'inputs/reuse_registry.csv']
    SIGNATURE=hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
    W.HERE=E;W.CONFIG_SHA='extension-NW4-'+SIGNATURE
    L.ROOT=E;L.CONFIG_SHA='extension-ML-'+SIGNATURE
    N.O=E/'nw2_windows';N.SIGNATURE='extension-NW2-'+SIGNATURE
    provenance=dict(signature=SIGNATURE,cohort_sha256=config['cohort_sha256'],
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        workers=args.workers,adopted_project_modified=False)
    (E/'inputs/processing_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    IP.load()
    for n in [1011,2011]:BASELINE_MT(np.sin(np.arange(n)*.13)+.01*np.cos(np.arange(n)*.017),.01)
    for n in [511,1011]:N.mt(np.sin(np.arange(n)*.13)+.01*np.cos(np.arange(n)*.017),.01)
    S.mt=BASELINE_MT
    start=time.monotonic();progress=[]
    print('Starting',len(cohort),'cases;',args.workers,'workers; unchanged gates',flush=True)
    with concurrent.futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        jobs={pool.submit(process,case):case['event_id'] for case in cohort.to_dict('records')}
        for i,job in enumerate(concurrent.futures.as_completed(jobs),1):
            try:row=job.result()
            except Exception as exc:row=dict(event_id=jobs[job],baseline_status='worker_error',error=repr(exc))
            progress.append(row)
            print(i,'/',len(cohort),f'elapsed {time.monotonic()-start:.0f}s',json.dumps(row),flush=True)
            if i%10==0 or i==len(cohort):pd.DataFrame(progress).to_csv(E/'tables/processing_progress.csv',index=False)
    print('Expanded waveform processing complete',flush=True)


if __name__=='__main__':main()
