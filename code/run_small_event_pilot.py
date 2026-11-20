"""Run the frozen small-event pilot with the existing audited measurement engines."""
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
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
import run_catalog as W
import local_magnitude as L
import instrument_processing as IP

R=Path(__file__).resolve().parents[1];P=R/'small_events'


def process(case):
    eid=case['event_id'];start=time.monotonic()
    if not case['input_available']:
        for folder in ['events','ml_events']:
            (P/folder/f'{eid}.json').write_text(json.dumps(dict(event_id=eid,status='missing_input',fits=[],summary={},reason='No retained picks/waveforms'))+'\n')
        return dict(event_id=eid,status='missing_input')
    mw=W.run_event(case)
    try:ml=L.run_event(case)
    except Exception as exc:
        ml=dict(event_id=eid,status='error',error=type(exc).__name__+':'+str(exc))
        (P/'ml_events'/f'{eid}.json').write_text(json.dumps(dict(event_id=eid,config_sha256=L.CONFIG_SHA,summary={},error=ml['error']))+'\n')
    return dict(event_id=eid,status=mw['status'],n_spectral_fits=mw.get('fits',0),
        measured_ML=ml.get('ML_Sheen'),ML_sites=ml.get('ML_Sheen_n',0),seconds=time.monotonic()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--events',nargs='*');args=parser.parse_args()
    config=P/'inputs/pilot_config.json'
    cohort=pd.read_csv(P/'inputs/pilot_cohort.csv',dtype={'event_id':str})
    if args.events:cohort=cohort[cohort.event_id.isin(args.events)]
    cases=cohort.to_dict('records')
    catpath=R.parent/'data/necis/catalog/necis_events_domestic_1978_2026.csv'
    cat=pd.read_csv(catpath)
    cat['seconds']=pd.to_datetime(cat.origin_utc,utc=True).astype('int64')/1e9
    for case in cases:
        t=float(UTCDateTime(case['origin_utc']));near=cat[abs(cat.seconds-t)<=180]
        case['neighbors']=[]
        for other in near.to_dict('records'):
            delta=other['seconds']-t
            if abs(delta)<.5:continue
            if gps2dist_azimuth(case['kma_lat'],case['kma_lon'],other['latitude'],other['longitude'])[0]/1000<=50:
                case['neighbors'].append(dict(dt_s=delta,lat=other['latitude'],lon=other['longitude'],ml=other['magnitude']))
    sourcepaths=[Path(__file__),R/'code/run_catalog.py',R/'code/local_magnitude.py',
        R/'code/instrument_processing.py',R/'code/spectra.py',R/'code/source_velocity.py',
        Path(L.RECIPE.__file__),R/'inputs/response_sources.json',R/'inputs/source_velocity_models.json',config]
    signature=hashlib.sha256(b''.join(path.read_bytes() for path in sourcepaths)).hexdigest()
    W.HERE=P;W.CONFIG_SHA='small-Mw-'+signature
    L.ROOT=P;L.CONFIG_SHA='small-ML-'+signature
    IP.load()
    for n in [1011,2011]:W.mt(np.sin(np.arange(n)*.13)+.01*np.cos(np.arange(n)*.017),.01)
    provenance=dict(signature=signature,cohort_sha256=hashlib.sha256((P/'inputs/pilot_cohort.csv').read_bytes()).hexdigest(),
        processing_sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sourcepaths},
        neighbor_catalog=str(catpath),neighbor_catalog_sha256=hashlib.sha256(catpath.read_bytes()).hexdigest(),
        known_neighbors_all_magnitudes=True,reference_project_modified=False,workers=args.workers)
    (P/'inputs/processing_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (P/'inputs/neighbor_counts.csv').write_text(pd.DataFrame([dict(event_id=c['event_id'],known_neighbors=len(c['neighbors']),neighbors_below_ML1p5=sum(n['ml']<1.5 for n in c['neighbors'])) for c in cases]).to_csv(index=False))
    print('Starting',len(cases),'events with',args.workers,'workers; same waveform engines, all-known-event overlap screen',flush=True)
    start=time.monotonic();status=[]
    with concurrent.futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(process,case):case['event_id'] for case in cases}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            try:row=future.result()
            except Exception as exc:row=dict(event_id=futures[future],status='worker_error',error=str(exc))
            status.append(row)
            print(i,'/',len(cases),f'elapsed {time.monotonic()-start:.0f}s',json.dumps(row),flush=True)
            pd.DataFrame(status).to_csv(P/'tables/processing_progress.csv',index=False)
    print('Pilot waveform processing complete',flush=True)


if __name__=='__main__':main()
