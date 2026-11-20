"""Controlled NW=2, K=3, 5.1/10.1-s sensitivity on the same, already-audited pilot records.

All records with a computable quiet original 10.1-s spectrum are eligible,
including its SNR failures and alternative sensors. The shorter windows are
contained in those audited signal/noise intervals; no newly contaminated
or previously rejected waveform is rescued. This is a diagnostic experiment.
"""
from pathlib import Path
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
import numpy as np
import pandas as pd
from obspy import read,UTCDateTime
import instrument_processing as IP
from spectra import station_spectrum,fit_stack,fit
from multitaper import MTSpec
import spectra as spectral_engine

# This process-local estimator substitution preserves the common spectrum
# construction, response/path corrections, and one-sided PSD conversion.
TAPERS={}
def mt(x,dt):
    key=(len(x),float(dt))
    if key not in TAPERS:
        model=MTSpec(x,2,3,dt);TAPERS[key]=(model.vn,model.lamb)
    else:
        vn,lamb=TAPERS[key];model=MTSpec(x,2,3,dt,vn=vn,lamb=lamb)
    f,power=model.rspec()
    return f[:,0],power[:,0]
spectral_engine.mt=mt

R=Path(__file__).resolve().parents[1];P=R/'small_events';O=P/'nw2_windows'
SIGNATURE=''


def pulse_controls():
    rows=[];dt=.01
    for corner in [3.,8.,15.,30.]:
        for onset in [.1,.5,1.]:
            for duration in [5.1,10.1,20.1]:
                t=np.arange(round(duration/dt)+1)*dt
                u=np.maximum(t-onset,0)*np.exp(-2*np.pi*corner*np.maximum(t-onset,0));u[t<onset]=0
                u/=u.sum()*dt
                f,power=mt(u,dt);effective=len(u)*dt;grid=np.geomspace(.5,20,200)
                amp=np.sqrt(effective*power/2)
                y=np.interp(np.log10(grid),np.log10(f[1:]),np.log10(np.maximum(amp[1:],1e-100)))
                ff=np.fft.rfftfreq(2*len(u)+1,dt);aa=dt*abs(np.fft.rfft(u-u.mean(),n=2*len(u)+1))
                yf=np.interp(np.log10(grid),np.log10(ff[1:]),np.log10(np.maximum(aa[1:],1e-100)))
                for method,curve in [('Multitaper',y),('Fourier',yf)]:
                    fitted,_=fit(grid,curve,grid>=.5,depth_km=10.)
                    rows.append(dict(corner=corner,onset=onset,duration=duration,method=method,
                        delta_Mw=np.log10(fitted['omega_r_m2s'])/1.5))
    d=pd.DataFrame(rows);d.to_csv(P/'tables/nw2_pulse_controls.csv',index=False)
    pivot=d.pivot(index=['corner','onset','method'],columns='duration',values='delta_Mw')
    change=pivot[[5.1,10.1]].sub(pivot[20.1],axis=0)
    return dict(unit_area_input=True,noise_free=True,known_plateau=1.,cases=len(d),band_Hz=[.5,20],nw=2,kspec=3,
        maximum_absolute_delta_Mw_short_minus_20s=float(abs(change.to_numpy()).max()),
        maximum_absolute_Mw_error_vs_true_input=float(abs(d.delta_Mw).max()),
        caveat='Pulse area is known after discrete normalization. This tests temporal/spectral operators, not path coda, real-noise selection, attenuation or site calibration. Absolute errors include source-shape/discretization effects; no correction is fitted.')


def process(case):
    eid=case['event_id'];path=O/'events'/f'{eid}.json'
    if path.exists():
        try:old=json.loads(path.read_text())
        except ValueError:old={}
        if old.get('config_sha256')==SIGNATURE:return dict(event_id=eid,status='cached')
    start=time.monotonic();fits=[];arrays={};curves={};failures={};meta=[]
    try:
        log=pd.read_csv(P/'station_logs'/f'{eid}.csv')
        if 'window_s' not in log or 'status' not in log:raise ValueError('No retained original-window records')
        log=log[log.window_s.eq(10.1)&log.status.isin(['ready','alternative'])].drop_duplicates(['station','sensor'])
        picks=pd.read_csv(case['picks_csv']);picks=picks[picks.Probability.ge(.3)].sort_values('Probability',ascending=False)
        phases={}
        for r in picks.to_dict('records'):phases.setdefault((r['Network'],r['Station'],r['Phase']),UTCDateTime(r['Time']))
        spectra={duration:[] for duration in [5.1,10.1]};records={duration:[] for duration in spectra}
        for row in log.itertuples():
            net,sta=row.station.split('.')
            try:
                pp,ss=phases[net,sta,'P'],phases[net,sta,'S']
                traces=[]
                for component in 'ZNE':
                    raw=read(str(Path(case['waveform_dir'])/f'{eid}.{net}.{sta}.{row.sensor}{component}.sac'))[0]
                    label,ch,units=IP.response(raw,UTCDateTime(case['origin_utc']))
                    trace,_=IP.displacement(raw,ch);traces.append(trace)
                for duration in spectra:
                    f,a,m=station_spectrum(traces,pp,ss,row.distance_m,float(ss-UTCDateTime(case['origin_utc'])),
                        window=duration,noise_shift=row.noise_shift_s)
                    keep=f>=.5
                    spectra[duration].append(a[:,keep])
                    records[duration].append(dict(station=row.station,sensor=row.sensor,**m))
            except Exception as exc:
                key=type(exc).__name__+':'+str(exc)[:120];failures[key]=failures.get(key,0)+1
        for duration,values in spectra.items():
            if not values:continue
            arr=np.asarray(values);rec=records[duration];tag=f'T{duration:g}';ff=f[f>=.5]
            arrays[tag+'_allrecords']=arr
            for sensor_set,allowed in [('ALL',['HH','EL','HG']),('HH_EL',['HH','EL']),('HG',['HG']),('HH',['HH'])]:
                chosen={}
                for i in sorted(range(len(rec)),key=lambda i:({'HH':0,'EL':1,'HG':2}[rec[i]['sensor']],rec[i]['station'])):
                    if rec[i]['sensor'] not in allowed or np.isfinite(arr[i,0]).sum()<20:continue
                    chosen.setdefault(rec[i]['station'],i)
                ids=list(chosen.values())
                if not ids:continue
                result,cc=fit_stack(ff,arr[ids],depth_km=case['z_central'])
                fits.extend(dict(event_id=eid,window_s=duration,sensor_set=sensor_set,**r) for r in result)
                for c in cc:
                    key=tag+'_'+sensor_set+'_'+c['method'].replace(' ','_')+'_'+c['shape']
                    for field in ['observed','predicted','good','count']:curves[key+'_'+field]=c[field]
                if sensor_set=='ALL':
                    arrays[tag]=arr[ids]
                    arrays[tag+'_stations']=np.array([rec[i]['station'] for i in ids])
                    arrays[tag+'_sensor']=np.array([rec[i]['sensor'] for i in ids])
        if arrays:np.savez_compressed(O/'spectra'/f'{eid}.npz',frequency_Hz=ff,**arrays,**curves)
        result=dict(event_id=eid,config_sha256=SIGNATURE,status='processed' if fits else 'no_usable_fit',
            fits=fits,rejections=failures,raw_usable_triplets=len(log),seconds=time.monotonic()-start,
            basis='All ready/alternative original 10.1-s records; shorter signal and pre-P noise intervals are contained in the original quiet intervals')
    except Exception as exc:
        result=dict(event_id=eid,config_sha256=SIGNATURE,status='no_retained_records',fits=[],
                    error=type(exc).__name__+':'+str(exc),seconds=time.monotonic()-start)
    path.write_text(json.dumps(result,indent=2)+'\n')
    return dict(event_id=eid,status=result['status'],fits=len(fits),seconds=result['seconds'])


def main():
    global SIGNATURE
    for folder in ['events','spectra']:(O/folder).mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),R/'code/instrument_processing.py',R/'code/spectra.py',R/'code/source_velocity.py',
             P/'inputs/pilot_config.json',P/'inputs/processing_provenance.json']
    SIGNATURE=hashlib.sha256(b''.join(path.read_bytes() for path in sources)).hexdigest()
    controls=pulse_controls()
    assert controls['maximum_absolute_delta_Mw_short_minus_20s']<.1
    config=dict(signature=SIGNATURE,windows_s=[5.1,10.1],band_Hz=[.5,20],nw=2,kspec=3,
        reason='NW=2 gives half-bandwidth about 0.39 Hz for the 5.1-s window; three tapers retain the original common 0.5-20-Hz band.',
        acceptance='Use the same A/B station, plateau, estimator/source-shape/window thresholds. Real ML>=2 controls assess transfer; no ad hoc amplitude adjustment or adopted-catalog replacement.',
        diagnostic_only=True,pulse_controls=controls)
    (O/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    cohort=pd.read_csv(P/'inputs/pilot_cohort.csv',dtype={'event_id':str})
    IP.load()
    for n in [511,1011]:mt(np.sin(np.arange(n)*.13)+.01*np.cos(np.arange(n)*.017),.01)
    start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('fork')) as pool:
        jobs={pool.submit(process,c):c['event_id'] for c in cohort.to_dict('records')}
        for i,future in enumerate(concurrent.futures.as_completed(jobs),1):
            try:row=future.result()
            except Exception as exc:row=dict(event_id=jobs[future],status='worker_error',error=str(exc))
            print(i,'/',len(cohort),f'{time.monotonic()-start:.0f}s',json.dumps(row),flush=True)
    print('NW2 window control complete',flush=True)


if __name__=='__main__':main()
