"""Read-only national waveform processing with resumable event results."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,multiprocessing,os,sys,time,warnings
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/korean_mw_ml_numba')
import numpy as np
import pandas as pd
from obspy import read,read_inventory,UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from spectra import station_spectrum,fit_stack,mt
import instrument_processing as IP

HERE=Path(__file__).resolve().parents[1]
INV={};META={};CONFIG_SHA=''
for sub in ['events','station_logs']:(HERE/sub).mkdir(exist_ok=True)

def config_signature():
    paths=['inputs/study_config.json','code/instrument_processing.py','code/run_catalog.py',
           'code/spectra.py','code/source_velocity.py','inputs/source_velocity_models.json',
           'inputs/response_sources.json']
    return hashlib.sha256(b''.join((IP.ROOT/p).read_bytes() for p in paths)).hexdigest()

def index_inventory(inv,label):
    for net in inv:
        for sta in net:
            for ch in sta:
                if ch.code[:2] not in ['HH','EL']:continue
                key=(net.code,sta.code,ch.code)
                INV.setdefault(key,[]).append((label,ch))

def response(tr,when):
    label,ch,units=IP.response(tr,when)
    return label,ch

def collision(interval,others):
    return any(interval[0]<b and interval[1]>a for a,b in others)

def run_event(case):
    eid=case['event_id'];output=HERE/'events'/(eid+'.json')
    if output.exists():
        old=json.loads(output.read_text())
        if old.get('config_sha256')==CONFIG_SHA:return dict(event_id=eid,status='cached',fits=len(old.get('fits',[])))
    start=time.monotonic();origin=UTCDateTime(case['origin_utc']);logs=[];records=[];fits=[];curve_data={};arrays={};checks=[]
    errors={};hashes={}
    def fail(reason):errors[reason]=errors.get(reason,0)+1
    try:
        p=pd.read_csv(case['picks_csv'])
        if 'Probability' in p:p=p[p.Probability>=.3]
        phase={}
        for r in p.sort_values('Probability',ascending=False).to_dict('records'):
            phase.setdefault((r['Network'],r['Station'],r['Phase']),UTCDateTime(r['Time']))
        roots=Path(case['waveform_dir']);groups={}
        for path in sorted(roots.glob('*.sac')):
            a=path.name.split('.')
            if len(a)!=5:continue
            _,net,sta,ch,_=a
            if ch[:2] in ['HH','EL','HG'] and ch[-1] in 'ZNE':groups.setdefault((net,sta,ch[:2]),{})[ch[-1]]=path
        done=set()
        for (net,sta,sensor),paths in sorted(groups.items(),key=lambda v:({'HH':0,'EL':1,'HG':2}[v[0][2]],v[0])):
            if IP.quarantined(eid,net,sta):fail('unresolved_colocated_response');continue
            if not all(c in paths for c in 'ZNE'):fail('incomplete_triplet');continue
            if (net,sta,'P') not in phase or (net,sta,'S') not in phase:fail('missing_usable_P_or_S');continue
            pp,ss=phase[net,sta,'P'],phase[net,sta,'S'];tp=float(pp-origin);ts=float(ss-origin)
            if not 0<tp<ts or ts-tp>60:fail('invalid_phase_times');continue
            try:
                traces=[read(paths[c])[0] for c in 'ZNE'];z=traces[0]
                lat,lon=float(z.stats.sac.stla),float(z.stats.sac.stlo)
                d_kma=gps2dist_azimuth(case['kma_lat'],case['kma_lon'],lat,lon)[0]/1000
                if d_kma>case.get('radius_km',100.):continue
                epic,az,baz=gps2dist_azimuth(case['lat'],case['lon'],lat,lon)
                distance=float(np.hypot(epic,case['z_central']*1000))
                if not IP.p_valid(distance/1000,tp):raise ValueError('P_timing_outside_4_8p5_km_s')
                sources=[];epochs=[];postissue=False;converted=[];nwarnings=0;gain_ratios=[]
                others=[]
                for other in case.get('neighbors',[]):
                    r_other=np.hypot(gps2dist_azimuth(other['lat'],other['lon'],lat,lon)[0]/1000,case['z_central'])
                    dr=r_other-distance/1000
                    other_p=other['dt_s']+tp+dr/6.0
                    other_s=other['dt_s']+ts+dr/3.5
                    tail=30. if other['ml']>=4.5 else (20. if other['ml']>=4 else 10.)
                    others.append((other_p-1.,other_s+tail))
                for c,tr in zip('ZNE',traces):
                    label,ch=response(tr,origin)
                    gain_ratios.append(getattr(ch,'_study_gain_ratio',float('nan')))
                    if float(tr.stats.delta)>.025:raise ValueError('sample_rate_too_low')
                    fragment=tr.slice(ss-.1,ss+20).data
                    if not len(fragment) or not np.isfinite(fragment).all():raise ValueError('invalid_signal_waveform')
                    if max(abs(float(fragment.min())),abs(float(fragment.max())))>=.98*2**23:
                        extreme=(fragment==fragment.max())|(fragment==fragment.min())
                        if np.convolve(extreme.astype(int),np.ones(5,dtype=int),mode='valid').max()>=5:raise ValueError('possible_ADC_clipping')
                    rawzero=(fragment==0)
                    if len(rawzero)>50 and np.convolve(rawzero.astype(int),np.ones(50,dtype=int),mode='valid').max()>=50:raise ValueError('signal_zero_gap')
                    hashes[str(paths[c])]=hashlib.sha256(paths[c].read_bytes()).hexdigest()
                    tr,warn=IP.displacement(tr,ch)
                    nwarnings+=len(warn)
                    converted.append(tr);sources.append(label)
                    epochs.append(str(ch.start_date))
                    postissue|=label=='master' and origin>UTCDateTime('2024-06-26T23:59:59')
                done.add((net,sta))
                records.append(dict(station=net+'.'+sta,sensor=sensor,traces=converted,p_abs=pp,s_abs=ss,
                  s_time=ts,p_time=tp,distance=distance,epicentral_km=epic/1000,azimuth=az,
                  response_source=';'.join(sorted(set(sources))),response_epoch=';'.join(epochs),
                  master_postissue=postissue,response_warnings=nwarnings,other_intervals=others,
                  response_gain_ratio_min=float(np.nanmin(gain_ratios)),response_gain_ratio_max=float(np.nanmax(gain_ratios))))
            except Exception as e:
                fail(type(e).__name__+':'+str(e)[:120])
                logs.append(dict(event_id=eid,station=net+'.'+sta,sensor=sensor,status='rejected',reason=str(e)[:200]))
        for duration in [10.1,20.1]:
            amp=[];meta=[]
            for rec in records:
                try:
                    lo=rec['s_time']-.1;hi=lo+duration
                    if collision((lo,hi),rec['other_intervals']):raise ValueError('catalog_signal_overlap')
                    noise_shift=None
                    for shift in [-1.,-11.,-21.,-31.,-41.]:
                        end=rec['p_time']+shift
                        if collision((end-duration,end),rec['other_intervals']):continue
                        if all(t.stats.starttime<=rec['p_abs']+shift-duration for t in rec['traces']):
                            noise_shift=shift;break
                    if noise_shift is None:raise ValueError('no_quiet_noise_window')
                    f,a,m=station_spectrum(rec['traces'],rec['p_abs'],rec['s_abs'],rec['distance'],rec['s_time'],duration,noise_shift=noise_shift)
                    amp.append(a);meta.append(dict(event_id=eid,station=rec['station'],sensor=rec['sensor'],window_s=duration,
                     epicentral_km=rec['epicentral_km'],distance_m=rec['distance'],azimuth=rec['azimuth'],
                     response_source=rec['response_source'],response_epoch=rec['response_epoch'],
                     master_postissue=rec['master_postissue'],response_warnings=rec['response_warnings'],
                     response_gain_ratio_min=rec['response_gain_ratio_min'],response_gain_ratio_max=rec['response_gain_ratio_max'],
                     noise_shift_s=noise_shift,status='ready',**m))
                except Exception as e:
                    logs.append(dict(event_id=eid,station=rec['station'],window_s=duration,status='rejected',reason=str(e)[:200]));fail(str(e)[:100])
            if not amp:continue
            arr=np.array(amp);tag=f'T{duration:g}'
            # Preserve alternatives for controls; every fit uses one record per site.
            arrays[tag+'_allrecords']=arr
            arrays[tag+'_allrecords_stations']=np.array([r['station'] for r in meta])
            arrays[tag+'_allrecords_sensor']=np.array([r['sensor'] for r in meta])
            primary=[]
            for sensor_set,sensors in [('ALL',['HH','EL','HG']),('HH_EL',['HH','EL']),('HG',['HG']),('HH',['HH'])]:
                chosen={}
                for i in sorted(range(len(meta)),key=lambda i:({'HH':0,'EL':1,'HG':2}[meta[i]['sensor']],meta[i]['station'])):
                    m=meta[i]
                    if m['sensor'] not in sensors or np.isfinite(arr[i,0]).sum()<20:continue
                    chosen.setdefault(m['station'],i)
                ids=list(chosen.values())
                if sensor_set=='ALL':primary=ids
                if not ids:continue
                rr,cc=fit_stack(f,arr[ids],depth_km=case['z_central'])
                for fit in rr:fits.append(dict(event_id=eid,window_s=duration,sensor_set=sensor_set,
                   master_postissue_stations=sum(meta[i]['master_postissue'] for i in ids),**fit))
                for curve in cc:
                    key=tag+'_'+sensor_set+'_'+curve['method'].replace(' ','_')+'_'+curve['shape']
                    for field in ['observed','predicted','good','count']:curve_data[key+'_'+field]=curve[field]
            if primary:
                arrays[tag]=arr[primary]
                arrays[tag+'_stations']=np.array([meta[i]['station'] for i in primary])
                arrays[tag+'_sensor']=np.array([meta[i]['sensor'] for i in primary])
            checks += [dict(m,status='ready' if i in primary else 'alternative') for i,m in enumerate(meta)]
        if arrays:np.savez_compressed(HERE/'spectra'/(eid+'.npz'),frequency_Hz=f,**arrays,**curve_data)
        pd.DataFrame(logs+checks).to_csv(HERE/'station_logs'/(eid+'.csv'),index=False)
        result=dict(event_id=eid,config_sha256=CONFIG_SHA,status='processed' if fits else 'no_usable_fit',
          raw_usable_triplets=len(records),fits=fits,rejections=errors,source_sha256=hashes,
          neighbors=case.get('neighbors',[]),seconds=time.monotonic()-start)
    except Exception as e:
        result=dict(event_id=eid,config_sha256=CONFIG_SHA,status='error',error=type(e).__name__+':'+str(e),fits=[])
    output.write_text(json.dumps(result,indent=2)+'\n')
    return dict(event_id=eid,status=result['status'],fits=len(result['fits']),seconds=result.get('seconds',0))

def main():
    global CONFIG_SHA
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--events',nargs='*');ap.add_argument('--limit',type=int)
    args=ap.parse_args();config=HERE/'inputs/study_config.json'
    CONFIG_SHA=config_signature()
    cohort=pd.read_csv(HERE/'inputs/cohort.csv',dtype={'event_id':str})
    target=json.loads((HERE.parent/'sangju_20261117/results/inputs/event.json').read_text())
    target.update(kma_lat=target['lat'],kma_lon=target['lon'],z_central=19.76,
      waveform_dir=str(IP.target_counts()),picks_csv=str(HERE.parent/'sangju_20261117/results/inputs/target_picks.csv'))
    cohort=pd.concat([cohort,pd.DataFrame([target])],ignore_index=True)
    if args.events:cohort=cohort[cohort.event_id.isin(args.events)]
    if args.limit:cohort=cohort.head(args.limit)
    cases=cohort.to_dict('records')
    # Neighbors come from the full magnitude catalog, including events whose depths are poor.
    catalog=pd.read_csv(HERE.parent/'data/necis/catalog/necis_events_domestic_1978_2026.csv')
    catalog=catalog[catalog.magnitude>=1.5].copy()
    catalog['seconds']=pd.to_datetime(catalog.origin_utc,utc=True).astype('int64')/1e9
    for case in cases:
        t=float(UTCDateTime(case['origin_utc']));near=catalog[abs(catalog.seconds-t)<=180]
        neighbors=[]
        for o in near.to_dict('records'):
            delta=float(o['seconds']-t)
            if abs(delta)<.5:continue
            km=gps2dist_azimuth(case['kma_lat'],case['kma_lon'],o['latitude'],o['longitude'])[0]/1000
            if km<=50:neighbors.append(dict(dt_s=delta,lat=o['latitude'],lon=o['longitude'],ml=o['magnitude']))
        case['neighbors']=neighbors
    IP.load()
    print('Response channel keys',len(IP.INV),'warming spectral tapers',flush=True)
    for n in [1011,2011]:mt(np.sin(np.arange(n)*.13)+.01*np.cos(np.arange(n)*.017),.01)
    print('Starting',len(cases),'events with',args.workers,'processes',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(run_event,c):c['event_id'] for c in cases}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            try:r=future.result()
            except Exception as e:r=dict(event_id=futures[future],status='worker_error',error=str(e))
            print(i,len(cases),json.dumps(r),flush=True)
    rows=[];status=[]
    for path in sorted((HERE/'events').glob('*.json')):
        item=json.loads(path.read_text());rows+=item.get('fits',[])
        status.append({k:v for k,v in item.items() if k not in ['fits','source_sha256','neighbors']})
    pd.DataFrame(rows).to_csv(HERE/'tables/catalog_spectral_fits.csv',index=False)
    pd.DataFrame(status).to_csv(HERE/'tables/catalog_processing_status.csv',index=False)

if __name__=='__main__':main()
