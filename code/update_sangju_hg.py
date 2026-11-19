"""Held-out Sangju controls and waveform arrays using the national HG pipeline."""
from pathlib import Path
import hashlib,json,sys,shutil,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from obspy import read,UTCDateTime
import run_catalog as C
import instrument_processing as IP
import local_magnitude as L
from spectra import fit_stack
from source_velocity import normalize_fit,log_moment_coefficient,CONFIG,CONFIG_SHA256
R=IP.ROOT;eid='20261117141017'
IP.load()
case=next(c for c in L.cases() if c['event_id']==eid)
case['radius_km']=110.
stage=Path('/tmp/sangju_hg_r110')
for sub in ['events','spectra','station_logs']:(stage/sub).mkdir(parents=True,exist_ok=True)
C.HERE=stage;C.CONFIG_SHA=C.config_signature()
existing=R/'tables/Sangju2026_spectral_fits.csv'
cached110=None
if existing.exists() and (R/'spectra/Sangju2026_r110.npz').exists():
    cached110=pd.read_csv(existing).query('radius_km==110').to_dict('records')
else:
    print(C.run_event(case),flush=True)
    shutil.copyfile(stage/'spectra'/(eid+'.npz'),R/'spectra/Sangju2026_r110.npz')
    shutil.copyfile(stage/'station_logs'/(eid+'.csv'),R/'tables/Sangju2026_stations_r110.csv')
rows=[];curves=[]
for radius,path in [(100,R/'events'/(eid+'.json')),(110,stage/'events'/(eid+'.json'))]:
    obj={'fits':cached110} if radius==110 and cached110 is not None else json.loads(path.read_text())
    for fit in obj['fits']:rows.append({**normalize_fit(fit,19.76),'radius_km':radius,'depth_km':19.76})
    b=np.load(R/'spectra'/((eid+'.npz') if radius==100 else 'Sangju2026_r110.npz'))
    f=b['frequency_Hz']
    for duration in [10.1,20.1]:
        tag=f'T{duration:g}'
        for sensor_set in ['ALL','HH_EL','HG','HH']:
            for method in ['Multitaper','Direct Fourier']:
                for shape in ['Boatwright','Brune']:
                    key=tag+'_'+sensor_set+'_'+method.replace(' ','_')+'_'+shape
                    if key+'_observed' not in b:continue
                    for j,hz in enumerate(f):curves.append(dict(radius_km=radius,window_s=duration,sensor_set=sensor_set,
                      method=method,shape=shape,frequency_Hz=hz,observed=b[key+'_observed'][j],
                      predicted=b[key+'_predicted'][j],usable=bool(b[key+'_good'][j]),count=int(b[key+'_count'][j])))
pd.DataFrame(rows).to_csv(R/'tables/Sangju2026_spectral_fits.csv',index=False)
pd.DataFrame(curves).to_csv(R/'tables/Sangju2026_spectral_curves.csv',index=False)
primary=next(r for r in rows if r['radius_km']==100 and r['window_s']==10.1 and r['sensor_set']=='ALL' and r['method']=='Multitaper' and r['shape']=='Boatwright')
ml=json.loads((R/'ml_events'/(eid+'.json')).read_text())
summary=dict(event_id=eid,origin_kst='2026-11-17 23:10:17',reported_ML=3.6,
 depth_km=19.76,depth_interval_km=[17.68,21.31],depth_grade='G3',Mw_constant=9.05,
 source_velocity_config_sha256=CONFIG_SHA256,source_velocity_scope=CONFIG['scope'],
 preferred=primary,local_magnitude=ml['summary'],
 uncertainty='Station-resampling intervals are conditional and exclude absolute path, source, site and response-history uncertainty.',
 primary_sensors=['HH','EL','HG'],aperture_km=100,
 target_exclusion='KS.BURB, both families: unresolved co-located horizontal amplitude discrepancy')
b=np.load(R/'spectra'/(eid+'.npz'));f=b['frequency_Hz'];arr=b['T10.1'][:,0]
good=np.isfinite(arr).sum(axis=0)>=max(3,.3*len(arr))
phi=-.5*np.log10(1+(f/primary['fc_Hz'])**(2*primary['falloff']))
rng=np.random.default_rng(20261117);boot=[]
for i in range(1000):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        curve=np.nanmean(arr[rng.integers(len(arr),size=len(arr))],axis=0)
    omega=float(np.nanmean((curve-phi)[good]))
    boot.append((omega+log_moment_coefficient(19.76)-9.05)/1.5)
summary['Mw_station_resampling_interval']=np.quantile(boot,[.025,.975]).tolist()
shift=primary['Mw_source_Vs_Kim2011']-primary['Mw']
summary['Mw_station_resampling_interval_Kim2011']=[x+shift for x in summary['Mw_station_resampling_interval']]
station_log=pd.read_csv(R/'station_logs'/(eid+'.csv'))
epic=[]
for name,sensor in zip(b['T10.1_stations'].astype(str),b['T10.1_sensor'].astype(str)):
    match=station_log[(station_log.station==name)&(station_log.sensor==sensor)&station_log.epicentral_km.notna()]
    epic.append(float(match.iloc[0].epicentral_km))
epic=np.asarray(epic);rcentral=np.hypot(epic,19.76)
endpoint_magnitudes={model:{'central':primary['Mw_source_Vs_'+model]} for model in CONFIG['models']}
for endpoint,depth in zip(['lower','upper'],summary['depth_interval_km']):
    adjust=np.log10(np.hypot(epic,depth)/rcentral)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        curve=np.nanmean(arr+adjust[:,None],axis=0)
    omega=float(np.nanmean((curve-phi)[good]))
    for model in CONFIG['models']:
        endpoint_magnitudes[model][endpoint]=(omega+log_moment_coefficient(depth,model)-9.05)/1.5
summary['depth_endpoint_magnitudes']=endpoint_magnitudes
summary['depth_endpoint_scope']='Frozen project depth envelope, distance and source Vs(z) only; fixed spectral shape and observed travel-time attenuation. Not a calibrated confidence interval.'
v=np.array([s['value'] for s in ml['selected']]);rng=np.random.default_rng(20261117)
summary['ML_station_resampling_interval']=np.quantile(np.median(rng.choice(v,(2000,len(v)),replace=True),axis=1),[.025,.975]).tolist()
(R/'tables/Sangju2026_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if (R/'spectra/Sangju2026_WA_examples.npz').exists() and (R/'tables/Sangju2026_WA_examples.csv').exists():
    print(json.dumps(summary,indent=2),flush=True)
    sys.exit(0)
# A few clear Wood--Anderson traces for direct review of added HG stations.
chosen=sorted(ml['selected'],key=lambda x:x['distance_km'])
show=[next(s for s in chosen if s['station']=='SAJB')]
show+= [s for s in chosen if s['sensor']=='HG'][:3]
show+= [s for s in chosen if s['sensor']=='HG' and s['distance_km']>65][:2]
origin=UTCDateTime(case['origin_utc']);arrays={};metadata=[]
for i,s in enumerate(show):
    c=next(c for c in ml['components'] if c['station']==s['station'] and c['sensor']==s['sensor'] and c['component']=='N' and c['status']=='processed')
    raw=read(c['file'])[0];label,ch,units=IP.response(raw,origin);w,_=L.wa_trace(raw,ch)
    cut=w.slice(origin+c['noise_start_s']-1,origin+c['signal_end_s']+3)
    arrays[f't{i}']=cut.times(reftime=origin);arrays[f'y{i}']=cut.data
    metadata.append(dict(index=i,station=s['station'],sensor=s['sensor'],distance_km=s['distance_km'],
      station_ML=s['value'],signal_start=c['signal_start_s'],signal_end=c['signal_end_s'],
      noise_start=c['noise_start_s'],noise_end=c['noise_end_s'],snr=c['snr']))
np.savez_compressed(R/'spectra/Sangju2026_WA_examples.npz',**arrays)
pd.DataFrame(metadata).to_csv(R/'tables/Sangju2026_WA_examples.csv',index=False)
print(json.dumps(summary,indent=2),flush=True)
