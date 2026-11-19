"""Physical recovery controls and co-located spectra for the adopted processing."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from obspy import Trace,read,read_inventory,UTCDateTime
from scipy.signal import detrend
import instrument_processing as IP
from spectra import station_spectrum,mt
import local_magnitude as L
R=IP.ROOT
pairs=json.loads((R/'hh_hg_comparison/inputs/pairs.json').read_text())
pair=next(p for p in pairs if p['event_id']=='20261117141017' and p['station']=='SAJB')
n=24000;dt=.01;t=np.arange(n)*dt
# A known displacement pulse, not a fitted source-moment calibration.
ground=1.e-7*np.exp(-((t-115)/2)**2)*sum(np.sin(2*np.pi*f*(t-115)) for f in [.7,1.,2.,5.,10.])
ff=np.fft.rfftfreq(n,dt);physical=[]
for sensor in ['HH','HG']:
    ch=read_inventory(pair['responses'][sensor+'Z'])[0][0][0]
    h=ch.response.get_evalresp_response_for_frequencies(ff[1:],output='DISP',hide_sensitivity_mismatch_warning=True)
    response=np.r_[0j,h]
    counts=np.fft.irfft(np.fft.rfft(ground)*response,n=n)
    tr=Trace(counts,header=dict(delta=dt,network='KS',station='SAJB',channel=sensor+'Z',starttime=UTCDateTime(0)))
    recovered,_=IP.displacement(tr,ch)
    use=(t>105)&(t<125)
    err=np.linalg.norm(recovered.data[use]-ground[use])/np.linalg.norm(ground[use])
    w,_=L.wa_trace(tr,ch)
    ideal=Trace(ground.copy(),header=dict(delta=dt)).filter('bandpass',**L.RECIPE.BANDPASS)
    ideal.simulate(paz_simulate=L.RECIPE.WOOD_ANDERSON_PAZ,taper=False,zero_mean=False)
    ratio=np.max(abs(w.data[use]))/(1000*np.max(abs(ideal.data[use])))
    physical.append(dict(sensor=sensor,relative_displacement_error=float(err),WA_peak_recovery=float(ratio)))
    assert err<.02 and abs(ratio-1)<.02,(sensor,err,ratio)
rows=[]
for p in pairs:
    if p['event_id']!='20261117141017':continue
    arr={}
    for sensor in ['HH','HG']:
        traces=[]
        for c in 'ZNE':
            ch=read_inventory(p['responses'][sensor+c])[0][0][0]
            w,_=IP.displacement(read(p['traces'][sensor+c])[0],ch)
            traces.append(w)
        f,a,m=station_spectrum(traces,UTCDateTime(p['P_utc']),UTCDateTime(p['S_utc']),
          np.hypot(p['epicentral_km'],p['depth_km'])*1000,
          float(UTCDateTime(p['S_utc'])-UTCDateTime(p['origin_utc'])),noise_shift=-1)
        arr[sensor]=a
    for low,high in [(.7,1.2),(2.,5.),(5.,10.)]:
        ix=(f>=low)&(f<=high)&np.isfinite(arr['HH'][0])&np.isfinite(arr['HG'][0])
        ratio=float(10**np.median(arr['HG'][0,ix]-arr['HH'][0,ix])) if ix.any() else np.nan
        rows.append(dict(station=p['station'],band_low_Hz=low,band_high_Hz=high,HG_HH_FAS=ratio,n_bins=int(ix.sum())))
rat=pd.DataFrame(rows)
check=rat[(rat.station!='BURB')&(rat.band_low_Hz==.7)]
assert len(check)==5 and check.HG_HH_FAS.between(.8,1.2).all(),check
rat.to_csv(R/'tables/HG_new_processing_pair_controls.csv',index=False)
result=dict(physical_recovery=physical,co_located_low_band=check.to_dict('records'),passed=True,
 limitations='Recovery checks response operators and units, not source, path or radiation calibration. Co-located discrepancies remain diagnostic; BURB is excluded from Sangju magnitudes.')
(R/'ml_validation/HG_processing_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
