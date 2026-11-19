"""Compare window and station-resampling stability before Mw-A selection.

Both estimators use the same station/sensor identities AND frequency support
in both windows. Existing SNR masks were obtained with multitaper, a limitation
retained explicitly; this is not an independent-noise or absolute-bias test.
"""
from pathlib import Path
import concurrent.futures,json,multiprocessing,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from spectra import fit
from source_velocity import metadata
R=Path(__file__).resolve().parents[1]

def event(case):
    eid=case['event_id'];path=R/'spectra'/(eid+'.npz')
    row=dict(event_id=eid,held_out=eid=='20261117141017',Mw_quality=case.get('Mw_quality','held out'),
      ML_KMA=case.get('ml',3.6),status='no_common_fit')
    row.update(metadata(case['z_central']))
    if not path.exists():return row
    b=np.load(path)
    if not all(t in b for t in ['T10.1','T20.1']):return row
    f=b['frequency_Hz'];a=b['T10.1'];aa=b['T20.1']
    keys=list(zip(b['T10.1_stations'].astype(str),b['T10.1_sensor'].astype(str)))
    keys2=list(zip(b['T20.1_stations'].astype(str),b['T20.1_sensor'].astype(str)))
    common=sorted(set(k for i,k in enumerate(keys) if np.isfinite(a[i,0]).sum()>=20)&
      set(k for i,k in enumerate(keys2) if np.isfinite(aa[i,0]).sum()>=20))
    row['n_common_sites']=len(common)
    if len(common)<8:return row
    x=a[[keys.index(k) for k in common]].copy();y=aa[[keys2.index(k) for k in common]].copy()
    mask=np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1)
    x=np.where(mask[:,None,:],x,np.nan);y=np.where(mask[:,None,:],y,np.nan)
    good=mask.sum(axis=0)>=max(3,.3*len(common))
    row['n_common_frequencies']=int(good.sum())
    allresolved=True;finite=True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        means=[np.nanmean(q,axis=0) for q in [x,y]]
    for i,name in enumerate(['MT','FFT']):
        values=[]
        for j,tag in enumerate(['10','20']):
            result,_=fit(f,means[j][i],good,depth_km=case['z_central'])
            value=result.get('Mw',np.nan);values.append(value)
            row[name+'_Mw_'+tag]=value
            row[name+'_Mw_'+tag+'_Kim2011']=result.get('Mw_source_Vs_Kim2011',np.nan)
            row[name+'_plateau_'+tag]=result.get('plateau_observed',False)
            row[name+'_RMS_'+tag]=result.get('rms_log10',np.nan)
            allresolved &= bool(result.get('plateau_observed',False))
            finite &= bool(np.isfinite(value))
        row[name+'_window_change']=values[1]-values[0]
        low=good&(f>=.7)&(f<=1.2)
        row[name+'_low_band_amplitude_change']=float((2/3)*np.median((means[1][i]-means[0][i])[low])) if low.sum()>=10 else np.nan
    row['both_methods_plateau']=allresolved
    if not finite:return row
    row['status']='fit'
    # Equal random site draws for both methods; source shape and usable band fixed.
    rng=np.random.default_rng(int(eid[-8:]))
    counts=rng.multinomial(len(common),np.ones(len(common))/len(common),size=250)
    denom=counts@mask.astype(float)
    for i,name in enumerate(['MT','FFT']):
        num=counts@np.nan_to_num(x[:,i],nan=0.)
        with np.errstate(divide='ignore',invalid='ignore'):curves=num/denom
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);level=np.nanmean(curves[:,good],axis=1)
        row[name+'_station_bootstrap_sigma']=float((2/3)*np.nanstd(level,ddof=1))
    return row

def main():
    cases=pd.read_csv(R/'tables/Mw_ML_catalog.csv',dtype={'event_id':str}).to_dict('records')
    cases.append(dict(event_id='20261117141017',ml=3.6,z_central=19.76))
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('fork')) as pool:
        for i,row in enumerate(pool.map(event,cases),1):
            rows.append(row)
            if i%100==0:print('Stability',i,'/',len(cases),flush=True)
    d=pd.DataFrame(rows);d.to_csv(R/'tables/estimator_stability_events.csv',index=False)
    selections=[('Common support, all finite',d[(d.status=='fit')&~d.held_out]),
      ('Common support, both plateaus',d[(d.status=='fit')&d.both_methods_plateau.fillna(False)&~d.held_out]),
      ('Existing Mw A, conditional',d[(d.status=='fit')&(d.Mw_quality=='A')&~d.held_out])]
    out=[]
    for name,q in selections:
        for method in ['MT','FFT']:
            v=q[method+'_window_change'].dropna();sig=q[method+'_station_bootstrap_sigma'].dropna()
            out.append(dict(selection=name,method=method,n=len(v),median_abs_window_change=float(v.abs().median()),
              q90_abs_window_change=float(v.abs().quantile(.9)),fraction_abs_gt_0p1=float((v.abs()>.1).mean()),
              median_station_bootstrap_sigma=float(sig.median())))
    pd.DataFrame(out).to_csv(R/'tables/estimator_stability_summary.csv',index=False)
    summary=dict(n_events=len(d),n_finite=int((d.status=='fit').sum()),
      method='Same station/sensor identities and common station-frequency support in 10.1 and 20.1 s',
      bootstrap='250 equal station resamples per event; fixed source shape and frequency support',
      selection='First two selections do not use Mw-A agreement or window-stability gates',
      limitations=['Existing SNR masks were derived with multitaper.',
       'Window/resampling stability is not an absolute moment-bias test.',
       'Raw time-domain Hong estimates were not recomputed on these records.'],results=out)
    (R/'tables/estimator_stability_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(pd.DataFrame(out).round(5).to_string(index=False),flush=True)

if __name__=='__main__':main()
