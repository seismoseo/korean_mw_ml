"""Reproduce saved spectra and test edge leakage on representative records.

Diagnostic controls do not replace adopted Mw or the response-removal recipe.
"""
from pathlib import Path
import json,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.signal.windows import tukey
from obspy import read,UTCDateTime
from multitaper import MTSpec
import instrument_processing as IP
from spectra import station_spectrum,fit_stack,mt
R=IP.ROOT
IP.load()
cases=pd.read_csv(R/'inputs/cohort.csv',dtype={'event_id':str}).set_index('event_id').to_dict('index')
target=json.loads((R.parent/'sangju_20261117/results/inputs/event.json').read_text())
target.update(waveform_dir=str(R/'inputs/Sangju_counts'),picks_csv=str(R.parent/'sangju_20261117/results/inputs/target_picks.csv'))
cases['20261117141017']=target
metrics=[];examples={};meta=[];controls=[]
for eid in ['20261117141017','20170105203113','20160829153946','20160502182909']:
    c=cases[eid];origin=UTCDateTime(c['origin_utc']);b=np.load(R/'spectra'/(eid+'.npz'))
    f=b['frequency_Hz'];arr=b['T10.1'];names=b['T10.1_stations'];sensors=b['T10.1_sensor']
    logs=pd.read_csv(R/'station_logs'/(eid+'.csv'))
    logs=logs[(logs.status=='ready')&(logs.window_s==10.1)].set_index('station')
    pp=pd.read_csv(c['picks_csv']);pp=pp[pp.Probability>=.3].sort_values('Probability',ascending=False)
    phase={}
    for p in pp.to_dict('records'):phase.setdefault((p['Network'],p['Station'],p['Phase']),UTCDateTime(p['Time']))
    # Examine the full problematic event, plus representative useful records.
    ids=list(range(len(names))) if eid=='20160829153946' else list(np.linspace(0,len(names)-1,5,dtype=int))
    if eid=='20261117141017':ids=[list(names).index('KS.SAJB')]
    for i in ids:
        name=str(names[i]);sensor=str(sensors[i]);net,sta=name.split('.');row=logs.loc[name]
        traces=[]
        for comp in 'ZNE':
            path=Path(c['waveform_dir'])/f'{eid}.{net}.{sta}.{sensor}{comp}.sac'
            raw=read(path)[0];label,ch,units=IP.response(raw,origin)
            w,_=IP.displacement(raw,ch);traces.append(w)
        p,s=phase[net,sta,'P'],phase[net,sta,'S']
        grid,a,m=station_spectrum(traces,p,s,row.distance_m,float(s-origin),noise_shift=row.noise_shift_s)
        valid=np.isfinite(a)&np.isfinite(arr[i]);error=float(np.max(abs(a[valid]-arr[i][valid])))
        assert error<1e-6,(eid,name,error)
        x=np.array([tr.slice(s-.1,s+10).data for tr in traces]);x=x-x.mean(axis=1,keepdims=True)
        dt=float(traces[0].stats.delta);T=x.shape[1]*dt
        # Controls intentionally change the window operator; none is adopted.
        bridge=x-(x[:,-1]-x[:,0])[:,None]*np.linspace(0,1,x.shape[1])[None,:]
        bridge-=bridge.mean(axis=1,keepdims=True)
        linear=detrend(x,axis=1,type='linear');cosine=x*tukey(x.shape[1],.1)[None,:]
        raw_f=np.fft.rfftfreq(2*x.shape[1]+1,dt)
        spectra={}
        for label,y in [('Fourier',x),('Bridge Fourier',bridge),('Linear Fourier',linear),('Cosine Fourier',cosine)]:
            z=np.linalg.norm(np.fft.rfft(y,n=2*y.shape[1]+1,axis=1)*dt,axis=0)
            spectra[label]=10**np.interp(np.log10(f),np.log10(raw_f[1:]),np.log10(np.maximum(z[1:],1e-100)))
        mt_power=[];scales=[]
        for xx in x:
            model=MTSpec(xx,4,7,dt);ff,power=model.rspec();mt_power.append(power[:,0])
            # Undo only the final variance rescaling to report its actual factor.
            from multitaper.utils import adaptspec
            unscaled,_,_=adaptspec(model.yk,model.sk,model.lamb,model.iadapt)
            factor=float(model.xvar/(np.sum(unscaled)*model.df));scales.append(factor/dt)
        power=np.sum(mt_power,axis=0)
        spectra['Multitaper']=10**np.interp(np.log10(f),np.log10(ff[1:,0]),np.log10(np.sqrt(T*power[1:]/2)))
        band=(f>=2)&(f<=10)&np.isfinite(a[0]);ratio={}
        for label,z in spectra.items():ratio[label]=float(np.median(z[band]/spectra['Fourier'][band])) if band.any() else np.nan
        jump=float(np.linalg.norm(x[:,-1]-x[:,0]));rms=float(np.sqrt(np.mean(np.sum(x*x,axis=0))))
        metrics.append(dict(event_id=eid,station=name,sensor=sensor,reproduction_error_log10=error,
          boundary_jump_m=jump,signal_rms_m=rms,boundary_to_RMS=jump/rms,
          MT_Fourier_ratio_2_10=ratio['Multitaper'],bridge_Fourier_ratio_2_10=ratio['Bridge Fourier'],
          linear_Fourier_ratio_2_10=ratio['Linear Fourier'],cosine_Fourier_ratio_2_10=ratio['Cosine Fourier'],
          MT_variance_rescaling_relative_min=min(scales),MT_variance_rescaling_relative_max=max(scales)))
        key=f'{eid}_{sta}';examples[key+'_time']=np.arange(x.shape[1])*dt-.1
        examples[key+'_displacement']=x;examples[key+'_frequency']=f
        for label,z in spectra.items():examples[key+'_'+label.replace(' ','_')]=z
        meta.append(dict(key=key,event_id=eid,station=name,sensor=sensor))
    print('Waveform audit',eid,'stations',len(ids),flush=True)
pd.DataFrame(metrics).to_csv(R/'tables/estimator_waveform_controls.csv',index=False)
np.savez_compressed(R/'spectra/estimator_waveform_controls.npz',**examples)
(R/'tables/estimator_waveform_examples.json').write_text(json.dumps(meta,indent=2)+'\n')
print(pd.DataFrame(metrics).groupby('event_id')[['MT_Fourier_ratio_2_10','bridge_Fourier_ratio_2_10','cosine_Fourier_ratio_2_10','boundary_to_RMS']].median().to_string())
