"""Spectral moments in physical Fourier-amplitude units; no ML calibration."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/korean_mw_ml_numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/korean_mw_ml_matplotlib')
import numpy as np
from multitaper import MTSpec
from source_velocity import C,RHO,RAD,FS,normalize_fit

TAPERS={};MODELS={}
Q0=229.2;ETA=.73

def mt(x,dt):
    key=(len(x),float(dt))
    if key not in TAPERS:
        m=MTSpec(x,4,7,dt);TAPERS[key]=(m.vn,m.lamb)
    else:
        vn,ev=TAPERS[key];m=MTSpec(x,4,7,dt,vn=vn,lamb=ev)
    f,p=m.rspec()
    return f[:,0],p[:,0]

def station_spectrum(traces,p_abs,s_abs,distance_m,s_time_s,window=10.1,low=.5,high=20.,noise_shift=0.):
    dt=float(traces[0].stats.delta)
    if not all(abs(t.stats.delta-dt)<1e-8 for t in traces):raise ValueError('component_sample_rates')
    sig=[];noise=[]
    for tr in traces:
        sw=tr.slice(s_abs-.1,s_abs+window-.1).data.astype(float)
        end=p_abs+noise_shift;nw=tr.slice(end-window,end).data.astype(float)
        expected=round(window/dt)+1
        if len(sw)!=expected or len(nw)!=expected:raise ValueError('incomplete_window')
        if not np.isfinite(sw).all() or not np.isfinite(nw).all():raise ValueError('nonfinite_waveform')
        sig.append(sw);noise.append(nw)
    sig=np.array(sig);noise=np.array(noise)
    ps=[];pn=[]
    for x,y in zip(sig,noise):
        f,a=mt(x,dt);fn,b=mt(y,dt);ps.append(a);pn.append(b)
    ps=np.sum(ps,axis=0);pn=np.sum(pn,axis=0)
    effective_T=sig.shape[1]*dt
    grid=np.geomspace(low,high,200);lf=np.log10(grid)
    amp=np.sqrt(np.maximum(effective_T*ps/2,1e-200))
    lmt=np.interp(lf,np.log10(f[1:]),np.log10(amp[1:]))
    snr=10**np.interp(lf,np.log10(f[1:]),.5*np.log10(np.maximum(ps[1:],1e-200)/np.maximum(pn[1:],1e-200)))
    y=sig-sig.mean(axis=1,keepdims=True)
    ff=np.fft.rfftfreq(2*y.shape[1]+1,dt)
    a=np.linalg.norm(np.fft.rfft(y,n=2*y.shape[1]+1,axis=1)*dt,axis=0)
    lfft=np.interp(lf,np.log10(ff[1:]),np.log10(np.maximum(a[1:],1e-100)))
    correction=np.log10(distance_m)+np.pi*grid*s_time_s/(Q0*grid**ETA)/np.log(10)
    arr=np.array([lmt,lfft])+correction
    good=(snr>=3)&np.isfinite(arr).all(axis=0)
    arr[:,~good]=np.nan
    return grid,arr,dict(SNR_median=float(np.median(snr)),valid_frequencies=int(good.sum()),
                          effective_T_s=effective_T,dt_s=dt)

def bank(f,shape):
    key=(float(f[0]),float(f[-1]),len(f),shape)
    if key not in MODELS:
        corners=np.geomspace(.2,f[-1],400)
        exponents=np.linspace(1.5,4.,26) if shape=='Boatwright' else np.array([2.])
        fc=np.repeat(corners,len(exponents));n=np.tile(exponents,len(corners))
        ratio=f[None,:]/fc[:,None]
        phi=-.5*np.log10(1+ratio**(2*n[:,None])) if shape=='Boatwright' else -np.log10(1+ratio**2)
        MODELS[key]=(fc,n,phi,len(exponents))
    return MODELS[key]

def fit(f,y,good,shape='Boatwright',*,depth_km):
    good=good&np.isfinite(y)
    if good.sum()<20 or np.ptp(np.log10(f[good]))<1:
        return dict(status='insufficient_band',Mw=np.nan),np.full(len(f),np.nan)
    fc,n,phi,nn=bank(f,shape);g=good.astype(float);ng=g.sum();yy=np.where(good,y,0.)
    om=(yy.sum()-phi@g)/ng
    var=(np.sum(yy**2)+(phi**2)@g-2*phi@yy)/ng-om**2
    j=int(np.argmin(var));omega=float(10**om[j])
    lower=j//nn==0;upper=j//nn==399
    plateau=f[good].min()<=fc[j]/2
    row=dict(status='ok' if plateau else 'plateau_extrapolated',omega_r_m2s=omega,
      fc_Hz=float(fc[j]),falloff=float(n[j]),corner_lower_rail=bool(lower),corner_upper_rail=bool(upper),
      plateau_observed=bool(plateau),rms_log10=float(np.sqrt(max(var[j],0))),
      n_frequency=int(ng),band_low_Hz=float(f[good].min()),band_high_Hz=float(f[good].max()))
    return normalize_fit(row,depth_km),om[j]+phi[j]

def fit_stack(f,arr,min_stations=5,*,depth_km):
    nsta=len(arr);count=np.isfinite(arr[:,0,:]).sum(axis=0)
    good=count>=max(3,.3*nsta)
    rows=[];curves=[]
    if nsta<min_stations:return rows,curves
    import warnings
    for method_index,method in enumerate(['Multitaper','Direct Fourier']):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            stack=np.nanmean(arr[:,method_index,:],axis=0)
        for shape in ['Boatwright','Brune']:
            row,pred=fit(f,stack,good,shape,depth_km=depth_km)
            row.update(method=method,shape=shape,n_stations=nsta)
            rows.append(row)
            curves.append(dict(method=method,shape=shape,observed=stack,predicted=pred,good=good,count=count))
    return rows,curves
