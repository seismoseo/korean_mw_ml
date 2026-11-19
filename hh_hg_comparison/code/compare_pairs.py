"""Matched-window HH/HG spectra, noise, coherence and response-removal controls."""
from pathlib import Path
import hashlib,json,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from obspy import read,read_inventory,UTCDateTime,Trace
from scipy.signal.windows import dpss
from scipy.signal import detrend,correlate,correlation_lags
from scipy.fft import rfft,rfftfreq,next_fast_len
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[0]/'code'))
from local_magnitude import RECIPE

POLICY=dict(primary='Same original station/location, UTC windows, components and physical units; no fitted amplitude renormalization',
    spectral_quantity='One-sided multitaper acceleration PSD; displacement-equivalent ASD = sqrt(PSD)/(2*pi*f)^2',
    response='Remove full dated response to ACC; water_level=None; pre_filt [.005,.02,25,40] Hz at 100 sps; scale high corners if needed',
    windows={'short':'S-.1 to S+10.0 s; equal 10.1 s pre-P noise ending P-1',
             'long':'origin to origin+80 s; equal 80 s pre-P noise ending P-1'},
    multitaper='DPSS: short NW=3 K=5; long NW=2 K=3; unit-energy tapers, equal weights; next_fast_len FFT',
    signal_quality='Amplitude SNR=sqrt(Psignal/Pnoise); common-signal ratio uses noise-subtracted power where both SNR>=3',
    coherence='Component magnitude-squared coherence and its power-weighted mean; also retain phase-sensitive vector coherence; amplitude ratios use both SNR>=3 without requiring axis agreement',
    timing='Original UTC timestamps; Lanczos interpolation onto common sample grid before spectral windows',
    water_level_control='Output DISP, water_level 60 vs None, otherwise identical processing; this isolates regularization',
    ML='Same 4--2 km/s hypocentral window, geometric horizontal WA peaks; causal .5--10 Hz; no water level; prefilter [.1,.25,15,20]',
    bands=[[.05,.2],[.2,.5],[.5,1.],[.7,1.2],[1.,2.],[2.,5.],[5.,10.],[10.,20.]])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_trace(tr,resp,output='ACC',water=None,ml=False):
    w=tr.copy().detrend('demean').detrend('linear').taper(.01,max_length=5.)
    nyq=w.stats.sampling_rate/2
    pre=[.005,.02,min(25.,.6*nyq),min(40.,.85*nyq)]
    if ml:
        w.filter('bandpass',**RECIPE.BANDPASS);pre=[.1,.25,15.,20.]
    w.stats.response=resp
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w.remove_response(output=output,water_level=water,pre_filt=pre,
                          zero_mean=False,taper=False,hide_sensitivity_mismatch_warning=True)
    messages=sorted(set(str(c.message) for c in caught))
    if any('not known to ObsPy' in s for s in messages):raise ValueError('unresolved response units')
    if not np.isfinite(w.data).all():raise ValueError('nonfinite response-corrected data')
    if ml:
        w.simulate(paz_simulate=RECIPE.WOOD_ANDERSON_PAZ);w.data*=1000
    return w,messages


def window(tr,start,duration,sr):
    count=int(round(duration*sr))
    # Identical UTC sample grid, including fractional origin/pick seconds.
    w=tr.copy().interpolate(sr,method='lanczos',starttime=start,npts=count,a=30)
    if len(w.data)!=count or not np.isfinite(w.data).all():raise ValueError('invalid matched window')
    return np.asarray(w.data,float)


def transforms(x,sr,nw=3.,k=5):
    n=x.shape[-1];tapers,eigen=dpss(n,nw,Kmax=k,sym=False,norm=2,return_ratios=True)
    x=detrend(x,axis=-1,type='linear')
    nfft=next_fast_len(n)
    j=rfft(x[:,None,:]*tapers[None,:,:],n=nfft,axis=-1)
    f=rfftfreq(nfft,1/sr)
    factor=np.ones(len(f))*2/sr;factor[0]/=2
    if nfft%2==0:factor[-1]/=2
    power=np.mean(abs(j)**2,axis=1)*factor[None,:]
    return f,power,j,factor


def spectral_case(traces,pair,tag,duration,sr):
    origin=UTCDateTime(pair['origin_utc']);p=UTCDateTime(pair['P_utc']);s=UTCDateTime(pair['S_utc'])
    start=s-.1 if tag=='short' else origin
    noise_start=p-1-duration
    raw={};xforms={};result={}
    for sensor in ['HH','HG']:
        x=np.array([window(traces[sensor+c],start,duration,sr) for c in 'ZNE'])
        noise=np.array([window(traces[sensor+c],noise_start,duration,sr) for c in 'ZNE'])
        kw=dict(nw=2.,k=3) if tag=='long' else {}
        f,ps,j,fac=transforms(x,sr,**kw);_,pn,_,_=transforms(noise,sr,**kw)
        result[sensor+'_signal_PSD']=ps
        result[sensor+'_noise_PSD']=pn
        total=ps.sum(axis=0);n_total=pn.sum(axis=0)
        snr=np.sqrt(np.divide(total,n_total,out=np.full_like(total,np.nan),where=n_total>0))
        result[sensor+'_SNR']=snr
        disp_factor=np.zeros(len(f));disp_factor[1:]=1/(2*np.pi*f[1:])**2
        result[sensor+'_ASD']=np.sqrt(total)*disp_factor
        result[sensor+'_noise_ASD']=np.sqrt(n_total)*disp_factor
        raw[sensor]=x;xforms[sensor]=j
    cross=np.mean(xforms['HH']*np.conj(xforms['HG']),axis=1)*fac[None,:]
    hh=result['HH_signal_PSD'];hg=result['HG_signal_PSD']
    coh=np.divide(abs(cross)**2,hh*hg,out=np.full_like(hh,np.nan),where=(hh*hg)>0)
    totalcoh=np.divide(abs(cross.sum(axis=0))**2,hh.sum(axis=0)*hg.sum(axis=0),
                       out=np.full(len(f),np.nan),where=(hh.sum(axis=0)*hg.sum(axis=0))>0)
    assert np.nanmax(coh)<=1+1e-10 and np.nanmax(totalcoh)<=1+1e-10
    result['coherence_components']=np.clip(coh,0,1);result['coherence_vector']=np.clip(totalcoh,0,1)
    weighted=np.divide(np.sum(abs(cross)**2,axis=0),np.sum(hh*hg,axis=0),
                       out=np.full(len(f),np.nan),where=np.sum(hh*hg,axis=0)>0)
    result['coherence_mean']=np.clip(weighted,0,1)
    result['phase_degrees']=np.rad2deg(np.angle(cross))
    excess_hh=(hh-result['HH_noise_PSD']).sum(axis=0)
    excess_hg=(hg-result['HG_noise_PSD']).sum(axis=0)
    ratio=np.sqrt(np.divide(excess_hg,excess_hh,out=np.full_like(f,np.nan),where=(excess_hh>0)&(excess_hg>0)))
    both=(result['HH_SNR']>=3)&(result['HG_SNR']>=3)
    result['amplitude_ratio']=ratio;result['both_SNR3']=both;result['frequency_Hz']=f
    result['start_utc']=np.array(str(start));result['noise_start_utc']=np.array(str(noise_start));result['duration_s']=np.array(duration)
    if tag=='short':
        for sensor in ['HH','HG']:result[sensor+'_waveforms_ACC']=raw[sensor]
        result['time_from_S_s']=np.arange(raw['HH'].shape[-1])/sr-.1
    return result


def process(pair):
    ident=pair['event_id']+'_'+pair['station'];origin=UTCDateTime(pair['origin_utc'])
    traces={};native={};unlimited={};wa={};responses={};audit=[]
    for key,path in pair['traces'].items():
        tr=read(path)[0];inv=read_inventory(pair['responses'][key])
        resp=inv.get_response(tr.id,origin)
        assert (tr.stats.network,tr.stats.station,tr.stats.channel)==(pair['network'],pair['station'],key)
        sig=resp.instrument_sensitivity
        eval_gain=abs(resp.get_evalresp_response_for_frequencies([float(sig.frequency)],output='DEF',hide_sensitivity_mismatch_warning=True)[0])
        a,messages=response_trace(tr,resp)
        traces[key]=a
        audit.append(dict(event_id=pair['event_id'],station=pair['station'],channel=key,input_units=sig.input_units,
                          full_to_stated_gain=eval_gain/abs(sig.value),sampling_rate=tr.stats.sampling_rate,
                          record_start=str(tr.stats.starttime),record_end=str(tr.stats.endtime),warnings=' | '.join(messages)))
        w,_=response_trace(tr,resp,output='DISP',ml=True);wa[key]=w
        if key.startswith('HG'):
            w,_=response_trace(tr,resp,output='DISP',water=60.);native[key]=w
            w,_=response_trace(tr,resp,output='DISP',water=None);unlimited[key]=w
    sr=min(t.stats.sampling_rate for t in traces.values())
    output={};bands=[]
    for tag,duration in [('short',10.1),('long',80.)]:
        result=spectral_case(traces,pair,tag,duration,sr)
        f=result['frequency_Hz'];both=result['both_SNR3']
        for key,value in result.items():output[tag+'_'+key]=value
        s=UTCDateTime(pair['S_utc']);start=s-.1 if tag=='short' else origin
        for label,tt in [('WL60',native),('noWL',unlimited)]:
            x=np.array([window(tt['HG'+c],start,duration,sr) for c in 'ZNE'])
            _,power,_,_=transforms(x,sr,**(dict(nw=2.,k=3) if tag=='long' else {}))
            output[tag+'_HG_DISP_ASD_'+label]=np.sqrt(power.sum(axis=0))
        with np.errstate(invalid='ignore',divide='ignore'):
            output[tag+'_HG_WL60_to_noWL']=output[tag+'_HG_DISP_ASD_WL60']/output[tag+'_HG_DISP_ASD_noWL']
        for low,high in POLICY['bands']:
            if (low<.5)!=(tag=='long'):continue
            mask=(f>=low)&(f<=high)
            common=mask&both&np.isfinite(result['amplitude_ratio'])
            use=common&(result['coherence_mean']>=.8)
            ratios=result['amplitude_ratio'][common]
            bands.append(dict(event_id=pair['event_id'],name=pair['name'],ML=pair['ML'],station=pair['station'],
                epicentral_km=pair['epicentral_km'],window=tag,window_s=duration,band_low_Hz=low,band_high_Hz=high,
                n_bins=int(mask.sum()),both_SNR3_fraction=float(np.mean(both[mask])) if mask.any() else np.nan,
                HH_SNR_median=float(np.nanmedian(result['HH_SNR'][mask])) if mask.any() else np.nan,
                HG_SNR_median=float(np.nanmedian(result['HG_SNR'][mask])) if mask.any() else np.nan,
                coherence_median=float(np.nanmedian(result['coherence_mean'][common])) if common.any() else np.nan,
                vertical_coherence_median=float(np.nanmedian(result['coherence_components'][0,common])) if common.any() else np.nan,
                coherent_SNR3_fraction=float(use.sum()/mask.sum()) if mask.any() else np.nan,
                HG_HH_ratio_median=float(np.median(ratios)) if len(ratios) else np.nan,
                HG_HH_ratio_q16=float(np.quantile(ratios,.16)) if len(ratios) else np.nan,
                HG_HH_ratio_q84=float(np.quantile(ratios,.84)) if len(ratios) else np.nan,
                equivalent_delta_Mw=float(2/3*np.log10(np.median(ratios))) if len(ratios) else np.nan,
                HG_WL60_amplitude_ratio=float(np.nanmedian(output[tag+'_HG_WL60_to_noWL'][mask])) if mask.any() else np.nan))
    # Same physical ML measurement at the same station: attenuation cancels in
    # the HH/HG difference, so this is a useful separate amplitude check.
    rtime=np.hypot(pair['epicentral_km'],pair['depth_km']);ml={}
    for sensor in ['HH','HG']:
        peaks={};snr={}
        for c in 'NE':
            w=wa[sensor+c]
            sig=window(w,origin+rtime/4,rtime/4,sr)
            noise=window(w,UTCDateTime(pair['P_utc'])-6,5.,sr)
            peaks[c]=float(max(abs(sig)));den=float(max(abs(noise)))
            snr[c]=peaks[c]/den if den>0 else np.nan
        amp=np.sqrt(peaks['N']*peaks['E'])
        ml[sensor+'_WA_mm']=float(amp);ml[sensor+'_ML']=float(RECIPE.ml_horizontal(amp,pair['epicentral_km']))
        ml[sensor+'_WA_SNR']=min(snr.values())
    ml['HG_minus_HH_ML']=ml['HG_ML']-ml['HH_ML']
    path=ROOT/'spectra'/(ident+'.npz');np.savez_compressed(path,**output)
    return bands,audit,dict(event_id=pair['event_id'],name=pair['name'],ML=pair['ML'],station=pair['station'],epicentral_km=pair['epicentral_km'],**ml)


def controls():
    from obspy.core.inventory.response import Response
    response=Response.from_paz(zeros=[],poles=[],stage_gain=1e6,input_units='M/S**2',output_units='COUNTS')
    sr=100.;t=np.arange(40000)/sr;rows=[]
    for f in [.1,.5,1.,2.,5.]:
        A=1e-6
        tr=Trace((-A*(2*np.pi*f)**2*np.cos(2*np.pi*f*t)*1e6).astype(float));tr.stats.sampling_rate=sr
        amplitudes={}
        for name,water in [('noWL',None),('WL60',60.)]:
            w,_=response_trace(tr,response,output='DISP',water=water)
            use=(t>=100)&(t<300)
            X=np.column_stack([np.cos(2*np.pi*f*t[use]),np.sin(2*np.pi*f*t[use]),np.ones(use.sum()),t[use]-200])
            coef=np.linalg.lstsq(X,w.data[use],rcond=None)[0];amplitudes[name]=float(np.hypot(*coef[:2]))
        expected=min(1.,f*f/(.001*(sr/2)**2))
        assert abs(amplitudes['noWL']/A-1)<.005
        assert abs(amplitudes['WL60']/amplitudes['noWL']-expected)<.005
        rows.append(dict(frequency_Hz=f,true_displacement_m=A,recovered_no_waterlevel=amplitudes['noWL'],
                         recovered_WL60=amplitudes['WL60'],analytic_WL60_attenuation=expected))
    # Coherence and amplitude ratio must recover known gain without fitting it.
    rng=np.random.default_rng(4);x=rng.normal(size=(3,4096));f,p,j,fac=transforms(x,100.)
    _,q,k,_=transforms(2*x,100.)
    ratio=np.sqrt(q.sum(axis=0)/p.sum(axis=0));cross=np.mean(j*np.conj(k),axis=1)*fac[None,:]
    coherence=abs(cross.sum(axis=0))**2/(p.sum(axis=0)*q.sum(axis=0))
    assert np.allclose(ratio,2.) and np.allclose(coherence,1.)
    return dict(known_acceleration_controls=rows,known_gain_and_coherence='passed')


def main():
    for sub in ['spectra','tables']:(ROOT/sub).mkdir(exist_ok=True)
    pairs=json.loads((ROOT/'inputs/pairs.json').read_text());bands=[];audit=[];ml=[]
    for pair in pairs:
        b,a,m=process(pair);bands+=b;audit+=a;ml.append(m)
        print(pair['event_id'],pair['station'],'spectra compared',flush=True)
    pd.DataFrame(bands).to_csv(ROOT/'tables/spectral_comparison.csv',index=False)
    pd.DataFrame(audit).to_csv(ROOT/'tables/response_checks.csv',index=False)
    pd.DataFrame(ml).to_csv(ROOT/'tables/paired_ML.csv',index=False)
    check=controls();check['n_station_event_pairs']=len(pairs)
    (ROOT/'tables/processing_validation.json').write_text(json.dumps(check,indent=2)+'\n')
    provenance=dict(policy=POLICY,code_sha256=sha(__file__),pair_inputs_sha256=sha(ROOT/'inputs/pairs.json'),
                    source_hashes_sha256=sha(ROOT/'inputs/source_hashes.json'))
    (ROOT/'tables/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Scientific controls passed; pairs',len(pairs),flush=True)


if __name__=='__main__':main()
