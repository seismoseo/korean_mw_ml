"""Independent Wood--Anderson ML from immutable counts, with no KMA anchoring.

The response and amplitude operations follow the frozen Haenam/Buan engine.
Primary ML uses observed P, two horizontal components, SNR>=3, 10--100 km.
No empirical station/era corrections enter these measurements.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, multiprocessing, os, sys, time, warnings
sys.dont_write_bytecode = True
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from obspy import read, read_inventory, UTCDateTime
from obspy.geodetics import gps2dist_azimuth

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('/home/msseo/works/27.Haenam_sequence_renewal/magnitudes/local_ml')
sys.path.insert(0, str(UPSTREAM))
import buan_sheen2018_snapshot as RECIPE
import instrument_processing as IP
INV = {}
CONFIG_SHA = ''
POLICY = {
    'version': 'independent-ML-HG-2',
    'scale': 'Sheen, Kang & Rhie (2018), equations 5--6, station corrections zero',
    'amplitude': 'revised Wood-Anderson zero-to-peak mm; geometric mean of N/E peaks',
    'processing': 'demean, linear detrend, one 1-percent taper capped 5 s; regularized response to DISP, waterlevel=None, .1/.2/25/40 Hz prefilter; causal six-pole .5--10 Hz Butterworth; revised WA, no additional taper',
    'WA': RECIPE.WOOD_ANDERSON_PAZ.__repr__(),
    'timing': 'origin+[hypocentral_km/4, hypocentral_km/2]; attenuation uses epicentral km',
    'noise': 'observed P+[ -6, -1 ] seconds, peak absolute amplitude',
    'picking': 'observed P probability >=.3; no theoretical-P rescued stations',
    'primary': 'median horizontal station ML; both N/E peak SNR>=3; >=3 stations; HH preferred then EL then HG; one observation per site; 10<=R_epi<=100 km',
    'comparison_quality': 'at least 5 horizontal stations, robust station scatter <=.35',
    'sensitivities': 'SNR2, H-preferred/V-fallback, all valid sensors incl HG, BB only, strict metadata, 40--100 km, epicentral timing, trimmed mean',
    'units': 'HH/EL must declare M/S; HG must declare M/S**2; unknown response units quarantined',
    'response': 'dated merged response override, historical master otherwise; coordinate check <=2 km; exclude KS.KWJU.HH and KS.SEJF.EL',
    'source_selection': 'KMA ML>=2 defines parent sample only; ML amplitudes/zero point/station corrections are not fitted to KMA or Mw',
    'overlap': 'full catalog M>=1.5 within 180 s and 50 km; exclude overlapping signal or pre-P noise; no quiet-window search',
    'published_calibration': 'published epicentral attenuation; 100-km aperture is narrower than original 10--600 km calibration; no independently fitted station terms',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collision(interval, others):
    return any(interval[0] < b and interval[1] > a for a, b in others)


def load_responses():
    IP.load()
    return json.loads((ROOT/'inputs/response_sources.json').read_text())


def response(tr,when):
    return IP.response(tr,when)


def wa_trace(tr,ch):
    w,messages=IP.displacement(tr,ch)
    w.filter('bandpass',**RECIPE.BANDPASS)
    w.simulate(paz_simulate=RECIPE.WOOD_ANDERSON_PAZ,taper=False,zero_mean=False)
    w.data=np.asarray(w.data,float)*1000.
    if not np.isfinite(w.data).all():raise ValueError('nonfinite_WA')
    return w,messages


def peak(w, start, end, origin):
    lo = float(origin + start - w.stats.starttime)*w.stats.sampling_rate
    hi = float(origin + end - w.stats.starttime)*w.stats.sampling_rate
    if lo < -.02 or hi > len(w.data)+.02:
        raise ValueError('incomplete_window')
    i, j = max(0, int(lo)), min(len(w.data), int(np.ceil(hi)))
    if j-i < 2:
        raise ValueError('empty_window')
    return float(np.max(abs(w.data[i:j])))


def station_rows(components):
    rows = []
    good = [c for c in components if c.get('status') == 'processed']
    keys = sorted(set((c['network'], c['station'], c['sensor']) for c in good))
    for net, sta, sensor in keys:
        cc = {c['component']: c for c in good if (c['network'], c['station'], c['sensor']) == (net, sta, sensor)}
        any_c = next(iter(cc.values()))
        for timing in ['hypocentral', 'epicentral']:
            suffix = '' if timing == 'hypocentral' else '_epic'
            for snr in [2., 3.]:
                pp = {k: c['peak_mm'+suffix] for k, c in cc.items() if c['snr'+suffix] >= snr}
                h = np.sqrt(pp['N']*pp['E']) if 'N' in pp and 'E' in pp else np.nan
                z = pp.get('Z', np.nan)
                if not np.isfinite(h) and not np.isfinite(z):
                    continue
                rows.append(dict(network=net, station=sta, sensor=sensor, snr_min=snr, timing=timing,
                  distance_km=any_c['distance_km'], azimuth=any_c['azimuth'],
                  ml_h=float(RECIPE.ml_horizontal(h, any_c['distance_km'])) if h>0 else np.nan,
                  ml_v=float(RECIPE.ml_vertical(z, any_c['distance_km'])) if z>0 else np.nan,
                  A_h_mm=float(h), A_z_mm=float(z),
                  response_source=';'.join(sorted(set(c['response_source'] for c in cc.values()))),
                  master_postissue=any(c['master_postissue'] for c in cc.values())))
    return rows


def aggregate(stations):
    modes = {
        'ML_Sheen': dict(snr=3, sensors=['HH','EL','HG'], horizontal=True),
        'ML_velocity': dict(snr=3, sensors=['HH','EL'], horizontal=True),
        'ML_SNR2': dict(snr=2, sensors=['HH','EL','HG'], horizontal=True),
        'ML_HV': dict(snr=3, sensors=['HH','EL','HG'], horizontal=False),
        'ML_BB': dict(snr=3, sensors=['HH'], horizontal=True),
        'ML_all_sensors': dict(snr=3, sensors=['HH','EL','HG'], horizontal=True),
        'ML_HG': dict(snr=3, sensors=['HG'], horizontal=True),
        'ML_strict_metadata': dict(snr=3, sensors=['HH','EL','HG'], horizontal=True, strict=True),
        'ML_far': dict(snr=3, sensors=['HH','EL','HG'], horizontal=True, mindist=40.),
        'ML_epicentral_timing': dict(snr=3, sensors=['HH','EL','HG'], horizontal=True, timing='epicentral'),
        'ML_vertical': dict(snr=3, sensors=['HH','EL','HG'], vertical=True),
    }
    summary = {}; selected = []
    for name, cfg in modes.items():
        valid = []
        for r in stations:
            if r['snr_min'] != cfg['snr'] or r['sensor'] not in cfg['sensors']:
                continue
            if r['timing'] != cfg.get('timing', 'hypocentral'):
                continue
            if cfg.get('strict') and r['master_postissue']:
                continue
            if r['distance_km'] < cfg.get('mindist', 10.):
                continue
            v = r['ml_v'] if cfg.get('vertical') else r['ml_h']
            if not cfg.get('horizontal', True) and not np.isfinite(v):
                v = r['ml_v']
            if np.isfinite(v):
                valid.append(dict(r, value=v))
        valid.sort(key=lambda r: ({'HH':0, 'EL':1, 'HG':2}[r['sensor']], r['network'], r['station']))
        chosen = {}; vals=[]
        for r in valid:
            key=(r['network'],r['station'])
            if key not in chosen:
                chosen[key]=r; vals.append(r['value'])
        vals=np.asarray(vals)
        med=float(np.median(vals)) if len(vals) else np.nan
        summary[name] = med if len(vals)>=3 else np.nan
        summary[name+'_n'] = len(vals)
        if name == 'ML_Sheen':
            scatter=float(1.4826*np.median(abs(vals-med))) if len(vals)>=2 else np.nan
            summary.update(ML_station_scatter=scatter,
                ML_station_std=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,
                ML_station_q16=float(np.quantile(vals,.16)) if len(vals) else np.nan,
                ML_station_q84=float(np.quantile(vals,.84)) if len(vals) else np.nan,
                ML_mean=float(np.mean(vals)) if len(vals)>=3 else np.nan,
                ML_min_distance=min((r['distance_km'] for r in chosen.values()),default=np.nan),
                ML_max_distance=max((r['distance_km'] for r in chosen.values()),default=np.nan),
                ML_median_distance=float(np.median([r['distance_km'] for r in chosen.values()])) if vals.size else np.nan,
                ML_postissue_n=sum(r['master_postissue'] for r in chosen.values()))
            for sensor in ['HH','EL','HG']:summary['ML_n_'+sensor]=sum(v['sensor']==sensor for v in chosen.values())
            # Same truncated-mean sensitivity as the published calculation, after choosing stations.
            vv=vals.copy()
            while len(vv)>=3:
                keep=abs(vv-vv.mean())<=.5
                if keep.all():break
                vv=vv[keep]
            summary['ML_trimmed_mean']=float(vv.mean()) if len(vv)>=3 else np.nan
            summary['ML_trimmed_mean_n']=len(vv)
            summary['ML_quality']='A' if len(vals)>=5 and scatter<=.35 else ('B' if len(vals)>=3 else 'F')
            for r in chosen.values():
                selected.append(dict(r, residual=r['value']-med))
    return summary, selected


def run_event(case):
    eid=case['event_id']; dest=ROOT/'ml_events'/(eid+'.json')
    if dest.exists():
        old=json.loads(dest.read_text())
        if old.get('config_sha256')==CONFIG_SHA:
            return dict(event_id=eid,cached=True,**old['summary'])
    tic=time.monotonic(); origin=UTCDateTime(case['origin_utc'])
    components=[]; hashes={}; phase={}
    pp=pd.read_csv(case['picks_csv'])
    pp=pp[(pp.Phase=='P') & (pp.Probability>=.3)].sort_values('Probability',ascending=False)
    for p in pp.to_dict('records'):
        phase.setdefault((p['Network'],p['Station']),float(UTCDateTime(p['Time'])-origin))
    for path in sorted(Path(case['waveform_dir']).glob('*.sac')):
        parts=path.name.split('.')
        if len(parts)!=5:continue
        _,net,sta,chan,_=parts
        if chan[:2] not in ['HH','EL','HG'] or chan[-1] not in 'ZNE':continue
        row=dict(network=net,station=sta,sensor=chan[:2],component=chan[-1],file=str(path),status='rejected')
        try:
            if IP.quarantined(eid,net,sta):raise ValueError('unresolved_colocated_response')
            tp=phase.get((net,sta))
            if tp is None:raise ValueError('no_observed_P')
            if not 0<tp<100:raise ValueError('invalid_observed_P')
            tr=read(path)[0]
            if (tr.stats.network,tr.stats.station,tr.stats.channel)!=(net,sta,chan):
                raise ValueError('filename_header_mismatch')
            r0=gps2dist_azimuth(case['kma_lat'],case['kma_lon'],tr.stats.sac.stla,tr.stats.sac.stlo)[0]/1000
            if r0>100:raise ValueError('outside_raw_aperture')
            metres,az,baz=gps2dist_azimuth(case['lat'],case['lon'],tr.stats.sac.stla,tr.stats.sac.stlo)
            dist=metres/1000
            if not 10<=dist<=100:raise ValueError('outside_10_100_km')
            hyp=float(np.hypot(dist,case['z_central'])); lo,hi=hyp/4,hyp/2
            if not IP.p_valid(hyp,tp):raise ValueError('P_timing_outside_4_8p5_km_s')
            row.update(distance_km=dist,azimuth=az,depth_km=case['z_central'],p_origin_s=tp,
                       signal_start_s=lo,signal_end_s=hi,noise_start_s=tp-6,noise_end_s=tp-1)
            others=[]
            for other in case.get('neighbors',[]):
                orad=np.hypot(gps2dist_azimuth(other['lat'],other['lon'],tr.stats.sac.stla,tr.stats.sac.stlo)[0]/1000,case['z_central'])
                other_p=other['dt_s']+tp+(orad-hyp)/6
                other_s=other['dt_s']+hyp/3.5+(orad-hyp)/3.5
                tail=30 if other['ml']>=4.5 else (20 if other['ml']>=4 else 10)
                others.append((other_p-1,other_s+tail))
            if collision((lo,hi),others):raise ValueError('catalog_signal_overlap')
            if collision((tp-6,tp-1),others):raise ValueError('catalog_noise_overlap')
            frag=tr.slice(origin+lo,origin+hi).data
            if len(frag)<2 or not np.isfinite(tr.data).all():raise ValueError('invalid_counts')
            if float(tr.stats.delta)>.025:raise ValueError('sampling_rate_below_40_Hz')
            if np.max(abs(frag))==0:raise ValueError('zero_signal')
            if len(frag)>50 and np.convolve((frag==0).astype(int),np.ones(50,dtype=int),'valid').max()>=50:
                raise ValueError('signal_zero_gap')
            if max(abs(float(frag.min())),abs(float(frag.max())))>=.98*2**23:
                extreme=(frag==frag.min())|(frag==frag.max())
                if np.convolve(extreme.astype(int),np.ones(5,dtype=int),'valid').max()>=5:
                    raise ValueError('possible_ADC_clipping')
            label,ch,units=response(tr,origin)
            hashes[str(path)]=sha(path)
            w,messages=wa_trace(tr,ch)
            a=peak(w,lo,hi,origin);noise=peak(w,tp-6,tp-1,origin)
            if a<=0 or noise<=0:raise ValueError('nonpositive_WA_signal_or_noise')
            # Timing-only sensitivity uses the same corrected full trace and noise.
            ae=peak(w,dist/4,dist/2,origin)
            if collision((dist/4,dist/2),others):ae=np.nan
            row.update(status='processed',peak_mm=a,noise_peak_mm=noise,snr=a/noise,
                peak_mm_epic=ae,snr_epic=ae/noise,response_source=label,response_units=units,
                response_sensitivity=float(ch.response.instrument_sensitivity.value),
                response_epoch_start=str(ch.start_date),response_epoch_end=str(ch.end_date),
                master_postissue=label=='master' and origin>UTCDateTime('2024-06-26T23:59:59'),warnings=messages)
        except Exception as exc:
            row['reason']=type(exc).__name__+':'+str(exc)[:160]
        components.append(row)
    stations=station_rows(components);summary,selected=aggregate(stations)
    result=dict(event_id=eid,config_sha256=CONFIG_SHA,summary=summary,components=components,stations=stations,
                selected=selected,source_sha256=hashes,neighbors=case.get('neighbors',[]),seconds=time.monotonic()-tic)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    return dict(event_id=eid,seconds=result['seconds'],**summary)


def cases():
    cohort=pd.read_csv(ROOT/'inputs/cohort.csv',dtype={'event_id':str}).to_dict('records')
    p=ROOT.parent/'sangju_20261117/results/inputs'
    target=json.loads((p/'event.json').read_text())
    target.update(kma_lat=target['lat'],kma_lon=target['lon'],z_central=19.76,
        waveform_dir=str(IP.target_counts()),picks_csv=str(p/'target_picks.csv'))
    cohort.append(target)
    cat=pd.read_csv(ROOT.parent/'data/necis/catalog/necis_events_domestic_1978_2026.csv')
    cat=cat[cat.magnitude>=1.5].copy()
    cat['seconds']=pd.to_datetime(cat.origin_utc,utc=True).astype('int64')/1e9
    for c in cohort:
        t=float(UTCDateTime(c['origin_utc']));near=cat[abs(cat.seconds-t)<=180]
        c['neighbors']=[]
        for other in near.to_dict('records'):
            dt=float(other['seconds']-t)
            if abs(dt)<.5:continue
            if gps2dist_azimuth(c['kma_lat'],c['kma_lon'],other['latitude'],other['longitude'])[0]/1000<=50:
                c['neighbors'].append(dict(dt_s=dt,lat=other['latitude'],lon=other['longitude'],ml=other['magnitude']))
    return cohort


def main():
    global CONFIG_SHA
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--events',nargs='*');ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    for folder in ['ml_events','ml_validation']:(ROOT/folder).mkdir(exist_ok=True)
    meta=json.loads((ROOT/'inputs/response_sources.json').read_text())
    signature=(ROOT/'code/instrument_processing.py').read_bytes()+Path(__file__).read_bytes()+Path(RECIPE.__file__).read_bytes()+json.dumps(POLICY,sort_keys=True).encode()+json.dumps(meta,sort_keys=True).encode()
    CONFIG_SHA=hashlib.sha256(signature).hexdigest()
    allcases=cases(); todo=allcases
    if args.events:todo=[c for c in todo if c['event_id'] in args.events]
    if args.limit:todo=todo[:args.limit]
    if any(not (ROOT/'ml_events'/(c['event_id']+'.json')).exists() or json.loads((ROOT/'ml_events'/(c['event_id']+'.json')).read_text()).get('config_sha256')!=CONFIG_SHA for c in todo):
        load_responses()
    metaout=dict(instrument_policy=IP.POLICY,policy=POLICY,config_sha256=CONFIG_SHA,recipe=dict(path=RECIPE.__file__,sha256=sha(RECIPE.__file__)),
                 driver=dict(path=str(Path(__file__)),sha256=sha(__file__)),response_sources=meta,
                 literature='https://doi.org/10.1785/0120180112',numpy=np.__version__)
    (ROOT/'ml_validation/provenance.json').write_text(json.dumps(metaout,indent=2)+'\n')
    tic=time.monotonic(); rows=[]
    with concurrent.futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(run_event,c):c['event_id'] for c in todo}
        for fut in concurrent.futures.as_completed(futures):
            r=fut.result(); rows.append(r)
            if len(rows)%20==0 or len(todo)<=5 or len(rows)==len(todo):
                print(f"{len(rows)}/{len(todo)}; {time.monotonic()-tic:.0f} s; last {r['event_id']}; ML {r['ML_Sheen']:.3f}, N={r['ML_Sheen_n']}",flush=True)
    # Rebuild aggregate table from matching caches, including target and previous batches.
    data=[]; stations=[]
    for c in allcases:
        path=ROOT/'ml_events'/(c['event_id']+'.json')
        if not path.exists():continue
        r=json.loads(path.read_text())
        if r.get('config_sha256')!=CONFIG_SHA:continue
        data.append(dict(event_id=c['event_id'],ML_KMA=c['ml'],**r['summary']))
        stations.extend(dict(event_id=c['event_id'],**s) for s in r['selected'])
    pd.DataFrame(data).sort_values('event_id').to_csv(ROOT/'tables/independent_ML_catalog.csv',index=False)
    pd.DataFrame(stations).to_csv(ROOT/'tables/independent_ML_stations.csv',index=False)
    print('Independent ML measured:',len(data),flush=True)


if __name__=='__main__':main()
