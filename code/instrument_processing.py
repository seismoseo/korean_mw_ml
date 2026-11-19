"""Common dated response selection and regularized HH/EL/HG deconvolution.

No water-level floor is imposed on the displacement response. The explicit
cosine prefilter limits integration of acceleration noise outside the fit band.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from obspy import read, read_inventory, UTCDateTime
from obspy.geodetics import gps2dist_azimuth

ROOT=Path(__file__).resolve().parents[1]
INV={}; COORD={}; SOURCES=[]
PRIORITY={'merged':0,'necis_current':1,'necis_october':2,'master':3}
POLICY=dict(version='HH_EL_HG_20261119_v1',water_level=None,
  spectral_prefilter_Hz=[.1,.2,25.,40.],
  full_record_taper='1 percent, capped at 5 s, once',
  spectral_noise='equal duration ending 1 s before P; quiet-window search',
  primary_sensors=['HH','EL','HG'],one_measurement_per_site=True,
  preference='HH, EL, HG among usable records; independent of amplitude or KMA ML',
  exclusions=['KS.KWJU.HH','KS.SEJF.EL'],
  target_pair_exclusion='Sangju KS.BURB: unresolved co-located horizontal response discrepancy; both families excluded',
  magnitude_constant=9.05)

def load():
    if INV:return
    meta=json.loads((ROOT/'inputs/response_sources.json').read_text())
    specs=[(Path(meta['master_source']),'master'),
      (Path(meta['merged_override']),'merged'),
      (ROOT.parent/'sangju_20261117/results/inputs/necis_KS_sangju.xml','necis_current')]
    for p in [ROOT.parent/'sangju_20261117/results/inputs/target_stations.csv',
      ROOT.parent.parent/'27.Haenam_sequence_renewal/detection/station_table/stations.csv']:
        for r in pd.read_csv(p).to_dict('records'):
            COORD.setdefault((r['Network'],r['Code']),(r['Latitude'],r['Longitude']))
    # Current NECIS download contains all KS sensor families with explicit epochs.
    # The audited merged inventory has first priority wherever its epoch matches.
    for p,label in specs:
        print('Loading response',label,flush=True)
        SOURCES.append(dict(path=str(p),label=label))
        for net in read_inventory(str(p)):
            for sta in net:
                for ch in sta:
                    if ch.code[:2] in ['HH','EL','HG']:
                        INV.setdefault((net.code,sta.code,ch.code),[]).append((label,ch))

def response(tr,when):
    key=(tr.stats.network,tr.stats.station,tr.stats.channel)
    if (key[0],key[1],key[2][:2]) in [('KS','KWJU','HH'),('KS','SEJF','EL')]:
        raise ValueError('documented_response_exclusion')
    choices=[(label,ch) for label,ch in INV.get(key,[]) if
      (ch.start_date is None or when>=ch.start_date) and
      (ch.end_date is None or when<=ch.end_date)]
    if not choices:raise ValueError('no_matching_response_epoch')
    rank=min(PRIORITY[a] for a,b in choices)
    choices=[v for v in choices if PRIORITY[v[0]]==rank]
    exact=[v for v in choices if v[1].location_code==tr.stats.location]
    choices=exact or choices
    if len(choices)!=1:raise ValueError('ambiguous_response_epoch')
    label,ch=choices[0]
    if ch.response is None or ch.response.instrument_sensitivity is None:
        raise ValueError('response_missing')
    units=ch.response.instrument_sensitivity.input_units.upper().replace(' ','')
    allowed=['M/S','M/SEC','M/S**1'] if key[2][:2]!='HG' else ['M/S**2','M/SEC**2','M/S/S','M/S^2']
    if units not in allowed:raise ValueError('response_units_unresolved:'+units)
    lat,lon=ch.latitude,ch.longitude
    if abs(lat)<.01 and abs(lon)<.01 and label.startswith('necis'):
        lat,lon=COORD[key[:2]]
    if gps2dist_azimuth(tr.stats.sac.stla,tr.stats.sac.stlo,lat,lon)[0]>2000:
        raise ValueError('response_station_coordinate_mismatch')
    if key[2][-1] in 'NE' and ch.dip is not None and abs(ch.dip)>10:
        raise ValueError('horizontal_channel_tilt')
    if not hasattr(ch,'_study_gain_ratio'):
        sens=ch.response.instrument_sensitivity
        try:
            v=ch.response.get_evalresp_response_for_frequencies([float(sens.frequency)],output='DEF',hide_sensitivity_mismatch_warning=True)[0]
            ch._study_gain_ratio=float(abs(v)/abs(sens.value))
        except Exception:ch._study_gain_ratio=float('nan')
    return label,ch,units

def displacement(tr,ch):
    w=tr.copy().detrend('demean').detrend('linear')
    w.stats.response=ch.response
    w.taper(.01,max_length=5.)
    nyq=w.stats.sampling_rate/2
    pre=[.1,.2,min(25.,.75*nyq),min(40.,.9*nyq)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w.remove_response(output='DISP',water_level=None,pre_filt=pre,
          zero_mean=False,taper=False,hide_sensitivity_mismatch_warning=True)
    messages=sorted(set(str(v.message) for v in caught))
    if any('not known to ObsPy' in v for v in messages):
        raise ValueError('response_units_unresolved_warning')
    if not np.isfinite(w.data).all():raise ValueError('nonfinite_displacement')
    return w,messages

def target_counts():
    """Canonical counts, with coordinates from the accompanying station table."""
    out=ROOT/'inputs/Sangju_counts';out.mkdir(exist_ok=True)
    coords={ (r['Network'],r['Code']):r for r in pd.read_csv(
      ROOT.parent/'sangju_20261117/results/inputs/target_stations.csv').to_dict('records') }
    raw=ROOT/'hh_hg_comparison/inputs/events/20261117141017'
    origin=UTCDateTime('2026-11-17T14:10:17')
    n=0
    for p in sorted(raw.glob('*/MSEED/*')):
        fields=p.name.split('.')
        if len(fields)<3:continue
        net,sta,chan=fields[:3]
        if chan[:2] not in ['HH','EL','HG'] or chan[-1] not in 'ZNE':continue
        if (net,sta) not in coords:continue
        c=coords[net,sta]
        if gps2dist_azimuth(36.52,127.96,c['Latitude'],c['Longitude'])[0]>110000:continue
        dest=out/f'20261117141017.{net}.{sta}.{chan}.sac'
        if dest.exists():continue
        stream=read(str(p))
        stream.merge(method=0)
        if len(stream)!=1 or np.ma.is_masked(stream[0].data):continue
        tr=stream[0]
        tr.stats.sac=dict(stla=c['Latitude'],stlo=c['Longitude'],stel=c['Elevation'],
          evla=36.52,evlo=127.96,evdp=19.76,o=float(origin-tr.stats.starttime))
        tr.write(str(dest),format='SAC');n+=1
    print('New canonical Sangju traces',n,flush=True)
    return out

def p_valid(hyp_km,tp):
    return hyp_km/8.5-.5 <= tp <= hyp_km/4.+.5

def quarantined(eid,net,sta):
    return eid=='20261117141017' and (net,sta)==('KS','BURB')
