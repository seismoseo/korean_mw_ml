"""Read original co-located recordings; never edit the Haenam source project."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from obspy import read,read_inventory,UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]
HAENAM=Path('/home/msseo/works/27.Haenam_sequence_renewal')
CONT=WORK/'data/necis/continuous/2026'
FETCHED=HAENAM/'data/stations/responses/necis_fetched_20261011/extracted'
GATHERED=HAENAM/'relocation/hypodd_dl_renewal_R4/waveforms_100km'
HASH={}


def sha(path):
    path=Path(path)
    if str(path) not in HASH:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        HASH[str(path)]=h.hexdigest()
    return HASH[str(path)]


def audit_haenam():
    p=HAENAM/'magnitudes/local_ml/full_period/component_measurements.csv'
    d=pd.read_csv(p,dtype={'eid':str},low_memory=False)
    d['sensor']=d.trace_id.str.split('.').str[-1].str[:2]
    good=d[d.status.eq('processed')&d.p_source.eq('pick')&d.snr_corrected.ge(2)]
    keys=['eid','network','station','sensor']
    pairs=good.pivot_table(index=keys,columns='component',values='peak_mm',aggfunc='first')
    pairs=pairs[pairs.N.notna()&pairs.E.notna()].reset_index()
    final=pd.read_csv(HAENAM/'magnitudes/local_ml/full_period/station_magnitudes.csv',dtype={'eid':str})
    used=final.merge(pairs[keys],on=['eid','network','station'],validate='one_to_one')
    summary=used.groupby('sensor').agg(station_event_pairs=('eid','size'),events=('eid','nunique'),stations=('station','nunique')).reset_index()
    used.to_csv(ROOT/'tables/Haenam_sensor_use.csv',index=False)
    summary.to_csv(ROOT/'tables/Haenam_sensor_use_summary.csv',index=False)
    r=HAENAM/'magnitudes/spectral_mw_R6/20260825004122';spectra=[]
    for path in sorted((r/'Spectrum').glob('*_disp.pkl')):
        sta=path.name.split('_disp')[0]
        matches=list((r/'SAC').glob(f'*.{sta}.??Z.removed.sac'))
        if len(matches)!=1:raise ValueError('ambiguous historical sensor '+sta)
        spectra.append(dict(event_id='20260825004122',station=sta,sensor=matches[0].name.split('.')[3][:2],
                            spectral_file=str(path),displacement_file=str(matches[0])))
    pd.DataFrame(spectra).to_csv(ROOT/'tables/Haenam_spectral_sensor_example.csv',index=False)
    obj=dict(local_ml=summary.to_dict('records'),spectral_example=dict(event_id='20260825004122',
        station_spectra=len(spectra),sensors=dict(Counter(s['sensor'] for s in spectra))),
        component_table=dict(path=str(p),sha256=sha(p)))
    (ROOT/'tables/Haenam_usage.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(obj,indent=2),flush=True)


def full_trace(files,start,end,network,station,channel):
    selected=[]
    for path in files:
        st=read(str(path),starttime=start,endtime=end)
        st=st.select(network=network,station=station,channel=channel)
        if len(st):selected.extend(st)
    if not selected:raise ValueError('no trace in requested interval')
    from obspy import Stream
    st=Stream(selected)
    gaps=st.get_gaps()
    if gaps:raise ValueError('gap or overlap in original segments: '+str(gaps[:2]))
    st.merge(method=0)
    if len(st)!=1 or np.ma.isMaskedArray(st[0].data):raise ValueError('unresolved input segments')
    tr=st[0]
    # NECIS event packets have component-dependent leading/trailing times.
    # Keep their real coverage; require sufficient common pre-event data for
    # the 80 s noise window and sufficient coda for the full-wave comparison.
    origin=start+220
    if tr.stats.starttime>origin-90 or tr.stats.endtime<origin+200:raise ValueError('incomplete source coverage')
    if not np.isfinite(tr.data).all():raise ValueError('nonfinite counts')
    return tr


def save_pair(case,sta,coords,model,kind):
    origin=UTCDateTime(case['origin_utc']);eid=case['event_id'];network='KS'
    cutdir=ROOT/'inputs/cutouts'/eid;cutdir.mkdir(parents=True,exist_ok=True)
    respdir=ROOT/'inputs/responses';respdir.mkdir(exist_ok=True)
    epic,az,baz=gps2dist_azimuth(case['lat'],case['lon'],coords[0],coords[1])
    row=dict(event_id=eid,station=sta,network=network,**{k:case[k] for k in ['name','origin_utc','ML','lat','lon','depth_km']},
             latitude=coords[0],longitude=coords[1],epicentral_km=epic/1000,azimuth=az,traces={},responses={},sources=[])
    for sensor in ['HH','HG']:
        for comp in 'ZNE':
            channel=sensor+comp
            if kind=='continuous':
                source=list((CONT/sta).glob(f'KS.{sta}.{channel}.{origin.year}.{origin.julday:03d}.*'))
                resp=FETCHED/f'RESP.KS.{sta}..{channel}'
                inventory=read_inventory(resp)
            else:
                root=ROOT/'inputs/events'/eid
                source=[p for p in root.rglob('*') if p.is_file() and p.parent.name=='MSEED' and p.name.startswith(f'KS.{sta}.{channel}.')]
                resp=WORK/'sangju_20261117/results/inputs/necis_KS_sangju.xml'
                inventory=read_inventory(resp).select(network=network,station=sta,channel=channel,time=origin)
            if not source:raise ValueError('missing original '+sta+'.'+channel)
            inv=inventory.select(network=network,station=sta,channel=channel,time=origin)
            channels=[c for n in inv for s in n for c in s]
            if len(channels)!=1:raise ValueError('ambiguous dated response '+sta+'.'+channel)
            units=channels[0].response.instrument_sensitivity.input_units.upper().replace(' ','')
            if (sensor=='HH' and units not in ['M/S','M/SEC']) or (sensor=='HG' and units not in ['M/S**2','M/SEC**2','M/S/S']):
                raise ValueError('invalid response units '+units)
            tr=full_trace(source,origin-220,origin+280,network,sta,channel)
            out=cutdir/(f'{network}.{sta}.{channel}.mseed');tr.write(out,format='MSEED')
            rs=respdir/(f'{eid}.{network}.{sta}.{channel}.xml');inv.write(rs,format='STATIONXML')
            row['traces'][channel]=str(out);row['responses'][channel]=str(rs)
            for path in source:
                if not any(s['path']==str(path) for s in row['sources']):row['sources'].append(dict(path=str(path),sha256=sha(path)))
            row['sources'].append(dict(path=str(resp),sha256=sha(resp)))
    # Prefer the observed exported absolute arrival times; the export's SAC b
    # and reference time need not equal the rounded event id or relocated origin.
    pp=ss=None
    source=GATHERED/eid if kind=='continuous' else WORK/'sangju_20261117/results/inputs/waveforms_100km'/eid
    for path in source.glob(f'*.{sta}.??Z.sac'):
        tr=read(path,headonly=True)[0];sac=tr.stats.sac
        ref=tr.stats.starttime-float(sac.b)
        if 'a' in sac and sac.a>-1e4:pp=ref+float(sac.a)
        if 't0' in sac and sac.t0>-1e4:ss=ref+float(sac.t0)
    if pp is None or ss is None:
        arrivals=model.get_travel_times(source_depth_in_km=case['depth_km'],distance_in_degree=epic/1000/111.195,
                                        phase_list=['p','P','Pg','Pn','s','S','Sg','Sn'])
        p=min(a.time for a in arrivals if a.name[0].upper()=='P')
        s=min(a.time for a in arrivals if a.name[0].upper()=='S')
        if pp is None:pp=origin+p;row['P_source']='Kim1983 TauP'
        if ss is None:ss=origin+s;row['S_source']='Kim1983 TauP'
    row.setdefault('P_source','observed SAC');row.setdefault('S_source','observed SAC')
    row['P_utc']=str(pp);row['S_utc']=str(ss)
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sangju',action='store_true');args=ap.parse_args()
    for sub in ['inputs','tables']:(ROOT/sub).mkdir(exist_ok=True)
    audit_haenam()
    model=TauPyModel(model=str(HAENAM/'data/stations/taup/kim1983_buan.npz'))
    registry=pd.read_csv(HAENAM/'registry/events.csv',dtype={'eid':str})
    cat=pd.read_csv(HAENAM/'relocation/hypodd_dl_renewal_R4/haenam_dl_relocated.csv',dtype={'event_id':str})
    stations=pd.read_csv(HAENAM/'detection/station_table/stations.csv')
    cases=[]
    for eid in ['20260822214301','20260825002354']:
        r=cat[cat.event_id==eid].iloc[0]
        cases.append(dict(event_id=eid,name='Haenam',origin_utc=str(r.origin_time),ML=float(r.ML_template),
                          lat=float(r.latitude),lon=float(r.longitude),depth_km=float(r.depth_km),kind='continuous',stations=['MOPB','HANB','JAEB','BGDB']))
    if args.sangju:
        cases.append(dict(event_id='20261117141017',name='Sangju',origin_utc='2026-11-17T14:10:17Z',ML=3.6,
                          lat=36.52,lon=127.96,depth_km=19.76,kind='event',stations=['SAJB','OKCB','CHJ3','KESB','JECB','BURB']))
    rows=[];failures=[]
    target_stations=pd.read_csv(WORK/'sangju_20261117/results/inputs/target_stations.csv')
    for case in cases:
        for sta in case['stations']:
            try:
                coords=(target_stations if case['kind']=='event' else stations)
                xyz=coords[(coords.Network=='KS')&(coords.Code==sta)].iloc[0]
                row=save_pair(case,sta,(float(xyz.Latitude),float(xyz.Longitude)),model,case['kind'])
                rows.append(row);print(case['event_id'],sta,'paired',flush=True)
            except Exception as exc:
                failures.append(dict(event_id=case['event_id'],station=sta,error=str(exc)));print('FAILED',case['event_id'],sta,str(exc),flush=True)
    (ROOT/'inputs/pairs.json').write_text(json.dumps(rows,indent=2)+'\n')
    (ROOT/'inputs/pair_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    (ROOT/'inputs/source_hashes.json').write_text(json.dumps(HASH,indent=2)+'\n')
    print('Paired recordings:',len(rows),'failures:',len(failures),flush=True)


if __name__=='__main__':main()
