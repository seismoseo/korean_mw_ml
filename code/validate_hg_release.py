"""Validate the current magnitude catalogs against selected spectra and counts."""
from pathlib import Path
import hashlib,json,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import fitz
from spectra import fit_stack
from source_velocity import validate_catalog,CONFIG_SHA256
R=Path(__file__).resolve().parents[1]
d=pd.read_csv(R/'tables/Mw_ML_catalog.csv',dtype={'event_id':str})
ml=pd.read_csv(R/'tables/independent_ML_catalog.csv',dtype={'event_id':str})
paired=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str})
sta=pd.read_csv(R/'tables/independent_ML_stations.csv',dtype={'event_id':str})
assert len(d)==642 and len(ml)==643 and len(paired)==642
assert d.event_id.is_unique and ml.event_id.is_unique
validate_catalog(d)
assert d.loc[d.Mw.notna(),'source_velocity_config_sha256'].eq(CONFIG_SHA256).all()
depths=dict(zip(d.event_id,d.z_central));depths['20261117141017']=19.76
assert '20261117141017' not in set(d.event_id)
assert sta.groupby(['event_id','network','station']).size().max()==1
assert sta.sensor.isin(['HH','EL','HG']).all()
signatures=set();mlsigs=set();units={};n_hg=0;fs=set();refits=[];source_checks=0
for eid in ml.event_id:
    raw=json.loads((R/'events'/(eid+'.json')).read_text())
    signatures.add(raw['config_sha256'])
    amp=json.loads((R/'ml_events'/(eid+'.json')).read_text());mlsigs.add(amp['config_sha256'])
    q=sta[sta.event_id==eid];row=ml[ml.event_id==eid].iloc[0]
    assert row.ML_Sheen_n==len(q),(eid,row.ML_Sheen_n,len(q))
    if len(q)>=3:assert abs(np.median(q.value)-row.ML_Sheen)<1e-9
    for c in amp['components']:
        if c['status']!='processed':continue
        assert c['response_source'] in ['merged','necis_current','master']
        if c['sensor']=='HG':
            assert c['response_units'] in ['M/S**2','M/SEC**2','M/S/S','M/S^2']
            n_hg+=1
        k=c['sensor']+':'+c['response_units'];units[k]=units.get(k,0)+1
    for s in amp['selected']:
        comp=[c for c in amp['components'] if c['status']=='processed' and
          (c['network'],c['station'],c['sensor'])==(s['network'],s['station'],s['sensor']) and c['component'] in 'NE']
        assert len(comp)==2 and min(c['snr'] for c in comp)>=3
    path=R/'spectra'/(eid+'.npz')
    if not path.exists():
        assert not raw['fits'];continue
    b=np.load(path)
    for tag in ['T10.1','T20.1']:
        if tag not in b:continue
        names=b[tag+'_stations'];assert len(names)==len(set(names)),(eid,tag)
        assert np.isin(b[tag+'_sensor'],['HH','EL','HG']).all()
        assert (np.isfinite(b[tag][:,0]).sum(axis=1)>=20).all()
    logs=pd.read_csv(R/'station_logs'/(eid+'.csv'))
    if 'dt_s' in logs:fs.update((1/logs.dt_s.dropna()).round(5))
    # Repeat the primary fit for a spread of events and for the target.
    if (int(eid[-2:])%7==0 or eid=='20261117141017') and 'T10.1' in b:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            fits,_=fit_stack(b['frequency_Hz'],b['T10.1'],depth_km=depths[eid])
        q=next((f for f in fits if f['method']=='Multitaper' and f['shape']=='Boatwright'),{})
        expected=next((f for f in raw['fits'] if f['sensor_set']=='ALL' and f['window_s']==10.1 and f['method']=='Multitaper' and f['shape']=='Boatwright'),{})
        if np.isfinite(expected.get('Mw',np.nan)):
            assert abs(q['Mw']-expected['Mw'])<1e-8
            refits.append(eid)
    # A deterministic station subset plus every selected target component:
    # verify immutable source hashes without rehashing the full national archive.
    paths=amp['source_sha256']
    for filename,h in paths.items():
        if eid!='20261117141017' and int(hashlib.sha256(filename.encode()).hexdigest()[:6],16)%100:continue
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==h
        source_checks+=1
assert len(signatures)==1 and len(mlsigs)==1
assert min(fs)>=80,fs  # Adopted spectral high prefilter must lie above 20 Hz.
assert n_hg>0
assert paired.ML_comparison_primary.equals((paired.Mw_quality=='A')&(paired.ML_quality=='A')&paired.primary_geography)
t=json.loads((R/'tables/Sangju2026_summary.json').read_text())
assert abs(t['local_magnitude']['ML_Sheen']-ml[ml.event_id=='20261117141017'].iloc[0].ML_Sheen)<1e-10
assert not ((sta.event_id=='20261117141017')&(sta.station=='BURB')).any()
assert json.loads((R/'ml_validation/HG_processing_validation.json').read_text())['passed']
atlas=pd.read_csv(R/'tables/spectral_atlas_index.csv',dtype={'event_id':str})
assert len(atlas)==642 and set(atlas.event_id)==set(d.event_id)
pages={p.name:len(fitz.open(p)) for p in (R/'reports').glob('*.pdf')}
assert pages['Spectral_fit_atlas.pdf']==161
with fitz.open(R.parent/'sangju_20261117/Sangju_20261117_magnitudes.pdf') as target_pdf:
    assert len(target_pdf)==4
    assert f"{t['preferred']['Mw']:.3f}" in target_pdf[0].get_text()
other=json.loads((R.parent/'sangju_20261117/results/tables/magnitudes/current_magnitudes.json').read_text())
assert other['preferred']==t['preferred']
assert json.dumps(other['local_magnitude'],sort_keys=True)==json.dumps(t['local_magnitude'],sort_keys=True)
result=dict(passed=True,n_historical=642,n_including_Sangju=643,n_selected_ML_sites=len(sta),
 n_processed_HG_components=n_hg,unit_counts=units,sampling_rates_Hz=sorted(fs),
 n_reproduced_spectral_fits=len(refits),n_verified_count_hashes=source_checks,
 spectral_signature=next(iter(signatures)),ML_signature=next(iter(mlsigs)),pdf_pages=pages,
 primary_Mw_A=int((d.Mw_quality=='A').sum()),paired_primary=int(paired.ML_comparison_primary.sum()))
(R/'tables/release_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
