"""Validate provenance, absolute timing, spectral units and report accounting."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.signal.windows import dpss
from obspy import read,read_inventory,UTCDateTime
import fitz
from compare_pairs import transforms
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


pairs=json.loads((ROOT/'inputs/pairs.json').read_text());assert len(pairs)==14
assert len({(p['event_id'],p['station']) for p in pairs})==14
assert sum(p['name']=='Sangju' for p in pairs)==6
checks=0;metadata={}
for p in pairs:
    origin=UTCDateTime(p['origin_utc']);locations=set()
    for key,path in p['traces'].items():
        tr=read(path)[0];locations.add(tr.stats.location)
        assert tr.id.startswith(p['network']+'.'+p['station']+'.')
        assert tr.stats.channel==key
        inv=read_inventory(p['responses'][key]);resp=inv.get_response(tr.id,origin)
        unit=resp.instrument_sensitivity.input_units.upper().replace(' ','')
        assert unit in (['M/S','M/SEC'] if key.startswith('HH') else ['M/S**2','M/SEC**2','M/S/S'])
        pp=UTCDateTime(p['P_utc']);ss=UTCDateTime(p['S_utc'])
        for start,end in [(pp-81,pp-1),(origin,origin+80),(ss-.1,ss+10)]:
            assert start>=tr.stats.starttime+5 and end<=tr.stats.endtime-5
        checks+=1
    assert len(locations)==1
    if p['name']=='Haenam':
        folder=Path('/home/msseo/works/27.Haenam_sequence_renewal/relocation/hypodd_dl_renewal_R4/waveforms_100km')/p['event_id']
    else:folder=ROOT.parents[1]/'sangju_20261117/results/inputs/waveforms_100km'/p['event_id']
    for path in folder.glob(f'*.{p["station"]}.??Z.sac'):
        tr=read(path,headonly=True)[0];ref=tr.stats.starttime-float(tr.stats.sac.b)
        for label,header in [('P','a'),('S','t0')]:
            if p[label+'_source']=='observed SAC' and header in tr.stats.sac:
                assert abs(UTCDateTime(p[label+'_utc'])-(ref+float(tr.stats.sac[header])))<1e-5
        metadata[str(path)]=sha(path)
    b=np.load(ROOT/'spectra'/(p['event_id']+'_'+p['station']+'.npz'))
    for tag in ['short','long']:
        f=b[tag+'_frequency_Hz'];assert np.all(np.diff(f)>0)
        for family in ['HH','HG']:
            assert np.isfinite(b[tag+'_'+family+'_signal_PSD']).all()
            assert (b[tag+'_'+family+'_signal_PSD']>=0).all()
        assert np.nanmin(b[tag+'_coherence_mean'])>=0 and np.nanmax(b[tag+'_coherence_mean'])<=1

source=json.loads((ROOT/'inputs/source_hashes.json').read_text())
for path,value in source.items():assert sha(path)==value,path

# Independent exact energy check for the one-sided PSD normalization.
rng=np.random.default_rng(7);x=rng.normal(size=(3,4096))
f,power,j,factor=transforms(x,100.)
integral=power.sum(axis=-1)*(f[1]-f[0])
h=dpss(4096,3.,Kmax=5,sym=False,norm=2)
expected=np.mean(np.sum(h[None,:,:]**2*detrend(x,axis=-1,type='linear')[:,None,:]**2,axis=-1),axis=1)
assert np.allclose(integral,expected,rtol=1e-12,atol=1e-14)

references=[Path('/home/msseo/works/27.Haenam_sequence_renewal')/s for s in [
    'relocation/hypodd_dl_renewal_R4/haenam_dl_relocated.csv','detection/station_table/stations.csv','data/stations/taup/kim1983_buan.npz']]
references.append(ROOT.parents[1]/'sangju_20261117/results/inputs/target_stations.csv')
metadata.update({str(p):sha(p) for p in references})
(ROOT/'inputs/metadata_source_hashes.json').write_text(json.dumps(metadata,indent=2)+'\n')
report=ROOT/'reports/HH_HG_paired_comparison.pdf'
with fitz.open(report) as pdf:
    assert len(pdf)==8 and all(len(p.get_text())>20 for p in pdf)
with fitz.open(ROOT.parent/'reports/Korean_Mw_ML_study.pdf') as pdf:combined_pages=len(pdf);assert combined_pages>=12
assert len(json.loads((ROOT/'tables/figure_manifest.json').read_text()))==6
response=pd.read_csv(ROOT/'tables/response_checks.csv');assert len(response)==84
assert abs(response.full_to_stated_gain-1).max()<.0061
result=dict(pairs=len(pairs),sensor_components=checks,source_files_content_hash_verified=len(source),
    metadata_files_content_hash_recorded=len(metadata),matched_location_codes=True,
    absolute_arrival_times='passed',window_coverage='passed',coherence_bounds='passed',
    one_sided_PSD_energy_error=float(max(abs(integral-expected))),
    known_acceleration_amplitude_tests='tables/processing_validation.json',
    report_pages=8,combined_report_pages=combined_pages,magnitude_estimation_performed_by_this_report=False)
(ROOT/'tables/validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
