"""Freeze the magnitude-independent depth cohort and local input availability."""
from pathlib import Path
import hashlib,json,os,sys
sys.dont_write_bytecode=True
import pandas as pd
HERE=Path(__file__).resolve().parents[1]
WORK=HERE.parent
DEPTH=WORK/'sangju_20261117/results/tables/depth/national_confident_events.csv'
SOURCE=Path('/home/msseo/works/38.SKP_lower_crustal_earthquakes')
for sub in ['inputs','tables','figures','spectra','reports','logs']:(HERE/sub).mkdir(parents=True,exist_ok=True)
d=pd.read_csv(DEPTH,dtype={'event_id':str})
d=d[(d.ml>=2)&d.grade.isin(['G2','G3'])].copy()
d['primary_geography']=d.region.ne('North Korea (land)')
d['target_holdout']=False
paths=[]
for r in d.to_dict('records'):
    eid=r['event_id'];found=False
    for label,root,out in [('pilot',SOURCE/'runs'/eid,'out'),
       ('national',SOURCE/'national/runs'/eid,'out'),
       ('kma16',Path('/home/msseo/works/16.kma_absolute_location/runs/kma_batch/runs')/eid,'')]:
        base=root/out
        picks=base/'picks'/(eid+'_picks.csv');waves=base/'waveforms_100km'/eid
        if picks.exists():
            files=list(waves.glob('*.sac')) if waves.exists() else []
            z=[p for p in files if p.name.split('.')[-2] in ['HHZ','ELZ']]
            paths.append(dict(event_id=eid,source=label,picks_csv=str(picks),waveform_dir=str(waves),
               waveform_files=len(files),velocity_Z_files=len(z),input_kind='raw_counts'))
            found=True;break
    if not found:paths.append(dict(event_id=eid,source='missing',waveform_files=0,velocity_Z_files=0))
d=d.merge(pd.DataFrame(paths),on='event_id',how='left')
d.to_csv(HERE/'inputs/cohort.csv',index=False)
# The complete processing configuration. run_catalog.py hashes this file into its cache signature,
# so the key order and wording must stay exactly as written here.
SV=HERE/'inputs/source_velocity_models.json';sv=json.loads(SV.read_text())
config=dict(population='KMA ML>=2 events with G2/G3 depths, 2010-01-01 to 2026-10-31; Sangju 2026-11-17 held out',
 selection_sha256=hashlib.sha256(DEPTH.read_bytes()).hexdigest(),candidate_events=len(d),
 primary_geography_events=int(d.primary_geography.sum()),G3_events=int((d.grade=='G3').sum()),
 quality='Tier 1, all station deletions located; G2 depth interval <=6 km and jackknife range <=4 km; G3 <=4 km and <=2 km',
 depth_interval_kind='Project model envelope and calibrated random error, not a calibrated 95% confidence interval',
 depth_analysis='Continuous depth and national D90=16.89 km; no hard cutoff at 20 km',
 Mw_constant=9.05,density_kg_m3=2800.,radiation=.63,free_surface=2.,Q0=229.2,Qeta=.73,
 primary_window_s=10.1,secondary_window_s=20.1,frequency_Hz=[.5,20.],nw=4,kspec=7,
 max_epicentral_km=100.,amplitude='sqrt(N*dt*P1/2); direct dt FFT cross-check',
 station_frequency_gate='amplitude SNR>=3; contributions>=max(3,30% of selected stations)',
 spectral_fit='Generalized Boatwright, continuously profiled log plateau, n=1.5..4; Brune sensitivity',
 moment_quality_A='>=8 distinct sites, observed plateau, usable band >=1 decade, MT/Fourier and Brune/Boatwright differences <=0.10 Mw, 10.1/20.1 s difference <=0.10 Mw, quiet windows',
 moment_quality_B='>=5 distinct sites with observed plateau; estimator/source-shape/window differences <=0.15 Mw; no known overlap',
 corner_policy='Upper-bound corner does not alone invalidate a resolved moment plateau; source radius/stress drop not inferred from it',
 overlap_policy='Flag other ML>=1.5 catalog events within 50 km whose origin+model S travel time falls in signal/noise windows; exclude affected primary windows',
 regression='Descriptive Mw on ML, family-balanced fit and NND-family block bootstrap; G3-only, offshore, era, and depth/Vs sensitivities',
 ML_role='Comparison only. No fit parameter, station selection, amplitude offset or acceptance gate uses Mw-ML agreement.',
 adopted_catalogs_modified=False,
 primary_sensors=['HH','EL','HG'],
 sensor_selection='One usable triplet per site; HH then EL then HG; no selection on amplitude or ML agreement',
 response_removal='DISP, water_level=None, cosine prefilter .1/.2/25/40 Hz capped by Nyquist; full-record linear detrend and one 1-percent taper capped at 5 s',
 noise_window='Equal duration ending P-1 s, moved earlier only for known catalog overlap/coverage',
 P_timing_QA='Hypocentral distance / apparent velocity 4..8.5 km/s, plus or minus .5 s; applied before measurements',
 reference_Vs_m_s=float(sv['reference_Vs_m_s']),source_velocity_model=sv['primary_model'],
 source_velocity_comparison=sv['comparison_model'],source_velocity_config='inputs/source_velocity_models.json',
 source_velocity_config_sha256=hashlib.sha256(SV.read_bytes()).hexdigest(),
 source_depth_policy=sv['depth_policy'],source_velocity_scope=sv['scope'])
(HERE/'inputs/study_config.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps(dict(events=len(d),primary_geography=int(d.primary_geography.sum()),
 sources=d.source.value_counts().to_dict(),with_velocity_waveforms=int((d.velocity_Z_files>0).sum()),
 channels=int(d.waveform_files.sum()),config_sha256=hashlib.sha256((HERE/'inputs/study_config.json').read_bytes()).hexdigest()),indent=2))
