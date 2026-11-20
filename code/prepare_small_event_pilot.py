"""Freeze a magnitude/depth/era-stratified small-earthquake feasibility pilot."""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
P=R/'small_events'
SEED=20261119


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def strata(frame):
    d=frame.copy()
    d['depth_stratum']=pd.cut(d.z_central,[0,10,16.89,100],right=False,
        labels=['0-10km','10-16p89km','ge16p89km']).astype(str)
    d['era_stratum']=pd.cut(pd.to_datetime(d.origin_utc).dt.year,[2009,2016,2021,2026],
        labels=['2010-2016','2017-2021','2022-2026']).astype(str)
    return d


def main():
    for sub in ['inputs','tables','figures','reports','events','spectra','station_logs','ml_events','logs']:
        (P/sub).mkdir(parents=True,exist_ok=True)
    depth=R.parent/'sangju_20261117/results/tables/depth/national_confident_events.csv'
    d=pd.read_csv(depth,dtype={'event_id':str})
    parent=strata(d[d.ml.ge(.5)&d.ml.lt(2)&d.region.ne('North Korea (land)')])
    parent['magnitude_stratum']=pd.cut(parent.ml,[.5,1,1.5,2],right=False,
        labels=['0p5-0p9','1p0-1p4','1p5-1p9']).astype(str)
    parent.to_csv(P/'inputs/eligible_small_events.csv',index=False)
    rng=np.random.default_rng(SEED);draws=[];counts=[]
    for keys,q in parent.groupby(['magnitude_stratum','depth_stratum','era_stratum'],sort=True):
        q=q.sort_values('event_id');n=min(6,len(q))
        chosen=q.iloc[np.sort(rng.choice(len(q),n,replace=False))].copy()
        chosen['selection_probability']=n/len(q);chosen['pilot_role']='small'
        draws.append(chosen)
        counts.append(dict(magnitude_stratum=keys[0],depth_stratum=keys[1],era_stratum=keys[2],parent_events=len(q),selected_events=n,selection_probability=n/len(q)))
    controls=strata(d[d.ml.ge(2)&d.ml.lt(2.5)&d.region.ne('North Korea (land)')])
    for keys,q in controls.groupby(['depth_stratum','era_stratum'],sort=True):
        q=q.sort_values('event_id');n=min(2,len(q))
        chosen=q.iloc[np.sort(rng.choice(len(q),n,replace=False))].copy()
        chosen['magnitude_stratum']='2p0-2p4_control';chosen['selection_probability']=n/len(q)
        chosen['pilot_role']='control';draws.append(chosen)
        counts.append(dict(magnitude_stratum='2p0-2p4_control',depth_stratum=keys[0],era_stratum=keys[1],parent_events=len(q),selected_events=n,selection_probability=n/len(q)))
    pilot=pd.concat(draws,ignore_index=True).sort_values(['pilot_role','event_id'])
    assert pilot.event_id.is_unique and len(parent)==9325 and (pilot.pilot_role=='small').sum()==162
    census=pd.DataFrame(counts);census.to_csv(P/'tables/sampling_strata.csv',index=False)
    # Inputs are resolved only after the draw; missing files are not replaced.
    availability=[]
    for row in pilot.to_dict('records'):
        eid=row['event_id'];chosen=None
        for label,base in [
            ('pilot',Path('/home/msseo/works/38.SKP_lower_crustal_earthquakes/runs')/eid/'out'),
            ('national',Path('/home/msseo/works/38.SKP_lower_crustal_earthquakes/national/runs')/eid/'out'),
            ('kma16',Path('/home/msseo/works/16.kma_absolute_location/runs/kma_batch/runs')/eid)]:
            picks=base/'picks'/f'{eid}_picks.csv';waves=base/'waveforms_100km'/eid
            if not picks.exists():continue
            paths=list(waves.glob('*.sac'))
            channels={}
            for path in paths:
                parts=path.name.split('.')
                if len(parts)!=5 or parts[3][:2] not in ['HH','EL','HG']:continue
                channels.setdefault((parts[1],parts[2],parts[3][:2]),set()).add(parts[3][-1])
            chosen=dict(event_id=eid,source=label,picks_csv=str(picks),waveform_dir=str(waves),
                waveform_files=len(paths),available_triplets=sum(set('ZNE').issubset(c) for c in channels.values()),
                input_available=bool(paths))
            break
        availability.append(chosen or dict(event_id=eid,source='missing',picks_csv='',waveform_dir='',waveform_files=0,available_triplets=0,input_available=False))
    pilot=pilot.merge(pd.DataFrame(availability),on='event_id',validate='one_to_one')
    pilot['primary_geography']=True;pilot['target_holdout']=False
    pilot['family_block']=[f'family_{int(f)}' if f>=0 else 'event_'+eid for f,eid in zip(pilot.family,pilot.event_id)]
    pilot.to_csv(P/'inputs/pilot_cohort.csv',index=False)
    config=dict(scope='Small-event feasibility pilot, not a complete national extension or an adopted conversion formula',
        seed=SEED,eligible_small_events=len(parent),small_events=int(pilot.pilot_role.eq('small').sum()),
        controls=int(pilot.pilot_role.eq('control').sum()),stratification='KMA ML [0.5,1),[1,1.5),[1.5,2); depths [0,10),[10,16.89),>=16.89 km; eras 2010-16,2017-21,2022-26',
        sampling='Six uniformly random events per small-event stratum; two ML2-2.4 controls per depth/era stratum. No replacement for missing data; selection made before waveform inspection.',
        source_catalog=str(depth),source_catalog_sha256=sha(depth),
        parent_catalog='inputs/eligible_small_events.csv',cohort='inputs/pilot_cohort.csv',
        primary_sensors=['HH','EL','HG'],quality_rules='../inputs/analysis_quality_rules.json',
        quality_policy='Same Mw/ML thresholds, source model, source constants, windows and band as the ML>=2 study. No acceptance based on Mw-ML agreement.',
        source_velocity_model='Kim1983',comparison_velocity_model='Kim2011',Mw_constant=9.05,
        overlap_policy='Use every known domestic catalog event (including ML<1.5), within 180 s and 50 km; same window-collision rules. Known-event coverage is incomplete.',
        frozen_reference_configuration_sha256=sha(R/'inputs/study_config.json'),
        pilot_cohort_sha256=sha(P/'inputs/pilot_cohort.csv'),
        eligibility_caveat='Conditional on successful original detection/picking and G2/G3 depth selection; not all Korean seismicity.',
        regression_caveat='Pilot sampling and SNR-dependent acceptance are not a complete selection-function correction. Any extension fit is descriptive and provisional.',
        recording_caveat='Modern station coverage and usable accelerometers can improve success; missing/failed cases remain in denominators.')
    (P/'inputs/pilot_config.json').write_text(json.dumps(config,indent=2)+'\n')
    protected=[*list((R/'reports').glob('*.pdf')),*[R/name for name in [
        'inputs/cohort.csv','inputs/study_config.json','inputs/analysis_quality_rules.json',
        'tables/Mw_ML_catalog.csv','tables/Mw_ML_primary.csv','tables/Mw_independent_ML_catalog.csv',
        'tables/Mw_independent_ML_primary.csv','tables/independent_ML_regressions.csv',
        'tables/Hong_regression_comparison.csv','tables/potency_event_catalog.csv','tables/Sangju2026_summary.json']]]
    (P/'inputs/protected_results.json').write_text(json.dumps({str(path.relative_to(R)):sha(path) for path in protected},indent=2)+'\n')
    print(json.dumps(dict(parent=len(parent),pilot=len(pilot),small=int(pilot.pilot_role.eq('small').sum()),controls=int(pilot.pilot_role.eq('control').sum()),input_available=int(pilot.input_available.sum()),source_counts=pilot.source.value_counts().to_dict(),magnitude_counts=pilot.magnitude_stratum.value_counts().to_dict()),indent=2),flush=True)


if __name__=='__main__':main()
