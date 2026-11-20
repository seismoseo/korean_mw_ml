"""Freeze the expansion before reading new waveform amplitudes or fitting Mw–ML."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from prepare_small_event_pilot import strata

R=Path(__file__).resolve().parents[1];P=R/'small_events';E=P/'extension'
SEED=20261120


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def neighbors(case,catalog):
    t=float(UTCDateTime(case['origin_utc']));out=[]
    times=catalog.seconds.to_numpy();lo,hi=np.searchsorted(times,[t-180,t+180],side='left')
    for row in catalog.iloc[lo:hi+1].to_dict('records'):
        dt=float(row['seconds']-t)
        if abs(dt)<.5 or abs(dt)>180:continue
        if gps2dist_azimuth(case['kma_lat'],case['kma_lon'],row['latitude'],row['longitude'])[0]/1000<=50:
            out.append(dict(dt_s=dt,lat=row['latitude'],lon=row['longitude'],ml=row['magnitude']))
    return out


def neighbor_key(items):
    return sorted(tuple(round(float(v[k]),6) for k in ['dt_s','lat','lon','ml']) for v in items)


def availability(row):
    eid=row['event_id']
    if row.get('source') in ['pilot','national','kma16'] and row.get('picks_csv'):
        pairs=[(row['source'],Path(row['picks_csv']),Path(row['waveform_dir']))]
    else:
        pairs=[]
        for label,base in [('pilot',Path('/home/msseo/works/38.SKP_lower_crustal_earthquakes/runs')/eid/'out'),
                           ('national',Path('/home/msseo/works/38.SKP_lower_crustal_earthquakes/national/runs')/eid/'out'),
                           ('kma16',Path('/home/msseo/works/16.kma_absolute_location/runs/kma_batch/runs')/eid)]:
            pairs.append((label,base/'picks'/f'{eid}_picks.csv',base/'waveforms_100km'/eid))
    for label,picks,waves in pairs:
        if not picks.exists():continue
        paths=list(waves.glob('*.sac'));channels={}
        for path in paths:
            a=path.name.split('.')
            if len(a)==5 and a[3][:2] in ['HH','EL','HG']:
                channels.setdefault((a[1],a[2],a[3][:2]),set()).add(a[3][-1])
        return dict(event_id=eid,source=label,picks_csv=str(picks),waveform_dir=str(waves),
                    waveform_files=len(paths),available_triplets=sum(set('ZNE').issubset(c) for c in channels.values()),
                    input_available=bool(paths))
    return dict(event_id=eid,source='missing',picks_csv='',waveform_dir='',waveform_files=0,
                available_triplets=0,input_available=False)


def main():
    if (E/'inputs/config.json').exists():
        config=json.loads((E/'inputs/config.json').read_text())
        assert sha(E/'inputs/cohort.csv')==config['cohort_sha256']
        print('Frozen extension already exists; selection preserved',flush=True);return
    for folder in ['inputs','tables','reports','figures','events','ml_events','station_logs','spectra','logs','nw2_windows/events','nw2_windows/spectra']:
        (E/folder).mkdir(parents=True,exist_ok=True)
    parent=pd.read_csv(P/'inputs/eligible_small_events.csv',dtype={'event_id':str})
    pilot=pd.read_csv(P/'inputs/pilot_cohort.csv',dtype={'event_id':str})
    reference=strata(pd.read_csv(R/'inputs/cohort.csv',dtype={'event_id':str}))
    rng=np.random.default_rng(SEED);draws=[];cells=[]
    for keys,q in parent.groupby(['magnitude_stratum','depth_stratum','era_stratum'],sort=True):
        q=q.sort_values('event_id');already=pilot[pilot.event_id.isin(q.event_id)]
        if keys[0]=='1p5-1p9':chosen=q.copy();target=len(q);role='census_1p5_1p9'
        elif keys[0]=='1p0-1p4':
            target=min(60,len(q));remaining=q[~q.event_id.isin(already.event_id)]
            extra=remaining.iloc[np.sort(rng.choice(len(remaining),target-len(already),replace=False))]
            chosen=pd.concat([q[q.event_id.isin(already.event_id)],extra]);role='nested_1p0_1p4'
        else:chosen=q[q.event_id.isin(already.event_id)].copy();target=len(chosen);role='pilot_only_0p5_0p9'
        chosen=chosen.copy();chosen['selection_probability']=target/len(q);chosen['extension_role']=role;draws.append(chosen)
        cells.append(dict(magnitude_stratum=keys[0],depth_stratum=keys[1],era_stratum=keys[2],parent_events=len(q),selected_events=target,selection_probability=target/len(q)))
    reference['selection_probability']=1.;reference['extension_role']='reference_MLge2'
    reference['magnitude_stratum']='MLge2_reference';draws.append(reference)
    cohort=pd.concat(draws,ignore_index=True).sort_values('event_id').reset_index(drop=True)
    assert cohort.event_id.is_unique and len(cohort)==3532
    cohort['pilot_role']=np.where(cohort.ml.lt(2),'small','control')
    cohort['in_original_pilot']=cohort.event_id.isin(pilot.event_id)
    cohort['confirmatory_small']=cohort.ml.lt(2)&~cohort.in_original_pilot
    cohort['family_block']=[f'family_{int(f)}' if f>=0 else 'event_'+eid for f,eid in zip(cohort.family,cohort.event_id)]
    cohort['primary_geography']=True;cohort['target_holdout']=False
    print('Resolving input paths for',len(cohort),'preselected events',flush=True)
    with ThreadPoolExecutor(12) as pool:resolved=list(pool.map(availability,cohort.to_dict('records')))
    cohort=cohort.drop(columns=[c for c in ['source','picks_csv','waveform_dir','waveform_files','available_triplets','input_available'] if c in cohort])
    cohort=cohort.merge(pd.DataFrame(resolved),on='event_id',validate='one_to_one')
    catpath=R.parent/'data/necis/catalog/necis_events_domestic_1978_2026.csv'
    catalog=pd.read_csv(catpath);catalog['seconds']=pd.to_datetime(catalog.origin_utc,utc=True).astype('int64')/1e9
    catalog=catalog.sort_values('seconds').reset_index(drop=True)
    all_neighbors={};registry=[];baseline_roots=[];candidate_roots=[]
    old_cases={str(root):frame.set_index('event_id') for root,frame in [(P,pilot),(R,reference)]}
    for case in cohort.to_dict('records'):
        eid=case['event_id'];near=neighbors(case,catalog);all_neighbors[eid]=near;base=E;candidate=E/'nw2_windows';reuse={}
        for root in [P,R]:
            known=old_cases[str(root)]
            if eid not in known.index:continue
            old=known.loc[eid]
            if not all(abs(float(case[k])-float(old[k]))<1e-8 for k in ['lat','lon','z_central','kma_lat','kma_lon']):continue
            if not all(str(case[k])==str(old[k]) for k in ['picks_csv','waveform_dir']):continue
            mw=root/'events'/f'{eid}.json';ml=root/'ml_events'/f'{eid}.json';log=root/'station_logs'/f'{eid}.csv'
            if not all(p.exists() for p in [mw,ml,log]):continue
            raw=json.loads(mw.read_text());local=json.loads(ml.read_text())
            if raw.get('status')=='error' or 'error' in local:continue
            if neighbor_key(raw.get('neighbors',[]))!=neighbor_key(near):continue
            if neighbor_key(local.get('neighbors',[]))!=neighbor_key(near):continue
            base=root
            for p in [mw,ml,log,root/'spectra'/f'{eid}.npz']:
                if p.exists():reuse[str(p)]=sha(p)
            if root==P and (P/'nw2_windows/events'/f'{eid}.json').exists():
                candidate=P/'nw2_windows'
                for p in [candidate/'events'/f'{eid}.json',candidate/'spectra'/f'{eid}.npz']:
                    if p.exists():reuse[str(p)]=sha(p)
            break
        baseline_roots.append(str(base));candidate_roots.append(str(candidate))
        registry.append(dict(event_id=eid,baseline_root=str(base),candidate_root=str(candidate),
                             reused_files_sha256=json.dumps(reuse,sort_keys=True),known_neighbors=len(near),neighbors_below_ML1p5=sum(n['ml']<1.5 for n in near)))
    cohort['baseline_root']=baseline_roots;cohort['candidate_root']=candidate_roots
    cohort.to_csv(E/'inputs/cohort.csv',index=False)
    pd.DataFrame(registry).to_csv(E/'inputs/reuse_registry.csv',index=False)
    pd.DataFrame(cells).to_csv(E/'tables/sampling_strata.csv',index=False)
    (E/'inputs/neighbors.json').write_text(json.dumps(all_neighbors,indent=2)+'\n')
    protect=json.loads((P/'inputs/protected_results.json').read_text())
    for p in [*P.rglob('*.pdf'),*P.glob('tables/*'),*P.glob('inputs/*'),R/'code/spectra.py',R/'code/source_velocity.py',R/'code/run_catalog.py',R/'code/local_magnitude.py',R/'code/instrument_processing.py']:
        if p.is_file() and E not in p.parents:protect[str(p.relative_to(R))]=sha(p)
    (E/'inputs/protected_results.json').write_text(json.dumps(protect,indent=2)+'\n')
    plan=dict(scope='Expanded selected-event Mw–ML study; preserve the adopted national catalog and pilot',seed=SEED,
        selected_events=len(cohort),small_events=int(cohort.ml.lt(2).sum()),reference_events=len(reference),
        design='Census of confident-depth catalog ML1.5–1.9; nested uniform total min(60,N) per depth/era cell at ML1–1.4; retain original 54 ML0.5–0.9 pilot cases; all 642 ML>=2 reference events.',
        new_small_cases=int(cohort.confirmatory_small.sum()),no_failed_cases_replaced=True,
        primary_spectral_scheme=dict(windows_s=[5.1,10.1],NW=2,K=3,band_Hz=[.5,20]),
        comparison_scheme=dict(windows_s=[10.1,20.1],NW=4,K=7,band_Hz=[.5,20]),
        frozen_quality_rules_sha256=sha(R/'inputs/analysis_quality_rules.json'),quality_policy='Unchanged A/B gates; independent ML; no acceptance by Mw–ML agreement.',
        source_model='Kim1983 at source depth; rho2800; C9.05',sensors=['HH','EL','HG'],
        overlap_policy='All known domestic magnitudes, 180 s and 50 km; reuse only if identical known-neighbor lists and event/input coordinates/paths.',
        cohort_sha256=sha(E/'inputs/cohort.csv'),reuse_registry_sha256=sha(E/'inputs/reuse_registry.csv'),neighbors_sha256=sha(E/'inputs/neighbors.json'),
        parent_sha256=sha(P/'inputs/eligible_small_events.csv'),reference_cohort_sha256=sha(R/'inputs/cohort.csv'),
        neighbor_catalog=str(catpath),neighbor_catalog_sha256=sha(catpath),
        transfer_gate=dict(minimum_shared_Mw_A=30,maximum_abs_median_delta=.05,maximum_RMS_delta=.10,
            per_catalog_ML_bin_maximum_abs_median=.05,minimum_per_bin=5,bins=[2,2.5,3,6]),
        regression_plan=dict(primary='Both A, measured ML>=1.5, uniform NW2 estimator; family-balanced OLS with whole-family bootstrap and CV.',
            comparisons=['linear','quadratic','continuous hinge at measured ML2'],
            sensitivities=['equal-axis ODR','A or B','design and family weights','measured-ML floor','exclude original pilot events','same reference event set for both estimators'],
            no_selection_function_correction_claimed=True),
        uncertainty='Conditional station-resampling bounds, not total uncertainty. Bootstrap/CV of accepted events does not recover missing low-SNR events.',
        previous_results_preserved=True)
    (E/'inputs/config.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(selected=len(cohort),roles=cohort.extension_role.value_counts().to_dict(),
        input_available=int(cohort.input_available.sum()),baseline_roots=cohort.baseline_root.value_counts().to_dict(),
        candidate_roots=cohort.candidate_root.value_counts().to_dict()),indent=2),flush=True)


if __name__=='__main__':main()
