"""Audit all expanded attempts, the fixed census/sample, controls and regression."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import fitz
from prepare_small_event_extension import neighbor_key
from source_velocity import validate_catalog
from analyze_small_event_extension import matrix

R=Path(__file__).resolve().parents[1];E=R/'small_events/extension';T=E/'tables'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def csv(path):return pd.read_csv(path,dtype={'event_id':str})
def close(a,b):return np.allclose(a,b,atol=1e-10,rtol=0,equal_nan=True)


def main():
    config=read(E/'inputs/config.json');cohort=csv(E/'inputs/cohort.csv')
    parent=csv(R/'small_events/inputs/eligible_small_events.csv');pilot=csv(R/'small_events/inputs/pilot_cohort.csv')
    assert len(cohort)==config['selected_events']==3532 and cohort.event_id.is_unique
    assert len(cohort[cohort.ml.lt(2)])==2890 and len(cohort[cohort.ml.ge(2)])==642
    assert sha(E/'inputs/cohort.csv')==config['cohort_sha256']
    assert sha(E/'inputs/reuse_registry.csv')==config['reuse_registry_sha256']
    assert sha(E/'inputs/neighbors.json')==config['neighbors_sha256']
    assert sha(R/'small_events/inputs/eligible_small_events.csv')==config['parent_sha256']
    assert sha(R/'inputs/cohort.csv')==config['reference_cohort_sha256']
    assert sha(R/'inputs/analysis_quality_rules.json')==config['frozen_quality_rules_sha256']
    census=cohort[cohort.extension_role.eq('census_1p5_1p9')]
    assert len(census)==2317 and set(census.event_id)==set(parent[parent.ml.ge(1.5)].event_id)
    assert census.selection_probability.eq(1).all()
    assert set(pilot.event_id).issubset(set(cohort.event_id))
    assert cohort.in_original_pilot.eq(cohort.event_id.isin(pilot.event_id)).all()
    assert cohort.confirmatory_small.eq(cohort.ml.lt(2)&~cohort.in_original_pilot).all()
    nested=cohort[cohort.extension_role.eq('nested_1p0_1p4')]
    assert len(nested)==519 and nested.ml.between(1,1.4).all()
    rng=np.random.default_rng(config['seed']);replay=[]
    for (mag,depth,era),q in parent.groupby(['magnitude_stratum','depth_stratum','era_stratum'],sort=True):
        if mag!='1p0-1p4':continue
        q=q.sort_values('event_id');old=q[q.event_id.isin(pilot.event_id)]
        rest=q[~q.event_id.isin(old.event_id)];target=min(60,len(q))
        chosen=rest.iloc[np.sort(rng.choice(len(rest),target-len(old),replace=False))]
        replay.extend(old.event_id);replay.extend(chosen.event_id)
        actual=nested[nested.depth_stratum.eq(depth)&nested.era_stratum.eq(era)]
        assert len(actual)==target and close(actual.selection_probability,np.full(target,target/len(q)))
    assert set(replay)==set(nested.event_id)
    print('Fixed cohort and sampling replay passed',flush=True)
    assert not cohort.target_holdout.any() and not cohort.event_id.eq('20261117141017').any()
    provenance=read(E/'inputs/processing_provenance.json');signature=provenance['signature']
    for filename,digest in provenance['sources_sha256'].items():assert sha(Path(filename))==digest,filename
    for filename,digest in read(R/'small_events/tables/report_validation.json')['source_sha256'].items():
        assert sha(R/filename)==digest,filename
    registry=csv(E/'inputs/reuse_registry.csv').set_index('event_id')
    neighbors=read(E/'inputs/neighbors.json')
    reused=[]
    for row in registry.itertuples():
        reused.extend(json.loads(row.reused_files_sha256).items())
    def check_digest(item):
        filename,digest=item;assert sha(Path(filename))==digest,filename
    with ThreadPoolExecutor(12) as pool:list(pool.map(check_digest,reused))
    checked_reuse=len(reused)
    print('Reused-cache hashes passed:',checked_reuse,flush=True)
    rules=read(R/'inputs/analysis_quality_rules.json');outcomes={}
    for name,filename,win,nw,k in [('NW4','NW4_Mw_ML_catalog.csv',[10.1,20.1],4,7),('NW2','NW2_Mw_ML_catalog.csv',[5.1,10.1],2,3)]:
        d=csv(T/filename).set_index('event_id',drop=False);outcomes[name]=d
        assert len(d)==len(cohort) and d.index.is_unique and set(d.index)==set(cohort.event_id)
        assert d.primary_window_s.eq(win[0]).all() and d.secondary_window_s.eq(win[1]).all()
        assert d.multitaper_NW.eq(nw).all() and d.multitaper_K.eq(k).all()
        assert d.paired_A.eq(d.Mw_quality.eq('A')&d.ML_quality.eq('A')).all()
        assert d.paired_A_or_B.eq(d.Mw_quality.isin(['A','B'])&d.ML_quality.isin(['A','B'])).all()
        validate_catalog(d)
        assert d.loc[d.Mw.isna(),'Mw_quality'].eq('F').all()
        finite=d[d.Mw.notna()]
        assert finite.band_high_Hz.div(finite.band_low_Hz).ge(10-1e-9).all()
        for grade in ['A','B']:
            q=d[d.Mw_quality.eq(grade)];g=rules['quality_'+grade]
            assert q.n_stations_SNR.ge(g['minimum_stations']).all()
            assert q.common_window_stations.ge(g['minimum_common_window_stations']).all()
            assert q.plateau_observed.eq(True).all()
            for col,key in [('difference_Fourier','maximum_estimator_difference_Mw'),('difference_Brune','maximum_source_shape_difference_Mw'),
                            ('window_difference_common','maximum_common_station_window_difference_Mw'),('rms_log10','maximum_fit_RMS_log10')]:
                assert q[col].le(g[key]+1e-12).all(),(name,grade,col)
        q=d[d.ML_quality.eq('A')];assert q.ML_Sheen_n.ge(5).all() and q.ML_station_scatter.le(.35).all()
        def check_cache(case):
            eid=case['event_id'];base=Path(case['baseline_root']);root=base if name=='NW4' else Path(case['candidate_root'])
            raw=read(root/'events'/f'{eid}.json');local=read(base/'ml_events'/f'{eid}.json')
            assert raw.get('status') not in ['error','worker_error'],(name,eid,raw.get('error'))
            assert 'error' not in local,(eid,local.get('error'))
            if root==E:assert raw['config_sha256']=='extension-NW4-'+signature
            if root==E/'nw2_windows':assert raw['config_sha256']=='extension-NW2-'+signature
            if base==E:assert local['config_sha256']=='extension-ML-'+signature
            original=read(base/'events'/f'{eid}.json')
            assert neighbor_key(original.get('neighbors',[]))==neighbor_key(neighbors[eid]),eid
            assert neighbor_key(local.get('neighbors',[]))==neighbor_key(neighbors[eid]),eid
            if name=='NW2' and base==E and raw.get('status')=='no_retained_records':
                # A normal no-record outcome must be traceable to retained input QA.
                assert not original.get('fits'),eid
        with ThreadPoolExecutor(8) as pool:list(pool.map(check_cache,cohort.to_dict('records')))
        print(name,'raw cache, quality and source-velocity checks passed',flush=True)
    old,new=outcomes['NW4'],outcomes['NW2']
    assert close(old.ML_Sheen,new.ML_Sheen) and old.ML_quality.eq(new.ML_quality).all()
    canonical=csv(R/'tables/Mw_independent_ML_catalog.csv').set_index('event_id')
    reused_reference=old[old.ml.ge(2)&old.baseline_root.ne(str(E))]
    for column in ['Mw','ML_Sheen']:
        assert close(reused_reference[column],canonical.loc[reused_reference.index,column]),column
    for column in ['Mw_quality','ML_quality']:
        assert reused_reference[column].eq(canonical.loc[reused_reference.index,column]).all(),column
    previous=csv(R/'small_events/tables/nw2_window_Mw_ML_catalog.csv').set_index('event_id')
    retained_pilot=new[new.in_original_pilot]
    for column in ['Mw','ML_Sheen','difference_Fourier','difference_Brune','window_difference_common']:
        assert close(retained_pilot[column],previous.loc[retained_pilot.index,column]),column
    assert retained_pilot.Mw_quality.eq(previous.loc[retained_pilot.index,'Mw_quality']).all()
    progress=csv(T/'processing_progress.csv');assert len(progress)==3532 and progress.event_id.is_unique
    assert set(progress.event_id)==set(cohort.event_id) and not progress.baseline_status.eq('worker_error').any()
    primary=csv(T/'paired_A_MLge1p5.csv');expected=new[new.paired_A&new.ML_Sheen.ge(1.5)]
    assert set(primary.event_id)==set(expected.event_id) and primary.event_id.is_unique
    public=csv(T/'Mw_ML_catalog.csv');assert set(public.event_id)==set(cohort.event_id)
    assert close(public.set_index('event_id').loc[new.index,'Mw'],new.Mw)
    assert close(public.set_index('event_id').loc[new.index,'M0_Nm'],new.M0_Nm)
    sensor=csv(T/'small_event_HH_HG_pairs.csv')
    assert set(sensor.event_id).issubset(set(new.loc[new.ml.lt(2)&new.paired_A,'event_id']))
    if len(sensor):
        assert sensor.coordinate_gap_km.le(.1+1e-10).all() and sensor.n_frequency.ge(8).all()
        assert close(np.log10(sensor.HG_over_HH),sensor.log10_HG_over_HH)
        assert close(sensor.equivalent_delta_Mw,sensor.log10_HG_over_HH/1.5)
        assert not sensor.used_for_amplitude_correction.any()
    # Independent direct weighted solve verifies the reported primary line.
    X=matrix(primary.ML_Sheen.to_numpy(),'linear');y=primary.Mw.to_numpy();blocks=primary.family_block.to_numpy()
    w=np.array([1/sum(blocks==key) for key in blocks]);coef=np.linalg.solve(X.T@(X*w[:,None]),X.T@(y*w))
    regressions=pd.read_csv(T/'regressions.csv');row=regressions[regressions.selection.eq('NW2 primary')&regressions.model.eq('linear')&regressions.method.eq('family-balanced OLS')].iloc[0]
    assert close(coef,[row.a_at_ML2,row.b_at_ML2]) and row.n_events==len(primary)
    design=regressions[regressions.selection.eq('NW2 event-design sensitivity')&regressions.model.eq('linear')&regressions.method.eq('event design-weighted OLS')].iloc[0]
    slope,intercept=np.polyfit(primary.ML_Sheen-2,primary.Mw,1,w=np.sqrt(1/primary.selection_probability))
    assert close([slope,intercept],[design.b_at_ML2,design.a_at_ML2])
    predictions=csv(T/'family_heldout_predictions.csv')
    pp=predictions[predictions.selection.eq('NW2 primary')&predictions.model.eq('linear')].set_index('event_id')
    for key in np.unique(blocks):
        held=blocks==key;xx=X[~held];yy=y[~held];ww=w[~held]
        other=np.linalg.solve(xx.T@(xx*ww[:,None]),xx.T@(yy*ww))
        assert close(pp.loc[primary.loc[held,'event_id'],'prediction'],X[held]@other),key
    summary=read(T/'summary.json');assert summary['complete'] and summary['attempted']==3532
    shared=new[new.ml.ge(2)&new.Mw_quality.eq('A')&old.Mw_quality.eq('A')]
    delta=shared.Mw-old.loc[shared.index,'Mw'];reported=summary['transfer']['all_controls']
    assert len(delta)==reported['n'] and close(np.median(delta),reported['median_delta']) and close(np.sqrt(np.mean(delta**2)),reported['RMS_delta'])
    preserved=read(E/'inputs/protected_results.json')
    for filename,digest in preserved.items():assert sha(R/filename)==digest,filename
    # The report revision reads archived amplitudes and the saved historical fits.
    comparison=read(T/'regression_comparison_validation.json');assert comparison['passed']
    assert comparison['source_sha256']==sha(R/'code/compare_extension_regressions.py')
    for filename,digest in comparison['protected_inputs_sha256'].items():assert sha(R/filename)==digest,filename
    for filename,digest in comparison['outputs_sha256'].items():assert sha(T/filename)==digest,filename
    audit=csv(T/'ML_measurement_audit.csv').set_index('event_id')
    stations=csv(T/'ML_station_measurements.csv')
    assert set(audit.index)==set(new.index) and audit.index.is_unique
    assert close(audit.loc[new.index,'ML_measured'],new.ML_Sheen)
    assert audit.loc[new.index,'ML_grade'].eq(new.ML_quality).all()
    assert len(stations)==comparison['station_measurements']==46117
    assert not stations.duplicated(['event_id','station']).any()
    assert close(np.sqrt(stations.A_N_mm*stations.A_E_mm),stations.A_H_mm)
    station_ml=np.log10(stations.A_H_mm)+.5869*np.log10(stations.epicentral_km/100)+.001680*(stations.epicentral_km-100)+3
    assert close(station_ml,stations.ML_station)
    assert stations.SNR_N.ge(3).all() and stations.SNR_E.ge(3).all() and stations.epicentral_km.between(10,100).all()
    grouped=stations.groupby('event_id').ML_station
    supported=grouped.count().ge(3);medians=grouped.median()[supported]
    assert len(medians)==2222 and new.ML_quality.eq('A').sum()==1698
    assert close(medians,new.loc[medians.index,'ML_Sheen'])
    for case in cohort.to_dict('records'):
        path=Path(case['baseline_root'])/'ml_events'/f'{case["event_id"]}.json'
        assert sha(path)==audit.loc[case['event_id'],'cache_sha256'],path
    cm=pd.read_csv(T/'regression_comparison_models.csv');assert len(cm)==32 and cm.case_id.is_unique
    historical=csv(R/'tables/Mw_independent_ML_primary.csv')
    previous_fits=pd.read_csv(R/'tables/magnitude_followup_regressions.csv')
    training={'previous':historical,'expanded_NW4':old[old.paired_A&old.ML_Sheen.ge(1.5)],
        'larger_NW2':primary[primary.ml.ge(2)],'expanded_NW2':primary}
    overlap=historical.merge(new[['event_id','family_block']].reset_index(drop=True),on='event_id',suffixes=('_old','_new'))
    assert overlap.family_block_old.eq(overlap.family_block_new).all()
    cp=csv(T/'regression_comparison_predictions.csv')
    assert len(cp)==len(primary)*32 and not cp.duplicated(['case_id','event_id']).any()
    for case in cm.itertuples():
        train=training[case.cohort];column='ML_Sheen' if case.axis=='our' else 'ml'
        assert len(train)==case.n_events and train.family_block.nunique()==case.n_families
        pred=cp[cp.case_id.eq(case.case_id)].set_index('event_id').loc[primary.event_id]
        assert close(pred.Mw_target,primary.Mw) and close(pred.ML_measured,primary.ML_Sheen) and close(pred.ML_KMA,primary.ml)
        if case.cohort=='previous':
            saved=previous_fits[previous_fits.case_id.eq(case.reused_previous_case)].iloc[0]
            a,b,c=saved.a_at_ML3,saved.slope_at_ML3,saved.quadratic_c
            assert close([case.a,case.b,case.c],[a-b+c,b-2*c,c])
        if case.method!='OLS':continue
        x=train[column].to_numpy()-2;yy=train.Mw.to_numpy();families=train.family_block.to_numpy()
        xx=np.polynomial.polynomial.polyvander(x,case.degree)
        ww=train.groupby('family_block').event_id.transform('size').rdiv(1).to_numpy()
        direct=np.linalg.solve(xx.T@(xx*ww[:,None]),xx.T@(yy*ww))
        assert close(direct,[case.a,case.b,case.c][:case.degree+1])
        tx=np.polynomial.polynomial.polyvander(primary[column].to_numpy()-2,case.degree)
        target_families=primary.family_block.to_numpy()
        for family in np.unique(target_families):
            keep=families!=family;held=target_families==family
            hX=xx[keep];hy=yy[keep];hw=ww[keep]
            coef=np.linalg.solve(hX.T@(hX*hw[:,None]),hX.T@(hy*hw))
            assert close(pred.loc[primary.loc[held,'event_id'],'predicted'],tx[held]@coef)
    objective=read(T/'fit_objective_validation.json');assert objective['passed'] and objective['n_models']==80
    assert objective['source_sha256']==sha(R/'code/audit_extension_fit_objectives.py')
    assert objective['engine_sha256']==sha(R/'code/potency_regression.py')
    for filename,digest in objective['protected_inputs_sha256'].items():assert sha(R/filename)==digest,filename
    for filename,digest in objective['outputs_sha256'].items():assert sha(T/filename)==digest,filename
    om=csv(T/'fit_objective_models.csv');op=csv(T/'fit_objective_predictions.csv');ob=csv(T/'fit_objective_bins.csv')
    ow=csv(T/'fit_objective_weights.csv')
    assert len(om)==80 and om.case_id.is_unique and not op.duplicated(['case_id','target_set','event_id']).any()
    def independent_bins(x,y):
        edges=np.linspace(min(x),max(x),51);idx=np.clip(np.searchsorted(edges,x,side='right')-1,0,49)
        result=[]
        for k in range(50):
            select=idx==k
            if not select.any():continue
            select=select&(np.abs(y-np.median(y[select]))<=.5)
            if select.any():result.append([np.median(x[select]),np.median(y[select])])
        return np.asarray(result)
    for name,train in training.items():
        weights=ow[ow.cohort.eq(name)].set_index('event_id').loc[train.event_id]
        expected=train.groupby('family_block').event_id.transform('size').rdiv(1).to_numpy()
        assert close(weights.family_weight,expected) and close(weights.normalized_family_weight,expected/expected.sum())
    held_families=sorted(set(primary.loc[primary.ML_Sheen.gt(4),'family_block'])|
        set(primary[primary.confirmatory_small].sort_values('event_id').family_block.head(6)))
    for case in om.itertuples():
        train=training[case.cohort];column='ML_Sheen' if case.axis=='our' else 'ml'
        shared=op[op.case_id.eq(case.case_id)&op.target_set.eq('shared_NW2')].set_index('event_id').loc[primary.event_id]
        native=op[op.case_id.eq(case.case_id)&op.target_set.eq('native')]
        assert len(native)==len(train) and set(native.event_id)==set(train.event_id)
        assert close(shared.Mw,primary.Mw) and close(shared.ML_measured,primary.ML_Sheen)
        x=train[column].to_numpy();y=train.Mw.to_numpy();families=train.family_block.to_numpy()
        if case.objective.startswith('bin_'):
            med=independent_bins(x,y);saved=ob[ob.case_id.eq(case.case_id)].sort_values('bin_index')
            assert close(med,saved[['ML_median','Mw_median']])
        if case.objective.endswith('ODR'):continue
        if case.objective=='bin_OLS':xx,yy=med[:,0],med[:,1];ww=np.ones(len(med))
        else:
            xx,yy=x,y;ww=np.ones(len(train))
            if case.objective=='family_OLS':ww=train.groupby('family_block').event_id.transform('size').rdiv(1).to_numpy()
        design=np.polynomial.polynomial.polyvander(xx-2,case.degree)
        direct=np.linalg.lstsq(design*np.sqrt(ww[:,None]),yy*np.sqrt(ww),rcond=None)[0]
        assert close(direct,[case.a,case.b,case.c][:case.degree+1])
        if case.objective!='bin_OLS':continue
        for family in held_families:
            keep=families!=family;held=primary.family_block.eq(family)
            med=independent_bins(x[keep],y[keep]);xx=np.polynomial.polynomial.polyvander(med[:,0]-2,case.degree)
            coef=np.linalg.lstsq(xx,med[:,1],rcond=None)[0]
            tx=np.polynomial.polynomial.polyvander(primary.loc[held,column]-2,case.degree)
            assert close(shared.loc[primary.loc[held,'event_id'],'predicted'],tx@coef)
    result=dict(passed=True,scope=config['scope'],selected_events=3532,small_events=2890,reference_events=642,
        all_attempts_accounted_for=True,sampling_replay_passed=True,identical_ML_across_estimators=True,
        all_reused_reference_values_and_grades_reproduced=True,all_original_pilot_values_reproduced=True,
        raw_cache_and_reuse_hashes_passed=True,reused_files_checked=checked_reuse,
        all_known_neighbor_policy_passed=True,unchanged_quality_gates_passed=True,
        source_Vs_and_C9p05_passed=True,primary_regression_and_whole_family_holdout_reproduced=True,
        independent_ML_station_amplitudes_replayed=True,station_ML_measurements_checked=len(stations),
        previous_and_expanded_regressions_reproduced=True,comparison_models_checked=len(cm),
        comparison_validation='small_events/extension/tables/regression_comparison_validation.json',
        comparison_validation_sha256=sha(T/'regression_comparison_validation.json'),
        fitting_objective_audit_passed=True,fitting_objective_models_checked=len(om),
        independent_bin_medians_and_large_family_holdouts_passed=True,
        fitting_objective_validation='small_events/extension/tables/fit_objective_validation.json',
        fitting_objective_validation_sha256=sha(T/'fit_objective_validation.json'),
        transfer_passed=summary['transfer']['passed'],adopted_and_pilot_results_preserved=True,
        summary_sha256=sha(T/'summary.json'),protected_results_sha256=preserved,
        sources_sha256={str(p.relative_to(R)):sha(p) for p in [R/'code/prepare_small_event_extension.py',R/'code/run_small_event_extension.py',R/'code/analyze_small_event_extension.py',Path(__file__)]})
    report=T/'report_validation.json'
    if report.exists():
        check=read(report);assert check['passed'];result['report_validation']='small_events/extension/tables/report_validation.json'
        assert check['source_sha256']==sha(R/'code/report_small_event_extension.py')
        assert check['summary_sha256']==sha(T/'summary.json')
        assert check['comparison_validation_sha256']==sha(T/'regression_comparison_validation.json')
        assert check['objective_validation_sha256']==sha(T/'fit_objective_validation.json')
        result['reports']=check['reports']
        for filename,record in check['reports'].items():
            path=R/filename;assert sha(path)==record['sha256']
            with fitz.open(path) as doc:assert len(doc)==record['pages']
    potency=read(T/'potency_validation.json');assert potency['passed']
    assert potency['source_sha256']==sha(R/'code/validate_extension_potency.py')
    assert potency['report_validation_sha256']==sha(T/'report_validation.json')
    assert potency['analysis_validation_sha256']==sha(T/'potency_analysis_validation.json')
    assert potency['controls_validation_sha256']==sha(T/'potency_controls_validation.json')
    assert potency['reports']==result['reports']
    result.update(extension_potency_passed=True,extension_potency_validation='small_events/extension/tables/potency_validation.json',
        extension_potency_validation_sha256=sha(T/'potency_validation.json'),extension_finite_Mp=potency['finite_potency'],
        extension_joint_magnitude_cases=potency['regression_cases'],previous_joint_magnitude_cases_reproduced=potency['previous_native_cases_reproduced'])
    (T/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['protected_results_sha256','sources_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
