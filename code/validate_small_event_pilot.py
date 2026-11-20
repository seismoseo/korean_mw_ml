"""Check sampling, all attempted cases, transfer controls and preserved results.

These checks establish computational consistency, not unbiased population
coverage or absolute accuracy of low-SNR spectral moment estimates.
"""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import fitz
import numpy as np
import pandas as pd
from prepare_small_event_pilot import strata
from source_velocity import validate_catalog, source_vs, log_moment_coefficient

R=Path(__file__).resolve().parents[1];P=R/'small_events';T=P/'tables'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def csv(path):return pd.read_csv(path,dtype={'event_id':str})
def close(a,b):return np.allclose(a,b,atol=1e-11,rtol=0,equal_nan=True)


def main():
    config=read(P/'inputs/pilot_config.json');cohort=csv(P/'inputs/pilot_cohort.csv')
    parent=csv(P/'inputs/eligible_small_events.csv');depth=csv(Path(config['source_catalog']))
    assert sha(Path(config['source_catalog']))==config['source_catalog_sha256']
    assert sha(P/'inputs/pilot_cohort.csv')==config['pilot_cohort_sha256']
    assert sha(R/'inputs/study_config.json')==config['frozen_reference_configuration_sha256']
    expected=depth[depth.ml.ge(.5)&depth.ml.lt(2)&depth.region.ne('North Korea (land)')]
    assert set(parent.event_id)==set(expected.event_id) and len(parent)==9325
    assert parent.grade.isin(['G2','G3']).all() and parent.event_id.is_unique
    assert len(cohort)==180 and cohort.event_id.is_unique and cohort.input_available.all()
    small=cohort[cohort.pilot_role.eq('small')]
    assert len(small)==162 and len(cohort[cohort.pilot_role.eq('control')])==18
    keys=['magnitude_stratum','depth_stratum','era_stratum']
    assert len(small.groupby(keys))==27 and small.groupby(keys).size().eq(6).all()
    assert small.groupby('magnitude_stratum').size().eq(54).all()
    # Replay the fixed draw without writing any input or measuring waveforms.
    rng=np.random.default_rng(config['seed']);selected=[]
    for _,q in parent.groupby(keys,sort=True):
        q=q.sort_values('event_id');chosen=q.iloc[np.sort(rng.choice(len(q),6,replace=False))]
        selected.extend(chosen.event_id)
        assert close(small.set_index('event_id').loc[chosen.event_id,'selection_probability'],np.full(6,6/len(q)))
    assert set(selected)==set(small.event_id)
    control_parent=strata(depth[depth.ml.ge(2)&depth.ml.lt(2.5)&depth.region.ne('North Korea (land)')])
    selected=[]
    for _,q in control_parent.groupby(['depth_stratum','era_stratum'],sort=True):
        q=q.sort_values('event_id');selected.extend(q.iloc[np.sort(rng.choice(len(q),2,replace=False))].event_id)
    assert set(selected)==set(cohort[cohort.pilot_role.eq('control')].event_id)

    provenance=read(P/'inputs/processing_provenance.json');signature=provenance['signature']
    assert provenance['cohort_sha256']==sha(P/'inputs/pilot_cohort.csv')
    for filename,digest in provenance['processing_sources'].items():assert sha(Path(filename))==digest,filename
    assert sha(Path(provenance['neighbor_catalog']))==provenance['neighbor_catalog_sha256']
    assert provenance['known_neighbors_all_magnitudes']
    rules=read(R/'inputs/analysis_quality_rules.json');results={};statistics={}
    schemes=[('original',P,'pilot_Mw_ML_catalog.csv',[10.1,20.1],4,7,.5),
             ('short_NW4',P/'short_windows','short_window_Mw_ML_catalog.csv',[3.1,5.1],4,7,1.5),
             ('short_NW2',P/'nw2_windows','nw2_window_Mw_ML_catalog.csv',[5.1,10.1],2,3,.5)]
    for name,root,filename,windows,nw,k,low in schemes:
        d=csv(T/filename).set_index('event_id',drop=False);results[name]=d
        assert len(d)==180 and d.index.is_unique and set(d.index)==set(cohort.event_id)
        assert d.primary_window_s.eq(windows[0]).all() and d.secondary_window_s.eq(windows[1]).all()
        assert d.multitaper_NW.eq(nw).all() and d.multitaper_K.eq(k).all()
        assert low>=nw/(windows[0]+.01)
        if name=='original':mw_signature='small-Mw-'+signature
        else:
            actual=read(root/'config.json');mw_signature=actual['signature']
            assert actual['windows_s']==windows and actual['band_Hz']==[low,20]
            driver='small_event_nw2_windows.py' if nw==2 else 'small_event_short_windows.py'
            paths=[R/'code'/driver,R/'code/instrument_processing.py',R/'code/spectra.py',R/'code/source_velocity.py',
                   P/'inputs/pilot_config.json',P/'inputs/processing_provenance.json']
            assert hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()==mw_signature
        validate_catalog(d)
        for eid in d.index:
            raw=read(root/'events'/f'{eid}.json');ml=read(P/'ml_events'/f'{eid}.json')
            assert raw['config_sha256']==mw_signature,(name,eid)
            assert ml['config_sha256']=='small-ML-'+signature,eid
            assert raw['status'] not in ['error','worker_error'],(name,eid,raw['status'])
            assert d.loc[eid,'Mw_quality'] in ['A','B','C','F']
        finite=d[d.Mw.notna()]
        assert np.allclose(finite.Mw,(np.log10(finite.omega_r_m2s)+log_moment_coefficient(finite.z_central.to_numpy())-9.05)/1.5,atol=1e-11,rtol=0)
        assert close(finite.source_Vs_m_s,source_vs(finite.z_central.to_numpy()))
        assert finite.band_high_Hz.div(finite.band_low_Hz).ge(10-1e-10).all()
        assert d.loc[d.Mw.isna(),'Mw_quality'].eq('F').all()
        for label in ['A','B']:
            q=d[d.Mw_quality.eq(label)];rule=rules['quality_'+label]
            assert q.n_stations_SNR.ge(rule['minimum_stations']).all()
            assert q.common_window_stations.ge(rule['minimum_common_window_stations']).all()
            assert q.plateau_observed.eq(True).all()
            for col,key in [('difference_Fourier','maximum_estimator_difference_Mw'),('difference_Brune','maximum_source_shape_difference_Mw'),
                            ('window_difference_common','maximum_common_station_window_difference_Mw'),('rms_log10','maximum_fit_RMS_log10')]:
                assert q[col].le(rule[key]+1e-12).all(),(name,label,col)
        ml_a=d[d.ML_quality.eq('A')]
        assert ml_a.ML_Sheen_n.ge(5).all() and ml_a.ML_station_scatter.le(.35).all()
        assert d.paired_A.eq(d.Mw_quality.eq('A')&d.ML_quality.eq('A')).all()
        assert d.paired_A_or_B.eq(d.Mw_quality.isin(['A','B'])&d.ML_quality.isin(['A','B'])).all()
        assert close(d.ML_Sheen,results['original'].loc[d.index,'ML_Sheen'])
        assert d.ML_quality.eq(results['original'].loc[d.index,'ML_quality']).all()
        s=d[d.pilot_role.eq('small')];pairs=s[s.paired_A]
        statistics[name]=dict(attempted=len(d),small=len(s),finite_small_Mw=int(s.Mw.notna().sum()),
            Mw_grade_counts=s.Mw_quality.value_counts().to_dict(),small_pairs_A=len(pairs),small_pairs_AB=int(s.paired_A_or_B.sum()),
            measured_ML_range_A=[float(pairs.ML_Sheen.min()),float(pairs.ML_Sheen.max())])
        if name!='original':
            base=results['original'];c=d[d.pilot_role.eq('control')&d.Mw_quality.eq('A')&base.Mw_quality.eq('A')]
            delta=c.Mw-base.loc[c.index,'Mw'];median=float(np.median(delta));rms=float(np.sqrt(np.mean(delta**2)))
            prefix='nw2_window' if nw==2 else 'short_window'
            plan=read(P/'inputs'/('nw2_assessment_plan.json' if nw==2 else 'short_window_assessment_plan.json'))['transfer_gate']
            gate=len(c)>=plan['minimum_shared_Mw_A_controls'] and abs(median)<=plan['maximum_absolute_median_delta_Mw'] and rms<=plan['maximum_RMS_delta_Mw']
            summary=read(T/(prefix+'_summary.json'))['transfer_controls']
            assert summary['n_shared_Mw_A']==len(c) and close(summary['median_delta_Mw'],median) and close(summary['RMS_delta_Mw'],rms)
            assert summary['passed']==gate and gate
            candidate=csv(T/(prefix+'_candidate_pairs.csv'))
            assert set(candidate.event_id)==set(pairs.index) and len(candidate)==len(pairs)
            assert close(candidate.Mw,pairs.loc[candidate.event_id,'Mw'])
            statistics[name]['transfer_controls']=dict(shared_Mw_A=len(c),median_delta_Mw=median,RMS_delta_Mw=rms,passed=gate)

    base=results['original'];canonical=csv(R/'tables/Mw_independent_ML_catalog.csv').set_index('event_id')
    controls=base[base.pilot_role.eq('control')];old=canonical.loc[controls.index]
    assert close(controls.Mw,old.Mw) and close(controls.ML_Sheen,old.ML_Sheen)
    assert controls.Mw_quality.eq(old.Mw_quality).all() and controls.ML_quality.eq(old.ML_quality).all()
    assert int(controls.Mw.notna().sum())==17 and int(controls.ML_Sheen.notna().sum())==17
    reference=csv(R/'tables/Mw_independent_ML_primary.csv')
    for filename,flag,count in [('combined_paired_A.csv','paired_A',364),('combined_paired_AB.csv','paired_A_or_B',377)]:
        combined=csv(T/filename);s=base[base.pilot_role.eq('small')&base[flag]]
        assert len(combined)==count and combined.event_id.is_unique
        assert set(combined.event_id)==set(reference.event_id)|set(s.index)
        assert not combined.target_holdout.any() and not combined.event_id.eq('20261117141017').any()
    model=csv(T/'exploratory_regressions_AB.csv');ref=model[model.selection.eq('Reference only')&model.model.eq('linear')&model.method.eq('family-balanced OLS')].iloc[0]
    canonical_fit=pd.read_csv(R/'tables/independent_ML_regressions.csv').query('selection == "Independent ML"').iloc[0]
    assert close(ref.b_at_ML2,canonical_fit.slope)
    assert not read(T/'pilot_summary.json')['regressions']['fit_extension']
    protected=read(P/'inputs/protected_results.json')
    for filename,digest in protected.items():assert sha(R/filename)==digest,filename
    report=read(T/'report_validation.json');assert report['passed'] and report['figure_font_and_page_bounds_passed']
    for filename,digest in report['source_sha256'].items():assert sha(R/filename)==digest,filename
    for filename,pages,key in [('Small_event_Mw_ML_pilot.pdf',7,'report_sha256'),('Small_event_spectral_atlas.pdf',45,'atlas_sha256')]:
        path=P/'reports'/filename
        with fitz.open(path) as doc:assert len(doc)==pages
        assert sha(path)==report[key]
    outcome=dict(passed=True,scope='Fixed-cohort feasibility and diagnostic window/taper experiments; no revised primary relation',
        eligible_events=9325,small_attempted=162,control_attempted=18,selection_replay_passed=True,
        every_attempt_accounted_for=True,all_cache_signatures_current=True,unchanged_ML_across_schemes=True,
        original_control_values_and_grades_reproduced=True,source_depth_Vs_and_C9p05_passed=True,
        all_A_B_gates_and_candidate_exports_passed=True,all_original_adopted_results_preserved=True,
        schemes=statistics,report_validation='small_events/tables/report_validation.json',
        report_sha256=report['report_sha256'],atlas_sha256=report['atlas_sha256'],
        protected_results_sha256=protected,validation_source_sha256=sha(Path(__file__)),
        limitations=['Conditional on detection and G2/G3 depths; balanced pilot fractions are not national detection rates.',
                    'Station bootstrap intervals are conditional; absolute source/path/site/response errors are excluded.',
                    'Transfer-control agreement does not establish absolute low-SNR accuracy or an unbiased conversion.'])
    (T/'validation.json').write_text(json.dumps(outcome,indent=2)+'\n')
    print(json.dumps({k:v for k,v in outcome.items() if k!='protected_results_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
