"""Normalization, published-catalog and historical-parity controls for Mp."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,multiprocessing,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from potency_regression import run_case

R=Path(__file__).resolve().parents[1];T=R/'small_events/extension/tables'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(p):return pd.read_csv(p,dtype={'event_id':str})


def main():
    audit=json.loads((T/'potency_analysis_validation.json').read_text());assert audit['passed']
    for name,digest in audit['outputs_sha256'].items():assert sha(T/name)==digest,name
    d=csv(T/'potency_event_catalog.csv');g=d[d.paired_A&d.ML_Sheen.ge(1.5)]
    members=csv(T/'magnitude_scaling_members.csv');normal=[];external=[]
    for model in ['Kim1983','Kim2011']:
        for col,label in [('Mp_Liu_','Liu'),('Mp36_C9p05_','36 GPa, C=9.05'),('Mpo28_C9p10_','28 GPa, C=9.10'),('Mpo28_C9p05_','28 GPa, C=9.05')]:
            delta=g[col+model]-g['Mp_Liu_'+model]
            assert delta.max()-delta.min()<1e-11
            normal.append(dict(model=model,definition=label,offset_from_Liu=float(delta.median())))
        for cohort,part in members.groupby('cohort'):
            for axis,x in [('our','ML_Sheen'),('KMA','ml')]:
                delta=part['Mp_Liu_'+model]-(.04*part[x]**2+.556*part[x]+1.133)
                external.append(dict(cohort=cohort,model=model,axis=axis,n=len(part),median=float(delta.median()),RMS=float(np.sqrt(np.mean(delta*delta)))))
    pd.DataFrame(normal).to_csv(T/'potency_normalization.csv',index=False)
    pd.DataFrame(external).to_csv(T/'potency_Liu_comparison.csv',index=False)
    h=csv(T/'potency_Hong_matches.csv');h=h[h.paired_A&h.ML_Sheen.ge(1.5)].copy()
    full=csv(R/'inputs/KR_Mw/Events_KP_2017_2022_Mw.csv').rename(columns={'evid':'event_id'})
    full['event_id']=full.event_id.astype(str);full['ml']=full.ml_KMA;full['family_block']='Hong_event_'+full.event_id
    for src,label in [('mw_spectra','spectral'),('mw_TD','TD')]:
        full['Mp_Hong_'+label]=(1.5*full[src]+9.05-np.log10(2700*3610**2)+1.4563)/1.5
    tasks=[]
    def add(frame,label,x,y,method,weight):
        spec=dict(case_id=f'Hong_{len(tasks):03d}',selection=label,x=x,y=y,method=method,weighting=weight,
            degree=1,n_bins=50,minimum=1,bootstrap_reps=1000,do_cv=False,
            block_policy='Whole matched event family' if weight=='family' else 'Event resampling; full-catalog family membership unavailable')
        tasks.append((spec,frame[x].to_numpy(),frame[y].to_numpy(),frame.family_block.to_numpy(),frame.event_id.to_numpy()))
    for method in ['OLS','ODR']:
        for model in ['Kim1983','Kim2011']:
            for quantity,y in [('Mw','Mw' if model=='Kim1983' else 'Mw_source_Vs_Kim2011'),('Mp','Mp_Liu_'+model)]:
                for label,x in [('spectral','mw_spectra'),('TD','mw_TD')]:
                    if quantity=='Mp':x='Mp_Hong_'+label
                    add(h,'Ours versus Hong: paired A',x,y,method,'family')
        for quantity in ['Mw','Mp']:
            for label,y in [('spectral','mw_spectra'),('TD','mw_TD')]:
                if quantity=='Mp':y='Mp_Hong_'+label
                for x in ['ML_Sheen','ml']:add(h,'Hong versus ML: paired A',x,y,method,'family')
                add(full,'Hong versus ML: full published catalog','ml',y,method,'event')
                limited=full[full.ml.between(h.ml.min(),h.ml.max())]
                add(limited,'Hong versus ML: matched magnitude range','ml',y,method,'event')
    rows=[];bootstrap={}
    with ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('fork')) as pool:
        for row,boot,_,_,_ in pool.map(run_case,tasks):rows.append(row);bootstrap[row['case_id']]=boot
    pd.DataFrame(rows).to_csv(T/'potency_Hong_regressions.csv',index=False)
    np.savez_compressed(T/'potency_Hong_bootstrap.npz',**bootstrap)
    # The same accepted horizontal peaks as the earlier regional-ML control.
    stations=csv(T/'ML_station_measurements.csv');stations['depth_km']=stations.event_id.map(d.set_index('event_id').z_central)
    r=stations.epicentral_km.to_numpy();rh=np.hypot(r,stations.depth_km);amp=stations.A_H_mm.to_numpy()
    original=np.log10(amp)+.5869*np.log10(r/100)+.001680*(r-100)+3
    assert np.allclose(original,stations.ML_station,atol=1e-11,rtol=0)
    stations['ML_HB87_same_WA2080']=np.log10(amp)+1.110*np.log10(rh/100)+.00189*(rh-100)+3
    stations['ML_HB87_WA2800_gain_control']=stations.ML_HB87_same_WA2080+np.log10(2800/2080)
    alt=stations.groupby('event_id')[['ML_HB87_same_WA2080','ML_HB87_WA2800_gain_control']].median()
    events=d.merge(alt,on='event_id',how='left',validate='one_to_one');events.loc[events.ML_Sheen.isna(),alt.columns]=np.nan
    events.to_csv(T/'potency_ML_definition_controls.csv',index=False)
    paired=events[events.paired_A&events.ML_Sheen.ge(1.5)];regional=[]
    for x in alt.columns:
        for y in ['Mp_Liu_Kim1983','Mp_Liu_Kim2011','Mw','Mw_source_Vs_Kim2011']:
            for method in ['OLS','BinnedOLS']:
                for degree in [1,2]:
                    spec=dict(case_id=f'regional_{len(regional):03d}',selection='Same WA amplitudes',x=x,y=y,method=method,
                        weighting='family',degree=degree,n_bins=50,minimum=1,bootstrap_reps=1000,do_cv=True,block_policy='Whole event family')
                    regional.append((spec,paired[x].to_numpy(),paired[y].to_numpy(),paired.family_block.to_numpy(),paired.event_id.to_numpy()))
    rr=[];rp=[];rb={}
    with ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('fork')) as pool:
        for row,boot,pred,_,_ in pool.map(run_case,regional):rr.append(row);rp+=pred;rb[row['case_id']]=boot
    regional_fits=pd.DataFrame(rr);regional_fits.to_csv(T/'potency_regional_regressions.csv',index=False)
    pd.DataFrame(rp).to_csv(T/'potency_regional_predictions.csv',index=False);np.savez_compressed(T/'potency_regional_bootstrap.npz',**rb)
    for (y,method,degree),part in regional_fits.groupby(['y','method','degree']):
        assert np.ptp(part.quadratic_c)<1e-10 and np.ptp(part.CV_RMS_families)<1e-10
    definitions=[]
    for x in ['ML_Sheen','ml']+list(alt.columns):
        delta=paired.Mp_Liu_Kim1983-(.04*paired[x]**2+.556*paired[x]+1.133)
        definitions.append(dict(ML_definition=x,n=len(paired),median_ML_minus_ours=float((paired[x]-paired.ML_Sheen).median()),
            median_Mp_minus_Liu=float(delta.median()),RMS_to_Liu=float(np.sqrt(np.mean(delta*delta)))))
    regional_summary=dict(scope='Change only attenuation/distance and the illustrative gain on identical saved N/E WA peaks and station sets',
        original_station_ML_error=float(max(abs(original-stations.ML_station))),n_station_readings=len(stations),
        gain_shift=float(np.log10(2800/2080)),gain_scope='2800/2080 is an illustrative gain control, not a claim about the observational Liu catalog gain',
        paired_events=len(paired),Hong_same_ML_range=len(limited),definitions=definitions)
    (T/'potency_regional_summary.json').write_text(json.dumps(regional_summary,indent=2)+'\n')
    parity=[('Physical definitions','Original P0 and Liu Mp; both velocity models; 28/36-GPa normalizations'),
        ('Magnitude regression','Every one of the earlier 128 Mw/Mp specifications and native-target scores reproduced'),
        ('Magnitude range','Previous 357, expanded NW4 582, larger NW2 375, expanded NW2 676'),
        ('Fit objectives','Family/event OLS and ODR; bin OLS; added bin ODR; 10/20/50-bin and support controls'),
        ('Uncertainty and prediction','Whole-family bootstrap; whole-family holdouts; fit/CV errors separated'),
        ('Large events','Same six measured-ML>4 events in five families; extrapolation flagged'),
        ('Depth dependence','Original 84 expanded-cohort depth controls plus 24 larger-only controls'),
        ('External comparisons','Unshifted Liu curve; Hong spectral/time-domain Mw and converted Mp'),
        ('Regional ML definitions','Same WA peaks: Sheen versus Hutton-Boore attenuation/distance; 2080/2800 gain control; matched Hong ML-range selection'),
        ('Catalog accounting','All 3532 attempts retained; Mp missing wherever Mw is missing'),
        ('Scope','Scalar source potency; effective-path-rigidity Mpv remains outside this spectral analysis')]
    pd.DataFrame(parity,columns=['topic','extension_coverage']).to_csv(T/'potency_report_coverage.csv',index=False)
    outputs=['potency_normalization.csv','potency_Liu_comparison.csv','potency_Hong_regressions.csv','potency_Hong_bootstrap.npz','potency_report_coverage.csv',
        'potency_ML_definition_controls.csv','potency_regional_regressions.csv','potency_regional_predictions.csv','potency_regional_bootstrap.npz','potency_regional_summary.json']
    result=dict(passed=True,analysis_validation_sha256=sha(T/'potency_analysis_validation.json'),
        source_sha256=sha(Path(__file__)),matched_Hong_paired_A=len(h),Hong_regressions=len(rows),regional_ML_regressions=len(rr),
        station_amplitudes_sha256=sha(T/'ML_station_measurements.csv'),original_regional_engine_sha256=sha(R/'code/potency_regional_controls.py'),
        outputs_sha256={name:sha(T/name) for name in outputs})
    (T/'potency_controls_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
