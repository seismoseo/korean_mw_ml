"""Apply the existing quality rules to every attempted small-event pilot case."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from spectra import fit
from source_velocity import log_moment_coefficient, normalize_fit

R=Path(__file__).resolve().parents[1];P=R/'small_events'
RULES=json.loads((R/'inputs/analysis_quality_rules.json').read_text())


def stack(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(a,axis=0)


def event_result(case,waveform_root=P,primary_window=10.1,secondary_window=20.1):
    eid=case['event_id'];path=waveform_root/'events'/f'{eid}.json';mpath=P/'ml_events'/f'{eid}.json'
    raw=json.loads(path.read_text())
    ml=json.loads(mpath.read_text()) if mpath.exists() else dict(summary={})
    row=dict(case,processing_status=raw.get('status'),raw_usable_triplets=raw.get('raw_usable_triplets',0),
             Mw=np.nan,Mw_quality='F',Mw_reasons='No usable primary spectral fit',
             n_stations_SNR=0,common_window_stations=0,window_difference_common=np.nan,
             difference_Fourier=np.nan,difference_Brune=np.nan,
             ML_Sheen=np.nan,ML_quality='F',ML_Sheen_n=0)
    row.update(ml.get('summary',{}))
    row.update(primary_window_s=primary_window,secondary_window_s=secondary_window)
    row['rejection_counts']=json.dumps(raw.get('rejections',{}),sort_keys=True)
    fits=[normalize_fit(f,case['z_central']) for f in raw.get('fits',[])]
    lookup={(f['window_s'],f['sensor_set'],f['method'],f['shape']):f for f in fits}
    primary=lookup.get((primary_window,'ALL','Multitaper','Boatwright'))
    row['primary_fit_status']=primary.get('status') if primary else 'absent'
    if primary is not None and np.isfinite(primary.get('Mw',np.nan)):
        row.update(primary)
        with np.load(waveform_root/'spectra'/f'{eid}.npz') as a:
            tag1=f'T{primary_window:g}';tag2=f'T{secondary_window:g}'
            f=a['frequency_Hz'];a10=a[tag1];names=a[tag1+'_stations'].astype(str);sensors=a[tag1+'_sensor'].astype(str)
            row['n_stations_SNR']=int((np.isfinite(a10[:,0,:]).sum(axis=1)>=20).sum())
            for sensor in ['HH','EL','HG']:row['n_'+sensor]=int((sensors==sensor).sum())
            for method,shape,label in [('Direct Fourier','Boatwright','Fourier'),('Multitaper','Brune','Brune')]:
                alt=lookup.get((primary_window,'ALL',method,shape),{})
                row['Mw_'+label]=alt.get('Mw',np.nan)
                row['difference_'+label]=abs(row['Mw_'+label]-row['Mw'])
            row['Mw_velocity']=lookup.get((primary_window,'HH_EL','Multitaper','Boatwright'),{}).get('Mw',np.nan)
            row['Mw_HG']=lookup.get((primary_window,'HG','Multitaper','Boatwright'),{}).get('Mw',np.nan)
            row['Mw_secondary_window']=lookup.get((secondary_window,'ALL','Multitaper','Boatwright'),{}).get('Mw',np.nan)
            if tag2 in a:
                a20=a[tag2];nn=a[tag2+'_stations'].astype(str);ss=a[tag2+'_sensor'].astype(str)
                k10=list(zip(names,sensors));k20=list(zip(nn,ss))
                useful10={k10[i] for i in range(len(k10)) if np.isfinite(a10[i,0]).sum()>=20}
                useful20={k20[i] for i in range(len(k20)) if np.isfinite(a20[i,0]).sum()>=20}
                common=sorted(useful10&useful20);row['common_window_stations']=len(common)
                if len(common)>=5:
                    vals=[]
                    for array,keys in [(a10,k10),(a20,k20)]:
                        ids=[keys.index(key) for key in common];selected=array[ids]
                        good=np.isfinite(selected[:,0]).sum(axis=0)>=max(3,.3*len(selected))
                        alt,_=fit(f,stack(selected[:,0]),good,depth_km=case['z_central'])
                        vals.append(alt.get('Mw',np.nan))
                    row['Mw_primary_common'],row['Mw_secondary_common']=vals
                    row['window_difference_common']=abs(vals[1]-vals[0])
            row['Mw_quality']='C';reasons=[]
            for label in ['A','B']:
                rule=RULES['quality_'+label];fails=[]
                if row['n_stations_SNR']<rule['minimum_stations']:fails.append('few_stations')
                if not row['plateau_observed']:fails.append('plateau_extrapolated')
                if row['common_window_stations']<rule['minimum_common_window_stations']:fails.append('few_common_window_stations')
                for col,cut,name in [('difference_Fourier','maximum_estimator_difference_Mw','estimator_sensitivity'),
                    ('difference_Brune','maximum_source_shape_difference_Mw','source_shape_sensitivity'),
                    ('window_difference_common','maximum_common_station_window_difference_Mw','window_sensitivity')]:
                    if not np.isfinite(row[col]) or row[col]>rule[cut]:fails.append(name)
                if row['rms_log10']>rule['maximum_fit_RMS_log10']:fails.append('poor_source_fit')
                if not fails:row['Mw_quality']=label;break
                if label=='A':reasons=fails
            row['Mw_reasons']='' if row['Mw_quality']=='A' else ';'.join(reasons)
            if row['Mw_quality'] in ['A','B']:
                good=np.isfinite(a10[:,0]).sum(axis=0)>=max(3,.3*len(a10))
                phi=-.5*np.log10(1+(f/row['fc_Hz'])**(2*row['falloff']))
                rng=np.random.default_rng(int(eid[-6:]));ids=rng.integers(len(a10),size=(300,len(a10)))
                av=stack(a10[ids,0,:].transpose(1,0,2))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    omega=np.nanmean((av-phi)[:,good],axis=1)
                boots=(omega+log_moment_coefficient(case['z_central'])-9.05)/1.5
                row['Mw_station_bootstrap_lo'],row['Mw_station_bootstrap_hi']=np.quantile(boots,[.025,.975])
            if not any(np.isfinite(a10[i,0]).sum()>=20 for i in range(len(a10))):raise AssertionError(eid)
    archive=waveform_root/'spectra'/f'{eid}.npz'
    if archive.exists():
        with np.load(archive) as a:
            tag=f'T{primary_window:g}'
            if tag in a:
                spectrum=a[tag];f=a['frequency_Hz']
                row['n_stations_SNR']=int((np.isfinite(spectrum[:,0]).sum(axis=1)>=20).sum())
                for sensor in ['HH','EL','HG']:row['n_'+sensor]=int((a[tag+'_sensor'].astype(str)==sensor).sum())
                good=np.isfinite(spectrum[:,0]).sum(axis=0)>=max(3,.3*len(spectrum))
                row['usable_frequency_count']=int(good.sum())
                row['usable_band_span_decades']=float(np.ptp(np.log10(f[good]))) if good.sum()>=2 else 0.
                tag2=f'T{secondary_window:g}'
                if row['Mw_quality']=='F' and tag2 in a:
                    def useful_keys(label):
                        aa=a[label];nn=a[label+'_stations'].astype(str);ss=a[label+'_sensor'].astype(str)
                        return {(nn[i],ss[i]) for i in range(len(nn)) if np.isfinite(aa[i,0]).sum()>=20}
                    row['common_window_stations']=len(useful_keys(tag)&useful_keys(tag2))
            else:row['usable_frequency_count']=0;row['usable_band_span_decades']=0.
    if row['Mw_quality']=='F':
        row['Mw_reasons']='insufficient_usable_frequency_band' if row['n_stations_SNR']>=5 else 'too_few_usable_triplets'
        if raw.get('status')=='error':row['Mw_reasons']='waveform_processing_error'
    selected=ml.get('selected',[])
    if len(selected)>=3:
        values=np.array([s['value'] for s in selected]);rng=np.random.default_rng(int(eid[-8:]))
        boots=np.median(rng.choice(values,(2000,len(values)),replace=True),axis=1)
        row['ML_station_bootstrap_lower'],row['ML_station_bootstrap_upper']=np.quantile(boots,[.025,.975])
    row['paired_A']=row['Mw_quality']=='A' and row.get('ML_quality')=='A'
    row['paired_A_or_B']=row['Mw_quality'] in ['A','B'] and row.get('ML_quality') in ['A','B']
    return row


def design_matrix(x,model):
    x=np.asarray(x)-2
    return np.column_stack([np.ones(len(x)),x]+([x*x] if model=='quadratic' else [np.minimum(x,0)] if model=='hinge2' else []))


def regressions(pilot,accept_b=False):
    old=pd.read_csv(R/'tables/Mw_independent_ML_primary.csv',dtype={'event_id':str})
    old['pilot_role']='reference';old['study_group']='ML>=2 reference'
    flag='paired_A_or_B' if accept_b else 'paired_A'
    suffix='_AB' if accept_b else ''
    small=pilot[pilot.pilot_role.eq('small')&pilot[flag]].copy();small['study_group']='ML<2 pilot'
    combined=pd.concat([old,small],ignore_index=True)
    assert combined.event_id.is_unique
    combined.to_csv(P/'tables'/f'combined_paired{suffix or "_A"}.csv',index=False)
    if len(small)<8 or small.family_block.nunique()<4:
        return dict(n_new_paired_A=len(small),n_new_blocks=int(small.family_block.nunique()),fit_extension=False)
    rows=[];predictions=[];rng=np.random.default_rng(20261119)
    for selection,g in [('Reference only',old),('Reference plus pilot',combined)]:
        x=g.ML_Sheen.to_numpy();y=g.Mw.to_numpy();blocks=g.family_block.to_numpy();unique=np.unique(blocks)
        for model in ['linear','quadratic','hinge2']:
            if selection=='Reference only' and model=='hinge2':continue
            X=design_matrix(x,model);p=X.shape[1]
            xx=np.array([X[blocks==key].T@X[blocks==key]/sum(blocks==key) for key in unique])
            xy=np.array([X[blocks==key].T@y[blocks==key]/sum(blocks==key) for key in unique])
            coef=np.linalg.lstsq(xx.sum(axis=0),xy.sum(axis=0),rcond=None)[0]
            draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=1500)
            mats=np.einsum('bi,ijk->bjk',draws,xx);rhs=draws@xy
            boot=[]
            for matrix,right in zip(mats,rhs):
                if np.linalg.matrix_rank(matrix)==p:boot.append(np.linalg.solve(matrix,right))
            boot=np.asarray(boot);lo,hi=np.quantile(boot,[.025,.975],axis=0)
            residual=y-X@coef
            held=np.full(len(g),np.nan)
            for i,key in enumerate(unique):
                matrix=xx.sum(axis=0)-xx[i];right=xy.sum(axis=0)-xy[i]
                if np.linalg.matrix_rank(matrix)<p:continue
                c=np.linalg.solve(matrix,right);held[blocks==key]=X[blocks==key]@c
            new=g.pilot_role.eq('small').to_numpy()
            c3=float(coef[2]) if p==3 else 0.
            row=dict(selection=selection,model=model,method='family-balanced OLS',n_events=len(g),n_blocks=len(unique),
                a_at_ML2=coef[0],b_at_ML2=coef[1],curvature_or_low_slope_change=c3,
                a_lo=lo[0],a_hi=hi[0],b_lo=lo[1],b_hi=hi[1],c_lo=lo[2] if p==3 else 0,c_hi=hi[2] if p==3 else 0,
                RMS=float(np.sqrt(np.mean(residual**2))),CV_RMS=float(np.sqrt(np.nanmean((held-y)**2))),
                CV_RMS_small=float(np.sqrt(np.nanmean((held[new]-y[new])**2))) if new.any() else np.nan,
                successful_bootstraps=len(boot))
            if model=='hinge2':
                row['slope_below_ML2']=float(coef[1]+coef[2])
                row['low_slope_lo'],row['low_slope_hi']=np.quantile(boot[:,1]+boot[:,2],[.025,.975])
            rows.append(row)
            for eid,role,xx0,yy0,pred in zip(g.event_id,g.pilot_role,x,y,held):
                predictions.append(dict(selection=selection,model=model,event_id=eid,pilot_role=role,ML_measured=xx0,Mw=yy0,heldout_prediction=pred))
        # Equal-axis orthogonal regression is a sensitivity, not a censoring correction.
        w=np.array([1/np.sum(blocks==key) for key in blocks]);w/=w.sum()
        mx=np.sum(w*x);my=np.sum(w*y);vx=np.sum(w*(x-mx)**2);vy=np.sum(w*(y-my)**2);cov=np.sum(w*(x-mx)*(y-my))
        slope=(vy-vx+np.sqrt((vy-vx)**2+4*cov*cov))/(2*cov)
        rows.append(dict(selection=selection,model='linear',method='equal-axis ODR; family weights',n_events=len(g),n_blocks=len(unique),a_at_ML2=my+slope*(2-mx),b_at_ML2=slope,curvature_or_low_slope_change=0))
    pd.DataFrame(rows).to_csv(P/'tables'/f'exploratory_regressions{suffix}.csv',index=False)
    pd.DataFrame(predictions).to_csv(P/'tables'/f'family_heldout_predictions{suffix}.csv',index=False)
    return dict(n_new_paired_A=len(small),n_new_blocks=int(small.family_block.nunique()),
                combined_events=len(combined),combined_blocks=int(combined.family_block.nunique()),fit_extension=True,
                regression_scope=('A/B-grade sensitivity; ' if accept_b else '')+'Exploratory selected pilot plus reference. Family bootstrap/CV and ODR do not correct for SNR-dependent missingness.')


def main(allow_incomplete=False):
    cohort=pd.read_csv(P/'inputs/pilot_cohort.csv',dtype={'event_id':str})
    missing=[eid for eid in cohort.event_id if not (P/'events'/f'{eid}.json').exists() or not (P/'ml_events'/f'{eid}.json').exists()]
    if missing and not allow_incomplete:raise RuntimeError(f'{len(missing)} cases still await processing')
    rows=[]
    for case in cohort.to_dict('records'):
        if case['event_id'] not in missing:rows.append(event_result(case))
    if not rows:raise RuntimeError('No completed cases')
    d=pd.DataFrame(rows);d['multitaper_NW']=4;d['multitaper_K']=7
    d.to_csv(P/'tables/pilot_Mw_ML_catalog.csv',index=False)
    groups=[]
    for keys,q in d.groupby(['pilot_role','magnitude_stratum'],sort=True):
        groups.append(dict(pilot_role=keys[0],magnitude_stratum=keys[1],attempted=len(q),
            finite_Mw=int(q.Mw.notna().sum()),Mw_A=int(q.Mw_quality.eq('A').sum()),Mw_B=int(q.Mw_quality.eq('B').sum()),
            ML_measured=int(q.ML_Sheen.notna().sum()),ML_A=int(q.ML_quality.eq('A').sum()),paired_A=int(q.paired_A.sum()),paired_A_or_B=int(q.paired_A_or_B.sum()),
            design_weighted_paired_A_fraction=float(np.average(q.paired_A,weights=1/q.selection_probability))))
    groups=pd.DataFrame(groups);groups.to_csv(P/'tables/yield_by_magnitude.csv',index=False)
    dimensions=[]
    for variable in ['depth_stratum','era_stratum']:
        for (mag,level),q in d[d.pilot_role.eq('small')].groupby(['magnitude_stratum',variable]):
            dimensions.append(dict(dimension=variable,magnitude_stratum=mag,level=level,attempted=len(q),Mw_A=int(q.Mw_quality.eq('A').sum()),paired_A=int(q.paired_A.sum())))
    pd.DataFrame(dimensions).to_csv(P/'tables/yield_depth_era.csv',index=False)
    failures=[]
    for (mag,reason),n in d[d.pilot_role.eq('small')].assign(reason=d.Mw_reasons.fillna('').str.split(';')).explode('reason').groupby(['magnitude_stratum','reason']).size().items():
        if reason:failures.append(dict(magnitude_stratum=mag,reason=reason,count=int(n)))
    pd.DataFrame(failures).to_csv(P/'tables/failure_reasons.csv',index=False)
    old=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str}).set_index('event_id')
    controls=[]
    for row in d[d.pilot_role.eq('control')].itertuples():
        ref=old.loc[row.event_id]
        controls.append(dict(event_id=row.event_id,Mw_pilot=row.Mw,Mw_reference=ref.Mw,
            delta_Mw=row.Mw-ref.Mw,ML_pilot=row.ML_Sheen,ML_reference=ref.ML_Sheen,
            delta_ML=row.ML_Sheen-ref.ML_Sheen,Mw_quality_pilot=row.Mw_quality,Mw_quality_reference=ref.Mw_quality,
            ML_quality_pilot=row.ML_quality,ML_quality_reference=ref.ML_quality))
    pd.DataFrame(controls).to_csv(P/'tables/overlap_policy_controls.csv',index=False)
    protected=json.loads((P/'inputs/protected_results.json').read_text())
    assert all(hashlib.sha256((R/name).read_bytes()).hexdigest()==value for name,value in protected.items())
    small=d[d.pilot_role.eq('small')];paired=small[small.paired_A]
    summary=dict(passed=True,complete=not missing,attempted=len(d),planned=len(cohort),small_attempted=len(small),
        small_Mw_quality_counts=small.Mw_quality.value_counts().to_dict(),small_ML_quality_counts=small.ML_quality.value_counts().to_dict(),
        small_paired_A=int(small.paired_A.sum()),small_paired_A_or_B=int(small.paired_A_or_B.sum()),
        measured_ML_range_paired_A=[float(paired.ML_Sheen.min()),float(paired.ML_Sheen.max())] if len(paired) else [],
        KMA_ML_range_paired_A=[float(paired.ml.min()),float(paired.ml.max())] if len(paired) else [],
        Mw_range_paired_A=[float(paired.Mw.min()),float(paired.Mw.max())] if len(paired) else [],
        finite_Mw=int(small.Mw.notna().sum()),yield_by_magnitude=groups.to_dict('records'),
        original_adopted_results_unchanged=True)
    if not missing:
        summary['regressions']=regressions(d)
        summary['regressions_A_or_B_sensitivity']=regressions(d,accept_b=True)
    (P/'tables/pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def analyze_short(nw2=False):
    root=P/('nw2_windows' if nw2 else 'short_windows')
    primary,secondary=(5.1,10.1) if nw2 else (3.1,5.1)
    prefix='nw2_window' if nw2 else 'short_window'
    cohort=pd.read_csv(P/'inputs/pilot_cohort.csv',dtype={'event_id':str})
    missing=[eid for eid in cohort.event_id if not (root/'events'/f'{eid}.json').exists()]
    if missing:raise RuntimeError(f'{len(missing)} short-window cases not complete')
    d=pd.DataFrame([event_result(case,root,primary,secondary) for case in cohort.to_dict('records')])
    d['multitaper_NW']=2 if nw2 else 4;d['multitaper_K']=3 if nw2 else 7
    original=pd.read_csv(P/'tables/pilot_Mw_ML_catalog.csv',dtype={'event_id':str}).set_index('event_id')
    d['Mw_short_minus_standard']=[row.Mw-original.loc[row.event_id,'Mw'] for row in d.itertuples()]
    d['paired_A_standard']=[bool(original.loc[eid,'paired_A']) for eid in d.event_id]
    d.to_csv(P/'tables'/f'{prefix}_Mw_ML_catalog.csv',index=False)
    groups=[]
    for (role,mag),q in d.groupby(['pilot_role','magnitude_stratum']):
        groups.append(dict(pilot_role=role,magnitude_stratum=mag,attempted=len(q),finite_Mw=int(q.Mw.notna().sum()),
            Mw_A=int(q.Mw_quality.eq('A').sum()),Mw_B=int(q.Mw_quality.eq('B').sum()),paired_A=int(q.paired_A.sum()),paired_A_or_B=int(q.paired_A_or_B.sum())))
    pd.DataFrame(groups).to_csv(P/'tables'/f'{prefix}_yields.csv',index=False)
    control=d[d.pilot_role.eq('control')].copy()
    common=control[control.Mw_quality.eq('A')&[original.loc[eid,'Mw_quality']=='A' for eid in control.event_id]]
    deltas=common.Mw_short_minus_standard.dropna()
    passed=len(deltas)>=5 and abs(deltas.median())<=.05 and np.sqrt(np.mean(deltas**2))<=.10
    controls=control[['event_id','Mw','Mw_quality','Mw_short_minus_standard','paired_A','paired_A_standard']]
    controls.to_csv(P/'tables'/f'{prefix}_controls.csv',index=False)
    small=d[d.pilot_role.eq('small')];paired=small[small.paired_A]
    fields=['event_id','origin_utc','lat','lon','z_central','lower','upper','ml','ML_Sheen','ML_quality','Mw','Mw_quality',
            'n_stations_SNR','common_window_stations','difference_Fourier','difference_Brune','window_difference_common',
            'primary_window_s','secondary_window_s','multitaper_NW','multitaper_K','n_HH','n_EL','n_HG',
            'Mw_station_bootstrap_lo','Mw_station_bootstrap_hi','ML_station_bootstrap_lower','ML_station_bootstrap_upper']
    paired.reindex(columns=fields).to_csv(P/'tables'/f'{prefix}_candidate_pairs.csv',index=False)
    summary=dict(complete=True,diagnostic_only=True,attempted=len(d),yield_by_magnitude=groups,
        small_paired_A=int(small.paired_A.sum()),small_paired_A_or_B=int(small.paired_A_or_B.sum()),
        measured_ML_range_paired_A=[float(paired.ML_Sheen.min()),float(paired.ML_Sheen.max())] if len(paired) else [],
        transfer_controls=dict(n_shared_Mw_A=len(deltas),median_delta_Mw=float(deltas.median()),
            RMS_delta_Mw=float(np.sqrt(np.mean(deltas**2))),max_abs_delta_Mw=float(abs(deltas).max()),
            criterion='At least 5 shared Mw-A controls; |median delta|<=0.05 and RMS delta<=0.10, fixed before the control outputs were assessed.',passed=bool(passed)),
        interpretation='Short-window estimates remain a sensitivity. Control agreement is necessary but not sufficient for an unbiased small-event scaling; no catalog replacement or residual-based amplitude offset.')
    summary['scheme']='NW2, K3, 5.1/10.1 s, 0.5-20 Hz' if nw2 else 'NW4, K7, 3.1/5.1 s, 1.5-20 Hz'
    (P/'tables'/f'{prefix}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--short-windows',action='store_true')
    parser.add_argument('--nw2-windows',action='store_true')
    args=parser.parse_args()
    if args.nw2_windows:analyze_short(nw2=True)
    elif args.short_windows:analyze_short()
    else:main(args.allow_incomplete)
