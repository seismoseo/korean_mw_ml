"""Grade every expanded attempt and compare uniform-estimator Mw–ML relations."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import analyze_small_event_pilot as A

R=Path(__file__).resolve().parents[1];E=R/'small_events/extension';T=E/'tables'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def cases_done(cohort):
    return [c for c in cohort.to_dict('records') if all(p.exists() for p in [
        Path(c['baseline_root'])/'events'/f"{c['event_id']}.json",
        Path(c['baseline_root'])/'ml_events'/f"{c['event_id']}.json",
        Path(c['candidate_root'])/'events'/f"{c['event_id']}.json"]) ]


def measure(case):
    A.P=Path(case['baseline_root'])
    old=A.event_result(case,Path(case['baseline_root']),10.1,20.1)
    new=A.event_result(case,Path(case['candidate_root']),5.1,10.1)
    for row,nw,k in [(old,4,7),(new,2,3)]:row.update(multitaper_NW=nw,multitaper_K=k)
    new['Mw_NW2_minus_NW4']=new['Mw']-old['Mw']
    return old,new


def matrix(x,model):
    u=np.asarray(x)-2
    return np.column_stack([np.ones(len(u)),u]+([u*u] if model=='quadratic' else [np.minimum(u,0)] if model=='hinge2' else []))


def model_fit(frame,selection,models=('linear','quadratic','hinge2'),design=False,bootstraps=2000):
    """Family-balanced OLS, family bootstrap and leave-one-whole-family-out CV."""
    x=frame.ML_Sheen.to_numpy();y=frame.Mw.to_numpy();block=frame.family_block.to_numpy()
    unique,ii=np.unique(block,return_inverse=True);n=len(unique)
    base=1/frame.selection_probability.to_numpy() if design else np.ones(len(frame))
    w=base if design=='event_IPW' else base/np.bincount(ii,weights=base)[ii]
    method=('event design-weighted OLS' if design=='event_IPW' else
            'family-balanced OLS with design weights' if design else 'family-balanced OLS')
    rng=np.random.default_rng(20261120);draws=rng.multinomial(n,np.full(n,1/n),size=bootstraps)
    rows=[];predictions=[];bands=[]
    for model in models:
        X=matrix(x,model);p=X.shape[1]
        xx=np.stack([X[ii==i].T@(X[ii==i]*w[ii==i,None]) for i in range(n)])
        xy=np.stack([X[ii==i].T@(y[ii==i]*w[ii==i]) for i in range(n)])
        totalxx=xx.sum(axis=0);totalxy=xy.sum(axis=0);coef=np.linalg.solve(totalxx,totalxy)
        boot=[]
        for mat,rhs in zip(np.einsum('bi,ijk->bjk',draws,xx),draws@xy):
            if np.linalg.matrix_rank(mat)==p:boot.append(np.linalg.solve(mat,rhs))
        boot=np.asarray(boot);bounds=np.quantile(boot,[.025,.975],axis=0)
        held=np.full(len(frame),np.nan)
        for i in range(n):
            mat=totalxx-xx[i]
            if np.linalg.matrix_rank(mat)==p:held[ii==i]=X[ii==i]@np.linalg.solve(mat,totalxy-xy[i])
        errors=(y-held)**2;new=frame.confirmatory_small.to_numpy(dtype=bool)
        low=new&(x>=1.5)&(x<2.)
        def rms(mask):return float(np.sqrt(np.mean(errors[mask]))) if mask.any() else np.nan
        def family_rms(mask):
            return float(np.sqrt(np.mean([np.mean(errors[mask&(ii==i)]) for i in np.unique(ii[mask])]))) if mask.any() else np.nan
        row=dict(selection=selection,method=method,
            model=model,n_events=len(frame),n_blocks=n,n_new_small=int(new.sum()),n_new_small_ML1p5_2=int(low.sum()),
            a_at_ML2=coef[0],b_at_ML2=coef[1],c=coef[2] if p==3 else 0,
            a_lo=bounds[0,0],a_hi=bounds[1,0],b_lo=bounds[0,1],b_hi=bounds[1,1],
            c_lo=bounds[0,2] if p==3 else 0,c_hi=bounds[1,2] if p==3 else 0,
            RMS=float(np.sqrt(np.mean((y-X@coef)**2))),CV_RMS=rms(np.ones(len(frame),bool)),
            CV_RMS_new_small=rms(new),CV_RMS_new_small_ML1p5_2=rms(low),
            CV_RMS_family=family_rms(np.ones(len(frame),bool)),CV_RMS_family_new_small=family_rms(new),
            CV_RMS_family_new_small_ML1p5_2=family_rms(low),successful_bootstraps=len(boot),
            slope_below2=coef[1]+coef[2] if model=='hinge2' else np.nan,
            x_min=float(x.min()),x_max=float(x.max()))
        if model=='hinge2':row['low_slope_lo'],row['low_slope_hi']=np.quantile(boot[:,1]+boot[:,2],[.025,.975])
        rows.append(row)
        for i,case in enumerate(frame.itertuples()):
            predictions.append(dict(selection=selection,model=model,event_id=case.event_id,family_block=case.family_block,
                confirmatory_small=bool(case.confirmatory_small),ML_Sheen=x[i],Mw=y[i],prediction=held[i],error=y[i]-held[i]))
        grid=np.linspace(max(1.,float(x.min())),float(x.max()),180);curves=boot@matrix(grid,model).T
        for xx0,estimate,lo,hi in zip(grid,matrix(grid,model)@coef,*np.quantile(curves,[.025,.975],axis=0)):
            bands.append(dict(selection=selection,model=model,ML_Sheen=xx0,Mw=estimate,lower=lo,upper=hi))
    # Equal-axis ODR and variance-ratio Deming; no claim of absolute axis errors.
    norm=w/w.sum();mom=np.column_stack([norm,norm*x,norm*y,norm*x*x,norm*y*y,norm*x*y])
    summaries=np.stack([mom[ii==i].sum(axis=0) for i in range(n)])
    total=summaries.sum(axis=0);samples=draws@summaries
    def deming(values,ratio):
        mass,sx,sy,sxx,syy,sxy=values.T;mx=sx/mass;my=sy/mass
        vx=sxx/mass-mx*mx;vy=syy/mass-my*my;cov=sxy/mass-mx*my
        b=(vy-ratio*vx+np.sqrt((vy-ratio*vx)**2+4*ratio*cov*cov))/(2*cov)
        return my+b*(2-mx),b
    for ratio in [.25,1.,4.]:
        a,b=deming(total[None,:],ratio);aa,bb=deming(samples,ratio)
        rows.append(dict(selection=selection,method='equal-axis ODR' if ratio==1 else f'Deming variance ratio {ratio:g}',model='linear',
            n_events=len(frame),n_blocks=n,a_at_ML2=float(a[0]),b_at_ML2=float(b[0]),c=0,
            a_lo=float(np.quantile(aa,.025)),a_hi=float(np.quantile(aa,.975)),b_lo=float(np.quantile(bb,.025)),b_hi=float(np.quantile(bb,.975)),
            x_min=float(x.min()),x_max=float(x.max()),successful_bootstraps=bootstraps))
    return rows,predictions,bands


def fit_all(new,old):
    flag=new.paired_A&new.ML_Sheen.ge(1.5);primary=new[flag].copy()
    primary.to_csv(T/'paired_A_MLge1p5.csv',index=False)
    rows=[];pred=[];bands=[]
    selections=[('NW2 primary',primary,False),
        ('NW2 without original pilot',primary[~primary.in_original_pilot],False),
        ('NW2 design-weight sensitivity',primary,True),
        ('NW2 event-design sensitivity',primary,'event_IPW'),
        ('NW2 census only',primary[primary.ml.ge(1.5)],False),
        ('NW2 A/B sensitivity',new[new.paired_A_or_B&new.ML_Sheen.ge(1.5)],False),
        ('NW2 reference only',primary[primary.ml.ge(2)],False),
        ('NW4 expanded',old[old.paired_A&old.ML_Sheen.ge(1.5)],False)]
    for label,frame,design in selections:
        rr,pp,bb=model_fit(frame,label,design=design);rows+=rr;pred+=pp;bands+=bb
        print(label,len(frame),'events',frame.family_block.nunique(),'families',flush=True)
    rr,pp,bb=model_fit(old[old.ml.ge(2)&old.paired_A&old.ML_Sheen.ge(1.5)],'NW4 reference only',models=('linear',))
    rows+=rr;pred+=pp;bands+=bb
    alternative=primary.assign(Mw=primary.Mw_source_Vs_Kim2011)
    rr,pp,bb=model_fit(alternative,'NW2 Kim2011 source Vs');rows+=rr;pred+=pp;bands+=bb
    pilot_families=set(new.loc[new.in_original_pilot,'family_block'])
    independent=primary[~primary.family_block.isin(pilot_families)]
    rr,pp,bb=model_fit(independent,'NW2 without pilot families');rows+=rr;pred+=pp;bands+=bb
    common=set(new[new.paired_A].event_id)&set(old[old.paired_A].event_id)
    for scheme,frame in [('NW2',new),('NW4',old)]:
        for suffix,q in [('same reference events',frame[frame.ml.ge(2)&frame.event_id.isin(common)&frame.ML_Sheen.ge(1.5)])]:
            rr,pp,bb=model_fit(q,f'{scheme} {suffix}',models=('linear',));rows+=rr;pred+=pp;bands+=bb
    for floor in [1.7,2.,2.3]:
        rr,pp,bb=model_fit(new[new.paired_A&new.ML_Sheen.ge(floor)],f'NW2 floor {floor:g}',models=('linear',))
        rows+=rr;pred+=pp;bands+=bb
    pd.DataFrame(rows).to_csv(T/'regressions.csv',index=False)
    predictions=pd.DataFrame(pred);predictions.to_csv(T/'family_heldout_predictions.csv',index=False)
    pd.DataFrame(bands).to_csv(T/'regression_bands.csv',index=False)
    # Paired whole-family bootstrap of held-out squared-error differences.
    p=predictions[predictions.selection.eq('NW2 primary')&predictions.confirmatory_small]
    pivot=p.pivot(index=['event_id','family_block','ML_Sheen'],columns='model',values='error').reset_index()
    comparison=[];rng=np.random.default_rng(20261120)
    for sub,mask in [('New small, ML>=1.5',np.ones(len(pivot),bool)),('New small, ML1.5–2',pivot.ML_Sheen.lt(2).to_numpy())]:
        q=pivot[mask]
        for alt in ['quadratic','hinge2']:
            delta=q[alt]**2-q.linear**2;family=delta.groupby(q.family_block).mean().to_numpy()
            bootstrap=rng.choice(family,(5000,len(family)),replace=True).mean(axis=1)
            lo,hi=np.quantile(bootstrap,[.025,.975])
            comparison.append(dict(selection=sub,comparison=alt+' minus linear',n_events=len(q),n_blocks=len(family),
                family_mean_delta_MSE=float(family.mean()),lower=float(lo),upper=float(hi),negative_favors_alternative=True))
    pd.DataFrame(comparison).to_csv(T/'model_prediction_comparison.csv',index=False)
    return primary


def transfer(new,old,plan):
    ref=new[new.ml.ge(2)].set_index('event_id',drop=False);base=old.set_index('event_id')
    shared=ref[ref.Mw_quality.eq('A')&base.loc[ref.index,'Mw_quality'].eq('A')].copy()
    shared['delta_Mw']=shared.Mw-base.loc[shared.index,'Mw']
    shared.to_csv(T/'shared_Mw_A_controls.csv',index=False)
    rows=[]
    for label,mask in [('All',np.ones(len(shared),bool)),('ML2–2.4',shared.ml.lt(2.5)),('ML2.5–2.9',shared.ml.ge(2.5)&shared.ml.lt(3)),('ML>=3',shared.ml.ge(3))]:
        d=shared.loc[mask,'delta_Mw'].to_numpy()
        rows.append(dict(bin=label,n=len(d),median_delta=float(np.median(d)) if len(d) else np.nan,
            RMS_delta=float(np.sqrt(np.mean(d*d))) if len(d) else np.nan,max_abs_delta=float(abs(d).max()) if len(d) else np.nan))
    d=pd.DataFrame(rows);d.to_csv(T/'transfer_controls.csv',index=False);g=plan['transfer_gate'];allrow=d.iloc[0]
    bins=d.iloc[1:];qualified=bins[bins.n.ge(g['minimum_per_bin'])]
    passed=(allrow.n>=g['minimum_shared_Mw_A'] and abs(allrow.median_delta)<=g['maximum_abs_median_delta'] and
        allrow.RMS_delta<=g['maximum_RMS_delta'] and qualified.median_delta.abs().le(g['per_catalog_ML_bin_maximum_abs_median']).all())
    return dict(passed=bool(passed),all_controls=rows[0],bins=rows[1:],gate=g,
        limit='Agreement assesses transfer between estimators on shared accepted records, not absolute accuracy or complete low-SNR coverage.')


def sensor_pairs(new):
    """Diagnostic only: co-located HH/HG at surviving small-event sites.

    Original NW4 archives retain the identity of every alternative record;
    the paired ratio is independent of between-site source/radiation effects.
    It is not used to tune acceptance or fit a sensor amplitude correction.
    """
    rows=[];coordinate_pairs=0
    for case in new[new.ml.lt(2)&new.paired_A].itertuples():
        root=Path(case.baseline_root);path=root/'spectra'/f'{case.event_id}.npz'
        with np.load(path) as archive:
            tag='T10.1_allrecords'
            if tag+'_stations' not in archive:continue
            arr=archive[tag];f=archive['frequency_Hz'];names=archive[tag+'_stations'].astype(str);sensors=archive[tag+'_sensor'].astype(str)
            log=pd.read_csv(root/'station_logs'/f'{case.event_id}.csv')
            log=log[log.window_s.eq(10.1)&log.status.isin(['ready','alternative'])].drop_duplicates(['station','sensor']).set_index(['station','sensor'])
            hh=np.flatnonzero(sensors=='HH');hg=np.flatnonzero(sensors=='HG')
            if not len(hh) or not len(hg):continue
            def xy(ids):
                records=log.loc[[(names[i],sensors[i]) for i in ids]]
                return np.column_stack([records.epicentral_km*np.sin(np.radians(records.azimuth)),
                                        records.epicentral_km*np.cos(np.radians(records.azimuth))])
            hxy,gxy=xy(hh),xy(hg);separation=np.linalg.norm(hxy[:,None,:]-gxy[None,:,:],axis=2)
            coordinate_pairs+=int((separation<=.1).sum())
            for ih,jg in zip(*np.where(separation<=.1)):
                i,j=hh[ih],hg[jg];delta=separation[ih,jg]
                for label,lo,hi in [('0.7–1.2 Hz',.7,1.2),('1–5 Hz',1,5),('5–10 Hz',5,10)]:
                    band=(f>=lo)&(f<=hi);good=band&np.isfinite(arr[i,0])&np.isfinite(arr[j,0])
                    if good.sum()<max(8,.5*band.sum()):continue
                    ratio=arr[j,0,good]-arr[i,0,good]
                    median=float(np.median(ratio))
                    station=names[i] if names[i]==names[j] else names[i]+'/'+names[j]
                    rows.append(dict(event_id=case.event_id,station=station,band=label,catalog_ML=case.ml,ML_Sheen=case.ML_Sheen,
                        n_frequency=int(good.sum()),coordinate_gap_km=float(delta),log10_HG_over_HH=median,
                        HG_over_HH=10**median,equivalent_delta_Mw=median/1.5,
                        diagnostic_scheme='Original 10.1 s NW4; both sensors SNR>=3; <100 m coordinate difference',
                        used_for_amplitude_correction=False))
    d=pd.DataFrame(rows,columns=['event_id','station','band','catalog_ML','ML_Sheen','n_frequency','coordinate_gap_km',
        'log10_HG_over_HH','HG_over_HH','equivalent_delta_Mw','diagnostic_scheme','used_for_amplitude_correction'])
    d.to_csv(T/'small_event_HH_HG_pairs.csv',index=False)
    if d.empty:
        return dict(event_station_band_pairs=0,coordinate_pairs=coordinate_pairs,scope='No paired usable records; no amplitude correction')
    station=d.groupby(['station','band']).agg(n_events=('event_id','nunique'),median_log_ratio=('log10_HG_over_HH','median')).reset_index()
    station['median_HG_over_HH']=10**station.median_log_ratio
    station.to_csv(T/'small_event_HH_HG_station_summary.csv',index=False)
    summaries=[]
    for band,q in d.groupby('band'):
        st=station[station.band.eq(band)]
        summaries.append(dict(band=band,event_station_pairs=len(q),events=int(q.event_id.nunique()),stations=int(q.station.nunique()),
            pair_median_HG_over_HH=float(10**q.log10_HG_over_HH.median()),
            station_balanced_median_HG_over_HH=float(10**st.median_log_ratio.median()),
            station_log_ratio_q16=float(st.median_log_ratio.quantile(.16)),station_log_ratio_q84=float(st.median_log_ratio.quantile(.84))))
    pd.DataFrame(summaries).to_csv(T/'small_event_HH_HG_summary.csv',index=False)
    return dict(event_station_band_pairs=len(d),coordinate_pairs=coordinate_pairs,bands=summaries,
        scope='Observed consistency for co-located, jointly SNR-passing records of accepted small events, not a low-SNR completeness test; no fitted correction')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-incomplete',action='store_true');ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args();cohort=pd.read_csv(E/'inputs/cohort.csv',dtype={'event_id':str});cases=cases_done(cohort)
    if len(cases)!=len(cohort) and not args.allow_incomplete:raise RuntimeError(f'{len(cohort)-len(cases)} cases not yet attempted')
    with concurrent.futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        result=list(pool.map(measure,cases,chunksize=4))
    old=pd.DataFrame([v[0] for v in result]);new=pd.DataFrame([v[1] for v in result])
    old.to_csv(T/'NW4_Mw_ML_catalog.csv',index=False);new.to_csv(T/'NW2_Mw_ML_catalog.csv',index=False)
    rows=[]
    for name,frame in [('NW4',old),('NW2',new)]:
        for (role,mag),q in frame.groupby(['extension_role','magnitude_stratum']):
            paired=q[q.paired_A]
            rows.append(dict(scheme=name,role=role,magnitude_stratum=mag,attempted=len(q),finite_Mw=int(q.Mw.notna().sum()),
                Mw_A=int(q.Mw_quality.eq('A').sum()),ML_A=int(q.ML_quality.eq('A').sum()),paired_A=int(q.paired_A.sum()),paired_AB=int(q.paired_A_or_B.sum()),
                measured_ML_min_A=float(paired.ML_Sheen.min()),measured_ML_max_A=float(paired.ML_Sheen.max())))
    pd.DataFrame(rows).to_csv(T/'yield_by_group.csv',index=False)
    failure=[]
    for (role,reason),n in new.assign(reason=new.Mw_reasons.fillna('').str.split(';')).explode('reason').groupby(['extension_role','reason']).size().items():
        if reason:failure.append(dict(role=role,reason=reason,count=int(n)))
    pd.DataFrame(failure).to_csv(T/'failure_reasons.csv',index=False)
    coverage=[]
    new['measured_ML_bin']=pd.cut(new.ML_Sheen,[0,1.5,1.75,2,2.25,2.5,3,4,6],right=False).astype(str)
    for dimension in ['depth_stratum','era_stratum','measured_ML_bin']:
        for (role,level),q in new.groupby(['extension_role',dimension]):
            coverage.append(dict(dimension=dimension,role=role,level=level,attempted=len(q),ML_measured=int(q.ML_Sheen.notna().sum()),
                ML_A=int(q.ML_quality.eq('A').sum()),Mw_A=int(q.Mw_quality.eq('A').sum()),paired_A=int(q.paired_A.sum())))
    pd.DataFrame(coverage).to_csv(T/'coverage.csv',index=False)
    source=read(E/'inputs/config.json');summary=dict(complete=len(cases)==len(cohort),attempted=len(new),planned=len(cohort),
        small_attempted=int(new.ml.lt(2).sum()),new_small_attempted=int(new.confirmatory_small.sum()),
        uniform_estimator='NW2 K3 5.1/10.1 s, 0.5–20 Hz',adopted_catalog_preserved=True)
    if summary['complete']:
        summary['transfer']=transfer(new,old,source);primary=fit_all(new,old)
        summary['small_event_sensor_check']=sensor_pairs(new)
        canonical=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str}).set_index('event_id')
        ref=old[old.ml.ge(2)].copy().set_index('event_id',drop=False)
        for field in ['Mw','ML_Sheen','Mw_quality','ML_quality']:ref['previous_'+field]=canonical.loc[ref.index,field]
        ref['delta_Mw_overlap']=ref.Mw-ref.previous_Mw;ref['delta_ML_overlap']=ref.ML_Sheen-ref.previous_ML_Sheen
        ref.to_csv(T/'reference_overlap_reassessment.csv',index=False)
        summary['reference_overlap_reassessment']=dict(reprocessed=int(ref.baseline_root.eq(str(E)).sum()),
            changed_finite_Mw=int(ref.delta_Mw_overlap.abs().gt(1e-10).sum()),changed_finite_ML=int(ref.delta_ML_overlap.abs().gt(1e-10).sum()),
            changed_Mw_grade=int(ref.Mw_quality.ne(ref.previous_Mw_quality).sum()),changed_ML_grade=int(ref.ML_quality.ne(ref.previous_ML_quality).sum()))
        pairs=new[new.ml.lt(2)&new.paired_A];fresh=pairs[pairs.confirmatory_small]
        summary.update(small_paired_A=len(pairs),small_paired_AB=int((new.ml.lt(2)&new.paired_A_or_B).sum()),
            new_small_paired_A=len(fresh),new_small_paired_A_measured_ML1p5_2=int((fresh.ML_Sheen.ge(1.5)&fresh.ML_Sheen.lt(2)).sum()),
            paired_A_measured_ML_range=[float(pairs.ML_Sheen.min()),float(pairs.ML_Sheen.max())],
            primary_events=len(primary),primary_blocks=int(primary.family_block.nunique()))
        new[new.ml.lt(2)&new.paired_A].to_csv(T/'small_paired_A.csv',index=False)
        fields=['event_id','origin_utc','lat','lon','z_central','lower','upper','grade','ml','ML_Sheen','ML_quality','ML_Sheen_n',
            'Mw','M0_Nm','Mw_quality','Mw_reasons','n_stations_SNR','common_window_stations','difference_Fourier','difference_Brune',
            'window_difference_common','primary_window_s','secondary_window_s','multitaper_NW','multitaper_K','source_Vs_m_s',
            'Mw_source_Vs_Kim2011','n_HH','n_EL','n_HG','Mw_station_bootstrap_lo','Mw_station_bootstrap_hi',
            'ML_station_bootstrap_lower','ML_station_bootstrap_upper','paired_A','paired_A_or_B','extension_role',
            'in_original_pilot','confirmatory_small','family_block','selection_probability']
        new.reindex(columns=fields).to_csv(T/'Mw_ML_catalog.csv',index=False)
        # All attempt outcomes remain in NW2_Mw_ML_catalog.csv, including blank Mw.
    for filename,digest in read(E/'inputs/protected_results.json').items():assert sha(R/filename)==digest,filename
    (T/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
