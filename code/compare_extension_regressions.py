"""Explicit previous/expanded linear and quadratic comparisons and ML replay.

Read saved measurements; never change a waveform, magnitude, selection or the
earlier reports. Both ML definitions use identical event sets within a cohort.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import multiprocessing
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from potency_regression import run_case,fit,prediction,weights_for

R=Path(__file__).resolve().parents[1];E=R/'small_events/extension';T=E/'tables'
AXES={'our':('ML_Sheen','Measured ML'),'KMA':('ml','KMA ML')}
LABELS={'previous':'Previous 357 (NW4)','expanded_NW4':'Expanded 582 (NW4)',
        'larger_NW2':'Larger 375 (NW2)','expanded_NW2':'Expanded 676 (NW2)'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csv(path):return pd.read_csv(path,dtype={'event_id':str})
def to_center2(coef):
    q=np.asarray(coef);return np.stack([q[...,0]-q[...,1]+q[...,2],q[...,1]-2*q[...,2],q[...,2]],axis=-1)


def ml_replay(case):
    eid=case['event_id'];path=Path(case['baseline_root'])/'ml_events'/f'{eid}.json'
    payload=path.read_bytes();raw=json.loads(payload);selected=raw.get('selected',[])
    components={(c['network'],c['station'],c['sensor'],c['component']):c for c in raw.get('components',[]) if c.get('status')=='processed'}
    stations=[];values=[];error=0.
    for row in selected:
        key=(row['network'],row['station'],row['sensor']);n=components[key+('N',)];e=components[key+('E',)]
        amp=float(np.sqrt(n['peak_mm']*e['peak_mm']));distance=row['distance_km']
        assert np.isclose(amp,row['A_h_mm'],rtol=1e-12,atol=1e-15),(eid,key)
        value=float(np.log10(amp)+.5869*np.log10(distance/100)+.001680*(distance-100)+3.)
        error=max(error,abs(value-row['value']));assert abs(value-row['value'])<1e-11,(eid,key)
        assert n['snr']>=3 and e['snr']>=3 and 10<=distance<=100
        values.append(value)
        stations.append(dict(event_id=eid,station=row['network']+'.'+row['station'],sensor=row['sensor'],
            epicentral_km=distance,A_N_mm=n['peak_mm'],A_E_mm=e['peak_mm'],A_H_mm=amp,SNR_N=n['snr'],SNR_E=e['snr'],
            ML_station=value,ML_event=case['ML_Sheen'],KMA_ML=case['ml'],ML_grade=case['ML_quality']))
    measured=float(np.median(values)) if len(values)>=3 else np.nan
    assert np.isclose(measured,case['ML_Sheen'],atol=1e-11,rtol=0,equal_nan=True),eid
    scatter=float(1.4826*np.median(abs(np.asarray(values)-np.median(values)))) if len(values)>=2 else np.nan
    grade='A' if len(values)>=5 and scatter<=.35 else 'B' if len(values)>=3 else 'F'
    assert grade==case['ML_quality'],eid
    return dict(event_id=eid,KMA_ML=case['ml'],ML_measured=measured,ML_grade=grade,stations=len(values),
        max_station_replay_error=error,cache_sha256=hashlib.sha256(payload).hexdigest()),stations


def evaluate(spec,frame,target,cached=None,boot=None):
    column,_=AXES[spec['axis']];x=frame[column].to_numpy();y=frame.Mw.to_numpy();blocks=frame.family_block.to_numpy()
    task=dict(case_id=spec['case_id'],selection=spec['cohort'],x='ML_Sheen' if spec['axis']=='our' else 'ML_KMA',
        y='Mw',method=spec['method'],weighting='family',degree=spec['degree'],n_bins=50,minimum=1,
        bootstrap_reps=2000 if spec['method']=='OLS' else 1000,do_cv=True,block_policy='Whole family')
    if cached is None:row,samples,_,_,_=run_case((task,x,y,blocks,frame.event_id.to_numpy()))
    else:row=dict(cached);samples=boot
    beta=np.array([row['a_at_ML3'],row['slope_at_ML3'],row['quadratic_c']])
    independent,_=fit(x,y,spec['degree'],spec['method'],weights_for(blocks,'family'))
    assert np.max(abs(independent-beta))<2e-5,(spec,np.max(abs(independent-beta)))
    center=to_center2(beta);joint=to_center2(samples);bounds=np.quantile(joint,[.025,.975],axis=0)
    model={**spec,'cohort_label':LABELS[spec['cohort']],'ML_definition':AXES[spec['axis']][1],
        'n_events':len(frame),'n_families':int(frame.family_block.nunique()),
        'a':center[0],'b':center[1],'c':center[2],
        'a_lower':bounds[0,0],'a_upper':bounds[1,0],'b_lower':bounds[0,1],'b_upper':bounds[1,1],
        'c_lower':bounds[0,2],'c_upper':bounds[1,2],
        'constant':row['constant'],'linear_coefficient':row['linear_b'],
        'x_min':float(x.min()),'x_max':float(x.max()),'bootstrap_replicates':len(samples),
        'native_CV_RMS_families':row['CV_RMS_families'],
        'reused_previous_case':cached['case_id'] if cached is not None else '',
        'point_reproduction_error':float(np.max(abs(independent-beta)))}
    tx=target[column].to_numpy();ty=target.Mw.to_numpy();tb=target.family_block.to_numpy()
    predictions=np.empty(len(target));inside=np.empty(len(target),bool)
    for family in np.unique(tb):
        test=tb==family;train=blocks!=family
        if train.all():coef=beta
        else:coef,_=fit(x[train],y[train],spec['degree'],spec['method'],weights_for(blocks[train],'family'))
        predictions[test]=prediction(coef,tx[test]);inside[test]=(tx[test]>=x[train].min())&(tx[test]<=x[train].max())
    predicted=[dict(case_id=spec['case_id'],cohort=spec['cohort'],axis=spec['axis'],method=spec['method'],degree=spec['degree'],
        event_id=eid,family_block=family,ML_measured=ml,ML_KMA=kma,Mw_target=value,predicted=estimate,error=value-estimate,
        in_training_range=bool(interior),confirmatory_small=bool(fresh))
        for eid,family,ml,kma,value,estimate,interior,fresh in zip(target.event_id,tb,target.ML_Sheen,target.ml,ty,predictions,inside,target.confirmatory_small)]
    grid=np.linspace(1.4,5.5,240) if spec['axis']=='our' else np.linspace(.5,5.3,240)
    X=np.polynomial.polynomial.polyvander(grid-2,2);lo,hi=np.quantile(joint@X.T,[.025,.975],axis=0)
    curves=[dict(case_id=spec['case_id'],cohort=spec['cohort'],axis=spec['axis'],method=spec['method'],degree=spec['degree'],
        ML=xx,Mw=yy,lower=ll,upper=hh,in_fitted_range=bool(x.min()<=xx<=x.max())) for xx,yy,ll,hh in zip(grid,X@center,lo,hi)]
    return model,joint,predicted,curves


def task_worker(task):return evaluate(*task)


def main():
    inputs=[R/'tables/Mw_independent_ML_primary.csv',R/'tables/independent_ML_regressions.csv',
        R/'tables/magnitude_followup_regressions.csv',R/'tables/magnitude_followup_bootstrap.npz',
        T/'NW2_Mw_ML_catalog.csv',T/'NW4_Mw_ML_catalog.csv',T/'Mw_ML_catalog.csv',T/'paired_A_MLge1p5.csv',
        T/'regressions.csv',T/'regression_bands.csv',T/'family_heldout_predictions.csv',T/'summary.json']
    protected={str(p.relative_to(R)):sha(p) for p in inputs}
    new=csv(T/'NW2_Mw_ML_catalog.csv');old=csv(T/'NW4_Mw_ML_catalog.csv');primary=csv(T/'paired_A_MLge1p5.csv')
    previous=csv(R/'tables/Mw_independent_ML_primary.csv')
    frames={'previous':previous,'expanded_NW4':old[old.paired_A&old.ML_Sheen.ge(1.5)],
            'larger_NW2':primary[primary.ml.ge(2)],'expanded_NW2':primary}
    for name,frame in frames.items():assert len(frame)==int(LABELS[name].split()[1]),(name,len(frame))
    cached=pd.read_csv(R/'tables/magnitude_followup_regressions.csv');old_boot=np.load(R/'tables/magnitude_followup_bootstrap.npz')
    jobs=[]
    for cohort,frame in frames.items():
        for axis in AXES:
            for method in ['OLS','ODR']:
                for degree in [1,2]:
                    spec=dict(case_id=f'{cohort}_{axis}_{method}_{degree}',cohort=cohort,axis=axis,method=method,degree=degree)
                    record=None;samples=None
                    if cohort=='previous':
                        match=cached[(cached.y=='Mw')&cached.x.eq('ML_Sheen' if axis=='our' else 'ML_KMA')&cached.method.eq(method)&cached.weighting.eq('family')&cached.degree.eq(degree)]
                        assert len(match)==1;record=match.iloc[0].to_dict();samples=old_boot[record['case_id']]
                    jobs.append((spec,frame,primary,record,samples))
    models=[];predictions=[];curves=[];bootstrap={}
    print('Replaying saved ML amplitudes for',len(new),'events',flush=True)
    event_audit=[];station_audit=[]
    with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('fork')) as pool:
        for event,stations in pool.map(ml_replay,new.to_dict('records'),chunksize=8):event_audit.append(event);station_audit.extend(stations)
    pd.DataFrame(event_audit).to_csv(T/'ML_measurement_audit.csv',index=False)
    pd.DataFrame(station_audit).to_csv(T/'ML_station_measurements.csv',index=False)
    print('ML replay passed;',len(station_audit),'station readings. Fitting/comparing',len(jobs),'cases',flush=True)
    with concurrent.futures.ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(task_worker,job):job[0]['case_id'] for job in jobs}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            row,boot,pred,curve=future.result();models.append(row);bootstrap[row['case_id']]=boot;predictions+=pred;curves+=curve
            print(i,'/',len(jobs),row['case_id'],flush=True)
    models=pd.DataFrame(models).sort_values('case_id');models.to_csv(T/'regression_comparison_models.csv',index=False)
    predictions=pd.DataFrame(predictions);predictions.to_csv(T/'regression_comparison_predictions.csv',index=False)
    pd.DataFrame(curves).to_csv(T/'regression_comparison_curves.csv',index=False)
    np.savez_compressed(T/'regression_comparison_bootstrap.npz',**bootstrap)
    metrics=[]
    for case,q in predictions.groupby('case_id'):
        tests=[('All paired-A targets',np.ones(len(q),bool)),
            ('New small, measured ML1.5–2',q.confirmatory_small&q.ML_measured.ge(1.5)&q.ML_measured.lt(2)),
            ('New small, measured ML>=1.5',q.confirmatory_small),('Measured ML>=4',q.ML_measured.ge(4))]
        for label,mask in tests:
            s=q[mask];family=s.groupby('family_block').error.agg(lambda v:np.mean(v*v))
            metrics.append(dict(case_id=case,cohort=q.cohort.iloc[0],axis=q.axis.iloc[0],method=q.method.iloc[0],degree=q.degree.iloc[0],
                target=label,n_events=len(s),n_families=len(family),RMS_family=float(np.sqrt(family.mean())),
                RMS_events=float(np.sqrt(np.mean(s.error**2))),n_extrapolated=int((~s.in_training_range).sum())))
    pd.DataFrame(metrics).to_csv(T/'regression_comparison_performance.csv',index=False)
    baseline=pd.read_csv(T/'regressions.csv')
    for degree,name in [(1,'linear'),(2,'quadratic')]:
        now=models[(models.cohort=='expanded_NW2')&(models.axis=='our')&(models.method=='OLS')&(models.degree==degree)].iloc[0]
        oldfit=baseline[(baseline.selection=='NW2 primary')&(baseline.method=='family-balanced OLS')&(baseline.model==name)].iloc[0]
        assert np.allclose([now.a,now.b,now.c],[oldfit.a_at_ML2,oldfit.b_at_ML2,oldfit.c],atol=1e-11,rtol=0)
    for path,digest in protected.items():assert sha(R/path)==digest,path
    counts={}
    for name,frame in [('All',new),('Catalog ML<2',new[new.ml.lt(2)]),('Catalog ML>=2',new[new.ml.ge(2)])]:
        counts[name]=dict(attempted=len(frame),ML_measured=int(frame.ML_Sheen.notna().sum()),ML_A=int(frame.ML_quality.eq('A').sum()),
            median_ML_A_minus_KMA=float((frame.loc[frame.ML_quality.eq('A'),'ML_Sheen']-frame.loc[frame.ML_quality.eq('A'),'ml']).median()))
    outputs=['ML_measurement_audit.csv','ML_station_measurements.csv','regression_comparison_models.csv',
        'regression_comparison_predictions.csv','regression_comparison_curves.csv','regression_comparison_bootstrap.npz','regression_comparison_performance.csv']
    validation=dict(passed=True,models=len(models),ML_counts=counts,station_measurements=len(station_audit),
        maximum_station_ML_replay_error=max(e['max_station_replay_error'] for e in event_audit),
        previous_coefficients_reproduced=True,same_event_sets_for_both_ML_definitions=True,
        test_target='Same 676 uniform-NW2 paired-A events; every target family omitted from each training cohort',
        target_ranges='Defined using independently measured ML for both ML definitions',
        sampling_policy='Existing quality selections; no waveform or magnitude changes',
        protected_inputs_sha256=protected,source_sha256=sha(Path(__file__)),
        outputs_sha256={name:sha(T/name) for name in outputs})
    (T/'regression_comparison_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k not in ['protected_inputs_sha256','outputs_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
