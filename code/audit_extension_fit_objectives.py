"""Reproduce the earlier fitting methods and diagnose magnitude-weight dilution.

All inputs are saved measurements. No magnitude or quality selection is changed.
Bin OLS repeats the earlier 50-bin implementation; bin ODR is an added check.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import multiprocessing
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from potency_regression import fit, prediction, weights_for, binned_medians, bootstrap_fit

R=Path(__file__).resolve().parents[1];E=R/'small_events/extension';T=E/'tables'
OBJECTIVES={'family_OLS':('OLS','family'),'family_ODR':('ODR','family'),
    'event_OLS':('OLS','event'),'bin_OLS':('BinnedOLS','event'),'bin_ODR':('BinnedODR','event')}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csv(path):return pd.read_csv(path,dtype={'event_id':str})
def center2(q):
    q=np.asarray(q);return np.stack([q[...,0]-q[...,1]+q[...,2],q[...,1]-2*q[...,2],q[...,2]],axis=-1)
def center3(q):
    q=np.asarray(q);return np.stack([q[...,0]+q[...,1]+q[...,2],q[...,1]+2*q[...,2],q[...,2]],axis=-1)


def estimate(x,y,blocks,objective,degree,edges=None):
    method,weight=OBJECTIVES[objective]
    if method=='BinnedODR':
        bins,_=binned_medians(x,y,50,1,edges)
        return fit(bins[:,0],bins[:,1],degree,'ODR')[0]
    return fit(x,y,degree,method,weights_for(blocks,weight),50,1,edges)[0]


def calculate(job):
    spec,frame,target,reference=job
    objective=spec['objective'];degree=spec['degree'];column='ML_Sheen' if spec['axis']=='our' else 'ml'
    x=frame[column].to_numpy();y=frame.Mw.to_numpy();blocks=frame.family_block.to_numpy()
    beta=estimate(x,y,blocks,objective,degree)
    if reference is not None:
        saved,samples=reference
        assert np.allclose(beta,saved,atol=2e-5,rtol=0),(spec,beta,saved)
        beta=saved
    else:
        method,weight=OBJECTIVES[objective]
        if method!='BinnedODR':
            samples=bootstrap_fit(x,y,blocks,dict(method=method,degree=degree,weighting=weight,n_bins=50,minimum=1),1000)
        else:
            keys,inverse=np.unique(blocks,return_inverse=True)
            counts=np.random.default_rng(20261121).multinomial(len(keys),np.full(len(keys),1/len(keys)),size=1000)
            edges=np.linspace(x.min(),x.max(),51);samples=[]
            for draw in counts:
                take=np.repeat(np.arange(len(x)),draw[inverse])
                try:samples.append(estimate(x[take],y[take],blocks[take],objective,degree,edges))
                except (ValueError,RuntimeError,np.linalg.LinAlgError):continue
            samples=np.asarray(samples)
    assert len(samples)>=900,(spec,len(samples))
    converted=center2(beta);boot=center2(samples);lo,hi=np.quantile(boot,[.025,.975],axis=0)
    bins,edges=binned_medians(x,y,50)
    row={**spec,'n_events':len(frame),'n_families':len(np.unique(blocks)),
        'a':converted[0],'b':converted[1],'c':converted[2],'c_lower':lo[2],'c_upper':hi[2],
        'x_min':x.min(),'x_max':x.max(),'n_bootstraps':len(samples),'n_occupied_bins':len(bins),
        'n_singleton_bins':int(sum(bins[:,3]==1)),'n_bin_trimmed_events':int(sum(bins[:,2]-bins[:,3])),
        'previous_case_reproduced':reference is not None}
    bin_rows=[dict(case_id=spec['case_id'],bin_index=int(v[4]),ML_median=v[0],Mw_median=v[1],
        n_initial=int(v[2]),n_retained=int(v[3]),left=edges[int(v[4])],right=edges[int(v[4])+1]) for v in bins]
    targets={'shared_NW2':target,'native':frame}
    outputs=[]
    all_keys=np.unique(np.concatenate([d.family_block.to_numpy() for d in targets.values()]))
    held={}
    for family in all_keys:
        train=blocks!=family
        held[family]=(beta if train.all() else estimate(x[train],y[train],blocks[train],objective,degree),
            x[train].min(),x[train].max())
    for label,data in targets.items():
        for event in data.to_dict('records'):
            coef,lbound,ubound=held[event['family_block']];xx=event[column]
            outputs.append(dict(case_id=spec['case_id'],cohort=spec['cohort'],axis=spec['axis'],
                objective=objective,degree=degree,target_set=label,event_id=event['event_id'],
                family_block=event['family_block'],ML_measured=event['ML_Sheen'],ML_KMA=event['ml'],
                Mw=event['Mw'],fitted=float(prediction(beta,[xx])[0]),
                predicted=float(prediction(coef,[xx])[0]),in_training_range=bool(lbound<=xx<=ubound),
                confirmatory_small=bool(event.get('confirmatory_small',False))))
    return row,boot,outputs,bin_rows


def paired_gain(part):
    mse=part.groupby('family_block')[['line_error','quad_error']].agg(lambda x:np.mean(x*x)).to_numpy()
    n=len(mse)
    draws=(np.array(list(itertools.product(range(n),repeat=n))) if n<=6 else
        np.random.default_rng(20261123).integers(n,size=(2500,n)))
    gain=np.sqrt(mse[draws,0].mean(1))-np.sqrt(mse[draws,1].mean(1))
    return dict(gain=float(np.sqrt(mse[:,0].mean())-np.sqrt(mse[:,1].mean())),
        gain_lower=float(np.quantile(gain,.025)),gain_upper=float(np.quantile(gain,.975)))


def main():
    inputs=[T/'NW2_Mw_ML_catalog.csv',T/'NW4_Mw_ML_catalog.csv',T/'paired_A_MLge1p5.csv',
        T/'regression_comparison_models.csv',T/'regression_comparison_bootstrap.npz',
        R/'tables/Mw_independent_ML_primary.csv',R/'tables/magnitude_followup_regressions.csv',
        R/'tables/magnitude_followup_bootstrap.npz',R/'tables/magnitude_followup_prediction_metrics.csv',
        R/'reports/Potency_magnitude_and_quadratic_scaling.pdf',R/'code/potency_regression.py']
    protected={str(p.relative_to(R)):sha(p) for p in inputs}
    previous=csv(inputs[5]);primary=csv(inputs[2]);old=csv(inputs[1])
    frames={'previous':previous,'expanded_NW4':old[old.paired_A&old.ML_Sheen.ge(1.5)],
        'larger_NW2':primary[primary.ml.ge(2)],'expanded_NW2':primary}
    current=csv(T/'regression_comparison_models.csv');current_boot=np.load(T/'regression_comparison_bootstrap.npz')
    historical=csv(R/'tables/magnitude_followup_regressions.csv');old_boot=np.load(R/'tables/magnitude_followup_bootstrap.npz')
    jobs=[];weight_rows=[]
    for cohort,frame in frames.items():
        family_weight=weights_for(frame.family_block.to_numpy(),'family')
        for event,w in zip(frame.to_dict('records'),family_weight):
            weight_rows.append(dict(cohort=cohort,event_id=event['event_id'],family_block=event['family_block'],
                ML_measured=event['ML_Sheen'],Mw=event['Mw'],family_members=round(1/w),
                family_weight=w,normalized_family_weight=w/family_weight.sum()))
        for axis in ['our','KMA']:
            for objective,(method,weight) in OBJECTIVES.items():
                for degree in [1,2]:
                    spec=dict(case_id=f'{cohort}_{axis}_{objective}_{degree}',cohort=cohort,axis=axis,objective=objective,degree=degree)
                    reference=None
                    if objective in ['family_OLS','family_ODR']:
                        saved=current[current.cohort.eq(cohort)&current.axis.eq(axis)&current.method.eq(method)&current.degree.eq(degree)].iloc[0]
                        reference=(center3([saved.a,saved.b,saved.c]),center3(current_boot[saved.case_id]))
                    elif cohort=='previous' and objective!='bin_ODR':
                        saved=historical[historical.x.eq('ML_Sheen' if axis=='our' else 'ML_KMA')&historical.y.eq('Mw')&
                            historical.method.eq(method)&historical.weighting.eq(weight)&historical.degree.eq(degree)&
                            historical.n_bins.eq(50)&historical.minimum.eq(1)].iloc[0]
                        reference=(np.array([saved.a_at_ML3,saved.slope_at_ML3,saved.quadratic_c]),old_boot[saved.case_id])
                    jobs.append((spec,frame,primary,reference))
    models=[];predictions=[];bins=[];bootstrap={}
    print('Audit',len(jobs),'fit specifications; historical and family fits are reused after point reproduction',flush=True)
    with ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('fork')) as pool:
        pending=[pool.submit(calculate,job) for job in jobs]
        for i,future in enumerate(as_completed(pending),1):
            row,samples,pred,medians=future.result();models.append(row);bootstrap[row['case_id']]=samples
            predictions+=pred;bins+=medians
            print(i,'/',len(jobs),row['case_id'],flush=True)
    models=pd.DataFrame(models).sort_values('case_id');pred=pd.DataFrame(predictions)
    pd.DataFrame(weight_rows).to_csv(T/'fit_objective_weights.csv',index=False)
    models.to_csv(T/'fit_objective_models.csv',index=False);pred.to_csv(T/'fit_objective_predictions.csv',index=False)
    pd.DataFrame(bins).to_csv(T/'fit_objective_bins.csv',index=False);np.savez_compressed(T/'fit_objective_bootstrap.npz',**bootstrap)
    metrics=[];gains=[]
    for (case,label),q in pred.groupby(['case_id','target_set']):
        for subset,mask in [('All',np.ones(len(q),bool)),('Measured ML>4',q.ML_measured.gt(4)),
            ('New measured ML1.5–2',q.confirmatory_small&q.ML_measured.ge(1.5)&q.ML_measured.lt(2))]:
            s=q[mask]
            if s.empty:continue
            fe=s.Mw-s.fitted;ce=s.Mw-s.predicted
            metrics.append(dict(case_id=case,cohort=q.cohort.iloc[0],axis=q.axis.iloc[0],objective=q.objective.iloc[0],
                degree=q.degree.iloc[0],target_set=label,subset=subset,n_events=len(s),n_families=s.family_block.nunique(),
                fit_RMS_events=float(np.sqrt(np.mean(fe*fe))),CV_RMS_events=float(np.sqrt(np.mean(ce*ce))),
                CV_RMS_families=float(np.sqrt(pd.Series((ce*ce).to_numpy()).groupby(s.family_block.to_numpy()).mean().mean())),
                n_extrapolated=int(sum(~s.in_training_range))))
            if q.degree.iloc[0]!=2:continue
            line=pred[pred.cohort.eq(q.cohort.iloc[0])&pred.axis.eq(q.axis.iloc[0])&pred.objective.eq(q.objective.iloc[0])&
                pred.degree.eq(1)&pred.target_set.eq(label)].set_index('event_id')
            loss=s[['family_block','event_id']].copy();loss['quad_error']=ce.to_numpy()
            loss['line_error']=s.Mw.to_numpy()-line.loc[s.event_id,'predicted'].to_numpy()
            gains.append(dict(case_id=case,target_set=label,subset=subset,n_families=s.family_block.nunique(),**paired_gain(loss)))
    metrics=pd.DataFrame(metrics);metrics.to_csv(T/'fit_objective_performance.csv',index=False)
    pd.DataFrame(gains).to_csv(T/'fit_objective_prediction_gains.csv',index=False)
    # Restore the earlier paper's native-target scores before comparing new targets.
    for m in models[models.cohort.eq('previous')&models.objective.ne('bin_ODR')].itertuples():
        method,weight=OBJECTIVES[m.objective]
        saved=historical[historical.x.eq('ML_Sheen' if m.axis=='our' else 'ML_KMA')&historical.y.eq('Mw')&
            historical.method.eq(method)&historical.weighting.eq(weight)&historical.degree.eq(m.degree)&
            historical.n_bins.eq(50)&historical.minimum.eq(1)].iloc[0]
        now=metrics[metrics.case_id.eq(m.case_id)&metrics.target_set.eq('native')&metrics.subset.eq('All')].iloc[0]
        assert abs(now.CV_RMS_events-saved.CV_RMS_events)<2e-5,(m.case_id,now.CV_RMS_events,saved.CV_RMS_events)
    for name,digest in protected.items():assert sha(R/name)==digest,name
    outputs=['fit_objective_'+name for name in ['weights.csv','models.csv','predictions.csv','bins.csv','bootstrap.npz','performance.csv','prediction_gains.csv']]
    validation=dict(passed=True,n_models=len(models),earlier_methods_and_native_CV_reproduced=True,
        previous_headline_method='Family OLS; 1/n_family_members per event',
        omitted_previous_alternatives='Equal-event OLS and 50-bin OLS',
        additional_method='Equal-axis ODR on the same 50-bin medians',
        bins='50 equal bins over training ML range; trim |Mw - initial bin median|>0.5; recompute ML/Mw medians; equal weight per occupied bin',
        bootstrap='Whole-family draws; previous bootstrap samples reused; bin edges fixed within a bootstrap, matching earlier implementation',
        CV='Hold out the entire target family; recompute training bins/trim/weights; native and shared NW2 target scores recorded separately',
        source_sha256=sha(Path(__file__)),engine_sha256=sha(R/'code/potency_regression.py'),
        protected_inputs_sha256=protected,outputs_sha256={name:sha(T/name) for name in outputs})
    (T/'fit_objective_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('All fitting-objective and historical-reproduction checks passed',flush=True)


if __name__=='__main__':main()
