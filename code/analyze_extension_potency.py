"""Carry the established Mw/ML/Mp study into the frozen smaller-event cohort.

Read saved moments and amplitudes only. Reuse the original physical conversion,
regression engine and depth controls; keep native and shared target tests apart.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,multiprocessing,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from analyze_potency import convert,LIU_OFFSET_SI,MU_REF_LIU
from analyze_magnitude_followup import depth_case
from potency_model import source_vs,CONFIG_SHA256
from potency_regression import fit,prediction,binned_medians,weights_for,bootstrap_fit
from audit_extension_fit_objectives import center2,center3,paired_gain

R=Path(__file__).resolve().parents[1];E=R/'small_events/extension';T=E/'tables'
RESPONSES={'Mw_Kim1983':'Mw','Mw_Kim2011':'Mw_source_Vs_Kim2011',
    'Mp_Kim1983':'Mp_Liu_Kim1983','Mp_Kim2011':'Mp_Liu_Kim2011'}
OBJECTIVES={'family_OLS':('OLS','family'),'family_ODR':('ODR','family'),
    'event_OLS':('OLS','event'),'event_ODR':('ODR','event'),
    'bin_OLS':('BinnedOLS','event'),'bin_ODR':('BinnedODR','event')}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(p):return pd.read_csv(p,dtype={'event_id':str})


def physical(frame):
    d=frame.copy()
    if 'ml' not in d:d['ml']=d.ML_KMA
    d['ML_KMA']=d.ml
    for model in ['Kim1983','Kim2011']:
        for endpoint in ['lower','upper']:
            beta=source_vs(np.maximum(d[endpoint].to_numpy(),0),model)
            d[f'Mw_at_depth_{endpoint}_{model}']=d.Mw_reference_Vs3500+2*np.log10(beta/3500)
        for side in ['lo','hi']:
            name=f'Mw_station_bootstrap_{side}'
            if name in d:d[name+'_Kim2011']=d[name]+d.Mw_source_Vs_Kim2011-d.Mw
    d=convert(d)
    d['ML_HG_fraction']=d.ML_n_HG/d.ML_Sheen_n
    d['Mw_HG_fraction']=d.n_HG/d.n_stations
    # Mp inherits the spectrum's acceptance; it is not an independent fit.
    d['Mp_quality']=d.Mw_quality
    return d


def estimate(x,y,blocks,spec,edges=None):
    method,weight=OBJECTIVES[spec['objective']]
    bins=spec.get('n_bins',50);minimum=spec.get('minimum',1)
    if method=='BinnedODR':
        med,_=binned_medians(x,y,bins,minimum,edges)
        return fit(med[:,0],med[:,1],spec['degree'],'ODR')[0]
    return fit(x,y,spec['degree'],method,weights_for(blocks,weight),bins,minimum,edges)[0]


def solve(job):
    spec,frame,target,reference=job
    column='ML_Sheen' if spec['axis']=='our' else 'ml';ykey=RESPONSES[spec['response']]
    x=frame[column].to_numpy();y=frame[ykey].to_numpy();blocks=frame.family_block.to_numpy()
    beta=estimate(x,y,blocks,spec);reps=spec['bootstrap_reps']
    if reference is not None:
        saved,samples=reference;assert np.allclose(beta,saved,atol=2e-5,rtol=0),(spec,beta,saved)
        beta=saved
    elif not reps:samples=np.empty((0,3))
    elif spec['objective']!='bin_ODR':
        method,weight=OBJECTIVES[spec['objective']]
        samples=bootstrap_fit(x,y,blocks,dict(method=method,degree=spec['degree'],weighting=weight,
            n_bins=spec['n_bins'],minimum=spec['minimum']),reps)
    else:
        keys,inv=np.unique(blocks,return_inverse=True)
        draws=np.random.default_rng(20261121).multinomial(len(keys),np.full(len(keys),1/len(keys)),size=reps)
        edges=np.linspace(x.min(),x.max(),spec['n_bins']+1);samples=[]
        for draw in draws:
            take=np.repeat(np.arange(len(x)),draw[inv])
            try:samples.append(estimate(x[take],y[take],blocks[take],spec,edges))
            except (ValueError,RuntimeError,np.linalg.LinAlgError):continue
        samples=np.asarray(samples)
    assert not reps or len(samples)>=.9*reps,(spec,len(samples))
    coef=center2(beta);boot=center2(samples) if len(samples) else np.empty((0,3))
    bounds=np.quantile(boot,[.025,.975],axis=0) if len(boot) else np.full((2,3),np.nan)
    med,edges=binned_medians(x,y,spec['n_bins'],spec['minimum'])
    row={**spec,'n_events':len(frame),'n_families':len(np.unique(blocks)),
        'a':coef[0],'b':coef[1],'c':coef[2],'a_lower':bounds[0,0],'a_upper':bounds[1,0],
        'b_lower':bounds[0,1],'b_upper':bounds[1,1],'c_lower':bounds[0,2],'c_upper':bounds[1,2],
        'x_min':x.min(),'x_max':x.max(),'n_successful_bootstraps':len(samples),
        'n_occupied_bins':len(med),'n_singleton_bins':int(sum(med[:,3]==1)),
        'n_trimmed_or_excluded_from_bins':int(len(x)-sum(med[:,3])),
        'reference_reproduced':reference is not None}
    binrows=[dict(case_id=spec['case_id'],bin_index=int(v[4]),ML=v[0],magnitude=v[1],n_initial=int(v[2]),
        n_retained=int(v[3]),left=edges[int(v[4])],right=edges[int(v[4])+1]) for v in med]
    outputs=[];targets={'shared_NW2':target,'native':frame}
    union=np.unique(np.concatenate([d.family_block.to_numpy() for d in targets.values()]))
    folds={}
    for key in union:
        train=blocks!=key
        folds[key]=(beta if train.all() else estimate(x[train],y[train],blocks[train],spec),x[train].min(),x[train].max())
    for label,data in targets.items():
        for event in data.to_dict('records'):
            coef,lo,hi=folds[event['family_block']];xx=event[column]
            outputs.append(dict(case_id=spec['case_id'],target_set=label,event_id=event['event_id'],family_block=event['family_block'],
                ML_measured=event['ML_Sheen'],ML_KMA=event['ml'],observed=event[ykey],
                fitted=float(prediction(beta,[xx])[0]),predicted=float(prediction(coef,[xx])[0]),
                in_training_range=bool(lo<=xx<=hi),confirmatory_small=bool(event.get('confirmatory_small',False))))
    return row,boot,outputs,binrows


def depth_worker(job):
    cohort,frame,x,y,control,case=job
    row,boot,pred=depth_case(frame,x,y,control,case)
    return {**row,'cohort':cohort},boot,pred


def main():
    sources=[Path(__file__),R/'code/analyze_potency.py',R/'code/potency_model.py',R/'code/potency_regression.py',
        R/'code/analyze_magnitude_followup.py',R/'code/audit_extension_fit_objectives.py']
    inputs=[T/'NW2_Mw_ML_catalog.csv',T/'NW4_Mw_ML_catalog.csv',T/'paired_A_MLge1p5.csv',
        T/'Mw_ML_catalog.csv',T/'fit_objective_models.csv',T/'fit_objective_predictions.csv',
        T/'fit_objective_bins.csv',T/'fit_objective_bootstrap.npz',
        R/'tables/Mw_independent_ML_primary.csv',R/'tables/magnitude_followup_regressions.csv',
        R/'tables/magnitude_followup_bootstrap.npz',R/'tables/magnitude_followup_prediction_metrics.csv',
        R/'tables/magnitude_followup_depth_regressions.csv',R/'tables/magnitude_followup_depth_bootstrap.npz',
        R/'tables/potency_event_catalog.csv',R/'tables/Sangju2026_summary.json',
        R/'inputs/KR_Mw/Events_KP_2017_2022_Mw.csv',R/'inputs/source_velocity_models.json']
    inputs += sorted((R/'reports').glob('*.pdf'))
    protected={str(p.relative_to(R)):sha(p) for p in inputs}
    new=physical(csv(T/'NW2_Mw_ML_catalog.csv'));old=physical(csv(T/'NW4_Mw_ML_catalog.csv'))
    new.to_csv(T/'potency_event_catalog.csv',index=False);old.to_csv(T/'potency_NW4_catalog.csv',index=False)
    primary=new[new.paired_A&new.ML_Sheen.ge(1.5)].copy()
    previous=physical(csv(R/'tables/Mw_independent_ML_primary.csv'))
    frames={'previous':previous,'expanded_NW4':old[old.paired_A&old.ML_Sheen.ge(1.5)].copy(),
        'larger_NW2':primary[primary.ml.ge(2)].copy(),'expanded_NW2':primary}
    members=pd.concat([d.assign(cohort=k) for k,d in frames.items()],ignore_index=True)
    members.to_csv(T/'magnitude_scaling_members.csv',index=False)
    legacy=csv(R/'tables/magnitude_followup_regressions.csv');legacy_boot=np.load(R/'tables/magnitude_followup_bootstrap.npz')
    existing=csv(T/'fit_objective_models.csv');existing_pred=csv(T/'fit_objective_predictions.csv')
    existing_bins=csv(T/'fit_objective_bins.csv');existing_boot=np.load(T/'fit_objective_bootstrap.npz')
    rows=[];predictions=[];bins=[];boots={};jobs=[]
    for cohort,frame in frames.items():
        for response,ykey in RESPONSES.items():
            for axis in ['our','KMA']:
                for objective,(method,weight) in OBJECTIVES.items():
                    for degree in [1,2]:
                        cid=f'{response}_{cohort}_{axis}_{objective}_{degree}'
                        spec=dict(case_id=cid,cohort=cohort,response=response,axis=axis,objective=objective,degree=degree,
                            n_bins=50,minimum=1,bootstrap_reps=2000 if method=='OLS' else 1000,selection='Main methods')
                        if response=='Mw_Kim1983' and objective!='event_ODR':
                            oldid=f'{cohort}_{axis}_{objective}_{degree}';saved=existing[existing.case_id.eq(oldid)].iloc[0]
                            row={**saved.to_dict(),**spec,'n_successful_bootstraps':len(existing_boot[oldid]),
                                'a_lower':np.quantile(existing_boot[oldid][:,0],.025),'a_upper':np.quantile(existing_boot[oldid][:,0],.975),
                                'b_lower':np.quantile(existing_boot[oldid][:,1],.025),'b_upper':np.quantile(existing_boot[oldid][:,1],.975)}
                            rows.append(row);boots[cid]=existing_boot[oldid]
                            pp=existing_pred[existing_pred.case_id.eq(oldid)].rename(columns={'Mw':'observed'}).copy();pp['case_id']=cid
                            predictions+=pp.to_dict('records')
                            bb=existing_bins[existing_bins.case_id.eq(oldid)].rename(columns={'ML_median':'ML','Mw_median':'magnitude'}).copy();bb['case_id']=cid
                            bins+=bb.to_dict('records');continue
                        ref=None
                        if cohort=='previous' and objective!='bin_ODR':
                            q=legacy[legacy.x.eq('ML_Sheen' if axis=='our' else 'ML_KMA')&legacy.y.eq(ykey)&legacy.method.eq(method)&
                                legacy.weighting.eq(weight)&legacy.degree.eq(degree)&legacy.n_bins.eq(50)&legacy.minimum.eq(1)]
                            assert len(q)==1;q=q.iloc[0]
                            ref=(np.array([q.a_at_ML3,q.slope_at_ML3,q.quadratic_c]),legacy_boot[q.case_id])
                        jobs.append((spec,frame,primary,ref))
    # Match the earlier bin-count and minimum-support controls; these are point/CV sensitivities.
    for cohort in ['previous','expanded_NW2']:
        for response in RESPONSES:
            for axis in ['our','KMA']:
                for nb,minimum in [(10,1),(20,1),(50,3)]:
                    for degree in [1,2]:
                        spec=dict(case_id=f'{response}_{cohort}_{axis}_bins{nb}_min{minimum}_{degree}',cohort=cohort,
                            response=response,axis=axis,objective='bin_OLS',degree=degree,n_bins=nb,minimum=minimum,
                            bootstrap_reps=0,selection='Bin support sensitivity')
                        jobs.append((spec,frames[cohort],primary,None))
    print('Reuse',len(rows),'Mw fits; calculate',len(jobs),'matched magnitude-method cases',flush=True)
    with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('fork')) as pool:
        pending=[pool.submit(solve,job) for job in jobs]
        for i,future in enumerate(as_completed(pending),1):
            row,boot,pp,bb=future.result();rows.append(row);predictions+=pp;bins+=bb
            if len(boot):boots[row['case_id']]=boot
            if i%20==0 or i==len(jobs):print('Magnitude cases',i,'/',len(jobs),flush=True)
    models=pd.DataFrame(rows).sort_values('case_id');pred=pd.DataFrame(predictions)
    meta=models[['case_id','cohort','response','axis','objective','degree','selection','n_bins','minimum']]
    pred=pred.drop(columns=[c for c in meta.columns if c!='case_id' and c in pred]).merge(meta,on='case_id',validate='many_to_one')
    models.to_csv(T/'magnitude_scaling_regressions.csv',index=False)
    pred.to_csv(T/'magnitude_scaling_predictions.csv',index=False)
    pd.DataFrame(bins).to_csv(T/'magnitude_scaling_bins.csv',index=False);np.savez_compressed(T/'magnitude_scaling_bootstrap.npz',**boots)
    metrics=[];gains=[]
    for (cid,label),q in pred.groupby(['case_id','target_set']):
        for subset,mask in [('All',np.ones(len(q),bool)),('Measured ML>4',q.ML_measured.gt(4)),
            ('New measured ML1.5–2',q.confirmatory_small&q.ML_measured.ge(1.5)&q.ML_measured.lt(2))]:
            s=q[mask]
            if s.empty:continue
            fr=s.observed-s.fitted;cr=s.observed-s.predicted
            row={k:q[k].iloc[0] for k in meta.columns};row.update(target_set=label,subset=subset,n_events=len(s),
                n_families=s.family_block.nunique(),fit_RMS_events=float(np.sqrt(np.mean(fr*fr))),
                CV_RMS_events=float(np.sqrt(np.mean(cr*cr))),
                CV_RMS_families=float(np.sqrt(pd.Series((cr*cr).to_numpy()).groupby(s.family_block.to_numpy()).mean().mean())),
                n_extrapolated=int(sum(~s.in_training_range)))
            metrics.append(row)
            if int(q.degree.iloc[0])!=2:continue
            lineid=cid[:-1]+'1';line=pred[pred.case_id.eq(lineid)&pred.target_set.eq(label)].set_index('event_id')
            loss=s[['event_id','family_block']].copy();loss['quad_error']=cr.to_numpy()
            loss['line_error']=s.observed.to_numpy()-line.loc[s.event_id,'predicted'].to_numpy()
            gains.append({**{k:row[k] for k in meta.columns},'target_set':label,'subset':subset,**paired_gain(loss)})
    metrics=pd.DataFrame(metrics);metrics.to_csv(T/'magnitude_scaling_performance.csv',index=False)
    pd.DataFrame(gains).to_csv(T/'magnitude_scaling_prediction_gains.csv',index=False)
    print('Magnitude scaling complete; repeat the established depth controls',flush=True)
    depthjobs=[];drows=[];dboot={};dpred=[]
    for response,y in RESPONSES.items():
        for x in ['ML_Sheen','ML_KMA']:
            controls=['Raw magnitude difference','ML linear','ML quadratic','ML quadratic with depth hinge']
            if x=='ML_Sheen':controls+=['Geographic plane','Station distance and sensors','One-degree spatial cells',
                'Half-degree spatial cells','Without Gyeongju/Pohang','Depth below 15 km','Depth at least 15 km',
                'Layer intercepts','Lower depth endpoints','Upper depth endpoints']
            for control in controls:
                cid=f'depth_{response}_{x}_{len(depthjobs):03d}'
                depthjobs.append(('expanded_NW2',primary,x,y,control,cid))
            for control in ['Raw magnitude difference','ML linear','ML quadratic']:
                cid=f'depth_larger_{response}_{x}_{len(depthjobs):03d}'
                depthjobs.append(('larger_NW2',frames['larger_NW2'],x,y,control,cid))
    # Fixed-Vs quantities isolate the inherited source-coefficient contribution.
    for x in ['ML_Sheen','ML_KMA']:
        for y in ['Mp_Liu_fixed_Vs3500','Mw_reference_Vs3500']:
            for control in ['Raw magnitude difference','ML linear','ML quadratic']:
                cid=f'depth_fixed_{len(depthjobs):03d}';depthjobs.append(('expanded_NW2',primary,x,y,control,cid))
    with ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('fork')) as pool:
        for row,boot,pp in pool.map(depth_worker,depthjobs):
            drows.append(row);dboot[row['case_id']]=boot;dpred+=pp
    pd.DataFrame(drows).to_csv(T/'potency_depth_regressions.csv',index=False)
    pd.DataFrame(dpred).to_csv(T/'potency_depth_predictions.csv',index=False);np.savez_compressed(T/'potency_depth_bootstrap.npz',**dboot)
    finish(new,old,primary,frames,models,metrics,protected,sources,len(depthjobs))


def finish(new,old,primary,frames,models,metrics,protected,sources,n_depth):
    # Published Hong moments are converted under Hong's own source constants.
    hong=csv(R/'inputs/KR_Mw/Events_KP_2017_2022_Mw.csv').rename(columns={'evid':'event_id'})
    hong['event_id']=hong.event_id.astype(str).str.strip()
    for src,label in [('mw_spectra','spectral'),('mw_TD','TD')]:
        hong['Mp_Hong_'+label]=(1.5*hong[src]+9.05-np.log10(2700*3610**2)+LIU_OFFSET_SI)/1.5
    matched=new.merge(hong,on='event_id',suffixes=('','_Hong'),validate='one_to_one')
    matched.to_csv(T/'potency_Hong_matches.csv',index=False)
    stats=[]
    for sample,q in [('All matches',matched),('Mw A',matched[matched.Mw_quality.eq('A')]),('Paired A',matched[matched.paired_A&matched.ML_Sheen.ge(1.5)])]:
        for response,column in RESPONSES.items():
            for label,hcolumn in [('spectral','mw_spectra'),('TD','mw_TD')]:
                if response.startswith('Mp'):hcolumn='Mp_Hong_'+label
                part=q.dropna(subset=[column,hcolumn]);delta=part[column]-part[hcolumn]
                stats.append(dict(sample=sample,response=response,Hong_method=label,n=len(part),median=float(delta.median()),
                    mean=float(delta.mean()),RMS=float(np.sqrt(np.mean(delta*delta)))))
    pd.DataFrame(stats).to_csv(T/'potency_Hong_statistics.csv',index=False)
    # Verify all repeated earlier native-target cases, including potency and the second model.
    earlier=csv(R/'tables/magnitude_followup_regressions.csv')
    errors=[]
    for case in models[models.cohort.eq('previous')&models.objective.ne('bin_ODR')].itertuples():
        method,weight=OBJECTIVES[case.objective]
        q=earlier[earlier.x.eq('ML_Sheen' if case.axis=='our' else 'ML_KMA')&earlier.y.eq(RESPONSES[case.response])&
            earlier.method.eq(method)&earlier.weighting.eq(weight)&earlier.degree.eq(case.degree)&
            earlier.n_bins.eq(case.n_bins)&earlier.minimum.eq(case.minimum)]
        assert len(q)==1;q=q.iloc[0]
        assert np.allclose([case.a,case.b,case.c],center2([q.a_at_ML3,q.slope_at_ML3,q.quadratic_c]),atol=2e-5,rtol=0)
        score=metrics[metrics.case_id.eq(case.case_id)&metrics.target_set.eq('native')&metrics.subset.eq('All')].iloc[0]
        err=abs(score.CV_RMS_events-q.CV_RMS_events);assert err<2e-5,(case.case_id,err);errors.append(err)
    sensitivity={}
    for name,delta in [('Mw',primary.Mw_source_Vs_Kim2011-primary.Mw),('Mp',primary.Mp_Liu_Kim2011-primary.Mp_Liu_Kim1983)]:
        sensitivity[name]=dict(median_abs=float(abs(delta).median()),max_abs=float(abs(delta).max()))
    for name,digest in protected.items():assert sha(R/name)==digest,name
    summary=dict(complete=True,attempted=len(new),finite_Mw=int(new.Mw.notna().sum()),finite_Mp=int(new.Mp_Liu_Kim1983.notna().sum()),
        primary_events=len(primary),primary_families=primary.family_block.nunique(),regression_cases=len(models),
        main_method_cases=int(models.selection.eq('Main methods').sum()),depth_control_cases=n_depth,
        previous_native_cases_reproduced=len(errors),maximum_previous_CV_error=max(errors),
        source_velocity_sensitivity=sensitivity,source_velocity_config_sha256=CONFIG_SHA256,
        definitions=dict(P0='M0/(rho*Vs(z)^2), m3',Mp_Liu='(2/3)*(log10(P0[m3])+1.4563)',
            density_kg_m3=2800,primary_model='Kim1983',comparison_model='Kim2011',
            equivalent_Liu_reference_rigidity_Pa=MU_REF_LIU,Mw_constant=9.05,
            Mp_alternatives='Mpo28_C9p10, Mpo28_C9p05, Mp36_C9p05; these constant shifts do not change free-fit curvature or depth slopes'),
        scope='Scalar source potency; Mp quality inherits Mw spectral quality. No new waveform processing, relocation or effective-path-rigidity Mpv inversion.')
    (T/'potency_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    outputs=['potency_event_catalog.csv','potency_NW4_catalog.csv','magnitude_scaling_members.csv','magnitude_scaling_regressions.csv',
        'magnitude_scaling_predictions.csv','magnitude_scaling_bins.csv','magnitude_scaling_bootstrap.npz',
        'magnitude_scaling_performance.csv','magnitude_scaling_prediction_gains.csv','potency_depth_regressions.csv',
        'potency_depth_predictions.csv','potency_depth_bootstrap.npz','potency_Hong_matches.csv','potency_Hong_statistics.csv','potency_summary.json']
    result=dict(passed=True,protected_inputs_sha256=protected,sources_sha256={str(p.relative_to(R)):sha(p) for p in sources},
        outputs_sha256={n:sha(T/n) for n in outputs},previous_native_cases_reproduced=len(errors),summary=summary)
    (T/'potency_analysis_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
