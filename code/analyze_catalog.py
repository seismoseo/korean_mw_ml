"""Quality-screen and compare independently computed spectral Mw with KMA ML."""
from pathlib import Path
import hashlib,json,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from spectra import fit,fit_stack
from source_velocity import (CONFIG,PRIMARY_MODEL,CONFIG_SHA256,normalize_fit,metadata,
                             source_vs,log_moment_coefficient,validate_catalog)
HERE=Path(__file__).resolve().parents[1]
cohort=pd.read_csv(HERE/'inputs/cohort.csv',dtype={'event_id':str})
rules=json.loads((HERE/'inputs/analysis_quality_rules.json').read_text())
bad={(x['station'],x['sensor']) for x in rules['known_response_exclusions']}
paths={p.stem:p for p in (HERE/'events').glob('*.json')}
missing=set(cohort.event_id)-set(paths)
if missing:raise RuntimeError(f'{len(missing)} events still await processing')
signatures={json.loads(paths[e].read_text()).get('config_sha256') for e in cohort.event_id}
if len(signatures)!=1:raise RuntimeError('Event processing signatures are not all identical')

allfits=[];entries=[];status=[];bootrows=[];staall=[]
def stack(arr):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(arr,axis=0)
def conditional_bootstrap(f,arr,result,seed):
    good=np.isfinite(arr[:,0,:]).sum(axis=0)>=max(3,.3*len(arr))
    phi=-.5*np.log10(1+(f/result['fc_Hz'])**(2*result['falloff']))
    rng=np.random.default_rng(seed);indices=rng.integers(len(arr),size=(300,len(arr)))
    av=stack(arr[indices,0,:].transpose(1,0,2))
    # av: bootstrap x frequency, station-resampled log means, fixed source shape.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        om=np.nanmean((av-phi)[:,good],axis=1)
    mw=(om+log_moment_coefficient(result['source_depth_km'])-9.05)/1.5
    return float(np.nanquantile(mw,.025)),float(np.nanquantile(mw,.975))

for case in cohort.to_dict('records'):
    eid=case['event_id'];raw=json.loads(paths[eid].read_text())
    fits=[normalize_fit(x,case['z_central']) for x in raw.get('fits',[])]
    status.append(dict(event_id=eid,status=raw.get('status'),raw_usable_triplets=raw.get('raw_usable_triplets',0),rejections=json.dumps(raw.get('rejections',{}))))
    npz=HERE/'spectra'/(eid+'.npz');clean={};meta=None
    if (HERE/'station_logs'/(eid+'.csv')).exists():
        meta=pd.read_csv(HERE/'station_logs'/(eid+'.csv'));staall.append(meta)
    if npz.exists():
        b=np.load(npz);f=b['frequency_Hz']
        for duration in [10.1,20.1]:
            tag=f'T{duration:g}'
            if tag not in b:continue
            names=b[tag+'_stations'];sensors=b[tag+'_sensor']
            keep=np.array([(str(s),str(k)) not in bad for s,k in zip(names,sensors)])
            clean[duration]=(b[tag][keep],names[keep],sensors[keep])
            if not keep.all():
                fits=[x for x in fits if x['window_s']!=duration]
                for sensor_set in ['ALL','HH_EL','HG','HH']:
                    ix=np.isin(sensors[keep],{'ALL':['HH','EL','HG'],'HH_EL':['HH','EL'],'HG':['HG'],'HH':['HH']}[sensor_set])
                    rr,_=fit_stack(f,b[tag][keep][ix],depth_km=case['z_central'])
                    for row in rr:fits.append(dict(event_id=eid,window_s=duration,sensor_set=sensor_set,**row))
    allfits+=fits
    lookup={(x['window_s'],x['sensor_set'],x['method'],x['shape']):x for x in fits}
    main=lookup.get((10.1,'ALL','Multitaper','Boatwright'))
    if main is None or not np.isfinite(main.get('Mw',np.nan)):
        entries.append(dict(**case,**metadata(case['z_central']),Mw=np.nan,Mw_quality='F',Mw_reasons='No usable primary spectral fit'));continue
    row={**case,**main};row['ML']=case['ml']
    arr10,names10,sensors10=clean[10.1]
    row['n_stations_SNR']=int((np.isfinite(arr10[:,0,:]).sum(axis=1)>=20).sum())
    for method,shape,label in [('Direct Fourier','Boatwright','Fourier'),('Multitaper','Brune','Brune')]:
        x=lookup.get((10.1,'ALL',method,shape),{})
        row['Mw_'+label]=x.get('Mw',np.nan)
        row['Mw_'+label+'_Kim2011']=x.get('Mw_source_Vs_Kim2011',np.nan)
        row['difference_'+label]=abs(row['Mw_'+label]-row['Mw'])
    row['Mw_velocity']=lookup.get((10.1,'HH_EL','Multitaper','Boatwright'),{}).get('Mw',np.nan)
    row['Mw_HG']=lookup.get((10.1,'HG','Multitaper','Boatwright'),{}).get('Mw',np.nan)
    for sensor in ['HH','EL','HG']:row['n_'+sensor]=int((sensors10==sensor).sum())
    row['Mw_HH']=lookup.get((10.1,'HH','Multitaper','Boatwright'),{}).get('Mw',np.nan)
    row['Mw_T20']=lookup.get((20.1,'ALL','Multitaper','Boatwright'),{}).get('Mw',np.nan)
    row['common_window_stations']=0;row['window_difference_common']=np.nan
    if 10.1 in clean and 20.1 in clean:
        a,n,s=clean[10.1];aa,nn,ss=clean[20.1]
        useful10={(str(n[i]),str(s[i])) for i in range(len(n)) if np.isfinite(a[i,0,:]).sum()>=20}
        useful20={(str(nn[i]),str(ss[i])) for i in range(len(nn)) if np.isfinite(aa[i,0,:]).sum()>=20}
        common=sorted(useful10&useful20);row['common_window_stations']=len(common)
        if len(common)>=5:
            means=[]
            for duration,(arr,names,sensors) in [(10.1,clean[10.1]),(20.1,clean[20.1])]:
                keys=list(zip(names.astype(str),sensors.astype(str)));ids=[keys.index(c) for c in common];q=arr[ids]
                good=np.isfinite(q[:,0,:]).sum(axis=0)>=max(3,.3*len(q))
                res,_=fit(f,stack(q[:,0,:]),good,depth_km=case['z_central'])
                means.append(res.get('Mw',np.nan))
            row['Mw_T10_common'],row['Mw_T20_common']=means
            row['window_difference_common']=abs(means[1]-means[0])
    reasons=[];quality='C'
    for label in ['A','B']:
        r=rules['quality_'+label];fails=[]
        if row['n_stations_SNR']<r['minimum_stations']:fails.append('few_stations')
        if not row['plateau_observed']:fails.append('plateau_extrapolated')
        if row['common_window_stations']<r['minimum_common_window_stations']:fails.append('few_common_window_stations')
        for col,threshold,name in [('difference_Fourier','maximum_estimator_difference_Mw','estimator_sensitivity'),
            ('difference_Brune','maximum_source_shape_difference_Mw','source_shape_sensitivity'),
            ('window_difference_common','maximum_common_station_window_difference_Mw','window_sensitivity')]:
            if not np.isfinite(row[col]) or row[col]>r[threshold]:fails.append(name)
        if row['rms_log10']>r['maximum_fit_RMS_log10']:fails.append('poor_source_fit')
        if not fails:quality=label;break
        if label=='A':reasons=fails
    row['Mw_quality']=quality;row['Mw_reasons']='' if quality=='A' else ';'.join(reasons)
    row['Mw_minus_ML']=row['Mw']-row['ML']
    row['family_block']=f'family_{int(case["family"])}' if case['family']>=0 else 'event_'+eid
    if meta is not None:
        ready=meta[(meta.status=='ready')&(meta.window_s==10.1)].copy()
        ready=ready[[((str(x.station),str(x.sensor)) not in bad) for x in ready.itertuples()]]
        row['master_postissue_stations']=int(ready.master_postissue.fillna(False).astype(bool).sum())
        row['all_master_epochs_confirmed_by_issue']=row['master_postissue_stations']==0
        row['max_response_gain_mismatch']=float(np.nanmax(abs(np.log10(ready[['response_gain_ratio_min','response_gain_ratio_max']].to_numpy()))))
    if quality in ['A','B'] and 10.1 in clean:
        arr,names,sensors=clean[10.1]
        lo,hi=conditional_bootstrap(f,arr,row,int(eid[-6:]))
        row['Mw_station_bootstrap_lo']=lo;row['Mw_station_bootstrap_hi']=hi
        shift=row['Mw_source_Vs_Kim2011']-row['Mw']
        row['Mw_station_bootstrap_lo_Kim2011']=lo+shift
        row['Mw_station_bootstrap_hi_Kim2011']=hi+shift
        # Propagate distance and source-layer velocity through the frozen depth
        # envelope. Observed travel times, source shape, and other path terms stay fixed.
        ready=ready.set_index('station');r_ep=np.array([ready.loc[str(n),'epicentral_km'] for n in names])
        rc=np.hypot(r_ep,case['z_central'])
        for endpoint,depth in [('lower',case['lower']),('upper',case['upper'])]:
            adjust=np.log10(np.hypot(r_ep,max(0,depth))/rc)
            y=stack(arr[:,0,:]+adjust[:,None]);good=np.isfinite(arr[:,0,:]).sum(axis=0)>=max(3,.3*len(arr))
            phi=-.5*np.log10(1+(f/row['fc_Hz'])**(2*row['falloff']))
            om=float(np.nanmean((y-phi)[good]))
            row['Mw_distance_only_at_depth_'+endpoint]=(om+log_moment_coefficient(case['z_central'])-9.05)/1.5
            for model in CONFIG['models']:
                row['Mw_at_depth_'+endpoint+'_'+model]=(om+log_moment_coefficient(max(0,depth),model)-9.05)/1.5
            row['Mw_at_depth_'+endpoint]=row['Mw_at_depth_'+endpoint+'_'+PRIMARY_MODEL]
    for model in CONFIG['models']:
        row['source_layer_boundary_crossed_'+model]=source_vs(max(0,case['lower']),model)!=source_vs(case['upper'],model)
    entries.append(row)

out=pd.DataFrame(entries)
validate_catalog(out)
out.to_csv(HERE/'tables/Mw_ML_catalog.csv',index=False)
pd.DataFrame(allfits).to_csv(HERE/'tables/clean_spectral_fits.csv',index=False)
pd.DataFrame(status).to_csv(HERE/'tables/processing_status.csv',index=False)
pd.concat(staall,ignore_index=True).to_csv(HERE/'tables/all_station_checks.csv.gz',index=False)

# Descriptive regression and a family-balanced regression. Neither tunes the moments.
reg=[];draws=[];rng=np.random.default_rng(20261118)
base=out[(out.Mw_quality=='A')&out.primary_geography].copy()
base.to_csv(HERE/'tables/Mw_ML_primary.csv',index=False)
def regression(g,label,column='Mw',balanced=True,depth=False,bootstrap=1500):
    g=g[np.isfinite(g[column])].copy()
    if len(g)<10:return
    X=np.column_stack([np.ones(len(g)),g.ML.to_numpy()-3.]+([(g.z_central.to_numpy()-10)/10.] if depth else []))
    y=g[column].to_numpy();blocks=g.family_block.to_numpy();unique=np.unique(blocks)
    xx=[];xy=[]
    if balanced:
        for key in unique:
            a=X[blocks==key];b=y[blocks==key];xx.append(a.T@a/len(a));xy.append(a.T@b/len(a))
        weights=np.array([1/np.sum(blocks==key) for key in blocks])
    else:
        # Still resample whole families for uncertainty, while retaining all-event weights.
        for key in unique:
            a=X[blocks==key];b=y[blocks==key];xx.append(a.T@a);xy.append(a.T@b)
        weights=np.ones(len(g))
    xx=np.array(xx);xy=np.array(xy);coef=np.linalg.lstsq(xx.sum(axis=0),xy.sum(axis=0),rcond=None)[0]
    pred=X@coef;res=y-pred
    count=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=bootstrap)
    mats=np.einsum('bi,ijk->bjk',count,xx);rhs=count@xy
    try:boot=np.linalg.solve(mats,rhs[...,None])[...,0]
    except np.linalg.LinAlgError:boot=np.array([np.linalg.lstsq(a,b,rcond=None)[0] for a,b in zip(mats,rhs)])
    row=dict(regression_method='OLS of response on ML',selection=label,balanced_by_family=balanced,depth_term=depth,response=column,n_events=len(g),n_blocks=len(unique),
      intercept_at_ML3=float(coef[0]),slope_ML=float(coef[1]),intercept_standard=float(coef[0]-3*coef[1]),
      RMSE=float(np.sqrt(np.average(res**2,weights=weights))),mean_Mw_minus_ML=float(np.average(y-g.ML,weights=weights)),
      depth_coefficient_per10km=float(coef[2]) if depth else np.nan)
    for j,name in enumerate(['intercept_at_ML3','slope_ML']+(['depth_coefficient_per10km'] if depth else [])):
        row[name+'_lo']=float(np.quantile(boot[:,j],.025));row[name+'_hi']=float(np.quantile(boot[:,j],.975))
    reg.append(row)
    if label=='Primary' and not depth:
        for a,b in boot:draws.append(dict(intercept_at_ML3=float(a),slope_ML=float(b)))
    return coef
regression(base,'Primary')
regression(base,'All-event OLS',balanced=False)
regression(base[base.grade=='G3'],'G3 depth only')
regression(out[out.Mw_quality.isin(['A','B'])&out.primary_geography],'A or B')
regression(base[base.region=='Land'],'Land only')
regression(base[base.all_master_epochs_confirmed_by_issue.fillna(False)],'Response history strict')
regression(base[base.family!=0],'Without Gyeongju family')
regression(base[base.family!=1],'Without Pohang family')
regression(base[~base.family.isin([0,1])],'Without Gyeongju/Pohang')
regression(base,'HH-only sensitivity',column='Mw_HH')
regression(base,'Velocity-only sensitivity',column='Mw_velocity')
regression(base,'HG-only sensitivity',column='Mw_HG')
regression(base,'Primary + depth',depth=True)
for model in ['Kim1983','Kim2011']:regression(base,'Source Vs '+model,column='Mw_source_Vs_'+model,depth=True)
regression(base,'Fixed Vs3500 control',column='Mw_reference_Vs3500',depth=True)
pd.DataFrame(reg).to_csv(HERE/'tables/regression_comparison.csv',index=False)
pd.DataFrame(draws).to_csv(HERE/'tables/regression_family_bootstrap.csv',index=False)
review_path=HERE/'tables/events_for_waveform_review.csv'
if review_path.exists():
    ids=pd.read_csv(review_path,dtype={'event_id':str}).event_id
    reviewed=out[out.event_id.isin(ids)][['event_id','origin_utc','ml','Mw','z_central']].copy()
    primary=next(x for x in reg if x['selection']=='Primary')
    reviewed['regression_residual']=reviewed.Mw-(primary['intercept_standard']+primary['slope_ML']*reviewed.ml)
    reviewed['selection_reason']='Retained waveform-review set; values refreshed for current source Vs(z)'
    reviewed.to_csv(review_path,index=False)
summary=dict(candidate_events=len(cohort),processed_events=len(cohort),primary_geography_events=int(cohort.primary_geography.sum()),
 Mw_quality_counts=out.Mw_quality.value_counts().to_dict(),primary_regression_events=len(base),
 primary_sequence_blocks=int(base.family_block.nunique()) if len(base) else 0,
 Mw_constant=9.05,depth_reference_D90_km=16.89,raw_processing_signature=list(signatures)[0],
 source_velocity_model=PRIMARY_MODEL,source_velocity_config_sha256=CONFIG_SHA256,
 source_velocity_comparison_model='Kim2011',source_velocity_scope=CONFIG['scope'],
 quality_rules_sha256=hashlib.sha256((HERE/'inputs/analysis_quality_rules.json').read_bytes()).hexdigest(),
 adopted_catalogs_modified=False,regressions=reg,
 limitations=['Absolute response/path/radiation/site calibration remains conditional.',
  'Station bootstrap intervals hold source shape and Earth assumptions fixed.',
  'Family-block regression intervals measure sample variation, not complete magnitude uncertainty.',
  'Reliable-depth selection and spectral SNR affect representativeness; no extrapolation below ML2.'])
(HERE/'tables/study_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['regressions']},indent=2))
print(pd.DataFrame(reg).round(4).to_string(index=False))
