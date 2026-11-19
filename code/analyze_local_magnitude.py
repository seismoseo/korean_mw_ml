"""Compare independently measured ML, KMA ML and frozen spectral Mw."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
mw=pd.read_csv(ROOT/'tables/Mw_ML_catalog.csv',dtype={'event_id':str})
ml=pd.read_csv(ROOT/'tables/independent_ML_catalog.csv',dtype={'event_id':str})
assert len(mw)==642 and len(ml)==643
assert mw.event_id.is_unique and ml.event_id.is_unique
d=mw.merge(ml,on='event_id',how='left',validate='one_to_one')
# P timing and exact dated response checks are applied before both measurements.
assert d.ML_quality.notna().all()
assert np.allclose(d.ml,d.ML_KMA)
d['ML_minus_KMA']=d.ML_Sheen-d.ML_KMA
d['Mw_minus_independent_ML']=d.Mw-d.ML_Sheen
d['ML_comparison_primary']=(d.ML_quality=='A')&(d.Mw_quality=='A')&d.primary_geography
d.to_csv(ROOT/'tables/Mw_independent_ML_catalog.csv',index=False)
g=d[d.ML_comparison_primary].copy()
g.to_csv(ROOT/'tables/Mw_independent_ML_primary.csv',index=False)


def regression(frame,xkey,ykey='Mw',depth=False,reps=1500,weighting='family'):
    fields=[xkey,ykey,'family_block']+(['z_central'] if depth else [])
    q=frame.dropna(subset=fields).copy()
    if len(q)<10:return None,None
    x=q[xkey].to_numpy();y=q[ykey].to_numpy()
    X=np.column_stack([np.ones(len(q)),x-3])
    if depth:X=np.column_stack([X,(q.z_central.to_numpy()-10)/10])
    blocks=q.family_block.to_numpy() if weighting=='family' else q.event_id.to_numpy()
    keys=np.unique(blocks);p=X.shape[1]
    moments=np.array([np.r_[np.mean(X[blocks==key,:,None]*X[blocks==key,None,:],axis=0).ravel(),
                      np.mean(X[blocks==key]*y[blocks==key,None],axis=0)] for key in keys])
    moment=moments.mean(axis=0)
    coef=np.linalg.solve(moment[:p*p].reshape(p,p),moment[p*p:])
    rng=np.random.default_rng(20261118)
    counts=rng.multinomial(len(keys),np.ones(len(keys))/len(keys),size=reps)
    samples=counts@moments/len(keys)
    coefboot=np.linalg.solve(samples[:,:p*p].reshape(-1,p,p),samples[:,p*p:,None])[...,0]
    residual=y-X@coef
    rmse=float(np.sqrt(np.mean([np.mean(residual[blocks==key]**2) for key in keys])))
    bounds=np.quantile(coefboot,[.025,.975],axis=0)
    result=dict(regression_method='OLS of y on x',x=xkey,y=ykey,n_events=len(q),n_blocks=len(keys),weighting=weighting,
        intercept_at_x3=float(coef[0]),slope=float(coef[1]),intercept_standard=float(coef[0]-3*coef[1]),
        slope_lo=float(bounds[0,1]),slope_hi=float(bounds[1,1]),RMSE=rmse,
        depth_coefficient=float(coef[2]) if depth else np.nan,
        depth_lo=float(bounds[0,2]) if depth else np.nan,depth_hi=float(bounds[1,2]) if depth else np.nan,
        x_min=float(x.min()),x_max=float(x.max()))
    boot=pd.DataFrame({'intercept_at_x3':coefboot[:,0],'slope':coefboot[:,1]})
    return result,boot


specs=[('Independent ML',g,'ML_Sheen','Mw',False),('KMA ML, same events',g,'ML_KMA','Mw',False),
       ('Independent ML + depth',g,'ML_Sheen','Mw',True),('KMA ML + depth, same events',g,'ML_KMA','Mw',True),
       ('BB-only ML',g,'ML_BB','Mw',False),('SNR2 ML',g,'ML_SNR2','Mw',False),
       ('Velocity-only ML',g,'ML_velocity','Mw',False),('HG-only ML',g,'ML_HG','Mw',False),('Velocity-only Mw',g,'ML_Sheen','Mw_velocity',False),('40--100 km ML',g,'ML_far','Mw',False),
       ('Strict response history',g,'ML_strict_metadata','Mw',False),('G3 depths',g[g.grade=='G3'],'ML_Sheen','Mw',False),
       ('All measured ML, Mw A',d[(d.Mw_quality=='A')&d.primary_geography],'ML_Sheen','Mw',False),
       ('Independent ML vs KMA',d[(d.ML_quality=='A')&d.primary_geography],'ML_KMA','ML_Sheen',False)]
for model in ['Kim1983','Kim2011']:
    specs.append(('Measured ML + depth, Vs '+model,g,'ML_Sheen','Mw_source_Vs_'+model,True))
    specs.append(('Measured ML, Vs '+model,g,'ML_Sheen','Mw_source_Vs_'+model,False))
    specs.append(('KMA ML, Vs '+model,g,'ML_KMA','Mw_source_Vs_'+model,False))
specs.append(('Measured ML + depth, fixed Vs3500',g,'ML_Sheen','Mw_reference_Vs3500',True))
rows=[]
for label,frame,x,y,depth in specs:
    r,boot=regression(frame,x,y,depth)
    if r is None:continue
    r['selection']=label;rows.append(r)
    if label in ['Independent ML','KMA ML, same events']:
        boot.to_csv(ROOT/'tables'/('ML_regression_bootstrap_'+('independent' if label=='Independent ML' else 'KMA')+'.csv'),index=False)
reg=pd.DataFrame(rows);reg.to_csv(ROOT/'tables/independent_ML_regressions.csv',index=False)

# Both axes contain measurement errors: report sensitivities, not an inferred
# true error ratio. Family statistics preserve equal weight for each sequence.
dem=[]
for xkey in ['ML_Sheen','ML_KMA']:
    stats=[]
    for _,q in g.groupby('family_block'):
        x=q[xkey].to_numpy();y=q.Mw.to_numpy()
        stats.append([x.mean(),y.mean(),np.mean(x*x),np.mean(y*y),np.mean(x*y)])
    stats=np.array(stats);mx,my,xx,yy,xy=stats.mean(axis=0)
    vx,vy,cov=xx-mx*mx,yy-my*my,xy-mx*my
    for ratio in [.25,1.,4.]:
        b=(vy-ratio*vx+np.sqrt((vy-ratio*vx)**2+4*ratio*cov**2))/(2*cov)
        dem.append(dict(x=xkey,variance_ratio_Mw_to_ML=ratio,slope=float(b),intercept=float(my-b*mx)))
pd.DataFrame(dem).to_csv(ROOT/'tables/independent_ML_Deming.csv',index=False)

# Held-out prediction excludes the complete sequence block.
held=[]
for key in ['ML_Sheen','ML_KMA']:
    blocks=g.family_block.to_numpy();x=g[key].to_numpy();y=g.Mw.to_numpy();keys=np.unique(blocks)
    stats=np.array([[x[blocks==b].mean(),y[blocks==b].mean(),np.mean(x[blocks==b]**2),np.mean(x[blocks==b]*y[blocks==b])] for b in keys])
    total=stats.sum(axis=0)
    for i,b in enumerate(keys):
        mx,my,xx,xy=(total-stats[i])/(len(keys)-1);slope=(xy-mx*my)/(xx-mx*mx);a=my-slope*mx
        for eid,xx0,yy0 in zip(g.loc[blocks==b,'event_id'],x[blocks==b],y[blocks==b]):
            held.append(dict(event_id=eid,family_block=b,scale=key,predicted_Mw=float(a+slope*xx0),residual=float(yy0-a-slope*xx0)))
held=pd.DataFrame(held);held.to_csv(ROOT/'tables/independent_ML_family_heldout.csv',index=False)
cv={key:float(np.sqrt(q.groupby('family_block').residual.apply(lambda s:np.mean(s**2)).mean())) for key,q in held.groupby('scale')}

summaries=[]
for label,q in [('All measured',d[d.ML_Sheen.notna()]),('ML A, primary domain',d[(d.ML_quality=='A')&d.primary_geography]),('Paired Mw A / ML A',g)]:
    difference=q.ML_minus_KMA.to_numpy();med=float(np.median(difference))
    summaries.append(dict(selection=label,n=len(q),median_ML_minus_KMA=med,robust_scatter=float(1.4826*np.median(abs(difference-med))),q16=float(np.quantile(difference,.16)),q84=float(np.quantile(difference,.84))))
pd.DataFrame(summaries).to_csv(ROOT/'tables/independent_ML_offsets.csv',index=False)

review=d[d.event_id.isin(pd.read_csv(ROOT/'tables/events_for_waveform_review.csv',dtype={'event_id':str}).event_id)].copy()
review[['event_id','ml','ML_Sheen','ML_Sheen_n','ML_quality','Mw','Mw_quality','z_central','ML_minus_KMA','Mw_minus_independent_ML']].to_csv(ROOT/'tables/independent_ML_review_events.csv',index=False)

target=ml[ml.event_id=='20261117141017'].iloc[0].to_dict()
result=dict(n_candidates=len(d),n_measured=int(d.ML_Sheen.notna().sum()),n_ML_A=int((d.ML_quality=='A').sum()),
 n_primary=len(g),n_blocks=int(g.family_block.nunique()),n_G3=int((g.grade=='G3').sum()),
 n_below_national_D90=int((g.z_central>=16.89).sum()),
 ML_quality_counts=d.ML_quality.value_counts().to_dict(),target=target,offsets=summaries,
 regressions=rows,Deming=dem,heldout_RMSE=cv,
 caveats=['KMA ML>=2 selects the parent sample; independent ML may fall below 2.',
 'Both magnitudes share count data and response uncertainties; this is an independent estimator, not independent observations.',
 'S=0 uses the published attenuation zero point; no local station terms are fitted.',
 '100-km archive aperture and reliable-depth/spectral selection limit generalization.',
 'No magnitude scale is adjusted to agree with KMA or with Mw.'])
(ROOT/'tables/independent_ML_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['n_candidates','n_measured','n_ML_A','n_primary','n_blocks','heldout_RMSE','offsets']},indent=2))
print(reg[['selection','n_events','slope','slope_lo','slope_hi','intercept_standard','RMSE']].to_string(index=False))
