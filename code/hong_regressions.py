"""Linear ODR/Deming and OLS on explicit, identical magnitude samples.

Equal-axis ODR is evaluated in closed form (weighted total least squares),
then checked independently against numerical ODRPACK. No published MAD is
silently interpreted as a measurement standard deviation.
"""
from pathlib import Path
import hashlib,json,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]

def coefficients(moment,ratio=None):
    moment=np.asarray(moment);n,sx,sy,sxx,sxy,syy=np.moveaxis(moment,-1,0)
    mx,my=sx/n,sy/n;vx=sxx/n-mx*mx;vy=syy/n-my*my;cov=sxy/n-mx*my
    if ratio is None:b=cov/vx
    else:b=(vy-ratio*vx+np.sqrt((vy-ratio*vx)**2+4*ratio*cov*cov))/(2*cov)
    a=my-b*mx
    return np.stack([a,b],axis=-1)

def regress(q,xkey,ykey,selection,weighting='event',family_bootstrap=True,reps=2500):
    q=q.dropna(subset=[xkey,ykey]).copy();x=q[xkey].to_numpy();y=q[ykey].to_numpy()
    blocks=q.family_block.to_numpy() if family_bootstrap else q.event_id.to_numpy()
    ids,inv=np.unique(blocks,return_inverse=True);n=np.bincount(inv)
    raw=np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y]);mom=np.zeros((len(ids),6))
    np.add.at(mom,inv,raw)
    if weighting=='family':mom=mom/n[:,None]
    weights=(1/n[inv]) if weighting=='family' else np.ones(len(q))
    weights=weights/weights.sum()
    rng=np.random.default_rng(20261120)
    draws=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=reps)@mom
    output=[];boots=[]
    for method,ratio in [('OLS',None),('Deming',.25),('ODR',1.),('Deming',4.)]:
        a,b=coefficients(mom.sum(axis=0),ratio);boot=coefficients(draws,ratio)
        residual=y-a-b*x;mx=np.sum(weights*x);my=np.sum(weights*y)
        vx=np.sum(weights*(x-mx)**2);vy=np.sum(weights*(y-my)**2);cov=np.sum(weights*(x-mx)*(y-my))
        ci=np.quantile(boot,[.025,.975],axis=0);at3=boot[:,0]+3*boot[:,1]
        row=dict(selection=selection,x=xkey,y=ykey,method=method,variance_ratio=ratio,
          weighting=weighting,bootstrap='family blocks' if family_bootstrap else 'events; sequence dependence unmodeled',
          n_events=len(q),n_blocks=len(ids),intercept=float(a),slope=float(b),intercept_at_x3=float(a+3*b),
          intercept_lo=float(ci[0,0]),intercept_hi=float(ci[1,0]),slope_lo=float(ci[0,1]),slope_hi=float(ci[1,1]),
          at_x3_lo=float(np.quantile(at3,.025)),at_x3_hi=float(np.quantile(at3,.975)),
          RMS_vertical=float(np.sqrt(np.sum(weights*residual**2))),
          RMS_orthogonal=float(np.sqrt(np.sum(weights*residual**2)/(1+b*b))),
          correlation_squared=float(cov*cov/(vx*vy)),
          R2_vertical=float(1-np.sum(weights*residual**2)/vy),
          mean_y_minus_x=float(np.sum(weights*(y-x))),x_min=float(x.min()),x_max=float(x.max()))
        output.append(row)
        if method=='ODR':
            boots.append(dict(selection=selection,x=xkey,y=ykey,weighting=weighting,
              intercept=boot[:,0].tolist(),slope=boot[:,1].tolist()))
    return output,boots

def main():
    d=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str})
    k=pd.read_csv(R/'inputs/KR_Mw/Events_KP_2017_2022_Mw.csv',dtype={'evid':str})
    k['event_id']=k.evid.str.strip();k['ML_KMA']=k.ml_KMA
    m=d.merge(k.drop(columns=['ML_KMA']),on='event_id',validate='one_to_one',suffixes=('','_KR'))
    missing=m.family_block.isna()
    m.loc[missing,'family_block']=[('family_'+str(int(a.family))) if a.family>=0 else 'event_'+a.event_id for a in m[missing].itertuples()]
    g=m[(m.Mw_quality=='A')&m.primary_geography].copy()
    h=g[g.ML_comparison_primary].copy()
    samples=[]
    # Reference reproduction of paper Figure 6 and full-catalog ML relations.
    samples.append(('KR full 490',k,'mw_spectra','mw_TD',False))
    for target in ['mw_spectra','mw_TD']:
        samples.append(('KR full 490',k,'ML_KMA',target,False))
        limited=k[k.ML_KMA.between(h.ML_KMA.min(),h.ML_KMA.max())]
        samples.append(('KR same ML range',limited,'ML_KMA',target,False))
    for target in ['mw_spectra','mw_TD']:
        samples.append(('All exact matches',m,'ML_KMA',target,True))
    for ref in ['mw_spectra','mw_TD']:
        for target in ['Mw','Mw_Fourier','Mw_source_Vs_Kim2011']:
            samples.append(('Matched Mw A',g,ref,target,True))
    for target in ['Mw','Mw_source_Vs_Kim2011','mw_spectra','mw_TD']:
        samples.append(('Matched Mw A',g,'ML_KMA',target,True))
        for x in ['ML_KMA','ML_Sheen']:samples.append(('Matched Mw A / ML A',h,x,target,True))
    samples.append(('Our full paired A',d[d.ML_comparison_primary],'ML_Sheen','Mw',True))
    samples.append(('Our full paired A',d[d.ML_comparison_primary],'ML_KMA','Mw',True))
    for x in ['ML_Sheen','ML_KMA']:
        samples.append(('Our full paired A',d[d.ML_comparison_primary],x,'Mw_source_Vs_Kim2011',True))
    samples.append(('Our full Mw A',d[(d.Mw_quality=='A')&d.primary_geography],'ML_KMA','Mw',True))
    rows=[];boot=[];members=[]
    for label,q,x,y,blocks in samples:
        for weighting in (['event','family'] if blocks else ['event']):
            rr,bb=regress(q,x,y,label,weighting,blocks)
            rows+=rr;boot+=bb
        members.extend(dict(selection=label,x=x,y=y,event_id=e) for e in q.event_id)
    pd.DataFrame(rows).to_csv(R/'tables/Hong_regression_comparison.csv',index=False)
    pd.DataFrame(members).to_csv(R/'tables/Hong_regression_members.csv',index=False)
    (R/'tables/Hong_ODR_bootstrap.json').write_text(json.dumps(boot)+'\n')
    # Independent numerical ODR check for every equal-axis fit, both weightings.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        from scipy import odr
    errors=[]
    for label,q,x,y,blocks in samples:
        for weighting in (['event','family'] if blocks else ['event']):
            expected=next(r for r in rows if r['selection']==label and r['x']==x and r['y']==y and
              r['method']=='ODR' and r['weighting']==weighting)
            q=q.dropna(subset=[x,y]);w=np.ones(len(q))
            if weighting=='family':w=1/q.groupby('family_block').family_block.transform('size').to_numpy()
            # Equal x/y uncertainties express the metric only, not measured sigma.
            data=odr.RealData(q[x].to_numpy(),q[y].to_numpy(),sx=1/np.sqrt(w),sy=1/np.sqrt(w))
            model=odr.Model(lambda beta,xx:beta[0]+beta[1]*xx,
              fjacb=lambda beta,xx:np.vstack([np.ones_like(xx),xx]),
              fjacd=lambda beta,xx:np.full_like(xx,beta[1]))
            engine=odr.ODR(data,model,beta0=[expected['intercept'],expected['slope']],maxit=500,sstol=1e-12,partol=1e-12)
            engine.set_job(deriv=3)
            fit=engine.run()
            err=float(np.max(abs(fit.beta-np.array([expected['intercept'],expected['slope']]))))
            assert err<2e-6,(label,x,y,weighting,err)
            errors.append(err)
    # Reproduce the previous headline OLS without changing its adopted table.
    old=pd.read_csv(R/'tables/independent_ML_regressions.csv').query("selection=='Independent ML'").iloc[0]
    new=next(r for r in rows if r['selection']=='Our full paired A' and r['x']=='ML_Sheen' and
      r['method']=='OLS' and r['weighting']=='family')
    assert abs(new['slope']-old.slope)<1e-10
    assert abs(new['intercept']-old.intercept_standard)<1e-10
    result=dict(passed=True,closed_form='Weighted total least squares; Deming lambda=1 is Euclidean ODR',
      scipy_ODRPACK_checks=len(errors),max_parameter_difference=max(errors),
      previous_headline_method='Family-balanced OLS of Mw on ML, with Deming sensitivities',
      previous_KR_comparison_slopes='Unweighted vertical OLS',
      equal_axis_assumption='lambda=sigma_y^2/sigma_x^2=1; sensitivities .25 and 4',
      uncertainty='2500 family-block resamples for matched/current samples; event resamples for full KR',
      caution='Shared waveform/response errors are correlated; standard ODR does not model their cross-axis covariance or selection bias.',
      primary_magnitudes_modified_by_regression=False,current_OLS_reproduced=True,
      source_velocity_model='Kim1983, Vp/Vs=1.73',comparison_model='Kim2011, published Table 2',
      n_common_Mw_A=len(g),n_common_paired_A=len(h),n_published=len(k))
    (R/'tables/Hong_regression_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    out=pd.DataFrame(rows);print(out[(out.method=='ODR')&(out.weighting=='event')][[
      'selection','x','y','n_events','intercept','slope','slope_lo','slope_hi','RMS_orthogonal']].round(5).to_string(index=False))

if __name__=='__main__':main()
