"""Independent checks for the four magnitude follow-up questions."""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import fitz

R=Path(__file__).resolve().parents[1];T=R/'tables'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((T/'magnitude_followup_summary.json').read_text())
for name,digest in summary['protected_outputs_sha256'].items():assert sha(R/name)==digest,name
for name,digest in summary['inputs_sha256'].items():assert sha(T/name)==digest,name
g=pd.read_csv(T/'magnitude_followup_events.csv',dtype={'event_id':str})
fits=pd.read_csv(T/'magnitude_followup_regressions.csv')
pred=pd.read_csv(T/'magnitude_followup_predictions.csv',dtype={'event_id':str})
metrics=pd.read_csv(T/'magnitude_followup_prediction_metrics.csv')
cv=pd.read_csv(T/'magnitude_followup_CV_comparison.csv')
depth=pd.read_csv(T/'magnitude_followup_depth_regressions.csv')
db=np.load(T/'magnitude_followup_depth_bootstrap.npz')
assert len(g)==357 and g.event_id.is_unique and g.family_block.nunique()==194
assert len(fits)==128 and len(depth)==summary['n_depth_fits']==84
assert len(g[g.ML_Sheen>4])==6 and g[g.ML_Sheen>4].family_block.nunique()==5
assert len(g[g.ML_KMA>4])==3 and g[g.ML_KMA>4].family_block.nunique()==2
requested=fits[fits.bootstrap_reps>0]
assert (requested.n_successful_bootstraps==requested.bootstrap_reps).all()
assert fits.monotone_over_observed_range.all()
assert fits.ODR_linear_closed_form_error.max()<2e-6
assert fits.ODR_projection_stationarity.max()<2e-5
def row(x='ML_Sheen',y='Mw',degree=2):
    q=fits[fits.x.eq(x)&fits.y.eq(y)&fits.degree.eq(degree)&fits.method.eq('OLS')&fits.weighting.eq('family')]
    assert len(q)==1
    return q.iloc[0]
def drow(x='ML_Sheen',y='Mp_Liu_Kim1983',control='ML quadratic'):
    q=depth[depth.x.eq(x)&depth.y.eq(y)&depth.control.eq(control)]
    assert len(q)==1
    return q.iloc[0]
tail_checks=[]
for y in ['Mp_Liu_Kim1983','Mw']:
    high=g[g.ML_Sheen>4].copy()
    for degree in [1,2]:
        f=row(y=y,degree=degree)
        q=pred[pred.case_id.eq(f.case_id)].set_index('event_id')
        high['r'+str(degree)]=high.event_id.map(q.residual)
        for family,part in high.groupby('family_block'):
            train=g[g.family_block!=family]
            counts=train.groupby('family_block').family_block.transform('size').to_numpy()
            beta=np.polyfit(train.ML_Sheen.to_numpy()-3,train[y].to_numpy(),degree,w=1/np.sqrt(counts))
            expected=part[y].to_numpy()-np.polyval(beta,part.ML_Sheen.to_numpy()-3)
            assert np.allclose(expected,part.event_id.map(q.residual),atol=2e-10)
        value=np.sqrt(np.mean(high['r'+str(degree)]**2))
        metric=metrics[metrics.case_id.eq(f.case_id)&metrics.subset.eq('ML > 4')].iloc[0]
        assert abs(value-metric.CV_RMSE_events)<1e-12
        assert metric.n_extrapolation==2
    matrix=high.groupby('family_block')[['r1','r2']].agg(lambda s:np.mean(s*s)).to_numpy()
    draws=np.array(list(itertools.product(range(5),repeat=5)))
    gain=np.sqrt(matrix[draws,0].mean(1))-np.sqrt(matrix[draws,1].mean(1))
    result=cv[cv.x.eq('ML_Sheen')&cv.y.eq(y)&cv.method.eq('OLS')&cv.weighting.eq('family')&cv.subset.eq('ML > 4')].iloc[0]
    assert np.allclose(np.quantile(gain,[.025,.975]),[result.gain_lo,result.gain_hi],atol=1e-12)
    assert result.gain_lo<0<result.gain_hi
    tail_checks.append(dict(y=y,linear_RMSE=float(np.sqrt(np.mean(high.r1**2))),
        quadratic_RMSE=float(np.sqrt(np.mean(high.r2**2))),gain_lo=result.gain_lo,gain_hi=result.gain_hi))

# Check the conditional depth coefficient using Frisch-Waugh-Lovell, rather
# than reusing the fitted normal equations or the production regression code.
blocks=g.family_block.to_numpy();keys,inverse,n=np.unique(blocks,return_inverse=True,return_counts=True)
w=1/n[inverse];sqrtw=np.sqrt(w)
z=(g.z_central.to_numpy()-15)/10
max_fwl=0.;max_bootstrap=0.
for x in ['ML_Sheen','ML_KMA']:
    m=g[x].to_numpy()-3;C=np.column_stack([np.ones(len(g)),m,m*m])
    zres=z-C@np.linalg.lstsq(C*sqrtw[:,None],z*sqrtw,rcond=None)[0]
    X=np.column_stack([C,z])
    for y in ['Mp_Liu_Kim1983','Mp_Liu_Kim2011','Mw','Mw_source_Vs_Kim2011']:
        yy=g[y].to_numpy()
        yres=yy-C@np.linalg.lstsq(C*sqrtw[:,None],yy*sqrtw,rcond=None)[0]
        slope=np.sum(w*zres*yres)/np.sum(w*zres*zres)
        rr=drow(x=x,y=y)
        max_fwl=max(max_fwl,abs(slope-rr.depth_slope_per_10km))
        assert abs(slope-rr.depth_slope_per_10km)<1e-10
        # Independently duplicate the sampled observations for the first twelve
        # complete-family bootstrap draws, retaining each family's total weight.
        counts=np.random.default_rng(20261124).multinomial(len(n),np.ones(len(n))/len(n),size=12)
        for k,count in enumerate(counts):
            ix=np.repeat(np.arange(len(g)),count[inverse])
            b=np.linalg.lstsq(X[ix]*sqrtw[ix,None],yy[ix]*sqrtw[ix],rcond=None)[0]
            err=np.max(abs(b-db[rr.case_id][k]));max_bootstrap=max(max_bootstrap,err)
            assert err<2e-8
for x in ['ML_Sheen','ML_KMA']:
    a=drow(x=x,y='Mp_Liu_fixed_Vs3500');b=drow(x=x,y='Mw_reference_Vs3500')
    assert abs(a.depth_slope_per_10km-b.depth_slope_per_10km)<1e-10
    assert np.allclose(db[a.case_id][:,-1],db[b.case_id][:,-1],atol=1e-9)
for y in ['Mp_Liu_Kim1983','Mp_Liu_Kim2011','Mw','Mw_source_Vs_Kim2011']:
    assert np.isfinite(db[drow(y=y).case_id]).all()
assert drow().depth_hi<0
assert drow(control='Depth at least 15 km').depth_lo<0<drow(control='Depth at least 15 km').depth_hi

# Verify an adding-depth CV fold independently and confirm no withheld members
# of the Gyeongju family appear in either training regression.
dp=pd.read_csv(T/'magnitude_followup_depth_predictions.csv',dtype={'event_id':str})
rr=drow();q=dp[dp.case_id.eq(rr.case_id)].set_index('event_id')
test=g.family_block.eq('family_0').to_numpy();train=~test
m=g.ML_Sheen.to_numpy()-3
X=np.column_stack([np.ones(len(g)),m,m*m,z]);yy=g.Mp_Liu_Kim1983.to_numpy()
wt=1/g[train].groupby('family_block').family_block.transform('size').to_numpy()
for use_z,col in [(False,'predicted_without_depth'),(True,'predicted_with_depth')]:
    C=X if use_z else X[:,:-1]
    b=np.linalg.lstsq(C[train]*np.sqrt(wt[:,None]),yy[train]*np.sqrt(wt),rcond=None)[0]
    assert np.allclose(C[test]@b,g.loc[test,'event_id'].map(q[col]),atol=1e-10)

# Check magnitude-scale identities with the user's C=9.05 convention retained.
mu=2800*g.source_Vs_Kim1983**2
ref=10**(9.1+1.4563)
assert np.allclose(g.Mp_Liu_Kim1983,g.Mw+(2/3)*np.log10(ref/mu)-.05/1.5,atol=1e-11)
metadata=json.loads((T/'potency_report_metadata.json').read_text())
pdf=R/'reports/Potency_magnitude_and_quadratic_scaling.pdf'
spans=0
with fitz.open(pdf) as doc:
    assert len(doc)==metadata['report_pages']==11
    text='\n'.join(p.get_text() for p in doc[6:])
    for required in ['0.1609','0.1114','0.1689','0.1164','36 GPa','0.061','0.0301','0.07523','0.07527']:
        assert required in text,required
    for page in doc:
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    if not span['text'].strip():continue
                    b=fitz.Rect(span['bbox']);spans+=1
                    assert b.x0>=-1 and b.y0>=-1 and b.x1<=page.rect.width+1 and b.y1<=page.rect.height+1,(page.number,span['text'],span['bbox'])
assert min(metadata['narrative_scales'])>=.95
scripts=['analyze_magnitude_followup.py','report_magnitude_followup.py','validate_magnitude_followup.py',
         'report_potency.py','validate_potency.py','potency_regression.py']
for name in scripts:ast.parse((R/'code'/name).read_text(),filename=name)
result=dict(passed=True,n_events=len(g),n_families=len(keys),n_magnitude_fits=len(fits),n_depth_fits=len(depth),
    identical_Mp_Mw_sample=True,source_catalogs_and_six_previous_reports_unchanged=True,
    bootstrap_success_fraction=1.,large_event_predictions_independently_refit=True,large_event_results=tail_checks,
    exact_five_family_gain_interval_verified=True,depth_FWL_max_error=max_fwl,
    independently_duplicated_cluster_bootstrap_max_error=max_bootstrap,
    depth_Gyeongju_holdout_verified=True,normalization_C9p05_identity_verified=True,
    ODR_linear_closed_form_max_error=float(fits.ODR_linear_closed_form_error.max()),
    ODR_quadratic_stationarity_max_error=float(fits.ODR_projection_stationarity.max()),
    report_pages=11,pdf_text_spans_checked=spans,report_sha256=sha(pdf),
    code_sha256={name:sha(R/'code'/name) for name in scripts})
(T/'magnitude_followup_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
