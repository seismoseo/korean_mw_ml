"""Error-in-both-axes sensitivity and leave-one-family-out prediction."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1]
d=pd.read_csv(HERE/'tables/Mw_ML_catalog.csv',dtype={'event_id':str})
g=d[(d.Mw_quality=='A')&d.primary_geography].copy()
blocks=g.family_block.to_numpy();keys=np.unique(blocks);x=g.ML.to_numpy();y=g.Mw.to_numpy()
statistics=[]
for key in keys:
    xx=x[blocks==key];yy=y[blocks==key]
    statistics.append([xx.mean(),yy.mean(),np.mean(xx**2),np.mean(yy**2),np.mean(xx*yy)])
statistics=np.array(statistics)
def deming(s,ratio):
    mx,my,xx,yy,xy=np.moveaxis(s,-1,0);vx=xx-mx**2;vy=yy-my**2;cov=xy-mx*my
    b=(vy-ratio*vx+np.sqrt((vy-ratio*vx)**2+4*ratio*cov**2))/(2*cov)
    return my-b*mx,b
rng=np.random.default_rng(20261118)
counts=rng.multinomial(len(keys),np.ones(len(keys))/len(keys),size=1500)
sample=(counts@statistics)/len(keys);rows=[]
for ratio in [.25,1.,4.]:
    a,b=deming(statistics.mean(axis=0),ratio);aa,bb=deming(sample,ratio)
    rows.append(dict(error_variance_ratio_Mw_to_ML=ratio,intercept=float(a),slope=float(b),
       slope_lo=float(np.quantile(bb,.025)),slope_hi=float(np.quantile(bb,.975)),
       note='Assumed error ratio, family-balanced Deming sensitivity; not an estimated true measurement-error ratio'))
pd.DataFrame(rows).to_csv(HERE/'tables/Deming_regression_sensitivity.csv',index=False)
held=[]
for i,key in enumerate(keys):
    s=np.delete(statistics,i,axis=0).mean(axis=0);mx,my,xx,yy,xy=s
    b=(xy-mx*my)/(xx-mx*mx);a=my-b*mx
    for eid,xx0,yy0 in zip(g.loc[blocks==key,'event_id'],x[blocks==key],y[blocks==key]):
        held.append(dict(event_id=eid,family_block=key,predicted_Mw=float(a+b*xx0),observed_Mw=float(yy0),
                         residual=float(yy0-a-b*xx0)))
held=pd.DataFrame(held);held.to_csv(HERE/'tables/leave_one_family_out.csv',index=False)
errors=held.groupby('family_block').residual.apply(lambda s:float(np.mean(s**2)))
result=dict(n_events=len(g),n_blocks=len(keys),family_balanced_prediction_RMSE=float(np.sqrt(errors.mean())),
 Deming_sensitivity=rows,interpretation='The relation is conditional on depth/magnitude quality selection and physical amplitude assumptions; no magnitude offset is calibrated to ML.')
(HERE/'tables/regression_controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
