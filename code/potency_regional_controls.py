"""Separate magnitude-range selection and regional ML-definition effects."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from potency_regression import run_case

R=Path(__file__).resolve().parents[1]
d=pd.read_csv(R/'tables/potency_event_catalog.csv',dtype={'event_id':str})
stations=pd.read_csv(R/'tables/independent_ML_stations.csv',dtype={'event_id':str})
depth=d.set_index('event_id').z_central
stations['source_depth_km']=stations.event_id.map(depth)
assert stations.source_depth_km.notna().all()
repi=stations.distance_km.to_numpy();rhyp=np.hypot(repi,stations.source_depth_km)
amp=stations.A_h_mm.to_numpy()
original=np.log10(amp)+.5869*np.log10(repi/100)+.001680*(repi-100)+3
assert np.allclose(original,stations.value,atol=1e-10)
stations['ML_HB87_same_WA2080']=np.log10(amp)+1.110*np.log10(rhyp/100)+.00189*(rhyp-100)+3
stations['ML_HB87_WA2800_gain_control']=stations.ML_HB87_same_WA2080+np.log10(2800/2080)
events=stations.groupby('event_id')[['ML_HB87_same_WA2080','ML_HB87_WA2800_gain_control']].median().reset_index()
events=d.merge(events,on='event_id',how='left',validate='one_to_one')
events.to_csv(R/'tables/potency_ML_definition_controls.csv',index=False)
g=events[events.ML_comparison_primary].copy()
tasks=[]
def add(q,selection,x,y,method,degree,bootstrap=0,cv=True):
    spec=dict(case_id=f'control_{len(tasks):03d}',selection=selection,x=x,y=y,method=method,weighting='family' if cv else 'event',
      degree=degree,n_bins=50,minimum=1,bootstrap_reps=bootstrap,do_cv=cv,
      block_policy='Whole national event-family blocks' if cv else 'Full family membership unavailable; no sequence CV claimed')
    tasks.append((spec,q[x].to_numpy(),q[y].to_numpy(),q.family_block.to_numpy(),q.event_id.to_numpy()))
for x in ['ML_HB87_same_WA2080','ML_HB87_WA2800_gain_control']:
    for degree in [1,2]:add(g,'ML attenuation control',x,'Mp_Liu_Kim1983','OLS',degree,1000)
kr=pd.read_csv(R/'inputs/KR_Mw/Events_KP_2017_2022_Mw.csv',dtype={'evid':str})
kr['event_id']=kr.evid.str.strip();kr['ML_KMA']=kr.ml_KMA;kr['family_block']='KR_event_'+kr.event_id
for field,name in [('mw_spectra','spectral'),('mw_TD','TD')]:
    kr['Mp_Hong_'+name]=(2/3)*(1.5*kr[field]+9.05-np.log10(2700*3610**2)+1.4563)
matched_paired=kr[kr.event_id.isin(g.event_id)]
limited=kr[kr.ML_KMA.between(matched_paired.ML_KMA.min(),matched_paired.ML_KMA.max())]
for y in ['Mp_Hong_spectral','Mp_Hong_TD']:
    for method in ['OLS','ODR','BinnedOLS']:
        for degree in [1,2]:add(limited,'Hong same ML range','ML_KMA',y,method,degree,cv=False)
rows=[];bins=[];predictions=[]
for task in tasks:
    row,boot,pred,fold,bb=run_case(task);rows.append(row);predictions+=pred
pd.DataFrame(rows).to_csv(R/'tables/potency_regional_control_regressions.csv',index=False)
pred=pd.DataFrame(predictions)
pred.to_csv(R/'tables/potency_regional_control_predictions.csv',index=False)
gains=[]
for x in ['ML_HB87_same_WA2080','ML_HB87_WA2800_gain_control']:
    specs=[row for row in rows if row['x']==x]
    pair=[]
    for degree in [1,2]:
        spec=next(row for row in specs if row['degree']==degree)
        q=pred[pred.case_id.eq(spec['case_id'])]
        pair.append(q.groupby('family_block',sort=True).residual.apply(lambda s:float(np.mean(s*s))).to_numpy())
    rng=np.random.default_rng(20261122);draw=rng.integers(len(pair[0]),size=(2500,len(pair[0])))
    gain=np.sqrt(pair[0][draw].mean(axis=1))-np.sqrt(pair[1][draw].mean(axis=1))
    gains.append(dict(x=x,linear_RMS_families=float(np.sqrt(pair[0].mean())),quadratic_RMS_families=float(np.sqrt(pair[1].mean())),
      gain=float(np.sqrt(pair[0].mean())-np.sqrt(pair[1].mean())),gain_lo=float(np.quantile(gain,.025)),gain_hi=float(np.quantile(gain,.975))))
stats=[]
for x in ['ML_Sheen','ML_KMA','ML_HB87_same_WA2080','ML_HB87_WA2800_gain_control']:
    diff=g.Mp_Liu_Kim1983-(.04*g[x]**2+.556*g[x]+1.133)
    stats.append(dict(ML_definition=x,n=len(g),median_ML_minus_ours=float((g[x]-g.ML_Sheen).median()),
      median_Mp_minus_Liu_curve=float(diff.median()),RMS_to_Liu_curve=float(np.sqrt(np.mean(diff**2)))))
summary=dict(scope='Change attenuation/distance formula on exactly the same accepted Korean horizontal WA amplitudes and station sets; not a full reproduction of Liu ML.',
  gain_scope='The 2800/2080 gain control is illustrative. Liu supplement S4 explicitly uses 2800 in the synthetic example; S1 does not fully specify the observational WA gain or component aggregation.',
  n_stations=len(stations),n_paired_A=len(g),Hong_full=len(kr),Hong_same_ML_range=len(limited),
  Hong_matched_paired_ML_range=[float(matched_paired.ML_KMA.min()),float(matched_paired.ML_KMA.max())],
  original_station_ML_reproduction_error=float(np.max(abs(original-stations.value))),
  gain_shift_ML=float(np.log10(2800/2080)),definitions=stats,heldout_gains=gains)
(R/'tables/potency_regional_controls_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));print(pd.DataFrame(rows)[['selection','x','y','method','degree','quadratic_c','c_lo','c_hi']].round(5).to_string(index=False))
