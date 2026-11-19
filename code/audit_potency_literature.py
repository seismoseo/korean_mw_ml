"""Resolve source/reference rigidity and compare Korean residuals with Liu S7."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from analyze_magnitude_followup import blocks_bootstrap
from potency_model import CONFIG,CONFIG_SHA256

R=Path(__file__).resolve().parents[1];T=R/'tables'
g=pd.read_csv(T/'magnitude_followup_events.csv',dtype={'event_id':str})
assert len(g)==357 and g.family_block.nunique()==194
rows=[];boots={};offsets=[]
for x in ['ML_Sheen','ML_KMA']:
    full_residual=g.Mp_Liu_Kim1983-(.04*g[x]**2+.556*g[x]+1.133)
    offsets.append(dict(ML=x,n=len(g),median=float(full_residual.median()),
        RMS=float(np.sqrt(np.mean(full_residual**2))),
        equivalent_potency_factor=float(10**(-1.5*full_residual.median()))))
    for label,q in [('Paired A',g),('ML > 3',g[g[x]>3])]:
        residual=q.Mp_Liu_Kim1983.to_numpy()-(.04*q[x].to_numpy()**2+.556*q[x].to_numpy()+1.133)
        X=np.column_stack([np.ones(len(q)),(q.z_central.to_numpy()-15)/10])
        w=1/q.groupby('family_block').family_block.transform('size').to_numpy()
        b=np.linalg.lstsq(X*np.sqrt(w[:,None]),residual*np.sqrt(w),rcond=None)[0]
        # A separate polynomial fit checks the centered depth calculation.
        independent=np.polyfit(q.z_central,residual,1,w=np.sqrt(w))
        assert abs(independent[0]*10-b[-1])<1e-12
        samples=blocks_bootstrap(X,residual,q.family_block.to_numpy(),5000)
        case=f'Liu_depth_{len(rows):02d}';boots[case]=samples
        ci=np.quantile(samples[:,-1],[.025,.975])
        rows.append(dict(case_id=case,ML=x,selection=label,n_events=len(q),n_families=q.family_block.nunique(),
            median_residual=float(np.median(residual)),intercept_at_15km=float(b[0]),
            slope_per_10km=float(b[-1]),slope_lo=float(ci[0]),slope_hi=float(ci[1])))
pd.DataFrame(rows).to_csv(T/'potency_Liu_depth_residual_comparison.csv',index=False)
np.savez_compressed(T/'potency_Liu_depth_bootstrap.npz',**boots)
rigidity=[]
for model in ['Kim1983','Kim2011']:
    m=CONFIG['models'][model]
    for i,(top,vs) in enumerate(zip(m['layer_tops_km'],m['Vs_km_s'])):
        bottom=m['layer_tops_km'][i+1] if i+1<len(m['layer_tops_km']) else None
        rigidity.append(dict(model=model,top_km=top,bottom_km=bottom,Vs_km_s=vs,
            mu_GPa=float(CONFIG['density_kg_m3']*(vs*1000)**2/1e9),
            n_paired=int(((g.z_central>=top)&(g.z_central<bottom if bottom is not None else True)).sum())))
mu=CONFIG['density_kg_m3']*g.source_Vs_Kim1983**2
assert np.allclose(mu/1e9,g.mu_GPa_Kim1983,atol=1e-12)
assert np.allclose(g.Mp_Liu_Kim1983,(2/3)*(np.log10(g.M0_Kim1983_Nm/mu)+1.4563),atol=1e-11)
literature=[
 dict(citation='Liu et al. (2026), GRL',DOI='10.1029/2026GL124983',
      primary_URL='https://onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2026GL124983',
      local_main='references/Liu2026_GRL.pdf',main_PDF_page=8,
      local_supplement='references/Liu2026_GRK_SM.pdf',supplement_PDF_page=15,figure='S7',
      result='Reports no clear systematic hypocentral-depth trend in residuals MP minus the published quadratic ML relation, across four datasets.',
      scope='Three newly analyzed regional datasets plus the existing regional potency catalog. The regional-catalog subset in S7 is restricted to ML>3 and within-bin outliers greater than 0.5 MP are removed. This is not proof of zero depth effects in every region.',
      access='Supplied full main paper and supplement inspected; S7 rendered and viewed.'),
 dict(citation='Dowrick and Rhoades (1998), Bulletin NZSEE 31, 260-280',DOI='10.5459/bnzsee.31.4.260-280',
      primary_URL='https://bulletin.nzsee.org.nz/article/view/533/510',PDF_page=18,printed_page=277,equation=11,
      result='Mw=0.96+0.84 ML-0.0055(hc-25), with centroid depth hc in km; reported depth-coefficient uncertainty 0.0015 and residual standard deviation 0.31.',
      scope='Larger historical/instrumental New Zealand earthquakes, roughly M5 and above and extending to much greater depths. Same sign as the Korean trend, but not a transferable Korean calibration.',
      access='Primary full-text publisher PDF and equation inspected.'),
 dict(citation='Rhoades et al. (2021), BSSA 111, 1056-1066',DOI='10.1785/0120200252',
      title='A Depth-Dependent Local Magnitude Scale for New Zealand Earthquakes Consistent with Moment Magnitude',
      primary_URL='https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/111/2/1056/593423/A-Depth-Dependent-Local-Magnitude-Scale-for-New',
      result='NZ20 adds a hypocentral-depth explanatory variable to ML attenuation to improve consistency with Mw over the New Zealand depth range.',
      scope='Regional calibration with heterogeneous crustal/subduction depths; does not validate its depth coefficients for Korean crustal earthquakes.',
      access='Primary publisher abstract verified through search; full text not retrieved.'),
 dict(citation='Lin, Huang and Wu (2026), TAO 37, 18',DOI='10.1007/s44195-026-00136-x',
      title='Discussion of local magnitude scale for deep earthquakes in Taiwan',
      primary_URL='https://link.springer.com/article/10.1007/s44195-026-00136-x',
      result='Hypocentral-distance-only ML attenuation leaves depth-dependent residuals. Including log depth produces recalibrated ML with no remaining depth trend and approximately one-to-one Mw scaling; reported scatter 0.16.',
      scope='649 events, Mw 3.0-6.6, depths 35-313 km; demonstrates calibration dependence, not a numerical correction for Korea.',
      access='Primary publisher full text inspected.'),
 dict(citation='Altindal et al. (2025), BSSA 115, 2385-2406',DOI='10.1785/0120250032',
      primary_URL='https://gfzpublic.gfz.de/pubman/item/item_5036752_1/component/file_5036760/5036752.pdf?mode=download',
      result='Station-ML minus event-ML residuals become negative for deeper-than-10-km, Foreland-dominated data; authors emphasize regional/path dependence.',
      scope='Related caution: station/event ML residuals are NOT direct Mw-ML or Mp-ML residuals. Do not present this as direct evidence for the question.',
      access='Primary institutional full-text PDF inspected, discussion and Figure 15.')]
audit=dict(passed=True,source_density_kg_m3=CONFIG['density_kg_m3'],source_velocity_config_sha256=CONFIG_SHA256,
    source_rigidity_layers=rigidity,
    plotted_MP='Mp_Liu_Kim1983 = (2/3)*(log10(P0[m3])+1.4563), identical to Liu equation 2 after unit conversion.',
    reference_rigidity_GPa=float(10**(9.10+1.4563)/1e9),
    other_28GPa_reference='Retained in explicitly named Mpo28 columns; not used as the MP axis in the Liu comparison.',
    C_status='The Liu MP axis uses the exact published definition. Current Mw retains C=9.05. There is no extra C or 28-to-36-GPa correction to apply to the plotted MP.',
    offset_statistics=offsets,
    measured_ML_minus_KMA_median=float((g.ML_Sheen-g.ML_KMA).median()),
    two_model_MP_median_absolute_difference=float(abs(g.Mp_Liu_Kim2011-g.Mp_Liu_Kim1983).median()),
    Liu_S7_analogue=rows,literature=literature,
    interpretation='A matched MP magnitude zero point rules out a reference-rigidity convention mismatch. It does not validate source amplitudes or ML calibration. P/S source inference, velocity/path/site corrections, amplitude conventions and selected event populations differ. A pooled cross-region curve does not establish identical regional offsets.',
    prior_omission='The prior depth discussion did not highlight Liu main-text p.8 and Figure S7. These directly relevant results have now been added to the report reading notes and discussion.',
    adopted_Mw_ML_changed=False,
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [T/'magnitude_followup_events.csv',R/'references/Liu2026_GRL.pdf',R/'references/Liu2026_GRK_SM.pdf']})
(T/'potency_rigidity_literature_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['passed','reference_rigidity_GPa','offset_statistics','Liu_S7_analogue']},indent=2))
