"""Evaluate potency and quadratic magnitude scaling without changing adopted Mw."""
from pathlib import Path
import concurrent.futures,hashlib,json,multiprocessing,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import scipy
from potency_regression import run_case,prediction,binned_medians
from potency_model import CONFIG,CONFIG_SHA256,source_vs

R=Path(__file__).resolve().parents[1]
LIU_OFFSET_SI=1.4563
MU_REF_LIU=10**(9.1+LIU_OFFSET_SI)
PROVENANCE=R/'tables/potency_provenance.json'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(frame):
    d=frame.copy()
    for model in ['Kim1983','Kim2011']:
        mu=2800*d['source_Vs_'+model]**2
        potency=d['M0_'+model+'_Nm']/mu
        d['mu_GPa_'+model]=mu/1e9
        d['P0_m3_'+model]=potency
        d['Mp_Liu_'+model]=(2/3)*(np.log10(potency)+LIU_OFFSET_SI)
        d['Mpo28_C9p10_'+model]=(np.log10(28e9*potency)-9.1)/1.5
        d['Mpo28_C9p05_'+model]=(np.log10(28e9*potency)-9.05)/1.5
        d['Mp36_C9p05_'+model]=(np.log10(36e9*potency)-9.05)/1.5
        # Moment/shape bootstrap conditions on the same model: only a constant
        # changes, so the station-sampling width is identical for Mp and Mw.
        shift=d['Mp_Liu_'+model]-d['Mw_source_Vs_'+model]
        for end in ['lo','hi']:
            name='Mw_station_bootstrap_'+end+('_Kim2011' if model=='Kim2011' else '')
            if name in d:d['Mp_Liu_bootstrap_'+end+'_'+model]=d[name]+shift
        for endpoint in ['lower','upper']:
            name='Mw_at_depth_'+endpoint+'_'+model
            if name in d:
                beta=source_vs(np.maximum(d[endpoint].to_numpy(),0),model)
                log_p=1.5*d[name]+9.05-np.log10(2800*beta**2)
                d['Mp_Liu_depth_'+endpoint+'_'+model]=(2/3)*(log_p+LIU_OFFSET_SI)
    d['Mp_Liu_fixed_Vs3500']=(2/3)*(np.log10(d.M0_reference_Vs3500_Nm/(2800*3500**2))+LIU_OFFSET_SI)
    return d


def main():
    protected=['tables/Mw_ML_catalog.csv','tables/Mw_ML_primary.csv','tables/Mw_independent_ML_catalog.csv',
      'tables/Mw_independent_ML_primary.csv','tables/independent_ML_catalog.csv','tables/Sangju2026_summary.json',
      'tables/clean_spectral_fits.csv','inputs/source_velocity_models.json']
    protected += [str(p.relative_to(R)) for p in (R/'reports').glob('*.pdf') if not p.name.startswith('Potency_')]
    before={s:digest(R/s) for s in protected}
    papers=[]
    for name,doi,pages in [('Trugman2026_SRL.pdf','10.1785/0220250419',10),
                           ('Liu2026_GRL.pdf','10.1029/2026GL124983',13),
                           ('Liu2026_GRK_SM.pdf','Supporting Information, Liu2026',16)]:
        papers.append(dict(file='references/'+name,sha256=digest(R/'references'/name),DOI=doi,pages=pages))
    provenance=dict(date='2026-11-19',protected_current_results_sha256=before,papers=papers,
      source_velocity_config_sha256=CONFIG_SHA256,numpy_version=np.__version__,scipy_version=scipy.__version__,
      scope='Ordinary scalar potency inferred from existing S-wave source spectra and prescribed source mu=rho*Vs^2. No new waveform fit, P-wave Bayesian inversion, or Mpv effective-path-rigidity correction.',
      definitions=dict(P0='M0/(rho*Vs^2), m^3; common controlled density 2800 kg/m^3',
       Mp_Liu='(2/3)*(log10(P0[m^3])+1.4563); exact Liu2026 equation 2, after cm km^2 -> m^3 conversion',
       Mpo28_C9p10='(log10(28e9*P0[m^3])-9.10)/1.5; source-rigidity-only analogue of Trugman2026 equation 5',
       Mpo28_C9p05='Same 28 GPa reference, with user-preferred 9.05; explicit alternative, not used against the published Liu curve',
       Liu_equivalent_reference_rigidity_Pa=MU_REF_LIU),
      published_curve=dict(a2=.040,a1=.556,a0=1.133,range_ML=[-1.8,7.5]),
      paper_regression='50 equally spaced ML bins; trim |Mp - initial bin median| > 0.5; recompute x/y bin medians; unweighted quadratic least squares; covariance/Student-t 95% intervals. This is not ODR.',
      current_uncertainty='Whole national event-family bootstrap, 2000 for primary OLS and 1000 for primary ODR/binned fits. Conditional on accepted spectra/model/depth. Equal-axis ODR expresses a metric assumption.',
      CV='Leave one entire event family out; training weights and binned medians refitted in each fold; vertical prediction RMSE is provided separately from orthogonal fitting loss.',
      archive_access='Paper-cited Zenodo 22092097 landing page and API could not be retrieved with the available web tool. Paper and supplied supplement specify the implemented regression; author code was not claimed to be reproduced.')
    PROVENANCE.write_text(json.dumps(provenance,indent=2)+'\n')
    d=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str})
    missing=d.family_block.isna()
    d.loc[missing,'family_block']=['family_'+str(int(v.family)) if v.family>=0 else 'event_'+v.event_id for v in d[missing].itertuples()]
    d=convert(d)
    target=json.loads((R/'tables/Sangju2026_summary.json').read_text());p=target['preferred'];ml=target['local_magnitude']
    targetrow=dict(event_id='20261117141017',origin_utc='2026-11-17 14:10:17',ML_KMA=3.6,ML_Sheen=ml['ML_Sheen'],
      z_central=19.76,lower=17.68,upper=21.31,ML_quality='A',Mw_quality='held out',family_block='held_out_Sangju',
      primary_geography=True,ML_comparison_primary=False,**{k:p[k] for k in ['Mw','Mw_source_Vs_Kim1983','Mw_source_Vs_Kim2011',
      'M0_Kim1983_Nm','M0_Kim2011_Nm','M0_reference_Vs3500_Nm','Mw_reference_Vs3500','source_Vs_Kim1983','source_Vs_Kim2011','omega_r_m2s']})
    targetframe=convert(pd.DataFrame([targetrow]))
    cat=pd.concat([d,targetframe],ignore_index=True)
    fields=['event_id','origin_utc','ML_KMA','ML_Sheen','Mw_quality','ML_quality','family_block','z_central','lower','upper',
       'primary_geography','ML_comparison_primary','Mw','Mw_source_Vs_Kim2011','Mw_reference_Vs3500','omega_r_m2s']
    fields += [c for c in cat if c.startswith(('P0_','Mp_','Mpo28_','Mp36_','mu_GPa_','source_Vs_','M0_Kim'))]
    cat[fields].to_csv(R/'tables/potency_event_catalog.csv',index=False)
    a=d[d.Mw_quality.eq('A')&d.primary_geography].copy();g=d[d.ML_comparison_primary].copy()
    kr=pd.read_csv(R/'inputs/KR_Mw/Events_KP_2017_2022_Mw.csv',dtype={'evid':str})
    kr['event_id']=kr.evid.str.strip();kr['ML_KMA']=kr.ml_KMA
    for src,label in [('mw_spectra','spectral'),('mw_TD','TD')]:
        logp=1.5*kr[src]+9.05-np.log10(2700*3610**2)
        kr['Mp_Hong_'+label]=(2/3)*(logp+LIU_OFFSET_SI)
    kr['family_block']='KR_event_'+kr.event_id
    kr.to_csv(R/'tables/potency_Hong_event_catalog.csv',index=False)
    matched=kr.drop(columns='family_block').merge(d[['event_id','family_block','Mw_quality','ML_comparison_primary']],on='event_id',validate='one_to_one')
    selections={'Paired A':g,'Mw A':a,'Without Gyeongju and Pohang':g[~g.family.isin([0,1])],
      'No source-layer crossing':g[g.source_layer_boundary_crossed_Kim1983.eq(False)&g.source_layer_boundary_crossed_Kim2011.eq(False)],
      'Hong full':kr,'Hong exact matches':matched,'Hong matched Mw A':matched[matched.Mw_quality.eq('A')],
      'Hong matched paired A':matched[matched.ML_comparison_primary]}
    tasks=[]
    def add(selection,x,y,method='OLS',weighting='event',degree=2,bins=50,minimum=1,bootstrap=0,cv=True):
        q=selections[selection].dropna(subset=[x,y]).copy()
        spec=dict(case_id=f'fit_{len(tasks):03d}',selection=selection,x=x,y=y,method=method,weighting=weighting,
          degree=degree,n_bins=bins,minimum=minimum,bootstrap_reps=bootstrap,do_cv=cv,
          block_policy='Event resampling only; full family membership unavailable' if selection=='Hong full' else 'Whole national event-family blocks')
        tasks.append((spec,q[x].to_numpy(),q[y].to_numpy(),q.family_block.to_numpy(),q.event_id.to_numpy()))
    for x in ['ML_Sheen','ML_KMA']:
        for y in ['Mp_Liu_Kim1983','Mp_Liu_Kim2011']:
            for method in ['OLS','ODR']:
                for weighting in ['event','family']:
                    for degree in [1,2]:add('Paired A',x,y,method,weighting,degree,bootstrap=2000 if method=='OLS' else 1000)
            for bins in [10,20,50]:
                for degree in [1,2]:add('Paired A',x,y,'BinnedOLS',degree=degree,bins=bins,bootstrap=1000 if bins==50 else 0)
        for y in ['Mw','Mw_source_Vs_Kim2011','Mp_Liu_fixed_Vs3500']:
            for degree in [1,2]:add('Paired A',x,y,weighting='family',degree=degree,bootstrap=2000)
        for method in ['Published','PublishedOffset']:add('Paired A',x,'Mp_Liu_Kim1983',method,'family')
        for selection in ['Without Gyeongju and Pohang','No source-layer crossing']:
            for degree in [1,2]:add(selection,x,'Mp_Liu_Kim1983',weighting='family',degree=degree,bootstrap=2000)
        add('Paired A',x,'Mp_Liu_Kim1983','BinnedOLS',bins=50,minimum=3,bootstrap=1000)
    for degree in [1,2]:add('Mw A','ML_KMA','Mp_Liu_Kim1983',weighting='family',degree=degree,bootstrap=2000)
    for selection in ['Hong full','Hong exact matches','Hong matched Mw A','Hong matched paired A']:
        for y in ['Mp_Hong_spectral','Mp_Hong_TD']:
            for method in ['OLS','ODR','BinnedOLS']:
                for degree in [1,2]:add(selection,'ML_KMA',y,method,degree=degree,cv=False,bootstrap=0)
    rows=[];boot={};predictions=[];folds=[];bins=[]
    print('Fitting',len(tasks),'cases',flush=True)
    with concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('fork')) as pool:
        for i,(row,samples,pp,ff,bb) in enumerate(pool.map(run_case,tasks),1):
            rows.append(row)
            if len(samples):boot[row['case_id']]=samples
            predictions+=pp
            folds.extend(dict(case_id=row['case_id'],**x) for x in ff)
            bins.extend(dict(case_id=row['case_id'],ML=x[0],Mp=x[1],n_initial=int(x[2]),n_retained=int(x[3]),bin_index=int(x[4])) for x in bb)
            if i%10==0:print('Completed',i,'/',len(tasks),flush=True)
    fits=pd.DataFrame(rows);fits.to_csv(R/'tables/potency_regressions.csv',index=False)
    np.savez_compressed(R/'tables/potency_bootstrap.npz',**boot)
    pred=pd.DataFrame(predictions);pred.to_csv(R/'tables/potency_heldout_predictions.csv',index=False)
    pd.DataFrame(folds).to_csv(R/'tables/potency_heldout_coefficients.csv',index=False)
    pd.DataFrame(bins).to_csv(R/'tables/potency_binned_medians.csv',index=False)
    # Paired uncertainty in predictive gain, using whole held-out event families.
    cv=[]
    for _,spec in fits[(fits.method.isin(['OLS','ODR','BinnedOLS']))&fits.degree.eq(2)&fits.do_cv].iterrows():
        match=fits[(fits.selection==spec.selection)&(fits.x==spec.x)&(fits.y==spec.y)&(fits.method==spec.method)&
                   (fits.weighting==spec.weighting)&fits.degree.eq(1)&fits.n_bins.eq(spec.n_bins)&fits.minimum.eq(spec.minimum)]
        if len(match)!=1:continue
        base=match.iloc[0]
        left=pred[pred.case_id.eq(base.case_id)].set_index('event_id')
        right=pred[pred.case_id.eq(spec.case_id)].set_index('event_id').loc[left.index]
        blocks=left.family_block.to_numpy();keys=np.unique(blocks)
        mse=np.array([[np.mean(q.residual**2) for _,q in z.groupby('family_block',sort=True)] for z in [left,right]])
        rng=np.random.default_rng(20261122);draw=rng.integers(len(keys),size=(2500,len(keys)))
        gain=np.sqrt(mse[0,draw].mean(axis=1))-np.sqrt(mse[1,draw].mean(axis=1))
        cv.append(dict(selection=spec.selection,x=spec.x,y=spec.y,method=spec.method,weighting=spec.weighting,n_bins=spec.n_bins,
          linear_case_id=base.case_id,quadratic_case_id=spec.case_id,n_blocks=len(keys),
          linear_RMS_families=base.CV_RMS_families,quadratic_RMS_families=spec.CV_RMS_families,
          gain_RMS_families=base.CV_RMS_families-spec.CV_RMS_families,
          gain_lo=float(np.quantile(gain,.025)),gain_hi=float(np.quantile(gain,.975))))
    pd.DataFrame(cv).to_csv(R/'tables/potency_CV_comparison.csv',index=False)
    sensitivity={}
    dm=a.Mw_source_Vs_Kim2011-a.Mw;dp=a.Mp_Liu_Kim2011-a.Mp_Liu_Kim1983
    for name,v in [('Mw',dm),('Mp',dp)]:sensitivity[name]=dict(median_abs=float(abs(v).median()),max_abs=float(abs(v).max()),min=float(v.min()),max=float(v.max()))
    curves=[]
    for label,q,x,y in [('Our paired, measured ML',g,'ML_Sheen','Mp_Liu_Kim1983'),('Our paired, KMA ML',g,'ML_KMA','Mp_Liu_Kim1983'),
                        ('Hong full spectral',kr,'ML_KMA','Mp_Hong_spectral'),('Hong full TD',kr,'ML_KMA','Mp_Hong_TD')]:
        res=q[y]-(.04*q[x]**2+.556*q[x]+1.133)
        curves.append(dict(sample=label,n=len(q),median=float(res.median()),mean=float(res.mean()),RMS=float(np.sqrt(np.mean(res**2)))))
    special=[]
    for q in [d[d.event_id.eq('20200503130714')].iloc[0],targetframe.iloc[0]]:
        special.append({key:(str(q[key]) if key=='event_id' else float(q[key])) for key in ['event_id','Mw','Mw_source_Vs_Kim2011','ML_KMA','ML_Sheen','P0_m3_Kim1983','P0_m3_Kim2011','Mp_Liu_Kim1983','Mp_Liu_Kim2011','Mpo28_C9p10_Kim1983','Mpo28_C9p05_Kim1983']})
    summary=dict(n_historical=len(d),n_Mw_A=len(a),n_paired_A=len(g),n_paired_blocks=g.family_block.nunique(),
      measured_ML_range=[float(g.ML_Sheen.min()),float(g.ML_Sheen.max())],KMA_ML_range=[float(g.ML_KMA.min()),float(g.ML_KMA.max())],
      n_fits=len(fits),n_bootstrapped_fits=len(boot),source_velocity_sensitivity=sensitivity,
      exact_coefficient_sensitivity_ratio='Delta Mp = Delta Mw / 3 for this source-coefficient-only experiment; algebraic, not independent validation',
      Liu_vs_Mpo28_offset=float((2/3)*np.log10(MU_REF_LIU/28e9)),published_curve_residuals=curves,examples=special,
      Mpv_status='Not computed. Requires matched full-waveform Green-function inversions and the actual model density/velocity and dominant periods for characteristic-rigidity estimation.',
      ODR_linear_closed_form_max_error=float(fits.ODR_linear_closed_form_error.max()),
      ODR_quadratic_projection_max_error=float(fits.ODR_projection_stationarity.max()),adopted_Mw_and_reports_modified=False)
    (R/'tables/potency_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,hsh in before.items():assert digest(R/name)==hsh,name
    print(json.dumps(summary,indent=2),flush=True)
    main=fits[(fits.selection=='Paired A')&(fits.y=='Mp_Liu_Kim1983')&fits.degree.eq(2)]
    print(main[['x','method','weighting','n_bins','a_at_ML3','slope_at_ML3','quadratic_c','c_lo','c_hi','CV_RMS_families']].round(6).to_string(index=False),flush=True)

if __name__=='__main__':main()
