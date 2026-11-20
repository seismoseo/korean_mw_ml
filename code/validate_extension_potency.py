"""Independently verify scalar potency, joint regressions and report parity."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import fitz
from potency_model import source_vs,CONFIG_SHA256

R=Path(__file__).resolve().parents[1];T=R/'small_events/extension/tables'
Y={'Mw_Kim1983':'Mw','Mw_Kim2011':'Mw_source_Vs_Kim2011','Mp_Kim1983':'Mp_Liu_Kim1983','Mp_Kim2011':'Mp_Liu_Kim2011'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(p):return pd.read_csv(p,dtype={'event_id':str})
def read(p):return json.loads(p.read_text())
def close(a,b,tol=1e-10):return np.allclose(a,b,atol=tol,rtol=0,equal_nan=True)


def bins(x,y,n,minimum):
    edges=np.linspace(min(x),max(x),n+1);edges[-1]=np.nextafter(edges[-1],np.inf)
    labels=pd.cut(x,edges,right=False,include_lowest=True)
    out=[]
    frame=pd.DataFrame(dict(x=x,y=y,label=labels))
    for _,q in frame.groupby('label',observed=True):
        keep=q[abs(q.y-q.y.median())<=.5]
        if len(keep)>=minimum:out.append([keep.x.median(),keep.y.median()])
    return np.asarray(out)


def main():
    audit=read(T/'potency_analysis_validation.json');controls=read(T/'potency_controls_validation.json')
    assert audit['passed'] and controls['passed']
    assert controls['analysis_validation_sha256']==sha(T/'potency_analysis_validation.json')
    assert controls['source_sha256']==sha(R/'code/extension_potency_controls.py')
    assert controls['station_amplitudes_sha256']==sha(T/'ML_station_measurements.csv')
    assert controls['original_regional_engine_sha256']==sha(R/'code/potency_regional_controls.py')
    for name,digest in audit['protected_inputs_sha256'].items():assert sha(R/name)==digest,name
    for name,digest in audit['sources_sha256'].items():assert sha(R/name)==digest,name
    for proof in [audit,controls]:
        for name,digest in proof['outputs_sha256'].items():assert sha(T/name)==digest,name
    original=csv(T/'NW2_Mw_ML_catalog.csv').set_index('event_id');d=csv(T/'potency_event_catalog.csv').set_index('event_id',drop=False)
    assert len(d)==3532 and d.index.is_unique and set(d.index)==set(original.index)
    for col in ['Mw','ML_Sheen','M0_Nm','z_central','ml']:
        if col=='M0_Nm':assert np.allclose(d[col],original.loc[d.index,col],rtol=2e-14,atol=0,equal_nan=True),col
        else:assert close(d[col],original.loc[d.index,col]),col
    assert d.Mp_quality.eq(d.Mw_quality).all() and d.Mp_Liu_Kim1983.isna().eq(d.Mw.isna()).all()
    finite=d[d.Mw.notna()];errors={}
    assert len(finite)==2000 and finite.Mp_quality.eq('A').sum()==801
    for model in ['Kim1983','Kim2011']:
        beta=source_vs(finite.z_central.to_numpy(),model);moment=finite['M0_'+model+'_Nm']
        p0=moment/(2800*beta*beta);direct=4*np.pi*beta*finite.omega_r_m2s/(2*.63)
        error=max(abs(p0/direct-1));assert error<1e-11
        assert close(p0,finite['P0_m3_'+model],1e-7)
        printed=(np.log10(p0/1e4)+5.4563)/1.5
        assert close(printed,finite['Mp_Liu_'+model])
        assert close(finite['Mpo28_C9p05_'+model]-finite['Mpo28_C9p10_'+model],.05/1.5)
        for endpoint in ['lower','upper']:
            endpoint_vs=source_vs(np.maximum(finite[endpoint],0),model)
            ep_mw=finite.Mw_reference_Vs3500+2*np.log10(endpoint_vs/3500)
            ep_mp=(1.5*ep_mw+9.05-np.log10(2800*endpoint_vs**2)+1.4563)/1.5
            assert close(ep_mp,finite[f'Mp_Liu_depth_{endpoint}_{model}'])
        mpwidth=finite['Mp_Liu_bootstrap_hi_'+model]-finite['Mp_Liu_bootstrap_lo_'+model]
        mwwidth=finite['Mw_station_bootstrap_hi']-finite['Mw_station_bootstrap_lo']
        assert close(mpwidth,mwwidth)
        errors[model]=float(error)
    assert close(finite.Mp_Liu_Kim2011-finite.Mp_Liu_Kim1983,(finite.Mw_source_Vs_Kim2011-finite.Mw)/3)
    g=d[d.paired_A&d.ML_Sheen.ge(1.5)].reset_index(drop=True)
    assert len(g)==676 and g.family_block.nunique()==374
    members=csv(T/'magnitude_scaling_members.csv');frames={k:q.reset_index(drop=True) for k,q in members.groupby('cohort')}
    old=csv(R/'tables/potency_event_catalog.csv').set_index('event_id')
    prev=frames['previous']
    for model in ['Kim1983','Kim2011']:
        assert close(prev['Mp_Liu_'+model],old.loc[prev.event_id,'Mp_Liu_'+model])
    f=csv(T/'magnitude_scaling_regressions.csv');pr=csv(T/'magnitude_scaling_predictions.csv')
    bm=csv(T/'magnitude_scaling_bins.csv');bootstrap=np.load(T/'magnitude_scaling_bootstrap.npz')
    assert len(f)==480 and f.case_id.is_unique and not pr.duplicated(['case_id','target_set','event_id']).any()
    point_errors=[];odr_errors=[]
    for row in f.itertuples():
        frame=frames[row.cohort];xcol='ML_Sheen' if row.axis=='our' else 'ml';ycol=Y[row.response]
        x=frame[xcol].to_numpy();y=frame[ycol].to_numpy();w=1/frame.groupby('family_block').event_id.transform('size').to_numpy() if row.objective.startswith('family') else np.ones(len(frame))
        b=bins(x,y,row.n_bins,row.minimum);saved=bm[bm.case_id.eq(row.case_id)].sort_values('bin_index')
        assert close(b,saved[['ML','magnitude']]),row.case_id
        if row.objective.startswith('bin'):x,y=b[:,0],b[:,1];w=np.ones(len(b))
        X=np.polynomial.polynomial.polyvander(x-2,row.degree)
        if row.objective.endswith('OLS'):
            coef=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
            error=max(abs(coef-np.array([row.a,row.b,row.c])[:row.degree+1]));assert error<1e-9,(row.case_id,error)
            point_errors.append(error)
        elif row.degree==1:
            mx=np.average(x,weights=w);my=np.average(y,weights=w)
            vx=np.average((x-mx)**2,weights=w);vy=np.average((y-my)**2,weights=w);cov=np.average((x-mx)*(y-my),weights=w)
            slope=(vy-vx+np.sqrt((vy-vx)**2+4*cov*cov))/(2*cov)
            assert close([row.a,row.b],[my-slope*(mx-2),slope],2e-5)
        elif row.cohort in ['previous','expanded_NW2']:
            # Independent closest-point cubic and coefficient normal equations.
            a,beta,c=row.a,row.b,row.c;projected=[]
            for xx,yy in zip(x,y):
                roots=np.roots([2*c*c,3*beta*c,1+beta*beta+2*c*(a-yy),2-xx+beta*(a-yy)])
                roots=roots.real[abs(roots.imag)<1e-7];assert len(roots)
                loss=(roots+2-xx)**2+(a+beta*roots+c*roots**2-yy)**2
                projected.append(roots[np.argmin(loss)])
            u=np.asarray(projected);res=a+beta*u+c*u*u-y
            gradient=np.sum(w[:,None]*res[:,None]*np.column_stack([np.ones(len(u)),u,u*u]),axis=0)/sum(w)
            error=max(abs(gradient));assert error<1e-4,(row.case_id,error);odr_errors.append(error)
        shared=pr[pr.case_id.eq(row.case_id)&pr.target_set.eq('shared_NW2')].set_index('event_id').loc[g.event_id]
        assert close(shared.observed,g[ycol]) and close(shared.ML_measured,g.ML_Sheen)
        if row.case_id in bootstrap:
            boot=bootstrap[row.case_id];assert len(boot)==row.n_successful_bootstraps and len(boot)>=900
            assert close(np.quantile(boot[:,2],[.025,.975]),[row.c_lower,row.c_upper])
    # The full Gyeongju family, plus independent fresh families, is omitted from bin fits.
    chosen=['family_0']+g[g.confirmatory_small].sort_values('event_id').family_block.drop_duplicates().head(6).tolist()
    q=f[f.cohort.eq('expanded_NW2')&f.axis.eq('our')&f.objective.eq('bin_OLS')&f.selection.eq('Main methods')]
    for row in q.itertuples():
        for family in chosen:
            train=g[g.family_block.ne(family)];test=g[g.family_block.eq(family)]
            b=bins(train.ML_Sheen.to_numpy(),train[Y[row.response]].to_numpy(),50,1)
            coef=np.polyfit(b[:,0]-2,b[:,1],row.degree)
            stored=pr[pr.case_id.eq(row.case_id)&pr.target_set.eq('shared_NW2')].set_index('event_id').loc[test.event_id]
            assert close(np.polyval(coef,test.ML_Sheen-2),stored.predicted)
    # Reconstruct all 108 joint depth designs independently from their column names.
    depth=csv(T/'potency_depth_regressions.csv');db=np.load(T/'potency_depth_bootstrap.npz')
    assert len(depth)==108
    for row in depth.itertuples():
        part=frames[row.cohort].copy();control=row.control
        if control=='Without Gyeongju/Pohang':part=part[~part.family_block.isin(['family_0','family_1'])]
        if control=='Depth below 15 km':part=part[part.z_central.lt(15)]
        if control=='Depth at least 15 km':part=part[part.z_central.ge(15)]
        z=part.z_central.to_numpy();response=part[row.y].to_numpy()
        if control.endswith('depth endpoints'):
            side='lower' if control.startswith('Lower') else 'upper';z=part[side].to_numpy();model='Kim2011' if row.y.endswith('Kim2011') else 'Kim1983'
            response=part[f'Mp_Liu_depth_{side}_{model}' if row.y.startswith('Mp_') else f'Mw_at_depth_{side}_{model}'].to_numpy()
        if control=='Raw magnitude difference':response=response-part[row.x].to_numpy()
        m=part[row.x].to_numpy()-3
        columns={'intercept':np.ones(len(part)),'ML_minus_3':m,'ML_minus_3_squared':m*m,'depth_10km':(z-15)/10,
            'depth_hinge_15km':np.maximum(z-15,0)/10,'latitude_minus_36':part.lat.to_numpy()-36,'longitude_minus_128':part.lon.to_numpy()-128,
            'log10_ML_median_distance_over_50':np.log10(part.ML_median_distance.to_numpy()/50),
            'ML_HG_fraction':part.ML_HG_fraction.to_numpy(),'Mw_HG_fraction':part.Mw_HG_fraction.to_numpy(),
            'Kim1983_layer_15':(z>=15).astype(float),'Kim2011_layer_7p29':(z>=7.29).astype(float),'Kim2011_layer_20p7':(z>=20.7).astype(float)}
        size=1 if control.startswith('One') else .5
        cells=np.floor(part.lat/size).astype(int).astype(str)+'_'+np.floor(part.lon/size).astype(int).astype(str)
        names=json.loads(row.names)
        X=np.column_stack([(cells.eq(name[5:]).astype(float).to_numpy() if name.startswith('cell_') else columns[name]) for name in names])
        w=1/part.groupby('family_block').event_id.transform('size').to_numpy()
        coef=np.linalg.lstsq(X*np.sqrt(w[:,None]),response*np.sqrt(w),rcond=None)[0]
        assert close(coef,json.loads(row.coefficients),1e-8),row.case_id
        assert close(np.quantile(db[row.case_id][:,-1],[.025,.975]),[row.depth_lo,row.depth_hi])
    # Repeat the earlier attenuation/distance and gain control on the saved peaks.
    stations=csv(T/'ML_station_measurements.csv');z=stations.event_id.map(d.z_central)
    rh=np.hypot(stations.epicentral_km,z)
    hbut=np.log10(stations.A_H_mm)+1.110*np.log10(rh/100)+.00189*(rh-100)+3
    recomputed=pd.DataFrame(dict(event_id=stations.event_id,value=hbut)).groupby('event_id').value.median()
    regional=csv(T/'potency_ML_definition_controls.csv').set_index('event_id')
    assert close(regional.loc[g.event_id,'ML_HB87_same_WA2080'],recomputed.loc[g.event_id])
    assert close(regional.loc[g.event_id,'ML_HB87_WA2800_gain_control']-regional.loc[g.event_id,'ML_HB87_same_WA2080'],np.log10(2800/2080))
    rf=csv(T/'potency_regional_regressions.csv');assert len(rf)==32
    for _,part in rf.groupby(['y','method','degree']):
        assert np.ptp(part.quadratic_c)<1e-10 and np.ptp(part.CV_RMS_families)<1e-10
    report=read(T/'report_validation.json');assert report['passed']
    assert report['potency_analysis_validation_sha256']==sha(T/'potency_analysis_validation.json')
    assert report['potency_controls_validation_sha256']==sha(T/'potency_controls_validation.json')
    assert report['potency_report_source_sha256']==sha(R/'code/report_extension_potency.py')
    for name,record in report['reports'].items():
        assert sha(R/name)==record['sha256']
        with fitz.open(R/name) as pdf:assert len(pdf)==record['pages']
    with fitz.open(R/'small_events/extension/reports/Potency_magnitude_and_quadratic_scaling.pdf') as pdf:
        assert len(pdf)==11;text='\n'.join(p.get_text() for p in pdf)
        for word in ['1.4563','28 GPa','9.05','Hong','Depth','480','Mpv']:assert word in text,word
    result=dict(passed=True,attempted=3532,finite_potency=2000,paired_events=676,paired_families=374,
        physical_formula_relative_errors=errors,units_and_normalizations_verified=True,source_depth_models_verified=True,
        missing_values_and_inherited_quality_verified=True,previous_potency_values_reproduced=True,
        regression_cases=480,previous_native_cases_reproduced=audit['previous_native_cases_reproduced'],
        independent_OLS_max_error=float(max(point_errors)),independent_ODR_max_normal_gradient=float(max(odr_errors)),
        independent_bins_and_Gyeongju_holdouts_verified=True,independent_depth_designs_verified=108,
        same_amplitude_ML_definition_controls_verified=True,regional_ML_regression_cases=32,
        report_validation_sha256=sha(T/'report_validation.json'),analysis_validation_sha256=sha(T/'potency_analysis_validation.json'),
        controls_validation_sha256=sha(T/'potency_controls_validation.json'),source_sha256=sha(Path(__file__)),
        reports=report['reports'])
    (T/'potency_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
