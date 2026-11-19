"""Independent unit, regression, held-out prediction and artifact checks."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import fitz
from potency_model import CONFIG_SHA256,source_vs

R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
provenance=json.loads((R/'tables/potency_provenance.json').read_text())
for name,digest in provenance['protected_current_results_sha256'].items():assert sha(R/name)==digest,name
for paper in provenance['papers']:assert sha(R/paper['file'])==paper['sha256']
assert provenance['source_velocity_config_sha256']==CONFIG_SHA256
d=pd.read_csv(R/'tables/potency_event_catalog.csv',dtype={'event_id':str})
assert len(d)==643 and d.event_id.is_unique
q=d[d.Mw.notna()].copy()
max_errors={}
for model in ['Kim1983','Kim2011']:
    beta=source_vs(q.z_central.to_numpy(),model)
    assert np.allclose(beta,q['source_Vs_'+model],rtol=0,atol=1e-9)
    from_moment=q['M0_'+model+'_Nm']/(2800*beta**2)
    direct=4*np.pi*beta*q.omega_r_m2s/(2*.63)
    error=np.max(abs(from_moment/direct-1))
    assert error<1e-11
    assert np.allclose(from_moment,q['P0_m3_'+model],rtol=1e-11)
    # Independently use the mixed units in the printed paper equation.
    printed=(2/3)*(np.log10(from_moment/10000)+5.4563)
    assert np.allclose(printed,q['Mp_Liu_'+model],atol=1e-11)
    assert np.allclose(q['Mpo28_C9p05_'+model]-q['Mpo28_C9p10_'+model],.05/1.5,atol=1e-11)
    max_errors[model]=float(error)
assert np.allclose(q.Mp_Liu_Kim2011-q.Mp_Liu_Kim1983,(q.Mw_source_Vs_Kim2011-q.Mw)/3,atol=1e-11)
h=d[d.event_id.eq('20200503130714')].iloc[0]
s=d[d.event_id.eq('20261117141017')].iloc[0]
assert np.isclose(h.P0_m3_Kim1983,1673.0321269352153,rtol=1e-12)
assert np.isclose(s.P0_m3_Kim1983,3940.102535559297,rtol=1e-12)
g=d[d.ML_comparison_primary].copy()
assert len(g)==357 and g.family_block.nunique()==194
fits=pd.read_csv(R/'tables/potency_regressions.csv')
def fit_row(degree,method='OLS',weight='family'):
    q=fits[fits.selection.eq('Paired A')&fits.x.eq('ML_Sheen')&fits.y.eq('Mp_Liu_Kim1983')&
      fits.method.eq(method)&fits.weighting.eq(weight)&fits.degree.eq(degree)&fits.n_bins.eq(50)&fits.minimum.eq(1)]
    assert len(q)==1
    return q.iloc[0]
r=fit_row(2)
w=1/g.groupby('family_block').family_block.transform('size').to_numpy()
expected=np.polyfit(g.ML_Sheen.to_numpy()-3,g.Mp_Liu_Kim1983.to_numpy(),2,w=np.sqrt(w))
assert np.allclose(expected,[r.quadratic_c,r.slope_at_ML3,r.a_at_ML3],atol=1e-10)
assert r.c_lo<0<r.c_hi
requested=fits[fits.bootstrap_reps>0]
assert (requested.n_successful_bootstraps==requested.bootstrap_reps).all()
assert fits.ODR_linear_closed_form_error.max()<2e-6
assert fits.ODR_projection_stationarity.max()<2e-5
assert fits.monotone_over_observed_range.all()
# Reconstruct paper-style bins with a separate pandas implementation.
edges=np.linspace(g.ML_Sheen.min(),g.ML_Sheen.max(),51);edges[-1]=np.nextafter(edges[-1],np.inf)
gg=g.copy();gg['bin']=pd.cut(gg.ML_Sheen,edges,right=False,include_lowest=True)
med=[]
for _,part in gg.groupby('bin',observed=True):
    keep=part[abs(part.Mp_Liu_Kim1983-part.Mp_Liu_Kim1983.median())<=.5]
    med.append([keep.ML_Sheen.median(),keep.Mp_Liu_Kim1983.median()])
med=np.asarray(med)
expected=np.polyfit(med[:,0]-3,med[:,1],2)
b=fit_row(2,'BinnedOLS','event')
assert np.allclose(expected,[b.quadratic_c,b.slope_at_ML3,b.a_at_ML3],atol=1e-10)
assert b.n_bins_fitted==35 and b.n_singleton_bins==8
# Refit the complete Gyeongju-held-out training set independently.
train=g[g.family_block!='family_0'];test=g[g.family_block=='family_0']
assert len(test)==93
w=1/train.groupby('family_block').family_block.transform('size').to_numpy()
beta=np.polyfit(train.ML_Sheen.to_numpy()-3,train.Mp_Liu_Kim1983.to_numpy(),2,w=np.sqrt(w))
pred=pd.read_csv(R/'tables/potency_heldout_predictions.csv',dtype={'event_id':str})
stored=pred[(pred.case_id==r.case_id)&pred.family_block.eq('family_0')].set_index('event_id').loc[test.event_id]
assert np.allclose(np.polyval(beta,test.ML_Sheen.to_numpy()-3),stored.predicted,atol=1e-10)
for degree in [1,2]:
    rr=fit_row(degree);pp=pred[pred.case_id.eq(rr.case_id)]
    rms=np.sqrt(pp.groupby('family_block').residual.apply(lambda s:float(np.mean(s*s))).mean())
    assert abs(rms-rr.CV_RMS_families)<1e-11
regional=json.loads((R/'tables/potency_regional_controls_summary.json').read_text())
assert regional['original_station_ML_reproduction_error']<1e-10
assert regional['Hong_same_ML_range']==483
controls=pd.read_csv(R/'tables/potency_regional_control_regressions.csv')
cc=controls[(controls.selection=='ML attenuation control')&controls.degree.eq(2)]
assert len(cc)==2
assert np.ptp(cc.quadratic_c)<1e-10 and np.ptp(cc.CV_RMS_families)<1e-10
metadata=json.loads((R/'tables/potency_report_metadata.json').read_text())
path=R/'reports/Potency_magnitude_and_quadratic_scaling.pdf'
spans=0
with fitz.open(path) as doc:
    assert len(doc)==metadata['report_pages']==11
    text='\n'.join(p.get_text() for p in doc)
    for required in ['1.4563','5.4563','28 GPa','9.05','0.040','3.120','3.368','Mpv']:
        assert required in text,required
    for page in doc:
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    if not span['text'].strip():continue
                    b=fitz.Rect(span['bbox']);spans+=1
                    assert b.x0>=-1 and b.y0>=-1 and b.x1<=page.rect.width+1 and b.y1<=page.rect.height+1,(page.number,span['text'],span['bbox'])
assert min(metadata['narrative_scales'])>=.95
audit=dict(passed=True,provided_papers=provenance['papers'],source_velocity_config_sha256=CONFIG_SHA256,
 protected_outputs_verified=len(provenance['protected_current_results_sha256']),
 adopted_magnitudes_and_six_reports_unchanged=True,physical_potency_formula_max_relative_error=max_errors,
 mixed_unit_conversion_verified=True,reference_rigidity_and_C_offsets_verified=True,
 independent_family_weighted_OLS_verified=True,independent_paper_binning_verified=True,
 complete_Gyeongju_holdout_verified=True,bootstrap_success_fraction=1.,
 ODR_linear_closed_form_max_error=float(fits.ODR_linear_closed_form_error.max()),
 ODR_quadratic_projection_max_error=float(fits.ODR_projection_stationarity.max()),
 report_pages=metadata['report_pages'],pdf_text_spans_checked=spans,report_sha256=sha(path),
 Mpv_status='Not estimated; source-only scalar potency is explicitly distinguished.',
 code_sha256={p.name:sha(p) for p in [R/'code'/n for n in ['potency_model.py','potency_regression.py','analyze_potency.py','potency_regional_controls.py','report_potency.py','validate_potency.py']]})
(R/'tables/potency_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
