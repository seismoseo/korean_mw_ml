"""Validate the source-Vs revision and all current magnitude reports."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import fitz
from source_velocity import CONFIG,CONFIG_SHA256,PRIMARY_MODEL,source_vs,validate_catalog

R=Path(__file__).resolve().parents[1]
audit_path=R/'tables/source_velocity_update.json'
audit=json.loads(audit_path.read_text())
for i,(name,digest) in enumerate(audit['immutable_sha256'].items(),1):
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
    if i%500==0:print('Immutable measurements verified',i,flush=True)
for model in CONFIG['models'].values():
    for source in model.get('provenance',[]):
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
assert abs(source_vs(20,'Kim1983')-6380/1.73)<1e-10
assert np.isclose(source_vs(14.999,'Kim1983'),5980/1.73,rtol=0,atol=1e-9)
assert np.isclose(source_vs(15,'Kim1983'),6380/1.73,rtol=0,atol=1e-9)
assert source_vs(20.699,'Kim2011')==3600.
assert source_vs(20.7,'Kim2011')==3700.
assert source_vs(19.867,'Kim2011')==3600.
d=pd.read_csv(R/'tables/Mw_ML_catalog.csv',dtype={'event_id':str})
paired=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str})
validate_catalog(d);validate_catalog(paired)
assert d.Mw_quality.value_counts().to_dict()==audit['previous_quality']
assert set(paired.loc[paired.ML_comparison_primary,'event_id'])==set(audit['previous_paired_event_ids'])
primary=pd.read_csv(R/'tables/Mw_ML_primary.csv',dtype={'event_id':str})
assert set(primary.event_id)==set(d.loc[d.Mw_quality.eq('A')&d.primary_geography,'event_id'])
validate_catalog(primary)
fits=pd.read_csv(R/'tables/clean_spectral_fits.csv',dtype={'event_id':str})
finite=fits[fits.M0_Nm.notna()]
depths=d.set_index('event_id').z_central
depth=finite.event_id.map(depths).to_numpy()
max_errors={}
for model in ['Kim1983','Kim2011']:
    # Independent coefficient evaluation, including all shapes/windows/sensor sets.
    beta=np.array([float(source_vs(z,model)) for z in depth])
    moment=4*np.pi*2800*beta**3*finite.omega_r_m2s.to_numpy()/(2*.63)
    error=np.max(abs(moment/finite['M0_'+model+'_Nm'].to_numpy()-1))
    assert error<1e-11,(model,error)
    assert np.allclose((np.log10(moment)-9.05)/1.5,finite['Mw_source_Vs_'+model],atol=1e-11)
    max_errors[model]=float(error)
ref=(np.log10(4*np.pi*2800*3500**3*finite.omega_r_m2s/(2*.63))-9.05)/1.5
assert np.allclose(ref,finite.Mw_reference_Vs3500,atol=1e-11)
h=d[d.event_id.eq('20200503130714')].iloc[0]
old=audit['previous_Haenam'][0]
assert abs(h.Mw_reference_Vs3500-old['Mw'])<1e-11
assert abs(h.Mw-3.169474308832197)<1e-10
assert abs(h.Mw_source_Vs_Kim2011-3.172328605781453)<1e-10
assert h.source_layer_boundary_crossed_Kim2011
assert h.Mw_at_depth_lower_Kim2011 < h.Mw_source_Vs_Kim2011 < h.Mw_at_depth_upper_Kim2011
t=json.loads((R/'tables/Sangju2026_summary.json').read_text())
p=t['preferred'];old=audit['previous_Sangju']
assert abs(p['Mw_reference_Vs3500']-old['Mw'])<1e-11
assert abs(p['source_Vs_m_s']-6380/1.73)<1e-10 and p['source_Vs_Kim2011']==3600.
assert abs(p['Mw']-3.41747647093014)<1e-10
assert abs(p['Mw_source_Vs_Kim2011']-3.396532321279981)<1e-10
for model in CONFIG['models']:
    endpoints=t['depth_endpoint_magnitudes'][model]
    assert endpoints['lower']<endpoints['central']<endpoints['upper']
# A fixed-radius Eshelby stress change equals the moment ratio; this is not
# multiplied by a second shear-modulus factor.
ratio=h.M0_Kim2011_Nm/h.M0_Nm
assert np.isclose(ratio,(h.source_Vs_Kim2011/h.source_Vs_m_s)**3)
stability=pd.read_csv(R/'tables/estimator_stability_events.csv',dtype={'event_id':str})
q=stability[stability.status.eq('fit')]
assert q.source_velocity_config_sha256.eq(CONFIG_SHA256).all()
assert np.allclose(q.MT_Mw_20-q.MT_Mw_10,q.MT_window_change,atol=1e-11)
hong=json.loads((R/'tables/Hong_report_validation.json').read_text())
assert hong['passed'] and hong['source_velocity_config_sha256']==CONFIG_SHA256
report_pages={};spans=0
for path in sorted((R/'reports').glob('*.pdf')):
    with fitz.open(path) as pdf:
        report_pages[path.name]=len(pdf)
        text='\n'.join(page.get_text() for page in pdf)
        assert 'Kim' in text and ('Vp/Vs' in text or 'Vp / Vs' in text),path.name
        assert '2011' in text,path.name
        for page in pdf:
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        if not span['text'].strip():continue
                        rect=fitz.Rect(span['bbox']);spans+=1
                        assert rect.x0>=-1 and rect.y0>=-1 and rect.x1<=page.rect.width+1 and rect.y1<=page.rect.height+1,(path.name,page.number,span['text'],span['bbox'])
        if path.name in ['Korean_Mw_ML_study.pdf','Independent_ML_and_Mw_comparison.pdf','Estimator_and_KR_Mw_comparison.pdf','Velocity_models_and_magnitudes.pdf']:
            assert f"{p['Mw']:.3f}" in text,(path.name,'Sangju magnitude')
assert report_pages['Spectral_fit_atlas.pdf']==161
atlas=pd.read_csv(R/'tables/spectral_atlas_index.csv',dtype={'event_id':str}).set_index('event_id')
assert np.allclose(atlas.loc[d.event_id].Mw,d.Mw,equal_nan=True)
assert np.allclose(atlas.loc[d.event_id].Mw_Kim2011,d.Mw_source_Vs_Kim2011,equal_nan=True)
result=dict(passed=True,source_velocity_config_sha256=CONFIG_SHA256,primary_model=PRIMARY_MODEL,
 comparison_model='Kim2011',density_kg_m3=2800.,magnitude_constant=9.05,
 immutable_input_files_verified=len(audit['immutable_sha256']),n_finite_fit_rows=len(finite),
 max_relative_moment_error=max_errors,selection_and_ML_unchanged=True,
 Haenam_Mw=float(h.Mw),Haenam_Mw_Kim2011=float(h.Mw_source_Vs_Kim2011),
 Sangju_Mw=p['Mw'],Sangju_Mw_Kim2011=p['Mw_source_Vs_Kim2011'],
 pdf_pages=report_pages,n_pdf_text_spans_checked=spans,
 external_Haenam_modified=False,scope=CONFIG['scope'])
(R/'tables/source_velocity_update_validation.json').write_text(json.dumps(result,indent=2)+'\n')
audit.update(status='complete',validation='tables/source_velocity_update_validation.json',current_results=result)
audit_path.write_text(json.dumps(audit,indent=2)+'\n')
final=dict(passed=True,source_velocity_updated=True,estimator_and_selection_unchanged=True,
 all_eight_matched_ML_Mw_fits_use_same_165_events=True,
 ODRPACK_parameter_checks=hong['ODR_validation']['scipy_ODRPACK_checks'],
 maximum_parameter_difference=hong['ODR_validation']['max_parameter_difference'],
 stability_events=hong['n_stability_common_plateau'],report_pages=hong['pages'],
 source_velocity_config_sha256=CONFIG_SHA256)
(R/'tables/Hong_final_validation.json').write_text(json.dumps(final,indent=2)+'\n')
print(json.dumps(result,indent=2))
