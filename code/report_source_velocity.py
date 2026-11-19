"""Source-velocity model comparison, with one frozen depth/measurement cohort."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import fitz
from source_velocity import CONFIG,CONFIG_SHA256,source_vs,validate_catalog

R=Path(__file__).resolve().parents[1]
d=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str})
validate_catalog(d)
a=d[(d.Mw_quality=='A')&d.primary_geography].copy()
g=d[d.ML_comparison_primary].copy()
reg=pd.read_csv(R/'tables/Hong_regression_comparison.csv')
target=json.loads((R/'tables/Sangju2026_summary.json').read_text())
h=d[d.event_id.eq('20200503130714')].iloc[0]
BLUE='#176da0';ORANGE='#bd591b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})

def regression(x,y,selection='Our full paired A',weight='event'):
    q=reg[(reg.selection==selection)&(reg.x==x)&(reg.y==y)&(reg.method=='ODR')&(reg.weighting==weight)]
    assert len(q)==1
    return q.iloc[0]

fig,axes=plt.subplots(2,2,figsize=(11,9.2))
fig.subplots_adjust(left=.1,right=.96,top=.87,bottom=.18,hspace=.42,wspace=.33)
fig.suptitle('Source velocity and moment magnitude',fontsize=15,y=.96)
ax=axes[0,0];z=np.linspace(0,35,1401)
for key,label,color in [('Kim1983','Kim & Kim (1983), Vp/Vs 1.73',BLUE),('Kim2011','Kim et al. (2011)',ORANGE)]:
    ax.plot(source_vs(z,key)/1000,z,c=color,lw=1.5,label=label)
ax.axvline(3.5,c='.6',lw=1,ls=':',label='Previous fixed Vs')
ax.set(xlabel='Source Vs (km/s)',ylabel='Depth (km)',ylim=(35,0),xlim=(3.3,4.75))
ax.legend(frameon=False,loc='lower left');ax.grid(color='.92')
ax.set_title('Published / project implementations',loc='left')
ax=axes[0,1];delta=a.Mw_source_Vs_Kim2011-a.Mw
ax.scatter(a.z_central,delta,s=18,c=BLUE,alpha=.65,edgecolors='none')
for boundary in [7.29,15,20.7,31.3]:ax.axvline(boundary,c='.83',ls=':',lw=.8)
ax.axhline(0,c='.5',lw=.8)
ax.set(xlabel='National event depth (km)',ylabel='Mw: Kim 2011 − Kim & Kim 1983',xlim=(0,32))
ax.grid(color='.94');ax.set_title(f'{len(a)} quality-A events; same spectra',loc='left')
ax=axes[1,0]
sc=ax.scatter(a.Mw,a.Mw_source_Vs_Kim2011,c=a.z_central,cmap='viridis',norm=Normalize(0,30),s=22,edgecolors='none',alpha=.8)
low=min(a.Mw.min(),a.Mw_source_Vs_Kim2011.min())-.08;high=max(a.Mw.max(),a.Mw_source_Vs_Kim2011.max())+.08
ax.plot([low,high],[low,high],c='.45',ls=':',lw=1)
ax.set(xlabel='Mw: Kim & Kim (1983), Vp/Vs 1.73',ylabel='Mw: Kim et al. (2011)',xlim=(low,high),ylim=(low,high))
ax.grid(color='.92');fig.colorbar(sc,ax=ax,fraction=.035,pad=.02,label='Depth (km)')
ax=axes[1,1]
for y,label,color in [('Mw','Kim & Kim 1983',BLUE),('Mw_source_Vs_Kim2011','Kim et al. 2011',ORANGE)]:
    rr=regression('ML_Sheen',y);x=np.linspace(g.ML_Sheen.min(),g.ML_Sheen.max(),150)
    ax.scatter(g.ML_Sheen,g[y],s=13,c=color,alpha=.25,edgecolors='none')
    ax.plot(x,rr.intercept+rr.slope*x,c=color,lw=1.5,label=f'{label}: slope {rr.slope:.3f}')
ax.set(xlabel='Measured ML',ylabel='Mw');ax.grid(color='.92');ax.legend(frameon=False,loc='upper left')
ax.set_title(f'{len(g)} identical paired-A events; ODR',loc='left')
caption=('Common density 2800 kg/m³, C=9.05, national depths, measured spectra and path corrections. Only the source Vs³ coefficient changes.\n'
 'Kim & Kim uses the Haenam Vp/Vs=1.73 implementation. Kim 2011 uses published Table 2; the old fixed 3.50 km/s estimate is a control.')
fig.text(.075,.045,caption,fontsize=8.5,color='.25',linespacing=1.4)
for ext in ['pdf','png']:fig.savefig(R/'figures'/('16_source_velocity_models.'+ext))
plt.close(fig)

comparisons=[]
for row in d[d.Mw.notna()].itertuples():
    comparisons.append(dict(event_id=row.event_id,depth_km=row.z_central,Mw_quality=row.Mw_quality,
      Mw_Kim1983=row.Mw,Mw_Kim2011=row.Mw_source_Vs_Kim2011,Mw_fixed_Vs3500=row.Mw_reference_Vs3500,
      Vs_Kim1983_m_s=row.source_Vs_Kim1983,Vs_Kim2011_m_s=row.source_Vs_Kim2011,
      M0_Kim1983_Nm=row.M0_Nm,M0_Kim2011_Nm=row.M0_Kim2011_Nm,
      delta_Mw_Kim2011_minus_Kim1983=row.Mw_source_Vs_Kim2011-row.Mw,
      fixed_radius_stress_ratio_Kim2011_to_Kim1983=row.M0_Kim2011_Nm/row.M0_Nm,
      depth_envelope_crosses_Kim1983_boundary=row.source_layer_boundary_crossed_Kim1983,
      depth_envelope_crosses_Kim2011_boundary=row.source_layer_boundary_crossed_Kim2011))
pd.DataFrame(comparisons).to_csv(R/'tables/source_velocity_model_comparison.csv',index=False)

# Retain the supplied-paper facts, with the old fixed-Vs prefactor explicitly named.
audit_path=R/'tables/Hong_paper_method_audit.json';paper=json.loads(audit_path.read_text())
old=paper.pop('source_prefactor_shift_Hong_minus_ours_Mw',None)
if old is not None:paper['source_prefactor_shift_Hong_minus_fixed_Vs3500_Mw']=old
paper.update(our_source_velocity_model=CONFIG['models']['Kim1983']['label'],
 our_source_velocity_comparison=CONFIG['models']['Kim2011']['label'],
 source_velocity_config_sha256=CONFIG_SHA256,magnitude_values_changed=True)
pref={}
for key in ['Kim1983','Kim2011']:
    values=np.log10(2700/2800)/1.5+2*np.log10(3610/source_vs(a.z_central.to_numpy(),key))
    pref[key]=dict(min=float(values.min()),median=float(np.median(values)),max=float(values.max()))
paper['source_prefactor_shift_Hong_minus_ours_by_model']=pref
audit_path.write_text(json.dumps(paper,indent=2)+'\n')
method_path=R/'tables/Hong_paper_method_comparison.csv';methods=pd.read_csv(method_path)
methods.loc[methods['item'].eq('Density / source Vs'),'our_study']='2800 kg/m3; source-depth Vs(z), Kim & Kim 1983 with Vp/Vs=1.73; published Kim et al. 2011 compared'
methods.to_csv(method_path,index=False)

def paragraph(text):return '<p>'+text+'</p>'
def table(head,rows):
    return '<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

layers=[]
for key in ['Kim1983','Kim2011']:
    m=CONFIG['models'][key]
    for i,(top,vs) in enumerate(zip(m['layer_tops_km'],m['Vs_km_s'])):
        bottom=m['layer_tops_km'][i+1] if i+1<len(m['layer_tops_km']) else 'half-space'
        layers.append([m['label'],f'{top:g}–{bottom}',f'{vs:.4f}'])
p=target['preferred']
examples=[['Haenam 2020',f'{h.z_central:.2f}',f'{h.Mw:.3f}',f'{h.Mw_source_Vs_Kim2011:.3f}',f'{h.Mw_reference_Vs3500:.3f}'],
 ['Sangju 2026','19.76',f'{p["Mw"]:.3f}',f'{p["Mw_source_Vs_Kim2011"]:.3f}',f'{p["Mw_reference_Vs3500"]:.3f}']]
model_reg=[]
for key,label in [('Mw','Kim & Kim 1983, 1.73'),('Mw_source_Vs_Kim2011','Kim et al. 2011')]:
    for weighting in ['event','family']:
        rr=regression('ML_Sheen',key,weight=weighting)
        model_reg.append([label,weighting,f'{rr.intercept:.4f}',f'{rr.slope:.4f}',f'{rr.slope_lo:.4f}–{rr.slope_hi:.4f}'])
pages=[]
pages.append('<h1>Source velocity: two explicit models</h1>'+paragraph('Updated 19 November 2026. This revision replaces the fixed 3.50 km/s source coefficient throughout the current Mw catalog, estimator controls, atlas and regression reports. Measured spectra, station selections, depth estimates, ML and the published Hong magnitudes are preserved.')+
 '<h2>Reference and comparison</h2>'+paragraph('The reference is the exact Kim & Kim (1983) implementation used by the Haenam relocation and rupture project: Vp=5.98/6.38/7.95 km/s and Vp/Vs=1.73. The comparison uses the regional average final S-wave velocities in Kim et al. (2011), Table 2. A common density of 2800 kg/m³ isolates the source-Vs effect; native model-file densities are not substituted.')+
 table(['Model','Depth interval (km)','Vs (km/s)'],layers)+
 paragraph('At an exact boundary the deeper layer is used. Each event retains its national z_central; the held-out Sangju depth is 19.76 km. The old Kim1983 file with lower-crustal Vs=3.79 km/s is a different implementation and is not the Haenam Vp/Vs=1.73 model. The old Kim2011 file also differs slightly from the published table in other layers.')+
 '<h2>What is recomputed</h2>'+paragraph('M0=4πρVs(z)³Ωr/(2×0.63); Mw=(log₁₀M0−9.05)/1.5. All estimators, source shapes, windows and sensor/aperture controls use the same event-specific coefficient. Kim2011 results and the old fixed-3.50 km/s control have explicit columns. New OLS/ODR fits use the updated values.')+
 paragraph('This is a source-coefficient comparison within the existing analytical spreading and empirical Q corrections. It is not a new relocation or a layered Green-function analysis. No empirical magnitude adjustment is fitted to Hong or KMA.'))
pages.append('<h1>Magnitude, depth and stress drop</h1>'+table(['Event','National depth (km)','Mw, Kim & Kim','Mw, Kim 2011','Mw, fixed Vs'],examples)+
 paragraph(f'Across {len(a)} quality-A events in the primary geographic domain, Kim2011 minus Kim & Kim gives median ΔMw={delta.median():+.4f}, range {delta.min():+.4f} to {delta.max():+.4f}. The adjustment depends on layer and depth; it is not one constant magnitude offset.')+
 '<h2>Haenam: retain the depth context</h2>'+paragraph('The national-study depth is 21.26 km, below the Kim2011 boundary at 20.7 km, so Vs=3.70 km/s. The dedicated Haenam TM R5 depth is 19.867 km, above that boundary, where published Kim2011 Vs=3.60 km/s. At a fixed corrected spectrum, the latter coefficient gives Mw 3.149. The dedicated depth is not substituted into one event in the systematic catalog. A full change of location/geometry would be a separate analysis.')+
 paragraph(f'The frozen national depth envelope crosses a source-layer boundary for {int(a.source_layer_boundary_crossed_Kim1983.sum())} Kim & Kim and {int(a.source_layer_boundary_crossed_Kim2011.sum())} Kim2011 quality-A events. Conditional depth-endpoint magnitudes now propagate hypocentral distance and Vs(z), at fixed spectral shape and observed travel times. Those envelopes are not calibrated 95% confidence intervals. Station bootstraps hold the model and central depth fixed.')+
 '<h2>Moment and Eshelby stress drop</h2>'+paragraph('For a fixed source radius a, Δσ=7M0/(16a³). Changing only source velocity therefore changes M0 and stress drop by [Vs,new/Vs,old]³, while ΔMw=2 log₁₀(Vs,new/Vs,old). There is no additional shear-modulus multiplier. If radius is instead recalculated as a=kVs/fc at fixed fc, k and corrected plateau, these direct Vs³ factors cancel.')+
 '<h2>Measured ML and Mw on the same events</h2>'+table(['Model','Weights','ODR a','ODR b','95% interval, b'],model_reg)+
 paragraph('Mw=a+b ML; equal-axis ODR, 2500 family-block resamples. The source-Vs change leaves the sample and estimator-agreement gates unchanged. ODR does not remove SNR/depth selection or shared response/site errors.'))

css='body{font-family:sans-serif;font-size:10pt;line-height:1.32;color:#222;}h1{font-size:19pt;margin:0 0 12pt;}h2{font-size:12pt;margin:12pt 0 6pt;}p{margin:0 0 8pt;}table{width:100%;border-collapse:collapse;font-size:9pt;margin:8pt 0 10pt;}td,th{padding:5pt 4pt;border-bottom:.5pt solid #ccd3d8;text-align:left;}th{background:#edf2f5;}'
doc=fitz.open()
for i,html in enumerate(pages):
    page=doc.new_page(width=595.276,height=841.89)
    spare,scale=page.insert_htmlbox(fitz.Rect(42,30,553,807),html,css=css,scale_low=.93)
    if spare<0:raise RuntimeError('Source-velocity narrative overflow')
    print('Source-model page',i+1,'scale',scale,flush=True)
with fitz.open(R/'figures/16_source_velocity_models.pdf') as part:doc.insert_pdf(part)
path=R/'reports/Velocity_models_and_magnitudes.pdf';doc.save(path,garbage=4,deflate=True)
with fitz.open(path) as output:path.with_suffix('.txt').write_text('\n\n'.join(p.get_text() for p in output))
summary=dict(source_velocity_config_sha256=CONFIG_SHA256,n_quality_A_primary=len(a),n_paired_A=len(g),
 delta_Kim2011_minus_Kim1983=dict(median=float(delta.median()),min=float(delta.min()),max=float(delta.max())),
 Haenam=examples[0],Sangju=examples[1],source_prefactor_Hong_minus_ours=pref,report_pages=len(doc))
(R/'tables/source_velocity_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
