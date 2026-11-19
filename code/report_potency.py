"""Potency definitions, Korean scaling tests and publication figures."""
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
from potency_regression import prediction
from potency_model import CONFIG,source_vs

R=Path(__file__).resolve().parents[1]
d=pd.read_csv(R/'tables/potency_event_catalog.csv',dtype={'event_id':str})
g=d[d.ML_comparison_primary].copy();a=d[d.Mw_quality.eq('A')&d.primary_geography].copy()
fits=pd.read_csv(R/'tables/potency_regressions.csv')
controls=pd.read_csv(R/'tables/potency_regional_control_regressions.csv')
allfits=pd.concat([fits,controls],ignore_index=True)
cv=pd.read_csv(R/'tables/potency_CV_comparison.csv')
bins=pd.read_csv(R/'tables/potency_binned_medians.csv')
boot=np.load(R/'tables/potency_bootstrap.npz')
summary=json.loads((R/'tables/potency_summary.json').read_text())
regional=json.loads((R/'tables/potency_regional_controls_summary.json').read_text())
BLUE='#176da0';ORANGE='#bd591b';RED='#a63027';GRAY='#666666'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'legend.fontsize':8.2,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':240})
manifest=[]

def row(x='ML_Sheen',y='Mp_Liu_Kim1983',method='OLS',weight='family',degree=2,selection='Paired A',n_bins=50,minimum=1):
    q=allfits[(allfits.selection==selection)&allfits.x.eq(x)&allfits.y.eq(y)&allfits.method.eq(method)&
        allfits.weighting.eq(weight)&allfits.degree.eq(degree)&allfits.n_bins.eq(n_bins)&allfits.minimum.eq(minimum)]
    assert len(q)==1,(x,y,method,weight,degree,selection,n_bins,minimum,len(q))
    return q.iloc[0]

def coef(r):return np.array([r.a_at_ML3,r.slope_at_ML3,r.quadratic_c])
def save(fig,name,caption):
    fig.text(.075,.045,caption,fontsize=8.5,color='.25',linespacing=1.4)
    for ext in ['pdf','png']:fig.savefig(R/'figures'/(name+'.'+ext))
    plt.close(fig);manifest.append(dict(stem=name,caption=caption))

fig,axes=plt.subplots(2,2,figsize=(11,9.5),gridspec_kw={'height_ratios':[1.6,1.]})
fig.subplots_adjust(left=.1,right=.94,top=.87,bottom=.18,wspace=.34,hspace=.36)
fig.suptitle('Potency magnitude and Korean local magnitude',fontsize=15,y=.96)
for j,(xkey,label) in enumerate([('ML_Sheen','Measured ML'),('ML_KMA','KMA ML')]):
    linear=row(x=xkey,degree=1);quadratic=row(x=xkey);binned=row(x=xkey,method='BinnedOLS',weight='event')
    x=np.linspace(g[xkey].min()-.04,g[xkey].max()+.04,250)
    ax=axes[0,j]
    sc=ax.scatter(g[xkey],g.Mp_Liu_Kim1983,c=g.z_central,cmap='viridis',norm=Normalize(0,30),s=18,edgecolors='none',alpha=.65)
    ax.plot(x,prediction(coef(linear),x),c=BLUE,ls='--',lw=1.2,label='Linear, family weights')
    ax.plot(x,prediction(coef(quadratic),x),c=BLUE,lw=1.6,label='Quadratic, family weights')
    ax.plot(x,.04*x*x+.556*x+1.133,c=RED,lw=1.3,label='Liu et al. (2026), unchanged')
    med=bins[bins.case_id.eq(binned.case_id)]
    ax.scatter(med.ML,med.Mp,s=23,marker='s',facecolor='none',edgecolor='.25',lw=.8,label='Filtered bin medians')
    ax.set(xlabel=label,ylabel=r'$M_P$ (Liu normalization)',xlim=(1.8,5.5),ylim=(1.8,5.5));ax.grid(color='.93')
    ax.set_title(f'n = {len(g)}; quadratic c = {quadratic.quadratic_c:.3f}',loc='left')
    ax.legend(frameon=False,loc='upper left',fontsize=7.7)
    bx=axes[1,j]
    base=prediction(coef(linear),g[xkey])
    bx.scatter(g[xkey],g.Mp_Liu_Kim1983-base,c=BLUE,s=15,edgecolors='none',alpha=.5)
    bx.plot(x,prediction(coef(quadratic),x)-prediction(coef(linear),x),c=BLUE,lw=1.5)
    samples=boot[quadratic.case_id]
    X=np.column_stack([np.ones(len(x)),x-3,(x-3)**2])
    curves=samples@X.T-prediction(coef(linear),x)
    lo,hi=np.quantile(curves,[.025,.975],axis=0)
    bx.fill_between(x,lo,hi,color=BLUE,alpha=.12,lw=0)
    bx.axhline(0,c='.5',ls=':',lw=.9)
    bx.set(xlabel=label,ylabel='Residual from linear fit',xlim=(1.8,5.5),ylim=(-.42,.42));bx.grid(color='.94')
    bx.text(.035,.94,f'95% interval for c: {quadratic.c_lo:+.3f} to {quadratic.c_hi:+.3f}',transform=bx.transAxes,va='top',fontsize=8.5)
fig.colorbar(sc,ax=axes[0,:],location='right',fraction=.023,pad=.025,label='Depth (km)')
save(fig,'17_potency_ML_scaling',
 '357 identical paired-A events; 194 event families. Potency comes from existing S-wave spectra and the Haenam Kim & Kim (1983), Vp/Vs=1.73.\n'
 'Both models and magnitude conventions are retained in the tables. Shading resamples whole families; the published Liu curve is not shifted.')

fig,axes=plt.subplots(2,2,figsize=(11,10))
fig.subplots_adjust(left=.18,right=.96,top=.88,bottom=.19,wspace=.55,hspace=.48)
fig.suptitle('Curvature depends on sampling and the fitting method',fontsize=15,y=.96)
specs=[('OLS, event weights','OLS','event','Paired A',1),('OLS, family weights','OLS','family','Paired A',1),
 ('ODR, family weights','ODR','family','Paired A',1),('50 bins','BinnedOLS','event','Paired A',1),
 ('50 bins, >=3 events','BinnedOLS','event','Paired A',3),('Without Gyeongju/Pohang','OLS','family','Without Gyeongju and Pohang',1)]
ax=axes[0,0]
for j,(label,method,weight,selection,minimum) in enumerate(specs):
    for k,(xkey,color) in enumerate([('ML_Sheen',BLUE),('ML_KMA',ORANGE)]):
        rr=row(x=xkey,method=method,weight=weight,selection=selection,minimum=minimum)
        ax.errorbar(rr.quadratic_c,j+(-.11 if k==0 else .11),xerr=[[max(0,rr.quadratic_c-rr.c_lo)],[max(0,rr.c_hi-rr.quadratic_c)]],fmt='o',ms=4,c=color,capsize=2)
ax.axvline(0,c='.5',lw=.8);ax.axvline(.04,c=RED,ls=':',lw=1)
ax.set(yticks=range(len(specs)),yticklabels=[s[0] for s in specs],xlabel='Quadratic coefficient c',xlim=(-.105,.17))
ax.invert_yaxis();ax.grid(axis='x',color='.93');ax.set_title('95% whole-family bootstrap intervals',loc='left',fontsize=10)
ax.plot([],[],c=BLUE,marker='o',ls='',label='Measured ML');ax.plot([],[],c=ORANGE,marker='o',ls='',label='KMA ML')
ax.legend(frameon=False,loc='lower right',fontsize=8)
ax=axes[0,1]
cv_specs=[('OLS, event weights','OLS','event'),('OLS, family weights','OLS','family'),('ODR, family weights','ODR','family'),('50 bins','BinnedOLS','event')]
for j,(label,method,weight) in enumerate(cv_specs):
    for k,(xkey,color) in enumerate([('ML_Sheen',BLUE),('ML_KMA',ORANGE)]):
        rr=cv[(cv.selection=='Paired A')&cv.x.eq(xkey)&cv.y.eq('Mp_Liu_Kim1983')&cv.method.eq(method)&cv.weighting.eq(weight)&cv.n_bins.eq(50)].iloc[0]
        ax.errorbar(rr.gain_RMS_families,j+(-.11 if k==0 else .11),xerr=[[max(0,rr.gain_RMS_families-rr.gain_lo)],[max(0,rr.gain_hi-rr.gain_RMS_families)]],fmt='o',c=color,ms=4,capsize=2)
ax.axvline(0,c='.4',lw=.9);ax.set(yticks=range(len(cv_specs)),yticklabels=[s[0] for s in cv_specs],xlabel='Family RMSE gain: linear − quadratic')
ax.invert_yaxis();ax.grid(axis='x',color='.93');ax.set_title('Whole event families held out',loc='left',fontsize=10)
ax=axes[1,0]
edges=np.arange(1.75,5.76,.25);counts,_=np.histogram(g.ML_Sheen,edges)
groupcount=np.array([g.loc[g.ML_Sheen.ge(l)&g.ML_Sheen.lt(u),'family_block'].nunique() for l,u in zip(edges[:-1],edges[1:])])
center=(edges[:-1]+edges[1:])/2
ax.bar(center,counts,width=.23,color='.75',label='Events')
ax.step(center,groupcount,c=BLUE,where='mid',lw=1.4,label='Families per bin')
ax.set(xlabel='Measured ML',ylabel='Count',yscale='log',ylim=(.8,160),xlim=(1.75,5.6));ax.grid(axis='y',color='.93')
ax.legend(frameon=False);ax.set_title('Limited independent coverage at high ML',loc='left',fontsize=10)
ax=axes[1,1]
stages=['Hong full','Hong exact matches','Hong matched Mw A','Hong matched paired A']
for y,label,color in [('Mp_Hong_spectral','Hong spectral',BLUE),('Mp_Hong_TD','Hong time domain',ORANGE)]:
    for method,ls,marker in [('OLS','-','o'),('BinnedOLS','--','s')]:
        vals=[row(x='ML_KMA',y=y,method=method,weight='event',selection=s).quadratic_c for s in stages]
        ax.plot(range(4),vals,c=color,ls=ls,marker=marker,ms=4,lw=1.1,label=label+(' / bins' if method=='BinnedOLS' else ' / OLS'))
        control=row(x='ML_KMA',y=y,method=method,weight='event',selection='Hong same ML range')
        ax.scatter(4,control.quadratic_c,c=color,marker=marker,s=25)
ax.axhline(0,c='.5',lw=.8);ax.axhline(.04,c=RED,ls=':',lw=.9)
ax.set(xticks=range(5),xticklabels=['All\n490','Depth\n294','Mw A\n199','Paired A\n165','ML 2–3.9\n483'],ylabel='Quadratic coefficient c',ylim=(-.12,.18))
ax.grid(axis='y',color='.94');ax.legend(frameon=False,fontsize=6.9,loc='upper right')
ax.set_title('Published Hong values; selection changes',loc='left',fontsize=10)
save(fig,'18_potency_curvature_validation',
 'Red dotted: Liu c=0.040. Positive RMSE gain favors the quadratic; error bars resample paired held-out family losses.\n'
 'Binned quadratic gains are relative to binned linear fits. Across the full sample, the ordinary family-weighted line predicts at least as well.\n'
 'Hong panel: final group independently restricts the full catalog ML range; it is not the next nested selection.')

fig,(ax,bx)=plt.subplots(1,2,figsize=(11,7.6))
fig.subplots_adjust(left=.1,right=.96,top=.85,bottom=.22,wspace=.32)
fig.suptitle('Velocity sensitivity and local-magnitude definitions',fontsize=15,y=.95)
dm=a.Mw_source_Vs_Kim2011-a.Mw;dp=a.Mp_Liu_Kim2011-a.Mp_Liu_Kim1983
ax.scatter(a.z_central,dm,c=BLUE,s=17,alpha=.6,edgecolors='none',label='ΔMw')
ax.scatter(a.z_central,dp,c=ORANGE,s=17,alpha=.6,edgecolors='none',label='ΔMP')
ax.axhline(0,c='.5',lw=.8);ax.set(xlabel='Source depth (km)',ylabel='Kim 2011 − Kim & Kim 1983',xlim=(0,30))
ax.grid(color='.93');ax.legend(frameon=False);ax.set_title('Source-coefficient sensitivity only',loc='left')
distance=np.linspace(10,100,300)
for depth,color in [(5,BLUE),(15,ORANGE),(20,'#34855b')]:
    rhyp=np.hypot(distance,depth)
    delta=1.110*np.log10(rhyp/100)+.00189*(rhyp-100)-(.5869*np.log10(distance/100)+.001680*(distance-100))
    bx.plot(distance,delta,c=color,lw=1.5,label=f'Depth {depth} km')
bx.axhline(0,c='.5',lw=.8);bx.set(xlabel='Epicentral distance (km)',ylabel='Hutton–Boore ML − Korean ML')
bx.grid(color='.93');bx.legend(frameon=False);bx.set_title('Same horizontal WA amplitude',loc='left')
save(fig,'19_potency_velocity_and_ML_definition',
 'Left: ΔMP=ΔMw/3 follows algebraically from P0=M0/(ρVs²) and M0∝ρVs³ in this analytical S-wave method.\n'
 'It is not an independent validation of full-waveform Mpv. Right: attenuation/distance definitions differ even at identical amplitude.\n'
 'The Korean relation uses epicentral distance; Liu supplement S1 uses Hutton–Boore with hypocentral distance.')
(R/'tables/potency_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def para(t):return '<p>'+t+'</p>'
def table(head,rows):
    return '<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def ci(r):return f'{r.c_lo:+.3f} to {r.c_hi:+.3f}'
primary=row();kma=row(x='ML_KMA');quad_odr=row(method='ODR');quad_bin=row(method='BinnedOLS',weight='event')
linear=row(degree=1);bin_linear=row(method='BinnedOLS',weight='event',degree=1)
hb=row(x='ML_HB87_same_WA2080',selection='ML attenuation control')
model_alt=row(y='Mp_Liu_Kim2011');fixed=row(y='Mp_Liu_fixed_Vs3500')
examples=summary['examples']
example_rows=[[('Haenam 2020' if x['event_id'].startswith('2020') else 'Sangju 2026'),f"{x['P0_m3_Kim1983']:.0f}",f"{x['Mw']:.3f}",f"{x['Mp_Liu_Kim1983']:.3f}",f"{x['Mpo28_C9p10_Kim1983']:.3f}"] for x in examples]
coeff_rows=[]
for label,r in [('Measured ML, family OLS',primary),('Measured ML, family ODR',quad_odr),('Measured ML, 50-bin OLS',quad_bin),('KMA ML, family OLS',kma)]:
    coeff_rows.append([label,f'{r.a_at_ML3:.4f}',f'{r.slope_at_ML3:.4f}',f'{r.quadratic_c:.4f}',ci(r)])
gain=cv[(cv.selection=='Paired A')&cv.x.eq('ML_Sheen')&cv.y.eq('Mp_Liu_Kim1983')&cv.method.eq('OLS')&cv.weighting.eq('family')].iloc[0]
residuals={r['sample']:r for r in summary['published_curve_residuals']}
papers=[
 ('Trugman and Ben-Zion (2026), SRL','10.1785/0220250419'),
 ('Liu et al. (2026), GRL and supplied supplement','10.1029/2026GL124983')]
pages=[]
pages.append('<h1>Potency magnitude in Korea</h1>'+para('19 November 2026 · A separate analysis of the current source spectra and magnitude catalog. Existing Mw/ML values, selections and six reports are preserved. The current reference is the Haenam Kim & Kim (1983) implementation, Vp/Vs=1.73; published Kim et al. (2011) is retained as the model comparison.')+
 '<h2>Applicable now: ordinary scalar potency</h2>'+para('For a shear source, P0=area×average slip=M0/μ, in m³, with μ=ρVs². Our S-wave amplitude conversion gives M0∝ρVs³Ωr, so P0=4πVsΩr/(2×0.63). Density cancels in this analytical coefficient; Vs dependence remains linear. The fitted plateau, corner frequency and spectral shape are preserved.')+
 para(f'Across {len(a)} quality-A events, the median absolute two-model difference decreases from {summary["source_velocity_sensitivity"]["Mw"]["median_abs"]:.4f} Mw to {summary["source_velocity_sensitivity"]["Mp"]["median_abs"]:.4f} MP. The exact factor of three is an algebraic property of this source-coefficient experiment, not independent evidence that the full velocity-model uncertainty has been removed. Response, attenuation, site and source-spectrum uncertainties remain. Conditional station-bootstrap widths are unchanged at a fixed model.')+
 '<h2>Two papers, different numerical normalizations</h2>'+table(['Quantity','Definition'],[
 ['Physical potency','P0=M0/(ρVs²), in m³'],
 ['Liu MP, equation 2','MP=(2/3)[log10(P0[m³])+1.4563]'],
 ['Trugman source-only Mpo analogue','Mpo=[log10(28 GPa × P0[m³])−9.10]/1.5'],
 ['Explicit 9.05 alternatives','28 GPa and 36 GPa reference versions retained separately']])+
 para(f'Liu uses P0 in cm km² with constant 5.4563; 1 cm km²=10⁴ m³, so the SI constant is 1.4563. That scale corresponds to approximately 36 GPa with C=9.10. Trugman (2026) uses 28 GPa with C=9.10; its source-only magnitude is {summary["Liu_vs_Mpo28_offset"]:.4f} lower for identical potency. Our adopted Mw remains at C=9.05. A C change does not change physical potency; any curve comparison must change its magnitude zero point consistently.')+
 table(['Event','P0 (m³)','Current Mw','Liu MP','28 GPa Mpo'],example_rows)+
 '<h2>Mpv requires a separate full-waveform test</h2>'+para('Trugman’s Mpv uses effective rigidity μw=√(μsrc μchar), where μchar is travel-time averaged to a quarter-wavelength depth at the dominant inversion period. It was tested with complete regional moment-tensor inversions, including model-dependent Green functions and depths. We have not applied this extra correction to our analytical S-wave spectra or claimed a Korean Mpv validation. Matched MTUQ inversions under both exact velocity/density models are the appropriate next test.'))

pages.append('<h1>Quadratic scaling: what the Korean data support</h1>'+
 para('Liu et al. divide −1.8≤ML≤7.5 into 50 equal bins, remove events more than 0.5 MP from the initial bin median, recompute median ML/MP, and fit unweighted quadratic least squares to those medians. Their coefficient intervals use the fit covariance and Student-t distribution. This procedure is not ODR. We implement it, then add event/family-weighted OLS, equal-axis ODR and whole-family bootstrap and prediction tests.')+
 para(f'Our main comparison contains {len(g)} paired-A events in 194 event families, measured ML {g.ML_Sheen.min():.2f}–{g.ML_Sheen.max():.2f}. Most events are small; the large-event tail has few independent sequences. The source-coefficient experiment uses fixed national depths. It does not recreate Liu’s P-wave windows, MTSPEC implementation, Bayesian source fitting, or attenuation/site corrections.')+
 '<h2>Fits on identical events</h2>'+para('Write MP=a+b(ML−3)+c(ML−3)². MP uses the Liu normalization and our Kim & Kim source model. Intervals below resample complete event families, conditional on model, depth and accepted spectra. Equal-axis ODR is a stated metric; measurement-error covariance is not inferred.')+
 table(['Fit','a','b','c','95% interval, c'],coeff_rows)+
 para(f'The 50-bin measured-ML fit has c={quad_bin.quadratic_c:.4f}. The paper-style covariance interval is {quad_bin.student_t_curvature_lo:.4f}–{quad_bin.student_t_curvature_hi:.4f}, while whole-family resampling gives {ci(quad_bin)}. There are {int(quad_bin.n_bins_fitted)} occupied bins, including {int(quad_bin.n_singleton_bins)} singletons. Requiring three events per bin gives c={row(method="BinnedOLS",weight="event",minimum=3).quadratic_c:.4f}. Apparent significance is sensitive to sampling and uncertainty assumptions.')+
 '<h2>Prediction: hold out whole earthquake sequences</h2>'+para(f'For family-weighted OLS and measured ML, held-out family RMSE is {gain.linear_RMS_families:.5f} for a line and {gain.quadratic_RMS_families:.5f} for a quadratic. The gain is {gain.gain_RMS_families:+.5f}, with 95% paired-family interval {gain.gain_lo:+.5f} to {gain.gain_hi:+.5f}. This full-sample metric does not establish a predictive improvement; the large-event subset is examined separately on pages 7–11. Binned quadratic fits improve on their own binned-linear baseline, but have RMSE {quad_bin.CV_RMS_families:.5f}, worse than the ordinary family-weighted line.')+
 para(f'Changing Vs model changes the family-weighted curvature only slightly: c={primary.quadratic_c:.4f} for Kim & Kim, {model_alt.quadratic_c:.4f} for Kim2011, and {fixed.quadratic_c:.4f} with a fixed-Vs potency coefficient. Removing Gyeongju/Pohang or layer-boundary-crossing events leaves broad intervals. Velocity-layer normalization is not the main explanation of the small fitted curvature in this sample.'))

pages.append('<h1>Regional transfer and the Haenam study</h1>'+
 '<h2>The published coefficients are not a Korean calibration</h2>'+para('Liu’s relation is MP=0.040 ML²+0.556 ML+1.133. With its exact magnitude normalization, our median MP minus that curve is '+f'{residuals["Our paired, measured ML"]["median"]:+.3f} for measured Korean ML and {residuals["Our paired, KMA ML"]["median"]:+.3f} for KMA ML. Hong’s full-catalog median differences are {residuals["Hong full spectral"]["median"]:+.3f} (spectral) and {residuals["Hong full TD"]["median"]:+.3f} (TD), with RMS {residuals["Hong full spectral"]["RMS"]:.3f} and {residuals["Hong full TD"]["RMS"]:.3f}. Agreement of a catalog median does not establish absolute calibration or transferability.')+
 para('The published Hong values themselves are held fixed in the selection test. Their fitted quadratic curvature changes with the depth-qualified, Mw-A and paired-A subsets, and also with the estimator/ML-range restriction. These are Korean events, but they do not establish a unique national quadratic law. The published multi-region relation should be plotted as an external reference, without forcing Korean magnitudes onto it.')+
 '<h2>ML definitions can change the apparent curvature</h2>'+para('The Korean ML attenuation is 0.5869 log10(Repi/100)+0.001680(Repi−100)+3; Liu supplement S1 uses 1.110 log10(Rhyp/100)+0.00189(Rhyp−100)+3. On exactly the same accepted horizontal WA amplitudes and station sets, replacing only that attenuation/distance convention shifts median event ML by '+f'{regional["definitions"][2]["median_ML_minus_ours"]:+.3f} and gives c={hb.quadratic_c:.4f} ({ci(hb)}). Its held-out family RMSE changes from {regional["heldout_gains"][0]["linear_RMS_families"]:.5f} to {regional["heldout_gains"][0]["quadratic_RMS_families"]:.5f}; gain uncertainty still spans zero. This is a controlled definition test, not a new adopted ML catalog or a complete reproduction of Liu’s component/WA processing.')+
 para('Our WA magnification is 2080. Liu’s synthetic supplement explicitly uses 2800; its observational subsection does not fully specify gain/component aggregation. The separate 2800/2080 control shifts ML by 0.1291 but leaves free-polynomial curvature and prediction unchanged. It is not evidence for correcting either catalog by that gain.')+
 '<h2>What to incorporate in Haenam later</h2>'+para('Retain P0 in m³ alongside M0 and Mw, with the exact source model, depth, density, rigidity and magnitude convention. Use the same μ for converting spectral M0 to potency and for converting rupture moment to slip. Propagate depth envelopes across layer boundaries; compare the two velocity models and preserve instrument/window/FFT controls. Add the P-wave Bayesian implementation as an independent analysis after its windows, response and attenuation are validated.')+
 para('Within a common Haenam source layer, converting Mw to MP adds a constant: event rankings and relative magnitude spacings do not change. Its main advantages are consistent kinematic quantities and comparisons across depth/model choices. It does not resolve missing source plateaus or difficult EGF/ASTF extraction.')+
 para('Potency is useful for kinematics, but fixed-radius stress drop remains Δσ=7μP0/(16a³); it still requires rigidity. Test Mpv using matched full-waveform inversions. Extend the Korean magnitude range with well-resolved small events and independently constrained larger events before adopting a national quadratic conversion. The present exploratory comparison supports retaining the current linear relation and reporting curvature as a sensitivity.')+
 '<h2>Sources and reproducibility</h2>'+para('Trugman and Ben-Zion (2026), SRL, doi:10.1785/0220250419; Liu et al. (2026), GRL, doi:10.1029/2026GL124983, with the supplied supplement. PDF hashes and page-referenced method notes are retained. The cited Zenodo record could not be retrieved with the available web tool; the described regression was independently implemented. Numerical tables, bootstrap coefficients, held-out predictions and all scripts accompany this report.'))

css='body{font-family:sans-serif;font-size:10pt;line-height:1.32;color:#20272c;}h1{font-size:18.5pt;margin:0 0 12pt;line-height:1.15;}h2{font-size:12.5pt;margin:12pt 0 6pt;}p{margin:0 0 8pt;}table{width:100%;border-collapse:collapse;font-size:8.9pt;margin:8pt 0 10pt;}td,th{padding:5pt 4pt;border-bottom:.5pt solid #ccd3d8;text-align:left;}th{background:#edf2f5;}'
doc=fitz.open();scales=[]
for i,html in enumerate(pages):
    page=doc.new_page(width=595.276,height=841.89)
    spare,scale=page.insert_htmlbox(fitz.Rect(42,30,553,807),html,css=css,scale_low=.95)
    if spare<0:raise RuntimeError(f'Potency narrative {i+1} overflows')
    scales.append(scale);print('Potency narrative',i+1,'scale',scale,flush=True)
for item in manifest:
    with fitz.open(R/'figures'/(item['stem']+'.pdf')) as part:doc.insert_pdf(part)
from report_magnitude_followup import append_followup
followup=append_followup(doc)
manifest+=followup['figures']
(R/'tables/potency_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
path=R/'reports/Potency_magnitude_and_quadratic_scaling.pdf'
doc.save(path,garbage=4,deflate=True)
with fitz.open(path) as pdf:
    path.with_suffix('.txt').write_text('\n\n'.join(p.get_text() for p in pdf))
    for i in range(len(pdf)):pdf[i].get_pixmap(matrix=fitz.Matrix(1.15,1.15)).save('/tmp/potency_report_'+str(i)+'.png')
metadata=dict(report_pages=len(doc),narrative_scales=scales+followup['narrative_scales'],figures=manifest,followup=followup,
  adopted_outputs_changed=False,headline_curvature=float(primary.quadratic_c),headline_curvature_CI=[float(primary.c_lo),float(primary.c_hi)],
  linear_family_CV_RMS=float(gain.linear_RMS_families),quadratic_family_CV_RMS=float(gain.quadratic_RMS_families))
(R/'tables/potency_report_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
