"""Publication figures for the supplied BSSA method audit and ODR comparisons."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fitz
from source_velocity import validate_catalog,CONFIG_SHA256
R=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
BLUE='#176da0';ORANGE='#bd591b';GREEN='#34855b'
fits=pd.read_csv(R/'tables/Hong_regression_comparison.csv')
boots=json.loads((R/'tables/Hong_ODR_bootstrap.json').read_text())
m=pd.read_csv(R/'tables/KR_Mw_matched_events.csv',dtype={'event_id':str})
k=pd.read_csv(R/'inputs/KR_Mw/Events_KP_2017_2022_Mw.csv',dtype={'evid':str});k['event_id']=k.evid.str.strip()
g=m[(m.Mw_quality=='A')&m.primary_geography];h=g[g.ML_comparison_primary]
validate_catalog(m)
stability=pd.read_csv(R/'tables/estimator_stability_events.csv',dtype={'event_id':str})
stable=stability[(stability.status=='fit')&stability.both_methods_plateau.eq(True)&~stability.held_out]
paper_audit=json.loads((R/'tables/Hong_paper_method_audit.json').read_text())
validation=json.loads((R/'tables/Hong_regression_validation.json').read_text());assert validation['passed']
manifest=[]

def row(selection,x,y,weight='event',estimator='ODR'):
    q=fits[(fits.selection==selection)&(fits.x==x)&(fits.y==y)&(fits.weighting==weight)&(fits.method==estimator)]
    assert len(q)==1,(selection,x,y,weight,estimator)
    return q.iloc[0]

def canvas(title,rows=1,cols=2,size=(11,7.6)):
    fig,ax=plt.subplots(rows,cols,figsize=size)
    fig.subplots_adjust(left=.09,right=.96,top=.86,bottom=.19,hspace=.37,wspace=.3)
    fig.suptitle(title,fontsize=15,y=.955)
    return fig,ax

def finish(fig,stem,caption):
    fig.text(.075,.04,caption,fontsize=8.5,color='.25',linespacing=1.4)
    for ext in ['pdf','png']:fig.savefig(R/'figures'/(stem+'.'+ext))
    plt.close(fig);manifest.append(dict(stem=stem,caption=caption))

fig,axes=canvas('Moment magnitudes: orthogonal regression')
for ax,reference,title in zip(axes,['mw_spectra','mw_TD'],['KR spectral Mw','KR time-domain Mw']):
    rr=row('Matched Mw A',reference,'Mw');rf=row('Matched Mw A',reference,'Mw','family')
    r2=row('Matched Mw A',reference,'Mw_source_Vs_Kim2011')
    x=np.linspace(g[reference].min()-.1,g[reference].max()+.1,150)
    ax.scatter(g[reference],g.Mw,s=24,c=BLUE,alpha=.75,edgecolors='none')
    ax.plot(x,x,c='.5',ls=':',lw=1,label='1:1')
    ax.plot(x,rr.intercept+rr.slope*x,c=BLUE,lw=1.4,label='Kim & Kim 1983 ODR')
    ax.plot(x,r2.intercept+r2.slope*x,c=ORANGE,lw=1.3,label='Kim 2011 ODR')
    ax.plot(x,rf.intercept+rf.slope*x,c='.4',ls='--',lw=1.1,label='1983, equal sequence weights')
    b=next(b for b in boots if b['selection']=='Matched Mw A' and b['x']==reference and b['y']=='Mw' and b['weighting']=='event')
    curves=np.asarray(b['intercept'])[:,None]+np.asarray(b['slope'])[:,None]*x
    lo,hi=np.quantile(curves,[.025,.975],axis=0);ax.fill_between(x,lo,hi,color=BLUE,alpha=.12,lw=0)
    ax.set(xlabel=title,ylabel='Our multitaper Mw');ax.grid(color='.92');ax.legend(frameon=False,loc='lower right')
    ax.set_title(f'y = {rr.intercept:.3f} + {rr.slope:.3f} x\nSlope interval {rr.slope_lo:.3f}–{rr.slope_hi:.3f}',loc='left')
finish(fig,'12_ODR_Mw_Mw',
 '199 identical quality-A events; 119 sequence blocks. ODR assumes equal error scales in the two magnitude axes.\n'
 'Source Vs(z): Haenam Kim & Kim (1983), Vp/Vs=1.73; published Kim et al. (2011) control. Shading: 1983 mean-line interval.')

fig,axes=canvas('ML–Mw relations on identical events')
for ax,xkey,label in zip(axes,['ML_KMA','ML_Sheen'],['KMA ML','Measured ML']):
    for ykey,name,color in [('Mw','Our MT, 1983',BLUE),('Mw_source_Vs_Kim2011','Our MT, 2011','#8c5a9f'),('mw_spectra','KR spectral',ORANGE),('mw_TD','KR time domain',GREEN)]:
        rr=row('Matched Mw A / ML A',xkey,ykey)
        ax.scatter(h[xkey],h[ykey],s=15,c=color,alpha=.35,edgecolors='none')
        x=np.linspace(h[xkey].min(),h[xkey].max(),100)
        ax.plot(x,rr.intercept+rr.slope*x,c=color,lw=1.5,
          label=f'{name}: {rr.intercept:.3f} + {rr.slope:.3f} ML')
    ax.plot([1.8,4.5],[1.8,4.5],c='.6',ls=':',lw=.8)
    ax.set(xlabel=label,ylabel='Mw',xlim=(1.8,4.3),ylim=(1.95,4.05));ax.grid(color='.92')
    ax.legend(frameon=False,loc='upper left',fontsize=8)
finish(fig,'13_ODR_ML_Mw',
 '165 identical events with our Mw-A and ML-A assessments. All lines use equal-axis, equal-event ODR.\n'
 'Our two Vs(z) models use common density and event depths. The right panel uses our measured ML for every Mw estimate.')

fig,axes=canvas('Selection changes the ML–Mw relation',2,2,size=(11,9.5))
stages=[('KR full 490',k,'Published catalog'),('All exact matches',m,'Depth-qualified matches'),
 ('Matched Mw A',g,'Our Mw-A selection'),('Matched Mw A / ML A',h,'Our Mw-A / ML-A selection')]
for ax,(label,q,title) in zip(axes.flat,stages):
    rr=row(label,'ML_KMA','mw_spectra')
    ax.scatter(k.ml_KMA,k.mw_spectra,s=13,c='.8',alpha=.6,edgecolors='none')
    ax.scatter(q.ml_KMA,q.mw_spectra,s=18,c=BLUE,alpha=.75,edgecolors='none')
    x=np.linspace(q.ml_KMA.min(),q.ml_KMA.max(),100);ax.plot(x,rr.intercept+rr.slope*x,c='black',lw=1.2)
    ax.plot([1.8,5.7],[1.8,5.7],c='.55',ls=':',lw=.8)
    ax.set(xlabel='KMA ML',ylabel='KR spectral Mw',xlim=(1.9,5.6),ylim=(1.9,5.8));ax.grid(color='.93')
    ax.set_title(f'{title}: n = {len(q)}; slope {rr.slope:.3f}',loc='left')
finish(fig,'14_ML_Mw_selection',
 'The KR magnitudes are unchanged across panels. Black: equal-event ODR; gray dots: the full KR catalog.\n'
 'Restricting only the full catalog ML range to 2.0–3.9 gives slope 0.963. The change above is not explained by that range cut alone.')

fig,(ax,bx)=canvas('Estimator stability under controlled changes')
for method,name,color in [('MT','Multitaper',BLUE),('FFT','Fourier',ORANGE)]:
    v=np.sort(abs(stable[method+'_window_change'].to_numpy()));p=100*np.arange(1,len(v)+1)/len(v)
    ax.step(v,p,where='post',c=color,lw=1.5,label=f'{name}; median {np.median(v):.4f}')
ax.set(xlabel='Absolute Mw change: 10.1 to 20.1 s',ylabel='Cumulative events (%)',xlim=(0,.12),ylim=(0,101))
ax.grid(color='.92');ax.legend(frameon=False,loc='lower right')
x=stable.FFT_station_bootstrap_sigma.to_numpy();y=stable.MT_station_bootstrap_sigma.to_numpy()
bx.scatter(x,y,s=18,c=BLUE,alpha=.55,edgecolors='none');limit=max(x.max(),y.max())*1.06
bx.plot([0,limit],[0,limit],c='.5',ls=':',lw=1);bx.set(xlabel='Fourier station-resampling σMw',ylabel='Multitaper station-resampling σMw',xlim=(0,limit),ylim=(0,limit));bx.grid(color='.92')
bx.set_title(f'Median σMw: MT {np.median(y):.4f}; FFT {np.median(x):.4f}',loc='left')
finish(fig,'15_estimator_stability',
 '568 events before the Mw-A agreement/window gates; both methods pass the fitted-plateau criterion on common station/frequency support.\n'
 '250 station resamples, fixed source shape. Existing SNR masks come from MT. These measure stability, not absolute bias; TD was not rerun.')
(R/'tables/Hong_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def equation(r,x='x',y='y'):
    return f'{y} = {r.intercept:.3f} + {r.slope:.3f} {x}'

def paragraph(text):return '<p>'+text+'</p>'
def table(head,rows):
    return '<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

mwrows=[]
for ref,label in [('mw_spectra','KR spectral'),('mw_TD','KR time domain')]:
    rr=row('Matched Mw A',ref,'Mw')
    mwrows.append([label,f'{rr.intercept:.3f}',f'{rr.slope:.3f}',f'{rr.slope_lo:.3f}–{rr.slope_hi:.3f}',f'{rr.RMS_orthogonal:.3f}'])
mlrows=[]
for target,label in [('Mw','Our MT'),('mw_spectra','KR spectral'),('mw_TD','KR time domain')]:
    rr=row('Matched Mw A / ML A','ML_KMA',target)
    mlrows.append([label,f'{rr.intercept:.3f}',f'{rr.slope:.3f}',f'{rr.slope_lo:.3f}–{rr.slope_hi:.3f}'])
old=row('Our full paired A','ML_Sheen','Mw','family','OLS')
new=row('Our full paired A','ML_Sheen','Mw','family','ODR')
paperfit=row('KR full 490','mw_spectra','mw_TD')
ss=pd.read_csv(R/'tables/estimator_stability_summary.csv').query("selection=='Common support, both plateaus'").set_index('method')
pages=[]
pages.append('<h1>Hong and Sheen (2026):<br>verified method differences</h1>'+paragraph('19 November 2026 · Source-depth Vs is now applied throughout our Mw results. Measured spectra, ML, event selection and the published KR magnitudes are preserved.')+
 '<h2>Common magnitude convention, different physical assumptions</h2>'+
 paragraph('Their equation (5) is equivalent to our <b>C=9.05</b> convention. Hong uses density 2700 kg/m³ and fixed source Vs=3.61 km/s at 10 km, citing Kim et al. (2011). We use density 2800 kg/m³ and event-depth Vs(z): the Haenam Kim & Kim (1983) implementation with Vp/Vs=1.73 as reference, and published Kim et al. (2011), Table 2, alongside it. Radiation 0.63, free surface 2 and Q(f)=229.2 f⁰·⁷³ are common (pp.4–5).')+
 paragraph('The density/Vs coefficient difference is now event dependent. The old +0.01635 Mw value referred only to the superseded fixed-3.50 km/s reference. Their spreading changes from 1/r to 1/√(70r) beyond 70 km; at 100 km that change alone lowers Mw by 0.0516 relative to our 1/r. These are source/path sensitivity controls, not a complete recipe reproduction.')+
 '<h2>Spectral reference</h2>'+
 paragraph('The paper uses Prieto multitaper on three-component displacement spectra, 1024-sample/10.24 s windows starting S−0.5 s, equal noise ending P−0.5 s, 50 log-spaced frequencies over 0.1–30 Hz, an SNR&gt;3 threshold and a 30% station-coverage starting-frequency rule. Brune Ω0 and fc are estimated with PyMC MCMC. They assume 10 km depth and select records within 100 km hypocentral distance; velocimeters have priority at co-located sites (pp.3–5,8).')+
 paragraph('Our reference uses NW=4/K=7, 10.1/20.1 s windows starting S−0.1 s, noise ending P−1 s, 0.5–20 Hz, event-specific depths, log-amplitude stacking and a generalized Boatwright model with a Brune sensitivity. The paper illustrates a median spectrum. Its exact DPSS settings, PSD-to-FAS conversion, response prefilter/water level and full MCMC likelihood are not specified.')+
 '<h2>Time-domain method</h2>'+
 paragraph('Ten vertical-component narrow-band peak amplitudes are measured using three-pole, one-way Butterworth filters of 0.2 Hz bandwidth. Equivalent Fourier amplitude is <b>(vertical peak / bandwidth) × 2.25 / 1.62</b>. The constants separately match narrow-band peaks to spectral amplitudes and compensate S-wave partitioning. At least four bands must exceed SNR 3; nested coarse/fine 50×50 Brune grids give station estimates, combined by a network median. Thirty-second windows establish the matching constants; final peak measurements use 10 s from S−0.5 s. Sensor corrections further reduce station scatter (pp.5–9).')+
 paragraph('The time-domain estimates therefore provide a useful operational cross-check, while their amplitude constants are calibrated to the multitaper reference. They are not wholly independent evidence about absolute spectral normalization. No time-domain waveforms were reprocessed in this comparison.'))

pages.append('<h1>ODR results and interpretation</h1>'+
 paragraph('The previous headline ML–Mw fits used <b>family-balanced OLS of Mw on ML</b>; the earlier KR-comparison slopes used ordinary vertical OLS. The earlier Deming sensitivities already included the equal-axis ODR special case at λ=1. The headline OLS coefficients, however, should not be described as ODR results. The new tables provide explicit ODR comparisons alongside OLS.')+
 paragraph(f'The fits use linear <b>ODR with equal error scales in both magnitude axes</b>, equivalent to Deming λ=σ²y/σ²x=1. Sensitivities use λ=0.25 and 4 and equal sequence weights. Matched/current samples use 2500 family-block resamples; full KR intervals resample events because full sequence assignments are unavailable. All {validation["scipy_ODRPACK_checks"]} equal-axis fits were independently checked with numerical ODRPACK.')+
 '<h2>Our Mw versus KR_Mw: same 199 Mw-A events</h2>'+
 paragraph('Here x is the indicated KR magnitude and y is our Kim & Kim (1983) source-Vs multitaper Mw; y=a+b x. Fits weight events equally; intervals resample 119 sequence blocks. Both source models are compared in the next table page and figure 12.')+
 table(['x','a','b','95% interval, b','RMS⊥'],mwrows)+
 '<h2>ML–Mw: same 165 Mw-A / ML-A events</h2>'+
 paragraph('All rows below use the identical <b>KMA ML</b> values as x. The KR catalog does not provide a separately remeasured local magnitude. Figure 13 also uses our measured ML as the common x-axis for all three moment estimates.')+
 table(['y','a','b','95% interval, b'],mlrows)+
 paragraph(f'For the full 357-event paired sample, our family-balanced measured-ML relation changes from <b>{equation(old,"ML","Mw")}</b> under OLS to <b>{equation(new,"ML","Mw")}</b> under equal-axis ODR. Different objectives give different slopes; vertical and perpendicular RMS are therefore supplied separately.')+
 paragraph(f'For the 490 published events, the spectral-to-time-domain ODR is {equation(paperfit,"spectral Mw","TD Mw")}; r²={paperfit.correlation_squared:.4f}, consistent with the paper’s rounded 0.97. The CSV mean TD−spectral is +0.041±0.076; p.8 describes a −0.04 residual in that stated order. We retain the CSV labels and make the subtraction order explicit.')+
 paragraph('Equal-error ODR is a stated metric assumption, not a measured error model. The published spectral errors are conditional MCMC uncertainties, whereas the TD errors describe station deviations. We do not treat those heterogeneous quantities as interchangeable 1σ errors. Shared waveform/response errors and intrinsic scatter remain; ordinary ODR does not remove their cross-axis covariance or selection bias.'))

pages.append('<h1>Selection and estimator stability</h1>'+
 '<h2>The catalog selection changes the slope</h2>'+
 paragraph('For the unchanged KR spectral magnitudes versus KMA ML, equal-event ODR gives slopes 0.943 for all 490 events, 0.961 for the 294 depth-qualified exact matches, 0.764 after our Mw-A selection (199 events), and 0.762 after the paired ML-A selection (165 events). Restricting only the full catalog’s ML range to 2.0–3.9 gives slope 0.963. The large flattening is therefore associated with our Mw-quality selection, not explained by that ML range restriction alone.')+
 paragraph('SNR, observed-plateau, usable-band and estimator/window agreement gates can select on magnitude and spectral shape even though no gate explicitly requires agreement with ML. The selected-sample regressions are valid descriptions of their retained events; they are not population-wide conversion formulas. ODR alone cannot correct this selection. This distinction matters more here than the modest OLS-to-ODR slope change.')+
 '<h2>Multitaper is not universally the most stable</h2>'+
 paragraph('We refitted both estimators on the same station/sensor identities and the intersection of their station-frequency support in 10.1 and 20.1 s windows. The test starts before the Mw-A agreement/window gates. Of 575 events with finite common fits, 568 pass the fitted-plateau criterion under both estimators in both windows. No choice of estimator was made from agreement with Hong.')+
 table(['Metric, 568 events','Multitaper','Direct Fourier'],[
  ['Median |ΔMw|, window',f'{ss.loc["MT","median_abs_window_change"]:.4f}',f'{ss.loc["FFT","median_abs_window_change"]:.4f}'],
  ['90th percentile |ΔMw|',f'{ss.loc["MT","q90_abs_window_change"]:.4f}',f'{ss.loc["FFT","q90_abs_window_change"]:.4f}'],
  ['Median station-resampling σMw',f'{ss.loc["MT","median_station_bootstrap_sigma"]:.4f}',f'{ss.loc["FFT","median_station_bootstrap_sigma"]:.4f}']])+
 paragraph('Multitaper reduces periodogram roughness and leakage through power averaging, at the cost of spectral resolution and possible smoothing bias. With NW=4, its nominal half-bandwidth changes from about 0.396 Hz to 0.199 Hz when this window doubles. A smoother spectrum does not guarantee the most stable or least biased Ω0, Mw or corner frequency.')+
 paragraph('In this controlled test, Fourier is slightly more stable to window changes and has slightly smaller conditional station-resampling scatter. Yet the previous cut-edge example shows that Fourier can also give a stable but contaminated spectrum. These are stability metrics, not absolute-accuracy tests. Existing SNR masks come from multitaper, and the resampling calculation fixes source shape; the time-domain method was not rerun on the same raw records.')+
 paragraph('Retain multitaper as the reference with Fourier, window, bandwidth and physical-calibration checks. Hong’s TD method adds an estimator, while this work does not establish a universal ranking. Source-depth Vs changes absolute moments and Mw; the within-event estimator/window differences remain unchanged because the same coefficient is applied to each.'))

model_rows=[]
for y,label in [('Mw','Kim & Kim 1983, 1.73'),('Mw_source_Vs_Kim2011','Kim et al. 2011')]:
    for ref,name in [('mw_spectra','Spectral'),('mw_TD','Time domain')]:
        rr=row('Matched Mw A',ref,y)
        diff=g[y]-g[ref]
        model_rows.append([label,name,f'{diff.median():+.4f}',f'{rr.intercept:.4f}',f'{rr.slope:.4f}',f'{rr.RMS_orthogonal:.4f}'])
material_rows=[]
for model,y,label in [('Kim1983','Mw','Kim & Kim 1983'),('Kim2011','Mw_source_Vs_Kim2011','Kim et al. 2011')]:
    for ref,name in [('mw_spectra','Spectral'),('mw_TD','Time domain')]:
        column='KR_'+ref+'_material_'+model
        diff=g[y]-g[column]
        material_rows.append([label,name,f'{diff.median():+.4f}',f'{np.sqrt(np.mean(diff**2)):.4f}'])
haenam=m[m.event_id.eq('20200503130714')].iloc[0]
pages.append('<h1>Both velocity models versus Hong</h1>'+
 paragraph('These comparisons use the same 199 quality-A event matches, their frozen national depths and the same measured source spectra. Only our source-depth Vs coefficient changes. Density stays 2800 kg/m³. The published Hong values use fixed Vs=3.61 km/s and density 2700 kg/m³.')+
 '<h2>Published Hong values as reported</h2>'+table(['Our source model','Hong x','Median y−x','ODR a','ODR b','RMS⊥'],model_rows)+
 paragraph('ODR y=a+b x, equal event weights and error scales. The reference uses the Haenam Kim & Kim implementation, Vp/Vs=1.73. The alternative stored Kim1983 file with Vs=3.79 km/s in the lower crust is a different implementation and is not used here.')+
 '<h2>Density/Vs coefficient control</h2>'+paragraph('For this separate control, the published Hong magnitude is shifted by (2/3)log₁₀(2800/2700)+2log₁₀[Vs(z)/3610]. This places the source coefficient on the same convention only; Hong’s fixed-depth geometry, station corrections, aggregation and spectra are not recomputed. It is not an empirical correction fitted to the residuals.')+
 table(['Our model','Hong estimator','Median difference','Raw RMS'],material_rows)+
 '<h2>Haenam, 3 May 2020</h2>'+paragraph(f'At the national depth 21.26 km, our MT Mw is {haenam.Mw:.3f} for the Haenam Kim & Kim implementation (Vs=3.6879 km/s), and {haenam.Mw_source_Vs_Kim2011:.3f} for Kim2011 (Vs=3.70 km/s). Hong reports spectral Mw {haenam.mw_spectra:.2f} and TD Mw {haenam.mw_TD:.2f}. The dedicated Haenam relocation is 19.867 km, above Kim2011’s 20.7-km boundary; its source Vs would be 3.60 km/s. That dedicated depth is not substituted into the national cohort.')+
 paragraph('Depth-envelope endpoint magnitudes now include distance and Vs(z) layer changes. Station bootstrap intervals still condition on the selected central depth and velocity model. The separate source-velocity report lists the model layers, stress-drop scaling and full paired-sample ODR comparison.'))

css='''body{font-family:sans-serif;font-size:10pt;line-height:1.35;color:#20272c;}h1{font-size:19pt;margin:0 0 12pt;line-height:1.15;}h2{font-size:12.5pt;margin:13pt 0 6pt;}p{margin:0 0 8pt;}table{width:100%;border-collapse:collapse;font-size:9.2pt;margin:8pt 0 11pt;}td,th{padding:5pt 4pt;border-bottom:.5pt solid #ccd3d8;text-align:left;}th{background:#edf2f5;}'''
doc=fitz.open()
for i,html in enumerate(pages):
    page=doc.new_page(width=595.276,height=841.89)
    spare,scale=page.insert_htmlbox(fitz.Rect(42,30,553,807),html,css=css,scale_low=.93)
    if spare<0:raise RuntimeError(f'Page {i+1} overflows')
    print('Narrative',i+1,'scale',scale,flush=True)
for figure in manifest:
    with fitz.open(R/'figures'/(figure['stem']+'.pdf')) as p:doc.insert_pdf(p)
with fitz.open(R/'figures/16_source_velocity_models.pdf') as p:doc.insert_pdf(p)
path=R/'reports/Hong_KR_Mw_regression_and_stability.pdf'
doc.save(path,garbage=4,deflate=True)
with fitz.open(path) as p:
    path.with_suffix('.txt').write_text('\n\n'.join(page.get_text() for page in p))
    for i in range(3):p[i].get_pixmap(matrix=fitz.Matrix(1.1,1.1)).save('/tmp/Hong_ODR_report_'+str(i)+'.png')
assert hashlib.sha256(Path(paper_audit['paper']).read_bytes()).hexdigest()==paper_audit['sha256']
validate_catalog(m)
result=dict(passed=True,paper_SHA256=paper_audit['sha256'],pages=len(doc),ODR_validation=validation,
 n_stability_common_plateau=len(stable),source_velocity_updated=True,source_velocity_config_sha256=CONFIG_SHA256,
 main_ODR='Equal error scales and equal event weights; family-block bootstrap; equal-family weights supplied separately')
(R/'tables/Hong_report_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('Report pages',len(doc))
