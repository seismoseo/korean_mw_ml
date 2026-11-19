"""Audit estimator choice and compare the pinned KR_Mw event catalog.

Comparison products are separate from adopted magnitudes. No empirical offset
is applied, and undocumented KR_Mw constants are not inferred from agreement.
"""
from pathlib import Path
import hashlib,json,sys,textwrap,warnings
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
 'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
BLUE='#176da0';ORANGE='#bd591b';RED='#a63027'
d=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str})
validate_catalog(d)
k=pd.read_csv(R/'inputs/KR_Mw/Events_KP_2017_2022_Mw.csv',dtype={'evid':str})
details=pd.read_csv(R/'inputs/KR_Mw/Details_KP_2017_2022_Mw.csv',dtype={'evid':str})
k['event_id']=k.evid.str.strip();details['event_id']=details.evid.str.strip()
assert k.event_id.is_unique and d.event_id.is_unique
matched=d.merge(k,on='event_id',how='inner',suffixes=('','_KR'),validate='one_to_one')
matched['origin_difference_s']=(pd.to_datetime(matched.origin_utc,utc=True)-pd.to_datetime(matched.origin_time,utc=True)).dt.total_seconds()
assert (matched.origin_difference_s==0).all() and np.allclose(matched.ml,matched.ml_KMA)
for col in ['mw_spectra','mw_TD']:
    for ours in ['Mw','Mw_Fourier','Mw_Brune','Mw_source_Vs_Kim1983','Mw_source_Vs_Kim2011']:
        matched[ours+'_minus_KR_'+col]=matched[ours]-matched[col]
    for model in ['Kim1983','Kim2011']:
        # Material-prefactor control only; the published catalog remains intact.
        shift=np.log10(2800/2700)/1.5+2*np.log10(matched['source_Vs_'+model]/3610)
        matched['KR_'+col+'_material_'+model]=matched[col]+shift
matched.to_csv(R/'tables/KR_Mw_matched_events.csv',index=False)
g=matched[matched.Mw_quality=='A'].copy()
stats=[];rng=np.random.default_rng(20261119)
for selection,frame in [('All finite',matched),('Mw A',g),('Paired Mw A / ML A',matched[matched.ML_comparison_primary])]:
    for reference in ['mw_spectra','mw_TD']:
        for ours in ['Mw','Mw_Fourier','Mw_Brune','Mw_source_Vs_Kim1983','Mw_source_Vs_Kim2011','Mw_reference_Vs3500']:
            q=frame.dropna(subset=[ours,reference]).copy();delta=q[ours]-q[reference]
            blocks=q.groupby('family_block',dropna=False).apply(lambda z:float((z[ours]-z[reference]).mean()),include_groups=False).to_numpy()
            draws=np.mean(rng.choice(blocks,(2000,len(blocks)),replace=True),axis=1)
            fit=np.polyfit(q[reference],q[ours],1)
            stats.append(dict(selection=selection,our_estimator=ours,KR_field=reference,regression_method='OLS y on x',n=len(q),
              median_difference=float(delta.median()),mean_difference=float(delta.mean()),
              robust_scatter=float(1.4826*np.median(abs(delta-delta.median()))),
              raw_RMS=float(np.sqrt(np.mean(delta**2))),centered_RMS=float(np.std(delta,ddof=0)),
              correlation=float(np.corrcoef(q[reference],q[ours])[0,1]),slope=float(fit[0]),intercept=float(fit[1]),
              n_blocks=len(blocks),family_mean_difference=float(blocks.mean()),
              family_mean_lo=float(np.quantile(draws,.025)),family_mean_hi=float(np.quantile(draws,.975))))
stats=pd.DataFrame(stats);stats.to_csv(R/'tables/KR_Mw_comparison_statistics.csv',index=False)
est=d[['event_id','Mw','Mw_Fourier','Mw_quality','Mw_reasons','ML_comparison_primary']].copy()
est['MT_minus_Fourier']=est.Mw-est.Mw_Fourier
est.to_csv(R/'tables/MT_Fourier_event_comparison.csv',index=False)
eststats={}
for quality,q in est.groupby('Mw_quality'):
    v=q.MT_minus_Fourier.dropna()
    eststats[quality]=dict(n_events=len(q),n_finite=len(v),median=float(v.median()) if len(v) else None,
      q16=float(v.quantile(.16)) if len(v) else None,q84=float(v.quantile(.84)) if len(v) else None,
      max_absolute=float(v.abs().max()) if len(v) else None,n_abs_gt_0p1=int((v.abs()>.1).sum()))
for f in json.loads((R/'inputs/KR_Mw/source.json').read_text())['files']:
    b=(R/'inputs/KR_Mw'/f['file']).read_bytes()
    assert hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()==f['git_blob_sha1']
# The station table and event table cannot be assumed to describe identical
# accepted station sets; retain the mismatch instead of silently repairing it.
n=details.groupby('event_id').size();med=details.groupby('event_id').mw_td.median()
alignment=k.set_index('event_id')[['nsta_spectra','nsta_TD','mw_spectra','mw_TD']].copy()
alignment['detail_rows']=n;alignment['detail_median_mw_td']=med
alignment['n_difference_spectra']=alignment.nsta_spectra-alignment.detail_rows
alignment['n_difference_TD']=alignment.nsta_TD-alignment.detail_rows
alignment.to_csv(R/'tables/KR_Mw_station_table_alignment.csv')
wave=pd.read_csv(R/'tables/estimator_waveform_controls.csv',dtype={'event_id':str})
q=wave[wave.event_id=='20160829153946']
summary=dict(adopted_estimator='Multitaper, NW=4, K=7; generalized Boatwright; 10.1 s; C=9.05',
 source_velocity_model='Kim & Kim (1983), Vp/Vs=1.73',source_velocity_comparison='Published Kim et al. (2011), Table 2',
 source_velocity_config_sha256=CONFIG_SHA256,
 Fourier_role='Sensitivity check; quality A requires |Mw_MT-Mw_FFT|<=0.10 by construction',
 KR_repository='https://github.com/dhsheen/KR_Mw',commit='61c24f655b83e320675cf64f01bc862792947d55',
 KR_events=len(k),KR_station_rows=len(details),exact_matches=len(matched),A_matches=len(g),
 paired_A_matches=int(matched.ML_comparison_primary.sum()),estimator_statistics=eststats,
 comparison=stats.query("selection=='Mw A' and our_estimator in ['Mw','Mw_Fourier']").to_dict('records'),
 station_table_alignment=dict(exact_spectra_counts=int((alignment.n_difference_spectra==0).sum()),
  exact_TD_counts=int((alignment.n_difference_TD==0).sum()),
  caution='Details table is not an exact station-list reconstruction of either event-summary column; do not infer event MAD or fit aggregation from these rows.'),
 edge_example=dict(event_id='20160829153946',quality='C',stations=len(q),
  median_MT_over_rectangular_FFT=float(q.MT_Fourier_ratio_2_10.median()),
  median_cosine_FFT_over_rectangular_FFT=float(q.cosine_Fourier_ratio_2_10.median()),
  band_Hz=[2,10]),
 limitations=['KR_Mw contains two CSV files and no estimation code or method README at this commit.',
  'Supplied Hong and Sheen (2026) paper establishes C=9.05, density 2700, Vs 3610, radiation .63, free surface 2, fixed depth 10 km and bilinear spreading; see Hong_paper_method_audit.json.',
  'Exact DPSS settings, PSD-to-FAS conversion, response prefilter/water level and event/detail version correspondence remain unspecified.',
  'The 2026 Sangju event is outside the 2017-2022 KR_Mw catalog.',
  'Magnitude agreement is not used to recalibrate either catalog.'],
 source_context='references/Hong_Sheen_2026_BSSA_Moment_magnitude.pdf; DOI 10.1785/0120250293')
(R/'tables/KR_Mw_comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
figures=[]
def save(fig,stem,caption):
    fig.text(.075,.035,caption,fontsize=8.5,color='.25',linespacing=1.4)
    for ext in ['pdf','png']:fig.savefig(R/'figures'/(stem+'.'+ext))
    plt.close(fig);figures.append(dict(stem=stem,caption=caption))

fig,axes=plt.subplots(2,2,figsize=(11,9))
fig.subplots_adjust(left=.09,right=.96,bottom=.17,top=.87,hspace=.33,wspace=.27)
fig.suptitle('KR_Mw: event-by-event comparison',fontsize=15,y=.96)
for j,(field,title) in enumerate([('mw_spectra','KR_Mw spectral Mw'),('mw_TD','KR_Mw time-domain Mw')]):
    st=stats.query("selection=='Mw A' and our_estimator=='Mw' and KR_field==@field").iloc[0]
    ax=axes[0,j];ax.scatter(g[field],g.Mw,s=22,c=BLUE,alpha=.75,edgecolors='none')
    lo=min(g[field].min(),g.Mw.min())-.1;hi=max(g[field].max(),g.Mw.max())+.1
    ax.plot([lo,hi],[lo,hi],c='.45',ls=':',lw=1)
    ax.set(xlabel=title,ylabel='Our multitaper Mw',xlim=(lo,hi),ylim=(lo,hi))
    ax.set_title(f'n = {len(g)}; median difference {st.median_difference:+.3f}',loc='left');ax.grid(color='.92')
    ax=axes[1,j]
    for our,label,color in [('Mw','Multitaper',BLUE),('Mw_Fourier','Fourier',ORANGE)]:
        ax.scatter(g[field],g[our]-g[field],s=16,alpha=.6,c=color,edgecolors='none',label=label)
    ax.axhline(0,c='.4',lw=.9);ax.set(xlabel=title,ylabel='Our Mw − KR_Mw');ax.grid(color='.92')
    if j==0:ax.legend(frameon=False)
save(fig,'09_KR_Mw_comparison',
 '199 quality-A events matched by exact UTC ID, origin time and KMA ML. Our source Vs(z): Kim & Kim (1983), Vp/Vs=1.73.\n'
 'Dotted: equal Mw. Published KR_Mw is used as reported; no empirical offset. Both estimators use the same events.')

fig,axes=plt.subplots(2,2,figsize=(11,9))
fig.subplots_adjust(left=.09,right=.96,bottom=.17,top=.87,hspace=.37,wspace=.29)
fig.suptitle('Multitaper and Fourier: accepted and rejected estimates',fontsize=14,y=.96)
for quality,color in [('C','.7'),('B',ORANGE),('A',BLUE)]:
    z=est[est.Mw_quality==quality];axes[0,0].scatter(z.Mw_Fourier,z.Mw,s=17,c=color,alpha=.65,label=f'{quality} ({len(z)})',edgecolors='none')
axes[0,0].plot([1,5.5],[1,5.5],c='.4',ls=':',lw=1);axes[0,0].set(xlabel='Fourier Mw',ylabel='Multitaper Mw');axes[0,0].legend(frameon=False,fontsize=8);axes[0,0].grid(color='.92')
for quality,color in [('A',BLUE),('B',ORANGE),('C','.55')]:
    z=est.loc[est.Mw_quality==quality,'MT_minus_Fourier'].dropna();axes[0,1].hist(z,bins=np.linspace(-.75,.75,51),histtype='step',lw=1.3,color=color,label=quality)
axes[0,1].axvline(-.1,c='.6',ls=':',lw=.8);axes[0,1].axvline(.1,c='.6',ls=':',lw=.8)
axes[0,1].set(xlabel='Multitaper Mw − Fourier Mw',ylabel='Events');axes[0,1].legend(frameon=False)
fits=pd.read_csv(R/'tables/clean_spectral_fits.csv',dtype={'event_id':str})
for ax,eid,title in [(axes[1,0],'20160829153946','29 August 2016; quality C'),(axes[1,1],'20160502182909','2 May 2016; quality C')]:
    b=np.load(R/'spectra'/(eid+'.npz'));f=b['frequency_Hz'];a=b['T10.1'];ok=np.isfinite(a[:,0]).sum(axis=0)>=max(3,.3*len(a))
    vals=[]
    for j,method,color in [(0,'Multitaper',BLUE),(1,'Direct Fourier',ORANGE)]:
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);y=np.nanmean(a[:,j],axis=0)
        ax.loglog(f[ok],10**y[ok],c=color,lw=1.3,label=method)
        row=fits[(fits.event_id==eid)&(fits.sensor_set=='ALL')&(fits.window_s==10.1)&(fits.method==method)&(fits['shape']=='Boatwright')].iloc[0]
        pred=np.log10(row.omega_r_m2s)-.5*np.log10(1+(f/row.fc_Hz)**(2*row.falloff))
        ax.loglog(f,10**pred,c=color,lw=1,ls='--');vals.append(f'{"MT" if j==0 else "FFT"} Mw {row.Mw:.2f}, fc {row.fc_Hz:.2f} Hz')
    ax.set_title(title,loc='left');ax.set(xlabel='Frequency (Hz)',ylabel=r'Corrected amplitude (m$^2$ s)');ax.grid(color='.92')
    ax.text(.04,.05,'\n'.join(vals),transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
save(fig,'10_MT_Fourier_diagnosis',
 'All finite fits are shown. A requires |ΔMw| ≤0.10; this is an acceptance rule.\n'
 'Bottom: cut-edge leakage (left), unresolved plateau and corner tradeoff (right). Both examples are excluded.')

# A representative good record and a rejected record with a strong cut-edge artifact.
arr=np.load(R/'spectra/estimator_waveform_controls.npz')
fig,axes=plt.subplots(2,2,figsize=(11,9))
fig.subplots_adjust(left=.09,right=.96,bottom=.17,top=.86,hspace=.33,wspace=.27)
fig.suptitle('Why window edges can change spectral amplitudes',fontsize=15,y=.96)
for j,(key,title) in enumerate([('20261117141017_SAJB','Sangju 2026; SAJB HH'),('20160829153946_CPR2','29 August 2016; CPR2 EL')]):
    tt=arr[key+'_time'];xx=arr[key+'_displacement'];ff=arr[key+'_frequency']
    ax=axes[0,j]
    for i,color,label in [(0,BLUE,'Z'),(1,ORANGE,'N'),(2,'#34855b','E')]:ax.plot(tt,xx[i]*1e6,c=color,lw=.7,label=label)
    ax.set_title(title,loc='left');ax.set(xlabel='Time after S (s)',ylabel='Displacement (µm)');ax.grid(color='.92')
    if j==0:ax.legend(frameon=False,ncol=3,fontsize=8)
    ax=axes[1,j]
    for tag,label,color,ls in [('Fourier','Rectangular FFT',ORANGE,'-'),('Multitaper','Multitaper',BLUE,'-'),('Cosine_Fourier','FFT, 5% cosine taper','#34855b','--')]:
        ax.loglog(ff,arr[key+'_'+tag],c=color,lw=1.2,ls=ls,label=label)
    if j==1:
        jump=np.linalg.norm(xx[:,-1]-xx[:,0]);ax.loglog(ff,jump/(2*np.pi*ff),c='.4',ls=':',lw=1,label=r'$|\Delta u|/(2\pi f)$')
    ax.set(xlabel='Frequency (Hz)',ylabel='Fourier amplitude (m s)',xlim=(.5,20));ax.grid(color='.92');ax.legend(frameon=False,fontsize=8)
save(fig,'11_Window_edge_controls',
 'Same corrected records and 10.1 s windows. Right: long-period motion cut at unequal endpoint values.\n'
 'Cosine tapering reduces its artificial FFT tail. Gray dots: endpoint scale. The taper is a diagnostic control.')

(R/'tables/comparison_figure_manifest.json').write_text(json.dumps(figures,indent=2)+'\n')
mtst=stats.query("selection=='Mw A' and our_estimator=='Mw' and KR_field=='mw_spectra'").iloc[0]
tdst=stats.query("selection=='Mw A' and our_estimator=='Mw' and KR_field=='mw_TD'").iloc[0]
target=json.loads((R/'tables/Sangju2026_summary.json').read_text())
targetfits=pd.read_csv(R/'tables/Sangju2026_spectral_fits.csv')
def target_value(method,window):
    return targetfits[(targetfits.radius_km==100)&(targetfits.sensor_set=='ALL')&
      (targetfits.method==method)&(targetfits.window_s==window)&(targetfits['shape']=='Boatwright')].iloc[0].Mw
prefactor=json.loads((R/'tables/Hong_paper_method_audit.json').read_text())['source_prefactor_shift_Hong_minus_ours_by_model']['Kim1983']
body=[('Which Mw is adopted',[
 'The adopted Mw is multitaper/generalized Boatwright, 10.1 s, NW=4/K=7, C=9.05, with source-depth Vs from Kim & Kim (1983), Vp/Vs=1.73. Published Kim et al. (2011) is the model comparison. Direct Fourier uses identical records and corrections. C/F estimates remain diagnostic; only quality A enters the primary relation.',
 f"Among 443 quality-A Mw events, the median MT−FFT difference is {eststats['A']['median']:+.3f}. All satisfy |ΔMw|≤0.10 by construction. Across all finite fits, 145 differences exceed 0.10; the atlas deliberately includes these rejected examples. That selection cannot independently demonstrate absolute calibration.",
 f"Sangju within 100 km: 10.1 s gives MT Mw {target_value('Multitaper',10.1):.3f} and Fourier {target_value('Direct Fourier',10.1):.3f}; 20.1 s gives {target_value('Multitaper',20.1):.3f} and {target_value('Direct Fourier',20.1):.3f}. Kim2011 gives MT Mw {target['preferred']['Mw_source_Vs_Kim2011']:.3f}. This revision updates the source coefficient, while preserving station selection and spectra.",
 'Multitaper combines power from seven DPSS-tapered transforms adaptively, then the library normalizes its PSD integral to the segment variance. Our FAS is sqrt(N*dt*P1/2). Direct Fourier is dt*abs(FFT) of the demeaned rectangular segment. Equal total energy does not imply equal amplitudes in every band; power averaging and square root do not commute.',
 f"In the rejected 29 August 2016 example, the median 2–10 Hz MT/FFT ratio over 18 stations is {summary['edge_example']['median_MT_over_rectangular_FFT']:.3f}. A 5% cosine taper reduces the FFT to {summary['edge_example']['median_cosine_FFT_over_rectangular_FFT']:.3f} of its rectangular amplitude. The traces expose strong long-period motion and unequal cut-edge values. At CPR2, the broadband tail follows the endpoint discontinuity scale. This diagnoses leakage in the rectangular FFT; it is not evidence for a universal amplitude multiplier.",
 'The rejected 2 May 2016 example has much closer observed spectra at low frequencies, but the MT corner reaches 0.20 Hz while the FFT corner is 0.96 Hz. Neither resolves the plateau below the usable band; their extrapolated Mw differ by 0.63. This is a source-fit tradeoff rather than a single spectral normalization error.'
 ]),('Comparison with KR_Mw',[
 f"The pinned public snapshot, commit 61c24f655b83e320675cf64f01bc862792947d55, contains 490 event rows for 2017–2022 and 12,413 station rows. There are {len(matched)} exact UTC event-ID matches with our depth-qualified cohort; all matched origin times and reported KMA ML values also agree. {len(g)} matched events have our Mw quality A; 165 also pass independent-ML quality A.",
 f"For the same 199 Mw-A events, our MT minus KR spectral Mw has median {mtst.median_difference:+.3f}, mean {mtst.mean_difference:+.3f}, raw RMS {mtst.raw_RMS:.3f} and centered RMS {mtst.centered_RMS:.3f}. Against KR time-domain Mw the corresponding values are {tdst.median_difference:+.3f}, {tdst.mean_difference:+.3f}, {tdst.raw_RMS:.3f} and {tdst.centered_RMS:.3f}. The observed residuals are retained without any zero-point adjustment.",
 'Our direct-Fourier estimates are also included on those same events. The spectral-Brune and source-Vs sensitivities remain available in the numerical comparison. These describe differences between processing/model choices; none supplies an independent absolute moment reference.',
 'The supplied Hong and Sheen (2026) BSSA paper establishes fixed source Vs=3.61 km/s at 10 km, citing Kim et al. (2011), density 2700 kg/m³, C=9.05, radiation 0.63, free surface 2 and the main spectral/time-domain methods. Exact PSD-to-FAS normalization, DPSS and response-removal settings remain unspecified.',
 'The station-detail file does not exactly reconstruct either event-summary station set: row counts equal nsta_spectra for 291 of 490 events and nsta_TD for 184. Consequently its mw_td field and the event MAD columns are not treated as a documented event-level likelihood or calibrated uncertainty. Event comparisons use the Events file as published.',
 f"At an identical plateau, Hong minus our density/Vs coefficient now ranges from {prefactor['min']:+.4f} to {prefactor['max']:+.4f} Mw over the quality-A source layers; it is not a global offset. Material-only controls are explicit columns. Their TD peak-to-spectrum factors 1.62 and 2.25 are empirical calibrations, not PSD folding factors. No empirical offset is fitted.",
 'The earlier slopes in KR_Mw_comparison_statistics.csv are vertical OLS. The new Hong_KR_Mw_regression_and_stability.pdf provides ODR, matched-event ML–Mw comparisons, selection tests and estimator stability. Sources: pinned github.com/dhsheen/KR_Mw CSVs and the supplied BSSA paper, DOI 10.1785/0120250293. Input files retain verified hashes.'
 ])]
part=R/'reports/.estimator_comparison_text.pdf'
with PdfPages(part) as pdf:
    for title,paragraphs in body:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.945,title,fontsize=17,va='top');y=.888
        for p in paragraphs:
            txt=textwrap.fill(p,96,break_long_words=False,break_on_hyphens=False)
            fig.text(.08,y,txt,fontsize=9.5,va='top',linespacing=1.45)
            y-=.0165*(txt.count('\n')+1)+.024
        if y<.055:raise RuntimeError('Narrative overflows: '+title+str(y))
        fig.text(.08,.035,'19 November 2026  |  Source Vs(z) updated; measured spectra and published KR_Mw preserved',fontsize=8,color='.4')
        pdf.savefig(fig);plt.close(fig)
doc=fitz.open()
with fitz.open(part) as p:doc.insert_pdf(p)
for figure in figures:
    with fitz.open(R/'figures'/(figure['stem']+'.pdf')) as p:doc.insert_pdf(p)
with fitz.open(R/'figures/16_source_velocity_models.pdf') as p:doc.insert_pdf(p)
doc.save(R/'reports/Estimator_and_KR_Mw_comparison.pdf',garbage=4,deflate=True);part.unlink()
(R/'reports/Estimator_and_KR_Mw_comparison.txt').write_text('\n\n'.join(title+'\n\n'+'\n\n'.join(paragraphs) for title,paragraphs in body)+'\n')
result=dict(passed=True,repository_blob_hashes='verified',exact_matching='294 unique IDs, origins and KMA ML verified',
 report_pages=len(doc),primary_catalog_modified_by_comparison=False,source_velocity_updated=True,
 source_velocity_config_sha256=CONFIG_SHA256,max_waveform_reproduction_error=float(wave.reproduction_error_log10.max()),
 repeated_waveforms=len(wave),universal_scale_correction_applied=False)
audit=json.loads((R/'tables/source_velocity_update.json').read_text())
expected={name:digest for name,digest in audit['immutable_sha256'].items() if name.startswith('inputs/KR_Mw/') or name=='tables/independent_ML_catalog.csv'}
for filename,digest in expected.items():assert hashlib.sha256((R/filename).read_bytes()).hexdigest()==digest,filename
result['unchanged_input_sha256']=expected
(R/'tables/comparison_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2));print(json.dumps(result,indent=2))
