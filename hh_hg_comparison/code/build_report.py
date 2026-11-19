"""Summarize the measured paired-sensor evidence and attach its figure pages."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import fitz
ROOT=Path(__file__).resolve().parents[1]
bands=pd.read_csv(ROOT/'tables/spectral_comparison.csv',dtype={'event_id':str})
ml=pd.read_csv(ROOT/'tables/paired_ML.csv',dtype={'event_id':str})
usage=json.loads((ROOT/'tables/Haenam_usage.json').read_text())
validation=json.loads((ROOT/'tables/processing_validation.json').read_text())
responses=pd.read_csv(ROOT/'tables/response_checks.csv')
water=pd.read_csv(ROOT/'tables/waterlevel_transfer.csv',dtype={'event_id':str})
s=bands[(bands.name=='Sangju')&(bands.band_low_Hz==.7)]
good=s[s.station!='BURB'];low=bands[(bands.name=='Sangju')&(bands.band_low_Hz==.05)]
hg=next(r for r in usage['local_ml'] if r['sensor']=='HG')
q=water[(water.name=='Sangju')&(water.sensor=='HG')&water.frequency_Hz.between(.7,1.2)].groupby('station').WL60_to_noWL.median()
summary=dict(n_pairs=validation['n_station_event_pairs'],Haenam_usage=usage,
    Sangju_source_band=s.to_dict('records'),Sangju_low_frequency=low.to_dict('records'),
    Sangju_five_station_amplitude_ratio_range=[float(good.HG_HH_ratio_median.min()),float(good.HG_HH_ratio_median.max())],
    Sangju_five_station_coherence_range=[float(good.coherence_median.min()),float(good.coherence_median.max())],
    HG_displacement_waterlevel60_band_attenuation_range=[float(q.min()),float(q.max())],
    response_stage_to_stated_gain_range=[float(responses.full_to_stated_gain.min()),float(responses.full_to_stated_gain.max())],
    interpretation='HG is demonstrably useful for ML and source-band spectra. Restricting all accelerometers by channel label is unsupported. Low-frequency MT, station anomalies and response-removal regularization require separate checks.',
    magnitude_estimation_performed_by_this_report=False)
(ROOT/'tables/comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def p(text):return '<p>'+text+'</p>'
def table(head,rows):
    return '<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
css='''body{font-family:sans-serif;font-size:10pt;line-height:1.36;color:#20272c;}
h1{font-size:20pt;line-height:1.12;margin:0 0 13pt;}h2{font-size:13pt;margin:14pt 0 6pt;}
p{margin:0 0 8pt;}table{width:100%;border-collapse:collapse;font-size:9pt;margin:8pt 0 10pt;}
td,th{padding:5pt 4pt;border-bottom:.5pt solid #cfd6dc;}th{text-align:left;background:#edf2f5;}a{color:#176da0;}'''
pages=[]
pages.append('<h1>HH and HG in Haenam<br>and Sangju</h1>'+p('19 November 2026 · Direct comparison of paired recordings')+
 p('<b>The Haenam project uses HG channels for both local magnitude and spectral moment magnitude.</b> The directly measured Sangju pairs also show that calibrated accelerometers provide useful spectra in the source-moment band. A channel-type exclusion is therefore too restrictive for the primary ML and spectral-Mw comparison.')+
 '<h2>What the existing Haenam project contains</h2>'+
 p(f'The full-period ML station table contains {hg["station_event_pairs"]:,} horizontal HG station–event measurements with observed P and component peak SNR ≥2, spanning {hg["events"]} events and {hg["stations"]} station codes. These counts describe retained station measurements; event-level ML gates are a further selection.')+
 p('For the 25 August 2026 spectral fit (00:41:22 UTC), the saved 59 station spectra comprise <b>40 HG, 16 HH and 3 EL</b>. The R6 source table identifies that fit as accepted. Thus HG use is present in computed products, as well as in the acquisition configuration.')+
 '<h2>Sangju: same stations, same windows</h2>'+
 p('Six HH/HG pairs were read from the original NECIS event packages. The table uses three-component, noise-subtracted spectra over 0.7–1.2 Hz, with a shared 10.1 s S window. Instrument responses are dated; amplitudes are in physical units and are never rescaled to force agreement.')+
 table(['Station','Distance km','HG SNR','HG / HH','Coherence'],[[r.station,f'{r.epicentral_km:.1f}',f'{r.HG_SNR_median:.1f}',f'{r.HG_HH_ratio_median:.3f}',f'{r.coherence_median:.3f}'] for _,r in s.iterrows()])+
 p(f'At five stations, median amplitude ratios are {good.HG_HH_ratio_median.min():.3f}–{good.HG_HH_ratio_median.max():.3f}, with component coherence {good.coherence_median.min():.3f}–{good.coherence_median.max():.3f}. Interpreting those ratios purely as moment ratios would change Mw by at most about 0.03. This is a relative sensor check, not an independent absolute-Mw determination.')+
 p('CHJ3 agrees closely across the broader band. BURB is anomalous: HG/HH is about 0.25 at 2–5 Hz and 0.21 at 5–10 Hz despite strong signals in both records. This warrants a response, installation and site audit. The pair alone does not establish which recording represents the desired ground motion more accurately.')+
 p('The waveforms also reveal horizontal phase differences at HANB. Amplitude ratios use the three-component power sum, which is insensitive to an orthogonal axis rotation. Coherence and individual waveforms retain these orientation diagnostics.'))

pages.append('<h1>Frequency, noise and processing</h1>'+
 '<h2>Small events and low frequencies are different tests</h2>'+
 p('The Haenam ML 3.1 and ML 1.3 examples use original HH/HG continuous recordings at MOPB, HANB, JAEB and BGDB, at roughly 17–44 km epicentral distance. HG has useful event energy at several Hz even for the small event. At JAEB, its Wood–Anderson peak SNR is 3.37, compared with 2.55 for HH, so a peak-SNR threshold of three retains that HG station measurement.')+
 p('That small event does not provide a shared, high-SNR 0.7–1.2 Hz plateau at these four stations. Detectability, a usable ML peak and a resolved spectral moment are separate requirements. A dense network can still add usable nearby HG sites when the closest velocity station is distant or unavailable.')+
 p(f'For Sangju, the 80 s full-wave test gives HG median amplitude SNR {low.HG_SNR_median.min():.2f}–{low.HG_SNR_median.max():.2f} over 0.05–0.2 Hz. None of the six HG band medians reaches three. HH is also weak in much of that band. This test supports separate quality screening for low-frequency MT; it does not justify rejecting HG from the higher-frequency ML or source-spectrum analyses.')+
 '<h2>A response-removal effect that can be tested</h2>'+
 p('The inherited Haenam displacement conversion uses a 60 dB water level. Its floor is relative to the largest gain of the requested displacement response. For an accelerometer, that response rises approximately as f² through its flat-acceleration band, so a global floor can attenuate useful lower frequencies.')+
 p(f'The actual Sangju HG responses give median inverse-response amplitude factors of {q.min():.2f}–{q.max():.2f} over 0.7–1.2 Hz under that floor; the paired HH response remains unattenuated there. The known-amplitude sinusoid control reproduces the analytic attenuation. <a href="https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.remove_response.html">ObsPy documents this non-native-output water-level issue.</a>')+
 p('The primary comparison removes responses to acceleration with water_level=None and a band-limiting prefilter, then converts the spectra to displacement-equivalent units. A prefilter is still required; simply removing the water level from a poorly band-limited displacement trace can admit drift and spectral leakage. The older Haenam spectral fits generally start above this low-frequency band, so this test does not establish a single correction to their Mw values.')+
 '<h2>Reproducible measurement choices</h2>'+
 p('Both sensors use the same UTC grid and windows. Short spectra: 10.1 s after S−0.1, DPSS NW=3/K=5; long spectra: 80 s after origin, NW=2/K=3. Equal-length noise ends one second before P. The displacement-equivalent ASD is √Pacc/(2πf)². Ratios use noise-subtracted power where both amplitude SNRs exceed three. Coherence is a power-weighted mean of component magnitude-squared coherences.')+
 p('Source records, response epochs and hashes are retained. All 84 responses have consistent full-stage and stated sensitivities, within 0.6%. The sensor pairs share station/location IDs; the RESP files do not independently establish physical separation, installation depth or axis orientation. The 80 s record contains four cycles at 0.05 Hz, and spectral bins are correlated. These are instrument checks rather than new source inversions.')+
 p('<b>Implication:</b> include HG when its calibration and analysis-band quality pass the same checks as velocity records, keep one contribution per station, and retain a velocity-only sensitivity. The subsequent HG-inclusive magnitude estimates are in the main Korean Mw–ML study; this appendix supplies the instrument validation.'))

doc=fitz.open()
for i,html in enumerate(pages):
    page=doc.new_page(width=595.276,height=841.89)
    spare,scale=page.insert_htmlbox(fitz.Rect(42,36,553,800),html,css=css,scale_low=.90)
    if spare<0:raise RuntimeError('Narrative page does not fit '+str(i+1))
    print('Narrative page',i+1,'scale',scale)
for entry in json.loads((ROOT/'tables/figure_manifest.json').read_text()):
    part=fitz.open(ROOT/'figures'/(entry['stem']+'.pdf'));doc.insert_pdf(part)
for i,page in enumerate(doc):page.insert_text((page.rect.width-55,page.rect.height-16),str(i+1),fontsize=8,color=(.4,.4,.4))
doc.set_metadata({'title':'HH and HG in Haenam and Sangju: paired recordings and spectra','author':'Korean earthquake magnitude study'})
path=ROOT/'reports/HH_HG_paired_comparison.pdf';doc.save(path,garbage=4,deflate=True)
path.with_suffix('.txt').write_text('\n\n'.join(p.get_text() for p in doc))
for i in range(2):doc[i].get_pixmap(matrix=fitz.Matrix(1.1,1.1)).save('/tmp/hh_hg_report_'+str(i+1)+'.png')

print('Standalone validation report pages',len(doc))
