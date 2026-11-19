"""Publish current held-out magnitudes in the single Sangju result tree."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import pandas as pd
import fitz
R=Path(__file__).resolve().parents[1];S=R.parent/'sangju_20261117'
t=json.loads((R/'tables/Sangju2026_summary.json').read_text());p=t['preferred'];m=t['local_magnitude']
fits=pd.read_csv(R/'tables/Sangju2026_spectral_fits.csv')
folder=S/'results/tables/magnitudes';folder.mkdir(exist_ok=True)
(folder/'current_magnitudes.json').write_text(json.dumps(t,indent=2)+'\n')
fits.to_csv(folder/'spectral_fits.csv',index=False)
stations=pd.read_csv(R/'tables/independent_ML_stations.csv',dtype={'event_id':str})
stations[stations.event_id==t['event_id']].to_csv(folder/'ML_stations.csv',index=False)
def value(sensor):
    return fits[(fits.radius_km==100)&(fits.window_s==10.1)&(fits.sensor_set==sensor)&
      (fits.method=='Multitaper')&(fits['shape']=='Boatwright')].iloc[0]
table=''
for sensor,label,key in [('ALL','HH + EL + HG','ML_Sheen'),('HH_EL','HH + EL','ML_velocity'),('HG','HG','ML_HG')]:
    r=value(sensor);table+=f'<tr><td>{label}</td><td>{r.Mw:.3f}</td><td>{r.n_stations}</td><td>{m[key]:.3f}</td><td>{m[key+"_n"]}</td></tr>'
lo,hi=t['Mw_station_resampling_interval'];llo,lhi=t['ML_station_resampling_interval']
depth_models=t['depth_endpoint_magnitudes']
html=f'''<h1>Sangju: moment and local magnitude</h1>
<p>17 November 2026, 23:10:17 KST · Updated 19 November 2026</p>
<h2>Mw {p['Mw']:.2f} · ML {m['ML_Sheen']:.2f}</h2>
<p>The preferred multitaper estimate is <b>M0 = {p['M0_Nm']:.3e} N m</b>, from {p['n_stations']} distinct sites within 100 km.
The independent local magnitude uses {m['ML_Sheen_n']} horizontal sites: {m['ML_n_HH']} HH, {m['ML_n_EL']} EL and {m['ML_n_HG']} HG.
Reported KMA ML is 3.6. Location depth is 19.76 km, with project uncertainty envelope 17.68–21.31 km (G3).</p>
<table><tr><th>Sensors</th><th>Mw</th><th>Sites</th><th>ML</th><th>Sites</th></tr>{table}</table>
<p>Each site contributes once. Among usable records, priority is HH, then EL, then HG. These sensor controls use identical processing.
HG greatly improves coverage. Velocity-only and HG-only Mw estimates agree closely; ML varies with the station set.</p>
<h2>Response removal and spectral estimation</h2>
<p>Counts are converted to displacement using exact channel/epoch responses, with <b>water_level=None</b> and a 0.1/0.2/25/40 Hz cosine prefilter.
A single 1% full-record taper is capped at 5 s. This avoids the tested HG attenuation caused by a global 60 dB floor on displacement response.
Known-displacement recovery controls pass for both sensor families. Five agreeing co-located Sangju pairs also pass the adopted source-band check.</p>
<p>Three-component S spectra use 10.1 s windows, with 20.1 s and direct Fourier controls. Prieto Python multitaper MTSpec uses NW=4, K=7;
one-sided PSD becomes Fourier amplitude as √(NΔt P₁/2). Amplitude SNR ≥3, geometric spreading, Q(f)=229.2 f⁰·⁷³,
ρ=2800 kg/m³, source β={p['source_Vs_m_s']/1000:.4f} km/s from Kim &amp; Kim (1983), Vp/Vs=1.73, radiation 0.63 and free surface 2.
Kim et al. (2011), β={p['source_Vs_Kim2011']/1000:.2f} km/s at 19.76 km, gives Mw {p['Mw_source_Vs_Kim2011']:.3f}.
Both models change the source coefficient only; geometry/Q stay fixed. Mw=(log₁₀ M0−9.05)/1.5.</p>
<h2>Local magnitude and uncertainty</h2>
<p>ML uses revised Wood–Anderson amplitudes, the causal 0.5–10 Hz Haenam filter, N/E geometric means and the published Sheen et al. epicentral attenuation.
Both horizontal peak SNRs must be ≥3; distances are 10–100 km. Station terms are zero; no magnitude is adjusted to agree with KMA or Mw.</p>
<p>Conditional station-resampling intervals are Mw {lo:.2f}–{hi:.2f} and ML {llo:.2f}–{lhi:.2f}.
They exclude absolute response-history, site, radiation, attenuation and source-model errors. The 110 km spectral result and estimator/window controls are plotted next.</p>
<p>Across the project depth envelope, distance and source-Vs changes give conditional Mw endpoints
{depth_models['Kim1983']['lower']:.3f}–{depth_models['Kim1983']['upper']:.3f} for Kim &amp; Kim,
and {depth_models['Kim2011']['lower']:.3f}–{depth_models['Kim2011']['upper']:.3f} for Kim2011.
These are model/depth sensitivities, not confidence intervals.</p>
<p>KS.BURB is excluded from both magnitudes because its co-located horizontal spectra disagree substantially. Neither sensor is assumed correct there.
These magnitude results do not change the arrival-time depth or the waveform focal-mechanism comparison.</p>
<p>Full national analysis: ../korean_mw_ml/reports/Korean_Mw_ML_study.pdf.<br>
Numerical Sangju results: results/tables/magnitudes/. Shared pipeline and full provenance: ../korean_mw_ml/.</p>'''
css='body{font-family:sans-serif;font-size:10pt;line-height:1.34;color:#222;}h1{font-size:20pt;margin:0 0 10pt;}h2{font-size:13pt;margin:13pt 0 6pt;}p{margin:0 0 8pt;}table{border-collapse:collapse;width:100%;font-size:10pt;margin:10pt 0;}td,th{text-align:left;padding:5pt;border-bottom:.5pt solid #ccc;}th{background:#edf2f5;}'
doc=fitz.open();page=doc.new_page(width=595.276,height=841.89)
spare,scale=page.insert_htmlbox(fitz.Rect(42,30,553,807),html,css=css,scale_low=.9)
if spare<0:raise RuntimeError('Sangju narrative does not fit')
for name in ['01_Sangju_magnitude','02_Sangju_local_magnitude','03_Sangju_WA_waveforms']:
    with fitz.open(R/'figures'/(name+'.pdf')) as q:doc.insert_pdf(q)
doc.save(S/'Sangju_20261117_magnitudes.pdf',garbage=4,deflate=True)
with fitz.open(S/'Sangju_20261117_magnitudes.pdf') as q:
    (folder/'magnitude_report.txt').write_text('\n\n'.join(p.get_text() for p in q))
summary=json.loads((S/'results/summary.json').read_text())
summary.update(current_spectral_Mw=p['Mw'],current_M0_Nm=p['M0_Nm'],current_local_ML=m['ML_Sheen'],
 current_magnitude_sensors=['HH','EL','HG'],magnitude_constant=9.05,
 spectral_magnitudes='tables/magnitudes/current_magnitudes.json; current physical Fourier-amplitude analysis',
 current_magnitude_report='Sangju_20261117_magnitudes.pdf')
summary['preferred_waveform']['Mw_9p05']=summary['preferred_waveform']['Mw_9p1']+.05/1.5
(S/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
readme=(S/'README.txt').read_text()
marker='CURRENT MAGNITUDES — 19 NOVEMBER 2026'
if marker in readme:readme=readme[:readme.index(marker)].rstrip()+'\n'
if '  Sangju_20261117_magnitudes.pdf' not in readme:
    readme=readme.replace('Reports:\n','Reports:\n  Sangju_20261117_magnitudes.pdf                   current Mw and ML, 4 pages\n')
readme=readme.replace('All current numerical results, input records, figures and code are in results/.','Depth, waveform and seismicity products are in results/. Current target magnitude\ntables are also there; the shared Mw/ML pipeline is in ../korean_mw_ml/.')
readme=readme.replace('Preferred waveform solution: Kim2011, 20 km, Mw 3.251, VR 82.09%.','Preferred waveform solution: Kim2011, 20 km, Mw 3.284 (C=9.05), VR 82.09%.\n  The same moment gives Mw 3.251 at C=9.1.')
readme=readme.replace('  Sangju_20261117_spectral_conventions.pdf          FFT/PSD and Mw audit, 5 pages',
 '  Sangju_20261117_spectral_conventions.pdf          convention audit, 5 pages; numerical examples are diagnostic')
# Insert the note once; earlier runs inserted it on every run, so collapse repeats first.
note='  These are convention-audit examples, not the current adopted magnitudes.\n'
anchor='  A convention conversion alone does not establish unbiased transient'
while note+note in readme:readme=readme.replace(note+note,note)
if note+anchor not in readme:readme=readme.replace(anchor,note+anchor)
readme+='\n'+marker+f'''\n\nStart with Sangju_20261117_magnitudes.pdf (4 pages) for current Mw and ML.
Spectral Mw = {p['Mw']:.3f}; M0 = {p['M0_Nm']:.3e} N m; {p['n_stations']} sites.
Source Vs: Kim & Kim (1983), Vp/Vs=1.73, {p['source_Vs_m_s']/1000:.4f} km/s.
Published Kim et al. (2011) source Vs gives Mw {p['Mw_source_Vs_Kim2011']:.3f} at the same 19.76 km depth.
Independent ML = {m['ML_Sheen']:.3f}; {m['ML_Sheen_n']} sites ({m['ML_n_HH']} HH, {m['ML_n_EL']} EL, {m['ML_n_HG']} HG).
Both use dated instrument responses and include qualified accelerometers.
The shared pipeline, 642-event national cohort and velocity-only controls are
in ../korean_mw_ml/. Compact current target tables are in results/tables/magnitudes/.
Convention-audit examples retain their explicitly stated processing and constants;
the current magnitude report supersedes those examples as the adopted Mw/ML estimate.
The waveform solution remains Mw {summary['preferred_waveform']['Mw_9p05']:.3f} at C=9.05
(Mw {summary['preferred_waveform']['Mw_9p1']:.3f} at C=9.1), with its existing depth and mechanism.
'''
(S/'README.txt').write_text(readme)
print('Sangju report pages',len(doc),'narrative scale',scale)
