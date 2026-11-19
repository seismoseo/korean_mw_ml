"""Current HH/EL/HG figures and reports. All numerical text comes from tables."""
from pathlib import Path
import json,sys,textwrap,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import fitz
from source_velocity import magnitude_shift,CONFIG
class PdfWriter:
    def __init__(self):self.doc=fitz.open()
    def append(self,path):
        with fitz.open(path) as part:self.doc.insert_pdf(part)
    def write(self,path):self.doc.save(path,garbage=4,deflate=True);self.doc.close()
R=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
 'axes.linewidth':.7,'savefig.dpi':220})
BLUE='#176da0';ORANGE='#bd591b';GREEN='#34855b';COL={'HH':BLUE,'EL':GREEN,'HG':ORANGE}
mw=pd.read_csv(R/'tables/Mw_ML_catalog.csv',dtype={'event_id':str})
d=pd.read_csv(R/'tables/Mw_independent_ML_catalog.csv',dtype={'event_id':str})
g=d[d.ML_comparison_primary].copy();a=mw[(mw.Mw_quality=='A')&mw.primary_geography]
reg=pd.read_csv(R/'tables/independent_ML_regressions.csv')
old=pd.read_csv(R/'inputs/previous_velocity_results.csv',dtype={'event_id':str})
change=d.merge(old,on='event_id',suffixes=('','_previous'),validate='one_to_one')
change['Mw_previous_reference_Vs3500']=change.Mw_previous
change['Mw_previous']=change.Mw_previous+magnitude_shift(change.z_central.to_numpy())
change['Mw_HG_increment']=change.Mw-change.Mw_velocity
change['ML_HG_increment']=change.ML_Sheen-change.ML_velocity
change['Mw_processing_change']=change.Mw_velocity-change.Mw_previous
change['ML_processing_change']=change.ML_velocity-change.ML_Sheen_previous
change.to_csv(R/'tables/HG_update_comparison.csv',index=False)
t=json.loads((R/'tables/Sangju2026_summary.json').read_text())
ts=pd.read_csv(R/'tables/Sangju2026_spectral_fits.csv')
tc=pd.read_csv(R/'tables/Sangju2026_spectral_curves.csv')
mlsta=pd.read_csv(R/'tables/independent_ML_stations.csv',dtype={'event_id':str})
targetsta=mlsta[mlsta.event_id=='20261117141017']
summary=json.loads((R/'tables/independent_ML_summary.json').read_text())
validation=json.loads((R/'ml_validation/HG_processing_validation.json').read_text())
figures=[]

def canvas(title,rows=1,cols=2,size=(11,7.6),**kw):
    fig,axes=plt.subplots(rows,cols,figsize=size,**kw)
    fig.subplots_adjust(left=.09,right=.96,top=.86,bottom=.17,hspace=.4,wspace=.3)
    fig.suptitle(title,y=.955,fontsize=15)
    return fig,axes

def finish(fig,name,caption):
    fig.text(.075,.045,caption,fontsize=8.6,color='.25',linespacing=1.5,va='bottom')
    fig.savefig(R/'figures'/(name+'.pdf'));fig.savefig(R/'figures'/(name+'.png'))
    plt.close(fig);figures.append(dict(stem=name,caption=caption))

def coast(ax):
    seg=[]
    for line in (R/'inputs/coastline_low.gmt').read_text().splitlines()+['>']:
        if line.startswith('>'):
            if seg:
                q=np.array(seg);ax.plot(q[:,0],q[:,1],c='.5',lw=.5,zorder=0)
            seg=[]
        else:
            try:seg.append([float(x) for x in line.split()[:2]])
            except ValueError:pass

# Held-out target: three independent sensor selections under common processing.
fig,(ax,bx)=canvas('Sangju, 17 November 2026: spectral moment')
for label,sensor,color in [('HH + EL + HG','ALL','black'),('HH + EL','HH_EL',BLUE),('HG','HG',ORANGE)]:
    q=tc[(tc.radius_km==100)&(tc.window_s==10.1)&(tc.sensor_set==sensor)&
      (tc.method=='Multitaper')&(tc['shape']=='Boatwright')]
    f=q.frequency_Hz.to_numpy();y=q.observed.to_numpy();ok=q.usable.to_numpy()
    ax.loglog(f[ok],10**y[ok],color=color,lw=1.35,label=label)
    ax.loglog(f,10**q.predicted.to_numpy(),color=color,lw=1,ls='--')
ax.set(xlabel='Frequency (Hz)',ylabel=r'Corrected amplitude (m$^2$ s)',xlim=(.5,20))
ax.legend(frameon=False);ax.grid(color='.9');ax.set_title('100 km; 10.1 s S window',loc='left')
labels=['Combined','Velocity','HG','110 km']
for j,(sensor,radius) in enumerate([('ALL',100),('HH_EL',100),('HG',100),('ALL',110)]):
    for k,(method,marker) in enumerate([('Multitaper','o'),('Direct Fourier','s')]):
        for duration,fill in [(10.1,True),(20.1,False)]:
            q=ts[(ts.sensor_set==sensor)&(ts.radius_km==radius)&(ts.window_s==duration)&
              (ts.method==method)&(ts['shape']=='Boatwright')].iloc[0]
            x=j+(-.12 if k==0 else .12)+(-.025 if fill else .025)
            bx.scatter(x,q.Mw,s=50,marker=marker,edgecolor=BLUE if k==0 else ORANGE,
              facecolor=(BLUE if k==0 else ORANGE) if fill else 'white',lw=1.1)
bx.set(xticks=range(4),xticklabels=labels,ylabel=r'$M_w$',ylim=(3.25,3.5))
bx.grid(axis='y',color='.9')
bx.legend(handles=[Line2D([],[],marker='o',color=BLUE,ls='',label='Multitaper'),
 Line2D([],[],marker='s',color=ORANGE,ls='',label='Fourier'),
 Line2D([],[],marker='o',color='.3',ls='',label='10.1 s'),
 Line2D([],[],marker='o',mfc='white',color='.3',ls='',label='20.1 s')],frameon=False,loc='lower right')
bx.set_title(f"Preferred Mw {t['preferred']['Mw']:.2f}; {t['preferred']['n_stations']} sites",loc='left')
finish(fig,'01_Sangju_magnitude',
 'One triplet per site; HH, then EL, then HG among usable records. Dashed: generalized Boatwright fit. Depth 19.76 km.\n'
 'Kim & Kim (1983), Vp/Vs = 1.73; source Vs = 3.688 km/s. C = 9.05. Kim et al. (2011) is compared separately.')

fig,(ax,bx)=canvas('Sangju: independently measured local magnitude')
for sensor,q in targetsta.groupby('sensor'):
    ax.scatter(q.distance_km,q.value,s=32,color=COL[sensor],alpha=.8,label=f'{sensor}: {len(q)}')
    bx.hist(q.value,bins=np.arange(2.3,4.91,.1),histtype='step',color=COL[sensor],lw=1.5,label=sensor)
for x in [ax,bx]:
    x.grid(color='.92');x.legend(frameon=False)
ax.axhline(t['local_magnitude']['ML_Sheen'],color='black',lw=1.1)
ax.axhline(3.6,color='.5',ls=':',lw=1)
ax.set(xlabel='Epicentral distance (km)',ylabel='Station ML',xlim=(0,102))
bx.axvline(t['local_magnitude']['ML_Sheen'],color='black',lw=1.1,label='Median')
bx.axvline(3.6,color='.5',ls=':',lw=1)
bx.set(xlabel='Station ML',ylabel='Sites')
ax.set_title(f"Combined ML {t['local_magnitude']['ML_Sheen']:.2f}; {len(targetsta)} sites",loc='left')
bx.set_title(f"Velocity ML {t['local_magnitude']['ML_velocity']:.2f}; HG ML {t['local_magnitude']['ML_HG']:.2f}",loc='left')
finish(fig,'02_Sangju_local_magnitude',
 'Revised Wood–Anderson; geometric mean of N/E zero-to-peak amplitudes; SNR ≥3 on both components; 10–100 km.\n'
 'Solid: event median. Dotted: reported ML 3.6. Published Sheen et al. attenuation, with station terms set to zero. No magnitude anchoring.')

fig,axes=canvas('Sangju: Wood–Anderson records',3,2,size=(11,9.5))
b=np.load(R/'spectra/Sangju2026_WA_examples.npz')
examples=pd.read_csv(R/'tables/Sangju2026_WA_examples.csv')
for ax,row in zip(axes.flat,examples.itertuples()):
    ax.plot(b[f't{row.index}'],b[f'y{row.index}'],c=COL[row.sensor],lw=.65)
    ax.axvspan(row.signal_start,row.signal_end,color=COL[row.sensor],alpha=.12)
    ax.axvspan(row.noise_start,row.noise_end,color='.5',alpha=.13)
    ax.set_title(f'{row.station} {row.sensor}N; {row.distance_km:.1f} km; station ML {row.station_ML:.2f}',loc='left',fontsize=10)
    ax.set(xlabel='Time after origin (s)',ylabel='WA displacement (mm)');ax.grid(color='.94')
finish(fig,'03_Sangju_WA_waveforms',
 'Physical Wood–Anderson amplitudes, with a separate vertical scale for each trace. Shading: signal (colored), noise (gray).\n'
 'The six-pole 0.5–10 Hz filter is causal, as in the established ML recipe. Response removal uses no displacement water-level floor.')

fig,(ax,bx)=canvas('Moment magnitude and local magnitude')
fig.subplots_adjust(right=.91)
norm=Normalize(0,30)
for axis,key,label,reglabel in [(ax,'ML_Sheen','Measured ML','Independent ML'),(bx,'ML_KMA','KMA ML','KMA ML, same events')]:
    sc=axis.scatter(g[key],g.Mw,c=g.z_central,cmap='viridis',norm=norm,s=24,edgecolors='none',alpha=.85)
    rr=reg[reg.selection==reglabel].iloc[0];x=np.linspace(g[key].min(),g[key].max(),150)
    axis.plot(x,rr.intercept_standard+rr.slope*x,color='black',lw=1.2)
    axis.plot([1.3,6],[1.3,6],color='.6',ls=':',lw=.8)
    targetx=t['local_magnitude']['ML_Sheen'] if key=='ML_Sheen' else 3.6
    axis.scatter(targetx,t['preferred']['Mw'],marker='*',s=150,color='#a63027',edgecolor='white',lw=.5,zorder=6)
    axis.set(xlabel=label,ylabel='Spectral Mw',xlim=(1.5,max(5.5,g[key].max()+.1)),ylim=(1.4,5.8))
    axis.set_title(f'Mw = {rr.intercept_standard:.3f} + {rr.slope:.3f} ML\nRMS {rr.RMSE:.3f}',loc='left')
    axis.grid(color='.92')
fig.colorbar(sc,ax=[ax,bx],location='right',fraction=.025,pad=.025,label='Depth (km)')
finish(fig,'04_Mw_ML_comparison',
 f'{len(g)} quality-A events; {g.family_block.nunique()} sequence blocks. Lines: family-balanced OLS. Red star: Sangju, held out.\n'
 'Shared response and site errors, and reliable-depth selection, limit interpretation of these descriptive fits.')

fig,axes=canvas('Effect of adding accelerometers',2,2,size=(11,9))
q=change[(change.Mw_quality=='A')&change.primary_geography]
plots=[(axes[0,0],q.n_HG,q.Mw_HG_increment,'HG sites in combined Mw','Combined Mw − velocity Mw'),
 (axes[0,1],change.ML_n_HG,change.ML_HG_increment,'HG sites in combined ML','Combined ML − velocity ML'),
 (axes[1,0],q.Mw_previous,q.Mw_processing_change,'Previous velocity Mw','Reprocessed velocity Mw − previous'),
 (axes[1,1],change.ML_Sheen_previous,change.ML_processing_change,'Previous velocity ML','Reprocessed velocity ML − previous')]
for axis,x,y,xlab,ylab in plots:
    ok=np.isfinite(x)&np.isfinite(y);axis.scatter(x[ok],y[ok],s=16,c=BLUE,alpha=.55,edgecolors='none')
    axis.axhline(0,c='.4',lw=.8);axis.set(xlabel=xlab,ylabel=ylab);axis.grid(color='.92')
    axis.text(.04,.95,f'n = {ok.sum()}; median {np.median(y[ok]):+.3f}',transform=axis.transAxes,va='top',fontsize=9)
finish(fig,'05_HG_effect',
 'Top: adding HG sites with common processing. Bottom: changing velocity processing, metadata and QA relative to the previous release.\n'
 'Both previous and current Mw use the same source Vs(z) coefficient. Finite pairs; current Mw A. No ML-based correction.')

fig,(ax,bx)=canvas('Coverage and depth')
coast(ax);ax.scatter(d.lon,d.lat,s=13,facecolor='none',edgecolor='.75',lw=.5)
sc=ax.scatter(g.lon,g.lat,c=g.z_central,cmap='viridis',norm=norm,s=24,edgecolors='none')
ax.scatter(127.96,36.52,c='#a63027',marker='*',s=120,edgecolor='white',lw=.5)
ax.annotate('Sangju',(127.96,36.52),xytext=(-27,22),textcoords='offset points',ha='right',fontsize=9,arrowprops=dict(arrowstyle='-',lw=.5))
ax.set(xlim=(124.5,131.1),ylim=(32.8,38.9),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
ax.set_aspect(1/np.cos(np.deg2rad(36)))
rr=reg[reg.selection=='Independent ML + depth'].iloc[0]
res=g.Mw-(rr.intercept_at_x3+rr.slope*(g.ML_Sheen-3))
bx.errorbar(g.z_central,res,xerr=[np.maximum(g.z_central-g.lower,0),np.maximum(g.upper-g.z_central,0)],fmt='none',ecolor='.8',alpha=.55,elinewidth=.6)
bx.scatter(g.z_central,res,c=BLUE,s=20,edgecolors='none',alpha=.75)
z=np.linspace(g.z_central.min(),g.z_central.max(),100);bx.plot(z,rr.depth_coefficient*(z-10)/10,c='black',lw=1.2)
bx.axvline(16.89,c='.45',ls=':',lw=1);bx.axhline(0,c='.6',lw=.7)
bx.set(xlabel='Depth (km)',ylabel='Mw after removing fitted ML term');bx.grid(color='.92')
fig.colorbar(sc,ax=ax,location='bottom',fraction=.04,pad=.14,label='Depth (km)')
finish(fig,'06_map_and_depth',
 'Gray: all depth-qualified candidates. Color: paired quality-A events. Depth bars show project uncertainty envelopes, not calibrated 95% CIs.\n'
 'Dotted: national D90 = 16.89 km, fixed from 10,233 reliable-depth events. The regression uses continuous depth; no hard deep-event cutoff.')

fig,(ax,bx)=canvas('Quality and station coverage')
bins=np.arange(2,6.01,.25);base=np.zeros(len(bins)-1)
for quality,color in [('A',BLUE),('B','#86b1ca'),('C','#b7b7b7'),('F','#e0e0e0')]:
    nn,_=np.histogram(d.loc[d.Mw_quality==quality,'ml'],bins)
    ax.bar((bins[:-1]+bins[1:])/2,nn,bottom=base,width=.23,color=color,label=f'{quality}: {(d.Mw_quality==quality).sum()}');base+=nn
ax.set(xlabel='KMA ML',ylabel='Events');ax.legend(frameon=False);ax.grid(axis='y',color='.93')
for sensor,color in COL.items():
    q=d[d.ML_quality=='A'];bx.scatter(q.ML_KMA,q['ML_n_'+sensor],s=15,color=color,alpha=.45,label=sensor)
bx.set(xlabel='KMA ML',ylabel='Sites in quality-A ML');bx.legend(frameon=False);bx.grid(color='.93')
finish(fig,'07_quality_and_coverage',
 'All 642 candidates remain in the catalog. A enters the primary Mw comparison; B is exploratory; C/F is excluded.\n'
 'Mw quality uses station count, plateau, bandwidth, residual and estimator/window/source-shape stability. ML quality A requires ≥5 sites and scatter ≤0.35.')

fig,(ax,bx)=canvas('Regression and source-velocity sensitivity',size=(11,8))
labels=['Independent ML','KMA ML, same events','Velocity-only ML','HG-only ML','Velocity-only Mw','G3 depths','40--100 km ML','Strict response history']
names=['Measured ML','KMA ML','Velocity ML','HG ML','Velocity Mw','G3 depth','40–100 km ML','Strict responses']
for i,(label,name) in enumerate(zip(labels,names)):
    q=reg[reg.selection==label]
    if len(q):
        rr=q.iloc[0];ax.errorbar(rr.slope,i,xerr=[[max(0,rr.slope-rr.slope_lo)],[max(0,rr.slope_hi-rr.slope)]],fmt='o',color=BLUE,ms=4,capsize=2)
ax.set(yticks=range(len(names)),yticklabels=names,xlabel='Mw–ML slope');ax.invert_yaxis();ax.grid(axis='x',color='.92')
for i,label in enumerate(['Independent ML + depth','Measured ML + depth, Vs Kim2011','Measured ML + depth, fixed Vs3500']):
    rr=reg[reg.selection==label].iloc[0]
    bx.errorbar(rr.depth_coefficient,i,xerr=[[rr.depth_coefficient-rr.depth_lo],[rr.depth_hi-rr.depth_coefficient]],fmt='o',color=ORANGE,capsize=2)
bx.set(yticks=range(3),yticklabels=['Kim & Kim 1983, 1.73','Kim et al. 2011','Fixed 3.50 km/s control'],xlabel='Depth coefficient (Mw / 10 km)')
bx.axvline(0,c='.5',lw=.8);bx.invert_yaxis();bx.grid(axis='x',color='.92')
fig.subplots_adjust(left=.17,wspace=.64)
finish(fig,'08_regression_controls',
 'Family-balanced OLS slopes. Bars: 95% sequence-block bootstrap intervals, conditional on measured magnitudes.\n'
 'The Vs(z) test changes the source Vs³ factor only. Depth trends remain sensitive to physical assumptions.')

(R/'tables/figure_manifest.json').write_text(json.dumps(figures,indent=2)+'\n')
# Numerical summary of fair paired differences.
stats={}
for key in ['Mw_HG_increment','ML_HG_increment','Mw_processing_change','ML_processing_change']:
    q=change[(change.Mw_quality=='A')&change.primary_geography] if key.startswith('Mw') else change[change.ML_quality=='A']
    v=q[key].dropna().to_numpy();stats[key]=dict(n=len(v),median=float(np.median(v)),q16=float(np.quantile(v,.16)),q84=float(np.quantile(v,.84)))
stats.update(n_Mw_A_with_HG=int(((d.Mw_quality=='A')&(d.n_HG>0)).sum()),
 n_ML_A_with_HG=int(((d.ML_quality=='A')&(d.ML_n_HG>0)).sum()),
 previous_Mw_A=int((old.Mw_quality=='A').sum()),current_Mw_A=int((d.Mw_quality=='A').sum()))
(R/'tables/HG_update_summary.json').write_text(json.dumps(stats,indent=2)+'\n')

rr=reg[reg.selection=='Independent ML'].iloc[0];rk=reg[reg.selection=='KMA ML, same events'].iloc[0]
ml=t['local_magnitude'];pref=t['preferred'];count=mw.Mw_quality.value_counts().to_dict()
velmw=ts[(ts.radius_km==100)&(ts.window_s==10.1)&(ts.sensor_set=='HH_EL')&(ts.method=='Multitaper')&(ts['shape']=='Boatwright')].iloc[0].Mw
hgmw=ts[(ts.radius_km==100)&(ts.window_s==10.1)&(ts.sensor_set=='HG')&(ts.method=='Multitaper')&(ts['shape']=='Boatwright')].iloc[0].Mw
paragraphs=[
 ('Sangju and the national Mw–ML study',[
  'Updated 19 November 2026. Multitaper/Boatwright Mw now uses source-depth Vs from the Haenam Kim & Kim (1983) implementation, Vp/Vs=1.73. Kim et al. (2011) is reported alongside it. HH, EL and HG each contribute once per site, in that priority among usable records. Selection never uses agreement with KMA ML or Mw.',
  f"Sangju: Mw {pref['Mw']:.3f}, M0 {pref['M0_Nm']:.3e} N m, {pref['n_stations']} spectral sites within 100 km. ML {ml['ML_Sheen']:.3f}, {ml['ML_Sheen_n']} horizontal sites ({ml['ML_n_HH']} HH, {ml['ML_n_EL']} EL, {ml['ML_n_HG']} HG). The phase-location depth is 19.76 km, with project interval 17.68–21.31 km (G3). Reported ML is 3.6.",
  f"Reprocessed velocity-only Sangju controls: Mw {velmw:.3f}; ML {ml['ML_velocity']:.3f}. HG-only: Mw {hgmw:.3f}; ML {ml['ML_HG']:.3f}. HG increases coverage substantially without requiring a common amplitude offset.",
  'KS.BURB is excluded from both Sangju magnitudes because its co-located horizontal spectra disagree. The exclusion is specific to this event.',
  f"The historical cohort is unchanged: 642 KMA ML≥2 events with G2/G3 depth assessments, 2010–October 2026. Mw quality counts: {count.get('A',0)} A, {count.get('B',0)} B, {count.get('C',0)} C and {count.get('F',0)} F. {summary['n_measured']} have measured ML and {summary['n_ML_A']} have quality-A ML. The paired primary sample contains {len(g)} events in {g.family_block.nunique()} sequence blocks; Sangju is held out."
 ]),
 ('Magnitude comparison and the HG contribution',[
  f"For the {len(g)} paired quality-A events, the family-balanced OLS fit of Mw on ML is Mw = {rr.intercept_standard:.4f} + {rr.slope:.4f} measured ML. Slope interval {rr.slope_lo:.4f}–{rr.slope_hi:.4f}; RMS {rr.RMSE:.4f}. On exactly the same events, KMA ML gives Mw = {rk.intercept_standard:.4f} + {rk.slope:.4f} KMA ML; RMS {rk.RMSE:.4f}.",
  f"Leave-one-sequence-out RMS: {summary['heldout_RMSE']['ML_Sheen']:.4f} for measured ML and {summary['heldout_RMSE']['ML_KMA']:.4f} for KMA ML. Both magnitude axes have errors; Deming fits for assumed variance ratios are supplied separately. These relations are descriptive and conditional on the selected sample.",
  f"Current quality-A Mw uses HG at {stats['n_Mw_A_with_HG']} events; quality-A ML uses HG at {stats['n_ML_A_with_HG']} events. With processing held fixed, adding HG changes paired Mw by median {stats['Mw_HG_increment']['median']:+.3f} and ML by {stats['ML_HG_increment']['median']:+.3f}. These medians include events with no added HG; the event table records the actual sensor counts and differences.",
  'The before/after table separates HG inclusion from response, metadata and timing QA. Both generations use the same source-depth Vs coefficient in that comparison. Original fixed-3.50 km/s values are explicitly named controls.',
  'National D90 remains 16.89 km, determined from 10,233 reliable-depth events before magnitude selection. Depth is fitted continuously. Source-Vs sensitivity changes only the beta-cubed moment factor; attenuation and other path assumptions remain fixed. A conditional depth coefficient cannot establish a physical source-depth trend.'
 ]),
 ('Processing in physical units',[
  '1. Read immutable count data. Match network, station, channel and response epoch. Prefer the audited merged inventory, then a matching dated NECIS response, then the historical master. Verify velocity or acceleration input units, station coordinates and horizontal orientation. Flag historical open epochs used after the inventory issue date.',
  '2. Demean and linearly detrend the full trace. Apply one 1% taper, capped at 5 s. Remove the response to displacement with water_level=None and a cosine prefilter of 0.1, 0.2, 25 and 40 Hz, capped by the sampling rate. The taper and lower prefilter limit low-frequency integration noise. No amplitude correction is fitted between sensors.',
  '3. Require observed P/S picks with probability ≥0.3 and physically plausible P travel time. Reject gaps, probable clipping and known catalog overlap. Extract 10.1 s and 20.1 s S windows, starting S−0.1 s; equal-duration noise windows end P−1 s and move earlier only for overlap or coverage. Sum the three component powers; amplitude SNR must be ≥3.',
  '4. Prieto Python multitaper MTSpec, NW=4, K=7, returns a one-sided PSD. Convert to Fourier-amplitude units with sqrt(N*dt*P1/2); N*dt is the actual sampled window length. This does not apply an additional sqrt(2) gain to a PSD. Compare with the direct dt-scaled FFT on the same records.',
  '5. Correct geometric spreading with hypocentral distance r and attenuation exp(pi*f*tS/Q(f)), Q(f)=229.2*f^0.73. Average station log amplitudes. Fit the generalized Boatwright spectrum over usable 0.5–20 Hz bins; require at least one decade. Brune, direct FFT, common station/sensor windows, aperture and sensor selections are sensitivity checks.',
  '6. M0 = 4*pi*rho*beta(z)^3*Omega_r/(2*0.63), rho=2800 kg/m³. Primary beta(z): Kim & Kim (1983), Vp/Vs=1.73; comparison: published Kim et al. (2011), Table 2. Both use each event’s national z_central. Mw=(log10(M0[N m])−9.05)/1.5. These are source-coefficient choices; empirical spreading/Q and event depths stay fixed.'
 ]),
 ('ML, validation and limits',[
  'Local magnitude follows the established Haenam/Buan recipe after the common corrected displacement: causal six-pole 0.5–10 Hz filtering and the revised Wood–Anderson response. WA poles are −6.283±4.7124i, two displacement zeros and magnification 2080. Horizontal amplitude is the geometric mean of N/E zero-to-peak amplitudes in millimeters.',
  'The ML signal window is origin + [Rhyp/4, Rhyp/2]; noise is observed P−6 to P−1 s. Both horizontal components require peak SNR≥3. For 10≤Repi≤100 km: ML=log10(Amm)+0.5869*log10(Repi/100)+0.001680*(Repi−100)+3; station terms are zero. Event ML is the median of at least 3 sites; quality A requires at least 5 and robust scatter≤0.35.',
  'The former DISP water_level=60 setting can suppress HG amplitudes because an accelerometer displacement response grows as frequency squared. Removing that floor requires an explicit highpass regularization. Synthetic response-forward/recovery tests pass for real HH and HG responses. The five agreeing Sangju co-located pairs remain within 20% at 0.7–1.2 Hz under the adopted processing; BURB remains discrepant and is quarantined.',
  'Station-resampling intervals hold the chosen model fixed. Separate depth-envelope endpoints propagate distance and source Vs(z), including layer changes; they are not calibrated 95% intervals. Radiation, site, attenuation, density and response-history uncertainties remain. SNR/depth/aperture selections limit completeness.',
  'The sensor inclusion in this release concerns spectral Mw and Wood–Anderson ML. Low-frequency moment-tensor inversion requires its own signal and response assessment. The paired validation appendix documents why short-window source spectra and 0.05–0.2 Hz waveform inversion have different usable station sets.',
  'Current catalogs, per-event count hashes, exact response sources, configuration signatures, station logs, spectral arrays and the full 642-event atlas accompany this report. Adopted external location and Haenam catalogs were read only.'
 ])]

# The new external comparison diagnoses selection without changing magnitudes.
hong=R/'tables/Hong_regression_comparison.csv'
if hong.exists():
    ht=pd.read_csv(hong)
    hs=ht[(ht.x=='ML_KMA')&(ht.y=='mw_spectra')&(ht.method=='ODR')&(ht.weighting=='event')].set_index('selection')
    paragraphs[1][1].insert(3,f"The Hong and Sheen catalog supplies a direct selection check: its unchanged spectral Mw gives an ODR ML–Mw slope of {hs.loc['All exact matches','slope']:.3f} for all exact matches, falling to {hs.loc['Matched Mw A','slope']:.3f} after our Mw-A filter. SNR, plateau and estimator-agreement gates can therefore affect the inferred population relation even without an explicit ML-agreement gate. ODR and OLS comparisons are in the separate Hong regression report; this report retains its labeled OLS predictive/descriptive fits.")

def narrative(pdf,selected):
    for title,paras in selected:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.085,.935,title,fontsize=16,va='top')
        y=.875
        for para in paras:
            wrapped=textwrap.fill(para,width=94,break_long_words=False,break_on_hyphens=False)
            fig.text(.085,y,wrapped,fontsize=10,va='top',linespacing=1.55)
            y-=.0189*(wrapped.count('\n')+1)+.028
        if y<.05:raise RuntimeError(f'Narrative overflows: {title}: {y}')
        fig.text(.085,.045,'19 November 2026  |  HH / EL / HG  |  C = 9.05',fontsize=8.5,color='.4')
        pdf.savefig(fig);plt.close(fig)

body=R/'reports/.current_narrative.pdf'
with PdfPages(body) as pdf:narrative(pdf,paragraphs)
writer=PdfWriter();writer.append(str(body))
for f in figures:writer.append(str(R/'figures'/(f['stem']+'.pdf')))
writer.append(str(R/'figures/16_source_velocity_models.pdf'))
appendix=R/'hh_hg_comparison/reports/HH_HG_paired_comparison.pdf'
writer.append(str(appendix));writer.write(str(R/'reports/Korean_Mw_ML_study.pdf'));body.unlink()
with PdfPages(body) as pdf:narrative(pdf,[paragraphs[0],paragraphs[1],paragraphs[3]])
writer=PdfWriter();writer.append(str(body))
for f in figures:
    if f['stem'][:2] in ['02','03','04','05','07','08']:writer.append(str(R/'figures'/(f['stem']+'.pdf')))
writer.append(str(R/'figures/16_source_velocity_models.pdf'))
writer.write(str(R/'reports/Independent_ML_and_Mw_comparison.pdf'));body.unlink()
plain='\n\n'.join(title+'\n\n'+'\n\n'.join(p) for title,p in paragraphs)+'\n'
for name in ['Korean_Mw_ML_study.txt','Independent_ML_and_Mw_comparison.txt']:(R/'reports'/name).write_text(plain)
print(json.dumps(dict(Sangju_Mw=pref['Mw'],Sangju_ML=ml['ML_Sheen'],paired_events=len(g),HG_effect=stats),indent=2))
