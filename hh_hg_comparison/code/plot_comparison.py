"""Publication figures for the directly observed HH/HG comparisons."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import read,read_inventory,UTCDateTime
from compare_pairs import response_trace,window
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'legend.fontsize':8,'grid.color':'.91','grid.linewidth':.5,'pdf.fonttype':42,'savefig.dpi':260})
HH='#176da0';HG='#bf591b';RED='#a42c2c'
PAIRS=json.loads((ROOT/'inputs/pairs.json').read_text())
BANDS=pd.read_csv(ROOT/'tables/spectral_comparison.csv',dtype={'event_id':str})
COLORS=['#176da0','#bf591b','#359777','#8b6ca8','#686868',RED]
manifest=[]


def finish(fig,name,caption):
    fig.text(.075,.023,caption,va='bottom',fontsize=8.4,color='.25',linespacing=1.4)
    for ext in ['pdf','png']:fig.savefig(ROOT/'figures'/(name+'.'+ext))
    plt.close(fig);manifest.append(dict(stem=name,caption=caption))


def pair(eid,sta):return next(p for p in PAIRS if p['event_id']==eid and p['station']==sta)


def load(p):return np.load(ROOT/'spectra'/(p['event_id']+'_'+p['station']+'.npz'))


def spectra(ax,p,tag,limits):
    b=load(p);f=b[tag+'_frequency_Hz'];use=(f>=limits[0])&(f<=limits[1])
    for sensor,color in [('HH',HH),('HG',HG)]:
        ax.loglog(f[use],b[tag+'_'+sensor+'_ASD'][use],c=color,lw=1.2)
        ax.loglog(f[use],b[tag+'_'+sensor+'_noise_ASD'][use],c=color,lw=.85,ls='--',alpha=.8)
    ax.set_xlim(*limits);ax.grid(which='major')
    ax.set_title(f"KS.{p['station']}   {p['epicentral_km']:.0f} km",loc='left')
    ax.set_ylabel(r'Displacement ASD (m/$\sqrt{Hz}$)')


def legend(fig,y):
    fig.legend(handles=[Line2D([],[],c=HH,label='HH signal'),Line2D([],[],c=HG,label='HG signal'),
        Line2D([],[],c=HH,ls='--',label='HH noise'),Line2D([],[],c=HG,ls='--',label='HG noise')],
        loc='upper center',bbox_to_anchor=(.5,y),ncol=4,frameon=False)


# The earthquake is the same, and every panel uses physically calibrated units.
stations=['SAJB','OKCB','CHJ3','KESB','JECB','BURB']
fig,axes=plt.subplots(3,2,figsize=(11,10.2),sharex=True)
fig.subplots_adjust(left=.09,right=.97,bottom=.13,top=.865,hspace=.38,wspace=.27)
fig.suptitle('Sangju: co-located HH and HG spectra',fontsize=15,y=.97);legend(fig,.935)
for ax,sta in zip(axes.ravel(),stations):spectra(ax,pair('20261117141017',sta),'short',(.5,20))
for ax in axes[-1]:ax.set_xlabel('Frequency (Hz)')
finish(fig,'01_Sangju_paired_spectra',
    '10.1 s S windows and equal pre-P noise; three-component power sums; same UTC sample grid and dated instrument responses.\n'
    'Response removed to acceleration before conversion to displacement-equivalent ASD. No amplitude matching or source/path correction.')

fig,axes=plt.subplots(4,2,figsize=(11,11.8),sharex=True)
fig.subplots_adjust(left=.09,right=.97,bottom=.12,top=.855,hspace=.48,wspace=.27)
fig.suptitle('Haenam: strong and small events at the same sites',fontsize=15,y=.975);legend(fig,.945)
fig.text(.31,.89,'22 August 2026 · reported ML 3.1',ha='center',fontsize=11)
fig.text(.77,.89,'25 August 2026 · reported ML 1.3',ha='center',fontsize=11)
for row,sta in enumerate(['MOPB','HANB','JAEB','BGDB']):
    for col,eid in enumerate(['20260822214301','20260825002354']):
        spectra(axes[row,col],pair(eid,sta),'short',(.5,20))
for ax in axes[-1]:ax.set_xlabel('Frequency (Hz)')
finish(fig,'02_Haenam_paired_spectra',
    'Same stations, response epochs, processing and 10.1 s windows. Dashed lines show measured pre-P noise.\n'
    'Weak-event usability changes with frequency and station; a large peak or a high coherence alone does not establish a resolved source plateau.')

fig,axes=plt.subplots(3,2,figsize=(11,10.2),sharex=True)
fig.subplots_adjust(left=.09,right=.97,bottom=.13,top=.865,hspace=.4,wspace=.27)
fig.suptitle('Sangju: low-frequency signal and noise',fontsize=15,y=.97);legend(fig,.935)
for ax,sta in zip(axes.ravel(),stations):
    spectra(ax,pair('20261117141017',sta),'long',(.04,.5))
    ax.axvspan(.05,.2,color='.6',alpha=.10,lw=0)
for ax in axes[-1]:ax.set_xlabel('Frequency (Hz)')
finish(fig,'03_Sangju_low_frequency',
    '80 s full-wave windows and equal pre-P noise; NW=2, K=3. Shading marks 0.05–0.2 Hz.\n'
    'The lowest frequency has four cycles per window. These spectra assess relative sensor performance; they are not new moment-tensor inversions.')

fig,axes=plt.subplots(2,2,figsize=(11.4,9.0))
fig.subplots_adjust(left=.085,right=.97,bottom=.15,top=.88,hspace=.44,wspace=.25)
fig.suptitle('Amplitude agreement and waveform coherence',fontsize=15,y=.97)
for sta,color in zip(stations,COLORS):
    b=load(pair('20261117141017',sta));f=b['short_frequency_Hz'];m=(f>=.5)&(f<=20)
    use=m&b['short_both_SNR3']
    y=np.where(b['short_both_SNR3'],b['short_amplitude_ratio'],np.nan)
    axes[0,0].semilogx(f[m],y[m],c=color,lw=1.,label=sta)
    axes[0,1].semilogx(f[m],b['short_coherence_mean'][m],c=color,lw=1.,label=sta)
axes[0,0].set_title('Sangju: three-component amplitude ratio',loc='left')
axes[0,1].set_title('Sangju: component coherence',loc='left')
for sta,color in zip(['MOPB','HANB','JAEB','BGDB'],COLORS):
    for ax,eid in zip(axes[1],['20260822214301','20260825002354']):
        b=load(pair(eid,sta));f=b['short_frequency_Hz'];use=(f>=.5)&(f<=20)&b['short_both_SNR3']
        m=(f>=.5)&(f<=20);y=np.where(b['short_both_SNR3'],b['short_amplitude_ratio'],np.nan)
        ax.semilogx(f[m],y[m],c=color,lw=1.,label=sta)
axes[1,0].set_title('Haenam: reported ML 3.1',loc='left')
axes[1,1].set_title('Haenam: reported ML 1.3',loc='left')
for i,ax in enumerate(axes.ravel()):
    ax.set_xlim(.5,20);ax.set_xlabel('Frequency (Hz)');ax.grid()
    ax.axvspan(.7,1.2,color='.65',alpha=.12,lw=0)
    if i==1:
        ax.set(ylim=(0,1.04),ylabel='Magnitude-squared coherence')
        ax.axhline(.8,color='.55',lw=.7,ls=':')
    else:
        ax.axhline(1,color='.45',lw=.8,ls=':');ax.set(ylabel='HG / HH amplitude',yscale='log',ylim=(.05,5.))
axes[0,0].legend(frameon=False,ncol=3,loc='upper left',fontsize=8)
axes[1,0].legend(frameon=False,ncol=2,loc='upper left',fontsize=8)
finish(fig,'04_amplitude_and_coherence',
    'Ratios use noise-subtracted three-component power where both amplitude SNRs ≥3; missing curves indicate insufficient shared SNR.\n'
    'Coherence is a power-weighted mean of matching component coherences. Shading marks 0.7–1.2 Hz; no fitted gain or polarity changes are applied.')

water=pd.read_csv(ROOT/'tables/waterlevel_transfer.csv',dtype={'event_id':str})
validation=json.loads((ROOT/'tables/processing_validation.json').read_text())
known=pd.DataFrame(validation['known_acceleration_controls'])
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,7.4))
fig.subplots_adjust(left=.09,right=.97,bottom=.23,top=.85,wspace=.3)
fig.suptitle('A displacement water level can attenuate accelerometer data',fontsize=14,y=.96)
f=np.geomspace(.04,20,350);expected=np.minimum(1.,f*f/(.001*50**2))
ax.loglog(f,expected,c=HG,lw=1.4,label='60 dB water level, analytic')
ax.loglog(known.frequency_Hz,known.recovered_WL60/known.true_displacement_m,'o',c=HG,ms=5,label='Recovered test amplitudes')
ax.loglog(known.frequency_Hz,known.recovered_no_waterlevel/known.true_displacement_m,'s',c=HH,ms=5,label='Prefilter, no water level')
ax.set_title('Known displacement, flat acceleration response',loc='left',fontsize=10)
for sta,color in [('SAJB',HH),('CHJ3',HG),('BURB',RED)]:
    q=water[(water.event_id=='20261117141017')&(water.station==sta)&(water.sensor=='HG')]
    bx.loglog(q.frequency_Hz,q.WL60_to_noWL,c=color,lw=1.2,label='HG '+sta)
q=water[(water.event_id=='20261117141017')&(water.station=='SAJB')&(water.sensor=='HH')]
bx.loglog(q.frequency_Hz,q.WL60_to_noWL,c='.4',ls='--',lw=1.1,label='HH SAJB')
bx.set_title('Actual Sangju inverse-response attenuation',loc='left',fontsize=10)
for xx in [ax,bx]:
    xx.axvspan(.7,1.2,color='.65',alpha=.12,lw=0);xx.set(xlim=(.04,20),ylim=(1e-4,1.3),xlabel='Frequency (Hz)',ylabel='Recovered / unattenuated amplitude')
    xx.grid();xx.legend(frameon=False,fontsize=8,loc='lower right')
finish(fig,'05_waterlevel_control',
    'The control uses 100 sps and a known displacement amplitude. The right panel is computed directly from each full instrument response.\n'
    'A 60 dB floor is relative to the largest displacement-response gain. Its frequency dependence differs for velocity and acceleration sensors.\n'
    'This demonstrates a processing effect; it does not by itself quantify the bias of any previously fitted earthquake magnitude.')

# A phase-sensitive check complements power spectra and reveals orientation issues.
wavecases=[('20261117141017','CHJ3'),('20261117141017','BURB'),('20260822214301','HANB')]
fig,axes=plt.subplots(3,3,figsize=(11.5,8.8),sharex=True)
fig.subplots_adjust(left=.08,right=.97,bottom=.15,top=.865,hspace=.46,wspace=.3)
fig.suptitle('Paired waveforms in physical units',fontsize=15,y=.97)
fig.legend(handles=[Line2D([],[],c=HH,label='HH'),Line2D([],[],c=HG,ls='--',label='HG')],
           loc='upper center',bbox_to_anchor=(.5,.937),ncol=2,frameon=False)
for row,(eid,sta) in enumerate(wavecases):
    p=pair(eid,sta);s=UTCDateTime(p['S_utc'])
    for col,comp in enumerate('ZNE'):
        ax=axes[row,col]
        for sensor,color,style in [('HH',HH,'-'),('HG',HG,'--')]:
            tr=read(p['traces'][sensor+comp])[0]
            resp=read_inventory(p['responses'][sensor+comp]).get_response(tr.id,UTCDateTime(p['origin_utc']))
            w,_=response_trace(tr,resp)
            w.filter('bandpass',freqmin=.7,freqmax=3.,corners=3,zerophase=True)
            data=window(w,s-1,10.,100.)*1000
            ax.plot(np.arange(len(data))/100-1,data,c=color,ls=style,lw=1.)
        ax.set_title(f'{p["name"]} {sta} · {comp}',loc='left',fontsize=9)
        ax.set_ylabel('mm/s²');ax.grid();ax.axvline(0,c='.6',lw=.6,ls=':')
for ax in axes[-1]:ax.set_xlabel('Time after S (s)')
finish(fig,'06_paired_waveforms',
    '0.7–3 Hz zero-phase filter after response correction; same UTC grid and physical amplitude units, with no fitted rescaling.\n'
    'CHJ3 closely agrees. BURB has an amplitude discrepancy; HANB’s horizontal phase differences warrant an orientation check.')

(ROOT/'tables/figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Wrote',len(manifest),'scientific figure pages')
