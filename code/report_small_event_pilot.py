"""Small-earthquake feasibility report; retain all trials and measurement failures."""
from pathlib import Path
import hashlib
import html
import json
import sys
import warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import fitz

R=Path(__file__).resolve().parents[1];P=R/'small_events';T=P/'tables';F=P/'figures'
D=pd.read_csv(T/'pilot_Mw_ML_catalog.csv',dtype={'event_id':str})
SHORT=pd.read_csv(T/'short_window_Mw_ML_catalog.csv',dtype={'event_id':str})
NW2=pd.read_csv(T/'nw2_window_Mw_ML_catalog.csv',dtype={'event_id':str})
S=json.loads((T/'pilot_summary.json').read_text())
S4=json.loads((T/'short_window_summary.json').read_text())
S2=json.loads((T/'nw2_window_summary.json').read_text())
OLD=pd.read_csv(R/'tables/Mw_independent_ML_primary.csv',dtype={'event_id':str})
FIT=pd.read_csv(R/'tables/independent_ML_regressions.csv')
REFERENCE=FIT[FIT.selection.eq('Independent ML')].iloc[0]
ABFIT=pd.read_csv(T/'exploratory_regressions_AB.csv')
small=D[D.pilot_role.eq('small')];qa=small[small.paired_A];qab=small[small.paired_A_or_B]
BLUE,RED,GREEN,ORANGE,GRAY='#176da0','#b32438','#298365','#bd641f','#91979b'
MAG=['0p5-0p9','1p0-1p4','1p5-1p9'];MAG_LABEL=['0.5–0.9','1.0–1.4','1.5–1.9']
plt.rcParams.update({'font.family':'Liberation Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':400})
FIGURES=[]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def grid(ax):ax.grid(color='.92',lw=.5)
def save(fig,stem,caption):
    fig.text(.065,.035,caption,fontsize=8,color='.25',linespacing=1.35,va='bottom')
    for ext in ['pdf','svg','png']:fig.savefig(F/f'{stem}.{ext}',dpi=400,facecolor='white')
    plt.close(fig);FIGURES.append(dict(stem=stem,caption=caption));print(stem,flush=True)


def coast(ax):
    points=[]
    def draw():
        if len(points)>1:
            p=np.asarray(points);ax.plot(p[:,0],p[:,1],color='.6',lw=.6,zorder=0)
    for line in (R/'inputs/coastline_low.gmt').read_text().splitlines():
        if line.startswith('>'):draw();points=[]
        elif line.strip():points.append(list(map(float,line.split()[:2])))
    draw()


def yield_data(frame,flag):
    s=frame[frame.pilot_role.eq('small')]
    return np.array([int(s[s.magnitude_stratum.eq(m)][flag].sum()) for m in MAG])


def sampling():
    fig=plt.figure(figsize=(11.7,9.3));gs=fig.add_gridspec(2,2,left=.085,right=.96,top=.87,bottom=.17,width_ratios=[.95,1.1],wspace=.30,hspace=.42)
    fig.suptitle('Smaller Korean earthquakes: a fixed feasibility pilot',fontsize=15,y=.96)
    fig.text(.085,.918,'162 events below catalog ML 2 · 18 ML 2.0–2.4 controls · every selected event had retained waveforms',fontsize=9,color='.3')
    ax=fig.add_subplot(gs[:,0]);coast(ax)
    ax.scatter(small.lon,small.lat,s=14,c='.72',alpha=.65,ec='white',lw=.2,label='Other small-event trials')
    ax.scatter(qa.lon,qa.lat,s=38,c=RED,ec='white',lw=.5,label='Both magnitudes grade A',zorder=5)
    c=D[D.pilot_role.eq('control')];ax.scatter(c.lon,c.lat,s=25,marker='s',fc='none',ec='.2',lw=.75,label='ML 2.0–2.4 controls')
    ax.set(xlim=(124.5,131.1),ylim=(32.8,38.9),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
    ax.set_aspect(1/np.cos(np.radians(36)))
    ax.set_title('a  Pilot coverage',loc='left');grid(ax);ax.legend(frameon=False,fontsize=7.2,loc='lower right')
    ax=fig.add_subplot(gs[0,1]);x=np.arange(3)
    for offset,flag,label,color in [(-.24,'Mw_A','Mw A',BLUE),(0,'paired_A','Both A',RED),(.24,'paired_A_or_B','Both A/B',GREEN)]:
        if flag=='Mw_A':counts=np.array([sum(small[small.magnitude_stratum.eq(m)].Mw_quality.eq('A')) for m in MAG])
        else:counts=yield_data(D,flag)
        bars=ax.bar(x+offset,counts,.23,color=color,label=label)
        ax.bar_label(bars,fontsize=8,padding=2)
    ax.set(xticks=x,xticklabels=MAG_LABEL,xlabel='Catalog ML',ylabel='Successful events',ylim=(0,22))
    ax.set_title('b  Same standards; 54 attempted in each bin',loc='left');grid(ax);ax.legend(frameon=False,ncol=3,fontsize=7.5)
    ax=fig.add_subplot(gs[1,1]);levels=['0-10km','10-16p89km','ge16p89km']
    matrix=np.array([[small[small.magnitude_stratum.eq(m)&small.depth_stratum.eq(z)].paired_A.mean() for m in MAG] for z in levels])
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=.35,aspect='auto')
    for i,z in enumerate(levels):
        for j,m in enumerate(MAG):
            q=small[small.magnitude_stratum.eq(m)&small.depth_stratum.eq(z)]
            ax.text(j,i,f'{int(q.paired_A.sum())}/{len(q)}',ha='center',va='center',fontsize=10)
    ax.set(xticks=np.arange(3),xticklabels=MAG_LABEL,yticks=np.arange(3),yticklabels=['<10 km','10–16.89 km','≥16.89 km'],xlabel='Catalog ML')
    ax.set_title('c  Both A, by magnitude and depth',loc='left')
    save(fig,'01_small_event_sampling',
         'Six randomly selected events per magnitude × depth × era cell; no replacement or selection by waveform amplitude or Mw–ML agreement.\n'
         'Era cells: 2010–2016, 2017–2021, 2022–Oct 2026. Fractions describe this balanced pilot, not the national detection rate.')


def scaling():
    fig,axs=plt.subplots(2,2,figsize=(11.7,9.4));fig.subplots_adjust(left=.09,right=.96,top=.87,bottom=.17,wspace=.29,hspace=.40)
    fig.suptitle('What is actually extended below catalog ML 2?',fontsize=15,y=.96)
    fig.text(.09,.918,'Independent Wood–Anderson ML is the scaling variable; catalog ML defines the pilot draw',fontsize=9,color='.3')
    x=np.linspace(.8,5.5,240);low=x<OLD.ML_Sheen.min();pred=REFERENCE.intercept_standard+REFERENCE.slope*x
    for ax,zoom in zip(axs[0],[False,True]):
        ax.scatter(OLD.ML_Sheen,OLD.Mw,s=12,c='.73',alpha=.5,ec='none',label='Existing paired A')
        b=qab[~qab.paired_A];ax.scatter(b.ML_Sheen,b.Mw,s=28,fc='none',ec=BLUE,lw=.8,label='Small pilot: A/B only')
        ax.scatter(qa.ML_Sheen,qa.Mw,s=37,c=RED,ec='white',lw=.5,label='Small pilot: both A',zorder=5)
        nw=NW2[NW2.pilot_role.eq('small')&NW2.paired_A]
        ax.scatter(nw.ML_Sheen,nw.Mw,s=36,marker='^',fc='none',ec=GREEN,lw=.9,label='NW2 candidate: both A',zorder=6)
        ax.plot(x[~low],pred[~low],c='.25',lw=1.2,label='Existing line')
        ax.plot(x[low],pred[low],c='.25',lw=1.2,ls='--')
        ext=ABFIT[ABFIT.selection.eq('Reference plus pilot')&ABFIT.model.eq('linear')&ABFIT.method.eq('family-balanced OLS')].iloc[0]
        ax.plot(x,ext.a_at_ML2+ext.b_at_ML2*(x-2),c=BLUE,lw=1,ls=':',label='A/B sensitivity line')
        ax.set(xlabel='Measured ML',ylabel='Mw',xlim=(1.15,2.65) if zoom else (.8,5.55),ylim=(1.4,2.9) if zoom else (1.2,5.5))
        grid(ax)
    axs[0,0].set_title('a  Existing study and the new trials',loc='left');axs[0,0].legend(frameon=False,fontsize=7.3,loc='upper left')
    axs[0,1].set_title('b  Low-magnitude detail',loc='left')
    ax=axs[1,0]
    ax.scatter(OLD.ML_Sheen,OLD.Mw-(REFERENCE.intercept_standard+REFERENCE.slope*OLD.ML_Sheen),s=10,c='.75',alpha=.5,ec='none')
    ax.scatter(qab.ML_Sheen,qab.Mw-(REFERENCE.intercept_standard+REFERENCE.slope*qab.ML_Sheen),s=30,fc='none',ec=BLUE,lw=.8)
    ax.scatter(qa.ML_Sheen,qa.Mw-(REFERENCE.intercept_standard+REFERENCE.slope*qa.ML_Sheen),s=34,c=RED,ec='white',lw=.4)
    ax.axhline(0,c='.4',ls=':',lw=.8);ax.set(xlabel='Measured ML',ylabel='Residual from the existing line',xlim=(1.15,5.5));grid(ax)
    ax.set_title('c  Retained pilot events and the old relation',loc='left')
    ax=axs[1,1]
    f=ABFIT[ABFIT.selection.eq('Reference plus pilot')&ABFIT.method.eq('family-balanced OLS')].set_index('model')
    bars=ax.bar(['Line','Quadratic','Knot at ML 2'],[f.loc[m,'CV_RMS_small'] for m in ['linear','quadratic','hinge2']],color=[BLUE,GREEN,ORANGE],width=.6)
    ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    ax.set(ylabel='Held-out RMS on small A/B events',ylim=(0,.17));grid(ax)
    ax.set_title('d  Family-held-out prediction: A/B sensitivity',loc='left')
    save(fig,'02_small_event_scaling',
         'Original processing gives seven paired-A events, measured ML 1.78–2.23; the NW2 window candidate reaches ML 1.48. No revised primary relation is adopted.\n'
         'The A/B sensitivity adds 20 events to the 357-event reference; bootstrap/CV resamples or withholds whole families. These controls do not remove SNR selection bias.')


def limitations():
    fig,axs=plt.subplots(2,2,figsize=(11.7,9.2));fig.subplots_adjust(left=.16,right=.96,top=.87,bottom=.17,wspace=.37,hspace=.43)
    fig.suptitle('The measurement limits of the unchanged pipeline',fontsize=15,y=.96)
    colors={'A':RED,'B':BLUE,'C':ORANGE,'F':'.73'}
    failed=pd.read_csv(T/'failure_reasons.csv').groupby('reason')['count'].sum().sort_values()
    labels={'too_few_usable_triplets':'Too few useful triplets','insufficient_usable_frequency_band':'Usable band too narrow','estimator_sensitivity':'Multitaper / Fourier mismatch',
            'window_sensitivity':'10 / 20 s mismatch','few_common_window_stations':'Too few common-window sites','few_stations':'Too few sites for A','plateau_extrapolated':'Plateau extrapolated',
            'poor_source_fit':'Poor source fit','source_shape_sensitivity':'Source-shape sensitivity'}
    ax=axs[0,0];ax.barh(np.arange(len(failed)),failed,color='.55');ax.set_yticks(np.arange(len(failed)),[labels.get(s,s) for s in failed.index],fontsize=7.5)
    ax.set(xlabel='Events failing gate');ax.set_title('a  Multiple failures may apply to an event',loc='left');grid(ax)
    for ax,col,title in [(axs[0,1],'n_stations_SNR','b  Useful sites'),(axs[1,0],'difference_Fourier','c  Spectral estimator agreement'),(axs[1,1],'window_difference_common','d  Common-station window agreement')]:
        for grade in ['F','C','B','A']:
            q=small[small.Mw_quality.eq(grade)]
            ax.scatter(q.ml,q[col],s=15,c=colors[grade],alpha=.65,ec='none',label=grade)
        ax.set(xlabel='Catalog ML',xlim=(.45,2.02));ax.set_title(title,loc='left');grid(ax)
        if col=='n_stations_SNR':
            ax.axhline(8,c='.35',ls='--',lw=.8);ax.axhline(5,c='.6',ls=':',lw=.8);ax.set_ylabel('SNR-passing sites');ax.legend(title='Mw grade',frameon=False,ncol=2,fontsize=7)
        else:
            ax.axhline(.10,c='.35',ls='--',lw=.8);ax.axhline(.15,c='.6',ls=':',lw=.8)
            ax.set_ylabel('|ΔMw|')
    save(fig,'03_small_event_limits',
         'Original 10.1/20.1 s, NW=4, seven tapers, 0.5–20 Hz; all HH/EL/HG inputs retained subject to the published response and SNR rules.\n'
         'Useful sites contribute ≥20 SNR-passing samples; the fitted stack must span ≥one decade. No threshold was relaxed to obtain a smaller-event relation.')


def window_controls():
    fig,axs=plt.subplots(2,2,figsize=(11.7,9.2));fig.subplots_adjust(left=.09,right=.96,top=.87,bottom=.17,wspace=.29,hspace=.43)
    fig.suptitle('Controlled window and taper experiments',fontsize=15,y=.96)
    schemes=[(D,'10/20 s · NW4',BLUE),(SHORT,'3/5 s · NW4',ORANGE),(NW2,'5/10 s · NW2',GREEN)]
    ax=axs[0,0];x=np.arange(3)
    for offset,(frame,label,color) in zip([-.24,0,.24],schemes):
        count=yield_data(frame,'paired_A');bars=ax.bar(x+offset,count,.23,label=label,color=color);ax.bar_label(bars,padding=2,fontsize=8)
    ax.set(xticks=x,xticklabels=MAG_LABEL,xlabel='Catalog ML',ylabel='Both A; 54 trials per bin');ax.set_title('a  Full pilot, including failures',loc='left');ax.legend(frameon=False,fontsize=7)
    ax=axs[0,1]
    for file,label,color in [('short_window_controls.csv','3/5 s · NW4',ORANGE),('nw2_window_controls.csv','5/10 s · NW2',GREEN)]:
        q=pd.read_csv(T/file,dtype={'event_id':str});valid=q.Mw_quality.eq('A');base=D.set_index('event_id')
        valid &= np.array([base.loc[eid,'Mw_quality']=='A' for eid in q.event_id])
        q=q[valid]
        ax.scatter(base.loc[q.event_id,'Mw'],q.Mw_short_minus_standard,s=30,c=color,alpha=.75,label=label)
    ax.axhline(0,c='.3',lw=.8);ax.axhline(.1,c='.7',ls=':',lw=.7);ax.axhline(-.1,c='.7',ls=':',lw=.7)
    ax.set(xlabel='Reference Mw of the ML 2–2.4 controls',ylabel='Candidate − reference Mw');ax.set_title('b  Controls passing Mw A in both methods',loc='left');ax.legend(frameon=False,fontsize=7.5)
    ax=axs[1,0]
    for frame,label,color in schemes[1:]:
        q=frame[frame.pilot_role.eq('small')&frame.Mw.notna()&frame.Mw_short_minus_standard.notna()]
        ax.scatter(q.ml,q.Mw_short_minus_standard,s=16,c=color,alpha=.5,label=label)
    ax.axhline(0,c='.3',lw=.8);ax.set(xlabel='Catalog ML',ylabel='Candidate − original Mw');ax.set_title('c  Where both methods provide a fit',loc='left');ax.legend(frameon=False,fontsize=7.5)
    ax=axs[1,1]
    pulse=pd.read_csv(T/'pulse_operator_controls.csv');arrays=[];ticks=[]
    for scheme,label in [('standard','10 s NW4'),('short_NW4','3 s NW4'),('short_NW2','5 s NW2')]:
        arrays.append(pulse[pulse.scheme.eq(scheme)&pulse.primary_window&pulse.method.eq('Multitaper')].delta_Mw.to_numpy());ticks.append(label)
    ax.boxplot(arrays,tick_labels=ticks,showfliers=True,widths=.5);ax.axhline(0,c='.4',lw=.8)
    ax.set(ylabel='Recovered Mw − known pulse-input Mw');ax.set_title('d  Noise-free pulse operators',loc='left')
    for ax in axs.flat:grid(ax)
    save(fig,'04_small_event_window_controls',
         '3/5 s NW4 uses 1.5–20 Hz to respect taper bandwidth; 5/10 s NW2 uses three tapers and the original 0.5–20 Hz band.\n'
         'Pulse controls have known unit area but omit real path coda/noise/site effects. Real control agreement is necessary, not sufficient, for a revised scaling.')


def spectral_view(eid,folder,tag,ax,color,label,draw_fit=True):
    path=folder/'spectra'/f'{eid}.npz'
    if not path.exists():return False
    with np.load(path) as a:
        f=a['frequency_Hz'];stem=tag+'_ALL_Multitaper_Boatwright'
        if stem+'_observed' in a:
            obs=a[stem+'_observed'];good=a[stem+'_good'];pred=a[stem+'_predicted']
        elif tag in a:
            array=a[tag];count=np.isfinite(array[:,0]).sum(axis=0);good=count>=max(3,.3*len(array))
            with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);obs=np.nanmean(array[:,0],axis=0)
            pred=np.full(len(f),np.nan)
        else:return False
    valid=good&np.isfinite(obs)
    if valid.any():ax.loglog(f[valid],10**obs[valid],'.',ms=2.4,color=color,alpha=.65,label=label)
    if draw_fit and np.isfinite(pred).any():ax.loglog(f,10**pred,c=color,lw=1.0)
    return bool(valid.any())


def spectra_examples():
    fig,axs=plt.subplots(3,2,figsize=(11.7,10.7));fig.subplots_adjust(left=.095,right=.96,top=.89,bottom=.15,wspace=.30,hspace=.48)
    fig.suptitle('Spectra of successful and unsuccessful small-event trials',fontsize=15,y=.96)
    selected=[]
    for i,mag in enumerate(MAG):
        group=small[small.magnitude_stratum.eq(mag)].sort_values('event_id')
        left=group[group.Mw_quality.eq('A')]
        if not len(left):left=group[group.Mw.notna()]
        left=left.iloc[0] if len(left) else group.iloc[0]
        right=group[group.Mw_quality.ne('A')&group.Mw.notna()]
        right=right.iloc[0] if len(right) else group.iloc[-1]
        for j,row in enumerate([left,right]):
            ax=axs[i,j];first=spectral_view(row.event_id,P,'T10.1',ax,BLUE,'10.1 s · NW4')
            second=spectral_view(row.event_id,P/'nw2_windows','T5.1',ax,GREEN,'5.1 s · NW2')
            if not first and not second:ax.text(.5,.5,'Insufficient usable stack',transform=ax.transAxes,ha='center')
            ax.set(xlim=(.45,22),xlabel='Frequency (Hz)',ylabel='Corrected FAS (m² s)')
            ax.set_title(f'{row.event_id} · catalog ML {row.ml:.1f}\nOriginal Mw {row.Mw_quality}; ML {row.ML_quality}',loc='left',fontsize=9)
            grid(ax);selected.append(dict(event_id=row.event_id,magnitude_stratum=mag,column='first Mw-A case' if j==0 else 'first finite non-A case'))
    fig.legend([Line2D([],[],color=BLUE),Line2D([],[],color=GREEN)],['Original 10.1 s · NW4','Diagnostic 5.1 s · NW2'],loc='lower center',bbox_to_anchor=(.53,.09),ncol=2,frameon=False)
    pd.DataFrame(selected).to_csv(T/'spectral_example_selection.csv',index=False)
    save(fig,'05_small_event_spectra',
         'Examples are the first chronological Mw-A and first finite non-A case in each catalog-magnitude bin; selection does not use Mw–ML residuals.\n'
         'Points show retained spectral samples; lines show fitted spectra, even when another quality criterion fails. The full atlas includes all 180 attempts.')


def pulse_matrix():
    from multitaper import MTSpec
    from spectra import fit
    rows=[];dt=.01;cache={}
    for scheme,nw,k,windows,low in [('standard',4,7,[10.1,20.1],.5),('short_NW4',4,7,[3.1,5.1],1.5),('short_NW2',2,3,[5.1,10.1],.5)]:
        for fc in [3.,8.,15.,30.]:
            for onset in [.1,.5,1.]:
                for duration in windows:
                    t=np.arange(round(duration/dt)+1)*dt;u=np.maximum(t-onset,0)*np.exp(-2*np.pi*fc*np.maximum(t-onset,0));u[t<onset]=0;u/=u.sum()*dt
                    key=(len(u),nw,k)
                    if key in cache:vn,ev=cache[key];m=MTSpec(u,nw,k,dt,vn=vn,lamb=ev)
                    else:m=MTSpec(u,nw,k,dt);cache[key]=(m.vn,m.lamb)
                    f,power=m.rspec();f=f[:,0];amp=np.sqrt(len(u)*dt*power[:,0]/2)
                    gridf=np.geomspace(.5,20,200);ym=np.interp(np.log10(gridf),np.log10(f[1:]),np.log10(np.maximum(amp[1:],1e-100)))
                    ff=np.fft.rfftfreq(2*len(u)+1,dt);aa=dt*abs(np.fft.rfft(u-u.mean(),n=2*len(u)+1));yf=np.interp(np.log10(gridf),np.log10(ff[1:]),np.log10(np.maximum(aa[1:],1e-100)))
                    for method,y in [('Multitaper',ym),('Fourier',yf)]:
                        fitted,_=fit(gridf,y,gridf>=low,depth_km=10.)
                        rows.append(dict(scheme=scheme,nw=nw,kspec=k,corner_Hz=fc,onset_s=onset,duration_s=duration,band_low_Hz=low,method=method,
                            primary_window=duration==windows[0],delta_Mw=np.log10(fitted['omega_r_m2s'])/1.5))
    pd.DataFrame(rows).to_csv(T/'pulse_operator_controls.csv',index=False)


def paragraph(text):return '<p>'+text+'</p>'
def table(head,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def narratives():
    counts=[]
    for label,mag in zip(MAG_LABEL,MAG):
        q=small[small.magnitude_stratum.eq(mag)];counts.append([label,len(q),int(q.Mw_quality.eq('A').sum()),int(q.paired_A.sum()),int(q.paired_A_or_B.sum())])
    first=('<h1>Extending Korean Mw–ML to smaller events</h1>'+paragraph('<b>The measurements can be extended for selected small events. This pilot does not yet establish an unbiased national low-magnitude relation.</b> Both the moment estimate and an independently measured ML must be reliable; a catalog ML below 2 does not guarantee that the measured ML is below 2.')+
      '<h2>A fixed pilot, including failures</h2>'+paragraph('The existing depth catalog contains 9,325 eligible South Korean/onshore–offshore events with 0.5≤catalog ML<2. From these, six events were drawn uniformly in each of 27 magnitude × depth × era strata: 162 small events. Another 18 ML 2.0–2.4 events are controls. The draw preceded waveform inspection; no missing or unsuccessful event was replaced. All 180 selected events had retained records.')+
      table(['Catalog ML','Attempted','Mw A','Both A','Both A/B'],counts)+
      paragraph(f'Original processing gives {S["small_Mw_quality_counts"]["A"]} Mw-A events and {S["small_paired_A"]} paired-A events. The paired-A measured-ML range is {qa.ML_Sheen.min():.2f}–{qa.ML_Sheen.max():.2f}, only a small extension below the old {OLD.ML_Sheen.min():.2f} minimum. The 20 A/B sensitivity pairs reach measured ML {qab.ML_Sheen.min():.2f}; these do not enter the adopted primary relation.')+
      paragraph(f'The controlled 5.1/10.1-s, NW=2/three-taper candidate gives {S2["small_paired_A"]} paired-A cases and reaches measured ML {S2["measured_ML_range_paired_A"][0]:.2f}. Its shared larger-event controls have RMS ΔMw={S2["transfer_controls"]["RMS_delta_Mw"]:.3f}. This supports a first target near measured ML 1.5, subject to further validation and selection analysis.')+
      '<h2>What was held fixed</h2>'+paragraph('HH, EL and HG were processed with the dated response inventory, displacement response removal, measured P/S picks, hypocentral spreading and the same empirical Q. Mw uses source-depth Vs from Kim & Kim (1983), density 2800 kg/m³ and C=9.05; Kim et al. (2011) remains a source-coefficient comparison. The original 10.1/20.1-s, NW=4/seven-taper, 0.5–20-Hz estimator and all A/B quality gates are unchanged. Independent Wood–Anderson ML uses the same published 10–100-km scale and site selection.')+
      paragraph('The overlap screen includes every known nearby domestic catalog event, including those below ML 1.5. The 18 larger controls reproduce the existing finite Mw and ML values to numerical precision. This tests implementation compatibility, not absolute accuracy of the magnitude scale.')+
      paragraph('Success fractions are conditional on prior event detection, G2/G3 depth selection and the balanced sampling design. Non-detections, narrow bands, too few sites and estimator/window failures are retained in the tables. Neither station selection nor magnitude acceptance uses agreement between Mw and ML.'))
    transfer2=S2['transfer_controls'];transfer4=S4['transfer_controls']
    second=('<h1>Interpretation and next steps</h1>'+paragraph('The first objective should be a validated extension toward measured ML about 1.5. The NW2 candidate produces some useful measurements there; the unchanged pipeline barely extends below measured ML 1.8. A large count of nominally small catalog events is not evidence that their independent magnitudes can all be measured.')+
      '<h2>Window and taper tests</h2>'+paragraph(f'The NW4 3.1/5.1-s experiment uses 1.5–20 Hz to respect taper bandwidth; it yields {S4["small_paired_A"]} small paired-A cases. Its higher low-frequency edge reduces the available span. The NW2 5.1/10.1-s experiment uses three tapers and retains 0.5–20 Hz; it yields {S2["small_paired_A"]} paired-A cases. Both tests used all the same selected events, the same response/path constants and unchanged quality gates.')+
      table(['Candidate','Shared Mw-A controls','Median ΔMw','RMS ΔMw','Transfer gate'],[
        ['3/5 s, NW4',transfer4['n_shared_Mw_A'],f'{transfer4["median_delta_Mw"]:+.3f}',f'{transfer4["RMS_delta_Mw"]:.3f}','Pass' if transfer4['passed'] else 'Not established'],
        ['5/10 s, NW2',transfer2['n_shared_Mw_A'],f'{transfer2["median_delta_Mw"]:+.3f}',f'{transfer2["RMS_delta_Mw"]:.3f}','Pass' if transfer2['passed'] else 'Not established']])+
      paragraph('The transfer criteria were fixed before assessing candidate output: at least five shared Mw-A controls, absolute median difference ≤0.05 and RMS ≤0.10, with no empirical amplitude offset. Noise-free unit-area pulse controls probe the window/spectral operators; they omit path coda, real noise and site effects and can have model-dependent absolute errors. Passing these controls does not validate a new population scaling. These method-development tests are exploratory and remain separate from the adopted catalog.')+
      '<h2>What a scaling fit can establish</h2>'+paragraph('The strict additions are too few and too narrow in measured ML for a new primary fit. The explicitly secondary A/B comparison adds 20 events to the existing 357. A line, quadratic and continuous line with a fixed knot at measured ML 2 were compared by whole-family bootstrap and held-out prediction. Equal-axis ODR is a sensitivity. These methods address dependence and errors in both axes; they do not recover events missing because of low signal-to-noise or depth selection. No physical break or national conversion is inferred.')+
      '<h2>A practical route downward</h2>'+paragraph('Expand the ML 1.5–1.9 parent cohort first, retaining failures and auditing the usable nearby sensors. For weaker events in well-recorded clusters, test relative moment ratios against several independently calibrated larger events, with spatial proximity, waveform similarity and mechanism checks. A separately validated microevent tier could use fewer well-audited stations with explicit uncertainty. It should not be created merely by lowering station-count or SNR thresholds until a curve appears.')+
      paragraph('<span class="small">Primary literature: Edwards et al. (2010), GJI 183, 407–420, doi:10.1111/j.1365-246X.2010.04743.x, demonstrates automated spectral Mw down to Swiss ML 1.3. Deichmann (2017), BSSA 107, 505–520, doi:10.1785/0120160318, explains why low-magnitude scaling can approach Mw versus ML slope 2/3 under suitable attenuation conditions; it is not imposed here. Parolai and Oth (2022), BSSA 112, 2364–2375, doi:10.1785/0120220050, documents small-event bandwidth and propagation trade-offs. Their regional results are not Korean calibrations.</span>'))
    return first,second


def atlas():
    destination=P/'reports/Small_event_spectral_atlas.pdf'
    with PdfPages(destination,metadata={'Title':'Small-event pilot: every attempted event','Author':'Korean Mw/ML project'}) as pdf:
        ordered=D.sort_values(['pilot_role','magnitude_stratum','event_id']).reset_index(drop=True)
        for start in range(0,len(ordered),4):
            fig,axs=plt.subplots(2,2,figsize=(11.7,8.8));fig.subplots_adjust(left=.09,right=.96,top=.86,bottom=.13,wspace=.27,hspace=.44)
            fig.suptitle(f'Small-event spectral atlas · {start+1}–{min(start+4,len(ordered))} of {len(ordered)}',fontsize=14,y=.96)
            for ax,row in zip(axs.flat,ordered.iloc[start:start+4].itertuples()):
                a=spectral_view(row.event_id,P,'T10.1',ax,BLUE,'Original')
                b=spectral_view(row.event_id,P/'nw2_windows','T5.1',ax,GREEN,'NW2 diagnostic')
                if not a and not b:ax.text(.5,.5,'No usable event stack',ha='center',transform=ax.transAxes)
                measured=f'{row.ML_Sheen:.2f}' if np.isfinite(row.ML_Sheen) else 'unmeasured'
                ax.set_title(f'{row.event_id} · {row.pilot_role}\nCatalog ML {row.ml:.1f}; measured ML {measured}; original Mw {row.Mw_quality}',fontsize=8.7,loc='left')
                ax.set(xlim=(.45,22),xlabel='Frequency (Hz)',ylabel='Corrected FAS (m² s)');grid(ax)
            fig.legend([Line2D([],[],c=BLUE),Line2D([],[],c=GREEN)],['Original 10.1 s, NW4','Diagnostic 5.1 s, NW2'],loc='lower center',bbox_to_anchor=(.53,.06),ncol=2,frameon=False)
            fig.text(.065,.025,'Every attempted case is shown. A curve does not establish an accepted magnitude; grades and rejection reasons are in pilot_Mw_ML_catalog.csv.',fontsize=7.5,color='.3')
            pdf.savefig(fig);plt.close(fig)
    return destination


def main():
    assert S['complete'] and len(D)==180 and len(NW2)==180
    pulse_matrix();sampling();scaling();limitations();window_controls();spectra_examples()
    first,last=narratives();doc=fitz.open();scales=[]
    css='body{font-family:sans-serif;font-size:10pt;line-height:1.29;color:#222;}h1{font-size:18pt;margin:0 0 11pt;}h2{font-size:11.5pt;margin:11pt 0 5pt;}p{margin:0 0 8pt;}table{width:100%;border-collapse:collapse;font-size:9pt;margin:8pt 0;}td,th{padding:5pt 4pt;border-bottom:.5pt solid #ccd3d8;text-align:left;}th{background:#edf2f5;}.small{font-size:8pt;color:#444;}'
    def insert(text):
        page=doc.new_page(width=595.276,height=841.89)
        spare,scale=page.insert_htmlbox(fitz.Rect(40,28,555,812),text,css=css,scale_low=.94)
        if spare<0:raise RuntimeError('Pilot narrative exceeds page')
        scales.append(scale)
    insert(first)
    for item in FIGURES:
        with fitz.open(F/(item['stem']+'.pdf')) as part:doc.insert_pdf(part)
    insert(last)
    doc.set_toc([[1,name,i+1] for i,name in enumerate(['Pilot design and findings','Sampling and yield','Scaling coverage','Measurement limits','Window controls','Spectral examples','Interpretation and literature'])])
    report=P/'reports/Small_event_Mw_ML_pilot.pdf';doc.save(report,garbage=4,deflate=True)
    atlas_path=atlas()
    with fitz.open(atlas_path) as check:assert len(check)==45
    assert len(doc)==7
    for output in [report,atlas_path]:
        with fitz.open(output) as check:
            for i,page in enumerate(check):
                assert '\x00' not in page.get_text()
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        for span in line.get('spans',[]):
                            x0,y0,x1,y1=span['bbox']
                            assert min(x0,y0)>=0 and x1<=page.rect.width and y1<=page.rect.height,(output,i,span)
                if output==report:page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(f'/tmp/korean_small_pilot_{i+1}.png')
    protected=json.loads((P/'inputs/protected_results.json').read_text())
    assert all(sha(R/name)==value for name,value in protected.items())
    metadata=dict(passed=True,report_pages=len(doc),narrative_scales=scales,
        report_sha256=sha(report),atlas_sha256=sha(atlas_path),atlas_pages=45,
        original_results_and_reports_unchanged=True,protected_sha256=protected,
        planned_and_attempted=180,small_events=162,controls=18,strict_small_pairs=int(small.paired_A.sum()),
        analysis_scope='Feasibility and exploratory sensitivity; no revised primary Mw-ML relation or adopted magnitudes',
        source_sha256={str(path.relative_to(R)):sha(path) for path in [R/'code/prepare_small_event_pilot.py',R/'code/run_small_event_pilot.py',R/'code/analyze_small_event_pilot.py',R/'code/small_event_short_windows.py',R/'code/small_event_nw2_windows.py',Path(__file__)]},
        figure_font_and_page_bounds_passed=True)
    (T/'report_validation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (T/'figure_manifest.json').write_text(json.dumps(FIGURES,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['protected_sha256','source_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
