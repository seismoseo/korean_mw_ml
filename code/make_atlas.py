"""A reviewable spectrum panel for every candidate, including rejected fits."""
from pathlib import Path
import json,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
HERE=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'legend.fontsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
d=pd.read_csv(HERE/'tables/Mw_ML_catalog.csv',dtype={'event_id':str}).sort_values(['ml','origin_utc'],ascending=[False,True])
fits=pd.read_csv(HERE/'tables/clean_spectral_fits.csv',dtype={'event_id':str})
fits=fits[(fits.window_s==10.1)&(fits.sensor_set=='ALL')&(fits['shape']=='Boatwright')]
groups={key:g for key,g in fits.groupby('event_id')}
rules=json.loads((HERE/'inputs/analysis_quality_rules.json').read_text());bad={(r['station'],r['sensor']) for r in rules['known_response_exclusions']}
out=HERE/'reports/Spectral_fit_atlas.pdf';index=[]
with PdfPages(out) as pdf:
    records=d.to_dict('records')
    for page,start in enumerate(range(0,len(records),4),1):
        fig,axes=plt.subplots(2,2,figsize=(10.5,8.3))
        fig.subplots_adjust(left=.095,right=.97,bottom=.12,top=.90,hspace=.42,wspace=.25)
        fig.suptitle('Korean ML ≥2 events: spectra and source fits',fontsize=13,y=.97)
        for ax,r in zip(axes.flat,records[start:start+4]):
            eid=r['event_id'];p=HERE/'spectra'/(eid+'.npz')
            title=r['origin_utc']+' UTC; ML '+f'{r["ml"]:.1f}'+'; '+r['Mw_quality']
            ax.set_title(title,loc='left',fontsize=8.8)
            if p.exists():
                b=np.load(p);f=b['frequency_Hz']
                if 'T10.1' in b:
                    names=b['T10.1_stations'];sensors=b['T10.1_sensor'];keep=np.array([(str(a),str(z)) not in bad for a,z in zip(names,sensors)])
                    arr=b['T10.1'][keep]
                    if len(arr):
                        for j in np.unique(np.linspace(0,len(arr)-1,min(20,len(arr)),dtype=int)):
                            ax.loglog(f,10**arr[j,1,:],c='.82',lw=.4,zorder=0)
                        good=np.isfinite(arr[:,0,:]).sum(axis=0)>=max(3,.3*len(arr))
                        for m,method,color in [(0,'Multitaper','#176da0'),(1,'Direct Fourier','#bd591b')]:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore',RuntimeWarning);y=np.nanmean(arr[:,m,:],axis=0)
                            ax.loglog(f[good],10**y[good],c=color,lw=.9)
                            q=groups.get(eid,pd.DataFrame())
                            if len(q):
                                q=q[q.method==method]
                                if len(q) and np.isfinite(q.iloc[0].get('omega_r_m2s',np.nan)):
                                    v=q.iloc[0];y=np.log10(v.omega_r_m2s)-.5*np.log10(1+(f/v.fc_Hz)**(2*v.falloff))
                                    ax.loglog(f,10**y,c=color,lw=1.2,ls='--')
            if not ax.lines:ax.text(.5,.5,'No usable primary spectrum',ha='center',transform=ax.transAxes,color='.4')
            mw=f'{r["Mw"]:.2f}' if np.isfinite(r.get('Mw',np.nan)) else '—'
            alternative=f'{r["Mw_source_Vs_Kim2011"]:.2f}' if np.isfinite(r.get('Mw_source_Vs_Kim2011',np.nan)) else '—'
            ax.text(.035,.05,f'MT Mw {mw}; Kim 2011 {alternative}\nDepth {r["z_central"]:.2f} km; interval {r["lower"]:.2f}–{r["upper"]:.2f} km',
              transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.82,pad=2))
            ax.set_xlim(.5,20);ax.set_xlabel('Frequency (Hz)');ax.set_ylabel(r'Corrected amplitude (m$^2$ s)');ax.grid(color='.9',lw=.5)
            index.append(dict(event_id=eid,origin_utc=r['origin_utc'],ML=r['ml'],Mw=r.get('Mw',np.nan),Mw_Kim2011=r.get('Mw_source_Vs_Kim2011',np.nan),source_Vs_m_s=r['source_Vs_m_s'],Mw_quality=r['Mw_quality'],page=page))
        for ax in axes.flat[len(records[start:start+4]):]:ax.axis('off')
        fig.text(.08,.035,'MT Mw: Kim & Kim (1983), Vp/Vs=1.73; comparison: published Kim et al. (2011). Source Vs evaluated at the plotted depth; C=9.05.\n'
         'Blue: multitaper; orange: Fourier; dashed: fits; gray: station spectra. A enters the primary relation; B exploratory; C/F excluded.',
         fontsize=8,linespacing=1.4)
        fig.text(.96,.025,str(page),ha='right',fontsize=8,color='.4')
        pdf.savefig(fig);plt.close(fig)
        if page%20==0:print('Atlas page',page,flush=True)
pd.DataFrame(index).to_csv(HERE/'tables/spectral_atlas_index.csv',index=False)
print('Atlas complete:',len(index),'event panels,',page,'pages',flush=True)
