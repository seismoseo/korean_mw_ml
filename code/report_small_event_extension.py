"""Figures and a concise report for the uniform-estimator small-event expansion."""
from pathlib import Path
import argparse
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
from report_extension_potency import PotencySection

R=Path(__file__).resolve().parents[1];E=R/'small_events/extension';T=E/'tables';F=E/'figures'
BLUE,RED,GREEN,ORANGE,GRAY='#176da0','#b32438','#298365','#bd641f','#969da1'
FIGURES=[]
plt.rcParams.update({'font.family':'Liberation Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csv(path):return pd.read_csv(path,dtype={'event_id':str})
def grid(ax):ax.grid(color='.92',lw=.5)
def save(fig,name,caption):
    fig.text(.065,.027,caption,fontsize=7.7,color='.25',va='bottom',linespacing=1.35)
    for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',dpi=400,facecolor='white')
    plt.close(fig);FIGURES.append(dict(stem=name,caption=caption));print(name,flush=True)
def frame_figure(title,left=.09):
    fig,axes=plt.subplots(2,2,figsize=(11.7,9.3));fig.subplots_adjust(left=left,right=.96,top=.87,bottom=.16,wspace=.31,hspace=.43)
    fig.suptitle(title,fontsize=15,y=.96);return fig,axes
def fit_row(selection='NW2 primary',model='linear',method='family-balanced OLS'):
    return FIT[FIT.selection.eq(selection)&FIT.model.eq(model)&FIT.method.eq(method)].iloc[0]
def comparison_row(cohort,axis='our',method='OLS',degree=1):
    return C_FIT[C_FIT.cohort.eq(cohort)&C_FIT.axis.eq(axis)&C_FIT.method.eq(method)&C_FIT.degree.eq(degree)].iloc[0]


def sampling():
    fig,axes=frame_figure('A fixed expansion of the smaller-event sample')
    ax=axes[0,0];points=[]
    def coast():
        if len(points)>1:
            p=np.asarray(points);ax.plot(p[:,0],p[:,1],c='.65',lw=.5,zorder=0)
    for line in (R/'inputs/coastline_low.gmt').read_text().splitlines():
        if line.startswith('>'):coast();points=[]
        elif line.strip():points.append(list(map(float,line.split()[:2])))
    coast();s=D[D.ml.lt(2)];a=s[s.paired_A]
    ax.scatter(s.lon,s.lat,s=4,c='.77',alpha=.45,ec='none')
    ax.scatter(a.lon,a.lat,s=12,c=RED,alpha=.8,ec='white',lw=.15)
    ax.legend([Line2D([],[],marker='o',ls='none',c='.7',ms=3),Line2D([],[],marker='o',ls='none',c=RED,ms=3)],
              ['Attempts','Both A'],loc='lower right',frameon=False,fontsize=7)
    ax.set(xlim=(124.5,131),ylim=(32.8,38.8),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
    ax.set_aspect(1/np.cos(np.radians(36)));ax.set_title('a  Small-event attempts and paired A',loc='left');grid(ax)
    ax=axes[0,1];roles=['pilot_only_0p5_0p9','nested_1p0_1p4','census_1p5_1p9','reference_MLge2'];labels=['0.5–0.9','1.0–1.4','1.5–1.9','≥2']
    x=np.arange(4)
    for off,flag,label,color in [(-.19,'Mw_quality','Mw A',BLUE),(.19,'paired_A','Both A',RED)]:
        groups=[D[D.extension_role.eq(role)] for role in roles]
        counts=np.array([q.Mw_quality.eq('A').sum() if flag=='Mw_quality' else q.paired_A.sum() for q in groups]);n=np.array([len(q) for q in groups])
        bars=ax.bar(x+off,100*counts/n,.36,label=label,color=color)
        ax.bar_label(bars,labels=[f'{a}/{b}' for a,b in zip(counts,n)],rotation=90,padding=3,fontsize=7)
    ax.set(xticks=x,xticklabels=labels,xlabel='Catalog ML',ylabel='Successful attempts (%)',ylim=(0,100));ax.legend(frameon=False,ncol=2,loc='upper left')
    ax.set_title('b  Counts and success fractions',loc='left');grid(ax)
    ax=axes[1,0];levels=['0-10km','10-16p89km','ge16p89km'];cats=[(1,1.5),(1.5,2),(2,6)]
    values=np.zeros((3,3));counts={}
    for i,level in enumerate(levels):
        for j,(lo,hi) in enumerate(cats):
            q=D[D.depth_stratum.eq(level)&D.ml.ge(lo)&D.ml.lt(hi)];n=len(q);m=int(q.paired_A.sum());values[i,j]=m/n;counts[i,j]=(m,n)
    ax.imshow(values,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for (i,j),(m,n) in counts.items():ax.text(j,i,f'{m}/{n}',ha='center',va='center',fontsize=9,color='white' if values[i,j]>.65 else '.15')
    ax.set(xticks=range(3),xticklabels=['1.0–1.4','1.5–1.9','≥2'],yticks=range(3),yticklabels=['<10 km','10–16.89 km','≥16.89 km'],xlabel='Catalog ML')
    ax.set_title('c  Both A, by depth',loc='left')
    ax=axes[1,1];q=D[D.ml.lt(2)]
    for grade,color in [('F','.78'),('B',ORANGE),('A',RED)]:
        group=q[q.ML_quality.eq(grade)]
        ax.scatter(group.ml,group.ML_Sheen,s=8,c=color,alpha=.35,ec='none',label='ML '+grade)
    ax.plot([.4,2.1],[.4,2.1],c='.35',ls=':',lw=.8);ax.set(xlim=(.4,2.05),xlabel='Catalog ML',ylabel='Measured ML')
    ax.set_title('d  Catalog and independently measured ML',loc='left');ax.legend(frameon=False,ncol=2,fontsize=7);grid(ax)
    save(fig,'01_extension_sampling','All 2,317 confident-depth catalog-ML 1.5–1.9 events were attempted; lower bins use the fixed nested sample.\n'
         'Fractions are conditional on event detection, confident depth and this sampling design. Paired A requires grade A for both Mw and independent ML.')


def scaling():
    fig,axes=frame_figure('Family-weighted reference: linear and quadratic fits')
    accepted=D[D.paired_A&D.ML_Sheen.ge(1.5)];ref=accepted[accepted.ml.ge(2)];small=accepted[accepted.ml.lt(2)]
    for ax,degree,color,title in [(axes[0,0],1,BLUE,'a  Linear fit'),(axes[0,1],2,GREEN,'b  Quadratic fit')]:
        ax.scatter(ref.ML_Sheen,ref.Mw,s=10,c='.72',alpha=.5,ec='none',label='Catalog ML≥2')
        ax.scatter(small.ML_Sheen,small.Mw,s=12,c=RED,alpha=.55,ec='none',label='Catalog ML<2')
        r=comparison_row('expanded_NW2',degree=degree)
        q=C_CURVE[C_CURVE.case_id.eq(r.case_id)&C_CURVE.in_fitted_range]
        ax.plot(q.ML,q.Mw,c=color,lw=1.5,label='Line' if degree==1 else 'Quadratic')
        ax.fill_between(q.ML,q.lower,q.upper,color=color,alpha=.13,lw=0)
        equation=f'Mw = {r.a:.6f} + {r.b:.6f} u'+(f' {r.c:+.6f} u²' if degree==2 else '')+'\nu = measured ML − 2'
        ax.text(.03,.97,equation,transform=ax.transAxes,va='top',fontsize=8,color=color)
        ax.set(xlabel='Measured ML',ylabel='Mw',xlim=(1.4,5.5),ylim=(1.4,5.35));ax.set_title(title,loc='left');grid(ax)
    axes[0,0].legend(frameon=False,fontsize=7,loc='lower right')
    ax=axes[1,0];line=comparison_row('expanded_NW2',degree=1);quad=comparison_row('expanded_NW2',degree=2)
    x=np.linspace(max(line.x_min,quad.x_min),min(line.x_max,quad.x_max),200);u=x-2
    difference=(quad.a-line.a)+(quad.b-line.b)*u+quad.c*u*u
    ax.plot(x,difference,c=GREEN,lw=1.5);ax.axhline(0,c='.4',ls=':',lw=.8)
    ax.text(.04,.08,f'Max |difference| = {abs(difference).max():.6f} Mw',transform=ax.transAxes,fontsize=8)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    ax.set(xlabel='Measured ML',ylabel='Quadratic − linear Mw');ax.set_title('c  Difference between the two fitted curves',loc='left');grid(ax)
    ax=axes[1,1];models=['linear','quadratic','hinge2'];x=np.arange(3)
    q=FIT[FIT.selection.eq('NW2 primary')&FIT.method.eq('family-balanced OLS')].set_index('model')
    for off,col,label,color in [(-.18,'CV_RMS_family_new_small','All new small',BLUE),(.18,'CV_RMS_family_new_small_ML1p5_2','New, measured ML 1.5–2',RED)]:
        values=[q.loc[m,col] for m in models];bars=ax.bar(x+off,values,.34,color=color,label=label);ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=7)
    ax.set(xticks=x,xticklabels=['Line','Quadratic','Knot at ML 2'],ylabel='Whole-family held-out RMS',ylim=(0,max(q.CV_RMS_family_new_small.max(),q.CV_RMS_family_new_small_ML1p5_2.max())*1.45));ax.legend(frameon=False,fontsize=7)
    ax.set_title('d  New small events; original pilot excluded',loc='left');grid(ax)
    save(fig,'02_extension_scaling','The quadratic is fitted freely; its small curvature makes it almost overlap the line. Panel (c) shows the actual difference on its own scale.\n'
         'Both panels use the same 676 events and family-balanced OLS. Shading: pointwise 95% family-bootstrap bands. The ML axis is independently measured Wood–Anderson ML.')


def local_magnitude_results():
    fig,axes=frame_figure('Local magnitude was measured from the waveforms')
    for ax,zoom in [(axes[0,0],False),(axes[0,1],True)]:
        for grade,color,label in [('B','.7','ML B'),('A',RED,'ML A')]:
            q=D[D.ML_quality.eq(grade)]
            x=q.ml;y=q.ML_Sheen if not zoom else q.ML_Sheen-q.ml
            ax.scatter(x,y,s=7,c=color,alpha=.35,ec='none',label=label)
        if zoom:ax.axhline(0,c='.3',ls=':',lw=.8)
        else:ax.plot([.3,5.4],[.3,5.4],c='.4',ls=':',lw=.8)
        ax.set(xlabel='KMA ML',ylabel='Measured ML − KMA ML' if zoom else 'Measured Wood–Anderson ML',xlim=(.35,5.3));grid(ax)
    axes[0,0].set_title('a  2,222 independently measured ML values',loc='left');axes[0,0].legend(frameon=False,fontsize=7)
    axes[0,1].set_title('b  Difference from the catalog magnitude',loc='left')
    ax=axes[1,0];eid='20250406110656';event=D[D.event_id.eq(eid)].iloc[0];st=C_STATION[C_STATION.event_id.eq(eid)]
    for sensor,color in [('HH',BLUE),('EL',ORANGE),('HG',GREEN)]:
        q=st[st.sensor.eq(sensor)];ax.scatter(q.epicentral_km,q.ML_station,s=20,c=color,alpha=.8,label=f'{sensor} ({len(q)})')
    ax.axhline(event.ML_Sheen,c=RED,lw=1.1,label=f'Median {event.ML_Sheen:.3f}')
    ax.axhline(event.ml,c='.3',ls=':',lw=1,label=f'KMA {event.ml:.1f}')
    ax.set(xlabel='Epicentral distance (km)',ylabel='Station ML',xlim=(8,102));ax.legend(frameon=False,fontsize=7,ncol=2)
    ax.set_title('c  Sangju area, 6 Apr 2025 11:06:56 UTC',loc='left');grid(ax)
    ax=axes[1,1]
    for grade,color in [('B','.7'),('A',RED)]:
        q=D[D.ML_quality.eq(grade)];ax.scatter(q.ML_Sheen,q.ML_Sheen_n,s=8,c=color,alpha=.3,ec='none',label='ML '+grade)
    ax.axhline(5,c='.4',ls=':',lw=.8);ax.set(xlabel='Measured ML',ylabel='Stations contributing to ML');ax.set_title('d  Station support for measured ML',loc='left');grid(ax)
    save(fig,'06_independent_local_magnitude','ML_Sheen is measured from response-corrected Wood–Anderson amplitudes; ml is the separate KMA catalog value. No event magnitude is copied or fitted to KMA.\n'
         'All 46,117 retained station readings reproduce from their saved N/E peak amplitudes and the published attenuation formula. Station corrections are zero.')


def regression_history():
    fig,axes=frame_figure('Previous larger-event regressions and the expanded fits')
    colors={'previous':'#8064a2','expanded_NW4':BLUE,'larger_NW2':ORANGE,'expanded_NW2':GREEN}
    target=D[D.paired_A&D.ML_Sheen.ge(1.5)]
    for i,(axis,column,label) in enumerate([('our','ML_Sheen','Measured ML'),('KMA','ml','KMA ML')]):
        for j,degree in enumerate([1,2]):
            ax=axes[i,j];ax.scatter(target[column],target.Mw,s=8,c='.75',alpha=.3,ec='none')
            for cohort in colors:
                r=comparison_row(cohort,axis,degree=degree);curve=C_CURVE[C_CURVE.case_id.eq(r.case_id)]
                good=curve.in_fitted_range.to_numpy();ax.plot(curve.ML,curve.Mw,c=colors[cohort],lw=1,ls='--',alpha=.65)
                ax.plot(curve.ML[good],curve.Mw[good],c=colors[cohort],lw=1.6)
            ax.set(xlabel=label,ylabel='Mw',xlim=(1.4,5.5) if axis=='our' else (.5,5.3),ylim=(1.3,5.35))
            ax.set_title(f'{"abcd"[2*i+j]}  {label}: '+('linear' if degree==1 else 'quadratic'),loc='left');grid(ax)
    handles=[Line2D([],[],c=color,lw=1.6) for color in colors.values()]
    labels=[comparison_row(cohort).cohort_label for cohort in colors]
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.095),ncol=4,frameon=False,fontsize=7.4)
    save(fig,'07_previous_and_expanded_regressions','Solid curves: observed ML range of each training cohort. Dashed curves: extrapolation. Points: the same 676 NW2 target events.\n'
         'Previous 357 curves reproduce the earlier saved coefficients. NW4 and NW2 use different windows/tapers; the two NW2 cohorts isolate the addition of smaller events.')


def regression_prediction():
    fig,axes=frame_figure('Prediction comparisons on identical target events')
    cohorts=['previous','expanded_NW4','larger_NW2','expanded_NW2'];ticks=['Previous\nNW4, 357','Expanded\nNW4, 582','Larger\nNW2, 375','Expanded\nNW2, 676']
    for i,axis in enumerate(['our','KMA']):
        for j,group in enumerate(['New small, measured ML1.5–2','Measured ML>=4']):
            ax=axes[i,j];subset=C_METRIC[C_METRIC.axis.eq(axis)&C_METRIC.method.eq('OLS')&C_METRIC.target.eq(group)]
            for off,degree,color,label in [(-.18,1,BLUE,'Linear'),(.18,2,GREEN,'Quadratic')]:
                values=[subset[subset.cohort.eq(c)&subset.degree.eq(degree)].RMS_family.iloc[0] for c in cohorts]
                bars=ax.bar(np.arange(4)+off,values,.34,color=color,label=label);ax.bar_label(bars,fmt='%.4f',fontsize=6.7,padding=2)
            n=subset.iloc[0];ax.set(xticks=np.arange(4),xticklabels=ticks,ylabel='Held-out RMS by family',ylim=(0,subset.RMS_family.max()*1.32))
            axis_label='Measured ML' if axis=='our' else 'KMA ML'
            group_label='new ML 1.5–2' if j==0 else 'measured ML≥4'
            ax.set_title(f'{"abcd"[2*i+j]}  {axis_label}; {group_label}\n{int(n.n_events)} events, {int(n.n_families)} families',loc='left',fontsize=9)
            if i==0 and j==0:ax.legend(frameon=False,fontsize=7,ncol=2)
            grid(ax)
    save(fig,'08_regression_prediction_comparison','Every model predicts the same NW2 target magnitudes; each target family is excluded from its training cohort. Target ranges use measured ML for both x-axis definitions.\n'
         'NW4-to-NW2 comparisons include a method change. The NW2 larger-only versus expanded comparison holds the method fixed. Large-event conclusions have few independent families.')


OBJECTIVE_STYLE=[('family_OLS',1,'Family line',GRAY,'--'),('family_OLS',2,'Family quadratic',BLUE,'-'),
    ('event_OLS',2,'Event quadratic',ORANGE,'-'),('bin_OLS',2,'Bin-median quadratic',RED,'-'),
    ('bin_ODR',2,'Bin-median ODR',GREEN,':')]
def objective_row(cohort='expanded_NW2',objective='bin_OLS',degree=2,axis='our'):
    return O_FIT[O_FIT.cohort.eq(cohort)&O_FIT.objective.eq(objective)&O_FIT.degree.eq(degree)&O_FIT.axis.eq(axis)].iloc[0]
def objective_metric(objective,degree,subset):
    return O_METRIC[O_METRIC.cohort.eq('expanded_NW2')&O_METRIC.axis.eq('our')&O_METRIC.objective.eq(objective)&
        O_METRIC.degree.eq(degree)&O_METRIC.target_set.eq('shared_NW2')&O_METRIC.subset.eq(subset)].iloc[0]


def objective_curves():
    fig,axes=frame_figure('Fitting across the magnitude range: restore the bin-median method')
    original=csv(R/'tables/Mw_independent_ML_primary.csv');primary=D[D.paired_A&D.ML_Sheen.ge(1.5)]
    for j,(cohort,points,title) in enumerate([('previous',original,'Previous: 357 events, NW4'),('expanded_NW2',primary,'Expanded: 676 events, NW2')]):
        for i in range(2):
            ax=axes[i,j];big=points[points.ML_Sheen.gt(4)].sort_values('event_id')
            ax.scatter(points.ML_Sheen,points.Mw,s=9 if i==0 else 24,c='.75' if i==0 else '.25',alpha=.55,zorder=3,ec='none')
            if i==0:
                r=objective_row(cohort);med=O_BINS[O_BINS.case_id.eq(r.case_id)]
                ax.scatter(med.ML_median,med.Mw_median,s=16,marker='s',facecolors='none',edgecolors='.35',lw=.6,zorder=3)
            else:
                for k,row in enumerate(big.itertuples()):ax.annotate(str(k+1),(row.ML_Sheen,row.Mw),xytext=(4,5),textcoords='offset points',fontsize=7)
            x=np.linspace(points.ML_Sheen.min(),points.ML_Sheen.max(),300);u=x-2
            for obj,degree,label,color,style in OBJECTIVE_STYLE:
                r=objective_row(cohort,obj,degree);ax.plot(x,r.a+r.b*u+r.c*u*u,c=color,ls=style,lw=1.3,zorder=2)
            ax.set(xlabel='Measured ML',ylabel='Mw',xlim=(1.4,5.5) if i==0 else (3.8,5.5),ylim=(1.4,5.35) if i==0 else (3.4,5.25))
            ax.set_title(f'{"abcd"[i*2+j]}  '+(title if i==0 else 'The six events with measured ML > 4'),loc='left');grid(ax)
    fig.legend([Line2D([],[],c=c,ls=s,lw=1.4) for _,_,_,c,s in OBJECTIVE_STYLE],
        [v[2] for v in OBJECTIVE_STYLE],loc='lower center',bbox_to_anchor=(.53,.087),ncol=5,frameon=False,fontsize=7.1)
    save(fig,'09_extension_fit_objectives','Each curve uses all accepted training magnitudes, with the stated weights. Open squares: the same filtered bin medians used for the bin fits.\n'
        'The previous headline and bin-OLS coefficients reproduce exactly. Bin-median ODR is an additional equal-axis check. NW4/NW2 magnitudes remain separate.\n'
        'Labels 1–6: 2015-12-21, 2016-09-12, 2016-09-19, 2019-07-21, 2023-11-29 and 2024-06-11 (UTC).')


def objective_diagnosis():
    fig,axes=frame_figure('Why the expanded family-weighted quadratic became nearly linear')
    ax=axes[0,0];edges=np.arange(1.5,5.751,.25);x=(edges[:-1]+edges[1:])/2
    for off,cohort,color,label in [(-.055,'previous',GRAY,'Previous, 194 families'),(.055,'expanded_NW2',BLUE,'Expanded, 374 families')]:
        q=O_WEIGHT[O_WEIGHT.cohort.eq(cohort)];mass=np.histogram(q.ML_measured,bins=edges,weights=100*q.normalized_family_weight)[0]
        ax.bar(x+off,mass,.105,color=color,label=label)
    ax.text(.035,.95,'Weight carried by ML > 4:\n1.815% → 0.606%',transform=ax.transAxes,va='top',fontsize=8)
    ax.set(xlabel='Measured ML',ylabel='Share of family-weighted loss (%)');ax.set_title('a  Equal family weight does not balance magnitudes',loc='left',fontsize=9)
    ax.legend(frameon=False,fontsize=7,loc='upper right');grid(ax)
    objectives=['family_OLS','family_ODR','event_OLS','bin_OLS','bin_ODR'];labels=['Family OLS','Family ODR','Event OLS','Bin OLS','Bin ODR']
    ax=axes[0,1]
    for off,cohort,color,label in [(-.10,'previous',GRAY,'Previous'),(.10,'expanded_NW2',RED,'Expanded')]:
        for k,obj in enumerate(objectives):
            r=objective_row(cohort,obj)
            ax.errorbar(r.c,k+off,xerr=[[max(0,r.c-r.c_lower)],[max(0,r.c_upper-r.c)]],fmt='o',ms=3.5,c=color,capsize=2,label=label if k==0 else None)
    ax.axvline(0,c='.5',ls=':',lw=.8);ax.set(yticks=range(5),yticklabels=labels,xlabel='Quadratic coefficient c');ax.invert_yaxis()
    ax.set_title('b  Curvature and whole-family bootstrap intervals',loc='left',fontsize=9);ax.legend(frameon=False,fontsize=7);grid(ax)
    for ax,subset,title in [(axes[1,0],'New measured ML1.5–2','c  New ML 1.5–2: 166 events, 112 families'),(axes[1,1],'Measured ML>4','d  ML > 4: six events, five families')]:
        for off,degree,color,label in [(-.18,1,BLUE,'Linear'),(.18,2,RED,'Quadratic')]:
            values=[objective_metric(obj,degree,subset).CV_RMS_events for obj in objectives]
            bars=ax.bar(np.arange(5)+off,values,.34,color=color,label=label)
            ax.bar_label(bars,fmt='%.4f',fontsize=6.5,padding=2)
        ax.set(xticks=np.arange(5),xticklabels=[s.replace(' ','\n') for s in labels],ylabel='Held-out event RMS (Mw)')
        ax.set_ylim(0,ax.get_ylim()[1]*1.22);ax.set_title(title,loc='left',fontsize=9);grid(ax)
    axes[1,0].legend(frameon=False,fontsize=7,ncol=2)
    fig.subplots_adjust(wspace=.39)
    save(fig,'10_extension_weight_and_prediction','All prediction scores use the same NW2 target magnitudes. Every target family, including its smaller members, is omitted and bin medians are recomputed.\n'
        'Bin fits give each occupied magnitude bin equal weight. Intervals resample whole families; six large events supply only five independent families. Event RMS differs from family RMS.')


def objective_narrative():
    q=objective_row();odr=objective_row(objective='bin_ODR');previous=objective_row('previous')
    weights=[]
    for cohort in ['previous','expanded_NW2']:
        part=O_WEIGHT[O_WEIGHT.cohort.eq(cohort)];weights.append(100*part.loc[part.ML_measured.gt(4),'normalized_family_weight'].sum())
    rows=[]
    for obj,degree,label in [('family_OLS',1,'Family OLS: line'),('family_OLS',2,'Family OLS: quadratic'),
        ('event_OLS',2,'Event OLS: quadratic'),('bin_OLS',1,'Bin OLS: line'),('bin_OLS',2,'Bin OLS: quadratic'),('bin_ODR',2,'Bin ODR: quadratic')]:
        m=objective_row(objective=obj,degree=degree);large=objective_metric(obj,degree,'Measured ML>4');small=objective_metric(obj,degree,'New measured ML1.5–2')
        rows.append([label,f'{m.c:+.6f}',f'{large.fit_RMS_events:.4f}',f'{large.CV_RMS_events:.4f}',f'{small.CV_RMS_events:.4f}'])
    return ('<h1>Fitting the large events</h1>'+paragraph('<b>The earlier headline used the same family-weighted least-squares engine, but its bin-median alternatives were missing from the extension.</b> The earlier coefficients and native-target prediction scores now reproduce for every repeated method. The near-zero family-weighted curvature was a valid result for that objective; it was an incomplete answer to the requested fit across the magnitude range.')+
        '<h2>How the fitting objective changed in practice</h2>'+paragraph(f'Under family weights, each event receives 1/(number of accepted members in its family). The six measured-ML>4 events carry {weights[0]:.3f}% of the previous loss but only {weights[1]:.3f}% after expansion. Adding smaller members also dilutes individual large events: Gyeongju grows from 93 to 122 members; the 2015 and 2019 large-event families each grow from one to two. Equal family weight does not give equal emphasis to magnitude ranges.')+
        '<h2>Restore the earlier 50-bin method</h2>'+paragraph(f'Divide the observed training ML range into 50 equal bins, trim |Mw−initial bin median|>0.5, recompute ML/Mw medians and fit those medians with equal bin weights. The expanded measured-ML sample has {int(q.n_occupied_bins)} occupied bins, including {int(q.n_singleton_bins)} singletons; <b>no event is trimmed</b>. This is the same bin-OLS procedure used in the earlier report. Equal-axis ODR on these bin medians is an added sensitivity.')+
        paragraph(f'With <b>u = measured ML − 2</b>, the magnitude-balanced descriptive quadratic is <b>Mw = {q.a:.6f} + {q.b:.6f}u + {q.c:.6f}u²</b>. The earlier bin-OLS curvature was {previous.c:+.6f}. The expanded bin-ODR curvature is {odr.c:+.6f}. Repeating the expanded analysis with NW4 gives bin-OLS curvature +0.035951, so the positive bin-fit curvature is present under both spectral methods.')+
        '<h2>Fit and prediction are separate tests</h2>'+table(['Expanded NW2 fit','c','Large fit RMS','Large CV RMS','New small CV RMS'],rows)+
        paragraph('RMS values above average events. Large: six measured-ML>4 events in five families. New small: 166 measured-ML1.5–2 events outside the original pilot. CV withholds the entire target family and recomputes bins, trimming and fit weights. All methods predict the same NW2 target magnitudes. Both Gyeongju events require extrapolation when their family is withheld.')+
        paragraph(f'The bin quadratic addresses the large-event fitting goal while retaining nearly the same small-event prediction error as the family line. Its whole-family curvature interval is [{q.c_lower:+.4f}, {q.c_upper:+.4f}]. Five-family paired held-out-loss resampling favors the bin quadratic over its bin line, but only resamples the existing five families; it does not validate a new large-event population. Use the bin curve to describe the observed magnitude range and retain the family-weighted curve as a different weighting sensitivity.')+
        paragraph('fit_objective_*.csv and fit_objective_bootstrap.npz retain 80 specifications, both ML definitions, previous/expanded NW4 and NW2 cohorts, bin medians, event weights, native/shared-target predictions and uncertainty. Measured magnitudes and quality selections are unchanged.'))


def controls():
    fig,axes=frame_figure('Separate the spectral-method change from the sample expansion')
    common=csv(T/'shared_Mw_A_controls.csv');base=OLD.set_index('event_id')
    ax=axes[0,0];ax.scatter(base.loc[common.event_id,'Mw'],common.delta_Mw,s=13,c=BLUE,alpha=.55,ec='none')
    ax.axhline(0,c='.3',lw=.8)
    for y in [-.1,.1]:ax.axhline(y,c='.65',ls=':',lw=.8)
    ax.set(xlabel='Reference Mw, original method',ylabel='NW2 − NW4 Mw');ax.set_title('a  Shared Mw-A reference events',loc='left');grid(ax)
    ax=axes[0,1];tab=pd.read_csv(T/'transfer_controls.csv').iloc[1:]
    for i,row in enumerate(tab.itertuples()):
        ax.scatter(i,row.median_delta,c=BLUE,s=35);ax.text(i,row.median_delta+.012,f'n={row.n}',ha='center',fontsize=8)
    ax.axhline(0,c='.3',lw=.8)
    for y in [-.05,.05]:ax.axhline(y,c='.6',ls=':',lw=.8)
    ax.set(xticks=range(3),xticklabels=['2.0–2.4','2.5–2.9','≥3'],xlabel='Catalog ML of controls',ylabel='Median NW2 − NW4 Mw',ylim=(min(-.08,tab.median_delta.min()-.04),max(.08,tab.median_delta.max()+.04)))
    ax.set_title('b  Fixed transfer gate, by magnitude',loc='left');grid(ax)
    ax=axes[1,0]
    selections=['NW4 reference only','NW4 same reference events','NW2 same reference events','NW2 reference only','NW2 primary']
    labels=['NW4: reference only','NW4: same reference','NW2: same reference','NW2: reference only','NW2: with smaller events']
    for i,label in enumerate(selections):
        row=fit_row(label);ax.errorbar(row.b_at_ML2,i,xerr=[[row.b_at_ML2-row.b_lo],[row.b_hi-row.b_at_ML2]],fmt='o',c=BLUE if i<4 else RED,ms=4,capsize=2)
    previous=pd.read_csv(R/'tables/independent_ML_regressions.csv').query('selection == "Independent ML"').iloc[0]
    ax.axvline(previous.slope,c='.6',lw=.8,ls=':')
    ax.set(yticks=range(5),yticklabels=labels,xlabel='Family-balanced linear slope');ax.invert_yaxis();ax.set_title('c  Matched-method and matched-event controls',loc='left');grid(ax)
    ax=axes[1,1];labels=['Primary','Census only','Exclude pilot','Exclude pilot families','Event design weights','A/B','ML≥2','ML≥2.3','Kim2011 Vs']
    selections=['NW2 primary','NW2 census only','NW2 without original pilot','NW2 without pilot families','NW2 event-design sensitivity','NW2 A/B sensitivity','NW2 floor 2','NW2 floor 2.3','NW2 Kim2011 source Vs']
    for i,label in enumerate(selections):
        method='event design-weighted OLS' if label=='NW2 event-design sensitivity' else 'family-balanced OLS';row=fit_row(label,method=method)
        ax.errorbar(row.b_at_ML2,i,xerr=[[row.b_at_ML2-row.b_lo],[row.b_hi-row.b_at_ML2]],fmt='o',c=RED if i==0 else '.4',ms=3.5,capsize=2)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Linear slope');ax.invert_yaxis();ax.set_title('d  Sensitivity of the extended line',loc='left');grid(ax)
    fig.subplots_adjust(left=.18,wspace=.60)
    save(fig,'03_extension_controls','Control gates were fixed before new Mw results: ≥30 shared Mw-A controls, |median ΔMw|≤0.05 and RMS≤0.10; populated magnitude bins also require |median|≤0.05.\n'
         'Intervals resample whole families; the dotted line in (c) is the earlier 357-event slope. Design-weight sensitivities remain conditional on accepted measurements.')


def coverage():
    fig,axes=frame_figure('What limits the smaller-event measurements?',left=.15)
    ax=axes[0,0];q=D[D.extension_role.eq('census_1p5_1p9')];x=np.arange(5);bottom=np.zeros(5)
    for grade,color in [('A',RED),('B',BLUE),('C',ORANGE),('F','.75')]:
        count=np.array([q[q.ml.eq(m)].Mw_quality.eq(grade).sum() for m in [1.5,1.6,1.7,1.8,1.9]])
        total=np.array([q.ml.eq(m).sum() for m in [1.5,1.6,1.7,1.8,1.9]])
        ax.bar(x,count/total*100,.65,bottom=bottom,color=color,label=grade);bottom+=count/total*100
    ax.set(xticks=x,xticklabels=['1.5','1.6','1.7','1.8','1.9'],xlabel='Catalog ML',ylabel='All census attempts (%)',ylim=(0,100))
    ax.set_title('a  Mw grades in the 2,317-event census',loc='left');ax.legend(title='Mw grade',frameon=False,ncol=4,fontsize=7,loc='upper left');grid(ax)
    ax=axes[0,1];small=D[D.ml.lt(2)];bins=np.arange(.7,3.21,.1)
    for mask,label,color in [(small.ML_quality.eq('A'),'ML A',GRAY),(small.paired_A,'Both A',RED)]:
        ax.hist(small.loc[mask,'ML_Sheen'],bins=bins,histtype='step',lw=1.4,color=color,label=label)
    ax.axvline(1.5,c='.4',ls=':',lw=.8);ax.set(xlabel='Measured ML',ylabel='Events');ax.legend(frameon=False)
    ax.set_title('b  Independent ML and joint acceptance',loc='left');grid(ax)
    ax=axes[1,0];failure=pd.read_csv(T/'failure_reasons.csv');failure=failure[failure.role.ne('reference_MLge2')].groupby('reason')['count'].sum().sort_values().tail(6)
    names={'insufficient_usable_frequency_band':'Usable band too narrow','too_few_usable_triplets':'Too few useful triplets','estimator_sensitivity':'Multitaper / Fourier mismatch',
        'window_sensitivity':'5 / 10 s mismatch','few_common_window_stations':'Too few common-window sites','few_stations':'Too few sites for A',
        'plateau_extrapolated':'Plateau extrapolated','poor_source_fit':'Poor source fit','source_shape_sensitivity':'Source-shape sensitivity'}
    ax.barh(np.arange(len(failure)),failure,color='.55');ax.set_yticks(np.arange(len(failure)),[names.get(v,v) for v in failure.index],fontsize=7.5)
    ax.set(xlabel='Small events failing gate');ax.set_title('c  Multiple failures may apply',loc='left');grid(ax)
    ax=axes[1,1];q=small[small.paired_A]
    ax.scatter(q.ML_Sheen,q.n_HG/q.n_stations_SNR*100,s=14,c=RED,alpha=.55,ec='none')
    ax.set(xlabel='Measured ML',ylabel='Selected Mw sites using HG (%)',ylim=(-3,103));ax.axvline(1.5,c='.5',ls=':',lw=.7)
    ax.set_title('d  Accelerometers retained by the same rules',loc='left');grid(ax)
    fig.subplots_adjust(wspace=.44)
    save(fig,'04_extension_measurement_limits','Mw A: ≥8 useful sites (≥20 frequency samples at SNR≥3 each), ≥8 common-window sites, an observed plateau and ≥one frequency decade.\n'
         '|ΔMw|≤0.10 for Fourier/source-shape/window checks; log10 fit RMS≤0.15. ML A: ≥5 sites; robust scatter≤0.35.\n'
         'Blank magnitudes and every failure remain in the catalog. Success is conditional on detection, confident depth and sufficient signal.')


def spectrum(ax,case,legend=False):
    eid=case.event_id;path=Path(case.candidate_root)/'spectra'/f'{eid}.npz';base=Path(case.baseline_root)/'spectra'/f'{eid}.npz'
    with np.load(path) as archive:
        f=archive['frequency_Hz']
        for method,color,label in [('Multitaper',BLUE,'Multitaper'),('Direct_Fourier',ORANGE,'Direct Fourier')]:
            key='T5.1_ALL_'+method+'_Boatwright'
            if key+'_observed' not in archive:continue
            y=archive[key+'_observed'];good=archive[key+'_good']&np.isfinite(y)
            ax.loglog(f[good],10**y[good],'.',ms=2,color=color,alpha=.6,label=label)
            if method=='Multitaper':ax.loglog(f,10**archive[key+'_predicted'],color=color,lw=1)
    if base.exists():
        with np.load(base) as archive:
            key='T10.1_ALL_Multitaper_Boatwright'
            if key+'_observed' in archive:
                f=archive['frequency_Hz'];good=archive[key+'_good']&np.isfinite(archive[key+'_observed'])
                ax.loglog(f[good],10**archive[key+'_observed'][good],c='.5',lw=.7,alpha=.65,label='Original NW4')
    ax.set(xlim=(.45,22),xlabel='Frequency (Hz)',ylabel='Corrected FAS (m² s)');grid(ax)
    if legend:ax.legend(frameon=False,fontsize=7)


def examples():
    fig,axes=plt.subplots(3,2,figsize=(11.7,10.7));fig.subplots_adjust(left=.10,right=.96,top=.89,bottom=.15,wspace=.30,hspace=.49)
    fig.suptitle('New smaller-event spectra, before any regression selection',fontsize=15,y=.96)
    chosen=[];used=set();small=D[D.confirmatory_small&D.paired_A]
    for i,(lo,hi) in enumerate([(1.5,1.65),(1.65,1.8),(1.8,2.)]):
        q=small[small.ML_Sheen.ge(lo)&small.ML_Sheen.lt(hi)].sort_values('event_id')
        for j in range(2):
            ax=axes[i,j];available=q[~q.family_block.isin(used)]
            if available.empty:
                ax.text(.5,.5,'No additional independent family in bin',ha='center',transform=ax.transAxes);continue
            row=available.iloc[0];used.add(row.family_block);chosen.append(dict(event_id=row.event_id,ML_bin=f'{lo:g}–{hi:g}',selection='First chronological new paired-A event from an unused family'))
            spectrum(ax,row)
            ax.set_title(f'{row.event_id} · {row.n_stations_SNR:g} sites\nML {row.ML_Sheen:.2f}; Mw {row.Mw:.2f}; catalog ML {row.ml:.1f}',loc='left',fontsize=9)
    fig.legend([Line2D([],[],c=BLUE),Line2D([],[],c=ORANGE),Line2D([],[],c='.5')],['NW2 multitaper and fit','NW2 direct Fourier','Original NW4 multitaper'],loc='lower center',bbox_to_anchor=(.53,.085),ncol=3,frameon=False,fontsize=8)
    pd.DataFrame(chosen).to_csv(T/'spectral_example_selection.csv',index=False)
    save(fig,'05_extension_small_event_spectra','First chronological fresh paired-A events from distinct families in measured-ML bins 1.50–1.65, 1.65–1.80 and 1.80–2.00. No selection uses Mw–ML residuals.\n'
         'Points are accepted SNR samples of the stacked spectrum; the line is the primary source fit. The separate atlas shows every accepted small-event case.')


def atlas():
    destination=E/'reports/Accepted_small_event_spectra.pdf';small=D[D.ml.lt(2)&D.paired_A].sort_values(['ML_Sheen','event_id'])
    with PdfPages(destination,metadata={'Title':'Expanded study: every accepted small-event spectrum','Author':'Korean Mw–ML project'}) as pdf:
        for start in range(0,len(small),4):
            fig,axes=plt.subplots(2,2,figsize=(11.7,8.8));fig.subplots_adjust(left=.09,right=.96,top=.86,bottom=.13,wspace=.30,hspace=.45)
            fig.suptitle(f'Accepted smaller events · {start+1}–{min(start+4,len(small))} of {len(small)}',fontsize=14,y=.96)
            for ax,row in zip(axes.flat,small.iloc[start:start+4].itertuples()):
                spectrum(ax,row)
                outside=' · outside fitted ML range' if row.ML_Sheen<1.5 else ''
                ax.set_title(f'{row.event_id} · {row.n_stations_SNR:g} sites{outside}\nML {row.ML_Sheen:.2f}; Mw {row.Mw:.2f}; catalog ML {row.ml:.1f}',loc='left',fontsize=8.5)
            for ax in axes.flat[len(small.iloc[start:start+4]):]:ax.axis('off')
            fig.legend([Line2D([],[],c=BLUE),Line2D([],[],c=ORANGE),Line2D([],[],c='.5')],['NW2 multitaper and fit','NW2 direct Fourier','Original NW4 multitaper'],loc='lower center',bbox_to_anchor=(.53,.06),ncol=3,frameon=False,fontsize=8)
            fig.text(.065,.025,'Every smaller event passing both A-quality checks is included. All attempted events and failures are retained in Mw_ML_catalog.csv.',fontsize=7.7,color='.3')
            pdf.savefig(fig);plt.close(fig)
    return destination


def paragraph(text):return '<p>'+text+'</p>'
def table(head,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def local_magnitude_methods():
    return ('<h1>Independent local magnitude</h1>'+paragraph('<b>ML was measured from the waveforms, separately from the KMA catalog.</b> Of 3,532 attempts, 2,222 have a measured ML: 1,698 grade A and 524 grade B. Among the 2,890 smaller catalog events, 1,614 have measured ML and 1,200 are grade A. All 676 primary regression events have both measured ML and Mw grade A.')+
        table(['CSV field','Meaning'],[['ml','KMA catalog ML; used to define the sampling cohort'],['ML_Sheen','Our Wood–Anderson local magnitude; primary regression x variable'],['ML_quality / ML_Sheen_n','ML quality grade / number of selected stations']])+
        '<h2>From recorded counts to station ML</h2>'+paragraph('1. Remove the dated instrument response to displacement, without a water level, using the 0.1/0.2/25/40-Hz response prefilter. Apply the established causal six-pole 0.5–10-Hz Butterworth filter and revised Wood–Anderson response (magnification 2080). HH, EL and HG channels are eligible.')+
        paragraph('2. Measure horizontal zero-to-peak amplitudes in the origin-time window [R<sub>hyp</sub>/4, R<sub>hyp</sub>/2] seconds. Require an observed P pick with probability ≥0.3 and peak SNR ≥3 on both N and E, relative to the P−6 to P−1-s noise window. Retain stations at epicentral distance 10–100 km.')+
        paragraph('3. Set A<sub>H</sub> = √(A<sub>N</sub>A<sub>E</sub>) in mm and apply the Sheen et al. (2018) relation, with R the <b>epicentral</b> distance in km:')+
        paragraph('<b>ML<sub>station</sub> = log<sub>10</sub>A<sub>H</sub> + 0.5869 log<sub>10</sub>(R/100) + 0.001680(R−100) + 3.</b> Station corrections are zero.')+
        paragraph('4. Use one sensor per network/station, in the established HH→EL→HG order. Event ML is the median of at least three stations. Grade A requires at least five stations and robust scatter 1.4826×MAD ≤0.35; other finite estimates are grade B. Known overlapping events are screened. No step fits or copies the KMA event magnitude.')+
        '<h2>Amplitude audit and a concrete example</h2>'+paragraph('All <b>46,117 retained station measurements</b> reproduce from their saved N/E Wood–Anderson peak amplitudes and distances, with maximum station-ML replay difference 0.000000. Every event median and grade also reproduces. This verifies the archived amplitude-to-magnitude calculation; it does not independently calibrate absolute responses or the regional attenuation scale.')+
        paragraph('For the Sangju-area event on <b>6 April 2025 at 11:06:56 UTC</b> (20:06:56 KST), KMA ML is 1.8 and our measured ML is <b>2.235774</b>, grade A, from 42 stations: 10 HH, 3 EL and 29 HG. The next page shows the individual station magnitudes.')+
        paragraph('The median measured-minus-KMA difference among ML-A cases is +0.191 overall and +0.212 for smaller catalog events. These offsets are observed results, not corrections applied to the measurements. ML and Mw use different amplitude estimators but can share response, path and site errors.')+
        paragraph('Audit files: ML_measurement_audit.csv (every event) and ML_station_measurements.csv (selected station amplitudes, SNR, distance and ML). The earlier Haenam Wood–Anderson recipe and quality rules are retained.'))


def coefficient_page(method):
    content=('<h1>'+('Family-balanced least squares' if method=='OLS' else 'Orthogonal distance regression')+'</h1>'+paragraph('<b>Mw = a + bu + cu²</b>, with u = the specified ML − 2. Linear fits fix c=0. All coefficients below use this same center; six decimal places are retained for reproducibility, not as an accuracy claim.')+
        table(['Training cohort','Events','Families','Spectral method'],[[label,int(comparison_row(cohort).n_events),int(comparison_row(cohort).n_families),scheme] for cohort,label,scheme in [('previous','Previous','NW4; earlier saved measurements'),('expanded_NW4','Expanded NW4','NW4; expanded overlap screen'),('larger_NW2','Larger NW2','NW2; catalog ML≥2 only'),('expanded_NW2','Expanded NW2','NW2; includes smaller events')]]))
    for axis,label in [('our','Measured Wood–Anderson ML'),('KMA','KMA catalog ML')]:
        rows=[]
        for cohort,name in [('previous','Previous'),('expanded_NW4','Expanded NW4'),('larger_NW2','Larger NW2'),('expanded_NW2','Expanded NW2')]:
            for degree in [1,2]:
                r=comparison_row(cohort,axis,method,degree)
                rows.append([name,'Line' if degree==1 else 'Quadratic',f'{r.a:.6f}',f'{r.b:.6f}',f'{r.c:+.6f}' if degree==2 else '0',f'[{r.c_lower:+.4f}, {r.c_upper:+.4f}]' if degree==2 else 'fixed'])
        content+='<h2>'+label+'</h2>'+table(['Cohort','Form','a','b','c','c: 95% interval'],rows)
    content+=paragraph('The same event membership is used for the two ML definitions within each cohort. Previous coefficients and joint bootstrap samples reproduce the earlier saved results after changing their center from ML 3 to ML 2. No previous magnitude or regression is replaced.')
    if method=='OLS':
        content+=paragraph('Each family has total weight one. Intervals resample whole families (2,000 draws). NW4 versus NW2 changes both estimator/window and potentially accepted membership; the two NW2 cohorts hold the method fixed while adding smaller events.')
    else:
        content+=paragraph('ODR assigns equal error variance to the two magnitude axes and equal total weight per family. This is a stated sensitivity assumption, not a measured event-by-event error model. Intervals use 1,000 whole-family bootstrap draws. Primary predictive comparisons use OLS; ODR predictions are retained in the CSV tables.')
    return content


def narratives():
    small=D[D.ml.lt(2)];pairs=small[small.paired_A];fresh=pairs[pairs.confirmatory_small];low=fresh[fresh.ML_Sheen.ge(1.5)&fresh.ML_Sheen.lt(2)]
    r=fit_row();odr=fit_row(method='equal-axis ODR');transfer=SUMMARY['transfer'];ctl=transfer['all_controls'];bin_fit=objective_row();mp_fit=POTENCY.fit()
    first=('<h1>Extending Korean ML, Mw and Mp</h1>'+paragraph(f'<b>The expansion gives {len(pairs)} smaller catalog events passing both A-quality checks, including {len(fresh)} outside the original pilot. Of the new accepted events, {len(low)} have independently measured ML 1.5–2.0.</b> Mw and scalar potency magnitude Mp are compared with both measured and KMA ML. The earlier adopted results and pilot are preserved.')+
        '<h2>Fixed cohort and method</h2>'+paragraph('All 2,317 confident-depth catalog-ML 1.5–1.9 events were attempted. Below that, a nested random sample totals 519 ML 1.0–1.4 events, capped at 60 per depth/era cell; the original 54 ML 0.5–0.9 pilot events are retained. The 642 ML≥2 reference events were also assessed with the same method. All 3,532 cases had retained waveforms; no failed event was replaced.')+
        paragraph('Primary spectra use 5.1/10.1-s S windows, NW=2, three tapers and 0.5–20 Hz. The original 10.1/20.1-s NW4 method is the control; the two methods have nearly matched primary half-bandwidths, about 0.39 Hz. HH, EL and HG, dated responses, the same path/Q corrections, Kim & Kim (1983) Vs at source depth, density 2800 kg/m³ and Mw constant 9.05 are retained. Independent Wood–Anderson ML uses the published Sheen et al. (2018) scale and the established 10–100-km station selection. All known domestic neighbors are screened; 126 reference cases required reprocessing under this expanded overlap screen.')+
        '<h2>All attempted events remain visible</h2>'+table(['Catalog ML','Attempted','Mw A','Both A','Both A/B'],[[label,len(q),int(q.Mw_quality.eq('A').sum()),int(q.paired_A.sum()),int(q.paired_A_or_B.sum())] for label,q in [('0.5–0.9',D[D.ml.lt(1)]),('1.0–1.4',D[D.ml.ge(1)&D.ml.lt(1.5)]),('1.5–1.9',D[D.ml.ge(1.5)&D.ml.lt(2)]),('≥2 reference',D[D.ml.ge(2)])]])+
        paragraph(f'The fixed transfer check {"passes" if transfer["passed"] else "does not pass"}: {ctl["n"]} shared Mw-A reference events have median NW2−NW4 Mw {ctl["median_delta"]:+.3f}, RMS {ctl["RMS_delta"]:.3f}. Magnitude-bin checks are shown separately. No empirical amplitude offset is fitted.')+
        '<h2>Two explicit fitting objectives</h2>'+paragraph(f'The family-weighted reference line is <b>Mw = {r.a_at_ML2:.3f} + {r.b_at_ML2:.3f}(ML − 2)</b>, using {int(r.n_events)} paired-A events in {int(r.n_blocks)} families at measured ML≥1.5. Its slope interval is {r.b_lo:.3f}–{r.b_hi:.3f}; family ODR gives {odr.b_at_ML2:.3f}.')+
        paragraph(f'<b>Matched bin-median curves across the magnitude range:</b> with u=measured ML−2, <b>Mw = {bin_fit.a:.6f} + {bin_fit.b:.6f}u + {bin_fit.c:.6f}u²</b> and <b>Mp = {mp_fit.a:.6f} + {mp_fit.b:.6f}u + {mp_fit.c:.6f}u²</b>. Equal occupied-bin weights restore emphasis on sparse large events. Both quadratics improve the observed large-event prediction comparison. The following section defines Mp and its reference rigidity.')+
        paragraph('Acceptance does not use agreement between Mw and ML. Catalog ML determines the selection; independently measured ML is the regression variable. The presence of catalog ML below 1 does not establish a reliable relation at measured ML below 1.5.')+
        paragraph('<b>Report guide:</b> independent ML is followed by the joint Mw/Mp analysis: definitions, equations, methods, large events, depth, velocity and Hong comparisons. The later sections retain detailed Mw weighting/transfer controls and spectra. Potency_magnitude_and_quadratic_scaling.pdf extracts the joint section; the regression comparison supplies the matched equations and fitting-objective audit.'))
    rows=[]
    for model,label in [('linear','Line'),('quadratic','Quadratic'),('hinge2','Knot at ML 2')]:
        q=fit_row(model=model)
        c=comparison_row('expanded_NW2',degree=2) if model=='quadratic' else None
        interval='0 (fixed)' if model=='linear' else (f'{c.c:+.6f} [{c.c_lower:+.3f}, {c.c_upper:+.3f}]' if c is not None else f'{q.c:+.4f} [{q.c_lo:+.3f}, {q.c_hi:+.3f}]')
        rows.append([label,interval,f'{q.CV_RMS_family_new_small:.5f}',f'{q.CV_RMS_family_new_small_ML1p5_2:.5f}'])
    comparison=pd.read_csv(T/'model_prediction_comparison.csv')
    delta=comparison[comparison.selection.eq('New small, ML1.5–2')]
    supported=delta[delta.upper.lt(0)]
    sensors=SUMMARY.get('small_event_sensor_check',{})
    if sensors.get('event_station_band_pairs',0):
        matched=next((r for r in sensors['bands'] if r['band']=='1–5 Hz'),None)
        sensor_note=(f' Co-located HH/HG at 1–5 Hz gives {matched["event_station_pairs"]} jointly usable event/site pairs; the equal-station median HG/HH ratio is {matched["station_balanced_median_HG_over_HH"]:.3f}. This is a conditional sensor diagnostic, without a fitted amplitude correction.' if matched else
            ' Co-located HH/HG diagnostics are listed separately for frequency bands with jointly usable records; no amplitude correction is fitted.')
    else:
        sensor_note=' This retained small-event sample supplies no co-located HH/HG pair with sufficient jointly usable spectral samples for the new check. The earlier dedicated sensor audit is not a direct lower-magnitude paired-record test.'
    conclusion=('At least one alternative reduces the paired family-level held-out squared error with its bootstrap interval below zero. This supports predictive improvement in this selected sample; it does not identify a physical scaling break.' if len(supported) else
        'Under family weights, these expanded models predict the new small events almost identically; neither alternative improves that test. Both curvature and the slope change at ML 2 have intervals spanning zero. Bin-median quadratics address the different goal of fitting across the magnitude range for both Mw and Mp.')
    census=fit_row('NW2 census only');no_pilot=fit_row('NW2 without pilot families');kim=fit_row('NW2 Kim2011 source Vs')
    weighted=fit_row('NW2 event-design sensitivity',method='event design-weighted OLS')
    low_test=delta.iloc[0]
    second=('<h1>Regression checks and interpretation</h1>'+paragraph('The primary expanded fits use uniform NW2 magnitudes. Historical NW4 curves retain their original measurements. The same-reference-event comparison separates the estimator change from changes in accepted membership; the matched NW2 cohorts isolate adding smaller events. No event-wise choice between methods or pooling by favorable Mw–ML residual is allowed.')+
        '<h2>Family-weighted prediction on new smaller events</h2>'+table(['Model','c [95% family interval]','New small RMS','New ML 1.5–2 RMS'],rows)+
        paragraph('With u=ML−2, the quadratic is a+bu+cu²; the fixed-knot model is a+bu+c min(u,0). Thus c is curvature for the quadratic and a low-ML slope increment for the knot model.')+
        paragraph(f'Errors above withhold entire families, including their reference members. The lower-range test contains {int(low_test.n_events)} new events in {int(low_test.n_blocks)} families and excludes the original pilot from scoring. Coefficient intervals use 2,000 family-bootstrap draws; paired held-out squared-error differences use 5,000 family resamples. '+conclusion)+
        paragraph('The earlier family-weighted measured-ML quadratic has c=+0.030111; the expanded family result is −0.000095. This flattening is not shared by bin fits: expanded measured-ML bin OLS gives +0.037806 and bin ODR +0.037289. KMA ML also differs: its expanded family curvature is +0.062359 (interval +0.0301 to +0.1038). The objective audit distinguishes family, event and bin weights; the joint Mw/Mp section applies the same methods to both quantities.')+
        '<h2>Sensitivity checks</h2>'+paragraph(f'The linear slope is {census.b_at_ML2:.3f} for the fully enumerated catalog-ML≥1.5 cohort, {no_pilot.b_at_ML2:.3f} after removing all pilot families and {kim.b_at_ML2:.3f} with Kim2011 source Vs. Event design weights give {weighted.b_at_ML2:.3f}; the family-bootstrap intervals overlap the primary result. The tables also retain ODR/Deming comparisons, A/B inclusion, within-family sampling weights and measured-ML floors 1.7, 2.0 and 2.3. Inverse-probability weights address the known draw, not unmeasurable events. No operation selects an event by Mw–ML agreement.')+
        '<h2>Coverage and uncertainty</h2>'+paragraph('This is conditional on original detection/picking and G2/G3 depth selection. The complete catalog-ML 1.5–1.9 census gives an exact measurement-success fraction for that eligible cohort; it is not a national detection-completeness estimate. Lower-magnitude cells have the recorded sampling probabilities. Independent-ML and Mw failures remain in the all-attempt catalog. Response/SNR/window gates were not relaxed.'+sensor_note)+
        paragraph('Station-resampling intervals hold the fitted source shape and source/path model fixed. They exclude absolute response, velocity, attenuation, radiation and site-model errors. Family-bootstrap regression intervals describe sampling variation among accepted families; they do not include systematic magnitude-scale uncertainty or recover missing events. Kim et al. (2011) source-velocity comparisons remain in the catalog.')+
        '<h2>Files and scope</h2>'+paragraph('potency_event_catalog.csv retains all 3,532 attempts with ML, Mw, scalar potency and Mp; missing Mw remains missing Mp. magnitude_scaling_* contains 480 matched Mw/Mp cases; potency_depth_* contains 108 depth controls. Earlier Mw-only tables and the 80-case objective audit remain available. The atlas includes every paired-A smaller event; measured ML below 1.5 is outside the fitted range.')+
        paragraph('Computational validation checks the fixed draw, complete event accounting, all quality gates, reuse hashes and all-known-event overlap policy, source-depth Vs/C=9.05, matched ML values, direct weighted regression and whole-family holdouts. Protected checksums retain the earlier catalog and PDF reports. A passing transfer check is evidence of consistency between methods, not proof of absolute small-event accuracy.'))
    return first,second


def main():
    global D,OLD,FIT,BANDS,SUMMARY,C_FIT,C_CURVE,C_METRIC,C_STATION,O_FIT,O_METRIC,O_BINS,O_WEIGHT,POTENCY
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-atlas',action='store_true',help='Reuse the existing spectral atlas after checking its saved checksum')
    args=parser.parse_args()
    prior=read(T/'report_validation.json') if args.reuse_atlas else None
    SUMMARY=read(T/'summary.json');assert SUMMARY['complete'] and SUMMARY['attempted']==3532
    D=csv(T/'NW2_Mw_ML_catalog.csv');OLD=csv(T/'NW4_Mw_ML_catalog.csv')
    FIT=pd.read_csv(T/'regressions.csv');BANDS=pd.read_csv(T/'regression_bands.csv')
    comparison=read(T/'regression_comparison_validation.json');assert comparison['passed']
    assert comparison['source_sha256']==sha(R/'code/compare_extension_regressions.py')
    for name,digest in comparison['outputs_sha256'].items():assert sha(T/name)==digest,name
    for name,digest in comparison['protected_inputs_sha256'].items():assert sha(R/name)==digest,name
    C_FIT=pd.read_csv(T/'regression_comparison_models.csv');C_CURVE=pd.read_csv(T/'regression_comparison_curves.csv')
    C_METRIC=pd.read_csv(T/'regression_comparison_performance.csv');C_STATION=csv(T/'ML_station_measurements.csv')
    audit=read(T/'fit_objective_validation.json');assert audit['passed']
    assert audit['source_sha256']==sha(R/'code/audit_extension_fit_objectives.py')
    for name,digest in audit['outputs_sha256'].items():assert sha(T/name)==digest,name
    for name,digest in audit['protected_inputs_sha256'].items():assert sha(R/name)==digest,name
    O_FIT=csv(T/'fit_objective_models.csv');O_METRIC=csv(T/'fit_objective_performance.csv')
    O_BINS=csv(T/'fit_objective_bins.csv');O_WEIGHT=csv(T/'fit_objective_weights.csv')
    sampling();scaling();controls();coverage();examples();local_magnitude_results();regression_history();regression_prediction();objective_curves();objective_diagnosis()
    POTENCY=PotencySection(save,frame_figure,grid);potency_pages=POTENCY.build()
    first,last=narratives();document=fitz.open();scales=[]
    css='body{font-family:sans-serif;font-size:9.5pt;line-height:1.26;color:#222;}h1{font-size:18pt;margin:0 0 11pt;}h2{font-size:11pt;margin:11pt 0 5pt;}p{margin:0 0 8pt;}table{width:100%;border-collapse:collapse;font-size:8.8pt;margin:8pt 0;}td,th{padding:5pt 4pt;border-bottom:.5pt solid #ccd3d8;text-align:left;}th{background:#edf2f5;}'
    def insert(text):
        page=document.new_page(width=595.276,height=841.89)
        spare,scale=page.insert_htmlbox(fitz.Rect(40,28,555,812),text,css=css,scale_low=.94)
        assert spare>=0,'Narrative exceeds page';scales.append(scale)
    def insert_figure(stem):
        with fitz.open(F/(stem+'.pdf')) as part:document.insert_pdf(part)
    insert(first)
    insert(local_magnitude_methods());insert_figure('06_independent_local_magnitude');insert_figure('01_extension_sampling')
    for kind,content,title in potency_pages:
        if kind=='text':insert(content)
        else:insert_figure(content)
    insert_figure('02_extension_scaling');insert_figure('07_previous_and_expanded_regressions');insert(coefficient_page('OLS'))
    insert_figure('08_regression_prediction_comparison');insert(coefficient_page('ODR'))
    insert(objective_narrative());insert_figure('09_extension_fit_objectives');insert_figure('10_extension_weight_and_prediction')
    for stem in ['03_extension_controls','04_extension_measurement_limits','05_extension_small_event_spectra']:insert_figure(stem)
    insert(last)
    titles=['Joint magnitude study: design and findings','Independent ML: method and amplitude audit','Independent ML: measurements','Cohort and coverage']+[entry[2] for entry in potency_pages]+['Family-weighted Mw fits','Previous and expanded Mw family curves','Mw family OLS equations','Mw family predictions','Mw family ODR equations','Fitting large events: objective audit','Previous and expanded Mw bin fits','Weight dilution and prediction','Spectral-method and event controls','Measurement limits','New smaller-event spectra','Regression checks and interpretation']
    document.set_toc([[1,title,i+1] for i,title in enumerate(titles)])
    document.set_metadata({'title':'Extending Korean ML, Mw and Mp toward ML 1.5','author':'Korean Mw–ML project'})
    destination=E/'reports/Extended_Mw_ML_scaling.pdf';document.save(destination,garbage=4,deflate=True)
    companion=E/'reports/Linear_quadratic_Mw_ML_comparison.pdf'
    joint_start=4;joint_end=joint_start+len(potency_pages)-1;family_start=joint_end+1;audit_start=family_start+5
    with fitz.open() as part:
        for first_page,last_page in [(joint_start,joint_start+5),(audit_start,audit_start+2),(family_start,family_start+4)]:part.insert_pdf(document,from_page=first_page,to_page=last_page)
        order=list(range(joint_start,joint_start+6))+list(range(audit_start,audit_start+3))+list(range(family_start,family_start+5))
        part.set_toc([[1,titles[i],j+1] for j,i in enumerate(order)])
        part.set_metadata({'title':'Linear and quadratic ML–Mw–Mp: previous and expanded fits','author':'Korean Mw–ML project'})
        part.save(companion,garbage=4,deflate=True)
    potency_path=E/'reports/Potency_magnitude_and_quadratic_scaling.pdf'
    with fitz.open() as part:
        part.insert_pdf(document,from_page=joint_start,to_page=joint_end)
        part.set_toc([[1,titles[i],i-joint_start+1] for i in range(joint_start,joint_end+1)])
        part.set_metadata({'title':'Expanded Korean potency magnitude and quadratic scaling','author':'Korean Mw–ML project'})
        part.save(potency_path,garbage=4,deflate=True)
    document.close()
    atlas_path=E/'reports/Accepted_small_event_spectra.pdf'
    if args.reuse_atlas:
        assert prior['passed'] and sha(atlas_path)==prior['reports'][str(atlas_path.relative_to(R))]['sha256']
    else:atlas_path=atlas()
    reports={}
    for path in [destination,companion,potency_path,atlas_path]:
        with fitz.open(path) as doc:
            if path==destination:assert len(doc)==27
            elif path==companion:assert len(doc)==14
            elif path==potency_path:assert len(doc)==11
            else:assert len(doc)==int(np.ceil((D.ml.lt(2)&D.paired_A).sum()/4))
            for i,page in enumerate(doc):
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        for span in line.get('spans',[]):
                            x0,y0,x1,y1=span['bbox']
                            assert min(x0,y0)>=0 and x1<=page.rect.width and y1<=page.rect.height,(path.name,i,span)
                if path==destination:page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(f'/tmp/korean_mw_ml_extension_{i+1}.png')
                if path==atlas_path and i==0:page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save('/tmp/korean_mw_ml_extension_atlas.png')
            reports[str(path.relative_to(R))]=dict(pages=len(doc),sha256=sha(path))
    for filename,digest in read(E/'inputs/protected_results.json').items():assert sha(R/filename)==digest,filename
    validation=dict(passed=True,reports=reports,narrative_scales=scales,
        PDF_page_bounds_passed=True,all_accepted_small_spectra_in_atlas=True,original_results_preserved=True,
        source_sha256=sha(Path(__file__)),summary_sha256=sha(T/'summary.json'),figures=FIGURES,
        comparison_validation_sha256=sha(T/'regression_comparison_validation.json'),spectral_atlas_reused=args.reuse_atlas,
        objective_validation_sha256=sha(T/'fit_objective_validation.json'),
        potency_analysis_validation_sha256=sha(T/'potency_analysis_validation.json'),
        potency_controls_validation_sha256=sha(T/'potency_controls_validation.json'),
        potency_report_source_sha256=sha(R/'code/report_extension_potency.py'))
    (T/'report_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='figures'},indent=2),flush=True)


if __name__=='__main__':main()
