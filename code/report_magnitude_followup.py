"""Append the four requested magnitude follow-ups to the existing potency report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import fitz
from potency_regression import prediction, binned_medians

R=Path(__file__).resolve().parents[1]
T=R/'tables'
BLUE='#176da0'; ORANGE='#bd591b'; GRAY='#777777'


def append_followup(doc):
    g=pd.read_csv(T/'magnitude_followup_events.csv',dtype={'event_id':str})
    fits=pd.read_csv(T/'magnitude_followup_regressions.csv')
    metrics=pd.read_csv(T/'magnitude_followup_prediction_metrics.csv')
    comparisons=pd.read_csv(T/'magnitude_followup_CV_comparison.csv')
    tail=pd.read_csv(T/'magnitude_followup_large_events.csv',dtype={'event_id':str})
    depth=pd.read_csv(T/'magnitude_followup_depth_regressions.csv')
    boot=np.load(T/'magnitude_followup_bootstrap.npz')
    depth_boot=np.load(T/'magnitude_followup_depth_bootstrap.npz')
    rigidity_audit=json.loads((T/'potency_rigidity_literature_audit.json').read_text())
    figures=[]
    def fit(x='ML_Sheen',y='Mw',degree=2,method='OLS',weight='family',bins=50,minimum=1):
        q=fits[fits.x.eq(x)&fits.y.eq(y)&fits.degree.eq(degree)&fits.method.eq(method)&
               fits.weighting.eq(weight)&fits.n_bins.eq(bins)&fits.minimum.eq(minimum)]
        assert len(q)==1
        return q.iloc[0]
    def coefficient(f):return np.array([f.a_at_ML3,f.slope_at_ML3,f.quadratic_c])
    def stat(x='ML_Sheen',y='Mw',degree=2,subset='All'):
        q=metrics[metrics.x.eq(x)&metrics.y.eq(y)&metrics.degree.eq(degree)&metrics.subset.eq(subset)&
                  metrics.method.eq('OLS')&metrics.weighting.eq('family')]
        assert len(q)==1
        return q.iloc[0]
    def comparison(x='ML_Sheen',y='Mw',subset='All'):
        q=comparisons[comparisons.x.eq(x)&comparisons.y.eq(y)&comparisons.subset.eq(subset)&
                      comparisons.method.eq('OLS')&comparisons.weighting.eq('family')]
        assert len(q)==1
        return q.iloc[0]
    def dep(y='Mp_Liu_Kim1983',x='ML_Sheen',control='ML quadratic'):
        q=depth[depth.x.eq(x)&depth.y.eq(y)&depth.control.eq(control)]
        assert len(q)==1
        return q.iloc[0]
    def save(fig,stem,caption):
        fig.text(.075,.045,caption,fontsize=8.5,color='.25',linespacing=1.4)
        for ext in ['pdf','png']:fig.savefig(R/'figures'/(stem+'.'+ext),dpi=240)
        plt.close(fig);figures.append(dict(stem=stem,caption=caption))

    fig,axes=plt.subplots(2,2,figsize=(11,9.5),gridspec_kw={'height_ratios':[1.6,1]})
    fig.subplots_adjust(left=.1,right=.94,top=.87,bottom=.18,wspace=.34,hspace=.36)
    fig.suptitle('Moment magnitude and Korean local magnitude',fontsize=15,y=.96)
    for j,(xkey,label) in enumerate([('ML_Sheen','Measured ML'),('ML_KMA','KMA ML')]):
        linear=fit(x=xkey,degree=1);quadratic=fit(x=xkey)
        x=np.linspace(g[xkey].min(),g[xkey].max(),250)
        ax=axes[0,j]
        sc=ax.scatter(g[xkey],g.Mw,c=g.z_central,cmap='viridis',norm=Normalize(0,30),s=18,alpha=.65,edgecolors='none')
        ax.plot(x,prediction(coefficient(linear),x),c=BLUE,ls='--',lw=1.2,label='Linear, family weights')
        ax.plot(x,prediction(coefficient(quadratic),x),c=BLUE,lw=1.6,label='Quadratic, family weights')
        med,_=binned_medians(g[xkey].to_numpy(),g.Mw.to_numpy(),50)
        ax.scatter(med[:,0],med[:,1],s=23,marker='s',facecolor='none',edgecolor='.3',lw=.8,label='Filtered bin medians')
        ax.set(xlabel=label,ylabel='Mw',xlim=(1.75,5.6),ylim=(1.9,5.4))
        ax.grid(color='.94');ax.legend(frameon=False,loc='upper left')
        ax.text(.04,.05,f'n = 357; quadratic c = {quadratic.quadratic_c:.3f}',transform=ax.transAxes,fontsize=9)
        bx=axes[1,j]
        bx.scatter(g[xkey],g.Mw-prediction(coefficient(linear),g[xkey]),c=g.z_central,
                   cmap='viridis',norm=Normalize(0,30),s=17,alpha=.65,edgecolors='none')
        samples=boot[quadratic.case_id]
        qcurve=samples[:,0,None]+samples[:,1,None]*(x-3)+samples[:,2,None]*(x-3)**2
        lo,hi=np.quantile(qcurve,[.025,.975],axis=0)-prediction(coefficient(linear),x)
        bx.fill_between(x,lo,hi,color=BLUE,alpha=.15,lw=0)
        bx.plot(x,prediction(coefficient(quadratic),x)-prediction(coefficient(linear),x),c=BLUE,lw=1.5)
        bx.axhline(0,c=GRAY,ls='--',lw=.8)
        bx.set(xlabel=label,ylabel='Residual from linear fit',xlim=(1.75,5.6),ylim=(-.42,.42))
        bx.grid(color='.94')
        bx.text(.04,.9,f'95% interval for c: {quadratic.c_lo:+.3f} to {quadratic.c_hi:+.3f}',transform=bx.transAxes,fontsize=9)
    fig.colorbar(sc,ax=axes,location='right',fraction=.025,pad=.025,label='Depth (km)')
    save(fig,'20_Mw_ML_scaling',
         '357 identical paired-A events; 194 families. Mw uses C=9.05 and the Haenam Kim & Kim (1983) source model.\n'
         'The fit weights, bootstrap blocks, bins and prediction tests match the potency analysis. Shading resamples whole families.\n'
         'These are selected-sample relations; the published Liu potency curve is not a moment-magnitude calibration.')

    fig,axes=plt.subplots(2,2,figsize=(11,9.5),gridspec_kw={'height_ratios':[1.15,1]})
    fig.subplots_adjust(left=.1,right=.96,top=.88,bottom=.23,wspace=.28,hspace=.42)
    fig.suptitle('Large events: fit and held-out prediction',fontsize=15,y=.96)
    q=g[g.ML_Sheen>4].sort_values('origin_utc')
    x=np.linspace(3.8,g.ML_Sheen.max(),200)
    for j,(y,label) in enumerate([('Mp_Liu_Kim1983','MP'),('Mw','Mw')]):
        ax=axes[0,j]
        for degree,style,legend in [(1,'--','Linear'),(2,'-','Quadratic')]:
            ax.plot(x,prediction(coefficient(fit(y=y,degree=degree)),x),c=BLUE,ls=style,lw=1.6,label=legend)
        ax.scatter(q.ML_Sheen,q[y],c=BLUE,s=35,zorder=5)
        for i,z in enumerate(q.itertuples()):
            ax.annotate(str(i+1),(z.ML_Sheen,getattr(z,y)),xytext=(5,6),textcoords='offset points',fontsize=9)
        ax.set(xlabel='Measured ML',ylabel=label,xlim=(3.8,5.5),ylim=(3.4,5.25))
        ax.grid(color='.94');ax.legend(frameon=False,loc='upper left')
        ax.set_title('Curves fitted to all 357 events',loc='left',fontsize=11)
        bx=axes[1,j]
        p=tail[tail.x.eq('ML_Sheen')&tail.y.eq(y)].set_index('event_id').loc[q.event_id]
        idx=np.arange(len(q));width=.35
        bx.bar(idx-width/2,abs(p.linear_residual),width,color='#aab4bc',label='Linear')
        bx.bar(idx+width/2,abs(p.quadratic_residual),width,color=BLUE,label='Quadratic')
        labels=[f'{i+1}\n{s[:4]}-{s[4:6]}-{s[6:8]}' for i,s in enumerate(q.event_id)]
        bx.set(xticks=idx,xticklabels=labels,ylabel='Absolute prediction error',ylim=(0,.35))
        bx.tick_params(axis='x',labelsize=7.4);bx.grid(axis='y',color='.94');bx.set_axisbelow(True)
        bx.set_title('Whole families held out',loc='left',fontsize=11);bx.legend(frameon=False)
        a=stat(y=y,degree=1,subset='ML > 4');b=stat(y=y,degree=2,subset='ML > 4')
        bx.text(.03,.9,f'Event RMSE: {a.CV_RMSE_events:.3f} → {b.CV_RMSE_events:.3f}',transform=bx.transAxes,fontsize=9)
    save(fig,'21_large_event_fit_comparison',
         'Six events with measured ML > 4, in five independent families. Dates are UTC. Events 2 and 3 are in the Gyeongju family.\n'
         'The quadratic improves large-event prediction, while changes below ML 4 are small. The five-family gain interval includes zero.\n'
         'The highest event is outside the training magnitude range when its entire family is held out. KMA ML > 4 gives only three events.')

    fig,axes=plt.subplots(2,2,figsize=(11,9.5))
    fig.subplots_adjust(left=.1,right=.94,top=.87,bottom=.2,wspace=.3,hspace=.35)
    fig.suptitle('Depth dependence of magnitude residuals',fontsize=15,y=.96)
    zgrid=np.linspace(5,24,200);zs=(zgrid-15)/10
    for i,(y,label,other) in enumerate([('Mp_Liu_Kim1983','MP','Mp_Liu_Kim2011'),('Mw','Mw','Mw_source_Vs_Kim2011')]):
        for j,control in enumerate(['Raw magnitude difference','ML quadratic']):
            ax=axes[i,j];rr=dep(y=y,control=control);beta=np.array(json.loads(rr.coefficients))
            if j==0:
                response=g[y]-g.ML_Sheen
                line=beta[0]+beta[-1]*zs
                samples=depth_boot[rr.case_id][:,0,None]+depth_boot[rr.case_id][:,-1,None]*zs
            else:
                m=g.ML_Sheen-3
                response=g[y]-(beta[0]+beta[1]*m+beta[2]*m*m)
                line=beta[-1]*zs;samples=depth_boot[rr.case_id][:,-1,None]*zs
            sc=ax.scatter(g.z_central,response,c=g.ML_Sheen,cmap='viridis',norm=Normalize(2,5.5),
                          s=17,alpha=.55,edgecolors='none')
            lo,hi=np.quantile(samples,[.025,.975],axis=0)
            ax.fill_between(zgrid,lo,hi,color=BLUE,alpha=.13,lw=0)
            ax.plot(zgrid,line,c=BLUE,lw=1.5,label='Kim & Kim 1983')
            alt=dep(y=other,control=control);bb=np.array(json.loads(alt.coefficients))
            ax.plot(zgrid,bb[-1]*zs+(bb[0] if j==0 else 0),c=ORANGE,ls='--',lw=1.2,label='Kim et al. 2011')
            for lower,upper in [(5,10),(10,15),(15,20),(20,25)]:
                keep=g.z_central.ge(lower)&g.z_central.lt(upper)
                ax.scatter(g.loc[keep,'z_central'].median(),response[keep].median(),marker='D',s=33,
                           facecolors='white',edgecolors='.25',lw=.8,zorder=5)
            ax.axvline(15,c='.6',ls=':',lw=.8);ax.axhline(0,c='.7',lw=.7,zorder=0)
            ax.set(xlabel='Source depth (km)',ylabel=(label+' − measured ML' if j==0 else label+' after ML adjustment'),xlim=(4.5,24.5))
            ax.grid(color='.95')
            ax.set_title('Raw difference' if j==0 else 'Joint quadratic ML + depth model',loc='left',fontsize=10.5)
            ax.text(.03,.96,f'{rr.depth_slope_per_10km:+.3f} per 10 km [{rr.depth_lo:+.3f}, {rr.depth_hi:+.3f}]',
                    transform=ax.transAxes,va='top',fontsize=8.8)
            if i==0 and j==0:ax.legend(frameon=False,loc='lower left',fontsize=8)
    fig.colorbar(sc,ax=axes,location='right',fraction=.025,pad=.025,label='Measured ML')
    save(fig,'22_magnitude_depth_residuals',
         '357 paired-A events. Right: remove the ML polynomial estimated jointly with depth; the remaining slope is conditional on ML.\n'
         'Shading: 95% whole-family coefficient intervals. Diamonds: 5-km-bin medians. Dotted: Kim & Kim boundary at 15 km.\n'
         'The decline is concentrated at 5–15 km; a slope within the 15–24 km subset is unresolved. Depth is not established as the cause.')

    def para(s):return '<p>'+s+'</p>'
    def table(head,rows):
        return '<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    def interval(a,b):return f'{a:+.3f} to {b:+.3f}'
    tailtable=[]
    for y,label in [('Mp_Liu_Kim1983','MP'),('Mw','Mw')]:
        l=stat(y=y,degree=1,subset='ML > 4');q=stat(y=y,degree=2,subset='ML > 4')
        c=comparison(y=y,subset='ML > 4')
        tailtable.append([label,f'{l.CV_RMSE_events:.4f}',f'{q.CV_RMSE_events:.4f}',
                          f'{c.gain_family_RMSE:+.4f}',interval(c.gain_lo,c.gain_hi)])
    p=fit(y='Mp_Liu_Kim1983');mw=fit();ml=fit(degree=1)
    shallow=dep(control='Depth below 15 km');deep=dep(control='Depth at least 15 km')
    prim=dep();other=dep(y='Mp_Liu_Kim2011');fixed=dep(y='Mp_Liu_fixed_Vs3500')
    trendtable=[]
    for y,label in [('Mp_Liu_Kim1983','MP, Kim & Kim 1983'),('Mp_Liu_Kim2011','MP, Kim 2011'),
                    ('Mw','Mw, Kim & Kim 1983'),('Mw_source_Vs_Kim2011','Mw, Kim 2011')]:
        rr=dep(y=y);trendtable.append([label,f'{rr.depth_slope_per_10km:+.3f}',interval(rr.depth_lo,rr.depth_hi)])
    mwtable=[]
    for x,label in [('ML_Sheen','Measured ML'),('ML_KMA','KMA ML')]:
        ll=fit(x=x,degree=1);qq=fit(x=x);cc=comparison(x=x)
        mwtable.append([label,f'{qq.quadratic_c:.4f}',interval(qq.c_lo,qq.c_hi),
                        f'{ll.CV_RMS_families:.5f}',f'{qq.CV_RMS_families:.5f}'])
    layer_mu=[v['mu_GPa'] for v in rigidity_audit['source_rigidity_layers'] if v['model']=='Kim1983']
    offsets={v['ML']:v['median'] for v in rigidity_audit['offset_statistics']}
    liu_depth=next(v for v in rigidity_audit['Liu_S7_analogue'] if v['ML']=='ML_Sheen' and v['selection']=='Paired A')
    normalized=('<h2>2. What the magnitude numbers mean</h2>'+para(
        'The original Trugman and Ben-Zion (2024) MP scale is designed to equal Mw at a reference rigidity of 36 GPa. '
        'Their author project report, pp. 4–5, states this choice and documents ODR on bin medians. The supplied Liu (2026) paper retains that MP definition but uses binned ordinary least squares. '
        'Trugman and Ben-Zion (2026), equation 5, instead use a 28-GPa reference representative of their model suite. All are Mw-like physical size scales; none is defined to equal local magnitude across sizes or regions.')+
        para('With the same C convention, MP=Mw+(2/3)log10(μref/μ). For Liu MP versus our C=9.05 Mw, '
        'MP≈Mw+(2/3)log10(36 GPa/μ)−0.03333. Thus the numerical comparison requires the reference rigidity and C to be stated. '
        'Liu’s empirical curve gives MP=2.405 at ML=2 and MP=3.569 at ML=3.5. Approximate agreement around moderate magnitudes does not imply an identity with ML.')+
        para(f'For the plotted Korean sample, physical source rigidity μsrc=ρVs(z)² is {layer_mu[0]:.3f} GPa at depths less than 15 km and {layer_mu[1]:.3f} GPa at 15–32 km, with ρ=2800 kg/m³. '
        'The plotted MP then uses Liu’s exact reference normalization, equivalent to 36 GPa at C=9.10. It does not use the alternative 28-GPa columns.')+
        para(f'Consequently, a reference-rigidity mismatch does not explain our offset from Liu: median MP minus the published curve is {offsets["ML_Sheen"]:+.3f} for measured ML and {offsets["ML_KMA"]:+.3f} for KMA ML. '
        f'Changing between our source models changes MP by a median absolute {rigidity_audit["two_model_MP_median_absolute_difference"]:.3f}. Differences in ML calibration, P- versus S-wave source estimation, attenuation/site corrections and event selection remain to be resolved; the pooled published curve is not a guaranteed regional calibration.'))
    pages=[]
    pages.append('<h1>Large events and magnitude scales</h1>'+para(
        'Follow-up to the initial six pages. All comparisons retain the same 357 paired-A events, 194 families, accepted waveforms and adopted magnitude values.')+
        '<h2>1. The quadratic predicts the large-event subset better</h2>'+para(
        'There are six events with measured ML&gt;4, in five independent families; two are in the 2016 Gyeongju family. '
        'The quadratic improves their fitted residuals and their predictions when entire families are held out. '
        'This subset result deserves separate reporting: the aggregate score is dominated by the other 351 events, for which changes are small.')+
        table(['Quantity','Event RMSE, line','Event RMSE, quadratic','Family RMSE gain','95% gain interval'],tailtable)+
        para('The gain intervals exhaustively resample the five existing held-out families (5⁵ paired draws); they include zero. '
        'This is uncertainty in the observed held-out losses, not an independent validation on a new large-event population. '
        'Both large Gyeongju events lie beyond the training ML range when their family is withheld. With KMA ML&gt;4, there are only three events in two families, and the quadratic has no prediction advantage.')+
        para('Large-event coverage is also limited by the existing quality rules. The KMA ML5.8 Gyeongju mainshock and ML4.6 Pohang event in the current depth-qualified cohort have spectral quality C because their plateaus are extrapolated; the ML4.1 Goesan event has no accepted independent ML. '
        'They are recorded in the coverage table and are excluded from these quality-A regressions. More well-constrained moderate and large events are needed to test the upper end.')+
        normalized)
    pages.append('<h1>Depth residuals and moment-magnitude scaling</h1>'+
        '<h2>3. A depth trend is present in the selected sample</h2>'+para(
        'The raw difference MP−ML decreases with depth. A joint family-weighted model MP=a+b(ML−3)+c(ML−3)²+d(z−15)/10 '
        'separates the depth coefficient from the magnitude distribution. With measured ML, its slopes are:')+
        table(['Magnitude and source model','d, per 10 km','95% family interval'],trendtable)+
        para(f'The MP trend remains after geographic, station-distance/sensor and sequence controls. A fixed-Vs coefficient gives d={fixed.depth_slope_per_10km:+.3f}; '
        'the trend is therefore not created solely by the layer-dependent potency conversion. Lower/upper depth-envelope endpoints retain its sign; these are sensitivities, not posterior depth samples.')+
        para(f'The trend is not uniform with depth. At depths of 5–15 km, d={shallow.depth_slope_per_10km:+.3f} '
        f'({interval(shallow.depth_lo,shallow.depth_hi)}; {int(shallow.n_events)} events). At 15–24 km, d={deep.depth_slope_per_10km:+.3f} '
        f'({interval(deep.depth_lo,deep.depth_hi)}; {int(deep.n_events)} events), which is unresolved. '
        'A continuous depth-hinge sensitivity gives a deeper upturn; the deeper behavior therefore depends on the depth parameterization. '
        'Depth remains correlated with region and recording/path conditions. These observations do not establish a source-physics cause or justify a uniform national depth correction.')+
        para(f'Adding linear depth to the measured-ML quadratic reduces MP held-out family RMSE from '
        f'{prim.CV_without_RMSE_families:.5f} to {prim.CV_with_depth_RMSE_families:.5f}; '
        f'gain interval {interval(prim.gain_lo,prim.gain_hi)}. This is an exploratory within-sample improvement.')+
        para('Liu2026 explicitly reports no clear depth trend in its quadratic residuals (main PDF p. 8; supplement Figure S7, p. 15). '
        f'For the Korean sample, the same published-curve residual definition gives a slope of {liu_depth["slope_per_10km"]:+.4f} per 10 km (95% family interval {liu_depth["slope_lo"]:+.4f} to {liu_depth["slope_hi"]:+.4f}). The contrast requires investigation under matched processing.')+
        '<h2>4. Mw gives the same broad curvature result</h2>'+para(
        f'Mw now has the same OLS, ODR, binning, family-bootstrap and held-out comparisons as MP, under both source models. '
        f'For measured ML and the Haenam Kim &amp; Kim model, Mw={ml.a_at_ML3:.4f}+{ml.slope_at_ML3:.4f}(ML−3) for the line, '
        f'and Mw={mw.a_at_ML3:.4f}+{mw.slope_at_ML3:.4f}(ML−3)+{mw.quadratic_c:.4f}(ML−3)² for the quadratic.')+
        table(['ML definition','Quadratic c','95% interval, c','CV line','CV quadratic'],mwtable)+
        para('The whole-sample predictive advantage remains unresolved, while the measured-ML&gt;4 subset benefits from curvature. '
        'Controlling for depth does not resolve the curvature uncertainty. Mw retains C=9.05; changing a constant magnitude zero point changes the intercept, not the curvature.')+
        para('Sources: supplied Trugman2026 SRL and Liu2026 GRL/supplement; Trugman and Ben-Zion (2024), doi:10.1785/0320240022, '
        '<a href="https://files.scec.org/s3fs-public/reports/2024/24169_report.pdf?h64GRlv2wy8iT.1M_nQV_Hvym9y9hCnX=">author SCEC report, pp. 4–5</a>. '
        'All follow-up numerical products use the prefix magnitude_followup in tables/.'))
    css='body{font-family:sans-serif;font-size:10pt;line-height:1.32;color:#20272c;}h1{font-size:18.5pt;margin:0 0 12pt;line-height:1.15;}h2{font-size:12.5pt;margin:12pt 0 6pt;}p{margin:0 0 8pt;}table{width:100%;border-collapse:collapse;font-size:8.7pt;margin:8pt 0 10pt;}td,th{padding:5pt 4pt;border-bottom:.5pt solid #ccd3d8;text-align:left;}th{background:#edf2f5;}'
    scales=[]
    for i,html in enumerate(pages):
        page=doc.new_page(width=595.276,height=841.89)
        spare,scale=page.insert_htmlbox(fitz.Rect(42,30,553,807),html,css=css,scale_low=.95)
        if spare<0:raise RuntimeError(f'Follow-up narrative {i+1} overflows')
        scales.append(scale)
    for item in figures:
        with fitz.open(R/'figures'/(item['stem']+'.pdf')) as part:doc.insert_pdf(part)
    metadata=dict(narrative_scales=scales,figures=figures,followup_first_page=7,followup_pages=5,
                  n_regressions=len(fits),n_depth_regressions=len(depth))
    (T/'magnitude_followup_report_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata
