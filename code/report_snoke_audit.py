"""Report the equation audit and diagnostic tests of Ahn & Kang (2026)."""
from pathlib import Path
import html
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import fitz
from snoke_audit import sha
from source_velocity import log_moment_coefficient, C

R=Path(__file__).resolve().parents[1];T=R/'tables';F=R/'figures'
S=json.loads((T/'snoke_audit_summary.json').read_text())
E=S['empirical'];A=S['analytic_controls'];Q=E['primary_quality_A']['quantiles']
data=pd.read_csv(T/'snoke_event_diagnostics.csv',dtype={'event_id':str})
q=data[data.Mw_quality.eq('A')&data.method.eq('Multitaper')&data.window_s.eq(10.1)&data.restriction.eq('full')]
examples=pd.read_csv(T/'snoke_named_event_examples.csv',dtype={'event_id':str})
fall=pd.read_csv(T/'snoke_falloff_sensitivity.csv',dtype={'event_id':str})
sens=pd.read_csv(T/'snoke_band_sensitivity.csv',dtype={'event_id':str})
sens=sens[sens.Mw_quality.eq('A')&sens.method.eq('Multitaper')&sens.window_s.eq(10.1)]
catalog=pd.read_csv(T/'Mw_ML_catalog.csv',dtype={'event_id':str})
BLUE,ORANGE,GREEN,RED='#176da0','#bd591b','#2d8062','#b32438'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':400})
figures=[]


def save(fig,stem,caption):
    fig.text(.065,.035,caption,fontsize=8,color='.25',linespacing=1.35,va='bottom')
    for extension in ['pdf','svg','png']:
        fig.savefig(F/f'{stem}.{extension}',dpi=400,facecolor='white')
    plt.close(fig);figures.append(dict(stem=stem,caption=caption))
    print(stem,flush=True)


def grid(ax):ax.grid(color='.92',lw=.5)


def reproduction():
    r=pd.read_csv(T/'snoke_coefficient_ratios.csv')
    errors=pd.read_csv(T/'snoke_true_parameter_errors.csv')
    fig,axs=plt.subplots(2,2,figsize=(11.7,9.1))
    fig.subplots_adjust(left=.09,right=.96,top=.87,bottom=.17,wspace=.30,hspace=.41)
    fig.suptitle('Independent reproduction of Ahn & Kang (2026)',fontsize=15,y=.96)
    fig.text(.09,.918,'Noise-free Brune spectrum · observed-band integrals evaluated analytically · tail equations solved jointly',fontsize=9,color='.3')
    ax=axs[0,0]
    ax.plot(r.low_over_true_fc,r.omega_printed_over_consistent,color=BLUE,lw=1.7)
    ax.axhline(1/np.sqrt(2),color='.5',ls=':',lw=.8)
    ax.set(xlabel='Lower band edge / true corner',ylabel='Printed / self-consistent spectral level',xlim=(0,1))
    ax.set_title('a  Same spectrum and band; upper edge = 10 fc',loc='left')
    ax=axs[0,1]
    for key,label,c in [('fc_jk','Corner from J / K',BLUE),('fc_jomega','Corner from J / Ω²',ORANGE)]:
        ax.plot(r.low_over_true_fc,r[key+'_printed_over_consistent'],label=label,c=c,lw=1.6)
    ax.set(xlabel='Lower band edge / true corner',ylabel='Printed / self-consistent corner',xlim=(0,1))
    ax.set_title('b  Corner-frequency pathway matters',loc='left');ax.legend(frameon=False)
    ax=axs[1,0]
    for key,label,c in [('stress_direct','Direct K, J',BLUE),('stress_jk','Ω × corner(J/K)³',GREEN),('stress_jomega','Ω × corner(J/Ω²)³',ORANGE)]:
        ax.plot(r.low_over_true_fc,r[key+'_printed_over_consistent'],label=label,c=c,lw=1.6)
    ax.set(xlabel='Lower band edge / true corner',ylabel='Printed / self-consistent stress proxy',xlim=(0,1))
    ax.set_title('c  Stress-drop pathway matters',loc='left');ax.legend(frameon=False,loc='center right')
    ax=axs[1,1]
    for high,c in [(1,RED),(2,ORANGE),(5,GREEN),(10,BLUE)]:
        z=errors[errors.high_over_true_fc.eq(high)]
        ax.semilogx(z.low_over_true_fc,z.delta_Mw_vs_true,label=f'Upper edge / fc = {high}',c=c,lw=1.4)
    ax.axhline(0,c='.5',ls=':',lw=.8)
    ax.set(xlabel='Lower band edge / true corner',ylabel='Self-consistent Mw − true Mw',xlim=(.01,1))
    ax.set_title('d  Tail error has a different reference',loc='left');ax.legend(frameon=False)
    for ax in axs.flat:grid(ax)
    save(fig,'23_Snoke_equation_controls',
         'Panels a–c compare the two implementations, each iterated to convergence; they are not errors relative to the true source.\n'
         'Panel d compares the self-consistent finite-band estimate with the true Brune input. The coefficient does not remove tail-approximation error.')


def bandwidth():
    fig,axs=plt.subplots(2,2,figsize=(11.7,9.1))
    fig.subplots_adjust(left=.09,right=.94,top=.87,bottom=.17,wspace=.33,hspace=.42)
    fig.suptitle('Bandwidth and source shape in the Korean catalog',fontsize=15,y=.96)
    fig.text(.09,.918,f'{len(q)} Mw grade-A events · 10.1 s multitaper stacks · same retained spectra and source constants',fontsize=9,color='.3')
    ax=axs[0,0]
    points=ax.scatter(q.low_over_Brune_fc,q.high_over_Brune_fc,c=q.depth_km,cmap='viridis',norm=Normalize(0,30),s=15,alpha=.6,ec='none')
    ax.axvline(.1,c='.5',ls='--',lw=.8);ax.axhline(10,c='.5',ls='--',lw=.8)
    ax.fill_between([.015,.1],10,30,color=GREEN,alpha=.1)
    ax.text(.018,18,'Paper benchmark',fontsize=8,color=GREEN)
    ax.set(xscale='log',yscale='log',xlabel='Lower edge / fitted Brune corner',ylabel='Upper edge / fitted Brune corner',xlim=(.015,.7),ylim=(.7,30))
    ax.set_title('a  Frequency coverage',loc='left')
    fig.colorbar(points,ax=ax,fraction=.035,pad=.025,label='Depth (km)')
    ax=axs[0,1]
    bins=np.arange(0,101,5)
    ax.hist(q.Brune_conditional_unobserved_J_fraction*100,bins=bins,color=BLUE,alpha=.45,label='Outside band: fitted Brune model')
    ax.hist(q.Snoke_J_high_tail_fraction*100,bins=bins,histtype='step',lw=1.5,color=ORANGE,label='Added high tail: Snoke estimate')
    ax.set(xlabel='Fraction of velocity-square integral J (%)',ylabel='Events',xlim=(0,100))
    ax.set_title('b  Dependence on unobserved frequencies',loc='left');ax.legend(frameon=False,fontsize=7.4)
    ax=axs[1,0]
    ax.scatter(q.fc_Brune_Hz,q.fc_Snoke_Hz,s=15,color=BLUE,alpha=.5,ec='none')
    ax.plot([0,22],[0,22],c='.5',lw=.8,ls=':')
    ax.set(xlabel='Direct Brune fit: corner (Hz)',ylabel='Snoke integral: corner (Hz)',xlim=(0,22),ylim=(0,22))
    ax.set_title('c  Same spectra; different corner estimators',loc='left')
    ax=axs[1,1]
    a=catalog[catalog.Mw_quality.eq('A')]
    ax.hist(a.falloff,bins=np.arange(1.45,4.16,.1),color=BLUE,ec='white',lw=.3)
    ax.set(xlabel='Generalized Boatwright falloff exponent',ylabel='Events',xlim=(1.4,4.1))
    ax.text(.97,.92,f'{int(a.falloff.eq(1.5).sum())}/{len(a)} at n = 1.5',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    ax.set_title('d  Fitted falloff reaches the lower bound',loc='left')
    for ax in axs.flat:grid(ax)
    save(fig,'24_Snoke_bandwidth',
         'Band ratios and the blue J fraction are conditional on the direct Brune fit; they are not measured missing energy.\n'
         'The stringent paper benchmark is a diagnostic reference, not a requirement for plateau-based Mw. Grade A assesses moment, not a resolved corner or stress drop.')


def comparisons():
    fig,axs=plt.subplots(3,2,figsize=(11.7,11))
    fig.subplots_adjust(left=.09,right=.96,top=.89,bottom=.14,wspace=.31,hspace=.47)
    fig.suptitle('Magnitude and sensitivity checks',fontsize=15,y=.966)
    fig.text(.09,.924,'Diagnostic alternatives to the adopted moment estimate · no fit is calibrated to ML or to another Mw catalog',fontsize=9,color='.3')
    ax=axs[0,0]
    ax.scatter(q.Mw_Brune_same_band,q.Mw_Snoke_consistent,s=13,color=BLUE,alpha=.55,ec='none')
    lo=min(q.Mw_Brune_same_band.min(),q.Mw_Snoke_consistent.min())-.1
    hi=max(q.Mw_Brune_same_band.max(),q.Mw_Snoke_consistent.max())+.1
    ax.plot([lo,hi],[lo,hi],ls=':',lw=.8,c='.5')
    ax.set(xlabel='Direct Brune fit Mw',ylabel='Self-consistent Snoke Mw',xlim=(lo,hi),ylim=(lo,hi))
    ax.set_title(f'a  Identical contiguous bands, {len(q)} events',loc='left')
    ax=axs[0,1]
    edges=np.linspace(-.2,.2,65)
    ax.hist(q.delta_Snoke_minus_Brune,bins=edges,color=BLUE,alpha=.6,label='Coefficient 2')
    ax.hist(q.Mw_Snoke_printed-q.Mw_Brune_same_band,bins=edges,histtype='step',lw=1.4,color=RED,label='Printed √2 (counterfactual)')
    ax.axvline(0,c='.45',ls=':',lw=.8)
    ax.set(xlabel='Integral Mw − direct Brune Mw',ylabel='Events');ax.legend(frameon=False,fontsize=7.7)
    ax.set_title('b  The two integral implementations',loc='left')
    ax=axs[1,0]
    ax.scatter(q.Snoke_J_high_tail_fraction*100,q.delta_Snoke_minus_Brune,s=13,c=BLUE,alpha=.5,ec='none')
    ax.axhline(0,c='.5',ls=':',lw=.8)
    ax.set(xlabel='Snoke high-tail share of J (%)',ylabel='Integral Mw − direct Brune Mw')
    ax.set_title('c  Tail dependence and moment offset',loc='left')
    ax=axs[1,1]
    ax.scatter(sens.Brune_Mw_change,sens.Snoke_Mw_change,s=14,c=BLUE,alpha=.55,ec='none')
    limit=max(abs(sens[['Brune_Mw_change','Snoke_Mw_change']].to_numpy()).max(),.06)*1.07
    ax.plot([-limit,limit],[-limit,limit],ls=':',lw=.8,c='.5');ax.axhline(0,c='.8',lw=.5);ax.axvline(0,c='.8',lw=.5)
    ax.set(xlabel='Direct Brune ΔMw',ylabel='Integral ΔMw',xlim=(-limit,limit),ylim=(-limit,limit))
    ax.set_title(f'd  Remove the upper half of the band: {len(sens)} events',loc='left')
    a=fall[fall.Mw_quality.eq('A')]
    ax=axs[2,0]
    ax.scatter(a.Mw_adopted,a.delta_Mw,c=np.where(a.falloff_adopted.eq(1.5),ORANGE,BLUE),s=14,alpha=.6,ec='none')
    review=a[a.delta_Mw.abs().gt(.1)|~a.extended_plateau_observed]
    ax.scatter(review.Mw_adopted,review.delta_Mw_with_observed_plateau,s=29,marker='D',
               fc='none',ec=GREEN,lw=.9,label='Plateau required: review cases')
    ax.axhline(0,c='.5',ls=':',lw=.8);ax.axhline(.1,c='.7',ls='--',lw=.7)
    ax.set(xlabel='Adopted Mw',ylabel='Extended-bound Mw − adopted Mw')
    ax.set_title('e  Allow finite-band falloff n ≥1 instead of ≥1.5',loc='left')
    ax.legend(frameon=False,fontsize=7.2,loc='upper right')
    ax=axs[2,1]
    ax.scatter(a.falloff_adopted,a.falloff_extended,s=15,color=BLUE,alpha=.3,ec='none')
    ax.plot([1,4],[1,4],ls=':',lw=.8,c='.5')
    ax.set(xlabel='Adopted falloff exponent',ylabel='Extended-bound exponent',xlim=(1,4.1),ylim=(.9,4.1))
    ax.set_title('f  Bounds affect source-shape interpretation',loc='left')
    for ax in axs.flat:grid(ax)
    save(fig,'25_Snoke_moment_sensitivity',
         'a–c: 443 grade-A events. d: only 210 retain one decade after trimming. e–f: 443 events, same frequency gate and source coefficient.\n'
         'Orange in e marks the original n=1.5 bound. Exponents ≤1.5 are finite-band sensitivity models; extending them to infinite frequency would make J diverge.')


def example_spectra():
    fig,axs=plt.subplots(3,1,figsize=(11.7,10.2))
    fig.subplots_adjust(left=.11,right=.95,top=.87,bottom=.20,hspace=.46)
    fig.suptitle('Three Korean earthquakes: the same spectra, three estimators',fontsize=15,y=.966)
    names={'20200503130714':'Haenam · 3 May 2020 · catalog ML 3.1',
           '20221028232749':'Goesan · 28 Oct 2022 · catalog ML 4.1',
           '20261117141017':'Sangju · 17 Nov 2026 · catalog ML 3.6'}
    for ax,row in zip(axs,examples.itertuples()):
        with np.load(R/'spectra'/f'{row.event_id}.npz') as a:
            f=a['frequency_Hz'];stem='T10.1_ALL_Multitaper_Boatwright'
            y=a[stem+'_observed'];good=a[stem+'_good']&np.isfinite(y)
            original=a[stem+'_predicted']
        low,high=row.low_Hz,row.high_Hz;selected=(f>=low)&(f<=high)
        # Magnitudes share the same moment coefficient: convert from plateau ratios.
        log_omega_brune=1.5*row.Mw_Brune_same_band+C-log_moment_coefficient(row.depth_km)
        log_omega_integral=1.5*row.Mw_Snoke_consistent+C-log_moment_coefficient(row.depth_km)
        ax.loglog(f[good],10**y[good],'.',ms=3.0,c='.6',label='Observed stack')
        ax.loglog(f,10**original,c=BLUE,lw=1.6,label='Adopted generalized fit')
        ax.loglog(f[selected],10**log_omega_brune/(1+(f[selected]/row.fc_Brune_Hz)**2),c=ORANGE,lw=1.5,label='Direct Brune fit')
        ax.loglog(f[selected],10**log_omega_integral/(1+(f[selected]/row.fc_Snoke_Hz)**2),c=GREEN,lw=1.5,ls='--',label='Consistent integral estimate')
        ax.set(xlabel='Frequency (Hz)',ylabel='Corrected FAS (m² s)',xlim=(.45,22))
        ax.set_title(names[row.event_id]+f'  |  Mw: {row.Mw_adopted:.3f} / {row.Mw_Brune_same_band:.3f} / {row.Mw_Snoke_consistent:.3f}',loc='left',pad=9)
        ax.text(.98,.94,f'Added J high tail: {row.Snoke_J_high_tail_fraction:.0%}',transform=ax.transAxes,ha='right',va='top',fontsize=8)
        grid(ax)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.092),ncol=4,frameon=False,fontsize=8.5)
    save(fig,'26_Snoke_event_examples',
         '10.1 s multitaper, original site selections and path corrections; Kim & Kim (1983) source-depth velocity, density 2800 kg/m³, C=9.05.\n'
         'Integral and direct Brune curves use their own estimated corners. They are diagnostic alternatives; the adopted magnitudes are unchanged.')


def paragraph(text):return '<p>'+text+'</p>'
def table(head,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in head)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</table>'


def narratives():
    fs=E['falloff_bound_sensitivity'];trim=E['upper_half_removal']
    pages=[]
    pages.append('<h1>Ahn & Kang (2026): what applies here</h1>'+
      paragraph('<b>The paper’s algebra is correct. Its coefficient correction does not apply to our adopted Korean Mw.</b> Our final spectral levels are obtained by direct fitting, without Snoke K–J initialization. This audit independently reproduces the paper’s equations and finite-band benchmark, then applies the two integral implementations to the existing Korean spectra as diagnostics.')+
      '<h2>The finding and its scope</h2>'+paragraph('For a Brune displacement spectrum U(f)=Ω₀/[1+(f/fc)²], and K=2∫₀∞|U|²df, J=2∫₀∞|2πfU|²df, direct integration gives K=(π/2)Ω₀²fc and J=2π³Ω₀²fc³. Therefore Ω₀=2(K³/J)<sup>1/4</sup>. The alternative √2 expression audited by Ahn and Kang is inconsistent with those specific definitions. It underestimates moment by 1/√2, or Mw by 0.100343, for a common K,J pair.')+
      paragraph('The primary Ω₀,J formulation, the J/K corner equation and the direct K,J stress-drop coefficient remain consistent. An iterative low-frequency J tail couples J to Ω₀, so the two implementations need not share J. Finite-band corrections are therefore pathway dependent. Ahn and Kang identify an equation/implementation audit problem; they do not demonstrate a coefficient bias in a final published catalog.')+
      table(['Workflow','Audit result'],[
        ['Our current Mw, including Sangju','Code-level: profiled direct generalized Boatwright fit; Brune sensitivity. No K,J moment formula or Snoke-dependent starting range. No coefficient rescaling.'],
        ['Hong & Sheen (2026), KR_Mw','Paper-level: direct Brune/MCMC spectral fit and a peak-amplitude-derived fit. No applicable K,J coefficient is shown. Unpublished implementation details are not certified.'],
        ['Lim et al. (2024), Goesan','Section 2.2 uses Snoke to choose the initial grid domain, then direct Brune fitting. Bounds/convergence would need a separate code audit; no automatic +0.10 correction.']])+
      '<h2>Strength and limits of the paper</h2>'+paragraph('The exact derivation, pathway distinctions and numerical benchmark are reproducible. The coefficient 2 was already known and used; the contribution is the careful localization of the inconsistency. The quoted 2.0% moment, 0.006 Mw, 3.85% corner and 9.35% stress departures compare the two finite-band procedures for noise-free Brune spectra with fh=10fc and 0≤fl/fc≤1. They are neither absolute error bars nor bounds for arbitrary Korean passbands, noise or spectral shapes.')+
      paragraph('<span class="small">Primary source: Ahn, B. S., and T.-S. Kang (2026), <i>The Seismic Record</i> 6(4), 493–501, doi:10.1785/0320260027; supplied nine-page PDF, published 18 Nov 2026. Equation/scope: pp.493–495; pathways/Tables 2–3: pp.496–498; exact residuals: p.501. Lim author manuscript, §2.2, PDF p.5: https://lowbontimp.github.io/papers/Lim_etal_2024_JS.pdf.</span>'))
    exrows=[]
    names={'20200503130714':'Haenam 2020','20221028232749':'Goesan 2022','20261117141017':'Sangju 2026'}
    for row in examples.itertuples():exrows.append([names[row.event_id],f'{row.Mw_adopted:.3f}',f'{row.Mw_Brune_same_band:.3f}',f'{row.Mw_Snoke_consistent:.3f}',f'{row.Snoke_J_high_tail_fraction:.0%}'])
    pages.append('<h1>Results for the Korean spectra</h1>'+
      paragraph(f'<b>All {E["n_adopted_replays"]} finite adopted Mw estimates replay to within {E["max_adopted_Mw_replay_error"]:.1e}.</b> The original spectra, Mw/ML catalogs, regressions and existing reports are preserved. The coefficient audit is independent of the source-velocity choice because the fixed-prefactor moment is linear in Ω₀.')+
      '<h2>Bandwidth matters for the integral estimate</h2>'+paragraph(f'All 443 Mw grade-A events have a usable contiguous band for this test. The median lower/Brune-corner ratio is {Q["low_over_Brune_fc"]["median"]:.3f}, and the median upper/corner ratio is {Q["high_over_Brune_fc"]["median"]:.2f}. None meets both fl≤0.1fc and fh≥10fc, the stringent benchmark. This does not invalidate a moment measured from a low-frequency plateau. It does limit conclusions that depend on the complete velocity-square integral or the high-frequency asymptote.')+
      paragraph(f'The fitted Brune model places a median {Q["Brune_conditional_unobserved_J_fraction"]["median"]:.1%} of J outside the observed band; this is a model-conditional quantity. In the actual Snoke calculation, the added high tail supplies a median {Q["Snoke_J_high_tail_fraction"]["median"]:.1%} of estimated J. The self-consistent integral Mw differs from a direct Brune fit on the same band by median {Q["delta_Snoke_minus_Brune"]["median"]:+.3f} (5th–95th percentiles {Q["delta_Snoke_minus_Brune"]["p05"]:+.3f} to {Q["delta_Snoke_minus_Brune"]["p95"]:+.3f}). No independent truth is available to interpret this as a bias correction.')+
      table(['Event','Adopted Mw','Direct Brune','Consistent integral','Added J high tail'],exrows)+
      paragraph('For Sangju, 3.491 is an alternative integral estimate, not 3.417 plus a coefficient correction. The direct Brune fit already gives 3.464, so much of the difference comes from the adopted generalized crossover/shape. The remaining difference is the estimator and its tail treatment.')+
      '<h2>Shape and band sensitivity</h2>'+paragraph(f'{fs["n_original_falloff_lower_bound"]}/443 generalized fits reach n=1.5. Extending the finite-band exponent to n≥1 changes Mw by median {fs["delta_Mw_quantiles"]["0.5"]:+.3f}. All {fs["n_absolute_Mw_change_gt_point1"]} shifts exceeding 0.10 lose the observed-plateau criterion. Enforcing that same criterion gives median {fs["constrained_plateau_delta_Mw_median"]:+.3f}, 95th percentile {fs["constrained_plateau_delta_Mw_p95"]:+.3f}; one case still exceeds 0.10. The {fs["n_review_cases"]} cases with loss of plateau or a large shift are listed in snoke_model_review_events.csv.')+
      paragraph('This is a sensitivity test, not a preferred infinite-band source model: J diverges for an unbroken generalized Boatwright tail with n≤1.5. Even n=2 differs from Brune; its exact integral coefficient is 8<sup>1/4</sup>, and using 2 creates a spurious +0.05017 Mw shift. Neither fit quality nor a coefficient change establishes a true moment correction.')+
      paragraph(f'After removing the upper half of the usable band, {trim["n"]} grade-A events retain a decade. Median absolute Mw changes are {trim["median_abs_Mw_change_Brune"]:.3f} for direct Brune and {trim["median_abs_Mw_change_Snoke"]:.3f} for the integral method. These descriptive checks do not establish corner-frequency or stress-drop accuracy.')+
      paragraph('Excluding the nine paired model-review cases changes the family-balanced Mw–measured-ML slope from 0.7271 to 0.7280 (357 to 348 events); quadratic curvature changes from 0.0301 to 0.0321. These are point-estimate sensitivities, not new significance tests or updated adopted relations.')+
      '<h2>The earlier √2 discussion</h2>'+paragraph('The PSD conversion √(TP₁/2) converts one-sided density P₁ into undoubled Fourier-amplitude units. The present paper concerns the coefficient of a different, K–J parameter formula at fixed integral definitions. The identical 0.100343 magnitude scale is not evidence of an additional correction, cancellation, or proof that a transient multitaper plateau is unbiased.')+
      paragraph('<span class="small">Protocol: retained 10.1/20.1 s stacks, both spectrum estimators, original source constants and SNR gate. Integrals use the longest contiguous usable band (≥one decade, ≥20 samples); no holes are bridged. The same-band direct Brune fit is the control. No band or estimator is selected by agreement with ML. Detailed tables: snoke_event_diagnostics.csv, snoke_band_sensitivity.csv and snoke_falloff_sensitivity.csv.</span>'))
    return pages


def independent_checks():
    """Independent numerical quadrature and bisection on the Sangju stack."""
    from scipy.integrate import quad
    from scipy.optimize import bisect
    from snoke_audit import measured_snoke
    eid='20261117141017'
    row=examples[examples.event_id.eq(eid)].iloc[0]
    with np.load(R/'spectra'/f'{eid}.npz') as a:
        f=a['frequency_Hz'];y=a['T10.1_ALL_Multitaper_Brune_observed']
    take=(f>=row.low_Hz)&(f<=row.high_Hz)
    ff,yy=f[take],y[take]
    def amp(hz):return 10**np.interp(np.log(hz),np.log(ff),yy)
    kb=2*quad(lambda hz:amp(hz)**2,ff[0],ff[-1],points=ff[1:-1],limit=600,epsabs=1e-12,epsrel=1e-11)[0]
    jb=8*np.pi**2*quad(lambda hz:hz*hz*amp(hz)**2,ff[0],ff[-1],points=ff[1:-1],limit=600,epsabs=1e-10,epsrel=1e-11)[0]
    k=kb+2*amp(ff[0])**2*ff[0]+2*amp(ff[-1])**2*ff[-1]/3
    j_without_low=jb+8*np.pi**2*ff[-1]**3*amp(ff[-1])**2
    def equation(omega):
        j=j_without_low+8*np.pi**2*omega**2*ff[0]**3/3
        return omega-2*(k**3/j)**.25
    direct=measured_snoke(ff,yy)
    omega=bisect(equation,direct['omega']/10,direct['omega']*10,xtol=1e-13,rtol=1e-13)
    error=abs(omega/direct['omega']-1)
    assert error<1e-10
    # Full-band Boatwright n=2, independently integrated rather than using CK/CJ.
    ck=2*quad(lambda x:1/(1+x**4),0,np.inf,epsabs=1e-11)[0]
    cj=8*np.pi**2*quad(lambda x:x*x/(1+x**4),0,np.inf,epsabs=1e-10)[0]
    coefficient=(cj/ck**3)**.25
    assert abs(coefficient-8**.25)<1e-11
    original=pd.read_csv(T/'independent_ML_regressions.csv')
    baseline=original[original.selection.eq('Independent ML')].iloc[0]
    relations=pd.read_csv(T/'snoke_Mw_ML_relation_sensitivity.csv')
    recomputed=relations[relations.selection.eq('All paired A')&relations.degree.eq(1)].iloc[0]
    assert abs(recomputed.slope_at_ML3-baseline.slope)<1e-11
    assert abs(recomputed.a_at_ML3-baseline.intercept_at_x3)<1e-11
    # The PSD identity is separate from the Snoke coefficient check.
    n=2048;dt=.01;x=np.sin(2*np.pi*57*np.arange(n)/n)
    fft=np.fft.rfft(x);p1=dt/n*abs(fft)**2;p1[1:-1]*=2
    fas=np.sqrt(n*dt*p1[1:-1]/2)
    reference=dt*abs(fft[1:-1]);psd_error=float(np.max(abs(fas-reference)))
    assert psd_error<1e-12
    return dict(Sangju_quadrature_bisection_relative_error=error,
        independently_integrated_Boatwright_n2_coefficient=coefficient,
        original_Mw_ML_point_estimate_reproduced=True,
        one_sided_PSD_to_undoubled_FAS_max_error=psd_error)


def main():
    reproduction();bandwidth();comparisons();example_spectra()
    css='body{font-family:sans-serif;font-size:10pt;line-height:1.29;color:#222;}h1{font-size:18pt;margin:0 0 11pt;}h2{font-size:11.4pt;margin:10pt 0 5pt;}p{margin:0 0 7pt;}table{width:100%;border-collapse:collapse;font-size:8.8pt;margin:7pt 0 9pt;}td,th{padding:4pt 4pt;border-bottom:.5pt solid #ccd3d8;text-align:left;}th{background:#edf2f5;}.small{font-size:8pt;color:#444;}'
    doc=fitz.open();scales=[]
    for text in narratives():
        page=doc.new_page(width=595.276,height=841.89)
        spare,scale=page.insert_htmlbox(fitz.Rect(40,28,555,812),text,css=css,scale_low=.94)
        if spare<0:raise RuntimeError('Snoke audit narrative exceeds page')
        scales.append(scale)
    for figure in figures:
        with fitz.open(F/(figure['stem']+'.pdf')) as part:doc.insert_pdf(part)
    doc.set_toc([[1,title,i+1] for i,title in enumerate([
        'Review and scope','Korean catalog: results','Independent equation checks',
        'Bandwidth and source shape','Magnitude sensitivity','Three Korean earthquakes'])])
    report=R/'reports/Snoke_coefficient_and_bandwidth_audit.pdf'
    doc.set_metadata({'title':'Ahn & Kang (2026): coefficient and bandwidth audit for Korean Mw','author':'Korean Mw/ML project'})
    doc.save(report,garbage=4,deflate=True)
    with fitz.open(report) as output:
        report.with_suffix('.txt').write_text('\n\n'.join(page.get_text() for page in output))
        assert len(output)==6
        for i,page in enumerate(output):
            assert '\x00' not in page.get_text()
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line.get('spans',[]):
                        x0,y0,x1,y1=span['bbox']
                        assert x0>=0 and y0>=0 and x1<=page.rect.width and y1<=page.rect.height,(i,span)
            page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(f'/tmp/korean_snoke_report_{i+1}.png')
    assert all(sha(R/name)==value for name,value in S['protected_sha256'].items())
    validation=dict(passed=True,report_pages=6,narrative_scales=scales,
        report_sha256=sha(report),paper_sha256=S['paper']['sha256'],
        adopted_Mw_coefficient_correction_applicable=False,
        independent_algebra_and_paper_benchmarks_passed=True,
        adopted_Mw_replay_count=E['n_adopted_replays'],adopted_Mw_replay_max_error=E['max_adopted_Mw_replay_error'],
        n_grade_A_diagnostic_events=len(q),original_results_and_reports_unchanged=True,
        protected_sha256=S['protected_sha256'],source_sha256={str(path.relative_to(R)):sha(path) for path in [
            R/'code/snoke_audit.py',R/'code/report_snoke_audit.py',T/'snoke_audit_summary.json']},
        new_analysis_csv_sha256={str(path.relative_to(R)):sha(path) for path in sorted(T.glob('snoke_*.csv'))},
        independent_numerical_checks=independent_checks(),
        figure_and_report_page_bounds_passed=True)
    (T/'snoke_audit_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (T/'snoke_figure_manifest.json').write_text(json.dumps(figures,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k not in ['protected_sha256','source_sha256','new_analysis_csv_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
