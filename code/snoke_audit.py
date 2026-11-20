"""Ahn & Kang (2026): equation audit and diagnostic spectral-integral replay.

This analysis never writes adopted magnitudes, spectra, regressions or ML.
The integral estimates use Snoke's Brune tails and are diagnostic, not a
replacement catalog. Frequencies are ordinary Hz and amplitudes are the
undoubled Fourier amplitudes retained by the Korean Mw project.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys
import warnings

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from spectra import fit
from source_velocity import log_moment_coefficient, C

R = Path(__file__).resolve().parents[1]
T = R / 'tables'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fk(x):
    x = np.asarray(x)
    return .5*(np.arctan(x)+x/(1+x*x))


def fj(x):
    x = np.asarray(x)
    return .5*(np.arctan(x)-x/(1+x*x))


def solve_integrals(k, j_without_low_tail, low, coefficient=2.):
    """Joint positive solution for Omega and J; input amplitudes may be scaled."""
    a = float(j_without_low_tail)
    b = 8*np.pi**2*low**3/3
    x0 = coefficient**2*np.sqrt(k**3/a)
    coupling = b*x0/a
    y = brentq(lambda q: q*q+coupling*q**3-1, 0, 1, xtol=5e-15, rtol=1e-14)
    omega = np.sqrt(x0*y)
    j = a+b*omega**2
    residual = abs(coefficient*(k**3/j)**.25/omega-1)
    fc_jk = np.sqrt(j/k)/(2*np.pi)
    fc_jo = (j/(2*np.pi**3*omega**2))**(1/3)
    return dict(omega=omega, fc_jk=fc_jk, fc_jomega=fc_jo,
                stress_direct=(j**5/k**3)**.25/(4*np.pi**3),
                stress_jk=omega*fc_jk**3, stress_jomega=omega*fc_jo**3,
                K=k, J=j, J_low=b*omega**2, J_without_low=a,
                fixed_point_relative_error=residual)


def analytic_brune(low, high, omega=1., corner=1., coefficient=2.):
    """Exact observed-band integrals followed by Snoke's actual tails."""
    xl, xh = low/corner, high/corner
    ul, uh = omega/(1+xl*xl), omega/(1+xh*xh)
    kb = 2*omega**2*corner*(fk(xh)-fk(xl))
    jb = 8*np.pi**2*omega**2*corner**3*(fj(xh)-fj(xl))
    kl, kh = 2*ul**2*low, 2*uh**2*high/3
    jh = 8*np.pi**2*high**3*uh**2
    row = solve_integrals(kb+kl+kh, jb+jh, low, coefficient)
    row.update(K_band=kb, K_low=kl, K_high=kh, J_band=jb, J_high=jh)
    return row


def loglog_integral(f, u, power):
    """Integrate f**power U**2, exactly for piecewise log-linear amplitude."""
    width = np.log(f[1:]/f[:-1])
    slope = 2*np.diff(np.log(u))/width
    exponent = slope+power+1
    z = exponent*width
    factor = np.empty_like(z)
    small = abs(z)<1e-8
    factor[small] = width[small]*(1+z[small]/2+z[small]**2/6)
    factor[~small] = np.expm1(z[~small])/exponent[~small]
    return float(np.sum(u[:-1]**2*f[:-1]**(power+1)*factor))


def measured_snoke(f, log_amplitude, coefficient=2.):
    # Scaling keeps the cubic well conditioned without changing any ratio.
    scale = 10**float(np.median(log_amplitude))
    u = 10**log_amplitude/scale
    low, high = f[0], f[-1]
    kb = 2*loglog_integral(f, u, 0)
    jb = 8*np.pi**2*loglog_integral(f, u, 2)
    kl, kh = 2*u[0]**2*low, 2*u[-1]**2*high/3
    jh = 8*np.pi**2*high**3*u[-1]**2
    row = solve_integrals(kb+kl+kh, jb+jh, low, coefficient)
    row.update(K_band=kb, K_low=kl, K_high=kh, J_band=jb, J_high=jh,
        K_tail_fraction=(kl+kh)/(kb+kl+kh), J_high_tail_fraction=jh/row['J'],
        J_all_tail_fraction=(jh+row['J_low'])/row['J'],
        K_quadrature_vs_trapezoid=kb/(2*np.trapezoid(u*u, f))-1,
        J_quadrature_vs_trapezoid=jb/(8*np.pi**2*np.trapezoid(f*f*u*u, f))-1)
    for key in ['omega', 'stress_direct', 'stress_jk', 'stress_jomega']:
        row[key] *= scale
    for key in ['K','J','J_low','J_without_low','K_band','K_low','K_high','J_band','J_high']:
        row[key] *= scale**2
    return row


def contiguous(mask):
    ids = np.flatnonzero(mask)
    if not len(ids): return ids
    return max(np.split(ids, np.flatnonzero(np.diff(ids)>1)+1), key=len)


def analytic_controls():
    # Independent quadrature of the full integrals, not the closed expressions.
    ik, ek = quad(lambda x: 1/(1+x*x)**2, 0, np.inf, epsabs=1e-12, epsrel=1e-12)
    ij, ej = quad(lambda x: x*x/(1+x*x)**2, 0, np.inf, epsabs=1e-12, epsrel=1e-12)
    assert abs(ik-np.pi/4)<1e-12 and abs(ij-np.pi/4)<1e-12
    k, j = 2*ik, 8*np.pi**2*ij
    assert abs(2*(k**3/j)**.25-1)<1e-12
    assert abs(np.sqrt(j/k)/(2*np.pi)-1)<1e-12
    assert abs((j**5/k**3)**.25/(4*np.pi**3)-1)<1e-12
    grid = np.linspace(0, 1, 1001)
    ratios = []
    for low in grid:
        s = analytic_brune(low, 10, coefficient=2)
        p = analytic_brune(low, 10, coefficient=np.sqrt(2))
        row = dict(low_over_true_fc=low, high_over_true_fc=10.,
                   omega_printed_over_consistent=p['omega']/s['omega'],
                   delta_Mw_printed_minus_consistent=np.log10(p['omega']/s['omega'])/1.5)
        for key in ['fc_jk','fc_jomega','stress_direct','stress_jk','stress_jomega']:
            row[key+'_printed_over_consistent'] = p[key]/s[key]
        ratios.append(row)
        assert s['fixed_point_relative_error'] < 1e-12
        assert p['fixed_point_relative_error'] < 1e-12
    ratios = pd.DataFrame(ratios)
    ratios.to_csv(T/'snoke_coefficient_ratios.csv', index=False)
    limit = ratios.iloc[0]
    assert abs(limit.omega_printed_over_consistent-2**(-.5))<1e-12
    assert abs(limit.fc_jomega_printed_over_consistent-2**(1/3))<1e-12
    assert abs(limit.stress_direct_printed_over_consistent-1)<1e-12
    departures = dict(moment=float(abs(np.sqrt(2)*ratios.omega_printed_over_consistent-1).max()),
        magnitude=float(abs(ratios.delta_Mw_printed_minus_consistent+np.log10(2)/3).max()),
        fc_jk=float(abs(ratios.fc_jk_printed_over_consistent-1).max()),
        fc_jomega=float(abs(2**(-1/3)*ratios.fc_jomega_printed_over_consistent-1).max()),
        stress_direct=float(abs(ratios.stress_direct_printed_over_consistent-1).max()),
        stress_jk=float(abs(np.sqrt(2)*ratios.stress_jk_printed_over_consistent-1).max()),
        stress_jomega=float(abs(ratios.stress_jomega_printed_over_consistent/np.sqrt(2)-1).max()))
    assert departures['moment']<.020 and departures['magnitude']<.006
    assert departures['fc_jk']<.0386 and departures['stress_direct']<.0936
    low, high = .1, 10.
    eps_k = (2*(fk(high)-fk(low))+2*low/(1+low*low)**2+2*high/(3*(1+high*high)**2))/(np.pi/2)-1
    eps_j = (8*np.pi**2*(fj(high)-fj(low))+8*np.pi**2*low**3/3+8*np.pi**2*high**3/(1+high*high)**2)/(2*np.pi**3)-1
    assert abs(eps_k-(-.00167085))<5e-9 and abs(eps_j-(-.00166249))<5e-9
    tail_grid = []
    for high in [1., 2., 3., 5., 10., 20.]:
        for low in np.geomspace(.01, min(.99, high*.8), 90):
            row = analytic_brune(low, high)
            tail_grid.append(dict(low_over_true_fc=low, high_over_true_fc=high,
                omega_over_true=row['omega'], delta_Mw_vs_true=np.log10(row['omega'])/1.5,
                fc_over_true=row['fc_jk'], stress_over_true=row['stress_direct']))
    pd.DataFrame(tail_grid).to_csv(T/'snoke_true_parameter_errors.csv', index=False)
    shapes = []
    for n in [1.5, 1.6, 1.8, 2., 2.4, 3., 4.]:
        if n<=1.5:
            shapes.append(dict(model='Generalized Boatwright', falloff=n, integrable_J=False))
            continue
        ck = np.pi/n/np.sin(np.pi/(2*n))
        cj = 4*np.pi**3/n/np.sin(3*np.pi/(2*n))
        coefficient = (cj/ck**3)**.25
        shapes.append(dict(model='Generalized Boatwright', falloff=n, integrable_J=True,
            correct_integral_coefficient=coefficient,
            Brune_coefficient_inferred_omega_over_true=2/coefficient,
            Brune_coefficient_delta_Mw=np.log10(2/coefficient)/1.5))
    shapes.append(dict(model='Brune', falloff=2., integrable_J=True, correct_integral_coefficient=2.,
        Brune_coefficient_inferred_omega_over_true=1., Brune_coefficient_delta_Mw=0.))
    pd.DataFrame(shapes).to_csv(T/'snoke_source_shape_controls.csv', index=False)
    model2 = next(row for row in shapes if row['model']=='Generalized Boatwright' and row['falloff']==2)
    assert abs(model2['Brune_coefficient_delta_Mw']-np.log10(2)/6)<1e-12
    f = np.geomspace(.1, 10, 200); u = -np.log10(1+f*f)
    numerical = measured_snoke(f, u)
    exact = analytic_brune(.1, 10)
    grid_error = abs(numerical['omega']/exact['omega']-1)
    assert grid_error<.0002
    return dict(full_integral_quadrature_error=max(abs(ik-np.pi/4),abs(ij-np.pi/4)),
        fixed_pair_Mw_bias=-np.log10(2)/3,
        paper_benchmark_eps_K=eps_k, paper_benchmark_eps_J=eps_j,
        paper_benchmark_omega_ratio=((1+eps_k)**3/(1+eps_j))**.25,
        paper_benchmark_fc_ratio=((1+eps_j)/(1+eps_k))**.5,
        paper_benchmark_stress_ratio=(1+eps_j)**1.25/(1+eps_k)**.75,
        broadband_backcorrection_max_departure=departures,
        interpolation_200_point_omega_relative_error=grid_error,
        source_shape_Boatwright_n2_spurious_Mw_shift=model2['Brune_coefficient_delta_Mw'])


def catalog_diagnostics():
    catalog = pd.read_csv(T/'Mw_ML_catalog.csv', dtype={'event_id':str})
    target = json.loads((T/'Sangju2026_summary.json').read_text())
    cases = catalog.to_dict('records') + [dict(event_id=target['event_id'],z_central=target['depth_km'],
        Mw_quality='held out', ml=target['reported_ML'], Mw=target['preferred']['Mw'],
        family_block='Sangju_holdout')]
    rows, replay, excluded = [], [], []
    for i, case in enumerate(cases):
        eid = case['event_id']; path = R/'spectra'/f'{eid}.npz'
        if not path.exists():
            excluded.append(dict(event_id=eid,reason='No retained event spectra')); continue
        with np.load(path) as archive:
            f = archive['frequency_Hz']
            for window in [10.1, 20.1]:
                for method in ['Multitaper', 'Direct Fourier']:
                    stem = f'T{window:g}_ALL_{method.replace(" ","_")}_Brune'
                    if stem+'_observed' not in archive: continue
                    y = archive[stem+'_observed']; mask = archive[stem+'_good']&np.isfinite(y)
                    if window==10.1 and method=='Multitaper' and np.isfinite(case.get('Mw', np.nan)):
                        fitted, _ = fit(f,y,mask,'Boatwright',depth_km=case['z_central'])
                        if np.isfinite(fitted.get('Mw', np.nan)):
                            replay.append(dict(event_id=eid, Mw_existing=case['Mw'], Mw_replayed=fitted['Mw'],
                                absolute_error=abs(fitted['Mw']-case['Mw'])))
                    ids = contiguous(mask)
                    if len(ids)<20 or np.ptp(np.log10(f[ids]))<1:
                        excluded.append(dict(event_id=eid,window_s=window,method=method,
                            reason='No contiguous usable band of one decade and 20 samples')); continue
                    for restriction in ['full', 'remove_high_half']:
                        use = ids if restriction=='full' else ids[f[ids]<=f[ids[-1]]/2]
                        if len(use)<20 or np.ptp(np.log10(f[use]))<1: continue
                        good = np.zeros(len(f), bool); good[use]=True
                        brune, _ = fit(f,y,good,'Brune',depth_km=case['z_central'])
                        if not np.isfinite(brune.get('Mw', np.nan)):continue
                        s = measured_snoke(f[use],y[use],2.)
                        wrong = measured_snoke(f[use],y[use],np.sqrt(2))
                        coeff = log_moment_coefficient(case['z_central'])
                        mw = (np.log10(s['omega'])+coeff-C)/1.5
                        mw_wrong = (np.log10(wrong['omega'])+coeff-C)/1.5
                        xl, xh = f[use[0]]/brune['fc_Hz'], f[use[-1]]/brune['fc_Hz']
                        jfrac = (fj(xh)-fj(xl))/(np.pi/4)
                        kfrac = (fk(xh)-fk(xl))/(np.pi/4)
                        rows.append(dict(event_id=eid,Mw_quality=case['Mw_quality'],ML_catalog=case['ml'],
                            family_block=case.get('family_block',''),depth_km=case['z_central'],
                            method=method,window_s=window,restriction=restriction,
                            n_frequency=len(use),n_usable_runs=len(np.split(np.flatnonzero(mask),np.flatnonzero(np.diff(np.flatnonzero(mask))>1)+1)),
                            low_Hz=f[use[0]],high_Hz=f[use[-1]],
                            Mw_adopted=case['Mw'],Mw_Brune_same_band=brune['Mw'],
                            fc_Brune_Hz=brune['fc_Hz'],Brune_plateau_observed=brune['plateau_observed'],
                            Brune_corner_upper_rail=brune['corner_upper_rail'],
                            low_over_Brune_fc=xl,high_over_Brune_fc=xh,
                            Brune_conditional_unobserved_K_fraction=1-kfrac,
                            Brune_conditional_unobserved_J_fraction=1-jfrac,
                            Mw_Snoke_consistent=mw,Mw_Snoke_printed=mw_wrong,
                            delta_Snoke_minus_Brune=mw-brune['Mw'],
                            delta_Snoke_minus_adopted=mw-case['Mw'],
                            delta_printed_minus_consistent=mw_wrong-mw,
                            residual_after_broadband_Mw_backcorrection=mw_wrong+np.log10(2)/3-mw,
                            fc_Snoke_Hz=s['fc_jk'],Snoke_K_tail_fraction=s['K_tail_fraction'],
                            Snoke_J_high_tail_fraction=s['J_high_tail_fraction'],
                            Snoke_J_all_tail_fraction=s['J_all_tail_fraction'],
                            K_quadrature_vs_trapezoid=s['K_quadrature_vs_trapezoid'],
                            J_quadrature_vs_trapezoid=s['J_quadrature_vs_trapezoid'],
                            fixed_point_error=max(s['fixed_point_relative_error'],wrong['fixed_point_relative_error'])))
        if i%100==0: print('Spectral-integral audit',i+1,'/',len(cases),flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(T/'snoke_event_diagnostics.csv',index=False)
    pd.DataFrame(excluded).to_csv(T/'snoke_excluded_bands.csv',index=False)
    replay = pd.DataFrame(replay); replay.to_csv(T/'snoke_adopted_Mw_replay.csv',index=False)
    assert replay.absolute_error.max()<1e-10
    assert frame.fixed_point_error.max()<1e-11
    def statistics(q):
        columns=['low_over_Brune_fc','high_over_Brune_fc','Brune_conditional_unobserved_J_fraction',
                 'Snoke_J_high_tail_fraction','delta_Snoke_minus_Brune',
                 'delta_printed_minus_consistent','residual_after_broadband_Mw_backcorrection']
        result=dict(n=len(q),n_paper_broadband_benchmark=int(((q.low_over_Brune_fc<=.1)&(q.high_over_Brune_fc>=10)).sum()),
                    n_Brune_upper_rail=int(q.Brune_corner_upper_rail.sum()))
        result['quantiles']={col:{'p05':float(q[col].quantile(.05)),'median':float(q[col].median()),'p95':float(q[col].quantile(.95))} for col in columns}
        return result
    primary=frame[(frame.Mw_quality=='A')&(frame.method=='Multitaper')&frame.window_s.eq(10.1)&frame.restriction.eq('full')]
    sensitivity=[]
    for (window,method),q in frame.groupby(['window_s','method']):
        full=q[q.restriction.eq('full')].set_index('event_id')
        trim=q[q.restriction.eq('remove_high_half')].set_index('event_id')
        for eid in full.index.intersection(trim.index):
            a,b=full.loc[eid],trim.loc[eid]
            sensitivity.append(dict(event_id=eid,Mw_quality=a.Mw_quality,window_s=window,method=method,
                Snoke_Mw_change=float(b.Mw_Snoke_consistent-a.Mw_Snoke_consistent),
                Brune_Mw_change=float(b.Mw_Brune_same_band-a.Mw_Brune_same_band),
                Snoke_fc_ratio=float(b.fc_Snoke_Hz/a.fc_Snoke_Hz),
                Brune_fc_ratio=float(b.fc_Brune_Hz/a.fc_Brune_Hz)))
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(T/'snoke_band_sensitivity.csv',index=False)
    sq=sensitivity[(sensitivity.Mw_quality=='A')&sensitivity.method.eq('Multitaper')&sensitivity.window_s.eq(10.1)]
    examples=frame[frame.event_id.isin(['20200503130714','20221028232749','20261117141017'])&frame.method.eq('Multitaper')&frame.window_s.eq(10.1)&frame.restriction.eq('full')]
    examples.to_csv(T/'snoke_named_event_examples.csv',index=False)
    return dict(total_diagnostic_rows=len(frame),n_adopted_replays=len(replay),
        max_adopted_Mw_replay_error=float(replay.absolute_error.max()),
        n_quality_A_catalog=int((catalog.Mw_quality=='A').sum()),primary_quality_A=statistics(primary),
        upper_half_removal=dict(n=len(sq),
          median_abs_Mw_change_Snoke=float(sq.Snoke_Mw_change.abs().median()),
          median_abs_Mw_change_Brune=float(sq.Brune_Mw_change.abs().median()),
          p95_abs_Mw_change_Snoke=float(sq.Snoke_Mw_change.abs().quantile(.95)),
          p95_abs_Mw_change_Brune=float(sq.Brune_Mw_change.abs().quantile(.95))),
        examples=examples.to_dict('records'))


def falloff_controls():
    """Finite-band model sensitivity; n<1.5 is not an infinite-band source model."""
    catalog = pd.read_csv(T/'Mw_ML_catalog.csv', dtype={'event_id':str})
    target = json.loads((T/'Sangju2026_summary.json').read_text())
    cases = catalog[catalog.Mw_quality.eq('A')].to_dict('records') + [dict(
        event_id=target['event_id'],z_central=target['depth_km'],Mw_quality='held out',
        Mw=target['preferred']['Mw'],fc_Hz=target['preferred']['fc_Hz'],falloff=target['preferred']['falloff'])]
    rows=[];model=None
    for case in cases:
        with np.load(R/'spectra'/f'{case["event_id"]}.npz') as a:
            f=a['frequency_Hz'];stem='T10.1_ALL_Multitaper_Boatwright'
            y=a[stem+'_observed'];mask=a[stem+'_good']&np.isfinite(y)
        if model is None:
            corners=np.geomspace(.2,f[-1],400);exponents=np.linspace(1.,4.,31)
            fc=np.repeat(corners,len(exponents));n=np.tile(exponents,len(corners))
            phi=-.5*np.log10(1+(f[None,:]/fc[:,None])**(2*n[:,None]));phi2=phi*phi
            model=True
        g=mask.astype(float);ng=g.sum();yy=np.where(mask,y,0.)
        om=(yy.sum()-phi@g)/ng
        variance=(np.sum(yy*yy)+phi2@g-2*phi@yy)/ng-om*om
        j=int(np.argmin(variance))
        mw=(float(om[j])+log_moment_coefficient(case['z_central'])-C)/1.5
        observed=fc>=2*f[mask].min()
        jp=int(np.argmin(np.where(observed,variance,np.inf)))
        mw_plateau=(float(om[jp])+log_moment_coefficient(case['z_central'])-C)/1.5
        rows.append(dict(event_id=case['event_id'],Mw_quality=case['Mw_quality'],
            Mw_adopted=case['Mw'],falloff_adopted=case['falloff'],fc_adopted_Hz=case['fc_Hz'],
            Mw_extended=mw,falloff_extended=float(n[j]),fc_extended_Hz=float(fc[j]),
            delta_Mw=mw-case['Mw'],fc_ratio=float(fc[j]/case['fc_Hz']),
            extended_lower_falloff_bound=bool(j%len(exponents)==0),
            extended_plateau_observed=bool(f[mask].min()<=fc[j]/2),
            rms_log10=float(np.sqrt(max(variance[j],0))),
            Mw_extended_with_observed_plateau=mw_plateau,
            delta_Mw_with_observed_plateau=mw_plateau-case['Mw'],
            falloff_extended_with_observed_plateau=float(n[jp]),
            fc_extended_with_observed_plateau_Hz=float(fc[jp])))
    frame=pd.DataFrame(rows);frame.to_csv(T/'snoke_falloff_sensitivity.csv',index=False)
    q=frame[frame.Mw_quality.eq('A')]
    review=q[q.delta_Mw.abs().gt(.1)|~q.extended_plateau_observed]
    review.to_csv(T/'snoke_model_review_events.csv',index=False)
    paired=pd.read_csv(T/'Mw_independent_ML_primary.csv',dtype={'event_id':str})
    relations=[]
    for name,exclude in [('All paired A',set()),('Exclude model-review cases',set(review.event_id))]:
        sample=paired[~paired.event_id.isin(exclude)]
        counts=sample.family_block.value_counts()
        w=1/sample.family_block.map(counts).to_numpy()
        x=sample.ML_Sheen.to_numpy()-3;y=sample.Mw.to_numpy()
        for degree in [1,2]:
            design=np.column_stack([x**i for i in range(degree+1)])
            coef=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
            relations.append(dict(selection=name,degree=degree,n_events=len(sample),n_families=len(counts),
                a_at_ML3=float(coef[0]),slope_at_ML3=float(coef[1]),quadratic_c=float(coef[2]) if degree==2 else 0.,
                **{f'prediction_ML{m}':float(np.polynomial.polynomial.polyval(m-3,coef)) for m in [2,3,4,5]}))
    pd.DataFrame(relations).to_csv(T/'snoke_Mw_ML_relation_sensitivity.csv',index=False)
    return dict(n_quality_A=len(q),n_original_falloff_lower_bound=int(q.falloff_adopted.eq(1.5).sum()),
        n_extended_falloff_lower_bound=int(q.extended_lower_falloff_bound.sum()),
        delta_Mw_quantiles={str(p):float(q.delta_Mw.quantile(p)) for p in [.05,.5,.95]},
        median_absolute_Mw_change=float(q.delta_Mw.abs().median()),
        n_absolute_Mw_change_gt_point1=int(q.delta_Mw.abs().gt(.1).sum()),
        n_gt_point1_also_loses_plateau=int((q.delta_Mw.abs().gt(.1)&~q.extended_plateau_observed).sum()),
        n_extended_plateau_extrapolated=int((~q.extended_plateau_observed).sum()),
        n_review_cases=len(review),
        constrained_plateau_delta_Mw_median=float(q.delta_Mw_with_observed_plateau.median()),
        constrained_plateau_delta_Mw_p95=float(q.delta_Mw_with_observed_plateau.quantile(.95)),
        constrained_plateau_n_abs_delta_gt_point1=int(q.delta_Mw_with_observed_plateau.abs().gt(.1).sum()),
        Mw_ML_relation_sensitivity=relations,
        median_fc_ratio=float(q.fc_ratio.median()),
        scope='Finite-band sensitivity to extending n from [1.5,4] to [1,4], same grid and source coefficient. Values n<=1.5 do not define finite infinite-band J; this is not an adopted physical source model.')


def main(falloff_only=False):
    protected_paths = [*list((R/'reports').glob('*.pdf')), *[R/name for name in [
        'tables/Mw_ML_catalog.csv','tables/Mw_ML_primary.csv','tables/Mw_independent_ML_catalog.csv',
        'tables/Mw_independent_ML_primary.csv','tables/clean_spectral_fits.csv','tables/Sangju2026_summary.json',
        'tables/potency_event_catalog.csv','tables/independent_ML_regressions.csv',
        'tables/Hong_regression_comparison.csv','inputs/study_config.json',
        'inputs/source_velocity_models.json','code/spectra.py','code/source_velocity.py']]]
    protected={str(p.relative_to(R)):sha(p) for p in protected_paths if p.name!='Snoke_coefficient_and_bandwidth_audit.pdf'}
    if falloff_only:
        previous=json.loads((T/'snoke_audit_summary.json').read_text())
        controls,empirical=previous['analytic_controls'],previous['empirical']
    else:
        controls=analytic_controls()
        empirical=catalog_diagnostics()
    empirical['falloff_bound_sensitivity']=falloff_controls()
    tree=ast.parse((R/'code/spectra.py').read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='fit')
    fit_text=ast.get_source_segment((R/'code/spectra.py').read_text(),function)
    assert 'om=(yy.sum()-phi@g)/ng' in fit_text and 'omega=float(10**om[j])' in fit_text
    assert not any(term in fit_text.lower() for term in ['snoke','trapz','trapezoid','quad('])
    audit=dict(paper=dict(title='Coefficient Consistency and Implementation Scope in Snoke’s Alternative K–J Spectral-Integral Parameterization',
        authors='Byeong Seok Ahn and Tae-Seob Kang',journal='The Seismic Record 6(4), 493–501',
        doi='10.1785/0320260027',publication_date='2026-11-18',
        supplied_pdf='references/Ahn2026_tsr.pdf',sha256=sha(R/'references/Ahn2026_tsr.pdf')),
        page_references=dict(coefficient_and_scope='493–495, equations 1–11',
            pathways='496–498, equations 12–19 and Tables 2–3',finite_band='498, 501, Appendix B',
            source_shape='501, Appendix A'),
        our_final_parameter_path='Profiled log spectral plateau in a direct generalized Boatwright fit; direct Brune is a sensitivity, neither uses K,J to obtain plateau or initialize its grid.',
        our_code_evidence=dict(path='code/spectra.py',function='fit',source=fit_text),
        coefficient_correction_applies_to_our_adopted_Mw=False,
        correction_applies_to_Hong_paper=False,
        Hong_scope='Paper-level audit: Hong and Sheen (2026), supplied PDF pp.4–7, directly fits the spectral and peak-amplitude-derived Brune models; no K,J final coefficient is specified. KR_Mw source code is not supplied, so this is not a code-level certification of that catalog.',
        Goesan_scope='Lim et al. (2024), section 2.2, PDF p.5, uses Snoke for initial search domain and a final Brune grid search; no fixed correction is justified without checking bounds/convergence. Author PDF: https://lowbontimp.github.io/papers/Lim_etal_2024_JS.pdf',
        FFT_scope='P1 is one-sided PSD. sqrt(T*P1/2) is the present PSD-to-undoubled-FAS conversion; the Snoke coefficient error is a separate algebraic issue. No cancellation or additional factor is inferred.',
        real_data_scope='The same retained, distance/Q-corrected log stack and frequency SNR gate are used. Integrate only the longest contiguous usable band, at least one decade/20 samples. Piecewise log-linear amplitudes; exact integral per segment. No observed holes are filled. Snoke Brune tails are iterated to convergence. Both window durations and both spectrum estimators are diagnostic checks.',
        limitations=['No measured M0 ground truth or empirical validation of Snoke moments.',
          'Source shapes differ: adopted generalized Boatwright is not Brune; high-frequency J can diverge at falloff=1.5.',
          'Q/site/response errors and coherent velocity-model errors are not calibrated here.',
          'Band ratios and unobserved Brune fractions are conditional on a fitted corner, not measured energy fractions.',
          'The paper’s 2%/0.006/3.85%/9.35% bounds compare two finite-band schemes under one noise-free Brune family; they do not bound absolute errors or other bandwidths.',
          'No source radius, radiated energy, or stress drop is adopted from this diagnostic.'],
        analytic_controls=controls,empirical=empirical,protected_sha256=protected)
    assert all(sha(R/name)==value for name,value in protected.items())
    (T/'snoke_audit_summary.json').write_text(json.dumps(audit,indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else float(x))+'\n')
    print(json.dumps(dict(analytic=controls,primary=empirical['primary_quality_A'],
                         sensitivity=empirical['upper_half_removal']),indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--falloff-only',action='store_true',help='Reuse the completed integral audit for the additional source-model-bound sensitivity')
    main(parser.parse_args().falloff_only)
