"""Large-event prediction, magnitude normalization, depth trends and Mw curvature.

Extend the potency analysis on exactly the same paired-A sample. Reuse verified
fits where the numerical specification is identical; do not change adopted data.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import itertools
import json
import multiprocessing
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from potency_regression import run_case, prediction, weights_for

R = Path(__file__).resolve().parents[1]
T = R / 'tables'
YKEYS = ['Mp_Liu_Kim1983', 'Mp_Liu_Kim2011', 'Mw', 'Mw_source_Vs_Kim2011']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def paired_gain(frame, left, right, blocks='family_block', seed=20261123):
    """Uncertainty of paired held-out losses, conditional on fitted CV models."""
    mse = frame.groupby(blocks, sort=True)[[left, right]].agg(lambda s: np.mean(s*s)).to_numpy()
    n = len(mse)
    if n <= 6:
        draws = np.array(list(itertools.product(range(n), repeat=n)))
        scheme = f'Exhaustive paired empirical bootstrap, {n} families'
    else:
        draws = np.random.default_rng(seed).integers(n, size=(2500, n))
        scheme = '2500 paired held-out-family resamples'
    gain = np.sqrt(mse[draws, 0].mean(1)) - np.sqrt(mse[draws, 1].mean(1))
    return dict(gain_family_RMSE=float(np.sqrt(mse[:, 0].mean())-np.sqrt(mse[:, 1].mean())),
                gain_lo=float(np.quantile(gain, .025)), gain_hi=float(np.quantile(gain, .975)),
                gain_interval_method=scheme)


def blocks_bootstrap(X, y, blocks, reps):
    """Cluster bootstrap with every sampled family retaining its total weight."""
    _, inverse, n = np.unique(blocks, return_inverse=True, return_counts=True)
    w = 1/n[inverse]
    xx = np.zeros((len(n), X.shape[1], X.shape[1]))
    xy = np.zeros((len(n), X.shape[1]))
    np.add.at(xx, inverse, w[:, None, None]*X[:, :, None]*X[:, None, :])
    np.add.at(xy, inverse, w[:, None]*X*y[:, None])
    counts = np.random.default_rng(20261124).multinomial(len(n), np.ones(len(n))/len(n), size=reps)
    matrices = np.einsum('bi,ijk->bjk', counts, xx)
    rhs = counts@xy
    # Missing spatial cells can make nuisance intercepts non-identifiable in a
    # bootstrap sample. The minimum-norm solution retains the estimable depth
    # coefficient. Refit examples independently in validate_magnitude_followup.
    beta = np.einsum('bij,bj->bi', np.linalg.pinv(matrices, rcond=1e-12), rhs)
    return beta


def depth_case(frame, xkey, ykey, control, case_id):
    q = frame.copy()
    if control == 'Without Gyeongju/Pohang':
        q = q[~q.family_block.isin(['family_0', 'family_1'])].copy()
    if control == 'Depth below 15 km':
        q = q[q.z_central < 15].copy()
    if control == 'Depth at least 15 km':
        q = q[q.z_central >= 15].copy()
    z = q.z_central.to_numpy()
    yy = q[ykey].to_numpy()
    if control in ['Lower depth endpoints', 'Upper depth endpoints']:
        endpoint = 'lower' if control.startswith('Lower') else 'upper'
        z = q[endpoint].to_numpy()
        if ykey.startswith('Mp_'):
            model = 'Kim2011' if ykey.endswith('Kim2011') else 'Kim1983'
            yy = q['Mp_Liu_depth_'+endpoint+'_'+model].to_numpy()
        else:
            model = 'Kim2011' if ykey.endswith('Kim2011') else 'Kim1983'
            yy = q['Mw_at_depth_'+endpoint+'_'+model].to_numpy()
    m = q[xkey].to_numpy()-3
    names = ['intercept']
    xx = [np.ones(len(q))]
    if control == 'Raw magnitude difference':
        yy = yy-q[xkey].to_numpy()
    else:
        xx.append(m); names.append('ML_minus_3')
        if control != 'ML linear':
            xx.append(m*m); names.append('ML_minus_3_squared')
    if control == 'Geographic plane':
        xx += [(q.lat-36).to_numpy(), (q.lon-128).to_numpy()]
        names += ['latitude_minus_36', 'longitude_minus_128']
    if control == 'Station distance and sensors':
        xx += [np.log10(q.ML_median_distance.to_numpy()/50),
               q.ML_HG_fraction.to_numpy(), q.Mw_HG_fraction.to_numpy()]
        names += ['log10_ML_median_distance_over_50', 'ML_HG_fraction', 'Mw_HG_fraction']
    if control in ['One-degree spatial cells', 'Half-degree spatial cells']:
        size = 1 if control.startswith('One') else .5
        cells = (np.floor(q.lat/size).astype(int).astype(str)+'_'+
                 np.floor(q.lon/size).astype(int).astype(str))
        for label, col in pd.get_dummies(cells, dtype=float, drop_first=True).items():
            xx.append(col.to_numpy()); names.append('cell_'+str(label))
    if control == 'Layer intercepts':
        for name, top in [('Kim1983_layer_15', 15), ('Kim2011_layer_7p29', 7.29),
                          ('Kim2011_layer_20p7', 20.7)]:
            xx.append((z >= top).astype(float)); names.append(name)
    if control == 'ML quadratic with depth hinge':
        xx.append(np.maximum(z-15, 0)/10); names.append('depth_hinge_15km')
    xx.append((z-15)/10); names.append('depth_10km')
    X = np.column_stack(xx)
    blocks = q.family_block.to_numpy()
    w = weights_for(blocks, 'family')
    assert np.isfinite(X).all() and np.isfinite(yy).all(), (xkey, ykey, control)
    beta = np.linalg.lstsq(X*np.sqrt(w[:, None]), yy*np.sqrt(w), rcond=None)[0]
    reps = 5000 if control in ['Raw magnitude difference', 'ML linear', 'ML quadratic'] else 2500
    samples = blocks_bootstrap(X, yy, blocks, reps)
    ci = np.quantile(samples[:, -1], [.025, .975])
    row = dict(case_id=case_id, x=xkey, y=ykey, control=control, n_events=len(q),
               n_families=len(np.unique(blocks)), depth_min=float(z.min()), depth_max=float(z.max()),
               depth_slope_per_10km=float(beta[-1]), depth_lo=float(ci[0]), depth_hi=float(ci[1]),
               depth_origin_km=15, names=json.dumps(names), coefficients=json.dumps(beta.tolist()),
               bootstrap_reps=reps, fit_RMSE=float(np.sqrt(np.average((yy-X@beta)**2, weights=w))),
               Spearman_depth_response=float(spearmanr(z, yy).statistic),
               endpoint_status='Sensitivity, not a depth posterior' if 'endpoints' in control else '')
    if 'ML_minus_3_squared' in names:
        k = names.index('ML_minus_3_squared')
        bounds = np.quantile(samples[:, k], [.025, .975])
        row.update(quadratic_c=float(beta[k]), c_lo=float(bounds[0]), c_hi=float(bounds[1]))
    if 'depth_hinge_15km' in names:
        k = names.index('depth_hinge_15km')
        bounds = np.quantile(samples[:, k]+samples[:, -1], [.025, .975])
        row.update(deep_slope_per_10km=float(beta[k]+beta[-1]),
                   deep_slope_lo=float(bounds[0]), deep_slope_hi=float(bounds[1]))
    predictions = []
    if control in ['ML linear', 'ML quadratic', 'ML quadratic with depth hinge'] and ykey in YKEYS:
        without = np.empty(len(q)); with_depth = np.empty(len(q))
        for key in np.unique(blocks):
            test = blocks == key; train = ~test
            ww = weights_for(blocks[train], 'family')
            for has_depth, output in [(False, without), (True, with_depth)]:
                design = X if has_depth else X[:, [i for i,n in enumerate(names) if not n.startswith('depth_')]]
                b = np.linalg.lstsq(design[train]*np.sqrt(ww[:, None]), yy[train]*np.sqrt(ww), rcond=None)[0]
                output[test] = design[test]@b
        loss = pd.DataFrame(dict(family_block=blocks, without=yy-without, with_depth=yy-with_depth))
        row.update(paired_gain(loss, 'without', 'with_depth'))
        for label, values in [('without', without), ('with_depth', with_depth)]:
            err = yy-values
            row['CV_'+label+'_RMSE_events'] = float(np.sqrt(np.mean(err**2)))
            row['CV_'+label+'_RMSE_families'] = float(np.sqrt(pd.Series(err**2).groupby(blocks).mean().mean()))
        predictions = [dict(case_id=case_id, event_id=e, family_block=b, observed=float(v),
                            predicted_without_depth=float(p), predicted_with_depth=float(pz), depth_km=float(zz))
                       for e, b, v, p, pz, zz in zip(q.event_id, blocks, yy, without, with_depth, z)]
    return row, samples, predictions


def main():
    baseline = json.loads((T/'potency_provenance.json').read_text())['protected_current_results_sha256']
    protected = {name:sha(R/name) for name in baseline}
    for name, value in protected.items():
        assert value == baseline[name], name
    d = pd.read_csv(T/'potency_event_catalog.csv', dtype={'event_id':str})
    g = d[d.ML_comparison_primary].copy()
    base = pd.read_csv(T/'Mw_independent_ML_catalog.csv', dtype={'event_id':str}).set_index('event_id')
    extra = ['lat','lon','place_ko','ML_median_distance','n_stations','ML_n_HG','ML_Sheen_n','n_HG',
             'Mw_at_depth_lower_Kim1983','Mw_at_depth_lower_Kim2011',
             'Mw_at_depth_upper_Kim1983','Mw_at_depth_upper_Kim2011']
    for key in extra:
        g[key] = g.event_id.map(base[key])
    g['ML_HG_fraction'] = g.ML_n_HG/g.ML_Sheen_n
    g['Mw_HG_fraction'] = g.n_HG/g.n_stations
    g.to_csv(T/'magnitude_followup_events.csv', index=False)
    existing = pd.read_csv(T/'potency_regressions.csv')
    old_predictions = pd.read_csv(T/'potency_heldout_predictions.csv', dtype={'event_id':str})
    old_boot = np.load(T/'potency_bootstrap.npz')
    tasks = []; specs = []
    for x in ['ML_Sheen','ML_KMA']:
        for y in YKEYS:
            for method, weight, bins, minimum in [
                ('OLS','event',50,1),('OLS','family',50,1),
                ('ODR','event',50,1),('ODR','family',50,1),
                ('BinnedOLS','event',10,1),('BinnedOLS','event',20,1),
                ('BinnedOLS','event',50,1),('BinnedOLS','event',50,3)]:
                for degree in [1,2]:
                    reps = 2000 if method == 'OLS' else 1000 if method == 'ODR' or bins == 50 else 0
                    specs.append(dict(case_id=f'magnitude_{len(specs):03d}', selection='Paired A',x=x,y=y,
                        method=method,weighting=weight,degree=degree,n_bins=bins,minimum=minimum,
                        bootstrap_reps=reps,do_cv=True,block_policy='Whole national event-family blocks'))
    rows = []; boot = {}; predictions = []
    for spec in specs:
        mask = np.ones(len(existing), dtype=bool)
        for key in ['selection','x','y','method','weighting','degree','n_bins','minimum']:
            mask &= existing[key].eq(spec[key]).to_numpy()
        found = existing[mask]
        if len(found) == 1 and found.iloc[0].bootstrap_reps >= spec['bootstrap_reps']:
            old = found.iloc[0]; row = {**old.to_dict(), **spec, 'reused_input_case_id':old.case_id}
            pp = old_predictions[old_predictions.case_id.eq(old.case_id)].copy()
            assert len(pp) == len(g)
            pp['case_id'] = spec['case_id']
            rows.append(row); predictions += pp.to_dict('records')
            if spec['bootstrap_reps']:
                boot[spec['case_id']] = old_boot[old.case_id][:spec['bootstrap_reps']]
        else:
            tasks.append((spec,g[spec['x']].to_numpy(),g[spec['y']].to_numpy(),
                          g.family_block.to_numpy(),g.event_id.to_numpy()))
    print('Reuse', len(rows), 'fits; compute', len(tasks), 'additional fits.', flush=True)
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('fork')) as pool:
        for i,(row, samples, pp, _, _) in enumerate(pool.map(run_case,tasks),1):
            rows.append(row); predictions += pp
            if len(samples):boot[row['case_id']] = samples
            if i % 8 == 0: print('New fits',i,'/',len(tasks),flush=True)
    fits = pd.DataFrame(rows).sort_values('case_id')
    fits.to_csv(T/'magnitude_followup_regressions.csv',index=False)
    np.savez_compressed(T/'magnitude_followup_bootstrap.npz',**boot)
    pred = pd.DataFrame(predictions)
    pred.to_csv(T/'magnitude_followup_predictions.csv',index=False)
    metrics = []; comparisons = []; tail_rows = []
    for _, f in fits.iterrows():
        q = g[['event_id','family_block',f.x,f.y]].copy()
        q['fit_residual'] = q[f.y]-prediction(np.array([f.a_at_ML3,f.slope_at_ML3,f.quadratic_c]),q[f.x])
        pp = pred[pred.case_id.eq(f.case_id)].set_index('event_id')
        q['CV_residual'] = q.event_id.map(pp.residual)
        q['in_training_range'] = q.event_id.map(pp.in_training_range)
        for label, select in [('All',np.ones(len(q),bool)),('ML <= 4',q[f.x] <= 4),('ML > 4',q[f.x] > 4)]:
            part = q[select]
            metrics.append(dict(case_id=f.case_id,x=f.x,y=f.y,method=f.method,weighting=f.weighting,
                degree=f.degree,n_bins=f.n_bins,minimum=f.minimum,subset=label,n_events=len(part),
                n_families=part.family_block.nunique(),fit_RMSE_events=float(np.sqrt(np.mean(part.fit_residual**2))),
                CV_RMSE_events=float(np.sqrt(np.mean(part.CV_residual**2))),
                CV_RMSE_families=float(np.sqrt(part.groupby('family_block').CV_residual.agg(lambda s:np.mean(s*s)).mean())),
                n_extrapolation=int((~part.in_training_range).sum())))
        if f.degree != 2:continue
        match = fits[(fits.x==f.x)&(fits.y==f.y)&(fits.method==f.method)&(fits.weighting==f.weighting)&
                     fits.degree.eq(1)&fits.n_bins.eq(f.n_bins)&fits.minimum.eq(f.minimum)]
        assert len(match) == 1
        linear = match.iloc[0]
        lp = pred[pred.case_id.eq(linear.case_id)].set_index('event_id')
        q['linear_residual'] = q.event_id.map(lp.residual)
        for label, select in [('All',np.ones(len(q),bool)),('ML <= 4',q[f.x] <= 4),('ML > 4',q[f.x] > 4)]:
            part=q[select]
            comparisons.append(dict(x=f.x,y=f.y,method=f.method,weighting=f.weighting,n_bins=f.n_bins,
                minimum=f.minimum,subset=label,n_events=len(part),n_families=part.family_block.nunique(),
                linear_case_id=linear.case_id,quadratic_case_id=f.case_id,
                **paired_gain(part,'linear_residual','CV_residual')))
        if f.method == 'OLS' and f.weighting == 'family':
            for z in q[q[f.x] > 4].itertuples(index=False):
                eid=z.event_id
                tail_rows.append(dict(x=f.x,y=f.y,event_id=eid,ML=float(g.set_index('event_id').loc[eid,f.x]),
                    family_block=z.family_block,linear_residual=float(z.linear_residual),quadratic_residual=float(z.CV_residual),
                    in_training_range=bool(z.in_training_range)))
    pd.DataFrame(metrics).to_csv(T/'magnitude_followup_prediction_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(T/'magnitude_followup_CV_comparison.csv',index=False)
    pd.DataFrame(tail_rows).to_csv(T/'magnitude_followup_large_events.csv',index=False)
    coverage = base[base.ML_KMA >= 4].reset_index()
    coverage[['event_id','place_ko','ML_KMA','ML_Sheen','Mw','Mw_quality','Mw_reasons','ML_quality',
              'ML_comparison_primary','plateau_observed','fc_Hz','band_low_Hz','band_high_Hz']].to_csv(T/'magnitude_followup_large_event_coverage.csv',index=False)
    print('Magnitude fits and large-event metrics complete; fitting depth controls.',flush=True)
    depth_rows=[]; depth_boot={}; depth_predictions=[]
    for x in ['ML_Sheen','ML_KMA']:
        for y in YKEYS+['Mp_Liu_fixed_Vs3500','Mw_reference_Vs3500']:
            controls=['Raw magnitude difference','ML linear','ML quadratic']
            if y in YKEYS:controls += ['ML quadratic with depth hinge']
            if x=='ML_Sheen' and y in YKEYS:
                controls += ['Geographic plane','Station distance and sensors','One-degree spatial cells',
                    'Half-degree spatial cells','Without Gyeongju/Pohang','Depth below 15 km',
                    'Depth at least 15 km','Layer intercepts','Lower depth endpoints','Upper depth endpoints']
            for control in controls:
                case_id=f'depth_{len(depth_rows):03d}'
                row,samples,pp=depth_case(g,x,y,control,case_id)
                depth_rows.append(row);depth_boot[case_id]=samples;depth_predictions += pp
            print('Depth controls',x,y,'complete.',flush=True)
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(T/'magnitude_followup_depth_regressions.csv',index=False)
    np.savez_compressed(T/'magnitude_followup_depth_bootstrap.npz',**depth_boot)
    pd.DataFrame(depth_predictions).to_csv(T/'magnitude_followup_depth_predictions.csv',index=False)
    bins=[]
    for x in ['ML_Sheen','ML_KMA']:
        for y in YKEYS:
            f=fits[(fits.x==x)&(fits.y==y)&fits.method.eq('OLS')&fits.weighting.eq('family')&fits.degree.eq(2)].iloc[0]
            q=g.copy();q['delta']=q[y]-q[x]
            q['residual']=q[y]-prediction(np.array([f.a_at_ML3,f.slope_at_ML3,f.quadratic_c]),q[x])
            for lo,hi in [(5,10),(10,15),(15,20),(20,25)]:
                part=q[q.z_central.ge(lo)&q.z_central.lt(hi)]
                bins.append(dict(x=x,y=y,lower_km=lo,upper_km=hi,n_events=len(part),
                    n_families=part.family_block.nunique(),median_delta=float(part.delta.median()),
                    median_ML_adjusted_residual=float(part.residual.median())))
    pd.DataFrame(bins).to_csv(T/'magnitude_followup_depth_bins.csv',index=False)
    source_summary=dict(
        reference='https://files.scec.org/s3fs-public/reports/2024/24169_report.pdf?h64GRlv2wy8iT.1M_nQV_Hvym9y9hCnX=',
        description='Primary author project report documenting Trugman and Ben-Zion (2024), accessed through web tool.',
        pages_4_5='Assumed nominal 36 GPa; MP equivalent to Mw at that reference rigidity by design. Their 2024 quadratic fits use ODR on bin medians, unlike the supplied Liu2026 binned ordinary least-squares procedure.',
        ML_status='Instrumental ML is related empirically; MP is not defined to equal ML for all sizes, depths or regions.',
        Trugman2026='Supplied PDF p.5 equation 5 chooses 28 GPa as representative of its models and allows a common Mw-like scale; it is not fitted to Korean ML.',
        C9p05='Our adopted Mw retains C=9.05. Liu MP is approximately the 36-GPa-reference Mw at C=9.10, including the corresponding -0.033333 magnitude offset.',
        approximate_Liu_formula=dict(ML_2=float(.04*2**2+.556*2+1.133),ML_3p5=float(.04*3.5**2+.556*3.5+1.133)))
    (T/'magnitude_followup_normalization.json').write_text(json.dumps(source_summary,indent=2)+'\n')
    summary=dict(n_events=len(g),n_families=g.family_block.nunique(),n_fits=len(fits),
        n_reused_fits=len(rows)-len(tasks),n_depth_fits=len(depth),
        depth_range=[float(g.z_central.min()),float(g.z_central.max())],
        n_measured_ML_above_4=int((g.ML_Sheen>4).sum()),
        n_measured_ML_above_4_families=g[g.ML_Sheen>4].family_block.nunique(),
        n_KMA_ML_above_4=int((g.ML_KMA>4).sum()),
        n_KMA_ML_above_4_families=g[g.ML_KMA>4].family_block.nunique(),
        methods='Identical paired-A events and family blocks; same OLS/ODR/binned specifications for Mp and Mw; leave whole families out.',
        interval_scope='Paired predictive-gain intervals resample existing held-out family losses. Depth-coefficient intervals resample/refit complete families. Conditional on selected events and measurement/velocity/depth assumptions.',
        limitations='Only five independent families have measured ML>4. Depth endpoints are sensitivities, not posterior samples; depth trends are observational and may reflect path/site/ML/source-estimation effects.',
        protected_outputs_sha256=protected,
        inputs_sha256={name:sha(T/name) for name in ['potency_event_catalog.csv','potency_regressions.csv',
            'potency_heldout_predictions.csv','Mw_independent_ML_catalog.csv']})
    for name,value in protected.items():assert sha(R/name)==value,name
    (T/'magnitude_followup_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith('sha256')},indent=2),flush=True)


if __name__ == '__main__':
    main()
