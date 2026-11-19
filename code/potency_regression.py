"""Linear/quadratic fits, paper-style binning and sequence-held-out prediction."""
import warnings
import numpy as np
from scipy.stats import t as student_t

CENTER = 3.


def design(x, degree):
    return np.polynomial.polynomial.polyvander(np.asarray(x)-CENTER, degree)


def binned_medians(x, y, n_bins=50, minimum=1, edges=None):
    x=np.asarray(x);y=np.asarray(y)
    if edges is None:edges=np.linspace(x.min(),x.max(),n_bins+1)
    index=np.clip(np.searchsorted(edges,x,side='right')-1,0,len(edges)-2)
    rows=[]
    for i in range(len(edges)-1):
        chosen=index==i
        if not chosen.any():continue
        center=np.median(y[chosen]);keep=chosen&(abs(y-center)<=.5)
        if keep.sum()<minimum:continue
        rows.append([np.median(x[keep]),np.median(y[keep]),int(chosen.sum()),int(keep.sum()),i])
    return np.asarray(rows,dtype=float).reshape(-1,5),np.asarray(edges)


def fit(x,y,degree=2,method='OLS',weights=None,n_bins=50,minimum=1,edges=None):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    weights=np.ones(len(x)) if weights is None else np.asarray(weights)
    keep=(weights>0)&np.isfinite(x)&np.isfinite(y)
    x=x[keep];y=y[keep];weights=weights[keep]
    if method=='Published':return np.array([3.161,.796,.04]),{}
    if method=='PublishedOffset':
        original=.04*x*x+.556*x+1.133
        return np.array([3.161+np.average(y-original,weights=weights),.796,.04]),{}
    bins=None
    if method=='BinnedOLS':
        bins,edges=binned_medians(x,y,n_bins,minimum,edges)
        x=bins[:,0];y=bins[:,1];weights=np.ones(len(x))
    if len(np.unique(x))<=degree or len(x)<=degree+1:raise ValueError('Insufficient independent x values')
    X=design(x,degree)
    initial=np.linalg.lstsq(X*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0]
    meta={}
    if method=='ODR':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            from scipy import odr
        def fn(beta,xx):return design(xx,degree)@beta
        def jb(beta,xx):return design(xx,degree).T
        def jd(beta,xx):
            return np.full_like(xx,beta[1]) if degree==1 else beta[1]+2*beta[2]*(xx-CENTER)
        model=odr.Model(fn,fjacb=jb,fjacd=jd)
        data=odr.RealData(x,y,sx=1/np.sqrt(weights),sy=1/np.sqrt(weights))
        solver=odr.ODR(data,model,beta0=initial,maxit=500,sstol=1e-11,partol=1e-11)
        solver.set_job(deriv=3)
        result=solver.run();coef=result.beta
        if result.info not in (1,2,3):raise RuntimeError('ODR did not converge: '+str(result.stopreason))
        meta['orthogonal_RMS']=float(np.sqrt(result.sum_square/weights.sum()))
        if degree==1:
            mx=np.average(x,weights=weights);my=np.average(y,weights=weights)
            vx=np.average((x-mx)**2,weights=weights);vy=np.average((y-my)**2,weights=weights)
            cov=np.average((x-mx)*(y-my),weights=weights)
            b=(vy-vx+np.sqrt((vy-vx)**2+4*cov**2))/(2*cov)
            expected=np.array([my-b*(mx-CENTER),b])
            meta['ODR_linear_closed_form_error']=float(np.max(abs(expected-coef)))
        else:
            # Independent stationarity check of the closest-point coordinates.
            adjusted=x+result.delta
            stationarity=result.delta+jd(coef,adjusted)*(fn(coef,adjusted)-y)
            meta['ODR_projection_stationarity']=float(np.max(abs(stationarity)))
    else:
        coef=initial
    residual=y-X@coef
    if method=='BinnedOLS':
        variance=float(np.sum(residual**2)/(len(x)-len(coef)))
        covariance=np.linalg.inv(X.T@X)*variance
        half=student_t.ppf(.975,len(x)-len(coef))*np.sqrt(np.diag(covariance))
        meta.update(n_bins_fitted=len(x),n_singleton_bins=int((bins[:,3]==1).sum()),
                    n_events_retained=int(bins[:,3].sum()),student_t_low=(coef-half).tolist(),
                    student_t_high=(coef+half).tolist(),bin_edges=edges.tolist())
    coef=np.pad(coef,(0,3-len(coef)))
    return coef,meta


def weights_for(blocks, weighting):
    _,inv,n=np.unique(blocks,return_inverse=True,return_counts=True)
    return np.ones(len(blocks)) if weighting=='event' else 1/n[inv]


def prediction(coef,x):return design(x,2)@coef


def bootstrap_fit(x,y,blocks,spec,reps):
    if not reps:return np.empty((0,3))
    keys,inv,n=np.unique(blocks,return_inverse=True,return_counts=True)
    rng=np.random.default_rng(20261121)
    count=rng.multinomial(len(keys),np.ones(len(keys))/len(keys),size=reps)
    method=spec['method'];degree=spec['degree']
    if method=='OLS':
        X=design(x,degree);w=weights_for(blocks,spec['weighting'])
        xx=np.zeros((len(keys),degree+1,degree+1));xy=np.zeros((len(keys),degree+1))
        np.add.at(xx,inv,w[:,None,None]*X[:,:,None]*X[:,None,:])
        np.add.at(xy,inv,w[:,None]*X*y[:,None])
        mats=np.einsum('bi,ijk->bjk',count,xx);rhs=count@xy
        try:result=np.linalg.solve(mats,rhs[...,None])[...,0]
        except np.linalg.LinAlgError:result=np.array([np.linalg.lstsq(a,b,rcond=None)[0] for a,b in zip(mats,rhs)])
        return np.pad(result,((0,0),(0,3-result.shape[1])))
    result=[]
    edges=np.linspace(np.min(x),np.max(x),spec.get('n_bins',50)+1)
    for c in count:
        try:
            if method=='BinnedOLS':
                indices=np.repeat(np.arange(len(x)),c[inv])
                beta,_=fit(x[indices],y[indices],degree,method,n_bins=spec.get('n_bins',50),
                           minimum=spec.get('minimum',1),edges=edges)
            else:
                w=c[inv].astype(float)
                if spec['weighting']=='family':w=w/n[inv]
                beta,_=fit(x,y,degree,method,weights=w)
            if np.isfinite(beta).all():result.append(beta)
        except (ValueError,RuntimeError,np.linalg.LinAlgError):continue
    return np.asarray(result).reshape(-1,3)


def heldout(x,y,blocks,spec):
    pred=np.empty(len(x));interior=np.zeros(len(x),dtype=bool);coeff=[]
    for key in np.unique(blocks):
        test=blocks==key;train=~test
        w=weights_for(blocks[train],spec['weighting'])
        beta,_=fit(x[train],y[train],spec['degree'],spec['method'],w,
                   spec.get('n_bins',50),spec.get('minimum',1))
        pred[test]=prediction(beta,x[test])
        interior[test]=(x[test]>=x[train].min())&(x[test]<=x[train].max())
        coeff.append(dict(block=str(key),a=float(beta[0]),b=float(beta[1]),c=float(beta[2]),
                          train_min=float(x[train].min()),train_max=float(x[train].max())))
    return pred,interior,coeff


def run_case(task):
    spec,x,y,blocks,ids=task
    w=weights_for(blocks,spec['weighting'])
    coef,meta=fit(x,y,spec['degree'],spec['method'],w,spec.get('n_bins',50),spec.get('minimum',1))
    residual=y-prediction(coef,x)
    row={**spec,'n_events':len(x),'n_blocks':len(np.unique(blocks)),
         'x_min':float(x.min()),'x_max':float(x.max()),
         'a_at_ML3':float(coef[0]),'slope_at_ML3':float(coef[1]),'quadratic_c':float(coef[2]),
         'constant':float(coef[0]-3*coef[1]+9*coef[2]),
         'linear_b':float(coef[1]-6*coef[2]),
         'RMS_events':float(np.sqrt(np.mean(residual**2))),
         'RMS_weighted':float(np.sqrt(np.average(residual**2,weights=w))),
         'monotone_over_observed_range':bool(np.min(coef[1]+2*coef[2]*(np.array([x.min(),x.max()])-CENTER))>0)}
    for k in ['n_bins_fitted','n_singleton_bins','n_events_retained','ODR_linear_closed_form_error','ODR_projection_stationarity','orthogonal_RMS']:
        if k in meta:row[k]=meta[k]
    if 'student_t_low' in meta:
        low=np.pad(meta['student_t_low'],(0,3-len(meta['student_t_low'])))
        high=np.pad(meta['student_t_high'],(0,3-len(meta['student_t_high'])))
        row.update(student_t_curvature_lo=float(low[2]),student_t_curvature_hi=float(high[2]))
    boot=bootstrap_fit(x,y,blocks,spec,spec['bootstrap_reps'])
    if len(boot):
        ci=np.quantile(boot,[.025,.975],axis=0)
        row.update(a_lo=float(ci[0,0]),a_hi=float(ci[1,0]),b_lo=float(ci[0,1]),b_hi=float(ci[1,1]),
                   c_lo=float(ci[0,2]),c_hi=float(ci[1,2]),n_successful_bootstraps=len(boot))
    predictions=[];coeff=[]
    if spec['do_cv']:
        pred,interior,coeff=heldout(x,y,blocks,spec)
        err=y-pred
        row.update(CV_RMS_events=float(np.sqrt(np.mean(err**2))),
                   CV_RMS_families=float(np.sqrt(np.mean([np.mean(err[blocks==b]**2) for b in np.unique(blocks)]))),
                   CV_n_interpolation=int(interior.sum()),
                   CV_RMS_interpolation=float(np.sqrt(np.mean(err[interior]**2))))
        predictions=[dict(case_id=spec['case_id'],event_id=e,family_block=b,predicted=float(p),observed=float(yy),
                          residual=float(ee),in_training_range=bool(inside))
                     for e,b,p,yy,ee,inside in zip(ids,blocks,pred,y,err,interior)]
    bins,_=binned_medians(x,y,spec.get('n_bins',50),spec.get('minimum',1)) if spec['method']=='BinnedOLS' else (np.empty((0,5)),None)
    return row,boot,predictions,coeff,bins
