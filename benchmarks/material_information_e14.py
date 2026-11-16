"""Conditional finite-secant information diagnostics, not a rendering model.

Gaussian noise is hypothetical, fixed in linear radiance. Deterministic codes
have intervals, not Fisher information. No probability floor or synthetic noise.
"""
import numpy as np
from scipy.special import log_ndtr
from infrastructure.signatures import require

SIGMA=.001


def rotated_light(light):
    axis=np.ones(3)/np.sqrt(3.);angle=np.pi/4
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    R=np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*K
    result={'ambient':list(light['ambient']),'lobes':[]}
    for old in light['lobes']:
        result['lobes'].append(dict(axis=(R@np.array(old['axis'])).tolist(),RGB=list(old['RGB']),power=old['power']))
    require(np.max(abs(R.T@R-np.eye(3)))<1e-14 and abs(np.linalg.det(R)-1)<1e-14,'Invalid prescribed rotation')
    return result


def linear_bin_edges():
    s=(np.arange(255,dtype=np.float64)+.5)/255
    linear=np.where(s<=.04045,s/12.92,((s+.055)/1.055)**2.4)
    return np.r_[-np.inf,linear,np.inf]


def deterministic_codes(mu):
    require(np.isfinite(mu).all(),'Nonfinite noiseless radiance')
    x=np.clip(mu,0,1)
    srgb=np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,np.finfo(np.float32).eps)**(1/2.4)-.055)
    return np.rint(srgb*255).astype(np.uint8)


def gaussian_bins(mu,sigma=SIGMA,edges=None):
    """Stable log-space probabilities, derivatives and quantized information.

    Tail differences use survival functions on the positive half-line. Both
    probability and phi differences use expm1, and I is formed from log(dP)^2/P.
    Underflow is reported; zero floating probabilities are never floored/divided.
    """
    mu=np.asarray(mu,dtype=np.float64).reshape(-1)
    e=linear_bin_edges() if edges is None else np.asarray(edges,dtype=float)
    require(np.isfinite(mu).all() and np.isfinite(sigma) and sigma>0 and e.ndim==1 and
            len(e)>=2 and e[0]==-np.inf and e[-1]==np.inf and np.all(np.diff(e)>0),'Invalid Gaussian/bin definition')
    a=(e[:-1][None]-mu[:,None])/sigma;b=(e[1:][None]-mu[:,None])/sigma
    positive=a>=0
    high=np.where(positive,log_ndtr(-a),log_ndtr(b))
    low=np.where(positive,log_ndtr(-b),log_ndtr(a))
    with np.errstate(divide='ignore',under='ignore'):
        logp=high+np.log(-np.expm1(low-high))
    require(np.isfinite(logp).all(),'Unresolved Gaussian tail probability')
    lp_a=-a*a/2-.5*np.log(2*np.pi);lp_b=-b*b/2-.5*np.log(2*np.pi)
    sign=(lp_a>lp_b).astype(float)-(lp_a<lp_b).astype(float)
    equal=lp_a==lp_b
    logd=np.full_like(logp,-np.inf)
    different=~equal
    largest=np.maximum(lp_a[different],lp_b[different]);smallest=np.minimum(lp_a[different],lp_b[different])
    with np.errstate(divide='ignore',under='ignore'):
        logd[different]=largest+np.log(-np.expm1(smallest-largest))-np.log(sigma)
        p=np.exp(logp);derivative=sign*np.exp(logd);terms=np.exp(2*logd-logp)
    information=terms.sum(1)
    require(np.isfinite(information).all() and np.max(abs(p.sum(1)-1))<=5e-13 and
            np.max(abs(derivative.sum(1)))*sigma<=5e-12,'Gaussian normalization/derivative identity failed')
    require(np.all(information>=0) and np.all(information<=sigma**-2*(1+1e-10)),
            'Quantization data-processing inequality failed; do not clamp information')
    return dict(probabilities=p,dP_dmu=derivative,information=information,
                probability_sum_max_error=float(np.max(abs(p.sum(1)-1))),
                derivative_sum_max_sigma_scaled_error=float(np.max(abs(derivative.sum(1)))*sigma),
                underflow_probability_bins=int((p==0).sum()))


def secants(components):
    rgb=components[...,2,:]
    return np.stack(((rgb[...,1,:]-rgb[...,2,:])/(np.pi/6),
                     (rgb[...,8,:]-rgb[...,9,:])/.2,
                     (rgb[...,10,:]-rgb[...,11,:])/.2),-1)


def matrix_summary(J):
    require(J.ndim==2 and J.shape[1]==3 and np.isfinite(J).all(),'Invalid response matrix')
    u,s,v=np.linalg.svd(J,full_matrices=False)
    weak=v[-1].copy();weak*=1 if weak[np.argmax(abs(weak))]>=0 else -1
    norm=np.linalg.norm(J,axis=0)
    pairs={}
    for i,j,name in ((0,1,'theta_logmean'),(0,2,'theta_logratio'),(1,2,'logmean_logratio')):
        pairs[name]=float(abs(J[:,i]@J[:,j])/(norm[i]*norm[j])) if norm[i]*norm[j]>0 else None
    rank=bool(s[-1]>max(J.shape)*np.finfo(float).eps*s[0])
    covariance=(v.T*(1/s**2))@v if rank else None
    condition=float(s[0]/s[-1]) if s[-1]>0 else None
    gram_condition=condition**2 if condition is not None and condition<np.sqrt(np.finfo(float).max) else None
    return dict(singular_values=s.tolist(),condition=condition,gram_condition=gram_condition,
                column_norms=norm.tolist(),absolute_column_cosines=pairs,weakest_direction=weak.tolist(),
                eigenvalues_JtJ=(s*s).tolist(),
                inverse_gram_axis_scales=np.sqrt(np.diag(covariance)).tolist() if rank else None,
                invertibility='machine_precision_full_rank' if rank else 'numerically_rank_deficient')


def line_interval(mu,J,direction):
    """Section through the truth in the secant-linearized quantization strips.

    Other coordinates are fixed along this line: these are NOT joint/marginal
    feasible parameter ranges or confidence intervals for the nonlinear BRDF.
    """
    code=deterministic_codes(mu);e=linear_bin_edges()
    lo=e[code.astype(int)];hi=e[code.astype(int)+1]
    slope=J@direction;active=slope!=0
    a=(lo[active]-mu[active])/slope[active];b=(hi[active]-mu[active])/slope[active]
    lower=float(np.maximum.reduce(np.minimum(a,b),initial=-np.inf))
    upper=float(np.minimum.reduce(np.maximum(a,b),initial=np.inf))
    require(lower<=0<=upper,'Truth outside its deterministic quantization strips')
    return dict(lower=lower if np.isfinite(lower) else None,upper=upper if np.isfinite(upper) else None,
                width=upper-lower if np.isfinite(lower) and np.isfinite(upper) else None,
                lower_unbounded=not np.isfinite(lower),upper_unbounded=not np.isfinite(upper))


def information_row(fine,coarse):
    """Input V/L selected observations x12 cases x3 components xRGB."""
    require(fine.shape==coarse.shape and fine.ndim==4 and fine.shape[1:]==(12,3,3) and len(fine)>0,
            'Nonempty same-observation paired captures required')
    require(np.isfinite(fine).all() and np.isfinite(coarse).all(),'Nonfinite admitted observations')
    n=len(fine);J=secants(fine).reshape(-1,3);K=secants(coarse).reshape(-1,3)
    mu=fine[:,0,2,:].reshape(-1);low_mu=coarse[:,0,2,:].reshape(-1)
    q=gaussian_bins(mu);ql=gaussian_bins(low_mu)
    result=dict(rgb_observations=n,scalar_channels=3*n,coordinates=['theta_radians','log_geometric_mean','log_ratio'],
                sigma_linear_RGB=SIGMA,noise_is_measured=False)
    for name,w,wl in [('ideal',np.ones(len(mu)),np.ones(len(mu))),
                      ('gaussian_linear',np.full(len(mu),1/SIGMA**2),np.full(len(mu),1/SIGMA**2)),
                      ('gaussian_quantized8',q['information'],ql['information'])]:
        A=J*np.sqrt(w)[:,None];B=K*np.sqrt(wl)[:,None]
        normalized=A/np.sqrt(n);stats=matrix_summary(normalized)
        delta=float(np.linalg.norm((A-B)/np.sqrt(n),ord=2))
        stats.update(grid_difference_operator_norm_RMS=delta,
                     grid_proxy_smin_lower=max(0.,stats['singular_values'][-1]-delta),
                     total_JtJ=(A.T@A).tolist(),per_RGB_observation_JtJ=(A.T@A/n).tolist())
        if name!='ideal':
            # Total data uncertainty scales use all observations, not one average observation.
            stats['total_data_local_std']=(np.array(stats['inverse_gram_axis_scales'])/np.sqrt(n)).tolist() if stats['inverse_gram_axis_scales'] is not None else None
            stats['total_information_eigenvalues']=(np.array(stats['eigenvalues_JtJ'])*n).tolist()
        else:
            stats.pop('inverse_gram_axis_scales')
            stats['interpretation']='Unweighted finite-secant Gram matrix; not a calibrated Fisher information matrix.'
        result[name]=stats
    code=deterministic_codes(mu);e=linear_bin_edges();bin_lo=e[code.astype(int)];bin_hi=e[code.astype(int)+1]
    widths=bin_hi-bin_lo
    result['deterministic8']=dict(clipped_channels=int(((mu<0)|(mu>1)).sum()),
        truth_codes=code.tolist(),analogue_bin_lower=[float(x) if np.isfinite(x) else None for x in bin_lo],
        analogue_bin_upper=[float(x) if np.isfinite(x) else None for x in bin_hi],
        finite_bin_width_min_median_max=np.quantile(widths[np.isfinite(widths)],[0,.5,1]).tolist() if np.isfinite(widths).any() else None,
        coarse_truth_code_changes=int((code!=deterministic_codes(low_mu)).sum()),
        identical_code_counterfactuals=[bool(np.array_equal(deterministic_codes(fine[:,case,2,:]),code.reshape(n,3))) for case in range(12)],
        coordinate_line_sections={name:line_interval(mu,J,np.eye(3)[i]) for i,name in enumerate(result['coordinates'])},
        weakest_ideal_line_section=line_interval(mu,J,np.array(result['ideal']['weakest_direction'])),
        interpretation='Noiseless code intervals; line sections are conditional secant-linearized, not joint/global uncertainty.')
    result['quantized_model_checks']={k:q[k] for k in ('probability_sum_max_error','derivative_sum_max_sigma_scaled_error','underflow_probability_bins')}
    result['quantized_model_checks']['maximum_information_fraction']=float(np.max(q['information'])*SIGMA**2)
    return result
