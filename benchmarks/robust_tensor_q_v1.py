"""Pseudo-Huber on weighted transported Q and fitting-support calibration."""
import math

import numpy as np

from infrastructure.signatures import require

CALIBRATION = dict(step=10000, radius_native=.01, maximum_normal_degrees=30., quantile=.9,
                   quantile_method='linear', minimum_pairs=32,
                   support='both reserved-edge endpoints near fitting observations')
TRANSITION_EDGES_NATIVE = (0.,.005,.01,.02,.035)


def terms(s, delta, torch):
    """Return rho(s) and its local multiplier on the quadratic gradient."""
    radius_squared = delta*delta
    require(math.isfinite(radius_squared) and radius_squared>0, 'Positive finite pseudo-Huber scale required')
    root = (s+radius_squared).sqrt()
    # Algebraically equal to the requested form, without s/delta² or 2*s overflow.
    penalty = s*((2*delta)/(root+delta))
    return penalty,delta/root


def fitting_support(points, normals, positions, observed_normals):
    from scipy.spatial import cKDTree

    radius = CALIBRATION['radius_native']
    cosine = np.cos(np.deg2rad(CALIBRATION['maximum_normal_degrees']))
    neighbors = cKDTree(positions).query_ball_point(points,radius)
    return np.asarray([bool(ids) and bool((observed_normals[ids]@normal>=cosine).any())
                       for ids,normal in zip(neighbors,normals)],dtype=bool)


def calibrate(domain, prediction_square, positions, normals, spatial_policy, weighting):
    ids = domain['diagnostic_ids']
    require(prediction_square.shape==ids.shape and np.isfinite(prediction_square).all()
            and (prediction_square>=0).all(), 'Invalid archived prediction differences')
    support = [fitting_support(domain['points'+side][ids],domain['normals'+side][ids],positions,normals)
               for side in ('0','1')]
    pick = support[0]&support[1]
    require(int(pick.sum())>=CALIBRATION['minimum_pairs'], 'Insufficient fitting-supported pseudo-Huber calibration pairs')
    selected = ids[pick]
    w = (spatial_policy['reference_distance']/domain['distance'][selected])**2
    if weighting is not None:
        w = np.minimum(w,weighting['cap'])*weighting['scale']
    amp = np.sqrt(w*prediction_square[pick])
    delta = float(np.quantile(amp,CALIBRATION['quantile']))
    require(math.isfinite(delta) and delta>0, 'Invalid calibrated delta')
    return dict(delta=delta,selected_pair_ids=selected.tolist(),samples=len(selected),
                amplitude_quantiles=dict(zip(('min','p50','p90','p95','max'),
                    np.quantile(amp,[0,.5,.9,.95,1]).tolist())))


def reduce(s, delta, coefficient, torch):
    penalty, derivative = terms(s,delta,torch)
    with torch.no_grad():
        raw_mean = s.mean()
        rho_mean = penalty.mean()
        ratio = torch.where(raw_mean>0,rho_mean/raw_mean,torch.ones_like(raw_mean))
        values = torch.stack((ratio,derivative.mean(),derivative.min(),(s>delta*delta).to(s.dtype).mean()))
    return (coefficient*penalty.mean(),coefficient*s.mean(),
            penalty.detach()*(coefficient/len(penalty)),values)


def width_profiles(distance, region, prediction, oracle, weights):
    """Fixed boundary-distance bins, with signs assigned by material region."""
    distance = np.asarray(distance)
    region = np.asarray(region)
    require(prediction.shape==oracle.shape==(len(distance),2) and region.shape==weights.shape==distance.shape
            and np.isfinite(distance).all() and (distance>=0).all()
            and np.isin(region,[0,1]).all() and np.isfinite(weights).all() and (weights>0).all(),
            'Invalid transition profile observations')
    rows = []
    for side in (0,1):
        for lo,hi in zip(TRANSITION_EDGES_NATIVE,TRANSITION_EDGES_NATIVE[1:]):
            upper = distance<=hi if hi==TRANSITION_EDGES_NATIVE[-1] else distance<hi
            keep = (region==side)&(distance>=lo)&upper
            row = dict(region=side,signed_distance_interval=[lo,hi] if side==0 else [-hi,-lo],
                       observations=int(keep.sum()),observed_area_weight=float(weights[keep].sum()))
            for name,weight in (('observation',np.ones(int(keep.sum()))),('observed_triangle_area',weights[keep])):
                row[name] = {k:np.average(v[keep],weights=weight,axis=0).tolist()
                             for k,v in (('predicted_widths',prediction),('oracle_widths',oracle),
                                         ('signed_width_error',prediction-oracle))} if keep.any() else None
            rows.append(row)
    return rows
