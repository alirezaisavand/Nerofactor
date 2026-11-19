"""Saved-data physical and spatial strata for the heterogeneous comparison."""
import numpy as np

from benchmarks import donut_heterogeneous_v1 as ref
from benchmarks import donut_heterogeneous_training_v1 as h
from infrastructure.signatures import require


def score(e, rgb, truth, weights):
    require(len(weights)==len(rgb)==len(truth) and np.isfinite(weights).all() and (weights>0).all(),
            'Observation weight shape or values differ')
    if not len(weights):
        return None

    def distribution(a,w):
        if not len(a):
            return None
        if (w==w[0]).all():
            median,p95=np.quantile(a,[.5,.95])
        else:
            order=np.argsort(a,kind='stable')
            cdf=np.cumsum(w[order])
            median,p95=a[order[np.searchsorted(cdf,np.asarray([.5,.95])*cdf[-1])]]
        return dict(mean=float(np.dot(a,w)/w.sum()),median=float(median),p95=float(p95))

    angular=e['orientation_eligible']
    mse=np.sum(np.square(rgb-truth),axis=1)
    truth2=np.sum(np.square(truth),axis=1)
    values={k:distribution(e[k],weights) for k in ('Q_relative','major_width','minor_width',
        'major_width_relative','minor_width_relative','color_max','metallic','gap','gap_reference')}
    values['orientation_degrees']=distribution(e['orientation_degrees'][angular],weights[angular])
    return dict(observations=len(weights),weight_sum=float(weights.sum()),orientation_eligible=int(angular.sum()),
        orientation_truth_eligible=int(e['orientation_truth_eligible'].sum()),
        orientation_prediction_eligible=int(e['orientation_prediction_eligible'].sum()),
        Q_at_most_005=int((e['Q_relative']<=.05).sum()),
        RGB_relative_L2=float(np.sqrt(np.dot(mse,weights)/max(np.dot(truth2,weights),1e-30))),
        RGB_RMSE=float(np.sqrt(np.dot(mse,weights)/(3*weights.sum()))),
        width_at_floor=int((e['minor_predict']<=.0401).sum()),
        width_at_ceiling=int((e['major_predict']>=.9999).sum()),**values)


def fitting_support(views, camera, *, fitting_cameras=None):
    """Count other fitting cameras with registered nearby points and highlights."""
    from scipy.spatial import cKDTree

    current=views[camera]
    mask=current['shader_valid']
    q,n=(current[k][mask] for k in ('position','normals'))
    support=np.zeros(len(q),dtype=np.uint8)
    highlights=np.zeros_like(support)
    for c in ref.SPLITS['fitting'] if fitting_cameras is None else fitting_cameras:
        if c==camera:
            continue
        v=views[c]
        valid=v['shader_valid']
        lookup=cKDTree(v['position'][valid]).query_ball_point(q,h.COVERAGE_RADIUS)
        normal=v['normals'][valid]
        bright=v['highlight_mask'][valid]
        for idx,neighbors in enumerate(lookup):
            if neighbors:
                candidate=np.asarray(neighbors)
                candidate=candidate[(normal[candidate]@n[idx])>=h.NORMAL_COSINE]
                support[idx]+=bool(candidate.size)
                highlights[idx]+=bool(bright[candidate].any())
    return support,highlights


def mesh_adjacency(faces):
    edges=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    owner=np.tile(np.arange(len(faces)),3)
    order=np.lexsort((edges[:,1],edges[:,0]))
    edges,owner=edges[order],owner[order]
    starts=np.r_[0,np.flatnonzero(np.any(edges[1:]!=edges[:-1],axis=1))+1,len(edges)]
    rows=starts[:-1][np.diff(starts)==2]
    return set(map(tuple,np.sort(np.stack((owner[rows],owner[rows+1]),axis=1),axis=1)))


def boundary_pairs(view, adjacency):
    """Opposite-region right/down image neighbors on the same or adjacent face."""
    valid=view['shader_valid']
    pixel=view['pixel_indices'][valid]
    lookup=np.full(ref.SIZE**2,-1,dtype=np.int64)
    lookup[pixel]=np.arange(len(pixel))
    x=pixel[(pixel%ref.SIZE)!=ref.SIZE-1]
    y=pixel[pixel<ref.SIZE*(ref.SIZE-1)]
    a,b=lookup[np.r_[x,y]],lookup[np.r_[x+1,y+ref.SIZE]]
    mask=b>=0
    a,b=a[mask],b[mask]
    p,n,region,face=(view[k][valid] for k in ('position','normals','material_region','triangle'))
    mask=((region[a]!=region[b]) & (np.linalg.norm(p[a]-p[b],axis=1)<=h.BOUNDARY_PAIR_RADIUS)
          & ((n[a]*n[b]).sum(-1)>=h.NORMAL_COSINE))
    a,b=a[mask],b[mask]
    mask=np.asarray([x==y or tuple(sorted((x,y))) in adjacency for x,y in zip(face[a],face[b])],dtype=bool)
    a,b=a[mask],b[mask]
    swap=region[a]!=0
    return np.stack((np.where(swap,b,a),np.where(swap,a,b)),axis=1)


def boundary_score(view, pairs, predicted_Q, predicted_widths):
    if not len(pairs):
        return None
    valid=view['shader_valid']
    n=view['normals'][valid]
    a,b=pairs.T
    source,target=n[b],n[a]
    v=np.cross(source,target)
    c=(source*target).sum(1)
    require((c>=h.NORMAL_COSINE).all(), 'Boundary transports require compatible normals')
    x,y,z=v.T
    o=np.zeros(len(x))
    k=np.stack((o,-z,y,z,o,-x,-y,x,o),axis=1).reshape(-1,3,3)
    R=np.eye(3)+k+(k@k)/(1+c[:,None,None])

    def jump(Q):
        return Q[a]-R@Q[b]@R.transpose(0,2,1)

    q_gt,q_pred=jump(view['Q_world'][valid]),jump(predicted_Q)
    magnitude=np.linalg.norm(q_gt,axis=(1,2))
    require((magnitude>0).all(), 'Degenerate prescribed roughness boundary')
    widths=view['widths_major_minor'][valid]
    w_gt,w_pred=widths[a]-widths[b],predicted_widths[a]-predicted_widths[b]
    ratios=np.linalg.norm(q_pred,axis=(1,2))/magnitude
    return dict(pairs=len(a),Q_jump_magnitude_ratio_median=float(np.median(ratios)),
                Q_jump_projection_median=float(np.median((q_pred*q_gt).sum((1,2))/magnitude**2)),
                Q_jump_relative_error_mean=float(np.mean(np.linalg.norm(q_pred-q_gt,axis=(1,2))/magnitude)),
                reference_signed_width_jump=w_gt.mean(0).tolist(),predicted_signed_width_jump=w_pred.mean(0).tolist(),
                signed_width_jump_MAE=abs(w_pred-w_gt).mean(0).tolist())
