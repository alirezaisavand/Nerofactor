"""Prospective fitting-calibration-only path, fixed prompt cameras 0 and 127."""
import numpy as np
from infrastructure.signatures import require


def camera_order(ids,poses):
    ids=list(map(int,ids));poses=np.asarray(poses,dtype=float)
    require(len(ids)==len(set(ids))==len(poses) and {0,127}<=set(ids) and poses.shape==(len(ids),3,4)
            and np.isfinite(poses).all(),'Unique fitting calibration rows and both anchors required')
    centers=-np.einsum('nji,nj->ni',poses[:,:,:3],poses[:,:,3])
    length=np.linalg.norm(centers,axis=1);require((length>0).all(),'Origin-centered calibration required')
    u=centers/length[:,None];distance=np.arccos(np.clip(u@u.T,-1,1));lookup={c:i for i,c in enumerate(ids)}
    start,end=lookup[0],lookup[127];order=[start];todo=set(range(len(ids)))-{start,end}
    while todo:
        j=min(todo,key=lambda i:(float(distance[order[-1],i]),ids[i]));order.append(j);todo.remove(j)
    order.append(end)
    # Deterministic first-improvement 2-opt, retaining both anchor endpoints.
    swaps=0;finished=False
    for _ in range(1000):
        changed=False
        for i in range(1,len(order)-2):
            for j in range(i+1,len(order)-1):
                a,b,c,d=order[i-1],order[i],order[j],order[j+1]
                if distance[a,c]+distance[b,d]<distance[a,b]+distance[c,d]-1e-12:
                    order[i:j+1]=reversed(order[i:j+1]);swaps+=1;changed=True;break
            if changed:break
        if not changed:finished=True;break
    require(finished,'Geometry-only ordering iteration cap exhausted')
    def summary(seq):
        angles=np.degrees([distance[a,b] for a,b in zip(seq[:-1],seq[1:])])
        return dict(count=len(angles),median=float(np.median(angles)),p95=float(np.quantile(angles,.95)),
                    maximum=float(max(angles)),sum=float(sum(angles)),adjacent_degrees=list(map(float,angles)))
    numeric=sorted(range(len(ids)),key=lambda i:ids[i])
    return dict(frame_to_camera=[str(ids[i]) for i in order],numerical_order=summary(numeric),
                angular_order=summary(order),two_opt_swaps=swaps,
                objective='sum of origin-centered camera-position great-circle distances',
                algorithm='fixed 0/127 endpoints; nearest angular neighbor, camera-ID tie break; first-improvement 2-opt',
                geometry_or_masks_used=False,RGB_or_model_outcomes_used=False,
                caveat='Angular proximity is not a guarantee of feature overlap or reliable propagation.')
