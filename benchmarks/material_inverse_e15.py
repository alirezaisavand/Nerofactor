"""Small physical inverse problem: unchanged additive Kurt BRDF, explicit LM.

No neural shader or geometry queries. Torch is dependency-injected. The four
half-vector slopes retain literal cross(n, axis), including finite frame errors.
"""
import numpy as np
from infrastructure.signatures import require
from benchmarks.material_information_e14 import linear_bin_edges,deterministic_codes

STARTS=((-np.pi/3,np.log(.22),np.log(1.5)),(0.,np.log(.30),np.log(2.)),
        (np.pi/3,np.log(.40),np.log(3.)))
POLICY=dict(outer_iterations=20,trials_per_outer=3,initial_damping=.001,
    minimum_damping=1e-10,maximum_damping=1e10,diagonal_floor=1e-12,
    scaled_radius=.35,acceptance_ratio=.1,gradient_infinity_stop=1e-10,
    accepted_step_stop=1e-8)


def correspondence_split(valid):
    require(valid.shape==(4,84) and valid.dtype==np.bool_ and valid[0].all(),'Fixed valid correspondences required')
    held=np.zeros_like(valid)
    for point in range(84):
        ids=np.flatnonzero(valid[:,point])
        if len(ids)>1:held[ids[-1],point]=True
    return valid&~held,held


def physical(p):
    p=np.asarray(p,dtype=np.float64)
    return np.stack((np.exp(p[...,1]+p[...,2]/2),np.exp(p[...,1]-p[...,2]/2)),-1)


def valid_parameters(p):
    p=np.asarray(p,dtype=np.float64);w=physical(p)
    return np.isfinite(p).all(-1)&np.isfinite(w).all(-1)&(p[...,2]>=0)&(w>=.04).all(-1)&(w<=1).all(-1)


def initial_parameters(points,fixed_ratio=None):
    p=np.tile(np.array(STARTS,dtype=np.float64),(points,1,1))
    if fixed_ratio is not None:
        require(np.asarray(fixed_ratio).shape==(points,),'Explicit oracle ratio shape required')
        p[...,2]=np.asarray(fixed_ratio)[:,None]
    require(valid_parameters(p).all(),'Predeclared starts outside material domain; no repair')
    return p


def residuals(prediction,target,mode):
    """Residual to linear truth or the actual noiseless sRGB code intervals."""
    require(mode in ('linear','codes8'),'Unknown observation model')
    if mode=='linear':return prediction-target,np.ones_like(prediction)
    require(np.asarray(target).dtype==np.uint8,'Actual integer codes required')
    e=linear_bin_edges();lo=e[target.astype(int)];hi=e[target.astype(int)+1]
    below=prediction<lo;above=prediction>hi
    r=np.where(below,prediction-lo,np.where(above,prediction-hi,0.))
    return r,(below|above).astype(float)


def fit_rows(y,j,target,mode,points,lights):
    """Observation x start x light x RGB -> independent point/start systems."""
    use=y[:,:,:lights];t=target[:,:lights][:,None]
    r,active=residuals(use,t,mode);jj=j[:,:,:lights]*active[...,None]
    P=int(points.max())+1;S=y.shape[1]
    h=np.zeros((P,S,3,3));g=np.zeros((P,S,3));loss=np.zeros((P,S));counts=np.zeros(P)
    np.add.at(h,points,np.einsum('oslcj,oslck->osjk',jj,jj))
    np.add.at(g,points,np.einsum('oslcj,oslc->osj',jj,r))
    np.add.at(loss,points,(r*r).sum((2,3)));np.add.at(counts,points,lights*3)
    require((counts>0).all(),'Every original point needs fitting observations')
    return loss/counts[:,None],h/counts[:,None,None,None],g/counts[:,None,None]


def fit(evaluate,target,points,initial,mode,lights,fixed_ratio=False,record=lambda row:None):
    """All acceptance/stopping uses fitting observations; no truth/held-out input.

    evaluator(parameters, jacobian) returns O x S x 2 x RGB, and its exact
    analytic derivative when requested. Every point/start has its own LM state.
    """
    p=initial.copy();P,S,_=p.shape;d=2 if fixed_ratio else 3
    damping=np.full((P,S),POLICY['initial_damping']);active=np.ones((P,S),bool)
    termination=np.full((P,S),'outer_limit',dtype='<U32');accepted=np.zeros((P,S),np.int64)
    trials=np.zeros_like(accepted);history=[]
    for outer in range(POLICY['outer_iterations']):
        if not active.any():break
        y,j=evaluate(p,True);old,h,g=fit_rows(y,j,target,mode,points,lights)
        grad=np.max(abs(2*g[...,:d]),axis=-1)
        stationary=active&(grad<=POLICY['gradient_infinity_stop'])
        termination[stationary]='fitting_stationary';active[stationary]=False
        pending=active.copy();row=dict(outer=outer,parameters=p.tolist(),loss_before=old.tolist(),
            gradient_infinity=grad.tolist(),normal_eigenvalues=np.linalg.eigvalsh(h[...,:d,:d]).tolist(),
            accepted_before=accepted.tolist(),trials=[])
        for trial in range(POLICY['trials_per_outer']):
            if not pending.any():break
            delta=np.zeros_like(p);predicted=np.zeros((P,S))
            for i,s in zip(*np.where(pending)):
                H=h[i,s,:d,:d];G=g[i,s,:d];diagonal=np.maximum(np.diag(H),POLICY['diagonal_floor'])
                step=np.linalg.solve(H+damping[i,s]*np.diag(diagonal),-G)
                step*=min(1.,POLICY['scaled_radius']/max(float(np.linalg.norm(step)),1e-300))
                delta[i,s,:d]=step;predicted[i,s]=-2*G@step-step@H@step
            candidate=p+delta;candidate[...,0]=(candidate[...,0]+np.pi/2)%np.pi-np.pi/2
            eligible=pending&valid_parameters(candidate)&(predicted>0)
            proposal=np.where(eligible[...,None],candidate,p)
            if eligible.any():
                yy,_=evaluate(proposal,False)
                # Only residuals matter here; zero derivative avoids additional scientific work.
                new,_,_=fit_rows(yy,np.zeros(yy.shape+(3,)),target,mode,points,lights)
            else:new=old.copy()
            rho=np.divide(old-new,predicted,out=np.full_like(old,-np.inf),where=predicted>0)
            take=eligible&(new<old)&(rho>POLICY['acceptance_ratio'])
            trials+=pending;accepted+=take;p=np.where(take[...,None],candidate,p)
            tiny=take&(np.linalg.norm(delta,axis=-1)<=POLICY['accepted_step_stop'])
            termination[tiny]='accepted_step_stationary';active[tiny]=False
            damping=np.where(pending&~take,damping*10,damping)
            damping=np.where(take&(rho>.75),damping/3,np.where(take&(rho<.25),damping*2,damping))
            damping=np.clip(damping,POLICY['minimum_damping'],POLICY['maximum_damping'])
            row['trials'].append(dict(index=trial,eligible=eligible.tolist(),accepted=take.tolist(),
                loss_proposed=new.tolist(),damping=damping.tolist(),step_norm=np.linalg.norm(delta,axis=-1).tolist()))
            pending&=~take
        row.update(parameters_after=p.tolist(),accepted_after=accepted.tolist())
        record(row);history.append(row)
    require(valid_parameters(p).all() and (not fixed_ratio or np.array_equal(p[...,2],initial[...,2])),
            'Terminal physical domain/fixed ratio violated')
    return dict(parameters=p,accepted=accepted,trials=trials,termination=termination,history=history,damping=damping)


class Prepared:
    """Exact finite-grid BRDF factorization, streamed in16384-node chunks."""
    def __init__(self,torch,slopes,weights,diffuse,point_ids,charge,phase):
        self.torch=torch;self.slopes=slopes;self.weights=weights;self.diffuse=diffuse
        self.point_ids=np.asarray(point_ids,np.int64);self.charge=charge;self.phase=phase

    def evaluate(self,parameters,jacobian,observations=None):
        t=self.torch;obs=np.arange(len(self.point_ids)) if observations is None else np.asarray(observations,np.int64)
        ids=t.as_tensor(obs,device=self.slopes.device);p=t.as_tensor(parameters[self.point_ids[obs]],device=self.slopes.device)
        O,S,_=p.shape;N=self.slopes.shape[1]
        self.charge(self.phase,O*S*N,bool(jacobian))
        theta,u,v=p.unbind(-1);c=t.cos(theta)[...,None];s=t.sin(theta)[...,None]
        a=t.exp(-2*u-v)[...,None];b=t.exp(-2*u+v)[...,None]
        result=t.zeros((O,S,6),dtype=t.float64,device=p.device)
        derivative=t.zeros((O,S,6,3),dtype=t.float64,device=p.device) if jacobian else None
        for start in range(0,N,16384):
            stop=min(N,start+16384);x,y,bx,by=self.slopes[ids,start:stop].unbind(-1)
            x=x[:,None];y=y[:,None];bx=bx[:,None];by=by[:,None]
            h1=c*x+s*y;h2=c*bx+s*by
            A=a*h1*h1;B=b*h2*h2;D=t.exp(-A-B-2*u[...,None])
            w=self.weights[ids,start:stop]
            result+=t.bmm(D,w)
            if jacobian:
                factors=(-2*(a*h1*(-s*x+c*y)+b*h2*(-s*bx+c*by)),2*(A+B)-2,A-B)
                for k,factor in enumerate(factors):derivative[...,k]+=t.bmm(D*factor,w)
        result=result.reshape(O,S,2,3)+self.diffuse[ids,None]
        require(bool(t.isfinite(result).all()) and bool((result>=0).all()),'Invalid factored radiance')
        y=result.cpu().numpy()
        j=derivative.reshape(O,S,2,3,3).cpu().numpy() if jacobian else None
        if j is not None:require(np.isfinite(j).all(),'Nonfinite analytic derivatives')
        return y,j


def prepare(torch,labels,cache,outgoing,lights,charge,device='cuda'):
    """Material-independent half-vector/Fresnel coefficients; no geometry rays.

    Known color/metal determine Fresnel and diffuse allocation. No unknown
    theta or widths enter these coefficients. No BRDF surrogate/interpolation.
    """
    from benchmarks.bell_reference_e11 import basis,environment
    valid=outgoing['valid'];views,points=np.nonzero(valid);O=len(points);N=len(cache['nodes'])
    slopes=torch.empty((O,N,4),dtype=torch.float64,device=device)
    weights=torch.empty((O,N,6),dtype=torch.float64,device=device)
    diffuse=torch.zeros((O,2,3),dtype=torch.float64,device=device)
    lab={k:torch.as_tensor(labels[k],device=device) for k in ('normals','T','B','color','metallic')}
    for first in range(0,len(labels['normals']),4):
        last=min(first+4,len(labels['normals']));n=lab['normals'][first:last];t,b=basis(n)
        obs=np.flatnonzero((points>=first)&(points<last));local=points[obs]-first
        idx=torch.as_tensor(obs,device=device);ix=torch.as_tensor(local,device=device)
        jpoints=torch.as_tensor(points[obs],device=device)
        normal=n[ix];T=lab['T'][jpoints];B=lab['B'][jpoints]
        BT=torch.cross(normal,T,dim=-1);BB=torch.cross(normal,B,dim=-1)
        wo=torch.as_tensor(outgoing['outgoing_directions'][views[obs],points[obs]],device=device)
        color=lab['color'][jpoints];metal=lab['metallic'][jpoints]
        f0=.04*(1-metal)+metal*color;fd=color*(1-metal)/np.pi
        for start in range(0,N,16384):
            stop=min(start+16384,N);nodes=torch.as_tensor(cache['nodes'][start:stop],device=device)
            wi=nodes[None,:,0,None]*t[:,None]+nodes[None,:,1,None]*b[:,None]+nodes[None,:,2,None]*n[:,None]
            charge('environment',2*len(n)*(stop-start),False)
            light=torch.stack([environment(wi,light) for light in lights],-2)
            packed=cache['visibility'][first:last,start//8:stop//8]
            visible=torch.as_tensor(np.unpackbits(packed,axis=1,bitorder='big').astype(bool),device=device)[ix]
            w=wi[ix];h=torch.nn.functional.normalize(w+wo[:,None],dim=-1)
            ci=(w*normal[:,None]).sum(-1);ch=(h*normal[:,None]).sum(-1);oh=(h*wo[:,None]).sum(-1)
            require(bool((ci>0).all()) and bool((ch>0).all()) and bool((oh>0).all()) and
                    bool(((normal*wo).sum(-1)>0).all()),'Invalid admitted hemisphere/half-vector')
            charge('coefficient',len(obs)*(stop-start),False)
            sl=torch.stack([(h*axis[:,None]).sum(-1)/ch for axis in (T,B,BT,BB)],-1)
            F=f0[:,None]+(1-f0[:,None])*(1-oh.clamp(0,1))[...,None]**5
            incident=light[ix]*((ci*visible*cache['weight'])[:,:,None,None])
            W=incident*F[:,:,None,:]/(4*np.pi*oh*ch**4)[:,:,None,None]
            require(bool(torch.isfinite(sl).all()) and bool(torch.isfinite(W).all()) and bool((W>=0).all()),
                    'Invalid exact integration coefficients')
            slopes[idx,start:stop]=sl;weights[idx,start:stop]=W.reshape(len(obs),stop-start,6)
            diffuse[idx]+=(incident*fd[:,None,None]).sum(1)
    return Prepared(torch,slopes,weights,diffuse,points,charge,'fit')


def metrics(parameters,truth,frame):
    w=physical(parameters);tw=physical(truth);th=parameters[...,0];tt=truth[...,0]
    def Q(theta,width):
        c=np.cos(theta);s=np.sin(theta);axis=np.stack((c,s),-1);other=np.stack((-s,c),-1)
        return width[...,0,None,None]**2*axis[...,None]*axis[...,None,:]+width[...,1,None,None]**2*other[...,None]*other[...,None,:]
    q=Q(th,w);tq=Q(tt,tw);basis=np.stack(frame,-1)
    world=basis[:,None]@q@basis[:,None].swapaxes(-1,-2)
    target=basis[:,None]@tq@basis[:,None].swapaxes(-1,-2)
    return dict(Q_local_relative=np.linalg.norm(q-tq,axis=(-2,-1))/np.maximum(np.linalg.norm(tq,axis=(-2,-1)),1e-15),
        Q_world_relative=np.linalg.norm(world-target,axis=(-2,-1))/np.maximum(np.linalg.norm(target,axis=(-2,-1)),1e-15),
        orientation_degrees=np.rad2deg(np.arccos(np.clip(abs(np.cos(th-tt)),0,1))),
        logmean_absolute=abs(parameters[...,1]-truth[...,1]),logratio_absolute=abs(parameters[...,2]-truth[...,2]),
        major_absolute=abs(w[...,0]-tw[...,0]),minor_absolute=abs(w[...,1]-tw[...,1]))
