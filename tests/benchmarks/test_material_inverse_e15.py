"""Small E15 synthetic forward/derivative/optimizer checks, no real observations."""
import numpy as np
import pytest
import torch

from benchmarks.material_inverse_e15 import (prepare,physical,initial_parameters,residuals,fit,
    metrics,correspondence_split,valid_parameters,POLICY)
from benchmarks.material_information_e14 import deterministic_codes
from experiments.bell_quadrature_e11e import midpoint_grid,weighted_components
from experiments.bell_highlight_pilot_e11d import kernel_axes
from benchmarks.bell_reference_e11 import basis,environment


def fixture():
    n=np.tile([0.,0.,1.],(2,1));T=np.tile([1.,0.,0.],(2,1));B=np.tile([0.,1.+1e-7,0.],(2,1))
    labels=dict(normals=n,T=T,B=B,color=np.full((2,3),.3),metallic=np.full((2,1),.2))
    wo=np.tile([0.,0.,1.],(2,2,1));wo[1]=[.5,0.,np.sqrt(.75)]
    outgoing=dict(valid=np.ones((2,2),bool),outgoing_directions=wo)
    nodes,weight=midpoint_grid(8);cache=dict(nodes=nodes,weight=weight,visibility=np.full((2,8),255,np.uint8))
    lights=[dict(ambient=[.1,.2,.3],lobes=[]),dict(ambient=[.3,.2,.1],lobes=[])]
    return labels,outgoing,cache,lights


def test_factored_forward_matches_literal_reference_cross_frame_and_known_inputs_only():
    labels,outgoing,cache,lights=fixture();work=[]
    model=prepare(torch,labels,cache,outgoing,lights,lambda *x:work.append(x),device='cpu')
    p=initial_parameters(2);y,j=model.evaluate(p,True)
    for obs,(view,point) in enumerate(zip(*np.nonzero(outgoing['valid']))):
        n=torch.tensor(labels['normals'][point:point+1]);t,b=basis(n);nodes=torch.tensor(cache['nodes'])
        wi=nodes[None,:,0,None]*t[:,None]+nodes[None,:,1,None]*b[:,None]+nodes[None,:,2,None]*n[:,None]
        wo=torch.tensor(outgoing['outgoing_directions'][view,point:point+1])
        for start in range(3):
            theta=p[point,start,0];axis=np.cos(theta)*labels['T'][point]+np.sin(theta)*labels['B'][point]
            m=dict(n=n,axis=torch.tensor(axis[None]),widths=torch.tensor(physical(p[point,start])[None]),
                   color=torch.tensor(labels['color'][point:point+1]),metallic=torch.tensor(labels['metallic'][point:point+1]))
            fd,fs=kernel_axes(torch,wi,wo,m)
            for light,definition in enumerate(lights):
                value=weighted_components(torch,fd,fs,environment(wi,definition),(wi*n[:,None]).sum(-1),torch.ones((1,64),dtype=torch.bool),cache['weight']).sum(1)
                np.testing.assert_allclose(y[obs,start,light],value[0].numpy(),atol=2e-14,rtol=2e-13)
    changed=dict(labels,theta_major=np.full(2,100.),widths_major_minor=np.full((2,2),.99))
    other=prepare(torch,changed,cache,outgoing,lights,lambda *x:None,device='cpu')
    np.testing.assert_array_equal(y,other.evaluate(p,False)[0])
    assert j.shape==(4,3,2,3,3) and work[-1]==('fit',4*3*64,True)


def test_analytic_derivatives_agree_with_independent_central_forward_differences():
    labels,outgoing,cache,lights=fixture()
    model=prepare(torch,labels,cache,outgoing,lights,lambda *x:None,device='cpu');p=initial_parameters(2)
    _,j=model.evaluate(p,True)
    for k in range(3):
        a=p.copy();b=p.copy();a[...,k]+=1e-5;b[...,k]-=1e-5
        fd=(model.evaluate(a,False)[0]-model.evaluate(b,False)[0])/2e-5
        np.testing.assert_allclose(j[...,k],fd,rtol=2e-6,atol=2e-10)


def test_derivative_matches_autograd_of_unmodified_independent_BRDF_kernel():
    labels,outgoing,cache,lights=fixture()
    model=prepare(torch,labels,cache,outgoing,lights,lambda *x:None,device='cpu');p=initial_parameters(2)
    _,j=model.evaluate(p,True);point=0;view=1;obs=2;start=1;light=1
    n=torch.tensor(labels['normals'][:1]);t,b=basis(n);nodes=torch.tensor(cache['nodes'])
    wi=nodes[None,:,0,None]*t[:,None]+nodes[None,:,1,None]*b[:,None]+nodes[None,:,2,None]*n[:,None]
    z=torch.tensor(p[point,start],requires_grad=True);theta,u,v=z.unbind()
    axis=torch.cos(theta)*torch.tensor(labels['T'][:1])+torch.sin(theta)*torch.tensor(labels['B'][:1])
    widths=torch.stack((torch.exp(u+v/2),torch.exp(u-v/2)))[None]
    m=dict(n=n,axis=axis,widths=widths,color=torch.tensor(labels['color'][:1]),metallic=torch.tensor(labels['metallic'][:1]))
    fd,fs=kernel_axes(torch,wi,torch.tensor(outgoing['outgoing_directions'][view,:1]),m)
    value=weighted_components(torch,fd,fs,environment(wi,lights[light]),(wi*n[:,None]).sum(-1),torch.ones((1,64),dtype=torch.bool),cache['weight']).sum()
    independent=torch.autograd.grad(value,z)[0].numpy()
    np.testing.assert_allclose(j[obs,start,light].sum(0),independent,atol=2e-14,rtol=2e-12)


def test_actual_codes_are_intervals_with_zero_interior_derivatives_and_infinite_saturation_bins():
    truth=np.array([0.,.2,1.]);codes=deterministic_codes(truth)
    r,j=residuals(truth,codes,'codes8');assert (r==0).all() and (j==0).all()
    r,j=residuals(np.array([-1.,.4,2.]),codes,'codes8')
    assert r[0]==r[-1]==0 and r[1]>0 and j.tolist()==[0.,1.,0.]
    with pytest.raises(ValueError):residuals(truth,codes.astype(float),'codes8')


@pytest.mark.parametrize('fixed',[False,True])
def test_LM_matches_budget_and_restores_identifiable_synthetic_parameters_without_truth_in_optimizer(fixed):
    points=np.array([0,0,1]);truth=np.array([[.15,np.log(.3),np.log(2.)],[-.2,np.log(.35),np.log(1.8)]])
    target=np.repeat((.2+.05*truth[points])[:,None,:],2,axis=1)
    calls=[]
    def evaluate(p,jacobian):
        calls.append(jacobian);y=np.repeat((.2+.05*p[points])[:,:,None,:],2,axis=2)
        j=np.broadcast_to(np.eye(3)*.05,y.shape+(3,)).copy() if jacobian else None
        return y,j
    initial=initial_parameters(2,truth[:,2] if fixed else None)
    result=fit(evaluate,target,points,initial,'linear',2,fixed)
    np.testing.assert_allclose(result['parameters'],np.repeat(truth[:,None,:],3,axis=1),atol=2e-6)
    assert len(calls)<=POLICY['outer_iterations']*(1+POLICY['trials_per_outer'])
    assert (result['accepted']<=20).all() and valid_parameters(result['parameters']).all()
    if fixed:np.testing.assert_array_equal(result['parameters'][...,2],initial[...,2])


def test_geometric_holdout_keeps_single_view_point_and_orientation_periodicity():
    valid=np.zeros((4,84),bool);valid[0]=True;valid[1,1:]=True
    train,held=correspondence_split(valid)
    assert train[0].all() and held[1,1:].all() and not held[:,0].any() and np.array_equal(train|held,valid)
    p=initial_parameters(2)[:,:1];q=p.copy();q[...,0]+=np.pi
    frame=(np.tile([1.,0.,0.],(2,1)),np.tile([0.,1.,0.],(2,1)))
    m=metrics(q,p,frame)
    assert np.max(m['Q_world_relative'])<1e-14 and np.max(m['orientation_degrees'])<2e-6


def test_zero_interval_loss_retains_distinct_physical_starts_instead_of_claiming_recovery():
    from benchmarks.material_information_e14 import linear_bin_edges
    mu=float(np.mean(linear_bin_edges()[100:102]));p=initial_parameters(1);points=np.array([0])
    target=deterministic_codes(np.full((1,2,3),mu))
    def evaluate(p,jacobian):
        y=np.repeat((mu+1e-6*p[points])[:,:,None],2,axis=2)
        j=np.broadcast_to(1e-6*np.eye(3),y.shape+(3,)).copy() if jacobian else None
        return y,j
    r=fit(evaluate,target,points,p,'codes8',2)
    np.testing.assert_array_equal(r['parameters'],p)
    assert (r['accepted']==0).all() and (r['termination']=='fitting_stationary').all()


def test_metadata_import_never_imports_torch_or_native_scientific_modules():
    import subprocess,sys
    code='''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','raytracing','_raytracing','network','train'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import experiments.bell_material_inverse_e15
'''
    subprocess.run([sys.executable,'-B','-c',code],check=True)


def test_all_native_panel_artifacts_use_numeric_archives_and_one_original_ratio_per_start(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    from experiments import bell_material_inverse_e15 as study
    from infrastructure.stage_sources import digest
    monkeypatch.setattr(torch,'__version__','1.8.1+cu111');monkeypatch.setattr(torch.version,'cuda','11.1')
    for name,value in [('device_count',lambda:1),('get_device_name',lambda i:'A100 synthetic'),
                       ('set_per_process_memory_fraction',lambda x:None),('max_memory_allocated',lambda:0),('max_memory_reserved',lambda:0)]:
        monkeypatch.setattr(torch.cuda,name,value)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None);monkeypatch.setattr(study,'baseline',lambda *a:None)
    binary=tmp_path/'synthetic.binary';binary.write_bytes(b'not a native extension')
    monkeypatch.setattr(study,'identity',lambda:dict(binary_path=str(binary),binary_sha256=digest(binary)))
    labels=dict(normals=np.tile([0.,0.,1.],(84,1)),T=np.tile([1.,0.,0.],(84,1)),B=np.tile([0.,1.,0.],(84,1)),
        theta_major=np.zeros(84),widths_major_minor=np.tile([.4,.2],(84,1)))
    valid=np.ones((4,84),bool);train,held=correspondence_split(valid)
    outgoing=dict(valid=valid,camera_ids=np.array([58,44,9,0]))
    targets={k:np.full((2,2,4,84,12,3,3),.1) for k in (256,512)}
    class Fake:
        phase='fit'
        def evaluate(self,p,jacobian,observations=None):
            O=int(valid.sum()) if observations is None else len(observations)
            return np.full((O,3,2,3),.1),np.zeros((O,3,2,3,3)) if jacobian else None
    monkeypatch.setattr(study,'prepare',lambda *a,**kw:Fake())
    monkeypatch.setattr(study,'numerical_gates',lambda p,d,r:r.update(admission=dict(status='pass',synthetic_only=True)))
    def fakefit(evaluate,target,points,initial,mode,lights,fixed,capture):
        evaluate(initial,True);capture(dict(synthetic_only=True))
        return dict(parameters=initial,accepted=np.zeros((84,3),np.int64),trials=np.zeros((84,3),np.int64),
            termination=np.full((84,3),'synthetic',dtype='<U32'),history=[],damping=np.ones((84,3)))
    monkeypatch.setattr(study,'fit',fakefit)
    r=dict(attempted={k:0 for k in study.LIMITS if k.endswith('_point_directions')},panels=[]);r['completed']=dict(r['attempted'])
    for name in ('forward_attempts','forward_returns','analytic_jacobian_attempts','analytic_jacobian_returns'):
        r[name]={k:0 for k in study.CALL_LIMITS}
    args=SimpleNamespace(output=tmp_path/'out');args.output.mkdir()
    data=(labels,np.arange(84),{256:{},512:{}},outgoing,[],targets,train,held)
    study.native(args,data,r,time.monotonic(),torch)
    assert len(r['panels'])==16 and len(list(args.output.glob('*-endpoint.npz')))==16
    for p in args.output.glob('*.npz'):
        with np.load(p,allow_pickle=False) as z:assert all(z[k].dtype.kind in 'biuf' for k in z.files)
