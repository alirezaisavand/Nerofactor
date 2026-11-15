"""New E11 CPU/synthetic tests. No real geometry rendering, native model or GPU."""
import ast
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from audits.bell_calibration_e11 import decode_camera,calibrations,rays
from benchmarks.bell_e11 import (ENV,RELIGHT,LIMITS,TOL,PROTOCOL,IMAGE,split,linear_to_srgb,
    convergence,geometry_gate,verify_authorization,verify_labels)
from benchmarks.bell_geometry_e11 import intersect,mesh_arrays
from benchmarks.bell_reference_e11 import field,kernel,samples,integrate
from benchmarks.generate_bell_e11 import frozen,primary,labels,write_json,write_arrays,run_with_cleanup
from infrastructure.stage_sources import ROOT,PINS,baseline,digest


def camera_bytes():
    P=np.concatenate([np.eye(3),np.array([[0.],[0.],[-3.]])],1)
    K=np.array([[800.,0.,400.],[0.,800.,400.],[0.,0.,1.]],dtype=np.float32)
    return pickle.dumps([P,K],protocol=4)


def test_data_only_camera_dialect_and_original_bytes(tmp_path):
    p=tmp_path/'camera.pkl';data=camera_bytes();p.write_bytes(data)
    a,b=decode_camera(data);P,K=calibrations([p]);assert P.dtype==K.dtype==np.float32
    assert np.array_equal(K[0][:2],b[:2]*.16) and np.array_equal(P[0],a.astype(np.float32))
    assert p.read_bytes()==data


@pytest.mark.parametrize('data',[b"\x80\x04cos\nsystem\n.",b'bad pickle',b'\x80\x04N.',camera_bytes()+b'extra'])
def test_unsafe_or_incomplete_camera_rejected(data):
    with pytest.raises((ValueError,IndexError)):decode_camera(data)


@pytest.mark.parametrize('kind',['rotation','K','shape','nonfinite','object'])
def test_bad_numeric_calibration_rejected(tmp_path,kind):
    P,K=decode_camera(camera_bytes())
    if kind=='rotation':P[0,0]=2
    if kind=='K':K[2,0]=1
    if kind=='shape':P=P[:,:3]
    if kind=='nonfinite':P[0,0]=np.nan
    if kind=='object':P=np.array(['unsafe'],dtype=object)
    p=tmp_path/'c';p.write_bytes(pickle.dumps([P,K],protocol=4))
    with pytest.raises(ValueError):calibrations([p])


def test_original_A_pinhole_arithmetic_matches_independent_numpy_without_rng():
    P,K=decode_camera(camera_bytes());P=P.astype(np.float32);K=K.astype(np.float32);K[:2]*=.16
    rng=torch.get_rng_state().clone();o,d=primary(torch,P,K);on,dn=rays(P,K,128,128)
    np.testing.assert_allclose(o,on,atol=2e-6,rtol=0);np.testing.assert_allclose(d,dn,atol=2e-6,rtol=0)
    assert np.array_equal(on[0],[0,0,3]) and torch.equal(rng,torch.get_rng_state())


def test_independent_triangle_intersection_preserves_winding():
    v=np.array([[-1,-1,0],[1,-1,0],[0,1,0]],dtype=float);f=np.array([[0,1,2]])
    r=intersect(np.array([0,0,2.]),np.array([0,0,-1.]),v,f)
    assert r['depth']==2 and np.array_equal(r['raw_normal'],[0,0,1])
    r2=intersect(np.array([0,0,2.]),np.array([0,0,-1.]),v,f[:,::-1]);assert np.array_equal(r2['raw_normal'],-r['raw_normal'])
    assert intersect(np.array([3,0,2.]),np.array([0,0,-1.]),v,f) is None


def test_ply_exact_payload_indices_and_counts(tmp_path):
    header=b'ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
    v=np.eye(3,dtype='<f4');p=tmp_path/'m.ply';p.write_bytes(header+v.tobytes()+b'\x03'+np.array([0,1,2],dtype='<i4').tobytes())
    a,b=mesh_arrays(p);assert np.array_equal(a,v) and np.array_equal(b,[[0,1,2]])
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError):mesh_arrays(p)


def material_fixture():
    p=torch.tensor([[.1,-.2,.3]],dtype=torch.float64);n=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
    bounds=torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]],dtype=torch.float64)
    return p,n,bounds


def test_world_tensor_labels_and_physical_chart_reconstruct_independently():
    p,n,bounds=material_fixture();m,a=labels(torch,p,n,bounds)
    rows={k:v.numpy() for k,v in a.items()};verify_labels(p.numpy(),n.numpy(),bounds.numpy(),rows)
    assert np.isfinite(rows['theta_major']).all() and np.linalg.eigvalsh(rows['Q_world'])[0,1]>0
    rows['Q_world'][0,0,0]+=.01
    with pytest.raises(ValueError,match='label'):verify_labels(p.numpy(),n.numpy(),bounds.numpy(),rows)


def test_isotropic_grain_direction_has_no_fabricated_orientation():
    p,n,bounds=material_fixture();n=torch.tensor([[1.,2.,3.]],dtype=torch.float64)/np.sqrt(14)
    _,a=labels(torch,p,n,bounds);assert not bool(a['orientation_defined'][0]) and torch.isnan(a['theta_major'][0])
    verify_labels(p.numpy(),n.numpy(),bounds.numpy(),{k:v.numpy() for k,v in a.items()})


def test_cartesian_BRDF_matches_independent_local_matrix_equation_and_reciprocity():
    p,n,bounds=material_fixture();m=field(p,n,bounds)
    wi=torch.tensor([[[.3,.2,np.sqrt(.87)],[0.,0.,1.]]],dtype=torch.float64)
    wo=torch.tensor([[.1,0,np.sqrt(.99)]],dtype=torch.float64)
    fd,fs=kernel(wi,wo,n,m)
    Q=m['Q_world'].numpy()[0,:2,:2];h=wi.numpy()[0]+wo.numpy()[0];h/=np.linalg.norm(h,axis=-1,keepdims=True)
    D=np.exp(-np.einsum('ni,ij,nj->n',h[:,:2],np.linalg.inv(Q),h[:,:2])/h[:,2]**2)/(np.pi*np.sqrt(np.linalg.det(Q))*h[:,2]**4)
    oh=h@wo.numpy()[0];F0=.04*(1-m['metallic'].numpy()[0])+m['metallic'].numpy()[0]*m['color'].numpy()[0]
    F=F0+(1-F0)*(1-oh[:,None])**5;expected=F*D[:,None]/(4*oh[:,None])
    np.testing.assert_allclose(fs.numpy()[0],expected,atol=1e-14,rtol=1e-12)
    _,reverse=kernel(wo[:,None],wi[:,0],n,m);np.testing.assert_allclose(fs[:,:1].numpy(),reverse.numpy(),rtol=1e-12,atol=1e-14)
    assert bool((fd>=0).all()) and bool(torch.isfinite(fs).all())


def test_reference_proposal_mixture_normalization_and_independent_quadrature():
    p,n,bounds=material_fixture();m=field(p,n,bounds);wo=n.clone()
    with np.load(ROOT/'benchmarks/inputs/bell_reference_tapes_e11_v1.npz',allow_pickle=False) as t:
        wi,pdf,valid=samples(n,wo,torch.from_numpy(t['diffuse']),torch.from_numpy(t['half']))
    assert wi.shape==(1,16384,3) and bool(torch.isfinite(pdf).all()) and bool((pdf>0).all())
    cosine=(wi*n[:,None]).sum(-1).clamp(min=0)
    assert abs(float((cosine/pdf*valid).mean())-np.pi)<.003
    fine,coarse=integrate(p,n,wo,m,wi,pdf,valid,ENV,4096)
    # Independent uniform solid-angle midpoint quadrature, not the MIS sampler/PDF.
    z=(torch.arange(128,dtype=torch.float64)+.5)/128;phi=(torch.arange(256,dtype=torch.float64)+.5)*2*np.pi/256
    zz,pp=torch.meshgrid(z,phi);directions=torch.stack([torch.sqrt(1-zz**2)*torch.cos(pp),torch.sqrt(1-zz**2)*torch.sin(pp),zz],-1).reshape(1,-1,3)
    from benchmarks.bell_reference_e11 import environment
    fd,fs=kernel(directions,wo,n,m);grid=((fd+fs)*environment(directions,ENV)*directions[:,:,2,None]).mean(1)*2*np.pi
    np.testing.assert_allclose(fine[:,2].numpy(),grid.numpy(),atol=.0004,rtol=.002)
    assert convergence(fine[:,2].numpy(),coarse[:,2].numpy())['passed']


def test_split_and_known_light_preserve_A_order_and_all_three_arm_settings():
    import yaml
    inventory=json.loads((ROOT/'docs/evidence/PAPER_BENCHMARK_E10_ASSET_INVENTORY_V1.json').read_text())['scenes'][0]
    assert split()=={k:inventory['proposed_new_benchmark_split'][k] for k in split()}
    configs=[yaml.safe_load((ROOT/('configs/material_corrected/bell_'+v+'_known_v1.yaml')).read_text()) for v in ('original','estimator','core')]
    for c in configs:
        assert c['shader_cfg']['known_environment']==ENV and c['shader_cfg']['indirect_mode']=='zero_bounce_v1'
        assert c['total_step']==2000 and c['lr_cfg']==configs[0]['lr_cfg'] and c['loss']==configs[0]['loss']
        assert c['shader_cfg']['diffuse_sample_num']==512 and c['shader_cfg']['specular_sample_num']==1024
    assert split()['validation']==['58'] and len(split()['train'])==122 and ENV!=RELIGHT


def test_linear_srgb_quantization_and_convergence_fail_closed():
    a=np.array([0,.0031308,.18,1.]);assert np.array_equal(linear_to_srgb(a),[0,10,118,255])
    for a in [np.array([-1.]),np.array([1.01]),np.array([np.nan])]:
        with pytest.raises(ValueError):linear_to_srgb(a)
    fine=np.ones((2,3));assert convergence(fine,fine)['passed'] and not convergence(fine,fine+.02)['passed']
    with pytest.raises(ValueError):convergence(fine*float('nan'),fine)


def test_authorization_binds_every_identity_and_output(tmp_path):
    out=Path('/outputs/one-fresh-run');f=tmp_path/'auth.json'
    a=dict(protocol=PROTOCOL,approved=True,freeze_sha256='a'*64,commit='b'*40,output_directory=str(out),image_id=IMAGE,work_limits=LIMITS)
    f.write_text(json.dumps(a));assert verify_authorization(f,digest(f),'a'*64,'b'*40,out)==a
    for key in ['approved','freeze_sha256','commit','output_directory','image_id','work_limits']:
        b=dict(a);b[key]=None;f.write_text(json.dumps(b))
        with pytest.raises(ValueError):verify_authorization(f,digest(f),'a'*64,'b'*40,out)


def test_freeze_hash_and_source_mismatch(tmp_path):
    f=tmp_path/'freeze';a=dict(protocol=PROTOCOL,work_limits=LIMITS,tolerances=TOL,files={'benchmarks/__init__.py':'0'*64});f.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='SHA-256'):frozen(f,'0'*64)
    with pytest.raises(ValueError,match='Incomplete source'):frozen(f,digest(f))


def test_new_outputs_refuse_collision_and_object_archives(tmp_path):
    p=tmp_path/'result.json';write_json(p,{'status':'fixture'})
    with pytest.raises(FileExistsError):write_json(p,{})
    with pytest.raises(ValueError):write_arrays(tmp_path/'bad.npz',x=np.array([{}],dtype=object))


@pytest.mark.parametrize('failure',[False,True])
def test_success_only_after_normal_resource_cleanup(failure):
    events=[]
    def sync():
        events.append('sync')
        if failure:raise RuntimeError('synthetic cleanup failed')
    t=SimpleNamespace(get_num_threads=lambda:8,set_num_threads=lambda n:events.append(('threads',n)),
                      cuda=SimpleNamespace(synchronize=sync,empty_cache=lambda:events.append('cache')))
    def work():events.append('work');return {'scientific_work':True}
    if failure:
        with pytest.raises(RuntimeError,match='cleanup'):run_with_cleanup(work,t)
        assert events==['work','sync']
    else:
        assert run_with_cleanup(work,t)=={'scientific_work':True}
        assert events==['work','sync','cache',('threads',8)]


def test_import_and_no_execute_path_never_import_native_science():
    script='''
import builtins
real=builtins.__import__
def guard(name,*a,**kw):
 assert name.split('.')[0] not in ('torch','network','train','dataset','_raytracing','scientific_variants'),name
 return real(name,*a,**kw)
builtins.__import__=guard
import benchmarks.generate_bell_e11
print('metadata-only import')
'''
    p=subprocess.run(['python3','-B','-c',script],cwd=ROOT,text=True,capture_output=True);assert p.returncode==0,p.stderr


def test_pinned_sources_unchanged_and_python38_grammar():
    for stage,commit in PINS.items():baseline(stage,commit)
    paths=list((ROOT/'benchmarks').glob('*.py'))+[ROOT/'audits/bell_calibration_e11.py',ROOT/'generate_bell_benchmark_e11.py']
    for path in paths:ast.parse(path.read_text(),feature_version=8)


def test_geometry_gate_detects_empty_hits_reversed_normals_and_calibration_mismatch():
    P=np.concatenate([np.diag([1.,-1.,-1.]),np.array([[0.],[0.],[3.]])],1)
    K=np.array([[128.,0,64.],[0,128.,64.],[0,0,1.]])
    o,d=rays(P,K,128,128);depth=-3/d[:,2];p=o+depth[:,None]*d
    n=np.broadcast_to([0.,0.,1.],o.shape);raw=-n;hit=np.ones(len(d),dtype=bool)
    good=geometry_gate(o,d,p,raw,n,depth,hit,hit,P,K);assert good['passed'] and good['silhouette_iou']==1
    assert not geometry_gate(o,d,p,raw,-n,depth,hit,hit,P,K)['passed']
    altered=K.copy();altered[0,0]+=1;assert not geometry_gate(o,d,p,raw,n,depth,hit,hit,P,altered)['passed']
    with pytest.raises(ValueError,match='Empty'):geometry_gate(o,d,p,raw,n,depth,~hit,hit,P,K)


def test_invalid_npz_never_allows_pickle_and_unsafe_input_paths(tmp_path):
    from benchmarks.bell_e11 import record
    target=tmp_path/'one';target.write_bytes(b'fixture');alias=tmp_path/'alias';alias.symlink_to(target)
    with pytest.raises(ValueError):record(alias)
    bad=tmp_path/'bad.npz';np.savez(bad,x=np.array([{}],dtype=object))
    with np.load(bad,allow_pickle=False) as saved:
        with pytest.raises(ValueError):saved['x']


def test_complete_source_input_freeze_required_and_no_implicit_runtime(tmp_path):
    from benchmarks.generate_bell_e11 import runtime
    a=SimpleNamespace(image_id=IMAGE,mesh=tmp_path/'m',cameras=tmp_path,authorization=tmp_path/'a',freeze=tmp_path/'f',
                      commit='0'*40,output=tmp_path/'output')
    with pytest.raises(ValueError):runtime(a)
    assert not a.output.exists()


def test_full_source_freeze_rejects_source_change(tmp_path):
    from benchmarks.bell_e11 import source_paths
    f=tmp_path/'freeze';a=dict(protocol=PROTOCOL,work_limits=LIMITS,tolerances=TOL,files={n:digest(ROOT/n) for n in source_paths()})
    f.write_text(json.dumps(a));assert frozen(f,digest(f))==a
    a['files']['benchmarks/bell_reference_e11.py']='0'*64;f.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='Changed generation'):frozen(f,digest(f))


def test_directional_energy_measurement_has_no_visibility_or_light_dependency():
    p,n,bounds=material_fixture();m=field(p,n,bounds)
    with np.load(ROOT/'benchmarks/inputs/bell_reference_tapes_e11_v1.npz',allow_pickle=False) as t:
        wi,pdf,valid=samples(n,n,torch.from_numpy(t['diffuse'][:2048]),torch.from_numpy(t['half'][:2048]))
    f,c,e,ec=integrate(p,n,n,m,wi,pdf,valid,ENV,1024,return_energy=True)
    _,_,dark,_=integrate(p,n,n,m,wi,pdf,torch.zeros_like(valid),RELIGHT,1024,return_energy=True)
    assert torch.equal(e,dark) and bool(torch.isfinite(e).all()) and bool((e>=0).all())


def test_foreground_launcher_grammar_and_owned_container_identity():
    p=ROOT/'benchmarks/run_bell_e11.sh'
    result=subprocess.run(['bash','-n',str(p)],text=True,capture_output=True);assert result.returncode==0,result.stderr
    code=p.read_text()
    assert '--cidfile "$E11_BATCH/container.cid"' in code
    assert 'test -s "$E11_BATCH/container.cid" || return 0' in code
    assert 'docker rm --force "$E11_OWNED_CID"' in code
    assert 'timeout --foreground --signal=TERM --kill-after=30s 7200s' in code
    assert '--network=none --read-only --cap-drop=ALL' in code and '--pull=never' in code
