"""CPU fixtures for split isolation, schedule identity, geometry gates and cleanup evidence."""
import ast
from contextlib import contextmanager
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bell_geometry_gate10_v1 as g
from experiments.bell_geometry_gate10_train_v1 import Acquisition
from experiments.bell_geometry_gate10_evaluate_v1 import sdf_consistency
from infrastructure.signatures import SignatureError


def method_class(name):
    """Run only numerical coefficient methods from the immutable source AST."""
    tree=ast.parse((g.ROOT/'scientific_sources/geometry_g/network/loss.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    cls.body=[n for n in cls.body if isinstance(n,ast.Assign) or
              isinstance(n,ast.FunctionDef) and n.name not in ('__call__',)]
    scope={'np':np,'Loss':object}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<coefficient-only-fixture>','exec'),scope)
    return scope[name]({})


def test_actual_loss_and_lr_schedules():
    rgb=method_class('NeRFRenderLoss')
    assert [rgb.cosine_weight_schedule(s) for s in (0,5000,10000,30000,299999)]==[1,5,1,1,1]
    occ=method_class('OccLoss')
    opacity=method_class('OpacityLoss')
    mask=method_class('MaskLoss')
    curv=method_class('CurvLoss')
    assert np.allclose([occ.get_occlusion_weight(s) for s in (0,20000,35000,50000)], [.01,.01,.505,1])
    assert np.allclose([opacity.get_opacity_weight(s) for s in (0,20000,35000,50000)],[.001,.001,.0055,.01])
    assert np.allclose([mask.get_mask_weight(s) for s in (0,15000,30000,299999)],[.3,.65,1,1])
    assert all(curv.get_curvature_weight(s)==.001 for s in (0,20000,35000,50000,299999))
    assert method_class('EikonalLoss').get_eikonal_weight(0)==.1
    assert method_class('StdRecorder').cfg['apply_std_loss'] is False
    tree=ast.parse((g.ROOT/'scientific_sources/geometry_g/train/lr_common_manager.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WarmUpCosLR')
    class Parent:
        def set_lr_for_all(self, opt, lr):
            opt['lr']=lr
    scope={'np':np,'LearningRateManager':Parent}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<lr-fixture>','exec'),scope)
    lr=scope['WarmUpCosLR']({})
    assert lr({},0)==0 and np.isclose(lr({},1),1e-7,rtol=1e-12,atol=0) and lr({},5000)==.0005
    assert np.isclose(lr({},300000),.000025)


def test_split_guard_actual_adapter_dispatch():
    class Base:
        def __init__(self,name):
            self.name=name
    class Original(Base):
        def get_image(self,c):return ('image',c)
        def get_mask(self,c):return ('mask',c)
        def get_depth(self,c):return ('depth',c)
    database=SimpleNamespace(GlossySyntheticDatabase=Original,BaseDatabase=Base,random=random.Random(23))
    def build(db,ids,nerf):
        return [db.get_image(c) for c in ids]
    renderer=SimpleNamespace(build_imgs_info=build)
    dummy=SimpleNamespace()
    a=Acquisition(Path('/scoped'),{c:(np.eye(3,4),np.eye(3)) for c in ('3','58')},
                  dict(train=['3'],validation=['58'],held_view_nvs=['105']))
    a.bind(database,renderer,dummy)
    db=renderer.parse_database_name('syn/bell','/scoped')
    assert renderer.get_database_split(db)==(['3'],['58'])
    assert renderer.build_imgs_info(db,['3'])==[('image','3')]
    assert a.scope is None
    with pytest.raises(SignatureError):db.get_image('3')
    with a.visiting('train'):
        with pytest.raises(SignatureError):db.get_image('105')
        with pytest.raises(SignatureError):db.get_depth('3')
    with a.visiting('validation'):
        assert db.get_depth('58')==('depth','58')
    assert dummy.parse_database_name is renderer.parse_database_name
    assert a.data_reads==dict(train=dict(rgb=1),validation=dict(depth=1))


def test_real_admitted_staging_excludes_oracle_and_heldout(tmp_path):
    # Only file copying/calibration arithmetic. No model, checkpoint load or renderer.
    f=g.definition(tmp_path/'unused')
    dst=tmp_path/'train'
    g.stage_training(f,dst)
    assert len(g.read_staged(f,dst))==123
    assert len(list(dst.rglob('*.png')))==247
    assert not list(dst.rglob('*.pkl'))
    assert not list(dst.rglob('*.ply'))
    assert not any((dst/'bell'/f'{c}.png').exists() for c in f['splits']['held_view_nvs'])
    assert (dst/'bell/58-depth.png').exists()
    (dst/'leaked.npy').write_bytes(b'oracle')
    with pytest.raises(SignatureError):g.read_staged(f,dst)


def surface(hit):
    n=np.tile([0,0,1.],(len(hit),1))
    return dict(hit=np.array(hit,bool),normal=n,depth=np.ones(len(hit)),position=np.zeros((len(hit),3)))


def test_signed_surface_and_full_exclusion_accounting():
    a,b=surface([1,1,0,0]),surface([1,0,1,0])
    row,m=g.paired_surface(a,b,np.tile([0,0,-1.],(4,1)))
    assert row['silhouette_iou']==1/3 and row['common']==1
    assert row['oracle_only']==row['reconstruction_only']==row['shared_misses']==1
    assert row['matched_support']==1 and not row['passed']
    b['normal'][0]*=-1
    row,m=g.paired_surface(a,b,np.tile([0,0,-1.],(4,1)))
    assert row['normal_failures']==row['facing_failures']==1 and not m['valid'].any()
    with pytest.raises(SignatureError):g.paired_surface(surface([0]),surface([0]),np.array([[0,0,-1.]]))


class AnalyticQuery:
    def __init__(self,offset=0):
        self.points=self.gradient_points=0
        self.offset=offset
    def __call__(self,p,gradients=False):
        self.points+=len(p)
        self.gradient_points+=len(p) if gradients else 0
        self.last_inside=np.linalg.norm(p,axis=1)<1
        y=p[:,2]+self.offset
        grad=np.tile([0.,0.,1.],(len(p),1)) if gradients else None
        return y,grad


@pytest.mark.parametrize('offset,flip,expected',[(0,False,True),(.01,False,False),(0,True,False)])
def test_sdf_raw_zero_crossing_and_signed_A_normal(offset,flip,expected):
    p=np.zeros((128,3),np.float32)
    p[:,0]=np.linspace(-.4,.4,128)
    normals=np.tile([0.,0.,-1. if flip else 1.],(128,1))
    r,a=sdf_consistency(p,normals,AnalyticQuery(offset))
    assert r['passed']==expected and r['raw_SDF_evaluations']==384
    assert r['eligible']==128
    assert a['consistent'].all()==(offset==0)


def test_train_receipt_rejects_missing_cleanup_and_wrong_split(tmp_path):
    from experiments.bell_geometry_gate10_v1 import completed_train
    args=SimpleNamespace(commit='a'*40,freeze_sha256='b'*64,authorization_sha256='c'*64)
    f=dict(host_output='/tmp/fixture',splits=g.split())
    g.write(tmp_path/'training.json',dict(status='complete',adam_returns=300000,cleanup_verified=False))
    with pytest.raises(SignatureError):completed_train(tmp_path,f,args)


def test_scientific_import_blocker_in_fresh_process():
    import subprocess
    import sys
    code="""import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','network','dataset','train','raytracing','open3d','_raytracing'):
   raise RuntimeError('Forbidden scientific import '+name)
sys.meta_path.insert(0,Block())
import experiments.bell_geometry_gate10_v1
"""
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=g.ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_offline_sdf_verifier_rejects_altered_scientific_gate(tmp_path):
    from experiments.bell_geometry_gate10_v1 import verify_sdf
    from experiments.donut_reconstructed_geometry_v1 import save
    points=np.zeros((128,3),np.float32)
    normals=np.tile([0.,0.,1.],(128,1))
    r,data=sdf_consistency(points,normals,AnalyticQuery())
    path=tmp_path/'sdf.npz'
    save(path,data)
    verify_sdf(path,r)
    r['passed']=False
    with pytest.raises(SignatureError):verify_sdf(path,r)


def test_full_resolution_mesh_sampling_exceeds_historical_audit_cap_and_has_no_rim():
    from experiments.bell_geometry_gate10_evaluate_v1 import surface_witnesses
    vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]],np.float32)
    faces=np.tile([0,1,2],(100001,1))
    centers,normals,ids,vids=surface_witnesses(vertices,faces)
    assert len(ids)==224 and len(vids)==3
    assert np.allclose(normals[ids],[0,0,1]) and (np.linalg.norm(centers[ids],axis=1)<.1).all()
