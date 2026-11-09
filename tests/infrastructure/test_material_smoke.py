"""CPU toy shader/operators with unchanged A loop/RGB/regularizer source bodies."""
import ast
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io
import inspect
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import yaml
try:import torch
except (ImportError,OSError):torch=None

from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.stage_execution import plan,trainer_io,execute
from infrastructure.material_smoke import MaterialProbe,inspect_mesh,admit_mesh,CONFIG,MESH_SHA256,GROUPS
from infrastructure.bounded_training import TwoUpdateTrace,run_boundaries
from infrastructure.bell_inference import runtime_environment
from infrastructure.runs import RunStore
from infrastructure.signatures import SignatureError
A=ROOT/'scientific_sources/material_a'


def compile_members(path,cls,names,scope):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls)
    members=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=members,type_ignores=[]),str(path),'exec'),scope)
    return {n.name:scope[n.name] for n in members}


def tiny_mesh(path,vertex_override=None,index=2):
    path.parent.mkdir(parents=True,exist_ok=True)
    vertices=[(0.,0.,0.),(1.,0.,1.),(0.,1.,0.)]
    if vertex_override is not None:vertices[0]=vertex_override
    header=b'ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
    path.write_bytes(header+b''.join(struct.pack('<fff',*v) for v in vertices)+struct.pack('<Biii',3,0,1,index))
    return path


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.data=self.root/'datasets';self.thesis=self.root/'thesis';self.out=self.root/'out'
        (self.data/'GlossyReal/bear').mkdir(parents=True)
        (self.data/'GlossyReal/bear/input.fixture').write_text('immutable fixture')
        self.mesh=tiny_mesh(self.thesis/'meshes-pretrained/real/bear.ply')
    def planning(self):return plan('material',PINS['material'],'C',self.data,self.thesis,self.out,ROOT/CONFIG,ROOT)
    def test_original_bear_configuration_no_overrides_and_cpu_only_preflight(self):
        state=random.getstate();p=self.planning()
        self.assertEqual(p['configuration'],yaml.safe_load((A/'configs/material/real/bear.yaml').read_text()))
        self.assertEqual(p['selected'].mesh,self.mesh);self.assertIsNone(p['selected'].checkpoint)
        self.assertEqual(random.getstate(),state);self.assertFalse(self.out.exists())
        with self.assertRaises(SignatureError):admit_mesh(p) # Synthetic bytes cannot claim historical hash.
        with patch('infrastructure.material_smoke.MESH_SHA256',digest(self.mesh)):
            value=admit_mesh(p)
        self.assertFalse(value['research_handoff_admitted']);self.assertEqual(value['triangles'],1)
    def test_missing_hash_malformed_and_incompatible_meshes_fail(self):
        with self.assertRaises(SignatureError):inspect_mesh(self.mesh,'0'*64)
        with self.assertRaises(SignatureError):inspect_mesh(self.root/'absent',MESH_SHA256)
        for vertex,index in [(None,3),((float('nan'),0.,0.),2)]:
            tiny_mesh(self.mesh,vertex,index)
            with self.assertRaises(SignatureError):inspect_mesh(self.mesh,digest(self.mesh))
        self.mesh.write_bytes(b'ply\nend_header\n')
        with self.assertRaises(SignatureError):inspect_mesh(self.mesh,digest(self.mesh))
        tiny_mesh(self.mesh);data=self.mesh.read_bytes();self.mesh.write_bytes(data[:-1])
        with self.assertRaises(SignatureError):inspect_mesh(self.mesh,digest(self.mesh))
        self.mesh.write_bytes(data);alias=self.root/'alias.ply';alias.symlink_to(self.mesh)
        with self.assertRaises(SignatureError):inspect_mesh(alias,digest(self.mesh))
    def test_default_research_execution_still_requires_G_handoff(self):
        p=self.planning()
        with patch('infrastructure.stage_execution.native_guards'),patch('infrastructure.stage_execution.native_run') as native:
            with self.assertRaisesRegex(SignatureError,'mesh-handoff'):execute(p,'not-a-smoke','image','0'*40)
            native.assert_not_called();self.assertFalse(self.out.exists())
        with self.assertRaises(SignatureError):
            with patch('infrastructure.material_smoke.MESH_SHA256',digest(self.mesh)):
                execute(p,'bad-handoff','image','0'*40,mesh_handoff=self.root/'fake.json',bounded_material=True)
    def test_preflight_import_guard_and_explicit_mesh_selection(self):
        script='''
import builtins,sys
from pathlib import Path
from infrastructure import material_smoke
material_smoke.MESH_SHA256=material_smoke.digest(Path(sys.argv[1]))
sys.argv.pop(1)
original=builtins.__import__
def guarded(name,*a,**kw):
 assert name.split('.')[0] not in {'torch','numpy','network','dataset','train','open3d','nvdiffrast'},name
 return original(name,*a,**kw)
builtins.__import__=guarded
from run_material_training_smoke import main
raise SystemExit(main(sys.argv[1:]))
'''
        args=['--scientific-commit',PINS['material'],'--datasets-root',str(self.data),'--thesis-root',str(self.thesis),
              '--output-root',str(self.out),'--mesh',str(self.mesh),'--mesh-sha256',digest(self.mesh),'--run-id','preflight','--validate-only']
        r=subprocess.run([sys.executable,'-B','-c',script,str(self.mesh),*args],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(self.out.exists())
    def test_bounded_orchestration_reuses_output_failure_and_cleanup_framework(self):
        script='''
import json,os,sys
from pathlib import Path
from infrastructure import stage_execution as stage,material_smoke as smoke
from infrastructure.stage_sources import PINS,ROOT
mesh,data,thesis,out,mode=map(Path,sys.argv[1:])
smoke.MESH_SHA256=smoke.digest(mesh)
p=stage.plan('material',PINS['material'],'C',data,thesis,out,ROOT/smoke.CONFIG,ROOT)
stage.native_guards=lambda *a:None # Explicit CPU filesystem fixture, not native execution.
def fake(p,run,report,persist):
 print('CPU fixture material backend')
 report['optimizer_calls_completed']=2
 report['updates']=[{'CPU_fixture':True},{'CPU_fixture':True}]
 if str(mode)=='failure':raise RuntimeError('CPU fixture native failure')
stage.native_run=fake
before=os.getcwd(),os.environ.get('HOME'),mesh.read_bytes()
try:stage.execute(p,'fixture','CPU-fixture-image','0'*40,bounded_material=True)
except RuntimeError:assert str(mode)=='failure'
else:assert str(mode)=='success'
assert (os.getcwd(),os.environ.get('HOME'),mesh.read_bytes())==before
r=json.loads((out/'stage-run.json').read_text())
assert r['diagnostic_only'] and r['mesh_handoff'] is None and r['mesh_identity']['research_handoff_admitted'] is False
assert r['status']==('failed' if str(mode)=='failure' else 'complete')
assert 'CPU fixture material backend' in (out/'logs/console.log').read_text()
assert not list((out/'checkpoints').iterdir())
from infrastructure.runs import RunStore
try:RunStore(p['resolver'],layout='mounted').create('fixture')
except ValueError:pass
else:raise AssertionError('Occupied run accepted')
'''
        for mode in ['success','failure']:
            out=self.root/mode;out.mkdir()
            r=subprocess.run([sys.executable,'-B','-c',script,str(self.mesh),str(self.data),str(self.thesis),str(out),mode],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)


def material_fixture():
    """Reuse loader/logger fixtures; toy radiance/BVH, original A dispatch/losses."""
    from training_smoke_fixtures import trainer_fixture,defaults
    f=trainer_fixture();scope=dict(f.Trainer.__init__.__globals__)
    class CPUOnlyTorch:
        def __getattr__(self,name):return getattr(torch,name)
        def set_default_tensor_type(self,name):f.events.append(('default_type',name))
    cpu=CPUOnlyTorch();shader_scope={'torch':cpu,'np':np,'F':torch.nn.functional}
    methods=compile_members(A/'network/field.py','MCShadingNetwork',
        {'forward','anisotropic_forward','predict_anisotropic_components','anisotropic_regularization','get_orthogonal_directions','compute_tangent_bitangent_flat','get_lights'},shader_scope)
    class Shader(torch.nn.Module):
        def __init__(self,cfg,trace=None):
            super().__init__();self.cfg={**defaults(A/'network/field.py','MCShadingNetwork'),**cfg}
            self.feats_network=torch.nn.Linear(3,4)
            for name,count in [('mx_predictor',1),('my_predictor',1),('rotation_predictor',2),('metallic_predictor',1),('kd_predictor',3)]:
                setattr(self,name,torch.nn.Sequential(torch.nn.Linear(7,count),torch.nn.Sigmoid()))
            for name in ['outer_light','inner_light','human_light']:setattr(self,name,torch.nn.Linear(3,3))
            self.disconnect=None;self.skip_regularizer=False
            self.ray_trace_fun=lambda points,directions:(points,torch.ones_like(points),torch.ones((len(points),1)),torch.arange(len(points))%2==0)
        def sample_aniso_ggx_directions(self,pts,normals):return self.compute_tangent_bitangent_flat(normals,None)
        def predict_outer_lights(self,points,directions):return self.outer_light(points)
        def get_inner_lights(self,points,view_dirs,normals):return self.inner_light(points)
        def get_human_light(self,points,directions,human_poses):return self.human_light(points),torch.ones((len(points),1))*.25
        def compute_radiance(self,pts,mx,my,metallic,kd,rotation,human_poses):
            lights,*_=self.get_lights(pts,pts,human_poses)
            terms={'mx_predictor':mx,'my_predictor':my,'metallic_predictor':metallic,'kd_predictor':kd,'rotation_predictor':rotation.mean(-1,keepdim=True)}
            # Explicit toy graph, never a production BRDF substitute.
            return (sum(v.detach() if n==self.disconnect else v for n,v in terms.items())+lights)*.01
        def shade_anisotropic_mixed(self,pts,normals,view_dirs,mx,my,metallic,kd,rotation,rotation_raw,human_poses,is_train):
            t,b=self.sample_aniso_ggx_directions(pts,normals)
            rgb=self.compute_radiance(pts,mx,my,metallic,kd,rotation,human_poses)
            return rgb,dict(tangents=t,bitangents=b,mx=mx,my=my,metallic=metallic,kd=kd,rotation=rotation,rotation_raw=rotation_raw,
                            f_d=kd,f_s_sum=(mx+my).expand(-1,3),L_spec=(mx+my).expand(-1,3),diffuse_light=rgb)
    for name,value in methods.items():setattr(Shader,name,value)
    render_scope={'torch':cpu,'np':np,'F':torch.nn.functional,'MCShadingNetwork':Shader}
    render_methods=compile_members(A/'network/renderer.py','NeROMaterialRenderer',
        {'forward','train_step','shade','compute_rgb_loss','compute_diffuse_light_regularization','_init_shader'},render_scope)
    class Renderer(torch.nn.Module):
        def __init__(self,cfg):
            super().__init__();self.cfg={**defaults(A/'network/renderer.py','NeROMaterialRenderer'),**cfg}
            self._init_shader();self.warned_normal=False;self.ray_tracer=SimpleNamespace()
            self.mesh=SimpleNamespace(vertices=np.asarray([[0,0,0],[1,0,1],[0,1,0]],np.float32),triangles=np.asarray([[0,1,2]]),has_vertex_normals=lambda:True)
            self.train_ids=['1','2'];self.test_ids=['0'];self.nvs_ids=['3'];self.train_batch_i=0;self.tbn=2048
            self.train_batch={k:torch.ones((2048,3))*.2 for k in ['inters','rays_d','rgb','normals']}
            self.train_batch['normals'][:]=torch.tensor([0.,0.,1.]);self.train_batch['human_poses']=torch.zeros((2048,3,4))
            self.fail_step=None;f.networks.append(self)
        def cuda(self):return self
        def test_step(self,*a,**kw):raise AssertionError('No validation')
        def nvs(self,*a,**kw):raise AssertionError('No final NVS')
        def predict_materials(self,*a,**kw):raise AssertionError('No material extraction')
    for name,value in render_methods.items():setattr(Renderer,name,value)
    losses={};exec(compile((A/'network/loss.py').read_text(),str(A/'network/loss.py'),'exec'),losses)
    lr={};exec(compile((A/'train/lr_common_manager.py').read_text(),str(A/'train/lr_common_manager.py'),'exec'),lr)
    class Evaluator:
        def __init__(self,*a):pass
        def __call__(self,*a,**kw):raise AssertionError('No validation evaluator')
    scope.update(name2renderer={'material':Renderer},name2loss=losses['name2loss'],name2metrics={'mat_render':lambda cfg:None},
                 name2lr_manager=lr['name2lr_manager'],ValidationEvaluator=Evaluator)
    cls=next(n for n in ast.parse((A/'train/trainer.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Trainer')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(A/'train/trainer.py'),'exec'),scope)
    f.Trainer=scope['Trainer'];f.Renderer=Renderer;f.Shader=Shader;f.losses=losses;f.Evaluator=Evaluator
    return f


@unittest.skipIf(torch is None,'CPU Torch unavailable; no native fallback')
class MaterialLoopTests(unittest.TestCase):
    planning=PlanningTests.planning
    def setUp(self):
        PlanningTests.setUp(self);states=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        self.addCleanup(lambda:(random.setstate(states[0]),np.random.set_state(states[1]),torch.set_rng_state(states[2])))
    def test_shared_no_grad_wrapper_allows_Adam_but_forbids_original_bodies(self):
        f=material_fixture();entered=[]
        class Renderer(f.Renderer):
            @torch.no_grad()
            def test_step(self):entered.append('validation')
            @torch.no_grad()
            def predict_materials(self):entered.append('extraction')
        # Compile the actual pinned NVS body/decorator without importing a native
        # renderer or executing its body. The guard must stop before its first line.
        Renderer.nvs=compile_members(A/'network/renderer.py','NeROMaterialRenderer',{'nvs'},{'torch':torch})['nvs']
        class Trainer:
            @torch.no_grad()
            def _save_model(self):entered.append('checkpoint')
        class Evaluator:
            @torch.no_grad()
            def __call__(self):entered.append('evaluator')
        probe=MaterialProbe(SimpleNamespace(),torch,np,{},lambda:None)
        probe.bind_calls(Renderer,f.Shader,f.losses['NeRFRenderLoss'],f.losses['MaterialRegLoss'],Trainer,Evaluator)
        parameter=torch.nn.Parameter(torch.tensor([1.0],device='cpu'))
        optimizer=torch.optim.Adam([parameter],lr=.001,eps=1e-4,amsgrad=True)
        parameter.grad=torch.ones_like(parameter)
        # Torch1.8 Adam.step uses the same no_grad decorator as NVS. Newer host
        # Torch uses another optimizer wrapper, so wrap the real bound Adam.step
        # in no_grad to reproduce exactly the shared decorate_context code frame.
        step=torch.no_grad()(optimizer.step)
        wrapper=Renderer.nvs.__code__
        self.assertIs(wrapper,step.__code__)
        forbidden=[Renderer.nvs,Renderer.test_step,Renderer.predict_materials,Trainer._save_model,Evaluator.__call__]
        self.assertTrue(all(method.__code__ is wrapper for method in forbidden))
        self.assertNotIn(wrapper,probe.forbidden)
        self.assertTrue(all(inspect.unwrap(method).__code__ in probe.forbidden for method in forbidden))
        grad_enabled=torch.is_grad_enabled()
        with probe.observe_calls():step()
        self.assertLess(parameter.item(),1.0)
        self.assertEqual(int(optimizer.state[parameter]['step']),1)
        after_step=parameter.detach().clone()
        for method,args in [(Renderer.nvs,(None,'unused-logs','unused-images')),
                            *[(method,(None,)) for method in forbidden[1:]]]:
            with self.subTest(method=method.__qualname__):
                with self.assertRaisesRegex(SignatureError,'Validation/checkpoint/NVS'):
                    with probe.observe_calls():method(*args)
                self.assertIsNone(sys.getprofile())
                self.assertEqual(torch.is_grad_enabled(),grad_enabled)
        self.assertEqual(entered,[])
        self.assertTrue(torch.equal(parameter.detach(),after_step))
        self.assertEqual(int(optimizer.state[parameter]['step']),1)
    def run_material(self,fail=None):
        from training_smoke_fixtures import cpu_rng
        p=self.planning();run=RunStore(p['resolver']).create('cpu-material')
        from infrastructure.stage_workspace import prepare_view
        view,_,_=prepare_view(run,p['selected'].dataset,'bear',p['science'],p['configuration'],{})
        f=material_fixture();report={'mesh_identity':{'vertices':3,'triangles':1},'optimizer_calls_completed':0}
        with runtime_environment(run),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            os.chdir(view);trainer=trainer_io(f.Trainer,run)(deepcopy(p['configuration']))
            if fail:
                original=trainer._init_network
                def initialized():
                    original()
                    if fail=='missing_path':
                        trainer.network.shader_network.anisotropic_forward=lambda *a,**kw:f.Shader.shade_anisotropic_mixed(
                            trainer.network.shader_network,a[0],a[2],a[1],*trainer.network.shader_network.predict_anisotropic_components(a[0]),a[3],a[5])
                    elif fail=='gradient':
                        # Detach material outputs in both render and regularizer inputs;
                        # missing gradients are tested directly in a separate group gate.
                        for param in trainer.network.shader_network.mx_predictor.parameters():param.requires_grad_(False)
                    else:raise RuntimeError('CPU fixture initialization failure')
                trainer._init_network=initialized
            probe=MaterialProbe(trainer,torch,np,report,lambda:None,rng=cpu_rng,memory=lambda:{'CPU_fixture':True})
            probe.bind_calls(f.Renderer,f.Shader,f.losses['NeRFRenderLoss'],f.losses['MaterialRegLoss'],f.Trainer,f.Evaluator)
            controller=TwoUpdateTrace(trainer,probe,A/'train/trainer.py')
            try:
                with patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self),probe.observe_calls():controller.run()
            except BaseException:
                self.last=f,report,probe,controller,run;raise
        self.last=f,report,probe,controller,run
        return report
    def test_original_material_loop_two_updates_calls_losses_gradients_and_cleanup(self):
        report=self.run_material();f,_,_,controller,run=self.last
        self.assertEqual(controller.completed,2);self.assertEqual(len(report['updates']),2)
        self.assertEqual([u['lr'] for u in report['updates']],[0.,5e-7])
        self.assertEqual(report['updates'][0]['changed_elements'],0);self.assertGreater(report['updates'][1]['changed_elements'],0)
        for update in report['updates']:
            self.assertTrue(all(g['gradient_l2_cpu64']>0 for g in update['groups_before_clip'].values()))
            self.assertEqual(set(update['groups_before_clip']),set(GROUPS))
            self.assertGreater(update['aggregate_loss'],0);self.assertEqual(update['calls']['shader.anisotropic_forward']['returned'],1)
            self.assertEqual(update['calls']['loss.mat_reg']['returned'],1)
            self.assertTrue(all(v['step']==update['update_index'] for v in update['optimizer']['state'].values()))
        self.assertEqual(report['effective_shader']['reg_energy_loss_lambda'],.05)
        self.assertEqual(report['declared_top_level_regularization']['reg_energy_loss_lambda'],.01)
        self.assertTrue(f.loaders[0].iterator.closed);self.assertTrue(f.writers[0].closed)
        self.assertFalse(list(run.checkpoints.iterdir()));self.assertIsNone(sys.gettrace());self.assertIsNone(sys.getprofile())
    def test_material_failure_retains_update_count_and_closes_resources(self):
        with self.assertRaisesRegex(SignatureError,'missing/zero material gradients'):self.run_material('gradient')
        f,report,_,controller,_=self.last
        self.assertEqual(controller.completed,0);self.assertEqual(report['optimizer_calls_completed'],0)
        self.assertTrue(f.writers[0].closed);self.assertTrue(f.loaders[0].iterator.closed)
        self.assertIsNone(sys.gettrace());self.assertIsNone(sys.getprofile())
    def test_missing_anisotropic_dispatch_is_rejected_before_backward(self):
        with self.assertRaisesRegex(SignatureError,'original anisotropic'):self.run_material('missing_path')
        f,report,_,controller,_=self.last
        self.assertEqual(controller.completed,0);self.assertTrue(f.writers[0].closed)
    def test_inactive_lighting_is_distinguished_from_unexpected_missing_material_gradients(self):
        report=self.run_material();_,_,probe,_,_=self.last
        records=deepcopy(report['updates'][0]['gradients_before_clip']);probe.current={'light_paths':{'inner':0,'outer':1,'human':1}}
        for n in records:
            if n.startswith('shader_network.inner_light.'):records[n]=None
        groups=probe.group_gradients(records);self.assertEqual(groups['inner_light']['status'],'expected_inactive')
        probe.current['light_paths']['inner']=1
        with self.assertRaisesRegex(SignatureError,'inner_light'):probe.group_gradients(records)
        records=deepcopy(report['updates'][0]['gradients_before_clip'])
        for n in records:
            if n.startswith('shader_network.outer_light.'):records[n]=None
        probe.current['light_paths']={'inner':1,'outer':0,'human':1}
        groups=probe.group_gradients(records)
        self.assertEqual(groups['outer_light']['status'],'expected_inactive')
        self.assertEqual(groups['human_light']['status'],'active')
