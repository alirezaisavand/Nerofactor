"""Disposable CPU ray/loop fixtures; never load original cameras or GPU models."""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import random
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

from infrastructure import bell_material_integration as integration
from infrastructure.bell_material_rays import summarize,TraceCapture,original_adapter,BOUNDS
from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.signatures import SignatureError
from test_material_smoke import tiny_mesh,compile_members,material_fixture

A=ROOT/'scientific_sources/material_a'


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.data=self.root/'data';self.thesis=self.root/'thesis';self.geom=self.root/'geometry';self.out=self.root/'out'
        (self.data/'GlossySynthetic/bell/masks_train').mkdir(parents=True);self.thesis.mkdir()
        (self.data/'GlossySynthetic/bell/masks_train/0.png').write_bytes(b'opaque CPU fixture')
        self.mesh=tiny_mesh(self.geom/integration.MESH_NAME)
        _,source=baseline('geometry',PINS['geometry'])
        self.proof=self.geom/'G-mesh-diagnostic.json'
        self.report={'protocol':'bell-historical-best-G-mesh-r128-v1','run_id':integration.EXTRACTION_RUN,
            'status':'complete','phase':'complete','scientific_commit':PINS['geometry'],'source':source,
            'strict_network_loaded':True,'network_unchanged':True,'checkpoint_selection':{'sha256':'d928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc'},
            'mesh':{'path':integration.MESH_NAME,'sha256':digest(self.mesh),'explicit_repairs_or_transforms':[]},
            'settings':{'resolution':128},'research_handoff_admitted':False}
        self.proof.write_text(json.dumps(self.report))
        for key,value in [('MESH_SHA256',digest(self.mesh)),('EXTRACTION_SHA256',digest(self.proof)),('MESH_COUNTS',(3,1))]:
            manager=patch.object(integration,key,value);manager.start();self.addCleanup(manager.stop)
    def planning(self,**kwargs):
        args=dict(commit=PINS['material'],datasets=self.data,thesis=self.thesis,geometry=self.geom,output=self.out,
                  mesh=self.mesh,mesh_sha256=integration.MESH_SHA256,extraction=self.proof,
                  extraction_sha256=integration.EXTRACTION_SHA256,action='rays')
        args.update(kwargs);return integration.plan(**args)
    def test_exact_config_independent_exception_and_no_RNG_or_allocation(self):
        state=random.getstate();p=self.planning()
        self.assertEqual(random.getstate(),state);self.assertFalse(self.out.exists())
        self.assertEqual(p['configuration'],yaml.safe_load((A/integration.CONFIG).read_text()))
        self.assertIsNone(p['selected'].checkpoint);self.assertEqual(p['selected'].mesh,self.mesh)
        from infrastructure.stage_workspace import handoff
        with self.assertRaises(SignatureError):handoff(self.proof,self.mesh,p['resolver'],'syn/bell')
        from infrastructure.stage_execution import plan
        from infrastructure.material_smoke import CONFIG
        with self.assertRaises(Exception):plan('material',PINS['material'],'C',self.data,self.thesis,self.out,ROOT/CONFIG,ROOT)
    def test_invalid_source_mesh_report_paths_and_missing_gate_fail_before_native(self):
        for override in [{'commit':PINS['geometry']},{'mesh_sha256':'0'*64},{'extraction_sha256':'0'*64},
                         {'mesh':self.mesh.parent/'absent.ply'},{'output':self.geom/'overlap'},
                         {'action':'two-update'},{'ray_sha256':'0'*64}]:
            with self.subTest(override=override),self.assertRaises((ValueError,OSError)):self.planning(**override)
        alias=self.root/'alias.ply';alias.symlink_to(self.mesh)
        with self.assertRaises(ValueError):self.planning(mesh=alias)
        self.assertFalse(self.out.exists())
    def test_mesh_loader_must_preserve_vertices_and_triangle_winding(self):
        p=self.planning()
        mesh=SimpleNamespace(vertices=np.asarray([[0.,0.,0.],[1.,0.,1.],[0.,1.,0.]]),triangles=np.asarray([[0,1,2]]))
        integration.verify_loaded_mesh(mesh,p,np)
        mesh.triangles=mesh.triangles[:,::-1]
        with self.assertRaisesRegex(ValueError,'triangle ordering/winding'):integration.verify_loaded_mesh(mesh,p,np)
    def gate(self):
        p=self.planning();self.raydir=self.root/'rays';self.raydir.mkdir();self.ray=self.raydir/'bell-material-diagnostic.json'
        from infrastructure.stage_workspace import source_records
        value={'schema_version':1,'protocol':integration.RAY_PROTOCOL,'status':'complete','ray_gate_passed':True,
               'scientific_commit':PINS['material'],'mesh_identity':p['mesh_identity'],'extraction_report':p['proof_identity'],
               'legacy_sha256':__import__('hashlib').sha256(p['raw']).hexdigest(),'infrastructure_files':integration.identity(),
               'bounds':BOUNDS,'optimizer_calls_completed':0,'input_files':source_records(p['selected'].dataset/'bell'),
               'splits':integration.expected_splits(),'views':[]}
        for role in ['train','validation','nvs']:
            q=self.raydir/(role+'.npz');q.write_bytes(b'CPU fixture artifact')
            value['views'].append({'role':role,'source_camera_id':value['splits'][role][0],'resolution':[800,800],
                                  'artifact':q.name,'sha256':digest(q),'statistics':{'passed':True,'checks':dict.fromkeys(
                    ['original_negation','unit_final_normals','intersection_consistency',
                     'front_facing_final_normals','raw_normals_opposite_view','silhouette_alignment'],True)}})
        self.ray.write_text(json.dumps(value));return value
    def test_two_update_requires_intact_gate_inputs_settings_and_artifacts(self):
        value=self.gate()
        p=self.planning(action='two-update',ray_report=self.ray,ray_sha256=digest(self.ray))
        self.assertEqual(p['ray']['protocol'],integration.RAY_PROTOCOL);self.assertFalse(self.out.exists())
        for key,bad in [('status','failed'),('ray_gate_passed',False),('bounds',{}),('infrastructure_files',{}),('views',[]),('splits',{})]:
            rejected={**value,key:bad};self.ray.write_text(json.dumps(rejected))
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.planning(action='two-update',ray_report=self.ray,ray_sha256=digest(self.ray))
        self.ray.write_text(json.dumps(value));(self.raydir/'train.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            self.planning(action='two-update',ray_report=self.ray,ray_sha256=digest(self.ray))
        value['views'][0]['source_camera_id']='0'
        self.ray.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'Incomplete/failed ray gates'):
            self.planning(action='two-update',ray_report=self.ray,ray_sha256=digest(self.ray))
    def test_validate_only_cannot_import_native_or_load_cameras(self):
        script='''
import builtins,sys
original=builtins.__import__
def guarded(name,*a,**kw):
 assert name.split('.')[0] not in {'torch','numpy','network','dataset','train','open3d','_raytracing'},name
 return original(name,*a,**kw)
builtins.__import__=guarded
from run_bell_material_diagnostic import main
raise SystemExit(main(sys.argv[1:]))
'''
        # Rejection of the synthetic hash is intentional; even rejection must not import science.
        result=subprocess.run([sys.executable,'-B','-c',script,'--action','rays','--scientific-commit',PINS['material'],
            '--datasets-root',str(self.data),'--thesis-root',str(self.thesis),'--geometry-root',str(self.geom),
            '--output-root',str(self.out),'--mesh',str(self.mesh),'--mesh-sha256',digest(self.mesh),
            '--extraction-report',str(self.proof),'--extraction-sha256',digest(self.proof),'--run-id','fixture','--validate-only'],
            cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,2,result.stderr);self.assertIn('admitted mesh/report hashes',result.stderr)
        self.assertFalse(self.out.exists())
    def test_owned_execution_failure_logging_collisions_and_no_input_mutation(self):
        # A subprocess preserves cold namespace selection; backend is an explicit CPU fixture.
        script='''
import sys,os,json
from pathlib import Path
from unittest.mock import patch
from infrastructure import bell_material_integration as b
from infrastructure.stage_sources import PINS,digest
data,thesis,geometry,out=map(Path,sys.argv[1:]);mesh=geometry/b.MESH_NAME;proof=geometry/'G-mesh-diagnostic.json'
b.MESH_SHA256=digest(mesh);b.EXTRACTION_SHA256=digest(proof);b.MESH_COUNTS=(3,1)
p=b.plan(PINS['material'],data,thesis,geometry,out,mesh,digest(mesh),proof,digest(proof),'rays')
before=os.getcwd(),os.environ.get('HOME'),mesh.read_bytes(),proof.read_bytes()
def fail(*a):print('CPU fixture failure');raise RuntimeError('explicit fixture error')
with patch('infrastructure.stage_execution.native_guards'),patch.object(b,'native',fail):
 try:b.execute(p,'fixture','CPU-fixture-image','0'*40,True)
 except RuntimeError:pass
 else:raise AssertionError('failure suppressed')
assert before==(os.getcwd(),os.environ.get('HOME'),mesh.read_bytes(),proof.read_bytes())
r=json.loads((out/'bell-material-diagnostic.json').read_text())
assert r['status']=='failed' and not r['research_handoff_admitted'] and r['optimizer_calls_completed']==0
assert 'CPU fixture failure' in (out/'logs/console.log').read_text()
assert not list((out/'checkpoints').iterdir())
from infrastructure.runs import RunStore
try:RunStore(p['resolver'],layout='mounted').create('new-id')
except ValueError:pass
else:raise AssertionError('occupied evidence overwritten')
'''
        result=subprocess.run([sys.executable,'-B','-c',script,*map(str,[self.data,self.thesis,self.geom,self.out])],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)


class RayStatisticsTests(unittest.TestCase):
    def arrays(self):
        return {'origins':np.tile([0.,0.,2.],(4,1)),'directions':np.tile([0.,0.,-1.],(4,1)),
                'positions':np.zeros((4,3)),'raw_normals':np.tile([0.,0.,-1.],(4,1)),
                'normals':np.tile([0.,0.,1.],(4,1)),'depth':np.ones((4,1))*2,'hit':np.ones((4,1),bool)}
    def test_normals_and_silhouette_report_without_repairs(self):
        arrays=self.arrays();before={k:v.copy() for k,v in arrays.items()}
        r=summarize(arrays,np.ones((2,2),bool),np)
        self.assertTrue(r['passed']);self.assertEqual(r['silhouette_iou'],1.);self.assertEqual(r['raw_back_facing_fraction'],1.)
        for k,v in arrays.items():np.testing.assert_array_equal(v,before[k])
        arrays['normals']*=-1
        self.assertFalse(summarize(arrays,np.ones((2,2),bool),np)['passed'])
        arrays=self.arrays();self.assertFalse(summarize(arrays,np.zeros((2,2),bool),np)['passed'])
    def test_invalid_depth_normals_intersections_and_no_hits_rejected(self):
        for key,value in [('depth',-1.),('positions',np.nan),('raw_normals',0.)]:
            a=self.arrays();a[key][:]=value
            with self.subTest(key=key),self.assertRaises(ValueError):summarize(a,np.ones((2,2),bool),np)
        a=self.arrays();a['depth'][:]=10;a['hit'][:]=False
        with self.assertRaisesRegex(ValueError,'No mesh intersections'):summarize(a,np.ones((2,2),bool),np)
        a=self.arrays();a['positions']+=.1
        self.assertFalse(summarize(a,np.ones((2,2),bool),np)['checks']['intersection_consistency'])


@unittest.skipIf(torch is None,'CPU Torch unavailable')
class OriginalBodyTests(unittest.TestCase):
    def test_original_camera_rays_trace_negation_and_observer_cleanup(self):
        class RayTracer:
            def trace(self,rays_o,rays_d,inplace=False):
                depth=-rays_o[:,2]/rays_d[:,2]
                return rays_o+depth[:,None]*rays_d,torch.tensor([0.,0.,-1.]).repeat(len(depth),1),depth
        geometry=SimpleNamespace(vertices=np.zeros((9,3)),triangles=np.zeros((9,3),int),has_vertex_normals=lambda:True)
        scope={'torch':torch,'np':np,'F':torch.nn.functional,'open3d':SimpleNamespace(io=SimpleNamespace(read_triangle_mesh=lambda p:geometry)),
               'raytracing':SimpleNamespace(RayTracer=lambda v,f:RayTracer())}
        names=['_init_geometry','trace','trace_in_batch','_warn_ray_tracing','get_human_coordinate_poses','_construct_ray_batch']
        methods=compile_members(A/'network/renderer.py','NeROMaterialRenderer',set(names),scope)
        renderer=type('OriginalMethods',(),{**methods,'default_cfg':{'fixed_camera':False}})
        owner=original_adapter(renderer,{'mesh':'CPU-fixture'})
        capture=TraceCapture(owner,torch);state=torch.get_rng_state().clone()
        info={'imgs':torch.ones((1,3,2,2)),'Ks':torch.eye(3)[None],
              'poses':torch.cat([torch.diag(torch.tensor([1.,-1.,-1.])),torch.tensor([[0.],[0.],[2.]])],dim=1)[None]}
        with torch.no_grad(),capture.active():batch=owner._construct_ray_batch(info,is_train=False)
        arrays=capture.arrays(np);self.assertTrue(summarize(arrays,np.ones((2,2),bool),np)['passed'])
        self.assertEqual(batch['hit_mask'].sum().item(),4);self.assertIsNone(sys.getprofile());self.assertTrue(torch.equal(state,torch.get_rng_state()))
        with self.assertRaises(RuntimeError):
            with torch.no_grad(),TraceCapture(owner,torch).active():raise RuntimeError('fixture')
        self.assertIsNone(sys.getprofile())
    def test_bell_original_loop_reuses_two_update_diagnostics_without_human_network(self):
        from infrastructure.stage_execution import trainer_io
        from infrastructure.runs import RunStore
        from infrastructure.paths import PathResolver
        from infrastructure.bell_inference import runtime_environment
        from infrastructure.bounded_training import TwoUpdateTrace
        from training_smoke_fixtures import cpu_rng
        states=random.getstate(),np.random.get_state(),torch.get_rng_state()
        self.addCleanup(lambda:(random.setstate(states[0]),np.random.set_state(states[1]),torch.set_rng_state(states[2])))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';thesis=root/'thesis';data.mkdir();thesis.mkdir()
            run=RunStore(PathResolver({'datasets':data,'thesis':thesis},root/'out')).create('bell-fixture')
            f=material_fixture();original=f.Shader.__init__
            def init(self,*a,**kw):
                original(self,*a,**kw)
                if not self.cfg['human_lights']:del self.human_light
            f.Shader.__init__=init
            cfg=yaml.safe_load((A/integration.CONFIG).read_text())
            report={'mesh_identity':{'vertices':3,'triangles':1},'ray_splits':{'train':['1','2'],'validation':['0'],'nvs':['3']},'optimizer_calls_completed':0}
            with runtime_environment(run),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                trainer=trainer_io(f.Trainer,run)(cfg)
                probe=integration.BellProbe(trainer,torch,np,report,lambda:None,rng=cpu_rng,memory=lambda:{'CPU_fixture':True})
                probe.bind_calls(f.Renderer,f.Shader,f.losses['NeRFRenderLoss'],f.losses['MaterialRegLoss'],f.Trainer,f.Evaluator)
                with patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self),probe.observe_calls():
                    TwoUpdateTrace(trainer,probe,A/'train/trainer.py').run()
            self.assertEqual(report['optimizer_calls_completed'],2);self.assertEqual([u['lr'] for u in report['updates']],[0.,5e-7])
            self.assertEqual(report['updates'][0]['changed_elements'],0);self.assertGreater(report['updates'][1]['changed_elements'],0)
            self.assertFalse(report['effective_shader']['human_lights']);self.assertFalse(report['effective_shader']['is_real'])
            self.assertEqual(report['effective_shader']['reg_energy_loss_lambda'],.05)
            self.assertEqual(report['declared_top_level_regularization']['reg_energy_loss_lambda'],.01)
            self.assertTrue(all('human_light' not in u['groups_before_clip'] for u in report['updates']))
            self.assertTrue(f.writers[0].closed);self.assertTrue(f.loaders[0].iterator.closed)
            self.assertFalse(list(run.checkpoints.iterdir()));self.assertIsNone(sys.gettrace());self.assertIsNone(sys.getprofile())


if __name__=='__main__':unittest.main()
