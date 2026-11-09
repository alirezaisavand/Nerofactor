"""CPU geometry and original-G function fixtures; no historical pickle/model loads."""
import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
try:import torch
except (ImportError,OSError):torch=None
from infrastructure import g_extraction as extraction
from infrastructure.mesh_diagnostics import geometry,export_diagnostic
from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.paths import PathResolver
from infrastructure.runs import RunStore
from infrastructure.signatures import SignatureError

G=ROOT/'scientific_sources/geometry_g'
V=np.asarray([[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]],np.float64)
F=np.asarray([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)


def original_functions(scope):
    tree=ast.parse((G/'network/field.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['extract_fields','extract_geometry']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(G/'network/field.py'),'exec'),scope)
    return scope


def mesh_library_fixture():
    events=[];holder={}
    class Mesh:
        def __init__(self,*args,**kwargs):
            events.append(('constructor',len(args),kwargs));self.vertices=np.asarray(args[0]).copy();self.faces=np.asarray(args[1]).copy();self.exported=False
            holder['mesh']=self
        def export(self,path):
            events.append(('export',path));self.exported=True
            Path(path).write_text('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 4\nproperty list uchar int vertex_indices\nend_header\nCPU fixture only\n')
        @property
        def face_normals(self):
            assert self.exported,'Lazy normal access changed the original pre-export cache state'
            events.append('face_normals');a,b,c=(self.vertices[self.faces[:,i]] for i in range(3))
            n=np.cross(b-a,c-a);return n/np.linalg.norm(n,axis=1,keepdims=True)
        @property
        def vertex_normals(self):
            assert self.exported;events.append('vertex_normals');return np.tile([1.,0.,0.],(len(self.vertices),1))
    def load_mesh(path,process):
        assert process is False and Path(path).is_file();events.append('readback_without_processing')
        m=holder['mesh'];return SimpleNamespace(vertices=m.vertices.astype(np.float32),faces=m.faces.copy())
    return SimpleNamespace(Trimesh=Mesh,load_mesh=load_mesh,events=events)


class ExtractionPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.datasets=self.root/'datasets';self.thesis=self.root/'thesis';self.proofdir=self.root/'compatibility';self.out=self.root/'outputs'
        for p in [self.datasets,self.thesis,self.proofdir]:p.mkdir()
        self.checkpoint=self.thesis/extraction.CHECKPOINT;self.checkpoint.parent.mkdir(parents=True);self.checkpoint.write_bytes(b'opaque synthetic checkpoint, not deserialized')
        science,source=baseline('geometry',PINS['geometry']);raw=(science/'configs/shape/syn/bell.yaml').read_bytes()
        catalog={'CPU.fixture':{'shape':[1],'dtype':'torch.float32'}}
        self.proof=self.proofdir/'G-diagnostic.json'
        self.prior={'protocol':'bell-G-checkpoint-compatibility-v1','action':'compatibility','status':'complete','phase':'complete',
                    'scientific_commit':PINS['geometry'],'checkpoint_sha256':digest(self.checkpoint),'strict_network_loaded':True,
                    'checkpoint_metadata':{'stored_step':150000},'source':{'scientific_files':source['files']},
                    'legacy_sha256':hashlib.sha256(raw).hexdigest(),'network_comparison':{'result':'exact_structural_match',
                    'missing_keys':[],'unexpected_keys':[],'shape_mismatches':[],'dtype_mismatches':[],
                    'checkpoint_network_catalog':catalog,'G_network_catalog':catalog},'loaded_network_identity':{'CPU.fixture':'fixture'},
                    'infrastructure_commit_declared':'0'*40}
        self.proof.write_text(json.dumps(self.prior))
    def prepare(self,**overrides):
        args=dict(commit=PINS['geometry'],datasets=self.datasets,thesis=self.thesis,outputs=self.out,
                  checkpoint=self.checkpoint,checkpoint_sha256=digest(self.checkpoint),compatibility=self.proof,compatibility_sha256=digest(self.proof))
        args.update(overrides)
        with patch.object(extraction,'CHECKPOINT_SHA256',digest(self.checkpoint)),patch.object(extraction,'COMPATIBILITY_SHA256',digest(self.proof)):
            return extraction.prepare(**args)
    def run_paths(self):return RunStore(PathResolver({'datasets':self.datasets,'thesis':self.thesis},self.out)).create('mesh-fixture')
    def test_preflight_is_metadata_only_no_RNG_allocation_or_checkpoint_loading(self):
        before=random.getstate();p=self.prepare()
        self.assertEqual(p['checkpoint'],self.checkpoint);self.assertEqual(p['resolution'],128)
        self.assertEqual(random.getstate(),before);self.assertFalse(self.out.exists())
        self.assertEqual(p['cfg']['name'],'bell_shape')
    def test_source_checkpoint_resolution_and_compatibility_gates(self):
        for key,value in [('commit',PINS['material']),('resolution',512),('resolution',True),('checkpoint_sha256','0'*64),
                          ('compatibility_sha256','0'*64),('checkpoint',self.thesis/'model.pth')]:
            with self.subTest(key=key),self.assertRaises(ValueError):self.prepare(**{key:value})
        for key,value in [('status','running'),('strict_network_loaded',False),('scientific_commit',PINS['material']),
                          ('loaded_network_identity',{})]:
            changed=deepcopy(self.prior);changed[key]=value;self.proof.write_text(json.dumps(changed))
            with self.subTest(key=key),self.assertRaises(SignatureError):self.prepare()
        self.assertFalse(self.out.exists())
    def test_output_containment_and_explicit_trust_before_native_work(self):
        for out in [ROOT,ROOT/'outputs',self.thesis/'output']:
            with self.assertRaises(ValueError):self.prepare(outputs=out)
        p=self.prepare()
        with patch.object(extraction,'native') as native,patch.object(extraction,'runtime_guards') as guards:
            with self.assertRaises(SignatureError):extraction.execute(p,'fixture',extraction.IMAGE,'0'*40)
            native.assert_not_called();guards.assert_not_called();self.assertFalse(self.out.exists())
        with patch.object(extraction,'native') as native,patch.object(extraction,'runtime_guards',side_effect=SignatureError('runtime blocked')):
            with self.assertRaises(SignatureError):extraction.execute(p,'fixture',extraction.IMAGE,'0'*40,trusted=True)
            native.assert_not_called();self.assertFalse(self.out.exists())
    def test_geometry_validation_winding_and_no_repairs(self):
        before=V.copy(),F.copy();r=geometry(V,F,np)
        self.assertEqual(r['nondegenerate_triangles'],4);self.assertTrue(r['watertight_by_edge_incidence']);self.assertEqual(r['boundary_edges'],0)
        self.assertEqual(r['inconsistently_wound_two_face_edges'],0);self.assertGreater(r['signed_volume_diagnostic'],0)
        reversed_faces=F[:,::-1].copy();flipped=geometry(V,reversed_faces,np)
        self.assertLess(flipped['signed_volume_diagnostic'],0);self.assertFalse(flipped['global_outward_orientation_verified'])
        inconsistent=F.copy();inconsistent[0]=inconsistent[0,::-1]
        self.assertGreater(geometry(V,inconsistent,np)['inconsistently_wound_two_face_edges'],0)
        for vertices,faces in [(V[:0],F),(V,F[:0]),(V,np.asarray([[0,0,1]])),(V,np.asarray([[0,1,4]])),
                               (V,F.astype(float)),(V+2,F),(V*np.nan,F)]:
            with self.assertRaises(SignatureError):geometry(vertices,faces,np)
        np.testing.assert_array_equal(V,before[0]);np.testing.assert_array_equal(F,before[1])
    def test_original_default_constructor_export_order_hashes_and_collision(self):
        run=self.run_paths();library=mesh_library_fixture();result=export_diagnostic(V,F,run,np,library)
        self.assertEqual(library.events[0],('constructor',2,{}));self.assertEqual(library.events[1][0],'export')
        self.assertEqual(result['sha256'],digest(run.output(result['path'])));self.assertEqual(result['raw']['faces'],4)
        self.assertEqual(result['explicit_repairs_or_transforms'],[])
        np.testing.assert_array_equal(np.load(run.output('meshes/raw-faces.npy'),allow_pickle=False),F)
        with self.assertRaises(SignatureError):export_diagnostic(V,F,run,np,library)
    def test_validation_only_CLI_has_no_model_or_numpy_imports(self):
        script='''
import builtins,sys
from pathlib import Path
from infrastructure import g_extraction as extraction
checkpoint,proof=sys.argv[1:3];del sys.argv[1:3]
extraction.CHECKPOINT_SHA256=extraction.digest(Path(checkpoint))
extraction.COMPATIBILITY_SHA256=extraction.digest(Path(proof))
original=builtins.__import__
def guarded(name,*a,**kw):
 assert name.split('.')[0] not in {'torch','numpy','trimesh','mcubes','network','train','dataset','open3d'},name
 return original(name,*a,**kw)
builtins.__import__=guarded
from run_g_mesh_diagnostic import main
raise SystemExit(main(sys.argv[1:]))
'''
        args=['--scientific-commit',PINS['geometry'],'--datasets-root',str(self.datasets),'--thesis-root',str(self.thesis),
              '--output-root',str(self.out),'--checkpoint',str(self.checkpoint),'--checkpoint-sha256',digest(self.checkpoint),
              '--compatibility-report',str(self.proof),'--compatibility-sha256',digest(self.proof),'--resolution','128','--run-id','fixture','--validate-only']
        r=subprocess.run([sys.executable,'-B','-c',script,str(self.checkpoint),str(self.proof),*args],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(self.out.exists())
    def test_orchestration_failure_logging_cleanup_collision_and_no_research_handoff(self):
        script='''
import json,os,sys
from pathlib import Path
from infrastructure import g_extraction as e
from infrastructure.stage_sources import PINS,digest
checkpoint,proof,data,thesis,out,mode=map(Path,sys.argv[1:])
e.CHECKPOINT_SHA256=digest(checkpoint);e.COMPATIBILITY_SHA256=digest(proof)
p=e.prepare(PINS['geometry'],data,thesis,out,checkpoint,digest(checkpoint),proof,digest(proof))
e.runtime_guards=lambda *a:None # Explicit CPU fixture, never native execution.
def fake(p,run,report,mark):
 mark('CPU_fixture_backend');print('CPU fixture extraction')
 if str(mode)=='failure':raise RuntimeError('CPU fixture extraction failure')
 if str(mode)=='missing-mesh':return
 path=run.output('meshes/bell_shape-best-150000-r128.ply');path.write_text('CPU fixture mesh, not research geometry')
 report.update(strict_network_loaded=True,network_unchanged=True,mesh={'path':path.relative_to(run.root).as_posix(),'sha256':digest(path)},
   extraction_calls={'extract_geometry_calls':1,'extract_fields_calls':1,'sdf_calls':8,'sdf_points':128**3,'maximum_query_points':64**3})
e.native=fake
before=os.getcwd(),os.environ.get('HOME'),checkpoint.read_bytes(),proof.read_bytes()
try:e.execute(p,'fixture',e.IMAGE,'0'*40,trusted=True)
except (RuntimeError,ValueError):assert str(mode)!='success'
else:assert str(mode)=='success'
assert (os.getcwd(),os.environ.get('HOME'),checkpoint.read_bytes(),proof.read_bytes())==before
r=json.loads((out/'G-mesh-diagnostic.json').read_text())
assert r['status']==('complete' if str(mode)=='success' else 'failed')
assert r['checkpoint_selection']['selected_artifact']=='model_best.pth' and not r['checkpoint_selection']['silent_substitution']
assert not r['research_handoff_admitted'] and not r['research_geometry_admitted'] and not r['training_resumed']
assert r['optimizer_updates']==0 and not r['datasets_read'] and not list((out/'checkpoints').iterdir())
assert 'CPU fixture extraction' in (out/'logs/console.log').read_text()
from infrastructure.runs import RunStore
try:RunStore(p['resolver'],layout='mounted').create('fixture')
except ValueError:pass
else:raise AssertionError('Collision accepted')
if str(mode)=='success':
 from infrastructure.stage_workspace import handoff
 h={'schema_version':1,'geometry_algorithm':'geometry-G','scientific_commit':PINS['geometry'],'geometry_run_id':'fixture',
 'geometry_run_manifest_sha256':digest(out/'G-mesh-diagnostic.json'),'geometry_run_manifest':{'root':'compatibility','path':'run.json'},
 'mesh_sha256':r['mesh']['sha256'],'database_name':'syn/bell','coordinates':'G-SDF','scale':1.,'normals':'original',
 'orientation':'unverified','material_preprocessing':'A-original-read-triangles-and-compute-missing-vertex-normals-no-file-write'}
 # An inspection-only manifest cannot establish an attributable G training run.
 target=proof.parent/'run.json';target.write_text(json.dumps(r))
 h['geometry_run_manifest_sha256']=digest(target)
 hp=proof.parent/'handoff.json';hp.write_text(json.dumps(h))
 try:handoff(hp,out/'meshes/bell_shape-best-150000-r128.ply',p['resolver'],'syn/bell')
 except ValueError:pass
 else:raise AssertionError('Diagnostic extraction admitted as research handoff')
'''
        for mode in ['success','failure','missing-mesh']:
            out=self.root/mode;out.mkdir()
            r=subprocess.run([sys.executable,'-B','-c',script,str(self.checkpoint),str(self.proof),str(self.datasets),str(self.thesis),str(out),mode],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)


@unittest.skipIf(torch is None,'CPU Torch unavailable; no native fallback')
class OriginalExtractionTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(lambda:torch.set_num_threads(self.threads))
    def fixture(self):
        seen={}
        def cubes(field,threshold):
            seen['field']=field.copy();seen['threshold']=threshold
            return (V+1)*127/2,F.copy() # Voxel-space fixture output, original conversion maps back.
        scope=original_functions({'torch':torch,'np':np,'mcubes':SimpleNamespace(marching_cubes=cubes)})
        tree=ast.parse((G/'network/field.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SDFNetwork')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sdf')
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(G/'network/field.py'),'exec'),scope)
        class SDF:
            def forward(self,x):return (x*x).sum(-1,keepdim=True)-.25
            sdf=scope['sdf']
        class Model:
            sdf_network=SDF()
            def forward(self,*a):raise AssertionError('No renderer forward')
            def _init_dataset(self):raise AssertionError('No dataset')
        return scope,Model(),seen
    def test_original_128_field_queries_sphere_mask_conversion_and_observer_RNG_cleanup(self):
        scope,model,seen=self.fixture();rng=torch.get_rng_state().clone();python_rng=random.getstate()
        audit=extraction.ExtractionAudit(model,scope['extract_geometry'],scope['extract_fields'],torch,np)
        with audit.active(),torch.no_grad():vertices,faces=scope['extract_geometry'](-torch.ones(3),torch.ones(3),128,0,lambda x:model.sdf_network.sdf(x))
        calls=audit.complete();self.assertEqual(calls['sdf_points'],128**3);self.assertEqual(seen['threshold'],0)
        self.assertEqual(seen['field'].dtype,np.float32);self.assertEqual(seen['field'][0,0,0],1.)
        self.assertLess(seen['field'][64,64,64],0);np.testing.assert_allclose(vertices,V,atol=0,rtol=0)
        np.testing.assert_array_equal(faces,F);self.assertIsNone(sys.getprofile());self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(random.getstate(),python_rng)
    def test_wrong_budget_and_nonfinite_SDF_fail_closed(self):
        scope,model,seen=self.fixture();audit=extraction.ExtractionAudit(model,scope['extract_geometry'],scope['extract_fields'],torch,np)
        with self.assertRaisesRegex(SignatureError,'extraction request'):
            with audit.active(),torch.no_grad():scope['extract_geometry'](-torch.ones(3),torch.ones(3),512,0,lambda x:model.sdf_network.sdf(x))
        self.assertIsNone(sys.getprofile());self.assertFalse(seen)
        model.sdf_network.forward=lambda x:torch.full((len(x),1),float('nan'))
        audit=extraction.ExtractionAudit(model,scope['extract_geometry'],scope['extract_fields'],torch,np)
        with self.assertRaisesRegex(SignatureError,'Nonfinite SDF'):
            with audit.active(),torch.no_grad():scope['extract_geometry'](-torch.ones(3),torch.ones(3),128,0,lambda x:model.sdf_network.sdf(x))
        self.assertIsNone(sys.getprofile());self.assertFalse(seen)
    def test_optional_real_CPU_PyMCubes_Trimesh_export_readback(self):
        try:import trimesh,mcubes
        except (ImportError,OSError):self.skipTest('Native CPU PyMCubes/Trimesh unavailable; original field/coordinate math and export ordering use explicit stand-ins')
        scope=original_functions({'torch':torch,'np':np,'mcubes':mcubes})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';thesis=root/'thesis';data.mkdir();thesis.mkdir()
            run=RunStore(PathResolver({'datasets':data,'thesis':thesis},root/'out')).create('analytic-cpu-fixture')
            with torch.no_grad():v,f=scope['extract_geometry'](-torch.ones(3),torch.ones(3),8,0,lambda x:(x*x).sum(-1,keepdim=True)-.25)
            info=export_diagnostic(v,f,run,np,trimesh)
            self.assertTrue(info['serialized']['watertight_by_edge_incidence']);self.assertEqual(info['sha256'],digest(run.output(info['path'])))
