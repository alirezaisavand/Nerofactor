"""Deterministic analytical fixtures; no model, original pickle, GPU or training."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from infrastructure import geometry_scientific_audit as audit
from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.signatures import SignatureError

SIZE=(6,8)


def fixture_mesh(path):
    vertices=np.asarray([[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],
                         [-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]],np.float32)
    faces=np.asarray([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,5,1],[0,4,5],
                      [3,2,6],[3,6,7],[0,3,7],[0,7,4],[1,5,6],[1,6,2]],np.int32)
    path.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 12\nproperty list uchar int vertex_indices\nend_header\n'+
        vertices.astype('<f4').tobytes()+b''.join(struct.pack('<Biii',3,*face) for face in faces))
    return {'vertices':8,'faces':12,'bounds':[[-.5,-.5,-.5],[.5,.5,.5]],'signed_volume_diagnostic':-1.}


def fixture_arrays(index):
    h,w=SIZE;theta=np.pi-.03-.01*index
    R=np.asarray([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]],np.float64)
    center=np.array([0.,0.,3.]);K=np.asarray([[80,0,w/2],[0,81,h/2],[0,0,1.]],np.float64)
    # Analytical camera rays, explicitly expanded without the audit's solve or
    # pixel projection code. The front face of the inward cube is world z=.5.
    rays=[]
    for row in range(h):
        for col in range(w):
            x=(col+.5-w/2)/80;y=(row+.5-h/2)/81
            world=np.array([x,np.cos(theta)*y+np.sin(theta),-np.sin(theta)*y+np.cos(theta)])
            rays.append(world/np.sqrt(np.dot(world,world)))
    directions=np.asarray(rays);depth=(-2.5/directions[:,2]);n=h*w
    points=center+directions*depth[:,None];hit=np.ones(n,dtype=bool);hit[:3]=False
    depth[~hit]=10.;points[~hit]=np.nan
    raw=np.tile([0.,0.,-1.],(n,1));normal=-raw;raw[~hit]=np.nan;normal[~hit]=np.nan
    return {'origins':np.tile(center,(n,1)).astype(np.float32),'directions':directions.astype(np.float32),
        'positions':points.astype(np.float32),'raw_normals':raw.astype(np.float32),'normals':normal.astype(np.float32),
        'raw_depth':depth.astype(np.float32),'depth':depth[:,None].astype(np.float32),'hit':hit[:,None],
        'mask':hit.reshape(h,w).copy(),'K':K[None].astype(np.float32),
        'pose':np.concatenate([R,(-R@center)[:,None]],axis=1)[None].astype(np.float32),'rgb':np.full((1,3,h,w),.5,np.float32)}


def statistics(arrays):
    hit=arrays['hit'].reshape(-1);mask=arrays['mask'].reshape(-1)
    tp=int(np.count_nonzero(hit & mask));union=int(np.count_nonzero(hit|mask))
    return {'passed':True,'checks':{'original_negation':True,'unit_final_normals':True,'intersection_consistency':True,
            'front_facing_final_normals':True,'raw_normals_opposite_view':True,'silhouette_alignment':True},
            'rays':len(hit),'hits':int(hit.sum()),'misses':int((~hit).sum()),'mask_pixels':int(mask.sum()),
            'mask_intersection':tp,'mask_union':union,'silhouette_iou':tp/union if union else 0.}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.inputs=self.root/'original';self.inputs.mkdir();self.output=self.root/'result'
        self.extraction=self.inputs/'G.json';self.rays=self.inputs/'A.json';self.mesh=self.inputs/'mesh.ply'
        measured=fixture_mesh(self.mesh)
        self.artifacts={'source_manifest':{'path':str(ROOT/'scientific_sources/manifest-v1.json'),'sha256':audit.SOURCE_MANIFEST_SHA},
                        'mesh':{'path':str(self.mesh),'sha256':digest(self.mesh)}}
        self.g={'schema_version':1,'protocol':'bell-historical-best-G-mesh-r128-v1','status':'complete','phase':'complete',
            'run_id':'bell-g-best-mesh-2f4a25e1828f47109433f42a56d53f4a','scientific_commit':PINS['geometry'],
            'source':baseline('geometry',PINS['geometry'])[1],'source_manifest_sha256':audit.SOURCE_MANIFEST_SHA,
            'checkpoint_selection':{'sha256':audit.CHECKPOINT_SHA},'checkpoint_metadata':{'stored_step':150000},
            'strict_network_loaded':True,'network_unchanged':True,'research_geometry_admitted':False,'research_handoff_admitted':False,
            'network_comparison':{'result':'exact_structural_match','missing_keys':[],'unexpected_keys':[],'shape_mismatches':[],'dtype_mismatches':[]},
            'settings':{'resolution':128,'bbox_min':[-1,-1,-1],'bbox_max':[1,1,1],'isovalue':0,'outside_val':1.,'explicit_mesh_transforms':[]},
            'mesh':{'sha256':digest(self.mesh),'serialized':measured,'explicit_repairs_or_transforms':[],
                    'triangle_order_and_winding_preserved_in_export':True}}
        self.rows=[];self.arrays=[]
        ids=[str(i) for i in range(128)];random.Random(6033).shuffle(ids)
        self.a={'schema_version':1,'protocol':'historical-G-bell-to-A-rays-v1','status':'complete','phase':'ray_gates_passed',
            'scientific_commit':PINS['material'],'source':baseline('material',PINS['material'])[1],'mesh_identity':{'sha256':digest(self.mesh)},
            'diagnostic_only':True,'research_handoff_admitted':False,'ray_gate_passed':True,'optimizer_calls_completed':0,
            'splits':{'train':ids[6:],'validation':ids[5:6],'nvs':ids[:5]},'views':self.rows,'output_hashes':{}}
        for i,(role,camera) in enumerate(audit.VIEWS):
            path=self.inputs/(role+'.npz');arrays=fixture_arrays(i);np.savez_compressed(path,**arrays);self.arrays.append(arrays)
            self.rows.append({'role':role,'source_camera_id':camera,'resolution':list(SIZE),'artifact':'evaluations/'+role+'-'+camera+'.npz',
                              'sha256':digest(path),'statistics':statistics(arrays)})
        self.selection={'schema_version':1,'protocol':audit.PROTOCOL,'artifacts':self.artifacts,
                        'views':[{'role':role,'source_camera_id':camera,'path':str(self.inputs/(role+'.npz')),
                                  'sha256':digest(self.inputs/(role+'.npz'))} for role,camera in audit.VIEWS]}
        self.selector=self.root/'inputs.json';self.sync()
        self.addCleanup(patch.stopall)
        patch.object(audit,'IMAGE_SIZE',SIZE).start()
    def sync(self):
        self.extraction.write_text(json.dumps(self.g));g_sha=digest(self.extraction)
        self.a['extraction_report']={'sha256':g_sha}
        for row,selected in zip(self.rows,self.selection['views']):
            row['sha256']=selected['sha256']=digest(Path(selected['path']))
            self.a['output_hashes'][row['artifact']]=row['sha256']
        self.rays.write_text(json.dumps(self.a))
        self.artifacts.update(extraction={'path':str(self.extraction),'sha256':g_sha},rays={'path':str(self.rays),'sha256':digest(self.rays)})
        self.selector.write_text(json.dumps(self.selection))
    def run_audit(self):
        # Synthetic identities are patched only in unit fixtures, never exposed
        # as CLI options or admitted as actual historical/native evidence.
        with patch.object(audit,'ACCEPTED',{name:ref['sha256'] for name,ref in self.artifacts.items() if name!='source_manifest'}):
            return audit.audit(self.selector,digest(self.selector),self.output)
    def mutate(self,change):
        change(self.arrays[0]);np.savez_compressed(self.selection['views'][0]['path'],**self.arrays[0]);self.sync()
    def test_valid_independent_audit_is_deterministic_and_preserves_inputs_and_RNG(self):
        before={p:p.read_bytes() for p in self.inputs.iterdir()};rng=random.getstate();state=np.random.get_state()
        with patch.object(np,'load',wraps=np.load) as loader:
            result=self.run_audit()
        self.assertEqual(result['status'],'pass',result['blockers'])
        self.assertEqual(loader.call_count,3);self.assertTrue(all(c.kwargs.get('allow_pickle') is False for c in loader.call_args_list))
        self.assertEqual(result['categories']['scientific_geometry_accuracy']['status'],'unresolved')
        self.assertEqual(result['categories']['3A2_SDF_orientation']['status'],'unresolved')
        self.assertEqual(result['categories']['G_architecture_compatibility']['status'],'unresolved')
        self.assertFalse(result['research_handoff_admitted']);self.assertEqual(random.getstate(),rng)
        now=np.random.get_state();self.assertEqual(state[0],now[0]);np.testing.assert_array_equal(state[1],now[1]);self.assertEqual(state[2:],now[2:])
        self.assertEqual(before,{p:p.read_bytes() for p in self.inputs.iterdir()})
        first=result['views'][0]['checks'];self.assertLess(first['camera_ray_directions']['measured'],2e-6)
        self.assertEqual(first['silhouette_iou']['measured']['iou'],1.)
        self.output=self.root/'repeat';repeat=self.run_audit()
        self.assertEqual(result['checks'],repeat['checks']);self.assertEqual(result['views'],repeat['views'])
    def test_wrong_proper_extrinsic_rotation_is_detected_by_camera_ray_checks(self):
        self.mutate(lambda a:a['pose'].__setitem__((0,slice(None),slice(0,3)),a['pose'][0,:,:3].T.copy()))
        result=self.run_audit();self.assertEqual(result['status'],'fail')
        self.assertEqual(result['views'][0]['checks']['camera_ray_directions']['status'],'fail')
    def test_altered_intrinsics_are_detected_even_when_still_positive_and_invertible(self):
        self.mutate(lambda a:a['K'].__setitem__((0,0,0),100.))
        result=self.run_audit();self.assertEqual(result['views'][0]['checks']['intrinsic_structure']['status'],'pass')
        self.assertEqual(result['views'][0]['checks']['camera_ray_directions']['status'],'fail')
    def test_reversed_final_normals_fail_without_input_repair(self):
        self.mutate(lambda a:a['normals'].__imul__(-1))
        before=Path(self.selection['views'][0]['path']).read_bytes();result=self.run_audit()
        self.assertEqual(result['views'][0]['checks']['existing_A_normal_negation']['status'],'fail')
        self.assertEqual(result['views'][0]['checks']['final_normal_facing']['status'],'fail')
        self.assertEqual(Path(self.selection['views'][0]['path']).read_bytes(),before)
    def test_empty_hits_fail_and_leave_hit_statistics_unresolved(self):
        def change(a):a['hit'][:]=False;a['depth'][:]=10;a['raw_depth'][:]=10
        self.mutate(change);self.rows[0]['statistics']=statistics(self.arrays[0]);self.sync()
        result=self.run_audit();checks=result['views'][0]['checks']
        self.assertEqual(result['status'],'fail');self.assertEqual(checks['nonempty_intersections']['status'],'fail')
        self.assertEqual(checks['hit_geometry']['status'],'unresolved')
    def test_incorrect_artifact_hash_fails_closed_and_writes_failure_report(self):
        path=Path(self.selection['views'][0]['path']);path.write_bytes(path.read_bytes()+b'changed')
        result=self.run_audit();self.assertEqual(result['status'],'fail');self.assertTrue(result['blockers'])
        self.assertIn('SHA-256',str(result['blockers']));self.assertTrue((self.output/'audit.json').is_file())
    def test_incompatible_camera_ids_cannot_be_substituted_for_iterator_zero(self):
        self.selection['views'][0]['source_camera_id']='0';self.selector.write_text(json.dumps(self.selection))
        result=self.run_audit();self.assertEqual(result['status'],'fail');self.assertFalse(result['views'])
    def test_incomplete_extraction_or_view_evidence_rejected(self):
        del self.g['checkpoint_metadata'];self.sync();result=self.run_audit()
        self.assertEqual(result['status'],'fail');self.assertFalse(result['views'])
    def test_object_NPZ_is_rejected_without_pickle_loading(self):
        self.mutate(lambda a:a.__setitem__('K',np.asarray([object()],dtype=object)))
        with patch('pickle.load',side_effect=AssertionError('Never deserialize')),patch('pickle.loads',side_effect=AssertionError('Never deserialize')):
            result=self.run_audit()
        self.assertEqual(result['status'],'fail');self.assertEqual(result['views'][0]['checks']['array_contract']['status'],'fail')
    def test_missing_or_extra_NPZ_members_fail_closed(self):
        self.mutate(lambda a:a.pop('pose'));result=self.run_audit()
        self.assertEqual(result['status'],'fail');self.assertIn('NPZ content',str(result['blockers']))
        self.arrays[0]=fixture_arrays(0);self.output=self.root/'extra-result'
        self.mutate(lambda a:a.__setitem__('extra',np.ones(1,np.float32)));result=self.run_audit()
        self.assertEqual(result['status'],'fail');self.assertIn('NPZ content',str(result['blockers']))
    def test_nonfinite_camera_and_bad_numerical_shapes_fail(self):
        self.mutate(lambda a:a['K'].__setitem__((0,0,0),np.nan));result=self.run_audit()
        self.assertEqual(result['status'],'fail');self.assertIn('Nonfinite',str(result['blockers']))
        self.arrays[0]=fixture_arrays(0);self.output=self.root/'bad-shape-result'
        self.mutate(lambda a:a.__setitem__('directions',a['directions'][:,:2]));result=self.run_audit()
        self.assertEqual(result['status'],'fail');self.assertIn('shape/dtype header',str(result['blockers']))
    def test_nonzip_and_standalone_NPY_are_not_treated_as_NPZ(self):
        path=Path(self.selection['views'][0]['path']);path.write_bytes(b'invalid NPZ content');self.sync()
        self.assertEqual(self.run_audit()['status'],'fail')
        self.output=self.root/'standalone-result'
        with path.open('wb') as stream:np.save(stream,np.ones(2),allow_pickle=False)
        self.sync();result=self.run_audit()
        self.assertEqual(result['status'],'fail');self.assertIn('Expected numerical NPZ',str(result['blockers']))
    def test_intersection_and_mask_errors_are_independently_reported(self):
        def change(a):a['positions'][:,0]+=.01;a['mask'][:]=~a['mask']
        self.mutate(change);self.rows[0]['statistics']=statistics(self.arrays[0]);self.sync()
        result=self.run_audit();checks=result['views'][0]['checks']
        self.assertEqual(checks['intersection_saved_unit_ray']['status'],'fail')
        self.assertEqual(checks['intersection_calibrated_ray']['status'],'fail')
        self.assertEqual(checks['silhouette_iou']['status'],'fail')
    def test_output_collision_aliases_and_input_overlap_are_refused(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):self.run_audit()
        self.output=self.inputs/'unwanted'
        with self.assertRaises(ValueError):self.run_audit()
        alias=self.root/'alias';alias.symlink_to(self.inputs,target_is_directory=True);self.output=alias/'unwanted'
        with self.assertRaises(ValueError):self.run_audit()
        self.assertFalse((self.inputs/'unwanted').exists())
    def test_selection_hash_and_threshold_override_fail_before_allocation(self):
        with self.assertRaises(ValueError):audit.audit(self.selector,'0'*64,self.output)
        with self.assertRaises(ValueError):audit.audit(Path(os.path.relpath(self.selector,Path.cwd())),digest(self.selector),self.output)
        self.selection['tolerances']={};self.selector.write_text(json.dumps(self.selection))
        with self.assertRaises(ValueError):self.run_audit()
        self.assertFalse(self.output.exists())
    def test_no_scientific_import_or_existing_summary_reference_in_CLI(self):
        # Run an actual CLI with synthetic metadata and denied model imports;
        # only fixture hashes and small image size differ from native admission.
        script='''
import builtins,json,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 assert name.split('.')[0] not in {'torch','network','dataset','train','open3d','nvdiffrast','raytracing'},name
 assert name!='infrastructure.bell_material_rays','Existing summarize is not an independent reference'
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from infrastructure import geometry_scientific_audit as a
from pathlib import Path
a.IMAGE_SIZE=(6,8)
selection=json.loads(Path(sys.argv[1]).read_text())
a.ACCEPTED={k:v['sha256'] for k,v in selection['artifacts'].items() if k!='source_manifest'}
from audit_geometry_scientific import main
raise SystemExit(main(['--inputs',sys.argv[1],'--inputs-sha256',sys.argv[2],'--output-dir',sys.argv[3]]))
'''
        result=subprocess.run([sys.executable,'-B','-c',script,str(self.selector),digest(self.selector),str(self.output)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('unresolved',result.stdout)


if __name__=='__main__':unittest.main()
