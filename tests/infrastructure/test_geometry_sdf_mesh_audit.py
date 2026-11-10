"""New bounded analytical tests only: no historical numerical/model execution."""
from copy import deepcopy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from infrastructure import geometry_sdf_mesh_audit as audit
from infrastructure.signatures import SignatureError
from infrastructure.stage_sources import ROOT, PINS, digest, baseline

try:
    import torch
except ImportError:
    torch=None


def plane_mesh():
    vertices=np.asarray([[-.2,-.2,0],[0,.2,0],[.2,-.2,0]],np.float64)
    return vertices,np.array([[0,1,2]],np.int64)  # Original cross normal is -z.


def plan(points,faces=0,normal=None,rays=None):
    points=np.asarray(points,np.float32).reshape(-1,3);rays=deepcopy(rays or [])
    vertices=len(points)-faces-len(rays)
    return {'points':points,'face_count':faces,'vertex_count':vertices,'rays':rays,
        'face_indices':np.arange(faces),'vertex_indices':np.arange(vertices),
        'face_normals':np.tile(normal or [0,0,-1.],(faces,1)),
        'face_areas':np.full(faces,.08),'sample_kind':np.asarray([0]*faces+[1]*vertices+[2]*len(rays),np.int32),
        'silhouettes':[],'mesh_population':{},'float32_point_rounding_maximum':0.}


def control(normal=(0,0,1.)):
    v,f=plane_mesh()
    row={'camera_id':'14','ray_index':4,'pixel_row':1,'pixel_column':1,'anomalous':False,
        'sample_index':1,'position':[0.,0.,0.],'origin':[0.,0.,2.],'direction':[0.,0.,-1.],
        'saved_depth':2.,'raw_normal':(-np.asarray(normal)).tolist(),'final_normal':list(normal),'viewing_cosine':float(normal[2])}
    row['CPU']=audit.ray_comparison(v,f,row)
    return row


@unittest.skipIf(torch is None,'CPU Torch unavailable for analytical derivatives')
class AnalyticalTests(unittest.TestCase):
    def measure(self,p,sdf):
        query=audit.TorchQueries(sdf,torch,device='cpu');result,numeric=audit.evaluate_samples(p,query)
        self.assertEqual(query.points,result['point_evaluations']);self.assertLessEqual(query.points,20000)
        json.dumps(result,allow_nan=False)
        self.assertEqual(result['statuses']['historical_checkpoint_provenance']['status'],'unresolved')
        self.assertEqual(result['statuses']['research_geometry_admission']['status'],'unresolved')
        return result,numeric
    def test_plane_zero_level_and_reversed_winding_preserve_both_signed_relations(self):
        p=plan([[0,0,0],[.1,0,0]],faces=1)
        first,n=self.measure(p,lambda x:x[:,2:3]);self.assertEqual(first['statuses']['sdf_mesh_zero_level_consistency']['status'],'pass')
        face=first['statuses']['face_winding_vs_sdf_orientation'];self.assertEqual(face['negative_cosines'],1)
        self.assertEqual(face['angle_degrees']['median'],180.);self.assertEqual(face['negated_face_angle_degrees']['median'],0.)
        p['face_normals']*=-1
        second,_=self.measure(p,lambda x:x[:,2:3]);self.assertEqual(second['statuses']['face_winding_vs_sdf_orientation']['positive_cosines'],1)
        np.testing.assert_array_equal(n['SDF_raw'],0)
    def test_sphere_and_nonunit_gradient_distance_proxy(self):
        p=plan([[.5,0,0],[0,-.5,0],[0,0,.5]])
        r,n=self.measure(p,lambda x:3*(torch.linalg.norm(x,dim=-1,keepdim=True)-.5))
        np.testing.assert_allclose(n['gradient_length'],3);np.testing.assert_array_equal(n['distance_proxy'],0)
        self.assertEqual(r['sign_probes']['gradient']['0.5']['sign_changes'],3)
        self.assertFalse(r['statuses']['sdf_gradient_quality']['unit_magnitude_assumed'])
    def test_near_zero_gradients_are_excluded_without_infinite_JSON_or_fabricated_direction(self):
        r,n=self.measure(plan([[0,0,0]],faces=1),lambda x:x[:,2:3]*1e-8)
        self.assertEqual(r['statuses']['sdf_gradient_quality']['status'],'unresolved')
        self.assertEqual(r['statuses']['sdf_mesh_zero_level_consistency']['status'],'unresolved')
        self.assertTrue(np.isnan(n['distance_proxy']).all());self.assertEqual(r['groups']['face_centroids']['near_zero_gradients'],1)
    def test_outside_raw_SDF_is_distinct_from_actual_extraction_mask(self):
        r,n=self.measure(plan([[1.01,0,0],[.999,0,0]]),lambda x:x[:,0:1]-1.)
        self.assertFalse(n['inside_sphere'][0]);self.assertAlmostEqual(float(n['SDF_raw'][0]),.01,places=6)
        self.assertEqual(float(n['base_masked_extraction_field'][0]),1.)
        self.assertEqual(r['groups']['vertices']['outside_sphere'],1)
    def test_camera_exterior_polarity_is_measured_for_both_SDF_signs(self):
        for sign in (1,-1):
            p=plan([[.1,0,0],[0,0,0]],rays=[control()])
            r,n=self.measure(p,lambda x:sign*x[:,2:3]);row=r['rays'][0]
            self.assertEqual(row['measured_camera_exterior_SDF_sign'],sign)
            self.assertEqual(row['gradient_vs_A_final_signed_cosine'],sign)
            self.assertEqual(row['A_alignment_with_exterior_signed_gradient'],1.)
            self.assertEqual(r['statuses']['A_shading_normal_consistency']['status'],'pass')
    def test_view_facing_but_bad_shading_normal_fails_without_normal_repair(self):
        normal=[float(np.sin(np.deg2rad(80))),0.,float(np.cos(np.deg2rad(80)))]
        r,_=self.measure(plan([[.1,0,0],[0,0,0]],rays=[control(normal)]),lambda x:x[:,2:3])
        self.assertEqual(r['statuses']['A_shading_normal_consistency']['status'],'fail')
        np.testing.assert_array_equal(r['rays'][0]['final_normal'],normal)
        self.assertAlmostEqual(r['statuses']['A_shading_normal_consistency']['gradient_angle_degrees']['median'],80.)
    def test_first_derivatives_fixed_float32_batches_budget_and_no_parameter_grad(self):
        weight=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float32));before=weight.detach().clone();seen=[]
        def sdf(x):seen.append((len(x),x.dtype,x.requires_grad));return x[:,2:3]*weight
        q=audit.TorchQueries(sdf,torch,device='cpu')
        with patch.object(torch.autograd,'grad',wraps=torch.autograd.grad) as derivative:
            q(np.zeros((257,3)),True)
        self.assertEqual([x[0] for x in seen],[128,128,1]);self.assertTrue(all(x[1]==torch.float32 for x in seen))
        self.assertTrue(all(c.kwargs['create_graph'] is False and c.kwargs['retain_graph'] is False for c in derivative.call_args_list))
        self.assertIsNone(weight.grad);self.assertTrue(torch.equal(before,weight))
        q.points=20000
        with self.assertRaises(SignatureError):q(np.zeros((1,3)))
        self.assertEqual(len(seen),3)
    def test_nonfinite_values_or_derivatives_fail_closed(self):
        q=audit.TorchQueries(lambda x:x[:,2:3]*float('nan'),torch,device='cpu')
        with self.assertRaises(SignatureError):q(np.zeros((1,3)),True)
        q=audit.TorchQueries(lambda x:torch.sqrt(torch.abs(x[:,2:3])),torch,device='cpu')
        with self.assertRaises(SignatureError):q(np.zeros((1,3)),True)
    def test_no_eligible_offset_direction_is_handled_without_hidden_queries(self):
        r,n=self.measure(plan([[0,0,0]]),lambda x:x[:,2:3]*0)
        self.assertEqual(len(n['probe_points']),0);self.assertEqual(r['point_evaluations'],1)


class GeometryTests(unittest.TestCase):
    def test_CPU_first_and_later_hit_anomaly_observations(self):
        v,f=plane_mesh();v=np.concatenate([v,v+[0,0,-1]]);f=np.concatenate([f,f+3])
        row=control();r=audit.ray_comparison(v,f,row)
        self.assertEqual(r['first']['distance'],2.);self.assertTrue(r['geometric_exterior_eligible'])
        row['saved_depth']=3.;row['position']=[0,0,-1.];r=audit.ray_comparison(v,f,row)
        self.assertEqual(r['classification'],'supported_by_evidence');self.assertIn('later',r['basis'])
        self.assertFalse(r['first_hit_match']);self.assertEqual(r['closest_saved_depth']['face_index'],1)
    def test_CPU_edge_tiny_triangle_and_first_hit_ties_are_ambiguous(self):
        v,f=plane_mesh();row=control();row['origin']=[0,.2,2];row['position']=[0,.2,0]
        self.assertEqual(audit.ray_comparison(v,f,row)['classification'],'ambiguous')
        row=control();small=v*1e-5
        self.assertTrue(audit.cpu_intersections(small,f,np.array([0,0,2.]),np.array([0,0,-1.]))[0]['tiny_triangle'])
        self.assertEqual(audit.ray_comparison(small,f,row)['classification'],'ambiguous')
        self.assertEqual(audit.ray_comparison(v,np.concatenate([f,f]),row)['first_tie_count'],2)
    def test_CPU_mismatched_raw_normal_or_empty_intersections_not_repaired(self):
        v,f=plane_mesh();row=control();row['raw_normal']=[1.,0,0]
        r=audit.ray_comparison(v,f,row);self.assertIn('normal disagrees',r['basis']);self.assertEqual(row['raw_normal'],[1.,0,0])
        row['origin']=[5.,0,2.];r=audit.ray_comparison(v,f,row)
        self.assertEqual(r['classification'],'unresolved');self.assertEqual(r['positive_intersections'],0)
    def test_sampling_preserves_all_3_anomalies_inputs_and_RNG(self):
        v,f=plane_mesh();views=[]
        for camera,count in [('14',0),('58',2),('11',1)]:
            n=48;normal=np.tile([0.,0,1],(n,1));normal[:count]*=-1
            views.append((camera,{'origins':np.tile([0.,0,2],(n,1)),'directions':np.tile([0.,0,-1],(n,1)),
                'positions':np.zeros((n,3)),'normals':normal,'raw_normals':-normal,
                'depth':np.full((n,1),2.),'hit':np.ones((n,1),bool),'mask':np.ones((6,8),bool)}))
        originals=deepcopy(views);rng=random.getstate();np_rng=np.random.get_state()
        p=audit.samples(v,f,views);q=audit.samples(v,f,views)
        bad=[(r['camera_id'],r['ray_index']) for r in p['rays'] if r['anomalous']]
        self.assertEqual(bad,[('58',0),('58',1),('11',0)]);self.assertEqual(len(p['rays']),99)
        self.assertEqual(p['rays'],q['rays']);self.assertEqual(random.getstate(),rng)
        now=np.random.get_state();np.testing.assert_array_equal(np_rng[1],now[1]);self.assertEqual(np_rng[2:],now[2:])
        for (_,before),(_,after) in zip(originals,views):
            for key in before:np.testing.assert_array_equal(before[key],after[key])
        views[0][1]['normals'][0]*=-1
        with self.assertRaises(SignatureError):audit.samples(v,f,views)
    def test_area_radius_stratification_includes_extrema_and_fixed_budget(self):
        centers=np.column_stack([np.linspace(0,.999,5000),np.zeros((5000,2))]);areas=np.linspace(.001,.01,5000)
        first=audit.select_faces(centers,areas);np.testing.assert_array_equal(first,audit.select_faces(centers,areas))
        self.assertEqual(len(first),1920);self.assertIn(0,first);self.assertIn(4999,first)
        self.assertEqual(len(set(first)),1920);self.assertEqual((1920+256+99)*5+4*1920+6*99,19649)
    def test_boundary_bands_separate_deep_disagreements_and_empty_reference(self):
        mask=np.zeros((21,21),bool);mask[5:16,5:16]=True;hit=mask.copy()
        hit[5,10]=False;hit[10,10]=False;hit[4,10]=True;hit[0,10]=True
        r=audit.silhouette_disagreements(hit,mask)
        self.assertEqual((r['false_positive'],r['false_negative']),(2,2))
        self.assertEqual(r['within_pixels']['1'],{'false_positive':1,'false_negative':1})
        self.assertEqual(r['deeper_than_4'],{'false_positive':1,'false_negative':1})
        empty=audit.silhouette_disagreements(np.ones((3,3),bool),np.zeros((3,3),bool))
        self.assertEqual(empty['boundary_pixels'],0);self.assertEqual(empty['deeper_than_4']['false_positive'],9)
    def test_original_cross_normals_and_degenerate_mesh_rejection(self):
        v,f=plane_mesh();_,norm,area=audit.face_geometry(v,f)
        np.testing.assert_array_equal(norm,[[0,0,-1]]);self.assertAlmostEqual(area[0],.08)
        with self.assertRaises(SignatureError):audit.face_geometry(v,np.array([[0,0,1]]))
    def test_original_pinned_scientific_source_closures_remain_exact(self):
        for stage in ('geometry','material'):
            path,item=baseline(stage,PINS[stage])
            self.assertEqual(path,ROOT/item['root'])
            self.assertTrue(item['files'])  # baseline verifies every recorded hash.
    def test_profile_observer_restores_owned_state_on_failure_and_unwraps_shared_decorators(self):
        def wrapper(method):
            from functools import wraps
            @wraps(method)
            def shared(*args,**kwargs):return method(*args,**kwargs)
            return shared
        def forbidden():pass
        def permitted():return 5
        original=forbidden;forbidden=wrapper(forbidden);permitted=wrapper(permitted)
        fake_renderer=SimpleNamespace(**{name:forbidden for name in ('forward','_init_dataset','train_step','test_step','nvs')})
        def sdf(x):return x
        def grad(*args,create_graph=False):return args
        fake_torch=SimpleNamespace(optim=SimpleNamespace(Adam=SimpleNamespace(__init__=forbidden,step=forbidden),
            Optimizer=SimpleNamespace(load_state_dict=forbidden)),Tensor=SimpleNamespace(backward=forbidden),save=forbidden,
            autograd=SimpleNamespace(grad=grad))
        modules={'network.field':SimpleNamespace(extract_geometry=forbidden,extract_fields=forbidden),
            'dataset.database':SimpleNamespace(GlossySyntheticDatabase=SimpleNamespace(__init__=forbidden)),
            'utils.base_utils':SimpleNamespace(read_pickle=forbidden,save_pickle=forbidden)}
        with patch.dict(sys.modules,modules):
            observer=audit.NativeCalls(fake_renderer,SimpleNamespace(gradient=forbidden,sdf=sdf),fake_torch)
        self.assertIn(original.__code__,observer.forbidden);self.assertNotIn(permitted.__code__,observer.forbidden)
        with self.assertRaises(SignatureError):
            with observer:
                self.assertEqual(permitted(),5)
                forbidden()
        self.assertIsNone(sys.getprofile());self.assertIsNone(sys.gettrace())


# Reuse ONLY disposable fixture builders from 3A-1, never execute its audit/tests.
_spec=importlib.util.spec_from_file_location('coordinate_fixture_builders',ROOT/'tests/infrastructure/test_geometry_scientific_audit.py')
fixtures=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(fixtures)


class AdmissionTests(unittest.TestCase):
    sync=fixtures.AuditTests.sync
    def setUp(self):
        fixtures.AuditTests.setUp(self)
        self.checkpoint=self.inputs/'opaque-checkpoint.pth';self.checkpoint.write_bytes(b'opaque fixture; never deserialize')
        self.g['checkpoint_selection']['sha256']=digest(self.checkpoint);self.sync()
        self.addCleanup(patch.stopall)
        patch.object(audit.evidence,'CHECKPOINT_SHA',digest(self.checkpoint)).start()
        patch.object(audit.evidence,'ACCEPTED',{k:v['sha256'] for k,v in self.artifacts.items() if k!='source_manifest'}).start()
        prior={'schema_version':1,'protocol':audit.evidence.PROTOCOL,'status':'pass','blockers':[],
            'research_handoff_admitted':False,'tolerances':audit.evidence.TOLERANCES,
            'categories':{'saved_evidence_coordinate_consistency':{'status':'pass'}},'artifacts':deepcopy(self.selection),
            'views':[{'role':role,'source_camera_id':camera,'checks':{'final_normal_facing':{'status':'pass',
                'measured':{'back_facing_count':audit.EXPECTED_BACK[camera]}}}} for role,camera in audit.evidence.VIEWS],
            'implementation':{n:digest(ROOT/n) for n in ['infrastructure/geometry_scientific_audit.py','audit_geometry_scientific.py',
                'infrastructure/signatures.py','infrastructure/stage_sources.py','infrastructure/paths.py','docs/GEOMETRY_SCIENTIFIC_AUDIT_V1.md']},
            'source_files':{stage:{str(ROOT/item['root']/n):record['sha256'] for n,record in item['files'].items()}
                for stage,item in [(s,baseline(s,PINS[s])[1]) for s in ['geometry','material']]}}
        self.prior=prior;self.prior_path=self.inputs/'coordinate.json';self.prior_path.write_text(json.dumps(prior))
        patch.object(audit,'COORDINATE_AUDIT_SHA',digest(self.prior_path)).start()
        self.new={'schema_version':1,'protocol':audit.PROTOCOL,'artifacts':{**self.artifacts,
            'checkpoint':{'path':str(self.checkpoint),'sha256':digest(self.checkpoint)},
            'coordinate_audit':{'path':str(self.prior_path),'sha256':digest(self.prior_path)}},'views':deepcopy(self.selection['views'])}
        selector_dir=self.root/'selection';selector_dir.mkdir();self.selector=selector_dir/'inputs.json';self.save_selection()
    def save_selection(self):self.selector.write_text(json.dumps(self.new))
    def save_prior(self):
        self.prior_path.write_text(json.dumps(self.prior));self.new['artifacts']['coordinate_audit']['sha256']=digest(self.prior_path)
        patch.object(audit,'COORDINATE_AUDIT_SHA',digest(self.prior_path)).start();self.save_selection()
    def prepare(self):return audit.prepare(self.selector,digest(self.selector),self.output)
    def test_metadata_admission_does_not_load_arrays_models_checkpoint_or_repeat_prior_audit(self):
        with patch.object(audit.evidence,'load_arrays',side_effect=AssertionError('No array load')), \
             patch.object(audit.evidence,'audit',side_effect=AssertionError('No old audit')), \
             patch.object(audit.evidence,'check_view',side_effect=AssertionError('No old numerical checks')), \
             patch.object(audit,'native',side_effect=AssertionError('No native execution')):
            p=self.prepare()
        self.assertEqual(p['cfg']['database_name'],'syn/bell');self.assertFalse(self.output.exists())
        self.assertEqual(self.checkpoint.read_bytes(),b'opaque fixture; never deserialize')
    def test_bad_checkpoint_prior_or_selected_array_hash_rejected_before_allocation(self):
        for name in ('checkpoint','coordinate_audit'):
            original=self.new['artifacts'][name]['sha256'];self.new['artifacts'][name]['sha256']='0'*64;self.save_selection()
            with self.assertRaises(SignatureError):self.prepare()
            self.new['artifacts'][name]['sha256']=original
        self.new['views'][0]['sha256']='1'*64;self.save_selection()
        with self.assertRaises(SignatureError):self.prepare()
        self.assertFalse(self.output.exists())
    def test_numerical_archive_reader_is_pickle_free_and_refuses_object_payloads(self):
        ref={k:self.new['views'][0][k] for k in ('path','sha256')}
        with patch.object(np,'load',wraps=np.load) as loader:
            arrays=audit.evidence.load_arrays(ref)
        self.assertTrue(loader.call_args.kwargs['allow_pickle'] is False)
        self.assertTrue(all(not a.flags.writeable for a in arrays.values()))
        bad=fixtures.fixture_arrays(0);bad['K']=np.asarray([object()],dtype=object)
        path=self.inputs/'invalid.npz';np.savez(path,**bad)
        with patch('pickle.load',side_effect=AssertionError('Never deserialize')),patch('pickle.loads',side_effect=AssertionError('Never deserialize')):
            with self.assertRaises((SignatureError,ValueError)):audit.evidence.load_arrays({'path':str(path),'sha256':digest(path)})
    def test_incomplete_prior_and_source_closure_fail_closed(self):
        original=deepcopy(self.prior)
        for change in [lambda p:p.update(status='fail'),lambda p:p['artifacts'].update(views=[]),
                       lambda p:p['source_files']['geometry'].clear(),lambda p:p['views'][1]['checks']['final_normal_facing']['measured'].update(back_facing_count=0)]:
            self.prior=deepcopy(original);change(self.prior);self.save_prior()
            with self.assertRaises((SignatureError,KeyError)):self.prepare()
        self.assertFalse(self.output.exists())
    def test_explicit_trust_and_isolation_required_before_activation_or_outputs(self):
        p=self.prepare()
        with patch.object(audit,'activate',side_effect=AssertionError('Must not activate')):
            with self.assertRaises(SignatureError):audit.execute(p,'new-audit','unused','0'*40,trusted=False)
            with patch.object(audit,'runtime_guards',side_effect=SignatureError('No isolation')):
                with self.assertRaises(SignatureError):audit.execute(p,'new-audit','unused','0'*40,trusted=True)
        self.assertFalse(self.output.exists())
    def test_output_containment_collision_and_source_pin_rejection(self):
        with self.assertRaises((SignatureError,ValueError)):audit.prepare(self.selector,digest(self.selector),self.inputs/'out')
        with self.assertRaises(SignatureError):audit.prepare(self.selector,digest(self.selector),self.output,g_commit='0'*40)
        p=self.prepare()
        from infrastructure.runs import RunStore
        run=RunStore(p['resolver'],layout='mounted').create('audit-fixture')
        with self.assertRaises((ValueError,FileExistsError)):RunStore(p['resolver'],layout='mounted').create('audit-fixture')
        self.assertTrue(run.root.exists());self.assertEqual(self.checkpoint.read_bytes(),b'opaque fixture; never deserialize')
    def test_native_failure_retains_failed_report_and_exits_owned_environment(self):
        p=self.prepare();cleaned=[]
        @contextmanager
        def environment(run):
            import os
            previous=Path.cwd()
            try:yield
            finally:os.chdir(previous);cleaned.append(True)
        with patch.object(audit,'runtime_guards'),patch.object(audit.os,'statvfs',return_value=SimpleNamespace(f_flag=audit.os.ST_RDONLY)), \
             patch.object(audit,'activate'),patch.object(audit.evidence,'load_arrays',return_value={}), \
             patch.object(audit,'samples',return_value=plan([[0,0,0]])), \
             patch('infrastructure.bell_inference.runtime_environment',environment), \
             patch.object(audit,'native',side_effect=SignatureError('Synthetic native failure')),patch.object(audit.traceback,'print_exc'):
            with self.assertRaises(SignatureError):audit.execute(p,'failure-fixture',audit.IMAGE,'0'*40,trusted=True)
        report=json.loads((self.output/'sdf-mesh-audit.json').read_text())
        self.assertEqual(report['status'],'failed');self.assertIn('Synthetic native failure',report['error'])
        self.assertEqual(cleaned,[True]);self.assertIsNone(sys.getprofile())
        self.assertFalse(list((self.output/'checkpoints').iterdir()));self.assertFalse(report['optimizer_state_restored'])
        self.assertEqual(report['statuses']['historical_checkpoint_provenance']['status'],'unresolved')


if __name__=='__main__':unittest.main()
