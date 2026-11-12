"""Focused C.1 synthetic fixtures: zero real shader calls or optimizer steps."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import complete_neural_material_validation_v1 as c
from tests.scientific.test_neural_material_validation_v1 import FakeShader
from infrastructure.signatures import SignatureError


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference,_=c.v.admit();cls.fit,_,_=c.v.h.inputs(cls.reference)

    def fixture(self):
        model=c.v.field.make_material('N0',c.v.h.fresh_latent(101),101)
        batch=c.v.h.MaterialBatch(self.reference['scene'],self.reference['scene']['observations'][:24],self.fit)
        target=torch.full((24,3),.9)
        arrays,info=c.v.capture(model,batch,FakeShader(),target,c.Budget(),'synthetic-test',[])
        inputs=dict(normals=batch.base.normals.numpy(),views=batch.base.views.numpy())
        return model,arrays,info,target,inputs

    def test_full_replay_row_assembly_with_real_capture_metadata(self):
        model,arrays,info,target,_=self.fixture()
        checks={key:c.v.comparison(value,value) for key,value in arrays.items()}
        original=copy.deepcopy(info)
        with tempfile.TemporaryDirectory() as root:
            with c.v.h.OwnedOutput(Path(root),'fixture') as output:
                output.path('raw').mkdir();raw=c.v.save_arrays(output,'fixture',arrays)
                row=c.v.replay_row('M1',101,'N0',raw,checks,info)
                output.persist(row)
                loaded=json.loads(output.path('report.json').read_text())
                self.assertEqual(loaded['material_name'],'M1');self.assertEqual(loaded['seed'],101)
                self.assertEqual(loaded['material'],info['material']);self.assertTrue(loaded['passed'])
                self.assertEqual(loaded['raw']['sha256'],raw['sha256'])
        self.assertEqual(info,original)

    def test_replay_collision_guard_and_failed_check_are_preserved(self):
        row=c.v.replay_row('M2',101,'D0',{}, {'x':{'passed':False}}, {'material':{'physical':'fixture'}})
        self.assertFalse(row['passed']);self.assertEqual(row['material_name'],'M2')
        with self.assertRaises(SignatureError):c.v.replay_row('M2',101,'D0',{}, {}, {'material_name':'spoofed'})

    def test_only_metadata_helper_and_one_call_site_changed(self):
        result=c.verify_reporting_patch();self.assertTrue(result['metadata_only'])
        self.assertTrue(result['helper_and_one_call_site_only']);self.assertEqual(len(result['similar_constructions_reviewed']),3)

    def test_reporting_patch_gate_rejects_a_scientific_change(self):
        real=Path.read_text;path=c.ROOT/'experiments/validate_neural_material_recovery_v1.py'
        def altered(p,*args,**kwargs):
            text=real(p,*args,**kwargs)
            return text.replace("2e-6 if key=='half_vectors'","9e-6 if key=='half_vectors'") if p==path else text
        with patch.object(Path,'read_text',new=altered):
            with self.assertRaises(SignatureError):c.verify_reporting_patch()

    def test_schedule_skips_all_completed_initials_and_first_replay(self):
        self.assertEqual(c.replay_schedule(),[('M1','N0'),('M1','N1'),('M2','D0'),('M2','N0'),('M2','N1'),('M3','D0'),('M3','N0'),('M3','N1')])
        p=c.validate_plan(c.plan());self.assertEqual(p['repeated_initial_renders'],0)
        self.assertEqual(p['caps']['shader_forward'],17);self.assertEqual(p['caps']['shader_backward'],8)
        self.assertEqual(p['caps']['field_vjp'],3);self.assertEqual(p['caps']['synthetic_Adam'],3)
        self.assertEqual(p['wall_seconds'],289)
        self.assertTrue(all(c.PRIOR_COUNTS[k]+c.CAPS[k]==c.v.CAPS[k] for k in c.CAPS))

    def test_remaining_caps_fail_closed_without_retry(self):
        b=c.Budget()
        with self.assertRaises(RuntimeError):
            with b.operation('shader_forward','fixture'):raise RuntimeError('fixture')
        self.assertEqual(b.counts['shader_forward'],{'attempted':1,'completed':0})
        with self.assertRaises(SignatureError):
            with b.operation('shader_forward','fixture'):pass
        b.counts['shader_forward']['attempted']=17
        with self.assertRaises(SignatureError):
            with b.operation('shader_forward','extra'):pass
        b.start-=290
        with self.assertRaises(c.v.Stopped):b.check()

    def test_independent_saved_tensor_checks_ignore_success_flags(self):
        model,arrays,info,target,inputs=self.fixture()
        self.assertTrue(c.recheck_capture(arrays,'N0',c.v.field.identity(model),target,inputs)['passed'])
        for key in ('mask','mean','gradient::latent','encoding'):
            altered={k:a.copy() for k,a in arrays.items()}
            if key=='mask':altered[key].flat[0]=False
            elif key=='gradient::latent':altered[key].fill(0)
            else:altered[key].flat[0]+=.01
            with self.assertRaises(SignatureError):c.recheck_capture(altered,'N0',c.v.field.identity(model),target,inputs)

    def test_source_topology_catalogue_matches_original_parameter_count(self):
        shapes=c.parameter_shapes('N1');self.assertEqual(len(shapes),84)
        self.assertEqual(sum(int(np.prod(s)) for s in shapes.values()),1485584)
        self.assertEqual(shapes['feats_network.module0.0.weight_v'],[256,51])
        self.assertEqual(shapes['feats_network.module1.0.weight_v'],[256,307])
        self.assertEqual(shapes['rotation_predictor.predictor.6.bias'],[2])
        self.assertEqual(c.parameter_shapes('D0'),{'latent':[7]})

    def test_old_archive_hash_gate_precedes_no_pickle_load(self):
        with tempfile.TemporaryDirectory() as root:
            with c.v.h.OwnedOutput(Path(root),'fixture') as output:
                output.path('raw').mkdir();a={'x':np.arange(4,dtype=np.float32)}
                manifest=c.v.save_arrays(output,'fixture',a);e=c.Evidence(Path(root))
                with patch.object(np,'load',wraps=np.load) as load:loaded=e.arrays(manifest)
                self.assertIs(load.call_args.kwargs['allow_pickle'],False);self.assertTrue(np.array_equal(loaded['x'],a['x']))
                manifest['sha256']='0'*64
                with patch.object(np,'load') as load:
                    with self.assertRaises(SignatureError):e.arrays(manifest)
                    load.assert_not_called()


if __name__=='__main__':unittest.main()
