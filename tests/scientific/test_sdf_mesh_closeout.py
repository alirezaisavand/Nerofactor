"""Disposable numerical evidence only; no repeat of original 3A-2 execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from audits import sdf_mesh_closeout as audit
from infrastructure.signatures import SignatureError


def fixture():
    p=np.array([[0,0,0],[.99,0,0],[.3,0,0]],np.float32);g=np.tile([0.,0,1.],(3,1)).astype(np.float32)
    y=np.array([0,.02,.0001],np.float32);indices=[];offsets=[];values=[];points=[]
    for i in range(3):
        for f in [.25,-.25,.5,-.5]:
            indices.append(i);offsets.append(f);points.append(p[i]+audit.H*f*g[i])
            values.append(y[i]+audit.H*f if i<2 else {.25:.001,-.25:-.004,.5:-.003,-.5:-.008}[f])
    return {'points':p,'SDF_raw':y,'base_masked_extraction_field':y.copy(),'gradient':g,
        'gradient_length':np.ones(3),'distance_proxy':abs(y).astype(np.float64),'inside_sphere':np.ones(3,bool),
        'float64_radius':np.linalg.norm(p.astype(np.float64),axis=1),'sample_kind':np.arange(3,dtype=np.int32),
        'probe_points':np.array(points,np.float32),'probe_SDF_raw':np.array(values,np.float32),
        'probe_masked_extraction_field':np.array(values,np.float32),'probe_inside_sphere':np.ones(12,bool),
        'probe_base_index':np.array(indices,np.int32),'probe_direction_code':np.zeros(12,np.int32),
        'probe_offset_over_h':np.array(offsets,np.float32),'face_indices':np.array([5]),'vertex_indices':np.array([3]),
        'face_areas':np.array([.01]),'face_cross_normals':np.array([[0,0,1.]]),'gradient_face_cosine':np.array([1.])}


class CloseoutTests(unittest.TestCase):
    def test_exact_failure_union_scale_localization_and_no_mutation(self):
        a=fixture();before=deepcopy(a);r=audit.localize(a)
        self.assertEqual((r['eligible'],r['excluded'],r['failures']),(3,0,2))
        self.assertEqual(r['failure_union'],{'proxy':1,'no_half_grid_crossing':2,'both':1})
        self.assertEqual(r['radial_bands']['1']['failed'],1)
        self.assertTrue(r['failed_samples'][1]['gradient_crossing_h4']);self.assertFalse(r['failed_samples'][1]['gradient_crossing_h2'])
        for k in a:np.testing.assert_array_equal(a[k],before[k])
        self.assertFalse(r['research_geometry_admitted'])
    def test_sphere_exclusion_keeps_original_mask_semantics(self):
        a=fixture();a['probe_inside_sphere'][6]=False
        r=audit.localize(a);self.assertEqual(r['excluded'],1);self.assertEqual(r['failures'],1)
    def test_missing_or_duplicate_probes_and_nonfinite_data_are_rejected(self):
        for change in [lambda a:a['probe_offset_over_h'].__setitem__(0,.75),
                       lambda a:a['probe_offset_over_h'].__setitem__(0,-.25),lambda a:a['gradient'].__setitem__((0,0),np.nan),
                       lambda a:a['distance_proxy'].__setitem__(0,.01)]:
            a=fixture();change(a)
            with self.assertRaises(SignatureError):audit.localize(a)
    def test_explicit_hash_binding_pickle_free_loading_and_original_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);npz=root/'numeric.npz';report=root/'saved.json';np.savez(npz,**fixture())
            sha=hashlib.sha256(npz.read_bytes()).hexdigest()
            original={'protocol':'bell-geometry-sdf-mesh-audit-v1','status':'complete','network_unchanged':True,
                'numeric_output':{'sha256':sha},'statuses':{'sdf_mesh_zero_level_consistency':{'eligible':3,'excluded':0,'failures':2}},
                'settings':{'direction_cosine':.866},'result':{'rays':[{'sample_index':2,'camera_id':'58','ray_index':20,'anomalous':False}]}}
            report.write_text(json.dumps(original));rsha=hashlib.sha256(report.read_bytes()).hexdigest()
            before=(report.read_bytes(),npz.read_bytes())
            with patch.object(audit,'REPORT_SHA',rsha),patch.object(audit,'ARRAY_SHA',sha),patch.object(np,'load',wraps=np.load) as reader:
                result=audit.analyze(report,rsha,npz,sha)
                self.assertEqual(result['result']['failures'],2);self.assertIs(reader.call_args.kwargs['allow_pickle'],False)
                npz.write_bytes(npz.read_bytes()+b'changed')
                with self.assertRaises(SignatureError):audit.analyze(report,rsha,npz,sha)
            self.assertEqual(report.read_bytes(),before[0])
    def test_unapproved_evidence_identity_fails_before_loading(self):
        with patch.object(np,'load',side_effect=AssertionError('Do not open')):
            with self.assertRaises(SignatureError):audit.analyze('/absent','0'*64,'/absent2','1'*64)


if __name__=='__main__':unittest.main()
