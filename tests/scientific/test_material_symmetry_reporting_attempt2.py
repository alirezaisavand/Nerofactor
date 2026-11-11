"""Reporting-only regressions for the explicitly approved fresh audit attempt."""
import ast
import io
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np
import torch

from experiments import material_symmetry_audit_v1 as audit
from test_material_symmetry_audit_v1 import capture, TinyBatch

FAILED_COMMIT='13fe645af355f66400d435f9e25f5fcc63870912'


class SymmetryReportingAttempt2Tests(unittest.TestCase):
    def test_exact_prior_numpy_boolean_collision_without_changing_predicate(self):
        source=subprocess.check_output(['git','show',FAILED_COMMIT+':experiments/material_symmetry_audit_v1.py'],cwd=audit.ROOT,text=True)
        tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='physical_equivalence')
        namespace=dict(vars(audit))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'frozen-reporting-helper-fixture','exec'),namespace)
        base=audit.physics(audit.reference.TRUTH.copy())
        old=namespace['physical_equivalence'](base,base);new=audit.physical_equivalence(base,base)
        self.assertIsInstance(old['checks']['orientation'],np.bool_)
        with self.assertRaisesRegex(TypeError,'bool_'):json.dumps(old,allow_nan=False)
        self.assertIs(type(new['checks']['orientation']),bool)
        self.assertEqual(old,new);json.dumps(new,allow_nan=False)

    def test_every_analytical_gauge_record_roundtrips_strict_json(self):
        for isotropic in (False,True):
            p=audit.reference.TRUTH.copy()
            if isotropic:p[:2]=.4
            base=audit.physics(p);rounded_base=audit.physics(p.astype(np.float32).astype(float))
            for label in ('P0','P1','P2'):
                transformed=audit.transform(p,label)
                f64=audit.physics(transformed);f32=audit.physics(transformed.astype(np.float32).astype(float))
                record=dict(physics=f64,rounded=f32,check=audit.physical_equivalence(f64,base),
                    rounded_check=audit.physical_equivalence(f32,rounded_base,rounded=True))
                payload=json.dumps(record,allow_nan=False);decoded=json.loads(payload)
                self.assertTrue(decoded['check']['passed']);self.assertTrue(decoded['rounded_check']['passed'])
                self.assertTrue(all(type(v) is bool for v in record['check']['checks'].values()))

    def test_complete_tiny_capture_and_comparison_record_is_serializable(self):
        a,p,_=capture();b,q,_=capture(audit.transform(audit.reference.TRUTH,'P1'))
        record=dict(forward=audit.serialize_forward(a,p,np.zeros((1,3,3)),TinyBatch.observations,torch.zeros(1,3)),
            radiance=audit.radiance_comparison(b,a,TinyBatch.observations),sampling=audit.sampling_comparison(b,a),
            objective=audit.objective_difference(2e-7,1e-7))
        stream=io.StringIO();audit.persist_report(stream,record)
        self.assertEqual(json.loads(stream.getvalue()),record)

    def test_unserializable_payload_keeps_previous_valid_snapshot_byte_exact(self):
        stream=io.StringIO();audit.persist_report(stream,dict(status='running',work=dict(shader_calls=0)))
        before=stream.getvalue();position=stream.tell()
        with self.assertRaises(TypeError):audit.persist_report(stream,dict(status='failed',bad=np.bool_(True)))
        self.assertEqual(stream.getvalue(),before);self.assertEqual(stream.tell(),position)
        with self.assertRaises(ValueError):audit.persist_report(stream,dict(bad=float('nan')))
        self.assertEqual(stream.getvalue(),before);self.assertEqual(stream.tell(),position)

    def test_successful_snapshot_truncates_previous_longer_snapshot(self):
        stream=io.StringIO();audit.persist_report(stream,dict(long='x'*1000))
        audit.persist_report(stream,dict(status='complete'))
        self.assertEqual(stream.getvalue(),json.dumps(dict(status='complete'),indent=2)+'\n')

    def test_all_frozen_scientific_function_bodies_are_unchanged(self):
        old=ast.parse(subprocess.check_output(['git','show',FAILED_COMMIT+':experiments/material_symmetry_audit_v1.py'],cwd=audit.ROOT,text=True))
        new=ast.parse(Path(audit.__file__).read_text())
        functions={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
        for n in old.body:
            if isinstance(n,ast.FunctionDef) and n.name not in ('physical_equivalence','main'):
                self.assertEqual(ast.dump(n,include_attributes=False),ast.dump(functions[n.name],include_attributes=False),n.name)
        # The analytical predicate itself is identical: only its result gets bool().
        fn=functions['physical_equivalence'];calls=[n for n in ast.walk(fn) if isinstance(n,ast.keyword) and n.arg=='orientation']
        self.assertEqual(len(calls),1);self.assertIsInstance(calls[0].value,ast.Call)
        self.assertEqual(calls[0].value.func.id,'bool')


if __name__=='__main__':unittest.main()
