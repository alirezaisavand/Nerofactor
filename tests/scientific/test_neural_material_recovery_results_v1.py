"""New offline-report checks only; no shader, model, derivative or Adam work."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments import analyze_neural_material_recovery_v1 as a


class OfflineResultsTests(unittest.TestCase):
    def fixture(self,root):
        values={'a0':np.arange(7,dtype=np.float32),'a1':np.array(250,dtype=np.float32),
            'a2':np.ones(7,np.float32),'a3':np.ones(7,np.float32),'a4':np.arange(20,dtype=np.uint8),
            'numpy_rng':np.arange(624,dtype=np.uint32)}
        roles=[('parameter','latent'),('adam','latent/step'),('adam','latent/exp_avg'),('adam','latent/exp_avg_sq'),('torch_rng','CPU')]
        entries=[dict(key='a'+str(i),role=role,name=name,shape=list(values['a'+str(i)].shape),dtype='torch.'+str(values['a'+str(i)].dtype),
            device='cpu',sha256=a.array_sha(values['a'+str(i)]),finite=True) for i,(role,name) in enumerate(roles)]
        frozen=dict(configuration_sha256='c'*64,sha256={'source':'d'*64},configuration=dict(optimizer_effective={'lr':.01}))
        row=dict(arm='N0',initial_model=dict(source=None,configuration={},parameters={'latent':entries[0]},buffers={}))
        archive=root/'state.npz';np.savez(archive,**values)
        manifest=dict(schema_version=1,protocol='neural-material-recovery-v1',arm='N0',source=None,model_configuration={},
            purpose='state',completed_updates=250,optimizer_effective={'lr':.01},entries=entries,
            execution_provenance=dict(freeze_sha256=a.FREEZE_SHA,configuration_sha256='c'*64,source_sha256=frozen['sha256'],run_id='fixture'),
            archive=archive.name,archive_sha256=a.digest(archive),archive_bytes=archive.stat().st_size,
            numpy_rng=dict(kind='MT19937',array_sha256=a.array_sha(values['numpy_rng'])))
        path=root/'state.json';path.write_text(json.dumps(manifest));selected=dict(manifest=path.name,manifest_sha256=a.digest(path),completed_updates=250)
        return selected,row,frozen,values,manifest

    def test_safe_numeric_state_and_hash_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);selected,row,frozen,_,_=self.fixture(root)
            result=a.inspect_checkpoint(root,selected,row,frozen,'fixture');self.assertEqual(result['verified_arrays'],6)
            self.assertFalse(result['allow_pickle']);self.assertTrue(result['all_finite'])
            with (root/'state.npz').open('ab') as stream:stream.write(b'changed')
            with mock.patch.object(a.np,'load',side_effect=AssertionError('must reject hash first')) as loader:
                with self.assertRaisesRegex(ValueError,'hash/size'):a.inspect_checkpoint(root,selected,row,frozen,'fixture')
                loader.assert_not_called()

    def test_object_state_rejected_without_pickle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);selected,row,frozen,values,manifest=self.fixture(root)
            values['a0']=np.array([object()],dtype=object);np.savez(root/'state.npz',**values)
            manifest.update(archive_sha256=a.digest(root/'state.npz'),archive_bytes=(root/'state.npz').stat().st_size)
            (root/'state.json').write_text(json.dumps(manifest));selected['manifest_sha256']=a.digest(root/'state.json')
            with self.assertRaisesRegex(ValueError,'allow_pickle=False'):a.inspect_checkpoint(root,selected,row,frozen,'fixture')

    def test_strict_json_and_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'a.json';path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError,'Duplicate'):a.selected_json(path,a.digest(path))
            with mock.patch.object(a.json,'load',side_effect=AssertionError('hash first')):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):a.selected_json(path,'0'*64)
            outside=root.parent/(root.name+'-outside.json');outside.write_text('{}')
            try:
                with self.assertRaises(ValueError):a.child(root,'../'+outside.name)
                (root/'link.json').symlink_to(outside)
                with self.assertRaises(ValueError):a.child(root,'link.json')
            finally:outside.unlink()

    def test_unchanged_simultaneous_thresholds(self):
        frozen=json.loads((a.ROOT/a.FREEZE).read_text());limits=frozen['configuration']['success']
        errors=dict(Q_relative_frobenius=.05,orientation_degrees=5.,albedo_absolute_errors=[.02,.01,0.],metallic_absolute_error=.03)
        metrics={'held_out':{'total':{'relative_L2':.01}}}
        self.assertTrue(a.assess(errors,metrics,limits)['all_passed'])
        for key in ('Q_relative_frobenius','orientation_degrees','metallic_absolute_error'):
            changed=copy.deepcopy(errors);changed[key]=float(np.nextafter(changed[key],np.inf))
            self.assertFalse(a.assess(changed,metrics,limits)['all_passed'])
        changed=copy.deepcopy(errors);changed['orientation_degrees']=None
        self.assertFalse(a.assess(changed,metrics,limits)['all_passed'])

    def test_censored_and_unstarted_not_in_success_denominator(self):
        rows=[dict(arm=arm,material=m,seed=s,status='admitted',completed_updates=0) for arm in ('D0','N0','N1') for m in ('M1','M2','M3') for s in (101,103,105)]
        rows[0].update(status='complete',completed_updates=5000,evaluation={'success':{'all_passed':False,'checks':{'held_out_radiance':True}}})
        rows[9].update(status='censored',completed_updates=4860)
        result=a.statistics(rows)
        self.assertEqual(result['D0']['eligible_complete'],1);self.assertEqual(result['D0']['physical_pass'],0)
        self.assertEqual(result['N0']['planned'],9);self.assertEqual(result['N0']['eligible_complete'],0)
        self.assertEqual(result['N0']['censored'],1);self.assertEqual(result['N1']['unstarted'],9)

    def test_offline_import_excludes_torch_and_scientific_execution(self):
        script='''import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='torch' or fullname.startswith(('experiments.neural_material_field_v1','experiments.neural_material_recovery_v1','experiments.material_shader')):raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from experiments import analyze_neural_material_recovery_v1
assert 'torch' not in sys.modules
'''
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=a.ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
