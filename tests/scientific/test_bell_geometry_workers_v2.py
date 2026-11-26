"""Focused CPU checks. No original scientific model or CUDA work."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from infrastructure.bell_geometry_workers_v2 import spawned_loaders, validate_worker
from infrastructure.signatures import SignatureError


def test_spawn_loader_changes_only_transport_and_restores_binding():
    calls=[]
    dataset=object()
    collate=object()
    def original(*args,**kwargs):
        calls.append((args,kwargs))
        return SimpleNamespace(dataset=dataset,multiprocessing_context=SimpleNamespace(get_start_method=lambda:'spawn'))
    module=SimpleNamespace(DataLoader=original)
    report={}
    with spawned_loaders(module,report):
        module.DataLoader(dataset,1,True,num_workers=8,collate_fn=collate)
    assert module.DataLoader is original
    assert calls==[((dataset,1,True),dict(num_workers=8,collate_fn=collate,
                                        multiprocessing_context='spawn',worker_init_fn=validate_worker))]
    assert report['loader_transport']==[dict(workers=8,start_method='spawn',dataset='object')]


def test_exception_restores_original_loader():
    original=object()
    module=SimpleNamespace(DataLoader=original)
    with pytest.raises(SignatureError):
        with spawned_loaders(module,{}):
            module.DataLoader(object(),num_workers=0)
    assert module.DataLoader is original


def test_real_eight_spawn_workers_preserve_payloads_indices_and_parent_rng(tmp_path):
    # A standalone file is necessary for spawn's importable main module.
    script=tmp_path/'spawn_fixture.py'
    script.write_text("""
import json
import sys
from types import SimpleNamespace
from infrastructure.bell_geometry_workers_v2 import bootstrap_geometry_worker
if __name__=='__mp_main__':
    bootstrap_geometry_worker()
    import dataset
    from infrastructure.stage_sources import ROOT, imported_sources
    assert list(dataset.__path__)==[str(ROOT/'scientific_sources/geometry_g/dataset')]
    imported_sources(ROOT/'scientific_sources/geometry_g')
import torch
from torch.utils.data import Dataset,DataLoader
from infrastructure.bell_geometry_workers_v2 import spawned_loaders

class Dummy(Dataset):
    def __init__(self,validation=False):self.validation=validation
    def __len__(self):return 17
    def __getitem__(self,index):return {'index':index} if self.validation else {}
def collate(items):return items[0]
def run(wrapped):
    torch.manual_seed(6033)
    module=SimpleNamespace(DataLoader=DataLoader)
    report={}
    if wrapped:
        with spawned_loaders(module,report):
            train=module.DataLoader(Dummy(),1,True,num_workers=8,collate_fn=collate)
            val=module.DataLoader(Dummy(True),1,False,num_workers=8,collate_fn=collate)
    else:
        train=DataLoader(Dummy(),1,True,num_workers=8,collate_fn=collate,
                         multiprocessing_context='spawn')
        val=DataLoader(Dummy(True),1,False,num_workers=8,collate_fn=collate,
                       multiprocessing_context='spawn')
    initial=torch.get_rng_state().clone()
    iterator=iter(train)
    samples=list(iterator)
    assert not any(p.is_alive() for p in iterator._workers)
    after_train=torch.get_rng_state().clone()
    iterator=iter(val)
    indices=[d['index'] for d in iterator]
    assert not any(p.is_alive() for p in iterator._workers)
    assert not torch.cuda.is_initialized()
    return samples,indices,initial,after_train,torch.get_rng_state()
if __name__=='__main__':
    a,b=run(False),run(True)
    assert a[:2]==b[:2] and a[0]==[{}]*17 and a[1]==list(range(17))
    assert all(torch.equal(x,y) for x,y in zip(a[2:],b[2:]))
    print(json.dumps(dict(status='passed',workers=8,parent_rng_parity=True,CUDA_initialized=False)))
""")
    import os
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    root=Path(__file__).resolve().parents[2]
    env['PYTHONPATH']=str(root)
    run=subprocess.run([sys.executable,'-B',str(script)],cwd=root,env=env,capture_output=True,text=True,timeout=120)
    assert run.returncode==0,run.stdout+run.stderr
    assert json.loads(run.stdout)['parent_rng_parity']


def test_unclosed_namespace_reproduces_failure_and_bootstrap_prevents_fallback(tmp_path):
    root=Path(__file__).resolve().parents[2]
    foreign=tmp_path/'foreign'
    (foreign/'dataset').mkdir(parents=True)
    (foreign/'dataset'/'foreign_only.py').write_text('marker=True')
    template="""
import sys
from infrastructure.stage_sources import ROOT, imported_sources
from infrastructure.signatures import SignatureError
from infrastructure.bell_geometry_workers_v2 import bootstrap_geometry_worker
root=ROOT/'scientific_sources/geometry_g'
sys.path.insert(0,FOREIGN)
sys.path.insert(0,str(root))
BOOTSTRAP
import dataset
if CLOSED:
 assert list(dataset.__path__)==[str(root/'dataset')]
 imported_sources(root)
 try:
  import dataset.foreign_only
 except ModuleNotFoundError:
  pass
 else:
  raise AssertionError('Foreign namespace fallback accepted')
else:
 assert len(list(dataset.__path__))>=2
 try:
  imported_sources(root)
 except SignatureError as e:
  assert 'Mixed namespace path' in str(e)
 else:
  raise AssertionError('Unclosed namespace accepted')
"""
    for closed in (False,True):
        code=template.replace('FOREIGN',repr(str(foreign))).replace('BOOTSTRAP','bootstrap_geometry_worker()' if closed else '').replace('CLOSED',repr(closed))
        run=subprocess.run([sys.executable,'-B','-c',code],cwd=root,capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr


def test_bootstrap_rejects_already_imported_mixed_namespace():
    root=Path(__file__).resolve().parents[2]
    code="""
import dataset
from infrastructure.bell_geometry_workers_v2 import bootstrap_geometry_worker
from infrastructure.signatures import SignatureError
try:
 bootstrap_geometry_worker()
except SignatureError as exc:
 assert 'Scientific namespace already imported' in str(exc)
else:
 raise AssertionError('Already mixed namespace was silently repaired')
"""
    run=subprocess.run([sys.executable,'-B','-c',code],cwd=root,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr


def test_frozen_entry_bootstraps_before_dataset_payload_import():
    import ast
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'experiments/bell_geometry_gate10_v1.py').read_text())
    guard=next(n for n in tree.body if isinstance(n,ast.If) and '__mp_main__' in ast.dump(n.test))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='bootstrap_geometry_worker'
               for n in ast.walk(guard))
    imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and
             n.module and n.module.startswith(('benchmarks','experiments'))]
    assert guard.lineno<min(n.lineno for n in imports)


def test_actual_frozen_entry_runs_closed_worker_bootstrap_without_scientific_execution():
    root=Path(__file__).resolve().parents[2]
    code="""
import runpy,sys
from infrastructure.stage_sources import ROOT,StageFinder,imported_sources
runpy.run_path(str(ROOT/'experiments/bell_geometry_gate10_v1.py'),run_name='__mp_main__')
import dataset
assert list(dataset.__path__)==[str(ROOT/'scientific_sources/geometry_g/dataset')]
assert sum(isinstance(f,StageFinder) for f in sys.meta_path)==1
imported_sources(ROOT/'scientific_sources/geometry_g')
assert 'torch' not in sys.modules
assert 'network.renderer' not in sys.modules
"""
    run=subprocess.run([sys.executable,'-B','-c',code],cwd=root,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
