"""One CPU toy fixture, actual pinned lazy Trainer.run and unchanged two-return trace."""
from copy import deepcopy
from types import SimpleNamespace
import random
import sys

import numpy as np
import pytest
import yaml

torch=pytest.importorskip('torch')
from test_material_smoke import material_fixture,A
from training_smoke_fixtures import cpu_rng
from infrastructure import corrected_material_smoke as corrected
from infrastructure.signatures import SignatureError
from infrastructure.stage_execution import trainer_io


@pytest.fixture
def lazy(tmp_path,monkeypatch):
    states=random.getstate(),np.random.get_state(),torch.get_rng_state()
    monkeypatch.chdir(tmp_path)
    (tmp_path/'data/model').mkdir(parents=True)
    logs=tmp_path/'logs';logs.mkdir();checkpoints=tmp_path/'checkpoints';checkpoints.mkdir()
    f=material_fixture()
    variant='A_corrected_core_v1'
    f.Renderer.material_variant=f.Shader.material_variant=variant
    cfg=yaml.safe_load((A/'configs/material/real/bear.yaml').read_text());cfg['material_variant']=variant
    trainer=trainer_io(f.Trainer,SimpleNamespace(logs=logs,checkpoints=checkpoints))(deepcopy(cfg))
    assert not hasattr(trainer,'network')
    initialized=[];original=trainer._init_network
    def init_once():
        initialized.append('original_init_network');original()
    trainer._init_network=init_once
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**kw:self)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    modules={'network.renderer':SimpleNamespace(name2renderer={'material':f.Renderer}),
             'network.loss':SimpleNamespace(NeRFRenderLoss=f.losses['NeRFRenderLoss'],MaterialRegLoss=f.losses['MaterialRegLoss']),
             'train.trainer':SimpleNamespace(Trainer=f.Trainer),
             'train.train_valid':SimpleNamespace(ValidationEvaluator=f.Evaluator)}
    for name,module in modules.items():monkeypatch.setitem(sys.modules,name,module)
    report={'optimizer_calls_completed':0};created=[];real=corrected.CorrectedProbe
    def probe_factory(*args,**kwargs):
        result=real(*args,**kwargs);created.append(result);return result
    monkeypatch.setattr(corrected,'CorrectedProbe',probe_factory)
    def prepare():
        p=real(trainer,torch,np,report,lambda:None,rng=cpu_rng,memory=lambda:{'CPU_fixture':True})
        before=cpu_rng()
        p.prepare_calls(f.Renderer,f.losses['NeRFRenderLoss'],f.losses['MaterialRegLoss'],f.Trainer,f.Evaluator)
        assert cpu_rng()==before and not initialized and not hasattr(trainer,'network')
        return p
    yield SimpleNamespace(f=f,trainer=trainer,report=report,initialized=initialized,created=created,prepare=prepare,
                          run=lambda:corrected.run_bounded(trainer,A,torch,np,report,lambda:None))
    random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
    assert sys.gettrace() is None and sys.getprofile() is None


def test_actual_lazy_run_binds_once_two_returns_and_cleanup(lazy):
    lazy.run();r=lazy.report
    assert lazy.initialized==['original_init_network'] and len(lazy.f.networks)==1
    assert r['observer_lifecycle']=={'guards_prepared_before_run':True,'method_bindings':1}
    assert r['optimizer_calls_completed']==2 and len(r['updates'])==2 and r['cleanup_errors']==[]
    assert [u['lr'] for u in r['updates']]==[0.,5e-7]
    assert r['updates'][0]['changed_elements']==0 and r['updates'][1]['changed_elements']>0
    for u in r['updates']:
        assert u['forward_path_verified'] and u['backward_returned']
        assert u['calls']['shader.anisotropic_forward']=={'entered':1,'returned':1}
        assert u['calls']['loss.nerf_render']=={'entered':1,'returned':1}
        assert all(v['step']==u['update_index'] for v in u['optimizer']['state'].values())
        assert all(v['gradient_l2_cpu64']>0 for v in u['groups_before_clip'].values())
    assert lazy.f.loaders[0].iterator.closed and lazy.f.writers[0].closed
    with pytest.raises(SignatureError,match='more than once'):
        lazy.created[0].event('initialized',{})
    assert lazy.initialized==['original_init_network']


@pytest.mark.parametrize('operation',['validation','nvs','extraction','checkpoint','evaluator'])
def test_forbidden_bodies_guarded_before_original_run(lazy,operation):
    p=lazy.prepare();f=lazy.f
    calls={'validation':lambda:f.Renderer.test_step(None), 'nvs':lambda:f.Renderer.nvs(None),
           'extraction':lambda:f.Renderer.predict_materials(None),
           'checkpoint':lambda:f.Trainer._save_model(None,0,0), 'evaluator':lambda:f.Evaluator.__call__(None)}
    with pytest.raises(SignatureError,match='Validation/checkpoint/NVS'):
        with p.observe_calls():calls[operation]()
    assert not lazy.initialized and not hasattr(lazy.trainer,'network')


@pytest.mark.parametrize('missing',['network','shader','boundary'])
def test_missing_lazy_boundary_or_objects_fail_explicitly(lazy,missing):
    p=lazy.prepare()
    if missing=='shader':lazy.trainer.network=SimpleNamespace()
    with pytest.raises(SignatureError,match='Missing initialized|network/shader missing'):
        p.event('before_forward' if missing=='boundary' else 'initialized',{})
    assert lazy.initialized==[] and lazy.report['optimizer_calls_completed']==0


def test_pre_backward_failure_preserves_count_and_cleanup(lazy,monkeypatch):
    original=corrected.CorrectedProbe  # Factory wraps the actual probe class.
    def fail_factory(*args,**kwargs):
        p=original(*args,**kwargs);before=p.before_backward
        def failed(values):
            before(values);raise RuntimeError('synthetic pending backward failure')
        p.before_backward=failed
        return p
    monkeypatch.setattr(corrected,'CorrectedProbe',fail_factory)
    with pytest.raises(RuntimeError,match='pending backward'):
        lazy.run()
    assert lazy.initialized==['original_init_network'] and lazy.report['optimizer_calls_completed']==0
    assert lazy.report['cleanup_errors']==[] and lazy.f.loaders[0].iterator.closed and lazy.f.writers[0].closed
