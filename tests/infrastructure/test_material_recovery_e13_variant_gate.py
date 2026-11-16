"""Actual E.13 initialized boundary with CPU stand-ins; zero training calls."""
from copy import deepcopy
from types import SimpleNamespace
import random
import json

import numpy as np
import pytest
import yaml
torch=pytest.importorskip('torch')

from infrastructure.material_recovery_e13 import ARMS, RecoveryProbe
from infrastructure.material_variants import configuration
from infrastructure.signatures import SignatureError
from infrastructure.stage_execution import trainer_io
from infrastructure.stage_sources import ROOT
from infrastructure.stage_sources import digest
from test_material_smoke import material_fixture
from training_smoke_fixtures import cpu_rng


@pytest.fixture
def initialized(tmp_path,monkeypatch):
    states=random.getstate(),np.random.get_state(),torch.get_rng_state()
    monkeypatch.chdir(tmp_path)
    def build(variant,declared='normal',renderer_variant=None,shader_variant=None):
        suffix=dict(zip(ARMS,('original','estimator','core')))[variant]
        raw=yaml.safe_load((ROOT/('configs/material_corrected/bell_'+suffix+'_known_v1.yaml')).read_text())
        cfg=configuration(raw,variant)
        if declared=='missing':cfg.pop('material_variant',None)
        elif declared!='normal':cfg['material_variant']=declared
        cfg['train_dataset_type']='dummy'  # Existing stage adapter's exact dummy class.
        f=material_fixture()
        f.Renderer.material_variant=renderer_variant or variant
        f.Shader.material_variant=shader_variant or variant
        run=tmp_path/variant;run.mkdir()
        logs=run/'logs';logs.mkdir();checkpoints=run/'checkpoints';checkpoints.mkdir()
        trainer=trainer_io(f.Trainer,SimpleNamespace(logs=logs,checkpoints=checkpoints))(deepcopy(cfg))
        assert not hasattr(trainer,'network')
        trainer.e13_run=SimpleNamespace(output=lambda name:run/name)
        trainer._init_network()  # Original initialization body with tiny CPU classes, no shader forward.
        report={'variant':variant,'optimizer_calls_completed':0}
        probe=RecoveryProbe(trainer,torch,np,report,lambda:None,rng=cpu_rng,memory=lambda:{'CPU_fixture':True})
        probe.prepare_calls(f.Renderer,f.losses['NeRFRenderLoss'],f.losses['MaterialRegLoss'],f.Trainer,f.Evaluator)
        return SimpleNamespace(trainer=trainer,report=report,probe=probe,run=run,fixture=f,raw=raw)
    yield build
    random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])


@pytest.mark.parametrize('variant',ARMS)
def test_actual_three_configurations_reach_E13_initialization_without_mutation(initialized,variant):
    f=initialized(variant);before=deepcopy(f.trainer.cfg);rng=cpu_rng()
    if variant=='A_original':
        assert 'material_variant' not in f.raw and 'material_variant' not in f.trainer.cfg
    else:assert f.trainer.cfg['material_variant']==variant
    f.probe.event('initialized',{})
    assert f.trainer.cfg==before and cpu_rng()==rng
    assert f.report['initialization_parity']=='first-arm anchor'
    assert f.report['observer_lifecycle']['method_bindings']==1
    assert (f.run/'initialization.json').is_file()
    assert f.report['optimizer_calls_completed']==0 and f.report['trajectory']==[]
    assert not f.trainer.optimizer.state
    assert not any(isinstance(e,tuple) and e[0]=='forward' for e in f.fixture.events)
    assert not list((f.run/'checkpoints').iterdir())


@pytest.mark.parametrize('variant',ARMS[1:])
def test_missing_derivative_declaration_does_not_default_to_derivative(initialized,variant):
    f=initialized(variant,declared='missing');before=deepcopy(f.trainer.cfg)
    with pytest.raises(SignatureError,match='shader/renderer not selected'):f.probe.event('initialized',{})
    assert f.trainer.cfg==before and not (f.run/'initialization.json').exists()
    assert f.report['optimizer_calls_completed']==0 and not f.trainer.optimizer.state


@pytest.mark.parametrize('which',['renderer','shader','explicit_none','foreign_declaration'])
def test_original_default_still_rejects_foreign_or_invalid_identities(initialized,which):
    kwargs={}
    if which=='renderer':kwargs['renderer_variant']=ARMS[1]
    elif which=='shader':kwargs['shader_variant']=ARMS[1]
    elif which=='explicit_none':kwargs['declared']=None
    else:kwargs['declared']='foreign'
    f=initialized(ARMS[0],**kwargs);before=deepcopy(f.trainer.cfg)
    with pytest.raises(SignatureError,match='shader/renderer not selected'):f.probe.event('initialized',{})
    assert f.trainer.cfg==before and not (f.run/'initialization.json').exists()
    assert f.report['optimizer_calls_completed']==0 and not f.trainer.optimizer.state


def test_v2_preserves_science_inputs_limits_and_v1_freeze():
    from experiments.bell_material_recovery_e13 import FREEZE,verify_freeze
    old=ROOT/'docs/protocols/BELL_THREE_ARM_RECOVERY_E13_V1.freeze.json'
    assert digest(old)=='5133290f15f2b902cd78449bf4c0a7cbe195757bb1a0e607a241f348e15c8e5e'
    before=json.loads(old.read_text());after=verify_freeze(digest(FREEZE))
    for key in ('inputs','limits','image_id','admission','host_benchmark','protocol'):
        assert after[key]==before[key]
    assert after['host_output_directory']!=before['host_output_directory']
    assert all(after['run_ids'][v]!=before['run_ids'][v] for v in ARMS)
    changed={p for p,h in before['files'].items() if digest(ROOT/p)!=h}
    assert changed=={'infrastructure/corrected_material_smoke.py','experiments/bell_material_recovery_e13.py',
                     'experiments/run_bell_material_recovery_e13.sh','tests/infrastructure/test_material_recovery_e13.py'}
