"""E2a ENGINEERING fixtures only. Never import/construct the scientific N1/shader."""
import copy
import hashlib
import json
import importlib.abc
import ast
from contextlib import contextmanager
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from infrastructure import neural_material_observer_e2a as o
from experiments import neural_material_early_step_e2a as r


@pytest.fixture(scope='session',autouse=True)
def forbid_scientific_imports():
    forbidden=('experiments.neural_material_field_v1','experiments.material_shader_v1',
               'experiments.neural_material_recovery_v1')
    assert not any(n.startswith(forbidden) for n in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,*args):
            if fullname.startswith(forbidden):raise AssertionError('Scientific import forbidden in engineering fixtures: '+fullname)
    block=Block();sys.meta_path.insert(0,block)
    try:yield
    finally:
        sys.meta_path.remove(block)
        assert not any(n.startswith(forbidden) for n in sys.modules)


class ToyHead(nn.Module):
    def __init__(self,width):
        super().__init__();self.predictor=nn.Sequential(nn.Linear(3,width),nn.Sigmoid())
    def forward(self,x):return self.predictor[-1](self.predictor[0](x)+.125)


class EngineeringModel(nn.Module):
    """Tiny deterministic, unrelated toy network with the observation interface."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            self.feats_network=nn.Sequential(nn.Linear(2,3),nn.ReLU())
            for name,width in zip(o.HEADS,(1,1,1,2,3)):setattr(self,name,ToyHead(width))
    def forward(self,shader):
        x=self.feats_network(torch.tensor([[.3,-.2]],dtype=torch.float32))
        q={h:getattr(self,h)(x) for h in o.HEADS}
        encoding=q['rotation_predictor']*2-1
        # Diagnostic theta is deliberately disconnected from the shader loss.
        theta=torch.atan2(encoding[:,1],encoding[:,0])
        material=torch.cat([q['mx_predictor'],q['my_predictor'],theta[:,None],q['kd_predictor'],q['metallic_predictor']],1)
        values=shader.shade(None,None,None,q['mx_predictor'],q['my_predictor'],q['kd_predictor'],q['metallic_predictor'],encoding)
        return values,material


class ToyShader:
    def shade(self,position,normal,view,mx,my,albedo,metallic,encoding):
        return albedo+mx+2*my+metallic+.1*encoding[:,:1]+.2*encoding[:,1:]


class Core:
    scientific=False
    def __init__(self,lr=.001):
        self.model=EngineeringModel();self.shader=ToyShader();self.opt=torch.optim.Adam(self.model.parameters(),lr=lr)
        self.calls=dict(forward=0,backward=0,optimizer=0,evaluation=0)
    def forward(self,shader,terminal):self.calls['forward']+=1;return self.model(shader)
    def loss(self,v):return (v*v).mean()
    def backward(self,loss):
        self.calls['backward']+=1;loss.backward();groups={}
        for n,p in self.model.named_parameters():
            g=groups.setdefault(n.split('.')[0],dict(squared_norm=0.,missing=0,finite=True,nonzero=0))
            g['squared_norm']+=float(torch.sum(p.grad.double()**2));g['nonzero']+=int(torch.count_nonzero(p.grad))
        return groups
    def step(self):self.calls['optimizer']+=1;self.opt.step()
    def physical(self,m):return dict(physical=m.detach()[0].tolist())
    def options(self):return dict(lr=self.opt.param_groups[0]['lr'])
    def evaluate(self):
        self.calls['evaluation']+=1
        with torch.no_grad():return dict(prediction=self.model(self.shader)[0].tolist())
    def native_work(self):return dict(self.calls)


def plan():
    mib=1024**2
    return dict(storage=dict(initial_archive=mib,final_archive=mib,failure_archive=mib,signal_archive=mib,
        xor_archive=mib,maximum_signal_arrays=130))


def test_enabled_disabled_exact_forward_gradients_and_rng_cleanup():
    a,b=EngineeringModel(),EngineeringModel();rng=torch.get_rng_state().clone()
    with o.GraphObserver(a,enabled=False) as observer:
        observer.begin();va,_=a(observer.wrap_shader(ToyShader()));va.square().mean().backward()
        assert observer.collect()==({},dict(enabled=False))
    with o.GraphObserver(b) as observer:
        observer.begin();vb,_=b(observer.wrap_shader(ToyShader()));vb.square().mean().backward()
        arrays,summary=observer.collect()
        assert arrays['shader.encoding.loss_gradient'].shape==(1,2)
        assert summary['nodes']['rotation_predictor.sigmoid']['loss_gradient']['nonzero']==2
        assert len(o.gradient_records(b))==12
        assert {n.split('.')[0] for n in o.gradient_records(b)}==set(o.GROUPS)
    assert torch.equal(va,vb) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(a.parameters(),b.parameters()))
    assert not observer.handles and not observer.nodes


def test_actual_calibrated_sigmoid_input_derivative_relu_and_encoding_not_theta():
    model=EngineeringModel()
    raw=[];handle=model.mx_predictor.predictor[0].register_forward_hook(lambda m,args,out:raw.append(out.detach().numpy().copy()))
    with torch.no_grad():
        model.feats_network[0].weight.zero_();model.feats_network[0].bias.copy_(torch.tensor([-2.,1.,2.]))
    with o.GraphObserver(model) as observer:
        observer.begin();v,material=model(observer.wrap_shader(ToyShader()));v.sum().backward()
        arrays,s=observer.collect();q=arrays['mx_predictor.sigmoid'];z=arrays['mx_predictor.logits']
        np.testing.assert_array_equal(z,raw[0]+np.float32(.125))
        np.testing.assert_allclose(q,1/(1+np.exp(-z)),rtol=1e-6)
        np.testing.assert_array_equal(arrays['mx_predictor.sigmoid.analytic_derivative'],q.astype(np.float64)*(1-q.astype(np.float64)))
        assert s['nodes']['relu.feats_network.1.post']['activity']==dict(positive=2,zero=1)
        assert arrays['relu.feats_network.1.pre.loss_gradient'][0,0]==0
        np.testing.assert_allclose(arrays['shader.encoding.loss_gradient'],[[.3,.6]],rtol=1e-6)
        assert 'theta' not in arrays and material.shape==(1,7)
    handle.remove()


def test_capture_cleanup_on_error_and_missing_gradient_rejection():
    m=EngineeringModel();observer=o.GraphObserver(m)
    with pytest.raises(ValueError,match='fixture'):
        with observer:observer.begin();m(observer.wrap_shader(ToyShader()));raise ValueError('fixture')
    assert not observer.handles and not observer.nodes
    with observer:
        observer.begin();m(observer.wrap_shader(ToyShader()))
        with pytest.raises(Exception,match='Missing retained'):observer.collect()


def test_nonfinite_tensor_and_missing_parameter_group_fail():
    with pytest.raises(Exception,match='Finite'):o.array_record(np.array([np.nan],np.float32))
    m=EngineeringModel()
    with pytest.raises(Exception,match='Missing parameter gradient'):o.gradient_records(m)
    for p in m.parameters():p.grad=torch.ones_like(p)
    del m.mx_predictor
    with pytest.raises(Exception,match='six original'):o.gradient_records(m)


def test_actual_float32_and_signed_zero_xor_reconstruction():
    before={'feats_network.p':np.array([1.,0.,-0.,1.],np.float32)}
    after={'feats_network.p':np.array([1.,-0.,0.,np.nextafter(np.float32(1),np.float32(2))],np.float32)}
    xor,record=o.displacement(before,after);t=record['per_tensor']['feats_network.p']
    assert t['changed_elements']==1 and t['bitwise_changed_elements']==3
    np.testing.assert_array_equal(o.reconstruct(before,xor)['feats_network.p'].view('u4'),after['feats_network.p'].view('u4'))
    with pytest.raises(Exception,match='dtype'):o.reconstruct(before,{'feats_network.p':xor['feats_network.p'].astype('f4')})


def test_adam_zero_current_gradient_can_move_and_predicted_roundoff_is_not_motion():
    # Toy optimizer engineering only: observe actual returned float32 parameters.
    p=nn.Parameter(torch.tensor([1.],dtype=torch.float32));opt=torch.optim.Adam([p],lr=.01)
    p.grad=torch.ones_like(p);opt.step();before={'feats_network.p':p.detach().numpy().copy()}
    p.grad.zero_();opt.step();after={'feats_network.p':p.detach().numpy().copy()}
    assert o.displacement(before,after)[1]['per_tensor']['feats_network.p']['changed_elements']==1
    # Nonzero moment-derived proposal can round away in the actual parameter.
    large=nn.Parameter(torch.tensor([1e10],dtype=torch.float32));other=torch.optim.Adam([large],lr=.01)
    initial=large.detach().clone();large.grad=torch.ones_like(large);other.step()
    assert other.state[large]['exp_avg'].item()!=0 and torch.equal(initial,large)


def test_exact_label_state_logic_and_saved_data_gate():
    expected={str(label):dict(update=label,loss_at_update=label-1,loss=.2,gradient_norm=.4,physical_before_update=[.3]) for label in (1,10)}
    r.parity(0,.2,.4,[.3],expected);r.parity(9,.2,.4,[.3],expected)
    for state,loss in ((10,.2),(9,.20000000001)):
        with pytest.raises(Exception):r.parity(state,loss,.4,[.3],expected)
    expected['10']['loss_at_update']=10
    with pytest.raises(Exception,match='parity'):r.parity(9,.2,.4,[.3],expected)


def test_work_caps_include_terminal_and_external():
    w=r.Work()
    for _ in range(10):
        w.attempt('forward','training');w.done('forward');w.attempt('backward');w.done('backward');w.attempt('optimizer');w.done('optimizer')
    with pytest.raises(Exception,match='cap'):w.attempt('optimizer')
    w.attempt('forward','terminal');w.done('forward');w.attempt('backward');w.done('backward')
    w.attempt('forward','external');w.done('forward');w.complete()
    assert w.value==r.LIMITS
    with pytest.raises(Exception,match='cap'):w.attempt('forward','external')


def test_toy_ten_returns_terminal_prepost_records_and_auditable_xor(tmp_path):
    store=o.Store(tmp_path/'engineering',tmp_path);core=Core();out=r.run_arm(core,store,'toy',.001,{},plan())
    assert out['status']=='complete' and out['completed_updates']==10 and out['work']==r.LIMITS
    assert core.calls==dict(forward=11,backward=11,optimizer=10,evaluation=1)
    values=o.read_numeric(store.root,out['initial_state']);parameters={n[len('parameter/'):]:a for n,a in values.items() if n.startswith('parameter/')}
    for state,selected in enumerate(out['records']):
        row=json.loads((store.root/selected['path']).read_text())
        assert row['pre_state']==row['loss_state']==state and row['post_state']==min(10,state+1)
        if state==0:assert row['Adam_before_return']=={}
        else:assert all(v['step']['max_absolute']==state for v in row['Adam_before_return'].values())
        if state<10:
            parameters=o.reconstruct(parameters,o.read_numeric(store.root,row['xor']))
            assert o.parameter_identity(parameters)==row['actual_displacement']['after_identity']
    assert o.parameter_identity(parameters)==out['final_parameter_identity'] and out['hooks_removed']


def test_toy_interruption_preserves_partial_counts_and_removes_hooks(tmp_path):
    core=Core();store=o.Store(tmp_path/'engineering',tmp_path);row=dict(material='toy',seed=1)
    def check(reserve=0):
        if core.calls['optimizer']==2:raise r.Censored('engineering interruption')
    with pytest.raises(r.Censored):r.run_arm(core,store,'toy',.001,{},plan(),check,result_row=row)
    assert row['status']=='censored' and row['completed_updates']==2 and row['hooks_removed']
    assert row['native_work']['optimizer']==2 and row['failure_state']
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.model.modules())
    assert json.loads((store.root/'toy/arm-result.json').read_text())['status']=='censored'


def test_every_state_onset_records_zero_groups_without_inventing_boundaries(tmp_path):
    core=Core();core.loss=lambda values:0*values.square().mean()  # Deliberately zero TOY objective, never scientific fallback.
    store=o.Store(tmp_path/'engineering',tmp_path);out=r.run_arm(core,store,'toy',.001,{},plan())
    assert out['first_all_zero_gradient_state']==0
    assert out['first_zero_gradient_states']==dict.fromkeys(o.GROUPS,0)
    assert out['first_zero_logit_gradient_states']==dict.fromkeys(o.HEADS,0)
    assert out['first_zero_shader_gradient_states']==dict.fromkeys(('mx','my','albedo','metallic','encoding'),0)
    assert out['first_boundary_states']=={} and out['completed_updates']==10


@pytest.mark.parametrize('reason',['source','initial','tape'])
def test_selected_identity_mismatch_rejection(reason):
    # Same strict primitive used for independent factory admission, without creating N1.
    from experiments.neural_material_early_step_e2a import require_identity
    with pytest.raises(Exception,match='identity'):require_identity({'sha':'a'},{'sha':'b'},reason)


def test_numeric_corruption_and_object_rejection(tmp_path):
    store=o.Store(tmp_path/'engineering',tmp_path);selection=store.numeric('toy/array.npz',{'x':np.array([1],np.float32)},4096)
    assert o.read_numeric(store.root,selection)['x'].tolist()==[1]
    with pytest.raises(Exception,match='Object'):store.numeric('toy/object.npz',{'x':np.array([{}],object)},4096)
    manifest=json.loads((store.root/selection['path']).read_text());p=store.root/manifest['archive'];p.write_bytes(b'corrupted')
    with pytest.raises(Exception,match='hash'):o.read_numeric(store.root,selection)


def test_output_paths_collisions_and_budgets(tmp_path):
    store=o.Store(tmp_path/'engineering',tmp_path,total_cap=800)
    store.json('toy/x.json',{})
    with pytest.raises(FileExistsError):store.json('toy/x.json',{})
    with pytest.raises(Exception):store.json('../escape.json',{})
    with pytest.raises(Exception):o.Store(store.root,tmp_path)
    with pytest.raises(Exception,match='capture cap'):store.json('toy/large.json',{'x':'x'*1000})
    with pytest.raises(Exception,match='Numeric catalog'):store.numeric('toy/big.npz',{'x':np.zeros(1000)},100)
    for i in range(63):store.json('count/'+str(i)+'.json',{})
    store.json('count/last.json',{})
    with pytest.raises(Exception,match='inventory cap'):store.json('count/overflow.json',{})


def test_symlink_output_escape_is_rejected(tmp_path):
    store=o.Store(tmp_path/'engineering',tmp_path)
    (store.root/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(Exception):store.json('link/escape.json',{})
    assert not (tmp_path/'escape.json').exists()


def test_runtime_cleanup_with_mocked_caps_and_timers(monkeypatch):
    # Engineering guard fixture: no real signals, timers or process limits changed.
    for k,v in dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items():monkeypatch.setenv(k,v)
    calls=[];handlers={r.signal.SIGALRM:'alarm-before',r.signal.SIGTERM:'term-before'}
    monkeypatch.setattr(r.signal,'getsignal',lambda s:handlers[s])
    monkeypatch.setattr(r.signal,'signal',lambda s,v:calls.append(('handler',s,v)))
    monkeypatch.setattr(r.signal,'getitimer',lambda _: (0.,0.))
    monkeypatch.setattr(r.signal,'setitimer',lambda *args:calls.append(('timer',args)))
    monkeypatch.setattr(r.resource,'getrlimit',lambda _: (-1,-1))
    monkeypatch.setattr(r.resource,'setrlimit',lambda *args:calls.append(('resource',args)))
    before=torch.get_rng_state().clone()
    with pytest.raises(r.Censored,match='engineering'):
        with r.limits() as (check,begin,end):
            begin();check();end();raise r.Censored('engineering')
    assert torch.equal(before,torch.get_rng_state())
    assert ('handler',r.signal.SIGALRM,'alarm-before') in calls
    assert ('handler',r.signal.SIGTERM,'term-before') in calls
    assert ('resource',(r.resource.RLIMIT_AS,(-1,-1))) in calls
    assert any(c[0]=='timer' and c[1]==(r.signal.ITIMER_REAL,0) for c in calls)


def test_no_scientific_imports_and_no_unguarded_execution_route():
    script="""
import sys, importlib.abc
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.startswith(('torch','experiments.neural_material_field_v1','experiments.material_shader_v1','experiments.neural_material_recovery_v1')):raise AssertionError('scientific import '+fullname)
sys.meta_path.insert(0,Reject())
from experiments import neural_material_early_step_e2a as r
from experiments import freeze_neural_material_early_step_e2a as f
try:r.validate_review({'scientific_execution_authorized':False},'0'*64)
except ValueError:pass
else:raise AssertionError('authorization bypass')
assert 'torch' not in sys.modules
print('zero scientific imports/constructions/queries/updates')
"""
    p=subprocess.run(['python3','-c',script],text=True,capture_output=True)
    assert p.returncode==0,p.stderr
    assert p.stdout.strip()=='zero scientific imports/constructions/queries/updates'


@pytest.mark.parametrize('change',[{},dict(phase='E.2a'),dict(scientific_execution_authorized=False),dict(freeze_sha256='a'*64)])
def test_separate_review_is_required_before_scientific_factory(change):
    review=dict(phase='E.2b',scientific_execution_authorized=True,freeze_sha256='0'*64,reviewed_branch='x',reviewed_commit='x',
        output_root='/tmp/forbidden',run_id='neural-e2b-forbidden',external_watchdog=dict(wall_seconds=1800,kill_grace_seconds=10))
    if not change:review={}
    else:review.update(change)
    with pytest.raises(Exception):r.validate_review(review,'0'*64)


def test_schema_and_shape_only_caps_without_model_calls():
    from experiments import freeze_neural_material_early_step_e2a as f
    old=json.loads((f.ROOT/f.e1.PINS['freeze'][0]).read_text())
    model=old['initial_models']['M1/101/N1'];cap=r.caps(model)
    assert cap['parameters']==5942336 and cap['parameter_tensors']==84 and cap['retained_signal_and_gradient_bytes']==98528
    assert cap['minimum_free_bytes']==2*cap['subtotal_bytes']
    assert len(r.CASES)==5 and len(r.RATES)==2 and 10*r.LIMITS['successful_updates']==100


def test_scientific_core_cannot_use_engineering_route(tmp_path):
    core=Core();core.scientific=True;store=o.Store(tmp_path/'engineering',tmp_path)
    with pytest.raises(Exception,match='admitted E2b factory'):r.run_arm(core,store,'toy',.001,{},plan())
    assert all(v==0 for v in core.calls.values()) and not list(store.root.iterdir())


def test_malformed_npz_is_rejected_even_with_rehashed_archive(tmp_path):
    store=o.Store(tmp_path/'engineering',tmp_path);selected=store.numeric('toy/a.npz',{'x':np.ones(1,np.float32)},4096)
    p=store.root/selected['path'];meta=json.loads(p.read_text());archive=store.root/meta['archive']
    np.savez(archive,a0000=np.array([{}],dtype=object));meta['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    p.write_text(json.dumps(meta));selected['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Object arrays'):o.read_numeric(store.root,selected)


def test_failure_nonfinite_archive_is_not_valid_regular_evidence(tmp_path):
    store=o.Store(tmp_path/'engineering',tmp_path)
    selected=store.numeric('toy/failure.npz',{'p':np.array([np.nan],np.float32)},4096,allow_nonfinite=True)
    with pytest.raises(Exception,match='nonfinite'):o.read_numeric(store.root,selected)
    assert np.isnan(o.read_numeric(store.root,selected,allow_nonfinite=True)['p'][0])


def test_initial_pair_signature_mismatch_stops_before_any_update(tmp_path):
    core=Core();store=o.Store(tmp_path/'engineering',tmp_path)
    with pytest.raises(Exception,match='Paired initial'):
        r.run_arm(core,store,'toy',.001,{},plan(),paired_initial=dict(loss=-1,gradient_norm=0,physical=[]))
    assert core.calls['optimizer']==0
    row=json.loads((store.root/'toy/arm-result.json').read_text())
    assert row['status']=='failed' and row['work']['forwards']==row['work']['backwards']==1


def test_baseline_gate_failure_stops_before_update(tmp_path):
    core=Core(.01);store=o.Store(tmp_path/'engineering',tmp_path)
    with pytest.raises(Exception,match='historical'):
        r.run_arm(core,store,'toy',.01,{'1':dict(update=1,loss_at_update=0,loss=-1,gradient_norm=0,physical_before_update=[])},plan())
    assert core.calls['optimizer']==0


def test_freeze_hash_and_inventory_gates_use_saved_metadata_only(tmp_path,monkeypatch):
    from experiments import freeze_neural_material_early_step_e2a as f
    monkeypatch.setattr(f,'ROOT',tmp_path);monkeypatch.setattr(f,'FREEZE','freeze.json');monkeypatch.setattr(f,'VALIDATION','validation.json')
    expected=dict(execution_authorized=False,source_sha256={'toy':'a'})
    validation=dict(scientific_model_calls=0,scientific_renderer_calls=0,scientific_optimizer_updates=0,
        tests=validation_fixture(),source_sha256={'toy':'a'})
    (tmp_path/'validation.json').write_text(json.dumps(validation))
    freeze=dict(expected,validation_sha256=f.sha(tmp_path/'validation.json'))
    (tmp_path/'freeze.json').write_text(json.dumps(freeze));selected=f.sha(tmp_path/'freeze.json')
    monkeypatch.setattr(f,'expected',lambda:copy.deepcopy(expected))
    assert f.validate_freeze(selected)==freeze
    with pytest.raises(Exception,match='SHA-256 mismatch'):f.validate_freeze('0'*64)
    expected['source_sha256']['toy']='changed'
    with pytest.raises(Exception,match='contract changed'):f.validate_freeze(selected)


def test_baseline_inherited_evidence_and_all_selected_saved_parity_rows():
    from experiments import freeze_neural_material_early_step_e2a as f
    evidence,models,parity,ids=f.inherited()
    assert len(models)==len(parity)==5 and len(ids)>=348
    for expected in parity.values():
        for state in (0,9):
            row=expected[str(state+1)]
            r.parity(state,row['loss'],row['gradient_norm'],row['physical_before_update'],expected)
    assert f.sha(f.ROOT/f.E1)==f.E1_SHA


def validation_fixture():
    xml='<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="engineering" /></testsuite></testsuites>'
    return dict(failed=0,errors=0,skipped=0,passed=1,process_exit_code=0,names=['engineering'],junit_xml=xml,
        junit_sha256=hashlib.sha256(xml.encode()).hexdigest(),stdout='',stderr='',
        stdout_sha256=hashlib.sha256(b'').hexdigest(),stderr_sha256=hashlib.sha256(b'').hexdigest())


def metadata_rows():
    # Pure synthetic dictionaries, not executed scientific arms.
    native=dict(shader_completed=12,reverse_completed=11,optimizer_returns=10)
    rows=[dict(material=m,seed=s,lr=lr,status='complete',completed_updates=10,hooks_removed=True,
        work=dict(r.LIMITS),native_work=dict(native)) for m,s in r.CASES for lr in r.RATES]
    return rows,dict(native_work_per_arm=native,total_work={k:10*v for k,v in r.LIMITS.items()})


@pytest.mark.parametrize('phase',['training','terminal','external',None,'invalid'])
def test_rejected_forward_is_not_an_attempt(phase):
    w=r.Work()
    if phase in ('training','terminal','external'):
        w.value[phase+'_forwards']=r.LIMITS[phase+'_forwards']
    before=dict(w.value)
    with pytest.raises(Exception):w.attempt('forward',phase)
    assert w.value==before


@contextmanager
def engineering_runtime():
    yield (lambda *args:None,lambda:None,lambda:None)


def lifecycle_fixture(tmp_path):
    rows,p=metadata_rows();store=o.Store(tmp_path/'engineering',tmp_path)
    result=dict(outcomes=rows)
    def integrity(checks):r.terminal_integrity(p,rows,'0'*64,lambda:None,checks,verify_freeze=lambda sha:copy.deepcopy(p))
    return result,store,p,integrity


def test_cleanup_success_precedes_integrity_and_sealed_finalization(tmp_path):
    result,store,p,integrity=lifecycle_fixture(tmp_path);events=[]
    @contextmanager
    def runtime():
        events.append('enter');yield (None,None,None)
        assert not result['cleanup_verified'] and not result['scientific_work_completed']
        events.append('cleanup-return')
    def check(checks):
        assert events==['enter','work','cleanup-return'] and result['cleanup_verified']
        assert not result['scientific_work_completed'];integrity(checks)
    out=r.finish_execution(result,store,lambda *args:events.append('work'),check,p,runtime)
    assert out['status']=='complete' and out['scientific_work_completed'] and out['evidence_finalized'] and out['cleanup_verified']
    saved=json.loads((store.root/'result.json').read_text());seal=json.loads((store.root/'finalization/seal.json').read_text())
    assert saved['status']=='work_and_cleanup_verified' and not saved['evidence_finalized']
    assert seal['outcome']=='complete' and seal['report']['sha256']==hashlib.sha256((store.root/'result.json').read_bytes()).hexdigest()
    assert seal['successful_process_exit_also_required']


def test_cleanup_failure_preserves_work_without_claiming_scientific_completion(tmp_path):
    result,store,p,integrity=lifecycle_fixture(tmp_path)
    @contextmanager
    def runtime():
        yield (None,None,None)
        raise ValueError('synthetic RNG/resource cleanup failure')
    with pytest.raises(ValueError,match='cleanup failure'):
        r.finish_execution(result,store,lambda *args:None,integrity,p,runtime)
    assert result['status']=='failed' and result['failure_stage']=='runtime_cleanup'
    assert result['work_counters_verified'] and not result['cleanup_verified'] and not result['scientific_work_completed']
    assert result['runtime_cleanup_status']=='failed' and result['evidence_finalized']
    assert all(v['status']=='not_verified' for v in result['terminal_integrity'].values())
    assert json.loads((store.root/'result.json').read_text())['status']=='failed'
    assert json.loads((store.root/'finalization/seal.json').read_text())['outcome']=='failed'


def test_interrupted_work_keeps_missing_verification_explicit(tmp_path):
    result,store,p,integrity=lifecycle_fixture(tmp_path)
    result['outcomes'][2]['status']='not_started'
    def work(*args):raise r.Censored('engineering interruption')
    with pytest.raises(r.Censored):r.finish_execution(result,store,work,integrity,p,engineering_runtime)
    assert result['status']=='censored' and not result['work_counters_verified']
    assert not result['cleanup_verified'] and not result['scientific_work_completed']
    assert result['runtime_cleanup_status']=='not_verified'
    assert result['outcomes'][2]['status']=='not_started' and result['evidence_finalized']


@pytest.mark.parametrize('kind',['sources','reference','tapes','counters'])
def test_terminal_integrity_failure_is_not_a_complete_run(tmp_path,kind):
    result,store,p,_=lifecycle_fixture(tmp_path)
    path=tmp_path/'selected.json';path.write_text('{"engineering":true}')
    admitted=hashlib.sha256(path.read_bytes()).hexdigest()
    tape=np.array([1.,2.],np.float32);admitted_tape=o.array_record(tape)
    def work(*args):
        if kind in ('sources','reference'):path.write_text('{"altered":true}')
        if kind=='tapes':tape[0]=3
    def check(checks):
        def frozen(sha):
            r.require_identity(hashlib.sha256(path.read_bytes()).hexdigest(),admitted,kind)
            return p
        if kind=='counters':result['outcomes'][0]['native_work']['optimizer_returns']=9
        r.terminal_integrity(p,result['outcomes'],'0'*64,
            lambda:r.require_identity(o.array_record(tape),admitted_tape,'tape'),checks,frozen)
    with pytest.raises(Exception):r.finish_execution(result,store,work,check,p,engineering_runtime)
    assert result['status']=='failed' and result['cleanup_verified'] and not result['scientific_work_completed']
    assert result['failure_stage']=='terminal_integrity'
    key='immutable_tapes' if kind=='tapes' else 'work_counters' if kind=='counters' else 'frozen_sources_and_evidence'
    assert result['terminal_integrity'][key]['status']=='failed'
    assert json.loads((store.root/'finalization/seal.json').read_text())['outcome']=='failed'


@pytest.mark.parametrize('target',['result.json','finalization/seal.json'])
def test_finalization_failure_never_returns_complete_metadata(tmp_path,monkeypatch,target):
    result,store,p,integrity=lifecycle_fixture(tmp_path);original=store.json
    original('partial/record.json',dict(engineering=True))
    def write(name,value):
        if name==target:raise OSError('synthetic finalization failure')
        return original(name,value)
    monkeypatch.setattr(store,'json',write)
    with pytest.raises(OSError):r.finish_execution(result,store,lambda *args:None,integrity,p,engineering_runtime)
    assert result['status']=='failed' and not result['evidence_finalized']
    assert result['cleanup_verified'] and result['scientific_work_completed']
    assert (store.root/'partial/record.json').exists() and not (store.root/'finalization/seal.json').exists()
    if target.endswith('seal.json'):
        assert not json.loads((store.root/'result.json').read_text())['evidence_finalized']


@pytest.mark.parametrize('kind',['matrix','native','work','hooks'])
def test_exact_ten_arm_counters_are_mandatory(kind):
    rows,p=metadata_rows()
    if kind=='matrix':rows[-1]=copy.deepcopy(rows[0])
    if kind=='native':rows[0]['native_work']['shader_completed']=11
    if kind=='work':rows[0]['work']['backwards']=10
    if kind=='hooks':rows[0]['hooks_removed']=False
    with pytest.raises(Exception):r.verify_completed_work(rows,p)


@pytest.mark.parametrize('kind',['exit','xml','inventory','stdout'])
def test_validation_rejects_incomplete_or_corrupt_test_evidence(kind):
    from experiments import freeze_neural_material_early_step_e2a as f
    value=dict(scientific_model_calls=0,scientific_renderer_calls=0,scientific_optimizer_updates=0,tests=validation_fixture())
    if kind=='exit':value['tests']['process_exit_code']=1
    if kind=='xml':value['tests']['junit_xml']+='corrupt'
    if kind=='inventory':value['tests']['names']=['wrong']
    if kind=='stdout':value['tests']['stdout']='altered'
    with pytest.raises(Exception):f.validate_validation(value)


def test_scientific_loop_and_primitives_are_unchanged_ast():
    old=subprocess.check_output(['git','show','e0b29b89b97e4c4b226a379ad66e3d634d6040a8:experiments/neural_material_early_step_e2a.py'],text=True)
    new=open(r.__file__).read()
    def node(source,name,kind=ast.FunctionDef):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,kind) and n.name==name)
    assert ast.dump(node(old,'run_arm'))==ast.dump(node(new,'run_arm'))
    assert ast.dump(node(old,'Core',ast.ClassDef))==ast.dump(node(new,'Core',ast.ClassDef))
    def loop(source):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.For) and ast.dump(n.iter)=='Name(id=\'CASES\', ctx=Load())')
    assert ast.dump(loop(old))==ast.dump(loop(new))


def test_previous_freeze_and_validation_remain_hash_identical():
    from experiments import freeze_neural_material_early_step_e2a as f
    assert f.FREEZE!=f.PREVIOUS_FREEZE and f.VALIDATION!=f.PREVIOUS_VALIDATION
    assert f.sha(f.ROOT/f.PREVIOUS_FREEZE)==f.PREVIOUS_SHA
    assert f.sha(f.ROOT/f.PREVIOUS_VALIDATION)==f.PREVIOUS_VALIDATION_SHA
