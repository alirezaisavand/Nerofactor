import ast
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bell_known_l2_v1 as spec
from infrastructure.signatures import SignatureError


def test_frozen_arm_change_is_only_light_and_diagnostic_work():
    base = spec.b.read(spec.ROOT/spec.CONTROL_FREEZE)
    for arm,control in spec.MATCH.items():
        before = base['arms'][control]
        after = spec.recipe(control,before)
        assert after.pop('learned_control')==control
        assert after.pop('illumination')=='known_L2'
        after['work']['component_reverse_sweeps'] -= 6
        after['limits']['component_reverse_sweeps'] -= 6
        assert before==after
        route = spec.route_plan(before['work'],arm!='K1')
        assert route['evaluation']==2448 and route['illumination_map']==6


def test_bell_runtime_selects_frozen_tensor_alias(monkeypatch):
    from benchmarks import bell_heterogeneous_training_v1 as native

    args = SimpleNamespace(phase='K2',freeze='fixture',freeze_sha256='hash',image_id=spec.IMAGE,
        authorization='approval',authorization_sha256='hash',commit='commit',output=spec.Path('/outputs/K2'),
        oracle='oracle',benchmark='dataset',admission='admit',verification='check',initial='base',arm_initial='arm',
        initial_sha256='base',benchmark_sha256=spec.GENERATION)
    base = spec.b.read(spec.ROOT/spec.CONTROL_FREEZE)
    target = dict(base,base_initial={'sha256':'base'},arms={'K2':spec.recipe('D8-C',base['arms']['D8-C'])})
    # Exercise the real runtime's recipe dispatch with fake admission/mount metadata.
    monkeypatch.setattr(spec,'freeze',lambda *a:target)
    monkeypatch.setattr(spec,'authorization',lambda *a:{'approved':True})
    monkeypatch.setattr(native.b,'read',lambda *a:{'approved':True})
    monkeypatch.setattr(native.b,'check_artifact',lambda *a:None)
    monkeypatch.setattr(native,'digest',lambda *a:'hash')
    monkeypatch.setattr(native.os,'geteuid',lambda :1109)
    monkeypatch.setattr(native.os,'statvfs',lambda p:SimpleNamespace(f_flag=0 if p=='/outputs' else native.os.ST_RDONLY))
    real_read = spec.Path.read_text
    monkeypatch.setattr(spec.Path,'read_text',lambda p:'x\ny\n lo: 0\n'
                        if p.as_posix()=='/proc/net/dev' else real_read(p))
    f,_ = native.runtime(args,spec=spec)
    assert f['tensor_training']['learned_control']=='D8-C'
    assert f['tensor_training']['arms']==['D8-C','D8-T']
    assert 'surface_q_training' not in f


@pytest.fixture
def native_formulas():
    torch = pytest.importorskip('torch')
    import torch.nn.functional as F
    from infrastructure.stage_sources import PINS,baseline

    root,_ = baseline('material',PINS['material'])
    scope={'torch':torch,'F':F,'np':np}
    names={'get_lights','get_orthogonal_directions','fresnel_schlick','fresnel_schlick_directions',
           'compute_spec_loss','rotate_tangent_bitangent','diffuse_term','compute_diffuse_light_regularization'}
    nodes=[]
    for path,klass in (('network/field.py','MCShadingNetwork'),('network/renderer.py','NeROMaterialRenderer')):
        tree=ast.parse((root/path).read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==klass)
        nodes.extend(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names)
    assert {n.name for n in nodes}==names
    linear = next(n for n in ast.parse((root/'utils/raw_utils.py').read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='linear_to_srgb')
    exec(compile(ast.Module(body=nodes+[linear],type_ignores=[]),'fixture-native-lighting','exec'),scope)
    return torch,{k:scope[k] for k in names|{'linear_to_srgb'}}


@pytest.mark.parametrize('mode',['core','D8-C','D8-T'])
def test_both_renderers_use_native_fixed_light_and_white_has_no_material_gradient(native_formulas,mode):
    torch,formula = native_formulas
    from infrastructure.known_l2_audit import KnownL2Audit
    from scientific_variants.donut_tensor_shader_d8 import tensor_shader
    from scientific_variants.material_a_v1 import native_classes
    from scientific_variants.donut_tensor_d8 import frame,chart_q
    from benchmarks.bell_reference_e11 import environment

    def forbidden(*a):
        raise AssertionError('Unselected fallback')

    class Base(torch.nn.Module):
        def __init__(self,cfg,trace):
            super().__init__()
            self.cfg=cfg
            self.ray_trace_fun=trace
            self.outer_light=torch.nn.Linear(3,3)
            self.inner_light=torch.nn.Linear(3,3)
            self.diffuse_direction_samples=torch.tensor([[0.,0.],[.25,0.],[.5,0.],[.75,0.]])

    for k in ('predict_anisotropic_components','compute_tangent_bitangent_flat','sample_diffuse_directions',
              'sample_aniso_ggx_directions','compute_Dh','compute_pdf_aniso_ggx','compute_radiance',
              'shade_anisotropic_mixed','anisotropic_regularization'):
        setattr(Base,k,forbidden)
    for k in formula.keys()-{'linear_to_srgb','compute_diffuse_light_regularization'}:
        setattr(Base,k,formula[k])
    rgb=formula['linear_to_srgb']
    cls,_=native_classes(Base,torch.nn.Module,'A_corrected_core_v1',rgb)
    if mode!='core':
        cls=tensor_shader(cls,rgb)
    dots=[]
    def trace(p,d):
        dots.append(d)
        hit=torch.arange(len(d))%7==0
        return p,torch.zeros_like(p),p.new_ones(len(d)),hit
    model=cls(dict(spec.SHADER,diffuse_sample_num=4,specular_sample_num=8),trace)
    audit=KnownL2Audit()
    with audit.attached(model),audit.path('training'):
        p=torch.zeros((2,3))
        n=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
        mx=torch.full((2,1),.33,requires_grad=True)
        my=torch.full((2,1),.17,requires_grad=True)
        kd=torch.full((2,3),.4,requires_grad=True)
        metallic=torch.full((2,1),.6,requires_grad=True)
        r=torch.tensor([[1.,0.],[1.,0.]])
        if mode=='core':
            _,o=model.shade_anisotropic_mixed(p,n,n,mx,my,metallic,kd,r,(r+1)/2,None,True)
        else:
            u=frame(n)
            q=chart_q(mx,my,u[...,0],u)
            o=model.shade_tensor(p,n,n,q,u,kd,metallic,15,True)
        d=dots[-1]
        want=environment(d,spec.LIGHT_POLICY['environment'])
        want[torch.arange(len(d))%7==0]=0
        torch.testing.assert_close(o['diffuse_light'],rgb(want.reshape(2,12,3)[:,:4].mean(1)).clamp(0,1))
        white=formula['compute_diffuse_light_regularization'](SimpleNamespace(cfg={'reg_diffuse_light_lambda':.01}),
                                                           o['diffuse_light']).mean()
        variables=(mx,my,metallic,kd)
        grads=torch.autograd.grad(white,variables,allow_unused=True,retain_graph=True) if white.requires_grad else ()
        assert all(g is None or torch.eq(g,0).all() for g in grads)
        photo=torch.autograd.grad(o['linear_rgb'].sum(),variables)
        assert all(torch.isfinite(v).all() for v in photo) and torch.count_nonzero(photo[-1])>0
    assert audit.routes['training']==dict(dispatches=1,outer_calls=1,inner_calls=1)
    assert 'predict_outer_lights' not in vars(model)


def test_light_audit_excludes_frozen_modules_from_optimizer_and_rejects_override():
    torch=pytest.importorskip('torch')
    from infrastructure.known_l2_audit import KnownL2Audit

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.outer_light=torch.nn.Linear(3,3).requires_grad_(False)
            self.material=torch.nn.Linear(3,3)
            self.cfg=spec.SHADER
            self._environment=None

        def predict_outer_lights(self,p,d):
            return self.outer_light(d)

        def get_inner_lights(self,p,d,n):
            return torch.zeros_like(p)

    model=Fixture()
    audit=KnownL2Audit()
    anchor=audit.lighting_hash(model)
    case=torch.optim.Adam(model.material.parameters())
    audit.checkpoint(0,model,case,anchor)
    with pytest.raises(SignatureError,match='material-only optimizer'):
        audit.checkpoint(0,model,torch.optim.Adam(model.parameters()),anchor)
    p=torch.zeros((1,3))
    with audit.attached(model),audit.path('training'):
        with pytest.raises(ValueError,match='Learned light called'):
            model.predict_outer_lights(p,p)
        model._environment=object()
        with pytest.raises(SignatureError,match='routing changed'):
            model.predict_outer_lights(p,p)
    with torch.no_grad():
        model.outer_light.weight[0,0]+=1
    with pytest.raises(SignatureError,match='changed'):
        audit.checkpoint(1,model,case,anchor)


def test_map_and_route_validation_reject_wrong_results():
    m=np.ones((3,4,3),dtype=np.float32)
    assert spec.validate_map(m,m)==dict(maximum_absolute=0.,relative_L2=0.)
    with pytest.raises(SignatureError,match='differs'):
        spec.validate_map(m,m*1.01)
    with pytest.raises(SignatureError,match='not observed'):
        spec.verify_routes({'training':dict(dispatches=1,outer_calls=0,inner_calls=1)},{'training':1})
