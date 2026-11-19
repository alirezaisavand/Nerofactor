import pytest

from benchmarks import bell_known_l2_completion_v2 as completion
from benchmarks import bell_known_l2_v1 as original
from experiments.donut_material_recovery_v1 import tensor_pair
from infrastructure.signatures import SignatureError


@pytest.mark.parametrize('phase,expected',[('K2',('D8-C','D8-T')),('K3',('D8-T','D8-C'))])
def test_known_light_alias_produces_both_native_initial_gate_captures(phase,expected,monkeypatch):
    torch = pytest.importorskip('torch')
    from benchmarks.donut_tensor_d8 import REPRESENTATION
    from scientific_variants.donut_tensor_d8 import frame
    from scientific_variants.donut_tensor_shader_d8 import paired_initial_gate

    class Fixture(torch.nn.Module):
        def __init__(self,name):
            super().__init__()
            self.tensor_arm=name

        def predict_tensor(self,points,normals):
            q=torch.eye(2,dtype=torch.float64).expand(len(points),2,2)*.25**2
            return q,frame(normals),points+.25,points[:,:1]+.5

        def shade_tensor(self,points,normals,outgoing,q,u,kd,metallic,seed,is_train):
            return dict(linear_rgb=kd,linear_diffuse=kd*.5,linear_specular=kd*.5,kd=kd,metallic=metallic)

    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda _:None)
    require_arms=['D8-C','D8-T']
    labels=tensor_pair(original.MATCH[phase],require_arms)
    assert labels==expected
    points=torch.zeros(24,3)
    normals=points.clone()
    normals[:,2]=1
    with torch.no_grad():
        gate,arrays=paired_initial_gate(*(Fixture(name) for name in labels),points,normals,normals,REPRESENTATION)
    assert gate['passed'] and {'D8-C_world_Q','D8-T_world_Q'}<=arrays.keys()


def test_pair_selector_rejects_aliases_and_duplicate_representations():
    for method,arms in (('K2',['D8-C','D8-T']),('D8-C',['D8-C','D8-C'])):
        with pytest.raises(SignatureError,match='Two distinct tensor'):
            tensor_pair(method,arms)


def test_completion_authorization_contains_only_remaining_recipes():
    prior=completion.prior_definition()
    draft=dict(prior,arms={k:prior['arms'][k] for k in completion.ARMS},prior_evidence={})
    a=completion.authorization(draft,'commit','freeze')
    assert a['order']==['K2','K3'] and set(a['recipes'])=={'K2','K3'}
    assert a['maximum_execution_launches']==1 and a['foreground_seconds']==7500
