"""Tiny metadata fixtures only; no Torch or scientific reruns."""
from copy import deepcopy
import math
import subprocess
import sys

import pytest

from experiments.verify_bell_material_smoke_e12b import ACTIVE,assess_updates
from infrastructure.signatures import SignatureError


@pytest.fixture
def metadata():
    groups=ACTIVE+('outer_light','inner_light');names=['shader_network.'+g+'.p' for g in groups]
    record=lambda sha:dict(shape=[1],dtype='float32',numel=1,sha256=sha,l2=.01)
    parameters={name:record('initial') for name in names}
    group=dict(lr=.001,betas=[.9,.999],eps=1e-4,weight_decay=0,amsgrad=True,parameter_names=names)
    r=dict(optimizer_calls_completed=2,cleanup_errors=[],observer_lifecycle=dict(guards_prepared_before_run=True,method_bindings=1),
           initial_parameters=deepcopy(parameters),initial_optimizer={'class':'torch.optim.adam.Adam','groups':[group],'state':{}},updates=[])
    for step in (0,1):
        grads={name:record('gradient') if g in ACTIVE else None for name,g in zip(names,groups)}
        stats={g:dict(parameters=1,gradients_present=int(g in ACTIVE),gradient_l2_cpu64=.01 if g in ACTIVE else 0.,
                      expected_active=g in ACTIVE,status='active' if g in ACTIVE else 'expected_inactive') for g in groups}
        after={name:record('changed') if step==1 and g in ACTIVE else deepcopy(parameters[name]) for name,g in zip(names,groups)}
        changes={name:dict(changed_elements=int(step==1 and g in ACTIVE),l2=1e-7 if step==1 and g in ACTIVE else 0.,max_abs=1e-7 if step==1 and g in ACTIVE else 0.) for name,g in zip(names,groups)}
        state={name:dict(step=step+1,exp_avg=record('m'),exp_avg_sq=record('v'),max_exp_avg_sq=record('vmax')) for name,g in zip(names,groups) if g in ACTIVE}
        calls={name:dict(entered=1,returned=1) for name in ['shader.'+n for n in ('forward','anisotropic_forward','shade_anisotropic_mixed','sample_aniso_ggx_directions','compute_tangent_bitangent_flat','compute_radiance')]+['renderer.compute_rgb_loss','loss.nerf_render']}
        u=dict(step=step,update_index=step+1,lr=[0.,5e-7][step],forward_path_verified=True,backward_returned=True,
               losses={'loss_rgb':.09},aggregate_loss=.09,calls=calls,parameters_before=deepcopy(parameters),parameters_after=after,
               gradients_before_clip=grads,gradients_after_clip=grads,gradient_l2_before_clip_cpu64=math.sqrt(6*.01**2),
               gradient_l2_after_clip_cpu64=math.sqrt(6*.01**2),groups_before_clip=stats,groups_after_clip=stats,clip_max_norm=.7,
               optimizer={'class':'torch.optim.adam.Adam','groups':[{**group,'lr':[0.,5e-7][step]}],'state':state},parameter_changes=changes,
               changed_elements=sum(v['changed_elements'] for v in changes.values()),group_changed_elements={g:int(step==1 and g in ACTIVE) for g in groups},
               cursor_after=(step+1)*512,sampling={'ray_count':512},elapsed_seconds=.1,memory_after={'peak_allocated_bytes':1})
        r['updates'].append(u);parameters=after
    return r


def test_two_recorded_returns_and_zero_rate_displacement(metadata):
    rows,groups=assess_updates(metadata)
    assert [r['changed_elements'] for r in rows]==[0,6]
    assert [r['Adam_step'] for r in rows]==[1,2] and len(groups)==16


@pytest.mark.parametrize('error',['missing_group','inactive_gradient','Adam_step','first_displacement','second_displacement','nonfinite_loss'])
def test_corrupt_or_incomplete_update_evidence_rejected(metadata,error):
    a,b=metadata['updates'];name='shader_network.mx_predictor.p'
    if error=='missing_group':del a['groups_before_clip']['mx_predictor']
    elif error=='inactive_gradient':a['gradients_before_clip']['shader_network.outer_light.p']=a['gradients_before_clip'][name]
    elif error=='Adam_step':b['optimizer']['state'][name]['step']=1
    elif error=='first_displacement':a['parameters_after'][name]['sha256']='different'
    elif error=='second_displacement':b['changed_elements']=0
    else:a['losses']['loss_rgb']=float('nan')
    with pytest.raises(SignatureError):assess_updates(metadata)


def test_offline_verifier_import_blocks_scientific_modules():
    script='''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','network','train','raytracing','_raytracing','open3d'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
import experiments.verify_bell_material_smoke_e12b
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True)
