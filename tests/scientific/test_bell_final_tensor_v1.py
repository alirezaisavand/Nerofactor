import ast

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from benchmarks import bell_final_tensor_v1 as f
from experiments import bell_final_tensor_admission_v1 as admission
from experiments import bell_final_tensor_trainer_v1 as overlay
from scientific_variants import donut_tensor_initialization_d9 as d9
from scientific_variants.donut_tensor_shader_d8 import tensor_shader
from test_donut_tensor_d8 import SyntheticCore


def test_final_recipe_uses_k1_objective_without_spatial_work():
    prior = f.b.read(f.ROOT/f.dense.FREEZE)
    base = prior['arms']['K1-Dense']
    c,t = [f.recipe(arm,prior,{'arm':arm}) for arm in f.ARMS]
    assert c['work']==t['work'] and c['limits']==t['limits']
    assert c['spatial_policy'] is t['spatial_policy'] is None
    assert c['work']['component_reverse_sweeps']==base['work']['component_reverse_sweeps']==30
    assert c['work']['shader_calls']-base['work']['shader_calls']==2
    assert not any(k.startswith('surface_q') for k in c['work'])
    assert all(c['limits'][k]==v for k,v in base['limits'].items()
               if k not in ('shader_calls','shadow_rays'))
    assert c['representation']==t['representation']==f.d9.REPRESENTATION


def test_reviewed_overlay_is_reversible_and_base_remains_pinned():
    modified = overlay.transformed_source()
    for before,after in reversed(overlay.EDITS):
        assert modified.count(after)==1
        modified = modified.replace(after,before,1)
    assert modified.encode()==(overlay.ROOT/overlay.SOURCE).read_bytes()
    parsed = ast.parse(overlay.transformed_source(),feature_version=(3,8))
    assert any(isinstance(n,ast.FunctionDef) and n.name=='train' for n in parsed.body)


def paired():
    cls=d9.weak_start_shader(tensor_shader(SyntheticCore,lambda x:x))
    models=[]
    for method in f.MATCH.values():
        torch.manual_seed(6033)
        model=cls()
        model.select_tensor_representation(method)
        models.append(model)
    rng=torch.Generator().manual_seed(442)
    points=torch.rand((24,3),generator=rng)
    normals=F.normalize(torch.randn((24,3),generator=rng),dim=-1)
    normals[0]=torch.tensor([1.,0.,0.])
    return models,points,normals


def test_paired_math_gpu_free_and_admission_leaves_no_gradients(tmp_path):
    models,points,normals=paired()
    calls=[]
    with torch.no_grad():
        report=admission.tensor_admission(*models,points,normals,normals,tmp_path,
                                          lambda key,n:calls.append((key,n)),torch)
    assert report['passed'] and report['points']==24 and report['directions_per_point']==16
    assert all(x>0 for row in report['gradient_norms'].values() for x in row.values())
    assert all(p.grad is None for model in models for p in model.parameters())
    assert calls==[('tensor_admission_predictor_calls',1),('tensor_admission_reverse_sweeps',1)]*2
    with np.load(tmp_path/'tensor-physical-admission.npz',allow_pickle=False) as data:
        assert data['D9-C_pdf'].shape==(24,16,1)
        np.testing.assert_allclose(data['D9-C_pdf'],data['D9-T_pdf'],atol=1e-8,rtol=1e-6)


def test_paired_math_rejects_different_brdf(tmp_path):
    models,points,normals=paired()
    with torch.no_grad():
        models[1].mx_predictor[6].bias.add_(.1)
    with pytest.raises(ValueError,match='Paired fixed-direction BRDF'):
        admission.tensor_admission(*models,points,normals,normals,tmp_path,lambda *_:None,torch)
    assert not list(tmp_path.iterdir())


def test_final_launcher_passes_frozen_flags_and_stops_on_failed_first_arm(tmp_path):
    import os
    import subprocess
    from hashlib import sha256

    repo = tmp_path/'source'
    freeze = repo/f.FREEZE
    freeze.parent.mkdir(parents=True)
    freeze.write_text('{}\n')
    auth = tmp_path/'fixture-auth.json'
    auth.write_text('{}\n')
    batch = tmp_path/'batch'
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    argv = tmp_path/'argv'

    def stub(name, body):
        path = bin_dir/name
        path.write_text('#!/bin/sh\n'+body+'\n')
        path.chmod(0o755)

    stub('id','printf "1000\\n"')
    stub('git','case "$3" in status) exit 0;; rev-parse) printf "fixture-commit\\n";; *) exit 2;; esac')
    stub('python3','cat >/dev/null\nprintf "%s\\n" "$FIXTURE_PREFLIGHT"')
    stub('docker',"""case "$1" in
    image) printf '%s\\n' "$FIXTURE_IMAGE";;
    ps) exit 0;;
    run) printf '%s\\n' "$@" > "$FIXTURE_ARGV"; exit 23;;
    *) exit 2;;
esac""")
    mounts = ('oracle.ply','reference','admission.json','verification.json','base-initial.json')
    preflight = '\n'.join([str(batch),f.IMAGE,*(f'{tmp_path/m}\t/inputs/{m}' for m in mounts),
                          *(str(tmp_path/(arm+'-initial.json')) for arm in f.ARMS)])
    env = dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],BELL_TRAIN_SOURCE=str(repo),
               BELL_TRAIN_AUTH=str(auth),FIXTURE_PREFLIGHT=preflight,FIXTURE_IMAGE=f.IMAGE,FIXTURE_ARGV=str(argv))
    script = f.ROOT/'experiments/run_bell_final_tensor_v1.sh'
    result = subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True,timeout=15)
    args = argv.read_text().splitlines()
    assert result.returncode!=0 and 'unbound variable' not in result.stderr
    assert args[args.index('--freeze')+1]=='/workspace/'+f.FREEZE
    assert args[args.index('--freeze-sha256')+1]==sha256(freeze.read_bytes()).hexdigest()
    assert args[args.index('--phase')+1]=='Final-Chart' and '--execute' in args
    assert (batch/'Final-Chart-exit-code.txt').read_text().strip()=='23'
    assert not (batch/'Final-Ambient-exit-code.txt').exists()


def test_width_angle_diagnostic_decomposes_squared_relative_q():
    from experiments.analyze_bell_final_tensor_v1 import width_angle_error
    theta = np.radians(37)
    c,s = np.cos(theta),np.sin(theta)
    transform = np.array([[c,-s],[s,c]])
    est,truth = np.array([[.52,.23]]),np.array([[.48,.17]])
    a = transform@np.diag(est[0]**2)@transform.T
    b = np.diag(truth[0]**2)
    width,angle = width_angle_error(a[None],b[None])
    expected_width = np.square(est**2-truth**2).sum()/np.square(b).sum()
    np.testing.assert_allclose(width,expected_width,atol=1e-15,rtol=0)
    np.testing.assert_allclose(width+angle,np.square(a-b).sum()/np.square(b).sum(),atol=1e-15,rtol=0)
