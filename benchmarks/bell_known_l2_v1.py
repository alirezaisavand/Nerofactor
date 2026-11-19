"""Known-illumination Bell controls using the admitted native recipes."""
from pathlib import Path

import numpy as np

from benchmarks.bell_heterogeneous_training_v1 import (
    ADMISSION, GENERATION, IMAGE, PANEL, POLICY, ROOT, SHADER, SPATIAL_POLICY, SPLITS, STEPS, TARGET_KEYS,
    VERIFICATION, b, boundary_score, check_domain_archive, check_probe_archive, check_sampler_capture,
    dataset, diagnostic_frames, diagnostics, fit_pool, geometry, numeric, point_errors,
)
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, baseline, digest

PROTOCOL = 'bell-kurt-known-L2-recovery-v1'
FREEZE = 'docs/protocols/BELL_KNOWN_L2_RECOVERY_V1.freeze.json'
CONTROL_FREEZE = 'docs/protocols/BELL_HETEROGENEOUS_TRAINING_V1.freeze.json'
CONTROL_FREEZE_SHA = '6c563597983ad80adbe235a7692e26089e6c24265fc6b2c9ec1b68197f6f10f7'
CONTROL_MANIFEST = 'docs/results/bell_heterogeneous_training_v1/artifact-manifest.json'
MATCH = {'K1':'D2','K2':'D8-C','K3':'D8-T'}
ARMS = tuple(MATCH)
FOREGROUND_SECONDS = 11160
LIGHT_POLICY = dict(direct='known_env_v1',indirect='zero_bounce_v1',environment=SHADER['known_environment'],
                    map_max_abs=2e-5,map_relative_L2=2e-6,diffuse_white_material_gradient=0.)


def recipe(control, original):
    # Six read-only white component probes at the original diagnostic steps.
    work = dict(original['work'],component_reverse_sweeps=original['work']['component_reverse_sweeps']+6)
    return dict(original,learned_control=control,illumination='known_L2',work=work,
                limits=dict(original['limits'],**work))


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha,'Frozen known-L2 execution differs')
    f = b.read(path)
    old = ROOT/CONTROL_FREEZE
    require(digest(old)==CONTROL_FREEZE_SHA,'Accepted learned-light freeze differs')
    base = b.read(old)
    inherited = {k:v for k,v in base.items() if k not in ('sources','arms','host_output','order','foreground_seconds','protocol')}
    require(all(f[k]==v for k,v in inherited.items()) and f['protocol']==PROTOCOL and f['order']==list(ARMS)
            and f['foreground_seconds']==FOREGROUND_SECONDS and f['lighting_policy']==LIGHT_POLICY
            and f['control_freeze_sha256']==CONTROL_FREEZE_SHA and f['host_output']!=base['host_output']
            and f['arms']=={k:recipe(v,base['arms'][v]) for k,v in MATCH.items()},
            'Known-L2 definition differs from approved controls')
    for name,expected in f['sources'].items():
        p = ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==expected,'Frozen known-light source differs: '+name)
    baseline('material',PINS['material'])
    return f


def authorization(f, commit, sha):
    from benchmarks import bell_heterogeneous_training_v1 as native
    return dict(native.authorization(f,commit,sha),protocol=PROTOCOL,order=list(ARMS),
                foreground_seconds=FOREGROUND_SECONDS,lighting_policy=LIGHT_POLICY)


def runtime(args):
    import sys
    from benchmarks import bell_heterogeneous_training_v1 as native
    return native.runtime(args,spec=sys.modules[__name__])


def verify_controls(f):
    accepted = b.read(ROOT/CONTROL_MANIFEST)
    require(f['controls']=={arm:accepted['arms'][method]['receipt'] for arm,method in MATCH.items()},
            'Foreign learned-light arm receipts')
    for arm,item in f['controls'].items():
        path = b.check_artifact(Path(item['path']),item)
        saved = b.read(path)
        require(saved['arm']==MATCH[arm] and saved['status']=='complete' and saved['scientific_work_complete']
                and saved['cleanup_verified'] and saved['optimizer_returns']==10000
                and saved['benchmark_sha256']==GENERATION and saved['dataset_admission_sha256']==ADMISSION
                and saved['freeze_sha256']==CONTROL_FREEZE_SHA
                and saved['effective_lighting']=='original A directional outer_light MLP'
                and saved['initial']==b.read(Path(f['arms'][arm]['initial']['path'])), 'Wrong learned-light control')


def route_plan(work, paired):
    return dict(compatibility=32,paired=2 if paired else 0,training=10000,
                evaluation=work['shader_calls']-10032-(2 if paired else 0),illumination_map=len(STEPS))


def verify_routes(routes, planned):
    require(routes.keys()==planned.keys(),'Unexpected known-L2 path inventory')
    for name,n in planned.items():
        row = routes[name]
        outer,inner = row['outer_calls'],row['inner_calls']
        require(row['dispatches']==n and 0<=outer<=n and 0<=inner<=n and (n==0 or outer>0)
                and (name!='illumination_map' or (outer==n and inner==0)),
                'Analytic illumination route not observed: '+name)


def validate_map(value, expected):
    require(value.shape==expected.shape and np.isfinite(value).all() and np.isfinite(expected).all()
            and (value>=0).all() and (expected>=0).all(),'Invalid known-light map')
    error = value.astype(np.float64)-expected
    abs_max = float(abs(error).max())
    relative = float(np.linalg.norm(error)/np.linalg.norm(expected))
    require(abs_max<=LIGHT_POLICY['map_max_abs'] and relative<=LIGHT_POLICY['map_relative_L2'],
            'Native illumination map differs from independent analytic L2')
    return dict(maximum_absolute=abs_max,relative_L2=relative)


def verify_illumination_receipt(root, run, recipe, history):
    require(run['effective_lighting']=='known analytic L2' and run['learned_control']==recipe['learned_control']
            and run['diffuse_white_in_objective'] and run['indirect_illumination']=='identically zero'
            and all(run['trainable_parameters'][key]==0 for key in ('outer_light','inner_light')),
            'Fixed-light training state differs')
    audit = run['known_lighting_audit']
    verify_routes(audit['routes'],route_plan(recipe['work'],recipe['representation'] is not None))
    snapshots = audit['checkpoints']
    require([r['step'] for r in snapshots]==STEPS and len({r['state_sha256'] for r in snapshots})==1
            and audit['learned_module_calls']==0
            and all(r['optimizer_light_parameters']==r['optimizer_light_states']==r['learned_light_gradients']==0
                    for r in snapshots)
            and all(r['gradient_groups']['outer_light']==r['gradient_groups']['inner_light']==0. for r in history),
            'Fixed-illumination gradients, optimizer or weights differ')
    probes = [r for r in history if 'gradient_interactions' in r]
    require([r['completed_updates'] for r in probes]==[1,*STEPS[1:]]
            and all(r['gradient_interactions']['terms']['diffuse_white']['gradient_l2']==0. for r in probes),
            'Fixed-illumination diffuse-white diagnostic differs')
    previous = None
    for endpoint in run['saved_endpoints']:
        step = endpoint['step']
        data = numeric(root/f'step-{step:05d}'/'illumination.npz')
        value,expected = (data[k] for k in ('linear_RGB','known_linear_RGB'))
        native = validate_map(value,expected)
        recorded = endpoint['metrics']['illumination']['analytic_agreement']
        np.testing.assert_allclose(list(native.values()),[recorded[k] for k in native],atol=2e-12,rtol=0)
        if previous is not None:
            np.testing.assert_array_equal(previous,value)
        previous = value
