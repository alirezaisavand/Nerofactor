"""Exact Gate 8 inputs and physical-validity accounting."""
from pathlib import Path

import numpy as np

from benchmarks import donut_nero_gate7_v1 as gate7
from benchmarks.donut_recovery_v1 import numeric
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS

ROOT, m, g6 = gate7.ROOT, gate7.m, gate7.gate6
read, check, digest = gate7.read, gate7.check, gate7.digest
ARM, CONTROL = 'A-Original-Oracle', 'K1-Oracle'
VARIANT = 'A_original_gate8_acquisition_v1'
PROTOCOL = 'donut-thesis-attribution-gate8-v1'
FREEZE = 'docs/protocols/DONUT_THESIS_ATTRIBUTION_GATE8_V1.freeze.json'
GATE7_SHA = 'bbc32d09a622ad8363c7c8a62737ac51ab10ce327990275b9de9c87340c74add'
WORK = m.work(False)
RELIGHT = dict(shader_calls=g6.DRAWS*sum((n+g6.CHUNK-1)//g6.CHUNK for n in g6.v1.COUNTS),
               shadow_rays=g6.DRAWS*sum(g6.v1.COUNTS)*1536)
RESOURCES = dict(g6.RESOURCES, watchdog_seconds=3600, inner_seconds=3540, output_bytes=8*1024**3)
FRAME_ATOL, RAW_NORM_MIN = g6.FRAME_ATOL, 1e-6
SOURCES = (
    'benchmarks/donut_thesis_attribution_gate8_v1.py',
    'experiments/train_donut_thesis_attribution_gate8_v1.py',
    'experiments/analyze_donut_thesis_attribution_gate8_v1.py',
    'experiments/run_donut_thesis_attribution_gate8_v1.sh',
    'tests/scientific/test_donut_thesis_attribution_gate8_v1.py',
    'docs/protocols/DONUT_THESIS_ATTRIBUTION_GATE8_V1.md',
    'docs/results/publication_gate8/README.md',
    'scientific_sources/manifest-v1.json',
    'scientific_sources/material_a/network/field.py',
    'scientific_sources/material_a/network/renderer.py',
    'scientific_sources/material_a/train/lr_common_manager.py',
    'scientific_variants/material_a_v1.py',
    'infrastructure/known_l2_audit.py',
    'infrastructure/material_variants.py',
    'infrastructure/sphere_visuals.py',
    'experiments/train_donut_matched_geometry_v1.py',
    'experiments/donut_material_recovery_v1.py',
    m.FREEZE, gate7.FREEZE,
)


def definition(output):
    from experiments.train_donut_thesis_attribution_gate8_v1 import trainer_source
    from hashlib import sha256

    frozen = gate7.freeze(GATE7_SHA)
    parent = gate7.gate5.parent()
    control = read(gate7.GATE5_MANIFEST)['arms'][CONTROL]['receipt']
    recipe = dict(parent['arms'][CONTROL], trainer_sha256=sha256(trainer_source().encode()).hexdigest())
    require(recipe['work']==WORK and recipe['limits']['inner_seconds']==RESOURCES['inner_seconds']
            and recipe['limits']['output_bytes']==RESOURCES['output_bytes'], 'Gate 4 work or resource limits differ')
    return dict(protocol=PROTOCOL, arm=ARM, variant=VARIANT, host_output=str(output),
                original_material_pin=PINS['material'], arm_recipe=recipe, image=frozen['image'],
                backend=frozen['backend'], support=frozen['support'], calibration=frozen['illuminations'],
                generation_sha256=m.GENERATION, admission_sha256=m.ADMISSION,
                initial=parent['base_initial'], K1_receipt=control, gate7_freeze_sha256=GATE7_SHA,
                relight_panels=frozen['panels'], work=WORK, relight_work=RELIGHT, resources=RESOURCES,
                steps=m.STEPS, sampling=dict(diffuse=512,specular=1024,relight_draws=g6.DRAWS,chunk=g6.CHUNK),
                primary_rays=0,new_reference_integrations=0,authorized_historical_checkpoint_loads=0,
                own_checkpoint_roundtrips=5, frame_atol=FRAME_ATOL, raw_orientation_norm_min=RAW_NORM_MIN,
                sources={name:digest(ROOT/name) for name in SOURCES})


def freeze(path, sha):
    require(path==ROOT/FREEZE and path.resolve()==path and digest(path)==sha,'Gate 8 freeze identity differs')
    f=read(path)
    out=Path(f['host_output'])
    require(out.parent==Path('/home/isavand/paper/evidence') and out.resolve()==out
            and out.name.startswith('donut-thesis-attribution-gate8-')
            and f==definition(out), 'Gate 8 definition differs')
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL, approved=True, arm=ARM, variant=VARIANT, commit=commit, freeze_sha256=sha,
                image_id=f['image'], selected_gpu='0', maximum_execution_launches=1, host_output=f['host_output'],
                researcher_admitted_dataset=True, benchmark_generation_sha256=m.GENERATION,
                dataset_admission_sha256=m.ADMISSION, gate3_receipt_sha256=m.GATE3_RECEIPT,
                optimizer_returns=10000, work=WORK, relight_work=RELIGHT, resources=RESOURCES,
                own_checkpoint_roundtrips=5, historical_checkpoint_loads_authorized=False)


def paths(f, native=False):
    base=gate7.paths(gate7.freeze(f['gate7_freeze_sha256']),native)
    del base['upstream']
    return base


def relight_reference(f, paths, views):
    panels={}
    for camera in g6.CAMERAS:
        a=numeric(check(paths[f'unseen_{camera}'],f['relight_panels'][str(camera)]))
        require(a['mask_common'].dtype==bool and a['reference'].shape==(g6.SIZE**2,3,3)
                and a['mask_expanded_foreground'].dtype==bool and a['mask_full_raster'].all()
                and np.array_equal(a['pixels'],np.flatnonzero(a['mask_expanded_foreground']))
                and np.array_equal(np.flatnonzero(a['mask_common']),views[camera]['pixel_indices']),
                'Gate 6 common support differs')
        panels[camera]=a
    return panels


def relight_geometry(paths,panels):
    old=g6.freeze(gate7.GATE6_SHA)
    views={}
    for camera,panel in panels.items():
        raw=numeric(check(paths['reference']/f'view-{camera}.npz',old['artifacts'][f'view_{camera}']))
        rows=np.searchsorted(raw['pixel_indices'],panel['pixels'])
        require(np.array_equal(raw['pixel_indices'][rows],panel['pixels']) and raw['shader_valid'][rows].all(),
                'Expanded Gate 6 relight geometry differs')
        views[camera]={key:raw[key][rows] for key in ('position','normals','outgoing','pixel_indices')}
    return views


def control_receipt(f):
    artifact=f['K1_receipt']
    path=check(Path(artifact['path']),artifact)
    receipt=read(path)
    require(receipt['status']=='complete' and receipt['cleanup_verified'] and receipt['counters']==WORK
            and receipt['optimizer_returns']==10000 and receipt['arm']==CONTROL
            and receipt['matched_support']==f['support']
            and receipt['outputs']['photometric-indices.u32le']['sha256']==f['support']['photometric_index_sha256'],
            'Accepted K1 comparison receipt differs')
    return path.parent,receipt


def vector_validity(mx,my,raw,t,b,n,expected_n):
    """Reject nonphysical original frames as evaluation observations. Never repair them."""
    unit=lambda v: abs(np.sum(v*v,axis=-1)-1)<=FRAME_ATOL
    dot=lambda a,b: abs(np.sum(a*b,axis=-1))<=FRAME_ATOL
    return dict(
        raw_orientation=np.linalg.norm(2*raw-1,axis=-1)>=RAW_NORM_MIN,
        normal=unit(n) & (abs(n-expected_n).max(axis=-1)<=FRAME_ATOL),
        axes=unit(t) & unit(b) & dot(t,b) & dot(n,t) & dot(n,b)
             & (abs(np.cross(t,b)-n).max(axis=-1)<=FRAME_ATOL),
        widths=((mx>=.04)&(mx<=1)&(my>=.04)&(my<=1)).reshape(-1),
    )


def frame_checks(a):
    mask=a['hit_mask']
    encoded=[2*a[k][mask]-1 for k in ('tangents','bitangents','normals')]
    return vector_validity(a['mx'][mask],a['my'][mask],a['rotation_raw'][mask],*encoded,a['normal'][mask])


def valid_mask(a):
    mask=a['hit_mask']
    valid=np.zeros_like(mask)
    valid[mask]=np.logical_and.reduce(list(frame_checks(a).values()))
    return valid


def completed(batch,f,sha):
    path=batch/'authorization.json'
    auth=read(path)
    require(auth==authorization(f,auth['commit'],sha) and digest(batch/'freeze.json')==sha
            and (batch/f'{ARM}-exit-code.txt').read_text().strip()=='0'
            and not (batch/'cleanup.txt').exists(), 'Exact complete foreground launch evidence required')
    root=batch/ARM
    r=read(root/'recovery.json')
    require(r['status']=='complete' and r['cleanup_verified'] and r['scientific_work_complete']
            and r['arm']==ARM and r['commit']==auth['commit'] and r['freeze_sha256']==sha
            and r['authorization_sha256']==digest(path) and r['optimizer_returns']==10000 and r['counters']==WORK
            and r['unseen']['work']==RELIGHT and r['unseen']['state_unchanged']
            and [item['step'] for item in r['saved_endpoints']]==m.STEPS, 'Incomplete or mismatched Gate 8 run')
    for name,item in r['outputs'].items():
        check(root/name,item)
    require(r['outputs']['photometric-indices.u32le']['sha256']==f['support']['photometric_index_sha256'],
            'Gate 8 photometric-index bytes differ from K1')
    return root,r
