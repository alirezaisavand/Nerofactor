"""Resume the dense Bell reference from the verified camera-127 prefix."""
from pathlib import Path

import numpy as np

from benchmarks import bell_dense_reference_v1 as v1
from benchmarks import donut_heterogeneous_v2 as refinement
from infrastructure.signatures import require

b = v1.b
PROTOCOL = 'bell-kurt-heterogeneous-dense-reference-v2'
FREEZE = 'docs/protocols/BELL_DENSE_REFERENCE_V2.freeze.json'
V1_SHA = 'b72efa61da2596166b358f7d2452af46c64fa2331e35682b16569916fcb73842'
AUDIT = 'docs/results/bell_dense_reference_failure_v1/analysis.json'
CAMERA = 127
PREFIX = 32
CAP = 64
POLICY = dict(b.REFINEMENT, per_view=CAP)
NEW_BASELINE_POINTS = sum(v1.NEW_COUNTS)-v1.COUNTS[CAMERA]
LIMITS = dict(v1.LIMITS, **b.counters(NEW_BASELINE_POINTS, b.MAX_TOTAL-PREFIX))
FAILURE_FILES = ('freeze.json','authorization.json','exit-code.txt','container.cid','console.txt','reference/generation.json')


def prior_definition():
    return v1.freeze(b.ROOT/v1.FREEZE, V1_SHA)


def freeze(path, sha):
    require(path.resolve()==path and b.digest(path)==sha, 'Dense Bell v2 freeze differs')
    f = b.read(path)
    old = prior_definition()
    require(f['protocol']==PROTOCOL and f['previous_freeze_sha256']==V1_SHA
            and f['failed_batch']==old['host_output'] and f['host_output']!=old['host_output']
            and f['limits']==LIMITS and f['refinement']==POLICY and f['prefix_patches']==PREFIX
            and set(f['failure_files'])==set(FAILURE_FILES)
            and f['sources'][AUDIT]==f['audit_sha256'], 'Dense Bell continuation definition differs')
    for name, expected in f['sources'].items():
        p = b.ROOT/name
        require(p.resolve()==p and b.ROOT in p.parents and b.digest(p)==expected,
                'Frozen dense Bell v2 source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(v1.authorization(prior_definition(),commit,sha), protocol=PROTOCOL,
                host_output=f['host_output'], limits=LIMITS, refinement=POLICY,
                failed_generation_sha256=f['failure_files']['reference/generation.json']['sha256'],
                prefix_patches=PREFIX, previous_freeze_sha256=V1_SHA)


def admit_prefix(f, failed):
    for name in FAILURE_FILES:
        b.check_artifact(failed/name,f['failure_files'][name])
    original = prior_definition()
    auth = b.read(failed/'authorization.json')
    audit = b.read(b.ROOT/AUDIT)
    receipt = b.read(failed/'reference/generation.json')
    require(auth==v1.authorization(original,receipt['commit'],V1_SHA)
            and b.digest(failed/'freeze.json')==V1_SHA
            and receipt['authorization_sha256']==b.digest(failed/'authorization.json')
            and receipt['status']=='failed' and receipt['error']=='SignatureError: Per-view refinement budget exhausted'
            and receipt['phase']==f'camera_{CAMERA}' and receipt['refined_points']==PREFIX and not receipt['views']
            and not receipt['scientific_work_complete'] and receipt['cleanup_verified']
            and receipt['counters']==b.counters(v1.COUNTS[CAMERA],PREFIX)
            and audit['status']=='verified_incomplete' and audit['experiment_batch']==f['failed_batch']
            and audit['generation_sha256']==b.digest(failed/'reference/generation.json')
            and (failed/'exit-code.txt').read_text().strip()=='1', 'Unexpected previous dense Bell failure')
    ordered = [x['pixel'] for x in audit['deterministic_trace']]
    require(len(set(ordered))==len(ordered)==PREFIX, 'Invalid verified patch order')
    names = {f'baseline-{CAMERA}.npz',*(f'patch-{CAMERA}-{pixel}.npz' for pixel in ordered)}
    require(set(receipt['outputs'])==names|set(v1.SPARSE_FILES)|{'surface-area.npz'},
            'Unexpected failed reference outputs')
    records = {name:receipt['outputs'][name] for name in sorted(names)}
    for name, item in records.items():
        b.check_artifact(failed/'reference'/name,item)
    return ordered,records,receipt['counters']


def assemble(baseline, coarse_counts, fetch, prefix=()):
    """Same selector and patch math as v1, with a bounded 64-row allowance."""
    patched = ('linear_components','coarse_components','visible_counts',
               'incident_reflectance','coarse_incident_reflectance')
    view = dict(baseline,**{key:baseline[key].copy() for key in patched})
    valid = view['shader_valid']
    require(coarse_counts.shape==valid.shape and coarse_counts.dtype.kind=='i'
            and ((coarse_counts[valid]>=0)&(coarse_counts[valid]<=b.GRIDS[0]**2)).all()
            and (coarse_counts[~valid]==0).all(), 'Invalid dense coarse visibility inventory')
    grids = dict(pixel_indices=view['pixel_indices'].copy(), coarse_grid=np.where(valid,b.GRIDS[0],0),
                 fine_grid=np.where(valid,b.GRIDS[1],0), coarse_visible_counts=coarse_counts.copy())
    chosen = []
    while True:
        row = refinement.next_row(view, grids['fine_grid']==refinement.HIGH[1])
        if row is None:
            require(len(chosen)>=len(prefix), 'Convergence changed within the archived patch prefix')
            grids['refined_pixels'] = np.asarray(chosen,dtype=np.int64)
            return view,grids
        require(len(chosen)<CAP, 'Per-view refinement budget exhausted')
        pixel = int(view['pixel_indices'][row])
        index = len(chosen)
        if index<len(prefix):
            require(pixel==prefix[index], 'Archived patch order differs')
        refinement.apply_patch(view,grids,row,fetch(row,pixel,index))
        chosen.append(pixel)
