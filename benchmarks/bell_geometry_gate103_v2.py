"""Versioned continuation of the stopped camera-prefix audit; no model imports."""
from benchmarks.bell_geometry_gate103 import *
from benchmarks import bell_geometry_gate103 as original

PROTOCOL='bell-geometry-evolution-gate10.3-v2'
FREEZE='docs/protocols/BELL_GEOMETRY_EVOLUTION_GATE103_V2.freeze.json'
FAILURE='docs/results/publication_gate103/failure-v1/artifact-manifest.json'
FAILED=Path(read(ROOT/FAILURE)['root'])
LIMITS=dict(original.LIMITS,dense_grids=2,dense_tiles=1024,dense_SDF_points=2*512**3,
    MC_calls=2,audit_BVH_rays=2*196608,CPU_ray_checks=2*384+4*64,
    consistency_SDF_points=2*1536,consistency_gradient_points=2*512,
    camera_precision_probe_calls=2)
NEURAL=dict(original.NEURAL,ray_builder='original CPU construction AND processing, then CUDA transfer',
    precision_witness='first selected camera only: original CUDA default and TF32-disabled prefix; never used for rendering')
SOURCES=tuple(dict.fromkeys((*original.SOURCES,original.FREEZE,FAILURE,
    'docs/results/publication_gate103/failure-v1/summary.json',
    'docs/results/publication_gate103/failure-v1/camera-cpu-replay.json',
    'benchmarks/bell_geometry_gate103_v2.py','experiments/bell_geometry_gate103_v2.py',
    'experiments/run_bell_geometry_gate103_v2.sh','tests/scientific/test_bell_geometry_gate103_camera_v2.py',
    'docs/protocols/BELL_GEOMETRY_EVOLUTION_GATE103_V2.md')))


def definition(out):
    f=original.definition(out)
    f.update(protocol=PROTOCOL,limits=LIMITS,neural=NEURAL,
        completed_15k=dict(root=str(FAILED),files=read(ROOT/FAILURE)['files']),
        sources={name:digest(ROOT/name) for name in SOURCES})
    return f


def freeze(sha):
    require(digest(ROOT/FREEZE)==sha,'Gate 10.3 v2 freeze differs')
    f=read(ROOT/FREEZE);out=Path(f['host_output'])
    require(out.resolve()==out and out.parent==TRAIN.parent.parent and
            out.name.startswith('bell-geometry-gate103-v2-') and f==definition(out),'Versioned continuation differs')
    return f


def completed_15k(f,root):
    root=Path(root)
    for name,identity in f['completed_15k']['files'].items():check(root/name,identity)
    r=read(root/'run/receipt.json');row=r['steps'][0]
    require(r['status']=='failed' and r['commit']=='d19f71fee36edd109a0999baed0536180f46423e' and
        r['image']==f['image'] and r['backend']==f['backend'] and
        r['freeze_sha256']=='8d0337fe10c02aa00f7ecf797ac02620888daa4e5a2766698ecf856918f20c82' and
        r['error']==dict(type='SignatureError',message='Neural/mesh rays differ') and
        r['observed']['loads']==1 and r['observed']['renders']==r['optimizer_returns']==0 and
        len(r['steps'])==1 and row['step']==15000 and row['MC_attempts']==row['MC_returns']==1 and
        row['grid_tiles']==512 and row['primary_BVH_rays']==196608 and row['CPU_ray_checks']==384 and
        len(row['views'])==6 and row['SDF_gate']['passed'] and
        row['checkpoint']==f['checkpoints']['15000']['identity'] and
        (root/'exit-code.txt').read_text().strip()=='1' and
        (root/'container-absent.txt').read_text().strip()=='absent','Unexpected or incomplete 15k capture')
    return row


def host_inputs(f):
    original.host_inputs(f)
    completed_15k(f,FAILED)


def authorization(f,commit,sha):
    result=original.authorization(f,commit,sha)
    result.update(protocol=PROTOCOL,limits=LIMITS,completed_15k=f['completed_15k'],
        camera_prefix=NEURAL['ray_builder'],precision_probe=NEURAL['precision_witness'])
    return result
