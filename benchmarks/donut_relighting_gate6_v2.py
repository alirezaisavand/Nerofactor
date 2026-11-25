"""Frozen material replay correction for the unchanged Gate 6 relighting comparison."""
from pathlib import Path

import numpy as np

from benchmarks import donut_relighting_gate6_v1 as v1
from benchmarks.donut_relighting_gate6_v1 import (
    ARMS, CAMERAS, CHUNK, DIRECTIONS, DIRECTION_CHUNK, DRAWS, FRAME_ATOL, HIGH, LIGHT, MATERIAL_ATOL,
    POINT_BATCH, RESOURCES, ROLES, ROOT, SIZE, STEP, TENSOR_ARMS, admit, base_paths, check, digest, endpoint_paths,
    gate5, physical_axes, read, refine, tape, tensor_from_axes,
)
from infrastructure.signatures import require


PROTOCOL = 'donut-unseen-light-gate6-v2'
FREEZE = 'docs/protocols/DONUT_RELIGHTING_GATE6_V2.freeze.json'
V1_SHA = '1d418aa790a68a26257be5ed759467b9cfda3d173fbd92f3bae8e7ac349a97a1'
FAILURE = 'docs/results/publication_gate6/v1-failure.json'
V1_RECEIPT = '5ae147dc86f002a45eac9b1b56709e7a5b42f85099e129df2928860afb0bc51d'
BATCH_POLICY = 'replay_Gate5_matched_batches_then_separate_extra_foreground_v2'
LIMITS = dict(v1.LIMITS,material_field_calls=len(ARMS)*sum((n+CHUNK-1)//CHUNK+1 for n in v1.MATCHED),
              material_field_points=len(ARMS)*sum(v1.COUNTS))
SOURCES = (
    'benchmarks/donut_relighting_gate6_v2.py', 'experiments/donut_relighting_gate6_v2.py',
    'experiments/analyze_donut_relighting_gate6_v2.py', 'experiments/run_donut_relighting_gate6_v2.sh',
    'tests/scientific/test_donut_relighting_gate6_v2.py', 'docs/protocols/DONUT_RELIGHTING_GATE6_V2.md',
    FAILURE, v1.FREEZE,
)


def definition(out):
    parent = v1.freeze(V1_SHA)
    failed = read(ROOT/FAILURE)
    require(failed['status']=='failed' and failed['files']['run/receipt.json']['sha256']==V1_RECEIPT
            and failed['files']['freeze.json']['sha256']==V1_SHA
            and failed['counters']['optimizer_updates']==failed['counters']['independent_BRDF_point_directions']==0,
            'Missing exact failed launch evidence')
    return dict(parent,protocol=PROTOCOL,host_output=str(out),limits=LIMITS,
                v1_freeze_sha256=V1_SHA,v1_failure_receipt_sha256=V1_RECEIPT,
                material_batch_policy=BATCH_POLICY,
                sources={**parent['sources'], **{name:digest(ROOT/name) for name in SOURCES}})


def freeze(sha):
    path = ROOT/FREEZE
    require(digest(path)==sha,'Gate 6 v2 freeze identity differs')
    f=read(path)
    out=Path(f['host_output'])
    require(out.resolve()==out and out.parent==Path('/home/isavand/paper/evidence')
            and out.name.startswith('donut-relight-gate6-v2-') and f==definition(out),'Gate 6 v2 definition differs')
    return f


def authorization(f,commit,sha):
    return dict(v1.authorization(f,commit,sha),protocol=PROTOCOL,limits=LIMITS,
                v1_failure_receipt_sha256=V1_RECEIPT,material_batch_policy=BATCH_POLICY)


def layout(view):
    """Exact historical material-query shape and order, then separate extra rows."""
    pixels, matched = (view[k] for k in ('pixel_indices','matched_pixels'))
    require(0<len(matched)<=len(pixels) and (np.diff(pixels)>0).all() and (np.diff(matched)>0).all(),
            'Expected increasing unique relighting supports')
    keep = np.searchsorted(pixels,matched)
    require((keep<len(pixels)).all() and np.array_equal(pixels[keep],matched),'Archived pixels missing')
    extra = np.setdiff1d(np.arange(len(pixels)),keep,assume_unique=True)
    return (keep,extra)


def batch_record(view):
    return {name:[min(CHUNK,len(rows)-i) for i in range(0,len(rows),CHUNK)]
            for name,rows in zip(('archived','extra'),layout(view))}


def reconstruct_material(view,predict):
    """Scatter material-only query results into the unchanged relighting raster order."""
    fields = {}
    count=len(view['pixel_indices'])
    for panel in layout(view):
        for begin in range(0,len(panel),CHUNK):
            rows=panel[begin:begin+CHUNK]
            values=predict(rows)
            require(values and (not fields or set(values)==set(fields)),'Changed material query fields')
            for key,value in values.items():
                require(len(value)==len(rows) and np.isfinite(value).all(),'Invalid material query field '+key)
                if key not in fields:
                    fields[key]=np.empty((count,*value.shape[1:]),value.dtype)
                require(fields[key].shape[1:]==value.shape[1:] and fields[key].dtype==value.dtype,
                        'Changed material query dimensions or dtype')
                fields[key][rows]=value
    return fields


def archived_q(mx,my,encoded_tangent,encoded_bitangent):
    """Exact float32 operation order of the historical material export."""
    t,b=2*encoded_tangent-1,2*encoded_bitangent-1
    return mx[:,:,None]**2*t[:,:,None]*t[:,None,:]+my[:,:,None]**2*b[:,:,None]*b[:,None,:]


def field_parity(arm,view,fields,old):
    pixels=view['matched_pixels']
    i,_=layout(view)
    mask=old['hit_mask']
    np.testing.assert_array_equal(np.flatnonzero(mask),pixels)
    qkey='Q_world' if TENSOR_ARMS[arm] is None else 'Q_tensor_direct'
    shifts,witnesses={},{}
    for name,key in (('Q_world',qkey),('color','kd'),('metallic','metallic')):
        a,b=fields[name][i],old[key][mask]
        require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                'Unexpected historical physical field layout '+name)
        per_pixel=abs(a.astype(np.float64)-b.astype(np.float64)).reshape(len(i),-1).max(1)
        row=int(per_pixel.argmax())
        shifts[name]=float(per_pixel[row])
        witnesses[name]=int(pixels[row])
    return dict(common_rows=len(i),field_max_abs=shifts,maximum_error_pixels=witnesses,
                batches=batch_record(view),passed=max(shifts.values())<=MATERIAL_ATOL)


def replay_source(name, replacements):
    """Apply exactly identified adapter sites to the accepted V1 source in memory."""
    path=ROOT/name
    require(digest(path)==v1.freeze(V1_SHA)['sources'][name],'V1 source identity differs')
    raw=path.read_text()
    for old,new in replacements:
        require(raw.count(old)==1,'Expected one V1 adapter site: '+old.splitlines()[0])
        raw=raw.replace(old,new,1)
    return raw
