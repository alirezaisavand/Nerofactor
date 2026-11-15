"""E11C sidecar backend contracts; metadata/CPU helpers, no native imports."""
from pathlib import Path
import json
import numpy as np
from infrastructure.signatures import require,read_report
from infrastructure.stage_sources import ROOT,digest

PROTOCOL='ray-predicate-native-e11c-v1'
SOURCE='29329268a86921f1c938b05277937369531209a4'
BASE='sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137'
ORIGINAL_BINARY='9cff02fdd0752c17a613874c0f71e47803a666234e5e5046cf194c5646102180'
PATCH='docker/paper/patches/raytracing-triangle-fp64-v1.patch'
PATCH_SHA='73fc52bb2d9de3d88fb12516ee5dad6e3a5edd978f68140ccc2ec0ff2f1a5b9e'
LIMITS=dict(whole_seconds=1800,kill_grace_seconds=30,build_seconds=900,image_seconds=120,probe_seconds=180,
            build_memory_gib=16,runtime_memory_gib=8,cpu_threads=4,output_bytes=2*1024**3,
            cuda_allocator_fraction=.1,warmups=5,timed_repeats=20,max_native_ray_queries=1638488)
TOL=dict(depth_abs=2e-4,position_abs=2e-4,normal_abs=2e-4,normal_unit_abs=2e-6)


def json_new(path,data):
    with Path(path).open('x') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def file_record(path):
    p=Path(path);require(p.is_file() and p.resolve()==p,'Unaliased regular file required')
    return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))


def authorize(path,sha,freeze_sha,commit,host_root):
    a,_=read_report(path,sha)
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==commit and
            a.get('freeze_sha256')==freeze_sha and a.get('host_evidence_directory')==str(host_root) and
            a.get('base_image')==BASE and a.get('work_limits')==LIMITS,'Exact E11C build+microtest authorization required')
    return a


def freeze_check(path,sha):
    f,_=read_report(path,sha)
    require(f.get('protocol')==PROTOCOL and f.get('work_limits')==LIMITS and f.get('tolerances')==TOL and
            f.get('upstream_commit')==SOURCE and f.get('patch_sha256')==PATCH_SHA,'E11C freeze mismatch')
    for name,expected in f['sources'].items():
        p=ROOT/name;require(p.resolve()==p and ROOT in p.parents and digest(p)==expected,'Changed E11C source: '+name)
    return f


def cube():
    v=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],np.float32)
    # Inward serialization, same final A -raw convention as dominant Bell shell.
    outward=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]],np.int32)
    cases=[([0,0,-3],[0,0,1]),([0,0,-3],[1e-4,0,1]),([0,0,-3],[-1e-4,0,1]),
           ([3,0,-3],[0,0,1]),([0,0,-3],[0,0,-1]),([0,0,0],[0,0,1]),([3,0,0],[-1,0,0]),([0,0,-3],[1,0,1e-7])]
    o=np.array([x[0] for x in cases],np.float32);d=np.array([x[1] for x in cases],np.float32)
    d=(d.astype(float)/np.linalg.norm(d.astype(float),axis=1,keepdims=True)).astype(np.float32)
    return v,outward[:,::-1].copy(),o,d


def reference(vertices,faces,o,d):
    from audits.bell_triangle_diagnostic_e11 import ray_hits
    out=[]
    for origin,direction in zip(o,d):
        hits=ray_hits(origin,direction,vertices[faces]);h=hits[0] if hits and hits[0]['depth']<10 else None
        if h:
            a,b,c=vertices[faces[h['face']]].astype(float);n=-np.cross(b-a,c-a);n/=np.linalg.norm(n)
            out.append(dict(hit=True,depth=h['depth'],normal=n.tolist(),face=h['face'],barycentric=h['barycentric']))
        else:out.append(dict(hit=False,depth=None,normal=None,face=None))
    return out


def compare(o,d,position,raw,depth,refs):
    """Compare actual BVH payloads; never substitute CPU outputs into them."""
    require(position.shape==raw.shape==o.shape==d.shape and depth.shape==(len(o),),'BVH result shape mismatch')
    require(np.isfinite(depth).all() and (depth>=0).all(),'Nonfinite/negative BVH depth')
    rows=[]
    for i,r in enumerate(refs):
        hit=bool(depth[i]<10);row=dict(hit=hit,expected_hit=r['hit'],passed=hit==r['hit'])
        if hit and r['hit']:
            length=float(np.linalg.norm(raw[i]));require(length>0,'Zero actual BVH normal')
            n=-raw[i]/length;normal=float(abs(n-np.array(r['normal'])).max());de=abs(float(depth[i])-r['depth'])
            pe=float(abs(position[i]-(o[i].astype(float)+r['depth']*d[i])).max())
            row.update(depth=float(depth[i]),expected_depth=r['depth'],depth_abs=de,position_abs=pe,normal_abs=normal,
                       unit_normal_abs=abs(float(np.linalg.norm(n))-1),facing_dot=float(n@-d[i]))
            row['passed']=bool(row['passed'] and np.isfinite(position[i]).all() and np.isfinite(n).all() and
                de<=TOL['depth_abs'] and pe<=TOL['position_abs'] and normal<=TOL['normal_abs'] and
                row['unit_normal_abs']<=TOL['normal_unit_abs'])
        rows.append(row)
    return rows


def selected_binary(build,variant,prefix=Path('/opt/e11c')):
    require(variant in ('original','predicate_fp64'),'Explicit backend selector required')
    item=build['original_binary'] if variant=='original' else build['corrected_binary']
    p=Path(item['path']) if variant=='original' else prefix/'corrected'/Path(item['path']).name
    require(p.is_absolute() and p.resolve()==p and digest(p)==item['sha256'],'Backend binary identity mismatch')
    if variant=='original':require(item['sha256']==ORIGINAL_BINARY,'Reference image original backend changed')
    else:require(p.parent==prefix/'corrected' and item['sha256']!=ORIGINAL_BINARY,'Corrected sidecar required, no historical fallback')
    return p
