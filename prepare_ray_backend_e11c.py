"""Data/source staging and CPU references only; never builds or invokes CUDA."""
import json,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
from infrastructure.stage_sources import ROOT,digest
from infrastructure.signatures import require
from infrastructure.ray_backend_e11c import SOURCE,PATCH,PATCH_SHA,file_record,json_new,cube,reference
from benchmarks.bell_geometry_e11 import mesh_arrays


def prepare(root):
    inputs=root/'inputs';original=inputs/'original';source=inputs/'source'
    tree=json.loads((inputs/'git-blobs.json').read_text())
    for row in tree:
        b=(original/row['path']).read_bytes()
        require(len(b)==row['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['sha'],'Pinned upstream blob mismatch')
    require(digest(ROOT/PATCH)==PATCH_SHA,'Reviewed patch changed');shutil.copytree(original,source)
    subprocess.run(['git','apply','--check',str(ROOT/PATCH)],cwd=source,check=True)
    subprocess.run(['git','apply',str(ROOT/PATCH)],cwd=source,check=True)
    changed=[r['path'] for r in tree if digest(original/r['path'])!=digest(source/r['path'])]
    require(changed==['include/raytracing/triangle.cuh'],'Unexpected upstream source changes')
    json_new(inputs/'source-manifest.json',dict(upstream_commit=SOURCE,git_blobs=tree,patch_sha256=PATCH_SHA,
        original={r['path']:digest(original/r['path']) for r in tree},patched={r['path']:digest(source/r['path']) for r in tree}))
    accepted=json.loads((ROOT/'docs/evidence/BELL_E11_TRIANGLE_DIAGNOSIS_V1.json').read_text())
    mesh=Path('/home/isavand/paper/final_results/meshes-pretrained/syn/bell.ply')
    old=Path('/home/isavand/paper/evidence/bell-oracle-e11-v2-7a327024561e4bf3bf4a7f0450107bc3/benchmark/geometry')
    paths=[mesh,old/'rays/0.npz',old/'rays/1.npz']
    for p in paths:
        expected=next(r['sha256'] for r in accepted['inputs'] if r['path']==str(p));require(digest(p)==expected,'Accepted evidence input changed')
    v,f=mesh_arrays(mesh);arch=[]
    for c in (0,1):
        with np.load(old/'rays'/(str(c)+'.npz'),allow_pickle=False) as z:arch.append({k:z[k] for k in z.files})
    ids=np.array([y*128+x for y in range(104,109) for x in range(51,56)])
    origins=list(arch[1]['origins'][ids]);directions=list(arch[1]['directions'][ids]);names=['saved-'+str(j) for j in ids]
    refs=[]
    # Reuse accepted25 CPU closest-hit references, not repeat that diagnosis.
    for row in accepted['neighborhood']:
        h=row['cpu_closest'];a,b,c=v[f[h['face']]].astype(float);n=-np.cross(b-a,c-a);n/=np.linalg.norm(n)
        refs.append(dict(hit=True,depth=h['depth'],normal=n.tolist(),face=h['face'],barycentric=h['barycentric']))
    a,b,c=v[f[181142]].astype(float);origin=arch[1]['origins'][13621]
    for fraction in (.25,.5,.75):
        edge=(1-fraction)*b+fraction*c
        for side in (-1e-4,0.,1e-4):
            target=edge+side*(a-edge);d=target-origin;d=(d/np.linalg.norm(d)).astype(np.float32)
            origins.append(origin.copy());directions.append(d);names.append('shared-edge-'+str(fraction)+'-'+str(side))
    closest=accepted['intersections'][0]['ray_position'];d=arch[1]['directions'][13621]
    for sign in (-1.,1.):
        dd=(d*sign).astype(np.float32);oo=np.array(closest,np.float32)+np.float32(1e-5)*dd
        origins.append(oo);directions.append(dd);names.append('visibility-'+str(sign))
    o=np.array(origins,np.float32);d=np.array(directions,np.float32)
    refs+=reference(v,f,o[25:],d[25:]);require(len(refs)==36,'Unexpected Bell case count')
    cv,cf,co,cd=cube();cube_refs=reference(cv,cf,co,cd)
    with (inputs/'panels.npz').open('xb') as out:np.savez(out,bell_o=o,bell_d=d,cube_v=cv,cube_f=cf,cube_o=co,cube_d=cd,
        perf_o=np.concatenate([z['origins'] for z in arch]),perf_d=np.concatenate([z['directions'] for z in arch]))
    json_new(inputs/'references.json',dict(names=names,bell=refs,cube=cube_refs,known_failure_index=12,
        saved_failure_depth=3.6032063961029053,expected_closest_face=181142,
        source_records=[file_record(p) for p in paths],description='25 accepted CPU refs reused;11 Bell controls +8 cube controls newly computed on exact quantized inputs'))
    return dict(source_manifest=file_record(inputs/'source-manifest.json'),panels=file_record(inputs/'panels.npz'),references=file_record(inputs/'references.json'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    json_new(a.root/'preparation.json',prepare(a.root))
