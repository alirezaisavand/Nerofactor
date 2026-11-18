"""CPU-only fitting scale match. Reads prediction arrays, geometry and work logs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from benchmarks.donut_log_tensor_d11 import coefficient, POLICY
from benchmarks.donut_recovery_v1 import json_new, SIZE, SPLITS
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest
from infrastructure.surface_q_regularization import normal_transport
from scientific_variants.donut_log_tensor_d11 import world_log_tensor
from scientific_variants.donut_tensor_d8 import frame, world_tensor

D8 = ROOT/'docs/results/donut_paired_tensor_d8_v1'
D8_FREEZE = ROOT/'docs/protocols/DONUT_KURT_PAIRED_TENSOR_D8_V1.freeze.json'
STEPS = (2000, 4000, 6000, 8000, 10000)


def checked(path, record):
    require(path.resolve()==path and path.stat().st_size==record['bytes']
            and digest(path)==record['sha256'], 'Scale input identity differs: '+str(path))
    return path


def array_subset(path, names):
    with np.load(path, allow_pickle=False) as a:
        return {name:a[name] for name in names}


def fitting_pairs(view, admitted_edges):
    ids = np.flatnonzero(view['shader_valid'])
    pixels = view['pixel_indices'][ids]
    lookup = np.full(SIZE*SIZE, -1)
    lookup[pixels] = ids
    # Ordered right/down pixel neighbors only, without repeated pairs.
    horizontal = pixels[pixels % SIZE != SIZE-1]
    vertical = pixels[pixels < SIZE*(SIZE-1)]
    l = lookup[np.r_[horizontal,vertical]]
    r = lookup[np.r_[horizontal+1,vertical+SIZE]]
    present = (l>=0)&(r>=0)
    l,r = l[present],r[present]
    face_l,face_r = view['triangle'][l],view['triangle'][r]
    adjacent = (face_l==face_r) | np.array([pair in admitted_edges for pair in zip(face_l,face_r)])
    distance = np.linalg.norm(view['position'][l]-view['position'][r], axis=-1)
    facing = (view['normals'][l]*view['normals'][r]).sum(-1)
    ok = (adjacent & (distance>=1e-6) & (distance<=POLICY['max_surface_distance']) &
          (facing>=np.cos(np.deg2rad(POLICY['max_face_dihedral_degrees']))))
    require(ok.any(), 'No geometry-matched fitting neighbors')
    return l[ok],r[ok],distance[ok]


def gradient_scale(q, normals, distance):
    n = torch.as_tensor(normals, dtype=torch.float64)
    u = frame(n.reshape(-1,3)).reshape(*n.shape,2)
    q = torch.as_tensor(q,dtype=torch.float64)
    q = (u.transpose(-1,-2)@q@u)
    q = ((q+q.transpose(-1,-2))/2).requires_grad_()
    r = normal_transport(n[:,1], n[:,0], torch)
    scale = torch.as_tensor((POLICY['reference_distance']/distance)**2)
    values = {}
    for tag, embed in [('raw',world_tensor),('log',world_log_tensor)]:
        w = embed(q.reshape(-1,2,2),u.reshape(-1,3,2)).reshape(-1,2,3,3)
        delta = w[:,0]-r@w[:,1]@r.transpose(-1,-2)
        loss = delta.square().sum((-1,-2))*scale
        grad, = torch.autograd.grad(loss.sum(), q)
        values[tag] = dict(gradient_square_sum=float(grad.square().sum()),
                           mean=float(loss.mean()), median=float(loss.median()))
    return values


def calibrate(output):
    torch.set_num_threads(2)
    seal = json.loads((D8/'finalization-validation.json').read_text())['artifact_sha256']
    manifest = D8/'artifact-manifest.json'
    require(digest(manifest)==seal[str(manifest.relative_to(ROOT))], 'D8 accepted evidence differs')
    manifest = json.loads(manifest.read_text())
    batch = Path(manifest['batch'])/'D8-T'
    records = manifest['records']
    receipt_path = checked(batch/'recovery.json', records[str(batch/'recovery.json')])
    receipt = json.loads(receipt_path.read_text())
    require(receipt['status']=='complete' and receipt['cleanup_verified'] and receipt['optimizer_returns']==10000,
            'D8-T control is incomplete')
    f = json.loads(D8_FREEZE.read_text())['tensor_training']
    ref = Path(f['benchmark_host_path'])
    gen_path = ref/'generation.json'
    require(digest(gen_path)==f['benchmark_generation_sha256'], 'Admitted generation changed')
    generated = json.loads(gen_path.read_text())['outputs']
    domain_path = checked(batch/'surface-q-domain.npz',receipt['outputs']['surface-q-domain.npz'])
    domain = array_subset(domain_path, ('face_pairs','training_ids'))
    faces = domain['face_pairs'][domain['training_ids']]
    edges = set(map(tuple, np.r_[faces,faces[:,::-1]]))
    identities = [receipt_path,gen_path,domain_path]
    rows = []
    for camera in SPLITS['fitting']:
        name = f'view-{camera}.npz'
        ref_path = checked(ref/name, generated[name])
        identities.append(ref_path)
        view = array_subset(ref_path, ('shader_valid','pixel_indices','triangle','position','normals'))
        l,r,distance = fitting_pairs(view,edges)
        normals = view['normals'][np.stack((l,r),axis=1)]
        pixel = view['pixel_indices'][np.stack((l,r),axis=1)]
        for step in STEPS:
            suffix = f'step-{step:05d}/camera-{camera}/materials.npz'
            path = checked(batch/suffix, receipt['outputs'][suffix])
            identities.append(path)
            maps = array_subset(path, ('Q_tensor_direct','hit_mask'))
            require(np.array_equal(np.flatnonzero(maps['hit_mask']), view['pixel_indices'][view['shader_valid']]),
                    'Archived D8-T fitting maps differ')
            values = gradient_scale(maps['Q_tensor_direct'].reshape(-1,3,3)[pixel], normals, distance)
            rows.append(dict(camera=camera, step=step, pairs=len(l), **values))
    count = sum(row['pairs'] for row in rows)
    raw,log = (float(np.sqrt(sum(row[k]['gradient_square_sum'] for row in rows)/count)) for k in ('raw','log'))
    chosen = coefficient(raw,log)
    for row in rows:
        row['weighted_raw_mean'] = row['raw']['mean']*POLICY['coefficient']
        row['weighted_log_mean'] = row['log']['mean']*chosen
    history_path = checked(batch/'training.jsonl',receipt['outputs']['training.jsonl'])
    identities.append(history_path)
    history = [r for line in history_path.read_text().splitlines() if 'gradient_interactions' in (r:=json.loads(line))]
    json_new(output, dict(schema_version=1, coefficient=chosen, base_coefficient=POLICY['coefficient'],
        pooled_rms_raw=raw,pooled_rms_log=log, fitting_pairs_across_checkpoints=count,
        policy='Match pooled RMS per-pair physical tangent-Q Frobenius gradient under the D6 distance factor.',
        scope='Fitting prediction arrays and geometry only; archived probes describe network-gradient context.',
        geometry_proxy='Right/down pixel neighbors on the same face or a D6 training edge; chord distance and 15 degree normal gate.',
        rows=rows, archived_training_gradient_context=[
            dict(step=v['completed_updates'],surface_q=v['gradient_interactions']['terms']['surface_q'],
                 photo=v['gradient_interactions']['terms']['photo']) for v in history],
        artifacts={str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in identities}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    calibrate(parser.parse_args().output)
