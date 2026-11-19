"""Geometry-only weighting and the two source-bound Bell D6 arms."""
import sys

import numpy as np

from benchmarks import bell_heterogeneous_training_v1 as bell
from benchmarks.bell_heterogeneous_training_v1 import (
    ADMISSION, GENERATION, IMAGE, PANEL, POLICY, ROOT, SHADER, SPATIAL_POLICY, SPLITS, STEPS, TARGET_KEYS,
    boundary_score, check_domain_archive, check_probe_archive, check_sampler_capture, dataset,
    diagnostic_frames, diagnostics, fit_pool, geometry, point_errors,
)
from benchmarks.donut_surface_q_d6 import NeighborhoodSampler
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, baseline, digest

PROTOCOL = 'bell-kurt-distance-robust-D6-v1'
FREEZE = 'docs/protocols/BELL_DISTANCE_ROBUST_V1.freeze.json'
ARMS = ('B1','B2')
FOREGROUND_SECONDS = 7500
BASE_FREEZE_SHA = '6c563597983ad80adbe235a7692e26089e6c24265fc6b2c9ec1b68197f6f10f7'
CAP_MULTIPLE = 4.
QUANTILES = (0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1)


def multipliers(distance):
    require(distance.ndim==1 and len(distance) and np.isfinite(distance).all() and (distance>0).all(),
            'Positive finite edge distances required')
    return (SPATIAL_POLICY['reference_distance']/distance)**2


def weighting_definitions(bell_distance, donut_distance, shortest):
    bell_median = float(np.median(multipliers(bell_distance)))
    donut_median = float(np.median(multipliers(donut_distance)))
    common = dict(bell_median=bell_median,donut_median=donut_median,
                  cap=CAP_MULTIPLE*bell_median,shortest_pair_id=int(shortest))
    return {'B1':dict(common,scale=1.),'B2':dict(common,scale=donut_median/bell_median)}


def weight_values(raw, definition):
    return np.minimum(raw,definition['cap'])*definition['scale']


def distribution(values):
    return dict(mean=float(values.mean()),std=float(values.std()),
                quantiles=dict(zip(map(str,QUANTILES),np.quantile(values,QUANTILES).tolist())))


def replay(domain, donut, snapshots, *, steps=10000):
    """Return the exact eligible-edge vectors and maximum share for every batch."""
    train,sample_distance = domain['training_ids'],domain['distance']
    shortest = int(train[np.argmin(sample_distance[train])])
    definitions = weighting_definitions(sample_distance[train],donut['distance'][donut['training_ids']],shortest)
    raw = multipliers(sample_distance)
    weights = dict(D6=raw,**{arm:weight_values(raw,defn) for arm,defn in definitions.items()})
    archive = dict(pair_ids=train,distance=sample_distance[train],
                   **{f'weight_{arm}':v[train] for arm,v in weights.items()})
    active_cap = raw>definitions['B1']['cap']
    shares = {arm:np.empty(steps) for arm in weights}
    maxima = {arm:np.empty(steps,dtype=np.int64) for arm in weights}
    capped = np.empty(steps,dtype=np.int64)
    sampler = NeighborhoodSampler(train,len(raw),SPATIAL_POLICY['rng_seed'])
    shortest_steps = []
    for i in range(steps):
        ids = sampler.draw(SPATIAL_POLICY['batch_pairs'])
        if shortest in ids:
            shortest_steps.append(i+1)
        capped[i] = int(active_cap[ids].sum())
        for arm,w in weights.items():
            value = w[ids]
            k = int(value.argmax())
            shares[arm][i] = value[k]/value.sum()
            maxima[arm][i] = ids[k]
        if str(i+1) in snapshots:
            require(bell.array_hash(sampler.counts)==snapshots[str(i+1)],'Accepted Bell sampler snapshot differs')
    require(not sampler.counts[domain['diagnostic_ids']].any(),'Reserved-edge sampling changed')
    summary = dict(definitions=definitions,training_edges=len(train),steps=steps,
        capped_edge_count=int(active_cap[train].sum()),capped_edge_fraction=float(active_cap[train].mean()),
        capped_draw_count=int(capped.sum()),capped_draw_fraction=float(capped.sum()/sampler.counts.sum()),
        batches_with_cap=int((capped>0).sum()),shortest_pair_steps=shortest_steps,
        shortest_step_weight_shares={str(step):{k:float(v[step-1]) for k,v in shares.items()}
                                     for step in shortest_steps},
        weight_distribution={arm:distribution(v[train]) for arm,v in weights.items()},
        max_batch_share_distribution={arm:distribution(v) for arm,v in shares.items()},
        largest_share_step={arm:int(v.argmax())+1 for arm,v in shares.items()})
    archive.update(capped_edges_per_batch=capped,**{f'max_share_{arm}':v for arm,v in shares.items()},
                   **{f'max_share_pair_id_{arm}':v for arm,v in maxima.items()})
    return summary,archive


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha,'Bell distance-robust freeze differs')
    f = bell.b.read(path)
    original = ROOT/bell.FREEZE
    require(digest(original)==BASE_FREEZE_SHA,'Accepted Bell D6 freeze differs')
    base = bell.b.read(original)
    same = ('dataset','host_reference','mesh','panel','splits','steps','policy','shader','image','reference_freeze_sha256',
            'base_initial','boundary_policy','boundary_probe_count','integer_probe_hashes','integer_domain_hashes',
            'sampler_snapshots')
    require(all(f[k]==base[k] for k in same) and f['protocol']==PROTOCOL and f['order']==list(ARMS)
            and f['foreground_seconds']==FOREGROUND_SECONDS and f['base_freeze_sha256']==BASE_FREEZE_SHA,
            'Bell distance-robust experimental identity differs')
    evidence = ROOT/f['weight_summary']['relative_path']
    require(evidence.resolve()==evidence and ROOT in evidence.parents and digest(evidence)==f['weight_summary']['sha256'],
            'Frozen weight summary differs')
    weights = bell.b.read(evidence)['definitions']
    for arm in ARMS:
        w = weights[arm]
        expected_scale = 1. if arm=='B1' else w['donut_median']/w['bell_median']
        require(w['cap']==4*w['bell_median'] and w['scale']==expected_scale,'Invalid frozen cap or median normalization')
        recipe = base['arms']['D6']
        work = dict(recipe['work'],edge_diagnostic_batches=10000)
        require(f['arms'][arm]==dict(recipe,work=work,limits=dict(recipe['limits'],edge_diagnostic_batches=10000),
                                    distance_weighting=w),'Distance-robust arm differs from accepted D6 recipe')
    for name,sha256 in f['sources'].items():
        file = ROOT/name
        require(file.resolve()==file and ROOT in file.parents and digest(file)==sha256,
                'Frozen Bell distance-robust source differs: '+name)
    baseline('material',PINS['material'])
    return f


def authorization(f, commit, sha):
    return dict(bell.authorization(f,commit,sha),protocol=PROTOCOL,order=list(ARMS),foreground_seconds=FOREGROUND_SECONDS)


def runtime(args):
    f,a = bell.runtime(args,spec=sys.modules[__name__])
    f['surface_q_training']['D6_domain_array_sha256'] = f['D6_native_domain_hashes']
    return f,a


def edge_diagnostic(indices, contributions, output, definition, torch):
    with torch.no_grad():
        max_value,max_index = contributions.max(dim=0)
        shortest = np.flatnonzero(indices==definition['shortest_pair_id'])
        width = torch.cat((output['mx'],output['my']),dim=-1).sort(dim=-1).values
        width_stats = torch.stack((width.min(dim=0).values,width.mean(dim=0),width.max(dim=0).values))
        floor = (width[:,0]<=.0401).sum()
        maximum,position,at_floor = torch.stack((max_value,max_index.to(max_value.dtype),floor.to(max_value.dtype))).tolist()
        return dict(max_contribution=maximum,max_pair_id=int(indices[int(position)]),
                    shortest_contribution=float(contributions[shortest[0]]) if shortest.size else None,
                    minor_major_min_mean_max=width_stats.tolist(),minor_at_floor=int(at_floor))


def audit_weighted_history(f, domain, arm, history):
    w = f['arms'][arm]['distance_weighting']
    sampler = NeighborhoodSampler(domain['training_ids'],len(domain['distance']),SPATIAL_POLICY['rng_seed'])
    events = []
    for step,row in enumerate(history,1):
        indices = sampler.draw(SPATIAL_POLICY['batch_pairs'])
        d = row['edge_diagnostic']
        maximum,total = d['max_contribution'],row['surface_q_loss']
        seen = w['shortest_pair_id'] in indices
        values = np.asarray(d['minor_major_min_mean_max'])
        require(d['max_pair_id'] in indices and np.isfinite([maximum,total]).all()
                and total>=0 and 0<=maximum<=total*(1+1e-6)
                and maximum>=(total/len(indices))*(1-1e-6)
                and seen==(d['shortest_contribution'] is not None)
                and values.shape==(3,2) and np.isfinite(values).all()
                and (np.diff(values,axis=0)>=-2e-6).all() and 0<=d['minor_at_floor']<=512,
                f'Invalid weighted diagnostic, {arm} update {step}')
        if seen:
            contribution = d['shortest_contribution']
            require(np.isfinite(contribution) and 0<=contribution<=maximum,'Invalid shortest-edge contribution')
            events.append(dict(update=step,surface_q_loss=total,preclip_gradient_l2=row['preclip_gradient_l2'],**d))
        if step in STEPS:
            require(bell.array_hash(sampler.counts)==f['sampler_snapshots'][str(step)],'Bell weighted sampler differs')
    return events
