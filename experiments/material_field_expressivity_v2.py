"""Frozen factorial CPU comparison of global and localized ambient tensor fields."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks.donut_recovery_v1 import json_new, numeric
from benchmarks.robust_tensor_q_v1 import width_profiles
from experiments import material_field_expressivity_v1 as v1
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest

PROTOCOL = 'supervised-localized-ambient-tensor-expressivity-v2'
FREEZE = 'docs/protocols/MATERIAL_FIELD_EXPRESSIVITY_V2.freeze.json'
V1_FREEZE_SHA = '409750e667ec68da0ccd9d399f7479aaf512d2e41faa97d5f6d95be955d1a6db'
V1_REPORT_MANIFEST = 'docs/results/material_field_expressivity_v1/artifact-manifest.json'
ARMS = {'E1':('sparse',False), 'E2':('dense',False), 'E3':('sparse',True), 'E4':('dense',True)}
DESIGN = dict(scenes=v1.DESIGN['scenes'],seeds=v1.DESIGN['seeds'],arms=list(ARMS),updates=2000,
              score_updates=[0,250,500,1000,1500,2000],saved_points_updates=[1000,2000],
              extra_fit_seed=74174,sparse_points=4096,dense_points=16384,
              grid=dict(long_axis_cells=[8,12,16,24,32,48,64],minimum_pitch_in_sparse_nn_medians=3,
                        global_parameters=1087244,maximum_extra_percent=8,components=6),
              spacing_bin_edges=[0,.5,1,2,4],gain_gate=dict(Q_jump_projection_gain=.10,
                        smooth_interior_max_relative_increase=.05),
              batch_size=v1.DESIGN['batch_size'],lr=v1.DESIGN['lr'],betas=v1.DESIGN['betas'],
              adam_epsilon=v1.DESIGN['adam_epsilon'],cpu_threads=4,wall_seconds=2400,
              address_space_bytes=12*1024**3,output_bytes=192*1024**2)


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha,'Frozen expressivity V2 definition differs')
    f = v1.donut.read(path)
    require(f['protocol']==PROTOCOL and f['design']==DESIGN and f['v1_freeze_sha256']==V1_FREEZE_SHA,
            'Frozen factorial policy differs')
    for name,expected in f['sources'].items():
        p = ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==expected,'Frozen expressivity source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=sha,maximum_execution_launches=1,
                host_output=f['host_output'],design=f['design'],geometry_plan=f['geometry_plan'],work=f['work'])


def grid_definition(vertices, sparse_positions):
    """Two predeclared levels selected using mesh span and fit-only spacing."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(vertices,dtype=np.float64)
    lower,upper = xyz.min(0),xyz.max(0)
    spans = upper-lower
    policy = DESIGN['grid']
    require((spans>0).all() and len(sparse_positions)>1,'Nondegenerate mesh and fitting spacing required')
    d = cKDTree(sparse_positions).query(sparse_positions,k=2)[0][:,1]
    typical = float(np.median(d))
    require(np.isfinite(typical) and typical>0,'Positive sparse nearest-neighbor spacing required')
    budget = policy['global_parameters']*policy['maximum_extra_percent']//100
    for fine in reversed(policy['long_axis_cells']):
        levels = []
        for k in (fine//2,fine):
            pitch = float(spans.max()/k)
            cells = np.ceil(spans/pitch).astype(int)  # per-axis endpoint padding
            cells[np.argmax(spans)] = k
            levels.append(dict(long_axis_cells=k,pitch=pitch,cells=cells.tolist(),
                               parameters=int(np.prod(cells+1)*policy['components'])))
        count = sum(row['parameters'] for row in levels)
        if levels[-1]['pitch']>=policy['minimum_pitch_in_sparse_nn_medians']*typical and count<=budget:
            return dict(lower=lower.tolist(),upper=upper.tolist(),levels=levels,
                        sparse_nn_median=typical,additional_parameters=count,parameter_cap=budget)
    raise ValueError('No declared grid fits the spacing and parameter limits')


def accepted_v1():
    plan = v1.freeze(ROOT/v1.FREEZE,V1_FREEZE_SHA)
    manifest = v1.donut.read(ROOT/V1_REPORT_MANIFEST)
    root = Path(plan['host_output'])
    report_item = manifest['report']
    require(manifest['status']=='complete' and manifest['work']['updates']==8000 and
            Path(report_item['path'])==root/'report.json' and manifest['output_root']==str(root),
            'Accepted V1 report identity differs')
    report = v1.donut.read(v1.donut.check_artifact(root/'report.json',report_item))
    require(report['outputs']==manifest['outputs'] and report['freeze_sha256']==V1_FREEZE_SHA,
            'Accepted V1 array inventory differs')
    return plan,root,report


def scene_data(prior, root, receipt, scene):
    """Add area-sampled observations on the original fitting triangles only."""
    from scipy.spatial import cKDTree

    geo,area,face_normals,target = v1.geometry(prior,scene)
    data = {}
    for name in ('fit','independent','boundary'):
        key = scene+'/'+name+'.npz'
        data[name] = numeric(v1.donut.check_artifact(root/key,receipt['outputs'][key]))
        require(v1.identity(data[name])==prior['scenes'][scene]['arrays'][name],'Frozen V1 panel differs: '+key)
    reserved = np.unique(data['boundary']['face'])
    rng = np.random.RandomState(v1.DESIGN['split_seed'])
    perm = rng.permutation(np.setdiff1d(np.arange(len(area)),reserved))
    cut = int(len(area)*v1.DESIGN['evaluation_face_fraction'])-len(reserved)
    fit_faces = np.sort(perm[cut:])
    held_faces = np.sort(np.r_[reserved,perm[:cut]])
    sparse = data.pop('fit')
    require(len(sparse['face'])==DESIGN['sparse_points'] and np.isin(sparse['face'],fit_faces).all()
            and np.isin(data['independent']['face'],held_faces).all(),'V1 triangle partition differs')
    extra = v1.sample_surface(geo,area,face_normals,fit_faces,DESIGN['dense_points']-DESIGN['sparse_points'],
                              DESIGN['extra_fit_seed'],target)
    dense = {k:np.concatenate((x,extra[k])) for k,x in sparse.items()}
    require(np.isin(dense['face'],fit_faces).all() and not np.intersect1d(dense['face'],held_faces).size,
            'Dense fit contains a held triangle')
    data.update(sparse=sparse,dense=dense)
    held = data['independent']
    grid = grid_definition(geo['vertices_native'],sparse['position'])
    # Labels are computed only for evaluation after selecting the geometry grid.
    held_labels = target(held['position'],held['normals'])
    np.testing.assert_array_equal(held_labels['Q_world'],held['Q_world'])
    aux = {k:held_labels[k] for k in ('material_region','boundary_distance_native')}
    dense_train_spacing = cKDTree(dense['position']).query(dense['position'],k=2)[0][:,1]
    for name in ('sparse','dense'):
        aux['nearest_'+name] = cKDTree(data[name]['position']).query(held['position'])[0]
    meta = dict(grid=grid,dense_nn_median=float(np.median(dense_train_spacing)),
                distinct_fitting_faces=dict(sparse=int(len(np.unique(sparse['face']))),
                                            dense=int(len(np.unique(dense['face'])))),
                fitting_area_fraction=float(area[fit_faces].sum()/area.sum()),
                arrays={name:v1.identity(panel) for name,panel in data.items()},
                held_auxiliary_arrays=v1.identity(aux),triangle_split=v1.identity(dict(fit=fit_faces,held=held_faces)))
    return data,aux,meta


def expected_work(panels):
    seeds = len(DESIGN['seeds'])
    updates = len(panels)*len(ARMS)*seeds*DESIGN['updates']
    scored,local = 0,0
    checkpoints = len(DESIGN['score_updates'])
    for panel in panels.values():
        parity = DESIGN['dense_points']*seeds
        base = panel['evaluation_points']
        section = sum((DESIGN[name+'_points']+base)*checkpoints*seeds for name in ('sparse','dense'))
        scored += 2*(parity+section)
        local += parity+section
    training_points = updates*DESIGN['batch_size']
    return dict(updates=updates,fitting_points=training_points,evaluation_points=scored,
                localized_point_levels=2*(training_points//2+local),RGB_renders=0,illumination_updates=0)


def make_field(torch, definitions, seed, local, grid):
    field = v1.build_field(torch,definitions,seed,'D8-T')
    if not local:
        return field
    from scientific_variants.localized_ambient_tensor_v1 import LocalizedAmbientField
    return LocalizedAmbientField(field,grid)


def local_spacing_scores(held, predicted, distances, reference_scale):
    from scipy.stats import spearmanr

    magnitudes = np.linalg.norm(held['Q_world'],axis=(1,2))
    error = np.linalg.norm(predicted['Q_world']-held['Q_world'],axis=(1,2))/magnitudes
    strata = {'all':np.ones(len(error),bool),'smooth_interior':~(held['near_boundary']|held['nearly_isotropic']),
               'boundary':held['near_boundary'],'near_isotropic':held['nearly_isotropic']}
    result = {}
    for key,points in strata.items():
        if not points.any():
            result[key] = None
            continue
        d,q = distances[points]/reference_scale,error[points]
        bins = []
        edges = DESIGN['spacing_bin_edges']+[None]
        for lo,hi in zip(edges,edges[1:]):
            keep = (d>=lo) if hi is None else ((d>=lo)&(d<hi))
            bins.append(dict(interval=[lo,hi],count=int(keep.sum()),
                             median_distance_native=float(np.median(d[keep])*reference_scale) if keep.any() else None,
                             Q_median=float(np.median(q[keep])) if keep.any() else None))
        association = float(spearmanr(d,q).correlation) if np.ptp(d)>0 and np.ptp(q)>0 else None
        result[key] = dict(spearman=association,bins=bins)
    return result


def compare_orientation(held, predictions):
    use = held['gap']>=v1.DESIGN['orientation_gap']
    for pred in predictions.values():
        w = pred['widths']**2
        use &= (w[:,0]-w[:,1])/w.sum(1)>=v1.DESIGN['orientation_gap']
    result = {}
    for name,p in predictions.items():
        dot = np.sum(p['axis'][use]*held['axis'][use],axis=1)
        result[name] = np.quantile(np.degrees(np.arccos(np.clip(abs(dot),0,1))),[.5,.95]).tolist() if use.any() else None
    return dict(count=int(use.sum()),angle_median_p95=result)


def photometric_gate(scenes):
    """Require independent boundary gains and stable interior Q for both seeds and scenes."""
    policy = DESIGN['gain_gate']
    rows = {}
    step = str(DESIGN['updates'])
    for scene,data in scenes.items():
        for seed in DESIGN['seeds']:
            base,local = (data['histories'][f'{seed}-{arm}']['scores'][step] for arm in ('E2','E4'))
            jump = local['boundary']['Q_jump_projection_median']-base['boundary']['Q_jump_projection_median']
            width_change = (np.asarray(local['boundary']['width_jump_MAE'])-base['boundary']['width_jump_MAE'])
            a,b = (result['independent']['smooth_interior']['Q_median'] for result in (base,local))
            stable = b<=a*(1+policy['smooth_interior_max_relative_increase'])
            rows[f'{scene}/{seed}'] = dict(Q_jump_projection_gain=jump,width_jump_MAE_change=width_change.tolist(),
                smooth_interior_median_Q_global=a,smooth_interior_median_Q_localized=b,
                passed=bool(jump>=policy['Q_jump_projection_gain'] and (width_change<0).all() and stable))
    return dict(eligible_for_followup=all(r['passed'] for r in rows.values()),comparisons=rows)


def train_history(torch, field, arm, seed, scene, data, aux, grid, destination, saved_root, saved_receipt, report, started):
    from scientific_variants.donut_tensor_d8 import world_tensor

    density,local = ARMS[arm]
    fit = data[density]
    p,n = (torch.as_tensor(fit[k],dtype=torch.float32) for k in ('position','normals'))
    target = torch.from_numpy(fit['Q_world'])
    active = [v for v in field.parameters() if v.requires_grad]
    count = sum(v.numel() for v in active)
    require(count==DESIGN['grid']['global_parameters']+(grid['additional_parameters'] if local else 0),
            'Global or localized parameter inventory differs')
    record = dict(parameters=count,updates=0,status='incomplete',optimization_seconds=0.,evaluation_seconds=0.,scores={})
    report['scenes'][scene]['histories'][f'{seed}-{arm}'] = record
    optimizer = torch.optim.Adam(active,lr=DESIGN['lr'],betas=tuple(DESIGN['betas']),eps=DESIGN['adam_epsilon'])
    rng = np.random.RandomState(seed+v1.DESIGN['batch_seed_offset'])
    trace = []
    work = report['work']
    batch_size = DESIGN['batch_size']
    try:
        for step in range(DESIGN['updates']+1):
            require(time.monotonic()-started<DESIGN['wall_seconds'],'Authorized CPU time exhausted')
            if step in DESIGN['score_updates']:
                begin = time.monotonic()
                score = {}
                for name,panel in (('fit',fit),('independent',data['independent']),('boundary',data['boundary'])):
                    slots = panel['position'].size//3
                    work['evaluation_points'] += slots
                    work['localized_point_levels'] += slots*2*local
                    pred = v1.prediction(torch,field,panel)
                    score[name] = (v1.boundary_scores(torch,panel,pred) if name=='boundary' else v1.population_scores(panel,pred))
                    if name=='independent' and step==DESIGN['updates']:
                        score['held_spacing'] = local_spacing_scores(panel,pred,aux['nearest_'+density],grid['sparse_nn_median'])
                        score['held_transition'] = width_profiles(aux['boundary_distance_native'],aux['material_region'],
                                                                 pred['widths'],panel['widths'],np.ones(slots))
                    if step in DESIGN['saved_points_updates']:
                        np.savez_compressed(destination/f'{seed}-{arm}-{step}-{name}.npz',**pred)
                    if arm=='E1' and step==1000:
                        key = f'{scene}/{seed}-D8-T-{name}-final.npz'
                        expected = numeric(v1.donut.check_artifact(saved_root/key,saved_receipt['outputs'][key]))
                        require(pred.keys()==expected.keys(),'V1 archived prediction fields differ')
                        for k in pred:
                            np.testing.assert_array_equal(pred[k],expected[k],err_msg=f'V1 parity {scene} {seed} {name} {k}')
                record['scores'][str(step)] = score
                record['evaluation_seconds'] += time.monotonic()-begin
                if arm=='E1' and step==1000:
                    record['V1_update_1000_exact'] = True
            if step==DESIGN['updates']:
                break
            begin = time.monotonic()
            if step%(len(p)//batch_size)==0:
                order = rng.permutation(len(p)).reshape(-1,batch_size)
            ix = order[step%len(order)]
            optimizer.zero_grad()
            work['fitting_points'] += len(ix)
            work['localized_point_levels'] += len(ix)*2*local
            q,frame,_,_ = field.predict_tensor(p[ix],n[ix])
            loss = (world_tensor(q,frame)-target[ix]).square().sum((1,2)).mean()
            require(bool(torch.isfinite(loss)),'Nonfinite physical Q loss')
            loss.backward()
            grad = sum(v.grad.square().sum() for v in active if v.grad is not None).sqrt()
            require(bool(torch.isfinite(grad)),'Nonfinite supervised gradient')
            optimizer.step()
            record['updates'] += 1
            work['updates'] += 1
            trace.append((float(loss.detach()),float(grad)))
            record['optimization_seconds'] += time.monotonic()-begin
        record['status'] = 'complete'
    finally:
        np.save(destination/f'{seed}-{arm}-training.npy',np.asarray(trace).reshape(-1,2),allow_pickle=False)


def execution(f, output, commit, sha, auth_sha):
    import scipy
    import torch
    from experiments.neural_material_field_v1 import source_definitions

    torch.set_num_threads(DESIGN['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    runtime = dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
                   python=sys.version.split()[0],device='cpu')
    require(runtime==f['runtime'] and not torch.cuda.is_available(),'Frozen CPU-only runtime required')
    prior,saved_root,saved_receipt = accepted_v1()
    definitions,source = source_definitions()
    output.mkdir()
    started = time.monotonic()
    report = dict(protocol=PROTOCOL,commit=commit,freeze_sha256=sha,authorization_sha256=auth_sha,source=source,
                  runtime=runtime,status='failed',work=dict.fromkeys(f['work'],0),scenes={})
    try:
        for scene in DESIGN['scenes']:
            data,aux,info = scene_data(prior,saved_root,saved_receipt,scene)
            require(info==f['geometry_plan'][scene]['inputs'],'Frozen geometry samples or grid differ')
            grid = info['grid']
            dest = output/scene
            dest.mkdir()
            for name,panel in dict(data,held_auxiliary=aux).items():
                np.savez_compressed(dest/(name+'.npz'),**panel)
            report['scenes'][scene] = dict(histories={},matched_orientation={},zero_residual_parity={})
            for seed in DESIGN['seeds']:
                global_field = make_field(torch,definitions,seed,False,grid)
                local_field = make_field(torch,definitions,seed,True,grid)
                g,l = global_field.state_dict(),local_field.global_field.state_dict()
                require(g.keys()==l.keys() and all(torch.equal(g[k],l[k]) for k in g),'Paired global initial state differs')
                start = [v1.prediction(torch,m,data['dense']) for m in (global_field,local_field)]
                report['work']['evaluation_points'] += 2*DESIGN['dense_points']
                report['work']['localized_point_levels'] += 2*DESIGN['dense_points']
                np.testing.assert_array_equal(start[0]['Q_world'],start[1]['Q_world'])
                n = data['dense']['normals']
                expected = v1.DESIGN['initial_width']**2*(np.eye(3)-n[:,:,None]*n[:,None,:])
                np.testing.assert_allclose(start[0]['Q_world'],expected,rtol=0,atol=v1.DESIGN['initial_Q_atol'])
                report['scenes'][scene]['zero_residual_parity'][str(seed)] = True
                for arm,(_,localized) in ARMS.items():
                    field = make_field(torch,definitions,seed,localized,grid)
                    train_history(torch,field,arm,seed,scene,data,aux,grid,dest,saved_root,saved_receipt,report,started)
                    require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<DESIGN['output_bytes'],
                            'Supervised output cap exceeded')
                report['scenes'][scene]['matched_orientation'][str(seed)] = {
                    str(step):compare_orientation(data['independent'],
                        {arm:numeric(dest/f'{seed}-{arm}-{step}-independent.npz') for arm in ARMS})
                    for step in DESIGN['saved_points_updates']}
        require(report['work']==f['work'],'Frozen factorial counters differ')
        report['photometric_followup_gate'] = photometric_gate(report['scenes'])
        report['status'] = 'complete'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['max_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['outputs'] = {p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
                             for p in sorted(output.rglob('*')) if p.is_file()}
        json_new(output/'report.json',report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--authorization',type=Path,required=True)
    args = p.parse_args()
    f = freeze(ROOT/FREEZE,args.freeze_sha256)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==args.commit
            and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
            'Approved clean source required')
    output = Path(f['host_output'])
    require(os.geteuid()!=0 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            and output.resolve()==output and output.parent.is_dir() and not output.exists(),
            'Fresh non-root CPU-only execution required')
    auth = args.authorization.resolve()
    auth_sha = digest(auth)
    require(v1.donut.read(auth)==authorization(f,args.commit,args.freeze_sha256) and digest(auth)==auth_sha,
            'Separate exact V2 execution authorization required')
    cap = DESIGN['address_space_bytes']
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))

    def terminate(signum,frame):
        raise TimeoutError('Foreground supervisor terminated the experiment')

    signal.signal(signal.SIGTERM,terminate)
    execution(f,output,args.commit,args.freeze_sha256,auth_sha)


if __name__=='__main__':
    main()
