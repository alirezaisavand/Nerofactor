"""CPU-only supervised fit of the existing chart and ambient anisotropy fields."""
import argparse
import hashlib
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

from benchmarks import bell_heterogeneous_training_v1 as bell
from benchmarks import donut_heterogeneous_v1 as donut
from benchmarks.donut_recovery_v1 import inputs, json_new
from benchmarks.donut_surface_q_d6 import neighborhoods, POLICY as DONUT_NEIGHBORHOODS
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest

PROTOCOL = 'supervised-chart-ambient-material-expressivity-v1'
FREEZE = 'docs/protocols/MATERIAL_FIELD_EXPRESSIVITY_V1.freeze.json'
DESIGN = dict(scenes=['donut','bell'],arms=['D8-C','D8-T'],seeds=[6033,6034],
              split_seed=74171,fit_seed=74172,evaluation_seed=74173,batch_seed_offset=94000,
              fit_points=4096,evaluation_points=8192,boundary_pairs_max=256,evaluation_face_fraction=.25,
              initial_width=.25,initial_Q_atol=2e-8,orientation_gap=.1,
              updates=1000,batch_size=256,score_updates=[0,250,500,1000],lr=.001,betas=[.9,.999],
              adam_epsilon=1e-8,cpu_threads=4,wall_seconds=1800,address_space_bytes=12*1024**3,output_bytes=128*1024**2)


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha,'Frozen supervised execution identity differs')
    f = donut.read(path)
    require(f['protocol']==PROTOCOL and f['design']==DESIGN and set(f['scenes'])==set(DESIGN['scenes']),
            'Supervised experiment definition differs')
    for name,expected in f['sources'].items():
        p = ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==expected,'Frozen supervised source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=sha,design=f['design'],
                host_output=f['host_output'],scenes=f['scenes'],maximum_execution_launches=1)


def geometry(f, scene):
    data = f['scenes'][scene]
    for item in (data['generation'],data['admission']):
        donut.check_artifact(Path(item['path']),item)
    admitted = donut.read(Path(data['admission']['path']))
    require(admitted['researcher_admitted_dataset'] and admitted['generation']==data['generation'],
            'Foreign analytic field admission')
    if scene=='bell':
        geo,_,_ = bell.geometry(data,Path(data['mesh']['path']))
        _,raw,area = bell.b.face_data(geo['vertices_native'],geo['triangles'])
        return geo,area,-raw,lambda p,n: bell.b.material_field(p,n,geo['bounds'])
    geo,_,_ = inputs(Path(data['oracle']),Path(data['cameras']))
    ring,area = donut.geometry_fields(geo)
    return geo,area,None,lambda p,n: donut.material_field(p,n,ring)


def boundary_panel(scene, geo, target):
    if scene=='bell':
        pairs = bell.boundary_probes(geo)
        select = np.linspace(0,len(pairs['face_ids'])-1,DESIGN['boundary_pairs_max']).astype(int)
        face = np.broadcast_to(pairs['face_ids'][select],(2,len(select))).copy()
        position = pairs['positions'][:,select]
        normal = np.broadcast_to(pairs['normals'][select],position.shape).copy()
    else:
        pairs = neighborhoods(geo,DONUT_NEIGHBORHOODS)
        p = np.stack((pairs['points0'],pairs['points1']))
        n = np.stack((pairs['normals0'],pairs['normals1']))
        region = [target(x,y)['material_region'] for x,y in zip(p,n)]
        cross = np.flatnonzero(region[0]!=region[1])
        require(len(cross)>0,'No Donut boundary probes')
        select = cross[np.linspace(0,len(cross)-1,min(len(cross),DESIGN['boundary_pairs_max'])).astype(int)]
        reverse = region[0][select]!=0
        face = pairs['face_pairs'][select].T
        position,normal = p[:,select],n[:,select]
        face[:,reverse] = face[::-1,reverse]
        position[:,reverse] = position[::-1,reverse]
        normal[:,reverse] = normal[::-1,reverse]
    oracle = [target(p,n) for p,n in zip(position,normal)]
    require(all((o['material_region']==i).all() for i,o in enumerate(oracle)),'Boundary pair orientation differs')
    return dict(face=face,position=position,normals=normal,Q_world=np.stack([o['Q_world'] for o in oracle]),
                widths=np.stack([o['widths_major_minor'] for o in oracle]))


def sample_surface(geo, area, face_normals, faces, count, seed, target):
    rng = np.random.RandomState(seed)
    face = rng.choice(faces,size=count,p=area[faces]/area[faces].sum())
    xy = rng.uniform(size=(count,2))
    root = np.sqrt(xy[:,0])
    bary = np.stack((1-root,root*(1-xy[:,1]),root*xy[:,1]),axis=1)
    position = (geo['vertices_native'][geo['triangles'][face]]*bary[:,:,None]).sum(1)
    normal = (face_normals[face] if face_normals is not None else
              donut.unit((geo['shading_normals_world'][geo['triangle_loops'][face]]*bary[:,:,None]).sum(1)))
    labels = target(position,normal)
    return dict(face=face,position=position,normals=normal,Q_world=labels['Q_world'],axis=labels['axis'],
                widths=labels['widths_major_minor'],gap=labels['anisotropy_gap'],
                near_boundary=labels['near_boundary'],nearly_isotropic=labels['nearly_isotropic'])


def panels(f, scene):
    geo,area,normals,target = geometry(f,scene)
    boundary = boundary_panel(scene,geo,target)
    reserved = np.unique(boundary['face'])
    rng = np.random.RandomState(DESIGN['split_seed'])
    perm = rng.permutation(np.setdiff1d(np.arange(len(area)),reserved))
    n_eval = int(len(area)*DESIGN['evaluation_face_fraction'])-len(reserved)
    require(0<n_eval<len(perm),'Reserved boundary faces exhaust evaluation split')
    evaluation = np.sort(np.r_[reserved,perm[:n_eval]])
    fitting = np.sort(perm[n_eval:])
    fit = sample_surface(geo,area,normals,fitting,DESIGN['fit_points'],DESIGN['fit_seed'],target)
    held = sample_surface(geo,area,normals,evaluation,DESIGN['evaluation_points'],DESIGN['evaluation_seed'],target)
    require(not np.intersect1d(fit['face'],np.r_[held['face'],reserved]).size
            and held['nearly_isotropic'].any() and held['near_boundary'].any(),'Independent evaluation coverage differs')
    summary = dict(fitting_faces=len(fitting),evaluation_faces=len(evaluation),boundary_faces=len(reserved),
                   evaluation_area_fraction=float(area[evaluation].sum()/area.sum()),
                   boundary_pairs=boundary['face'].shape[1],evaluation_nearly_isotropic=int(held['nearly_isotropic'].sum()),
                   evaluation_boundary=int(held['near_boundary'].sum()))
    return dict(fit=fit,independent=held,boundary=boundary),summary


def identity(arrays):
    return {name:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
            for name,a in arrays.items()}


def build_field(torch, definitions, seed, arm):
    import torch.nn as nn
    from benchmarks.donut_recovery_v1 import SHADER
    from scientific_variants.donut_tensor_shader_d8 import tensor_shader
    from scientific_variants.material_a_v1 import native_classes

    def forbidden(*args):
        raise RuntimeError('This field-only diagnostic has no RGB, renderer, or lighting path')

    class FieldOnly(nn.Module):
        default_cfg = {}
        predict_anisotropic_components = definitions['predict_anisotropic_components']

        def __init__(self,cfg,ray_trace_fun):
            super().__init__()
            self.cfg = dict(cfg)
            self.feats_network = definitions['MaterialFeatsNetwork']()
            make = definitions['make_predictor']
            for name,size in (('mx',1),('my',1),('metallic',1),('rotation',2),('kd',3)):
                setattr(self,name+'_predictor',make(259,size))

    # The factory also defines its unused original-shader branch. Give it an
    # explicit failure for each non-field base method that branch would need.
    for name in ('compute_tangent_bitangent_flat','sample_diffuse_directions','sample_aniso_ggx_directions',
                 'compute_Dh','compute_pdf_aniso_ggx','compute_radiance','shade_anisotropic_mixed',
                 'anisotropic_regularization'):
        setattr(FieldOnly,name,forbidden)

    core,_ = native_classes(FieldOnly,nn.Module,'A_corrected_core_v1',forbidden)
    torch.manual_seed(seed)
    field = tensor_shader(core,forbidden)(SHADER,forbidden)
    field.select_tensor_representation(arm)
    field.kd_predictor.requires_grad_(False)
    field.metallic_predictor.requires_grad_(False)
    return field


def boundary_scores(torch, data, predicted):
    from infrastructure.surface_q_regularization import normal_transport

    n = torch.from_numpy(data['normals'])
    R = normal_transport(n[1],n[0],torch).numpy()
    oracle = data['Q_world']
    q = predicted['Q_world'].reshape(oracle.shape)
    widths = predicted['widths'].reshape(data['widths'].shape)
    gt_jump = oracle[0]-R@oracle[1]@R.transpose(0,2,1)
    jump = q[0]-R@q[1]@R.transpose(0,2,1)
    denom = np.square(gt_jump).sum((1,2))
    keep = denom>1e-12
    width_jump = widths[0]-widths[1]
    wanted = data['widths'][0]-data['widths'][1]
    return dict(pairs=len(denom),nonzero_Q_jump_pairs=int(keep.sum()),
                Q_jump_projection_median=float(np.median((jump[keep]*gt_jump[keep]).sum((1,2))/denom[keep])) if keep.any() else None,
                Q_jump_absolute_error_mean=float(np.linalg.norm(jump-gt_jump,axis=(1,2)).mean()),
                signed_predicted_width_jump=width_jump.mean(0).tolist(),signed_reference_width_jump=wanted.mean(0).tolist(),
                width_jump_MAE=abs(width_jump-wanted).mean(0).tolist())


def population_scores(data, pred):
    error = np.linalg.norm(pred['Q_world']-data['Q_world'],axis=(1,2))/np.linalg.norm(data['Q_world'],axis=(1,2))
    width = abs(pred['widths']-data['widths'])
    square = pred['widths']**2
    gap = (square[:,0]-square[:,1])/square.sum(1)
    truth_ok = data['gap']>=DESIGN['orientation_gap']
    eligible = truth_ok&(gap>=DESIGN['orientation_gap'])
    angles = np.degrees(np.arccos(np.clip(abs((pred['axis']*data['axis']).sum(1)),0,1)))
    result = {}
    nearby,weak = data['near_boundary'],data['nearly_isotropic']
    for label,keep in dict(all=np.ones(len(error),bool),smooth_interior=~(nearby|weak),
                           boundary=nearby,near_isotropic=weak,exactly_isotropic=data['gap']==0).items():
        if not keep.any():
            result[label] = None
            continue
        valid = keep&eligible
        result[label] = dict(count=int(keep.sum()),Q_median=float(np.median(error[keep])),
            Q_p95=float(np.quantile(error[keep],.95)),Q_at_most_005=int((error[keep]<=.05).sum()),
            width_MAE=width[keep].mean(0).tolist(),predicted_gap_median=float(np.median(gap[keep])),
            oracle_orientation_eligible=int((keep&truth_ok).sum()),joint_orientation_eligible=int(valid.sum()),
            orientation_median=float(np.median(angles[valid])) if valid.any() else None,
            orientation_p95=float(np.quantile(angles[valid],.95)) if valid.any() else None)
    return result


def prediction(torch, field, data):
    from scientific_variants import donut_tensor_d8 as tensors
    p,n = (torch.as_tensor(data[k].reshape(-1,3),dtype=torch.float32) for k in ('position','normals'))
    out = dict(Q_world=[],widths=[],axis=[])
    with torch.no_grad():
        for pp,nn in zip(p.split(512),n.split(512)):
            local,frame,_,_ = field.predict_tensor(pp,nn)
            out['Q_world'].append(tensors.world_tensor(local,frame).numpy())
            mx,my,axis,_ = tensors.export_axes(local,frame)
            out['widths'].append(torch.cat((mx,my),1).numpy())
            out['axis'].append(axis.numpy())
    return {k:np.concatenate(v) for k,v in out.items()}


def experiment(f, output, commit, sha, authentication):
    import scipy
    import torch
    from experiments.neural_material_field_v1 import source_definitions
    from scientific_variants.donut_tensor_d8 import world_tensor

    torch.set_num_threads(DESIGN['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    runtime = dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
                   python=sys.version.split()[0],device='cpu')
    require(runtime==f['runtime'],'Frozen supervised CPU runtime differs')
    definitions,source = source_definitions()
    require(not torch.cuda.is_available(),'CUDA must be hidden for this CPU experiment')
    started = time.monotonic()
    output.mkdir()
    report = dict(protocol=PROTOCOL,commit=commit,freeze_sha256=sha,authorization_sha256=authentication,
        runtime=runtime,
        status='failed',source=source,work=dict(updates=0,fit_points=0,evaluation_points=0,
                                              RGB_renders=0,illumination_updates=0),scenes={})
    try:
        for scene in DESIGN['scenes']:
            data,sample_summary = panels(f,scene)
            require(sample_summary==f['scenes'][scene]['sample_summary']
                    and {k:identity(v) for k,v in data.items()}==f['scenes'][scene]['arrays'],
                    'Frozen supervised benchmark arrays differ')
            captured = output/scene
            captured.mkdir()
            for name,panel in data.items():
                np.savez_compressed(captured/(name+'.npz'),**panel)
            results = report['scenes'][scene] = dict(samples=sample_summary,histories={})
            p,n = (torch.as_tensor(data['fit'][k],dtype=torch.float32) for k in ('position','normals'))
            q = torch.from_numpy(data['fit']['Q_world'])
            for seed in DESIGN['seeds']:
                fields = {arm:build_field(torch,definitions,seed,arm) for arm in DESIGN['arms']}
                chart,ambient = (fields[k] for k in DESIGN['arms'])
                for name in ('feats_network','kd_predictor','metallic_predictor'):
                    for a,b in zip(getattr(chart,name).state_dict().values(),getattr(ambient,name).state_dict().values()):
                        require(torch.equal(a,b),'Paired shared weights differ')
                initial = {arm:prediction(torch,field,data['fit']) for arm,field in fields.items()}
                report['work']['evaluation_points'] += 2*len(p)
                atol = DESIGN['initial_Q_atol']
                anchor = DESIGN['initial_width']**2*(np.eye(3)-data['fit']['normals'][:,:,None]*data['fit']['normals'][:,None,:])
                np.testing.assert_allclose(initial['D8-C']['Q_world'],anchor,rtol=0,atol=atol)
                np.testing.assert_allclose(initial['D8-T']['Q_world'],anchor,rtol=0,atol=atol)
                np.testing.assert_allclose(initial['D8-C']['Q_world'],initial['D8-T']['Q_world'],rtol=0,atol=atol)
                for arm,field in fields.items():
                    label = f'{seed}-{arm}'
                    active = [v for v in field.parameters() if v.requires_grad]
                    record = results['histories'][label] = dict(parameters=sum(v.numel() for v in active),
                                                               updates=0,scores={},status='incomplete')
                    optimizer = torch.optim.Adam(active,lr=DESIGN['lr'],betas=tuple(DESIGN['betas']),eps=DESIGN['adam_epsilon'])
                    rng = np.random.RandomState(seed+DESIGN['batch_seed_offset'])
                    t0 = time.monotonic()
                    trace = []
                    try:
                        for step in range(DESIGN['updates']+1):
                            require(time.monotonic()-started<DESIGN['wall_seconds'],'Supervised experiment runtime exhausted')
                            if step in DESIGN['score_updates']:
                                scores = {}
                                for key,panel in data.items():
                                    report['work']['evaluation_points'] += panel['position'].size//3
                                    pred = prediction(torch,field,panel)
                                    scores[key] = (boundary_scores(torch,panel,pred) if key=='boundary' else population_scores(panel,pred))
                                    if step==DESIGN['updates']:
                                        np.savez_compressed(captured/f'{label}-{key}-final.npz',**pred)
                                record['scores'][str(step)] = scores
                            if step==DESIGN['updates']:
                                break
                            if step%(len(p)//DESIGN['batch_size'])==0:
                                order = rng.permutation(len(p)).reshape(-1,DESIGN['batch_size'])
                            ix = order[step%len(order)]
                            optimizer.zero_grad()
                            report['work']['fit_points'] += len(ix)
                            pred,frame,_,_ = field.predict_tensor(p[ix],n[ix])
                            loss = (world_tensor(pred,frame)-q[ix]).square().sum((1,2)).mean()
                            require(bool(torch.isfinite(loss)),'Nonfinite supervised Q loss')
                            loss.backward()
                            grad = sum(v.grad.square().sum() for v in active if v.grad is not None).sqrt()
                            require(bool(torch.isfinite(grad)),'Nonfinite supervised Q gradient')
                            optimizer.step()
                            record['updates'] += 1
                            report['work']['updates'] += 1
                            trace.append((float(loss.detach()),float(grad)))
                        record['status'] = 'complete'
                    except ValueError as exc:
                        if arm!='D8-C' or 'Undefined zero/near-zero orientation' not in str(exc):
                            raise
                        record.update(status='chart_orientation_failed',error=str(exc))
                    finally:
                        np.save(captured/f'{label}-training.npy',np.asarray(trace).reshape(-1,2),allow_pickle=False)
                        record['elapsed_seconds'] = time.monotonic()-t0
                    require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<DESIGN['output_bytes'],
                            'Supervised outputs exceed their cap')
        require(report['work']['updates']<=8000 and report['work']['fit_points']<=2048000
                and report['work']['evaluation_points']<=440320,'Supervised work cap exceeded')
        histories = [r for scene in report['scenes'].values() for r in scene['histories'].values()]
        require(len(histories)==8,'Incomplete supervised history inventory')
        report['status'] = ('complete' if all(r['status']=='complete' for r in histories)
                            else 'completed_with_chart_failures')
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['max_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['outputs'] = {str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=digest(p))
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
            'Approved clean supervised source required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.geteuid()!=0,'Non-root CPU-only execution required')
    output = Path(f['host_output'])
    require(output.resolve()==output and output.parent.is_dir() and not output.exists(),'Fresh supervised output required')
    auth = args.authorization.resolve()
    args_sha = digest(auth)
    auth_bytes = auth.read_bytes()
    require(hashlib.sha256(auth_bytes).hexdigest()==args_sha
            and json.loads(auth_bytes)==authorization(f,args.commit,args.freeze_sha256),
            'Separate exact supervised execution authorization required')
    cap = DESIGN['address_space_bytes']
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))

    def timed_out(signum, frame):
        raise TimeoutError('Supervised foreground execution was terminated')

    signal.signal(signal.SIGTERM,timed_out)
    experiment(f,output,args.commit,args.freeze_sha256,args_sha)


if __name__=='__main__':
    main()
