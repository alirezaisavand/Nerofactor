"""Prepare, execute and independently verify the single authorized Gate 10 batch."""
import argparse
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__=='__mp_main__':
    from infrastructure.bell_geometry_workers_v2 import bootstrap_geometry_worker
    bootstrap_geometry_worker()

import numpy as np

from benchmarks import bell_geometry_gate10_v1 as g
from benchmarks.generate_bell_e11 import run_with_cleanup
from experiments.donut_reconstructed_geometry_v1 import interrupted
from experiments import bell_geometry_gate10_evaluate_v1 as evaluation
from experiments import bell_geometry_gate10_train_v1 as training
from infrastructure.paths import PathResolver
from infrastructure.signatures import require
from infrastructure.stage_sources import activate, baseline, imported_sources
from infrastructure.stage_workspace import handoff
from infrastructure.verified_ray_backend import load


def approval(path, f, args):
    require(len(args.commit)==40 and all(c in '0123456789abcdef' for c in args.commit) and
            path.resolve()==path and g.digest(path)==args.authorization_sha256
            and g.read(path)==g.authorization(f,args.commit,args.freeze_sha256),'Exact Gate 10 authorization required')


def context(f, args):
    return dict(protocol=g.PROTOCOL,run_id=Path(f['host_output']).name,commit=args.commit,
                freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,
                stage='geometry',scientific_commit=g.PINS['geometry'],image_id=g.IMAGE,seed=g.SEED,
                splits=f['splits'])


def completed_train(root, f, args):
    r=g.read(root/'training.json')
    require(all(r.get(k)==v for k,v in context(f,args).items()) and r.get('status')=='complete'
            and r.get('cleanup_verified') and not r.get('cleanup_errors') and r.get('phase')=='train'
            and r.get('adam_returns')==g.STEP and r.get('validation_calls')==r.get('periodic_saves')==300
            and r.get('endpoint_network_sha256') and r.get('step_1000_gate',{}).get('passed')
            and r['step_1000_gate']['validation']['view']=='58' and r['last_validation']['view']=='58'
            and r['step_1000_gate']['validation']['checkpoint_step']==g.CADENCE
            and r['last_validation']['checkpoint_step']==g.STEP
            and r.get('loader_transport')==[dict(workers=8,start_method='spawn',dataset='DummyDataset')]*2
            and [r['training_witnesses'][i]['parameters_changed'] for i in (0,1)]==[False,True],
            'G training completion/cleanup receipt is missing or incompatible')
    require(set(r.get('retained_checkpoints',{}))==set(map(str,g.KEEP_STEPS)),'Missing intermediate checkpoints')
    for step,item in r['retained_checkpoints'].items():
        g.check(root/f'checkpoints/model-step{step}.pth',item)
    g.check(root/'checkpoints/model-step1000.pth',r['step_1000_gate']['checkpoint'])
    g.check(root/'checkpoints/model.pth',r['endpoint_checkpoint'])
    return r


def native(f,args):
    import torch
    import yaml
    from infrastructure.geometry_training_smoke import runtime_identity

    approval(Path('/authorization.json'),f,args)
    out=Path('/outputs')
    require(out.is_dir() and not any(out.iterdir()),'Native Gate 10 output must be empty')
    require(os.getuid()!=0 and sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111'
            and torch.version.cuda=='11.1' and torch.cuda.is_available() and torch.cuda.device_count()==1
            and torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe','Wrong native Gate 10 environment')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu',
            'Original Torch CPU float32 constructor context required')
    torch.cuda.set_per_process_memory_fraction(g.RESOURCES['cuda_allocator_fraction'])
    torch.set_num_threads(g.RESOURCES['cpu_threads'])
    torch.set_num_interop_threads(g.RESOURCES['cpu_threads'])
    if args.phase=='evaluate':
        backend=load('predicate_fp64',g.IMAGE)
        require(backend==f['backend'],'Corrected Gate 10 tracing backend differs')
    source,manifest=activate('geometry',g.PINS['geometry'])
    cfg=yaml.safe_load((g.ROOT/g.CFG).read_text())
    report=dict(**context(f,args),original_configuration=cfg,phase=args.phase,status='incomplete',
                cleanup_verified=False,runtime=runtime_identity(torch),
                source=dict(stage='geometry',algorithm='geometry-G',scientific_commit=g.PINS['geometry'],
                            scientific_files=manifest['files'],source_manifest_sha256=f['sources']['scientific_sources/manifest-v1.json']),
                limits=g.LIMITS, eval_only_oracle_mesh=True, held_out_seen_by_trainer=False)
    if args.phase=='train':
        report.update(adam_returns=0,validation_calls=0,periodic_saves=0,best_saves=0,training_witnesses=[])
    else:
        report.update(views=[],primary_BVH_ray_slots=0,cpu_witnesses=0,backend=backend,quality_passed=False)
    started=time.monotonic()
    receipt=out/('training.json' if args.phase=='train' else 'evaluation.json')
    temp=receipt.with_suffix('.next')

    def progress():
        temp.unlink(missing_ok=True)
        report['wall_seconds']=time.monotonic()-started
        g.write(temp,report)
        temp.replace(receipt)

    def budget(first=False):
        elapsed=time.monotonic()-started
        deadline=g.RESOURCES[('train' if args.phase=='train' else 'eval')+'_watchdog']-g.RESOURCES['inner_margin_seconds']
        size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
        require(elapsed<deadline and (not first or elapsed<g.RESOURCES['step_1000_seconds'])
                and size<=g.RESOURCES['owned_outputs_bytes'],'Gate 10 native time or owned-output budget exceeded')
        report['owned_output_bytes_at_check']=size

    def work():
        try:
            if args.phase=='train':
                training.run(out,source,f,report,torch,cfg,budget,progress)
            else:
                prior=Path('/inputs/training-result')
                accepted=completed_train(prior,f,args)
                report['training_receipt']=g.item(prior/'training.json')
                evaluation.run(out,source,f,report,torch,cfg,prior,accepted,budget,progress)
        finally:
            torch.set_default_tensor_type('torch.FloatTensor')

    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    progress()
    try:
        run_with_cleanup(work,torch)
        report['cleanup_verified']=sys.gettrace() is None and sys.getprofile() is None and torch.tensor(0.).device.type=='cpu'
        require(report['cleanup_verified'],'Native Gate 10 cleanup verification failed')
        budget()
        baseline('geometry',g.PINS['geometry'])
        g.freeze(args.freeze_sha256)
        names=('checkpoint-events.jsonl',) if args.phase=='train' else (
            evaluation.MESH,'sdf-witnesses.npz',*(f'rasters/{i}.npz' for i in g.EVAL_IDS))
        report['artifacts']={name:g.item(out/name) for name in names}
        report.update(peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      imported_scientific_modules=imported_sources(source))
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        progress()


def verify_rasters(out, views):
    split_rows={}
    for row in views:
        camera=row['camera']
        with np.load(out/'rasters'/f'{camera}.npz',allow_pickle=False) as z:
            names=('oracle','reconstructed')
            arrays={name:{k:z[f'{name}_{k}'] for k in ('hit','normal','position','depth')} for name in names}
            expected,masks=g.paired_surface(arrays['oracle'],arrays['reconstructed'],z['directions'])
            require(all(row[k]==v for k,v in expected.items()) and
                    all(np.array_equal(mask,z[k]) for k,mask in masks.items()),'Gate 10 saved raster decisions differ')
            bucket=split_rows.setdefault(row['split'],[])
            a,b=(arrays[n] for n in names)
            common=masks['common']
            ndot=(a['normal'][common]*b['normal'][common]).sum(axis=-1)
            bucket.append(dict(oracle=a['hit'],reconstructed=b['hit'],common=common,eligible=masks['valid'],
                               depth=b['depth'][common].astype(float)-a['depth'][common].astype(float),
                               displacement=np.linalg.norm(b['position'][common]-a['position'][common],axis=1),
                               normal_degrees=np.degrees(np.arccos(np.clip(ndot,-1,1)))))
    output={}
    for split,entries in split_rows.items():
        a,b,common,eligible=(np.concatenate([r[k] for r in entries]) for k in ('oracle','reconstructed','common','eligible'))
        output[split]=dict(oracle=int(a.sum()),reconstructed=int(b.sum()),common=int(common.sum()),eligible=int(eligible.sum()),
                           silhouette_iou=float(common.sum()/(a|b).sum()),coverage=float(eligible.sum()/a.sum()),
                           oracle_only=int((a&~b).sum()),reconstructed_only=int((b&~a).sum()),
                           signed_depth=g.distribution(np.concatenate([r['depth'] for r in entries])),
                           absolute_depth=g.distribution(np.abs(np.concatenate([r['depth'] for r in entries]))),
                           displacement=g.distribution(np.concatenate([r['displacement'] for r in entries])),
                           normal_degrees=g.distribution(np.concatenate([r['normal_degrees'] for r in entries])))
    return output



def verify_sdf(path, expected):
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    class SavedQueries:
        def __init__(self):
            self.points=self.gradient_points=0
        def __call__(self,p,gradients=False):
            self.points+=len(p)
            if gradients:
                require(self.gradient_points==0 and np.array_equal(p,data['points']),'SDF base witness differs')
                self.gradient_points=len(p)
                self.last_inside=data['inside_unit_ball']
                return data['raw_sdf'],data['gradient']
            self.last_inside=np.r_[data['inward_inside_unit_ball'],data['outward_inside_unit_ball']]
            return np.r_[data['raw_sdf_inward'],data['raw_sdf_outward']],None
    observed,arrays=evaluation.sdf_consistency(data['points'],data['native_normals'],SavedQueries())
    require(observed==expected and all(np.array_equal(data[k],v) for k,v in arrays.items()),
            'Saved raw SDF witness decisions differ')


def finalize(f,args):
    root=Path(f['host_output'])
    approval(root/'authorization.json',f,args)
    require(all((root/f'{p}-exit-code.txt').read_text().strip()=='0' and
                (root/f'{p}-container-absent.txt').read_text().strip()=='absent' for p in ('train','evaluate')),
            'Gate 10 container exit/cleanup evidence differs')
    train=completed_train(root/'train',f,args)
    out=root/'evaluate'
    report=g.read(out/'evaluation.json')
    require(all(report.get(k)==v for k,v in context(f,args).items()) and report.get('status')=='complete'
            and report.get('cleanup_verified') and report.get('phase')=='evaluate' and report.get('network_state_preserved')
            and report.get('training_receipt')==g.item(root/'train/training.json') and report.get('backend')==f['backend']
            and report.get('checkpoint')==train['endpoint_checkpoint'] and report.get('endpoint_checkpoint_loads')==1
            and report.get('extraction_tiles')==512 and report.get('open3d_A_mesh_parity') and not report.get('optimizer_state_restored')
            and report.get('primary_BVH_ray_slots')==g.LIMITS['primary_BVH_rays']
            and report.get('cpu_witnesses')==g.LIMITS['cpu_closest_hit_witnesses']
            and tuple(r['camera'] for r in report.get('views',()))==g.EVAL_IDS,'Incomplete native Bell geometry evidence')
    for path,receipt in ((root/'train',train),(out,report)):
        for name,item in receipt['artifacts'].items():
            g.check(path/name,item)
    for name,item in f['eval_files'].items():
        g.check(Path(f['acquisition'])/'geometry'/name,item)
    g.read_staged(f,root/'training')
    summaries=verify_rasters(out,report['views'])
    verify_sdf(out/'sdf-witnesses.npz',report['sdf'])
    mesh=out/evaluation.MESH
    digest=g.digest(mesh)
    require(digest==report['mesh_outputs'][evaluation.MESH],'Gate 10 mesh detached from producing receipt')
    passed=report['sdf']['passed'] and all(r['passed'] and all(w['passed'] for ws in r['cpu_closest_hit'].values() for w in ws)
                                         for r in report['views'])
    require(passed==report['quality_passed'],'Native Gate 10 quality decision differs')
    summary=dict(**context(f,args),status='verified_pending_researcher_review' if passed else 'quality_gate_failed',
                 geometry_admissible=passed,training_receipt=g.item(root/'train/training.json'),
                 evaluation_receipt=g.item(out/'evaluation.json'),mesh=g.item(mesh),splits_summary=summaries,
                 views=[{k:v for k,v in row.items() if k!='cpu_closest_hit'} for row in report['views']],
                 sdf=report['sdf'],mesh_topology=report['mesh_topology'])
    if passed:
        record=dict(schema_version=1,geometry_algorithm='geometry-G',scientific_commit=g.PINS['geometry'],
                    geometry_run_id=report['run_id'],geometry_run_manifest={'root':'geometry','path':'evaluation.json'},
                    geometry_run_manifest_sha256=summary['evaluation_receipt']['sha256'],mesh_sha256=digest,
                    database_name='syn/bell',coordinates='Original admitted Bell v3 right-handed XYZ and world-to-camera calibration',
                    scale='Unchanged Bell v3 normalized scene units',
                    normals='Pinned A negates raw BVH face normals, with signed raster and SDF checks recorded',
                    orientation='No mesh reorientation. Sampled SDF signs and camera-facing correspondences verified',
                    material_preprocessing='A-original-read-triangles-and-compute-missing-vertex-normals-no-file-write')
        candidate=out/'handoff.json'
        g.write(candidate,record)
        resolver=PathResolver(dict(datasets=Path(f['acquisition'])/'datasets',thesis=root/'train',geometry=out),root/'future-material')
        require(handoff(candidate,mesh,resolver,'syn/bell')==record,'Existing G-to-A validator rejected handoff')
        summary['handoff']=g.item(candidate)
    g.write(root/'offline-verification.json',summary)
    require(passed,'Bell geometry failed the frozen quality gate. See offline-verification.json')
    print(root/'offline-verification.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','stage','train','evaluate','verify-train','finalize'))
    parser.add_argument('--commit',required=True)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--authorization-sha256',required=True)
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    f=g.freeze(args.freeze_sha256)
    if args.phase in ('train','evaluate'):
        native(f,args)
        return
    out=Path(f['host_output'])
    approval(args.authorization or out/'authorization.json',f,args)
    if args.phase=='preflight':
        require(not os.path.lexists(out) and out.parent.is_dir(),'Gate 10 requires a fresh output root')
        require(g.definition(out)==f,'Gate 10 frozen inputs differ from researcher admission')
        baseline('geometry',g.PINS['geometry'])
        print(out)
        print(g.IMAGE)
        print(Path(f['acquisition'])/'geometry')
    elif args.phase=='stage':
        g.stage_training(f,out/'training')
    elif args.phase=='verify-train':
        completed_train(out/'train',f,args)
    else:
        finalize(f,args)


if __name__=='__main__':
    main()
