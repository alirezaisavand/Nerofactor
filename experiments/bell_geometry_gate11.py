"""One separately authorized O1 training history and four-checkpoint geometry audit."""
import argparse
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__=='__mp_main__':
    from infrastructure.bell_geometry_workers_v2 import bootstrap_geometry_worker
    bootstrap_geometry_worker()

from benchmarks import bell_geometry_gate11 as g
from benchmarks.generate_bell_e11 import run_with_cleanup
from experiments import bell_geometry_gate10_train_v1 as training
from experiments import bell_geometry_gate11_evaluate as evaluation
from experiments.donut_reconstructed_geometry_v1 import interrupted
from infrastructure.signatures import require
from infrastructure.stage_sources import activate,baseline,imported_sources,PINS
from infrastructure.verified_ray_backend import load
from scientific_variants.geometry_background_opacity_v1 import select

MASK_PROBES={1,2,1000,5000,15000,15001,20001,30000,30001,50000,50001,100000,200000,g.STEP}


def approval(path,f,args):
    require(path.resolve()==path and g.digest(path)==args.authorization_sha256 and
        g.read(path)==g.authorization(f,args.commit,args.freeze_sha256),'Exact Gate 11 approval required')


def context(f,args):
    return dict(protocol=g.PROTOCOL,variant=g.VARIANT,run_id=Path(f['host_output']).name,
        commit=args.commit,freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,
        stage='geometry',scientific_commit=PINS['geometry'],image_id=f['image'],seed=g.SEED,splits=f['splits'])


def completed_train(root,f,args):
    r=g.read(root/'training.json')
    require(all(r.get(k)==v for k,v in context(f,args).items()) and r['status']=='complete' and
        r['cleanup_verified'] and not r['cleanup_errors'] and r['phase']=='train' and
        r['adam_returns']==g.STEP and r['validation_calls']==r['periodic_saves']==300 and
        r['initial_state_sha256']==f['O0']['initial_state_sha256'] and r['initial_parity_passed'] and
        r['step_1000_gate']['passed'] and r['step_1000_gate']['validation']['view']=='58' and
        r['step_1000_gate']['validation']['checkpoint_step']==1000 and r['step_1000_gate']['own_cpu_roundtrip'] and
        not r['held_out_seen_by_trainer'] and
        r['last_validation']['checkpoint_step']==g.STEP and r['last_validation']['view']=='58' and
        r['loader_transport']==[dict(workers=8,start_method='spawn',dataset='DummyDataset')]*2 and
        [x['parameters_changed'] for x in r['training_witnesses'][:2]]==[False,True] and
        r['mask_forward_calls']==g.STEP and r['mask_trace_rows']==15006,
        'O1 training lifecycle, initial parity or objective inventory incomplete')
    require(set(r['retained_checkpoints'])==set(map(str,g.KEEP_STEPS)),'Missing retained checkpoint')
    require(r['retained_checkpoints']['1000']==r['step_1000_gate']['checkpoint'],'Detached natural checkpoint gate')
    for step,identity in r['retained_checkpoints'].items():g.check(root/f'checkpoints/model-step{step}.pth',identity)
    g.check(root/'checkpoints/model.pth',r['endpoint_checkpoint'])
    for name,identity in r['artifacts'].items():g.check(root/name,identity)
    return r


def native(f,args):
    approval(Path('/authorization.json'),f,args)
    import torch
    import torch.utils.cpp_extension
    import yaml
    from infrastructure.geometry_training_smoke import runtime_identity
    out=Path('/outputs');require(out.is_dir() and not any(out.iterdir()),'Fresh owned output required')
    require(os.environ.get('TORCH_EXTENSIONS_DIR')==f['extension_cache']==str(out/'torch_extensions'),
            'Owned executable cache required')
    require(os.getuid()!=0 and sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and
        torch.version.cuda=='11.1' and torch.cuda.is_available() and torch.cuda.device_count()==1 and
        torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe','Frozen runtime required')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu','CPU float32 constructor required')
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.cuda.set_per_process_memory_fraction(.75)
    backend=load('predicate_fp64',f['image']) if args.phase=='evaluate' else None
    if backend is not None:require(backend==f['backend'],'Corrected backend differs')
    source,manifest=activate('geometry',PINS['geometry'])
    cfg=yaml.safe_load((g.ROOT/g.CFG).read_text())
    r=dict(**context(f,args),phase=args.phase,status='incomplete',cleanup_verified=False,
        original_configuration=cfg,runtime=runtime_identity(torch),scientific_files=manifest['files'],
        objective=f['objective'],geometry_admitted=False,handoff_created=False,held_out_seen_by_trainer=False)
    expected_runtime=f['O0']['runtime']
    require(all(r['runtime'][key]==expected_runtime[key] for key in
                ('matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark','cudnn_deterministic','default_dtype')),
            'Arithmetic policy differs from O0')
    if args.phase=='train':
        r.update(adam_returns=0,validation_calls=0,periodic_saves=0,best_saves=0,training_witnesses=[],
                 mask_forward_calls=0,mask_trace_rows=0)
    else:r.update(steps=[],observed={},optimizer_returns=0,optimizer_state_restored=False,backend=backend)
    start=time.monotonic();receipt=out/('training.json' if args.phase=='train' else 'evaluation.json')
    def progress():
        if 'initial_state_sha256' in r:
            require(r['initial_state_sha256']==f['O0']['initial_state_sha256'],'O1/O0 initial tensor parity failed')
            r['initial_parity_passed']=True
        r['wall_seconds']=time.monotonic()-start
        temp=receipt.with_suffix('.next');temp.unlink(missing_ok=True);g.write(temp,r);temp.replace(receipt)
    def budget(first=False):
        r['owned_output_bytes_at_check']=sum(x.stat().st_size for x in out.rglob('*') if x.is_file() and not x.is_symlink())
        cap=g.TRAIN_RESOURCES['train_watchdog']-60 if args.phase=='train' else g.RESOURCES['inner']
        require(time.monotonic()-start<cap and (not first or time.monotonic()-start<3600) and
            r['owned_output_bytes_at_check']<=16*1024**3,'Time/output cap reached')
    def mask_observer(step,foreground,background,total):
        require(step==r['adam_returns']==r['mask_forward_calls']<g.STEP,'O1 mask call order differs')
        r['mask_forward_calls']+=1
        if (step+1)%20==0 or step+1 in MASK_PROBES:
            from network.loss import MaskLoss
            weight=float(MaskLoss(cfg).get_mask_weight(step))
            values={k:float(v.cpu().item()) for k,v in dict(foreground=foreground,background=background,total=total).items()}
            require(all(math.isfinite(v) for v in values.values()),'Nonfinite mask component')
            row=dict(completed=step+1,mask_weight=weight,**values,
                     weighted_foreground=weight*values['foreground'],weighted_background=weight*values['background'])
            import json
            with (out/'mask-components.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            r['mask_trace_rows']+=1
    def work():
        for name,identity in f['jit_cache'].items():
            source_cache=g.check(Path('/inputs/jit')/name,identity);dest=out/'torch_extensions'/name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_cache,dest)
        if args.phase=='train':
            from network import renderer
            import nvdiffrast.torch.ops as ops
            plugin=ops._get_plugin()
            require(g.item(Path(plugin.__file__))==f['jit_cache']['nvdiffrast_plugin/nvdiffrast_plugin.so'],
                    'Accepted plugin identity must pass before constructing/training O1')
            with select(renderer,mask_observer):training.run(out,source,f,r,torch,cfg,budget,progress)
        else:
            prior=completed_train(Path('/inputs/training-result'),f,args)
            r['training_receipt']=g.item(Path('/inputs/training-result/training.json'))
            evaluation.run(f,prior,source,out,r,torch,budget,progress)
        plugin=out/'torch_extensions/nvdiffrast_plugin/nvdiffrast_plugin.so'
        require(g.item(plugin)==f['jit_cache']['nvdiffrast_plugin/nvdiffrast_plugin.so'],'Accepted plugin changed')
        r['scientific_work_complete']=True
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted);progress()
    try:
        try:run_with_cleanup(work,torch)
        finally:torch.set_default_tensor_type('torch.FloatTensor')
        require(sys.gettrace() is None and sys.getprofile() is None and torch.tensor(0.).device.type=='cpu','Cleanup incomplete')
        budget();baseline('geometry',PINS['geometry']);g.freeze(args.freeze_sha256)
        if args.phase=='train':require(r['mask_forward_calls']==g.STEP and r['mask_trace_rows']==15006,'Incomplete O1 loss history')
        r.update(status='complete',cleanup_verified=True,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),imported_scientific_modules=imported_sources(source))
        r['artifacts']={x.relative_to(out).as_posix():g.item(x) for x in sorted(out.rglob('*'))
            if x.is_file() and x.resolve()==x and x!=receipt and x.suffix!='.next'}
    except BaseException as exc:r.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)));raise
    finally:progress()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','stage','train','evaluate','verify-train','finalize'))
    for name in ('commit','freeze-sha256','authorization-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--authorization',type=Path);args=parser.parse_args();f=g.freeze(args.freeze_sha256)
    if args.phase in ('train','evaluate'):native(f,args);return
    root=Path(f['host_output']);approval(args.authorization or root/'authorization.json',f,args)
    if args.phase=='preflight':
        require(not os.path.lexists(root),'Fresh Gate 11 output required');g.host_inputs(f)
        print(root);print(f['image']);print(Path(f['acquisition'])/'geometry')
    elif args.phase=='stage':g.stage_training(f,root/'training')
    elif args.phase=='verify-train':completed_train(root/'train',f,args)
    else:
        from experiments.bell_geometry_gate11_verify import finalize
        finalize(f,args)


if __name__=='__main__':main()
