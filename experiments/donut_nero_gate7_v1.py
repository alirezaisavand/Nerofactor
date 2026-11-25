"""Bounded upstream NeRO Stage II with admitted Donut hits and fixed direct light."""
from argparse import ArgumentParser
from pathlib import Path
import os
import random
import resource
import signal
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_nero_gate7_v1 as g
from benchmarks.donut_recovery_v1 import json_new
from experiments.donut_reconstructed_geometry_v1 import interrupted, save
from infrastructure.signatures import require
from infrastructure.stage_sources import imported_sources
from infrastructure.verified_ray_backend import load


def execute(args,f,p,views,unseen,geometry):
    import torch
    from benchmarks.bell_reference_e11 import environment
    from benchmarks.generate_bell_e11 import run_with_cleanup
    from infrastructure.material_recovery_e13_evaluation import native_tensor_default
    from infrastructure.sphere_replay import evaluation_scope

    out=Path('/outputs/run')
    out.mkdir()
    started=checked=time.monotonic()
    r=dict(protocol=g.PROTOCOL,status='incomplete',cleanup_verified=False,commit=args.commit,
           freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,
           upstream_commit=g.UPSTREAM_COMMIT,counters=dict.fromkeys(g.LIMITS,0),outputs={},views={},
           training_domain=f['objective_domain'])

    def budget(extra=0):
        require(time.monotonic()-started<g.RESOURCES['inner_seconds'],'Gate 7 wall limit')
        require(sum(x.stat().st_size for x in Path('/outputs').rglob('*') if x.is_file())+extra
                <=g.RESOURCES['output_bytes'],'Gate 7 output limit')

    def charge(key,n):
        nonlocal checked
        require(r['counters'][key]+n<=g.LIMITS[key],'Gate 7 work cap: '+key)
        now=time.monotonic()
        if now-checked>=5:
            budget()
            checked=now
        r['counters'][key]+=n

    def capture(name,arrays):
        budget(sum(a.nbytes for a in arrays.values())+65536)
        save(out/name,arrays)

    def work():
        g.activate_nero(p['upstream'],f)
        from network import field, renderer
        from network.loss import NeRFRenderLoss, MaterialRegLoss
        from train.lr_common_manager import WarmUpCosLR
        from raytracing import RayTracer

        torch.set_num_threads(g.RESOURCES['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(g.RESOURCES['cuda_allocator_fraction'])
        random.seed(g.SEED)
        np.random.seed(g.SEED)
        torch.manual_seed(g.SEED)
        torch.cuda.manual_seed_all(g.SEED)
        tracer=RayTracer(geometry['vertices_native'].astype('f4'),geometry['triangles'].astype('i4'))

        def trace(origin,direction):
            charge('shadow_slots',len(origin))
            hits,normals,depth=tracer.trace(origin,direction)
            d=depth.reshape(-1,1)
            require(bool(torch.isfinite(d).all()),'Nonfinite visibility depth')
            return hits,normals,d,d.reshape(-1)<10

        class KnownLightNeRO(field.MCShadingNetwork):
            def predict_outer_lights(self,points,directions):
                self.light_directions+=len(directions)
                return environment(directions,self.incident)

            def get_inner_lights(self,points,directions,normals):
                return torch.zeros_like(points)

        def tensor(a):
            return torch.as_tensor(a,dtype=torch.float32,device='cuda')

        def shade(model,pv,n,wo,training,step):
            charge('shader_calls',1)
            return model(pv,wo,n,None,step,training)

        # Upstream constructs the field in the host default dtype and moves it
        # to CUDA before using its CUDA default within the forward path.
        require(torch.tensor([]).type()=='torch.FloatTensor','Unexpected upstream initialization default')
        model=KnownLightNeRO(g.SHADER,trace).cuda()
        model.light_directions=0
        model.incident=f['illuminations']['L2']
        for name,param in model.named_parameters():
            param.requires_grad_(name.split('.')[0] in g.TRAINABLE)
        active={name:param for name,param in model.named_parameters() if param.requires_grad}
        require(set(n.split('.')[0] for n in active)==set(g.TRAINABLE),'Upstream material heads differ')
        r['trainable_parameters']=sum(p.numel() for p in active.values())
        cfg=SimpleNamespace(cfg=g.RENDERER)
        lr=WarmUpCosLR(g.LR)
        optimizer=lr.construct_optimizer(torch.optim.Adam,SimpleNamespace(parameters=active.values))
        photo_loss,mat_loss=NeRFRenderLoss({}),MaterialRegLoss({})
        pool=g.m.old.fit_pool(views)
        require(len(pool['position'])==f['support']['fitting_observations']
                and set(pool)==set(g.m.FIT_KEYS),'Fitting pool differs')
        from benchmarks.bell_dense_known_l2_v1 import index_batches
        from hashlib import sha256
        stream_hash=sha256()
        with (out/'fit-indices.bin').open('xb') as stream,(out/'training.csv').open('x') as log:
            log.write('update,lr,srgb_charbonnier,material_regularizer,diffuse_light_regularizer\n')
            model.train()
            for step,index in enumerate(index_batches(len(pool['position']),g.SEED,g.STEPS,g.BATCH)):
                data=index.astype('<u4').tobytes()
                stream.write(data)
                stream_hash.update(data)
                pv,n,wo=(tensor(pool[k][index]) for k in ('position','normals','outgoing'))
                target=field.linear_to_srgb(tensor(pool['linear_components'][index,2]))
                optimizer.zero_grad()
                rate=lr(optimizer,step)
                with native_tensor_default(torch):
                    rgb,outputs=shade(model,pv,n,wo,True,step)
                    outputs['loss_rgb']=renderer.NeROMaterialRenderer.compute_rgb_loss(cfg,rgb,target)
                    outputs['loss_mat_reg']=model.material_regularization(pv,n,outputs['metallic'],
                                                                          outputs['roughness'],outputs['albedo'],step)
                    outputs['loss_diffuse_light']=renderer.NeROMaterialRenderer.compute_diffuse_light_regularization(
                        cfg,outputs['diffuse_light'])
                terms=dict(photo_loss(outputs,{},step),**mat_loss(outputs,{},step))
                require(set(terms)=={'loss_rgb','loss_mat_reg','loss_diffuse_light'},'Native objective differs')
                means={k:v.mean() for k,v in terms.items()}
                loss=sum(means.values())
                require(bool(torch.isfinite(loss)),'Nonfinite upstream objective')
                charge('backwards',1)
                loss.backward()
                require(all(p.grad is not None for p in active.values())
                        and bool(torch.stack([torch.isfinite(p.grad).all() for p in active.values()]).all()),
                        'Missing or nonfinite material gradient')
                if step==0:
                    norms={name:float(param.grad.norm()) for name,param in active.items()}
                    require(all(np.isfinite(v) for v in norms.values())
                            and any(v>0 for v in norms.values()) and model.light_directions>0,
                            'Inactive material or fixed illumination at first update')
                    r['initial_forward_backward']=dict(finite=True,gradient_norms=norms,
                        known_light_directions=model.light_directions)
                require(r['counters']['optimizer_returns']<g.STEPS,'Unexpected extra Adam call')
                optimizer.step()
                charge('optimizer_returns',1)
                log.write(f'{step+1},{rate},{float(means["loss_rgb"])},{float(means["loss_mat_reg"])},'
                          f'{float(means["loss_diffuse_light"])}\n')
                if (step+1)%1000==0:
                    require(all(bool(torch.isfinite(p).all()) for p in active.values()),'Nonfinite material weights')
                    log.flush()
                    stream.flush()
                    print(f'NeRO update {step+1}',flush=True)
        require(stream_hash.hexdigest()==g.digest(out/'fit-indices.bin')==f['support']['photometric_index_sha256'],
                'Recorded fitting indices differ')
        r['fitting_index_sha256']=stream_hash.hexdigest()
        require(all(int(optimizer.state[p]['step'])==g.STEPS for p in active.values()),'Adam moments did not advance')
        endpoint={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for label,states in (('L2',views),('unseen',{c:v['view'] for c,v in unseen.items()})):
            model.incident=f['illuminations'][label]
            before=model.light_directions
            with evaluation_scope(model,torch,np) as scoped, torch.no_grad(), native_tensor_default(torch):
                model.eval()
                for camera,view in states.items():
                    arrays={key:[] for key in ('components','albedo','metallic','roughness_native')}
                    nrows=len(view['pixel_indices'])
                    for i in range(0,nrows,g.BATCH):
                        sl=slice(i,i+g.BATCH)
                        pv,n,wo=(tensor(view[k][sl]) for k in ('position','normals','outgoing'))
                        events=[]
                        with g.observe_linear(field,events):
                            rgb,outputs=shade(model,pv,n,wo,False,g.STEPS)
                        d,s,total=g.linear_components(events)
                        require(torch.equal(d+s,total) and torch.equal(field.linear_to_srgb(total),rgb),
                                'Upstream linear readout differs')
                        values=dict(components=torch.stack((d,s,total),dim=1),albedo=outputs['albedo'],
                                    metallic=outputs['metallic'],roughness_native=outputs['roughness'])
                        for key,val in values.items():
                            require(bool(torch.isfinite(val).all() & (val>=0).all()),'Invalid linear render or material')
                            arrays[key].append(val.cpu().numpy())
                    name=f'{label}-camera-{camera}.npz'
                    capture(name,dict(pixels=view['pixel_indices'],**{k:np.concatenate(v) for k,v in arrays.items()}))
                    r['views'][name]=nrows
            require(scoped['RNG_and_modes_restored'] and model.light_directions>before,'Invalid evaluation scope or light route')
        require(all(torch.equal(v.detach().cpu(),endpoint[k]) for k,v in model.state_dict().items()),
                'Endpoint changed during evaluation')
        r['frozen_endpoint_weights_unchanged']=True
        r['known_light_directions']=model.light_directions
        r['imported_sources']=imported_sources(p['upstream'])
        # Save only the newly fitted endpoint. The run never reads a checkpoint.
        torch.save({'upstream_commit':g.UPSTREAM_COMMIT,'completed_updates':g.STEPS,
                    'network_state_dict':endpoint},out/'checkpoint-10000.pth')

    try:
        require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1'
                and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count()==1
                and 'A100' in torch.cuda.get_device_name(0),'Reviewed A100 runtime required')
        r['runtime']=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0))
        r['backend']=load('predicate_fp64',f['image'])
        require(r['backend']==f['backend'],'Corrected visibility backend differs')
        run_with_cleanup(work,torch)
        r['cleanup_verified']=True
        require(r['counters']==g.LIMITS,'Unexpected external baseline work')
        g.freeze(args.freeze_sha256)
        g.source_check(p['upstream'],f)
        budget()
        r['status']='complete'
    except BaseException as exc:
        r.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        r['elapsed_seconds']=time.monotonic()-started
        r['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch.cuda.is_initialized():
            r['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            r['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        r['outputs']={x.name:dict(bytes=x.stat().st_size,sha256=g.digest(x)) for x in sorted(out.iterdir()) if x.is_file()}
        json_new(out/'receipt.json',r)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--commit')
    cli.add_argument('--authorization-sha256')
    cli.add_argument('--execute',action='store_true')
    args=cli.parse_args()
    f=g.freeze(args.freeze_sha256)
    paths=g.paths(f,args.execute)
    if args.execute:
        auth=Path('/authorization.json')
        require(args.commit and g.digest(auth)==args.authorization_sha256
                and g.read(auth)==g.authorization(f,args.commit,args.freeze_sha256),'Exact Gate 7 execution approval required')
        require(os.geteuid()!=0 and all(os.statvfs(p).f_flag&os.ST_RDONLY for p in
                    (Path('/'),g.ROOT,auth,*paths.values())) and not os.statvfs('/outputs').f_flag&os.ST_RDONLY
                and {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
                'Read-only, non-root, networkless execution required')
    original,unseen,geometry=g.admit(f,paths)
    if args.execute:
        signal.signal(signal.SIGTERM,interrupted)
        execute(args,f,paths,original,unseen,geometry)
    else:
        print('Gate 7 metadata only: official source, exact fitting indices and both acquisition supports verified')


if __name__=='__main__':
    main()
