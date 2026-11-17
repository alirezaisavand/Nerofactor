"""Two separately authorized full-field arms; linear oracle observations only.

Frozen core owns BRDF, sampling, networks and existing material priors. This
small observation adapter deliberately does not invoke the historical loader,
tone-mapped RGB loss inputs, learned indirect network or automatic NVS.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np

from benchmarks.donut_recovery_v1 import (TRAINING, TRAIN_LIMITS as LIMITS, SHADER,
    POLICY, PANEL, SPLITS, SIZE, COLOR, METALLIC, WIDTHS, inputs, numeric, unit,
    physical_errors, surface_frames, runtime, verify_freeze, json_new)
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest


def shader_lighting(shader, original_shader, arm):
    """Existing environment callback supplies ONLY native direct MLP for D2.

    Both arms retain the identical validated zero_bounce_v1 configuration.
    No modification to corrected-core, network capacity or BRDF is required.
    """
    require(arm in ('D1','D2') and shader.cfg['indirect_mode']=='zero_bounce_v1',
            'Matched direct-only acquisition required')
    if arm=='D1':return nullcontext()
    return shader.environment(lambda p,d:original_shader.predict_outer_lights(shader,p,d),
                              'donut-D2-native-directional-MLP-zero-indirect-v1')


def benchmark(root, expected):
    require(root.resolve()==root and digest(root/'generation.json')==expected,'Explicit admitted generation hash required')
    g=json.loads((root/'generation.json').read_text())
    from benchmarks.donut_recovery_v1 import REFERENCE, ENVIRONMENT
    from infrastructure.verified_ray_backend import identity
    require(g['status']=='complete_pending_researcher_admission' and g['protocol'] in (REFERENCE,'donut-kurt-reference-v2') and
        g['cleanup_verified'] and g['splits']==SPLITS and g['environment']==ENVIRONMENT and
        g['backend']==identity() and g['material']['color']==COLOR and g['material']['widths_major_minor']==WIDTHS and
        g['material']['metallic']==METALLIC and g['BRDF']=='Kurt-a0-metallic-additive-v1' and
        g['indirect']=='zero' and g['point_light']=='absent',
        'Incomplete/foreign reference dataset')
    for name,item in g['outputs'].items():
        p=root/name
        require(p.resolve()==p and root in p.parents and p.stat().st_size==item['bytes'] and
                digest(p)==item['sha256'],'Reference artifact changed: '+name)
    require(len(g['views'])==6 and all(r['radiance_convergence']['passed'] and r['energy_convergence']['passed'] for r in g['views']),
            'Reference scientific gates incomplete')
    if g['protocol']=='donut-kurt-reference-v2':
        require(g.get('reflectance_orders')==[64,128] and
            g.get('validation',{}).get('grazing',{}).get('slope_two_grid',{}).get('passed') is True and
            all(g.get('validation',{}).get('ordinary',{}).get(key,{}).get('passed') is True
                for key in ('slope_two_grid','old_incident_two_grid','slope_vs_independent_incident')) and
            g.get('camera0_reuse',{}).get('array_equality_verified') is True and
            g['camera0_reuse'].get('new_camera0_primary_rays')==g['camera0_reuse'].get('new_camera0_RGB_integrations')==0,
            'V2 slope validation/camera0 reuse evidence incomplete')
    return g,{camera:numeric(root/('view-%d.npz'%camera)) for camera in PANEL}


def train(args):
    freeze,auth=runtime(args,args.phase)
    import os
    require(auth.get('researcher_admitted_dataset') is True and auth.get('benchmark_generation_sha256')==args.benchmark_sha256
        and os.statvfs(args.benchmark).f_flag&os.ST_RDONLY, 'Separate explicit dataset admission/training approval required')
    generation,data=benchmark(args.benchmark,args.benchmark_sha256)
    require(generation['freeze_sha256']==args.freeze_sha256,'Reference and training source/configuration freeze differ')
    geometry,_,records=inputs(args.oracle,args.cameras)
    expected_initial=None
    if args.phase=='D2':
        require(args.initial is not None and digest(args.initial)==args.initial_sha256 and
            os.statvfs(args.initial).f_flag&os.ST_RDONLY,'Explicit immutable D1 initialization anchor required')
        expected_initial=json.loads(args.initial.read_text())
    args.output.mkdir();(args.output/'runtime/cache').mkdir(parents=True);started=time.monotonic()
    report=dict(schema_version=1,protocol=TRAINING,status='incomplete',arm=args.phase,
        commit=args.commit,freeze_sha256=args.freeze_sha256,benchmark_sha256=args.benchmark_sha256,
        input_identities=records,optimizer_returns=0,cleanup_verified=False,
        effective_shader=deepcopy(SHADER),policy=POLICY,saved_endpoints=[],
        effective_lighting=('known analytic L2' if args.phase=='D1' else 'original A directional outer_light MLP'),
        indirect_illumination='identically zero in D1/D2',research_G_handoff_admitted=False,
        counters=dict(shader_calls=0,shadow_rays=0,component_reverse_sweeps=0))
    def budget(key=None,count=0):
        require(time.monotonic()-started<LIMITS['inner_seconds'],'Authorized arm runtime reached; preserve partials')
        if key:
            require(report['counters'][key]+count<=LIMITS[key], 'Native work cap: '+key)
            report['counters'][key]+=count
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],'Owned output cap reached')
    def progress():
        p=args.output/'progress.next';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(args.output/'progress.json')
    try:
        import torch
        import random
        from infrastructure.verified_ray_backend import load
        from benchmarks.generate_bell_e11 import run_with_cleanup
        from benchmarks.neural_observation_e16 import compatibility
        from experiments.bell_neural_material_e16 import state_hash
        from infrastructure.material_recovery_e13_evaluation import native_tensor_default
        from infrastructure.sphere_replay import evaluation_scope
        from infrastructure.sphere_visuals import write_view
        from infrastructure.surface_q_regularization import (surface_pairs,world_q,normal_transport,
            gradient_diagnostics,legacy_energy_specular)
        require(torch.__version__=='1.8.1+cu111' and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Reviewed native runtime required')
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
        report['runtime']=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
            CXX11_ABI=bool(torch._C._GLIBCXX_USE_CXX11_ABI),numpy=np.__version__,declared_image=args.image_id)
        report['backend']=load('predicate_fp64',args.image_id);activate('material',PINS['material'])
        import raytracing
        from network import field,renderer
        from train.lr_common_manager import WarmUpCosLR
        from scientific_variants.material_a_v1 import native_classes,orientation
        tracer=raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),geometry['triangles'].astype(np.int32))
        def trace(o,d):
            p,n,depth=tracer.trace(o,d)
            return p,n,depth.reshape(-1,1),depth.reshape(-1)<10
        random.seed(POLICY['seed']);np.random.seed(POLICY['seed'])
        torch.manual_seed(POLICY['seed']);torch.cuda.manual_seed_all(POLICY['seed'])
        shader_class,_=native_classes(field.MCShadingNetwork,renderer.NeROMaterialRenderer,
                                      'A_corrected_core_v1',field.linear_to_srgb)
        shader=shader_class(deepcopy(SHADER),trace).cuda()
        initial=dict(state_sha256=state_hash(shader),variant='A_corrected_core_v1',seed=POLICY['seed'])
        require(expected_initial is None or initial==expected_initial,'Full initial weights/buffers differ from D1')
        json_new(args.output/'initial.json',initial);report['initial']=initial
        material_names=('feats_network','mx_predictor','my_predictor','rotation_predictor','kd_predictor','metallic_predictor')
        for name,p in shader.named_parameters():
            p.requires_grad_(name.split('.')[0] in material_names or (args.phase=='D2' and name.startswith('outer_light.')))
        report['trainable_parameters']={name:sum(p.numel() for n,p in shader.named_parameters() if p.requires_grad and n.startswith(name+'.'))
            for name in (*material_names,'outer_light','inner_light')}
        require(all(report['trainable_parameters'][name]>0 for name in material_names),'Full native material field required')
        require(sum(report['trainable_parameters'][name] for name in material_names)==1485584,
                'Original full material capacity changed')
        def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
        def shade(p,wo,n,step,training):
            budget('shader_calls',1);budget('shadow_rays',len(p)*1536)
            _,o=shader(p,wo,n,None,step,training)
            require(all(bool(torch.isfinite(v).all()) for v in o.values() if isinstance(v,torch.Tensor)),'Nonfinite native output')
            return o
        # No training gradient or state update enters this independent parity gate.
        first=data[PANEL[0]];valid=np.flatnonzero(first['shader_valid'])
        ix=valid[np.linspace(0,len(valid)-1,24).astype(int)]
        p,n,wo=(tensor(first[k][ix]) for k in ('position','normals','outgoing'))
        T,B=shader.compute_tangent_bitangent_flat(n,None);axis=tensor(first['axis'][ix])
        angle=torch.atan2((axis*B).sum(-1),(axis*T).sum(-1))
        rotation=torch.stack([torch.cos(2*angle),torch.sin(2*angle)],-1)
        captures=[]
        with evaluation_scope(shader,torch,np),torch.no_grad(),native_tensor_default(torch):
            for draw in range(32):
                torch.manual_seed(900000+draw);torch.cuda.manual_seed_all(900000+draw)
                budget('shader_calls',1);budget('shadow_rays',len(ix)*1536)
                _,o=shader.shade_anisotropic_mixed(p,n,wo,n.new_full((24,1),WIDTHS[0]),n.new_full((24,1),WIDTHS[1]),
                    n.new_full((24,1),METALLIC),n.new_tensor(COLOR).expand(24,3),rotation,(rotation+1)/2,None,False)
                captures.append(o['linear_rgb'].cpu().numpy())
        reference=first['linear_components'][ix,2];coarse=first['coarse_components'][ix,2]
        gate=compatibility(np.stack(captures),reference,abs(reference-coarse),POLICY['compatibility_absolute'],POLICY['compatibility_relative'])
        np.savez_compressed(args.output/'native-reference-gate.npz',native_draws=np.stack(captures),reference=reference,coarse=coarse,pixel_indices=first['pixel_indices'][ix])
        report['native_reference_gate']=gate;progress()
        require(gate['passed'],'Native/reference compatibility or precision gate failed before optimization')
        pool={key:np.concatenate([data[i][key][data[i]['shader_valid']] for i in SPLITS['fitting']])
              for key in ('position','normals','outgoing','linear_components')}
        require(len(pool['position'])>=512,'Insufficient original fitting observations')
        sample_rng=np.random.RandomState(POLICY['seed']);order=sample_rng.permutation(len(pool['position']));cursor=0
        schedule=WarmUpCosLR(dict(end_warm=1000,end_iter=100000))
        optimizer=schedule.construct_optimizer(torch.optim.Adam,shader)
        # Geometric, edge-connected diagnostic only; NEVER added to the loss.
        pairs=surface_pairs(geometry['vertices_native'],geometry['triangles'])
        pair_ix=np.linspace(0,len(pairs['points0'])-1,min(1024,len(pairs['points0']))).astype(int)
        pair_points=[pairs[k][pair_ix] for k in ('points0','points1')]
        pair_frames=[surface_frames(geometry,v) for v in pair_points]
        def evaluate(step):
            destination=args.output/('step-%05d'%step);destination.mkdir()
            metrics={};shader.eval()
            with evaluation_scope(shader,torch,np),torch.no_grad(),native_tensor_default(torch):
                for camera,d in data.items():
                    indices=np.flatnonzero(d['shader_valid']);chunks=[]
                    for start in range(0,len(indices),512):
                        ids=indices[start:start+512];pp,nn,vv=(tensor(d[k][ids]) for k in ('position','normals','outgoing'))
                        draws=[];last=None
                        for draw in range(8):
                            seed=800000+camera*1000+start+draw
                            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                            last=shade(pp,vv,nn,step,False)
                            draws.append(np.stack([last[k].cpu().numpy() for k in ('linear_rgb','linear_diffuse','linear_specular')],1))
                        # Deterministic field maps from the same returned forward, no extra material query.
                        chunks.append((ids,{k:v.cpu().numpy() for k,v in last.items() if k in
                            ('kd','mx','my','metallic','rotation','rotation_raw','tangents','bitangents','normals')},np.mean(draws,axis=0),np.std(draws,axis=0,ddof=1)/np.sqrt(8)))
                    arrays={k:np.zeros((SIZE,SIZE,width),np.float32) for k,width in
                        dict(kd=3,mx=1,my=1,metallic=1,rotation=2,rotation_raw=2,tangents=3,bitangents=3,normals=3,
                             linear_rgb=3,linear_diffuse=3,linear_specular=3).items()}
                    standard_error=np.zeros((SIZE*SIZE,3,3),np.float32)
                    for ids,physical,mean,se in chunks:
                        pixels=d['pixel_indices'][ids]
                        for key,value in physical.items():arrays[key].reshape(-1,value.shape[-1])[pixels]=value
                        for k,key in enumerate(('linear_rgb','linear_diffuse','linear_specular')):arrays[key].reshape(-1,3)[pixels]=mean[:,k]
                        standard_error[pixels]=se
                    # Preserve geometric mask separately from one-sided BRDF domain.
                    mask=np.zeros(SIZE*SIZE,bool);mask[d['pixel_indices'][indices]]=True;mask=mask.reshape(SIZE,SIZE)
                    truth=np.zeros((SIZE*SIZE,3));truth[d['pixel_indices']]=d['linear_components'][:,2]
                    arrays.update(hit_mask=mask,rgb_gt=tensor(truth).reshape(SIZE,SIZE,3).cpu().numpy(),
                        rgb_pr=field.linear_to_srgb(tensor(arrays['linear_rgb'])).cpu().numpy(),
                        diffuse_color=field.linear_to_srgb(tensor(arrays['linear_diffuse'])).cpu().numpy(),
                        specular_color=field.linear_to_srgb(tensor(arrays['linear_specular'])).cpu().numpy(),
                        position=np.zeros((SIZE,SIZE,3),np.float32),normal=np.zeros((SIZE,SIZE,3),np.float32))
                    # write_view expects display GT; numerical linear GT remains separate.
                    arrays['linear_gt']=arrays['rgb_gt'].copy();arrays['rgb_gt']=field.linear_to_srgb(tensor(arrays['linear_gt'])).cpu().numpy()
                    arrays['position'].reshape(-1,3)[d['pixel_indices']]=d['position']
                    arrays['normal'].reshape(-1,3)[d['pixel_indices']]=d['normals']
                    write_view(destination/('camera-%d'%camera),arrays,dict(arm=args.phase,completed_updates=step,camera_id=camera),np)
                    point_pred={k:arrays[k].reshape(-1,arrays[k].shape[-1])[d['pixel_indices'][indices]] for k in ('kd','mx','my','metallic')}
                    t=2*arrays['tangents'].reshape(-1,3)[d['pixel_indices'][indices]]-1
                    b=2*arrays['bitangents'].reshape(-1,3)[d['pixel_indices'][indices]]-1
                    point_pred['axis']=np.where(point_pred['mx']>=point_pred['my'],t,b)
                    point_pred['Q_world']=point_pred['mx'][:,:,None]**2*t[:,:,None]*t[:,None,:]+point_pred['my'][:,:,None]**2*b[:,:,None]*b[:,None,:]
                    errors,physical=physical_errors(point_pred,{k:d[k][indices] for k in ('normals','axis','Q_world')})
                    difference=arrays['linear_rgb'][mask]-arrays['linear_gt'][mask]
                    metrics[str(camera)]=dict(split='fitting' if camera in SPLITS['fitting'] else 'held_out',physical=physical,
                        RGB_linear_RMSE=float(np.sqrt(np.mean(difference**2))),RGB_relative_L2=float(np.linalg.norm(difference)/max(np.linalg.norm(arrays['linear_gt'][mask]),1e-15)),
                        preview_RGB_channels_above_one=int((arrays['linear_rgb'][mask]>1).sum()),
                        native_standard_error_RMS=float(np.sqrt(np.mean(standard_error.reshape(SIZE,SIZE,3,3)[mask]**2))))
                    np.savez_compressed(destination/('camera-%d-errors.npz'%camera),**errors,native_component_SE=standard_error,geometry_hit_mask=d['hit_mask'])
                q=[]
                for pp,ff in zip(pair_points,pair_frames):
                    a=shader.predict_anisotropic_components(tensor(pp))
                    q.append(world_q(a[0],a[1],a[4],tensor(ff['normals']),torch))
                R=normal_transport(tensor(pair_frames[1]['normals']),tensor(pair_frames[0]['normals']),torch)
                delta=q[0]-R@q[1]@R.transpose(-1,-2)
                gt0,gt1=(tensor(ff['Q_world']) for ff in pair_frames)
                truth_delta=gt0-R@gt1@R.transpose(-1,-2)
                metrics['surface_Q_smoothness_diagnostic']=dict(pairs=len(pair_ix),predicted_mean_squared=float(delta.square().sum((-1,-2)).mean()),
                    oracle_mean_squared=float(truth_delta.square().sum((-1,-2)).mean()),new_loss=False)
                y,x=np.mgrid[:128,:256];phi=(x+.5)/256*2*np.pi;theta=(y+.5)/128*np.pi
                directions=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],-1)
                light=shader.predict_outer_lights(tensor(np.zeros((32768,3))),tensor(directions.reshape(-1,3))).cpu().numpy().reshape(128,256,3)
                require(np.isfinite(light).all() and (light>=0).all(),'Invalid illumination map')
                from benchmarks.bell_reference_e11 import environment
                from benchmarks.donut_recovery_v1 import ENVIRONMENT
                from benchmarks.bell_e11 import linear_to_srgb
                from PIL import Image
                truth_light=environment(tensor(directions),ENVIRONMENT).cpu().numpy()
                np.savez_compressed(destination/'illumination.npz',directions=directions,linear_RGB=light,known_linear_RGB=truth_light)
                Image.fromarray(linear_to_srgb(light)).save(destination/'illumination-preview.png')
                weight=np.sin(theta)[...,None]
                metrics['illumination']=dict(minimum=light.min((0,1)).tolist(),maximum=light.max((0,1)).tolist(),
                    mean_spherical=((light*weight).sum((0,1))/weight.sum()).tolist(),
                    relative_L2=float(np.sqrt(((light-truth_light)**2*weight).sum()/max((truth_light**2*weight).sum(),1e-15))),
                    indirect='exactly zero',mode=report['effective_lighting'])
                json_new(destination/'metrics.json',metrics)
            shader.train();report['saved_endpoints'].append(dict(step=step,metrics=metrics))
            progress()
        def work():
            nonlocal cursor,order
            with shader_lighting(shader,field.MCShadingNetwork,args.phase),native_tensor_default(torch):
                evaluate(0)
                with (args.output/'training.jsonl').open('x') as log:
                    for step in range(10000):
                        budget();shader.train()
                        if cursor+512>len(order):order=sample_rng.permutation(len(order));cursor=0
                        selected=order[cursor:cursor+512];cursor+=512
                        pp,nn,vv=(tensor(pool[k][selected]) for k in ('position','normals','outgoing'))
                        optimizer.zero_grad();lr=schedule(optimizer,step)
                        o=shade(pp,vv,nn,step,True)
                        photo=renderer.NeROMaterialRenderer.compute_rgb_loss(SimpleNamespace(cfg={'rgb_loss':'charbonier'}),o['linear_rgb'],tensor(pool['linear_components'][selected,2])).mean()
                        reg=shader.anisotropic_regularization(pp,nn,o['tangents'],o['bitangents'],o['mx'],o['my'],o['metallic'],o['kd'],o['f_d'],o['f_s_sum'],o['L_spec'],o['rotation'],o['rotation_raw'],step).mean()
                        white=renderer.NeROMaterialRenderer.compute_diffuse_light_regularization(SimpleNamespace(cfg={'reg_diffuse_light_lambda':.01}),o['diffuse_light']).mean()
                        loss=photo+reg+white
                        require(bool(torch.isfinite(loss)),'Nonfinite joint loss')
                        row=dict(completed_updates=step+1,lr=lr,RGB_loss=float(photo.detach()),material_prior=float(reg.detach()),
                            diffuse_white_prior=float(white.detach()),kd_mean=o['kd'].detach().mean(0).cpu().tolist(),
                            metallic_mean=float(o['metallic'].detach().mean()),diffuse_allocation=o['f_d'].detach().mean(0).cpu().tolist(),
                            mx_mean=float(o['mx'].detach().mean()),my_mean=float(o['my'].detach().mean()))
                        if step==0 or (step+1)%2000==0:
                            energy,spec=legacy_energy_specular(o['f_d'],o['f_s_sum'],o['L_spec'],.05,.05,torch)
                            # Four extra reverse sweeps at each of six declared boundaries.
                            report['counters']['component_reverse_sweeps']+=4
                            require(report['counters']['component_reverse_sweeps']<=24,'Gradient diagnostic cap')
                            named=[(n,p) for n,p in shader.named_parameters() if p.requires_grad]
                            row['gradient_interactions']=gradient_diagnostics(dict(photo=photo,material=reg,energy=energy,specular=spec),[p for _,p in named],torch,[n for n,_ in named])
                        loss.backward()
                        active=[p for p in shader.parameters() if p.requires_grad and p.grad is not None]
                        require(active and all(bool(torch.isfinite(p.grad).all()) for p in active),'Missing/nonfinite native gradients')
                        row['gradient_groups']={group:sum(float(torch.sum(p.grad.detach()**2)) for name,p in shader.named_parameters()
                            if name.startswith(group+'.') and p.grad is not None)**.5 for group in (*material_names,'outer_light','inner_light')}
                        require(all(any(name.startswith(group+'.') and p.grad is not None for name,p in shader.named_parameters())
                            for group in material_names),'A material predictor has no gradient')
                        require(args.phase!='D2' or any(name.startswith('outer_light.') and p.grad is not None
                            for name,p in shader.named_parameters()),'Learned direct light is disconnected')
                        norm=torch.nn.utils.clip_grad_norm_(shader.parameters(),.7);row['preclip_gradient_l2']=float(norm)
                        require(bool(torch.isfinite(norm)), 'Nonfinite total gradient norm')
                        before=[p.detach().clone() for p in active] if step<2 else []
                        optimizer.step();report['optimizer_returns']+=1
                        if step<2:
                            displacement=sum(float(abs(p.detach()-q).sum()) for p,q in zip(active,before))
                            require(displacement==0 if step==0 else displacement>0,'Original warm-up parameter displacement failed')
                            require(all(int(optimizer.state[p]['step'])==step+1 and
                                bool(torch.isfinite(optimizer.state[p]['exp_avg']).all()) and
                                bool(torch.isfinite(optimizer.state[p]['exp_avg_sq']).all()) for p in active), 'Adam state did not advance finitely')
                            row['parameter_displacement_L1']=displacement
                        log.write(json.dumps(row,allow_nan=False)+'\n')
                        if (step+1)%2000==0:
                            log.flush()
                            checkpoint=args.output/('checkpoint-%05d.pth'%(step+1))
                            saved=dict(protocol=TRAINING,variant='A_corrected_core_v1',arm=args.phase,
                                benchmark_sha256=args.benchmark_sha256,freeze_sha256=args.freeze_sha256,
                                completed_updates=step+1,network_state_dict=shader.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                                torch_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),
                                sample_RNG=sample_rng.get_state(),sample_order=order,sample_cursor=cursor)
                            torch.save(saved,checkpoint)
                            # Trust is limited to this process's own newly written checkpoint.
                            loaded=torch.load(checkpoint,map_location='cpu')
                            require(all(loaded[k]==saved[k] for k in ('protocol','variant','arm','benchmark_sha256','freeze_sha256','completed_updates')), 'Checkpoint identity mismatch')
                            require(set(loaded['network_state_dict'])==set(shader.state_dict()) and all(torch.equal(v.cpu(),loaded['network_state_dict'][k]) for k,v in shader.state_dict().items()), 'Checkpoint round-trip differs')
                            shader.load_state_dict(loaded['network_state_dict'],strict=True)
                            del loaded,saved
                            evaluate(step+1)
                require(report['optimizer_returns']==10000,'Incomplete recovery history')
            return {}
        run_with_cleanup(work,torch)
        report['cleanup_verified']=True
        report['cleanup_scope']='Native synchronization/cache cleanup and lighting/default/RNG scopes; process/container exit handled by foreground launcher'
        import resource
        report['resources']=dict(CUDA_peak_allocated=torch.cuda.max_memory_allocated(),
            CUDA_peak_reserved=torch.cuda.max_memory_reserved(),host_peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        verify_freeze(args.freeze,args.freeze_sha256);benchmark(args.benchmark,args.benchmark_sha256);inputs(args.oracle,args.cameras)
        report.update(status='complete',scientific_work_complete=True)
        report['outputs']={p.relative_to(args.output).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
            for p in args.output.rglob('*') if p.is_file() and p.name not in ('progress.json','progress.next')}
    except BaseException as exc:
        report.update(status='failed',scientific_work_complete=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        json_new(args.output/'recovery.json',report)
