"""One authorized Donut relight. Frozen checkpoint inference and shared quadrature only."""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import os
import resource
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_relighting_gate6_v1 as g
from benchmarks.bell_e11 import convergence
from benchmarks.donut_recovery_v1 import SHADER, json_new, numeric
from experiments.donut_reconstructed_geometry_v1 import interrupted, save
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate
from infrastructure.verified_ray_backend import load


def native_pass(model, arm, views, archives, torch, charge, save_capture):
    from benchmarks.bell_reference_e11 import environment
    from experiments.bell_neural_material_e16 import state_hash
    from infrastructure.material_recovery_e13_evaluation import native_tensor_default
    from infrastructure.sphere_replay import evaluation_scope

    materials, parity = {}, {}
    before = state_hash(model)
    model.requires_grad_(False)
    routed = dict(calls=0,directions=0)

    def relight(_,directions):
        routed['calls']+=1
        routed['directions']+=len(directions)
        return environment(directions,g.LIGHT)

    with evaluation_scope(model,torch,np) as scoped, torch.no_grad(), native_tensor_default(torch), model.environment(
            relight,g.PROTOCOL):
        model.eval()
        for camera,view in views.items():
            count = len(view['pixel_indices'])
            fields = dict(Q_world=np.empty((count,3,3),np.float64),
                          color=np.empty((count,3),np.float64),metallic=np.empty((count,1),np.float64))
            draws = np.empty((g.DRAWS,count,3,3),np.float32)
            for begin in range(0,count,g.CHUNK):
                end = min(count,begin+g.CHUNK)
                sl = slice(begin,end)
                p,n,wo = (torch.as_tensor(view[k][sl],dtype=torch.float32,device='cuda')
                          for k in ('position','normals','outgoing'))
                for draw in range(g.DRAWS):
                    seed,dedicated = g.tape(camera,begin,draw)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    charge('native_shader_calls',1)
                    if g.TENSOR_ARMS[arm] is None:
                        _,result = model(p,wo,n,None,g.STEP,False)
                    else:
                        result = model(p,wo,n,None,g.STEP,False,tape_seed=dedicated,export=draw==0)
                    draws[draw,sl] = np.stack([result[k].cpu().numpy()
                                               for k in ('linear_diffuse','linear_specular','linear_rgb')],1)
                    if draw==0:
                        q = result.get('Q')
                        if q is None:
                            t,b = (2*result[k]-1 for k in ('tangents','bitangents'))
                            q = result['mx'][:,:,None]**2*t[:,:,None]*t[:,None,:]+result['my'][:,:,None]**2*b[:,:,None]*b[:,None,:]
                        for key,value in (('Q_world',q),('color',result['kd']),('metallic',result['metallic'])):
                            fields[key][sl] = value.cpu().numpy()
            require(np.isfinite(draws).all() and (draws>=0).all()
                    and np.array_equal(draws[:,:,0]+draws[:,:,1],draws[:,:,2]), 'Invalid native draw or component sum')
            widths,axis,error = g.physical_axes(fields['Q_world'],view['normals'])
            require(all(np.isfinite(fields[k]).all() and ((fields[k]>=0)&(fields[k]<=1)).all()
                        for k in ('color','metallic')), 'Invalid restored linear material values')
            material = dict(fields,widths_major_minor=widths,axis=axis)
            old = archives[arm,camera]
            mask = old['hit_mask']
            rows = np.searchsorted(view['pixel_indices'],view['matched_pixels'])
            qkey = 'Q_world' if g.TENSOR_ARMS[arm] is None else 'Q_tensor_direct'
            shifts = {name:float(abs(fields[name][rows]-old[key][mask]).max()) for name,key in
                      (('Q_world',qkey),('color','kd'),('metallic','metallic'))}
            require(max(shifts.values())<=g.MATERIAL_ATOL,'Restored physical field differs from accepted endpoint')
            parity[str(camera)] = dict(common_rows=len(rows),field_max_abs=shifts,Q_axis_reconstruction_max_abs=error)
            save_capture(f'{arm}-camera-{camera}-native.npz',pixels=view['pixel_indices'],draws=draws,**material)
            materials[camera] = material
    require(scoped['RNG_and_modes_restored'] and model._environment is None and state_hash(model)==before,
            'Relighting mutated model state, environment, modes or RNG')
    require(routed['calls']>0 and routed['directions']>0,'No native rays reached the unseen environment')
    return materials,dict(field_parity=parity,state_sha256=before,state_unchanged=True,unseen_environment=routed)


def integrate(view, materials, rows, grid, tracer, torch, charge):
    """Same independent Donut angular rule and BRDF, with shared shadow and light terms."""
    from benchmarks.bell_reference_e11 import basis, environment
    from experiments.bell_highlight_pilot_e11d import kernel_axes
    from experiments.bell_quadrature_e11e import midpoint_grid, query_visibility, weighted_components

    local,weight = midpoint_grid(grid)
    values = np.zeros((len(g.ROLES),len(rows),3,3),np.float64)
    count = np.zeros(len(rows),np.int64)
    for begin in range(0,len(rows),g.POINT_BATCH):
        end = min(begin+g.POINT_BATCH,len(rows))
        idx = rows[begin:end]
        p,n,wo = (torch.from_numpy(view[k][idx]).cuda() for k in ('position','normals','outgoing'))
        t,b = basis(n)
        params = [dict(n=n, **{k:torch.from_numpy(materials[arm][name][idx]).cuda()
            for k,name in (('axis','axis'),('widths','widths_major_minor'),('color','color'),('metallic','metallic'))})
            for arm in g.ROLES]
        rgb = torch.zeros((len(g.ROLES),len(idx),2,3),dtype=torch.float64,device='cuda')
        for start in range(0,len(local),g.DIRECTION_CHUNK):
            loc = torch.from_numpy(local[start:start+g.DIRECTION_CHUNK]).cuda()
            wi = loc[None,:,0,None]*t[:,None]+loc[None,:,1,None]*b[:,None]+loc[None,:,2,None]*n[:,None]
            cost = len(idx)*len(loc)
            charge('independent_shadow_directions',cost)
            visible,_ = query_visibility(torch,tracer,p,wi)
            count[begin:end] += visible.sum(1).cpu().numpy()
            light = environment(wi,g.LIGHT)
            cosine = (wi*n[:,None]).sum(-1)
            for i,material in enumerate(params):
                charge('independent_BRDF_point_directions',cost)
                fd,fs = kernel_axes(torch,wi,wo,material)
                rgb[i] += weighted_components(torch,fd,fs,light,cosine,visible,weight)
        values[:,begin:end,:2] = rgb.cpu().numpy()
    values[:,:,2] = values[:,:,0]+values[:,:,1]
    return dict(pixels=view['pixel_indices'][rows],grid=np.array(grid),components=values,visibility=count)


def execute(args, f, paths, exports, receipts, views, archives):
    import torch
    from benchmarks.generate_bell_e11 import run_with_cleanup
    from experiments.bell_quadrature_e11e import controls
    from scientific_variants.material_a_v1 import native_classes

    out = Path('/outputs/run')
    out.mkdir()
    started = last = time.monotonic()
    r = dict(protocol=g.PROTOCOL, commit=args.commit, freeze_sha256=args.freeze_sha256,
             authorization_sha256=args.authorization_sha256, status='incomplete', cleanup_verified=False,
             image=f['image'], counters=dict.fromkeys(g.LIMITS,0), outputs={}, restoration={}, views={})

    def budget(extra=0):
        require(time.monotonic()-started<g.RESOURCES['inner_seconds'],'Relighting wall limit')
        require(sum(p.stat().st_size for p in Path('/outputs').rglob('*') if p.is_file())+extra
                <=g.RESOURCES['output_bytes'],'Relighting output limit')

    def charge(key, count):
        nonlocal last
        require(r['counters'][key]+count<=g.LIMITS[key],'Relighting work cap: '+key)
        if time.monotonic()-last>=5:
            budget()
            tmp = out/'progress.next'
            if tmp.exists():
                tmp.unlink()
            json_new(tmp,r)
            tmp.replace(out/'progress.json')
            last = time.monotonic()
        r['counters'][key] += count

    def save_capture(name, **values):
        budget(sum(v.nbytes for v in values.values())+65536)
        save(out/name,values)
        r['outputs'][name] = dict(bytes=(out/name).stat().st_size,sha256=g.digest(out/name))

    def work():
        torch.set_num_threads(g.RESOURCES['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(g.RESOURCES['cuda_allocator_fraction'])
        activate('material',PINS['material'])
        from raytracing import RayTracer
        from network import field, renderer
        from scientific_variants.donut_tensor_shader_d8 import tensor_shader
        from scientific_variants.donut_tensor_initialization_d9 import weak_start_shader

        geom = numeric(paths['oracle'])
        tracer = RayTracer(geom['vertices_native'].astype('f4'),geom['triangles'].astype('i4'))

        def trace(o,d):
            charge('native_shadow_slots',len(o))
            p,n,depth = tracer.trace(o,d)
            depth = depth.reshape(-1,1)
            require(bool(torch.isfinite(depth).all()),'Nonfinite native shadow depth')
            return p,n,depth,depth.reshape(-1)<10

        predictions = {}
        corrected,_ = native_classes(field.MCShadingNetwork,renderer.NeROMaterialRenderer,
                                     'A_corrected_core_v1',field.linear_to_srgb)
        tensor_class = weak_start_shader(tensor_shader(corrected,field.linear_to_srgb))
        for arm in g.ARMS:
            recipe = f['methods'][arm]
            tensor_arm = recipe['tensor_arm']
            shader = (corrected if tensor_arm is None else tensor_class)(deepcopy(SHADER),trace)
            if tensor_arm is not None:
                shader.select_tensor_representation(tensor_arm)
            shader.cuda()
            hist = receipts[arm]
            expected = dict(arm=arm,variant=recipe['variant'],completed_updates=g.STEP,
                            **{k:hist[k] for k in ('protocol','benchmark_sha256','freeze_sha256')})
            # The pinned 1.8 loader uses pickle. Admission binds each trusted file
            # by size/hash before this isolated CPU load. Restore weights only.
            charge('checkpoint_loads',1)
            saved = torch.load(paths[recipe['checkpoint']],map_location='cpu')
            require(all(saved[k]==v for k,v in expected.items()),'Foreign endpoint checkpoint header')
            state = saved['network_state_dict']
            target = shader.state_dict()
            require(state.keys()==target.keys() and all(v.device.type=='cpu' and v.shape==target[k].shape
                    and v.dtype==target[k].dtype and bool(torch.isfinite(v).all()) for k,v in state.items()),
                    'Strict checkpoint state inventory differs')
            shader.load_state_dict(state,strict=True)
            require(all(torch.equal(shader.state_dict()[k].detach().cpu(),v) for k,v in state.items()),
                    'Restored weights or buffers differ from checkpoint')
            del saved,state,target
            predictions[arm],record = native_pass(shader,arm,views,archives,torch,charge,save_capture)
            r['restoration'][arm] = dict(record,metadata=expected,strict=True,optimizer_restored=False,
                                        checkpoint_sha256=f['artifacts'][recipe['checkpoint']]['sha256'])
            print(f'{arm} restored and rendered on both cameras',flush=True)
            del shader

        total=0
        for camera,view in views.items():
            print(f'Independent relighting camera {camera}',flush=True)
            fields = {'oracle':view, **{arm:predictions[arm][camera] for arm in g.ARMS}}

            def fetch(rows, grid, prefix):
                r['in_progress'] = dict(camera=camera,prefix=prefix,grid=grid)
                with torch.no_grad():
                    result = integrate(view,fields,rows,grid,tracer,torch,charge)
                save_capture(f'camera-{camera}-{prefix}-{grid}.npz',**result)
                return result

            rgb,_,_,pixels = g.refine(view['pixel_indices'],fetch,total)
            total += len(pixels)
            r['views'][str(camera)] = dict(refined_pixels=pixels,
                convergence={arm:convergence(rgb[1,i,:,2],rgb[0,i,:,2]) for i,arm in enumerate(g.ROLES)})
        r.pop('in_progress',None)
        r['refined_pixels_total']=total

    try:
        require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1'
                and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count()==1
                and 'A100' in torch.cuda.get_device_name(0),'Reviewed A100 runtime required')
        r['runtime'] = dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0))
        r['backend'] = load('predicate_fp64',f['image'])
        require(r['backend']==f['backend'],'Corrected backend identity differs')
        r['quadrature_controls'] = controls()
        run_with_cleanup(work,torch)
        r['cleanup_verified']=True
        shadow = g.DIRECTIONS+r['refined_pixels_total']*sum(n*n for n in g.HIGH)
        require(r['counters']==dict(g.LIMITS,independent_shadow_directions=shadow,
                    independent_BRDF_point_directions=len(g.ROLES)*shadow),'Incomplete relighting work')
        g.freeze(args.freeze_sha256)
        g.admit(f,paths,exports)
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
        json_new(out/'receipt.json',r)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--commit')
    cli.add_argument('--authorization-sha256')
    cli.add_argument('--execute',action='store_true')
    args = cli.parse_args()
    f = g.freeze(args.freeze_sha256)
    paths,exports = g.base_paths(f,args.execute),g.endpoint_paths(f,args.execute)
    if args.execute:
        auth = Path('/authorization.json')
        require(args.commit and g.digest(auth)==args.authorization_sha256
                and g.read(auth)==g.authorization(f,args.commit,args.freeze_sha256),
                'Exact one-launch authorization including checkpoint trust required')
        require(os.geteuid()!=0 and all(os.statvfs(p).f_flag&os.ST_RDONLY
                    for p in (Path('/'),g.ROOT,auth,*paths.values(),*exports.values()))
                and not os.statvfs('/outputs').f_flag&os.ST_RDONLY
                and {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
                'Isolated read-only checkpoint runtime required')
    receipts,views,archives = g.admit(f,paths,exports)
    if args.execute:
        signal.signal(signal.SIGTERM,interrupted)
        execute(args,f,paths,exports,receipts,views,archives)
    else:
        print('Gate 6 metadata only: verified three endpoints and two admitted held-out rasters')


if __name__=='__main__':
    main()
