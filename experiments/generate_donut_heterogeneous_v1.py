"""One frozen heterogeneous Donut generation; corrected BVH for shadows only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_heterogeneous_v1 as d
from benchmarks.bell_e11 import convergence, linear_to_srgb
from benchmarks.beckmann_reflectance_v2 import integrate, local_views
from benchmarks.donut_recovery_v1 import json_new
from benchmarks.generate_donut_recovery_v1 import highlight_record
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest

ROOT = d.ROOT


def incident_grid(view, size, tracer, torch, charge, device='cuda'):
    """Same incident grid, kernels, visibility radius and reduction as Donut v2."""
    from benchmarks.bell_reference_e11 import basis, environment
    from experiments.bell_highlight_pilot_e11d import kernel_axes
    from experiments.bell_quadrature_e11e import midpoint_grid, query_visibility, weighted_components

    field = d.LIMITS
    local, weight = midpoint_grid(size)
    values = np.zeros((len(view['position']),3,3),np.float64)
    energy = np.zeros_like(values[:,0])
    visible_count = np.zeros(len(values),dtype=np.int64)
    valid = np.flatnonzero(view['shader_valid'])
    for start in range(0,len(valid),field['point_batch']):
        i = valid[start:start+field['point_batch']]
        pts, n, wo = (torch.from_numpy(view[name][i]).to(device) for name in ('position','normals','outgoing'))
        t,b = basis(n)
        material = dict(n=n, **{name:torch.from_numpy(view[key][i]).to(device)
                            for name,key in [('axis','axis'),('widths','widths_major_minor'),
                                             ('color','color'),('metallic','metallic')]})
        rgb = torch.zeros((len(i),2,3),dtype=torch.float64,device=device)
        R = torch.zeros_like(rgb[:,0])
        for first in range(0,len(local),field['direction_chunk']):
            loc = torch.from_numpy(local[first:first+field['direction_chunk']]).to(device)
            wi = loc[None,:,0,None]*t[:,None]+loc[None,:,1,None]*b[:,None]+loc[None,:,2,None]*n[:,None]
            count = len(i)*len(loc)
            charge('shadow_directions',count)
            visible,_ = query_visibility(torch,tracer,pts,wi)
            visible_count[i] += visible.sum(1).cpu().numpy()
            charge('BRDF_point_directions',count)
            fd,fs = kernel_axes(torch,wi,wo,material)
            cosine = (wi*n[:,None]).sum(-1)
            rgb += weighted_components(torch,fd,fs,environment(wi,d.ENVIRONMENT),cosine,visible,weight)
            R += ((fd+fs)*cosine[...,None]*weight).sum(1)
        values[i,:2] = rgb.cpu().numpy()
        values[i,2] = values[i,0]+values[i,1]
        energy[i] = R.cpu().numpy()
    return values, energy, visible_count


def slope_pair(view, charge):
    i = view['shader_valid']
    local = local_views(view['outgoing'][i],view['normals'][i],view['axis'][i])
    result = {}
    for order in d.ORDERS:
        slope = integrate(local,view['widths_major_minor'][i],view['color'][i],view['metallic'][i],order,
            charge=lambda cost,ok:charge('slope_nodes_returned' if ok else 'slope_nodes_attempted',cost))
        result[order] = {}
        for name in ('total','diffuse','specular'):
            array = np.zeros((len(i),3),dtype=np.float64)
            array[i] = slope[name]
            result[order][name] = array
    return result


def ordinary_probes(v):
    eligible = np.flatnonzero(v['shader_valid'] & ((v['normals']*v['outgoing']).sum(1)>=.25))
    require(len(eligible)>=8, 'Insufficient ordinary-angle cross-checks')
    return eligible[np.linspace(0,len(eligible)-1,8,dtype=int)]


def compare(v):
    valid = v['shader_valid']
    ordinary = ordinary_probes(v)
    return dict(
        radiance_convergence=convergence(v['linear_components'][valid,2],v['coarse_components'][valid,2]),
        reflectance_convergence=convergence(v['reflectance'][valid],v['coarse_reflectance'][valid]),
        ordinary_incident_convergence=convergence(v['incident_reflectance'][ordinary],v['coarse_incident_reflectance'][ordinary]),
        ordinary_slope_vs_incident=convergence(v['reflectance'][ordinary],v['incident_reflectance'][ordinary]))


def previews(out, camera, view):
    from PIL import Image
    rows = []
    for slot,name in enumerate(('diffuse','specular','RGB')):
        rgb = view['linear_components'][:,slot]
        canvas = np.zeros((d.SIZE*d.SIZE,3),dtype=np.float64)
        canvas[view['pixel_indices']] = np.clip(rgb,0,1)
        path = out/f'view-{camera}-{name}.png'
        Image.fromarray(linear_to_srgb(canvas.reshape(d.SIZE,d.SIZE,3))).save(path)
        rows.append(dict(component=name,channels_above_one=int((rgb>1).sum())))
    return rows


def execute(f, sha, commit, auth_sha, path_inputs, geometry, views):
    import torch
    from PIL import Image
    from benchmarks.generate_bell_e11 import run_with_cleanup
    from infrastructure.verified_ray_backend import load

    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1'
            and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count()==1
            and 'A100' in torch.cuda.get_device_name(0), 'Reviewed CUDA environment required')
    out = Path('/outputs/reference')
    out.mkdir()
    started = time.monotonic()
    work_keys = ('primary_rays','CPU_primary_crosschecks','shadow_directions','BRDF_point_directions',
                 'slope_nodes_attempted','slope_nodes_returned')
    report = dict(protocol=d.PROTOCOL,status='incomplete',freeze_sha256=sha,commit=commit,
                  authorization_sha256=auth_sha,image=d.IMAGE,source_generation_sha256=d.SOURCE_REPORT,
                  source_admission_sha256=d.SOURCE_ADMISSION,source_view_records=f['geometry_view_records'],
                  panel=d.PANEL,splits=d.SPLITS,field=d.FIELD,environment=d.ENVIRONMENT,
                  limits=d.LIMITS,tolerances=d.TOLERANCES,grids=list(d.GRIDS),slope_orders=list(d.ORDERS),
                  BRDF=d.BRDF,indirect='zero',point_light='absent',cleanup_verified=False,
                  scientific_work_complete=False,counters=dict.fromkeys(work_keys,0),views=[],
                  training_authorized=False,optimizer_returns=0,native_model_calls=0)
    last_check = started

    def budget():
        require(time.monotonic()-started < d.LIMITS['inner_seconds'], 'Reference runtime exhausted')
        require(sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())<d.LIMITS['output_bytes'],
                'Reference output limit reached')

    def charge(key,count):
        nonlocal last_check
        require(report['counters'][key]+count<=d.LIMITS[key], 'Reference work limit: '+key)
        report['counters'][key]+=count
        now = time.monotonic()
        if now-last_check>=5:
            budget()
            path = out/'progress.next'
            path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            path.replace(out/'progress.json')
            last_check=now

    def work():
        torch.set_num_threads(d.LIMITS['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(d.LIMITS['cuda_allocator_fraction'])
        report['backend'] = load('predicate_fp64',d.IMAGE)
        activate('material',PINS['material'])
        import raytracing
        report['runtime']=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
                               numpy=np.__version__)
        tracer = raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),
                                     geometry['triangles'].astype(np.int32))
        labels, radius, area = d.labels(geometry,views)
        report['ring_radius_native'] = radius
        np.savez_compressed(out/'surface-area.npz',triangle_area_native=area)
        for camera,v in labels.items():
            report['camera_in_progress'] = camera
            print(f'Donut heterogeneous camera {camera} started',flush=True)
            slopes = slope_pair(v,charge)
            coarse,fine = (slopes[key] for key in d.ORDERS)
            valid = v['shader_valid']
            gate = convergence(fine['total'][valid],coarse['total'][valid])
            if not gate['passed']:
                np.savez_compressed(out/f'view-{camera}-slope-failure.npz',**v,
                                    reflectance=fine['total'],coarse_reflectance=coarse['total'])
                report['failed_slope_convergence'] = gate
                raise ValueError(f'Camera {camera} slope convergence failed')
            with torch.no_grad():
                lo, le, _ = incident_grid(v,d.GRIDS[0],tracer,torch,charge)
                hi, he, visibility = incident_grid(v,d.GRIDS[1],tracer,torch,charge)
            highlight,mask = highlight_record(hi[valid])
            highlight_mask = np.zeros(len(valid),dtype=bool)
            highlight_mask[valid] = mask
            v.update(linear_components=hi,coarse_components=lo,visible_counts=visibility,
                     incident_reflectance=he,coarse_incident_reflectance=le,
                     reflectance=fine['total'],coarse_reflectance=coarse['total'],
                     reflectance_diffuse=fine['diffuse'],reflectance_specular=fine['specular'],
                     highlight_mask=highlight_mask)
            np.savez_compressed(out/f'view-{camera}.npz',**v)
            gates = compare(v)
            report['views'].append(dict(camera=camera,valid=int(valid.sum()),**gates,
                   geometry_hits=len(valid),ordinary_probe_pixels=v['pixel_indices'][ordinary_probes(v)].tolist(),
                   maximum_reflectance=float(fine['total'].max()),reflectance_channels_above_one=int((fine['total']>1).sum()),
                   highlight_diagnostics=highlight))
            require(all(g['passed'] for g in gates.values()), f'Camera {camera} reference convergence failed')
            report['views'][-1]['previews'] = previews(out,camera,v)
            print(f'Donut heterogeneous camera {camera} verified',flush=True)

    try:
        run_with_cleanup(work,torch)
        report['cleanup_verified'] = True
        require(report['counters']=={k:d.LIMITS[k] for k in work_keys}, 'Incomplete reference work accounting')
        d.freeze(ROOT/d.FREEZE,sha)
        d.source(f,*path_inputs)
        sheet = Image.new('RGB',(d.SIZE*3,d.SIZE*len(d.PANEL)))
        for row,c in enumerate(d.PANEL):
            for col,tag in enumerate(('RGB','diffuse','specular')):
                with Image.open(out/f'view-{c}-{tag}.png') as image:
                    sheet.paste(image,(col*d.SIZE,row*d.SIZE))
        sheet.save(out/'reference-components.png')
        budget()
        report['outputs'] = {p.name:dict(bytes=p.stat().st_size,sha256=digest(p))
                             for p in sorted(out.iterdir()) if p.name not in ('progress.json','progress.next')}
        report.update(status='complete_pending_researcher_admission',scientific_work_complete=True)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        import resource
        report['wall_seconds'] = time.monotonic()-started
        report['host_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch.cuda.is_initialized():
            report['CUDA_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['CUDA_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        json_new(out/'generation.json',report)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--commit')
    cli.add_argument('--authorization-sha256')
    cli.add_argument('--execute',action='store_true')
    a = cli.parse_args()
    f = d.freeze(ROOT/d.FREEZE,a.freeze_sha256)
    if a.execute:
        paths = tuple(Path(p) for p in ('/inputs/oracle','/inputs/cameras.json',
                                       '/inputs/reference','/inputs/admission.json'))
        auth = Path('/authorization.json')
        require(a.commit and digest(auth)==a.authorization_sha256
                and d.read(auth)==d.authorization(f,a.commit,a.freeze_sha256),
                'Exact one-launch generation authorization required')
        require(os.getuid()!=0 and all(os.statvfs(p).f_flag&os.ST_RDONLY for p in
                [*paths,ROOT,Path('/'),auth]) and not os.statvfs('/outputs').f_flag&os.ST_RDONLY
                and {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
                'Expected networkless non-root runtime with read-only inputs')
    else:
        paths = tuple(Path(f['host_inputs'][key]) for key in ('oracle','cameras','reference','admission'))
    geometry,views = d.source(f,*paths)
    if a.execute:
        execute(f,a.freeze_sha256,a.commit,a.authorization_sha256,paths,geometry,views)
    else:
        fields,radius,_ = d.labels(geometry,views)
        print(json.dumps(dict(status='metadata_only',radius=radius,points={
            c:int(v['shader_valid'].sum()) for c,v in fields.items()})))


if __name__=='__main__':
    main()
