"""One authorized oracle generation; no native material model or Blender calls."""
import json
import time
import traceback
import numpy as np

from benchmarks.donut_recovery_v1 import (REFERENCE, REFERENCE_LIMITS as LIMITS,
    SIZE, GRIDS, COLOR, METALLIC, WIDTHS, PANEL, SPLITS, ENVIRONMENT, INPUTS,
    inputs, camera_rays, surface_frames, runtime, json_new, verify_freeze)
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest


def highlight_record(components):
    """Keep the existing scalar summary in JSON and the Boolean mask in NPZ."""
    from experiments.bell_highlight_e11d import measures
    summary,mask=measures(components)
    require(isinstance(summary,dict) and mask.dtype==np.bool_ and mask.shape==(len(components),),
            'Unexpected saved-component highlight result')
    json.dumps(summary,allow_nan=False)  # No generic ndarray/string fallback.
    return summary,mask


def generate(args):
    freeze,_=runtime(args,'reference')
    geometry,meta,records=inputs(args.oracle,args.cameras)
    args.output.mkdir();(args.output/'runtime/cache').mkdir(parents=True);started=time.monotonic()
    report=dict(schema_version=1,protocol=REFERENCE,status='incomplete',commit=args.commit,
        freeze_sha256=args.freeze_sha256,inputs=records,panel=PANEL,splits=SPLITS,
        material=dict(color=COLOR,metallic=METALLIC,widths_major_minor=WIDTHS,
                      orientation='exported world principal projector interpolated with smooth shading normals'),
        BRDF='Kurt-a0-metallic-additive-v1',environment=ENVIRONMENT,
        indirect='zero',point_light='absent',coordinates='original A half-scale Blender world',
        native_model_calls=0,optimizer_returns=0,Blender_calls=0,
        cleanup_verified=False,views=[],counters=dict(primary_rays=0,shadow_directions=0,BRDF_point_directions=0,CPU_primary_crosschecks=0))
    def bump(key,count):
        require(time.monotonic()-started<LIMITS['inner_seconds'],'Reference runtime exhausted')
        require(report['counters'][key]+count<=LIMITS[key],'Reference work cap: '+key)
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'], 'Reference output cap')
        report['counters'][key]+=count
    try:
        import torch
        from infrastructure.verified_ray_backend import load
        from benchmarks.generate_bell_e11 import run_with_cleanup
        from benchmarks.bell_geometry_e11 import intersect
        from benchmarks.bell_reference_e11 import basis,environment
        from benchmarks.bell_e11 import convergence,linear_to_srgb
        from experiments.bell_highlight_pilot_e11d import kernel_axes
        from experiments.bell_quadrature_e11e import midpoint_grid,controls,weighted_components
        require(torch.__version__=='1.8.1+cu111' and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0), 'Reviewed native GPU runtime required')
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
        report['runtime']=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
            CXX11_ABI=bool(torch._C._GLIBCXX_USE_CXX11_ABI),numpy=np.__version__,declared_image=args.image_id)
        report['backend']=load('predicate_fp64',args.image_id)
        activate('material',PINS['material'])
        import raytracing
        tracer=raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),geometry['triangles'].astype(np.int32))
        report['quadrature_controls']=controls()
        def work():
            from PIL import Image
            for camera in PANEL:
                origin,direction,calibration=camera_rays(meta,camera)
                bump('primary_rays',len(origin))
                with torch.no_grad():
                    pos,raw,depth=tracer.trace(torch.from_numpy(origin.astype(np.float32)).cuda(),torch.from_numpy(direction.astype(np.float32)).cuda())
                pos,raw,depth=(v.cpu().numpy() for v in (pos,raw,depth));depth=depth.reshape(-1)
                require(np.isfinite(depth).all() and (depth>=0).all(),'Invalid native primary depths')
                hit=depth<10; ids=np.flatnonzero(hit);require(len(ids)>0,'Empty oracle image')
                p=pos[ids].astype(np.float64)
                require(abs(p-(origin[ids]+depth[ids,None]*direction[ids])).max()<=2e-5,'Primary ray/position mismatch')
                frame=surface_frames(geometry,p);wo=-direction[ids];valid=(wo*frame['normals']).sum(-1)>0
                # Domain is the existing one-sided BRDF, not a normal flip or a
                # favorable mask: preserve ALL geometry hits and label validity.
                require(valid.any(),'No upper-hemisphere shading observations')
                checks=[]
                for index in np.linspace(0,len(ids)-1,8).astype(int):
                    bump('CPU_primary_crosschecks',1)
                    check=intersect(origin[ids[index]],direction[ids[index]],geometry['vertices_native'],geometry['triangles'])
                    require(check is not None,'Independent primary hit missing')
                    error=abs(check['depth']-float(depth[ids[index]]))
                    require(error<=2e-4,'Independent closest-hit depth mismatch')
                    checks.append(dict(pixel=int(ids[index]),depth_error=error))
                row=dict(camera=camera,calibration=calibration,geometry_hits=len(ids),
                    shader_valid=int(valid.sum()),one_sided_zero_RGB=int((~valid).sum()),
                    max_surface_residual=float(frame['surface_residual'].max()),CPU_checks=checks,grids=[])
                report['views'].append(row)
                results={};energies={};shadow_counts={}
                for grid in GRIDS:
                    local,weight=midpoint_grid(grid); N=len(local)
                    values=np.zeros((len(ids),3,3),np.float64);energy=np.zeros((len(ids),3),np.float64)
                    visibility_count=np.zeros(len(ids),np.int64)
                    selected=np.flatnonzero(valid)
                    for first in range(0,len(selected),LIMITS['point_batch']):
                        ix=selected[first:first+LIMITS['point_batch']]
                        n=torch.from_numpy(frame['normals'][ix]).cuda();view=torch.from_numpy(wo[ix]).cuda()
                        points=torch.from_numpy(p[ix]).cuda();t,b=basis(n)
                        material=dict(n=n,axis=torch.from_numpy(frame['axis'][ix]).cuda(),
                            widths=n.new_tensor(WIDTHS).expand(len(ix),2),color=n.new_tensor(COLOR).expand(len(ix),3),
                            metallic=n.new_full((len(ix),1),METALLIC))
                        pair=torch.zeros((len(ix),2,3),dtype=torch.float64,device='cuda');R=torch.zeros_like(pair[:,0])
                        for start in range(0,N,LIMITS['direction_chunk']):
                            loc=torch.from_numpy(local[start:start+LIMITS['direction_chunk']]).cuda()
                            wi=loc[None,:,0,None]*t[:,None]+loc[None,:,1,None]*b[:,None]+loc[None,:,2,None]*n[:,None]
                            count=len(ix)*len(loc);bump('shadow_directions',count)
                            wf=wi.float();o=points.float()[:,None]+1e-5*wf
                            _,_,dep=tracer.trace(o,wf);visible=dep>=10
                            require(bool(torch.isfinite(dep).all()),'Nonfinite reference shadow depths')
                            visibility_count[ix]+=visible.sum(1).cpu().numpy()
                            bump('BRDF_point_directions',count)
                            fd,fs=kernel_axes(torch,wi,view,material);cosine=(wi*n[:,None]).sum(-1)
                            pair+=weighted_components(torch,fd,fs,environment(wi,ENVIRONMENT),cosine,visible,weight)
                            R+=((fd+fs)*cosine[...,None]*weight).sum(1)
                        values[ix,:2]=pair.cpu().numpy();values[ix,2]=values[ix,0]+values[ix,1]
                        energy[ix]=R.cpu().numpy()
                        if first%128==0:
                            progress=dict(status='incomplete',camera=camera,grid=grid,
                                completed_shading_points=first+len(ix),counters=report['counters'].copy())
                            tmp=args.output/'progress.next';tmp.write_text(json.dumps(progress)+'\n');tmp.replace(args.output/'progress.json')
                            print(json.dumps(progress),flush=True)
                    require(np.isfinite(values).all() and (values>=0).all() and np.isfinite(energy).all() and (energy>=0).all(), 'Invalid independent image/energy')
                    results[grid]=values;energies[grid]=energy;shadow_counts[grid]=visibility_count
                    row['grids'].append(dict(size=grid,valid_points=int(valid.sum()),directions_per_point=N))
                cg=convergence(results[512][valid,2],results[256][valid,2])
                eg=convergence(energies[512][valid],energies[256][valid])
                highlight_summary,strong=highlight_record(results[512][valid])
                highlight_mask=np.zeros(len(ids),bool);highlight_mask[valid]=strong
                row.update(radiance_convergence=cg,energy_convergence=eg,
                    highlight_diagnostics=highlight_summary,
                    directional_reflectance_max=energies[512][valid].max(0).tolist(),
                    reflectance_channels_above_one=int((energies[512][valid]>1).sum()),
                    energy_interpretation='Unoccluded directional-hemispherical reflectance; additive core has no conservation guarantee; never normalized or clipped.')
                np.savez_compressed(args.output/('view-%d.npz'%camera),pixel_indices=ids,
                    hit_mask=hit.reshape(SIZE,SIZE),shader_valid=valid,position=p,outgoing=wo,
                    raw_BVH_normals=raw[ids],depth=depth[ids],**frame,
                    color=np.broadcast_to(COLOR,(len(ids),3)),metallic=np.full((len(ids),1),METALLIC),
                    widths_major_minor=np.broadcast_to(WIDTHS,(len(ids),2)),
                    highlight_mask=highlight_mask,
                    linear_components=results[512],coarse_components=results[256],
                    reflectance=energies[512],coarse_reflectance=energies[256],
                    visible_counts=shadow_counts[512])
                image=np.zeros((SIZE*SIZE,3));image[ids]=results[512][:,2]
                Image.fromarray(linear_to_srgb(image.reshape(SIZE,SIZE,3))).save(args.output/('view-%d-preview.png'%camera))
                for component,name in ((0,'diffuse'),(1,'specular')):
                    image=np.zeros((SIZE*SIZE,3));image[ids]=results[512][:,component]
                    Image.fromarray(linear_to_srgb(image.reshape(SIZE,SIZE,3))).save(args.output/('view-%d-%s-preview.png'%(camera,name)))
                print(json.dumps(row),flush=True)
                require(cg['passed'] and eg['passed'],'Independent image/energy convergence failed; preserve arrays, no tuning/retry')
            return {}
        run_with_cleanup(work,torch);del tracer
        report['cleanup_verified']=True
        from PIL import Image
        sheet=Image.new('RGB',(3*SIZE,len(PANEL)*SIZE))
        for row,camera in enumerate(PANEL):
            for column,suffix in enumerate(('preview','diffuse-preview','specular-preview')):
                with Image.open(args.output/('view-%d-%s.png'%(camera,suffix))) as image:
                    sheet.paste(image,(column*SIZE,row*SIZE))
        sheet.save(args.output/'reference-RGB-diffuse-specular.png')
        import resource
        report['resources']=dict(CUDA_peak_allocated=torch.cuda.max_memory_allocated(),
            CUDA_peak_reserved=torch.cuda.max_memory_reserved(),host_peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        require(report['counters']['primary_rays']==LIMITS['primary_rays'] and report['counters']['CPU_primary_crosschecks']==48,'Incomplete camera panel/crosschecks')
        expected=sum(r['shader_valid'] for r in report['views'])*sum(g*g for g in GRIDS)
        require(report['counters']['shadow_directions']==report['counters']['BRDF_point_directions']==expected,'Incomplete integration work')
        inputs(args.oracle,args.cameras);verify_freeze(args.freeze,args.freeze_sha256)
        report['outputs']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir()
                           if p.is_file() and p.name not in ('progress.json','progress.next')}
        report['status']='complete_pending_researcher_admission'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        json_new(args.output/'generation.json',report)
