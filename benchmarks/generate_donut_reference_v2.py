"""One gated continuation: CPU slope validation, camera0 reuse, five new views."""
import json
import shutil
import time
import traceback
import numpy as np

from benchmarks.donut_recovery_v1 import (SIZE,GRIDS,PANEL,SPLITS,COLOR,METALLIC,WIDTHS,
    ENVIRONMENT,inputs,camera_rays,surface_frames,runtime,verify_freeze,json_new,numeric)
from benchmarks.donut_reference_v2 import PROTOCOL,LIMITS,previous,validation_indices
from benchmarks.beckmann_reflectance_v2 import ORDERS,NODES_PER_POINT,local_views,integrate,nodes
from benchmarks.generate_donut_recovery_v1 import highlight_record
from benchmarks.bell_e11 import convergence,linear_to_srgb
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS,activate,digest


def completed_view_arrays(data,low,high):
    """Replace only the reflectance reference; preserve every reused RGB label."""
    v=data['shader_valid'];summary,strong=highlight_record(data['linear_components'][v])
    require(low.shape==high.shape==(len(v),3) and np.isfinite(low).all() and np.isfinite(high).all(), 'Invalid completed reflectance arrays')
    mask=np.zeros(len(v),bool);mask[v]=strong
    out=dict(data,hemisphere_reflectance_diagnostic=data['reflectance'],coarse_hemisphere_reflectance_diagnostic=data['coarse_reflectance'],
        reflectance=high,coarse_reflectance=low,highlight_mask=mask)
    diffuse=np.zeros_like(high);diffuse[v]=data['color'][v]*(1-data['metallic'][v])
    out['reflectance_diffuse']=diffuse;out['reflectance_specular']=high-diffuse
    return out,summary


def render_remaining(camera,geometry,meta,tracer,torch,charge):
    """Same v1 primary/RGB quadrature and independent kernel, no new material law.

    Incident-grid reflectance is retained as a diagnostic from the already
    evaluated BRDF values. It no longer stands in for the validated slope rule.
    """
    from benchmarks.bell_geometry_e11 import intersect
    from benchmarks.bell_reference_e11 import basis,environment
    from experiments.bell_highlight_pilot_e11d import kernel_axes
    from experiments.bell_quadrature_e11e import midpoint_grid,weighted_components
    origin,direction,calibration=camera_rays(meta,camera)
    charge('primary_rays',len(origin))
    with torch.no_grad():
        pos,raw,depth=tracer.trace(torch.from_numpy(origin.astype(np.float32)).cuda(),torch.from_numpy(direction.astype(np.float32)).cuda())
    pos,raw,depth=(v.cpu().numpy() for v in (pos,raw,depth));depth=depth.reshape(-1)
    require(np.isfinite(depth).all() and (depth>=0).all(),'Invalid native primary depths')
    hit=depth<10;ids=np.flatnonzero(hit);require(len(ids)>0,'Empty oracle image')
    p=pos[ids].astype(np.float64)
    require(abs(p-(origin[ids]+depth[ids,None]*direction[ids])).max()<=2e-5,'Primary ray/position mismatch')
    frame=surface_frames(geometry,p);wo=-direction[ids];valid=(wo*frame['normals']).sum(-1)>0
    require(valid.any(),'No upper-hemisphere shading observations')
    checks=[]
    for i in np.linspace(0,len(ids)-1,8).astype(int):
        charge('CPU_primary_crosschecks',1)
        check=intersect(origin[ids[i]],direction[ids[i]],geometry['vertices_native'],geometry['triangles'])
        require(check is not None,'Independent primary hit missing')
        error=abs(check['depth']-float(depth[ids[i]]));require(error<=2e-4,'Independent closest-hit mismatch')
        checks.append(dict(pixel=int(ids[i]),depth_error=error))
    row=dict(camera=camera,calibration=calibration,geometry_hits=len(ids),shader_valid=int(valid.sum()),
             one_sided_zero_RGB=int((~valid).sum()),CPU_checks=checks,max_surface_residual=float(frame['surface_residual'].max()),grids=[])
    values_by_grid={};energy_by_grid={};counts_by_grid={}
    for grid in GRIDS:
        local,weight=midpoint_grid(grid);N=len(local)
        values=np.zeros((len(ids),3,3),np.float64);energy=np.zeros((len(ids),3),np.float64)
        visibility_count=np.zeros(len(ids),np.int64);selected=np.flatnonzero(valid)
        for first in range(0,len(selected),LIMITS['point_batch']):
            ix=selected[first:first+LIMITS['point_batch']]
            n=torch.from_numpy(frame['normals'][ix]).cuda();view=torch.from_numpy(wo[ix]).cuda()
            points=torch.from_numpy(p[ix]).cuda();t,b=basis(n)
            material=dict(n=n,axis=torch.from_numpy(frame['axis'][ix]).cuda(),widths=n.new_tensor(WIDTHS).expand(len(ix),2),
                          color=n.new_tensor(COLOR).expand(len(ix),3),metallic=n.new_full((len(ix),1),METALLIC))
            pair=torch.zeros((len(ix),2,3),dtype=torch.float64,device='cuda');R=torch.zeros_like(pair[:,0])
            for start in range(0,N,LIMITS['direction_chunk']):
                loc=torch.from_numpy(local[start:start+LIMITS['direction_chunk']]).cuda()
                wi=loc[None,:,0,None]*t[:,None]+loc[None,:,1,None]*b[:,None]+loc[None,:,2,None]*n[:,None]
                count=len(ix)*len(loc);charge('shadow_directions',count)
                wf=wi.float();o=points.float()[:,None]+1e-5*wf
                _,_,dep=tracer.trace(o,wf);visible=dep>=10
                require(bool(torch.isfinite(dep).all()),'Nonfinite shadow depth')
                visibility_count[ix]+=visible.sum(1).cpu().numpy();charge('BRDF_point_directions',count)
                fd,fs=kernel_axes(torch,wi,view,material);cosine=(wi*n[:,None]).sum(-1)
                pair+=weighted_components(torch,fd,fs,environment(wi,ENVIRONMENT),cosine,visible,weight)
                R+=((fd+fs)*cosine[...,None]*weight).sum(1)
            values[ix,:2]=pair.cpu().numpy();values[ix,2]=values[ix,0]+values[ix,1];energy[ix]=R.cpu().numpy()
        require(np.isfinite(values).all() and (values>=0).all() and np.isfinite(energy).all() and (energy>=0).all(),'Invalid reference values')
        values_by_grid[grid]=values;energy_by_grid[grid]=energy;counts_by_grid[grid]=visibility_count
        row['grids'].append(dict(size=grid,directions_per_point=N,valid_points=int(valid.sum())))
        print(json.dumps(dict(phase='new_RGB_grid_complete',camera=camera,grid=grid)),flush=True)
    data=dict(pixel_indices=ids,hit_mask=hit.reshape(SIZE,SIZE),shader_valid=valid,position=p,outgoing=wo,
        raw_BVH_normals=raw[ids],depth=depth[ids],**frame,color=np.broadcast_to(COLOR,(len(ids),3)),
        metallic=np.full((len(ids),1),METALLIC),widths_major_minor=np.broadcast_to(WIDTHS,(len(ids),2)),
        linear_components=values_by_grid[512],coarse_components=values_by_grid[256],
        reflectance=energy_by_grid[512],coarse_reflectance=energy_by_grid[256],visible_counts=counts_by_grid[512])
    row['radiance_convergence']=convergence(data['linear_components'][valid,2],data['coarse_components'][valid,2])
    return data,row


def generate(args):
    freeze,_=runtime(args,'reference')
    import os
    require(args.previous is not None and os.statvfs(args.previous).f_flag&os.ST_RDONLY,'Immutable previous reference mount required')
    old,evidence=previous(args.previous);geometry,meta,input_records=inputs(args.oracle,args.cameras)
    args.output.mkdir();(args.output/'runtime/cache').mkdir(parents=True)
    started=time.monotonic();torch=None;last_progress=[0.]
    report=dict(schema_version=1,protocol=PROTOCOL,status='incomplete',commit=args.commit,
        freeze_sha256=args.freeze_sha256,inputs=input_records,panel=PANEL,splits=SPLITS,
        declared_image_id=args.image_id,limits=LIMITS,scientific_work_complete=False,
        numpy_version=np.__version__,quadrature_dtype='float64',
        material=dict(color=COLOR,metallic=METALLIC,widths_major_minor=WIDTHS,orientation='unchanged exported world principal director'),
        BRDF='Kurt-a0-metallic-additive-v1',environment=ENVIRONMENT,indirect='zero',point_light='absent',
        coordinates='original A half-scale Blender world',reflectance_method='analytic Lambert + exact-support Gaussian slopes',
        reflectance_orders=list(ORDERS),previous_source_artifacts=evidence['original_artifacts'],
        previous_source_root=str(args.previous),previous_terminal_report_valid_JSON=False,
        cleanup_verified=False,native_model_calls=0,optimizer_returns=0,Blender_calls=0,
        views=[],validation={},phase='grazing_validation_before_any_new_rays',
        counters={k:0 for k in ('primary_rays','shadow_directions','BRDF_point_directions','CPU_primary_crosschecks','slope_nodes_attempted','slope_nodes_returned')})
    def charge(key,n):
        require(time.monotonic()-started<LIMITS['inner_seconds'],'Reference continuation runtime exhausted')
        require(report['counters'][key]+n<=LIMITS[key],'Reference continuation work cap: '+key)
        report['counters'][key]+=n
        if time.monotonic()-last_progress[0]>5:
            require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],'Output cap exceeded')
            tmp=args.output/'progress.next';tmp.write_text(json.dumps(report,allow_nan=False,indent=2)+'\n');tmp.replace(args.output/'progress.json')
            print(json.dumps(dict(phase=report['phase'],counters=report['counters'])),flush=True);last_progress[0]=time.monotonic()
    def slope_charge(n,returned):charge('slope_nodes_returned' if returned else 'slope_nodes_attempted',n)
    def pair(data,indices):
        views=local_views(data['outgoing'][indices],data['normals'][indices],data['axis'][indices])
        return {order:integrate(views,data['widths_major_minor'][indices],data['color'][indices],data['metallic'][indices],order,charge=slope_charge) for order in ORDERS}
    def save_view(data,row,low,high,reused=False):
        v=data['shader_valid'];out,summary=completed_view_arrays(data,low,high)
        row.update(energy_convergence=convergence(high[v],low[v]),highlight_diagnostics=summary,
            directional_reflectance_max=high[v].max(0).tolist(),reflectance_channels_above_one=int((high[v]>1).sum()),
            energy_interpretation='Independent slope-space unoccluded reflectance; never clipped/normalized',reused_camera0_RGB_and_geometry=reused)
        np.savez_compressed(args.output/('view-%d.npz'%row['camera']),**out)
        from PIL import Image
        for component,suffix in ((2,'preview'),(0,'diffuse-preview'),(1,'specular-preview')):
            name='view-%d-%s.png'%(row['camera'],suffix)
            if reused:
                shutil.copyfile(args.previous/'reference'/name,args.output/name)
                require(digest(args.output/name)==digest(args.previous/'reference'/name),'Reused PNG changed')
            else:
                image=np.zeros((SIZE*SIZE,3));image[data['pixel_indices']]=data['linear_components'][:,component]
                Image.fromarray(linear_to_srgb(image.reshape(SIZE,SIZE,3))).save(args.output/name)
        if reused:
            saved=numeric(args.output/'view-0.npz')
            names=[k for k in data if k not in ('reflectance','coarse_reflectance')]
            require(all(saved[k].dtype==data[k].dtype and saved[k].shape==data[k].shape and
                        saved[k].tobytes(order='C')==data[k].tobytes(order='C') for k in names),'Reused camera0 arrays changed')
            report['camera0_reuse']=dict(source_NPZ_sha256=digest(args.previous/'reference/view-0.npz'),
                unchanged_array_names=names,array_equality_verified=True,new_camera0_primary_rays=0,new_camera0_RGB_integrations=0,
                array_verification='dtype,shape,and exact C-order bytes',
                replaced_fields=['reflectance','coarse_reflectance'],old_reflectance_retained_as_diagnostic=True)
        require(row['radiance_convergence']['passed'] and row['energy_convergence']['passed'],'View convergence failed; no retry or threshold change')
    try:
        for order in ORDERS:
            u,wu,phi,wp=nodes(order)
            np.savez_compressed(args.output/('slope-nodes-%d.npz'%order),radial_CDF_nodes=u,
                                radial_CDF_weights=wu,quadrant_angles=phi,azimuth_weights=wp)
        grazing,ordinary=validation_indices(old)
        cached={order:np.zeros((len(old['pixel_indices']),3),np.float64) for order in ORDERS}
        done=np.zeros(len(old['pixel_indices']),bool)
        for name,indices in (('grazing',grazing),('ordinary',ordinary)):
            report['phase']=name+'_validation_before_any_new_rays'
            values=pair(old,indices)
            gate=convergence(values[128]['total'],values[64]['total'])
            result=dict(slope_two_grid=gate,pixel_indices=old['pixel_indices'][indices].tolist())
            if name=='ordinary':
                result['old_incident_two_grid']=convergence(old['reflectance'][indices],old['coarse_reflectance'][indices])
                result['slope_vs_independent_incident']=convergence(values[128]['total'],old['reflectance'][indices])
            report['validation'][name]=result
            np.savez_compressed(args.output/(name+'-validation.npz'),pixel_indices=old['pixel_indices'][indices],
                slope_coarse=values[64]['total'],slope_fine=values[128]['total'],analytic_diffuse=values[128]['diffuse'],
                incident_old_fine=old['reflectance'][indices],incident_old_coarse=old['coarse_reflectance'][indices],
                local_outgoing=local_views(old['outgoing'][indices],old['normals'][indices],old['axis'][indices]))
            json_new(args.output/(name+'-validation.json'),result)
            require(gate['passed'] and all(v['passed'] for k,v in result.items() if k in ('old_incident_two_grid','slope_vs_independent_incident')),
                    'Predeclared slope/independent incident validation failed; no new rays or generation')
            for order in ORDERS:cached[order][indices]=values[order]['total']
            done[indices]=True
        report['phase']='all_camera0_reflectance_no_RGB_or_ray_replay'
        remaining=np.flatnonzero(old['shader_valid']&~done);values=pair(old,remaining)
        for order in ORDERS:cached[order][remaining]=values[order]['total']
        c0=dict(camera=0,geometry_hits=len(done),shader_valid=int(old['shader_valid'].sum()),
            one_sided_zero_RGB=int((~old['shader_valid']).sum()),CPU_checks_reused_count=8,
            calibration=dict(focal_px=.5*SIZE/np.tan(float(meta['camera_angle_x'])/2),principal_point=[SIZE/2]*2,
                pixel_offset=0,c2w=meta['frames'][0]['transform_matrix'],world_to_native_scale=.5),
            calibration_source='unchanged original camera JSON; no primary rays recomputed',
            radiance_convergence=convergence(old['linear_components'][old['shader_valid'],2],old['coarse_components'][old['shader_valid'],2]))
        report['views'].append(c0);save_view(old,c0,cached[64],cached[128],True)
        report['phase']='conditional_remaining_five_RGB_views'
        import torch as torch_module
        torch=torch_module
        from benchmarks.generate_bell_e11 import run_with_cleanup
        from infrastructure.verified_ray_backend import load
        from experiments.bell_quadrature_e11e import controls
        require(torch.__version__=='1.8.1+cu111' and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Reviewed native runtime required')
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
        report['backend']=load('predicate_fp64',args.image_id);activate('material',PINS['material'])
        report['runtime']=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),numpy=np.__version__,declared_image=args.image_id)
        import raytracing
        tracer=raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),geometry['triangles'].astype(np.int32))
        report['quadrature_controls']=controls()
        def work():
            for camera in PANEL[1:]:
                report['phase']='remaining_camera_%d'%camera
                data,row=render_remaining(camera,geometry,meta,tracer,torch,charge);report['views'].append(row)
                if not row['radiance_convergence']['passed']:
                    np.savez_compressed(args.output/('view-%d-incomplete-RGB.npz'%camera),**data)
                    raise ValueError('Remaining camera RGB convergence failed; preserve data, no reflectance/training/retry')
                v=data['shader_valid'];indices=np.flatnonzero(v);values=pair(data,indices)
                arrays={order:np.zeros((len(v),3),np.float64) for order in ORDERS}
                for order in ORDERS:arrays[order][indices]=values[order]['total']
                save_view(data,row,arrays[64],arrays[128])
            return {}
        run_with_cleanup(work,torch);del tracer
        report['cleanup_verified']=True
        expected=sum(v['shader_valid'] for v in report['views'])*NODES_PER_POINT
        expected_RGB=sum(v['shader_valid'] for v in report['views'][1:])*(256**2+512**2)
        require(len(report['views'])==6 and report['counters']['primary_rays']==LIMITS['primary_rays'] and report['counters']['CPU_primary_crosschecks']==40 and
            report['counters']['slope_nodes_attempted']==report['counters']['slope_nodes_returned']==expected and
            report['counters']['shadow_directions']==report['counters']['BRDF_point_directions']==expected_RGB,'Incomplete scientific work')
        previous(args.previous);inputs(args.oracle,args.cameras);verify_freeze(args.freeze,args.freeze_sha256)
        from PIL import Image
        sheet=Image.new('RGB',(3*SIZE,len(PANEL)*SIZE))
        for row,camera in enumerate(PANEL):
            for column,suffix in enumerate(('preview','diffuse-preview','specular-preview')):
                with Image.open(args.output/('view-%d-%s.png'%(camera,suffix))) as image:sheet.paste(image,(column*SIZE,row*SIZE))
        sheet.save(args.output/'reference-RGB-diffuse-specular.png')
        report['outputs']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.is_file() and p.name not in ('progress.json','progress.next')}
        require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],'Terminal output cap exceeded')
        report.update(status='complete_pending_researcher_admission',scientific_work_complete=True)
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        if torch is None:report.update(cleanup_verified=True,cleanup_scope='CPU-only validation; no native CUDA runtime initialized')
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        import resource
        report['host_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch is not None and torch.cuda.is_initialized():
            report['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        if report['status']=='failed':
            report['partial_outputs']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p))
                for p in args.output.iterdir() if p.is_file() and p.name not in ('progress.json','progress.next')}
        json_new(args.output/'generation.json',report)
