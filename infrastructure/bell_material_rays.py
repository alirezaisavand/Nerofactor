"""Observe original A ray construction/BVH/normal negation without a shader.

Native dependencies are injected after isolation. No ray or normal is replaced.
Miss payloads can be unspecified by the BVH; finite checks apply to hits only.
"""
from contextlib import contextmanager
from types import SimpleNamespace, MethodType
import sys

from infrastructure.signatures import require

BOUNDS={'minimum_silhouette_iou':0.90,'minimum_front_facing_fraction':0.99,
        'normal_absolute_error':2e-6,'intersection_absolute_error':2e-4}


def summarize(arrays,mask,np):
    o,d,p,raw,n,depth,hit=[arrays[k] for k in ['origins','directions','positions','raw_normals','normals','depth','hit']]
    hit=hit.reshape(-1);depth=depth.reshape(-1);mask=mask.reshape(-1)
    require(hit.dtype==np.bool_ and mask.dtype==np.bool_ and len(mask)==len(hit)>0,'Invalid ray/mask shapes')
    require(all(v.shape==(len(hit),3) for v in [o,d,p,raw,n]) and depth.shape==hit.shape,'Invalid ray payload shapes')
    require(np.isfinite(o).all() and np.isfinite(d).all() and np.isfinite(depth).all(),'Nonfinite camera rays/depth')
    require(np.array_equal(hit,depth<10),'Original depth<10 hit policy changed')
    require(hit.any(),'No mesh intersections; camera/mesh compatibility blocked')
    require(all(np.isfinite(v[hit]).all() for v in [p,raw,n]) and (depth[hit]>0).all(),'Invalid hit geometry/depth')
    raw_length=np.linalg.norm(raw[hit],axis=-1)
    require((raw_length>0).all(),'Zero BVH hit normals')
    expected=-raw[hit]/np.maximum(raw_length[:,None],1e-12)
    flip_error=float(np.abs(n[hit]-expected).max())
    position_error=float(np.abs(p[hit]-(o[hit]+depth[hit,None]*d[hit])).max())
    dot_raw=(raw[hit]*(-d[hit])).sum(-1)/raw_length
    dot_final=(n[hit]*(-d[hit])).sum(-1)
    front=float((dot_final>0).mean());back_raw=float((dot_raw<0).mean())
    union=int((hit|mask).sum());intersection=int((hit&mask).sum())
    iou=intersection/union if union else 0.
    checks={'original_negation':flip_error<=BOUNDS['normal_absolute_error'],
            'unit_final_normals':float(np.abs(np.linalg.norm(n[hit],axis=-1)-1).max())<=BOUNDS['normal_absolute_error'],
            'intersection_consistency':position_error<=BOUNDS['intersection_absolute_error'],
            'front_facing_final_normals':front>=BOUNDS['minimum_front_facing_fraction'],
            'raw_normals_opposite_view':back_raw>=BOUNDS['minimum_front_facing_fraction'],
            'silhouette_alignment':iou>=BOUNDS['minimum_silhouette_iou']}
    return {'rays':len(hit),'hits':int(hit.sum()),'misses':int((~hit).sum()),'hit_rate':float(hit.mean()),
            'depth_minimum':float(depth[hit].min()),'depth_maximum':float(depth[hit].max()),
            'intersection_maximum_error':position_error,'normal_negation_maximum_error':flip_error,
            'final_normal_dot_view_minimum':float(dot_final.min()),'final_normal_dot_view_mean':float(dot_final.mean()),
            'final_front_facing_fraction':front,'raw_back_facing_fraction':back_raw,
            'mask_pixels':int(mask.sum()),'mask_intersection':intersection,'mask_union':union,'silhouette_iou':iou,
            'checks':checks,'passed':all(checks.values())}


class TraceCapture:
    """Scoped return observer, not a replacement for the original BVH/trace."""
    def __init__(self,owner,torch):
        self.owner=owner;self.torch=torch;self.raw=[];self.final=[]
        self.raw_code=owner.ray_tracer.trace.__func__.__code__;self.final_code=owner.trace.__func__.__code__
    def observe(self,frame,event,value):
        if event!='return' or value is None:return
        code=frame.f_code
        if code not in (self.raw_code,self.final_code):return
        require(not self.torch.is_grad_enabled(),'Ray diagnostic must remain no_grad')
        require(frame.f_locals['self'] is (self.owner.ray_tracer if code is self.raw_code else self.owner),'Foreign ray tracer')
        copy=lambda v:v.detach().cpu().numpy().copy()
        if code is self.raw_code:
            require(frame.f_locals.get('inplace') is False,'In-place BVH diagnostic refused')
            self.raw.append({'origins':copy(frame.f_locals['rays_o']),'directions':copy(frame.f_locals['rays_d']),
                             'positions':copy(value[0]),'raw_normals':copy(value[1]),'raw_depth':copy(value[2])})
        else:self.final.append({k:copy(v) for k,v in zip(['positions','normals','depth','hit'],value)})
        require(len(self.raw)<=1 and len(self.final)<=1,'Diagnostic view exceeded one original BVH batch')
    @contextmanager
    def active(self):
        require(sys.implementation.name=='cpython' and sys.getprofile() is None and sys.gettrace() is None,'Unobserved CPython required')
        sys.setprofile(self.observe)
        try:yield
        finally:sys.setprofile(None)
    def arrays(self,np):
        require(len(self.raw)==len(self.final)==1,'Missing original BVH/renderer trace return')
        raw,final=self.raw[0],self.final[0]
        require(np.array_equal(raw['positions'],final['positions'],equal_nan=True) and
                np.array_equal(raw['raw_depth'].reshape(-1),final['depth'].reshape(-1),equal_nan=True),'A modified BVH intersections/depth')
        return {**raw,**final}


def original_adapter(renderer,cfg):
    """Bind only original methods on an external I/O facade; no model/shader."""
    owner=SimpleNamespace(cfg={**renderer.default_cfg,**cfg},warned_normal=False)
    for name in ['_init_geometry','trace','trace_in_batch','_warn_ray_tracing','get_human_coordinate_poses','_construct_ray_batch']:
        setattr(owner,name,MethodType(getattr(renderer,name),owner))
    owner._init_geometry()
    return owner


def native_views(p,run,report,persist,torch,np):
    from network.renderer import NeROMaterialRenderer,build_imgs_info,imgs_info_to_torch
    from dataset.database import parse_database_name,get_database_split
    from infrastructure.bell_material_integration import verify_loaded_mesh,expected_splits
    owner=original_adapter(NeROMaterialRenderer,report['runtime_configuration'])
    verify_loaded_mesh(owner.mesh,p,np)
    database=parse_database_name('syn/bell',report['runtime_configuration']['dataset_dir'])
    train,val,nvs=get_database_split(database,'validation')  # Original call/seeding/shuffle, once.
    report['splits']={'train':train,'validation':val,'nvs':nvs}
    require(database.get_img_ids()==[str(i) for i in range(128)] and report['splits']==expected_splits(),
            'Unexpected original ordered bell split')
    report['effective_ray_renderer']=owner.cfg
    report['shader_constructed']=False
    report['views']=[]
    for role,image_id in [('train',train[0]),('validation',val[0]),('nvs',nvs[0])]:
        report['phase']='original_A_rays_'+role+'_'+image_id;persist()
        info=build_imgs_info(database,[image_id],False)
        mask=info['masks'][0].copy()
        require(info['imgs'].shape==(1,800,800,3) and mask.shape==(800,800),'Original full material resolution/mask required')
        info=imgs_info_to_torch(info,'cpu');capture=TraceCapture(owner,torch)
        with torch.no_grad(),capture.active():batch=owner._construct_ray_batch(info,is_train=False)
        arrays=capture.arrays(np)
        require(np.array_equal(batch['hit_mask'].numpy(),arrays['hit'].reshape(-1)),'Ray batch differs from observed trace')
        stats=summarize(arrays,mask,np)
        path=run.output('evaluations/'+role+'-'+image_id+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream,**arrays,mask=mask,K=info['Ks'].numpy(),pose=info['poses'].numpy(),rgb=info['imgs'].numpy())
        from infrastructure.stage_sources import digest
        report['views'].append({'role':role,'source_camera_id':image_id,'resolution':[800,800],
            'mask_source':'original masks_train/'+image_id+'.png > 0.5 (A get_mask)',
            'artifact':path.relative_to(run.root).as_posix(),'sha256':digest(path),'statistics':stats})
        persist()
    report['ray_gate_passed']=all(v['statistics']['passed'] for v in report['views'])
    require(report['ray_gate_passed'],'Ray/normal/silhouette gates failed; preserve evidence, do not flip/align/resize')
