"""One separately approved Bell fixed-material rendering batch; no models/loads."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks import bell_geometry_rgb_gate12 as g
from benchmarks.generate_bell_e11 import run_with_cleanup
from experiments.donut_reconstructed_geometry_v1 import interrupted,save
from infrastructure.signatures import require
from infrastructure.stage_sources import activate,PINS

VISIBILITY_FIELDS=('own_visible','source_visible','xor','own_open_source_blocked','own_blocked_source_open',
                   'own_blocker_radial_band','source_blocker_radial_band',
                   'light_weighted_xor_R','light_weighted_xor_G','light_weighted_xor_B',
                   'unoccluded_Li_cos_R','unoccluded_Li_cos_G','unoccluded_Li_cos_B')


def capture_name(camera,arm,prefix,size):return f'{camera}-{arm}-{prefix}-{size}.npz'


def integrate(view,size,own_tracer,oracle_tracer,is_oracle,torch,charge,band,device_name='cuda'):
    """Accepted hemisphere nodes/kernel. Source visibility shares WORLD directions.

    Primary uses the same global analytic field on each geometry. Transported
    source controls are camera-conditioned diagnostics, never primary labels.
    """
    from benchmarks.bell_reference_e11 import basis,environment
    from experiments.bell_highlight_pilot_e11d import kernel_axes
    from experiments.bell_quadrature_e11e import midpoint_grid,query_visibility,weighted_components
    local,weight=midpoint_grid(size);count=len(view['pixels'])
    values=np.zeros((count,len(g.KINDS),3,3),np.float64);stats=np.zeros((count,len(VISIBILITY_FIELDS)),np.float64)
    for start in range(0,count,4):
        ix=slice(start,min(start+4,count))
        def device(key):return torch.from_numpy(np.ascontiguousarray(view[key][ix])).to(device_name)
        p,n,wo,source=(device(k) for k in ('position','normals','outgoing','source_position'))
        t,b=basis(n);material=dict(n=n,axis=device('axis'),widths=device('widths_major_minor'),
                                  color=device('color'),metallic=device('metallic'))
        transported=material if is_oracle else dict(n=n,axis=device('transported_axis'),widths=device('transported_widths_major_minor'),
                                                color=device('transported_color'),metallic=device('transported_metallic'))
        accum=torch.zeros((len(p),len(g.KINDS),2,3),dtype=torch.float64,device=device_name)
        vis=torch.zeros((len(p),len(VISIBILITY_FIELDS)),dtype=torch.float64,device=device_name)
        for first in range(0,len(local),16384):
            loc=torch.from_numpy(local[first:first+16384]).to(device_name)
            wi=loc[None,:,0,None]*t[:,None]+loc[None,:,1,None]*b[:,None]+loc[None,:,2,None]*n[:,None]
            work=len(p)*len(loc);charge('shadow_directions',work)
            own,depth=query_visibility(torch,own_tracer,p,wi)
            if is_oracle:src,source_depth=own,depth
            else:
                charge('shadow_directions',work);src,source_depth=query_visibility(torch,oracle_tracer,source,wi)
            charge('BRDF_point_directions',work);fd,fs=kernel_axes(torch,wi,wo,material)
            cosine=(wi*n[:,None]).sum(-1);charge('environment_point_directions',work);light=environment(wi,g.ENV)
            accum[:,0]+=weighted_components(torch,fd,fs,light,cosine,own,weight)
            accum[:,1]+=weighted_components(torch,fd,fs,light,cosine,src,weight)
            accum[:,2]+=weighted_components(torch,fd,fs,light,cosine,torch.ones_like(own),weight)
            if is_oracle:td,ts=fd,fs
            else:
                charge('BRDF_point_directions',work);td,ts=kernel_axes(torch,wi,wo,transported)
            accum[:,3]+=weighted_components(torch,td,ts,light,cosine,own,weight)
            accum[:,4]+=weighted_components(torch,td,ts,light,cosine,torch.ones_like(own),weight)
            own_hit=p.float()[:,None]+(1e-5+depth)[...,None]*wi.float()
            source_hit=source.float()[:,None]+(1e-5+source_depth)[...,None]*wi.float()
            own_band=(~own)&((torch.linalg.norm(own_hit,dim=-1)-1).abs()<=band)
            source_band=(~src)&((torch.linalg.norm(source_hit,dim=-1)-1).abs()<=band)
            masks=(own,src,own^src,own&~src,~own&src,own_band,source_band)
            for k,mask in enumerate(masks):vis[:,k]+=mask.sum(1)
            weighted=light*cosine[...,None]*weight
            vis[:,7:10]+=(weighted*(own^src)[...,None]).sum(1);vis[:,10:13]+=weighted.sum(1)
        components=accum.cpu().numpy();values[ix,:,:2]=components;values[ix,:,2]=components.sum(2)
        stats[ix]=vis.cpu().numpy()
    return dict(components=values,visibility=stats,pixels=view['pixels'],grid=np.array(size))


def undefined_visibility(view,size,tracer,oracle_tracer,torch,charge):
    """Geometry-only check at unmatched hits; no fabricated material or RGB.

    Both tracers receive the same reconstructed location/direction. Such an
    origin can be inside the oracle: this is not a material correspondence.
    """
    from benchmarks.bell_reference_e11 import basis
    from experiments.bell_quadrature_e11e import midpoint_grid,query_visibility
    local,_=midpoint_grid(size);counts=np.zeros((len(view['pixels']),5),np.int64)
    for start in range(0,len(counts),4):
        ix=slice(start,start+4);p=torch.from_numpy(view['position'][ix]).cuda();n=torch.from_numpy(view['normals'][ix]).cuda()
        t,b=basis(n)
        for first in range(0,len(local),16384):
            d=torch.from_numpy(local[first:first+16384]).cuda()
            wi=d[None,:,0,None]*t[:,None]+d[None,:,1,None]*b[:,None]+d[None,:,2,None]*n[:,None]
            charge('shadow_directions',2*len(p)*len(d))
            a,_=query_visibility(torch,tracer,p,wi);b0,_=query_visibility(torch,oracle_tracer,p,wi)
            counts[ix]+=torch.stack([x.sum(1) for x in (a,b0,a^b0,a&~b0,~a&b0)],-1).cpu().numpy()
    return dict(pixels=view['pixels'],counts=counts,grid=np.array(size))


def validate_capture(v,size,a):
    n=len(v['pixels']);x=a['components'];s=a['visibility']
    require(set(a)=={'components','visibility','pixels','grid'} and int(a['grid'])==size and
            np.array_equal(a['pixels'],v['pixels']) and x.shape==(n,len(g.KINDS),3,3) and s.shape==(n,13) and
            x.dtype==s.dtype==np.float64 and np.isfinite(x).all() and np.isfinite(s).all() and
            (x>=0).all() and (s>=0).all() and np.array_equal(x[:,:,0]+x[:,:,1],x[:,:,2]) and
            (s[:,:7]<=size**2).all() and np.array_equal(s[:,:7],np.floor(s[:,:7])) and
            np.array_equal(s[:,2],s[:,3]+s[:,4]),'Invalid Gate 12 capture')


def execute(f,args):
    auth=Path('/authorization.json')
    require(g.digest(auth)==args.authorization_sha256 and
            g.read(auth)==g.authorization(f,args.commit,args.freeze_sha256),'Exact separate Gate 12 approval required')
    p=g.paths(True);g.admit(p);out=Path('/outputs/render');out.mkdir()
    require(os.getuid()!=0 and all(os.statvfs(x).f_flag&os.ST_RDONLY for x in
            (Path('/'),g.ROOT,auth,*p.values())) and not os.statvfs(out).f_flag&os.ST_RDONLY and
            {x.split(':')[0].strip() for x in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Read-only networkless non-root runtime required')
    import torch
    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and not torch._C._GLIBCXX_USE_CXX11_ABI
            and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Reviewed CUDA runtime required')
    from infrastructure.verified_ray_backend import load
    backend=load('predicate_fp64',f['image']);require(backend==f['backend'],'Corrected BVH differs')
    activate('material',PINS['material'])
    from raytracing import RayTracer
    counters={k:0 for k in f['limits']};started=time.monotonic()
    report=dict(protocol=g.PROTOCOL,commit=args.commit,freeze_sha256=args.freeze_sha256,
        authorization_sha256=args.authorization_sha256,status='incomplete',cleanup_verified=False,
        backend=backend,counters=counters,views={},visibility_fields=VISIBILITY_FIELDS,
        handoff_sha256=f['handoff']['sha256'],geometry_admission_scope='perfect-synthetic-mask-assisted O1 derivative')
    report['runtime']=dict(torch=torch.__version__,CUDA=torch.version.cuda,python=sys.version.split()[0],
                           numpy=np.__version__,device=torch.cuda.get_device_name(0))
    def storage():
        require(sum(x.stat().st_size for x in out.parent.rglob('*') if x.is_file())<=f['resources']['output_bytes'],'Output limit')
    last=[0.]
    def charge(key,count):
        elapsed=time.monotonic()-started
        require(elapsed<f['resources']['inner_seconds'] and counters[key]+count<=f['limits'][key],'Gate 12 time/work limit '+key)
        counters[key]+=count
        if elapsed-last[0]>5:
            storage();last[0]=elapsed
            q=out/'progress.next';q.write_text(json.dumps(report,allow_nan=False)+'\n');q.replace(out/'progress.json')
    try:
        def work():
            torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
            meshes={arm:g.mesh_arrays(p[arm]/name) for arm,name in
                    [('oracle','geometry/mesh.ply'),('O0','domain-aware-300000.ply'),('O1','candidate.ply')]}
            tracers={arm:RayTracer(v.astype('f4'),faces.astype('i4')) for arm,(v,faces) in meshes.items()}
            vertices=meshes['oracle'][0];bounds=np.stack([vertices.min(0),vertices.max(0)]).astype(float)
            band=g.read(g.ROOT/g.INVENTORY)['domain_band'];total=0
            for camera in g.PANEL:
                print('Gate 12 camera '+camera+' started',flush=True)
                data=g.prepare(camera,p,bounds)
                expected=next(x for x in f['counts']['per_camera'] if x['camera']==camera)
                require(len(data['views']['oracle']['pixels'])==expected['oracle'] and
                    all(len(data['views'][arm]['pixels'])==expected[arm]['defined'] and
                        len(data['undefined_views'][arm]['pixels'])==expected[arm]['undefined'] for arm in ('O0','O1')),
                    'Frozen rendering/support inventory differs')
                for arm,values in data['transfers'].items():save(out/f'{camera}-{arm}-transfer.npz',values)
                for arm,v in data['undefined_views'].items():
                    for size in g.BASE:
                        with torch.no_grad():a=undefined_visibility(v,size,tracers[arm],tracers['oracle'],torch,charge)
                        save(out/f'{camera}-{arm}-undefined-{size}.npz',a)
                def fetch(arm,v,size,prefix):
                    with torch.no_grad():a=integrate(v,size,tracers[arm],tracers['oracle'],arm=='oracle',torch,charge,band)
                    validate_capture(v,size,a);save(out/capture_name(camera,arm,prefix,size),a);storage();return a
                states,selected=g.paired(data['views'],fetch,total);total+=len(selected)
                gates={arm:[g.convergence(s['components'][1,:,k,c],s['components'][0,:,k,c])
                            for k in range(len(g.KINDS)) for c in range(3)] for arm,s in states.items()}
                s=states['oracle'];replay=g.convergence(s['components'][1,:,0,2],data['archived'][s['pixels'],2])
                require(replay['passed'],'Oracle RGB replay differs from admitted v3 reference')
                report['views'][camera]=dict(refined_pixels=selected,convergence=gates,oracle_replay=replay)
                print('Gate 12 camera '+camera+' verified',flush=True)
            report['refined_pixels_total']=total
        run_with_cleanup(work,torch);report['cleanup_verified']=True
        require(set(report['views'])==set(g.PANEL),'Incomplete camera inventory')
        g.freeze(args.freeze_sha256);g.admit(p);storage();report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['outputs']={x.name:g.item(x) for x in sorted(out.glob('*.npz'))}
        g.write(out/'receipt.json',report)


def main():
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('phase',choices=('preflight','execute'))
    for key in ('commit','freeze-sha256','authorization-sha256'):c.add_argument('--'+key,required=True)
    c.add_argument('--authorization',type=Path);a=c.parse_args();f=g.freeze(a.freeze_sha256)
    if a.phase=='execute':execute(f,a);return
    require(a.authorization is not None and g.digest(a.authorization)==a.authorization_sha256 and
            g.read(a.authorization)==g.authorization(f,a.commit,a.freeze_sha256),'Separate exact execution approval required')
    require(not Path(f['host_output']).exists(),'Fresh output directory required');g.admit(g.paths())
    print(f['host_output']);print(f['image'])
    for key,value in g.paths().items():print(str(value)+'\t/inputs/'+key)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted);main()
