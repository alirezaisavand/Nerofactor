"""Gate 12: saved-ray material transfer and a separately authorized RGB audit.

Metadata/NumPy only. Importing this module does not load Torch or a checkpoint.
"""
from pathlib import Path
import numpy as np
from benchmarks import bell_geometry_gate11 as prior
from benchmarks.bell_e11 import ENV, convergence, verify_labels
from benchmarks.bell_geometry_e11 import mesh_arrays
from benchmarks.donut_reconstructed_geometry_v1 import minimal_rotation, rotate_tensor
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest

PROTOCOL='bell-authenticated-geometry-rgb-gate12-v1'
FREEZE='docs/protocols/BELL_GEOMETRY_RGB_GATE12_V1.freeze.json'
INVENTORY='docs/results/publication_gate12/input-inventory.json'
HANDOFF='docs/results/publication_gate12/O1-controlled-handoff.json'
PANEL=tuple(prior.original.EVAL_IDS)
ARMS=('oracle','O0','O1')
KINDS=('primary','oracle_visibility','unoccluded','transported_source','transported_source_unoccluded')
BASE=(256,512)
HIGH=(1024,2048)
MAX_VIEW,MAX_TOTAL=32,128
RESOURCES=dict(cpu_threads=4,host_memory_gib=64,pids=128,cuda_allocator_fraction=.25,
               output_bytes=2*1024**3,watchdog_seconds=7200,inner_seconds=7080,kill_grace_seconds=30)
SOURCES=(
    'benchmarks/bell_geometry_rgb_gate12.py','experiments/bell_geometry_rgb_gate12.py',
    'experiments/verify_bell_geometry_rgb_gate12.py','experiments/run_bell_geometry_rgb_gate12.sh',
    'tests/scientific/test_bell_geometry_rgb_gate12.py','docs/protocols/BELL_GEOMETRY_RGB_GATE12_V1.md',
    'experiments/prepare_bell_geometry_rgb_gate12.py','benchmarks/bell_sam2_ordering_gate12.py',
    INVENTORY,HANDOFF,'docs/results/publication_gate12/sam2-ordering-v2.json',
    'benchmarks/bell_geometry_gate11.py','benchmarks/bell_geometry_gate10_v1.py',
    'benchmarks/bell_e11.py','benchmarks/bell_reference_e11.py','benchmarks/bell_geometry_e11.py',
    'benchmarks/donut_reconstructed_geometry_v1.py','benchmarks/donut_oracle_geometry_rgb_v1.py',
    'experiments/verify_donut_oracle_geometry_rgb_v1.py','experiments/bell_quadrature_e11e.py',
    'experiments/bell_highlight_pilot_e11d.py','experiments/bell_highlight_e11d.py',
    'benchmarks/generate_bell_e11.py','experiments/donut_reconstructed_geometry_v1.py',
    'infrastructure/verified_ray_backend.py','infrastructure/stage_sources.py','infrastructure/signatures.py',
    'scientific_sources/manifest-v1.json','scientific_sources/material_a/raytracing/raytracer.py',
    'docs/results/publication_gate102/results-v2/artifact-manifest.json',
    'docs/results/publication_gate11/results-v1/artifact-manifest.json',
)
read,write,item,check=prior.read,prior.write,prior.item,prior.check


def definition(out):
    inv=read(ROOT/INVENTORY)
    # One oracle kernel and two kernels per reconstructed geometry; five
    # visibility calls (oracle plus own/source pairs), with equal worst caps.
    points=inv['counts']['oracle_total']
    cost=5*points*sum(n*n for n in BASE)+5*MAX_TOTAL*sum(n*n for n in HIGH)
    return dict(protocol=PROTOCOL,host_output=str(out),panel=list(PANEL),arms=list(ARMS),kinds=list(KINDS),
        image=inv['image'],backend=inv['backend'],input_inventory=item(ROOT/INVENTORY),
        handoff=item(ROOT/HANDOFF),illumination=ENV,indirect='zero',exposure=1,
        baseline_grids=list(BASE),refined_grids=list(HIGH),max_refined_per_view=MAX_VIEW,
        max_refined_total=MAX_TOTAL,counts=inv['counts'],
        limits=dict(shadow_directions=cost+2*inv['counts']['undefined_total']*sum(n*n for n in BASE),
                    BRDF_point_directions=cost,
                    environment_point_directions=3*points*sum(n*n for n in BASE)+3*MAX_TOTAL*sum(n*n for n in HIGH),
                    primary_rays=0,
                    checkpoint_loads=0,native_models=0,optimizer_updates=0),resources=RESOURCES,
        sources={name:digest(ROOT/name) for name in SOURCES})


def freeze(sha):
    require(digest(ROOT/FREEZE)==sha,'Gate 12 freeze differs')
    f=read(ROOT/FREEZE);out=Path(f['host_output'])
    require(out.resolve()==out and out.parent==Path('/home/isavand/paper/evidence') and
            out.name.startswith('bell-geometry-rgb-gate12-') and f==definition(out),'Gate 12 definition differs')
    return f


def authorization(f,commit,sha):
    return dict(approved=True,protocol=PROTOCOL,commit=commit,freeze_sha256=sha,host_output=f['host_output'],
        image=f['image'],backend=f['backend'],handoff_sha256=f['handoff']['sha256'],selected_gpu='0',
        limits=f['limits'],resources=f['resources'],maximum_execution_launches=1,
        checkpoint_deserialization=False,training=False,retries=False)


def paths(native=False):
    inv=read(ROOT/INVENTORY)
    return {key:Path('/inputs')/key if native else Path(value) for key,value in inv['roots'].items()}


def admit(p):
    inv=read(ROOT/INVENTORY)
    O0=read(ROOT/'docs/results/publication_gate102/results-v2/artifact-manifest.json')['files']
    O1=read(ROOT/'docs/results/publication_gate11/results-v1/artifact-manifest.json')['native_files']
    for root,files in inv['files'].items():
        for name,identity in files.items():
            if root=='O0':require(identity==O0['run/'+name],'O0 differs from its accepted artifact identity')
            if root=='O1':require(identity==O1['evaluate/300000/'+name],'O1 differs from accepted Gate 11 artifacts')
            check(p[root]/name,identity)
    h=read(ROOT/HANDOFF)
    require(h['status']=='admitted_for_controlled_downstream_evaluation' and
            h['variant']==prior.VARIANT and h['perfect_synthetic_mask_assisted'] and
            not h['unchanged_original_G'] and not h['realistic_SAM2_validation'] and
            h['mesh']==inv['files']['O1']['candidate.ply'],'Variant-aware O1 admission required')
    require(digest(p['oracle']/'geometry/generation.json')==prior.original.GEN_SHA,'Admitted Bell generation differs')
    return inv


def tensor(axis,width,n):
    b=np.cross(n,axis)
    return width[:,0,None,None]**2*axis[:,:,None]*axis[:,None,:]+width[:,1,None,None]**2*b[:,:,None]*b[:,None,:]


def analytic(p,n,bounds):
    """Unchanged Bell analytic field; apply only on admitted rendering support."""
    q=(p-bounds.mean(0))/((bounds[1]-bounds[0]).max()/2)
    a=.23+.03*q[:,0];b=.35+.03*q[:,1];g=np.array([1.,2.,3.])/np.sqrt(14)
    u=g-(n@g)[:,None]*n;length=np.linalg.norm(u,axis=1)
    # At exact isotropy the director is undefined; a deterministic tangent
    # basis represents the same Q, never repairs a learned orientation.
    t=np.cross(n,np.eye(3)[np.abs(n).argmin(1)]);t/=np.linalg.norm(t,axis=1,keepdims=True)
    axis=np.divide(u,length[:,None],out=t.copy(),where=length[:,None]>1e-12)
    width=np.stack([np.sqrt(a*a+b*b*length*length),a],-1)
    color=np.stack([.35+.08*q[:,0],.50+.06*q[:,1],.20+.04*q[:,2]],-1)
    metal=(.25+.05*np.sin(np.pi*q[:,0])*np.cos(np.pi*q[:,1]))[:,None]
    return dict(axis=axis,widths_major_minor=width,color=color,metallic=metal,Q=tensor(axis,width,n))


def support(oracle_hit,reconstructed_hit,valid):
    require(oracle_hit.dtype==reconstructed_hit.dtype==valid.dtype==np.bool_ and
            oracle_hit.shape==reconstructed_hit.shape==valid.shape and
            not (valid&~(oracle_hit&reconstructed_hit)).any(),'Invalid saved correspondence domain')
    return dict(common=oracle_hit&reconstructed_hit,valid=valid,oracle_only=oracle_hit&~reconstructed_hit,
                reconstruction_only=~oracle_hit&reconstructed_hit,background=~oracle_hit&~reconstructed_hit,
                known=~reconstructed_hit|valid,undefined=reconstructed_hit&~valid)


def prepare(camera,p,bounds):
    with np.load(p['oracle']/'geometry/labels'/f'{camera}.npz',allow_pickle=False) as z:l={k:z[k] for k in z.files}
    with np.load(p['oracle']/'geometry/rays'/f'{camera}.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(p['oracle']/'references'/f'{camera}.npz',allow_pickle=False) as z:
        archived=z['linear_components'].reshape(-1,3,3);archived_hit=z['hit'].ravel()
    ids=l['pixel_indices'];n=l['normals'];position=l['position'];wo=-a['directions'][ids].astype(float)
    wo/=np.linalg.norm(wo,axis=1,keepdims=True)
    np.testing.assert_array_equal(ids,np.flatnonzero(a['hit']));np.testing.assert_array_equal(a['hit'],archived_hit)
    verify_labels(position,n,bounds,l)
    reconstructed=analytic(position,n,bounds)
    for name,key in [('Q','Q_world'),('widths_major_minor','widths_major_minor'),('color','color'),('metallic','metallic')]:
        require(np.max(abs(reconstructed[name]-l[key]))<=2e-12,'Primary field differs from admitted Bell labels')
    angle=np.where(l['orientation_defined'],l['theta_major'],0.)
    axis=np.cos(angle)[:,None]*l['T']+np.sin(angle)[:,None]*l['B']
    require(np.max(abs(tensor(axis,l['widths_major_minor'],n)-l['Q_world']))<=2e-12,'Oracle Q decode differs')
    require(((n*wo).sum(1)>0).all(),'Unexpected Bell oracle horizon case; no silent deletion')
    oracle=dict(pixels=ids,position=position,normals=n,outgoing=wo,source_position=position,
                axis=axis,widths_major_minor=l['widths_major_minor'],color=l['color'],metallic=l['metallic'],
                Q=l['Q_world'])
    result=dict(oracle=oracle);cohorts={};transfers={};undefined_views={}
    index=np.full(128**2,-1,dtype=int);index[ids]=np.arange(len(ids))
    for arm in ('O0','O1'):
        with np.load(p[arm]/'rasters'/f'{camera}.npz',allow_pickle=False) as z:r={k:z[k] for k in z.files}
        for key in ('origins','directions'):np.testing.assert_array_equal(a[key],r[key])
        np.testing.assert_array_equal(a['hit'],r['oracle_hit'])
        # Replay the accepted audit, never loosen correspondence eligibility.
        _,m=prior.original.paired_surface(
            {k:r['oracle_'+k] for k in ('hit','normal','position','depth')},
            {k:r['reconstructed_'+k] for k in ('hit','normal','position','depth')},a['directions'])
        require(all(np.array_equal(r[k],v) for k,v in m.items()),'Frozen correspondence replay differs')
        c=support(a['hit'],r['reconstructed_hit'],r['valid']);pix=np.flatnonzero(c['valid']);j=index[pix]
        undefined=np.flatnonzero(c['undefined'])
        undefined_views[arm]=dict(pixels=undefined,position=r['reconstructed_position'][undefined].astype(float),
                                  normals=r['reconstructed_normal'][undefined])
        n1=r['reconstructed_normal'][pix];n0=n[j];rot=minimal_rotation(n0,n1)
        require(np.max(abs(np.linalg.norm(n1,axis=1)-1))<=2e-6 and
                np.isfinite(r['reconstructed_position'][r['reconstructed_hit']]).all(),
                'Invalid captured reconstructed shading data')
        transported_axis=np.einsum('nij,nj->ni',rot,axis[j]);Q=rotate_tensor(l['Q_world'][j],rot)
        require(np.max(abs(tensor(transported_axis,l['widths_major_minor'][j],n1)-Q))<=2e-12,'Transported Q mismatch')
        position1=r['reconstructed_position'][pix].astype(float)
        spatial=analytic(position1,n0,bounds)
        physical=analytic(position1,n1,bounds)
        result[arm]=dict(pixels=pix,position=position1,normals=n1,outgoing=wo[j],source_position=position[j],
            axis=physical['axis'],widths_major_minor=physical['widths_major_minor'],color=physical['color'],metallic=physical['metallic'],Q=physical['Q'],
            transported_axis=transported_axis,transported_widths_major_minor=l['widths_major_minor'][j],
            transported_color=l['color'][j],transported_metallic=l['metallic'][j])
        # Evaluation-only boundary proximity from existing topology; no new rays.
        from scipy.spatial import cKDTree
        meshname='candidate.ply' if arm=='O1' else 'domain-aware-300000.ply'
        v,_=mesh_arrays(p[arm]/meshname)
        with np.load(p[arm]/'topology-after-cleanup.npz',allow_pickle=False) as z:edges=z['boundary_edges']
        near=np.zeros(128**2,bool);all_hits=np.flatnonzero(r['reconstructed_hit'])
        require(np.isfinite(r['reconstructed_normal'][all_hits]).all() and
                np.max(abs(np.linalg.norm(r['reconstructed_normal'][all_hits],axis=1)-1))<=2e-6,
                'Undefined-hit geometry visibility still requires valid unit normals')
        if len(edges):near[all_hits]=cKDTree(v[edges].astype(float).mean(1)).query(r['reconstructed_position'][all_hits])[0]<=.02
        theta=np.degrees(np.arccos(np.clip((n0*n1).sum(1),-1,1)))
        c['near_open_boundary']=near
        c['large_depth_shift']=m['common']&(abs(r['reconstructed_depth']-r['oracle_depth'])>.05)
        ndot=(r['oracle_normal']*r['reconstructed_normal']).sum(1)
        c['normal_gt15']=m['common']&(ndot<np.cos(np.radians(15)))
        c['grazing']=m['common']&(np.minimum(-(r['oracle_normal']*a['directions']).sum(1),
                                                         -(r['reconstructed_normal']*a['directions']).sum(1))<.1)
        c['orientation_eligible']=np.zeros(128**2,bool);c['orientation_eligible'][ids]=l['orientation_eligible']
        c['weak_or_isotropic']=np.zeros(128**2,bool);c['weak_or_isotropic'][ids]=~l['orientation_eligible']
        c['normal_failure']=m['common']&~m['signed_normal_ok'];c['distance_failure']=m['common']&~m['depth_close']
        c['facing_failure']=m['common']&~m['both_front'];c['raster']=np.ones(128**2,bool)
        cohorts[arm]=c
        transfers[arm]=dict(pixels=pix,source_position=position[j],target_position=position1,source_Q=l['Q_world'][j],
            transported_Q=Q,coordinate_Q=spatial['Q'],primary_Q=physical['Q'],
            coordinate_displacement=position1-position[j],normal_degrees=theta,
            Q_coordinate_shift=np.linalg.norm(spatial['Q']-l['Q_world'][j],axis=(1,2)),
            Q_frame_shift=np.linalg.norm(Q-l['Q_world'][j],axis=(1,2)),
            Q_projection_vs_transport=np.linalg.norm(physical['Q']-rotate_tensor(spatial['Q'],rot),axis=(1,2)),
            audit_vs_label_normal_max_abs=float(abs(r['oracle_normal'][pix]-n0).max()))
    # All three arms' matched comparison is an additional scope, never a replacement.
    joint=cohorts['O0']['valid']&cohorts['O1']['valid']
    for c in cohorts.values():c['three_arm_joint']=joint
    return dict(views=result,cohorts=cohorts,transfers=transfers,undefined_views=undefined_views,
                archived=archived,oracle_hit=a['hit'])


def choose(states,done):
    candidates=[]
    for order,arm in enumerate(ARMS):
        s=states[arm];d=s['components'][1]-s['components'][0]
        gates=[convergence(s['components'][1,:,k,c],s['components'][0,:,k,c]) for k in range(len(KINDS)) for c in range(3)]
        if all(g['passed'] for g in gates):continue
        absolute=abs(d).max(axis=(1,2,3))>.003;refined=np.isin(s['pixels'],done)
        require(not (absolute&refined).any(),'Refined component/contrast absolute convergence failed')
        rank=abs(d).max(axis=(1,2,3)) if absolute.any() else (d*d).sum(axis=(1,2,3))
        eligible=np.flatnonzero(~refined&(absolute if absolute.any() else True))
        require(len(eligible)>0,'Pooled convergence unresolved after allowed refinement')
        j=eligible[np.lexsort((s['pixels'][eligible],-rank[eligible]))[0]]
        candidates.append((int(not absolute.any()),-float(rank[j]),int(s['pixels'][j]),order))
    return min(candidates)[2] if candidates else None


def paired(views,fetch,total):
    states={};done=[]
    for arm,v in views.items():
        pair=[fetch(arm,v,n,'base') for n in BASE]
        states[arm]=dict(pixels=v['pixels'],components=np.stack([a['components'] for a in pair]),
                        visibility=np.stack([a['visibility'] for a in pair]),grids=np.tile(np.array(BASE)[:,None],(1,len(v['pixels']))))
    while True:
        pix=choose(states,done)
        if pix is None:return states,done
        require(len(done)<MAX_VIEW and total+len(done)<MAX_TOTAL,'Gate 12 refinement budget exhausted')
        for arm,v in views.items():
            rows=np.flatnonzero(v['pixels']==pix)
            if not len(rows):continue
            selected={k:a[rows] for k,a in v.items()}
            pair=[fetch(arm,selected,n,'pixel-'+str(pix)) for n in HIGH]
            states[arm]['components'][:,rows]=np.stack([a['components'] for a in pair])
            states[arm]['visibility'][:,rows]=np.stack([a['visibility'] for a in pair])
            states[arm]['grids'][:,rows]=np.array(HIGH)[:,None]
        done.append(pix)
