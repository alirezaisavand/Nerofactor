"""Cold metadata and geometry-only policies for the four-checkpoint G audit."""
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion

from benchmarks import bell_geometry_gate102 as policy
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT,PINS,baseline,digest

PROTOCOL='bell-geometry-evolution-gate10.3-v1'
FREEZE='docs/protocols/BELL_GEOMETRY_EVOLUTION_GATE103_V1.freeze.json'
INVENTORY='docs/results/publication_gate103/input-inventory.json'
STEPS=(15000,30000,50000,300000)
TRAIN=policy.previous.PRIOR/'train'
ACCEPTED=Path(policy.read(ROOT/policy.FREEZE)['host_output'])
SOURCES=tuple(dict.fromkeys((*policy.SOURCES,policy.FREEZE,
    'docs/results/publication_gate102/results-v2/summary.json',
    'docs/results/publication_gate102/results-v2/artifact-manifest.json',
    'benchmarks/bell_geometry_gate103.py','experiments/bell_geometry_gate103.py',
    'experiments/run_bell_geometry_gate103.sh','tests/scientific/test_bell_geometry_gate103.py',
    'docs/protocols/BELL_GEOMETRY_EVOLUTION_GATE103_V1.md',
    'docs/results/publication_gate103/PREPARATION.md',INVENTORY,
    'audits/bell_mask_provenance_gate103.py','docs/RESEARCH_BACKLOG.md',
    'docs/results/publication_gate103/MASK_SUPERVISION_CLARIFICATION_V1.md',
    'docs/results/publication_gate103/mask-provenance-v1.json',
    'docs/results/publication_gate103/sam2-artifact-search-v1.json',
    'docs/results/publication_gate103/clarification-validation-v1.json')))
LIMITS=dict(checkpoint_loads=4,dense_grids=3,dense_tiles=1536,dense_SDF_points=3*512**3,MC_calls=3,
    max_faces_per_mesh=4000000,audit_BVH_rays=3*196608,fitting_BVH_rays=4*122*128**2,
    CPU_ray_checks=3*384+4*64,consistency_SDF_points=3*1536,consistency_gradient_points=3*512,
    neural_render_calls=128,neural_rays=4*16*256*2,neural_final_sample_slots=4*16*256*2*160,
    neural_SDF_forward_points=35127296,neural_gradient_points=10485760,
    occ_probe_origins=3*16*2*2048,validation_probe_origins=0,
    appearance_forward_calls=128,appearance_point_slots=4*16*256*2*160,
    optimizer_returns=0,checkpoint_writes=0,material_calls=0)
RESOURCES=dict(watchdog=3600,inner=3540,grace=30,cpu_threads=4,memory_gib=128,pids=128,
    allocator=.75,output_bytes=8*1024**3,MAX_JOBS=4)
NEURAL=dict(cameras=16,per_stratum=64,boundary_width=4,draws=2,seed='6033+1000*camera_ID+draw',thresholds=[.1,.5,.9],
    primary_threshold=.5,perturb_overwrite=-1,cos_anneal='original get_anneal_val(checkpoint-step-minus-one)',
    step='checkpoint-step-minus-one',is_train=True,is_nerf=False,forward_only=True,
    ray_builder='original CPU construction then original CUDA ray processing')
read,write,item,check=policy.read,policy.write,policy.item,policy.check


def angular_panel(ids,poses,count=16):
    """First training ID, then max-min world camera-direction angle; numeric-ID ties."""
    require(len(ids)>=count and len(set(ids))==len(ids),'Unique training camera population required')
    centers=np.array([-poses[int(k),:3,:3].T@poses[int(k),:3,3] for k in ids])
    length=np.linalg.norm(centers,axis=1)
    require(np.isfinite(centers).all() and (length>0).all(),'Invalid camera directions')
    directions=centers/length[:,None];chosen=[0]
    while len(chosen)<count:
        score=np.min(1-np.clip(directions@directions[chosen].T,-1,1),axis=1)
        score[chosen]=-np.inf
        best=np.flatnonzero(score==score.max())
        chosen.append(min(best,key=lambda i:int(ids[i])))
    return [ids[i] for i in chosen]


def pixel_panel(hit,count=64,width=4):
    require(hit.dtype==bool and hit.shape==(128**2,),'Frozen flattened boolean foreground required')
    mask=hit.reshape(128,128)
    # Chebyshev bands, using the original image boundary as exterior background.
    inner=mask&~binary_erosion(mask,structure=np.ones((3,3)),iterations=width,border_value=0)
    outer=~mask&binary_dilation(mask,structure=np.ones((3,3)),iterations=width,border_value=0)
    groups={}
    for name,m in dict(foreground=mask,background=~mask,inner_band=inner,outer_band=outer).items():
        ids=np.flatnonzero(m.ravel(order='C'))
        groups[name]=ids[np.linspace(0,len(ids)-1,min(count,len(ids)),dtype=np.int64)].tolist() if len(ids) else []
    union=sorted({i for ids in groups.values() for i in ids})
    return dict(indices=union,strata=groups,stratum_population={k:int(v.sum()) for k,v in
        dict(foreground=mask,background=~mask,inner_band=inner,outer_band=outer).items()})


def silhouette(oracle,candidate):
    require(oracle.shape==candidate.shape and oracle.dtype==candidate.dtype==bool and oracle.any(),
            'Matching nonempty reference masks required')
    tp=int((oracle&candidate).sum());fp=int((~oracle&candidate).sum());fn=int((oracle&~candidate).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=int((~oracle&~candidate).sum()),iou=tp/(tp+fp+fn),
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn))


def mask_terms(mask,acc,lambda_bg=1.):
    """Synthetic/prospective only; never applied to a live G model in this study."""
    require(mask.shape==acc.shape and np.isfinite(acc).all() and np.isin(mask,[0,1]).all() and lambda_bg>=0,
            'Binary mask/finite accumulation required')
    original=np.abs(mask-mask*acc)
    added=lambda_bg*(1-mask)*acc
    return dict(original=float(original.mean()),background=float(added.mean()),
                candidate=float((original+added).mean()))


def opacity_summary(panel,oracle,mesh,draws):
    indices=np.array(panel['indices'])
    require(draws.shape==(2,len(indices)) and oracle.shape==mesh.shape==(len(indices),) and
            np.isfinite(draws).all(),'Paired opacity inventory differs')
    result={}
    for name,ids in dict(unique_panel=panel['indices'],**panel['strata']).items():
        selected=np.isin(indices,ids);a=draws[:,selected];truth=oracle[selected];hit=mesh[selected]
        if not selected.any():result[name]=dict(count=0);continue
        decisions=a>.5
        result[name]=dict(count=int(selected.sum()),mean_acc_by_draw=a.mean(1).tolist(),
            mean_absolute_draw_difference=float(abs(a[0]-a[1]).mean()),
            threshold_flips=int((decisions[0]!=decisions[1]).sum()),
            stable_false_positive=int((decisions.all(0)&~truth).sum()),
            stable_false_negative=int(((~decisions).all(0)&truth).sum()),
            mesh_false_positive=int((hit&~truth).sum()),mesh_false_negative=int((~hit&truth).sum()),
            threshold_diagnostics={str(t):dict(false_positive=((a>t)&~truth).sum(1).tolist(),
                false_negative=((a<=t)&truth).sum(1).tolist(),mesh_disagreements=((a>t)!=hit).sum(1).tolist())
                for t in (.1,.5,.9)})
    return result


def negative_extension(raw,outside):
    require(raw.shape==outside.shape==(512,512,512),'Original grid layout required')
    result={}
    for name,mask in (('inside',~outside),('outside',outside)):
        values=raw[mask]
        result[name]=dict(nodes=len(values),negative=int((values<-1e-7).sum()),
                          near_zero=int((abs(values)<=1e-7).sum()),minimum=float(values.min()),maximum=float(values.max()))
    lower=raw[:,:90,:]  # Original geometry-only y < -.65 band, no node lies on -.65.
    result['lower_cube_y_below_minus065']=dict(nodes=lower.size,negative=int((lower<-1e-7).sum()))
    edges={}
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        edges[str(axis)]=int(((outside[a]!=outside[b])&(raw[a]<-1e-7)&(raw[b]<-1e-7)).sum())
    result['raw_negative_domain_crossing_edges']=edges
    return result


def definition(out):
    inputs=read(ROOT/INVENTORY)
    old=read(ROOT/policy.FREEZE)
    return dict(protocol=PROTOCOL,host_output=str(out),steps=list(STEPS),publication_endpoint=300000,
        image=old['image'],backend=old['backend'],scientific_commit=PINS['geometry'],
        checkpoints=inputs['checkpoints'],training_receipt=inputs['training_receipt'],
        training_root=str(TRAIN),acquisition=old['acquisition'],eval_files=inputs['geometry_files'],
        fitting_ids=inputs['fitting_ids'],audit_ids=list(policy.original.EVAL_IDS),
        neural_panel=inputs['neural_panel'],neural=NEURAL,domain=old['domain'],outside_mask=old['outside_mask'],
        raw300=old['raw_grid'],raw300_path=str(Path(old['diagnostic'])/'run/stages/raw-SDF-grid.npy'),
        reuse=dict(root=str(ACCEPTED),files=inputs['accepted_files']),jit_cache=inputs['jit_cache'],
        jit_cache_root=str(TRAIN/'torch_extensions'),quality=old['quality'],cell_policy=old['cell_policy'],
        numerical_policy=old['numerical_policy'],limits=LIMITS,resources=RESOURCES,
        sources={name:digest(ROOT/name) for name in SOURCES})


def freeze(sha):
    require(digest(ROOT/FREEZE)==sha,'Gate 10.3 freeze differs')
    f=read(ROOT/FREEZE);out=Path(f['host_output'])
    require(out.resolve()==out and out.parent==TRAIN.parent.parent and out.name.startswith('bell-geometry-gate103-') and
            f==definition(out),'Frozen evaluation definition differs')
    return f


def host_inputs(f):
    t=read(check(TRAIN/'training.json',f['training_receipt']))
    require(t['status']=='complete' and t['adam_returns']==300000 and t['cleanup_verified'],'Authenticated training incomplete')
    for step,ref in f['checkpoints'].items():
        expected=t['endpoint_checkpoint'] if step=='300000' else t['retained_checkpoints'][step]
        require(ref['identity']==expected,'Checkpoint producing receipt differs')
        check(TRAIN/'checkpoints'/ref['file'],ref['identity'])
    for name,identity in f['eval_files'].items():check(Path(f['acquisition'])/'geometry'/name,identity)
    for name,identity in f['reuse']['files'].items():check(ACCEPTED/name,identity)
    check(Path(f['raw300_path']),f['raw300'])
    old=read(ROOT/policy.FREEZE)
    check(Path(old['diagnostic'])/'run/stages/outside-sphere.npy',f['outside_mask'])
    for name,identity in f['jit_cache'].items():check(TRAIN/'torch_extensions'/name,identity)
    baseline('geometry',PINS['geometry'])


def authorization(f,commit,sha):
    return dict(approved=True,protocol=PROTOCOL,commit=commit,freeze_sha256=sha,host_output=f['host_output'],
        image=f['image'],backend=f['backend'],selected_gpu='0',checkpoints=f['checkpoints'],
        CPU_pickle='exactly these four authenticated researcher checkpoints, once each, CPU only',
        limits=LIMITS,resources=RESOURCES,maximum_execution_launches=1,training_authorized=False,
        existing_jit_cache=f['jit_cache'],original_runtime_JIT_validation_or_identical_rebuild=True)
