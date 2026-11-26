"""Hash-bound relighting of saved Donut materials jointly fitted with unknown light."""
from pathlib import Path
from types import FunctionType

import numpy as np

from benchmarks import donut_nero_gate7_v1 as gate7
from benchmarks.donut_recovery_v1 import numeric
from infrastructure.signatures import require

g6 = gate7.gate6.v1
ROOT, read, check, digest = gate7.ROOT, gate7.read, gate7.check, gate7.digest
PROTOCOL = 'donut-learned-light-relighting-gate9-v1'
FREEZE = 'docs/protocols/DONUT_LEARNED_LIGHT_RELIGHTING_GATE9_V1.freeze.json'
GATE7_SHA = 'bbc32d09a622ad8363c7c8a62737ac51ab10ce327990275b9de9c87340c74add'
MANIFEST = 'docs/results/donut_heterogeneous_training_v1/artifact-manifest.json'
INVENTORY = 'docs/results/publication_gate9/learned-light-endpoints.json'
ARMS = ('D2','D8-C','D8-T')
VARIANTS = ('A_corrected_core_v1','donut_tensor_chart_v1','donut_tensor_ambient_v1')
CAMERAS, SIZE = g6.CAMERAS, g6.SIZE
RESOURCES = g6.RESOURCES
LIMITS = dict(independent_shadow_directions=g6.MAX_DIRECTIONS,
              independent_BRDF_point_directions=len(ARMS)*g6.MAX_DIRECTIONS,
              optimizer_updates=0,checkpoint_loads=0,primary_rays=0,native_shader_calls=0,new_reference_integrations=0)
SOURCES = (
    'benchmarks/donut_learned_light_relighting_gate9_v1.py',
    'experiments/donut_learned_light_relighting_gate9_v1.py',
    'experiments/run_donut_learned_light_relighting_gate9_v1.sh',
    'tests/scientific/test_donut_learned_light_relighting_gate9_v1.py',
    'docs/protocols/DONUT_LEARNED_LIGHT_RELIGHTING_GATE9_V1.md',
    MANIFEST, INVENTORY, gate7.FREEZE,
    'experiments/donut_relighting_gate6_v1.py','benchmarks/donut_relighting_gate6_v1.py',
    'experiments/analyze_donut_relighting_gate6_v1.py',
    'benchmarks/bell_reference_e11.py','benchmarks/bell_e11.py',
    'experiments/bell_highlight_pilot_e11d.py','experiments/bell_quadrature_e11e.py',
    'benchmarks/generate_bell_e11.py','infrastructure/verified_ray_backend.py',
)


def definition(output):
    prior=gate7.freeze(GATE7_SHA)
    relight=gate7.gate6.freeze(gate7.GATE6_SHA)
    a=read(ROOT/MANIFEST)
    assets={k:relight['artifacts'][k] for k in ('oracle','generation','admission','verification')}
    assets.update({f'geometry_{c}':relight['artifacts'][f'view_{c}'] for c in CAMERAS})
    assets.update({f'reference_{c}':prior['panels'][str(c)] for c in CAMERAS})
    inventory=read(ROOT/INVENTORY)
    require(inventory['source_manifest_sha256']==digest(ROOT/MANIFEST),'Historical result manifest differs')
    methods={}
    for arm,variant in zip(ARMS,VARIANTS):
        rec=a['arms'][arm]['receipt']
        path=Path(rec['path'])
        keys={f'export_{arm}_{c}':g6.endpoint_name(c) for c in CAMERAS}
        light=f'illumination_{arm}'
        keys[light]='step-10000/illumination.npz'
        items=inventory['arms'][arm]
        require(set(items)==set(keys.values()),'Unexpected historical artifact inventory')
        assets.update({key:g6.with_path(items[name],path.parent/name) for key,name in keys.items()})
        assets[f'receipt_{arm}']=rec
        methods[arm]=dict(variant=variant,endpoint_fields=keys,seed=6033)
    return dict(protocol=PROTOCOL,host_output=str(output),image=prior['image'],backend=prior['backend'],
                assets=assets,methods=methods,roles=list(ARMS),cameras=list(CAMERAS),stage=10000,
                gate6_freeze_sha256=gate7.GATE6_SHA,gate6_common_support=relight['support'],
                light=relight['environment'],integration=dict(base=list(g6.BASE),high=list(g6.HIGH),
                per_view_refinements=g6.MAX_VIEW,total_refinements=g6.MAX_TOTAL,tolerances=g6.TOL,
                visibility='Gate 6 corrected FP64-predicate BVH and exact existing quadrature route'),
                limits=LIMITS,resources=RESOURCES,sources={name:digest(ROOT/name) for name in SOURCES})


def freeze(sha):
    path=ROOT/FREEZE
    require(digest(path)==sha,'Gate 9 freeze differs')
    f=read(path)
    output=Path(f['host_output'])
    require(output.resolve()==output and output.parent==Path('/home/isavand/paper/evidence')
            and output.name.startswith('donut-learned-relight-gate9-') and f==definition(output),'Gate 9 definition differs')
    return f


def authorization(f,commit,sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=sha,
                host_output=f['host_output'],image=f['image'],selected_gpu='0',maximum_execution_launches=1,
                methods=list(ARMS),limits=LIMITS,resources=RESOURCES,training_authorized=False,
                checkpoint_loading_authorized=False,reference_generation_authorized=False)


def paths(f,native=False):
    # Only archive identities and material/geometry arrays reach the renderer.
    # Evaluator radiance targets and learned-light maps stay on the host.
    return {key:Path(f'/inputs/{key}{Path(item["path"]).suffix}' if native else item['path'])
            for key,item in f['assets'].items()
            if not native or not key.startswith(('reference_','illumination_'))}


def read_input(f,p,key):
    return check(p[key],f['assets'][key])


def material_fields(maps,v,tensor):
    mask=maps['hit_mask']
    require(mask.shape==(SIZE,SIZE) and mask.dtype==np.bool_
            and np.array_equal(np.flatnonzero(mask),v['pixel_indices']),'Historical raster support differs')
    for k,k2 in (('position','position'),('normal','normals')):
        np.testing.assert_array_equal(maps[k][mask],v[k2].astype('f4'))
    q=maps['Q_tensor_direct' if tensor else 'Q_world'][mask]
    if tensor:
        require(np.max(abs(maps['Q_world'][mask]-q))<=g6.MATERIAL_ATOL,'Archived tensor export disagrees with its axes')
    widths,axis,error=g6.physical_axes(q,v['normals'])
    color,metallic=(maps[k][mask].astype(np.float64) for k in ('kd','metallic'))
    require(color.shape==(len(q),3) and metallic.shape==(len(q),1)
            and all(((x>=0)&(x<=1)).all() for x in (color,metallic)), 'Invalid endpoint scalar materials')
    return dict(Q_world=q.astype(np.float64),axis=axis,widths_major_minor=widths,color=color,metallic=metallic),error


def admit(f,p):
    gen,verify,admission=(read(read_input(f,p,k)) for k in ('generation','verification','admission'))
    require(gen['cleanup_verified'] and gen['scientific_work_complete'] and gen['backend']==f['backend']
            and gen['outputs']==verify['outputs']==admission['output_identities']
            and admission['researcher_admitted_dataset'] is True,'Expected accepted Donut dataset')
    receipts={}
    for arm in ARMS:
        row=read(read_input(f,p,f'receipt_{arm}'))
        require(row['status']=='complete' and row['scientific_work_complete'] and row['cleanup_verified']
                and row['optimizer_returns']==row['saved_endpoints'][-1]['step']==f['stage']
                and row['arm']==arm and row['backend']==f['backend']
                and row['benchmark_sha256']==f['assets']['generation']['sha256']
                and row['dataset_admission_sha256']==f['assets']['admission']['sha256']
                and row['effective_lighting']=='original A directional outer_light MLP'
                and row['indirect_illumination']=='identically zero'
                and row['initial']['variant']==f['methods'][arm]['variant']
                and row['initial']['seed']==6033,'Expected accepted learned-light fit')
        for key,name in f['methods'][arm]['endpoint_fields'].items():
            require(row['outputs'][name]=={k:f['assets'][key][k] for k in ('bytes','sha256')},
                    'Historical output identity differs')
        receipts[arm]=row
    chart,ambient=(receipts[key] for key in ('D8-C','D8-T'))
    require(chart['counters']==ambient['counters'] and chart['representation']==ambient['representation']
            and chart['paired_initial_gate']['passed'] and ambient['paired_initial_gate']['passed']
            and chart['initial']['common_base_state_sha256']==ambient['initial']['common_base_state_sha256'],
            'Historical tensor pairing differs')
    views,materials,parity={},{},{}
    for camera,count in zip(CAMERAS,g6.COUNTS):
        raw=numeric(read_input(f,p,f'geometry_{camera}'))
        valid=raw['shader_valid']
        v={k:raw[k][valid].astype(np.float64) for k in ('position','normals','outgoing')}
        v['pixel_indices']=raw['pixel_indices'][valid]
        spec=f['gate6_common_support'][str(camera)]
        require(len(v['pixel_indices'])==count and (np.diff(v['pixel_indices'])>0).all()
                and ((v['normals']*v['outgoing']).sum(1)>0).all()
                and g6.pixels_sha(v['pixel_indices'])==spec['foreground_sha256'],'Gate 6 foreground differs')
        views[camera]=v
        for arm in ARMS:
            maps=numeric(read_input(f,p,f'export_{arm}_{camera}'))
            fields,error=material_fields(maps,v,arm!='D2')
            np.testing.assert_array_equal(maps['linear_gt'][maps['hit_mask']],raw['linear_components'][valid,2].astype('f4'))
            materials.setdefault(camera,{})[arm]=fields
            parity.setdefault(str(camera),{})[arm]=dict(pixels=count,Q_axis_reconstruction_max_abs=error)
    # Verify the mesh bytes without decoding or invoking visibility here.
    read_input(f,p,'oracle')
    return views,materials,parity


def with_roles(function, **overrides):
    """Use the frozen Gate 6 numerics with a local role inventory."""
    return FunctionType(function.__code__,dict(function.__globals__,**overrides),function.__name__)


def refine(pixels,fetch,prior):
    verify=with_roles(g6.verify_grid,ROLES=ARMS)
    return with_roles(g6.refine,verify_grid=verify)(pixels,fetch,prior)


def expected_work(refined):
    require(0<=refined<=g6.MAX_TOTAL,'Invalid refinement count')
    directions=g6.DIRECTIONS+refined*sum(n*n for n in g6.HIGH)
    return dict(LIMITS,independent_shadow_directions=directions,
                independent_BRDF_point_directions=len(ARMS)*directions)


def illumination_score(pred,truth,weights):
    """Separate raw radiance error from an evaluation-only global brightness gauge."""
    require(pred.shape==truth.shape==(*weights.shape,3) and weights.size and np.isfinite(weights).all()
            and (weights>0).all() and all(np.isfinite(a).all() and (a>=0).all() for a in (pred,truth)),
            'Invalid linear radiance samples')
    w=weights[...,None]
    n=np.sum(w*truth**2)
    d=np.sum(w*pred**2)
    require(n>0 and d>0,'Zero signal in illumination score')
    gain=float(np.sum(w*pred*truth)/d)

    def error(rgb):
        return float(np.sqrt(np.sum(w*(rgb-truth)**2)/n))

    def mean(rgb):
        return (np.sum((w*rgb).reshape(-1,3),axis=0)/weights.sum()).tolist()

    return dict(raw_relative_L2=error(pred),gain_aligned_relative_L2=error(gain*pred),
                diagnostic_global_gain=gain,mean_RGB=mean(pred),reference_mean_RGB=mean(truth))
