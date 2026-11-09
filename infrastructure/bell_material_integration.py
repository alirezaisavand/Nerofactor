"""Diagnostic-only historical G bell mesh admission into pinned A material.

Ordinary stage execution and attributable research handoff are not consulted or
relaxed. Planning reads only YAML/JSON/opaque mesh bytes, never camera pickles.
"""
from copy import deepcopy
import datetime
import hashlib
import os
from pathlib import Path
import sys
import traceback

from infrastructure.paths import PathResolver,InputSelection,InputRef
from infrastructure.runs import RunStore,validate_run_id
from infrastructure.signatures import require,read_report,write_report
from infrastructure.stage_sources import ROOT,PINS,baseline,activate,imported_sources,digest,infrastructure_identity
from infrastructure.material_smoke import MaterialProbe,GROUPS,inspect_mesh,run_bounded
from infrastructure.stage_workspace import prepare_view,source_records
from infrastructure.bell_material_rays import BOUNDS
from infrastructure.bell_inference import proposed_split

RAY_PROTOCOL='historical-G-bell-to-A-rays-v1'
TRAIN_PROTOCOL='historical-G-bell-to-A-two-update-v1'
MESH_SHA256='eaa17f8a6b4ff12c769443195a00f78f2fd08b7adfd3e51ba463fea5a55cefd5'
EXTRACTION_SHA256='caa850d761f968cbb28f2a17890fc6864ad4e2c73cd0185135c1aca679f2461f'
EXTRACTION_RUN='bell-g-best-mesh-2f4a25e1828f47109433f42a56d53f4a'
MESH_NAME='meshes/bell_shape-best-150000-r128.ply'
MESH_COUNTS=(30806,61612)
CONFIG='configs/material/syn/bell.yaml'


def expected_splits():
    v=proposed_split()  # Existing local Random(6033) verifier, no global RNG changes.
    return {'train':v['train_ids'],'validation':v['test_ids'],'nvs':v['nvs_ids']}


def identity():
    return {**infrastructure_identity(),**{n:digest(ROOT/n) for n in
        ['run_bell_material_diagnostic.py','docs/BELL_MATERIAL_INTEGRATION_DIAGNOSTIC.md']}}


def plan(commit,datasets,thesis,geometry,output,mesh,mesh_sha256,extraction,extraction_sha256,
         action,ray_report=None,ray_sha256=None):
    science,source=baseline('material',commit)
    _,g_source=baseline('geometry',PINS['geometry'])
    require(action in ('rays','two-update'),'Unknown bell diagnostic action')
    require(mesh_sha256==MESH_SHA256 and extraction_sha256==EXTRACTION_SHA256,'Explicit admitted mesh/report hashes required')
    roots={'datasets':datasets,'thesis':thesis,'geometry':geometry}
    if ray_report is not None:
        q=Path(ray_report);require(q.is_absolute() and q.resolve()==q,'Explicit unaliased ray evidence required')
        roots['ray_evidence']=q.parent
    resolver=PathResolver(roots,output)
    require(resolver.output_root!=ROOT and ROOT not in resolver.output_root.parents and resolver.output_root not in ROOT.parents,'Output overlaps repository')
    selected=resolver.resolve_inputs(InputSelection('GlossySynthetic',InputRef('geometry',MESH_NAME)))
    require(Path(mesh).is_absolute() and Path(mesh).resolve()==Path(mesh)==selected.mesh,'Explicit admitted mesh required; no substitution')
    proof=resolver.input(InputRef('geometry','G-mesh-diagnostic.json'))
    require(Path(extraction).is_absolute() and Path(extraction)==Path(extraction).resolve()==proof,'Explicit extraction report required')
    evidence,proof_identity=read_report(proof,EXTRACTION_SHA256)
    from infrastructure.g_checkpoint import CHECKPOINT_SHA256
    require(evidence.get('protocol')=='bell-historical-best-G-mesh-r128-v1' and
            evidence.get('run_id')==EXTRACTION_RUN and evidence.get('status')==evidence.get('phase')=='complete' and
            evidence.get('scientific_commit')==PINS['geometry'] and evidence.get('source',{}).get('files')==g_source['files'] and
            evidence.get('strict_network_loaded') is True and evidence.get('network_unchanged') is True and
            evidence.get('checkpoint_selection',{}).get('sha256')==CHECKPOINT_SHA256 and
            evidence.get('mesh',{}).get('sha256')==MESH_SHA256 and evidence['mesh'].get('path')==MESH_NAME and
            evidence.get('settings',{}).get('resolution')==128 and evidence['mesh'].get('explicit_repairs_or_transforms')==[] and
            evidence.get('research_handoff_admitted') is False,'Extraction evidence is not the admitted historical diagnostic')
    mesh_identity=inspect_mesh(selected.mesh,MESH_SHA256)
    require(mesh_identity['vertices']==MESH_COUNTS[0] and mesh_identity['triangles']==mesh_identity['nondegenerate_triangles']==MESH_COUNTS[1],
            'Admitted diagnostic geometry differs')
    require((selected.dataset/'bell/masks_train').is_dir(),'Original A bell masks_train inputs required')
    raw=(science/CONFIG).read_bytes()
    import yaml
    cfg=yaml.load(raw,Loader=yaml.FullLoader)
    require(cfg['database_name']=='syn/bell' and cfg['shader_cfg']['human_lights'] is False,'Original A bell configuration required')
    p={'stage':'material','scientific_commit':commit,'science':science,'source':source,'resolver':resolver,'selected':selected,
       'scene':'bell','configuration':cfg,'raw':raw,'mesh_identity':mesh_identity,'proof_identity':proof_identity,'action':action}
    if action=='two-update':
        require(ray_report is not None and ray_sha256 is not None,'Two-update diagnostic requires explicit completed ray report and SHA-256')
        ray,ray_identity=read_report(ray_report,ray_sha256)
        require(ray.get('schema_version')==1 and ray.get('protocol')==RAY_PROTOCOL and ray.get('status')=='complete' and
                ray.get('ray_gate_passed') is True and ray.get('scientific_commit')==commit and
                ray.get('mesh_identity',{}).get('sha256')==MESH_SHA256 and
                ray.get('extraction_report',{}).get('sha256')==EXTRACTION_SHA256 and
                ray.get('legacy_sha256')==hashlib.sha256(raw).hexdigest() and ray.get('infrastructure_files')==identity() and
                ray.get('bounds')==BOUNDS and ray.get('optimizer_calls_completed')==0,'Ray gate identities/implementation/bounds differ')
        views=ray.get('views',[])
        require(len(views)==3 and [v.get('role') for v in views]==['train','validation','nvs'] and
                ray.get('splits')==expected_splits() and
                all(v.get('source_camera_id')==ray['splits'][v['role']][0] and v.get('resolution')==[800,800] for v in views) and
                all(v.get('statistics',{}).get('passed') is True and v['statistics'].get('checks') and
                    set(v['statistics']['checks'])=={'original_negation','unit_final_normals','intersection_consistency',
                        'front_facing_final_normals','raw_normals_opposite_view','silhouette_alignment'} and
                    all(x is True for x in v['statistics']['checks'].values()) for v in views),'Incomplete/failed ray gates')
        for v in views:
            artifact=resolver.input(InputRef('ray_evidence',v['artifact']))
            require(digest(artifact)==v['sha256'],'Ray evidence artifact changed')
        require(ray.get('input_files')==source_records(selected.dataset/'bell'),'Ray/training source dataset differs')
        p.update(ray=ray,ray_identity=ray_identity)
    else:require(ray_report is None and ray_sha256 is None,'Ray-only action cannot consume a training gate')
    return p


def verify_loaded_mesh(mesh,p,np):
    """Open3D must preserve the exact PLY coordinate/triangle order and winding."""
    with p['selected'].mesh.open('rb') as stream:
        for _ in range(32):
            line=stream.readline(256)
            require(line and len(line)<256,'Invalid selected PLY header during native verification')
            if line==b'end_header\n':break
        else:require(False,'Oversized selected PLY header during native verification')
        xyz=np.frombuffer(stream.read(12*MESH_COUNTS[0]),dtype='<f4').reshape(-1,3)
        faces=np.frombuffer(stream.read(),dtype=np.dtype([('count','u1'),('indices','<i4',(3,))]))['indices']
    require(np.array_equal(np.asarray(mesh.vertices),xyz) and np.array_equal(np.asarray(mesh.triangles),faces),
            'Open3D changed selected mesh vertices/triangle ordering/winding')


class BellProbe(MaterialProbe):
    database_name='syn/bell'
    is_real=False
    human_lights=False
    groups=tuple(g for g in GROUPS if g!='human_light')
    def initialize(self):
        super().initialize()
        require(self.report['splits']=={'train_ids':self.report['ray_splits']['train'],
                'test_ids':self.report['ray_splits']['validation'],'nvs_ids':self.report['ray_splits']['nvs']},'Ray/training split differs')
        require(not hasattr(self.trainer.network.shader_network,'human_light'),'Original bell disables human light network')
        self.report['expected_absent_groups']={'human_light':'Absent by original A bell shader_cfg.human_lights=false'}
        self.persist()


def native(p,run,report,persist):
    report['phase']='native_runtime_imports';persist()
    import torch
    import numpy as np
    require(sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and
            torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe',
            'Reviewed Python 3.8/Torch/CUDA/one A100 required')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu','Original CPU float32 setup required')
    from infrastructure.geometry_training_smoke import runtime_identity
    import _raytracing
    report['runtime']=runtime_identity(torch)
    report['runtime']['raytracing_binary']={'path':_raytracing.__file__,'sha256':digest(_raytracing.__file__)}
    persist()
    try:
        if p['action']=='rays':
            from infrastructure.bell_material_rays import native_views
            report['phase']='original_A_mesh_BVH_and_camera_initialization';persist()
            native_views(p,run,report,persist,torch,np)
        else:
            require(report['runtime']==p['ray']['runtime'],'Ray/training runtime identities differ')
            from train.trainer import Trainer
            from infrastructure.stage_execution import trainer_io
            require(Path(Trainer.run.__code__.co_filename).resolve()==p['science']/'train/trainer.py','Foreign A trainer')
            report['phase']='original_A_trainer_constructor';persist()
            trainer=trainer_io(Trainer,run)(report['runtime_configuration'])
            report['ray_splits']=p['ray']['splits']
            run_bounded(trainer,p['science'],torch,np,report,persist,probe_class=BellProbe)
            verify_loaded_mesh(trainer.network.mesh,p,np)
    finally:
        torch.set_default_tensor_type('torch.FloatTensor');torch.cuda.empty_cache()
        report['runtime_cleanup']={'CPU_float_default_restored':True,'profile_restored':sys.getprofile() is None,'trace_restored':sys.gettrace() is None}


def execute(p,run_id,image_id,infrastructure_commit,acknowledged=False):
    validate_run_id(run_id)
    require(acknowledged is True,'Explicit diagnostic-only historical mesh/camera trust acknowledgement required')
    from infrastructure.stage_execution import native_guards
    native_guards(p,image_id)
    if p['action']=='two-update':require(image_id==p['ray']['image_id_declared'],'Ray/training image differs')
    activate('material',p['scientific_commit'])
    run=RunStore(p['resolver'],layout='mounted').create(run_id)
    files=identity()
    report={'schema_version':1,'protocol':RAY_PROTOCOL if p['action']=='rays' else TRAIN_PROTOCOL,
            'status':'running','phase':'allocated','run_id':run_id,'scientific_commit':p['scientific_commit'],
            'source':p['source'],'infrastructure_commit_declared':infrastructure_commit,'infrastructure_files':files,
            'image_id_declared':image_id,'mesh_identity':p['mesh_identity'],'extraction_report':p['proof_identity'],
            'original_configuration':p['configuration'],'legacy_sha256':hashlib.sha256(p['raw']).hexdigest(),
            'diagnostic_only':True,'research_handoff_admitted':False,'historical_producing_revision':'unresolved',
            'canonical_resolution_512':False,'bounds':deepcopy(BOUNDS),'optimizer_calls_completed':0,
            'mesh_policy':{'transforms_or_repairs':[],'winding':'preserve serialized PLY triangle order',
                           'normal_operation':'original A trace negation and F.normalize only'},
            'scope':'Three current-A camera views then separately gated fresh two-update prefix; no thesis split or research handoff attribution',
            'ray_gate_reference':p.get('ray_identity'),'checkpoint_input':None,'optimizer_state_restored':False,
            'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with run.output('bell-material-diagnostic.json').open('x') as stream:
        def persist():write_report(stream,report)
        persist()
        try:
            report['phase']='owned_dataset_view_setup';persist()
            workspace,family,inputs=prepare_view(run,p['selected'].dataset,'bell',p['science'],p['configuration'],p['source'])
            report['input_files']=inputs
            cfg=deepcopy(p['configuration']);cfg['dataset_dir']=str(family);cfg['mesh']=str(p['selected'].mesh)
            report['runtime_configuration']=cfg
            run.output('config.original.yaml').write_bytes(p['raw'])
            from infrastructure.bell_inference import runtime_environment
            with runtime_environment(run):
                os.chdir(workspace)
                try:native(p,run,report,persist)
                except BaseException:traceback.print_exc();raise
            require(source_records(p['selected'].dataset/'bell')==inputs and digest(p['selected'].mesh)==MESH_SHA256 and
                    digest(p['proof_identity']['path'])==EXTRACTION_SHA256,'Original inputs/evidence changed')
            require(files==identity(),'Diagnostic implementation changed')
            baseline('material',p['scientific_commit'])
            require(not list(run.checkpoints.iterdir()) and not any(q.is_file() for q in run.output('renders').rglob('*')),
                    'Unexpected checkpoint/validation/NVS output')
            if p['action']=='rays':require(report.get('ray_gate_passed') is True and report['optimizer_calls_completed']==0,'Ray diagnostic did not pass')
            else:
                require(digest(p['ray_identity']['path'])==p['ray_identity']['sha256'],'Ray gate report changed')
                require(report['optimizer_calls_completed']==2 and len(report.get('updates',[]))==2 and not report.get('cleanup_errors'),
                        'Invalid two-update bounded completion')
            report['imported_scientific_modules']=imported_sources(p['science'])
            report['output_hashes']={q.relative_to(run.root).as_posix():digest(q) for q in run.root.rglob('*')
                if q.is_file() and q.resolve()==q and q.name not in ['bell-material-diagnostic.json','.experiment.json'] and
                not any(part in ('torch_extensions','home','tmp') for part in q.relative_to(run.root).parts)}
            report.update(status='complete',phase='ray_gates_passed' if p['action']=='rays' else 'bounded_stop_after_second_optimizer_return')
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist();raise
        report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        persist()
    return report
