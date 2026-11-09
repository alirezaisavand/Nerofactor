"""Diagnostic-only historical bell best-checkpoint extraction using pinned G code.

Native imports and trusted pickle loading occur only after explicit isolated
execution gates. No dataset, trainer, optimizer, renderer forward or material run.
"""
from contextlib import contextmanager
from copy import deepcopy
import datetime
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import sys
import traceback

from infrastructure.paths import PathResolver,InputRef
from infrastructure.runs import RunStore,validate_run_id
from infrastructure.signatures import CHECKPOINT_SHA256,require,read_report,write_report
from infrastructure.stage_sources import ROOT,PINS,baseline,activate,digest,imported_sources,infrastructure_identity
from infrastructure.g_checkpoint import STORED_STEP,historical_schema,load_checked,network_comparison,strict_network_load,identities
from infrastructure.g_lifecycle import runtime_guards,IMAGE
from infrastructure.bell_inference import runtime_environment

PROTOCOL='bell-historical-best-G-mesh-r128-v1'
COMPATIBILITY_SHA256='73e405b99a3041bb6bc7195e220bd128b82ea3ee54d5a4f5f4df14476b20435a'
CHECKPOINT='model/bell_shape/model_best.pth'
DOC='docs/G_MESH_EXTRACTION_DIAGNOSTIC.md'
SETTINGS={'resolution':128,'canonical_original_CLI_resolution':512,'diagnostic_resolution':True,
          'bbox_min':[-1,-1,-1],'bbox_max':[1,1,1],'isovalue':0,'outside_val':1.0,
          'outside_mask':'norm(points) >= 1','field_dtype':'float32','chunk_axis_size':64,
          'sdf_query':'lambda x: network.sdf_network.sdf(x)','explicit_mesh_transforms':[]}


def identity():
    return {**infrastructure_identity(),**{n:digest(ROOT/n) for n in ['run_g_mesh_diagnostic.py',DOC]}}


def prepare(commit,datasets,thesis,outputs,checkpoint,checkpoint_sha256,compatibility,compatibility_sha256,resolution=128):
    require(type(resolution) is int and resolution==128,'Only reviewed diagnostic resolution 128 is supported')
    science,source=baseline('geometry',commit)
    require(checkpoint_sha256==CHECKPOINT_SHA256 and compatibility_sha256==COMPATIBILITY_SHA256,
            'Explicit admitted checkpoint and completed G compatibility hashes required')
    proof=Path(compatibility)
    require(proof.is_absolute() and proof.resolve()==proof,'Compatibility evidence must be explicit/unaliased')
    resolver=PathResolver({'datasets':datasets,'thesis':thesis,'compatibility':proof.parent},outputs)
    require(ROOT!=resolver.output_root and ROOT not in resolver.output_root.parents and resolver.output_root not in ROOT.parents,
            'Extraction outputs overlap source repository')
    selected=resolver.input(InputRef('thesis',CHECKPOINT))
    explicit=Path(checkpoint)
    require(explicit.is_absolute() and explicit.resolve()==explicit and explicit==selected and digest(selected)==CHECKPOINT_SHA256,
            'Select the exact admitted historical model_best.pth explicitly; no fallback/substitution')
    prior,proof_id=read_report(proof,COMPATIBILITY_SHA256)
    import yaml
    raw=(science/'configs/shape/syn/bell.yaml').read_bytes();cfg=yaml.load(raw,Loader=yaml.FullLoader)
    comparison=prior.get('network_comparison',{})
    require(prior.get('protocol')=='bell-G-checkpoint-compatibility-v1' and prior.get('action')=='compatibility' and
            prior.get('status')==prior.get('phase')=='complete' and prior.get('scientific_commit')==commit and
            prior.get('checkpoint_sha256')==CHECKPOINT_SHA256 and prior.get('strict_network_loaded') is True and
            prior.get('checkpoint_metadata',{}).get('stored_step')==STORED_STEP and
            prior.get('source',{}).get('scientific_files')==source['files'] and
            prior.get('legacy_sha256')==hashlib.sha256(raw).hexdigest() and
            comparison.get('result')=='exact_structural_match' and
            all(comparison.get(k)==[] for k in ['missing_keys','unexpected_keys','shape_mismatches','dtype_mismatches']) and
            comparison.get('checkpoint_network_catalog')==comparison.get('G_network_catalog') and
            bool(prior.get('loaded_network_identity')),'Completed G checkpoint compatibility evidence does not match source/configuration')
    return {'science':science,'source':source,'commit':commit,'resolver':resolver,'checkpoint':selected,
            'cfg':cfg,'raw':raw,'prior':prior,'proof_identity':proof_id,'resolution':resolution}


class ExtractionAudit:
    """Observe the original functions without wrapping/replacing SDF query calls."""
    def __init__(self,network,extract_geometry,extract_fields,torch,np):
        self.query=network.sdf_network.sdf.__func__.__code__
        self.geometry=extract_geometry.__code__;self.fields=extract_fields.__code__;self.torch=torch;self.np=np
        self.forbidden={inspect.unwrap(f).__code__ for f in
                        [network.forward,network._init_dataset,torch.Tensor.backward,torch.optim.Adam.step,torch.optim.Optimizer.load_state_dict]}
        self.record={'extract_geometry_calls':0,'extract_fields_calls':0,'sdf_calls':0,'sdf_points':0,'maximum_query_points':0}
    def observe(self,frame,event,value):
        code=frame.f_code
        require(not(event=='call' and code in self.forbidden),'Training/rendering/dataset/optimizer call during extraction')
        if event=='call' and code in (self.geometry,self.fields):
            key='extract_geometry_calls' if code is self.geometry else 'extract_fields_calls';self.record[key]+=1
            require(self.record[key]==1 and frame.f_locals['resolution']==128,'Unexpected/repeated extraction request')
        if event=='return' and value is not None and code is self.query:
            require(not self.torch.is_grad_enabled(),'Original extraction no_grad context missing')
            pts=frame.f_locals['x'];r=self.record
            require(list(pts.shape)==[64**3,3] and list(value.shape)==[64**3,1] and value.dtype==self.torch.float32,
                    'Original 128/64 sampling or SDF shape/dtype changed')
            arr=value.detach().cpu().numpy()
            require(self.np.isfinite(arr).all(),'Nonfinite SDF query; no masking/repair to hide it')
            r['sdf_calls']+=1;r['sdf_points']+=len(pts);r['maximum_query_points']=max(r['maximum_query_points'],len(pts))
            require(r['sdf_calls']<=8,'SDF query budget exceeded')
        if event=='return' and value is not None and code is self.fields:
            require(value.shape==(128,128,128) and value.dtype==self.np.float32 and self.np.isfinite(value).all(),
                    'Invalid original dense SDF field')
            self.record.update(field_minimum=float(value.min()),field_maximum=float(value.max()))
    @contextmanager
    def active(self):
        require(sys.implementation.name=='cpython' and sys.gettrace() is None and sys.getprofile() is None,
                'Untraced/unprofiled CPython extraction process required')
        sys.setprofile(self.observe)
        try:yield
        finally:sys.setprofile(None)
    def complete(self):
        require(all(self.record[k]==v for k,v in {'extract_geometry_calls':1,'extract_fields_calls':1,'sdf_calls':8,
                'sdf_points':128**3,'maximum_query_points':64**3}.items()),'Incomplete original G extraction call budget')
        return self.record


def native(p,run,report,mark):
    import numpy as np
    import torch
    import trimesh
    import mcubes
    require(sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and
            torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe',
            'Reviewed Python 3.8/Torch 1.8.1 CUDA 11.1/one A100 runtime required')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu','Original CPU float32 constructor context required')
    from network.renderer import NeROShapeRenderer
    from network.field import extract_geometry,extract_fields
    from infrastructure.geometry_training_smoke import runtime_identity
    from infrastructure.mesh_diagnostics import export_diagnostic
    report['runtime']=runtime_identity(torch)
    report['runtime']['mesh_packages']={n:importlib.metadata.version(n) for n in ['trimesh','PyMCubes']}
    require(report['runtime']['mesh_packages']['PyMCubes']=='0.1.4','Reviewed PyMCubes 0.1.4 required')
    report['runtime']['mesh_modules']={n:{'path':m.__file__,'sha256':digest(m.__file__)} for n,m in [('trimesh',trimesh),('mcubes',mcubes)]}
    network=None;audit=None
    try:
        mark('original_G_extraction_mode_construction')
        network=NeROShapeRenderer(deepcopy(p['cfg']),training=False)  # Original CLI constructor; no Trainer/seed override.
        mark('trusted_best_checkpoint_strict_load')
        state=load_checked(p['checkpoint'],CHECKPOINT_SHA256,torch,trusted=True)
        report['checkpoint_metadata']=historical_schema(state)
        report['network_comparison']=network_comparison(network,state['network_state_dict'],torch)
        require(report['network_comparison']==p['prior']['network_comparison'],'Extraction architecture differs from admitted G catalog')
        strict_network_load(network,state['network_state_dict'],torch);del state
        network.eval().cuda()  # Original ordering, strict state loading before this transfer.
        before=identities(network.state_dict())
        require(before==p['prior']['loaded_network_identity'],'Loaded weights differ from admitted G compatibility identities')
        report.update(strict_network_loaded=True,network_state_before=before,
                      effective_renderer=network.cfg,effective_shader=network.color_network.cfg,
                      sdf_configuration={'class':type(network.sdf_network).__name__,'scale':network.sdf_network.scale,
                                         'skip_in':list(network.sdf_network.skip_in),'num_layers':network.sdf_network.num_layers})
        audit=ExtractionAudit(network,extract_geometry,extract_fields,torch,np)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        mark('original_G_extract_geometry_r128')
        bbox_min=-torch.ones(3);bbox_max=torch.ones(3)
        with audit.active(),torch.no_grad():
            vertices,triangles=extract_geometry(bbox_min,bbox_max,128,0,lambda x:network.sdf_network.sdf(x))
        torch.cuda.synchronize()
        report['extraction_calls']=audit.complete()
        require(identities(network.state_dict())==before,'Extraction mutated network weights/buffers')
        report.update(network_unchanged=True,network_state_after=before)
        mark('original_default_Trimesh_export_and_read_only_inspection')
        report['mesh']=export_diagnostic(vertices,triangles,run,np,trimesh)
    finally:
        if audit is not None:report['extraction_calls']=dict(audit.record)
        torch.set_default_tensor_type('torch.FloatTensor')
        report['cleanup']={'default_tensor_type_restored':'torch.FloatTensor','profile_restored':sys.getprofile() is None}
        del network
        torch.cuda.empty_cache()


def execute(p,run_id,image_id,infrastructure_commit,trusted=False):
    validate_run_id(run_id)
    require(trusted is True,'Explicit trusted-checkpoint acknowledgement required; Torch 1.8 loads pickle')
    runtime_guards(p,image_id)
    activate('geometry',p['commit'])
    run=RunStore(p['resolver'],layout='mounted').create(run_id)
    files=identity();s=p['checkpoint'].stat()
    report={'schema_version':1,'protocol':PROTOCOL,'status':'running','phase':'allocated','run_id':run_id,
            'purpose':'diagnostic-only','research_geometry_admitted':False,'research_handoff_admitted':False,
            'checkpoint_producing_revision':'unresolved; historical artifact, not an attributable new G training run',
            'checkpoint_selection':{'protocol':'explicit-historical-best','path':str(p['checkpoint']),'sha256':CHECKPOINT_SHA256,
                'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,
                'original_CLI_selection':'model.pth','selected_artifact':'model_best.pth','silent_substitution':False},
            'scientific_commit':p['commit'],'source':p['source'],'source_manifest_sha256':digest(ROOT/'scientific_sources/manifest-v1.json'),
            'infrastructure_commit_declared':infrastructure_commit,'infrastructure_files':files,'image_id_declared':image_id,
            'compatibility_reference':p['proof_identity'],'compatibility_producing_infrastructure_declared':p['prior']['infrastructure_commit_declared'],
            'original_configuration':p['cfg'],'legacy_sha256':hashlib.sha256(p['raw']).hexdigest(),'settings':deepcopy(SETTINGS),
            'coordinates':{'frame':'Unmodified selected-checkpoint G SDF coordinates; producing camera normalization unverified',
                'units':'checkpoint SDF units, physical scale unknown','voxel_to_coordinate':'v/(resolution-1)*(bbox_max-bbox_min)+bbox_min',
                'axes':'unchanged XYZ','rescaling_or_reorientation_applied':False,'global_normal_orientation_verified':False},
            'datasets_read':False,'optimizer_state_restored':False,'optimizer_updates':0,'training_resumed':False,
            'rng_policy':'Original extraction has no explicit seed; no seed override, extra sampling or SDF query is introduced.',
            'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with run.output('G-mesh-diagnostic.json').open('x') as stream:
        def mark(phase):report['phase']=phase;write_report(stream,report);print('G-MESH-DIAGNOSTIC '+phase,flush=True)
        write_report(stream,report)
        try:
            workspace=run.output('cache/extraction-workspace');workspace.mkdir()
            for name in ['assets','configs']:(workspace/name).symlink_to(p['science']/name,target_is_directory=True)
            run.output('config.original.yaml').write_bytes(p['raw'])
            with runtime_environment(run):
                os.chdir(workspace)
                try:native(p,run,report,mark)
                except BaseException:traceback.print_exc();raise
            require(digest(p['checkpoint'])==CHECKPOINT_SHA256,'Historical checkpoint changed')
            require(digest(p['proof_identity']['path'])==p['proof_identity']['sha256'],'Compatibility evidence changed')
            baseline('geometry',p['commit']);require(files==identity(),'Extraction code/protocol changed during execution')
            mesh=report.get('mesh',{});output=run.output('meshes/bell_shape-best-150000-r128.ply')
            require(report.get('strict_network_loaded') is True and report.get('network_unchanged') is True and
                    mesh.get('path')=='meshes/bell_shape-best-150000-r128.ply' and output.is_file() and not output.is_symlink() and
                    digest(output)==mesh.get('sha256'),'Missing or inconsistent completed extraction/mesh evidence')
            require(all(report.get('extraction_calls',{}).get(k)==v for k,v in
                        {'extract_geometry_calls':1,'extract_fields_calls':1,'sdf_calls':8,'sdf_points':128**3,'maximum_query_points':64**3}.items()),
                    'Incomplete original extraction call budget')
            require(not list(run.checkpoints.iterdir()),'Extraction unexpectedly wrote checkpoints')
            report['imported_scientific_modules']=imported_sources(p['science'])
            report['output_hashes']={q.relative_to(run.root).as_posix():digest(q) for q in run.root.rglob('*')
                if q.is_file() and q.resolve()==q and q.name not in ['.experiment.json','G-mesh-diagnostic.json'] and
                not any(part in ('torch_extensions','home','tmp') for part in q.relative_to(run.root).parts)}
            report.update(status='complete',phase='complete',finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_report(stream,report);raise
        write_report(stream,report)
    return report
