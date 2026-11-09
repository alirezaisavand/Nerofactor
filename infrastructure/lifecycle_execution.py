"""Bounded dual-stage lifecycle actions, using existing stage/run isolation."""
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from infrastructure.paths import PathResolver,InputRef
from infrastructure.runs import RunStore,validate_run_id
from infrastructure.signatures import require,read_report,write_report
from infrastructure.stage_sources import ROOT,PINS,baseline,activate,digest,infrastructure_identity,imported_sources
from infrastructure.stage_workspace import prepare_view,source_records
from infrastructure.g_lifecycle import runtime_guards,IMAGE,admitted_prefix
from infrastructure.lifecycle_state import original_save,state_record,restore_original,inspect_state,canonical

PROTOCOL='dual-stage-bell-lifecycle-v1'
A_ACCEPTED_SHA='6396208c186f9583585d80095b918a537784bed3bd58c559a2f12329f8caff38'
A_RAY_SHA='88262c38e0fb40930e4c026fe92aa971a6a4ef841d47e51b4f1f6efa4b267510'
DOC='docs/TRAINING_CHECKPOINT_EVALUATION_LIFECYCLE.md'
SCHEMA='docs/protocols/training-lifecycle-manifest-v1.schema.json'


def identity():
    return {**infrastructure_identity(),**{n:digest(ROOT/n) for n in [DOC,SCHEMA,'run_lifecycle_diagnostic.py']}}


def validate_manifest(value):
    import jsonschema
    try:jsonschema.Draft7Validator(json.loads((ROOT/SCHEMA).read_text())).validate(value)
    except jsonschema.ValidationError as exc:
        require(False,'Lifecycle manifest '+str(list(exc.absolute_path))+': '+exc.message[:500])


def splits(stage):
    if stage=='geometry':return {'train_ids':[str(i) for i in range(1,128)],'test_ids':['0'],'nvs_ids':None}
    from infrastructure.bell_inference import proposed_split
    return proposed_split()


def loaded_binaries():
    """Inspect already-loaded extension modules; never trigger plugin compilation."""
    modules={n:m for n,m in list(sys.modules.items()) if n=='_raytracing' or n.startswith('nvdiffrast')}
    ops=modules.get('nvdiffrast.torch.ops')
    cached=getattr(ops,'_cached_plugin',{})
    if isinstance(cached,dict):modules.update({'nvdiffrast-cache-'+str(k):v for k,v in cached.items()})
    result={}
    for name,module in modules.items():
        file=getattr(module,'__file__',None)
        if file and Path(file).suffix=='.so':result[name]={'path':file,'sha256':digest(file)}
    return {'observed':result,'limit':'Already-loaded extension modules/cache only; not a complete CUDA/shared-library closure.'}


def plan(stage,action,commit,datasets,thesis,output,*,geometry=None,mesh=None,mesh_sha=None,extraction=None,
         extraction_sha=None,accepted_material=None,accepted_ray=None,checkpoint_root=None,checkpoint=None,
         checkpoint_sha=None,producer=None,producer_sha=None):
    require(action in ('save','restore','evaluate'),'Unknown bounded lifecycle action')
    science,source=baseline(stage,commit)
    if stage=='material':
        from infrastructure.bell_material_integration import plan as mesh_plan,MESH_SHA256
        require(all(v is not None for v in [geometry,mesh,mesh_sha,extraction,extraction_sha,accepted_material,accepted_ray]),
                'Material requires explicit admitted historical mesh/extraction and accepted native reports')
        p=mesh_plan(commit,datasets,thesis,geometry,output,mesh,mesh_sha,extraction,extraction_sha,'rays')
        accepted,accepted_id=read_report(accepted_material,A_ACCEPTED_SHA)
        ray,ray_id=read_report(accepted_ray,A_RAY_SHA)
        require(accepted.get('protocol')=='historical-G-bell-to-A-two-update-v1' and accepted.get('status')=='complete' and
                accepted.get('run_id')=='bell-ga-train-a3b03c5c9d8245e6b067137cba9d2d06' and
                accepted.get('scientific_commit')==commit and accepted.get('source',{}).get('files')==source['files'] and
                accepted.get('optimizer_calls_completed')==2 and accepted.get('mesh_identity',{}).get('sha256')==MESH_SHA256 and
                accepted.get('original_configuration')==p['configuration'] and accepted.get('ray_gate_reference',{}).get('sha256')==A_RAY_SHA and
                ray.get('status')=='complete' and ray.get('ray_gate_passed') is True and ray.get('splits',{}).get('validation')==['58'],
                'Accepted material evidence scope/source/configuration/mesh differs')
        p.update(accepted_material=accepted,prerequisites={'material':accepted_id,'rays':ray_id})
        roots=dict(p['resolver'].input_roots)
        for key,path in [('accepted_material',accepted_material),('accepted_ray',accepted_ray)]:
            q=Path(path);require(q.is_absolute() and q.resolve()==q,'Unaliased accepted evidence required');roots[key]=q.parent
    else:
        require(all(v is None for v in [geometry,mesh,mesh_sha,extraction,extraction_sha,accepted_material,accepted_ray]),'Unused material selections for G')
        import yaml
        raw=(science/'configs/shape/syn/bell.yaml').read_bytes()
        roots={'datasets':datasets,'thesis':thesis}
        p={'configuration':yaml.load(raw,Loader=yaml.FullLoader),'raw':raw,'mesh_identity':None,
           'proof_identity':None,'prerequisites':{'geometry':admitted_prefix()}}
    require(not any(Path(thesis).iterdir()),'All lifecycle actions require empty read-only thesis placeholder; no historical checkpoint loading')
    if action=='save':
        require(all(v is None for v in [checkpoint_root,checkpoint,checkpoint_sha,producer,producer_sha]),'Fresh save cannot select an existing checkpoint')
    else:
        require(all(v is not None for v in [checkpoint_root,checkpoint,checkpoint_sha,producer,producer_sha]),'Explicit checkpoint/root/hash and producing report/hash required')
        roots['checkpoint']=checkpoint_root
    resolver=PathResolver(roots,output)
    require(resolver.output_root!=ROOT and ROOT not in resolver.output_root.parents and resolver.output_root not in ROOT.parents,'Outputs overlap code')
    family=resolver.input(InputRef('datasets','GlossySynthetic'),directory=True)
    require((family/'bell').is_dir(),'Missing bell dataset')
    p.update(stage=stage,action=action,science=science,source=source,scientific_commit=commit,resolver=resolver,
             family=family,checkpoint=None,producer=None,producer_identity=None)
    if action!='save':
        selected=resolver.input(InputRef('checkpoint','checkpoints/model.pth'))
        proof_path=resolver.input(InputRef('checkpoint','lifecycle.json'))
        require(Path(checkpoint).is_absolute() and Path(checkpoint)==Path(checkpoint).resolve()==selected and
                Path(producer).is_absolute() and Path(producer)==Path(producer).resolve()==proof_path and digest(selected)==checkpoint_sha,
                'Explicit original diagnostic checkpoint/report selection differs')
        prior,prior_id=read_report(proof_path,producer_sha)
        validate_manifest(prior)
        require(prior.get('manifest_version')==1 and prior.get('protocol')==PROTOCOL and prior.get('action')=='save' and
                prior.get('status')=='complete' and prior.get('stage')==stage and prior.get('scientific_commit')==commit and
                prior.get('scientific_source')==source and prior.get('infrastructure_files')==identity() and
                prior.get('original_configuration')==p['configuration'] and prior.get('splits')==splits(stage) and
                prior.get('research_handoff_admitted') is False and prior.get('checkpoint',{}).get('sha256')==checkpoint_sha and
                prior['checkpoint'].get('step')==2 and prior.get('optimizer_calls_completed')==2,
                'Checkpoint producing evidence does not admit this exact lifecycle source/configuration/action')
        require(prior.get('mesh_identity',{}).get('sha256')==mesh_sha if stage=='material' else prior.get('mesh_identity') is None,
                'Checkpoint mesh/scene identity differs')
        p.update(checkpoint=selected,checkpoint_sha=checkpoint_sha,producer=prior,producer_identity=prior_id)
    return p


def io_trainer(base,p,run,report,torch,np):
    class LifecycleTrainer(base):
        def __init__(self,cfg):
            super().__init__(cfg)
            self.model_dir=str(run.logs);self.best_pth_fn=str(run.checkpoints/'model_best.pth')
            self.pth_fn=str(p['checkpoint'] if p['action']=='restore' else run.checkpoints/'model.pth')
        def _load_model(self):
            if p['action']=='save':
                require(not Path(self.pth_fn).exists() and not Path(self.best_pth_fn).exists(),'Implicit checkpoint reuse refused')
                return base._load_model(self)
            require(p['action']=='restore','Evaluation must not invoke original resume loader')
            best,step,details=restore_original(self,base,p['checkpoint'],p['checkpoint_sha'],p['producer']['checkpoint'],torch,np,True)
            report['restoration']=details
            return best,step
    return LifecycleTrainer


def native(p,run,report,persist):
    import numpy as np
    import torch
    require(sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and
            torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe',
            'Reviewed Python/Torch/CUDA/selected A100 required')
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu','Original CPU float32 constructor context required')
    from train.trainer import Trainer
    from train.train_valid import ValidationEvaluator
    from infrastructure.geometry_training_smoke import runtime_identity
    report['runtime']=runtime_identity(torch)
    if p['stage']=='material':
        import _raytracing
        report['runtime']['raytracing_binary']={'path':_raytracing.__file__,'sha256':digest(_raytracing.__file__)}
    if p['producer'] is not None:require(report['runtime']==p['producer']['runtime'],'Producer/consumer runtime identity differs')
    trainer=io_trainer(Trainer,p,run,report,torch,np)(report['runtime_configuration'])
    try:
        if p['action']=='evaluate':
            trainer._init_dataset();trainer._init_network()
            from infrastructure.lifecycle_evaluation import evaluate
            state=inspect_state(trainer,p['checkpoint'],p['checkpoint_sha'],p['producer']['checkpoint'],torch,np,True)
            from infrastructure.g_checkpoint import strict_network_load
            strict_network_load(trainer.network,state['network_state_dict'],torch);del state
            evaluate(trainer,p,run,report,persist,torch,np)
        elif p['action']=='save':
            if p['stage']=='geometry':
                from infrastructure.g_training import GTrainingProbe,GTwoUpdateTrace
                GTwoUpdateTrace(trainer,GTrainingProbe(trainer,torch,np,report,persist),p['science']/'train/trainer.py').run()
            else:
                from infrastructure.material_smoke import run_bounded
                from infrastructure.bell_material_integration import BellProbe
                run_bounded(trainer,p['science'],torch,np,report,persist,probe_class=BellProbe)
            require(report['optimizer_calls_completed']==2 and not list(run.checkpoints.iterdir()),'Invalid fresh prefix/checkpoint budget')
            report['checkpoint']=original_save(trainer,Trainer,run.checkpoints/'model.pth',torch,np)
            report['save_sampling_cursor']=trainer.network.train_batch_i
        else:
            from infrastructure.lifecycle_training import GResumeProbe,AResumeProbe,ReloadTrace,ForbidCalls
            probe=(GResumeProbe if p['stage']=='geometry' else AResumeProbe)(trainer,torch,np,report,persist)
            control=ReloadTrace(trainer,probe,p['science']/'train/trainer.py',p['stage'])
            if p['stage']=='material':
                from network.renderer import NeROMaterialRenderer
                from network.field import MCShadingNetwork
                from network.loss import NeRFRenderLoss,MaterialRegLoss
                probe.bind_calls(NeROMaterialRenderer,MCShadingNetwork,NeRFRenderLoss,MaterialRegLoss,Trainer,ValidationEvaluator)
                with probe.observe_calls():control.run()
            else:
                from network.renderer import NeROShapeRenderer
                with ForbidCalls([Trainer._save_model,ValidationEvaluator.__call__,NeROShapeRenderer.nvs]):control.run()
            require(report['optimizer_calls_completed']==1 and len(report['updates'])==1,'Invalid restored update budget')
            report['state_after_update']=state_record(trainer,np)
        if p['stage']=='material':
            from infrastructure.bell_material_integration import verify_loaded_mesh
            verify_loaded_mesh(trainer.network.mesh,p,np)
        report['effective_configuration']={'trainer':trainer.cfg,'renderer':trainer.network.cfg,
            'shader':(trainer.network.color_network if p['stage']=='geometry' else trainer.network.shader_network).cfg,
            'scheduler':canonical(vars(trainer.lr_manager)),'losses':[{'class':type(v).__name__,'cfg':getattr(v,'cfg',None)} for v in trainer.val_losses]}
        report['loaded_extension_binaries']=loaded_binaries()
    finally:
        torch.set_default_tensor_type('torch.FloatTensor');torch.cuda.empty_cache()
        report['runtime_cleanup']={'default_tensor_type_restored':'torch.FloatTensor','profile_restored':sys.getprofile() is None,
                                   'trace_restored':sys.gettrace() is None}


def execute(p,run_id,image_id,infrastructure_commit,trusted=False):
    validate_run_id(run_id);require(trusted is True,'Explicit isolated camera/checkpoint trust/manual-save acknowledgement required')
    runtime_guards(p,image_id)
    if p['producer'] is not None:
        require(p['producer']['run_id']!=run_id and p['producer']['image_id_declared']==image_id,'Fresh consumer ID/matching image required')
    activate(p['stage'],p['scientific_commit'])
    run=RunStore(p['resolver'],layout='mounted').create(run_id)
    report={'manifest_version':1,'protocol':PROTOCOL,'action':p['action'],'stage':p['stage'],'run_id':run_id,
            'status':'running','phase':'allocated','scientific_commit':p['scientific_commit'],'scientific_source':p['source'],
            'infrastructure_commit_declared':infrastructure_commit,'infrastructure_files':identity(),'image_id_declared':image_id,
            'original_configuration':p['configuration'],'legacy_sha256':hashlib.sha256(p['raw']).hexdigest(),
            'mesh_identity':p['mesh_identity'],'prerequisites':p['prerequisites'],'splits':splits(p['stage']),
            'ray_splits':{'train':splits('material')['train_ids'],'validation':['58'],'nvs':splits('material')['nvs_ids']},
            'research_handoff_admitted':False,'natural_schedule_tested':False,'uninterrupted_equivalence_proven':False,
            'RNG_loader_cursor_state_in_original_checkpoint':False,'optimizer_calls_completed':0,
            'checkpoint_input':({'path':str(p['checkpoint']),'sha256':p['checkpoint_sha']} if p['checkpoint'] else None),
            'producer_reference':p['producer_identity'],'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    report.update(runtime_configuration=None,input_files=None)
    if p['producer'] is not None:report['producer_sampling']=p['producer']['updates'][-1]['sampling']
    with run.output('lifecycle.json').open('x') as stream:
        def persist():write_report(stream,report)
        persist()
        try:
            report['phase']='owned_workspace';persist()
            workspace,family,inputs=prepare_view(run,p['family'],'bell',p['science'],p['configuration'],p['source'])
            report['input_files']=inputs
            if p['stage']=='material':require(inputs==p['accepted_material']['input_files'],'Accepted material scene inputs changed')
            if p['producer'] is not None:require(inputs==p['producer']['input_files'],'Checkpoint scene inputs changed')
            cfg=deepcopy(p['configuration']);cfg['dataset_dir']=str(family)
            if p['stage']=='material':cfg['mesh']=str(p['selected'].mesh)
            report['runtime_configuration']=cfg;run.output('config.original.yaml').write_bytes(p['raw'])
            from infrastructure.bell_inference import runtime_environment
            with runtime_environment(run):
                os.chdir(workspace)
                try:native(p,run,report,persist)
                except BaseException:traceback.print_exc();raise
            require(report.get('runtime_cleanup',{}).get('profile_restored') is True and
                    report['runtime_cleanup'].get('trace_restored') is True,'Lifecycle observer cleanup incomplete')
            require(source_records(p['family']/'bell')==inputs and report['infrastructure_files']==identity(),'Inputs/implementation changed')
            baseline(p['stage'],p['scientific_commit'])
            if p['checkpoint']:
                require(digest(p['checkpoint'])==p['checkpoint_sha'] and digest(p['producer_identity']['path'])==p['producer_identity']['sha256'],
                        'Input checkpoint/producing report changed')
            if p['stage']=='material':require(digest(p['selected'].mesh)==p['mesh_identity']['sha256'],'Selected mesh changed')
            require(list(run.checkpoints.iterdir())==([run.checkpoints/'model.pth'] if p['action']=='save' else []),'Unexpected checkpoint writer')
            report['imported_scientific_modules']=imported_sources(p['science'])
            report['output_hashes']={q.relative_to(run.root).as_posix():digest(q) for q in run.root.rglob('*')
                if q.is_file() and q.resolve()==q and q.name not in ['lifecycle.json','.experiment.json'] and
                not any(part in ('torch_extensions','home','tmp') for part in q.relative_to(run.root).parts)}
            report.update(status='complete',phase='complete',finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            # Existing inventoried dependency; local schema has no external refs.
            validate_manifest(report)
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc));persist();raise
        finally:
            report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();persist()
    return report
