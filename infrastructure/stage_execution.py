"""Fresh stage-aware execution. Native work requires an isolated researcher process."""
from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import traceback

from infrastructure.paths import PathResolver, InputSelection
from infrastructure.runs import RunStore, validate_run_id
from infrastructure.signatures import require, write_report
from infrastructure.stage_sources import ROOT, PINS, baseline, activate, imported_sources, digest, infrastructure_identity
from infrastructure.stage_workspace import prepare_view, source_records, handoff


def plan(stage, commit, arm, datasets, thesis, output, cfg_path=None, config_root=None, extra_roots=None):
    science, source = baseline(stage, commit)
    require(arm in ('R','R-repeat','C') and (stage=='geometry' or arm=='C'), 'Invalid stage/configuration arm')
    roots={'datasets':datasets,'thesis':thesis}
    for name,path in (extra_roots or {}).items():
        require(name not in roots,'Duplicate input root');roots[name]=path
    resolver=PathResolver(roots,output)
    require(resolver.output_root!=ROOT and ROOT not in resolver.output_root.parents and resolver.output_root not in ROOT.parents,
            'Outputs overlap repository')
    legacy=science/('configs/shape/syn/bell.yaml' if stage=='geometry' else 'configs/material/real/bear.yaml')
    if arm=='C':
        from infrastructure.config import load_config
        config=load_config(cfg_path,config_root=config_root)
        require(config.stage==stage and config.legacy_source==legacy and config.original.get('overrides')=={},
                'Envelope must select exact stage scientific configuration, without overrides')
        require(config.inputs.checkpoint is None,'No checkpoint initialization/resume admitted by reset')
        require(config.preprocessing.cache=='cache/stage-workspace','This version implements only the stage filesystem cache')
        raw=legacy.read_bytes();values=deepcopy(config.legacy);selection=config.inputs
        envelope=Path(cfg_path).read_bytes()
    else:
        import yaml
        require(cfg_path is None and config_root is None,'Reference loads pinned legacy YAML independently')
        raw=legacy.read_bytes();values=yaml.load(raw,Loader=yaml.FullLoader)
        selection=InputSelection('GlossySynthetic');envelope=None
    require(values.get('network')==('shape' if stage=='geometry' else 'material'),'Foreign renderer selection')
    selected=resolver.resolve_inputs(selection)
    require(selection.dataset==('GlossySynthetic' if stage=='geometry' else 'GlossyReal'),'Pinned stage dataset family required')
    require((selected.mesh is not None)==(stage=='material') and selected.checkpoint is None,'Ambiguous stage inputs')
    scene='bell' if stage=='geometry' else 'bear'
    require((selected.dataset/scene).is_dir(),'Required source scene is missing')
    return {'stage':stage,'scientific_commit':commit,'science':science,'source':source,'arm':arm,
            'route':'schema-1-pinned-stage' if arm=='C' else 'independent-pinned-legacy-FullLoader',
            'resolver':resolver,'selected':selected,'scene':scene,'configuration':values,'raw':raw,'envelope':envelope}


def native_guards(p, image_id):
    import multiprocessing as mp
    require((mp.get_start_method(allow_none=True) or mp.get_all_start_methods()[0])=='fork',
            'This version preserves the Linux fork loader context; spawn import isolation needs separate review')
    require(os.geteuid()!=0 and os.statvfs('/').f_flag & os.ST_RDONLY,'Non-root/read-only root filesystem required')
    for root in [ROOT,*p['resolver'].input_roots.values()]:
        require(os.statvfs(str(root)).f_flag & os.ST_RDONLY,'Read-only source and input mounts required')
    require({line.split(':',1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No-network runtime required')
    require(image_id=='sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137','Reviewed image identity required')
    if p['stage']=='geometry':require(not any(p['resolver'].input_roots['thesis'].iterdir()),'Geometry smoke requires empty read-only thesis placeholder')


def trainer_io(base, run):
    """Change only storage attributes, after original constructor/seed operations."""
    class StageTrainer(base):
        def __init__(self,cfg):
            super().__init__(cfg)
            self.model_dir=str(run.logs)
            self.pth_fn=str(run.checkpoints/'model.pth')
            self.best_pth_fn=str(run.checkpoints/'model_best.pth')
        def _load_model(self):
            require(not Path(self.pth_fn).exists() and not Path(self.best_pth_fn).exists(),
                    'Fresh stage run refuses implicit checkpoint resume')
            return super()._load_model()
    return StageTrainer


def native_run(p,run,report,persist):
    report['phase']='scientific_imports';persist()
    import numpy as np
    import torch
    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and
            torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe',
            'Reviewed Torch/CUDA runtime and one selected A100 80GB PCIe required')
    import sys
    require(sys.version_info[:2]==(3,8),'Reviewed Python 3.8 runtime required')
    from train.trainer import Trainer
    require(Path(Trainer.run.__code__.co_filename).resolve()==p['science']/'train/trainer.py','Mixed stage trainer')
    from infrastructure.geometry_training_smoke import runtime_identity
    report['runtime']=runtime_identity(torch)
    import multiprocessing as mp
    report['runtime']['loader_start_method']=mp.get_start_method(allow_none=True) or mp.get_all_start_methods()[0]
    persist()
    report['phase']='trainer_constructor';persist()
    trainer=trainer_io(Trainer,run)(report['runtime_configuration'])
    if p['stage']=='geometry':
        from infrastructure.g_training import GTrainingProbe,GTwoUpdateTrace
        probe=GTrainingProbe(trainer,torch,np,report,persist)
        GTwoUpdateTrace(trainer,probe,p['science']/'train/trainer.py').run()
        require(report['optimizer_calls_completed']==2 and not list(run.checkpoints.glob('*.pth')),'Unexpected G checkpoint/update budget')
    elif report.get('bounded_material') is True:
        from infrastructure.material_smoke import PROTOCOL,run_bounded
        require(report['protocol']==PROTOCOL,'Unknown bounded material protocol')
        run_bounded(trainer,p['science'],torch,np,report,persist)
        require(not list(run.checkpoints.iterdir()),'Bounded material diagnostic wrote a checkpoint')
    else:
        # Actual A loop, clipping, NVS and selection; never invoked in CPU checks.
        report['phase']='original_material_loop';persist()
        def capture():
            from infrastructure.training_probe import optimizer_record
            report.update(effective_trainer=trainer.cfg,effective_renderer=trainer.network.cfg,
                          effective_shader=trainer.network.shader_network.cfg,
                          effective_losses=[{'registered_name':name,'class':type(loss).__module__+'.'+type(loss).__name__,
                             'configuration':getattr(loss,'cfg',None)} for name,loss in zip(trainer.cfg['loss'],trainer.train_losses)],
                          scheduler={'class':type(trainer.lr_manager).__name__,'attributes':dict(vars(trainer.lr_manager))},
                          initial_optimizer=optimizer_record(trainer.optimizer,trainer.network,np))
            persist()
        run_material_owned(trainer,report,persist,capture)
    report['imported_scientific_modules']=imported_sources(p['science'])


def run_material_owned(trainer,report,persist,capture=None):
    """Observe only resource locals; execute the original A run body unchanged."""
    import sys
    previous=sys.gettrace()
    require(previous is None and sys.implementation.name=='cpython','Untraced CPython required for owned cleanup')
    code=trainer.run.__func__.__code__;resources={}
    from infrastructure.bounded_training import run_boundaries
    _,points=run_boundaries(Path(code.co_filename))
    def observe(frame,event,arg):
        if frame.f_code is not code:return None
        require(frame.f_locals.get('self') is trainer,'Foreign material run frame')
        if event=='line':
            if frame.f_lineno==points['initialized'] and capture is not None:capture()
            for name in ['train_iter','pbar']:
                if name in frame.f_locals:resources[name]=frame.f_locals[name]
        return observe
    errors=[]
    try:
        sys.settrace(observe)
        require(sys.gettrace() is observe,'Material cleanup observation unavailable')
        trainer.run()
    finally:
        sys.settrace(previous)
        targets=[('workers',resources.get('train_iter'),'_shutdown_workers'),('progress',resources.get('pbar'),'close'),
                 ('writer',getattr(getattr(trainer,'logger',None),'writer',None),'close')]
        for name,target,method in targets:
            if callable(getattr(target,method,None)):
                try:getattr(target,method)()
                except Exception as exc:errors.append({'resource':name,'error':str(exc)})
        report['cleanup_errors']=errors;persist()
    require(not errors,'Material owned resource cleanup failed')


def execute(p,run_id,image_id,infrastructure_commit,mesh_handoff=None,bounded_material=False):
    validate_run_id(run_id)
    require(type(bounded_material) is bool,'Explicit bounded-material boolean required')
    diagnostic_mesh=None
    if bounded_material:
        from infrastructure.material_smoke import PROTOCOL,admit_mesh,diagnostic_files
        require(mesh_handoff is None,'Historical diagnostic cannot claim a G mesh handoff')
        diagnostic_mesh=admit_mesh(p)
    require(p['stage']!='geometry' or mesh_handoff is None,'Geometry cannot accept a material mesh-handoff argument')
    native_guards(p,image_id)
    transfer=handoff(mesh_handoff,p['selected'].mesh,p['resolver'],p['configuration']['database_name']) if p['stage']=='material' and not bounded_material else None
    # Cold import isolation is activated before Torch/scientific imports or allocation.
    activate(p['stage'],p['scientific_commit'])
    run=RunStore(p['resolver'],layout='mounted').create(run_id)
    source_context={'stage':p['stage'],'algorithm':p['source']['algorithm'],'scientific_commit':p['scientific_commit'],
                    'scientific_files':p['source']['files'],'source_manifest_sha256':digest(ROOT/'scientific_sources/manifest-v1.json'),
                    'declared_infrastructure_commit':infrastructure_commit,
                    'declared_image_id':image_id,
                    'infrastructure_files':infrastructure_identity()}
    if bounded_material:source_context['diagnostic_files']=diagnostic_files()
    report={'schema_version':1,'protocol':PROTOCOL if bounded_material else ('bell-G-two-update-training-v1' if p['stage']=='geometry' else 'material-A-stage-execution-v1'),
            'bounded_material':bounded_material,
            'status':'running','phase':'allocated','stage':p['stage'],'scientific_commit':p['scientific_commit'],
            'run_id':run_id,'arm':p['arm'],'route':p['route'],'source':source_context,'declared_image_id':image_id,
            'original_configuration':p['configuration'],'legacy_sha256':__import__('hashlib').sha256(p['raw']).hexdigest(),
            'mesh_handoff':transfer,'checkpoint_input':None,'optimizer_calls_completed':0 if p['stage']=='geometry' else None,
            'mesh_handoff_identity':{'path':str(mesh_handoff),'sha256':digest(mesh_handoff)} if transfer else None,
            'mesh_identity':diagnostic_mesh if bounded_material else ({'path':str(p['selected'].mesh),'sha256':digest(p['selected'].mesh),
                             'bytes':p['selected'].mesh.stat().st_size} if transfer else None),
            'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'limitations':['Fresh weights only; historical checkpoint G production/resume eligibility not established.',
                           'Native execution and this source selection require separate review; CPU fixtures are not native acceptance.',
                           'Original relative-path writers use an experiment-owned filesystem view; input files are mounted read-only.',
                           'Infrastructure/image commit declarations and handoff metadata are not independent historical attestation.']}
    if bounded_material:
        report.update(optimizer_calls_completed=0,diagnostic_only=True,research_geometry_admitted=False,
                      acceptance_scope='A anisotropic two-update diagnostic only; no G handoff, full training or publication result')
    with run.output('stage-run.json').open('x') as stream:
        def persist():write_report(stream,report)
        persist()
        try:
            report['phase']='cache_view_setup';persist()
            workspace,family,inputs=prepare_view(run,p['selected'].dataset,p['scene'],p['science'],p['configuration'],source_context)
            report['input_files']=inputs
            cfg=deepcopy(p['configuration']);cfg['dataset_dir']=str(family)
            for nested in [cfg.get('train_dataset_cfg',{}),*[v['cfg'] for v in cfg.get('val_set_list',[])]]:
                if 'dataset_dir' in nested:nested['dataset_dir']=str(family)
            if p['stage']=='material':cfg['mesh']=str(p['selected'].mesh)
            report['runtime_configuration']=cfg
            run.output('config.original.yaml').write_bytes(p['raw'])
            if p['envelope'] is not None:run.output('config.envelope.yaml').write_bytes(p['envelope'])
            from infrastructure.bell_inference import runtime_environment
            with runtime_environment(run):
                os.chdir(str(workspace))
                try:native_run(p,run,report,persist)
                except BaseException:traceback.print_exc();raise
            require(source_records(p['selected'].dataset/p['scene'])==inputs,'Source input changed during execution')
            if transfer:
                require(digest(p['selected'].mesh)==transfer['mesh_sha256'] and
                        digest(mesh_handoff)==report['mesh_handoff_identity']['sha256'],'Input mesh/handoff changed during execution')
                require(handoff(mesh_handoff,p['selected'].mesh,p['resolver'],p['configuration']['database_name'])==transfer,
                        'Producing geometry evidence changed during execution')
            if bounded_material:
                require(digest(p['selected'].mesh)==diagnostic_mesh['sha256'] and diagnostic_files()==source_context['diagnostic_files'],
                        'Diagnostic mesh/definition changed during execution')
                require(report['optimizer_calls_completed']==2 and len(report.get('updates',[]))==2 and
                        not list(run.checkpoints.iterdir()),'Invalid bounded material completion')
                require(not any(q.is_file() for q in run.output('renders').rglob('*')),'Unexpected material validation/NVS render')
            baseline(p['stage'],p['scientific_commit'])
            require(digest(ROOT/'scientific_sources/manifest-v1.json')==source_context['source_manifest_sha256'] and
                    infrastructure_identity()==source_context['infrastructure_files'],'Source manifest/infrastructure changed')
            report.update(status='complete',phase='bounded_stop_after_second_optimizer_return' if p['stage']=='geometry' or bounded_material else 'training_complete',
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            report['output_hashes']={q.relative_to(run.root).as_posix():digest(q) for q in run.root.rglob('*')
                if q.is_file() and q.resolve()==q and q!=run.output('stage-run.json') and '.experiment.json'!=q.name
                and not any(part in ('torch_extensions','home','tmp') for part in q.relative_to(run.root).parts)}
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc));persist();raise
        persist()
    return report
