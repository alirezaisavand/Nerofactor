"""Explicit isolated G compatibility/validation/manual serialization scopes.

All native imports and checkpoint pickle loading are lazy and researcher-side.
No natural training/validation schedule or extraction acceptance is implied.
"""
from copy import deepcopy
import datetime
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import traceback

from infrastructure.paths import PathResolver,InputRef
from infrastructure.runs import RunStore,validate_run_id
from infrastructure.signatures import require,read_report,write_report,CHECKPOINT_SHA256
from infrastructure.stage_sources import ROOT,PINS,baseline,activate,digest,imported_sources,infrastructure_identity
from infrastructure.stage_workspace import prepare_view,source_records
from infrastructure.g_checkpoint import STORED_STEP

ACCEPTANCE='docs/evidence/g-geometry-training-acceptance-v1.json'
ACCEPTANCE_SHA256='0526626944896d3f4cdfb5490d24b83a5650fc62e5b804b4a23ab7a28dd8d957'
IMAGE='sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137'
PROTOCOLS={'compatibility':'bell-G-checkpoint-compatibility-v1','optimizer':'bell-G-historical-optimizer-diagnostic-v1',
           'validation':'bell-G-camera0-validation-v1','roundtrip':'bell-G-manual-two-update-roundtrip-v1'}
DOCS=('docs/protocols/G_CHECKPOINT_COMPATIBILITY_V1.md','docs/protocols/G_CAMERA0_VALIDATION_V1.md',
      'docs/protocols/G_MANUAL_CHECKPOINT_ROUNDTRIP_V1.md')
MONTAGE='train_vis/bell_shape-val/149999-index-0.jpg'
MONTAGE_SHA256='2800003a665d47f757b1ec30b98d225bcfe19751b2608f1caa75587f8bcb3a44'


def admitted_prefix():
    value,_=read_report(ROOT/ACCEPTANCE,ACCEPTANCE_SHA256)
    require(value['result']=='G_regression_passed' and value['scientific_commit']==PINS['geometry'] and
            value['accepted_scopes']==['fresh-two-update-prefix'],'G training prerequisite scope mismatch')
    return value


def identity():
    return {**infrastructure_identity(),**{n:digest(ROOT/n) for n in (*DOCS,'run_g_geometry_diagnostic.py',ACCEPTANCE)}}


def prepare(action,commit,datasets,thesis,outputs,compatibility=None,compatibility_sha=None):
    require(action in PROTOCOLS,'Unknown diagnostic action')
    science,source=baseline('geometry',commit);admitted_prefix()
    roots={'datasets':datasets,'thesis':thesis}
    if compatibility is not None:
        path=Path(compatibility)
        require(path.is_absolute() and path.resolve()==path,'Compatibility report must be an explicit unaliased absolute path')
        roots['compatibility']=path.parent
    resolver=PathResolver(roots,outputs)
    require(ROOT!=resolver.output_root and ROOT not in resolver.output_root.parents and resolver.output_root not in ROOT.parents,
            'Diagnostic outputs overlap code')
    import yaml
    config_path=science/'configs/shape/syn/bell.yaml';raw=config_path.read_bytes()
    cfg=yaml.load(raw,Loader=yaml.FullLoader)
    require(cfg['network']=='shape' and cfg['database_name']=='syn/bell' and cfg['lr_type']=='warm_up_cos' and cfg['lr_cfg']=={},
            'Pinned bell scientific configuration required')
    family=resolver.input(InputRef('datasets','GlossySynthetic'),directory=True)
    if action in ('validation','roundtrip'):require((family/'bell').is_dir(),'Bell scene missing')
    checkpoint=None;prior=None
    if action!='roundtrip':
        checkpoint=resolver.input(InputRef('thesis','model/bell_shape/model_best.pth'))
        require(checkpoint.resolve()==checkpoint and digest(checkpoint)==CHECKPOINT_SHA256,'Historical checkpoint SHA-256 mismatch')
    else:require(not any(resolver.input_roots['thesis'].iterdir()),'Roundtrip requires an empty read-only thesis placeholder')
    if action in ('validation','optimizer'):
        require(compatibility is not None and compatibility_sha is not None,'Explicit completed G compatibility report/hash required')
        prior,prior_id=read_report(Path(compatibility),compatibility_sha)
        require(prior.get('protocol')==PROTOCOLS['compatibility'] and prior.get('action')=='compatibility' and
                prior.get('status')=='complete' and prior.get('scientific_commit')==commit and
                prior.get('checkpoint_sha256')==CHECKPOINT_SHA256 and prior.get('strict_network_loaded') is True and
                prior.get('network_comparison',{}).get('result')=='exact_structural_match' and
                prior.get('source',{}).get('scientific_files')==source['files'] and prior.get('infrastructure_files')==identity(),
                'Compatibility evidence does not admit this exact G diagnostic implementation')
    else:require(compatibility is None and compatibility_sha is None,'Unused compatibility reference is not accepted in this action')
    return {'action':action,'science':science,'source':source,'commit':commit,'resolver':resolver,'family':family,
            'checkpoint':checkpoint,'cfg':cfg,'raw':raw,'prior':prior,
            'prior_identity':prior_id if prior is not None else None}


def runtime_guards(p,image_id):
    import multiprocessing as mp
    require(os.geteuid()!=0 and os.statvfs('/').f_flag & os.ST_RDONLY,'Non-root/read-only root filesystem required')
    require(image_id==IMAGE,'Use the reviewed immutable image')
    for path in [ROOT,*p['resolver'].input_roots.values()]:
        require(os.statvfs(str(path)).f_flag & os.ST_RDONLY,'Code and every original input/evidence mount must be read-only')
    require({x.split(':',1)[0].strip() for x in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No-network runtime required')
    require((mp.get_start_method(allow_none=True) or mp.get_all_start_methods()[0])=='fork','Original Linux fork loader context required')


def setup_workspace(p,run,context):
    if p['action'] in ('validation','roundtrip'):
        return prepare_view(run,p['family'],'bell',p['science'],p['cfg'],context)
    workspace=run.output('cache/stage-workspace');workspace.mkdir()
    for name in ('assets','configs'):(workspace/name).symlink_to(p['science']/name,target_is_directory=True)
    return workspace,p['family'],None


def model_setup(p,run,cfg,torch,full=False):
    from train.trainer import Trainer
    from infrastructure.stage_execution import trainer_io
    trainer=trainer_io(Trainer,run)(cfg)  # Original G seed order; I/O attributes only.
    if full:
        trainer._init_dataset();trainer._init_network()
    else:
        from network.renderer import NeROShapeRenderer
        from train.lr_common_manager import name2lr_manager
        trainer.network=NeROShapeRenderer(trainer.cfg,training=False).cuda()
        trainer.lr_manager=name2lr_manager[trainer.cfg['lr_type']](trainer.cfg['lr_cfg'])
        trainer.optimizer=trainer.lr_manager.construct_optimizer(torch.optim.Adam,trainer.network)
    return trainer,Trainer


def read_observer(torch,allow_forward=True,allow_manual_save=False):
    from train.trainer import Trainer
    from dataset.database import GlossySyntheticDatabase,get_database_split
    from network.renderer import NeROShapeRenderer,build_imgs_info
    from utils.base_utils import read_pickle
    from infrastructure.g_validation_observer import ReadAudit
    forbidden=[Trainer.run,torch.optim.Adam.step,torch.Tensor.backward]
    if not allow_manual_save:forbidden.append(Trainer._save_model)
    return ReadAudit(GlossySyntheticDatabase,NeROShapeRenderer,build_imgs_info,get_database_split,read_pickle,
                     forbidden=forbidden,allow_forward=allow_forward)


def historical_context(trainer,state,report,torch):
    from infrastructure.g_checkpoint import historical_schema,network_comparison,strict_network_load
    report['checkpoint_metadata']=historical_schema(state)
    from inspect_bell_checkpoint import summarize_checkpoint
    report['checkpoint_metadata']['primitive_provenance']=summarize_checkpoint(state,torch)['primitive_provenance']
    report['network_comparison']=network_comparison(trainer.network,state['network_state_dict'],torch)
    require(report['network_comparison']['result']=='exact_structural_match','G checkpoint network incompatible; no partial load/evaluation')
    if report.get('compatibility_reference') is not None:
        require(report['network_comparison']==report['compatibility_reference']['network_comparison'],
                'Validation/optimizer model differs from the separately admitted G architecture')
    strict_network_load(trainer.network,state['network_state_dict'],torch)
    report['strict_network_loaded']=True


def validation(p,run,cfg,report,torch,np,mark):
    from train.train_valid import tqdm
    from utils.base_utils import color_map_backward
    from infrastructure.g_checkpoint import load_checked,identities
    from infrastructure.bell_inference import rng_identity,scalar_metric
    audit=read_observer(torch)
    try:
        with audit:
            mark('original_G_dataset_and_network_initialization')
            trainer,_=model_setup(p,run,cfg,torch,full=True)
            require(not trainer.network.cfg['val_geometry'] and trainer.network.cfg['test_downsample_ratio'] is True and
                    trainer.network.cfg['downsample_ratio']==.25 and trainer.network.cfg['test_ray_num']==512,'Original G validation settings changed')
            require(len(trainer.val_set_list)==1 and trainer.val_set_names==['val'] and len(trainer.val_set_list[0].dataset)==1,
                    'Exactly one original G validation dataset/view required')
            mark('trusted_checkpoint_strict_load')
            state=load_checked(p['checkpoint'],CHECKPOINT_SHA256,torch,trusted=True)
            historical_context(trainer,state,report,torch);del state
            before=identities(trainer.network.state_dict());cursor=trainer.network.train_batch_i
            report['rng_before_validation']=rng_identity(torch,np)
            audit.watch_loader(trainer.val_set_list[0],tqdm);audit.phase='evaluation'
            mark('original_G_validation_evaluator')
            values,key=trainer.val_evaluator(trainer.network,trainer.val_losses+trainer.val_metrics,trainer.val_set_list[0],
                                            149999,trainer.model_name,val_set_name='val')
            torch.cuda.synchronize()
            report['rng_after_validation']=rng_identity(torch,np)
    finally:
        report['validation_observation']={'split_calls':audit.splits,'getter_reads':audit.reads,
            'image_info_builds':audit.builds,'camera_pickle_reads':audit.pickles,
            'forward_calls':audit.forwards,'cleanup_errors':audit.cleanup_errors,
            'limit':'Partial observation is not successful camera/validation admission.'}
    report['read_provenance']=audit.record(trainer.network)
    model_database=report['read_provenance']['model_database']
    source_images=[r for r in audit.reads if r['database']==model_database and r['method']=='get_image']
    require(len(source_images)==128 and all(r['returned']['shape']==[800,800,3] for r in source_images),
            'Original full-resolution bell image reads were not preserved')
    require(identities(trainer.network.state_dict())==before and trainer.network.train_batch_i==cursor,
            'Validation mutated checkpoint network state or training-ray cursor')
    require(not trainer.optimizer.state,'Read-only validation unexpectedly populated optimizer state')
    outputs=audit.outputs[0]
    arrays={name:outputs[key].detach().cpu().numpy() for name,key in
            [('prediction','ray_rgb'),('ground_truth','gt_rgb'),('mask','gt_mask')]}
    require(arrays['prediction'].shape==arrays['ground_truth'].shape==(200,200,3) and
            arrays['prediction'].dtype==arrays['ground_truth'].dtype==np.float32 and
            arrays['mask'].shape==(200,200,1) and arrays['mask'].dtype==np.int32 and
            all(np.isfinite(x).all() for x in arrays.values()),'Unexpected original G validation RGB/mask')
    run.output('arrays').mkdir()
    for name,value in arrays.items():np.save(str(run.output('arrays/'+name+'.npy')),value,allow_pickle=False)
    from PIL import Image
    for name in ['prediction','ground_truth']:Image.fromarray(color_map_backward(arrays[name])).save(str(run.output('renders/'+name+'.png')))
    metrics={'psnr':scalar_metric(key),'ssim':float(np.asarray(values['ssim']).reshape(-1)[0])}
    require(np.isfinite(metrics['ssim']),'Original G SSIM is nonfinite')
    require(metrics['psnr']==scalar_metric(values['psnr']),'Original G key metric differs from evaluator output')
    report.update(metrics=metrics,evaluation_step=149999,stored_step=STORED_STEP,source_camera_id='0',
                  network_state_before=before,network_state_after=identities(trainer.network.state_dict()),
                  network_unchanged=True,optimizer_updates=0,optimizer_state_restored=False,
                  original_G_validation_aggregates={k:{'shape':list(np.asarray(v).shape),'dtype':str(np.asarray(v).dtype),
                    'sha256':hashlib.sha256(np.asarray(v).tobytes()).hexdigest(),'reporting_mean':scalar_metric(np.mean(v))} for k,v in values.items()},
                  effective_trainer=trainer.cfg,effective_renderer=trainer.network.cfg,effective_shader=trainer.network.color_network.cfg)
    # Pure diagnostic comparison; neither historical equality nor JPEG closeness is an acceptance requirement.
    from infrastructure.bell_camera_comparison import historical_comparison
    montage=p['resolver'].input(InputRef('thesis',MONTAGE))
    require(digest(montage)==MONTAGE_SHA256,'Historical montage identity changed')
    report['historical_comparison']=historical_comparison(color_map_backward(arrays['ground_truth']),metrics,montage,run)
    generated=run.output('renders/validation/bell_shape-val/149999-index-0.jpg')
    require(generated.is_file(),'Original G diagnostic montage was not generated')
    report['montage_comparison']={'historical_sha256':MONTAGE_SHA256,'G_sha256':digest(generated),
                                  'byte_identical':digest(generated)==MONTAGE_SHA256,'historical_acceptance_tolerance':None}
    with run.output('metrics.json').open('x') as f:write_report(f,metrics)


def native(p,run,cfg,report,mark):
    mark('scientific_imports')
    import sys
    import numpy as np
    import torch
    require(sys.version_info[:2]==(3,8) and torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and
            torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA A100 80GB PCIe',
            'Reviewed Python 3.8/Torch 1.8.1 CUDA 11.1 and one A100 80GB PCIe required')
    from infrastructure.geometry_training_smoke import runtime_identity
    from infrastructure.g_checkpoint import load_checked,identities,inspect_optimizer_load,manual_roundtrip,group_settings
    report['runtime']=runtime_identity(torch)
    if p['action']=='validation':validation(p,run,cfg,report,torch,np,mark);return
    if p['action']=='roundtrip':
        from train.trainer import Trainer
        from infrastructure.stage_execution import trainer_io
        from infrastructure.g_training import GTrainingProbe,GTwoUpdateTrace
        from network.renderer import NeROShapeRenderer
        from train.lr_common_manager import name2lr_manager
        mark('original_G_two_update_prefix_for_manual_roundtrip')
        donor=trainer_io(Trainer,run)(cfg)
        prefix={'status':'running','updates':[],'optimizer_calls_completed':0};report['prefix']=prefix
        def persist_prefix():mark('manual_roundtrip_prefix_'+prefix.get('phase','initializing'))
        GTwoUpdateTrace(donor,GTrainingProbe(donor,torch,np,prefix,persist_prefix),p['science']/'train/trainer.py').run()
        require(prefix['optimizer_calls_completed']==2 and not list(run.checkpoints.iterdir()),'Invalid manual roundtrip prefix/checkpoint state')
        prefix.update(status='complete',phase='bounded_stop_after_second_optimizer_return')
        mark('manual_G_save_load_after_two_updates')
        with read_observer(torch,allow_forward=False,allow_manual_save=True):
            receiver=SimpleNamespace(network=NeROShapeRenderer(donor.cfg,training=False).cuda())
            receiver.optimizer=name2lr_manager[donor.cfg['lr_type']](donor.cfg['lr_cfg']).construct_optimizer(torch.optim.Adam,receiver.network)
            report['manual_roundtrip']=manual_roundtrip(donor,receiver,Trainer,run.checkpoints/'manual-G-after-two-calls.pth',2,0.0,torch,np)
        require(report['manual_roundtrip']['optimizer_state_entries']>0,'Populated G Adam state required in this roundtrip')
        report.update(optimizer_updates=2,post_load_optimizer_updates=0,natural_schedule_tested=False,research_geometry_admitted=False)
        return
    with read_observer(torch,allow_forward=False) as observation:
        mark('original_G_extraction_mode_network_construction')
        trainer,_=model_setup(p,run,cfg,torch,full=False)
        report['G_fresh_optimizer_settings']=group_settings(trainer.optimizer.param_groups)
        state=load_checked(p['checkpoint'],CHECKPOINT_SHA256,torch,trusted=True)
        historical_context(trainer,state,report,torch)
        report.update(forward_calls=0,optimizer_updates=0,training_resumed=False,G_producing_provenance='unresolved',G_continuation_admitted=False,
                      loaded_network_identity=identities(trainer.network.state_dict()),effective_trainer=trainer.cfg,
                      effective_renderer=trainer.network.cfg,effective_shader=trainer.network.color_network.cfg)
        if p['action']=='optimizer':
            mark('separate_historical_optimizer_state_diagnostic')
            report['optimizer_diagnostic']=inspect_optimizer_load(trainer.optimizer,state['optimizer_state_dict'],trainer.network,torch)
            report['optimizer_state_restored']=True
        else:report['optimizer_state_restored']=False
        del state
    require(not observation.forwards,'Compatibility/optimizer diagnostic unexpectedly executed a forward pass')


def execute(p,run_id,image_id,infrastructure_commit,trusted=False,manual_ack=False,optimizer_ack=False):
    validate_run_id(run_id)
    require(optimizer_ack is (p['action']=='optimizer'),'Explicit, action-specific optimizer diagnostic acknowledgement required')
    require((p['action']=='roundtrip' and manual_ack is True and not trusted) or
            (p['action']!='roundtrip' and trusted is True and not manual_ack),'Explicit action-specific trust/manual-save gate required')
    runtime_guards(p,image_id)
    activate('geometry',p['commit'])
    run=RunStore(p['resolver'],layout='mounted').create(run_id)
    code=identity()
    report={'schema_version':1,'protocol':PROTOCOLS[p['action']],'action':p['action'],'purpose':'diagnostic-only',
            'status':'running','phase':'allocated','run_id':run_id,'scientific_commit':p['commit'],
            'source':{'stage':'geometry','algorithm':'geometry-G','scientific_commit':p['commit'],
                      'source_manifest_sha256':digest(ROOT/'scientific_sources/manifest-v1.json'),'scientific_files':p['source']['files']},
            'infrastructure_commit_declared':infrastructure_commit,'infrastructure_files':code,'image_id_declared':image_id,
            'checkpoint_sha256':CHECKPOINT_SHA256 if p['checkpoint'] else None,'compatibility_reference':p['prior'],
            'compatibility_reference_identity':p['prior_identity'],
            'natural_schedule_tested':False,'research_geometry_admitted':False,
            'original_configuration':p['cfg'],'legacy_sha256':hashlib.sha256(p['raw']).hexdigest(),
            'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'limitations':['New G diagnostic only; no historical producing revision inferred.',
                'Historical checkpoint pickle loading occurs only with explicit trust in the isolated native runtime.',
                'Direct validation, optimizer diagnostic, manual serialization and natural training schedule are distinct scopes.',
                'No 1000-step training, extraction, material execution or full training reproducibility admitted.']}
    with run.output('G-diagnostic.json').open('x') as stream:
        def mark(phase):report['phase']=phase;write_report(stream,report);print('G-DIAGNOSTIC '+phase,flush=True)
        write_report(stream,report)
        try:
            workspace,family,inputs=setup_workspace(p,run,report['source'])
            report['input_files']=inputs
            if p['checkpoint']:
                s=p['checkpoint'].stat()
                report['checkpoint_file_identity']={'path':str(p['checkpoint']),'bytes':s.st_size,
                    'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino,'sha256':CHECKPOINT_SHA256}
            cfg=deepcopy(p['cfg']);cfg['dataset_dir']=str(family)
            report['runtime_configuration']=cfg
            run.output('config.original.yaml').write_bytes(p['raw'])
            from infrastructure.bell_inference import runtime_environment
            with runtime_environment(run):
                os.chdir(workspace)
                try:native(p,run,cfg,report,mark)
                except BaseException:traceback.print_exc();raise
                report['imported_scientific_modules']=imported_sources(p['science'])
                sys_stdout=__import__('sys').stdout;sys_stdout.flush();__import__('sys').stderr.flush()
                require(code==identity(),'Diagnostic implementation/protocol changed')
                baseline('geometry',p['commit'])
                if inputs is not None:require(source_records(p['family']/'bell')==inputs,'Original scene changed')
                if p['checkpoint']:require(digest(p['checkpoint'])==CHECKPOINT_SHA256,'Historical checkpoint changed')
                if p['action']!='roundtrip':require(not any(run.checkpoints.iterdir()),'Read-only diagnostic wrote a checkpoint')
                report['output_hashes']={f.relative_to(run.root).as_posix():digest(f) for f in run.root.rglob('*')
                    if f.is_file() and f.resolve()==f and f!=run.output('G-diagnostic.json') and
                    f.name!='.experiment.json' and not any(x in ('torch_extensions','home','tmp') for x in f.relative_to(run.root).parts)}
                report.update(status='complete',phase='complete',finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_report(stream,report);raise
        write_report(stream,report)
    return report
