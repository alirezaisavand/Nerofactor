"""Strict original-format state admission. Torch is injected after isolation.

Hash checks precede pickle deserialization. Structural/semantic checks precede
any live weight or optimizer restoration. No RNG/loader/cursor restoration.
"""
import hashlib
import json
from pathlib import Path

from infrastructure.signatures import require
from infrastructure.stage_sources import digest
from infrastructure.g_checkpoint import (mapping,load_checked,identities,network_comparison,
                                         optimizer_preflight,group_settings)
from infrastructure.training_probe import optimizer_record


def canonical(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def record_hash(value):
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def state_record(trainer,np):
    network=identities(trainer.network.state_dict())
    optimizer=canonical(optimizer_record(trainer.optimizer,trainer.network,np))
    types=optimizer_types(trainer.optimizer,trainer.network)
    return {'network':network,'network_sha256':record_hash(network),'optimizer':optimizer,
            'optimizer_sha256':record_hash({'record':optimizer,**types}),**types}


def optimizer_types(optimizer,model):
    names={id(p):n for n,p in model.named_parameters()}
    describe=lambda x:type(x).__module__+'.'+type(x).__name__
    return {'optimizer_group_types':[{k:describe(v) for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
            'optimizer_counter_types':{names[id(p)]:{'type':describe(v['step']),
                'tensor':identities({'step':v['step']})['step'] if hasattr(v['step'],'detach') else None}
                for p,v in optimizer.state.items()}}


def original_save(trainer,base,path,torch,np):
    path=Path(path)
    require(not path.exists() and not path.is_symlink() and not Path(trainer.best_pth_fn).exists(),'Checkpoint collision')
    before=state_record(trainer,np)
    base._save_model(trainer,2,0,save_fn=str(path))  # Preserve original initialized best_para integer; early diagnostic call.
    require(path.is_file() and path.resolve()==path and path.stat().st_nlink==1,'Unsafe owned checkpoint')
    require(before==state_record(trainer,np),'Serialization mutated live model/optimizer')
    return {'path':'checkpoints/model.pth','sha256':digest(path),'bytes':path.stat().st_size,
            'step':2,'best_para':0,'invocation':'explicit early diagnostic, not natural cadence or best selection',
            'state':before,'scheduler':canonical(vars(trainer.lr_manager))}


def inspect_state(trainer,path,sha,proof,torch,np,trusted):
    state=load_checked(path,sha,torch,trusted=trusted)
    require(mapping(state) and set(state)=={'step','best_para','network_state_dict','optimizer_state_dict'},'Incomplete/foreign original checkpoint fields')
    require(type(state['step']) is int and state['step']==proof['step']==2 and
            type(state['best_para']) is int and state['best_para']==proof['best_para']==0,
            'Stored step/best differs from diagnostic admission')
    require(network_comparison(trainer.network,state['network_state_dict'],torch)['result']=='exact_structural_match',
            'Strict network catalog mismatch; no partial restoration')
    require(identities(state['network_state_dict'])==proof['state']['network'],'Saved network differs from producing report')
    optimizer_preflight(trainer.optimizer,state['optimizer_state_dict'],trainer.network,torch)
    saved_groups=canonical(group_settings(state['optimizer_state_dict']['param_groups']))
    fresh_groups=canonical(group_settings(trainer.optimizer.param_groups))
    require([{k:v for k,v in g.items() if k!='lr'} for g in saved_groups]==
            [{k:v for k,v in g.items() if k!='lr'} for g in fresh_groups],
            'Checkpoint would replace original optimizer hyperparameters')
    # Compare every saved moment/counter/group before restoration using a detached
    # parameter-name view, without loading state into even a temporary optimizer.
    from types import SimpleNamespace
    detached_state={};groups=[]
    for saved,live in zip(state['optimizer_state_dict']['param_groups'],trainer.optimizer.param_groups):
        groups.append({**saved,'params':live['params']})
        for index,parameter in zip(saved['params'],live['params']):
            if index in state['optimizer_state_dict']['state']:detached_state[parameter]=state['optimizer_state_dict']['state'][index]
    view=SimpleNamespace(param_groups=groups,state=detached_state)
    # optimizer_record includes class identity; supply the original identity only
    # after its actual native class/parameter mapping has been checked.
    actual=canonical(optimizer_record(view,trainer.network,np))
    actual['class']=type(trainer.optimizer).__module__+'.'+type(trainer.optimizer).__name__
    expected=proof['state']['optimizer']
    types=optimizer_types(view,trainer.network)
    require(actual==expected and record_hash({'record':actual,**types})==proof['state']['optimizer_sha256'],
            'Saved optimizer moments/counters/groups differ from producing report')
    for key,value in types.items():
        require(value==proof['state'][key],'Optimizer counter/group types differ from producing report')
    require(record_hash(proof['state']['network'])==proof['state']['network_sha256'] and
            canonical(vars(trainer.lr_manager))==proof['scheduler'],'Network digest/scheduler differs from admission')
    require(bool(actual['state']),'Diagnostic checkpoint has no populated optimizer state')
    return state


def restore_original(trainer,base,path,sha,proof,torch,np,trusted=False):
    require(str(trainer.pth_fn)==str(path),'Original loader must target explicitly selected input only')
    before=state_record(trainer,np)
    require(before['network']!=proof['state']['network'],'Fresh receiver must differ from trained checkpoint')
    state=inspect_state(trainer,path,sha,proof,torch,np,trusted);del state
    require(before==state_record(trainer,np),'Preflight mutated live state')
    best,step=base._load_model(trainer)  # Original torch.load/strict default network/optimizer restoration.
    require(best==proof['best_para'] and step==proof['step'] and digest(path)==sha,'Original load returned wrong progress/file')
    after=state_record(trainer,np)
    require(after==proof['state'],'Original restoration changed tensor bytes, moments, counters or groups')
    return best,step,{'exact_restoration':True,'before':before,'restored':after,'stored_step':step,
                      'best_para':float(best),'scheduler':canonical(vars(trainer.lr_manager)),
                      'restored_progress_lr':[g['lr'] for g in after['optimizer']['groups']]}
