"""Strict checkpoint/optimizer diagnostics; Torch is injected, never imported here.

PyTorch 1.8 torch.load deserializes trusted pickle, including optimizer payload.
The native caller must establish isolation and exact artifact identity first.
"""
from collections import OrderedDict
import hashlib
import math
import numbers
import os
from pathlib import Path
import stat

from infrastructure.signatures import CHECKPOINT_SHA256, require

STORED_STEP = 150000
STORED_BEST = 23.787122565451323


def mapping(value):
    return type(value) in (dict, OrderedDict)


def hash_stream(stream):
    stream.seek(0); h = hashlib.sha256()
    for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    stream.seek(0)
    return h.hexdigest()


def load_checked(path, expected_sha256, torch, trusted=False):
    """Only the CLI/native wrapper supplies the fixed historical SHA; tests use fixtures."""
    require(trusted is True, 'Explicit trusted-checkpoint acknowledgement required before pickle deserialization')
    path = Path(path)
    require(path.resolve() == path and path.is_file(), 'Checkpoint path is missing or aliased')
    with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and hash_stream(stream) == expected_sha256, 'Checkpoint SHA-256 mismatch')
        value = torch.load(stream, map_location='cpu')  # Full pickle container. No weights_only API in Torch 1.8.
        after = os.fstat(stream.fileno())
        require(hash_stream(stream) == expected_sha256 and
                (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), 'Checkpoint changed during loading')
    return value


def historical_schema(state):
    require(mapping(state) and all(type(k) is str for k in state), 'Checkpoint must be a string-keyed mapping')
    require(type(state.get('step')) is int and state['step'] == STORED_STEP, 'Unexpected stored checkpoint step')
    require(isinstance(state.get('best_para'), numbers.Real) and not isinstance(state['best_para'],bool) and
            math.isfinite(float(state['best_para'])) and float(state['best_para']) == STORED_BEST, 'Stored best metric differs from admission')
    require(mapping(state.get('network_state_dict')) and mapping(state.get('optimizer_state_dict')), 'Missing original G-format state mappings')
    return {'top_level_keys': sorted(state), 'top_level_types': {k:type(v).__module__+'.'+type(v).__name__ for k,v in state.items()},
            'stored_step': state['step'], 'stored_best_para': float(state['best_para']),
            'limit': 'Container schema/structural compatibility do not identify producing source or resume eligibility.'}


def tensor_catalog(values, torch):
    require(mapping(values), 'Network state must be a mapping')
    result = {}
    for name, value in values.items():
        require(type(name) is str and torch.is_tensor(value), 'Invalid network tensor/name')
        require(bool(torch.isfinite(value.detach()).all().item()), 'Nonfinite network tensor: '+name)
        result[name] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
    return result


def network_comparison(model, saved, torch):
    expected = tensor_catalog(model.state_dict(), torch)
    actual = tensor_catalog(saved, torch)
    common = expected.keys() & actual.keys()
    missing, unexpected = sorted(expected.keys()-actual.keys()), sorted(actual.keys()-expected.keys())
    mismatches = {key:[{'name':n,'checkpoint':actual[n][key],'G':expected[n][key]} for n in sorted(common)
                      if actual[n][key]!=expected[n][key]] for key in ('shape','dtype')}
    return {'result':'exact_structural_match' if not missing and not unexpected and not any(mismatches.values()) else 'network_incompatible',
            'missing_keys':missing,'unexpected_keys':unexpected,'shape_mismatches':mismatches['shape'],
            'dtype_mismatches':mismatches['dtype'],'G_network_catalog':expected,'checkpoint_network_catalog':actual,
            'G_tensor_count':len(expected),'checkpoint_tensor_count':len(actual)}


def identities(values):
    return {n:{'shape':list(v.shape),'dtype':str(v.dtype),
               'sha256':hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()} for n,v in values.items()}


def strict_network_load(model, saved, torch):
    report = network_comparison(model, saved, torch)
    require(report['result']=='exact_structural_match','G network state keys/shapes/dtypes incompatible')
    model.load_state_dict(saved, strict=True)
    require(identities(model.state_dict())==identities(saved),'Strict load did not preserve selected checkpoint tensor bytes')
    return report


def group_settings(groups):
    def primitive(v):
        if v is None or type(v) in (str,bool,int,float): return v
        if isinstance(v,numbers.Integral):return int(v)
        if isinstance(v,numbers.Real):return float(v)
        if type(v) in (tuple,list): return [primitive(x) for x in v]
        return {'unreported_type':type(v).__module__+'.'+type(v).__name__}
    return [{k:primitive(v) for k,v in group.items() if k!='params'} for group in groups]


def optimizer_preflight(optimizer, saved, model, torch):
    require(mapping(saved) and set(saved)=={'state','param_groups'} and mapping(saved['state']) and
            type(saved['param_groups']) is list, 'Invalid optimizer state container')
    groups = saved['param_groups']; live = optimizer.param_groups
    require(len(groups)==len(live),'Optimizer parameter-group count mismatch; no regrouping permitted')
    names = {id(p):n for n,p in model.named_parameters()}
    mapping_rows=[]; known=set()
    for saved_group, live_group in zip(groups,live):
        require(mapping(saved_group) and type(saved_group.get('params')) is list and
                len(saved_group['params'])==len(live_group['params']),'Optimizer parameter count/order cannot be mapped')
        require({'lr','betas','eps','weight_decay','amsgrad'}<=saved_group.keys(),'Missing optimizer hyperparameter fields')
        require(type(saved_group['amsgrad']) is bool and type(saved_group['betas']) in (list,tuple) and
                len(saved_group['betas'])==2 and all(isinstance(b,numbers.Real) and not isinstance(b,bool) and 0<=b<1 for b in saved_group['betas']),
                'Invalid Adam betas/AMSGrad values')
        require(all(isinstance(saved_group[k],numbers.Real) and not isinstance(saved_group[k],bool) and
                    math.isfinite(float(saved_group[k])) and saved_group[k]>=0 for k in ['lr','eps','weight_decay']),
                'Invalid optimizer scalar settings')
        for key,p in zip(saved_group['params'],live_group['params']):
            require(type(key) is int and key>=0 and key not in known and id(p) in names,'Invalid/duplicate optimizer parameter identity')
            known.add(key)
            mapping_rows.append({'saved_id':key,'G_parameter':names[id(p)],'shape':list(p.shape),'dtype':str(p.dtype)})
            if key not in saved['state']: continue  # Inactive parameters need not have Adam state.
            state=saved['state'][key]
            fields={'step','exp_avg','exp_avg_sq'} | ({'max_exp_avg_sq'} if saved_group.get('amsgrad',False) else set())
            require(mapping(state) and set(state)==fields,'Missing/extra/incompatible Adam state fields')
            step=state['step'].item() if torch.is_tensor(state['step']) else state['step']
            require(isinstance(step,numbers.Real) and not isinstance(step,bool) and math.isfinite(float(step)) and int(step)==step and step>=0,
                    'Invalid per-parameter Adam counter')
            for name in fields-{'step'}:
                tensor=state[name]
                require(torch.is_tensor(tensor) and tensor.shape==p.shape and tensor.dtype==p.dtype and
                        bool(torch.isfinite(tensor).all().item()),'Incompatible optimizer tensor: '+names[id(p)]+':'+name)
    require(set(saved['state'])<=known,'Optimizer state contains unknown parameter IDs')
    return mapping_rows


def inspect_optimizer_load(optimizer, saved, model, torch):
    rows=optimizer_preflight(optimizer,saved,model,torch)
    before=group_settings(optimizer.param_groups); network_before=identities(model.state_dict())
    optimizer.load_state_dict(saved)  # Diagnostic object only. Never call step or resume Trainer.run.
    after=group_settings(optimizer.param_groups)
    require(identities(model.state_dict())==network_before,'Optimizer state loading mutated network tensors')
    differences=[{k:{'G_fresh':a.get(k),'loaded':b.get(k)} for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
                 for a,b in zip(before,after)]
    return {'load_state_dict_succeeded':True,'G_fresh_group_settings':before,'loaded_group_settings':after,
            'loaded_group_scalar_types':[{k:type(v).__module__+'.'+type(v).__name__ for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
            'scientific_hyperparameter_changes':[{k:v for k,v in d.items() if k!='lr'} for d in differences],
            'progress_lr_changes':[d.get('lr') for d in differences], 'positional_parameter_mapping':rows,
            'network_unchanged':True,'optimizer_updates':0,'training_resumed':False,'G_continuation_admitted':False,
            'limits':['A successful library load is not semantic/provenance compatibility.',
                      'Loading may replace G eps/betas/AMSGrad/other settings; no resetting or repair is applied.',
                      'Saved IDs have no recovered names: mapping to G parameter order is positional only.',
                      'Stored LR/counters reflect progress, not necessarily a different optimizer algorithm.']}


def manual_roundtrip(donor, receiver, original_trainer_class, checkpoint, step, best, torch, np):
    """Owned G-produced state only. Original format/API; no optimizer calls here."""
    from infrastructure.training_probe import optimizer_record
    checkpoint=Path(checkpoint)
    require(not checkpoint.exists() and not checkpoint.is_symlink(),'Manual checkpoint collision')
    source_weights=identities(donor.network.state_dict())
    source_optimizer=optimizer_record(donor.optimizer,donor.network,np)
    receiver_before=identities(receiver.network.state_dict())
    require(receiver_before!=source_weights,'Fresh receiver must differ so roundtrip cannot pass as a no-op')
    original_trainer_class._save_model(donor,step,best,save_fn=str(checkpoint))
    require(checkpoint.is_file() and checkpoint.resolve()==checkpoint and checkpoint.stat().st_nlink==1,'Unsafe owned checkpoint file')
    from infrastructure.stage_sources import digest
    checksum=digest(checkpoint)
    # G's original method defaults to strict=True and restores optimizer ONLY in this controlled roundtrip.
    receiver.pth_fn=str(checkpoint)
    loaded_best,loaded_step=original_trainer_class._load_model(receiver)
    require(loaded_best==best and loaded_step==step,'Roundtrip step/best mismatch')
    require(identities(receiver.network.state_dict())==source_weights and
            optimizer_record(receiver.optimizer,receiver.network,np)==source_optimizer,'Roundtrip network/optimizer mismatch')
    require(digest(checkpoint)==checksum and identities(donor.network.state_dict())==source_weights and
            optimizer_record(donor.optimizer,donor.network,np)==source_optimizer,'Manual source state/file changed')
    return {'kind':'manually-triggered-G-serialization','checkpoint_sha256':checksum,'stored_step':step,'stored_best':best,
            'strict_network_roundtrip':True,'optimizer_roundtrip':True,'optimizer_state_entries':len(source_optimizer['state']),
            'receiver_initially_different':True,'post_load_optimizer_updates':0,'natural_schedule_tested':False,
            'research_geometry_admitted':False,'source_network_identity':source_weights}
