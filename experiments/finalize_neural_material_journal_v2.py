"""Read-only execution-journal admission; outputs go to a separate fresh root."""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.scientific_reporting_adapter_v2 import _apply_v2
from infrastructure.neural_material_journal_runtime_v2 import PROTOCOL,ANCHOR,END
from infrastructure.paths import contained_path
from infrastructure.signatures import require


def read_selected(path,sha):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_file() and log.file_sha(path)==sha,'Explicit artifact hash mismatch: '+str(path))
    return log.decode(path.read_bytes())


def scan(root,anchor,journal_sha,allow_prefix):
    root=Path(root);path=root/'journal/events.jsonl'
    require(path.resolve()==path and path.is_file() and log.file_sha(path)==journal_sha,'Explicit full journal hash mismatch')
    size=path.stat().st_size;limit=anchor['journal_bytes']
    require(type(limit) is int and 0<limit<=size and (limit==size or allow_prefix),'Uncommitted tail requires explicit prefix-only diagnostic selection')
    identity=anchor['identity'];log.validate_identity(identity)
    read_selected(root/'journal/identity.json',anchor['identity_manifest_sha256'])
    manifest=log.decode((root/'journal/identity.json').read_bytes())
    require(manifest==dict(schema_version=2,identity=identity,durability='flush'),'Journal source/configuration identity differs')
    seq=0;head=log.ZERO;used=0;persist=0;pending=0;report=None
    with path.open('rb') as stream:
        while used<limit:
            line=stream.readline(min(log.MAX_RECORD_BYTES+1,limit-used+1))
            require(line.endswith(b'\n') and len(line)<=log.MAX_RECORD_BYTES and used+len(line)<=limit,'Torn/oversized selected journal record')
            r=log.decode(line);require(type(r) is dict and set(r)=={'schema_version','seq','previous_sha256','identity_sha256','kind','association','payload','sha256'},'Exact event schema required')
            require(log.canonical(r)+b'\n'==line and r['schema_version']==2 and type(r['seq']) is int and r['seq']==seq+1 and
                r['previous_sha256']==head and r['identity_sha256']==log.digest(identity) and
                log.digest({k:v for k,v in r.items() if k!='sha256'})==r['sha256'],'Changed, missing, duplicate or reordered journal event')
            kind,payload,scope=r['kind'],r['payload'],r['association']
            if kind=='initialize':
                require(seq==0 and scope is None and set(payload)=={'report'},'Invalid initialization');report=payload['report'];log._validate_rows(report)
            elif kind=='boundary':
                require(report is not None and scope is None and payload==dict(persist_id=persist+1,changes=pending,previous_seq=seq),'Invalid/missing persistence boundary')
                persist+=1;pending=0
            else:
                require(report is not None and kind in log.KINDS-{'initialize','finalize'},'Unexpected/finalized execution-journal record')
                _apply_v2(report,kind,scope,payload);pending+=len(payload['changes'])
            seq+=1;head=r['sha256'];used+=len(line)
    require(pending==0 and persist==anchor['persist_id'] and seq==anchor['seq'] and head==anchor['head_sha256'] and
        used==anchor['journal_bytes'] and report.get('status')==anchor['status'],'Selected commit anchor does not bind this exact committed prefix')
    return report,dict(records=seq,persist_boundaries=persist,committed_bytes=used,uncommitted_tail_bytes=size-used)


def checkpoint_evidence(root,report,freeze,freeze_sha):
    references=list(report.get('failure_artifacts',[]));identified=[];associated={}
    for row in report['outcomes']:
        references.extend(row.get('snapshots',{}).values())
        associated.update({id(s):row for s in row.get('snapshots',{}).values()})
        if row.get('failure_checkpoint'):references.append(row['failure_checkpoint'])
    for selection in references:
        require(type(selection) is dict and {'manifest','manifest_sha256'}<=set(selection),'Missing checkpoint identity')
        path=contained_path(root,selection['manifest'],writable=False);meta=read_selected(path,selection['manifest_sha256'])
        identified.append(dict(path=str(path),sha256=selection['manifest_sha256']))
        if 'completed_updates' in selection:
            require(meta.get('protocol')=='neural-material-recovery-v1' and meta.get('completed_updates')==selection['completed_updates'],'Invalid checkpoint protocol/counter')
            require(meta.get('execution_provenance')==dict(freeze_sha256=freeze_sha,configuration_sha256=freeze['configuration_sha256'],source_sha256=freeze['sha256'],run_id=report['run_id']),'Checkpoint execution provenance differs')
            if 'compact_state' in selection:
                require(meta['purpose']=='state','Failure-only state cannot be a completed snapshot')
                row=associated[id(selection)];model=freeze['initial_models'][row['material']+'/'+str(row['seed'])+'/'+row['arm']]
                require(meta['arm']==row['arm'] and meta.get('source')==model['source'] and meta.get('model_configuration')==model['configuration'],'Frozen checkpoint architecture/source changed')
                compact=selection['compact_state']
                require(set(compact['parameters'])==set(model['parameters']) and set(compact['buffers'])==set(model['buffers']),'Checkpoint parameter/buffer names changed')
                for name,p in compact['parameters'].items():
                    require(all(p[k]==model['parameters'][name][k] for k in ('dtype','device','shape')),'Checkpoint shape/dtype changed')
                for name,b in compact['buffers'].items():
                    require(all(b[k]==model['buffers'][name][k] for k in ('dtype','device','shape','sha256')),'Immutable calibration buffer changed')
                require(set(compact['adam'])==set(model['parameters']),'Missing parameter Adam state')
                for name,state in compact['adam'].items():
                    require(set(state)=={'step','exp_avg','exp_avg_sq'},'Missing/unknown Adam moment or counter')
                    require(state['step']['dtype']=='torch.float32' and state['step']['device']=='cpu' and state['step']['shape']==[] and
                        state['step']['sha256']==hashlib.sha256(struct.pack('<f',selection['completed_updates'])).hexdigest(),'Adam step counter identity differs')
                    for key in ('exp_avg','exp_avg_sq'):
                        require(all(state[key][k]==model['parameters'][name][k] for k in ('dtype','device','shape')),'Adam moment shape/dtype differs')
                entries={(e['role'],e['name']):e for e in meta['entries']}
                require(len(entries)==len(meta['entries']),'Duplicate tensor catalog entry')
                expected={('parameter',n) for n in model['parameters']}|{('buffer',n) for n in model['buffers']}|{('adam',n+'/'+k) for n in model['parameters'] for k in ('step','exp_avg','exp_avg_sq')}|{('torch_rng','CPU')}
                require(set(entries)==expected,'Missing/unexpected checkpoint tensor catalog entry')
                for role,key in (('parameter','parameters'),('buffer','buffers')):
                    for name,want in compact[key].items():
                        require((role,name) in entries and all(entries[role,name][k]==want[k] for k in ('dtype','device','shape','sha256')),'Snapshot tensor catalog differs')
                for name,state in compact['adam'].items():
                    for key,want in state.items():
                        require(('adam',name+'/'+key) in entries and all(entries['adam',name+'/'+key][k]==want[k] for k in ('dtype','device','shape','sha256')),'Snapshot Adam catalog differs')
                require(meta['optimizer_effective']==compact['optimizer_effective']==freeze['configuration']['optimizer_effective'],'Effective checkpoint Adam configuration changed')
        if 'archive' in meta:
            archive=contained_path(path.parent,meta['archive'],writable=False)
            sha=meta.get('archive_sha256',meta.get('sha256'))
            require(archive.is_file() and archive.resolve()==archive and log.file_sha(archive)==sha,'Missing/corrupt numeric checkpoint or failure artifact')
            if 'archive_bytes' in meta:require(archive.stat().st_size==meta['archive_bytes'],'Numeric archive size mismatch')
            identified.append(dict(path=str(archive),sha256=sha))
    return identified


def completion_checks(report,freeze):
    cfg=freeze['configuration'];expected=[(m,s,a) for m in cfg['materials'] for s in cfg['seeds'] for a in cfg['arms']]
    rows=report['outcomes'];checks={}
    checks['matrix']=len(rows)==27 and [(r['material'],r['seed'],r['arm']) for r in rows]==expected
    checks['all_complete']=all(r.get('status')=='complete' and r.get('completed_updates')==5000 for r in rows)
    checks['initial_gates']=len(report.get('initial_admission',[]))==9 and all(r.get('passed') is True and
        all(r.get('checks',{}).values()) and all(r.get('D0_N0_exact',{}).values()) and
        r.get('arithmetic',{}).get('passed') is True for r in report.get('initial_admission',[]))
    schema=freeze.get('initial_gate_schema')
    if schema is not None:
        checks['initial_gates']=checks['initial_gates'] and all(set(r.get('checks',{}))==set(schema['checks']) and
            set(r.get('D0_N0_exact',{}))==set(schema['D0_N0_exact']) and len(r.get('gradients',[]))==3 and
            all(groups and all(g.get('finite') is True and g.get('missing')==0 and g.get('nonzero',0)>0 for g in groups.values()) for groups in r.get('gradients',[])) for r in report.get('initial_admission',[]))
    else:checks['initial_gates']=False
    checks['settings']=report.get('settings')==freeze['settings'] and report.get('tapes')==freeze['tapes']
    checks['runtime_cleanup']=all(report.get(k) is True for k in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored'))
    checks['work']=checks['matrix'] and all(r.get('work')==cfg['work_per_history'][r['arm']] for r in rows)
    checks['snapshots']=all(set(r.get('snapshots',{}))=={'250','1000','5000'} and
        all(r['snapshots'][str(n)].get('completed_updates')==n for n in (250,1000,5000)) for r in rows)
    checks['initial_model_identities']=checks['matrix'] and all(r.get('initial_model')==freeze['initial_models'][r['material']+'/'+str(r['seed'])+'/'+r['arm']] for r in rows)
    checks['targets']=checks['matrix'] and all(r.get('fit_target_float32_sha256')==freeze['inputs']['targets'][r['material']]['fit_float32_sha256'] for r in rows)
    lookup={(r['material'],r['seed'],r['arm']):r for r in rows}
    controls=True
    for m in cfg['materials']:
        for seed in cfg['seeds']:
            a,b=lookup.get((m,seed,'D0')),lookup.get((m,seed,'N0'))
            if not a or not b or not a.get('prefix_reproduction_passed') or a.get('prefix_verified_through')!=1000:controls=False;continue
            if a.get('trajectory_sha256')!=b.get('trajectory_sha256') or not a.get('trajectory_sha256') or a.get('final_state')!=b.get('final_state'):controls=False
            for n in ('250','1000','5000'):
                if a.get('snapshots',{}).get(n,{}).get('compact_state')!=b.get('snapshots',{}).get(n,{}).get('compact_state'):controls=False
    checks['exact_controls']=controls
    checks['endpoint_records']=all('evaluation' in r and r.get('scientific_outcome') in ('complete_pass','complete_fail') for r in rows)
    checks['not_synthetic']=report.get('synthetic_fixture') is not True
    return checks


def atomic_json(root,name,value,limit):
    """Own fresh postprocessing output only; preserve interrupted temporaries."""
    root=Path(root);path=root/name;temporary=root/('.'+name+'.tmp')
    require(not path.exists(),'Postprocessing file collision')
    with temporary.open('x') as stream:
        class Bounded:
            bytes=0
            def write(self,text):
                n=len(text.encode('utf-8'));require(self.bytes+n<=limit,'Frozen postprocessing file cap exceeded')
                self.bytes+=n;return stream.write(text)
            def flush(self):return stream.flush()
        sink=Bounded();json.dump(value,sink,sort_keys=True,indent=2,allow_nan=False)
        require(sink.bytes+1<=limit,'Frozen postprocessing file cap exceeded')
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return path


def finalize_selected(root,anchor_sha,journal_sha,freeze,freeze_sha,output,end_sha=None,allow_prefix=False):
    root,output=Path(root),Path(output)
    require(root.is_absolute() and root.resolve()==root,'Unaliased read-only execution root required')
    anchor=read_selected(root/ANCHOR,anchor_sha)
    require(set(anchor)=={'schema_version','protocol','identity','identity_manifest_sha256','seq','head_sha256','journal_bytes','persist_id','status','authorization_sha256'} and
        anchor['schema_version']==1 and anchor['protocol']==PROTOCOL,'Invalid runtime commit anchor')
    identity=anchor['identity'];require(identity['protocol']==PROTOCOL and identity['source_sha256']==log.digest(freeze['sha256']) and
        identity['configuration_sha256']==freeze['configuration_sha256'],'Frozen journal identity mismatch')
    end=read_selected(root/END,end_sha) if end_sha is not None else None
    if end is not None:
        require(end['protocol']==PROTOCOL and end['identity']==identity and end['anchor_sha256']==anchor_sha and
            end['source_inventory_sha256']==log.digest(freeze['sha256']) and end['scientific_status']==anchor['status'],'Execution seal/anchor mismatch')
    else:require(allow_prefix,'Missing execution seal requires explicit prefix-only diagnostic mode')
    require(output.is_absolute() and output.resolve()==output and not output.exists() and output.parent.is_dir() and
        output.parent.stat().st_uid==os.geteuid(),'Fresh separate owned post-execution directory required')
    require(root!=output and root not in output.parents and output not in root.parents and not any((p/'.git').is_file() or (p/'.git/HEAD').is_file() for p in (output,*output.parents)),'Post-execution output overlaps execution/source')
    from infrastructure.neural_material_resources_v2 import ROOT,capacity
    for p in (ROOT.parent/'data',ROOT.parent/'final_results'):
        require(output!=p and p not in output.parents and output not in p.parents,'Post-execution output overlaps original inputs')
    capacity(output.parent,2*freeze['resource_budget']['materialized_report_cap_bytes'])
    report,scan_result=scan(root,anchor,journal_sha,allow_prefix)
    require(report.get('run_id')==identity['run_id'] and report.get('reviewed_freeze_sha256')==freeze_sha,'Report run/freeze identity mismatch')
    checkpoints=checkpoint_evidence(root,report,freeze,freeze_sha)
    checks=completion_checks(report,freeze)
    checks['authorization_record']=False
    if anchor['authorization_sha256'] is not None:
        review=read_selected(root/'execution-review.json',anchor['authorization_sha256'])
        checks['authorization_record']=(review.get('phase')=='D.logged-v2' and review.get('scientific_execution_authorized') is True and
            review.get('freeze_sha256')==freeze_sha and review.get('source_inventory_sha256')==identity['source_sha256'] and
            review.get('approved_output_root')==str(root) and review.get('approved_run_id')==identity['run_id'] and
            review.get('approved_output_parent')==freeze['output_policy']['approved_parent'] and
            root==Path(review.get('approved_output_parent',''))/identity['run_id'] and
            review.get('external_watchdog')==dict(wall_seconds=28800,kill_grace_seconds=10))
    scientifically_complete=(not allow_prefix and end is not None and end.get('exception_type') is None and scan_result['uncommitted_tail_bytes']==0 and
        report.get('status')=='complete' and all(checks.values()))
    # No status repair: invalid complete claims are a blocking integrity error.
    if report.get('status')=='complete' and end is not None and not allow_prefix:require(scientifically_complete,'Complete claim lacks mandatory scientific evidence')
    output.mkdir(exist_ok=False)
    path=atomic_json(output,'report.json',report,freeze['resource_budget']['materialized_report_cap_bytes'])
    final=dict(schema_version=1,protocol=PROTOCOL,freeze_sha256=freeze_sha,identity=identity,
        selected_anchor_sha256=anchor_sha,selected_journal_sha256=journal_sha,selected_execution_end_sha256=end_sha,
        source_inventory_sha256=log.digest(freeze['sha256']),scan=scan_result,original_scientific_status=report['status'],
        classification='complete_scientific_trajectory' if scientifically_complete else 'sealed_negative_outcome' if end is not None else 'valid_committed_prefix_only',
        scientifically_complete=scientifically_complete,recovery_success_not_implied=True,completion_checks=checks,
        materialized_report_sha256=log.file_sha(path),materialized_canonical_sha256=log.digest(report),checkpoint_artifacts=checkpoints)
    atomic_json(output,'final.json',final,freeze['resource_budget']['control_metadata_bytes'])
    return dict(final_path=str(output/'final.json'),final_sha256=log.file_sha(output/'final.json'),**final)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','anchor-sha256','journal-sha256','freeze','freeze-sha256','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--execution-end-sha256');p.add_argument('--prefix-only',action='store_true');a=p.parse_args()
    from experiments.freeze_neural_material_recovery_logged_v2 import validate_freeze
    freeze=validate_freeze(Path(a.freeze),a.freeze_sha256)
    result=finalize_selected(a.run_root,a.anchor_sha256,a.journal_sha256,freeze,a.freeze_sha256,a.output,a.execution_end_sha256,a.prefix_only)
    print(json.dumps(dict(final_path=result['final_path'],final_sha256=result['final_sha256'],classification=result['classification'],scientifically_complete=result['scientifically_complete']),sort_keys=True))


if __name__=='__main__':main()
