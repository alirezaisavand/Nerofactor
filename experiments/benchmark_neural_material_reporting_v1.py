"""D.1 saved-data audit and bounded synthetic reporting benchmark, never recovery."""
import argparse
import ast
import copy
import importlib.abc
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from infrastructure import append_only_scientific_log_v1 as journal

ROOT=Path(__file__).resolve().parents[1]
BASE='de5277686653033d5655f71bcaa3d70627c12945'
RAW_SHA='4104e5e178eb41331dd29983648b28a8ef8d5c70a88d487dc12063ff4ab7de49'
RESULT_SHA='1f7f0d2cb9d41104eca447dc09a8f66016b7fd9e87354881763ba410526dac7e'
FREEZE_SHA='730a88458148d92c34fe4429c7f3cd712b2962b1fb64f56a09b42993b97871ad'
CASES=('early','middle','censor_scale','planned_27_synthetic')
PERSIST_REPETITIONS=3
WALL_CAP=180
CODE=('experiments/benchmark_neural_material_reporting_v1.py','infrastructure/append_only_scientific_log_v1.py',
      'tests/scientific/test_append_only_scientific_log_v1.py','docs/protocols/NEURAL_MATERIAL_REPORTING_OPTIMIZATION_V1_DRAFT.md')


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        forbidden=('experiments.material_shader','experiments.neural_material_field','experiments.neural_material_recovery',
            'experiments.run_known_material_recovery','experiments.multi_material_recovery','scientific_sources')
        if fullname.startswith(forbidden) or fullname=='network' or fullname.startswith('network.'):
            raise AssertionError('Forbidden scientific import: '+fullname)


def read_selected(path,sha):
    path=Path(path)
    journal.require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink(),'Unaliased explicit input required')
    journal.require(path.stat().st_size<128*1024**2 and journal.file_sha(path)==sha,'Input hash/size mismatch before parsing')
    return journal.decode(path.read_bytes())


def identity():
    return dict(protocol='synthetic-reporting-audit-v1',run_id='synthetic-benchmark',source_sha256=journal.file_sha(ROOT/CODE[0]),
                configuration_sha256=journal.digest(dict(cases=list(CASES),repetitions=PERSIST_REPETITIONS,durability='flush')))


def timed(call):
    begin,cpu=time.perf_counter(),time.process_time();value=call()
    return value,dict(wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu)


def admit(raw_path):
    raw=read_selected(raw_path,RAW_SHA)
    result=read_selected(ROOT/'docs/evidence/NEURAL_MATERIAL_FIELD_RECOVERY_V1.json',RESULT_SHA)
    freeze=read_selected(ROOT/'docs/protocols/NEURAL_MATERIAL_FIELD_RECOVERY_V1.freeze.json',FREEZE_SHA)
    for name,sha in freeze['sha256'].items():journal.require(journal.file_sha(ROOT/name)==sha,'Frozen source mismatch: '+name)
    for meta in result['control_artifacts'].values():journal.require(journal.file_sha(meta['path'])==meta['sha256'],'Original control artifact changed')
    for name,meta in result['output_inventory'].items():journal.require(journal.file_sha(Path(raw_path).parent/name)==meta['sha256'],'Original output changed')
    journal.require(raw['status']=='failed' and result['process']['invocations']==1 and raw['repository']['HEAD']=='ffed2b075b997e7b6faf37840fdec7e1c489d8ef','Original execution identity changed')
    return raw,result,freeze


def runtime_audit(raw):
    order=sorted((r for r in raw['outcomes'] if r['completed_updates']),key=lambda r:(('D0','N0','N1').index(r['arm']),r['material'],r['seed']))
    accumulated=0;histories=[]
    for ordinal,row in enumerate(order):
        entries=row['history'];points=[(e['update'],e['active_wall_seconds']) for e in entries]
        journal.require(all(b[0]>a[0] and b[1]>a[1] for a,b in zip(points,points[1:])),'Nonmonotonic saved timing evidence')
        intervals=[]
        for (u0,t0),(u1,t1) in zip(points,points[1:]):
            intervals.append(dict(start_update=u0,end_update=u1,updates=u1-u0,seconds=t1-t0,seconds_per_update=(t1-t0)/(u1-u0),
                includes_after_checkpoint=u0 in (250,1000),prefix_extra_forward=u0==1000))
        windows={}
        for label,lo,hi in (('early',1,1000),('middle',2000,3000),('late',4000,row['completed_updates'])):
            selected=[(u,t) for u,t in points if lo<=u<=hi]
            if len(selected)>1:
                a,b=selected[0],selected[-1];windows[label]=dict(start_update=a[0],end_update=b[0],seconds=b[1]-a[1],
                    seconds_per_update=(b[1]-a[1])/(b[0]-a[0]),updates_per_second=(b[0]-a[0])/(b[1]-a[1]),
                    seconds_per_100_updates=100*(b[1]-a[1])/(b[0]-a[0]))
        chunks=[]
        for stop in range(100,row['completed_updates']+1,100):
            selected=[(u,t) for u,t in points if max(1,stop-100)<=u<=stop]
            if len(selected)>1:
                a,b=selected[0],selected[-1];chunks.append([a[0],b[0],b[1]-a[1]])
        history=dict(ordinal=ordinal,material=row['material'],seed=row['seed'],arm=row['arm'],status=row['status'],
            completed_updates=row['completed_updates'],active_wall_seconds=row['active_wall_seconds'],CPU_seconds=row['CPU_seconds'],
            logged_updates=len(entries),accumulated_prior_log_records=accumulated,snapshot_count=len(row['snapshots']),work=row['work'],
            failure=row.get('failure'),termination=row['termination'],timing_points_columns=['completed_update','observed_active_wall_seconds'],
            timing_points=points,windows=windows,hundred_update_windows=chunks,
            checkpoint_intervals=[i for i in intervals if i['includes_after_checkpoint']],
            other_ten_update_interval_median_seconds=statistics.median(i['seconds'] for i in intervals if not i['includes_after_checkpoint']),
            observed_final_row_pretty_bytes=len(json.dumps(row,indent=2,allow_nan=False).encode()),
            late_over_early_seconds_per_update=windows['late']['seconds_per_update']/windows['early']['seconds_per_update'])
        histories.append(history);accumulated+=len(entries)
    complete=[h for h in histories if h['status']=='complete'];x=np.array([h['accumulated_prior_log_records'] for h in complete],float)
    y=np.array([h['active_wall_seconds']/5000 for h in complete],float)
    slope,intercept=np.polyfit(x,y,1)
    return dict(histories=histories,observed_final_report_bytes=len(json.dumps(raw,indent=2,allow_nan=False).encode())+1,
        association=dict(Pearson_prior_log_records_vs_seconds_per_update=float(np.corrcoef(x,y)[0,1]),
            least_squares_seconds_per_update_intercept=float(intercept),seconds_per_update_per_prior_record=float(slope),
            causal=False,limitations='Order/material/arm/report growth are confounded; no original persist duration or historical file sizes were recorded'),
        first_to_last_complete_active_time_ratio=complete[-1]['active_wall_seconds']/complete[0]['active_wall_seconds'],
        persistence_timing_scope='Logged timestamp is after update bookkeeping but before persist/checkpoint; intervals include prior persistence and possible checkpoint/prefix work',
        measured_persistence_durations_available=False,exact_prior_persisted_report_sizes_available=False)


def synthetic_record(template,update,arm,n1_groups):
    item=copy.deepcopy(template);item.update(update=update,loss_at_update=update-1,loss=.0125,gradient_norm=.25,active_wall_seconds=update*.05)
    item['physical_before_update']=[.2,.5,-.3,.1,.2,.3,.4]
    if arm=='N1':item['gradient_groups']=copy.deepcopy(n1_groups)
    item['work']=dict(field_attempts=update+3,field_completed=update+3,calibration_forwards=int(arm=='N1'),shader_attempts=update+1,
        shader_completed=update+1,fit_forwards=update+1,evaluation_forwards=0,attempted_points=(update+1)*96,completed_points=(update+1)*96,
        diffuse_directions=(update+1)*96*512,specular_directions=(update+1)*96*1024,reverse_attempts=update+1,reverse_completed=update+1,
        optimizer_attempts=update,optimizer_returns=update,initial_forwards=1,prefix_forwards=0,terminal_forwards=0)
    return item


def fixture(raw,case):
    report=copy.deepcopy(raw);report.update(synthetic=True,benchmark_case=case,status='running',run_id='synthetic-benchmark')
    for key in ('failure','failure_type','failure_category','active_history'):report.pop(key,None)
    rows=sorted(report['outcomes'],key=lambda r:(('D0','N0','N1').index(r['arm']),r['material'],r['seed']))
    template=raw['outcomes'][0]['history'][0]
    n1_groups=copy.deepcopy(raw['initial_admission'][0]['gradients'][2])
    for group in n1_groups.values():group.update(missing=0,finite=True,nonzero=group['parameters'],squared_norm=.01)
    active,prior,end={'early':(0,0,1000),'middle':(13,13,1000),'censor_scale':(15,15,4860),'planned_27_synthetic':(26,26,4970)}[case]
    for i,row in enumerate(rows):
        count=5000 if i<prior else end if i==active else 0
        row['history']=[synthetic_record(template,u,row['arm'],n1_groups) for u in ([1,*range(10,count+1,10)] if count else [])]
        row.update(status='complete' if i<prior else 'optimizing' if i==active else 'admitted',completed_updates=count,
            work=copy.deepcopy(row['history'][-1]['work']) if count else dict(template['work']))
        # Preserve realistic field/identity/snapshot payload SHAPES, not scientific values or claims.
        compact=copy.deepcopy(raw['outcomes'][0]['snapshots']['250']['compact_state'])
        if row['arm']=='N1':
            params={n:{k:m[k] for k in ('dtype','device','shape','sha256')} for n,m in row['initial_model']['parameters'].items()}
            compact=dict(parameters=params,buffers=row['initial_model']['buffers'],optimizer_effective=compact['optimizer_effective'],
                adam={n:dict(step=dict(dtype='torch.float32',device='cpu',shape=[],sha256='f'*64),exp_avg=p,exp_avg_sq=p) for n,p in params.items()})
        row['snapshots']={str(u):dict(completed_updates=u,manifest='synthetic/u'+str(u)+'.json',manifest_sha256='d'*64,
            compact_state=copy.deepcopy(compact)) for u in (250,1000,5000) if u<=count}
        for key in ('evaluation','final_state','final_gradient_groups','failure_checkpoint','termination','failure','scientific_outcome'):row.pop(key,None)
    report['outcomes']=rows
    events=[]
    for delta in (10,20,30):
        record=synthetic_record(template,end+delta,rows[active]['arm'],n1_groups)
        events.append(dict(kind='history',association=journal.association(rows[active]),changes=[journal.change('append',['outcomes',active,'history'],record),
            journal.change('set',['outcomes',active,'completed_updates'],end+delta),journal.change('set',['outcomes',active,'work'],record['work'])]))
    return dict(synthetic=True,case=case,source_report_sha256=RAW_SHA,report=report,events=events,
        expected_prefix_log_records=sum(len(r['history']) for r in rows),N1_payload='synthetic metadata from saved initial gradient-group schema; no N1 training measurement')


def persist_r0(stream,report):
    begin,cpu=time.perf_counter(),time.process_time()
    payload=json.dumps(report,indent=2,allow_nan=False)+'\n'
    encoded=time.perf_counter();encoded_cpu=time.process_time()
    stream.seek(0);stream.write(payload);stream.truncate();stream.flush()
    finished,finished_cpu=time.perf_counter(),time.process_time()
    return dict(encoding_wall_seconds=encoded-begin,writing_flush_wall_seconds=finished-encoded,
        total_wall_seconds=finished-begin,encoding_CPU_seconds=encoded_cpu-cpu,writing_flush_CPU_seconds=finished_cpu-encoded_cpu,
        logical_bytes_written=len(payload.encode()),fsync=False)


def seed_journal(w,desired):
    for i,row in enumerate(desired['outcomes']):
        scope=journal.association(row)
        for record in row['history']:
            w.append('history',[journal.change('append',['outcomes',i,'history'],record)],scope)
        for step,snapshot in row['snapshots'].items():w.append('checkpoint',[journal.change('set',['outcomes',i,'snapshots',step],snapshot)],scope)
        fields=[journal.change('set',['outcomes',i,k],v) for k,v in row.items() if k not in ('material','seed','arm','history','snapshots','status')]
        if fields:w.append('metadata',fields,scope)
        kind='censor' if row['status']=='censored' else 'failure' if row['status'].startswith('failed') else 'status'
        w.append(kind,[journal.change('set',['outcomes',i,'status'],row['status'])],scope)


def reporting_worker(fixture_path,fixture_sha,output,arm):
    item=read_selected(fixture_path,fixture_sha);report=item['report'];events=item['events'];output=Path(output);output.mkdir(exist_ok=False)
    before=time.perf_counter();result=dict(arm=arm,case=item['case'],fixture_sha256=fixture_sha,synthetic=True,events=[],durability='flush_no_fsync',
        prefix_log_records=item['expected_prefix_log_records'],window_events=len(events),setup_scope='Prefix is bootstrapped once; not a replay of all historical R0 rewrites')
    initial_bytes=len(json.dumps(report,indent=2,allow_nan=False).encode())+1
    result['initial_pretty_report_bytes']=initial_bytes
    if arm=='R0':
        with (output/'report.json').open('x') as stream:
            result['bootstrap']=persist_r0(stream,report)
            for event in events:
                journal.apply_event(report,event['kind'],event['association'],dict(changes=event['changes']))
                result['events'].append(persist_r0(stream,report))
            report['status']='incomplete';result['final_persistence']=persist_r0(stream,report)
        result['reconstruction_seconds']=0.;result['hash_verification_seconds']=timed(lambda:journal.file_sha(output/'report.json'))[1]['wall_seconds']
        result['window_wall_seconds']=sum(x['total_wall_seconds'] for x in result['events'])
        result['window_CPU_seconds']=sum(x['encoding_CPU_seconds']+x['writing_flush_CPU_seconds'] for x in result['events'])
        result['window_logical_bytes_written']=sum(x['logical_bytes_written'] for x in result['events'])
        result['window_data_writes']=len(events);result['window_summary_updates']=0
    else:
        skeleton=copy.deepcopy(report)
        for row in skeleton['outcomes']:row.update(history=[],snapshots={},status='admitted',completed_updates=0)
        start=time.perf_counter()
        w=journal.ScientificLog(output/'journal',identity(),skeleton,approved_parent=output,
            preserved_roots=[ROOT,ROOT.parent/'data',ROOT.parent/'final_results'],durability='flush')
        seed_journal(w,report);w.update_summary()
        result['bootstrap_seconds']=time.perf_counter()-start;result['bootstrap_journal_events']=w.seq
        result['bootstrap_journal_bytes']=w.bytes
        journal.require(journal.canonical(w.report)==journal.canonical(report),'Synthetic prefix seed lost data')
        for event in events:
            start,cpu=time.perf_counter(),time.process_time();old_bytes=w.bytes
            w.append(event['kind'],event['changes'],event['association']);appended=time.perf_counter();summary=w.update_summary()
            end=time.perf_counter();end_cpu=time.process_time()
            result['events'].append(dict(append_validate_encode_write_wall_seconds=appended-start,summary_wall_seconds=end-appended,
                total_wall_seconds=end-start,CPU_seconds=end_cpu-cpu,logical_bytes_written=w.bytes-old_bytes+(output/'journal/summary.json').stat().st_size))
        w.append('status',[journal.change('set',['status'],'incomplete')]);final,measurement=timed(w.finalize);w.close()
        result['finalization_seconds']=measurement['wall_seconds'];result['trusted_final_manifest']=final
        verified,measurement=timed(lambda:journal.verify(output/'journal',identity(),expected_final_sha256=final['sha256']))
        result['verification_and_reconstruction_seconds']=measurement['wall_seconds']
        # Combined chain verification/reconstruction is reported honestly; an independent hash scan is separate.
        result['hash_verification_seconds']=timed(lambda:journal.file_sha(output/'journal/events.jsonl'))[1]['wall_seconds']
        result['reconstruction_seconds']=None
        def materialize():
            with (output/'report.json').open('x') as stream:json.dump(verified['report'],stream,indent=2,allow_nan=False);stream.write('\n')
        _,measurement=timed(materialize);result['materialization_seconds']=measurement['wall_seconds']
        expected=copy.deepcopy(report)
        for event in events:journal.apply_event(expected,event['kind'],event['association'],dict(changes=copy.deepcopy(event['changes'])))
        expected['status']='incomplete'
        journal.require(journal.canonical(verified['report'])==journal.canonical(expected),'R1 final report is not semantically exact')
        result['semantic_equality']=True;result['summary_bytes']=(output/'journal/summary.json').stat().st_size
        result['window_wall_seconds']=sum(x['total_wall_seconds'] for x in result['events'])
        result['window_CPU_seconds']=sum(x['CPU_seconds'] for x in result['events'])
        result['window_logical_bytes_written']=sum(x['logical_bytes_written'] for x in result['events'])
        result['window_data_writes']=len(events);result['window_summary_updates']=len(events)
    result.update(per_persist_seconds=result['window_wall_seconds']/len(events),final_canonical_report_sha256=journal.digest(journal.decode((output/'report.json').read_bytes())),
        final_artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        wall_seconds=time.perf_counter()-before,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        memory_scope='Fresh isolated worker high-water RSS, includes identical fixture loading; not incremental allocation or physical disk traffic')
    with (output/'measurement.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return result


def synthetic_observer_costs(raw,output):
    """Exact isolated observer definitions on inert tensors, never a model or Adam."""
    import torch
    from unittest import mock
    tree=ast.parse((ROOT/'experiments/neural_material_field_v1.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('record','gradient_summary')]
    ns={'torch':torch,'hashlib':__import__('hashlib')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'admitted-observer-only-definitions','exec'),ns)
    result={};options=dict(lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0.)
    with mock.patch.object(torch.optim,'Adam',side_effect=AssertionError('No optimizer construction')),\
         mock.patch.object(torch.autograd,'backward',side_effect=AssertionError('No backward')):
        for arm in ('D0','N1'):
            source=next(r for r in raw['outcomes'] if r['arm']==arm)['initial_model']['parameters']
            params=[]
            for name,meta in source.items():
                p=torch.full(meta['shape'],.125,dtype=torch.float32);p.grad=torch.full_like(p,.25);params.append((name,p))
            fake=SimpleNamespace(named_parameters=lambda:iter(params));groups=ns['gradient_summary'](fake)
            def repeats(fn,n):
                _,cost=timed(lambda:[fn() for _ in range(n)]);return dict(repetitions=n,per_call_wall_seconds=cost['wall_seconds']/n,**cost)
            result[arm]=dict(parameters=sum(p.numel() for _,p in params),tensor_count=len(params),
                gradient_summary=repeats(lambda:ns['gradient_summary'](fake),20),
                parameter_moment_hash_proxy=repeats(lambda:[ns['record'](p) for _,p in params for _ in range(3)],20),
                groups=list(groups),scope='Synthetic constant CPU tensors, no shader/model/gradient computation; N1 state hashing is checkpoint-only in the frozen harness')
            if arm=='D0':
                htree=ast.parse((ROOT/'experiments/neural_material_recovery_v1.py').read_text())
                helpers=[n for n in htree.body if isinstance(n,ast.FunctionDef) and n.name in ('compact_state','vector_transition_digest')]
                helper_ns=dict(field=SimpleNamespace(record=ns['record']),previous=SimpleNamespace(lm=SimpleNamespace(accepted=SimpleNamespace(
                    optimizer_options=lambda optimizer:options))),require=journal.require,json=json,hashlib=__import__('hashlib'))
                exec(compile(ast.fix_missing_locations(ast.Module(body=helpers,type_ignores=[])),'admitted-observer-only-state-hash','exec'),helper_ns)
                fake.arm='D0';fake.latent=params[0][1];fake.named_buffers=lambda:iter(())
                optimizer=SimpleNamespace(state={fake.latent:dict(step=torch.tensor(1.,dtype=torch.float32),exp_avg=torch.ones(7),exp_avg_sq=torch.ones(7))})
                loss=torch.tensor(.01)
                result[arm]['exact_observer_transition_digest']=repeats(lambda:helper_ns['vector_transition_digest']('0'*64,fake,optimizer,loss,1),100)
                result[arm]['state_hash_helper_AST_sha256']={n.name:journal.digest(ast.dump(n)) for n in helpers}
            arrays={str(i):p.numpy() for i,(_,p) in enumerate(params)}
            for i,(_,p) in enumerate(params):
                arrays['first'+str(i)]=np.full(p.shape,.001,np.float32);arrays['second'+str(i)]=np.full(p.shape,.002,np.float32)
            path=Path(output)/('synthetic-'+arm+'-state.npz')
            def save():
                with path.open('xb') as stream:np.savez(stream,**arrays)
            _,cost=timed(save);result[arm]['numeric_snapshot_write']=dict(bytes=path.stat().st_size,sha256=journal.file_sha(path),**cost)
        _,cost=timed(lambda:[resource.getrusage(resource.RUSAGE_SELF).ru_maxrss for _ in range(1000)])
        result['resource_check']=dict(repetitions=1000,per_call_wall_seconds=cost['wall_seconds']/1000,**cost)
        result['scientific_calls']=result['backwards']=result['optimizer_steps']=0
    return result


def run(raw_path,output):
    begin=time.perf_counter();cpu=time.process_time();raw,result,freeze=admit(raw_path);output=Path(output)
    journal.require(output.is_absolute() and output.resolve()==output and not output.exists() and output.parent.stat().st_uid==os.geteuid(),'Fresh owned benchmark output required')
    for preserved in (ROOT,ROOT.parent/'data',ROOT.parent/'final_results',Path(raw_path).parent,Path(result['process']['control'])):
        journal.require(output!=preserved and preserved not in output.parents and output not in preserved.parents,'Benchmark output overlaps preserved inputs')
    output.mkdir(exist_ok=False)
    for name in CODE:
        copied=output/'source-epoch'/name;copied.parent.mkdir(parents=True,exist_ok=True)
        with copied.open('xb') as stream:stream.write((ROOT/name).read_bytes())
    audit=dict(schema_version=1,protocol='neural-material-reporting-audit-v1',status='running',baseline=BASE,source_report=dict(path=str(raw_path),sha256=RAW_SHA),
        original_results_sha256=RESULT_SHA,original_freeze_sha256=FREEZE_SHA,scientific_work=0,benchmarks=[],
        source_sha256={n:journal.file_sha(ROOT/n) for n in CODE},runtime=dict(python=platform.python_version(),numpy=np.__version__,threads='one via explicit environment',
            filesystem_device=output.stat().st_dev),output_root=str(output),benchmark_plan=dict(wall_cap_seconds=WALL_CAP,cases=CASES,repetitions=PERSIST_REPETITIONS,
            original_persistence_every_updates=10,durability='flush',full_planned_scale='synthetic projection, not scientific execution'))
    def persist():
        # This bounded audit summary is written after each CASE, never scientific training.
        path=output/'audit.partial.json';path.write_text(json.dumps(audit,sort_keys=True,indent=2,allow_nan=False)+'\n')
    def alarm(signum,frame):raise TimeoutError('Offline benchmark wall cap; preserve partial artifacts')
    previous=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,WALL_CAP-(time.perf_counter()-begin)))
    try:
        audit['saved_runtime_analysis']=runtime_audit(raw);persist()
        with (output/'original-report-copy.json').open('x') as stream:
            audit['original_final_report_copy_R0']=[persist_r0(stream,raw) for _ in range(PERSIST_REPETITIONS)]
        journal.require(journal.file_sha(output/'original-report-copy.json')==RAW_SHA,'Exact original persistence copy differs')
        audit['original_final_report_copy_sha256']=RAW_SHA
        for case in CASES:
            item=fixture(raw,case);path=output/(case+'-synthetic-fixture.json')
            with path.open('x') as stream:json.dump(item,stream,allow_nan=False);stream.write('\n')
            sha=journal.file_sha(path);pair={}
            for arm in ('R0','R1'):
                destination=output/(case+'-'+arm)
                command=['python3','-B','-m','experiments.benchmark_neural_material_reporting_v1','--worker',arm,'--fixture',str(path),
                    '--fixture-sha256',sha,'--output',str(destination)]
                with (output/(case+'-'+arm+'.console.log')).open('x') as console:
                    completed=subprocess.run(command,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT,timeout=max(1,WALL_CAP-(time.perf_counter()-begin)))
                journal.require(completed.returncode==0,'Reporting worker failed; preserve console and do not invent a measurement')
                pair[arm]=journal.decode((destination/'measurement.json').read_bytes())
            journal.require(pair['R0']['final_canonical_report_sha256']==pair['R1']['final_canonical_report_sha256'],'R0/R1 final semantic mismatch')
            audit['benchmarks'].append(dict(case=case,fixture=dict(path=str(path),sha256=sha,bytes=path.stat().st_size),R0=pair['R0'],R1=pair['R1'],
                steady_state_speedup=pair['R0']['window_wall_seconds']/pair['R1']['window_wall_seconds'],
                logical_window_write_reduction=pair['R0']['window_logical_bytes_written']/pair['R1']['window_logical_bytes_written'],semantic_equality=True))
            persist();print(json.dumps(dict(case=case,status='verified',steady_state_speedup=audit['benchmarks'][-1]['steady_state_speedup'])),flush=True)
            journal.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'Offline parent RSS limit')
        audit['observer_costs']=synthetic_observer_costs(raw,output)
        for name,sha in audit['source_sha256'].items():journal.require(journal.file_sha(ROOT/name)==sha,'Benchmark source changed during measurement')
        audit.update(status='complete',wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            largest_worker_RSS_bytes=max(b[a]['peak_RSS_bytes'] for b in audit['benchmarks'] for a in ('R0','R1')))
    except BaseException as exc:
        audit.update(status='failed',failure_type=type(exc).__name__,failure=str(exc),wall_seconds=time.perf_counter()-begin);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);persist()
        with (output/'audit.json').open('x') as stream:json.dump(audit,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return audit


def main():
    sys.meta_path.insert(0,NoScience())
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',choices=('R0','R1'));parser.add_argument('--fixture',type=Path);parser.add_argument('--fixture-sha256');args=parser.parse_args()
    journal.require(all(os.environ.get(k)==v for k,v in {'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}.items()),'Hidden GPU/one-thread environment required')
    soft,hard=resource.getrlimit(resource.RLIMIT_AS);cap=8*1024**3
    resource.setrlimit(resource.RLIMIT_AS,(min(cap,soft) if soft>=0 else min(cap,hard) if hard>=0 else cap,hard))
    if args.worker:
        journal.require(args.fixture is not None and args.fixture_sha256 is not None,'Explicit synthetic fixture selection required')
        reporting_worker(args.fixture,args.fixture_sha256,args.output,args.worker)
    else:
        journal.require(args.report is not None,'Explicit original report required');r=run(args.report,args.output)
        print(json.dumps(dict(status=r['status'],audit=str(args.output/'audit.json'),sha256=journal.file_sha(args.output/'audit.json'))),flush=True)


if __name__=='__main__':main()
