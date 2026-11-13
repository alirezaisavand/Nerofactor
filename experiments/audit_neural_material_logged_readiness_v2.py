"""Bounded synthetic D.3 storage/memory/commit audit; zero scientific calls."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import signal
import time
from unittest import mock

import numpy as np
import torch

from experiments import freeze_neural_material_recovery_logged_v2 as f
from experiments import neural_material_recovery_logged_v2_execution as execution
from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.neural_material_journal_runtime_v2 import RuntimeOutput
from infrastructure.scientific_reporting_adapter_v2 import Tracker,plain
from infrastructure.neural_material_resources_v2 import budget,capacity,selected,D2,D2_SHA


def micro(parent,old,resources,status):
    root=parent/status;root.mkdir()
    class Output(RuntimeOutput):pass
    Output.freeze=dict(resource_budget=resources,sha256=old['sha256'],configuration_sha256=old['configuration_sha256'])
    report=Tracker(dict(status='running',run_id='synthetic-'+status,synthetic_fixture=True,settings=old['settings'],
        reviewed_freeze_sha256='a'*64,outcomes=[dict(material='M1',seed=101,arm='N1',status='optimizing',completed_updates=0,history=[],work={})])).root
    output=Output(root,'synthetic-'+status);output.persist(report);costs=[]
    evidence=selected(f.ROOT/D2,D2_SHA)
    saved=log.decode((Path(evidence['reporting_benchmarks']['complete']['measurement']['path']).parent/'report.json').read_bytes())
    template=next(r for r in saved['outcomes'] if r['arm']=='N1')['history'][0];del saved
    for i in range(1,33):
        start=time.perf_counter();row=report['outcomes'][0]
        for k in old['configuration']['work_per_history']['N1']:row['work'][k]=i
        row['completed_updates']=i*10;record=copy.deepcopy(template);record.update(update=i*10,loss_at_update=i*10-1)
        row['history'].append(record);output.persist(report);costs.append(time.perf_counter()-start)
    report['status']=status;report['outcomes'][0]['status']='complete' if status=='complete' else 'censored';output.persist(report)
    before=time.perf_counter()
    # Fails if shutdown attempts the old whole-state digest or materialization.
    with mock.patch.object(log,'digest',side_effect=AssertionError('No full-state digest at shutdown')):
        output.close()
    elapsed=time.perf_counter()-before
    assert elapsed<1.,'Predeclared1-second bounded seal readiness gate'
    assert not (root/'report.json').exists() and not (root/'journal/final.json').exists()
    return dict(synthetic=True,status=status,persist_calls=output.persist_calls,progress_samples=32,
        persist_seconds=costs,mean_progress_seconds=sum(costs)/len(costs),max_progress_seconds=max(costs),
        total_commit_fsync_anchor_seconds=output.commit_seconds,shutdown_seconds=elapsed,
        anchor_sha256=log.file_sha(root/'commit.json'),end_sha256=log.file_sha(root/'execution-end.json'),
        journal_sha256=log.file_sha(root/'journal/events.jsonl'),journal_bytes=(root/'journal/events.jsonl').stat().st_size,
        scientific_updates=0)


def memory_fixture(old,resources):
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    # Inert, touched arrays only; no real model, gradient calculation or Adam.
    arrays=[]
    for model in old['initial_models'].values():
        for p in model['parameters'].values():
            for role in ('parameter','gradient','first_moment','second_moment'):
                a=np.empty(p['shape'],dtype=np.float32);a.fill(.125);arrays.append(a)
        for b in model['buffers'].values():
            a=np.empty(b['shape'],dtype=np.float32);a.fill(.125);arrays.append(a)
    e=selected(f.ROOT/D2,D2_SHA);path=Path(e['reporting_benchmarks']['complete']['measurement']['path']).parent/'report.json'
    original=log.decode(path.read_bytes());live=Tracker(original).root;reconstruction=plain(live);del original
    # Source-derived conservative capture footprint (not autograd execution).
    workspace=np.empty(resources['diagnostic_archive_cap_bytes']//3,dtype=np.uint8);workspace.fill(1)
    state=np.empty(resources['state_file_caps_bytes']['N1']*2,dtype=np.uint8);state.fill(1)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<2*1024**3,'Bounded synthetic memory audit2GiB cap'
    return dict(before_peak_RSS_bytes=before,peak_RSS_bytes=peak,touched_model_gradient_moment_buffer_bytes=sum(a.nbytes for a in arrays),
        workspace_proxy_bytes=workspace.nbytes,checkpoint_proxy_bytes=state.nbytes,report_copies=2,
        actual_model_constructors=0,shader_forwards=0,backwards=0,optimizer_steps=0,
        limitation='Inert array/report fixture does not execute or guarantee real autograd allocation; original4GiB/8GiB guards remain')


def run(output):
    begin=time.perf_counter();old=f.lineage();resources=budget(old);capacity(f.PARENT,resources['minimum_free_bytes'])
    output=Path(output);assert output.is_absolute() and output.resolve()==output and not output.exists();output.mkdir()
    h=execution.d2._scientific_v1();result=dict(protocol='neural-material-logged-v2-D3-bounded-readiness',status='running',scientific_calls=0,source_audit=execution.source_equivalence(),resource_budget=resources)
    saved=signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('D3 engineering120second cap')))
    signal.setitimer(signal.ITIMER_REAL,max(.01,120-(time.perf_counter()-begin)))
    soft,hard=resource.getrlimit(resource.RLIMIT_AS);resource.setrlimit(resource.RLIMIT_AS,(min(8*1024**3,soft) if soft>=0 else min(8*1024**3,hard) if hard>=0 else 8*1024**3,hard))
    try:
        with mock.patch.object(h.previous.original,'MaterialShaderV1',side_effect=AssertionError('No scientific shader')),             mock.patch.object(h.field,'make_material',side_effect=AssertionError('No scientific model')),             mock.patch.object(torch.optim,'Adam',side_effect=AssertionError('No real optimizer')),             mock.patch.object(torch.autograd,'backward',side_effect=AssertionError('No real backward')):
            result['commits']=[micro(output,old,resources,s) for s in ('complete','censored')]
            result['memory']=memory_fixture(old,resources)
        result['status']='passed'
    except BaseException as exc:result.update(status='failed',failure_type=type(exc).__name__,failure=str(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,saved)
        result.update(wall_seconds=time.perf_counter()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (output/'audit.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert all(os.environ.get(k)==v for k,v in dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items())
    torch.set_num_threads(1);r=run(a.output);print(json.dumps(dict(status=r['status'],path=str(a.output/'audit.json'),sha256=log.file_sha(a.output/'audit.json'))))


if __name__=='__main__':main()
