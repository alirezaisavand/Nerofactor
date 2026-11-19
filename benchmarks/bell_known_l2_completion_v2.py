"""Finish K2/K3 with a corrected pair selector and retain completed K1."""
from pathlib import Path
import sys

from benchmarks import bell_known_l2_v1 as previous
from benchmarks.bell_known_l2_v1 import (
    ADMISSION, GENERATION, IMAGE, PANEL, ROOT, SPATIAL_POLICY, SPLITS, STEPS,
    boundary_score, check_domain_archive, check_probe_archive, check_sampler_capture,
    dataset, diagnostic_frames, diagnostics, fit_pool, geometry, point_errors,
    route_plan, validate_map, verify_illumination_receipt,
)
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, baseline, digest

PROTOCOL = 'bell-kurt-known-L2-tensor-completion-v2'
FREEZE = 'docs/protocols/BELL_KNOWN_L2_COMPLETION_V2.freeze.json'
PRIOR_SHA = '50bfdc692b1b72030779eb346358de32b5d473bb55aaaf2c1ab4acf907f1d431'
PRIOR_COMMIT = '0f47e5ee4a09a06b0d6c3daeab8f5d06fda37507'
PRIOR_AUTH_SHA = 'abe82a0d7cbe7800935be2a929a44e1c89ae3e3743ab2e1e4e5ea365370f472f'
ARMS = ('K2','K3')
FOREGROUND_SECONDS = 7500


def prior_definition():
    path = ROOT/previous.FREEZE
    require(digest(path)==PRIOR_SHA,'Original known-light freeze differs')
    return previous.b.read(path)


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha,'Frozen Bell completion identity differs')
    f = previous.b.read(path)
    old = prior_definition()
    inherited = {k:v for k,v in old.items() if k not in ('sources','arms','host_output','order','foreground_seconds','protocol')}
    require(all(f[k]==v for k,v in inherited.items()) and f['protocol']==PROTOCOL and f['order']==list(ARMS)
            and f['foreground_seconds']==FOREGROUND_SECONDS and f['prior_freeze_sha256']==PRIOR_SHA
            and f['host_output']!=old['host_output'] and f['arms']=={k:old['arms'][k] for k in ARMS},
            'Remaining tensor recipes or known-light definition changed')
    for name,sha256 in f['sources'].items():
        p = ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==sha256,'Frozen completion source differs: '+name)
    baseline('material',PINS['material'])
    return f


def authorization(f, commit, sha):
    return dict(previous.authorization(f,commit,sha),protocol=PROTOCOL,order=list(ARMS),
                foreground_seconds=FOREGROUND_SECONDS,prior_evidence=f['prior_evidence'])


def runtime(args):
    from benchmarks import bell_heterogeneous_training_v1 as native
    return native.runtime(args,spec=sys.modules[__name__])


def verify_controls(f):
    old = prior_definition()
    previous.verify_controls(old)
    batch = Path(old['host_output'])
    analysis_root = batch.with_name(batch.name+'-K1-analysis-v1')
    expected = {name:batch/name for name in ('K1/recovery.json','K1-exit-code.txt','K2/recovery.json','K2-exit-code.txt',
                'K2.cid','K2-console.txt','freeze.json','authorization.json')}
    expected['K1-analysis'] = analysis_root/'analysis.json'
    records = f['prior_evidence']
    require(records.keys()==expected.keys(),'Earlier run evidence inventory differs')
    for name,p in expected.items():
        item = records[name]
        require(Path(item['path'])==p,'Unexpected preserved evidence location')
        previous.b.check_artifact(p,item)
    require(digest(expected['freeze.json'])==PRIOR_SHA and digest(expected['authorization.json'])==PRIOR_AUTH_SHA
            and previous.b.read(expected['authorization.json'])==previous.authorization(old,PRIOR_COMMIT,PRIOR_SHA),
            'Earlier run freeze or approval differs')
    k1 = previous.b.read(expected['K1/recovery.json'])
    k2 = previous.b.read(expected['K2/recovery.json'])
    k1_analysis = previous.b.read(expected['K1-analysis'])
    require(all(r['commit']==PRIOR_COMMIT and r['freeze_sha256']==PRIOR_SHA
                and r['authorization_sha256']==PRIOR_AUTH_SHA for r in (k1,k2)),
            'Earlier receipt source or approval differs')
    require(k1['arm']=='K1' and k1['status']=='complete' and k1['scientific_work_complete'] and k1['cleanup_verified']
            and k1['optimizer_returns']==10000 and k1['counters']==old['arms']['K1']['work']
            and expected['K1-exit-code.txt'].read_text().strip()=='0'
            and k1_analysis['status']=='verified_complete' and k1_analysis['execution_commit']==PRIOR_COMMIT
            and k1_analysis['freeze_sha256']==PRIOR_SHA and set(k1_analysis['arms'])=={'K1'}
            and k1_analysis['arms']['K1']['receipt_sha256']==records['K1/recovery.json']['sha256']
            and [(r['step'],r['split']) for r in k1_analysis['results']]==[(s,p) for s in STEPS for p in SPLITS],
            'Completed K1 evidence differs')
    require(k2['arm']=='K2' and k2['status']=='failed' and k2['optimizer_returns']==0 and k2['saved_endpoints']==[]
            and k2['error']=="KeyError: 'D8-T_world_Q'" and expected['K2-exit-code.txt'].read_text().strip()=='1'
            and not (batch/'K3').exists(),'Prior K2 failure or unexecuted K3 differs')
    cid = expected['K2.cid'].read_text().strip()
    require(len(cid)==64 and all(c in '0123456789abcdef' for c in cid),'Invalid previous K2 container ID')
    return cid,batch.name+'-K2'
