"""Retain K1 and finish the unchanged tensor arms with the full Bell adapter."""
from pathlib import Path
import sys

from benchmarks import bell_heterogeneous_training_v1 as native
from benchmarks import bell_known_l2_completion_v2 as previous
from benchmarks import bell_known_l2_v1 as known
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, ROOT, baseline, digest

PROTOCOL = 'bell-kurt-known-L2-tensor-completion-v3'
FREEZE = 'docs/protocols/BELL_KNOWN_L2_COMPLETION_V3.freeze.json'
PREVIOUS_SHA = '5fa9323b00d472a64e852c6d209b12e4e5183c602d82a4a858ce34e155b41acb'
PREVIOUS_COMMIT = '66e2520e57569d7a1b9567dbfe37d782524fbfdc'
PREVIOUS_AUTH_SHA = '61e13709403b7e5c02d09c5f19d35ece8cfe7b32a8b17739ded2d7938dff411d'
ARMS = previous.ARMS
FOREGROUND_SECONDS = previous.FOREGROUND_SECONDS
FAILURE_ARTIFACTS = (
    'K2/recovery.json', 'K2-exit-code.txt', 'K2.cid', 'K2-console.txt', 'freeze.json', 'authorization.json',
    'previous-run-cleanup.json', 'K2/paired-initial-gate.npz', 'K2/step-00000/camera-0/view.json',
    'K2/step-00000/camera-0/materials.npz',
)


def __getattr__(name):
    # Use the complete established known-light interface, including TARGET_KEYS.
    return getattr(known,name)


def v2_definition():
    path = ROOT/previous.FREEZE
    require(digest(path)==PREVIOUS_SHA,'Original v2 completion freeze differs')
    return known.b.read(path)


def prior_definition():
    return previous.prior_definition()


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha,'Frozen Bell v3 identity differs')
    f = known.b.read(path)
    old = v2_definition()
    inherited = {k:v for k,v in old.items() if k not in ('sources','host_output','protocol')}
    require(all(f[k]==v for k,v in inherited.items()) and f['protocol']==PROTOCOL
            and f['previous_freeze_sha256']==PREVIOUS_SHA and f['host_output']!=old['host_output'],
            'Tensor recipes or known-light definition changed')
    for name,expected in f['sources'].items():
        p = ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==expected,'Frozen v3 source differs: '+name)
    baseline('material',PINS['material'])
    return f


def authorization(f, commit, sha):
    return dict(previous.authorization(f,commit,sha),protocol=PROTOCOL,second_failure=f['second_failure'])


def runtime(args):
    return native.runtime(args,spec=sys.modules[__name__])


def verify_controls(f):
    old = v2_definition()
    first_id,first_label = previous.verify_controls(old)
    batch = Path(old['host_output'])
    receipts = f['second_failure']
    require(receipts.keys()==set(FAILURE_ARTIFACTS),'V2 failure evidence inventory differs')
    for name in FAILURE_ARTIFACTS:
        p,record = batch/name,receipts[name]
        require(Path(record['path'])==p,'Foreign v2 failure evidence path')
        known.b.check_artifact(p,record)
    require(digest(batch/'freeze.json')==PREVIOUS_SHA and digest(batch/'authorization.json')==PREVIOUS_AUTH_SHA
            and known.b.read(batch/'authorization.json')==previous.authorization(old,PREVIOUS_COMMIT,PREVIOUS_SHA),
            'V2 freeze or approval differs')
    cleanup = known.b.read(batch/'previous-run-cleanup.json')
    require(cleanup['container_id']==first_id and cleanup['label']==first_label
            and cleanup['absent_by_id'] is True and cleanup['absent_by_owner_label'] is True,
            'Original K2 cleanup evidence differs')
    failure = known.b.read(batch/'K2/recovery.json')
    require(failure['commit']==PREVIOUS_COMMIT and failure['freeze_sha256']==PREVIOUS_SHA
            and failure['authorization_sha256']==PREVIOUS_AUTH_SHA and failure['arm']=='K2'
            and failure['status']=='failed' and failure['optimizer_returns']==0 and failure['saved_endpoints']==[]
            and failure['native_reference_gate']['passed'] and failure['paired_initial_gate']['passed']
            and "'TARGET_KEYS'" in failure['error'] and (batch/'K2-exit-code.txt').read_text().strip()=='1'
            and not (batch/'K3').exists(),'V2 failure or unexecuted K3 differs')
    cid = (batch/'K2.cid').read_text().strip()
    require(len(cid)==64 and all(c in '0123456789abcdef' for c in cid),'Invalid v2 K2 container ID')
    return cid,batch.name+'-K2'
