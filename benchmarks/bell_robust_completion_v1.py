"""Standalone execution of the previously frozen Bell-R1 recipe."""
from pathlib import Path

import numpy as np

from benchmarks import robust_tensor_q_training_v1 as prior
from benchmarks.donut_heterogeneous_v1 import check_artifact, read
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, ROOT, baseline, digest

PROTOCOL = 'bell-robust-tensor-q-completion-v1'
FREEZE = 'docs/protocols/BELL_ROBUST_TENSOR_Q_COMPLETION_V1.freeze.json'
IMAGE = prior.IMAGE
ARMS = ('Bell-R1',)
FOREGROUND_SECONDS = 3780
PRIOR_SHA = '9dcbf267369e04ab2e18076e34144ee55b78b15a3d167846e8ae8284246f94a3'
FAILURE_RECORD = 'docs/results/robust_tensor_q_v1/failure-analysis.json'


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha,'Bell completion freeze differs')
    f = read(path)
    old_path = ROOT/prior.FREEZE
    require(digest(old_path)==PRIOR_SHA,'Previously authorized paired freeze differs')
    old = read(old_path)
    require(f['protocol']==PROTOCOL and f['order']==list(ARMS) and f['image']==IMAGE
            and f['foreground_seconds']==FOREGROUND_SECONDS and f['prior_freeze_sha256']==PRIOR_SHA
            and f['host_output']!=old['host_output'] and f['calibration']==old['calibration']
            and f['scenes']=={'Bell-R1':old['scenes']['Bell-R1']},'Previously frozen Bell recipe differs')
    for name,expected in f['sources'].items():
        file = ROOT/name
        require(file.resolve()==file and ROOT in file.parents and digest(file)==expected,
                'Frozen Bell completion source differs: '+name)
    baseline('material',PINS['material'])
    return f


def authorization(f, commit, sha):
    return dict(prior.authorization(f,commit,sha),protocol=PROTOCOL,order=list(ARMS),
                foreground_seconds=FOREGROUND_SECONDS,previous_failed_container=f['previous_failed_container'],
                previous_donut_run_reusable=False)


def verify_inputs(f):
    """Check saved admission and failure identities without recalibrating delta."""
    saved = read(ROOT/FAILURE_RECORD)
    for name in ('freeze.json','authorization.json','Donut-R1-exit-code.txt','recovery.json'):
        item = saved['artifacts'][name]
        check_artifact(Path(item['path']),item)
    require(saved['status']=='failed' and saved['bell_R1_status']=='unexecuted'
            and saved['completed_updates']==2568 and saved['freeze_sha256']==PRIOR_SHA,
            'Prior paired failure differs')
    item = f['previous_failed_container']
    check_artifact(Path(item['path']),item)
    cid = Path(item['path']).read_text().strip()
    require(Path(item['path'])==Path(saved['batch'])/'Donut-R1.cid'
            and len(cid)==64 and all(c in '0123456789abcdef' for c in cid), 'Foreign prior container ID')
    cal = ROOT/f['calibration']['relative_path']
    require(digest(cal)==f['calibration']['sha256']
            and read(cal)['arms']['Bell-R1']['delta']==f['scenes']['Bell-R1']['recipe']['robust_delta'],
            'Previously calibrated Bell delta differs')
    return cid,Path(saved['batch']).name+'-Donut-R1'


class Acquisition(prior.Acquisition):
    FREEZE, PROTOCOL = FREEZE, PROTOCOL
    CAPTURE_ORIENTATION_FAILURE = True

    def __init__(self, arm):
        require(arm=='Bell-R1','Bell completion admits no other arm')
        super().__init__(arm)

    def freeze(self, path, sha):
        batch = freeze(path,sha)
        scene = batch['scenes'][self.arm]
        return dict(scene,arms={self.arm:scene['recipe']},host_output=batch['host_output'],batch=batch)

    def authorization(self, f, commit, sha):
        return authorization(f['batch'],commit,sha)


def orientation_failure(exc, pending_update):
    """Capture retained local tensors from the existing exception, with zero model calls."""
    if 'Undefined zero/near-zero orientation' not in str(exc):
        return None
    frames = {}
    cause = exc
    while cause is not None:
        tb = cause.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            key = (Path(frame.f_code.co_filename).name,frame.f_code.co_name)
            frames[key] = frame.f_locals
            tb = tb.tb_next
        cause = cause.__cause__
    core = 'material_a_v1.py'
    predictor = frames.get((core,'predict_anisotropic_components'))
    guard = frames.get((core,'orientation'))
    if predictor is None or guard is None or 'norm' not in guard:
        return dict(status='unavailable',pending_update=pending_update,reason='Raw predictor and guard frames unavailable')
    to_numpy = lambda x: x.detach().cpu().numpy()
    points = to_numpy(predictor['pts'])
    values = predictor['values']
    widths = np.column_stack([to_numpy(values[i]).reshape(-1) for i in (0,1)]).astype(np.float64)
    raw = to_numpy(guard['r'])
    norms = to_numpy(guard['norm']).reshape(-1)
    from scientific_variants.material_a_v1 import ORIENTATION_MIN
    bad = np.flatnonzero(norms<ORIENTATION_MIN)
    require(len(bad)>0 and points.shape==(len(norms),3) and widths.shape==(len(norms),2)
            and raw.shape==(len(norms),2),'Retained orientation failure tensors differ')
    spatial = frames.get(('donut_surface_q_d6.py','surface_q_term'))
    work = frames.get(('donut_material_recovery_v1.py','work'))
    suffix = spatial['suffix'] if spatial is not None else None
    ids = work['indices'] if spatial is not None and work is not None else None
    rows = []
    for i in bad[:16]:
        w = widths[i]
        squared = w*w
        gap = float(abs(squared[0]-squared[1])/squared.sum()) if np.isfinite(squared).all() and squared.sum()>0 else None
        rows.append(dict(row=int(i),point_native=[float(v) if np.isfinite(v) else None for v in points[i]],
            orientation_vector=raw[i].tolist(),raw_orientation_norm=float(norms[i]),
            chart_widths=[float(v) if np.isfinite(v) else None for v in w],anisotropy_gap=gap,
            surface_pair_id=int(ids[i]) if ids is not None else None))
    return dict(status='captured',pending_update=pending_update,threshold=ORIENTATION_MIN,surface_endpoint=suffix,
                batch_size=len(norms),offending_rows=int(len(bad)),rows=rows)
