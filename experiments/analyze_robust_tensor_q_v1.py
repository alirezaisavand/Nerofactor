"""Replay saved evidence for an explicitly selected robust tensor-Q arm."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import robust_tensor_q_training_v1 as h
from benchmarks.donut_heterogeneous_v1 import check_artifact, geometry_fields, read, seen_area_weights
from benchmarks.donut_recovery_v1 import inputs, json_new
from benchmarks.robust_tensor_q_v1 import width_profiles
from experiments.analyze_donut_heterogeneous_training_v1 import analyze, read_endpoint
from infrastructure.signatures import require


def historical_profiles(acquisition, f, *, candidate_root=None):
    """Score the same signed distance bins from the accepted control archives."""
    files = f['dataset']
    _,views = acquisition.dataset(f,Path(f['host_reference']),
                                   Path(files['admission']['path']),Path(files['verification']['path']))
    if acquisition.FACETED:
        geometry,_,_ = acquisition.geometry(f,Path(f['mesh']['path']))
        area = geometry['area']
    else:
        paths = acquisition.host_inputs(f)
        geometry,_,_ = inputs(paths['oracle'],paths['cameras'])
        _,area = geometry_fields(geometry)
    controls = {}
    eligibility = {}
    if candidate_root is not None:
        eligibility[acquisition.arm] = {
            camera:read_endpoint(candidate_root,10000,camera,views[camera])[0] for camera in acquisition.PANEL}
    for name in h.CONTROL_RESULTS[acquisition.arm]:
        manifest = read(h.result_manifest(acquisition.arm,name))
        item = manifest['arms'][name]['receipt']
        receipt = read(check_artifact(Path(item['path']),item))
        root = Path(item['path']).parent
        require(receipt['status']=='complete' and receipt['scientific_work_complete'] and receipt['cleanup_verified']
                and receipt['arm']==name and receipt['optimizer_returns']==10000
                and receipt['benchmark_sha256']==acquisition.GENERATION
                and receipt['commit']==manifest['execution_commit'],'Foreign boundary control receipt')
        widths, eligibility[name] = {},{}
        for camera in acquisition.PANEL:
            for file in (f'camera-{camera}/materials.npz',f'camera-{camera}-errors.npz'):
                key = 'step-10000/'+file
                check_artifact(root/key,receipt['outputs'][key])
            result = read_endpoint(root,10000,camera,views[camera])
            eligibility[name][camera],widths[camera] = result[0],result[-1]
        profiles = {}
        for split,cameras in acquisition.SPLITS.items():
            def valid(key):
                return np.concatenate([views[c][key][views[c]['shader_valid']] for c in cameras])

            weights,_ = seen_area_weights(valid('triangle'),area)
            profiles[split] = width_profiles(valid('boundary_distance_native'),valid('material_region'),
                np.concatenate([widths[c] for c in cameras]),valid('widths_major_minor'),weights)
        controls[name] = dict(receipt=item,boundary_transition_profiles=profiles)
    common = {}
    if candidate_root is not None:
        for split,cameras in acquisition.SPLITS.items():
            for control in controls:
                arms = (acquisition.arm,control)
                by_arm = {arm:{k:np.concatenate([eligibility[arm][c][k] for c in cameras])
                               for k in ('orientation_eligible','orientation_degrees')} for arm in arms}
                keep = np.logical_and.reduce([v['orientation_eligible'] for v in by_arm.values()])
                common.setdefault(split,{})[control] = dict(count=int(keep.sum()),orientation_degrees={
                    arm:dict(zip(('median','p95'),np.quantile(v['orientation_degrees'][keep],[.5,.95]).tolist()))
                    if keep.any() else None for arm,v in by_arm.items()})
    return controls,common


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--phase',choices=h.ARMS,required=True)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--bell-completion',action='store_true')
    x = cli.parse_args()
    acquisition = h.Acquisition
    if x.bell_completion:
        from benchmarks.bell_robust_completion_v1 import Acquisition
        acquisition = Acquisition
    acquired = acquisition(x.phase)
    f = acquired.freeze(h.ROOT/acquired.FREEZE,x.freeze_sha256)
    if x.bell_completion:
        from benchmarks.bell_robust_completion_v1 import verify_inputs
        cid,label = verify_inputs(f['batch'])
        cleanup = read(Path(f['host_output'])/'previous-run-cleanup.json')
        require(cleanup['container_id']==cid and cleanup['label']==label
                and cleanup['absent_by_exact_id'] is True and cleanup['absent_by_owner_label'] is True,
                'Standalone Bell cleanup witness differs')
    count = analyze(x.freeze_sha256,x.output,acquisition=acquired)
    historical,orientation = historical_profiles(acquired,f,candidate_root=Path(f['host_output'])/x.phase)
    json_new(x.output/'control-boundary-profiles.json',dict(schema_version=1,step=10000,controls=historical,
                                                             common_orientation=orientation))
    print(f'Verified {count} {x.phase} checkpoint/split combinations')
