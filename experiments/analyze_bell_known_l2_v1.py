"""File-only matched illumination comparison for completed Bell K1/K2/K3."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_known_l2_v1 as spec
from benchmarks.donut_heterogeneous_v1 import seen_area_weights
from benchmarks.donut_recovery_v1 import json_new,numeric
from experiments.analyze_donut_heterogeneous_training_v1 import analyze,read_endpoint
from infrastructure.signatures import require


def endpoint(root, receipt, camera, view):
    for name in (f'camera-{camera}/materials.npz',f'camera-{camera}-errors.npz'):
        key = 'step-10000/'+name
        spec.b.check_artifact(root/key,receipt['outputs'][key])
    return read_endpoint(root,10000,camera,view)


def compare_saved(f, output, analysis, *, known_roots=None):
    batch = Path(f['host_output'])
    known_roots = {arm:batch/arm for arm in spec.MATCH} if known_roots is None else known_roots
    files = f['dataset']
    _,views = spec.dataset(f,Path(f['host_reference']),Path(files['admission']['path']),Path(files['verification']['path']))
    geometry,_,_ = spec.geometry(f,Path(f['mesh']['path']))
    spec.verify_controls(f)
    scores = {}
    recovered = {}
    baseline_manifest = spec.b.read(spec.ROOT/spec.CONTROL_MANIFEST)
    baseline = baseline_manifest['offline_analysis'][0]
    old_analysis = spec.b.read(spec.b.check_artifact(Path(baseline['path']),baseline))
    for arm,known in known_roots.items():
        control = spec.MATCH[arm]
        original = f['controls'][arm]
        old = Path(original['path']).parent
        before,after = (spec.b.read(root/'recovery.json') for root in (old,known))
        require(before['initial']==after['initial'],'Known-light initial state changed')
        old_gate = 'native-reference-gate.npz'
        check = numeric(spec.b.check_artifact(old/old_gate,before['outputs'][old_gate]))
        native = numeric(known/old_gate)
        require(check.keys()==native.keys(),'Known-light compatibility inventory changed')
        for key,value in check.items():
            np.testing.assert_array_equal(native[key],value)
        for camera in spec.PANEL:
            key = f'step-00000/camera-{camera}/materials.npz'
            anchor = numeric(spec.b.check_artifact(old/key,before['outputs'][key]))
            value = numeric(known/key)
            for k in ('kd','metallic','mx','my','Q_world'):
                np.testing.assert_array_equal(value[k],anchor[k])
        histories = []
        for root,r in ((old,before),(known,after)):
            path = spec.b.check_artifact(root/'training.jsonl',r['outputs']['training.jsonl'])
            histories.append([json.loads(line) for line in path.read_text().splitlines()])
        require([r['lr'] for r in histories[0]]==[r['lr'] for r in histories[1]],'Learning-rate stream changed')
        prior_values = {}
        for name,history in zip((control,arm),histories):
            fields = ('RGB_loss','material_prior','diffuse_white_prior','preclip_gradient_l2')
            prior_values[name] = {k:dict(zip(('median','p95'),
                np.quantile([row[k] for row in history[-2000:]],[.5,.95]).tolist())) for k in fields}
        data = {name:{c:endpoint(root,r,c,views[c]) for c in spec.PANEL}
                for name,root,r in ((arm,known,after),(control,old,before))}
        recovered.update(data)
        detail = {}
        for split,cameras in spec.SPLITS.items():
            gt = np.concatenate([views[c]['widths_major_minor'][views[c]['shader_valid']] for c in cameras])
            ids = np.concatenate([views[c]['triangle'][views[c]['shader_valid']] for c in cameras])
            area_weights,_ = seen_area_weights(ids,geometry['area'])
            signed = {}
            for name,by_cam in data.items():
                widths = np.concatenate([by_cam[c][-1] for c in cameras])
                err = widths-gt
                signed[name] = dict(observation_weighted=err.mean(0).tolist(),
                                    area_weighted=np.average(err,axis=0,weights=area_weights).tolist())
            detail[split] = dict(signed_major_minor_bias=signed,common_orientation=common_orientation(data,cameras))
        learned_metrics = [row for row in old_analysis['results'] if row['arm']==control and row['step']==10000]
        fixed_metrics = [row for row in analysis['results'] if row['arm']==arm and row['step']==10000]
        scores[arm] = dict(learned_control=control,learned_receipt=original,splits=detail,
                           learned_metrics=learned_metrics,known_metrics=fixed_metrics,
                           learned_boundary=old_analysis['arms'][control]['boundary_history'][-1],
                           known_boundary=analysis['arms'][arm]['boundary_history'][-1],
                           learned_gradient_probes=old_analysis['arms'][control]['gradient_probes'],
                           known_gradient_probes=analysis['arms'][arm]['gradient_probes'],
                           last_2000_updates=prior_values,
                           initial_material_and_reference_checks_exact=True,matching_learning_rates=True)
    pairs = [(a,b) for a,b in (('K2','K3'),('K1','K3')) if {a,b}<=recovered.keys()]
    json_new(output/'matched-controls.json',dict(schema_version=1,learned_analysis=baseline,arms=scores,
        known_pairs={a+'-'+b:{split:common_orientation({k:recovered[k] for k in (a,b)},cameras)
                               for split,cameras in spec.SPLITS.items()} for a,b in pairs}))


def common_orientation(panels, cameras):
    values = {}
    for name,by_camera in panels.items():
        values[name] = [np.concatenate([by_camera[c][0][key] for c in cameras])
                        for key in ('orientation_eligible','orientation_degrees')]
    keep = np.logical_and.reduce([mask for mask,_ in values.values()])
    return dict(observations=int(keep.sum()),angle_median_p95={name:np.quantile(a[keep],[.5,.95]).tolist()
                    if keep.any() else None for name,(_,a) in values.items()})


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--output',type=Path,required=True)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--completion',action='store_true')
    mode.add_argument('--completion-v3',action='store_true')
    a = cli.parse_args()
    if a.completion or a.completion_v3:
        if a.completion_v3:
            from benchmarks import bell_known_l2_completion_v3 as completion
        else:
            from benchmarks import bell_known_l2_completion_v2 as completion
        f = completion.freeze(spec.ROOT/completion.FREEZE,a.freeze_sha256)
        cid,label = completion.verify_controls(f)
        cleanup = spec.b.read(Path(f['host_output'])/'previous-run-cleanup.json')
        require(cleanup['container_id']==cid and cleanup['label']==label
                and cleanup['absent_by_id'] and cleanup['absent_by_owner_label'],'Previous K2 cleanup evidence differs')
        count = analyze(a.freeze_sha256,a.output,acquisition=completion)
        earlier = completion.prior_definition()
        prior = spec.b.read(Path(f['prior_evidence']['K1-analysis']['path']))
        latest = spec.b.read(a.output/'analysis.json')
        combined = dict(latest,arms={**prior['arms'],**latest['arms']},results=prior['results']+latest['results'],
                        per_camera_endpoint=prior['per_camera_endpoint']+latest['per_camera_endpoint'],
                        common_orientation_arms=list(completion.ARMS))
        json_new(a.output/'combined-analysis.json',dict(combined,evidence_origins={'K1':f['prior_evidence']['K1-analysis'],
            'K2_K3':dict(path=str(a.output/'analysis.json'),sha256=spec.digest(a.output/'analysis.json'))}))
        compare_saved(earlier,a.output,combined,known_roots={
            'K1':Path(earlier['host_output'])/'K1',**{arm:Path(f['host_output'])/arm for arm in completion.ARMS}})
        count += len(prior['results'])
    else:
        f = spec.freeze(spec.ROOT/spec.FREEZE,a.freeze_sha256)
        count = analyze(a.freeze_sha256,a.output,acquisition=spec)
        compare_saved(f,a.output,spec.b.read(a.output/'analysis.json'))
    print(f'Verified {count} Bell checkpoint/split combinations and three matched controls')
