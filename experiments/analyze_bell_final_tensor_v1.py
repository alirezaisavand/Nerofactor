"""Verify final pair receipts and compare matched saved trajectories with K1-Dense."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_final_tensor_v1 as f
from benchmarks.donut_recovery_v1 import json_new
from experiments.analyze_bell_known_l2_v1 import common_orientation
from experiments.analyze_donut_heterogeneous_training_v1 import analyze, read_endpoint
from experiments.bell_final_tensor_trainer_v1 import REGULARIZATION
from infrastructure.signatures import require

b = f.b
HISTORY_FIELDS = ('RGB_loss','material_prior','mx_mean','my_mean','preclip_gradient_l2')


def width_angle_error(predicted, truth):
    """Separate spectral width and eigenframe contributions in world Q."""
    def symmetric(value):
        matrix = value.astype(np.float64)
        return (matrix+matrix.transpose(0,2,1))/2
    predicted,truth = symmetric(predicted),symmetric(truth)
    eigenvalues = np.linalg.eigvalsh(predicted)-np.linalg.eigvalsh(truth)
    norm_sq = np.square(truth).sum((1,2))
    width_sq = np.square(eigenvalues).sum(1)/norm_sq
    total_sq = np.square(predicted-truth).sum((1,2))/norm_sq
    angle_sq = total_sq-width_sq
    require(np.isfinite(angle_sq).all() and (angle_sq>=-1e-12).all(),
            'Invalid eigenframe contribution to Q error')
    return width_sq,np.maximum(angle_sq,0)


def summarize(sha, output):
    freeze = f.freeze(f.ROOT/f.FREEZE,sha)
    f.verify_controls(freeze)
    batch = Path(freeze['host_output'])
    dense_report = b.read(output/'analysis.json')
    k1_path = Path(freeze['controls']['K1-Dense']['path'])
    paths = dict({arm:batch/arm/'recovery.json' for arm in f.ARMS},**{'K1-Dense':k1_path})
    original = f.dense.read_artifact(freeze['dense_analysis'])
    require(dense_report['status']=='verified_complete' and dense_report['freeze_sha256']==sha,
            'Missing verified final pair analysis')
    inputs = freeze['dataset']
    _,views = f.dataset(freeze,Path(freeze['host_reference']),Path(inputs['admission']['path']),
                        Path(inputs['verification']['path']))
    receipts,logs,by_state,admission = {},{},{},{}
    pair_files = ('paired-initial-gate.npz','tensor-physical-admission.npz','native-reference-gate.npz','boundary-probes.npz')
    pair = {}
    init = {}
    width_angle = []
    for arm,path in paths.items():
        r = receipts[arm] = b.read(path)
        control = (dense_report if arm in f.ARMS else original)['arms'][arm]
        require(b.digest(path)==control['receipt_sha256'],'Arm receipt hash differs')
        root = path.parent

        def capture(name):
            return b.numeric(b.check_artifact(root/name,r['outputs'][name]))

        if arm in f.ARMS:
            require(r['representation']==freeze['arms'][arm]['representation']
                    and r['effective_regularization']==REGULARIZATION and
                    r['optimization_terms']==['RGB_loss','material_prior','diffuse_white_prior']
                    and not any(k.startswith('surface_q_') for k in r['counters']) and
                    r['paired_initial_gate']['passed'] and r['physical_tensor_admission']['passed'],
                    'Final pair admission or no-spatial objective differs')
            stream = r['outputs'][f.dense.STREAM]
            require(stream==dict(bytes=20480000,sha256=freeze['photometric_index_sha256']),
                    'Final pair photometric indices differ')
            b.check_artifact(root/f.dense.STREAM,stream)
            admission[arm] = dict(initial_gate=r['paired_initial_gate'],tensor_math=r['physical_tensor_admission'],
                                  parameters=r['trainable_parameters'])
            arrays = {name:capture(name) for name in pair_files}
            if pair:
                for name,values in arrays.items():
                    require(values.keys()==pair[name].keys(),'Paired admission archive inventory differs')
                    for key,value in values.items():
                        np.testing.assert_array_equal(value,pair[name][key])
            else:
                pair = arrays
            for camera in f.PANEL:
                name = f'step-00000/camera-{camera}/materials.npz'
                values = capture(name)
                initial = {k:values[k][values['hit_mask']] for k in ('Q_tensor_direct','kd','metallic','linear_rgb')}
                if camera in init:
                    for key in ('kd','metallic'):
                        np.testing.assert_array_equal(init[camera][key],initial[key])
                    for key,tol in (('Q_tensor_direct',f.d9.REPRESENTATION['initial_Q_atol']),
                                    ('linear_rgb',f.d9.REPRESENTATION['initial_RGB_atol'])):
                        np.testing.assert_allclose(init[camera][key],initial[key],atol=tol,rtol=0)
                else:
                    init[camera] = initial
        with b.check_artifact(root/'training.jsonl',r['outputs']['training.jsonl']).open() as stream:
            logs[arm] = [json.loads(line) for line in stream]
        require(len(logs[arm])==10000,'Unexpected training history count')
        for step in f.STEPS:
            for camera in f.SPLITS['held_out']:
                view = views[camera]
                for name in (f'step-{step:05d}/camera-{camera}/materials.npz',
                             f'step-{step:05d}/camera-{camera}-errors.npz'):
                    b.check_artifact(root/name,r['outputs'][name])
                scores = read_endpoint(root,step,camera,view)
                by_state.setdefault(step,{}).setdefault(arm,{})[camera] = scores
                errors,_,_,predicted,_ = scores
                width,angle = width_angle_error(predicted,view['Q_world'][view['shader_valid']])
                np.testing.assert_allclose(np.sqrt(width+angle),errors['Q_relative'],atol=2e-6,rtol=2e-6)
                width_angle.append(dict(arm=arm,step=step,camera=camera,
                    mean_normalized_squared_width=float(width.mean()),mean_normalized_squared_orientation=float(angle.mean())))
    rates = [[row['lr'] for row in logs[arm]] for arm in paths]
    require(all(cur==rates[0] for cur in rates), 'Final pair or K1-Dense schedule differs')
    trajectories = [r for report in (dense_report,original) for r in report['results']
                    if r['arm'] in paths and r['split']=='held_out']
    common = {str(step):{'paired':common_orientation({a:rows[a] for a in f.ARMS},f.SPLITS['held_out']),
                         'all':common_orientation(rows,f.SPLITS['held_out'])} for step,rows in by_state.items()}
    chart,ambient = (logs[arm] for arm in f.ARMS)
    first = {field:next((i for i,(a,z) in enumerate(zip(chart,ambient),1) if a[field]!=z[field]),None)
             for field in HISTORY_FIELDS}
    windows = {arm:[dict(first_update=i+1,last_update=i+500,**{
                  k:float(np.mean([v[k] for v in rows[i:i+500]])) for k in HISTORY_FIELDS})
                  for i in range(0,10000,500)] for arm,rows in logs.items()}
    diagnostics = {arm:dict(elapsed_seconds=r['elapsed_seconds'],resources=r['resources'],work=r['counters'],
        tensor_forward_CUDA_seconds=r.get('tensor_training_forward_CUDA_seconds'),
        boundary=(dense_report if arm in f.ARMS else original)['arms'][arm]['boundary_history'])
        for arm,r in receipts.items()}
    json_new(output/'method-comparison.json',dict(schema_version=1,initial_admission=admission,
        trajectories=trajectories,common_orientation=common,width_orientation_Q_squared=width_angle,
        first_numeric_difference_before_named_update=first,training_windows=windows,arms=diagnostics,
        offline_scientific_queries=0,primary_step=10000))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--output',required=True,type=Path)
    args = cli.parse_args()
    print(f'Verified {analyze(args.freeze_sha256,args.output,acquisition=f)} arm/state/split records')
    summarize(args.freeze_sha256,args.output)
