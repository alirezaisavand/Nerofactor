"""Compare verified Dense Bell outcomes with their sparse counterparts, file-only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_dense_known_l2_v1 as d
from benchmarks import donut_heterogeneous_metrics_v1 as metrics
from benchmarks.donut_recovery_v1 import json_new
from experiments.analyze_bell_known_l2_v1 import common_orientation, endpoint
from experiments.analyze_donut_heterogeneous_training_v1 import analyze
from infrastructure.signatures import require

b = d.b


def compare(sha, out):
    f = d.freeze(d.ROOT/d.FREEZE,sha)
    d.verify_controls(f)
    batch = Path(f['host_output'])
    manifest = f['dataset']
    _,views = d.dataset(f,Path(f['host_reference']),Path(manifest['admission']['path']),Path(manifest['verification']['path']))
    support_item = f['held_out_support']
    support = b.numeric(b.check_artifact(Path(support_item['path']),support_item))
    geometry,_,_ = d.geometry(f,Path(f['mesh']['path']))
    dense_result = b.read(out/'analysis.json')
    sparse_result = d.read_artifact(f['sparse_analysis'])
    require(dense_result['status']=='verified_complete' and dense_result['freeze_sha256']==sha,
            'Dense analysis verification differs')
    groups = dict(held_out=d.SPLITS['held_out'],original_fitting=d.dense.ORIGINAL)
    results, captured = {}, {}
    stream_bytes = f['training_projection']['training_slots']*d.INDEX_DTYPE.itemsize

    for arm,control in d.MATCH.items():
        old_path = Path(f['controls'][arm]['path'])
        old_root,new_root = old_path.parent,batch/arm
        old = d.read_artifact(f['controls'][arm])
        new = b.read(new_root/'recovery.json')
        require(b.digest(new_root/'recovery.json')==dense_result['arms'][arm]['receipt_sha256']
                and sparse_result['arms'][control]['receipt_sha256']==f['controls'][arm]['sha256']
                and old['initial']==new['initial'], 'Dense/sparse initial or terminal evidence differs')
        lr = []
        for root,receipt in ((old_root,old),(new_root,new)):
            path = b.check_artifact(root/'training.jsonl',receipt['outputs']['training.jsonl'])
            with path.open() as stream:
                lr.append([json.loads(line)['lr'] for line in stream])
        require(len(lr[0])==10000 and lr[0]==lr[1],'Dense and sparse learning rates differ')
        identity = dict(bytes=stream_bytes,sha256=f['photometric_index_sha256'])
        require(new['outputs'][d.STREAM]==identity,'Native photometric trace inventory differs')
        b.check_artifact(new_root/d.STREAM,identity)
        def saved(root, receipt, key):
            return b.numeric(b.check_artifact(root/key,receipt['outputs'][key]))
        keys = ['native-reference-gate.npz','boundary-probes.npz']
        if f['arms'][arm]['representation'] is not None:
            keys.append('paired-initial-gate.npz')
        for key in keys:
            former,current = saved(old_root,old,key),saved(new_root,new,key)
            require(former.keys()==current.keys(),'Dense/sparse control capture inventory differs')
            for name in former:
                np.testing.assert_array_equal(former[name],current[name])
        by_arm = {}
        for label,root,receipt in ((control,old_root,old),(arm,new_root,new)):
            by_arm[label] = {c:endpoint(root,receipt,c,views[c])
                             for c in (*d.SPLITS['held_out'],*d.dense.ORIGINAL)}
        for c in (*d.SPLITS['held_out'],*d.dense.ORIGINAL):
            key = f'step-00000/camera-{c}/materials.npz'
            former,current = saved(old_root,old,key),saved(new_root,new,key)
            for name in ('Q_world','mx','my','kd','metallic'):
                np.testing.assert_array_equal(former[name],current[name])
        captured.update(by_arm)
        result = results[arm] = dict(sparse_control=control,groups={},common_orientation={},
                boundary=dict(sparse=sparse_result['arms'][control]['boundary_history'][-1],
                              dense=dense_result['arms'][arm]['boundary_history'][-1]))
        for label,cameras in groups.items():
            valid = [views[c]['shader_valid'] for c in cameras]
            widths = np.concatenate([views[c]['widths_major_minor'][mask] for c,mask in zip(cameras,valid)])
            triangles = np.concatenate([views[c]['triangle'][mask] for c,mask in zip(cameras,valid)])
            area,_ = b.seen_area_weights(triangles,geometry['area'])
            all_points = np.ones(len(widths),bool)
            cohorts = dict(all=all_points,boundary=np.concatenate([views[c]['near_boundary'][i]
                           for c,i in zip(cameras,valid)]))
            if label=='held_out':
                for kind in ('nearby_fitting_count','nearby_highlight_count'):
                    sparse = np.concatenate([support[f'sparse_{c}_{kind}'] for c in cameras])
                    dense = np.concatenate([support[f'dense_{c}_{kind}'] for c in cameras])
                    for split, counts in (('sparse',sparse),('dense',dense)):
                        for n in (0,1,2,3,4,5):
                            cohorts[f'{split}_{kind}_{n if n<5 else "5plus"}'] = counts==n if n<5 else counts>=5
                    cohorts[f'newly_supported_{kind}'] = (sparse==0)&(dense>0)
            result['common_orientation'][label] = common_orientation(by_arm,cameras)
            result['groups'][label] = {}
            for method,capture in by_arm.items():
                errors = {key:np.concatenate([capture[c][0][key] for c in cameras]) for key in capture[cameras[0]][0]}
                rgb,truth = (np.concatenate([capture[c][i] for c in cameras]) for i in (1,2))
                estimate = np.concatenate([capture[c][4] for c in cameras])
                result['groups'][label][method] = {name:dict(
                    observation=metrics.score({k:v[mask] for k,v in errors.items()},rgb[mask],truth[mask],np.ones(mask.sum())),
                    area=metrics.score({k:v[mask] for k,v in errors.items()},rgb[mask],truth[mask],area[mask]),
                    signed_major_minor_bias=(estimate[mask]-widths[mask]).mean(0).tolist() if mask.any() else None)
                    for name,mask in cohorts.items()}
    pairs = {name:common_orientation({label:captured[label] for label in labels},d.SPLITS['held_out'])
             for name,labels in (('dense_representation',('K2-Dense','K3-Dense')),
                                 ('dense_recipe_gap',('K1-Dense','K3-Dense')),
                                 ('sparse_recipe_gap',('K1','K3')))}
    def q(arm,control):
        return results[arm]['groups']['held_out'][control]['all']['observation']['Q_relative']['median']
    gap = dict(sparse=q('K3-Dense','K3')-q('K1-Dense','K1'),
               dense=q('K3-Dense','K3-Dense')-q('K1-Dense','K1-Dense'))
    json_new(out/'dense-vs-sparse.json',dict(schema_version=1,arms=results,common_orientation=pairs,
        Q_median_K3_minus_K1=gap,photometric_stream_sha256=f['photometric_index_sha256'],
        exposure=dict(dense=f['training_projection']['effective_passes'],
                      sparse=10000*512/sum(d.dense.COUNTS[c] for c in d.dense.ORIGINAL)),
        offline_scientific_queries=0))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--output',required=True,type=Path)
    args = cli.parse_args()
    print(f'Verified {analyze(args.freeze_sha256,args.output,acquisition=d)} arm/state/split records')
    compare(args.freeze_sha256,args.output)
