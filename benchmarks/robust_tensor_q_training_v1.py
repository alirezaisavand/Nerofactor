"""One source-bound batch with two existing acquisition/control recipes."""
from pathlib import Path

import numpy as np

from benchmarks import bell_distance_robust_v1 as capped
from benchmarks import bell_heterogeneous_training_v1 as bell
from benchmarks import donut_heterogeneous_training_v1 as donut
from benchmarks import robust_tensor_q_v1 as robust
from benchmarks.donut_recovery_v1 import inputs
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, ROOT, baseline, digest

PROTOCOL = 'robust-pseudo-huber-transported-Q-v1'
FREEZE = 'docs/protocols/ROBUST_TENSOR_Q_TRAINING_V1.freeze.json'
ARMS = ('Donut-R1','Bell-R1')
IMAGE = bell.IMAGE
FOREGROUND_SECONDS = 7500
CONTROL_FREEZES = {
    'Donut-R1': (donut.FREEZE,'d6be4b2ab1d3885a4f35a27d435f345780ee7c1f263c98ec4103b3b2f4129ed3','D6'),
    'Bell-R1': (capped.FREEZE,'d0766caa6f362c8b698522ae54061db7127b5a8520edbd4b8e85655000cd9ecf','B2'),
}
CONTROL_RESULTS = {
    'Donut-R1': {'D2':'donut_heterogeneous_training_v1','D6':'donut_heterogeneous_training_v1'},
    'Bell-R1': {'D2':'bell_heterogeneous_training_v1','B2':'bell_distance_robust_v1'},
}


def recipe(control, delta):
    extra = dict(component_reverse_sweeps=control['work']['component_reverse_sweeps']+6,
                 robust_diagnostic_batches=10000)
    work = dict(control['work'],**extra)
    return dict(control,robust_delta=delta,work=work,limits={**control['limits'],**work})


def result_manifest(arm, control):
    return ROOT/'docs/results'/CONTROL_RESULTS[arm][control]/'artifact-manifest.json'


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha,'Robust tensor-Q freeze differs')
    f = bell.b.read(path)
    require(f['protocol']==PROTOCOL and f['order']==list(ARMS) and f['image']==IMAGE
            and f['foreground_seconds']==FOREGROUND_SECONDS and f['delta_policy']==robust.CALIBRATION,
            'Robust batch definition differs')
    evidence_path = ROOT/f['calibration']['relative_path']
    require(evidence_path.resolve()==evidence_path and ROOT in evidence_path.parents
            and digest(evidence_path)==f['calibration']['sha256'],'Frozen fitting-only calibration differs')
    evidence = bell.b.read(evidence_path)
    for arm,(relative,base_sha,control) in CONTROL_FREEZES.items():
        record = ROOT/relative
        require(digest(record)==base_sha,'Historical control freeze differs: '+arm)
        base = bell.b.read(record)
        expected = {k:v for k,v in base.items() if k not in ('sources','arms','host_output','order','foreground_seconds')}
        expected['recipe'] = recipe(base['arms'][control],evidence['arms'][arm]['delta'])
        require(f['scenes'][arm]==expected,'Robust arm differs from its accepted control: '+arm)
    for name,value in f['sources'].items():
        file = ROOT/name
        require(file.resolve()==file and ROOT in file.parents and digest(file)==value,'Frozen robust source differs: '+name)
    baseline('material',PINS['material'])
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL,approved=True,commit=commit,freeze_sha256=sha,image_id=IMAGE,selected_gpu='0',
        host_output=f['host_output'],order=list(ARMS),maximum_execution_launches=1,foreground_seconds=FOREGROUND_SECONDS,
        researcher_admitted_dataset=True,
        scenes={k:dict(dataset=v['dataset'],recipe=v['recipe']) for k,v in f['scenes'].items()})


def verify_calibration(f):
    """Replay the frozen fit-support filter without opening material or held-out arrays."""
    evidence = bell.b.read(ROOT/f['calibration']['relative_path'])
    require(evidence['policy']==robust.CALIBRATION,'Fitting calibration policy differs')
    for arm, record in evidence['arms'].items():
        scene = f['scenes'][arm]
        acquired = Acquisition(arm)

        def checked(item):
            return bell.b.check_artifact(Path(item['path']),item)

        require(record['generation']==scene['dataset']['generation'],'Calibration dataset differs')
        generation = bell.b.read(checked(record['generation']))
        receipt = bell.b.read(checked(record['control_receipt']))
        control = CONTROL_FREEZES[arm][2]
        manifest = bell.b.read(result_manifest(arm,control))
        require(record['control_receipt']==manifest['arms'][control]['receipt']
                and record['control']==receipt['arm']==control and receipt['optimizer_returns']==10000
                and receipt['status']=='complete' and receipt['cleanup_verified'] and receipt['scientific_work_complete']
                and receipt['benchmark_sha256']==acquired.GENERATION
                and receipt['freeze_sha256']==CONTROL_FREEZES[arm][1], 'Calibration control completion differs')
        root = Path(record['control_receipt']['path']).parent
        domain_key = 'surface-q-domain.npz'
        prediction_key = 'step-10000/surface-q-diagnostic.npz'
        for k, key in (('domain',domain_key),('prediction_differences',prediction_key)):
            require(record[k]==dict(receipt['outputs'][key],path=str(root/key)),
                    'Calibration array is foreign to the accepted control: '+k)

        positions, normals = [], []
        cameras = acquired.SPLITS['fitting']
        require(len(cameras)==len(record['fitting_geometry']),'Calibration fitting panel differs')
        for camera,item in zip(cameras,record['fitting_geometry']):
            name = f'view-{camera}.npz'
            require(item==dict(generation['outputs'][name],path=str(Path(scene['host_reference'])/name)),
                    'Non-fitting input selected for calibration')
            with np.load(checked(item),allow_pickle=False) as data:
                mask = data['shader_valid']
                positions.append(data['position'][mask])
                normals.append(data['normals'][mask])
        fields = ('diagnostic_ids','distance','points0','points1','normals0','normals1')
        with np.load(checked(record['domain']),allow_pickle=False) as data:
            domain = {k:data[k] for k in fields}
        with np.load(checked(record['prediction_differences']),allow_pickle=False) as data:
            np.testing.assert_array_equal(data['pair_ids'],domain['diagnostic_ids'])
            square = data['transported_square']
        recipe = scene['recipe']
        actual = robust.calibrate(domain,square,np.concatenate(positions),np.concatenate(normals),
                                  recipe['spatial_policy'],recipe.get('distance_weighting'))
        require(all(record[k]==v for k,v in actual.items()),'Frozen numerical calibration differs: '+arm)


class Acquisition:
    ROOT, FREEZE, PROTOCOL, IMAGE = ROOT, FREEZE, PROTOCOL, IMAGE

    def __init__(self, arm):
        require(arm in ARMS,'Unknown robust arm')
        self.arm, self.ARMS = arm,(arm,)
        self.FACETED = arm=='Bell-R1'
        self.control = bell if self.FACETED else donut
        layout = bell if self.FACETED else donut.ref
        self.PANEL,self.SPLITS = layout.PANEL,layout.SPLITS

    def __getattr__(self, name):
        return getattr(self.control,name)

    def freeze(self, path, sha):
        batch = freeze(path,sha)
        scene = batch['scenes'][self.arm]
        return dict(scene,arms={self.arm:scene['recipe']},host_output=batch['host_output'],batch=batch)

    def authorization(self, f, commit, sha):
        return authorization(f['batch'],commit,sha)

    def runtime(self, args):
        f,a = self.control.runtime(args,spec=self)
        if self.FACETED:
            f['surface_q_training']['D6_domain_array_sha256'] = f['D6_native_domain_hashes']
        return f,dict(a,benchmark_generation_sha256=self.GENERATION)

    def host_inputs(self, f):
        if self.FACETED:
            mesh = Path(f['mesh']['path'])
            self.geometry(f,mesh)
            return dict(oracle=mesh)
        paths = {k:Path(v) for k,v in f['host_inputs'].items()}
        inputs(paths['oracle'],paths['cameras'])
        return paths

    def edge_diagnostic(self, *args):
        return capped.edge_diagnostic(*args)

    def audit_weighted_history(self, *args):
        return capped.audit_weighted_history(*args)

    def check_donut_domain(self, f, domain, views, hashes, saved):
        expected = dict(domain,fitting_camera_count=bell.d6.fitting_face_counts(
            views,self.SPLITS['fitting'],len(domain['face_area'])))
        bell.check_probe_archive(expected,saved)
        require(hashes==f['D6_domain_array_sha256']=={k:bell.array_hash(v) for k,v in saved.items()},
                'Archived Donut robust domain differs')

    def check_sampler_capture(self, f, domain, step, capture):
        snapshots = f['sampler_snapshots'] if self.FACETED else f['D6_sampler_snapshots']
        bell.check_sampler_capture(dict(sampler_snapshots=snapshots),domain,step,capture)


def audit_robust_history(history):
    names = ('penalty_ratio','mean_gradient_multiplier','min_gradient_multiplier','fraction_above_delta')
    summaries = []
    for row in history:
        d = row['robust_diagnostic']
        values = [d[k] for k in names]
        q,loss = d['quadratic_loss'],row['surface_q_loss']
        require(np.isfinite([q,loss,*values]).all() and 0<=loss<=q*(1+1e-6)
                and all(0<=v<=1+1e-6 for v in values)
                and abs((loss/q if q>0 else 1.)-d['penalty_ratio'])<=2e-6,
                'Invalid robust loss or attenuation history')
        summaries.append(values)
    values = np.asarray(summaries)
    return {name:dict(zip(('min','p50','p95','max'),np.quantile(values[:,i],[0,.5,.95,1]).tolist()))
            for i,name in enumerate(names)}
