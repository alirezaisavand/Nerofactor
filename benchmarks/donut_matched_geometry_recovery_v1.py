"""Bind Gate 4 to the accepted physical labels, observations, and trainer recipes."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from benchmarks import bell_final_tensor_v1 as bell
from benchmarks import donut_oracle_geometry_rgb_v1 as rgb
from benchmarks import donut_heterogeneous_training_v1 as old
from benchmarks.donut_recovery_v1 import numeric
from experiments.donut_oracle_geometry_rgb_v1 import capture_name
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest


PROTOCOL = 'donut-matched-geometry-recovery-v1'
FREEZE = 'docs/protocols/DONUT_MATCHED_GEOMETRY_RECOVERY_V1.freeze.json'
GATE3_SHA = '77653de0486e47ff1e4b52475b1afb5a1e8492ae15e2b520ccf5c97e2c3c4040'
GATE3 = ROOT/'docs/results/publication_gate3'
GATE3_RECEIPT = '832fa3706725e8abad345fd72036b5ee6ad163dfc18190217b785a9d016c4d62'
BELL_SHA = '8c881df7ce05f512a3f784e423bc52e07e1e8a850b028f52de2bd35cc8d72af3'
ARMS = ('K1-Oracle','K1-Reconstruction','Ambient-Oracle','Ambient-Reconstruction')
PANEL, SPLITS = rgb.geo.PANEL, rgb.geo.SPLITS
COUNTS = dict(zip(PANEL, (4513,2605,4131,3942,4830,2476)))
GENERATION, ADMISSION = old.GENERATION, old.ADMISSION
STEPS, POLICY, SHADER, IMAGE = old.STEPS, old.POLICY, old.SHADER, old.IMAGE
TARGET_KEYS, FIT_KEYS = old.TARGET_KEYS, old.FIT_KEYS
FOREGROUND_SECONDS = old.FOREGROUND_SECONDS
MOUNTS = dict(rgb.MOUNTS, gate3='/inputs/geometry-rgb', initial='/inputs/initial.json',
              ambient_initial='/inputs/ambient-initial.json')
SOURCES = ('benchmarks/donut_matched_geometry_recovery_v1.py', 'experiments/train_donut_matched_geometry_v1.py',
           'experiments/analyze_donut_matched_geometry_v1.py', 'experiments/run_donut_matched_geometry_v1.sh',
           'tests/scientific/test_donut_matched_geometry_v1.py', 'docs/protocols/DONUT_MATCHED_GEOMETRY_RECOVERY_V1.md',
           'docs/results/publication_gate3/artifact-manifest.json', 'docs/results/publication_gate3/rgb-metrics.json',
           'experiments/analyze_donut_heterogeneous_training_v1.py', 'benchmarks/donut_heterogeneous_metrics_v1.py')


def geometry_name(arm):
    require(arm in ARMS, 'Unknown matched geometry history')
    return arm.split('-')[1].lower()


def work(ambient):
    paired = 2 if ambient else 0
    result = dict(shader_calls=10000+32+paired+48*sum((n+511)//512 for n in COUNTS.values()),
                  shadow_rays=(10000*512+(32+paired)*24+48*sum(COUNTS.values()))*1536,
                  component_reverse_sweeps=30)
    if ambient:
        result.update(tensor_admission_predictor_calls=2,tensor_admission_reverse_sweeps=2)
    return result


def definition(out, support):
    from experiments.train_donut_matched_geometry_v1 import trainer_source
    g = rgb.freeze(GATE3_SHA)
    parent = bell.freeze(ROOT/bell.FREEZE, BELL_SHA)
    manifest = rgb.geo.h.read(GATE3/'artifact-manifest.json')
    require(manifest['receipt']['sha256'] == GATE3_RECEIPT and manifest['freeze']['sha256'] == GATE3_SHA,
            'Accepted Gate 3 receipt differs')
    rgb.geo.h.check_artifact(GATE3/'rgb-metrics.json',manifest['packaged_files']['rgb-metrics.json'])
    inherited = bell.dense_definition()['arms']['K1-Dense']
    arms = {}
    for arm in ARMS:
        ambient = arm.startswith('Ambient-')
        prior = parent['arms']['Final-Ambient'] if ambient else inherited
        counts = work(ambient)
        require(not prior['spatial_policy'] and prior['illumination']=='known_L2', 'Unexpected prior recipe')
        arms[arm] = dict(prior, work=counts, limits=dict(prior['limits'],**counts), geometry=geometry_name(arm),
                         trainer_sha256=sha256(trainer_source(ambient).encode()).hexdigest())
    return dict(protocol=PROTOCOL, host_output=str(out), gate3=g, gate3_receipt=manifest['receipt'],
                gate3_outputs=manifest['native_outputs'], arms=arms, order=list(ARMS), support=support,
                base_initial=parent['base_initial'], ambient_initial=parent['arms']['Final-Ambient']['initial'],
                policy=POLICY, shader=SHADER, steps=STEPS, foreground_seconds=FOREGROUND_SECONDS,
                reference_freeze_sha256=old.REFERENCE_FREEZE, sources={key:digest(ROOT/key) for key in SOURCES},
                history_checkpoint_loads_authorized=False, historical_geometry_provenance='unresolved')


def freeze(path, sha):
    require(path == ROOT/FREEZE and path.resolve()==path and digest(path)==sha, 'Matched recovery freeze differs')
    f = rgb.geo.h.read(path)
    out = Path(f['host_output'])
    require(out.resolve()==out and out.parent==Path('/home/isavand/paper/evidence')
            and out.name.startswith('donut-matched-geometry-gate4-') and f==definition(out,f['support']),
            'Geometry recovery definition differs')
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL, approved=True, commit=commit, freeze_sha256=sha, host_output=f['host_output'],
                order=f['order'], recipes=f['arms'], image_id=IMAGE, maximum_execution_launches=1,
                researcher_admitted_dataset=True, benchmark_generation_sha256=GENERATION, dataset_admission_sha256=ADMISSION,
                selected_gpu='0', foreground_seconds=FOREGROUND_SECONDS, gate3_receipt_sha256=GATE3_RECEIPT,
                historical_geometry_provenance='unresolved')


def paths(f, native=False):
    if native:
        return {k:Path(v) for k,v in MOUNTS.items()}
    return dict(rgb.paths(f['gate3']), gate3=Path(f['gate3_receipt']['path']).parent,
                initial=Path(f['base_initial']['path']), ambient_initial=Path(f['ambient_initial']['path']))


def common_rows(view, c, primary):
    # Oracle positivity is the admitted shader mask. Keep matching transfer
    # pixels in source order and require positive reconstructed facing too.
    candidate = c['pixel'][c['label_rows']]
    common = np.intersect1d(view['pixel_indices'][view['shader_valid']],candidate,assume_unique=True)
    normal = primary['reconstruction_normal'][common]
    facing = -(normal*primary['direction'][common]).sum(1)
    return common[facing>0]


def common_frame(oracle, reconstruction, q):
    rotation = rgb.geo.minimal_rotation(reconstruction['normals'],oracle['normals'])
    return rgb.geo.rotate_tensor(q.astype(np.float64),rotation)


def support_identity(data):
    h = sha256()
    for camera in PANEL:
        v = data['oracle'][camera]
        r = data['reconstruction'][camera]
        for key in ('pixel_indices','linear_components','outgoing'):
            np.testing.assert_array_equal(v[key],r[key])
        h.update(np.array([camera,len(v['pixel_indices'])],dtype='<u4').tobytes())
        h.update(v['pixel_indices'].astype('<u4').tobytes())
        h.update(v['linear_components'][:,2].astype('<f8').tobytes())
    return dict(camera_counts={str(k):v for k,v in COUNTS.items()}, observations_and_RGB_sha256=h.hexdigest(),
                fitting_observations=sum(COUNTS[c] for c in SPLITS['fitting']),
                held_out_observations=sum(COUNTS[c] for c in SPLITS['held_out']),
                photometric_index_sha256=bell.dense.index_hash(sum(COUNTS[c] for c in SPLITS['fitting'])))


def admit(f,p, *, check_support=True):
    from benchmarks.bell_geometry_e11 import mesh_arrays

    g = f['gate3']
    geometry, source = rgb.admit(g,p)
    generation = rgb.geo.h.read(p['reference']/'generation.json')
    rgb.geo.h.check_artifact(p['initial'],f['base_initial'])
    rgb.geo.h.check_artifact(p['ambient_initial'],f['ambient_initial'])
    receipt = rgb.geo.h.read(rgb.geo.h.check_artifact(p['gate3']/'receipt.json',f['gate3_receipt']))
    require(receipt['status']=='complete' and receipt['cleanup_verified'] and receipt['outputs']==f['gate3_outputs']
            and receipt['backend']==g['backend'] and receipt['freeze_sha256']==GATE3_SHA
            and receipt['optimizer_updates']==receipt['primary_rays']==0, 'Gate 3 capture admission failed')
    fields = (*TARGET_KEYS,'pixel_indices','position','outgoing','shader_valid','linear_components','material_region','triangle')
    views = {arm:{} for arm in rgb.ARMS}
    compatibility, coverage, all_files = {}, {}, set()
    total = 0
    for camera, scene in source.items():
        def fetch(arm, selected, size, prefix):
            name = capture_name(camera,arm,prefix,size)
            require(name not in all_files, 'Duplicate Gate 3 capture')
            all_files.add(name)
            return numeric(rgb.geo.h.check_artifact(p['gate3']/name,receipt['outputs'][name]))
        states, refined = rgb.paired(scene['views'],fetch,total)
        total += len(refined)
        images,_ = rgb.raster(scene,states)
        require(refined==receipt['views'][str(camera)]['refined_pixels'], 'Gate 3 refinement replay differs')
        with np.load(p['reference']/f'view-{camera}.npz',allow_pickle=False) as z:
            oracle = {key:z[key] for key in fields}
        with np.load(p['gate2']/f'primary-{camera}.npz',allow_pickle=False) as z:
            primary = {key:z[key] for key in ('direction','oracle_hit','reconstruction_hit','reconstruction_normal','archived_hit')}
        c = numeric(p['gate2']/f'correspondence-{camera}.npz')
        pixels = common_rows(oracle,c,primary)
        require(len(pixels)==COUNTS[camera] and (camera!=1 or 8239 in pixels), 'Matched observation inventory differs')
        original = np.searchsorted(oracle['pixel_indices'],pixels)
        mapped = np.searchsorted(c['pixel'][c['label_rows']],pixels)
        mask = np.zeros(rgb.geo.SIZE**2,bool)
        mask[pixels] = True
        o = {key:value[original] for key,value in oracle.items()}
        ri = np.searchsorted(scene['views']['reconstruction']['pixel_indices'],pixels)
        r = {key:value[ri] for key,value in scene['views']['reconstruction'].items()}
        r.update(Q_world=c['transported_world_Q'][mapped], material_region=c['labels_material_region'][mapped],
                 triangle=oracle['triangle'][original], linear_components=o['linear_components'].copy())
        np.testing.assert_array_equal(o['outgoing'], r['outgoing'])
        for arm,v in zip(rgb.ARMS,(o,r)):
            # Output metadata retains actual geometry hits, while native exports
            # separately record the supported rendering mask.
            views[arm][camera] = dict(v,hit_mask=primary[arm+'_hit'].reshape(rgb.geo.SIZE,rgb.geo.SIZE))
            require(v['shader_valid'].all() and np.array_equal(v['pixel_indices'],pixels), 'Arm pixel parity differs')
        coverage[camera] = dict(pool=mask, oracle_hit=primary['oracle_hit'], reconstruction_hit=primary['reconstruction_hit'],
                                defined_reconstruction=scene['known'], **scene['cohorts'])
        coverage[camera]['oracle_positive'] = np.isin(np.arange(len(mask)),oracle['pixel_indices'][oracle['shader_valid']])
        coverage[camera]['reconstruction_positive'] = primary['reconstruction_hit'] & (
            -(primary['reconstruction_normal']*primary['direction']).sum(1)>0)
        for reason in rgb.geo.REASONS:
            row = np.zeros_like(mask); row[c['pixel']]=c[reason]; coverage[camera][reason] = row
        for arm in rgb.ARMS:
            compatibility[arm,camera] = images[arm][:,pixels,2]
    require(all_files == set(receipt['outputs']) and total == receipt['refined_pixels_total'], 'Gate 3 capture inventory differs')
    support = support_identity(views)
    if check_support:
        require(support==f['support'], 'Matched observation or fitting stream hash differs')
    return generation, views, compatibility, coverage, dict(oracle=geometry,
           reconstruction=dict(zip(('vertices_native','triangles'),mesh_arrays(p['mesh']))))


def completed(batch, arm, f, commit, sha, authorization_sha):
    root = batch/arm
    r = rgb.geo.h.read(root/'recovery.json')
    recipe = f['arms'][arm]
    require((batch/f'{arm}-exit-code.txt').read_text().strip()=='0' and r['status']=='complete'
            and r['scientific_work_complete'] and r['cleanup_verified'] and r['historical_arm_initial_parity_verified']
            and r['arm']==arm and r['commit']==commit and r['freeze_sha256']==sha
            and r['authorization_sha256']==authorization_sha and r['protocol']==PROTOCOL
            and r['geometry']==recipe['geometry'] and r['matched_support']==f['support']
            and r['reference_Gate3_receipt_sha256']==GATE3_RECEIPT and r['counters']==recipe['work']
            and r['optimizer_returns']==10000 and [step['step'] for step in r['saved_endpoints']]==STEPS
            and r['known_lighting_audit']['learned_module_calls']==0, 'Incomplete or foreign geometry history')
    if recipe['representation'] is not None:
        require(r['paired_initial_gate']['passed'] and r['physical_tensor_admission']['passed']
                and r['representation']==recipe['representation'] and
                r['effective_regularization']=='legacy_scalar_energy_specular_white_no_surface_Q_v1',
                'Ambient recipe parity differs')
    stream = bell.dense.STREAM
    expected = dict(bytes=20480000,sha256=f['support']['photometric_index_sha256'])
    require(r['outputs'][stream]==expected, 'Native fitting stream identity differs')
    rgb.geo.h.check_artifact(root/stream,expected)
    return r
