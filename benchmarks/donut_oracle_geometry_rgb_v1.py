"""Immutable Gate 2 inputs and paired refinement for the fixed-material comparison."""
import csv
from pathlib import Path

import numpy as np

from benchmarks import donut_reconstructed_geometry_v1 as geo
from benchmarks.bell_e11 import convergence
from benchmarks.donut_recovery_v1 import camera_rays, numeric
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest


PROTOCOL = 'donut-geometry-fixed-material-rgb-v1'
FREEZE = 'docs/protocols/DONUT_ORACLE_GEOMETRY_RGB_FLOOR_V1.freeze.json'
GATE2 = ROOT/'docs/results/publication_gate2'
GATE2_SHA = '25b53011c62ae5bb36539c813178ef3c0c1b6b9217dc8283e6f5ca3c5e1f7cc9'
MANIFEST_SHA = 'd87ed9189da0d9af3c2f5c0cc108ad79564b17106a92074f28cd911a925caca0'
ARMS = ('oracle', 'reconstruction')
BASE = (256, 512)
HIGH = (1024, 2048)
COUNTS = dict(oracle=22630, reconstruction=22520)
MAX_VIEW, MAX_TOTAL = 32, 128
BASE_COST = sum(COUNTS.values())*sum(n*n for n in BASE)
HIGH_COST = 2*MAX_TOTAL*sum(n*n for n in HIGH)
LIMITS = dict(shadow_directions=BASE_COST+HIGH_COST, BRDF_point_directions=BASE_COST+HIGH_COST)
RESOURCES = dict(cpu_threads=4, host_memory_gib=64, pids=128, allocator_fraction=.25,
                 output_bytes=512*1024**2, watchdog_seconds=1800, kill_grace_seconds=30, inner_seconds=1740)
MOUNTS = dict(geo.MOUNTS, gate2='/inputs/geometry-audit')
FIELDS = ('axis', 'widths_major_minor', 'color', 'metallic')
SOURCES = (
    'benchmarks/donut_oracle_geometry_rgb_v1.py', 'experiments/donut_oracle_geometry_rgb_v1.py',
    'experiments/verify_donut_oracle_geometry_rgb_v1.py', 'experiments/run_donut_oracle_geometry_rgb_v1.sh',
    'tests/scientific/test_donut_oracle_geometry_rgb_v1.py', 'docs/protocols/DONUT_ORACLE_GEOMETRY_RGB_FLOOR_V1.md',
    'docs/results/publication_gate2/artifact-manifest.json', 'docs/results/publication_gate2/geometry-metrics.json',
    'docs/results/publication_gate2/depth-outliers.csv', 'experiments/generate_donut_heterogeneous_v1.py',
    'benchmarks/donut_heterogeneous_v2.py', 'benchmarks/bell_e11.py', 'benchmarks/bell_reference_e11.py',
    'benchmarks/beckmann_reflectance_v2.py', 'experiments/bell_highlight_pilot_e11d.py',
    'experiments/bell_quadrature_e11e.py',
)


def definition(out):
    g = geo.freeze(GATE2_SHA)
    path = GATE2/'artifact-manifest.json'
    require(digest(path) == MANIFEST_SHA, 'Accepted Gate 2 manifest differs')
    manifest = geo.h.read(path)
    for name in ('geometry-metrics.json', 'depth-outliers.csv'):
        geo.h.check_artifact(GATE2/name, manifest['packaged_files'][name])
    return dict(protocol=PROTOCOL, schema_version=1, host_output=str(out), geometry_audit=g,
                gate2_receipt=manifest['receipt'], gate2_outputs=manifest['native_outputs'],
                gate2_manifest_sha256=MANIFEST_SHA, backend=g['backend'], panel=geo.PANEL, splits=geo.SPLITS,
                tolerances=geo.h.TOLERANCES, baseline_grids=list(BASE), refined_grids=list(HIGH),
                max_refined_per_view=MAX_VIEW, max_refined_total=MAX_TOTAL, integrated_counts=COUNTS,
                illumination=geo.h.ENVIRONMENT, indirect='zero', point_light='absent',
                limits=LIMITS, resources=RESOURCES, sources={name: digest(ROOT/name) for name in SOURCES},
                historical_geometry_provenance='unresolved', training_authorized=False)


def freeze(sha):
    p = ROOT/FREEZE
    require(digest(p) == sha, 'Fixed-material RGB freeze differs')
    f = geo.h.read(p)
    out = Path(f['host_output'])
    require(out.resolve() == out and out.parent == Path('/home/isavand/paper/evidence')
            and out.name.startswith('donut-geometry-rgb-gate3-') and f == definition(out), 'Frozen RGB definition differs')
    return f


def authorization(f, commit, sha):
    return dict(protocol=PROTOCOL, approved=True, commit=commit, freeze_sha256=sha, host_output=f['host_output'],
                backend=f['backend'], gate2_receipt_sha256=f['gate2_receipt']['sha256'], selected_gpu='0',
                limits=LIMITS, resources=RESOURCES, maximum_execution_launches=1, training_authorized=False)


def paths(f, native=False):
    if native:
        return {key: Path(value) for key, value in MOUNTS.items()}
    return dict(geo.paths(f['geometry_audit']), gate2=Path(f['gate2_receipt']['path']).parent)


def admit(f, p):
    geometry, meta, views = geo.admit(f['geometry_audit'], p)
    receipt = geo.h.read(geo.h.check_artifact(p['gate2']/'receipt.json', f['gate2_receipt']))
    inventory = f['gate2_outputs']
    require(receipt['status'] == 'complete' and receipt['cleanup_verified'] and all(receipt['gates'].values())
            and receipt['view_products'] == geo.PANEL and receipt['backend'] == f['backend']
            and receipt['outputs'] == inventory and len(inventory) == 18, 'Incomplete Gate 2 evidence')
    for name, item in inventory.items():
        geo.h.check_artifact(p['gate2']/name, item)
    with (GATE2/'depth-outliers.csv').open(newline='') as handle:
        outliers = list(csv.DictReader(handle))
    data = {}
    for camera, view in views.items():
        path = p['reference']/f'view-{camera}.npz'
        with np.load(path, allow_pickle=False) as z:
            reference = {key: z[key] for key in (*FIELDS, 'linear_components', 'near_boundary')}
        # Miss payloads from native tracing are uninterpreted. Restrict readout to hit masks below.
        with np.load(p['gate2']/f'primary-{camera}.npz', allow_pickle=False) as z:
            primary = {key: z[key] for key in z.files}
        c = numeric(p['gate2']/f'correspondence-{camera}.npz')
        data[camera] = prepare(camera, meta, view, reference, primary, c, outliers)
    require({key: sum(len(v['views'][key]['pixel_indices']) for v in data.values()) for key in ARMS} == COUNTS
            and sum((~v['known']).sum() for v in data.values()) == 61
            and sum(v['cohorts']['depth_outlier'].sum() for v in data.values()) == len(outliers) == 10,
            'Ray, transfer or outlier inventory differs')
    return geometry, data


def prepare(camera, meta, view, reference, primary, c, outliers):
    o, r = (primary[key+'_hit'] for key in ARMS)
    org, direction, _ = camera_rays(meta, camera)
    np.testing.assert_array_equal(org.astype('f4'), primary['origin'])
    np.testing.assert_array_equal(direction.astype('f4'), primary['direction'])
    np.testing.assert_array_equal(o, view['hit_mask'].ravel())
    np.testing.assert_array_equal(c['pixel'], np.flatnonzero(r))
    valid = c['valid']
    np.testing.assert_array_equal(c['label_rows'], np.flatnonzero(valid))
    np.testing.assert_array_equal(valid, ~np.logical_or.reduce([c[key] for key in geo.REASONS]))
    ids = view['pixel_indices']
    a = view['shader_valid']
    np.testing.assert_array_equal(primary['oracle_position'][ids], view['position'])
    np.testing.assert_array_equal(primary['oracle_smooth_normal'][ids], view['normals'])
    # Replay the admitted camera comparison allowance and retain the original
    # float64 vectors wherever the oracle archive supplies them.
    np.testing.assert_allclose(view['outgoing'], -direction[ids], rtol=0, atol=2e-12)
    outgoing = -direction
    outgoing[ids] = view['outgoing']
    rec = c['pixel'][valid]
    rec_n = primary['reconstruction_normal'][rec]
    rec_view = dict(position=primary['reconstruction_position'][rec].astype('f8'), normals=rec_n,
                    outgoing=outgoing[rec], axis=c['transported_axis'], pixel_indices=rec,
                    shader_valid=np.ones(len(rec), bool), **{key: c['labels_'+key] for key in FIELDS[1:]})
    oracle_view = dict(pixel_indices=ids[a], shader_valid=np.ones(a.sum(), bool), position=view['position'][a],
                       normals=view['normals'][a], outgoing=view['outgoing'][a],
                       **{key: reference[key][a] for key in FIELDS})
    axis = rec_view['axis']
    other = np.cross(rec_n, axis)
    width = rec_view['widths_major_minor']**2
    Q = width[:, 0, None, None]*np.einsum('ni,nj->nij', axis, axis)+width[:, 1, None, None]*np.einsum('ni,nj->nij', other, other)
    np.testing.assert_allclose(Q, c['transported_world_Q'], rtol=0, atol=geo.TOL['transport_width'])
    require(all(np.isfinite(x).all() and x.dtype == np.float64 for v in (oracle_view,rec_view)
                for key,x in v.items() if key not in ('pixel_indices','shader_valid'))
            and ((rec_n*rec_view['outgoing']).sum(1) > 0).all(), 'Saved shader inputs differ')
    known = ~r.copy()
    known[rec] = True
    boundary = np.zeros(len(o), bool)
    boundary[ids] = reference['near_boundary']
    boundary[rec] |= c['labels_near_boundary']
    boundary[c['pixel']] |= c['boundary_side_ambiguity']
    fo = -(primary['oracle_smooth_normal']*direction).sum(1)
    fr = -(primary['reconstruction_normal']*direction).sum(1)
    grazing = (o & (fo > 0) & (fo <= geo.TOL['grazing_cosine'])) | (r & (fr > 0) & (fr <= geo.TOL['grazing_cosine']))
    depth = o & r & (abs(primary['reconstruction_depth']-primary['oracle_depth']) > .05)
    selected = [row for row in outliers if int(row['camera']) == camera]
    np.testing.assert_array_equal(np.flatnonzero(depth), [int(row['pixel']) for row in selected])
    for row in selected:
        j = np.searchsorted(c['pixel'], int(row['pixel']))
        for current, saved in (('valid','transfer_valid'), ('boundary_side_ambiguity','material_side_ambiguous'),
                               ('oracle_hidden','oracle_hidden')):
            require(c[current][j] == bool(int(row[saved])), 'Depth-outlier transfer classification changed')
    saved = np.zeros((len(o), 3, 3), np.float64)
    saved[ids] = reference['linear_components']
    require((saved[ids[~a]] == 0).all(), 'Archived one-sided-zero convention differs')
    return dict(views=dict(oracle=oracle_view, reconstruction=rec_view), known=known, archived=saved,
                cohorts=dict(common=o&r, oracle_only=o&~r, reconstruction_only=r&~o, background=~o&~r,
                             boundary=boundary, grazing=grazing, degenerate_adjacent=primary['near_degenerate'],
                             depth_outlier=depth), outliers=selected)


def verify_grid(view, size, capture):
    n = len(view['pixel_indices'])
    rgb, count = capture['components'], capture['visibility']
    require(set(capture) == {'components', 'visibility', 'pixels', 'grid'} and int(capture['grid']) == size
            and np.array_equal(capture['pixels'], view['pixel_indices']) and rgb.shape == (n,3,3)
            and rgb.dtype == np.float64 and np.isfinite(rgb).all() and (rgb >= 0).all()
            and np.array_equal(rgb[:,0]+rgb[:,1], rgb[:,2]) and count.shape == (n,)
            and count.dtype.kind in 'iu' and ((count >= 0)&(count <= size**2)).all(), 'Invalid quadrature grid capture')


def choose(states, done):
    candidates = []
    for n, arm in enumerate(ARMS):
        pixels, rgb = states[arm]['pixels'], states[arm]['rgb'][:, :, 2]
        gate = convergence(rgb[1], rgb[0])
        if gate['passed']:
            continue
        d = rgb[1]-rgb[0]
        abs_error = abs(d).max(1)
        absolute = abs_error > geo.h.TOLERANCES['convergence_rgb_abs']
        refined = np.isin(pixels, done)
        require(not (absolute & refined).any(), 'Refined absolute RGB convergence failed')
        rank = abs_error if absolute.any() else np.square(d).sum(1)
        eligible = np.flatnonzero(~refined & (absolute if absolute.any() else True))
        require(len(eligible), 'Refinement exhausted without pooled RGB convergence')
        i = eligible[np.lexsort((pixels[eligible], -rank[eligible]))[0]]
        candidates.append((int(not absolute.any()), -float(rank[i]), int(pixels[i]), n))
    return min(candidates)[2] if candidates else None


def paired(view, fetch, total):
    states, done = {}, []
    def capture(arm, selected, grids, prefix):
        pair = [fetch(arm, selected, n, prefix) for n in grids]
        for n, a in zip(grids, pair):
            verify_grid(selected, n, a)
        return {key: np.stack([a[key] for a in pair]) for key in ('components','visibility')}
    for arm, v in view.items():
        values = capture(arm, v, BASE, 'base')
        states[arm] = dict(pixels=v['pixel_indices'], rgb=values['components'], visibility=values['visibility'],
                           grids=np.broadcast_to(np.array(BASE)[:,None], values['visibility'].shape).copy())
    while True:
        pixel = choose(states, done)
        if pixel is None:
            return states, done
        require(len(done) < MAX_VIEW and total+len(done) < MAX_TOTAL, 'Paired refinement budget exhausted')
        for arm, v in view.items():
            row = np.flatnonzero(v['pixel_indices'] == pixel)
            if len(row):
                selected = {key: value[row] for key,value in v.items()}
                values = capture(arm, selected, HIGH, f'pixel-{pixel}')
                states[arm]['rgb'][:, row] = values['components']
                states[arm]['visibility'][:, row] = values['visibility']
                states[arm]['grids'][:, row] = np.array(HIGH)[:,None]
        done.append(pixel)


def raster(camera, states):
    up = {key: np.zeros((2, len(camera['known']), 3, 3), dtype=np.float64) for key in ARMS}
    for arm, s in states.items():
        up[arm][:, s['pixels']] = s['rgb']
    oracle = up['oracle'][1,:,2]
    gate = convergence(oracle, camera['archived'][:,2])
    require(gate['passed'], 'Fresh oracle radiance differs from admitted independent reference')
    gates = {arm: convergence(s['rgb'][1,:,2], s['rgb'][0,:,2]) for arm,s in states.items()}
    require(all(value['passed'] for value in gates.values()), 'Incomplete paired integration convergence')
    return up, dict(oracle_replay=gate, convergence=gates)
