"""Frozen heterogeneous Donut continuation and RGB-only refinement rules."""
from pathlib import Path

import numpy as np

from benchmarks import donut_heterogeneous_v1 as v1
from benchmarks.bell_e11 import convergence
from benchmarks.donut_recovery_v1 import numeric
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest

PROTOCOL = 'donut-kurt-heterogeneous-reference-v2'
FREEZE = 'docs/protocols/DONUT_HETEROGENEOUS_REFERENCE_V2.freeze.json'
V1_SHA = '330eafdac4dc70555cd8d5d1e6da54d2a92b91b9047f89d8c7f04329fc90e4e1'
FAILURE = 'docs/results/donut_heterogeneous_reference_failure_v1/analysis.json'
PROBE_PIXELS = [7317,7955,6577]  # witness (21,57), next residual (19,62), first ordinary
HIGH = (1024,2048)
BASE = v1.GRIDS
PROBE_CAMERA = 50
REUSED = (0,50)
NEW_POINTS = sum(n for c,n in zip(v1.PANEL,v1.COUNTS) if c not in REUSED)
MAX_PER_VIEW, MAX_TOTAL = 32,128
HIGH_COST = sum(n*n for n in HIGH)
NEW_BASE_COST = NEW_POINTS*sum(n*n for n in BASE)
LIMITS = dict(v1.LIMITS, shadow_directions=NEW_BASE_COST+(len(PROBE_PIXELS)+MAX_TOTAL)*HIGH_COST,
              BRDF_point_directions=NEW_BASE_COST+(len(PROBE_PIXELS)+MAX_TOTAL)*HIGH_COST,
              slope_nodes_attempted=NEW_POINTS*v1.NODES_PER_POINT,
              slope_nodes_returned=NEW_POINTS*v1.NODES_PER_POINT)
POLICY = dict(diagnostic_camera=PROBE_CAMERA, diagnostic_pixels=PROBE_PIXELS,
              absolute_rank='descending maximum RGB channel difference, then ascending pixel ID',
              relative_rank='descending squared RGB difference, then ascending pixel ID',
              high_grids=list(HIGH), baseline_grids=list(BASE), camera0='byte copy',
              max_refined_per_view=MAX_PER_VIEW, max_refined_total=MAX_TOTAL,
              max_refinements_per_pixel=1, relative_scope='entire valid view',
              diagnostic_scope='three-point combined RGB panel', coarse_visibility_missing=-1)
WORK = ('primary_rays','CPU_primary_crosschecks','shadow_directions','BRDF_point_directions',
        'slope_nodes_attempted','slope_nodes_returned')


def freeze(path, sha):
    require(path.resolve()==path and digest(path)==sha, 'Heterogeneous v2 freeze identity differs')
    f = v1.read(path)
    require(f['protocol']==PROTOCOL and f['refinement']==POLICY and f['limits']==LIMITS
            and f['image']==v1.IMAGE and f['v1_freeze_sha256']==V1_SHA
            and f['tolerances']==v1.TOLERANCES, 'Frozen v2 definition differs')
    for name, hash_ in f['sources'].items():
        p=ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==hash_, 'Frozen v2 source differs: '+name)
    return f


def authorization(f, commit, sha):
    return dict(approved=True, protocol=PROTOCOL, commit=commit, freeze_sha256=sha, image=v1.IMAGE,
                selected_gpu='0', limits=LIMITS, maximum_execution_launches=1, host_output=f['host_output'],
                retained_failed_generation_sha256=f['failed_records']['reference/generation.json']['sha256'],
                v1_freeze_sha256=V1_SHA, training_authorized=False)


def failed_inputs(f, prior):
    require(prior.is_absolute() and prior.resolve()==prior, 'Explicit failed v1 batch required')
    for name, item in f['failed_records'].items():
        require(prior in (prior/name).parents, 'Invalid failed artifact relative path')
        v1.check_artifact(prior/name,item)
    v1_f = v1.freeze(prior/'freeze.json',V1_SHA)
    r = v1.read(prior/'reference/generation.json')
    require(r['status']=='failed' and not r['scientific_work_complete'] and r['freeze_sha256']==V1_SHA
            and [v['camera'] for v in r['views']]==[0,50] and r['views'][0]['radiance_convergence']['passed']
            and not r['views'][1]['radiance_convergence']['passed']
            and (prior/'exit-code.txt').read_text().strip()=='1', 'Expected the preserved camera 50 failure')
    views = {c:numeric(prior/f'reference/view-{c}.npz') for c in REUSED}
    return v1_f,views


def source(f, prior, paths):
    v1_f, prior_views = failed_inputs(f,prior)
    geometry, geom_views = v1.source(v1_f,*paths)
    labels,radius,area = v1.labels(geometry,geom_views)
    np.testing.assert_array_equal(numeric(prior/'reference/surface-area.npz')['triangle_area_native'],area)
    for camera, captured in prior_views.items():
        for key, truth in labels[camera].items():
            require(captured[key].dtype==truth.dtype and captured[key].shape==truth.shape,
                    'Failed-run geometry/label schema differs')
            if key in v1.GEOMETRY_KEYS or truth.dtype.kind!='f':
                np.testing.assert_array_equal(captured[key],truth)
            else:
                np.testing.assert_allclose(captured[key],truth,atol=2e-12,rtol=0)
    return geometry, labels, radius, prior_views


def pixel_rows(view, pixels):
    ids = view['pixel_indices']
    rows = np.searchsorted(ids,pixels)
    require((rows<len(ids)).all() and np.array_equal(ids[rows],pixels) and view['shader_valid'][rows].all(),
            'Missing valid frozen diagnostic pixel')
    return rows


def rgb_gate(view):
    mask = view['shader_valid']
    return convergence(view['linear_components'][mask,2],view['coarse_components'][mask,2])


def next_row(view, refined):
    """Refine an absolute offender first, otherwise the largest pooled contributor."""
    gate = rgb_gate(view)
    if gate['passed']:
        return None
    delta = view['linear_components'][:,2]-view['coarse_components'][:,2]
    absolute = view['shader_valid'] & (abs(delta).max(1) > v1.TOLERANCES['convergence_rgb_abs'])
    require(not (refined&absolute).any(), 'Refined absolute RGB convergence failed')
    candidates = view['shader_valid']&~refined
    if absolute.any():
        candidates &= absolute
        scores = abs(delta).max(1)
    else:
        scores = np.square(delta).sum(1)
    rows = np.flatnonzero(candidates)
    require(len(rows)>0, 'No unrefined observations available for pooled RGB convergence')
    return int(rows[np.lexsort((view['pixel_indices'][rows],-scores[rows]))[0]])


def counters(new_high_points):
    require(0<=new_high_points<=MAX_TOTAL, 'Refinement count exceeds cap')
    directions = NEW_BASE_COST+(len(PROBE_PIXELS)+new_high_points)*HIGH_COST
    return dict(primary_rays=0,CPU_primary_crosschecks=0,shadow_directions=directions,
                BRDF_point_directions=directions,
                slope_nodes_attempted=LIMITS['slope_nodes_attempted'],
                slope_nodes_returned=LIMITS['slope_nodes_returned'])


def check_capture(pixels, capture):
    count = len(pixels)
    require(set(capture)=={'pixels','grids','components','incident_reflectance','visible_counts'}
            and np.array_equal(capture['pixels'],pixels)
            and np.array_equal(capture['grids'],HIGH), 'High-grid capture identity differs')
    rgb,e,n = (capture[k] for k in ('components','incident_reflectance','visible_counts'))
    require(rgb.shape==(2,count,3,3) and e.shape==(2,count,3) and n.shape==(2,count)
            and rgb.dtype==e.dtype==np.float64 and n.dtype.kind in 'iu'
            and np.isfinite(rgb).all() and np.isfinite(e).all() and (rgb>=0).all() and (e>=0).all()
            and np.array_equal(rgb[:,:,0]+rgb[:,:,1],rgb[:,:,2])
            and ((n>=0)&(n<=np.square(HIGH)[:,None])).all(), 'Invalid high-grid component or visibility diagnostics')
    return capture


def diagnostic_gates(capture):
    checked = check_capture(np.asarray(PROBE_PIXELS),capture)
    return {name:convergence(value[1],value[0]) for name,value in
            [('RGB',checked['components'][:,:,2]),('diffuse',checked['components'][:,:,0]),
             ('specular',checked['components'][:,:,1]),('unoccluded_incident_reflectance',checked['incident_reflectance'])]}


def apply_patch(view, quadrature, row, capture):
    check_capture(view['pixel_indices'][[row]],capture)
    require(quadrature['fine_grid'][row]==BASE[1], 'Duplicate or invalid refinement')
    for fields,name in ((('coarse_components','linear_components'),'components'),
                        (('coarse_incident_reflectance','incident_reflectance'),'incident_reflectance')):
        for grid,field in enumerate(fields):
            view[field][row] = capture[name][grid,0]
    view['visible_counts'][row] = capture['visible_counts'][1,0]
    quadrature['coarse_visible_counts'][row] = capture['visible_counts'][0,0]
    quadrature['coarse_grid'][row],quadrature['fine_grid'][row] = HIGH


def assemble(baseline, coarse_counts, fetch):
    """Only the selected RGB/incident rows can change; return selection in order."""
    patched = ('linear_components','coarse_components','visible_counts',
               'incident_reflectance','coarse_incident_reflectance')
    view = dict(baseline,**{name:baseline[name].copy() for name in patched})
    valid = view['shader_valid']
    require(coarse_counts.shape==valid.shape and coarse_counts.dtype.kind=='i'
            and ((coarse_counts[valid]>=-1)&(coarse_counts[valid]<=BASE[0]**2)).all()
            and (coarse_counts[~valid]==0).all(), 'Invalid coarse visibility inventory')
    quadrature = dict(pixel_indices=view['pixel_indices'].copy(),
                      coarse_grid=np.where(valid,BASE[0],0),fine_grid=np.where(valid,BASE[1],0),
                      coarse_visible_counts=coarse_counts.copy())
    selected = []
    while True:
        row = next_row(view,quadrature['fine_grid']==HIGH[1])
        if row is None:
            quadrature['refined_pixels']=np.asarray(selected,dtype=np.int64)
            return view,quadrature
        require(len(selected)<MAX_PER_VIEW, 'Per-view refinement budget exhausted')
        captured = fetch(row)
        apply_patch(view,quadrature,row,captured)
        selected.append(int(view['pixel_indices'][row]))


def reused_coarse_counts(view):
    return np.where(view['shader_valid'],POLICY['coarse_visibility_missing'],0).astype(np.int64)


def probe_slice(panel, pixel):
    i = PROBE_PIXELS.index(pixel)
    return dict(pixels=panel['pixels'][i:i+1],grids=panel['grids'],
                **{k:panel[k][:,i:i+1] for k in ('components','incident_reflectance','visible_counts')})
