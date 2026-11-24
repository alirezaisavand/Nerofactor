"""Replay the recorded dense Bell camera-127 convergence stop without rendering."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_dense_reference_v1 as d
from benchmarks import donut_heterogeneous_v2 as refinement
from benchmarks.donut_recovery_v1 import json_new
from experiments.generate_donut_heterogeneous_v1 import compare
from experiments.verify_bell_heterogeneous_v1 import compare_geometry_metadata
from experiments.verify_donut_heterogeneous_v2 import view_identity
from infrastructure.signatures import SignatureError, require


b = d.b
CAMERA = 127


def residual(view):
    valid = view['shader_valid']
    delta = view['linear_components'][valid] - view['coarse_components'][valid]
    norm = np.linalg.norm(delta, axis=(0,2))
    absolute = np.max(abs(delta[:,2]), axis=1)
    return dict(view=compare(view), absolute_offenders=int((absolute>b.TOLERANCES['convergence_rgb_abs']).sum()),
                diffuse_residual_L2=float(norm[0]), specular_residual_L2=float(norm[1]),
                RGB_residual_L2=float(norm[2]))


def replay(out, baseline, counts, recorded):
    seen = []

    def fetch(row):
        pixel = int(baseline['pixel_indices'][row])
        seen.append(pixel)
        return b.numeric(out/f'patch-{CAMERA}-{pixel}.npz')

    try:
        refinement.assemble(baseline, counts, fetch)
    except SignatureError as error:
        require(str(error)=='Per-view refinement budget exhausted', 'Different dense Bell failure: '+str(error))
    else:
        raise SignatureError('Archived dense Bell stop did not replay')
    require(len(seen)==recorded==b.MAX_PER_VIEW and len(set(seen))==recorded, 'Unexpected patch count')
    view = {name: value.copy() for name, value in baseline.items()}
    quadrature = dict(coarse_grid=np.full(len(counts), b.GRIDS[0]), fine_grid=np.full(len(counts), b.GRIDS[1]),
                      coarse_visible_counts=counts.copy())
    sequence = []
    for pixel in seen:
        row = refinement.next_row(view, quadrature['fine_grid']==refinement.HIGH[1])
        require(int(baseline['pixel_indices'][row])==pixel, 'Baseline selection replay differs')
        refinement.apply_patch(view, quadrature, row, b.numeric(out/f'patch-{CAMERA}-{pixel}.npz'))
        sequence.append(dict(pixel=pixel, **refinement.rgb_gate(view)))
    return view, quadrature, sequence


def audit(sha, destination):
    f = d.freeze(b.ROOT/d.FREEZE, sha)
    batch = Path(f['host_output'])
    require(destination.is_absolute() and destination.resolve()==destination and destination.parent.is_dir()
            and batch not in destination.parents and all(p not in destination.parents for p in d.host_paths(f)),
            'Separate local audit output required')
    auth = batch/'authorization.json'
    approval = b.read(auth)
    result = batch/'reference/generation.json'
    r = b.read(result)
    require(approval==d.authorization(f,approval['commit'],sha)
            and b.digest(auth)==b.digest(batch.with_name(batch.name+'-authorization.json'))
            and b.digest(batch/'freeze.json')==sha and r['freeze_sha256']==sha
            and r['authorization_sha256']==b.digest(auth) and r['commit']==approval['commit']
            and r['protocol']==d.PROTOCOL and r['image']==b.IMAGE and r['backend']==b.identity()
            and r['input_records']==f['input_records'] and r['reused_sparse_records']==f['sparse']['outputs']
            and r['status']=='failed' and r['phase']==f'camera_{CAMERA}'
            and r['error']=='SignatureError: Per-view refinement budget exhausted'
            and not r['scientific_work_complete'] and r['cleanup_verified'] and not r['views']
            and r['native_model_calls']==r['optimizer_returns']==0
            and r['counters']==b.counters(d.COUNTS[CAMERA],r['refined_points'])
            and (batch/'exit-code.txt').read_text().strip()=='1', 'Unexpected native dense Bell failure receipt')
    out = result.parent
    patches = {name for name in r['outputs'] if name.startswith(f'patch-{CAMERA}-')}
    expected = {f'baseline-{CAMERA}.npz','surface-area.npz',*d.SPARSE_FILES,*patches}
    require(set(r['outputs'])==expected=={p.name for p in out.iterdir() if p.is_file()
                 and p.name not in ('generation.json','progress.json')}, 'Unexpected failed batch inventory')
    for name, item in r['outputs'].items():
        b.check_artifact(out/name,item)
    require({name:r['outputs'][name] for name in d.SPARSE_FILES}==f['sparse']['outputs'],
            'Sparse evidence differs')
    geometry, labels, _, _ = d.source(f,*d.host_paths(f))
    np.testing.assert_array_equal(b.numeric(out/'surface-area.npz')['triangle_area_native'],geometry['area'])
    compare_geometry_metadata(r['geometry_checks'],{str(c):b.view_geometry_summary(v) for c,v in labels.items()})
    baseline = b.numeric(out/f'baseline-{CAMERA}.npz')
    counts = baseline.pop('baseline_coarse_visible_counts')
    view_identity(baseline,labels[CAMERA])
    valid = baseline['shader_valid']
    before = residual(baseline)
    view,q,sequence = replay(out,baseline,counts,r['refined_points'])
    require(patches=={f'patch-{CAMERA}-{row["pixel"]}.npz' for row in sequence}, 'Patch inventory differs')
    after = residual(view)
    eligible = q['fine_grid']==b.GRIDS[1]
    next_row = refinement.next_row(view,~eligible)
    errors = view['linear_components']-view['coarse_components']
    squared = np.square(errors[:,2]).sum(1)
    rows = np.flatnonzero(eligible)
    order = rows[np.lexsort((view['pixel_indices'][rows],-squared[rows]))[:32]]
    assumed = np.sqrt(squared[valid].sum()-np.cumsum(squared[order]))/np.linalg.norm(view['linear_components'][valid,2])
    cos = (view['outgoing'][order]*view['normals'][order]).sum(1)
    json_new(destination, dict(schema_version=1, status='verified_incomplete', experiment_batch=str(batch),
        execution_commit=r['commit'], freeze_sha256=sha, authorization_sha256=b.digest(auth),
        generation_sha256=b.digest(result), wall_seconds=r['wall_seconds'], cleanup_verified=r['cleanup_verified'],
        original_exit_code=1, native_outputs_checked=len(expected), view=CAMERA, counters=r['counters'],
        original=b.TOLERANCES, baseline=before, after_saved_patches=after, deterministic_trace=sequence,
        next_selected_pixel=int(view['pixel_indices'][next_row]), leading_32_remaining=dict(
            n_dot_view_min_median_max=np.quantile(cos,[0,.5,1]).tolist(),
            specular_to_diffuse_residual_L2=float(np.linalg.norm(errors[order,1])/np.linalg.norm(errors[order,0]))),
        scenario_fixed_fine_RGB_norm_and_zero_new_patch_residual=assumed.tolist(),
        proposed_continuation=dict(per_view_cap=64, panel_cap=b.MAX_TOTAL, saved_patches=len(sequence),
            maximum_new_patches=b.MAX_TOTAL-len(sequence),
            maximum_new_work=b.counters(sum(d.NEW_COUNTS)-d.COUNTS[CAMERA], b.MAX_TOTAL-len(sequence))),
        offline_primary_rays=0, offline_visibility_rays=0, offline_BRDF_queries=0))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    audit(args.freeze_sha256,args.output)
