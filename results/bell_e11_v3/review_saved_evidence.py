"""File-only review of the explicitly selected Bell v3 capture; no rendering.

Run from the repository root with PYTHONPATH=. and python3. All output is new;
the benchmark, its candidate admission and earlier verification remain untouched.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audits.bell_calibration_e11 import calibrations, rays
from benchmarks.bell_e11 import geometry_gate, mask_for_primary_rays, split
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, ROOT, baseline, digest
from infrastructure.verified_ray_backend import identity


REPORT_SHA = '5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7'
FREEZE_SHA = 'e32cda8967d386115dfe3c38409b742283706f469e6f86220ed531f256ed99bf'
COMMIT = '32046599ae9eb100ed3e5b0ff70ae605d4dcf3c9'


def record(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def review(batch, verification, output):
    from benchmarks.generate_bell_e11 import frozen

    root = batch / 'benchmark'
    report_path = root / 'geometry/generation.json'
    require(digest(report_path) == REPORT_SHA, 'Unexpected generation report')
    report = json.loads(report_path.read_text())
    seal = json.loads((root / 'completion.json').read_text())
    launch = json.loads((batch / 'launch.json').read_text())
    checked = json.loads(verification.read_text())
    require(checked['status'] == 'pass' and checked['generation_sha256'] == REPORT_SHA
            and len(checked['views']) == 128 and checked['scientific_queries'] == 0,
            'Explicit completed file-only verification required')
    require(report['status'] == 'complete' and report['cleanup_verified'] is True
            and report['scientific_work_complete'] is True
            and seal['generation_sha256'] == REPORT_SHA
            and seal['candidate_sha256'] == digest(root / 'geometry/oracle-admission.candidate.json')
            and seal['cleanup_verified'] is True and seal['scientific_work_complete'] is True
            and (batch / 'exit-code.txt').read_text().strip() == '0', 'Terminal evidence incomplete')
    require(not (root / 'failure.json').exists(), 'Generation failure evidence present')
    freeze = ROOT / 'docs/protocols/BELL_ORACLE_GENERATION_E11_V3.freeze.json'
    frozen(freeze, FREEZE_SHA)
    require(report['freeze_sha256'] == launch['freeze_sha256'] == FREEZE_SHA
            and report['commit'] == launch['commit'] == COMMIT
            and report['image_id'] == launch['image_id'] == identity()['image_id']
            and report['ray_backend'] == identity()
            and report['runtime']['backend']['sha256'] == identity()['binary_sha256'],
            'Launch/backend identity mismatch')
    auth_path = Path(launch['authorization'])
    require(digest(auth_path) == launch['authorization_sha256'], 'Authorization changed')
    auth = json.loads(auth_path.read_text())
    require(auth['approved'] is True and auth['commit'] == COMMIT
            and auth['freeze_sha256'] == FREEZE_SHA and auth['image_id'] == report['image_id']
            and auth['host_output_directory'] == str(batch)
            and auth['output_directory'] == '/outputs/benchmark'
            and auth['work_limits'] == report['work_limits'], 'Authorization binding mismatch')
    for name, sha in report['source_files'].items():
        require(digest(ROOT / name) == sha, 'Scientific/runtime source changed: ' + name)
    for stage in ('geometry', 'material'):
        baseline(stage, PINS[stage])  # Byte-only snapshot verification, no scientific imports.
    for item in report['inputs']:
        name = Path(item['path'])
        path = (Path(launch['source_mesh']) if str(name) == '/inputs/mesh.ply'
                else Path(launch['source_cameras']) / name.relative_to('/inputs/bell'))
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'],
                'Original input changed: ' + str(path))
    require(report['splits'] == split(), 'Ordered benchmark split changed')
    require(all(report[k] == 0 for k in
                ('native_model_calls', 'native_shader_calls', 'optimizer_returns', 'checkpoint_loads')),
            'Unexpected native scientific work')
    paths = [Path(launch['source_cameras']) / (str(i) + '-camera.pkl') for i in range(128)]
    P, K = calibrations(paths)  # Restricted numeric decoder, never pickle.load.
    with np.load(root / 'geometry/calibrations-safe.npz', allow_pickle=False) as saved:
        require(np.array_equal(P, saved['pose']) and np.array_equal(K, saved['K'])
                and np.array_equal(saved['source_camera_ids'], np.arange(128)), 'Calibration mismatch')
    geometry, energy, radiance = [], [], []
    ranges = {k: [float('inf'), -float('inf')] for k in
              ('width_major', 'width_minor', 'metallic', 'color', 'linear_rgb', 'energy')}
    eligible = undefined = 0
    index = np.floor((np.arange(128) + .5) * 800 / 128).astype(int)
    component_error = 0.
    for i in range(128):
        with np.load(root / 'geometry/rays' / (str(i) + '.npz'), allow_pickle=False) as z:
            with Image.open(Path(launch['source_cameras']) / 'masks_train' / (str(i) + '.png')) as image:
                mask = mask_for_primary_rays(np.asarray(image)[np.ix_(index, index)] > 127)
            stats = geometry_gate(z['origins'], z['directions'], z['positions'], z['raw_normals'],
                                  z['normals'], z['depth'], z['hit'], mask, P[i], K[i])
            o, d = rays(P[i], K[i], 128, 128)
            stats['independent_ray_error'] = float(max(abs(o-z['origins']).max(), abs(d-z['directions']).max()))
            require(stats['passed'] and stats['independent_ray_error'] <= 2e-6, 'Saved geometry failed')
            geometry.append(dict(camera_id=str(i), **stats))
            if i == 1:
                j = 106 * 128 + 53
                witness = dict(camera_id='1', pixel_xy=[53, 106], ray_index=j,
                               depth=float(z['depth'][j]),
                               facing_dot=float(np.dot(z['normals'][j], -z['directions'][j])))
        with np.load(root / 'geometry/labels' / (str(i) + '.npz'), allow_pickle=False) as z:
            eligible += int(z['orientation_eligible'].sum())
            undefined += int((~z['orientation_defined']).sum())
            values = dict(width_major=z['widths_major_minor'][:, 0], width_minor=z['widths_major_minor'][:, 1],
                          metallic=z['metallic'], color=z['color'])
            for key, value in values.items():
                ranges[key][0] = min(ranges[key][0], float(value.min()))
                ranges[key][1] = max(ranges[key][1], float(value.max()))
        for relight in (False, True):
            if relight and str(i) not in report['splits']['validation'] + report['splits']['held_view_nvs']:
                continue
            path = root / 'references' / ('relight' if relight else '') / (str(i) + '.npz')
            with np.load(path, allow_pickle=False) as z:
                from benchmarks.bell_e11 import convergence
                hit = np.load(root / 'references' / (str(i) + '.npz'), allow_pickle=False)
                try:
                    h = hit['hit']; fine, coarse = z['linear_components'], z['coarse_components']
                    gate = convergence(fine[h, 2], coarse[h, 2])
                    require(gate['passed'] and np.isfinite(fine).all() and (fine >= 0).all(), 'Radiance failed')
                    err = float(abs(fine[:, :, 0] + fine[:, :, 1] - fine[:, :, 2]).max())
                    component_error = max(component_error, err)
                    require(np.array_equal(fine[:, :, 0] + fine[:, :, 1], fine[:, :, 2]), 'Component sum mismatch')
                    radiance.append(dict(camera_id=str(i), relight=relight, **gate))
                    ranges['linear_rgb'][0] = min(ranges['linear_rgb'][0], float(fine[h, 2].min()))
                    ranges['linear_rgb'][1] = max(ranges['linear_rgb'][1], float(fine[h, 2].max()))
                    if not relight:
                        e = z['unoccluded_directional_energy'][h]
                        require(np.isfinite(e).all() and (e >= 0).all(), 'Invalid recorded energy')
                        ranges['energy'][0] = min(ranges['energy'][0], float(e.min()))
                        ranges['energy'][1] = max(ranges['energy'][1], float(e.max()))
                        energy.append(int((e > 1).sum()))
                finally:
                    hit.close()
    hits = sum(g['hits'] for g in geometry)
    relight_hits = sum(g['hits'] for g in geometry if g['camera_id'] in
                       report['splits']['validation'] + report['splits']['held_view_nvs'])
    counts = report['counters']
    require(counts['primary_rays'] == 128**3 and counts['shadow_directions'] == hits*16384
            and counts['brdf_direction_evaluations'] == (hits+relight_hits)*16384
            and counts['brdf_direction_returns'] == counts['brdf_direction_evaluations']
            and counts['valid_shadow_queries'] == counts['valid_shadow_query_returns'], 'Work counter mismatch')
    candidate = json.loads((root / 'geometry/oracle-admission.candidate.json').read_text())
    require(candidate['researcher_reviewed_generation'] is False and
            report['researcher_reviewed_generation'] is False, 'Unexpected researcher admission')
    files = [p for p in batch.rglob('*') if p.is_file()]
    summary = dict(schema_version=1, protocol='bell-e11-v3-saved-evidence-review-v1',
                   status='pass_pending_researcher_dataset_admission', scientific_queries=0,
                   generation=record(report_path), completion=record(root/'completion.json'),
                   authorization=record(auth_path), launch=record(batch/'launch.json'), freeze=record(freeze),
                   verification=record(verification), original_offline_verification=record(batch/'offline-verification-v1.json'),
                   candidate=record(root/'geometry/oracle-admission.candidate.json'),
                   console=record(batch/'console.log'), progress=record(root/'progress.json'),
                   original_inputs_rehashed=len(report['inputs']), frozen_files_verified=len(report['source_files']),
                   pinned_scientific_snapshots=PINS, splits=report['splits'], ray_backend=report['ray_backend'],
                   geometry=geometry, radiance_convergence=radiance, foreground_ray_observations=hits,
                   corrected_pixel=witness, ranges=ranges, orientation_eligible=eligible, orientation_undefined=undefined,
                   additive_energy_channels_above_one=sum(energy), diffuse_specular_sum_max_abs=component_error,
                   work=counts, elapsed_seconds=report['elapsed_seconds'], runtime=report['runtime'],
                   resources={k:report[k] for k in ('host_peak_rss_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')},
                   batch_file_bytes=sum(p.stat().st_size for p in files), batch_file_count=len(files),
                   output_inventory=report['outputs'], dataset_inventory=report['dataset_files'],
                   limitations=['No renderer/BRDF/shadow queries repeated; billions of raw shadow decisions are not archived.',
                                'Two-grid convergence is not a statistical error bound.',
                                'Oracle mesh is benchmark-defined, not measured ground-truth geometry.',
                                'Additive BRDF is not globally energy conserving.',
                                'Native corrected material construction/backpropagation remain unexecuted.'])
    output.mkdir(exist_ok=False)
    (output/'review.json').write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+'\n')
    write_csv(summary, output/'per_view.csv')
    figures(root, output)
    return summary


def write_csv(summary, path):
    fields=['camera_id','split','hits','silhouette_iou','normal_dot_view_min','intersection_abs',
            'reprojection_pixels','maximum_rgb_difference','relative_l2',
            'relight_maximum_rgb_difference','relight_relative_l2']
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for g in summary['geometry']:
            i=g['camera_id'];row={k:g[k] for k in fields if k in g}
            row['split']=next(k for k,ids in summary['splits'].items() if i in ids)
            for r in summary['radiance_convergence']:
                if r['camera_id']==i:
                    for k in ('maximum_rgb_difference','relative_l2'):
                        row[('relight_' if r['relight'] else '')+k]=r[k]
            writer.writerow(row)


def figures(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'bell-e11-v3-review-v1'
    scene = root/'datasets/GlossySynthetic/bell'
    fig, ax = plt.subplots(2, 6, figsize=(12, 4))
    for j, i in enumerate((0, 1, 58, 11, 105, 76)):
        for row, path in enumerate((scene/(str(i)+'.png'), scene/'masks_train'/(str(i)+'.png'))):
            with Image.open(path) as image:
                ax[row,j].imshow(np.asarray(image), cmap='gray', vmin=0, vmax=255)
            ax[row,j].axis('off')
        ax[0,j].set_title('Camera '+str(i))
    fig.suptitle('New independent Bell observations (sRGB); oracle silhouettes')
    save(fig, output, 'observations')
    fig, ax = plt.subplots(2, 3, figsize=(7, 5))
    for row, i in enumerate((58, 11)):
        for col, path in enumerate((scene/(str(i)+'.png'),root/'references/relight'/(str(i)+'.png'))):
            with Image.open(path) as image:
                ax[row,col].imshow(np.asarray(image));ax[row,col].axis('off')
        with np.load(root/'references'/(str(i)+'.npz'),allow_pickle=False) as a, np.load(root/'references/relight'/(str(i)+'.npz'),allow_pickle=False) as b:
            delta=np.abs(a['linear_components'][:,:,2]-b['linear_components'][:,:,2]).max(-1)
        im=ax[row,2].imshow(delta,vmin=0,vmax=.35,cmap='magma');ax[row,2].axis('off')
        ax[row,0].set_title('Camera '+str(i)+': original light');ax[row,1].set_title('+90° Y light');ax[row,2].set_title('Max |Δlinear RGB|')
    fig.colorbar(im,ax=ax[:,2],shrink=.5)
    save(fig, output, 'relighting')
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    with np.load(root/'geometry/labels/58.npz',allow_pickle=False) as z:
        for j,(label,values,lo,hi) in enumerate((('Major slope width',z['widths_major_minor'][:,0],.04,1),('Minor slope width',z['widths_major_minor'][:,1],.04,1),('Metallic',z['metallic'][:,0],0,1))):
            image=np.full(128*128,np.nan);image[z['pixel_indices']]=values
            im=ax[j].imshow(image.reshape(128,128),vmin=lo,vmax=hi,cmap='viridis');ax[j].axis('off');ax[j].set_title(label);fig.colorbar(im,ax=ax[j],shrink=.7)
    fig.suptitle('Camera58: spatial truth labels (background excluded)')
    save(fig, output, 'material_labels')
    plt.close('all')


def save(fig, output, name):
    svg=output/(name+'.svg')
    fig.savefig(svg,bbox_inches='tight',metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(output/(name+'.png'),bbox_inches='tight',dpi=140)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--verification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(all(x.is_absolute() and x.resolve()==x for x in (a.batch,a.verification,a.output)), 'Explicit unaliased paths required')
    result=review(a.batch,a.verification,a.output)
    print(json.dumps({k:result[k] for k in ('status','foreground_ray_observations','ranges','resources','corrected_pixel')}))
