"""File-only E11F analysis: cached arrays, no Torch, BRDF or ray queries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.bell_anisotropy_e11f import ROOT, LIMITS, PROTOCOL, GRIDS, NEW_CASES, inputs, summarize
from experiments.analyze_bell_highlight_e11d import record, replay_equal, response_fields, verify_layout
from experiments.bell_highlight_e11d import write_json, save_figure
from infrastructure.signatures import require
from infrastructure.stage_sources import digest, baseline, PINS
from infrastructure.verified_ray_backend import identity

COMMIT = 'cbfe1e235ec2062363600ca0f277682a6e7487fc'
FREEZE_SHA = 'f129d3b8015333d036afe7010792d210be80680d4417cee3a869c731b234dbda'
REPORT_SHA = '9f8c8a3e52d0aaea434c11f623a05c6da948d08d266ee38fef4f3e940bcbadae'


def verify_grid(data, original, pixels):
    require(set(data) == {'components', 'pixel_indices', 'reused_case_mask'}, 'Wrong grid inventory')
    a = data['components']
    require(a.shape == (84, 8, 3, 3) and a.dtype == np.float64 and
            np.isfinite(a).all() and (a >= 0).all(), 'Invalid saved components')
    require(data['pixel_indices'].dtype == np.int64 and np.array_equal(data['pixel_indices'], pixels),
            'Probe IDs changed')
    require(data['reused_case_mask'].dtype == np.bool_ and
            np.array_equal(data['reused_case_mask'], [False]*7+[True]), 'Wrong reused-case declaration')
    require(np.array_equal(a[:, :, 0]+a[:, :, 1], a[:, :, 2]), 'Component sum mismatch')
    require(np.array_equal(a[:, 7], original[:, 7]), 'Reused isotropic bytes changed')
    require(np.array_equal(a[:, :, 0], original[:, :, 0]), 'Fixed diffuse components changed')
    return a


def verify_material(data, labels, pixels):
    require(set(data) == {'pixel_indices', 'original_widths', 'stronger_widths', 'canonical_mean'},
            'Wrong material-factor inventory')
    old = labels['widths_major_minor']; new = data['stronger_widths']
    require(data['pixel_indices'].dtype == np.int64 and np.array_equal(data['pixel_indices'], pixels) and
            data['original_widths'].dtype == np.float64 and np.array_equal(data['original_widths'], old),
            'Original material/probe identity changed')
    factor = np.sqrt(old[:, 0]/old[:, 1])
    expected = np.stack([old[:, 0]*factor, old[:, 1]/factor], -1)
    require(new.shape == old.shape and new.dtype == np.float64 and np.isfinite(new).all() and
            np.allclose(new, expected, atol=0, rtol=16*np.finfo(np.float64).eps),
            'Requested material transformation changed')
    mean = data['canonical_mean']
    require(mean.shape == (84,) and mean.dtype == np.float64 and
            np.allclose(mean, np.sqrt(old.prod(-1)), atol=0, rtol=16*np.finfo(np.float64).eps),
            'Canonical isotropic mean changed')
    require(np.allclose(new.prod(-1), old.prod(-1), atol=0, rtol=16*np.finfo(np.float64).eps),
            'Product invariant failed')
    return dict(original_ratio_min_median_max=[float(v) for v in np.quantile(old[:, 0]/old[:, 1], [0, .5, 1])],
                stronger_ratio_min_median_max=[float(v) for v in np.quantile(new[:, 0]/new[:, 1], [0, .5, 1])],
                product_max_relative_error=float(np.max(abs(new.prod(-1)/old.prod(-1)-1))),
                CPU_CUDA_width_max_absolute_difference=float(abs(new-expected).max()),
                canonical_two_axis_width_sha256=hashlib.sha256(
                    np.repeat(mean[:, None], 2, axis=1).astype('<f8').tobytes()).hexdigest())


def response_rows(old, old_low, new, new_low):
    fields = {'original': response_fields(old, old_low), 'stronger': response_fields(new, new_low)}
    rows = []
    for name in fields['original']:
        a, b = fields['original'][name], fields['stronger'][name]
        row = dict(factor=name, newly_resolved=int((b['resolved'] & ~a['resolved']).sum()),
                   no_longer_resolved=int((a['resolved'] & ~b['resolved']).sum()))
        for label, v in [('original', a), ('stronger', b)]:
            row.update({label+'_resolved': int(v['resolved'].sum()),
                        label+'_median_response': float(np.median(v['amplitude'])),
                        label+'_absolute_gate': int(v['absolute_gate'].sum()),
                        label+'_grid_gate': int(v['grid_gate'].sum()),
                        label+'_two_codes': int(v['code_gate'].sum()),
                        label+'_zero_codes': int((v['code_difference'] == 0).sum())})
        rows.append(row)
    return rows, fields


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def analyze(batch, benchmark, pilot, reference, output):
    require(all(p.is_absolute() and p.resolve() == p for p in (batch, benchmark, pilot, reference, output)),
            'Explicit unaliased paths required')
    require(not output.exists() and all(p not in output.parents for p in (batch, benchmark, pilot, reference)),
            'Fresh separate analysis directory required')
    native = batch/'anisotropy'; rp = native/'anisotropy.json'
    require(digest(rp) == REPORT_SHA, 'Selected native report changed')
    r = json.loads(rp.read_text()); launch = json.loads((batch/'launch.json').read_text())
    ap = Path(str(batch)+'-authorization.json'); a = json.loads(ap.read_text())
    fp = ROOT/'docs/protocols/BELL_ANISOTROPY_E11F_V1.freeze.json'
    require(digest(fp) == FREEZE_SHA, 'Frozen definition changed'); f = json.loads(fp.read_text())
    require(r['status'] == 'complete' and r['cleanup_verified'] is True and
            (batch/'exit-code.txt').read_text().strip() == '0', 'Incomplete execution or cleanup')
    require(r['protocol'] == a['protocol'] == f['protocol'] == PROTOCOL and
            r['commit'] == a['commit'] == launch['commit'] == COMMIT and
            r['freeze_sha256'] == a['freeze_sha256'] == launch['freeze_sha256'] == FREEZE_SHA and
            r['authorization_sha256'] == launch['authorization_sha256'] == digest(ap) and a['approved'] is True and
            a['host_output_directory'] == launch['batch'] == str(batch) and
            a['output_directory'] == '/outputs/anisotropy' and a['selected_gpu'] == launch['gpu'] == '0' and
            a['image_id'] == r['cached_tracing_identity']['image_id'] == identity()['image_id'],
            'Launch/authorization identity mismatch')
    for key, path in [('benchmark', benchmark), ('pilot', pilot), ('reference', reference)]:
        require(a['host_'+key] == launch[key] == str(path), 'Input mount identity mismatch')
    require(r['source_files'] == f['files'] and r['limits'] == a['limits'] == f['limits'] == LIMITS and
            r['new_cases'] == list(NEW_CASES) == f['new_cases'] and r['grids'] == list(GRIDS) == f['grids'] and
            r['lighting'] == f['lighting'] and r['camera_id'] == '58' and r['probes'] == 84 and
            r['reused_case'] == 'isotropic' and r['cached_tracing_identity'] == identity(), 'Scientific identity mismatch')
    for name, sha in r['source_files'].items():
        require(digest(ROOT/name) == sha, 'Frozen source changed: '+name)
    for stage in PINS:
        baseline(stage, PINS[stage])
    labels, wo, pixels, cache, consumed = inputs(benchmark, pilot, reference)
    mounts = {'v3': benchmark, 'e11d': pilot, 'e11e': reference}
    native_inputs = []
    for item in r['inputs']:
        p = Path(item['path']); require(p.parts[:2] == ('/', 'inputs') and p.parts[2] in mounts,
                                       'Unexpected native input')
        host = mounts[p.parts[2]].joinpath(*p.parts[3:])
        require(host.resolve() == host and mounts[p.parts[2]] in host.parents and
                digest(host) == item['sha256'] and host.stat().st_size == item['bytes'], 'Consumed input changed')
        native_inputs.append(record(host))
    require(native_inputs == consumed, 'Consumed-input inventory mismatch')
    require(a['immutable_inputs'] == consumed, 'Authorized inputs changed')
    expected = {'comparison.json', 'grid-256.npz', 'grid-512.npz', 'material-factor.npz'}
    require(len(r['outputs']) == 4 and {x['path'] for x in r['outputs']} == expected, 'Wrong native inventory')
    for item in r['outputs']:
        p = native/item['path']
        require(digest(p) == item['sha256'] and p.stat().st_size == item['bytes'], 'Native artifact changed')
    verify_layout(native, expected | {'anisotropy.json', 'progress.json'})
    material = load_npz(native/'material-factor.npz'); material_check = verify_material(material, labels, pixels)
    require(r['isotropic_equivalence']['mathematical_equivalence'] is True and
            material_check['canonical_two_axis_width_sha256'] ==
            r['isotropic_equivalence']['canonical_mean_sha256'], 'Isotropic canonical identity failed')
    grids = {size: verify_grid(load_npz(native/('grid-'+str(size)+'.npz')), cache[size]['original'], pixels)
             for size in GRIDS}
    require(r['completed_grids'] == [dict(size=s, completed_new_probes=84, diffuse_control_max_abs=0.,
                                         reused_isotropic_exact=True) for s in GRIDS], 'Incomplete grid records')
    comparison = summarize(grids[256], grids[512], cache[256]['original'], cache[512]['original'],
                           labels, cache[512]['visible_counts'], pixels)
    replay_equal(comparison, r['comparison'])
    replay_equal(comparison, json.loads((native/'comparison.json').read_text()))
    require(r['counters'] == dict(brdf_attempts=192675840, brdf_returns=192675840,
            environment_attempts=27525120, environment_returns=27525120, new_shadow_queries=0,
            new_primary_rays=0, original_BRDF_queries=0, new_isotropic_BRDF_queries=0) and
            all(r[k] == 0 for k in ('native_model_calls', 'native_shader_calls', 'optimizer_returns')),
            'Work counters differ from authorized study')
    size_bytes = sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())
    require(r['elapsed_seconds'] < LIMITS['wall_seconds'] and r['host_peak_rss_bytes'] <= 64*1024**3 and
            size_bytes <= LIMITS['output_bytes'], 'Recorded resource limit exceeded')
    rows, fields = response_rows(cache[512]['original'], cache[256]['original'], grids[512], grids[256])
    probes = []
    for j, pixel in enumerate(pixels):
        row = dict(pixel_index=int(pixel), x=int(pixel % 128), y=int(pixel//128),
                   cell_id=int((pixel//128//16)*8+pixel % 128//16),
                   visible_fraction=float(cache[512]['visible_counts'][j]/512**2))
        for label, factors in fields.items():
            for name, values in factors.items():
                for key, value in values.items():
                    row[label+'_'+name+'_'+key] = value[j].item()
        probes.append(row)
    convergence = [dict(case=k, reused=(k == 'isotropic'), **v)
                   for k, v in comparison['two_grid_convergence'].items()]
    output.mkdir()
    result = dict(schema_version=1, protocol='bell-anisotropy-e11f-offline-analysis-v1', verification_status='pass',
                  decision='FIX_AND_PROCEED', full_dataset_admitted=False, native_report=record(rp),
                  freeze=record(fp), authorization=record(ap), scientific_sources=r['source_files'],
                  input_records=consumed, native_outputs=[record(native/p) for p in sorted(expected)],
                  execution_receipts=[record(batch/p) for p in ('launch.json', 'exit-code.txt', 'console.log', 'container.cid')],
                  runtime=r['runtime'], cached_tracing_identity=r['cached_tracing_identity'], counters=r['counters'],
                  elapsed_seconds=r['elapsed_seconds'], batch_bytes=size_bytes,
                  resources={k: r[k] for k in ('host_peak_rss_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')},
                  material_transformation=material_check, isotropic_equivalence=r['isotropic_equivalence'],
                  comparison=comparison, response_summary=rows, original_E11D_admission_changed=False,
                  original_E11E_replaced=False, scientific_reexecution=False, new_BRDF_queries=0, new_GPU_queries=0)
    write_json(output/'analysis.json', result)
    for name, data in [('response_summary', rows), ('probe_responses', probes), ('convergence_summary', convergence)]:
        with (output/(name+'.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(data)
    figures(output, pixels, fields, rows, comparison, grids, cache)
    return result


def figures(output, pixels, fields, rows, comparison, grids, cache):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'e11f-analysis-v1'
    names = list(fields['original'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7)); x = np.arange(4)
    for offset, label in [(-.18, 'original'), (.18, 'stronger')]:
        axes[0].bar(x+offset, [row[label+'_resolved'] for row in rows], .36, label=label)
        axes[1].bar(x+offset, [row[label+'_median_response'] for row in rows], .36, label=label)
    axes[0].set_ylabel('Resolved probes /84'); axes[0].axhline(21, color='grey', linestyle=':', label='25% screen')
    axes[1].set_ylabel('Median max-channel response (linear RGB)'); axes[1].axhline(.003, color='grey', linestyle=':')
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names); ax.legend(fontsize=8)
    fig.suptitle('Same L2 and84 probes: stronger anisotropy improves orientation, not width coverage')
    fig.tight_layout(); save(fig, output, 'factor_comparison')
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    # Display actual probes only; blank pixels are unmeasured, not zero response.
    for row, label in enumerate(('original', 'stronger')):
        for col, name in enumerate(names):
            ax = axes[row, col]; f = fields[label][name]; vmax = max(fields[k][name]['amplitude'].max() for k in fields)
            sc = ax.scatter(pixels % 128, pixels//128, c=f['amplitude'], vmin=0, vmax=vmax, s=22, cmap='magma')
            mask = f['resolved']; ax.scatter(pixels[mask] % 128, pixels[mask]//128, facecolors='none', edgecolors='cyan', s=36, linewidths=.6)
            ax.set_xlim(0, 127); ax.set_ylim(127, 0); ax.set_aspect('equal'); ax.set_title(label+' '+name, fontsize=9)
            fig.colorbar(sc, ax=ax, fraction=.046)
    fig.suptitle('Sparse camera58 sensitivity; cyan circles pass all frozen local gates')
    fig.tight_layout(); save(fig, output, 'spatial_sensitivity')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, array in [('original', cache[512]['original']), ('stronger', grids[512])]:
        rgb = array[:, :, 2]; J = np.stack([(rgb[:, a]-rgb[:, b])/2 for a, b in [(1, 2), (3, 4), (5, 6)]], -1)
        singular = np.linalg.svd(J, compute_uv=False)
        axes[0].plot(np.arange(1, 85)/84, np.sort(singular[:, 0]/singular[:, -1]), label=label)
        norms = np.linalg.norm(J, axis=1)
        axes[1].scatter(norms[:, 1], norms[:, 2], s=15, alpha=.65, label=label)
    axes[0].set_yscale('log'); axes[0].set_xlabel('Empirical probe quantile'); axes[0].set_ylabel('Conditional scaled J condition number')
    axes[1].set_xlabel('Major-width column norm'); axes[1].set_ylabel('Minor-width column norm')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle('One-view conditional finite differences: stronger signal does not remove coupling')
    fig.tight_layout(); save(fig, output, 'conditional_coupling')


def save(fig, output, name):
    import matplotlib.pyplot as plt
    save_figure(fig, output/(name+'.svg')); fig.savefig(output/(name+'.png'), dpi=130, bbox_inches='tight'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'benchmark', 'pilot', 'reference', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    result = analyze(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['verification_status'], decision=result['decision'], response_summary=result['response_summary'])))


if __name__ == '__main__':
    main()
