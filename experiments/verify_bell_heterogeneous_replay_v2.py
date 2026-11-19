"""Versioned saved-array replay with a narrow highlight-summary ULP guard."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from benchmarks import bell_heterogeneous_v1 as b
from benchmarks import donut_heterogeneous_v2 as refinement
from benchmarks.bell_e11 import linear_to_srgb
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.generate_bell_heterogeneous_v1 import add_highlights, sheet
from experiments.generate_donut_heterogeneous_v1 import compare, ordinary_probes
from experiments.verify_bell_heterogeneous_v1 import compare_geometry_metadata
from experiments.verify_donut_heterogeneous_v2 import replay_gates, validate_physical_view, view_identity
from infrastructure.signatures import require
from infrastructure.stage_sources import digest


# Only these post-render summary scalars showed one-ULP differences
# between the producer and replay NumPy environments.
ROUNDING_FIELDS = ('specular_luminance_p50', 'specular_contrast_ratio',
                   'quantization_rmse_over_specular_rms')


def highlight_replay(saved, actual):
    require(saved.keys() == actual.keys(), 'Bell highlight summary fields differ')
    roundoff = {}
    for name, value in actual.items():
        reference = saved[name]
        require(type(reference) is type(value) and
                (not isinstance(value, float) or np.isfinite([reference, value]).all()),
                'Bell highlight summary type or finite check failed: '+name)
        if reference == value:
            continue
        require(name in ROUNDING_FIELDS and type(value) is float,
                'Bell highlight summary differs: '+name)
        require(np.nextafter(reference, value) == value, 'Bell highlight summary exceeds one ULP: '+name)
        roundoff[name] = abs(reference-value)
    return roundoff


def verify(sha, destination):
    f = b.freeze(b.ROOT/b.FREEZE, sha)
    batch = Path(f['host_output'])
    require(destination.resolve() == destination and not destination.exists() and destination.parent.is_dir()
            and batch not in destination.parents, 'Fresh separate Bell verification file required')
    geometry, labels = b.source(f, *(Path(f['host_inputs'][k]) for k in ('oracle', 'admission')))
    auth = batch/'authorization.json'
    approval = b.read(auth)
    require(approval == b.authorization(f, approval['commit'], sha)
            and digest(auth) == digest(batch.with_name(batch.name+'-authorization.json'))
            and digest(batch/'freeze.json') == sha and (batch/'exit-code.txt').read_text().strip() == '0',
            'Invalid Bell execution or authorization evidence')
    out = batch/'reference'
    path = out/'generation.json'
    r = b.read(path)
    expected = dict(status='complete_pending_researcher_admission', phase='complete',
                    scientific_work_complete=True, cleanup_verified=True, training_authorized=False,
                    protocol=b.PROTOCOL, image=b.IMAGE, backend=b.identity(), commit=approval['commit'],
                    freeze_sha256=sha, authorization_sha256=digest(auth), panel=b.PANEL, splits=b.SPLITS,
                    source_generation_sha256=b.SOURCE_GENERATION, source_admission_sha256=b.SOURCE_ADMISSION,
                    input_records=f['input_records'], limits=b.LIMITS, environment=b.ENVIRONMENT, field=b.FIELD,
                    BRDF=b.BRDF, indirect='zero', point_light='absent', tolerances=b.TOLERANCES,
                    refinement=b.REFINEMENT, native_model_calls=0, optimizer_returns=0)
    require(all(r[k] == v for k, v in expected.items())
            and [row['camera'] for row in r['views']] == b.PANEL
            and r['wall_seconds'] < b.LIMITS['inner_seconds']
            and r['host_peak_RSS_bytes'] < b.LIMITS['host_memory_gib']*1024**3
            and sum(p.stat().st_size for p in batch.rglob('*') if p.is_file()) < b.LIMITS['output_bytes'],
            'Terminal Bell reference or resource conditions differ')
    compare_geometry_metadata(r['coverage'], b.coverage(geometry, labels))
    compare_geometry_metadata(r['geometry_checks'], {str(c): b.view_geometry_summary(v) for c, v in labels.items()})
    names = {'surface-area.npz', 'reference-components.png'}
    for row in r['views']:
        c = row['camera']
        names.update(f'{name}-{c}.npz' for name in ('view', 'baseline', 'quadrature'))
        names.update(f'view-{c}-{tag}.png' for tag in ('RGB', 'diffuse', 'specular'))
        names.update(f'patch-{c}-{pix}.npz' for pix in row['selected_pixels'])
    require(set(r['outputs']) == names == {p.name for p in out.iterdir() if p.is_file()
            and p.name not in ('progress.json', 'progress.next', 'generation.json')},
            'Bell reference artifact inventory differs')
    for name, item in r['outputs'].items():
        b.check_artifact(out/name, item)
    np.testing.assert_array_equal(numeric(out/'surface-area.npz')['triangle_area_native'], geometry['area'])

    total = 0
    summary = []
    for row in r['views']:
        camera = row['camera']
        old = numeric(out/f'baseline-{camera}.npz')
        coarse_counts = old.pop('baseline_coarse_visible_counts')
        view_identity(old, labels[camera])
        baseline = compare(old)
        require(all(gate['passed'] for k, gate in baseline.items() if k != 'radiance_convergence'),
                'Baseline reflectance control differs')
        selected = []

        def fetch(i):
            nonlocal total
            pixel = int(old['pixel_indices'][i])
            require(total < b.MAX_TOTAL and row['selected_pixels'][len(selected):len(selected)+1] == [pixel],
                    'Bell refinement order or panel budget differs')
            selected.append(pixel)
            total += 1
            return numeric(out/f'patch-{camera}-{pixel}.npz')

        view, grids = refinement.assemble(old, coarse_counts, fetch)
        highlight = add_highlights(view)
        require(selected == row['selected_pixels'], 'Bell refinement inventory differs')
        highlight_roundoff = highlight_replay(row['highlight_diagnostics'], highlight)
        for name, predicted in ((f'view-{camera}.npz', view), (f'quadrature-{camera}.npz', grids)):
            saved = numeric(out/name)
            require(saved.keys() == predicted.keys(), 'Bell array fields differ')
            for key, value in predicted.items():
                require(saved[key].dtype == value.dtype, 'Bell array dtype differs')
                np.testing.assert_array_equal(saved[key], value)
        validate_physical_view(view, grids)
        final = compare(view)
        replay_gates(final, {k: row[k] for k in final})
        require(all(gate['passed'] for gate in final.values())
                and row['valid'] == int(view['shader_valid'].sum()) == b.COUNTS[b.PANEL.index(camera)]
                and row['ordinary_probe_pixels'] == view['pixel_indices'][ordinary_probes(view)].tolist(),
                'Final Bell gate or valid hit inventory differs')
        for slot, tag in enumerate(('diffuse', 'specular', 'RGB')):
            canvas = np.zeros((b.SIZE**2, 3))
            rgb = view['linear_components'][:, slot]
            canvas[view['pixel_indices']] = np.clip(rgb, 0, 1)
            with Image.open(out/f'view-{camera}-{tag}.png') as png:
                np.testing.assert_array_equal(np.asarray(png), linear_to_srgb(canvas.reshape(b.SIZE, b.SIZE, 3)))
            require(row['previews'][slot] == dict(component=tag, channels_above_one=int((rgb > 1).sum())),
                    'Bell PNG clipping record differs')
        summary.append(dict(camera=camera, valid=row['valid'], refined_pixels=selected,
                            baseline_RGB=baseline['radiance_convergence'], final_gates=final,
                            highlight_summary_roundoff=highlight_roundoff,
                            **{k: int(view[k].sum()) for k in ('near_boundary', 'nearly_isotropic',
                                                              'exactly_isotropic', 'highlight_mask')}))
    with Image.open(out/'reference-components.png') as saved:
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(sheet(out)))
    require(r['refined_points'] == total and r['counters'] == b.counters(b.POINTS, total),
            'Bell work and refinement counters differ')
    json_new(destination, dict(schema_version=1, offline_replay_version=2, protocol=b.PROTOCOL,
        status='verified_pending_researcher_admission', replay_source_sha256=digest(Path(__file__)),
        highlight_roundoff_allowance=dict(fields=list(ROUNDING_FIELDS), maximum_float64_ULPs=1),
        native_commit=r['commit'], freeze_sha256=sha, authorization_sha256=digest(auth),
        generation_sha256=digest(path), package=str(out), outputs=r['outputs'], counters=r['counters'],
        coverage=r['coverage'], geometry_checks=r['geometry_checks'], per_view=summary,
        training_authorized=False, researcher_admitted_dataset=False))
    return len(summary)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256', required=True)
    cli.add_argument('--output', required=True, type=Path)
    a = cli.parse_args()
    print(f'Verified {verify(a.freeze_sha256, a.output)} heterogeneous Bell reference views')
