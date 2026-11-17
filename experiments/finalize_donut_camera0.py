"""Verify saved camera0 evidence and copy small results; never start Blender."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from experiments.donut_camera0_reproduction import (
    ROOT, authorize, comparison, digest, require, verify_freeze,
)

COMMIT = 'c2ced59da6f20804a452ec4ff9705498f714c942'
FREEZE_SHA = 'e435d3190b22d50b708903d574bd72d3780d3d72d4aad4428f953a6f21dbf2c2'


def identity(path):
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def finalize(evidence, destination):
    evidence = evidence.resolve(strict=True)
    destination = destination.resolve()
    require(ROOT / 'docs/results' in destination.parents and not destination.exists(),
            'Select a fresh repository results directory')
    freeze = verify_freeze(evidence / 'freeze.json', FREEZE_SHA)
    require(str(evidence) == freeze['output_directory'], 'Wrong execution directory')
    authorize(evidence / 'authorization.json', freeze, FREEZE_SHA, COMMIT)
    native = json.loads((evidence / 'render-receipt.json').read_text())
    execution = json.loads((evidence / 'execution-receipt.json').read_text())
    saved = json.loads((evidence / 'comparison.json').read_text())
    require(native['status'] == 'complete' and native['commit'] == COMMIT and
            native['freeze_sha256'] == FREEZE_SHA and native['render_attempts'] == 1 and
            native['successful_render_returns'] == 1 and native['operator_return'] == ['FINISHED'] and
            native['image_only_roundtrip'] and native['source_unchanged'] and
            native['physical_settings_before'] == native['physical_settings_after'],
            'Incomplete or inconsistent native receipt')
    require(execution['exit_code'] == 0 and execution['stop_reason'] is None and
            execution['supervisor_cleanup_complete'] and
            execution['wall_seconds'] < freeze['limits']['wall_seconds'] and
            execution['observed_peak_RSS_bytes'] < freeze['limits']['RSS_bytes'],
            'Execution/cleanup/resource gate failed')
    for name, record in native['outputs'].items():
        path = evidence / name
        require(evidence in path.resolve().parents and identity(path) == record,
                'Native output changed: ' + name)
    require(saved['status'] == 'file_only_comparison_complete' and
            saved['render_receipt_sha256'] == digest(evidence / 'render-receipt.json'),
            'Comparison receipt mismatch')
    with np.load(evidence / 'comparison-inputs.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    require(set(arrays) == {'reference_RGB16', 'reproduced_RGB16', 'roundtrip_RGB16',
                           'linear_RGB', 'reference_mask', 'render_mask'}, 'Unexpected array inventory')
    for name, array in arrays.items():
        shape = (1080, 1080) if name.endswith('mask') else (1080, 1080, 3)
        dtype = 'bool' if name.endswith('mask') else 'float32' if name == 'linear_RGB' else 'uint16'
        require(array.shape == shape and str(array.dtype) == dtype and np.isfinite(array).all(),
                'Invalid numerical archive: ' + name)
    ref, pred = arrays['reference_RGB16'], arrays['reproduced_RGB16']
    mask, render_mask = arrays['reference_mask'], arrays['render_mask']
    measured, error, ssim = comparison(ref, pred, mask, render_mask)
    require(measured == saved['comparison'], 'Recomputed metrics differ')
    roundtrip = abs(pred.astype(np.int32) - arrays['roundtrip_RGB16'].astype(np.int32))
    require(saved['EXR_to_AgX_roundtrip'] == dict(max_LSB=int(roundtrip.max()),
            different_components=int((roundtrip != 0).sum()), within_2_LSB=bool((roundtrip <= 2).all())),
            'Round-trip report differs from saved pixels')
    require(saved['EXR_to_AgX_roundtrip']['within_2_LSB'], 'Frozen round-trip gate failed')
    with np.load(evidence / 'RGB-errors.npz', allow_pickle=False) as archive:
        for name, value in dict(absolute_RGB_error=error, SSIM_map=ssim,
                                reference_mask=mask, render_mask=render_mask).items():
            require(np.array_equal(archive[name], value), 'Saved error map differs: ' + name)
    failure = json.loads((ROOT / 'docs/evidence/donut_camera0_correction_v2/failure.json').read_text())
    failed = Path(failure['batch'])
    require(set(p.name for p in failed.iterdir() if p.is_file()) == set(failure['files']),
            'Failed v1 evidence inventory changed')
    for name, record in failure['files'].items():
        require(identity(failed / name) == record, 'Failed v1 evidence changed: ' + name)
    preserved = ['scientific_sources', 'scientific_variants', 'docs/protocols',
                 'experiments/donut_camera0_reproduction.py', 'experiments/render_donut_camera0.py',
                 'tests/infrastructure/test_donut_camera0_reproduction.py']
    subprocess.run(['git', 'diff', '--exit-code', COMMIT, '--'] + preserved, cwd=ROOT, check=True)
    junit = ROOT / 'docs/evidence/donut_camera0_correction_v2/tests.junit.xml'
    cases = ET.parse(junit).getroot().findall('.//testcase')
    counts = dict(tests=len(cases), failures=sum(c.find('failure') is not None for c in cases),
                  errors=sum(c.find('error') is not None for c in cases),
                  skipped=sum(c.find('skipped') is not None for c in cases))
    counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
    require(counts == dict(tests=5, failures=0, errors=0, skipped=0, passed=5), 'Preparation test evidence differs')
    files = {p.relative_to(evidence).as_posix(): identity(p) for p in sorted(evidence.rglob('*'))
             if p.is_file() and 'runtime' not in p.relative_to(evidence).parts}
    require(sum(r['bytes'] for r in files.values()) < freeze['limits']['output_bytes'], 'Output cap exceeded')
    boundary = mask ^ binary_erosion(mask)
    distance = distance_transform_edt(~boundary)
    disagreement = mask ^ render_mask
    manifest = dict(schema_version=1, protocol='donut-camera0-results-finalization-v1',
        status='verified_single_frame_near_reproduction_NOT_historical_build_attribution',
        source_commit=COMMIT, freeze_sha256=FREEZE_SHA, evidence_root=str(evidence),
        external_artifacts=files, runtime_cache_excluded=True,
        original_inputs=freeze['inputs'], frozen_sources=freeze['sources'],
        failed_v1_preserved=failure, recorded_preparation_tests=dict(counts=counts, JUnit=identity(junit), rerun=False),
        scientific_subtrees={name: subprocess.check_output(['git', 'rev-parse', COMMIT + ':' + name],
                              cwd=ROOT, text=True).strip() for name in ('scientific_sources', 'scientific_variants')},
        verification=dict(input_identities=len(freeze['inputs']), frozen_sources=len(freeze['sources']),
            original_scientific_sources_and_protocols_unchanged=True, native_output_hashes=True,
            saved_metric_and_error_array_recomputation=True, numpy_allow_pickle=False,
            native_exit_code=execution['exit_code'], cleanup_complete=True,
            outer_timeout_exit_code=None, outer_timeout_exit_code_note='Not separately persisted; native receipt and completed comparison verified.'),
        supplementary_diagnostics=dict(mask_disagreement_distance_to_original_boundary_px_quantiles_0_50_95_100=
            np.quantile(distance[disagreement], [0, .5, .95, 1]).tolist(),
            exact_RGB_components_fraction=float((error == 0).mean()),
            exact_RGB_pixels_fraction=float((error == 0).all(-1).mean())),
        new_scientific_render_calls=0, new_training_updates=0)
    destination.mkdir(parents=True)
    for name in ('camera0-comparison.png', 'comparison.json', 'render-receipt.json',
                 'execution-receipt.json', 'authorization.json', 'console.log'):
        copied_name = 'console.txt' if name == 'console.log' else name
        shutil.copyfile(evidence / name, destination / copied_name)
        require(identity(destination / copied_name) == files[name], 'Copied artifact mismatch: ' + name)
    manifest['tracked_copies'] = {p.name: identity(p) for p in sorted(destination.iterdir())}
    manifest['renamed_copy_sources'] = {'console.txt': 'console.log'}
    manifest['finalization_code'] = identity(Path(__file__))
    (destination / 'artifact-manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=manifest['status'], destination=str(destination),
                         scientific_render_calls=0, input_identities=len(freeze['inputs']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.evidence, args.destination)
