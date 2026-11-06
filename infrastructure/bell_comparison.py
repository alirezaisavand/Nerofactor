"""CPU-only lossless RGB/report comparison, with repeatability as the first gate."""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from infrastructure.bell_inference import (COMPARISON_SHA256, PROTOCOL, REFERENCE_SHA256,
                                         STORED_STEP, THRESHOLDS, file_hash, proposed_split, scalar_metric)
from infrastructure.paths import relative_path
from infrastructure.signatures import ADMITTED_REPORT_SHA256, CHECKPOINT_SHA256, read_report, require


def rgb_metrics(prediction, ground_truth):
    """Original network.metrics uint8 conversion, float32 two-stage MSE and SSIM."""
    import numpy as np
    from skimage.metrics import structural_similarity
    pr = np.clip(prediction * 255, 0, 255).astype(np.uint8)
    gt = np.clip(ground_truth * 255, 0, 255).astype(np.uint8)
    mse = np.mean((gt.reshape(-1, 3).astype(np.float32) - pr.reshape(-1, 3).astype(np.float32)) ** 2, 0)
    mse = np.mean(mse)
    with np.errstate(divide='ignore'):
        psnr = 10 * np.log10(255 * 255 / mse)
    return {'psnr': scalar_metric(psnr),
            'ssim': float(structural_similarity(gt, pr, win_size=11, channel_axis=2, data_range=255))}


def artifact(root, name, hashes):
    relative = relative_path(name)
    path = root / relative
    require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'Unsafe/missing output artifact')
    require(name in hashes and file_hash(path) == hashes[name], 'Output artifact identity mismatch')
    return path


def load_arm(path, digest, expected_arm):
    import numpy as np
    report, identity = read_report(path, digest)
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1 and report.get('protocol') == PROTOCOL and
            report.get('status') == report.get('phase') == 'complete' and report.get('arm') == expected_arm,
            'Incomplete/wrong arm report')
    require(type(report.get('step')) is int and report['step'] == STORED_STEP and
            type(report.get('validation_calls')) is int and report['validation_calls'] == 1 and
            report.get('checkpoint_sha256') == CHECKPOINT_SHA256, 'Wrong checkpoint/step/view count')
    require(report.get('evidence') == {'metadata': ADMITTED_REPORT_SHA256, 'reference': REFERENCE_SHA256,
                                      'comparison': COMPARISON_SHA256}, 'Unadmitted architecture evidence')
    require(report.get('splits') == proposed_split(), 'Unexpected ordered A split')
    require(type(report.get('run_id')) is str and report['run_id'], 'Missing unique run ID')
    fields = ('used_inputs', 'effective_trainer', 'effective_renderer', 'effective_shader', 'rng_before',
              'rng_after', 'runtime', 'implementation', 'preprocessing', 'legacy_sha256', 'declared_source_commit',
              'declared_image_id', 'metrics', 'arrays', 'output_hashes')
    require(all(key in report and report[key] is not None for key in fields), 'Missing run identities/metadata')
    require(type(report['runtime']) is dict and type(report['runtime'].get('packages')) is dict and
            all(type(report[key]) is dict and report[key] for key in
                ('effective_trainer', 'effective_renderer', 'effective_shader', 'rng_before', 'rng_after', 'implementation', 'preprocessing')) and
            type(report['used_inputs']) is list and bool(report['used_inputs']), 'Invalid run identity structure')
    require(report['runtime']['packages'].get('numpy') == importlib.metadata.version('numpy') and
            report['runtime']['packages'].get('scikit-image') == importlib.metadata.version('scikit-image'),
            'Use the same NumPy/scikit-image runtime for metric validation')
    hashes = report['output_hashes']
    required = {'arrays/prediction.npy', 'arrays/ground_truth.npy', 'arrays/mask.npy',
                'renders/prediction.png', 'renders/ground_truth.png', 'metrics.json'}
    require(type(hashes) is dict and required <= hashes.keys(), 'Incomplete output manifest')
    root = Path(path).resolve().parent
    for name in hashes:
        artifact(root, name, hashes)
    arrays = {}
    for key in ('prediction', 'ground_truth', 'mask'):
        require(report['arrays'].get(key) == 'arrays/' + key + '.npy', 'Invalid array selector')
        arrays[key] = np.load(str(artifact(root, report['arrays'][key], hashes)), allow_pickle=False)
    pr, gt = arrays['prediction'], arrays['ground_truth']
    require(pr.dtype == gt.dtype == np.float32 and pr.shape == gt.shape and
            pr.ndim == 3 and pr.shape[-1] == 3 and min(pr.shape[:2]) >= 11 and
            np.isfinite(pr).all() and np.isfinite(gt).all(), 'Invalid RGB arrays')
    require(arrays['mask'].dtype == np.int32 and arrays['mask'].shape == pr.shape[:2] + (1,),
            'Invalid original geometry mask shape/dtype')
    computed = rgb_metrics(pr, gt)
    require(computed == report['metrics'], 'Reported PSNR/SSIM disagree with lossless arrays')
    metric_file, _ = read_report(root / 'metrics.json', hashes['metrics.json'])
    require(metric_file == computed, 'Metric artifact disagrees with report')
    return report, arrays, identity


def metric_delta(a, b):
    if a['status'] == b['status'] == 'positive_infinity':
        return 0.0
    if a['status'] != 'finite' or b['status'] != 'finite':
        return None
    return abs(a['value'] - b['value'])


def pair_difference(reference, candidate, reference_arrays, candidate_arrays):
    import numpy as np
    identity_fields = ('evidence', 'checkpoint_sha256', 'step', 'legacy_sha256', 'splits', 'used_inputs',
                       'effective_trainer', 'effective_renderer', 'effective_shader', 'runtime',
                       'implementation', 'preprocessing', 'rng_before', 'rng_after',
                       'declared_image_id', 'declared_source_commit')
    mismatch = [key for key in identity_fields if reference.get(key) != candidate.get(key)]
    require(reference['run_id'] != candidate['run_id'], 'Arms must have distinct run IDs')
    require(reference_arrays['prediction'].shape == candidate_arrays['prediction'].shape, 'View shapes differ')
    if not np.array_equal(reference_arrays['ground_truth'], candidate_arrays['ground_truth']):
        mismatch.append('ground_truth')
    if not np.array_equal(reference_arrays['mask'], candidate_arrays['mask']):
        mismatch.append('mask')
    r, c = reference_arrays['prediction'].astype(np.float64), candidate_arrays['prediction'].astype(np.float64)
    signed = c - r
    absolute = np.abs(signed)
    violations = absolute > (THRESHOLDS['float_atol'] + THRESHOLDS['float_rtol'] * np.abs(r))
    q_r = np.clip(reference_arrays['prediction'] * 255, 0, 255).astype(np.uint8).astype(np.int16)
    q_c = np.clip(candidate_arrays['prediction'] * 255, 0, 255).astype(np.uint8).astype(np.int16)
    quantized = np.abs(q_c - q_r)
    nonzero = np.abs(r) > 0
    relative = absolute[nonzero] / np.abs(r[nonzero])
    psnr = metric_delta(reference['metrics']['psnr'], candidate['metrics']['psnr'])
    ssim = abs(reference['metrics']['ssim'] - candidate['metrics']['ssim'])
    metrics = {'reference': reference['metrics'], 'other': candidate['metrics'], 'psnr_delta_db': psnr, 'ssim_delta': ssim}
    passed = not mismatch and not violations.any() and quantized.max() <= THRESHOLDS['uint8_max'] and \
             psnr is not None and psnr <= THRESHOLDS['psnr_db'] and ssim <= THRESHOLDS['ssim']
    return {'passed': bool(passed), 'identity_mismatches': mismatch, 'shape': list(r.shape),
            'max_absolute_error': float(absolute.max()), 'mean_absolute_error': float(absolute.mean()),
            'max_relative_error_nonzero_reference': float(relative.max()) if relative.size else 0.0,
            'mean_relative_error_nonzero_reference': float(relative.mean()) if relative.size else 0.0,
            'nonzero_error_at_zero_reference_count': int(((~nonzero) & (absolute > 0)).sum()),
            'relative_error_limit': 'Relative errors at zero reference are not silently epsilon-clamped; their nonzero count is separate.',
            'float_violation_count': int(violations.sum()), 'uint8_max_error': int(quantized.max()),
            'uint8_mismatch_count': int((quantized > 0).sum()), 'uint8_mean_absolute_error': float(quantized.mean()),
            'metrics': metrics}, signed, violations


def compare_sequence(r, repeat, candidate, output, record=None):
    """Tuples are (validated report, arrays, identity); candidate can remain unexamined."""
    import numpy as np
    report = {'schema_version': 1, 'protocol': PROTOCOL, 'result': 'blocked', 'thresholds': dict(THRESHOLDS),
              'reference': r[2], 'repeat': repeat[2], 'candidate': None, 'candidate_interpreted': False,
              'output_hashes': {}}
    def compare(name, other):
        diagnostics, signed, violations = pair_difference(r[0], other[0], r[1], other[1])
        for suffix, array in [('signed-difference', signed), ('violation-mask', violations)]:
            path = output / (name + '-' + suffix + '.npy')
            with path.open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
            report['output_hashes'][path.name] = file_hash(path)
        return diagnostics
    report['repeatability'] = compare('R-repeat-minus-R', repeat)
    if not report['repeatability']['passed']:
        report['result'] = 'repeatability_failed'
        return report
    report['result'] = 'repeatability_passed'
    if record is not None:
        record(report)
    if candidate is not None:
        if callable(candidate):
            candidate = candidate()
        require(candidate[0]['run_id'] not in (r[0]['run_id'], repeat[0]['run_id']), 'Candidate reuses a run ID')
        report['candidate'] = candidate[2]
        report['candidate_interpreted'] = True
        report['candidate_comparison'] = compare('C-minus-R', candidate)
        report['result'] = 'regression_passed' if report['candidate_comparison']['passed'] else 'candidate_mismatch'
    return report
