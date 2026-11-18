"""Freeze checks and an axes-free replay of archived reserved-edge diagnostics."""
import json
from pathlib import Path
import math

import numpy as np

from benchmarks.donut_tensor_d8 import POLICY, REPRESENTATION, limits as paired_limits
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest

ARMS = ('D11',)
PROTOCOL = 'donut-kurt-log-euclidean-tensor-d11-v1'
FREEZE = 'docs/protocols/DONUT_KURT_LOG_TENSOR_D11_V1.freeze.json'
COEFFICIENT_REPORT = 'docs/evidence/donut_log_tensor_d11_preparation_v1/coefficient.json'
LOSS_ID = 'transported_log_euclidean_Q_v1'
CONTROL_FILES = ('initial.json', 'paired-initial-gate.npz')
MATHEMATICAL_CONTROL = 'D8-T'
# Absolute allowance for subtracting float32 raw archive squared residuals
# and sorted float32 widths. Report any excursions to this allowance.
ARCHIVE_SQUARE_ATOL = 1e-6


def limits(base):
    return dict(paired_limits(base), wall_seconds=3720)


def coefficient(raw_gradient_rms, log_gradient_rms):
    require(all(math.isfinite(v) and v > 0 for v in (raw_gradient_rms, log_gradient_rms)),
            'Expected positive finite fitting scale')
    return POLICY['coefficient'] * raw_gradient_rms / log_gradient_rms


def verify_policy(plan, root=ROOT):
    expected = json.loads((root/COEFFICIENT_REPORT).read_text())
    proposed = expected['coefficient']
    require(proposed == coefficient(expected['pooled_rms_raw'], expected['pooled_rms_log'])
            and plan['spatial_policy'] == dict(POLICY, coefficient=proposed)
            and plan['spatial_regularizer'] == LOSS_ID
            and plan['coefficient_sha256'] == digest(root/COEFFICIENT_REPORT)
            and set(plan['D8_T_initial_controls']) == set(CONTROL_FILES),
            'Frozen D11 spatial definition differs')


def verify_initial_controls(directory, records):
    for name in CONTROL_FILES:
        p = Path(directory)/name
        require(p.is_file() and p.resolve()==p and p.stat().st_size==records[name]['bytes']
                and digest(p)==records[name]['sha256'], 'D8-T initial control differs: '+name)


def archived_log_square(widths, raw_square):
    """Recover log-distance without recovering directors from an axis gap."""
    width = np.asarray(widths, dtype=np.float64)
    raw = np.asarray(raw_square, dtype=np.float64)
    require(width.ndim==3 and width.shape[1:]==(2, 2) and raw.shape==(len(width),)
            and np.isfinite(width).all() and (width > 0).all() and (width <= 1.000001).all()
            and np.isfinite(raw).all() and (raw >= 0).all(),
            'Invalid historical edge widths or raw-Q squared distances')
    major, minor = np.sort(width**2, axis=-1)[..., ::-1].transpose(2,0,1)
    m, r = (major+minor)/2, (major-minor)/2
    log_minor = np.log(minor)
    log_gap = np.log1p((major-minor)/minor)
    center, radius = log_minor+log_gap/2, log_gap/2
    scale = np.divide(log_gap, major-minor, out=1/minor, where=major>minor)

    def squared_difference(x):
        return 2*np.square(x[:,0]-x[:,1])

    # Q squared difference is its isotropic and radial contributions plus
    # nonnegative relative-axis mismatch, scaled by both spectral radii.
    angular = raw-squared_difference(m)-squared_difference(r)
    ceiling = 8*r[:,0]*r[:,1]
    violation = np.maximum(-angular, angular-ceiling)
    require((violation <= ARCHIVE_SQUARE_ATOL).all(), 'Archived widths and Q distance are inconsistent')
    delta = (squared_difference(center)+squared_difference(radius)
             + np.clip(angular, 0, ceiling)*scale[:,0]*scale[:,1])
    return delta, dict(raw_square_clamped=int((violation > 0).sum()),
                       maximum_raw_square_adjustment=float(np.maximum(violation, 0).max(initial=0)))


def authorization(freeze, commit, freeze_sha):
    plan = freeze['tensor_training']
    names = ('arms','protocol','representation','limits','spatial_policy','spatial_regularizer',
             'coefficient_sha256','D8_T_initial_controls','benchmark_generation_sha256',
             'dataset_admission_sha256','initial_sha256','initial_host_path','benchmark_host_path')
    return dict(**{k:plan[k] for k in names}, approved=True, commit=commit,
                freeze_sha256=freeze_sha, image_id=freeze['image_id'], selected_gpu='0',
                host_output_directory=freeze['host_tensor_output'], output_directories={'D11':'/outputs/D11'},
                dataset_admission_path=plan['dataset_admission_host_path'], maximum_execution_launches=1,
                wall_seconds=plan['limits']['wall_seconds'], diffuse_white_in_objective=True,
                researcher_admitted_dataset=True, input_identities=freeze['inputs'],
                accepted_control_artifacts=plan['controls'])
