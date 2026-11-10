"""Read-only 3C-1 trajectory analysis. NumPy only; no shader execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.material_recovery_metrics_v1 import roughness_tensor, gauge_metrics
from infrastructure.signatures import require

ROOT = Path(__file__).resolve().parents[1]
BASE = '72629688bee84c0b06f4fa3c19b207917d2dd450'
EVIDENCE_SHA = 'f3c7a93e8f24ef7e3aac2de66b91da52962cb2f75f0d64cc66a199ba159da451'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit(path, expected_sha):
    require(expected_sha == EVIDENCE_SHA and digest(path) == expected_sha, '3C-1 evidence SHA-256 mismatch')
    report = json.loads(Path(path).read_text())
    require(report['status'] == 'complete' and report['gates'] == {'reference': 'passed', 'identifiability': 'passed'}, 'Incomplete/unadmitted recovery evidence')
    identities = dict(report['implementation_sha256'])
    identities.update({'docs/protocols/KNOWN_MATERIAL_RECOVERY_V1.md': report['protocol_sha256'],
                       'docs/KNOWN_MATERIAL_RECOVERY_V1_EXECUTION.md': report['execution_notes_sha256']})
    for name, sha in identities.items():
        require(ROOT in (ROOT / name).resolve().parents,
                'Source identity path escaped repository')
        require(digest(ROOT / name) == sha, 'Accepted source/protocol identity mismatch: ' + name)
    name = 'docs/KNOWN_MATERIAL_RECOVERY_V1_RESULTS.md'
    original = subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT)
    require((ROOT / name).read_bytes() == original, 'Accepted results document changed')
    identities[name] = hashlib.sha256(original).hexdigest()
    require([r['seed'] for r in report['initializations']] == list(range(101, 106)) and
            all(r['status'] == 'complete' and r['completed_updates'] == 250 for r in report['initializations']), 'Unexpected start/update identities')
    return report, identities


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b); a = a - a.mean(); b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else None


def branch_brackets(points):
    result = []
    for left, right in zip(points, points[1:]):
        a, b = left['physical'][2], right['physical'][2]
        low, high = sorted((a, b))
        for k in range(int(np.floor((low - np.pi / 2) / np.pi)) - 1,
                       int(np.ceil((high - np.pi / 2) / np.pi)) + 2):
            boundary = np.pi / 2 + k * np.pi
            if low < boundary <= high:
                result.append({'left_id': left['id'], 'right_id': right['id'],
                               'left_update': left['update'], 'right_update': right['update'],
                               'boundary_theta': float(boundary),
                               'scope': 'Recorded bracket only; exact crossing update/causality unavailable'})
    return result


def compensation(predictions, targets):
    d = np.asarray(predictions) - np.asarray(targets)
    rd, rs, rt = (d[:, i].reshape(-1) for i in range(3))
    denominator = np.linalg.norm(rd) + np.linalg.norm(rs)
    return {'component_RMSE': np.sqrt(np.mean(d * d, axis=(0, 2))).tolist(),
            'mean_signed_residual_RGB': np.mean(d, axis=0).tolist(),
            'diffuse_specular_residual_correlation': correlation(rd, rs),
            'diffuse_specular_residual_cosine': float(rd @ rs / (np.linalg.norm(rd) * np.linalg.norm(rs))) if np.linalg.norm(rd) * np.linalg.norm(rs) > 1e-12 else None,
            'cancellation_fraction': float(1 - np.linalg.norm(rt) / denominator) if denominator > 1e-12 else None,
            'maximum_component_additivity_roundoff': float(np.max(abs(d[:, 0] + d[:, 1] - d[:, 2])))}


def analyze(report):
    truth = np.asarray(report['truth']); target = np.asarray(report['targets']['fine'])
    records, selected = [], []
    for row in report['initializations']:
        seed = row['seed']
        points = [{'id': 'seed-{}/u-0'.format(seed), 'update': 0,
                   'physical': row['initial_effective_physical_float32']}]
        points += [{'id': 'seed-{}/u-{}'.format(seed, h['completed_updates']), 'update': h['completed_updates'],
                    'physical': h['physical_after_update']} for h in row['history']]
        values = np.asarray([p['physical'] for p in points]); q = np.array([roughness_tensor(*p[:3]) for p in values])
        k, m = values[:, 3:6].mean(1), values[:, 6]
        mean_d = k * (1 - m); mean_f0 = .04 * (1 - m) + m * k
        correlations = {'mean_albedo_metallic': correlation(k, m),
                        'minor_width_metallic': correlation(np.min(values[:, :2], axis=1), m),
                        'major_width_metallic': correlation(np.max(values[:, :2], axis=1), m),
                        'mean_diffuse_F0': correlation(mean_d, mean_f0),
                        'Qxx_minus_Qyy_metallic': correlation(q[:, 0, 0] - q[:, 1, 1], m),
                        'twice_Qxy_metallic': correlation(2 * q[:, 0, 1], m),
                        'mean_albedo_twice_Qxy': correlation(k, 2 * q[:, 0, 1]),
                        'loss_gradient_norm': correlation([h['fit_MSE_before_update'] for h in row['history']],
                                                          [h['gradient_norm_before_update'] for h in row['history']])}
        brackets = branch_brackets(points); ids = {x for b in brackets for x in (b['left_id'], b['right_id'])}
        chosen = [p for p in points if p['update'] in (0, 10, 100, 200, 230, 240, 250) or p['id'] in ids]
        selected.extend(chosen)
        first, mid, last = row['history'][-3:]
        l0, l1, l2 = [h['fit_MSE_before_update'] for h in (first, mid, last)]
        delta = np.asarray(last['physical_after_update']) - np.asarray(mid['physical_after_update'])
        qe = [gauge_metrics(h['physical_after_update'][:3], truth[:3])['Q_relative_frobenius'] for h in (first, mid, last)]
        late = {'recorded_post_updates': [h['completed_updates'] for h in (first, mid, last)],
                'loss_evaluated_at_updates': [h['loss_at_update_count'] for h in (first, mid, last)],
                'losses': [l0, l1, l2], 'loss_drop_last10': l1 - l2, 'relative_drop_last10': (l1 - l2) / l1,
                'relative_drop_last20': (l0 - l2) / l0, 'last_gradient_norm': last['gradient_norm_before_update'],
                'physical_delta_240_to_250': delta.tolist(), 'physical_delta_norm': float(np.linalg.norm(delta)),
                'Q_delta_240_to_250': float(np.linalg.norm(q[-1] - q[-2])), 'Q_relative_errors': qe,
                'stationarity': 'Not established; nonzero changes/gradients and declining loss are not convergence proof'}
        residuals = {}
        for split in ('fit', 'held_out', 'relight'):
            mask = [o['split'] == split for o in report['scene']['observations']]
            residuals[split] = compensation(np.asarray(row['final_predictions'])[mask], target[mask])
        records.append({'seed': seed, 'points': points, 'late': late,
                        'correlations': correlations, 'correlation_scope': 'Dependent trajectory samples; descriptive, no causal/p-value claim',
                        'branch_brackets': brackets, 'saved_evaluation_tape_compensation': residuals,
                        'final_recovery_errors': row['recovery_errors']})
    require(len(selected) <= 45, 'Predeclared trajectory replay budget exceeded')
    return {'schema_version': 1, 'protocol': 'material-recovery-trajectories-v1', 'mode': 'offline_no_shader_no_optimizer',
            'input_sha256': EVIDENCE_SHA, 'records': records, 'selected_replay_states': selected,
            'selection_rule': 'initial + updates10,100,200,230,240,250 + all recorded angle-branch brackets; at most45 states',
            'optimizer_convergence_claim': False, 'causal_attribution': 'Deferred to new measurements; no single cause established offline'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    require(args.output.is_absolute() and args.output.resolve() == args.output, 'Explicit unaliased new output required')
    report, identities = admit(args.input, args.sha256)
    result = analyze(report); result['source_identities'] = identities; result['input_path'] = str(args.input)
    result['implementation_sha256'] = digest(__file__)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print('Offline records:', len(result['records']), 'selected replay states:', len(result['selected_replay_states']))


if __name__ == '__main__': main()
