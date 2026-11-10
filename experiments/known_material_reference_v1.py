"""Independent, bounded NumPy targets/Jacobians for the single flat v1 case.

No Torch, shader, source sampler or optimizer is used by this module.
"""
import hashlib
import numpy as np

from audits import anisotropic_brdf as reference
from infrastructure.signatures import require

PROTOCOL = 'known-material-recovery-v1'
TRUTH = np.array([.20, .55, .31, .30, .45, .60, .25], np.float64)
PARAMETERS = ['mx', 'my', 'theta', 'k_R', 'k_G', 'k_B', 'metallic']
SCALES = np.array([.96, .96, np.pi, 1., 1., 1., 1.])
GRIDS = ((256, 512), (512, 1024))
FD_STEPS = (1e-4, 5e-5)
ATOL, RTOL, RANK_RATIO = 2e-6, 5e-5, 1e-4


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def direction(polar, azimuth):
    p, a = np.deg2rad([polar, azimuth])
    return np.array([np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)])


def design():
    views = []
    for prefix, polars, azimuths in [('v_train', (20, 50), (0, 120, 240)),
                                    ('v_hold', (35, 65), (60, 180, 300))]:
        for p in polars:
            for a in azimuths:
                i = sum(v['id'].startswith(prefix) for v in views)
                views.append({'id': prefix + '_' + str(i), 'polar': p, 'azimuth': a,
                              'direction': direction(p, a).tolist()})
    lights = [{'id': 'l0', 'axis': None, 'power': None}]
    for p, azimuths, power in [(40, (0, 120, 240), 8), (55, (60, 180, 300), 16)]:
        for a in azimuths:
            lights.append({'id': 'l' + str(len(lights)), 'axis': direction(p, a).tolist(), 'power': power})
    observations = []
    for split, selected_views, selected_lights in [('fit', range(6), range(4)),
                                                   ('held_out', range(6, 12), range(4)),
                                                   ('relight', range(12), range(4, 7))]:
        for v in selected_views:
            for l in selected_lights:
                observations.append({'id': views[v]['id'] + '/' + lights[l]['id'],
                                     'split': split, 'view': v, 'light': l})
    return {'positions': [[0., 0., 0.]], 'normals': [[0., 0., 1.]],
            'base_tangent': [0., 1., 0.], 'base_bitangent': [-1., 0., 0.],
            'views': views, 'lights': lights, 'observations': observations}


def lighting(directions, lights):
    values = []
    for light in lights:
        values.append(np.ones(len(directions)) if light['axis'] is None else
                      .1 + .9 * np.maximum(directions @ np.asarray(light['axis']), 0.) ** light['power'])
    return np.stack(values, axis=1)


def make_grids(specifications=GRIDS):
    lights = design()['lights']
    grids = []
    for n, m in specifications:
        wi, weights = reference.hemisphere(n, m)
        illumination = lighting(wi, lights)
        grids.append({'directions': wi, 'weights': weights, 'lighting': illumination,
                      'diffuse_integral': (weights * wi[:, 2]) @ illumination / np.pi,
                      'shape': [n, m], 'directions_sha256': sha_array(wi),
                      'weights_sha256': sha_array(weights), 'lighting_sha256': sha_array(illumination)})
    return grids


def render(parameters, observations, grids, budget, check=lambda: None):
    """Cache exactly identical per-view kernels across lights and RGB channels.

    Fresnel is affine in F0: F=s+(1-s)*F0. This permits two scalar integrals
    per view/light while retaining independent Cartesian NDF quadrature.
    """
    p = np.asarray(parameters, np.float64)
    require(p.shape == (7,) and np.isfinite(p).all(), 'Seven finite physical parameters required')
    require(np.all((p[:2] >= .04) & (p[:2] <= 1)) and np.all((p[3:] >= 0) & (p[3:] <= 1)),
            'Reference parameters outside physical domain')
    scene = design(); d = p[3:6] * (1 - p[6]); f0 = .04 * (1 - p[6]) + p[6] * p[3:6]
    levels = []
    for grid in grids:
        check(); wi, weights, lights = [grid[k] for k in ('directions', 'weights', 'lighting')]
        out = np.empty((len(observations), 3, 3), np.float64)
        for v in sorted({o['view'] for o in observations}):
            check(); wo = np.asarray(scene['views'][v]['direction'])
            require(budget.get('reference_direction_evaluations', 0) + len(wi) <= 495452160,
                    'Declared combined reference budget exceeded')
            h = reference.unit(wi + wo); dot = h @ wo
            density = reference.ndf(h, p[0], p[1], p[2])
            s = (1 - np.clip(dot, 0, 1)) ** 5
            factor = weights * wi[:, 2] * density / (4 * dot)
            a = (factor * (1 - s)) @ lights
            b = (factor * s) @ lights
            budget['reference_direction_evaluations'] = budget.get('reference_direction_evaluations', 0) + len(wi)
            for i, obs in enumerate(observations):
                if obs['view'] != v: continue
                l = obs['light']; diffuse = grid['diffuse_integral'][l] * d
                specular = a[l] * f0 + b[l]
                out[i] = np.stack([diffuse, specular, diffuse + specular])
        require(np.isfinite(out).all(), 'Nonfinite independent reference')
        levels.append(out)
    require(len(levels) == 2, 'Exactly two reference grids required')
    coarse, fine = levels; delta = abs(coarse - fine); limit = ATOL + RTOL * abs(fine)
    converged = delta <= limit
    converged[:, 2] &= converged[:, 0] & converged[:, 1]
    return {'coarse': coarse, 'fine': fine, 'absolute_delta': delta, 'limit': limit,
            'converged': converged, 'passed': bool(converged.all())}


def serial_reference(result):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}


def jacobian_gate(matrices):
    """Thresholds unchanged; Weyl bound certifies robustness to observed drift.

    matrices[step,grid,observation*RGB,parameter] are scale-normalized columns.
    No fitted-model derivative enters this gate.
    """
    x = np.asarray(matrices, np.float64)
    require(x.ndim == 4 and x.shape[:2] == (2, 2) and x.shape[-1] == 7 and
            x.shape[-2] >= 7 and np.isfinite(x).all(), 'Two-step/two-grid finite seven-column Jacobians required')
    reports = []
    for i in range(2):
        u, singular, vh = np.linalg.svd(x[i, 1], full_matrices=False)
        ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.
        reports.append({'step': FD_STEPS[i], 'singular_values': singular.tolist(), 'ratio': ratio,
                        'rank_at_declared_ratio': int(np.count_nonzero(singular >= RANK_RATIO * singular[0])) if singular[0] > 0 else 0,
                        'right_singular_vectors': vh.tolist(),
                        'null_directions_at_declared_ratio': vh[singular < RANK_RATIO * singular[0]].tolist() if singular[0] > 0 else vh.tolist()})
    step_delta = float(np.linalg.norm(x[0, 1] - x[1, 1], ord=2))
    grid_deltas = [float(np.linalg.norm(x[i, 1] - x[i, 0], ord=2)) for i in range(2)]
    smallest = min(r['singular_values'][-1] for r in reports)
    robust = max(step_delta, *grid_deltas) < smallest
    return {'passed': all(r['ratio'] >= RANK_RATIO for r in reports) and robust,
            'threshold': RANK_RATIO, 'steps': reports, 'step_spectral_discrepancy': step_delta,
            'grid_spectral_discrepancies': grid_deltas, 'smallest_singular_value': smallest,
            'stable_rank_under_observed_discrepancy': bool(robust),
            'scope': 'Local independent-integral rank only; does not establish global uniqueness'}


def identifiability(grids, observations, budget, check, progress):
    observations = [o for o in observations if o['split'] == 'fit']
    matrices = np.empty((2, 2, 3 * len(observations), 7), np.float64)
    evaluations = []
    for i, step in enumerate(FD_STEPS):
        for column, scale in enumerate(SCALES):
            values = []
            for sign in (-1, 1):
                p = TRUTH.copy(); p[column] += sign * step * scale
                result = render(p, observations, grids, budget, check)
                evaluations.append({'step': step, 'parameter': PARAMETERS[column], 'sign': sign,
                                    'physical_parameters': p.tolist(), **serial_reference(result)})
                if not result['passed']:
                    return {'passed': False, 'reason': 'Perturbed reference convergence failed',
                            'evaluations': evaluations, 'jacobians': None}
                values.append(result)
            for level, name in enumerate(('coarse', 'fine')):
                # J_physical * scale: derivative w.r.t. normalized coordinates.
                matrices[i, level, :, column] = ((values[1][name][:, 2] - values[0][name][:, 2]) / (2 * step)).reshape(-1)
            progress(i, column)
    return {**jacobian_gate(matrices), 'jacobians': matrices.tolist(),
            'parameter_order': PARAMETERS, 'parameter_scales': SCALES.tolist(), 'evaluations': evaluations}
