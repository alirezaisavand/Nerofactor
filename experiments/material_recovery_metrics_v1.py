"""Small analytical diagnostics/metrics; no fitting, data generation or I/O."""
import numpy as np

from audits import anisotropic_brdf as reference
from infrastructure.signatures import require

PROTOCOL = 'known-material-recovery-metrics-v1'
ANISOTROPY_GAP = .1


def roughness_tensor(mx, my, theta):
    """Physical 2x2 tensor in the declared unrotated tangent coordinates."""
    require(np.isfinite([mx, my, theta]).all() and .04 <= mx <= 1 and .04 <= my <= 1,
            'Finite original physical widths [.04,1] and orientation required')
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.diag([mx * mx, my * my]) @ rotation.T


def gauge_metrics(predicted, truth):
    """(mx,my,theta) errors; orientation is not scored near isotropy."""
    p, g = np.asarray(predicted, np.float64), np.asarray(truth, np.float64)
    require(p.shape == g.shape == (3,), 'Explicit width/width/theta triples required')
    qp, qg = roughness_tensor(*p), roughness_tensor(*g)
    gap = lambda v: abs(v[0] ** 2 - v[1] ** 2) / (v[0] ** 2 + v[1] ** 2)
    pg, gg = gap(p), gap(g)
    valid = gg >= ANISOTROPY_GAP and pg >= ANISOTROPY_GAP
    major_angle = lambda v: v[2] + (np.pi / 2 if v[1] > v[0] else 0.)
    delta = major_angle(p) - major_angle(g)
    orientation = abs(.5 * np.arctan2(np.sin(2 * delta), np.cos(2 * delta)))
    return {'protocol': PROTOCOL, 'sorted_axis_absolute_errors': abs(np.sort(p[:2]) - np.sort(g[:2])).tolist(),
            'Q_absolute_frobenius': float(np.linalg.norm(qp - qg)),
            'Q_relative_frobenius': float(np.linalg.norm(qp - qg) / np.linalg.norm(qg)),
            'truth_anisotropy_gap': float(gg), 'prediction_anisotropy_gap': float(pg),
            'orientation_identifiable': bool(valid),
            'major_axis_error_radians': float(orientation) if valid else None,
            'orientation_reason': None if valid else 'Truth or prediction is near-isotropic; retain Q/axis errors, do not score orientation as zero'}


def integrate_unit_energy(mx, my, theta, albedo, metallic, view, quadrature):
    """Independent float64 Cartesian hemisphere integral, n=+Z.

    Base tangent=+Y, bitangent=-X matches A. Caller explicitly supplies a grid;
    no adaptive retry, random samples or source Monte Carlo weights are used.
    Results are linear RGB, not gamma-encoded colors. No physical clipping.
    """
    roughness_tensor(mx, my, theta)
    k, view = np.asarray(albedo, np.float64), np.asarray(view, np.float64)
    require(k.shape == view.shape == (3,) and np.isfinite(k).all() and np.isfinite(view).all() and
            ((k >= 0) & (k <= 1)).all() and np.isfinite(metallic) and 0 <= metallic <= 1,
            'Finite RGB albedo, metallic and view required')
    require(abs(np.linalg.norm(view) - 1) < 1e-12 and view[2] > 0, 'Unit upper view required')
    wi, weights = [np.asarray(v, np.float64) for v in quadrature]
    require(wi.ndim == 2 and wi.shape[1] == 3 and weights.shape == (len(wi),) and len(wi) > 0,
            'Explicit nonempty hemisphere quadrature required')
    require(np.isfinite(wi).all() and np.isfinite(weights).all() and (weights > 0).all() and
            (wi[:, 2] > 0).all() and np.max(abs(np.linalg.norm(wi, axis=1) - 1)) < 1e-12,
            'Finite unit upper directions and positive weights required')
    require(abs(weights.sum() - 2 * np.pi) < 1e-10, 'Hemisphere weights must integrate unit solid angle')
    d = k * (1 - metallic)
    f0 = .04 * (1 - metallic) + metallic * k
    spec = np.array([np.sum(weights * wi[:, 2] * reference.kernel(wi, view, mx, my, theta, f)) for f in f0])
    diffuse = (weights * wi[:, 2]).sum() * d / np.pi
    require(np.isfinite(spec).all() and np.isfinite(diffuse).all(), 'Nonfinite independent integral')
    return {'diffuse': diffuse, 'specular': spec, 'total': diffuse + spec}


def energy_diagnostics(result, coarse, fine, absolute_tolerance=2e-5):
    """Detached observations only: never a loss or automatic regularizer change.

    One point per comparison; reference values are explicitly provisional on
    failed two-grid convergence. A matching integral is not energy conservation.
    """
    require(np.isfinite(absolute_tolerance) and absolute_tolerance > 0, 'Positive predeclared tolerance required')
    rows = {}
    for component in ('diffuse', 'specular', 'total'):
        a, b = np.asarray(coarse[component]), np.asarray(fine[component])
        require(a.shape == b.shape == (3,) and np.isfinite(a).all() and np.isfinite(b).all(),
                'Finite RGB references required')
        rows[component] = {'coarse': a.tolist(), 'fine': b.tolist(),
                           'absolute_delta': abs(a - b).tolist(),
                           'converged': bool(np.all(abs(a - b) <= absolute_tolerance))}
    rows['total']['converged'] &= rows['diffuse']['converged'] and rows['specular']['converged']
    def rgb(key):
        tensor = result[key]
        require(tuple(tensor.shape) == (1, 3) and tensor.device.type == 'cpu', 'One explicit CPU point required')
        value = tensor.detach().numpy()[0].astype(np.float64)
        require(np.isfinite(value).all(), 'Nonfinite energy diagnostic')
        return value
    conditional, unconditional, d = rgb('original_f_s_sum'), rgb('unconditional_reflectance_proxy'), rgb('f_d')
    return {'protocol': PROTOCOL, 'mode': 'read_only_no_loss', 'shader_identity': result['identity'],
            'conditional_specular_proxy': conditional.tolist(), 'unconditional_specular_proxy': unconditional.tolist(),
            'conditional_total_proxy': (d + conditional).tolist(), 'unconditional_total_proxy': (d + unconditional).tolist(),
            'original_L_spec': rgb('original_L_spec').tolist(), 'independent_integrals': rows,
            'reference_status': 'converged' if rows['total']['converged'] else 'unresolved',
            'unconditional_minus_reference': (unconditional - fine['specular']).tolist() if rows['specular']['converged'] else None,
            'loss_coefficients_changed': False}
