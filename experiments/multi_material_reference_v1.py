"""3D-1 explicit-truth NumPy reference adapters; no historical global mutation."""
import numpy as np

from experiments import known_material_reference_v1 as original
from infrastructure.signatures import require

PANEL = {
    'M1': [.28, .49, -.62, .36, .58, .24, .10],
    'M2': [.17, .62, .91, .56, .31, .18, .65],
    'M3': [.37, .47, .19, .24, .43, .70, .35],
}
PER_MATERIAL_DIRECTIONS = 117964800
REFERENCE_CAP = 3 * PER_MATERIAL_DIRECTIONS


def truth(material):
    require(material in PANEL, 'Unknown preregistered material')
    return validate(PANEL[material])


def validate(value):
    p = np.asarray(value, np.float64)
    require(p.shape == (7,) and np.isfinite(p).all() and
            ((p[:2] >= .04) & (p[:2] <= 1)).all() and
            ((p[3:] >= 0) & (p[3:] <= 1)).all(), 'Invalid physical material domain')
    return p.copy()


def render(parameters, observations, grids, budget, check=lambda: None):
    p = validate(parameters)
    expected = len({o['view'] for o in observations}) * sum(len(g['directions']) for g in grids)
    before = budget.get('reference_direction_evaluations', 0)
    require(before + expected <= REFERENCE_CAP, 'Frozen panel reference work cap exhausted')
    result = original.render(p, observations, grids, budget, check)
    require(budget['reference_direction_evaluations'] - before == expected, 'Reference cached-view accounting changed')
    result.update(physical_parameters=p.tolist(), physical_sha256=original.sha_array(p),
                  coarse_sha256=original.sha_array(result['coarse']), fine_sha256=original.sha_array(result['fine']),
                  maximum_bound_ratio=float(np.max(result['absolute_delta'] / result['limit'])),
                  component_maximum_bound_ratios={name:float(np.max(result['absolute_delta'][:, i] / result['limit'][:, i]))
                      for i,name in enumerate(('diffuse','specular','total'))},
                  direction_evaluations=expected)
    return result


def identifiability(parameters, grids, observations, budget, check, progress):
    """Original normalized two-step/two-grid central differences at explicit truth.

    Complete exactly 28 perturbations; retain all failed perturbation gates.
    No renderer/model derivatives, grid refinement or observation redesign.
    """
    truth_value = validate(parameters)
    selected = [o for o in observations if o['split'] == 'fit']
    require(len(selected) == 24, 'Original 24 fitting observations required')
    matrices = np.empty((2, 2, 72, 7), np.float64)
    evaluations = []
    for i, step in enumerate(original.FD_STEPS):
        for column, scale in enumerate(original.SCALES):
            values = []
            for sign in (-1, 1):
                p = truth_value.copy()
                p[column] += sign * step * scale
                result = render(p, selected, grids, budget, check)
                evaluations.append(dict(step=step, parameter=original.PARAMETERS[column], sign=sign,
                    **original.serial_reference(result)))
                values.append(result)
            for level,name in enumerate(('coarse','fine')):
                matrices[i,level,:,column] = ((values[1][name][:,2]-values[0][name][:,2])/(2*step)).reshape(-1)
            progress(i,column)
    gate = original.jacobian_gate(matrices)
    convergence = all(e['passed'] for e in evaluations)
    smallest = gate['smallest_singular_value']
    drift = max(gate['step_spectral_discrepancy'], *gate['grid_spectral_discrepancies'])
    conditions = [[float(np.linalg.cond(matrices[i,j])) for j in range(2)] for i in range(2)]
    singular = [[np.linalg.svd(matrices[i,j],compute_uv=False) for j in range(2)] for i in range(2)]
    return dict(**{**gate, 'passed': bool(gate['passed'] and convergence)},
        explicit_truth=truth_value.tolist(), perturbation_convergence_passed=convergence,
        jacobian_gate_passed=gate['passed'], evaluations=evaluations, jacobians=matrices.tolist(),
        jacobians_sha256=original.sha_array(matrices), parameter_order=original.PARAMETERS,
        parameter_scales=original.SCALES.tolist(),
        condition_numbers=[[v if np.isfinite(v) else None for v in row] for row in conditions],
        condition_number_nonfinite=[[not bool(np.isfinite(v)) for v in row] for row in conditions],
        grid_singular_values=[[v.tolist() for v in row] for row in singular],
        grid_numerical_ranks=[[int(np.count_nonzero(v >= original.RANK_RATIO*v[0])) if v[0] > 0 else 0 for v in row] for row in singular],
        smallest_singular_value_stability_margin=smallest-drift,
        failure_reason=None if convergence and gate['passed'] else
            'Perturbation convergence failed' if not convergence else 'Original independent Jacobian rank/stability gate failed')
