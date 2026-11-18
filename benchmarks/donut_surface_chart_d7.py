"""Chart-coordinate control on the exact D6 surface neighborhoods."""

import hashlib
import json

import numpy as np

from benchmarks import donut_surface_q_d6 as d6
from infrastructure.signatures import require

PROTOCOL = 'donut-kurt-surface-chart-d7-v1'
POLICY = dict(d6.POLICY, angle_weight=.00638, coordinate_loss='squared_widths_and_axial_chord_v1')
LIMITS = d6.LIMITS


def array_hash(values):
    array = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder('<'))
    sha = hashlib.sha256(json.dumps([array.dtype.str, array.shape]).encode('ascii'))
    sha.update(array.tobytes())
    return sha.hexdigest()


def chart_difference(widths0, widths1, rotation0, rotation1, weight, torch):
    """Compare fixed labels of squared widths and chart-local double angles."""
    rotations = []
    for vector in (rotation0, rotation1):
        length = torch.linalg.norm(vector, dim=-1, keepdim=True)
        require(bool(torch.isfinite(length).all() and (length >= 1e-6).all()),
                'Invalid chart orientation encoding')
        rotations.append(vector / length)
    return ((widths0.square() - widths1.square()).square().sum(-1)
            + weight * (rotations[0] - rotations[1]).square().sum(-1))


def surface_chart_term(shader, batch, policy, torch):
    # The same two full-field predictions as D6. No albedo/metallic companion.
    a = shader.predict_anisotropic_components(batch['points0'])
    b = shader.predict_anisotropic_components(batch['points1'])
    d = batch['distance']
    require(d.shape == (len(a[0]),) and bool((torch.isfinite(d) & (d >= 1e-6)).all()),
            'Invalid surface distance')
    loss = chart_difference(torch.cat(a[:2], dim=-1), torch.cat(b[:2], dim=-1),
                            a[4], b[4], policy['angle_weight'], torch)
    return policy['coefficient'] * (loss * (policy['reference_distance'] / d).square()).mean()
