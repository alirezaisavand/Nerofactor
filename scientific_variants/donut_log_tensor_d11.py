"""Log-Euclidean penalty for the D8-T physical tangent tensor."""
import torch

from infrastructure.signatures import require
from infrastructure.surface_q_regularization import transported_q_loss
from scientific_variants.donut_tensor_d8 import symmetric, world_tensor


def tangent_log(q):
    """Logarithm of a symmetric positive definite 2 by 2 tangent tensor."""
    q = q.double()
    a, d = q[..., 0, 0], q[..., 1, 1]
    b = (q[..., 0, 1]+q[..., 1, 0])/2
    mean, difference = (a+d)/2, (a-d)/2
    determinant = a*d-b.square()
    require(bool((torch.isfinite(q).all()) & (mean > 0).all() & (determinant > 0).all()),
            'D11 requires finite positive definite tangent Q')
    z = (difference/mean).square()+(b/mean).square()
    # Replace the unused argument before sqrt and division. At isotropy,
    # autograd sees a smooth polynomial in the squared normalized radius.
    near = z <= 1e-6
    r = torch.sqrt(torch.where(near, torch.full_like(z, .25), z))
    ratio = torch.where(near, 1+z*(1/3+z*(1/5+z/7)), (torch.log1p(r)-torch.log1p(-r))/(2*r))/mean
    diagonal = .5*torch.log(determinant)
    return symmetric(diagonal+ratio*difference, diagonal-ratio*difference, ratio*b)


def world_log_tensor(q, tangent):
    # A has eigenvalue 1 along n, so log(A) has zero normal component.
    return world_tensor(tangent_log(q), tangent)


def surface_log_q_term(shader, batch, policy, torch):
    fields = []
    for suffix in ('0', '1'):
        normal = batch['normals'+suffix]
        q, frame, _, _ = shader.predict_tensor(batch['points'+suffix], normal)
        fields.append(world_log_tensor(q, frame).to(normal.dtype))
    _, delta = transported_q_loss(*fields, batch['normals0'], batch['normals1'],
                                  torch.ones_like(batch['distance']), torch)
    return policy['coefficient']*(delta*(policy['reference_distance']/batch['distance']).square()).mean()
