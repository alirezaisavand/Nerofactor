"""E7 generic final-bias projection; no original N1/shader imports."""
from contextlib import contextmanager
from types import FunctionType
import numpy as np
from infrastructure.signatures import require

HEADS = (('mx_predictor', 1), ('my_predictor', 1), ('metallic_predictor', 1),
         ('rotation_predictor', 2), ('kd_predictor', 3))
RHO = .5


def target_logits(physical):
    """Seven physical values -> eight head logits; fixed nonzero radial gauge.

    No material truth, optimizer or renderer is used. A near-isotropic teacher
    retains its supplied sampling gauge; no angle prior/eigengap division.
    """
    import torch
    p = physical
    require(p.shape == (7,) and p.device.type == 'cpu' and bool(torch.isfinite(p).all()), 'Finite seven-value CPU teacher required')
    e = torch.stack((torch.cos(2*p[2]), torch.sin(2*p[2])))
    q = torch.cat(((p[:2]-.04)/.96, p[6:7], (1+RHO*e)/2, p[3:6]))
    require(bool(((q > 0) & (q < 1)).all()), 'Projection requires interior probabilities; no clipping or saturation repair')
    return torch.log(q/(1-q))


def bias_parameters(model):
    result = {}
    for name, width in HEADS:
        seq = getattr(model, name).predictor; layer = seq[-2]; p = layer.bias
        require(p is not None and tuple(p.shape) == (width,), 'Original final linear bias missing: '+name)
        key = name+'.predictor.'+str(len(seq)-2)+'.bias'
        require(dict(model.named_parameters()).get(key) is p, 'Ambiguous final bias registration')
        result[key] = p
    return result


def capture(model, positions):
    """One actual field query, plus noninvasive final-sigmoid input hooks."""
    import torch
    values = {}; handles = []
    try:
        for name, _ in HEADS:
            def keep(module, args, name=name):
                require(name not in values, 'Repeated head call during projection capture')
                values[name] = args[0].detach().clone()
            handles.append(getattr(model, name).predictor[-1].register_forward_pre_hook(keep))
        with torch.no_grad(): material = model(positions)
    finally:
        for h in handles: h.remove()
    require(set(values) == {n for n, _ in HEADS}, 'Missing actual head logits')
    logits = torch.cat([values[n].reshape(-1) for n, _ in HEADS])
    require(logits.shape == (8,) and bool(torch.isfinite(logits).all()), 'Constant-anchor logits required')
    return material, logits


@contextmanager
def bias_transaction(model, old_logits, desired):
    """Minimum Euclidean bias displacement for the prescribed logit target.

    da/db=I at fixed features/weights/calibration. Commit is explicit; failures
    and rejected candidates restore exact old bias bytes, never old evidence.
    """
    import torch
    params = bias_parameters(model); saved = {n: p.detach().clone() for n, p in params.items()}
    delta = target_logits(desired).to(old_logits.dtype)-old_logits
    require(delta.shape == (8,) and bool(torch.isfinite(delta).all()), 'Invalid projection displacement')
    committed = [False]; normal_exit = False
    def commit(): committed[0] = True
    try:
        with torch.no_grad():
            offset = 0
            for p in params.values():
                width = p.numel(); p.add_(delta[offset:offset+width].reshape_as(p)); offset += width
        yield dict(commit=commit, bias_names=list(params), delta=delta.detach().numpy().copy(),
                   changed_coordinates=int(torch.count_nonzero(delta)))
        normal_exit = True
    finally:
        if not (committed[0] and normal_exit):
            with torch.no_grad():
                for n, p in params.items(): p.copy_(saved[n])


def realization(physical, encoding, desired):
    """No angle score near isotropy: same physical tensor and supplied encoding."""
    p, t = np.asarray(physical, np.float64), np.asarray(desired, np.float64)
    e = np.array(encoding, np.float64, copy=True); norm = np.linalg.norm(e)
    if not np.isfinite(norm) or norm < 1e-6:
        return dict(passed=False, reason='Invalid projected orientation encoding')
    e /= norm
    target_e = np.array([np.cos(2*t[2]), np.sin(2*t[2])])
    def q(v):
        c, s = np.cos(v[2]), np.sin(v[2]); r = np.array([[c, -s], [s, c]])
        return r@np.diag(v[:2]**2)@r.T
    physical_error = float(np.max(abs(p[[0, 1, 3, 4, 5, 6]]-t[[0, 1, 3, 4, 5, 6]])))
    q_error = float(np.linalg.norm(q(p)-q(t))/max(np.linalg.norm(q(t)), 1e-30))
    encoding_error = float(np.max(abs(e-target_e)))
    return dict(passed=physical_error <= 2e-6 and q_error <= 1e-5 and encoding_error <= 2e-5,
        physical_max_error=physical_error, relative_Q_error=q_error, unit_encoding_max_error=encoding_error)


def accept(old, new, projection_passed):
    require(np.isfinite([old, new]).all() and old >= 0 and new >= 0, 'Finite nonnegative objective values required')
    floor = 16*np.finfo(np.float32).eps*max(old, new, 1e-12)
    return bool(projection_passed and old-new > floor)


def restore100(model, optimizer, arrays, rng, options, identity):
    """Reuse original strict restorer with a private start-100 binding."""
    from experiments import neural_material_continuation_e3 as old
    original_start = old.START
    fn = FunctionType(old.restore_state.__code__, dict(old.restore_state.__globals__, START=100),
                      'restore_E7_state100', old.restore_state.__defaults__, old.restore_state.__closure__)
    result = fn(model, optimizer, arrays, rng, options, identity)
    require(old.START == original_start and result['Adam_step'] == 100, 'Frozen restorer globals changed')
    return result
