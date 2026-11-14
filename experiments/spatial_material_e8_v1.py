"""E8 spatial benchmark/head projection. Scientific construction is explicit/lazy."""
from contextlib import contextmanager
import numpy as np
from infrastructure.signatures import require
from experiments.neural_projection_e7_v1 import HEADS

TRAIN = ((-.1, -.1, 0.), (-.1, .1, 0.), (.1, -.1, 0.), (.1, .1, 0.))
HELD = ((-.04, .02, 0.), (.045, -.025, 0.))
SEEDS = (101, 103)
ARMS = ('Adam', 'guided', 'no_retention')
POLICY = dict(rounds=12, damping=.01, retention=1., ridge_fraction=.001,
              feature_scale_floor=.001, logit_radius=.5, retention_radius=.05,
              other_loss_fraction=.1, other_loss_atol=1e-10, rotation_radius=.5)


def truth(points):
    """Smooth analytic *benchmark only*, never used by correction/projection."""
    x = np.asarray(points, np.float64)
    require(x.ndim == 2 and x.shape[1] == 3 and np.isfinite(x).all() and
            (abs(x[:, :2]) <= .1).all() and (x[:, 2] == 0).all(), 'Flat pilot coordinates required')
    u, v = x[:, 0]/.1, x[:, 1]/.1
    return np.stack((.28+.035*u+.015*v, .49-.025*u+.02*v, -.62+.18*u-.12*v,
                     .36+.04*u, .58+.03*v, .24+.02*(u+v), .25+.05*u-.04*v), -1)


def make_spatial(latent, seed):
    """Same original N1 modules/state inventory; query actual coordinates."""
    import torch
    from experiments import neural_material_field_v1 as field
    from experiments.material_shader_v1 import encoding_status
    class Spatial(field.NeuralMaterial):
        def forward(self, positions):
            field.cpu_float(positions, (len(positions), 3), 'spatial query positions')
            require(len(positions) > 0, 'Empty spatial field query')
            for name, p in self.named_parameters(): field.cpu_float(p, tuple(p.shape), 'parameter '+name)
            mx, my, metal, kd, encoding, raw = self._predict(self, positions)
            status = encoding_status(encoding); require(bool(status['valid'].all()), 'Invalid spatial orientation; no fallback')
            unit = encoding/status['norm']; angle = .5*torch.atan2(unit[:, 1:2], unit[:, :1])
            return field.Material(torch.cat((mx, my, angle, kd, metal), -1), encoding, raw,
                                  torch.norm(2*raw-1, dim=-1, keepdim=True)).validate()
    return Spatial(latent, seed)


@contextmanager
def monitor(model):
    """Passive hooks; usable on the actual differentiable shader path."""
    features, logits, probabilities, handles = {}, {}, {}, []
    try:
        for name, _ in HEADS:
            def feature(module, args, name=name):
                require(name not in features, 'Repeated feature hook')
                features[name] = args[0].detach().numpy().copy()
            def logit(module, args, name=name):
                require(name not in logits, 'Repeated logit hook')
                logits[name] = args[0].detach().numpy().copy()
            def probability(module, args, value, name=name):
                require(name not in probabilities, 'Repeated probability hook')
                probabilities[name] = value.detach().numpy().copy()
            seq = getattr(model, name).predictor
            handles.append(seq[-2].register_forward_pre_hook(feature))
            handles.append(seq[-1].register_forward_pre_hook(logit))
            handles.append(seq[-1].register_forward_hook(probability))
        yield features, logits, probabilities
    finally:
        for h in handles: h.remove()
    require(set(features) == set(logits) == set(probabilities) == {n for n, _ in HEADS}, 'Incomplete spatial capture')


def capture(model, points):
    """One field call: actual penultimate activations, centered logits, outputs."""
    import torch
    with monitor(model) as (features, logits, _):
        with torch.no_grad(): material = model(points)
    return material, features, np.concatenate([logits[n] for n, _ in HEADS], -1)


def shade_material(shader, batch, material):
    """Original A2 batched component/seed means with explicit spatial material."""
    import torch
    n = len(batch.positions); p = material.physical.expand(n, 7)
    result = shader.shade(batch.positions, batch.normals, batch.views, p[:, :1], p[:, 1:2],
        p[:, 3:6], p[:, 6:7], material.encoding.expand(n, 2), batch.illumination, batch.tapes, is_train=True)
    require(bool(result['frame_orthonormal'].all()), 'Invalid tangent frame')
    values = torch.stack([result[k].reshape(batch.seeds, batch.n, 3).mean(0) for k in ('diffuse_rgb','specular_rgb','linear_rgb')], 1)
    require(bool(torch.isfinite(values).all()), 'Nonfinite spatial rendering')
    return values


def embedding(p):
    """Continuous gauge-invariant physical trust coordinates; no eigenframes."""
    p = np.asarray(p, np.float64); c, s = np.cos(p[..., 2]), np.sin(p[..., 2])
    a, b = p[..., 0]**2, p[..., 1]**2
    return np.stack((a*c*c+b*s*s, np.sqrt(2)*(a-b)*c*s, a*s*s+b*c*c,
                     p[..., 3], p[..., 4], p[..., 5], p[..., 6]), -1)


def proposed_updates(features, logits, physical, selected, teacher, retention, account_solve=None):
    """Eight small dual ridge solves in preconditioned last-layer coordinates."""
    p = np.asarray(physical, np.float64); z = np.asarray(logits, np.float64); t = np.asarray(teacher, np.float64)
    require(p.shape == (4, 7) and z.shape == (4, 8) and t.shape == (7,) and selected in range(4) and
            retention in (0., 1.) and np.isfinite(p).all() and np.isfinite(z).all() and np.isfinite(t).all(), 'Invalid projection input')
    q = 1/(1+np.exp(-z))
    require(((q > 0)&(q < 1)).all(), 'Saturated input projection; no repair')
    e = np.array([np.cos(2*t[2]), np.sin(2*t[2])])
    target = np.r_[(t[:2]-.04)/.96, t[6], (1+.5*e)/2, t[3:6]]
    require(((target > 0)&(target < 1)).all(), 'Interior physical teacher required')
    wanted = np.log(target/(1-target))-z[selected]
    sensitivity = q*(1-q)
    gap = abs(p[:, 0]**2-p[:, 1]**2)/np.maximum(p[:, 0]**2, p[:, 1]**2)
    radius = np.linalg.norm(2*q[:, 3:5]-1, axis=1)
    require((radius >= 1e-6).all(), 'Invalid orientation radius')
    sensitivity[:, 3:5] *= (2*gap/radius)[:, None]
    weights = np.full(4, retention); weights[selected] = 1.
    updates = {}; predicted = np.zeros_like(z); offset = 0; systems = []
    for name, width in HEADS:
        h = np.asarray(features[name], np.float64)
        require(h.ndim == 2 and len(h) == 4 and np.isfinite(h).all(), 'Invalid shared features')
        mean = h.mean(0); scale = np.maximum(np.sqrt(np.mean((h-mean)**2, axis=0)), POLICY['feature_scale_floor'])
        design = np.c_[(h-mean)/scale, np.ones(4)]
        ridge = POLICY['ridge_fraction']*np.mean(np.sum(design**2, axis=1)); rows = []
        for j in range(width):
            k = offset+j; sw = np.sqrt(weights)*sensitivity[:, k]; a = sw[:, None]*design
            rhs = np.zeros(4); rhs[selected] = sw[selected]*wanted[k]
            if account_solve: account_solve('attempt')
            system = a@a.T+ridge*np.eye(4); chol = np.linalg.cholesky(system)
            v = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs)); coef = a.T@v
            require(np.isfinite(coef).all(), 'Nonfinite head solve')
            if account_solve: account_solve('complete')
            rows.append(coef); predicted[:, k] = design@coef
            systems.append(dict(head=name, channel=j, condition=float(np.linalg.cond(system)), ridge=float(ridge),
                                solve_residual_L2=float(np.linalg.norm(system@v-rhs))))
        coef = np.stack(rows); dw = coef[:, :-1]/scale; db = coef[:, -1]-dw@mean
        updates[name] = (dw, db); offset += width
    alpha = min(1., POLICY['logit_radius']/max(float(abs(predicted).max()), 1e-30))
    return {n: (w*alpha, b*alpha) for n, (w, b) in updates.items()}, dict(alpha=alpha,
        predicted_logit_changes=(predicted*alpha).tolist(), desired_logit_change=wanted.tolist(), systems=systems)


@contextmanager
def transfer(model, updates):
    """Atomic weight-norm-aware last-layer update, with exact rejection rollback."""
    import torch
    saved = {}; committed = False; normal = False
    def commit(): nonlocal committed; committed = True
    try:
        with torch.no_grad():
            for name, (dw, db) in updates.items():
                layer = getattr(model, name).predictor[-2]
                for key in ('weight_g', 'weight_v', 'bias'):
                    saved[name, key] = getattr(layer, key).detach().clone()
                effective = layer.weight_g*layer.weight_v/torch.norm(layer.weight_v, dim=1, keepdim=True)
                w = effective+torch.from_numpy(dw).to(effective.dtype); norm = torch.norm(w, dim=1, keepdim=True)
                require(bool(torch.isfinite(w).all()) and bool((norm > 0).all()), 'Invalid weight-norm target')
                layer.weight_v.copy_(w); layer.weight_g.copy_(norm)
                layer.bias.add_(torch.from_numpy(db).to(layer.bias.dtype))
        yield commit
        normal = True
    finally:
        if not (normal and committed):
            with torch.no_grad():
                for (name, key), value in saved.items(): getattr(getattr(model, name).predictor[-2], key).copy_(value)


def accept(before_p, after_p, target, selected, before_loss, after_loss, probabilities):
    """Only training outputs/losses and teacher; held-out/truth cannot enter."""
    before, after, desired = embedding(before_p), embedding(after_p), embedding(target)
    old, new = np.asarray(before_loss), np.asarray(after_loss); q = np.asarray(probabilities)
    require(old.shape == new.shape == (4,) and np.isfinite(old).all() and np.isfinite(new).all() and (old >= 0).all() and (new >= 0).all(), 'Invalid training objective')
    others = [j for j in range(4) if j != selected]
    intended = np.linalg.norm(desired-before[selected]); residual = np.linalg.norm(after[selected]-desired)
    drift = np.linalg.norm(after[others]-before[others], axis=1)
    gates = dict(realization=bool(intended > 1e-12 and residual < intended),
        retention=bool((drift <= POLICY['retention_radius']).all()),
        others_photometric=bool((new[others] <= (1+POLICY['other_loss_fraction'])*old[others]+POLICY['other_loss_atol']).all()),
        interior=bool(np.isfinite(q).all() and ((q > 0)&(q < 1)).all()),
        joint_photometric=bool(old.mean()-new.mean() > 16*np.finfo(np.float32).eps*max(old.mean(),new.mean(),1e-12)))
    return dict(accepted=bool(all(gates.values())), gates=gates, intended_embedding_L2=float(intended),
        residual_embedding_L2=float(residual), other_embedding_drift=drift.tolist(),
        before_loss=old.tolist(), after_loss=new.tolist())
