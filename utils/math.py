import torch
import torch.nn.functional as F

def log10(x):
    return torch.log(x) / torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype))

class SafeAtan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, eps=1e-6):
        ctx.save_for_backward(x, y)
        ctx.eps = eps
        return torch.atan2(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        denom = x ** 2 + y ** 2 + ctx.eps
        dzdx = y / denom
        dzdy = -x / denom
        return grad_output * dzdx, grad_output * dzdy, None

def safe_atan2(x, y, eps=1e-6):
    return SafeAtan2.apply(x, y, eps)

class SafeAcos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-6):
        x_clip = torch.clamp(x, -1.0, 1.0)
        ctx.save_for_backward(x_clip)
        ctx.eps = eps
        return torch.acos(x_clip)

    @staticmethod
    def backward(ctx, grad_output):
        (x_clip,) = ctx.saved_tensors
        in_sqrt = 1.0 - x_clip ** 2 + ctx.eps
        denom = torch.sqrt(in_sqrt) + ctx.eps
        dydx = -1.0 / denom
        return grad_output * dydx, None

def safe_acos(x, eps=1e-6):
    return SafeAcos.apply(x, eps)

def safe_l2_normalize(x, dim=None, eps=1e-6):
    norm = torch.linalg.norm(x, dim=dim, keepdim=True) + eps
    return x / norm

def safe_cumprod(x, eps=1e-6):
    x = x + eps
    return torch.cumprod(x, dim=-1, exclusive=True)

def inv_transform_sample(val, weights, n_samples, det=False, eps=1e-5):
    denom = torch.sum(weights, dim=-1, keepdim=True) + eps
    pdf = weights / denom
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1)

    if det:
        u = torch.linspace(0.0, 1.0, n_samples, device=val.device, dtype=val.dtype)
        u = u.expand(*cdf.shape[:-1], n_samples)
    else:
        u = torch.rand(*cdf.shape[:-1], n_samples, device=val.device, dtype=val.dtype)

    ind = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(ind - 1, min=0)
    above = torch.clamp(ind, max=cdf.shape[-1] - 1)

    ind_g = torch.stack((below, above), dim=-1)
    cdf_g = torch.gather(cdf, -1, ind_g)
    val_g = torch.gather(val, -1, ind_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom
    samples = val_g[..., 0] + t * (val_g[..., 1] - val_g[..., 0])
    return samples
