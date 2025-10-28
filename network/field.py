import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import nvdiffrast.torch as dr
import mcubes

from utils.base_utils import az_el_to_points, sample_sphere
from utils.raw_utils import linear_to_srgb
from utils.ref_utils import generate_ide_fn
import open3d as o3d
from scipy.spatial import cKDTree


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 sdf_activation='none',
                 layer_activation='softplus'):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if layer_activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        elif layer_activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def sdf_normal(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return y[..., :1].detach(), gradients.detach()


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation='exp'):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        if self.act == 'exp':
            return torch.ones([*x.shape[:-1], 1]) * torch.exp(self.variance * 10.0)
        elif self.act == 'linear':
            return torch.ones([*x.shape[:-1], 1]) * self.variance * 10.0
        elif self.act == 'square':
            return torch.ones([*x.shape[:-1], 1]) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        return torch.ones([*x.shape[:-1], 1]) * inv_s


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRFNetwork(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRFNetwork, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

    def density(self, input_pts):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha


class IdentityActivation(nn.Module):
    def forward(self, x): return x


class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light = max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))


def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid',
                   exp_max=0.0) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation == 'none':
        activation = IdentityActivation()
    elif activation == 'relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256
    if weight_norm:
        module = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module = nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module


def get_camera_plane_intersection(pts, dirs, poses):
    """
    compute the intersection between the rays and the camera XoY plane
    :param pts:      pn,3
    :param dirs:     pn,3
    :param poses:    pn,3,4
    :return:
    """
    R, t = poses[:, :, :3], poses[:, :, 3:]

    # transfer into human coordinate
    pts_ = (R @ pts[:, :, None] + t)[..., 0]  # pn,3
    dirs_ = (R @ dirs[:, :, None])[..., 0]  # pn,3

    hits = torch.abs(dirs_[..., 2]) > 1e-4
    dirs_z = dirs_[:, 2]
    dirs_z[~hits] = 1e-4
    dist = -pts_[:, 2] / dirs_z
    inter = pts_ + dist.unsqueeze(-1) * dirs_
    return inter, dist, hits


def expected_sin(mean, var):
    """Compute the mean of sin(x), x ~ N(mean, var)."""
    return torch.exp(-0.5 * var) * torch.sin(mean)  # large var -> small value.


def IPE(mean, var, min_deg, max_deg):
    scales = 2 ** torch.arange(min_deg, max_deg)
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
    scaled_var = torch.reshape(var[..., None, :] * scales[:, None] ** 2, shape)
    return expected_sin(torch.cat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1),
                        torch.cat([scaled_var] * 2, dim=-1))


def offset_points_to_sphere(points):
    points_norm = torch.norm(points, dim=-1)
    mask = points_norm > 0.999
    if torch.sum(mask) > 0:
        points = torch.clone(points)
        points[mask] /= points_norm[mask].unsqueeze(-1)
        points[mask] *= 0.999
        # points[points_norm>0.999] = 0
    return points


def get_sphere_intersection(pts, dirs):
    dtx = torch.sum(pts * dirs, dim=-1, keepdim=True)  # rn,1
    xtx = torch.sum(pts ** 2, dim=-1, keepdim=True)  # rn,1
    dist = dtx ** 2 - xtx + 1
    assert torch.sum(dist < 0) == 0
    dist = -dtx + torch.sqrt(dist + 1e-6)  # rn,1
    return dist


# this function is borrowed from NeuS
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_weights(sdf_fun, inv_fun, z_vals, origins, dirs):
    points = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2)  # pn,sn,3
    inv_s = inv_fun(points[:, :-1, :])[..., 0]  # pn,sn-1
    sdf = sdf_fun(points)[..., 0]  # pn,sn

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # pn,sn-1
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # pn,sn-1
    surface_mask = (cos_val < 0)  # pn,sn-1
    cos_val = torch.clamp(cos_val, max=0)

    dist = next_z_vals - prev_z_vals  # pn,sn-1
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # pn, sn-1
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) * surface_mask.float()
    weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    mid_sdf[~surface_mask] = -1.0
    return weights, mid_sdf


def get_intersection(sdf_fun, inv_fun, pts, dirs, sn0=128, sn1=9):
    """
    :param sdf_fun:
    :param inv_fun:
    :param pts:    pn,3
    :param dirs:   pn,3
    :param sn0:
    :param sn1:
    :return:
    """
    inside_mask = torch.norm(pts, dim=-1) < 0.999  # left some margin
    pn, _ = pts.shape
    hit_z_vals = torch.zeros([pn, sn1 - 1])
    hit_weights = torch.zeros([pn, sn1 - 1])
    hit_sdf = -torch.ones([pn, sn1 - 1])
    if torch.sum(inside_mask) > 0:
        pts = pts[inside_mask]
        dirs = dirs[inside_mask]
        max_dist = get_sphere_intersection(pts, dirs)  # pn,1
        with torch.no_grad():
            z_vals = torch.linspace(0, 1, sn0)  # sn0
            z_vals = max_dist * z_vals.unsqueeze(0)  # pn,sn0
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, pts, dirs)  # pn,sn0-1
            z_vals_new = sample_pdf(z_vals, weights, sn1, True)  # pn,sn1
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, pts, dirs)  # pn,sn1-1
            z_vals_mid = (z_vals_new[:, 1:] + z_vals_new[:, :-1]) * 0.5

        hit_z_vals[inside_mask] = z_vals_mid
        hit_weights[inside_mask] = weights
        hit_sdf[inside_mask] = mid_sdf
    return hit_z_vals, hit_weights, hit_sdf


class AppShadingNetwork(nn.Module):
    default_cfg = {
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 0.0,
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        feats_dim = 256

        # material MLPs
        self.metallic_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['metallic_init'] != 0:
            nn.init.constant_(self.metallic_predictor[-2].bias, self.cfg['metallic_init'])

        # Todo reset to isotropic roughness
        # self.roughness_predictor = make_predictor(feats_dim + 3, 1)
        self.roughness_predictor_x = make_predictor(feats_dim + 3, 1)
        self.roughness_predictor_y = make_predictor(feats_dim + 3, 1)
        if self.cfg['roughness_init'] != 0:
            # Todo reset to isotropic roughness
            # nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
            nn.init.constant_(self.roughness_predictor_x[-2].bias, self.cfg['roughness_init'])
            nn.init.constant_(self.roughness_predictor_y[-2].bias, self.cfg['roughness_init'])
        self.albedo_predictor = make_predictor(feats_dim + 3, 3)

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)
        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        if self.cfg['sphere_direction']:
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=exp_max)
        else:
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        # inner lights are indirect lights
        self.inner_light = make_predictor(pos_dim + 72 * 2, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='none')
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        # human lights are the lights reflected from the photo capturer
        if self.cfg['human_light']:
            self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))

    def predict_human_light(self, points, reflective, human_poses, roughness):
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_specular_lights(self, points, feature_vectors, reflective, roughness_x, roughness_y, human_poses, step):
        human_light, human_weight = 0, 0
        # Todo reset to isotropic roughness
        ref_roughness_x = self.sph_enc(reflective, roughness_x)
        ref_roughness_y = self.sph_enc(reflective, roughness_y)
        roughness = torch.sqrt(roughness_x * roughness_y + 1e-6)  # pn,1
        ref_roughness = torch.cat([ref_roughness_x, ref_roughness_y], -1)  # pn,72*2
        pts = self.pos_enc(points)
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1, eps=1e-6)
            sph_points = self.sph_enc(sph_points, roughness)
            direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1))
        else:
            direct_light = self.outer_light(ref_roughness)

        if self.cfg['human_light']:
            human_light, human_weight = self.predict_human_light(points, reflective, human_poses, roughness)

        indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
        ref_ = self.dir_enc(reflective)
        occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1))  # this is occlusion prob
        occ_prob = occ_prob * 0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob, min=0, max=1)

        light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
                1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, points, feature_vectors, normals):
        roughness_x = torch.ones([normals.shape[0], 1])
        roughness_y = torch.ones([normals.shape[0], 1])
        roughness = torch.sqrt(roughness_x * roughness_y + 1e-6)
        ref_x = self.sph_enc(normals, roughness_x)  # von Mises-Fisher distribution
        ref_y = self.sph_enc(normals, roughness_y)  # von Mises-Fisher distribution
        ref = torch.cat([ref_x, ref_y], -1)  # pn,72*2
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1, eps=1e-6)
            sph_points = self.sph_enc(sph_points, roughness)
            light = self.outer_light(torch.cat([ref, sph_points], -1))
        else:
            light = self.outer_light(ref)
        return light

    def forward(self, points, normals, view_dirs, feature_vectors, human_poses, inter_results=False, step=None):
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)

        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        # Todo reset to isotropic roughness
        # roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        roughness_x = self.roughness_predictor_x(torch.cat([feature_vectors, points], -1))
        roughness_y = self.roughness_predictor_y(torch.cat([feature_vectors, points], -1))
        roughness = torch.sqrt(roughness_x * roughness_y + 1e-6)
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))

        # diffuse light
        diffuse_albedo = (1 - metallic) * albedo
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        diffuse_color = diffuse_albedo * diffuse_light

        # specular light
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        specular_light, occ_prob, indirect_light, human_light = self.predict_specular_lights(points, feature_vectors,
                                                                                             reflective,
                                                                                             roughness_x, roughness_y,
                                                                                             human_poses, step)

        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = points.shape[0], 1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
                               boundary_mode='clamp').reshape(pn, 2)
        specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
        specular_color = specular_ref * specular_light

        # integrated together
        color = diffuse_color + specular_color

        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color)
        specular_color = linear_to_srgb(specular_color)
        color = linear_to_srgb(color)
        color = torch.clamp(color, min=0.0, max=1.0)

        # changed here calculating the weights

        occ_info = {
            'reflective': reflective,
            'occ_prob': occ_prob,
        }

        if inter_results:
            intermediate_results = {
                'specular_albedo': specular_albedo,
                'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
                'specular_light': torch.clamp(linear_to_srgb(specular_light), min=0.0, max=1.0),
                'specular_color': torch.clamp(specular_color, min=0.0, max=1.0),

                'diffuse_albedo': diffuse_albedo,
                'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),

                'metallic': metallic,
                'roughness': roughness,

                'occ_prob': torch.clamp(occ_prob, max=1.0, min=0.0),
                'indirect_light': indirect_light,
            }
            if self.cfg['human_light']:
                intermediate_results['human_light'] = linear_to_srgb(human_light)
            return color, occ_info, intermediate_results
        else:
            return color, occ_info, diffuse_color, color

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        # Todo reset to isotropic roughness
        # roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        roughness_x = self.roughness_predictor_x(torch.cat([feature_vectors, points], -1))
        roughness_y = self.roughness_predictor_y(torch.cat([feature_vectors, points], -1))
        roughness = torch.sqrt(roughness_x * roughness_y + 1e-6)
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        return metallic, roughness, albedo


class MaterialFeatsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_enc, input_dim = get_embedder(8, 3)
        run_dim = 256
        self.module0 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
        )
        self.module1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim + run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        input = x
        x = self.module0(x)
        return self.module1(torch.cat([x, input], -1))


def saturate_dot(v0, v1):
    return torch.clamp(torch.sum(v0 * v1, dim=-1, keepdim=True), min=0.0, max=1.0)


class MCShadingNetwork(nn.Module):
    default_cfg = {
        'diffuse_sample_num': 512,
        'specular_sample_num': 256,
        'human_lights': True,
        'light_exp_max': 5.0,
        'inner_light_exp_max': 5.0,
        'outer_light_version': 'direction',
        'geometry_type': 'schlick',

        'reg_change': True,
        'change_eps': 0.05,
        'change_type': 'gaussian',
        'reg_lambda1': 0.05,  # Prev value was 0.05
        'reg_min_max': True,

        'random_azimuth': True,
        'is_real': False,
        'anisotropy': True,
        'max_n_exp': 20,
        'max_alpha_exp': 10,
        'reg_energy_loss': True,
        'reg_energy_loss_lambda': 0.05,
        'reg_spec_loss': True,
        'reg_spec_loss_lambda': 0.05,
    }

    def __init__(self, cfg, ray_trace_fun):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__()

        # material part
        self.feats_network = MaterialFeatsNetwork()
        # self.metallic_predictor = make_predictor(256 + 3, 1)
        # self.roughness_predictor = make_predictor(256 + 3, 1)
        # self.albedo_predictor = make_predictor(256 + 3, 3)

        if self.cfg['anisotropy']:
            # self.mx_predictor = make_predictor(256 + 3, 1, activation='exp', exp_max=self.cfg['max_n_exp'])
            #
            # self.my_predictor = make_predictor(256 + 3, 1, activation='exp', exp_max=self.cfg['max_n_exp'])
            self.mx_predictor = make_predictor(256 + 3, 1)
            self.my_predictor = make_predictor(256 + 3, 1)
            self.alpha_predictor = make_predictor(256 + 3, 1)
            self.metallic_predictor = make_predictor(256 + 3, 1)
            self.rotation_predictor = make_predictor(256 + 3, 2)
            self.source_predictor = make_predictor(256 + 3, 3)

            self.kd_predictor = make_predictor(256 + 3, 3)

        # light part
        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        # Todo changed here
        self.pos_enc, pos_dim = get_embedder(6, 3)
        if self.cfg['outer_light_version'] == 'direction':
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
            # self.outer_light = make_predictor(dir_dim, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
        else:
            raise NotImplementedError
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))
        if self.cfg['human_lights']:
            self.human_light = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light[-2].bias, np.log(0.02))
        # self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=self.cfg['inner_light_exp_max'])

        self.inner_light = make_predictor(pos_dim + dir_dim, 3, activation='exp',
                                          exp_max=self.cfg['inner_light_exp_max'])
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))

        # predefined diffuse sample directions
        az, el = sample_sphere(self.cfg['diffuse_sample_num'], 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
        self.diffuse_direction_samples = np.stack([az, el], -1)
        self.diffuse_direction_samples = torch.from_numpy(
            self.diffuse_direction_samples.astype(np.float32)).cuda()  # [dn0,2]

        az, el = sample_sphere(self.cfg['specular_sample_num'], 0)
        az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
        self.specular_direction_samples = np.stack([az, el], -1)
        self.specular_direction_samples = torch.from_numpy(
            self.specular_direction_samples.astype(np.float32)).cuda()  # [dn1,2]

        az, el = sample_sphere(8192, 0)
        light_pts = az_el_to_points(az, el)
        self.register_buffer('light_pts', torch.from_numpy(light_pts.astype(np.float32)))
        self.ray_trace_fun = ray_trace_fun

    def get_orthogonal_directions(self, directions):
        x, y, z = torch.split(directions, 1, dim=-1)  # pn,1
        otho0 = torch.cat([y, -x, torch.zeros_like(x)], -1)
        otho1 = torch.cat([-z, torch.zeros_like(x), x], -1)
        mask0 = torch.norm(otho0, dim=-1) > torch.norm(otho1, dim=-1)
        mask1 = ~mask0
        otho = torch.zeros_like(directions)
        otho[mask0] = otho0[mask0]
        otho[mask1] = otho1[mask1]
        otho = F.normalize(otho, dim=-1)
        return otho

    # def compute_tangent_bitangent_flat(self, normals, switch_width=0.15):
    #     """
    #     Smooth reference-vector method: blend between X- and Y-axis bases
    #     to avoid discontinuities when the normal aligns with a reference axis.

    #     Args:
    #         normals (torch.Tensor): (N, 3) unit (or near-unit) normals
    #         switch_width (float): width of the smooth transition near |nx|≈1

    #     Returns:
    #         tangent (N,3), bitangent (N,3)
    #     """
    #     import torch
    #     import torch.nn.functional as F

    #     n = F.normalize(normals, dim=-1)
    #     device, dtype = n.device, n.dtype

    #     v0 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)  # X
    #     v1 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)  # Y

    #     # How parallel is n to X? (|cos θ_x|)
    #     a = n[..., 0].abs()

    #     # Smoothstep from (1 - switch_width) → 1
    #     # t in [0,1] only near |nx| ~ 1; elsewhere ~0
    #     edge0 = 1.0 - switch_width
    #     edge1 = 1.0
    #     t = torch.clamp((a - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    #     w = t * t * (3.0 - 2.0 * t)  # smoothstep

    #     # Blend the base direction: near X-alignment, slide toward Y
    #     base = (1.0 - w).unsqueeze(-1) * v0 + w.unsqueeze(-1) * v1

    #     # Tangent ⟂ n via cross with blended base
    #     tangent = torch.cross(n, base.expand_as(n), dim=-1)

    #     # Rare numerical degeneracy guard (e.g., if n not normalized initially)
    #     zero_mask = tangent.norm(dim=-1) < 1e-8
    #     if zero_mask.any():
    #         v2 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    #         tangent[zero_mask] = torch.cross(n[zero_mask], v2.unsqueeze(0), dim=-1)

    #     tangent = F.normalize(tangent, dim=-1)

    #     # Bitangent from right-handed frame
    #     bitangent = F.normalize(torch.cross(n, tangent, dim=-1), dim=-1)

    #     return tangent, bitangent

    def compute_tangent_bitangent_flat(self, normals, sources):
        """
        Compute tangents and bitangents for flat surfaces with no UVs using a reference direction.

        Args:
            normals (torch.Tensor): (N, 3) tensor of vertex normals

        Returns:
            torch.Tensor: (N, 3) tensor of tangents
            torch.Tensor: (N, 3) tensor of bitangents
        """
        if sources is None:
            # Set a global reference direction (here, along the X-axis)
            ref_dir = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # Example: X-axis

            # Compute tangent by crossing the normal with the reference direction
            tangent = torch.cross(normals, ref_dir.unsqueeze(0).expand(normals.size(0), -1), dim=-1)

            # Handle cases where tangent is zero due to alignment with the reference direction
            zero_tangent_mask = tangent.norm(dim=1) < 1e-6
            if zero_tangent_mask.any():
                # Recalculate tangent using the Y-axis if the normal is aligned with X-axis
                ref_dir = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)  # Example: Y-axis
                tangent[zero_tangent_mask] = torch.cross(normals[zero_tangent_mask], ref_dir.unsqueeze(0), dim=-1)

            # Normalize the tangents
            tangent = torch.nn.functional.normalize(tangent, p=2, dim=-1, eps=1e-6)

            # Compute bitangent via cross product with normal
            bitangent = torch.cross(normals, tangent, dim=-1)

            # Normalize bitangents
            bitangent = torch.nn.functional.normalize(bitangent, p=2, dim=-1, eps=1e-6)

            return tangent, bitangent
        else:
            # Set a global reference direction (here, along the X-axis)
            ref_dirs = sources

            # Compute tangent by crossing the normal with the reference direction
            tangent = torch.cross(normals, ref_dirs, dim=-1)

            # Normalize the tangents
            tangent = torch.nn.functional.normalize(tangent, p=2, dim=-1, eps=1e-6)

            # Compute bitangent via cross product with normal
            bitangent = torch.cross(normals, tangent, dim=-1)

            # Normalize bitangents
            bitangent = torch.nn.functional.normalize(bitangent, p=2, dim=-1, eps=1e-6)

            return tangent, bitangent

    def sample_diffuse_directions(self, normals, is_train):
        # normals [pn,3]
        z = normals  # pn,3
        x = self.get_orthogonal_directions(normals)  # pn,3
        y = torch.cross(z, x, dim=-1)  # pn,3
        # y = torch.cross(z, x, dim=-1) # pn,3

        # project onto this tangent space
        az, el = torch.split(self.diffuse_direction_samples, 1, dim=1)  # sn,1
        el, az = el.unsqueeze(0), az.unsqueeze(0)
        az = az * np.pi * 2
        el_sqrt = torch.sqrt(el + 1e-7)
        if is_train and self.cfg['random_azimuth']:
            az = (az + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
        coeff_z = torch.sqrt(1 - el + 1e-7)
        coeff_x = el_sqrt * torch.cos(az)
        coeff_y = el_sqrt * torch.sin(az)

        directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)  # pn,sn,3
        return directions

    def sample_specular_directions(self, reflections, roughness, is_train):
        # roughness [pn,1]
        z = reflections  # pn,3
        x = self.get_orthogonal_directions(reflections)  # pn,3
        y = torch.cross(z, x, dim=-1)  # pn,3
        a = roughness  # we assume the predicted roughness is already squared

        az, el = torch.split(self.specular_direction_samples, 1, dim=1)  # sn,1
        phi = np.pi * 2 * az  # sn,1
        a, el = a.unsqueeze(1), el.unsqueeze(0)  # [pn,1,1] [1,sn,1]
        cos_theta = torch.sqrt((1.0 - el + 1e-6) / (1.0 + (a ** 2 - 1.0) * el + 1e-6) + 1e-6)  # pn,sn,1
        sin_theta = torch.sqrt(1 - cos_theta ** 2 + 1e-6)  # pn,sn,1

        phi = phi.unsqueeze(0)  # 1,sn,1
        if is_train and self.cfg['random_azimuth']:
            phi = (phi + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
        coeff_x = torch.cos(phi) * sin_theta  # pn,sn,1
        coeff_y = torch.sin(phi) * sin_theta  # pn,sn,1
        coeff_z = cos_theta  # pn,sn,1

        directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)  # pn,sn,3
        return directions

    def get_inner_lights(self, points, view_dirs, normals):
        pos_enc = self.pos_enc(points)
        normals = F.normalize(normals, dim=-1, eps=1e-6)
        view_dirs = F.normalize(view_dirs, dim=-1, eps=1e-6)
        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs

        # Todo changed here dir_enc = self.sph_enc(reflections, 0)
        dir_enc = self.dir_enc(reflections)
        return self.inner_light(torch.cat([pos_enc, dir_enc], -1))

    def get_human_light(self, points, directions, human_poses):
        inter, dists, hits = get_camera_plane_intersection(points, directions, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits

        var = torch.zeros_like(mean)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_outer_lights(self, points, directions):
        # Todo changed sph_enc to dir_enc
        if self.cfg['outer_light_version'] == 'direction':
            outer_enc = self.sph_enc(directions, 0)
            # outer_enc = self.dir_enc(directions)
            outer_lights = self.outer_light(outer_enc)
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            outer_dirs = directions
            outer_pts = points
            outer_enc = self.sph_enc(outer_dirs, 0)
            # outer_enc = self.dir_enc(outer_dirs)
            mask = torch.norm(outer_pts, dim=-1) > 0.999
            if torch.sum(mask) > 0:
                outer_pts = torch.clone(outer_pts)
                outer_pts[mask] *= 0.999  # shrink this point a little bit
            dists = get_sphere_intersection(outer_pts, outer_dirs)
            sphere_pts = outer_pts + outer_dirs * dists
            sphere_pts = self.sph_enc(sphere_pts, 0)
            # sphere_pts = self.dir_enc(sphere_pts)
            outer_lights = self.outer_light(torch.cat([outer_enc, sphere_pts], -1))
        else:
            raise NotImplementedError
        return outer_lights

    def get_lights(self, points, directions, human_poses):
        # trace
        shape = points.shape[:-1]  # pn,sn
        eps = 1e-5
        inters, normals, depth, hit_mask = self.ray_trace_fun(points.reshape(-1, 3) + directions.reshape(-1, 3) * eps,
                                                              directions.reshape(-1, 3))
        inters, normals, depth, hit_mask = inters.reshape(*shape, 3), normals.reshape(*shape, 3), depth.reshape(*shape,
                                                                                                                1), hit_mask.reshape(
            *shape)
        miss_mask = ~hit_mask

        # hit_mask
        lights = torch.zeros(*shape, 3)
        human_lights, human_weights = torch.zeros([1, 3]), torch.zeros([1, 1])
        if torch.sum(miss_mask) > 0:
            outer_lights = self.predict_outer_lights(points[miss_mask], directions[miss_mask])
            if self.cfg['human_lights']:
                human_lights, human_weights = self.get_human_light(points[miss_mask], directions[miss_mask],
                                                                   human_poses[miss_mask])
            else:
                human_lights, human_weights = torch.zeros_like(outer_lights), torch.zeros(outer_lights.shape[0], 1)
            lights[miss_mask] = outer_lights * (1 - human_weights) + human_lights * human_weights

        if torch.sum(hit_mask) > 0:
            lights[hit_mask] = self.get_inner_lights(inters[hit_mask], -directions[hit_mask], normals[hit_mask])

        near_mask = (depth > eps).float()
        lights = lights * near_mask  # very near surface does not bring lights
        return lights, human_lights * human_weights, inters, normals, hit_mask

    def fresnel_schlick(self, F0, HoV):
        return F0 + (1.0 - F0) * torch.clamp(1.0 - HoV, min=0.0, max=1.0) ** 5.0

    def fresnel_schlick_directions(self, F0, view_dirs, directions):
        H = (view_dirs + directions)  # [pn,sn0,3]
        H = F.normalize(H, dim=-1, eps=1e-6)
        HoV = torch.clamp(torch.sum(H * view_dirs, dim=-1, keepdim=True), min=0.0, max=1.0)  # [pn,sn0,1]
        fresnel = self.fresnel_schlick(F0, HoV)  # [pn,sn0,1]
        return fresnel, H, HoV

    def geometry_schlick_ggx(self, NoV, roughness):
        a = roughness  # a = roughness**2: we assume the predicted roughness is already squared

        k = a / 2
        num = NoV
        denom = NoV * (1 - k) + k
        return num / (denom + 1e-5)

    def geometry_schlick(self, NoV, NoL, roughness):
        ggx2 = self.geometry_schlick_ggx(NoV, roughness)
        ggx1 = self.geometry_schlick_ggx(NoL, roughness)
        return ggx2 * ggx1

    def geometry_ggx_smith_correlated(self, NoV, NoL, roughness):
        def fun(alpha2, cos_theta):
            # cos_theta = torch.clamp(cos_theta,min=1e-7,max=1-1e-7)
            cos_theta2 = cos_theta ** 2
            tan_theta2 = (1 - cos_theta2) / (cos_theta2 + 1e-7)
            return 0.5 * torch.sqrt(1 + alpha2 * tan_theta2) - 0.5

        alpha_sq = roughness ** 2
        return 1.0 / (1.0 + fun(alpha_sq, NoV) + fun(alpha_sq, NoL))

    def predict_materials(self, pts):
        feats = self.feats_network(pts)
        metallic = self.metallic_predictor(torch.cat([feats, pts], -1))
        roughness = self.roughness_predictor(torch.cat([feats, pts], -1))
        rmax, rmin = 1.0, 0.04 ** 2
        roughness = roughness * (rmax - rmin) + rmin
        albedo = self.albedo_predictor(torch.cat([feats, pts], -1))
        return metallic, roughness, albedo

    def distribution_ggx(self, NoH, roughness):
        a = roughness
        a2 = a ** 2
        NoH2 = NoH ** 2
        denom = NoH2 * (a2 - 1.0) + 1.0
        return a2 / (np.pi * denom ** 2 + 1e-4)

    def geometry(self, NoV, NoL, roughness):
        if self.cfg['geometry_type'] == 'schlick':
            geometry = self.geometry_schlick(NoV, NoL, roughness)
        elif self.cfg['geometry_type'] == 'ggx_smith':
            geometry = self.geometry_ggx_smith_correlated(NoV, NoL, roughness)
        else:
            raise NotImplementedError
        return geometry

    def shade_mixed(self, pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses, is_train):
        F0 = 0.04 * (1 - metallic) + metallic * albedo  # [pn,1]

        # sample diffuse directions
        diffuse_directions = self.sample_diffuse_directions(normals, is_train)  # [pn,sn0,3]
        point_num, diffuse_num, _ = diffuse_directions.shape
        # sample specular directions
        specular_directions = self.sample_specular_directions(reflections, roughness, is_train)  # [pn,sn1,3]
        specular_num = specular_directions.shape[1]

        # diffuse sample prob
        NoL_d = saturate_dot(diffuse_directions, normals.unsqueeze(1))
        diffuse_probability = NoL_d / np.pi * (diffuse_num / (specular_num + diffuse_num))

        # specualr sample prob
        H_s = (view_dirs.unsqueeze(1) + specular_directions)  # [pn,sn0,3] half vector
        H_s = F.normalize(H_s, dim=-1, eps=1e-6)
        NoH_s = saturate_dot(normals.unsqueeze(1), H_s)
        VoH_s = saturate_dot(view_dirs.unsqueeze(1), H_s)
        specular_probability = self.distribution_ggx(NoH_s, roughness.unsqueeze(1)) * NoH_s / (4 * VoH_s + 1e-5) * (
                specular_num / (specular_num + diffuse_num))  # D * NoH / (4 * VoH)

        # combine
        directions = torch.cat([diffuse_directions, specular_directions], 1)
        probability = torch.cat([diffuse_probability, specular_probability], 1)
        sn = diffuse_num + specular_num

        # specular
        fresnel, H, HoV = self.fresnel_schlick_directions(F0.unsqueeze(1), view_dirs.unsqueeze(1), directions)
        NoV = saturate_dot(normals, view_dirs).unsqueeze(1)  # pn,1,3
        NoL = saturate_dot(normals.unsqueeze(1), directions)  # pn,sn,3
        geometry = self.geometry(NoV, NoL, roughness.unsqueeze(1))
        NoH = saturate_dot(normals.unsqueeze(1), H)

        human_poses = human_poses.unsqueeze(1).repeat(1, sn, 1, 1) if human_poses is not None else None
        pts_ = pts.unsqueeze(1).repeat(1, sn, 1)
        lights, hl, light_pts, light_normals, light_pts_mask = self.get_lights(pts_, directions, human_poses)  # pn,sn,3
        specular_weights = distribution * geometry / (4 * NoV * probability + 1e-5)
        specular_lights = lights * specular_weights
        specular_colors = torch.mean(fresnel * specular_lights, 1)
        specular_weights = specular_weights * fresnel

        # diffuse only consider diffuse directions
        kd = (1 - metallic.unsqueeze(1))
        diffuse_lights = lights[:, :diffuse_num]
        diffuse_colors = albedo.unsqueeze(1) * kd[:, :diffuse_num] * diffuse_lights
        diffuse_colors = torch.mean(diffuse_colors, 1)

        colors = diffuse_colors + specular_colors
        colors = linear_to_srgb(colors)

        outputs = {}
        outputs['albedo'] = albedo
        outputs['roughness'] = roughness
        outputs['metallic'] = metallic
        outputs['human_lights'] = hl.reshape(-1, 3)
        outputs['diffuse_light'] = torch.clamp(linear_to_srgb(torch.mean(diffuse_lights, dim=1)), min=0, max=1)
        outputs['specular_light'] = torch.clamp(linear_to_srgb(torch.mean(specular_lights, dim=1)), min=0, max=1)
        diffuse_colors = torch.clamp(linear_to_srgb(diffuse_colors), min=0, max=1)
        specular_colors = torch.clamp(linear_to_srgb(specular_colors), min=0, max=1)
        outputs['diffuse_color'] = diffuse_colors
        outputs['specular_color'] = specular_colors
        outputs['approximate_light'] = torch.clamp(
            linear_to_srgb(torch.mean(kd[:, :diffuse_num] * diffuse_lights, dim=1) + specular_colors), min=0, max=1)
        return colors, outputs

    def predict_anisotropic_components(self, pts):
        feats = self.feats_network(pts)
        mx_min, mx_max = 0.005, 1.0
        my_min, my_max = 0.005, 1.0
        mx = self.mx_predictor(torch.cat([feats, pts], -1))
        mx = mx_min + (mx_max - mx_min) * mx
        my = self.my_predictor(torch.cat([feats, pts], -1))
        my = my_min + (my_max - my_min) * my
        alpha = self.alpha_predictor(torch.cat([feats, pts], -1))
        # alpha = torch.ones_like(mx)
        metallic = self.metallic_predictor(torch.cat([feats, pts], -1))
        kd = self.kd_predictor(torch.cat([feats, pts], -1))
        rotation_raw = self.rotation_predictor(torch.cat([feats, pts], -1))
        # print(f"rotatotion[:,0] min: {rotation_raw[:,0].min()}, max: {rotation_raw[:,0].max()}")
        # print(f"rotatotion[:,1] min: {rotation_raw[:,1].min()}, max: {rotation_raw[:,1].max()}")
        # print(f"rotation norm min: {torch.norm(rotation_raw, dim=-1).min()}, max: {torch.norm(rotation_raw, dim=-1).max()}")
        rotation = rotation_raw * 2.0 - 1.0
        rotation = F.normalize(rotation, dim=-1, eps=1e-6)
        sources = self.source_predictor(torch.cat([feats, pts], -1))
        sources = sources * 2.0 - 1.0
        return mx, my, alpha, metallic, kd, rotation, rotation_raw, sources

    def compute_Dh(self, m_x, m_y, theta_h, phi_h):
        """
        Compute D(h) based on the provided equation.

        Args:
            m_x (torch.Tensor): (N, 1) tensor of roughness in the x direction
            m_y (torch.Tensor): (N, 1) tensor of roughness in the y direction
            theta_h (torch.Tensor): (N, M) tensor of the angles theta_h
            phi_h (torch.Tensor): (N, M) tensor of the angles phi_h

        Returns:
            torch.Tensor: (N, M, 1) tensor of D(h)
        """
        # Step 1: Compute q(h)
        cos_phi_h = torch.cos(phi_h)  # (N, M)
        sin_phi_h = torch.sin(phi_h)  # (N, M)

        cos_theta_h = torch.cos(theta_h)  # (N, M)
        sin_theta_h = torch.sin(theta_h)  # (N, M)
        eps = 1e-6
        cos_th2 = (cos_theta_h * cos_theta_h).clamp_min(eps)  # avoid blow-up at grazing
        tan2 = (sin_theta_h * sin_theta_h) / cos_th2
        # Calculate q(h)
        log_q = -tan2 * (
                (cos_phi_h ** 2) / (m_x ** 2) + (sin_phi_h ** 2) / (m_y ** 2)
        )
        log_q = torch.clamp(log_q, min=-40.0, max=40.0)  # prevent overflow
        q_h = torch.exp(log_q)

        # Step 2: Compute D(h) based on the formula
        denom = (np.pi * m_x * m_y * cos_theta_h ** 4).clamp(min=eps)
        D_h = (1 / denom) * q_h  # (N, M, 1)

        return D_h.unsqueeze(-1)

    def compute_spec_loss(self, tem, m_x, m_y, theta_h, phi_h):
        D = self.compute_Dh(m_x, m_y, theta_h, phi_h).detach()
        soft_d = torch.nn.functional.softmax(D / tem, dim=1)
        return soft_d

    def compute_pdf_aniso_ggx(self,
                              m_x: torch.Tensor,  # (N,1)
                              m_y: torch.Tensor,  # (N,1)
                              w_o: torch.Tensor,  # (N, 3)
                              h: torch.Tensor,  # (N, M, 3), half‐vectors in tangent‐space
                              theta_h,
                              phi_h,
                              eps: float = 1e-8
                              ) -> torch.Tensor:
        """
        Returns:
          p: (N, M, 1) the sampling PDF p(ω_i | ω_o) per Eqn.(20)&(4).
        """
        N, M, _ = h.shape
        cos_phi = torch.cos(phi_h).unsqueeze(-1)
        sin_phi = torch.sin(phi_h).unsqueeze(-1)
        tan_th = torch.tan(theta_h).unsqueeze(-1)
        cos_th = torch.cos(theta_h).unsqueeze(-1)
        cos_th2 = (cos_th * cos_th).clamp_min(eps)
        sin_th = torch.sin(theta_h).unsqueeze(-1)
        tan2 = (sin_th * sin_th) / cos_th2
        # reshape roughness for broadcast
        m_x = m_x.view(N, 1, 1)  # (N,1,1)
        m_y = m_y.view(N, 1, 1)

        # exponent: tan^2θ_h * (cos^2φ_h/m_x^2 + sin^2φ_h/m_y^2)
        #  -> (h_x^2 + h_y^2)/h_z^2 * ( h_x^2/(h_x^2+h_y^2)/m_x^2 + h_y^2/(h_x^2+h_y^2)/m_y^2 )
        # simplifies to:
        # print('tan_th, cos_phi, sin_phi, cos_th:', tan_th.shape, cos_phi.shape, sin_phi.shape, cos_th.shape)
        exp_term = (-tan2) * ((sin_phi * sin_phi) / (m_y * m_y) + (cos_phi * cos_phi) / (m_x * m_x))
        # print((cos_phi / m_x).shape)
        exp_term = torch.clamp(exp_term, min=-50.0, max=50.0)  # prevent overflow
        q = torch.exp(exp_term)

        # denominator: 4π m_x m_y cos^3θ_h (ω_o · h)
        #   compute dot(ω_o, h) → (N,M,1)
        wo = w_o.unsqueeze(1)  # (N,1,3)
        dot_wo_h = (wo * h).sum(dim=-1, keepdim=True)  # (N,M,1)
        # print('cos_th, mx, my, dot_wo, q:', cos_th.shape, m_x.shape, m_y.shape, dot_wo_h.shape, q.shape)
        denom = (4.0 * np.pi) * m_x * m_y * (cos_th ** 3) * dot_wo_h
        p = q / (denom.clamp(min=0) + eps)

        return p  # (N, M, 1)

    def rotate_tangent_bitangent(self, tangent, bitangent, cos_theta, sin_theta):
        T_rot = cos_theta * tangent + sin_theta * bitangent
        B_rot = -sin_theta * tangent + cos_theta * bitangent
        return T_rot, B_rot

    def sample_aniso_ggx_directions(self,
                                    rotation: torch.Tensor,
                                    pts: torch.Tensor,
                                    m_x: torch.Tensor,
                                    m_y: torch.Tensor,
                                    wo: torch.Tensor,
                                    M: int,
                                    normals: torch.Tensor,
                                    sources: torch.Tensor,
                                    device: torch.device = None,
                                    eps: float = 1e-6):
        if device is None:
            device = wo.device

        z = normals  # pn,3

        x, y = self.compute_tangent_bitangent_flat(normals, sources)
        # rotate tangent and bitangent
        cos_2rot = rotation[:, 0:1]  # (N,1)
        sin_2rot = rotation[:, 1:2]  # (N,1)

        cos_rot = torch.sqrt(eps + (cos_2rot + 1.0) / 2.0).clamp(min=0.0, max=1.0)  # (N,1)
        self.nan_inf_check(cos_rot, 'cos_rot')
        sin_rot = torch.sign(sin_2rot) * torch.sqrt(eps + (1.0 - cos_2rot) / 2.0).clamp(min=0.0, max=1.0)  # (N,1)
        self.nan_inf_check(sin_rot, 'sin_rot')
        if sources is None:
            x, y = self.rotate_tangent_bitangent(x, y, cos_rot, sin_rot)

        m_x = m_x.to(device)  # (N,1)
        m_y = m_y.to(device)  # (N,1)
        wo = wo.to(device)  # (N,3)
        N = wo.shape[0]

        # 1) Uniformsk
        xi1 = torch.rand((N, M), device=device).clamp(min=eps)
        xi2 = torch.rand((N, M), device=device)

        two_pi_xi2 = 2.0 * np.pi * xi2  # (N,M)
        s, c = torch.sin(two_pi_xi2), torch.cos(two_pi_xi2)
        # 2) Azimuth via atan2
        phi_h = torch.atan2(m_y * s, m_x * c)  # <-- full-range

        # 3) Elevation
        cos_phi = torch.cos(phi_h)
        sin_phi = torch.sin(phi_h)
        denom = (cos_phi * cos_phi) / (m_x * m_x) + (sin_phi * sin_phi) / (m_y * m_y)
        theta_h = torch.atan2(torch.sqrt(-torch.log(xi1)), torch.sqrt(denom + eps))

        # 4) half-vector
        sin_th = torch.sin(theta_h)
        cos_th = torch.cos(theta_h)
        coeff_x = sin_th * cos_phi
        coeff_y = sin_th * sin_phi
        coeff_z = cos_th
        # print('coeff_x shape', coeff_x.shape, x.shape)
        h = (coeff_x.unsqueeze(2) * x.unsqueeze(1).expand(coeff_x.shape[0], coeff_x.shape[1], 3) +
             coeff_y.unsqueeze(2) * y.unsqueeze(1).expand(coeff_x.shape[0], coeff_x.shape[1], 3) +
             coeff_z.unsqueeze(2) * z.unsqueeze(1).expand(coeff_x.shape[0], coeff_x.shape[1], 3))

        dot = (wo.unsqueeze(1) * h).sum(-1, keepdim=True)
        wi = 2 * dot * h - wo.unsqueeze(1)
        wi = torch.nn.functional.normalize(wi, dim=-1, eps=eps)

        cos_theta_h = cos_th.unsqueeze(-1)  # (N,M,1)
        pdf = self.compute_pdf_aniso_ggx(m_x, m_y, wo, h, theta_h, phi_h)

        return h.float(), wi.float(), cos_theta_h.float(), pdf.float(), x, y, z, theta_h, phi_h

    def compute_radiance(self,
                         f_d: torch.Tensor,
                         diffuse_lights: torch.Tensor,
                         specular_lights: torch.Tensor,
                         F: torch.Tensor, #(N, M, 3)
                         diffuse_directions: torch.Tensor,
                         wi: torch.Tensor,
                         n: torch.Tensor,
                         alpha: torch.Tensor,
                         cos_theta_h: torch.Tensor,
                         wo: torch.Tensor,
                         pdf: torch.Tensor,
                         theta_h: torch.Tensor,
                         phi_h: torch.Tensor,
                         mx: torch.Tensor,
                         my: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
        """
        Compute outgoing radiance R for N points, M samples each, via:

          R = (1/M) sum_i [ f_d * L_i ]
            + (1/M) sum_i [ L_i * k_s * F * (wi·n)^(1-alpha) / (cosθ_h * (wo·n)^alpha) ]

        Parameters:
        -----------
        f_d : Tensor (N, M, 3)
            Diffuse BRDF term per sample.
        lights : Tensor (N, M, 3)
            Incoming radiance Li per sample.
        F : Tensor (N, M, 3)
            Fresnel term per sample.
        wi : Tensor (N, M, 3)
            Incident directions per sample.
        n : Tensor (N, 3)
            Surface normals per point.
        alpha : Tensor (N, 1)
            Roughness exponent per point.
        cos_theta_h : Tensor (N, M, 1)
            cos(theta_h) per sample.
        wo : Tensor (N, 3)
            View/outgoing direction per point.
        eps : float
            Small epsilon for stability.

        Returns:
        --------
        R : Tensor (N, 3)
            Estimated outgoing radiance per point.
        """
        f_d = f_d.unsqueeze(1)  # (N,1,3)
        N_spec, M_spec, _ = specular_lights.shape
        N_diff, M_diff, _ = diffuse_lights.shape

        #    shape (N, M)
        mask = ((wi * n.unsqueeze(1)).sum(dim=-1) > 0.0)
        valid_counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

        # Expand k_s, alpha, wo·n to match (N, M, *)
        alpha_exp = alpha.unsqueeze(1)  # (N, 1, 1)
        cos_on = torch.clamp((wo * n).sum(dim=1, keepdim=True), min=0.0)  # (N,1)
        cos_on_exp = cos_on.unsqueeze(1)  # (N,1,1)

        # Dot product wi·n
        cos_in = torch.clamp((wi * n.unsqueeze(1)).sum(dim=2, keepdim=True), min=0.0)  # (N,M,1)

        # --- Diffuse component: (1/M) ∑ f_d * Li ---
        diff_weighted = diffuse_lights * f_d  # (N,M,3)
        diffuse = diff_weighted.sum(dim=1) / M_diff  # (N,3)

        # --- Specular component ---
        # term1 = L * k_s * F

        # term2 = (wi·n)^(1-alpha)
        exponent = 1.0 - alpha_exp  # (N,1,1)

        pow_in = torch.pow(cos_in + eps, exponent)  # (N,M,1)
        pow_in_denom = torch.pow(cos_in + eps, -alpha_exp)

        pow_on = torch.pow(cos_on_exp + eps, alpha_exp) # (N,1,1)
        denom = (cos_theta_h * pow_on).clamp(min=1e-6)  # (N,M,1)
        f_s = (F * pow_in / denom) * mask.unsqueeze(-1) # (N,M,3)
        # spec_brdf = (F * pow_in_denom * pdf / denom) * mask.unsqueeze(-1)
        weighted_specular_light = (pow_in_denom * pdf / denom) * mask.unsqueeze(-1) * specular_lights

        spec_weighted = f_s * specular_lights  # (N,M,3)
        specular = spec_weighted.sum(dim=1) / valid_counts  # (N,3)
        f_s_sum = f_s.sum(dim=1) / valid_counts # (N,3)
        # Total radiance
        R = diffuse + specular  # (N,3)

        tem = 1
        L_spec = self.compute_spec_loss(tem, mx, my, theta_h, phi_h)
        L_spec = torch.sum(L_spec * f_d * mask.unsqueeze(-1), dim=1) / (valid_counts * 3)
        return R, diffuse, specular, f_s_sum, L_spec, weighted_specular_light

    def nan_inf_check(self, A, name):
        if torch.isinf(A).any():
            print('inf in', name)
        if torch.isnan(A).any():
            print('nan in', name)

    def fresnel_schlick_batch(self,
                              f0: torch.Tensor,
                              wo: torch.Tensor,
                              h: torch.Tensor,
                              eps: float = 1e-6) -> torch.Tensor:
        """
        Schlick's Fresnel for batched queries:
            F = f0 + (1 - f0) * (1 - (wo · h))^5

        Parameters:
        -----------
        f0 : torch.Tensor
            Base reflectivity, shape (N, 1)  (per-point scalar or channel count 1).
        wo : torch.Tensor
            Outgoing/view directions, shape (N, 3).
        h : torch.Tensor
            Half-vectors, shape (N, M, 3), M samples per point.
        eps : float
            Small epsilon to avoid numerical issues.

        Returns:
        --------
        F : torch.Tensor
            Fresnel terms, shape (N, M, 1).
        """
        # Ensure shapes
        N, M, _ = h.shape
        # Broadcast wo to (N, M, 3)
        wo_exp = wo.unsqueeze(1).expand(N, M, 3)
        # Compute cos(theta) = wo · h, shape (N, M, 1)
        cos_theta = torch.clamp(torch.sum(wo_exp * h, dim=-1, keepdim=True), min=0.0, max=1.0)
        # Broadcast f0 to (N, M, 1)
        f0_exp = f0.unsqueeze(1)  # (N, 1, 1) -> broadcast over M
        # Compute Fresnel
        F = f0_exp + (1.0 - f0_exp) * (1.0 - cos_theta) ** 5
        return F

    def diffuse_term(self,
                     kd: torch.Tensor,
                     metallic: torch.Tensor,
                     is_seperate: bool = True) -> torch.Tensor:
        """
        Compute the diffuse term f_d = (k_d / π) * (1 - F) for anisotropic Cook-Torrance.

        Parameters:
        -----------
        kd : torch.Tensor
            Diffuse albedo, shape (N, 3).
        F : torch.Tensor
            Fresnel term from fresnel_schlick_batch,
            shape (N, M, 1).

        Returns:
        --------
        f_d : torch.Tensor
            Diffuse BRDF term, shape (N, M, 3).
        """
        f_d = kd * (1-metallic)  # (N,3)
        return f_d

    def shade_anisotropic_mixed(self, pts, normals, sources, view_dirs, mx, my, alpha, metallic, kd, rotation, rotation_raw,
                                human_poses, is_train):
        sources_norm = torch.nn.functional.normalize(sources, dim=-1, eps=1e-6)
        num_spec_samples = self.cfg['specular_sample_num']
        # Todo sources is not passed here
        hs, wis, cos_ths, pdfs, t, b, n, theta_h, phi_h = self.sample_aniso_ggx_directions(rotation, pts, mx, my,
                                                                                           view_dirs, num_spec_samples,
                                                                                           normals, None, 'cuda')
        f0 = 0.04 * (1 - metallic) + metallic * kd  # [pn,1]
        diffuse_directions = self.sample_diffuse_directions(normals, is_train)

        point_num, diffuse_num, _ = diffuse_directions.shape

        pts_ = pts.unsqueeze(1).repeat(1, num_spec_samples + diffuse_num, 1)
        directions = torch.cat([diffuse_directions, wis], 1)
        sn = diffuse_num + num_spec_samples
        human_poses = human_poses.unsqueeze(1).repeat(1, sn, 1, 1) if human_poses is not None else None
        lights, hl, light_pts, light_normals, light_pts_mask = self.get_lights(pts_, directions, human_poses)

        diffuse_lights = lights[:, :diffuse_num]
        specular_lights = lights[:, diffuse_num:]
        F, H, HoV = self.fresnel_schlick_directions(f0.unsqueeze(1), view_dirs.unsqueeze(1), wis)

        f_d = self.diffuse_term(kd, metallic, is_seperate=False)

        R, diffuse_color, specular_color, f_s_sum, L_spec, weighted_specular_lights = self.compute_radiance(
            f_d, diffuse_lights, specular_lights, F, diffuse_directions, wis, normals, alpha, cos_ths, view_dirs,
            pdfs, theta_h, phi_h, mx, my)

        colors = linear_to_srgb(R)

        diffuse_color = linear_to_srgb(diffuse_color)

        specular_color = linear_to_srgb(specular_color)

        outputs = {}
        outputs['tangents'] = (t + 1) / 2
        outputs['bitangents'] = (b + 1) / 2
        outputs['normals'] = (n + 1) / 2
        outputs['human_lights'] = hl.reshape(-1, 3)
        outputs['kd'] = kd
        outputs['metallic'] = metallic
        outputs['alpha'] = alpha
        outputs['mx'] = mx
        outputs['my'] = my
        outputs['diffuse_color'] = diffuse_color
        outputs['specular_color'] = specular_color
        outputs['diffuse_light'] = torch.clamp(linear_to_srgb(torch.mean(diffuse_lights, dim=1)), min=0, max=1)
        outputs['specular_light'] = torch.clamp(linear_to_srgb(torch.mean(weighted_specular_lights, dim=1)), min=0,
                                                max=1)
        outputs['f_d'] = f_d
        outputs['f_s_sum'] = f_s_sum
        outputs['L_spec'] = L_spec
        outputs['rotation'] = rotation
        outputs['sources'] = (sources + 1) / 2
        outputs['sources_norm'] = (sources_norm + 1) / 2
        outputs['rotation_raw'] = rotation_raw
        return colors, outputs

    def anisotropic_forward(self, pts, view_dirs, normals, human_poses, step, is_train, is_seperate=True):
        # print('anisotropic_forward:')
        mx, my, alpha, metallic, kd, rotation, rotation_raw, sources = self.predict_anisotropic_components(pts)
        return self.shade_anisotropic_mixed(pts, normals, sources, view_dirs, mx, my, alpha, metallic, kd, rotation, rotation_raw,
                                            human_poses, is_train)

    def forward(self, pts, view_dirs, normals, human_poses, step, is_train):
        if self.cfg['anisotropy']:
            return self.anisotropic_forward(pts, view_dirs, normals, human_poses, step, is_train, is_seperate=True)
        view_dirs, normals = F.normalize(view_dirs, dim=-1 , eps=1e-6), F.normalize(normals, dim=-1 , eps=1e-6)
        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        metallic, roughness, albedo = self.predict_materials(pts)  # [pn,1] [pn,1] [pn,3]
        return self.shade_mixed(pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses,
                                is_train)

    def env_light(self, h, w, gamma=True):
        azs = torch.linspace(1.0, 0.0, w) * np.pi * 2 - np.pi / 2
        els = torch.linspace(1.0, -1.0, h) * np.pi / 2

        els, azs = torch.meshgrid(els, azs)
        if self.cfg['is_real']:
            x = torch.cos(els) * torch.cos(azs)
            y = torch.cos(els) * torch.sin(azs)
            z = torch.sin(els)
        else:
            z = torch.cos(els) * torch.cos(azs)
            x = torch.cos(els) * torch.sin(azs)
            y = torch.sin(els)
        xyzs = torch.stack([x, y, z], -1)  # h,w,3
        xyzs = xyzs.reshape(h * w, 3)
        # xyzs = xyzs @ torch.from_numpy(np.asarray([[0,0,1],[0,1,0],[-1,0,0]],np.float32)).cuda()

        batch_size = 8192
        lights = []
        for ri in range(0, h * w, batch_size):
            with torch.no_grad():
                light = self.predict_outer_lights_pts(xyzs[ri:ri + batch_size])
            lights.append(light)
        if gamma:
            lights = linear_to_srgb(torch.cat(lights, 0)).reshape(h, w, 3)
        else:
            lights = (torch.cat(lights, 0)).reshape(h, w, 3)
        return lights

    def predict_outer_lights_pts(self, pts):
        if self.cfg['outer_light_version'] == 'direction':
            # return self.outer_light(self.sph_enc(pts, 0))
            return self.outer_light(self.dir_enc(pts))
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            # return self.outer_light(torch.cat([self.sph_enc(pts, 0), self.sph_enc(pts, 0)], -1))
            return self.outer_light(torch.cat([self.dir_enc(pts), self.dir_enc(pts)], -1))
        else:
            raise NotImplementedError

    def get_env_light(self):
        return self.predict_outer_lights_pts(self.light_pts)

    def length_floor_loss(self, s, tau=0.2, beta=5, eps=1e-8):
        r = s.norm(dim=-1)  # (...,)

        if beta is None:
            # squared hinge
            return torch.relu(tau - r).pow(2).mean()
        else:
            # smooth hinge via softplus
            z = beta * (tau - r)
            return (F.softplus(z).pow(2) / (beta * beta))

    def unit_norm_prior(self, s, kind="huber", delta=0.1, eps=1e-8):
        r = (s.pow(2).sum(dim=-1) + eps).sqrt()  # safe radius
        if kind == "l2":  # (r-1)^2
            e = r - 1.0
            return (e * e).mean()
        elif kind == "huber":  # robust
            e = (r - 1.0).abs()
            return torch.where(e < delta, 0.5 * e * e / delta, e - 0.5 * delta).mean()
        elif kind == "log":  # (log r)^2
            return (r.log().pow(2)).mean()
        else:
            raise ValueError("unknown kind")

    def anisotropic_regularization(self, pts, normals, sources, t, b, mx, my, alpha, metallic, kd, f_d, f_s_sum,
                                   L_spec, rotation, rotation_raw, step):
        reg = 0
        if self.cfg['reg_change']:
            normals = F.normalize(normals, dim=-1, eps=1e-6)
            x = self.get_orthogonal_directions(normals)
            y = torch.cross(normals, x)
            ang = torch.rand(pts.shape[0], 1) * np.pi * 2
            if self.cfg['change_type'] == 'constant':
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * self.cfg['change_eps']
            elif self.cfg['change_type'] == 'gaussian':
                eps = torch.normal(mean=0.0, std=self.cfg['change_eps'], size=[x.shape[0], 1])
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * eps
            else:
                raise NotImplementedError
            # sources_normalized = F.normalize(sources, dim=-1)
            # dot = (sources_normalized * normals).sum(dim=-1, keepdim=True)
            # alignment_loss = (dot ** 2)
            # Penalize alignment (i.e., |dot| close to 1)
            # Using squared absolute dot product ensures smoothness and symmetry
            rotation_raw_mapped = 2.0 * rotation_raw - 1.0
            tau = 0.3
            non_zero_loss = torch.max(torch.zeros_like(mx), tau-torch.norm(rotation_raw_mapped, dim=-1, keepdim=True))**2
            lambda_non_zero = step*0.1 / (100.0 * 1000.0)
            mx_ch, my_ch, alpha_ch, metallic_ch, kd_ch, rotation_ch, rotation_raw_ch, sources_ch = self.predict_anisotropic_components(
                pts + change)

            # sources_ch_normalized = F.normalize(sources_ch, dim=-1)
            curv_dot = (rotation * rotation_ch).sum(dim=-1, keepdim=True)
            curv_loss = (1 - curv_dot) ** 2
            # length_loss = self.unit_norm_prior(sources, kind="huber", delta=0.1)
            # the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]

            mat_reg = torch.mean(
                (
                        torch.abs(kd - kd_ch) +
                        torch.abs(mx - mx_ch) +
                        torch.abs(my - my_ch) +
                        torch.abs(alpha - alpha_ch) +
                        torch.abs(metallic - metallic_ch)
                        # + non_zero_loss * lambda_non_zero
                        + curv_loss * lambda_non_zero
                ),
                dim=1)
            # print(f"length loss: {length_loss.mean().item():.6f}, alignment loss: {alignment_loss.mean().item():.6f}, mat reg loss: {mat_reg.mean().item():.6f}")
            # source_loss = (alignment_loss + length_loss) * 0.001
            reg = reg + (mat_reg) * self.cfg['reg_lambda1']
            if self.cfg['reg_energy_loss']:
                f_r_loss = (f_d + f_s_sum) - 1
                f_r_loss = torch.nn.functional.relu(f_r_loss)
                energy_reg = torch.mean(
                    f_r_loss.sum(dim=1),
                    dim=0
                ) * self.cfg['reg_energy_loss_lambda']
                reg = reg + energy_reg

            if self.cfg['reg_spec_loss']:
                spec_reg = torch.mean(
                    L_spec.sum(dim=1),
                    dim=0
                ) * self.cfg['reg_spec_loss_lambda']
                reg = reg + spec_reg

        return reg

    def material_regularization(self, pts, normals, metallic, roughness, albedo, step):
        # metallic, roughness, albedo = self.predict_materials(pts)
        reg = 0

        if self.cfg['reg_change']:
            normals = F.normalize(normals, dim=-1, eps=1e-6)
            x = self.get_orthogonal_directions(normals)
            y = torch.cross(normals, x)
            ang = torch.rand(pts.shape[0], 1) * np.pi * 2
            if self.cfg['change_type'] == 'constant':
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * self.cfg['change_eps']
            elif self.cfg['change_type'] == 'gaussian':
                eps = torch.normal(mean=0.0, std=self.cfg['change_eps'], size=[x.shape[0], 1])
                change = (torch.cos(ang) * x + torch.sin(ang) * y) * eps
            else:
                raise NotImplementedError
            m0, r0, a0 = self.predict_materials(pts + change)
            reg = reg + torch.mean(
                (torch.abs(m0 - metallic) + torch.abs(r0 - roughness) + torch.abs(a0 - albedo)) * self.cfg[
                    'reg_lambda1'], dim=1)

        if self.cfg['reg_min_max'] and step is not None and step < 2000:
            # sometimes the roughness and metallic saturate with the sigmoid activation in the early stage
            reg = reg + torch.sum(torch.clamp(roughness - 0.98 ** 2, min=0))
            reg = reg + torch.sum(torch.clamp(0.02 ** 2 - roughness, min=0))
            reg = reg + torch.sum(torch.clamp(metallic - 0.98, min=0))
            reg = reg + torch.sum(torch.clamp(0.02 - metallic, min=0))

        return reg


def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).detach()
                    outside_mask = torch.norm(pts, dim=-1) >= 1.0
                    val[outside_mask] = outside_val
                    val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, outside_val=1.0):
    u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles