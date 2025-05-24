# import torch.nn.functional as F
# import torch.nn as nn
# import torch
# import numpy as np
# import nvdiffrast.torch as dr
# import mcubes
#
# from utils.base_utils import az_el_to_points, sample_sphere
# from utils.raw_utils import linear_to_srgb
# from utils.ref_utils import generate_ide_fn
# from utils import math as mathutil
# import tensorflow as tf
#
#
# # Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
# class Embedder:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.create_embedding_fn()
#
#     def create_embedding_fn(self):
#         embed_fns = []
#         d = self.kwargs['input_dims']
#         out_dim = 0
#         if self.kwargs['include_input']:
#             embed_fns.append(lambda x: x)
#             out_dim += d
#
#         max_freq = self.kwargs['max_freq_log2']
#         N_freqs = self.kwargs['num_freqs']
#
#         if self.kwargs['log_sampling']:
#             freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
#         else:
#             freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)
#
#         for freq in freq_bands:
#             for p_fn in self.kwargs['periodic_fns']:
#                 embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
#                 out_dim += d
#
#         self.embed_fns = embed_fns
#         self.out_dim = out_dim
#
#     def embed(self, inputs):
#         return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
#
#
# def get_embedder(multires, input_dims=3):
#     embed_kwargs = {
#         'include_input': True,
#         'input_dims': input_dims,
#         'max_freq_log2': multires - 1,
#         'num_freqs': multires,
#         'log_sampling': True,
#         'periodic_fns': [torch.sin, torch.cos],
#     }
#
#     embedder_obj = Embedder(**embed_kwargs)
#
#     def embed(x, eo=embedder_obj): return eo.embed(x)
#
#     return embed, embedder_obj.out_dim
#
#
# class SDFNetwork(nn.Module):
#     def __init__(self,
#                  d_in,
#                  d_out,
#                  d_hidden,
#                  n_layers,
#                  skip_in=(4,),
#                  multires=0,
#                  bias=0.5,
#                  scale=1,
#                  geometric_init=True,
#                  weight_norm=True,
#                  inside_outside=False,
#                  sdf_activation='none',
#                  layer_activation='softplus'):
#         super(SDFNetwork, self).__init__()
#
#         dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
#
#         self.embed_fn_fine = None
#
#         if multires > 0:
#             embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
#             self.embed_fn_fine = embed_fn
#             dims[0] = input_ch
#
#         self.num_layers = len(dims)
#         self.skip_in = skip_in
#         self.scale = scale
#
#         for l in range(0, self.num_layers - 1):
#             if l + 1 in self.skip_in:
#                 out_dim = dims[l + 1] - dims[0]
#             else:
#                 out_dim = dims[l + 1]
#
#             lin = nn.Linear(dims[l], out_dim)
#
#             if geometric_init:
#                 if l == self.num_layers - 2:
#                     if not inside_outside:
#                         torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
#                         torch.nn.init.constant_(lin.bias, -bias)
#                     else:
#                         torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
#                         torch.nn.init.constant_(lin.bias, bias)
#                 elif multires > 0 and l == 0:
#                     torch.nn.init.constant_(lin.bias, 0.0)
#                     torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
#                     torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
#                 elif multires > 0 and l in self.skip_in:
#                     torch.nn.init.constant_(lin.bias, 0.0)
#                     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
#                     torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
#                 else:
#                     torch.nn.init.constant_(lin.bias, 0.0)
#                     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
#
#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)
#
#             setattr(self, "lin" + str(l), lin)
#
#         if layer_activation == 'softplus':
#             self.activation = nn.Softplus(beta=100)
#         elif layer_activation == 'relu':
#             self.activation = nn.ReLU()
#         else:
#             raise NotImplementedError
#
#     def forward(self, inputs):
#         inputs = inputs * self.scale
#         if self.embed_fn_fine is not None:
#             inputs = self.embed_fn_fine(inputs)
#
#         x = inputs
#         for l in range(0, self.num_layers - 1):
#             lin = getattr(self, "lin" + str(l))
#
#             if l in self.skip_in:
#                 x = torch.cat([x, inputs], -1) / np.sqrt(2)
#
#             x = lin(x)
#
#             if l < self.num_layers - 2:
#                 x = self.activation(x)
#
#         return x
#
#     def sdf(self, x):
#         return self.forward(x)[..., :1]
#
#     def sdf_hidden_appearance(self, x):
#         return self.forward(x)
#
#     def gradient(self, x):
#         x.requires_grad_(True)
#         with torch.enable_grad():
#             y = self.sdf(x)
#         d_output = torch.ones_like(y, requires_grad=False, device=y.device)
#         gradients = torch.autograd.grad(
#             outputs=y,
#             inputs=x,
#             grad_outputs=d_output,
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True)[0]
#         return gradients
#
#     def sdf_normal(self, x):
#         x.requires_grad_(True)
#         with torch.enable_grad():
#             y = self.sdf(x)
#         d_output = torch.ones_like(y, requires_grad=False, device=y.device)
#         gradients = torch.autograd.grad(
#             outputs=y,
#             inputs=x,
#             grad_outputs=d_output,
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True)[0]
#         return y[..., :1].detach(), gradients.detach()
#
#
# class SingleVarianceNetwork(nn.Module):
#     def __init__(self, init_val, activation='exp'):
#         super(SingleVarianceNetwork, self).__init__()
#         self.act = activation
#         self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
#
#     def forward(self, x):
#         if self.act == 'exp':
#             return torch.ones([*x.shape[:-1], 1]) * torch.exp(self.variance * 10.0)
#         elif self.act == 'linear':
#             return torch.ones([*x.shape[:-1], 1]) * self.variance * 10.0
#         elif self.act == 'square':
#             return torch.ones([*x.shape[:-1], 1]) * (self.variance * 10.0) ** 2
#         else:
#             raise NotImplementedError
#
#     def warp(self, x, inv_s):
#         return torch.ones([*x.shape[:-1], 1]) * inv_s
#
#
# # This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
# class NeRFNetwork(nn.Module):
#     def __init__(self,
#                  D=8,
#                  W=256,
#                  d_in=3,
#                  d_in_view=3,
#                  multires=0,
#                  multires_view=0,
#                  output_ch=4,
#                  skips=[4],
#                  use_viewdirs=False):
#         super(NeRFNetwork, self).__init__()
#         self.D = D
#         self.W = W
#         self.d_in = d_in
#         self.d_in_view = d_in_view
#         self.input_ch = 3
#         self.input_ch_view = 3
#         self.embed_fn = None
#         self.embed_fn_view = None
#
#         if multires > 0:
#             embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
#             self.embed_fn = embed_fn
#             self.input_ch = input_ch
#
#         if multires_view > 0:
#             embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
#             self.embed_fn_view = embed_fn_view
#             self.input_ch_view = input_ch_view
#
#         self.skips = skips
#         self.use_viewdirs = use_viewdirs
#
#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(self.input_ch, W)] +
#             [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])
#
#         ### Implementation according to the official code release
#         ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
#         self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])
#
#         ### Implementation according to the paper
#         # self.views_linears = nn.ModuleList(
#         #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
#
#         if use_viewdirs:
#             self.feature_linear = nn.Linear(W, W)
#             self.alpha_linear = nn.Linear(W, 1)
#             self.rgb_linear = nn.Linear(W // 2, 3)
#         else:
#             self.output_linear = nn.Linear(W, output_ch)
#
#     def forward(self, input_pts, input_views):
#         if self.embed_fn is not None:
#             input_pts = self.embed_fn(input_pts)
#         if self.embed_fn_view is not None:
#             input_views = self.embed_fn_view(input_views)
#
#         h = input_pts
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([input_pts, h], -1)
#
#         if self.use_viewdirs:
#             alpha = self.alpha_linear(h)
#             feature = self.feature_linear(h)
#             h = torch.cat([feature, input_views], -1)
#
#             for i, l in enumerate(self.views_linears):
#                 h = self.views_linears[i](h)
#                 h = F.relu(h)
#
#             rgb = self.rgb_linear(h)
#             return alpha, rgb
#         else:
#             assert False
#
#     def density(self, input_pts):
#         if self.embed_fn is not None:
#             input_pts = self.embed_fn(input_pts)
#
#         h = input_pts
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([input_pts, h], -1)
#
#         alpha = self.alpha_linear(h)
#         return alpha
#
#
# class IdentityActivation(nn.Module):
#     def forward(self, x): return x
#
#
# class ExpActivation(nn.Module):
#     def __init__(self, max_light=5.0):
#         super().__init__()
#         self.max_light = max_light
#
#     def forward(self, x):
#         return torch.exp(torch.clamp(x, max=self.max_light))
#
#
# def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid',
#                    exp_max=0.0) -> object:
#     if activation == 'sigmoid':
#         activation = nn.Sigmoid()
#     elif activation == 'exp':
#         activation = ExpActivation(max_light=exp_max)
#     elif activation == 'none':
#         activation = IdentityActivation()
#     elif activation == 'relu':
#         activation = nn.ReLU()
#     else:
#         raise NotImplementedError
#
#     run_dim = 256
#     if weight_norm:
#         module = nn.Sequential(
#             nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
#             activation,
#         )
#     else:
#         module = nn.Sequential(
#             nn.Linear(feats_dim, run_dim),
#             nn.ReLU(),
#             nn.Linear(run_dim, run_dim),
#             nn.ReLU(),
#             nn.Linear(run_dim, run_dim),
#             nn.ReLU(),
#             nn.Linear(run_dim, output_dim),
#             activation,
#         )
#
#     return module
#
#
# def get_camera_plane_intersection(pts, dirs, poses):
#     """
#     compute the intersection between the rays and the camera XoY plane
#     :param pts:      pn,3
#     :param dirs:     pn,3
#     :param poses:    pn,3,4
#     :return:
#     """
#     R, t = poses[:, :, :3], poses[:, :, 3:]
#
#     # transfer into human coordinate
#     pts_ = (R @ pts[:, :, None] + t)[..., 0]  # pn,3
#     dirs_ = (R @ dirs[:, :, None])[..., 0]  # pn,3
#
#     hits = torch.abs(dirs_[..., 2]) > 1e-4
#     dirs_z = dirs_[:, 2]
#     dirs_z[~hits] = 1e-4
#     dist = -pts_[:, 2] / dirs_z
#     inter = pts_ + dist.unsqueeze(-1) * dirs_
#     return inter, dist, hits
#
#
# def expected_sin(mean, var):
#     """Compute the mean of sin(x), x ~ N(mean, var)."""
#     return torch.exp(-0.5 * var) * torch.sin(mean)  # large var -> small value.
#
#
# def IPE(mean, var, min_deg, max_deg):
#     scales = 2 ** torch.arange(min_deg, max_deg)
#     shape = mean.shape[:-1] + (-1,)
#     scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
#     scaled_var = torch.reshape(var[..., None, :] * scales[:, None] ** 2, shape)
#     return expected_sin(torch.cat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1),
#                         torch.cat([scaled_var] * 2, dim=-1))
#
#
# def offset_points_to_sphere(points):
#     points_norm = torch.norm(points, dim=-1)
#     mask = points_norm > 0.999
#     if torch.sum(mask) > 0:
#         points = torch.clone(points)
#         points[mask] /= points_norm[mask].unsqueeze(-1)
#         points[mask] *= 0.999
#         # points[points_norm>0.999] = 0
#     return points
#
#
# def get_sphere_intersection(pts, dirs):
#     dtx = torch.sum(pts * dirs, dim=-1, keepdim=True)  # rn,1
#     xtx = torch.sum(pts ** 2, dim=-1, keepdim=True)  # rn,1
#     dist = dtx ** 2 - xtx + 1
#     assert torch.sum(dist < 0) == 0
#     dist = -dtx + torch.sqrt(dist + 1e-6)  # rn,1
#     return dist
#
#
# # this function is borrowed from NeuS
# def sample_pdf(bins, weights, n_samples, det=False):
#     # This implementation is from NeRF
#     # Get pdf
#     weights = weights + 1e-5  # prevent nans
#     pdf = weights / torch.sum(weights, -1, keepdim=True)
#     cdf = torch.cumsum(pdf, -1)
#     cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
#     # Take uniform samples
#     if det:
#         u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
#         u = u.expand(list(cdf.shape[:-1]) + [n_samples])
#     else:
#         u = torch.rand(list(cdf.shape[:-1]) + [n_samples])
#
#     # Invert CDF
#     u = u.contiguous()
#     inds = torch.searchsorted(cdf, u, right=True)
#     below = torch.max(torch.zeros_like(inds - 1), inds - 1)
#     above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
#     inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
#
#     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
#     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
#     bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
#
#     denom = (cdf_g[..., 1] - cdf_g[..., 0])
#     denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
#     t = (u - cdf_g[..., 0]) / denom
#     samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
#
#     return samples
#
#
# def get_weights(sdf_fun, inv_fun, z_vals, origins, dirs):
#     points = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2)  # pn,sn,3
#     inv_s = inv_fun(points[:, :-1, :])[..., 0]  # pn,sn-1
#     sdf = sdf_fun(points)[..., 0]  # pn,sn
#
#     prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # pn,sn-1
#     prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
#     mid_sdf = (prev_sdf + next_sdf) * 0.5
#     cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # pn,sn-1
#     surface_mask = (cos_val < 0)  # pn,sn-1
#     cos_val = torch.clamp(cos_val, max=0)
#
#     dist = next_z_vals - prev_z_vals  # pn,sn-1
#     prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # pn, sn-1
#     next_esti_sdf = mid_sdf + cos_val * dist * 0.5
#     prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
#     next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
#     alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) * surface_mask.float()
#     weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
#     mid_sdf[~surface_mask] = -1.0
#     return weights, mid_sdf
#
#
# def get_intersection(sdf_fun, inv_fun, pts, dirs, sn0=128, sn1=9):
#     """
#     :param sdf_fun:
#     :param inv_fun:
#     :param pts:    pn,3
#     :param dirs:   pn,3
#     :param sn0:
#     :param sn1:
#     :return:
#     """
#     inside_mask = torch.norm(pts, dim=-1) < 0.999  # left some margin
#     pn, _ = pts.shape
#     hit_z_vals = torch.zeros([pn, sn1 - 1])
#     hit_weights = torch.zeros([pn, sn1 - 1])
#     hit_sdf = -torch.ones([pn, sn1 - 1])
#     if torch.sum(inside_mask) > 0:
#         pts = pts[inside_mask]
#         dirs = dirs[inside_mask]
#         max_dist = get_sphere_intersection(pts, dirs)  # pn,1
#         with torch.no_grad():
#             z_vals = torch.linspace(0, 1, sn0)  # sn0
#             z_vals = max_dist * z_vals.unsqueeze(0)  # pn,sn0
#             weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, pts, dirs)  # pn,sn0-1
#             z_vals_new = sample_pdf(z_vals, weights, sn1, True)  # pn,sn1
#             weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, pts, dirs)  # pn,sn1-1
#             z_vals_mid = (z_vals_new[:, 1:] + z_vals_new[:, :-1]) * 0.5
#
#         hit_z_vals[inside_mask] = z_vals_mid
#         hit_weights[inside_mask] = weights
#         hit_sdf[inside_mask] = mid_sdf
#     return hit_z_vals, hit_weights, hit_sdf
#
#
# class AppShadingNetwork(nn.Module):
#     default_cfg = {
#         'human_light': False,
#         'sphere_direction': False,
#         'light_pos_freq': 8,
#         'inner_init': -0.95,
#         'roughness_init': 0.0,
#         'metallic_init': 0.0,
#         'light_exp_max': 0.0,
#     }
#
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = {**self.default_cfg, **cfg}
#         feats_dim = 256
#
#         # material MLPs
#         self.metallic_predictor = make_predictor(feats_dim + 3, 1)
#         if self.cfg['metallic_init'] != 0:
#             nn.init.constant_(self.metallic_predictor[-2].bias, self.cfg['metallic_init'])
#         self.roughness_predictor = make_predictor(feats_dim + 3, 1)
#         if self.cfg['roughness_init'] != 0:
#             nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
#         self.albedo_predictor = make_predictor(feats_dim + 3, 3)
#
#         FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
#         self.register_buffer('FG_LUT', FG_LUT)
#
#         self.sph_enc = generate_ide_fn(5)
#         self.dir_enc, dir_dim = get_embedder(6, 3)
#         self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)
#         exp_max = self.cfg['light_exp_max']
#         # outer lights are direct lights
#         if self.cfg['sphere_direction']:
#             self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=exp_max)
#         else:
#             self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
#         nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))
#
#         # inner lights are indirect lights
#         self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
#         nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
#         self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='none')
#         nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])
#
#         # human lights are the lights reflected from the photo capturer
#         if self.cfg['human_light']:
#             self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation='exp')
#             nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))
#
#     def predict_human_light(self, points, reflective, human_poses, roughness):
#         inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
#         scale_factor = 0.3
#         mean = inter[..., :2] * scale_factor
#         var = roughness * (dists[:, None] * scale_factor) ** 2
#         hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
#         hits = hits.float().unsqueeze(-1)
#         mean = mean * hits
#         var = var * hits
#
#         var = var.expand(mean.shape[0], 2)
#         pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
#         human_lights = self.human_light_predictor(pos_enc)
#         human_lights = human_lights * hits
#         human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
#         human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
#         return human_lights, human_weights
#
#     def predict_specular_lights(self, points, feature_vectors, reflective, roughness, human_poses, step):
#         human_light, human_weight = 0, 0
#         ref_roughness = self.sph_enc(reflective, roughness)
#         pts = self.pos_enc(points)
#         if self.cfg['sphere_direction']:
#             sph_points = offset_points_to_sphere(points)
#             sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)
#             sph_points = self.sph_enc(sph_points, roughness)
#             direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1))
#         else:
#             direct_light = self.outer_light(ref_roughness)
#
#         if self.cfg['human_light']:
#             human_light, human_weight = self.predict_human_light(points, reflective, human_poses, roughness)
#
#         indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
#         ref_ = self.dir_enc(reflective)
#         occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1))  # this is occlusion prob
#         occ_prob = occ_prob * 0.5 + 0.5
#         occ_prob_ = torch.clamp(occ_prob, min=0, max=1)
#
#         light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (
#                 1 - occ_prob_)
#         indirect_light = indirect_light * occ_prob_
#         return light, occ_prob, indirect_light, human_light * human_weight
#
#     def predict_diffuse_lights(self, points, feature_vectors, normals):
#         roughness = torch.ones([normals.shape[0], 1])
#         ref = self.sph_enc(normals, roughness)  # von Mises-Fisher distribution
#         if self.cfg['sphere_direction']:
#             sph_points = offset_points_to_sphere(points)
#             sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1)
#             sph_points = self.sph_enc(sph_points, roughness)
#             light = self.outer_light(torch.cat([ref, sph_points], -1))
#         else:
#             light = self.outer_light(ref)
#         return light
#
#     def forward(self, points, normals, view_dirs, feature_vectors, human_poses, inter_results=False, step=None):
#         normals = F.normalize(normals, dim=-1)
#         view_dirs = F.normalize(view_dirs, dim=-1)
#         reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
#         NoV = torch.sum(normals * view_dirs, -1, keepdim=True)
#
#         metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
#         roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
#         albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
#
#         # diffuse light
#         diffuse_albedo = (1 - metallic) * albedo
#         diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
#         diffuse_color = diffuse_albedo * diffuse_light
#
#         # specular light
#         specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
#         specular_light, occ_prob, indirect_light, human_light = self.predict_specular_lights(points, feature_vectors,
#                                                                                              reflective, roughness,
#                                                                                              human_poses, step)
#
#         fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
#         pn, bn = points.shape[0], 1
#         fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode='linear',
#                                boundary_mode='clamp').reshape(pn, 2)
#         specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
#         specular_color = specular_ref * specular_light
#
#         # integrated together
#         color = diffuse_color + specular_color
#
#         # gamma correction
#         diffuse_color = linear_to_srgb(diffuse_color)
#         specular_color = linear_to_srgb(specular_color)
#         color = linear_to_srgb(color)
#         color = torch.clamp(color, min=0.0, max=1.0)
#
#         occ_info = {
#             'reflective': reflective,
#             'occ_prob': occ_prob,
#         }
#
#         if inter_results:
#             intermediate_results = {
#                 'specular_albedo': specular_albedo,
#                 'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
#                 'specular_light': torch.clamp(linear_to_srgb(specular_light), min=0.0, max=1.0),
#                 'specular_color': torch.clamp(specular_color, min=0.0, max=1.0),
#
#                 'diffuse_albedo': diffuse_albedo,
#                 'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
#                 'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),
#
#                 # 'metallic': metallic,
#                 # 'roughness': roughness,
#
#                 'occ_prob': torch.clamp(occ_prob, max=1.0, min=0.0),
#                 'indirect_light': indirect_light,
#             }
#             if self.cfg['human_light']:
#                 intermediate_results['human_light'] = linear_to_srgb(human_light)
#             return color, occ_info, intermediate_results
#         else:
#             return color, occ_info
#
#     def predict_materials(self, points, feature_vectors):
#         metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
#         roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
#         albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
#         return metallic, roughness, albedo
#
#
# class MaterialFeatsNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pos_enc, input_dim = get_embedder(8, 3)
#         run_dim = 256
#         self.module0 = nn.Sequential(
#             nn.utils.weight_norm(nn.Linear(input_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
#             nn.ReLU(),
#         )
#         self.module1 = nn.Sequential(
#             nn.utils.weight_norm(nn.Linear(input_dim + run_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
#             nn.ReLU(),
#             nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
#         )
#
#     def forward(self, x):
#         x = self.pos_enc(x)
#         input = x
#         x = self.module0(x)
#         return self.module1(torch.cat([x, input], -1))
#
#
# def saturate_dot(v0, v1):
#     return torch.clamp(torch.sum(v0 * v1, dim=-1, keepdim=True), min=0.0, max=1.0)
#
#
# from nerfactor.third_party.xiuminglib import xiuminglib as xm
# from nerfactor.nerfactor.models.brdf import Model as BRDFModel
# # from nerfactor.nerfactor.networks.embedder import Embedder as nerfactor_Embedder
# from nerfactor.nerfactor.util import config as configutil, \
#     io as ioutil
#
# from utils import geom as geomutil
# from network.embedder import Embedder as nerfactor_Embedder
#
#
# # Custom PyTorch autograd function
# # class TensorFlowBridge(torch.autograd.Function):
# #     tf_model = None  # Class-level attribute to hold the TensorFlow model
# #
# #     @staticmethod
# #     def initialize_model(model):
# #         """Initialize the TensorFlow model."""
# #         TensorFlowBridge.tf_model = model
# #
# #
# #     @staticmethod
# #     def forward(ctx, input_tensor):
# #         # Check if the TensorFlow model has been initialized
# #         if TensorFlowBridge.tf_model is None:
# #             raise ValueError("TensorFlow model is not initialized. Call `TensorFlowBridge.initialize_model()` first.")
# #
# #         # Convert PyTorch tensor to TensorFlow tensor
# #         tf_input = tf.convert_to_tensor(input_tensor.detach().cpu().numpy(), dtype=tf.float32)
# #
# #         # Perform inference using the frozen TensorFlow model
# #         with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
# #             brdf_mlp = TensorFlowBridge.tf_model.net['brdf_mlp']
# #             brdf_out = TensorFlowBridge.tf_model.net['brdf_out']
# #
# #             # Forward pass through the frozen model
# #             mlp_output = brdf_mlp(tf_input)
# #             tf_output = brdf_out(mlp_output)
# #
# #         # Convert TensorFlow output back to PyTorch tensor
# #         output_tensor = torch.tensor(tf_output.numpy(), device=input_tensor.device)
# #
# #         # Save TensorFlow input and the intermediate outputs for backward pass
# #         ctx.save_for_backward(input_tensor)
# #         ctx.tf_input = tf_input
# #         ctx.mlp_output = mlp_output
# #
# #         return output_tensor
# #
# #     @staticmethod
# #     def backward(ctx, grad_output):
# #         # Retrieve saved tensors
# #         input_tensor, = ctx.saved_tensors
# #         tf_input = ctx.tf_input
# #         mlp_output = ctx.mlp_output
# #
# #         # Convert PyTorch gradient to TensorFlow tensor
# #         tf_grad_output = tf.convert_to_tensor(grad_output.cpu().numpy(), dtype=tf.float32)
# #
# #         # Compute gradient w.r.t. input using TensorFlow's `GradientTape`
# #         with tf.GradientTape() as tape:
# #             tape.watch(tf_input)  # Track input tensor for gradients
# #             tf_output = mlp_output  # Frozen MLP already evaluated
# #         tf_grad_input = tape.gradient(tf_output, tf_input, output_gradients=tf_grad_output)
# #
# #         # Convert TensorFlow gradient back to PyTorch tensor
# #         grad_input = torch.tensor(tf_grad_input.numpy(), device=input_tensor.device)
# #
# #         return grad_input
#
#
# class MCShadingNetwork(nn.Module):
#     default_cfg = {
#         'diffuse_sample_num': 512,
#         'specular_sample_num': 256,
#         'human_lights': True,
#         'light_exp_max': 5.0,
#         'inner_light_exp_max': 5.0,
#         'outer_light_version': 'direction',
#         'geometry_type': 'schlick',
#
#         'reg_change': True,
#         'change_eps': 0.01,
#         'change_type': 'gaussian',
#         'reg_lambda1': 0.05,
#         'reg_min_max': True,
#
#         'random_azimuth': True,
#         'is_real': False,
#     }
#
#     def __init__(self, cfg, ray_trace_fun):
#         self.cfg = {**self.default_cfg, **cfg}
#         super().__init__()
#
#         # Configurations
#         config_ini = "nerfactor/nerfactor/config/nerfactor.ini"
#         config = ioutil.read_config(config_ini)
#         self.nerfactor_config = config
#         # BRDF nerfactor part
#         brdf_ckpt = config.get('DEFAULT', 'brdf_model_ckpt')
#         brdf_config_path = configutil.get_config_ini(brdf_ckpt)
#         print(brdf_config_path)
#         self.nerfactor_config_brdf = ioutil.read_config(brdf_config_path)
#         self.nerfactor_pred_brdf = config.getboolean('DEFAULT', 'pred_brdf')
#         self.nerfactor_z_dim = self.nerfactor_config_brdf.getint('DEFAULT', 'z_dim')
#         self.nerfactor_normalize_brdf_z = self.nerfactor_config_brdf.getboolean(
#             'DEFAULT', 'normalize_z')
#
#         # Pre-trained BRDF Decoder
#         self.nerfactor_albedo_smooth_weight = config.getfloat(
#             'DEFAULT', 'albedo_smooth_weight')
#         self.nerfactor_brdf_smooth_weight = config.getfloat(
#             'DEFAULT', 'brdf_smooth_weight')
#         self.nerfactor_brdf_model = BRDFModel(self.nerfactor_config_brdf)
#         ioutil.restore_model(self.nerfactor_brdf_model, brdf_ckpt)
#         self.nerfactor_brdf_model.trainable = False
#         self.nerfactor_brdf_smooth_weight = config.getfloat(
#             'DEFAULT', 'brdf_smooth_weight')
#         # TensorFlowBridge.initialize_model(self.nerfactor_brdf_model)
#
#         # BRDF Encoder
#         mlp_width = config.getint('DEFAULT', 'mlp_width')
#         mlp_depth = config.getint('DEFAULT', 'mlp_depth')
#         mlp_skip_at = config.getint('DEFAULT', 'mlp_skip_at')
#         self.nerfactor_net = {}
#         # BRDF Z
#         from network.mlp_brdf import MLPNetwork as brdf_mlp
#         self.nerfactor_embedder = self._init_nerfactor_embedder()
#         self.nerfactor_net['brdf_z_mlp'] = brdf_mlp(
#             [self.nerfactor_embedder['xyz'].out_dims] + [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
#             skip_at=[mlp_skip_at])
#         self.nerfactor_net['brdf_z_out'] = brdf_mlp([mlp_width, self.nerfactor_z_dim], act=None)
#
#         # setting up the nerfactor embedder
#
#         self.nerfactor_xyz_scale = self.nerfactor_config.getfloat(
#             'DEFAULT', 'xyz_scale', fallback=1.)
#         self.nerfactor_smooth_use_l1 = self.nerfactor_config.getboolean('DEFAULT', 'smooth_use_l1')
#         self.nerfactor_smooth_loss = nn.L1Loss() if self.nerfactor_smooth_use_l1 else nn.MSELoss()
#         # PSNR calculator
#         self.psnr = xm.metric.PSNR('uint8')
#         ######
#
#         self.feats_network = MaterialFeatsNetwork()
#         self.metallic_predictor = make_predictor(256 + 3, 1)
#         self.roughness_predictor = make_predictor(256 + 3, 1)
#         self.albedo_predictor = make_predictor(256 + 3, 3)
#
#         # light part
#         self.sph_enc = generate_ide_fn(5)
#         self.dir_enc, dir_dim = get_embedder(6, 3)
#         self.pos_enc, pos_dim = get_embedder(8, 3)
#         if self.cfg['outer_light_version'] == 'direction':
#             self.outer_light = make_predictor(72, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
#         elif self.cfg['outer_light_version'] == 'sphere_direction':
#             self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
#         else:
#             raise NotImplementedError
#         nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))
#         if self.cfg['human_lights']:
#             self.human_light = make_predictor(2 * 2 * 6, 4, activation='exp')
#             nn.init.constant_(self.human_light[-2].bias, np.log(0.02))
#         self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=self.cfg['inner_light_exp_max'])
#         nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
#
#         # predefined diffuse sample directions
#         az, el = sample_sphere(self.cfg['diffuse_sample_num'], 0)
#         az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
#         self.diffuse_direction_samples = np.stack([az, el], -1)
#         self.diffuse_direction_samples = torch.from_numpy(
#             self.diffuse_direction_samples.astype(np.float32)).cuda()  # [dn0,2]
#
#         az, el = sample_sphere(self.cfg['specular_sample_num'], 0)
#         az, el = az * 0.5 / np.pi, 1 - 2 * el / np.pi  # scale to [0,1]
#         self.specular_direction_samples = np.stack([az, el], -1)
#         self.specular_direction_samples = torch.from_numpy(
#             self.specular_direction_samples.astype(np.float32)).cuda()  # [dn1,2]
#
#         az, el = sample_sphere(8192, 0)
#         light_pts = az_el_to_points(az, el)
#         self.register_buffer('light_pts', torch.from_numpy(light_pts.astype(np.float32)))
#         self.ray_trace_fun = ray_trace_fun
#
#     def get_orthogonal_directions(self, directions):
#         x, y, z = torch.split(directions, 1, dim=-1)  # pn,1
#         otho0 = torch.cat([y, -x, torch.zeros_like(x)], -1)
#         otho1 = torch.cat([-z, torch.zeros_like(x), x], -1)
#         mask0 = torch.norm(otho0, dim=-1) > torch.norm(otho1, dim=-1)
#         mask1 = ~mask0
#         otho = torch.zeros_like(directions)
#         otho[mask0] = otho0[mask0]
#         otho[mask1] = otho1[mask1]
#         otho = F.normalize(otho, dim=-1)
#         return otho
#
#     def sample_diffuse_directions(self, normals, is_train):
#         # normals [pn,3]
#         z = normals  # pn,3
#         x = self.get_orthogonal_directions(normals)  # pn,3
#         y = torch.cross(z, x, dim=-1)  # pn,3
#         # y = torch.cross(z, x, dim=-1) # pn,3
#
#         # project onto this tangent space
#         az, el = torch.split(self.diffuse_direction_samples, 1, dim=1)  # sn,1
#         el, az = el.unsqueeze(0), az.unsqueeze(0)
#         az = az * np.pi * 2
#         el_sqrt = torch.sqrt(el + 1e-7)
#         if is_train and self.cfg['random_azimuth']:
#             az = (az + torch.rand(z.shape[0], 1, 1) * np.pi * 2) % (2 * np.pi)
#         coeff_z = torch.sqrt(1 - el + 1e-7)
#         coeff_x = el_sqrt * torch.cos(az)
#         coeff_y = el_sqrt * torch.sin(az)
#
#         directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1)  # pn,sn,3
#         return directions
#
#     def reflect(self, v, n):
#         """
#         Reflects vector v about the normal n.
#         v and n can be of shape (..., 3).
#         """
#         dot = (v * n).sum(dim=-1, keepdim=True)
#         return v - 2 * dot * n
#
#     def sample_specular_rays(self, normals, view_dirs, roughness, n_samples):
#         """
#         Samples specular ray directions for a batch of surface points.
#
#         Parameters:
#             normals   : (B, 3) torch.Tensor of surface normals.
#             view_dirs : (B, 3) torch.Tensor of view directions (from surface point to camera).
#             roughness : scalar or (B,) torch.Tensor in [0, 1]. 0 is smooth (sharp reflection).
#             n_samples : int, number of samples per surface point.
#             device    : device to use ('cpu' or 'cuda').
#
#         Returns:
#             sample_dirs: (B, n_samples, 3) torch.Tensor of sampled incident directions.
#             pdfs       : (B, n_samples) torch.Tensor of PDF values for each sample.
#         """
#         device = normals.device
#         B = normals.shape[0]
#
#         # Ensure normals and view_dirs are normalized.
#         normals = mathutil.safe_l2_normalize(normals, axis=-1)
#         view_dirs = mathutil.safe_l2_normalize(view_dirs, axis=-1)
#
#         # Compute the perfect reflection direction for each point.
#         # Note: we reflect -view_dir about the normal.
#         r = self.reflect(-view_dirs, normals)
#         r = mathutil.safe_l2_normalize(r, axis=-1)
#
#         # Ensure roughness is a tensor of shape (B,)
#         if not torch.is_tensor(roughness):
#             roughness = torch.tensor(roughness, device=device, dtype=normals.dtype).expand(B)
#         elif roughness.dim() == 0:
#             roughness = roughness.expand(B)
#
#         # Map roughness to a cosine-power exponent.
#         k = 100.0  # scaling constant (adjust as needed)
#         exponent = k * (1 - roughness) ** 2  # shape (B,)
#
#         # --- Build an Orthonormal Basis for Each Surface Point ---
#         # We want to compute two vectors (tangent and bitangent) perpendicular to r.
#         # For each batch element, we choose an arbitrary vector that is not parallel to r.
#         cond = (torch.abs(r[:, 0]) < 0.99).unsqueeze(-1)  # shape (B, 1)
#         arbitrary = torch.where(cond,
#                                 torch.tensor([1.0, 0.0, 0.0], device=device, dtype=r.dtype),
#                                 torch.tensor([0.0, 1.0, 0.0], device=device, dtype=r.dtype))
#         arbitrary = arbitrary.expand_as(r)  # (B, 3)
#
#         tangent = torch.cross(r, arbitrary, dim=-1)  # (B, 3)
#         tangent = mathutil.safe_l2_normalize(tangent, axis=-1)
#         bitangent = torch.cross(r, tangent, dim=-1)  # (B, 3)
#         bitangent = mathutil.safe_l2_normalize(bitangent, axis=-1)
#
#         # --- Sample from the Cosine-Power Distribution ---
#         # Generate random numbers for each batch element and sample.
#         u1 = torch.rand(B, n_samples, device=device, dtype=normals.dtype)
#         u2 = torch.rand(B, n_samples, device=device, dtype=normals.dtype)
#
#         # Unsqueeze exponent to match sample shape: (B, 1)
#         exp_unsq = exponent.unsqueeze(1)
#
#         # Invert the CDF to sample theta for each point:
#         val = u1 ** (1.0 / (exp_unsq + 1.0))
#         val = torch.clamp(val, min=-1.0, max=1.0)  # Ensure it's in [-1, 1]
#         theta = torch.acos(val)  # (B, n_samples)
#         phi = 2 * np.pi * u2  # (B, n_samples)
#
#         sin_theta = torch.sin(theta)
#         cos_theta = torch.cos(theta)  # equals z in local coordinates
#
#         # Local spherical coordinates (x, y, z) for each sample:
#         x = sin_theta * torch.cos(phi)  # (B, n_samples)
#         y = sin_theta * torch.sin(phi)  # (B, n_samples)
#         z = cos_theta  # (B, n_samples)
#
#         # Stack to form local directions: (B, n_samples, 3)
#         local_dirs = torch.stack([x, y, z], dim=-1)
#
#         # --- Transform to World Coordinates ---
#         # For each surface point, combine the local coordinates with the orthonormal basis.
#         # tangent, bitangent, and r are each (B, 3); we unsqueeze them to (B, 1, 3) for broadcasting.
#         sample_dirs = (local_dirs[..., 0:1] * tangent.unsqueeze(1) +
#                        local_dirs[..., 1:2] * bitangent.unsqueeze(1) +
#                        local_dirs[..., 2:3] * r.unsqueeze(1))
#         sample_dirs = mathutil.safe_l2_normalize(sample_dirs, axis=-1)  # (B, n_samples, 3)
#
#         # --- Compute the PDF for Each Sample ---
#         # The PDF for the cosine-power distribution:
#         #   pdf(theta) = (exponent + 1)/(2π) * (cos(theta))^(exponent)
#
#         if torch.isinf(exp_unsq).any():
#             print('inf in exp_unsq', exp_unsq)
#         if torch.isnan(exp_unsq).any():
#             print('nan in exp_unsq', exp_unsq)
#
#         if torch.isinf(cos_theta).any():
#             print('inf in cos_theta', cos_theta)
#         if torch.isnan(cos_theta).any():
#             print('nan in cos_theta', cos_theta)
#
#         eps = 1e-6
#         cos_theta_clamped = torch.clamp(torch.cos(theta), min=eps)
#
#         exp_component = cos_theta_clamped ** exp_unsq  # (B, n_samples)
#         if torch.isinf(exp_component).any():
#             print('inf in exp_component', exp_component)
#         if torch.isnan(exp_component).any():
#             print('nan in exp_component', exp_component)
#
#         pdfs = (exp_unsq + 1) / (2 * np.pi) * (cos_theta_clamped ** exp_unsq)  # (B, n_samples)
#         if torch.isinf(pdfs).any():
#             print('*inf in pdfs', pdfs)
#         if torch.isnan(pdfs).any():
#             print('*nan in pdfs', pdfs)
#         return sample_dirs, pdfs
#
#     def sample_cosine_weighted_rays(self, normals, num_samples=1):
#         """
#         Samples cosine-weighted ray directions for a batch of points.
#
#         Parameters:
#             normals (torch.Tensor): A tensor of shape (B, 3) containing surface normals (assumed normalized).
#             num_samples (int): The number of ray samples per point.
#
#         Returns:
#             world_dirs (torch.Tensor): A tensor of shape (B, num_samples, 3) containing the sampled ray directions in world space.
#             pdf (torch.Tensor): A tensor of shape (B, num_samples) containing the probability density function values for each sample.
#                                For cosine-weighted sampling, pdf = cos(theta) / pi.
#         """
#         B = normals.shape[0]
#         device = normals.device
#
#         # Ensure normals are normalized
#         normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
#
#         # Sample two independent uniform random variables for each ray sample: shape (B, num_samples)
#         u1 = torch.rand(B, num_samples, device=device)
#         u2 = torch.rand(B, num_samples, device=device)
#
#         # Convert uniform random numbers to polar coordinates for cosine-weighted sampling.
#         # The radial distance r = sqrt(u1) ensures cosine weighting.
#         r = torch.sqrt(u1)
#         theta = 2 * np.pi * u2
#
#         # Local (tangent-space) coordinates: x, y, z.
#         # Here, z = sqrt(1 - u1) so that the distribution is cosine-weighted.
#         x = r * torch.cos(theta)
#         y = r * torch.sin(theta)
#         z = torch.sqrt(torch.clamp(1.0 - u1, min=0.0))
#
#         # Stack to create local direction vectors of shape (B, num_samples, 3)
#         local_dirs = torch.stack([x, y, z], dim=-1)
#
#         # --- Build an Orthonormal Basis for Each Normal (Tangent Space) ---
#         # For each normal we choose an "up" vector. A common strategy is:
#         # if |normal_z| < 0.999, use up = [0, 0, 1], otherwise use up = [1, 0, 0].
#         up = torch.empty_like(normals)
#         # Create a mask for normals that are not nearly vertical.
#         mask = (normals[:, 2].abs() < 0.999)
#         up[mask] = torch.tensor([0.0, 0.0, 1.0], device=device)
#         up[~mask] = torch.tensor([1.0, 0.0, 0.0], device=device)
#
#         # Compute tangent = normalize(cross(up, normal))
#         tangent = torch.cross(up, normals, dim=-1)
#         tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
#
#         # Compute bitangent = cross(normal, tangent)
#         bitangent = torch.cross(normals, tangent, dim=-1)
#         bitangent = bitangent / (bitangent.norm(dim=-1, keepdim=True) + 1e-8)
#
#         # Expand tangent space vectors to shape (B, 1, 3) for broadcasting.
#         tangent = tangent.unsqueeze(1)  # Shape: (B, 1, 3)
#         bitangent = bitangent.unsqueeze(1)  # Shape: (B, 1, 3)
#         normals_exp = normals.unsqueeze(1)  # Shape: (B, 1, 3)
#
#         # Transform local directions to world space:
#         # world_dir = local_x * tangent + local_y * bitangent + local_z * normal.
#         world_dirs = (local_dirs[..., 0:1] * tangent +
#                       local_dirs[..., 1:2] * bitangent +
#                       local_dirs[..., 2:3] * normals_exp)
#
#         # Normalize the resulting world directions (optional but often good to do)
#         world_dirs = world_dirs / (world_dirs.norm(dim=-1, keepdim=True) + 1e-8)
#
#         # Compute PDF for each sample:
#         # For cosine-weighted sampling, pdf = cos(theta) / pi.
#         # In the local coordinate system, cos(theta) is the z component.
#         pdf = local_dirs[..., 2] / np.pi  # Shape: (B, num_samples)
#
#         return world_dirs, pdf
#
#     def get_inner_lights(self, points, view_dirs, normals):
#         pos_enc = self.pos_enc(points)
#         normals = F.normalize(normals, dim=-1)
#         view_dirs = F.normalize(view_dirs, dim=-1)
#         reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
#         dir_enc = self.sph_enc(reflections, 0)
#         return self.inner_light(torch.cat([pos_enc, dir_enc], -1))
#
#     def get_human_light(self, points, directions, human_poses):
#         inter, dists, hits = get_camera_plane_intersection(points, directions, human_poses)
#         scale_factor = 0.3
#         mean = inter[..., :2] * scale_factor
#         hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)
#         hits = hits.float().unsqueeze(-1)
#         mean = mean * hits
#
#         var = torch.zeros_like(mean)
#         pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
#         human_lights = self.human_light(pos_enc)
#         human_lights = human_lights * hits
#         human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
#         human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
#         return human_lights, human_weights
#
#     def predict_outer_lights(self, points, directions):
#         if self.cfg['outer_light_version'] == 'direction':
#             outer_enc = self.sph_enc(directions, 0)
#             outer_lights = self.outer_light(outer_enc)
#         elif self.cfg['outer_light_version'] == 'sphere_direction':
#             outer_dirs = directions
#             outer_pts = points
#             outer_enc = self.sph_enc(outer_dirs, 0)
#             mask = torch.norm(outer_pts, dim=-1) > 0.999
#             if torch.sum(mask) > 0:
#                 outer_pts = torch.clone(outer_pts)
#                 outer_pts[mask] *= 0.999  # shrink this point a little bit
#             dists = get_sphere_intersection(outer_pts, outer_dirs)
#             sphere_pts = outer_pts + outer_dirs * dists
#             sphere_pts = self.sph_enc(sphere_pts, 0)
#             outer_lights = self.outer_light(torch.cat([outer_enc, sphere_pts], -1))
#         else:
#             raise NotImplementedError
#         return outer_lights
#
#     def get_lights(self, points, directions, human_poses):
#         # trace
#         shape = points.shape[:-1]  # pn,sn
#         eps = 1e-5
#         inters, normals, depth, hit_mask = self.ray_trace_fun(points.reshape(-1, 3) + directions.reshape(-1, 3) * eps,
#                                                               directions.reshape(-1, 3))
#         inters, normals, depth, hit_mask = inters.reshape(*shape, 3), normals.reshape(*shape, 3), depth.reshape(*shape,
#                                                                                                                 1), hit_mask.reshape(
#             *shape)
#         miss_mask = ~hit_mask
#
#         # hit_mask
#         lights = torch.zeros(*shape, 3)
#         human_lights, human_weights = torch.zeros([1, 3]), torch.zeros([1, 1])
#         if torch.sum(miss_mask) > 0:
#             outer_lights = self.predict_outer_lights(points[miss_mask], directions[miss_mask])
#             if self.cfg['human_lights']:
#                 human_lights, human_weights = self.get_human_light(points[miss_mask], directions[miss_mask],
#                                                                    human_poses[miss_mask])
#             else:
#                 human_lights, human_weights = torch.zeros_like(outer_lights), torch.zeros(outer_lights.shape[0], 1)
#             lights[miss_mask] = outer_lights * (1 - human_weights) + human_lights * human_weights
#
#         if torch.sum(hit_mask) > 0:
#             lights[hit_mask] = self.get_inner_lights(inters[hit_mask], -directions[hit_mask], normals[hit_mask])
#
#         near_mask = (depth > eps).float()
#         lights = lights * near_mask  # very near surface does not bring lights
#         return lights, human_lights * human_weights, inters, normals, hit_mask
#
#     def fresnel_schlick(self, F0, HoV):
#         return F0 + (1.0 - F0) * torch.clamp(1.0 - HoV, min=0.0, max=1.0) ** 5.0
#
#     def fresnel_schlick_directions(self, F0, view_dirs, directions):
#         H = (view_dirs + directions)  # [pn,sn0,3]
#         H = F.normalize(H, dim=-1)
#         HoV = torch.clamp(torch.sum(H * view_dirs, dim=-1, keepdim=True), min=0.0, max=1.0)  # [pn,sn0,1]
#         fresnel = self.fresnel_schlick(F0, HoV)  # [pn,sn0,1]
#         return fresnel, H, HoV
#
#     def geometry_schlick_ggx(self, NoV, roughness):
#         a = roughness  # a = roughness**2: we assume the predicted roughness is already squared
#
#         k = a / 2
#         num = NoV
#         denom = NoV * (1 - k) + k
#         return num / (denom + 1e-5)
#
#     def geometry_schlick(self, NoV, NoL, roughness):
#         ggx2 = self.geometry_schlick_ggx(NoV, roughness)
#         ggx1 = self.geometry_schlick_ggx(NoL, roughness)
#         return ggx2 * ggx1
#
#     def geometry_ggx_smith_correlated(self, NoV, NoL, roughness):
#         def fun(alpha2, cos_theta):
#             # cos_theta = torch.clamp(cos_theta,min=1e-7,max=1-1e-7)
#             cos_theta2 = cos_theta ** 2
#             tan_theta2 = (1 - cos_theta2) / (cos_theta2 + 1e-7)
#             return 0.5 * torch.sqrt(1 + alpha2 * tan_theta2) - 0.5
#
#         alpha_sq = roughness ** 2
#         return 1.0 / (1.0 + fun(alpha_sq, NoV) + fun(alpha_sq, NoL))
#
#     def predict_materials(self, pts):
#         feats = self.feats_network(pts)
#         # metallic = self.metallic_predictor(torch.cat([feats, pts], -1))
#         metallic = None
#         # roughness = self.roughness_predictor(torch.cat([feats, pts], -1))
#         roughness = None
#         # rmax, rmin = 1.0, 0.04 ** 2
#         # roughness = roughness * (rmax - rmin) + rmin
#         albedo = self.albedo_predictor(torch.cat([feats, pts], -1))
#         return metallic, roughness, albedo
#
#     def distribution_ggx(self, NoH, roughness):
#         a = roughness
#         a2 = a ** 2
#         NoH2 = NoH ** 2
#         denom = NoH2 * (a2 - 1.0) + 1.0
#         return a2 / (np.pi * denom ** 2 + 1e-4)
#
#     def geometry(self, NoV, NoL, roughness):
#         if self.cfg['geometry_type'] == 'schlick':
#             geometry = self.geometry_schlick(NoV, NoL, roughness)
#         elif self.cfg['geometry_type'] == 'ggx_smith':
#             geometry = self.geometry_ggx_smith_correlated(NoV, NoL, roughness)
#         else:
#             raise NotImplementedError
#         return geometry
#
#     def _init_nerfactor_embedder(self):
#         # Read configuration values
#         pos_enc = self.nerfactor_config.getboolean('DEFAULT', 'pos_enc')
#         n_freqs_xyz = self.nerfactor_config.getint('DEFAULT', 'n_freqs_xyz')
#         n_freqs_ldir = self.nerfactor_config.getint('DEFAULT', 'n_freqs_ldir')
#         n_freqs_vdir = self.nerfactor_config.getint('DEFAULT', 'n_freqs_vdir')
#
#         # Shortcircuit if not using embedders
#         if not pos_enc:
#             embedder = {
#                 'xyz': torch.identity, 'ldir': torch.identity, 'vdir': torch.identity
#             }
#             return embedder
#
#         # Position embedder
#         kwargs = {
#             'incl_input': True,
#             'in_dims': 3,
#             'log2_max_freq': n_freqs_xyz - 1,
#             'n_freqs': n_freqs_xyz,
#             'log_sampling': True,
#             'periodic_func': [torch.sin, torch.cos]
#         }
#         embedder_xyz = nerfactor_Embedder(**kwargs)
#
#         # Light direction embedder
#         kwargs['log2_max_freq'] = n_freqs_ldir - 1
#         kwargs['n_freqs'] = n_freqs_ldir
#         embedder_ldir = nerfactor_Embedder(**kwargs)
#
#         # View direction embedder
#         kwargs['log2_max_freq'] = n_freqs_vdir - 1
#         kwargs['n_freqs'] = n_freqs_vdir
#         embedder_vdir = nerfactor_Embedder(**kwargs)
#
#         # Combine embedders
#         embedder = {
#             'xyz': embedder_xyz, 'ldir': embedder_ldir, 'vdir': embedder_vdir
#         }
#
#         pos_enc = self.nerfactor_config.getboolean('DEFAULT', 'pos_enc')
#         n_freqs_rusink = self.nerfactor_config_brdf.getint('DEFAULT', 'n_freqs')
#
#         if not pos_enc:
#             embedder['rusink'] = lambda x: x  # Identity function
#             return embedder
#
#         # Parameters for the rusink embedder
#         kwargs = {
#             'incl_input': True,
#             'in_dims': 3,
#             'log2_max_freq': n_freqs_rusink - 1,
#             'n_freqs': n_freqs_rusink,
#             'log_sampling': True,
#             'periodic_func': [torch.sin, torch.cos]
#         }
#         embedder_rusink = nerfactor_Embedder(**kwargs)
#         embedder['rusink'] = embedder_rusink
#         return embedder
#
#     def _pred_brdf_at(self, pts):
#         """
#         Predicts a latent BRDF (represented as a “z” code) at the given 3D positions.
#         """
#         mlp_layers = self.nerfactor_net['brdf_z_mlp'].cuda()
#         out_layer = self.nerfactor_net['brdf_z_out'].cuda()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         embedder = self.nerfactor_embedder['xyz']
#         # Scale points appropriately (user is unaware of the scaling)
#         pts_scaled = self.nerfactor_xyz_scale * pts
#
#         def chunk_func(surf):
#             # Embed the input positions, run them through an MLP and then the output layer.
#             surf_embed = embedder(surf)
#             brdf_z = out_layer(mlp_layers(surf_embed))
#             return brdf_z
#
#         # Apply the network in chunks (to avoid memory issues) along the last dimension.
#         brdf_z = chunk_func(pts_scaled)
#         return brdf_z  # Tensor of shape (N, z_dim)
#
#     def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, brdf_prop):
#         """
#         Evaluates the BRDF (as a combination of a Lambertian and a learned specular
#         component) at the given view and light directions.
#
#         Arguments:
#           pts2l: Tensor of shape (N, L, 3) giving directions from each surface point to each light.
#           pts2c: Tensor of shape (N, 3) giving directions from each surface point to the camera.
#           normal: Tensor of shape (N, 3) with world-space normals.
#           albedo: Tensor of shape (N, 3) containing the Lambertian (diffuse) albedo.
#           brdf_prop: Tensor of shape (N, z_dim) (the learned BRDF properties).
#
#         Returns:
#           A tensor of shape (N, L, 3) with the final BRDF evaluated for each ray–light pair.
#         """
#
#         # Get a learned BRDF scaling factor from configuration.
#         brdf_scale = self.nerfactor_config.getfloat('DEFAULT', 'learned_brdf_scale')
#         z = brdf_prop
#
#         # Get the rotation matrices to map world-space normals into a local frame.
#         world2local = geomutil.gen_world2local(normal)  # shape: (N, 3, 3)
#
#         # Transform directions into local frames.
#         # Here we use torch.einsum to mimic TensorFlow’s einsum behavior.
#         # (pts2c: (N, 3); world2local: (N, 3, 3) -> vdir: (N, 3))
#         vdir = torch.einsum('jkl,jl->jk', world2local, pts2c)
#         # (pts2l: (N, L, 3); world2local: (N, 3, 3) -> ldir: (N, L, 3))
#         ldir = torch.einsum('jkl,jnl->jnk', world2local, pts2l)
#
#         # Flatten light directions for per-direction processing.
#         ldir_flat = ldir.reshape(-1, 3)
#         # For the view direction, repeat each ray’s view vector for all L light directions.
#         vdir_rep = vdir.unsqueeze(1).repeat(1, ldir.shape[1], 1)
#         vdir_flat = vdir_rep.reshape(-1, 3)
#
#         # Convert the pair of (local) directions into Rusink coordinates.
#         rusink = geomutil.dir2rusink(ldir_flat, vdir_flat)  # shape: (N*L, 3)
#
#         # Repeat the BRDF “z” property for each light direction.
#         z_rep = z.unsqueeze(1).repeat(1, ldir.shape[1], 1)
#         z_flat = z_rep.reshape(-1, self.nerfactor_z_dim)
#
#         # Mask out back‐lit directions.
#         # Create a local “up” vector. (Ensure it’s on the same device as ldir.)
#         local_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=ldir.device).reshape(3, 1)
#         # Compute the cosine between each light direction and the local normal.
#         cos = ldir_flat @ local_normal  # shape: (N*L, 1)
#         front_lit = cos.reshape(-1) > 0  # Boolean mask: True for front-lit directions.
#         rusink_fl = rusink[front_lit]
#         z_fl = z_flat[front_lit]
#
#         # Predict the specular (BRDF) response in the Rusink coordinate space.
#         mlp_layers = self.nerfactor_brdf_model.net['brdf_mlp']
#         out_layer = self.nerfactor_brdf_model.net['brdf_out']
#         embedder = self.nerfactor_embedder['rusink']
#
#         def chunk_func(rusink_z):
#             # Split the input into Rusink coordinates and the corresponding BRDF z.
#             rusink_val = rusink_z[:, :3]
#             z_val = rusink_z[:, 3:]
#             # Embed the Rusink coordinates.
#             rusink_embed = embedder(rusink_val)
#             # Concatenate the latent BRDF z and the Rusink embedding.
#             z_rusink = torch.cat((z_val, rusink_embed), dim=1)
#             z_rusink_tf = tf.convert_to_tensor(z_rusink.detach().cpu().numpy(), dtype=tf.float32)
#             # (Optionally, you may check that z_rusink.shape[1] == self.z_dim + embedder.out_dims)
#             brdf = out_layer(mlp_layers(z_rusink_tf))
#             return torch.tensor(brdf.numpy()).cuda()
#
#         # Concatenate the Rusink coordinates and BRDF z values.
#         rusink_z = torch.cat((rusink_fl, z_fl), dim=1)
#         brdf_fl = chunk_func(rusink_z)  # shape: (N_front, 1)
#
#         # Scatter the computed front-lit BRDF values back into a tensor of shape (N*L, 1).
#         # (In PyTorch, boolean indexing can be used to assign into a pre-created tensor.)
#         brdf_flat = torch.zeros((ldir_flat.shape[0], 1), dtype=brdf_fl.dtype, device=ldir_flat.device)
#         brdf_flat[front_lit] = brdf_fl
#
#         # Reshape the flat tensor back into (N, L, 1) and expand to 3 channels (achromatic specular).
#         spec = brdf_flat.reshape(ldir.shape[0], ldir.shape[1], 1)
#         spec = spec.expand(-1, -1, 3)
#
#         # Combine the diffuse (Lambertian) and specular components.
#         # (albedo: (N, 3) → unsqueeze to (N, 1, 3) for broadcasting over L.)
#         brdf = spec * brdf_scale
#         return brdf  # Tensor of shape (N, L, 3)
#
#     def sample_diffuse_seperate(self, normals, num_samples):
#         # Cosine-weighted sampling (as before)
#         normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
#         B = normals.shape[0]
#         device = normals.device
#         u1 = torch.rand(B, num_samples, device=device)
#         u2 = torch.rand(B, num_samples, device=device)
#         r = torch.sqrt(u1)
#         theta = 2 * np.pi * u2
#         x = r * torch.cos(theta)
#         y = r * torch.sin(theta)
#         z = torch.sqrt(torch.clamp(1 - u1, min=0.0))
#         local_dirs = torch.stack([x, y, z], dim=-1)
#         # Transform from local (with +Z aligned with normal) to world space
#         world_dirs = self.transform_local_to_world(normals, local_dirs)
#         pdf_diff = z / np.pi  # cosθ/π
#         return world_dirs, pdf_diff
#
#     def sample_specular_seperate(self, normals, view_dirs, exponent, num_samples):
#         """
#         Sample specular directions according to a Blinn-Phong like distribution.
#         `exponent` is the lobe exponent.
#         """
#         normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
#         B = normals.shape[0]
#         device = normals.device
#         u1 = torch.rand(B, num_samples, device=device)
#         u2 = torch.rand(B, num_samples, device=device)
#         # Sample half-vectors in local space:
#         # theta_h ~ arccos(u1^(1/(exponent+1))) and phi_h ~ 2*pi*u2.
#         theta_h = torch.acos(torch.clamp(u1, max=1.0) ** (1 / (exponent + 1)))
#         phi_h = 2 * np.pi * u2
#         sin_theta_h = torch.sin(theta_h)
#         cos_theta_h = torch.cos(theta_h)
#         # Half-vectors in local coordinate system:
#         h_local = torch.stack([
#             sin_theta_h * torch.cos(phi_h),
#             sin_theta_h * torch.sin(phi_h),
#             cos_theta_h
#         ], dim=-1)
#         # Transform half-vector to world space:
#         h_world = self.transform_local_to_world(normals, h_local)
#         # Reflect view direction around half-vector:
#         # reflection: ω_i = 2 (v·h) h - v
#         v_dot_h = (view_dirs.unsqueeze(1) * h_world).sum(dim=-1, keepdim=True)
#         spec_dirs = 2 * v_dot_h * h_world - view_dirs.unsqueeze(1)
#         # Compute PDF for half-vector sampling
#         # PDF_h = (exponent + 1) / (2π) * (cosθ_h)^(exponent)
#         pdf_h = (exponent + 1) / (2 * np.pi) * (cos_theta_h ** exponent)
#         # Jacobian for the half-vector to direction mapping:
#         # pdf_spec = pdf_h / (4 * |v · h|)
#         eps = 1e-5  # or a small value that makes sense in your context
#         pdf_spec = pdf_h / (4 * (torch.abs(v_dot_h.squeeze(-1)) + eps))
#         return spec_dirs, pdf_spec
#
#     def transform_local_to_world(self, normals, local_dirs):
#         """
#         Build an orthonormal basis for each normal and transform local_dirs (BxN x 3) to world space.
#         """
#         # For each normal, compute tangent and bitangent.
#         B = normals.shape[0]
#         device = normals.device
#         # Use a similar technique as before:
#         # Choose an up vector that is not colinear.
#         up = torch.where(torch.abs(normals[:, 2:3]) < 0.999,
#                          torch.tensor([0, 0, 1], dtype=torch.float32, device=device),
#                          torch.tensor([1, 0, 0], dtype=torch.float32, device=device))
#         tangent = torch.nn.functional.normalize(torch.cross(up, normals, dim=1), dim=1)
#         bitangent = torch.cross(normals, tangent, dim=1)
#         # Expand basis vectors for broadcasting with samples.
#         tangent = tangent.unsqueeze(1)  # (B, 1, 3)
#         bitangent = bitangent.unsqueeze(1)
#         normals_exp = normals.unsqueeze(1)
#         # Combine local directions: local_dirs assumed shape (B, num_samples, 3)
#         world_dirs = local_dirs[..., 0:1] * tangent + local_dirs[..., 1:2] * bitangent + local_dirs[...,
#                                                                                          2:3] * normals_exp
#         world_dirs = mathutil.safe_l2_normalize(world_dirs, axis=-1)
#         return world_dirs
#
#     def shade_mixed(self, pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses, is_train):
#
#         # sample specular directions
#         # specular_directions, spec_pdfs = self.sample_specular_seperate(normals, view_dirs, 30, self.cfg['specular_sample_num'])
#         specular_directions, spec_pdfs = self.sample_cosine_weighted_rays(normals, self.cfg['specular_sample_num'])
#         specular_num = specular_directions.shape[1]
#
#         # sample diffuse directions
#         # diffuse_directions, diff_pdfs = self.sample_diffuse_seperate(normals, self.cfg['diffuse_sample_num'])  # [pn,sn0,3]
#         diff_pdfs = spec_pdfs
#         diffuse_directions = specular_directions
#         point_num, diffuse_num, _ = diffuse_directions.shape
#
#         # combine
#         # directions = torch.cat([diffuse_directions, specular_directions], 1)
#         # sn = diffuse_num + specular_num
#         directions = specular_directions
#         sn = specular_num
#
#         # specular
#         human_poses = human_poses.unsqueeze(1).repeat(1, sn, 1, 1) if human_poses is not None else None
#         pts_ = pts.unsqueeze(1).repeat(1, sn, 1)
#         lights, hl, light_pts, light_normals, light_pts_mask = self.get_lights(pts_, directions, human_poses)  # pn,sn,3
#         # specular_lights = lights[:, diffuse_num:]
#         specular_lights = lights
#         # Change here for using /nerfactor BRDF model
#
#         brdf_prop = self._pred_brdf_at(pts)
#         brdf_prop_jitter = None
#         if self.nerfactor_normalize_brdf_z:
#             brdf_prop = mathutil.safe_l2_normalize(brdf_prop, axis=1)
#             if brdf_prop_jitter is not None:
#                 brdf_prop_jitter = mathutil.safe_l2_normalize(
#                     brdf_prop_jitter, axis=1)
#         surf2l = mathutil.safe_l2_normalize(specular_directions, axis=-1)
#         surf2c = mathutil.safe_l2_normalize(-view_dirs, axis=-1)
#         brdf_normals = mathutil.safe_l2_normalize(normals, axis=-1)
#         albedo_nerfacor = albedo * 0.77 + 0.03
#         spec_brdf = self._eval_brdf_at(
#             surf2l, surf2c, brdf_normals, albedo_nerfacor, brdf_prop)  # NxLx3
#
#         cos_theta_spec = torch.clamp(
#             (specular_directions * brdf_normals.unsqueeze(1)).sum(dim=-1), min=0.0)  # (B, n_samples)
#         # cos_theta_diff = torch.clamp(
#         #     (diffuse_directions * brdf_normals.unsqueeze(1)).sum(dim=-1), min=0.0)  # (B, n_samples)
#         cos_theta_diff = cos_theta_spec
#
#         cos_theta_spec = cos_theta_spec.unsqueeze(-1)
#         cos_theta_diff = cos_theta_diff.unsqueeze(-1)
#         spec_pdfs = spec_pdfs.unsqueeze(-1)
#         diff_pdfs = diff_pdfs.unsqueeze(-1)
#
#         if torch.isnan(specular_directions).any():
#             print('nan in specular_directions')
#         if torch.isinf(specular_directions).any():
#             print('inf in specular_directions')
#
#         if torch.isnan(spec_brdf).any():
#             print('nan in spec_brdf')
#         if torch.isinf(spec_brdf).any():
#             print('inf in spec_brdf')
#
#         if torch.isnan(specular_lights).any():
#             print('nan in specular_lights')
#         if torch.isinf(specular_lights).any():
#             print('inf in specular_lights')
#
#         if torch.isnan(cos_theta_spec).any():
#             print('nan in cos_theta')
#         if torch.isinf(cos_theta_spec).any():
#             print('inf in cos_theta')
#
#         if torch.isnan(spec_pdfs).any():
#             print('nan in pdfs')
#         if torch.isinf(spec_pdfs).any():
#             print('inf in pdfs')
#
#         specular_colors = torch.mean(spec_brdf * specular_lights * cos_theta_spec / (0.001 + spec_pdfs), 1)
#         spec_brdf_avg = torch.mean(spec_brdf, 1)
#
#         # diffuse_lights = lights[:, :diffuse_num]
#         diffuse_lights = lights
#         #
#
#         diffuse_colors = torch.mean(albedo_nerfacor.unsqueeze(1) / np.pi * diffuse_lights * cos_theta_diff / (0.001+diff_pdfs), 1)
#
#         colors = diffuse_colors + specular_colors
#         colors = torch.clamp(colors, min=0.0, max=1.0)
#         colors = linear_to_srgb(colors)
#
#         outputs = {}
#         outputs['albedo'] = albedo
#         outputs['spec_brdf'] = spec_brdf
#
#         outputs['human_lights'] = hl.reshape(-1, 3)
#         outputs['diffuse_light'] = torch.clamp(linear_to_srgb(torch.mean(diffuse_lights, dim=1)), min=0, max=1)
#         outputs['specular_light'] = torch.clamp(linear_to_srgb(torch.mean(specular_lights, dim=1)), min=0, max=1)
#         diffuse_colors = torch.clamp(linear_to_srgb(diffuse_colors), min=0, max=1)
#         specular_colors = torch.clamp(linear_to_srgb(specular_colors), min=0, max=1)
#         outputs['diffuse_color'] = diffuse_colors
#         outputs['specular_color'] = specular_colors
#         outputs['spec_brdf'] = spec_brdf_avg
#
#         return colors, outputs
#
#     def forward(self, pts, view_dirs, normals, human_poses, step, is_train):
#         view_dirs, normals = F.normalize(view_dirs, dim=-1), F.normalize(normals, dim=-1)
#         reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
#         metallic, roughness, albedo = self.predict_materials(pts)  # [pn,1] [pn,1] [pn,3]
#         return self.shade_mixed(pts, normals, view_dirs, reflections, metallic, roughness, albedo, human_poses,
#                                 is_train)
#
#     def env_light(self, h, w, gamma=True):
#         azs = torch.linspace(1.0, 0.0, w) * np.pi * 2 - np.pi / 2
#         els = torch.linspace(1.0, -1.0, h) * np.pi / 2
#
#         els, azs = torch.meshgrid(els, azs)
#         if self.cfg['is_real']:
#             x = torch.cos(els) * torch.cos(azs)
#             y = torch.cos(els) * torch.sin(azs)
#             z = torch.sin(els)
#         else:
#             z = torch.cos(els) * torch.cos(azs)
#             x = torch.cos(els) * torch.sin(azs)
#             y = torch.sin(els)
#         xyzs = torch.stack([x, y, z], -1)  # h,w,3
#         xyzs = xyzs.reshape(h * w, 3)
#         # xyzs = xyzs @ torch.from_numpy(np.asarray([[0,0,1],[0,1,0],[-1,0,0]],np.float32)).cuda()
#
#         batch_size = 8192
#         lights = []
#         for ri in range(0, h * w, batch_size):
#             with torch.no_grad():
#                 light = self.predict_outer_lights_pts(xyzs[ri:ri + batch_size])
#             lights.append(light)
#         if gamma:
#             lights = linear_to_srgb(torch.cat(lights, 0)).reshape(h, w, 3)
#         else:
#             lights = (torch.cat(lights, 0)).reshape(h, w, 3)
#         return lights
#
#     def predict_outer_lights_pts(self, pts):
#         if self.cfg['outer_light_version'] == 'direction':
#             return self.outer_light(self.sph_enc(pts, 0))
#         elif self.cfg['outer_light_version'] == 'sphere_direction':
#             return self.outer_light(torch.cat([self.sph_enc(pts, 0), self.sph_enc(pts, 0)], -1))
#         else:
#             raise NotImplementedError
#
#     def get_env_light(self):
#         return self.predict_outer_lights_pts(self.light_pts)
#
#     def material_regularization(self, pts, normals, albedo, step):
#         # metallic, roughness, albedo = self.predict_materials(pts)
#         reg = 0
#
#         if self.cfg['reg_change']:
#             normals = F.normalize(normals, dim=-1)
#             x = self.get_orthogonal_directions(normals)
#             y = torch.cross(normals, x)
#             ang = torch.rand(pts.shape[0], 1) * np.pi * 2
#             if self.cfg['change_type'] == 'constant':
#                 change = (torch.cos(ang) * x + torch.sin(ang) * y) * self.cfg['change_eps']
#             elif self.cfg['change_type'] == 'gaussian':
#                 eps = torch.normal(mean=0.0, std=self.cfg['change_eps'], size=[x.shape[0], 1])
#                 change = (torch.cos(ang) * x + torch.sin(ang) * y) * eps
#             else:
#                 raise NotImplementedError
#             m0, r0, a0 = self.predict_materials(pts + change)
#
#             # reg = reg + torch.mean(
#             #     (torch.abs(m0 - metallic) + torch.abs(r0 - roughness) + torch.abs(a0 - albedo)) * self.cfg[
#             #         'reg_lambda1'], dim=1)
#             brdf_prop_jitter = self._pred_brdf_at(pts + change)
#             brdf_prop_pred = self._pred_brdf_at(pts)
#             brdf_smooth_loss = self.nerfactor_smooth_loss(brdf_prop_pred, brdf_prop_jitter)  # N
#             # todo modify nerfactor_brdf_smooth_weight based on loss function
#             reg += self.nerfactor_brdf_smooth_weight * brdf_smooth_loss
#
#             reg = reg + torch.mean(
#                 torch.abs(a0 - albedo) * self.nerfactor_albedo_smooth_weight, dim=1)
#
#         # if self.cfg['reg_min_max'] and step is not None and step < 2000:
#         #     # sometimes the roughness and metallic saturate with the sigmoid activation in the early stage
#         #     reg = reg + torch.sum(torch.clamp(roughness - 0.98 ** 2, min=0))
#         #     reg = reg + torch.sum(torch.clamp(0.02 ** 2 - roughness, min=0))
#         #     reg = reg + torch.sum(torch.clamp(metallic - 0.98, min=0))
#         #     reg = reg + torch.sum(torch.clamp(0.02 - metallic, min=0))
#
#         return reg
#
#
# def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
#     N = batch_size
#     X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
#     Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
#     Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
#
#     u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
#     with torch.no_grad():
#         for xi, xs in enumerate(X):
#             for yi, ys in enumerate(Y):
#                 for zi, zs in enumerate(Z):
#                     xx, yy, zz = torch.meshgrid(xs, ys, zs)
#                     pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
#                     val = query_func(pts).detach()
#                     outside_mask = torch.norm(pts, dim=-1) >= 1.0
#                     val[outside_mask] = outside_val
#                     val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
#                     u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
#     return u
#
#
# def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, outside_val=1.0):
#     u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
#     vertices, triangles = mcubes.marching_cubes(u, threshold)
#     b_max_np = bound_max.detach().cpu().numpy()
#     b_min_np = bound_min.detach().cpu().numpy()
#
#     vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
#     return vertices, triangles

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
    return expected_sin(torch.concat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1),
                        torch.concat([scaled_var] * 2, dim=-1))


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
        self.roughness_predictor = make_predictor(feats_dim + 3, 1)
        if self.cfg['roughness_init'] != 0:
            nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
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
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))

        # inner lights are indirect lights
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
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

    def predict_specular_lights(self, points, feature_vectors, reflective, roughness, human_poses, step):
        human_light, human_weight = 0, 0
        ref_roughness = self.sph_enc(reflective, roughness)
        pts = self.pos_enc(points)
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)
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
        roughness = torch.ones([normals.shape[0], 1])
        ref = self.sph_enc(normals, roughness)  # von Mises-Fisher distribution
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)
            sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1)
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
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))

        # diffuse light
        diffuse_albedo = (1 - metallic) * albedo
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        diffuse_color = diffuse_albedo * diffuse_light

        # specular light
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        specular_light, occ_prob, indirect_light, human_light = self.predict_specular_lights(points, feature_vectors,
                                                                                             reflective, roughness,
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
            return color, occ_info

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
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
        'reg_lambda1': 0.005,
        'reg_min_max': True,

        'random_azimuth': True,
        'is_real': False,
        'anisotropy': True,
        'max_n_exp': 20,
        'max_alpha_exp': 10
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
            # self.my_predictor = make_predictor(256 + 3, 1, activation='exp', exp_max=self.cfg['max_n_exp'])
            self.mx_predictor = make_predictor(256 + 3, 1)
            self.my_predictor = make_predictor(256 + 3, 1)
            self.alpha_predictor = make_predictor(256 + 3, 1)
            self.F0_predictor = make_predictor(256 + 3, 1)
            self.kd_predictor = make_predictor(256 + 3, 3)
            self.ks_predictor = make_predictor(256 + 3, 3)

        # light part
        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(8, 3)
        if self.cfg['outer_light_version'] == 'direction':
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            self.outer_light = make_predictor(72 * 2, 3, activation='exp', exp_max=self.cfg['light_exp_max'])
        else:
            raise NotImplementedError
        nn.init.constant_(self.outer_light[-2].bias, np.log(0.5))
        if self.cfg['human_lights']:
            self.human_light = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light[-2].bias, np.log(0.02))
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=self.cfg['inner_light_exp_max'])
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
        normals = F.normalize(normals, dim=-1)
        view_dirs = F.normalize(view_dirs, dim=-1)
        reflections = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        dir_enc = self.sph_enc(reflections, 0)
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
        if self.cfg['outer_light_version'] == 'direction':
            outer_enc = self.sph_enc(directions, 0)
            outer_lights = self.outer_light(outer_enc)
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            outer_dirs = directions
            outer_pts = points
            outer_enc = self.sph_enc(outer_dirs, 0)
            mask = torch.norm(outer_pts, dim=-1) > 0.999
            if torch.sum(mask) > 0:
                outer_pts = torch.clone(outer_pts)
                outer_pts[mask] *= 0.999  # shrink this point a little bit
            dists = get_sphere_intersection(outer_pts, outer_dirs)
            sphere_pts = outer_pts + outer_dirs * dists
            sphere_pts = self.sph_enc(sphere_pts, 0)
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
        H = F.normalize(H, dim=-1)
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
        H_s = F.normalize(H_s, dim=-1)
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
        mx = self.mx_predictor(torch.cat([feats, pts], -1)) + 1e-5
        my = self.my_predictor(torch.cat([feats, pts], -1)) + 1e-5
        alpha = self.alpha_predictor(torch.cat([feats, pts], -1))
        F0 = self.F0_predictor(torch.cat([feats, pts], -1))
        kd = self.kd_predictor(torch.cat([feats, pts], -1))
        ks = self.ks_predictor(torch.cat([feats, pts], -1))
        print('mx    range:', mx.min(), '-', mx.max())
        print('my    range:', my.min(), '-', my.max())
        print('alpha range:', alpha.min(), '-', alpha.max())
        return mx, my, alpha, F0, kd, ks



    def sample_aniso_ggx_directions(self,
                                    mesh,
                                    tangents,
                                    bitangents,
                                    tree,
                                    pts: torch.Tensor,
                                    m_x: torch.Tensor,
                                    m_y: torch.Tensor,
                                    wo: torch.Tensor,
                                    M: int,
                                    normals: torch.Tensor,
                                    device: torch.device = None,
                                    eps: float = 1e-6):
        if device is None:
            device = wo.device
        T = tangents
        B = bitangents
        # z = normals  # pn,3
        # x = self.get_orthogonal_directions(normals)  # pn,3
        # y = torch.cross(z, x, dim=-1)  # pn,3
        vertices = torch.from_numpy(np.asarray(mesh.vertices))
        faces = torch.from_numpy(np.asarray(mesh.triangles))
        x, y = self.get_tangent_bitangent_via_kdtree(vertices, faces, T, B, pts, normals, tree)
        z = normals

        m_x = m_x.to(device)  # (N,1)
        m_y = m_y.to(device)  # (N,1)
        wo = wo.to(device)  # (N,3)
        N = wo.shape[0]

        # 1) Uniformsk
        xi1 = torch.rand((N, M), device=device).clamp(min=eps)
        xi2 = torch.rand((N, M), device=device)

        two_pi_xi2 = 2.0 * np.pi * xi2  # (N,M)

        # 2) Azimuth via atan2
        phi_h = torch.atan2(m_y*torch.tan(two_pi_xi2), m_x)  # <-- full-range

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

        return h, wi, cos_theta_h

    def compute_radiance(self,
                         f_d: torch.Tensor,
                         diffuse_lights: torch.Tensor,
                         specular_lights: torch.Tensor,
                         k_s: torch.Tensor,
                         F: torch.Tensor,
                         diffuse_directions: torch.Tensor,
                         wi: torch.Tensor,
                         n: torch.Tensor,
                         alpha: torch.Tensor,
                         cos_theta_h: torch.Tensor,
                         wo: torch.Tensor,
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
        k_s : Tensor (N, 3)
            Specular reflectivity per point.
        F : Tensor (N, M, 1)
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
        N_spec, M_spec, _ = specular_lights.shape
        N_diff, M_diff, _ = diffuse_lights.shape

        #    shape (N, M)
        mask = ((wi * n.unsqueeze(1)).sum(dim=-1) > 0.0)
        valid_counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

        # Expand k_s, alpha, wo·n to match (N, M, *)
        k_s_exp = k_s.unsqueeze(1)  # (N, 1, 3)
        alpha_exp = alpha.unsqueeze(1)  # (N, 1, 1)
        cos_on = torch.clamp((wo * n).sum(dim=1, keepdim=True), min=0.0)  # (N,1)
        cos_on_exp = cos_on.unsqueeze(1)  # (N,1,1)

        # Dot product wi·n
        cos_in = torch.clamp((wi * n.unsqueeze(1)).sum(dim=2, keepdim=True), min=0.0)  # (N,M,1)

        # --- Diffuse component: (1/M) ∑ f_d * Li ---
        diff_weighted = f_d * diffuse_lights # (N,M,3)
        diffuse = diff_weighted.sum(dim=1) / M_diff  # (N,3)

        # --- Specular component ---
        # term1 = L * k_s * F
        spec_base = specular_lights * k_s_exp * F  # (N,M,3)

        # term2 = (wi·n)^(1-alpha)
        exponent = 1.0 - alpha_exp  # (N,1,1)

        pow_in = torch.pow(cos_in + eps, exponent)  # (N,M,1)
        # if cos_in.min() <= 0:
        #     print('cos wi.n min:', cos_in.min(), torch.pow(cos_in + eps, exponent))

        # denom = cos_theta_h * (wo·n)^alpha
        pow_on = torch.pow(cos_on_exp + eps, alpha_exp)  # (N,1,1)
        denom = cos_theta_h * pow_on + eps  # (N,M,1)

        spec_weighted = (spec_base * pow_in / denom) * mask.unsqueeze(-1)  # (N,M,3)
        specular = spec_weighted.sum(dim=1) / valid_counts  # (N,3)

        # Total radiance
        R = diffuse + specular  # (N,3)
        return R, diffuse, specular

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
                     F: torch.Tensor,
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
        # Ensure shapes
        # kd: (N,3) -> (N,1,3)
        kd_exp = kd.unsqueeze(1)  # (N,1,3)

        # (1 - F): shape (N, M, 1)
        one_minus_F = 1.0 - F  # broadcastable to (N,M,3)

        # f_d = (k_d / π) * (1 - F)
        if is_seperate:
            f_d = kd_exp / np.pi  # (N,M,3)
        else:
            f_d = kd_exp / np.pi * one_minus_F  # (N,M,3)

        return f_d

    def find_triangles_and_barycentrics_kdtree(
            self,
            verts: torch.Tensor,
            faces: torch.LongTensor,
            pts: torch.Tensor,
            tree: cKDTree,
            k: int = 5,
    ):
        """
        For each point in pts, search k nearest triangle centroids,
        test barycentric coordinates, and return the containing triangle.

        Args:
          verts: (V,3)
          faces: (F,3)
          pts: (N,3)
          tree: cKDTree over centroids
          centroids: (F,3) numpy array
          k: number of candidate triangles

        Returns:
          tri_indices: (N,) LongTensor of face IDs
          bary_coords: (N,3) tensor of barycentric coords
        """
        device = pts.device
        pts_np = pts.cpu().numpy()  # (N,3)
        _, idxs = tree.query(pts_np, k=k)  # (N,k)

        tri_indices = []
        bary_coords = []
        for pi, candidates in enumerate(idxs):
            p = pts[pi]
            for fid in candidates:
                tri = verts[faces[fid]]  # (3,3)
                v0, v1 = tri[1] - tri[0], tri[2] - tri[0]
                v2 = p - tri[0]
                d00 = torch.dot(v0, v0);
                d01 = torch.dot(v0, v1)
                d11 = torch.dot(v1, v1);
                d20 = torch.dot(v2, v0)
                d21 = torch.dot(v2, v1)
                denom = d00 * d11 - d01 * d01 + 1e-8
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1 - v - w
                bary = torch.stack([u, v, w])
                if (bary >= -1e-4).all() and (bary <= 1 + 1e-4).all():
                    tri_indices.append(int(fid))
                    bary_coords.append(bary)
                    break

        tri_indices = torch.LongTensor(tri_indices).to(device)
        bary_coords = torch.stack(bary_coords, dim=0).to(device)
        return tri_indices, bary_coords

    def get_tangent_bitangent_via_kdtree(
            self,
            verts: torch.Tensor,
            faces: torch.LongTensor,
            vert_tangents: torch.Tensor,
            vert_bitangents: torch.Tensor,
            pts: torch.Tensor,
            query_normals: torch.Tensor,
            tree: cKDTree,
            k: int = 5,
    ):
        """
        Full pipeline: KD-tree lookup → barycentric coords → frame interpolation.

        Returns:
          (tangents, bitangents): each (N,3)
        """
        # 1) find face IDs and barycentrics via KD-tree
        tri_idx, baryc = self.find_triangles_and_barycentrics_kdtree(
            verts, faces, pts, tree, k
        )

        # 2) interpolate & orthonormalize
        return self.interpolate_tangent_bitangent(
            vert_tangents, vert_bitangents, faces,
            tri_idx, baryc, query_normals
        )

    def interpolate_tangent_bitangent(
            self,
            vert_tangents: torch.Tensor,
            vert_bitangents: torch.Tensor,
            faces: torch.LongTensor,
            tri_indices: torch.LongTensor,
            barycentric_coords: torch.Tensor,
            query_normals: torch.Tensor,
    ):
        """
        vert_tangents:       (V, 3) per-vertex tangent vectors
        vert_bitangents:     (V, 3) per-vertex bitangent vectors
        faces:               (F, 3) triangle vertex indices
        tri_indices:         (N,) face index for each query point
        barycentric_coords:  (N, 3) barycentric coords per query point
        query_normals:       (N, 3) interpolated normals at query points

        Returns:
          (tangents, bitangents): each (N, 3)
        """
        # 1. Gather the three vertex indices for each sampled point
        face_verts = faces[tri_indices]  # shape: (N, 3)

        # 2. Pull per-vertex tangents & bitangents
        t0 = vert_tangents[face_verts[:, 0]]
        t1 = vert_tangents[face_verts[:, 1]]
        t2 = vert_tangents[face_verts[:, 2]]
        b0 = vert_bitangents[face_verts[:, 0]]
        b1 = vert_bitangents[face_verts[:, 1]]
        b2 = vert_bitangents[face_verts[:, 2]]

        # 3. Barycentric interpolation
        bc = barycentric_coords
        t_interp = t0 * bc[:, 0:1] + t1 * bc[:, 1:2] + t2 * bc[:, 2:3]
        b_interp = b0 * bc[:, 0:1] + b1 * bc[:, 1:2] + b2 * bc[:, 2:3]

        # 4. Orthonormalize the tangent relative to the normal
        n = query_normals
        t_proj = t_interp - n * torch.sum(n * t_interp, dim=1, keepdim=True)
        t_norm = F.normalize(t_proj, eps=1e-6, dim=1)

        # 5. Orthonormalize the bitangent relative to both normal & tangent
        b_proj = b_interp
        b_proj = b_proj - n * torch.sum(n * b_proj, dim=1, keepdim=True)
        b_proj = b_proj - t_norm * torch.sum(t_norm * b_proj, dim=1, keepdim=True)
        b_norm = F.normalize(b_proj, eps=1e-6, dim=1)

        return t_norm, b_norm

    def shade_anisotropic_mixed(self, mesh, tangents, bitangents, tree, pts, normals, view_dirs, mx, my, alpha, F0, kd, ks, human_poses, is_train, is_seperate=True):
        self.nan_inf_check(normals, 'normals')
        self.nan_inf_check(mx, 'mx')
        self.nan_inf_check(my, 'my')
        self.nan_inf_check(alpha, 'alpha')
        self.nan_inf_check(F0, 'F0')
        self.nan_inf_check(kd, 'kd')
        self.nan_inf_check(ks, 'ks')

        num_spec_samples = self.cfg['specular_sample_num']
        hs, wis, cos_ths = self.sample_aniso_ggx_directions(mesh, tangents, bitangents, tree, pts, mx, my, view_dirs, num_spec_samples, normals,'cuda')
        self.nan_inf_check(hs, 'hs')
        self.nan_inf_check(wis, 'wis')
        self.nan_inf_check(cos_ths, 'cos_ths')
        if is_seperate:
            diffuse_directions = self.sample_diffuse_directions(normals, is_train)
        else:
            diffuse_directions = wis
        point_num, diffuse_num, _ = diffuse_directions.shape
        self.nan_inf_check(diffuse_directions, 'diffuse_directions')

        pts_ = pts.unsqueeze(1).repeat(1, num_spec_samples+diffuse_num, 1)
        directions = torch.cat([diffuse_directions, wis], 1)
        lights, hl, light_pts, light_normals, light_pts_mask = self.get_lights(pts_, directions, human_poses)

        diffuse_lights = lights[:, :diffuse_num]
        specular_lights = lights[:, diffuse_num:]
        self.nan_inf_check(lights, 'lights')

        F = self.fresnel_schlick_batch(F0, view_dirs, hs)
        self.nan_inf_check(F, 'F (fresnel schlick)')

        f_d = self.diffuse_term(kd, F, is_seperate)
        self.nan_inf_check(f_d, 'diffuse term (f_d)')

        R, diffuse_color, specular_color  = self.compute_radiance(f_d, diffuse_lights, specular_lights, ks, F, diffuse_directions, wis, normals, alpha, cos_ths, view_dirs)
        self.nan_inf_check(R, 'R')
        self.nan_inf_check(diffuse_color, 'diffuse_color')
        self.nan_inf_check(specular_color, 'specular_color')

        colors = linear_to_srgb(R)
        self.nan_inf_check(colors, 'colors')

        diffuse_color = linear_to_srgb(diffuse_color)
        self.nan_inf_check(diffuse_color, 'diffuse_color_srgb')

        specular_color = linear_to_srgb(specular_color)
        self.nan_inf_check(specular_color, 'specular_color_srgb')

        outputs = {}
        outputs['human_lights'] = hl.reshape(-1, 3)
        outputs['kd'] = kd
        outputs['ks'] = ks
        outputs['F0'] = F0
        outputs['alpha'] = alpha
        outputs['mx'] = mx
        outputs['my'] = my
        outputs['diffuse_color'] = diffuse_color
        outputs['specular_color'] = specular_color
        outputs['diffuse_light'] = torch.clamp(linear_to_srgb(torch.mean(diffuse_lights, dim=1)), min=0, max=1)
        outputs['specular_light'] = torch.clamp(linear_to_srgb(torch.mean(specular_lights, dim=1)), min=0, max=1)
        return colors, outputs


    def anisotropic_forward(self, pts, view_dirs, normals, human_poses, step, is_train, mesh, tangents, bitangents, tree, is_seperate=True):
        # print('anisotropic_forward:')
        mx, my, alpha, F0, kd, ks = self.predict_anisotropic_components(pts)
        return self.shade_anisotropic_mixed(mesh, tangents, bitangents, tree, pts, normals, view_dirs, mx, my, alpha, F0, kd, ks, human_poses, is_train, is_seperate)


    def forward(self, pts, view_dirs, normals, human_poses, step, is_train, mesh, tangents, bitangents, tree):
        if mesh is not None:
            return self.anisotropic_forward(pts, view_dirs, normals, human_poses, step, is_train, mesh, tangents, bitangents, tree, is_seperate=True)
        view_dirs, normals = F.normalize(view_dirs, dim=-1), F.normalize(normals, dim=-1)
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
            return self.outer_light(self.sph_enc(pts, 0))
        elif self.cfg['outer_light_version'] == 'sphere_direction':
            return self.outer_light(torch.cat([self.sph_enc(pts, 0), self.sph_enc(pts, 0)], -1))
        else:
            raise NotImplementedError

    def get_env_light(self):
        return self.predict_outer_lights_pts(self.light_pts)

    def anisotropic_regularization(self, pts, normals, mx, my, alpha, F0, kd, ks):
        reg = 0
        print("I should not be here")
        if self.cfg['reg_change']:
            normals = F.normalize(normals, dim=-1)
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

            mx_ch, my_ch, alpha_ch, F0_ch, kd_ch, ks_ch = self.predict_anisotropic_components(pts + change)
            reg = reg + torch.mean(
                (torch.abs(kd - kd_ch) +
                 torch.abs(ks - ks_ch) +
                 torch.abs(mx - mx_ch) +
                 torch.abs(my - my_ch) +
                 torch.abs(alpha - alpha_ch) +
                 torch.abs(F0 - F0_ch)) *
                self.cfg['reg_lambda1'],
                        dim=1)
        return reg

    def material_regularization(self, pts, normals, metallic, roughness, albedo, step):
        # metallic, roughness, albedo = self.predict_materials(pts)
        reg = 0

        if self.cfg['reg_change']:
            normals = F.normalize(normals, dim=-1)
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