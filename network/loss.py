import numpy as np
import torch


class Loss:
    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass


class NeRFRenderLoss(Loss):
    default_cfg = {
        'render_loss_weight_begin': 1,
        'render_loss_weight_end': 5,
        'render_weight_decay_begin': 1000,
        'render_weight_decay_end': 5000,
    }
    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}



    def get_render_weight(self, step):
        nom = max(step - self.cfg['render_weight_decay_begin'], 0)
        mx = max(self.cfg['render_loss_weight_end'], self.cfg['render_loss_weight_begin'])
        mn = min(self.cfg['render_loss_weight_end'], self.cfg['render_loss_weight_begin'])
        denom = self.cfg['render_weight_decay_end'] - self.cfg['render_weight_decay_begin']
        nom = min(nom, denom)
        coef = mx - mn
        bias = mn
        rot = 0
        if self.cfg['render_loss_weight_end'] - self.cfg['render_loss_weight_begin'] > 0:
            rot = np.pi
        anneal_weights = (np.cos((nom / denom) * np.pi + rot) + 1) / 2

        return anneal_weights * coef + bias

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}

        if 'loss_rgb' in data_pr: outputs['loss_rgb'] = data_pr['loss_rgb'] * self.get_render_weight(step)
        if 'loss_rgb_fine' in data_pr: outputs['loss_rgb_fine'] = data_pr['loss_rgb_fine']
        if 'loss_global_rgb' in data_pr: outputs['loss_global_rgb'] = data_pr['loss_global_rgb']
        if 'loss_rgb_inner' in data_pr: outputs['loss_rgb_inner'] = data_pr['loss_rgb_inner']
        if 'loss_rgb0' in data_pr: outputs['loss_rgb0'] = data_pr['loss_rgb0']
        if 'loss_rgb1' in data_pr: outputs['loss_rgb1'] = data_pr['loss_rgb1']
        if 'loss_masks' in data_pr: outputs['loss_masks'] = data_pr['loss_masks']
        return outputs


class EikonalLoss(Loss):
    default_cfg = {
        "eikonal_weight": 0.1, # changed from 0.1
        'eikonal_weight_anneal_begin': 0,
        'eikonal_weight_anneal_end': 0,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def get_eikonal_weight(self, step):
        if step < self.cfg['eikonal_weight_anneal_begin']:
            return 0.0
        elif self.cfg['eikonal_weight_anneal_begin'] <= step < self.cfg['eikonal_weight_anneal_end']:
            return self.cfg['eikonal_weight'] * (step - self.cfg['eikonal_weight_anneal_begin']) / \
                (self.cfg['eikonal_weight_anneal_end'] - self.cfg['eikonal_weight_anneal_begin'])
        else:
            return self.cfg['eikonal_weight']

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        weight = self.get_eikonal_weight(step)
        outputs = {'loss_eikonal': data_pr['gradient_error'] * weight}
        return outputs


class MaterialRegLoss(Loss):
    default_cfg = {
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_mat_reg' in data_pr: outputs['loss_mat_reg'] = data_pr['loss_mat_reg']
        if 'loss_diffuse_light' in data_pr: outputs['loss_diffuse_light'] = data_pr['loss_diffuse_light']
        return outputs


class StdRecorder(Loss):
    default_cfg = {
        'apply_std_loss': False,
        'std_loss_weight': 0.05,
        'std_loss_weight_type': 'constant',
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'std' in data_pr:
            outputs['std'] = data_pr['std']
            if self.cfg['apply_std_loss']:
                if self.cfg['std_loss_weight_type'] == 'constant':
                    outputs['loss_std'] = data_pr['std'] * self.cfg['std_loss_weight']
                else:
                    raise NotImplementedError
        if 'inner_std' in data_pr: outputs['inner_std'] = data_pr['inner_std']
        if 'outer_std' in data_pr: outputs['outer_std'] = data_pr['outer_std']
        return outputs


class OccLoss(Loss):
    default_cfg = {
        'occ_loss_weight': 1,  # changed here from 0.01 to 1
        'occ_loss_weight_begin': 0.01,
        'occ_loss_weight_end': 1,
        'occ_weight_decay_begin': 20000,
        'occ_weight_decay_end': 50000,
    }

    def map_range_val(self, input_val, input_start, input_end, output_start, output_end):
        input_clamped = max(input_start, min(input_end, input_val))
        return output_start + ((output_end - output_start) / (input_end - input_start)) * (
                input_clamped - input_start
        )

    def get_occlusion_weight(self, step):
        # return self.cfg['occ_loss_weight']
        nom = max(0, step - self.cfg['occ_weight_decay_begin'])
        denom = self.cfg['occ_weight_decay_end'] - self.cfg['occ_weight_decay_begin']
        nom = min(nom, denom)
        coef = self.cfg['occ_loss_weight_end'] - self.cfg['occ_loss_weight_begin']
        bias = self.cfg['occ_loss_weight_begin']
        anneal_weights = (np.cos((nom / denom) * np.pi + np.pi) + 1) / 2 #[0-1]
        return anneal_weights * coef + bias # [begin-end]

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_occ' in data_pr:
            outputs['loss_occ'] = torch.mean(data_pr['loss_occ']).reshape(1) * self.get_occlusion_weight(step)
        return outputs


class InitSDFRegLoss(Loss):
    default_cfg = {
        'SDF_loss_weight': 5,
    }
    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        reg_step = 1000
        small_threshold = 0.1   
        large_threshold = 1.05
        if 'sdf_vals' in data_pr and 'sdf_pts' in data_pr and step < reg_step:
            norm = torch.norm(data_pr['sdf_pts'], dim=-1)
            sdf = data_pr['sdf_vals']
            small_mask = norm < small_threshold
            if torch.sum(small_mask) > 0:
                bounds = norm[small_mask] - small_threshold  # 0-small_threshold -> 0
                # we want sdf - bounds < 0
                small_loss = torch.mean(torch.clamp(sdf[small_mask] - bounds, min=0.0))
                small_loss = torch.sum(small_loss) / (torch.sum(small_loss > 1e-5) + 1e-3)
            else:
                small_loss = torch.zeros(1)

            large_mask = norm > large_threshold

            
            if torch.sum(large_mask) > 0:
                bounds = norm[large_mask] - large_threshold  # 0 -> 1 - large_threshold
                # we want sdf - bounds > 0 => bounds - sdf < 0
                large_loss = torch.clamp(bounds - sdf[large_mask], min=0.0)
                large_loss = torch.sum(large_loss) / (torch.sum(large_loss > 1e-5) + 1e-3)
            else:
                large_loss = torch.zeros(1)

            anneal_weights = (np.cos((step / reg_step) * np.pi) + 1) / 2
            anneal_weights = anneal_weights * self.cfg['SDF_loss_weight']
            return {'loss_sdf_large': large_loss * anneal_weights, 'loss_sdf_small': small_loss * anneal_weights}
        else:
            return {}


class MaskLoss(Loss):
    default_cfg = {
        'mask_loss_weight_begin': 0.3,
        'mask_loss_weight_end': 1,
        'mask_weight_decay_begin': 0,
        'mask_weight_decay_end': 30000,
    }

    def get_mask_weight(self, step):
        nom = max(step - self.cfg['mask_weight_decay_begin'], 0)
        mx = max(self.cfg['mask_loss_weight_end'], self.cfg['mask_loss_weight_begin'])
        mn = min(self.cfg['mask_loss_weight_end'], self.cfg['mask_loss_weight_begin'])
        denom = self.cfg['mask_weight_decay_end'] - self.cfg['mask_weight_decay_begin']
        nom = min(nom, denom)
        coef = mx - mn
        bias = mn
        rot = 0
        if self.cfg['mask_loss_weight_end'] - self.cfg['mask_loss_weight_begin'] > 0:
            rot = np.pi
        anneal_weights = (np.cos((nom / denom) * np.pi + rot) + 1) / 2

        return anneal_weights * coef + bias

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}


    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_mask' in data_pr and (step < self.cfg['mask_weight_decay_end'] or self.cfg['mask_loss_weight_end'] > 0):
            outputs['loss_mask'] = data_pr['loss_mask'].reshape(1) * self.get_mask_weight(step)
        return outputs

class FGLoss(Loss):
    default_cfg = {
        'fg_loss_weight': 4, #changed here from 0.01 to 0.1
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_fg' in data_pr:
            outputs['loss_fg'] = data_pr['loss_fg'] * self.cfg['fg_loss_weight']
        return outputs

class BGLoss(Loss):
    default_cfg = {
        'bg_loss_weight': 0.1, #changed here from 0.01 to 0.1
        'bg_weight_decay_begin': 0,
        'bg_weight_decay_end': 0,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def get_bg_cosine_weight(self, step):
        begin = self.cfg.get('bg_weight_decay_begin', 0)
        end = self.cfg.get('bg_weight_decay_end', begin)
        if end <= begin:
            return 1.0
        span = end - begin
        nom = np.clip(step - begin, 0, span)
        return (np.cos((nom / span) * np.pi) + 1.0) * 0.5

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_bg' in data_pr:
            weight = self.cfg['bg_loss_weight'] * self.get_bg_cosine_weight(step)
            outputs['loss_bg'] = data_pr['loss_bg'] * weight
        return outputs

class CurvLoss(Loss):
    default_cfg = {
        'curv_loss_weight_begin': 0.001,
        'curv_loss_weight_end': 0.001,
        'curv_weight_decay_begin': 20000,
        'curv_weight_decay_end': 50000,
    }

    def map_range_val(self, input_val, input_start, input_end, output_start, output_end):
        input_clamped = max(input_start, min(input_end, input_val))
        return output_start + ((output_end - output_start) / (input_end - input_start)) * (
                input_clamped - input_start
        )

    def get_curvature_weight(self, step):
        # return self.cfg['curv_loss_weight_begin']
        nom = max(step - self.cfg['curv_weight_decay_begin'], 0)
        denom = self.cfg['curv_weight_decay_end'] - self.cfg['curv_weight_decay_begin']
        nom = min(nom, denom)
        coef = self.cfg['curv_loss_weight_begin'] - self.cfg['curv_loss_weight_end']
        bias = self.cfg['curv_loss_weight_end']
        anneal_weights = (np.cos((nom / denom) * np.pi) + 1) / 2

        return anneal_weights * coef + bias

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_curv' in data_pr:
            outputs['loss_curv'] = data_pr['loss_curv'].reshape(1) * self.get_curvature_weight(step)
        return outputs

class OpacityLoss(Loss):
    default_cfg = {
        'opacity_loss_weight': 1,  # changed here from 0.01 to 1
        'opacity_loss_weight_begin': 0.001,
        'opacity_loss_weight_end': 0.01,
        'opacity_weight_decay_begin': 20000,
        'opacity_weight_decay_end': 50000,
    }


    def get_opacity_weight(self, step):
        nom = max(0, step - self.cfg['opacity_weight_decay_begin'])
        denom = self.cfg['opacity_weight_decay_end'] - self.cfg['opacity_weight_decay_begin']
        nom = min(nom, denom)
        coef = self.cfg['opacity_loss_weight_end'] - self.cfg['opacity_loss_weight_begin']
        bias = self.cfg['opacity_loss_weight_begin']
        anneal_weights = (np.cos((nom / denom) * np.pi + np.pi) + 1) / 2 #[0-1]
        return anneal_weights * coef + bias # [begin-end]

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        outputs = {}
        if 'loss_opacity' in data_pr:
            outputs['loss_opacity'] = data_pr['loss_opacity'].reshape(1)  * self.get_opacity_weight(step)
        return outputs

name2loss = {
    'nerf_render': NeRFRenderLoss,
    'eikonal': EikonalLoss,
    'std': StdRecorder,
    'init_sdf_reg': InitSDFRegLoss,
    'occ': OccLoss,
    'mask': MaskLoss,

    'mat_reg': MaterialRegLoss,
    'curv': CurvLoss,
    'opacity': OpacityLoss,
    'fg': FGLoss,
    'bg': BGLoss
}
