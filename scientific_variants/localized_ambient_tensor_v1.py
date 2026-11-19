"""Opt-in local residual of the six ambient tensor logits."""
import torch
from torch import nn
from torch.nn import functional as F

from infrastructure.signatures import require
from scientific_variants import donut_tensor_d8 as tensors
from scientific_variants.material_a_v1 import finite, unit


class LocalTensorResidual(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.register_buffer('lower',torch.tensor(grid['lower'],dtype=torch.float32))
        self.tables = nn.ParameterList()
        for i,level in enumerate(grid['levels']):
            x,y,z = level['cells']
            self.register_buffer(f'extent_{i}',torch.tensor([x,y,z],dtype=torch.float32)*level['pitch'])
            self.tables.append(nn.Parameter(torch.zeros(1,6,z+1,y+1,x+1)))

    def forward(self, points):
        require(points.ndim==2 and points.shape[1]==3 and points.dtype==self.lower.dtype,
                'Expected the same float32 XYZ coordinates as the global field')
        residual = points.new_zeros((len(points),6))
        for i,table in enumerate(self.tables):
            uvw = 2*(points-self.lower)/getattr(self,f'extent_{i}')-1
            # The closed mesh AABB includes every observation. Only rounding at
            # a representable endpoint may exceed an interpolation bound.
            tolerance = 64*torch.finfo(uvw.dtype).eps
            require(bool(torch.isfinite(uvw).all() & (uvw.abs()<=1+tolerance).all()),
                    'Point outside frozen localized grid')
            value = F.grid_sample(table,uvw.clamp(-1,1).reshape(1,1,1,-1,3),mode='bilinear',
                                  padding_mode='border',align_corners=True)
            residual = residual + value.reshape(6,-1).T
        return residual


class LocalizedAmbientField(nn.Module):
    """Keep the existing global D8-T field as a submodule and add logits before decoding."""
    def __init__(self, global_field, grid):
        super().__init__()
        require(global_field.tensor_arm=='D8-T','Localized residual requires the ambient D8-T field')
        self.global_field = global_field
        self.local = LocalTensorResidual(grid)

    def predict_tensor(self, points, normals):
        unit(normals,'tensor normals')
        base = self.global_field
        frame = tensors.frame(normals)
        features = base.features(points)
        logits = torch.cat([base.mx_predictor(features),base.my_predictor(features),base.rotation_predictor(features)],-1)
        logits = logits + self.local(points)
        finite(logits,'localized ambient tensor logits')
        q = tensors.ambient_q(logits,frame,initial_width=base.initial_width)
        return q,frame,base.kd_predictor(features),base.metallic_predictor(features)
