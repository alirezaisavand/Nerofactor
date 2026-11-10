"""Differentiable known-input shader, experimental only; no training or I/O.

The immutable reference is (A0, original). Estimator and decoder identities are
always separate. Hash-admitted A algebra executes in plain objects, not models.
"""
from dataclasses import dataclass
import ast
import hashlib
from types import FunctionType, MethodType

import numpy as np
import torch
import torch.nn.functional as F

from experiments import anisotropic_estimator_v1 as algebra
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline, PINS

PROTOCOL = 'known-input-material-shader-v1'
ENCODING_MIN_NORM = 1e-6
VARIANTS = {'A0': 'B0', 'A1': 'B3', 'A2': 'B3'}
DECODERS = ('original', 'atan2')


def _tensor(value, shape, name):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
            value.dtype == torch.float32 and tuple(value.shape) == tuple(shape),
            name + ': explicit CPU float32 tensor with shape ' + str(shape) + ' required')
    require(bool(torch.isfinite(value).all()), name + ': nonfinite input')


def encoding_status(encoding):
    """No fallback: zero and near-zero are distinct, explicitly invalid cases."""
    require(isinstance(encoding, torch.Tensor) and encoding.device.type == 'cpu' and
            encoding.ndim == 2 and encoding.shape[-1] == 2 and
            encoding.dtype in (torch.float32, torch.float64) and
            bool(torch.isfinite(encoding).all()), 'Finite CPU double-angle encodings [N,2] required')
    norm = torch.norm(encoding, dim=-1, keepdim=True)
    require(bool(torch.isfinite(norm).all()), 'Encoding norm overflow; no fallback policy is approved')
    zero = (encoding == 0).all(dim=-1, keepdim=True)
    return {'norm': norm, 'zero': zero,
            'near_zero': (~zero) & (norm < ENCODING_MIN_NORM),
            'valid': norm >= ENCODING_MIN_NORM}


def decode_atan2(encoding):
    status = encoding_status(encoding)
    require(bool(status['valid'].all()),
            'Invalid zero/near-zero anisotropy encoding; no fallback policy is approved')
    r = encoding / status['norm']
    theta = .5 * torch.atan2(r[:, 1:2], r[:, 0:1])
    return torch.cos(theta), torch.sin(theta)


@dataclass(frozen=True)
class RandomInputs:
    """Explicit draws in original order; callers own private seeding."""
    u: torch.Tensor
    v: torch.Tensor
    azimuth: torch.Tensor

    def validate(self, points, count):
        for name, value, shape in [('u', self.u, (points, count)),
                                   ('v', self.v, (points, count)),
                                   ('azimuth', self.azimuth, (points, 1, 1))]:
            _tensor(value, shape, name)
            require(not value.requires_grad and bool(((value >= 0) & (value < 1)).all()),
                    name + ': fixed uniforms in [0,1) required')


class _ExplicitTorch:
    """Per-call namespace: inject draws without patching global torch/RNG state."""
    def __init__(self, arrays):
        self.arrays = arrays
        self.calls = 0

    def __getattr__(self, name):
        return getattr(torch, name)

    def rand(self, *shape, **kwargs):
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
        require(self.calls < len(self.arrays), 'Unexpected random draw')
        value = self.arrays[self.calls]
        require(tuple(value.shape) == shape and kwargs.get('device') in (None, 'cpu', torch.device('cpu')),
                'Random draw shape/order/device differs from the original')
        self.calls += 1
        return value


def _bind_namespace(obj, name, torch_namespace):
    function = getattr(type(obj), name)
    namespace = dict(function.__globals__, torch=torch_namespace)
    bound = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    setattr(obj, name, MethodType(bound, obj))


class MaterialShaderV1:
    """CPU float32 A algebra with caller-supplied material and illumination.

    Defaults are original A bell's 512 diffuse / 1024 specular counts. Explicit
    smaller counts support tiny fixtures; no variant changes counts implicitly.
    Illumination(positions[N,S,3], directions[N,S,3]) returns linear RGB[N,S,3].
    It must be a differentiable torch function for pathwise material gradients.
    """
    def __init__(self, variant, decoder, diffuse_samples=512, specular_samples=1024):
        require(variant in VARIANTS and decoder in DECODERS, 'Unknown estimator/decoder identity')
        require(all(type(n) is int and 0 < n <= 4096 for n in (diffuse_samples, specular_samples)),
                'Explicit positive sample counts <=4096 required')
        objects, identity = algebra.shaders(diffuse_samples)
        self.template = objects[VARIANTS[variant]]
        self.variant, self.decoder = variant, decoder
        self.diffuse_samples, self.specular_samples = diffuse_samples, specular_samples
        if variant == 'A2':
            self.template.underlying_heights = (torch.arange(diffuse_samples, dtype=torch.float32) + .5) / diffuse_samples
        self.identity = dict(protocol=PROTOCOL, variant=variant, decoder=decoder,
                             diffuse_samples=diffuse_samples, specular_samples=specular_samples,
                             radial_rule='midpoint' if variant == 'A2' else ('legacy' if variant == 'A0' else 'left_endpoint'),
                             height_sha256=hashlib.sha256(self.template.underlying_heights.numpy().tobytes()).hexdigest(),
                             original_source=identity)
        root, _ = baseline('material', PINS['material'])
        tree = ast.parse((root / 'utils/raw_utils.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'linear_to_srgb')
        namespace = {'torch': torch, 'np': np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                     str(root / 'utils/raw_utils.py'), 'exec'), namespace)
        self.to_srgb = namespace['linear_to_srgb']

    def shade(self, positions, normals, views, mx, my, albedo, metallic, encoding,
              illumination, random_inputs, is_train=True):
        require(isinstance(positions, torch.Tensor) and positions.ndim == 2 and positions.shape[0] > 0,
                'Nonempty positions [N,3] required')
        count = positions.shape[0]
        for name, value, shape in [('positions', positions, (count, 3)), ('normals', normals, (count, 3)),
                                  ('views', views, (count, 3)), ('mx', mx, (count, 1)), ('my', my, (count, 1)),
                                  ('albedo', albedo, (count, 3)), ('metallic', metallic, (count, 1)),
                                  ('encoding', encoding, (count, 2))]:
            _tensor(value, shape, name)
        require(type(is_train) is bool, 'Explicit Boolean is_train required')
        for value in (normals, views):
            require(bool((abs(torch.norm(value, dim=-1) - 1) <= 2e-6).all()), 'Known unit normals/views required; no repair')
        require(bool(((views * normals).sum(-1) > 0).all()), 'Views must lie in the open outgoing hemisphere')
        require(bool(((mx >= .04) & (mx <= 1) & (my >= .04) & (my <= 1)).all()), 'Original physical slope widths [.04,1] required')
        require(bool(((albedo >= 0) & (albedo <= 1)).all()) and
                bool(((metallic >= 0) & (metallic <= 1)).all()), 'Albedo/metallic in [0,1] required')
        require(isinstance(random_inputs, RandomInputs), 'Explicit RandomInputs required')
        random_inputs.validate(count, self.specular_samples)
        status = encoding_status(encoding)
        rotation = F.normalize(encoding, dim=-1, eps=1e-6)  # Original head normalization.
        model = type(self.template)()
        model.__dict__.update(self.template.__dict__)
        tape = _ExplicitTorch([random_inputs.u, random_inputs.v] + ([random_inputs.azimuth] if is_train else []))
        for method in ('sample_aniso_ggx_directions', 'sample_diffuse_directions'):
            _bind_namespace(model, method, tape)
        if self.decoder == 'atan2':
            c, s = decode_atan2(encoding)
            # Only replace rotation coefficients. Retain the entire source
            # sampler, including clamps, basis, PDFs, masks and float casts.
            def rotate(_self, tangent, bitangent, _original_c, _original_s):
                return c * tangent + s * bitangent, -s * tangent + c * bitangent
            model.rotate_tangent_bitangent = MethodType(rotate, model)
        spec = model.sample_aniso_ggx_directions(rotation, positions, mx, my, views,
                                                self.specular_samples, normals, None, device='cpu')
        diffuse = model.sample_diffuse_directions(normals, is_train)
        require(tape.calls == (3 if is_train else 2), 'Original draw order/count changed')
        h, wi, ch, pdf, t, b, n, theta_h, phi_h = spec
        directions = torch.cat([diffuse, wi], dim=1)
        points = positions.unsqueeze(1).repeat(1, directions.shape[1], 1)
        lights = illumination(points, directions)
        _tensor(lights, directions.shape, 'illumination RGB')
        require(bool((lights >= 0).all()), 'Nonnegative linear incident radiance required')
        f0 = .04 * (1 - metallic) + metallic * albedo
        fresnel, _, _ = model.fresnel_schlick_directions(f0.unsqueeze(1), views.unsqueeze(1), wi)
        fd = model.diffuse_term(albedo, metallic)
        values = model.compute_radiance(fd, lights[:, :self.diffuse_samples], lights[:, self.diffuse_samples:],
                                        fresnel, diffuse, wi, normals, ch, views, pdf, theta_h, phi_h, mx, my)
        mask = (wi * normals.unsqueeze(1)).sum(-1, keepdim=True) > 0
        ci = (wi * normals.unsqueeze(1)).sum(-1, keepdim=True).clamp(min=0)
        unconditional = ((fresnel * ci / ch.clamp(min=1e-6)) * mask).sum(1) / self.specular_samples
        require(all(bool(torch.isfinite(v).all()) for v in (*values, *spec, diffuse)), 'Nonfinite shader output')
        basis = torch.stack([t, b, n], dim=-1)
        gram = basis.transpose(-1, -2) @ basis
        frame_valid = (abs(gram - torch.eye(3, dtype=gram.dtype)).amax(dim=(-1, -2)) <= 2e-6)
        return {'linear_rgb': values[0], 'srgb': self.to_srgb(values[0]),
                'diffuse_rgb': values[1], 'specular_rgb': values[2],
                'original_f_s_sum': values[3], 'original_L_spec': values[4],
                'unconditional_reflectance_proxy': unconditional, 'f_d': fd,
                'diffuse_directions': diffuse, 'specular_directions': wi, 'half_vectors': h,
                'tangent': t, 'bitangent': b, 'normal': n, 'pdf': pdf,
                'valid_specular_counts': mask.sum(1), 'encoding_status': status,
                'frame_orthonormal': frame_valid,
                'random_calls': tape.calls, 'identity': self.identity}
