"""Real material I/O/dispatch methods with explicit CPU geometry/device fakes."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geometry_fakes import REPO, Tensor, compile_nodes, defaults, trainer_fixture


def material_fixture():
    fixture = trainer_fixture()
    scope = fixture.Trainer.run.__globals__
    torch = fixture.torch
    torch.device = lambda name: name
    torch.set_default_tensor_type = lambda name: fixture.events.append(("tensor_type", name))
    parse = scope['name2dataset']['dummy'].__init__.__globals__['parse_database_name']
    split = scope['name2dataset']['dummy'].__init__.__globals__['get_database_split']

    class Shader:
        def __init__(self, cfg, trace):
            self.cfg = {**defaults('network/field.py', 'MCShadingNetwork'), **cfg}

    class Base(fixture.Renderer):
        def __init__(self):
            self.loaded = None
            fixture.networks.append(self)

        def state_dict(self):
            return {'shader_network.rotation_predictor.weight': Tensor([[1], [2]]),
                    'shader_network.kd_predictor.weight': Tensor([[1], [2], [3]])}

        def _construct_ray_batch(self, info):
            return {'rays_o': np.zeros((len(info['imgs']) * 4, 3), np.float32)}

        _construct_nerf_ray_batch = _construct_ray_batch

        def _shuffle_train_batch(self):
            fixture.events.append('shuffle_rays')

        def outputs(self, index=0):
            values = {'loss_rgb': Tensor([0.2]), 'loss_mat_reg': Tensor([0.05]),
                      'loss_diffuse_light': Tensor([0.1]),
                      'rgb_gt': Tensor(np.full((12, 12, 3), 0.25 + index / 100)),
                      'rgb_pr': Tensor(np.full((12, 12, 3), 0.5))}
            for key in ['tangents', 'bitangents', 'normals', 'kd', 'specular_light', 'diffuse_light',
                        'specular_color', 'diffuse_color', 'mx', 'my', 'metallic', 'f_d', 'f_s_sum', 'L_spec']:
                values[key] = Tensor(np.full((12, 12, 3), 0.3))
            return values

        def __call__(self, data):
            return self.outputs()

        def test_step(self, index, is_nvs=False):
            fixture.events.append(('test_step', index, is_nvs))
            return self.outputs(index)

    def read_mesh(path):
        fixture.events.append(('mesh_read', path, Path(path).read_bytes()))
        return SimpleNamespace(vertices=np.zeros((3, 3)), triangles=np.array([[0, 1, 2]]),
                               has_vertex_normals=lambda: False, compute_vertex_normals=lambda: None)

    def write_cloud(path, points):
        Path(path).write_text('CPU point-cloud fixture\n' + repr(points.points.tolist()))

    open3d = SimpleNamespace(io=SimpleNamespace(read_triangle_mesh=read_mesh, write_point_cloud=write_cloud),
                             geometry=SimpleNamespace(PointCloud=lambda: SimpleNamespace()),
                             utility=SimpleNamespace(Vector3dVector=lambda value: value))
    draw_scope = {'np': np, 'cv2': SimpleNamespace(BORDER_CONSTANT=0,
                  copyMakeBorder=lambda image, top, bottom, left, right, **kwargs:
                  np.pad(image, ((top, bottom), (left, right), (0, 0))))}
    compile_nodes('utils/draw_utils.py', draw_scope,
                  lambda n: isinstance(n, ast.FunctionDef) and n.name in {'concat_images', 'concat_images_list'})
    methods = {'__init__', '_init_geometry', '_init_dataset', '_init_shader', 'save_pointcloud', 'nvs'}
    tree = ast.parse((REPO / 'network/renderer.py').read_text())
    material = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NeROMaterialRenderer')
    material.body = [n for n in material.body if isinstance(n, ast.Assign)
                     or isinstance(n, ast.FunctionDef) and n.name in methods]
    for node in material.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []  # Fake device has no autograd decorator.
    renderer_scope = {'nn': SimpleNamespace(Module=Base), 'np': np, 'torch': torch, 'open3d': open3d,
                      'raytracing': SimpleNamespace(RayTracer=lambda *args: object()),
                      'parse_database_name': parse, 'get_database_split': split,
                      'build_imgs_info': lambda db, ids, is_nerf: {'imgs': np.zeros((len(ids), 3, 2, 2))},
                      'imgs_info_to_torch': lambda info, device: info, 'MCShadingNetwork': Shader,
                      'os': __import__('os'), 'color_map_backward': fixture.metric_scope['color_map_backward'],
                      'compute_psnr': fixture.metric_scope['compute_psnr']}
    exec(compile(ast.Module(body=[material], type_ignores=[]), 'material_fixture', 'exec'), renderer_scope)
    metric_scope = {**fixture.metric_scope, 'concat_images_list': draw_scope['concat_images_list']}
    compile_nodes('network/metrics.py', metric_scope,
                  lambda n: isinstance(n, ast.ClassDef) and n.name == 'MaterialRenderMetrics')
    scope['name2renderer']['material'] = renderer_scope['NeROMaterialRenderer']
    scope['name2metrics']['mat_render'] = metric_scope['MaterialRenderMetrics']
    fixture.MaterialRenderer = renderer_scope['NeROMaterialRenderer']
    fixture.modules = {'open3d': open3d,
                       'utils.draw_utils': SimpleNamespace(concat_images_list=draw_scope['concat_images_list'])}
    fixture.metric_scope = metric_scope
    return fixture
