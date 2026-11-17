"""Sphere-only fresh reconstruction adapter; no scientific imports at module import.

This explicit historical-mesh replay never grants a research G-to-A handoff.
The pinned Trainer.run, dataset, losses, Adam and scheduler remain the owners of
scientific work. Subclasses below add observation and destination routing only.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from infrastructure.material_variants import CORE, configuration
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, PINS, baseline, digest

PROTOCOL = 'sphere-corrected-core-replay-v1'
IMAGE = 'sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331'
MESH_SHA = '3e512585ad62aafdbdfdde785f64d1be2454a891fde2f730c027f5b02ed60eba'
CAMERA_SHA = 'c39d4423b0a0ae7da607c822e2b6d22fb20a0582cd85d4ddb93ceaf9fca46630'
ORIGINAL_CFG = 'configs/material/nerf/sphere.yaml'
LIMITS = dict(updates=100000, validation_interval=2000, novel_view_interval=10000,
              outer_seconds=86400, inner_seconds=85800, memory_gib=128, cpus=4,
              pids=256, output_bytes=64 * 1024**3, gpu='0')
CAPTURE_KEYS = ('rotation', 'rotation_raw', 'linear_rgb', 'linear_diffuse', 'linear_specular')
VIEW_INTERFACE = 'sphere-unit-outgoing-at-shade-v1'


def unit_outgoing_views(view_dirs, torch):
    """Adapt original nonunit NeRF rays ONLY at corrected shader's boundary.

    Preserve the raw camera-ray batch and tracer's depth/intersection convention.
    This explicitly changes shader inputs relative to historical raw-ray A;
    corrected-core's existing unit-vector contract and formulas stay unchanged.
    """
    require(isinstance(view_dirs, torch.Tensor) and view_dirs.ndim == 2 and
            view_dirs.shape[-1] == 3 and view_dirs.is_floating_point(),
            'Sphere outgoing views must be floating Nx3 vectors')
    lengths = torch.linalg.norm(view_dirs, dim=-1, keepdim=True)
    require(bool(torch.isfinite(view_dirs).all() & torch.isfinite(lengths).all() & (lengths > 0).all()),
            'Sphere outgoing views must be finite and nonzero; no fallback')
    return view_dirs / lengths  # Out-of-place: no ray-batch or normal mutation.


def split_ids():
    """Independent metadata check, with a LOCAL RNG, never a new split policy."""
    ids = [str(i) for i in range(100)]
    random.Random(6033).shuffle(ids)
    return dict(train=ids[6:], validation=ids[5:6], nvs=ids[:5])


def effective_configuration(raw, dataset_root, mesh, name):
    """Preserve proven ignored legacy fields as diagnostics, not working options."""
    cfg = deepcopy(raw)
    from infrastructure.runs import validate_run_id
    validate_run_id(name)
    require(cfg['database_name'] == 'nerf/sphere' and cfg['is_nerf'] is True and
            cfg['total_step'] == 100000 and cfg['lr_cfg'] == dict(end_warm=1000, end_iter=100000),
            'Only the unchanged original sphere preset is admitted')
    ignored_top = ('anisotropy', 'max_n_exp', 'max_alpha_exp', 'reg_energy_loss',
                   'reg_energy_loss_lambda', 'reg_spec_loss', 'reg_spec_loss_lambda')
    diagnostics = {'ignored_top_level': {k: cfg.pop(k) for k in ignored_top},
                   'unused_shader_fields': {k: cfg['shader_cfg'].pop(k) for k in
                                             ('shading_feature_num', 'num_viewdir_freqs')}}
    cfg['name'] = name
    cfg['mesh'], cfg['dataset_dir'] = str(mesh), str(dataset_root)
    cfg['train_dataset_cfg']['dataset_dir'] = str(dataset_root)
    for entry in cfg['val_set_list']:
        entry['cfg']['dataset_dir'] = str(dataset_root)
    cfg.update(material_variant=CORE, random_seed=6033, worker_num=8,
               novel_view_interval=10000)
    # These are the ORIGINAL effective shader defaults, NOT the ignored .01
    # top-level energy/specular declarations. Keep the learned lighting networks.
    cfg['shader_cfg'].update(anisotropy=True, regularization_version='legacy_v1',
        reg_change=True, change_eps=.02, change_type='gaussian', reg_lambda1=.05,
        random_azimuth=True, reg_energy_loss=True, reg_energy_loss_lambda=.05,
        reg_spec_loss=True, reg_spec_loss_lambda=.05, is_real=False,
        illumination_mode='learned_native', indirect_mode='native_learned')
    return configuration(cfg, CORE), diagnostics


def input_records(dataset_root, mesh, historical):
    """Hash only the 100 active original frames/masks, calibration and comparisons."""
    dataset_root, mesh, historical = map(Path, (dataset_root, mesh, historical))
    require(all(p.is_absolute() and p.resolve() == p and p.is_dir() for p in (dataset_root,historical)),
            'Explicit absolute, unredirected dataset and historical input roots required')
    require(mesh == historical / 'meshes/sphere_shape-300000.ply', 'Explicit approved historical sphere mesh required')
    scene = dataset_root / 'sphere'
    require(mesh.is_file() and digest(mesh) == MESH_SHA, 'Explicit historical sphere mesh changed/missing')
    camera = scene / 'transforms_train.json'
    require(camera.is_file() and digest(camera) == CAMERA_SHA, 'Sphere camera identity changed/missing')
    meta = json.loads(camera.read_text())
    require(len(meta['frames']) == 100 and math.isfinite(meta['camera_angle_x']), 'Expected 100 original cameras')
    paths = [camera]
    for frame in meta['frames']:
        parts = frame['file_path'].split('\\')
        require(len(parts) == 2 and parts[0] == 'train' and Path(parts[1]).name == parts[1],
                'Unsafe/unsupported original frame name')
        paths.extend((scene / 'train' / parts[1], scene / 'masks_train' / parts[1]))
    paths.append(mesh)
    paths.extend(historical / 'nvs_imgs/sphere_material' / ('test_' + i + '.png') for i in split_ids()['nvs'])
    paths.append(historical / 'logs/sphere_material/log.txt')
    # A step-labelled montage is comparison evidence, not checkpoint attribution.
    paths.append(historical / 'train_vis/sphere_material-val/99999-index-0.jpg')
    records = []
    for path in paths:
        require(path.is_file() and path.resolve() == path, 'Missing/redirected sphere input: ' + str(path))
        root = dataset_root if path == camera or scene in path.parents else historical
        records.append(dict(root='datasets' if root == dataset_root else 'thesis',
                            path=path.relative_to(root).as_posix(),
                            bytes=path.stat().st_size, sha256=digest(path)))
    return records


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, root, seconds=LIMITS['inner_seconds'] - 120):
        self.root = Path(root)
        self.deadline = time.monotonic() + seconds
        self.next_size_check = 0.

    def check(self):
        now = time.monotonic()
        if now >= self.deadline:
            raise BudgetStop('Authorized runtime reached; preserve partial checkpoints, no extension')
        if now >= self.next_size_check:
            total = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())
            if total >= LIMITS['output_bytes']:
                raise BudgetStop('Output cap reached; preserve partial results')
            self.next_size_check = now + 20


@contextmanager
def evaluation_scope(model, torch, np):
    """Extra visualization calls cannot perturb the subsequent training stream."""
    py, numpy = random.getstate(), np.random.get_state()
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    default = torch.tensor([]).type()
    modes = [(m, m.training) for m in model.modules()]
    verification = {'RNG_and_modes_restored':False}
    try:
        yield verification
    finally:
        random.setstate(py)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        torch.set_default_tensor_type(default)
        for module, mode in modes:
            module.training = mode
        numpy_after = np.random.get_state()
        cuda_after = torch.cuda.get_rng_state_all()
        require(random.getstate() == py and np.array_equal(numpy_after[1],numpy[1]) and
                numpy_after[0] == numpy[0] and numpy_after[2:] == numpy[2:] and
                torch.equal(torch.get_rng_state(),cpu) and
                len(cuda_after) == len(cuda) and all(torch.equal(a,b) for a,b in zip(cuda_after,cuda)) and
                torch.tensor([]).type() == default and all(m.training == mode for m,mode in modes),
                'Extra NVS changed training RNG, modes or default tensor type')
        verification['RNG_and_modes_restored'] = True


def bind_optimizer(trainer, report, persist, budget, torch):
    """Count actual original Adam returns; reject nonfinite gradients before call."""
    original = trainer.optimizer.step

    def observed(*args, **kwargs):
        budget.check()
        step = report['optimizer_returns']
        require(step < LIMITS['updates'], 'Unexpected extra original optimizer call')
        params = [p for p in trainer.network.parameters() if p.requires_grad]
        active = [p for p in params if p.grad is not None]
        require(active and bool(torch.stack([torch.isfinite(p.grad).all() for p in active]).all()),
                'Missing/nonfinite gradients')
        # Only first two updates copy parameters: do not add repeated model probes.
        before = [p.detach().clone() for p in active] if step < 2 else []
        result = original(*args, **kwargs)
        report['optimizer_returns'] += 1
        if step < 2:
            change = sum(float((p.detach() - q).abs().sum().cpu()) for p, q in zip(active, before))
            require(change == 0 if step == 0 else change > 0, 'Unexpected original warm-up displacement')
            states = [trainer.optimizer.state[p] for p in active]
            require(all(int(s['step']) == step + 1 and bool(torch.isfinite(s['exp_avg']).all()) and
                        bool(torch.isfinite(s['exp_avg_sq']).all()) for s in states), 'Adam state did not advance finitely')
            report.setdefault('initial_updates', []).append(dict(zero_based_step=step,
                lr=[g['lr'] for g in trainer.optimizer.param_groups], gradient_parameters=len(active),
                absolute_parameter_displacement=change, adam_step=step + 1,
                parameter_gradients={name:None if p.grad is None else
                    dict(nonzero=int(torch.count_nonzero(p.grad).cpu()),
                         l2=float(torch.linalg.norm(p.grad.detach()).cpu()))
                    for name,p in trainer.network.named_parameters() if p.requires_grad}))
            persist()
        return result

    trainer.optimizer.step = observed


def renderer_adapter(base, output, report, budget, torch, np):
    """Observe original shade outputs; never issue additional material predictions."""
    from infrastructure.sphere_visuals import write_view

    class SphereRenderer(base):
        _captures = None
        _nvs_step = None

        def train_step(self, step):
            budget.check()
            result = super().train_step(step)
            require(all(bool(torch.isfinite(v).all()) for k, v in result.items() if k.startswith('loss')),
                    'Nonfinite original training loss')
            return result

        def shade(self, pts, view_dirs, normals, human_poses, is_train, step=None):
            shader_views = unit_outgoing_views(view_dirs, torch)
            adapter = report.setdefault('view_direction_adapter',dict(identity=VIEW_INTERFACE,
                raw_camera_rays_preserved=True, normals_unchanged=True, first_calls={}))
            mode = 'training' if is_train else 'evaluation'
            if mode not in adapter['first_calls']:
                lengths = torch.linalg.norm(view_dirs,dim=-1)
                adapter['first_calls'][mode] = dict(raw_min=float(lengths.min().cpu()),
                    raw_max=float(lengths.max().cpu()), dtype=str(view_dirs.dtype),device=str(view_dirs.device))
            value = super().shade(pts, shader_views, normals, human_poses, is_train, step)
            if self._captures is not None:
                require(not is_train, 'Visualization capture entered training')
                capture = {k: value[k].detach().cpu().numpy() for k in CAPTURE_KEYS}
                capture.update(position=pts.detach().cpu().numpy(), normal=normals.detach().cpu().numpy())
                self._captures.append(capture)
            return value

        def test_step(self, index, is_nvs=False):
            budget.check()
            step = report['optimizer_returns']
            require(step > 0 and step % (10000 if is_nvs else 2000) == 0,
                    'Unexpected visualization interval')
            selected = list(map(str,self.nvs_ids if is_nvs else self.test_ids))
            require(0 <= int(index) < len(selected), 'Invalid fixed source camera index')
            camera = selected[int(index)]
            identity = dict(completed_updates=step,split='nvs' if is_nvs else 'validation',camera_id=camera)
            require(camera in split_ids()['nvs' if is_nvs else 'validation'] and
                    identity not in report.get('saved_views',[]), 'Unexpected/duplicate fixed camera')
            require(self._captures is None, 'Nested rendering capture')
            self._captures = []
            try:
                value = super().test_step(index, is_nvs=is_nvs)
                arrays = {k: v.detach().cpu().numpy() for k, v in value.items() if hasattr(v, 'detach')}
                # Original test_step fills encoded normals only at hit pixels.
                mask = np.any(arrays['normals'] != 0, axis=-1)
                require(mask.any(), 'Sphere validation view has no intersections')
                capture = {k: np.concatenate([c[k] for c in self._captures], axis=0)
                           for k in (*CAPTURE_KEYS, 'position', 'normal')}
                require(all(len(v) == int(mask.sum()) and np.isfinite(v).all() for v in capture.values()),
                        'Captured hit outputs do not match original raster ordering')
                for k, v in capture.items():
                    full = np.zeros((*mask.shape, v.shape[-1]), dtype=v.dtype)
                    full[mask] = v
                    arrays[k] = full
                arrays['hit_mask'] = mask
                step = report['optimizer_returns']
                write_view(output / 'renders' / ('step-%06d' % step) /
                           ('nvs' if is_nvs else 'validation') / ('camera-' + camera), arrays,
                           dict(camera_id=camera, completed_updates=step, zero_based_validation_step=step - 1,
                                split='nvs' if is_nvs else 'validation'), np)
                report.setdefault('saved_views', []).append(identity)
                return value
            finally:
                self._captures = None

        def nvs(self, log_dir_str, imgs_dir):
            step = report['optimizer_returns']
            require(step % 10000 == 0 and step > 0 and self._nvs_step != step, 'Invalid/duplicate NVS schedule')
            self._nvs_step = step
            path = output / 'renders' / ('step-%06d' % step) / 'legacy_nvs'
            path.mkdir(parents=True, exist_ok=False)
            with evaluation_scope(self, torch, np) as verification:
                result = super().nvs(str(path), str(path))
            report.setdefault('novel_view_scopes',[]).append(dict(completed_updates=step,**verification))
            return result

    return SphereRenderer


def trainer_adapter(base, output, report, persist, budget, torch, np):
    class SphereTrainer(base):
        def __init__(self, cfg):
            super().__init__(cfg)
            # Base constructor performs the original three seed calls first.
            self.model_dir = str(output / 'logs')
            self.pth_fn = str(output / 'checkpoints/model.pth')
            self.best_pth_fn = str(output / 'checkpoints/model_best.pth')

        def _init_network(self):
            require(not hasattr(self, 'network'), 'Unexpected repeated original initialization')
            super()._init_network()
            actual = dict(train=list(map(str, self.network.train_ids)),
                          validation=list(map(str, self.network.test_ids)), nvs=list(map(str, self.network.nvs_ids)))
            require(actual == split_ids(), 'Pinned original sphere split differs')
            shader = self.network.shader_network
            require(self.network.is_nerf and shader.cfg['illumination_mode'] == 'learned_native' and
                    shader.cfg['regularization_version'] == 'legacy_v1', 'Wrong sphere renderer/lighting/priors')
            report.update(split_ids=actual, effective_trainer=self.cfg,
                          effective_renderer=self.network.cfg, effective_shader=shader.cfg,
                          optimizer={k: v for k, v in self.optimizer.param_groups[0].items() if k != 'params'},
                          scheduler=dict(vars(self.lr_manager)), native_initialized=True)
            bind_optimizer(self, report, persist, budget, torch)
            persist()

        def _load_model(self):
            require(not Path(self.pth_fn).exists() and not Path(self.best_pth_fn).exists(),
                    'Sphere replay requires fresh weights; implicit resume forbidden')
            return super()._load_model()

        def _save_model(self, step, best_para, save_fn=None):
            budget.check()
            super()._save_model(step, best_para, save_fn)
            if save_fn is None:
                path = output / 'checkpoints' / ('step-%06d.pth' % step)
                require(not path.exists(), 'Checkpoint interval collision')
                shutil.copyfile(self.pth_fn, path)
                report['last_checkpoint'] = dict(completed_updates=step, path=str(path), sha256=digest(path))
                (path.with_suffix('.json')).write_text(json.dumps(dict(protocol=PROTOCOL,
                    variant=CORE, scientific_commit=PINS['material'], mesh_sha256=MESH_SHA,
                    completed_updates=step, checkpoint_sha256=digest(path),
                    historical_initialization=False, implicit_resume_permitted=False,
                    config_sha256=digest(output/'config.effective.yaml')), indent=2)+'\n')
                persist()
                # Final NVS belongs to the original Trainer.run after its loop.
                if step % 10000 == 0 and step < LIMITS['updates']:
                    self.network.nvs('', '')

        def _log_data(self, results, step, prefix='train', verbose=False):
            super()._log_data(results, step, prefix, verbose)
            row = dict(completed_updates=step, prefix=prefix)
            for k, v in results.items():
                row[k] = float(np.mean(v.detach().cpu().numpy() if hasattr(v, 'detach') else v))
            with (output / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            budget.check()

    return SphereTrainer


def native_execute(cfg, output, report, persist, image):
    import resource
    started = time.monotonic()
    import numpy as np
    import torch
    from infrastructure.verified_ray_backend import load
    require(torch.__version__ == '1.8.1+cu111' and torch.version.cuda == '11.1' and
            torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == 'NVIDIA A100 80GB PCIe', 'Reviewed single-A100 runtime required')
    report['ray_backend'] = load('predicate_fp64', image)
    from infrastructure.stage_sources import activate, imported_sources
    science, _ = activate('material', PINS['material'])
    from train.trainer import Trainer
    import network.renderer as renderers
    import network.field as fields
    from infrastructure.material_variants import install
    from infrastructure.stage_execution import run_material_owned
    from infrastructure.geometry_training_smoke import runtime_identity
    require(Path(Trainer.run.__code__.co_filename).resolve() == science / 'train/trainer.py', 'Wrong Trainer')
    report['runtime'] = runtime_identity(torch)
    budget = Budget(output)
    base = install(renderers, fields, CORE, cfg)
    renderers.name2renderer['material'] = renderer_adapter(base, output, report, budget, torch, np)
    trainer = trainer_adapter(Trainer, output, report, persist, budget, torch, np)(cfg)
    report['phase'] = 'original_trainer_run'
    persist()
    try:
        run_material_owned(trainer, report, persist)
    finally:
        report['resources'] = dict(native_wall_seconds=time.monotonic()-started,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        persist()
    require(report['optimizer_returns'] == 100000 and report.get('cleanup_errors') == [],
            'Incomplete original loop or cleanup')
    expected = [(step, 'validation', '36') for step in range(2000,100001,2000)]
    expected += [(step,'nvs',camera) for step in range(10000,100001,10000) for camera in split_ids()['nvs']]
    actual = [(r['completed_updates'],r['split'],r['camera_id']) for r in report.get('saved_views',[])]
    require(sorted(expected) == sorted(actual), 'Incomplete/duplicate fixed-camera visualization schedule')
    report.update(scientific_work_complete=True, owned_trainer_cleanup_verified=True,
                  imported_sources=imported_sources(science))
    baseline('material', PINS['material'])


def workspace(output, science):
    work = output / 'work'
    work.mkdir()
    (work / 'data').mkdir()
    (work / 'data/train_vis').symlink_to(output / 'renders/legacy_validation', target_is_directory=True)
    (output / 'renders/legacy_validation').mkdir()
    # Original field loads the BSDF lookup table by a relative input path.
    for name in ('assets',):
        require((science / name).is_dir(), 'Pinned scientific asset directory missing')
        (work / name).symlink_to(science / name, target_is_directory=True)
    return work
