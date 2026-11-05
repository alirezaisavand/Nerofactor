"""Isolated, fresh-weight native initialization and one forward; no training loop.

Importing/preflighting this module does not import scientific/native dependencies.
Only an explicit execute request may enter the native backend.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import traceback

from infrastructure.config import ConfigError
from infrastructure.geometry import (CODE_ROOT, check_geometry_selection, prepare_geometry,
                                     validate_geometry)
from infrastructure.geometry_execution import execution_environment
from infrastructure.material import (check_material_selection, prepare_material,
                                     validate_material)
from infrastructure.runs import RunError


PROTOCOL = 'native-training-forward-v1'


def validate_smoke(configuration, resolver, run_id, layout='collection'):
    if configuration.schema_version != 1 or configuration.stage not in {'geometry', 'material'}:
        raise ConfigError('Smoke tests require a schema-1 geometry or material envelope')
    if configuration.inputs.checkpoint is not None:
        raise ConfigError('Smoke protocol uses fresh weights only; checkpoint inputs/resume are forbidden')
    root = resolver.output_root
    if root == CODE_ROOT or CODE_ROOT in root.parents or root in CODE_ROOT.parents:
        raise ConfigError('Smoke outputs must be separate from source storage')
    validate = validate_geometry if configuration.stage == 'geometry' else validate_material
    check = check_geometry_selection if configuration.stage == 'geometry' else check_material_selection
    plan = validate(configuration, resolver)
    check(plan, run_id, mode='new', layout=layout)
    return plan


def readonly_filesystem(path):
    """Inspect mount flags without probing writes or changing input permissions."""
    return bool(os.statvfs(str(path)).f_flag & os.ST_RDONLY)


def check_mounts(plan, readonly=readonly_filesystem):
    paths = {CODE_ROOT, plan.configuration.config_root, plan.configuration.source,
             plan.configuration.legacy_source, plan.selected.dataset,
             plan.selected.dataset / plan.pipeline_config['database_name'].split('/')[1],
             *plan.resolver.input_roots.values()}
    if plan.selected.mesh is not None:
        paths.add(plan.selected.mesh)
    for path in sorted(paths):
        if not readonly(path):
            raise RunError('Native smoke requires OS read-only code/config/input filesystems: {}'.format(path))
    output = plan.resolver.output_root
    while not output.exists():
        output = output.parent
    if readonly(output):
        raise RunError('Smoke output filesystem must be writable')
    return {'read_only_paths': [str(p) for p in sorted(paths)],
            'output_filesystem': str(output), 'enforcement': 'statvfs ST_RDONLY; no write probe'}


@contextmanager
def smoke_environment(experiment):
    home = experiment.run.output('cache/runtime/home')
    home.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get('HOME')
    try:
        os.environ['HOME'] = str(home)
        with execution_environment(experiment):
            yield
    finally:
        if previous is None:
            os.environ.pop('HOME', None)
        else:
            os.environ['HOME'] = previous


def tensor_summary(outputs, torch, stage):
    primary = 'ray_rgb' if stage == 'geometry' else 'rgb_pr'
    if not isinstance(outputs, dict) or primary not in outputs or not torch.is_tensor(outputs[primary]):
        raise RunError('Native forward did not return the expected {} tensor'.format(primary))
    rgb = outputs[primary]
    if len(rgb.shape) != 2 or rgb.shape[0] == 0 or rgb.shape[1] != 3:
        raise RunError('Expected nonempty N-by-3 primary RGB output')
    result = {}
    for name, value in outputs.items():
        if torch.is_tensor(value):
            finite = bool(torch.isfinite(value.detach()).all().item())
            result[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                            'device': str(value.device), 'finite': finite}
            if not finite:
                raise RunError('Nonfinite native output: ' + str(name))
    return result


def one_forward(experiment, trainer_type, renderers, torch, mark):
    """Use unchanged constructor/seed/dataset and renderer forward interfaces.

    Trainer._init_network/run are intentionally not called: they create optimizers
    or enter training. No DataLoader iterator/workers or loss-class evaluation.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RunError('Expose exactly one CUDA device explicitly for native smoke execution')
    stage = experiment.plan.configuration.stage
    mark('initializing')
    trainer = trainer_type(deepcopy(experiment.plan.pipeline_config), experiment=experiment)
    trainer._init_dataset()
    options = {'preprocessing': experiment.preprocessing}
    if stage == 'geometry':
        options['isolated_geometry'] = True
    else:
        options['output_paths'] = experiment.run
    model = renderers[trainer.cfg['network']](trainer.cfg, **options).cuda()
    experiment.record_owner('renderer', model.cfg)
    if stage == 'material':
        experiment.record_owner('shader', model.shader_network.cfg)
    torch.cuda.synchronize()
    details = {'renderer': type(model).__name__, 'seed': trainer.cfg['random_seed'],
               'train_ray_num': model.cfg['train_ray_num'],
               'available_training_rays': int(model.tbn),
               'splits': {name: [str(i) for i in getattr(model, name)]
                          for name in ['train_ids', 'test_ids', 'nvs_ids']}}
    mark('initialized', initialization=details)
    # DummyDataset's training item is {}; renderer owns real rays/images and its
    # unchanged stochastic sampling. Avoid the enormous dummy random sampler.
    data = trainer.train_set.dataset[0]
    if not isinstance(data, dict) or data:
        raise RunError('Smoke protocol expects the existing empty dummy training item')
    data['step'] = 0
    model.train()
    mark('forward')
    # Autograd must stay enabled: geometry computes SDF derivatives internally.
    outputs = model(data)
    torch.cuda.synchronize()
    summary = tensor_summary(outputs, torch, stage)
    primary = 'ray_rgb' if stage == 'geometry' else 'rgb_pr'
    if not summary[primary]['device'].startswith('cuda'):
        raise RunError('Native forward primary RGB output must reside on CUDA')
    return {'outputs': summary, 'torch': torch.__version__,
            'cuda_compiled': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'forward_calls': 1, 'step': 0}


def native_backend(experiment, mark):
    # Reached only after run/cache/mount setup under explicit --execute.
    mark('native_imports')
    import torch
    from train.trainer import Trainer
    from network.renderer import name2renderer
    return one_forward(experiment, Trainer, name2renderer, torch, mark)


def execute_smoke(plan, run_id, layout='collection', image_id=None, *,
                  backend=None, readonly=readonly_filesystem):
    if plan.selected.checkpoint is not None:
        raise RunError('Checkpoint inputs are forbidden in native smoke tests')
    mounts = check_mounts(plan, readonly)
    stage = plan.configuration.stage
    prepare = prepare_geometry if stage == 'geometry' else prepare_material
    experiment = prepare(plan, run_id, mode='new', layout=layout)
    report = {'schema_version': 1, 'protocol': PROTOCOL, 'stage': stage,
              'run_id': run_id, 'status': 'running', 'phase': 'allocated',
              'backend': 'native' if backend is None else 'injected_cpu_fixture',
              'declared_image_id': image_id, 'mounts': mounts,
              'compatibility_signature': experiment.signature,
              'source_files': {p: hashlib.sha256((CODE_ROOT / p).read_bytes()).hexdigest()
                               for p in ['run_native_smoke.py', 'infrastructure/native_smoke.py']},
              'policy': {'fresh_weights': True, 'optimizer_constructed': False,
                         'backward_called': False, 'checkpoint_io': False,
                         'evaluation_or_nvs': False, 'configuration_overrides_by_harness': {}},
              'limits': ['Initialization eagerly loads the unchanged scene and builds native rays/BVH.',
                         'One training-mode forward at step 0; no full training step or loss-class aggregation.',
                         'Dummy DataLoader iteration is omitted; no training-loop RNG trajectory parity claimed.',
                         'Fresh weights and finite outputs do not establish reconstruction quality or scientific parity.',
                         'Image identity is caller supplied; use a separately inspected immutable image ID.',
                         'Single-use subprocess only; native global tensor defaults after failure are not repaired.']}

    def mark(phase, **values):
        report.update(phase=phase, **values)
        experiment.run.output('smoke.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print('SMOKE {} {}'.format(stage, phase), flush=True)

    try:
        with smoke_environment(experiment):
            try:
                if stage == 'material':
                    experiment.check_output_storage()
                    experiment.verify_inputs()
                mark('runtime_ready')
                result = (native_backend if backend is None else backend)(experiment, mark)
                if stage == 'material':
                    experiment.verify_inputs()
                if any(experiment.run.checkpoints.iterdir()):
                    raise RunError('Unexpected checkpoint output in smoke directory')
                report.update(result=result, status='passed')
                mark('complete')
            except BaseException:
                traceback.print_exc()
                raise
    except BaseException as exc:
        report.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        experiment.run.output('smoke.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        raise
    return experiment.run, report
