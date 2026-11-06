"""A-based one-view checkpoint regression. Native work is explicit and researcher-side."""

from contextlib import contextmanager
from copy import deepcopy
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import traceback
from types import SimpleNamespace

from infrastructure.signatures import (ADMITTED_REPORT_SHA256, CHECKPOINT_SHA256, compare_catalogs,
                                       read_report, require, validate_checkpoint, validate_reference, write_report)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'checkpoint-inference-validation-v1'
REFERENCE_SHA256 = '768ce796e7d3356b17cdf76042666c71d73462442be2a99a02480214a047b0f8'
COMPARISON_SHA256 = '4a2413bb174a94507dbb771477652b356e5e9269c891f8c1d4ec5fde9fc4886c'
STORED_STEP = 150000
THRESHOLDS = {'float_atol': 1e-4, 'float_rtol': 1e-4, 'uint8_max': 1, 'psnr_db': 0.01, 'ssim': 1e-4}
IMPLEMENTATION_FILES = ('infrastructure/bell_inference.py', 'infrastructure/bell_comparison.py',
                        'run_bell_inference.py', 'compare_bell_inference.py', 'configs/experiments/geometry.example.yaml',
                        'train/trainer.py', 'train/train_tools.py', 'train/train_valid.py', 'train/lr_common_manager.py')


def implementation_identity():
    return {path: file_hash(ROOT / path) for path in IMPLEMENTATION_FILES}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evidence(metadata, reference, comparison):
    cp, cp_id = read_report(metadata, ADMITTED_REPORT_SHA256)
    ref, ref_id = read_report(reference, REFERENCE_SHA256)
    result, result_id = read_report(comparison, COMPARISON_SHA256)
    expected = compare_catalogs(validate_checkpoint(cp), validate_reference(ref))
    require(expected['result'] == 'exact_structural_match' and
            all(result.get(k) == v for k, v in expected.items()), 'Architecture evidence differs from recomputation')
    require(result['checkpoint_identity']['sha256'] == cp_id['sha256'] and
            result['reference_identity']['sha256'] == ref_id['sha256'], 'Evidence identities disagree')
    require(cp['inspection']['stored_step']['value'] == STORED_STEP, 'Unexpected checkpoint step')
    for path, digest in ref['source']['files_sha256'].items():
        require(file_hash(ROOT / path) == digest, 'Pinned reference source/configuration changed: ' + path)
    return {'metadata': cp, 'reference': ref,
            'identities': {'metadata': cp_id['sha256'], 'reference': ref_id['sha256'], 'comparison': result_id['sha256']}}


def resolve_reference(datasets, thesis, output):
    """R/R-repeat: independent legacy FullLoader resolution; no schema adapter."""
    import yaml
    from infrastructure.cache import PreprocessingSettings
    from infrastructure.paths import PathResolver
    path = ROOT / 'configs/shape/syn/bell.yaml'
    raw = path.read_bytes()
    values = yaml.load(raw, Loader=yaml.FullLoader)
    require(type(values) is dict and values.get('database_name') == 'syn/bell', 'Invalid direct bell config')
    resolver = PathResolver({'datasets': datasets, 'thesis': thesis}, output)
    family = resolver.input_roots['datasets'] / 'GlossySynthetic'
    require(family.resolve() == family and (family / 'bell').is_dir(), 'Bell dataset missing or aliased')
    cfg = deepcopy(values)
    cfg['dataset_dir'] = str(family)
    for nested in [cfg['train_dataset_cfg'], *[v['cfg'] for v in cfg['val_set_list']]]:
        if 'dataset_dir' in nested:
            nested['dataset_dir'] = str(family)
    return cfg, resolver, PreprocessingSettings(), raw, 'direct-legacy-FullLoader'


def resolve_candidate(datasets, thesis, output):
    """C: approved schema-1/path resolution, independent of R's legacy loader."""
    from infrastructure.config import load_config
    from infrastructure.geometry import validate_geometry
    from infrastructure.paths import PathResolver
    cfg = load_config(ROOT / 'configs/experiments/geometry.example.yaml', config_root=ROOT / 'configs')
    require(cfg.original.get('overrides') == {}, 'Scientific overrides are not admitted')
    resolver = PathResolver({'datasets': datasets, 'thesis': thesis}, output)
    plan = validate_geometry(cfg, resolver)
    return plan.pipeline_config, resolver, cfg.preprocessing, plan.legacy_bytes, 'schema-1-PathResolver'


def ordered_split(model):
    initial = model.database.get_img_ids()
    require(initial == [str(i) for i in range(128)], 'Unexpected ordered bell input IDs')
    shuffled = initial.copy()
    random.Random(6033).shuffle(shuffled)  # Local verifier; no additional global RNG consumption.
    expected = {'train_ids': shuffled[6:], 'test_ids': shuffled[5:6], 'nvs_ids': shuffled[:5]}
    actual = {name: [str(v) for v in getattr(model, name)] for name in expected}
    require(actual == expected and actual['test_ids'] == ['58'], 'Actual A split differs from admitted proposal')
    return actual


def proposed_split():
    ids = [str(i) for i in range(128)]
    random.Random(6033).shuffle(ids)
    return {'train_ids': ids[6:], 'test_ids': ids[5:6], 'nvs_ids': ids[:5]}


def used_inputs(scene, splits):
    """Hash exactly selected camera/RGB/mask/depth inputs, not unrelated dataset families."""
    ids = splits['train_ids'] + splits['test_ids']
    names = [str(i) + '-camera.pkl' for i in range(128)]
    names += [name for image in ids for name in (image + '.png', 'masks_train/' + image + '.png')]
    names += [splits['test_ids'][0] + '-depth.png']
    result = []
    for name in sorted(names):
        path = scene / name
        require(path.resolve() == path and path.is_file(), 'Unsafe/missing selected dataset input')
        stat = path.stat()
        result.append({'path': name, 'bytes': stat.st_size, 'sha256': file_hash(path)})
        require((stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns),
                'Dataset input changed during hashing')
    return result


@contextmanager
def runtime_environment(run):
    from infrastructure.geometry_execution import execution_environment
    home = run.output('cache/runtime/home')
    home.mkdir(parents=True)
    previous = os.environ.get('HOME')
    os.environ['HOME'] = str(home)
    try:
        with execution_environment(SimpleNamespace(run=run)):
            yield
    finally:
        if previous is None:
            os.environ.pop('HOME', None)
        else:
            os.environ['HOME'] = previous


def rng_identity(torch, np):
    state = np.random.get_state()
    return {'python': hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            'numpy': hashlib.sha256(repr((state[0], state[2:])).encode() + state[1].tobytes()).hexdigest(),
            'torch_cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'torch_cuda': [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()]}


def seed_and_merge(cfg, defaults, torch, np):
    effective = {**defaults, **cfg}  # Original Trainer shallow merge and three-call seed policy.
    torch.manual_seed(effective['random_seed'])
    np.random.seed(effective['random_seed'])
    random.seed(effective['random_seed'])
    return effective


def strict_load(model, state, torch):
    require(type(state) is dict and type(state.get('step')) is int and state['step'] == STORED_STEP,
            'Checkpoint container/step differs from admitted metadata')
    saved, current = state.get('network_state_dict'), model.state_dict()
    require(isinstance(saved, dict) and saved.keys() == current.keys(), 'Checkpoint network key mismatch')
    for key, value in saved.items():
        require(type(value) in (torch.Tensor, torch.nn.Parameter) and value.device.type == 'cpu',
                'Unexpected checkpoint tensor/device')
        require(value.shape == current[key].shape and value.dtype == current[key].dtype,
                'Checkpoint network shape/dtype mismatch: ' + key)
    model.load_state_dict(saved, strict=True)


def scalar_metric(value):
    import math
    value = float(value)
    if math.isinf(value) and value > 0:
        return {'status': 'positive_infinity', 'value': None}
    require(math.isfinite(value), 'Invalid metric value')
    return {'status': 'finite', 'value': value}


def native_arm(cfg, cache, run, checkpoint, packet, mark):
    mark('scientific_imports')
    import numpy as np
    import torch
    require(torch.__version__ == '1.8.1+cu111' and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Expected the reviewed runtime and one selected GPU')
    from train.trainer import Trainer  # Class defaults only; no Trainer/optimizer instance.
    from dataset.train_dataset import DummyDataset
    from network.renderer import NeROShapeRenderer
    from network.metrics import ShapeRenderMetrics
    from utils.base_utils import color_map_backward
    from generate_bell_reference import extract_signature
    from infrastructure.signatures import tensor_catalog
    import imageio

    effective = seed_and_merge(cfg, Trainer.default_cfg, torch, np)
    mark('dataset_model_initialization')
    require(len(effective['val_set_list']) == 1 and effective['val_set_list'][0]['name'] == 'val', 'One validation set required')
    scene = Path(effective['dataset_dir']) / 'bell'
    require(len(list(scene.glob('*.pkl'))) == 128, 'Unexpected camera pickle inventory')
    inputs = used_inputs(scene, proposed_split())
    # Same dataset-construction/split order as Trainer._init_dataset, without DataLoader iteration.
    DummyDataset(effective['train_dataset_cfg'], True, dataset_dir=effective['dataset_dir'], preprocessing=cache)
    validation = DummyDataset(effective['val_set_list'][0]['cfg'], False,
                              dataset_dir=effective['dataset_dir'], preprocessing=cache)
    require(len(validation) == 1, 'Expected one validation view')
    model = NeROShapeRenderer(effective, preprocessing=cache, isolated_geometry=True).cuda()
    require(not model.cfg['val_geometry'], 'Mesh extraction is outside this protocol')
    splits = ordered_split(model)
    actual = tensor_catalog(extract_signature(model), len(model.state_dict()), reference=True)
    require(compare_catalogs(actual, validate_reference(packet['reference']))['result'] == 'exact_structural_match',
            'Actual live model differs from the pinned architecture reference')
    mark('strict_weight_load')
    descriptor = os.open(str(checkpoint), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        from inspect_bell_checkpoint import _hash_stream
        require(_hash_stream(stream) == CHECKPOINT_SHA256, 'Checkpoint hash mismatch before loading')
        state = torch.load(stream, map_location='cpu')  # Trusted full pickle container, including unused optimizer payload.
        strict_load(model, state, torch)
        require(float(state.get('best_para')) == packet['metadata']['inspection']['best_para']['value'], 'Best metric differs')
        require(_hash_stream(stream) == CHECKPOINT_SHA256, 'Checkpoint changed during loading')
    del state  # No optimizer construction/restoration; do not retain unused payloads.
    require(inputs == used_inputs(scene, splits), 'Selected input content changed during initialization')
    model.eval()
    data = validation[0]
    require(data == {'index': 0}, 'Unexpected validation dispatch')
    data.update(eval=True, step=STORED_STEP)
    before = rng_identity(torch, np)
    mark('one_validation_view')
    with torch.no_grad():  # Original validation boundary; SDF.gradient internally re-enables its derivatives.
        outputs = model(data)
    torch.cuda.synchronize()
    after = rng_identity(torch, np)
    prediction, gt = [outputs[key].detach().cpu().numpy() for key in ('ray_rgb', 'gt_rgb')]
    require(prediction.dtype == gt.dtype == np.float32 and prediction.shape == gt.shape and
            prediction.ndim == 3 and prediction.shape[-1] == 3 and
            np.isfinite(prediction).all() and np.isfinite(gt).all(), 'Invalid full validation RGB')
    mark('legacy_validation_metrics')
    metrics = ShapeRenderMetrics(effective, output_paths=run)(outputs, data, STORED_STEP,
                data_index=0, model_name=effective['name'] + '-val')
    paths = {}
    for name, array in [('prediction', prediction), ('ground_truth', gt),
                        ('mask', outputs['gt_mask'].detach().cpu().numpy())]:
        path = run.output('arrays/' + name + '.npy'); path.parent.mkdir(exist_ok=True)
        np.save(str(path), array, allow_pickle=False); paths[name] = path.relative_to(run.root).as_posix()
    for name, array in [('prediction', prediction), ('ground_truth', gt)]:
        path = run.output('renders/' + name + '.png')
        imageio.imwrite(str(path), color_map_backward(array))
    result_metrics = {'psnr': scalar_metric(metrics['psnr'][0]), 'ssim': float(metrics['ssim'][0])}
    require(np.isfinite(result_metrics['ssim']), 'Nonfinite SSIM')
    with run.output('metrics.json').open('x') as stream:
        write_report(stream, result_metrics)
    require(inputs == used_inputs(scene, splits), 'Selected input content changed')
    packages = {}
    for name in ('numpy', 'scikit-image', 'imageio', 'Pillow', 'opencv-python', 'open3d', 'nvdiffrast', 'PyYAML', 'scipy'):
        packages[name] = importlib.metadata.version(name)
    diagnostics = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                         'finite': bool(torch.isfinite(value.detach()).all().item())}
                   for key, value in outputs.items() if torch.is_tensor(value)}
    require(not any(run.checkpoints.iterdir()), 'Inference must not write checkpoints')
    driver = Path('/proc/driver/nvidia/version')
    return {'splits': splits, 'used_inputs': inputs, 'effective_trainer': effective,
            'effective_renderer': model.cfg, 'effective_shader': model.color_network.cfg,
            'rng_before': before, 'rng_after': after, 'metrics': result_metrics, 'arrays': paths,
            'diagnostics': diagnostics, 'runtime': {'python': sys.version, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'packages': packages,
            'driver': driver.read_text() if driver.is_file() else None,
            'default_dtype': str(torch.get_default_dtype()),
            'torch_cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': getattr(torch.backends.cudnn, 'allow_tf32', None),
            'matmul_allow_tf32': getattr(torch.backends.cuda.matmul, 'allow_tf32', None)},
            'preprocessing': cache.bind('syn/bell', scene).identity}


def execute_arm(arm, run_id, datasets, thesis, output, metadata, reference, comparison, image_id, source_commit):
    from infrastructure.cache import PreprocessingCache
    from infrastructure.runs import RunStore
    packet = evidence(metadata, reference, comparison)
    cfg, resolver, settings, original, route = (resolve_candidate if arm == 'C' else resolve_reference)(datasets, thesis, output)
    require(hashlib.sha256(original).hexdigest() == packet['reference']['configuration']['sha256'], 'Legacy config differs from reference')
    checkpoint = resolver.input_roots['thesis'] / 'model/bell_shape/model_best.pth'
    for path in (ROOT, ROOT / 'configs/shape/syn/bell.yaml', resolver.input_roots['datasets'],
                 resolver.input_roots['thesis'], checkpoint, metadata, reference, comparison):
        require(os.statvfs(str(path)).f_flag & os.ST_RDONLY, 'OS read-only input/source mounts required')
    require(os.geteuid() != 0 and os.statvfs('/').f_flag & os.ST_RDONLY, 'Non-root/read-only rootfs required')
    interfaces = {line.split(':', 1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}
    require(interfaces == {'lo'}, 'No-network runtime required')
    require(image_id == packet['reference']['declared_image_id'], 'Use the admitted reference image')
    require(checkpoint.resolve() == checkpoint and checkpoint.is_file() and not checkpoint.is_symlink(),
            'Checkpoint path must be the explicitly selected unaliased artifact')
    require(file_hash(checkpoint) == CHECKPOINT_SHA256, 'Checkpoint hash mismatch before scientific imports')
    run = RunStore(resolver, layout='mounted').create(run_id)
    report = {'schema_version': 1, 'protocol': PROTOCOL, 'status': 'running', 'phase': 'initializing',
              'arm': arm, 'run_id': run_id, 'route': route, 'evidence': packet['identities'],
              'checkpoint_sha256': CHECKPOINT_SHA256, 'step': STORED_STEP, 'declared_image_id': image_id,
              'path_resolved_legacy_values': cfg,
              'implementation': implementation_identity(),
              'declared_source_commit': source_commit, 'legacy_sha256': hashlib.sha256(original).hexdigest(),
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'argv': list(sys.argv),
              'unresolved_provenance': ['Producing checkpoint revision/configuration/split and final-version status.',
                                       'Image build chain, complete native library/extension binary and runtime JIT hashes.',
                                       'Caller image/commit declarations are not independently attested inside the container.'],
              'limits': ['A-based current-code validation regression, not thesis held-out/NVS reproduction.',
                        'Producing revision/configuration/split/final-version status unresolved.',
                        'No DataLoader iteration/workers or training RNG trajectory claim.',
                        'Explicit isolated_geometry=True preserves A three-list split; legacy entry point remains unchanged.',
                        'Full checkpoint pickle is deserialized; only network state loaded strictly.']}
    with run.output('inference.json').open('x') as stream:
        def mark(phase):
            report['phase'] = phase
            write_report(stream, report)
            print('BELL ' + arm + ' ' + phase, flush=True)
        write_report(stream, report)
        try:
            run.output('config.original.yaml').write_bytes(original)
            if arm == 'C':
                run.output('config.envelope.yaml').write_bytes((ROOT / 'configs/experiments/geometry.example.yaml').read_bytes())
            with runtime_environment(run):
                cache = PreprocessingCache(resolver, run, settings)
                try:
                    result = native_arm(cfg, cache, run, checkpoint, packet, mark)
                except Exception:
                    traceback.print_exc()
                    raise
                evidence(metadata, reference, comparison)  # Recheck pinned source and evidence after execution.
                report.update(result)
                require(report['implementation'] == implementation_identity(), 'Regression adapter source changed during execution')
                output_files = [p for directory in ('arrays', 'renders') for p in run.output(directory).rglob('*') if p.is_file()]
                output_files += [run.output('metrics.json')]
                report['output_hashes'] = {p.relative_to(run.root).as_posix(): file_hash(p) for p in output_files}
                report.update(status='complete', phase='complete', validation_calls=1,
                              finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        except Exception as exc:
            report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_report(stream, report)
            raise
        write_report(stream, report)
    return report
