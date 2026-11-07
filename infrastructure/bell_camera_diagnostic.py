"""Explicit source-camera diagnostic; no scientific source or split mutation."""

from copy import deepcopy
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

from infrastructure.bell_inference import (CHECKPOINT_SHA256, evidence, file_hash, ordered_split, proposed_split,
    resolve_reference, rng_identity, runtime_environment, scalar_metric, seed_and_merge, strict_load, used_inputs)
from infrastructure.signatures import read_report, require, write_report

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'bell-camera0-validation-diagnostic-v1'
ADMISSION_PATH = 'docs/evidence/bell-camera0-diagnostic-admission.json'
ADMISSION_SHA256 = '0ea7ba3a7ee55159a4d538211bfed0af49d4fe03e93ed0c91827606434fa12b9'
IMAGE58_PATH = 'docs/evidence/bell-inference-regression-success.json'
IMAGE58_SHA256 = '75bd66dc79aad793ae0313d93a8764276c949b62a8998bcf9d3695c918918acc'
DIAGNOSTIC_FILES = ('infrastructure/bell_camera_diagnostic.py', 'infrastructure/bell_camera_comparison.py',
                    'run_bell_camera_diagnostic.py', 'docs/protocols/BELL_CAMERA0_DIAGNOSTIC_V1.md')


def identity():
    return {name: file_hash(ROOT / name) for name in DIAGNOSTIC_FILES}


def admission():
    value, _ = read_report(ROOT / ADMISSION_PATH, ADMISSION_SHA256)
    baseline, _ = read_report(ROOT / IMAGE58_PATH, IMAGE58_SHA256)
    for name, digest in baseline['source']['implementation'].items():
        require(file_hash(ROOT / name) == digest, 'Original image-58 implementation changed: ' + name)
    return value, baseline


def diagnostic_inputs(scene, approved):
    inputs = used_inputs(scene, proposed_split())
    canonical = json.dumps(inputs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(hashlib.sha256(canonical).hexdigest() == approved['inputs']['used_manifest_canonical_sha256'],
            'Original admitted bell inputs changed')
    depth = scene / '0-depth.png'
    require(depth.resolve() == depth and depth.is_file(), 'Missing/aliased camera-0 depth')
    inputs.append({'path': '0-depth.png', 'bytes': depth.stat().st_size, 'sha256': file_hash(depth)})
    return sorted(inputs, key=lambda v: v['path'])


class CameraDatabaseReads:
    """Read-only delegating view; records actual selected getter IDs."""

    def __init__(self, database, source_id):
        self.database, self.source_id, self.reads = database, source_id, []

    def __getattr__(self, name):
        return getattr(self.database, name)

    def _read(self, method, source_id):
        require(type(source_id) is str and source_id == self.source_id, 'Getter requested a different source camera')
        self.reads.append({'method': method, 'source_id': source_id})
        return getattr(self.database, method)(source_id)

    def get_image(self, source_id): return self._read('get_image', source_id)
    def get_pose(self, source_id): return self._read('get_pose', source_id)
    def get_K(self, source_id): return self._read('get_K', source_id)
    def get_mask(self, source_id): return self._read('get_mask', source_id)
    def get_depth(self, source_id): return self._read('get_depth', source_id)


def select_camera(model, source_id, renderer_class, build_info, info_to_torch, torch):
    """Only this external adapter owns the diagnostic ID/table. Model stays intact.

    Original forward/test_step execute on the facade; their downstream methods
    remain bound to the real renderer, preserving network/weight_norm dispatch.
    The facade has no registered parameters, optimizer or child model.
    """
    require(type(source_id) is str and source_id == '0', 'Only explicit source camera string "0" is admitted')
    require(type(model) is renderer_class and not model.training and not model.is_nerf and not model.cfg['val_geometry'],
            'Expected the unmodified evaluated synthetic shape renderer, without extraction')
    for name in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks', '_backward_pre_hooks'):
        require(not getattr(model, name, {}), 'Root renderer hooks require separate review')
    module = torch.nn.modules.module
    for name in ('_global_forward_hooks', '_global_forward_pre_hooks', '_global_backward_hooks', '_global_backward_pre_hooks'):
        require(not getattr(module, name, {}), 'Global module hooks require separate review')
    require(source_id in model.database.get_img_ids(), 'Selected source ID is not in the actual database')
    reads = CameraDatabaseReads(model.database, source_id)
    info = info_to_torch(build_info(reads, [source_id], model.is_nerf), 'cpu')
    methods = [r['method'] for r in reads.reads]
    require(methods == ['get_image', 'get_pose', 'get_K', 'get_mask'], 'Unexpected selected-camera preparation path')
    original_ids, original_info = list(model.test_ids), model.test_imgs_info
    original_train = [str(i) for i in model.train_ids]
    original_nvs = list(model.nvs_ids)

    class SelectedCamera(torch.nn.Module):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, '_renderer', model)  # Do not register/reparent original modules.
            self.database = reads
            self.test_ids = [source_id]
            self.test_imgs_info = info
            self.eval()

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(object.__getattribute__(self, '_renderer'), name)

        def forward(self, data):
            require(type(data) is dict and set(data) == {'index','eval','step'} and
                    type(data.get('index')) is int and data['index'] == 0 and data.get('eval') is True and
                    type(data.get('step')) is int and data['step'] in (149999, 150000), 'Diagnostic evaluation dispatch only')
            return renderer_class.forward(self, data)

        def test_step(self, index, step):
            require(type(index) is int and index == 0 and self.test_ids[index] == source_id,
                    'Iterator index does not map to the selected source camera')
            return renderer_class.test_step(self, index, step)

        def zero_grad(self, *args, **kwargs):
            return model.zero_grad(*args, **kwargs)

        def selection_record(self):
            require(list(model.test_ids) == original_ids and model.test_imgs_info is original_info and
                    [str(i) for i in model.train_ids] == original_train and list(model.nvs_ids) == original_nvs,
                    'Original renderer split/table changed')
            require(all(r['source_id'] == source_id for r in reads.reads), 'Mixed source-camera reads')
            return {'database_source_id': source_id, 'diagnostic_iterator_index': 0,
                    'diagnostic_index_to_source_id': {'0': source_id}, 'getter_reads': list(reads.reads),
                    'original_A_validation_ids': original_ids, 'original_A_tables_unchanged': True}
    return SelectedCamera()


def fingerprints(model):
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                   'sha256': hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()}
            for name, value in model.state_dict().items()}


def native_evaluate(cfg, cache, run, checkpoint, packet, eval_step, source_id, historical, inputs, mark):
    mark('scientific_imports')
    import numpy as np
    import torch
    require(torch.__version__ == '1.8.1+cu111' and torch.version.cuda == '11.1' and sys.version_info[:2] == (3, 8) and
            torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == packet['reference']['runtime']['gpu'],
            'Use the reviewed native runtime and one GPU of the admitted class')
    from train.trainer import Trainer  # Defaults only; never construct an optimizer/trainer.
    from dataset.train_dataset import DummyDataset
    from dataset.database import GlossySyntheticDatabase
    from network.renderer import NeROShapeRenderer, build_imgs_info, imgs_info_to_torch
    from network.metrics import ShapeRenderMetrics
    from utils.base_utils import color_map_backward
    from generate_bell_reference import extract_signature
    from infrastructure.signatures import compare_catalogs, tensor_catalog, validate_reference
    from infrastructure.bell_camera_comparison import historical_comparison
    import imageio

    effective = seed_and_merge(cfg, Trainer.default_cfg, torch, np)
    mark('original_A_initialization')
    DummyDataset(effective['train_dataset_cfg'], True, dataset_dir=effective['dataset_dir'], preprocessing=cache)
    DummyDataset(effective['val_set_list'][0]['cfg'], False, dataset_dir=effective['dataset_dir'], preprocessing=cache)
    model = NeROShapeRenderer(effective, preprocessing=cache, isolated_geometry=True).cuda()
    split = ordered_split(model)
    require(type(model.database) is GlossySyntheticDatabase, 'Unexpected source-camera database implementation')
    actual = tensor_catalog(extract_signature(model), len(model.state_dict()), reference=True)
    require(compare_catalogs(actual, validate_reference(packet['reference']))['result'] == 'exact_structural_match',
            'Live architecture differs from the admitted 137-tensor reference')
    mark('strict_weight_load')
    from inspect_bell_checkpoint import _hash_stream
    with os.fdopen(os.open(str(checkpoint), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        require(_hash_stream(stream) == CHECKPOINT_SHA256, 'Checkpoint hash mismatch before loading')
        state = torch.load(stream, map_location='cpu')  # Explicitly trusted pickle container; optimizer is not restored.
        strict_load(model, state, torch)
        require(float(state.get('best_para')) == 23.787122565451323, 'Stored metric differs from admitted metadata')
        require(_hash_stream(stream) == CHECKPOINT_SHA256, 'Checkpoint changed during loading')
    del state
    model.eval()
    weights = fingerprints(model)
    before_selection = rng_identity(torch, np)
    adapter = select_camera(model, source_id, NeROShapeRenderer, build_imgs_info, imgs_info_to_torch, torch)
    before = rng_identity(torch, np)
    require(before_selection == before, 'Camera selection consumed scientific RNG')
    mark('explicit_camera0_evaluation')
    with torch.no_grad():  # Original forward controls CUDA tensor defaults; internal SDF gradients remain enabled.
        outputs = adapter({'index': 0, 'eval': True, 'step': eval_step})
    torch.cuda.synchronize()
    after = rng_identity(torch, np)
    selection = adapter.selection_record()
    require(selection['getter_reads'][-1] == {'method': 'get_depth', 'source_id': '0'}, 'Camera-0 depth selection not observed')
    require(weights == fingerprints(model), 'Network weights/buffers changed during evaluation')
    prediction, gt, mask = [outputs[k].detach().cpu().numpy() for k in ('ray_rgb', 'gt_rgb', 'gt_mask')]
    require(prediction.shape == gt.shape == (200, 200, 3) and prediction.dtype == gt.dtype == np.float32 and
            mask.shape == (200, 200, 1) and mask.dtype == np.int32 and
            all(np.isfinite(v).all() for v in (prediction, gt, mask)), 'Unexpected diagnostic RGB/mask data')
    mark('unchanged_validation_metrics')
    metrics = ShapeRenderMetrics(effective, output_paths=run)(outputs, {'index': 0, 'eval': True, 'step': eval_step},
                    eval_step, data_index=0, model_name=effective['name']+'-camera0')
    measured = {'psnr': scalar_metric(metrics['psnr'][0]), 'ssim': float(metrics['ssim'][0])}
    require(np.isfinite(measured['ssim']), 'Nonfinite SSIM')
    run.output('arrays').mkdir(exist_ok=True)
    for name, value in [('prediction', prediction), ('ground_truth', gt), ('mask', mask)]:
        np.save(str(run.output('arrays/'+name+'.npy')), value, allow_pickle=False)
    for name, value in [('prediction', prediction), ('ground_truth', gt)]:
        imageio.imwrite(str(run.output('renders/'+name+'.png')), color_map_backward(value))
    comparison = historical_comparison(color_map_backward(gt), measured, historical, run)
    with run.output('metrics.json').open('x') as stream:
        write_report(stream, measured)
    scene = Path(effective['dataset_dir']) / 'bell'
    _, approved = admission()
    require(inputs == diagnostic_inputs(scene, approved), 'Diagnostic input contents changed')
    driver = Path('/proc/driver/nvidia/version')
    return {'selection': selection, 'original_A_splits': split, 'metrics': measured, 'historical_comparison': comparison,
            'effective_trainer': effective, 'effective_renderer': model.cfg, 'effective_shader': model.color_network.cfg,
            'rng_before': before, 'rng_after': after, 'selection_rng_unchanged': True,
            'network_state_fingerprints': weights, 'weights_and_buffers_unchanged': True,
            'auxiliary_outputs': {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'finite': bool(torch.isfinite(v.detach()).all().item()),
                'sha256': hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()}
                for k, v in outputs.items() if torch.is_tensor(v)},
            'preprocessing': cache.bind('syn/bell', scene).identity,
            'runtime': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'driver': driver.read_text() if driver.is_file() else None,
                'torch_cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI), 'default_dtype': str(torch.get_default_dtype()),
                'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads(),
                'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_allow_tf32': getattr(torch.backends.cudnn, 'allow_tf32', None),
                'matmul_allow_tf32': getattr(torch.backends.cuda.matmul, 'allow_tf32', None),
                'packages': {name: importlib.metadata.version(name) for name in
                    ('numpy', 'scipy', 'scikit-image', 'Pillow', 'imageio', 'open3d', 'nvdiffrast', 'PyYAML')}}}


def execute_diagnostic(run_id, source_id, eval_step, datasets, thesis, output, metadata, reference, comparison, image_id, source_commit):
    from infrastructure.cache import PreprocessingCache
    from infrastructure.runs import RunStore
    require(type(source_id) is str and source_id == '0' and type(eval_step) is int and eval_step in (149999, 150000),
            'Explicit camera 0 and admitted evaluation steps only')
    selected, approved = admission()
    packet = evidence(metadata, reference, comparison)
    cfg, resolver, settings, raw, route = resolve_reference(datasets, thesis, output)
    require(resolver.output_root != ROOT and ROOT not in resolver.output_root.parents and
            resolver.output_root not in ROOT.parents, 'Diagnostic outputs overlap source')
    checkpoint = resolver.input_roots['thesis'] / selected['checkpoint']['path']
    historical = resolver.input_roots['thesis'] / selected['historical_montage']['path']
    for path in (ROOT, resolver.input_roots['datasets'], resolver.input_roots['thesis'], checkpoint, historical, metadata, reference, comparison):
        require(os.statvfs(str(path)).f_flag & os.ST_RDONLY, 'Read-only source and input mounts required')
    require(os.geteuid() != 0 and os.statvfs('/').f_flag & os.ST_RDONLY, 'Non-root/read-only rootfs required')
    require({s.split(':',1)[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]} == {'lo'}, 'No-network runtime required')
    require(image_id == approved['runtime']['declared_image_id'], 'Use the admitted immutable image')
    for path, digest in [(checkpoint, CHECKPOINT_SHA256), (historical, selected['historical_montage']['sha256'])]:
        require(path.is_file() and path.resolve() == path and file_hash(path) == digest, 'Missing/aliased/changed admitted artifact')
    require(hashlib.sha256(raw).hexdigest() == approved['configuration']['legacy_sha256'], 'Original bell configuration changed')
    scene = resolver.input_roots['datasets'] / 'GlossySynthetic/bell'
    require(len(list(scene.glob('*.pkl'))) == 128, 'Unexpected bell camera inventory')
    inputs = diagnostic_inputs(scene, approved)
    for file in selected['selected_source']['files']:
        require(file in inputs, 'Camera-0 input differs from diagnostic admission')
    implementation = identity()
    run = RunStore(resolver, layout='mounted').create(run_id)
    report = {'schema_version': 1, 'protocol': PROTOCOL, 'status': 'running', 'phase': 'allocated',
              'run_id': run_id, 'source_id': source_id, 'evaluation_step': eval_step, 'checkpoint_stored_step': 150000,
              'checkpoint_sha256': CHECKPOINT_SHA256, 'evidence': packet['identities'], 'diagnostic_admission_sha256': ADMISSION_SHA256,
              'implementation': implementation, 'declared_image_id': image_id, 'declared_source_commit': source_commit,
              'legacy_sha256': hashlib.sha256(raw).hexdigest(), 'used_inputs': inputs, 'route': route, 'argv': list(sys.argv),
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'limitations': ['Explicit source-camera diagnostic on maintained A; no producing branch/split attribution.',
                 'Researcher visually reports historical camera 0 and current camera 58; unshuffled producing split remains a hypothesis.',
                 'A train/validation/NVS splits remain unchanged; only the external evaluation table selects source 0.',
                 'Historical metric proximity/JPEG differences do not establish producing history.',
                 'Full trusted checkpoint pickle deserialization occurs; optimizer payload is discarded, never restored.',
                 'Root hooks must be absent; original downstream scientific methods and child/weight_norm hooks remain active.',
                 'No historical RNG/optimizer restoration, full NVS, training or image-58 evidence overwrite.']}
    with run.output('camera-diagnostic.json').open('x') as stream:
        def mark(phase):
            report['phase'] = phase; write_report(stream, report); print('CAMERA0 '+phase, flush=True)
        write_report(stream, report)
        try:
            run.output('config.original.yaml').write_bytes(raw)
            with runtime_environment(run):
                try:
                    report.update(native_evaluate(cfg, PreprocessingCache(resolver, run, settings), run, checkpoint, packet,
                                                  eval_step, source_id, historical, inputs, mark))
                except BaseException:
                    traceback.print_exc(); raise
                require(identity() == implementation, 'Diagnostic source/protocol changed')
                evidence(metadata, reference, comparison); admission()
                require(file_hash(historical) == selected['historical_montage']['sha256'], 'Historical montage changed')
                require(not any(run.checkpoints.iterdir()), 'Unexpected checkpoint output')
                sys.stdout.flush(); sys.stderr.flush()
                files = [p for directory in ('arrays','renders') for p in run.output(directory).rglob('*') if p.is_file()]
                files += [run.output('metrics.json'), run.output('config.original.yaml'), run.output('logs/console.log')]
                report['output_hashes'] = {p.relative_to(run.root).as_posix(): file_hash(p) for p in files}
                report.update(status='complete', phase='complete', evaluation_calls=1,
                              finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        except BaseException as exc:
            report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_report(stream, report); raise
        write_report(stream, report)
    return report
