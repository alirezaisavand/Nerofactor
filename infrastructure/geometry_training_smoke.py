"""Explicit fresh two-update geometry execution; scientific imports are lazy."""

from copy import deepcopy
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from infrastructure.bell_inference import file_hash, resolve_reference, runtime_environment, used_inputs, proposed_split
from infrastructure.signatures import read_report, require, write_report

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'bell-two-update-training-v1'
BASELINE = 'ef7e6b270137c1315c1ef01f303af910ee1dcbaf'
ADMISSION = 'docs/evidence/bell-inference-regression-success.json'
ADMISSION_SHA256 = '75bd66dc79aad793ae0313d93a8764276c949b62a8998bcf9d3695c918918acc'
STRUCTURE = 'docs/evidence/bell-structural-match.json'
STRUCTURE_SHA256 = 'ea1df7ab996c6b80e0b586e9e002e81e3597cd6561e700543ebb72c81b45fa8d'
NEW_FILES = ('infrastructure/bounded_training.py', 'infrastructure/training_probe.py',
             'infrastructure/geometry_training_smoke.py', 'infrastructure/training_comparison.py',
             'run_geometry_training_smoke.py', 'compare_geometry_training_smoke.py',
             'docs/protocols/GEOMETRY_TWO_UPDATE_V1.md')


def source_identity():
    admitted, _ = read_report(ROOT / ADMISSION, ADMISSION_SHA256)
    structure, _ = read_report(ROOT / STRUCTURE, STRUCTURE_SHA256)
    protected = {**structure['reference_source']['files_sha256'], **admitted['source']['implementation']}
    for path, digest in protected.items():
        require(file_hash(ROOT / path) == digest, 'Pinned scientific/infrastructure source changed: ' + path)
    return {'protected': protected, 'smoke': {name: file_hash(ROOT / name) for name in NEW_FILES},
            'admission_sha256': ADMISSION_SHA256, 'structural_manifest_sha256': STRUCTURE_SHA256}


def resolve_plan(arm, datasets, thesis, output):
    """Only storage is shared. R never calls C's schema loader/validator."""
    require(arm in ('R', 'R-repeat', 'C'), 'Unknown arm')
    from infrastructure.paths import InputSelection, PathResolver
    from infrastructure.geometry import GeometryPlan, validate_geometry
    if arm == 'C':
        from infrastructure.config import load_config
        config = load_config(ROOT / 'configs/experiments/geometry.example.yaml', config_root=ROOT / 'configs')
        require(config.original.get('overrides') == {} and config.inputs.checkpoint is None and config.inputs.mesh is None,
                'No scientific overrides, weights or mesh permitted')
        plan = validate_geometry(config, PathResolver({'datasets': datasets, 'thesis': thesis}, output))
        route = 'schema-1-PathResolver'
    else:
        import yaml
        cfg, resolver, settings, raw, route = resolve_reference(datasets, thesis, output)
        original = yaml.load(raw, Loader=yaml.FullLoader)
        config = SimpleNamespace(legacy=original, preprocessing=settings, diagnostics=[])
        selected = resolver.resolve_inputs(InputSelection('GlossySynthetic'))
        identity = {'schema_version': 1, 'stage': 'geometry', 'reference': 'direct-legacy',
                    'scientific_configuration': original, 'resolved_dataset': str(selected.dataset),
                    'preprocessing': settings.cache}
        plan = GeometryPlan(config, resolver, selected, cfg, identity, raw, raw)
    require(plan.selected.checkpoint is None and plan.selected.mesh is None, 'Fresh geometry only')
    require(plan.pipeline_config['database_name'] == 'syn/bell', 'Only the admitted bell scene')
    return plan, route


def runtime_guard(plan, image_id, admitted):
    require(os.geteuid() != 0 and os.statvfs('/').f_flag & os.ST_RDONLY, 'Non-root/read-only rootfs required')
    for path in (ROOT, plan.selected.dataset, plan.resolver.input_roots['thesis']):
        require(os.statvfs(str(path)).f_flag & os.ST_RDONLY, 'Read-only source/input mounts required')
    require(not any(plan.resolver.input_roots['thesis'].iterdir()), 'Use an empty read-only thesis placeholder; no historical weights mounted')
    require(plan.resolver.output_root.is_dir() and not os.statvfs(str(plan.resolver.output_root)).f_flag & os.ST_RDONLY,
            'Existing writable output mount required')
    interfaces = {s.split(':', 1)[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]}
    require(interfaces == {'lo'}, 'No-network runtime required')
    require(image_id == admitted['runtime']['declared_image_id'], 'Use the admitted immutable image')
    require(plan.resolver.output_root != ROOT and ROOT not in plan.resolver.output_root.parents and
            plan.resolver.output_root not in ROOT.parents, 'Outputs overlap code')


def runtime_identity(torch):
    driver = Path('/proc/driver/nvidia/version')
    return {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'torch_cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            'driver': driver.read_text() if driver.is_file() else None,
            'packages': {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'scikit-image', 'opencv-python', 'open3d', 'nvdiffrast', 'tensorboardX', 'PyYAML')},
            'default_dtype': str(torch.get_default_dtype()), 'num_threads': torch.get_num_threads(),
            'num_interop_threads': torch.get_num_interop_threads(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': getattr(torch.backends.cudnn, 'allow_tf32', None),
            'matmul_allow_tf32': getattr(torch.backends.cuda.matmul, 'allow_tf32', None)}


def native_smoke(experiment, report, persist):
    report['phase'] = 'scientific_imports'; persist()
    import numpy as np
    import torch
    require(torch.__version__ == '1.8.1+cu111' and torch.version.cuda == '11.1' and
            torch.cuda.is_available() and torch.cuda.device_count() == 1 and sys.version_info[:2] == (3, 8) and
            torch.cuda.get_device_name(0) == 'NVIDIA A100 80GB PCIe',
            'Requires the reviewed Python 3.8/Torch 1.8.1 CUDA 11.1 runtime and one selected A100 80GB PCIe')
    from train.trainer import Trainer
    require(Path(Trainer.run.__code__.co_filename).resolve() == ROOT / 'train/trainer.py',
            'Imported Trainer is outside the pinned source')
    from infrastructure.bounded_training import TwoUpdateTrace
    from infrastructure.training_probe import TrainingProbe
    report['phase'] = 'trainer_constructor'; persist()
    trainer = Trainer(deepcopy(experiment.plan.pipeline_config), experiment=experiment)
    report['runtime'] = runtime_identity(torch)
    observer = TrainingProbe(trainer, torch, np, report, persist)
    controller = TwoUpdateTrace(trainer, observer, ROOT / 'train/trainer.py')
    controller.run()
    require(controller.completed == 2 and len(report['updates']) == 2, 'Incomplete two-update evidence')
    require(not any(experiment.run.checkpoints.iterdir()), 'No checkpoint expected before its natural interval')
    report['preprocessing'] = experiment.preprocessing.bind('syn/bell', experiment.plan.selected.dataset / 'bell').identity
    return trainer


def execute_smoke(arm, run_id, datasets, thesis, output, image_id, source_commit):
    from infrastructure.geometry import prepare_geometry
    require(arm in ('R', 'R-repeat', 'C'), 'Unknown arm')
    source = source_identity()
    admitted, _ = read_report(ROOT / ADMISSION, ADMISSION_SHA256)
    plan, route = resolve_plan(arm, datasets, thesis, output)
    runtime_guard(plan, image_id, admitted)
    # Hashing/configuration validation creates no model, seed, optimizer or cache.
    require(len(list((plan.selected.dataset / 'bell').glob('*.pkl'))) == 128, 'Unexpected bell camera inventory')
    inputs = used_inputs(plan.selected.dataset / 'bell', proposed_split())
    canonical = json.dumps(inputs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(hashlib.sha256(canonical).hexdigest() == admitted['inputs']['used_manifest_canonical_sha256'],
            'Bell input contents differ from the admitted inference baseline')
    require(plan.legacy_bytes and hashlib.sha256(plan.legacy_bytes).hexdigest() == admitted['configuration']['legacy_sha256'],
            'Original bell YAML changed')
    experiment = prepare_geometry(plan, run_id, mode='new', layout='mounted')
    report = {'schema_version': 1, 'protocol': PROTOCOL, 'status': 'running', 'phase': 'allocated',
              'arm': arm, 'run_id': run_id, 'route': route, 'optimizer_calls_completed': 0, 'updates': [],
              'source': source, 'declared_source_commit': source_commit, 'declared_image_id': image_id,
              'legacy_sha256': hashlib.sha256(plan.legacy_bytes).hexdigest(), 'used_inputs': inputs,
              'execution_mode': 'fresh-initialization-two-update-bound', 'checkpoint_input': None,
              'legacy_total_step': 300000, 'requested_optimizer_updates': 2, 'argv': list(sys.argv),
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'limitations': ['Current A infrastructure smoke, not historical thesis training reproduction.',
                  'Step 0 LR is zero: Adam moments/step advance but parameters are expected unchanged.',
                  'External CPython line observer aborts the original loop immediately after Adam call 2.',
                  'Diagnostics synchronize/copy detached tensors; wall time/memory differ from an unobserved run.',
                  'Dummy sample indices/worker RNG states are not exported; native DataLoader behavior is unchanged.',
                  'No later loss schedules, natural validation/checkpoint interval or convergence tested.',
                  'Image/source declarations are not independent launch attestation; binary/JIT closure incomplete.']}
    with experiment.run.output('training-smoke.json').open('x') as stream:
        def persist():
            write_report(stream, report)
        persist()
        try:
            with runtime_environment(experiment.run):
                try:
                    native_smoke(experiment, report, persist)
                except BaseException:
                    traceback.print_exc()
                    raise
                require(inputs == used_inputs(plan.selected.dataset / 'bell', report['splits']), 'Input content changed')
                require(source == source_identity(), 'Pinned source/adapter/protocol changed during execution')
                require(not any(experiment.run.checkpoints.iterdir()), 'Unexpected checkpoint output')
                sys.stdout.flush(); sys.stderr.flush()
                report['output_hashes'] = {p.relative_to(experiment.run.root).as_posix(): file_hash(p)
                    for p in experiment.run.root.rglob('*') if p.is_file()
                    and p != experiment.run.output('training-smoke.json') and 'cache' not in p.relative_to(experiment.run.root).parts
                    and 'tmp' not in p.relative_to(experiment.run.root).parts and p.name != '.experiment.json'}
                report.update(status='complete', phase='bounded_stop_after_second_optimizer_return',
                              finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        except BaseException as exc:
            report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            persist()
            raise
        persist()
    return report
