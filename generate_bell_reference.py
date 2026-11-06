"""Researcher-side GPU construction/state_dict signature only; never loads weights."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # Explicit trusted read-only code root, including under python -I.
from infrastructure.signatures import (REFERENCE_PROTOCOL, require, reserve_output,
                                       tensor_catalog, validate_reference, write_report)

CONFIG = ROOT / 'configs/shape/syn/bell.yaml'


def runtime_guards(output):
    require(sys.platform == 'linux' and os.geteuid() != 0, 'Linux non-root runtime required')
    require(ROOT != output and ROOT not in output.parents and output not in ROOT.parents,
            'Code and outputs must be separate')
    require(os.statvfs(str(ROOT)).f_flag & os.ST_RDONLY and os.statvfs(str(CONFIG)).f_flag & os.ST_RDONLY,
            'Read-only source/config mounts required')
    require(os.statvfs('/').f_flag & os.ST_RDONLY, 'Read-only root filesystem required')
    require(not os.statvfs(str(output)).f_flag & os.ST_RDONLY, 'Writable output mount required')
    interfaces = {line.split(':', 1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}
    require(interfaces == {'lo'}, 'No-network runtime required')
    return {'uid': os.geteuid(), 'interfaces': sorted(interfaces), 'source_read_only': True,
            'classification': 'Observed guards, not independent Docker attestation'}


def source_identity(commit, diff_sha256, tree_state):
    paths = [ROOT / 'generate_bell_reference.py', ROOT / 'configs/shape/syn/bell.yaml',
             ROOT / 'assets/bsdf_256_256.bin']
    for directory in ('network', 'dataset', 'utils', 'raytracing', 'colmap', 'infrastructure'):
        paths.extend(sorted((ROOT / directory).glob('*.py')))
    return {'declared_commit': commit, 'declared_diff_sha256': diff_sha256,
            'declared_tree_state': tree_state,
            'files_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in paths},
            'limits': 'Commit/diff/state are caller declarations; file hashes are observed. No full build identity claim.'}


def extract_signature(model):
    """Full real state_dict, including persistent buffers; no tensor values or reductions."""
    state = model.state_dict()
    parameters = {name for name, _ in model.named_parameters()}
    buffers = {name for name, _ in model.named_buffers()}
    entries = []
    for name, value in state.items():
        require(name in parameters or name in buffers, 'Unclassified registered state entry')
        entries.append({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                        'device': str(value.device),
                        'role': 'parameter' if name in parameters else 'persistent_buffer'})
    tensor_catalog(entries, len(entries), reference=True)
    return sorted(entries, key=lambda entry: entry['name'])


def construct_reference(cfg, renderer_class):
    require(type(cfg) is dict and cfg.get('network') == 'shape' and cfg.get('database_name') == 'syn/bell',
            'Expected the actual bell shape configuration')
    # Built-in mode skips dataset/ray construction; module definitions/defaults are unchanged.
    model = renderer_class(cfg, training=False)
    return {'network_tensors': extract_signature(model),
            'effective_renderer_configuration': model.cfg,
            'effective_shader_configuration': model.color_network.cfg}


def native_reference(cfg):
    import torch
    require(torch.__version__ == '1.8.1+cu111', 'Reviewed Torch 1.8.1+cu111 runtime required')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one selected GPU required')
    from network.renderer import NeROShapeRenderer
    result = construct_reference(cfg, NeROShapeRenderer)
    result['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'cuda_compiled': torch.version.cuda,
                         'gpu': torch.cuda.get_device_name(0),
                         'packages': {name: importlib.metadata.version(name)
                                      for name in ('numpy', 'open3d', 'PyYAML', 'nvdiffrast')}}
    return result


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--source-diff-sha256', required=True)
    parser.add_argument('--source-tree-state', choices=['clean', 'dirty'], required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--confirm-isolated-runtime', action='store_true', required=True)
    args = parser.parse_args(argv)
    for value, pattern in [(args.source_commit, r'[a-f0-9]{40}'),
                           (args.source_diff_sha256, r'[a-f0-9]{64}'),
                           (args.image_id, r'sha256:[a-f0-9]{64}')]:
        if re.fullmatch(pattern, value) is None:
            parser.error('Invalid caller-declared source/image identity')
    output = None
    report = {'schema_version': 1, 'protocol': REFERENCE_PROTOCOL, 'status': 'running',
              'phase': 'runtime_guards', 'algorithm': 'maintained-A-bell-geometry',
              'constructor': {'class': 'NeROShapeRenderer', 'training': False},
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'declared_image_id': args.image_id,
              'policy': {'checkpoint_loading': False, 'forward_backward': False, 'optimizer': False,
                         'dataset_initialization': False, 'tensor_values_exported': False}}
    try:
        report['runtime_guards'] = runtime_guards(args.output_root)
        output = reserve_output(args.output_root, 'reference-signature.json')
        write_report(output, report)
        report['phase'] = 'source_configuration'
        report['source'] = source_identity(args.source_commit, args.source_diff_sha256, args.source_tree_state)
        original_bytes = CONFIG.read_bytes()
        original = original_bytes.decode('utf-8')
        import yaml
        cfg = yaml.safe_load(original)
        report['configuration'] = {'path': str(CONFIG), 'original_yaml': original,
                                   'sha256': hashlib.sha256(original_bytes).hexdigest(),
                                   'raw_values': cfg, 'overrides': {},
                                   'seed_policy': 'No seed overrides; only signatures, not initialization values, are compared.'}
        runtime = args.output_root / 'runtime'
        runtime.mkdir()
        for name in ('HOME', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME',
                     'MPLCONFIGDIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                     'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR'):
            directory = runtime / name.lower()
            directory.mkdir()
            os.environ[name] = str(directory)
        report['phase'] = 'construction'
        write_report(output, report)
        previous_cwd = Path.cwd()
        try:
            os.chdir(ROOT)  # Preserve the original constructor's relative BSDF input path.
            result = native_reference(cfg)
        finally:
            os.chdir(previous_cwd)
        report['network_tensors'] = result['network_tensors']
        report['tensor_count'] = len(result['network_tensors'])
        report['state_dict_keys'] = [entry['name'] for entry in result['network_tensors']]
        for name in ('effective_renderer_configuration', 'effective_shader_configuration'):
            report['configuration'][name] = result[name]
        report['runtime'] = result['runtime']
        report['phase'] = 'source_recheck'
        require(source_identity(args.source_commit, args.source_diff_sha256, args.source_tree_state) == report['source'],
                'Source/configuration changed during construction')
        report['phase'] = 'signature_validation'
        completed = dict(report, status='reference_generated', phase='complete',
                         finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        validate_reference(completed)
        report = completed
        write_report(output, report)
        print('Reference generated: ' + str(args.output_root / 'reference-signature.json'))
        return 0
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__,
                      finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if output is not None:
            write_report(output, report)
        print('Reference generation failed at ' + report['phase'] + ' (' + type(exc).__name__ + ')', file=sys.stderr)
        return 1
    finally:
        if output is not None:
            output.close()


if __name__ == '__main__':
    raise SystemExit(main())
