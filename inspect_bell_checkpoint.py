"""Metadata only for one trusted bell checkpoint; run in the documented CPU container.

torch.load in PyTorch 1.8 deserializes pickle, including the checkpoint container's
optimizer payload. Nothing here sandboxes pickle or authenticates its history.
"""

import argparse
from collections import OrderedDict
import datetime
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import stat
import sys

APPROVED_SHA256 = 'd928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc'
CHECKPOINT = Path('/inputs/checkpoint/model_best.pth')
OUTPUT_ROOT = Path('/outputs')
REPORT_NAME = 'checkpoint-metadata.json'
EXPECTED_TORCH = '1.8.1+cu111'
PROTOCOL = 'bell-checkpoint-metadata-v1'
PROVENANCE_KEYS = ('git_commit', 'commit', 'source_commit', 'branch', 'algorithm',
                   'stage', 'schema_version', 'run_id', 'seed', 'random_seed',
                   'config_sha256', 'dataset', 'database_name', 'split_type')
PROVENANCE_CONTAINERS = ('provenance', 'metadata', 'experiment')
MAX_ITEMS = 10000


class InspectionError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise InspectionError(message)


def _type_name(value):
    cls = type(value)
    return (cls.__module__ + '.' + cls.__qualname__)[:200]


def _mapping(value):
    # Avoid arbitrary mapping protocols, representations or recursive traversal.
    return type(value) in (dict, OrderedDict)


def _identity(info):
    return {name: getattr(info, 'st_' + name)
            for name in ('size', 'mtime_ns', 'ctime_ns', 'dev', 'ino')}


def _hash_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _verify_hash(stream):
    digest = _hash_stream(stream)
    _require(digest == APPROVED_SHA256, 'Checkpoint SHA-256 differs from the approved bell artifact')
    return digest


def _plain_scalar(value):
    if value is None or type(value) in (str, bool, int):
        if type(value) is str and len(value) > 512:
            return False, None
        if type(value) is int and value.bit_length() > 128:
            return False, None
        return True, value
    if type(value) is float:
        return (True, value) if math.isfinite(value) else (False, None)
    # Historical best_para may be a NumPy scalar. Do not read tensor values.
    if type(value).__module__ == 'numpy':
        if isinstance(value, numbers.Integral):
            converted = int(value)
            return (True, converted) if converted.bit_length() <= 128 else (False, None)
        if isinstance(value, numbers.Real):
            converted = float(value)
            return (True, converted) if math.isfinite(converted) else (False, None)
    return False, None


def _scalar_field(container, key, integer=False):
    if key not in container:
        return {'status': 'absent'}
    raw = container[key]
    supported, value = _plain_scalar(raw)
    valid = supported and (type(value) is int if integer else type(value) in (int, float))
    if integer and valid:
        valid = value >= 0
    result = {'status': 'recorded' if valid else 'unsupported_or_nonfinite', 'type': _type_name(raw)}
    if valid:
        result['value'] = value
    return result


def summarize_checkpoint(state, torch):
    """Inspect CPU tensor attributes; never execute a model or traverse optimizer state."""
    _require(_mapping(state), 'Checkpoint container must be a dict or OrderedDict')
    _require(len(state) <= MAX_ITEMS, 'Too many checkpoint fields')
    _require(all(type(k) is str and len(k) <= 512 for k in state), 'Invalid checkpoint field names')
    top = [{'name': k, 'type': _type_name(state[k])} for k in sorted(state)]
    network = state.get('network_state_dict')
    _require(_mapping(network) and 0 < len(network) <= MAX_ITEMS,
             'Missing, empty or unsupported network_state_dict')
    _require(all(type(k) is str and len(k) <= 512 for k in network), 'Invalid network tensor names')
    tensors = []
    for key in sorted(network):
        value = network[key]
        _require(type(value) in (torch.Tensor, torch.nn.Parameter), 'Network state contains a nonstandard tensor')
        _require(value.device.type == 'cpu', 'Network state contains a non-CPU tensor')
        tensors.append({'name': key, 'shape': list(value.shape), 'dtype': str(value.dtype),
                        'device': str(value.device)})
    provenance = []
    containers = [('', state)]
    for key in PROVENANCE_CONTAINERS:
        if key in state and _mapping(state[key]):
            _require(len(state[key]) <= MAX_ITEMS, 'Too many provenance fields')
            containers.append((key + '.', state[key]))
    for prefix, container in containers:
        for key in PROVENANCE_KEYS:
            if key in container:
                value = container[key]
                supported, scalar = _plain_scalar(value)
                entry = {'path': prefix + key, 'type': _type_name(value),
                         'status': 'recorded' if supported else 'omitted_nonprimitive_or_oversize'}
                if supported:
                    entry['value'] = scalar
                provenance.append(entry)
    return {'top_level_fields': top, 'network_tensors': tensors, 'tensor_count': len(tensors),
            'stored_step': _scalar_field(state, 'step', integer=True),
            'best_para': _scalar_field(state, 'best_para'), 'primitive_provenance': provenance,
            'limits': ['Provenance keys are a bounded allowlist; absence from this report is not absence from the artifact.',
                       'No tensor values or optimizer contents are reported; the full container was deserialized.',
                       'Tensor metadata is not architecture compatibility, producing-history or inference evidence.']}


def _mounts(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 7 and ' - ' in line:
            result[fields[4]] = set(fields[5].split(','))
    return result


def _runtime_guards(script):
    """Defense-in-depth checks, not a sandbox or proof of Docker policy/authenticity."""
    _require(sys.platform == 'linux' and os.geteuid() != 0, 'Linux non-root execution required')
    _require(OUTPUT_ROOT.is_dir() and OUTPUT_ROOT.resolve() == OUTPUT_ROOT, 'Invalid output root')
    _require(CHECKPOINT.resolve() == CHECKPOINT and not CHECKPOINT.is_symlink(), 'Checkpoint path alias rejected')
    mounts = _mounts(Path('/proc/self/mountinfo').read_text())
    for path in (script, CHECKPOINT):
        _require(str(path) in mounts and 'ro' in mounts[str(path)], 'Explicit read-only file mounts required')
        _require(os.statvfs(str(path)).f_flag & os.ST_RDONLY, 'Input/source filesystem must be read-only')
    _require('/' in mounts and 'ro' in mounts['/'], 'Read-only root filesystem required')
    _require('/outputs' in mounts and 'rw' in mounts['/outputs'], 'Separate writable /outputs mount required')
    interfaces = {line.split(':', 1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}
    _require(interfaces == {'lo'}, 'Only the isolated loopback interface may be visible')
    devices = list(Path('/dev').glob('nvidia*')) + [p for p in (Path('/dev/dri'), Path('/dev/kfd')) if p.exists()]
    _require(not devices, 'GPU device nodes must not be exposed')
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void',
             'Explicit CPU-only device environment required')
    _require(not any(OUTPUT_ROOT.iterdir()), 'Output root must be fresh and empty')
    return {'non_root_uid': os.geteuid(), 'interfaces': sorted(interfaces), 'gpu_nodes': [],
            'read_only_file_mounts': [str(script), str(CHECKPOINT)], 'root_read_only': True,
            'output_mount': '/outputs', 'classification': 'Observed runtime guards, not independent Docker attestation'}


def _load_metadata(stream, torch):
    # PyTorch 1.8 API: deliberately no weights_only argument or fallback.
    state = torch.load(stream, map_location='cpu')
    return summarize_checkpoint(state, torch)


def _import_torch():
    import torch
    return torch


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, choices=[str(CHECKPOINT)])
    parser.add_argument('--output-root', required=True, choices=[str(OUTPUT_ROOT)])
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--execute', required=True, action='store_true')
    parser.add_argument('--acknowledge-trusted-checkpoint', required=True, action='store_true')
    parser.add_argument('--confirm-isolated-runtime', required=True, action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', args.image_id):
        parser.error('--image-id must be an immutable sha256 image ID')
    report = None
    output = None
    phase = 'runtime_guards'
    try:
        script = Path(__file__).resolve()
        guards = _runtime_guards(script)
        phase = 'hash_verification'
        descriptor = os.open(str(CHECKPOINT), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode), 'Checkpoint must be a regular file')
            digest = _verify_hash(stream)
            _require(_identity(before) == _identity(os.fstat(stream.fileno())), 'Input changed during hashing')
            report = {'schema_version': 1, 'protocol': PROTOCOL, 'status': 'running',
                      'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'checkpoint': {'path': str(CHECKPOINT), 'sha256': digest, **_identity(before)},
                      'inspector_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
                      'declared_image_id': args.image_id, 'runtime_guards': guards,
                      'trust': 'Researcher confirmed ownership/trust for the exact approved hash; producing revision/final-version status unknown.',
                      'classification': 'Metadata extraction only; architecture compatibility and producing history unverified',
                      'harness_policy': {'model_construction': False, 'model_weight_loading': False,
                                         'optimizer_restoration': False, 'inference': False},
                      'pickle_deserialization_requested': True,
                      'checkpoint_deserialization_attempted': False, 'inspection': None}
            # Exclusive creation precedes Torch import/load; retain partial failure evidence.
            output = (OUTPUT_ROOT / REPORT_NAME).open('x', encoding='utf-8')
            phase = 'runtime_setup'
            runtime = OUTPUT_ROOT / 'runtime'
            runtime.mkdir()
            for name in ('HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'TORCH_HOME', 'TMPDIR', 'TEMP', 'TMP'):
                target = runtime / name.lower()
                target.mkdir()
                os.environ[name] = str(target)
            report['phase'] = phase
            json.dump(report, output, indent=2, allow_nan=False)
            output.flush()
            phase = 'torch_import'
            torch = _import_torch()
            _require(torch.__version__ == EXPECTED_TORCH, 'Inspection requires the reviewed Torch 1.8.1+cu111 runtime')
            report['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'executable': sys.executable}
            phase = 'checkpoint_deserialization'
            report['checkpoint_deserialization_attempted'] = True
            report['inspection'] = _load_metadata(stream, torch)
            phase = 'input_recheck'
            _require(_identity(before) == _identity(os.fstat(stream.fileno())) and _hash_stream(stream) == digest,
                     'Input changed during inspection')
            report.update(status='metadata_extracted', phase='complete',
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        output.seek(0)
        json.dump(report, output, indent=2, allow_nan=False)
        output.truncate()
        print('Metadata extracted: /outputs/' + REPORT_NAME + '; compatibility and producing history remain unverified.')
        return 0
    except Exception as exc:
        # Arbitrary exception text/repr may contain tensor values or private payloads.
        if report is not None and output is not None:
            report.update(status='failed', phase=phase, error_type=_type_name(exc),
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            report['inspection'] = None
            output.seek(0)
            json.dump(report, output, indent=2, allow_nan=False)
            output.truncate()
        print('Inspection failed at ' + phase + ' (' + _type_name(exc) + '); no retry or fallback.', file=sys.stderr)
        return 1
    finally:
        if output is not None:
            output.close()


if __name__ == '__main__':
    raise SystemExit(main())
