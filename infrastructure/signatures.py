"""Strict JSON tensor-signature validation/comparison; no native/scientific imports."""

import hashlib
import json
import math
from pathlib import Path
import re

CHECKPOINT_SHA256 = 'd928a03d9eb9f993ab8f4a0384b31cf1c7bdc875316a1c247dcf9722b8b79ccc'
ADMITTED_REPORT_SHA256 = '6108d45e49f502e48b78f71a8dd329b8f37b95fd249a04fb2a386688d2eb6720'
REFERENCE_PROTOCOL = 'bell-geometry-reference-signature-v1'
MAX_JSON_BYTES = 8 * 1024 * 1024
DTYPES = {'torch.' + name for name in ('float16', 'float32', 'float64', 'bfloat16',
          'int8', 'uint8', 'int16', 'int32', 'int64', 'bool', 'complex64', 'complex128')}


class SignatureError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SignatureError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON object key')
        result[key] = value
    return result


def _nonfinite(value):
    raise SignatureError('Nonfinite JSON number')


def _finite_tree(value, depth=0):
    require(depth <= 64, 'JSON nesting exceeds limit')
    if type(value) is float:
        require(math.isfinite(value), 'Nonfinite JSON number')
    elif type(value) is dict:
        for item in value.values():
            _finite_tree(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _finite_tree(item, depth + 1)


def read_report(path, expected_sha256):
    path = Path(path)
    require(type(expected_sha256) is str and re.fullmatch(r'[a-f0-9]{64}', expected_sha256) is not None,
            'Invalid report SHA-256')
    require(path.is_file() and not path.is_symlink(), 'Report must be a regular non-symlink file')
    before = path.stat()
    require(before.st_size <= MAX_JSON_BYTES, 'Report exceeds size limit')
    data = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns), 'Report changed during reading')
    digest = hashlib.sha256(data).hexdigest()
    require(digest == expected_sha256, 'Report SHA-256 mismatch')
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SignatureError('Invalid report JSON') from exc
    require(type(value) is dict, 'Report must be a JSON object')
    _finite_tree(value)
    return value, {'path': str(path.resolve()), 'sha256': digest, 'bytes': len(data)}


def tensor_catalog(entries, count, reference=False):
    require(type(entries) is list and type(count) is int and 0 < count <= 20000 and len(entries) == count,
            'Invalid tensor list/count')
    result = {}
    for entry in entries:
        require(type(entry) is dict, 'Tensor entry must be an object')
        name, shape, dtype = entry.get('name'), entry.get('shape'), entry.get('dtype')
        require(type(name) is str and 0 < len(name) <= 1024 and name not in result,
                'Invalid or duplicate tensor name')
        require(type(shape) is list and len(shape) <= 64 and
                all(type(d) is int and 0 <= d <= 2 ** 63 - 1 for d in shape), 'Invalid tensor shape')
        require(type(dtype) is str and dtype in DTYPES, 'Invalid tensor dtype')
        require(type(entry.get('device')) is str and
                re.fullmatch(r'cpu|cuda(?::[0-9]+)?', entry['device']) is not None, 'Invalid tensor device')
        if reference:
            require(entry.get('role') in {'parameter', 'persistent_buffer'}, 'Unknown reference tensor role')
        result[name] = entry
    return result


def validate_checkpoint(report, required_count=137):
    require(report.get('schema_version') == 1 and type(report.get('schema_version')) is int and
            report.get('protocol') == 'bell-checkpoint-metadata-v1' and
            report.get('status') == 'metadata_extracted' and report.get('phase') == 'complete',
            'Checkpoint report is not a completed schema-1 extraction')
    require(type(report.get('checkpoint')) is dict and report['checkpoint'].get('sha256') == CHECKPOINT_SHA256,
            'Checkpoint identity differs from admitted artifact')
    inspection = report.get('inspection')
    require(type(inspection) is dict, 'Missing inspection')
    catalog = tensor_catalog(inspection.get('network_tensors'), inspection.get('tensor_count'))
    require(len(catalog) == required_count, 'Unexpected admitted checkpoint tensor count')
    require(all(e['device'] == 'cpu' for e in catalog.values()), 'Checkpoint metadata must describe CPU tensors')
    step, best = inspection.get('stored_step'), inspection.get('best_para')
    require(type(step) is dict and step.get('status') == 'recorded' and
            type(step.get('value')) is int and step['value'] >= 0, 'Invalid stored step')
    require(type(best) is dict and best.get('status') == 'recorded' and
            type(best.get('value')) in (int, float) and
            (type(best['value']) is not int or best['value'].bit_length() <= 128) and
            math.isfinite(best['value']), 'Invalid best metric')
    return catalog


def validate_reference(report):
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1 and
            report.get('protocol') == REFERENCE_PROTOCOL and report.get('status') == 'reference_generated' and
            report.get('phase') == 'complete', 'Reference signature is incomplete or unsupported')
    require(report.get('algorithm') == 'maintained-A-bell-geometry' and
            report.get('constructor') == {'class': 'NeROShapeRenderer', 'training': False},
            'Wrong reference algorithm/constructor')
    for field in ('configuration', 'source', 'runtime'):
        require(type(report.get(field)) is dict and bool(report[field]), 'Missing reference provenance')
    cfg, source, runtime = report['configuration'], report['source'], report['runtime']
    require(type(cfg.get('original_yaml')) is str and bool(cfg['original_yaml']) and cfg.get('sha256') ==
            hashlib.sha256(cfg['original_yaml'].encode()).hexdigest(), 'Reference configuration identity mismatch')
    require(type(cfg.get('raw_values')) is dict and cfg['raw_values'].get('database_name') == 'syn/bell' and
            cfg['raw_values'].get('network') == 'shape' and
            cfg.get('overrides') == {} and type(cfg.get('effective_renderer_configuration')) is dict and
            type(cfg.get('effective_shader_configuration')) is dict, 'Missing effective bell configuration')
    require(all(key in cfg['effective_renderer_configuration'] and cfg['effective_renderer_configuration'][key] == value
                for key, value in cfg['raw_values'].items()), 'Raw configuration differs from effective renderer settings')
    files = source.get('files_sha256')
    require(type(files) is dict and all(type(k) is str and type(v) is str and
            re.fullmatch('[a-f0-9]{64}', v) for k, v in files.items()) and
            all(k in files for k in ('network/renderer.py', 'network/field.py', 'configs/shape/syn/bell.yaml')) and
            files['configs/shape/syn/bell.yaml'] == cfg['sha256'], 'Missing/inconsistent reference source hashes')
    require(type(source.get('declared_commit')) is str and re.fullmatch('[a-f0-9]{40}', source['declared_commit']) and
            type(source.get('declared_diff_sha256')) is str and re.fullmatch('[a-f0-9]{64}', source['declared_diff_sha256']) and
            source.get('declared_tree_state') in ('clean', 'dirty'), 'Missing source snapshot declarations')
    require(runtime.get('torch') == '1.8.1+cu111' and runtime.get('cuda_compiled') == '11.1' and
            type(runtime.get('python')) is str and bool(runtime['python']) and
            type(runtime.get('gpu')) is str and bool(runtime['gpu']), 'Missing reviewed reference runtime')
    catalog = tensor_catalog(report.get('network_tensors'), report.get('tensor_count'), reference=True)
    require(report.get('state_dict_keys') == sorted(catalog), 'Reference state_dict key inventory mismatch')
    for name, entry in catalog.items():
        for suffix, other in (('.weight_g', '.weight_v'), ('.weight_v', '.weight_g')):
            if name.endswith(suffix):
                pair = name[:-len(suffix)] + other
                require(pair in catalog and catalog[pair]['role'] == entry['role'] == 'parameter',
                        'Incomplete/misclassified reference weight-normalization pair')
    return catalog


def compare_catalogs(checkpoint, reference):
    missing, unexpected = sorted(reference.keys() - checkpoint.keys()), sorted(checkpoint.keys() - reference.keys())
    shapes, dtypes = [], []
    for name in sorted(reference.keys() & checkpoint.keys()):
        for field, mismatches in [('shape', shapes), ('dtype', dtypes)]:
            if checkpoint[name][field] != reference[name][field]:
                mismatches.append({'name': name, 'checkpoint': checkpoint[name][field],
                                   'reference': reference[name][field]})
    buffers = sorted(n for n, e in reference.items() if e['role'] == 'persistent_buffer')
    groups = sorted({n[:-len('.weight_g')] for n in reference if n.endswith('.weight_g')} |
                    {n[:-len('.weight_v')] for n in reference if n.endswith('.weight_v')})
    norm = [{'prefix': prefix,
             'reference_pair': all(prefix + suffix in reference for suffix in ('.weight_g', '.weight_v')),
             'checkpoint_pair': all(prefix + suffix in checkpoint for suffix in ('.weight_g', '.weight_v'))}
            for prefix in groups]
    return {'result': 'structural_mismatch' if missing or unexpected or shapes or dtypes else 'exact_structural_match',
            'key_sets_equal': not missing and not unexpected,
            'checkpoint_tensor_count': len(checkpoint), 'reference_tensor_count': len(reference),
            'missing_keys': missing, 'unexpected_keys': unexpected,
            'shape_mismatches': shapes, 'dtype_mismatches': dtypes,
            'coverage': {'reference_parameter_count': sum(e['role'] == 'parameter' for e in reference.values()),
                         'reference_persistent_buffers': buffers,
                         'checkpoint_contains_reference_buffers': all(n in checkpoint for n in buffers),
                         'checkpoint_roles': 'Inferred by exact reference names, not recorded independently by metadata inspector',
                         'weight_normalization': norm},
            'limits': ['Only full key/shape/dtype equality is tested; device placement is recorded but not compared.',
                       'A structural match proves neither producing branch, A/G split, tensor values nor scientific reproducibility.']}


def reserve_output(root, filename, inputs=()):
    root = Path(root)
    require(root.is_absolute() and root.is_dir() and root.resolve() == root, 'Output root must be an existing unaliased directory')
    code = Path(__file__).resolve().parents[1]
    require(root != code and code not in root.parents and root not in code.parents, 'Output root overlaps repository')
    for path in inputs:
        parent = Path(path).resolve().parent
        require(root != parent and root not in parent.parents and parent not in root.parents,
                'Output root overlaps an input report directory')
    require(Path(filename).name == filename and filename not in ('', '.', '..'), 'Invalid report filename')
    require(not any(root.iterdir()), 'Output root must be fresh and empty')
    return (root / filename).open('x', encoding='utf-8')


def write_report(stream, value):
    stream.seek(0)
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')
    stream.truncate()
    stream.flush()
