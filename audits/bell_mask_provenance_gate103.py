"""File-only Bell mask provenance audit. No scientific modules or model loads."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'docs/protocols/BELL_AUTHENTICATED_GEOMETRY_GATE10_V4.freeze.json'


def read(path):
    return json.loads(path.read_text())


def identity(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    actual = identity(path)
    require(actual == {k: expected[k] for k in actual}, 'Identity differs: ' + str(path))
    return actual


def png(path, shape):
    with Image.open(path) as image:
        a = np.array(image)
    require(a.shape == shape and a.dtype == np.uint8 and np.isin(a, [0, 255]).all(),
            'Expected binary uint8 PNG: ' + str(path))
    return a


def audit():
    f = read(FREEZE)
    training_root = Path(f['host_output'])
    receipt_path = training_root / 'train/training.json'
    receipt = read(receipt_path)
    require(identity(FREEZE)['sha256'] == receipt['freeze_sha256'], 'Producing freeze differs')
    require(receipt['status'] == 'complete' and receipt['adam_returns'] == 300000
            and receipt['splits'] == f['splits'] and not receipt['held_out_seen_by_trainer'],
            'Authenticated training metadata differs')
    acquisition = Path(f['acquisition'])
    generation_path = acquisition / 'geometry/generation.json'
    require(identity(generation_path)['sha256'] == f['receipts']['generation'],
            'Generating receipt differs')
    generation = read(generation_path)
    require(generation['status'] == 'complete' and generation['scientific_work_complete'],
            'Incomplete reference generation')
    records = {r['path']: r for r in generation['outputs']}
    sources = ['benchmarks/generate_bell_e11.py', 'benchmarks/bell_e11.py']
    for name in sources:
        require(identity(ROOT / name)['sha256'] == generation['source_files'][name],
                'Reference-generation source differs: ' + name)
    original_rows = {r['path']: r for r in read(
        ROOT / 'benchmarks/inputs/bell_mask_identities_e11_v1.json')['files']}
    original_root = Path('/home/isavand/paper/data/GlossySynthetic/bell')
    rows = []
    for camera in map(str, range(128)):
        name = 'masks_train/' + camera + '.png'
        relative = 'datasets/GlossySynthetic/bell/' + name
        mask_path = acquisition / relative
        mask_identity = checked(mask_path, records[relative])
        ray_relative = 'geometry/rays/' + camera + '.npz'
        checked(acquisition / ray_relative, records[ray_relative])
        with np.load(acquisition / ray_relative, allow_pickle=False) as rays:
            hit, depth = rays['hit'], rays['depth']
            require(hit.dtype == bool and hit.shape == depth.shape == (128 * 128,)
                    and np.array_equal(hit, depth < 10), 'Ray-hit definition differs')
        mask = png(mask_path, (128, 128))
        require(np.array_equal(mask.ravel(order='C') > 127, hit), 'Mask differs from ray hits')
        original_identity = checked(original_root / name, original_rows[name])
        png(original_root / name, (800, 800))
        generation_input = next(r for r in generation['inputs']
                                if r['path'] == '/inputs/bell/' + name)
        require(original_identity == {k: generation_input[k] for k in original_identity},
                'Historical admission-mask identity differs')
        role = next(k for k, ids in f['splits'].items() if camera in ids)
        staged = role != 'held_view_nvs'
        if staged:
            stage_name = 'bell/' + name
            require(mask_identity == f['train_files'][stage_name], 'Training freeze mask differs')
            checked(training_root / 'training' / stage_name, mask_identity)
        rows.append(dict(camera=camera, role=role, foreground_pixels=int(hit.sum()),
                         acquisition=mask_identity, historical_admission_mask=original_identity,
                         staged=staged, exact_ray_hit_equality=True))
    staged_ids = {p.stem for p in (training_root / 'training/bell/masks_train').glob('*.png')}
    require(staged_ids == set(f['splits']['train'] + f['splits']['validation']),
            'Unexpected staged mask inventory')
    inspected = [ROOT / n for n in sources] + [ROOT / n for n in (
        'experiments/bell_geometry_gate10_train_v1.py',
        'scientific_sources/geometry_g/dataset/database.py',
        'scientific_sources/geometry_g/network/renderer.py',
        'scientific_sources/geometry_g/network/loss.py')] + [Path(n) for n in (
        '/home/isavand/sam2/sam2.py', '/home/isavand/sam2/click.py',
        '/home/isavand/sam2/convert_jpg.py',
        '/home/isavand/sam2/sam2/sam2_video_predictor.py',
        '/home/isavand/sam2/sam2/build_sam.py',
        '/home/isavand/sam_codes/__pycache__/sam2.cpython-310.pyc')]
    return dict(schema_version=1, audit='bell-mask-provenance-gate103-v1',
        observed_date_UTC='2026-11-27', scope='PNG/NPZ/JSON/source reads only',
        training_receipt=dict(path=str(receipt_path), **identity(receipt_path)),
        training_freeze=dict(path=str(FREEZE), **identity(FREEZE)),
        generation_receipt=dict(path=str(generation_path), **identity(generation_path)),
        scientific_commit=receipt['scientific_commit'], producing_commit=receipt['commit'],
        data_access=receipt['data_access'],
        counts=dict(acquisition_masks=128, exact_ray_hit_matches=128,
                    fitting_masks=122, validation_masks=1, held_out_staged_masks=0),
        semantics='Binary oracle-mesh primary-ray hits; not SAM2 predictions',
        historical_admission_mask_origin='Producing segmentation/rendering procedure unresolved',
        sources={str(p): identity(p) for p in inspected}, masks=rows,
        work=dict(checkpoint_deserializations=0, model_queries=0, native_rays=0,
                  reference_generation=0, optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['counts'], sort_keys=True))
