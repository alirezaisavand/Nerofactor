"""Source- and dataset-bound Bell replication for two new initialization seeds."""
import os
from pathlib import Path

from benchmarks import bell_final_tensor_v1 as final
from infrastructure.signatures import require

b, dense, ROOT = final.b, final.dense, final.ROOT
SEEDS = (6034, 6035)
METHODS = ('K1-Dense', *final.ARMS)
ARMS = tuple(f'{method}-{seed}' for seed in SEEDS for method in METHODS)
PROTOCOL = 'bell-publication-matched-seeds-v1'
FREEZE = 'docs/protocols/BELL_PUBLICATION_SEEDS_V1.freeze.json'
PARENT_SHA = '8c881df7ce05f512a3f784e423bc52e07e1e8a850b028f52de2bd35cc8d72af3'
FOREGROUND_SECONDS = 22200
HISTORICAL_INITIAL_HASHES = {
    'K1-Dense': '80b5015fe7333449c22f33861e00f9006583e609b1cc99a913dab9308cfae4e9',
    'Final-Chart': '2f29a57339f9186c029eb55a40f80cff28e7cdde794eb4b8d2b127a2709d656d',
    'Final-Ambient': 'bd699bc79504aa87f4ff69ee6f7467f09fe0aafe300361bd215f4dd9bab579b3',
}
FILES = (
    'benchmarks/bell_publication_seeds_v1.py',
    'experiments/train_bell_publication_seeds_v1.py',
    'experiments/run_bell_publication_seeds_v1.sh',
    'experiments/analyze_bell_publication_seeds_v1.py',
    'experiments/analyze_donut_heterogeneous_training_v1.py',
    'docs/protocols/BELL_PUBLICATION_SEEDS_V1.md',
    'tests/scientific/test_bell_publication_seeds_v1.py',
)


def __getattr__(name):
    return getattr(final, name)


def seed_method(arm):
    require(arm in ARMS, 'Unknown publication seed history')
    method, seed = arm.rsplit('-', 1)
    return int(seed), method


def policy(arm):
    return dict(final.POLICY, seed=seed_method(arm)[0])


def definition(output):
    from experiments.train_bell_publication_seeds_v1 import trainer_source

    parent = final.freeze(ROOT/final.FREEZE, PARENT_SHA)
    k1 = final.dense_definition()['arms']['K1-Dense']
    originals = dict(parent['arms'], **{'K1-Dense': k1})
    inherited = {key: value for key, value in parent.items() if key not in
                 ('protocol', 'host_output', 'arms', 'order', 'foreground_seconds', 'sources',
                  'photometric_index_sha256', 'trainer_overlay_sha256')}
    arms = {}
    for arm in ARMS:
        seed, method = seed_method(arm)
        recipe = {key: value for key, value in originals[method].items() if key != 'initial'}
        arms[arm] = dict(recipe, seed=seed, method=method)
    return dict(inherited, protocol=PROTOCOL, host_output=str(output), order=list(ARMS), arms=arms,
                foreground_seconds=FOREGROUND_SECONDS, parent_freeze_sha256=PARENT_SHA,
                historical_initial_hashes=HISTORICAL_INITIAL_HASHES,
                photometric_index_sha256={str(seed): dense.index_hash(65419, seed) for seed in SEEDS},
                trainer_overlay_sha256={method: trainer_source(method)[1] for method in METHODS},
                sources={**parent['sources'], **{name: b.digest(ROOT/name) for name in FILES}})


def freeze(path, sha):
    require(path.resolve() == path and b.digest(path) == sha, 'Publication seed freeze differs')
    record = b.read(path)
    out = Path(record['host_output'])
    require(out.resolve() == out and out.parent == Path('/home/isavand/paper/evidence')
            and out.name.startswith('bell-publication-seeds-v1-') and record == definition(out),
            'Publication seed recipe, inputs or sources differ')
    return record


def authorization(f, commit, sha):
    # The historical authorization builder binds paths, caps, and the full ordered recipe inventory.
    return dict(dense.authorization(f, commit, sha), protocol=PROTOCOL, order=list(ARMS),
                foreground_seconds=FOREGROUND_SECONDS, parent_freeze_sha256=PARENT_SHA,
                fresh_seeds=list(SEEDS), tensor_admission=final.AUDIT)


def validate_initial(record, arm, base=None):
    seed, method = seed_method(arm)
    typed = {'K1-Dense': 'A_corrected_core_v1', 'Final-Chart': 'donut_tensor_chart_weak_start_v1',
             'Final-Ambient': 'donut_tensor_ambient_weak_start_v1'}
    expected = {'state_sha256', 'variant', 'seed'}
    if method != 'K1-Dense':
        expected.add('common_base_state_sha256')
    require(set(record) == expected and record['seed'] == seed and record['variant'] == typed[method]
            and len(record['state_sha256']) == 64 and all(c in '0123456789abcdef' for c in record['state_sha256']),
            'Invalid fresh seed initialization identity')
    if method != 'K1-Dense':
        require(base is not None and record['common_base_state_sha256'] == base['state_sha256'],
                'Tensor and seed-matched K1 base weights differ')


def completed(batch, arm, f, commit, sha, auth_sha):
    path = batch/arm
    r = b.read(path/'recovery.json')
    recipe = f['arms'][arm]
    require((batch/f'{arm}-exit-code.txt').read_text().strip() == '0' and r['status'] == 'complete'
            and r['scientific_work_complete'] and r['cleanup_verified'] and r['fresh_seed_initialization_verified']
            and r['arm'] == arm and r['commit'] == commit and r['freeze_sha256'] == sha
            and r['authorization_sha256'] == auth_sha and r['protocol'] == PROTOCOL
            and r['policy'] == policy(arm) and r['counters'] == recipe['work'] and r['optimizer_returns'] == 10000
            and r['dataset_admission_sha256'] == f['dataset']['admission']['sha256']
            and [v['step'] for v in r['saved_endpoints']] == f['steps']
            and r['known_lighting_audit']['learned_module_calls'] == 0,
            'Incomplete or foreign publication seed history: '+arm)
    if recipe['method'] != 'K1-Dense':
        require(r['paired_initial_gate']['passed'] and r['physical_tensor_admission']['passed']
                and r['effective_regularization'] == 'legacy_scalar_energy_specular_white_no_surface_Q_v1',
                'Matched tensor admission differs')
    initial = b.read(b.check_artifact(path/'initial.json', r['outputs']['initial.json']))
    expected = dict(bytes=20480000, sha256=f['photometric_index_sha256'][str(recipe['seed'])])
    require(r['outputs'][dense.STREAM] == expected and initial == r['initial'], 'Seed state or sample receipt differs')
    b.check_artifact(path/dense.STREAM, expected)
    return r
