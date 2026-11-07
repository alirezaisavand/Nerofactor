"""JSON-only two-update evidence comparison; no Torch or tensor deserialization."""

import math
from pathlib import Path

from infrastructure.bell_comparison import artifact
from infrastructure.bell_inference import proposed_split
from infrastructure.geometry_training_smoke import PROTOCOL, source_identity
from infrastructure.signatures import read_report, require

# Conservative first protocol: exact tensor/state hashes AND unit-specific scalar
# bounds. Norm closeness alone cannot certify gradient/parameter element equality.
TOLERANCES = {'loss': (1e-7, 1e-5), 'gradient': (1e-7, 1e-5),
              'parameter': (1e-8, 1e-6), 'parameter_delta': (1e-12, 1e-4), 'moment': (1e-12, 1e-5)}
IDENTITIES = ('protocol', 'source', 'declared_source_commit', 'declared_image_id', 'legacy_sha256', 'used_inputs',
              'execution_mode', 'checkpoint_input', 'legacy_total_step', 'requested_optimizer_updates',
              'splits', 'effective_trainer', 'effective_renderer', 'effective_shader', 'effective_losses', 'loader', 'scheduler',
              'initial_parameters', 'initial_buffers', 'initial_optimizer', 'preprocessing', 'runtime')


def _int(value, expected):
    return type(value) is int and value == expected


def _records(value):
    require(type(value) is dict and value, 'Missing parameter/tensor records')
    for record in value.values():
        if record is None:
            continue
        require(type(record) is dict and all(k in record for k in ('sha256', 'shape', 'dtype', 'numel', 'l2', 'mean', 'max_abs')),
                'Incomplete tensor metadata')
        require(type(record['sha256']) is str and len(record['sha256']) == 64 and
                all(c in '0123456789abcdef' for c in record['sha256']), 'Invalid tensor hash')
        require(type(record['shape']) is list and all(type(n) is int and n >= 0 for n in record['shape']) and
                type(record['numel']) is int and math.prod(record['shape']) == record['numel'], 'Invalid tensor shape')
        require(record['dtype'] == 'float32' and all(type(record[k]) in (int, float) and
                math.isfinite(record[k]) for k in ('l2', 'mean', 'max_abs')) and
                record['l2'] >= 0 and record['max_abs'] >= 0, 'Invalid tensor precision/statistics')


def validate_training_report(report, arm):
    require(_int(report.get('schema_version'), 1) and report.get('protocol') == PROTOCOL and
            report.get('status') == 'complete' and report.get('phase') == 'bounded_stop_after_second_optimizer_return' and
            report.get('arm') == arm, 'Incomplete/wrong training arm')
    require(report.get('route') == ('schema-1-PathResolver' if arm == 'C' else 'direct-legacy-FullLoader'), 'Wrong independent configuration route')
    require(type(report.get('run_id')) is str and report['run_id'] and _int(report.get('optimizer_calls_completed'), 2) and
            _int(report.get('requested_optimizer_updates'), 2) and _int(report.get('legacy_total_step'), 300000) and
            report.get('checkpoint_input') is None and report.get('execution_mode') == 'fresh-initialization-two-update-bound' and
            report.get('cleanup_errors') == [], 'Invalid update budget/resume/cleanup state')
    require(all(k in report for k in IDENTITIES) and report['splits'] == proposed_split(), 'Missing identity or wrong ordered split')
    _records(report['initial_parameters']); _records(report['initial_buffers'])
    require(report['initial_optimizer']['state'] == {}, 'Optimizer was not freshly initialized')
    require(report['initial_optimizer']['class'] == 'torch.optim.adam.Adam', 'Unexpected optimizer class')
    require(type(report.get('updates')) is list and len(report['updates']) == 2, 'Expected two complete update records')
    previous = report['initial_parameters']
    previous_optimizer = {}
    for i, update in enumerate(report['updates']):
        require(_int(update.get('step'), i) and _int(update.get('update_index'), i+1) and
                update.get('lr') == (0.0, 5e-4*(1/5000))[i], 'Step/LR semantics changed')
        require(update['parameters_before'] == previous, 'Parameter evidence is not a contiguous update chain')
        for key in ('parameters_before', 'parameters_after', 'gradients_before_clip', 'gradients_after_clip'):
            _records(update[key]); require(update[key].keys() == previous.keys(), 'Parameter/gradient key mismatch')
        require(type(update['losses']) is dict and update['losses'] and
                all(type(x) in (float,int) and math.isfinite(x) for x in update['losses'].values()) and
                type(update['aggregate_loss']) in (float,int) and math.isfinite(update['aggregate_loss']), 'Invalid loss metadata')
        require(update['clip_max_norm'] == 0.7 and update['gradient_l2_before_clip_cpu64'] > 0 and
                0 <= update['gradient_l2_after_clip_cpu64'] <= 0.70001 and
                update['native_clip_returned_norm'] > 0, 'Invalid gradient/clipping evidence')
        require(type(update['parameter_changes']) is dict and update['parameter_changes'].keys() == previous.keys(), 'Missing parameter deltas')
        for name, delta in update['parameter_changes'].items():
            require(type(delta['changed_elements']) is int and 0 <= delta['changed_elements'] <= previous[name]['numel'] and
                    all(type(delta[k]) in (int,float) and math.isfinite(delta[k]) and delta[k] >= 0 for k in ('l2','max_abs')),
                    'Invalid parameter-change metadata')
            same = previous[name]['sha256'] == update['parameters_after'][name]['sha256']
            require(same == (delta['changed_elements'] == 0) and same == (delta['l2'] == 0) and same == (delta['max_abs'] == 0),
                    'Hash and parameter-change evidence disagree')
        changed = sum(x['changed_elements'] for x in update['parameter_changes'].values())
        require(type(update['changed_elements']) is int and update['changed_elements'] == changed and
                ((i == 0 and changed == 0) or (i == 1 and changed > 0)), 'Unexpected zero/nonzero parameter update')
        active = {n for n,v in update['gradients_after_clip'].items() if v is not None}
        state = update['optimizer']['state']
        require(update['optimizer']['class'] == report['initial_optimizer']['class'] and
                len(update['optimizer']['groups']) == len(report['initial_optimizer']['groups']), 'Optimizer identity changed')
        for before, after in zip(report['initial_optimizer']['groups'], update['optimizer']['groups']):
            require({k:v for k,v in before.items() if k != 'lr'} == {k:v for k,v in after.items() if k != 'lr'} and
                    after['lr'] == update['lr'], 'Optimizer hyperparameters/learning rate changed')
        require(active and set(state) == (set(previous_optimizer) | active), 'Missing/extra Adam state')
        for name, values in state.items():
            count = previous_optimizer.get(name, {}).get('step', 0) + int(name in active)
            require(set(values) == {'step','exp_avg','exp_avg_sq','max_exp_avg_sq'} and _int(values['step'], count), 'Invalid Adam step/state fields')
            _records({key:value for key,value in values.items() if key != 'step'})
            require(all(values[key]['shape'] == previous[name]['shape'] for key in values if key != 'step'), 'Wrong Adam state shape')
        require(all(type(update[k]) is dict and update[k] for k in ('rng_before_batch','rng_before_forward','rng_after','memory_before','memory_after')),
                'Missing RNG/memory identities')
        sample = update['sampling']
        require(_int(sample['cursor_before'], i*512) and _int(sample['ray_count'],512) and
                type(sample['pool_size']) is int and sample['pool_size'] > 1024 and _int(update['cursor_after'],(i+1)*512) and
                type(sample['selected']) is dict and sample['selected'], 'Wrong ray sampling cursor')
        previous = update['parameters_after']
        previous_optimizer = state
    return report


def load_training_arm(path, digest, arm):
    report, identity = read_report(path, digest)
    validate_training_report(report, arm)
    require(report['source'] == source_identity(), 'Current protocol/adapter/pinned sources differ from execution')
    hashes = report.get('output_hashes')
    require(type(hashes) is dict and 'config.effective.yaml' in hashes and 'config.legacy.yaml' in hashes and
            'logs/console.log' in hashes, 'Incomplete owned output manifest')
    for name in hashes:
        require(not name.startswith('checkpoints/'), 'Unexpected checkpoint artifact')
        artifact(Path(path).resolve().parent, name, hashes)
    return report, identity


def compare_training_pair(reference, other):
    exact = [k for k in IDENTITIES if reference[k] != other[k]]
    # Initial identities include exact seeded parameter hashes. Scalar diagnostics
    # below expose every recorded training float; all hash/string/integer fields
    # remain exact, including active gradients, weights and Adam moments.
    numeric = []; mismatches = []
    def walk(a, b, path, category):
        if type(a) is dict and type(b) is dict:
            if a.keys() != b.keys():
                mismatches.append(path + ':keys'); return
            for key in a:
                walk(a[key],b[key],path+'/'+key,category)
        elif type(a) is list and type(b) is list:
            if len(a)!=len(b):
                mismatches.append(path+':length'); return
            for i,(left,right) in enumerate(zip(a,b)):walk(left,right,path+'/'+str(i),category)
        elif type(a) is float and type(b) is float:
            absolute=abs(a-b); atol,rtol=TOLERANCES[category]; limit=atol+rtol*abs(a)
            numeric.append([path,a,b,absolute,limit,absolute<=limit])
        elif type(a) != type(b) or a != b:
            mismatches.append(path)
    categories={'losses':'loss','aggregate_loss':'loss','native_clip_returned_norm':'gradient',
                'gradient_l2_before_clip_cpu64':'gradient','gradient_l2_after_clip_cpu64':'gradient',
                'gradients_before_clip':'gradient','gradients_after_clip':'gradient',
                'parameters_before':'parameter','parameters_after':'parameter','parameter_changes':'parameter_delta',
                'optimizer':'moment'}
    for index,(a,b) in enumerate(zip(reference['updates'],other['updates'])):
        for key,category in categories.items():walk(a[key],b[key],str(index)+'/'+key,category)
        for key in ('step','update_index','lr','clip_max_norm','changed_elements','sampling','cursor_after',
                    'rng_before_batch','rng_before_forward','rng_after'):
            if a[key]!=b[key]:mismatches.append(str(index)+'/'+key)
    return {'passed':not exact and not mismatches and all(row[-1] for row in numeric),
            'identity_mismatches':exact,'exact_state_mismatches':mismatches,
            'scalar_columns':['path','reference','other','absolute_difference','limit','passed'],
            'scalar_differences':numeric,'scalar_violation_count':sum(not row[-1] for row in numeric),
            'tensor_policy':'Exact shape/dtype/hash and discrete-state equality; norms alone cannot establish vector equivalence.'}


def compare_training_sequence(reference, repeat, candidate=None, record=None):
    require(reference[0]['run_id'] != repeat[0]['run_id'], 'R and R-repeat must be independent runs')
    result={'schema_version':1,'protocol':PROTOCOL,'result':'blocked','reference':reference[1],'repeat':repeat[1],
            'candidate':None,'candidate_interpreted':False,'tolerances':TOLERANCES}
    result['repeatability']=compare_training_pair(reference[0],repeat[0])
    if not result['repeatability']['passed']:
        result['result']='repeatability_failed';return result
    result['result']='repeatability_passed'
    if record is not None:record(result)
    if candidate is not None:
        value=candidate() if callable(candidate) else candidate
        require(value[0]['run_id'] not in (reference[0]['run_id'],repeat[0]['run_id']), 'Candidate reused a run ID')
        result.update(candidate=value[1],candidate_interpreted=True)
        result['candidate_comparison']=compare_training_pair(reference[0],value[0])
        result['result']='bounded_training_regression_passed' if result['candidate_comparison']['passed'] else 'candidate_mismatch'
    return result
