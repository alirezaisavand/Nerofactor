"""E8R explicit endpoint handling and read-only reference admission; NumPy only."""
import json
from pathlib import Path
import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure.signatures import require, read_report

POLICY = 'float64-uniform-to-float32-open-upper-endpoint-v1'
UPPER = np.nextafter(np.float32(1.), np.float32(0.))


def convert(source):
    a = np.asarray(source)
    require(a.dtype == np.float64 and np.isfinite(a).all() and ((a >= 0)&(a < 1)).all(),
            'Finite original float64 uniforms in [0,1) required')
    original = a.astype(np.float32); result = original.copy(); mask = original == np.float32(1.)
    result[mask] = UPPER
    require(np.isfinite(result).all() and ((result >= 0)&(result < 1)).all(), 'Corrected float32 domain failed')
    require(np.array_equal(result[~mask].view(np.uint32), original[~mask].view(np.uint32)) and
            np.array_equal(result.view(np.uint32) != original.view(np.uint32), mask), 'Unexpected non-endpoint bit change')
    return original, result, mask


def identity(arrays):
    import hashlib
    return {k:dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for k,a in arrays.items()}


def preflight(original_identities, expected=None):
    arrays = {}; corrected_ids = {}; changes = []; values = 0; changed_arrays = 0
    for i in range(6):
        corrected_ids[str(i)] = {}
        for name,start,points in (('fit',6000,24),('evaluation',7000,84)):
            old_columns = {k:[] for k in ('u','v','azimuth')}; new_columns = {k:[] for k in old_columns}
            for seed in range(start+10*i,start+10*i+4):
                rng = np.random.RandomState(seed)
                for k,shape in zip(old_columns,((points,1024),(points,1024),(points,1,1))):
                    source = rng.uniform(size=shape); old,new,mask = convert(source)
                    old_columns[k].append(old); new_columns[k].append(new); values += new.size
                    for index in zip(*np.where(mask)):
                        changes.append(dict(location=i,split=name,seed=seed,array=k,index=list(map(int,index)),
                            float64_value=float(source[index]),old_float32_value=float(old[index]),new_float32_value=float(new[index])))
            old = {k:np.concatenate(v) for k,v in old_columns.items()}; new = {k:np.concatenate(v) for k,v in new_columns.items()}
            require(identity(old) == original_identities[str(i)][name], 'Original E8 tape identity changed')
            for k in old:
                changed_arrays += int(not np.array_equal(old[k].view(np.uint32),new[k].view(np.uint32)))
                require(new[k].shape == old[k].shape and new[k].dtype == np.float32 and np.isfinite(new[k]).all() and
                        ((new[k] >= 0)&(new[k] < 1)).all(), 'Incomplete float32 tape preflight')
            arrays[i,name] = new; corrected_ids[str(i)][name] = identity(new)
    summary = dict(policy=POLICY,upper_float32=float(UPPER),checked_values=values,arrays_checked=36,
        changed_arrays=changed_arrays,corrections=changes,corrected_identities=corrected_ids,
        unchanged_values_bitwise_verified=True,strict_domain_verified=True)
    require(values == 5311008 and changed_arrays == len(changes) == 1 and changes[0] == dict(
        location=5,split='evaluation',seed=7050,array='v',index=[42,982],float64_value=.9999999982641566,
        old_float32_value=1.,new_float32_value=float(UPPER)), 'Unexpected endpoint correction inventory')
    if expected is not None: require(summary == expected, 'Frozen corrected preflight differs')
    return arrays, summary


def admit_references(root, selection, design):
    """Verify accepted artifacts/compatibility; never integrate or measure rank."""
    root = Path(root); report,_ = read_report(root/selection['path'],selection['sha256'])
    require(report['status'] == 'failed' and report['scientific_work_completed'] is False and
            report['completed']['reference_calls'] == 118 and report['completed']['reference_directions'] == 487587840 and
            all(v == 0 for k,v in report['completed'].items() if k not in ('reference_calls','reference_directions')),
            'Expected pre-neural failed E8 attempt required')
    rows = report['references']; require(len(rows) == 6 and [r['location'] for r in rows] == list(range(6)), 'Incomplete reference inventory')
    fine = []
    for i,row in enumerate(rows):
        require(row['truth'] == design['truth'][i] and row['passed'] is True, 'Reference truth/quality mismatch')
        a = obs.read_numeric(root,row['arrays'])
        require(set(a) == {'coarse','fine','absolute_delta','limit'} and all(v.shape == (84,3,3) and v.dtype == np.float64 and
            np.isfinite(v).all() for v in a.values()), 'Invalid independent reference arrays')
        delta = abs(a['coarse']-a['fine']); limit = 2e-6+5e-5*abs(a['fine'])
        require(np.array_equal(delta,a['absolute_delta']) and np.array_equal(limit,a['limit']) and (delta <= limit).all(), 'Reference convergence evidence changed')
        if i < 4:
            gate,_ = read_report(root/row['rank']['path'],row['rank']['sha256'])
            require(gate['explicit_truth'] == row['truth'] and gate['passed'] is True and gate['perturbation_convergence_passed'] is True and
                gate['jacobian_gate_passed'] is True and gate['stable_rank_under_observed_discrepancy'] is True and
                np.asarray(gate['jacobians']).shape == (2,2,72,7) and
                all(s['ratio'] >= 1e-4 for s in gate['steps']) and
                gate['smallest_singular_value'] > max(gate['step_spectral_discrepancy'],*gate['grid_spectral_discrepancies']),
                'Accepted rank evidence incompatible')
        else: require('rank' not in row, 'Unexpected held-out rank measurement')
        fine.append(a['fine'])
    return fine, json.loads(json.dumps(rows))
