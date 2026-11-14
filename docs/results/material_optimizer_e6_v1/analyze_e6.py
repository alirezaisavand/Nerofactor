"""E6 offline verification and figures; no Torch/scientific module imports."""
import argparse
import ast
import csv
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name == 'torch' or name.startswith(('torch.', 'experiments.')):
            raise RuntimeError('Offline analysis forbids '+name)


sys.meta_path.insert(0, NoScience())
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import require, read_report

RUN = 'material-e6-1d760020180f45248f9e0ae41bb6b376'
FREEZE = 'ca53a5ec9bcb8279ee7d8f5c84a769ff5aed1eebdc550eb955657c09ceeac8f4'
REPORT = 'fbcfa7a1c8abb974a985e32c3c66000ef89418005b598e6b4fb9f73b20f9da6c'
SEAL = '4b5cf4946ea8fcb565dafb312296030f8a0568652b880b041fc1f2537c5e45ec'
RECEIPT = '05f39b5e3bebe03d457941f028a22961014c6287394da52134c9ab210b1d36ef'
COMMIT = '00d795227faba37642662f2df9dd7aa02d0b98f8'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(p): return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def numeric_metrics(plan):
    name = 'experiments/run_known_material_recovery_v1.py'; assert sha(ROOT/name) == plan['source_sha256'][name]
    tree = ast.parse((ROOT/name).read_bytes())
    cfg = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in n.targets))
    success = next(ast.literal_eval(v) for k, v in zip(cfg.keys, cfg.values) if k.value == 'success')
    ns = dict(np=np, require=require, SETTINGS=dict(success=success))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('radiance_errors', 'assess_success')]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT/name), 'exec'), ns)
    return ns


def components(a, b):
    delta = a.astype(np.float64)-b; d, s, t = (delta[:, i] for i in range(3))
    return dict(component_MSE=[float(np.mean(v*v)) for v in (d, s, t)],
        component_squared_error=float(np.sum(d*d)+np.sum(s*s)), total_squared_error=float(np.sum(t*t)),
        diffuse_specular_inner_product=float(np.sum(d*s)),
        normalized_residual_inner_product=float(np.sum(d*s)/max(np.linalg.norm(d)*np.linalg.norm(s), 1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--evidence-parent', type=Path, required=True); parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); parent = args.evidence_parent; raw = parent/RUN; out = args.output_dir
    assert raw.is_absolute() and raw.resolve() == raw and out.is_absolute() and out.resolve() == out and out.parent == parent and not out.exists()
    plan, _ = read_report(ROOT/'docs/protocols/MATERIAL_OPTIMIZER_E6_V1.freeze.json', FREEZE)
    for p, h in plan['source_sha256'].items(): assert sha(ROOT/p) == h
    for p, h in plan['input_sha256'].items(): assert sha(p) == h
    report, _ = read_report(raw/'result.json', REPORT); seal, _ = read_report(raw/'finalization/seal.json', SEAL)
    receipt, _ = read_report(parent/(RUN+'.process.json'), RECEIPT)
    assert report['scientific_work_completed'] and report['cleanup_verified'] and report['terminal_integrity_verified'] and report['initial_gates_passed']
    assert seal['report']['sha256'] == REPORT and seal['outcome'] == 'complete' and receipt['process_exit_code'] == receipt['retry_attempts'] == 0
    for k in ('authorization', 'launch', 'console'): assert identity(Path(receipt[k]['path'])) == receipt[k]
    launch = json.loads(Path(receipt['launch']['path']).read_text()); assert launch['commit'] == COMMIT and launch['freeze_sha256'] == FREEZE and launch['foreground_supervisor']
    rp = ROOT/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'; assert sha(rp) == plan['inputs']['accepted_3D1_SHA256']
    reference = json.loads(rp.read_bytes()); metrics = numeric_metrics(plan); scales = np.asarray(plan['scales']); policy = plan['LM_policy']
    gamma72 = 72*np.finfo(np.float32).eps/(1-72*np.finfo(np.float32).eps)
    endpoints = []; trials = []; trajectories = []; spectra = []; groups = []; snapshots = 0; new_fd = []
    for spec, case in zip(plan['cases'], report['cases']):
        name = case['case']; assert name == spec['case'] and case['initial_parity'] and case['initial_physical'] == spec['physical']
        info = next(v for v in reference['materials'] if v['material'] == spec['material']); truth = np.asarray(info['truth'], np.float64)
        target = np.asarray(info['targets']['fine'], np.float64); fit = target[:24, 2].astype(np.float32)
        assert hashlib.sha256(target.tobytes()).hexdigest() == plan['inputs']['targets'][spec['material']]['fine_sha256']
        lm, adam = case['LM'], case['Adam']; current = np.asarray(spec['physical'], np.float32); loss = case['initial_loss']; damping = .01; accepted = queried = 0
        c = 2; path = [dict(case=name, method='LM', cost=c, loss=loss, physical=current.tolist(), Q=diagnostic.recovery_errors(current, truth)['Q_relative_frobenius'])]
        for h in lm['history']:
            arrays = obs.read_numeric(raw, h['arrays']); assert arrays['J_scaled'].shape == (72, 7) and arrays['J_scaled'].dtype == np.float32
            assert arrays['physical'].tolist() == h['physical_before'] == current.tolist() and arrays['predictions'].shape == (24, 3, 3)
            check_loss = float(np.mean((arrays['predictions'][:, 2]-fit)**2)); assert abs(check_loss-loss) <= 2*gamma72*loss
            j = arrays['J_scaled'].astype(np.float64)/math.sqrt(72)
            residual = (arrays['predictions'][:, 2]-fit).astype(np.float64).reshape(72)/math.sqrt(72)
            hh, g = j.T@j, j.T@residual; _, sv, vt = np.linalg.svd(j, full_matrices=False)
            np.testing.assert_array_equal(sv, h['singular_values']); np.testing.assert_array_equal(2*g, h['scaled_gradient'])
            c += 42
            if h['outer'] == 1 and spec['new_directional_FD']:
                fd = arrays['initial_FD']; direction = arrays['initial_direction']; error = np.linalg.norm(fd-j@direction, axis=1); limit = 1e-4+.05*np.linalg.norm(j@direction)
                np.testing.assert_array_equal(error, arrays['initial_FD_errors']); assert np.all(error <= limit)
                new_fd.append(dict(case=name, errors=error.tolist(), limit=float(limit))); c += 8
            spectra.append(dict(case=name, outer=h['outer'], singular_values=sv.tolist(), condition_ratio=float(sv[0]/sv[-1]) if sv[-1] > 0 else None,
                scaled_gradient_L2=float(np.linalg.norm(2*g)), physical=current.tolist()))
            for t in h['trials']:
                assert t['damping_before'] == damping
                a = hh+damping*np.eye(7); raw_delta = np.asarray(t['raw_scaled_delta']); system_error = np.linalg.norm(a@raw_delta+g)
                assert system_error <= 1e-10*max(np.linalg.norm(a)*np.linalg.norm(raw_delta)+np.linalg.norm(g), 1e-30)
                candidate = np.asarray(t['physical'], np.float32); step = (candidate.astype(float)-current.astype(float))/scales
                np.testing.assert_array_equal(step, t['represented_scaled_delta'])
                predicted = float(-2*g@step-step@hh@step)
                np.testing.assert_allclose(predicted, t['predicted_reduction'], atol=1e-26, rtol=1e-12)
                assert np.isfinite(candidate).all() and np.all(candidate[:2] >= np.float32(.04)) and np.all(candidate[:2] <= 1) and np.all((candidate[3:] >= 0) & (candidate[3:] <= 1))
                if t['queried']:
                    queried += 1; c += 2; actual = loss-t['loss']; ratio = actual/predicted
                    floor = 16*np.finfo(np.float32).eps*max(loss, t['loss'], 1e-12)
                    take = bool(ratio > .1 and actual > floor)
                    assert actual == t['actual_reduction'] and ratio == t['ratio'] and floor == t['reduction_floor'] and take == t['accepted']
                else:
                    assert not t['accepted'] and t['loss'] is None and (predicted <= 0 or np.array_equal(candidate, current)); actual = None; ratio = None
                if t['accepted']:
                    current = candidate; loss = t['loss']; accepted += 1
                    damping = max(1e-8, min(1e8, damping/3 if ratio > .75 else damping*2 if ratio < .25 else damping))
                else: damping = max(1e-8, min(1e8, damping*10))
                assert damping == t['damping_after']
                trials.append(dict(case=name, outer=h['outer'], accepted_index=accepted, cost=c, **t))
                path.append(dict(case=name, method='LM', cost=c, loss=loss, physical=current.tolist(), Q=diagnostic.recovery_errors(current, truth)['Q_relative_frobenius']))
            assert accepted == h['accepted_updates'] and current.tolist() == h['physical_after'] and loss == h['loss_after']
        assert accepted == lm['accepted_updates'] < 20 and lm['status'] == 'censored_algorithm' and lm['termination'] == 'four_trial_limit'
        assert len(lm['history'][-1]['trials']) == 4 and not any(t['accepted'] for t in lm['history'][-1]['trials'])
        assert not case['matched_accepted_update_available'] and not lm['matched_update_available']
        c += 11; assert c == lm['render_cost']
        assert current.tolist() == lm['terminal']['physical'] and loss == lm['terminal']['loss']
        path.append(dict(case=name, method='LM', cost=c, loss=loss, physical=current.tolist(), Q=diagnostic.recovery_errors(current, truth)['Q_relative_frobenius']))
        lw = case['LM_work']; aw = case['Adam_work']; n = adam['updates']; events = lw['events']
        assert lw['done'] == lw['attempts'] == dict(fit_forward=2+7*len(lm['history'])+queried+4*int(spec['new_directional_FD']), reverse=1+14*len(lm['history']), external=1, updates=0)
        assert events['trials'] == events['solves_attempted'] == events['solves_completed'] == sum(len(h['trials']) for h in lm['history'])
        assert events['accepted_steps'] == accepted and events['rejected_trials'] == events['trials']-accepted
        assert aw['done'] == aw['attempts'] == dict(fit_forward=n+1, reverse=n+1, external=2, updates=n)
        cost_for = lambda k: 4*k+4+7*(1+int(k > 20))
        assert cost_for(n) == adam['render_cost'] <= c < cost_for(n+1) and c-adam['render_cost'] == case['work_cost_difference']
        assert len(adam['history']) == n and adam['target_updates'] == n
        before = np.asarray(spec['physical'], np.float32)
        for i, h in enumerate(adam['history']):
            assert h['physical_before'] == before.tolist() and h['update'] == i+1 and h['loss_at_state'] == i
            path.append(dict(case=name, method='Adam', cost=4*i+4+7*int(i >= 20), loss=h['loss'], physical=before.tolist(), Q=diagnostic.recovery_errors(before, truth)['Q_relative_frobenius']))
            before = np.asarray(h['physical_after'], np.float32)
        assert before.tolist() == adam['terminal']['physical']
        path.append(dict(case=name, method='Adam', cost=adam['render_cost'], loss=adam['terminal']['loss'], physical=before.tolist(), Q=diagnostic.recovery_errors(before, truth)['Q_relative_frobenius']))
        trajectories.extend(path)
        for label, arm in (('LM', lm), ('Adam', adam)):
            state = obs.read_numeric(raw, arm['final_state']); snapshots += 1
            assert state['physical'].tolist() == arm['terminal']['physical'] == arm['evaluation']['physical']
            if label == 'Adam':
                assert state['latent'].shape == state['exp_avg'].shape == state['exp_avg_sq'].shape == (7,) and float(state['step']) == n and np.all(state['exp_avg_sq'] >= 0)
            else: assert int(state['accepted_updates']) == accepted and float(state['damping']) == lm['damping']
            selections = [('work_endpoint', arm['evaluation'])]
            if label == 'Adam': selections.append(('unmatched_Adam20_context', arm['matched_update_evaluation']))
            for role, e in selections:
                arrays = obs.read_numeric(raw, e['arrays']); pred = arrays['predictions']; assert pred.shape == (84, 3, 3) and arrays['per_tape'].shape == (4, 84, 3, 3)
                assert arrays['physical'].tolist() == e['physical']
                errors = diagnostic.recovery_errors(np.asarray(e['physical'], np.float32), truth); assert errors == e['recovery_errors']
                m = {}
                for split in ('fit', 'held_out', 'relight'):
                    mask = [o['split'] == split for o in reference['scene']['observations']]
                    m[split] = {key: metrics['radiance_errors'](pred[mask, i], target[mask, i]) for i, key in enumerate(('diffuse', 'specular', 'total'))}
                    assert m[split] == e['metrics'][split]['errors'] and components(pred[mask], target[mask]) == {k: v for k, v in e['metrics'][split].items() if k != 'errors'}
                assert metrics['assess_success'](errors, m) == e['success']
                endpoint = dict(case=name, method=label, role=role, physical=e['physical'], physical_errors=errors, success=e['success'], metrics=e['metrics'],
                    updates=20 if role != 'work_endpoint' else accepted if label == 'LM' else n,
                    fitting_MSE=adam['history'][20]['loss'] if role != 'work_endpoint' else arm['terminal']['loss'],
                    cost=None if role != 'work_endpoint' else arm['render_cost'],
                    wall_seconds=None if role != 'work_endpoint' else arm['wall_seconds'],
                    terminal_gradient_L2=None if role != 'work_endpoint' else float(np.linalg.norm(arm['terminal']['scaled_gradient'])))
                endpoints.append(endpoint)
        groups.append(dict(case=name, matched_update_20_available=False, LM_status=lm['status'], LM_accepted_updates=accepted,
            Adam_matched_updates=n, LM_cost=c, Adam_cost=adam['render_cost'], cost_gap=case['work_cost_difference'],
            queried_trials=queried, rejected_trials=events['rejected_trials']))
    totals = dict(shader_forwards=2+sum(c[k+'_work']['done']['fit_forward']+c[k+'_work']['done']['external'] for c in report['cases'] for k in ('LM', 'Adam')),
        reverse_sweeps=sum(c[k+'_work']['done']['reverse'] for c in report['cases'] for k in ('LM', 'Adam')),
        Adam_returns=sum(c['Adam']['updates'] for c in report['cases']), LM_accepted_steps=sum(c['LM']['accepted_updates'] for c in report['cases']),
        Jacobians=sum(len(c['LM']['history']) for c in report['cases']), trial_attempts=len(trials), external_forwards=6, shared_replays=2)
    assert totals == report['totals'] and all(v <= plan['maxima'][k] for k, v in totals.items())
    assert report['shared_replay_work'] == dict(attempts=2, completed=2)
    comparisons = []
    for c in groups:
        es = {e['method']: e for e in endpoints if e['case'] == c['case'] and e['role'] == 'work_endpoint'}; l, a = es['LM'], es['Adam']
        comparisons.append(dict(**c, LM_Q=l['physical_errors']['Q_relative_frobenius'], Adam_Q=a['physical_errors']['Q_relative_frobenius'],
            LM_success=l['success']['all_passed'], Adam_success=a['success']['all_passed'], LM_loss=l['fitting_MSE'], Adam_loss=a['fitting_MSE'],
            LM_heldout=l['metrics']['held_out']['errors']['total']['relative_L2'], Adam_heldout=a['metrics']['held_out']['errors']['total']['relative_L2'],
            LM_relight=l['metrics']['relight']['errors']['total']['relative_L2'], Adam_relight=a['metrics']['relight']['errors']['total']['relative_L2']))
    out.mkdir(exist_ok=False)
    def save(name, value):
        with (out/name).open('x') as f: json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    def table(name, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open('x', newline='') as f:
            w = csv.DictWriter(f, keys, lineterminator='\n'); w.writeheader()
            for r in rows: w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    save('comparison.json', dict(schema_version=1, protocol='material-optimizer-e6-offline-v1', comparisons=comparisons,
        endpoints=endpoints, scientific_work=totals, scientific_execution_complete=True, algorithm_censored_LM_arms=2,
        matched_update_20_comparisons_available=0, additional_shader_queries=0, additional_optimizer_updates=0))
    table('comparisons.csv', comparisons); table('endpoints.csv', endpoints); table('trials.csv', trials); table('trajectories.csv', trajectories); table('conditioning.csv', spectra)
    save('integrity.json', dict(source_hashes_verified=len(plan['source_sha256']), input_hashes_verified=len(plan['input_sha256']),
        numeric_Jacobian_archives=24, bound_final_snapshots=snapshots, external_metric_sets_recomputed=6, new_M3_FD_checks=new_fd,
        scientific_counters_reconstructed=totals, precision_sensitive_accepted_trials=sum(t['accepted'] and t['ratio'] is not None and t['ratio'] > 2 for t in trials),
        trial_RGB_limit='Rejected-trial RGB arrays and all Adam fit RGB arrays were not archived; checks use sealed loss scalars, all LM pre-Jacobian RGB arrays, explicit step/system reconstruction, and fixed independent endpoint arrays; no renderer rerun or exhaustive Adam moment replay claimed'))
    save('evidence_manifest.json', dict(schema_version=1, execution_commit=COMMIT, freeze_sha256=FREEZE, report_sha256=REPORT, seal_sha256=SEAL, receipt_sha256=RECEIPT,
        raw_files=[identity(p) for p in sorted(raw.rglob('*')) if p.is_file()], sidecars=[identity(parent/(RUN+s)) for s in ('.authorization.json', '.launch.json', '.console.log', '.process.json')],
        analysis_source=identity(Path(__file__).resolve()), original_raw_evidence_unchanged=True))
    figures(out, comparisons, endpoints, trajectories, spectra, trials)
    print(json.dumps(dict(status='complete_offline_analysis', comparisons=comparisons, output=str(out), scientific_queries=0)))


def figures(out, comparisons, endpoints, trajectories, spectra, trials):
    os.environ['MPLCONFIGDIR'] = str(out/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(LM='#D55E00', Adam='#0072B2'); cases = [c['case'] for c in comparisons]
    folder = out/'figures'; folder.mkdir()
    def emit(fig, name):
        fig.savefig(folder/(name+'.png'), dpi=180, bbox_inches='tight'); fig.savefig(folder/(name+'.pdf'), bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, suffix, title in zip(axes, ('Q', 'heldout', 'loss'), ('Relative Q error', 'Held-out RGB relative L2', 'Fitting MSE')):
        for i, method in enumerate(('LM', 'Adam')):
            ax.bar(np.arange(2)+(i-.5)*.32, [c[method+'_'+suffix] for c in comparisons], width=.32, color=colors[method], label=method)
        ax.set_xticks(range(2), cases); ax.set_title(title); ax.grid(axis='y', alpha=.2)
    axes[0].axhline(.05, ls='--', color='black', lw=.8); axes[1].axhline(.01, ls='--', color='black', lw=.8); axes[2].set_yscale('log'); axes[0].legend()
    fig.suptitle('Work-matched endpoints: opposite physical-recovery winners; LM20 comparison unavailable'); fig.tight_layout(); emit(fig, 'work_matched_endpoints')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, case in enumerate(cases):
        for method in ('LM', 'Adam'):
            rows = [r for r in trajectories if r['case'] == case and r['method'] == method]
            for ax, key in zip(axes[i], ('loss', 'Q')): ax.plot([r['cost'] for r in rows], [r[key] for r in rows], color=colors[method], label=method)
        axes[i, 0].set_yscale('log'); axes[i, 0].set_ylabel(case); axes[i, 1].axhline(.05, ls='--', color='black', lw=.8)
        for ax in axes[i]: ax.grid(alpha=.2); ax.set_xlabel('Cumulative declared renderer cost C (not wall time)')
    axes[0, 0].set_title('Fitting MSE'); axes[0, 1].set_title('Physical Q error'); axes[0, 0].legend()
    fig.suptitle('Accepted-state trajectories, including query/derivative/rejection and evaluation cost'); fig.tight_layout(); emit(fig, 'work_trajectories')
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, case in enumerate(cases):
        rows = [r for r in spectra if r['case'] == case]
        axes[i, 0].plot(range(1, 8), rows[0]['singular_values'], 'o-', label='initial'); axes[i, 0].plot(range(1, 8), rows[-1]['singular_values'], 's--', label='last Jacobian'); axes[i, 0].set_yscale('log'); axes[i, 0].set_ylabel(case); axes[i, 0].legend()
        ts = [t for t in trials if t['case'] == case and t['accepted']]
        axes[i, 1].plot([t['accepted_index'] for t in ts], [t['weak_direction_step_fraction'] for t in ts], 'o-', color=colors['LM']); axes[i, 1].set_ylim(0, 1.05)
        ts = [t for t in trials if t['case'] == case and t['ratio'] is not None]
        for take, marker in ((True, 'o'), (False, 'x')):
            selected = [t for t in ts if t['accepted'] == take]
            axes[i, 2].scatter([t['outer'] for t in selected], [t['ratio'] for t in selected], marker=marker, label='accepted' if take else 'rejected')
        axes[i, 2].set_yscale('symlog', linthresh=2); axes[i, 2].legend(fontsize=7)
    for ax, title in zip(axes[0], ('Physical J singular values', 'Step fraction along current weakest direction', 'Actual / predicted reduction; near-floor ratios unstable')): ax.set_title(title)
    for ax, label in zip(axes[-1], ('SV index', 'Accepted LM update', 'Jacobian / outer index')): ax.set_xlabel(label)
    fig.tight_layout(); emit(fig, 'curvature_and_step_diagnostics')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, case in enumerate(cases):
        rows = {e['method']: e for e in endpoints if e['case'] == case and e['role'] == 'work_endpoint'}
        for j, method in enumerate(('LM', 'Adam')):
            m = rows[method]['metrics']['held_out']; values = [m['errors'][c]['relative_L2'] for c in ('diffuse', 'specular', 'total')]
            axes[i, 0].bar(np.arange(3)+(j-.5)*.3, values, width=.3, label=method, color=colors[method])
            err = rows[method]['physical_errors']; values = [max(err['sorted_axis_absolute_errors']), max(err['albedo_absolute_errors']), err['metallic_absolute_error']]
            axes[i, 1].bar(np.arange(3)+(j-.5)*.3, values, width=.3, label=method, color=colors[method])
        axes[i, 0].set_xticks(range(3), ('diffuse', 'specular', 'total')); axes[i, 0].set_ylabel(case+' held-out relative L2'); axes[i, 1].set_xticks(range(3), ('max axis error', 'max albedo error', 'metallic error')); axes[i, 1].set_ylabel('Absolute physical error')
    axes[0, 0].legend(); axes[0, 1].legend(); fig.suptitle('Work-matched physical/component errors; unchanged metrics and thresholds'); fig.tight_layout(); emit(fig, 'physical_and_component_errors')


if __name__ == '__main__': main()
