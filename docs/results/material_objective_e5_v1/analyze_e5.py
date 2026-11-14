"""E5 offline saved-array verification/figures. No Torch or scientific imports."""
import argparse
import ast
import csv
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name == 'torch' or name.startswith(('torch.', 'experiments.')):
            raise RuntimeError('Offline analysis forbids '+name)


sys.meta_path.insert(0, NoScience())
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import require, read_report

RUN = 'material-e5-b4cfa064e4df41e49fd865635bf5056b'
FREEZE = '2888b9f867ca86b00e2ee90d418fcb0dfb40fd7c52b5c91bab94c42e2015fe7d'
REPORT = 'f4fc926841cf79e8619cd326646f81fc490e442d8705e0de472d792e2c1b43a8'
SEAL = '4d819808774a8d5de447b52e010213eb399e88b3b8129dca53ebe64a30d9f700'
RECEIPT = '64d74a806ea143c9d40c822197ce493952b2bb1bea79e12820c57bdcbfaa3015'
EXECUTION = '93e54858a0a0a7bc3ff8975d12ee29d27fc6c8e4'
BASE = 'a832f020d4d9618295ccfe818131a0f8193eb248'
POINTS = ('truth', 'N1', 'AS1')
TAPES = ('accepted', 'independent_1', 'independent_2')
LABELS = ('smaller width', 'larger width', 'theta', 'albedo R', 'albedo G', 'albedo B', 'metallic')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(p): return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def components(predictions, target):
    delta = predictions.astype(np.float64)-target
    d, s, t = (delta[:, i] for i in range(3))
    return dict(component_MSE=[float(np.mean(v*v)) for v in (d, s, t)],
                diffuse_specular_inner_product=float(np.sum(d*s)),
                normalized_residual_inner_product=float(np.sum(d*s)/max(np.linalg.norm(d)*np.linalg.norm(s), 1e-30)),
                total_squared_error=float(np.sum(t*t)), component_squared_error=float(np.sum(d*d)+np.sum(s*s)))


def spectra(j):
    _, s, vt = np.linalg.svd(j, full_matrices=False)
    weak = vt[-1]; weak *= 1 if weak[np.argmax(abs(weak))] >= 0 else -1
    return s, weak


def q(p):
    x, y, angle = np.asarray(p, np.float64)[:3]; c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    return r@np.diag([x*x, y*y])@r.T


def metric_functions(plan):
    path = 'experiments/run_known_material_recovery_v1.py'
    assert sha(REPO/path) == plan['source_sha256'][path]
    tree = ast.parse((REPO/path).read_bytes())
    cfg = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in n.targets))
    success = next(ast.literal_eval(v) for k, v in zip(cfg.keys, cfg.values) if k.value == 'success')
    ns = dict(np=np, require=require, SETTINGS=dict(success=success))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('radiance_errors', 'assess_success')]
    assert all(not any(isinstance(v, (ast.Import, ast.ImportFrom)) for v in ast.walk(n)) for n in nodes)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPO/path), 'exec'), ns)
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-parent', type=Path, required=True); parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); parent = args.evidence_parent; root = parent/RUN; out = args.output_dir
    assert root.is_absolute() and root.resolve() == root and out.is_absolute() and out.resolve() == out and out.parent == parent and not out.exists()
    plan, _ = read_report(REPO/'docs/protocols/MATERIAL_OBJECTIVE_E5_V1.freeze.json', FREEZE)
    for p, h in plan['source_sha256'].items(): assert sha(REPO/p) == h, p
    for p, h in plan['input_sha256'].items(): assert sha(p) == h, p
    report, _ = read_report(root/'result.json', REPORT); seal, _ = read_report(root/'finalization/seal.json', SEAL)
    receipt, _ = read_report(parent/(RUN+'.process.json'), RECEIPT)
    assert report['scientific_work_completed'] and report['cleanup_verified'] and report['terminal_integrity_verified']
    assert report['completed'] == report['attempts'] == plan['work'] and report['native_shader_calls'] == 385
    assert seal['report']['sha256'] == REPORT and seal['outcome'] == 'complete' and receipt['process_exit_code'] == receipt['retry_attempts'] == 0
    for k in ('authorization', 'launch', 'console'): assert identity(Path(receipt[k]['path'])) == receipt[k]
    launch = json.loads(Path(receipt['launch']['path']).read_text()); assert launch['commit'] == EXECUTION and launch['freeze_sha256'] == FREEZE and launch['foreground_supervisor']
    ref_path = REPO/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'; assert sha(ref_path) == plan['inputs']['accepted_3D1_SHA256']
    ref = json.loads(ref_path.read_bytes()); info = next(x for x in ref['materials'] if x['material'] == 'M2')
    target = np.asarray(info['targets']['fine'], np.float64); fit = target[:24, 2].astype(np.float32)
    assert target.shape == (84, 3, 3) and hashlib.sha256(target.tobytes()).hexdigest() == plan['inputs']['targets']['M2']['fine_sha256']
    numeric = metric_functions(plan); truth = np.asarray(plan['points']['truth'], np.float64); scales = np.asarray(plan['scales'])
    old_path = 'docs/results/material_parameterization_e4_v1/comparison.json'
    assert (REPO/old_path).read_bytes() == subprocess.check_output(['git', 'show', BASE+':'+old_path], cwd=REPO)
    old = next(x for x in json.loads((REPO/old_path).read_bytes())['endpoints'] if x['case'] == 'M2/103')
    objective = []; sensitivity = []; external = []; jacobians = {}; fd_ratios = []; gradient_errors = []
    eps = np.finfo(np.float32).eps; gamma72 = 72*eps/(1-72*eps)
    def check_mse(a, value):
        computed = float(np.mean((a[:, 2]-fit)**2))
        assert abs(computed-value) <= 2*gamma72*max(value, 1e-30), (computed, value)
    assert len(report['replays']) == 2 and all(r['passed'] and r['actual_loss'] == r['expected_loss'] for r in report['replays'])
    for m in report['measurements']:
        label, tape = m['point'], m['tapes']; a = obs.read_numeric(root, m['arrays'])
        assert a['predictions'].shape == (24, 3, 3) and a['predictions'].dtype == np.float32
        assert a['per_tape'].shape == (4, 24, 3, 3) and a['J_scaled'].shape == (24, 3, 3, 7) and a['J_scaled'].dtype == np.float32
        assert a['FD_scaled'].shape == (2, 24, 3, 3, 7) and a['FD_scaled'].dtype == np.float64
        np.testing.assert_allclose(a['per_tape'].mean(0), a['predictions'], atol=1e-7, rtol=1e-6)
        check_mse(a['predictions'], m['objective_float32']); assert components(a['predictions'], target[:24]) == m['errors']
        assert a['physical'].tolist() == np.asarray(plan['points'][label], np.float32).tolist()
        g = a['physical_gradient'].astype(np.float64)*scales
        assert g.tolist() == m['scaled_physical_gradient'] and float(np.linalg.norm(g)) == m['gradient_norm']
        j, fd = a['J_scaled'].astype(np.float64), a['FD_scaled']
        errors = np.linalg.norm(fd.reshape(2, -1, 7)-j.reshape(1, -1, 7), axis=1)
        limits = 1e-4+.05*np.linalg.norm(j.reshape(-1, 7), axis=0)
        np.testing.assert_array_equal(errors, m['derivative_check']['L2_column_errors'])
        assert np.all(errors <= limits) and m['derivative_check']['passed']; fd_ratios.append(float(np.max(errors/limits)))
        r = (a['predictions'][:, 2]-fit).astype(np.float64)
        chain = 2*j[:, 2].reshape(72, 7).T@r.reshape(72)/72
        gradient_errors.append(float(np.linalg.norm(chain-g))); assert gradient_errors[-1] <= 1e-7+1e-3*np.linalg.norm(g)
        order = m['sensitivity']['reporting_column_order']; aligned = j[..., order]
        jt = aligned[:, 2].reshape(72, 7)/np.sqrt(72); jc = aligned[:, :2].reshape(144, 7)/np.sqrt(72)
        s, weak = spectra(jt); sc, _ = spectra(jc)
        np.testing.assert_array_equal(s, m['sensitivity']['total']['singular_values'])
        np.testing.assert_allclose(2*jt.T@jt, m['sensitivity']['total']['Gauss_Newton_matrix'], rtol=1e-14, atol=1e-17)
        norms = np.linalg.norm(jt, axis=0); corr = jt.T@jt/np.outer(norms, norms)
        assert np.sum(s >= 1e-4*s[0]) == m['sensitivity']['total']['effective_rank']
        jacobians[label, tape] = dict(total=jt, components=jc, correlations=corr)
        objective.append(dict(point=label, tapes=tape, fitting_MSE=m['objective_float32'], scaled_gradient_norm=m['gradient_norm'],
            scaled_gradient=g.tolist(), individual_tape_MSE=m['per_tape_MSE'], physical_errors=diagnostic.recovery_errors(a['physical'], truth)))
        sensitivity.append(dict(point=label, tapes=tape, singular_values=s.tolist(), condition_ratio=float(s[0]/s[-1]),
            effective_rank=int(np.sum(s >= 1e-4*s[0])), weak_aligned_scaled_direction=weak.tolist(),
            component_singular_values=sc.tolist(), component_minimum_SV_gain=float(sc[-1]/s[-1]), correlations=corr.tolist()))
    for m in report['gauge_measurements']:
        a = obs.read_numeric(root, m['arrays']); check_mse(a['predictions'], m['objective_float32'])
        assert components(a['predictions'], target[:24]) == m['errors']
        objective.append(dict(point='truth_swapped', tapes=m['tapes'], fitting_MSE=m['objective_float32'], scaled_gradient_norm=None))
    assert len(objective) == 12 and len(sensitivity) == 9
    for e in report['evaluations']:
        a = obs.read_numeric(root, e['arrays']); predictions = a['predictions']; assert predictions.shape == (84, 3, 3) and a['per_tape'].shape == (4, 84, 3, 3)
        errors = diagnostic.recovery_errors(np.asarray(e['physical'], np.float32), truth); assert errors == e['recovery_errors']
        metrics = {}
        for split in ('fit', 'held_out', 'relight'):
            mask = [o['split'] == split for o in ref['scene']['observations']]
            metrics[split] = {name: numeric['radiance_errors'](predictions[mask, i], target[mask, i]) for i, name in enumerate(('diffuse', 'specular', 'total'))}
            assert metrics[split] == e['metrics'][split]['errors'] and components(predictions[mask], target[mask]) == {k: v for k, v in e['metrics'][split].items() if k != 'errors'}
            external.append(dict(point=e['point'], split=split, total_MSE=e['metrics'][split]['component_MSE'][2],
                total_relative_L2=metrics[split]['total']['relative_L2'], diffuse_relative_L2=metrics[split]['diffuse']['relative_L2'],
                specular_relative_L2=metrics[split]['specular']['relative_L2'], residual_inner_product=e['metrics'][split]['normalized_residual_inner_product'],
                Q_error=errors['Q_relative_frobenius'], physical=e['physical'], physical_errors=errors))
        if e['point'] == 'N1': assert metrics == {k: old['baseline']['radiance_errors100'][k] for k in metrics}
        if e['point'] == 'AS1': assert metrics == {k: old['AS1']['radiance_errors100'][k] for k in metrics}
        if e['point'] == 'direct_final': direct_success = numeric['assess_success'](errors, metrics)
    direct = report['direct']; assert direct['status'] == 'complete' and direct['updates'] == len(direct['history']) == 50
    assert direct['history'][0]['loss'] == next(x['fitting_MSE'] for x in objective if x['point'] == 'N1' and x['tapes'] == 'accepted')
    trajectory = []
    for state in range(51):
        p = np.asarray(plan['points']['N1'] if state == 0 else direct['history'][state-1]['physical'], np.float32)
        errors = diagnostic.recovery_errors(p, truth)
        if state: assert errors == direct['history'][state-1]['recovery_errors'] and direct['history'][state-1]['update'] == state
        loss = direct['history'][state]['loss'] if state < 50 else direct['final_objective']
        trajectory.append(dict(state=state, fitting_MSE=loss, Q_error=errors['Q_relative_frobenius'], physical=p.tolist(),
            latent_gradient_norm=direct['history'][state]['latent_gradient_norm'] if state < 50 else None,
            albedo_max_error=max(errors['albedo_absolute_errors']), metallic_error=errors['metallic_absolute_error']))
    snapshot_path = root/'direct/final-state.npz.json'; snapshot = obs.read_numeric(root, dict(path=str(snapshot_path.relative_to(root)), sha256=sha(snapshot_path)))
    assert snapshot['step'].shape == () and float(snapshot['step']) == 50 and np.all(snapshot['exp_avg_sq'] >= 0)
    assert snapshot['latent'].shape == snapshot['physical'].shape == snapshot['gradient'].shape == snapshot['exp_avg'].shape == snapshot['exp_avg_sq'].shape == (7,)
    assert snapshot['physical'].tolist() == direct['history'][-1]['physical']
    assert snapshot['gradient'].tolist() == direct['final_physical_gradient']
    objective.append(dict(point='direct_final', tapes='accepted', fitting_MSE=direct['final_objective'], scaled_gradient_norm=float(np.linalg.norm(snapshot['gradient']*scales))))
    rows = {(r['point'], r['tapes']): r for r in objective}; ex = {(r['point'], r['split']): r for r in external}
    ratios = {point: {t: rows[point, t]['fitting_MSE']/rows['truth', t]['fitting_MSE'] for t in TAPES} for point in ('N1', 'AS1')}
    gradient_cosines = {}
    for label in POINTS:
        gs = [np.asarray(rows[label, t]['scaled_gradient']) for t in TAPES]
        gradient_cosines[label] = [float(gs[0]@v/np.linalg.norm(gs[0])/np.linalg.norm(v)) for v in gs[1:]]
    improvement = dict(fitting_MSE=1-direct['final_objective']/rows['N1', 'accepted']['fitting_MSE'],
        Q=1-trajectory[-1]['Q_error']/trajectory[0]['Q_error'],
        heldout=1-ex['direct_final', 'held_out']['total_relative_L2']/ex['N1', 'held_out']['total_relative_L2'],
        relighting=1-ex['direct_final', 'relight']['total_relative_L2']/ex['N1', 'relight']['total_relative_L2'])
    comparison = dict(schema_version=1, protocol='material-e5-offline-v1', run_id=RUN, scientific_execution_complete=True,
        truth_preferred_all_three_quartets=all(v > 1 for d in ratios.values() for v in d.values()), endpoint_truth_MSE_ratios=ratios,
        scaled_gradient_cosines_to_accepted=gradient_cosines, direct_relative_improvements=improvement, direct_success=direct_success,
        full_objective_measurements=objective, full_sensitivity_measurements=sensitivity, full_external_measurements=external,
        final_direct_parameters=snapshot['physical'].tolist(), reference_convergence_maximum_bound_ratio=info['targets']['maximum_bound_ratio'],
        truth_gauge_Q_relative_difference=float(np.linalg.norm(q(np.asarray(plan['points']['truth_swapped'], np.float32))-q(np.asarray(plan['points']['truth'], np.float32)))/np.linalg.norm(q(truth))),
        additional_model_queries=0, additional_shader_queries=0, additional_derivative_queries=0, additional_optimizer_updates=0)
    out.mkdir(exist_ok=False)
    def save(name, value):
        with (out/name).open('x') as f: json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    def table(name, values):
        keys = list(dict.fromkeys(k for r in values for k in r))
        with (out/name).open('x', newline='') as f:
            writer = csv.DictWriter(f, keys, lineterminator='\n'); writer.writeheader()
            for row in values: writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    save('comparison.json', comparison); table('objectives.csv', objective); table('sensitivity.csv', sensitivity); table('external.csv', external); table('direct_trajectory.csv', trajectory)
    save('integrity.json', dict(frozen_sources_verified=len(plan['source_sha256']), accepted_inputs_verified=len(plan['input_sha256']),
        numeric_sensitivity_archives=9, numerical_derivative_gate_passes=9, maximum_FD_gate_bound_ratio=max(fd_ratios),
        maximum_scaled_gradient_J_chain_discrepancy=max(gradient_errors), external_metric_sets_recomputed=5, direct_updates=50,
        original_N1_AS1_external_metrics_reproduced_exactly=True, direct_final_state_consistent=True,
        direct_snapshot_limit='Final Adam snapshot has its own numeric manifest but is not selected by the sealed execution report; inventoried/hash-pinned offline, with step/physical/gradient consistency checks, not an exhaustive optimizer-transition reconstruction'))
    save('evidence_manifest.json', dict(schema_version=1, execution_commit=EXECUTION, freeze_sha256=FREEZE, report_sha256=REPORT,
        seal_sha256=SEAL, receipt_sha256=RECEIPT, raw_root=str(root), original_raw_files=[identity(p) for p in sorted(root.rglob('*')) if p.is_file()],
        launch_sidecars=[identity(parent/(RUN+s)) for s in ('.authorization.json', '.launch.json', '.console.log', '.process.json')],
        analysis_source=identity(Path(__file__).resolve()), accepted_control=identity(REPO/old_path), large_numeric_evidence_not_copied=True))
    figures(out, objective, sensitivity, external, trajectory, jacobians)
    print(json.dumps(dict(status='complete_offline_analysis', output=str(out), truth_preferred=True, direct_improvements=improvement, direct_success=direct_success, scientific_queries=0)))


def figures(out, objective, sensitivity, external, trajectory, jacobians):
    os.environ['MPLCONFIGDIR'] = str(out/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(truth='#009E73', N1='#0072B2', AS1='#D55E00', truth_swapped='#CC79A7', direct_final='#6A51A3')
    directory = out/'figures'; directory.mkdir()
    def emit(fig, name):
        fig.savefig(directory/(name+'.png'), dpi=180, bbox_inches='tight'); fig.savefig(directory/(name+'.pdf'), bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, label in enumerate((*POINTS, 'truth_swapped')):
        rows = [next(r for r in objective if r['point'] == label and r['tapes'] == t) for t in TAPES]
        axes[0].bar(np.arange(3)+(i-1.5)*.19, [r['fitting_MSE'] for r in rows], width=.19, label=label, color=colors[label])
        if label != 'truth_swapped': axes[1].plot(TAPES, [r['scaled_gradient_norm'] for r in rows], 'o-', label=label, color=colors[label])
    axes[0].set_xticks(np.arange(3), TAPES); axes[0].set_ylabel('Fitting MSE (log scale)'); axes[0].set_yscale('log'); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Scaled physical gradient L2'); axes[1].set_yscale('log'); axes[1].legend()
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.suptitle('M2/103: truth remains preferred across all three fixed quartets'); fig.tight_layout(); emit(fig, 'objective_and_sampling')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for label in POINTS:
        r = next(r for r in sensitivity if r['point'] == label and r['tapes'] == 'accepted')
        axes[0].plot(range(1, 8), r['singular_values'], 'o-', label=label+' total', color=colors[label])
        axes[0].plot(range(1, 8), r['component_singular_values'], ':', color=colors[label], label=label+' components')
        axes[1].plot(range(7), r['weak_aligned_scaled_direction'], 'o-', label=label, color=colors[label])
    axes[0].set_yscale('log'); axes[0].set_xlabel('Singular-value index'); axes[0].set_ylabel('Scaled J / sqrt(72) singular values'); axes[0].legend(fontsize=7)
    axes[1].set_xticks(range(7), LABELS, rotation=70); axes[1].set_ylabel('Weakest scaled right-singular vector'); axes[1].legend(); axes[1].axhline(0, lw=.5, color='black')
    im = axes[2].imshow(jacobians['N1', 'accepted']['correlations'], vmin=-1, vmax=1, cmap='RdBu_r')
    axes[2].set_xticks(range(7), LABELS, rotation=70); axes[2].set_yticks(range(7), LABELS); axes[2].set_title('N1 total-RGB column cosines'); fig.colorbar(im, ax=axes[2], shrink=.7)
    fig.suptitle('Local radiance sensitivity: full rank, correlated physical directions (accepted tapes)'); fig.tight_layout(); emit(fig, 'local_sensitivity')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    chosen = ('truth', 'N1', 'AS1', 'direct_final')
    for i, component in enumerate(('diffuse', 'specular', 'total')):
        values = [next(r for r in external if r['point'] == label and r['split'] == 'held_out')[component+'_relative_L2'] for label in chosen]
        axes[0].bar(np.arange(4)+(i-1)*.24, values, width=.24, label=component)
    axes[0].set_xticks(range(4), chosen); axes[0].set_yscale('log'); axes[0].set_ylabel('Held-out component relative L2'); axes[0].legend()
    values = [next(r for r in external if r['point'] == label and r['split'] == 'held_out')['residual_inner_product'] for label in chosen]
    axes[1].bar(chosen, values, color=[colors[label] for label in chosen]); axes[1].set_ylim(-1.05, 1); axes[1].axhline(0, lw=.5, color='black'); axes[1].set_ylabel('Normalized diffuse/specular residual inner product')
    fig.suptitle('Independent evaluation tapes: total RGB conceals component compensation'); fig.tight_layout(); emit(fig, 'component_compensation')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key, title in zip(axes.ravel()[:3], ('fitting_MSE', 'Q_error', 'metallic_error'), ('Fitting MSE', 'Relative Q error', 'Absolute metallic error')):
        ax.plot([r['state'] for r in trajectory], [r[key] for r in trajectory], color=colors['direct_final']); ax.set_title(title); ax.set_xlabel('Direct Adam updates'); ax.grid(alpha=.2)
    axes[0, 0].set_yscale('log'); axes[0, 1].axhline(.05, color='black', ls='--', lw=.8, label='Original Q threshold'); axes[0, 1].legend(fontsize=8)
    ax = axes[1, 1]
    for i, split in enumerate(('held_out', 'relight')):
        values = [next(r for r in external if r['point'] == label and r['split'] == split)['total_relative_L2'] for label in ('N1', 'direct_final')]
        ax.bar(np.arange(2)+(i-.5)*.3, values, width=.3, label=split)
    ax.set_xticks(range(2), ('N1 start', 'Direct state 50')); ax.set_ylabel('Independent RGB relative L2'); ax.legend()
    fig.suptitle('One local direct probe: physical and RGB improvement, no full recovery'); fig.tight_layout(); emit(fig, 'direct_probe')


if __name__ == '__main__': main()
