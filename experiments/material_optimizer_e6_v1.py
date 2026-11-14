"""E6 paired physical LM/direct Adam pilot. Preparation CLI; execution gated."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from experiments import material_objective_e5_v1 as previous
from experiments import physical_lm_e6_v1 as method
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import require, read_report
from infrastructure.runs import validate_run_id

ROOT = previous.ROOT
BASE = '438bb28d0f2fad046a50cf1941c0e44de399ffc9'
PROTOCOL = 'physical-material-optimizer-e6-v1'
PARENT_SHA = '2888b9f867ca86b00e2ee90d418fcb0dfb40fd7c52b5c91bab94c42e2015fe7d'
FREEZE = 'docs/protocols/MATERIAL_OPTIMIZER_E6_V1.freeze.json'
VALIDATION = 'docs/evidence/MATERIAL_OPTIMIZER_E6_PREPARATION_V1.json'
CODE = ('experiments/physical_lm_e6_v1.py', 'experiments/material_optimizer_e6_v1.py',
        'tests/scientific/test_material_optimizer_e6_v1.py', 'docs/protocols/MATERIAL_OPTIMIZER_E6_V1.md')
CASES = ('M2/103', 'M3/101')
sha = previous.sha


def make_plan():
    parent = previous.validate_freeze(PARENT_SHA); ids = dict(parent['input_sha256'])
    def accepted(name):
        p = ROOT/name; require(p.read_bytes() == subprocess.check_output(['git', 'show', BASE+':'+name], cwd=ROOT), 'Accepted file changed: '+name)
        ids[str(p)] = sha(p); return json.loads(p.read_text())
    manifest = accepted('docs/results/material_objective_e5_v1/evidence_manifest.json')
    for e in manifest['original_raw_files']+manifest['launch_sidecars']:
        p = Path(e['path']); require(p.stat().st_size == e['bytes'] and sha(p) == e['sha256'], 'Accepted E5 evidence changed'); ids[str(p)] = e['sha256']
    ids[str(ROOT/previous.FREEZE)] = PARENT_SHA
    panel = accepted('docs/results/neural_material_e3_complete_v1/comparison.json'); cases = []
    for name in CASES:
        old = next(x for x in panel['endpoints'] if x['case'] == name and x['lr'] == .001)
        root = Path(parent['output_parent'])/old['endpoint_source_run']; path = root/'result.json'
        require(str(path) in ids and sha(path) == ids[str(path)], 'Unadmitted producing report')
        report = json.loads(path.read_text()); a = next(a for a in report['outcomes'] if a['material']+'/'+str(a['seed']) == name and a['lr'] == .001)
        require(a['status'] == 'complete', 'Missing complete state100'); trace = root/a['trajectory']['path']
        require(sha(trace) == a['trajectory']['sha256'], 'Selected trace changed'); ids[str(trace)] = sha(trace)
        row = next(json.loads(line) for line in trace.read_text().splitlines() if json.loads(line)['state'] == 100)
        arrays = obs.read_numeric(root, row['signals']); physical = np.asarray(old['physical100'], np.float32)
        require(physical.tolist() == row['physical']['physical'] and row['loss'] == old['loss100'], 'Endpoint disagrees with accepted panel')
        z = np.r_[arrays['mx_predictor.logits'].ravel(), arrays['my_predictor.logits'].ravel(), physical[2],
                  arrays['kd_predictor.logits'].ravel(), arrays['metallic_predictor.logits'].ravel()].astype(np.float32)
        selection = root/row['signals']['path']; meta, _ = read_report(selection, row['signals']['sha256'])
        ids[str(selection)] = row['signals']['sha256']; ids[str(root/meta['archive'])] = meta['archive_sha256']
        cases.append(dict(case=name, material=a['material'], physical=physical.tolist(), initial_logits=z.tolist(),
            source_run=str(root), signals=row['signals'], saved_loss=row['loss'], new_directional_FD=name == 'M3/101'))
    sources = dict(parent['source_sha256']); sources.update({p: sha(ROOT/p) for p in CODE})
    return dict(schema_version=1, protocol=PROTOCOL, execution_authorized=False, base_commit=BASE, cases=cases,
        source_sha256=sources, input_sha256=ids, runtime=parent['runtime'], arithmetic_runtime=parent['arithmetic_runtime'],
        inputs=parent['inputs'], tapes={k: parent['tapes'][k] for k in ('accepted', 'evaluation')},
        optimizer=parent['optimizer'], optimizer_effective=parent['optimizer_effective'], scales=method.SCALES.tolist(),
        LM_policy=method.POLICY, accepted_target=20, trials_per_Jacobian=4,
        maxima=dict(shader_forwards=956, reverse_sweeps=1062, Adam_returns=498, LM_accepted_steps=40,
                    Jacobians=40, trial_attempts=160, external_forwards=6, shared_replays=2),
        work_matching='C=2*(fit forwards+fit reverse sweeps)+7*external forwards; Adam largest n with 4n+4+7*(1+[n>20]) <= actual LM C',
        output_parent=parent['output_parent'], storage=parent['storage'], resources=parent['resources'])


def validate_freeze(selected):
    p, _ = read_report(ROOT/FREEZE, selected)
    require(p['protocol'] == PROTOCOL and p['execution_authorized'] is False and [x['case'] for x in p['cases']] == list(CASES) and
        p['LM_policy'] == method.POLICY and p['scales'] == method.SCALES.tolist() and p['accepted_target'] == method.ACCEPTED_TARGET and p['trials_per_Jacobian'] == method.TRIALS,
        'Frozen E6 method differs')
    for n, h in p['source_sha256'].items(): require(sha(ROOT/n) == h, 'Frozen source changed: '+n)
    for n, h in p['input_sha256'].items(): require(sha(n) == h, 'Accepted input changed: '+n)
    v, _ = read_report(ROOT/VALIDATION, p['validation_sha256'])
    require(v['exit_code'] == v['failed'] == v['errors'] == v['skipped'] == v['scientific_queries'] == 0 and
        v['new_source_sha256'] == {n: p['source_sha256'][n] for n in CODE} and sha(ROOT/v['JUnit']['path']) == v['JUnit']['sha256'], 'E6 focused validation differs')
    return p


def validate_review(review, selected):
    require(set(review) == {'phase', 'scientific_execution_authorized', 'freeze_sha256', 'reviewed_commit', 'reviewed_branch',
        'run_id', 'output_root', 'external_watchdog'} and review['phase'] == 'E.6' and review['scientific_execution_authorized'] is True, 'Explicit E6 execution approval required')
    p = validate_freeze(selected); git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    require(review['freeze_sha256'] == selected and review['external_watchdog'] == dict(wall_seconds=1800, kill_grace_seconds=10), 'E6 freeze/watchdog mismatch')
    require(git('rev-parse', 'HEAD') == review['reviewed_commit'] and git('branch', '--show-current') == review['reviewed_branch'] == 'feat/neural-material-early-step-e2a' and not git('status', '--porcelain'), 'Clean reviewed E6 checkout required')
    validate_run_id(review['run_id']); path = Path(review['output_root'])
    require(path.is_absolute() and path.resolve() == path and path.parent == Path(p['output_parent']) and path.name == review['run_id'] and path.name.startswith('material-e6-') and not path.exists(), 'Fresh E6 output required')
    require(shutil.disk_usage(path.parent).free >= p['storage']['minimum_free_bytes'], 'Insufficient output storage')
    return p


def execute_reviewed(review_path, review_sha, freeze_sha):
    review, _ = read_report(Path(review_path), review_sha); plan = validate_review(review, freeze_sha)
    import platform
    import torch
    from experiments import run_known_material_recovery_v1 as original
    from experiments import material_lm_gauge_trajectories_v2 as accepted
    from experiments import neural_material_early_step_e2a as resource_owner
    require(dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device=str(torch.get_default_device()),
        torch_threads=torch.get_num_threads()) == plan['runtime'] and accepted.arithmetic_runtime() == plan['arithmetic_runtime'], 'Accepted CPU runtime required')
    store = obs.Store(review['output_root'], plan['output_parent'], total_cap=plan['storage']['output_cap_bytes'])
    store.json('execution-review.json', review)
    result = dict(protocol=PROTOCOL, status='running', freeze_sha256=freeze_sha, cases=[], initial_gates_passed=False,
        cleanup_verified=False, terminal_integrity_verified=False, scientific_work_completed=False, evidence_finalized=False,
        shared_replay_work=dict(attempts=0, completed=0))
    phase = 'setup'; cores = []; shared = 0
    try:
        with resource_owner.limits() as (check, begin, end):
            begin(); reference_path = ROOT/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'
            require(sha(reference_path) == plan['inputs']['accepted_3D1_SHA256'], 'Reference identity differs')
            reference = json.loads(reference_path.read_text()); scene = reference['scene']; observations = scene['observations']
            arrays = dict(accepted=previous.tape_arrays((3000, 3001, 3002, 3003), 24), evaluation=previous.tape_arrays((1000, 1001, 1002, 1003), 84))
            require({n: previous.tape_identity(a) for n, a in arrays.items()} == plan['tapes'], 'Tapes changed')
            batches = {n: original.ObservationBatch(scene, observations[:24] if n == 'accepted' else observations,
                original.RandomInputs(*(torch.from_numpy(a[k]) for k in ('u', 'v', 'azimuth'))), 4) for n, a in arrays.items()}
            shader = original.MaterialShaderV1('A2', 'atan2'); require(shader.identity == plan['inputs']['shader'], 'Shader identity differs')
            class Core:
                def __init__(self, spec, name, target):
                    self.spec = spec; self.name = name; self.work = method.Work(name, spec['new_directional_FD'] if name == 'LM' else False)
                    self.prefix = spec['case'].replace('/', '-').lower()+'-'+name.lower(); self.target = target; self.native = 0
                    self.p = np.asarray(spec['physical'], np.float32); self.pending = {}; self.last = None
                def tensor_forward(self, p, phase='trial', capture=None):
                    check(); self.work.attempt('forward', phase); wrapped = previous.CapturedShader(shader)
                    y = batches['evaluation' if phase == 'external' else 'accepted'](wrapped, p)
                    self.native += wrapped.calls; require(wrapped.calls == 1 and bool(torch.isfinite(y).all()), 'Invalid native renderer output')
                    self.work.complete('forward', phase)
                    if capture is not None: capture.update(per_tape=wrapped.per_tape)
                    return y
                def value(self, y):
                    loss = torch.mean((y[:, 2]-self.target)**2)
                    require(bool(torch.isfinite(loss)), 'Nonfinite original MSE')
                    return loss
                def forward(self, p, phase='trial'):
                    with torch.no_grad(): y = self.tensor_forward(torch.from_numpy(p.copy()), phase)
                    return dict(loss=float(self.value(y)), residual=(y[:, 2].numpy()-self.target.numpy()).astype(np.float64).reshape(72)/math.sqrt(72), values=y.numpy().copy())
                def jacobian(self):
                    p = torch.from_numpy(self.p.copy()); expected = torch.from_numpy(self.current['values'][:, 2].copy()).reshape(72)
                    fn = lambda x, phase: self.tensor_forward(x, phase)[:, 2].reshape(72)
                    raw = previous.columns(fn, p, expected, self.work, self.pending, check); j = raw.astype(np.float64)/math.sqrt(72)
                    if self.spec['new_directional_FD'] and self.work.events['jacobians_attempted'] == 1:
                        _, _, vt = np.linalg.svd(j, full_matrices=False); direction = vt[-1]; fd = []
                        for h in previous.FD_STEPS:
                            pair = []
                            for sign in (-1, 1):
                                proposed = (self.p.astype(np.float64)+sign*h*method.SCALES*direction).astype(np.float32); method.valid(proposed)
                                with torch.no_grad(): y = self.tensor_forward(torch.from_numpy(proposed), 'initial_fd')[:, 2].numpy().astype(np.float64).reshape(72)/math.sqrt(72)
                                pair.append(y)
                            fd.append((pair[1]-pair[0])/(2*h))
                        error = np.linalg.norm(np.asarray(fd)-j@direction, axis=1); limit = 1e-4+.05*np.linalg.norm(j@direction)
                        self.pending.update(initial_direction=direction, initial_FD=np.asarray(fd), initial_FD_errors=error)
                        require(bool(np.all(error <= limit)), 'New M3 directional derivative admission failed')
                    self.last = dict(physical=self.p.copy(), predictions=self.current['values'].copy(), J_scaled=raw.copy(), **self.pending)
                    return j
                def terminal(self):
                    p = torch.from_numpy(self.p.copy()).requires_grad_(); y = self.tensor_forward(p, 'terminal'); l = self.value(y)
                    self.work.attempt('reverse', 'terminal'); g, = torch.autograd.grad(l, p); self.work.complete('reverse', 'terminal')
                    require(bool(torch.isfinite(g).all()), 'Nonfinite terminal physical gradient')
                    return dict(loss=float(l.detach()), physical=self.p.tolist(), scaled_gradient=(g.numpy()*method.SCALES).tolist())
                def external(self, label, p):
                    capture = {}
                    with torch.no_grad(): y = self.tensor_forward(torch.from_numpy(p.copy()), 'external', capture).numpy().astype(np.float64)
                    metrics = {}
                    for split in ('fit', 'held_out', 'relight'):
                        mask = [o['split'] == split for o in observations]
                        metrics[split] = dict(errors={n: original.radiance_errors(y[mask, i], self.fine[mask, i]) for i, n in enumerate(('diffuse', 'specular', 'total'))}, **previous.component_errors(y[mask], self.fine[mask]))
                    errors = diagnostic.recovery_errors(p, self.reporting_truth)
                    return dict(physical=p.tolist(), metrics=metrics, recovery_errors=errors,
                        success=original.assess_success(errors, {s: v['errors'] for s, v in metrics.items()}),
                        arrays=store.numeric(self.prefix+'/'+label+'.npz', dict(predictions=y, per_tape=capture['per_tape'], physical=p), 1024**2))
            phase = 'all_initial_admission'
            admission_start = time.perf_counter(); admission_cpu = time.process_time()
            for spec in plan['cases']:
                info = next(v for v in reference['materials'] if v['material'] == spec['material']); fine = np.asarray(info['targets']['fine'], np.float64)
                require(fine.shape == (84, 3, 3) and original.reference.sha_array(fine) == plan['inputs']['targets'][spec['material']]['fine_sha256'], 'Target changed')
                target = torch.from_numpy(fine[:24, 2].astype(np.float32)); signals = obs.read_numeric(Path(spec['source_run']), spec['signals'])
                lm, adam = (Core(spec, name, target) for name in ('LM', 'Adam'))
                row = dict(case=spec['case'], initial_parity=False,
                    LM_work=dict(attempts=lm.work.attempts, done=lm.work.done, events=lm.work.events),
                    Adam_work=dict(attempts=adam.work.attempts, done=adam.work.done, events=adam.work.events))
                result['cases'].append(row); cores.append((lm, adam, row))
                wrapped = previous.CapturedShader(shader, torch.from_numpy(signals['shader.encoding'].copy()))
                result['shared_replay_work']['attempts'] += 1
                with torch.no_grad(): replay = batches['accepted'](wrapped, torch.tensor(spec['physical'], dtype=torch.float32))
                require(wrapped.calls == 1, 'Unexpected shared replay count'); result['shared_replay_work']['completed'] += 1
                shared += 1; replay_loss = float(torch.mean((replay[:, 2]-target)**2)); require(replay_loss == spec['saved_loss'], 'Archived N1 replay failed')
                for core in (lm, adam): core.fine = fine; core.reporting_truth = np.asarray(info['truth'], np.float64)
                lm.current = lm.forward(lm.p, 'initial')
                adam.z = torch.tensor(spec['initial_logits'], dtype=torch.float32, requires_grad=True)
                p = original.physical(adam.z); require(torch.equal(p.detach(), torch.from_numpy(lm.p)), 'Exact physical starts differ')
                adam.initial_p = p; adam.initial_y = adam.tensor_forward(p, 'initial')
                method.verify_start(lm.p, p.detach().numpy(), lm.current['values'], adam.initial_y.detach().numpy(),
                    lm.current['loss'], float(adam.value(adam.initial_y).detach()))
                delta = float(np.max(abs(lm.current['values']-replay.numpy()))); loss_delta = abs(lm.current['loss']-spec['saved_loss'])
                require(delta <= 1e-6 and loss_delta <= 1e-10+1e-4*spec['saved_loss'], 'Canonical bridge exceeds E5 bounds')
                row.update(initial_parity=True, initial_physical=lm.p.tolist(), initial_loss=lm.current['loss'],
                    archived_replay_loss=replay_loss, canonical_RGB_delta=delta, canonical_MSE_delta=loss_delta); check()
            result.update(initial_gates_passed=True, admission_wall_seconds=time.perf_counter()-admission_start,
                admission_CPU_seconds=time.process_time()-admission_cpu); end()
            for lm, adam, row in cores:
                phase = lm.prefix; begin(); start = time.perf_counter(); cpu = time.process_time()
                row['LM'] = dict(status='running', history=[]); progress = row['LM']['history']
                def record(item, j):
                    item['arrays'] = store.numeric(lm.prefix+'/jacobian-'+str(item['outer'])+'.npz', lm.last, 1024**2)
                    progress.append(item); lm.pending.clear()
                row['LM'] = method.run_lm(lm, record, check)
                terminal = lm.terminal(); require(terminal['loss'] == lm.current['loss'], 'LM terminal replay changed')
                row['LM'].update(terminal=terminal, evaluation=lm.external('final', lm.p),
                    final_state=store.numeric(lm.prefix+'/final-state.npz', dict(physical=lm.p, damping=np.array(row['LM']['damping']),
                        accepted_updates=np.array(row['LM']['accepted_updates'])), 1024**2))
                row['LM']['wall_seconds'] = time.perf_counter()-start; row['LM']['CPU_seconds'] = time.process_time()-cpu
                lm.work.verify(); cost = lm.work.cost(); row['LM']['render_cost'] = cost; end()
                phase = adam.prefix; begin(); start = time.perf_counter(); cpu = time.process_time(); n = method.matched_adam_updates(cost)
                cfg = plan['optimizer']; opt = torch.optim.Adam([adam.z], lr=cfg['lr'], betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
                adam.opt = opt
                effective = json.loads(json.dumps({k: v for k, v in opt.param_groups[0].items() if k != 'params'}))
                require(effective == plan['optimizer_effective'] and not opt.state, 'Original fresh Adam settings differ')
                row['Adam'] = dict(target_updates=n, optimizer_effective=effective, history=[])
                for state in range(n):
                    check(10); opt.zero_grad()
                    p, y = (adam.initial_p, adam.initial_y) if state == 0 else (original.physical(adam.z), None)
                    if y is None: y = adam.tensor_forward(p, 'update')
                    l = adam.value(y)
                    if state == method.ACCEPTED_TARGET:
                        row['Adam']['matched_update_evaluation'] = adam.external('accepted-20', p.detach().numpy().copy())
                    adam.work.attempt('reverse', 'Adam'); gz, gp = torch.autograd.grad(l, (adam.z, p)); adam.work.complete('reverse', 'Adam')
                    require(bool(torch.isfinite(gz).all()) and bool(torch.isfinite(gp).all()), 'Nonfinite direct Adam gradient')
                    before = adam.z.detach().clone(); adam.z.grad = gz
                    adam.work.attempt('update', 'Adam'); opt.step(); adam.work.complete('update', 'Adam')
                    require(float(opt.state[adam.z]['step']) == state+1 and bool(torch.isfinite(adam.z).all()) and all(bool(torch.isfinite(v).all()) for v in opt.state[adam.z].values()), 'Invalid Adam return/state')
                    adam.p = original.physical(adam.z).detach().numpy().copy()
                    row['Adam']['history'].append(dict(update=state+1, loss_at_state=state, loss=float(l.detach()),
                        physical_before=p.detach().tolist(), physical_after=adam.p.tolist(), scaled_physical_gradient=(gp.numpy()*method.SCALES).tolist(),
                        latent_gradient_norm=float(torch.norm(gz)), latent_displacement_L2=float(torch.norm(adam.z.detach()-before))))
                terminal = adam.terminal(); evaluation = adam.external('work-final', adam.p)
                if n == method.ACCEPTED_TARGET: row['Adam']['matched_update_evaluation'] = evaluation
                row['Adam'].update(updates=n, terminal=terminal, evaluation=evaluation,
                    final_state=store.numeric(adam.prefix+'/final-state.npz', dict(latent=adam.z.detach().numpy(), physical=adam.p,
                        **{k: v.numpy() for k, v in opt.state[adam.z].items()}), 1024**2),
                    wall_seconds=time.perf_counter()-start, CPU_seconds=time.process_time()-cpu, render_cost=adam.work.cost())
                adam.work.verify(); require(adam.work.cost() == method.adam_cost(n) <= cost, 'Work matching failed')
                row['matched_accepted_update_available'] = row['LM']['matched_update_available'] and 'matched_update_evaluation' in row['Adam']
                row['work_cost_difference'] = cost-adam.work.cost()
                require(lm.native == lm.work.done['fit_forward']+lm.work.done['external'] and adam.native == adam.work.done['fit_forward']+adam.work.done['external'], 'Native call counters differ')
                end()
        result['cleanup_verified'] = True; phase = 'terminal_integrity'; validate_freeze(freeze_sha)
        require({n: previous.tape_identity(a) for n, a in arrays.items()} == plan['tapes'] and shared == 2 and
            result['shared_replay_work'] == dict(attempts=2, completed=2), 'Terminal input/tape inventory differs')
        for lm, adam, row in cores: lm.work.verify(); adam.work.verify(); require('evaluation' in row['LM'] and 'evaluation' in row['Adam'], 'Missing fixed endpoint')
        totals = dict(shader_forwards=shared+sum(c.work.done['fit_forward']+c.work.done['external'] for pair in cores for c in pair[:2]),
            reverse_sweeps=sum(c.work.done['reverse'] for pair in cores for c in pair[:2]),
            Adam_returns=sum(a.work.done['updates'] for _, a, _ in cores),
            LM_accepted_steps=sum(l.work.events['accepted_steps'] for l, _, _ in cores),
            Jacobians=sum(l.work.events['jacobians_completed'] for l, _, _ in cores),
            trial_attempts=sum(l.work.events['trials'] for l, _, _ in cores),
            external_forwards=sum(c.work.done['external'] for pair in cores for c in pair[:2]), shared_replays=shared)
        require(all(0 <= n <= plan['maxima'][k] for k, n in totals.items()), 'Aggregate scientific cap exceeded')
        result.update(status='work_and_cleanup_verified', scientific_work_completed=True, terminal_integrity_verified=True, totals=totals)
    except BaseException as exc:
        result.update(status='failed', failure_stage=phase, failure=type(exc).__name__+': '+str(exc))
        for lm, adam, row in cores:
            try:
                states = dict(LM_physical=lm.p, **lm.pending)
                if hasattr(adam, 'z'): states['Adam_latent'] = adam.z.detach().numpy()
                if hasattr(adam, 'opt'): states.update({'Adam/'+k: v.numpy() for k, v in adam.opt.state.get(adam.z, {}).items()})
                row['partial_states'] = store.numeric('failure/'+lm.prefix+'.npz', states, 8*1024**2, allow_nonfinite=True)
            except BaseException as other: row['failure_capture_error'] = str(other)
        raise
    finally:
        selected = store.json('result.json', result)
        seal = store.json('finalization/seal.json', dict(report=selected, outcome='complete' if result['scientific_work_completed'] else result['status'], external_success_required=True))
        actual, _ = read_report(store.root/seal['path'], seal['sha256']); require(actual['report'] == selected, 'Seal mismatch')
    result.update(status='complete', evidence_finalized=True, seal=seal); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--describe', action='store_true'); group.add_argument('--prepare', action='store_true'); group.add_argument('--verify-sha256')
    args = parser.parse_args()
    if args.prepare:
        p = make_plan(); v = json.loads((ROOT/VALIDATION).read_text()); require(v['exit_code'] == v['scientific_queries'] == 0 and v['new_source_sha256'] == {n: p['source_sha256'][n] for n in CODE}, 'Preparation validation differs')
        p['validation_sha256'] = sha(ROOT/VALIDATION)
        with (ROOT/FREEZE).open('x') as f: json.dump(p, f, sort_keys=True, indent=2); f.write('\n')
        print(json.dumps(dict(freeze=FREEZE, sha256=sha(ROOT/FREEZE), execution_authorized=False)))
    elif args.verify_sha256: print(json.dumps(dict(verified=bool(validate_freeze(args.verify_sha256)), execution_authorized=False)))
    else: print(json.dumps(dict(protocol=PROTOCOL, cases=CASES, LM_accepted_target_each=20, max_Adam_updates=498,
        max_forwards=956, max_reverse_sweeps=1062, scientific_execution_authorized=False)))


if __name__ == '__main__': main()
