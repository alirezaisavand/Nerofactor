"""E7 cached physical correction -> original N1 bias projection. CLI prepares only."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from experiments import material_optimizer_e6_v1 as prior
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import require, read_report
from infrastructure.runs import validate_run_id

ROOT = prior.ROOT; BASE = 'e7ae53cf991f8fe597547d8757612be1cf089c0a'
PROTOCOL = 'hybrid-neural-physical-e7-v1'
PARENT = 'ca53a5ec9bcb8279ee7d8f5c84a769ff5aed1eebdc550eb955657c09ceeac8f4'
RAW = Path('/home/isavand/paper/evidence/material-e6-1d760020180f45248f9e0ae41bb6b376')
FREEZE = 'docs/protocols/HYBRID_MATERIAL_E7_V1.freeze.json'
VALIDATION = 'docs/evidence/HYBRID_MATERIAL_E7_PREPARATION_V1.json'
CODE = ('experiments/neural_projection_e7_v1.py', 'experiments/hybrid_material_e7_v1.py',
        'tests/scientific/test_hybrid_material_e7_v1.py', 'docs/protocols/HYBRID_MATERIAL_E7_V1.md')
CAPS = dict(diagnostic_forwards=8, neural_forwards=293, backwards=283, Adam_returns=277,
            projections=4, field_probes=8, spatial_probes=8, constructors=6)
sha = prior.sha


class Work:
    def __init__(self): self.attempts = {k: 0 for k in CAPS}; self.done = dict(self.attempts)
    def attempt(self, key):
        require(key in CAPS and self.attempts[key] < CAPS[key], 'E7 cap: '+key); self.attempts[key] += 1
    def complete(self, key):
        require(self.done[key] < self.attempts[key], 'Unattempted E7 completion'); self.done[key] += 1
    def verify(self): require(self.attempts == self.done, 'Incomplete attempted scientific work')


def verify_native(work, native):
    """Independent original counters; constructor/probe work is separate."""
    work.verify()
    for attempted, completed in (('shader_attempts', 'shader_completed'),
                                 ('reverse_attempts', 'reverse_completed'),
                                 ('optimizer_attempts', 'optimizer_returns')):
        require(all(n[attempted] == n[completed] for n in native), 'Incomplete native work')
    require(sum(n['shader_completed'] for n in native) == work.done['neural_forwards'] and
        sum(n['reverse_completed'] for n in native) == work.done['backwards'] and
        sum(n['optimizer_returns'] for n in native)-600 == work.done['Adam_returns'] and
        sum(n['calibration_forwards'] for n in native) == work.done['constructors'] == 6,
        'Native/aggregate work mismatch')


def make_plan():
    parent = prior.validate_freeze(PARENT); ids = dict(parent['input_sha256'])
    def accepted(name):
        p = ROOT/name; require(p.read_bytes() == subprocess.check_output(['git', 'show', BASE+':'+name], cwd=ROOT), 'Accepted E6 file changed')
        ids[str(p)] = sha(p); return json.loads(p.read_text())
    manifest = accepted('docs/results/material_optimizer_e6_v1/evidence_manifest.json')
    for e in manifest['raw_files']+manifest['sidecars']:
        p = Path(e['path']); require(p.stat().st_size == e['bytes'] and sha(p) == e['sha256'], 'E6 evidence changed'); ids[str(p)] = e['sha256']
    ids[str(ROOT/prior.FREEZE)] = PARENT
    e6 = json.loads((RAW/'result.json').read_text()); require(e6['scientific_work_completed'], 'Completed E6 evidence required')
    cases = []; common_rng = None
    for spec in parent['cases']:
        root = Path(spec['source_run']); r = json.loads((root/'result.json').read_text())
        a = next(a for a in r['outcomes'] if a['material']+'/'+str(a['seed']) == spec['case'] and a['lr'] == .001)
        state = a['checkpoints']['100']; arrays = obs.read_numeric(root, state['selection'])
        require(len(arrays) == 350 and all(float(v) == 100 for n, v in arrays.items() if n.endswith('/step')), 'Exact N1 state100 with Adam required')
        rng = dict(metadata=state['RNG'], arrays={n: obs.array_record(arrays[n]) for n in ('torch_CPU_RNG', 'numpy_MT19937')})
        if common_rng is None: common_rng = rng
        require(rng == common_rng, 'Common unchanged scientific RNG required across saved initializations')
        meta, _ = read_report(root/state['selection']['path'], state['selection']['sha256'])
        ids[str(root/state['selection']['path'])] = state['selection']['sha256']; ids[str(root/meta['archive'])] = meta['archive_sha256']
        identity = obs.parameter_identity({n[10:]: v for n, v in arrays.items() if n.startswith('parameter/')})
        old = next(c for c in e6['cases'] if c['case'] == spec['case']); teachers = {}
        for method in ('LM', 'Adam'):
            selected = old[method]; target = obs.read_numeric(RAW, selected['final_state'])['physical']
            require(target.tolist() == selected['evaluation']['physical'], 'Bound teacher snapshot differs')
            teachers[method] = dict(physical=target.tolist(), inherited_cost=selected['render_cost'], fitting_loss=selected['terminal']['loss'],
                checkpoint=selected['final_state'], status=selected.get('status', 'complete'), updates=selected.get('accepted_updates', selected.get('updates')))
        cost = min(t['inherited_cost']+13 for t in teachers.values()); steps = (cost-11)//4
        require(steps == (115 if spec['case'] == 'M2/103' else 162), 'Predeclared work-match endpoint differs')
        cases.append(dict(**spec, checkpoint=state['selection'], RNG=state['RNG'], optimizer=state['optimizer_effective'],
            parameter_identity=identity, seed=a['seed'], control_updates=steps, teachers=teachers,
            reused_LM_fit=old['LM']['history'][-1]['arrays']))
    sources = dict(parent['source_sha256']); sources.update({n: sha(ROOT/n) for n in CODE})
    diag = {n: prior.previous.tape_identity(prior.previous.tape_arrays(seeds, 24)) for n, seeds in
        dict(accepted=(3000, 3001, 3002, 3003), independent_1=(4000, 4001, 4002, 4003), independent_2=(5000, 5001, 5002, 5003)).items()}
    return dict(schema_version=1, protocol=PROTOCOL, execution_authorized=False, base_commit=BASE, source_sha256=sources,
        input_sha256=ids, cases=cases, limits=CAPS, runtime=parent['runtime'], arithmetic_runtime=parent['arithmetic_runtime'],
        inputs=parent['inputs'], tapes=parent['tapes'], diagnostic_tapes=diag, output_parent=parent['output_parent'],
        resources=parent['resources'], storage=parent['storage'], projection=dict(radial_gauge=.5, physical_atol=2e-6, Q_rtol=1e-5, encoding_atol=2e-5),
        canonical_probe_points=[[x, y, 0.] for x in (-.1, 0., .1) for y in (-.1, 0., .1)],
        spatial_probe_points=[[.1, 0., 0.], [0., .1, 0.], [-.1, -.1, 0.]])


def validate_freeze(selected):
    p, _ = read_report(ROOT/FREEZE, selected)
    require(p['protocol'] == PROTOCOL and p['execution_authorized'] is False and p['limits'] == CAPS and
        [c['case'] for c in p['cases']] == list(prior.CASES) and [c['control_updates'] for c in p['cases']] == [115, 162], 'Frozen E7 design differs')
    for n, h in p['source_sha256'].items(): require(sha(ROOT/n) == h, 'Source changed: '+n)
    for n, h in p['input_sha256'].items(): require(sha(n) == h, 'Input changed: '+n)
    v, _ = read_report(ROOT/VALIDATION, p['validation_sha256'])
    require(v['exit_code'] == v['failed'] == v['errors'] == v['skipped'] == v['scientific_queries'] == 0 and
        v['new_source_sha256'] == {n: p['source_sha256'][n] for n in CODE} and sha(ROOT/v['JUnit']['path']) == v['JUnit']['sha256'], 'E7 tests differ')
    return p


def validate_review(review, selected):
    require(set(review) == {'phase', 'scientific_execution_authorized', 'freeze_sha256', 'reviewed_commit', 'reviewed_branch', 'run_id',
        'output_root', 'external_watchdog'} and review['phase'] == 'E.7' and review['scientific_execution_authorized'] is True, 'Explicit E7 execution approval required')
    p = validate_freeze(selected); git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    require(review['freeze_sha256'] == selected and review['external_watchdog'] == dict(wall_seconds=1800, kill_grace_seconds=10), 'E7 watchdog/freeze mismatch')
    require(git('rev-parse', 'HEAD') == review['reviewed_commit'] and git('branch', '--show-current') == review['reviewed_branch'] == 'feat/neural-material-early-step-e2a' and not git('status', '--porcelain'), 'Clean reviewed E7 commit required')
    validate_run_id(review['run_id']); path = Path(review['output_root'])
    require(path.is_absolute() and path.resolve() == path and path.parent == Path(p['output_parent']) and path.name == review['run_id'] and path.name.startswith('material-e7-') and not path.exists(), 'Fresh E7 output required')
    require(shutil.disk_usage(path.parent).free >= p['storage']['minimum_free_bytes'], 'Insufficient storage'); return p


def execute_reviewed(review_path, review_sha, freeze_sha):
    review, _ = read_report(Path(review_path), review_sha); plan = validate_review(review, freeze_sha)
    import platform
    import torch
    from experiments import neural_material_recovery_v1 as h
    from experiments import neural_material_continuation_e3 as restorer
    from experiments import neural_projection_e7_v1 as projection
    require(dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device=str(torch.get_default_device()),
        torch_threads=torch.get_num_threads()) == plan['runtime'] and h.previous.accepted.arithmetic_runtime() == plan['arithmetic_runtime'], 'Accepted runtime required')
    work = Work(); store = obs.Store(review['output_root'], plan['output_parent'], total_cap=plan['storage']['output_cap_bytes'])
    store.json('execution-review.json', review)
    result = dict(protocol=PROTOCOL, status='running', freeze_sha256=freeze_sha, attempts=work.attempts, completed=work.done,
        cases=[], diagnostic=[], cleanup_verified=False, scientific_work_completed=False, terminal_integrity_verified=False, evidence_finalized=False)
    cores = []; phase = 'setup'; started = time.perf_counter()
    common = obs.read_numeric(plan['cases'][0]['source_run'], plan['cases'][0]['checkpoint'])
    try:
        with restorer.saved_rng_context(common, plan['cases'][0]['RNG']):
            with restorer.prior.limits() as (check, begin, end):
                begin(); rp = ROOT/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'; require(sha(rp) == plan['inputs']['accepted_3D1_SHA256'], 'Reference changed')
                reference = json.loads(rp.read_text()); observations = reference['scene']['observations']
                ft, et, tape_manifest = h.inputs(reference); require({k: tape_manifest[k]['arrays'] for k in ('fit', 'evaluation')} == dict(fit=plan['tapes']['accepted'], evaluation=plan['tapes']['evaluation']), 'Runtime tapes differ')
                shader = h.previous.original.MaterialShaderV1('A2', 'atan2'); require(shader.identity == plan['inputs']['shader'], 'Shader changed')
                fit = h.MaterialBatch(reference['scene'], observations[:24], ft); evaluation = h.MaterialBatch(reference['scene'], observations, et)
                phase = 'M3_independent_tapes'
                m3 = plan['cases'][1]; m3info = next(m for m in reference['materials'] if m['material'] == 'M3'); fine = np.asarray(m3info['targets']['fine'], np.float64)
                old = obs.read_numeric(RAW, m3['reused_LM_fit']); require(old['physical'].tolist() == m3['teachers']['LM']['physical'], 'Reused diagnostic point differs')
                diagnostic_arrays = {}
                for name, seeds in dict(accepted=(3000, 3001, 3002, 3003), independent_1=(4000, 4001, 4002, 4003), independent_2=(5000, 5001, 5002, 5003)).items():
                    arrays = prior.previous.tape_arrays(seeds, 24); diagnostic_arrays[name] = arrays
                    require(prior.previous.tape_identity(arrays) == plan['diagnostic_tapes'][name], 'Diagnostic draws changed')
                    batch = h.previous.original.ObservationBatch(reference['scene'], observations[:24], h.previous.original.RandomInputs(*(torch.from_numpy(arrays[k]) for k in ('u', 'v', 'azimuth'))), 4)
                    for point in ('truth', 'LM', 'Adam'):
                        p = np.asarray(m3info['truth'] if point == 'truth' else m3['teachers'][point]['physical'], np.float32)
                        reused = name == 'accepted' and point == 'LM'
                        if reused: values = old['predictions'].copy()
                        else:
                            work.attempt('diagnostic_forwards')
                            with torch.no_grad(): values = batch(shader, torch.from_numpy(p.copy())).numpy()
                            work.complete('diagnostic_forwards')
                        loss = float(torch.mean((torch.from_numpy(values[:, 2])-torch.from_numpy(fine[:24, 2].astype(np.float32)))**2))
                        if name == 'accepted' and point != 'truth': require(loss == m3['teachers'][point]['fitting_loss'], 'Original diagnostic replay differs')
                        result['diagnostic'].append(dict(point=point, tapes=name, loss=loss, reused=reused, errors=prior.previous.component_errors(values, fine[:24]),
                            arrays=store.numeric('diagnostic/'+name+'-'+point+'.npz', dict(physical=p, predictions=values), 1024**2)))
                end(); phase = 'restore_all_N1'; begin()
                class Core:
                    def __init__(self, spec, arm, info, state):
                        construction_started = time.perf_counter()
                        self.spec = spec; self.arm = arm; self.prefix = spec['case'].replace('/', '-').lower()+'-'+arm
                        self.fine = np.asarray(info['targets']['fine'], np.float64); self.truth = np.asarray(info['truth'], np.float64)
                        self.target = torch.from_numpy(self.fine[:24, 2].astype(np.float32)); self.row = dict(arm=arm, status='restoring')
                        work.attempt('constructors'); self.model = h.field.make_material('N1', torch.from_numpy(state['buffer/initial_latent'].copy()), spec['seed']); work.complete('constructors')
                        self.opt = h.optimizer_for(self.model)
                        for g in self.opt.param_groups: g['lr'] = .001
                        self.opt.defaults['lr'] = .001
                        self.row['restoration'] = projection.restore100(self.model, self.opt, state, spec['RNG'], spec['optimizer'], spec['parameter_identity'])
                        self.c = h.counts('N1'); self.c.update(optimizer_attempts=100, optimizer_returns=100)
                        self.row['construction_restore_seconds'] = time.perf_counter()-construction_started
                    def measure(self, state, backward, save=False):
                        self.opt.zero_grad(); work.attempt('neural_forwards')
                        with (obs.GraphObserver(self.model, expected_relus=22) if backward else torch.no_grad()) as observer:
                            if backward: observer.begin()
                            values, mat, _ = fit.render(self.model, observer.wrap_shader(shader) if backward else shader, self.c)
                            work.complete('neural_forwards'); loss = h.loss_value(values, self.target)
                            row = dict(state=state, loss=float(loss.detach()), physical=mat.physical[0].detach().tolist(), encoding=mat.encoding[0].detach().tolist())
                            if backward:
                                work.attempt('backwards'); groups = h.backward(self.model, loss, self.c); work.complete('backwards'); signals, graph = observer.collect()
                                row['gradient_groups'] = groups; row['global_gradient_L2'] = math.sqrt(sum(g['squared_norm'] for g in groups.values()))
                                row['shader_gradients'] = {n: graph['nodes']['shader.'+n]['loss_gradient'] for n in ('mx', 'my', 'albedo', 'metallic', 'encoding')}
                                row['heads'] = {name: dict(probabilities=signals[name+'.sigmoid'].tolist(), logit_gradient_L2=float(np.linalg.norm(signals[name+'.logits.loss_gradient'])),
                                    sigmoid_gradient_L2=float(np.linalg.norm(signals[name+'.sigmoid.loss_gradient']))) for name, _ in projection.HEADS}
                                row['exact_sigmoid_boundaries'] = sum(int(np.sum((signals[n+'.sigmoid'] == 0)|(signals[n+'.sigmoid'] == 1))) for n, _ in projection.HEADS)
                                if save: row['signals'] = store.numeric(self.prefix+'/signals-'+str(state)+'.npz', signals, 1024**2)
                        return row
                    def capture(self, label):
                        work.attempt('field_probes'); material, logits = projection.capture(self.model, torch.tensor(plan['canonical_probe_points'], dtype=torch.float32)); work.complete('field_probes')
                        require(torch.equal(material.physical, material.physical[:1].expand_as(material.physical)), 'Original N1 constant-anchor convention changed')
                        selected = store.numeric(self.prefix+'/'+label+'.npz', dict(physical=material.physical.detach().numpy(), encoding=material.encoding.detach().numpy(), logits=logits.numpy()), 1024**2)
                        return material, logits, selected
                    def spatial(self, label):
                        work.attempt('spatial_probes'); points = torch.tensor(plan['spatial_probe_points'], dtype=torch.float32)
                        with torch.no_grad(): mx, my, metal, kd, encoding, raw = self.model._predict(self.model, points)
                        work.complete('spatial_probes')
                        return store.numeric(self.prefix+'/'+label+'.npz', dict(points=points.numpy(), widths=torch.cat((mx, my), -1).numpy(),
                            albedo=kd.numpy(), metallic=metal.numpy(), encoding=encoding.numpy(), rotation_raw=raw.numpy()), 1024**2)
                    def external(self):
                        c = h.counts('N1'); c['calibration_forwards'] = 0; work.attempt('neural_forwards')
                        with torch.no_grad(): values, mat, _ = evaluation.render(self.model, shader, c, evaluation=True)
                        work.complete('neural_forwards'); prediction = values.numpy().astype(np.float64); metrics = {}
                        for split in ('fit', 'held_out', 'relight'):
                            mask = [o['split'] == split for o in observations]
                            metrics[split] = dict(errors={k: h.previous.original.radiance_errors(prediction[mask, i], self.fine[mask, i]) for i, k in enumerate(('diffuse', 'specular', 'total'))}, **prior.previous.component_errors(prediction[mask], self.fine[mask]))
                        p = mat.physical[0].numpy(); errors = diagnostic.recovery_errors(p, self.truth); self.row['external_native'] = c
                        return dict(physical=p.tolist(), metrics=metrics, recovery_errors=errors, success=h.previous.original.assess_success(errors, {s: m['errors'] for s, m in metrics.items()}),
                            arrays=store.numeric(self.prefix+'/external.npz', dict(predictions=prediction, physical=p, encoding=mat.encoding[0].numpy()), 1024**2))
                for spec in plan['cases']:
                    state = obs.read_numeric(spec['source_run'], spec['checkpoint']); info = next(m for m in reference['materials'] if m['material'] == spec['material'])
                    case = dict(case=spec['case'], arms=[]); result['cases'].append(case); pair = []
                    for arm in ('control', 'hybrid_LM', 'hybrid_Adam'):
                        core = Core(spec, arm, info, state); core.state = state; cores.append(core); pair.append(core); case['arms'].append(core.row)
                        core.initial = core.measure(100, arm == 'control', save=arm == 'control')
                        require(core.initial['physical'] == spec['physical'] and core.initial['loss'] == spec['saved_loss'], 'Exact restored N1 parity failed')
                        core.row.update(initial=core.initial, native=core.c, status='initial_verified')
                    control = pair[0]; control.base_material, control.base_logits, selected = control.capture('baseline-probe'); case['baseline_probe'] = selected
                    case['baseline_spatial'] = control.spatial('baseline-spatial'); case['initial_parameter_identity'] = spec['parameter_identity']
                    case['_cores'] = pair
                end(); result['all_initial_gates_passed'] = True
                for case in result['cases']:
                    pair = case.pop('_cores'); control = pair[0]; spec = control.spec
                    for core, teacher_name in zip(pair[1:], ('LM', 'Adam')):
                        phase = core.prefix; arm_started = time.perf_counter(); begin(); work.attempt('projections'); teacher = spec['teachers'][teacher_name]; desired = torch.tensor(teacher['physical'], dtype=torch.float32)
                        with projection.bias_transaction(core.model, control.base_logits, desired) as transfer:
                            work.complete('projections'); mat, logits, selected = core.capture('proposed-probe')
                            realized = projection.realization(mat.physical[0].numpy(), mat.encoding[0].numpy(), desired.numpy())
                            core.row.update(teacher=teacher, proposed_probe=selected, realization=realized, projected_bias_names=transfer['bias_names'], bias_delta=transfer['delta'].tolist())
                            q = torch.sigmoid(logits); realized['nonsaturated'] = bool(((q > 0)&(q < 1)).all())
                            if realized['passed'] and realized['nonsaturated']:
                                trial = core.measure(100, True, save=True); core.row['trial'] = trial
                                agreement = abs(trial['loss']-teacher['fitting_loss']) <= 1e-10+1e-4*max(trial['loss'], teacher['fitting_loss'])
                                core.row['teacher_loss_agreement'] = agreement
                                take = projection.accept(core.initial['loss'], trial['loss'], agreement)
                            else: take = False
                            if take: transfer['commit']()
                            core.row['accepted'] = take
                        current = obs.parameter_arrays(core.model); changed = transfer['bias_names']
                        require(all(np.array_equal(a, core.state['parameter/'+n]) for n, a in current.items() if n not in changed) and
                            all(np.array_equal(b.detach().numpy(), core.state['buffer/'+n]) for n, b in core.model.named_buffers()), 'Projection modified non-bias state or calibration')
                        if not take: require(obs.parameter_identity(current) == spec['parameter_identity'], 'Rejected projection did not restore exact weights')
                        core.row.update(status='projected' if take else 'projection_rejected', spatial=core.spatial('final-spatial'), evaluation=core.external())
                        a, rng = restorer.prior.state_arrays(core.model, core.opt); a = {n: v for n, v in a.items() if not n.startswith('adam/')}
                        core.row['checkpoint'] = dict(role='weights_only_not_resumable', RNG=rng, selection=store.numeric(core.prefix+'/final-weights.npz', a, 8*1024**2))
                        core.row['fitting_loss'] = core.row['trial']['loss'] if take else core.initial['loss']; core.row['new_renderer_cost'] = 13 if 'trial' in core.row else 9
                        core.row['including_inherited_cost'] = teacher['inherited_cost']+core.row['new_renderer_cost']; core.row['arm_seconds'] = time.perf_counter()-arm_started; end()
                    core = control; phase = core.prefix; arm_started = time.perf_counter(); begin(); path = store.path(core.prefix+'/trajectory.jsonl'); path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as stream:
                        for update in range(spec['control_updates']):
                            check(10); row = core.initial if update == 0 else core.measure(100+update, True)
                            work.attempt('Adam_returns'); h.step(core.model, core.opt, core.c); work.complete('Adam_returns')
                            row = dict(row, completed_new_updates=update+1, absolute_Adam_step=101+update)
                            data=(json.dumps(row, sort_keys=True, allow_nan=False)+'\n').encode(); store.account(len(data)); stream.write(data); stream.flush()
                    terminal = core.measure(100+spec['control_updates'], True, save=True)
                    _, _, capture = core.capture('final-probe'); core.row.update(status='complete', terminal=terminal, probe=capture,
                        spatial=core.spatial('final-spatial'), evaluation=core.external(), trajectory=dict(path=str(path.relative_to(store.root)), sha256=sha(path)))
                    a, rng = restorer.prior.state_arrays(core.model, core.opt)
                    require(all(float(v) == 100+spec['control_updates'] for n, v in a.items() if n.endswith('/step')), 'Absolute Adam counters differ')
                    core.row['checkpoint'] = dict(role='continuation', RNG=rng, selection=store.numeric(core.prefix+'/final-state.npz', a, 18*1024**2))
                    core.row['new_renderer_cost'] = 4*spec['control_updates']+11
                    core.row['arm_seconds'] = time.perf_counter()-arm_started
                    end()
        result['cleanup_verified'] = True; phase = 'terminal_integrity'; validate_freeze(freeze_sha); h.verify_tapes(ft, et, tape_manifest)
        require(all(prior.previous.tape_identity(a) == plan['diagnostic_tapes'][n] for n, a in diagnostic_arrays.items()), 'Diagnostic tapes mutated')
        verify_native(work, [n for core in cores for n in (core.c, core.row['external_native'])])
        require(len(result['cases']) == 2 and all(len(c['arms']) == 3 and all('evaluation' in a for a in c['arms']) for c in result['cases']), 'Incomplete endpoint panel')
        result.update(status='work_and_cleanup_verified', scientific_work_completed=True, terminal_integrity_verified=True)
    except BaseException as exc:
        result.update(status='failed', failure_stage=phase, failure=type(exc).__name__+': '+str(exc))
        for c in result['cases']: c.pop('_cores', None)
        for core in cores:
            try:
                a, rng = restorer.prior.state_arrays(core.model, core.opt)
                core.row['partial_state'] = dict(RNG=rng, selection=store.numeric(core.prefix+'/failure.npz', a, 18*1024**2, allow_nonfinite=True))
            except BaseException as other: core.row['partial_capture_error'] = str(other)
        raise
    finally:
        import resource
        result['wall_seconds'] = time.perf_counter()-started
        result['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        for c in result['cases']: c.pop('_cores', None)
        selected = store.json('result.json', result)
        seal = store.json('finalization/seal.json', dict(report=selected, outcome='complete' if result['scientific_work_completed'] else result['status'], external_success_required=True))
        check_seal, _ = read_report(store.root/seal['path'], seal['sha256']); require(check_seal['report'] == selected, 'Final seal mismatch')
    result.update(status='complete', evidence_finalized=True, seal=seal); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--describe',action='store_true'); g.add_argument('--prepare',action='store_true'); g.add_argument('--verify-sha256'); args=parser.parse_args()
    if args.prepare:
        p=make_plan(); v=json.loads((ROOT/VALIDATION).read_text()); require(v['exit_code']==v['scientific_queries']==0 and v['new_source_sha256']=={n:p['source_sha256'][n] for n in CODE}, 'E7 validation differs'); p['validation_sha256']=sha(ROOT/VALIDATION)
        with (ROOT/FREEZE).open('x') as f:json.dump(p,f,sort_keys=True,indent=2);f.write('\n')
        print(json.dumps(dict(freeze=FREEZE,sha256=sha(ROOT/FREEZE),execution_authorized=False)))
    elif args.verify_sha256:print(json.dumps(dict(verified=bool(validate_freeze(args.verify_sha256)),execution_authorized=False)))
    else:print(json.dumps(dict(protocol=PROTOCOL,limits=CAPS,scientific_execution_authorized=False)))


if __name__=='__main__':main()
