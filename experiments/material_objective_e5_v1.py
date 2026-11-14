"""E5 physical-objective diagnosis. CLI prepares only; real calls need a review.

Imports are scientific-execution free. Synthetic functions accept fake renderers;
the accepted shader and direct-parameter control are imported only after admission.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostics
from infrastructure.runs import validate_run_id
from infrastructure.signatures import require, read_report

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a832f020d4d9618295ccfe818131a0f8193eb248'
PROTOCOL = 'material-objective-e5-v1'
FREEZE = 'docs/protocols/MATERIAL_OBJECTIVE_E5_V1.freeze.json'
VALIDATION = 'docs/evidence/MATERIAL_OBJECTIVE_E5_PREPARATION_V1.json'
CODE = ('experiments/material_objective_e5_v1.py',
        'tests/scientific/test_material_objective_e5_v1.py',
        'docs/protocols/MATERIAL_OBJECTIVE_E5_V1.md')
E4_FREEZE = 'a3318d2e2633d5fd7e952224e65293ec0fb810e4de04f2c38f887e2a3dadf643'
E4_RAW = Path('/home/isavand/paper/evidence/material-e4-5290efddb03c4842837176191a90558d')
E3_RAW = Path('/home/isavand/paper/evidence/neural-e3-b4d1389b174e4b20b2f892725050a0ae')
TAPE_SEEDS = {'accepted': (3000, 3001, 3002, 3003),
              'independent_1': (4000, 4001, 4002, 4003),
              'independent_2': (5000, 5001, 5002, 5003)}
PARAMETERS = ('mx', 'my', 'theta', 'k_R', 'k_G', 'k_B', 'metallic')
SCALES = np.array([.96, .96, np.pi, 1., 1., 1., 1.])
FD_STEPS = (1e-3, 5e-4)
FORWARDS = dict(replay=2, base=9, jacobian=63, finite_difference=252,
                external=5, direct=51, gauge=3)
REVERSES = dict(gradient=10, JVP=126, direct=50)
UPDATES = 50
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tape_arrays(seeds, points):
    """Exactly original make_tapes draw order; private NumPy RNG, no shader/Torch."""
    columns = [[], [], []]
    for seed in seeds:
        rng = np.random.RandomState(seed)
        for out, shape in zip(columns, ((points, 1024), (points, 1024), (points, 1, 1))):
            out.append(rng.uniform(size=shape).astype(np.float32))
    return dict(zip(('u', 'v', 'azimuth'), (np.concatenate(a) for a in columns)))


def tape_identity(arrays):
    return {k: dict(dtype=str(a.dtype), shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())
            for k, a in arrays.items()}


class Work:
    def __init__(self):
        self.attempts = {kind: {k: 0 for k in limits} for kind, limits in
                         (('forward', FORWARDS), ('reverse', REVERSES), ('update', {'direct': UPDATES}))}
        self.done = json.loads(json.dumps(self.attempts))

    def attempt(self, kind, phase):
        limits = {'forward': FORWARDS, 'reverse': REVERSES, 'update': {'direct': UPDATES}}
        require(kind in limits and phase in limits[kind], 'Unknown E5 work phase')
        require(self.attempts[kind][phase] < limits[kind][phase], 'E5 cap: '+kind+'/'+phase)
        self.attempts[kind][phase] += 1

    def complete(self, kind, phase):
        require(self.done[kind][phase] < self.attempts[kind][phase], 'Unattempted completion')
        self.done[kind][phase] += 1

    def verify_complete(self):
        expected = dict(forward=FORWARDS, reverse=REVERSES, update=dict(direct=UPDATES))
        require(self.attempts == self.done == expected, 'E5 incomplete scientific work')


def columns(function, physical, expected, work, pending, check=lambda: None):
    """Original two-reverse JVP construction, seven scaled physical columns.

    The implementation follows the accepted directional_jvp control. Each
    primal is checked bitwise; no finite-difference fallback is allowed.
    """
    import torch
    result = []
    for j, scale in enumerate(SCALES):
        check(); x = physical.detach().requires_grad_(); y = function(x, 'jacobian')
        require(torch.equal(y.detach(), expected), 'JVP primal changed with fixed samples')
        w = torch.zeros_like(y, requires_grad=True)
        work.attempt('reverse', 'JVP')
        vjp, = torch.autograd.grad(y, x, grad_outputs=w, create_graph=True)
        work.complete('reverse', 'JVP')
        direction = torch.zeros_like(x); direction[j] = float(scale)
        work.attempt('reverse', 'JVP')
        value, = torch.autograd.grad(vjp, w, grad_outputs=direction)
        work.complete('reverse', 'JVP')
        require(bool(torch.isfinite(value).all()), 'Nonfinite physical JVP')
        result.append(value.detach().numpy().copy()); pending['J_partial'] = np.stack(result, -1)
    return np.stack(result, -1)


def finite_differences(function, physical, pending, check=lambda: None):
    import torch
    matrices = []
    for h in FD_STEPS:
        cols = []
        for j, scale in enumerate(SCALES):
            check(); offset = torch.zeros_like(physical); offset[j] = h*float(scale)
            with torch.no_grad():
                low = function(physical-offset, 'finite_difference').numpy().astype(np.float64)
                high = function(physical+offset, 'finite_difference').numpy().astype(np.float64)
            cols.append((high-low)/(2*h)); pending['FD_partial'] = np.stack(cols, -1)
        matrices.append(np.stack(cols, -1))
    return np.stack(matrices)


def derivative_agreement(jacobian, fd):
    """Predeclared float32, piecewise-sampler diagnostic gate; no adaptive h."""
    j = np.asarray(jacobian, np.float64).reshape(-1, 7)
    f = np.asarray(fd, np.float64).reshape(2, -1, 7)
    errors = np.linalg.norm(f-j[None], axis=1)
    limits = 1e-4 + .05*np.linalg.norm(j, axis=0)
    # Both fixed steps must agree: no favorable step selection after observation.
    return dict(passed=bool((errors <= limits).all()), L2_column_errors=errors.tolist(),
                L2_column_limits=limits.tolist(), fixed_steps=list(FD_STEPS),
                FD_step_discrepancy_L2=np.linalg.norm(f[0]-f[1], axis=0).tolist(),
                limitation='Pathwise float32 derivative; horizon/clamp discontinuities may fail this gate')


def spectra(matrix):
    j = np.asarray(matrix, np.float64)
    require(j.ndim == 2 and j.shape[1] == 7 and np.isfinite(j).all(), 'Finite seven-column sensitivity required')
    _, s, vt = np.linalg.svd(j, full_matrices=False)
    norms = np.linalg.norm(j, axis=0); denom = np.outer(norms, norms)
    correlations = np.divide(j.T@j, denom, out=np.zeros((7, 7)), where=denom > 0)
    weak = vt[-1]; weak *= 1 if weak[np.argmax(abs(weak))] >= 0 else -1
    return dict(singular_values=s.tolist(), effective_rank=int(np.sum(s >= 1e-4*s[0])) if s[0] > 0 else 0,
                rank_threshold_relative=1e-4, condition_ratio=float(s[0]/s[-1]) if s[-1] > 0 else None,
                weak_scaled_direction=weak.tolist(), column_norms=norms.tolist(),
                column_correlations=correlations.tolist(), Gauss_Newton_eigenvalues=(2*s*s).tolist(),
                Gauss_Newton_matrix=(2*j.T@j).tolist())


def sensitivity(jacobian, physical):
    j = np.asarray(jacobian, np.float64).reshape(24, 3, 3, 7)
    p = np.asarray(physical); order = [0, 1, 2, 3, 4, 5, 6]
    if p[0] > p[1]: order[:2] = [1, 0]
    # Reporting permutation only: never canonicalize renderer inputs/samples.
    aligned = j[..., order]
    total = aligned[:, 2].reshape(72, 7)/np.sqrt(72)
    components = aligned[:, :2].reshape(144, 7)/np.sqrt(72)
    gap = abs(p[0]**2-p[1]**2)/(p[0]**2+p[1]**2)
    return dict(total=spectra(total), separate_components=spectra(components),
                diffuse=spectra(aligned[:, 0].reshape(72, 7)/np.sqrt(72)),
                specular=spectra(aligned[:, 1].reshape(72, 7)/np.sqrt(72)),
                reporting_column_order=order, scales=SCALES.tolist(), predicted_anisotropy_gap=float(gap),
                angle_scored=bool(gap >= .1),
                limitation='Local, scaled, fixed-tape radiance sensitivity; not global identifiability or inverse covariance')


def component_errors(predictions, target):
    delta = np.asarray(predictions, np.float64)-np.asarray(target, np.float64)
    d, s, t = (delta[:, i] for i in range(3))
    return dict(component_MSE=[float(np.mean(x*x)) for x in (d, s, t)],
                diffuse_specular_inner_product=float(np.sum(d*s)),
                normalized_residual_inner_product=float(np.sum(d*s)/max(np.linalg.norm(d)*np.linalg.norm(s), 1e-30)),
                total_squared_error=float(np.sum(t*t)), component_squared_error=float(np.sum(d*d)+np.sum(s*s)))


def archived_physical_gradient(signals):
    """Chain-derived from existing shader-input gradients, not a new AD call.

    Angle displacement rotates the actual incoming encoding at fixed radius.
    This does not replace the new canonical-encoding derivative admission.
    """
    encoding = signals['shader.encoding'].astype(np.float64)
    eg = signals['shader.encoding.loss_gradient'].astype(np.float64)
    angle = np.sum(eg*np.column_stack((-2*encoding[:, 1], 2*encoding[:, 0])))
    sums = {n: signals['shader.'+n+'.loss_gradient'].astype(np.float64).sum(axis=0)
            for n in ('mx', 'my', 'albedo', 'metallic')}
    g = np.r_[sums['mx'], sums['my'], angle, sums['albedo'], sums['metallic']]
    require(g.shape == (7,) and np.isfinite(g).all(), 'Invalid archived physical gradient')
    scaled = SCALES*g
    return dict(physical_gradient=g.tolist(), scaled_gradient=scaled.tolist(),
                scaled_norm=float(np.linalg.norm(scaled)),
                provenance='Offline chain from archived actual shader-input gradients; fixed encoding radius; no new derivative query')


def admit_inputs():
    from experiments import material_parameterization_e4_v1 as e4
    parent = e4.validate_freeze(E4_FREEZE)
    ids = dict(parent['input_sha256'])
    def accepted(name):
        p = ROOT/name
        require(p.read_bytes() == subprocess.check_output(['git', 'show', BASE+':'+name], cwd=ROOT), 'Accepted E4 file changed')
        ids[str(p)] = sha(p); return json.loads(p.read_text())
    panel = accepted('docs/results/material_parameterization_e4_v1/comparison.json')
    manifest = accepted('docs/results/material_parameterization_e4_v1/evidence_manifest.json')
    for record in manifest['new_raw_files']+manifest['sidecars']:
        p = Path(record['path']); require(p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], 'E4 evidence changed')
        ids[str(p)] = record['sha256']
    paired = next(a for a in panel['endpoints'] if a['case'] == 'M2/103')
    baseline = paired['baseline']; candidate = paired['AS1']
    points = dict(truth=parent['inputs']['targets']['M2']['truth'],
                  N1=baseline['physical100'], AS1=candidate['physical100'])
    truth = np.asarray(points['truth'], np.float64); swapped = truth.copy()
    swapped[:2] = truth[1::-1]; swapped[2] -= math.pi/2
    points['truth_swapped'] = swapped.tolist()
    replays = {}
    for name, root in (('N1', E3_RAW), ('AS1', E4_RAW)):
        report_path = root/'result.json'; require(str(report_path) in ids and sha(report_path) == ids[str(report_path)], 'Unadmitted endpoint report')
        report = json.loads(report_path.read_text()); a = next(a for a in report['outcomes'] if
            a['material'] == 'M2' and a['seed'] == 103 and a.get('lr', .001) == .001)
        require(a['status'] == 'complete', 'Selected endpoint is incomplete')
        p = root/a['trajectory']['path']; require(sha(p) == a['trajectory']['sha256'], 'Endpoint trace changed'); ids[str(p)] = sha(p)
        row = next(json.loads(l) for l in p.read_text().splitlines() if json.loads(l)['state'] == 100 and json.loads(l).get('kind', 'fit') == 'fit')
        signals = obs.read_numeric(root, row['signals']); selection = root/row['signals']['path']
        m, _ = read_report(selection, row['signals']['sha256'])
        ids[str(selection)] = row['signals']['sha256']; ids[str(root/m['archive'])] = m['archive_sha256']
        expected = np.asarray(points[name], np.float32)
        require(all(np.array_equal(signals['shader.'+key], np.broadcast_to(value, signals['shader.'+key].shape))
            for key, value in (('mx', expected[:1]), ('my', expected[1:2]), ('albedo', expected[3:6]), ('metallic', expected[6:]))), 'Saved endpoint shader inputs differ')
        replays[name] = dict(root=str(root), signals=row['signals'], loss=row['loss'],
                             saved_physical_gradient=archived_physical_gradient(signals))
        if name == 'N1':
            z = np.concatenate((signals['mx_predictor.logits'].ravel(), signals['my_predictor.logits'].ravel(),
                expected[2:3], signals['kd_predictor.logits'].ravel(), signals['metallic_predictor.logits'].ravel()))
            require(z.shape == (7,) and z.dtype == np.float32, 'Exact direct-control logits missing')
    return parent, ids, points, replays, z.tolist()


def make_plan():
    parent, ids, points, replays, z = admit_inputs()
    sources = dict(parent['source_sha256']); sources.update({p: sha(ROOT/p) for p in CODE})
    seeds = dict(TAPE_SEEDS, evaluation=(1000, 1001, 1002, 1003))
    tapes = {name: tape_identity(tape_arrays(s, 84 if name == 'evaluation' else 24)) for name, s in seeds.items()}
    require(tapes['accepted'] == parent['tapes']['fit']['arrays'] and tapes['evaluation'] == parent['tapes']['evaluation']['arrays'], 'Original draw identities differ')
    return dict(schema_version=1, protocol=PROTOCOL, execution_authorized=False, base_commit=BASE,
                source_sha256=sources, input_sha256=ids, points=points, replays=replays, direct_initial_logits=z,
                tape_seeds=seeds, tapes=tapes, scales=SCALES.tolist(), FD_steps=list(FD_STEPS),
                work=dict(forward=FORWARDS, reverse=REVERSES, update=dict(direct=UPDATES)),
                totals=dict(shader_forwards=385, reverse_sweeps=186, Adam_returns=50, reference_queries=0, neural_queries=0),
                optimizer=dict(lr=.01, betas=[.9, .999], eps=1e-8, weight_decay=0.),
                optimizer_effective=dict(parent['optimizer'], lr=.01),
                runtime=parent['runtime'], arithmetic_runtime=parent['arithmetic_runtime'], inputs=parent['inputs'],
                output_parent=parent['output_parent'], storage=parent['storage'], resources=parent['resources'])


def validate_freeze(selected):
    p, _ = read_report(ROOT/FREEZE, selected)
    require(p['protocol'] == PROTOCOL and p['execution_authorized'] is False and
            p['work'] == dict(forward=FORWARDS, reverse=REVERSES, update=dict(direct=UPDATES)) and
            p['scales'] == SCALES.tolist() and p['FD_steps'] == list(FD_STEPS) and
            p['tape_seeds'] == {k: list(v) for k, v in dict(TAPE_SEEDS, evaluation=(1000, 1001, 1002, 1003)).items()}, 'Frozen E5 design differs')
    for n, h in p['source_sha256'].items(): require(sha(ROOT/n) == h, 'Frozen source changed: '+n)
    for n, h in p['input_sha256'].items(): require(sha(n) == h, 'Accepted input changed: '+n)
    v, _ = read_report(ROOT/VALIDATION, p['validation_sha256'])
    require(v['exit_code'] == v['failed'] == v['errors'] == v['skipped'] == v['scientific_queries'] == 0 and
            v['new_source_sha256'] == {n: p['source_sha256'][n] for n in CODE}, 'E5 synthetic validation differs')
    require(sha(ROOT/v['JUnit']['path']) == v['JUnit']['sha256'], 'E5 JUnit changed')
    return p


def validate_review(review, selected):
    require(set(review) == {'phase', 'scientific_execution_authorized', 'freeze_sha256', 'reviewed_commit',
        'reviewed_branch', 'run_id', 'output_root', 'external_watchdog'} and review['phase'] == 'E.5' and
        review['scientific_execution_authorized'] is True, 'Explicit E5 execution approval required')
    p = validate_freeze(selected); git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    require(review['freeze_sha256'] == selected and review['external_watchdog'] == dict(wall_seconds=1800, kill_grace_seconds=10), 'Frozen watchdog mismatch')
    require(git('rev-parse', 'HEAD') == review['reviewed_commit'] and git('branch', '--show-current') == review['reviewed_branch'] == 'feat/neural-material-early-step-e2a' and not git('status', '--porcelain'), 'Clean reviewed E5 commit required')
    validate_run_id(review['run_id']); out = Path(review['output_root'])
    require(out.is_absolute() and out.resolve() == out and out.parent == Path(p['output_parent']) and
        out.name == review['run_id'] and out.name.startswith('material-e5-') and not out.exists(), 'Fresh E5 output required')
    require(shutil.disk_usage(out.parent).free >= p['storage']['minimum_free_bytes'], 'Insufficient storage')
    return p


class CapturedShader:
    """Read-only observation/explicit archived encoding replay; no shader math edits."""
    def __init__(self, shader, encoding=None):
        self.shader = shader; self.encoding = encoding; self.per_tape = None; self.calls = 0

    def shade(self, *args, **kwargs):
        if self.encoding is not None: args = (*args[:7], self.encoding, *args[8:])
        self.calls += 1; result = self.shader.shade(*args, **kwargs)
        self.per_tape = np.stack([result[k].detach().numpy().reshape(4, -1, 3)
            for k in ('diffuse_rgb', 'specular_rgb', 'linear_rgb')], 2)
        return result


def execute_reviewed(review_path, review_sha, freeze_sha):
    review, _ = read_report(Path(review_path), review_sha); plan = validate_review(review, freeze_sha)
    # No real shader, control or Torch import above the explicit trust/review gate.
    import platform
    import torch
    from experiments import run_known_material_recovery_v1 as original
    from experiments import material_lm_gauge_trajectories_v2 as accepted
    from experiments import neural_material_early_step_e2a as resource_owner
    require(dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        device=str(torch.get_default_device()), torch_threads=torch.get_num_threads()) == plan['runtime'] and
        accepted.arithmetic_runtime() == plan['arithmetic_runtime'], 'Accepted CPU runtime required')
    store = obs.Store(review['output_root'], plan['output_parent'], total_cap=plan['storage']['output_cap_bytes'])
    store.json('execution-review.json', review); work = Work(); pending = {}; native = [0]
    result = dict(protocol=PROTOCOL, status='running', freeze_sha256=freeze_sha, cleanup_verified=False,
        terminal_integrity_verified=False, scientific_work_completed=False, evidence_finalized=False,
        attempts=work.attempts, completed=work.done, measurements=[], evaluations=[], replays=[])
    stage = 'setup'
    try:
        with resource_owner.limits() as (check, begin, end):
            begin()
            reference_path = ROOT/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json'
            require(sha(reference_path) == plan['inputs']['accepted_3D1_SHA256'], 'Accepted reference identity differs')
            reference = json.loads(reference_path.read_text()); scene = reference['scene']; observations = scene['observations']
            target = np.asarray(next(r for r in reference['materials'] if r['material'] == 'M2')['targets']['fine'], np.float64)
            require(target.shape == (84, 3, 3) and hashlib.sha256(target.tobytes()).hexdigest() == plan['inputs']['targets']['M2']['fine_sha256'], 'Accepted target changed')
            target_fit = torch.from_numpy(target[:24, 2].astype(np.float32))
            arrays = {n: tape_arrays(seeds, 84 if n == 'evaluation' else 24) for n, seeds in plan['tape_seeds'].items()}
            require({n: tape_identity(a) for n, a in arrays.items()} == plan['tapes'], 'Frozen sample tapes differ')
            batches = {n: original.ObservationBatch(scene, observations if n == 'evaluation' else observations[:24],
                original.RandomInputs(*(torch.from_numpy(a[k]) for k in ('u', 'v', 'azimuth'))), 4) for n, a in arrays.items()}
            shader = original.MaterialShaderV1('A2', 'atan2'); require(shader.identity == plan['inputs']['shader'], 'A2 shader identity differs')
            truth = np.asarray(plan['points']['truth'], np.float64)
            def forward(p, phase, tape='accepted', capture=None, encoding=None):
                check(); work.attempt('forward', phase); wrapped = CapturedShader(shader, encoding)
                y = batches[tape](wrapped, p)
                require(wrapped.calls == 1 and bool(torch.isfinite(y).all()), 'Invalid actual shader call/output')
                native[0] += wrapped.calls; work.complete('forward', phase)
                if capture is not None: capture['per_tape'] = wrapped.per_tape
                return y
            def loss(y): return torch.mean((y[:, 2]-target_fit)**2)
            def gradient(l, p, phase):
                work.attempt('reverse', phase); g, = torch.autograd.grad(l, p)
                work.complete('reverse', phase); require(bool(torch.isfinite(g).all()), 'Nonfinite physical gradient')
                return g
            def external(label, p):
                capture = {}
                with torch.no_grad(): y = forward(p, 'external', 'evaluation', capture).numpy().astype(np.float64)
                metrics = {}
                for split in ('fit', 'held_out', 'relight'):
                    mask = [o['split'] == split for o in observations]
                    metrics[split] = dict(errors={name: original.radiance_errors(y[mask, i], target[mask, i])
                        for i, name in enumerate(('diffuse', 'specular', 'total'))}, **component_errors(y[mask], target[mask]))
                row = dict(point=label, physical=p.tolist(), metrics=metrics,
                    recovery_errors=diagnostics.recovery_errors(p.numpy(), truth))
                row['arrays'] = store.numeric('external/'+label+'.npz', dict(predictions=y, per_tape=capture['per_tape']), 1024**2)
                result['evaluations'].append(row)
            stage = 'archived_encoding_replay'
            replays = {}
            for label, spec in plan['replays'].items():
                p = torch.tensor(plan['points'][label], dtype=torch.float32)
                signals = obs.read_numeric(Path(spec['root']), spec['signals']); encoding = torch.from_numpy(signals['shader.encoding'].copy())
                with torch.no_grad(): y = forward(p, 'replay', encoding=encoding)
                actual = float(loss(y)); row = dict(point=label, actual_loss=actual, expected_loss=spec['loss'], passed=actual == spec['loss'])
                result['replays'].append(row); store.json('replay/'+label+'.json', row)
                require(row['passed'], 'Archived actual-encoding objective replay failed')
                replays[label] = y.numpy().copy()
            end()
            stage = 'physical_sensitivity'
            canonical_loss = {}
            for label in ('truth', 'N1', 'AS1'):
                for tape in TAPE_SEEDS:
                    begin(); p = torch.tensor(plan['points'][label], dtype=torch.float32, requires_grad=True)
                    result['active_point'] = label+'/'+tape; pending.clear(); capture = {}
                    y = forward(p, 'base', tape, capture); l = loss(y); g = gradient(l, p, 'gradient')
                    pending.update(physical=p.detach().numpy(), predictions=y.detach().numpy(), physical_gradient=g.numpy(), per_tape=capture['per_tape'])
                    if tape == 'accepted' and label != 'truth':
                        delta = float(np.max(abs(y.detach().numpy()-replays[label])))
                        ld = abs(float(l)-plan['replays'][label]['loss'])
                        require(delta <= 1e-6 and ld <= 1e-10+1e-4*plan['replays'][label]['loss'], 'Canonical encoding bridge exceeds frozen diagnostic bounds')
                    else: delta = ld = None
                    fn = lambda x, phase: forward(x, phase, tape)
                    j = columns(fn, p, y.detach(), work, pending, check)
                    fd = finite_differences(fn, p.detach(), pending, check)
                    pending.update(J_scaled=j, FD_scaled=fd); agreement = derivative_agreement(j, fd)
                    r = (y[:, 2].detach().numpy()-target_fit.numpy()).astype(np.float64)
                    chain = 2*j[:, 2].astype(np.float64).reshape(72, 7).T@r.reshape(72)/72
                    scaled = g.numpy().astype(np.float64)*SCALES
                    chain_error = float(np.linalg.norm(chain-scaled))
                    row = dict(point=label, tapes=tape, physical=p.detach().tolist(), objective_float32=float(l.detach()),
                        physical_gradient=g.tolist(), scaled_physical_gradient=scaled.tolist(), gradient_norm=float(np.linalg.norm(scaled)),
                        exact_zero_gradient=bool(np.all(scaled == 0)), local_negative_gradient_directional_derivative=-float(scaled@scaled),
                        J_gradient_chain_error=chain_error, derivative_check=agreement,
                        canonical_encoding_RGB_max_delta=delta, canonical_encoding_MSE_delta=ld,
                        errors=component_errors(y.detach().numpy(), target[:24]),
                        per_tape_MSE=[float(np.mean((v[:, 2].astype(np.float64)-target[:24, 2])**2)) for v in capture['per_tape']],
                        sensitivity=sensitivity(j, p.detach().numpy()))
                    row['arrays'] = store.numeric('sensitivity/'+label+'-'+tape+'.npz', pending, 1024**2)
                    result['measurements'].append(row); store.json('measurements/'+label+'-'+tape+'.json', row)
                    require(agreement['passed'] and chain_error <= 1e-7+1e-3*np.linalg.norm(scaled), 'Physical derivative admission failed; no optimization permitted')
                    if tape == 'accepted': canonical_loss[label] = float(l.detach())
                    pending.clear(); end()
                begin(); external(label, p.detach()); end()
            stage = 'auxiliary_physical_gauge'
            begin(); p = torch.tensor(plan['points']['truth_swapped'], dtype=torch.float32)
            for tape in TAPE_SEEDS:
                with torch.no_grad(): y = forward(p, 'gauge', tape).numpy()
                row = dict(point='truth_swapped', tapes=tape, objective_float32=float(torch.mean((torch.from_numpy(y[:, 2])-target_fit)**2)), errors=component_errors(y, target[:24]))
                row['arrays'] = store.numeric('gauge/'+tape+'.npz', dict(predictions=y, physical=p.numpy()), 1024**2)
                result.setdefault('gauge_measurements', []).append(row)
            external('truth_swapped', p); end()
            stage = 'direct_parameter_probe'
            begin(); z = torch.tensor(plan['direct_initial_logits'], dtype=torch.float32, requires_grad=True)
            require(torch.equal(original.physical(z).detach(), torch.tensor(plan['points']['N1'], dtype=torch.float32)), 'Saved logits do not reproduce exact N1 endpoint')
            cfg = plan['optimizer']; opt = torch.optim.Adam([z], lr=cfg['lr'], betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
            require(not opt.state, 'Direct probe requires explicitly fresh Adam, not neural continuation')
            effective = json.loads(json.dumps({k: v for k, v in opt.param_groups[0].items() if k != 'params'}))
            require(effective == plan['optimizer_effective'], 'Original direct Adam effective options differ')
            result['direct'] = dict(status='running', start='N1 state100', updates=0, optimizer=cfg,
                                    optimizer_effective=effective, history=[])
            for step in range(UPDATES):
                check(10); opt.zero_grad(); p = original.physical(z); y = forward(p, 'direct'); l = loss(y)
                if step == 0: require(float(l.detach()) == canonical_loss['N1'], 'Direct probe initial objective changed')
                g = gradient(l, z, 'direct'); z.grad = g
                before = z.detach().clone(); work.attempt('update', 'direct'); opt.step(); work.complete('update', 'direct')
                require(float(opt.state[z]['step']) == step+1 and bool(torch.isfinite(z).all()) and
                    all(bool(torch.isfinite(t).all()) for t in opt.state[z].values()), 'Nonfinite/incorrect original Adam state')
                actual = original.physical(z).detach().numpy()
                result['direct']['updates'] = step+1
                result['direct']['history'].append(dict(update=step+1, loss_at_state=step, loss=float(l.detach()),
                    latent_gradient_norm=float(torch.norm(g)), displacement_L2=float(torch.norm(z.detach()-before)),
                    physical=actual.tolist(), recovery_errors=diagnostics.recovery_errors(actual, truth)))
            p = original.physical(z).detach().requires_grad_(); y = forward(p, 'direct'); l = loss(y); g = gradient(l, p, 'gradient')
            result['direct'].update(status='complete', final_objective=float(l.detach()), final_physical_gradient=g.tolist(),
                final_errors=component_errors(y.detach().numpy(), target[:24]), final=diagnostics.recovery_errors(p.detach().numpy(), truth))
            store.numeric('direct/final-state.npz', dict(latent=z.detach().numpy(), physical=p.detach().numpy(),
                gradient=g.numpy(), **{k: v.numpy() for k, v in opt.state[z].items()}), 1024**2)
            external('direct_final', p.detach()); store.json('direct/history.json', result['direct']); end()
            require(native[0] == sum(work.done['forward'].values()), 'Native/driver forward counters differ')
            require({n: tape_identity(a) for n, a in arrays.items()} == plan['tapes'], 'Runtime sample tapes mutated')
            work.verify_complete()
        result['cleanup_verified'] = True
        stage = 'terminal_integrity'; validate_freeze(freeze_sha)
        require({n: tape_identity(a) for n, a in arrays.items()} == plan['tapes'], 'Terminal tapes changed')
        work.verify_complete()
        result.update(status='work_and_cleanup_verified', scientific_work_completed=True, terminal_integrity_verified=True, native_shader_calls=native[0])
    except BaseException as exc:
        result.update(status='failed', failure_stage=stage, failure=type(exc).__name__+': '+str(exc))
        if pending:
            try: result['partial_numeric'] = store.numeric('failure/partial.npz', pending, 8*1024**2, allow_nonfinite=True)
            except BaseException as secondary: result['partial_capture_failure'] = str(secondary)
        if 'opt' in locals():
            try:
                result['partial_direct_state'] = store.numeric('failure/direct-state.npz',
                    dict(latent=z.detach().numpy(), **{k: v.numpy() for k, v in opt.state.get(z, {}).items()}),
                    1024**2, allow_nonfinite=True)
            except BaseException as secondary: result['direct_capture_failure'] = str(secondary)
        raise
    finally:
        report = store.json('result.json', result)
        seal = store.json('finalization/seal.json', dict(report=report, outcome='complete' if result['scientific_work_completed'] else result['status'], external_success_required=True))
        verified, _ = read_report(store.root/seal['path'], seal['sha256']); require(verified['report'] == report, 'Final seal mismatch')
    result.update(status='complete', evidence_finalized=True, seal=seal); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--describe', action='store_true'); g.add_argument('--prepare', action='store_true'); g.add_argument('--verify-sha256')
    args = parser.parse_args()
    if args.prepare:
        p = make_plan(); v = json.loads((ROOT/VALIDATION).read_text())
        require(v['exit_code'] == v['scientific_queries'] == 0 and v['new_source_sha256'] == {n: p['source_sha256'][n] for n in CODE}, 'E5 validation mismatch')
        p['validation_sha256'] = sha(ROOT/VALIDATION)
        with (ROOT/FREEZE).open('x') as f: json.dump(p, f, sort_keys=True, indent=2); f.write('\n')
        print(json.dumps(dict(freeze=FREEZE, sha256=sha(ROOT/FREEZE), execution_authorized=False)))
    elif args.verify_sha256:
        print(json.dumps(dict(verified=bool(validate_freeze(args.verify_sha256)), execution_authorized=False)))
    else: print(json.dumps(dict(protocol=PROTOCOL, forwards=sum(FORWARDS.values()), backwards=sum(REVERSES.values()), updates=UPDATES, scientific_execution_authorized=False)))


if __name__ == '__main__': main()
