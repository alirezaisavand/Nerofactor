"""E.13 endpoint observation and CPU metrics. No scientific imports at import time.

The native Trainer/optimizer are neither replaced nor called by these metrics.
The first two updates reuse E.12B's observer inside each full history.
"""
import json
import math
from pathlib import Path
import sys
import time

from infrastructure.bounded_training import TwoUpdateTrace, _TwoUpdatesComplete
from infrastructure.corrected_material_smoke import CorrectedProbe
from infrastructure.signatures import require
from infrastructure.stage_sources import digest
from infrastructure.training_probe import parameter_copies, snapshot, optimizer_record, tensor_copy

PROTOCOL = 'bell-native-three-arm-recovery-e13-v1'
ARMS = ('A_original', 'A_estimator_v1', 'A_corrected_core_v1')
RETURNS = 2000
MILESTONES = (100, 500, 1000, 1500, 2000)
FRAME_TOL = 1e-5


def write_json(path, value):
    """Only fresh named artifacts; reports use a separately owned open stream."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def initialization_gate(actual, expected):
    require(set(actual) == set(expected) == {'parameters', 'splits'}, 'Incomplete initialization parity packet')
    require(actual == expected, 'Cross-arm initial parameters/split parity failed; no weights are copied')


def strict_roundtrip(model, optimizer, path, variant, torch, np, context):
    """Serialize our own trusted endpoint; strict same-variant network roundtrip.

    Full pickle is used only inside the approved isolated native runtime. Optimizer
    payload is preserved, never restored. No historical checkpoint is loaded.
    """
    before = snapshot(parameter_copies(model, np), np)
    state = model.state_dict()
    tag = state.get('shader_network._a_variant_signature')
    require(variant in ARMS and model.material_variant == variant and tag is not None,
            'Missing/foreign native checkpoint identity')
    catalog = {n: (tuple(v.shape), str(v.dtype)) for n, v in state.items()}
    buffer_hashes = {n: __import__('hashlib').sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                     for n, v in state.items()}
    with Path(path).open('xb') as stream:
        torch.save(dict(protocol=PROTOCOL, material_variant=variant, step=RETURNS,
                        evaluation_step=RETURNS-1, context=context,
                        network_state_dict=state, optimizer_state_dict=optimizer.state_dict(),
                        best_para=None, selection='fixed endpoint; no best-metric selection'), stream)
    sha = digest(path)
    loaded = torch.load(str(path), map_location='cpu')
    require(digest(path) == sha and loaded['protocol'] == PROTOCOL and loaded['material_variant'] == variant and
            loaded['step'] == RETURNS and loaded['context'] == context, 'Endpoint checkpoint identity changed')
    saved = loaded['network_state_dict']
    require({n: (tuple(v.shape), str(v.dtype)) for n, v in saved.items()} == catalog,
            'Endpoint checkpoint keys/shapes/dtypes differ')
    require(all(__import__('hashlib').sha256(saved[n].numpy().tobytes()).hexdigest() == h
                for n, h in buffer_hashes.items()), 'Endpoint tensor content changed in serialization')
    result = model.load_state_dict(saved, strict=True)  # Native variant gate remains authoritative.
    require(not result.missing_keys and not result.unexpected_keys and
            snapshot(parameter_copies(model, np), np) == before, 'Strict endpoint roundtrip failed')
    return dict(sha256=sha, bytes=Path(path).stat().st_size, step=RETURNS, strict=True,
                optimizer_restored=False, tensor_hashes=buffer_hashes)


class EndpointTrace(TwoUpdateTrace):
    """Same inspected statement boundaries/owned cleanup, fixed endpoint stop."""
    def _trace(self, frame, event, argument):
        if frame.f_code is not self.code:
            return None
        if event == 'call':
            require(frame.f_locals.get('self') is self.trainer and self.frame is None, 'Foreign/reentrant run')
            self.frame = frame
        if event != 'line':
            return self._trace
        values = frame.f_locals
        self.iterator = values.get('train_iter', self.iterator)
        self.progress = values.get('pbar', self.progress)
        boundary = self.points.get(frame.f_lineno)
        if boundary is None:
            return self._trace
        if boundary in ('iteration', 'before_forward', 'before_backward', 'before_clip', 'before_optimizer'):
            require(type(values.get('step')) is int and values['step'] == self.completed < RETURNS,
                    'Unexpected step or attempt beyond E.13 endpoint')
        if boundary == 'after_optimizer':
            require(values['step'] == self.completed < RETURNS, 'Unexpected optimizer completion')
            self.completed += 1
            self.observer.after_optimizer(values, self.completed)
            if self.completed == RETURNS:
                raise _TwoUpdatesComplete(self)
        else:
            self.observer.event(boundary, values)
        return self._trace

    def run(self):
        previous = sys.gettrace()
        require(sys.implementation.name == 'cpython' and previous is None and not self.started,
                'Untraced fresh CPython required')
        self.started = True
        stopped = False
        try:
            sys.settrace(self._trace)
            require(sys.gettrace() == self._trace, 'Endpoint observer unavailable')
            try:
                self.trainer.run()
            except _TwoUpdatesComplete as exc:
                require(exc.owner is self and self.completed == RETURNS, 'Foreign/premature endpoint stop')
                stopped = True
        finally:
            sys.settrace(previous)
            for name, target, method in (
                    ('train_iterator', self.iterator, '_shutdown_workers'),
                    ('progress', self.progress, 'close'),
                    ('writer', getattr(getattr(self.trainer, 'logger', None), 'writer', None), 'close')):
                fn = getattr(target, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        self.cleanup_errors.append(dict(resource=name, error=str(exc)))
            self.frame = self.iterator = self.progress = None
            self.observer.cleanup(self.completed, self.cleanup_errors)
        require(stopped and self.completed == RETURNS and not self.cleanup_errors,
                'Endpoint stop/owned resource cleanup incomplete')


class RecoveryProbe(CorrectedProbe):
    def initialize(self):
        super().initialize()
        t = self.trainer
        require(t.cfg['total_step'] == RETURNS and not t.network.cfg['reg_mat'] and
                not t.network.cfg['reg_diffuse_light'] and
                t.network.shader_network.cfg['illumination_mode'] == 'known_env_v1', 'E.13 settings differ')
        packet = dict(parameters=self.report['initial_parameters'], splits=self.report['splits'])
        expected = self.report.get('expected_initialization')
        if expected is not None:
            initialization_gate(packet, expected)
        self.report['initialization_parity'] = 'first-arm anchor' if expected is None else 'exact'
        write_json(self.trainer.e13_run.output('initialization.json'), packet)
        self.report['scheduler'] = dict(vars(t.lr_manager))
        self.report['effective_losses'] = [dict(name=n, configuration=v.cfg) for n, v in zip(t.cfg['loss'], t.train_losses)]
        self.report['trajectory'] = []
        self.report['milestones'] = {}
        self.persist()

    def mark(self, phase):
        self.phase = self.report['phase'] = phase
        # Persist first-two full diagnostics; later rows flush on every actual return.
        if len(self.report.get('trajectory', [])) < 2:
            self.persist()

    def before_forward(self, values):
        if values['step'] < 2:
            return super().before_forward(values)
        t, c = self.trainer, self.current
        require(c is not None and values['train_data'] == {'step': c['step']} and
                t.network.training and self.torch.is_grad_enabled(), 'Invalid native training dispatch')
        lr = values['lr']  # Never call the scheduler from the observer.
        expected = 5e-4 * (c['step']/1000) if c['step'] < 1000 else (
            5e-4 * ((self.np.cos(self.np.pi * ((c['step']-1000)/(100000-1000))) + 1)*.5*(1-.2) + .2))
        require(lr == expected and all(g['lr'] == lr for g in t.optimizer.param_groups), 'Native schedule differs')
        start, pool = t.network.train_batch_i, t.network.tbn
        require(0 <= start and start+512 <= pool, 'Invalid original ray-pool cursor')
        c.update(lr=lr, calls={}, light_paths={'inner': 0, 'outer': 0, 'human': 0},
                 sampling={'cursor_before': start, 'pool_size': pool, 'ray_count': 512})

    def before_clip(self, values):
        if values['step'] < 2:
            return super().before_clip(values)
        # Native derivative hooks already check each finite gradient. Also require
        # full active-group presence, inactive known-light heads, nonzero aggregate.
        t, torch = self.trainer, self.torch
        active = []; norms = {}
        for group in self.groups:
            ps = [p for n, p in t.network.named_parameters() if n.startswith('shader_network.'+group+'.')]
            present = [p.grad for p in ps if p.grad is not None]
            inactive = group in ('inner_light', 'outer_light', 'human_light')
            require((not present) if inactive else len(present) == len(ps) > 0,
                    'Unexpected active/missing gradient group: '+group)
            if present:
                flat = torch.cat([g.detach().reshape(-1) for g in present])
                require(bool(torch.isfinite(flat).all()), 'Nonfinite gradients: '+group)
                norm = float(torch.norm(flat).cpu().item())
                norms[group] = norm
                active.append(norm)
        require(sum(active) > 0, 'All active gradients vanished')
        self.current['group_gradient_l2_pre_clip'] = norms

    def before_optimizer(self, values):
        if values['step'] < 2:
            return super().before_optimizer(values)
        require(values['max_grad_norm'] == .7, 'Clipping policy changed')
        norm = float(tensor_copy(values['total_norm'], self.np).item())
        require(math.isfinite(norm) and norm > 0, 'Invalid native gradient norm')
        self.current['native_clip_returned_norm'] = norm

    def after_optimizer(self, values, completed):
        self.report['optimizer_calls_completed'] = completed
        if completed <= 2:
            super().after_optimizer(values, completed)
            c = self.report['updates'][-1]
        else:
            c = self.current
            require(c['step']+1 == completed and c.get('forward_path_verified') and
                    'native_clip_returned_norm' in c, 'Incomplete forward/backward evidence')
            require(all(int(s['step']) == completed for s in self.trainer.optimizer.state.values()),
                    'Adam per-parameter step differs')
            start, pool = c['sampling']['cursor_before'], c['sampling']['pool_size']
            expected_cursor = 0 if start+1024 >= pool else start+512
            require(self.trainer.network.train_batch_i == expected_cursor, 'Original shuffle/cursor behavior differs')
        row = dict(update=completed, lr=c['lr'], loss=c['aggregate_loss'],
                   gradient_l2=c['native_clip_returned_norm'], elapsed_seconds=time.monotonic()-c.get('started_monotonic', time.monotonic()))
        if 'elapsed_seconds' in c:
            row['elapsed_seconds'] = c['elapsed_seconds']
        if 'group_gradient_l2_pre_clip' in c:
            row['groups'] = c['group_gradient_l2_pre_clip']
        row['cursor_after'] = self.trainer.network.train_batch_i
        self.report['trajectory'].append(row)
        if completed in MILESTONES:
            self.report['milestones'][str(completed)] = dict(
                parameters=snapshot(parameter_copies(self.trainer.network, self.np), self.np),
                optimizer=optimizer_record(self.trainer.optimizer, self.trainer.network, self.np), rng=self.rng())
        self.current = self.before_parameters = None
        self.persist()


def material_errors(pred, truth, np):
    """Nominal Q and actual-frame validity are deliberately distinct outcomes."""
    mx, my = pred['mx'].reshape(-1), pred['my'].reshape(-1)
    r = pred['rotation']; t, b = pred['T'], pred['B']
    theta = .5*np.arctan2(r[:, 1], r[:, 0])
    u = np.cos(theta)[:, None]*t + np.sin(theta)[:, None]*b
    v = -np.sin(theta)[:, None]*t + np.cos(theta)[:, None]*b
    Q = mx[:, None, None]**2*u[:, :, None]*u[:, None, :] + my[:, None, None]**2*v[:, :, None]*v[:, None, :]
    frame = np.stack([truth['T'], truth['B']], axis=-1)
    local = np.einsum('nki,nkl,nlj->nij', frame, Q, frame)
    widths = np.stack([np.maximum(mx, my), np.minimum(mx, my)], -1)
    major = np.where((mx >= my)[:, None], u, v)
    gtmajor = np.cos(truth['theta_major'])[:, None]*truth['T'] + np.sin(truth['theta_major'])[:, None]*truth['B']
    angle = np.degrees(np.arccos(np.clip(abs((major*gtmajor).sum(-1)), 0, 1)))
    actual = np.stack([pred['actual_T'], pred['actual_B'], truth['normals']], axis=-1)
    gram = np.einsum('nki,nkj->nij', actual, actual)
    invalid_frame = ((np.max(abs(gram-np.eye(3)), axis=(1, 2)) > FRAME_TOL) |
                     (abs(np.linalg.det(actual)-1) > FRAME_TOL))
    invalid_material = (~np.isfinite(np.concatenate([mx[:, None], my[:, None], r, pred['color'], pred['metallic']], -1)).all(-1) |
                        (mx <= 0) | (my <= 0) | (mx > 1) | (my > 1) |
                        (abs(np.linalg.norm(r, axis=-1)-1) > FRAME_TOL) |
                        ((pred['color'] < 0) | (pred['color'] > 1)).any(-1) |
                        ((pred['metallic'] < 0) | (pred['metallic'] > 1)).any(-1))
    return dict(Q_world_absolute=np.linalg.norm(Q-truth['Q_world'], axis=(1, 2)),
                Q_world_relative=np.linalg.norm(Q-truth['Q_world'], axis=(1, 2))/np.maximum(np.linalg.norm(truth['Q_world'], axis=(1, 2)), 1e-15),
                Q_local_absolute=np.linalg.norm(local-truth['Q_local'], axis=(1, 2)),
                Q_local_relative=np.linalg.norm(local-truth['Q_local'], axis=(1, 2))/np.maximum(np.linalg.norm(truth['Q_local'], axis=(1, 2)), 1e-15),
                major_width_absolute=abs(widths[:, 0]-truth['widths_major_minor'][:, 0]),
                minor_width_absolute=abs(widths[:, 1]-truth['widths_major_minor'][:, 1]),
                orientation_degrees=angle, orientation_eligible=truth['orientation_eligible'] & truth['orientation_defined'],
                orientation_prediction_defined=abs(mx-my) > 1e-6,
                albedo_max_absolute=abs(pred['color']-truth['color']).max(-1),
                metallic_absolute=abs(pred['metallic']-truth['metallic']).reshape(-1),
                invalid_frame=invalid_frame, invalid_material=invalid_material,
                Q_world=Q, Q_local=local)


def summarize_material(rows, np):
    require(rows, 'Empty material aggregate')
    joined = {k: np.concatenate([a[k] for a in rows]) for k in rows[0] if k not in ('Q_world', 'Q_local')}
    n = len(joined['invalid_frame']); valid = ~(joined['invalid_frame'] | joined['invalid_material'])
    result = dict(observations=n, invalid_frames=int(joined['invalid_frame'].sum()),
                  invalid_materials=int(joined['invalid_material'].sum()), valid_observations=int(valid.sum()))
    for key, a in joined.items():
        if a.dtype == np.bool_:
            continue
        mask = (joined['orientation_eligible'] & joined['orientation_prediction_defined']) if key == 'orientation_degrees' else np.ones(n, bool)
        x = a[mask]
        result[key] = dict(count=len(x), mean=float(x.mean()) if len(x) else None,
                           p95=float(np.quantile(x, .95)) if len(x) else None,
                           maximum=float(x.max()) if len(x) else None)
    thresholds = dict(Q_world_relative=.05, Q_local_relative=.05, orientation_degrees=5.,
                      albedo_max_absolute=.02, metallic_absolute=.03)
    result['descriptive_pass_fractions'] = {}
    for key, limit in thresholds.items():
        eligible = joined['orientation_eligible'] if key == 'orientation_degrees' else np.ones(n, bool)
        defined = joined['orientation_prediction_defined'] if key == 'orientation_degrees' else np.ones(n, bool)
        denom = int(eligible.sum())
        result['descriptive_pass_fractions'][key] = dict(
            threshold=limit, numerator=int(((joined[key] <= limit) & valid & eligible & defined).sum()),
            denominator=denom, fraction=float(((joined[key] <= limit) & valid & eligible & defined).sum()/denom) if denom else None)
    result['gt_orientation_eligible'] = int(joined['orientation_eligible'].sum())
    result['predicted_orientation_undefined_on_gt_eligible'] = int((joined['orientation_eligible'] & ~joined['orientation_prediction_defined']).sum())
    return result
