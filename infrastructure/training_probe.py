"""Read-only training diagnostics; native tensors are copied, never modified.

NumPy/Torch are supplied after runtime isolation. Diagnostics reduce detached
CPU copies; they never feed back into losses, clipping, optimizer or RNG state.
"""

import hashlib
import time

from infrastructure.bell_inference import ordered_split, rng_identity
from infrastructure.signatures import require


def tensor_copy(value, np, context='unspecified'):
    array = value.detach().cpu().numpy().copy()
    require(np.isfinite(array).all(), 'Nonfinite training tensor: ' + context)
    return array


def array_record(array, np):
    values = array.astype(np.float64)
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'numel': int(array.size),
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            'l2': float(np.linalg.norm(values.reshape(-1))),
            'mean': float(values.mean()) if values.size else 0.0,
            'max_abs': float(np.abs(values).max()) if values.size else 0.0}


def parameter_copies(network, np):
    return {name: tensor_copy(value, np, 'parameter:' + name) for name, value in network.named_parameters()}


def snapshot(values, np):
    return {name: array_record(value, np) for name, value in values.items()}


def parameter_changes(before, after, np):
    require(before.keys() == after.keys(), 'Parameter keys changed during update')
    result = {}
    for name in before:
        require(before[name].shape == after[name].shape and before[name].dtype == after[name].dtype,
                'Parameter shape/dtype changed')
        difference = after[name].astype(np.float64) - before[name].astype(np.float64)
        result[name] = {'changed_elements': int((difference != 0).sum()),
                        'l2': float(np.linalg.norm(difference.reshape(-1))),
                        'max_abs': float(np.abs(difference).max()) if difference.size else 0.0}
    return result


def optimizer_record(optimizer, network, np):
    names = {id(value): name for name, value in network.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        groups.append({**{k: v for k, v in group.items() if k != 'params'},
                       'parameter_names': [names[id(p)] for p in group['params']]})
    state = {}
    for parameter, values in optimizer.state.items():
        require(id(parameter) in names, 'Optimizer owns an unknown parameter')
        entry = {}
        for key, value in values.items():
            if key == 'step':
                number = value.item() if hasattr(value, 'item') else value
                require(not isinstance(number, bool) and int(number) == number, 'Invalid Adam step metadata')
                entry['step'] = int(number)
            else:
                require(key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'), 'Unexpected Adam state field')
                entry[key] = array_record(tensor_copy(value, np, 'Adam:' + names[id(parameter)] + ':' + key), np)
        state[names[id(parameter)]] = entry
    return {'class': type(optimizer).__module__ + '.' + type(optimizer).__name__, 'groups': groups, 'state': state}


class TrainingProbe:
    """Events come from the existing loop's exact statement boundaries."""

    def __init__(self, trainer, torch, np, report, persist, rng=None, memory=None):
        self.trainer, self.torch, self.np = trainer, torch, np
        self.report, self.persist = report, persist
        self.rng = rng if rng is not None else lambda: rng_identity(torch, np)
        self.memory = memory if memory is not None else lambda: {
            'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        self.current = None
        self.before_parameters = None
        self.previous_state = None
        self.phase = 'constructing'
        self.report['updates'] = []

    def mark(self, phase):
        self.phase = self.report['phase'] = phase
        self.persist()

    def event(self, boundary, values):
        self.mark(boundary)
        if boundary == 'initialized':
            self.initialize()
        elif boundary == 'fresh_restore_gate':
            require(not any(self.trainer.experiment.run.checkpoints.iterdir()), 'Fresh run contains checkpoint artifacts')
        elif boundary == 'iteration':
            require(values['start_step'] == 0 and values['best_para'] == 0, 'Unexpected resume/progress state')
            self.current = {'step': values['step'], 'rng_before_batch': self.rng(), 'started_monotonic': time.monotonic()}
        elif boundary == 'before_forward':
            self.before_forward(values)
        elif boundary == 'before_backward':
            self.before_backward(values)
        elif boundary == 'before_clip':
            self.before_clip(values)
        elif boundary == 'before_optimizer':
            self.before_optimizer(values)

    def initialize(self):
        t, torch = self.trainer, self.torch
        require(type(t.optimizer) is torch.optim.Adam, 'Expected the actual Adam class, not an optimizer wrapper')
        for key, expected in {'random_seed': 6033, 'total_step': 300000, 'val_interval': 1000,
                              'save_interval': 1000, 'train_log_step': 20, 'worker_num': 8}.items():
            require(type(t.cfg[key]) is int and t.cfg[key] == expected, 'Unadmitted scientific setting: ' + key)
        require(t.cfg['lr_type'] == 'warm_up_cos' and t.cfg['lr_cfg'] == {} and t.cfg['optimizer_type'] == 'adam',
                'Unadmitted optimizer/scheduler configuration')
        require(t.train_network is t.network and not t.cfg['multi_gpus'], 'Expected unchanged single-GPU geometry')
        cfg = t.network.cfg
        require(cfg['train_ray_num'] == 512 and cfg['n_samples'] == 64 and cfg['n_importance'] == 64 and
                cfg['n_bg_samples'] == 32 and cfg['up_sample_steps'] == 4 and not cfg['val_geometry'],
                'Unexpected geometry sampling or extraction settings')
        require(t.network.isolated_geometry is True, 'Explicit approved A split shim required')
        initial = parameter_copies(t.network, self.np)
        require(initial, 'No initialized model parameters')
        self.report.update(initial_parameters=snapshot(initial, self.np), splits=ordered_split(t.network),
            effective_trainer=t.cfg, effective_renderer=cfg, effective_shader=t.network.color_network.cfg,
            effective_losses=[{'registered_name': name,
                               'class': type(loss).__module__ + '.' + type(loss).__name__,
                               'configuration': getattr(loss, 'cfg', None)}
                              for name, loss in zip(t.cfg['loss'], t.train_losses)],
            initial_buffers={name: array_record(tensor_copy(value, self.np), self.np) for name, value in t.network.named_buffers()},
            scheduler={'class': type(t.lr_manager).__name__, 'attributes': dict(vars(t.lr_manager)),
                       'hard_coded_cosine_endpoint': 100000},
            loader={'batch_size': t.train_set.batch_size, 'num_workers': t.train_set.num_workers,
                    'drop_last': t.train_set.drop_last, 'sampler': type(t.train_set.sampler).__name__,
                    'dataset_size': len(t.train_set.dataset), 'validation_loader_count': len(t.val_set_list),
                    'collate': t.train_set.collate_fn.__name__})
        self.previous_state = optimizer_record(t.optimizer, t.network, self.np)
        require(not self.previous_state['state'] and len(self.previous_state['groups']) == 1, 'Optimizer is not fresh')
        group = self.previous_state['groups'][0]
        require(group['lr'] == 1e-3 and group['eps'] == 1e-4 and group['amsgrad'] is True and
                tuple(group['betas']) == (0.9, 0.999) and group['weight_decay'] == 0,
                'Unexpected native Adam defaults')
        require(self.report['loader'] == {'batch_size': 1, 'num_workers': 8, 'drop_last': False,
                'sampler': 'RandomSampler', 'dataset_size': 99999999, 'validation_loader_count': 1,
                'collate': 'dummy_collate_fn'}, 'Unexpected DataLoader behavior')
        self.report['initial_optimizer'] = self.previous_state
        self.persist()

    def before_forward(self, values):
        t, c, np = self.trainer, self.current, self.np
        require(c is not None and c['step'] == values['step'] and values['train_data'] == {'step': c['step']},
                'Unexpected dummy training dispatch')
        expected_lr = (0.0, 5e-4 * (1 / 5000))[c['step']]  # Preserve Python's original float arithmetic.
        require(values['lr'] == expected_lr and all(g['lr'] == expected_lr for g in t.optimizer.param_groups),
                'Legacy warm-up schedule changed')
        require(t.network.training and self.torch.is_grad_enabled(), 'Training gradients are not enabled')
        c.update(lr=float(values['lr']), rng_before_forward=self.rng(), memory_before=self.memory())
        start, count = t.network.train_batch_i, t.network.cfg['train_ray_num']
        require(start == c['step'] * count and start + count < t.network.tbn, 'Unexpected ray cursor/reshuffle')
        c['sampling'] = {'cursor_before': start, 'ray_count': count, 'pool_size': t.network.tbn,
                         'selected': {k: array_record(tensor_copy(v[start:start+count], np), np)
                                      for k, v in t.network.train_batch.items()}}
        self.before_parameters = parameter_copies(t.network, np)
        c['parameters_before'] = snapshot(self.before_parameters, np)

    def before_backward(self, values):
        require(self.current is not None, 'Backward without a selected batch')
        torch, np = self.torch, self.np
        losses = {}
        for key, value in values['log_info'].items():
            require(torch.is_tensor(value), 'Unexpected non-tensor loss/recorder output')
            tensor_copy(value, np, 'loss:' + key)  # Full finite check, not only a potentially cancelling mean.
            losses[key] = float(torch.mean(value.detach()).cpu().item())
        require(losses and any(k.startswith('loss') for k in losses), 'No actual training losses')
        aggregate = tensor_copy(values['loss'], np)
        require(aggregate.size == 1 and values['loss'].requires_grad, 'Invalid differentiable aggregate loss')
        self.current.update(losses=losses, aggregate_loss=float(aggregate.item()))
        self.persist()

    def gradients(self):
        result = {}; squared = 0.0
        for name, p in self.trainer.network.named_parameters():
            if p.grad is None:
                result[name] = None
            else:
                record = array_record(tensor_copy(p.grad, self.np, 'gradient:' + name), self.np)
                result[name] = record; squared += record['l2'] ** 2
        return result, squared ** 0.5

    def before_clip(self, values):
        gradients, norm = self.gradients()
        require(norm > 0, 'No nonzero finite gradients from backward')
        self.current.update(gradients_before_clip=gradients, gradient_l2_before_clip_cpu64=norm)
        self.persist()

    def before_optimizer(self, values):
        require(values['max_grad_norm'] == 0.7, 'Legacy clipping threshold changed')
        source_norm = tensor_copy(values['total_norm'], self.np)
        gradients, norm = self.gradients()
        require(norm <= 0.70001, 'Post-clip gradient norm exceeds the preserved bound')
        self.current.update(native_clip_returned_norm=float(source_norm.item()), clip_max_norm=0.7,
                            gradients_after_clip=gradients, gradient_l2_after_clip_cpu64=norm)
        self.persist()

    def after_optimizer(self, values, completed):
        self.mark('after_optimizer')
        c, t, np = self.current, self.trainer, self.np
        self.report['optimizer_calls_completed'] = completed
        require(c is not None and c['step'] + 1 == completed and 'gradients_after_clip' in c,
                'Optimizer completion lacks matching forward/backward/clipping evidence')
        after = parameter_copies(t.network, np)
        changes = parameter_changes(self.before_parameters, after, np)
        changed = sum(v['changed_elements'] for v in changes.values())
        require((completed == 1 and changed == 0) or (completed == 2 and changed > 0),
                'Unexpected parameter-change evidence for the preserved zero-LR first step')
        state = optimizer_record(t.optimizer, t.network, np)
        active = {name for name, g in c['gradients_after_clip'].items() if g is not None}
        previous = self.previous_state['state']
        require(set(state['state']) == (set(previous) | active) and all(
                value['step'] == previous.get(name, {}).get('step', 0) + int(name in active)
                for name, value in state['state'].items()),
                'Adam per-parameter step accounting changed')
        require(all(set(v) == {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'} for v in state['state'].values()),
                'Missing Adam/AMSGrad state metadata')
        c.update(update_index=completed, parameters_after=snapshot(after, np), parameter_changes=changes,
                 changed_elements=changed, optimizer=state, rng_after=self.rng(), memory_after=self.memory(),
                 elapsed_seconds=time.monotonic()-c.pop('started_monotonic'),
                 cursor_after=t.network.train_batch_i)
        require(c['cursor_after'] == completed * t.network.cfg['train_ray_num'], 'Unexpected post-update ray cursor')
        self.report['updates'].append(c)
        self.previous_state = state
        self.current = self.before_parameters = None
        self.persist()

    def cleanup(self, completed, errors):
        self.report.update(optimizer_calls_completed=completed, cleanup_errors=errors)
        if self.current is not None:
            partial = dict(self.current); partial.pop('started_monotonic', None)
            self.report['partial_update'] = partial
        self.persist()
