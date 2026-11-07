"""External CPython line observer around the unchanged Trainer.run method.

No optimizer/loss/loader replacement, source rewriting, trace into native ops or
scientific RNG draws. The private stop is raised at the first line after Adam
returns the second time. This implementation is for review, not GPU approval.
"""

import ast
from pathlib import Path
import sys

from infrastructure.signatures import require


class _TwoUpdatesComplete(BaseException):
    def __init__(self, owner):
        self.owner = owner


def run_boundaries(path):
    """Locate existing statements; never compile/modify the native trainer."""
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in run.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    def call_name(node):
        if isinstance(node, ast.Call):
            node = node.func
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return call_name(node.value) + '.' + node.attr
        return ''
    def find(body, target):
        matches = [n for n in body if call_name(getattr(n, 'value', None)) == target]
        require(len(matches) == 1, 'Unrecognized Trainer.run statement: ' + target)
        return matches[0]
    optimizer = find(loop.body, 'self.optimizer.step')
    following = loop.body[loop.body.index(optimizer) + 1]
    points = {'dataset_initialization': find(run.body, 'self._init_dataset').lineno,
              'network_initialization': find(run.body, 'self._init_network').lineno,
              'initialized': find(run.body, 'self._init_logger').lineno,
              'fresh_restore_gate': find(run.body, 'self._load_model').lineno,
              'iteration': loop.body[0].lineno,
              'before_forward': find(loop.body, 'self.train_network').lineno,
              'before_backward': find(loop.body, 'loss.backward').lineno,
              'before_clip': find(loop.body, 'torch.nn.utils.clip_grad_norm_').lineno,
              'before_optimizer': optimizer.lineno, 'after_optimizer': following.lineno}
    require(len(set(points.values())) == len(points), 'Ambiguous line boundaries')
    return run.lineno, points


class TwoUpdateTrace:
    """Observe one owned Trainer.run frame; require fresh steps 0 and 1 only."""

    def __init__(self, trainer, observer, source_path):
        self.trainer, self.observer = trainer, observer
        self.code = trainer.run.__func__.__code__
        first, points = run_boundaries(source_path)
        require(Path(self.code.co_filename).resolve() == Path(source_path).resolve() and
                self.code.co_firstlineno == first, 'Trainer.run does not match inspected source')
        self.points = {line: name for name, line in points.items()}
        self.completed = 0
        self.iterator = self.progress = None
        self.cleanup_errors = []
        self.frame = None
        self.started = False

    def _trace(self, frame, event, argument):
        if frame.f_code is not self.code:
            return None
        if event == 'call':
            require(frame.f_locals.get('self') is self.trainer and self.frame is None, 'Unexpected/reentrant training frame')
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
            require(type(values.get('step')) is int and values['step'] == self.completed and self.completed < 2,
                    'Unexpected step or attempt beyond the two-update budget')
        if boundary == 'after_optimizer':
            require(self.completed < 2 and values['step'] == self.completed, 'Unexpected optimizer completion')
            self.completed += 1  # Actual Adam returned; count even if the subsequent diagnostic fails.
            self.observer.after_optimizer(values, self.completed)
            if self.completed == 2:
                raise _TwoUpdatesComplete(self)
        else:
            self.observer.event(boundary, values)
        return self._trace

    def run(self):
        previous = sys.gettrace()
        require(sys.implementation.name == 'cpython' and previous is None and not self.started,
                'Requires untraced CPython; do not override an existing debugger/coverage trace')
        self.started = True
        stopped = False
        try:
            sys.settrace(self._trace)
            require(sys.gettrace() == self._trace, 'Observer unavailable; training is not entered')
            try:
                self.trainer.run()
            except _TwoUpdatesComplete as exc:
                require(exc.owner is self and self.completed == 2, 'Foreign or premature bounded stop')
                stopped = True
        finally:
            sys.settrace(previous)
            # Owned DataLoader workers, progress bar and TensorBoard writer only.
            targets = [('train_iterator', self.iterator, '_shutdown_workers'),
                       ('progress', self.progress, 'close'),
                       ('writer', getattr(getattr(self.trainer, 'logger', None), 'writer', None), 'close')]
            for name, target, method in targets:
                function = getattr(target, method, None)
                if callable(function):
                    try:
                        function()
                    except Exception as exc:
                        self.cleanup_errors.append({'resource': name, 'error_type': type(exc).__name__, 'error': str(exc)})
            self.frame = self.iterator = self.progress = None
            self.observer.cleanup(self.completed, self.cleanup_errors)
        require(stopped and self.completed == 2, 'Trainer returned without the explicit two-update stop')
        require(not self.cleanup_errors, 'Owned runtime resource cleanup failed')
