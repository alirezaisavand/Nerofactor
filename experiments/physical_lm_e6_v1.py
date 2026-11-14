"""E6 physically scaled LM policy and work matching; NumPy only, no oracle."""
import math
import numpy as np
from infrastructure.signatures import require

SCALES = np.array([.96, .96, np.pi, 1., 1., 1., 1.])
ACCEPTED_TARGET = 20
TRIALS = 4
LM_FIT_CAP = 503
ADAM_MAX = 250
POLICY = dict(initial_damping=.01, minimum_damping=1e-8, maximum_damping=1e8,
              accepted_ratio=.1, good_ratio=.75, poor_ratio=.25,
              rejected_multiplier=10., good_divisor=3., poor_multiplier=2.,
              scaled_step_radius=.1, boundary_fraction=.99,
              reduction_epsilon_multiplier=16.)
LOW = np.array([float(np.float32(.04))]*2+[-np.inf]+[0.]*4)
HIGH = np.array([1., 1., np.inf, 1., 1., 1., 1.])


def valid(p):
    p = np.asarray(p)
    require(p.shape == (7,) and p.dtype == np.float32 and np.isfinite(p).all() and
            np.all(p >= LOW) and np.all(p <= HIGH), 'Finite bounded original float32 material required')
    return p


def verify_start(pa, pb, ya, yb, la, lb):
    valid(pa); valid(pb); a, b = np.asarray(ya), np.asarray(yb)
    require(np.array_equal(pa, pb) and a.shape == b.shape == (24, 3, 3) and
            a.dtype == b.dtype == np.float32 and np.isfinite(a).all() and np.array_equal(a, b) and
            math.isfinite(la) and la >= 0 and la == lb, 'Paired physical states/radiance/objectives differ')
    return True


def proposal(p, j, residual, damping):
    """Half-MSE normal system; predicted reduction is for full MSE/actual f32 step."""
    p = valid(p); j = np.asarray(j, np.float64); r = np.asarray(residual, np.float64)
    require(j.shape == (72, 7) and r.shape == (72,) and np.isfinite(j).all() and np.isfinite(r).all() and
            POLICY['minimum_damping'] <= damping <= POLICY['maximum_damping'], 'Invalid physical LM system')
    h, g = j.T@j, j.T@r
    a = h+damping*np.eye(7)
    lower = np.linalg.cholesky(a)
    raw = np.linalg.solve(lower.T, np.linalg.solve(lower, -g))
    error = float(np.linalg.norm(a@raw+g))
    require(np.isfinite(raw).all() and error <= 1e-10*max(np.linalg.norm(a)*np.linalg.norm(raw)+np.linalg.norm(g), 1e-30),
            'Invalid Cholesky solve; no alternate solver or jitter retry')
    delta = raw*min(1., POLICY['scaled_step_radius']/max(float(np.linalg.norm(raw)), 1e-300))
    dp = SCALES*delta; alpha = 1.
    for i in range(7):
        if dp[i] < 0 and np.isfinite(LOW[i]): bound = (LOW[i]-float(p[i]))/dp[i]
        elif dp[i] > 0 and np.isfinite(HIGH[i]): bound = (HIGH[i]-float(p[i]))/dp[i]
        else: continue
        if bound < 1: alpha = min(alpha, POLICY['boundary_fraction']*max(0., bound))
    candidate = (p.astype(np.float64)+alpha*dp).astype(np.float32); valid(candidate)
    actual = (candidate.astype(np.float64)-p.astype(np.float64))/SCALES
    predicted = float(-2*g@actual-actual@h@actual)
    branch = lambda t: math.floor((float(t)+math.pi/2)/math.pi)
    return candidate, dict(raw_scaled_delta=raw.tolist(), represented_scaled_delta=actual.tolist(),
        raw_step_L2=float(np.linalg.norm(raw)), represented_step_L2=float(np.linalg.norm(actual)), boundary_fraction=alpha,
        predicted_reduction=predicted, solve_residual_L2=error, scaled_gradient=(2*g).tolist(),
        decoder_branch_crossed=branch(p[2]) != branch(candidate[2]))


def acceptance(old, new, predicted):
    require(math.isfinite(old) and math.isfinite(new) and old >= 0 and new >= 0, 'Finite nonnegative fitting losses required')
    floor = POLICY['reduction_epsilon_multiplier']*np.finfo(np.float32).eps*max(old, new, 1e-12)
    ratio = (old-new)/predicted if math.isfinite(predicted) and predicted > 0 else None
    return bool(ratio is not None and math.isfinite(ratio) and ratio > POLICY['accepted_ratio'] and old-new > floor), ratio, float(floor)


def next_damping(value, accepted, ratio):
    if not accepted: value *= POLICY['rejected_multiplier']
    elif ratio > POLICY['good_ratio']: value /= POLICY['good_divisor']
    elif ratio < POLICY['poor_ratio']: value *= POLICY['poor_multiplier']
    return float(min(POLICY['maximum_damping'], max(POLICY['minimum_damping'], value)))


def adam_cost(updates):
    require(type(updates) is int and 1 <= updates <= ADAM_MAX, 'Invalid direct Adam update count')
    return 4*updates+4+7*(1+int(updates > ACCEPTED_TARGET))


def matched_adam_updates(lm_cost):
    choices = [n for n in range(1, ADAM_MAX+1) if adam_cost(n) <= lm_cost]
    require(bool(choices), 'LM work too small for a valid Adam match')
    return choices[-1]


class Work:
    """All renderer calls/reverse sweeps, including initial/terminal/evaluation work."""
    def __init__(self, method, fd_admission=False):
        require(method in ('LM', 'Adam'), 'Unknown optimizer'); self.method = method
        require(type(fd_admission) is bool and (method == 'LM' or not fd_admission), 'Invalid derivative admission mode')
        self.fd_admission = fd_admission
        self.attempts = dict(fit_forward=0, reverse=0, external=0, updates=0)
        self.done = dict(self.attempts)
        self.events = dict(jacobians_attempted=0, jacobians_completed=0, trials=0,
                           solves_attempted=0, solves_completed=0, accepted_steps=0, rejected_trials=0)
    def key(self, kind, phase):
        if kind == 'forward':
            require(phase in ('initial', 'jacobian', 'trial', 'update', 'terminal', 'external', 'initial_fd'), 'Unknown forward phase')
            return 'external' if phase == 'external' else 'fit_forward'
        if kind == 'reverse':
            require(phase in ('JVP', 'Adam', 'terminal'), 'Unknown reverse phase'); return 'reverse'
        require(kind == 'update' and phase == 'Adam' and self.method == 'Adam', 'Unknown work event'); return 'updates'
    def attempt(self, kind, phase):
        key = self.key(kind, phase)
        limits = dict(fit_forward=222+4*int(self.fd_admission) if self.method == 'LM' else ADAM_MAX+1,
                      reverse=281 if self.method == 'LM' else ADAM_MAX+1,
                      external=1 if self.method == 'LM' else 2, updates=0 if self.method == 'LM' else ADAM_MAX)
        require(self.attempts[key] < limits[key], 'E6 work cap: '+key)
        if key in ('fit_forward', 'reverse'):
            require(self.remaining_fit() > 0, 'E6 fitting work exhausted')
        self.attempts[key] += 1
    def complete(self, kind, phase):
        key = self.key(kind, phase); require(self.done[key] < self.attempts[key], 'Unattempted work completion'); self.done[key] += 1
    def remaining_fit(self):
        cap = LM_FIT_CAP+4*int(self.fd_admission) if self.method == 'LM' else 2*ADAM_MAX+2
        return cap-self.attempts['fit_forward']-self.attempts['reverse']
    def cost(self): return 2*(self.done['fit_forward']+self.done['reverse'])+7*self.done['external']
    def verify(self):
        require(self.attempts == self.done and self.remaining_fit() >= 0 and
            self.events['jacobians_attempted'] == self.events['jacobians_completed'] and
            self.events['solves_attempted'] == self.events['solves_completed'], 'Incomplete native optimization work')


def run_lm(core, record, check=lambda reserve=0: None):
    """Dependency-injected fixed policy. Core exposes p/current/forward/jacobian/work.

    No material truth, component targets, external errors or recovery criteria
    enter this optimizer. Four rejected trials stop the arm without a new batch.
    """
    damping = POLICY['initial_damping']; accepted = 0; rows = []
    for outer in range(ACCEPTED_TARGET):
        check(10)
        if core.work.remaining_fit() < 24: termination = 'fitting_budget'; break
        core.work.events['jacobians_attempted'] += 1
        j = core.jacobian(); core.work.events['jacobians_completed'] += 1
        residual = np.asarray(core.current['residual'], np.float64)
        _, singular, vt = np.linalg.svd(j, full_matrices=False)
        row = dict(outer=outer+1, physical_before=core.p.tolist(), loss_before=core.current['loss'],
                   singular_values=singular.tolist(), scaled_gradient=(2*j.T@residual).tolist(),
                   weakest_scaled_direction=vt[-1].tolist(), trials=[])
        changed = False
        for trial in range(TRIALS):
            check(10); core.work.events['trials'] += 1; core.work.events['solves_attempted'] += 1
            candidate, detail = proposal(core.p, j, residual, damping); core.work.events['solves_completed'] += 1
            before = damping; predicted = detail['predicted_reduction']; measured = False
            if predicted > 0 and not np.array_equal(candidate, core.p):
                if core.work.remaining_fit() < 3: termination = 'fitting_budget'; break
                value = core.forward(candidate); measured = True
                take, ratio, floor = acceptance(core.current['loss'], value['loss'], predicted)
            else: value = None; take = False; ratio = None; floor = None
            damping = next_damping(damping, take, ratio)
            detail.update(trial=trial+1, physical=candidate.tolist(), queried=measured, accepted=take,
                loss=None if value is None else value['loss'], actual_reduction=None if value is None else core.current['loss']-value['loss'],
                ratio=ratio, reduction_floor=floor, damping_before=before, damping_after=damping)
            step = np.asarray(detail['represented_scaled_delta']); length = np.linalg.norm(step)
            detail['weak_direction_step_fraction'] = float(abs(vt[-1]@step)/length) if length > 0 else None
            row['trials'].append(detail)
            if take:
                accepted += 1; core.work.events['accepted_steps'] += 1
                core.p = candidate; core.current = value; changed = True; break
            core.work.events['rejected_trials'] += 1
        row.update(accepted_updates=accepted, physical_after=core.p.tolist(), loss_after=core.current['loss'])
        rows.append(row); record(row, j)
        if not changed: termination = 'four_trial_limit' if len(row['trials']) == TRIALS else 'fitting_budget'; break
    else: termination = 'accepted_update_target'
    return dict(status='complete' if accepted == ACCEPTED_TARGET else 'censored_algorithm', termination=termination,
                accepted_updates=accepted, damping=damping, history=rows, matched_update_available=accepted == ACCEPTED_TARGET)
