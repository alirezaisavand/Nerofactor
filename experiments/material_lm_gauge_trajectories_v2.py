"""3C-9c: one 15-arm LM comparison, independent initialization gates A--E."""
import argparse
import copy
import ctypes
import datetime
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

import numpy as np
import torch

from experiments import material_lm_gauge_trajectories_v1 as v1
from experiments import material_gauge_roundoff_v1 as roundoff
from infrastructure.signatures import require, SignatureError
from infrastructure.stage_sources import baseline, PINS

lm, original, reference, coupling = v1.lm, v1.original, v1.reference, v1.coupling
ROOT = Path(__file__).resolve().parents[1]
BASE = 'c37337e195db57b9a331021d9a55147abf866bb1'
NAME = 'MATERIAL_LM_GAUGE_TRAJECTORIES_V2'
PROTOCOL = ROOT/('docs/protocols/'+NAME+'.md')
OUTPUT = ROOT/('docs/evidence/'+NAME+'.json')
CODE = ('experiments/material_lm_gauge_trajectories_v2.py', 'tests/scientific/test_material_lm_gauge_trajectories_v2.py')
TORCH_GIT = 'e4ee3be4063b7c430974252fdf7db42273388d86'
INPUTS = {
    'docs/evidence/MATERIAL_GAUGE_ROUNDOFF_V1.json':'4753e1e6d971b033d24d686e146ab0d22dfd4941eee3d03df9780dc0fce37c5b',
    'docs/evidence/MATERIAL_GAUGE_ROUNDOFF_V1_OFFLINE_SUMMARY.json':'510775832a45f9ec1956f2cc6ab5fc27ce0a1f9d5a88a9ff53e8944a4dbf934c',
    'docs/protocols/MATERIAL_GAUGE_ROUNDOFF_V1.freeze.json':'87788c2dbae00cce036080f44f6b66151e0ad4cf192e03e54c55c98f99a78527',
    'docs/MATERIAL_GAUGE_ROUNDOFF_V1_RESULTS.md':'ec0b709230dd4348ee8d0db832e5c03dd39c6af5649a68ac79ed8d8af216132f',
}
POLICY = dict(Q_atol=2e-7, Q_rtol=2e-6, frame_atol=2e-6,
    h_atol=2e-6, wi_atol=4e-6, vector_rtol=2e-5, angle_atol=2e-5,
    RGB_atol=2e-6, RGB_rtol=2e-5, old_scalar_atol=1e-12, old_scalar_rtol=5e-7,
    grazing_indicator=1e-5, arithmetic_u_power=-24, gamma_k=75,
    arithmetic_assumptions='finite CPU float32, round-to-nearest, nonzero operands/results normal, no overflow, source-pinned sum-of-72-squares then scalar mean')
CAPS = dict(v1.CAPS, initial_forward_completions=1, candidate_forward_completions=320,
    terminal_fit_forward_completions=1, terminal_gradient_completions=1, evaluation_forward_attempts=1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def admit():
    old, continuation, failed, identities = roundoff.admit()  # Admission only, no historical execution.
    _, history, _, more = v1.admit()
    identities.update(more)
    for name, sha in INPUTS.items():
        require(digest(ROOT/name) == sha, 'Accepted 3C-9b artifact changed: '+name)
    evidence = json.loads((ROOT/'docs/evidence/MATERIAL_GAUGE_ROUNDOFF_V1.json').read_text())
    freeze = json.loads((ROOT/'docs/protocols/MATERIAL_GAUGE_ROUNDOFF_V1.freeze.json').read_text())
    require(evidence['status'] == 'complete' and evidence['frozen_execution'] == freeze and
            evidence['work']['completed_shader_calls'] == evidence['work']['attempted_shader_calls'] == 6 and
            evidence['work']['completed_points'] == 576 and evidence['no_optimizer_or_derivatives'] and
            evidence['original_gate_unchanged'], 'Complete accepted six-call roundoff diagnostic required')
    for flag in ('global_torch_CPU_RNG_unchanged', 'global_numpy_RNG_unchanged', 'threads_restored',
                 'memory_limit_restored', 'profile_hook_restored', 'original_and_coupled_tapes_unchanged'):
        require(evidence[flag] is True, 'Roundoff integrity failure: '+flag)
    for group in (evidence['accepted_identities'], freeze['sha256']):
        for name, sha in group.items():
            require(ROOT in (ROOT/name).resolve().parents and digest(ROOT/name) == sha, 'Prior source/protocol changed: '+name)
            identities[name] = sha
    require(evidence['comparisons'][1]['native_vs_P0']['objective']['original_gate']['passed'] is False and
            failed['status'] == 'failed', 'Original failed gate must remain failed')
    identities.update(INPUTS)
    return old, history, continuation, identities


def work():
    return dict(v1.work(), initial_forward_completions=0, candidate_forward_completions=0,
                terminal_fit_forward_completions=0, terminal_gradient_completions=0, evaluation_forward_attempts=0)


def check_caps(counts):
    for name, cap in CAPS.items():
        v1.invariant(counts[name] <= cap, 'v2 arm work cap exceeded: '+name)
    for complete,attempt in (('initial_forward_completions','initial_forwards'),
        ('candidate_forward_completions','candidate_forwards'),('terminal_fit_forward_completions','terminal_fit_forwards'),
        ('terminal_gradient_completions','terminal_gradients'),('evaluation_forward_evaluations','evaluation_forward_attempts'),
        ('full_batch_forward_evaluations','forward_attempts'),('reverse_completed','reverse_mode_gradient_evaluations'),
        ('JVP_completed','jacobian_vector_evaluations'),('jacobians_completed','jacobian_constructions'),
        ('linear_solves_completed','linear_solves')):
        v1.invariant(counts[complete] <= counts[attempt], 'Completed work exceeds attempts: '+complete)
    v1.invariant(sum(counts[k] for k in ('accepted_parameter_steps','rejected_parameter_steps','branch_blocked_proposals'))
                 <= counts['linear_solves'], 'Trial classifications exceed attempted solves')


def arithmetic_runtime():
    require(torch.version.git_version == TORCH_GIT and torch.get_default_dtype() == torch.float32,
            'Reviewed Torch revision/default float32 required for arithmetic proof')
    mode = int(ctypes.CDLL(None).fegetround())
    require(mode == 0, 'FE_TONEAREST required; do not change rounding mode')
    one = torch.tensor(1., dtype=torch.float32)
    require(float(one+2.**-24) == 1. and float(one+3*2.**-24) == 1.+2.**-22,
            'Independent float32 ties-to-even check failed')
    return dict(torch_git_version=TORCH_GIT, CPU_capability=torch.backends.cpu.get_cpu_capability(),
                fegetround=mode, ties_to_even_probe=True, n=72, maximum_nontrivial_sum_depth=71,
                square_roundings=1, final_store_roundings_at_most=1,
                mean_scaling_roundings_at_most=2, conservative_gamma_k=75,
                binary_identity_limit='Source/runtime/probe evidence, not a binary build-identity proof')


def upward(value):
    return math.nextafter(float(value), math.inf)


def objective_envelope(base, candidate, target, loss0, loss1):
    """Exact rational construction of the predeclared gamma75 consistency bound."""
    a, b, t = (np.asarray(v, np.float32) for v in (base, candidate, target))
    require(a.shape == b.shape == t.shape == (24, 3) and all(np.isfinite(v).all() for v in (a,b,t)),
            'Gate E requires exactly72 finite RGB residuals')
    l0, r0, q0 = roundoff.mse32(a, t)
    l1, r1, q1 = roundoff.mse32(b, t)
    require(l0 == loss0 and l1 == loss1, 'Original float32 fitting arithmetic replay mismatch')
    tiny, largest = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    arrays = (a,b,t,r0,r1,q0,q1,a.astype(float)-t,b.astype(float)-t,
              r0.astype(float)**2,r1.astype(float)**2)
    require(all(np.all((v == 0) | (abs(v) >= tiny)) for v in arrays), 'Gate E normal-number assumption failed')
    require(all(np.sum(v.astype(float)**2) < largest/8 for v in (r0,r1)) and
            all(l == 0 or l >= tiny for l in (l0,l1)), 'Gate E no-overflow/normal-mean assumption failed')
    u = Fraction(1, 2**24)
    gamma = 75*u/(1-75*u)
    residual_bound = Fraction(0)
    squared_means = Fraction(0)
    for p0, p1, target_i, x, y in zip(a.flat,b.flat,t.flat,r0.flat,r1.flat):
        p0,p1,target_i,x,y = (Fraction(float(v)) for v in (p0,p1,target_i,x,y))
        eps = abs(p1-p0)+u*(abs(p0-target_i)+abs(p1-target_i))
        residual_bound += 2*abs(x)*eps+eps**2
        squared_means += x*x+y*y
    residual_bound /= 72
    reduction_bound = gamma*squared_means/72
    total = residual_bound+reduction_bound
    observed = abs(Fraction(loss1)-Fraction(loss0))
    return dict(passed=bool(observed <= total), residual_bound=upward(residual_bound),
        reduction_bound=upward(reduction_bound), total_bound=upward(total), observed_absolute_difference=float(observed),
        gamma_75=upward(gamma), exact_rational_comparison=True, normal_finite_assumptions_verified=True,
        bound_is_independent_equivalence_evidence=False,
        limits='Conditional arithmetic consistency only; cannot override physical/sample/mask/component gates')


class InitialObserver(lm.ShaderObserver):
    def __init__(self, shader, arm):
        super().__init__(shader)
        self.arm = arm

    def shade(self, *args, **kwargs):
        result = super().shade(*args, **kwargs)  # Original call exactly once; returned object unchanged.
        if self.arm.stage == 'initial':
            data = {k:result[k].detach().numpy().copy() for k in
                ('half_vectors','specular_directions','diffuse_directions','pdf','tangent','bitangent','normal')}
            data['per_tape'] = np.stack([result[k].detach().numpy().reshape(4,24,3).copy()
                for k in ('diffuse_rgb','specular_rgb','linear_rgb')], 2)
            data['views'] = args[2].detach().numpy().copy()
            data['encoding'] = args[7].detach().numpy().copy()
            data['mask'] = self.mask.copy()
            data['tape_arrays'] = [reference.sha_array(v.numpy()) for v in (args[9].u,args[9].v,args[9].azimuth)]
            self.arm.initial_capture = data
        return result


class Arm(v1.Arm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shader = InitialObserver(self.shader.shader, self)
        self.initial_capture = None

    def forward(self, z, evaluation=False, physical=False):
        phase = self.stage
        jacobian = self.in_jacobian
        if evaluation:
            self.row['work']['evaluation_forward_attempts'] += 1
        value = super().forward(z, evaluation=evaluation, physical=physical)
        key = ('initial_forward_completions' if phase == 'initial' else
               'candidate_forward_completions' if phase == 'optimization' and not jacobian else
               'terminal_fit_forward_completions' if phase == 'endpoint' and not evaluation else None)
        if key:
            self.row['work'][key] += 1
        if phase == 'initial':
            v1.invariant(self.initial_capture is not None, 'Missing initial source capture')
            self.initial_capture['mean'] = value['values'].detach().numpy().copy()
        check_caps(self.row['work'])
        return value

    def gradient(self, state, z):
        result = super().gradient(state, z)
        if self.stage == 'endpoint':
            self.row['work']['terminal_gradient_completions'] += 1
        check_caps(self.row['work'])
        return result


def bounded(a, b, atol, rtol, vector=False):
    a,b = np.asarray(a),np.asarray(b)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Finite same-layout initialization capture required')
    out = roundoff.compare(a,b,per_tape=a.shape[0] == 4,vector=vector)
    beyond = abs(a.astype(float)-b.astype(float)) > atol+rtol*abs(b.astype(float))
    out.update(atol=atol,rtol=rtol,exceeding_elements=int(beyond.sum()),
               per_channel_maximum=np.max(abs(a.astype(float)-b),axis=tuple(range(a.ndim-1))).tolist())
    if a.shape[0] == 4:
        out['exceeding_elements_per_tape'] = [int(v.sum()) for v in beyond]
    out['passed'] = bool(not beyond.any() and (not vector or out['max_angle_radians'] <= POLICY['angle_atol']))
    return out


def initialization_gates(p0, pu, pc, archived, target, observations, construction):
    rows = [a.row for a in (p0,pu,pc)]
    d0,dc = p0.initial_capture,pc.initial_capture
    require(d0 is not None and dc is not None, 'Initial captures absent')
    exact_keys = ('initial_state','initial_fit_predictions','initial_fit_prediction_sha256','initial_fit_MSE','initial_mask')
    ga = dict(passed=all(rows[0][k] == archived[k] for k in exact_keys),
              exact_fields={k:rows[0][k] == archived[k] for k in exact_keys},
              exact_source_target_tape_observation_runtime=True)
    physical0 = np.asarray(rows[0]['initial_state']['physical']['values'])
    physical1 = np.asarray(rows[2]['initial_state']['physical']['values'])
    q0,q1 = (coupling.accepted.tensor_q(p) for p in (physical0,physical1))
    f0,f1 = (np.stack((d['tangent'],d['bitangent']),-1) for d in (d0,dc))
    uniform = all(np.array_equal(f,np.broadcast_to(f[0],f.shape)) for f in (f0,f1))
    try:
        mapping = coupling.frame_sign(f0[0],f1[0])
        mapped = mapping['sign'] == construction['mapping']['sign']
    except SignatureError as exc:
        mapping,mapped = dict(failure=str(exc)),False
    same_p1 = rows[1]['initial_state'] == rows[2]['initial_state']
    gb = dict(passed=bool(np.all(abs(q1-q0) <= 2e-7+2e-6*abs(q0)) and
        physical0[3:].tobytes() == physical1[3:].tobytes() and uniform and mapped and same_p1),
        Q0=q0.tolist(),Q1=q1.tolist(),Q_relative=v1.q_difference(physical1,physical0),
        albedo_metallic_identical=physical0[3:].tobytes() == physical1[3:].tobytes(),
        batch_frames_uniform=uniform,actual_frame_mapping=mapping,signed_mapping_matches_construction=mapped,
        P1U_P1C_initial_state_identical=same_p1)
    sample = {}
    for key,atol in (('half_vectors',POLICY['h_atol']),('specular_directions',POLICY['wi_atol'])):
        sample[key] = bounded(dc[key].reshape(4,24,1024,3),d0[key].reshape(4,24,1024,3),atol,POLICY['vector_rtol'],True)
    diffuse_same = d0['diffuse_directions'].tobytes() == dc['diffuse_directions'].tobytes()
    xor = int(np.count_nonzero(d0['mask'] != dc['mask']))
    original_tape = [reference.sha_array(t.numpy()) for t in (p0.fit.tapes.u,p0.fit.tapes.v,p0.fit.tapes.azimuth)]
    coupled_tape = [r['sha256'] for r in construction['coupled_tape']['coupled_arrays']]
    tape_same = d0['tape_arrays'] == original_tape and dc['tape_arrays'] == coupled_tape
    dot0 = np.sum(d0['half_vectors'].astype(float)*d0['views'][:,None].astype(float),-1)
    dotc = np.sum(dc['half_vectors'].astype(float)*dc['views'][:,None].astype(float),-1)
    grazing = (abs(dot0) <= POLICY['grazing_indicator']) | (abs(dotc) <= POLICY['grazing_indicator'])
    pdf = roundoff.compare(dc['pdf'].reshape(4,24,1024,1),d0['pdf'].reshape(4,24,1024,1),True)
    pdf.update(grazing_count=int(grazing.sum()), indicator='abs(world wo-dot-h), reconstructed float64; not source-clamped PDF input',
        grazing_errors=roundoff.compare(dc['pdf'][...,0][grazing],d0['pdf'][...,0][grazing]) if grazing.any() else None,
        source_PDF_is_not_an_initial_gate=True,historical_grazing_exceptions=43)
    gc = dict(passed=bool(all(r['passed'] for r in sample.values()) and xor == 0 and diffuse_same and tape_same),
        fields=sample,reflection_mask_XOR_count=xor,diffuse_sampling_bitwise_equal=diffuse_same,
        sample_tape_identities_unchanged=tape_same,PDF=pdf)
    rgb,mean_replay = {},[]
    for d in (d0,dc):
        mean = np.stack([torch.from_numpy(d['per_tape'][:,:,i].copy()).mean(0).numpy() for i in range(3)],1)
        mean_replay.append(mean.tobytes() == d['mean'].tobytes())
    for i,name in enumerate(('diffuse','specular','total')):
        for key,a,b in (('per_tape',dc['per_tape'][:,:,i],d0['per_tape'][:,:,i]),('mean',dc['mean'][:,i],d0['mean'][:,i])):
            r = bounded(a,b,POLICY['RGB_atol'],POLICY['RGB_rtol'])
            if name == 'diffuse':
                r['passed'] = r['bitwise_equal']
                r['require_bitwise'] = True
            rgb[key+'_'+name] = r
    gd = dict(passed=bool(all(r['passed'] for r in rgb.values()) and all(mean_replay)),
              fields=rgb,original_four_tape_component_mean_replay=mean_replay)
    envelope = objective_envelope(d0['mean'][:,2],dc['mean'][:,2],target,rows[0]['initial_fit_MSE'],rows[2]['initial_fit_MSE'])
    decomposition = roundoff.mse_decomposition(d0['mean'][:,2],dc['mean'][:,2],target,observations)
    warning = decomposition['original_gate']
    ge = dict(passed=envelope['passed'],arithmetic_envelope=envelope,residual_decomposition=decomposition,
        L_P0=rows[0]['initial_fit_MSE'],L_P1U=rows[1]['initial_fit_MSE'],L_P1C=rows[2]['initial_fit_MSE'],
        old_scalar_gate_passed=warning['passed'],old_scalar_gate_difference=warning['absolute_error'],
        old_scalar_gate_bound=warning['frozen_bound'],old_scalar_gate_excess=warning['signed_distance_from_bound'],
        old_scalar_gate_is_warning_only=True)
    gates = dict(A=ga,B=gb,C=gc,D=gd,E=ge)
    return dict(seed=rows[0]['seed'],passed=all(g['passed'] for g in gates.values()),gates=gates)


def initial_gradient_diagnostics(row):
    if 'initial_J_normalized_float64' not in row:
        return dict(available=False,reason='Initial linearization did not complete')
    z = torch.tensor(row['initial_state']['unconstrained']['values'],dtype=torch.float32)
    y = torch.sigmoid(z).numpy().astype(float)
    d = y*(1-y)
    d[:2] *= .96
    d[2] = 1.
    require(np.isfinite(d).all() and (d > 0).all(), 'Initial analytic transform must be nonsingular')
    j = np.asarray(row['initial_J_normalized_float64'])
    r = np.asarray(row['initial_residual_float32'])
    jp = j/d
    return dict(available=True,latent_gradient=(2*j.T@r).tolist(),
        physical_gradient_chain_derived=(2*jp.T@r).tolist(),transform_diagonal_float64=d.tolist(),
        physical_normalized_J_chain_derived=jp.tolist(),
        limits='Analytical inverse transform at float32 sigmoid outputs; no independent physical-coordinate AD or additional derivative calls; float32 chain ordering is not reconstructed bitwise')


def analyze(rows):
    comparisons = v1.comparisons(rows)
    for record in comparisons:
        a,b = (next(r for r in rows if r['seed'] == record['seed'] and r['arm'] == name) for name in record['pair'])
        if all(r['initial_gradients']['available'] for r in (a,b)):
            record['initial_latent_gradient'] = v1.derivatives.gradient_comparison(
                np.asarray(b['initial_gradients']['latent_gradient'])[v1.PERM],a['initial_gradients']['latent_gradient'])
            record['initial_physical_gradient_chain_derived'] = v1.derivatives.gradient_comparison(
                np.asarray(b['initial_gradients']['physical_gradient_chain_derived'])[v1.PERM],a['initial_gradients']['physical_gradient_chain_derived'])
    return comparisons


def make_figures(report, destination):
    """Offline saved-state figures only; never render, differentiate or solve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'material-lm-gauge-trajectories-v2'
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    colors = dict(zip(v1.ARMS,('#222222','#c65102','#0868ac')))
    outputs = {}
    for name,xkey,ykey,xlabel,ylabel in (
        ('loss-outer','outer','training_MSE','Attempted outer iteration','Training MSE'),
        ('loss-accepted','accepted','training_MSE','Accepted updates','Training MSE'),
        ('loss-wall','wall','training_MSE','Active wall seconds','Training MSE'),
        ('loss-forwards','forwards','training_MSE','Full batch forwards (includes JVP primals)','Training MSE'),
        ('Q-accepted','accepted','Q_error','Accepted updates','Relative Q error to truth'),
        ('damping-outer','outer','damping','Attempted outer iteration','LM damping')):
        fig,axes = plt.subplots(1,5,figsize=(16,3.3),squeeze=False)
        for ax,seed in zip(axes[0],range(101,106)):
            for row in report['outcomes']:
                if row['seed'] != seed:
                    continue
                xs,ys = [],[]
                if ykey == 'damping':
                    for rec in row['iterations']:
                        xs.append(rec['outer_iteration'])
                        ys.append(rec['lambda_before'])
                    if row['iterations']:
                        xs.append(row['iterations'][-1]['outer_iteration'])
                        ys.append(row['damping'])
                else:
                    for point in row.get('trajectory_analysis',{}).get('points',[]):
                        if xkey == 'outer':
                            x = (0 if point['phase'] == 'initial' else point['outer_iteration']
                                 if point['outer_iteration'] is not None else
                                 row['iterations'][-1]['outer_iteration'] if row['iterations'] else 0)
                        else:
                            x = {'accepted':point['accepted_updates'],'wall':point['active_wall_seconds'],
                                 'forwards':point['work']['full_batch_forward_evaluations']}[xkey]
                        xs.append(x)
                        ys.append(point[ykey])
                if xs:
                    ax.semilogy(xs,ys,label=row['arm'],color=colors[row['arm']],marker='.',markersize=3)
                    if row['censored']:
                        ax.plot(xs[-1],ys[-1],marker='x',color=colors[row['arm']],markersize=7)
            ax.set_title('Seed '+str(seed))
            ax.set_xlabel(xlabel,fontsize=8)
            ax.grid(alpha=.25)
        axes[0,0].set_ylabel(ylabel)
        axes[0,-1].legend(fontsize=8)
        fig.tight_layout()
        path = destination/(name+'.svg')
        fig.savefig(path,metadata={'Date':None})
        plt.close(fig)
        # Normalize only newly generated SVG whitespace BEFORE its first identity hash.
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        outputs[str(path.relative_to(ROOT))] = digest(path)
    return outputs


SAFE_NUMERICAL_FAILURES = {'Nonfinite original shader predictions', 'Nonfinite directional derivative; no alternative derivative method'}


def execute(old, history, continuation, report, persist, whole):
    runtime = dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1)
    require(runtime == history['runtime'],'Original numerical runtime required')
    report.update(runtime=runtime,arithmetic_runtime=arithmetic_runtime())
    for stage,pin in PINS.items():
        baseline(stage,pin)
    require(old['scene'] == reference.design() and old['targets']['passed'] and old['identifiability']['passed'], 'Accepted scene/reference/gates required')
    targets = np.asarray(old['targets']['fine'],np.float64)
    target = torch.tensor(targets[:24,2],dtype=torch.float32)
    require(reference.sha_array(targets) == old['fine_targets_sha256'] and reference.sha_array(target.numpy()) == old['fit_target_float32_sha256'], 'Target identity changed')
    fit_tape,fit_ids = original.make_tapes(list(range(4)),24)
    eval_tape,eval_ids = original.make_tapes(list(range(1000,1004)),84)
    require(fit_ids == old['tapes']['fit'] and eval_ids == old['tapes']['evaluation'], 'Original fixed tapes changed')
    obs = old['scene']['observations']
    require(obs[:24] == [o for o in obs if o['split'] == 'fit'], 'Original fitting observation order required')
    shader = original.MaterialShaderV1('A2','atan2')
    require(shader.identity == old['shader'],'Accepted shader identity changed')
    evaluation = original.ObservationBatch(old['scene'],obs,eval_tape,4)
    arms,all_tapes,construction = [],[fit_tape,eval_tape],[]
    report.update(scene=old['scene'],shader=shader.identity,reused_gates=old['gates'],
        fine_targets_sha256=old['fine_targets_sha256'],fit_target_float32_sha256=old['fit_target_float32_sha256'],
        tapes=dict(fit=fit_ids,evaluation=eval_ids),outcomes=[],construction=construction,initialization=[])
    for saved in continuation['initializations']:
        z = v1.restore(saved)
        z1 = v1.gauge(z)
        p0,p1 = original.physical(z),original.physical(z1)
        mapping = coupling.frame_sign(coupling.decoded_frame(shader,p0,96),coupling.decoded_frame(shader,p1,96))
        copied,tape_record = coupling.copy_coupled_tape(fit_tape,mapping['sign'])
        construction.append(dict(seed=saved['seed'],mapping=mapping,coupled_tape=tape_record))
        all_tapes.append(copied)
        for name in v1.ARMS:
            selected = copied if name == 'P1-C' else fit_tape
            fit = original.ObservationBatch(old['scene'],obs[:24],selected,4)
            row = dict(seed=saved['seed'],arm=name,status='restored',budget_limited=False,censored=False,
                termination=None,completed_outer_iterations=0,work=work(),trajectory=[],iterations=[])
            arm = Arm(row,lm.ArmClock(whole),shader,fit,evaluation,target,(z if name == 'P0' else z1).clone().requires_grad_(),sign=mapping['sign'])
            arm.save_state()
            row['initial_state'] = copy.deepcopy(row['final_state'])
            arms.append(arm)
            report['outcomes'].append(row)
    hashes = [[reference.sha_array(t.numpy()) for t in (tape.u,tape.v,tape.azimuth)] for tape in all_tapes]
    report.update(restoration_gate='passed',phase='initial_forwards',optimization_began=False)
    persist()
    for arm in arms:
        report.update(active_seed=arm.row['seed'],active_arm=arm.row['arm'])
        with arm.clock.active():
            arm.current = lm.detached(arm.forward(arm.z))
            arm.row.update(initial_fit_MSE=float(arm.current['loss']),initial_fit_predictions=arm.current['values'].tolist(),
                initial_fit_prediction_sha256=reference.sha_array(arm.current['values'].numpy()),initial_mask=arm.current['mask_identity'])
            arm.row['initial_capture_identities'] = {k:roundoff.array_record(v, k == 'per_tape')
                for k,v in arm.initial_capture.items() if isinstance(v,np.ndarray)}
            arm.log(phase='initial',training_MSE=float(arm.current['loss']),unconstrained=arm.z.detach().tolist(),physical=original.physical(arm.z).detach().tolist())
        persist()
    report['phase'] = 'initial_gates'
    for i,seed in enumerate(range(101,106)):
        archived = next(r for r in history['outcomes'] if r['seed'] == seed and r['arm'] == 'LM')
        try:
            result = initialization_gates(*arms[3*i:3*i+3],archived,target.numpy(),obs[:24],construction[i])
        except SignatureError as exc:
            result = dict(seed=seed,passed=False,integrity_failure=str(exc))
        report['initialization'].append(result)
        persist()
    report['initialization_gate'] = 'passed' if all(r['passed'] for r in report['initialization']) else 'failed'
    persist()
    require(report['initialization_gate'] == 'passed','Mandatory v2 initialization failure; no optimization')
    require(hashes == [[reference.sha_array(t.numpy()) for t in (t.u,t.v,t.azimuth)] for t in all_tapes], 'Initial tapes mutated')
    for arm in arms:
        arm.initial_capture = None
    for arm in arms:
        report.update(phase='optimization',active_seed=arm.row['seed'],active_arm=arm.row['arm'],optimization_began=True)
        persist()
        try:
            v1.run_arm(arm,targets,obs,persist)
        except SignatureError as exc:
            # Only declared nonfinite-output failures with unchanged finite latent state are safe to isolate.
            if str(exc) not in SAFE_NUMERICAL_FAILURES or not bool(torch.isfinite(arm.z).all()):
                raise
            arm.row.update(status='failed_numerical',termination='nonfinite_numerical_failure',failure=str(exc))
            require(hashes == [[reference.sha_array(t.numpy()) for t in (t.u,t.v,t.azimuth)] for t in all_tapes], 'Tape mutation after numerical failure')
        arm.row['trajectory_analysis'] = v1.trajectory_analysis(arm.row)
        arm.row['initial_gradients'] = initial_gradient_diagnostics(arm.row)
        if arm.row['arm'] == 'P0':
            archived = next(r for r in history['outcomes'] if r['seed'] == arm.row['seed'] and r['arm'] == 'LM')
            arm.row['historical_control'] = v1.historical_control(arm.row,archived)
        print('{} {}: {}, {}, accepted={}, MSE={}'.format(arm.row['seed'],arm.row['arm'],arm.row['status'],arm.row['termination'],
            arm.row['work']['accepted_parameter_steps'],arm.row.get('final_fit_MSE')),flush=True)
        whole.check()
    require(hashes == [[reference.sha_array(t.numpy()) for t in (t.u,t.v,t.azimuth)] for t in all_tapes], 'Fixed tapes mutated')
    for arm in arms:
        check_caps(arm.row['work'])
    totals = {k:sum(a.row['work'][k] for a in arms) for k in work()}
    for k,cap in CAPS.items():
        require(totals[k] <= 15*cap,'Whole-run work cap exceeded: '+k)
    report.update(status='complete',phase='finished',total_work=totals,fixed_tapes_unchanged=True,comparisons=analyze(report['outcomes']))
    persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(all(os.environ.get(k) == v for k,v in original.REQUIRED_ENV.items()),'Explicit CPU/one-thread environment required')
    require(args.output.is_absolute() and args.output.resolve() == args.output == OUTPUT,'Fresh explicit versioned output required')
    freeze = json.loads((ROOT/('docs/protocols/'+NAME+'.freeze.json')).read_text())
    for name,sha in freeze['sha256'].items():
        require(digest(ROOT/name) == sha,'Pre-execution freeze mismatch: '+name)
    require(identity(POLICY) == freeze['numerical_policy_sha256'] and identity(CAPS) == freeze['arm_work_caps_sha256'], 'Gate/budget freeze mismatch')
    report = dict(schema_version=2,protocol='material-lm-gauge-trajectories-v2',milestone='3C-9c',base_commit=BASE,
        status='running',phase='admission',optimization_began=False,frozen_execution=freeze,protocol_sha256=digest(PROTOCOL),
        implementation_sha256={n:digest(ROOT/n) for n in CODE},numerical_policy=POLICY,
        settings=dict(original=original.SETTINGS,arms=list(v1.ARMS),seeds=list(range(101,106)),initial_lambda=.01,
            minimum_lambda=1e-8,maximum_lambda=1e8,outer_cap=80,trial_cap=4,arm_wall_seconds=120,
            endpoint_reserve_seconds=10,whole_wall_seconds=2400,peak_RSS_bytes=v1.RSS_BYTES,address_bytes=v1.ADDRESS_BYTES,
            arm_work_caps=CAPS,whole_work_caps={k:15*v for k,v in CAPS.items()},
            cross_arm_similarity=dict(Q_relative=.005,albedo_max=.002,metallic=.002)),
        GPU_execution=False,neural_training=False,new_targets=False,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        argv=[sys.executable,*sys.argv])
    threads,rng,nrng = torch.get_num_threads(),torch.get_rng_state().clone(),np.random.get_state()
    handlers = {s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
    address = resource.getrlimit(resource.RLIMIT_AS)
    whole = v1.WholeClock()
    begin,cpu = time.perf_counter(),time.process_time()
    with args.output.open('x') as stream:
        def persist():
            coupling.accepted.persist_report(stream,report)
        persist()
        try:
            for s in handlers:
                signal.signal(s,whole.alarm)
            signal.setitimer(signal.ITIMER_REAL,2400)
            cap = min(v1.ADDRESS_BYTES,address[0]) if address[0] >= 0 else v1.ADDRESS_BYTES
            if address[1] >= 0:
                cap = min(cap,address[1])
            resource.setrlimit(resource.RLIMIT_AS,(cap,address[1]))
            torch.set_num_threads(1)
            old,history,continuation,identities = admit()
            report['accepted_identities'] = identities
            execute(old,history,continuation,report,persist,whole)
            for name,sha in identities.items():
                require(digest(ROOT/name) == sha,'Accepted input changed during experiment: '+name)
            whole.check()
        except BaseException as exc:
            report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc))
            raise
        finally:
            after = np.random.get_state()
            report['global_torch_CPU_RNG_unchanged'] = torch.equal(rng,torch.get_rng_state())
            report['global_numpy_RNG_unchanged'] = nrng[0] == after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:] == after[2:]
            torch.set_num_threads(threads)
            resource.setrlimit(resource.RLIMIT_AS,address)
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,handler in handlers.items():
                signal.signal(s,handler)
            report.update(threads_restored=torch.get_num_threads() == threads,memory_limit_restored=resource.getrlimit(resource.RLIMIT_AS) == address,
                wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if not all(report[k] for k in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored')):
                report.update(status='failed',failure='Runtime state integrity failed')
            persist()
    print('Final status:',report['status'],flush=True)
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
