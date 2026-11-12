"""ONE frozen CPU three-material reference -> fresh Adam -> paired LM benchmark."""
import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
from contextlib import contextmanager
from types import FunctionType, SimpleNamespace

import numpy as np
import torch

from experiments import multi_material_reference_v1 as reference
from experiments import material_independent_tape_lm_v1 as previous
from experiments.material_recovery_metrics_v1 import gauge_metrics
from infrastructure.signatures import require, SignatureError
from infrastructure.stage_sources import baseline, PINS

accepted, original, v1, lm = previous.accepted, previous.original, previous.v1, previous.lm
oldref, coupling, roundoff = previous.reference, previous.coupling, previous.roundoff
ROOT = Path(__file__).resolve().parents[1]
BASE = '34548db58e7e0d3adb4cc2be1d2a4f1ed45443c7'
NAME = 'MULTI_MATERIAL_RECOVERY_V1'
SEEDS = (101, 103, 105)
TRAIN_SEEDS, EVAL_SEEDS = (3000,3001,3002,3003), (1000,1001,1002,1003)
CODE = ('experiments/multi_material_recovery_v1.py','experiments/multi_material_reference_v1.py',
        'tests/scientific/test_multi_material_recovery_v1.py')
PROTOCOL = ROOT/('docs/protocols/'+NAME+'.md')
OUTPUT = ROOT/('docs/evidence/'+NAME+'.json')
INPUTS = {
    'docs/evidence/MATERIAL_INDEPENDENT_TAPE_LM_V1.json':'bf1d8b4687244fb4625377031e59a5606ce354eace429aab3470894d11f9f77c',
    'docs/evidence/MATERIAL_INDEPENDENT_TAPE_LM_V1_ANALYSIS.json':'5cc6947679d18bcae19abef10c264d482fc5889fb27ccfd9f67998ea2ca26577',
    'docs/protocols/MATERIAL_INDEPENDENT_TAPE_LM_V1.freeze.json':'b8cbfc515d6bbcd5e37941564b29f7fff68e46a9e81ad99b9d0dbf6628b1aa0d',
    'docs/MATERIAL_INDEPENDENT_TAPE_LM_V1_RESULTS.md':'a4896b847ee371afd72a8f7a9dd8b67670cb9f9f41d4e5766dbc49ee37ab73b7',
}
ADAM_CAPS = dict(forward_attempts=1001, full_batch_forward_evaluations=1001,
    reverse_mode_gradient_evaluations=1000, reverse_completed=1000, optimizer_attempts=1000,
    optimizer_returns=1000, attempted_points=96096, fit_points=96096,
    diffuse_directions=49201152, specular_directions=98402304)
BUDGETS = dict(whole_wall_seconds=7200, Adam_active_seconds=180, LM_active_seconds=120,
    LM_endpoint_reserve=10, RSS_bytes=v1.RSS_BYTES, address_bytes=v1.ADDRESS_BYTES,
    reference_direction_evaluations=reference.REFERENCE_CAP, Adam_runs=9, LM_arms=27,
    Adam_per_run=ADAM_CAPS, LM_per_arm=accepted.CAPS,
    total_forward_attempts=9*1001+27*883, total_JVPs=27*560,
    total_reverse_sweeps=9*1000+27*1121, total_LM_solves=27*320,
    total_diffuse_directions=9*49201152+27*43524096,
    total_specular_directions=9*98402304+27*87048192)


def admit():
    archival, _, _, identities = previous.admit()
    for name,sha in INPUTS.items():
        require(accepted.digest(ROOT/name) == sha, 'Accepted 3C-10 artifact changed: '+name)
    history = json.loads((ROOT/'docs/evidence/MATERIAL_INDEPENDENT_TAPE_LM_V1.json').read_text())
    analysis = json.loads((ROOT/'docs/evidence/MATERIAL_INDEPENDENT_TAPE_LM_V1_ANALYSIS.json').read_text())
    freeze = json.loads((ROOT/'docs/protocols/MATERIAL_INDEPENDENT_TAPE_LM_V1.freeze.json').read_text())
    require(history['status'] == 'complete' and history['initialization_gate'] == 'passed' and
            len(history['outcomes']) == 15 and history['frozen_execution'] == freeze and
            analysis['source_report_sha256'] == INPUTS['docs/evidence/MATERIAL_INDEPENDENT_TAPE_LM_V1.json'],
            'Complete accepted 3C-10 evidence required')
    require(analysis['additional_shader_calls'] == analysis['additional_derivative_calls'] == analysis['additional_solves'] == 0,
            'Prior offline analysis provenance changed')
    for flag in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored','fixed_tapes_unchanged'):
        require(history[flag] is True, 'Prior runtime integrity failure')
    for group in (history['accepted_identities'], freeze['sha256'], analysis['figures'], INPUTS):
        for name,sha in group.items():
            require(ROOT in (ROOT/name).resolve().parents and accepted.digest(ROOT/name) == sha, 'Prior evidence changed: '+name)
            identities[name] = sha
    return archival, history, identities


def fresh_initializations():
    # Original constructor/private draws unchanged; select only the three approved seeds.
    rows = [r for r in original.initializations() if r['seed'] in SEEDS]
    require([r['seed'] for r in rows] == list(SEEDS), 'Original seed construction changed')
    for r in rows:
        z = lm.accepted.float32_exact(r['initial_unconstrained_float32'], (7,), 'fresh latent')
        r['initial_state'] = dict(unconstrained=lm.accepted.describe(z), physical=lm.accepted.describe(original.physical(z)))
    return rows


def tapes(expected=None):
    require(TRAIN_SEEDS == (3000,3001,3002,3003) and
        not set(TRAIN_SEEDS) & set((0,1,2,3,2000,2001,2002,2003,*EVAL_SEEDS)), 'Preregistered disjoint tapes required')
    rng,nrng = torch.get_rng_state().clone(), np.random.get_state()
    fit,fi = original.make_tapes(list(TRAIN_SEEDS),24)
    evaluation,ei = original.make_tapes(list(EVAL_SEEDS),84)
    after = np.random.get_state()
    require(torch.equal(rng,torch.get_rng_state()) and nrng[0] == after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:] == after[2:], 'Global RNG consumed')
    manifest = {}
    for name,t,ids in (('fit',fit,fi),('evaluation',evaluation,ei)):
        require(len({v.data_ptr() for v in (t.u,t.v,t.azimuth)}) == 3, 'Independent owned tape arrays required')
        manifest[name] = dict(per_tape=ids, arrays={k:roundoff.array_record(v.numpy()) for k,v in zip(('u','v','azimuth'),(t.u,t.v,t.azimuth))})
    if expected is not None:
        require(manifest == expected, 'Frozen train/evaluation tape mismatch; no replacement')
    return fit,evaluation,manifest


def material_target(row, material):
    p = reference.truth(material)
    require(row['material'] == material and row['truth'] == p.tolist() and row['targets']['passed'], 'Material-target association failed')
    values = np.asarray(row['targets']['fine'],np.float64)
    require(values.shape == (84,3,3) and oldref.sha_array(values) == row['targets']['fine_sha256'], 'Material target hash/layout mismatch')
    fit = values[:24,2].astype(np.float32)
    require(oldref.sha_array(fit) == row['fit_target_float32_sha256'], 'Material float32 target mismatch')
    return values,torch.from_numpy(fit)


def recovery_errors(p, truth):
    p, truth = reference.validate(p), reference.validate(truth)
    g = gauge_metrics(p[:3],truth[:3]); dk = p[3:6]-truth[3:6]
    return dict(**g, sorted_axis_relative_errors=(abs(np.sort(p[:2])-np.sort(truth[:2]))/np.sort(truth[:2])).tolist(),
        orientation_degrees=float(np.rad2deg(g['major_axis_error_radians'])) if g['orientation_identifiable'] else None,
        albedo_absolute_errors=abs(dk).tolist(), albedo_RMSE=float(np.sqrt(np.mean(dk**2))),
        albedo_relative_L2=float(np.linalg.norm(dk)/max(np.linalg.norm(truth[3:6]),1e-4)),
        metallic_absolute_error=float(abs(p[6]-truth[6])), metallic_near_boundary=bool(p[6] <= .01 or p[6] >= .99))


class WholeClock(v1.WholeClock):
    def remaining(self):
        return 7200-(time.perf_counter()-self.start)

    def check(self):
        v1.memory_check()
        if self.remaining() <= 0:
            raise lm.WholeLimit('Single 7200-second process cap; no retry')


class AdamClock(lm.ArmClock):
    def check(self, reserve=0.):
        self.whole.check()
        if self.elapsed() >= 180:
            raise lm.ArmLimit('Adam 180-second accumulated active cap')

    @contextmanager
    def active(self):
        require(self.started is None and self.whole.active is None, 'Nested active clock')
        self.check(); self.started=(time.perf_counter(),time.process_time()); self.whole.active=self
        signal.setitimer(signal.ITIMER_REAL,max(1e-6,min(self.whole.remaining(),180-self.wall)))
        try:
            yield
        finally:
            self.wall+=time.perf_counter()-self.started[0]; self.cpu+=time.process_time()-self.started[1]
            self.started=None; self.whole.active=None
            signal.setitimer(signal.ITIMER_REAL,max(1e-6,self.whole.remaining()))


def check_adam(counts):
    for k,cap in ADAM_CAPS.items():
        require(counts[k] <= cap, 'Adam work cap exceeded: '+k)


def check_total_work(report):
    adam=report.get('Adam_runs',[]); arms=report.get('outcomes',[])
    require(len(adam) <= 9 and len(arms) <= 27, 'Preregistered population exceeded')
    for r in adam:
        if 'work' in r:check_adam(r['work'])
    for r in arms:accepted.check_caps(r['work'])
    fields={'total_forward_attempts':'forward_attempts','total_reverse_sweeps':'reverse_mode_gradient_evaluations',
        'total_diffuse_directions':'diffuse_directions','total_specular_directions':'specular_directions'}
    totals={k:sum(r.get('work',{}).get(field,0) for r in adam+arms) for k,field in fields.items()}
    totals.update(total_JVPs=sum(r['work']['jacobian_vector_evaluations'] for r in arms),
        total_LM_solves=sum(r['work']['linear_solves'] for r in arms))
    for k,value in totals.items():require(value <= BUDGETS[k], 'Whole benchmark work cap exceeded: '+k)
    return totals


def adam_forward(row, batch, shader, z, target):
    c=row['work']; check_adam(c)
    require(c['forward_attempts'] < 1001, 'Adam forward cap exhausted')
    c['forward_attempts']+=1; c['attempted_points']+=96
    prediction=batch(shader,original.physical(z))
    c['full_batch_forward_evaluations']+=1; c['fit_points']+=96
    c['diffuse_directions']+=96*512; c['specular_directions']+=96*1024
    loss=torch.mean((prediction[:,2]-target)**2)
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(loss)), 'Nonfinite Adam predictions/loss')
    check_adam(c)
    return loss


def fit_adam(row, shader, batch, target, clock, persist):
    require(row['status'] == 'not_started' and row['completed_updates'] == 0, 'Fresh uninterrupted Adam only')
    z=lm.accepted.float32_exact(row['initial_state']['unconstrained']['values'],(7,),'fresh Adam').requires_grad_()
    require(lm.accepted.describe(original.physical(z)) == row['initial_state']['physical'], 'Fresh physical mismatch')
    config=original.SETTINGS['optimizer']
    optimizer=torch.optim.Adam([z],lr=config['lr'],betas=tuple(config['betas']),eps=config['eps'],weight_decay=config['weight_decay'])
    row.update(status='optimizing',work={k:0 for k in ADAM_CAPS},snapshots={})
    try:
        with clock.active():
            for update in range(1,1001):
                clock.check(); optimizer.zero_grad()
                loss=adam_forward(row,batch,shader,z,target)
                if update == 251:
                    require(lm.accepted.describe(z) == row['snapshots']['250']['unconstrained'], 'Snapshot/next-forward state mismatch')
                    row['snapshots']['250'].update(objective=float(loss.detach()),objective_at_update=250,
                        objective_source='Reused original update251 forward, before its gradient/update')
                row['work']['reverse_mode_gradient_evaluations']+=1
                loss.backward(); row['work']['reverse_completed']+=1
                require(z.grad is not None and bool(torch.isfinite(z.grad).all()), 'Missing/nonfinite Adam gradient')
                grad=float(torch.norm(z.grad))
                clock.check(); row['work']['optimizer_attempts']+=1
                optimizer.step(); row['work']['optimizer_returns']+=1; row['completed_updates']+=1
                require(float(optimizer.state[z]['step']) == update and bool(torch.isfinite(z).all()) and
                    all(bool(torch.isfinite(v).all()) for v in optimizer.state[z].values()), 'Adam state/counter invalid')
                if update in (250,1000):
                    row['snapshots'][str(update)] = lm.accepted.snapshot(z,optimizer)
                if update == 1 or update % 10 == 0:
                    row['latest_state']=lm.accepted.snapshot(z,optimizer)
                    row['history'].append(dict(update=update,loss_at_update=update-1,loss=float(loss.detach()),gradient_norm=grad,
                        physical=original.physical(z).detach().tolist(),unconstrained=z.detach().tolist(),work=copy.deepcopy(row['work']),
                        active_wall_seconds=clock.elapsed()))
                    persist()
                check_adam(row['work'])
            clock.check()
            with torch.no_grad():
                endpoint=adam_forward(row,batch,shader,z,target)
            row['snapshots']['1000'].update(objective=float(endpoint),objective_at_update=1000,objective_source='One counted terminal forward; no backward/update')
            row.update(status='complete',latest_state=lm.accepted.snapshot(z,optimizer),final=original.physical(z).detach().tolist(),
                termination='exact_1000_update_ceiling')
            clock.check()
    except BaseException as exc:
        row.update(status='failed',failure_type=type(exc).__name__,failure=str(exc),latest_state=lm.accepted.snapshot(z,optimizer))
        raise
    finally:
        row.update(active_wall_seconds=clock.wall,CPU_seconds=clock.cpu); persist()


def initial_gates(arms, saved, target, observations, construction, manifest):
    # Reuse B--E unchanged. The helper's self-comparison A is discarded: no old RGB identity claim.
    p0,pu,pc=arms
    result=accepted.initialization_gates(p0,pu,pc,p0.row,target,observations,construction)
    replays=[]
    original_hashes=[manifest['fit']['arrays'][k]['sha256'] for k in ('u','v','azimuth')]
    copied_hashes=[v['sha256'] for v in construction['coupled_tape']['coupled_arrays']]
    for arm,hashes in zip(arms,(original_hashes,original_hashes,copied_hashes)):
        d=arm.initial_capture
        replay=np.stack([torch.from_numpy(d['per_tape'][:,:,i].copy()).mean(0).numpy() for i in range(3)],1)
        mean=d['mean']; saved_mean=roundoff.saved_mean(arm.row)
        passed=(replay.tobytes() == mean.tobytes() == saved_mean.tobytes() and
            roundoff.mse32(mean[:,2],target)[0] == arm.row['initial_fit_MSE'] and d['tape_arrays'] == hashes)
        replays.append(dict(arm=arm.row['arm'],passed=passed))
    checks=dict(exact_new_Adam_state=p0.row['initial_state'] == {k:saved['latest_state'][k] for k in ('unconstrained','physical')},
        exact_1000_returns=saved['status'] == 'complete' and saved['completed_updates'] == 1000,
        target_identity=all(a.row['fit_target_float32_sha256'] == oldref.sha_array(np.asarray(target,np.float32)) for a in arms),
        internal_replays=all(r['passed'] for r in replays),source_scene_runtime=True)
    result['gates']['A']=dict(passed=all(checks.values()),checks=checks,replays=replays,archival_RGB_parity_required=False)
    result.update(passed=all(v['passed'] for v in result['gates'].values()),material=saved['material'])
    return result


def evaluate(arm, targets, observations, truth):
    p=original.physical(arm.z).detach(); mapped=p.clone()
    if arm.row['arm'] != 'P0':
        mapped=p[v1.PERM].clone(); mapped[2]=p[2]-math.pi/2
    ideal=p.double()[v1.PERM].numpy() if arm.row['arm'] != 'P0' else p.double().numpy()
    if arm.row['arm'] != 'P0':
        ideal[2]-=math.pi/2
    arm.row.update(final=p.tolist(),evaluation_gauge=dict(physical=mapped.tolist(),ideal_float64=ideal.tolist(),
        rounding_delta=(mapped.double().numpy()-ideal).tolist(),tapes='original 1000--1003',
        Q_relative_mapping_error=v1.q_difference(mapped.tolist(),p.tolist())))
    arm.check()
    with torch.no_grad():
        predictions=arm.forward(mapped,evaluation=True,physical=True).numpy().astype(np.float64)
    metrics={}
    for split in ('fit','held_out','relight','relight_seen_views','relight_unseen_views'):
        mask=[o['split'] == split if split in ('fit','held_out','relight') else
            o['split']=='relight' and (o['view']<6 if split=='relight_seen_views' else o['view']>=6) for o in observations]
        metrics[split]={name:original.radiance_errors(predictions[mask,i],targets[mask,i]) for i,name in enumerate(('diffuse','specular','total'))}
    errors=recovery_errors(mapped.numpy(),truth)
    delta=predictions-targets; compensation={}
    for split in ('fit','held_out','relight'):
        mask=[o['split']==split for o in observations]
        d,s,t=(delta[mask,i] for i in range(3))
        compensation[split]=dict(diffuse_specular_inner_product=float(np.sum(d*s)),
            residual_correlation=float(np.sum(d*s)/max(np.linalg.norm(d)*np.linalg.norm(s),1e-30)),
            total_squared_error=float(np.sum(t*t)),component_squared_error=float(np.sum(d*d)+np.sum(s*s)))
    arm.row.update(final_predictions=predictions.tolist(),prediction_sha256=oldref.sha_array(predictions),
        recovery_errors=errors,unmapped_recovery_errors=recovery_errors(p.numpy(),truth),radiance_errors=metrics,
        success=original.assess_success(errors,metrics),component_compensation=compensation)


def run_arm(arm, targets, observations, truth, persist):
    # Private observational namespace: original branch/LM/endpoint code object, explicit-truth evaluator only.
    namespace=dict(v1.run_arm.__globals__,evaluate=lambda a,t,o:evaluate(a,t,o,truth))
    FunctionType(v1.run_arm.__code__,namespace)(arm,targets,observations,persist)


def trajectory_analysis(row, truth):
    # Same offline historical function, with only its explicit metric dependency bound locally.
    adapter=SimpleNamespace(recovery_errors=lambda p:recovery_errors(p,truth))
    return FunctionType(v1.trajectory_analysis.__code__,dict(v1.trajectory_analysis.__globals__,original=adapter))(row)


def state_difference(a, b):
    # Preserve inherited physical similarity bounds; nullable orientation is descriptive only.
    g=gauge_metrics(np.asarray(a)[:3],np.asarray(b)[:3])
    q=v1.q_difference(a,b); k=float(np.max(abs(np.asarray(a)[3:6]-np.asarray(b)[3:6]))); metal=abs(a[6]-b[6])
    return dict(Q_relative=q,max_albedo_absolute=k,metallic_absolute=metal,
        orientation_degrees=float(np.rad2deg(g['major_axis_error_radians'])) if g['orientation_identifiable'] else None,
        orientation_identifiable=g['orientation_identifiable'],practical_similarity=bool(q<=.005 and k<=.002 and metal<=.002))


def comparisons(rows):
    # Original formula/thresholds; restrict the sole hard-coded outer seed enumeration.
    def selected_range(*args):
        require(args == (101,106), 'Unexpected historical comparison enumeration')
        return SEEDS
    result=FunctionType(v1.comparisons.__code__,dict(v1.comparisons.__globals__,range=selected_range,state_difference=state_difference))(rows)
    for c in result:
        a,b=(next(r for r in rows if r['seed']==c['seed'] and r['arm']==n) for n in c['pair'])
        if all(r['initial_gradients']['available'] for r in (a,b)):
            for field,key in [('latent_gradient','initial_latent_gradient'),('physical_gradient_chain_derived','initial_physical_gradient_chain_derived')]:
                c[key]=v1.derivatives.gradient_comparison(np.asarray(b['initial_gradients'][field])[v1.PERM],a['initial_gradients'][field])
    return result


def execute(report,persist,whole,freeze):
    archival,history,identities=admit(); report['accepted_identities']=identities
    runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1)
    require(runtime==history['runtime'], 'Exact accepted numerical runtime required')
    report.update(runtime=runtime,arithmetic_runtime=accepted.arithmetic_runtime())
    for stage,pin in PINS.items():
        baseline(stage,pin)
    require(reference.PANEL==freeze['material_panel'] and fresh_initializations()==freeze['initializations'], 'Frozen panel/initializations changed')
    scene=oldref.design(); obs=scene['observations']; require(scene==archival['scene'],'Original scene/order changed')
    report.update(scene=scene,materials=[],Adam_runs=[],outcomes=[],initialization=[],comparisons=[],reference_work={'reference_direction_evaluations':0},optimization_began=False)
    fit_tape,eval_tape,manifest=tapes(freeze['tapes'])
    def verify_original_tapes():
        for key,t in (('fit',fit_tape),('evaluation',eval_tape)):
            require(manifest[key]['arrays']=={n:roundoff.array_record(v.numpy()) for n,v in zip(('u','v','azimuth'),(t.u,t.v,t.azimuth))}, 'Original runtime tapes mutated')
        report['fixed_tapes_unchanged']=True
    require(manifest['evaluation']['per_tape']==archival['tapes']['evaluation'],'Original evaluation tapes changed')
    report.update(tapes=manifest,phase='reference_targets'); persist()
    grids=oldref.make_grids(); whole.check()
    report['reference_grid_identities']=[{k:g[k] for k in ('shape','directions_sha256','weights_sha256','lighting_sha256')} for g in grids]
    require(report['reference_grid_identities']==archival['reference_grid_identities']==freeze['reference_grid_identities'], 'Frozen independent reference grids changed')
    report['reference_work'].update(reference_grid_directions=sum(len(g['directions']) for g in grids),
        reference_precomputed_scalar_lights=7*sum(len(g['directions']) for g in grids))
    for material in reference.PANEL:
        result=reference.render(reference.truth(material),obs,grids,report['reference_work'],whole.check)
        row=dict(material=material,truth=reference.truth(material).tolist(),targets=oldref.serial_reference(result),
            fit_target_float32_sha256=oldref.sha_array(result['fine'][:24,2].astype(np.float32)))
        report['materials'].append(row); persist()
        print(material,'target convergence:',result['passed'],'max bound ratio',result['maximum_bound_ratio'],flush=True)
    if not all(r['targets']['passed'] for r in report['materials']):
        verify_original_tapes()
        report.update(status='stopped_reference_gate',reason='Mandatory material target convergence failed; no optimization'); persist(); return
    report['phase']='identifiability'; persist()
    for row in report['materials']:
        row['identifiability']=reference.identifiability(row['truth'],grids,obs,report['reference_work'],whole.check,
            lambda i,j:print(row['material'],'reference Jacobian',i+1,j+1,flush=True))
        persist()
    require(report['reference_work']['reference_direction_evaluations']==reference.REFERENCE_CAP,'Exact prescribed reference work required')
    if not all(r['identifiability']['passed'] for r in report['materials']):
        verify_original_tapes()
        report.update(status='stopped_identifiability_gate',reason='Mandatory independent local rank/stability gate failed; no optimization'); persist(); return
    del grids
    report.update(reference_gate='passed',phase='Adam',optimization_began=True); persist()
    shader=original.MaterialShaderV1('A2','atan2'); require(shader.identity==history['shader'],'Approved shader changed')
    report['shader']=shader.identity
    fit=original.ObservationBatch(scene,obs[:24],fit_tape,4)
    evaluation=original.ObservationBatch(scene,obs,eval_tape,4)
    for row in report['materials']:
        _,target=material_target(row,row['material'])
        for init in fresh_initializations():
            saved=copy.deepcopy(init); saved.update(material=row['material'],fit_target_float32_sha256=row['fit_target_float32_sha256'])
            report['Adam_runs'].append(saved)
            report.update(active_material=row['material'],active_seed=saved['seed']); persist()
            fit_adam(saved,shader,fit,target,AdamClock(whole),persist)
            verify_original_tapes(); check_total_work(report)
            print(row['material'],saved['seed'],'Adam returned',saved['completed_updates'],flush=True)
    require(len(report['Adam_runs'])==9 and all(r['status']=='complete' and r['completed_updates']==1000 for r in report['Adam_runs']), 'All nine new Adam endpoints required')
    arms=[]; all_tapes=[fit_tape,eval_tape]; constructions=[]
    for saved in report['Adam_runs']:
        row=next(r for r in report['materials'] if r['material']==saved['material'])
        _,target=material_target(row,saved['material'])
        z=v1.restore(saved); z1=v1.gauge(z)
        mapping=coupling.frame_sign(coupling.decoded_frame(shader,original.physical(z),96),coupling.decoded_frame(shader,original.physical(z1),96))
        copied,record=coupling.copy_coupled_tape(fit_tape,mapping['sign']); all_tapes.append(copied)
        construction=dict(material=saved['material'],seed=saved['seed'],mapping=mapping,coupled_tape=record); constructions.append(construction)
        for name in v1.ARMS:
            batch=original.ObservationBatch(scene,obs[:24],copied if name=='P1-C' else fit_tape,4)
            r=dict(material=saved['material'],seed=saved['seed'],arm=name,status='restored',budget_limited=False,censored=False,
                termination=None,completed_outer_iterations=0,work=accepted.work(),trajectory=[],iterations=[],
                fit_target_float32_sha256=row['fit_target_float32_sha256'])
            arm=accepted.Arm(r,lm.ArmClock(whole),shader,batch,evaluation,target,(z if name=='P0' else z1).clone().requires_grad_(),sign=mapping['sign'])
            arm.save_state(); r['initial_state']=copy.deepcopy(r['final_state']); arms.append(arm); report['outcomes'].append(r)
    hashes=[[oldref.sha_array(v.numpy()) for v in (t.u,t.v,t.azimuth)] for t in all_tapes]
    report.update(phase='LM_initial_forwards',construction=constructions,LM_optimization_began=False); persist()
    for arm in arms:
        with arm.clock.active():
            arm.current=lm.detached(arm.forward(arm.z))
            arm.row.update(initial_fit_MSE=float(arm.current['loss']),initial_fit_predictions=arm.current['values'].tolist(),
                initial_fit_prediction_sha256=oldref.sha_array(arm.current['values'].numpy()),initial_mask=arm.current['mask_identity'],
                initial_capture_identities={k:roundoff.array_record(v,k=='per_tape') for k,v in arm.initial_capture.items() if isinstance(v,np.ndarray)})
            arm.log(phase='initial',training_MSE=float(arm.current['loss']),unconstrained=arm.z.detach().tolist(),physical=original.physical(arm.z).detach().tolist())
        persist()
    report['phase']='LM_initial_gates'
    for i,saved in enumerate(report['Adam_runs']):
        try:
            g=initial_gates(arms[3*i:3*i+3],saved,arms[3*i].target.numpy(),obs[:24],constructions[i],manifest)
        except SignatureError as exc:
            g=dict(material=saved['material'],seed=saved['seed'],passed=False,integrity_failure=str(exc))
        report['initialization'].append(g); persist()
    report['initialization_gate']='passed' if all(g['passed'] for g in report['initialization']) else 'failed'; persist()
    require(hashes==[[oldref.sha_array(v.numpy()) for v in (t.u,t.v,t.azimuth)] for t in all_tapes],'Initial tape mutation')
    verify_original_tapes()
    if report['initialization_gate'] != 'passed':
        report.update(status='stopped_initialization_gate',reason='Mandatory gauge gate failed; no LM optimization'); persist(); return
    for arm in arms:
        arm.initial_capture=None
    for arm in arms:
        row=next(r for r in report['materials'] if r['material']==arm.row['material'])
        targets,_=material_target(row,arm.row['material'])
        report.update(phase='LM',LM_optimization_began=True,active_material=row['material'],active_seed=arm.row['seed'],active_arm=arm.row['arm']); persist()
        try:
            run_arm(arm,targets,obs,row['truth'],persist)
        except SignatureError as exc:
            if str(exc) not in accepted.SAFE_NUMERICAL_FAILURES or not bool(torch.isfinite(arm.z).all()):
                raise
            arm.row.update(status='failed_numerical',termination='nonfinite_numerical_failure',failure=str(exc))
        arm.row['trajectory_analysis']=trajectory_analysis(arm.row,row['truth'])
        arm.row['initial_gradients']=accepted.initial_gradient_diagnostics(arm.row)
        require(hashes==[[oldref.sha_array(v.numpy()) for v in (t.u,t.v,t.azimuth)] for t in all_tapes],'Fixed tape mutation')
        print(row['material'],arm.row['seed'],arm.row['arm'],arm.row['status'],arm.row.get('success'),flush=True)
        persist(); whole.check()
    for material in reference.PANEL:
        for c in comparisons([r for r in report['outcomes'] if r['material']==material]):
            c['material']=material; report['comparisons'].append(c)
    for arm in arms:
        accepted.check_caps(arm.row['work'])
    report.update(status='complete',phase='finished',fixed_tapes_unchanged=True,total_work=check_total_work(report)); persist()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    require(all(os.environ.get(k)==v for k,v in original.REQUIRED_ENV.items()),'Explicit one-thread CPU environment required')
    require(args.output.is_absolute() and args.output.resolve()==args.output==OUTPUT,'Exclusive explicit versioned output required')
    freeze=json.loads((ROOT/('docs/protocols/'+NAME+'.freeze.json')).read_text())
    for name,sha in freeze['sha256'].items():
        require(accepted.digest(ROOT/name)==sha,'Pre-execution source freeze mismatch: '+name)
    require(freeze['budgets']==BUDGETS and freeze['numerical_policy_sha256']==accepted.identity(accepted.POLICY),'Frozen policy/budgets changed')
    report=dict(schema_version=1,protocol='multi-material-recovery-v1',milestone='3D-1',base_commit=BASE,status='running',phase='admission',
        frozen_execution=freeze,material_panel=reference.PANEL,budgets=BUDGETS,settings=dict(shader='A2/atan2',optimizer=original.SETTINGS['optimizer'],
        success=original.SETTINGS['success'],LM=dict(initial_lambda=.01,minimum_lambda=1e-8,maximum_lambda=1e8,outers=80,trials=4)),
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_execution=False,neural_training=False,argv=[sys.executable,*sys.argv])
    threads,rng,nrng=torch.get_num_threads(),torch.get_rng_state().clone(),np.random.get_state()
    address=resource.getrlimit(resource.RLIMIT_AS); handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
    whole=WholeClock(); begin,cpu=time.perf_counter(),time.process_time()
    with args.output.open('x') as stream:
        def persist():
            coupling.accepted.persist_report(stream,report)
        persist()
        try:
            for s in handlers:signal.signal(s,whole.alarm)
            signal.setitimer(signal.ITIMER_REAL,7200)
            cap=min(v1.ADDRESS_BYTES,address[0]) if address[0]>=0 else v1.ADDRESS_BYTES
            if address[1]>=0:cap=min(cap,address[1])
            resource.setrlimit(resource.RLIMIT_AS,(cap,address[1])); torch.set_num_threads(1)
            execute(report,persist,whole,freeze)
            for name,sha in report.get('accepted_identities',{}).items():
                require(accepted.digest(ROOT/name)==sha,'Accepted artifact modified during execution')
            whole.check()
        except BaseException as exc:
            report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc)); raise
        finally:
            after=np.random.get_state()
            report.update(global_torch_CPU_RNG_unchanged=torch.equal(rng,torch.get_rng_state()),
                global_numpy_RNG_unchanged=nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:])
            torch.set_num_threads(threads); resource.setrlimit(resource.RLIMIT_AS,address); signal.setitimer(signal.ITIMER_REAL,0)
            for s,h in handlers.items():signal.signal(s,h)
            report.update(threads_restored=torch.get_num_threads()==threads,memory_limit_restored=resource.getrlimit(resource.RLIMIT_AS)==address,
                wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if not all(report[k] for k in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored')):
                report.update(status='failed',failure='Runtime state integrity failed')
            persist()
    print('Final status:',report['status'],flush=True)
    return 0 if report['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
