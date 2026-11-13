"""Explicitly authorized remaining-history continuation, never an original-run repair."""
import argparse
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import time

from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.signatures import require, read_report
from infrastructure.scientific_reporting_adapter_v2 import Tracker, plain
from infrastructure.neural_material_journal_runtime_v2 import RuntimeOutput
from infrastructure.paths import contained_path

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = 'dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af'
PARENT_REPORT_SHA = 'bded2f341b90d3480c2c1c6107bb712d46d5a863cd1b9a8a674753c295858eb2'
PARENT_FINAL_SHA = '36e95cb46a7ae47ad4e1b9121c5b194a6f0da04f80927bb67499756438504413'
REPLACEMENTS = (
    ("require(row['status']=='admitted' and row['completed_updates']==0,'Fresh admitted history required')",
     "require(row['status']=='optimizing' and row['completed_updates']==1000 and model.arm=='N1','Explicit update1000 N1 resume required')"),
    ("row.update(status='optimizing',history=[],snapshots={})", "row.update(status='optimizing')"),
    ("require(not optimizer.state,'Fresh Adam state required; no implicit continuation')",
     "require(bool(optimizer.state),'Explicit restored Adam state required')"),
    ('range(1,UPDATES+1)', 'range(1001,UPDATES+1)'),
)


def resume_loop(h, factory):
    original = inspect.getsource(h.train_history)
    changed = original
    for before, after in REPLACEMENTS:
        require(changed.count(before) == 1, 'Pinned resume edit site changed')
        changed = changed.replace(before, after)
    normalized = changed
    for before, after in reversed(REPLACEMENTS):
        normalized = normalized.replace(after, before)
    require(ast.dump(ast.parse(normalized)) == ast.dump(ast.parse(original)), 'Scientific loop AST changed')
    namespace = dict(vars(h), optimizer_for=factory)
    exec(compile(changed, 'explicit-update1000-continuation', 'exec'), namespace)
    return namespace['train_history']


def resumed_row(old):
    require((old['material'], old['seed'], old['arm'], old['completed_updates']) ==
            ('M1', 105, 'N1', 4270), 'Unexpected interrupted history')
    row = copy.deepcopy(old)
    logs = {r['update']: r for r in row['history']}
    require(1000 in logs and 1010 in logs and set(row['snapshots']) == {'250', '1000'},
            'Missing exact checkpoint/prefix accounting')
    # At checkpoint1000, the recorded log precedes the already completed no-grad
    # prefix forward. Infer its counters independently from the next saved log.
    c = copy.deepcopy(logs[1000]['work'])
    for k in ('field_attempts', 'field_completed', 'shader_attempts', 'shader_completed',
              'fit_forwards', 'prefix_forwards'):
        c[k] += 1
    for k, n in (('attempted_points', 96), ('completed_points', 96),
                 ('diffuse_directions', 96*512), ('specular_directions', 96*1024)):
        c[k] += n
    check = copy.deepcopy(c)
    for k in ('field_attempts', 'field_completed', 'shader_attempts', 'shader_completed',
              'fit_forwards', 'reverse_attempts', 'reverse_completed',
              'optimizer_attempts', 'optimizer_returns'):
        check[k] += 10
    for k, n in (('attempted_points', 960), ('completed_points', 960),
                 ('diffuse_directions', 960*512), ('specular_directions', 960*1024)):
        check[k] += n
    require(check == logs[1010]['work'], 'Inherited checkpoint counter reconstruction differs')
    row.update(status='optimizing', completed_updates=1000, work=c,
               history=[r for r in row['history'] if r['update'] <= 1000])
    row['continuation'] = dict(restored_update=1000, previously_logged_through=4270,
        replay_updates=3270, new_progress_updates=730, restored_checkpoint_exact=False,
        replay_verified_logged_updates=[], historical_checkpoint_clock_upper_seconds=logs[1010]['active_wall_seconds'],
        full_historical_update4270_state_available=False)
    return row


class ReadOnlyInput:
    def __init__(self, root): self.root = Path(root)
    def path(self, name): return contained_path(self.root, name, writable=False)


def admit(review):
    from experiments.freeze_neural_material_recovery_logged_v2 import validate_freeze
    from experiments import neural_material_recovery_v1 as h
    require(review['phase'] == 'D.logged-v2-continuation-v1' and review['authorized'] is True and
            review['replay_from_update1000_authorized'] is True and review['maximum_new_optimizer_returns'] == 34000,
            'Separate remaining-history/replay authorization required')
    freeze = validate_freeze(Path(review['freeze_path']), FREEZE_SHA)
    require(review['freeze_sha256'] == FREEZE_SHA and
            all(log.file_sha(ROOT/p) == sha for p, sha in review['continuation_sources'].items()),
            'Continuation source/protocol identity mismatch')
    require(review['external_watchdog'] == dict(wall_seconds=28800, kill_grace_seconds=10), 'Watchdog changed')
    require(review['prior_observed_seconds'] == 8280, 'Conservative prior wall charge changed')
    require(all(os.environ.get(k) == v for k, v in h.previous.original.REQUIRED_ENV.items()), 'CPU environment changed')
    import platform
    require(dict(python=platform.python_version(), numpy=h.np.__version__, torch=h.torch.__version__,
                 device='cpu', torch_threads=h.torch.get_num_threads()) == freeze['inputs']['runtime'], 'Runtime changed')
    require(h.previous.accepted.arithmetic_runtime() == freeze['inputs']['arithmetic_runtime'], 'Arithmetic changed')
    final, _ = read_report(review['parent_final_path'], PARENT_FINAL_SHA)
    from experiments.analyze_neural_material_recovery_v1 import selected_json
    raw, _ = selected_json(Path(review['parent_report_path']), PARENT_REPORT_SHA)
    require(final['classification'] == 'valid_committed_prefix_only' and
            final['materialized_report_sha256'] == PARENT_REPORT_SHA and
            all(final['completion_checks'][k] for k in ('exact_controls', 'initial_gates', 'initial_model_identities', 'settings', 'targets')),
            'Prior control/admission evidence failed')
    for key, filename in (('selected_anchor_sha256', 'commit.json'), ('selected_journal_sha256', 'journal/events.jsonl')):
        require(log.file_sha(Path(review['parent_run_root'])/filename) == final[key], 'Parent evidence changed')
    require(raw['run_id'] == Path(review['parent_run_root']).name, 'Parent run selection changed')
    selected = [r for r in raw['outcomes'] if r['arm'] == 'N1' and r['status'] != 'complete']
    require([(r['material'], r['seed']) for r in selected] ==
            [('M1',105), ('M2',101), ('M2',103), ('M2',105), ('M3',101), ('M3',103), ('M3',105)], 'Remaining matrix changed')
    require(all(r['status']=='admitted' and r['completed_updates']==0 for r in selected[1:]), 'Unstarted selection changed')
    from infrastructure.neural_material_resources_v2 import capacity
    root=Path(review['output_root']);parent=Path('/home/isavand/paper/evidence')
    require(root.is_absolute() and root.resolve()==root and root.parent==parent and
            root.name==review['run_id'] and root.name.startswith('3d2-continuation-v2-') and not root.exists(),
            'Fresh explicit continuation output required')
    capacity(parent, freeze['resource_budget']['minimum_free_bytes'])
    return h, freeze, raw, selected


def execute(review):
    h, freeze, prior, selected = admit(review)
    reference, _ = h.admit_reference()
    fit_tape, eval_tape, tape_manifest = h.inputs(reference)
    require(h.field.__name__ == 'experiments.neural_material_field_v1', 'Field implementation changed')
    class Output(RuntimeOutput): pass
    Output.freeze=freeze;Output.review=review
    root=Path(review['output_root']);root.mkdir(exist_ok=False)
    report=Tracker(dict(schema_version=1, protocol='neural-material-recovery-continuation-v1',
        run_id=review['run_id'], status='running', phase='remaining_histories', outcomes=[], failure_artifacts=[],
        parent_run_id=prior['run_id'], parent_report_sha256=PARENT_REPORT_SHA,
        reviewed_freeze_sha256=FREEZE_SHA, continuation_review_sha256=log.digest(review),
        original_run_scientifically_complete=False, replay_authorization_explicit=True,
        inherited_initial_admission=prior['initial_admission'], construction_calibrations=0)).root
    with Output(root,review['run_id']) as output:
        def persist():output.persist(report)
        persist()
        try:
            with h.controlled_runtime(report) as whole:
                # Charge the observed prior execution window against the same
                # whole-experiment limit. Replay is counted as extra physical work.
                whole.start -= review['prior_observed_seconds']
                scene=reference['scene'];obs=scene['observations']
                fit=h.MaterialBatch(scene,obs[:24],fit_tape);evaluation=h.MaterialBatch(scene,obs,eval_tape)
                shader=h.previous.original.MaterialShaderV1('A2','atan2')
                require(shader.identity == reference['shader'], 'Shader identity changed')
                models=[]
                for old in selected:
                    m,s=old['material'],old['seed'];model=h.field.make_material('N1',h.fresh_latent(s),s)
                    require(h.field.identity(model)==freeze['initial_models'][m+'/'+str(s)+'/N1']==old['initial_model'],
                            'Reconstructed model initialization differs')
                    report['construction_calibrations']+=1
                    row=resumed_row(old) if m=='M1' else copy.deepcopy(old)
                    report['outcomes'].append(row);models.append(model)
                persist()
                for model,row,old in zip(models,report['outcomes'],selected):
                    m,s=row['material'],row['seed'];report['active_history']=dict(material=m,seed=s,arm='N1')
                    stored=next(x for x in reference['materials'] if x['material']==m)
                    fine,target=h.previous.material_target(stored,m)
                    require(h.previous.oldref.sha_array(target.numpy())==row['fit_target_float32_sha256'], 'Target changed')
                    clock=h.HistoryClock(whole)
                    replay={x['update']:x for x in old.get('history',[]) if 1000<x['update']<=4270}
                    base_work=copy.deepcopy(plain(row['work']));row['segment_base_work']=base_work
                    def row_persist():
                        if m=='M1' and row.get('history'):
                            last=plain(row['history'][-1]);u=last['update']
                            if u in replay and u not in row['continuation']['replay_verified_logged_updates']:
                                for k in ('loss','loss_at_update','gradient_norm','physical_before_update',
                                          'material_before_update','gradient_groups','work'):
                                    require(last[k]==replay[u][k], 'Replay differs at update '+str(u)+' / '+k)
                                row['continuation']['replay_verified_logged_updates'].append(u)
                        persist()
                    def checkpoint(update,model,opt,current):
                        saved=h.save_checkpoint(output,m.lower()+'-'+str(s)+'-n1-u'+str(update),model,opt,update,
                            provenance=dict(freeze_sha256=FREEZE_SHA,configuration_sha256=freeze['configuration_sha256'],
                                source_sha256=freeze['sha256'],run_id=review['run_id']))
                        return dict(**saved,compact_state=h.compact_state(model,opt))
                    def failure(model,opt,current):
                        current['failure_checkpoint']=h.save_checkpoint(output,m.lower()+'-'+str(s)+'-n1-failure',
                            model,opt,current['completed_updates'],diagnostic=True)
                    loop=h.train_history
                    if m=='M1':
                        clock.wall=row['continuation']['historical_checkpoint_clock_upper_seconds']
                        def restore_factory(model):
                            opt=h.optimizer_for(model)
                            expected=dict(freeze_sha256=FREEZE_SHA,configuration_sha256=freeze['configuration_sha256'],
                                source_sha256=freeze['sha256'],run_id=prior['run_id'])
                            h.restore_checkpoint(ReadOnlyInput(review['parent_run_root']),plain(row['snapshots']['1000']),model,opt,expected)
                            require(h.compact_state(model,opt)==plain(row['snapshots']['1000']['compact_state']),
                                    'Restored weights/Adam/buffers differ')
                            row['continuation']['restored_checkpoint_exact']=True
                            return opt
                        loop=resume_loop(h,restore_factory)
                    opt=loop(model,fit,shader,target,row,clock,row_persist,checkpoint,
                             failure_state=failure,expected_optimizer=freeze['configuration']['optimizer_effective'])
                    require(row['status']=='optimized', 'History did not reach the frozen endpoint')
                    if m=='M1':require(row['continuation']['replay_verified_logged_updates']==sorted(replay), 'Incomplete replay verification')
                    with clock.active():
                        row['evaluation']=h.evaluate(model,evaluation,shader,row['work'],h.previous.reference.truth(m),fine,obs)
                        clock.check()
                    row.update(status='complete',scientific_outcome='complete_pass' if row['evaluation']['success']['all_passed'] else 'complete_fail',
                               active_wall_seconds=clock.wall,CPU_seconds=clock.cpu)
                    require(plain(row['work'])==h.work_limits('N1'), 'Logical trajectory work differs')
                    row['new_physical_work']={k:row['work'][k]-base_work[k] for k in base_work}
                    row_persist();h.verify_tapes(fit_tape,eval_tape,tape_manifest);whole.check()
                require(sum(r['new_physical_work']['optimizer_returns'] for r in report['outcomes'])==34000,
                        'Continuation update count differs')
            report.update(status='complete',phase='remaining_histories_finished');persist()
        except BaseException as exc:
            report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc));persist();raise
    return review['output_root']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',required=True);p.add_argument('--review-sha256',required=True)
    a=p.parse_args();review,_=read_report(Path(a.review),a.review_sha256);print(execute(review))


if __name__=='__main__':main()
