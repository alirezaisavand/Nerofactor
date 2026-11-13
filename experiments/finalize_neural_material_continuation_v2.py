"""Read-only admission of explicitly selected continuation and combined endpoints."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from experiments import analyze_neural_material_recovery_v1 as metrics
from experiments.analyze_neural_material_recovery_logged_v2_results import inspect_numeric_checkpoint, inventory, trajectory
from experiments.finalize_neural_material_journal_v2 import scan, atomic_json, checkpoint_evidence
from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.signatures import require

ROOT=Path(__file__).resolve().parents[1]
FREEZE_SHA='dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af'
PRIOR_SUMMARY_SHA='7a9ca7ae1e930c1a90143079f83e7376ebd00ed8d0bea514bf321686ea37084e'


def selected(record):return metrics.selected_json(Path(record['path']),record['sha256'])[0]


def finalize(selection, output):
    review=selected(selection['review']);process=selected(selection['process'])
    anchor=selected(selection['anchor']);end=selected(selection['end'])
    freeze,_=metrics.selected_json(Path(review['freeze_path']),FREEZE_SHA)
    require(all(metrics.digest(ROOT/p)==s for p,s in freeze['sha256'].items()),'Frozen scientific source changed')
    require(all(metrics.digest(ROOT/p)==s for p,s in review['continuation_sources'].items()),'Continuation source changed')
    require(process['returncode']==0 and process['invocations']==1 and process['retried'] is False and
            process['review_sha256']==selection['review']['sha256'],'Continuation process did not complete cleanly')
    require(end['anchor_sha256']==selection['anchor']['sha256'] and end['exception_type'] is None and
            end['scientific_status']==anchor['status']=='complete' and end['identity']==anchor['identity'],
            'Terminal seal or review identity mismatch')
    root=Path(review['output_root']);before=inventory(root)
    require(root==Path(selection['anchor']['path']).parent and
            anchor['identity']['run_id']==review['run_id'] and
            anchor['identity']['source_sha256']==log.digest(freeze['sha256']) and
            anchor['identity']['configuration_sha256']==freeze['configuration_sha256'], 'Run/source identity differs')
    archived_review_sha=metrics.digest(root/'execution-review.json')
    archived_review,_=metrics.selected_json(root/'execution-review.json',archived_review_sha)
    require(archived_review==review and anchor['authorization_sha256']==archived_review_sha,
            'Archived review semantics/anchor binding differs')
    raw,verified=scan(root,anchor,selection['journal']['sha256'],False)
    require(raw['run_id']==review['run_id'] and raw['protocol']=='neural-material-recovery-continuation-v1' and
            raw['status']=='complete' and raw['original_run_scientifically_complete'] is False and
            all(raw[k] is True for k in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged',
                                        'threads_restored','memory_limit_restored')),'Continuation cleanup/status failed')
    require(raw['construction_calibrations']==7,'Setup reconstruction count differs')
    prior=selected(selection['prior_summary'])
    require(selection['prior_summary']['sha256']==PRIOR_SUMMARY_SHA and
            prior['scientifically_complete'] is False,'Prior results identity/status changed')
    prior_root=Path(prior['output_root'])
    for item in prior['original_artifact_inventory']:
        require(metrics.digest(prior_root/item['path'])==item['sha256'],'Original output changed')
    original,_=metrics.selected_json(Path(review['parent_report_path']),prior['identities']['materialized_report']['sha256'])
    reference,_=metrics.selected_json(ROOT/metrics.REFERENCE,metrics.REFERENCE_SHA)
    expected=[('M1',105),('M2',101),('M2',103),('M2',105),('M3',101),('M3',103),('M3',105)]
    require([(r['material'],r['seed']) for r in raw['outcomes']]==expected,'Remaining matrix differs')
    own_catalog=dict(run_id=raw['run_id'],outcomes=[dict(row,snapshots={step:snap for step,snap in row['snapshots'].items()
        if not (row['material']=='M1' and step in ('250','1000'))}) for row in raw['outcomes']])
    catalog_bindings=checkpoint_evidence(root,own_catalog,freeze,FREEZE_SHA)
    new={};archives=[];physical_work=Counter()
    for row in raw['outcomes']:
        key=(row['material'],row['seed'],row['arm'])
        require(row['arm']=='N1' and row['status']=='complete' and row['completed_updates']==5000 and
                row['work']==freeze['configuration']['work_per_history']['N1'],'Endpoint work is incomplete')
        require(row['initial_model']==freeze['initial_models'][row['material']+'/'+str(row['seed'])+'/N1'],
                'Initial scientific model differs')
        require(set(row['snapshots'])=={'250','1000','5000'},'Snapshot inventory differs')
        difference={k:row['work'][k]-row['segment_base_work'][k] for k in row['work']}
        require(difference==row['new_physical_work'],'Physical work delta differs')
        physical_work.update(difference)
        if row['material']=='M1':
            continuation=row['continuation'];old=next(r for r in original['outcomes'] if
                (r['material'],r['seed'],r['arm'])==key)
            require(continuation['restored_checkpoint_exact'] is True and
                    continuation['replay_verified_logged_updates']==list(range(1010,4271,10)), 'Replay/restore incomplete')
            old_logs={r['update']:r for r in old['history']}
            for record in row['history']:
                if 1000<record['update']<=4270:
                    for name in ('loss','loss_at_update','gradient_norm','physical_before_update',
                                 'material_before_update','gradient_groups','work'):
                        require(record[name]==old_logs[record['update']][name],'Offline replay log mismatch')
        for step,snap in row['snapshots'].items():
            inherited=row['material']=='M1' and step in ('250','1000')
            archives.append(inspect_numeric_checkpoint(prior_root if inherited else root,snap,row,freeze,
                original['run_id'] if inherited else raw['run_id']))
        new[key]=dict(material=row['material'],seed=row['seed'],arm='N1',origin_run_id=raw['run_id'],
            status='complete',completed_updates=5000,final_fit_MSE=row['final_fit_MSE'],
            evaluation=metrics.verify_evaluation(row,reference,freeze),trajectory=trajectory(row),
            active_wall_seconds=row['active_wall_seconds'],CPU_seconds=row['CPU_seconds'],
            timing_scope='New segment CPU; active wall includes conservative historical checkpoint bound' if
                row['material']=='M1' else 'New history only',work=row['work'],new_physical_work=difference,
            final_gradient_groups=row['final_gradient_groups'],continuation=row.get('continuation'))
    require(physical_work['optimizer_returns']==physical_work['optimizer_attempts']==34000,'Additional updates differ')
    combined=[]
    for old in prior['histories']:
        key=(old['material'],old['seed'],old['arm'])
        if key in new:combined.append(new[key])
        else:
            require(old['original_status']=='complete' and old['completed_updates']==5000,'Missing prior endpoint')
            value={k:old[k] for k in ('material','seed','arm','completed_updates','final_fit_MSE','evaluation',
                   'trajectory','active_wall_seconds','CPU_seconds','work','final_gradient_groups')}
            value.update(status='complete',origin_run_id=prior['run_id']);combined.append(value)
    require(len(combined)==27 and len({(r['material'],r['seed'],r['arm']) for r in combined})==27,'Combined matrix differs')
    logical_work={k:sum(r['work'][k] for r in combined) for k in freeze['expected_total_work']}
    require(logical_work==freeze['expected_total_work'],'Combined logical work differs')
    require(before==inventory(root),'Continuation output changed during offline verification')
    stats={}
    for arm in ('D0','N0','N1'):
        rows=[r for r in combined if r['arm']==arm]
        stats[arm]=dict(complete=len(rows),physical_pass=sum(r['evaluation']['success']['all_passed'] for r in rows),
            held_out_RGB_pass=sum(r['evaluation']['success']['checks']['held_out_radiance'] for r in rows),
            per_material={m:dict(physical_pass=sum(r['material']==m and r['evaluation']['success']['all_passed'] for r in rows),
                complete=sum(r['material']==m for r in rows)) for m in ('M1','M2','M3')})
    output=Path(output)
    require(output.is_absolute() and output.resolve()==output and not output.exists() and output.parent.stat().st_uid==os.geteuid()
            and output.parent==Path('/home/isavand/paper/evidence'),'Fresh separate evidence output required')
    output.mkdir(exist_ok=False)
    path=atomic_json(output,'continuation-report.json',raw,freeze['resource_budget']['materialized_report_cap_bytes'])
    result=dict(schema_version=1,protocol='neural-material-recovery-segmented-completion-v1',
        combined_catalog_complete=True,original_run_scientifically_complete=False,continuation_verified_complete=True,
        inputs=selection,authorization=review,archived_review_sha256=archived_review_sha,
        process=process,journal_verification=verified,
        continuation_materialized_report=dict(path=str(path),sha256=metrics.digest(path)),
        inherited_control_verification=prior['verification']['controls'],checkpoint_verification=archives,
        new_checkpoint_catalog_bindings=catalog_bindings,
        new_physical_work=dict(physical_work),combined_logical_work=logical_work,
        actual_optimizer_returns_in_both_invocations=prior['work_totals']['optimizer_returns']+physical_work['optimizer_returns'],
        duplicate_replay_returns=3270,new_logical_progress_returns=30730,
        reconstruction_calibrations=7,statistics=stats,histories=combined,
        resources={k:raw[k] for k in ('wall_seconds','CPU_seconds','peak_RSS_bytes')},
        continuation_artifacts=before,offline_scientific_operations=0,
        limitations=['Two execution segments, not a single sealed original run.',
            'Restored exact update1000, replayed3270 updates; historical full4270 state unavailable.',
            'Original process termination cause is researcher-reported token limit; original exit totals remain unavailable.',
            'Architecture/encoding/redundancy are confounded; three selected materials and starts do not establish population reliability.'])
    result['analysis_source_sha256']=metrics.digest(Path(__file__).resolve())
    final=atomic_json(output,'results.json',result,freeze['resource_budget']['materialized_report_cap_bytes'])
    return dict(path=str(final),sha256=metrics.digest(final),statistics=stats,combined_catalog_complete=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',required=True);p.add_argument('--selection-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value,_=metrics.selected_json(Path(a.selection),a.selection_sha256)
    print(json.dumps(finalize(value,a.output),sort_keys=True))


if __name__=='__main__':main()
