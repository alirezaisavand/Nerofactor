"""Hash-selected offline v2 results. Never imports Torch or executes a model."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path

import numpy as np

from experiments import analyze_neural_material_recovery_v1 as saved
from infrastructure.signatures import require

ROOT = Path(__file__).resolve().parents[1]
BASE = '678f3bf3230ea9e36aa97065491466a816c10310'
FREEZE_SHA = 'dbd948b23b456b8d3344156a4b5d8d1699cc9988aa938742e14bf00dbed7f2af'


def selected(path, sha):
    return saved.selected_json(Path(path), sha)


def outcome(row):
    """Keep raw status; do not turn a lost-session prefix into native censoring."""
    status, updates = row['status'], row['completed_updates']
    require(type(updates) is int and 0 <= updates <= 5000, 'Invalid update count')
    if status == 'complete':
        require(updates == 5000 and 'evaluation' in row, 'Incomplete endpoint claim')
        return 'complete'
    require('evaluation' not in row, 'Unevaluated outcome has endpoint data')
    if status == 'admitted':
        require(updates == 0, 'Admitted history has optimization updates')
        return 'unstarted'
    if status == 'optimizing':
        return 'interrupted_committed_prefix'
    require(status == 'censored' or status.startswith('failed'), 'Unknown outcome')
    return status


def trajectory(row):
    history = row.get('history', [])
    if not history:
        return None
    updates = [x['update'] for x in history]
    require(updates == sorted(set(updates)) and updates[-1] <= row['completed_updates'],
            'Invalid saved log ordering')
    losses = np.asarray([x['loss'] for x in history], np.float64)
    gradients = np.asarray([x['gradient_norm'] for x in history], np.float64)
    require(np.isfinite(losses).all() and np.isfinite(gradients).all(), 'Nonfinite saved log')
    zero = np.flatnonzero(gradients == 0)
    radii = [x['material_before_update']['centered_radius']['values'][0][0]
             for x in history if x['material_before_update'].get('centered_radius')]
    keep = {1, 10, 20, 100, 250, 1000, 2500, 5000, updates[-1]}
    return dict(logged_records=len(history), first_logged_loss=float(losses[0]),
                last_logged_loss=float(losses[-1]), minimum_logged_loss=float(losses.min()),
                maximum_logged_gradient_norm=float(gradients.max()),
                zero_gradient_logged_records=int(len(zero)),
                first_zero_gradient_logged_update=updates[int(zero[0])] if len(zero) else None,
                all_later_logged_gradients_zero=bool(len(zero) and np.all(gradients[zero[0]:] == 0)),
                centered_radius_range=[min(radii), max(radii)] if radii else None,
                last_logged_active_seconds=history[-1]['active_wall_seconds'],
                last_logged_material=history[-1]['material_before_update'],
                last_logged_gradient_groups=history[-1]['gradient_groups'],
                records=[{k: x[k] for k in ('update', 'loss_at_update', 'loss',
                         'gradient_norm', 'physical_before_update', 'active_wall_seconds')}
                         for x in history if x['update'] in keep],
                scope='Saved pre-update log samples; no reconstruction of unlogged updates')


def inspect_numeric_checkpoint(root, selection, row, freeze, run_id):
    """Inspect numeric archives with no deserialization of objects/model state."""
    path = saved.child(root, selection['manifest'])
    meta, identity = selected(path, selection['manifest_sha256'])
    require(meta['execution_provenance'] == dict(freeze_sha256=FREEZE_SHA,
            configuration_sha256=freeze['configuration_sha256'], source_sha256=freeze['sha256'],
            run_id=run_id), 'Checkpoint provenance mismatch')
    require(meta['arm'] == row['arm'] and meta['source'] == row['initial_model']['source'] and
            meta['model_configuration'] == row['initial_model']['configuration'] and
            meta['completed_updates'] == selection['completed_updates'] and
            meta['optimizer_effective'] == freeze['configuration']['optimizer_effective'],
            'Checkpoint model/optimizer mismatch')
    archive = saved.child(path.parent, meta['archive'])
    require(saved.digest(archive) == meta['archive_sha256'] and
            archive.stat().st_size == meta['archive_bytes'], 'Numeric archive identity mismatch')
    entries = meta['entries']
    require(len({e['key'] for e in entries}) == len(entries), 'Duplicate array key')
    with np.load(archive, allow_pickle=False) as arrays:
        require(set(arrays.files) == {e['key'] for e in entries} | {'numpy_rng'}, 'Array inventory mismatch')
        for e in entries:
            a = arrays[e['key']]
            require(a.dtype.kind != 'O' and list(a.shape) == e['shape'] and
                    'torch.' + str(a.dtype) == e['dtype'] and e['device'] == 'cpu' and
                    saved.array_sha(a) == e['sha256'] and
                    bool(np.isfinite(a).all()) == e['finite'], 'Numeric tensor mismatch')
            require(e['finite'], 'Unexpected nonfinite snapshot')
            if e['role'] == 'adam':
                if e['name'].endswith('/step'):
                    require(a.shape == () and float(a) == meta['completed_updates'], 'Adam counter mismatch')
                if e['name'].endswith('/exp_avg_sq'):
                    require(bool((a >= 0).all()), 'Negative Adam second moment')
        rng = arrays['numpy_rng']
        require(rng.dtype == np.uint32 and rng.shape == (624,) and
                saved.array_sha(rng) == meta['numpy_rng']['array_sha256'], 'RNG array mismatch')
    return dict(manifest=identity, archive=dict(path=str(archive), sha256=meta['archive_sha256'],
                bytes=meta['archive_bytes']), completed_updates=meta['completed_updates'],
                arrays_verified=len(entries) + 1, allow_pickle=False)


def inventory(root):
    result = []
    for path in sorted(Path(root).rglob('*')):
        require(not path.is_symlink(), 'Evidence tree contains a symlink')
        if path.is_file():
            st = path.stat()
            result.append(dict(path=str(path.relative_to(root)), bytes=st.st_size,
                               allocated_bytes=st.st_blocks * 512, sha256=saved.digest(path)))
    return result


def analyze(selection_path, selection_sha, final_path, final_sha):
    selection, selection_id = selected(selection_path, selection_sha)
    final, final_id = selected(final_path, final_sha)
    a = selection['artifacts']
    require(a['freeze']['sha256'] == FREEZE_SHA, 'Wrong v2 freeze')
    freeze, freeze_id = selected(a['freeze']['path'], FREEZE_SHA)
    require(all(saved.digest(ROOT / name) == sha for name, sha in freeze['sha256'].items()),
            'Frozen source changed')
    run = Path(a['anchor']['path']).parent
    before = inventory(run)
    for name in ('anchor', 'journal', 'authorization', 'protocol', 'console'):
        require(saved.digest(a[name]['path']) == a[name]['sha256'], 'Selected input changed: ' + name)
    require(final['selected_anchor_sha256'] == a['anchor']['sha256'] and
            final['selected_journal_sha256'] == a['journal']['sha256'] and
            final['freeze_sha256'] == FREEZE_SHA and final['identity']['run_id'] == run.name,
            'Finalization identity mismatch')
    require(final['classification'] == 'valid_committed_prefix_only' and
            final['scientifically_complete'] is False and final['original_scientific_status'] == 'running' and
            final['selected_execution_end_sha256'] is None and not (run / 'execution-end.json').exists(),
            'This analyzer requires the selected incomplete prefix, not a complete claim')
    raw, report_id = selected(Path(final_path).parent / 'report.json', final['materialized_report_sha256'])
    require(raw['repository'] == dict(HEAD=BASE, branch='feat/neural-material-logged-v2-freeze',
            tracked_status=[]) and raw['run_id'] == run.name and raw['reviewed_freeze_sha256'] == FREEZE_SHA,
            'Scientific source/run identity mismatch')
    require(all(final['completion_checks'][k] for k in ('authorization_record', 'exact_controls',
            'initial_gates', 'initial_model_identities', 'matrix', 'settings', 'targets', 'not_synthetic')),
            'Mandatory prefix admission failed')
    controller = Path(selection_path).parent
    preflight = json.loads((controller / 'preflight.json').read_text())
    start = json.loads((controller / 'process-started.json').read_text())
    claim = json.loads((controller / 'invocation-claim.json').read_text())
    require(preflight['status'] == 'passed' and preflight['scientific_model_constructions'] ==
            preflight['shader_calls'] == preflight['optimizer_calls'] == 0 and
            preflight['prior_invocation_claim_exists'] is False and preflight['run_root_exists'] is False,
            'Non-scientific preflight failed')
    require(start['invocations'] == selection['scientific_invocations'] == 1 and selection['retry'] is False and
            start['command'][:4] == ['timeout', '--signal=TERM', '--kill-after=10s', '28800s'] and
            preflight['source_commit'] == BASE and preflight['run_id'] == raw['run_id'] and
            preflight['authorization_sha256'] == a['authorization']['sha256'] and
            saved.digest(controller / 'controller.py') == preflight['controller_sha256'] and
            saved.digest(controller / 'preflight.py') == preflight['preflight_script_sha256'],
            'Controller invocation/preflight identity mismatch')
    require(not (controller / 'process-outcome.json').exists(), 'Inspect new process outcome before analysis')
    reference, reference_id = selected(ROOT / saved.REFERENCE, saved.REFERENCE_SHA)
    controls = saved.verify_controls(raw['outcomes'], reference)
    rows, checkpoints = [], []
    counters = Counter()
    expected = [(m, s, arm) for m in freeze['configuration']['materials']
                for s in freeze['configuration']['seeds'] for arm in freeze['configuration']['arms']]
    require([(r['material'], r['seed'], r['arm']) for r in raw['outcomes']] == expected, 'Matrix changed')
    for row in raw['outcomes']:
        state = outcome(row)
        work, limit = row['work'], freeze['configuration']['work_per_history'][row['arm']]
        require(set(work) == set(limit) and all(type(v) is int and 0 <= v <= limit[k] for k, v in work.items()),
                'Invalid work accounting')
        require(work['optimizer_attempts'] == work['optimizer_returns'] == row['completed_updates'] and
                work['shader_attempts'] == work['shader_completed'] == work['fit_forwards'] + work['evaluation_forwards'] and
                work['field_attempts'] == work['field_completed'] == work['shader_completed'] + 2 and
                work['reverse_attempts'] == work['reverse_completed'] == row['completed_updates'] + 1 + work['terminal_forwards'] and
                work['attempted_points'] == work['completed_points'] == 96 * work['fit_forwards'] + 336 * work['evaluation_forwards'],
                'Committed work counter mismatch')
        if state == 'complete':
            require(work == limit, 'Complete history has incomplete work')
        counters.update(work)
        evaluation = saved.verify_evaluation(row, reference, freeze) if state == 'complete' else None
        snapshots = []
        for _, snap in sorted(row.get('snapshots', {}).items(), key=lambda kv: int(kv[0])):
            checkpoints.append(inspect_numeric_checkpoint(run, snap, row, freeze, raw['run_id']))
            snapshots.append({k: v for k, v in snap.items() if k != 'compact_state'})
        rows.append(dict(material=row['material'], seed=row['seed'], arm=row['arm'],
            original_status=row['status'], observed_outcome=state, completed_updates=row['completed_updates'],
            final_fit_MSE=row.get('final_fit_MSE') if evaluation else None, evaluation=evaluation,
            physical_recovery_pass=evaluation['success']['all_passed'] if evaluation else None,
            active_wall_seconds=row.get('active_wall_seconds'), CPU_seconds=row.get('CPU_seconds'),
            reason=row.get('termination') if evaluation else 'Execution session lost; no terminal seal or endpoint' if
                state == 'interrupted_committed_prefix' else 'Admitted initially; optimization not started',
            trajectory_sha256=row.get('trajectory_sha256'), trajectory=trajectory(row),
            final_gradient_groups=row.get('final_gradient_groups'), snapshots=snapshots, work=work))
    require(before == inventory(run), 'Original evidence changed during offline analysis')
    stats = {}
    for arm in ('D0', 'N0', 'N1'):
        rr = [r for r in rows if r['arm'] == arm]
        done = [r for r in rr if r['evaluation'] is not None]
        stats[arm] = dict(planned=len(rr), outcomes=dict(Counter(r['observed_outcome'] for r in rr)),
            endpoint_physical_pass=sum(r['physical_recovery_pass'] for r in done), endpoint_denominator=len(done),
            held_out_RGB_pass=sum(r['evaluation']['success']['checks']['held_out_radiance'] for r in done),
            per_material={m: dict(outcomes=dict(Counter(r['observed_outcome'] for r in rr if r['material'] == m)),
                physical_pass=sum(r['physical_recovery_pass'] for r in done if r['material'] == m),
                endpoint_denominator=sum(r['material'] == m for r in done)) for m in ('M1', 'M2', 'M3')})
    index = {(r['material'], r['seed'], r['arm']): r for r in rows}
    pairs = []
    for r in rows:
        if r['arm'] == 'N1' and r['evaluation'] is not None:
            d = index[r['material'], r['seed'], 'D0']
            pairs.append(dict(material=r['material'], seed=r['seed'],
                D0=d['evaluation'], N1=r['evaluation'],
                active_wall_ratio_N1_over_D0=r['active_wall_seconds'] / d['active_wall_seconds'],
                scope='Available paired endpoints only; six N1 starts unexecuted and one interrupted'))
    started = datetime.datetime.fromisoformat(start['started_UTC'])
    committed = datetime.datetime.fromisoformat(a['anchor']['mtime_UTC'])
    return dict(schema_version=1, protocol='neural-material-recovery-logged-v2-results-v1',
        classification='incomplete_execution_valid_committed_prefix', scientifically_complete=False,
        source_commit=BASE, run_id=raw['run_id'], output_root=str(run),
        analysis_source_sha256=saved.digest(Path(__file__).resolve()),
        identities=dict(selection=selection_id, freeze=freeze_id, authorization=a['authorization'],
            finalization=final_id, materialized_report=report_id, reference=reference_id,
            anchor=a['anchor'], journal=a['journal'], seal=None,
            source_inventory_sha256=final['source_inventory_sha256']),
        process=dict(actual_exit_status=None, actual_end_time=None, actual_duration_seconds=None,
            interruption_cause='Unresolved: execution session unavailable, no outcome or terminal seal',
            started_UTC=start['started_UTC'], last_committed_UTC=a['anchor']['mtime_UTC'],
            observed_start_to_last_commit_seconds=(committed - started).total_seconds(),
            command=start['command'], invocations=1, retried=False, claim=claim),
        runtime=preflight['CPU_runtime'], environment=preflight['environment'],
        settings=raw['settings'], optimizer=freeze['configuration']['optimizer_effective'],
        success_thresholds=freeze['configuration']['success'],
        verification=dict(frozen_source_files=len(freeze['sha256']), finalizer_scan=final['scan'],
            finalizer_checks=final['completion_checks'], controls=controls,
            endpoint_metrics_offline_recomputed=len([r for r in rows if r['evaluation']]),
            numeric_checkpoints=checkpoints, original_evidence_unchanged=True,
            scientific_operations_during_analysis=0, original_raw_status=raw['status']),
        outcomes=dict(Counter(r['observed_outcome'] for r in rows)), statistics=stats,
        work_totals=dict(counters), histories=rows, available_D0_N1_pairs=pairs,
        resources=dict(completed_history_active_wall_seconds=sum(r['active_wall_seconds'] or 0 for r in rows),
            completed_history_CPU_seconds=sum(r['CPU_seconds'] or 0 for r in rows),
            incomplete_last_logged_active_seconds=max((r['trajectory']['last_logged_active_seconds'] for r in rows
                if r['observed_outcome'] == 'interrupted_committed_prefix' and r['trajectory']), default=None),
            process_CPU_seconds=None, peak_RSS_bytes=None, terminal_resource_totals_available=False,
            original_output_bytes=sum(x['bytes'] for x in before),
            original_output_allocated_bytes=sum(x['allocated_bytes'] for x in before),
            original_output_files=len(before)),
        original_artifact_inventory=before,
        controller_inventory=[f for f in inventory(controller)
                              if not f['path'].startswith('offline-analysis')],
        limitations=['Not a complete 27-history experiment; no retry/resume authorized or performed.',
            'Endpoint failures differ from execution failures; zero finite gradients can accompany optimizer returns.',
            'Partial N1/105 training logs are not an endpoint or a physical recovery result.',
            'No N1 endpoint for M2/M3; no aggregate representation-reliability conclusion.',
            'Architecture, orientation encoding and redundant parameterization are confounded; not depth alone.',
            'Process exit, shutdown cleanup, whole-run CPU and peak RSS are unavailable.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'selection-sha256', 'final', 'final-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    out = Path(a.output)
    require(out.is_absolute() and out.resolve() == out and not out.exists() and
            out.parent.is_dir() and out.parent.stat().st_uid == os.geteuid(), 'Fresh owned output required')
    require(Path(a.selection).parent not in out.parents and Path(a.final).parent not in out.parents,
            'Do not write into selected original evidence')
    report = analyze(a.selection, a.selection_sha256, a.final, a.final_sha256)
    for preserved in (Path(report['output_root']), ROOT.parent / 'data', ROOT.parent / 'final_results',
                      Path(report['identities']['freeze']['path']).parents[2]):
        require(out != preserved and preserved not in out.parents,
                'Output overlaps original scientific evidence/source')
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(out), sha256=saved.digest(out), outcomes=report['outcomes'],
                         statistics=report['statistics'], scientifically_complete=False), sort_keys=True))


if __name__ == '__main__':
    main()
