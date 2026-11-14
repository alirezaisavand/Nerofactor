"""Offline E7 verification/figures. Reuses accepted pure NumPy E6 analysis helpers."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
HELPER = ROOT/'docs/results/material_optimizer_e6_v1/analyze_e6.py'
assert HELPER.read_bytes() == subprocess.check_output(['git', 'show', 'e7ae53cf991f8fe597547d8757612be1cf089c0a:'+str(HELPER.relative_to(ROOT))], cwd=ROOT)
spec = importlib.util.spec_from_file_location('accepted_e6_offline_helpers', HELPER)
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
# The accepted helper installs a process-wide import blocker for torch/experiments.
import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure import material_diagnostics_e3r_v1 as diagnostic
from infrastructure.signatures import read_report, require

RUN = 'material-e7-b44cedc3bbcd443f8fc47976ebdd1c18'
FREEZE = 'd250f4ef5ef55f2ccdce06e85a7461823b2a30fd5155e7f7a017e25425e251fe'
REPORT = '917058aeefa2e15b497e56869cc96fd461415c75e37d6f34f82784034a0431e8'
SEAL = 'aa1048bfa74716606a119aa1c4ae8e59508d29ac86bf6dfe991ee1ffd7f2d42e'
RECEIPT = '82ddbfb2b6a54cf57a959710fcc41317154cae9acbe1148af18357fd6c65c2d5'
COMMIT = 'c1c5bff12fb3b0abb070748458133ba60b383a0a'
sha = helpers.sha
identity = helpers.identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-parent', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); parent = args.evidence_parent; out = args.output_dir; raw = parent/RUN
    assert raw.is_absolute() and raw.resolve() == raw and out.is_absolute() and out.resolve() == out and out.parent == parent and not out.exists()
    plan, _ = read_report(ROOT/'docs/protocols/HYBRID_MATERIAL_E7_V1.freeze.json', FREEZE)
    for name, h in plan['source_sha256'].items(): assert sha(ROOT/name) == h
    for name, h in plan['input_sha256'].items(): assert sha(name) == h
    report, _ = read_report(raw/'result.json', REPORT); seal, _ = read_report(raw/'finalization/seal.json', SEAL)
    receipt, _ = read_report(parent/(RUN+'.process.json'), RECEIPT)
    assert report['all_initial_gates_passed'] and report['cleanup_verified'] and report['terminal_integrity_verified'] and report['scientific_work_completed']
    assert seal['report']['sha256'] == REPORT and seal['outcome'] == 'complete' and receipt['process_exit_code'] == receipt['retry_attempts'] == 0
    for key in ('authorization', 'launch', 'console'): assert identity(Path(receipt[key]['path'])) == receipt[key]
    launch = json.loads(Path(receipt['launch']['path']).read_text())
    assert launch['commit'] == COMMIT and launch['freeze_sha256'] == FREEZE and launch['foreground_supervisor']
    assert report['attempts'] == report['completed'] == plan['limits']
    authorization = json.loads(Path(receipt['authorization']['path']).read_text())
    assert authorization['external_watchdog'] == dict(wall_seconds=1800, kill_grace_seconds=10)
    numeric = helpers.numeric_metrics(plan)
    reference = json.loads((ROOT/'docs/evidence/MULTI_MATERIAL_RECOVERY_V1.json').read_text())
    observations = reference['scene']['observations']; endpoints = []; gradients = []; spatial = []; trajectories = {}
    numeric_count = 0
    # Check every numerical manifest, not only the displayed endpoints.
    for path in sorted(raw.rglob('*.npz.json')):
        obs.read_numeric(raw, dict(path=str(path.relative_to(raw)), sha256=sha(path))); numeric_count += 1
    for case, frozen in zip(report['cases'], plan['cases']):
        assert case['case'] == frozen['case']
        info = next(m for m in reference['materials'] if m['material'] == frozen['material'])
        fine = np.asarray(info['targets']['fine'], np.float64); truth = np.asarray(info['truth'], np.float64)
        initial = obs.read_numeric(frozen['source_run'], frozen['checkpoint'])
        baseline = obs.read_numeric(raw, case['baseline_spatial'])
        for arm in case['arms']:
            a = arm['arm']; saved = arm['evaluation']; arrays = obs.read_numeric(raw, saved['arrays']); p = arrays['physical']
            errors = diagnostic.recovery_errors(p, truth); assert errors == saved['recovery_errors']
            metrics = {}
            for split in ('fit', 'held_out', 'relight'):
                mask = [o['split'] == split for o in observations]
                values = arrays['predictions'][mask]; targets = fine[mask]
                metrics[split] = dict(errors={k: numeric['radiance_errors'](values[:, i], targets[:, i]) for i, k in enumerate(('diffuse', 'specular', 'total'))}, **helpers.components(values, targets))
                assert metrics[split] == saved['metrics'][split]
            success = numeric['assess_success'](errors, {s: m['errors'] for s, m in metrics.items()}); assert success == saved['success']
            final = obs.read_numeric(raw, arm['checkpoint']['selection'])
            assert arm['restoration']['exact'] and arm['restoration']['Adam_step'] == 100
            if a == 'control':
                path = raw/arm['trajectory']['path']; assert sha(path) == arm['trajectory']['sha256']
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                assert len(rows) == frozen['control_updates'] and [r['absolute_Adam_step'] for r in rows] == list(range(101, 101+len(rows)))
                assert len(final) == 350 and all(float(v) == 100+len(rows) for k, v in final.items() if k.endswith('/step'))
                rows.append(arm['terminal']); trajectories[case['case']] = rows
                cost = arm['new_renderer_cost']; inherited = 0; loss = arm['terminal']['loss']
            else:
                assert arm['accepted'] and arm['realization']['passed'] and arm['realization']['nonsaturated'] and arm['teacher_loss_agreement']
                allowed = {'parameter/'+n for n in arm['projected_bias_names']}
                assert len(allowed) == 5 and len(arm['bias_delta']) == 8 and len(final) == 98
                assert all(np.array_equal(value, initial[name]) for name, value in final.items() if name not in allowed)
                assert not any(k.startswith('adam/') for k in final)
                rows = [arm['trial']]; cost = arm['including_inherited_cost']; inherited = arm['teacher']['inherited_cost']; loss = arm['fitting_loss']
            assert all(np.isfinite(r['global_gradient_L2']) and r['global_gradient_L2'] > 0 and r['exact_sigmoid_boundaries'] == 0 for r in rows)
            assert all(g['finite'] and g['missing'] == 0 and g['nonzero'] > 0 for r in rows for g in r['gradient_groups'].values())
            gradients.append(dict(case=case['case'], arm=a, observed_backward_rows=len(rows), exact_boundary_channels=0,
                global_gradient_L2_min=min(r['global_gradient_L2'] for r in rows), global_gradient_L2_max=max(r['global_gradient_L2'] for r in rows),
                heads={n:dict(logit_gradient_L2_min=min(r['heads'][n]['logit_gradient_L2'] for r in rows),
                              sigmoid_gradient_L2_min=min(r['heads'][n]['sigmoid_gradient_L2'] for r in rows)) for n in rows[0]['heads']}))
            ext = obs.read_numeric(raw, arm['spatial'])
            spatial.append(dict(case=case['case'], arm=a,
                width_change_Linf=float(abs(ext['widths']-baseline['widths']).max()),
                albedo_change_Linf=float(abs(ext['albedo']-baseline['albedo']).max()),
                metallic_change_Linf=float(abs(ext['metallic']-baseline['metallic']).max()),
                encoding_change_Linf=float(abs(ext['encoding']-baseline['encoding']).max()),
                raw_encoder_radius_min=float(np.linalg.norm(2*ext['rotation_raw']-1, axis=1).min()),
                width_spread_Linf=float(np.ptp(ext['widths'], axis=0).max()),
                limitation='Three-point underlying predictor extension, not deployed constant N1 or spatial ground truth'))
            probe = obs.read_numeric(raw, arm.get('proposed_probe', arm.get('probe')))
            assert np.array_equal(probe['physical'], np.broadcast_to(probe['physical'][:1], probe['physical'].shape))
            correction = None
            if a != 'control':
                target = np.asarray(arm['teacher']['physical'], np.float64)
                before = np.asarray(arm['initial']['physical'], np.float64)
                idx = [0, 1, 3, 4, 5, 6]
                signal = obs.read_numeric(raw, arm['trial']['signals'])
                radius = np.linalg.norm(2*signal['rotation_predictor.sigmoid']-1, axis=-1)
                correction = dict(teacher_nonangle_displacement_L2=float(np.linalg.norm(target[idx]-before[idx])),
                    realized_nonangle_displacement_L2=float(np.linalg.norm(p[idx]-before[idx])),
                    bias_displacement_L2=float(np.linalg.norm(arm['bias_delta'])), actual_anchor_encoder_radius=radius.tolist())
            endpoints.append(dict(case=case['case'], arm=a, physical=p.tolist(), recovery_errors=errors, metrics=metrics, success=success, correction=correction,
                fitting_MSE=loss, new_C=arm['new_renderer_cost'], inherited_C=inherited, total_C=cost,
                realization=arm.get('realization'), bias_delta=arm.get('bias_delta'), arm_seconds=arm['arm_seconds'],
                construction_restore_seconds=arm['construction_restore_seconds']))
    diagnosis = []
    fine = np.asarray(next(m for m in reference['materials'] if m['material'] == 'M3')['targets']['fine'], np.float64)[:24]
    for d in report['diagnostic']:
        arrays = obs.read_numeric(raw, d['arrays']); component = helpers.components(arrays['predictions'], fine); assert component == d['errors']
        diagnosis.append(dict(point=d['point'], tapes=d['tapes'], fitting_MSE_float32=d['loss'], reused=d['reused'], **component))
    assert len(diagnosis) == 9 and sum(d['reused'] for d in diagnosis) == 1
    native = [n for c in report['cases'] for a in c['arms'] for n in (a['native'], a['external_native'])]
    assert sum(n['shader_completed'] for n in native) == 293 and sum(n['reverse_completed'] for n in native) == 283
    assert sum(n['optimizer_returns'] for n in native)-600 == 277
    assert sum(n['calibration_forwards'] for n in native) == 6
    assert not any(n == 'torch' or n.startswith(('torch.', 'experiments.')) for n in sys.modules)
    out.mkdir()
    def dump(name, value):
        with (out/name).open('x') as stream: json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    comparison = dict(schema_version=1, protocol='hybrid-neural-physical-e7-v1-offline', run=RUN, execution_commit=COMMIT,
        decision='FIX AND PROCEED', completed=report['completed'], sources_verified=len(plan['source_sha256']), inputs_verified=len(plan['input_sha256']),
        numerical_archives_verified=numeric_count, shader_queries=0, optimizer_updates=0, neural_constructions=0,
        endpoints=endpoints, diagnostic=diagnosis, gradients=gradients, spatial=spatial,
        control_progress={case: dict(initial_Q=diagnostic.recovery_errors(np.asarray(rows[0]['physical'],np.float32),
            np.asarray(next(m for m in reference['materials'] if m['material']==case.split('/')[0])['truth']))['Q_relative_frobenius'],
            initial_loss=rows[0]['loss'],final_loss=rows[-1]['loss']) for case,rows in trajectories.items()},
        external_wall_seconds=receipt['wall_seconds'], scientific_wall_seconds=report['wall_seconds'], peak_RSS_bytes=receipt['peak_RSS_bytes'])
    dump('comparison.json', comparison)
    with (out/'endpoints.csv').open('x', newline='') as stream:
        fields = ('case','arm','fitting_MSE','Q_relative','orientation_degrees','max_albedo_error','metallic_error','held_out_relative_L2','relight_relative_L2','full_recovery','new_C','inherited_C','total_C','physical')
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for e in endpoints:
            er=e['recovery_errors'];writer.writerow(dict(case=e['case'],arm=e['arm'],fitting_MSE=e['fitting_MSE'],Q_relative=er['Q_relative_frobenius'],orientation_degrees=er['orientation_degrees'],
                max_albedo_error=max(er['albedo_absolute_errors']),metallic_error=er['metallic_absolute_error'],held_out_relative_L2=e['metrics']['held_out']['errors']['total']['relative_L2'],
                relight_relative_L2=e['metrics']['relight']['errors']['total']['relative_L2'],full_recovery=e['success']['all_passed'],new_C=e['new_C'],inherited_C=e['inherited_C'],total_C=e['total_C'],physical=json.dumps(e['physical'])))
    with tempfile.TemporaryDirectory(prefix='nerofactor-e7-mpl-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels=['N1 continuation','LM → N1','Adam → N1']; colors=['#777777','#2878b5','#e07b39']
        fig, axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        for j,case in enumerate(('M2/103','M3/101')):
            rows=[e for e in endpoints if e['case']==case]
            for i,(title,values,threshold) in enumerate((('Relative Q error',[e['recovery_errors']['Q_relative_frobenius'] for e in rows],.05),('Held-out RGB relative L2',[e['metrics']['held_out']['errors']['total']['relative_L2'] for e in rows],.01))):
                ax=axes[i,j];ax.bar(labels,values,color=colors);ax.axhline(threshold,color='black',ls='--',lw=1);ax.set_title(case+' — '+title);ax.tick_params(axis='x',labelsize=8)
                for k,v in enumerate(values):ax.text(k,v,f'{v:.4f}',ha='center',va='bottom',fontsize=9)
        fig.savefig(out/'recovery_endpoints.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True);names=['accepted','independent_1','independent_2'];x=np.arange(3)
        for i,point in enumerate(('truth','LM','Adam')):
            rows=[next(d for d in diagnosis if d['tapes']==n and d['point']==point) for n in names];offset=(i-1)*.24
            for ax,values in zip(axes,([r['fitting_MSE_float32'] for r in rows],[sum(r['component_MSE'][:2]) for r in rows],[r['normalized_residual_inner_product'] for r in rows])):ax.bar(x+offset,values,width=.24,label=point)
        for ax in axes:ax.set_xticks(x);ax.set_xticklabels(['Original','Independent 1','Independent 2'],fontsize=8);ax.legend(fontsize=8)
        axes[0].set_title('M3 total-RGB fitting MSE');axes[0].set_yscale('log');axes[1].set_title('M3 diffuse + specular error MSE');axes[1].set_yscale('log');axes[2].set_title('Diffuse/specular residual cosine');axes[2].axhline(0,color='black',lw=.7)
        fig.savefig(out/'m3_sampling_compensation.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        for j,(case,rows) in enumerate(trajectories.items()):
            x=[r['state'] for r in rows];axes[0,j].plot(x,[r['loss'] for r in rows]);axes[1,j].plot(x,[r['global_gradient_L2'] for r in rows])
            axes[0,j].set_title(case+' original N1 continuation');axes[0,j].set_ylabel('Fitting MSE');axes[1,j].set_ylabel('Full-network gradient L2')
            for ax in axes[:,j]:ax.set_yscale('log');ax.set_xlabel('Absolute Adam state')
        fig.savefig(out/'neural_continuation.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        hybrids=[e for e in endpoints if e['realization']];axes[0].bar([e['case']+'\n'+e['arm'].replace('hybrid_','') for e in hybrids],[e['realization']['physical_max_error'] for e in hybrids]);axes[0].axhline(2e-6,color='black',ls='--',label='Admission limit');axes[0].set_yscale('log');axes[0].set_title('Anchor non-angle projection max error');axes[0].legend(fontsize=8)
        axes[1].bar([s['case']+'\n'+s['arm'].replace('hybrid_','') for s in spatial],[max(s['width_change_Linf'],s['albedo_change_Linf'],s['metallic_change_Linf']) for s in spatial]);axes[1].set_title('Underlying predictor: other-position drift');axes[1].set_ylabel('Max change in width / albedo / metallic');axes[1].tick_params(axis='x',labelsize=8)
        fig.savefig(out/'projection_spatial_drift.png',dpi=170);plt.close(fig)
    raw_files=[identity(p) for p in sorted(raw.rglob('*')) if p.is_file()]
    sidecars=[identity(parent/(RUN+s)) for s in ('.authorization.json','.launch.json','.console.log','.process.json')]
    dump('evidence_manifest.json',dict(schema_version=1,run=RUN,raw_files=raw_files,sidecars=sidecars,raw_evidence_copied_to_Git=False,
        report_sha256=REPORT,seal_sha256=SEAL,receipt_sha256=RECEIPT,freeze_sha256=FREEZE,execution_commit=COMMIT,
        analysis_source=identity(Path(__file__).resolve()),accepted_offline_helper=identity(HELPER)))
    dump('analysis_integrity.json',dict(schema_version=1,analysis_only=True,scientific_imports_blocked=True,scientific_queries=0,
        numerical_archives_verified=numeric_count,sources_verified=len(plan['source_sha256']),inputs_verified=len(plan['input_sha256']),
        files=[identity(p) for p in sorted(out.iterdir()) if p.is_file()]))
    print(json.dumps(dict(status='offline_analysis_complete',output=str(out),endpoints=6,scientific_queries=0)))


if __name__=='__main__':main()
