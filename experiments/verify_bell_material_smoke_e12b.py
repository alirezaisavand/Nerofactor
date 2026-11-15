"""Read-only saved E12B smoke verification. No Torch, checkpoint or event loading."""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path

from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, PINS, baseline, digest
from infrastructure.stage_workspace import source_records
from infrastructure.verified_ray_backend import identity

REPORT_SHA = '09ff4545f0e4b1ac973003c393fe34897bd7a048a907802739a5fb6173eddbe2'
FREEZE_SHA = '54e53195734bfef17275d366f44e894a6c42ab60a8aa4b96dbb71adcddfa8d58'
COMMIT = '8247fe3a5107577660c7a211294b639deb897280'
ACTIVE = ('mx_predictor', 'my_predictor', 'rotation_predictor', 'metallic_predictor', 'kd_predictor', 'feats_network')


def record(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def finite_metadata(value):
    if isinstance(value, dict):
        return all(finite_metadata(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_metadata(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def assess_updates(r):
    require(r['optimizer_calls_completed'] == 2 and len(r['updates']) == 2 and r['cleanup_errors'] == [] and
            r['observer_lifecycle'] == dict(guards_prepared_before_run=True, method_bindings=1), 'Stop/binding/cleanup evidence incomplete')
    initial = r['initial_optimizer']
    require(initial['class'] == 'torch.optim.adam.Adam' and initial['state'] == {} and len(initial['groups']) == 1, 'Adam was not fresh')
    group = initial['groups'][0]
    require(group['lr'] == .001 and group['betas'] == [.9, .999] and group['eps'] == 1e-4 and
            group['amsgrad'] is True and group['weight_decay'] == 0, 'Optimizer settings changed')
    previous = r['initial_parameters']; states = {}; rows = []; gradients = []
    required = ['shader.'+n for n in ('forward', 'anisotropic_forward', 'shade_anisotropic_mixed',
                'sample_aniso_ggx_directions', 'compute_tangent_bitangent_flat', 'compute_radiance')]
    required += ['renderer.compute_rgb_loss', 'loss.nerf_render']
    for index, u in enumerate(r['updates']):
        require(finite_metadata(u) and u['step'] == index and u['update_index'] == index+1 and
                u['lr'] == [0., 5e-7][index] and u['forward_path_verified'] is True and
                u['backward_returned'] is True and 'loss_rgb' in u['losses'], 'Update/finite/RGB evidence failed')
        require(all(u['calls'][n] == dict(entered=1, returned=1) for n in required), 'Required native call path missing')
        require(u['parameters_before'] == previous, 'Parameter continuity failed')
        require(set(u['groups_before_clip']) == set(u['groups_after_clip']) == set(ACTIVE)|{'outer_light','inner_light'},
                'Material/shared/inactive group inventory changed')
        require(set(u['gradients_before_clip']) == set(u['gradients_after_clip']) == set(previous), 'Gradient parameter inventory changed')
        require(u['optimizer']['class'] == initial['class'] and len(u['optimizer']['groups']) == 1 and
                {k:v for k,v in u['optimizer']['groups'][0].items() if k!='lr'} == {k:v for k,v in group.items() if k!='lr'} and
                u['optimizer']['groups'][0]['lr'] == u['lr'], 'Optimizer configuration/learning rate changed')
        require(u['clip_max_norm'] == .7 and 0 < u['gradient_l2_after_clip_cpu64'] <= .70001, 'Clipping evidence failed')
        for when in ('before', 'after'):
            values = u['gradients_'+when+'_clip']; norm = math.sqrt(sum(v['l2']**2 for v in values.values() if v is not None))
            require(math.isclose(norm, u['gradient_l2_'+when+'_clip_cpu64'], rel_tol=1e-12, abs_tol=1e-15), 'Gradient norm records inconsistent')
            for name, g in u['groups_'+when+'_clip'].items():
                selected = {k: v for k, v in values.items() if k.startswith('shader_network.'+name+'.')}
                require(g['parameters'] == len(selected) and g['gradients_present'] == sum(v is not None for v in selected.values()), 'Gradient group inventory changed')
                require(math.isclose(g['gradient_l2_cpu64'], math.sqrt(sum(v['l2']**2 for v in selected.values() if v is not None)), rel_tol=1e-12, abs_tol=1e-15), 'Gradient group norm mismatch')
                require((g['expected_active'] is True and g['status'] == 'active' and g['gradient_l2_cpu64'] > 0 and
                         all(v is not None for v in selected.values())) if name in ACTIVE else
                        (name in ('outer_light', 'inner_light') and g['expected_active'] is False and
                         g['status'] == 'expected_inactive' and g['gradients_present'] == 0 and g['gradient_l2_cpu64'] == 0), 'Active/inactive gradient mismatch')
                if when == 'before':
                    gradients.append(dict(step=index, group=name, expected_active=g['expected_active'], gradient_l2=g['gradient_l2_cpu64'],
                                          gradients_present=g['gradients_present'], changed_elements=u['group_changed_elements'][name]))
        require(set(u['optimizer']['state']) == {name for name, v in u['gradients_after_clip'].items() if v is not None}, 'Unexpected Adam state ownership')
        for name, state in u['optimizer']['state'].items():
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'} and state['step'] == index+1,
                    'Adam/AMSGrad step or moment inventory mismatch')
            require(all(state[k]['shape'] == u['parameters_after'][name]['shape'] and state[k]['dtype'] == 'float32'
                        for k in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')), 'Moment shape/dtype mismatch')
        changed = sum(v['changed_elements'] for v in u['parameter_changes'].values())
        require(changed == u['changed_elements'] == sum(u['group_changed_elements'].values()), 'Displacement count mismatch')
        if index == 0:
            require(changed == 0 and u['parameters_before'] == u['parameters_after'] and
                    any(s['exp_avg']['l2'] > 0 and s['exp_avg_sq']['l2'] > 0 for s in u['optimizer']['state'].values()),
                    'Zero-LR first return did not preserve weights and advance moments')
        else:
            require(changed > 0 and all(u['group_changed_elements'][name] > 0 for name in ACTIVE) and
                    all(u['parameters_before'][name]['sha256'] != u['parameters_after'][name]['sha256']
                        for name, v in u['parameter_changes'].items() if v['changed_elements']), 'Second-return material displacement missing')
        require(u['cursor_after'] == (index+1)*512 and u['sampling']['ray_count'] == 512, 'Original ray cursor changed')
        rows.append(dict(step=index, lr=u['lr'], loss_rgb=u['losses']['loss_rgb'], aggregate_loss=u['aggregate_loss'],
                         gradient_l2=u['gradient_l2_before_clip_cpu64'], changed_elements=changed,
                         displacement_l2=math.sqrt(sum(v['l2']**2 for v in u['parameter_changes'].values())),
                         maximum_displacement=max(v['max_abs'] for v in u['parameter_changes'].values()),
                         Adam_state_parameters=len(u['optimizer']['state']), Adam_step=index+1,
                         observed_step_seconds=u['elapsed_seconds'], CUDA_peak_allocated_bytes=u['memory_after']['peak_allocated_bytes']))
        previous = u['parameters_after']; states = u['optimizer']['state']
    return rows, gradients


def verify(batch, output):
    require(batch.is_absolute() and batch.resolve() == batch and output.is_absolute() and output.resolve() == output and
            not output.exists() and batch not in output.parents, 'Explicit fresh separate analysis output required')
    rp = batch/'run/stage-run.json'; require(digest(rp) == REPORT_SHA, 'Selected native report changed')
    r = json.loads(rp.read_text()); launch = json.loads((batch/'launch.json').read_text())
    ap = Path(str(batch)+'-authorization.json'); a = json.loads(ap.read_text())
    fp = ROOT/'docs/protocols/BELL_MATERIAL_SMOKE_E12B_V1.freeze.json'; require(digest(fp) == FREEZE_SHA, 'Freeze changed')
    f = json.loads(fp.read_text())
    require(r['status'] == 'complete' and r['phase'] == 'bounded_stop_after_second_optimizer_return' and
            (batch/'exit-code.txt').read_text().strip() == '0' and r['checkpoint_input'] is None and
            r['corrected_material_smoke'] is True, 'Native completion gate failed')
    require(a['approved'] is True and a['commit'] == launch['commit'] == r['source']['declared_infrastructure_commit'] == COMMIT and
            a['freeze_sha256'] == launch['freeze_sha256'] == FREEZE_SHA and launch['authorization_sha256'] == digest(ap) and
            a['run_id'] == launch['run_id'] == r['run_id'] and launch['batch'] == a['host_output_directory'] == str(batch) and
            a['selected_gpu'] == launch['gpu'] == '0' and a['limits'] == f['limits'], 'Authorization/launch identity mismatch')
    require(a['image_id'] == r['declared_image_id'] == f['image_id'] and r['ray_backend'] == identity() and
            r['scientific_commit'] == PINS['material'] and r['source']['material_implementation'] == f['material_implementation'], 'Image/scientific identity mismatch')
    for name, sha in f['files'].items():
        require(digest(ROOT/name) == sha, 'Frozen source changed: '+name)
    science, unused = baseline('material', PINS['material']); baseline('geometry', PINS['geometry'])
    for name, sha in r['source']['infrastructure_files'].items():
        require(digest(ROOT/name) == sha, 'Native infrastructure changed')
    for item in r['imported_scientific_modules'].values():
        require(digest(science/item['path']) == item['sha256'], 'Imported scientific source changed')
    admission = Path(a['host_admission']); require(digest(admission) == f['admission']['sha256'] == a['admission_sha256'] ==
                                                r['oracle_benchmark']['admission']['sha256'], 'Oracle admission changed')
    bench = Path(a['host_benchmark']); require(source_records(bench/'datasets/GlossySynthetic/bell') ==
                                               r['input_files'] == r['oracle_benchmark']['evidence']['dataset_files'], 'Dataset changed')
    require(digest(bench/'geometry/mesh.ply') == r['mesh_identity']['sha256'] and
            digest(bench/'geometry/generation.json') == r['oracle_benchmark']['evidence']['generation_report']['sha256'] and
            digest(bench/'geometry/labels.json') == r['oracle_benchmark']['evidence']['material_labels']['sha256'], 'Oracle geometry/label provenance changed')
    for name, sha in r['output_hashes'].items():
        p = batch/'run'/name; require(p.resolve() == p and batch in p.parents and digest(p) == sha, 'Native output changed: '+name)
    require(not any(p.is_file() for p in (batch/'run/checkpoints').rglob('*')) and
            not any(p.is_file() for p in (batch/'run/renders').rglob('*')), 'Unexpected checkpoint/validation/NVS output')
    require(r['expected_absent_groups'] == {'human_light': 'disabled by explicit shader_cfg'} and
            r['effective_shader']['illumination_mode'] == 'known_env_v1' and r['effective_shader']['indirect_mode'] == 'zero_bounce_v1' and
            r['effective_renderer']['reg_mat'] is False, 'Known-light/prior configuration changed')
    rows, gradients = assess_updates(r)
    duration = (datetime.datetime.fromisoformat(r['finished_at_utc'])-datetime.datetime.fromisoformat(r['started_at_utc'])).total_seconds()
    require(0 < duration < 900, 'Recorded run time outside limit')
    output.mkdir()
    result = dict(schema_version=1, protocol='bell-e12b-native-smoke-acceptance-v1', decision='PROCEED', scope='corrected-core native two-update execution only',
        verification_status='pass', native_report=record(rp), freeze=record(fp), authorization=record(ap), admission=record(admission),
        execution_receipts=[record(batch/p) for p in ('launch.json', 'exit-code.txt', 'console.log', 'container.cid')],
        native_outputs=[record(batch/'run'/p) for p in r['output_hashes']], original_dataset_files=512,
        frozen_source_files=len(f['files']), scientific_snapshots=PINS, runtime=r['runtime'], ray_backend=r['ray_backend'],
        observer_lifecycle=r['observer_lifecycle'], cleanup_errors=r['cleanup_errors'], cleanup_interpretation='Owned iterator/progress/writer cleanup returned without errors; no independent Docker engine inspection.',
        update_summary=rows, gradient_summary=gradients, splits=r['splits'], recorded_run_seconds=duration,
        scientific_reexecution=False, new_model_calls=0, new_optimizer_returns=0, physical_recovery_established=False,
        limitations=['Hash/finite-statistic/state-continuity verification of recorded detached metadata, not independent recomputation of tensors/backpropagation.',
                    'Both recorded losses precede the first nonzero-LR update; different batches/tapes prevent interpreting their difference as learning progress.',
                    'A_original/A_estimator, checkpoint/evaluation/export/relighting interfaces and long-run stability remain unvalidated by this smoke.'])
    (output/'acceptance.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    for name, data in [('updates', rows), ('gradients', gradients)]:
        with (output/(name+'.csv')).open('x', newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(data)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    r=verify(**vars(p.parse_args()));print(json.dumps(dict(decision=r['decision'],updates=r['update_summary'])))


if __name__=='__main__': main()
