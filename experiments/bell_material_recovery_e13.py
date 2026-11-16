#!/usr/bin/env python3
"""One isolated E.13 arm. Without --execute: metadata/hash checks only."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, baseline, digest, activate, imported_sources
from infrastructure.stage_execution import plan, native_guards, trainer_io
from infrastructure.material_variants import admit_oracle, identity, install
from infrastructure.material_recovery_e13 import (PROTOCOL, ARMS, RETURNS, RecoveryProbe,
    EndpointTrace, strict_roundtrip, write_json)

IMAGE = 'sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331'
ADMISSION = 'f89fe72d13e9f6d05067b012ff4575852eef26ba35b5b3c072ac236e51e74605'
SUFFIXES = ('original', 'estimator', 'core')
FREEZE = ROOT/'docs/protocols/BELL_THREE_ARM_RECOVERY_E13_V2.freeze.json'


def verify_freeze(expected):
    require(digest(FREEZE) == expected, 'E.13 freeze identity mismatch')
    freeze = json.loads(FREEZE.read_text())
    require(freeze['protocol'] == PROTOCOL and freeze['image_id'] == IMAGE and
            all(digest(ROOT/p) == h for p, h in freeze['files'].items()), 'Frozen source mismatch')
    baseline('material', PINS['material'])
    return freeze


def verify_inputs(root, freeze):
    for rel, item in freeze['inputs'].items():
        p = root/rel
        require(p.resolve() == p and p.is_file() and p.stat().st_size == item['bytes'] and digest(p) == item['sha256'],
                'Consumed oracle input mismatch: '+rel)


def prepared(args, freeze):
    require(args.variant in ARMS and digest(args.admission) == ADMISSION, 'Foreign variant/admission')
    p = plan('material', PINS['material'], 'C', args.inputs/'datasets', args.thesis, args.output,
             ROOT/('configs/experiments/material_a_bell_'+SUFFIXES[ARMS.index(args.variant)]+'_known_v1.yaml'), ROOT,
             {'geometry': args.inputs/'geometry', 'admission': args.admission.parent},
             material_variant=args.variant, oracle_benchmark=True)
    proof = admit_oracle(p, args.admission)
    verify_inputs(args.inputs, freeze)
    require(p['configuration']['total_step'] == RETURNS, 'Foreign update budget')
    return p, proof


def execute(args, freeze, p, proof):
    from infrastructure.runs import RunStore
    from infrastructure.stage_workspace import prepare_view, source_records
    from infrastructure.bell_inference import runtime_environment, rng_identity
    from infrastructure.signatures import write_report
    native_guards(p, IMAGE)
    require(os.statvfs(str(args.inputs/'references')).f_flag & os.ST_RDONLY, 'Read-only references required')
    auth = json.loads(args.authorization.read_text())
    require(digest(args.authorization) == args.authorization_sha256 and auth.get('approved') is True and
            auth.get('protocol') == PROTOCOL and auth.get('commit') == args.commit and
            auth.get('freeze_sha256') == args.freeze_sha256 and auth.get('image_id') == IMAGE and
            auth.get('admission_sha256') == ADMISSION and auth.get('limits') == freeze['limits'] and
            auth.get('run_ids', {}).get(args.variant) == args.run_id,
            'Exact E.13 three-arm execution authorization required')
    expected = None
    if args.variant != ARMS[0]:
        require(args.initial is not None and digest(args.initial) == args.initial_sha256 and
                os.statvfs(str(args.initial)).f_flag & os.ST_RDONLY, 'Read-only hash-bound first-arm initialization required')
        expected = json.loads(args.initial.read_text())
    else:
        require(args.initial is None and args.initial_sha256 is None, 'First arm cannot substitute initialization')
    activate('material', PINS['material'])
    run = RunStore(p['resolver'], layout='mounted').create(args.run_id)
    report = dict(schema_version=1, protocol=PROTOCOL, status='running', phase='allocated', variant=args.variant,
                  infrastructure_commit=args.commit, scientific_commit=PINS['material'], run_id=args.run_id,
                  freeze_sha256=args.freeze_sha256, source_files=freeze['files'], declared_image_id=IMAGE,
                  authorization_sha256=args.authorization_sha256, oracle_benchmark=proof,
                  optimizer_calls_completed=0, cleanup_verified=False, scientific_work_complete=False,
                  expected_initialization=expected, material_implementation=identity(args.variant, p['configuration']))
    started = time.monotonic(); last_budget = [0.]
    with run.output('recovery.json').open('x') as stream:
        def persist():
            report['elapsed_seconds'] = time.monotonic()-started
            write_report(stream, report)
            require(report['elapsed_seconds'] < freeze['limits']['arm_seconds'], 'E.13 arm time budget exhausted')
            if time.monotonic()-last_budget[0] >= 5:
                last_budget[0] = time.monotonic()
                size = sum(q.stat().st_size for q in run.root.rglob('*') if q.is_file() and not q.is_symlink())
                require(size < freeze['limits']['output_bytes_per_arm'], 'E.13 output cap exhausted')
        persist()
        try:
            workspace, family, original_inputs = prepare_view(run, p['selected'].dataset, 'bell', p['science'], p['configuration'], report['material_implementation'])
            cfg = deepcopy(p['configuration']); cfg['dataset_dir'] = str(family); cfg['mesh'] = str(p['selected'].mesh)
            for nested in [cfg.get('train_dataset_cfg', {}), *[v['cfg'] for v in cfg.get('val_set_list', [])]]:
                if 'dataset_dir' in nested:
                    nested['dataset_dir'] = str(family)
            report['runtime_configuration'] = cfg
            run.output('config.original.yaml').write_bytes(p['raw'])
            run.output('config.envelope.yaml').write_bytes(p['envelope'])
            with runtime_environment(run):
                os.chdir(str(workspace))
                import numpy as np
                import torch
                require(sys.version_info[:2] == (3,8) and torch.__version__ == '1.8.1+cu111' and
                        torch.version.cuda == '11.1' and torch.cuda.is_available() and torch.cuda.device_count() == 1 and
                        torch.cuda.get_device_name(0) == 'NVIDIA A100 80GB PCIe', 'Reviewed native runtime required')
                from infrastructure.verified_ray_backend import load
                report['ray_backend'] = load('predicate_fp64', IMAGE)
                import network.renderer as renderer
                import network.field as field
                from train.trainer import Trainer
                from train.train_valid import ValidationEvaluator
                from network.loss import NeRFRenderLoss, MaterialRegLoss
                from infrastructure.geometry_training_smoke import runtime_identity
                report['runtime'] = runtime_identity(torch)
                require(Path(Trainer.run.__code__.co_filename).resolve() == p['science']/'train/trainer.py', 'Mixed scientific Trainer')
                install(renderer, field, args.variant, cfg)
                trainer = trainer_io(Trainer, run)(cfg); trainer.e13_run = run
                probe = RecoveryProbe(trainer, torch, np, report, persist)
                probe.prepare_calls(renderer.name2renderer['material'], NeRFRenderLoss, MaterialRegLoss, Trainer, ValidationEvaluator)
                report['phase'] = 'original_native_training'; persist()
                with probe.observe_calls():
                    EndpointTrace(trainer, probe, p['science']/'train/trainer.py').run()
                require(probe._binding_count == 1 and report['optimizer_calls_completed'] == RETURNS and
                        len(report['trajectory']) == RETURNS and not list(run.checkpoints.iterdir()), 'Incomplete native training endpoint')
                report['completed_training_work'] = dict(original_forwards=RETURNS, original_backwards=RETURNS,
                                                        original_adam_returns=RETURNS, direction_slots=RETURNS*512*1536)
                model = trainer.network
                report['training_rng_endpoint'] = rng_identity(torch, np)
                from infrastructure.training_probe import optimizer_record
                report['final_optimizer'] = optimizer_record(trainer.optimizer, model, np)
                report['phase'] = 'strict_endpoint_checkpoint'; persist()
                report['checkpoint'] = strict_roundtrip(model, trainer.optimizer, run.checkpoints/'endpoint-2000.pth',
                    args.variant, torch, np, dict(commit=args.commit, freeze_sha256=args.freeze_sha256,
                                                oracle_admission_sha256=ADMISSION, material_implementation=report['material_implementation']))
                require(rng_identity(torch,np) == report['training_rng_endpoint'], 'Checkpoint adapter consumed scientific RNG')
                from infrastructure.material_recovery_e13_evaluation import material_export, render_evaluation, native_tensor_default
                generation = json.loads((args.inputs/'geometry/generation.json').read_text())
                require(report['splits'] == dict(train_ids=generation['splits']['train'],test_ids=generation['splits']['validation'],
                                                nvs_ids=generation['splits']['held_view_nvs']), 'Native/oracle split differs')
                report['phase'] = 'fixed_endpoint_material_export'; persist()
                model.eval()
                with native_tensor_default(torch):
                    # Original A's pure tangent routine creates reference tensors
                    # under the CUDA default established by its native forward.
                    report['material_recovery'] = material_export(model,args.inputs/'geometry',run,generation['splits'],torch,np)
                persist()  # Preserve invalid baseline outcomes before any mandatory stop.
                require(all(v['invalid_frames'] == v['invalid_materials'] == 0
                            for v in report['material_recovery']['split_aggregates'].values()),
                        'Physically invalid endpoint material/frame; preserved without repair; remaining arms must stop')
                require(rng_identity(torch,np) == report['training_rng_endpoint'], 'Material export consumed scientific RNG')
                report['phase'] = 'bounded_native_evaluation'; persist()
                report['image_evaluation'] = render_evaluation(model,args.inputs/'geometry',args.inputs/'references',generation,run,torch,np)
                report['evaluation_rng_endpoint'] = rng_identity(torch,np)
                report['gpu_memory'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                report['imported_scientific_modules'] = imported_sources(p['science'])
                # Owned Trainer workers/writer/progress were already closed by EndpointTrace.
                del probe, model, trainer
                import gc
                gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
            report['cleanup_verified'] = True
            verify_freeze(args.freeze_sha256); verify_inputs(args.inputs,freeze)
            require(args.initial is None or digest(args.initial) == args.initial_sha256, 'Initialization anchor changed')
            require(source_records(p['selected'].dataset/'bell') == original_inputs and admit_oracle(p,args.admission) == proof,
                    'Original source inputs/admission changed')
            report.update(scientific_work_complete=True, phase='complete', status='complete')
            import resource
            report['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            report['output_hashes'] = {q.relative_to(run.root).as_posix(): digest(q) for q in run.root.rglob('*')
                if q.is_file() and not q.is_symlink() and q.name != 'recovery.json' and
                not any(s in ('home','tmp','torch_extensions') for s in q.relative_to(run.root).parts)}
            persist()
        except BaseException as exc:
            report.update(status='failed', scientific_work_complete=False, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            write_report(stream, report)
            raise
    write_json(run.output('completion.json'), dict(status='complete',run_id=args.run_id,variant=args.variant,
               recovery_sha256=digest(run.output('recovery.json')),optimizer_returns=RETURNS,
               scientific_work_complete=True,cleanup_verified=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=ARMS, required=True)
    for name in ('inputs','thesis','admission','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--authorization', type=Path)
    for name in ('authorization-sha256','commit','run-id','initial-sha256'):
        p.add_argument('--'+name)
    p.add_argument('--initial', type=Path)
    args = p.parse_args()
    freeze = verify_freeze(args.freeze_sha256)
    plan_value, proof = prepared(args, freeze)
    if not args.execute:
        require(not any(n.split('.')[0] in ('torch','network','train','raytracing','_raytracing') for n in sys.modules), 'Metadata validation imported science')
        print(json.dumps(dict(status='metadata_only', variant=args.variant, native_calls=0, optimizer_returns=0)))
        return
    require(all((args.authorization,args.authorization_sha256,args.commit,args.run_id)), 'Bound execution authorization required')
    execute(args, freeze, plan_value, proof)


if __name__ == '__main__':
    main()
