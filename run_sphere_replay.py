"""Explicit fresh sphere reconstruction; --execute requires a hash-bound approval."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import sys
import traceback

# Also works with python3 -I -B /workspace/run_sphere_replay.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, PINS, baseline, digest
from infrastructure.sphere_replay import (PROTOCOL, IMAGE, LIMITS, ORIGINAL_CFG, BudgetStop,
    effective_configuration, input_records, native_execute, workspace)


def verify_freeze(path, sha):
    require(path.is_file() and digest(path) == sha, 'Freeze identity mismatch')
    f = json.loads(path.read_text())
    require(f['protocol'] == PROTOCOL and f['limits'] == LIMITS and f['image_id'] == IMAGE and
            f['scientific_commit'] == PINS['material'], 'Freeze definition mismatch')
    for name, expected in f['files'].items():
        target = ROOT / name
        require(target.resolve() == target and ROOT in target.parents and digest(target) == expected,
                'Frozen source changed: ' + name)
    packet = ROOT / f['input_manifest']['path']
    require(packet.resolve() == packet and ROOT in packet.parents and digest(packet) == f['input_manifest']['sha256'],
            'Frozen sphere input manifest changed')
    inputs = json.loads(packet.read_text())
    require(inputs['protocol'] == PROTOCOL and len(inputs['records']) == 209, 'Incomplete sphere input inventory')
    f['inputs'] = inputs['records']
    baseline('material', PINS['material'])
    baseline('geometry', PINS['geometry'])
    return f


def verify_authorization(path, sha, freeze, freeze_sha, commit):
    require(path is not None and path.is_file() and digest(path) == sha, 'Authorization required/identity mismatch')
    a = json.loads(path.read_text())
    require(a.get('approved') is True and a.get('protocol') == PROTOCOL and a.get('commit') == commit and
            a.get('freeze_sha256') == freeze_sha and a.get('image_id') == IMAGE and a.get('limits') == LIMITS and
            a.get('host_output_directory') == freeze['host_output_directory'] and
            a.get('output_directory') == '/outputs/sphere' and a.get('selected_gpu') == '0',
            'Exact one-time sphere execution authorization required')
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, required=True, help='Explicit nerf_synthetic family directory')
    parser.add_argument('--thesis', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--image-id', default=IMAGE)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze, args.freeze_sha256)
    records = input_records(args.datasets, args.mesh, args.thesis)
    require(records == freeze['inputs'], 'Frozen sphere inputs changed')
    import yaml
    raw = yaml.safe_load((ROOT / ORIGINAL_CFG).read_text())
    cfg, diagnostics = effective_configuration(raw, args.datasets, args.mesh, freeze['model_name'])
    require(yaml.safe_load((ROOT / 'configs/replays/sphere_corrected_core_v1.yaml').read_text()) ==
            effective_configuration(raw, '/inputs/datasets/nerf_synthetic',
                                    '/inputs/thesis/meshes/sphere_shape-300000.ply', freeze['model_name'])[0],
            'Committed working configuration differs from reviewed effective semantics')
    if not args.execute:
        print(json.dumps(dict(status='metadata_validated_only', scientific_execution=False,
                              inputs=len(records), ignored_legacy_fields=diagnostics), indent=2))
        return
    verify_authorization(args.authorization, args.authorization_sha256, freeze, args.freeze_sha256, args.commit)
    require(sys.version_info[:2] == (3,8) and args.image_id == IMAGE, 'Reviewed Python3.8/image required')
    require(args.output == Path('/outputs/sphere') and not args.output.exists(), 'Fresh fixed sphere output required')
    require(os.geteuid() != 0 and os.statvfs('/').f_flag & os.ST_RDONLY, 'Non-root/read-only runtime required')
    for path in (ROOT, args.datasets, args.thesis, args.mesh):
        require(os.statvfs(str(path)).f_flag & os.ST_RDONLY, 'Read-only inputs/source required')
    require({line.split(':',1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]} == {'lo'},
            'No-network execution required')
    from infrastructure.paths import PathResolver
    from infrastructure.runs import RunStore
    from infrastructure.bell_inference import runtime_environment
    resolver = PathResolver({'datasets':args.datasets, 'thesis':args.thesis}, args.output)
    run = RunStore(resolver, layout='mounted').create(freeze['run_id'])
    report = dict(schema_version=1, protocol=PROTOCOL, status='running', phase='allocated',
        scientific_work_complete=False, cleanup_verified=False, evidence_finalized=False,
        commit=args.commit, freeze_sha256=args.freeze_sha256, authorization_sha256=args.authorization_sha256,
        run_id=freeze['run_id'], declared_image_id=args.image_id, input_records=records,
        historical_mesh_diagnostic_replay=True, research_G_handoff_admitted=False,
        optimizer_returns=0, configuration=cfg, legacy_diagnostics=diagnostics,
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (run.root / 'sphere-replay.json').open('x') as stream:
        def persist():
            stream.seek(0);json.dump(report, stream, indent=2, allow_nan=False);stream.write('\n');stream.truncate();stream.flush()
        persist()
        (run.root / 'config.original.yaml').write_bytes((ROOT / ORIGINAL_CFG).read_bytes())
        (run.root / 'config.effective.yaml').write_text(yaml.safe_dump(cfg))
        previous = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise BudgetStop('External foreground watchdog/user termination; no automatic resume')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            science, _ = baseline('material', PINS['material'])
            work = workspace(run.root, science)
            with runtime_environment(run):
                os.chdir(str(work))
                native_execute(cfg, run.root, report, persist, args.image_id)
            # Only after runtime_environment exits normally acknowledge cleanup.
            report['scientific_work_complete'] = report['optimizer_returns'] == 100000
            report['cleanup_verified'] = True
            require(input_records(args.datasets, args.mesh, args.thesis) == records, 'Original sphere inputs changed')
            verify_freeze(args.freeze, args.freeze_sha256)
            report['phase'] = 'saved_image_comparison'
            persist()
            from infrastructure.sphere_visuals import historical_comparison
            report['comparison'] = historical_comparison(run.root, args.thesis)
            report.update(status='complete', phase='complete', evidence_finalized=True)
        except BaseException as exc:
            traceback.print_exc()
            report.update(status='incomplete' if isinstance(exc,(BudgetStop,KeyboardInterrupt)) else 'failed',
                          error_type=type(exc).__name__, error=str(exc))
            persist()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)
            report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            # Raw checkpoints/images stay outside Git. A small output inventory
            # binds them without copying scientific artifacts into the repository.
            report['outputs'] = {p.relative_to(run.root).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
                for folder in ('checkpoints','renders','comparison') for p in (run.root/folder).rglob('*')
                if p.is_file() and not p.is_symlink()}
            persist()


if __name__ == '__main__':
    main()
