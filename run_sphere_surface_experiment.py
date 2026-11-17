"""Explicitly authorized 4000-update sphere surface-regularizer arm."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import signal
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
from infrastructure.stage_sources import ROOT,PINS,digest,baseline
from infrastructure.signatures import require
from infrastructure.sphere_replay import IMAGE,ORIGINAL_CFG,input_records,effective_configuration,workspace,BudgetStop
from infrastructure.sphere_surface_experiment import PROTOCOL,ARMS,native

LIMITS=dict(whole_seconds=3600,kill_grace_seconds=30,arm_seconds=1680,inner_grace_seconds=20,
 updates_per_arm=4000,absolute_start=4000,absolute_end=8000,arms=2,gradient_probe_steps=[4000,5999,7999],extra_reverse_sweeps=36,
 surface_pair_predictor_points=8192000,visibility_queries=2048,
 training_direction_slots=6291456000,validation_direction_slots=7166361600,
 cpu_threads=4,loader_workers=8,host_memory_gib=128,pids=256,output_bytes_per_arm=8*1024**3,gpu='0')


def preflight(path,sha):
    require(digest(path)==sha,'Surface experiment freeze hash mismatch')
    f=json.loads(path.read_text())
    require(f['protocol']==PROTOCOL and f['limits']==LIMITS and f['arms']==list(ARMS) and f['image_id']==IMAGE,
            'Frozen bounded experiment definition differs')
    for name,h in f['files'].items():
        p=ROOT/name
        require(p.resolve()==p and ROOT in p.parents and digest(p)==h,'Changed source/evidence: '+name)
    for stage in PINS:baseline(stage,PINS[stage])
    return f


def authorize(path,sha,f,freeze_sha,commit):
    require(path is not None and digest(path)==sha,'Explicit surface execution authorization required')
    a=json.loads(path.read_text())
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==commit and
            a.get('freeze_sha256')==freeze_sha and a.get('host_output_directory')==f['host_output_directory'] and
            a.get('image_id')==IMAGE and a.get('limits')==LIMITS and a.get('selected_gpu')=='0',
            'One exact researcher-approved paired launch required')
    require(a.get('trusted_initial_checkpoint_sha256')==f['initial_checkpoint']['sha256'],
            'Explicit trust/CPU-deserialization authorization for the selected own checkpoint required')
    return a


def completed_arm(path,arm,freeze_sha,commit):
    """Exit zero alone is insufficient: require the actual bounded native receipt."""
    require(path.is_file() and path.resolve()==path,'Missing or redirected completed arm report: '+str(path))
    value=json.loads(path.read_text())
    require(value.get('status')=='complete' and value.get('protocol')==PROTOCOL and
            value.get('arm')==arm and value.get('commit')==commit and
            value.get('freeze_sha256')==freeze_sha and value.get('optimizer_returns')==8000 and
            value.get('new_optimizer_returns')==4000 and value.get('cleanup_verified') is True and
            value.get('owned_cleanup_verified') is True,
            'Arm did not complete its authorized native work and cleanup')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('datasets','thesis','output','freeze'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True);p.add_argument('--commit',required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--control',type=Path)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--authorization',type=Path);p.add_argument('--authorization-sha256')
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    f=preflight(args.freeze,args.freeze_sha256)
    mesh=args.thesis/'meshes/sphere_shape-300000.ply'
    packet=json.loads((ROOT/'docs/evidence/SPHERE_CORRECTED_REPLAY_INPUTS_V1.json').read_text())
    records=input_records(args.datasets,mesh,args.thesis)
    require(records==packet['records'],'Source sphere input identity mismatch')
    require(args.checkpoint.is_absolute() and args.checkpoint.resolve()==args.checkpoint and
            args.checkpoint.is_file() and args.checkpoint.stat().st_size==f['initial_checkpoint']['bytes'] and
            digest(args.checkpoint)==f['initial_checkpoint']['sha256'],'Selected initial checkpoint identity mismatch')
    if not args.execute:
        print(json.dumps(dict(status='metadata_only',arm=args.arm,scientific_execution=False)));return
    authorize(args.authorization,args.authorization_sha256,f,args.freeze_sha256,args.commit)
    require(sys.version_info[:2]==(3,8) and args.output==Path('/outputs/run') and not args.output.exists(),
            'Reviewed Python3.8 and fresh arm directory required')
    require(args.checkpoint==Path('/inputs/initial.pth') and digest(args.checkpoint)==f['initial_checkpoint']['sha256'] and
            os.statvfs(str(args.checkpoint)).f_flag & os.ST_RDONLY,'Explicit read-only initial checkpoint identity mismatch')
    control=None
    if args.arm==ARMS[1]:
        require(args.control==Path('/inputs/control.json'),'Explicit completed control packet required')
        control=completed_arm(args.control,ARMS[0],args.freeze_sha256,args.commit)
    else:require(args.control is None,'First arm cannot use a control packet')
    from infrastructure.paths import PathResolver
    from infrastructure.runs import RunStore
    from infrastructure.stage_execution import native_guards
    from infrastructure.bell_inference import runtime_environment
    resolver=PathResolver(dict(datasets=args.datasets,thesis=args.thesis),args.output)
    native_guards(dict(stage='material',resolver=resolver,ray_backend='predicate_fp64'),IMAGE)
    run=RunStore(resolver,layout='mounted').create(f['run_id']+('-chart' if args.arm==ARMS[0] else '-q'))
    import yaml
    raw=(ROOT/ORIGINAL_CFG).read_bytes();cfg,diagnostics=effective_configuration(yaml.safe_load(raw),args.datasets,mesh,'sphere_surface')
    report=dict(schema_version=1,protocol=PROTOCOL,status='running',arm=args.arm,commit=args.commit,
        freeze_sha256=args.freeze_sha256,optimizer_returns=0,cleanup_verified=False,
        effective_configuration=cfg,legacy_diagnostics=diagnostics,energy_specular_mode='legacy_v1',
        external_spatial_mode=args.arm,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inherited_optimizer_updates=4000,new_optimizer_returns=0,
        fork_policy='Exact weights+Adam at4000, common seed6033 construction/sampler reset; NOT an exact historical resume',
        teacher_or_GT_material_supervision=False,control_packet_sha256=digest(args.control) if control else None)
    with (run.root/'surface-run.json').open('x') as stream:
        def persist():
            stream.seek(0);json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n');stream.truncate();stream.flush()
        persist()
        (run.root/'config.original.yaml').write_bytes(raw)
        (run.root/'config.effective.yaml').write_text(yaml.safe_dump(cfg))
        previous=signal.getsignal(signal.SIGTERM)
        def terminated(signum,frame):raise BudgetStop('Foreground watchdog termination; preserve partial arm')
        signal.signal(signal.SIGTERM,terminated)
        try:
            science,_=baseline('material',PINS['material']);work=workspace(run.root,science)
            with runtime_environment(run):
                os.chdir(work)
                native(cfg,run,report,persist,args.arm,IMAGE,control,(args.checkpoint,f['initial_checkpoint']))
            require(input_records(args.datasets,mesh,args.thesis)==records,'Original sphere assets changed')
            preflight(args.freeze,args.freeze_sha256)
            report.update(status='complete',cleanup_verified=True)
        except BaseException as exc:
            report.update(status='incomplete' if isinstance(exc,(KeyboardInterrupt,BudgetStop)) else 'failed',
                          error_type=type(exc).__name__,error=str(exc));traceback.print_exc();raise
        finally:
            signal.signal(signal.SIGTERM,previous)
            report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            report['outputs']={p.relative_to(run.root).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
                for directory in ('checkpoints','renders') for p in (run.root/directory).rglob('*') if p.is_file() and not p.is_symlink()}
            persist()


if __name__=='__main__':main()
