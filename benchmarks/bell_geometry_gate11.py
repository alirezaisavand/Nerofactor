"""Cold full-run definition for the isolated oracle-mask O1 diagnostic."""
from pathlib import Path
from benchmarks import bell_geometry_gate10_v1 as original
from benchmarks import bell_geometry_gate103_v2 as evolution
from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.signatures import require

PROTOCOL='bell-background-opacity-oracle-gate11-v1'
VARIANT='G_background_opacity_oracle_v1'
FREEZE='docs/protocols/BELL_BACKGROUND_OPACITY_GATE11_V1.freeze.json'
INVENTORY='docs/results/publication_gate11/input-inventory.json'
read,write,item,check=original.read,original.write,original.item,original.check
STEP,SEED,SIZE,CADENCE,KEEP_STEPS=original.STEP,original.SEED,original.SIZE,original.CADENCE,original.KEEP_STEPS
CFG=original.CFG
policy=evolution.policy
negative_extension,silhouette,opacity_summary=evolution.negative_extension,evolution.silhouette,evolution.opacity_summary
STEPS=(15000,30000,50000,300000)
TRAIN_LIMITS=dict(original.LIMITS,owned_checkpoint_loads=1,mask_observer_calls=STEP,mask_trace_rows=15006,extra_backwards=0)
TRAIN_RESOURCES=original.RESOURCES
RESOURCES=dict(evolution.RESOURCES,watchdog=7200,inner=7140,pids=256,output_bytes=16*1024**3)
LIMITS=dict(evolution.LIMITS,dense_grids=4,dense_tiles=2048,dense_SDF_points=4*512**3,MC_calls=4,
    audit_BVH_rays=4*196608,CPU_ray_checks=4*384+4*64,consistency_SDF_points=4*1536,
    consistency_gradient_points=4*512,camera_precision_probe_calls=0)
NEURAL={k:v for k,v in evolution.NEURAL.items() if k!='precision_witness'}
SOURCES=tuple(dict.fromkeys((*evolution.SOURCES,evolution.FREEZE,
    'docs/results/publication_gate103/RESULTS_V2.md',
    'docs/results/publication_gate103/results-v2/summary.json',
    'docs/results/publication_gate103/results-v2/verification.json',
    'docs/results/publication_gate103/results-v2/artifact-manifest.json',
    'benchmarks/bell_geometry_gate11.py','scientific_variants/geometry_background_opacity_v1.py',
    'experiments/bell_geometry_gate11.py','experiments/bell_geometry_gate11_evaluate.py',
    'experiments/bell_geometry_gate11_verify.py','experiments/analyze_bell_geometry_gate11.py',
    'experiments/analyze_bell_geometry_gate103_v3.py',
    'benchmarks/bell_mask_trimap_gate11.py','docs/results/publication_gate11/realistic-mask-definition.json',
    'experiments/run_bell_geometry_gate11.sh','tests/scientific/test_bell_geometry_gate11.py',
    'benchmarks/bell_geometry_gate103_replay_v3.py',
    'docs/protocols/BELL_BACKGROUND_OPACITY_GATE11_V1.md',
    'docs/protocols/BELL_IMPERFECT_MASK_ACQUISITION_GATE11_V1.md',INVENTORY,
    'docs/RESEARCH_BACKLOG_GATE11.md',
    'docs/results/publication_gate11/sam2-readiness.json','docs/results/publication_gate11/READINESS.md')))


def definition(out):
    inputs=read(ROOT/INVENTORY);old=inputs['original_freeze'];e=inputs['evolution_inputs']
    return dict(protocol=PROTOCOL,variant=VARIANT,host_output=str(out),scientific_commit=PINS['geometry'],
        image=old['image'],backend=old['backend'],seed=SEED,steps=list(STEPS),
        retained_checkpoints=list(KEEP_STEPS),publication_endpoint=STEP,
        acquisition=old['acquisition'],splits=old['splits'],train_files=old['train_files'],
        eval_files=e['geometry_files'],calibration_subset_sha256=old['calibration_subset_sha256'],
        receipts=old['receipts'],O0=inputs['O0'],O0_evolution_summary=inputs['O0_evolution_summary'],
        O0_evolution_root=read(ROOT/'docs/results/publication_gate103/results-v2/artifact-manifest.json')['native_root'],
        O0_evolution_receipt=read(ROOT/'docs/results/publication_gate103/results-v2/summary.json')['native_receipt'],
        objective=dict(foreground='unchanged original F.l1_loss(mask,mask*fg_acc,reduction=mean)',
            addition='mean((1-mask)*fg_acc)',background_coefficient=1.0,mask_schedule='original MaskLoss.get_mask_weight',
            mask_semantics='synthetic oracle-hit masks; not a realistic SAM2 comparison'),
        extension_cache='/outputs/torch_extensions',jit_cache=e['jit_cache'],
        jit_cache_root=str(evolution.TRAIN/'torch_extensions'),
        worker_transport=old['worker_transport'],fitting_ids=e['fitting_ids'],
        neural_panel=e['neural_panel'],neural=NEURAL,
        outside_mask=evolution.policy.read(ROOT/evolution.policy.FREEZE)['outside_mask'],
        outside_mask_path=str(Path(evolution.policy.read(ROOT/evolution.policy.FREEZE)['diagnostic'])/'run/stages/outside-sphere.npy'),
        domain=evolution.policy.read(ROOT/evolution.policy.FREEZE)['domain'],
        cell_policy=old_cell_policy(),numerical_policy='float32-exact-coordinate-weld-and-exact-zero-area-removal',
        quality=old['quality'],train_limits=TRAIN_LIMITS,limits=LIMITS,
        training_resources=TRAIN_RESOURCES,evaluation_resources=RESOURCES,whole_cooperative_seconds=138000,
        sources={name:digest(ROOT/name) for name in SOURCES})


def old_cell_policy():
    return evolution.policy.read(ROOT/evolution.policy.FREEZE)['cell_policy']


def freeze(sha):
    require(digest(ROOT/FREEZE)==sha,'Gate 11 freeze differs')
    f=read(ROOT/FREEZE);out=Path(f['host_output'])
    require(out.resolve()==out and out.parent==evolution.TRAIN.parent.parent and
            out.name.startswith('bell-background-opacity-gate11-') and f==definition(out),'Gate 11 definition differs')
    return f


def host_inputs(f):
    o0=read(check(Path(f['O0']['root'])/'training.json',f['O0']['training_receipt']))
    require(o0['status']=='complete' and o0['cleanup_verified'] and o0['adam_returns']==STEP and
            o0['initial_state_sha256']==f['O0']['initial_state_sha256'],'Authenticated O0 differs')
    for name,identity in f['train_files'].items():check(Path(f['acquisition'])/'datasets/GlossySynthetic'/name,identity)
    for name,identity in f['eval_files'].items():check(Path(f['acquisition'])/'geometry'/name,identity)
    for name,identity in f['jit_cache'].items():check(Path(f['jit_cache_root'])/name,identity)
    check(Path(f['outside_mask_path']),f['outside_mask'])
    check(ROOT/'docs/results/publication_gate103/results-v2/summary.json',f['O0_evolution_summary'])
    check(Path(f['O0_evolution_root'])/'run/receipt.json',f['O0_evolution_receipt'])
    for stage,commit in PINS.items():baseline(stage,commit)


def authorization(f,commit,sha):
    return dict(approved=True,protocol=PROTOCOL,variant=VARIANT,commit=commit,freeze_sha256=sha,
        host_output=f['host_output'],image=f['image'],backend=f['backend'],selected_gpu='0',
        maximum_execution_launches=1,optimizer_returns=STEP,
        CPU_pickle='only this new history: natural step1000 roundtrip and 15k/30k/50k/300k evaluation, once each (five loads)',
        train_limits=TRAIN_LIMITS,evaluation_limits=LIMITS,training_resources=TRAIN_RESOURCES,
        evaluation_resources=RESOURCES,whole_cooperative_seconds=f['whole_cooperative_seconds'],
        original_runtime_JIT_validation_or_identical_rebuild=True,segmentation_inference=False,
        R0_R1_training=False,material_or_external_training=False,retries=False)


stage_training,read_staged=original.stage_training,original.read_staged
