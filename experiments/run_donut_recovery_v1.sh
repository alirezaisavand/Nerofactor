#!/usr/bin/env bash
# Exactly one approved reference or training batch, foreground only.
set -euo pipefail
: "${DONUT_SOURCE:?}" "${DONUT_ORACLE:?}" "${DONUT_CAMERAS:?}" "${DONUT_BATCH:?}" "${DONUT_AUTH:?}" "${DONUT_GPU:?}" "${DONUT_PHASE:?reference, training, calibrated, ablation, surface, chart or tensor}"
DONUT_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
DONUT_COMMIT=$(git -C "$DONUT_SOURCE" rev-parse HEAD)
test -z "$(git -C "$DONUT_SOURCE" status --porcelain)"
DONUT_FREEZE_NAME=${DONUT_FREEZE_NAME:-DONUT_KURT_RECOVERY_V1.freeze.json}
case "$DONUT_FREEZE_NAME" in DONUT_KURT_RECOVERY_V1.freeze.json|DONUT_KURT_REFERENCE_V2.freeze.json|DONUT_KURT_RECOVERY_V2.freeze.json|DONUT_KURT_CALIBRATED_D3_V1.freeze.json|DONUT_KURT_DIFFUSE_WHITE_ABLATION_V1.freeze.json|DONUT_KURT_SURFACE_Q_D6_V1.freeze.json|DONUT_KURT_SURFACE_CHART_D7_V1.freeze.json|DONUT_KURT_SURFACE_CHART_D7_V2.freeze.json|DONUT_KURT_PAIRED_TENSOR_D8_V1.freeze.json|DONUT_KURT_PAIRED_TENSOR_D9_V1.freeze.json) ;; *) exit 2 ;; esac
DONUT_FREEZE=$(sha256sum "$DONUT_SOURCE/docs/protocols/$DONUT_FREEZE_NAME" | cut -d' ' -f1)
DONUT_AUTH_SHA=$(sha256sum "$DONUT_AUTH" | cut -d' ' -f1)
python3 -B - "$DONUT_SOURCE" "$DONUT_AUTH" "$DONUT_BATCH" "$DONUT_PHASE" "$DONUT_COMMIT" "$DONUT_FREEZE" "$DONUT_IMAGE" "$DONUT_GPU" "$DONUT_FREEZE_NAME" "${DONUT_BENCHMARK:-}" "${DONUT_INITIAL:-}" <<'PY'
import hashlib
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from benchmarks.donut_recovery_v1 import ABLATION_ARMS,REFERENCE,TRAINING,TRAIN_LIMITS,verify_freeze
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());batch=Path(sys.argv[3]);phase=sys.argv[4]
require(phase in ('reference','training','calibrated','ablation','surface','chart','tensor'),'Explicit phase required')
f=verify_freeze(Path(sys.argv[1])/'docs/protocols'/sys.argv[9],sys.argv[6])
d3=f.get('calibrated_training')
ablation=f.get('diffuse_white_ablation')
d6=f.get('surface_q_training')
d7=f.get('surface_chart_training')
paired=f.get('tensor_training')
require((phase=='calibrated')==(d3 is not None),'D3 requires its dedicated execution freeze')
require((phase=='ablation')==(ablation is not None),'D4/D5 require their dedicated execution freeze')
require((phase=='surface')==(d6 is not None),'D6 requires its dedicated execution freeze')
require((phase=='chart')==(d7 is not None),'D7 requires its dedicated execution freeze')
require((phase=='tensor')==(paired is not None),'Paired tensor training requires its dedicated execution freeze')
protocol=f.get('reference_protocol',REFERENCE) if phase=='reference' else TRAINING
limits=f['reference_limits'] if phase=='reference' else TRAIN_LIMITS
bound=d6 or d7 or paired or d3 or ablation
if bound:
 protocol,limits=bound['protocol'],bound['limits']
 arms=ABLATION_ARMS if ablation else ({'surface':('D6',),'chart':('D7',),'calibrated':('D3',),'tensor':tuple(paired['arms']) if paired else ()}[phase])
 if ablation:
  require(a.get('arms')==list(arms) and a.get('wall_seconds')==ablation['wall_seconds'],
   'D4/D5 order or outer watchdog differs')
 if d6 or d7 or paired:
  require(a.get('spatial_policy')==bound['spatial_policy'] and a.get('diffuse_white_in_objective') is True,
   'Frozen spatial or diffuse-white objective differs')
 def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 benchmark=Path(sys.argv[10]);initial=Path(sys.argv[11])
 require(benchmark.is_absolute() and benchmark.resolve()==benchmark and
  digest(benchmark/'generation.json')==a.get('benchmark_generation_sha256')==bound['benchmark_generation_sha256'] and
  initial.is_absolute() and initial.resolve()==initial and digest(initial)==a.get('initial_sha256')==bound['initial_sha256'],
  'Dataset or initialization anchor differs')
 admission=Path(a['dataset_admission_path'])
 require(admission.is_absolute() and digest(admission)==a.get('dataset_admission_sha256')==bound['dataset_admission_sha256'] and
  json.loads(admission.read_text()).get('researcher_admitted_dataset') is True and
  a.get('researcher_admitted_dataset') is True and a.get('maximum_execution_launches')==1 and
  a.get('output_directories')=={arm:'/outputs/'+arm for arm in arms}, 'Admission or one-launch authorization differs')
 if paired:
  require(a.get('arms')==list(arms) and a.get('wall_seconds')==limits['wall_seconds'] and
   a.get('representation')==paired['representation'],'Tensor paired arm order, policy or watchdog differs')
 if d7 or paired:
  from infrastructure.stage_sources import digest as file_digest
  require(benchmark==Path(bound['benchmark_host_path']) and initial==Path(bound['initial_host_path']) and
   admission==Path(bound['dataset_admission_host_path']), 'Frozen spatial input paths differ')
  for item in bound['controls'].values():
   p=Path(item['path'])
   require(p.resolve()==p and file_digest(p)==item['sha256'], 'Accepted control changed: '+str(p))
require(a.get('approved') is True and a.get('protocol')==protocol and a.get('limits')==limits and
 a.get('commit')==sys.argv[5] and a.get('freeze_sha256')==sys.argv[6] and a.get('image_id')==sys.argv[7] and
 a.get('selected_gpu')==sys.argv[8]=='0' and a.get('host_output_directory')==str(batch)==f['host_'+phase+'_output'], 'Exact separate approval required')
require(batch.is_absolute() and batch.resolve()==batch and not batch.exists(),'Fresh batch required')
PY
test "$(docker image inspect --format '{{.Id}}' "$DONUT_IMAGE")" = "$DONUT_IMAGE"
test -z "$(docker ps -q --filter label=nerofactor.donut-recovery)"
mkdir "$DONUT_BATCH"
cp "$DONUT_AUTH" "$DONUT_BATCH/authorization.json"
cp "$DONUT_SOURCE/docs/protocols/$DONUT_FREEZE_NAME" "$DONUT_BATCH/freeze.json"
DONUT_OWNED_CID=''
cleanup_owned() {
 test -n "$DONUT_OWNED_CID" && test -s "$DONUT_OWNED_CID" || return 0
 local cid
 cid=$(cat "$DONUT_OWNED_CID")
 if test "$(docker inspect --format '{{ index .Config.Labels "nerofactor.donut-recovery" }}' "$cid" 2>/dev/null || true)" = "$(basename "$DONUT_BATCH")"; then
  docker rm --force "$cid" >> "$DONUT_BATCH/cleanup.txt" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
arms=(reference);seconds=7200;inner=7080
if test "$DONUT_PHASE" = training; then
 : "${DONUT_BENCHMARK:?}"
 arms=(D1 D2);seconds=3600;inner=3540
fi
if test "$DONUT_PHASE" = calibrated; then
 : "${DONUT_BENCHMARK:?}" "${DONUT_INITIAL:?}"
 arms=(D3);seconds=3600;inner=3540
fi
if test "$DONUT_PHASE" = ablation; then
 : "${DONUT_BENCHMARK:?}" "${DONUT_INITIAL:?}"
 arms=(D4 D5);seconds=3600;inner=3540
fi
if [[ "$DONUT_PHASE" == surface || "$DONUT_PHASE" == chart ]]; then
 : "${DONUT_BENCHMARK:?}" "${DONUT_INITIAL:?}"
 arms=(D6);seconds=3600;inner=3540
 if test "$DONUT_PHASE" = chart; then arms=(D7); fi
fi
if test "$DONUT_PHASE" = tensor; then
 : "${DONUT_BENCHMARK:?}" "${DONUT_INITIAL:?}"
 arms=(D8-C D8-T);seconds=3600;inner=3540
 if test "$DONUT_FREEZE_NAME" = DONUT_KURT_PAIRED_TENSOR_D9_V1.freeze.json; then arms=(D9-C D9-T); fi
fi
for arm in "${arms[@]}"; do
 extra=();mounts=()
 if test "$DONUT_PHASE" = reference && test "$DONUT_FREEZE_NAME" = DONUT_KURT_REFERENCE_V2.freeze.json; then
  : "${DONUT_PREVIOUS:?}"
  extra+=(--previous /inputs/previous)
  mounts+=(--mount "type=bind,src=$DONUT_PREVIOUS,dst=/inputs/previous,readonly")
 fi
 if test "$DONUT_PHASE" != reference; then
  BENCHMARK_SHA=$(sha256sum "$DONUT_BENCHMARK/generation.json" | cut -d' ' -f1)
  extra+=(--benchmark /inputs/benchmark --benchmark-sha256 "$BENCHMARK_SHA")
  mounts+=(--mount "type=bind,src=$DONUT_BENCHMARK,dst=/inputs/benchmark,readonly")
  if test "$arm" = D2; then
   python3 -B - "$DONUT_BATCH/D1/recovery.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['status']=='complete' and r['optimizer_returns']==10000 and r['cleanup_verified']
PY
   INITIAL_SHA=$(sha256sum "$DONUT_BATCH/D1/initial.json" | cut -d' ' -f1)
   extra+=(--initial /inputs/initial.json --initial-sha256 "$INITIAL_SHA")
   mounts+=(--mount "type=bind,src=$DONUT_BATCH/D1/initial.json,dst=/inputs/initial.json,readonly")
  fi
 fi
 if [[ "$arm" == D3 || "$arm" == D4 || "$arm" == D5 || "$arm" == D6 || "$arm" == D7 || "$arm" == D8-C || "$arm" == D8-T || "$arm" == D9-C || "$arm" == D9-T ]]; then
  INITIAL_SHA=$(sha256sum "$DONUT_INITIAL" | cut -d' ' -f1)
  extra+=(--initial /inputs/initial.json --initial-sha256 "$INITIAL_SHA")
  mounts+=(--mount "type=bind,src=$DONUT_INITIAL,dst=/inputs/initial.json,readonly")
 fi
 DONUT_OWNED_CID="$DONUT_BATCH/$arm.cid"
 set +e
 timeout --foreground --signal=TERM --kill-after=30s "${seconds}s" \
 docker run --rm --pull=never --init --cidfile "$DONUT_OWNED_CID" \
 --name "$(basename "$DONUT_BATCH")-$arm" --label "nerofactor.donut-recovery=$(basename "$DONUT_BATCH")" \
 --user "$(id -u):$(id -g)" --gpus "device=$DONUT_GPU" --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
 --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env "HOME=/outputs/$arm/runtime" --env "XDG_CACHE_HOME=/outputs/$arm/runtime/cache" \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$DONUT_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$DONUT_ORACLE,dst=/inputs/oracle,readonly" \
 --mount "type=bind,src=$DONUT_CAMERAS,dst=/inputs/cameras.json,readonly" \
 --mount "type=bind,src=$DONUT_AUTH,dst=/authorization.json,readonly" \
 --mount "type=bind,src=$DONUT_BATCH,dst=/outputs" "${mounts[@]}" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$DONUT_IMAGE" \
 --signal=TERM --kill-after=20s "${inner}s" /usr/bin/python3 -I -B /workspace/run_donut_recovery.py \
 --phase "$arm" --oracle /inputs/oracle --cameras /inputs/cameras.json --output "/outputs/$arm" \
 --freeze "/workspace/docs/protocols/$DONUT_FREEZE_NAME" --freeze-sha256 "$DONUT_FREEZE" \
 --authorization /authorization.json --authorization-sha256 "$DONUT_AUTH_SHA" \
 --commit "$DONUT_COMMIT" --image-id "$DONUT_IMAGE" "${extra[@]}" --execute > "$DONUT_BATCH/$arm-console.txt" 2>&1
 code=$?
 set -e
 printf '%s\n' "$code" > "$DONUT_BATCH/$arm-exit-code.txt"
 cleanup_owned
 cat "$DONUT_BATCH/$arm-console.txt"
 test "$code" -eq 0
 python3 -B - "$DONUT_BATCH" "$arm" <<'PY'
import json,sys
from pathlib import Path
arm=sys.argv[2];p=Path(sys.argv[1])/arm
r=json.loads((p/('generation.json' if arm=='reference' else 'recovery.json')).read_text())
assert r['status']==('complete_pending_researcher_admission' if arm=='reference' else 'complete') and r['cleanup_verified']
if arm!='reference':assert r['arm']==arm and r['scientific_work_complete'] and r['optimizer_returns']==10000
if arm=='D3':
 f=json.loads((p.parent/'freeze.json').read_text())['calibrated_training']
 assert r['protocol']==f['protocol'] and r['initialization_anchor_sha256']==f['initial_sha256']
 assert all(r['counters'][k]==f['limits'][k] for k in ('calibration_evaluations','calibration_directions','component_reverse_sweeps'))
if arm in ('D4','D5'):
 f=json.loads((p.parent/'freeze.json').read_text())['diffuse_white_ablation']
 a=json.loads((p.parent/'authorization.json').read_text())
 assert r['protocol']==f['protocol'] and r['initialization_anchor_sha256']==f['initial_sha256']
 assert r['commit']==a['commit'] and r['freeze_sha256']==a['freeze_sha256']
 assert r['diffuse_white_in_objective'] is False and r['mean_calibration']==(arm=='D4')
 assert r['counters']['component_reverse_sweeps']==f['limits'][arm]['component_reverse_sweeps']
 for key in ('calibration_evaluations','calibration_directions'):
  assert r['counters'].get(key,0)==f['limits'][arm].get(key,0)
if arm in ('D6','D7','D8-C','D8-T','D9-C','D9-T'):
 key={'D6':'surface_q_training','D7':'surface_chart_training'}.get(arm,'tensor_training')
 f=json.loads((p.parent/'freeze.json').read_text())[key]
 a=json.loads((p.parent/'authorization.json').read_text())
 assert r['protocol']==f['protocol'] and r['initialization_anchor_sha256']==f['initial_sha256']
 assert r['commit']==a['commit'] and r['freeze_sha256']==a['freeze_sha256']
 assert r['diffuse_white_in_objective'] is True and r['mean_calibration'] is False
 assert r['surface_q_policy']==f['spatial_policy']
 if arm=='D7':
  assert r['D6_domain_parity_verified'] is True
  assert r['optimization_terms']==['RGB_loss','material_prior','diffuse_white_prior','surface_chart_loss']
  assert [o['step'] for o in r['saved_endpoints']]==[0,2000,4000,6000,8000,10000]
 if key=='tensor_training':
  assert arm in f['arms']
  assert r['D6_domain_parity_verified'] is True and r['paired_initial_gate']['passed'] is True
  assert r['representation']==f['representation']
  assert r['optimization_terms']==['RGB_loss','material_prior','diffuse_white_prior','surface_q_loss']
  assert [o['step'] for o in r['saved_endpoints']]==[0,2000,4000,6000,8000,10000]
 keys=('component_reverse_sweeps','surface_q_batches','surface_q_pairs','surface_q_predictor_calls','surface_q_predictor_points')
 assert all(r['counters'][key]==f['limits'][key] for key in keys)
PY
done
trap - EXIT
