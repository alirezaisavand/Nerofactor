#!/usr/bin/env bash
# Exactly one approved reference or training batch, foreground only.
set -euo pipefail
: "${DONUT_SOURCE:?}" "${DONUT_ORACLE:?}" "${DONUT_CAMERAS:?}" "${DONUT_BATCH:?}" "${DONUT_AUTH:?}" "${DONUT_GPU:?}" "${DONUT_PHASE:?reference, training or calibrated}"
DONUT_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
DONUT_COMMIT=$(git -C "$DONUT_SOURCE" rev-parse HEAD)
test -z "$(git -C "$DONUT_SOURCE" status --porcelain)"
DONUT_FREEZE_NAME=${DONUT_FREEZE_NAME:-DONUT_KURT_RECOVERY_V1.freeze.json}
case "$DONUT_FREEZE_NAME" in DONUT_KURT_RECOVERY_V1.freeze.json|DONUT_KURT_REFERENCE_V2.freeze.json|DONUT_KURT_RECOVERY_V2.freeze.json|DONUT_KURT_CALIBRATED_D3_V1.freeze.json) ;; *) exit 2 ;; esac
DONUT_FREEZE=$(sha256sum "$DONUT_SOURCE/docs/protocols/$DONUT_FREEZE_NAME" | cut -d' ' -f1)
DONUT_AUTH_SHA=$(sha256sum "$DONUT_AUTH" | cut -d' ' -f1)
python3 -B - "$DONUT_SOURCE" "$DONUT_AUTH" "$DONUT_BATCH" "$DONUT_PHASE" "$DONUT_COMMIT" "$DONUT_FREEZE" "$DONUT_IMAGE" "$DONUT_GPU" "$DONUT_FREEZE_NAME" "${DONUT_BENCHMARK:-}" "${DONUT_INITIAL:-}" <<'PY'
import hashlib
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from benchmarks.donut_recovery_v1 import REFERENCE,TRAINING,REFERENCE_LIMITS,TRAIN_LIMITS,verify_freeze
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());batch=Path(sys.argv[3]);phase=sys.argv[4]
require(phase in ('reference','training','calibrated'),'Explicit phase required')
f=verify_freeze(Path(sys.argv[1])/'docs/protocols'/sys.argv[9],sys.argv[6])
d3=f.get('calibrated_training')
require((phase=='calibrated')==(d3 is not None),'D3 requires its dedicated execution freeze')
protocol=f.get('reference_protocol',REFERENCE) if phase=='reference' else TRAINING
limits=f['reference_limits'] if phase=='reference' else TRAIN_LIMITS
if d3:
 protocol,limits=d3['protocol'],d3['limits']
 def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 benchmark=Path(sys.argv[10]);initial=Path(sys.argv[11])
 require(benchmark.is_absolute() and benchmark.resolve()==benchmark and
  digest(benchmark/'generation.json')==a.get('benchmark_generation_sha256')==d3['benchmark_generation_sha256'] and
  initial.is_absolute() and initial.resolve()==initial and digest(initial)==a.get('initial_sha256')==d3['initial_sha256'],
  'D3 dataset or initialization anchor differs')
 admission=Path(a['dataset_admission_path'])
 require(admission.is_absolute() and digest(admission)==a.get('dataset_admission_sha256')==d3['dataset_admission_sha256'] and
  json.loads(admission.read_text()).get('researcher_admitted_dataset') is True and
  a.get('researcher_admitted_dataset') is True and a.get('maximum_execution_launches')==1 and
  a.get('output_directories')=={'D3':'/outputs/D3'}, 'D3 admission or one-launch authorization differs')
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
 if test "$arm" = D3; then
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
PY
done
trap - EXIT
