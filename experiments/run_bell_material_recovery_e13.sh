#!/usr/bin/env bash
# One foreground, sequential A/B/C batch. No retry/resume; each arm is a cold process.
set -euo pipefail
: "${E13_SOURCE:?}" "${E13_BATCH:?}" "${E13_AUTH:?}"
E13_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E13_BENCHMARK=/home/isavand/paper/evidence/bell-oracle-e11-v3-fea6430e2a644b679b0b35989569c240/benchmark
E13_ADMISSION=/home/isavand/paper/evidence/bell-oracle-e12-admission-v1/oracle-admission.json
E13_COMMIT=$(git -C "$E13_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E13_SOURCE" status --porcelain)"
E13_FREEZE=$(sha256sum "$E13_SOURCE/docs/protocols/BELL_THREE_ARM_RECOVERY_E13_V2.freeze.json" | cut -d' ' -f1)
E13_AUTH_SHA=$(sha256sum "$E13_AUTH" | cut -d' ' -f1)
python3 -I -B - "$E13_SOURCE" "$E13_AUTH" "$E13_BATCH" "$E13_COMMIT" "$E13_FREEZE" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from infrastructure.signatures import require
from experiments.bell_material_recovery_e13 import verify_freeze,verify_inputs,IMAGE,ADMISSION
f=verify_freeze(sys.argv[5]);a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
require(a.get('approved') is True and a.get('protocol')==f['protocol'] and a.get('commit')==sys.argv[4] and
 a.get('freeze_sha256')==sys.argv[5] and a.get('image_id')==IMAGE and a.get('admission_sha256')==ADMISSION and
 a.get('host_output_directory')==f['host_output_directory']==str(out) and a.get('run_ids')==f['run_ids'] and
 a.get('limits')==f['limits'] and a.get('selected_gpu')=='0','Exact E.13 authorization required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh E.13 batch required; no reuse')
verify_inputs(Path(f['host_benchmark']),f)
print('E.13 metadata/source/input preflight passed; zero native model calls')
PY
test "$(docker image inspect --format '{{.Id}}' "$E13_IMAGE")" = "$E13_IMAGE"
mkdir "$E13_BATCH"
cp --no-clobber "$E13_AUTH" "$E13_BATCH/authorization.json"
python3 -I -B - "$E13_BATCH" "$E13_COMMIT" "$E13_FREEZE" "$E13_AUTH_SHA" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
 json.dump(dict(batch=sys.argv[1],commit=sys.argv[2],freeze_sha256=sys.argv[3],authorization_sha256=sys.argv[4]),f,indent=2);f.write('\n')
with (Path(sys.argv[1])/'panel-status.json').open('x') as f:
 json.dump({v:{'status':'unexecuted'} for v in ('A_original','A_estimator_v1','A_corrected_core_v1')},f,indent=2);f.write('\n')
PY
record_arm() {
 python3 -I -B - "$E13_BATCH" "$E13_VARIANT" "$1" "${2:-}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'panel-status.json';value=json.loads(p.read_text())
value[sys.argv[2]]={'status':sys.argv[3],'exit_code':int(sys.argv[4]) if sys.argv[4] else None}
tmp=p.with_suffix('.next');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
PY
}
E13_ACTIVE_CID=
cleanup_owned() {
 test -n "$E13_ACTIVE_CID" && test -s "$E13_ACTIVE_CID" || return 0
 E13_CID=$(cat "$E13_ACTIVE_CID")
 if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e13.batch" }}' "$E13_CID" 2>/dev/null || true)" = "$(basename "$E13_BATCH")"; then
  docker rm --force "$E13_CID" >> "$E13_BATCH/cleanup.log" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
E13_INDEX=0
for E13_VARIANT in A_original A_estimator_v1 A_corrected_core_v1; do
 E13_RUN_ID=$(python3 -I -B - "$E13_SOURCE/docs/protocols/BELL_THREE_ARM_RECOVERY_E13_V2.freeze.json" "$E13_VARIANT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['run_ids'][sys.argv[2]])
PY
)
 E13_ARM_DIR="$E13_BATCH/$E13_VARIANT"
 mkdir "$E13_ARM_DIR"
 E13_ACTIVE_CID="$E13_BATCH/$E13_VARIANT.cid"
 E13_PARITY=()
 if test "$E13_INDEX" != 0; then
  E13_INITIAL="$E13_BATCH/A_original/initialization.json"
  test -s "$E13_BATCH/A_original/completion.json"
  E13_INITIAL_SHA=$(sha256sum "$E13_INITIAL" | cut -d' ' -f1)
  E13_PARITY+=(--mount "type=bind,src=$E13_INITIAL,dst=/inputs/initialization.json,readonly")
 fi
 E13_ARGS=()
 if test "$E13_INDEX" != 0; then
  E13_ARGS+=(--initial /inputs/initialization.json --initial-sha256 "$E13_INITIAL_SHA")
 fi
 record_arm running
 set +e
 timeout --foreground --signal=TERM --kill-after=30s 2100s \
 docker run --rm --pull=never --init --cidfile "$E13_ACTIVE_CID" \
 --name "$E13_RUN_ID" --label "nerofactor.e13.batch=$(basename "$E13_BATCH")" \
 --user "$(id -u):$(id -g)" --gpus device=0 --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
 --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$E13_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$E13_BENCHMARK/datasets,dst=/inputs/datasets,readonly" \
 --mount "type=bind,src=$E13_BENCHMARK/geometry,dst=/inputs/geometry,readonly" \
 --mount "type=bind,src=$E13_BENCHMARK/references,dst=/inputs/references,readonly" \
 --mount "type=bind,src=$(dirname "$E13_ADMISSION"),dst=/inputs/admission,readonly" \
 --mount type=bind,src=/home/isavand/paper/final_results,dst=/inputs/thesis,readonly \
 --mount "type=bind,src=$E13_AUTH,dst=/authorization.json,readonly" \
 --mount "type=bind,src=$E13_ARM_DIR,dst=/outputs" "${E13_PARITY[@]}" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$E13_IMAGE" \
 --signal=TERM --kill-after=20s 2040s /usr/bin/python3 -I -B /workspace/experiments/bell_material_recovery_e13.py \
 --variant "$E13_VARIANT" --inputs /inputs --thesis /inputs/thesis \
 --admission /inputs/admission/oracle-admission.json --output /outputs \
 --freeze-sha256 "$E13_FREEZE" --execute --authorization /authorization.json \
 --authorization-sha256 "$E13_AUTH_SHA" --commit "$E13_COMMIT" --run-id "$E13_RUN_ID" \
 "${E13_ARGS[@]}" > "$E13_BATCH/$E13_VARIANT.console.log" 2>&1
 E13_EXIT=$?
 set -e
 printf '%s\n' "$E13_EXIT" > "$E13_BATCH/$E13_VARIANT.exit-code.txt"
 cleanup_owned
 E13_ACTIVE_CID=
 if test "$E13_EXIT" != 0; then
  record_arm failed_or_incomplete "$E13_EXIT"
  printf 'Stopped on failed/incomplete arm %s. Remaining arms are unexecuted.\n' "$E13_VARIANT"
  tail -60 "$E13_BATCH/$E13_VARIANT.console.log"
  exit "$E13_EXIT"
 fi
 test -s "$E13_ARM_DIR/completion.json"
 record_arm exited_zero_pending_offline_verification "$E13_EXIT"
 E13_INDEX=$((E13_INDEX+1))
done
printf 'All three native arms exited zero. File-only report/hash verification remains required.\n'
