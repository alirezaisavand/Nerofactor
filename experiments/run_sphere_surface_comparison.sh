#!/usr/bin/env bash
set -euo pipefail
: "${SURFACE_SOURCE:?}" "${SURFACE_DATASETS:?}" "${SURFACE_THESIS:?}" "${SURFACE_BATCH:?}" "${SURFACE_AUTH:?}" "${SURFACE_GPU:?}"
if test "${1:-}" != --supervised; then
 exec timeout --foreground --signal=TERM --kill-after=30s 3600s bash "$0" --supervised
fi
SURFACE_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
SURFACE_COMMIT=$(git -C "$SURFACE_SOURCE" rev-parse HEAD)
test -z "$(git -C "$SURFACE_SOURCE" status --porcelain)"
SURFACE_FREEZE=$(sha256sum "$SURFACE_SOURCE/docs/protocols/SPHERE_SURFACE_COMPARISON_V2.freeze.json" | cut -d' ' -f1)
SURFACE_AUTH_SHA=$(sha256sum "$SURFACE_AUTH" | cut -d' ' -f1)
python3 -B - "$SURFACE_SOURCE" "$SURFACE_BATCH" "$SURFACE_AUTH" "$SURFACE_AUTH_SHA" "$SURFACE_FREEZE" "$SURFACE_COMMIT" "$SURFACE_GPU" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_sphere_surface_experiment import preflight,authorize
from infrastructure.signatures import require
f=preflight(Path(sys.argv[1])/'docs/protocols/SPHERE_SURFACE_COMPARISON_V2.freeze.json',sys.argv[5])
authorize(Path(sys.argv[3]),sys.argv[4],f,sys.argv[5],sys.argv[6])
out=Path(sys.argv[2]);require(str(out)==f['host_output_directory'] and out.is_absolute() and out.resolve()==out and not out.exists() and sys.argv[7]=='0','Fresh exact batch/GPU required')
from infrastructure.stage_sources import digest
c=Path(f['initial_checkpoint']['host_path']);require(c.is_absolute() and c.resolve()==c and digest(c)==f['initial_checkpoint']['sha256'],'Selected initial checkpoint changed')
PY
test "$(docker image inspect --format '{{.Id}}' "$SURFACE_IMAGE")" = "$SURFACE_IMAGE"
SURFACE_PRIOR_CID=$(python3 -B - "$SURFACE_SOURCE/docs/protocols/SPHERE_SURFACE_COMPARISON_V2.freeze.json" <<'PY'
import json,re,sys
value=json.load(open(sys.argv[1]))['previous_container_id']
assert re.fullmatch('[a-f0-9]{64}',value)
print(value)
PY
)
SURFACE_RUNNING=$(docker ps --no-trunc --format '{{.ID}}')
while IFS= read -r SURFACE_CID; do
 if test "$SURFACE_CID" = "$SURFACE_PRIOR_CID"; then
  echo 'Previous sphere container is still running; stop for researcher review.' >&2
  exit 1
 fi
done <<< "$SURFACE_RUNNING"
SURFACE_INITIAL_CHECKPOINT=$(python3 -B - "$SURFACE_SOURCE/docs/protocols/SPHERE_SURFACE_COMPARISON_V2.freeze.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['initial_checkpoint']['host_path'])
PY
)
mkdir "$SURFACE_BATCH"
cp -- "$SURFACE_AUTH" "$SURFACE_BATCH/authorization.json"
cp -- "$SURFACE_SOURCE/docs/protocols/SPHERE_SURFACE_COMPARISON_V2.freeze.json" "$SURFACE_BATCH/freeze.json"
SURFACE_CURRENT=
cleanup_owned() {
 test -n "$SURFACE_CURRENT" && test -s "$SURFACE_CURRENT/container.cid" || return 0
 SURFACE_CID=$(cat "$SURFACE_CURRENT/container.cid")
 if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.surface.batch" }}' "$SURFACE_CID" 2>/dev/null || true)" = "$(basename "$SURFACE_BATCH")"; then
  docker rm --force "$SURFACE_CID" >> "$SURFACE_CURRENT/cleanup.log" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for SURFACE_ARM in surface_chart_control_v1 surface_world_q_v1; do
 SURFACE_CURRENT="$SURFACE_BATCH/$SURFACE_ARM"
 mkdir "$SURFACE_CURRENT"
 SURFACE_CONTROL_MOUNT=()
 SURFACE_CONTROL_ARGS=()
 if test "$SURFACE_ARM" = surface_world_q_v1; then
  SURFACE_CONTROL_MOUNT=(--mount "type=bind,src=$SURFACE_BATCH/surface_chart_control_v1/run/surface-run.json,dst=/inputs/control.json,readonly")
  SURFACE_CONTROL_ARGS=(--control /inputs/control.json)
 fi
 set +e
 timeout --foreground --signal=TERM --kill-after=30s 1740s \
 docker run --rm --pull=never --init --cidfile "$SURFACE_CURRENT/container.cid" \
 --label "nerofactor.surface.batch=$(basename "$SURFACE_BATCH")" \
 --user "$(id -u):$(id -g)" --gpus "device=$SURFACE_GPU" --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=128g --memory-swap=128g --cpus=4 --pids-limit=256 \
 --shm-size=4g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env HOME=/outputs/runtime --env XDG_CACHE_HOME=/outputs/runtime/cache \
 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$SURFACE_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$SURFACE_DATASETS,dst=/inputs/datasets/nerf_synthetic,readonly" \
 --mount "type=bind,src=$SURFACE_THESIS,dst=/inputs/thesis,readonly" \
 --mount "type=bind,src=$SURFACE_AUTH,dst=/authorization.json,readonly" \
 --mount "type=bind,src=$SURFACE_INITIAL_CHECKPOINT,dst=/inputs/initial.pth,readonly" \
 --mount "type=bind,src=$SURFACE_CURRENT,dst=/outputs" "${SURFACE_CONTROL_MOUNT[@]}" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$SURFACE_IMAGE" \
 --signal=TERM --kill-after=20s 1680s /usr/bin/python3 -I -B /workspace/run_sphere_surface_experiment.py \
 --datasets /inputs/datasets/nerf_synthetic --thesis /inputs/thesis --output /outputs/run --checkpoint /inputs/initial.pth \
 --arm "$SURFACE_ARM" "${SURFACE_CONTROL_ARGS[@]}" --execute \
 --freeze /workspace/docs/protocols/SPHERE_SURFACE_COMPARISON_V2.freeze.json --freeze-sha256 "$SURFACE_FREEZE" \
 --commit "$SURFACE_COMMIT" --authorization /authorization.json --authorization-sha256 "$SURFACE_AUTH_SHA" \
 > "$SURFACE_CURRENT/console.log" 2>&1
 SURFACE_EXIT=$?
 set -e
 printf '%s\n' "$SURFACE_EXIT" > "$SURFACE_CURRENT/exit-code.txt"
 cleanup_owned
 tail -n 30 "$SURFACE_CURRENT/console.log"
 test "$SURFACE_EXIT" = 0 || exit "$SURFACE_EXIT"
 python3 -B - "$SURFACE_SOURCE" "$SURFACE_CURRENT/run/surface-run.json" "$SURFACE_ARM" "$SURFACE_FREEZE" "$SURFACE_COMMIT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_sphere_surface_experiment import completed_arm
completed_arm(Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5])
PY
done
