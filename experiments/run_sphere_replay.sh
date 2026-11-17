#!/usr/bin/env bash
# One explicitly approved fresh sphere reconstruction. Keep this foreground.
set -euo pipefail
: "${SPHERE_SOURCE:?}" "${SPHERE_DATASETS:?}" "${SPHERE_THESIS:?}" "${SPHERE_BATCH:?}" "${SPHERE_AUTH:?}" "${SPHERE_GPU:?}"
SPHERE_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
SPHERE_COMMIT=$(git -C "$SPHERE_SOURCE" rev-parse HEAD)
test -z "$(git -C "$SPHERE_SOURCE" status --porcelain)"
SPHERE_FREEZE=$(sha256sum "$SPHERE_SOURCE/docs/protocols/SPHERE_CORRECTED_REPLAY_V2.freeze.json" | cut -d' ' -f1)
SPHERE_AUTH_SHA=$(sha256sum "$SPHERE_AUTH" | cut -d' ' -f1)
python3 -B - "$SPHERE_SOURCE" "$SPHERE_AUTH" "$SPHERE_AUTH_SHA" "$SPHERE_BATCH" "$SPHERE_COMMIT" "$SPHERE_FREEZE" "$SPHERE_DATASETS" "$SPHERE_THESIS" "$SPHERE_GPU" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_sphere_replay import verify_freeze,verify_authorization
from infrastructure.sphere_replay import input_records
from infrastructure.signatures import require
f=verify_freeze(Path(sys.argv[1])/'docs/protocols/SPHERE_CORRECTED_REPLAY_V2.freeze.json',sys.argv[6])
a=verify_authorization(Path(sys.argv[2]),sys.argv[3],f,sys.argv[6],sys.argv[5])
batch=Path(sys.argv[4]);dataset=Path(sys.argv[7]);thesis=Path(sys.argv[8])
require(str(batch)==f['host_output_directory'] and batch.is_absolute() and batch.resolve()==batch and
        not batch.exists(),'Fresh exactly frozen output directory required')
require(sys.argv[9]=='0' and input_records(dataset,thesis/'meshes/sphere_shape-300000.ply',thesis)==f['inputs'],
        'Selected GPU or original sphere input identities differ')
PY
test "$(docker image inspect --format '{{.Id}}' "$SPHERE_IMAGE")" = "$SPHERE_IMAGE"
mkdir "$SPHERE_BATCH"
cp -- "$SPHERE_AUTH" "$SPHERE_BATCH/authorization.json"
cp -- "$SPHERE_SOURCE/docs/protocols/SPHERE_CORRECTED_REPLAY_V2.freeze.json" "$SPHERE_BATCH/freeze.json"
python3 -B - "$SPHERE_BATCH" "$SPHERE_COMMIT" "$SPHERE_FREEZE" "$SPHERE_AUTH_SHA" "$SPHERE_IMAGE" "$SPHERE_GPU" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
 json.dump(dict(zip(('batch','commit','freeze_sha256','authorization_sha256','image_id','gpu'),sys.argv[1:])),f,indent=2);f.write('\n')
PY
SPHERE_CONTAINER=$(basename "$SPHERE_BATCH")
cleanup_owned() {
 test -s "$SPHERE_BATCH/container.cid" || return 0
 SPHERE_CID=$(cat "$SPHERE_BATCH/container.cid")
 if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.sphere.run" }}' "$SPHERE_CID" 2>/dev/null || true)" = "$SPHERE_CONTAINER"; then
  docker rm --force "$SPHERE_CID" >> "$SPHERE_BATCH/cleanup.log" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 86400s \
 docker run --rm --pull=never --init --cidfile "$SPHERE_BATCH/container.cid" \
 --name "$SPHERE_CONTAINER" --label "nerofactor.sphere.run=$SPHERE_CONTAINER" \
 --user "$(id -u):$(id -g)" --gpus "device=$SPHERE_GPU" --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=128g --memory-swap=128g --cpus=4 --pids-limit=256 \
 --shm-size=4g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env HOME=/outputs/runtime --env XDG_CACHE_HOME=/outputs/runtime/cache \
 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$SPHERE_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$SPHERE_DATASETS,dst=/inputs/datasets/nerf_synthetic,readonly" \
 --mount "type=bind,src=$SPHERE_THESIS,dst=/inputs/thesis,readonly" \
 --mount "type=bind,src=$SPHERE_AUTH,dst=/authorization.json,readonly" \
 --mount "type=bind,src=$SPHERE_BATCH,dst=/outputs" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$SPHERE_IMAGE" \
 --signal=TERM --kill-after=20s 85800s /usr/bin/python3 -I -B /workspace/run_sphere_replay.py \
 --datasets /inputs/datasets/nerf_synthetic --thesis /inputs/thesis \
 --mesh /inputs/thesis/meshes/sphere_shape-300000.ply --output /outputs/sphere \
 --freeze /workspace/docs/protocols/SPHERE_CORRECTED_REPLAY_V2.freeze.json --freeze-sha256 "$SPHERE_FREEZE" \
 --authorization /authorization.json --authorization-sha256 "$SPHERE_AUTH_SHA" \
 --commit "$SPHERE_COMMIT" --image-id "$SPHERE_IMAGE" --execute > "$SPHERE_BATCH/console.log" 2>&1
SPHERE_EXIT=$?
set -e
printf '%s\n' "$SPHERE_EXIT" > "$SPHERE_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
tail -n 60 "$SPHERE_BATCH/console.log"
exit "$SPHERE_EXIT"
