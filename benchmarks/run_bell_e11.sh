#!/usr/bin/env bash
# One foreground launch, only after explicit authorization. No builds or retries.
set -euo pipefail
: "${E11_SOURCE:?}" "${E11_CAMERAS:?}" "${E11_MESH:?}" "${E11_BATCH:?}" "${E11_AUTH:?}" "${E11_GPU:?}"
E11_IMAGE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
E11_COMMIT=$(git -C "$E11_SOURCE" rev-parse HEAD)
test "$(git -C "$E11_SOURCE" branch --show-current)" = feat/neural-material-early-step-e2a
test -z "$(git -C "$E11_SOURCE" status --porcelain)"
test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$E11_IMAGE"
E11_FREEZE=$(sha256sum "$E11_SOURCE/docs/protocols/BELL_ORACLE_GENERATION_E11_V1.freeze.json" | cut -d' ' -f1)
E11_AUTH_SHA=$(sha256sum "$E11_AUTH" | cut -d' ' -f1)
# Data-only host preflight. No CUDA, native imports, source camera unpickling or rendering.
python3 -B - "$E11_SOURCE" "$E11_AUTH" "$E11_AUTH_SHA" "$E11_FREEZE" "$E11_COMMIT" "$E11_BATCH" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from benchmarks.bell_e11 import verify_authorization
from infrastructure.signatures import require
a=verify_authorization(Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5],Path('/outputs/benchmark'))
require(a.get('host_output_directory')==sys.argv[6],'Authorization is bound to another host output directory')
p=Path(sys.argv[6]);require(p.is_absolute() and p.resolve()==p and not p.exists(),'Fresh unaliased evidence batch required')
PY
mkdir "$E11_BATCH"
python3 -B - "$E11_BATCH" "$E11_SOURCE" "$E11_CAMERAS" "$E11_MESH" "$E11_AUTH" "$E11_AUTH_SHA" "$E11_FREEZE" "$E11_COMMIT" "$E11_IMAGE" "$E11_GPU" <<'PYMETA'
import json,sys
from pathlib import Path
keys=['host_output_directory','source','source_cameras','source_mesh','authorization','authorization_sha256','freeze_sha256','commit','image_id','gpu_selection']
with (Path(sys.argv[1])/'launch.json').open('x') as f:json.dump(dict(zip(keys,sys.argv[1:])),f,indent=2);f.write('\n')
PYMETA
E11_CONTAINER=$(basename "$E11_BATCH")
cleanup_owned() {
  test -s "$E11_BATCH/container.cid" || return 0
  E11_OWNED_CID=$(cat "$E11_BATCH/container.cid")
  if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e11.run" }}' "$E11_OWNED_CID" 2>/dev/null || true)" = "$E11_CONTAINER"; then
    docker rm --force "$E11_OWNED_CID" >> "$E11_BATCH/container-cleanup.log" 2>&1
  fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 7200s \
  docker run --rm --pull=never --init --cidfile "$E11_BATCH/container.cid" --name "$E11_CONTAINER" --label "nerofactor.e11.run=$E11_CONTAINER" \
  --user "$(id -u):$(id -g)" --gpus "device=$E11_GPU" --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
  --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
  --env HOME=/outputs/benchmark/runtime --env XDG_CACHE_HOME=/outputs/benchmark/runtime/cache \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$E11_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$E11_CAMERAS,dst=/inputs/bell,readonly" \
  --mount "type=bind,src=$E11_MESH,dst=/inputs/mesh.ply,readonly" \
  --mount "type=bind,src=$E11_AUTH,dst=/authorization.json,readonly" \
  --mount "type=bind,src=$E11_BATCH,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout "$E11_IMAGE" \
  --signal=TERM --kill-after=20s 7140s /usr/bin/python3 -I -B /workspace/generate_bell_benchmark_e11.py \
  --mesh /inputs/mesh.ply --cameras /inputs/bell \
  --freeze /workspace/docs/protocols/BELL_ORACLE_GENERATION_E11_V1.freeze.json --freeze-sha256 "$E11_FREEZE" \
  --authorization /authorization.json --authorization-sha256 "$E11_AUTH_SHA" \
  --commit "$E11_COMMIT" --image-id "$E11_IMAGE" --output /outputs/benchmark --execute \
  > "$E11_BATCH/console.log" 2>&1
E11_EXIT=$?
set -e
printf '%s\n' "$E11_EXIT" > "$E11_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E11_BATCH/console.log"
exit "$E11_EXIT"
