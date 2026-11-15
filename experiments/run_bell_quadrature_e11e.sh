#!/usr/bin/env bash
# One foreground quadrature study after an exact researcher authorization; no rebuild/retry.
set -euo pipefail
: "${E11E_SOURCE:?}" "${E11E_BENCHMARK:?}" "${E11E_PILOT:?}" "${E11E_BATCH:?}" "${E11E_AUTH:?}" "${E11E_GPU:?}"
E11E_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E11E_COMMIT=$(git -C "$E11E_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E11E_SOURCE" status --porcelain)"
test "$(docker image inspect --format '{{.Id}}' "$E11E_IMAGE")" = "$E11E_IMAGE"
E11E_FREEZE=$(sha256sum "$E11E_SOURCE/docs/protocols/BELL_QUADRATURE_E11E_V1.freeze.json" | cut -d' ' -f1)
E11E_AUTH_SHA=$(sha256sum "$E11E_AUTH" | cut -d' ' -f1)
python3 -B - "$E11E_SOURCE" "$E11E_AUTH" "$E11E_BATCH" "$E11E_COMMIT" "$E11E_FREEZE" "$E11E_IMAGE" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from experiments.bell_quadrature_e11e import LIMITS,PROTOCOL
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
require(a.get('approved') is True and a.get('protocol')==PROTOCOL and
        a.get('host_output_directory')==str(out) and a.get('output_directory')=='/outputs/quadrature' and
        a.get('commit')==sys.argv[4] and a.get('freeze_sha256')==sys.argv[5] and
        a.get('image_id')==sys.argv[6] and a.get('limits')==LIMITS,'Exact researcher quadrature authorization required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh quadrature batch required')
PY
mkdir "$E11E_BATCH"
python3 -B - "$E11E_BATCH" "$E11E_SOURCE" "$E11E_BENCHMARK" "$E11E_COMMIT" "$E11E_FREEZE" "$E11E_AUTH_SHA" "$E11E_GPU" "$E11E_PILOT" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
    json.dump(dict(zip(['batch','source','benchmark','commit','freeze_sha256','authorization_sha256','gpu','pilot'],sys.argv[1:])),f,indent=2);f.write('\n')
PY
E11E_CONTAINER=$(basename "$E11E_BATCH")
cleanup_owned() {
  test -s "$E11E_BATCH/container.cid" || return 0
  E11E_CID=$(cat "$E11E_BATCH/container.cid")
  if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e11e.run" }}' "$E11E_CID" 2>/dev/null || true)" = "$E11E_CONTAINER"; then
    docker rm --force "$E11E_CID" >> "$E11E_BATCH/cleanup.log" 2>&1
  fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 900s \
  docker run --rm --pull=never --init --cidfile "$E11E_BATCH/container.cid" \
  --name "$E11E_CONTAINER" --label "nerofactor.e11e.run=$E11E_CONTAINER" \
  --user "$(id -u):$(id -g)" --gpus "device=$E11E_GPU" --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
  --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
  --env HOME=/outputs/quadrature/runtime --env XDG_CACHE_HOME=/outputs/quadrature/runtime/cache \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$E11E_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$E11E_BENCHMARK,dst=/inputs/v3,readonly" \
  --mount "type=bind,src=$E11E_PILOT,dst=/inputs/e11d,readonly" \
  --mount "type=bind,src=$E11E_AUTH,dst=/authorization.json,readonly" \
  --mount "type=bind,src=$E11E_BATCH,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout "$E11E_IMAGE" \
  --signal=TERM --kill-after=20s 840s /usr/bin/python3 -I -B /workspace/run_bell_quadrature_e11e.py \
  --benchmark /inputs/v3 --pilot /inputs/e11d --output /outputs/quadrature \
  --freeze /workspace/docs/protocols/BELL_QUADRATURE_E11E_V1.freeze.json --freeze-sha256 "$E11E_FREEZE" \
  --authorization /authorization.json --authorization-sha256 "$E11E_AUTH_SHA" \
  --commit "$E11E_COMMIT" --image-id "$E11E_IMAGE" --execute > "$E11E_BATCH/console.log" 2>&1
E11E_EXIT=$?
set -e
printf '%s\n' "$E11E_EXIT" > "$E11E_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E11E_BATCH/console.log"
exit "$E11E_EXIT"
