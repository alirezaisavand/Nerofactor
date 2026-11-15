#!/usr/bin/env bash
# One foreground cached-visibility material-factor study after an exact researcher authorization; no rebuild/retry.
set -euo pipefail
: "${E11F_SOURCE:?}" "${E11F_BENCHMARK:?}" "${E11F_PILOT:?}" "${E11F_REFERENCE:?}" "${E11F_BATCH:?}" "${E11F_AUTH:?}" "${E11F_GPU:?}"
E11F_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E11F_COMMIT=$(git -C "$E11F_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E11F_SOURCE" status --porcelain)"
test "$(docker image inspect --format '{{.Id}}' "$E11F_IMAGE")" = "$E11F_IMAGE"
E11F_FREEZE=$(sha256sum "$E11F_SOURCE/docs/protocols/BELL_ANISOTROPY_E11F_V1.freeze.json" | cut -d' ' -f1)
E11F_AUTH_SHA=$(sha256sum "$E11F_AUTH" | cut -d' ' -f1)
python3 -B - "$E11F_SOURCE" "$E11F_AUTH" "$E11F_BATCH" "$E11F_COMMIT" "$E11F_FREEZE" "$E11F_IMAGE" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from experiments.bell_anisotropy_e11f import LIMITS,PROTOCOL
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
require(a.get('approved') is True and a.get('protocol')==PROTOCOL and
        a.get('host_output_directory')==str(out) and a.get('output_directory')=='/outputs/anisotropy' and
        a.get('commit')==sys.argv[4] and a.get('freeze_sha256')==sys.argv[5] and
        a.get('image_id')==sys.argv[6] and a.get('limits')==LIMITS,'Exact researcher material-factor authorization required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh material-factor batch required')
PY
mkdir "$E11F_BATCH"
python3 -B - "$E11F_BATCH" "$E11F_SOURCE" "$E11F_BENCHMARK" "$E11F_COMMIT" "$E11F_FREEZE" "$E11F_AUTH_SHA" "$E11F_GPU" "$E11F_PILOT" "$E11F_REFERENCE" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
    json.dump(dict(zip(['batch','source','benchmark','commit','freeze_sha256','authorization_sha256','gpu','pilot','reference'],sys.argv[1:])),f,indent=2);f.write('\n')
PY
E11F_CONTAINER=$(basename "$E11F_BATCH")
cleanup_owned() {
  test -s "$E11F_BATCH/container.cid" || return 0
  E11F_CID=$(cat "$E11F_BATCH/container.cid")
  if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e11f.run" }}' "$E11F_CID" 2>/dev/null || true)" = "$E11F_CONTAINER"; then
    docker rm --force "$E11F_CID" >> "$E11F_BATCH/cleanup.log" 2>&1
  fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 900s \
  docker run --rm --pull=never --init --cidfile "$E11F_BATCH/container.cid" \
  --name "$E11F_CONTAINER" --label "nerofactor.e11f.run=$E11F_CONTAINER" \
  --user "$(id -u):$(id -g)" --gpus "device=$E11F_GPU" --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
  --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
  --env HOME=/outputs/anisotropy/runtime --env XDG_CACHE_HOME=/outputs/anisotropy/runtime/cache \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$E11F_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$E11F_BENCHMARK,dst=/inputs/v3,readonly" \
  --mount "type=bind,src=$E11F_PILOT,dst=/inputs/e11d,readonly" \
  --mount "type=bind,src=$E11F_REFERENCE,dst=/inputs/e11e,readonly" \
  --mount "type=bind,src=$E11F_AUTH,dst=/authorization.json,readonly" \
  --mount "type=bind,src=$E11F_BATCH,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout "$E11F_IMAGE" \
  --signal=TERM --kill-after=20s 840s /usr/bin/python3 -I -B /workspace/run_bell_anisotropy_e11f.py \
  --benchmark /inputs/v3 --pilot /inputs/e11d --reference /inputs/e11e --output /outputs/anisotropy \
  --freeze /workspace/docs/protocols/BELL_ANISOTROPY_E11F_V1.freeze.json --freeze-sha256 "$E11F_FREEZE" \
  --authorization /authorization.json --authorization-sha256 "$E11F_AUTH_SHA" \
  --commit "$E11F_COMMIT" --image-id "$E11F_IMAGE" --execute > "$E11F_BATCH/console.log" 2>&1
E11F_EXIT=$?
set -e
printf '%s\n' "$E11F_EXIT" > "$E11F_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E11F_BATCH/console.log"
exit "$E11F_EXIT"
