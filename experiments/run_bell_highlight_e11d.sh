#!/usr/bin/env bash
# One foreground pilot after an exact researcher authorization; no rebuild/retry.
set -euo pipefail
: "${E11D_SOURCE:?}" "${E11D_BENCHMARK:?}" "${E11D_BATCH:?}" "${E11D_AUTH:?}" "${E11D_GPU:?}"
E11D_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E11D_COMMIT=$(git -C "$E11D_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E11D_SOURCE" status --porcelain)"
test "$(docker image inspect --format '{{.Id}}' "$E11D_IMAGE")" = "$E11D_IMAGE"
E11D_FREEZE=$(sha256sum "$E11D_SOURCE/docs/protocols/BELL_HIGHLIGHT_PILOT_E11D_V1.freeze.json" | cut -d' ' -f1)
E11D_AUTH_SHA=$(sha256sum "$E11D_AUTH" | cut -d' ' -f1)
python3 -B - "$E11D_SOURCE" "$E11D_AUTH" "$E11D_BATCH" "$E11D_COMMIT" "$E11D_FREEZE" "$E11D_IMAGE" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from experiments.bell_highlight_pilot_e11d import LIMITS,PROTOCOL
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
require(a.get('approved') is True and a.get('protocol')==PROTOCOL and
        a.get('host_output_directory')==str(out) and a.get('output_directory')=='/outputs/pilot' and
        a.get('commit')==sys.argv[4] and a.get('freeze_sha256')==sys.argv[5] and
        a.get('image_id')==sys.argv[6] and a.get('limits')==LIMITS,'Exact researcher pilot authorization required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh pilot batch required')
PY
mkdir "$E11D_BATCH"
python3 -B - "$E11D_BATCH" "$E11D_SOURCE" "$E11D_BENCHMARK" "$E11D_COMMIT" "$E11D_FREEZE" "$E11D_AUTH_SHA" "$E11D_GPU" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
    json.dump(dict(zip(['batch','source','benchmark','commit','freeze_sha256','authorization_sha256','gpu'],sys.argv[1:])),f,indent=2);f.write('\n')
PY
E11D_CONTAINER=$(basename "$E11D_BATCH")
cleanup_owned() {
  test -s "$E11D_BATCH/container.cid" || return 0
  E11D_CID=$(cat "$E11D_BATCH/container.cid")
  if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e11d.run" }}' "$E11D_CID" 2>/dev/null || true)" = "$E11D_CONTAINER"; then
    docker rm --force "$E11D_CID" >> "$E11D_BATCH/cleanup.log" 2>&1
  fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 1800s \
  docker run --rm --pull=never --init --cidfile "$E11D_BATCH/container.cid" \
  --name "$E11D_CONTAINER" --label "nerofactor.e11d.run=$E11D_CONTAINER" \
  --user "$(id -u):$(id -g)" --gpus "device=$E11D_GPU" --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
  --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
  --env HOME=/outputs/pilot/runtime --env XDG_CACHE_HOME=/outputs/pilot/runtime/cache \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$E11D_SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$E11D_BENCHMARK,dst=/inputs/v3,readonly" \
  --mount "type=bind,src=$E11D_AUTH,dst=/authorization.json,readonly" \
  --mount "type=bind,src=$E11D_BATCH,dst=/outputs" \
  --workdir /workspace --entrypoint /usr/bin/timeout "$E11D_IMAGE" \
  --signal=TERM --kill-after=20s 1740s /usr/bin/python3 -I -B /workspace/run_bell_highlight_pilot_e11d.py \
  --root /inputs/v3 --output /outputs/pilot \
  --freeze /workspace/docs/protocols/BELL_HIGHLIGHT_PILOT_E11D_V1.freeze.json --freeze-sha256 "$E11D_FREEZE" \
  --authorization /authorization.json --authorization-sha256 "$E11D_AUTH_SHA" \
  --commit "$E11D_COMMIT" --image-id "$E11D_IMAGE" --execute > "$E11D_BATCH/console.log" 2>&1
E11D_EXIT=$?
set -e
printf '%s\n' "$E11D_EXIT" > "$E11D_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E11D_BATCH/console.log"
exit "$E11D_EXIT"
