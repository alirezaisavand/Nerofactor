#!/usr/bin/env bash
# One approved batch: two new Q-retention histories, foreground only.
set -euo pipefail
: "${E17_SOURCE:?}" "${E17_BENCHMARK:?}" "${E17_PILOT:?}" "${E17_REFERENCE:?}" "${E17_FACTOR:?}" "${E17_MULTIVIEW:?}" "${E17_TWO_LIGHT:?}" "${E17_PHYSICAL:?}" "${E17_NEURAL:?}" "${E17_BATCH:?}" "${E17_AUTH:?}" "${E17_GPU:?}"
E17_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E17_COMMIT=$(git -C "$E17_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E17_SOURCE" status --porcelain)"
E17_FREEZE=$(sha256sum "$E17_SOURCE/docs/protocols/BELL_NEURAL_RETENTION_E17_V1.freeze.json" | cut -d' ' -f1)
E17_AUTH_SHA=$(sha256sum "$E17_AUTH" | cut -d' ' -f1)
python3 -B - "$E17_SOURCE" "$E17_AUTH" "$E17_BATCH" "$E17_COMMIT" "$E17_FREEZE" "$E17_IMAGE" "$E17_GPU" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from experiments.bell_neural_retention_e17 import LIMITS,PROTOCOL
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
f=json.loads((Path(sys.argv[1])/'docs/protocols/BELL_NEURAL_RETENTION_E17_V1.freeze.json').read_text())
require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==sys.argv[4] and
 a.get('freeze_sha256')==sys.argv[5] and a.get('image_id')==sys.argv[6] and a.get('limits')==LIMITS and
 a.get('host_output_directory')==f['host_output_directory']==str(out) and
 a.get('output_directory')=='/outputs/retention' and a.get('selected_gpu')==sys.argv[7]=='0','Exact E17 approval required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh batch required')
PY
test "$(docker image inspect --format '{{.Id}}' "$E17_IMAGE")" = "$E17_IMAGE"
mkdir "$E17_BATCH"
python3 -B - "$E17_BATCH" "$E17_SOURCE" "$E17_BENCHMARK" "$E17_PILOT" "$E17_REFERENCE" "$E17_FACTOR" "$E17_MULTIVIEW" "$E17_TWO_LIGHT" "$E17_PHYSICAL" "$E17_NEURAL" "$E17_COMMIT" "$E17_FREEZE" "$E17_AUTH_SHA" "$E17_GPU" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
 json.dump(dict(zip(['batch','source','benchmark','pilot','reference','factor','multiview','two_light','physical','neural','commit','freeze_sha256','authorization_sha256','gpu'],sys.argv[1:])),f,indent=2);f.write('\n')
PY
E17_CONTAINER=$(basename "$E17_BATCH")
cleanup_owned() {
 test -s "$E17_BATCH/container.cid" || return 0
 E17_CID=$(cat "$E17_BATCH/container.cid")
 if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e17.run" }}' "$E17_CID" 2>/dev/null || true)" = "$E17_CONTAINER"; then
  docker rm --force "$E17_CID" >> "$E17_BATCH/cleanup.log" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 1800s \
 docker run --rm --pull=never --init --cidfile "$E17_BATCH/container.cid" \
 --name "$E17_CONTAINER" --label "nerofactor.e17.run=$E17_CONTAINER" \
 --user "$(id -u):$(id -g)" --gpus "device=$E17_GPU" --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
 --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env HOME=/outputs/retention/runtime --env XDG_CACHE_HOME=/outputs/retention/runtime/cache \
 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$E17_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$E17_BENCHMARK,dst=/inputs/v3,readonly" \
 --mount "type=bind,src=$E17_PILOT,dst=/inputs/e11d,readonly" \
 --mount "type=bind,src=$E17_REFERENCE,dst=/inputs/e11e,readonly" \
 --mount "type=bind,src=$E17_FACTOR,dst=/inputs/e11f,readonly" \
 --mount "type=bind,src=$E17_MULTIVIEW,dst=/inputs/e11g,readonly" \
 --mount "type=bind,src=$E17_TWO_LIGHT,dst=/inputs/e14,readonly" \
 --mount "type=bind,src=$E17_PHYSICAL,dst=/inputs/e15,readonly" \
 --mount "type=bind,src=$E17_NEURAL,dst=/inputs/e16,readonly" \
 --mount "type=bind,src=$E17_AUTH,dst=/authorization.json,readonly" \
 --mount "type=bind,src=$E17_BATCH,dst=/outputs" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$E17_IMAGE" \
 --signal=TERM --kill-after=20s 1680s /usr/bin/python3 -I -B /workspace/run_bell_neural_retention_e17.py \
 --benchmark /inputs/v3 --pilot /inputs/e11d --reference /inputs/e11e --factor /inputs/e11f \
 --multiview /inputs/e11g --two-light /inputs/e14 --physical /inputs/e15 --neural /inputs/e16 --output /outputs/retention \
 --freeze /workspace/docs/protocols/BELL_NEURAL_RETENTION_E17_V1.freeze.json --freeze-sha256 "$E17_FREEZE" \
 --authorization /authorization.json --authorization-sha256 "$E17_AUTH_SHA" \
 --commit "$E17_COMMIT" --image-id "$E17_IMAGE" --execute > "$E17_BATCH/console.log" 2>&1
E17_EXIT=$?
set -e
printf '%s\n' "$E17_EXIT" > "$E17_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E17_BATCH/console.log"
exit "$E17_EXIT"
