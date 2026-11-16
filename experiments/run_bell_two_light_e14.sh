#!/usr/bin/env bash
# Exactly one approved cached-geometry second-light study, foreground only.
set -euo pipefail
: "${E14_SOURCE:?}" "${E14_BENCHMARK:?}" "${E14_PILOT:?}" "${E14_REFERENCE:?}" "${E14_FACTOR:?}" "${E14_MULTIVIEW:?}" "${E14_BATCH:?}" "${E14_AUTH:?}" "${E14_GPU:?}"
E14_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E14_COMMIT=$(git -C "$E14_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E14_SOURCE" status --porcelain)"
E14_FREEZE=$(sha256sum "$E14_SOURCE/docs/protocols/BELL_TWO_LIGHT_OBSERVABILITY_E14_V1.freeze.json" | cut -d' ' -f1)
E14_AUTH_SHA=$(sha256sum "$E14_AUTH" | cut -d' ' -f1)
python3 -B - "$E14_SOURCE" "$E14_AUTH" "$E14_BATCH" "$E14_COMMIT" "$E14_FREEZE" "$E14_IMAGE" "$E14_GPU" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from experiments.bell_two_light_e14 import LIMITS,PROTOCOL
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
f=json.loads((Path(sys.argv[1])/'docs/protocols/BELL_TWO_LIGHT_OBSERVABILITY_E14_V1.freeze.json').read_text())
require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==sys.argv[4] and
 a.get('freeze_sha256')==sys.argv[5] and a.get('image_id')==sys.argv[6] and a.get('limits')==LIMITS and
 a.get('host_output_directory')==f['host_output_directory']==str(out) and
 a.get('output_directory')=='/outputs/two-light' and a.get('selected_gpu')==sys.argv[7]=='0','Exact E14 approval required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh batch required')
PY
test "$(docker image inspect --format '{{.Id}}' "$E14_IMAGE")" = "$E14_IMAGE"
mkdir "$E14_BATCH"
python3 -B - "$E14_BATCH" "$E14_SOURCE" "$E14_BENCHMARK" "$E14_PILOT" "$E14_REFERENCE" "$E14_FACTOR" "$E14_MULTIVIEW" "$E14_COMMIT" "$E14_FREEZE" "$E14_AUTH_SHA" "$E14_GPU" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
 json.dump(dict(zip(['batch','source','benchmark','pilot','reference','factor','multiview','commit','freeze_sha256','authorization_sha256','gpu'],sys.argv[1:])),f,indent=2);f.write('\n')
PY
E14_CONTAINER=$(basename "$E14_BATCH")
cleanup_owned() {
 test -s "$E14_BATCH/container.cid" || return 0
 E14_CID=$(cat "$E14_BATCH/container.cid")
 if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e14.run" }}' "$E14_CID" 2>/dev/null || true)" = "$E14_CONTAINER"; then
  docker rm --force "$E14_CID" >> "$E14_BATCH/cleanup.log" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 900s \
 docker run --rm --pull=never --init --cidfile "$E14_BATCH/container.cid" \
 --name "$E14_CONTAINER" --label "nerofactor.e14.run=$E14_CONTAINER" \
 --user "$(id -u):$(id -g)" --gpus "device=$E14_GPU" --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
 --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env HOME=/outputs/two-light/runtime --env XDG_CACHE_HOME=/outputs/two-light/runtime/cache \
 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$E14_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$E14_BENCHMARK,dst=/inputs/v3,readonly" \
 --mount "type=bind,src=$E14_PILOT,dst=/inputs/e11d,readonly" \
 --mount "type=bind,src=$E14_REFERENCE,dst=/inputs/e11e,readonly" \
 --mount "type=bind,src=$E14_FACTOR,dst=/inputs/e11f,readonly" \
 --mount "type=bind,src=$E14_MULTIVIEW,dst=/inputs/e11g,readonly" \
 --mount "type=bind,src=$E14_AUTH,dst=/authorization.json,readonly" \
 --mount "type=bind,src=$E14_BATCH,dst=/outputs" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$E14_IMAGE" \
 --signal=TERM --kill-after=20s 840s /usr/bin/python3 -I -B /workspace/run_bell_two_light_e14.py \
 --benchmark /inputs/v3 --pilot /inputs/e11d --reference /inputs/e11e --factor /inputs/e11f \
 --multiview /inputs/e11g --output /outputs/two-light \
 --freeze /workspace/docs/protocols/BELL_TWO_LIGHT_OBSERVABILITY_E14_V1.freeze.json --freeze-sha256 "$E14_FREEZE" \
 --authorization /authorization.json --authorization-sha256 "$E14_AUTH_SHA" \
 --commit "$E14_COMMIT" --image-id "$E14_IMAGE" --execute > "$E14_BATCH/console.log" 2>&1
E14_EXIT=$?
set -e
printf '%s\n' "$E14_EXIT" > "$E14_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E14_BATCH/console.log"
exit "$E14_EXIT"
