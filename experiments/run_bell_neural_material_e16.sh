#!/usr/bin/env bash
# Exactly one approved cached-geometry paired full-field neural recovery, foreground only.
set -euo pipefail
: "${E16_SOURCE:?}" "${E16_BENCHMARK:?}" "${E16_PILOT:?}" "${E16_REFERENCE:?}" "${E16_FACTOR:?}" "${E16_MULTIVIEW:?}" "${E16_TWO_LIGHT:?}" "${E16_PHYSICAL:?}" "${E16_BATCH:?}" "${E16_AUTH:?}" "${E16_GPU:?}"
E16_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E16_COMMIT=$(git -C "$E16_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E16_SOURCE" status --porcelain)"
E16_FREEZE=$(sha256sum "$E16_SOURCE/docs/protocols/BELL_NEURAL_OBSERVATION_E16_V1.freeze.json" | cut -d' ' -f1)
E16_AUTH_SHA=$(sha256sum "$E16_AUTH" | cut -d' ' -f1)
python3 -B - "$E16_SOURCE" "$E16_AUTH" "$E16_BATCH" "$E16_COMMIT" "$E16_FREEZE" "$E16_IMAGE" "$E16_GPU" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from experiments.bell_neural_material_e16 import LIMITS,PROTOCOL
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
f=json.loads((Path(sys.argv[1])/'docs/protocols/BELL_NEURAL_OBSERVATION_E16_V1.freeze.json').read_text())
require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==sys.argv[4] and
 a.get('freeze_sha256')==sys.argv[5] and a.get('image_id')==sys.argv[6] and a.get('limits')==LIMITS and
 a.get('host_output_directory')==f['host_output_directory']==str(out) and
 a.get('output_directory')=='/outputs/neural' and a.get('selected_gpu')==sys.argv[7]=='0','Exact E16 approval required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh batch required')
PY
test "$(docker image inspect --format '{{.Id}}' "$E16_IMAGE")" = "$E16_IMAGE"
mkdir "$E16_BATCH"
python3 -B - "$E16_BATCH" "$E16_SOURCE" "$E16_BENCHMARK" "$E16_PILOT" "$E16_REFERENCE" "$E16_FACTOR" "$E16_MULTIVIEW" "$E16_TWO_LIGHT" "$E16_PHYSICAL" "$E16_COMMIT" "$E16_FREEZE" "$E16_AUTH_SHA" "$E16_GPU" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
 json.dump(dict(zip(['batch','source','benchmark','pilot','reference','factor','multiview','two_light','physical','commit','freeze_sha256','authorization_sha256','gpu'],sys.argv[1:])),f,indent=2);f.write('\n')
PY
E16_CONTAINER=$(basename "$E16_BATCH")
cleanup_owned() {
 test -s "$E16_BATCH/container.cid" || return 0
 E16_CID=$(cat "$E16_BATCH/container.cid")
 if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e16.run" }}' "$E16_CID" 2>/dev/null || true)" = "$E16_CONTAINER"; then
  docker rm --force "$E16_CID" >> "$E16_BATCH/cleanup.log" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 3600s \
 docker run --rm --pull=never --init --cidfile "$E16_BATCH/container.cid" \
 --name "$E16_CONTAINER" --label "nerofactor.e16.run=$E16_CONTAINER" \
 --user "$(id -u):$(id -g)" --gpus "device=$E16_GPU" --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
 --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env HOME=/outputs/neural/runtime --env XDG_CACHE_HOME=/outputs/neural/runtime/cache \
 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$E16_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$E16_BENCHMARK,dst=/inputs/v3,readonly" \
 --mount "type=bind,src=$E16_PILOT,dst=/inputs/e11d,readonly" \
 --mount "type=bind,src=$E16_REFERENCE,dst=/inputs/e11e,readonly" \
 --mount "type=bind,src=$E16_FACTOR,dst=/inputs/e11f,readonly" \
 --mount "type=bind,src=$E16_MULTIVIEW,dst=/inputs/e11g,readonly" \
 --mount "type=bind,src=$E16_TWO_LIGHT,dst=/inputs/e14,readonly" \
 --mount "type=bind,src=$E16_PHYSICAL,dst=/inputs/e15,readonly" \
 --mount "type=bind,src=$E16_AUTH,dst=/authorization.json,readonly" \
 --mount "type=bind,src=$E16_BATCH,dst=/outputs" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$E16_IMAGE" \
 --signal=TERM --kill-after=20s 3480s /usr/bin/python3 -I -B /workspace/run_bell_neural_material_e16.py \
 --benchmark /inputs/v3 --pilot /inputs/e11d --reference /inputs/e11e --factor /inputs/e11f \
 --multiview /inputs/e11g --two-light /inputs/e14 --physical /inputs/e15 --output /outputs/neural \
 --freeze /workspace/docs/protocols/BELL_NEURAL_OBSERVATION_E16_V1.freeze.json --freeze-sha256 "$E16_FREEZE" \
 --authorization /authorization.json --authorization-sha256 "$E16_AUTH_SHA" \
 --commit "$E16_COMMIT" --image-id "$E16_IMAGE" --execute > "$E16_BATCH/console.log" 2>&1
E16_EXIT=$?
set -e
printf '%s\n' "$E16_EXIT" > "$E16_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E16_BATCH/console.log"
exit "$E16_EXIT"
