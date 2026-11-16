#!/usr/bin/env bash
# Exactly one approved cached-geometry physical recovery panel, foreground only.
set -euo pipefail
: "${E15_SOURCE:?}" "${E15_BENCHMARK:?}" "${E15_PILOT:?}" "${E15_REFERENCE:?}" "${E15_FACTOR:?}" "${E15_MULTIVIEW:?}" "${E15_TWO_LIGHT:?}" "${E15_BATCH:?}" "${E15_AUTH:?}" "${E15_GPU:?}"
E15_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E15_COMMIT=$(git -C "$E15_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E15_SOURCE" status --porcelain)"
E15_FREEZE=$(sha256sum "$E15_SOURCE/docs/protocols/BELL_PHYSICAL_RECOVERY_E15_V1.freeze.json" | cut -d' ' -f1)
E15_AUTH_SHA=$(sha256sum "$E15_AUTH" | cut -d' ' -f1)
python3 -B - "$E15_SOURCE" "$E15_AUTH" "$E15_BATCH" "$E15_COMMIT" "$E15_FREEZE" "$E15_IMAGE" "$E15_GPU" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from experiments.bell_material_inverse_e15 import LIMITS,PROTOCOL
from infrastructure.signatures import require
a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
f=json.loads((Path(sys.argv[1])/'docs/protocols/BELL_PHYSICAL_RECOVERY_E15_V1.freeze.json').read_text())
require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==sys.argv[4] and
 a.get('freeze_sha256')==sys.argv[5] and a.get('image_id')==sys.argv[6] and a.get('limits')==LIMITS and
 a.get('host_output_directory')==f['host_output_directory']==str(out) and
 a.get('output_directory')=='/outputs/recovery' and a.get('selected_gpu')==sys.argv[7]=='0','Exact E15 approval required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh batch required')
PY
test "$(docker image inspect --format '{{.Id}}' "$E15_IMAGE")" = "$E15_IMAGE"
mkdir "$E15_BATCH"
python3 -B - "$E15_BATCH" "$E15_SOURCE" "$E15_BENCHMARK" "$E15_PILOT" "$E15_REFERENCE" "$E15_FACTOR" "$E15_MULTIVIEW" "$E15_TWO_LIGHT" "$E15_COMMIT" "$E15_FREEZE" "$E15_AUTH_SHA" "$E15_GPU" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as f:
 json.dump(dict(zip(['batch','source','benchmark','pilot','reference','factor','multiview','two_light','commit','freeze_sha256','authorization_sha256','gpu'],sys.argv[1:])),f,indent=2);f.write('\n')
PY
E15_CONTAINER=$(basename "$E15_BATCH")
cleanup_owned() {
 test -s "$E15_BATCH/container.cid" || return 0
 E15_CID=$(cat "$E15_BATCH/container.cid")
 if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e15.run" }}' "$E15_CID" 2>/dev/null || true)" = "$E15_CONTAINER"; then
  docker rm --force "$E15_CID" >> "$E15_BATCH/cleanup.log" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 3600s \
 docker run --rm --pull=never --init --cidfile "$E15_BATCH/container.cid" \
 --name "$E15_CONTAINER" --label "nerofactor.e15.run=$E15_CONTAINER" \
 --user "$(id -u):$(id -g)" --gpus "device=$E15_GPU" --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
 --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env HOME=/outputs/recovery/runtime --env XDG_CACHE_HOME=/outputs/recovery/runtime/cache \
 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$E15_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$E15_BENCHMARK,dst=/inputs/v3,readonly" \
 --mount "type=bind,src=$E15_PILOT,dst=/inputs/e11d,readonly" \
 --mount "type=bind,src=$E15_REFERENCE,dst=/inputs/e11e,readonly" \
 --mount "type=bind,src=$E15_FACTOR,dst=/inputs/e11f,readonly" \
 --mount "type=bind,src=$E15_MULTIVIEW,dst=/inputs/e11g,readonly" \
 --mount "type=bind,src=$E15_TWO_LIGHT,dst=/inputs/e14,readonly" \
 --mount "type=bind,src=$E15_AUTH,dst=/authorization.json,readonly" \
 --mount "type=bind,src=$E15_BATCH,dst=/outputs" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$E15_IMAGE" \
 --signal=TERM --kill-after=20s 3480s /usr/bin/python3 -I -B /workspace/run_bell_material_inverse_e15.py \
 --benchmark /inputs/v3 --pilot /inputs/e11d --reference /inputs/e11e --factor /inputs/e11f \
 --multiview /inputs/e11g --two-light /inputs/e14 --output /outputs/recovery \
 --freeze /workspace/docs/protocols/BELL_PHYSICAL_RECOVERY_E15_V1.freeze.json --freeze-sha256 "$E15_FREEZE" \
 --authorization /authorization.json --authorization-sha256 "$E15_AUTH_SHA" \
 --commit "$E15_COMMIT" --image-id "$E15_IMAGE" --execute > "$E15_BATCH/console.log" 2>&1
E15_EXIT=$?
set -e
printf '%s\n' "$E15_EXIT" > "$E15_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E15_BATCH/console.log"
exit "$E15_EXIT"
