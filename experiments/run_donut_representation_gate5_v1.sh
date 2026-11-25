#!/usr/bin/env bash
# Foreground execution of one exactly authorized Final-Chart-Oracle history.
set -euo pipefail
: "${DONUT_REPRESENTATION_SOURCE:?}" "${DONUT_REPRESENTATION_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$DONUT_REPRESENTATION_SOURCE")" = "$DONUT_REPRESENTATION_SOURCE"
test -z "$(git -C "$DONUT_REPRESENTATION_SOURCE" status --porcelain)"
commit=$(git -C "$DONUT_REPRESENTATION_SOURCE" rev-parse HEAD)
freeze_name=docs/protocols/DONUT_REPRESENTATION_GATE5_V1.freeze.json
freeze="$DONUT_REPRESENTATION_SOURCE/$freeze_name"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$DONUT_REPRESENTATION_AUTH" | cut -d' ' -f1)
spec=$(cd "$DONUT_REPRESENTATION_SOURCE" && python3 -B - "$DONUT_REPRESENTATION_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_representation_gate5_v1 as g
from infrastructure.signatures import require

auth,commit,sha=sys.argv[1:]
f=g.freeze(g.ROOT/g.FREEZE,sha)
require(g.read(Path(auth))==g.authorization(f,commit,sha),'Exact separate Gate 5 approval required')
out=Path(f['host_output'])
require(not out.exists() and out.parent.is_dir(),'Fresh Gate 5 batch required')
parent=g.parent()
p=g.paths(f)
_,_,controls=g.control_receipts(parent)
require(all(controls[g.AMBIENT]['outputs'][name]==f['ambient_captures'][key]
            for key,name in g.CAPTURES.items()),'Ambient capture inventory differs from the accepted receipt')
g.check_inputs(f,p)
g.m.admit(parent,p)
print(out)
print(f['image_id'])
print(len(p))
for key, native in g.MOUNTS.items():
    print(f'{p[key]}\t{native}')
print(g.ARM)
PY
)
mapfile -t settings <<< "$spec"
batch=${settings[0]}
image=${settings[1]}
mount_count=${settings[2]}
arm=${settings[$((3+mount_count))]}
test "$arm" = Final-Chart-Oracle
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
for active_label in nerofactor.donut-recovery nerofactor.donut-heterogeneous nerofactor.bell-heterogeneous nerofactor.donut-geometry nerofactor.donut-geometry-rgb; do
    active=$(docker ps -q --filter "label=$active_label")
    test -z "$active"
done
mkdir "$batch"
cp "$DONUT_REPRESENTATION_AUTH" "$batch/authorization.json"
cp "$freeze" "$batch/freeze.json"
test "$(sha256sum "$batch/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$batch/freeze.json" | cut -d' ' -f1)" = "$sha"
mounts=()
for item in "${settings[@]:3:mount_count}"; do
    IFS=$'\t' read -r host native <<< "$item"
    mounts+=(--mount "type=bind,src=$host,dst=$native,readonly")
done
cidfile=''
name=''
cleanup_owned() {
    if test -n "$cidfile" && test -s "$cidfile"; then
        local id label
        id=$(cat "$cidfile")
        label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.donut-recovery" }}' "$id" 2>/dev/null || true)
        if test "$label" = "$name"; then
            docker rm --force "$id" >>"$batch/cleanup.txt" 2>&1
        fi
    fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cidfile="$batch/$arm.cid"
name="$(basename "$batch")-$arm"
printf 'Starting %s\n' "$arm"
set +e
timeout --foreground --signal=TERM --kill-after=30s 3600s \
    docker run --rm --pull=never --init --cidfile "$cidfile" \
    --name "$name" --label "nerofactor.donut-recovery=$name" \
    --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --memory 64g --memory-swap 64g --cpus 4 --pids-limit 128 \
    --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --env "HOME=/outputs/$arm/runtime" --env "XDG_CACHE_HOME=/outputs/$arm/runtime/cache" \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$DONUT_REPRESENTATION_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$batch/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$batch,dst=/outputs" "${mounts[@]}" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
    --signal=TERM --kill-after=20s 3540s /usr/bin/python3 -I -B /workspace/experiments/train_donut_representation_gate5_v1.py \
    --freeze-sha256 "$sha" --authorization-sha256 "$auth_sha" --commit "$commit" --execute \
    >"$batch/$arm-console.txt" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$batch/$arm-exit-code.txt"
cleanup_owned
cat "$batch/$arm-console.txt"
test "$code" -eq 0
(cd "$DONUT_REPRESENTATION_SOURCE" && python3 -B - "$batch" "$sha" "$auth_sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_representation_gate5_v1 as g
path,sha,auth_sha=sys.argv[1:]
batch=Path(path)
auth=g.read(batch/'authorization.json')
g.completed(batch,g.freeze(g.ROOT/g.FREEZE,sha),auth,sha,auth_sha)
PY
)
trap - EXIT
