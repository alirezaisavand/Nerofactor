#!/usr/bin/env bash
# Foreground execution of one exactly authorized A-Original-Oracle history.
set -euo pipefail
: "${DONUT_THESIS_SOURCE:?}" "${DONUT_THESIS_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$DONUT_THESIS_SOURCE")" = "$DONUT_THESIS_SOURCE"
test -z "$(git -C "$DONUT_THESIS_SOURCE" status --porcelain)"
commit=$(git -C "$DONUT_THESIS_SOURCE" rev-parse HEAD)
freeze_name=docs/protocols/DONUT_THESIS_ATTRIBUTION_GATE8_V1.freeze.json
freeze="$DONUT_THESIS_SOURCE/$freeze_name"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$DONUT_THESIS_AUTH" | cut -d' ' -f1)
spec=$(cd "$DONUT_THESIS_SOURCE" && python3 -B - "$DONUT_THESIS_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_thesis_attribution_gate8_v1 as g
from infrastructure.signatures import require

auth,commit,sha=sys.argv[1:]
f=g.freeze(g.ROOT/g.FREEZE,sha)
out=Path(f['host_output'])
require(Path(auth)==out.with_name(out.name+'-authorization.json')
        and g.read(Path(auth))==g.authorization(f,commit,sha),'Exact separate Gate 8 approval required')
require(not out.exists() and out.parent.is_dir(),'Fresh Gate 8 batch required')
p=g.paths(f)
_,views,_,_,_=g.m.admit(g.gate7.gate5.parent(),p)
g.relight_reference(f,p,views['oracle'])
g.control_receipt(f)
mounts=g.paths(f,native=True)
print(out)
print(f['image'])
print(len(mounts))
for key,native in mounts.items():
    print(f'{p[key]}\t{native}')
print(g.ARM)
PY
)
mapfile -t settings <<< "$spec"
batch=${settings[0]}
image=${settings[1]}
mount_count=${settings[2]}
arm=${settings[$((3+mount_count))]}
test "$arm" = A-Original-Oracle
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
for active_label in nerofactor.donut-recovery nerofactor.donut-heterogeneous nerofactor.bell-heterogeneous nerofactor.donut-geometry nerofactor.donut-geometry-rgb nerofactor.donut-relight nerofactor.donut-nero; do
    active=$(docker ps -q --filter "label=$active_label")
    test -z "$active"
done
mkdir "$batch"
cp "$DONUT_THESIS_AUTH" "$batch/authorization.json"
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
    --mount "type=bind,src=$DONUT_THESIS_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$batch/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$batch,dst=/outputs" "${mounts[@]}" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
    --signal=TERM --kill-after=20s 3540s /usr/bin/python3 -I -B /workspace/experiments/train_donut_thesis_attribution_gate8_v1.py \
    --freeze-sha256 "$sha" --authorization-sha256 "$auth_sha" --commit "$commit" --execute \
    >"$batch/$arm-console.txt" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$batch/$arm-exit-code.txt"
cleanup_owned
cat "$batch/$arm-console.txt"
test "$code" -eq 0
(cd "$DONUT_THESIS_SOURCE" && python3 -B - "$batch" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_thesis_attribution_gate8_v1 as g
path,sha=sys.argv[1:]
batch=Path(path)
g.completed(batch,g.freeze(g.ROOT/g.FREEZE,sha),sha)
PY
)
trap - EXIT
