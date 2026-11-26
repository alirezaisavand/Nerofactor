#!/usr/bin/env bash
# One approved batch. Training and endpoint evaluation run sequentially in the foreground.
set -euo pipefail
: "${BELL_GEOMETRY_SOURCE:?}" "${BELL_GEOMETRY_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$BELL_GEOMETRY_SOURCE")" = "$BELL_GEOMETRY_SOURCE"
test "$(realpath "$BELL_GEOMETRY_AUTH")" = "$BELL_GEOMETRY_AUTH"
test -z "$(git -C "$BELL_GEOMETRY_SOURCE" status --porcelain)"
commit=$(git -C "$BELL_GEOMETRY_SOURCE" rev-parse HEAD)
cd "$BELL_GEOMETRY_SOURCE"
freeze=docs/protocols/BELL_AUTHENTICATED_GEOMETRY_GATE10_V1.freeze.json
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$BELL_GEOMETRY_AUTH" | cut -d' ' -f1)
entry=experiments/bell_geometry_gate10_v1.py
args=(--commit "$commit" --freeze-sha256 "$sha" --authorization-sha256 "$auth_sha")
settings=$(python3 -B "$entry" preflight "${args[@]}" --authorization "$BELL_GEOMETRY_AUTH")
mapfile -t lines <<< "$settings"
test "${#lines[@]}" -eq 3
out=${lines[0]}
image=${lines[1]}
oracle=${lines[2]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
test "$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)" = 'NVIDIA A100 80GB PCIe'
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
for owner in nerofactor.bell-geometry nerofactor.bell-heterogeneous nerofactor.donut-recovery nerofactor.donut-geometry; do
    test -z "$(docker ps -q --filter "label=$owner")"
done
python3 -B - "$out" <<'PY'
from pathlib import Path
from shutil import disk_usage
import sys
from benchmarks import bell_geometry_gate10_v1 as g
from infrastructure.signatures import require
require(disk_usage(Path(sys.argv[1]).parent).free > 2*g.RESOURCES['owned_outputs_bytes'],
        'Insufficient evidence space for both Gate 10 phases')
PY
mkdir "$out"
cp "$BELL_GEOMETRY_AUTH" "$out/authorization.json"
cp "$freeze" "$out/freeze.json"
test "$(sha256sum "$out/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$out/freeze.json" | cut -d' ' -f1)" = "$sha"
python3 -B "$entry" stage "${args[@]}"
name=$(basename "$out")
owned_phase=
cleanup_owned() {
    if test -n "$owned_phase"; then
        local owned_name label existing
        owned_name="$name-$owned_phase"
        existing=$(docker ps -aq --filter "name=^/$owned_name$")
        if test -n "$existing"; then
            label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.bell-geometry" }}' "$existing")
            test "$label" = "$name"
            docker rm --force "$existing" >>"$out/cleanup.txt" 2>&1
        fi
        test -z "$(docker ps -aq --filter "name=^/$owned_name$")"
        printf 'absent\n' >"$out/$owned_phase-container-absent.txt"
    fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch() {
    local phase=$1 outer=$2 inner=$3 code
    shift 3
    mkdir "$out/$phase"
    owned_phase=$phase
    printf 'Gate 10 %s console is at %s/%s-console.txt\n' "$phase" "$out" "$phase"
    code=0
    timeout --foreground --signal=TERM --kill-after=30s "${outer}s" \
        docker run --rm --pull=never --init --cidfile "$out/$phase-container.cid" \
        --name "$name-$phase" --label "nerofactor.bell-geometry=$name" \
        --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
        --security-opt no-new-privileges:true --memory 128g --memory-swap 128g --cpus 4 --pids-limit 256 \
        --shm-size 4g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=512m \
        --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
        --env OPENBLAS_NUM_THREADS=4 --mount "type=bind,src=$BELL_GEOMETRY_SOURCE,dst=/workspace,readonly" \
        --mount "type=bind,src=$out/authorization.json,dst=/authorization.json,readonly" \
        --mount "type=bind,src=$out/$phase,dst=/outputs" "$@" \
        --workdir /workspace --entrypoint /usr/bin/timeout "$image" --signal=TERM --kill-after=30s "${inner}s" \
        /usr/bin/python3 -I -B "/workspace/$entry" "$phase" "${args[@]}" \
        >"$out/$phase-console.txt" 2>&1 || code=$?
    printf '%s\n' "$code" >"$out/$phase-exit-code.txt"
    cleanup_owned
    owned_phase=
    if test "$code" -ne 0; then
        tail -n 36 "$out/$phase-console.txt" >&2
        return "$code"
    fi
}
launch train 129600 129540 --mount "type=bind,src=$out/training,dst=/inputs/training,readonly"
python3 -B "$entry" verify-train "${args[@]}"
launch evaluate 7200 7140 \
    --mount "type=bind,src=$out/train,dst=/inputs/training-result,readonly" \
    --mount "type=bind,src=$oracle,dst=/inputs/oracle,readonly"
python3 -B "$entry" finalize "${args[@]}"
