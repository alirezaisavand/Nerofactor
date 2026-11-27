#!/usr/bin/env bash
# One approved batch. Training and endpoint evaluation run sequentially in the foreground.
set -euo pipefail
: "${BELL_BG_SOURCE:?}" "${BELL_BG_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$BELL_BG_SOURCE")" = "$BELL_BG_SOURCE"
test "$(realpath "$BELL_BG_AUTH")" = "$BELL_BG_AUTH"
test -z "$(git -C "$BELL_BG_SOURCE" status --porcelain)"
commit=$(git -C "$BELL_BG_SOURCE" rev-parse HEAD)
cd "$BELL_BG_SOURCE"
freeze=docs/protocols/BELL_BACKGROUND_OPACITY_GATE11_V1.freeze.json
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$BELL_BG_AUTH" | cut -d' ' -f1)
entry=experiments/bell_geometry_gate11.py
args=(--commit "$commit" --freeze-sha256 "$sha" --authorization-sha256 "$auth_sha")
settings=$(python3 -B "$entry" preflight "${args[@]}" --authorization "$BELL_BG_AUTH")
mapfile -t lines <<< "$settings"
test "${#lines[@]}" -eq 3
out=${lines[0]}
image=${lines[1]}
oracle=${lines[2]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
test "$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)" = 'NVIDIA A100 80GB PCIe'
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
for owner in nerofactor.bell-background-opacity nerofactor.bell-geometry-evolution nerofactor.bell-geometry nerofactor.bell-heterogeneous nerofactor.donut-recovery nerofactor.donut-geometry; do
    test -z "$(docker ps -q --filter "label=$owner")"
done
python3 -B - "$out" <<'PY'
from pathlib import Path
from shutil import disk_usage
import sys
from benchmarks import bell_geometry_gate11 as g
from infrastructure.signatures import require
require(disk_usage(Path(sys.argv[1]).parent).free > 2*g.TRAIN_RESOURCES['owned_outputs_bytes'],
        'Insufficient evidence space for both Gate 11 phases')
PY
extra=$(python3 -B - <<'PY1'
from benchmarks import bell_geometry_gate11 as g
f=g.read(g.ROOT/g.FREEZE)
print(f['jit_cache_root'])
print(f['outside_mask_path'])
PY1
)
mapfile -t extra_lines <<< "$extra"
jit=${extra_lines[0]}
mask=${extra_lines[1]}
mkdir "$out"
docker image inspect --format '{{.Id}}' "$image" >"$out/image-id.txt"
cp "$BELL_BG_AUTH" "$out/authorization.json"
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
            label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.bell-background-opacity" }}' "$existing")
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
    printf 'Gate 11 %s console is at %s/%s-console.txt\n' "$phase" "$out" "$phase"
    code=0
    timeout --foreground --signal=TERM --kill-after=30s "${outer}s" \
        docker run --rm --pull=never --init --cidfile "$out/$phase-container.cid" \
        --name "$name-$phase" --label "nerofactor.bell-background-opacity=$name" \
        --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
        --security-opt no-new-privileges:true --memory 128g --memory-swap 128g --cpus 4 --pids-limit 256 \
        --shm-size 4g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=512m \
        --env MAX_JOBS=4 --env TORCH_EXTENSIONS_DIR=/outputs/torch_extensions \
        --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
        --env OPENBLAS_NUM_THREADS=4 --mount "type=bind,src=$BELL_BG_SOURCE,dst=/workspace,readonly" \
        --mount "type=bind,src=$out/authorization.json,dst=/authorization.json,readonly" \
        --mount "type=bind,src=$out/$phase,dst=/outputs" "$@" \
        --mount "type=bind,src=$jit,dst=/inputs/jit,readonly" \
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
batch_started=$SECONDS
launch train 129600 129540 --mount "type=bind,src=$out/training,dst=/inputs/training,readonly"
python3 -B "$entry" verify-train "${args[@]}"
test "$((SECONDS-batch_started))" -lt 130740
launch evaluate 7200 7140 \
    --mount "type=bind,src=$out/train,dst=/inputs/training-result,readonly" \
    --mount "type=bind,src=$oracle,dst=/inputs/oracle,readonly" \
    --mount "type=bind,src=$mask,dst=/inputs/mask.npy,readonly"
test "$((SECONDS-batch_started))" -lt 138000
python3 -B "$entry" finalize "${args[@]}"
