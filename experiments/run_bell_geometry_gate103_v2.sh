#!/usr/bin/env bash
set -euo pipefail
: "${BELL_EVOLUTION_SOURCE:?}" "${BELL_EVOLUTION_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$BELL_EVOLUTION_SOURCE")" = "$BELL_EVOLUTION_SOURCE"
test "$(realpath "$BELL_EVOLUTION_AUTH")" = "$BELL_EVOLUTION_AUTH"
test -z "$(git -C "$BELL_EVOLUTION_SOURCE" status --porcelain)"
commit=$(git -C "$BELL_EVOLUTION_SOURCE" rev-parse HEAD)
cd "$BELL_EVOLUTION_SOURCE"
freeze=docs/protocols/BELL_GEOMETRY_EVOLUTION_GATE103_V2.freeze.json
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$BELL_EVOLUTION_AUTH" | cut -d' ' -f1)
entry=experiments/bell_geometry_gate103_v2.py
args=(--commit "$commit" --freeze-sha256 "$sha" --authorization-sha256 "$auth_sha")
settings=$(python3 -B "$entry" preflight "${args[@]}" --authorization "$BELL_EVOLUTION_AUTH")
mapfile -t lines <<< "$settings"
test "${#lines[@]}" -eq 3
out=${lines[0]}
image=${lines[1]}
oracle=${lines[2]}
actual_image=$(docker image inspect --format '{{.Id}}' "$image")
test "$actual_image" = "$image"
test "$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)" = 'NVIDIA A100 80GB PCIe'
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
for owner in nerofactor.bell-geometry nerofactor.bell-geometry-diagnosis nerofactor.bell-domain-extraction nerofactor.bell-geometry-evolution; do
    test -z "$(docker ps -q --filter "label=$owner")"
done
prior=/home/isavand/paper/evidence/bell-geometry-gate10-v4-5eba8365e19e4ab9aa2320491429cffb
python3 -B - "$out" "$prior" <<'PY'
import sys
from pathlib import Path
from shutil import disk_usage
from infrastructure.signatures import require
from benchmarks import bell_geometry_gate103_v2 as p
out,old=map(Path,sys.argv[1:])
require(disk_usage(out.parent).free>=2*p.RESOURCES['output_bytes'],'Insufficient diagnostic evidence space')
require(all((old/(s+'-container-absent.txt')).read_text().strip()=='absent' for s in ('train','evaluate')),
        'Prior container-removal receipts missing')
PY
settings_extra=$(python3 -B - <<'PY3'
from benchmarks import bell_geometry_gate103_v2 as p
f=p.read(p.ROOT/p.FREEZE)
print(p.ACCEPTED)
print(p.FAILED)
print(f['raw300_path'])
print(p.policy.read(p.ROOT/p.policy.FREEZE)['diagnostic']+'/run/stages/outside-sphere.npy')
PY3
)
mapfile -t extra <<< "$settings_extra"
accepted=${extra[0]}
completed=${extra[1]}
raw300=${extra[2]}
mask=${extra[3]}
mkdir "$out" "$out/run"
printf '%s\n' "$actual_image" >"$out/image-id.txt"
cp "$BELL_EVOLUTION_AUTH" "$out/authorization.json"
cp "$freeze" "$out/freeze.json"
test "$(sha256sum "$out/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$out/freeze.json" | cut -d' ' -f1)" = "$sha"
name=$(basename "$out")
printf 'Gate 10.3 console is at %s\n' "$out/console.txt"
cleanup_owned() {
    local found label
    found=$(docker ps -aq --filter "name=^/$name$")
    if test -n "$found"; then
        label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.bell-geometry-evolution" }}' "$found")
        test "$label" = "$name"
        docker rm --force "$found" >>"$out/cleanup.txt" 2>&1
    fi
    test -z "$(docker ps -aq --filter "name=^/$name$")"
    printf 'absent\n' >"$out/container-absent.txt"
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
code=0
timeout --foreground --signal=TERM --kill-after=30s 3600s \
    docker run --rm --pull=never --init --cidfile "$out/container.cid" \
    --name "$name" --label "nerofactor.bell-geometry-evolution=$name" \
    --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --memory 128g --memory-swap 128g --cpus 4 --pids-limit 128 \
    --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --env MAX_JOBS=4 --env TORCH_EXTENSIONS_DIR=/outputs/torch_extensions --env HOME=/tmp --env XDG_CACHE_HOME=/tmp \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$BELL_EVOLUTION_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$out/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$out/run,dst=/outputs" \
    --mount "type=bind,src=$accepted,dst=/inputs/accepted,readonly" \
    --mount "type=bind,src=$completed,dst=/inputs/completed15k,readonly" \
    --mount "type=bind,src=$raw300,dst=/inputs/raw300.npy,readonly" \
    --mount "type=bind,src=$mask,dst=/inputs/mask.npy,readonly" \
    --mount "type=bind,src=$prior/train/checkpoints/model-step15000.pth,dst=/inputs/checkpoints/model-step15000.pth,readonly" \
    --mount "type=bind,src=$prior/train/checkpoints/model-step30000.pth,dst=/inputs/checkpoints/model-step30000.pth,readonly" \
    --mount "type=bind,src=$prior/train/checkpoints/model-step50000.pth,dst=/inputs/checkpoints/model-step50000.pth,readonly" \
    --mount "type=bind,src=$prior/train/checkpoints/model.pth,dst=/inputs/checkpoints/model.pth,readonly" \
    --mount "type=bind,src=$prior/train/torch_extensions,dst=/inputs/jit,readonly" \
    --mount "type=bind,src=$oracle,dst=/inputs/oracle,readonly" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" --signal=TERM --kill-after=30s 3540s \
    /usr/bin/python3 -I -B "/workspace/$entry" execute "${args[@]}" \
    >"$out/console.txt" 2>&1 || code=$?
printf '%s\n' "$code" >"$out/exit-code.txt"
cleanup_owned
trap - EXIT
if test "$code" -ne 0; then
    tail -n 40 "$out/console.txt" >&2
    exit "$code"
fi
python3 -B "$entry" verify "${args[@]}"
