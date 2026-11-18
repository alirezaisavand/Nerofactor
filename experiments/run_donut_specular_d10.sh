#!/usr/bin/env bash
# One foreground D10 camera-28 render batch. No optimizer, retry or resume.
set -euo pipefail
: "${D10_SOURCE:?}" "${D10_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$D10_SOURCE")" = "$D10_SOURCE"
test -z "$(git -C "$D10_SOURCE" status --porcelain)"
commit=$(git -C "$D10_SOURCE" rev-parse HEAD)
freeze="$D10_SOURCE/docs/protocols/DONUT_SPECULAR_ATTRIBUTION_D10_V1.freeze.json"
freeze_sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$D10_AUTH" | cut -d' ' -f1)
spec=$(cd "$D10_SOURCE" && python3 -B - "$D10_AUTH" "$commit" "$freeze_sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_specular_d10 as d
from infrastructure.signatures import require

path, commit, sha = sys.argv[1:]
f = d.freeze(d.ROOT/d.FREEZE, sha)
require(d.read(Path(path)) == d.authorization(f, commit, sha), 'Exact D10 launch authorization required')
batch = Path(f['host_output'])
require(batch.parent.is_dir() and batch.resolve() == batch and not batch.exists(), 'Fresh batch required')
paths = {role: Path(item['host']) for role, item in f['artifacts'].items()}
d.panel(f, paths)
print(batch)
print(d.IMAGE)
print(f['selected_gpu'])
print(d.LIMITS['foreground_seconds'])
print(d.LIMITS['inner_seconds'])
print(f'{paths["generation"].parent}\t/inputs/reference')
print(f'{paths["geometry"].parent}\t/inputs/oracle')
for role in ('admission', 'camera', 'receipt', 'checkpoint', 'maps', 'errors'):
    print(f'{paths[role]}\t{f["artifacts"][role]["native"]}')
PY
)
mapfile -t settings <<< "$spec"
batch=${settings[0]}
image=${settings[1]}
gpu=${settings[2]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
test -z "$(docker ps -q --filter label=nerofactor.donut-recovery)"
test -z "$(docker ps -q --filter label=nerofactor.donut-d10)"
mkdir "$batch"
cp "$D10_AUTH" "$batch/authorization.json"
test "$(sha256sum "$batch/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
cp "$freeze" "$batch/freeze.json"
test "$(sha256sum "$batch/freeze.json" | cut -d' ' -f1)" = "$freeze_sha"
mounts=()
for item in "${settings[@]:5}"; do
    IFS=$'\t' read -r host native <<< "$item"
    mounts+=(--mount "type=bind,src=$host,dst=$native,readonly")
done
cidfile="$batch/container.cid"
cleanup_owned() {
    if test -s "$cidfile"; then
        local cid label
        cid=$(cat "$cidfile")
        label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.donut-d10" }}' "$cid" 2>/dev/null || true)
        if test "$label" = "$(basename "$batch")"; then
            docker rm --force "$cid" >> "$batch/cleanup.txt" 2>&1
        fi
    fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s "${settings[3]}s" \
    docker run --rm --pull=never --init --cidfile "$cidfile" \
    --name "$(basename "$batch")" --label "nerofactor.donut-d10=$(basename "$batch")" \
    --user "$(id -u):$(id -g)" --gpus "device=$gpu" --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --memory 64g --memory-swap 64g --cpus 4 --pids-limit 128 \
    --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --env OPENBLAS_NUM_THREADS=4 --mount "type=bind,src=$D10_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$batch/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$batch,dst=/outputs" "${mounts[@]}" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
    --signal=TERM --kill-after=20s "${settings[4]}s" /usr/bin/python3 -I -B /workspace/experiments/donut_specular_attribution_d10.py \
    --freeze-sha256 "$freeze_sha" --authorization-sha256 "$auth_sha" --commit "$commit" --execute \
    > "$batch/console.txt" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$batch/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$batch/console.txt"
test "$code" -eq 0
cd "$D10_SOURCE"
python3 -B - "$batch/run/render-receipt.json" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_specular_d10 as d
from infrastructure.signatures import require
r = d.read(Path(sys.argv[1]))
require(r['status'] == 'complete' and r['cleanup_verified'] and r['archived_replay']['passed']
        and r['RNG_modes_and_weights_unchanged'] and r['optimizer_returns'] == r['primary_rays'] == 0,
        'Incomplete D10 run; preserve partial evidence')
PY
