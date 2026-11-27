#!/usr/bin/env bash
# One exact, separately authorized foreground run. No training or checkpoint load.
set -euo pipefail
: "${BELL_RGB_SOURCE:?}" "${BELL_RGB_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$BELL_RGB_SOURCE")" = "$BELL_RGB_SOURCE"
test "$(realpath "$BELL_RGB_AUTH")" = "$BELL_RGB_AUTH"
test -z "$(git -C "$BELL_RGB_SOURCE" status --porcelain)"
cd "$BELL_RGB_SOURCE"
commit=$(git rev-parse HEAD)
freeze=docs/protocols/BELL_GEOMETRY_RGB_GATE12_V1.freeze.json
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$BELL_RGB_AUTH" | cut -d' ' -f1)
args=(--commit "$commit" --freeze-sha256 "$sha" --authorization-sha256 "$auth_sha")
settings=$(python3 -B experiments/bell_geometry_rgb_gate12.py preflight "${args[@]}" --authorization "$BELL_RGB_AUTH")
mapfile -t lines <<< "$settings"
test "${#lines[@]}" -eq 5
out=${lines[0]}
image=${lines[1]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
test "$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)" = 'NVIDIA A100 80GB PCIe'
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
for label in nerofactor.bell-geometry-rgb nerofactor.bell-background-opacity nerofactor.bell-geometry nerofactor.donut-recovery; do
    test -z "$(docker ps -q --filter "label=$label")"
done
mounts=()
for entry in "${lines[@]:2}"; do
    IFS=$'\t' read -r host native <<< "$entry"
    mounts+=(--mount "type=bind,src=$host,dst=$native,readonly")
done
mkdir "$out"
cp "$BELL_RGB_AUTH" "$out/authorization.json"
cp "$freeze" "$out/freeze.json"
test "$(sha256sum "$out/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$out/freeze.json" | cut -d' ' -f1)" = "$sha"
docker image inspect --format '{{.Id}}' "$image" >"$out/image-id.txt"
name=$(basename "$out")
cleanup_owned() {
    local id label
    id=$(docker ps -aq --filter "name=^/$name$")
    if test -n "$id"; then
        label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.bell-geometry-rgb" }}' "$id")
        test "$label" = "$name"
        docker rm --force "$id" >>"$out/cleanup.txt" 2>&1
    fi
    test -z "$(docker ps -aq --filter "name=^/$name$")"
    printf 'absent\n' >"$out/container-absent.txt"
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
code=0
printf 'Gate 12 console: %s/console.txt\n' "$out"
timeout --foreground --signal=TERM --kill-after=30s 7200s \
    docker run --rm --pull=never --init --cidfile "$out/container.cid" \
    --name "$name" --label "nerofactor.bell-geometry-rgb=$name" \
    --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --memory 64g --memory-swap 64g --cpus 4 --pids-limit 128 \
    --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --mount "type=bind,src=$BELL_RGB_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$out/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$out,dst=/outputs" "${mounts[@]}" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" --signal=TERM --kill-after=20s 7080s \
    /usr/bin/python3 -I -B /workspace/experiments/bell_geometry_rgb_gate12.py execute "${args[@]}" \
    >"$out/console.txt" 2>&1 || code=$?
printf '%s\n' "$code" >"$out/exit-code.txt"
cleanup_owned
trap - EXIT
if test "$code" -ne 0; then tail -n 36 "$out/console.txt" >&2; exit "$code"; fi
MPLCONFIGDIR=/tmp/bell-gate12-matplotlib timeout --foreground --signal=TERM --kill-after=20s 600s \
    python3 -B experiments/verify_bell_geometry_rgb_gate12.py "${args[@]}" \
    >"$out/verify-console.txt" 2>&1
cat "$out/verify-console.txt"
