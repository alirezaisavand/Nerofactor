#!/usr/bin/env bash
# Run the one exact approved geometry audit in the foreground, with read-only inputs.
set -euo pipefail
: "${DONUT_GEOMETRY_SOURCE:?}" "${DONUT_GEOMETRY_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$DONUT_GEOMETRY_SOURCE")" = "$DONUT_GEOMETRY_SOURCE"
test -z "$(git -C "$DONUT_GEOMETRY_SOURCE" status --porcelain)"
commit=$(git -C "$DONUT_GEOMETRY_SOURCE" rev-parse HEAD)
freeze="$DONUT_GEOMETRY_SOURCE/docs/protocols/DONUT_RECONSTRUCTED_GEOMETRY_VALIDATION_V1.freeze.json"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$DONUT_GEOMETRY_AUTH" | cut -d' ' -f1)
spec=$(cd "$DONUT_GEOMETRY_SOURCE" && python3 -B - "$DONUT_GEOMETRY_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_reconstructed_geometry_v1 as g
from infrastructure.signatures import require
auth, commit, sha = sys.argv[1:]
f = g.freeze(sha)
require(g.h.read(Path(auth)) == g.authorization(f, commit, sha), 'Exact geometry approval required')
out = Path(f['host_output'])
require(not out.exists() and out.parent.is_dir(), 'Fresh geometry output required')
p = g.paths(f)
g.admit(f, p)
print(out)
print(f['image'])
for key, native in g.MOUNTS.items():
    print(f'{p[key]}\t{native}')
PY
)
mapfile -t settings <<< "$spec"
out=${settings[0]}
image=${settings[1]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
for owner in nerofactor.donut-recovery nerofactor.donut-heterogeneous nerofactor.bell-heterogeneous nerofactor.donut-geometry; do
    test -z "$(docker ps -q --filter "label=$owner")"
done
mkdir "$out"
cp "$DONUT_GEOMETRY_AUTH" "$out/authorization.json"
cp "$freeze" "$out/freeze.json"
test "$(sha256sum "$out/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$out/freeze.json" | cut -d' ' -f1)" = "$sha"
mounts=()
for item in "${settings[@]:2}"; do
    IFS=$'\t' read -r host native <<< "$item"
    mounts+=(--mount "type=bind,src=$host,dst=$native,readonly")
done
cidfile="$out/container.cid"
name=$(basename "$out")
cleanup_owned() {
    if test -s "$cidfile"; then
        local id label
        id=$(cat "$cidfile")
        label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.donut-geometry" }}' "$id" 2>/dev/null || true)
        if test "$label" = "$name"; then
            docker rm --force "$id" >>"$out/cleanup.txt" 2>&1
        fi
    fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 900s \
    docker run --rm --pull=never --init --cidfile "$cidfile" \
    --name "$name" --label "nerofactor.donut-geometry=$name" \
    --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --memory 16g --memory-swap 16g --cpus 4 --pids-limit 128 \
    --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --env MKL_NUM_THREADS=4 --mount "type=bind,src=$DONUT_GEOMETRY_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$out/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$out,dst=/outputs" "${mounts[@]}" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
    --signal=TERM --kill-after=20s 870s /usr/bin/python3 -I -B /workspace/experiments/donut_reconstructed_geometry_v1.py \
    --freeze-sha256 "$sha" --commit "$commit" --authorization-sha256 "$auth_sha" --execute \
    >"$out/console.txt" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$out/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$out/console.txt"
test "$code" -eq 0
python3 -B - "$out/audit/receipt.json" <<'PY'
from pathlib import Path
import json
import sys
r = json.loads(Path(sys.argv[1]).read_text())
assert r['status'] == 'complete' and r['cleanup_verified'] and all(r['gates'].values())
assert r['renderer_calls'] == r['optimizer_updates'] == r['checkpoint_loads'] == 0
PY
