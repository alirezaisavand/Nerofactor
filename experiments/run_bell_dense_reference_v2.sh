#!/usr/bin/env bash
# Launch the one approved dense Bell continuation in the existing corrected image.
set -euo pipefail
: "${BELL_HET_SOURCE:?}" "${BELL_HET_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$BELL_HET_SOURCE")" = "$BELL_HET_SOURCE"
test -z "$(git -C "$BELL_HET_SOURCE" status --porcelain)"
commit=$(git -C "$BELL_HET_SOURCE" rev-parse HEAD)
freeze="$BELL_HET_SOURCE/docs/protocols/BELL_DENSE_REFERENCE_V2.freeze.json"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$BELL_HET_AUTH" | cut -d' ' -f1)
spec=$(cd "$BELL_HET_SOURCE" && python3 -B - "$BELL_HET_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import bell_dense_reference_v1 as d
from benchmarks import bell_dense_reference_v2 as v2
from benchmarks import bell_heterogeneous_v1 as b
from infrastructure.signatures import require

auth, commit, sha = sys.argv[1:]
f = v2.freeze(b.ROOT/v2.FREEZE, sha)
require(b.read(Path(auth)) == v2.authorization(f, commit, sha), 'Exact separate dense Bell v2 authorization required')
out = Path(f['host_output'])
require(out.resolve() == out and out.parent.is_dir() and not out.exists(), 'Fresh Bell reference batch required')
prior = v2.prior_definition()
paths = d.host_paths(prior)
d.source(prior, *paths)
failed = Path(f['failed_batch'])
v2.admit_prefix(f, failed)
cid = (failed/'container.cid').read_text().strip()
require(len(cid)==64 and all(c in '0123456789abcdef' for c in cid), 'Invalid failed Bell container ID')
for value in (out, b.IMAGE, *paths, failed, cid):
    print(value)
PY
)
mapfile -t settings <<< "$spec"
out=${settings[0]}
image=${settings[1]}
oracle=${settings[2]}
admission=${settings[3]}
sparse=${settings[4]}
sparse_admission=${settings[5]}
sparse_verification=${settings[6]}
failed=${settings[7]}
previous_id=${settings[8]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
previous_by_id=$(docker ps -aq --no-trunc --filter "id=$previous_id")
previous_by_label=$(docker ps -aq --filter "label=nerofactor.bell-heterogeneous=$(basename "$failed")")
test -z "$previous_by_id" && test -z "$previous_by_label" || exit 1
active=$(docker ps -q --filter label=nerofactor.bell-heterogeneous)
test -z "$active"
mkdir "$out"
cp "$BELL_HET_AUTH" "$out/authorization.json"
cp "$freeze" "$out/freeze.json"
test "$(sha256sum "$out/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$out/freeze.json" | cut -d' ' -f1)" = "$sha"
cidfile="$out/container.cid"
name=$(basename "$out")
cleanup_owned() {
    if test -s "$cidfile"; then
        local id label
        id=$(cat "$cidfile")
        label=$(docker container inspect --format '{{ index .Config.Labels "nerofactor.bell-heterogeneous" }}' "$id" 2>/dev/null || true)
        if test "$label" = "$name"; then
            docker rm --force "$id" >>"$out/cleanup.txt" 2>&1
        fi
    fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 7200s \
    docker run --rm --pull=never --init --cidfile "$cidfile" \
    --name "$name" --label "nerofactor.bell-heterogeneous=$name" \
    --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --memory 64g --memory-swap 64g --cpus 4 --pids-limit 128 \
    --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --env MKL_NUM_THREADS=4 --mount "type=bind,src=$BELL_HET_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$out/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$out,dst=/outputs" \
    --mount "type=bind,src=$oracle,dst=/inputs/oracle,readonly" \
    --mount "type=bind,src=$admission,dst=/inputs/admission.json,readonly" \
    --mount "type=bind,src=$sparse,dst=/inputs/sparse,readonly" \
    --mount "type=bind,src=$sparse_admission,dst=/inputs/sparse-admission.json,readonly" \
    --mount "type=bind,src=$sparse_verification,dst=/inputs/sparse-verification.json,readonly" \
    --mount "type=bind,src=$failed,dst=/inputs/failed,readonly" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
    --signal=TERM --kill-after=20s 7080s /usr/bin/python3 -I -B /workspace/experiments/bell_dense_reference_v2.py \
    --freeze-sha256 "$sha" --commit "$commit" --authorization-sha256 "$auth_sha" --execute \
    >"$out/console.txt" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$out/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$out/console.txt"
test "$code" -eq 0
python3 -B - "$out/reference/generation.json" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text())
assert r['status'] == 'complete_pending_researcher_admission' and r['cleanup_verified'] and r['scientific_work_complete']
assert r['native_model_calls'] == r['optimizer_returns'] == r['counters']['primary_rays'] == 0
PY
