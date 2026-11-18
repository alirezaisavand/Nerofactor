#!/usr/bin/env bash
# One foreground v2 diagnostic and conditional completion under exact approval.
set -euo pipefail
: "${HET_SOURCE:?}" "${HET_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$HET_SOURCE")" = "$HET_SOURCE"
test -z "$(git -C "$HET_SOURCE" status --porcelain)"
commit=$(git -C "$HET_SOURCE" rev-parse HEAD)
freeze="$HET_SOURCE/docs/protocols/DONUT_HETEROGENEOUS_REFERENCE_V2.freeze.json"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$HET_AUTH" | cut -d' ' -f1)
spec=$(cd "$HET_SOURCE" && python3 -B - "$HET_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_heterogeneous_v1 as v1
from benchmarks import donut_heterogeneous_v2 as d
from infrastructure.signatures import require

auth,commit,sha=sys.argv[1:]
f=d.freeze(d.ROOT/d.FREEZE,sha)
require(v1.read(Path(auth))==d.authorization(f,commit,sha),'Exact heterogeneous v2 approval required')
out=Path(f['host_output'])
require(out.resolve()==out and out.parent.is_dir() and not out.exists(),'Fresh heterogeneous reference batch required')
paths=f['host_inputs']
prior=Path(f['retained_failed_batch'])
d.source(f,prior,tuple(Path(paths[k]) for k in ('oracle','cameras','reference','admission')))
print(out)
print(v1.IMAGE)
print(f'{prior}\t/inputs/failed-v1')
for k,n in [('oracle','oracle'),('cameras','cameras.json'),('reference','reference'),('admission','admission.json')]:
    print(f'{paths[k]}\t/inputs/{n}')
PY
)
mapfile -t settings <<< "$spec"
out=${settings[0]}
image=${settings[1]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
test -z "$(docker ps -q --filter label=nerofactor.donut-recovery)"
test -z "$(docker ps -q --filter label=nerofactor.donut-heterogeneous)"
mkdir "$out"
cp "$HET_AUTH" "$out/authorization.json"
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
        label=$(docker container inspect --format '{{ index .Config.Labels "nerofactor.donut-heterogeneous" }}' "$id" 2>/dev/null || true)
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
    --name "$name" --label "nerofactor.donut-heterogeneous=$name" \
    --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --memory 64g --memory-swap 64g --cpus 4 --pids-limit 128 \
    --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --env MKL_NUM_THREADS=4 --mount "type=bind,src=$HET_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$out/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$out,dst=/outputs" "${mounts[@]}" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
    --signal=TERM --kill-after=20s 7080s /usr/bin/python3 -I -B /workspace/experiments/generate_donut_heterogeneous_v2.py \
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
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['status']=='complete_pending_researcher_admission' and r['cleanup_verified'] and r['scientific_work_complete']
assert r['native_model_calls']==r['optimizer_returns']==r['counters']['primary_rays']==0
PY
