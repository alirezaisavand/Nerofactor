#!/usr/bin/env bash
# One authorized rendering batch, foreground only.
set -euo pipefail
: "${DONUT_RELIGHT_SOURCE:?}" "${DONUT_RELIGHT_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$DONUT_RELIGHT_SOURCE")" = "$DONUT_RELIGHT_SOURCE"
test -z "$(git -C "$DONUT_RELIGHT_SOURCE" status --porcelain)"
commit=$(git -C "$DONUT_RELIGHT_SOURCE" rev-parse HEAD)
freeze="$DONUT_RELIGHT_SOURCE/docs/protocols/DONUT_RELIGHTING_GATE6_V1.freeze.json"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$DONUT_RELIGHT_AUTH" | cut -d' ' -f1)
spec=$(cd "$DONUT_RELIGHT_SOURCE" && python3 -B - "$DONUT_RELIGHT_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_relighting_gate6_v1 as g
from infrastructure.signatures import require

auth,commit,sha=sys.argv[1:]
f=g.freeze(sha)
out=Path(f['host_output'])
auth=Path(auth)
require(auth==out.with_name(out.name+'-authorization.json')
        and g.read(auth)==g.authorization(f,commit,sha), 'Exact separate Gate 6 approval required')
require(not out.exists() and out.parent.is_dir(), 'Fresh relighting output required')
p,e=g.base_paths(f),g.endpoint_paths(f)
g.admit(f,p,e)
print(out)
print(f['image'])
for host,native in ((p,g.base_paths(f,True)),(e,g.endpoint_paths(f,True))):
    for key,value in native.items():
        print(f'{host[key]}\t{value}')
PY
)
mapfile -t settings <<< "$spec"
out=${settings[0]}
image=${settings[1]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
for label in nerofactor.donut-recovery nerofactor.donut-heterogeneous nerofactor.bell-heterogeneous nerofactor.donut-geometry nerofactor.donut-geometry-rgb nerofactor.donut-relight; do
    active=$(docker ps -q --filter "label=$label")
    test -z "$active"
done
mkdir "$out"
cp "$DONUT_RELIGHT_AUTH" "$out/authorization.json"
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
        label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.donut-relight" }}' "$id" 2>/dev/null || true)
        if test "$label" = "$name"; then
            docker rm --force "$id" >>"$out/cleanup.txt" 2>&1
        fi
    fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 1800s \
    docker run --rm --pull=never --init --cidfile "$cidfile" \
    --name "$name" --label "nerofactor.donut-relight=$name" \
    --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --memory 64g --memory-swap 64g --cpus 4 --pids-limit 128 \
    --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
    --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --mount "type=bind,src=$DONUT_RELIGHT_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$out/authorization.json,dst=/authorization.json,readonly" \
    --mount "type=bind,src=$out,dst=/outputs" "${mounts[@]}" \
    --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
    --signal=TERM --kill-after=20s 1740s /usr/bin/python3 -I -B /workspace/experiments/donut_relighting_gate6_v1.py \
    --freeze-sha256 "$sha" --commit "$commit" --authorization-sha256 "$auth_sha" --execute \
    >"$out/console.txt" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$out/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$out/console.txt"
test "$code" -eq 0
(cd "$DONUT_RELIGHT_SOURCE" && python3 -B - "$out" "$sha" "$auth_sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import donut_relighting_gate6_v1 as g
from infrastructure.signatures import require

out,sha,auth_sha=sys.argv[1:]
r=g.read(Path(out)/'run/receipt.json')
require(r['status']=='complete' and r['cleanup_verified'] and r['freeze_sha256']==sha
        and r['authorization_sha256']==auth_sha and set(r['views'])==set(map(str,g.CAMERAS))
        and r['counters']['optimizer_updates']==r['counters']['primary_rays']==0,
        'Gate 6 did not complete')
PY
)
