#!/usr/bin/env bash
# One separately approved foreground pair; never retries a failed history.
set -euo pipefail
: "${BELL_MATERIAL_SOURCE:?}" "${BELL_MATERIAL_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$BELL_MATERIAL_SOURCE")" = "$BELL_MATERIAL_SOURCE"
test -z "$(git -C "$BELL_MATERIAL_SOURCE" status --porcelain)"
commit=$(git -C "$BELL_MATERIAL_SOURCE" rev-parse HEAD)
freeze_name=docs/protocols/BELL_MATCHED_MATERIAL_GATE13_V1.freeze.json
freeze="$BELL_MATERIAL_SOURCE/$freeze_name"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$BELL_MATERIAL_AUTH" | cut -d' ' -f1)
spec=$(cd "$BELL_MATERIAL_SOURCE" && python3 -B - "$BELL_MATERIAL_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import bell_material_geometry_gate13 as m
auth,commit,sha=sys.argv[1:]
f=m.freeze(m.ROOT/m.FREEZE,sha)
m.require(len(commit)==40 and m.read(Path(auth))==m.authorization(f,commit,sha),'Separate exact-source approval required')
out=Path(f['host_output']);m.require(not out.exists() and out.parent.is_dir(),'Fresh Gate 13 batch required')
p=m.paths();m.admit(p)
print(out);print(m.IMAGE);print(len(p))
for key,path in p.items():print(str(path)+'\t/inputs/'+key)
print(*f['order'],sep='\n')
PY
)
mapfile -t settings <<< "$spec"
batch=${settings[0]};image=${settings[1]};mount_count=${settings[2]}
arms=("${settings[@]:$((3+mount_count))}")
test "${#arms[@]}" -eq 2
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
for label in nerofactor.donut-recovery nerofactor.bell-heterogeneous nerofactor.bell-geometry nerofactor.bell-material-gate13; do
    test -z "$(docker ps -q --filter "label=$label")"
done
mkdir "$batch"
cp "$BELL_MATERIAL_AUTH" "$batch/authorization.json"
cp "$freeze" "$batch/freeze.json"
printf '%s\n' "$image" >"$batch/image-id.txt"
test "$(sha256sum "$batch/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$batch/freeze.json" | cut -d' ' -f1)" = "$sha"
mounts=()
for item in "${settings[@]:3:mount_count}"; do
    IFS=$'\t' read -r host native <<< "$item"
    mounts+=(--mount "type=bind,src=$host,dst=$native,readonly")
done
cidfile='';name=''
cleanup_owned() {
    if test -n "$cidfile" && test -s "$cidfile"; then
        local id label
        id=$(cat "$cidfile")
        label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.bell-material-gate13" }}' "$id" 2>/dev/null || true)
        if test "$label" = "$name"; then
            docker rm --force "$id" >>"$batch/cleanup.txt" 2>&1
        fi
    fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for arm in "${arms[@]}"; do
    cidfile="$batch/$arm.cid";name="$(basename "$batch")-$arm"
    pair_mounts=()
    if test "$arm" = K1-O1; then
        pair_mounts+=(--mount "type=bind,src=$batch/K1-Oracle,dst=/paired,readonly")
        pair_mounts+=(--mount "type=bind,src=$batch/K1-Oracle,dst=/outputs/K1-Oracle,readonly")
    fi
    printf 'Gate 13 %s console: %s\n' "$arm" "$batch/$arm-console.txt"
    set +e
    timeout --foreground --signal=TERM --kill-after=30s 9000s \
        docker run --rm --pull=never --init --cidfile "$cidfile" \
        --name "$name" --label "nerofactor.bell-material-gate13=$name" \
        --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
        --security-opt no-new-privileges:true --memory 64g --memory-swap 64g --cpus 4 --pids-limit 128 \
        --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
        --env "HOME=/outputs/$arm/runtime" --env "XDG_CACHE_HOME=/outputs/$arm/runtime/cache" \
        --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
        --mount "type=bind,src=$BELL_MATERIAL_SOURCE,dst=/workspace,readonly" \
        --mount "type=bind,src=$batch/authorization.json,dst=/authorization.json,readonly" \
        --mount "type=bind,src=$batch,dst=/outputs" "${pair_mounts[@]}" "${mounts[@]}" \
        --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
        --signal=TERM --kill-after=20s 8940s /usr/bin/python3 -I -B /workspace/experiments/train_bell_material_geometry_gate13.py \
        --phase "$arm" --freeze-sha256 "$sha" --authorization-sha256 "$auth_sha" --commit "$commit" --execute \
        >"$batch/$arm-console.txt" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" >"$batch/$arm-exit-code.txt"
    cleanup_owned
    test -z "$(docker ps -aq --filter "name=^/$name$")"
    printf 'absent\n' >"$batch/$arm-container-absent.txt"
    if test "$code" -ne 0; then tail -80 "$batch/$arm-console.txt";exit "$code";fi
    (cd "$BELL_MATERIAL_SOURCE" && python3 -B - "$batch" "$arm" "$commit" "$sha" "$auth_sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import bell_material_geometry_gate13 as m
batch,arm,commit,sha,auth_sha=sys.argv[1:];batch=Path(batch)
m.completed(batch,arm,m.read(batch/'freeze.json'),commit,sha,auth_sha)
PY
    )
done
trap - EXIT
(cd "$BELL_MATERIAL_SOURCE" && python3 -B experiments/analyze_bell_material_geometry_gate13.py --freeze-sha256 "$sha")
