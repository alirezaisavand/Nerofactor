#!/usr/bin/env bash
# Foreground execution of one exactly authorized batch, stopping at the first failed arm.
set -euo pipefail
: "${BELL_TRAIN_SOURCE:?}" "${BELL_TRAIN_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$BELL_TRAIN_SOURCE")" = "$BELL_TRAIN_SOURCE"
test -z "$(git -C "$BELL_TRAIN_SOURCE" status --porcelain)"
commit=$(git -C "$BELL_TRAIN_SOURCE" rev-parse HEAD)
freeze_name=docs/protocols/BELL_PUBLICATION_SEEDS_V1.freeze.json
freeze="$BELL_TRAIN_SOURCE/$freeze_name"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$BELL_TRAIN_AUTH" | cut -d' ' -f1)
spec=$(cd "$BELL_TRAIN_SOURCE" && python3 -B - "$BELL_TRAIN_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import bell_publication_seeds_v1 as h
from infrastructure.signatures import require

auth,commit,sha=sys.argv[1:]
f=h.freeze(h.ROOT/h.FREEZE,sha)
require(h.b.read(Path(auth))==h.authorization(f,commit,sha),'Exact separate seed execution approval required')
out=Path(f['host_output'])
require(not out.exists() and out.parent.is_dir(),'Fresh publication seed batch required')
files=f['dataset']
h.geometry(f,Path(f['mesh']['path']))
h.dataset(f,Path(f['host_reference']),Path(files['admission']['path']),Path(files['verification']['path']))
h.final.verify_controls(h.final.freeze(h.ROOT/h.final.FREEZE,h.PARENT_SHA))
h.b.check_artifact(Path(f['held_out_support']['path']),f['held_out_support'])
print(out)
print(h.IMAGE)
print(f'{f["mesh"]["path"]}\t/inputs/oracle.ply')
print(f'{f["host_reference"]}\t/inputs/reference')
for key in ('admission','verification'):
    print(f'{files[key]["path"]}\t/inputs/{key}.json')
print(*f['order'],sep='\n')
PY
)
mapfile -t settings <<< "$spec"
batch=${settings[0]}
image=${settings[1]}
arms=("${settings[@]:6}")
test "${#arms[@]}" -eq 6
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
for active_label in nerofactor.donut-recovery nerofactor.donut-heterogeneous nerofactor.bell-heterogeneous; do
    active=$(docker ps -q --filter "label=$active_label")
    test -z "$active"
done
mkdir "$batch"
cp "$BELL_TRAIN_AUTH" "$batch/authorization.json"
cp "$freeze" "$batch/freeze.json"
test "$(sha256sum "$batch/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$batch/freeze.json" | cut -d' ' -f1)" = "$sha"
mounts=()
for item in "${settings[@]:2:4}"; do
    IFS=$'\t' read -r host native <<< "$item"
    mounts+=(--mount "type=bind,src=$host,dst=$native,readonly")
done
cidfile=''
name=''
cleanup_owned() {
    if test -n "$cidfile" && test -s "$cidfile"; then
        local id label
        id=$(cat "$cidfile")
        label=$(docker inspect --format '{{ index .Config.Labels "nerofactor.donut-recovery" }}' "$id" 2>/dev/null || true)
        if test "$label" = "$name"; then
            docker rm --force "$id" >>"$batch/cleanup.txt" 2>&1
        fi
    fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for arm in "${arms[@]}"; do
    cidfile="$batch/$arm.cid"
    name="$(basename "$batch")-$arm"
    anchor_mount=()
    anchor_args=()
    if [[ "$arm" != K1-Dense-* ]]; then
        seed=${arm##*-}
        anchor="$batch/K1-Dense-$seed/initial.json"
        anchor_sha=$(sha256sum "$anchor" | cut -d' ' -f1)
        anchor_mount=(--mount "type=bind,src=$anchor,dst=/inputs/seed-initial.json,readonly")
        anchor_args=(--initial /inputs/seed-initial.json --initial-sha256 "$anchor_sha")
    fi
    printf 'Starting %s\n' "$arm"
    set +e
    timeout --foreground --signal=TERM --kill-after=30s 3600s \
        docker run --rm --pull=never --init --cidfile "$cidfile" \
        --name "$name" --label "nerofactor.donut-recovery=$name" \
        --user "$(id -u):$(id -g)" --gpus device=0 --network none --read-only --cap-drop ALL \
        --security-opt no-new-privileges:true --memory 64g --memory-swap 64g --cpus 4 --pids-limit 128 \
        --shm-size 2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
        --env "HOME=/outputs/$arm/runtime" --env "XDG_CACHE_HOME=/outputs/$arm/runtime/cache" \
        --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
        --mount "type=bind,src=$BELL_TRAIN_SOURCE,dst=/workspace,readonly" \
        --mount "type=bind,src=$batch/authorization.json,dst=/authorization.json,readonly" \
        --mount "type=bind,src=$batch,dst=/outputs" "${mounts[@]}" "${anchor_mount[@]}" \
        --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
        --signal=TERM --kill-after=20s 3540s /usr/bin/python3 -I -B /workspace/experiments/train_bell_publication_seeds_v1.py \
        --phase "$arm" --oracle /inputs/oracle.ply \
        --benchmark /inputs/reference --admission /inputs/admission.json --verification /inputs/verification.json \
        --output "/outputs/$arm" --freeze "/workspace/$freeze_name" --freeze-sha256 "$sha" \
        --authorization /authorization.json --authorization-sha256 "$auth_sha" --commit "$commit" --image-id "$image" \
        "${anchor_args[@]}" --execute >"$batch/$arm-console.txt" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" >"$batch/$arm-exit-code.txt"
    cleanup_owned
    cat "$batch/$arm-console.txt"
    test "$code" -eq 0
    (cd "$BELL_TRAIN_SOURCE" && python3 -B - "$batch" "$arm" "$commit" "$sha" "$auth_sha" <<'PY'
from pathlib import Path
import sys
from benchmarks import bell_publication_seeds_v1 as h
path,arm,commit,sha,auth_sha=sys.argv[1:]
batch=Path(path)
h.completed(batch,arm,h.b.read(batch/'freeze.json'),commit,sha,auth_sha)
PY
    )
done
trap - EXIT
