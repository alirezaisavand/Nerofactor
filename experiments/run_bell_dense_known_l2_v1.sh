#!/usr/bin/env bash
# One foreground Dense Bell batch under its exact admission and execution records.
set -euo pipefail
: "${BELL_TRAIN_SOURCE:?}" "${BELL_TRAIN_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$BELL_TRAIN_SOURCE")" = "$BELL_TRAIN_SOURCE"
test -z "$(git -C "$BELL_TRAIN_SOURCE" status --porcelain)"
commit=$(git -C "$BELL_TRAIN_SOURCE" rev-parse HEAD)
freeze="$BELL_TRAIN_SOURCE/docs/protocols/BELL_DENSE_KNOWN_L2_RECOVERY_V1.freeze.json"
arms=(K1-Dense K2-Dense K3-Dense)
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$BELL_TRAIN_AUTH" | cut -d' ' -f1)
spec=$(cd "$BELL_TRAIN_SOURCE" && python3 -B - "$BELL_TRAIN_AUTH" "$commit" "$sha" <<'PY'
from pathlib import Path
import sys
from infrastructure.signatures import require
from benchmarks import donut_heterogeneous_v1 as ref

from benchmarks import bell_dense_known_l2_v1 as h

auth,commit,sha=sys.argv[1:]
f=h.freeze(h.ROOT/h.FREEZE,sha)
require(ref.read(Path(auth))==h.authorization(f,commit,sha),'Exact separate Bell training approval required')
out=Path(f['host_output'])
require(out.resolve()==out and out.parent.is_dir() and not out.exists(),'Fresh Bell training batch required')
files=f['dataset']
h.geometry(f,Path(f['mesh']['path']))
h.dataset(f,Path(f['host_reference']),Path(files['admission']['path']),Path(files['verification']['path']))
ref.check_artifact(Path(f['base_initial']['path']),f['base_initial'])
h.verify_controls(f)
ref.check_artifact(Path(f['held_out_support']['path']),f['held_out_support'])
print(out)
print(h.IMAGE)
print(f'{f["mesh"]["path"]}\t/inputs/oracle.ply')
print(f'{f["host_reference"]}\t/inputs/reference')
for key in ('admission','verification'):
    print(f'{files[key]["path"]}\t/inputs/{key}.json')
print(f'{f["base_initial"]["path"]}\t/inputs/base-initial.json')
for arm in h.ARMS:
    item=f['arms'][arm]['initial']
    ref.check_artifact(Path(item['path']),item)
    print(item['path'])
PY
)
mapfile -t settings <<< "$spec"
batch=${settings[0]}
image=${settings[1]}
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
for item in "${settings[@]:2:5}"; do
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
for i in "${!arms[@]}"; do
    arm=${arms[$i]}
    initial=${settings[$((7+i))]}
    cidfile="$batch/$arm.cid"
    name="$(basename "$batch")-$arm"
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
        --mount "type=bind,src=$initial,dst=/inputs/arm-initial.json,readonly" \
        --mount "type=bind,src=$batch,dst=/outputs" "${mounts[@]}" \
        --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
        --signal=TERM --kill-after=20s 3540s /usr/bin/python3 -I -B /workspace/experiments/train_bell_dense_known_l2_v1.py \
        --phase "$arm" --oracle /inputs/oracle.ply \
        --benchmark /inputs/reference --admission /inputs/admission.json --verification /inputs/verification.json \
        --initial /inputs/base-initial.json --arm-initial /inputs/arm-initial.json --output "/outputs/$arm" \
        --freeze "/workspace/$freeze_name" --freeze-sha256 "$sha" \
        --authorization /authorization.json --authorization-sha256 "$auth_sha" --commit "$commit" --image-id "$image" --execute \
        >"$batch/$arm-console.txt" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" >"$batch/$arm-exit-code.txt"
    cleanup_owned
    cat "$batch/$arm-console.txt"
    test "$code" -eq 0
    python3 -B - "$batch" "$arm" "$commit" "$sha" "$auth_sha" <<'PY'
import json,sys
from pathlib import Path
path,arm,commit,sha,auth_sha=sys.argv[1:]
batch=Path(path)
f=json.loads((batch/'freeze.json').read_text())
r=json.loads((batch/arm/'recovery.json').read_text())
assert r['arm']==arm and r['status']=='complete' and r['scientific_work_complete'] and r['cleanup_verified']
assert r['commit']==commit and r['freeze_sha256']==sha and r['authorization_sha256']==auth_sha
assert r['optimizer_returns']==10000 and r['historical_arm_initial_parity_verified']
assert r['dataset_admission_sha256']==f['dataset']['admission']['sha256']
assert r['outputs']['photometric-indices.u32le']==dict(bytes=20480000,sha256=f['photometric_index_sha256'])
assert r['counters']==f['arms'][arm]['work'] and [x['step'] for x in r['saved_endpoints']]==f['steps']
assert r['known_lighting_audit']['learned_module_calls']==0
PY
done
trap - EXIT
