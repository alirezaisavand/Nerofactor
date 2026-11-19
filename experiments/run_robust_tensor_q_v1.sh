#!/usr/bin/env bash
# Run the two admitted acquisitions in one foreground, source-bound batch.
set -euo pipefail
: "${ROBUST_SOURCE:?}" "${ROBUST_AUTH:?}"
test "$(id -u)" -ne 0
test "$(realpath "$ROBUST_SOURCE")" = "$ROBUST_SOURCE"
test -z "$(git -C "$ROBUST_SOURCE" status --porcelain)"
commit=$(git -C "$ROBUST_SOURCE" rev-parse HEAD)
freeze_name=docs/protocols/ROBUST_TENSOR_Q_TRAINING_V1.freeze.json
freeze="$ROBUST_SOURCE/$freeze_name"
sha=$(sha256sum "$freeze" | cut -d' ' -f1)
auth_sha=$(sha256sum "$ROBUST_AUTH" | cut -d' ' -f1)
spec=$(cd "$ROBUST_SOURCE" && python3 -B - "$ROBUST_AUTH" "$commit" "$sha" <<'PYTHON'
from pathlib import Path
import sys

from benchmarks import robust_tensor_q_training_v1 as h
from benchmarks.donut_heterogeneous_v1 import check_artifact, read
from infrastructure.signatures import require

auth,commit,sha=sys.argv[1:]
f=h.freeze(h.ROOT/h.FREEZE,sha)
require(read(Path(auth))==h.authorization(f,commit,sha),'Exact separate robust tensor-Q training authorization required')
out=Path(f['host_output'])
require(out.resolve()==out and out.parent.is_dir() and not out.exists(),'Fresh robust batch required')
h.verify_calibration(f)
print(out)
print(h.IMAGE)
for arm in h.ARMS:
    acquired=h.Acquisition(arm)
    scene=f['scenes'][arm]
    paths=acquired.host_inputs(scene)
    reference=Path(scene['host_reference'])
    admission,verification=(Path(scene['dataset'][k]['path']) for k in ('admission','verification'))
    acquired.dataset(scene,reference,admission,verification)
    base,initial=scene['base_initial'],scene['recipe']['initial']
    for item in (base,initial):
        check_artifact(Path(item['path']),item)
    row=(arm,paths['oracle'],paths.get('cameras','-'),reference,admission,verification,base['path'],initial['path'])
    print('\t'.join(map(str,row)))
PYTHON
)
mapfile -t settings <<< "$spec"
batch=${settings[0]}
image=${settings[1]}
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
test -z "$(docker ps -q --filter label=nerofactor.donut-recovery)"
test -z "$(docker ps -q --filter label=nerofactor.donut-heterogeneous)"
test -z "$(docker ps -q --filter label=nerofactor.bell-heterogeneous)"
mkdir "$batch"
cp "$ROBUST_AUTH" "$batch/authorization.json"
cp "$freeze" "$batch/freeze.json"
test "$(sha256sum "$batch/authorization.json" | cut -d' ' -f1)" = "$auth_sha"
test "$(sha256sum "$batch/freeze.json" | cut -d' ' -f1)" = "$sha"
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
for row in "${settings[@]:2}"; do
    IFS=$'\t' read -r arm oracle cameras reference admission verification base_initial initial <<< "$row"
    mounts=(
        --mount "type=bind,src=$oracle,dst=/inputs/oracle,readonly"
        --mount "type=bind,src=$reference,dst=/inputs/reference,readonly"
        --mount "type=bind,src=$admission,dst=/inputs/admission.json,readonly"
        --mount "type=bind,src=$verification,dst=/inputs/verification.json,readonly"
        --mount "type=bind,src=$base_initial,dst=/inputs/base-initial.json,readonly"
    )
    native_cameras=()
    if test "$cameras" != '-'; then
        mounts+=(--mount "type=bind,src=$cameras,dst=/inputs/cameras.json,readonly")
        native_cameras=(--cameras /inputs/cameras.json)
    fi
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
        --mount "type=bind,src=$ROBUST_SOURCE,dst=/workspace,readonly" \
        --mount "type=bind,src=$batch/authorization.json,dst=/authorization.json,readonly" \
        --mount "type=bind,src=$initial,dst=/inputs/arm-initial.json,readonly" \
        --mount "type=bind,src=$batch,dst=/outputs" "${mounts[@]}" \
        --workdir /workspace --entrypoint /usr/bin/timeout "$image" \
        --signal=TERM --kill-after=20s 3540s /usr/bin/python3 -I -B /workspace/experiments/train_robust_tensor_q_v1.py \
        --phase "$arm" --oracle /inputs/oracle "${native_cameras[@]}" \
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
scene=f['scenes'][arm]
assert r['counters']==scene['recipe']['work'] and [x['step'] for x in r['saved_endpoints']]==scene['steps']
assert r['surface_q_penalty']==dict(name='pseudo_huber_v1',delta=scene['recipe']['robust_delta'])
PY
done
trap - EXIT
