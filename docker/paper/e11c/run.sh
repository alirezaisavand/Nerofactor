#!/usr/bin/env bash
# Exactly one bounded build + two cold-process BVH probes. No generation/training.
set -euo pipefail
if test "${1:-}" != --supervised; then
  exec timeout --foreground --signal=TERM --kill-after=30s 1800s bash "$0" --supervised
fi
: "${E11C_SOURCE:?}" "${E11C_ROOT:?}" "${E11C_AUTH:?}" "${E11C_GPU:?}"
E11C_BASE=sha256:2572950d9f662b7603b604878613e85c1a0fec90842447e9a256751a130ee137
E11C_TAG=nerofactor-paper:ray-predicate-fp64-e11c-ffbde4d0e261
E11C_JOB="$E11C_ROOT/execution"
E11C_COMMIT=$(git -C "$E11C_SOURCE" rev-parse HEAD)
E11C_FREEZE=$(sha256sum "$E11C_SOURCE/docs/protocols/RAY_BACKEND_E11C_V1.freeze.json" | cut -d' ' -f1)
E11C_AUTH_SHA=$(sha256sum "$E11C_AUTH" | cut -d' ' -f1)
test -z "$(git -C "$E11C_SOURCE" status --porcelain)"
python3 -B - "$E11C_SOURCE" "$E11C_AUTH" "$E11C_AUTH_SHA" "$E11C_FREEZE" "$E11C_COMMIT" "$E11C_ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from infrastructure.ray_backend_e11c import authorize,freeze_check
r=Path(sys.argv[6]);assert r.is_absolute() and r.resolve()==r and not (r/'execution').exists()
authorize(Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5],r)
freeze_check(Path(sys.argv[1])/'docs/protocols/RAY_BACKEND_E11C_V1.freeze.json',sys.argv[4])
PY
test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$E11C_BASE"
test -z "$(docker image ls --quiet "$E11C_TAG")"
mkdir "$E11C_JOB"
docker image inspect --format '{{json .RootFS.Layers}}' "$E11C_BASE" > "$E11C_JOB/base-layers.json"
cleanup_owned() {
  for E11C_CIDFILE in "$E11C_JOB"/*/container.cid; do
    test -s "$E11C_CIDFILE" || continue
    E11C_CID=$(cat "$E11C_CIDFILE")
    if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e11c.owner" }}' "$E11C_CID" 2>/dev/null || true)" = "$(basename "$E11C_ROOT")"; then
      docker rm --force "$E11C_CID" >> "$E11C_JOB/cleanup.log" 2>&1
    fi
  done
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
E11C_COMMON=(--freeze /workspace/docs/protocols/RAY_BACKEND_E11C_V1.freeze.json --freeze-sha256 "$E11C_FREEZE"
 --authorization /authorization.json --authorization-sha256 "$E11C_AUTH_SHA" --commit "$E11C_COMMIT" --host-root "$E11C_ROOT")
run_stage() {
  E11C_STAGE=$1; E11C_IMAGE=$2; E11C_MEMORY=$3; E11C_DEADLINE=$4; shift 4
  mkdir "$E11C_JOB/$E11C_STAGE" "$E11C_JOB/$E11C_STAGE/out"
  set +e
  docker run --rm --pull=never --init --cidfile "$E11C_JOB/$E11C_STAGE/container.cid" \
    --name "$(basename "$E11C_ROOT")-$E11C_STAGE" --label "nerofactor.e11c.owner=$(basename "$E11C_ROOT")" \
    --user "$(id -u):$(id -g)" --read-only --network=none --cap-drop=ALL --security-opt=no-new-privileges:true \
    --memory="$E11C_MEMORY" --memory-swap="$E11C_MEMORY" --cpus=4 --pids-limit=256 --shm-size=1g --ulimit core=0 \
    --tmpfs /tmp:rw,nosuid,nodev,size=512m --env HOME=/outputs/home --env TMPDIR=/tmp \
    --env XDG_CACHE_HOME=/outputs/cache --env TORCH_EXTENSIONS_DIR=/outputs/cache/extensions --env CUDA_CACHE_PATH=/outputs/cache/cuda \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
    --mount "type=bind,src=$E11C_SOURCE,dst=/workspace,readonly" \
    --mount "type=bind,src=$E11C_ROOT/inputs,dst=/inputs,readonly" \
    --mount "type=bind,src=$E11C_AUTH,dst=/authorization.json,readonly" \
    --mount type=bind,src=/home/isavand/paper/final_results/meshes-pretrained/syn/bell.ply,dst=/mesh.ply,readonly \
    --mount "type=bind,src=$E11C_JOB/$E11C_STAGE/out,dst=/outputs" \
    --workdir /outputs "$@" --entrypoint /usr/bin/timeout "$E11C_IMAGE" \
    --signal=TERM --kill-after=20s "${E11C_DEADLINE}s" /usr/bin/python3 -I -B /workspace/run_ray_backend_e11c.py \
    --mode "$([[ "$E11C_STAGE" = build ]] && echo build || echo probe)" \
    $([[ "$E11C_STAGE" = original ]] && echo '--backend original' || true) \
    $([[ "$E11C_STAGE" = corrected ]] && echo '--backend predicate_fp64' || true) "${E11C_COMMON[@]}" \
    > "$E11C_JOB/$E11C_STAGE/console.log" 2>&1
  E11C_EXIT=$?
  set -e
  printf '%s\n' "$E11C_EXIT" > "$E11C_JOB/$E11C_STAGE/exit-code.txt"
  test "$E11C_EXIT" -eq 0 || { cat "$E11C_JOB/$E11C_STAGE/console.log"; return "$E11C_EXIT"; }
}
run_stage build "$E11C_BASE" 16g 900 --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= --env TORCH_CUDA_ARCH_LIST=8.0 --env MAX_JOBS=4 --env CC=/usr/bin/gcc --env CXX=/usr/bin/g++ --env CUDA_HOME=/usr/local/cuda
python3 -B - "$E11C_SOURCE" "$E11C_ROOT" <<'PY'
import sys,shutil,json
from pathlib import Path
source,root=map(Path,sys.argv[1:]);job=root/'execution';ctx=job/'image-context';dst=ctx/'e11c';dst.mkdir(parents=True)
shutil.copytree(job/'build/out/corrected',dst/'corrected');shutil.copyfile(job/'build/out/build.json',dst/'build.json')
shutil.copytree(root/'inputs/source',dst/'source');shutil.copyfile(root/'inputs/source-manifest.json',dst/'source-manifest.json')
shutil.copyfile(source/'docker/paper/patches/raytracing-triangle-fp64-v1.patch',dst/'predicate.patch')
shutil.copyfile(source/'docker/paper/e11c/Dockerfile',ctx/'Dockerfile')
assert sum(p.stat().st_size for p in job.rglob('*') if p.is_file())<=2*1024**3
PY
set +e
timeout --foreground --signal=TERM --kill-after=20s 120s docker build --pull=false --network=none \
  --build-arg E11C_BASE_IMAGE=nerofactor-paper:quick-candidate-1 --tag "$E11C_TAG" "$E11C_JOB/image-context" \
  > "$E11C_JOB/image-build.log" 2>&1
E11C_IMAGE_EXIT=$?
set -e
printf '%s\n' "$E11C_IMAGE_EXIT" > "$E11C_JOB/image-build-exit.txt"
test "$E11C_IMAGE_EXIT" -eq 0
E11C_IMAGE=$(docker image inspect --format '{{.Id}}' "$E11C_TAG")
docker image inspect --format '{{json .RootFS.Layers}}' "$E11C_IMAGE" > "$E11C_JOB/derivative-layers.json"
python3 -B - "$E11C_JOB" "$E11C_BASE" "$E11C_IMAGE" "$E11C_TAG" <<'PY'
import json,sys
from pathlib import Path
job=Path(sys.argv[1]);old=json.loads((job/'base-layers.json').read_text());new=json.loads((job/'derivative-layers.json').read_text())
assert new[:len(old)]==old and sys.argv[2]!=sys.argv[3]
with (job/'image.json').open('x') as f:json.dump(dict(base_id=sys.argv[2],image_id=sys.argv[3],tag=sys.argv[4],base_layers=old,layers=new),f,indent=2);f.write('\n')
PY
test "$(docker image inspect --format '{{.Id}}' nerofactor-paper:quick-candidate-1)" = "$E11C_BASE"
run_stage original "$E11C_IMAGE" 8g 180 --gpus "device=$E11C_GPU"
run_stage corrected "$E11C_IMAGE" 8g 180 --gpus "device=$E11C_GPU"
# Analysis is file-only. Worker outputs live in their separate owned subdirectories.
python3 -B "$E11C_SOURCE/run_ray_backend_e11c.py" --mode analyze --root "$E11C_JOB"
