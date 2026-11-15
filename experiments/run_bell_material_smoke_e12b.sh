#!/usr/bin/env bash
# One approved two-return smoke through the existing original-A Trainer adapter.
set -euo pipefail
: "${E12B_SOURCE:?}" "${E12B_BENCHMARK:?}" "${E12B_ADMISSION:?}" "${E12B_BATCH:?}" "${E12B_AUTH:?}" "${E12B_RUN_ID:?}" "${E12B_GPU:?}"
E12B_IMAGE=sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331
E12B_COMMIT=$(git -C "$E12B_SOURCE" rev-parse HEAD)
test -z "$(git -C "$E12B_SOURCE" status --porcelain)"
E12B_FREEZE=$(sha256sum "$E12B_SOURCE/docs/protocols/BELL_MATERIAL_SMOKE_E12B_V1.freeze.json" | cut -d' ' -f1)
E12B_AUTH_SHA=$(sha256sum "$E12B_AUTH" | cut -d' ' -f1)
python3 -B - "$E12B_SOURCE" "$E12B_AUTH" "$E12B_BATCH" "$E12B_RUN_ID" "$E12B_COMMIT" "$E12B_FREEZE" "$E12B_IMAGE" "$E12B_BENCHMARK" "$E12B_ADMISSION" "$E12B_GPU" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from infrastructure.signatures import require
from infrastructure.stage_sources import digest,PINS
from infrastructure.stage_execution import plan
from infrastructure.material_variants import admit_oracle
root=Path(sys.argv[1]);a=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3]);bench=Path(sys.argv[8]);admission=Path(sys.argv[9])
f=json.loads((root/'docs/protocols/BELL_MATERIAL_SMOKE_E12B_V1.freeze.json').read_text())
require(a.get('approved') is True and a.get('protocol')==f['protocol']=='bell-native-core-two-update-e12b-v1' and
 a.get('commit')==sys.argv[5] and a.get('freeze_sha256')==sys.argv[6] and a.get('image_id')==f['image_id']==sys.argv[7] and
 a.get('host_output_directory')==str(out) and a.get('run_id')==sys.argv[4] and a.get('limits')==f['limits'] and
 a.get('admission_sha256')==f['admission']['sha256']==digest(admission) and a.get('selected_gpu')==sys.argv[10]=='0',
 'Exact separate native smoke authorization required')
require(out.is_absolute() and out.resolve()==out and not out.exists(),'Fresh explicit smoke batch required')
require(all(digest(root/p)==sha for p,sha in f['files'].items()),'Frozen source changed')
require(str(bench)==f['host_benchmark'] and str(admission)==f['admission']['path'],'Explicit admitted inputs required')
p=plan('material',PINS['material'],'C',bench/'datasets',Path('/home/isavand/paper/final_results'),out/'run',
 root/'configs/experiments/material_a_bell_core_known_v1.yaml',root,
 {'geometry':bench/'geometry','admission':admission.parent},material_variant='A_corrected_core_v1',oracle_benchmark=True)
proof=admit_oracle(p,admission)
require(proof['admission']['sha256']==f['admission']['sha256'] and not any(n.split('.')[0] in
 ('torch','network','dataset','train','raytracing','_raytracing') for n in sys.modules),'Cold admission or scientific import boundary failed')
PY
test "$(docker image inspect --format '{{.Id}}' "$E12B_IMAGE")" = "$E12B_IMAGE"
mkdir "$E12B_BATCH"
mkdir "$E12B_BATCH/run"
python3 -B - "$E12B_BATCH" "$E12B_SOURCE" "$E12B_BENCHMARK" "$E12B_ADMISSION" "$E12B_RUN_ID" "$E12B_COMMIT" "$E12B_FREEZE" "$E12B_AUTH_SHA" "$E12B_GPU" <<'PY'
import json,sys
from pathlib import Path
with (Path(sys.argv[1])/'launch.json').open('x') as stream:
 json.dump(dict(zip(['batch','source','benchmark','admission','run_id','commit','freeze_sha256','authorization_sha256','gpu'],sys.argv[1:])),stream,indent=2);stream.write('\n')
PY
E12B_CONTAINER=$(basename "$E12B_BATCH")
cleanup_owned() {
 test -s "$E12B_BATCH/container.cid" || return 0
 E12B_CID=$(cat "$E12B_BATCH/container.cid")
 if test "$(docker container inspect --format '{{ index .Config.Labels "nerofactor.e12b.run" }}' "$E12B_CID" 2>/dev/null || true)" = "$E12B_CONTAINER"; then
  docker rm --force "$E12B_CID" >> "$E12B_BATCH/cleanup.log" 2>&1
 fi
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
timeout --foreground --signal=TERM --kill-after=30s 900s \
 docker run --rm --pull=never --init --cidfile "$E12B_BATCH/container.cid" \
 --name "$E12B_CONTAINER" --label "nerofactor.e12b.run=$E12B_CONTAINER" \
 --user "$(id -u):$(id -g)" --gpus "device=$E12B_GPU" --network=none --read-only --cap-drop=ALL \
 --security-opt=no-new-privileges:true --memory=64g --memory-swap=64g --cpus=4 --pids-limit=128 \
 --shm-size=2g --ulimit core=0 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m \
 --env HOME=/outputs --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$E12B_SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$E12B_BENCHMARK/datasets,dst=/inputs/datasets,readonly" \
 --mount "type=bind,src=$E12B_BENCHMARK/geometry,dst=/inputs/geometry,readonly" \
 --mount "type=bind,src=$(dirname "$E12B_ADMISSION"),dst=/inputs/admission,readonly" \
 --mount type=bind,src=/home/isavand/paper/final_results,dst=/inputs/thesis,readonly \
 --mount "type=bind,src=$E12B_BATCH/run,dst=/outputs" \
 --workdir /workspace --entrypoint /usr/bin/timeout "$E12B_IMAGE" \
 --signal=TERM --kill-after=20s 840s /usr/bin/python3 -I -B /workspace/run_stage_experiment.py \
 --stage material --scientific-commit e1e859779301faced46d6476691e20b546c27ae0 \
 --arm C --material-variant A_corrected_core_v1 --material-smoke \
 --cfg /workspace/configs/experiments/material_a_bell_core_known_v1.yaml --config-root /workspace \
 --datasets-root /inputs/datasets --thesis-root /inputs/thesis --output-root /outputs \
 --input-root geometry=/inputs/geometry --input-root admission=/inputs/admission \
 --oracle-benchmark /inputs/admission/oracle-admission.json \
 --run-id "$E12B_RUN_ID" --execute --confirm-isolated-runtime \
 --image-id "$E12B_IMAGE" --infrastructure-commit "$E12B_COMMIT" > "$E12B_BATCH/console.log" 2>&1
E12B_EXIT=$?
set -e
printf '%s\n' "$E12B_EXIT" > "$E12B_BATCH/exit-code.txt"
cleanup_owned
trap - EXIT
cat "$E12B_BATCH/console.log"
exit "$E12B_EXIT"
