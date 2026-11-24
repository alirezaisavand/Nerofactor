
import numpy as np
import pytest

from benchmarks import bell_dense_known_l2_v1 as d
from experiments.train_bell_dense_known_l2_v1 import RecordedPositions
from infrastructure.signatures import SignatureError


def test_admission_candidate_remains_pending_and_expected_record_is_bound():
    pending=d.b.read(d.ROOT/d.CANDIDATE)
    assert not pending['researcher_admitted_dataset'] and not pending['training_authorized']
    accepted=d.admission_record(pending)
    assert accepted['researcher_admitted_dataset'] and not accepted['training_authorized']
    assert d.json_identity(accepted)['sha256']==d.ADMISSION
    assert pending['scope']==list(d.ARMS) and pending['splits']['held_out']==[105,114]


def test_dense_pool_uses_ordered_fitting_views_and_only_native_observations(monkeypatch):
    monkeypatch.setattr(d,'SPLITS',dict(fitting=[4,0],held_out=[105,114]))
    def view(camera):
        return dict({k:np.array([camera,-1,camera]) for k in d.FIT_KEYS},
                    shader_valid=np.array([True,False,True]), **{'Q_world':None,'material_region':None})
    pool=d.fit_pool({c:view(c) for c in (0,4,105,114)})
    assert tuple(pool)==d.FIT_KEYS
    assert all(np.array_equal(v,[4,4,0,0]) for v in pool.values())


def test_exact_shuffle_resets_and_drops_the_incomplete_tail():
    rng=np.random.RandomState(6033)
    order=rng.permutation(9)
    second=rng.permutation(9)
    expected=[order[:4],order[4:8],second[:4],second[4:8]]
    actual=list(d.index_batches(9,6033,4,4))
    for got,want in zip(actual,expected):
        np.testing.assert_array_equal(got,want)
    assert len(actual)==4


def test_recorded_positions_verify_native_indices_and_return_ordinary_array(tmp_path):
    values=np.arange(1026*3).reshape(1026,3)
    observed=RecordedPositions(values,tmp_path)
    first=next(d.index_batches(len(values),6033,1,512))
    actual=observed[first]
    assert type(actual) is np.ndarray
    np.testing.assert_array_equal(actual,values[first])
    assert (tmp_path/d.STREAM).read_bytes()==first.astype(d.INDEX_DTYPE).tobytes()
    with pytest.raises(SignatureError,match='Native fitting indices differ'):
        observed[first]
    with pytest.raises(SignatureError,match='Incomplete'):
        observed.verify(d.index_hash(len(values)))


@pytest.mark.parametrize('arm',d.ARMS)
def test_recipe_changes_only_evaluation_work(arm):
    sparse=d.b.read(d.ROOT/d.known.FREEZE)
    original=sparse['arms'][d.MATCH[arm]]
    actual=d.recipe(arm,sparse)
    changed={'shader_calls','shadow_rays'}
    for k in ('work','limits'):
        assert {name for name in original[k] if original[k][name]!=actual[k][name]}==changed
    assert {k:v for k,v in original.items() if k not in ('work','limits')}=={
        k:v for k,v in actual.items() if k not in ('work','limits')}
    assert actual['limits']['updates_per_arm']==10000
    assert actual['work']['shader_calls']==(17184 if arm=='K1-Dense' else 17186)


@pytest.mark.parametrize('arm',d.ARMS)
def test_runtime_configures_the_original_native_route_and_dense_admission(arm,monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    sparse=d.b.read(d.ROOT/d.known.FREEZE)
    plan=d.recipe(arm,sparse)
    f=dict(arms={arm:plan},base_initial={'sha256':'base'},integer_domain_hashes={'face_pairs':'bound'},
           sampler_snapshots={'0':'count'})
    mount=Path('/outputs')
    args=SimpleNamespace(phase=arm,freeze=Path('/freeze.json'),freeze_sha256='freeze',
        authorization=Path('/authorization.json'),authorization_sha256='auth',commit='commit',image_id=d.IMAGE,
        output=mount/arm,oracle=Path('/oracle.ply'),benchmark=Path('/benchmark'),admission=Path('/admission.json'),
        verification=Path('/verification.json'),initial=Path('/initial.json'),arm_initial=Path('/arm-initial.json'),
        initial_sha256='base',benchmark_sha256=d.GENERATION)
    monkeypatch.setattr(d,'freeze',lambda *_:f)
    monkeypatch.setattr(d,'authorization',lambda *_:{'approved':True})
    monkeypatch.setattr(d.b,'read',lambda *_:{'approved':True})
    monkeypatch.setattr(d.b,'digest',lambda *_:'auth')
    monkeypatch.setattr(d.b,'check_artifact',lambda path,_:path)
    monkeypatch.setattr(d.os,'geteuid',lambda:1000)
    monkeypatch.setattr(d.os,'statvfs',lambda path:SimpleNamespace(f_flag=0 if Path(path)==mount else d.os.ST_RDONLY))
    monkeypatch.setattr(Path,'read_text',lambda _: 'header\nheader\nlo: 0\n')
    active,auth=d.runtime(args)
    assert auth=={'approved':True}
    assert active['arm_training']['benchmark_generation_sha256']==d.GENERATION
    assert active['arm_training']['learned_control']==d.known.MATCH[d.MATCH[arm]]
    assert d.ADMISSION!=d.known.ADMISSION
    if arm=='K1-Dense':
        assert 'tensor_training' not in active
    else:
        assert active['tensor_training']['arms']==['D8-C','D8-T']


def test_launcher_passes_the_same_freeze_to_docker(tmp_path):
    """Exercise the actual shell expansion with a failing stand-in container."""
    import os
    import subprocess
    from hashlib import sha256

    repo = tmp_path/'source'
    repo.mkdir()
    freeze = repo/d.FREEZE
    freeze.parent.mkdir(parents=True)
    freeze.write_text('{}\n')
    authorization = tmp_path/'authorized-fixture.json'
    authorization.write_text('{}\n')
    batch = tmp_path/'batch'

    # The subprocess remains host-only. Stub preflight verification, image
    # inspection and Docker, while using the actual launcher under set -u.
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()

    def stub(name, body):
        path = bin_dir/name
        path.write_text('#!/bin/sh\n'+body+'\n')
        path.chmod(0o755)

    stub('id','case "$1" in -u|-g) printf "1000\\n";; *) exit 2;; esac')
    stub('git','case "$3" in status) exit 0;; rev-parse) printf "fixture-commit\\n";; *) exit 2;; esac')
    stub('python3','cat >/dev/null\nprintf "%s\\n" "$FIXTURE_PREFLIGHT"')
    stub('docker',"""case "$1" in
    image) printf '%s\\n' "$FIXTURE_IMAGE";;
    ps) exit 0;;
    run) printf '%s\\n' "$@" > "$FIXTURE_ARGV"; exit 23;;
    *) exit 2;;
esac""")
    argv_file = tmp_path/'docker-argv'
    expected_mounts = ('oracle.ply','reference','admission.json','verification.json','base-initial.json')
    preflight = '\n'.join([str(batch),d.IMAGE,
                           *(f'{tmp_path/name}\t/inputs/{name}' for name in expected_mounts),
                           *(str(tmp_path/(arm+'-initial.json')) for arm in d.ARMS)])
    env = dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],BELL_TRAIN_SOURCE=str(repo),
               BELL_TRAIN_AUTH=str(authorization),FIXTURE_PREFLIGHT=preflight,FIXTURE_IMAGE=d.IMAGE,
               FIXTURE_ARGV=str(argv_file))
    launch = d.ROOT/'experiments/run_bell_dense_known_l2_v1.sh'
    result = subprocess.run(['bash',str(launch)],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'unbound variable' not in result.stderr, result.stderr
    args = argv_file.read_text().splitlines()
    assert args[args.index('--freeze')+1]=='/workspace/'+d.FREEZE
    assert args[args.index('--freeze-sha256')+1]==sha256(freeze.read_bytes()).hexdigest()
    assert args[args.index('--phase')+1]=='K1-Dense'
    assert (batch/'K1-Dense-exit-code.txt').read_text().strip()=='23'
    assert not (batch/'K2-Dense-exit-code.txt').exists()
