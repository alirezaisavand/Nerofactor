"""Exercise the real shell launcher using fake Docker; no scientific execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from infrastructure.signatures import SignatureError
from infrastructure.stage_sources import ROOT
from run_sphere_surface_experiment import completed_arm


def receipt(arm='surface_chart_control_v1',freeze='f'*64,commit='c'*40):
    return dict(status='complete',protocol='sphere-surface-regularizer-comparison-v1',
                arm=arm,commit=commit,freeze_sha256=freeze,optimizer_returns=8000,
                new_optimizer_returns=4000,cleanup_verified=True,owned_cleanup_verified=True)


def test_completed_receipt_requires_real_work_identity_and_cleanup(tmp_path):
    path=tmp_path/'surface-run.json';args=(path,'surface_chart_control_v1','f'*64,'c'*40)
    with pytest.raises(SignatureError,match='Missing'):completed_arm(*args)
    for key,value in [('status','metadata_only'),('new_optimizer_returns',0),('optimizer_returns',4000),
                      ('cleanup_verified',False),('owned_cleanup_verified',False),
                      ('freeze_sha256','0'*64),('commit','0'*40),('arm','surface_world_q_v1'),
                      ('protocol','foreign')]:
        packet=receipt();packet[key]=value;path.write_text(json.dumps(packet))
        with pytest.raises(SignatureError,match='did not complete'):completed_arm(*args)
    path.write_text(json.dumps(dict(status='metadata_only',scientific_execution=False)))
    with pytest.raises(SignatureError,match='did not complete'):completed_arm(*args)
    path.write_text(json.dumps(receipt()));assert completed_arm(*args)==receipt()
    path.unlink();path.symlink_to(tmp_path/'elsewhere.json')
    (tmp_path/'elsewhere.json').write_text(json.dumps(receipt()))
    with pytest.raises(SignatureError,match='redirected'):completed_arm(*args)


def executable(path,code):
    path.write_text('#!'+sys.executable+'\n'+code);path.chmod(0o700)


@pytest.mark.parametrize('scenario',['complete','metadata_only','missing'])
def test_actual_shell_explicit_execute_and_stop_before_missing_control_mount(tmp_path,scenario):
    fake=tmp_path/'bin';fake.mkdir();source=tmp_path/'source';source.mkdir()
    defs=source/'docs/protocols';defs.mkdir(parents=True)
    checkpoint=tmp_path/'initial.pth';checkpoint.write_bytes(b'not a checkpoint; never deserialized')
    freeze=defs/'SPHERE_SURFACE_COMPARISON_V2.freeze.json'
    freeze.write_text(json.dumps(dict(previous_container_id='d'*64,
                                     initial_checkpoint=dict(host_path=str(checkpoint)))))
    sha=hashlib.sha256(freeze.read_bytes()).hexdigest()
    auth=tmp_path/'authorization.json';auth.write_text('{}')
    calls=tmp_path/'docker-calls.jsonl';batch=tmp_path/'fresh-batch'
    # Bypass ONLY the host source-admission stub in this launcher-routing fixture.
    # The real post-arm completed_arm() function executes in a fresh process.
    executable(fake/'python3',"""import os,sys
code=sys.stdin.read()
if 'preflight,authorize' in code: raise SystemExit(0)
sys.path.insert(0,os.environ['FIXTURE_REAL_SOURCE'])
sys.argv=['-']+sys.argv[3:]
exec(compile(code,'<launcher-stdin>','exec'))
""")
    executable(fake/'git',"""import sys
if 'rev-parse' in sys.argv:print('c'*40)
""")
    executable(fake/'timeout',"""import os,sys
index=sys.argv.index('docker');os.execvp('docker',sys.argv[index:])
""")
    executable(fake/'docker',"""import json,os,sys
from pathlib import Path
a=sys.argv[1:]
if a[:2]==['image','inspect']:
 print('sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331');raise SystemExit(0)
if a[0]=='ps':raise SystemExit(0)
assert a[0]=='run',a
with open(os.environ['FIXTURE_CALLS'],'a') as stream:stream.write(json.dumps(a)+'\\n')
arm=a[a.index('--arm')+1];freeze=a[a.index('--freeze-sha256')+1]
mount=next(x for x in a if x.startswith('type=bind,src=') and ',dst=/outputs' in x)
out=Path(mount.split(',src=',1)[1].split(',dst=',1)[0])
if os.environ['FIXTURE_SCENARIO']=='metadata_only':
 print(json.dumps(dict(status='metadata_only',arm=arm,scientific_execution=False)));raise SystemExit(0)
if os.environ['FIXTURE_SCENARIO']=='missing':raise SystemExit(0)
p=out/'run/surface-run.json';p.parent.mkdir()
p.write_text(json.dumps(dict(status='complete',protocol='sphere-surface-regularizer-comparison-v1',
 arm=arm,commit='c'*40,freeze_sha256=freeze,optimizer_returns=8000,new_optimizer_returns=4000,
 cleanup_verified=True,owned_cleanup_verified=True)))
""")
    env=os.environ.copy();env.update(PATH=str(fake)+os.pathsep+env['PATH'],
        FIXTURE_REAL_SOURCE=str(ROOT),FIXTURE_CALLS=str(calls),FIXTURE_SCENARIO=scenario,
        SURFACE_SOURCE=str(source),SURFACE_DATASETS=str(tmp_path),SURFACE_THESIS=str(tmp_path),
        SURFACE_BATCH=str(batch),SURFACE_AUTH=str(auth),SURFACE_GPU='0',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run(['bash',str(ROOT/'experiments/run_sphere_surface_comparison.sh'),'--supervised'],
                          env=env,text=True,capture_output=True,timeout=20)
    actual=[json.loads(line) for line in calls.read_text().splitlines()]
    assert all('--execute' in args and args[args.index('--freeze-sha256')+1]==sha for args in actual)
    if scenario=='complete':
        assert result.returncode==0,result.stdout+result.stderr
        assert len(actual)==2
        assert '--control' in actual[1]
    else:
        assert result.returncode!=0 and len(actual)==1,result.stdout+result.stderr
        assert not (batch/'surface_world_q_v1').exists()
        assert (batch/'surface_chart_control_v1/exit-code.txt').read_text().strip()=='0'
        assert 'Missing or redirected completed arm report' in result.stderr
