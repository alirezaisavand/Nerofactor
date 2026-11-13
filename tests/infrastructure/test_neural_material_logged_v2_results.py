"""New saved-data reporting checks only; no scientific experiment execution."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_neural_material_recovery_logged_v2_results as audit


def test_offline_analysis_does_not_import_torch():
    subprocess.run([sys.executable, '-B', '-c',
                    'import sys; from experiments import analyze_neural_material_recovery_logged_v2_results; '
                    'assert "torch" not in sys.modules'], check=True)


def test_lost_session_is_not_native_censor_or_endpoint():
    row = dict(status='optimizing', completed_updates=4270)
    assert audit.outcome(row) == 'interrupted_committed_prefix'
    assert audit.outcome(dict(status='admitted', completed_updates=0)) == 'unstarted'
    with pytest.raises(Exception, match='Incomplete endpoint claim'):
        audit.outcome(dict(status='complete', completed_updates=4270, evaluation={}))
    with pytest.raises(Exception, match='endpoint data'):
        audit.outcome(dict(row, evaluation={}))


def test_physical_failure_is_still_completed_execution():
    row = dict(status='complete', completed_updates=5000,
               evaluation=dict(success=dict(all_passed=False)))
    assert audit.outcome(row) == 'complete'


def test_wrong_hash_and_unsafe_path_fail_before_parsing(tmp_path):
    p = tmp_path / 'evidence.json'
    p.write_text('{invalid')
    with pytest.raises(Exception, match='hash mismatch'):
        audit.selected(p, '0' * 64)
    link = tmp_path / 'link.json'
    link.symlink_to(p)
    with pytest.raises(Exception, match='unaliased'):
        audit.selected(link, audit.saved.digest(p))


def test_logged_zero_gradients_and_preupdate_labels_are_retained():
    records = [dict(update=i, loss_at_update=i-1, loss=loss,
                    gradient_norm=g, physical_before_update=[1.] * 7,
                    active_wall_seconds=float(i), material_before_update={}, gradient_groups={})
               for i, loss, g in [(1, .02, .01), (10, .03, 0.), (20, .03, 0.)]]
    value = audit.trajectory(dict(history=records, completed_updates=20))
    assert value['first_zero_gradient_logged_update'] == 10
    assert value['all_later_logged_gradients_zero'] is True
    assert value['last_logged_loss'] == .03
    assert value['records'][-1]['loss_at_update'] == 19
    records[-1]['gradient_norm'] = float('nan')
    with pytest.raises(Exception, match='Nonfinite'):
        audit.trajectory(dict(history=records, completed_updates=20))


def test_object_checkpoint_archive_is_never_unpickled(tmp_path):
    archive = tmp_path / 'synthetic.npz'
    np.savez(archive, tensor=np.asarray({'forbidden': True}, dtype=object),
             numpy_rng=np.zeros(624, np.uint32))
    freeze = dict(configuration_sha256='config', sha256={'file': 'hash'},
                  configuration=dict(optimizer_effective={}))
    row = dict(arm='D0', initial_model=dict(source={}, configuration={}))
    meta = dict(arm='D0', source={}, model_configuration={}, completed_updates=250,
                optimizer_effective={}, archive=archive.name,
                archive_sha256=audit.saved.digest(archive), archive_bytes=archive.stat().st_size,
                execution_provenance=dict(freeze_sha256=audit.FREEZE_SHA,
                    configuration_sha256='config', source_sha256={'file': 'hash'}, run_id='fixture'),
                entries=[dict(key='tensor', role='parameter', name='latent', shape=[],
                              dtype='torch.object', device='cpu', sha256='unused', finite=True)])
    manifest = tmp_path / 'synthetic.json'
    manifest.write_text(json.dumps(meta))
    selection = dict(manifest=manifest.name, manifest_sha256=audit.saved.digest(manifest),
                     completed_updates=250)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='allow_pickle=False'):
        audit.inspect_numeric_checkpoint(tmp_path, selection, row, freeze, 'fixture')
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before
