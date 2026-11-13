"""Read-only continuation verifier admission; no scientific calls."""
import subprocess
import sys

import pytest

from experiments import finalize_neural_material_continuation_v2 as final


def test_verifier_import_never_imports_torch():
    subprocess.run([sys.executable,'-B','-c',
        'import sys; from experiments import finalize_neural_material_continuation_v2; '
        'assert "torch" not in sys.modules'],check=True)


def test_selected_artifact_hash_is_checked_before_json_parsing(tmp_path):
    path=tmp_path/'altered.json';path.write_text('not json')
    with pytest.raises(Exception,match='hash mismatch'):
        final.selected(dict(path=str(path),sha256='0'*64))
