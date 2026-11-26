"""Focused checks of cache routing and executable library loading. No CUDA/model."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from benchmarks import bell_geometry_gate10_v1 as g


def test_launcher_routes_only_extension_cache_to_owned_output():
    source=(g.ROOT/'experiments/run_bell_geometry_gate10_v1.sh').read_text()
    tokens=shlex.split(source.replace('\\\n',' '))
    values=[tokens[i+1] for i,t in enumerate(tokens[:-1]) if t=='--env']
    assert values.count('TORCH_EXTENSIONS_DIR=/outputs/torch_extensions')==1
    assert 'HOME=/tmp' in values and 'XDG_CACHE_HOME=/tmp' in values
    assert '/tmp:rw,nosuid,nodev,noexec,size=512m' in tokens
    assert 'type=bind,src=$out/$phase,dst=/outputs' in tokens


def test_cache_guard_precedes_scientific_execution_and_hashes_artifacts():
    tree=ast.parse((g.ROOT/'experiments/bell_geometry_gate10_v1.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native')
    guards=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id=='require' and 'TORCH_EXTENSIONS_DIR' in ast.dump(n)]
    assert len(guards)==1
    activations=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='activate']
    assert guards[0].lineno<min(n.lineno for n in activations)
    assert 'runtime_extension_files' in ast.dump(function)


def test_owned_evidence_mount_supports_cpu_shared_library_mapping():
    # Match /outputs' host filesystem without creating any scientific run directory.
    with tempfile.TemporaryDirectory(prefix='.gate10-cache-cpu-',dir=g.ARCHIVE.parent) as directory:
        cache=Path(directory)/'torch_extensions'
        cache.mkdir()
        source=cache/'fixture.c'
        library=cache/'fixture.so'
        source.write_text('int fixture_value(void) { return 77; }\n')
        built=subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(library)],
                             capture_output=True,text=True,timeout=30)
        assert built.returncode==0,built.stderr
        env=dict(os.environ,TORCH_EXTENSIONS_DIR=str(cache))
        script="""import ctypes,os
from pathlib import Path
p=Path(os.environ['TORCH_EXTENSIONS_DIR'])/'fixture.so'
assert ctypes.CDLL(str(p)).fixture_value()==77
"""
        loaded=subprocess.run(['python3','-B','-c',script],env=env,capture_output=True,text=True,timeout=10)
        assert loaded.returncode==0,loaded.stderr
