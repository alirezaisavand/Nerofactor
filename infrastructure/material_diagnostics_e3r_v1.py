"""Representation-aware, value-preserving diagnostics; no model/shader imports."""
import ast
import copy
from functools import lru_cache
import hashlib
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
from infrastructure.signatures import require

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'experiments/material_recovery_metrics_v1.py':'316e79b5e06615c9fa46e714f35f515b204c1f175c6c56636a6c4d9235ccabc4',
    'experiments/multi_material_recovery_v1.py':'bd1d406f374c436375d58c987940d8364c2dc433f72b5445a1a93e8172e30ef4',
    'experiments/multi_material_reference_v1.py':'cabf7d60648f8daa6cc3b1f6bef8fec07f7ac5e4b4e4b5d1d8679a195ff7dd5f'}
PROTOCOL='e3-recovery-representation-aware-diagnostics-v1'
FLOAT32_FLOOR=float(np.float32(.04))


def validate_material(value):
    """Validate in the input's declared precision, then promote without changing values."""
    p=np.asarray(value)
    require(p.dtype in (np.dtype('float32'),np.dtype('float64')) and p.shape==(7,) and np.isfinite(p).all(),
            'Finite float32/float64 physical material required')
    floor=FLOAT32_FLOOR if p.dtype==np.float32 else .04
    q=p.astype(np.float64)
    require(((q[:2]>=floor)&(q[:2]<=1)).all() and ((q[3:]>=0)&(q[3:]<=1)).all(),
            'Invalid physical material domain for declared representation')
    return q


def source_node(path,name):
    data=(ROOT/path).read_bytes()
    require(hashlib.sha256(data).hexdigest()==SOURCES[path],'Frozen diagnostic source changed: '+path)
    node=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name==name)
    require(not node.decorator_list and not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(node)),
            'Unexpected diagnostic definition dependency')
    return node


@lru_cache(maxsize=1)
def diagnostic_functions():
    """Private AST namespace: original formulas, two explicit lower-bound guard replacements.

    No original module is imported or monkeypatched. Only roughness_tensor's
    domain guard changes; entry validation retains the original input precision.
    """
    path='experiments/material_recovery_metrics_v1.py'
    rough=source_node(path,'roughness_tensor');original=copy.deepcopy(rough)
    # Docstring, domain require, then unchanged tensor formula.
    guard=rough.body[1];require(isinstance(guard,ast.Expr) and isinstance(guard.value,ast.Call) and
        isinstance(guard.value.func,ast.Name) and guard.value.func.id=='require','Unexpected roughness domain guard')
    bounds=[n for n in ast.walk(guard) if isinstance(n,ast.Constant) and n.value==.04]
    require(len(bounds)==2,'Expected exactly two roughness lower-bound guards')
    for n in bounds:n.value=FLOAT32_FLOOR
    require(ast.dump(ast.Module(body=rough.body[2:],type_ignores=[]))==
        ast.dump(ast.Module(body=original.body[2:],type_ignores=[])),'Diagnostic tensor mathematics changed')
    gauge=source_node(path,'gauge_metrics')
    errors=source_node('experiments/multi_material_recovery_v1.py','recovery_errors')
    ns=dict(np=np,require=require,PROTOCOL='known-material-recovery-metrics-v1',ANISOTROPY_GAP=.1,
            reference=SimpleNamespace(validate=validate_material))
    nodes=[rough,gauge,errors]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__,'exec'),ns)
    return ns


def recovery_errors(predicted,truth):
    # Preserve origin dtype through admission; no float32 recast of float64 inputs.
    return diagnostic_functions()['recovery_errors'](predicted,truth)


def evaluation_adapter(original):
    """Same frozen evaluation code, private diagnostic dependency; original globals untouched."""
    previous=original.__globals__['previous']
    class DiagnosticPrevious:
        recovery_errors=staticmethod(recovery_errors)
        def __getattr__(self,name):return getattr(previous,name)
    ns=dict(original.__globals__,previous=DiagnosticPrevious())
    result=FunctionType(original.__code__,ns,original.__name__,original.__defaults__,original.__closure__)
    result.__kwdefaults__=original.__kwdefaults__
    require(result.__code__ is original.__code__ and original.__globals__['previous'] is previous,
            'Original evaluation code/globals changed')
    return result
