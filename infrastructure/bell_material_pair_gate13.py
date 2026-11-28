"""Read-only illumination observation and RNG tape accounting for Gate 13."""
from contextlib import contextmanager
from hashlib import sha256
import json
import time
from pathlib import Path
from types import FunctionType
from infrastructure.known_l2_audit import KnownL2Audit,LIGHT_POLICY
from infrastructure.signatures import require
from benchmarks.bell_e11 import ENV

# Reuse every accepted guard, in a PRIVATE function namespace. Do not mutate
# LIGHT_POLICY or the existing KnownL2Audit used by historical experiments.
_base=KnownL2Audit.attached.__wrapped__
_attached=FunctionType(_base.__code__,dict(_base.__globals__,LIGHT_POLICY=dict(LIGHT_POLICY,environment=ENV)),
                       name='bell_v3_known_light_attached')


class KnownBellV3Audit(KnownL2Audit):
    attached=contextmanager(_attached)


def same_record(a,b):
    require(a==b,'Actual per-step sampling tape differs from paired oracle history')


class SamplingAudit:
    """No reseeding/replacement draws. Observe original rand/normal requests.

    Full CUDA Philox state + exact request shapes/dtypes identify each tape in
    this pinned runtime. Hash actual random outputs at six predeclared steps.
    """
    PROBES={0,1999,3999,5999,7999,9999}
    def __init__(self,output,reference=None):
        self.path=Path(output)/'sampling-tapes.jsonl';self.steps=0
        self.observer_seconds=0.
        self.stream=self.path.open('x')
        self.reference=Path(reference).open() if reference is not None else None

    @contextmanager
    def step(self,step,torch):
        require(step==self.steps<10000,'Sampling observer step order differs')
        requests=[];original={name:getattr(torch,name) for name in ('rand','normal')}
        def state():return sha256(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()).hexdigest()
        def wrap(name):
            def call(*args,**kwargs):
                t0=time.monotonic();before=state();self.observer_seconds+=time.monotonic()-t0
                value=original[name](*args,**kwargs);t0=time.monotonic();after=state()
                row=dict(function=name,shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),
                         RNG_before=before,RNG_after=after)
                if step in self.PROBES:row['values_sha256']=sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                requests.append(row);self.observer_seconds+=time.monotonic()-t0;return value
            return call
        before=state()
        for name in original:setattr(torch,name,wrap(name))
        try:yield
        finally:
            for name,value in original.items():setattr(torch,name,value)
        expected=[('rand',[512,1024]),('rand',[512,1024]),('rand',[512,1,1]),('rand',[512,1]),('normal',[512,1])]
        require([(r['function'],r['shape']) for r in requests]==expected,'K1 random request policy differs')
        require(all(r['device']=='cuda:0' and r['dtype']=='torch.float32' for r in requests),
                'Expected original CUDA float32 draws; unobserved CPU/different-dtype RNG is not admitted')
        t0=time.monotonic();row=dict(step=step,RNG_before=before,RNG_after=state(),requests=requests)
        if self.reference:
            line=self.reference.readline();require(bool(line),'Paired tape reference ended early');same_record(row,json.loads(line))
        self.stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');self.stream.flush();self.steps+=1
        self.observer_seconds+=time.monotonic()-t0

    def close(self):
        require(self.steps==10000,'Incomplete original sampling stream')
        if self.reference:
            require(not self.reference.readline(),'Extra paired tape reference rows');self.reference.close()
        self.stream.close()
