"""Observe the existing analytic illumination route during known-light Bell runs."""
from contextlib import ExitStack, contextmanager
import hashlib

from benchmarks.bell_known_l2_v1 import LIGHT_POLICY, verify_routes
from infrastructure.signatures import require

MODULES = ('outer_light','inner_light','inner_weight','human_light')
PATHS = ('compatibility','paired','training','evaluation','illumination_map')


def light_name(name):
    return name.split('.')[0] in MODULES


class KnownL2Audit:
    def __init__(self):
        self.current = None
        self.routes = {k:dict(dispatches=0,outer_calls=0,inner_calls=0) for k in PATHS}
        self.checkpoints = []

    @contextmanager
    def path(self, name, dispatches=1):
        require(self.current is None and name in self.routes,'Unexpected nested or unknown lighting path')
        self.current = name
        try:
            yield
            self.routes[name]['dispatches'] += dispatches
        finally:
            self.current = None

    @contextmanager
    def attached(self, shader):
        """Check routing and guard against bypasses into learned modules."""
        from functools import wraps

        def forbid(module, args):
            raise ValueError('Learned light called during fixed analytic L2')

        def observe(original, slot):
            @wraps(original)
            def call(*args,**kwargs):
                cfg = shader.cfg
                require(self.current is not None and shader._environment is None
                        and cfg['illumination_mode']==LIGHT_POLICY['direct']
                        and cfg['indirect_mode']==LIGHT_POLICY['indirect']
                        and cfg['known_environment']==LIGHT_POLICY['environment']
                        and cfg['human_lights'] is False,'Known-L2 native routing changed')
                out = original(*args,**kwargs)
                self.routes[self.current][slot] += 1
                return out
            return call

        with ExitStack() as restore:
            for name in MODULES:
                module = getattr(shader,name,None)
                if module is not None:
                    restore.callback(module.register_forward_pre_hook(forbid).remove)
            for key,slot in (('predict_outer_lights','outer_calls'),('get_inner_lights','inner_calls')):
                require(key not in vars(shader),'Native light method already overridden')
                setattr(shader,key,observe(getattr(shader,key),slot))
                restore.callback(delattr,shader,key)
            yield

    @staticmethod
    def lighting_hash(shader):
        h = hashlib.sha256()
        states = {name:value for name,value in shader.state_dict().items() if light_name(name)}
        require(bool(states),'Expected original light modules for initialization parity')
        for name,value in sorted(states.items()):
            a = value.detach().cpu().numpy()
            h.update(name.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def checkpoint(self, step, shader, optimizer, anchor):
        named = list(shader.named_parameters())
        lights = [p for name,p in named if light_name(name)]
        active = {id(p) for _,p in named if p.requires_grad}
        selected = {id(p) for group in optimizer.param_groups for p in group['params']}
        moments = {id(p) for p in optimizer.state}
        require(bool(lights) and all(not p.requires_grad and p.grad is None for p in lights)
                and selected==active and moments<=active,'Known L2 requires material-only optimizer state')
        value = self.lighting_hash(shader)
        require(value==anchor,'Frozen illumination parameters or buffers changed')
        self.checkpoints.append(dict(step=step,state_sha256=value,optimizer_light_parameters=0,
                                     optimizer_light_states=0,learned_light_gradients=0))

    def final(self, planned, steps):
        verify_routes(self.routes,planned)
        require([c['step'] for c in self.checkpoints]==steps,'Light checkpoint cadence differs')
        return dict(routes=self.routes,checkpoints=self.checkpoints,learned_module_calls=0)
