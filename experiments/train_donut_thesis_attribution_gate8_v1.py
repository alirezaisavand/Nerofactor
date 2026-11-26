"""One original-A Donut history under the accepted Gate 4 fitting recipe."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_thesis_attribution_gate8_v1 as g
from benchmarks.donut_recovery_v1 import json_new
from experiments import train_donut_matched_geometry_v1 as parent
from infrastructure.signatures import require


# The E10 original/known-light class also replaces these methods. Gate 8
# deliberately dispatches back to the pinned subclass's original implementation.
ORIGINAL_METHODS = ('predict_anisotropic_components','compute_tangent_bitangent_flat',
                    'sample_diffuse_directions','sample_aniso_ggx_directions','compute_Dh',
                    'compute_pdf_aniso_ggx','anisotropic_regularization')


def shader_type(field, renderer, native_classes, torch):
    original=field.MCShadingNetwork
    lighting,_=native_classes(original,renderer.NeROMaterialRenderer,'A_original',field.linear_to_srgb)

    class ThesisShader(lighting):
        def __init__(self,cfg,trace):
            super().__init__(cfg,trace)
            tagged=sha256(bytes(self._a_variant_signature.cpu().tolist())+g.VARIANT.encode()
                          +bytes.fromhex(g.digest(Path(__file__).resolve()))).digest()
            self._a_variant_signature.copy_(torch.as_tensor(list(tagged),dtype=torch.uint8))
            self._captured=None
            self._capturing=False

        def compute_radiance(self,*args,**kwargs):
            values=original.compute_radiance(self,*args,**kwargs)
            require(self._capturing and self._captured is None,'Expected one original radiance call per shade')
            self._captured=values[:3]
            return values

        def shade_anisotropic_mixed(self,*args,**kwargs):
            require(not self._capturing,'Unexpected nested original shade')
            self._capturing=True
            try:
                colors,out=original.shade_anisotropic_mixed(self,*args,**kwargs)
                require(self._captured is not None,'Missing pre-transfer original RGB capture')
                out.update(zip(('linear_rgb','linear_diffuse','linear_specular'),self._captured))
                return colors,out
            finally:
                self._captured=None
                self._capturing=False

    for name in ORIGINAL_METHODS:
        setattr(ThesisShader,name,getattr(original,name))
    return ThesisShader


def initial_identity(shader, cfg, expected, state_hash, torch):
    """Compare against archived K1 with only its behavior signature substituted in a read-only view."""
    from infrastructure.material_variants import CORE, identity

    k1_identity=identity(CORE,dict(network='material',reg_mat=True,shader_cfg=cfg))
    k1_tag=sha256(json.dumps(dict(implementation=k1_identity,shader=shader.cfg),
                            sort_keys=True,allow_nan=False).encode()).digest()
    actual=shader.state_dict()
    projected=dict(actual,_a_variant_signature=torch.as_tensor(list(k1_tag),dtype=torch.uint8))
    require(expected['variant']==CORE and expected['seed']==g.m.POLICY['seed']
            and state_hash(SimpleNamespace(state_dict=lambda:projected))==expected['state_sha256'],
            'Fresh original A parameters or buffers differ from archived K1')
    require(not torch.equal(actual['_a_variant_signature'].cpu(),projected['_a_variant_signature']),
            'Original execution would share the K1 checkpoint identity')
    return dict(state_sha256=state_hash(shader),variant=g.VARIANT,seed=expected['seed'],
                K1_projection_sha256=expected['state_sha256'],shared_state_excludes=['_a_variant_signature'])


def valid_capture(draws):
    require(np.isfinite(draws).all() and (draws>=0).all()
            and np.array_equal(draws[...,0,:]+draws[...,1,:],draws[...,2,:]),
            'Nonfinite, negative or non-additive original linear radiance')


def trainer_source():
    code=parent.trainer_source(False)
    replacements=(
        ("""        shader_class,_=native_classes(field.MCShadingNetwork,renderer.NeROMaterialRenderer,
                                      'A_corrected_core_v1',field.linear_to_srgb)""",
         """        shader_class=acquisition.shader_type(field,renderer,native_classes,torch)"""),
        ("""        initial=dict(state_sha256=state_hash(shader),variant='A_corrected_core_v1',seed=POLICY['seed'])
        require(expected_initial is None or initial==expected_initial,'Full initial weights/buffers differ from the admitted anchor')""",
         """        initial=acquisition.initial_identity(shader,SHADER,expected_initial,state_hash,torch)"""),
        ("""            require(initial==json.loads(args.arm_initial.read_text()),
                    'Fresh arm weights or buffers differ from its accepted historical initialization')""",
         """            require(initial['K1_projection_sha256']==json.loads(args.arm_initial.read_text())['state_sha256'],
                    'Fresh shared K1 state differs')"""),
        ("""        T,B=shader.compute_tangent_bitangent_flat(n,None);axis=tensor(first['axis'][ix])""",
         """        with native_tensor_default(torch):
            T,B=shader.compute_tangent_bitangent_flat(n,None)
        axis=tensor(first['axis'][ix])"""),
        ("""                captures.append(o['linear_rgb'].cpu().numpy())""",
         """                capture=np.stack([o[k].cpu().numpy() for k in ('linear_diffuse','linear_specular','linear_rgb')],axis=1)
                acquisition.valid_capture(capture)
                captures.append(capture[:,2])"""),
        ("""        require(gate['passed'],'Native/reference compatibility or precision gate failed before optimization')""",
         """        report['native_reference_gate_disposition']='Original model discrepancy is measured; K1 agreement is not an admission gate'"""),
        ("""                    write_view(destination/('camera-%d'%camera),arrays,view_identity,np)""",
         """                    acquisition.write_view(destination/('camera-%d'%camera),arrays,view_identity,write_view)"""),
        ("""                        errors,physical=hetero.point_errors(point_pred,{k:d[k][indices] for k in hetero.TARGET_KEYS})""",
         """                        errors,physical=acquisition.physical_errors(point_pred,{k:d[k][indices] for k in hetero.TARGET_KEYS},
                                                                     arrays['physical_valid'][mask])"""),
        ("""            return {}
        run_with_cleanup(work,torch)""",
         """            light_handles.close()
            acquisition.relight(shader,acquisition.unseen,args.output,report,torch,budget)
            return {}
        run_with_cleanup(work,torch)"""),
    )
    for old,new in replacements:
        require(code.count(old)==1,'Unexpected Gate 4 adapter site: '+old.splitlines()[0])
        code=code.replace(old,new,1)
    return code


class Acquisition(parent.Acquisition):
    freeze=staticmethod(g.freeze)
    shader_type=staticmethod(shader_type)
    initial_identity=staticmethod(initial_identity)
    valid_capture=staticmethod(valid_capture)

    def __init__(self,f,paths,args):
        self.frozen=f
        super().__init__(dict(g.gate7.gate5.parent(),arms={g.ARM:f['arm_recipe']}),paths,args)
        self.unseen=g.relight_geometry(paths,g.relight_reference(f,paths,self.views['oracle']))

    def runtime(self,args):
        auth=g.read(args.authorization)
        require(auth==g.authorization(self.frozen,args.commit,args.freeze_sha256)
                and g.digest(args.authorization)==args.authorization_sha256,'Exact Gate 8 approval required')
        recipe=dict(self.recipe,initial_sha256=args.initial_sha256,
                    benchmark_generation_sha256=g.m.GENERATION,protocol=g.PROTOCOL)
        return dict(self.f,arm_training=recipe),auth

    @staticmethod
    def physical_errors(pred,truth,valid):
        errors,summary={},None
        if valid.any():
            errors,summary=g.m.old.point_errors({k:v[valid] for k,v in pred.items()},
                                               {k:v[valid] for k,v in truth.items()})
        return dict(errors,physical_valid=valid),dict(valid=int(valid.sum()),invalid=int((~valid).sum()),
                                                     valid_frame_only=summary)

    @staticmethod
    def write_view(output,arrays,identity,write_view):
        arrays['physical_valid']=g.valid_mask(arrays)
        write_view(output,arrays,identity,np)
        metadata=g.read(output/'view.json')
        metadata.update(orientation='Raw head and actual original sampling axes; consult physical_valid mask',
                        world_Q='Raw frame-derived diagnostic. Physical Q only where physical_valid is true')
        # Preserve RGB and raw widths everywhere and mask invalid Q/orientation display pixels.
        from PIL import Image
        displays=('orientation.png','orientation_world_projector.png')
        for name in (*displays,*(f'Q_world_{i}{j}.png' for i in range(3) for j in range(i,3))):
            path=output/name
            with Image.open(path) as image:
                preview=np.array(image)
            preview[~arrays['physical_valid']]=0
            Image.fromarray(preview).save(path)
            metadata['files'][name]=dict(bytes=path.stat().st_size,sha256=g.digest(path))
        (output/'view.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')

    @staticmethod
    def relight(shader,views,output,report,torch,budget):
        from benchmarks.bell_reference_e11 import environment
        from experiments.bell_neural_material_e16 import state_hash
        from infrastructure.known_l2_audit import MODULES
        from infrastructure.material_recovery_e13_evaluation import native_tensor_default
        from infrastructure.sphere_replay import evaluation_scope

        before=state_hash(shader)
        work=dict(shader_calls=0,shadow_rays=0)
        routed=dict(calls=0,directions=0)
        unseen=report['unseen']=dict(work=work,unseen_environment=routed,state_unchanged=False)

        def light(_,directions):
            routed['calls']+=1
            routed['directions']+=len(directions)
            return environment(directions,g.g6.LIGHT)

        def forbidden(module,args):
            raise ValueError('Learned light called during Gate 8 relighting')

        with evaluation_scope(shader,torch,np) as scoped, torch.no_grad(), native_tensor_default(torch), ExitStack() as guards:
            shader.eval()
            guards.enter_context(shader.environment(light,g.PROTOCOL))
            for name in MODULES:
                module=getattr(shader,name,None)
                if module is not None:
                    guards.callback(module.register_forward_pre_hook(forbidden).remove)
            for camera in g.g6.CAMERAS:
                view=views[camera]
                count=len(view['pixel_indices'])
                draws=np.empty((g.g6.DRAWS,count,3,3),np.float32)
                for start in range(0,count,g.g6.CHUNK):
                    rows=slice(start,start+g.g6.CHUNK)
                    p,n,wo=(torch.as_tensor(view[k][rows],device='cuda',dtype=torch.float32)
                            for k in ('position','normals','outgoing'))
                    for draw in range(g.g6.DRAWS):
                        budget()
                        require(work['shader_calls']<g.RELIGHT['shader_calls']
                                and work['shadow_rays']+len(p)*1536<=g.RELIGHT['shadow_rays'],
                                'Relighting work cap reached')
                        work['shader_calls']+=1
                        work['shadow_rays']+=len(p)*1536
                        seed,_=g.g6.tape(camera,start,draw)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        _,out=shader(p,wo,n,None,g.g6.STEP,False)
                        draws[draw,rows]=np.stack([out[k].cpu().numpy()
                                                  for k in ('linear_diffuse','linear_specular','linear_rgb')],axis=1)
                    valid_capture(draws[:,rows])
                np.savez_compressed(output/f'unseen-camera-{camera}.npz',pixels=view['pixel_indices'],draws=draws)
                budget()
        require(work==g.RELIGHT and routed['calls']>0 and 0<routed['directions']<=work['shadow_rays']
                and shader._environment is None and scoped['RNG_and_modes_restored']
                and state_hash(shader)==before,'Invalid endpoint relighting, work accounting or restoration')
        unseen.update(state_unchanged=True,state_sha256=before,learned_module_calls=0)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for flag in ('freeze-sha256','commit','authorization-sha256'):
        cli.add_argument('--'+flag,required=True)
    cli.add_argument('--execute',action='store_true',required=True)
    args=cli.parse_args()
    args.freeze=g.ROOT/g.FREEZE
    f=g.freeze(args.freeze,args.freeze_sha256)
    p=g.paths(f,native=True)
    args.phase=g.ARM
    args.output=Path('/outputs')/g.ARM
    args.authorization=Path('/authorization.json')
    require(os.geteuid()!=0 and args.output.resolve()==args.output and not args.output.exists()
            and not os.statvfs('/outputs').f_flag&os.ST_RDONLY
            and all(os.statvfs(path).f_flag&os.ST_RDONLY for path in (Path('/'),g.ROOT,args.authorization,*p.values()))
            and {ln.split(':')[0].strip() for ln in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},
            'Fresh isolated non-root Gate 8 runtime required')
    args.image_id=f['image']
    args.initial,args.oracle,args.benchmark=(p[k] for k in ('initial','oracle','reference'))
    args.admission,args.verification=p['admission'],p['verification']
    args.initial_sha256=f['initial']['sha256']
    args.benchmark_sha256=f['generation_sha256']
    args.arm_initial=p['initial']
    code=trainer_source()
    require(sha256(code.encode()).hexdigest()==f['arm_recipe']['trainer_sha256'],'Gate 8 trainer overlay differs')
    acquisition=Acquisition(f,p,args)
    namespace=dict(__name__='donut_gate8_native',__file__=str(g.ROOT/parent.native.SOURCE))
    exec(compile(code,'donut-thesis-attribution-gate8-v1','exec'),namespace)
    namespace['train'](args,acquisition=acquisition)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,parent.interrupted)
    main()
