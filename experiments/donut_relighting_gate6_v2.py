"""Replay accepted material-query batches before the unchanged Gate 6 shaders."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_relighting_gate6_v2 as g
from infrastructure.signatures import require


V1 = 'experiments/donut_relighting_gate6_v1.py'


def material_query(model, arm, points, normals):
    """Return the actual render parameters and their accepted world-Q export."""
    if g.TENSOR_ARMS[arm] is not None:
        from scientific_variants.donut_tensor_d8 import world_tensor

        q,frame,kd,metallic = model.predict_tensor(points,normals)
        tensors = dict(Q_world=world_tensor(q,frame),color=kd,metallic=metallic,local_Q=q,frame=frame)
        return {key:value.cpu().numpy() for key,value in tensors.items()}

    from scientific_variants.material_a_v1 import orientation

    mx,my,metallic,kd,rotation,raw = model.predict_anisotropic_components(points)
    tangent,bitangent = model.compute_tangent_bitangent_flat(normals,None)
    cosine,sine = orientation(rotation)
    tangent,bitangent = model.rotate_tangent_bitangent(tangent,bitangent,cosine,sine)
    vectors = dict(mx=mx,my=my,color=kd,metallic=metallic,rotation=rotation,rotation_raw=raw,
                   tangent=(tangent+1)/2,bitangent=(bitangent+1)/2)
    values = {key:value.cpu().numpy() for key,value in vectors.items()}
    values['Q_world'] = g.archived_q(values['mx'],values['my'],values.pop('tangent'),values.pop('bitangent'))
    return values


def native_pass(model, arm, views, archives, torch, charge, save_capture, admission):
    from benchmarks.bell_reference_e11 import environment
    from experiments.bell_neural_material_e16 import state_hash
    from infrastructure.material_recovery_e13_evaluation import native_tensor_default
    from infrastructure.sphere_replay import evaluation_scope

    material_keys = ('Q_world','color','metallic')
    materials,params = {},{}
    parity = admission[arm] = {}
    before = state_hash(model)
    model.requires_grad_(False)
    routed = dict(calls=0,directions=0)

    def relight(_,directions):
        routed['calls']+=1
        routed['directions']+=len(directions)
        return environment(directions,g.LIGHT)

    def tensor(values):
        return torch.as_tensor(values,device='cuda')

    with evaluation_scope(model,torch,np) as scoped, torch.no_grad(), native_tensor_default(torch):
        model.eval()
        for camera,view in views.items():
            def predict(rows):
                charge('material_field_calls',1)
                charge('material_field_points',len(rows))
                return material_query(model,arm,tensor(view['position'][rows].astype('f4')),
                                       tensor(view['normals'][rows].astype('f4')))

            fields = g.reconstruct_material(view,predict)
            row = parity[str(camera)] = g.field_parity(arm,view,fields,archives[arm,camera])
            require(row['passed'],f'Restored physical field differs from accepted endpoint for {arm} camera {camera}: '
                    f'{row["field_max_abs"]}, pixels {row["maximum_error_pixels"]}')
            widths,axis,error = g.physical_axes(fields['Q_world'],view['normals'])
            require(all(((fields[k]>=0)&(fields[k]<=1)).all() for k in ('color','metallic')),
                    'Invalid restored linear material coefficients')
            row['Q_axis_reconstruction_max_abs']=error
            materials[camera] = {key:fields[key].astype(np.float64) for key in material_keys}
            materials[camera].update(widths_major_minor=widths,axis=axis)
            params[camera]=fields

        # The scientific sampling still uses the original V1 expanded raster
        # chunking and seeds. Only neural material query batching has changed.
        with model.environment(relight,g.PROTOCOL):
            for camera,view in views.items():
                count = len(view['pixel_indices'])
                draws = np.empty((g.DRAWS,count,3,3),np.float32)
                for begin in range(0,count,g.CHUNK):
                    sl = slice(begin,min(begin+g.CHUNK,count))
                    p,n,wo = (tensor(view[k][sl].astype('f4')) for k in ('position','normals','outgoing'))
                    fields = params[camera]
                    is_tensor = g.TENSOR_ARMS[arm] is not None
                    keys = ('local_Q','frame','color','metallic') if is_tensor else (
                        'mx','my','metallic','color','rotation','rotation_raw')
                    values = [tensor(fields[key][sl]) for key in keys]
                    for draw in range(g.DRAWS):
                        seed,dedicated = g.tape(camera,begin,draw)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        charge('native_shader_calls',1)
                        if is_tensor:
                            result = model.shade_tensor(p,n,wo,*values,dedicated,False)
                        else:
                            _,result = model.shade_anisotropic_mixed(p,n,wo,*values,None,False)
                        draws[draw,sl] = np.stack([result[k].cpu().numpy()
                                                   for k in ('linear_diffuse','linear_specular','linear_rgb')],1)
                require(np.isfinite(draws).all() and (draws>=0).all()
                        and np.array_equal(draws[:,:,0]+draws[:,:,1],draws[:,:,2]), 'Invalid native draw or component sum')
                save_capture(f'{arm}-camera-{camera}-native.npz',pixels=view['pixel_indices'],draws=draws,**materials[camera])
    require(scoped['RNG_and_modes_restored'] and model._environment is None and state_hash(model)==before,
            'Relighting mutated model state, environment, modes or RNG')
    require(routed['calls']>0 and routed['directions']>0,'No native rays reached the unseen environment')
    return materials,dict(field_parity=parity,state_sha256=before,state_unchanged=True,unseen_environment=routed)


def runner_source():
    source=g.replay_source(V1,(
        ('from benchmarks import donut_relighting_gate6_v1 as g','from benchmarks import donut_relighting_gate6_v2 as g'),
        ('image=f[\'image\'], counters=dict.fromkeys(g.LIMITS,0), outputs={}, restoration={}, views={})',
         'image=f[\'image\'], counters=dict.fromkeys(g.LIMITS,0), outputs={}, restoration={}, views={}, field_admission={})'),
        ('native_pass(shader,arm,views,archives,torch,charge,save_capture)',
         'native_pass(shader,arm,views,archives,torch,charge,save_capture,r[\'field_admission\'])'),
    ))
    start, end = 'def native_pass(', 'def integrate('
    require(source.count(start)==source.count(end)==1 and source.index(start)<source.index(end),
            'Expected ordered V1 native adapter and independent integrator')
    return source[:source.index(start)]+source[source.index(end):]


if __name__=='__main__':
    # Keep the V1 checkpoint loader, independent integrator, limits plumbing and
    # evidence lifecycle. The new function replaces only material replay/native dispatch.
    exec(compile(runner_source(),str(g.ROOT/V1),'exec'),
         dict(__name__='__main__',__file__=__file__,native_pass=native_pass))
