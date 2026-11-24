"""Bounded paired BRDF, proposal density and material-gradient diagnostics."""
from infrastructure.signatures import require

AUDIT = dict(points=24, directions_per_point=16, seed=6806033, output_atol=1e-8,
             output_rtol=1e-6, groups=['mx_predictor','my_predictor','rotation_predictor','kd_predictor','metallic_predictor'])


def tensor_admission(first,second,points,normals,outgoing,output,budget,torch):
    import numpy as np
    from scientific_variants import donut_tensor_d8 as t

    require(len(points)==AUDIT['points'], 'Unexpected final pair audit inventory')
    models = {model.tensor_arm:model for model in (first,second)}
    require(set(models)=={'D9-C','D9-T'},'Final pair representation identities differ')
    n = torch.nn.functional.normalize(normals.double(),dim=-1)
    wo = torch.nn.functional.normalize(outgoing.double(),dim=-1)
    basis = t.frame(n)
    gaussian = torch.randn((len(n),AUDIT['directions_per_point'],3),dtype=torch.float64,device=n.device,
                           generator=torch.Generator(device=n.device).manual_seed(AUDIT['seed']))
    incident = torch.nn.functional.normalize(n[:,None]+.18*gaussian,dim=-1)
    require(bool(((incident*n[:,None]).sum(-1)>0).all()),'Fixed BRDF audit directions left the hemisphere')
    capture, norms, pdf_error = {}, {}, {}

    def equal(a,b):
        return torch.allclose(a,b,atol=AUDIT['output_atol'],rtol=AUDIT['output_rtol'])

    for name,model in sorted(models.items()):
        with torch.enable_grad():
            budget('tensor_admission_predictor_calls',1)
            q,_,kd,metallic = model.predict_tensor(points,normals)
            diffuse,specular = t.brdf(q,basis,n,wo,incident,kd.double(),metallic.double())
            half,reflected,cosine,ndf,pdf = t.sample(q,basis,n,wo,gaussian,density=True)
            direct = t.ndf(q,basis,n,half)
            dot = (half*wo[:,None]).sum(-1,keepdim=True).abs()
            support = dot>0
            identity = torch.where(support,direct*cosine/(4*torch.where(support,dot,1.)),0.)
            require(equal(direct,ndf) and equal(pdf,identity) and bool((pdf>=0).all()),
                    'Direct NDF or reflected proposal density differs')
            params = [getattr(model,key)[6].bias for key in AUDIT['groups']]
            # Physical objective for diagnostics only. autograd.grad leaves all
            # .grad fields untouched, including on the two shared scalar heads.
            target = (diffuse+specular).mean() + half[...,0].mean()
            budget('tensor_admission_reverse_sweeps',1)
            grads = torch.autograd.grad(target,params)
            require(all(bool(torch.isfinite(g).all()) and float(g.norm())>0 and p.grad is None
                        for g,p in zip(grads,params)), 'Nonfinite, zero or applied initial material gradient')
            norms[name] = {key:float(g.norm()) for key,g in zip(AUDIT['groups'],grads)}
        pdf_error[name] = float((pdf-identity).abs().max())
        capture[name] = {key:value.detach() for key,value in dict(diffuse=diffuse,specular=specular,
                                                                 half=half,reflected=reflected,ndf=ndf,pdf=pdf).items()}
    a,b = capture['D9-C'],capture['D9-T']
    require(all(bool(torch.isfinite(a[k]).all() & torch.isfinite(b[k]).all()) and equal(a[k],b[k]) for k in a),
            'Paired fixed-direction BRDF or common Gaussian proposal differs')
    path = output/'tensor-physical-admission.npz'
    with path.open('xb') as file:
        np.savez_compressed(file,**{name+'_'+k:v.cpu().numpy()
                                    for name,fields in capture.items() for k,v in fields.items()})
    return dict(passed=True,points=len(points),directions_per_point=AUDIT['directions_per_point'],
                shared_math_max_abs={key:float((a[key]-b[key]).abs().max()) for key in a},
                direct_pdf_max_abs=pdf_error,gradient_norms=norms)
