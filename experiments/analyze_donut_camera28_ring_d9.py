"""Localize D8 camera-28 errors from saved arrays, with no renderer calls."""

import argparse
from pathlib import Path

import numpy as np

from analyze_donut_paired_tensor_d8 import ARMS, ROOT, replay

CAMERA, STEP = 28, 10000
LUMA = np.array([.2126,.7152,.0722])


def sample_environment(array, directions):
    """Bilinear lookup on the saved equirectangular grid, periodic in azimuth."""
    h,w,_=array.shape
    x=np.arctan2(directions[:,1],directions[:,0])*w/(2*np.pi)-.5
    y=np.clip(np.arccos(np.clip(directions[:,2],-1,1))*h/np.pi-.5,0,h-1)
    i,j=np.floor(x).astype(int),np.floor(y).astype(int)
    u,v=(x-i)[:,None],(y-j)[:,None]
    upper=(1-u)*array[j,i%w]+u*array[j,(i+1)%w]
    lower=(1-u)*array[np.minimum(j+1,h-1),i%w]+u*array[np.minimum(j+1,h-1),(i+1)%w]
    return (1-v)*upper+v*lower


def correlations(y, fields):
    return {name:float(np.corrcoef(y,v)[0,1]) if np.std(v)>0 and np.std(y)>0 else None
            for name,v in fields.items()}


def analyze(out):
    originals=(replay.REFERENCE,*(p.parent for p in ARMS.values()))
    assert not out.exists() and not any(p in out.resolve().parents for p in originals)
    package=ROOT/'docs/results/donut_paired_tensor_d8_v1'
    manifest_path=package/'artifact-manifest.json'
    sealed=replay.read(package/'finalization-validation.json')['artifact_sha256']
    assert replay.digest(manifest_path)==sealed[str(manifest_path.relative_to(ROOT))]
    manifest=replay.read(manifest_path)['records']
    generation_path=replay.REFERENCE/'generation.json'
    assert replay.digest(generation_path)==replay.GENERATION
    reference_path=replay.REFERENCE/f'view-{CAMERA}.npz'
    replay.verify_files(replay.REFERENCE,{reference_path.name:
        replay.read(generation_path)['outputs'][reference_path.name]})
    ref=replay.numeric(reference_path)
    use=ref['shader_valid']
    pixels=ref['pixel_indices'][use]
    y,x=np.divmod(pixels,128)
    center=np.array([x.mean(),y.mean()])
    radius=np.linalg.norm(np.column_stack((x,y))-center,axis=1)
    cuts=np.quantile(radius,[.25,.5,.75])
    quartiles=np.digitize(radius,cuts)
    components=ref['linear_components'][use].astype(float)
    np.testing.assert_allclose(components[:,0]+components[:,1],components[:,2],rtol=0,atol=1e-12)
    n,wo=ref['normals'][use],ref['outgoing'][use]
    mirror=2*(n*wo).sum(-1,keepdims=True)*n-wo
    mirror/=np.linalg.norm(mirror,axis=-1,keepdims=True)
    views={}
    all_paths=[manifest_path,generation_path,reference_path]
    known_light=None
    for arm,root in ARMS.items():
        receipt=root/'recovery.json'
        assert replay.digest(receipt)==manifest[str(receipt)]['sha256']
        report=replay.read(receipt)
        step=f'step-{STEP:05d}'
        names=[f'{step}/camera-{CAMERA}/materials.npz',f'{step}/camera-{CAMERA}-errors.npz',f'{step}/illumination.npz']
        replay.verify_files(root,{name:report['outputs'][name] for name in names})
        m,e,l=(replay.numeric(root/name) for name in names)
        if known_light is None:
            known_light=l['known_linear_RGB']
        else:
            np.testing.assert_array_equal(l['known_linear_RGB'],known_light)
        mask=m['hit_mask']
        np.testing.assert_array_equal(np.flatnonzero(mask),pixels)
        np.testing.assert_array_equal(m['linear_gt'][mask],components[:,2].astype(np.float32))
        observed=np.stack([m[k][mask] for k in ('linear_diffuse','linear_specular','linear_rgb')],axis=1).astype(float)
        np.testing.assert_allclose(observed[:,0]+observed[:,1],observed[:,2],rtol=0,atol=2e-7)
        delta=observed-components
        mu,sigma=sample_environment(l['linear_RGB'],mirror),sample_environment(l['known_linear_RGB'],mirror)
        fields=dict(diffuse_error_luma=delta[:,0]@LUMA,specular_error_luma=delta[:,1]@LUMA,
                    RGB_error_luma=delta[:,2]@LUMA,major_width=m['major_width'][mask,0],
                    minor_width=m['minor_width'][mask,0],gap=m['anisotropy_gap'][mask,0],
                    relative_Q=e['Q_relative'],axial_orientation_degrees=e['orientation_degrees'],
                    mirror_light_error_luma=(mu-sigma)@LUMA)
        views[arm]=dict(delta=delta,observed=observed,fields=fields,environment=l,
                        eligible=e['orientation_eligible'],
                        illumination_relative_L2=report['saved_endpoints'][-1]['metrics']['illumination']['relative_L2'])
        all_paths.extend((receipt,*(root/name for name in names)))
    # A post-hoc region for localization only, applied unchanged to every arm.
    excess=views['D8-T']['fields']['RGB_error_luma']
    brightest=np.argsort(-excess,kind='stable')[:int(np.ceil(len(excess)/10))]
    region=np.zeros(len(excess),bool)
    region[brightest]=True
    assert (excess[region]>0).all()
    subsets={'foreground':np.ones_like(region),'D8_T_positive_excess_decile':region,
             'elsewhere':~region,**{f'radius_Q{i+1}':quartiles==i for i in range(4)}}
    result={}
    for arm,d in views.items():
        fields=d['fields']
        panels={}
        for label,mask in subsets.items():
            delta=d['delta'][mask]
            eligible=mask&d['eligible']
            diffuse_sse=float((delta[:,0]**2).sum())
            specular_sse=float((delta[:,1]**2).sum())
            cross=float((2*delta[:,0]*delta[:,1]).sum())
            total=float((delta[:,2]**2).sum())
            np.testing.assert_allclose(diffuse_sse+specular_sse+cross,total,rtol=1e-5,atol=1e-7)
            means={k:float(v[mask].mean()) for k,v in fields.items() if k!='axial_orientation_degrees'}
            panels[label]=dict(observations=int(mask.sum()),means=means,
                eligible_orientation_count=int(eligible.sum()),
                eligible_orientation_median_degrees=float(np.median(fields['axial_orientation_degrees'][eligible]))
                if eligible.any() else None,
                diffuse_SSE=diffuse_sse,specular_SSE=specular_sse,diffuse_specular_cross=cross,total_RGB_SSE=total)
        corr={k:fields[k] for k in ('diffuse_error_luma','specular_error_luma','major_width','minor_width',
                                   'gap','relative_Q','mirror_light_error_luma')}
        corr['eligible_orientation_degrees']=fields['axial_orientation_degrees'][d['eligible']]
        orient=correlations(fields['RGB_error_luma'][d['eligible']],
                            {'eligible_orientation_degrees':corr.pop('eligible_orientation_degrees')})
        result[arm]=dict(subsets=panels,RGB_excess_correlations=dict(correlations(fields['RGB_error_luma'],corr),**orient),
                         illumination_relative_L2=d['illumination_relative_L2'])
    out.mkdir(parents=True)
    replay.write_json(out/'ring-analysis.json',dict(camera=CAMERA,updates=STEP,
        foreground_pixels=len(pixels),oracle_widths=[.35,.08],center_xy=center.tolist(),radius_quartiles=cuts.tolist(),
        D8_T_positive_excess_decile=dict(pixels=pixels[region].tolist(),radius_median=float(np.median(radius[region])),
            count_by_radius_quartile=np.bincount(quartiles[region],minlength=4).tolist()),
        oracle_in_decile=dict(component_mean_luma=(components[region]@LUMA).mean(0).tolist(),
                             mirror_mean_luma=float((sigma[region]@LUMA).mean())),
        comparisons=result,artifacts={str(p):replay.record(p) for p in all_paths},
        analysis_work=dict(renderer_calls=0,native_models=0,optimizer_returns=0,checkpoint_deserializations=0)))
    figures(out,pixels,region,components,views)


def figures(out,pixels,region,oracle,views):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def grid(ax,values,title,vmin=None,vmax=None,cmap='magma'):
        shape=(128*128,*values.shape[1:])
        im=np.full(shape,np.nan)
        im[pixels]=values
        ax.set(title=title,xticks=[],yticks=[])
        return ax.imshow(im.reshape(128,128,*values.shape[1:]),vmin=vmin,vmax=vmax,cmap=cmap)

    names=('RGB_error_luma','specular_error_luma','diffuse_error_luma','mirror_light_error_luma')
    limit=max(float(np.quantile(abs(d['fields'][k]),.99)) for d in views.values() for k in names[:3])
    illum_limit=max(float(np.quantile(abs(d['fields'][names[-1]]),.99)) for d in views.values())
    fig,axes=plt.subplots(4,5,figsize=(12.5,9),layout='constrained')
    rgb=replay.srgb(oracle[:,2])
    grid(axes[0,0],rgb,'Oracle RGB')
    grid(axes[0,1],oracle[:,1]@LUMA,'Oracle specular Y',0,limit)
    grid(axes[0,2],oracle[:,0]@LUMA,'Oracle diffuse Y',0,limit)
    grid(axes[0,3],region.astype(float),'D8-T excess decile',0,1,cmap='Greys')
    axes[0,4].axis('off')
    for i,(arm,d) in enumerate(views.items(),1):
        grid(axes[i,0],replay.srgb(d['observed'][:,2]),arm+' RGB')
        for j,(k,label) in enumerate(zip(names,('RGB ΔY','Specular ΔY','Diffuse ΔY','Mirror env. ΔY')),1):
            bound=limit if j<4 else illum_limit
            img=grid(axes[i,j],d['fields'][k],label,-bound,bound,cmap='coolwarm')
            if i==len(views) and j in (3,4):
                fig.colorbar(img,ax=axes[1:4,1:4] if j==3 else axes[1:4,4],orientation='horizontal')
    fig.savefig(out/'camera28-radiance.png',dpi=140)
    plt.close(fig)

    fig,axes=plt.subplots(3,4,figsize=(11,7.5),layout='constrained')
    keys=('major_width','minor_width','relative_Q','axial_orientation_degrees')
    for i,(arm,d) in enumerate(views.items()):
        for j,(k,limits,label) in enumerate(zip(keys,((.1,.5),(.04,.16),(0,1),(0,45)),
                                                ('Major width','Minor width','Relative Q error','Eligible axial error (°)'))):
            value=d['fields'][k]
            if j==3:
                value=np.where(d['eligible'],value,np.nan)
            img=grid(axes[i,j],value,f'{arm} · {label}',*limits)
            if i==len(views)-1:
                fig.colorbar(img,ax=axes[:,j],orientation='horizontal')
    fig.savefig(out/'camera28-material.png',dpi=140)
    plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    analyze(cli.parse_args().output)
