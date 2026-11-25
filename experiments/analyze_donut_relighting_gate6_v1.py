"""File-only replay of Gate 6 convergence, metrics and publication figures."""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import rankdata

from benchmarks import donut_relighting_gate6_v1 as g
from benchmarks.bell_e11 import convergence
from benchmarks.donut_recovery_v1 import json_new, numeric
from benchmarks.generate_donut_recovery_v1 import highlight_record
from experiments.analyze_donut_representation_gate5_v1 import cohorts_and_views
from infrastructure.signatures import require

COMPONENTS = ('diffuse','specular','RGB')


def score(pred, truth):
    require(pred.shape==truth.shape and pred.ndim==2 and pred.shape[1]==3
            and np.isfinite(pred).all() and np.isfinite(truth).all(), 'Invalid linear metric arrays')
    if not len(truth):
        return None
    norm = np.linalg.norm(truth)
    delta = pred-truth
    rmse = float(np.sqrt(np.mean(delta**2)))
    return dict(pixels=len(truth),rmse=rmse,relative_L2=float(np.linalg.norm(delta)/norm) if norm else None,
                PSNR_unit_peak_dB=float(-20*np.log10(rmse)) if rmse else None,
                exact=rmse==0.)


def scores(pred, truth, mask):
    return {name:score(pred[mask,i],truth[mask,i]) for i,name in enumerate(COMPONENTS)}


def correlation(a,b):
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:
        return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def raster(data, pixels):
    result=np.zeros((g.SIZE**2,*data.shape[1:]),dtype=data.dtype)
    result[pixels]=data
    return result


def scopes(view, new_highlight, original):
    pixels = view['pixel_indices']
    matched = np.isin(pixels,view['matched_pixels'])
    values = dict(common=matched, expanded_foreground=np.ones(len(pixels),bool), extra_foreground=~matched)
    fixed = dict(region_0=view['material_region']==0,region_1=view['material_region']==1,
                 near_isotropic=view['nearly_isotropic'], boundary=view['near_boundary'],
                 grazing=(view['normals']*view['outgoing']).sum(1)<g.gate5.m.rgb.geo.TOL['grazing_cosine'],
                 fitting_light_highlight=view['highlight_mask'], relight_highlight=new_highlight,
                 relight_non_highlight=~new_highlight)
    values.update({key:matched&mask for key,mask in fixed.items()})
    for key in ('other_fitting_view','no_other_fitting_view','nearby_highlight','covered_without_highlight'):
        a=np.zeros_like(matched)
        a[matched]=original[key]
        values[key]=a
    result = {key:raster(v,pixels) for key,v in values.items()}
    result['full_raster']=np.ones(g.SIZE**2,bool)
    return result


def analyze(sha, out):
    f=g.freeze(sha)
    batch=Path(f['host_output'])
    run=batch/'run'
    auth=g.read(batch/'authorization.json')
    auth_sha=g.digest(batch/'authorization.json')
    r=g.read(run/'receipt.json')
    require((batch/'exit-code.txt').read_text().strip()=='0'
            and auth==g.authorization(f,auth['commit'],sha) and g.digest(batch/'freeze.json')==sha
            and auth_sha==g.digest(batch.with_name(batch.name+'-authorization.json'))
            and r['protocol']==g.PROTOCOL and r['status']=='complete' and r['cleanup_verified']
            and r['commit']==auth['commit'] and r['authorization_sha256']==auth_sha and r['freeze_sha256']==sha
            and r['image']==f['image'] and r['backend']==f['backend']
            and r['elapsed_seconds']<g.RESOURCES['inner_seconds']
            and r['host_peak_rss_bytes']<g.RESOURCES['host_memory_gib']*1024**3
            and sum(v['bytes'] for v in r['outputs'].values())<=g.RESOURCES['output_bytes'],
            'Incomplete or foreign Gate 6 result')
    require(set(r['outputs'])=={p.name for p in run.glob('*.npz')},'Capture inventory differs')
    require(out.resolve()==out and not out.exists() and out.parent.is_dir() and batch not in out.parents,
            'Fresh file-only analysis path required')
    previous,views,archives=g.admit(f,g.base_paths(f),g.endpoint_paths(f))
    parent=g.gate5.parent()
    pp=g.gate5.m.paths(parent)
    _,gate4,_,coverage,_=g.gate5.m.admit(parent,pp)
    original=cohorts_and_views(gate4['oracle'],pp['reference'],coverage)
    images, used, rows={},set(),{}
    panels={}
    total=0

    def capture(name):
        require(name not in used,'Duplicate relighting capture '+name)
        a=numeric(g.check(run/name,r['outputs'][name]))
        used.add(name)
        return a

    for camera in g.CAMERAS:
        v,pixels=views[camera],views[camera]['pixel_indices']
        np.testing.assert_array_equal(gate4['oracle'][camera]['pixel_indices'],v['matched_pixels'])
        matched=np.isin(pixels,v['matched_pixels'])

        def fetch(ix, grid, prefix):
            return capture(f'camera-{camera}-{prefix}-{grid}.npz')

        grids,vis,resolution,done=g.refine(pixels,fetch,total)
        total+=len(done)
        gates={arm:convergence(grids[1,i,:,2],grids[0,i,:,2]) for i,arm in enumerate(g.ROLES)}
        require(r['views'][str(camera)]==dict(refined_pixels=done,convergence=gates),'Grid refinement or convergence differs')
        ref=grids[1,0]
        highlights, is_highlight=highlight_record(ref)
        masks=scopes(v,is_highlight,original[camera])
        reference=raster(ref,pixels)
        coarse=raster(grids[0,0],pixels)
        panels[camera]=dict(pixels=pixels,oracle_hit=v['hit_mask'].ravel(),reference=reference,
            reference_coarse=coarse,resolution=resolution,visible_counts=vis,
            **{f'mask_{k}':a for k,a in masks.items()})
        support=dict(total_pixels=g.SIZE**2,primary_common=int(matched.sum()),oracle_shader_valid=len(pixels),
                     extra_valid=int((~matched).sum()), geometrical_hits=int(v['hit_mask'].sum()),
                     below_horizon=int(v['hit_mask'].sum())-len(pixels),
                     known_black_background=int((~v['hit_mask']).sum()),undefined=0,
                     masks={k:int(a.sum()) for k,a in masks.items()})
        report=dict(support=support,refined_pixels=done,convergence=gates,oracle_highlights=highlights,
                    independent={},native={},uncertainty={},physical_associations={})
        images[camera]=dict(masks=masks,reference=reference,coarse=coarse,variants={})
        for i,arm in enumerate(g.ARMS,1):
            d=capture(f'{arm}-camera-{camera}-native.npz')
            require(set(d)=={'pixels','draws','Q_world','color','metallic','widths_major_minor','axis'}
                    and np.array_equal(d['pixels'],pixels) and d['draws'].shape==(g.DRAWS,len(pixels),3,3)
                    and d['draws'].dtype==np.float32 and np.isfinite(d['draws']).all()
                    and (d['draws']>=0).all() and np.array_equal(d['draws'][:,:,0]+d['draws'][:,:,1],d['draws'][:,:,2]),
                    'Invalid saved native draw panel')
            widths,_,axis_error=g.physical_axes(d['Q_world'],v['normals'])
            np.testing.assert_allclose(widths,d['widths_major_minor'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(d['axis'],axis=1),1,rtol=0,atol=g.FRAME_ATOL)
            np.testing.assert_allclose((d['axis']*v['normals']).sum(1),0,rtol=0,atol=g.FRAME_ATOL)
            np.testing.assert_allclose(g.tensor_from_axes(widths,d['axis'],v['normals']),d['Q_world'],
                                       rtol=0,atol=g.MATERIAL_ATOL)
            restore=r['restoration'][arm]
            old=archives[arm,camera]
            qkey='Q_world' if g.TENSOR_ARMS[arm] is None else 'Q_tensor_direct'
            shifts={k:float(abs(d[k][matched]-old[key][old['hit_mask']]).max()) for k,key in
                    (('Q_world',qkey),('color','kd'),('metallic','metallic'))}
            require(restore['strict'] and restore['state_unchanged'] and not restore['optimizer_restored']
                    and restore['checkpoint_sha256']==auth['checkpoints'][arm]
                    and restore['unseen_environment']['calls']>0 and restore['unseen_environment']['directions']>0
                    and restore['field_parity'][str(camera)]['common_rows']==matched.sum()
                    and restore['field_parity'][str(camera)]['field_max_abs']==shifts
                    and max(shifts.values())<=g.MATERIAL_ATOL
                    and axis_error<=g.MATERIAL_ATOL,'Restored material parity receipt differs')
            draws=d['draws'].astype(np.float64)
            mean=draws.mean(0)
            se=draws.std(0,ddof=1)/np.sqrt(g.DRAWS)
            half=draws[:g.DRAWS//2].mean(0)-draws[g.DRAWS//2:].mean(0)
            independent=raster(grids[1,i],pixels)
            lower=raster(grids[0,i],pixels)
            native=raster(mean,pixels)
            panels[camera].update({f'{arm}_{name}':value for name,value in
                (('independent',independent),('independent_coarse',lower),('native',native),
                 ('native_SE',raster(se,pixels)))})
            images[camera]['variants'][arm]=dict(independent=independent,coarse=lower,native=native)
            report['independent'][arm]={k:scores(independent,reference,a) for k,a in masks.items()}
            report['native'][arm]={k:scores(native,reference,a) for k,a in masks.items()}
            sel=masks['common']
            pair=independent-reference
            lower_pair=lower-coarse
            report['uncertainty'][arm]=dict(
                prediction_grid_difference=scores(independent,lower,sel),
                reference_grid_difference=scores(reference,coarse,sel),
                paired_grid_residual_change=scores(pair,lower_pair,sel),
                coarse_endpoint_error=scores(lower,coarse,sel),
                native_minus_independent=scores(native,independent,sel),
                native_standard_error_RMS=np.sqrt(np.mean(se[matched]**2,axis=(0,2))).tolist(),
                native_split_half_difference_RMS=np.sqrt(np.mean(half[matched]**2,axis=(0,2))).tolist())
            path=Path(f['methods'][arm]['folder'])/f'step-{g.STEP:05d}/camera-{camera}-errors.npz'
            prior=numeric(g.check(path,previous[arm]['outputs'][f'step-{g.STEP:05d}/{path.name}']))
            error=np.sqrt(np.mean((grids[1,i][matched]-ref[matched])**2,axis=2))
            require(len(error)==len(prior['Q_relative']),'Prior physical errors do not match')
            report['physical_associations'][arm]={name:{comp:correlation(prior[name],error[:,j])
                for j,comp in enumerate(COMPONENTS)} for name in ('Q_relative','major_width','minor_width','color_max','metallic')}
            eligible=prior['orientation_eligible']
            report['physical_associations'][arm]['orientation_eligible_rows']=int(eligible.sum())
            report['physical_associations'][arm]['orientation_degrees']={comp:correlation(
                prior['orientation_degrees'][eligible],error[eligible,j]) for j,comp in enumerate(COMPONENTS)}
        rows[str(camera)]=report

    valid=g.DIRECTIONS+total*sum(n*n for n in g.HIGH)
    require(used==set(r['outputs']) and total==r['refined_pixels_total']
            and r['counters']==dict(g.LIMITS,independent_shadow_directions=valid,
                 independent_BRDF_point_directions=len(g.ROLES)*valid),'Observed work or capture coverage differs')
    first=images[g.CAMERAS[0]]
    full_ref=np.concatenate([v['reference'] for v in images.values()])
    full_coarse=np.concatenate([v['coarse'] for v in images.values()])
    common=np.concatenate([v['masks']['common'] for v in images.values()])
    pooled={}
    for arm in g.ARMS:
        pooled[arm]={}
        for route in ('independent','native'):
            pred=np.concatenate([v['variants'][arm][route] for v in images.values()])
            pooled[arm][route]={group:scores(pred,full_ref,np.concatenate([v['masks'][group] for v in images.values()]))
                               for group in first['masks']}
        lower=np.concatenate([v['variants'][arm]['coarse'] for v in images.values()])
        fine=np.concatenate([v['variants'][arm]['independent'] for v in images.values()])
        pooled[arm]['paired_grid_common']=dict(coarse_endpoint=scores(lower,full_coarse,common),
            residual_change=scores(fine-full_ref,lower-full_coarse,common))
    out.mkdir()
    for camera,panel in panels.items():
        with (out/f'camera-{camera}-relighting.npz').open('xb') as file:
            np.savez_compressed(file,**panel)
    display=plot(images,out)
    json_new(out/'relighting-results.json',dict(protocol=g.PROTOCOL,status='verified_complete',
        execution_commit=auth['commit'],freeze_sha256=sha,authorization_sha256=auth_sha,receipt_sha256=g.digest(run/'receipt.json'),
        views=rows,pooled=pooled,work=r['counters'],runtime_seconds=r['elapsed_seconds'],
        display=display,component_order=list(COMPONENTS),optimizer_updates=0,
        PSNR='20 log10(1/RMSE), fixed unit linear radiance; exact matches null with exact=true',
        association='Spearman by camera on frozen common pixels; orientation uses its historical per-arm eligibility',
        uncertainty='Grid differences are numerical diagnostics. Native diffuse strata are fixed and their bias is not captured by draw SE.'))


def plot(images,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmarks.bell_e11 import linear_to_srgb

    errors=[np.sqrt(np.mean((v['variants'][arm][route][:,2]-v['reference'][:,2])**2,axis=1))
            for v in images.values() for arm in g.ARMS for route in ('independent','native')]
    error_max=max(float(np.max(a)) for a in errors)
    limit=max(error_max,1e-12)
    clipped={}
    titles=('Oracle','K1','Final-Chart','Final-Ambient')
    for route in ('independent','native'):
        fig,axes=plt.subplots(4,4,figsize=(10,9.4),layout='constrained')
        for i,(camera,v) in enumerate(images.items()):
            truth=v['reference'][:,2]
            kept=[]
            rgb=[truth,*[v['variants'][arm][route][:,2] for arm in g.ARMS]]
            for j,(title,image) in enumerate(zip(titles,rgb)):
                clipped[f'{route}/{camera}/{title}']=int((image>1).sum())
                ax=axes[2*i,j]
                ax.imshow(linear_to_srgb(np.minimum(image.reshape(g.SIZE,g.SIZE,3),1)),interpolation='nearest')
                if i==0:
                    ax.set_title(title)
                if j==0:
                    ax.set_ylabel(f'Camera {camera} RGB')
                ax.set_xticks([])
                ax.set_yticks([])
                err=np.sqrt(np.mean((image-truth)**2,axis=1)).reshape(g.SIZE,g.SIZE)
                ax=axes[2*i+1,j]
                if j==0:
                    ax.imshow(v['masks']['common'].reshape(g.SIZE,g.SIZE),cmap='gray',vmin=0,vmax=1)
                    ax.set_ylabel('Primary mask')
                else:
                    kept.append(ax)
                    picture=ax.imshow(err,cmap='magma',vmin=0,vmax=limit,interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
            fig.colorbar(picture,ax=kept,shrink=.75,label='Linear RGB pixel RMSE')
        for ext in ('pdf','png'):
            fig.savefig(out/f'{route}-relighting.{ext}',dpi=200)
        plt.close(fig)
    return dict(error_map_min=0.,error_map_max=limit,RGB='unit exposure, standard sRGB; display-only clipping',
                channels_clipped_in_previews=clipped)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.freeze_sha256,args.output)
