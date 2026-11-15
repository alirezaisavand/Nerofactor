"""E.11D saved-array diagnostics and fixed pilot definitions; no native imports."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.bell_e11 import ENV, linear_to_srgb
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest

GENERATION_SHA = '5035b22b594848900426fd5dfaa8b96382b57a76cfde97864b7ef7591c46e6d7'
PROTOCOL = 'bell-highlight-pilot-e11d-v1'
VIEWS = ('0', '58', '105')
CASES = ('truth', 'theta_plus', 'theta_minus', 'major_plus', 'major_minus',
         'minor_plus', 'minor_minus', 'isotropic')
Y = np.array([.2126, .7152, .0722])


def mean_environment(definition):
    # Integral over the complete sphere /4pi; integral max(w.axis,0)^p =2pi/(p+1).
    return np.array(definition['ambient'], dtype=float) + sum(
        (np.array(v['RGB'])/(2*(v['power']+1)) for v in definition['lobes']), np.zeros(3))


def candidates():
    result = {'L0_v3': ENV}
    for name, power, axes in [
            ('L1_three_oblique', 48, [(1,1,1),(-1,-1,1),(1,-1,-1)]),
            ('L2_six_cardinal', 64, [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])]:
        amplitude = (mean_environment(ENV)-.01)*2*(power+1)/len(axes)
        result[name] = {'ambient':[.01]*3, 'lobes':[
            {'axis':(np.array(axis)/np.linalg.norm(axis)).tolist(),
             'RGB':amplitude.tolist(), 'power':power} for axis in axes]}
    return result


def subset(hit):
    """At most192 points/view: three fixed-order foreground ranks in each16x16 cell."""
    require(hit.shape == (128,128) and hit.dtype == np.bool_, '128x128 Boolean hit image required')
    chosen=[]
    for y in range(0,128,16):
        for x in range(0,128,16):
            yy,xx=np.nonzero(hit[y:y+16,x:x+16]);ids=(yy+y)*128+xx+x
            if len(ids): chosen.extend(ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).astype(int))].tolist())
    return np.array(sorted(chosen),dtype=np.int64)


def decoded_png(rgb):
    v=linear_to_srgb(rgb)/255.
    return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)


def measures(components):
    require(components.ndim==3 and components.shape[1:]==(3,3) and len(components)>0,
            'Nonempty Nx(diffuse,specular,total)xRGB required')
    require(np.isfinite(components).all() and (components>=0).all(), 'Invalid components')
    d,s,t=[components[:,i] for i in range(3)]
    require(np.array_equal(d+s,t), 'Diffuse/specular sum mismatch')
    yd,ys,yt=d@Y,s@Y,t@Y;fraction=ys/np.maximum(yt,1e-30)
    highlight=(ys>=.01)&(ys>=3*np.median(ys))
    result=dict(pixels=len(t),mean_diffuse_luminance=float(yd.mean()),mean_specular_luminance=float(ys.mean()),
                integrated_specular_to_diffuse=float(ys.sum()/max(yd.sum(),1e-30)),
                integrated_specular_fraction=float(ys.sum()/max(yt.sum(),1e-30)),
                specular_fraction_median=float(np.median(fraction)),
                specular_dominant_fraction=float((fraction>=.5).mean()),
                highlight_coverage=float(highlight.mean()),
                specular_luminance_p50=float(np.median(ys)),specular_luminance_p99=float(np.quantile(ys,.99)),
                specular_contrast_p99_minus_p50=float(np.quantile(ys,.99)-np.median(ys)),
                specular_contrast_ratio=float(np.quantile(ys,.99)/max(np.median(ys),1e-30)),
                linear_rgb_max=float(t.max()),out_of_png_domain=bool((t>1).any()))
    if not result['out_of_png_domain']:
        delta=np.abs(linear_to_srgb(t).astype(int)-linear_to_srgb(d).astype(int)).max(1)
        error=decoded_png(t)-t
        result.update(specular_zero_code_fraction=float((delta==0).mean()),
                      specular_at_least_two_codes_fraction=float((delta>=2).mean()),
                      quantization_linear_rgb_rmse=float(np.sqrt(np.mean(error**2))),
                      quantization_rmse_over_specular_rms=float(np.linalg.norm(error)/max(np.linalg.norm(s),1e-30)))
    return result,highlight


def percentile(values):
    return {str(q):float(np.quantile(values,q)) for q in (0,.05,.25,.5,.75,.95,1)}


def write_json(path,value):
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def offline(root,output):
    from benchmarks.verify_bell_e11 import verify
    require(root.is_absolute() and root.resolve()==root,'Explicit original benchmark root required')
    require(digest(root/'geometry/generation.json')==GENERATION_SHA,'v3 report changed')
    require(not output.exists() and output.is_absolute() and root not in output.parents,'Fresh separate output required')
    verified=verify(root,GENERATION_SHA)  # Saved arrays only; no shader, camera pickle execution or CUDA.
    report=json.loads((root/'geometry/generation.json').read_text())
    output.mkdir()
    rows=[];ratios=[];all_components=[];weak=[];regions=[];cue_components={}
    for i in range(128):
        with np.load(root/'references'/(str(i)+'.npz'),allow_pickle=False) as z:
            hit=z['hit'];components=z['linear_components'][hit];r,strong=measures(components)
            all_components.append(components)
            with np.load(root/'geometry/labels'/(str(i)+'.npz'),allow_pickle=False) as labels:
                widths=labels['widths_major_minor'];ratio=widths[:,0]/widths[:,1];ratios.append(ratio)
                r.update(camera_id=str(i),ratio_median=float(np.median(ratio)),ratio_ge_1_5=float((ratio>=1.5).mean()),
                         weak_specular_and_anisotropic=float((((components[:,1]@Y)<.005)&(ratio>=1.5)).mean()),
                         pilot_points=len(subset(hit)))
                # Fixed world Z tertiles in coordinate bounds, independent of highlight/material labels.
                p=labels['position'];zbin=np.digitize(p[:,2],[-.3333333333333333,.3333333333333333])
                quadrant=(p[:,0]>=0).astype(int)+2*(p[:,1]>=0).astype(int)
                for k in range(12):
                    selected=zbin*4+quadrant==k
                    if selected.any():
                        regions.append(dict(camera_id=str(i),region_id=k,**measures(components[selected])[0]))
            mask=np.zeros((128,128),bool);mask[hit]=strong
            from scipy.ndimage import label
            components_id,num=label(mask);sizes=np.bincount(components_id.ravel())[1:]
            cue_components[str(i)]={'components':int(num),'pixels':int(strong.sum()),
                                   'pixels_in_components_at_most_two':int(sizes[sizes<=2].sum())}
            weak.append((components[:,1]@Y)<.005);rows.append(r)
    combined=np.concatenate(all_components);summary,unused=measures(combined)
    ratio=np.concatenate(ratios)
    summary['highlight_coverage']=sum(v['pixels'] for v in cue_components.values())/len(combined)
    summary.update(width_ratio_quantiles=percentile(ratio),ratio_ge_1_5=float((ratio>=1.5).mean()),
                   ratio_ge_2=float((ratio>=2).mean()),weak_specular_luminance_lt_005=float(np.concatenate(weak).mean()),
                   highlight_connected_components=cue_components)
    result=dict(schema_version=1,protocol='bell-e11d-v3-saved-cue-analysis-v1',status='complete',
                generation_sha256=GENERATION_SHA,root=str(root),scientific_renderer_queries=0,
                original_verification_status=verified['status'],definitions={
                    'luminance':'linear RGB dot(.2126,.7152,.0722)',
                    'highlight':'Y_spec>=.01 AND Y_spec>=3*median_foreground(Y_spec), per view; descriptive only',
                    'weak_proxy':'Y_spec<.005; NOT an orientation derivative or identifiability decision',
                    'specular_quantization':'max absolute channel code difference: encode(D+S) versus encode(D)',
                    'regions':'world Z boundaries -1/3,+1/3 and XY sign quadrants; untransformed oracle coordinates'},
                aggregate=summary,views=rows,regions=regions,
                limitation='Observed width gaps, specular strength and code differences do not establish orientation sensitivity or seven-parameter identifiability.')
    write_json(output/'observations.json',result)
    with (output/'per_view.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    diagnostic_figures(root,output,rows)
    return result


def diagnostic_figures(root,output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='e11d-cue-analysis-v1'
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for row,i in enumerate((0,58,105)):
        with np.load(root/'references'/(str(i)+'.npz'),allow_pickle=False) as z, np.load(root/'geometry/labels'/(str(i)+'.npz'),allow_pickle=False) as labels:
            c=z['linear_components'];hit=z['hit'];ratio=np.full(128*128,np.nan);ratio[labels['pixel_indices']]=labels['widths_major_minor'][:,0]/labels['widths_major_minor'][:,1]
            yd=c[:,:,0]@Y;ys=c[:,:,1]@Y;fraction=ys/np.maximum(yd+ys,1e-30)
            panels=[(linear_to_srgb(c[:,:,2]),'New observation (sRGB)',None,None,None),
                    (np.where(hit,ys,np.nan),'Specular linear luminance',0,.04,'magma'),
                    (np.where(hit,fraction,np.nan),'Specular fraction',0,1,'viridis'),
                    (ratio.reshape(128,128),'Major/minor width ratio',1,2,'viridis')]
            for col,(a,title,lo,hi,cmap) in enumerate(panels):
                im=axes[row,col].imshow(a,vmin=lo,vmax=hi,cmap=cmap);axes[row,col].axis('off')
                axes[row,col].set_title(('Camera'+str(i)+'\n' if col==0 else '')+title,fontsize=10)
                if col:fig.colorbar(im,ax=axes[row,col],shrink=.6)
    fig.tight_layout();save_figure(fig,output/'cue_maps.svg');fig.savefig(output/'cue_maps.png',dpi=120)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6));ids=np.arange(128)
    ax[0].plot(ids,[r['integrated_specular_fraction'] for r in rows],label='Integrated specular fraction')
    ax[0].plot(ids,[r['highlight_coverage'] for r in rows],label='Highlight coverage');ax[0].set_xlabel('Camera ID');ax[0].set_ylabel('Fraction');ax[0].legend()
    ax[1].plot(ids,[r['specular_zero_code_fraction'] for r in rows],label='Specular addition changes zero codes')
    ax[1].plot(ids,[r['quantization_rmse_over_specular_rms'] for r in rows],label='Quantization RMS / specular RMS')
    ax[1].set_xlabel('Camera ID');ax[1].set_ylabel('Fraction / ratio');ax[1].legend(fontsize=8)
    fig.tight_layout();save_figure(fig,output/'view_statistics.svg');fig.savefig(output/'view_statistics.png',dpi=120);plt.close('all')


def save_figure(fig,path):
    fig.savefig(path,bbox_inches='tight',metadata={'Date':None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=offline(a.root,a.output)
    print(json.dumps({k:v for k,v in r['aggregate'].items() if k!='highlight_connected_components'},sort_keys=True))


if __name__=='__main__':main()
