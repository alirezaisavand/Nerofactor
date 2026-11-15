"""File-only evaluation of the completed, explicitly selected E11D pilot.

No torch/native modules, reference rendering, new samples or model execution.
Sparse sensitivity plots show only the actual frozen probe pixels.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.bell_highlight_e11d import (ROOT, GENERATION_SHA, VIEWS, CASES, Y,
    candidates, subset, measures, linear_to_srgb, write_json, save_figure)
from experiments.bell_highlight_pilot_e11d import LIMITS, sensitivity, recommend
from benchmarks.bell_e11 import convergence
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, baseline, digest
from infrastructure.verified_ray_backend import identity

COMMIT='3dcd2218d36e9afd9e17349a672d7388649c77fa'
FREEZE_SHA='33e500cad37190b4bd0e4f9adc69b39533d631e92d2fad7ec11538b8f79e496c'
REPORT_SHA='4ad7c79fb63713c5710aa8407c3229fb8bd71341dc81d39e985cb24608a3f1f1'
FACTORS={'orientation':(1,2),'major':(3,4),'minor':(5,6),'isotropic':(7,)}


def replay_equal(measured, saved, path='report'):
    """Exact decisions/counts/identities; finite float64 diagnostics allow replay roundoff.

    atol1e-12,rtol1e-10 apply ONLY to comparing diagnostic numbers across NumPy/
    BLAS versions. All scientific thresholds and their recomputed decisions stay exact.
    """
    if isinstance(measured,dict):
        require(isinstance(saved,dict) and set(measured)==set(saved),'Field mismatch: '+path)
        for k in measured:replay_equal(measured[k],saved[k],path+'.'+k)
    elif isinstance(measured,list):
        require(isinstance(saved,list) and len(measured)==len(saved),'List mismatch: '+path)
        for i,(a,b) in enumerate(zip(measured,saved)):replay_equal(a,b,path+'.'+str(i))
    elif type(measured) in (bool,int,str) or measured is None:
        require(type(measured) is type(saved) and measured==saved,'Exact value mismatch: '+path)
    else:
        require(isinstance(saved,(float,int)) and not isinstance(saved,bool) and
                np.isfinite(measured) and np.isfinite(saved) and
                np.isclose(measured,saved,atol=1e-12,rtol=1e-10),'Numerical replay mismatch: '+path)


def record(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)}


def verify_layout(pilot,expected):
    paths=list(pilot.rglob('*'))
    require(all(not p.is_symlink() for p in paths),'Unexpected symlink in native capture')
    require({p.relative_to(pilot).as_posix() for p in paths if p.is_file()}==set(expected),
            'Unexpected partial/unmanifested file')
    directories={p.relative_to(pilot).as_posix() for p in paths if p.is_dir()}
    # CUDA may create these empty HOME directories; no cache payload is admitted.
    require(directories<={'runtime','runtime/.nv','runtime/.nv/ComputeCache'},'Unexpected native output directory')
    return sorted(directories)


def response_fields(fine,coarse):
    require(fine.shape==coarse.shape and fine.ndim==4 and fine.shape[1:]==(8,3,3),'Eight paired cases required')
    require(np.isfinite(fine).all() and np.isfinite(coarse).all() and (fine>=0).all() and (coarse>=0).all(), 'Invalid capture')
    total=fine[:,:,2];low=coarse[:,:,2]
    code=linear_to_srgb(total).astype(np.int16)
    fields={}
    for name,ids in FACTORS.items():
        amplitude=np.abs(total[:,ids]-total[:,0,None]).max((1,2))
        grid=np.maximum(np.abs(total[:,ids]-low[:,ids]).max((1,2)),np.abs(total[:,0]-low[:,0]).max(1))
        codes=np.abs(code[:,ids]-code[:,0,None]).max((1,2))
        # Exact frozen gates; none of the replay tolerances change these masks.
        resolved=(amplitude>=.003)&(amplitude>=4*grid)&(codes>=2)
        fields[name]={'amplitude':amplitude,'grid_difference':grid,'code_difference':codes,'resolved':resolved,
                      'absolute_gate':amplitude>=.003,'grid_gate':amplitude>=4*grid,'code_gate':codes>=2}
    return fields


def verify_capture(z,reference,probe_ids):
    expected={'linear_components','coarse_components','hit','pixel_indices','paired_components',
              'paired_coarse_components','component_effective_samples','component_max_sample_fraction'}
    require(set(z)==expected,'Missing/extra capture arrays')
    require(z['hit'].shape==(128,128) and z['hit'].dtype==np.bool_ and
            np.array_equal(z['hit'],reference['hit']) and np.array_equal(z['pixel_indices'],probe_ids), 'Hit/probe selection mismatch')
    require(z['pixel_indices'].dtype==np.int64,'Probe index dtype mismatch')
    for name,shape in [('linear_components',(128,128,3,3)),('coarse_components',(128,128,3,3)),
                       ('paired_components',(len(probe_ids),8,3,3)),('paired_coarse_components',(len(probe_ids),8,3,3)),
                       ('component_effective_samples',(128,128,2)),('component_max_sample_fraction',(128,128,2))]:
        a=z[name];require(a.shape==shape and a.dtype==np.float64 and np.isfinite(a).all() and (a>=0).all(),'Invalid capture array: '+name)
    for name in ('linear_components','coarse_components','paired_components','paired_coarse_components'):
        a=z[name];require(np.array_equal(a[...,0,:]+a[...,1,:],a[...,2,:]),'Component sum mismatch: '+name)
    for paired,full in [('paired_components','linear_components'),('paired_coarse_components','coarse_components')]:
        require(np.array_equal(z[paired][:,0],z[full].reshape(-1,3,3)[probe_ids]),'Paired truth differs from full image')
    require((z['component_effective_samples']<=16384+1e-9).all() and (z['component_max_sample_fraction']<=1).all(), 'Invalid concentration diagnostics')


def analyze(batch,benchmark,output):
    from benchmarks.verify_bell_e11 import verify
    require(all(p.is_absolute() and p.resolve()==p for p in (batch,benchmark,output)),'Explicit unaliased paths required')
    require(not output.exists() and batch not in output.parents and benchmark not in output.parents,'Fresh separate analysis output required')
    pilot=batch/'pilot';report_path=pilot/'pilot.json'
    require(digest(report_path)==REPORT_SHA,'Selected native report hash mismatch')
    report=json.loads(report_path.read_text());launch=json.loads((batch/'launch.json').read_text())
    auth_path=Path(str(batch)+'-authorization.json');auth=json.loads(auth_path.read_text())
    freeze_path=ROOT/'docs/protocols/BELL_HIGHLIGHT_PILOT_E11D_V1.freeze.json'
    require(digest(freeze_path)==FREEZE_SHA,'Pilot freeze changed')
    freeze=json.loads(freeze_path.read_text())
    require(report['status']=='complete' and report['cleanup_verified'] is True and
            (batch/'exit-code.txt').read_text().strip()=='0','Native work/cleanup/exit not complete')
    require(report['commit']==launch['commit']==auth['commit']==COMMIT and
            report['freeze_sha256']==launch['freeze_sha256']==auth['freeze_sha256']==FREEZE_SHA and
            report['authorization_sha256']==launch['authorization_sha256']==digest(auth_path) and
            auth['approved'] is True and auth['host_output_directory']==launch['batch']==str(batch) and
            auth['host_input_benchmark']==launch['benchmark']==str(benchmark) and
            auth['output_directory']=='/outputs/pilot' and report['backend']==identity() and
            auth['image_id']==identity()['image_id'] and launch['gpu']==auth['selected_gpu']=='0', 'Execution/approval identity mismatch')
    require(report['generation_sha256']==digest(benchmark/'geometry/generation.json')==GENERATION_SHA and
            report['lighting']==freeze['lighting']==candidates() and
            report['limits']==auth['limits']==freeze['limits']==LIMITS and
            report['source_files']==freeze['files'],'Frozen scientific definition mismatch')
    for name,sha in report['source_files'].items():require(digest(ROOT/name)==sha,'Frozen source changed: '+name)
    for stage in PINS:baseline(stage,PINS[stage])
    verified=verify(benchmark,GENERATION_SHA)  # Saved hashes/encodings only, never renders.
    require(verified['status']=='pass','Original v3 verification failed')
    require(len(report['outputs'])==27 and {x['path'] for x in report['outputs']}==
            {v+'-'+l+suffix for v in VIEWS for l in candidates() for suffix in ('.npz','.json','.png')},'Incomplete native inventory')
    for item in report['outputs']:
        path=pilot/item['path'];require(path.resolve()==path and pilot in path.parents and
            path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],'Native artifact changed: '+item['path'])
    empty_runtime_dirs=verify_layout(pilot,{x['path'] for x in report['outputs']}|{'pilot.json'})
    require(all(report[k]==0 for k in ('native_model_calls','native_shader_calls','optimizer_returns')),'Unexpected training work')
    rows=[];probe_rows=[];captures={};expected_shadow=expected_brdf=0;parity=0.
    for camera in VIEWS:
        with np.load(benchmark/'references'/(camera+'.npz'),allow_pickle=False) as f:
            reference={k:f[k].copy() for k in ('hit','linear_components','coarse_components')}
        ids=subset(reference['hit']);expected_shadow+=int(reference['hit'].sum())*16384
        expected_brdf+=(int(reference['hit'].sum())+7*len(ids))*16384
        with np.load(benchmark/'geometry/labels'/(camera+'.npz'),allow_pickle=False) as f:
            offset=np.searchsorted(f['pixel_indices'],ids)
            labels={k:f[k][offset] for k in ('position','normals','widths_major_minor','orientation_eligible')}
            require(np.array_equal(f['pixel_indices'][offset],ids),'Probe/source label identity mismatch')
        for light in candidates():
            with np.load(pilot/(camera+'-'+light+'.npz'),allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
            verify_capture(z,reference,ids);hit=z['hit'];fine=z['linear_components'];coarse=z['coarse_components']
            if light=='L0_v3':
                error=float(max(abs(fine-reference['linear_components']).max(),abs(coarse-reference['coarse_components']).max()))
                require(error<=2e-10,'L0/v3 reference parity failed');parity=max(parity,error)
            metrics,_=measures(fine[hit]);conv=convergence(fine[hit,2],coarse[hit,2])
            gates={case:convergence(z['paired_components'][:,k,2],z['paired_coarse_components'][:,k,2]) for k,case in enumerate(CASES)}
            sens=sensitivity(z['paired_components'],z['paired_coarse_components'])
            eligible=conv['passed'] and all(v['passed'] for v in gates.values()) and not metrics['out_of_png_domain'] and sens['png_domain']
            row=dict(camera_id=camera,lighting=light,metrics=metrics,convergence=conv,paired_convergence=gates,sensitivity=sens,eligible=eligible)
            ess=z['component_effective_samples'].reshape(-1,2)[np.flatnonzero(hit),1]
            row['specular_weight_concentration']={'minimum_effective_samples':float(ess.min()),'median_effective_samples':float(np.median(ess)),
                'maximum_sample_fraction':float(z['component_max_sample_fraction'][hit,1].max()),
                'meaning':'Fixed QMC contribution concentration; not an independent variance estimate'}
            native=next(v for v in report['views'] if v['camera_id']==camera and v['lighting']==light)
            replay_equal(row,native);replay_equal(row,json.loads((pilot/(camera+'-'+light+'.json')).read_text()))
            with Image.open(pilot/(camera+'-'+light+'.png')) as image:
                require(np.array_equal(np.asarray(image),linear_to_srgb(fine[:,:,2])),'Native PNG differs from saved reference')
            fields=response_fields(z['paired_components'],z['paired_coarse_components'])
            ratio=labels['widths_major_minor'][:,0]/labels['widths_major_minor'][:,1]
            for j,pixel in enumerate(ids):
                data=dict(camera_id=camera,lighting=light,pixel_index=int(pixel),pixel_x=int(pixel%128),pixel_y=int(pixel//128),
                          world_x=float(labels['position'][j,0]),world_y=float(labels['position'][j,1]),world_z=float(labels['position'][j,2]),
                          width_ratio=float(ratio[j]),orientation_label_eligible=bool(labels['orientation_eligible'][j]))
                for name,v in fields.items():
                    data.update({name+'_'+key:(bool(a[j]) if a.dtype==np.bool_ else int(a[j]) if a.dtype.kind in 'iu' else float(a[j]))
                                 for key,a in v.items()})
                probe_rows.append(data)
            captures[camera,light]=(z,fields);rows.append(row)
    recommendation=recommend(rows);replay_equal(recommendation,report['recommendation'])
    counts=report['counters'];require(counts['shadow_directions']==expected_shadow and
        counts['brdf_point_directions']==counts['brdf_point_direction_returns']==expected_brdf and
        counts['light_weighted_point_directions']==counts['light_weighted_point_direction_returns']==expected_brdf*3 and
        counts['valid_shadow_queries']==counts['valid_shadow_returns']<=expected_shadow,'Scientific work accounting mismatch')
    require(report['elapsed_seconds']<LIMITS['wall_seconds'] and report['host_peak_rss_bytes']<=64*1024**3 and
            sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())<=LIMITS['output_bytes'],'Recorded resource ceiling exceeded')
    pooled={}
    for light in candidates():
        records=[x for x in probe_rows if x['lighting']==light]
        combined=np.concatenate([captures[v,light][0]['linear_components'][captures[v,light][0]['hit']] for v in VIEWS])
        pooled[light]={'probes':len(records),'integrated_specular_fraction':measures(combined)[0]['integrated_specular_fraction'],
                      'highlight_coverage':sum(x['metrics']['highlight_coverage']*x['metrics']['pixels'] for x in rows if x['lighting']==light)/len(combined)}
        for name in FACTORS:
            pooled[light][name]={k:float(np.mean([x[name+'_'+k] for x in records])) for k in ('resolved','code_gate','absolute_gate','grid_gate')}
            pooled[light][name]['zero_code_fraction']=float(np.mean([x[name+'_code_difference']==0 for x in records]))
            pooled[light][name]['median_response']=float(np.median([x[name+'_amplitude'] for x in records]))
    output.mkdir()
    summary=dict(schema_version=1,protocol='bell-highlight-pilot-file-analysis-e11d-v1',verification_status='pass',
                 decision=recommendation['decision'],recommended_lighting=recommendation['recommended'],
                 native_report=record(report_path),execution_receipts=[record(batch/x) for x in ('launch.json','exit-code.txt','console.log','container.cid')],
                 authorization=record(auth_path),freeze=record(freeze_path),v3_generation=record(benchmark/'geometry/generation.json'),
                 source_files=report['source_files'],native_artifacts=report['outputs'],runtime=report['runtime'],backend=report['backend'],
                 counters=counts,elapsed_seconds=report['elapsed_seconds'],
                 empty_runtime_directories=empty_runtime_dirs,
                 host_peak_rss_bytes=report['host_peak_rss_bytes'],cuda_peak_allocated_bytes=report['cuda_peak_allocated_bytes'],
                 cuda_peak_reserved_bytes=report['cuda_peak_reserved_bytes'],native_cleanup_verified=True,L0_v3_max_absolute_difference=parity,
                 replay_diagnostic_tolerance={'atol':1e-12,'rtol':1e-10,'scientific_decisions_counts_identities':'exact'},
                 rows=rows,pooled=pooled,selection=recommendation,new_scientific_queries=0,new_gpu_calls=0,
                 limitations=['All L1/L2 sensitivity values remain provisional because their references fail convergence.',
                              'Sparse maps show only246 actual probe pixels; no interpolation or new rendering.',
                              'Conditional one-view RGB rank is not seven-parameter material identifiability.',
                              'No independent-tape variance or full128-view coverage estimate.'])
    write_json(output/'analysis.json',summary)
    write_csv(output/'probe_comparisons.csv',probe_rows)
    compact=[]
    for r in rows:
        compact.append(dict(camera_id=r['camera_id'],lighting=r['lighting'],reference_eligible=r['eligible'],
            convergence_relative_l2=r['convergence']['relative_l2'],convergence_max_abs=r['convergence']['maximum_rgb_difference'],
            linear_rgb_max=r['metrics']['linear_rgb_max'],highlight_coverage=r['metrics']['highlight_coverage'],
            specular_fraction=r['metrics']['integrated_specular_fraction'],
            **{name+'_resolved':r['sensitivity'][name]['resolved_fraction'] for name in FACTORS},
            orientation_two_codes=r['sensitivity']['orientation']['two_code_fraction'],
            specular_median_effective_samples=r['specular_weight_concentration']['median_effective_samples']))
    write_csv(output/'view_comparisons.csv',compact)
    figures(captures,output)
    return summary


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def figures(captures,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm,ListedColormap
    plt.rcParams['svg.hashsalt']='e11d-pilot-results-v1'
    lights=list(candidates());titles=['L0 (v3)','L1 (oblique)','L2 (cardinal)']
    def setup():
        f,ax=plt.subplots(3,3,figsize=(9,9))
        for r,v in enumerate(VIEWS):
            for c in range(3):ax[r,c].set_title('Cam'+v+' / '+titles[c],fontsize=10);ax[r,c].axis('off')
        return f,ax
    def save(f,name):
        save_figure(f,output/(name+'.svg'));f.savefig(output/(name+'.png'),dpi=130,bbox_inches='tight');plt.close(f)
    fig,axes=setup()
    for r,v in enumerate(VIEWS):
        for c,l in enumerate(lights):axes[r,c].imshow(linear_to_srgb(captures[v,l][0]['linear_components'][:,:,2]))
    fig.suptitle('Saved independent RGB; unchanged sRGB encoding, no exposure adjustment');save(fig,'rgb_comparison')
    fig,axes=setup();maximum=max(float((z['linear_components'][:,:,1]@Y).max()) for z,f in captures.values())
    for r,v in enumerate(VIEWS):
        for c,l in enumerate(lights):
            z,_=captures[v,l];im=axes[r,c].imshow(np.where(z['hit'],z['linear_components'][:,:,1]@Y,np.nan),vmin=0,vmax=maximum,cmap='magma')
    fig.colorbar(im,ax=list(axes.ravel()),shrink=.6,label='Specular linear luminance (common scale)')
    fig.suptitle('Saved specular component; L1/L2 references unconverged');save(fig,'specular_comparison')
    for name in ('orientation','major','minor','isotropic'):
        fig,axes=setup();maximum=max(float(f[name]['amplitude'].max()) for z,f in captures.values())
        for r,v in enumerate(VIEWS):
            for c,l in enumerate(lights):
                z,fields=captures[v,l];values=fields[name];pixels=z['pixel_indices'];a=axes[r,c]
                a.imshow(np.where(z['hit'],.8,np.nan),vmin=0,vmax=1,cmap='gray')
                im=a.scatter(pixels%128,pixels//128,c=values['amplitude'],s=15,cmap='magma',vmin=0,vmax=maximum)
                ok=values['resolved'];a.scatter(pixels[ok]%128,pixels[ok]//128,s=27,facecolors='none',edgecolors='cyan',linewidths=.6)
                a.text(.5,-.04,str(int(ok.sum()))+'/'+str(len(pixels))+' resolved',transform=a.transAxes,fontsize=8,ha='center')
        fig.colorbar(im,ax=list(axes.ravel()),shrink=.6,label='Max channel |counterfactual − truth| (linear RGB)')
        fig.suptitle(name+': actual probe pixels only; cyan = frozen local gates, not dataset admission')
        save(fig,name+'_sensitivity')
    fig,axes=setup();colors=ListedColormap(['#222222','#b34e32','#6c86b4','#e8b32c']);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    for r,v in enumerate(VIEWS):
        for c,l in enumerate(lights):
            z,fields=captures[v,l];p=z['pixel_indices'];d=fields['orientation'];category=np.where(d['resolved'],3,np.where(d['code_gate'],2,np.where(d['code_difference']>0,1,0)))
            axes[r,c].imshow(np.where(z['hit'],.8,np.nan),vmin=0,vmax=1,cmap='gray')
            im=axes[r,c].scatter(p%128,p//128,c=category,s=17,cmap=colors,norm=norm)
    bar=fig.colorbar(im,ax=list(axes.ravel()),shrink=.6,ticks=[0,1,2,3]);bar.ax.set_yticklabels(['0 codes','1 code','≥2; local gates fail','All local gates pass'])
    fig.suptitle('Orientation and native quantization; L1/L2 lack global reference convergence')
    save(fig,'orientation_quantization');plt.close('all')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','benchmark','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=analyze(a.batch,a.benchmark,a.output)
    print(json.dumps({'verification':r['verification_status'],'decision':r['decision'],'recommended':r['recommended_lighting'],'pooled':r['pooled']},indent=2))


if __name__=='__main__':main()
