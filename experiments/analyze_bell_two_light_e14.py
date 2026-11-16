"""File-only E.14 verification, summaries and paper figures; no scientific queries."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import bell_two_light_e14 as study
from experiments.analyze_bell_highlight_e11d import record,replay_equal,verify_layout
from experiments.bell_highlight_e11d import write_json,save_figure
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline,PINS,digest

COMMIT='2bff40f8c9f323c912325d61712468498e78e5d7'
FREEZE_SHA='662d14190b3394655a2cd497ab08c0aec9e64d3094b85d2657bed11a7ee1d102'
AUTH_SHA='c410b09fb4adde2559be083ec8b57915dafce3a921b32108fa114d3a50a844f1'
ARMS=('original','stronger')
OBSERVATIONS=('L2','rotated_L2','two_light')
MODELS=('ideal','gaussian_linear','gaussian_quantized8')


def check_work(r):
    require(r['counters']==dict(brdf_attempts=1658060800,brdf_returns=1658060800,
        environment_attempts=27525120,environment_returns=27525120,new_primary_rays=0,
        new_shadow_queries=0,new_outgoing_queries=0,new_L2_brdf_queries=0,model_calls=0,optimizer_returns=0),
        'Exact scientific work/zero-query boundary differs')
    require(r['status']=='complete' and r['scientific_work_complete'] is True and
            r['cleanup_verified'] is True and r['completed_grids']==[256,512],
            'Incomplete scientific work, grids or native cleanup')


def check_grid(z,valid,pixels,ids,L2):
    require(set(z)=={'components','valid','pixel_indices','camera_ids','completed_probes'},'Grid inventory mismatch')
    require(z['valid'].dtype==z['completed_probes'].dtype==np.bool_ and
            np.array_equal(z['valid'],valid) and z['completed_probes'].shape==(84,) and z['completed_probes'].all(),
            'Incomplete/changed correspondence masks')
    require(z['camera_ids'].dtype==z['pixel_indices'].dtype==np.int64 and
            np.array_equal(z['camera_ids'],ids) and np.array_equal(z['pixel_indices'],pixels),'Grid identity mismatch')
    a=z['components']
    require(a.shape==(2,2,4,84,12,3,3) and a.dtype==np.float64 and np.isnan(a[:,:,~valid]).all(),
            'Invalid dtype/layout or fabricated unavailable observations')
    require(np.array_equal(a[:,0],L2,equal_nan=True),'Preserved L2 values differ')
    for arm in range(2):
        for light in range(2):
            measured=a[arm,light,valid]
            require(np.isfinite(measured).all() and (measured>=0).all() and
                    np.array_equal(measured[:,:,0]+measured[:,:,1],measured[:,:,2]),'Invalid RGB/component sum')
            require(np.array_equal(measured[:,:,0],np.repeat(measured[:,:1,0],12,axis=1)),
                    'Diffuse changed with roughness/orientation')
    require(np.array_equal(a[0,:,valid,7],a[1,:,valid,7]) and
            np.array_equal(a[0,:,valid,:,0],a[1,:,valid,:,0]),'Shared canonical isotropic/diffuse control changed')
    return a


def verify(batch,authorization):
    fp=ROOT/'docs/protocols/BELL_TWO_LIGHT_OBSERVABILITY_E14_V1.freeze.json'
    require(digest(fp)==FREEZE_SHA and digest(authorization)==AUTH_SHA,'Freeze/authorization identity mismatch')
    f=json.loads(fp.read_text());a=json.loads(authorization.read_text());launch=json.loads((batch/'launch.json').read_text())
    roots={k:Path(v) for k,v in f['host_inputs'].items()}
    require(all(p.is_absolute() and p.resolve()==p for p in (batch,authorization,*roots.values())),
            'Explicit unaliased input paths required')
    require(str(batch)==f['host_output_directory']==a['host_output_directory']==launch['batch'] and
            a['output_directory']=='/outputs/two-light' and a['approved'] is True and
            a['commit']==launch['commit']==COMMIT and a['freeze_sha256']==launch['freeze_sha256']==FREEZE_SHA and
            launch['authorization_sha256']==AUTH_SHA and a['image_id']==f['image_id']==study.identity()['image_id'] and
            a['protocol']==f['protocol']==study.PROTOCOL and a['limits']==f['limits']==study.LIMITS and
            a['selected_gpu']==launch['gpu']=='0','Launch/approval binding mismatch')
    require(all(str(p)==launch[k] for k,p in roots.items()),'Declared input mount mismatch')
    require((batch/'exit-code.txt').read_text().strip()=='0','Native process did not exit successfully')
    native=batch/'two-light';r=json.loads((native/'two-light.json').read_text());check_work(r)
    require(r['protocol']==study.PROTOCOL and r['commit']==COMMIT and r['freeze_sha256']==FREEZE_SHA and
            r['authorization_sha256']==AUTH_SHA and r['source_files']==f['files'] and r['limits']==study.LIMITS and
            r['lighting']==f['lights']==list(study.LIGHTS) and r['cases']==list(study.CASES) and
            r['cached_tracing_identity']==study.identity(),'Native scientific identity mismatch')
    args=argparse.Namespace(**roots,freeze=fp,freeze_sha256=FREEZE_SHA)
    _,data=study.definition(args)  # Safe metadata and numeric-cache admission only.
    for stage in PINS:baseline(stage,PINS[stage])
    mounts=dict(v3=roots['benchmark'],e11d=roots['pilot'],e11e=roots['reference'],e11f=roots['factor'],e11g=roots['multiview'])
    mapped=[]
    for row in r['inputs']:
        p=Path(row['path']);require(p.parts[:2]==('/','inputs') and p.parts[2] in mounts,'Unexpected native input')
        host=mounts[p.parts[2]].joinpath(*p.parts[3:])
        require(host.resolve()==host and mounts[p.parts[2]] in host.parents and
                host.stat().st_size==row['bytes'] and digest(host)==row['sha256'],'Consumed artifact changed')
        mapped.append(record(host))
    require(mapped==data[-1],'Consumed input inventory differs')
    expected={'comparison.json','grid-256.npz','grid-512.npz'}
    require(len(r['outputs'])==3 and {x['path'] for x in r['outputs']}==expected,'Output inventory differs')
    for row in r['outputs']:
        p=native/row['path'];require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'Native output hash differs')
    verify_layout(native,expected|{'two-light.json','progress.json'})
    labels,pixels,cache,outgoing,_=data;grids={}
    for size in study.GRIDS:
        with np.load(native/('grid-'+str(size)+'.npz'),allow_pickle=False) as z:
            grids[size]=check_grid({k:z[k] for k in z.files},outgoing['valid'],pixels,outgoing['camera_ids'],cache[size]['L2'])
    replay=study.analyze(grids[512],grids[256],outgoing['valid'])
    replay_equal(replay,r['comparison']);replay_equal(replay,json.loads((native/'comparison.json').read_text()))
    progress=json.loads((native/'progress.json').read_text())
    require(progress['grid']==512 and progress['completed_probes']==84 and progress['counters']==r['counters'],
            'Terminal progress counters differ')
    lines=[json.loads(line) for line in (batch/'console.log').read_text().splitlines() if line.strip()]
    require([x['grid'] for x in lines]==[256,512] and lines[-1]['counters']==r['counters'],'Console work receipts differ')
    size=sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())
    require(r['elapsed_seconds']<840 and r['host_peak_rss_bytes']<=64*1024**3 and size<study.LIMITS['output_bytes'],
            'Recorded resources exceeded limits')
    return f,r,replay,data,grids,size


def summaries(c):
    details=[];deterministic=[];summary=[]
    for row in c['per_probe_information']:
        for obs in OBSERVATIONS:
            x=row['information'][obs];q=x['deterministic8']
            for model in MODELS:
                v=x[model];s=v['singular_values'];std=v.get('total_data_local_std')
                details.append(dict(material=ARMS[row['material']],probe_index=row['probe_index'],valid_views=row['valid_view_count'],
                    observations=obs,rgb_observations=x['rgb_observations'],model=model,RMS_smin=s[-1],
                    condition=v['condition'],gram_condition=v['gram_condition'],
                    grid_operator_RMS=v['grid_difference_operator_norm_RMS'],grid_proxy_positive=v['grid_proxy_smin_lower']>0,
                    mean_ratio_abs_cosine=v['absolute_column_cosines']['logmean_logratio'],
                    theta_local_std_degrees=float(np.rad2deg(std[0])) if std else None,
                    mean_local_std=std[1] if std else None,ratio_local_std=std[2] if std else None))
            deterministic.append(dict(material=ARMS[row['material']],probe_index=row['probe_index'],valid_views=row['valid_view_count'],
                observations=obs,ratio_plus_codes_unchanged=q['identical_code_counterfactuals'][10],
                ratio_minus_codes_unchanged=q['identical_code_counterfactuals'][11],
                ratio_conditional_line_width=q['coordinate_line_sections']['log_ratio']['width'],
                weak_conditional_line_width=q['weakest_ideal_line_section']['width'],
                coarse_truth_code_changes=q['coarse_truth_code_changes'],clipped_channels=q['clipped_channels']))
    groups=[('all',lambda n:True),('at_least_two',lambda n:n>=2)]+[('exactly_'+str(n),lambda v,n=n:v==n) for n in range(1,5)]
    for material in ARMS:
        for group,selected in groups:
            for obs in OBSERVATIONS:
                for model in MODELS:
                    rows=[r for r in details if r['material']==material and selected(r['valid_views']) and r['observations']==obs and r['model']==model]
                    result=dict(material=material,group=group,observations=obs,model=model,points=len(rows),
                        grid_proxy_nonpositive=sum(not r['grid_proxy_positive'] for r in rows))
                    for key in ('RMS_smin','condition','gram_condition','mean_ratio_abs_cosine','theta_local_std_degrees','mean_local_std','ratio_local_std'):
                        values=[r[key] for r in rows if r[key] is not None]
                        result[key+'_median']=float(np.median(values)) if values else None
                        result[key+'_p95']=float(np.quantile(values,.95)) if values else None
                    summary.append(result)
    return details,deterministic,summary


def figures(output,details,coverage,pixels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='bell-two-light-e14-results-v1'
    fig,axes=plt.subplots(2,2,figsize=(9,7))
    for arm,material in enumerate(ARMS):
        rows={obs:[r for r in details if r['material']==material and r['observations']==obs and r['model']=='ideal' and r['valid_views']>=2] for obs in ('L2','two_light')}
        for col,key in enumerate(('RMS_smin','condition')):
            x=[r[key] for r in rows['L2']];y=[r[key] for r in rows['two_light']]
            ax=axes[arm,col];ax.scatter(x,y,s=13,alpha=.7);lo=min(x+y)*.7;hi=max(x+y)*1.4
            ax.plot([lo,hi],[lo,hi],color='gray',ls='--');ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),
                xlabel='L2 multiview',ylabel='Two-light multiview',title=material+': '+key+' (83 matched points)')
    fig.suptitle('Observation-normalized responses; diagonal = no gain');fig.tight_layout();save_figure(fig,output/'conditioning.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));factors=('orientation','major','minor','logmean','logratio')
    for ax,arm in zip(axes,range(2)):
        rows=[next(r for r in coverage if r['material']==arm and r['group']=='at_least_two' and r['factor']==f) for f in factors]
        x=np.arange(5)
        ax.bar(x-.18,[100*r['L2_resolved']/83 for r in rows],.36,label='L2')
        ax.bar(x+.18,[100*r['two_light_resolved']/83 for r in rows],.36,label='Two lights')
        ax.axhline(25,color='gray',ls='--',lw=1,label='Width coverage floor')
        ax.set(xticks=x,xticklabels=('theta','major','minor','log mean','log ratio'),ylim=(0,100),ylabel='Resolved matched points (%)',title=ARMS[arm]);ax.legend(fontsize=8)
    fig.suptitle('Unchanged amplitude, grid-error and two-code conjunction');fig.tight_layout();save_figure(fig,output/'response_coverage.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,7))
    for arm,material in enumerate(ARMS):
        for col,key in enumerate(('theta_local_std_degrees','ratio_local_std')):
            ax=axes[arm,col]
            for obs,style in (('L2','--'),('two_light','-')):
                for model,color in (('gaussian_linear','C0'),('gaussian_quantized8','C1')):
                    x=np.sort([r[key] for r in details if r['material']==material and r['valid_views']>=2 and r['observations']==obs and r['model']==model and r[key] is not None])
                    ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',ls=style,color=color,label=obs+'/'+model)
            ax.set(xscale='log',xlabel=key,ylabel='Empirical CDF, 83 matched points',title=material);ax.legend(fontsize=6)
    fig.suptitle('Conditional local-linearized scales, hypothetical sigma=0.001; not confidence intervals')
    fig.tight_layout();save_figure(fig,output/'noise_quantization.svg');plt.close(fig)


def analyze(batch,authorization,output):
    require(output.is_absolute() and output.resolve()==output and not output.exists() and
            batch not in output.parents and output not in batch.parents,'Fresh separate analysis directory required')
    f,r,c,data,grids,size=verify(batch,authorization)
    details,deterministic,summary=summaries(c)
    output.mkdir()
    for name,rows in [('information_by_probe',details),('deterministic_quantization',deterministic),
                      ('information_summary',summary),('response_coverage',c['response_coverage']),('retained_screen',c['retained_screen'])]:
        with (output/(name+'.csv')).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    figures(output,details,c['response_coverage'],data[1])
    result=dict(schema_version=1,protocol='bell-two-light-e14-offline-analysis-v1',verification_status='pass',decision='FIX_AND_PROCEED',
        native_scientific_decision=c['decision'],full_benchmark_admitted=False,source_commit=COMMIT,freeze_sha256=FREEZE_SHA,
        native_report=record(batch/'two-light/two-light.json'),authorization=record(authorization),
        native_artifacts=[record(p) for p in sorted(batch.rglob('*')) if p.is_file()],input_artifacts=data[-1],frozen_sources=f['files'],
        replay_policy='Existing E11D float64 diagnostic atol1e-12/rtol1e-10 only; decisions, counts, classifications and identities exact.',
        convergence=dict(cases=len(c['convergence']),passed=sum(x['passed'] for x in c['convergence']),
            maximum_RGB_difference=max(x['maximum_rgb_difference'] for x in c['convergence']),maximum_relative_L2=max(x['relative_l2'] for x in c['convergence'])),
        coverage=c['response_coverage'],screens=c['retained_screen'],information_summary=summary,
        counters=r['counters'],runtime=r['runtime'],backend=r['cached_tracing_identity'],
        native_elapsed_seconds=r['elapsed_seconds'],native_batch_bytes=size,
        resources={k:r[k] for k in ('host_peak_rss_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')},
        verification=dict(source_hashes=len(f['files']),input_hashes=len(data[-1]),native_cleanup_reported=r['cleanup_verified'],
            original_L2_bitwise_preserved=True,shared_isotropic_controls_exact=True,saved_decisions_replayed=True,
            direct_Docker_daemon_inspection=False),analysis_new_BRDF_queries=0,analysis_new_geometry_queries=0,
        analysis_new_model_or_optimizer_calls=0,previous_evidence_replaced=False,
        analysis_sources={p:digest(ROOT/p) for p in ('experiments/analyze_bell_two_light_e14.py','experiments/bell_two_light_e14_analysis.py','benchmarks/material_information_e14.py')})
    write_json(output/'analysis.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','authorization','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=analyze(a.batch,a.authorization,a.output)
    print(json.dumps(dict(verification=r['verification_status'],decision=r['native_scientific_decision'],convergence=r['convergence'])))


if __name__=='__main__':main()
