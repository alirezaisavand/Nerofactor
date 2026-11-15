"""File-only E11E verification and scientific comparisons; no rendering or Torch.

Uses only recorded components, quadrature nodes, packed visibility and metadata.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.bell_quadrature_e11e import (ROOT,GRIDS,CASES,LIMITS,PROTOCOL,GENERATION_SHA,
    PILOT_SHA,MIS_SHA,midpoint_grid,inputs,compare,controls)
from experiments.analyze_bell_highlight_e11d import replay_equal,record,response_fields
from experiments.bell_highlight_e11d import save_figure,write_json
from infrastructure.stage_sources import digest,baseline,PINS
from infrastructure.signatures import require
from infrastructure.verified_ray_backend import identity

REPORT_SHA='589f894d653b715a55739aad2e7667ce0105b14eb0119bb5b54e69010f7cc383'
FREEZE_SHA='9ba88ae03ae73be9c7a10f465d6262eba3738c15b66fa9a4ca9ee742ef2fa11f'
COMMIT='c661349e9c5bb4fda79a8dd220bd655185ea8288'


def verify_nodes(size,data,native):
    require(set(data)=={'local_directions','solid_angle_weight','proposal_pdf'},'Quadrature node inventory mismatch')
    d=data['local_directions'];expected,w=midpoint_grid(size)
    require(d.shape==(size*size,3) and d.dtype==np.float64 and np.isfinite(d).all(), 'Invalid node shape/dtype')
    require(hashlib.sha256(d.astype('<f8').tobytes()).hexdigest()==native['node_float64_little_endian_sha256'], 'Native node bytes changed')
    # Byte hashes bind actual nodes. Formula comparison permits libm float64 roundoff only.
    require(abs(d-expected).max()<=2e-15 and (d[:,2]>0).all() and (d[:,2]<1).all(), 'Wrong midpoint domain/order')
    require(data['solid_angle_weight'].shape==data['proposal_pdf'].shape==() and
            float(data['solid_angle_weight'])==native['solid_angle_weight']==w and
            float(data['proposal_pdf'])==native['pdf']==1/(2*np.pi), 'Wrong solid-angle weight/PDF')
    require(abs(np.linalg.norm(d,axis=-1)-1).max()<=2e-14 and abs(w*len(d)-2*np.pi)<=2e-13,'Node unit/area gate failed')


def verify_grid(size,data,pixels,native):
    require(set(data)=={'components','pixel_indices','visibility_packed','visible_counts'},'Grid capture inventory mismatch')
    a=data['components'];packed=data['visibility_packed'];visible=data['visible_counts'];N=size*size
    require(a.shape==(84,8,3,3) and a.dtype==np.float64 and np.isfinite(a).all() and (a>=0).all(), 'Invalid grid radiance')
    require(np.array_equal(a[:,:,0]+a[:,:,1],a[:,:,2]) and np.array_equal(data['pixel_indices'],pixels)
            and data['pixel_indices'].dtype==np.int64,'Component sum/probe identity mismatch')
    require(packed.shape==(84,N//8) and packed.dtype==np.uint8 and visible.shape==(84,) and visible.dtype==np.int64,
            'Invalid packed visibility shape/dtype')
    counts=np.unpackbits(packed,axis=1,bitorder='big').sum(1)
    require(np.array_equal(counts,visible) and int(counts.sum())==native['visible_count'],'Visibility count mismatch')
    require(np.array_equal(a[:,:,0],np.repeat(a[:,:1,0],8,axis=1)), 'Diffuse unexpectedly changes with fixed-color/metal counterfactuals')
    return dict(size=size,visible_fraction=float(counts.sum()/(84*N)),minimum_visible_fraction=float(counts.min()/N),
                maximum_visible_fraction=float(counts.max()/N),visible_count=int(counts.sum()))


def contrast_stability(coarse,fine,mis,mis_coarse):
    q=response_fields(fine,coarse);m=response_fields(mis,mis_coarse);cq=response_fields(coarse,fine);rows=[]
    for name in q:
        difference=q[name]['amplitude']-m[name]['amplitude']
        rows.append(dict(factor=name,probes=len(difference),quadrature_resolved=int(q[name]['resolved'].sum()),
            MIS_resolved=int(m[name]['resolved'].sum()),resolved_membership_changes=int((q[name]['resolved']!=m[name]['resolved']).sum()),
            two_code_membership_changes=int((q[name]['code_gate']!=m[name]['code_gate']).sum()),
            median_quadrature_response=float(np.median(q[name]['amplitude'])),median_MIS_response=float(np.median(m[name]['amplitude'])),
            median_quad256_response=float(np.median(cq[name]['amplitude'])),
            maximum_quad512_vs_quad256_response_change=float(abs(q[name]['amplitude']-cq[name]['amplitude']).max()),
            maximum_response_change=float(abs(difference).max()),median_absolute_response_change=float(np.median(abs(difference))),
            quadrature_zero_codes=int((q[name]['code_difference']==0).sum()),MIS_zero_codes=int((m[name]['code_difference']==0).sum())))
    return rows,q,m


def analyze(batch,benchmark,pilot,output):
    require(all(p.is_absolute() and p.resolve()==p for p in (batch,benchmark,pilot,output)), 'Explicit unaliased paths required')
    require(not output.exists() and all(p not in output.parents for p in (batch,benchmark,pilot)), 'New separate analysis directory required')
    root=batch/'quadrature';report_path=root/'quadrature.json'
    require(digest(report_path)==REPORT_SHA,'Selected E11E report changed')
    r=json.loads(report_path.read_text());launch=json.loads((batch/'launch.json').read_text())
    auth_path=Path(str(batch)+'-authorization.json');auth=json.loads(auth_path.read_text())
    freeze_path=ROOT/'docs/protocols/BELL_QUADRATURE_E11E_V1.freeze.json'
    require(digest(freeze_path)==FREEZE_SHA,'E11E freeze changed');freeze=json.loads(freeze_path.read_text())
    require(r['status']=='complete' and r['cleanup_verified'] is True and (batch/'exit-code.txt').read_text().strip()=='0', 'Execution/cleanup/exit incomplete')
    require(r['protocol']==PROTOCOL and r['commit']==launch['commit']==auth['commit']==COMMIT and
            r['freeze_sha256']==launch['freeze_sha256']==auth['freeze_sha256']==FREEZE_SHA and
            r['authorization_sha256']==launch['authorization_sha256']==digest(auth_path) and
            auth['approved'] is True and auth['host_output_directory']==launch['batch']==str(batch) and
            auth['host_benchmark']==launch['benchmark']==str(benchmark) and auth['host_E11D_pilot']==launch['pilot']==str(pilot)
            and auth['output_directory']=='/outputs/quadrature' and auth['selected_gpu']==launch['gpu']=='0'
            and r['backend']==identity() and auth['image_id']==identity()['image_id'], 'Execution/approval identity mismatch')
    require(r['source_files']==freeze['files'] and r['grids']==freeze['grids']==list(GRIDS) and
            r['limits']==freeze['limits']==auth['limits']==LIMITS and r['cases']==list(CASES)
            and r['camera_id']=='58' and r['probes']==84 and r['lighting']==freeze['lighting'], 'Scientific definition mismatch')
    for name,sha in r['source_files'].items():require(digest(ROOT/name)==sha,'Frozen source changed: '+name)
    for stage in PINS:baseline(stage,PINS[stage])
    labels,wo,pixels,mis,mis_coarse,unused=inputs(benchmark,pilot)
    for item in r['inputs']:
        p=Path(item['path']);require(p.parts[:2]==('/','inputs'),'Unexpected native input path')
        host=(benchmark/p.relative_to('/inputs/v3') if p.parts[:3]==('/','inputs','v3')
              else pilot/p.relative_to('/inputs/e11d'))
        require(digest(host)==item['sha256'] and host.stat().st_size==item['bytes'],'Consumed input changed')
    expected={'comparison.json','directions-256.npz','directions-512.npz','grid-256.npz','grid-512.npz'}
    require({x['path'] for x in r['outputs']}==expected and len(r['outputs'])==5,'Scientific artifact inventory mismatch')
    for item in r['outputs']:
        p=root/item['path'];require(p.resolve()==p and root in p.parents and digest(p)==item['sha256']
                                  and p.stat().st_size==item['bytes'],'Scientific output changed')
    actual_files={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual_files==expected|{'quadrature.json','progress.json'} and
            all(not p.is_symlink() for p in root.rglob('*')),'Unexpected output payload/symlink')
    replay_equal(controls(),r['synthetic_controls'])
    grids={};visibility=[]
    for size in GRIDS:
        native=next(x for x in r['completed_grids'] if x['size']==size)
        require(native['directions_per_point']==size*size and native['max_world_unit_error']<=2e-12 and
                native['max_world_cosine_error']<=2e-12,'World quadrature direction gate failed')
        with np.load(root/('directions-'+str(size)+'.npz'),allow_pickle=False) as z:verify_nodes(size,{k:z[k] for k in z.files},native)
        with np.load(root/('grid-'+str(size)+'.npz'),allow_pickle=False) as z:
            data={k:z[k].copy() for k in z.files};visibility.append(verify_grid(size,data,pixels,native));grids[size]=data['components']
    coarse,fine=grids[256],grids[512];comparison=compare(coarse,fine,mis,mis_coarse)
    replay_equal(comparison,r['comparison']);replay_equal(comparison,json.loads((root/'comparison.json').read_text()))
    require(r['counters']=={'shadow_attempts':27525120,'shadow_returns':27525120,'brdf_attempts':220200960,'brdf_returns':220200960}
            and all(r[k]==0 for k in ('native_model_calls','native_shader_calls','optimizer_returns','new_primary_rays')),
            'Work counters differ from authorized study')
    require(r['elapsed_seconds']<900 and r['host_peak_rss_bytes']<=64*1024**3 and
            sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())<=LIMITS['output_bytes'],'Recorded resource ceiling exceeded')
    rows,q,m=contrast_stability(coarse,fine,mis,mis_coarse);probe_rows=[]
    for j,pixel in enumerate(pixels):
        row=dict(pixel_index=int(pixel),x=int(pixel%128),y=int(pixel//128))
        for name in q:
            for label,fields in [('quadrature',q),('MIS',m)]:
                for key in ('amplitude','grid_difference','code_difference','resolved'):
                    a=fields[name][key];row[name+'_'+label+'_'+key]=(bool(a[j]) if a.dtype==np.bool_ else int(a[j]) if a.dtype.kind in 'iu' else float(a[j]))
        probe_rows.append(row)
    error=abs(fine[:,0,2]-coarse[:,0,2]).max(1);j=int(error.argmax())
    outlier=dict(pixel_index=int(pixels[j]),x=int(pixels[j]%128),y=int(pixels[j]//128),truth_grid_max_abs=float(error[j]),
                 truth_diffuse_error=float(abs(fine[j,0,0]-coarse[j,0,0]).max()),
                 truth_specular_error=float(abs(fine[j,0,1]-coarse[j,0,1]).max()))
    output.mkdir()
    result=dict(schema_version=1,protocol='bell-quadrature-e11e-offline-analysis-v1',verification_status='pass',
        followup_all_cases_two_grid_converged=comparison['all_cases_two_grid_converged'],
        original_E11D_admission_changed=False,original_E11D_recommended=None,
        native_report=record(report_path),freeze=record(freeze_path),authorization=record(auth_path),
        execution_receipts=[record(batch/p) for p in ('launch.json','exit-code.txt','console.log','container.cid')],
        native_outputs=r['outputs'],source_files=r['source_files'],runtime=r['runtime'],backend=r['backend'],
        counters=r['counters'],elapsed_seconds=r['elapsed_seconds'],
        memory={k:r[k] for k in ('host_peak_rss_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')},
        visibility=visibility,comparison=comparison,response_stability=rows,maximum_truth_grid_error_probe=outlier,
        diffuse_identical_across_cases=True,scientific_reexecution=False,new_BRDF_queries=0,new_GPU_queries=0,
        limits_of_conclusion='Conditional84-probe two-grid/method agreement; no absolute-accuracy, statistical-bias, full-scene or recovery proof.')
    write_json(output/'analysis.json',result);write_csv(output/'response_summary.csv',rows);write_csv(output/'probe_responses.csv',probe_rows)
    case_rows=[]
    for case in CASES:
        gate=comparison['two_grid_convergence'][case];v=comparison['comparison'][case]
        case_rows.append(dict(case=case,**gate,quad512_minus_MIS16384_relative=v['fine_quadrature_minus_MIS_fine']['relative_l2'],
                             quad512_minus_MIS4096_relative=v['fine_quadrature_minus_MIS_coarse']['relative_l2']))
    write_csv(output/'convergence_summary.csv',case_rows)
    figures(output,pixels,rows,q,m,comparison)
    return result


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def figures(output,pixels,rows,q,m,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='e11e-analysis-v1'
    factors=list(q);fig,ax=plt.subplots(2,2,figsize=(8,7))
    for a,name in zip(ax.ravel(),factors):
        old,new=m[name]['amplitude'],q[name]['amplitude'];maximum=max(old.max(),new.max())*1.03
        a.scatter(old,new,s=12);a.plot([0,maximum],[0,maximum],color='grey',linewidth=.8)
        a.axhline(.003,color='red',linestyle=':',linewidth=.8);a.axvline(.003,color='red',linestyle=':',linewidth=.8)
        a.set_title(name);a.set_xlabel('MIS16384 response');a.set_ylabel('Midpoint512² response');a.set_xlim(0,maximum);a.set_ylim(0,maximum)
    fig.suptitle('Same84 probes: max-channel finite perturbation response (linear RGB)');fig.tight_layout();save(fig,output,'response_agreement')
    fig,ax=plt.subplots(1,2,figsize=(9,3.7));x=np.arange(8)
    a=np.array([comparison['two_grid_convergence'][case]['relative_l2'] for case in CASES])
    b=np.array([comparison['comparison'][case]['fine_quadrature_minus_MIS_fine']['relative_l2'] for case in CASES])
    c=np.array([comparison['comparison'][case]['fine_quadrature_minus_MIS_coarse']['relative_l2'] for case in CASES])
    ax[0].plot(x,a,'o-',label='512² versus256²');ax[0].plot(x,b,'o-',label='512² versus MIS16384');ax[0].plot(x,c,'o-',label='512² versus MIS4096')
    ax[0].axhline(.002,color='red',linestyle=':',label='Frozen convergence bound');ax[0].set_xticks(x);ax[0].set_xticklabels(CASES,rotation=65,fontsize=7)
    ax[0].set_ylabel('Relative L2 (explicit comparison denominator)');ax[0].legend(fontsize=7)
    for label,fields in [('MIS16384',m),('Midpoint512²',q)]:
        ax[1].plot(range(4),[np.median(fields[name]['grid_difference']) for name in factors],'o-',label=label)
    ax[1].set_xticks(range(4));ax[1].set_xticklabels(factors);ax[1].set_yscale('log');ax[1].set_ylabel('Median local two-grid RGB difference');ax[1].legend(fontsize=8)
    fig.tight_layout();save(fig,output,'reference_accuracy')
    fig,axes=plt.subplots(2,4,figsize=(11,5))
    for col,name in enumerate(factors):
        maximum=max(q[name]['amplitude'].max(),m[name]['amplitude'].max())
        for row,(label,fields) in enumerate([('MIS16384',m),('512² midpoint',q)]):
            a=axes[row,col];v=fields[name];im=a.scatter(pixels%128,pixels//128,c=v['amplitude'],vmin=0,vmax=maximum,s=13,cmap='magma')
            ok=v['resolved'];a.scatter(pixels[ok]%128,pixels[ok]//128,s=24,facecolors='none',edgecolors='cyan',linewidths=.6)
            a.set_xlim(-.5,127.5);a.set_ylim(127.5,-.5);a.set_aspect('equal');a.axis('off');a.set_title(label+' / '+name,fontsize=9)
        fig.colorbar(im,ax=axes[:,col],shrink=.6)
    fig.suptitle('Camera58 only: actual probe positions; cyan=unchanged local gates, no interpolation')
    save(fig,output,'probe_sensitivity');plt.close('all')


def save(fig,output,name):
    save_figure(fig,output/(name+'.svg'));fig.savefig(output/(name+'.png'),bbox_inches='tight',dpi=130);__import__('matplotlib.pyplot',fromlist=['close']).close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','benchmark','pilot','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=analyze(a.batch,a.benchmark,a.pilot,a.output)
    print(json.dumps({'verification':r['verification_status'],'all_eight_converged':r['followup_all_cases_two_grid_converged'],
                     'response_stability':r['response_stability'],'max_grid_error_probe':r['maximum_truth_grid_error_probe']},indent=2))


if __name__=='__main__':main()
