"""File-only verification and paper figures for the completed v2 reference.

Never calls a shader/integrator, starts Docker, traces rays or queries a model.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from benchmarks.donut_recovery_v1 import (ROOT,PANEL,SPLITS,COLOR,METALLIC,WIDTHS,ENVIRONMENT,
    numeric,json_new,inputs,verify_freeze)
from benchmarks.donut_reference_v2 import PROTOCOL,LIMITS,previous,GRAZING_PIXELS
from benchmarks.bell_e11 import convergence
from infrastructure.stage_sources import digest
from infrastructure.signatures import require
from infrastructure.verified_ray_backend import identity as backend_identity

COMMIT='dbd60d9f45eeb04a32b3f670d56797a934a417ec'
FREEZE='1c0af2688f36bef6e4da7e15666e0245a76c9a11294a3f719bfe5601c53b7730'
AUTH='060452975b60efe770522abd7271514dc55c15346dce9952a3ed495af9e76ede'


def metric_replay(actual,saved):
    require(set(actual)==set(saved) and actual['passed'] is saved['passed'], 'Scientific acceptance decision differs')
    # Summation replay guard only; both decisions are independently evaluated
    # against the unchanged .003/.002 scientific thresholds, without this guard.
    guard=32*np.finfo(np.float64).eps
    delta={k:abs(actual[k]-saved[k]) for k in ('maximum_rgb_difference','relative_l2')}
    require(all(np.isfinite(saved[k]) and delta[k]<=guard*max(1,abs(saved[k])) for k in delta), 'Saved metric differs beyond float64 summation replay')
    return delta


def finalize(batch,prior,destination):
    batch=batch.resolve(strict=True);prior=prior.resolve(strict=True);destination=destination.resolve()
    require(ROOT/'docs/results' in destination.parents and not destination.exists(),'Fresh tracked small-results destination required')
    frozen=verify_freeze(batch/'freeze.json',FREEZE)
    require(frozen['host_reference_output']==str(batch) and frozen['previous_host_reference']==str(prior),'Explicit experiment identities differ')
    require(digest(batch/'authorization.json')==AUTH,'Execution authorization differs')
    auth=json.loads((batch/'authorization.json').read_text())
    require(auth['approved'] is True and auth['commit']==COMMIT and auth['freeze_sha256']==FREEZE and auth['protocol']==PROTOCOL
            and auth['limits']==LIMITS,'Exact executed approval binding differs')
    package=batch/'reference';report=json.loads((package/'generation.json').read_text())
    require((batch/'reference-exit-code.txt').read_text().strip()=='0' and report['status']=='complete_pending_researcher_admission' and
        report['scientific_work_complete'] and report['cleanup_verified'] and report['commit']==COMMIT and report['freeze_sha256']==FREEZE,
        'Incomplete execution or cleanup')
    require(report['backend']==backend_identity() and report['splits']==SPLITS and report['panel']==PANEL and
        report['declared_image_id']==auth['image_id'] and report['environment']==ENVIRONMENT and
        report['BRDF']=='Kurt-a0-metallic-additive-v1' and report['indirect']=='zero' and report['point_light']=='absent' and
        report['native_model_calls']==report['optimizer_returns']==report['Blender_calls']==0,'Scientific definition differs')
    for name,item in report['outputs'].items():
        p=package/name
        require(package in p.resolve().parents and p.resolve()==p and p.stat().st_size==item['bytes'] and digest(p)==item['sha256'], 'Output artifact differs: '+name)
    old,evidence=previous(prior)
    _,meta,input_records=inputs(Path(frozen['inputs']['oracle-geometry-and-frames.npz']['path']).parent,
                               Path(frozen['inputs']['transforms_train.json']['path']))
    views={};rows=[];replay=[];total=0;new=0
    for row in report['views']:
        camera=row['camera'];d=numeric(package/('view-%d.npz'%camera));views[camera]=d
        ids=d['pixel_indices'];valid=d['shader_valid'];n=d['normals'];a=d['axis'];wo=d['outgoing'];q=d['Q_world']
        require(d['hit_mask'].shape==(128,128) and d['hit_mask'].dtype==bool and ids.dtype.kind in 'iu' and
            np.array_equal(np.flatnonzero(d['hit_mask'].reshape(-1)),ids) and valid.dtype==bool and valid.shape==ids.shape,
            'Invalid observation/mask ordering')
        require(np.array_equal(valid,(n*wo).sum(-1)>0),'One-sided geometry domain changed')
        require(all(x.shape==(len(ids),3) and abs(np.linalg.norm(x,axis=1)-1).max()<2e-12 for x in (n,a,wo)), 'Invalid unit frame')
        require(abs((n*a).sum(-1)).max()<2e-12 and ((n*d['face_normal']).sum(-1)>0).all(), 'Invalid normal/director convention')
        require(np.array_equal(d['color'],np.broadcast_to(COLOR,(len(ids),3))) and np.array_equal(d['metallic'],np.full((len(ids),1),METALLIC)) and
            np.array_equal(d['widths_major_minor'],np.broadcast_to(WIDTHS,(len(ids),2))),'Material labels changed')
        b=np.cross(n,a);b/=np.linalg.norm(b,axis=1,keepdims=True)
        expected=WIDTHS[0]**2*a[:,:,None]*a[:,None,:]+WIDTHS[1]**2*b[:,:,None]*b[:,None,:]
        Q_error=float(abs(q-expected).max());require(Q_error<2e-12,'World-Q labels disagree with physical axes')
        origin=np.array(meta['frames'][camera]['transform_matrix'])[:3,3]/2
        position_error=float(abs(d['position']-(origin-d['depth'][:,None]*wo)).max())
        require(position_error<=2e-5 and d['surface_residual'].max()<=2e-6,'Saved position/primary-ray/surface mismatch')
        require(np.array_equal(d['linear_components'][:,0]+d['linear_components'][:,1],d['linear_components'][:,2]),'RGB component identity differs')
        require((d['linear_components']>=0).all() and (d['reflectance']>=0).all(),'Negative physical references')
        image=convergence(d['linear_components'][valid,2],d['coarse_components'][valid,2])
        energy=convergence(d['reflectance'][valid],d['coarse_reflectance'][valid])
        replay.extend([metric_replay(image,row['radiance_convergence']),metric_replay(energy,row['energy_convergence'])])
        require(image['passed'] and energy['passed'],'Recomputed scientific convergence fails')
        require(np.array_equal(d['reflectance_diffuse'][valid],d['color'][valid]*(1-d['metallic'][valid])) and
            abs(d['reflectance']-d['reflectance_diffuse']-d['reflectance_specular']).max()<2e-15,'Analytic/component reflectance mismatch')
        require(row['geometry_hits']==len(ids) and row['shader_valid']==int(valid.sum()),'Recorded point counts differ')
        total+=int(valid.sum());new+=int(valid.sum()) if camera!=0 else 0
        rows.append(dict(camera=camera,split='fitting' if camera in SPLITS['fitting'] else 'held_out',geometry_hits=len(ids),valid=int(valid.sum()),
            image_max_difference=image['maximum_rgb_difference'],image_relative_L2=image['relative_l2'],
            reflectance_max_difference=energy['maximum_rgb_difference'],reflectance_relative_L2=energy['relative_l2'],
            reflectance_max=float(d['reflectance'].max()),reflectance_channels_above_one=int((d['reflectance']>1).sum()),
            highlight_fraction=row['highlight_diagnostics']['highlight_coverage'],specular_fraction=row['highlight_diagnostics']['integrated_specular_fraction'],
            linear_RGB_max=float(d['linear_components'][:,2].max()),world_Q_max_error=Q_error,primary_position_max_error=position_error))
    require(list(views)==PANEL and len(views)==6,'Incomplete/duplicated camera panel')
    for name in report['camera0_reuse']['unchanged_array_names']:
        x=views[0][name];y=old[name]
        require(x.dtype==y.dtype and x.shape==y.shape and x.tobytes(order='C')==y.tobytes(order='C'),'Camera0 reused array differs: '+name)
    for suffix in ('preview','diffuse-preview','specular-preview'):
        name='view-0-'+suffix+'.png';require(digest(package/name)==digest(prior/'reference'/name),'Camera0 preview changed')
    for name in ('grazing','ordinary'):
        v=numeric(package/(name+'-validation.npz'));ids=v['pixel_indices']
        expected_pixels=frozen['grazing_pixels' if name=='grazing' else 'ordinary_pixels']
        require(ids.tolist()==expected_pixels,'Validation correspondence changed')
        indices=np.searchsorted(views[0]['pixel_indices'],ids)
        require(np.array_equal(views[0]['pixel_indices'][indices],ids) and
            np.array_equal(v['slope_fine'],views[0]['reflectance'][indices]) and
            np.array_equal(v['slope_coarse'],views[0]['coarse_reflectance'][indices]) and
            np.array_equal(v['incident_old_fine'],old['reflectance'][indices]) and
            np.array_equal(v['incident_old_coarse'],old['coarse_reflectance'][indices]),'Cached validation values differ')
        actual={'slope_two_grid':convergence(v['slope_fine'],v['slope_coarse'])}
        if name=='ordinary':actual.update(old_incident_two_grid=convergence(v['incident_old_fine'],v['incident_old_coarse']),
            slope_vs_independent_incident=convergence(v['slope_fine'],v['incident_old_fine']))
        for key,value in actual.items():
            require(value['passed'],'Validation gate fails on replay');replay.append(metric_replay(value,report['validation'][name][key]))
    counters=report['counters']
    require(counters==dict(primary_rays=5*128**2,CPU_primary_crosschecks=40,
        shadow_directions=new*327680,BRDF_point_directions=new*327680,
        slope_nodes_attempted=total*81920,slope_nodes_returned=total*81920),'Scientific work totals differ')
    artifacts={p.relative_to(batch).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(batch.rglob('*')) if p.is_file() and 'runtime' not in p.relative_to(batch).parts}
    size=sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())
    require(size<LIMITS['output_bytes'] and report['elapsed_seconds']<LIMITS['inner_seconds'] and report['host_peak_RSS_bytes']<64*1024**3,'Recorded resources exceed limits')
    result=dict(schema_version=1,status='TECHNICAL_VERIFICATION_PASS_PENDING_RESEARCHER_ADMISSION',
        researcher_admitted_dataset=False,source_commit=COMMIT,freeze_sha256=FREEZE,authorization_sha256=AUTH,
        generation_sha256=digest(package/'generation.json'),package_root=str(package),batch_root=str(batch),
        runtime=report['runtime'],backend=report['backend'],validation=report['validation'],camera0_reuse=report['camera0_reuse'],
        per_view=rows,counters=counters,valid_observations=total,geometry_hits=sum(r['geometry_hits'] for r in rows),
        fitting_observations=sum(r['valid'] for r in rows if r['split']=='fitting'),held_out_observations=sum(r['valid'] for r in rows if r['split']=='held_out'),
        elapsed_seconds=report['elapsed_seconds'],host_peak_RSS_bytes=report['host_peak_RSS_bytes'],output_bytes=size,
        GPU_peak_allocated_bytes=report['GPU_peak_allocated_bytes'],GPU_peak_reserved_bytes=report['GPU_peak_reserved_bytes'],
        native_exit_code=0,cleanup_verified_from_saved_receipt=True,Docker_cleanup_independently_queried=False,
        maximum_metric_replay_roundoff=max(v for d in replay for v in d.values()),
        scientific_thresholds_unchanged=True,original_failed_evidence_unchanged=True,input_artifacts_verified=input_records,
        new_reference_or_model_queries_during_finalization=0,training_authorized=False)
    destination.mkdir(parents=True)
    json_new(destination/'verification.json',result)
    with (destination/'view-metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copyfile(package/'reference-RGB-diffuse-specular.png',destination/'reference-RGB-diffuse-specular.png')
    figure(package,prior,destination,rows)
    json_new(destination/'artifact-manifest.json',dict(schema_version=1,external_artifacts=artifacts,batch_root=str(batch),
        prior_batch_root=str(prior),prior_artifacts=evidence['original_artifacts'],freeze_sha256=FREEZE,
        finalization_code=dict(path=str(Path(__file__)),sha256=digest(Path(__file__))),
        tracked_artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in destination.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=result['status'],generation_sha256=result['generation_sha256'],observations=total,
                         output_bytes=size,metric_replay_roundoff=result['maximum_metric_replay_roundoff'])))


def figure(package,prior,destination,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    v=numeric(package/'grazing-validation.npz')
    old_error=abs(v['incident_old_fine']-v['incident_old_coarse']).max(1)
    new_error=abs(v['slope_fine']-v['slope_coarse']).max(1)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    x=np.arange(5)
    axes[0].bar(x-.18,old_error,.36,label='Incident 256² vs512²')
    axes[0].bar(x+.18,new_error,.36,label='Slopes 64 vs128')
    axes[0].set_xticks(x);axes[0].set_xticklabels(['(%d,%d)'%(i%128,i//128) for i in v['pixel_indices']])
    axes[0].set_xlabel('Five original grazing pixels');axes[0].set_title('Unoccluded reflectance convergence')
    x=np.arange(6)
    axes[1].bar(x-.18,[r['image_max_difference'] for r in rows],.36,label='RGB image')
    axes[1].bar(x+.18,[r['reflectance_max_difference'] for r in rows],.36,label='Slope reflectance')
    axes[1].set_xticks(x);axes[1].set_xticklabels([str(r['camera']) for r in rows]);axes[1].set_xlabel('Frozen camera ID')
    axes[1].set_title('Full-view convergence')
    for ax in axes:
        ax.set_yscale('log');ax.axhline(.003,color='crimson',ls='--',label='Frozen absolute bound0.003')
        ax.set_ylabel('Maximum channel difference');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(destination/'convergence.png',dpi=180);fig.savefig(destination/'convergence.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','prior','destination'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();finalize(a.batch,a.prior,a.destination)
