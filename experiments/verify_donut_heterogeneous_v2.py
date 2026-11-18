"""File-only replay of all v2 decisions and outputs; no dataset admission."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_heterogeneous_v1 as v1
from benchmarks import donut_heterogeneous_v2 as d
from benchmarks.bell_e11 import linear_to_srgb
from benchmarks.donut_recovery_v1 import json_new, numeric
from benchmarks.generate_donut_recovery_v1 import highlight_record
from experiments.finalize_donut_reference_v2 import metric_replay
from experiments.generate_donut_heterogeneous_v1 import compare, ordinary_probes
from infrastructure.signatures import require
from infrastructure.stage_sources import digest
from infrastructure.verified_ray_backend import identity


def replay_gates(actual, saved):
    require(set(actual)==set(saved), 'Gate inventory differs')
    return [metric_replay(g,saved[k]) for k,g in actual.items()]


def view_identity(view, expected):
    for key,w in expected.items():
        v = view[key]
        require(v.dtype==w.dtype and v.shape==w.shape, 'Mismatched per-observation schema')
        if key in v1.GEOMETRY_KEYS or w.dtype.kind!='f':
            np.testing.assert_array_equal(v,w)
        else:
            np.testing.assert_allclose(v,w,atol=2e-12,rtol=0)


def validate_physical_view(view, resolutions):
    i = view['shader_valid']
    for name,grid in [('visible_counts',resolutions['fine_grid']),
                      ('coarse_visible_counts',resolutions['coarse_grid'])]:
        counts = view[name] if name=='visible_counts' else resolutions[name]
        missing = -1 if name=='coarse_visible_counts' else 0
        require(counts.shape==i.shape and counts.dtype.kind in 'iu'
                and ((counts[i]>=missing)&(counts[i]<=grid[i]**2)).all()
                and (counts[~i]==0).all(), 'Invalid count for recorded angular resolution')
    require(all(np.array_equal(view[k][:,0]+view[k][:,1],view[k][:,2])
                and (view[k]>=0).all() and (view[k][~i]==0).all()
                for k in ('linear_components','coarse_components')),
            'Nonphysical or inconsistent RGB components')
    require(np.array_equal(view['reflectance_diffuse'][i],view['color'][i]*(1-view['metallic'][i]))
            and np.max(abs(view['reflectance']-view['reflectance_diffuse']-view['reflectance_specular']))<=2e-15
            and all((view[k]>=0).all() and (view[k][~i]==0).all()
                    for k in ('reflectance','coarse_reflectance','incident_reflectance','coarse_incident_reflectance')),
            'Nonphysical or inconsistent reflectance components')


def verify(sha, destination):
    from PIL import Image

    f=d.freeze(d.ROOT/d.FREEZE,sha)
    batch=Path(f['host_output'])
    prior=Path(f['retained_failed_batch'])
    require(destination.resolve()==destination and not destination.exists() and destination.parent.is_dir()
            and all(p!=destination and p not in destination.parents for p in (batch,prior)),
            'Fresh separate verification file required')
    paths=tuple(Path(f['host_inputs'][k]) for k in ('oracle','cameras','reference','admission'))
    _,labels,radius,old=d.source(f,prior,paths)
    out=batch/'reference'
    authority=batch/'authorization.json'
    a=v1.read(authority)
    require(a==d.authorization(f,a['commit'],sha) and digest(batch/'freeze.json')==sha
            and digest(authority)==digest(batch.with_name(batch.name+'-authorization.json'))
            and (batch/'exit-code.txt').read_text().strip()=='0','Invalid v2 launch evidence')
    path=out/'generation.json'
    r=v1.read(path)
    require(r['status']=='complete_pending_researcher_admission' and r['scientific_work_complete']
            and r['cleanup_verified'] and r['training_authorized'] is False
            and r['optimizer_returns']==r['native_model_calls']==0 and r['protocol']==d.PROTOCOL
            and r['freeze_sha256']==sha and r['commit']==a['commit'] and r['authorization_sha256']==digest(authority)
            and r['backend']==identity() and r['image']==v1.IMAGE
            and r['retained_failed_generation_sha256']==a['retained_failed_generation_sha256']
            and r['v1_freeze_sha256']==d.V1_SHA and r['panel']==v1.PANEL and r['splits']==v1.SPLITS
            and r['limits']==d.LIMITS and r['tolerances']==v1.TOLERANCES and r['refinement']==d.POLICY
            and r['ring_radius_native']==radius and r['environment']==v1.ENVIRONMENT and r['field']==v1.FIELD
            and r['BRDF']==v1.BRDF and r['indirect']=='zero' and r['point_light']=='absent'
            and r['wall_seconds']<d.LIMITS['inner_seconds']
            and r['host_peak_RSS_bytes']<d.LIMITS['host_memory_gib']*1024**3
            and sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())<d.LIMITS['output_bytes'],
            'Terminal v2 identity or resource conditions differ')
    require([row['camera'] for row in r['views']]==v1.PANEL,'Incomplete or duplicated final panel')

    expected={'surface-area.npz','reference-components.png','camera50-high-grid-diagnostic.npz',
              *(f'view-{c}.npz' for c in v1.PANEL),*(f'quadrature-{c}.npz' for c in v1.PANEL),
              *(f'baseline-{c}.npz' for c in v1.PANEL if c!=0),
              *(f'view-{c}-{tag}.png' for c in v1.PANEL for tag in ('RGB','diffuse','specular'))}
    for row in r['views']:
        expected.update(f'patch-{row["camera"]}-{pix}.npz' for pix in row['selected_pixels'])
    require(set(r['outputs'])==expected and expected=={p.name for p in out.iterdir()
             if p.is_file() and p.name not in ('progress.json','progress.next','generation.json')},'Unexpected v2 output inventory')
    for name,item in r['outputs'].items():
        v1.check_artifact(out/name,item)
    for new,original in [('view-0.npz','view-0.npz'),('baseline-50.npz','view-50.npz'),
                         ('surface-area.npz','surface-area.npz'),
                         *[(f'view-0-{tag}.png',f'view-0-{tag}.png') for tag in ('RGB','diffuse','specular')]]:
        require(digest(out/new)==digest(prior/'reference'/original),'Archived byte-identical v1 artifact differs: '+new)

    diagnostic=numeric(out/'camera50-high-grid-diagnostic.npz')
    gates=d.diagnostic_gates(diagnostic)
    require(gates['RGB']['passed'],'High-grid diagnostic did not converge on replay')
    replay_gates(gates,r['diagnostic'])
    selected=0
    new_high=0
    summary=[]
    sheet=Image.new('RGB',(3*v1.SIZE,len(v1.PANEL)*v1.SIZE))
    for row,s in enumerate(r['views']):
        camera=s['camera']
        baseline=old[camera] if camera in d.REUSED else numeric(out/f'baseline-{camera}.npz')
        cohort=d.reused_coarse_counts(baseline) if camera in d.REUSED else baseline.pop('baseline_coarse_visible_counts')
        view_identity(baseline,labels[camera])
        before=compare(baseline)
        require(all(g['passed'] for k,g in before.items() if k!='radiance_convergence')
                and (camera!=0 or before['radiance_convergence']['passed']),'Baseline reflectance or camera0 gate differs')
        selected_here=[]

        def retrieve(i):
            nonlocal selected,new_high
            pixel=int(baseline['pixel_indices'][i])
            require(selected<d.MAX_TOTAL and s['selected_pixels'][len(selected_here):len(selected_here)+1]==[pixel],
                    'Saved refinement order differs from replay or panel budget')
            patch=numeric(out/f'patch-{camera}-{pixel}.npz')
            if camera==50 and pixel in d.PROBE_PIXELS:
                for k,v in d.probe_slice(diagnostic,pixel).items():
                    np.testing.assert_array_equal(patch[k],v)
            else:
                new_high+=1
            selected+=1
            selected_here.append(pixel)
            return patch

        reconstructed,quadrature=d.assemble(baseline,cohort,retrieve)
        require(selected_here==s['selected_pixels'],'Unused or missing high-grid patches')
        valid=reconstructed['shader_valid']
        _,highlights=highlight_record(reconstructed['linear_components'][valid])
        mask=np.zeros(len(valid),dtype=bool)
        mask[valid]=highlights
        reconstructed['highlight_mask']=mask
        v=numeric(out/f'view-{camera}.npz')
        q=numeric(out/f'quadrature-{camera}.npz')
        require(v.keys()==reconstructed.keys() and q.keys()==quadrature.keys(),'Assembled array inventory differs')
        for actual,replay in ((v,reconstructed),(q,quadrature)):
            for key, expected_array in replay.items():
                require(actual[key].dtype==expected_array.dtype,'Assembled array dtype differs')
                np.testing.assert_array_equal(actual[key],expected_array)
        validate_physical_view(v,q)
        final=compare(v)
        replay_gates(final,{k:s[k] for k in final})
        require(all(g['passed'] for g in final.values()) and s['valid']==int(valid.sum())
                and s['ordinary_probe_pixels']==v['pixel_indices'][ordinary_probes(v)].tolist(),
                'Final scientific admission gate differs')
        for column,(index,tag) in enumerate(((2,'RGB'),(0,'diffuse'),(1,'specular'))):
            img=np.zeros((v1.SIZE**2,3))
            img[v['pixel_indices']]=np.clip(v['linear_components'][:,index],0,1)
            with Image.open(out/f'view-{camera}-{tag}.png') as saved:
                np.testing.assert_array_equal(np.array(saved),linear_to_srgb(img.reshape(v1.SIZE,v1.SIZE,3)))
                sheet.paste(saved,(column*v1.SIZE,row*v1.SIZE))
        summary.append(dict(camera=camera,valid=int(valid.sum()),selected_pixels=selected_here,
                original_RGB=before['radiance_convergence'],final_gates=final,
                highlight_count=int(mask.sum()),near_boundary=int(v['near_boundary'][valid].sum()),
                nearly_isotropic=int(v['nearly_isotropic'][valid].sum())))
    with Image.open(out/'reference-components.png') as saved:
        np.testing.assert_array_equal(np.array(saved),np.array(sheet))
    require(r['new_high_points']==new_high and r['refined_points']==selected
            and r['counters']==d.counters(new_high),'Scientific charge counters differ from recorded selections')
    json_new(destination,dict(schema_version=1,protocol=d.PROTOCOL,status='verified_pending_researcher_admission',
        native_commit=r['commit'],freeze_sha256=sha,authorization_sha256=digest(authority),
        generation_sha256=digest(path),package=str(out),retained_failed_batch=str(prior),
        retained_failed_records=f['failed_records'],outputs=r['outputs'],counters=r['counters'],
        per_view=summary,diagnostic=gates,refined_points=selected,new_high_points=new_high,
        camera0_byte_identity_verified=True,training_authorized=False,researcher_admitted_dataset=False))
    return len(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(f'Independently verified {verify(a.freeze_sha256,a.output)} heterogeneous v2 reference views')
