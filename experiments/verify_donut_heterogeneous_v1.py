"""File-only replay of heterogeneous Donut generation, without issuing admission."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_heterogeneous_v1 as d
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.finalize_donut_reference_v2 import metric_replay
from experiments.generate_donut_heterogeneous_v1 import compare, ordinary_probes
from infrastructure.signatures import require
from infrastructure.stage_sources import digest
from infrastructure.verified_ray_backend import identity


def verify(sha, output):
    f = d.freeze(d.ROOT/d.FREEZE,sha)
    batch = Path(f['host_output'])
    require(output.resolve()==output and not output.exists() and batch not in output.parents
            and output.parent.is_dir(), 'Fresh separate verification path required')
    out = batch/'reference'
    auth = batch/'authorization.json'
    a = d.read(auth)
    require(digest(batch/'freeze.json')==sha
            and a==d.authorization(f,a['commit'],sha)
            and digest(auth)==digest(batch.with_name(batch.name+'-authorization.json'))
            and (batch/'exit-code.txt').read_text().strip()=='0', 'Invalid generation launch evidence')
    path = out/'generation.json'
    r = d.read(path)
    require(r['status']=='complete_pending_researcher_admission' and r['scientific_work_complete']
            and r['cleanup_verified'] and r['backend']==identity() and r['commit']==a['commit']
            and r['freeze_sha256']==sha and r['authorization_sha256']==digest(auth)
            and r['native_model_calls']==r['optimizer_returns']==0
            and r['training_authorized'] is False and r['panel']==d.PANEL and r['splits']==d.SPLITS
            and r['source_generation_sha256']==d.SOURCE_REPORT and r['source_admission_sha256']==d.SOURCE_ADMISSION
            and r['source_view_records']==f['geometry_view_records']
            and r['field']==d.FIELD and r['environment']==d.ENVIRONMENT and r['image']==d.IMAGE
            and r['BRDF']==d.BRDF and r['indirect']=='zero' and r['point_light']=='absent'
            and r['tolerances']==d.TOLERANCES and r['grids']==list(d.GRIDS) and r['slope_orders']==list(d.ORDERS),
            'Terminal generation receipt differs')
    require(r['counters']=={k:d.LIMITS[k] for k in ('primary_rays','CPU_primary_crosschecks',
                'shadow_directions','BRDF_point_directions','slope_nodes_attempted','slope_nodes_returned')},
            'Incomplete scientific work counters')
    expected = {'surface-area.npz','reference-components.png',
                *(f'view-{c}.npz' for c in d.PANEL),
                *(f'view-{c}-{component}.png' for c in d.PANEL for component in ('RGB','diffuse','specular'))}
    require(set(r['outputs'])==expected and [row['camera'] for row in r['views']]==d.PANEL,
            'Incomplete output or panel inventory')
    for name,item in r['outputs'].items():
        d.check_artifact(out/name,item)
    geometry,old = d.source(f,*(Path(f['host_inputs'][k]) for k in ('oracle','cameras','reference','admission')))
    fields,radius,areas=d.labels(geometry,old)
    np.testing.assert_array_equal(numeric(out/'surface-area.npz')['triangle_area_native'],areas)
    require(r['ring_radius_native']==radius, 'Geometry-derived radius differs')
    from benchmarks.generate_donut_recovery_v1 import highlight_record
    from benchmarks.bell_e11 import linear_to_srgb
    from PIL import Image

    summaries, views = [], {}
    for frozen_row in r['views']:
        c = frozen_row['camera']
        v = numeric(out/f'view-{c}.npz')
        views[c] = v
        for k,w in fields[c].items():
            require(v[k].shape==w.shape and v[k].dtype==w.dtype, 'Archived geometry or label shape changed')
            if k in d.GEOMETRY_KEYS or w.dtype.kind!='f':
                np.testing.assert_array_equal(v[k],w)
            else:
                np.testing.assert_allclose(v[k],w,atol=2e-12,rtol=0)
        i = v['shader_valid']
        require(all(np.array_equal(v[k][:,0]+v[k][:,1],v[k][:,2]) and (v[k][~i]==0).all()
                    for k in ('linear_components','coarse_components'))
                and np.array_equal(v['reflectance_diffuse'][i],v['color'][i]*(1-v['metallic'][i]))
                and np.max(abs(v['reflectance']-v['reflectance_diffuse']-v['reflectance_specular']))<=2e-15
                and ((v['visible_counts']>=0)&(v['visible_counts']<=d.GRIDS[-1]**2)).all(),
                'RGB, reflectance or visibility diagnostics differ')
        gates = compare(v)
        for k,g in gates.items():
            metric_replay(g,frozen_row[k])
        require(all(g['passed'] for g in gates.values()) and int(i.sum())==frozen_row['valid']
                and v['pixel_indices'][ordinary_probes(v)].tolist()==frozen_row['ordinary_probe_pixels'],
                'Replayed convergence or panel validity differs')
        _,h = highlight_record(v['linear_components'][i])
        np.testing.assert_array_equal(v['highlight_mask'][i],h)
        for slot,tag in enumerate(('diffuse','specular','RGB')):
            im = np.zeros((d.SIZE*d.SIZE,3))
            im[v['pixel_indices']] = np.clip(v['linear_components'][:,slot],0,1)
            with Image.open(out/f'view-{c}-{tag}.png') as saved:
                np.testing.assert_array_equal(np.array(saved),linear_to_srgb(im.reshape(d.SIZE,d.SIZE,3)))
        summaries.append(dict(camera=c,valid=int(i.sum()),near_boundary=int(v['near_boundary'][i].sum()),
                    nearly_isotropic=int(v['nearly_isotropic'][i].sum()),
                    region_counts=np.bincount(v['material_region'][i],minlength=2).tolist(),
                    above_one_RGB_channels=int((v['linear_components'][:,2]>1).sum()),**gates))
    split_coverage={}
    for name,cams in d.SPLITS.items():
        triangle=np.concatenate([views[c]['triangle'][views[c]['shader_valid']] for c in cams])
        weight,coverage=d.seen_area_weights(triangle,areas)
        split_coverage[name] = dict(observations=len(triangle),fraction_mesh_area_on_observed_triangles=coverage,
            area_fraction_with_reference_highlight=float(weight[np.concatenate([
                views[c]['highlight_mask'][views[c]['shader_valid']] for c in cams])].sum()/weight.sum()))
    json_new(output,dict(schema_version=1,protocol=d.PROTOCOL,status='verified_pending_researcher_admission',
        source_commit=r['commit'],freeze_sha256=sha,authorization_sha256=digest(auth),
        package=str(out),generation_sha256=digest(path),outputs=r['outputs'],
        counters=r['counters'],per_view=summaries,area_coverage=split_coverage,
        training_authorized=False))
    return len(summaries)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(f'Independently verified {verify(args.freeze_sha256,args.output)} heterogeneous Donut views')
