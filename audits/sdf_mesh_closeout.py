"""Offline localization of saved 3A-2 measurements, without model/source execution."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from infrastructure.signatures import require, read_report

PROTOCOL='geometry-sdf-mesh-closeout-v1'
H=2/127
REPORT_SHA='839f101d31d349e9b45bf6cfcf559681ea381cc013bb375f23930c6cdc8a8cb8'
ARRAY_SHA='522bf62ce7e02d2148866e35a6dec1fa86a16b73c681118809efdedc4309b66e'
FIELDS={'points','SDF_raw','base_masked_extraction_field','gradient','gradient_length','distance_proxy','inside_sphere',
        'float64_radius','sample_kind','probe_points','probe_SDF_raw','probe_masked_extraction_field','probe_inside_sphere',
        'probe_base_index','probe_direction_code','probe_offset_over_h','face_indices','vertex_indices','face_areas',
        'face_cross_normals','gradient_face_cosine'}


def quantiles(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return dict(zip(['min','p05','median','p95','max'],map(float,np.quantile(x,[0,.05,.5,.95,1])))) if x.size else None


def localize(a):
    """Reconstruct the FROZEN eligibility/criterion, never reinterpret its bounds."""
    n=len(a['points']);nf=len(a['face_indices']);nv=len(a['vertex_indices'])
    require(0<n<=2275 and nf<=1920 and nv<=256 and n+len(a['probe_points'])<=20000,'Invalid saved sample budget')
    require(a['points'].shape==(n,3) and a['gradient'].shape==(n,3),'Invalid base point/gradient shape')
    for name in ['SDF_raw','gradient_length','distance_proxy','inside_sphere','float64_radius','sample_kind']:
        require(a[name].shape==(n,),'Invalid saved base field: '+name)
    require(a['inside_sphere'].dtype==a['probe_inside_sphere'].dtype==np.bool_,'Sphere masks must be Boolean')
    require(a['face_areas'].shape==(nf,) and np.isfinite(a['face_areas']).all() and (a['face_areas']>0).all(),'Invalid face areas')
    require(np.array_equal(a['sample_kind'],np.array([0]*nf+[1]*nv+[2]*(n-nf-nv))),'Saved sample kinds/order differ')
    for name in ['points','gradient','SDF_raw','gradient_length','float64_radius','probe_points','probe_SDF_raw']:
        require(np.isfinite(a[name]).all(),'Nonfinite saved measurements: '+name)
    radius=np.linalg.norm(a['points'].astype(np.float64),axis=-1)
    lengths=np.linalg.norm(a['gradient'].astype(np.float64),axis=-1)
    require(np.allclose(radius,a['float64_radius'],rtol=0,atol=1e-12) and np.allclose(lengths,a['gradient_length'],rtol=0,atol=1e-12),'Saved derived norms differ')
    valid=lengths>1e-6;require(np.allclose(a['distance_proxy'][valid],np.abs(a['SDF_raw'][valid])/lengths[valid],rtol=0,atol=1e-12),'Saved distance proxies differ')
    count=len(a['probe_points'])
    require(a['probe_points'].shape==(count,3) and all(a[k].shape==(count,) for k in ['probe_base_index','probe_direction_code','probe_offset_over_h','probe_inside_sphere','probe_SDF_raw']),'Invalid probe fields')
    require(np.issubdtype(a['probe_base_index'].dtype,np.integer) and ((a['probe_base_index']>=0)&(a['probe_base_index']<n)).all(),'Invalid probe indices')
    keys=list(zip(map(int,a['probe_base_index']),map(int,a['probe_direction_code']),map(float,a['probe_offset_over_h'])))
    require(len(set(keys))==len(keys),'Duplicate probe bindings')
    lookup=dict(zip(keys,range(count)));eligible=valid & a['inside_sphere'];crossings={}
    for scale in (.25,.5):
        cross=np.zeros(n,bool)
        for i in range(n):
            if not valid[i]:continue
            require((i,0,scale) in lookup and (i,0,-scale) in lookup,'Missing saved gradient probe')
            plus,minus=[lookup[i,0,f] for f in (scale,-scale)]
            y,z=a['probe_SDF_raw'][[plus,minus]]
            cross[i]=(y>1e-7 and z<-1e-7) or (z>1e-7 and y<-1e-7)
            if scale==.5:eligible[i]&=a['probe_inside_sphere'][plus] and a['probe_inside_sphere'][minus]
        crossings[scale]=cross
    proxy_fail=a['distance_proxy']>H/2;cross_fail=~crossings[.5];failed=eligible & (proxy_fail|cross_fail)
    def group(s):
        return {'count':int(s.sum()),'radius':quantiles(radius[s]),'signed_raw_SDF':quantiles(a['SDF_raw'][s]),
            'absolute_raw_SDF':quantiles(abs(a['SDF_raw'][s])),'proxy_over_h':quantiles(a['distance_proxy'][s]/H),
            'face_area':quantiles(a['face_areas'][s[:nf]]),'tiny_faces':int((s[:nf]&(a['face_areas']<=1e-10)).sum()),
            'positive_raw_SDF':int((s&(a['SDF_raw']>0)).sum()),'negative_raw_SDF':int((s&(a['SDF_raw']<0)).sum())}
    rows=[]
    for i in np.flatnonzero(failed):
        kind=int(a['sample_kind'][i]);row={'sample_index':int(i),'kind':['face','vertex','ray'][kind],
            'source_index':int(a['face_indices'][i]) if kind==0 else (int(a['vertex_indices'][i-nf]) if kind==1 else None),
            'point':a['points'][i].tolist(),'radius':float(radius[i]),'sphere_distance_over_h':float((1-radius[i])/H),
            'raw_SDF':float(a['SDF_raw'][i]),'gradient_length':float(lengths[i]),'proxy_over_h':float(a['distance_proxy'][i]/H),
            'face_area':float(a['face_areas'][i]) if kind==0 else None,'proxy_failure':bool(proxy_fail[i]),
            'gradient_crossing_h4':bool(crossings[.25][i]),'gradient_crossing_h2':bool(crossings[.5][i]),
            'gradient_probes':{str(f):float(a['probe_SDF_raw'][lookup[i,0,f]]) for f in [.25,-.25,.5,-.5]}}
        rows.append(row)
    return {'eligible':int(eligible.sum()),'excluded':int((~eligible).sum()),'failures':int(failed.sum()),
        'failure_union':{'proxy':int((eligible&proxy_fail).sum()),'no_half_grid_crossing':int((eligible&cross_fail).sum()),
                         'both':int((eligible&proxy_fail&cross_fail).sum())},
        'groups':{'failed':group(failed),'passed_eligible':group(eligible&~failed),'excluded':group(~eligible)},
        'failure_kind_counts':{label:int((failed&(a['sample_kind']==k)).sum()) for k,label in enumerate(['face','vertex','ray'])},
        'radial_bands':{str(scale):{'failed':int((failed&(abs(radius-1)<=scale*H)).sum()),
                      'eligible':int((eligible&(abs(radius-1)<=scale*H)).sum())} for scale in [.25,.5,1,2,4]},
        'failed_samples':rows,'research_geometry_admitted':False,
        'interpretation':'Offline localization only; original failed/unresolved scientific statuses remain unchanged.'}


def analyze(report,report_sha,measurements,measurement_sha):
    require(report_sha==REPORT_SHA and measurement_sha==ARRAY_SHA,'Only the explicitly accepted 3A-2 evidence is selected')
    report=Path(report);measurements=Path(measurements)
    require(report.is_absolute() and measurements.is_absolute() and report.resolve()==report and measurements.resolve()==measurements,'Explicit unaliased absolute paths required')
    r,identity=read_report(report,report_sha)
    require(r['protocol']=='bell-geometry-sdf-mesh-audit-v1' and r['status']=='complete' and r['network_unchanged'] is True and r['numeric_output']['sha256']==measurement_sha,'Incomplete/unbound 3A-2 evidence')
    require(measurements.stat().st_size<=16*1024*1024,'Measurement size limit exceeded')
    data=measurements.read_bytes();require(hashlib.sha256(data).hexdigest()==measurement_sha,'Measurement SHA mismatch')
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        require(len(z.infolist())==len(FIELDS) and {v.filename for v in z.infolist()}=={k+'.npy' for k in FIELDS} and sum(v.file_size for v in z.infolist())<16*1024*1024,'Unreviewed numerical NPZ layout/resource size')
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    require(all(v.dtype.kind in 'bifu' for v in a.values()),'Only non-object numerical arrays permitted')
    result=localize(a)
    ray_ids={v['sample_index']:v for v in r['result']['rays']}
    for failed in result['failed_samples']:
        if failed['kind']=='ray':
            require(failed['sample_index'] in ray_ids,'Failed ray lacks saved source-camera identity')
            record=ray_ids[failed['sample_index']]
            failed.update(camera_id=record['camera_id'],ray_index=record['ray_index'])
    expected=r['statuses']['sdf_mesh_zero_level_consistency']
    require(all(result[k]==expected[k] for k in ['eligible','excluded','failures']),'Offline reconstruction differs from frozen 3A-2 result')
    require(hashlib.sha256(measurements.read_bytes()).hexdigest()==measurement_sha and hashlib.sha256(report.read_bytes()).hexdigest()==report_sha,'Evidence changed during analysis')
    return {'schema_version':1,'protocol':PROTOCOL,'report_identity':identity,
        'measurements_identity':{'path':str(measurements),'sha256':measurement_sha},'result':result,
        'original_statuses':r['statuses'],'other_findings':{
            'failed_A_controls':[{k:v[k] for k in ['camera_id','ray_index','radius','A_alignment_with_exterior_signed_gradient']}
                for v in r['result']['rays'] if not v['anomalous'] and v.get('A_alignment_with_exterior_signed_gradient',1)<r['settings']['direction_cosine']],
            'anomalies':[{k:v[k] for k in ['camera_id','ray_index','CPU']} for v in r['result']['rays'] if v['anomalous']]}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True);p.add_argument('--report-sha256',required=True)
    p.add_argument('--measurements',required=True);p.add_argument('--measurements-sha256',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();value=analyze(args.report,args.report_sha256,args.measurements,args.measurements_sha256)
    target=Path(args.output);require(target.resolve() not in [Path(args.report).resolve(),Path(args.measurements).resolve()],'Output overlaps evidence')
    with target.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print('Offline close-out: {} frozen failures localized; original evidence unchanged'.format(value['result']['failures']))


if __name__=='__main__':main()
