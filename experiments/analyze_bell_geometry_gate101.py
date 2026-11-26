"""File-only Gate 10.1 completion verification, stage attribution and SDF figures."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import bell_geometry_gate101 as p
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline,PINS

COMMIT='1852244c140c1e825003f44743865ece36aede35'
SHA='b1ad431e11a2bd59e97734de566eb767f7d9cd1339d08afabb80ae2c67608933'


def quantiles(a):
    return dict(zip(('min','p05','median','p95','max'),map(float,np.quantile(a,[0,.05,.5,.95,1]))))


def write_csv(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def figures(out,stages,sphere,lower,table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'savefig.dpi':200})
    raw=np.load(stages/'raw-SDF-grid.npy',mmap_mode='r',allow_pickle=False)
    masked=np.load(stages/'extraction-field.npy',mmap_mode='r',allow_pickle=False)
    coordinate=np.linspace(-1,1,512)
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    images=[raw[255,:,:],masked[255,:,:],raw[:,:,255].T,masked[:,:,255].T]
    titles=['Raw SDF, X≈−0.00196','Masked field, X≈−0.00196',
            'Raw SDF, Z≈−0.00196','Masked field, Z≈−0.00196']
    angle=np.linspace(0,2*np.pi,300)
    for i,(ax,image,title) in enumerate(zip(axes.flat,images,titles)):
        heat=ax.imshow(image,origin='lower',extent=(-1,1,-1,1),cmap='RdBu_r',
                       norm=TwoSlopeNorm(vmin=-.7,vcenter=0,vmax=.7),interpolation='nearest')
        if image.min()<=0<=image.max():
            ax.contour(coordinate,coordinate,image,levels=[0],colors='black',linewidths=.8)
        ax.plot(np.cos(angle),np.sin(angle),color='#0f766e',linestyle=':',linewidth=1)
        ax.set(title=title,xlabel='Z' if i<2 else 'X',ylabel='Y',aspect='equal')
    fig.colorbar(heat,ax=list(axes.flat),label='Saved field value (normalized scene units)',extend='both',shrink=.85)
    fig.savefig(out/'raw-versus-sentinel-fields.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    rows=[('Global sphere',slice(0,256)),('Lower-region sphere',slice(256,None))]
    x=np.arange(2)
    for key,color,label in [('sentinel_crossing','#e76f51','Sentinel-induced evidence'),
                            ('no_radial_crossing','#2a9d8f','No sampled crossing'),
                            ('ambiguous','#e9c46a','Ambiguous')]:
        below=np.zeros(2)
        if key=='no_radial_crossing':below=np.array([sphere['sentinel_crossing'][sl].sum() for _,sl in rows])
        if key=='ambiguous':below=np.array([(sphere['sentinel_crossing'][sl]|sphere['no_radial_crossing'][sl]).sum() for _,sl in rows])
        count=[sphere[key][sl].sum() for _,sl in rows]
        axes[0].bar(x,count,bottom=below,color=color,label=label,width=.6)
    axes[0].set_xticks(x,('Global sphere\n256 directions','Lower region\n128 directions'))
    axes[0].set_ylabel('Fixed spatial directions')
    axes[0].legend(fontsize=8,frameon=False)
    radii=np.array([-1,-.5,0,.5,1.])
    y=sphere['raw_values'][256:]
    transformed=np.where(sphere['inside_sphere'][256:],y,1)
    for values,color,label in [(y,'#155e75','Raw learned SDF'),(transformed,'#d97706','Sentinel-masked values')]:
        q=np.quantile(values,[.1,.5,.9],axis=0)
        axes[1].fill_between(radii,q[0],q[2],color=color,alpha=.15)
        axes[1].plot(radii,q[1],'-o',color=color,label=label,markersize=4)
    axes[1].axhline(0,color='.4',linewidth=.8)
    axes[1].axvline(0,color='.5',linestyle=':',linewidth=.8)
    axes[1].set(xlabel='Radial offset from unit sphere / h',ylabel='Field value',
                title='Lower 128 directions, median and 10–90%')
    axes[1].legend(fontsize=8,frameon=False)
    fig.savefig(out/'sphere-crossing-evidence.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    counts=[r['zero_area_count'] for r in table]
    axes[0].bar(np.arange(4),counts,color=['#64748b','#64748b','#d97706','#dc2626'],width=.6)
    axes[0].set_xticks(np.arange(4),('Marching\ncubes','World\ncoordinates','Trimesh\nprocessing','Float32\nPLY'))
    axes[0].set_ylabel('Exact zero-area triangles')
    for i,n in enumerate(counts):axes[0].text(i,n+.15,str(n),ha='center')
    axes[0].set_ylim(0,12)
    points=lower['points']
    heat=axes[1].scatter(points[:,2],points[:,1],c=lower['raw_sdf'],s=15,cmap='viridis')
    supported=lower['raw_support']
    axes[1].scatter(points[supported,2],points[supported,1],s=70,facecolors='none',edgecolors='red',label=f'Raw supported ({int(supported.sum())})')
    axes[1].set(xlabel='Z',ylabel='Y',title='256 area-quantile lower-surface witnesses',aspect='equal')
    axes[1].legend(fontsize=8,frameon=False)
    fig.colorbar(heat,ax=axes[1],label='Raw learned SDF')
    fig.savefig(out/'defect-stages-and-surface-support.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    require(not out.exists(),'Fresh offline output required')
    plan=p.freeze(SHA);p.verify_host_inputs(plan)
    root=Path(plan['host_output']);run=root/'run';stages=run/'stages'
    r=p.read(run/'receipt.json');verification=p.read(root/'verification.json')
    require(p.read(root/'authorization.json')==p.authorization(plan,COMMIT,SHA) and
            r['commit']==COMMIT and r['freeze_sha256']==SHA and r['checkpoint']==plan['checkpoint'] and
            r['network_state_preserved'] and r['status']=='complete' and r['scientific_work_complete'] and
            r['cleanup_verified'] and r['checkpoint_loads']==1 and r['extraction_tiles']==512 and
            r['extraction_tile_attempts']==512 and r['marching_cubes_calls']==1 and
            r['primary_BVH_rays']==r['CPU_ray_checks']==r['optimizer_returns']==r['shader_calls']==0 and
            r['decision']=='BLOCKED_substantive_geometry_review_required' and
            verification['receipt']==p.item(run/'receipt.json') and not verification['geometry_admitted'] and
            not verification['handoff_created'] and (root/'exit-code.txt').read_text().strip()=='0' and
            (root/'container-absent.txt').read_text().strip()=='absent','Unexpected diagnostic receipt or outcome')
    for name,row in r['artifacts'].items():p.check(run/name,row)
    mesh={}
    table=[]
    for name in ('marching','returned','processed','serialized'):
        v=np.load(stages/f'{name}-vertices.npy',allow_pickle=False)
        f=np.load(stages/f'{name}-faces.npy',allow_pickle=False)
        stage,area=p.mesh_stage(v,f)
        require(stage==r['stages'][name],'Stage replay differs')
        mesh[name]=(v,f,area)
        table.append(dict(stage=name,vertices=len(v),faces=len(f),vertex_dtype=str(v.dtype),
                          face_dtype=str(f.dtype),zero_area_count=stage['zero_area_count'],
                          repeated_index_count=len(stage['repeated_index_face_ids']),
                          coincident_coordinate_count=len(stage['coincident_coordinate_face_ids'])))
    mapping,change=p.exact_vertex_mapping(mesh['returned'][0],mesh['processed'][0],mesh['processed'][1],mesh['returned'][1])
    require(change==r['vertex_processing'] and
            np.array_equal(mapping,np.load(stages/'returned-to-processed.npy',allow_pickle=False)),'Vertex map differs')
    rows=[]
    masked=np.load(stages/'extraction-field.npy',mmap_mode='r',allow_pickle=False)
    raw=np.load(stages/'raw-SDF-grid.npy',mmap_mode='r',allow_pickle=False)
    outside=np.load(stages/'outside-sphere.npy',mmap_mode='r',allow_pickle=False)
    for start in range(0,512,32):
        require(np.array_equal(masked[start:start+32],np.where(outside[start:start+32],1,raw[start:start+32])),
                'Recorded raw/masked field relation differs')
    corner=[]
    for face in p.BAD_FACES:
        row=dict(serialized_face_id=face)
        first=None
        for name,(v,f,area) in mesh.items():
            row[name+'_area']=float(area[face])
            row[name+'_unique_coordinates']=len(np.unique(v[f[face]],axis=0))
            if area[face]==0 and first is None:first=name
        row['first_zero_area_stage']=first
        rows.append(row)
        if first=='marching':
            for point in np.unique(mesh['marching'][0][mesh['marching'][1][face]],axis=0):
                if np.array_equal(point,np.rint(point)):
                    index=tuple(point.astype(int))
                    corner.append(dict(face=face,grid_index=[int(x) for x in index],raw_SDF=float(raw[index]),
                                       extraction_value=float(masked[index]),sentinel_applied=bool(outside[index])))
    require([dict(serialized_face_id=x['serialized_face_id'],first_zero_area_stage=x['first_zero_area_stage'])
             for x in rows]==r['defect_localization'],'Defect localization differs')
    with np.load(run/'sphere-witnesses.npz',allow_pickle=False) as z:sphere={k:z[k] for k in z.files}
    with np.load(run/'lower-surface-witnesses.npz',allow_pickle=False) as z:lower={k:z[k] for k in z.files}
    labels=p.radial_labels(sphere['raw_values'])
    require(all(np.array_equal(sphere[k],a) for k,a in labels.items()),'Radial classification differs')
    for z in (sphere,lower):
        support=p.raw_support(z['raw_sdf'],z['gradient'],z['minus'],z['plus'])
        require(np.array_equal(z['raw_support'],support),'SDF support differs')
    require(int(lower['raw_support'].sum())==r['broad_SDF_diagnosis']['lower_raw_supported'] and
            r['broad_SDF_diagnosis']['substantive_review_required'],'Frozen review stop missing')
    groups={}
    for name,sl in [('global_256',slice(0,256)),('lower_128',slice(256,None))]:
        groups[name]={k:int(v[sl].sum()) for k,v in labels.items()}
        groups[name]['raw_SDF_at_radius1']=quantiles(sphere['raw_values'][sl,2])
    gridcounts={}
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        changed=(outside[a]!=outside[b])&(raw[a]<-1e-7)&(raw[b]<-1e-7)&(masked[a]*masked[b]<0)
        gridcounts[str(axis)]=int(changed.sum())
    require(gridcounts==r['sentinel_grid_edges_raw_negative_on_both_sides'],'Grid-edge counts differ')
    length=np.linalg.norm(lower['gradient'].astype(float),axis=1)
    out.mkdir()
    write_csv(out/'mesh-stages.csv',table)
    write_csv(out/'face-localization.csv',rows)
    p.write(out/'exact-zero-grid-corners.json',corner)
    summary=dict(execution_status='complete',scientific_status=r['decision'],geometry_admitted=False,handoff_created=False,
                 execution_commit=COMMIT,freeze_sha256=SHA,checkpoint=plan['checkpoint'],
                 receipt=p.item(run/'receipt.json'),verification=p.item(root/'verification.json'),
                 wall_seconds=r['wall_seconds'],cleanup_verified=True,containers_absent=True,
                 work=dict(checkpoint_loads=1,extraction_nodes=512**3,extraction_tiles=512,marching_cubes_calls=1,
                           diagnostic_SDF_points=r['broad_SDF_diagnosis']['point_evaluations'],
                           diagnostic_gradient_points=r['broad_SDF_diagnosis']['gradient_points'],
                           original_geometry_audit_executed=False,optimizer_returns=0,shader_calls=0),
                 stages=table,face_localization=rows,vertex_processing=change,
                 serialized_coordinate_shift=r['PLY_maximum_coordinate_shift'],float64_export=r['float64_export'],
                 sphere_direction_groups=groups,sentinel_grid_edges=gridcounts,
                 total_sentinel_grid_edges=sum(gridcounts.values()),lower_surface=dict(samples=len(lower['points']),
                     unique_original_faces=len(np.unique(lower['original_face_ids'])),
                     raw_supported=int(lower['raw_support'].sum()),raw_unsupported=int((~lower['raw_support']).sum()),
                     raw_SDF=quantiles(lower['raw_sdf']),gradient_length=quantiles(length),
                     zero_level_distance_proxy=quantiles(abs(lower['raw_sdf'])/length)),
                 conditional_correction_path_executed=False,diagnostic_positive_area_view_constructed=True,
                 derived_candidate_exported=False,
                 missing_original_audit=['silhouette','depth','signed_normals','correspondence','full_raster_exclusions',
                                         'original_complete_SDF_mesh_quality_gate'],
                 follow_up='Researcher review of domain-aware raw-SDF extraction from saved field with no artificial sphere closure.',
                 additional_scientific_execution=dict(checkpoint_loads=0,SDF_queries=0,extractions=0,ray_queries=0,optimizer_returns=0))
    p.write(out/'summary.json',summary)
    figures(out,stages,sphere,lower,table)
    for stage,pin in PINS.items():baseline(stage,pin)
    require(not (run/'handoff.json').exists() and not (run/'derived-exact-zero-filter.ply').exists(),
            'Unexpected admitted/corrected mesh')
    p.write(out/'artifact-manifest.json',dict(original_output=str(root),
          files={x.relative_to(root).as_posix():p.item(x) for x in sorted(root.rglob('*')) if x.is_file() and x.resolve()==x},
          analysis_source=p.item(Path(__file__).resolve()),
          outputs={x.name:p.item(x) for x in sorted(out.iterdir()) if x.is_file()}))
    print('Execution verified. Substantive geometry review required.')
    print(out/'summary.json')


if __name__=='__main__':
    main()
