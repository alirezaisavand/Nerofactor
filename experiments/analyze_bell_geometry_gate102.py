"""File-only Gate 10.2 results, verifier replay and saved-raster figures."""
import argparse
from argparse import Namespace
import csv
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import bell_geometry_gate102 as p
from benchmarks.bell_geometry_e11 import mesh_arrays
from experiments import bell_geometry_gate102 as execution
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline,PINS

COMMIT='f26cea4ae99b7b465d0ed9b357be714a87950b2e'
SHA='8b6cc635502d9794acaa69a7fe1c8de470c5acaa46ae1bc4f8e7e7bf70e27285'


def distribution(values):
    a=np.asarray(values,dtype=float)
    require(np.isfinite(a).all(),'Nonfinite offline measurement')
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),maximum=float(a.max()),mean=float(a.mean())) if len(a) else dict(count=0)


def table(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def replay(plan,root):
    p.host_inputs(plan)
    require(p.digest(root/'freeze.json')==SHA and
            p.read(root/'authorization.json')==p.authorization(plan,COMMIT,SHA),'Execution bindings differ')
    record=p.read(root/'verification.json');identity=p.item(root/'verification.json')
    # Preserve the original write-once verifier report. All computation is the
    # committed verifier; its final write is compared with the existing report.
    write=p.write;observed=[]
    def compare(path,value):
        require(Path(path)==root/'verification.json' and value==record,'Verifier replay differs')
        observed.append(value)
    p.write=compare
    try:
        execution.verify(plan,Namespace(commit=COMMIT,freeze_sha256=SHA,
                                       authorization_sha256=p.digest(root/'authorization.json')))
    finally:p.write=write
    require(len(observed)==1 and identity==p.item(root/'verification.json'),'Verification report changed')
    require(not record['handoff_created'] and not record['geometry_admitted'] and
            not (root/'run/handoff.json').exists(),'Unexpected handoff/admission')
    return record


def area_accounting(run,receipt):
    world=np.load(run/'raw-world-vertices.npy',allow_pickle=False,mmap_mode='r')
    faces=np.load(run/'raw-faces.npy',allow_pickle=False,mmap_mode='r')
    area=p.areas(world,faces)
    with np.load(run/'face-domain.npz',allow_pickle=False) as z:
        keep=z['retained'];reason=z['reason']
    require(np.array_equal(keep,reason==0),'Area-accounting domain masks differ')
    rows=[]
    for code,name in enumerate(('retained_domain','mixed_generators','boundary_cells','outside_corner_cells')):
        mask=reason==code
        rows.append(dict(group=name,faces=int(mask.sum()),float64_area=float(area[mask].sum()),
                         fraction_of_global_raw_area=float(area[mask].sum()/area.sum())))
    require(sum(row['faces'] for row in rows)==len(faces) and
            np.isclose(sum(row['float64_area'] for row in rows),area.sum(),rtol=1e-12),
            'Area inventory does not partition the raw extraction')
    with np.load(run/'cleanup-mappings.npz',allow_pickle=False) as z:removed=z['removed_exact_zero_raw_faces']
    v=np.load(run/'candidate-vertices.npy',allow_pickle=False,mmap_mode='r')
    f=np.load(run/'candidate-faces.npy',allow_pickle=False,mmap_mode='r')
    candidate=p.areas(v,f)
    require(np.isclose(candidate.sum(),receipt['historical_comparison']['candidate_area'],rtol=1e-12),
            'Candidate area differs')
    with np.load(run/'topology-after-cleanup.npz',allow_pickle=False) as z:edges=z['boundary_edges']
    midpoint=v[edges].astype(float).mean(axis=1)
    lengths=np.linalg.norm(v[edges[:,0]].astype(float)-v[edges[:,1]].astype(float),axis=1)
    radius=np.linalg.norm(midpoint,axis=1)
    endpoint_degree=np.bincount(edges.ravel(),minlength=len(v))
    band=receipt['domain_cells']['maximum_boundary_omission_depth']
    support=dict(boundary_edges=len(edges),boundary_vertices=int((endpoint_degree>0).sum()),
        endpoint_degree_histogram={str(k):int((endpoint_degree==k).sum()) for k in np.unique(endpoint_degree) if k>0},
        total_boundary_edge_length=float(lengths.sum()),midpoint_radius=distribution(radius),
        boundary_edges_in_conservative_radial_band=int((radius>=1-band).sum()),band_depth=band)
    return dict(global_raw_area=float(area.sum()),domain_retained_area_before_rounding=float(area[keep].sum()),
        excluded_area=float(area[~keep].sum()),candidate_area=float(candidate.sum()),
        float32_rounding_net_area_change=float(candidate.sum()-area[keep].sum()),
        removed_face_ids=removed.tolist(),removed_faces_raw_exact_zero=int((area[removed]==0).sum()),
        removed_faces_raw_positive=int((area[removed]>0).sum()),removed_faces_original_area=float(area[removed].sum()),
        groups=rows,open_boundary=support)


def raster_metrics(row,z,band):
    a,b=z['oracle_hit'],z['reconstructed_hit'];common=a&b
    extra=b&~a
    n0,n1=z['oracle_normal'],z['reconstructed_normal']
    angle=np.degrees(np.arccos(np.clip((n0*n1).sum(1),-1,1)))
    delta=z['reconstructed_depth'].astype(float)-z['oracle_depth'].astype(float)
    grazing=common&(np.minimum(-(n0*z['directions']).sum(1),-(n1*z['directions']).sum(1))<p.original.QUALITY['grazing_cosine'])
    radius=np.linalg.norm(z['reconstructed_position'][extra].astype(float),axis=1)
    exclude=common&~z['valid']
    d_bad=common&~z['depth_close'];n_bad=common&~z['signed_normal_ok']
    return dict(camera=row['camera'],split=row['split'],total_pixels=len(a),oracle=int(a.sum()),
        reconstructed=int(b.sum()),common=int(common.sum()),shared_miss=int((~a&~b).sum()),
        reconstruction_only=int(extra.sum()),oracle_only=int((a&~b).sum()),matched=int(z['valid'].sum()),
        excluded_common=int(exclude.sum()),distance_failures=int(d_bad.sum()),normal_failures=int(n_bad.sum()),
        distance_normal_overlap=int((d_bad&n_bad).sum()),grazing_common=int(grazing.sum()),
        absolute_depth=distribution(abs(delta[common])),signed_depth=distribution(delta[common]),
        signed_normal_degrees=distribution(angle[common]),
        grazing_absolute_depth=distribution(abs(delta[grazing])),
        nongrazing_absolute_depth=distribution(abs(delta[common&~grazing])),
        reconstruction_only_radius=distribution(radius),
        reconstruction_only_radial_interior=int((radius<1-band).sum()),
        reconstruction_only_radial_boundary_band=int((radius>=1-band).sum()),
        large_depth_over_point1=int((common&(abs(delta)>.1)).sum()),
        oracle_domain_supported=int(z['oracle_conservative_domain'].sum()),
        missing_inside_domain=int(((a&~b)&z['oracle_conservative_domain']).sum()),
        missing_outside_domain=int(((a&~b)&~z['oracle_conservative_domain']).sum()),
        silhouette_iou=row['silhouette_iou'],matched_fraction_oracle=row['matched_fraction_oracle'],
        passed=row['passed'])


def figures(out,plan,run,receipt,rasters,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size':10,'savefig.dpi':200,'pdf.fonttype':42})
    def export(fig,name):
        fig.savefig(out/(name+'.png'),dpi=200);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    depth=np.concatenate([z[k+'_depth'][z[k+'_hit']] for z in rasters.values() for k in ('oracle','reconstructed')])
    dmin,dmax=float(depth.min()),float(depth.max())
    fig,axes=plt.subplots(6,7,figsize=(19,16))
    fig.subplots_adjust(left=.07,right=.99,top=.94,bottom=.10,hspace=.09,wspace=.07)
    maskcmap=ListedColormap(['white','#cbd5e1','#dc2626','#2563eb'])
    masknorm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    titles=('Oracle normals','Candidate normals','Oracle depth','Candidate depth',
            'Silhouette support','Signed depth error','Signed normal error')
    for j,name in enumerate(titles):axes[0,j].set_title(name,fontsize=10,pad=8)
    for i,row in enumerate(receipt['views']):
        z=rasters[row['camera']];a,b=z['oracle_hit'],z['reconstructed_hit'];common=a&b
        for j,key in enumerate(('oracle','reconstructed')):
            image=np.ones((len(a),3));image[z[key+'_hit']]=np.clip((z[key+'_normal'][z[key+'_hit']]+1)/2,0,1)
            axes[i,j].imshow(image.reshape(128,128,3),interpolation='nearest')
            dep=np.ma.array(z[key+'_depth'].reshape(128,128),mask=~z[key+'_hit'].reshape(128,128))
            depth_image=axes[i,2+j].imshow(dep,cmap='viridis',vmin=dmin,vmax=dmax,interpolation='nearest')
        mask=np.zeros(len(a),np.uint8);mask[common]=1;mask[b&~a]=2;mask[a&~b]=3
        axes[i,4].imshow(mask.reshape(128,128),cmap=maskcmap,norm=masknorm,interpolation='nearest')
        delta=z['reconstructed_depth'].astype(float)-z['oracle_depth'].astype(float)
        delta=np.ma.array(delta.reshape(128,128),mask=~common.reshape(128,128))
        error_image=axes[i,5].imshow(delta,cmap='RdBu_r',vmin=-.02,vmax=.02,interpolation='nearest')
        angle=np.degrees(np.arccos(np.clip((z['oracle_normal']*z['reconstructed_normal']).sum(1),-1,1)))
        angle=np.ma.array(angle.reshape(128,128),mask=~common.reshape(128,128))
        normal_image=axes[i,6].imshow(angle,cmap='magma',vmin=0,vmax=30,interpolation='nearest')
        axes[i,0].set_ylabel(f"Camera {row['camera']}\nIoU {row['silhouette_iou']:.4f}",fontsize=10)
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Authenticated 300k Bell — domain-aware learned surface',fontsize=16,y=.98)
    for x,width,artist,label in ((.35,.20,depth_image,'Depth (scene units; full range)'),
                                (.60,.16,error_image,'Candidate − oracle depth (±0.02 display)'),
                                (.81,.16,normal_image,'Normal difference (degrees; 30° display cap)')):
        bar=fig.colorbar(artist,cax=fig.add_axes([x,.055,width,.012]),orientation='horizontal')
        bar.set_label(label,fontsize=9)
    fig.legend(handles=[Patch(color='#cbd5e1',label='Common foreground'),Patch(color='#dc2626',label='Candidate only'),
                        Patch(color='#2563eb',label='Oracle only (none)')],loc='lower left',bbox_to_anchor=(.025,.018),
               fontsize=9,frameon=False)
    fig.text(.60,.018,'White = undefined/background. Error display caps do not alter metrics or eligibility.',fontsize=9)
    export(fig,'raster-geometry-normal-depth-errors')
    historical=mesh_arrays(p.previous.PRIOR/'evaluate/bell_shape-300000.ply')[0]
    candidate=np.load(run/'candidate-vertices.npy',allow_pickle=False,mmap_mode='r')
    oracle=mesh_arrays(Path(plan['acquisition'])/'geometry/mesh.ply')[0]
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for j,(title,vertices) in enumerate((('Historical sentinel-masked',historical),('Domain-aware raw SDF',candidate),('Oracle',oracle))):
        shown=vertices[::max(1,int(np.ceil(len(vertices)/60000)))]
        for i,(x,y) in enumerate(((0,1),(2,1))):
            axes[i,j].scatter(shown[:,x],shown[:,y],s=.08,color='#155e75',rasterized=True)
            theta=np.linspace(0,2*np.pi,361)
            axes[i,j].plot(np.cos(theta),np.sin(theta),':',color='.6',lw=.7)
            axes[i,j].set(xlim=(-1.04,1.04),ylim=(-1.04,1.04),aspect='equal',xlabel='XZ'[i],ylabel='Y')
            if i==0:axes[i,j].set_title(title)
    fig.suptitle('Saved vertex projections — display striding only; no component selection or new ray queries',fontsize=11)
    export(fig,'historical-domain-aware-oracle-projections')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    x=np.arange(6);rows=summary['cameras']
    for name,label,color in (('silhouette_iou','Silhouette IoU','#155e75'),('matched_fraction_oracle','Valid correspondence / oracle','#d97706')):
        axes[0,0].plot(x,[r[name] for r in rows],'-o',label=label,color=color)
    axes[0,0].axhline(.9,color='#155e75',ls=':',lw=1);axes[0,0].axhline(.8,color='#d97706',ls=':',lw=1)
    axes[0,0].set(xticks=x,xticklabels=[r['camera'] for r in rows],ylim=(.75,1.01),xlabel='Camera',ylabel='Fraction')
    axes[0,0].legend(fontsize=8,frameon=False)
    for split,color in (('validation','#155e75'),('held_view_nvs','#d97706')):
        values=np.sort(np.concatenate([abs(z['reconstructed_depth'][z['common']].astype(float)-z['oracle_depth'][z['common']].astype(float))
            for row in receipt['views'] for z in [rasters[row['camera']]] if row['split']==split]))
        positive=values>0
        axes[0,1].plot(values[positive],np.arange(1,len(values)+1)[positive]/len(values),color=color,label=split)
    axes[0,1].axvline(.02,color='.4',ls=':');axes[0,1].set(xscale='log',xlabel='Absolute depth error (scene units)',ylabel='Common-hit cumulative fraction')
    axes[0,1].legend(fontsize=8,frameon=False)
    with np.load(run/'sdf-witnesses.npz',allow_pickle=False) as z:
        proxy=abs(z['raw_sdf'].astype(float))/z['gradient_length']
    axes[1,0].hist(proxy,bins=30,color='#155e75');axes[1,0].axvline(p.original.QUALITY['sdf_step'],ls=':',color='#dc2626',label='Frozen bound 1/511')
    axes[1,0].set(xlabel='Raw-SDF first-order distance proxy',ylabel='Saved witness count');axes[1,0].legend(fontsize=8,frameon=False)
    comparison=receipt['historical_comparison']
    total=[comparison['original_area'],comparison['candidate_area']]
    band=[comparison['original_near_sphere_area'],comparison['candidate_near_sphere_area']]
    axes[1,1].bar([0,1],np.array(total)-band,color='#155e75',label='Remaining surface')
    axes[1,1].bar([0,1],band,bottom=np.array(total)-band,color='#d97706',label='All corners within 2/511 of unit sphere')
    axes[1,1].set(xticks=[0,1],xticklabels=['Historical masked','Domain-aware'],ylabel='Surface area (scene units²)')
    axes[1,1].legend(fontsize=8,frameon=False)
    export(fig,'quality-support-and-sdf-summary')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve();require(not out.exists(),'Fresh offline output required')
    plan=p.freeze(SHA);root=Path(plan['host_output']);run=root/'run'
    verification=replay(plan,root);r=p.read(run/'receipt.json')
    require(r['status']=='complete' and r['scientific_work_complete'] and r['cleanup_verified'] and
            r['network_state_preserved'] and r['checkpoint']==plan['checkpoint'] and
            r['decision']=='BLOCKED_original_geometry_quality_gate' and r['original_SDF_gate']['passed'],
            'Unexpected completion or scientific decision')
    area=area_accounting(run,r);rasters={};rows=[]
    for row in r['views']:
        with np.load(run/'rasters'/f"{row['camera']}.npz",allow_pickle=False) as z:
            rasters[row['camera']]={key:z[key] for key in z.files}
        rows.append(raster_metrics(row,rasters[row['camera']],area['open_boundary']['band_depth']))
    require(sum(row['oracle_only'] for row in rows)==0 and sum(row['reconstruction_only'] for row in rows)==2807 and
            sum(not row['passed'] for row in rows)==4,'Scientific summary differs from six-camera report')
    cpu=[w for row in r['views'] for group in row['cpu_closest_hit'].values() for w in group]
    require(len(cpu)==384 and all(w['passed'] for w in cpu),'Incomplete independent closest-hit witnesses')
    inventory={key:sum(row[key] for row in rows) for key in
        ('total_pixels','oracle','reconstructed','common','shared_miss','reconstruction_only','oracle_only',
         'matched','excluded_common','distance_failures','normal_failures','distance_normal_overlap',
         'grazing_common','large_depth_over_point1','reconstruction_only_radial_interior',
         'reconstruction_only_radial_boundary_band','oracle_domain_supported','missing_inside_domain','missing_outside_domain')}
    require(inventory['common']+inventory['reconstruction_only']+inventory['oracle_only']+inventory['shared_miss']==inventory['total_pixels']
            and inventory['matched']+inventory['excluded_common']==inventory['common'],
            'Full-raster accounting does not close')
    summary=dict(decision='BLOCKED',execution_status='complete_successful',quality_passed=False,geometry_admitted=False,
        handoff_created=False,execution_commit=COMMIT,freeze_sha256=SHA,checkpoint=plan['checkpoint'],
        receipt=p.item(run/'receipt.json'),verification=p.item(root/'verification.json'),verifier_replay='exact_nonmutating',
        image=r['image_id'],backend=r['backend'],cleanup_verified=True,network_state_preserved=True,
        wall_seconds=r['wall_seconds'],peak_RSS_bytes=r['peak_RSS_bytes'],peak_cuda_allocated_bytes=r['peak_cuda_allocated_bytes'],
        work=dict(MC_attempts=r['MC_attempts'],MC_returns=r['MC_returns'],primary_BVH_rays=r['primary_BVH_rays'],
                  CPU_ray_checks=r['CPU_ray_checks'],checkpoint_loads=r['checkpoint_loads'],SDF=r['SDF_work'],
                  optimizer_returns=r['optimizer_returns'],shader_calls=r['shader_calls']),
        mesh=r['mesh_outputs'],cleanup=r['cleanup'],topology_before=r['topology_before_cleanup'],
        topology_after=r['topology_after_cleanup'],historical_comparison=r['historical_comparison'],area_accounting=area,
        cameras=rows,full_raster=inventory,independent_CPU_witnesses=dict(total=len(cpu),passed=len(cpu)),
        split_summaries=verification['split_summaries'],original_SDF_gate=r['original_SDF_gate'],
        additional_scientific_execution=dict(checkpoint_loads=0,SDF_queries=0,extractions=0,ray_queries=0,optimizer_returns=0))
    out.mkdir();p.write(out/'summary.json',summary);table(out/'surface-area-accounting.csv',area['groups'])
    flat=[]
    for row in rows:
        v={k:value for k,value in row.items() if not isinstance(value,dict)}
        for name in ('absolute_depth','signed_depth','signed_normal_degrees'):
            for k,value in row[name].items():v[name+'_'+k]=value
        flat.append(v)
    table(out/'per-camera.csv',flat)
    tails=[]
    for row in r['views']:
        z=rasters[row['camera']]
        delta=z['reconstructed_depth'].astype(float)-z['oracle_depth'].astype(float)
        angle=np.degrees(np.arccos(np.clip((z['oracle_normal']*z['reconstructed_normal']).sum(1),-1,1)))
        for pixel in np.flatnonzero(z['common']&(abs(delta)>.1)):
            tails.append(dict(camera=row['camera'],pixel=int(pixel),x=int(pixel%128),y=int(pixel//128),
                oracle_depth=float(z['oracle_depth'][pixel]),candidate_depth=float(z['reconstructed_depth'][pixel]),
                signed_depth_error=float(delta[pixel]),normal_difference_degrees=float(angle[pixel]),
                correspondence_valid=bool(z['valid'][pixel])))
    table(out/'large-depth-shifts.csv',tails)
    figures(out,plan,run,r,rasters,summary)
    for stage,pin in PINS.items():baseline(stage,pin)
    p.write(out/'artifact-manifest.json',dict(original_output=str(root),
        files={x.relative_to(root).as_posix():p.item(x) for x in sorted(root.rglob('*')) if x.is_file() and x.resolve()==x},
        analysis_source=p.item(Path(__file__).resolve()),
        outputs={x.name:p.item(x) for x in sorted(out.iterdir()) if x.is_file()}))
    print(out/'summary.json')


if __name__=='__main__':main()
