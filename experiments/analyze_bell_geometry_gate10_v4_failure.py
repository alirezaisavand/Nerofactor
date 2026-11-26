"""File-only verification and diagnosis of completed G training and rejected mesh."""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import bell_geometry_gate10_v1 as g
from benchmarks.bell_geometry_e11 import mesh_arrays
from experiments.bell_geometry_gate10_v1 import completed_train, context
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline,PINS

COMMIT='6638d3233e439b2c1ec9279eb46c2d514ba2509a'
SHA='6c77069cc577735a63638674094e6e7ce23666f57a12e72a3ea91a6044683a8e'


def logs(path,prefix,interval):
    records=[]
    for i,line in enumerate(path.read_text().splitlines(),1):
        words=line.split()
        require(words[0]==prefix and len(words)%2==1,'Malformed native log')
        row={words[j]:float(words[j+1]) for j in range(1,len(words),2)}
        require(all(np.isfinite(v) for v in row.values()),'Nonfinite saved log')
        records.append(dict(step=i*interval,**row))
    return records


def face_diagnostics(vertices,faces):
    require(np.isfinite(vertices).all() and faces.min()>=0 and faces.max()<len(vertices),'Invalid stored mesh')
    triangle=vertices[faces].astype(np.float64)
    twice=np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)
    require(np.isfinite(twice).all(),'Nonfinite mesh triangle area')
    bad=np.flatnonzero(twice==0)
    repeated=(faces[:,0]==faces[:,1])|(faces[:,1]==faces[:,2])|(faces[:,0]==faces[:,2])
    radius=np.linalg.norm(triangle,axis=2)
    h=2/511
    details=[]
    for index in bad:
        points=triangle[index]
        neighbor=np.flatnonzero(np.isin(faces,faces[index]).any(axis=1))
        details.append(dict(face=int(index),indices=faces[index].tolist(),position=points.mean(axis=0).tolist(),
                            vertices=points.tolist(),unique_coordinate_count=len(np.unique(points,axis=0)),
                            repeated_index=bool(repeated[index]),minimum_radius=float(radius[index].min()),
                            maximum_radius=float(radius[index].max()),near_unit_sphere=bool((abs(radius[index]-1)<=h).any()),
                            incident_faces=len(neighbor),positive_area_incident_faces=int((twice[neighbor]>0).sum())))
    report=dict(vertices=len(vertices),faces=len(faces),finite_vertices=True,finite_triangle_areas=True,
                zero_area_faces=len(bad),zero_area_fraction=len(bad)/len(faces),repeated_index_faces=int(repeated.sum()),
                distinct_index_zero_area_faces=int((~repeated[bad]).sum()),
                coincident_coordinate_zero_area_faces=sum(d['unique_coordinate_count']<3 for d in details),
                minimum_positive_area=float(twice[twice>0].min()/2),
                total_positive_area=float(twice.sum()/2),bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
                unit_sphere_band_width=h,zero_area_near_sphere=sum(d['near_unit_sphere'] for d in details),
                faces_touching_sphere_band=int((abs(radius-1)<=h).any(axis=1).sum()),
                faces_entirely_in_sphere_band=int((abs(radius-1)<=h).all(axis=1).sum()),
                area_fraction_entirely_in_sphere_band=float(twice[(abs(radius-1)<=h).all(axis=1)].sum()/twice.sum()),
                zero_area_away_from_sphere=sum(not d['near_unit_sphere'] for d in details),
                sphere_cap_attribution='unverified without raw SDF and staged extraction witnesses')
    return report,details


def table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def figures(out,train,val,bins,vertices,oracle,details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':160,'savefig.dpi':200})
    fig,axes=plt.subplots(2,1,figsize=(9,6.5),sharex=True,constrained_layout=True)
    x=np.array([r['step'] for r in bins])/1000
    for key,label in [('loss_rgb','RGB'),('loss_occ','Occlusion'),('loss_mask','Mask'),('loss_eikonal','Eikonal'),('loss_opacity','Opacity')]:
        axes[0].plot(x,[r.get(key,np.nan) for r in bins],label=label,linewidth=1.3)
    axes[0].set_yscale('symlog',linthresh=1e-5)
    axes[0].set_ylabel('Weighted loss, 1k-bin median')
    axes[0].legend(ncol=3,frameon=False)
    axes[1].plot([r['step']/1000 for r in val],[r['val-psnr'] for r in val],color='#155e75',linewidth=1)
    axes[1].set_ylabel('Native validation PSNR (dB)')
    axes[1].set_xlabel('Completed original Adam returns (thousands)')
    for ax in axes:
        for step in (1,5,10,15,20,30,50):ax.axvline(step,color='.7',linestyle=':',linewidth=.6)
        ax.grid(alpha=.2)
    fig.savefig(out/'training-trajectories.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,4.2),constrained_layout=True)
    selected=vertices[np.linspace(0,len(vertices)-1,10000,dtype=int)]
    ref=oracle[np.linspace(0,len(oracle)-1,5000,dtype=int)]
    bad=np.array([d['position'] for d in details])
    angle=np.linspace(0,2*np.pi,200)
    for ax,(i,j,label) in zip(axes,[(0,1,'X–Y'),(0,2,'X–Z'),(2,1,'Z–Y')]):
        ax.scatter(ref[:,i],ref[:,j],s=.15,color='#60a5fa',alpha=.5,rasterized=True,label='Oracle vertices')
        ax.scatter(selected[:,i],selected[:,j],s=.15,color='.25',alpha=.5,rasterized=True,label='Extracted vertices')
        ax.scatter(bad[:,i],bad[:,j],s=22,facecolor='none',edgecolor='#dc2626',linewidth=.8,label='Zero-area face centers')
        ax.plot(np.cos(angle),np.sin(angle),':',color='.5',linewidth=.7)
        ax.set(title=label,aspect='equal',xlim=(-1.05,1.05),ylim=(-1.05,1.05))
        ax.set_xlabel('Normalized scene coordinate')
    axes[0].legend(loc='upper left',fontsize=7,markerscale=2,frameon=False)
    fig.savefig(out/'mesh-defect-locations.png');plt.close(fig)
    from PIL import Image
    steps=(1000,15000,30000,50000,100000,200000,300000)
    fig,axes=plt.subplots(len(steps),3,figsize=(5,10),constrained_layout=True)
    images={}
    for row,step in enumerate(steps):
        path=train/'workspace/data/train_vis/bell_shape-val'/f'{step-1}-index-0.jpg'
        image=np.array(Image.open(path).convert('RGB'))
        require(image.shape==(160,128,3),'Unexpected native visualization montage shape')
        images[str(step)]=dict(path=str(path),**g.item(path))
        for col in range(3):
            axes[row,col].imshow(image[:32,col*32:(col+1)*32],interpolation='nearest')
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(('Native GT','Native RGB','Native normal')[col])
            if col==0:axes[row,col].set_ylabel(f'{step//1000}k')
    fig.savefig(out/'saved-validation-progression.png');plt.close(fig)
    return dict(validation_visualization_sources=images)



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    require(not out.exists(),'Analysis output must be fresh')
    f=g.freeze(SHA)
    root=Path(f['host_output'])
    require(g.definition(root)==f,'Frozen inputs/source differ')
    auth=root/'authorization.json'
    a=SimpleNamespace(commit=COMMIT,freeze_sha256=SHA,authorization_sha256=g.digest(auth))
    require(g.read(auth)==g.authorization(f,COMMIT,SHA),'Run authorization differs')
    train=completed_train(root/'train',f,a)
    require((root/'train-exit-code.txt').read_text().strip()=='0' and
            all((root/(phase+'-container-absent.txt')).read_text().strip()=='absent' for phase in ('train','evaluate')),
            'Missing process cleanup receipts')
    evaluate=g.read(root/'evaluate/evaluation.json')
    require(all(evaluate[k]==v for k,v in context(f,a).items()) and evaluate['status']=='failed' and
            evaluate['error']['message']=='Degenerate or nonfinite triangles; no repair permitted' and
            evaluate['training_receipt']==g.item(root/'train/training.json') and
            evaluate['checkpoint']==train['endpoint_checkpoint'] and evaluate['network_sha256']==train['endpoint_network_sha256']
            and evaluate['extraction_tiles']==512 and evaluate['primary_BVH_ray_slots']==0 and
            not evaluate['cleanup_verified'] and not evaluate['views'] and not evaluate['quality_passed'] and
            (root/'evaluate-exit-code.txt').read_text().strip()=='1','Unexpected failure/provenance')
    for phase,r in [('train',train),('evaluate',evaluate)]:
        for name,row in r.get('artifacts',{}).items():g.check(root/phase/name,row)
        for name,row in r['runtime_extension_files'].items():g.check(root/phase/name,row)
    for stage,pin in PINS.items():baseline(stage,pin)
    out.mkdir()
    mesh=root/'evaluate/bell_shape-300000.ply'
    vertices,faces=mesh_arrays(mesh)
    diagnostic,details=face_diagnostics(vertices,faces)
    diagnostic['raw_extraction_counts']=evaluate['mesh_raw_counts']
    diagnostic['vertices_removed_before_PLY']=evaluate['mesh_raw_counts']['vertices']-len(vertices)
    require(diagnostic['zero_area_faces']>0,'Expected true geometric gate failure')
    g.write(out/'degenerate-faces.json',details)
    table(out/'degenerate-faces.csv',[{k:v for k,v in d.items() if k not in ('vertices','indices','position')} for d in details])
    tr=logs(root/'train/logs/train.txt','train',20)
    val=logs(root/'train/logs/val.txt','val',1000)
    require(len(tr)==15000 and len(val)==300,'Native log cadence/count differs')
    bins=[]
    for start in range(0,len(tr),50):
        block=tr[start:start+50]
        keys=set().union(*(r.keys() for r in block))-{'step'}
        bins.append(dict(step=block[-1]['step'],**{k:float(np.median([r.get(k,0) for r in block])) for k in sorted(keys)}))
    table(out/'training-loss-bin-medians.csv',bins)
    table(out/'validation.csv',val)
    oracle,_=mesh_arrays(Path(f['acquisition'])/'geometry/mesh.ply')
    visual=figures(out,root/'train',val,bins,vertices,oracle,details)
    periods={}
    for start,end in [(0,1000),(1000,10000),(10000,20000),(20000,50000),(50000,100000),(100000,200000),(200000,300000)]:
        rows=[r for r in val if start<r['step']<=end]
        periods[str(start)+'-'+str(end)]=dict(native_PSNR_median=float(np.median([r['val-psnr'] for r in rows])),
                                            native_PSNR_min=float(min(r['val-psnr'] for r in rows)),
                                            native_PSNR_max=float(max(r['val-psnr'] for r in rows)))
    summary=dict(status='BLOCKED_geometry_admission_failed',training_complete=True,geometry_admitted=False,handoff_created=False,
                 commit=COMMIT,freeze_sha256=SHA,authorization_sha256=a.authorization_sha256,original_output=str(root),
                 training_receipt=g.item(root/'train/training.json'),evaluation_receipt=g.item(root/'evaluate/evaluation.json'),
                 endpoint_checkpoint=train['endpoint_checkpoint'],mesh=g.item(mesh),
                 adam_returns=train['adam_returns'],validations=train['validation_calls'],training_hours=train['wall_seconds']/3600,
                 retained_checkpoints=train['retained_checkpoints'],training_cleanup_verified=True,evaluation_cleanup_verified=False,
                 containers_absent=True,peak_RSS_bytes=train['peak_RSS_bytes'],peak_cuda_allocated_bytes=train['peak_cuda_allocated_bytes'],
                 peak_cuda_reserved_bytes=train['peak_cuda_reserved_bytes'],mesh_diagnostics=diagnostic,phase_validation=periods,
                 final_native_validation=train['last_validation'],training_witnesses=train['training_witnesses'],
                 mesh_packages=evaluate['mesh_packages'],extraction_grid_nodes=512**3,extraction_tiles=512,
                 missing_evaluation=['oracle replay','FP64 ray witnesses','silhouette IoU','depth errors','signed normal errors',
                                     'correspondence/foreground exclusions','raw SDF/mesh consistency','unit-sphere cap attribution'],
                 additional_scientific_execution=dict(Adam_updates=0,checkpoint_deserializations=0,SDF_queries=0,
                                                      geometry_rays=0,shader_calls=0,mesh_repairs=0),
                 known_limit='No raw pre-Trimesh or pre-PLY vertex/face arrays were saved; stage attribution remains unresolved.',
                 **visual)
    g.write(out/'summary.json',summary)
    inventory={p.relative_to(root).as_posix():g.item(p) for p in sorted(root.rglob('*')) if p.is_file() and p.resolve()==p}
    g.write(out/'artifact-manifest.json',dict(original_output=str(root),files=inventory,
                                           analysis_source=g.item(Path(__file__).resolve()),
                                           outputs={p.name:g.item(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(g.read(out/'summary.json')['mesh_diagnostics'])
    print('Saved offline analysis',out)


if __name__=='__main__':
    main()
