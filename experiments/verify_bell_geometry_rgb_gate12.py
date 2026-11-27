"""File-only replay, coverage and uncertainty accounting for Gate 12."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from benchmarks import bell_geometry_rgb_gate12 as g
from experiments.bell_geometry_rgb_gate12 import capture_name,validate_capture,VISIBILITY_FIELDS
from experiments.verify_donut_oracle_geometry_rgb_v1 import score
from infrastructure.signatures import require


def gates_equal(saved,observed):
    """Only the norm reduction gets the accepted Gate 3 finite roundoff allowance."""
    require(set(saved)==set(observed) and saved['passed'] is observed['passed'] is True and
            saved['maximum_rgb_difference']==observed['maximum_rgb_difference'] and
            np.isfinite(saved['relative_l2']) and
            np.isclose(saved['relative_l2'],observed['relative_l2'],atol=1e-15,rtol=1e-12),
            'Gate 12 convergence replay differs')


def expand(data,states):
    images={}
    for arm,s in states.items():
        x=np.zeros((2,128**2,len(g.KINDS),3,3),np.float64);x[:,s['pixels']]=s['components'];images[arm]=x
        if arm!='oracle':
            for name,rows in [('incident_disagreement_any',s['visibility'][1,:,2]>0),
                              ('shadow_radial_band_seen',(s['visibility'][1,:,5:7]>0).any(1))]:
                mask=np.zeros(128**2,bool);mask[s['pixels']]=rows;data['cohorts'][arm][name]=mask
    return dict(data,images=images,states=states)


def metrics(records):
    result={}
    for arm in ('O0','O1'):
        target=np.concatenate([v['images']['oracle'][:,:,0] for v in records],axis=1)
        prediction=np.concatenate([v['images'][arm][:,:,0] for v in records],axis=1)
        keys=records[0]['cohorts'][arm].keys();cohorts={k:np.concatenate([v['cohorts'][arm][k] for v in records]) for k in keys}
        known=cohorts['known'];scores={k:score(target,prediction,m,known) for k,m in cohorts.items() if k not in ('known','undefined')}
        for scope,measurement in scores.items():
            use=cohorts[scope]&known;delta=prediction[1,use]-target[1,use]
            for c,name in enumerate(('diffuse','specular','RGB')):
                rmse=np.sqrt(np.mean(delta[:,c]**2,axis=1))
                measurement[name]['per_pixel_RMSE_distribution']=g.prior.original.distribution(rmse)
                measurement[name]['per_pixel_max_channel_distribution']=g.prior.original.distribution(abs(delta[:,c]).max(1))
        # Controlled substitutions share exactly the SAME BRDF samples per arm.
        contrast={}
        for k,label in [(1,'visibility_function_change'),(2,'occlusion_effect'),(3,'global_field_vs_transported_source')]:
            substitute=np.concatenate([v['images'][arm][:,:,k] for v in records],axis=1)
            contrast[label]=score(substitute,prediction,cohorts['valid'],known)
        unoccluded=np.concatenate([v['images'][arm][:,:,4] for v in records],axis=1)
        oracle_unoccluded=np.concatenate([v['images']['oracle'][:,:,4] for v in records],axis=1)
        contrast['unoccluded_normal_frame_change']=score(oracle_unoccluded,unoccluded,cohorts['valid'],known)
        # Visibility statistics are observation-normalized, not node-count weighted.
        visibility=np.concatenate([v['states'][arm]['visibility'][1] for v in records])
        nodes=np.concatenate([v['states'][arm]['grids'][1]**2 for v in records])
        fractional=visibility[:,:7]/nodes[:,None]
        weighted=np.divide(visibility[:,7:10],visibility[:,10:13],out=np.zeros_like(visibility[:,7:10]),where=visibility[:,10:13]>0)
        transfer={k:g.prior.original.distribution(np.concatenate([v['transfers'][arm][k] for v in records]))
                  for k in ('normal_degrees','Q_coordinate_shift','Q_frame_shift','Q_projection_vs_transport')}
        transfer['coordinate_displacement']=g.prior.original.distribution(np.concatenate([
            np.linalg.norm(v['transfers'][arm]['coordinate_displacement'],axis=1) for v in records]))
        result[arm]=dict(scopes=scores,controlled_contrasts=contrast,material_coordinate_and_frame=transfer,
            incident_visibility=dict(observations=len(nodes),fraction_fields=list(VISIBILITY_FIELDS[:7]),
                mean_per_observation_fractions=fractional.mean(0).tolist(),
                light_weighted_disagreement_RGB=weighted.mean(0).tolist(),
                quadrature_is_deterministic_not_MC=True),
            undefined_support=int(cohorts['undefined'].sum()),
            exclusion_reasons_overlap={k:int(cohorts[k].sum()) for k in
                                      ('reconstruction_only','normal_failure','distance_failure','facing_failure')})
    return result


def figure(records,out,component=2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cameras=[v['camera'] for v in records];fig,axes=plt.subplots(len(records),5,figsize=(12,2.1*len(records)),squeeze=False)
    clipping=[]
    for row,v in enumerate(records):
        target=v['images']['oracle'][1,:,0,component];axes[row,0].imshow(np.clip(target,0,1).reshape(128,128,3))
        for j,arm in enumerate(('O0','O1')):
            a=v['images'][arm][1,:,0,component];known=v['cohorts'][arm]['known']
            preview=np.clip(a,0,1);preview[~known]=[1,0,1]
            axes[row,j+1].imshow(preview.reshape(128,128,3))
            err=np.sqrt(np.mean((a-target)**2,axis=1));err[~known]=np.nan
            cmap=plt.get_cmap('magma').copy();cmap.set_bad('magenta')
            im=axes[row,j+3].imshow(err.reshape(128,128),vmin=0,vmax=.05,cmap=cmap)
            clipping.append(dict(camera=v['camera'],arm=arm,preview_channels_above_one=int(((a>1)&known[:,None]).sum()),
                                 error_pixels_above_display=int(((err>.05)&known).sum()),undefined=int((~known).sum())))
        clipping.append(dict(camera=v['camera'],arm='oracle',preview_channels_above_one=int((target>1).sum())))
        axes[row,0].set_ylabel('Camera '+cameras[row])
        for j in range(5):
            axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
            if row==0:axes[row,j].set_title(('Oracle linear preview','O0 diagnostic','O1 admitted derivative','O0 RGB RMSE','O1 RGB RMSE')[j])
    fig.colorbar(im,ax=axes[:,3:].ravel().tolist(),label='Linear RGB RMSE (display 0–.05)')
    name=('diffuse','specular','rgb')[component]
    fig.suptitle(name+': fixed material / original Bell v3 light; magenta = undefined material transfer')
    fig.savefig(out/(name+'-comparison.png'),dpi=200,bbox_inches='tight');fig.savefig(out/(name+'-comparison.pdf'),bbox_inches='tight');plt.close(fig)
    return clipping


def verify(f,args):
    root=Path(f['host_output']);p=g.paths();g.admit(p)
    require(g.digest(root/'authorization.json')==args.authorization_sha256 and
            g.read(root/'authorization.json')==g.authorization(f,args.commit,args.freeze_sha256),'Detached authorization')
    require((root/'exit-code.txt').read_text().strip()=='0' and
            (root/'container-absent.txt').read_text().strip()=='absent' and
            (root/'image-id.txt').read_text().strip()==f['image'],'Host completion/cleanup/image receipt differs')
    receipt=g.read(root/'render/receipt.json');native=root/'render'
    require(receipt['status']=='complete' and receipt['cleanup_verified'] and receipt['backend']==f['backend'] and
            receipt['commit']==args.commit and receipt['freeze_sha256']==args.freeze_sha256 and
            receipt['authorization_sha256']==args.authorization_sha256 and receipt['handoff_sha256']==f['handoff']['sha256'] and
            set(receipt['views'])==set(g.PANEL) and list(receipt['visibility_fields'])==list(VISIBILITY_FIELDS),
            'Incomplete or foreign Gate 12 receipt')
    for name,identity in receipt['outputs'].items():g.check(native/name,identity)
    require(set(receipt['outputs'])=={x.name for x in native.glob('*.npz')},'Missing/extra native captures')
    vertices,_=g.mesh_arrays(p['oracle']/'geometry/mesh.ply');bounds=np.stack([vertices.min(0),vertices.max(0)]).astype(float)
    consumed=set();work=dict.fromkeys(f['limits'],0);records=[];total=0;transfer_roundoff=[]
    for camera in g.PANEL:
        data=g.prepare(camera,p,bounds)
        for arm,expected in data['transfers'].items():
            name=f'{camera}-{arm}-transfer.npz';consumed.add(name)
            with np.load(native/name,allow_pickle=False) as z:
                require(set(z.files)==set(expected),'Transfer fields missing/extra')
                for k,want in expected.items():
                    got=z[k];want=np.asarray(want)
                    require(got.dtype==want.dtype and got.shape==want.shape and np.isfinite(got).all(),'Invalid transfer dtype/value')
                    if not np.array_equal(got,want):
                        require(got.dtype==np.float64 and np.allclose(got,want,atol=2e-12,rtol=0),'Material/frame transfer replay differs')
                        transfer_roundoff.append(dict(camera=camera,arm=arm,field=k,max_abs=float(abs(got-want).max())))
        undefined_records={}
        for arm,v in data['undefined_views'].items():
            counts=[]
            for size in g.BASE:
                name=f'{camera}-{arm}-undefined-{size}.npz';consumed.add(name)
                with np.load(native/name,allow_pickle=False) as z:
                    a=z['counts'];require(set(z.files)=={'pixels','counts','grid'} and int(z['grid'])==size and
                        np.array_equal(z['pixels'],v['pixels']) and a.shape==(len(v['pixels']),5) and
                        a.dtype==np.int64 and (a>=0).all() and (a<=size*size).all() and
                        np.array_equal(a[:,2],a[:,3]+a[:,4]),'Invalid undefined-hit visibility capture')
                    counts.append(a/(size*size))
                work['shadow_directions']+=2*len(v['pixels'])*size*size
            undefined_records[arm]=dict(observations=len(v['pixels']),
                same_reconstructed_origin_mean_fractions=[a.mean(0).tolist() if len(a) else None for a in counts],
                coarse_fine_mean_absolute_change=float(abs(counts[1]-counts[0]).mean()) if len(v['pixels']) else None,
                material_and_RGB_remain_undefined=True)
        def fetch(arm,v,n,prefix):
            name=capture_name(camera,arm,prefix,n);require(name not in consumed,'Unexpected capture replay')
            consumed.add(name)
            with np.load(native/name,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            validate_capture(v,n,a);cost=len(v['pixels'])*n*n*(1 if arm=='oracle' else 2)
            work['shadow_directions']+=cost;work['BRDF_point_directions']+=cost
            work['environment_point_directions']+=len(v['pixels'])*n*n;return a
        states,selected=g.paired(data['views'],fetch,total);total+=len(selected);saved=receipt['views'][camera]
        require(set(saved)=={'refined_pixels','convergence','oracle_replay'} and saved['refined_pixels']==selected and
                set(saved['convergence'])==set(g.ARMS),'Refinement/arm inventory differs')
        for arm,s in states.items():
            observed=[g.convergence(s['components'][1,:,k,c],s['components'][0,:,k,c]) for k in range(len(g.KINDS)) for c in range(3)]
            require(len(saved['convergence'][arm])==len(observed),'Convergence count differs')
            for a,b in zip(saved['convergence'][arm],observed):gates_equal(a,b)
        s=states['oracle'];gates_equal(saved['oracle_replay'],g.convergence(s['components'][1,:,0,2],data['archived'][s['pixels'],2]))
        records.append(dict(camera=camera,undefined_visibility=undefined_records,**expand(data,states)))
    require(consumed==set(receipt['outputs']) and receipt['counters']==work and
            receipt['refined_pixels_total']==total and all(work[k]<=v for k,v in f['limits'].items()),'Work/capture inventory differs')
    out=root/'analysis';out.mkdir()
    summary=dict(status='complete_verified_pending_scientific_review',protocol=g.PROTOCOL,
        receipt=g.item(native/'receipt.json'),per_camera={v['camera']:metrics([v]) for v in records},
        validation=metrics([v for v in records if v['camera']=='58']),
        held=metrics([v for v in records if v['camera']!='58']),all_views=metrics(records),
        counters=work,transfer_roundoff_events=transfer_roundoff,refined_pixels_total=total,
        undefined_geometry_visibility={v['camera']:v['undefined_visibility'] for v in records},
        interpretation='O0 diagnostic only; O1 perfect-mask derivative; fixed-material differences are not an inverse-rendering lower bound.',
        new_scientific_execution=False)
    summary['visualization_clipping']={name:figure(records,out,k) for k,name in enumerate(('diffuse','specular','RGB'))}
    g.write(out/'metrics.json',summary)
    require(sum(x.stat().st_size for x in root.rglob('*') if x.is_file())<=f['resources']['output_bytes'],'Final output limit')
    g.write(out/'artifact-manifest.json',dict(native_outputs=receipt['outputs'],receipt=g.item(native/'receipt.json'),
        files={x.name:g.item(x) for x in out.iterdir() if x.is_file()},freeze=g.item(g.ROOT/g.FREEZE),handoff=f['handoff']))
    print(out/'metrics.json')


if __name__=='__main__':
    c=argparse.ArgumentParser(description=__doc__)
    for key in ('commit','freeze-sha256','authorization-sha256'):c.add_argument('--'+key,required=True)
    a=c.parse_args();verify(g.freeze(a.freeze_sha256),a)
