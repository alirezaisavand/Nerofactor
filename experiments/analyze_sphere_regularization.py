"""File-only sphere audit: saved maps/logs + small independent CPU quadrature.

No checkpoints are deserialized and no native renderer/model is constructed.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from infrastructure.signatures import require
from infrastructure.stage_sources import digest
from audits import anisotropic_brdf as reference


def q_numpy(a):
    t=2*a['tangents'].astype(np.float64)-1;b=2*a['bitangents'].astype(np.float64)-1
    return a['mx'].astype(float)[...,None]**2*t[..., :,None]*t[...,None,:]+a['my'].astype(float)[...,None]**2*b[..., :,None]*b[...,None,:]


def prior_values(a,mask):
    d=a['f_d'][mask].astype(float);s=a['f_s_sum'][mask].astype(float);ls=a['L_spec'][mask].astype(float)
    return dict(energy=.05*np.maximum(d+s-1,0).sum(-1).mean(),
                specular=.05*ls.sum(-1).mean(),energy_violating_fraction=(np.max(d+s,axis=-1)>1).mean())


def local_specular_gradient(kd,metal,L_spec):
    """Exact local physical partials with legacy detached NDF/masks fixed.

    Does not recover neural gradients or historical stochastic graph/tapes.
    """
    d=kd*(1-metal)
    require((d>0).all(),'Cannot reconstruct detached specular coefficient at zero allocation; no silent zero derivative')
    coefficient=np.divide(L_spec,d,out=np.zeros_like(d),where=d>0)
    return .05*coefficient*(1-metal),-.05*(coefficient*kd).sum(-1)


def quadrature_check(a,mask,count=6):
    """Fixed raster-spaced materials, no favorable material/view selection."""
    flat=np.flatnonzero(mask);chosen=flat[np.linspace(0,len(flat)-1,count,dtype=int)]
    rows=[]
    for pixel in chosen:
        y,x=np.unravel_index(pixel,mask.shape)
        mx,my=float(a['mx'][y,x,0]),float(a['my'][y,x,0])
        t=2*a['tangents'][y,x].astype(float)-1;b=2*a['bitangents'][y,x].astype(float)-1
        n=a['normal'][y,x].astype(float)
        # Source ID36 original c2w is read separately by caller, not a camera pickle.
        wo=a['_camera_center']-a['position'][y,x];wo=wo/np.linalg.norm(wo)
        local=np.array([wo@t,wo@b,wo@n]);canonical_t,canonical_b=reference.frame()
        outgoing=local[0]*canonical_t+local[1]*canonical_b+np.array([0,0,local[2]])
        if outgoing[2]<=0:
            rows.append(dict(pixel=int(pixel),status='blocked_nonpositive_view'));continue
        kd=a['kd'][y,x].astype(float);metal=float(a['metallic'][y,x,0]);d=kd*(1-metal);f0=.04*(1-metal)+metal*kd
        integrals=[];acceptances=[]
        for resolution in (128,256):
            wi,w=reference.hemisphere(resolution,2*resolution)
            spec=np.stack([np.sum(reference.kernel(wi,outgoing,mx,my,f0=c)*wi[:,2]*w) for c in f0])
            acceptance=float(np.sum(reference.directional_pdf(wi,outgoing,mx,my)*w))
            integrals.append(d+spec);acceptances.append(acceptance)
        saved=d+a['f_s_sum'][y,x].astype(float)
        expected_conditional=d+(integrals[-1]-d)/acceptances[-1]
        rows.append(dict(pixel=int(pixel),status='measured',widths=[mx,my],cosine=float(outgoing[2]),
            reflectance_coarse=integrals[0].tolist(),reflectance_fine=integrals[1].tolist(),
            grid_max_absolute_difference=float(np.max(abs(integrals[1]-integrals[0]))),
            two_grid_bound_pass=bool(np.all(abs(integrals[1]-integrals[0])<=.001+.002*abs(integrals[1]))),
            analytical_acceptance=reference.acceptance(outgoing,mx,my,n_phi=4096),
            accepted_mass=acceptances,saved_conditional_energy_proxy=saved.tolist(),
            expected_conditional_limit=expected_conditional.tolist(),
            saved_proxy_minus_true_DHR=(saved-integrals[-1]).tolist(),
            caution='Saved proxy is one finite1024-sample conditional estimate; no bitwise/MC-convergence claim.'))
    return rows


def chart_statistics(a,mask):
    q=q_numpy(a);n=a['normal'].astype(float);p=a['position'].astype(float);r=a['rotation'].astype(float)
    raw=[];physical=[]
    for axis in (0,1):
        s0=[slice(None)]*2;s1=s0.copy();s0[axis]=slice(None,-1);s1[axis]=slice(1,None)
        s0,s1=tuple(s0),tuple(s1)
        ok=mask[s0]&mask[s1]&(np.linalg.norm(p[s0]-p[s1],axis=-1)<.02)&((n[s0]*n[s1]).sum(-1)>.99)
        u,v=n[s1][ok],n[s0][ok];cross=np.cross(u,v);c=(u*v).sum(-1)
        K=np.zeros((len(c),3,3));K[:,0,1]=-cross[:,2];K[:,0,2]=cross[:,1];K[:,1,0]=cross[:,2]
        K[:,1,2]=-cross[:,0];K[:,2,0]=-cross[:,1];K[:,2,1]=cross[:,0]
        R=np.eye(3)[None]+K+(K@K)/(1+c)[:,None,None]
        diff=q[s0][ok]-R@q[s1][ok]@R.transpose(0,2,1)
        physical.extend(np.linalg.norm(diff,axis=(1,2)).tolist())
        raw.extend((.5*np.arccos(np.clip((r[s0][ok]*r[s1][ok]).sum(-1),-1,1))).tolist())
    raw,physical=np.asarray(raw),np.asarray(physical)
    return dict(projected_neighbor_pairs=len(raw),raw_angle_jump_p95_degrees=float(np.degrees(np.quantile(raw,.95))),
        transported_Q_absolute_difference_p95=float(np.quantile(physical,.95)),
        large_raw_jump_small_Q_count=int(((raw>np.pi/4)&(physical<.05)).sum()),
        large_raw_jump_large_Q_count=int(((raw>np.pi/4)&(physical>=.05)).sum()),
        diagnostic_only_thresholds=dict(raw_angle_degrees=45,Q_absolute=.05,world_distance=.02,normal_cosine=.99),
        limitation='Screen-neighbor diagnostic; no source face IDs, so not a certified mesh-surface neighborhood.')


def analyze(batch,output,cameras):
    require(not output.exists(),'Fresh audit output required');output.mkdir(parents=True)
    sphere=batch/'sphere';run=json.loads((sphere/'sphere-replay.json').read_text())
    require(run['commit']=='92d433dd809bab5fe8667a2920dc54ce94071f35' and
            run['effective_shader']['regularization_version']=='legacy_v1','Expected admitted sphere execution')
    camera=json.loads(cameras.read_text())['frames'][36];center=np.asarray(camera['transform_matrix'],np.float32)[:3,3]/2
    records={};rows=[];selected={};quadrature=[]
    for directory in sorted((sphere/'renders').glob('step-*')):
        p=directory/'validation/camera-36/materials.npz'
        if not p.exists():continue
        info=json.loads((p.parent/'view.json').read_text())
        require(digest(p)==info['files']['materials.npz']['sha256'],'Material archive hash mismatch')
        step=int(directory.name.split('-')[1]);records[str(p)]=dict(sha256=digest(p),bytes=p.stat().st_size)
        with np.load(p,allow_pickle=False) as source:
            a={k:source[k] for k in ('kd','metallic','mx','my','f_d','f_s_sum','L_spec','linear_diffuse','linear_specular',
                'rgb_gt','rgb_pr','diffuse_color','specular_color','diffuse_light','specular_light','normal','position','tangents','bitangents','rotation','rotation_raw','hit_mask')}
        m=a['hit_mask'];kd=a['kd'][m].astype(float);metal=a['metallic'][m].astype(float);d=kd*(1-metal)
        require(m.any() and np.allclose(d,a['f_d'][m],atol=1e-7),'Diffuse allocation mismatch')
        ld=a['linear_diffuse'][m].astype(float);ls=a['linear_specular'][m].astype(float)
        light=np.divide(ld,d,out=np.full_like(ld,np.nan),where=d>1e-8)
        gkd,gm=local_specular_gradient(kd,metal,a['L_spec'][m].astype(float))
        row=dict(step=step,pixels=int(m.sum()),kd_red=kd[:,0].mean(),kd_mean=kd.mean(),metallic=metal.mean(),
            diffuse_allocation_red=d[:,0].mean(),diffuse_red=ld[:,0].mean(),specular_red=ls[:,0].mean(),
            effective_diffuse_incident_red=np.nanmean(light[:,0]),
            mx_mean=a['mx'][m].astype(float).mean(),my_mean=a['my'][m].astype(float).mean(),
            raw_orientation_norm_p05=np.quantile(np.linalg.norm(2*a['rotation_raw'][m]-1,axis=-1),.05),
            specular_direct_dkd_l2=np.linalg.norm(gkd)/len(kd),specular_direct_dmetal_l2=np.linalg.norm(gm)/len(kd),
            **prior_values(a,m))
        rows.append(row)
        if step in (2000,10000,20000,40000):
            a['_camera_center']=center
            selected[step]=a
            quadrature.append(dict(step=step,rows=quadrature_check(a,m),chart=chart_statistics(a,m)))
    checkpoints=[]
    for p in sorted((sphere/'checkpoints').glob('step-*.json')):
        identity=json.loads(p.read_text());checkpoint=p.with_suffix('.pth')
        require(digest(checkpoint)==identity['checkpoint_sha256'],'Checkpoint identity mismatch')
        checkpoints.append(dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,**identity))
    with (output/'progression.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    logs=[json.loads(line) for line in (sphere/'training.jsonl').read_text().splitlines()]
    report=dict(schema_version=1,status='offline_audit_complete',source_run_status=run['status'],
        source_optimizer_returns=run['optimizer_returns'],source_report_sha256=digest(sphere/'sphere-replay.json'),
        effective_shader=run['effective_shader'],artifacts=records,checkpoints=checkpoints,progression=rows,quadrature=quadrature,
        declared_vs_effective_energy_specular=dict(top_level=[.01,.01],effective_shader=[.05,.05],mode='legacy_v1'),
        unavailable=['Historical per-component neural/illumination gradients and their directions were not saved.',
                     'Training minibatch spatial/energy/specular split and sampled-neighbor graphs were not saved.',
                     'Full HDR learned environment and per-sample incident radiance were not saved; diffuse irradiance ratio is an effective sample mean.',
                     'No independent material ground truth for this sphere.'],
        derivative_scope='Analytical local physical partials of archived detached-NDF specular loss, not historical network gradients',
        quadrature_rule='Gauss-Legendre mu / uniform midpoint azimuth at128x256 and256x512; .001+.002|fine| two-grid bound, no absolute-accuracy proof',
        new_native_model_calls=0,new_GPU_work=0,checkpoint_deserialization=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    figures(rows,selected,logs,output)
    return report


def figures(rows,selected,logs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[r['step'] for r in rows];fig,axes=plt.subplots(2,3,figsize=(13,7))
    for axis,keys,title in zip(axes.flat,[['kd_red','diffuse_allocation_red'],['metallic','mx_mean','my_mean'],
         ['effective_diffuse_incident_red'],['diffuse_red','specular_red'],['energy','specular'],['specular_direct_dkd_l2','specular_direct_dmetal_l2']],
         ['Material red channel','Material scalar estimates','Effective red diffuse illumination','Linear red contributions','Archived validation prior diagnostics','Local physical prior partials (not neural gradients)']):
        for key in keys:axis.plot(x,[r[key] for r in rows],label=key)
        axis.set_title(title,fontsize=9);axis.legend(fontsize=7);axis.set_xlabel('Completed updates')
    fig.tight_layout();fig.savefig(output/'progression.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(len(selected),6,figsize=(15,3*len(selected)))
    for row,(step,a) in enumerate(selected.items()):
        fields=[np.clip(a['rgb_pr'],0,1),np.clip(a['kd'],0,1),np.clip(a['metallic'][...,0],0,1),
                np.clip(a['diffuse_color'],0,1),np.clip(a['specular_color'],0,1),
                ((2*a['tangents']-1)*.5+.5)*a['hit_mask'][...,None]]
        for col,(image,label) in enumerate(zip(fields,['RGB','kd (linear display)','metallic','diffuse sRGB','specular sRGB','world axis (sign ambiguous)'])):
            axes[row,col].imshow(image[::4,::4],vmin=0,vmax=1,cmap='gray' if image.ndim==2 else None)
            axes[row,col].set_title(str(step)+' '+label,fontsize=8);axes[row,col].axis('off')
    fig.tight_layout();fig.savefig(output/'saved-map-progression.png',dpi=130);plt.close(fig)
    fig,axis=plt.subplots(figsize=(8,4))
    for key in ('loss_rgb','loss_mat_reg','loss_diffuse_light'):
        selected_logs=[r for r in logs if r['prefix']=='train' and key in r]
        axis.plot([r['completed_updates'] for r in selected_logs],[r[key] for r in selected_logs],label=key)
    axis.set_yscale('log');axis.legend();axis.set_xlabel('Completed updates');axis.set_ylabel('Actual logged weighted loss')
    fig.tight_layout();fig.savefig(output/'logged-losses.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cameras',type=Path,required=True)
    args=parser.parse_args();r=analyze(args.batch,args.output,args.cameras)
    print(json.dumps(dict(status=r['status'],views=len(r['progression']),checkpoints=len(r['checkpoints']))))
