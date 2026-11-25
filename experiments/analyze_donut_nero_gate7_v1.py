"""Verify and score Gate 7 against archived project outcomes without re-rendering."""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_nero_gate7_v1 as g
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.analyze_donut_relighting_gate6_v1 import raster, scores
from infrastructure.signatures import require


def verify(sha,out):
    f=g.freeze(sha)
    batch=Path(f['host_output'])
    run=batch/'run'
    auth_path=batch/'authorization.json'
    auth=g.read(auth_path)
    r=g.read(run/'receipt.json')
    require(g.digest(batch/'freeze.json')==sha and auth==g.authorization(f,auth['commit'],sha)
            and g.digest(auth_path)==g.digest(batch.with_name(batch.name+'-authorization.json'))
            and (batch/'exit-code.txt').read_text().strip()=='0'
            and r['status']=='complete' and r['cleanup_verified'] and r['frozen_endpoint_weights_unchanged']
            and r['protocol']==g.PROTOCOL and r['freeze_sha256']==sha and r['commit']==auth['commit']
            and r['authorization_sha256']==g.digest(auth_path) and r['counters']==g.LIMITS
            and r['backend']==f['backend'] and r['upstream_commit']==g.UPSTREAM_COMMIT
            and r['elapsed_seconds']<g.RESOURCES['inner_seconds']
            and r['host_peak_rss_bytes']<g.RESOURCES['host_memory_gib']*1024**3
            and r['known_light_directions']>0 and r['initial_forward_backward']['finite'],
            'Incomplete or foreign upstream NeRO execution')
    require(set(r['outputs'])=={p.name for p in run.iterdir() if p.is_file() and p.name!='receipt.json'}
            and sum(v['bytes'] for v in r['outputs'].values())<=g.RESOURCES['output_bytes'],'Output inventory differs')
    for name,item in r['outputs'].items():
        g.check(run/name,item)
    require(all(f['upstream']['files'][v['path']]['sha256']==v['sha256'] for v in r['imported_sources'].values())
            and {'network.field','network.renderer','network.loss','train.lr_common_manager','raytracing.raytracer'}
                <=set(r['imported_sources']),'Native import provenance differs')
    stream=run/'fit-indices.bin'
    log=np.loadtxt(run/'training.csv',delimiter=',',skiprows=1)
    require(g.digest(stream)==r['fitting_index_sha256']==f['support']['photometric_index_sha256']
            and stream.stat().st_size==g.STEPS*g.BATCH*4 and log.shape==(g.STEPS,5) and np.isfinite(log).all()
            and np.array_equal(log[:,0],np.arange(1,g.STEPS+1)),'Fitting evidence differs')
    p=g.paths(f)
    views,unseen,_=g.admit(f,p)
    accepted5=g.read(g.GATE5_MANIFEST.parent/'representation-results.json')
    accepted6=g.read(g.GATE6_MANIFEST.parent/'relighting-results.json')
    panels=assemble(f,p,r,run,views,unseen)
    out=out.resolve()
    require(not out.exists() and out.parent.is_dir() and batch not in out.parents,'Fresh analysis output required')
    out.mkdir()
    result=dict(protocol=g.PROTOCOL,status='verified_complete',freeze_sha256=sha,receipt_sha256=g.digest(run/'receipt.json'),
                native_work=r['counters'],elapsed_seconds=r['elapsed_seconds'],runtime=r['runtime'],
                comparable_project_L2=accepted5['rows'][-1]['arms'],comparable_project_unseen_native={
                    arm:{'pooled':a['native'],'views':{c:v['native'][arm] for c,v in accepted6['views'].items()}}
                    for arm,a in accepted6['pooled'].items()}, upstream={},linear_PSNR_peak=1,
                training_domain=f['objective_domain'],roughness_conversion=None,
                offline_checkpoint_loads=0,offline_shader_calls=0)
    for light,data in panels.items():
        by_view={str(c):measure(v,v['masks']) for c,v in data.items()}
        groups=g.m.SPLITS if light=='L2' else {'held_out':g.gate6.CAMERAS}
        pooled={}
        for split,cameras in groups.items():
            merged={key:np.concatenate([data[c][key] for c in cameras]) for key in
                    ('pred','ref','color_error','metallic_error','roughness_native')}
            masks={key:np.concatenate([data[c]['masks'][key] for c in cameras]) for key in data[cameras[0]]['masks']}
            pooled[split]=measure(merged,masks)
        result['upstream'][light]=dict(by_view=by_view,pooled=pooled)
    result['display']=plot_unseen(panels['unseen'],out)
    json_new(out/'comparison.json',result)
    json_new(out/'artifact-manifest.json',dict(protocol=g.PROTOCOL,execution_commit=r['commit'],freeze_sha256=sha,
        authorization_sha256=g.digest(auth_path),native_receipt_sha256=g.digest(run/'receipt.json'),
        native_output=str(batch),native_files=r['outputs'],upstream_commit=g.UPSTREAM_COMMIT,
        prior_gate5_sha256=f['gate5_manifest_sha256'],prior_gate6_sha256=f['gate6_result_manifest_sha256'],
        files={x.name:dict(bytes=x.stat().st_size,sha256=g.digest(x)) for x in sorted(out.iterdir())}))


def assemble(f,p,receipt,run,views,unseen):
    panels={}
    expected={}
    for light,states in (('L2',views),('unseen',{c:v['view'] for c,v in unseen.items()})):
        a={}
        for camera,v in states.items():
            name=f'{light}-camera-{camera}.npz'
            d=numeric(run/name)
            pix=v['pixel_indices']
            expected[name]=len(pix)
            require(set(d)=={'pixels','components','albedo','metallic','roughness_native'}
                    and np.array_equal(d['pixels'],pix) and d['components'].shape==(len(pix),3,3)
                    and (d['components']>=0).all() and np.array_equal(d['components'][:,0]+d['components'][:,1],d['components'][:,2])
                    and d['albedo'].shape==(len(pix),3) and d['metallic'].shape==d['roughness_native'].shape==(len(pix),1)
                    and all(((d[k]>=0)&(d[k]<=1)).all() for k in ('albedo','metallic','roughness_native')),
                    'Malformed external baseline endpoint')
            mask=raster(np.ones(len(pix),bool),pix)
            masks={'common':mask} if light=='L2' else unseen[camera]['masks']
            ref=raster(v['linear_components'],pix) if light=='L2' else unseen[camera]['reference']
            a[camera]=dict(pred=raster(d['components'].astype(np.float64),pix),ref=ref,masks=masks,
                           color_error=raster(abs(d['albedo']-v['color']).max(1),pix),
                           metallic_error=raster(abs(d['metallic']-v['metallic']).reshape(-1),pix),
                           roughness_native=raster(d['roughness_native'].reshape(-1),pix))
            if light=='unseen':
                archive=numeric(g.check(p[f'unseen_{camera}'],f['panels'][str(camera)]))
                a[camera]['project_native']={arm:archive[arm+'_native'] for arm in g.gate6.ARMS}
        panels[light]=a
    require(receipt['views']==expected and set(receipt['outputs'])==set(expected)|{
        'fit-indices.bin','training.csv','checkpoint-10000.pth'},'Expected exactly one external baseline endpoint')
    return panels


def measure(panel,masks):
    common=masks['common']
    return dict(appearance={k:scores(panel['pred'],panel['ref'],m) for k,m in masks.items()},
                common_scalar_material=dict(albedo_max_channel_MAE=float(panel['color_error'][common].mean()),
                    metallic_MAE=float(panel['metallic_error'][common].mean()),
                    roughness_native_quantiles=np.quantile(panel['roughness_native'][common],[.05,.5,.95]).tolist(),
                    anisotropic_width_Q_orientation='unavailable for isotropic GGX'))


def plot_unseen(panels,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmarks.bell_e11 import linear_to_srgb

    arms=('K1-Oracle','Final-Chart-Oracle','Ambient-Oracle')
    blocks={}
    for camera,v in panels.items():
        blocks[camera]=[v['ref'][:,2],v['pred'][:,2],*[v['project_native'][a][:,2] for a in arms]]
    errors={c:[np.sqrt(np.mean((rgb-block[0])**2,axis=1)) for rgb in block[1:]] for c,block in blocks.items()}
    limit=max(float(max(x.max() for seq in errors.values() for x in seq)),1e-15)
    clipped={}
    fig,axes=plt.subplots(4,5,figsize=(12,8.9),layout='constrained')
    for i,(c,block) in enumerate(blocks.items()):
        for j,(name,rgb) in enumerate(zip(('Oracle','NeRO','K1','Final-Chart','Final-Ambient'),block)):
            clipped[f'{c}/{name}']=int((rgb>1).sum())
            axes[2*i,j].imshow(linear_to_srgb(np.minimum(rgb.reshape(g.SIZE,g.SIZE,3),1)),interpolation='nearest')
            if i==0:
                axes[0,j].set_title(name)
            if j==0:
                axes[2*i+1,j].imshow(panels[c]['masks']['common'].reshape(g.SIZE,g.SIZE),cmap='gray',vmin=0,vmax=1)
            else:
                bar=axes[2*i+1,j].imshow(errors[c][j-1].reshape(g.SIZE,g.SIZE),cmap='magma',vmin=0,vmax=limit)
        axes[2*i,0].set_ylabel(f'Camera {c}')
        axes[2*i+1,0].set_ylabel('Primary mask')
        fig.colorbar(bar,ax=axes[2*i+1,1:].tolist(),shrink=.75,label='Linear RGB pixel RMSE')
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for suffix in ('png','pdf'):
        fig.savefig(out/f'held-out-native-relighting.{suffix}',dpi=200)
    plt.close(fig)
    return dict(error_max=limit,preview_clipped_channels=clipped,exposure=1)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    verify(a.freeze_sha256,a.output)
