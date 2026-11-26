"""Package checked Gate 7 endpoints, comparative tables and physical diagnostics."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from benchmarks import donut_nero_gate7_v1 as g
from benchmarks.donut_recovery_v1 import json_new, numeric
from experiments.analyze_donut_relighting_gate6_v1 import scores
from infrastructure.signatures import require


def identity(path):
    return dict(bytes=path.stat().st_size,sha256=g.digest(path))


def known_light_project(views):
    prior=g.read(g.GATE5_MANIFEST)
    project={}
    for arm in g.gate6.ARMS:
        path=Path(prior['arms'][arm]['receipt']['path'])
        receipt=g.read(g.check(path,prior['arms'][arm]['receipt']))
        pred={}
        for camera,v in views.items():
            name=f'step-10000/camera-{camera}/materials.npz'
            maps=numeric(g.check(path.parent/name,receipt['outputs'][name]))
            mask=maps['hit_mask']
            np.testing.assert_array_equal(np.flatnonzero(mask),v['pixel_indices'])
            pred[camera]=np.stack([maps[k][mask] for k in ('linear_diffuse','linear_specular','linear_rgb')],axis=1).astype(float)

        def measure(cameras):
            prediction=np.concatenate([pred[c] for c in cameras])
            truth=np.concatenate([views[c]['linear_components'] for c in cameras])
            return scores(prediction,truth,np.ones(len(truth),bool))

        project[arm]={**{str(c):measure([c]) for c in views},
                      **{split:measure(cams) for split,cams in g.m.SPLITS.items()}}
    return project


def summary_rows(data,project_L2):
    rows=[]

    def append(light,arm,group,values):
        for scope,components in values.items():
            for component,x in components.items():
                metric=x if x is not None else dict(pixels=0,rmse=None,relative_L2=None,PSNR_unit_peak_dB=None,exact=None)
                rows.append(dict(light=light,arm=arm,group=group,scope=scope,component=component,**metric))

    for light,panel in data['upstream'].items():
        for group,row in {**panel['by_view'],**panel['pooled']}.items():
            append(light,'NeRO',group,row['appearance'])
    for arm,panels in project_L2.items():
        for group,x in panels.items():
            known=data['comparable_project_L2'][arm]
            source=known[group]['all'] if group in g.m.SPLITS else known['cameras'][group]['all']
            np.testing.assert_allclose([x['RGB']['relative_L2'],x['RGB']['rmse']],
                [source['RGB_relative_L2'],source['RGB_RMSE']],rtol=0,atol=1e-12)
            assert x['RGB']['pixels']==source['observations']
            append('L2',arm,group,{'common':x})
    for arm,panels in data['comparable_project_unseen_native'].items():
        for group,scopes in {**panels['views'],'held_out':panels['pooled']}.items():
            append('unseen',arm,group,scopes)
    require(len({tuple(r[k] for k in ('light','arm','group','scope','component')) for r in rows})==len(rows),
            'Duplicate numerical table row')
    return rows


def material_diagnostics(run,views,unseen):
    y=np.array([.2126,.7152,.0722])
    diagnostics={}
    for light,split,cameras in [('L2',s,cams) for s,cams in g.m.SPLITS.items()]+[
        ('unseen','held_out',g.gate6.CAMERAS)]:
        predicted,truth,metal,metal_truth=[],[],[],[]
        for camera in cameras:
            d=numeric(run/f'{light}-camera-{camera}.npz')
            pixels=d['pixels']
            if light=='unseen':
                support=unseen[camera]['masks']['common'][pixels]
                r=unseen[camera]['reference'][pixels]
                field=unseen[camera]['view']
            else:
                support=np.ones(len(pixels),bool)
                field=views[camera]
                r=field['linear_components']
            np.testing.assert_array_equal(pixels,field['pixel_indices'])
            predicted.append(d['components'][support])
            truth.append(r[support])
            metal.append(d['metallic'][support,0])
            metal_truth.append(field['metallic'][support,0])
        m,mo=np.concatenate(metal),np.concatenate(metal_truth)
        a,b=np.concatenate(predicted),np.concatenate(truth)
        diagnostics[f'{light}_{split}']=dict(observations=len(m),metallic_median=float(np.median(m)),
            metallic_above_098=int((m>.98).sum()),oracle_metallic_median=float(np.median(mo)),
            oracle_metallic_above_098=int((mo>.98).sum()),predicted_component_mean_Y=(a@y).mean(0).tolist(),
            oracle_component_mean_Y=(b@y).mean(0).tolist(),maximum_linear_RGB=float(a[:,2].max()))
    return diagnostics


def package(sha,analysis,out):
    f=g.freeze(sha)
    batch=Path(f['host_output'])
    native=batch/'run/receipt.json'
    r=g.read(native)
    verified=g.read(analysis/'artifact-manifest.json')
    report=g.read(analysis/'comparison.json')
    a=g.read(batch/'authorization.json')
    require(a==g.authorization(f,a['commit'],sha) and r['commit']==a['commit']
            and (batch/'exit-code.txt').read_text().strip()=='0' and r['status']=='complete' and r['cleanup_verified']
            and g.digest(batch/'freeze.json')==r['freeze_sha256']==verified['freeze_sha256']==report['freeze_sha256']==sha
            and verified['authorization_sha256']==r['authorization_sha256']==g.digest(batch/'authorization.json')
            and g.digest(native)==verified['native_receipt_sha256']==report['receipt_sha256']
            and r['outputs']==verified['native_files'] and r['counters']==report['native_work']==g.LIMITS,
            'Exact verified external baseline required')
    for name,item in r['outputs'].items():
        g.check(native.parent/name,item)
    for name,item in verified['files'].items():
        g.check(analysis/name,item)
    views,unseen,_=g.admit(f,g.paths(f))
    rows=summary_rows(report,known_light_project(views))
    diag=material_diagnostics(native.parent,views,unseen)
    require(out.resolve()==out and not out.exists() and out.parent.is_dir() and batch not in out.parents
            and analysis not in out.parents,'Fresh Gate 7 publication directory required')
    out.mkdir()
    for name in verified['files']:
        shutil.copyfile(analysis/name,out/name)
        require(identity(out/name)==identity(analysis/name),'Packaged analysis copy differs')
    with (out/'image-metrics.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    json_new(out/'material-diagnostics.json',dict(component_order=['diffuse','specular','RGB'],
             data=diag,observed_fit='RGB fit and near-unit metallic are endpoint observations; no causal intervention'))
    json_new(out/'artifact-manifest.json',dict(
        **{k:v for k,v in verified.items() if k!='files'},status='verified_complete',
        prior_analysis=dict(identity(analysis/'artifact-manifest.json'),path=str(analysis/'artifact-manifest.json')),
        packaged_files={p.name:identity(p) for p in sorted(out.iterdir())},
        native_counters=r['counters'],elapsed_seconds=r['elapsed_seconds'],
        trainable_parameters=r['trainable_parameters'],cleanup_verified=r['cleanup_verified'],
        offline_shader_calls=0,offline_checkpoint_loads=0))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    package(args.freeze_sha256,args.analysis,args.output)
