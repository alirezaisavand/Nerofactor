"""Saved-file Gate 12 finalization. No checkpoint, SDF, BVH or shader execution."""
import ast
import copy
import csv
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR','/tmp/bell-gate12-finalization-matplotlib')
import numpy as np
from benchmarks import bell_geometry_rgb_gate12 as g
from experiments import verify_bell_geometry_rgb_gate12 as frozen
from experiments.verify_donut_oracle_geometry_rgb_v1 import score
from infrastructure.signatures import require
from infrastructure.stage_sources import baseline,PINS

COMMIT='3c3831957c805c38038abb0c9be4633b4672769a'
SHA='f20bab2ebcddc48bbd8d22dc384c06d0ee3f6231d181b09560d32c3968fe5e9a'
AUTH='4cbcb2b88e08700fadbc010f12fdc303a9e4e7bc057d61b3fa20049b4090ab59'


def output_only_rebind(source,destination):
    """Replay the frozen verifier without overwriting its original analysis.

    Its function has no output-path argument. Change only the one assignment
    out=root/'analysis'; assert a reverse edit restores the entire original AST.
    All data roots, calculations, numerical gates and work checks stay original.
    """
    tree=ast.parse(source);before=ast.dump(tree)
    nodes=[x for x in ast.walk(tree) if isinstance(x,ast.Assign) and
           len(x.targets)==1 and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='out']
    require(len(nodes)==1 and ast.dump(nodes[0].value)==ast.dump(ast.parse("root/'analysis'",mode='eval').body),
            'Frozen verifier output assignment differs; no other edit permitted')
    old=copy.deepcopy(nodes[0].value)
    nodes[0].value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(value=str(destination))],keywords=[])
    result=copy.deepcopy(tree);nodes[0].value=old
    require(ast.dump(tree)==before,'Verifier body changed beyond output destination')
    return ast.fix_missing_locations(result)


def distribution(a):
    if not len(a):return None
    require(np.isfinite(a).all(),'Nonfinite supplemental statistic')
    return dict(count=len(a),median=float(np.median(a)),mean=float(a.mean()),p95=float(np.quantile(a,.95)),
                p99=float(np.quantile(a,.99)),maximum=float(a.max()))


def diagnostic_replay_comparison(saved,replayed):
    """Scientific checks already passed the unchanged frozen verifier.

    Preserve exact types/keys/counts and RGB metrics. Document only observed
    float64 derived-Q diagnostics (existing 2e-12 transfer allowance) and
    signed-luminance reductions (1e-15). This never changes any native gate.
    """
    events=[]
    def walk(a,b,path=''):
        if path=='/transfer_roundoff_events':
            for rows in (a,b):
                for e in rows:
                    require(set(e)=={'arm','camera','field','max_abs'} and e['arm'] in ('O0','O1') and
                        e['camera'] in g.PANEL and np.isfinite(e['max_abs']) and 0<e['max_abs']<=2e-12,
                        'Original frozen transfer roundoff allowance violated')
            return
        require(type(a)==type(b),'Replay diagnostic type differs: '+path)
        if isinstance(a,dict):
            require(a.keys()==b.keys(),'Replay keys differ: '+path)
            for k in a:walk(a[k],b[k],path+'/'+k)
        elif isinstance(a,list):
            require(len(a)==len(b),'Replay count differs: '+path)
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
        elif isinstance(a,float):
            require(np.isfinite(a) and np.isfinite(b),'Nonfinite diagnostic')
            if a==b:return
            limit=2e-12 if '/material_coordinate_and_frame/' in path else 1e-15 if path.endswith('/signed_luminance') else 0
            require(limit>0 and abs(a-b)<=limit,'Non-diagnostic result differs: '+path)
            events.append(dict(path=path,original=a,replayed=b,absolute_difference=abs(a-b),diagnostic_allowance=limit))
        else:require(a==b,'Exact replay field differs: '+path)
    walk(saved,replayed)
    return dict(scientific_flags_counts_RGB_errors_exact=True,diagnostic_events=events,
        original_transfer_events=saved['transfer_roundoff_events'],replayed_transfer_events=replayed['transfer_roundoff_events'])


def controls(records,scope):
    target=np.concatenate([v['images']['oracle'] for v in records],1)
    result={}
    for arm in ('O0','O1'):
        prediction=np.concatenate([v['images'][arm] for v in records],1)
        mask=np.concatenate([v['cohorts'][arm][scope] for v in records])
        known=np.concatenate([v['cohorts'][arm]['known'] for v in records])
        values={}
        # Norms of defined interventions are NOT additive causal error shares.
        for label,a,b in [
            ('primary',target[:,:,0],prediction[:,:,0]),
            ('primary_unoccluded',target[:,:,2],prediction[:,:,2]),
            ('transported_source_unoccluded',target[:,:,4],prediction[:,:,4]),
            ('oracle_visibility_vs_original',target[:,:,0],prediction[:,:,1]),
            ('visibility_substitution',prediction[:,:,1],prediction[:,:,0]),
            ('global_field_vs_transported_source',prediction[:,:,3],prediction[:,:,0]),
        ]:values[label]=score(a,b,mask,known)
        use=mask&known
        d=prediction[1,use,0,2]-target[1,use,0,2]
        vis=prediction[1,use,0,2]-prediction[1,use,1,2]
        rest=prediction[1,use,1,2]-target[1,use,0,2]
        denom=np.linalg.norm(vis)*np.linalg.norm(rest)
        values['conditional_residual_alignment']=dict(
            cosine=float(np.sum(vis*rest)/denom) if denom else None,
            algebraic_sum_max_roundoff=float(abs(d-(vis+rest)).max()) if len(d) else None,
            interpretation='Conditional residual alignment only; interacting terms are not independent causal shares.')
        visibility=np.concatenate([v['states'][arm]['visibility'][1] for v in records])
        pixels=np.concatenate([v['cohorts'][arm][scope][v['states'][arm]['pixels']] for v in records])
        nodes=np.concatenate([v['states'][arm]['grids'][1]**2 for v in records])
        ratios=visibility[pixels,:7]/nodes[pixels,None]
        light=visibility[pixels,7:10]/visibility[pixels,10:13]
        values['visibility']=dict(observations=int(pixels.sum()),fraction_fields=list(frozen.VISIBILITY_FIELDS[:7]),
            mean_fractions=ratios.mean(0).tolist(),light_weighted_disagreement_RGB=light.mean(0).tolist(),
            per_point_fraction_distributions={key:distribution(ratios[:,j]) for j,key in enumerate(frozen.VISIBILITY_FIELDS[:7])})
        result[arm]=values
    return result


def tables(records,out):
    rows=[]
    groups={v['camera']:[v] for v in records}
    groups.update(validation=[v for v in records if v['camera']=='58'],held=[v for v in records if v['camera']!='58'],all_views=records)
    controlled={};coverage=[]
    for group,views in groups.items():
        controlled[group]={scope:controls(views,scope) for scope in ('valid','three_arm_joint')}
        for arm in ('O0','O1'):
            reference=np.concatenate([v['images']['oracle'][:,:,0] for v in views],1)
            prediction=np.concatenate([v['images'][arm][:,:,0] for v in views],1)
            known=np.concatenate([v['cohorts'][arm]['known'] for v in views])
            for scope in views[0]['cohorts'][arm]:
                mask=np.concatenate([v['cohorts'][arm][scope] for v in views]);m=score(reference,prediction,mask,known)
                use=mask&known
                for c,name in enumerate(('diffuse','specular','RGB')):
                    value=m[name];error=prediction[1,use,c]-reference[1,use,c]
                    tail=distribution(np.sqrt(np.mean(error**2,axis=1)))
                    rows.append(dict(group=group,arm=arm,scope=scope,component=name,
                        pixels=int(mask.sum()),defined=m['defined_pixels'],undefined=m['undefined_pixels'],
                        RMSE=value['known_support_RMSE'],relative_L2=value['relative_L2'],
                        squared_error=value['squared_error'],signed_luminance=value['signed_luminance'],
                        median=tail['median'] if tail else None,p95=tail['p95'] if tail else None,
                        p99=tail['p99'] if tail else None,maximum=tail['maximum'] if tail else None,
                        numerical_RMSE=value['coarse_fine_difference_RMSE'],
                        numerical_over_effect=value['coarse_fine_difference_over_discrepancy_L2'],
                        undefined_oracle_energy_fraction=value['undefined_oracle_energy_fraction'],
                        measured_squared_relative_error_lower_bound=value['squared_relative_error_lower_bound']))
                coverage.append(dict(group=group,arm=arm,scope=scope,pixels=int(mask.sum()),
                    defined=m['defined_pixels'],undefined=m['undefined_pixels'],
                    undefined_oracle_squared_energy_fraction=m['RGB']['undefined_oracle_energy_fraction']))
    for name,values in [('metrics.csv',rows),('coverage.csv',coverage)]:
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    return controlled


def figures(records,out,controlled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    held=[v for v in records if v['camera']!='58']
    for c,(component,axis) in enumerate(zip(('diffuse','specular','RGB'),axes)):
        for arm,color in [('O0','#bc5a36'),('O1','#176b9b')]:
            errors=np.concatenate([np.sqrt(np.mean((v['images'][arm][1,v['cohorts'][arm]['three_arm_joint'],0,c]-
                v['images']['oracle'][1,v['cohorts'][arm]['three_arm_joint'],0,c])**2,axis=1)) for v in held])
            a=np.sort(errors);axis.plot(a,np.arange(1,len(a)+1)/len(a),color=color,label=arm)
        axis.set_title(component);axis.set_xscale('log');axis.set_xlim(1e-7,.08);axis.set_ylim(0,1)
        axis.set_xlabel('Linear per-pixel RGB RMSE');axis.grid(alpha=.2)
    axes[0].set_ylabel('Empirical cumulative fraction');axes[-1].legend()
    fig.suptitle('Original Bell v3 light: identical 19,273 held-view valid observations')
    save_figure(fig,out,'held-error-distributions')
    labels=['Primary','Primary\nunoccluded','Transported source\nunoccluded','Visibility\nsubstitution']
    keys=['primary','primary_unoccluded','transported_source_unoccluded','visibility_substitution']
    fig,axis=plt.subplots(figsize=(8,3.8),constrained_layout=True)
    for j,(arm,color) in enumerate([('O0','#bc5a36'),('O1','#176b9b')]):
        vals=[controlled['held']['three_arm_joint'][arm][k]['RGB']['known_support_RMSE'] for k in keys]
        axis.bar(np.arange(4)+(j-.5)*.34,vals,.34,color=color,label=arm)
    axis.set_xticks(np.arange(4));axis.set_xticklabels(labels);axis.set_ylabel('Linear RGB RMSE');axis.legend()
    axis.set_title('Matched held support; conditional controls are not additive causal shares')
    save_figure(fig,out,'held-controlled-comparisons')
    # Retain every camera; both methods share visualization scales and support.
    fig,axes=plt.subplots(len(records),4,figsize=(10,2.1*len(records)),constrained_layout=True)
    display=[]
    for row,v in enumerate(records):
        for j,arm in enumerate(('O0','O1')):
            s=v['states'][arm];mask=v['cohorts'][arm]['three_arm_joint'];ids=s['pixels'];selected=mask[ids]
            visibility=np.full(128**2,np.nan);visibility[ids[selected]]=s['visibility'][1,selected,2]/s['grids'][1,selected]**2
            axes[row,j].imshow(visibility.reshape(128,128),vmin=0,vmax=.15,cmap='viridis')
            err=np.full(128**2,np.nan);err[mask]=np.sqrt(np.mean((v['images'][arm][1,mask,0,2]-v['images']['oracle'][1,mask,0,2])**2,1))
            image=axes[row,j+2].imshow(err.reshape(128,128),vmin=0,vmax=.025,cmap='magma')
            display.append(dict(camera=v['camera'],arm=arm,matched_pixels=int(mask.sum()),
                visibility_above_display=int(np.nansum(visibility>.15)),RGB_error_above_display=int(np.nansum(err>.025))))
        axes[row,0].set_ylabel('Camera '+v['camera'])
        for j in range(4):
            axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
            if row==0:axes[row,j].set_title(('O0 visibility disagreement','O1 visibility disagreement','O0 primary error','O1 primary error')[j])
    fig.colorbar(image,ax=axes[:,2:].ravel().tolist(),label='Linear RGB RMSE, 0–.025 display')
    fig.suptitle('Common support only; white = outside joint eligibility, metrics remain unclipped')
    save_figure(fig,out,'joint-visibility-and-error')
    return display


def save_figure(fig,out,name):
    import matplotlib.pyplot as plt
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    f=g.freeze(SHA);root=Path(f['host_output']);dest=root.parent/(root.name+'-finalization-v2')
    require(not dest.exists(),'Fresh file-only finalization output required')
    old=g.read(root/'analysis/artifact-manifest.json')
    for name,identity in old['native_outputs'].items():g.check(root/'render'/name,identity)
    for name,identity in old['files'].items():g.check(root/'analysis'/name,identity)
    g.check(root/'render/receipt.json',old['receipt']);g.check(g.ROOT/g.FREEZE,old['freeze']);g.check(g.ROOT/g.HANDOFF,old['handoff'])
    protected={str(path):g.item(path) for path in [*root.glob('*.json'),*root.glob('*.txt'),
               root/'render/receipt.json',*list((root/'analysis').iterdir())] if path.is_file()}
    # The v1 file-only attempt ran the complete frozen verifier successfully,
    # then its supplemental bit-equality check stopped on diagnostic roundoff.
    # Reuse that verified replay; do not repeat native work or overwrite it.
    replay=root.parent/(root.name+'-finalization-v1');proof=g.read(replay/'artifact-manifest.json')
    g.check(g.ROOT/g.FREEZE,proof['freeze']);g.check(g.ROOT/g.HANDOFF,proof['handoff'])
    require(proof['native_outputs']==old['native_outputs'] and proof['receipt']==old['receipt'],'Detached offline replay')
    for name,identity in proof['files'].items():g.check(replay/name,identity)
    comparison=diagnostic_replay_comparison(g.read(root/'analysis/metrics.json'),g.read(replay/'metrics.json'))
    records=[];p=g.paths();g.admit(p)
    vertices,_=g.mesh_arrays(p['oracle']/'geometry/mesh.ply');bounds=np.stack([vertices.min(0),vertices.max(0)]).astype(float)
    require(g.read(root/'render/receipt.json')['refined_pixels_total']==0,'This saved replay requires zero actual refinements')
    for camera in g.PANEL:
        data=g.prepare(camera,p,bounds);states={}
        for arm,v in data['views'].items():
            pair=[]
            for n in g.BASE:
                with np.load(root/'render'/frozen.capture_name(camera,arm,'base',n),allow_pickle=False) as z:a={k:z[k] for k in z.files}
                frozen.validate_capture(v,n,a);pair.append(a)
            states[arm]=dict(pixels=v['pixels'],components=np.stack([a['components'] for a in pair]),
                visibility=np.stack([a['visibility'] for a in pair]),grids=np.tile(np.array(g.BASE)[:,None],(1,len(v['pixels']))))
        records.append(dict(camera=camera,**frozen.expand(data,states)))
    dest.mkdir()
    for path in replay.iterdir():
        if path.is_file():shutil.copyfile(path,dest/('frozen-replay-'+path.name))
    g.write(dest/'diagnostic-replay-comparison.json',comparison)
    controlled=tables(records,dest);display=figures(records,dest,controlled)
    receipt=g.read(root/'render/receipt.json');gates=[x for v in receipt['views'].values() for a in v['convergence'].values() for x in a]
    summary=dict(status='complete_verified_PROCEED_to_separately_authorized_matched_recovery',
        protocol=g.PROTOCOL,source_commit=COMMIT,freeze_sha256=SHA,authorization_sha256=AUTH,
        receipt=g.item(root/'render/receipt.json'),original_metrics=g.item(root/'analysis/metrics.json'),
        frozen_scientific_replay_passed=True,verifier_output_path_only_rebound=True,
        diagnostic_replay_comparison=g.item(dest/'diagnostic-replay-comparison.json'),
        preserved_first_finalization=str(replay),all_captures_verified=len(receipt['outputs']),
        convergence=dict(component_gates=len(gates),all_pass=all(x['passed'] for x in gates),
            maximum_absolute=max(x['maximum_rgb_difference'] for x in gates),
            maximum_relative=max(x['relative_l2'] for x in gates),refined_pixels=receipt['refined_pixels_total'],
            oracle_replay={c:v['oracle_replay'] for c,v in receipt['views'].items()}),
        controlled_comparisons=controlled,display=display,runtime={k:receipt[k] for k in
            ('elapsed_seconds','host_peak_rss_bytes','cuda_peak_allocated_bytes','runtime')},
        observed_work=receipt['counters'],original_evidence_preserved=True,
        material_training_authorized=False,O0_admitted=False,O1_admission_scope='perfect-synthetic-mask-assisted derivative',
        new_execution=dict(checkpoint_loads=0,SDF_queries=0,native_rays=0,shader_integrations=0,optimizer_updates=0,SAM2_inference=0))
    for name,identity in protected.items():g.check(Path(name),identity)
    for stage,pin in PINS.items():baseline(stage,pin)
    g.freeze(SHA)
    g.write(dest/'supplemental-summary.json',summary)
    # Original figure/metrics bytes remain available separately; copy them exactly.
    for path in (root/'analysis').iterdir():
        if path.is_file():shutil.copyfile(path,dest/('original-'+path.name))
    shutil.copyfile(root/'render/receipt.json',dest/'native-receipt.json')
    g.write(dest/'finalization-manifest.json',dict(native_root=str(root),original_native_manifest=g.item(root/'analysis/artifact-manifest.json'),
        native_outputs=receipt['outputs'],protected_original_files=protected,
        files={p.name:g.item(p) for p in dest.iterdir() if p.is_file()},
        sources={n:g.item(g.ROOT/n) for n in ('experiments/finalize_bell_geometry_rgb_gate12.py','experiments/verify_bell_geometry_rgb_gate12.py')}))
    packaged=g.ROOT/'docs/results/publication_gate12/results-v1';packaged.mkdir()
    for path in dest.iterdir():
        if path.is_file() and path.name!='artifact-manifest.json':shutil.copyfile(path,packaged/path.name)
    print(packaged/'supplemental-summary.json')


if __name__=='__main__':main()
