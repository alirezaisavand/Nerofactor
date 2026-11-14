"""Read-only E8 failure analysis; no Torch, experiments, references or model calls."""
import argparse
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
HELPER=ROOT/'docs/results/material_optimizer_e6_v1/analyze_e6.py'
assert HELPER.read_bytes()==subprocess.check_output(['git','show','3cd6bb8e11a8972cd6a85536471f8c8b8a8a9af0:'+str(HELPER.relative_to(ROOT))],cwd=ROOT)
spec=importlib.util.spec_from_file_location('accepted_numpy_analysis',HELPER)
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)  # Blocks torch/all experiments imports.
import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure.signatures import read_report,require

RUN='material-e8-565980ba2b1245a39b221288ea2b7fac'
FREEZE='425956188c550f3139996f70e06f70b19a36c9a574af88a2fddc8df862d68aca'
REPORT='42abc3d3d4a672cc8faa4f1733701ca488b52642690a9eda632321a475fc9ecb'
SEAL='db979494bebaffdfcac779182f171c9e471a9faa5c0ca7d8eb345fdb24b61b5a'
RECEIPT='f13055a1b53bd198b30417e726d70fffae66dd46199c7e39b604c524d19a276f'
COMMIT='9a52ef5286e6c540371aafef3f634be7bc9912a6'
sha=helpers.sha;identity=helpers.identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-parent',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();parent=args.evidence_parent;raw=parent/RUN;out=args.output_dir
    assert raw.is_absolute() and raw.resolve()==raw and out.is_absolute() and out.resolve()==out and out.parent==parent and not out.exists()
    plan,_=read_report(ROOT/'docs/protocols/SPATIAL_MATERIAL_E8_V1.freeze.json',FREEZE)
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h
    for n,h in plan['input_sha256'].items():assert sha(n)==h
    r,_=read_report(raw/'result.json',REPORT);seal,_=read_report(raw/'finalization/seal.json',SEAL);receipt,_=read_report(parent/(RUN+'.process.json'),RECEIPT)
    assert r['status']=='failed' and not r['scientific_work_completed'] and not r['cleanup_verified'] and not r['terminal_integrity_verified']
    assert seal['report']['sha256']==REPORT and seal['outcome']=='failed' and receipt['process_exit_code']==1 and receipt['retry_attempts']==0
    for k in ('authorization','launch','console'):assert identity(Path(receipt[k]['path']))==receipt[k]
    launch=json.loads(Path(receipt['launch']['path']).read_text());assert launch['commit']==COMMIT and launch['freeze_sha256']==FREEZE and launch['foreground_supervisor']
    assert r['arms']==[] and r['attempts']==r['completed']
    assert r['completed']['reference_calls']==118 and r['completed']['reference_directions']==487587840
    assert all(v==0 for k,v in r['completed'].items() if k not in ('reference_calls','reference_directions'))
    refpath=ROOT/'experiments/known_material_reference_v1.py';tree=ast.parse(refpath.read_text())
    rank=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='jacobian_gate')
    ns=dict(np=np,require=require,FD_STEPS=(1e-4,5e-5),RANK_RATIO=1e-4)
    exec(compile(ast.Module(body=[rank],type_ignores=[]),str(refpath),'exec'),ns)
    references=[]
    for row in r['references']:
        a=obs.read_numeric(raw,row['arrays']);delta=abs(a['coarse']-a['fine']);limit=2e-6+5e-5*abs(a['fine'])
        assert np.array_equal(delta,a['absolute_delta']) and np.array_equal(limit,a['limit']) and (delta<=limit).all() and row['passed']
        v=dict(location=row['location'],spatial_split='train' if row['location']<4 else 'held_out',truth=row['truth'],
            convergence_passed=True,maximum_bound_ratio=float(np.max(delta/limit)))
        if 'rank' in row:
            g,_=read_report(raw/row['rank']['path'],row['rank']['sha256']);gate=ns['jacobian_gate'](g['jacobians'])
            assert g['passed'] and gate['passed'] and all(e['passed'] for e in g['evaluations'])
            # Independently reconstruct the saved finite differences from perturbation radiance.
            j=np.empty((2,2,72,7));ev=g['evaluations']
            for s,step in enumerate((1e-4,5e-5)):
                for col in range(7):
                    lo,hi=ev[s*14+col*2:s*14+col*2+2]
                    for level,name in enumerate(('coarse','fine')):j[s,level,:,col]=((np.array(hi[name])[:,2]-np.array(lo[name])[:,2])/(2*step)).reshape(72)
            assert np.array_equal(j,np.asarray(g['jacobians']))
            v.update(rank_passed=True,rank_ratios=[x['ratio'] for x in gate['steps']],
                     smallest_singular_value=gate['smallest_singular_value'],
                     stability_margin=gate['smallest_singular_value']-max(gate['step_spectral_discrepancy'],*gate['grid_spectral_discrepancies']))
        else:v['rank_passed']=None
        references.append(v)
    # Pure private-RNG reconstruction of the FROZEN tapes, with exact hash checks.
    # No shader or reference generation, sample repair, resampling, or changed seeds.
    invalid=[];tape_identities=0;values=0
    for i in range(6):
        for name,start,points in (('fit',6000,24),('evaluation',7000,84)):
            columns={k:[] for k in ('u','v','azimuth')}
            for seed in range(start+10*i,start+10*i+4):
                rng=np.random.RandomState(seed)
                for k,shape in zip(columns,((points,1024),(points,1024),(points,1,1))):
                    original=rng.uniform(size=shape);a=original.astype(np.float32);columns[k].append(a);values+=a.size
                    for index in zip(*np.where((a<0)|(a>=1))):
                        invalid.append(dict(location=i,split=name,seed=seed,array=k,index=list(map(int,index)),
                            float64_value=float(original[index]),float32_value=float(a[index]),
                            cast_rounds_to_one=bool(original[index]<1 and a[index]==1)))
            for k,column in columns.items():
                a=np.concatenate(column);expected=plan['tapes'][str(i)][name][k]
                assert dict(dtype=str(a.dtype),shape=list(a.shape),sha256=sha_array(a))==expected;tape_identities+=1
    assert invalid and all(v['cast_rounds_to_one'] for v in invalid)
    assert not any(n=='torch' or n.startswith(('torch.','experiments.')) for n in sys.modules)
    out.mkdir()
    def dump(name,value):
        with (out/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    analysis=dict(schema_version=1,decision='BLOCKED',run=RUN,execution_commit=COMMIT,process_exit_code=1,
        failure=r['failure'],saved_failure_stage=r['failure_stage'],actual_failure_stage='fixed_tape_batch_construction',
        cleanup_verified=False,terminal_integrity_verified=False,scientific_work_completed=False,
        completed=r['completed'],reference_gates=references,invalid_tape_entries=invalid,
        reconstructed_frozen_tape_array_identities=tape_identities,checked_float32_uniform_values=values,
        sources_verified=len(plan['source_sha256']),inputs_verified=len(plan['input_sha256']),
        additional_scientific_queries=0,additional_reference_queries=0,additional_optimizer_updates=0,
        external_wall_seconds=receipt['wall_seconds'],peak_RSS_bytes=receipt['peak_RSS_bytes'])
    dump('failure_analysis.json',analysis)
    with (out/'planned_arms.csv').open('x',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['seed','arm','status','recovery_error','held_out_error'])
        for seed in plan['seeds']:
            for arm in plan['arms']:writer.writerow([seed,arm,'not_executed','',''])
    with tempfile.TemporaryDirectory(prefix='nerofactor-e8-mpl-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        axes[0].bar([v['location'] for v in references],[v['maximum_bound_ratio'] for v in references]);axes[0].axhline(1,color='black',ls='--');axes[0].set_yscale('log');axes[0].set_title('New reference convergence');axes[0].set_xlabel('Location:0–3 train,4–5 spatial holdout');axes[0].set_ylabel('Max error / original convergence bound')
        axes[1].bar(np.arange(4),[min(v['rank_ratios']) for v in references[:4]]);axes[1].set_xticks(np.arange(4));axes[1].set_yscale('log');axes[1].axhline(1e-4,color='black',ls='--');axes[1].set_title('Independent local rank admission');axes[1].set_xlabel('Training location');axes[1].set_ylabel('Smallest / largest singular value')
        fig.savefig(out/'reference_admission.png',dpi=170);plt.close(fig)
    dump('evidence_manifest.json',dict(schema_version=1,run=RUN,report_sha256=REPORT,seal_sha256=SEAL,receipt_sha256=RECEIPT,freeze_sha256=FREEZE,
        raw_files=[identity(p) for p in sorted(raw.rglob('*')) if p.is_file()],
        sidecars=[identity(parent/(RUN+s)) for s in ('.authorization.json','.launch.json','.console.log','.process.json')],
        analysis_source=identity(Path(__file__).resolve()),accepted_offline_helper=identity(HELPER),raw_evidence_copied_to_Git=False))
    dump('analysis_integrity.json',dict(schema_version=1,scientific_imports_blocked=True,scientific_queries=0,
        files=[identity(p) for p in sorted(out.iterdir()) if p.is_file()]))
    print(json.dumps(dict(status='failure_evidence_verified',invalid_tape_entries=invalid,output=str(out),new_scientific_queries=0)))


def sha_array(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()


if __name__=='__main__':main()
