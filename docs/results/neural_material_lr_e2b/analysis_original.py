"""Offline E2b recorded-data analysis. No scientific model/renderer imports."""
import csv
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/isavand/paper/evidence/neural-e2b-d1f760d630964a2786ff45bde7438825')
OUT=Path(__file__).resolve().parent
REPO=Path('/home/isavand/paper/Nerofactor-3d2-e2a')
FREEZE_SHA='f9d8f28a46641c8f3aa777118b7f106f120918e7c60e6605387fcb7c46d984ad'
REPORT_SHA='bb12d3627e778634da17da7a2b484ebe608a88d94aab0bc6c033cd58a1f15373'
SEAL_SHA='cabee3c788264f60b127882274b8f03d81b28248defacc509bb4ee20843786f5'
RECEIPT_SHA='20f52bbcec66b1a06e1dfb9320f848304c536aababe63e8ac39630ec9938cbcd'
CASES=('M1/101','M1/103','M2/103','M3/101','M3/103')
HEADS=('mx_predictor','my_predictor','metallic_predictor','rotation_predictor','kd_predictor')
GROUPS=('feats_network',*HEADS)
SHADERS=('mx','my','metallic','encoding','albedo')
class BlockScientific(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname=='torch' or fullname.startswith(('torch.','experiments.neural_material','experiments.material_shader')):
            raise RuntimeError('Offline analysis forbids scientific imports: '+fullname)
sys.meta_path.insert(0,BlockScientific())
sys.path.insert(0,str(REPO))
os.environ['MPLCONFIGDIR']=str(OUT/'plot-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from infrastructure import neural_material_observer_e2a as audit
from infrastructure.signatures import read_report

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def table(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def bits_equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def total_norm(groups):return math.sqrt(math.fsum(g['L2']**2 for g in groups.values()))

assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='a1314e192ed87b70bcf5f78c223e319d8e3ce56f'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
freeze,_=read_report(REPO/'docs/protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A_V2.freeze.json',FREEZE_SHA)
report,_=read_report(ROOT/'result.json',REPORT_SHA)
seal,_=read_report(ROOT/'finalization/seal.json',SEAL_SHA)
receipt,_=read_report(ROOT.parent/(ROOT.name+'.process.json'),RECEIPT_SHA)
assert seal['report']['sha256']==REPORT_SHA and seal['outcome']=='complete' and receipt['process_exit_code']==0
assert report['scientific_work_completed'] and report['cleanup_verified'] and report['work_counters_verified']
assert all(v['status']=='passed' for v in report['terminal_integrity'].values())
assert report['verified_totals']==freeze['total_work'] and len(report['outcomes'])==10
critical={str(ROOT/'result.json'):REPORT_SHA,str(ROOT/'finalization/seal.json'):SEAL_SHA,
          str(ROOT.parent/(ROOT.name+'.process.json')):RECEIPT_SHA,
          str(REPO/'docs/protocols/NEURAL_MATERIAL_EARLY_STEP_LR_E2A_V2.freeze.json'):FREEZE_SHA}
for path,s in critical.items():assert sha(path)==s
all_inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file()]
for suffix in ('.review.json','.launch.json','.console.log','.process.json'):
    p=ROOT.parent/(ROOT.name+suffix);all_inputs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
save('inputs.json',dict(protocol='e2b-offline-recorded-comparison-v1',critical=critical,artifacts=all_inputs,
    analysis_sha256=sha(__file__),model_queries=0,renderer_queries=0,optimizer_updates=0))

endpoints=[];trajectories=[];routes=[];motions=[];saturation=[];series={};verified=[];initials={}
for arm in report['outcomes']:
    case=arm['material']+'/'+str(arm['seed']);lr=arm['lr'];key=(case,lr)
    assert arm['status']=='complete' and arm['completed_updates']==10 and arm['work']==freeze['work_per_arm']
    assert arm['native_work']==freeze['native_work_per_arm'] and arm['hooks_removed']
    if lr==.01:assert arm['historical_parity']=={'0':True,'9':True}
    initial=audit.read_numeric(ROOT,arm['initial_state']);final=audit.read_numeric(ROOT,arm['final_state'])
    initials[key]=initial
    parameters={n[len('parameter/'):]:a for n,a in initial.items() if n.startswith('parameter/')}
    initial_parameters={n:a.copy() for n,a in parameters.items()}
    assert audit.parameter_identity(parameters)==arm['initial_parameter_identity']
    assert all(bits_equal(a,final[n]) for n,a in initial.items() if not n.startswith('parameter/'))
    rows=[];last_sha=None;path_length=0.;zero_gradient_moving=0
    for state,selection in enumerate(arm['records']):
        row,_=read_report(ROOT/selection['path'],selection['sha256'])
        assert row['pre_state']==row['loss_state']==state and row['post_state']==min(state+1,10)
        assert row['previous_record_sha256']==last_sha;last_sha=selection['sha256']
        nodes=row['observations']['nodes'];signals=audit.read_numeric(ROOT,row['signals'])
        for name,node in nodes.items():
            assert audit.array_record(signals[name])==node['value']
            assert audit.array_record(signals[name+'.loss_gradient'])==node['loss_gradient']
        nonzero=sum(g['nonzero'] for g in row['parameter_gradients'].values())
        assert math.sqrt(sum(row['original_gradient_groups'][g]['squared_norm'] for g in GROUPS))==row['original_gradient_norm']
        for head,shader in zip(HEADS,SHADERS):
            q=signals[head+'.sigmoid'];gq=signals[head+'.sigmoid.loss_gradient'];gz=signals[head+'.logits.loss_gradient']
            z=signals[head+'.logits'];b=(q==0)|(q==1)
            assert not np.any(b&(gz!=0))
            routes.append(dict(case=case,lr=lr,state=state,head=head,shader_input=shader,
                sigmoid_values=q.tolist(),logits=z.tolist(),hard_boundary_channels=int(np.count_nonzero(b)),
                upstream_nonzero_but_boundary_blocks_channels=int(np.count_nonzero(b&(gq!=0)&(gz==0))),
                shader_gradient_L2=nodes['shader.'+shader]['loss_gradient']['L2'],
                shader_gradient_nonzero=nodes['shader.'+shader]['loss_gradient']['nonzero'],
                sigmoid_output_gradient_L2=nodes[head+'.sigmoid']['loss_gradient']['L2'],
                logit_gradient_L2=nodes[head+'.logits']['loss_gradient']['L2']))
        relu=[v for n,v in nodes.items() if n.startswith('relu.') and n.endswith('.post')]
        tr=dict(case=case,lr=lr,state=state,loss=row['loss'],original_gradient_norm=row['original_gradient_norm'],
            Q_relative=row['physical']['errors']['Q_relative_frobenius'],physical=row['physical']['physical'],
            physical_boundary_distances=row['physical']['boundary_distances'],centered_radius=row['physical']['centered_radius'][0][0],
            min_sigmoid_boundary=float(min(signals[h+'.sigmoid.boundary_distance'].min() for h in HEADS)),
            boundary_channels=sum(nodes[h+'.sigmoid']['analytic_derivative']['zero'] for h in HEADS),
            parameter_gradient_nonzero=nonzero,parameter_gradient_nonzero_fraction=nonzero/1485584,
            shared_feature_gradient_L2=nodes['shared_features']['loss_gradient']['L2'],
            relu_positive_fraction=sum(v['activity']['positive'] for v in relu)/(22*256))
        for h in HEADS:tr[h+'_logit_gradient_L2']=nodes[h+'.logits']['loss_gradient']['L2']
        trajectories.append(tr);rows.append(tr)
        for head in HEADS:
            saturation.append(dict(case=case,lr=lr,state=state,head=head,
                first_observed_boundary_state=arm['first_boundary_states'].get(head),
                boundary_channels=nodes[head+'.sigmoid']['analytic_derivative']['zero'],
                sigmoid_values=signals[head+'.sigmoid'].tolist(),
                minimum_boundary=float(signals[head+'.sigmoid.boundary_distance'].min()),
                minimum_derivative=float(signals[head+'.sigmoid.analytic_derivative'].min()),
                first_zero_group_gradient=arm['first_zero_gradient_states'].get(head)))
        if state<10:
            xor=audit.read_numeric(ROOT,row['xor']);after=audit.reconstruct(parameters,xor)
            assert audit.parameter_identity(parameters)==row['actual_displacement']['before_identity']
            assert audit.parameter_identity(after)==row['actual_displacement']['after_identity']
            _,recomputed=audit.displacement(parameters,after)
            assert recomputed['per_tensor']==row['actual_displacement']['per_tensor']
            for g in GROUPS:
                for field in ('changed_elements','bitwise_changed_elements','max_absolute'):
                    assert recomputed['per_group'][g][field]==row['actual_displacement']['per_group'][g][field]
            d=row['actual_displacement'];motion=total_norm(d['per_tensor']);path_length+=motion
            changed=sum(v['changed_elements'] for v in d['per_tensor'].values())
            bit_changed=sum(v['bitwise_changed_elements'] for v in d['per_tensor'].values())
            zero_gradient_moving+=int(nonzero==0 and bit_changed>0)
            item=dict(case=case,lr=lr,update=state+1,pre_state=state,post_state=state+1,
                current_gradient_nonzero=nonzero,parameter_gradient_nonzero_fraction=nonzero/1485584,
                actual_parameter_L2=motion,changed_elements=changed,bitwise_changed_elements=bit_changed,
                bitwise_changed_fraction=bit_changed/1485584,max_absolute=max(v['max_absolute'] for v in d['per_tensor'].values()),
                before_identity=d['before_identity'],after_identity=d['after_identity'])
            for g in GROUPS:item[g+'_displacement_L2']=d['per_group'][g]['L2']
            motions.append(item);parameters=after
        assert all(v['step']['max_absolute']==min(state+1,10) for v in row['Adam_after_return'].values())
    assert len(rows)==11 and arm['terminal_record_sha256']==last_sha
    assert audit.parameter_identity(parameters)==arm['final_parameter_identity']
    assert all(bits_equal(a,final['parameter/'+n]) for n,a in parameters.items())
    terminal,_=read_report(ROOT/arm['records'][-1]['path'],arm['records'][-1]['sha256'])
    for name in parameters:
        for field in ('step','exp_avg','exp_avg_sq'):
            assert audit.array_record(final['adam/'+name+'/'+field])==terminal['Adam_after_return'][name][field]
    first_boundary=min(arm['first_boundary_states'].values(),default=None)
    net=math.sqrt(math.fsum(float(np.sum((parameters[n].astype(np.float64)-a.astype(np.float64))**2)) for n,a in initial_parameters.items()))
    e=arm['endpoint'];errors=e['recovery_errors']
    endpoint=dict(case=case,lr=lr,loss0=rows[0]['loss'],loss10=rows[-1]['loss'],loss10_over_initial=rows[-1]['loss']/rows[0]['loss'],
        Q0=rows[0]['Q_relative'],Q10=errors['Q_relative_frobenius'],orientation_identifiable=errors['orientation_identifiable'],
        orientation_error_degrees=errors['orientation_degrees'],roughness_axis_relative_errors=errors['sorted_axis_relative_errors'],
        albedo_max_absolute_error=max(errors['albedo_absolute_errors']),albedo_absolute_errors=errors['albedo_absolute_errors'],
        metallic_absolute_error=errors['metallic_absolute_error'],physical=e['physical'],
        heldout_relative_RGB=e['radiance_errors']['held_out']['total']['relative_L2'],
        heldout_diffuse_relative=e['radiance_errors']['held_out']['diffuse']['relative_L2'],
        heldout_specular_relative=e['radiance_errors']['held_out']['specular']['relative_L2'],
        external_fit_relative_RGB=e['radiance_errors']['fit']['total']['relative_L2'],relight_relative_RGB=e['radiance_errors']['relight']['total']['relative_L2'],
        physical_success=e['success']['all_passed'],success_checks=e['success']['checks'],
        first_boundary_state=first_boundary,first_all_zero_gradient_state=arm['first_all_zero_gradient_state'],
        final_global_gradient_norm=rows[-1]['original_gradient_norm'],final_minimum_sigmoid_boundary=rows[-1]['min_sigmoid_boundary'],
        zero_current_gradient_moving_updates=zero_gradient_moving,net_parameter_L2=net,parameter_path_length=path_length,
        first_boundary_by_head=arm['first_boundary_states'])
    endpoints.append(endpoint);series[key]=rows
    verified.append(dict(case=case,lr=lr,records=11,exact_displacement_transitions=10,final_parameter_bits=True,
        final_Adam_moments_and_counters=True,buffers_and_global_RNG_unchanged=True,signal_archives_reverified=11))
for case in CASES:
    a,b=initials[case,.01],initials[case,.001]
    assert set(a)==set(b) and all(bits_equal(a[n],b[n]) for n in a)
    pair=[r for r in report['outcomes'] if r['material']+'/'+str(r['seed'])==case]
    assert pair[0]['initial_rng']==pair[1]['initial_rng'] and pair[0]['initial_signature']==pair[1]['initial_signature']

for name,rows in (('endpoints.csv',endpoints),('trajectories.csv',trajectories),('gradient_routes.csv',routes),
                  ('adam_displacements.csv',motions),('sigmoid_boundaries.csv',saturation)):table(name,rows)
by={(r['case'],r['lr']):r for r in endpoints}
summary=dict(schema_version=1,protocol='e2b-offline-recorded-comparison-v1',status='complete',run_id=ROOT.name,
    accepted_commit='a1314e192ed87b70bcf5f78c223e319d8e3ce56f',critical_input_sha256=critical,
    exact_work=report['verified_totals'],wall_seconds=receipt['wall_seconds'],peak_RSS_bytes=report['peak_RSS_bytes'],
    cleanup_and_terminal_integrity=report['terminal_integrity'],integrity_audit=verified,
    raw_artifact_files=len(all_inputs)-4,raw_artifact_bytes=sum(p['bytes'] for p in all_inputs[:-4]),
    paired_initial_full_states_equal=True,historical_parity_cases=5,historical_parity_comparisons=10,
    endpoints=endpoints,counts=dict(lower_LR_training_MSE_better=sum(by[c,.001]['loss10']<by[c,.01]['loss10'] for c in CASES),
        lower_LR_Q_better=sum(by[c,.001]['Q10']<by[c,.01]['Q10'] for c in CASES),
        lower_LR_heldout_better=sum(by[c,.001]['heldout_relative_RGB']<by[c,.01]['heldout_relative_RGB'] for c in CASES),
        exact_saturation_arms_by_LR={str(lr):sum(r['first_boundary_state'] is not None for r in endpoints if r['lr']==lr) for lr in (.01,.001)},
        full_gradient_zero_arms_by_LR={str(lr):sum(r['first_all_zero_gradient_state'] is not None for r in endpoints if r['lr']==lr) for lr in (.01,.001)},
        physical_recovery_passes=sum(r['physical_success'] for r in endpoints),
        zero_current_gradient_but_nonzero_actual_motion_updates=sum(r['zero_current_gradient_moving_updates'] for r in endpoints)),
    new_model_queries=0,new_renderer_queries=0,new_optimizer_updates=0,new_reference_generation=0,
    interpretation_scope='Five selected fixed starts, ten updates per arm. No population estimate or 5000-update inference.',
    audit_note='Exact per-tensor displacement statistics and group counts/maxima verified; grouped norm re-reductions can depend on summation order. No tolerances or scientific criteria changed.')
save('comparison.json',summary)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={.01:'#D55E00',.001:'#0072B2'}
def emit(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(5,3,figsize=(14,15),sharex=True)
for i,case in enumerate(CASES):
    for lr in (.01,.001):
        rows=series[case,lr];x=[r['state'] for r in rows]
        for j,field in enumerate(('loss','original_gradient_norm','Q_relative')):
            axes[i,j].plot(x,[r[field] for r in rows],'-o',ms=3,color=colors[lr],label='LR '+str(lr))
            axes[i,j].grid(alpha=.2)
    axes[i,0].set_yscale('log');axes[i,1].set_yscale('symlog',linthresh=1e-9);axes[i,2].set_yscale('log')
    axes[i,0].set_ylabel(case);axes[i,1].set_ylabel('symlog; exact zero retained')
for j,title in enumerate(('Fitting MSE, original tapes','Original N1 gradient norm','Relative anisotropy-tensor Q error')):axes[0,j].set_title(title)
for ax in axes[-1]:ax.set_xlabel('Completed updates / pre-update state');ax.set_xticks(range(0,11,2))
axes[0,0].legend();fig.suptitle('Recorded early trajectories — same initial states, only LR differs',y=1.005)
fig.tight_layout();emit(fig,'early_trajectories')

positive=[row[h+'_logit_gradient_L2'] for row in trajectories for h in HEADS if row[h+'_logit_gradient_L2']>0]
vmin=math.floor(math.log10(min(positive)));vmax=math.ceil(math.log10(max(positive)))
fig,axes=plt.subplots(5,2,figsize=(14,13),sharex=True,sharey=True,constrained_layout=True)
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#c9c9c9')
for i,case in enumerate(CASES):
    for j,lr in enumerate((.01,.001)):
        rows=series[case,lr];values=np.array([[row[h+'_logit_gradient_L2'] for row in rows] for h in HEADS]);masked=np.ma.masked_all(values.shape)
        nonzero=values>0;masked[nonzero]=np.log10(values[nonzero])
        im=axes[i,j].imshow(masked,aspect='auto',origin='upper',cmap=cmap,vmin=vmin,vmax=vmax,extent=(-.5,10.5,4.5,-.5))
        for hidx,h in enumerate(HEADS):
            for row in saturation:
                if row['case']==case and row['lr']==lr and row['head']==h and row['boundary_channels']:
                    axes[i,j].text(row['state'],hidx,'B',ha='center',va='center',color='red',weight='bold',fontsize=8)
        axes[i,j].set_title(case+' | LR '+str(lr));axes[i,j].set_yticks(range(5));axes[i,j].set_yticklabels(['mx','my','metallic','rotation','albedo'])
        axes[i,j].set_xticks(range(0,11,2))
for ax in axes[-1]:ax.set_xlabel('Pre-update state (state 10 is terminal)')
fig.colorbar(im,ax=axes.ravel().tolist(),label='log10 actual logit-loss-gradient L2; gray = exact zero',shrink=.7)
fig.suptitle('Actual graph gradients at calibrated logits; B = at least one exact sigmoid boundary')
emit(fig,'logit_gradients_and_boundaries')

fig,axes=plt.subplots(5,2,figsize=(13,14),sharex=True)
for i,case in enumerate(CASES):
    for lr in (.01,.001):
        rows=[r for r in motions if r['case']==case and r['lr']==lr];x=[r['update'] for r in rows]
        axes[i,0].plot(x,[r['actual_parameter_L2'] for r in rows],'-o',ms=3,color=colors[lr],label='LR '+str(lr))
        axes[i,1].plot(x,[100*r['bitwise_changed_fraction'] for r in rows],'-',color=colors[lr],label=str(lr)+' weights changed')
        axes[i,1].plot(x,[100*r['parameter_gradient_nonzero_fraction'] for r in rows],'--',color=colors[lr],label=str(lr)+' nonzero gradients')
        for row in rows:
            if row['current_gradient_nonzero']==0:axes[i,0].plot(row['update'],row['actual_parameter_L2'],'x',color='black',ms=8)
    axes[i,0].set_ylabel(case+' | L2 displacement');axes[i,1].set_ylabel('% of 1,485,584 parameters')
    for ax in axes[i]:ax.grid(alpha=.2)
axes[0,0].set_title('Actual float32 parameter movement; x = zero current gradient')
axes[0,1].set_title('Gradient zeros and unchanged weights are different observations')
axes[0,0].legend();axes[0,1].legend(fontsize=8)
for ax in axes[-1]:ax.set_xlabel('Update number (gradient measured at pre-state update−1)')
fig.tight_layout();emit(fig,'adam_motion_vs_gradient')

assert 'torch' not in sys.modules
artifacts=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()]
save('analysis-manifest.json',dict(schema_version=1,artifacts=artifacts,scientific_model_queries=0,renderer_queries=0,optimizer_updates=0))
assert all(sha(x['path'])==x['sha256'] for x in all_inputs)
print(json.dumps(dict(status='complete',analysis_directory=str(OUT),counts=summary['counts'],exact_transition_audits=100,
    additional_scientific_queries=0,comparison_sha256=sha(OUT/'comparison.json')),sort_keys=True))
