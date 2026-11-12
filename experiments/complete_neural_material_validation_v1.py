"""Bounded Phase C.1 remainder. Admit old arrays; never repeat initial renders."""
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import platform
import signal
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import validate_neural_material_recovery_v1 as v
from infrastructure.paths import contained_path
from infrastructure.signatures import require, read_report

ROOT=v.ROOT
BASE='42c8f4b08a0db11a47413f1007fb504f6e995378'
PROTOCOL='neural-material-validation-completion-v1'
PRIOR_PATH='docs/evidence/NEURAL_MATERIAL_FIELD_VALIDATION_V1.json'
PRIOR_SHA='03bb86b7c3300839cebad5313a22b50e7169e5e7b81720eef7de0446b52705bf'
REPORT_SHA='a5dff2d22167c00ba1cd144c81b9dbce49f6903f26c9197854d6b118b877d9e5'
REPLAY_SHA='f3b7d45f2fb61445ad5c74b682dde419513fc025645a24692f1314393b073603'
WALL_SECONDS=289
CAPS=dict(shader_forward=17,shader_backward=8,field_vjp=3,synthetic_Adam=3,field_query=20,
          construct_D0=12,construct_N0=4,construct_N1=4,N1_calibration=4,original_components=0,original_component_forward=0)
PRIOR_COUNTS={k:v.CAPS[k]-CAPS[k] for k in CAPS}
CODE=('experiments/complete_neural_material_validation_v1.py','experiments/validate_neural_material_recovery_v1.py',
      'tests/scientific/test_neural_material_validation_completion_v1.py',
      'docs/protocols/NEURAL_MATERIAL_FIELD_VALIDATION_COMPLETION_V1.md')


def replay_schedule():return [(m,a) for m in v.h.previous.reference.PANEL for a in v.h.ARMS if (m,a)!=('M1','D0')]


def plan():
    return dict(protocol=PROTOCOL,base_commit=BASE,prior_evidence_sha256=PRIOR_SHA,
        replay=[dict(material=m,seed=101,arm=a) for m,a in replay_schedule()],
        gauges=[dict(seed=s,variant=k) for s in v.field.SEEDS for k in ('periodic','uncoupled_swap','coupled_swap')],
        synthetic=[dict(seed=101,arm=a,updates=1) for a in v.h.ARMS],caps=CAPS,
        wall_seconds=WALL_SECONDS,RSS_bytes=4*1024**3,address_bytes=8*1024**3,
        threads=1,GPU_hidden=True,repeated_initial_renders=0,recovery_updates=0,reference_calls=0)


def validate_plan(value):
    require(value==plan() and len(replay_schedule())==8,'Exact approved completion schedule required')
    require(CAPS['shader_forward']==8+9 and CAPS['shader_backward']==8 and CAPS['field_vjp']==CAPS['synthetic_Adam']==3,
        'Unexpected additional scientific work')
    require(all(PRIOR_COUNTS[k]+CAPS[k]==v.CAPS[k] for k in CAPS),'Additive original work plan mismatch')
    return value


class Budget(v.Budget):
    def check(self):
        super().check()
        if time.perf_counter()-self.start>=WALL_SECONDS:raise v.Stopped('289-second completion cap; no retry')
    @contextmanager
    def operation(self,kind,label):
        self.check();key=(kind,label);require(key not in self.labels,'Operation retry forbidden')
        c=self.counts[kind];require(c['attempted']<CAPS[kind],'Approved remainder exhausted: '+kind)
        self.labels.add(key);c['attempted']+=1;yield;c['completed']+=1;self.check()
    def finished(self):return all(c['attempted']==c['completed']==CAPS[k] for k,c in self.counts.items())


def verify_reporting_patch():
    source=subprocess.check_output(['git','show',BASE+':experiments/validate_neural_material_recovery_v1.py'],cwd=ROOT)
    old=ast.parse(source);new=ast.parse((ROOT/'experiments/validate_neural_material_recovery_v1.py').read_text())
    helper=ast.parse('''def replay_row(material_name,seed,arm,raw,checks,info):
    """Keep material identity separate from captured material diagnostics."""
    metadata=dict(material_name=material_name,seed=seed,arm=arm,raw=raw,
        passed=all(c['passed'] for c in checks.values()),checks=checks)
    require(not metadata.keys() & info.keys(),'Replay metadata/diagnostic key collision')
    return dict(metadata,**info)
''').body[0]
    actual=[n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='replay_row']
    require(len(actual)==1 and ast.dump(actual[0])==ast.dump(helper),'Reporting helper changed beyond reviewed metadata fix')
    new.body=[n for n in new.body if n is not actual[0]]
    original=ast.parse("row=dict(material=m,seed=101,arm=arm,raw=save_arrays(output,label,arrays),passed=all(c['passed'] for c in checks.values()),checks=checks,**info)").body[0]
    corrected=ast.parse('row=replay_row(m,101,arm,save_arrays(output,label,arrays),checks,info)').body[0]
    class Restore(ast.NodeTransformer):
        hits=0
        def visit_Assign(self,node):
            if ast.dump(node)==ast.dump(corrected):self.hits+=1;return copy.deepcopy(original)
            return self.generic_visit(node)
    transform=Restore();new=transform.visit(new)
    require(transform.hits==1 and ast.dump(new)==ast.dump(old),'Non-reporting change in original validation runner')
    info_keys={'loss','gradient_groups','original_arithmetic_replay','forward_seconds','backward_seconds','material'}
    constructions=[]
    for node in ast.walk(ast.parse((ROOT/'experiments/validate_neural_material_recovery_v1.py').read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg is None and isinstance(k.value,ast.Name) and k.value.id=='info' for k in node.keywords):
            explicit={k.arg for k in node.keywords if k.arg};require(not explicit & info_keys,'Similar duplicate info keyword remains')
            constructions.append(dict(line=node.lineno,explicit_keywords=sorted(explicit)))
    return dict(metadata_only=True,old_sha256=hashlib.sha256(source).hexdigest(),new_sha256=v.h.digest(ROOT/'experiments/validate_neural_material_recovery_v1.py'),
        helper_and_one_call_site_only=True,similar_constructions_reviewed=constructions)


class Evidence:
    def __init__(self,root):
        self.root=Path(root);require(self.root.is_absolute() and self.root.resolve()==self.root,'Explicit unaliased prior evidence root required')
        self.verified={}
    def path(self,name):return contained_path(self.root,name,writable=False)
    def arrays(self,manifest):
        path=self.path(manifest['path']);require(path.is_file() and path.stat().st_size==manifest['bytes'],'Missing or wrong-size previous archive')
        arrays=v.read_arrays(self,manifest)
        require(all(a.dtype.kind!='O' and np.isfinite(a).all() for a in arrays.values()),'Invalid prior numeric evidence')
        self.verified[str(path)]=dict(path=str(path),sha256=manifest['sha256'],bytes=manifest['bytes'],arrays=len(arrays),
            catalogue_sha256=identity(manifest['arrays']),allow_pickle=False)
        return arrays


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def parameter_shapes(arm):
    if arm!='N1':return {'latent':[7]}
    shapes={}
    def linear(name,inputs,outputs):
        for suffix,shape in (('bias',[outputs]),('weight_g',[outputs,1]),('weight_v',[outputs,inputs])):shapes[name+'.'+suffix]=shape
    for block,start in (('module0',51),('module1',307)):
        for index in (0,2,4,6):linear('feats_network.'+block+'.'+str(index),start if index==0 else 256,256)
    for name,width in v.field.HEADS:
        for index in (0,2,4,6):linear(name+'.predictor.'+str(index),259 if index==0 else 256,width if index==6 else 256)
    return shapes


def recheck_capture(arrays,arm,model,target,inputs):
    shapes=dict(tangent=(96,3),bitangent=(96,3),normal=(96,3),half_vectors=(96,1024,3),specular_directions=(96,1024,3),
        diffuse_directions=(96,512,3),pdf=(96,1024,1),valid_specular_counts=(96,1),frame_orthonormal=(96,),
        physical=(1,7),encoding=(96,2),material_encoding=(1,2),mean=(24,3,3),loss=(),per_tape=(4,24,3,3),mask=(96,1024),
        views=(96,3),encoding_gradient=(96,2))
    expected=parameter_shapes(arm)
    require(set(model['parameters'])==set(expected),'Prior model topology/key mismatch')
    require(model['parameter_count']==sum(int(np.prod(s)) for s in expected.values())==(1485584 if arm=='N1' else 7),'Prior parameter count mismatch')
    for name,shape in expected.items():
        meta=model['parameters'][name];require(meta['shape']==shape and meta['dtype']=='torch.float32' and meta['device']=='cpu' and meta['requires_grad'],
            'Prior parameter shape/dtype/trainability mismatch')
        shapes['gradient::'+name]=tuple(shape)
    if arm=='N1':shapes.update(rotation_raw=(1,2),centered_radius=(1,1))
    require(set(arrays)==set(shapes),'Missing/unexpected prior capture tensors')
    for name,shape in shapes.items():
        dtype=np.bool_ if name in ('mask','frame_orthonormal') else np.int64 if name=='valid_specular_counts' else np.float32
        require(arrays[name].shape==shape and arrays[name].dtype==dtype,'Prior capture shape/dtype mismatch: '+name)
    require(np.array_equal(arrays['views'],inputs['views']) and np.array_equal(arrays['normal'],inputs['normals']), 'Prior camera/normal identity mismatch')
    mean=np.stack([torch.from_numpy(arrays['per_tape'][:,:,i].copy()).mean(0).numpy() for i in range(3)],1)
    require(np.array_equal(mean,arrays['mean']) and v.h.previous.roundoff.mse32(mean[:,2],target.numpy())[0]==float(arrays['loss']),
        'Saved float32 mean/MSE replay mismatch')
    mask=np.sum(arrays['specular_directions']*arrays['normal'][:,None],axis=-1)>0
    require(np.array_equal(mask,arrays['mask']) and np.array_equal(mask.sum(1,keepdims=True),arrays['valid_specular_counts']),'Saved sampling mask/count mismatch')
    basis=torch.from_numpy(np.stack([arrays[k] for k in ('tangent','bitangent','normal')],axis=-1))
    error=float((basis.transpose(-1,-2)@basis-torch.eye(3)).abs().max())
    require(error<=2e-6 and arrays['frame_orthonormal'].all(),'Saved frame orthonormality mismatch')
    v.field.Material(torch.from_numpy(arrays['physical']),torch.from_numpy(arrays['material_encoding'])).validate()
    require(np.array_equal(arrays['encoding'],np.broadcast_to(arrays['material_encoding'],(96,2))),'Actual shader encoding differs from field output')
    if arm=='N1':
        decoded=F.normalize(torch.from_numpy(arrays['rotation_raw'])*2-1,dim=-1,eps=1e-6).numpy()
        require(np.array_equal(decoded,arrays['material_encoding']),'Saved neural encoding violates original head normalization')
    groups={}
    for name in expected:
        a=arrays['gradient::'+name];group=name.split('.')[0];g=groups.setdefault(group,dict(parameters=0,nonzero=0,finite=True,squared_norm=0.))
        g['parameters']+=a.size;g['nonzero']+=int(np.count_nonzero(a));g['finite'] &= bool(np.isfinite(a).all());g['squared_norm']+=float(np.sum(a.astype(float)**2))
    require(all(g['finite'] and g['nonzero']>0 for g in groups.values()) and bool(np.count_nonzero(arrays['encoding_gradient'])), 'Disconnected/nonfinite prior material gradients')
    return dict(passed=True,frame_maximum_error=error,mean_MSE_replayed=True,mask_recomputed=True,gradient_groups=groups)


def verify_previous():
    reference,ids=v.admit();patch=verify_reporting_patch()
    versioned,_=read_report(ROOT/PRIOR_PATH,PRIOR_SHA);selected=versioned['original_report']
    require(selected['sha256']==REPORT_SHA,'Unapproved original report selection')
    path=Path(selected['path']);require(path.resolve()==path and path.stat().st_size==selected['bytes'],'Missing/aliased prior report')
    old,_=read_report(path,REPORT_SHA);evidence=Evidence(path.parent)
    require(old['status']=='failed' and old['failure_type']=='TypeError' and old['failure']=="dict() got multiple values for keyword argument 'material'",'Prior failure identity changed')
    require(old['work']=={k:dict(attempted=n,completed=n) for k,n in PRIOR_COUNTS.items()},'Prior work counters inconsistent with approved remainder')
    require(old['recovery_updates']==0 and not old['replay'] and not old['gauge'] and not old['storage'],'Unexpected prior executed fixtures')
    require(old['fixture_plan']==v.plan(),'Prior fixed plan changed')
    for name,sha in old['validation_source_sha256'].items():
        blob=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest()==sha,'Old executed source Git identity mismatch')
        if name!='experiments/validate_neural_material_recovery_v1.py':require(v.h.digest(ROOT/name)==sha,'Old test/protocol changed')
    for name,sha in old['accepted_identities'].items():require(v.h.digest(ROOT/name)==sha,'Inherited scientific evidence changed: '+name)
    pre,_=read_report(evidence.path('pre-execution.json'),v.h.digest(evidence.path('pre-execution.json')))
    outcome,_=read_report(evidence.path('process-outcome.json'),v.h.digest(evidence.path('process-outcome.json')))
    require(pre==versioned['pre_execution'] and outcome==versioned['external_execution'] and pre['sha256']==old['validation_source_sha256'],
        'Original pre-execution/process manifest mismatch')
    require(v.h.digest(evidence.path('console.log'))==outcome['console_sha256'],'Original console hash mismatch')
    require(outcome['process_wall_seconds']+WALL_SECONDS<300,'Completion would extend original total wall budget')
    fit,external,tapes=v.h.inputs(reference);require(tapes==old['tapes'],'Fixed tape identities changed')
    inputs=evidence.arrays(old['input_archive']);batch=v.h.MaterialBatch(reference['scene'],reference['scene']['observations'][:24],fit)
    actual=dict(u=fit.u.numpy(),v=fit.v.numpy(),azimuth=fit.azimuth.numpy(),positions=batch.base.positions.numpy(),normals=batch.base.normals.numpy(),views=batch.base.views.numpy())
    targets={m:v.h.previous.material_target(next(row for row in reference['materials'] if row['material']==m),m)[1] for m in v.h.previous.reference.PANEL}
    actual.update({'target_'+m:t.numpy() for m,t in targets.items()})
    require(v.array_catalog(inputs)==v.array_catalog(actual),'Saved inputs/targets differ from accepted source and evidence')
    expected_order=[(m,s) for m in v.h.previous.reference.PANEL for s in v.field.SEEDS]
    require([(a['material'],a['seed']) for a in old['initial_admission']]==expected_order,'Prior complete initial matrix required')
    verified=[]
    for entry in old['initial_admission']:
        m,s=entry['material'],entry['seed'];arrays={};checks={}
        require(set(entry['arms'])==set(v.h.ARMS),'Missing previous arm')
        for arm in v.h.ARMS:
            row=entry['arms'][arm];arrays[arm]=evidence.arrays(row['raw'])
            checks[arm]=recheck_capture(arrays[arm],arm,row['initial_model'],targets[m],inputs)
            for group,g in checks[arm]['gradient_groups'].items():
                saved=row['gradient_groups'][group];require(saved['missing']==0 and saved['finite'] and saved['parameters']==g['parameters'] and saved['nonzero']==g['nonzero'],
                    'Stored gradient metadata inconsistent with actual arrays')
        require(entry['arms']['D0']['initial_model']['parameters']==entry['arms']['N0']['initial_model']['parameters'],'Original latent identities mismatch')
        z=v.h.fresh_latent(s);physical=v.h.previous.original.physical(z).numpy().reshape(1,7)
        require(np.array_equal(arrays['D0']['physical'],physical) and np.array_equal(arrays['N0']['physical'],physical),'Prior direct/constant initialization changed')
        q=(1+.5*torch.stack((torch.cos(2*z[2]),torch.sin(2*z[2]))))/2
        calibrated=torch.sigmoid(torch.log(q/(1-q))).reshape(1,2).numpy()
        require(np.array_equal(arrays['N1']['rotation_raw'],calibrated),'Prior rho=.5 neural calibration changed')
        comparisons=dict(D0_N0=v.parity(arrays['N0'],arrays['D0'],targets[m],exact=True),D0_N1=v.parity(arrays['N1'],arrays['D0'],targets[m]))
        require(all(c['passed'] for c in comparisons.values()),'Saved initial numerical gate independently failed')
        require(comparisons==entry['comparisons'],'Recorded comparison diagnostics disagree with recalculation')
        verified.append(dict(material_name=m,seed=s,passed=True,captures=checks,comparisons=comparisons))
    for audit in old['field_audits']:
        jac=np.asarray(audit['physical_bias_jacobian'],np.float64)
        require(jac.shape==(7,8) and np.isfinite(jac).all() and np.linalg.matrix_rank(jac)==7,'Saved physical reachability matrix invalid')
        source,_=v.field.source_definitions();require(audit['source_definition_AST_sha256']==_['definition_AST_sha256'],'Saved source AST identity mismatch')
    require(len(old['field_audits'])==3 and [r['seed'] for r in old['field_audits']]==list(v.field.SEEDS),'Prior field audit incomplete')
    captured=versioned['offline_verification']['recovered_first_replay'];require(captured['sha256']==REPLAY_SHA,'Saved replay selection changed')
    relative=str(Path(captured['path']).relative_to(evidence.root));manifest=dict(path=relative,sha256=REPLAY_SHA,bytes=captured['bytes'],arrays=captured['arrays'])
    replay=evidence.arrays(manifest);base=evidence.arrays(old['initial_admission'][0]['arms']['D0']['raw'])
    require(v.array_catalog(replay)==v.array_catalog(base),'Previously captured first replay not bitwise identical')
    for flag in ('global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged','threads_restored','memory_limit_restored'):require(old[flag] is True,'Prior runtime integrity failure')
    summary=dict(passed=True,previous_failure_preserved=True,versioned_sha256=PRIOR_SHA,original_report=selected,metadata_patch=patch,
        independent_initial_admission=verified,verified_archives=list(evidence.verified.values()),
        recovered_replay=dict(material_name='M1',seed=101,arm='D0',sha256=REPLAY_SHA,passed=True,all_arrays_exact=True,new_scientific_calls=0),
        prior_work=old['work'],saved_reachability_ranks=[7,7,7],new_shader_calls=0,new_backward_calls=0)
    return reference,old,evidence,targets,(fit,external,tapes),ids,summary


def execute(output,budget,report,persist):
    reference,old,evidence,targets,tapes,ids,verified=verify_previous()
    report.update(previous_evidence_verification=verified,accepted_identities=ids,
        implementation_sha256=old['implementation_sha256'],validation_source_sha256={n:v.h.digest(ROOT/n) for n in CODE},runtime=reference['runtime'])
    require(reference['runtime']==dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1),'Accepted numerical CPU runtime required')
    output.path('raw').mkdir();persist();fit,external,manifest=tapes
    shader=v.h.previous.original.MaterialShaderV1('A2','atan2');require(shader.identity==old['shader'],'Shader identity changed')
    batch=v.h.MaterialBatch(reference['scene'],reference['scene']['observations'][:24],fit)
    entries={(r['material'],r['seed']):r for r in old['initial_admission']}
    def failure_sink(label,arrays):report['failure_artifacts'].append(v.save_arrays(output,label+'-failed',arrays));persist()
    for m,arm in replay_schedule():
        label='replay-'+m+'-'+arm;model=v.make_model(arm,101,budget,label,report['timings']);original=entries[(m,101)]['arms'][arm]
        require(v.field.identity(model)==original['initial_model'],'Fresh replay network identity changed')
        arrays,info=v.capture(model,batch,shader,targets[m],budget,label,report['timings'],failure_sink=failure_sink)
        base=evidence.arrays(original['raw']);checks={k:v.comparison(arrays[k],base[k]) for k in base}
        row=v.replay_row(m,101,arm,v.save_arrays(output,label,arrays),checks,info)
        report['new_replay'].append(row);persist();require(row['passed'],'Deterministic replay failed')
    for seed in v.field.SEEDS:
        base=evidence.arrays(entries[('M1',seed)]['arms']['D0']['raw']);p0=torch.from_numpy(base['physical'][0].copy())
        for variant in ('periodic','uncoupled_swap','coupled_swap'):
            label='gauge-'+str(seed)+'-'+variant
            with budget.operation('construct_D0',label):
                p=p0.clone()
                if variant=='periodic':p[2]=p[2]+np.pi
                else:p[:2]=p0[[1,0]];p[2]=p[2]+np.pi/2
                model=v.PhysicalFixture(p)
            selected=batch;mapping=None;tape_record=None
            if variant!='periodic':mapping=v.h.previous.coupling.frame_sign(v.h.previous.coupling.decoded_frame(shader,p0,96),v.h.previous.coupling.decoded_frame(shader,p,96))
            if variant=='coupled_swap':
                tape,tape_record=v.h.previous.coupling.copy_coupled_tape(fit,mapping['sign']);selected=v.h.MaterialBatch(reference['scene'],reference['scene']['observations'][:24],tape)
            arrays,info=v.capture(model,selected,shader,targets['M1'],budget,label,report['timings'],backward=False,failure_sink=failure_sink)
            row=dict(material_name='M1',seed=seed,variant=variant,raw=v.save_arrays(output,label,arrays),mapping=mapping,coupled_tape=tape_record,**info)
            if variant=='uncoupled_swap':
                row['diagnostic_difference']={k:v.comparison(arrays[k],base[k],mandatory=False) for k in ('half_vectors','specular_directions','pdf','mean')}
                q0,q1=(v.h.previous.coupling.accepted.tensor_q(x) for x in (p0.numpy(),p.numpy()))
                row['physical_Q']=v.comparison(q1,q0,2e-7,2e-6);row['passed']=row['physical_Q']['passed']
            else:row['comparison']=v.parity(arrays,base,targets['M1'],gauge=variant=='coupled_swap');row['passed']=row['comparison']['passed']
            report['new_gauge'].append(row);persist();require(row['passed'],'Mandatory gauge fixture failed: '+label)
    report['storage']=v.benchmark(output,budget,report['timings'])
    report['runtime_feasibility']=v.feasibility(old['timings']+report['timings'],report['storage'])
    v.h.verify_tapes(fit,external,manifest)
    for name,sha in {**ids,**report['validation_source_sha256']}.items():require(v.h.digest(ROOT/name)==sha,'Source/evidence changed during completion')
    require(v.h.digest(evidence.path('report.json'))==REPORT_SHA and v.h.digest(ROOT/PRIOR_PATH)==PRIOR_SHA,'Prior report changed')
    require(budget.finished(),'Incomplete additional work matrix')
    combined={k:{n:old['work'][k][n]+budget.counts[k][n] for n in ('attempted','completed')} for k in CAPS}
    require(all(c['attempted']==c['completed']==v.CAPS[k] for k,c in combined.items()),'Original aggregate work matrix not complete')
    report.update(combined_work=combined,all_prescribed_fixtures_completed=True,full_shader_admission_passed=True,
        runtime_feasibility_passed=report['runtime_feasibility']['passed'])
    if not report['runtime_feasibility_passed']:raise v.Stopped('900-second active-history budget appears infeasible; review without changing settings')
    report['status']='passed'


def run(root,run_id):
    validate_plan(plan());budget=Budget()
    report=dict(schema_version=1,protocol=PROTOCOL,base_commit=BASE,status='running',run_id=run_id,fixture_plan=plan(),work=budget.counts,
        new_replay=[],new_gauge=[],storage=[],timings=[],failure_artifacts=[],recovery_updates=0,repeated_initial_renders=0,
        full_shader_admission_passed=False,runtime_feasibility_passed=False,all_prescribed_fixtures_completed=False,
        final_freeze_created=False,Phase_D_authorized=False)
    with v.h.OwnedOutput(root,run_id) as output:
        def persist():output.persist(report)
        persist()
        try:
            with v.runtime(budget,report):
                signal.setitimer(signal.ITIMER_REAL,max(1e-6,WALL_SECONDS-(time.perf_counter()-budget.start)))
                execute(output,budget,report,persist)
        except BaseException as exc:report.update(status='blocked' if isinstance(exc,v.Stopped) else 'failed',failure_type=type(exc).__name__,failure=str(exc))
        finally:
            report['shader_points']={k:96*n for k,n in budget.counts['shader_forward'].items()};persist()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only',action='store_true');mode.add_argument('--execute-once',action='store_true')
    parser.add_argument('--output-root',type=Path);parser.add_argument('--run-id');args=parser.parse_args()
    if args.verify_only:
        *_,verified=verify_previous();print(json.dumps(dict(plan=validate_plan(plan()),old_admission_reverified=len(verified['independent_initial_admission']),
            verified_archives=len(verified['verified_archives']),metadata_patch=verified['metadata_patch']),indent=2));return
    require(args.output_root is not None and args.run_id is not None,'Explicit fresh output root and unique run ID required')
    report=run(args.output_root,args.run_id);print(json.dumps(dict(status=report['status'],failure=report.get('failure'),work=report['work']),indent=2))
    if report['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
