"""One forward-only, fixed-state A2 symmetry audit; no optimization or new targets."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

import numpy as np
import torch

from audits import anisotropic_brdf as independent
from experiments import material_recovery_optimizer_comparison_v1 as previous
from experiments import run_known_material_recovery_v1 as original
from experiments import known_material_reference_v1 as reference
from experiments.material_shader_v1 import MaterialShaderV1, decode_atan2
from infrastructure.signatures import require, write_report
from infrastructure.stage_sources import baseline, PINS

ROOT = Path(__file__).resolve().parents[1]
BASE = 'f4f5f4d57fc9f57c4d6387663ac9ef19b40144c0'
PROTOCOL = ROOT/'docs/protocols/MATERIAL_SYMMETRY_AUDIT_V1.md'
COMPARISON_SHA = '66e626379dbdfc9c7c7c5116e94cada4a748e3db29ccb2c14cf96f6fe724537f'
RESULTS_SHA = '699fe50ebdfe2cd087147814787e8ee23b1267e3753c9ad92d5961382579d5f2'
MAX_CALLS, MAX_POINTS, WALL_SECONDS = 36, 7776, 1800
RSS_BYTES, ADDRESS_BYTES = 4*1024**3, 8*1024**3
Q_ATOL, Q_RTOL, ANGLE_ATOL = 5e-14, 5e-12, 1e-11
KERNEL_ATOL, KERNEL_RTOL = 1e-13, 1e-10
Q32_ATOL, Q32_RTOL = 2e-7, 2e-6
RGB_ATOL, RGB_RTOL, OBJECTIVE_ATOL, OBJECTIVE_RTOL = 2e-6, 2e-5, 1e-12, 5e-7
WI_RAW = [[.2,.4,1.],[-.4,.1,1.],[.6,-.3,1.],[-.2,-.7,1.],[.8,.2,.45]]
WO_RAW = [[.3,-.1,1.],[-.5,.35,1.],[.1,.6,.7]]
COMPONENTS = ('diffuse','specular','total')
SAMPLES = ('diffuse_directions','specular_directions','half_vectors','pdf')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit():
    # Read-only admission helpers only. Never call old restore/run/execute helpers.
    old, continuation, identities = previous.admit()
    path=ROOT/'docs/evidence/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.json'
    require(digest(path)==COMPARISON_SHA,'3C-4 evidence SHA-256 mismatch')
    comparison=json.loads(path.read_text())
    require(comparison['status']=='complete' and comparison['restoration_gate']=='passed' and
            comparison['initial_objective_gate']=='passed' and comparison['scene']==old['scene'] and
            comparison['tapes']=={k:old['tapes'][k] for k in ('fit','evaluation')} and
            comparison['shader']==old['shader'],'Incomplete/foreign comparison evidence')
    for flag in ('fixed_tapes_unchanged','global_torch_CPU_RNG_unchanged','global_numpy_RNG_unchanged'):
        require(comparison[flag] is True,'Comparison randomness not preserved')
    for name,sha in {**comparison['accepted_identities'],**comparison['implementation_sha256'],
                    'docs/protocols/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1.md':comparison['protocol_sha256']}.items():
        require(ROOT in (ROOT/name).resolve().parents and digest(ROOT/name)==sha,'Accepted identity changed: '+name)
        identities[name]=sha
    name='docs/MATERIAL_RECOVERY_OPTIMIZER_COMPARISON_V1_RESULTS.md'
    require(digest(ROOT/name)==RESULTS_SHA,'Accepted comparison results changed')
    identities[name]=RESULTS_SHA;identities[str(path.relative_to(ROOT))]=COMPARISON_SHA
    diagnostic=json.loads((ROOT/'docs/evidence/MATERIAL_RECOVERY_DIAGNOSTICS_V1.json').read_text())
    lm=[r for r in comparison['outcomes'] if r['arm']=='LM']
    require([r['seed'] for r in lm]==list(range(101,106)) and all(r['status']=='complete' for r in lm),
            'Exactly five completed LM endpoints required')
    for r in lm:
        p=r['final_state']['physical']
        a=np.asarray(p['values'],np.float32)
        require(p['dtype']=='torch.float32' and p['device']=='cpu' and p['shape']==[7] and
                reference.sha_array(a)==p['sha256'] and p['values']==r['final'] and
                np.array_equal(a.astype(np.float64),np.asarray(r['final'])),'LM physical-state identity mismatch')
    states=[dict(id='truth',nominal_physical=old['truth'],source='3C-1 truth; direct physical float32 shader input')]
    states += [dict(id='LM-{}'.format(r['seed']),nominal_physical=r['final'],source='3C-4 final LM state',seed=r['seed']) for r in lm]
    require(np.array_equal(np.asarray(old['truth']),reference.TRUTH),'Original truth changed')
    return old,diagnostic,comparison,states,identities


def transform(p,representation):
    p=np.asarray(p,np.float64)
    require(p.shape==(7,) and np.isfinite(p).all() and (.04<=p[:2]).all() and (p[:2]<=1).all() and
            ((p[3:]>=0)&(p[3:]<=1)).all(),'Finite original material bounds required')
    q=p.copy()
    if representation=='P1':q[:2]=p[1::-1];q[2]+=np.pi/2
    elif representation=='P2':q[2]+=np.pi
    else:require(representation=='P0','Unknown declared representation')
    return q


def tensor_q(p):
    p=np.asarray(p,np.float64)
    c,s=np.cos(p[2]),np.sin(p[2]);r=np.array([[c,-s],[s,c]],np.float64)
    return r@np.diag(np.asarray(p[:2])**2)@r.T


def matrix_difference(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    return dict(absolute_frobenius=float(np.linalg.norm(a-b)),
                relative_frobenius=float(np.linalg.norm(a-b)/np.linalg.norm(b)),max_absolute=float(np.max(abs(a-b))))


def physics(p):
    wi,wo=independent.unit(WI_RAW),independent.unit(WO_RAW)
    f0=.04*(1-p[6])+p[6]*p[3:6]
    kernels=np.stack([np.stack([independent.kernel(wi,o,p[0],p[1],p[2],f) for f in f0],-1) for o in wo])
    pdfs=np.stack([independent.directional_pdf(wi,o,p[0],p[1],p[2]) for o in wo])
    q=tensor_q(p);eigenvalues,eigenvectors=np.linalg.eigh(q)
    return dict(physical=np.asarray(p).tolist(),Q=q.tolist(),eigenvalues=eigenvalues.tolist(),
        major_axis=eigenvectors[:,-1].tolist(),anisotropy_gap=float((eigenvalues[1]-eigenvalues[0])/eigenvalues.sum()),
        independent_specular_kernel=kernels.tolist(),independent_fixed_direction_pdf=pdfs.tolist(),F0=f0.tolist())


def physical_equivalence(current,base,rounded=False):
    a,b=np.asarray(current['Q']),np.asarray(base['Q'])
    atol,rtol=(Q32_ATOL,Q32_RTOL) if rounded else (Q_ATOL,Q_RTOL)
    result=dict(Q=matrix_difference(a,b),kernel=matrix_difference(current['independent_specular_kernel'],base['independent_specular_kernel']),
                fixed_direction_pdf=matrix_difference(current['independent_fixed_direction_pdf'],base['independent_fixed_direction_pdf']))
    identifiable=min(current['anisotropy_gap'],base['anisotropy_gap'])>=.1
    if identifiable:
        u,v=np.asarray(current['major_axis']),np.asarray(base['major_axis'])
        d=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)
        angle=abs(.5*np.arctan2(np.sin(2*d),np.cos(2*d)))
    else:angle=None
    result.update(orientation_identifiable=identifiable,major_axis_error_radians=float(angle) if angle is not None else None,
        orientation_reason=None if identifiable else 'Near/isotropic orientation is unidentifiable; not scored as zero',
        eigenvalue_max_absolute=float(np.max(abs(np.asarray(current['eigenvalues'])-base['eigenvalues']))))
    checks=dict(Q=bool(np.allclose(a,b,atol=atol,rtol=rtol)),eigenvalues=bool(np.allclose(
        current['eigenvalues'],base['eigenvalues'],atol=atol,rtol=rtol)))
    if not rounded:
        checks.update(kernel=bool(np.allclose(current['independent_specular_kernel'],base['independent_specular_kernel'],
            atol=KERNEL_ATOL,rtol=KERNEL_RTOL)),fixed_direction_pdf=bool(np.allclose(
            current['independent_fixed_direction_pdf'],base['independent_fixed_direction_pdf'],atol=KERNEL_ATOL,rtol=KERNEL_RTOL)),
            orientation=angle is None or angle<=ANGLE_ATOL)
    result.update(checks=checks,passed=all(checks.values()),scope='Rounded-input Q/eigenvalue check; kernel/orientation differences descriptive' if rounded else 'Nominal float64 physical equivalence')
    return result


def array_identity(a,values=False):
    a=np.asarray(a);record=dict(dtype=str(a.dtype),shape=list(a.shape),sha256=reference.sha_array(a))
    if values:record['values']=a.tolist()
    return record


def difference(a,b):
    return original.radiance_errors(a,b)


def splits(observations):
    result={k:np.array([o['split']==k for o in observations]) for k in ('fit','held_out','relight')}
    return {k:v for k,v in result.items() if v.any()}


def radiance_comparison(a,b,observations):
    require(a['per_tape'].shape==b['per_tape'].shape==(4,len(observations),3,3),'Matching four-tape component/RGB arrays required')
    result={}
    for split,mask in splits(observations).items():
        result[split]={}
        for i,name in enumerate(COMPONENTS):
            current,base=a['mean'][mask,i],b['mean'][mask,i]
            result[split][name]=dict(four_tape_mean=difference(current,base),
                individual_tapes=[difference(a['per_tape'][j,mask,i],b['per_tape'][j,mask,i]) for j in range(4)],
                values_beyond_declared_RGB_roundoff_bound=int(np.count_nonzero(abs(current-base)>RGB_ATOL+RGB_RTOL*abs(base))))
    return result


def objective_difference(a,b):
    d=float(a-b);bound=OBJECTIVE_ATOL+OBJECTIVE_RTOL*abs(b)
    return dict(signed_difference=d,absolute_difference=abs(d),relative_signed_difference=d/max(abs(b),1e-12),
                relative_denominator_floor=1e-12,roundoff_bound=bound,beyond_roundoff_bound=abs(d)>bound)


def mask_summary(mask):
    require(mask.ndim==3 and mask.shape[0]==4 and mask.dtype==np.bool_,'Four-tape Boolean sample masks required')
    return dict(shape=list(mask.shape),sha256_packbits=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
                valid_counts=mask.sum(-1).tolist(),valid_total=int(mask.sum()),rejected_total=int(mask.size-mask.sum()))


def mask_difference(a,b):
    require(a.shape==b.shape and a.dtype==b.dtype==np.bool_,'Matching Boolean masks required')
    return dict(flipped_total=int(np.count_nonzero(a!=b)),valid_to_rejected=int(np.count_nonzero(b&~a)),
        rejected_to_valid=int(np.count_nonzero(~b&a)),per_tape_observation_flips=np.count_nonzero(a!=b,axis=-1).tolist())


def sampling_comparison(a,b):
    require(a['observation_identity']==b['observation_identity'] and a['tape_identities']==b['tape_identities'],
            'Common observation/sample tape identities required')
    fields={}
    for name in SAMPLES:
        x,y=a['samples'][name],b['samples'][name]
        require(x.shape==y.shape and x.shape[0]==4,'Common sample correspondence shape required')
        fields[name]=dict(indexwise_all_tapes=difference(x,y),individual_tapes=[difference(x[j],y[j]) for j in range(4)],
            meaning='Same random-index correspondence; different world samples are not a physical PDF/kernel invariance test')
        if name!='pdf':
            cosine=np.sum(x.astype(np.float64)*y,axis=-1)/(np.linalg.norm(x.astype(np.float64),axis=-1)*np.linalg.norm(y.astype(np.float64),axis=-1))
            angles=np.arccos(np.clip(cosine,-1,1))
            fields[name]['mean_angle_radians']=float(angles.mean());fields[name]['max_angle_radians']=float(angles.max())
    frames={name:difference(a['frames'][name],b['frames'][name]) for name in ('tangent','bitangent','normal')}
    ta,ba,tb,bb=[v[0,0] for v in (a['frames']['tangent'],a['frames']['bitangent'],b['frames']['tangent'],b['frames']['bitangent'])]
    return dict(fields=fields,masks=mask_difference(a['mask'],b['mask']),frame_differences=frames,
                first_frame_alignment_to_P0=[[float(ta@tb),float(ta@bb)],[float(ba@tb),float(ba@bb)]])


class Budget:
    def __init__(self,report):
        self.start=time.perf_counter();self.report=report
        report['work']=dict(shader_calls=0,completed_shader_calls=0,points=0,diffuse_directions=0,specular_directions=0,
            independent_kernel_scalar_values=0,independent_pdf_scalar_values=0)
    def check(self):
        if time.perf_counter()-self.start>=WALL_SECONDS:raise original.RuntimeLimit('Single1800-second audit boundary')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RSS_BYTES,'Declared4GiB peak RSS budget exceeded')
    def before_forward(self,points):
        self.check();w=self.report['work']
        require(w['shader_calls']<MAX_CALLS and w['points']+points<=MAX_POINTS,'Declared shader-call/point budget exceeded')
        w['shader_calls']+=1;w['points']+=points


class Capture:
    def __init__(self,shader):self.shader=shader;self.result=None;self.encoding=None
    def shade(self,*args,**kwargs):
        self.encoding=args[7][0].detach().clone()
        self.result=self.shader.shade(*args,**kwargs)
        return self.result


def capture_forward(shader,batch,p,budget,tape_ids):
    points=batch.n*batch.seeds;budget.before_forward(points)
    capture=Capture(shader)
    with torch.no_grad():mean=batch(capture,p)
    r=capture.result;require(r is not None,'Missing shader capture')
    w=budget.report['work'];w['completed_shader_calls']+=1
    w['diffuse_directions']+=points*shader.diffuse_samples;w['specular_directions']+=points*shader.specular_samples
    require(mean.shape==(batch.n,3,3) and mean.dtype==torch.float32 and not mean.requires_grad and
            bool(torch.isfinite(mean).all()),'Original finite float32 RGB component layout required')
    per=torch.stack([r[k].reshape(4,batch.n,3) for k in ('diffuse_rgb','specular_rgb','linear_rgb')],dim=2)
    samples={k:r[k].detach().numpy().reshape(4,batch.n,*r[k].shape[1:]) for k in SAMPLES}
    frames={k:r[k].detach().numpy().reshape(4,batch.n,3) for k in ('tangent','bitangent','normal')}
    require(all(np.isfinite(v).all() for v in (*samples.values(),*frames.values())),'Nonfinite sample/frame capture')
    require(samples['pdf'].shape==(4,batch.n,shader.specular_samples,1) and (samples['pdf']>=0).all(),'PDF shape/nonnegativity failed')
    for k in SAMPLES[:-1]:
        require(samples[k].shape==(4,batch.n,shader.diffuse_samples if k=='diffuse_directions' else shader.specular_samples,3) and
                np.max(abs(np.linalg.norm(samples[k],axis=-1)-1))<=2e-6,'Unit sampled direction/half-vector check failed: '+k)
    mask=np.sum(samples['specular_directions']*frames['normal'][:,:,None],axis=-1)>0
    require(np.array_equal(mask.sum(-1),r['valid_specular_counts'].detach().numpy().reshape(4,batch.n)),
            'Observed acceptance mask and original valid counts disagree')
    require(torch.allclose(per[:,:,0]+per[:,:,1],per[:,:,2],atol=2e-7,rtol=2e-6),'Diffuse/specular/total consistency failed')
    # Reconstruct physical Q from actual world T/B, using T0=+Y,B0=-X.
    project=np.array([[0.,1.,0.],[-1.,0.,0.]])
    t,b=frames['tangent'][0,0]@project.T,frames['bitangent'][0,0]@project.T
    frame_q=float(p[0])**2*np.outer(t,t)+float(p[1])**2*np.outer(b,b)
    require(np.allclose(frame_q,tensor_q(p.numpy()),atol=Q32_ATOL,rtol=Q32_RTOL),'Actual decoded frame and rounded physical Q disagree')
    c,s=decode_atan2(capture.encoding.reshape(1,2))
    budget.check()
    return dict(mean=mean.detach().numpy(),mean_tensor=mean.detach(),per_tape=per.detach().numpy(),samples=samples,frames=frames,
        mask=mask,encoding=capture.encoding.numpy(),decoder_coefficients=[float(c),float(s)],frame_Q=frame_q,
        observation_identity=hashlib.sha256(json.dumps(batch.observations,sort_keys=True).encode()).hexdigest(),tape_identities=tape_ids)


def reuse_saved(actual,saved,eligible,expected_sha=None,expected_dtype=None,reason=None):
    if not eligible:return actual,dict(reused=False,reason=reason)
    cached=np.asarray(saved,dtype=expected_dtype)
    require(reference.sha_array(cached)==expected_sha,'Saved prediction SHA-256 mismatch')
    require(cached.shape==actual.shape and np.array_equal(actual.astype(expected_dtype),cached),
            'Admitted exact saved/computed prediction identity mismatch; no retry')
    return cached.astype(np.float32),dict(reused=True,verified_bitwise=True,source_sha256=expected_sha,
                                        reason='Same admitted original ObservationBatch mean arithmetic and float32 physical inputs')


def serialize_forward(data,p,targets,observations,fit_target):
    mean,per=data['mean'],data['per_tape'];metrics={}
    for split,mask in splits(observations).items():
        metrics[split]={name:dict(four_tape_mean=difference(mean[mask,i],targets[mask,i]),
            individual_tapes=[difference(per[j,mask,i],targets[mask,i]) for j in range(4)]) for i,name in enumerate(COMPONENTS)}
    samples={}
    for name,value in data['samples'].items():
        samples[name]={**array_identity(value), 'first_observation_first4_samples_each_tape':value[:,0,:4].tolist(),
                       'minimum':float(value.min()),'maximum':float(value.max())}
    frames={k:{**array_identity(v),'first':v[0,0].tolist(),
               'max_difference_from_first':float(np.max(abs(v-v[0,0])))} for k,v in data['frames'].items()}
    result=dict(actual_physical_float32=array_identity(p.numpy(),True),actual_encoding_float32=data['encoding'].tolist(),
        decoder_cos_sin_float32=data['decoder_coefficients'],principal_angle_from_coefficients_float64=float(np.arctan2(*data['decoder_coefficients'][::-1])),
        decoded_frame_Q=data['frame_Q'].tolist(),mean_predictions=array_identity(mean,True),per_tape_predictions=array_identity(per,True),
        target_metrics=metrics,samples=samples,frames=frames,masks=mask_summary(data['mask']),
        observation_identity=data['observation_identity'],tape_identities=data['tape_identities'])
    if fit_target is not None:
        result['objective_float32']=float(torch.mean((data['mean_tensor'][:,2]-fit_target)**2))
        result['individual_tape_objectives_float32']=[float(torch.mean((torch.from_numpy(per[j,:,2])-fit_target)**2)) for j in range(4)]
    return result


def execute(old,diagnostic,comparison,states,report,persist):
    begin=time.perf_counter();cpu=time.process_time();threads=torch.get_num_threads()
    rng=torch.get_rng_state().clone();nrng=np.random.get_state();torch.set_num_threads(1);budget=Budget(report)
    try:
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',torch_threads=1)
        report['runtime']=runtime
        require(all(runtime[k]==comparison['runtime'][k] for k in runtime) and torch.get_default_dtype()==torch.float32,
                'Original numerical runtime/default dtype required')
        for stage,pin in PINS.items():baseline(stage,pin)
        report.update(states=[],phase='physical_validation',fixed_directions=dict(incoming=independent.unit(WI_RAW).tolist(),outgoing=independent.unit(WO_RAW).tolist()))
        # All eighteen nominal/rounded physical checks precede any shader evaluation.
        for state in states:
            row={**state,'representations':[]};report['states'].append(row);base64=base32=None
            for label in ('P0','P1','P2'):
                nominal=transform(state['nominal_physical'],label);rounded=nominal.astype(np.float32).astype(np.float64)
                f64,f32=physics(nominal),physics(rounded)
                report['work']['independent_kernel_scalar_values']+=90;report['work']['independent_pdf_scalar_values']+=30
                if base64 is None:base64,base32=f64,f32
                check=physical_equivalence(f64,base64);check32=physical_equivalence(f32,base32,rounded=True)
                rep=dict(id=label,nominal_float64=nominal.tolist(),rounded_float32=rounded.tolist(),physics_float64=f64,
                    physics_at_rounded_inputs=f32,physical_equivalence=check,rounded_equivalence=check32,
                    rounding_Q_difference=matrix_difference(f32['Q'],f64['Q']),forwards={},comparisons_to_P0={})
                row['representations'].append(rep);persist()
                require(check['passed'] and check32['passed'],'Physical-equivalence gate failed for '+state['id']+'/'+label)
                budget.check()
        report['physical_gate']='passed';persist()
        require(old['scene']==reference.design() and comparison['scene']==old['scene'],'Original scene/observation ordering changed')
        targets=np.asarray(old['targets']['fine'],np.float64)
        require(targets.shape==(84,3,3) and reference.sha_array(targets)==old['fine_targets_sha256'] and old['targets']['passed'],
                'Original independently generated targets invalid')
        target=torch.tensor(targets[:24,2],dtype=torch.float32)
        require(reference.sha_array(target.numpy())==old['fit_target_float32_sha256'],'Original fitting targets changed')
        shader=MaterialShaderV1('A2','atan2');require(shader.identity==old['shader']==comparison['shader'],'Accepted shader identity changed')
        report.update(shader=shader.identity,scene=old['scene'],fine_targets_sha256=old['fine_targets_sha256'],tapes={},phase='forward_comparisons')
        batches={};tapes=[]
        for phase,seeds,observations in [('fit',list(range(4)),old['scene']['observations'][:24]),
                                         ('evaluation',list(range(1000,1004)),old['scene']['observations'])]:
            tape,ids=original.make_tapes(seeds,len(observations));require(ids==old['tapes'][phase],'Original tape identity mismatch')
            tapes.append(tape);report['tapes'][phase]=ids
            batches[phase]=original.ObservationBatch(old['scene'],observations,tape,4)
        tape_hashes=[reference.sha_array(t.numpy()) for tape in tapes for t in (tape.u,tape.v,tape.azimuth)]
        for row in report['states']:
            report['active_state']=row['id'];lm=next((r for r in comparison['outcomes'] if r['arm']=='LM' and r['seed']==row.get('seed')),None)
            for phase,batch in batches.items():
                base_data=base_objective=None
                for rep in row['representations']:
                    report.update(active_phase=phase,active_representation=rep['id'])
                    p=torch.tensor(rep['rounded_float32'],dtype=torch.float32)
                    data=capture_forward(shader,batch,p,budget,report['tapes'][phase])
                    reused=dict(reused=False,reason='Transformed representation; no saved prediction substituted')
                    if rep['id']=='P0':
                        if lm is not None:
                            prefix='final_fit_' if phase=='fit' else 'final_'
                            values=lm[prefix+'predictions'];sha=lm['final_fit_prediction_sha256' if phase=='fit' else 'prediction_sha256']
                            cached,reused=reuse_saved(data['mean'],values,True,sha,np.float32 if phase=='fit' else np.float64)
                            data['mean']=cached
                        else:
                            old_forward=diagnostic['states'][0]['forward'] if phase=='fit' else diagnostic['truth_evaluation_tapes']
                            _,reused=reuse_saved(data['mean'],old_forward['mean_predictions'],False,
                                reason='3C-2 stacks before the mean; exact arithmetic identity is not assumed. One counted baseline captures unavailable sampling arrays.')
                            reused['observed_difference_to_old_mean']=difference(data['mean'],old_forward['mean_predictions'])
                        reused['baseline_forward_reason']='Per-tape LM predictions and raw sample/frame arrays were not saved; one capture forward, never an extra retry'
                    serial=serialize_forward(data,p,targets[:batch.n],batch.observations,target if phase=='fit' else None)
                    serial['saved_prediction_reuse']=reused;rep['forwards'][phase]=serial
                    if rep['id']=='P0':
                        base_data=data;base_objective=serial.get('objective_float32')
                        if lm is not None and phase=='fit':require(base_objective==lm['final_fit_MSE'],'Original LM fitting objective mismatch')
                    else:
                        compare=dict(radiance=radiance_comparison(data,base_data,batch.observations),sampling=sampling_comparison(data,base_data),
                                     encoding_difference=(data['encoding'].astype(np.float64)-base_data['encoding']).tolist())
                        if phase=='fit':compare['objective']=objective_difference(serial['objective_float32'],base_objective)
                        rep['comparisons_to_P0'][phase]=compare
                    persist()
            print('Forward symmetry audit complete:',row['id'],flush=True)
        w=report['work']
        require(w['shader_calls']==w['completed_shader_calls']==MAX_CALLS and w['points']==MAX_POINTS and
                w['diffuse_directions']==MAX_POINTS*512 and w['specular_directions']==MAX_POINTS*1024 and
                w['independent_kernel_scalar_values']==1620 and w['independent_pdf_scalar_values']==540,'Exact diagnostic work accounting failed')
        require(tape_hashes==[reference.sha_array(t.numpy()) for tape in tapes for t in (tape.u,tape.v,tape.azimuth)],'Fixed common tapes mutated')
        report.update(status='complete',phase='finished',fixed_tapes_unchanged=True)
    except BaseException as exc:
        report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc));raise
    finally:
        after=np.random.get_state()
        report['global_torch_CPU_RNG_unchanged']=torch.equal(rng,torch.get_rng_state())
        report['global_numpy_RNG_unchanged']=nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:]
        if not report['global_torch_CPU_RNG_unchanged'] or not report['global_numpy_RNG_unchanged']:
            report.update(status='failed',failure='Unexpected global RNG use')
        torch.set_num_threads(threads);report['threads_restored']=torch.get_num_threads()==threads
        report.update(wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);persist()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(all(os.environ.get(k)==v for k,v in original.REQUIRED_ENV.items()),'Explicit hidden-GPU/one-thread environment required')
    require(args.output.is_absolute() and args.output.resolve()==args.output and
            args.output==ROOT/'docs/evidence/MATERIAL_SYMMETRY_AUDIT_V1.json','Explicit fresh versioned output required')
    report=dict(schema_version=1,protocol='material-symmetry-audit-v1',milestone='3C-5',base_commit=BASE,status='running',
        physical_gate='not_started',protocol_sha256=digest(PROTOCOL),implementation_sha256={'experiments/material_symmetry_audit_v1.py':digest(__file__)},
        optimizer_creation=False,parameter_updates=False,new_targets=False,GPU_execution=False,
        settings=dict(max_shader_calls=MAX_CALLS,max_points=MAX_POINTS,wall_seconds=WALL_SECONDS,peak_RSS_budget_bytes=RSS_BYTES,
            address_space_budget_bytes=ADDRESS_BYTES,Q_atol=Q_ATOL,Q_rtol=Q_RTOL,orientation_atol_radians=ANGLE_ATOL,
            kernel_atol=KERNEL_ATOL,kernel_rtol=KERNEL_RTOL,Q32_atol=Q32_ATOL,Q32_rtol=Q32_RTOL,
            RGB_atol=RGB_ATOL,RGB_rtol=RGB_RTOL,objective_atol=OBJECTIVE_ATOL,objective_rtol=OBJECTIVE_RTOL),
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=[sys.executable,*sys.argv])
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
    old_limit=resource.getrlimit(resource.RLIMIT_AS)
    def stop(signum,frame):raise original.RuntimeLimit('Single1800-second forward-only audit terminated; no retry')
    with args.output.open('x') as stream:
        def persist():write_report(stream,report)
        persist()
        try:
            for s in handlers:signal.signal(s,stop)
            signal.alarm(WALL_SECONDS)
            cap=min(ADDRESS_BYTES,old_limit[0]) if old_limit[0]>=0 else ADDRESS_BYTES
            if old_limit[1]>=0:cap=min(cap,old_limit[1])
            resource.setrlimit(resource.RLIMIT_AS,(cap,old_limit[1]));report['effective_address_space_soft_limit']=cap
            old,diagnostic,comparison,states,identities=admit();report['accepted_identities']=identities
            execute(old,diagnostic,comparison,states,report,persist)
        except BaseException as exc:
            report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc));raise
        finally:
            resource.setrlimit(resource.RLIMIT_AS,old_limit);report['memory_limit_restored']=resource.getrlimit(resource.RLIMIT_AS)==old_limit
            signal.alarm(0)
            for s,handler in handlers.items():signal.signal(s,handler)
            report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();persist()
    print('Final status:',report['status'],flush=True)
    return 0 if report['status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
