"""ONE fixed, bounded CPU-only 3B-2 experiment; no grids/tolerances can be overridden."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from unittest.mock import patch

from audits import anisotropic_brdf as reference
from infrastructure.signatures import require,write_report
from infrastructure.stage_sources import ROOT,digest
from experiments import anisotropic_estimator_v1 as experiment

PROTOCOL_FILE=ROOT/'docs/ANISOTROPIC_ESTIMATOR_ABLATION_V1_PROTOCOL.md'
SETTINGS={'roughness':[[.04,.04],[.1,.6],[.3,.7],[.6,.9],[1.,1.]],'outgoing_degrees':[0,30,60,80,89],
    'outgoing_azimuth':0.,'theta':.31,'diffuse_samples':512,'specular_samples':1024,'seeds':list(range(16)),
    'quadrature_grids':[[256,512],[512,1024]],'convergence_absolute':2e-5,'standard_error_multiplier':3.,
    't95_df15':2.131449545559323,'roundoff_budget':16*float(np.finfo(np.float32).eps),'relative_bias_floor':1e-12,
    'metallic':.001,'kd':.999,'alpha_exponent':0.,'random_azimuth':True,'is_train':True,
    'lighting':['constant','cosine_squared','azimuthal_lobe'],'light_lobe_axis':[1.,1.,1.],
    'light_lobe_floor':.1,'light_lobe_amplitude':.9,'light_lobe_power':8,'torch_threads':1,
    'reciprocity_anchor':[.2,.5,1.],'decoder_samples':16,'external_timeout_seconds':600}


def summary(values,target=None,converged=True,reference_delta=0.):
    x=np.asarray(values,np.float64);require(x.shape==(16,) and np.isfinite(x).all(),'Exactly 16 finite paired seed observations required')
    mean=float(x.mean());variance=float(x.var(ddof=1));se=float(np.sqrt(variance/len(x)))
    row={'mean':mean,'sample_variance':variance,'standard_error':se,
         'mean_CI95':[mean-SETTINGS['t95_df15']*se,mean+SETTINGS['t95_df15']*se]}
    if target is not None:
        difference=mean-target;budget=3*se+SETTINGS['convergence_absolute']+SETTINGS['roundoff_budget']
        row.update(reference=float(target),reference_grid_delta=float(reference_delta),
            signed_bias=difference if converged else None,absolute_bias=abs(difference) if converged else None,
            relative_bias=difference/abs(target) if converged and abs(target)>SETTINGS['relative_bias_floor'] else None,
            absolute_relative_bias=abs(difference/target) if converged and abs(target)>SETTINGS['relative_bias_floor'] else None,
            comparison_budget=budget if converged else None,
            agreement=('within_predeclared_budget' if abs(difference)<=budget else 'outside_predeclared_budget') if converged else 'unresolved_reference',
            reason=None if converged else 'The fixed two-grid absolute convergence test failed; no accuracy claim')
    return row


def make_reference_grids():
    result=[]
    for mu,phi in SETTINGS['quadrature_grids']:
        directions,weights=reference.hemisphere(mu,phi)
        result.append((directions,weights,experiment.lighting(directions)))
    return result


def reference_integrals(mx,my,wo,grids):
    d=SETTINGS['kd']*(1-SETTINGS['metallic']);f0=.04*(1-SETTINGS['metallic'])+SETTINGS['metallic']*SETTINGS['kd'];levels=[]
    for directions,weights,lights in grids:
        spec=reference.kernel(directions,wo,mx,my,theta=SETTINGS['theta'],f0=f0)
        factor=weights*directions[:,2]
        diffuse=np.sum((factor*d/np.pi)[:,None]*lights,axis=0)
        specular=np.sum((factor*spec)[:,None]*lights,axis=0)
        levels.append(np.stack([diffuse,specular,diffuse+specular],axis=-1))
    coarse,fine=levels;delta=abs(coarse-fine);converged=delta<=SETTINGS['convergence_absolute']
    converged[:,2]=converged[:,0]&converged[:,1]&(delta[:,2]<=SETTINGS['convergence_absolute'])
    return {'coarse':coarse.tolist(),'fine':fine.tolist(),'absolute_grid_difference':delta.tolist(),
            'converged':converged.tolist(),'status':'converged' if converged.all() else 'partly_unresolved',
            'reason':None if converged.all() else 'Fixed grids disagree by more than 2e-5 in one or more components; no retry'}


def reciprocity(mx,my,wo):
    anchor=reference.unit(SETTINGS['reciprocity_anchor']);d=SETTINGS['kd']*(1-SETTINGS['metallic'])
    f0=.04*(1-SETTINGS['metallic'])+SETTINGS['metallic']*SETTINGS['kd'];theta=SETTINGS['theta']
    forward=float(reference.kernel(anchor,wo,mx,my,theta=theta,f0=f0));reverse=float(reference.kernel(wo,anchor,mx,my,theta=theta,f0=f0))
    pv=reference.acceptance(wo,mx,my,theta);pa=reference.acceptance(anchor,mx,my,theta)
    variants={}
    for name in experiment.VARIANTS:
        a,b=(forward,reverse) if name in ('B1','B3') else (forward/pv,reverse/pa)
        da,db=(d/np.pi,d/np.pi) if name in ('B2','B3') else (float(d*reference.diffuse_density(anchor[2])/anchor[2]),float(d*reference.diffuse_density(wo[2])/wo[2]))
        variants[name]={'specular_forward':a,'specular_reverse':b,'specular_absolute_difference':abs(a-b),
            'diffuse_forward':da,'diffuse_reverse':db,'total_forward':a+da,'total_reverse':b+db,
            'total_absolute_difference':abs(a+da-b-db)}
    return {'scope':'Ideal valid-frame continuum/large-M effective kernels, NOT finite-pattern observations or trained BRDFs',
        'anchor':anchor.tolist(),'P_view':pv,'P_anchor':pa,'intended_specular':[forward,reverse],
        'intended_total':[d/np.pi+forward,d/np.pi+reverse],'intended_absolute_difference':abs(forward-reverse),
        'variants':variants}


def decoder_diagnostic(shader,shared_stream=None):
    near=reference.unit([-1.,1e-7]);near_minus=reference.unit([-1.,-1e-7])
    encodings=[('negative_axis',[-1.,0.]),('zero',[0.,0.]),('unit_near_negative_plus',near),('unit_near_negative_minus',near_minus),
        ('positive_axis',[1.,0.]),('regular_unit',[np.cos(.62),np.sin(.62)]),('small_nonunit_x',[1e-8,0.]),('small_nonunit_y',[0.,1e-8])]
    shared_stream=experiment.sample_stream(0) if shared_stream is None else shared_stream
    rows=[];stream=[v[:,:SETTINGS['decoder_samples']] for v in shared_stream[:2]]
    for label,encoding in encodings:
        tape=experiment.RandomTape(stream[:2])
        with patch.object(torch,'rand',tape):
            value=shader.sample_aniso_ggx_directions(experiment.tensor([encoding]),experiment.tensor([[0,0,0]]),experiment.tensor([[.3]]),
                experiment.tensor([[.7]]),experiment.tensor([[0,0,1]]),SETTINGS['decoder_samples'],experiment.tensor([[0,0,1]]),None,device='cpu')
        h,t,b,n=value[0],value[4],value[5],value[6];length=torch.norm(h,dim=-1).numpy().reshape(-1)
        rows.append({'label':label,'encoding':np.asarray(encoding).tolist(),'encoding_norm':float(np.linalg.norm(encoding)),
            'T_norm':float(torch.norm(t)),'B_norm':float(torch.norm(b)),'N_norm':float(torch.norm(n)),
            'T_dot_B':float((t*b).sum()),'T_dot_N':float((t*n).sum()),'B_dot_N':float((b*n).sum()),
            'handedness':float((torch.cross(t,b,dim=-1)*n).sum()),'half_vector_norm_min':float(length.min()),
            'half_vector_norm_mean':float(length.mean()),'half_vector_norm_max':float(length.max()),
            'finite':all(bool(torch.isfinite(x).all()) for x in value),'random_calls':tape.calls})
    return {'unchanged_source_decoder':True,'unit_vectors_cannot_approach_zero':'A unit vector is always distance 1 from (0,0); near-zero nonunit encodings are labeled separately',
            'cases':rows,'half_vectors':len(rows)*SETTINGS['decoder_samples']}


def summarize_case(raw,refs):
    fine=np.asarray(refs['fine']);delta=np.asarray(refs['absolute_grid_difference']);valid=np.asarray(refs['converged']);out={}
    for v,name in enumerate(experiment.VARIANTS):
        out[name]=[[summary(raw[:,v,l,c],fine[l,c],bool(valid[l,c]),delta[l,c]) for c in range(3)] for l in range(3)]
    pairs={}
    for label,i,j in [('B1_minus_B0',1,0),('B2_minus_B0',2,0),('B3_minus_B0',3,0),('B3_minus_B1',3,1),('B3_minus_B2',3,2)]:
        pairs[label]=[]
        for light in range(3):
            row=[]
            for component in range(3):
                a,b=raw[:,i,light,component],raw[:,j,light,component]
                row.append({'estimate_or_signed_error_difference':summary(a-b),
                    'absolute_error_difference':summary(abs(a-fine[light,component])-abs(b-fine[light,component])) if valid[light,component] else None})
            pairs[label].append(row)
    return out,pairs


def compact_conclusions(cases,decoder):
    matches={};unresolved=[]
    for name in experiment.VARIANTS:
        matches[name]={}
        for c,label in enumerate(['diffuse','specular','total']):
            entries=[case['statistics'][name][l][c]['agreement'] for case in cases for l in range(3)]
            matches[name][label]={k:entries.count(k) for k in ['within_predeclared_budget','outside_predeclared_budget','unresolved_reference']}
    for case in cases:
        if case['references']['status']!='converged':unresolved.append({'mx':case['mx'],'my':case['my'],'angle':case['outgoing_degrees'],
            'absolute_grid_difference':case['references']['absolute_grid_difference'],'converged':case['references']['converged']})
    energy={name:sum(case['statistics'][name][0][2]['mean_CI95'][0]>1 for case in cases) for name in experiment.VARIANTS}
    return {'component_agreement_counts':matches,'unresolved_cases':unresolved,'unit_light_CI95_entirely_above_one':energy,
        'H1':'Fixed-M weights have the intended unconditioned valid-domain expectation; finite observations are counted separately',
        'H2':'Correct cosine measure recovered; fixed left-endpoint heights retain deterministic finite-pattern integration error',
        'H3':'Both component corrections combined without other changes; the numerical bounds are NOT assumed satisfied in every case',
        'H4':{'energy_not_guaranteed':energy['B3']>0,'B3_energy_counterexample_count':energy['B3'],'original_negative_axis_T_norm':decoder['cases'][0]['T_norm'],
              'decoder_changed':False,'geometry_or_training_admitted':False}}


def run(report,persist):
    start=time.perf_counter();cpu_start=time.process_time();old_threads=torch.get_num_threads();rng=torch.get_rng_state().clone();numpy_rng=np.random.get_state()
    require(torch.get_default_dtype()==torch.float32 and torch.tensor(0.).device.type=='cpu','Default original float32/CPU context required')
    torch.set_num_threads(1)
    budget={'cases':0,'draw_calls':0,'draw_float_values':0,'specular_half_vectors':0,'diffuse_directions':0,'radiance_evaluations':0,
            'sample_scalar_lighting_evaluations':0,'quadrature_direction_evaluations':0,'nonfinite_observations':None}
    report['budget']=budget;report['cases']=[]
    try:
        models,source=experiment.shaders();report['source']=source
        report['runtime']={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','torch_threads':torch.get_num_threads()}
        report['setup_seconds']=time.perf_counter()-start
        qstart=time.perf_counter();grids=make_reference_grids();report['quadrature_setup_seconds']=time.perf_counter()-qstart
        budget['reference_scalar_lighting_evaluations']=3*sum(len(g[0]) for g in grids)
        streams=[experiment.sample_stream(s) for s in SETTINGS['seeds']]
        stream_ids=[hashlib.sha256(b''.join(v.tobytes() for v in stream)).hexdigest() for stream in streams];report['paired_stream_sha256']=stream_ids
        budget['unique_private_uniforms']=sum(v.size for stream in streams for v in stream)
        report['layout']={'variants':list(experiment.VARIANTS),'lighting':SETTINGS['lighting'],'components':['diffuse','specular','total'],
            'seed_estimates_dimensions':'seed[16], variant[4], lighting[3], component[3]; achromatic RGB channels are equal'}
        with torch.no_grad():
            for mx,my in SETTINGS['roughness']:
                for angle in SETTINGS['outgoing_degrees']:
                    begin=time.perf_counter();radians=np.deg2rad(angle);wo=np.array([np.sin(radians),0.,np.cos(radians)])
                    qbegin=time.perf_counter();refs=reference_integrals(mx,my,wo,grids);qtime=time.perf_counter()-qbegin
                    budget['quadrature_direction_evaluations']+=sum(len(g[0]) for g in grids)
                    raw=[];counts=[];timings={n:{'sampling_seconds':0.,'radiance_seconds':0.} for n in experiment.VARIANTS}
                    for seed,stream in zip(SETTINGS['seeds'],streams):
                        values=[];pervariant=[]
                        for name in experiment.VARIANTS:
                            then=time.perf_counter();spec,diffuse,count=experiment.draw(models[name],mx,my,wo,stream,SETTINGS['theta'])
                            timings[name]['sampling_seconds']+=time.perf_counter()-then
                            require(float(torch.max(abs(torch.norm(spec[0],dim=-1)-1)))<2e-6 and abs(float(torch.norm(spec[4]))-1)<2e-6 and abs(float(torch.norm(spec[5]))-1)<2e-6,'Regular frame unexpectedly invalid; no decoder fix allowed')
                            then=time.perf_counter();values.append(experiment.estimate(models[name],spec,diffuse,wo,mx,my));timings[name]['radiance_seconds']+=time.perf_counter()-then
                            pervariant.append(count);budget['draw_calls']+=count['calls'];budget['draw_float_values']+=count['floats']
                            budget['specular_half_vectors']+=len(spec[0][0]);budget['diffuse_directions']+=len(diffuse[0]);budget['radiance_evaluations']+=3
                            budget['sample_scalar_lighting_evaluations']+=3*(len(spec[0][0])+len(diffuse[0]))
                        require(all(v==pervariant[0] for v in pervariant),'Variant sample counts/acceptance/random consumption differ')
                        counts.append({'seed':seed,'valid':pervariant[0]['valid'],'rejected':pervariant[0]['rejected']});raw.append(values)
                    raw=np.asarray(raw);stats,pairs=summarize_case(raw,refs);reciprocal=reciprocity(mx,my,wo)
                    acceptance=summary([row['valid']/SETTINGS['specular_samples'] for row in counts],reciprocal['P_view'])
                    case={'mx':mx,'my':my,'outgoing_degrees':angle,'outgoing_unit_direction_float64':wo.tolist(),
                        'references':refs,'reciprocity':reciprocal,'seed_estimates':raw.tolist(),'seed_counts':counts,
                        'acceptance':acceptance,'statistics':stats,'paired_error_differences':pairs,'finite':True,
                        'timing':{'quadrature_seconds':qtime,'variants':timings,'case_wall_seconds':time.perf_counter()-begin}}
                    report['cases'].append(case);budget['cases']+=1;persist()
                    print('completed case {}/25: mx={}, my={}, angle={}, reference={}'.format(budget['cases'],mx,my,angle,refs['status']),flush=True)
            report['decoder_diagnostic']=decoder_diagnostic(models['B0'],streams[0])
        require(torch.equal(rng,torch.get_rng_state()),'Infrastructure consumed global Torch CPU RNG')
        after=np.random.get_state();require(numpy_rng[0]==after[0] and np.array_equal(numpy_rng[1],after[1]) and numpy_rng[2:]==after[2:],'Global NumPy RNG changed')
        require(stream_ids==[hashlib.sha256(b''.join(v.tobytes() for v in stream)).hexdigest() for stream in streams],'Paired stream inputs mutated')
        budget['decoder_half_vectors']=report['decoder_diagnostic']['half_vectors'];budget['decoder_draw_calls']=sum(r['random_calls'] for r in report['decoder_diagnostic']['cases'])
        budget['decoder_requested_uniforms']=2*budget['decoder_half_vectors'];budget['nonfinite_observations']=0
        report['conclusions']=compact_conclusions(report['cases'],report['decoder_diagnostic'])
        report['global_RNG_unchanged']=True;report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        torch.set_num_threads(old_threads);report['threads_restored']=torch.get_num_threads()==old_threads
        report['wall_seconds']=time.perf_counter()-start;report['process_cpu_seconds']=time.process_time()-cpu_start;persist()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    require(args.output.resolve()==args.output and args.output.is_absolute(),'Explicit unaliased absolute fresh output required')
    report={'schema_version':1,'protocol':experiment.PROTOCOL,'status':'running','base_commit':'759a1253bf1db207221d6989ec4ca6f9c3ae3225',
        'settings':SETTINGS,'protocol_sha256':digest(PROTOCOL_FILE),'implementation_sha256':{n:digest(ROOT/n) for n in [
            'experiments/anisotropic_estimator_v1.py','experiments/run_anisotropic_ablation_v1.py','audits/anisotropic_brdf.py']},
        'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scientific_changes':['specular total-count denominator','cosine diffuse radial mapping only'],
        'training_integration':False,'neural_network_execution':False,'checkpoint_loading':False,'GPU_execution':False}
    with args.output.open('x') as stream:
        def persist():write_report(stream,report)
        persist()
        try:run(report,persist)
        finally:
            report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();persist()
    print('Complete: inspect unresolved references, finite-pattern bias, energy and decoder limitations.')


if __name__=='__main__':main()
