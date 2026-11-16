"""One authorized physical Bell recovery panel; no native neural training/rays."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

from benchmarks.material_inverse_e15 import (POLICY,STARTS,physical,valid_parameters,
    initial_parameters,correspondence_split,prepare,fit,fit_rows,metrics)
from benchmarks.material_information_e14 import deterministic_codes
from experiments import bell_two_light_e14 as prior
from experiments.analyze_bell_two_light_e14 import check_grid
from experiments.bell_highlight_e11d import write_json
from infrastructure.signatures import require,read_report
from infrastructure.stage_sources import ROOT,PINS,baseline,digest
from infrastructure.verified_ray_backend import identity

PROTOCOL='bell-physical-recovery-e15-v1'
E14_REPORT='215eb11f9dd0a2fff9074cf6bab5eca6be832402a1860e3dcf27138f8458bab5'
LIMITS=dict(wall_seconds=3600,grace_seconds=30,inner_seconds=3480,cpu_threads=4,
    host_memory_gib=64,cuda_allocator_fraction=.25,output_bytes=1024**3,
    fit_point_directions=137908715520,endpoint_point_directions=3460300800,
    gate_point_directions=519045120,environment_point_directions=55050240,
    coefficient_point_directions=72089600,new_geometry_queries=0,new_references=0,
    neural_calls=0,Adam_returns=0)
PANELS=tuple(dict(material=m,observation=o,ratio=r,lights=l) for m in ('original','stronger')
             for o in ('linear','codes8') for r in ('fixed','free') for l in (1,2))
CALL_LIMITS=dict(fit=1280,gate=9,endpoint=32)
JACOBIAN_LIMITS=dict(fit=320,gate=1,endpoint=0)


def source_paths():
    return sorted(set(prior.source_paths())|{'benchmarks/material_inverse_e15.py',
        'experiments/bell_material_inverse_e15.py','experiments/run_bell_material_inverse_e15.sh',
        'run_bell_material_inverse_e15.py','tests/benchmarks/test_material_inverse_e15.py',
        'docs/protocols/BELL_PHYSICAL_RECOVERY_E15_V1.md',
        'docs/protocols/BELL_TWO_LIGHT_OBSERVABILITY_E14_V1.freeze.json',
        'docs/BELL_TWO_LIGHT_E14_RESULTS_V1.md','docs/evidence/BELL_TWO_LIGHT_E14_RESULTS_V1.json',
        'experiments/analyze_bell_two_light_e14.py'})


def record(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))


def load_inputs(args):
    data=prior.inputs(**{k:getattr(args,k) for k in ('benchmark','pilot','reference','factor','multiview')})
    labels,pixels,cache,outgoing,records=data
    rp=args.two_light/'two-light.json';require(digest(rp)==E14_REPORT,'Accepted E14 report changed')
    r=json.loads(rp.read_text());require(r['status']=='complete' and r['cleanup_verified'] is True and
        r['scientific_work_complete'] is True and r['comparison']['all_cases_converged'] is True and
        r['comparison']['decision']=='FROZEN_SCREEN_NOT_MET' and r['cached_tracing_identity']==identity(),
        'E14 completion/failed-screen identity differs')
    records.append(record(rp))
    for row in r['outputs']:
        p=args.two_light/row['path'];require(p.resolve()==p and args.two_light in p.parents and
            p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'E14 artifact changed')
        records.append(record(p))
    targets={}
    for size in (256,512):
        with np.load(args.two_light/('grid-'+str(size)+'.npz'),allow_pickle=False) as z:
            targets[size]=check_grid({k:z[k] for k in z.files},outgoing['valid'],pixels,outgoing['camera_ids'],cache[size]['L2'])
    train,held=correspondence_split(outgoing['valid'])
    require(int(train.sum())==137 and int(held.sum())==83,'Predeclared geometric holdout differs')
    return labels,pixels,cache,outgoing,records,targets,train,held


def logical_inputs(records,args):
    roots={k:getattr(args,k) for k in ('benchmark','pilot','reference','factor','multiview','two_light')}
    out=[]
    for r in records:
        p=Path(r['path']);names=[k for k,v in roots.items() if v in p.parents]
        require(len(names)==1,'Ambiguous input root');name=names[0]
        out.append(dict(root=name,path=p.relative_to(roots[name]).as_posix(),bytes=r['bytes'],sha256=r['sha256']))
    return out


def definition(args):
    f,_=read_report(args.freeze,args.freeze_sha256)
    require(f['protocol']==PROTOCOL and f['limits']==LIMITS and f['policy']==POLICY and
        f['starts']==[list(x) for x in STARTS] and f['panels']==list(PANELS) and
        f['forward_call_limits']==CALL_LIMITS and f['jacobian_call_limits']==JACOBIAN_LIMITS and
        f['E14_report_sha256']==E14_REPORT and set(f['files'])==set(source_paths()),'E15 definition mismatch')
    for p,h in f['files'].items():require(digest(ROOT/p)==h,'Frozen source changed: '+p)
    data=load_inputs(args);require(logical_inputs(data[4],args)==f['inputs'],'Frozen input identities differ')
    require(f['train_mask']==data[6].tolist() and f['held_mask']==data[7].tolist(),'Frozen geometric masks differ')
    return f,data


def authorize(args,f):
    a,_=read_report(args.authorization,args.authorization_sha256)
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and
        a.get('commit')==args.commit and a.get('freeze_sha256')==args.freeze_sha256 and
        a.get('limits')==LIMITS and a.get('image_id')==args.image_id==identity()['image_id'] and
        a.get('output_directory')==str(args.output),'Separate exact E15 authorization required')
    require(os.geteuid()!=0 and args.output.is_absolute() and args.output.resolve()==args.output and
        args.output.parent==Path('/outputs') and not args.output.exists(),'Fresh non-root owned output required')
    for p in (Path('/'),ROOT,args.freeze,args.authorization,*[getattr(args,k) for k in
              ('benchmark','pilot','reference','factor','multiview','two_light')]):
        require(os.statvfs(p).f_flag&os.ST_RDONLY,'Read-only source/inputs/root required')
    require(not os.statvfs('/outputs').f_flag&os.ST_RDONLY,'Writable outputs required')
    require({s.split(':')[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'Network must be disabled')


def truth_parameters(labels,strong):
    w=labels['widths_major_minor'].copy()
    if strong:
        scale=np.sqrt(w[:,0]/w[:,1]);w=np.stack((w[:,0]*scale,w[:,1]/scale),-1)
    p=np.stack((labels['theta_major'],np.log(np.sqrt(w.prod(-1))),np.log(w[:,0]/w[:,1])),-1)
    require(valid_parameters(p).all(),'Invalid admitted truth')
    return p[:,None]


def captured_target(components,arm,views,points):
    return np.stack([components[arm,light,views,points,0,2] for light in range(2)],1)


def numerical_gates(prepared,data,r):
    labels,pixels,cache,outgoing,records,targets,train,held=data
    views,points=np.nonzero(outgoing['valid']);prepared.phase='gate'
    parity=[]
    for arm in range(2):
        expected=captured_target(targets[512],arm,views,points)
        y,_=prepared.evaluate(truth_parameters(labels,arm==1),False)
        error=float(abs(y[:,0]-expected).max())
        require(np.allclose(y[:,0],expected,atol=2e-12,rtol=2e-10),'Factored finite-grid forward differs from accepted E14')
        parity.append(dict(material=arm,maximum_RGB_difference=error))
    p=initial_parameters(84)[:,:1];y,j=prepared.evaluate(p,True);errors=[];h=1e-5
    for k in range(3):
        a=p.copy();b=p.copy();a[...,k]+=h;b[...,k]-=h
        ya,_=prepared.evaluate(a,False);yb,_=prepared.evaluate(b,False);fd=(ya-yb)/(2*h)
        error=float(abs(j[...,k]-fd).max())
        require(np.all(abs(j[...,k]-fd)<=2e-8+5e-5*abs(fd)),'Native analytic derivative gate failed')
        errors.append(dict(coordinate=k,maximum_absolute_error=error))
    r['admission']=dict(status='pass',forward_parity=parity,derivative_errors=errors,
                        finite_difference_step=h,derivative_absolute=2e-8,derivative_relative=5e-5)


def endpoint(prepared,coarse,parameters,truth,target,mode,lights,train,held,views,points,labels,outfile):
    from benchmarks.generate_bell_e11 import write_arrays
    prepared.phase=coarse.phase='endpoint'
    y,_=prepared.evaluate(parameters,False);low,_=coarse.evaluate(parameters,False)
    errors=metrics(parameters,truth,(labels['T'],labels['B']));true=target[:,None]
    ycodes=deterministic_codes(y);tcodes=deterministic_codes(target)
    w=physical(parameters);tw=physical(truth)
    eligible=(tw[...,0]-tw[...,1])>=.1;defined=(w[...,0]-w[...,1])>1e-6
    masks=dict(train_views=train[views,points],held_views=held[views,points])
    use=masks['train_views'];observation=target if mode=='linear' else deterministic_codes(target)
    training,_,_=fit_rows(y[use],np.zeros(y[use].shape+(3,)),observation[use],mode,points[use],lights)
    selected=np.argmin(training,axis=1)  # Exact training objective only; first start wins an exact tie.
    payload=dict(parameters=parameters,truth=truth,predicted_RGB=y,coarse_RGB=low,target_RGB=target,
        predicted_codes=ycodes,target_codes=tcodes,orientation_eligible=eligible,orientation_prediction_defined=defined,
        observation_view_indices=views,observation_point_indices=points,training_objective=training,
        selected_by_training=selected,clipped_prediction_channels=((y<0)|(y>1)).sum((0,2,3)),**errors)
    for name,mask in masks.items():payload[name]=mask
    write_arrays(outfile,**payload)
    # No held-out quantity is returned to fitting or start selection.
    return dict(endpoint=record(outfile),maximum_grid_RGB_difference=float(abs(y-low).max()),
        grid_relative_L2=float(np.linalg.norm(y-low)/max(np.linalg.norm(y),1e-15)),
        Q_relative_median_by_start=np.median(errors['Q_world_relative'],axis=0).tolist(),
        Q_005_pass_by_start=(errors['Q_world_relative']<=.05).sum(0).tolist(),
        orientation_eligible_points=int(eligible.sum()),
        orientation_5deg_pass_by_start=((errors['orientation_degrees']<=5)&eligible&defined).sum(0).tolist(),
        interpretation='Oracle fixed-ratio/same-kernel diagnostic, not practical neural recovery. Endpoint selection remains fitting-only.')


def native(args,data,r,start,torch):
    from benchmarks.generate_bell_e11 import write_arrays
    require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and
        torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0),'Reviewed native runtime required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    require(digest(Path(identity()['binary_path']))==identity()['binary_sha256'],'Reviewed image dependency changed')
    for stage in PINS:baseline(stage,PINS[stage])
    r['runtime']=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    labels,pixels,cache,outgoing,records,targets,train,held=data;views,points=np.nonzero(outgoing['valid'])
    fitobs=np.flatnonzero(train[views,points]);r['split']=dict(train_mask=train.tolist(),held_mask=held.tolist(),
        pixel_ids=pixels.tolist(),camera_ids=outgoing['camera_ids'].tolist(),missing_heldout_points=np.flatnonzero(~held.any(0)).tolist())
    def charge(phase,n,jacobian):
        key=phase+'_point_directions';limit=LIMITS[key]
        require(time.monotonic()-start<LIMITS['inner_seconds']-30 and r['attempted'][key]+n<=limit and
            sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],
            'E15 fixed scientific/resource ceiling reached')
        if phase in CALL_LIMITS:
            require(r['forward_attempts'][phase]<CALL_LIMITS[phase] and
                (not jacobian or r['analytic_jacobian_attempts'][phase]<JACOBIAN_LIMITS[phase]),
                'E15 forward/Jacobian call ceiling reached')
            r['forward_attempts'][phase]+=1
            if jacobian:
                r['analytic_jacobian_attempts'][phase]+=1;r['analytic_derivative_point_directions'][phase]+=3*n
        r['attempted'][key]+=n
        # Pure finite-grid array contractions are synchronous at the returned CPU capture.
        # Completed work is acknowledged below after each evaluated/prepared operation.
    def ready(size):
        c=prepare(torch,labels,cache[size],outgoing,list(prior.LIGHTS),charge)
        for key in ('environment_point_directions','coefficient_point_directions'):r['completed'][key]=r['attempted'][key]
        return c
    with torch.no_grad():
        fine=ready(512);coarse=ready(256)
        # Wrap each evaluator to mark completion only after its finite CPU capture returns.
        for prepared in (fine,coarse):
            original=prepared.evaluate
            def wrapped(*a,_prepared=prepared,_evaluate=original,**kw):
                result=_evaluate(*a,**kw);key=_prepared.phase+'_point_directions'
                r['completed'][key]=r['attempted'][key]
                r['forward_returns'][_prepared.phase]=r['forward_attempts'][_prepared.phase]
                r['analytic_jacobian_returns'][_prepared.phase]=r['analytic_jacobian_attempts'][_prepared.phase]
                return result
            prepared.evaluate=wrapped
        numerical_gates(fine,data,r)
        for index,panel in enumerate(PANELS):
            require(r['admission']['status']=='pass','All admission gates before any optimization')
            name='panel-'+str(index).zfill(2);r['active_panel']=dict(index=index,**panel)
            truth=truth_parameters(labels,panel['material']=='stronger')
            initial=initial_parameters(84,truth[:,0,2] if panel['ratio']=='fixed' else None)
            target=captured_target(targets[512],int(panel['material']=='stronger'),views,points)
            observed=target if panel['observation']=='linear' else deterministic_codes(target)
            fine.phase='fit';rows=[]
            def capture(row):
                rows.append(row);p=args.output/(name+'-progress.json');tmp=p.with_suffix('.next')
                tmp.write_text(json.dumps(dict(panel=panel,history=rows,attempted=r['attempted'],completed=r['completed']),allow_nan=False)+'\n');tmp.replace(p)
            result=fit(lambda p,j:fine.evaluate(p,j,fitobs),observed[fitobs],points[fitobs],initial,
                panel['observation'],panel['lights'],panel['ratio']=='fixed',capture)
            write_json(args.output/(name+'-trajectory.json'),dict(panel=panel,initial=initial.tolist(),
                termination=result['termination'].tolist(),history=result.pop('history')))
            write_arrays(args.output/(name+'-optimizer.npz'),**{k:result[k] for k in ('parameters','accepted','trials','damping')})
            report=endpoint(fine,coarse,result['parameters'],truth,target,panel['observation'],panel['lights'],train,held,views,points,labels,args.output/(name+'-endpoint.npz'))
            report.update(panel=panel,index=index,accepted_updates=int(result['accepted'].sum()),
                trial_attempts=int(result['trials'].sum()),termination_counts={str(k):int(v) for k,v in zip(*np.unique(result['termination'],return_counts=True))})
            r['panels'].append(report);print(json.dumps(report),flush=True)
        r.pop('active_panel',None)
        require(r['attempted']==r['completed'] and r['forward_attempts']==r['forward_returns'] and
            r['analytic_jacobian_attempts']==r['analytic_jacobian_returns'] and len(r['panels'])==16,'Incomplete authorized panel work')
        require(not any(k.split('.')[0] in ('network','train','raytracing','_raytracing') for k in sys.modules),'Forbidden native scientific import')
        import resource
        r.update(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def execute(args,f,data):
    authorize(args,f);args.output.mkdir();start=time.monotonic()
    keys=[k for k in LIMITS if k.endswith('_point_directions')]
    r=dict(schema_version=1,protocol=PROTOCOL,status='incomplete',cleanup_verified=False,scientific_work_complete=False,
        commit=args.commit,freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,
        files=f['files'],inputs=data[4],limits=LIMITS,policy=POLICY,starts=[list(x) for x in STARTS],
        panels=[],attempted={k:0 for k in keys},completed={k:0 for k in keys},new_geometry_queries=0,new_reference_datasets=0,
        forward_attempts={k:0 for k in CALL_LIMITS},forward_returns={k:0 for k in CALL_LIMITS},
        analytic_jacobian_attempts={k:0 for k in CALL_LIMITS},analytic_jacobian_returns={k:0 for k in CALL_LIMITS},
        analytic_derivative_point_directions={k:0 for k in CALL_LIMITS},autograd_reverse_calls=0,
        neural_calls=0,Adam_returns=0,backend=identity())
    try:
        import torch
        from benchmarks.generate_bell_e11 import run_with_cleanup
        run_with_cleanup(lambda:native(args,data,r,start,torch),torch)
        r['cleanup_verified']=True
        require(all(digest(ROOT/p)==h for p,h in f['files'].items()) and
                all(digest(Path(x['path']))==x['sha256'] for x in data[4]),'Terminal source/input integrity failed')
        require(time.monotonic()-start<LIMITS['inner_seconds'] and
                sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes']-1024**2,
                'Terminal resource ceiling exceeded')
        r['outputs']=[record(p) for p in sorted(args.output.iterdir()) if p.is_file()]
        for x in r['outputs']:x['path']=str(Path(x['path']).relative_to(args.output))
        r.update(status='complete',scientific_work_complete=True)
    except BaseException as exc:
        r.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
    finally:
        r['elapsed_seconds']=time.monotonic()-start;write_json(args.output/'recovery.json',r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark','pilot','reference','factor','multiview','two-light','freeze'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True);p.add_argument('--execute',action='store_true')
    for name in ('authorization','output'):p.add_argument('--'+name,type=Path)
    for name in ('commit','image-id','authorization-sha256'):p.add_argument('--'+name)
    a=p.parse_args();f,data=definition(a)
    if not a.execute:
        print(json.dumps(dict(status='metadata_only',panels=16,point_histories=4032,train_observations=137,heldout_observations=83,new_scientific_calls=0)));return
    require(all((a.authorization,a.output,a.commit,a.image_id,a.authorization_sha256)),'Separate exact scientific approval required')
    execute(a,f,data)


if __name__=='__main__':main()
