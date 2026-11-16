"""One explicitly authorized native full-field E16 paired experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np

from benchmarks.neural_observation_e16 import POLICY,LIMITS,teacher,world_Q,compatibility,compute_match
from experiments import bell_material_inverse_e15 as previous
from experiments.bell_highlight_e11d import write_json
from infrastructure.stage_sources import ROOT,PINS,digest,baseline
from infrastructure.signatures import require,read_report
from infrastructure.verified_ray_backend import identity

PROTOCOL='bell-observation-guided-neural-e16-v1'
E15_REPORT='78245f0ab91c2d18b83c87e36556c79e618859ccfeb7f8cd956988a98c38178a'
FIELDS=('original','stronger')


def source_paths():
    return sorted(set(previous.source_paths())|{'benchmarks/neural_observation_e16.py',
        'experiments/bell_neural_material_e16.py','experiments/run_bell_neural_material_e16.sh',
        'run_bell_neural_material_e16.py','tests/benchmarks/test_neural_observation_e16.py',
        'docs/protocols/BELL_NEURAL_OBSERVATION_E16_V1.md',
        'docs/protocols/BELL_PHYSICAL_RECOVERY_E15_V1.freeze.json',
        'docs/BELL_PHYSICAL_RECOVERY_E15_RESULTS_V1.md','docs/evidence/BELL_PHYSICAL_RECOVERY_E15_RESULTS_V1.json',
        'scientific_variants/material_a_v1.py','infrastructure/material_variants.py',
        'infrastructure/material_recovery_e13_evaluation.py','infrastructure/bell_material_rays.py',
        'configs/material_corrected/bell_core_known_v1.yaml','tests/scientific/test_native_material_a_e10.py'})


def record(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))


def load_inputs(args):
    data=previous.load_inputs(args);labels,pixels,cache,outgoing,records,targets,train,held=data
    rp=args.physical/'recovery.json';require(digest(rp)==E15_REPORT,'Accepted E15 report changed')
    old=json.loads(rp.read_text());require(old['status']=='complete' and old['cleanup_verified'] is True and
        old['scientific_work_complete'] is True,'Incomplete accepted teacher execution')
    records.append(record(rp));teachers={};references={};confidence={};cost={}
    for material,index in zip(FIELDS,(3,11)):
        require(old['panels'][index]['panel']==dict(material=material,observation='linear',ratio='free',lights=2),
                'Only the admitted free/two-light/linear teacher is allowed')
        paths={}
        for suffix in ('endpoint.npz','trajectory.json'):
            p=args.physical/('panel-%02d-%s'%(index,suffix));item=next(x for x in old['outputs'] if x['path']==p.name)
            require(p.resolve()==p and p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'Teacher artifact changed')
            records.append(record(p));paths[suffix]=p
        history=json.loads(paths['trajectory.json'].read_text())
        forwards=len(history['history'])
        forwards+=sum(bool(np.asarray(trial['eligible']).any()) for row in history['history'] for trial in row['trials'])
        cost[material]=dict(fitting_forward_calls=forwards,analytic_Jacobian_calls=len(history['history']),
            fitting_point_directions=forwards*137*3*512**2,endpoint_point_directions=3*220*(512**2+256**2),
            per_panel_wall_time_available=False,charged_wall_seconds=float(old['elapsed_seconds']))
        with np.load(paths['endpoint.npz'],allow_pickle=False) as z:
            p,Q,c=teacher(z,history,labels['T'],labels['B']);selected=z['selected_by_training']
            ids=z['observation_point_indices'];views=z['observation_view_indices']
            require(np.array_equal((views,ids),np.nonzero(outgoing['valid'])),'Teacher observation IDs differ')
            references[material]={k:z[k][np.arange(len(ids)),selected[ids]].copy() for k in ('predicted_RGB','coarse_RGB')}
            teachers[material]=dict(parameters=p,Q=Q);confidence[material]=c
    known={k:labels[k] for k in ('position','normals','T','B','color','metallic')}
    return dict(legacy=data,known=known,teachers=teachers,compat_reference=references,confidence=confidence,
        records=records,teacher_wall_charge=float(old['elapsed_seconds']),teacher_cost=cost,
        teacher_context=dict(full_E15_work=old['completed'],full_E15_elapsed_seconds=float(old['elapsed_seconds']),
            cost_policy='Full prior-run time conservatively charged to each guided field; no amortized speedup claim.'),
        mesh=args.benchmark/'geometry/mesh.ply')


def logical_inputs(records,args):
    roots={k:getattr(args,k) for k in ('benchmark','pilot','reference','factor','multiview','two_light','physical')}
    out=[]
    for row in records:
        p=Path(row['path']);matches=[k for k,v in roots.items() if v in p.parents];require(len(matches)==1,'Ambiguous input root')
        name=matches[0];out.append(dict(root=name,path=p.relative_to(roots[name]).as_posix(),bytes=row['bytes'],sha256=row['sha256']))
    return out


def definition(args):
    f,_=read_report(args.freeze,args.freeze_sha256)
    require(f['protocol']==PROTOCOL and f['policy']==POLICY and f['limits']==LIMITS and
        set(f['files'])==set(source_paths()) and f['E15_report_sha256']==E15_REPORT,'E16 definition mismatch')
    for p,h in f['files'].items():require(digest(ROOT/p)==h,'Frozen source changed: '+p)
    d=load_inputs(args);require(logical_inputs(d['records'],args)==f['inputs'],'Frozen inputs differ')
    require(d['teacher_wall_charge']==f['teacher_wall_charge_seconds'],'Teacher compute charge changed')
    return f,d


def authorize(args):
    a,_=read_report(args.authorization,args.authorization_sha256)
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==args.commit and
        a.get('freeze_sha256')==args.freeze_sha256 and a.get('limits')==LIMITS and
        a.get('image_id')==args.image_id==identity()['image_id'] and a.get('output_directory')==str(args.output),
        'Separate exact E16 execution authorization required')
    require(os.geteuid()!=0 and args.output.is_absolute() and args.output.resolve()==args.output and
        args.output.parent==Path('/outputs') and not args.output.exists(),'Fresh non-root output required')
    for p in (Path('/'),ROOT,args.authorization,args.freeze,*[getattr(args,k) for k in
              ('benchmark','pilot','reference','factor','multiview','two_light','physical')]):
        require(os.statvfs(p).f_flag&os.ST_RDONLY,'Read-only original inputs/source/root required')
    require(not os.statvfs('/outputs').f_flag&os.ST_RDONLY,'Writable owned output required')
    require({s.split(':')[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No network required')


def state_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def shader_config():
    import yaml
    c=yaml.safe_load((ROOT/'configs/material_corrected/bell_core_known_v1.yaml').read_text())['shader_cfg']
    c['known_environment']=previous.prior.LIGHTS[0]
    return c


class Native:
    def __init__(self,torch,data,output,report,check):
        self.t=torch;self.data=data;self.out=output;self.r=report;self.check=check;self.phase='gate'
        self.r['phase']='native_construction'
        from infrastructure.verified_ray_backend import load
        from infrastructure.stage_sources import activate,imported_sources
        load('predicate_fp64',identity()['image_id']);science,_=activate('material',PINS['material'])
        from network import field,renderer
        from scientific_variants.material_a_v1 import native_classes
        from infrastructure.bell_material_rays import original_adapter
        self.shader_class,_=native_classes(field.MCShadingNetwork,renderer.NeROMaterialRenderer,'A_corrected_core_v1',field.linear_to_srgb)
        # Explicitly selected admitted oracle mesh; no new geometry reconstruction.
        self.owner=original_adapter(renderer.NeROMaterialRenderer,dict(mesh=str(data['mesh'])))
        self.known={k:torch.as_tensor(v,dtype=torch.float32,device='cuda') for k,v in data['known'].items()}
        self.known['positions']=self.known.pop('position')
        positions=data['known']['position'].astype(np.float32)
        require(positions.shape==(84,3) and len(np.unique(positions,axis=0))==84,'Distinct original field coordinates required')
        self.r['field_coordinates_sha256']=hashlib.sha256(positions.tobytes()).hexdigest()
        self.views,self.points=np.nonzero(data['legacy'][3]['valid'])
        self.wo=torch.as_tensor(data['legacy'][3]['outgoing_directions'][self.views,self.points],dtype=torch.float32,device='cuda')
        self.fit=np.flatnonzero(data['legacy'][6][self.views,self.points]);self.all=np.arange(len(self.points))
        self.models={}
        for material in FIELDS:
            self.r['phase']='construct_'+material
            for arm in ('A','B'):
                torch.manual_seed(POLICY['seed']);torch.cuda.manual_seed_all(POLICY['seed'])
                self.models[material,arm]=self.shader_class(shader_config(),self.trace).cuda()
        hashes={k:state_hash(v) for k,v in self.models.items()}
        require(len(set(hashes.values()))==1,'All initial full shader weights/buffers must match before training')
        names=('feats_network','mx_predictor','my_predictor','metallic_predictor','rotation_predictor','kd_predictor')
        model=next(iter(self.models.values()));material_params=[p for n,p in model.named_parameters() if n.split('.')[0] in names]
        require(sum(p.numel() for p in material_params)==1485584 and len(material_params)==84 and
            all(p.requires_grad for m in self.models.values() for p in m.parameters()),'Original full material capacity/trainability changed')
        self.r['initial_state_sha256']=next(iter(hashes.values()));self.r['material_parameter_count']=1485584
        self.inactive={k:{n:p.detach().cpu().clone() for n,p in m.named_parameters() if n.split('.')[0] in
            ('metallic_predictor','kd_predictor','inner_light','outer_light')} for k,m in self.models.items()}
        self.r['full_shader_parameter_count']=sum(p.numel() for p in model.parameters())
        self.r['imported_pinned_sources']=imported_sources(science);self.r['effective_shader']=model.cfg
        T,B=model.compute_tangent_bitangent_flat(self.known['normals'],None)
        frame=max(float(abs(T-self.known['T']).max()),float(abs(B-self.known['B']).max()))
        require(frame<=2e-6,'Native/accepted tangent-frame mismatch; no frame repair')
        self.r['native_reference_frame_max_abs']=frame
        self.prepare_reference()

    def bump(self,key,n=1):
        self.check();require(key in LIMITS and self.r['attempted'][key]+n<=LIMITS[key],'E16 fixed work cap: '+key)
        self.r['attempted'][key]+=n

    def done(self,key,n=1):self.r['completed'][key]+=n

    def trace(self,o,d):
        n=len(o);self.bump('shadow_rays',n);result=self.owner.trace(o,d);self.done('shadow_rays',n);return result

    def prepare_reference(self):
        from benchmarks.material_inverse_e15 import prepare
        self.references={}
        def charge(phase,n,j):
            key=('independent' if phase=='independent' else phase)+'_point_directions';self.bump(key,n)
        for size in (512,256):
            c=prepare(self.t,self.data['known'],self.data['legacy'][2][size],self.data['legacy'][3],list(previous.prior.LIGHTS),charge)
            c.phase='independent';original=c.evaluate
            def wrapped(*a,_fn=original,**kw):
                y=_fn(*a,**kw);self.r['completed']['independent_point_directions']=self.r['attempted']['independent_point_directions'];return y
            c.evaluate=wrapped;self.references[size]=c
        for k in ('environment_point_directions','coefficient_point_directions'):self.r['completed'][k]=self.r['attempted'][k]

    def field(self,model):
        from benchmarks.neural_observation_e16 import tensor_quantities
        self.bump('field_calls');q=tensor_quantities(self.t,model,self.known);self.done('field_calls');return q

    def render(self,model,q,observations,seed,phase,per_observation=False):
        t=self.t;ids=t.as_tensor(self.points[observations],device='cuda');p=self.known['positions'][ids]
        n=self.known['normals'][ids];wo=self.wo[observations];outputs=[]
        from benchmarks.bell_reference_e11 import environment
        for light,definition in enumerate(previous.prior.LIGHTS):
            self.bump(phase+'_shader_calls')
            with t.random.fork_rng(devices=[0]):
                t.manual_seed(seed);t.cuda.manual_seed(seed)
                with model.environment(lambda pts,d,definition=definition:environment(d,definition),'e16-light-'+str(light)):
                    values={k:q[k] if per_observation else q[k][ids] for k in ('mx','my','rotation','raw')}
                    _,a=model.shade_anisotropic_mixed(p,n,wo,values['mx'],values['my'],self.known['metallic'][ids],
                        self.known['color'][ids],values['rotation'],values['raw'],None,True)
            value=t.stack([a[k] for k in ('linear_rgb','linear_diffuse','linear_specular')],-2)
            require(bool(t.isfinite(value).all()) and bool((value>=0).all()),'Nonfinite/negative native linear radiance')
            outputs.append(value);self.done(phase+'_shader_calls')
        return t.stack(outputs,1)

    def explicit(self,model,p,observations):
        """Teacher chart to native chart, with differentiable coordinate conversion."""
        t=self.t;ids=t.as_tensor(self.points[observations],device='cuda')
        T,B=model.compute_tangent_bitangent_flat(self.known['normals'][ids],None)
        axis=t.cos(p[:,:1])*self.known['T'][ids]+t.sin(p[:,:1])*self.known['B'][ids]
        theta=t.atan2((axis*B).sum(-1,keepdim=True),(axis*T).sum(-1,keepdim=True))
        e=t.cat((t.cos(2*theta),t.sin(2*theta)),-1);raw=(1+.5*e)/2
        # render() indexes the original physical point IDs. Expand only inputs,
        # not teacher outputs from a neural field, for this compatibility control.
        dtype=self.known['positions'].dtype
        return dict(mx=t.exp(p[:,1:2]+p[:,2:3]/2).to(dtype),my=t.exp(p[:,1:2]-p[:,2:3]/2).to(dtype),
            rotation=e.to(dtype),raw=raw.to(dtype))

    def compatibility(self):
        t=self.t;fit=self.fit;ids=self.points[fit];view=self.views[fit]
        pick=(view==0)&np.isin(ids,POLICY['gradient_probe_indices']);selected=fit[pick]
        require(len(selected)==6,'Fixed geometric derivative-control panel changed')
        reports={}
        for material in FIELDS:
            self.r['phase']='compatibility_'+material
            model=self.models[material,'A'];p=self.data['teachers'][material]['parameters'];values=[];gradients=[]
            # Existing E15 total capture plus independent known diffuse component.
            total=self.data['compat_reference'][material]['predicted_RGB'][fit]
            coarse=self.data['compat_reference'][material]['coarse_RGB'][fit]
            old=self.data['legacy'][5];arm=FIELDS.index(material)
            diffuse=np.stack([old[512][arm,l,self.views[fit],ids,0,0] for l in range(2)],1)
            diffuse_low=np.stack([old[256][arm,l,self.views[fit],ids,0,0] for l in range(2)],1)
            ref=np.stack((total,diffuse,total-diffuse),-2);low=np.stack((coarse,diffuse_low,coarse-diffuse_low),-2)
            for repeat in range(POLICY['compatibility_repeats']):
                # Differentiate the declared chart in float64, but pass only
                # native float32 widths/encoding into the unchanged shader.
                z=t.tensor(p[ids],dtype=t.float64,device='cuda',requires_grad=True)
                direct=self.explicit(model,z,fit)
                out=self.render(model,direct,fit,700000+repeat,'gate',per_observation=True)
                g=[]
                for channel in range(3):
                    # Sum both lights separately to retain their independent response columns.
                    parts=[]
                    for light in range(2):
                        self.bump('gate_reverse_sweeps')
                        parts.append(t.autograd.grad(out[pick,light,0,channel].sum(),z,retain_graph=True)[0][pick])
                        self.done('gate_reverse_sweeps')
                    g.append(t.stack(parts,1))
                gradients.append(t.stack(g,-2).detach().cpu().numpy());values.append(out.detach().cpu().numpy())
            refs={}
            for size in (512,256):
                _,j=self.references[size].evaluate(p[:,None],True,selected);refs[size]=j[:,0]
            forward=compatibility(np.array(values),ref,abs(ref-low),POLICY['compatibility_absolute'],POLICY['compatibility_relative'])
            gradient=compatibility(np.array(gradients),refs[512],abs(refs[512]-refs[256]),POLICY['gradient_absolute'],POLICY['gradient_relative'])
            from benchmarks.generate_bell_e11 import write_arrays
            write_arrays(self.out/(material+'-compatibility.npz'),native=np.array(values),reference=ref,coarse_reference=low,
                native_gradient=np.array(gradients),reference_gradient=refs[512],coarse_reference_gradient=refs[256],
                fitting_observations=fit,gradient_observations=selected)
            reports[material]=dict(forward=forward,gradient=gradient)
            self.r.setdefault('compatibility',{})[material]=reports[material]
            require(forward['passed'] and gradient['passed'],'Native/reference compatibility failed; stop before neural updates')
        self.r['compatibility']=reports

    def optimizer(self,model):
        p=POLICY
        return self.t.optim.Adam(model.parameters(),lr=p['lr'],betas=tuple(p['betas']),eps=p['eps'],
            amsgrad=p['amsgrad'],weight_decay=p['weight_decay'])

    def update(self,model,opt,loss,phase):
        t=self.t;require(bool(t.isfinite(loss)),'Nonfinite neural objective')
        self.bump(phase+'_backwards');loss.backward();self.done(phase+'_backwards')
        gradients={};finite_checks=[];inactive=('metallic_predictor','kd_predictor','outer_light','inner_light')
        for name,p in model.named_parameters():
            require(p.requires_grad,'Head-only/full-field restriction forbidden')
            if p.grad is not None:
                finite_checks.append(t.isfinite(p.grad).all())
                group=name.split('.')[0];gradients[group]=gradients.get(group,0.)+(p.grad*p.grad).sum()
            elif name.split('.')[0] not in inactive:raise ValueError('Unexpected missing active field gradient: '+name)
        require(not(set(gradients)&set(inactive)),'Known color/metal/lighting unexpectedly learned')
        require(finite_checks and bool(t.stack(finite_checks).all()),'Nonfinite active neural gradients')
        pre=float(t.nn.utils.clip_grad_norm_(model.parameters(),POLICY['gradient_clip']))
        require(np.isfinite(pre),'Nonfinite gradient norm')
        key='photometric_Adam_returns' if phase=='photo' else 'distillation_Adam_returns'
        self.bump(key);opt.step();t.cuda.synchronize();self.done(key)
        norms=t.sqrt(t.stack(list(gradients.values()))).detach().cpu().numpy()
        return dict(loss=float(loss.detach()),preclip_gradient_L2=pre,group_gradient_L2=dict(zip(gradients,norms.tolist())))

    def checkpoint(self,model,opt,name):
        from benchmarks.generate_bell_e11 import write_arrays
        arrays={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()};write_arrays(self.out/(name+'-weights.npz'),**arrays)
        if opt is not None:
            names={id(p):n for n,p in model.named_parameters()};states={}
            for p,state in opt.state.items():
                for k,v in state.items():states[names[id(p)]+'.'+k]=v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
            write_arrays(self.out/(name+'-Adam.npz'),**states)

    def snapshot(self,material,arm,stage,model,opt=None):
        self.r['phase']='snapshot_'+material+'_'+arm+'_'+stage
        from benchmarks.generate_bell_e11 import write_arrays
        from benchmarks.material_information_e14 import deterministic_codes
        with self.t.no_grad():
            q=self.field(model);draws=[]
            for repeat in range(POLICY['evaluation_repeats']):draws.append(self.render(model,q,self.all,900000+repeat,'eval').cpu().numpy())
        arrays={k:v.detach().cpu().numpy() for k,v in q.items()}
        a=arrays['axis'].astype(float);theta=np.arctan2((a*self.data['known']['B']).sum(1),(a*self.data['known']['T']).sum(1))
        mx=arrays['mx'][:,0].astype(float);my=arrays['my'][:,0].astype(float)
        p=np.stack((theta,np.log(np.sqrt(mx*my)),np.log(mx/my)),-1)[:,None]
        independent={}
        for size in (512,256):independent[size]=self.references[size].evaluate(p,False)[0][:,0]
        labels=self.data['legacy'][0];truth=previous.truth_parameters(labels,material=='stronger')[:,0]
        true_Q=world_Q(truth,labels['T'],labels['B']);Q=arrays['Q'].astype(float)
        target=previous.captured_target(self.data['legacy'][5][512],FIELDS.index(material),self.views,self.points)
        name=material+'-'+arm+'-'+stage;mean=np.asarray(draws,dtype=np.float64).mean(axis=0)[:,:,0]
        write_arrays(self.out/(name+'.npz'),**arrays,native_components=np.array(draws),independent_fine=independent[512],
            independent_coarse=independent[256],target_RGB=target,true_parameters=truth,true_Q=true_Q,
            teacher_Q=self.data['teachers'][material]['Q'],native_mean_codes=deterministic_codes(mean),target_codes=deterministic_codes(target),
            observation_points=self.points,observation_views=self.views,fitting_mask=np.isin(self.all,self.fit))
        self.checkpoint(model,opt,name)
        relative=np.linalg.norm(Q-true_Q,axis=(1,2))/np.linalg.norm(true_Q,axis=(1,2))
        held=self.data['legacy'][7][self.views,self.points]
        row=dict(material=material,arm=arm,stage=stage,Q_median=float(np.median(relative)),Q_pass=int((relative<=.05).sum()),
            held_RGB_relative_L2=float(np.linalg.norm((mean-target)[held])/np.linalg.norm(target[held])),
            weight_hash=state_hash(model),artifact=name+'.npz')
        self.r['snapshots'].append(row);return arrays,mean

    def run(self):
        from benchmarks.neural_observation_e16 import Q_loss
        from benchmarks.generate_bell_e11 import write_arrays
        write_arrays(self.out/'acquisition.npz',**self.data['known'],observation_points=self.points,
            observation_views=self.views,fitting_indices=self.fit,pixel_ids=self.data['legacy'][1])
        for material,c in self.data['confidence'].items():
            write_json(self.out/(material+'-confidence.json'),{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in c.items()})
        self.r['inherited_teacher_cost']=self.data.get('teacher_cost',{})
        self.r['inherited_teacher_context']=self.data.get('teacher_context',{})
        initials={}
        for material in FIELDS:
            for arm in ('A','B'):initials[material,arm]=self.snapshot(material,arm,'initial',self.models[material,arm])
        for material in FIELDS:
            for k in initials[material,'A'][0]:require(np.array_equal(initials[material,'A'][0][k],initials[material,'B'][0][k]),'Initial native field outputs differ')
            require(np.array_equal(initials[material,'A'][1],initials[material,'B'][1]),'Initial native renderer parity failed')
        self.compatibility()
        self.r['training_admitted']=True
        for material in FIELDS:
            B=self.models[material,'B'];A=self.models[material,'A'];distill=self.optimizer(B);rows=[];active_D=0.
            target=self.t.as_tensor(self.data['teachers'][material]['Q'],dtype=self.t.float32,device='cuda')
            for step in range(POLICY['distillation_updates']):
                self.r['phase']='distillation_'+material+'_'+str(step)
                start=time.monotonic();distill.zero_grad(set_to_none=True);q=self.field(B)
                value=self.update(B,distill,Q_loss(self.t,q['Q'],target),'distillation');active_D+=time.monotonic()-start
                rows.append(dict(phase='distillation',step=step,**value));self.progress(material,rows)
                if (step+1)%100==0:self.checkpoint(B,distill,material+'-B-distillation-progress'+str(step+1))
            self.snapshot(material,'B','post_distillation',B,distill)
            opt_B=self.optimizer(B);require(not opt_B.state,'Fresh matching photometric Adam state required')
            fit_target=self.data['legacy'][5][512]
            target_np=previous.captured_target(fit_target,FIELDS.index(material),self.views,self.points)[self.fit]
            fit_RGB=self.t.as_tensor(target_np,dtype=self.t.float32,device='cuda')
            active_B=self.photo(B,opt_B,fit_RGB,material,'B',POLICY['photometric_updates'],rows)
            self.snapshot(material,'B','photo1000',B,opt_B)
            opt_A=self.optimizer(A);rows_A=[];active_A=self.photo(A,opt_A,fit_RGB,material,'A',POLICY['photometric_updates'],rows_A)
            self.snapshot(material,'A','photo1000',A,opt_A);extra=0
            match=compute_match(active_A,active_B,active_D,self.data['teacher_wall_charge'],extra)
            while match['continue_baseline']:
                active_A+=self.photo(A,opt_A,fit_RGB,material,'A',1,rows_A,offset=POLICY['photometric_updates']+extra)
                extra+=1;match=compute_match(active_A,active_B,active_D,self.data['teacher_wall_charge'],extra)
            self.snapshot(material,'A','compute_endpoint',A,opt_A)
            self.r['compute_comparisons'][material]=dict(**match,extra_baseline_updates=extra,
                teacher_charge_seconds=self.data['teacher_wall_charge'],teacher_charge='Conservative entire E15 run per guided field; per-panel timing unavailable.',
                distillation_active_seconds=active_D,guided_photo_active_seconds=active_B)
        require(self.r['attempted']==self.r['completed'],'Incomplete native work/updates')
        for key,initial in self.inactive.items():
            current=dict(self.models[key].named_parameters())
            require(all(self.t.equal(current[n].detach().cpu(),p) for n,p in initial.items()),
                'Known color/metal/lighting parameters changed')
        self.r['inactive_parameters_verified_unchanged']=True
        for name,tensor in self.known.items():
            original='position' if name=='positions' else name
            require(np.array_equal(tensor.detach().cpu().numpy(),self.data['known'][original].astype(np.float32)),
                'Known geometry/color/metal/frame inputs changed')
        self.r['known_inputs_verified_unchanged']=True

    def progress(self,material,rows,arm='B'):
        p=self.out/(material+'-'+arm+'-training.json');temp=p.with_suffix('.next')
        temp.write_text(json.dumps(rows,allow_nan=False)+'\n');temp.replace(p)

    def photo(self,model,opt,target,material,arm,count,rows,offset=0):
        from benchmarks.neural_observation_e16 import photo_loss
        active=0.
        for step in range(offset,offset+count):
            self.r['phase']='photo_'+material+'_'+arm+'_'+str(step)
            start=time.monotonic();opt.zero_grad(set_to_none=True);q=self.field(model)
            y=self.render(model,q,self.fit,800000+step,'photo')[:,:,0]
            value=self.update(model,opt,photo_loss(self.t,y,target),'photo');active+=time.monotonic()-start
            rows.append(dict(phase='photo',step=step,**value));self.progress(material,rows,arm)
            if (step+1)%250==0:self.checkpoint(model,opt,material+'-'+arm+'-photo-progress'+str(step+1))
        return active


def execute(args,f,data):
    authorize(args);args.output.mkdir();start=time.monotonic()
    r=dict(schema_version=1,protocol=PROTOCOL,status='incomplete',cleanup_verified=False,scientific_work_complete=False,
        commit=args.commit,freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,files=f['files'],
        inputs=data['records'],policy=POLICY,limits=LIMITS,attempted={k:0 for k in LIMITS if k not in
        ('wall_seconds','inner_seconds','grace_seconds','cpu_threads','host_memory_gib','cuda_allocator_fraction','output_bytes')},
        snapshots=[],compute_comparisons={},training_admitted=False,phase='allocated',
        planned_histories=[dict(material=m,arm=a) for m in FIELDS for a in ('A','B')])
    r['completed']=dict(r['attempted'])
    def check():
        require(time.monotonic()-start<LIMITS['inner_seconds']-30 and
            sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],'E16 resource ceiling reached')
    try:
        import torch
        from infrastructure.material_recovery_e13_evaluation import native_tensor_default
        from benchmarks.generate_bell_e11 import run_with_cleanup
        require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and torch.cuda.device_count()==1 and
            'A100' in torch.cuda.get_device_name(0),'Reviewed native runtime required')
        torch.cuda.set_per_process_memory_fraction(.25);old_threads=torch.get_num_threads();torch.set_num_threads(4)
        r['runtime']=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),image_id=identity()['image_id'])
        old_np=np.random.get_state();old_py=random.getstate();old_cpu=torch.random.get_rng_state();old_gpu=torch.cuda.get_rng_state_all()
        old_type=torch.tensor([]).type()
        def work():
            with native_tensor_default(torch):Native(torch,data,args.output,r,check).run()
        try:run_with_cleanup(work,torch)
        finally:
            np.random.set_state(old_np);random.setstate(old_py);torch.random.set_rng_state(old_cpu);torch.cuda.set_rng_state_all(old_gpu)
            torch.set_num_threads(old_threads)
        current_np=np.random.get_state();current_gpu=torch.cuda.get_rng_state_all()
        require(current_np[0]==old_np[0] and np.array_equal(current_np[1],old_np[1]) and current_np[2:]==old_np[2:] and
            random.getstate()==old_py and torch.equal(torch.random.get_rng_state(),old_cpu) and
            len(current_gpu)==len(old_gpu) and all(torch.equal(a,b) for a,b in zip(current_gpu,old_gpu)) and
            torch.get_num_threads()==old_threads and torch.tensor([]).type()==old_type,'RNG/thread/default-type cleanup verification failed')
        r['cleanup_verified']=True
        import resource
        r.update(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        require(all(digest(ROOT/p)==h for p,h in f['files'].items()) and all(digest(Path(x['path']))==x['sha256'] for x in data['records']),
                'Terminal scientific source/input identity changed')
        check()
        r['outputs']=[record(p) for p in sorted(args.output.iterdir()) if p.is_file()]
        for x in r['outputs']:x['path']=str(Path(x['path']).relative_to(args.output))
        r.update(status='complete',scientific_work_complete=True,phase='finalized')
    except BaseException as exc:r.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
    finally:
        r['elapsed_seconds']=time.monotonic()-start;write_json(args.output/'neural-recovery.json',r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark','pilot','reference','factor','multiview','two-light','physical','freeze'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True);p.add_argument('--execute',action='store_true')
    for name in ('output','authorization'):p.add_argument('--'+name,type=Path)
    for name in ('commit','image-id','authorization-sha256'):p.add_argument('--'+name)
    args=p.parse_args();f,d=definition(args)
    if not args.execute:print(json.dumps(dict(status='metadata_only',new_scientific_queries=0,histories=4)));return
    require(all((args.output,args.authorization,args.commit,args.image_id,args.authorization_sha256)),'Separate scientific approval required')
    execute(args,f,d)


if __name__=='__main__':main()
