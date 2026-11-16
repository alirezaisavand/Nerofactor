"""One authorized E17 retention extension; accepted E16 implementation is immutable."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np

from experiments import bell_neural_material_e16 as prior
from experiments.analyze_bell_neural_material_e16 import numeric,state_hash
from benchmarks.neural_retention_e17 import POLICY,LIMITS,COEFFICIENT,scale_evidence,objective,strict_restore
from benchmarks.neural_observation_e16 import Q_loss
from infrastructure.stage_sources import ROOT,PINS,activate,imported_sources,digest
from infrastructure.signatures import require,read_report
from infrastructure.verified_ray_backend import identity
from experiments.bell_highlight_e11d import write_json

PROTOCOL='bell-neural-Q-retention-e17-v1'
E16_REPORT='f14aaa8dde21afbb1dd6626c49568e3083fa8d548979f8033f749bdd8af37f2b'
ROOT_NAMES=('benchmark','pilot','reference','factor','multiview','two_light','physical','neural')


def source_paths():
    return sorted(set(prior.source_paths())|{'benchmarks/neural_retention_e17.py','experiments/bell_neural_retention_e17.py',
        'experiments/run_bell_neural_retention_e17.sh','run_bell_neural_retention_e17.py',
        'tests/benchmarks/test_neural_retention_e17.py','docs/protocols/BELL_NEURAL_RETENTION_E17_V1.md',
        'docs/protocols/BELL_NEURAL_OBSERVATION_E16_V1.freeze.json','docs/BELL_NEURAL_OBSERVATION_E16_RESULTS_V1.md',
        'docs/evidence/BELL_NEURAL_OBSERVATION_E16_RESULTS_V1.json','experiments/analyze_bell_neural_material_e16.py'})


def load_inputs(args):
    data=prior.load_inputs(args);rp=args.neural/'neural-recovery.json'
    require(digest(rp)==E16_REPORT,'Accepted E16 report changed');r=json.loads(rp.read_text())
    require(r['status']=='complete' and all(r[k] is True for k in ('scientific_work_complete','cleanup_verified','training_admitted')) and
        r['commit']=='71687127e36527052aff733e3fcb96ebd0b67887' and r['policy']==prior.POLICY and
        all(x['forward']['passed'] and x['gradient']['passed'] for x in r['compatibility'].values()),'E16 admission unavailable')
    data['records'].append(prior.record(rp));restoration={};captures={};training={}
    for material in prior.FIELDS:
        paths={}
        for suffix in ('B-post_distillation-weights.npz','B-post_distillation.npz','B-photo1000.npz','A-photo1000.npz','B-training.json'):
            p=args.neural/(material+'-'+suffix);old=next(x for x in r['outputs'] if x['path']==p.name)
            require(p.parent==args.neural and p.resolve()==p and digest(p)==old['sha256'] and p.stat().st_size==old['bytes'],'E16 artifact changed: '+p.name)
            data['records'].append(prior.record(p));paths[suffix]=p
        with np.load(paths['B-post_distillation.npz'],allow_pickle=False) as z:
            initial={k:z[k].copy() for k in ('mx','my','rotation','raw','Q','axis','other','T','B')}
        old=next(x for x in r['snapshots'] if x['material']==material and x['arm']=='B' and x['stage']=='post_distillation')
        arrays=numeric(paths['B-post_distillation-weights.npz']);require(state_hash(arrays)==old['weight_hash'],'Checkpoint state identity differs')
        restoration[material]=dict(arrays=arrays,hash=old['weight_hash'],initial=initial)
        for stage in ('post_distillation','photo1000'):
            with np.load(paths['B-'+stage+'.npz'],allow_pickle=False) as z:
                captures[material,stage]={k:z[k].copy() for k in ('Q','teacher_Q')}
                require(np.array_equal(z['teacher_Q'],data['teachers'][material]['Q']),'E16/E15 teacher identity differs')
        training[material]=json.loads(paths['B-training.json'].read_text())
    data.update(restoration=restoration,coefficient_evidence=scale_evidence(captures,training),E16_effective_shader=r['effective_shader'])
    return data


def logical_inputs(records,args):
    result=[]
    for row in records:
        p=Path(row['path']);names=[k for k in ROOT_NAMES if getattr(args,k) in p.parents]
        require(len(names)==1,'Ambiguous input root');k=names[0]
        result.append(dict(root=k,path=p.relative_to(getattr(args,k)).as_posix(),sha256=row['sha256'],bytes=row['bytes']))
    return result


def definition(args):
    f,_=read_report(args.freeze,args.freeze_sha256)
    require(f['protocol']==PROTOCOL and f['policy']==POLICY and f['limits']==LIMITS and
        f['optimizer_policy']==prior.POLICY and f['E16_report_sha256']==E16_REPORT and set(f['files'])==set(source_paths()),'E17 definition mismatch')
    for p,h in f['files'].items():require(digest(ROOT/p)==h,'Frozen source changed: '+p)
    d=load_inputs(args);require(logical_inputs(d['records'],args)==f['inputs'] and d['coefficient_evidence']==f['coefficient_evidence'],'Frozen input/scale mismatch')
    return f,d


def authorize(args):
    a,_=read_report(args.authorization,args.authorization_sha256)
    require(a.get('approved') is True and a.get('protocol')==PROTOCOL and a.get('commit')==args.commit and
        a.get('freeze_sha256')==args.freeze_sha256 and a.get('limits')==LIMITS and
        a.get('image_id')==args.image_id==identity()['image_id'] and a.get('output_directory')==str(args.output),'Separate exact E17 approval required')
    require(os.geteuid()!=0 and args.output.is_absolute() and args.output.resolve()==args.output and args.output.parent==Path('/outputs') and
        not args.output.exists(),'Fresh non-root output required')
    for p in (Path('/'),ROOT,args.authorization,args.freeze,*[getattr(args,k) for k in ROOT_NAMES]):
        require(os.statvfs(p).f_flag&os.ST_RDONLY,'Read-only source/inputs/root required')
    require(not os.statvfs('/outputs').f_flag&os.ST_RDONLY,'Writable owned output required')
    require({s.split(':')[0].strip() for s in Path('/proc/net/dev').read_text().splitlines()[2:]}=={'lo'},'No network required')


class Retention(prior.Native):
    # Inherit the admitted field, native render/trace, Adam, gradient/clipping,
    # snapshot and primitive-checkpoint methods. No E16 module globals are changed.
    def __init__(self,torch,data,output,report,check):
        self.t=torch;self.data=data;self.out=output;self.r=report;self.check=check
        from infrastructure.verified_ray_backend import load
        load('predicate_fp64',identity()['image_id']);science,_=activate('material',PINS['material'])
        from network import field,renderer
        from scientific_variants.material_a_v1 import native_classes
        from infrastructure.bell_material_rays import original_adapter
        cls,_=native_classes(field.MCShadingNetwork,renderer.NeROMaterialRenderer,'A_corrected_core_v1',field.linear_to_srgb)
        self.owner=original_adapter(renderer.NeROMaterialRenderer,dict(mesh=str(data['mesh'])))
        self.known={k:torch.as_tensor(v,dtype=torch.float32,device='cuda') for k,v in data['known'].items()}
        self.known['positions']=self.known.pop('position');self.views,self.points=np.nonzero(data['legacy'][3]['valid'])
        self.wo=torch.as_tensor(data['legacy'][3]['outgoing_directions'][self.views,self.points],dtype=torch.float32,device='cuda')
        self.fit=np.flatnonzero(data['legacy'][6][self.views,self.points]);self.all=np.arange(len(self.points));self.models={};self.inactive={}
        self.r['restoration']={}
        for material in prior.FIELDS:
            self.r['phase']='restore_'+material;torch.manual_seed(prior.POLICY['seed']);torch.cuda.manual_seed_all(prior.POLICY['seed'])
            model=cls(prior.shader_config(),self.trace).cuda();old=data['restoration'][material]
            require(model.cfg==data['E16_effective_shader'],'Effective native shader settings changed')
            strict_restore(torch,model,old['arrays'],old['hash']);self.models[material,'C']=model
            material_parameters=[p for n,p in model.named_parameters() if n.split('.')[0] in
                ('feats_network','mx_predictor','my_predictor','rotation_predictor','metallic_predictor','kd_predictor')]
            require(sum(p.numel() for p in material_parameters)==1485584 and len(material_parameters)==84 and
                all(p.requires_grad for p in model.parameters()),'Full native material field required')
            self.inactive[material]={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.split('.')[0] in
                ('metallic_predictor','kd_predictor','inner_light','outer_light')}
            with torch.no_grad():q=self.field(model)
            require(set(q)==set(old['initial']) and all(np.array_equal(v.detach().cpu().numpy(),old['initial'][k]) for k,v in q.items()),
                'Restored native field differs from captured post-distillation outputs')
            self.r['restoration'][material]=dict(state_sha256=prior.state_hash(model),strict=True,field_outputs_bitwise_equal=True)
        self.r['restoration_admitted']=True;self.r['material_parameter_count']=1485584
        self.r['imported_pinned_sources']=imported_sources(science);self.r['effective_shader']=model.cfg
        start=time.monotonic();self.prepare_reference();self.r['reference_preparation_seconds']=time.monotonic()-start

    def bump(self,key,n=1):
        self.check();require(key in LIMITS and self.r['attempted'][key]+n<=LIMITS[key],'E17 fixed work cap: '+key)
        self.r['attempted'][key]+=n

    def component_probes(self,model,photo,retention):
        """Read component gradients without changing .grad, RNG or an optimizer."""
        t=self.t;names,params=zip(*model.named_parameters());before=t.random.get_rng_state();gpu=t.cuda.get_rng_state_all()
        gradients={};elapsed={};start=time.monotonic()
        for name,loss in (('photo',photo),('weighted_retention',COEFFICIENT*retention)):
            t.cuda.synchronize();begin=time.monotonic();self.bump('retention_reverse_sweeps')
            values=t.autograd.grad(loss,params,retain_graph=True,allow_unused=True);t.cuda.synchronize();self.done('retention_reverse_sweeps')
            elapsed[name]=time.monotonic()-begin;gradients[name]=values
        require(t.equal(t.random.get_rng_state(),before) and all(t.equal(a,b) for a,b in zip(gpu,t.cuda.get_rng_state_all())) and
            all(p.grad is None for p in params),'Diagnostic gradients altered RNG or .grad')
        groups={}
        for (name,p),a,b in zip(zip(names,params),gradients['photo'],gradients['weighted_retention']):
            group=name.split('.')[0]
            if a is None or b is None:
                require(a is None and b is None and group in ('metallic_predictor','kd_predictor','inner_light','outer_light'),'Missing active component gradient')
                continue
            require(bool(t.isfinite(a).all()&t.isfinite(b).all()),'Nonfinite component gradient')
            old=groups.setdefault(group,[0.,0.,0.]);old[0]+=(a*a).sum();old[1]+=(b*b).sum();old[2]+=(a*b).sum()
        result={k:dict(photo_L2=float(v[0].sqrt()),weighted_retention_L2=float(v[1].sqrt()),
            cosine=float(v[2]/t.sqrt(v[0]*v[1])) if float(v[0]*v[1])>0 else None) for k,v in groups.items()}
        return dict(groups=result,reverse_seconds=elapsed,wall_seconds=time.monotonic()-start)

    def run(self):
        require(self.r['restoration_admitted'],'All strict restoration gates must pass before updates')
        for material in prior.FIELDS:
            model=self.models[material,'C'];opt=self.optimizer(model);require(not opt.state,'Fresh photometric Adam required')
            teacher=self.t.as_tensor(self.data['teachers'][material]['Q'],dtype=self.t.float32,device='cuda')
            initial=self.t.as_tensor(self.data['restoration'][material]['initial']['Q'],dtype=self.t.float32,device='cuda')
            target=prior.previous.captured_target(self.data['legacy'][5][512],prior.FIELDS.index(material),self.views,self.points)[self.fit]
            target=self.t.as_tensor(target,dtype=self.t.float32,device='cuda');rows=[];active=0.;Q_seconds=0.;probe_seconds=0.
            for step in range(POLICY['photometric_updates']):
                self.r['phase']='photo_'+material+'_C_'+str(step);start=time.monotonic();opt.zero_grad(set_to_none=True)
                q=self.field(model);y=self.render(model,q,self.fit,800000+step,'photo')[:,:,0]
                self.t.cuda.synchronize();begin=time.monotonic()
                total,photo,retention=objective(self.t,y,target,q['Q'],teacher)
                drift=Q_loss(self.t,q['Q'].detach(),initial);self.t.cuda.synchronize();Q_seconds+=time.monotonic()-begin
                diagnostic=0.
                if step in POLICY['gradient_probe_steps']:
                    probe=self.component_probes(model,photo,retention);diagnostic=probe['wall_seconds'];probe_seconds+=diagnostic
                    self.r['gradient_probes'].append(dict(material=material,step=step,**probe))
                row=self.update(model,opt,total,'photo');active+=time.monotonic()-start-diagnostic
                rows.append(dict(phase='photo',step=step,photo_loss=float(photo.detach()),retention_loss=float(retention.detach()),
                    weighted_retention_loss=float(COEFFICIENT*retention.detach()),initial_Q_drift=float(drift),**row))
                self.progress(material,rows,'C')
                if (step+1)%250==0:self.checkpoint(model,opt,material+'-C-photo-progress'+str(step+1))
            require({int(s['step']) for s in opt.state.values()}=={POLICY['photometric_updates']},'Photometric Adam state differs')
            before=time.monotonic();self.snapshot(material,'C','photo1000',model,opt)
            self.r['timing'][material]=dict(active_training_seconds_excluding_probes=active,
                added_objective_and_drift_seconds=Q_seconds,component_gradient_probe_seconds=probe_seconds,
                endpoint_evaluation_and_serialization_seconds=time.monotonic()-before)
        require(self.r['attempted']==self.r['completed'] and self.r['completed']['photometric_Adam_returns']==2*POLICY['photometric_updates'],'Incomplete retention work')
        for material,initial in self.inactive.items():
            current=dict(self.models[material,'C'].named_parameters())
            require(all(self.t.equal(current[k].detach().cpu(),v) for k,v in initial.items()),'Known parameter group changed')
        self.r['inactive_parameters_verified_unchanged']=True
        require(all(np.array_equal(v.detach().cpu().numpy(),self.data['known']['position' if k=='positions' else k].astype(np.float32))
            for k,v in self.known.items()),'Known physical input changed')
        self.r['known_inputs_verified_unchanged']=True


def execute(args,f,data):
    authorize(args);args.output.mkdir();start=time.monotonic()
    r=dict(schema_version=1,protocol=PROTOCOL,status='incomplete',cleanup_verified=False,scientific_work_complete=False,
        commit=args.commit,freeze_sha256=args.freeze_sha256,authorization_sha256=args.authorization_sha256,files=f['files'],inputs=data['records'],
        policy=POLICY,optimizer_policy=prior.POLICY,limits=LIMITS,coefficient_evidence=data['coefficient_evidence'],
        restoration_admitted=False,phase='allocated',snapshots=[],gradient_probes=[],timing={})
    r['attempted']={k:0 for k in LIMITS if k not in ('wall_seconds','inner_seconds','grace_seconds','cpu_threads','host_memory_gib','cuda_allocator_fraction','output_bytes')}
    r['completed']=dict(r['attempted'])
    def check():
        require(time.monotonic()-start<LIMITS['inner_seconds']-30 and sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<LIMITS['output_bytes'],
            'E17 resource cap reached')
    try:
        import torch
        from infrastructure.material_recovery_e13_evaluation import native_tensor_default
        from benchmarks.generate_bell_e11 import run_with_cleanup
        require(torch.__version__=='1.8.1+cu111' and torch.version.cuda=='11.1' and torch.cuda.device_count()==1 and
            'A100' in torch.cuda.get_device_name(0),'Reviewed runtime required')
        torch.cuda.set_per_process_memory_fraction(.25);old_threads=torch.get_num_threads();torch.set_num_threads(4)
        r['runtime']=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0),cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),image_id=identity()['image_id'])
        old_np=np.random.get_state();old_py=random.getstate();old_cpu=torch.random.get_rng_state();old_gpu=torch.cuda.get_rng_state_all();old_type=torch.tensor([]).type()
        def work():
            with native_tensor_default(torch):Retention(torch,data,args.output,r,check).run()
        try:run_with_cleanup(work,torch)
        finally:
            np.random.set_state(old_np);random.setstate(old_py);torch.random.set_rng_state(old_cpu);torch.cuda.set_rng_state_all(old_gpu);torch.set_num_threads(old_threads)
        current=np.random.get_state();gpu=torch.cuda.get_rng_state_all()
        require(current[0]==old_np[0] and np.array_equal(current[1],old_np[1]) and current[2:]==old_np[2:] and random.getstate()==old_py and
            torch.equal(torch.random.get_rng_state(),old_cpu) and len(gpu)==len(old_gpu) and all(torch.equal(a,b) for a,b in zip(gpu,old_gpu)) and
            torch.get_num_threads()==old_threads and torch.tensor([]).type()==old_type,'Cleanup/RNG verification failed')
        r['cleanup_verified']=True
        import resource
        r.update(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        require(all(digest(ROOT/p)==h for p,h in f['files'].items()) and all(digest(Path(x['path']))==x['sha256'] for x in data['records']),'Terminal source/input change')
        require(len(r['snapshots'])==2 and all(r['completed'][k]==LIMITS[k] for k in
            ('photometric_Adam_returns','photo_backwards','photo_shader_calls','eval_shader_calls','retention_reverse_sweeps',
             'field_calls','independent_point_directions','coefficient_point_directions','environment_point_directions')),
            'Incomplete fixed retention/evaluation work')
        check();r['outputs']=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(args.output.iterdir()) if p.is_file()]
        r.update(status='complete',scientific_work_complete=True,phase='finalized')
    except BaseException as exc:r.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
    finally:r['elapsed_seconds']=time.monotonic()-start;write_json(args.output/'retention.json',r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in (*ROOT_NAMES,'freeze'):p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True);p.add_argument('--execute',action='store_true')
    for k in ('output','authorization'):p.add_argument('--'+k,type=Path)
    for k in ('commit','image-id','authorization-sha256'):p.add_argument('--'+k)
    args=p.parse_args();f,d=definition(args)
    if not args.execute:print(json.dumps(dict(status='metadata_only',new_scientific_queries=0,histories=2)));return
    require(all((args.output,args.authorization,args.commit,args.image_id,args.authorization_sha256)),'Separate execution approval required')
    execute(args,f,d)


if __name__=='__main__':main()
