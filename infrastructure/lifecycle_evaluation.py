"""Bounded original evaluators/test_step and independent uint8 metric checks."""
import sys
from pathlib import Path

from infrastructure.signatures import require
from infrastructure.lifecycle_state import state_record,canonical
from infrastructure.lifecycle_training import ForbidCalls
from infrastructure.bell_inference import rng_identity,scalar_metric

METRIC_BOUNDS={'psnr_absolute_db':1e-3,'ssim_absolute':1e-12}


def admit_view(stage,role,image_id,actual):
    from infrastructure.lifecycle_execution import splits
    expected=splits(stage)
    require(actual==expected,'Original renderer split differs from admitted stage split')
    ids=actual['test_ids'] if role=='validation' else actual['nvs_ids']
    require(ids and image_id==ids[0] and image_id not in actual['train_ids'],'Evaluation split leakage/selection mismatch')
    require((stage,role,image_id) in [('geometry','validation','0'),('material','validation','58'),('material','nvs','11')],
            'Unreviewed evaluation camera/role')


def metric_comparison(prediction,ground_truth,original,np,ssim):
    """Independent PSNR reduction; SSIM recomputed with the same declared library."""
    encode=lambda x:np.clip(x*255,0,255).astype(np.uint8)  # Original float scaling/truncation, not rounding/gamma.
    pred,truth=encode(prediction),encode(ground_truth)
    mse=float(np.mean((truth.astype(np.float64)-pred.astype(np.float64))**2))
    psnr=float(10*np.log10(255.**2/mse)) if mse else float('inf')
    similarity=float(ssim(truth,pred,win_size=11,channel_axis=2,data_range=255))
    measured=float(original['psnr']);measured_ssim=float(original['ssim'])
    delta=0. if psnr==measured else abs(psnr-measured)
    require(np.isfinite(similarity) and np.isfinite(measured_ssim),'Nonfinite original/independent SSIM')
    result={'original':{'psnr':scalar_metric(measured),'ssim':measured_ssim},
            'independent':{'psnr':scalar_metric(psnr),'ssim':similarity,'mse_uint8_cpu_float64':mse},
            'psnr_absolute_difference_db':delta if np.isfinite(delta) else None,
            'ssim_absolute_difference':abs(similarity-measured_ssim),'bounds':METRIC_BOUNDS,
            'passed':bool(delta<=METRIC_BOUNDS['psnr_absolute_db'] and abs(similarity-measured_ssim)<=METRIC_BOUNDS['ssim_absolute'])}
    return result,pred,truth


class EvaluationCapture(ForbidCalls):
    def __init__(self,trainer,stage,role,functions,progress):
        super().__init__(functions)
        self.network=trainer.network;self.stage=stage;self.role=role
        self.code=self.network.test_step.__func__.__code__;self.outputs=[];self.calls=[];self.mask=None
        self.loader=trainer.val_set_list[0];self.loader_code=type(self.loader).__iter__.__code__
        self.bar_code=progress.__init__.__code__;self.iterators=[];self.bars=[];self.cleanup_errors=[]
    def observe(self,frame,event,value):
        super().observe(frame,event,value)
        v=frame.f_locals;code=frame.f_code
        if code is self.code:
            require(v['self'] is self.network,'Foreign model evaluation')
            if event=='call':
                require(not self.calls and v['index']==0,'More than one view/incorrect iterator index')
                if self.stage=='material':require(v.get('is_nvs') is (self.role=='nvs'),'Incorrect A NVS/validation selection')
                self.calls.append({'index':0,'is_nvs':self.role=='nvs','step':v.get('step')})
            elif event=='return' and value is not None:
                self.outputs.append(value)
                if self.stage=='material':self.mask=v['ray_batch']['hit_mask'].detach().cpu().numpy().copy()
                self.camera={'K':v['test_imgs_info']['Ks'] if self.stage=='material' else v['imgs_info']['Ks'],
                             'pose':v['test_imgs_info']['poses'] if self.stage=='material' else v['imgs_info']['poses']}
        elif event=='return' and code is self.loader_code and v.get('self') is self.loader:self.iterators.append(value)
        elif event=='call' and code is self.bar_code:self.bars.append(v['self'])
    def __exit__(self,*args):
        super().__exit__(*args)
        for name,items,method in [('workers',self.iterators,'_shutdown_workers'),('progress',self.bars,'close')]:
            for obj in items:
                if callable(getattr(obj,method,None)):
                    try:getattr(obj,method)()
                    except Exception as exc:self.cleanup_errors.append({'resource':name,'error':str(exc)})
        if args[0] is None:require(not self.cleanup_errors,'Evaluation owned cleanup failed')


class SingleNVSView:
    """External dispatch only; same default tensor context as original A nvs()."""
    def __init__(self,network,torch):self.network=network;self.torch=torch
    def eval(self):self.network.eval()
    def __call__(self,data):
        require(data['index']==0 and data['eval'] is True,'Unexpected bounded NVS dispatch')
        self.torch.set_default_tensor_type('torch.cuda.FloatTensor')
        try:return self.network.test_step(0,is_nvs=True)
        finally:self.torch.set_default_tensor_type('torch.FloatTensor')


def evaluate(trainer,p,run,report,persist,torch,np):
    from train.trainer import Trainer
    from train.train_valid import tqdm
    from utils.base_utils import color_map_backward
    from skimage.metrics import structural_similarity
    from PIL import Image
    network=trainer.network
    if p['stage']=='geometry':
        require(network.cfg['val_geometry'] is False and network.cfg['test_downsample_ratio'] is True and
                network.cfg['downsample_ratio']==.25 and network.cfg['test_ray_num']==512,
                'Original G validation settings changed; no extraction or resizing substitution')
    else:
        require(network.cfg['test_ray_num']==1024 and network.is_nerf is False,'Original A full-resolution camera/ray settings changed')
    actual={'train_ids':[str(i) for i in network.train_ids],'test_ids':[str(i) for i in network.test_ids],
            'nvs_ids':[str(i) for i in network.nvs_ids] if p['stage']=='material' else None}
    require(len(trainer.val_set_list)==1 and len(trainer.val_set_list[0].dataset)==1,'One original held-out validation loader required')
    roles=[('validation','0')] if p['stage']=='geometry' else [('validation','58'),('nvs','11')]
    before=state_record(trainer,np);cursor=network.train_batch_i
    report.update(evaluation=[],evaluation_step=1,checkpoint_stored_step=2,optimizer_state_restored=False,
                  evaluation_before=before,rng_before_evaluation=rng_identity(torch,np))
    for role,image_id in roles:
        admit_view(p['stage'],role,image_id,actual)
        report['phase']='original_'+p['stage']+'_'+role+'_'+image_id;persist()
        forbidden=[Trainer.run,Trainer._save_model,Trainer._load_model,torch.optim.Adam.step,torch.Tensor.backward,network.train_step,network.nvs]
        observation=EvaluationCapture(trainer,p['stage'],role,forbidden,tqdm)
        try:
            with observation:
                model=network if role=='validation' else SingleNVSView(network,torch)
                values,key=trainer.val_evaluator(model,trainer.val_losses+trainer.val_metrics,trainer.val_set_list[0],1,
                                                 trainer.model_name,val_set_name='lifecycle-'+role)
        finally:
            report['evaluation_cleanup']={'errors':observation.cleanup_errors,'profile_restored':sys.getprofile() is None};persist()
        require(len(observation.outputs)==len(observation.calls)==1,'Missing genuine original test_step return')
        outputs=observation.outputs[0];size=200 if p['stage']=='geometry' else 800
        pred_key,gt_key=('ray_rgb','gt_rgb') if p['stage']=='geometry' else ('rgb_pr','rgb_gt')
        arrays={name:v.detach().cpu().numpy().copy() for name,v in outputs.items() if torch.is_tensor(v)}
        require(arrays[pred_key].shape==arrays[gt_key].shape==(size,size,3) and
                all(np.isfinite(v).all() for v in arrays.values()),'Wrong dimensions/nonfinite original RGB/shading output')
        if p['stage']=='material':
            for name,width in [('mx',1),('my',1),('metallic',1),('kd',3)]:
                require(arrays[name].shape==(800,800,width),'Incorrect material channel dimensions')
            mask=observation.mask.reshape(800,800)
            require((arrays[pred_key][~mask]==0).all() and (arrays[gt_key][~mask]==0).all(),'Original A missed-ray black fill changed')
            arrays['ray_hit_mask']=mask;arrays['source_object_mask']=network.database.get_mask(image_id)
        require(float(key)==float(np.asarray(values['psnr']).reshape(-1)[0]),'Original key metric aggregation differs')
        comparison,rgb,truth=metric_comparison(arrays[pred_key],arrays[gt_key],
            {'psnr':key,'ssim':np.asarray(values['ssim']).reshape(-1)[0]},np,structural_similarity)
        require(np.array_equal(rgb,color_map_backward(arrays[pred_key])) and
                np.array_equal(truth,color_map_backward(arrays[gt_key])),'Independent RGB conversion disagrees with original')
        directory='evaluations/'+role+'-'+image_id;run.output(directory).mkdir()
        for name,v in arrays.items():
            with run.output(directory+'/'+name+'.npy').open('xb') as f:np.save(f,v,allow_pickle=False)
        for name,v in observation.camera.items():
            with run.output(directory+'/'+name+'.npy').open('xb') as f:np.save(f,v.detach().cpu().numpy(),allow_pickle=False)
        for name,v in [('prediction',rgb),('ground-truth',truth)]:Image.fromarray(v).save(str(run.output(directory+'/'+name+'.png')))
        entry={'role':role,'source_camera_id':image_id,'training_member':False,'resolution':[size,size],
               'calls':observation.calls,'metrics':comparison,
               'original_aggregates':{k:scalar_metric(float(np.asarray(v).mean())) for k,v in values.items()},
               'channels':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'minimum':float(v.min()),'maximum':float(v.max())} for k,v in arrays.items()},
               'rotation':'not exposed by original A test_step' if p['stage']=='material' else None,
               'alpha':'allocated but unfilled by original A test_step' if p['stage']=='material' else None,
               'normal_encoding':'Original visualization encoding (unit normal + 1)/2; G additionally masks inner samples; not raw BVH normals',
               'metric_domain':'full uint8 image; A missed rays black in prediction AND returned ground truth',
               'outputs':directory}
        report['evaluation'].append(entry);persist()
        require(comparison['passed'],'Independent metrics outside frozen comparison bounds')
        require(state_record(trainer,np)==before and network.train_batch_i==cursor,'Evaluation modified weights/optimizer/ray cursor')
    report.update(evaluation_after=state_record(trainer,np),rng_after_evaluation=rng_identity(torch,np),
                  model_optimizer_unchanged=True,optimizer_calls_completed=0,
                  evaluation_implementation={'metric_PSNR':'original float32 per-channel reduction on clipped/truncated uint8 RGB',
                    'independent_PSNR':'CPU float64 mean over same uint8 RGB',
                    'SSIM':'original and independently re-invoked same skimage implementation; win_size11/channel_axis2/data_range255',
                    'masking':'No foreground mask applied by original PSNR/SSIM',
                    'RGB_scaling':'Renderer RGB preserved; A shader already applies original linear_to_srgb; dataset RGB/255; no added gamma, clip*255 and truncate only for metrics/PNG',
                    'quality_claim':'functional evidence only; diagnostic checkpoint is not a converged model'})
