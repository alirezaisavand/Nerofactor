"""Original forward/test_step/build/SDF-gradient AST with explicit CPU I/O/operator stand-ins."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]


def methods(path, class_name, names, scope):
    tree=ast.parse((ROOT/path).read_text())
    if class_name is not None:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),'exec'),scope)


@contextmanager
def fixture_renderer():
    events=[]
    class TorchFacade:
        def __getattr__(self,name):return getattr(torch,name)
        def set_default_tensor_type(self,value):events.append(('default_tensor_type',value))
    facade=TorchFacade()
    class DB:
        def __init__(self):self.calls=[]
        def get_img_ids(self):return [str(i) for i in range(128)]
        def get_image(self,source_id):
            self.calls.append(('image',source_id));return np.full((48,48,3),64 if source_id=='0' else 192,np.uint8)
        def get_pose(self,source_id):self.calls.append(('pose',source_id));return np.zeros((3,4),np.float32)
        def get_K(self,source_id):self.calls.append(('K',source_id));return np.eye(3,dtype=np.float32)
        def get_mask(self,source_id):self.calls.append(('mask',source_id));return np.ones((48,48),bool)
        def get_depth(self,source_id):
            self.calls.append(('depth',source_id));return np.full((48,48),int(source_id)+1,np.float32),np.ones((48,48),bool)
    def resize(array,size,interpolation):
        h,w=size[1],size[0];return array[::array.shape[0]//h,::array.shape[1]//w][:h,:w].copy()
    scope={'torch':facade,'np':np,'BaseDatabase':object,'cv2':SimpleNamespace(resize=resize,INTER_NEAREST=0),
           'color_map_forward':lambda a:a.astype(np.float32)/255}
    methods('network/renderer.py',None,{'build_imgs_info','imgs_info_to_torch','imgs_info_slice'},scope)
    def downsample(info,ratio):
        events.append(('downsample',ratio));d=dict(info);d['imgs']=info['imgs'][...,::4,::4];return d
    scope['imgs_info_downsample']=downsample
    methods('network/renderer.py','NeROShapeRenderer',{'forward','test_step'},scope)
    sdfscope={'torch':torch}
    methods('network/field.py','SDFNetwork',{'gradient'},sdfscope)
    class SDF:
        def sdf(self,x):return (x*x).sum(dim=-1,keepdim=True)
        gradient=sdfscope['gradient']
    class Renderer(torch.nn.Module):
        forward=scope['forward'];test_step=scope['test_step']
        def __init__(self):
            super().__init__();self.database=DB();self.is_nerf=False
            self.cfg={'database_name':'syn/bell','val_geometry':False,'test_downsample_ratio':True,'downsample_ratio':.25,
                      'test_ray_num':512,'use_refscores':False}
            self.color_network=SimpleNamespace(cfg={'human_light':False})
            self.child=torch.nn.Linear(1,1).cpu()
            self.child.register_forward_pre_hook(lambda *args:events.append('child_hook'))
            from infrastructure.bell_inference import proposed_split
            split=proposed_split();self.train_ids=np.asarray(split['train_ids']);self.test_ids=split['test_ids'];self.nvs_ids=split['nvs_ids']
            self.test_imgs_info=scope['imgs_info_to_torch'](scope['build_imgs_info'](self.database,['58'],False),'cpu')
            self.zero_calls=0;self.eval()
        def _construct_ray_batch(self,info,is_train):
            images=info['imgs'];h,w=images.shape[-2:];rgbs=images[0].permute(1,2,0).reshape(-1,3)
            return {'rgbs':rgbs},torch.zeros((1,3,4),device='cpu'),h*w,h,w
        def _process_ray_batch(self,batch,poses):
            n=len(batch['rgbs']);return (torch.zeros((n,3),device='cpu'),torch.zeros((n,3),device='cpu'),None,None,None)
        def render(self,rays_o,rays_d,near,far,human,perturb,anneal,*,is_train,step,is_nerf):
            events.append(('render',step,perturb,anneal,is_train))
            n=len(rays_o)
            x=torch.ones((n,3),device='cpu')
            gradient=SDF().gradient(x)
            events.append(('sdf_gradient',bool(torch.isfinite(gradient).all())))
            self.child(torch.ones((n,1),device='cpu'))
            rgb=torch.full((n,3),.3,device='cpu')
            keys=['ray_rgb','normal','diffuse_albedo','diffuse_light','diffuse_color','specular_albedo','specular_light',
                  'specular_color','specular_ref','occ_prob','indirect_light','occ_prob_gt','fg_rgb','bg_rgb','fg_acc']
            output={k:rgb.clone() for k in keys};output.update(gradient_error=torch.zeros(n,device='cpu'),depth=torch.ones((n,1),device='cpu'))
            return output
        def compute_rgb_loss(self,p,g,w):return ((p-g)**2).mean(-1)
        def zero_grad(self,*args,**kwargs):self.zero_calls+=1;return super().zero_grad(*args,**kwargs)
    model=Renderer()
    # Only this CPU fixture makes existing CUDA transfer calls no-ops. Production
    # code never patches Torch or scientific classes/operators.
    with patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
        yield SimpleNamespace(model=model,Renderer=Renderer,build=scope['build_imgs_info'],
                              to_torch=scope['imgs_info_to_torch'],events=events,torch=torch)
