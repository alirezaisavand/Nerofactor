"""3B-2 CPU-only B0/B1/B2/B3 algebra, isolated from frozen scientific files."""
import ast
from copy import deepcopy
import hashlib
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from infrastructure.signatures import require
from infrastructure.stage_sources import baseline,PINS,digest

PROTOCOL='anisotropic-estimator-ablation-v1'
SOURCE_SHA='b8386007dc009442cdefdeccc3c6ecc0114f6d0e7feaf804008b0bd9020091fe'
VARIANTS=('B0','B1','B2','B3')
METHODS={'get_orthogonal_directions','compute_tangent_bitangent_flat','sample_diffuse_directions',
         'compute_Dh','compute_pdf_aniso_ggx','compute_spec_loss','rotate_tangent_bitangent',
         'sample_aniso_ggx_directions','compute_radiance','nan_inf_check','fresnel_schlick',
         'fresnel_schlick_directions','diffuse_term'}


def tensor(value):return torch.tensor(np.asarray(value),dtype=torch.float32,device='cpu')


def cosine_diffuse(self,normals,is_train):
    """Only corrected radial construction; original basis, azimuth and RNG order."""
    z_axis=normals
    x=self.get_orthogonal_directions(normals)
    y=torch.cross(z_axis,x,dim=-1)
    az=self.diffuse_direction_samples[:,0:1].unsqueeze(0)*np.pi*2
    if is_train and self.cfg['random_azimuth']:
        az=(az+torch.rand(z_axis.shape[0],1,1)*np.pi*2)%(2*np.pi)
    z=self.underlying_heights.reshape(1,-1,1)
    # Exact approved formulas, no added epsilon/jitter/normal repair.
    radial=torch.sqrt(1-z);height=torch.sqrt(z)
    return radial*torch.cos(az)*x.unsqueeze(1)+radial*torch.sin(az)*y.unsqueeze(1)+height*z_axis.unsqueeze(1)


def source_classes():
    root,source=baseline('material',PINS['material']);path=root/'network/field.py'
    require(digest(path)==SOURCE_SHA,'Frozen A algebra identity differs')
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MCShadingNetwork')
    functions=[deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in METHODS]
    require({n.name for n in functions}==METHODS,'Incomplete original CPU method set')
    for node in functions:
        require(not node.decorator_list and not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(node)),'Unexpected source imports/decorators')
        require(not any(isinstance(n,ast.Attribute) and n.attr=='cuda' for n in ast.walk(node)),'Native CUDA call in pure source method')
    sphere_tree=ast.parse((root/'utils/base_utils.py').read_text())
    sphere=deepcopy(next(n for n in sphere_tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample_sphere'))
    def compile_class(nodes,name):
        module=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name=name,bases=[],keywords=[],body=nodes,decorator_list=[]),deepcopy(sphere)],type_ignores=[]))
        namespace={'torch':torch,'F':F,'np':np};exec(compile(module,str(path),'exec'),namespace)
        return namespace[name],namespace['sample_sphere']
    original,sample_sphere=compile_class(deepcopy(functions),'B0FrozenAlgebra')
    changed=deepcopy(functions);radiance=next(n for n in changed if n.name=='compute_radiance')
    candidates=[n for n in ast.walk(radiance) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='specular' for t in n.targets)]
    expected=ast.parse('specular = spec_weighted.sum(dim=1) / valid_counts').body[0]
    require(len(candidates)==1 and ast.dump(candidates[0])==ast.dump(expected),'Original specular normalization boundary changed')
    candidates[0].value.right=ast.Name(id='M_spec',ctx=ast.Load())
    corrected,_=compile_class(changed,'B1TotalCountAlgebra')
    classes={'B0':original,'B1':corrected,
             'B2':type('B2CosineAlgebra',(original,),{'sample_diffuse_directions':cosine_diffuse}),
             'B3':type('B3CombinedAlgebra',(corrected,),{'sample_diffuse_directions':cosine_diffuse})}
    fingerprints={n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in functions}
    return classes,sample_sphere,{'commit':source['commit'],'source_sha256':SOURCE_SHA,'method_AST_sha256':fingerprints,
        'approved_AST_change':'compute_radiance: specular denominator valid_counts -> M_spec; all other statements unchanged'}


def shaders(diffuse_count=512):
    """Recover exact original heights from the pinned construction, not inverse-rounded storage."""
    require(diffuse_count>0,'Positive sample count required')
    classes,sphere,identity=source_classes();az,el=sphere(diffuse_count,0)
    pattern=np.stack([az*.5/np.pi,1-2*el/np.pi],-1).astype(np.float32)
    # sample_sphere(N,0) uses indices N..2N-1 and num_points=2N.
    heights=2*np.arange(diffuse_count,2*diffuse_count,dtype=np.float64)/(2*diffuse_count)-1
    require(np.max(abs(np.sin(el)-heights))<1e-12,'Original sample-height reconstruction mismatch')
    require(np.array_equal(heights.astype(np.float32),np.arange(diffuse_count,dtype=np.float32)/diffuse_count),'Height stratification mismatch')
    rounded_inverse=np.sin((1-pattern[:,1].astype(np.float64))*np.pi/2)
    identity['height_recovery']={'strategy':'original sample_sphere integer indices, z=j/N',
        'maximum_unrounded_sine_error':float(np.max(abs(np.sin(el)-heights))),
        'maximum_inverse_rounded_storage_error':float(np.max(abs(rounded_inverse-heights))),
        'height_sha256':hashlib.sha256(heights.astype(np.float32).tobytes()).hexdigest(),
        'original_pattern_sha256':hashlib.sha256(pattern.tobytes()).hexdigest()}
    objects={}
    for name,cls in classes.items():
        obj=cls();obj.cfg={'random_azimuth':True};obj.diffuse_direction_samples=tensor(pattern)
        obj.underlying_heights=tensor(heights);objects[name]=obj
    return objects,identity


def sample_stream(seed,specular_count=1024):
    rng=np.random.RandomState(seed)
    values=[rng.uniform(size=(1,specular_count)).astype(np.float32),rng.uniform(size=(1,specular_count)).astype(np.float32),rng.uniform(size=(1,1,1)).astype(np.float32)]
    return values


class RandomTape:
    def __init__(self,values):self.values=values;self.calls=0;self.floats=0
    def __call__(self,*shape,**kwargs):
        require(self.calls<len(self.values),'Unapproved extra random draw')
        shape=tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(tuple,list)) else tuple(shape)
        require(shape==self.values[self.calls].shape,'Random sample shape/order differs')
        require(kwargs.get('device') in (None,'cpu',torch.device('cpu')),'CPU-only draw required')
        value=tensor(self.values[self.calls]);self.calls+=1;self.floats+=value.numel();return value


def draw(shader,mx,my,wo,stream,theta=.31,rotation=None):
    rotation=tensor([[np.cos(2*theta),np.sin(2*theta)]]) if rotation is None else tensor([rotation])
    tape=RandomTape(stream)
    with patch.object(torch,'rand',tape):
        spec=shader.sample_aniso_ggx_directions(rotation,tensor([[0,0,0]]),tensor([[mx]]),tensor([[my]]),tensor([wo]),
                stream[0].shape[1],tensor([[0,0,1]]),None,device='cpu')
        diffuse=shader.sample_diffuse_directions(tensor([[0,0,1]]),True)
    require(tape.calls==3,'Original two specular + one azimuth draw order not preserved')
    require(all(x.device.type=='cpu' and bool(torch.isfinite(x).all()) for x in spec) and bool(torch.isfinite(diffuse).all()),'Nonfinite/non-CPU samples')
    return spec,diffuse,{'calls':tape.calls,'floats':tape.floats,'valid':int((spec[1][:,:,2]>0).sum()),'rejected':int((spec[1][:,:,2]<=0).sum())}


def lighting(directions):
    """Fixed nonnegative analytical lighting; input directions are never repaired."""
    x=np.asarray(directions,dtype=np.float64);c=np.maximum(x[...,2],0.)
    axis=np.ones(3)/np.sqrt(3);lobe=.1+.9*np.maximum(x@axis,0.)**8
    return np.stack([np.ones_like(c),c*c,lobe],-1)


def estimate(shader,spec,diffuse,wo,mx,my,metallic=.001,kd=.999):
    h,wi,ch,pdf,t,b,n,theta,phi=spec
    metal=tensor([[metallic]]);albedo=tensor([[kd]*3]);d=shader.diffuse_term(albedo,metal)
    F0=.04*(1-metal)+metal*albedo
    fresnel,_,_=shader.fresnel_schlick_directions(F0.unsqueeze(1),tensor([[wo]]),wi)
    ld=lighting(diffuse.numpy()[0]);ls=lighting(wi.numpy()[0]);estimates=[]
    for light in range(3):
        diffuse_light=tensor(np.repeat(ld[None,:,light,None],3,axis=-1))
        specular_light=tensor(np.repeat(ls[None,:,light,None],3,axis=-1))
        values=shader.compute_radiance(d,diffuse_light,specular_light,fresnel,diffuse,wi,tensor([[0,0,1]]),ch,tensor([wo]),pdf,
                theta,phi,tensor([[mx]]),tensor([[my]]))
        require(all(bool(torch.isfinite(v).all()) for v in values),'Nonfinite radiance or retained legacy diagnostic')
        for value in values[:3]:require(bool((value[0]==value[0,0]).all()),'Expected achromatic RGB fixture')
        estimates.append([float(values[1][0,0]),float(values[2][0,0]),float(values[0][0,0])])
    return np.asarray(estimates,np.float64)
