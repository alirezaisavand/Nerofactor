"""E9 trainable-set, numeric initialization and passive frozen-graph observations."""
import numpy as np
from infrastructure import neural_material_observer_e2a as obs
from infrastructure.signatures import require
from experiments.neural_projection_e7_v1 import HEADS


def final_names():
    return {head+'.predictor.6.'+key for head,_ in HEADS for key in ('bias','weight_g','weight_v')}


def freeze(model):
    """Preserve module/parameter values and original registration order."""
    allowed=final_names();parameters=dict(model.named_parameters())
    require(allowed <= set(parameters), 'Original final weight-normalized heads required')
    for head,width in HEADS:
        seq=getattr(model,head).predictor
        require(len(seq)==8 and tuple(seq[-2].bias.shape)==(width,) and hasattr(seq[-2],'weight_g') and hasattr(seq[-2],'weight_v'), 'Final head architecture mismatch')
    for n,p in parameters.items():p.requires_grad_(n in allowed);p.grad=None
    for _,b in model.named_buffers():b.requires_grad_(False)
    return {n:p for n,p in parameters.items() if n in allowed}


class TrainableView:
    """Use original backward/Adam functions with their correct parameter scope."""
    arm='N1'
    def __init__(self,model):self.selected={n:p for n,p in model.named_parameters() if p.requires_grad}
    def named_parameters(self):return self.selected.items()
    def parameters(self):return self.selected.values()


def verify_frozen(model,initial,optimizer):
    allowed=final_names();parameters=dict(model.named_parameters())
    require({n for n,p in parameters.items() if p.requires_grad}==allowed, 'Trainable set changed')
    require(all(p.grad is None and p.detach().numpy().tobytes()==initial['parameter/'+n].tobytes() and p not in optimizer.state
        for n,p in parameters.items() if n not in allowed), 'Frozen parameter changed or received gradient/Adam state')
    require(all(not b.requires_grad and b.detach().numpy().tobytes()==initial['buffer/'+n].tobytes() for n,b in model.named_buffers()), 'Calibration buffer changed')
    require(len(optimizer.param_groups)==1 and {id(p) for p in optimizer.param_groups[0]['params']}=={id(parameters[n]) for n in allowed}, 'Optimizer includes foreign parameters')


def restore_initial(model,optimizer,arrays,rng,expected_identity):
    """Exact saved initial weights/buffers/RNG; fresh head-only Adam, no resume."""
    import torch
    from experiments import neural_material_continuation_e3 as runtime
    params=dict(model.named_parameters());buffers=dict(model.named_buffers())
    wanted={'parameter/'+n for n in params}|{'buffer/'+n for n in buffers}|{'torch_CPU_RNG','numpy_MT19937'}
    require(set(arrays)==wanted and not optimizer.state, 'Saved initialization/fresh optimizer inventory mismatch')
    for prefix,items in (('parameter/',params),('buffer/',buffers)):
        for n,t in items.items():
            a=arrays[prefix+n]
            require(t.device.type=='cpu' and t.dtype==torch.float32 and a.dtype==np.float32 and a.shape==tuple(t.shape) and np.isfinite(a).all(), 'Saved initialization shape/dtype/value mismatch')
    require(obs.parameter_identity({n:arrays['parameter/'+n] for n in params})==expected_identity, 'Initial parameter identity mismatch')
    runtime.install_rng(arrays,rng)
    with torch.no_grad():
        for prefix,items in (('parameter/',params),('buffer/',buffers)):
            for n,t in items.items():t.copy_(torch.from_numpy(arrays[prefix+n].copy()))
    actual,metadata=runtime.prior.state_arrays(model,optimizer)
    require(metadata==rng and set(actual)==wanted and all(actual[n].dtype==a.dtype and actual[n].shape==a.shape and actual[n].tobytes()==a.tobytes() for n,a in arrays.items()), 'Exact initial state restoration failed')
    verify_frozen(model,arrays,optimizer)
    return dict(exact=True,parameter_identity=expected_identity,arrays=len(wanted),fresh_Adam=True,Adam_step=0,RNG_restored=True)


class HeadObserver(obs.GraphObserver):
    """Observe connected heads/shader; frozen activations correctly lack gradients."""
    def __enter__(self):
        import torch
        try:
            for head,_ in HEADS:
                seq=getattr(self.model,head).predictor;last=seq[-1]
                require(isinstance(last,torch.nn.Sigmoid), 'Original sigmoid head required')
                self.handles.append(last.register_forward_pre_hook(lambda m,args,n=head:self._keep(n+'.logits',args[0])))
                self.handles.append(last.register_forward_hook(lambda m,args,out,n=head:self._keep(n+'.sigmoid',out)))
                def frozen_input(module,args):require(not args[0].requires_grad, 'Unexpected differentiable frozen hidden features')
                self.handles.append(seq[-2].register_forward_pre_hook(frozen_input))
            def frozen_features(module,args,out):require(not out.requires_grad, 'Frozen trunk unexpectedly connected')
            self.handles.append(self.model.feats_network.register_forward_hook(frozen_features))
            return self
        except BaseException:self.close();raise
    def collect(self):
        expected={h+'.'+k for h,_ in HEADS for k in ('logits','sigmoid')}|{'shader.'+n for n in ('mx','my','albedo','metallic','encoding')}
        require(set(self.nodes)==expected, 'Missing connected head/shader nodes')
        arrays={};summary={}
        for name,t in self.nodes.items():
            require(t.grad is not None, 'Missing connected head/shader gradient')
            a=obs.tensor_array(t);g=obs.tensor_array(t.grad)
            require(np.isfinite(a).all() and np.isfinite(g).all(), 'Nonfinite head/shader observation')
            arrays[name]=a;arrays[name+'.loss_gradient']=g
            summary[name]=dict(value=obs.array_record(a),loss_gradient=obs.array_record(g),phase='pre_update')
            if name.endswith('.sigmoid'):
                arrays[name+'.analytic_derivative']=a.astype(float)*(1-a.astype(float))
                summary[name]['analytic_derivative']=obs.array_record(arrays[name+'.analytic_derivative'])
        return arrays,dict(enabled=True,nodes=summary,frozen_activation_gradients='expected_absent',parameter_gradient_scope='15 final head tensors only')
