"""Phase B field-only CPU fixtures. No shader rendering or recovery optimizer."""
import ast
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import neural_material_field_v1 as f
from experiments import neural_material_recovery_v1 as h
from infrastructure.signatures import SignatureError


def latent(seed=101):return h.fresh_latent(seed)
def positions(n=1):return torch.arange(n*3,dtype=torch.float32).reshape(n,3)/7


class Tests(unittest.TestCase):
    def test_source_admission_closed_definitions_and_no_native_modules(self):
        ns,identity=f.source_definitions()
        self.assertEqual(identity['scientific_commit'],'e1e859779301faced46d6476691e20b546c27ae0')
        self.assertEqual(identity['source_sha256'],f.SOURCE_SHA)
        self.assertEqual(set(identity['definition_AST_sha256']),set(f.DEFINITIONS)|{'predict_anisotropic_components'})
        self.assertNotIn('MCShadingNetwork',ns)
        self.assertNotIn('nvdiffrast',ns)
        with patch.object(f,'SOURCE_SHA','0'*64):
            with self.assertRaises(SignatureError):f.source_definitions()

    def test_original_topology_weight_norm_and_live_parameter_count(self):
        n=f.NeuralMaterial(latent(),101)
        self.assertEqual(sum(p.numel() for p in n.parameters()),1485584)
        self.assertEqual(len(list(n.named_parameters())),84)
        self.assertEqual(len(list(n.feats_network.module0)),8)
        self.assertEqual(len(list(n.feats_network.module1)),7)
        self.assertEqual(n.feats_network.module0[0].weight_v.shape,(256,51))
        self.assertEqual(n.feats_network.module1[0].weight_v.shape,(256,307))
        linears=[m for m in n.modules() if isinstance(m,nn.Linear)]
        self.assertEqual(len(linears),28)
        for m in linears:
            self.assertTrue(hasattr(m,'weight_v') and hasattr(m,'weight_g'))
            self.assertGreater(float(m.weight_v.norm()),0)
        for name,size in f.HEADS:
            head=getattr(n,name).predictor
            self.assertEqual(head[0].weight_v.shape,(256,259))
            self.assertEqual(head[-2].weight_v.shape,(size,256))
            self.assertIsInstance(head[-1],nn.Sigmoid)
        self.assertFalse(any(isinstance(m,(nn.Dropout,nn.BatchNorm1d)) for m in n.modules()))

    def test_encoding_and_skip_are_the_original_definitions(self):
        ns,_=f.source_definitions();encode,width=ns['get_embedder'](8,3)
        self.assertEqual(width,51)
        x=torch.tensor([[.1,.2,.3]])
        expected=[x]
        for freq in 2.**torch.linspace(0.,7.,8):expected.extend((torch.sin(x*freq),torch.cos(x*freq)))
        self.assertTrue(torch.equal(encode(x),torch.cat(expected,-1)))
        net=ns['MaterialFeatsNetwork']();encoded=net.pos_enc(x)
        self.assertTrue(torch.equal(net(x),net.module1(torch.cat((net.module0(encoded),encoded),-1))))

    def test_head_order_and_exact_six_scalars_for_all_approved_seeds(self):
        self.assertEqual(f.HEADS,(('mx_predictor',1),('my_predictor',1),('metallic_predictor',1),('rotation_predictor',2),('kd_predictor',3)))
        for seed in f.SEEDS:
            z=latent(seed);a=f.DirectMaterial(z)(positions());b=f.NeuralMaterial(z,seed)(positions())
            indices=[0,1,3,4,5,6]
            self.assertTrue(torch.equal(a.physical[:,indices],b.physical[:,indices]))
            q0,q1=(h.previous.coupling.accepted.tensor_q(p.physical[0].detach().numpy()) for p in (a,b))
            self.assertTrue(np.all(abs(q1-q0)<=2e-7+2e-6*abs(q0)))

    def test_constant_field_ignores_both_spatial_inputs_across_batches(self):
        n=f.NeuralMaterial(latent(),101)
        original=n(positions(1))
        for count in (1,9,96):
            value=n(positions(count))
            self.assertTrue(torch.equal(value.physical,original.physical.expand(count,7)))
            self.assertTrue(torch.equal(value.encoding,original.encoding.expand(count,2)))
        seen=[]
        hook=n.feats_network.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        n(positions(9));hook.remove()
        self.assertTrue(torch.equal(seen[0],torch.zeros((1,3))))
        incoming=[]
        hook=n.mx_predictor.register_forward_pre_hook(lambda module,args:incoming.append(args[0].detach().clone()))
        n(positions(9));hook.remove()
        self.assertTrue(torch.equal(incoming[0][:,-3:],torch.zeros((1,3))))

    def test_rng_restoration_and_deterministic_parameter_buffer_identities(self):
        rng=torch.get_rng_state().clone();nrng=np.random.get_state()
        a=f.NeuralMaterial(latent(),101);b=f.NeuralMaterial(latent(),101)
        self.assertEqual(f.identity(a),f.identity(b))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        after=np.random.get_state();self.assertTrue(np.array_equal(nrng[1],after[1]));self.assertEqual(nrng[2:],after[2:])
        self.assertEqual(f.identity(a)['configuration']['weight_seed'],10101)
        self.assertIsNotNone(f.identity(a)['source'])

    def test_calibration_buffers_do_not_depend_on_truth_and_are_not_parameters(self):
        n=f.NeuralMaterial(latent(),101)
        self.assertFalse(any('center' in key or 'target' in key for key,_ in n.named_parameters()))
        self.assertFalse(any(b.requires_grad for _,b in n.named_buffers()))
        features=torch.cat((n.feats_network(n.anchor),n.anchor),-1)
        for name,_ in f.HEADS:
            head=getattr(n,name)
            raw=head.pre_sigmoid(features)
            self.assertTrue(torch.equal(raw,head.center))
            calibrated=head.target+(raw-head.center)
            self.assertTrue(torch.equal(calibrated,head.target))
            self.assertTrue(torch.equal(head(features),torch.sigmoid(head.target)))
        self.assertNotIn('truth',f.identity(n)['configuration'])

    def test_centered_head_keeps_original_activation_and_live_derivative(self):
        ns,_=f.source_definitions();head=ns['make_predictor'](259,1)
        x=torch.linspace(-.1,.2,259).reshape(1,259)
        raw=x
        for layer in list(head.children())[:-1]:raw=layer(raw)
        wrapper=f._CenteredHead(head,x,raw.detach())
        a=head(x);b=wrapper(x)
        self.assertTrue(torch.equal(a,b))
        ga=torch.autograd.grad(a.sum(),tuple(head.parameters()))
        gb=torch.autograd.grad(b.sum(),tuple(head.parameters()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(ga,gb)))

    def test_double_angle_radius_and_decoder_are_not_scalar_reencoding(self):
        z=latent();n=f.NeuralMaterial(z,101);value=n(positions())
        q=(1+.5*torch.stack((torch.cos(2*z[2]),torch.sin(2*z[2]))))/2
        self.assertTrue(torch.equal(n.rotation_predictor.target,torch.log(q/(1-q)).reshape(1,2)))
        expected=F.normalize(value.rotation_raw*2-1,dim=-1,eps=1e-6)
        self.assertTrue(torch.equal(expected,value.encoding))
        self.assertLess(float(abs(value.centered_radius-.5)),2e-7)
        # This is a permitted angular rounding difference, not bitwise theta equivalence.
        self.assertLess(float(abs(value.physical[0,2]-z[2])),2e-6)

    def test_bounds_live_gradients_and_expected_relu_zeros(self):
        n=f.NeuralMaterial(latent(),101);value=n(positions(9))
        value.physical[0].sum().backward()
        groups=f.gradient_summary(n)
        self.assertEqual(set(groups),{'feats_network',*(name for name,_ in f.HEADS)})
        self.assertTrue(all(g['finite'] and g['missing']==0 and g['nonzero']>0 for g in groups.values()))
        self.assertTrue(any(g['nonzero']<g['parameters'] for g in groups.values()))
        self.assertTrue(bool((value.physical[:,3:]>=0).all() and (value.physical[:,3:]<=1).all()))

    def test_detached_feature_path_is_reported_as_invalid(self):
        n=f.NeuralMaterial(latent(),101)
        hook=n.feats_network.register_forward_hook(lambda module,args,out:out.detach())
        n(positions()).physical.sum().backward();hook.remove()
        self.assertGreater(f.gradient_summary(n)['feats_network']['missing'],0)

    def test_seven_physical_directions_reachable_through_original_final_biases(self):
        n=f.NeuralMaterial(latent(),101)
        biases=[getattr(n,name).predictor[-2].bias for name,_ in f.HEADS]
        physical=n(positions()).physical[0]
        rows=[]
        for value in physical:
            gradients=torch.autograd.grad(value,biases,retain_graph=True)
            rows.append(torch.cat(gradients).detach().double().numpy())
        jacobian=np.stack(rows)
        self.assertEqual(jacobian.shape,(7,8))
        self.assertTrue(np.isfinite(jacobian).all())
        self.assertEqual(np.linalg.matrix_rank(jacobian),7)

    def test_zero_orientation_rejected_without_repair(self):
        n=f.NeuralMaterial(latent(),101)
        n.rotation_predictor.target.zero_()
        with self.assertRaisesRegex(SignatureError,'orientation encoding'):n(positions())
        self.assertTrue(torch.equal(n.rotation_predictor.target,torch.zeros_like(n.rotation_predictor.target)))

    def test_direct_and_N0_bitwise_material_and_gradient_parity(self):
        a=f.DirectMaterial(latent());b=f.ConstantMaterial(latent())
        pa,pb=(model(positions()).physical for model in (a,b))
        self.assertTrue(torch.equal(pa,pb))
        pa.square().sum().backward();pb.square().sum().backward()
        self.assertTrue(torch.equal(a.latent.grad,b.latent.grad))
        self.assertEqual(sum(p.numel() for p in b.parameters()),7)
        self.assertEqual(len(list(b.parameters())),1)
        self.assertTrue(torch.equal(pa[0],h.previous.original.physical(a.latent)))

    def test_pure_original_source_tangent_convention(self):
        # Only the pure basis function, no sampler, renderer or scientific shader call.
        root,_=h.baseline('material',h.PINS['material'])
        tree=ast.parse((root/'network/field.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MCShadingNetwork')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_tangent_bitangent_flat')
        namespace={'torch':torch,'F':F};exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),'original-basis','exec'),namespace)
        t,b=namespace[node.name](None,torch.tensor([[0.,0.,1.]]),None)
        self.assertTrue(torch.equal(t,torch.tensor([[0.,1.,0.]])))
        self.assertTrue(torch.equal(b,torch.tensor([[-1.,0.,0.]])))

    def test_invalid_dtype_seed_and_configuration_fail(self):
        with self.assertRaises(SignatureError):f.NeuralMaterial(latent(),102)
        with self.assertRaises(SignatureError):f.ConstantMaterial(latent().double())
        n=f.NeuralMaterial(latent(),101)
        with self.assertRaises(SignatureError):n(positions().double())
        n.double()
        with self.assertRaises(SignatureError):n(positions())
        with self.assertRaises(SignatureError):f.make_material('N2',latent(),101)


if __name__=='__main__':unittest.main()
