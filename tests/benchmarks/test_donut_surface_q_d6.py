"""Focused synthetic checks for geometry sampling and additive D6 Q loss."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import donut_recovery_v1 as recovery
from benchmarks import donut_surface_q_d6 as d6
from infrastructure.stage_sources import digest
from infrastructure.surface_q_regularization import candidate_pair_loss


def test_mesh_sampling_uses_smooth_normals_and_reserves_diagnostics():
    vertices = np.array([[0.,0,0],[.005,0,0],[0,.005,0],[-.005,0,0],[0,-.005,0]])
    faces = np.array([[0,1,2],[0,2,3],[0,3,4],[0,4,1]])
    smooth = np.array([.1,0,1.])
    smooth /= np.linalg.norm(smooth)
    geometry = dict(vertices_native=vertices, triangles=faces, triangle_loops=np.arange(12).reshape(4,3),
                    shading_normals_world=np.tile(smooth,(12,1)))
    domain = d6.neighborhoods(geometry, dict(d6.POLICY, diagnostic_pairs=1))
    np.testing.assert_allclose(domain['normals0'], np.tile(smooth,(4,1)))
    assert len(domain['training_ids']) == 3 and len(domain['diagnostic_ids']) == 1
    rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    sampler = d6.NeighborhoodSampler(domain['training_ids'], 4, 42)
    control = d6.NeighborhoodSampler(domain['training_ids'], 4, 42)
    a, b = sampler.draw(2), sampler.draw(2)
    np.testing.assert_array_equal(np.r_[a,b], np.r_[control.draw(2),control.draw(2)])
    np.testing.assert_array_equal(np.sort(np.r_[a,b][:3]), domain['training_ids'])
    assert not sampler.counts[domain['diagnostic_ids']].any()
    report = d6.coverage_summary(domain, sampler.counts, np.array([1,0,0,2]))
    assert report['visited_faces'] == 4 and report['unique_pairs'] == 3
    assert report['unsampled_in_fit']['visited_faces'] == 2 and report['surface_area_fraction'] == 1
    assert sampler.checkpoint()['cursor'] == 1
    after = np.random.get_state()
    assert rng[0]==after[0] and rng[2:]==after[2:]
    np.testing.assert_array_equal(rng[1],after[1])
    assert torch.equal(cpu_rng, torch.get_rng_state())


def test_q_only_term_matches_legacy_candidate_and_differentiates_width_and_rotation():
    x = torch.tensor([.07,.035,.44], requires_grad=True, dtype=torch.float64)

    class Predictor:
        swapped = False

        def predict_anisotropic_components(self, points):
            mx, my = .35 + x[0] * points[:,:1], .08 + x[1] * points[:,1:2]
            theta = x[2] * points[:,:2].sum(-1, keepdim=True)
            rotation = torch.cat([torch.cos(2*theta),torch.sin(2*theta)],dim=-1)
            if self.swapped:
                mx, my, rotation = my, mx, -rotation
            # Nonconstant scalar heads ensure the old scalar companion is nonzero.
            return mx,my,points[:,:1],points,rotation,rotation

    model = Predictor()
    p0 = torch.tensor([[.1,.1,0],[.1,.2,0]], dtype=x.dtype)
    p1 = torch.tensor([[.2,.2,0],[.4,.4,0]], dtype=x.dtype)
    normal = torch.tensor([[0.,0,1.],[0.,0,1.]], dtype=x.dtype)
    batch = dict(points0=p0,points1=p1,normals0=normal,normals1=normal,
                 distance=x.new_tensor([.01,.02]))
    loss = d6.surface_q_term(model,batch,d6.POLICY,torch)
    legacy = candidate_pair_loss(model,dict(batch,weights=x.new_ones(2)),torch)
    assert legacy['loss_surface_scalar'] > 0 and torch.equal(loss,legacy['loss_surface_q'])
    assert torch.allclose(d6.surface_q_term(model,dict(batch,distance=batch['distance']*2),d6.POLICY,torch),
                          loss/4,rtol=1e-12,atol=0)
    model.swapped = True
    assert torch.allclose(d6.surface_q_term(model,batch,d6.POLICY,torch),loss,rtol=1e-12,atol=0)
    loss.backward()
    assert torch.isfinite(x.grad).all() and (x.grad.abs() > 0).all()


def test_d6_runtime_rejects_a_modified_coefficient_or_wrong_phase(tmp_path, monkeypatch):
    definition = dict(protocol=d6.PROTOCOL, limits=d6.LIMITS, spatial_policy=d6.POLICY,
                      benchmark_generation_sha256='dataset', dataset_admission_sha256='admitted', initial_sha256='initial')
    freeze = dict(surface_q_training=definition, host_surface_output='batch')
    monkeypatch.setattr(recovery, 'verify_freeze', lambda *args: freeze)
    monkeypatch.setattr(os,'geteuid',lambda:0)
    auth = tmp_path/'authorization.json'
    value=dict(definition, approved=True, commit='commit', freeze_sha256='freeze', image_id=recovery.IMAGE,
               selected_gpu='0',maximum_execution_launches=1,host_output_directory='batch',
               diffuse_white_in_objective=True, output_directories={'D6':'/outputs/D6'})
    args=SimpleNamespace(freeze=tmp_path/'freeze.json',freeze_sha256='freeze',authorization=auth,
                         commit='commit',image_id=recovery.IMAGE,output=Path('/outputs/D6'))
    auth.write_text(json.dumps(value));args.authorization_sha256=digest(auth)
    with pytest.raises(ValueError,match='Fresh owned'):
        recovery.runtime(args,'D6')
    with pytest.raises(ValueError,match='D6 requires'):
        recovery.runtime(args,'D2')
    auth.write_text(json.dumps(dict(value,spatial_policy=dict(d6.POLICY,coefficient=.1))))
    args.authorization_sha256=digest(auth)
    with pytest.raises(ValueError,match='D6 authorization differs'):
        recovery.runtime(args,'D6')
