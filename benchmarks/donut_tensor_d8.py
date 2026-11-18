"""Frozen work and initialization definition for the paired D8 comparison."""

from benchmarks.donut_surface_q_d6 import LIMITS as D6_LIMITS, POLICY

ARMS=('D8-C','D8-T')
PROTOCOL='donut-kurt-paired-tensor-d8-v1'
REPRESENTATION=dict(initial_width=.25,training_tape_seed_offset=603300000,
                   evaluation_tape_seed_offset=1600000,initial_gate_seed=120008,
                   native_gate_points=24,initial_Q_atol=2e-8,initial_RGB_atol=1e-6,
                   export_world_Q_atol=1e-6,
                   perturbation='legacy_scalar_only_v1')


def limits(base):
    result=dict(base,**D6_LIMITS)
    result.update(wall_seconds=7500,
        shader_calls=base['shader_calls']+2,
        shadow_rays=base['shadow_rays']+2*REPRESENTATION['native_gate_points']*1536)
    return result
