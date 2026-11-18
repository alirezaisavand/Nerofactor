"""One explicitly authorized, forward-only D8-T camera 28 intervention."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import donut_specular_d10 as d10
from benchmarks.donut_recovery_v1 import COLOR, METALLIC, SHADER, WIDTHS, inputs, json_new
from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, activate, digest


def render(plan, files, view, maps, selected, out, args):
    """All native imports and checkpoint deserialization follow the runtime gate."""
    import torch
    from benchmarks.generate_bell_e11 import run_with_cleanup
    from experiments.bell_neural_material_e16 import state_hash
    from experiments.donut_material_recovery_v1 import shader_lighting
    from infrastructure.material_recovery_e13_evaluation import native_tensor_default
    from infrastructure.sphere_replay import evaluation_scope
    from infrastructure.verified_ray_backend import load

    require(torch.__version__ == '1.8.1+cu111' and torch.version.cuda == '11.1'
            and not torch._C._GLIBCXX_USE_CXX11_ABI and torch.cuda.device_count() == 1
            and 'A100' in torch.cuda.get_device_name(0), 'Reviewed D8 CUDA runtime required')
    report = dict(protocol=d10.PROTOCOL, status='incomplete', cleanup_verified=False,
                  optimizer_returns=0, primary_rays=0, shader_calls=0, shadow_slots=0,
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  numpy=np.__version__, image=d10.IMAGE, scientific_source=PINS['material'],
                  commit=args.commit, freeze_sha256=args.freeze_sha256, authorization_sha256=args.authorization_sha256,
                  checkpoint_sha256=plan['artifacts']['checkpoint']['sha256'], outputs={})
    start_time = time.monotonic()
    ids, pixels, ring, archive, archive_se = selected
    ref = view['linear_components'][ids]
    limits = d10.LIMITS

    def budget():
        require(time.monotonic()-start_time < limits['inner_seconds'], 'Authorized D10 wall limit')
        require(sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file()) < limits['output_bytes'],
                'Authorized D10 output limit')

    def save(name, **values):
        budget()
        path = out/name
        with path.open('xb') as stream:
            np.savez_compressed(stream, **values)
        report['outputs'][name] = dict(bytes=path.stat().st_size, sha256=digest(path))
        budget()

    def work():
        torch.set_num_threads(limits['cpus'])
        torch.cuda.set_per_process_memory_fraction(limits['cuda_allocator_fraction'])
        report['backend'] = load('predicate_fp64', d10.IMAGE)
        activate('material', PINS['material'])
        import raytracing
        from network import field, renderer
        from scientific_variants import donut_tensor_d8 as tensor_math
        from scientific_variants.donut_tensor_shader_d8 import tensor_shader
        from scientific_variants.material_a_v1 import native_classes

        geometry, _, _ = inputs(files['geometry'].parent, files['camera'])
        tracer = raytracing.RayTracer(geometry['vertices_native'].astype(np.float32),
                                      geometry['triangles'].astype(np.int32))

        def trace(origins, directions):
            points, normals, depth = tracer.trace(origins, directions)
            return points, normals, depth.reshape(-1, 1), depth.reshape(-1) < 10

        def tensor(a):
            return torch.as_tensor(a, dtype=torch.float32, device='cuda')

        corrected, _ = native_classes(field.MCShadingNetwork, renderer.NeROMaterialRenderer,
                                      'A_corrected_core_v1', field.linear_to_srgb)
        shader = tensor_shader(corrected, field.linear_to_srgb)(deepcopy(SHADER), trace).cuda()
        shader.select_tensor_representation('D8-T')
        shader.cuda()
        # PyTorch 1.8 uses pickle. The exact researcher's checkpoint is authorized
        # for CPU loading only inside the read-only networkless container.
        saved = torch.load(files['checkpoint'], map_location='cpu')
        historical = d10.read(files['receipt'])
        expected = dict(protocol=historical['protocol'], variant='donut_tensor_ambient_v1', arm='D8-T',
                        benchmark_sha256=historical['benchmark_sha256'],
                        freeze_sha256=plan['D8_freeze_sha256'], completed_updates=10000)
        require(all(saved[name] == value for name, value in expected.items()),
                'D8-T checkpoint header differs')
        state = saved['network_state_dict']
        model_state = shader.state_dict()
        require(state.keys() == model_state.keys() and
                all(v.device.type == 'cpu' and v.dtype == model_state[k].dtype
                    and v.shape == model_state[k].shape and bool(torch.isfinite(v).all())
                    for k, v in state.items()), 'D8-T strict weight compatibility failed')
        shader.load_state_dict(state, strict=True)
        del saved, state, model_state
        before = state_hash(shader)
        report['restoration'] = dict(metadata=expected, strict=True, loaded_device='cpu',
                                     optimizer_restored=False, state_sha256=before)
        shader.requires_grad_(False)
        measured = {}
        chunks = []

        def render_role(role, indices, oracle_light=False):
            scope = nullcontext() if oracle_light else shader_lighting(shader, field.MCShadingNetwork, 'D8-T')
            capture = np.empty((len(indices), len(ids), 3, 3), dtype=np.float32)
            with scope:
                for begin, end, (p, n, wo), frame, values in chunks:
                    q, kd, metallic = values[role]
                    for i, draw in enumerate(indices):
                        budget()
                        require(report['shader_calls'] < limits['shader_calls'] and
                                report['shadow_slots']+len(p)*1536 <= limits['shadow_slots'],
                                'Authorized shader work limit')
                        report['shader_calls'] += 1
                        report['shadow_slots'] += len(p)*1536
                        seed, dedicated = d10.tape(begin, draw)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        result = shader.shade_tensor(p, n, wo, q, frame, kd, metallic, dedicated, False)
                        capture[i, begin:end] = np.stack([result[k].cpu().numpy() for k in d10.COMPONENTS], 1)
            return capture

        with evaluation_scope(shader, torch, np) as scoped, torch.no_grad(), native_tensor_default(torch):
            shader.eval()
            eligible = []
            recovered_checks = dict(Q=0., kd=0., metallic=0., oracle_Q=0.)
            for begin in range(0, len(ids), d10.CHUNK):
                end = min(begin+d10.CHUNK, len(ids))
                sample, px = ids[begin:end], pixels[begin:end]
                p, n, wo = (tensor(view[k][sample]) for k in ('position', 'normals', 'outgoing'))
                q, frame, kd, m = shader.predict_tensor(p, n)
                axis = tensor(view['axis'][sample])
                wide, narrow = (n.new_full((len(sample), 1), value) for value in WIDTHS)
                oracle = tensor_math.chart_q(wide, narrow, axis, frame)
                a, b, director, _ = tensor_math.export_axes(q, frame)
                valid = d10.orientation_eligible(a, b)
                eligible.extend(valid[:, 0].cpu().numpy().tolist())
                width_q = tensor_math.chart_q(wide, narrow, director, frame)
                axis_q = tensor_math.chart_q(a, b, axis, frame)
                width_q = torch.where(valid[..., None], width_q, q)
                axis_q = torch.where(valid[..., None], axis_q, q)
                for key, prediction, target in (
                    ('Q', tensor_math.world_tensor(q, frame), maps['Q_tensor_direct'].reshape(-1, 3, 3)[px]),
                    ('oracle_Q', tensor_math.world_tensor(oracle, frame), view['Q_world'][sample]),
                    ('kd', kd, maps['kd'].reshape(-1, 3)[px]),
                    ('metallic', m, maps['metallic'].reshape(-1, 1)[px]),
                ):
                    recovered_checks[key] = max(recovered_checks[key],
                        float(np.max(abs(prediction.cpu().numpy()-target))))
                values = dict(R1=(q, kd, m), R2=(oracle, kd, m), R3=(q, kd, m), R4=(oracle, kd, m),
                              R5=(oracle, n.new_tensor(COLOR).expand_as(kd), n.new_full(m.shape, METALLIC)),
                              width_only=(width_q, kd, m), orientation_only=(axis_q, kd, m))
                chunks.append((begin, end, (p, n, wo), frame, values))
            eligible = np.asarray(eligible, dtype=bool)
            report['field_gate_max_absolute'] = recovered_checks
            require(max(recovered_checks.values()) <= d10.POLICY['replay_atol'],
                    'Recovered or oracle Q/scalar fields differ from admitted arrays')

            first = render_role('R1', range(d10.ARCHIVED_DRAWS))
            save('R1-archived-replay.npz', component_draws=first)
            report['archived_replay'] = d10.verify_replay(first, archive, archive_se)
            require(report['archived_replay']['passed'], 'R1 failed archived eight-draw means or SE replay')
            for role in d10.BASE:
                begin = d10.ARCHIVED_DRAWS if role == 'R1' else 0
                draw = render_role(role, range(begin, d10.DRAWS), oracle_light=role in ('R3', 'R4', 'R5'))
                if role == 'R1':
                    draw = np.concatenate([first, draw])
                save(role+'.npz', component_draws=draw)
                if role != 'R5':
                    measured[role] = d10.draw_summary(draw, ref, ring)
            context = d10.split_context(measured)
            report['conditional_split_light'] = context
            if context is not None:
                for role in d10.SWAPS:
                    draw = render_role(role, range(d10.DRAWS), oracle_light=context == 'oracle')
                    save(role+'.npz', component_draws=draw)
            save('panel.npz', pixels=pixels, ring=ring, orientation_eligible=eligible)
        roles = len(d10.BASE) + (len(d10.SWAPS) if context is not None else 0)
        require(scoped['RNG_and_modes_restored'] and before == state_hash(shader)
                and shader._environment is None, 'Render mutated restored model or runtime state')
        require(report['shader_calls'] == roles*d10.DRAWS*len(chunks)
                and report['shadow_slots'] == roles*d10.DRAWS*d10.POINTS*1536,
                'Incomplete matched scientific work')
        report['RNG_modes_and_weights_unchanged'] = True
        report['resources'] = dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        budget()

    try:
        run_with_cleanup(work, torch)
        report['cleanup_verified'] = True
        d10.freeze(d10.ROOT/d10.FREEZE, args.freeze_sha256)
        for name, item in plan['artifacts'].items():
            d10.checked(files[name], item)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-start_time
        json_new(out/'render-receipt.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--commit')
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = d10.freeze(d10.ROOT/d10.FREEZE, args.freeze_sha256)
    paths = {name: Path(item['native' if args.execute else 'host'])
             for name, item in plan['artifacts'].items()}
    if args.execute:
        auth = Path('/authorization.json')
        require(args.commit and digest(auth) == args.authorization_sha256
                and d10.read(auth) == d10.authorization(plan, args.commit, args.freeze_sha256),
                'Exact one-launch approval including checkpoint trust required')
        require(os.getuid() != 0 and all(os.statvfs(p).f_flag & os.ST_RDONLY
                    for p in [*paths.values(), auth, d10.ROOT, Path('/')])
                and {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]} == {'lo'},
                'Expected non-root, networkless read-only runtime and inputs')
    files, views, maps, selected = d10.panel(plan, paths)
    if not args.execute:
        print(f'D10 metadata only: verified {len(selected[1])} camera points and {sum(selected[2])} ring pixels')
        return
    out = Path('/outputs/run')
    out.mkdir()
    render(plan, files, views[d10.POLICY['camera']], maps, selected, out, args)


if __name__ == '__main__':
    main()
