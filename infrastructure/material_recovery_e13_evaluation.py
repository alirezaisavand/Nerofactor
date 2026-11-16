"""Fixed-endpoint export and native per-view rendering, E.13 only.

Scientific dependencies are supplied/imported only after runtime admission.
NPZ input readers disallow pickle; no historical checkpoints are read.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

from infrastructure.signatures import require
from infrastructure.material_recovery_e13 import material_errors, summarize_material


@contextmanager
def native_tensor_default(torch):
    """Match original A forward/NVS default, restoring it even on failure."""
    previous = torch.tensor([]).type()
    try:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        yield
    finally:
        torch.set_default_tensor_type(previous)


def numeric(path, np):
    with np.load(str(path), allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    require(result and all(a.dtype.kind in 'biuf' and np.isfinite(a).all() for a in result.values()),
            'Invalid/nonfinite numeric evidence: '+str(path))
    return result


def material_export(model, geometry, run, splits, torch, np):
    grouped = {k: [] for k in ('train', 'validation', 'held_view_nvs')}
    report = {}
    shader = model.shader_network
    device = next(shader.parameters()).device
    count = 0; batches = 0
    with torch.no_grad():
        for image in map(str, range(128)):
            truth = numeric(geometry/'labels'/(image+'.npz'), np)
            pieces = {k: [] for k in ('mx', 'my', 'metallic', 'color', 'rotation', 'rotation_raw', 'T', 'B', 'actual_T', 'actual_B')}
            for start in range(0, len(truth['position']), 4096):
                pts = torch.as_tensor(truth['position'][start:start+4096], dtype=torch.float32, device=device)
                n = torch.as_tensor(truth['normals'][start:start+4096], dtype=torch.float32, device=device)
                mx, my, m, color, r, raw = shader.predict_anisotropic_components(pts)
                batches += 1
                t, b = shader.compute_tangent_bitangent_flat(n, None)
                if model.material_variant == 'A_corrected_core_v1':
                    from scientific_variants.material_a_v1 import orientation
                    c, s = orientation(r)
                else:
                    # Exact original sampler decoder, WITHOUT calling its RNG/sampling.
                    c = torch.sqrt(((r[:, :1]+1)/2).clamp(min=1e-6)).clamp(0, 1)
                    s = torch.sign(r[:, 1:])*torch.sqrt(((1-r[:, :1])/2).clamp(min=1e-6)).clamp(0, 1)
                values = (mx, my, m, color, r, raw, t, b, c*t+s*b, -s*t+c*b)
                for key, value in zip(pieces, values):
                    pieces[key].append(value.detach().cpu().numpy())
            pred = {k: np.concatenate(v) for k, v in pieces.items()}
            require(all(np.isfinite(v).all() for v in pred.values()), 'Nonfinite native material export')
            errors = material_errors(pred, truth, np)
            count += len(truth['position'])
            split = next(k for k, ids in splits.items() if image in ids)
            grouped[split].append(errors)
            report[image] = summarize_material([errors], np)
            np.savez_compressed(str(run.output('materials/'+image+'.npz')), pixel_indices=truth['pixel_indices'],
                                **pred, **errors)
    require(count == 515698, 'Material export observation count differs from admitted labels')
    require(batches <= 256, 'Material-export predictor-call cap exceeded')
    return dict(observations=count, predictor_batches=batches, per_view=report,
                split_aggregates={k: summarize_material(v, np) for k, v in grouped.items()})


@contextmanager
def capture_native(model, torch, np):
    """Observe actual returned linear components; never invert clipped sRGB.

    The native test_step still chooses cameras, rays, hit masks, chunks and shade.
    The profiler copies returned values only; it makes no shader/RNG queries.
    """
    require(sys.getprofile() is None, 'Evaluation requires unprofiled process')
    shader = model.shader_network
    radiance_code = shader.compute_radiance.__func__.__code__
    rays_code = model._construct_ray_batch.__func__.__code__
    chunks = []; batches = []
    def observe(frame, event, value):
        if event != 'return' or value is None:
            return
        if frame.f_code is radiance_code and frame.f_locals.get('self') is shader:
            require(isinstance(value, tuple) and len(value) == 6, 'Native radiance return changed')
            chunks.append(np.stack([v.detach().cpu().numpy() for v in value[:3]], axis=1))
        elif frame.f_code is rays_code and frame.f_locals.get('self') is model:
            batches.append({k: v.detach().cpu().numpy() for k, v in value.items()})
    try:
        sys.setprofile(observe)
        yield chunks, batches
    finally:
        sys.setprofile(None)


def image_metrics(pred, gt, hit, native_gt, np, native_pred=None):
    """Independent reference errors plus exact native uint8 PSNR/SSIM conventions."""
    from skimage.metrics import structural_similarity
    require(pred.shape == gt.shape == (128, 128, 3, 3) and hit.shape == (128, 128) and hit.any(),
            'Invalid complete prediction/reference image')
    require(np.isfinite(pred).all() and (pred >= 0).all(), 'Nonfinite/negative predicted radiance')
    def srgb(x):
        return np.where(x <= .0031308, x*12.92, 1.055*np.maximum(x, np.finfo(np.float32).eps)**(1/2.4)-.055)
    p = srgb(pred[:, :, 0]) if native_pred is None else native_pred
    g = native_gt
    # Native color_map_backward deliberately truncates; do not round to nearest.
    pu = np.clip(p*255, 0, 255).astype(np.uint8)
    gu = np.clip(g*255, 0, 255).astype(np.uint8)
    mse = float(np.mean((gu.astype(np.float32).reshape(-1, 3)-pu.astype(np.float32).reshape(-1, 3))**2, axis=0).mean())
    result = dict(native_psnr=10*float(np.log10(255**2/mse)) if mse > 0 else None,
                  native_psnr_infinite=(mse == 0),
                  native_ssim=float(structural_similarity(gu, pu, win_size=11, channel_axis=2, data_range=255)),
                  foreground_pixels=int(hit.sum()), clipped_prediction_channels=int((p > 1).sum()))
    # References order diffuse,specular,total; native captured order total,diffuse,specular.
    for pi, gi, name in ((0, 2, 'total'), (1, 0, 'diffuse'), (2, 1, 'specular')):
        a = pred[:, :, pi][hit].astype(np.float64); b = gt[:, :, gi][hit]
        error = a-b
        result[name] = dict(rmse=float(np.sqrt(np.mean(error**2))),
                            relative_l2=float(np.linalg.norm(error)/max(np.linalg.norm(b), 1e-15)),
                            squared_error=float((error**2).sum()), reference_squared=float((b**2).sum()), channels=int(b.size))
    result['foreground_srgb_rmse'] = float(np.sqrt(np.mean((p[hit]-srgb(gt[:, :, 2])[hit])**2)))
    return result


def render_evaluation(model, geometry, references, generation, run, torch, np):
    import imageio
    from benchmarks.bell_reference_e11 import environment
    results = {}; calls = 0; rays = 0; directions = 0
    model.eval()
    for mode in ('source', 'relight'):
        for image in generation['splits']['validation']+generation['splits']['held_view_nvs']:
            nvs = image in list(map(str, model.nvs_ids))
            ids = list(map(str, model.nvs_ids if nvs else model.test_ids))
            index = ids.index(image)
            require(str(ids[index]) == image, 'Evaluation source camera mismatch')
            definition = generation['known_environment'] if mode == 'source' else generation['relighting_environment']
            # Source path uses actual configured lighting; relight changes only its
            # explicit environment callback, through the accepted native interface.
            from contextlib import ExitStack
            with native_tensor_default(torch):
                with torch.no_grad(), ExitStack() as stack:
                    if mode == 'relight':
                        stack.enter_context(model.shader_network.environment(lambda p, d: environment(d, definition), 'e11-v3-unseen-plus90Y'))
                    chunks, batches = stack.enter_context(capture_native(model, torch, np))
                    outputs = model.test_step(index, is_nvs=nvs)
            require(len(batches) == 1 and chunks, 'Native ray/radiance captures incomplete')
            batch = batches[0]; hit = batch['hit_mask'].reshape(-1)
            expected = numeric(geometry/'rays'/(image+'.npz'), np)
            geometry_errors = dict(position_max=float(np.max(abs(batch['inters'][hit]-expected['positions'][hit]))),
                                   normal_max=float(np.max(abs(batch['normals'][hit]-expected['normals'][hit]))),
                                   direction_max=float(np.max(abs(batch['rays_d']-expected['directions']))),
                                   origin_max=float(np.max(abs(batch['rays_o']-expected['origins']))),
                                   depth_max=float(np.max(abs(batch['depth'].reshape(-1)[hit]-expected['depth'][hit]))),
                                   hits=int(hit.sum()), misses=int((~hit).sum()))
            require(np.array_equal(hit, expected['hit']) and geometry_errors['position_max'] <= 2e-4 and
                    geometry_errors['depth_max'] <= 2e-4 and geometry_errors['normal_max'] <= 2e-6 and
                    geometry_errors['origin_max'] <= 2e-6 and geometry_errors['direction_max'] <= 2e-6,
                    'Native/source geometry, ray or normal convention mismatch')
            linear = np.zeros((16384, 3, 3), np.float32)
            values = np.concatenate(chunks)
            require(len(values) == int(hit.sum()), 'Captured linear output count differs from native hits')
            linear[hit] = values
            srgb = outputs['rgb_pr'].detach().cpu().numpy()
            rgb_gt = outputs['rgb_gt'].detach().cpu().numpy()
            ref = numeric(references/(('relight/' if mode == 'relight' else '')+image+'.npz'), np)
            # Relight native GT comes from the source observation. Use the frozen
            # independently rendered relight PNG for the native-equivalent metric.
            if mode == 'relight':
                rgb_gt = imageio.imread(str(references/'relight'/(image+'.png'))).astype(np.float32)/255
            key = mode+'-'+image
            results[key] = image_metrics(linear.reshape(128,128,3,3), ref['linear_components'], hit.reshape(128,128), rgb_gt, np, native_pred=srgb)
            results[key]['geometry_comparison'] = geometry_errors
            results[key]['native_rgb_vs_linear_transfer_max'] = float(np.max(abs(
                srgb-np.where(linear.reshape(128,128,3,3)[:, :, 0] <= .0031308,
                              linear.reshape(128,128,3,3)[:, :, 0]*12.92,
                              1.055*np.maximum(linear.reshape(128,128,3,3)[:, :, 0], np.finfo(np.float32).eps)**(1/2.4)-.055))))
            require(results[key]['native_rgb_vs_linear_transfer_max'] <= 2e-6, 'Captured/native RGB transfer mismatch')
            np.savez_compressed(str(run.output('evaluations/'+key+'.npz')), linear_components=linear.reshape(128,128,3,3),
                                native_srgb=srgb, ground_truth_srgb=rgb_gt, hit=hit.reshape(128,128))
            imageio.imwrite(str(run.output('renders/'+key+'.png')), np.clip(srgb*255,0,255).astype(np.uint8))
            rays += 16384; directions += int(hit.sum())*1536; calls += len(chunks)
    require(len(results) == 12 and rays == 196608 and directions <= 301989888 and calls <= 192, 'Evaluation budget mismatch')
    aggregates = {}
    for mode in ('source', 'relight'):
        for name, ids in (('validation', generation['splits']['validation']), ('held', generation['splits']['held_view_nvs'])):
            rows = [results[mode+'-'+i] for i in ids]
            aggregates[mode+'-'+name] = dict(
                native_psnr_mean=float(np.mean([r['native_psnr'] for r in rows])) if all(r['native_psnr'] is not None for r in rows) else None,
                native_ssim_mean=float(np.mean([r['native_ssim'] for r in rows])),
                pooled_linear_relative_l2=float(np.sqrt(sum(r['total']['squared_error'] for r in rows)/
                                                        max(sum(r['total']['reference_squared'] for r in rows),1e-15))))
    return dict(per_view=results, aggregates=aggregates, primary_rays=rays, shader_chunks=calls, direction_slots=directions,
                prediction_component_order=['total','diffuse','specular'], reference_component_order=['diffuse','specular','total'])
