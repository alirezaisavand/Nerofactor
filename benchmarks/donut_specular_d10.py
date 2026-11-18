"""Frozen inputs and paired-draw statistics for the Donut D10 intervention."""
import json
from pathlib import Path

import numpy as np

from benchmarks.donut_recovery_v1 import SIZE, numeric
from infrastructure.signatures import require
from infrastructure.stage_sources import digest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'donut-specular-attribution-d10-v1'
FREEZE = 'docs/protocols/DONUT_SPECULAR_ATTRIBUTION_D10_V1.freeze.json'
IMAGE = 'sha256:40024bfedce45251e359d6af63f26bf5a497136d2916d8b06046cc09b5f45331'
COMPONENTS = ('linear_diffuse', 'linear_specular', 'linear_rgb')
BASE = ('R1', 'R2', 'R3', 'R4', 'R5')
SWAPS = ('width_only', 'orientation_only')
DRAWS, ARCHIVED_DRAWS, POINTS, CHUNK = 64, 8, 4849, 512
LUMA = np.array([.2126, .7152, .0722])
POLICY = dict(camera=28, ring_pixels=485, draws=DRAWS, archived_draws=ARCHIVED_DRAWS,
              chunk=CHUNK, points=POINTS, replay_atol=1e-6, split_SSE_fraction=.25,
              orientation_gap=.1, t_critical_63=1.998340542520741,
              mixed_components=list(COMPONENTS), coverage_radii=[.005, .01, .02],
              coverage_normal_degrees=30, highlight_min_specular_luminance=.01,
              highlight_min_view_median_multiple=3.)
MAX_CONDITIONS = len(BASE) + len(SWAPS)
LIMITS = dict(shader_calls=MAX_CONDITIONS*DRAWS*((POINTS+CHUNK-1)//CHUNK),
              shadow_slots=MAX_CONDITIONS*DRAWS*POINTS*1536,
              optimizer_returns=0, inner_seconds=1200, foreground_seconds=1320,
              output_bytes=512*1024**2, host_memory_gib=64, cuda_allocator_fraction=.25,
              cpus=4, pids=128)


def read(path):
    return json.loads(path.read_text())


def checked(path, record):
    require(path.is_file() and path.resolve() == path and path.stat().st_size == record['bytes']
            and digest(path) == record['sha256'], 'Frozen artifact differs: '+str(path))
    return path


def freeze(path, expected):
    require(path.resolve() == path and digest(path) == expected, 'D10 freeze hash differs')
    plan = read(path)
    require(plan['protocol'] == PROTOCOL and plan['policy'] == POLICY and plan['limits'] == LIMITS
            and plan['image'] == IMAGE and plan['selected_gpu'] == '0', 'D10 freeze definition differs')
    for name, sha in plan['sources'].items():
        file = ROOT / name
        require(ROOT in file.parents and file.resolve() == file and digest(file) == sha,
                'Frozen source differs: '+name)
    return plan


def authorization(plan, commit, freeze_sha):
    return dict(approved=True, protocol=PROTOCOL, commit=commit, freeze_sha256=freeze_sha,
                image=IMAGE, selected_gpu='0', host_output=plan['host_output'], limits=LIMITS,
                maximum_execution_launches=1, trusted_checkpoint_sha256=plan['artifacts']['checkpoint']['sha256'],
                isolated_cpu_pickle_authorized=True, restore_optimizer=False)


def tape(start, draw):
    seed = 800000 + POLICY['camera']*1000 + start + draw
    return seed, 1600000 + seed


def camera_inputs(view, maps, errors, ring):
    ids = np.flatnonzero(view['shader_valid'])
    pixels = view['pixel_indices'][ids]
    require(len(ids) == POINTS and np.array_equal(pixels, np.flatnonzero(maps['hit_mask']))
            and len(np.unique(pixels)) == POINTS and view['hit_mask'].shape == (SIZE, SIZE),
            'Admitted camera and archived pixel order differ')
    region = np.asarray(ring)
    require(region.dtype.kind in 'iu' and region.shape == (POLICY['ring_pixels'],)
            and len(np.unique(region)) == len(region) and np.isin(region, pixels).all(),
            'Frozen ring is incomplete or out of bounds')
    archived = np.stack([maps[name].reshape(-1, 3)[pixels] for name in COMPONENTS], axis=1)
    standard_error = errors['native_component_SE'][pixels][:, [1, 2, 0]]
    reference = view['linear_components'][ids]
    require(reference.shape == archived.shape == standard_error.shape == (POINTS, 3, 3),
            'Unexpected component axes')
    return ids, pixels, np.isin(pixels, region), archived, standard_error


def verify_replay(draws, archived, standard_error):
    require(draws.shape[0] == ARCHIVED_DRAWS and draws.shape[1:] == archived.shape == standard_error.shape,
            'Incomplete archived replay')
    delta = np.abs(draws.mean(0) - archived)
    delta_se = np.abs(draws.std(0, ddof=1)/np.sqrt(ARCHIVED_DRAWS) - standard_error)
    tol = POLICY['replay_atol']
    return dict(passed=bool((delta <= tol).all() and (delta_se <= tol).all()),
                component_mean_max_absolute=delta.max(axis=(0, 2)).tolist(),
                component_SE_max_absolute=delta_se.max(axis=(0, 2)).tolist(), tolerance=tol)


def orientation_eligible(major, minor):
    return (major**2 - minor**2)/(major**2 + minor**2) >= POLICY['orientation_gap']


def draw_summary(capture, reference, region):
    """Return point estimates and leave-one-whole-draw estimates in the same order."""
    draws = capture[:, region].astype(np.float64)
    require(len(draws) == DRAWS and draws.shape[1]>0 and np.isfinite(draws).all(),
            'Missing or nonfinite matched draw block')
    mean = draws.mean(axis=0)
    ref = reference[region]
    residual = mean-ref
    loo = (draws.sum(axis=0)-draws)/(DRAWS-1)-ref
    # Last dimension in the summaries is diffuse, specular, combined RGB.
    return dict(SSE=(np.square(residual).sum(axis=(0, 2)),
                     np.square(loo).sum(axis=(1, 3))),
                mean_signed_luminance=((residual@LUMA).mean(axis=0),
                                       (loo@LUMA).mean(axis=1)))


def interval(estimate, leave_one_out):
    se = np.sqrt((DRAWS-1)/DRAWS*np.square(leave_one_out-leave_one_out.mean(axis=0)).sum(axis=0))
    width = se*POLICY['t_critical_63']
    return dict(value=estimate.tolist(), paired_MC_SE=se.tolist(),
                paired_MC_95_lower=(estimate-width).tolist(), paired_MC_95_upper=(estimate+width).tolist())


def effect(stats, metric, weights):
    return interval(*(sum(weight*stats[role][metric][i] for role, weight in weights.items())
                      for i in (0, 1)))


def split_context(stats):
    """Trigger only on a large, positive, matched reduction in ring RGB SSE."""
    for light, before, after in [('recovered', 'R1', 'R2'), ('oracle', 'R3', 'R4')]:
        total = stats[before]['SSE'][0][-1]
        reduction = effect(stats, 'SSE', {before: 1, after: -1})
        if (total > 0 and reduction['value'][-1] >= POLICY['split_SSE_fraction']*total
                and reduction['paired_MC_95_lower'][-1] > 0):
            return light
    return None


def panel(plan, paths):
    from experiments.donut_material_recovery_v1 import benchmark
    files = {name: checked(paths[name], item) for name, item in plan['artifacts'].items()}
    admission = read(files['admission'])
    generation_sha = plan['artifacts']['generation']['sha256']
    require(admission['researcher_admitted_dataset'] is True
            and admission['generation']['sha256'] == generation_sha, 'Expected the admitted Donut oracle')
    _, views = benchmark(files['generation'].parent, generation_sha)
    receipt = read(files['receipt'])
    require(receipt['status'] == 'complete' and receipt['scientific_work_complete']
            and receipt['cleanup_verified'] and receipt['optimizer_returns'] == 10000
            and receipt['arm'] == 'D8-T' and receipt['commit'] == plan['D8_commit']
            and receipt['freeze_sha256'] == plan['D8_freeze_sha256']
            and receipt['initial']['variant'] == 'donut_tensor_ambient_v1'
            and receipt['benchmark_sha256'] == generation_sha, 'Expected the complete admitted D8-T endpoint')
    for role in ('checkpoint', 'maps', 'errors'):
        artifact = plan['artifacts'][role]
        name = str(Path(artifact['host']).relative_to(Path(plan['artifacts']['receipt']['host']).parent))
        require(all(artifact[k] == receipt['outputs'][name][k] for k in ('bytes', 'sha256')),
                'Endpoint identity differs from the D8 completion receipt')
    ring = read(ROOT/plan['ring_source'])['D8_T_positive_excess_decile']['pixels']
    view = views[POLICY['camera']]
    maps = numeric(files['maps'])
    selected = camera_inputs(view, maps, numeric(files['errors']), ring)
    return files, views, maps, selected
