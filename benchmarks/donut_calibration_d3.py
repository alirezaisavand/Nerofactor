"""Fixed spherical-mean calibration of the original directional light MLP."""
from contextlib import contextmanager
from time import perf_counter

import numpy as np

from infrastructure.signatures import require

PROTOCOL = 'donut-kurt-calibrated-recovery-d3-v1'
GRID = (128, 256)
NODES = GRID[0] * GRID[1]
CALIBRATION = {
    'grid': list(GRID),
    'quadrature': 'equirectangular midpoint, normalized sin(theta) weights',
    'mean_linear_RGB': [0.13995771766028667, 0.14655455327313444, 0.1674445384474473],
    'absolute_mean_tolerance': 2e-6,
    'reference_map_sha256': '5380d6b6977decd196da8b18a5017ebad6c55c4c77b341a29d31c986e53d93a4',
}
NEW_LIMITS = dict(component_reverse_sweeps=30, calibration_evaluations=10006,
                  calibration_directions=10006 * NODES, wall_seconds=3720)


def quadrature():
    y, x = np.mgrid[:GRID[0], :GRID[1]]
    phi = (x + .5) / GRID[1] * 2 * np.pi
    theta = (y + .5) / GRID[0] * np.pi
    directions = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                           np.cos(theta)], axis=-1)
    weights = np.sin(theta)
    return directions, weights / weights.sum()


class CalibratedEnvironment:
    """One differentiable denominator per update, one no-grad denominator per endpoint."""

    def __init__(self, raw_environment, directions, weights, target, torch):
        self.raw_environment = raw_environment
        self.directions = directions
        self.points = torch.zeros_like(directions)
        self.weights = weights
        self.target = target
        self.scale = None
        self.raw_mean = None
        self.grid = None

    @contextmanager
    def scope(self, budget, timing, torch, *, evaluation):
        require(self.scale is None, 'Nested D3 calibration scope')
        budget('calibration_evaluations', 1)
        budget('calibration_directions', len(self.directions))
        started = perf_counter()
        start, stop = None, None
        if self.directions.is_cuda:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        grid = self.raw_environment(self.points, self.directions)
        require(grid.shape == self.directions.shape, 'D3 grid radiance shape differs')
        raw_mean = (grid * self.weights).sum(dim=0)
        scale = self.target / raw_mean
        if stop is not None:
            stop.record()
        valid = (torch.isfinite(grid).all() & (grid >= 0).all() &
                 torch.isfinite(raw_mean).all() & (raw_mean > 0).all() &
                 torch.isfinite(scale).all() & (scale > 0).all())
        require(bool(valid), 'Invalid D3 calibration radiance or denominator')
        require(evaluation or scale.requires_grad, 'D3 calibration denominator lost its gradient')
        key = 'evaluation' if evaluation else 'training'
        timing[key]['setup_wall_seconds'] += perf_counter() - started
        if stop is not None:
            timing[key]['grid_forward_and_reduction_CUDA_seconds'] += start.elapsed_time(stop) / 1000
        self.scale, self.raw_mean, self.grid = scale, raw_mean, grid if evaluation else None
        try:
            yield
        finally:
            self.scale = self.raw_mean = self.grid = None

    def __call__(self, points, directions):
        require(self.scale is not None, 'D3 light used outside calibration scope')
        return self.raw_environment(points, directions) * self.scale

    def grid_radiance(self):
        require(self.grid is not None, 'D3 grid export requires an evaluation scope')
        return self.grid * self.scale
