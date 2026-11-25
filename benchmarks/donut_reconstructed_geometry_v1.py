"""Frozen Donut geometry inputs and bounded, independent surface calculations."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from benchmarks import donut_heterogeneous_v1 as h
from benchmarks.donut_recovery_v1 import INPUTS, PANEL, SIZE, SPLITS, select_panel, unit
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest
from infrastructure.verified_ray_backend import identity


PROTOCOL = 'donut-reconstructed-geometry-validation-v1'
FREEZE = 'docs/protocols/DONUT_RECONSTRUCTED_GEOMETRY_VALIDATION_V1.freeze.json'
PRELIMINARY = 'docs/results/publication_gate1/donut-geometry-readiness-v1.json'
PREPARATION = 'docs/protocols/DONUT_RECONSTRUCTED_GEOMETRY_VALIDATION_V1.md'
SOURCES = (PRELIMINARY, PREPARATION, 'benchmarks/donut_reconstructed_geometry_v1.py',
           'experiments/donut_reconstructed_geometry_v1.py', 'experiments/run_donut_reconstructed_geometry_v1.sh',
           'experiments/verify_donut_reconstructed_geometry_v1.py',
           'benchmarks/donut_heterogeneous_v1.py', 'benchmarks/donut_recovery_v1.py',
           'benchmarks/bell_geometry_e11.py', 'benchmarks/generate_bell_e11.py',
           'infrastructure/verified_ray_backend.py', 'infrastructure/stage_sources.py',
           'scientific_sources/manifest-v1.json', 'tests/scientific/test_donut_reconstructed_geometry_v1.py')
MESH = '/home/isavand/nero/final_results/meshes/donut_shape-300000.ply'
LIMITS = dict(primary_rays=196608, visibility_rays=98304, cpu_ray_checks=384, nearest_searches=196608,
              projection_candidates=40000000)
RESOURCES = dict(cpu_threads=4, host_memory_gib=16, pids=128, allocator_fraction=.1,
                 output_bytes=1024**3, watchdog_seconds=900, kill_grace_seconds=30, inner_seconds=870)
TOL = dict(depth=2e-4, position=2e-4, normal=2e-4, unit_normal=2e-6, edge=1e-5,
           radius=.005, normal_degrees=30., side_tie=2e-4, gap=.1, transport_width=2e-10,
           coincident_direction=1e-12, grazing_cosine=.1)
MOUNTS = dict(mesh='/inputs/reconstruction.ply', oracle='/inputs/oracle.npz', cameras='/inputs/cameras.json',
              reference='/inputs/reference', verification='/inputs/verification.json',
              admission='/inputs/admission.json')
REASONS = ('no_oracle_candidate', 'reverse_unresolved', 'invalid_reconstructed_normal', 'normal_mismatch',
           'boundary_side_ambiguity', 'oracle_hidden')


def definition(output):
    p = h.read(ROOT/PRELIMINARY)
    i = p['inputs']
    return dict(schema_version=1, protocol=PROTOCOL, host_output=str(output), image=identity()['image_id'],
                backend=identity(), mesh=i[MESH], oracle=i['oracle-geometry-and-frames.npz'],
                cameras=i['transforms_train.json'], dataset=p['dataset'],
                views={f'view-{c}.npz': i[f'view-{c}.npz'] for c in PANEL},
                sources={name: digest(ROOT/name) for name in SOURCES}, panel=PANEL, splits=SPLITS, size=SIZE,
                tolerances=TOL, limits=LIMITS, resources=RESOURCES,
                historical_geometry_provenance='unresolved', material_optimization_authorized=False)


def freeze(sha):
    path = ROOT/FREEZE
    require(digest(path) == sha, 'Full-raster geometry freeze differs')
    f = h.read(path)
    out = Path(f['host_output'])
    require(out.resolve() == out and out.parent == Path('/home/isavand/paper/evidence')
            and out.name.startswith('donut-geometry-gate2-') and f == definition(out), 'Geometry definition differs')
    return f


def authorization(f, commit, sha):
    return dict(approved=True, protocol=PROTOCOL, commit=commit, freeze_sha256=sha, host_output=f['host_output'],
                backend=f['backend'], selected_gpu='0', maximum_execution_launches=1, limits=LIMITS,
                resources=RESOURCES, material_optimization_authorized=False)


def paths(f, native=False):
    if native:
        return {k: Path(v) for k, v in MOUNTS.items()}
    return dict(mesh=Path(f['mesh']['path']), oracle=Path(f['oracle']['path']), cameras=Path(f['cameras']['path']),
                reference=Path(f['dataset']['generation']['path']).parent,
                verification=Path(f['dataset']['verification']['path']), admission=Path(f['dataset']['admission']['path']))


def admit(f, p):
    for k in ('mesh', 'oracle', 'cameras'):
        h.check_artifact(p[k], f[k])
    files = f['dataset']
    g, v, a = (h.read(h.check_artifact(path, files[k])) for k, path in
               (('generation', p['reference']/'generation.json'), ('verification', p['verification']),
                ('admission', p['admission'])))
    require(g['backend'] == f['backend'] and g['panel'] == PANEL and g['splits'] == a['dataset_splits'] == SPLITS
            and g['field'] == h.FIELD and g['scientific_work_complete'] and g['cleanup_verified']
            and v['status'] == 'verified_pending_researcher_admission' and a['researcher_admitted_dataset'] is True
            and a['training_authorized'] is False and g['outputs'] == v['outputs'] == a['output_identities'],
            'Expected admitted heterogeneous oracle evidence')
    keys = (*h.GEOMETRY_KEYS, 'Q_world', 'axis', 'widths_major_minor', 'anisotropy_gap', 'material_region')
    views = {}
    for c in PANEL:
        name = f'view-{c}.npz'
        item = f['views'][name]
        require(all(item[k] == g['outputs'][name][k] for k in ('bytes', 'sha256')), 'Oracle array identity differs')
        with np.load(h.check_artifact(p['reference']/name, item), allow_pickle=False) as z:
            view = {key: z[key] for key in keys}
        require(view['hit_mask'].shape == (SIZE, SIZE) and view['hit_mask'].dtype == bool
                and np.array_equal(view['pixel_indices'], np.flatnonzero(view['hit_mask']))
                and all(np.isfinite(x).all() for x in view.values()), 'Invalid admitted view arrays')
        views[c] = view
    require(digest(p['oracle']) == INPUTS['oracle-geometry-and-frames.npz'], 'Oracle frame archive differs')
    from benchmarks.donut_recovery_v1 import numeric
    geometry, meta = numeric(p['oracle']), h.read(p['cameras'])
    require(select_panel(meta) == PANEL, 'Camera panel differs')
    return geometry, meta, views


def sampled_pixels(mask):
    groups = (np.flatnonzero(mask), np.flatnonzero(~mask))
    require(all(len(ids) >= 16 for ids in groups), 'Expected foreground and background witnesses')
    return np.concatenate([ids[np.linspace(0, len(ids)-1, 16, dtype=int)] for ids in groups])


class Surface:
    def __init__(self, vertices, faces):
        self.tri = np.asarray(vertices, dtype=float)[faces]
        self.e1 = self.tri[:, 1]-self.tri[:, 0]
        self.e2 = self.tri[:, 2]-self.tri[:, 0]
        cross = np.cross(self.e1, self.e2)
        length = np.linalg.norm(cross, axis=1)
        self.valid = length > 0
        self.normal = np.divide(cross, length[:, None], out=np.zeros_like(cross), where=self.valid[:, None])
        self.lo, self.hi = self.tri.min(axis=1), self.tri.max(axis=1)
        self.half_extent = float(np.max((self.hi-self.lo)/2))
        self.tree = cKDTree((self.lo+self.hi)/2)

    def candidates(self, point):
        radius = TOL['radius']
        ids = np.asarray(self.tree.query_ball_point(point, np.nextafter(radius+self.half_extent, np.inf), p=np.inf), dtype=int)
        return ids[((self.lo[ids] <= point+radius) & (self.hi[ids] >= point-radius)).all(axis=1)]

    def nearest(self, point, charge):
        charge('nearest_searches', 1)
        ids = np.sort(self.candidates(point))
        ids = ids[self.valid[ids]]
        if not len(ids):
            return None
        charge('projection_candidates', len(ids))
        points, bary = closest_points(point, self.tri[ids], self.normal[ids])
        distances = np.linalg.norm(points-point, axis=1)
        best = int(distances.argmin())
        if distances[best] > TOL['radius']:
            return None
        tied = distances <= min(distances[best]+TOL['side_tie'], TOL['radius'])
        return int(ids[best]), points[best], bary[best], float(distances[best]), points[tied]

    def independent_hit(self, origin, direction):
        # The accepted CPU Moller–Trumbore control with all possible normal witnesses retained.
        q = np.cross(direction, self.e2)
        det = np.einsum('ij,ij->i', self.e1, q)
        support = self.valid & (abs(det) > 1e-14)
        inv = np.divide(1., det, out=np.zeros_like(det), where=support)
        s = origin-self.tri[:, 0]
        r = np.cross(s, self.e1)
        u = np.einsum('ij,ij->i', s, q)*inv
        v = np.einsum('ij,j->i', r, direction)*inv
        depth = np.einsum('ij,ij->i', self.e2, r)*inv
        low = np.minimum(np.minimum(u, v), 1-u-v)
        forward = support & (depth > 0) & (depth < 10)
        hits = forward & (low >= 0)
        if not hits.any():
            return None
        index = int(np.argmin(np.where(hits, depth, np.inf)))
        tied = (np.flatnonzero(forward & (low >= -TOL['edge']) & (abs(depth-depth[index]) <= TOL['depth']))
                if low[index] <= TOL['edge'] else np.array([index]))
        return index, float(depth[index]), float(low[index]), tied


def closest_points(point, tri, normal):
    """Exact interior projection or closest edge segment on nondegenerate triangles."""
    a, b, c = tri.transpose(1, 0, 2)
    e, f = b-a, c-a
    q = point-a
    normal_area = np.einsum('ij,ij->i', np.cross(e, f), normal)
    u = np.einsum('ij,ij->i', np.cross(q, f), normal)/normal_area
    v = np.einsum('ij,ij->i', np.cross(e, q), normal)/normal_area
    bary = np.stack((1-u-v, u, v), axis=1)
    inside = bary.min(axis=1) >= 0
    outside, = np.nonzero(~inside)
    if len(outside):
        t = tri[outside]
        edge = np.roll(t, -1, axis=1)-t
        frac = np.clip(np.einsum('nkj,nkj->nk', point-t, edge)/np.square(edge).sum(2), 0, 1)
        dist = np.square(point-t-frac[..., None]*edge).sum(2)
        selected = dist.argmin(axis=1)
        weight = frac[np.arange(len(outside)), selected]
        bary[outside] = 0
        bary[outside, selected] = 1-weight
        bary[outside, (selected+1) % 3] = weight
    return np.einsum('ni,nij->nj', bary, tri), bary


def smooth_normals(geometry, ids, bary):
    normals = geometry['shading_normals_world'][geometry['triangle_loops'][ids]]
    return unit(np.einsum('ni,nij->nj', bary, normals))


def minimal_rotation(a, b):
    cosine = (a*b).sum(axis=-1)
    require((cosine > -1+1e-6).all(), 'Antipodal normal transport is undefined')
    x, y, z = np.cross(a, b).T
    o = np.zeros_like(x)
    k = np.stack((o, -z, y, z, o, -x, -y, x, o), axis=1).reshape(-1, 3, 3)
    return np.eye(3)+k+k@k/(1+cosine[:, None, None])


def rotate_tensor(q, rotation):
    return rotation@q@rotation.transpose(0, 2, 1)


def material_side(points):
    x, y = points[:, :2].T
    return np.cos(2*np.arctan2(y, x)) < 0
