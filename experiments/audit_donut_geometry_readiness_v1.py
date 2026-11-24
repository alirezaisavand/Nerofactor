"""Read-only CPU audit of the previously registered historical Donut mesh."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks.bell_geometry_e11 import intersect, mesh_arrays
from benchmarks.donut_recovery_v1 import camera_rays, inputs, json_new, numeric, PANEL, SPLITS
from benchmarks.donut_heterogeneous_v1 import check_artifact, read
from infrastructure.signatures import require
from infrastructure.stage_sources import ROOT, digest, PINS


MESH = Path('/home/isavand/nero/final_results/meshes/donut_shape-300000.ply')
MESH_SHA = '1206fa06e7b1b724731082dd8cc22185afa62bc2b14600ad54fb6593a758f20c'
REFERENCE = 'docs/protocols/DONUT_HETEROGENEOUS_TRAINING_V1.freeze.json'
SAMPLES_PER_CLASS = 48


def stats(value):
    values = np.asarray(value)
    return dict(median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                max=float(values.max())) if values.size else None


def sample(mask):
    pixels = np.flatnonzero(mask)
    return pixels[np.linspace(0, len(pixels)-1, min(len(pixels), SAMPLES_PER_CLASS), dtype=int)]


def transport_axis(axis, normal, other):
    # Shortest rotation between outward normals, well-defined away from antipodal pairs.
    cosine = float(normal@other)
    if cosine <= -1+1e-10:
        return None
    cross = np.cross(normal, other)
    return axis+np.cross(cross, axis)+np.cross(cross, np.cross(cross, axis))/(1+cosine)


def audit(output):
    f = read(ROOT/REFERENCE)
    base = Path(f['host_reference'])
    files = f['dataset']
    require(output.resolve() == output and not output.exists()
            and output.parent.is_dir() and base not in output.parents and MESH.parent not in output.parents,
            'Expected fresh audit path separate from historical inputs')
    g = read(check_artifact(base/'generation.json', files['generation']))
    v = read(check_artifact(Path(files['verification']['path']), files['verification']))
    admission = read(check_artifact(Path(files['admission']['path']), files['admission']))
    require(admission['researcher_admitted_dataset'] and g['outputs'] == v['outputs'] and f['splits'] == SPLITS,
            'Expected the admitted heterogeneous Donut panel')
    oracle, meta, identities = inputs(Path(f['host_inputs']['oracle']), Path(f['host_inputs']['cameras']))
    vertices, faces = mesh_arrays(check_artifact(MESH, dict(sha256=MESH_SHA, bytes=7062117)))
    a, b, c = (vertices[faces[:, i]].astype(float) for i in range(3))
    area2 = np.linalg.norm(np.cross(b-a, c-a), axis=-1)
    signed_volume = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()/6)
    identities[str(MESH)] = dict(path=str(MESH), bytes=MESH.stat().st_size, sha256=digest(MESH))
    reports = {}
    for camera in PANEL:
        path = base/f'view-{camera}.npz'
        identities[path.name] = dict(g['outputs'][path.name], path=str(path))
        view = numeric(check_artifact(path, g['outputs'][path.name]))
        mask = view['hit_mask'].ravel()
        row = dict(zip(view['pixel_indices'].tolist(), range(len(view['pixel_indices']))))
        origin, direction, _ = camera_rays(meta, camera)
        scores = dict(position=[], abs_depth=[], normal_deg=[], facing=[], frame_Q_relative=[], director_deg=[])
        counts = dict(reference_hit=0, reference_miss=0, both_hit=0, both_miss=0, oracle_only=0,
                      reconstruction_only=0, transport_eligible=0, transport_unresolved=0)
        for pixel in np.concatenate((sample(mask), sample(~mask))):
            expected = bool(mask[pixel])
            counts['reference_hit' if expected else 'reference_miss'] += 1
            hit = intersect(origin[pixel], direction[pixel], vertices, faces)
            if not expected or hit is None:
                key = ('oracle_only' if expected else ('reconstruction_only' if hit else 'both_miss'))
                counts[key] += 1
                continue
            counts['both_hit'] += 1
            i = row[pixel]
            normal = view['normals'][i]
            other = -hit['raw_normal']  # Observe the original A renderer's sign convention.
            scores['position'].append(np.linalg.norm(hit['position']-view['position'][i]))
            scores['abs_depth'].append(abs(hit['depth']-view['depth'][i]))
            scores['normal_deg'].append(np.degrees(np.arccos(np.clip(normal@other, -1, 1))))
            facing = float(other@-direction[pixel])
            scores['facing'].append(facing)
            axis = view['axis'][i]
            transported = transport_axis(axis, normal, other)
            if transported is None or facing <= 0 or view['anisotropy_gap'][i] < .1:
                counts['transport_unresolved'] += 1
                continue
            counts['transport_eligible'] += 1
            scores['director_deg'].append(np.degrees(np.arccos(np.clip(abs(axis@transported), 0, 1))))
            widths = view['widths_major_minor'][i]
            minor_axis = np.cross(other, transported)
            tensor = widths[0]**2*np.outer(transported, transported)+widths[1]**2*np.outer(minor_axis, minor_axis)
            Q = view['Q_world'][i]
            scores['frame_Q_relative'].append(np.linalg.norm(tensor-Q)/np.linalg.norm(Q))
        reports[str(camera)] = dict(counts=counts, minimum_facing=min(scores['facing'], default=None),
                                    nonpositive_facing=int(np.count_nonzero(np.asarray(scores['facing']) <= 0)),
                                    **{k: stats(x) for k, x in scores.items() if k != 'facing'})
    json_new(output, dict(schema_version=1, status='sampled_coordinate_compatibility_only', inputs=identities,
             dataset=files, source_pins=PINS, samples_per_camera_per_reference_class=SAMPLES_PER_CLASS,
             views=reports, vertices=len(vertices), faces=len(faces), exact_degenerate_faces=int((area2 == 0).sum()),
             fitted_transform=False, signed_volume=signed_volume,
             bounds=np.stack((vertices.min(0), vertices.max(0))).tolist(),
             oracle_bounds=np.stack((oracle['vertices_native'].min(0), oracle['vertices_native'].max(0))).tolist(),
             checkpoint_loaded=False, BVH_or_RGB_calls=0, optimizer_updates=0))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    audit(cli.parse_args().output)
