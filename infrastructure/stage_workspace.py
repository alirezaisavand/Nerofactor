"""Filesystem-only views for pristine stage code; no preprocessing arithmetic."""
import json
import importlib.metadata
import os
from pathlib import Path
import sys

from infrastructure.paths import relative_path
from infrastructure.signatures import require,read_report
from infrastructure.stage_sources import digest


def source_records(scene):
    records = []
    for path in sorted(scene.rglob('*')):
        require(not path.is_symlink(), 'Source dataset symlink requires separate review: ' + str(path))
        if path.is_file():
            s = path.stat()
            records.append({'path': path.relative_to(scene).as_posix(), 'bytes': s.st_size,
                            'mtime_ns': s.st_mtime_ns, 'sha256': digest(path)})
    require(records, 'Selected scene has no source files')
    return records


def prepare_view(run, selected, scene_name, science, cfg, identity):
    """Original files are RO links; missing derived files are owned cache outputs.

    Native execution must first verify the mounted dataset is read-only. No link
    is created by following a source symlink. Existing original cache products
    retain their original input semantics and cannot be overwritten through RO mounts.
    """
    relative_path(scene_name)
    require('/' not in scene_name and cfg.get('preprocess_cache') is None, 'Invalid scene/cache selection')
    scene = selected / scene_name
    require(scene.resolve() == scene and scene.is_dir(), 'Missing/aliased scene')
    records = source_records(scene)
    workspace = run.output('cache/stage-workspace'); workspace.mkdir()
    family = workspace / 'data' / selected.name; destination = family / scene_name
    destination.mkdir(parents=True)
    for record in records:
        source = scene / record['path']; target = destination / record['path']
        target.parent.mkdir(exist_ok=True, parents=True)
        target.symlink_to(source)
    # All pristine relative-path writers land here. Assets/configs are immutable.
    (workspace/'assets').symlink_to(science/'assets', target_is_directory=True)
    (workspace/'configs').symlink_to(science/'configs', target_is_directory=True)
    for name, target in [('data/train_vis',run.output('renders/validation')),
                         ('data/meshes',run.output('meshes')),
                         ('nvs_imgs',run.output('renders/nvs')),
                         ('logs',run.output('logs/nvs'))]:
        target.mkdir(exist_ok=True,parents=True)
        (workspace/name).symlink_to(target,target_is_directory=True)
    cache_identity = {'schema_version': 1, 'policy': 'stage-filesystem-cache-v1', 'scientific_source': identity,
                      'original_configuration': cfg, 'dataset_root': str(selected), 'scene': scene_name, 'sources': records,
                      'python':sys.version,'packages':{}}
    for name in ['torch','numpy','scipy','opencv-python','scikit-image','Pillow','imageio','open3d','transforms3d']:
        try:cache_identity['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:cache_identity['packages'][name]=None
    run.output('cache/stage-cache-identity.json').write_text(json.dumps(cache_identity, sort_keys=True, indent=2)+'\n')
    return workspace, family, records


def handoff(path, mesh, resolver, database_name):
    """Admission of an explicit, hashed G-run mesh; no mesh decoding/conversion."""
    require(path and Path(path).is_file(), 'Material requires an explicit mesh-handoff manifest')
    path=Path(path)
    require(path.resolve()==path and any(root in path.parents for root in resolver.input_roots.values()),
            'Mesh handoff must be an unaliased file in an explicit read-only input root')
    value,_ = read_report(path,digest(path))
    from infrastructure.stage_sources import PINS
    require(value.get('schema_version') == 1 and value.get('geometry_algorithm') == 'geometry-G' and
            value.get('scientific_commit') == PINS['geometry'], 'Unattributed/foreign geometry is not a G-to-A handoff')
    for key in ('geometry_run_id','geometry_run_manifest_sha256','coordinates','scale','normals','orientation','material_preprocessing'):
        require(value.get(key) is not None and value[key] != '' and value[key] != 'unknown', 'Incomplete mesh handoff: '+key)
    require(value.get('mesh_sha256') == digest(mesh), 'Selected mesh differs from handoff')
    from infrastructure.paths import InputRef
    ref=value.get('geometry_run_manifest')
    require(type(ref) is dict and set(ref)=={'root','path'},'Explicit geometry run evidence reference required')
    evidence=resolver.input(InputRef(**ref))
    require(digest(evidence)==value['geometry_run_manifest_sha256'],'Geometry run evidence hash mismatch')
    run,_=read_report(evidence,value['geometry_run_manifest_sha256'])
    from infrastructure.stage_sources import baseline,ROOT
    _,geometry_source=baseline('geometry',PINS['geometry'])
    context=run.get('source',{})
    require(context.get('stage')=='geometry' and context.get('algorithm')=='geometry-G' and
            context.get('scientific_commit')==PINS['geometry'] and
            context.get('scientific_files')==geometry_source['files'] and
            context.get('source_manifest_sha256')==digest(ROOT/'scientific_sources/manifest-v1.json'),
            'Producing mesh evidence lacks verified G source context')
    require(value.get('database_name')==database_name and
            run.get('original_configuration',{}).get('database_name')==database_name,
            'Geometry mesh is not from the selected material scene/camera normalization')
    require(run.get('status')=='complete' and run.get('stage')=='geometry' and
            run.get('scientific_commit')==PINS['geometry'] and run.get('run_id')==value['geometry_run_id'] and
            value['mesh_sha256'] in run.get('mesh_outputs',{}).values(),
            'Producing G run must explicitly bind the selected mesh; a training-smoke report alone is insufficient')
    require(value['material_preprocessing'] == 'A-original-read-triangles-and-compute-missing-vertex-normals-no-file-write',
            'Unreviewed mesh conversion/preprocessing')
    return value
