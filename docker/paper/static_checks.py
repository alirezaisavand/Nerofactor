"""Parse definitions and source grammar only; no builds, staging or model execution."""

import ast
from pathlib import Path
import re
import subprocess

import yaml

from validate import DEFINITION_FILES, HERE, definition, require


def main():
    definition()
    ignore = (HERE / '.dockerignore').read_text().splitlines()
    require(all('!definition/' + name in ignore for name in DEFINITION_FILES),
            'Required staged definition file excluded from context')
    require(all('!' + name in ignore for name in ['PREFLIGHT.sha256', 'PREFLIGHT.base-image']),
            'Preflight gate files excluded from context')
    for name in ['compose.yaml', 'compose.geometry-input.yaml']:
        service = yaml.safe_load((HERE / name).read_text())['services']['paper']
        expected_targets = {'/inputs/geometry'} if name == 'compose.geometry-input.yaml' else {'/inputs/datasets', '/inputs/thesis', '/outputs'}
        require({v['target'] for v in service['volumes']} == expected_targets, 'Unexpected mount destination')
        for volume in service['volumes']:
            require(volume['type'] == 'bind' and volume['bind']['create_host_path'] is False,
                    'Mount must use an existing explicit host directory')
            require(volume['read_only'] == (volume['target'] != '/outputs'), 'Wrong mount access mode')
        if name == 'compose.yaml':
            require(service['read_only'] and service['network_mode'] == 'none', 'Missing runtime isolation')
            require(service['cap_drop'] == ['ALL'] and service['security_opt'] == ['no-new-privileges:true'], 'Unexpected capabilities')
            require(not service.get('privileged', False), 'Privileged execution is forbidden')
            require('build' not in service and '${PAPER_IMAGE:?' in service['image'], 'Runtime must not build or reuse an implicit image')
            require('${PAPER_UID:?' in service['user'] and '${PAPER_GID:?' in service['user'], 'Explicit user required')
            devices = service['deploy']['resources']['reservations']['devices']
            require(len(devices) == 1 and devices[0]['device_ids'] == ['${PAPER_GPU:?select a GPU explicitly}'], 'Explicit GPU selection required')
    logical = re.sub(r'\\\n\s*', ' ', (HERE / 'Dockerfile').read_text())
    runs = [line for line in logical.splitlines() if line.startswith('RUN ')]
    require('ARG PAPER_PREFLIGHT_SHA256\n' in logical and
            'sha256sum --strict --check PREFLIGHT.sha256' in runs[0], 'Missing mandatory pre-install gate')
    require(not re.search(r'\b(dpkg|pip|apt-get|python3)\b', runs[0]),
            'Installer or base Python used before complete context verification')
    require('dpkg --unpack' in runs[1], 'Debian install must follow preflight gate')
    shell_count = 0
    for line in logical.splitlines():
        if line.startswith('RUN '):
            prefix = 'RUN --network=none '
            require(line.startswith(prefix), 'Networked build step found')
            subprocess.run(['bash', '-n'], input=line[len(prefix):], text=True, check=True)
            shell_count += 1
    repo = HERE.parents[1]
    paths = [p for folder in ['infrastructure', 'dataset', 'network', 'train', 'utils', 'raytracing']
             for p in (repo / folder).glob('*.py')]
    paths += [repo / 'run_experiment.py', repo / 'run_material_experiment.py', *HERE.glob('*.py')]
    paths += [repo / 'colmap/read_write_model.py', repo / 'colmap/plyfile.py']
    for path in paths:
        ast.parse(path.read_text(), filename=str(path), feature_version=8)
    print('PASS: Compose mount/non-root/GPU policy and {} Docker RUN shell parses'.format(shell_count))
    print('PASS: complete staged definition allowlist and pre-install validation boundary')
    print('PASS: Python 3.8 grammar for {} source/definition files'.format(len(paths)))
    print('NOT TESTED: Python 3.8 runtime, Docker/Compose engine semantics, package resolution, builds, ABI or GPU behavior')


if __name__ == '__main__':
    main()
