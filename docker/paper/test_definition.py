"""Static regression fixtures only: no real sources, packages, Docker or models."""

import copy
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import json

import preflight
import validate


HERE = validate.HERE
BASE = 'nvidia/cuda:11.1.1-devel-ubuntu20.04@sha256:' + 'a' * 64


class DefinitionTests(unittest.TestCase):
    def setUp(self):
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in validate.DEFINITION_FILES:
            self.put('definition/' + name, (HERE / name).read_bytes())
        for name in ['Dockerfile', '.dockerignore']:
            self.put(name, (HERE / name).read_bytes())
        self.put('workspace/README.md', b'Synthetic paper fixture\n')
        self.reference = copy.deepcopy(validate.read_json(HERE / 'reference.json'))
        # Synthetic source identities; never export or modify historical checkouts.
        for name, source in self.reference['extensions'].items():
            self.put('inputs/sources/' + name + '/setup.py', b'# synthetic, never executed\n')
            source['setup_py_sha256'] = validate.digest(self.root / 'inputs/sources' / name / 'setup.py')
        packages = dict(self.reference['verified_live_packages'])
        packages.update({name: '0.0.fixture' for name in self.reference['unresolved_python_packages'][:-1]})
        self.put('inputs/wheels/fixture.whl', b'Synthetic wheel bytes; never installed\n')
        sha = validate.digest(self.root / 'inputs/wheels/fixture.whl')
        self.put('inputs/python-complete.txt', ''.join(
            '{}=={} --hash=sha256:{}\n'.format(k, v, sha) for k, v in packages.items()).encode())
        self.put('inputs/apt/fixture.deb', b'Synthetic Debian bytes; never installed\n')
        sha = validate.digest(self.root / 'inputs/apt/fixture.deb')
        self.put('inputs/apt/SHA256SUMS', (sha + '  fixture.deb\n').encode())
        resolved = {
            'schema_version': 1, 'reviewed': True, 'base_image': BASE,
            'workspace_commit': 'b' * 40,
            'workspace_files': validate.inventory(self.root / 'workspace'),
            'files': validate.inventory(self.root / 'inputs'),
            'python_packages': packages, 'historical_extras_decision': 'omit-unused',
            'system_packages': {name: 'fixture' for name in [
                'python3', 'python3-dev', 'python3-pip', 'build-essential', 'libeigen3-dev']},
            'toolchain': {'gcc': 'fixture', 'g++': 'fixture', 'eigen': 'fixture',
                          'cmake': 'fixture', 'nvcc': '11.1.105', 'cxx11_abi': 0,
                          'cxx_flags': ['fixture'], 'nvcc_flags': ['fixture']},
            'sources': {k: {'commit': v['commit']} for k, v in self.reference['extensions'].items()},
        }
        self.put('inputs/resolved.json', json.dumps(resolved).encode())

    def put(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def prepare(self):
        with patch.object(preflight, 'definition', return_value=self.reference):
            return preflight.prepare(self.root, BASE)

    def gate(self, sha, base=BASE):
        logical = re.sub(r'\\\n\s*', ' ', (HERE / 'Dockerfile').read_text())
        runs = [line for line in logical.splitlines() if line.startswith('RUN ')]
        command = runs[0][len('RUN --network=none '):]
        self.assertNotRegex(command, r'\b(dpkg|pip|apt-get|python3)\b')
        self.assertIn('sha256sum --strict --check PREFLIGHT.sha256', command)
        self.assertIn('dpkg --unpack', runs[1])
        command = command.replace('cd /opt/paper-context', 'cd ' + shlex.quote(str(self.root)))
        return subprocess.run(['bash', '-euo', 'pipefail', '-c', command],
                              env={'PATH': os.defpath, 'PAPER_BASE_IMAGE': base,
                                   'PAPER_PREFLIGHT_SHA256': sha},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_complete_definition_inventory(self):
        rules = (HERE / '.dockerignore').read_text().splitlines()
        for name in validate.DEFINITION_FILES:
            self.assertIn('!definition/' + name, rules)
        for name in [preflight.RECEIPT, preflight.BASE_FILE]:
            self.assertIn('!' + name, rules)
        # A minimal staged definition must run the same definition checks.
        with patch.object(validate, 'HERE', self.root / 'definition'):
            validate.definition()
            for name in validate.DEFINITION_FILES:
                path = self.root / 'definition' / name
                content = path.read_bytes()
                path.unlink()
                with self.subTest(missing=name), self.assertRaisesRegex(ValueError, 'Missing required'):
                    validate.definition()
                path.write_bytes(content)

    def test_valid_context_passes_shell_gate_without_python(self):
        sha = self.prepare()
        self.assertEqual(self.gate(sha).returncode, 0)

    def test_unreviewed_resolution_rejected_before_receipt(self):
        path = self.root / 'inputs/resolved.json'
        resolved = json.loads(path.read_text())
        resolved['reviewed'] = False
        path.write_text(json.dumps(resolved))
        with self.assertRaisesRegex(ValueError, 'reviewed resolution'):
            self.prepare()
        self.assertFalse((self.root / preflight.RECEIPT).exists())

    def test_invalid_artifact_rejected_before_receipt(self):
        self.put('inputs/apt/fixture.deb', b'changed')
        with self.assertRaisesRegex(ValueError, 'complete hash manifest'):
            self.prepare()
        self.assertFalse((self.root / preflight.RECEIPT).exists())

    def test_missing_definition_rejected_before_receipt(self):
        (self.root / 'definition/requirements.reference.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'definition inventory'):
            self.prepare()

    def test_changed_definition_rejected_before_receipt(self):
        self.put('definition/validate.py', b'# changed')
        with self.assertRaisesRegex(ValueError, 'definition inventory'):
            self.prepare()

    def test_extra_context_file_rejected(self):
        self.put('unreviewed.txt', b'extra')
        with self.assertRaisesRegex(ValueError, 'extra files'):
            self.prepare()

    def test_existing_receipt_not_overwritten(self):
        sha = self.prepare()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(validate.digest(self.root / preflight.RECEIPT), sha)

    def test_missing_or_wrong_external_digest_rejected(self):
        self.prepare()
        for sha in ['', 'c' * 64]:
            with self.subTest(sha=sha):
                self.assertNotEqual(self.gate(sha).returncode, 0)

    def test_changed_manifest_rejected(self):
        sha = self.prepare()
        with (self.root / preflight.RECEIPT).open('a') as stream:
            stream.write('changed\n')
        self.assertNotEqual(self.gate(sha).returncode, 0)

    def test_changed_copied_bytes_rejected(self):
        sha = self.prepare()
        for name in ['inputs/resolved.json', 'inputs/apt/fixture.deb',
                     'definition/validate.py', 'workspace/README.md', '.dockerignore']:
            path = self.root / name
            content = path.read_bytes()
            path.write_bytes(content + b'changed')
            with self.subTest(changed=name):
                self.assertNotEqual(self.gate(sha).returncode, 0)
            path.write_bytes(content)

    def test_missing_or_added_copied_file_rejected(self):
        sha = self.prepare()
        self.put('extra.txt', b'extra')
        self.assertNotEqual(self.gate(sha).returncode, 0)
        (self.root / 'extra.txt').unlink()
        (self.root / 'definition/requirements.reference.txt').unlink()
        self.assertNotEqual(self.gate(sha).returncode, 0)

    def test_changed_base_argument_rejected(self):
        sha = self.prepare()
        self.assertNotEqual(self.gate(sha, BASE[:-1] + 'b').returncode, 0)

    def test_redirected_copied_file_rejected(self):
        sha = self.prepare()
        path = self.root / 'workspace/README.md'
        path.unlink()
        path.symlink_to(self.root / 'Dockerfile')
        self.assertNotEqual(self.gate(sha).returncode, 0)


if __name__ == '__main__':
    unittest.main()
