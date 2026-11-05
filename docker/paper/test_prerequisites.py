"""CPU-only source/export and metadata fixtures; no original artifacts or installs."""

import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import collect_runtime
import stage_sources
from validate import HERE, HEX, digest, inventory, read_json


class SourceStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'original'
        self.repo.mkdir()
        self.git('init', '--quiet')
        for name, data in {'setup.py': b'# never execute setup\n', 'README.md': b'fixture\n',
                           'LICENSE.txt': b'fixture licensing retained byte-for-byte\n',
                           'nvdiffrast/__init__.py': b'# fixture\n',
                           'data/original.png': b'never export\n'}.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.commit = self.commit_all()
        self.reference = read_json(HERE / 'reference.json')
        self.reference['extensions']['nvdiffrast'].update(
            commit=self.commit, setup_py_sha256=digest(self.repo / 'setup.py'))
        self.destination = self.root / 'export'
        self.report = self.root / 'provenance.json'
        self.protected = self.root / 'datasets'
        self.protected.mkdir()

    def git(self, *args):
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(self.repo),
                                        *args], stderr=subprocess.DEVNULL).decode().strip()

    def commit_all(self):
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 '-c', 'core.hooksPath=/dev/null', 'commit', '--quiet', '-m', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def stage(self):
        with patch.object(stage_sources, 'read_json', return_value=self.reference):
            return stage_sources.stage('nvdiffrast', self.repo, self.commit,
                                       self.destination, self.report, [self.protected])

    def test_verified_export_retains_licence_hashes_and_original_bytes(self):
        before = inventory(self.repo / 'nvdiffrast')
        (self.repo / 'untracked-secret.env').write_text('never export')
        record = self.stage()
        self.assertEqual(inventory(self.destination), record['files'])
        self.assertEqual((self.repo / 'LICENSE.txt').read_bytes(), (self.destination / 'LICENSE.txt').read_bytes())
        self.assertEqual(inventory(self.repo / 'nvdiffrast'), before)
        self.assertFalse((self.destination / 'data').exists())
        self.assertFalse((self.destination / 'untracked-secret.env').exists())
        self.assertEqual(record['commit'], self.commit)
        self.assertEqual(record['git_tree'], self.git('rev-parse', 'HEAD^{tree}'))

    def test_existing_destination_never_overwritten(self):
        self.stage()
        before = inventory(self.destination)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.stage()
        self.assertEqual(inventory(self.destination), before)

    def test_protected_and_source_destinations_rejected(self):
        for parent in [self.repo, self.protected]:
            self.destination = parent / 'new'
            with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, 'protected'):
                self.stage()
            self.assertFalse(self.destination.exists())

    def test_report_must_be_separate_and_fresh(self):
        self.report = self.destination / 'provenance.json'
        with self.assertRaises(ValueError):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_wrong_commit_and_setup_hash_rejected(self):
        self.reference['extensions']['nvdiffrast']['commit'] = 'a' * 40
        with self.assertRaisesRegex(ValueError, 'approved pin'):
            self.stage()
        self.reference['extensions']['nvdiffrast']['commit'] = self.commit
        self.reference['extensions']['nvdiffrast']['setup_py_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'setup.py SHA-256'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_dirty_checkout_rejected(self):
        (self.repo / 'README.md').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Tracked source'):
            self.stage()

    def test_selected_symlink_rejected_before_writes(self):
        (self.repo / 'nvdiffrast/link.py').symlink_to(self.repo / 'setup.py')
        self.commit = self.commit_all()
        self.reference['extensions']['nvdiffrast']['commit'] = self.commit
        with self.assertRaisesRegex(ValueError, 'Symlink/special'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_bundle_roundtrip_verifies_git_objects_and_hash(self):
        bundle = self.root / 'source.bundle'
        content = subprocess.check_output(['git', '--no-optional-locks', '-C', str(self.repo),
                                           'bundle', 'create', '-', 'HEAD'], stderr=subprocess.DEVNULL)
        bundle.write_bytes(content)
        self.assertFalse((self.repo / '-').exists())
        with stage_sources.bundle_repository(bundle, digest(bundle)) as (repo, head):
            self.assertEqual(head, self.commit)
            with patch.object(stage_sources, 'read_json', return_value=self.reference):
                record = stage_sources.stage('nvdiffrast', repo, head, self.destination,
                                             self.report, [self.protected])
        self.assertEqual(record['files'], inventory(self.destination))
        with self.assertRaisesRegex(ValueError, 'Bundle SHA-256'):
            with stage_sources.bundle_repository(bundle, 'c' * 64):
                self.fail('Wrong bundle digest accepted')

    def test_raytracing_requires_reviewed_paths_and_retains_root_licence(self):
        self.reference['extensions']['raytracing'].update(self.reference['extensions']['nvdiffrast'])
        with patch.object(stage_sources, 'read_json', return_value=self.reference):
            with self.assertRaisesRegex(ValueError, 'allowlist'):
                stage_sources.stage('raytracing', self.repo, self.commit, self.destination,
                                    self.report, [self.protected])
            record = stage_sources.stage('raytracing', self.repo, self.commit, self.destination,
                                         self.report, [self.protected], ['setup.py', 'nvdiffrast'])
        self.assertIn('LICENSE.txt', record['files'])

    def test_unsafe_source_selection_rejected(self):
        for path in ['data/original.png', '../setup.py', 'weights.pth', '.env', 'keys/secret.pem']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                stage_sources.allowed(path, 'raytracing')

    def test_paper_snapshot_uses_pinned_commit_and_asset_allowlist(self):
        for path in stage_sources.PAPER_PATHS:
            target = self.repo / path
            if target.suffix:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('# fixture')
            else:
                target.mkdir(exist_ok=True)
                (target / 'fixture.py').write_text('# fixture')
        self.commit = self.commit_all()
        record = stage_sources.stage('paper', self.repo, self.commit, self.destination,
                                     self.report, [self.protected])
        self.assertEqual(record['commit'], self.commit)
        self.assertIn('assets/bsdf_256_256.bin', record['files'])
        self.assertNotIn('data/original.png', record['files'])


class DependencyMetadataTests(unittest.TestCase):
    def test_recovered_runtime_manifest_consistency_and_artifact_boundary(self):
        record = read_json(HERE / 'runtime.reference.json')
        reference = read_json(HERE / 'reference.json')
        packages = record['python_packages']
        self.assertFalse(record['reviewed'])
        self.assertFalse(record['build_ready'])
        self.assertEqual(len(packages), 102)
        self.assertEqual(record['provenance']['sha256'], reference['recovered_inventory_manifest']['input_sha256'])
        self.assertTrue(HEX.fullmatch(record['provenance']['sha256']))
        self.assertEqual(record['runtime']['python'], reference['runtime']['python'])
        for name, version in reference['verified_live_packages'].items():
            self.assertEqual(packages[collect_runtime.normalize(name)]['version'], version)
        for name, source in reference['extensions'].items():
            self.assertEqual(packages[name]['version'], source['version'])
        edges, declared = 0, 0
        for name, package in packages.items():
            self.assertEqual(name, collect_runtime.normalize(name))
            self.assertTrue(Version(package['version']))
            if package['requires_python'] is not None:
                declared += 1
                self.assertIn(record['runtime']['python'], SpecifierSet(package['requires_python']))
            for text in package['requires']:
                dependency = Requirement(text)
                self.assertIsNone(dependency.url)
                self.assertFalse(dependency.extras)
                self.assertIsNone(dependency.marker)  # collector-selected active edges
                self.assertIn(dependency.name, packages)
                self.assertIn(packages[dependency.name]['version'], dependency.specifier)
                edges += 1
        def reachable(roots):
            result, pending = set(), list(roots)
            while pending:
                name = collect_runtime.normalize(pending.pop())
                if name not in result:
                    result.add(name)
                    pending.extend(Requirement(text).name for text in packages[name]['requires'])
            return result
        closure = record['closure']
        self.assertEqual(set(packages), reachable(collect_runtime.RUNTIME_ROOTS + collect_runtime.BUILD_ROOTS))
        self.assertEqual(edges, closure['active_requirement_edges'])
        self.assertEqual(edges, 142)
        self.assertEqual(declared, closure['declared_python_constraints_satisfied'])
        self.assertEqual(declared, 82)
        self.assertEqual(sorted(n for n, p in packages.items() if p['requires_python'] is None),
                         closure['missing_requires_python'])
        self.assertEqual(len(reachable(collect_runtime.RUNTIME_ROOTS)), closure['runtime_closure_count'])
        self.assertEqual(len(reachable(['open3d'])), closure['open3d_closure_count_including_root'])
        self.assertEqual(len(reachable(['raytracing'])), closure['raytracing_closure_count_including_root'])
        self.assertEqual(len(record['debian_role_versions']), 21)
        self.assertTrue(all(record['debian_role_versions'].values()))
        self.assertEqual(len(record['unresolved_debian_roles']), 6)
        self.assertNotIn('ninja-build', record['debian_role_versions'])
        self.assertEqual(set(record['source_built_distributions']), {'raytracing', 'nvdiffrast'})
        self.assertEqual(closure['prebuilt_wheel_distribution_count_excluding_native_sources'], 100)
        self.assertEqual(closure['verified_downloaded_wheel_artifacts'], 0)
        for candidate in record['publisher_candidates']['packages']:
            self.assertEqual(candidate['version'], packages[candidate['name']]['version'])
            self.assertTrue(HEX.fullmatch(candidate['publisher_sha256']))
        self.assertIsNone(reference['image']['base_digest'])
        self.assertFalse(record['cuda_base_candidate']['registry_verified_by_codex'])
        self.assertFalse(record['cuda_base_candidate']['historical_identity_established'])

    def test_roots_cover_manifest_runtime_and_build_requirements(self):
        reference = read_json(HERE / 'reference.json')
        selected = {collect_runtime.normalize(n) for n in collect_runtime.RUNTIME_ROOTS + collect_runtime.BUILD_ROOTS}
        required = set(reference['verified_live_packages']) | set(reference['unresolved_python_packages'][:-1]) | set(reference['extensions'])
        self.assertEqual(selected, {collect_runtime.normalize(n) for n in required})
        for candidate in read_json(HERE / 'artifact_candidates.json')['candidates']:
            self.assertEqual(candidate['version'], reference['verified_live_packages'][candidate['package']])
            sha = candidate['publisher_sha256']
            self.assertTrue(sha is None or HEX.fullmatch(sha))

    def test_closure_honours_markers_extras_and_reports_missing_or_incompatible_versions(self):
        distributions = {
            'root': SimpleNamespace(version='1', metadata={}, requires=[
                'child[feature]>=2', 'not-on-linux; sys_platform == "win32"',
                'private @ https://example.invalid/private?token=secret']),
            'child': SimpleNamespace(version='1', metadata={}, requires=['missing; extra == "feature"']),
            'private': SimpleNamespace(version='1', metadata={}, requires=[]),
        }
        def installed(name):
            if name not in distributions:
                raise md.PackageNotFoundError(name)
            return distributions[name]
        packages, issues = collect_runtime.closure(['root'], installed, Requirement,
                                                 dict(default_environment(), sys_platform='linux'))
        self.assertEqual(set(packages), {'root', 'child', 'private'})
        self.assertEqual(packages['child']['requested_extras'], ['feature'])
        text = json.dumps(issues)
        self.assertIn('missing distribution', text)
        self.assertIn('violates metadata', text)
        self.assertIn('direct URL omitted', text)
        self.assertNotIn('secret', text)
        self.assertNotIn('example.invalid', text)


if __name__ == '__main__':
    unittest.main()
