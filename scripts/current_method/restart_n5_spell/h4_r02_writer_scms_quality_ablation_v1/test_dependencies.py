"""Real frozen-identity and repository guards, using temporary offline files."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ..h4_r02_basic_miner_v3_off_paired_v1 import dependencies as shared_dependencies
from . import dependencies
from .runtime import GuardrailStop


class FrozenDependencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.run = self.base / 'run'
        self.code = self.base / 'repository/new_code'
        self.code.mkdir(parents=True)
        (self.run / 'audit').mkdir(parents=True)
        self.paths = {
            'code': self.code / 'stage.py',
            'shared': self.base / 'repository/shared.py',
            'inputs': self.run / 'input.json',
            'sources': self.base / 'historical_source.json',
            'config': self.base / 'fake_nonsecret_config.json',
            'preflight': self.run / 'audit/preflight_report.json',
        }
        for key, path in self.paths.items():
            path.write_text('{"status":"PASS"}' if key == 'preflight' else 'VALUE = 1\n')
        self.manifest_path = self.run / 'audit/frozen_manifest.json'
        self.environment = {'python': 'offline-test', 'libraries': {'test': '1.0'},
                            'bytecode_writes_disabled': True}
        self.manifest = {
            'version': dependencies.VERSION,
            'schedule': copy.deepcopy(dependencies.SCHEDULE),
            'code': [dependencies.record(self.paths['code']), dependencies.record(self.paths['shared'])],
            'inputs': [dependencies.record(self.paths['inputs'])],
            'sources': [dependencies.record(self.paths['sources'])],
            'config': dependencies.record(self.paths['config']),
            'preflight': dependencies.record(self.paths['preflight']),
            'environment': copy.deepcopy(self.environment),
        }
        self.save_manifest()
        # Real coverage functions see a small temporary repository; no network,
        # historical file, credential, client, ledger or experiment is used.
        for context in (
            dependencies.forbid_network(),
            patch.object(dependencies, 'CODE', self.code),
            patch.object(dependencies, 'CONFIG', self.paths['config']),
            patch.object(shared_dependencies, 'REPO', self.base / 'repository'),
            patch.object(dependencies, 'environment_identity', return_value=self.environment),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)
        module = types.ModuleType('_writer_frozen_test_known')
        module.__file__ = str(self.paths['shared'])
        context = patch.dict(sys.modules, {module.__name__: module})
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def verify(self, **kwargs):
        return dependencies.verify_frozen(self.run, **kwargs)

    def test_full_frozen_identity_passes_real_file_and_loaded_module_checks(self):
        result = self.verify()
        self.assertEqual(result, self.manifest)
        self.assertIn(self.paths['shared'], dependencies.loaded_repository_files())
        self.assertEqual(self.verify(config_path=self.paths['config']), self.manifest)

    def test_missing_manifest_blocks(self):
        self.manifest_path.unlink()
        with self.assertRaisesRegex(GuardrailStop, 'PREPARATION_NOT_FROZEN'):
            self.verify()

    def test_version_or_schedule_drift_blocks(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (('version', 'wrong-experiment'), ('schedule', [])):
            with self.subTest(key=key):
                self.manifest = copy.deepcopy(original)
                self.manifest[key] = value
                self.save_manifest()
                with self.assertRaisesRegex(GuardrailStop, 'FROZEN_SCOPE_DRIFT'):
                    self.verify()

    def test_config_path_identity_blocks_even_equal_bytes(self):
        alternative = self.base / 'other_config.json'
        alternative.write_bytes(self.paths['config'].read_bytes())
        with self.assertRaisesRegex(GuardrailStop, 'CONFIG_PATH_DRIFT'):
            self.verify(config_path=alternative)

    def test_each_code_input_source_config_preflight_sha_drift_blocks(self):
        for category, path in self.paths.items():
            original = path.read_bytes()
            # Equal length guarantees the SHA comparison itself is exercised.
            changed = original[:-1] + (b'X' if original[-1:] != b'X' else b'Y')
            with self.subTest(category=category):
                path.write_bytes(changed)
                with self.assertRaisesRegex(GuardrailStop, 'FROZEN_IDENTITY_DRIFT:'):
                    self.verify()
                path.write_bytes(original)
        self.assertEqual(self.verify(), self.manifest)

    def test_each_identity_category_missing_or_size_changed_blocks(self):
        for category, path in self.paths.items():
            original = path.read_bytes()
            with self.subTest(category=category, mutation='size'):
                path.write_bytes(original + b' ')
                with self.assertRaisesRegex(GuardrailStop, 'FROZEN_IDENTITY_DRIFT:'):
                    self.verify()
                path.write_bytes(original)
            with self.subTest(category=category, mutation='missing'):
                path.unlink()
                with self.assertRaisesRegex(GuardrailStop, 'FROZEN_IDENTITY_DRIFT:'):
                    self.verify()
                path.write_bytes(original)

    def test_new_source_or_nonpython_inventory_file_blocks(self):
        for filename in ('unfrozen.py', 'unfrozen_prompt.txt'):
            path = self.code / filename
            with self.subTest(filename=filename):
                path.write_text('new dependency')
                with self.assertRaisesRegex(GuardrailStop, 'NEW_CODE_INVENTORY_DRIFT'):
                    self.verify()
                path.unlink()
        cache = self.code / '__pycache__/irrelevant.pyc'
        cache.parent.mkdir()
        cache.write_bytes(b'not-loaded-bytecode')
        self.assertEqual(self.verify(), self.manifest)

    def test_environment_drift_blocks(self):
        self.environment['libraries']['test'] = '2.0'
        with self.assertRaisesRegex(GuardrailStop, 'ENVIRONMENT_DRIFT'):
            self.verify()

    def test_each_frozen_input_inventory_rejects_added_files(self):
        for folder in ('inputs', 'policy_snapshot', 'audit/service_evidence'):
            with self.subTest(folder=folder):
                directory = self.run / folder / 'nested'
                directory.mkdir(parents=True)
                known = directory / 'frozen.json'
                known.write_text('{"frozen":true}')
                self.manifest['inputs'].append(dependencies.record(known))
                self.save_manifest()
                self.assertEqual(self.verify(), self.manifest)
                added = directory / 'added_tokenizer_config.json'
                added.write_text('{"unexpected":true}')
                with self.assertRaisesRegex(GuardrailStop, 'FROZEN_INPUT_INVENTORY_DRIFT:' + folder):
                    self.verify()
                added.unlink()
                self.assertEqual(self.verify(), self.manifest)

    def test_preflight_must_pass_even_if_hash_is_frozen(self):
        self.paths['preflight'].write_text('{"status":"FAIL"}')
        self.manifest['preflight'] = dependencies.record(self.paths['preflight'])
        self.save_manifest()
        with self.assertRaisesRegex(GuardrailStop, 'PREFLIGHT_NOT_PASS'):
            self.verify()

    def test_real_loaded_coverage_rejects_unfrozen_repository_module(self):
        unknown_path = self.base / 'repository/unfrozen_import.py'
        unknown_path.write_text('VALUE = 1\n')
        module = types.ModuleType('_writer_frozen_test_unknown')
        module.__file__ = str(unknown_path)
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(GuardrailStop, 'unfrozen_loaded_repository_dependency:'):
                self.verify()

    def test_real_profile_guard_allows_frozen_execution_and_restores_profile(self):
        namespace = {}
        exec(compile('def execute():\n    return 7\n', str(self.paths['code']), 'exec'), namespace)
        previous = sys.getprofile()
        with dependencies.repository_execution_guard(self.manifest['code']):
            self.assertEqual(namespace['execute'](), 7)
        self.assertIs(sys.getprofile(), previous)

    def test_real_profile_guard_stops_unfrozen_code_before_body_and_restores(self):
        unknown = self.base / 'repository/unfrozen_dynamic.py'
        namespace = {'visited': []}
        exec(compile('def execute():\n    visited.append("must-not-run")\n', str(unknown), 'exec'), namespace)
        previous = sys.getprofile()
        with self.assertRaisesRegex(GuardrailStop, 'unfrozen_executed_repository_dependency:'):
            with dependencies.repository_execution_guard(self.manifest['code']):
                namespace['execute']()
        self.assertEqual(namespace['visited'], [])
        self.assertIs(sys.getprofile(), previous)


if __name__ == '__main__':
    unittest.main()
