"""Offline CLI delivery guards; synthetic state stays in temporary directories."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import delivery, prepare, reporting, run, runtime


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CliPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / 'synthetic-config.json'
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(run, 'DEFAULT_RUN', self.root))
        self.stack.enter_context(patch.object(run, 'CONFIG', self.config))
        self.stack.enter_context(patch.object(prepare, 'DEFAULT_RUN', self.root))
        self.stack.enter_context(patch.object(prepare, 'CONFIG', self.config))
        self.stack.enter_context(patch('socket.socket', side_effect=AssertionError('network forbidden')))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def existing_status(self):
        (self.root / 'reports').mkdir(exist_ok=True)
        (self.root / 'RUN_STATUS.md').write_text('SYNTHETIC EXISTING LIVE STATE: primary_calls=12\n')
        (self.root / 'reports/preparation_report.md').write_text('SYNTHETIC EXISTING REPORT; preserve bytes.\n')
        return {p: p.read_bytes() for p in (self.root / 'RUN_STATUS.md', self.root / 'reports/preparation_report.md')}

    def test_frozen_prepare_verifies_without_rewriting_existing_delivery(self):
        before = self.existing_status()
        write(self.root / 'audit/frozen_manifest.json', {'fixture': True})
        write(self.root / 'audit/preflight_report.json', {'status': 'PASS'})
        with patch.object(prepare, 'verify_frozen') as verify, patch.object(delivery, 'write_delivery') as deliver, patch.object(runtime, 'execute_live', side_effect=AssertionError('no live')) as live, redirect_stdout(StringIO()) as output:
            code = run.main(['prepare'])
        self.assertEqual(code, 0)
        verify.assert_called_once_with(self.root.resolve(), self.config)
        deliver.assert_not_called()
        live.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result['mode'], 'FROZEN_IDENTITY_VERIFIED')
        self.assertFalse(result['artifacts_rewritten'])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_default_mode_is_prepare_and_never_calls_live(self):
        with patch.object(prepare, 'prepare', return_value={'status': 'PASS'}) as prepare_call, patch.object(delivery, 'write_delivery') as deliver, patch.object(runtime, 'execute_live', side_effect=AssertionError('no live')) as live, redirect_stdout(StringIO()):
            self.assertEqual(run.main([]), 0)
        prepare_call.assert_called_once_with(self.root)
        deliver.assert_called_once_with(self.root)
        live.assert_not_called()
        self.assertFalse(any('authorization' in p.name.lower() for p in self.root.rglob('*')))

    def test_live_without_explicit_flag_is_rejected_before_execution(self):
        with patch.object(runtime, 'execute_live', side_effect=AssertionError('no live')) as live, redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            run.main(['live'])
        self.assertEqual(raised.exception.code, 2)
        live.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def seed_delivery_reads(self):
        for name in ('preflight_report', 'test_report', 'token_budget_summary', 'token_identity', 'dependency_coverage_report', 'source_preservation_report'):
            write(self.root / 'audit' / (name + '.json'), {'status': 'PASS'} if name == 'preflight_report' else {})

    def test_delivery_with_request_ledger_never_claims_zero_calls(self):
        before = self.existing_status()
        self.seed_delivery_reads()
        write(self.root / 'arms/WITH_CURRENT_INTENT/qa/shards/shard-00/roots/synthetic-root/ledger/requests/synthetic.json', {'fixture': True})
        with patch.object(reporting, 'report', return_value={'status': 'PREPARED_NOT_RUN', 'errors': []}), self.assertRaisesRegex(RuntimeError, 'ZERO_REAL_STATE_AND_CLEAN_REPORT'):
            delivery.write_delivery(self.root)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertFalse((self.root / 'EXECUTE_AFTER_AUTHORIZATION.md').exists())
        self.assertFalse((self.root / 'audit/delivery_check.json').exists())

    def test_delivery_rejects_dirty_report_without_rewriting_status(self):
        before = self.existing_status()
        self.seed_delivery_reads()
        with patch.object(reporting, 'report', return_value={'status': 'BLOCKED', 'errors': ['synthetic report error']}), self.assertRaisesRegex(RuntimeError, 'ZERO_REAL_STATE_AND_CLEAN_REPORT'):
            delivery.write_delivery(self.root)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
