"""Offline recovery guards; network disabled throughout."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import recover_collection as r

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.blocker = patch('socket.socket', side_effect=AssertionError('No network in recovery tests'))
        self.blocker.start(); self.addCleanup(self.blocker.stop)
        self.plan = r.c.verify_plan()

    def test_exact_frozen_interrupted_request_replays_without_http(self):
        chunk, run, folder = r.interrupted_identity(self.plan)
        self.assertEqual(chunk['chunk_name'], 'direct_c01')
        self.assertEqual(len(list((run / 'requests').glob('*/HTTP_ATTEMPT.json'))), 1)
        self.assertFalse(any((folder / name).exists() for name in r.MISSING))
        gen, _ = r.c.modules()
        cost = gen.request_costs(run)['question_generator']
        self.assertEqual(cost['http_attempts'], 1)
        self.assertEqual(cost['unknown_request_count'], 1)
        self.assertIsNone(cost['input_tokens'])
        self.assertIsNone(cost['output_tokens'])

    def test_wrong_first_chunk_is_rejected(self):
        changed = deepcopy(self.plan); changed['chunks'][0]['arm'] = 'staged'
        with self.assertRaisesRegex(ValueError, 'first_chunk_changed'):
            r.interrupted_identity(changed)

    def test_remaining_budget_and_order_are_unchanged(self):
        remaining = self.plan['chunks'][1:]
        self.assertEqual([c['chunk_name'] for c in remaining],
            ['staged_c01', 'staged_c02', 'direct_c02', 'direct_c03', 'staged_c03'])
        self.assertEqual(1 + sum(c['http_max'] for c in remaining), 14)
        self.assertLessEqual(14, self.plan['remote_http_max'])

    def test_already_started_recovery_cannot_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            here = Path(temp); (here / 'generation.STARTED.json').write_text('{}')
            with patch.object(r, 'HERE', here), patch.object(r, 'verify', return_value=(self.plan, {})), \
                 patch.object(r, 'live_processes', return_value=[]), patch.object(r.c, 'command') as command:
                with self.assertRaisesRegex(ValueError, 'already_started_no_replay'): r.generation()
                command.assert_not_called()

    def test_live_process_blocks_remote_resume(self):
        with patch.object(r, 'verify', return_value=(self.plan, {})), \
             patch.object(r, 'live_processes', return_value=[{'pid': 123}]), patch.object(r.c, 'command') as command:
            with self.assertRaisesRegex(ValueError, 'live_generation_process'): r.generation()
            command.assert_not_called()

    def test_sha_guard_detects_existing_file_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'artifact.json'; path.write_text('{"truth":"original"}')
            bound = r.c.binding(path); path.write_text('{"truth":"changed"}')
            with self.assertRaisesRegex(ValueError, 'frozen_file_changed'): r.c.verify_binding(bound)

if __name__ == '__main__': unittest.main()
