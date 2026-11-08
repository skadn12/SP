"""Synthetic integration: a late defect stops successors and preserves history."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_review import Fixture, ROOT, AUTHOR, REVIEWER, write, quiet
import late_source_hold_v1 as late


class LateHoldTests(Fixture):
    def setUp(self):
        super().setUp()
        write(self.unit / 'selection.json', {'root_id': ROOT, 'a_continuations': [], 'allocations': []})
        self.scene = 'R02:A:S01'
        self.draft(self.scene)
        self.read_scene(self.scene)
        self.save_scene(self.scene)
        self.prior = self.folder / 'scenes/A_S01/SOURCE_REVIEW.json'
        self.prior_bytes = self.prior.read_bytes()

    def hold(self):
        raw = self.judgment(self.scene)
        raw['decision'] = 'HOLD'
        raw['read_attestation']['all_reserved_native_events_read'] = True
        raw['checks']['cross_scene_continuity'] = {
            'status': 'FAIL', 'reason': 'Synthetic late cumulative contradiction.'}
        raw['findings'] = [{'severity': 'BLOCKING', 'reason': 'Synthetic late defect.',
                            'citations': [{'turn_ref': self.scene + ':T001', 'speaker': 'Emily',
                                           'quote': 'explicit report number 1.'}]}]
        return raw

    def test_only_assigned_reader_can_start(self):
        with self.assertRaisesRegex(ValueError, 'EXECUTOR_NOT_ASSIGNED_REVIEWER'):
            quiet(late.start, self.batch, ROOT, AUTHOR)
        self.assertFalse(list((self.unit / 'evidence').glob('late_source_recheck_*')))

    def test_late_hold_preserves_review_and_blocks_successor(self):
        quiet(late.start, self.batch, ROOT, REVIEWER)
        quiet(late.save, self.batch, ROOT, REVIEWER, self.supplied(self.hold()))
        self.assertEqual(self.prior.read_bytes(), self.prior_bytes)
        state = late.review.controller_state(self.unit, ROOT)
        self.assertTrue(state['root_stopped'])
        from codex_sequential_ab import materialize_scene
        result = quiet(materialize_scene, self.unit, ROOT, 'R02:A:S02')
        self.assertEqual(result['status'], 'ROOT_STOPPED')

    def test_late_path_cannot_return_ready(self):
        raw = self.hold()
        raw['decision'] = 'READY'
        with self.assertRaisesRegex(ValueError, 'LATE_RECHECK_CAN_ONLY_HOLD'):
            late.validate(raw, late.inputs(self.batch, ROOT, REVIEWER))

    def test_false_citation_preserves_rejection_and_does_not_stop_root(self):
        quiet(late.start, self.batch, ROOT, REVIEWER)
        raw = self.hold()
        raw['findings'][0]['citations'][0]['quote'] = 'not in this source'
        with self.assertRaisesRegex(ValueError, 'CITATION_NOT_EXACT'):
            quiet(late.save, self.batch, ROOT, REVIEWER, self.supplied(raw))
        directory = late.inputs(self.batch, ROOT, REVIEWER)['directory']
        self.assertTrue((directory / 'RAW_JUDGMENT.json').exists())
        self.assertTrue((directory / 'SAVE_FAILURE.json').exists())
        self.assertFalse(late.review.controller_state(self.unit, ROOT)['root_stopped'])

    def test_changed_prior_review_is_rejected(self):
        quiet(late.start, self.batch, ROOT, REVIEWER)
        changed = late.review.read(self.prior)
        changed['unreviewed_addition'] = 'synthetic mutation'
        write(self.prior, changed)
        with self.assertRaisesRegex(ValueError, 'RECHECK_INPUTS_CHANGED'):
            quiet(late.save, self.batch, ROOT, REVIEWER, self.supplied(self.hold()))


if __name__ == '__main__':
    unittest.main()
