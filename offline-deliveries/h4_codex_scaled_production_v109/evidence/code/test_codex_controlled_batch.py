"""Offline invariants for immutable batch inputs and arbitrary configured roots."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('controlled', Path(__file__).with_name('codex_controlled_batch.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.batch = self.base / 'arbitrary_batch_name'
        self.root = 'cc-train-episode-987654'
        self.old = {'root_id': self.root, 'turns': [
            {'turn_ref': 'OLD:T001', 'session_ref': 'OLD', 'scene_ref': 'OLD',
             'speaker': 'Morgan', 'text': 'The planned visit is tomorrow.', 'origin': 'historical'}]}
        self.baseline = self.base / 'baseline.json'; c.save(self.baseline, self.old)
        self.pool = self.base / 'pool.json'
        c.save(self.pool, {'rows': [{'row': {'original_index': 12345}, 'payload': 'event'}]})
        self.selection = self.base / 'selection.json'
        self.selection_value = {'source_pool_path': str(self.pool),
            'selection_method': 'offline test only', 'consumed_source_ids': [],
            'consumption_scan': {'test': True}, 'allocations': [{
                'root_id': self.root, 'baseline_path': str(self.baseline),
                'baseline_sha256': c.digest(self.baseline), 'source_id': 'soda:train:12345',
                'source_original_index': 12345, 'role_mapping': {}, 'selection_reason': 'test'}]}
        c.save(self.selection, self.selection_value)

    def tearDown(self):
        self.tmp.cleanup()

    def init(self):
        c.initialize(self.batch, self.selection)

    def draft(self, speaker='Morgan'):
        value = {'root_id': self.root, 'source_id': 'soda:train:12345', 'turns': []}
        for i in range(1, 13):
            value['turns'].append({'turn_ref': f'R02:C:S01:T{i:03d}',
                'session_ref': 'R02:C:S01', 'scene_ref': 'R02:C:S01',
                'speaker': speaker, 'text': f'Concrete test utterance {i}.',
                'origin': 'codex_session_generated'})
        return value

    def test_arbitrary_root_and_prefix_preserved(self):
        self.init(); f = c.root_folder(self.batch, self.root)
        c.save(f / 'EXPANSION_DRAFT.json', self.draft()); c.materialize(self.batch)
        actual = c.read(f / 'CUMULATIVE_SOURCE.json')
        self.assertEqual(actual['turns'][:1], self.old['turns'])
        self.assertEqual(actual['new_turn_count'], 12)
        reservation = c.read(self.batch / 'SOURCE_RESERVATIONS.json')['allocation'][0]
        self.assertEqual(reservation['source_item_sha256'], c.digest(reservation['source_item_path']))
        self.assertEqual(reservation['source_item_canonical_sha256'], c.canonical_digest(c.read(reservation['source_item_path'])))
        self.assertEqual(reservation['source_item_hash_mode'], 'file_bytes')

    def test_consumed_event_rejected_before_reservation(self):
        value = copy.deepcopy(self.selection_value); value['consumed_source_ids'] = ['soda:train:12345']
        p = self.base / 'consumed.json'; c.save(p, value)
        with self.assertRaises(AssertionError): c.initialize(self.batch, p)
        self.assertFalse((self.batch / 'SOURCE_RESERVATIONS.json').exists())

    def test_baseline_hash_mismatch_rejected_before_reservation(self):
        self.baseline.write_text('changed', encoding='utf-8')
        with self.assertRaises(AssertionError): self.init()
        self.assertFalse((self.batch / 'SOURCE_RESERVATIONS.json').exists())

    def test_unestablished_speaker_keeps_failure_not_source(self):
        self.init(); f = c.root_folder(self.batch, self.root)
        c.save(f / 'EXPANSION_DRAFT.json', self.draft('Unknown Actor')); c.materialize(self.batch)
        self.assertTrue((f / 'STRUCTURAL_FAILURE.json').exists())
        self.assertFalse((f / 'CUMULATIVE_SOURCE.json').exists())

    def test_reinitialize_cannot_overwrite_reserved_batch(self):
        self.init(); before = c.digest(self.batch / 'RUN_CONFIG.json')
        with self.assertRaises(AssertionError): self.init()
        self.assertEqual(c.digest(self.batch / 'RUN_CONFIG.json'), before)

    def test_exclusive_save_preserves_existing_review(self):
        p = self.base / 'review.json'; c.save(p, {'first': True})
        with self.assertRaises(FileExistsError): c.save(p, {'second': True})
        self.assertEqual(c.read(p), {'first': True})

if __name__ == '__main__':
    unittest.main()
