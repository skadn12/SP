"""Offline source/feedback integrity checks; no model or network calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import source_adapter as s

ROOTS = ['cc-train-episode-' + n for n in ['103866','109164','112812','122711','130221']]

class SourceAdapterTests(unittest.TestCase):
    def test_five_sources_and_locked_feedback_coverage(self):
        rows = s.root_rows()
        for root, count in zip(ROOTS, [36,35,35,36,30]):
            with self.subTest(root=root):
                checked = s.checked_source(rows[root]['canonical_dialogue_path'], root)
                self.assertEqual(checked['source'], s.read(rows[root]['canonical_dialogue_path']))
                self.assertEqual(checked['audit']['new_expansion_tokens'], 0)
                self.assertEqual(checked['audit']['source_admission_status'], 'FROZEN_FORMAL_ORIGINAL_PLUS_R01')
                self.assertFalse(any(t['turn_ref'].startswith(('R02:', 'CANDIDATE:')) for t in checked['source']['turns']))
                feedback = s.feedback_data(root)
                self.assertEqual(len(feedback['candidate_history']), count)
                self.assertFalse(feedback['receipt']['current_solver_data_used'])
                self.assertEqual(feedback['source_lock']['evaluation_run_id'], 'h4_round0_trained_solver_train_reeval_v1_live_1')
        self.assertTrue(s.feedback_data(ROOTS[0])['receipt']['existing_compiled_v4_byte_identical'])

    def test_configured_new_roots_preserve_source_and_previous_feedback(self):
        config = s.read(Path(__file__).parent / 'CONFIG.json')
        rows = s.root_rows()
        selected = sorted({slot['root_id'] for slot in config['slots']})
        self.assertEqual(len(selected), 5)
        self.assertTrue(set(selected).isdisjoint(ROOTS))
        for root in selected:
            with self.subTest(root=root):
                row = rows[root]
                checked = s.checked_source(row['canonical_dialogue_path'], root)
                self.assertEqual(checked['source'], s.read(row['canonical_dialogue_path']))
                self.assertTrue(checked['audit']['all_turns_preserved'])
                self.assertEqual(checked['audit']['new_expansion_tokens'], 0)
                feedback = s.feedback_data(root)
                self.assertEqual(len(feedback['candidate_history']), row['round0_candidate_count'])
                self.assertFalse(feedback['receipt']['current_solver_data_used'])
                self.assertFalse(feedback['source_lock']['compilation']['new_evaluation'])

    def test_source_rejects_unbound_path_and_mutated_dialogue(self):
        root = ROOTS[0]; row = s.root_rows()[root]
        with self.assertRaisesRegex(ValueError, 'canonical_path'):
            s.checked_source('/tmp/unbound-source.json', root)
        with self.assertRaisesRegex(ValueError, 'outside_fixed_70'):
            s.checked_source(row['canonical_dialogue_path'], 'cc-train-episode-not-train')
        src = s.read(row['canonical_dialogue_path']); changed = deepcopy(src); changed['turns'].pop()
        with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
            s.validate_original_r01(changed, row)
        changed = deepcopy(src); changed['turns'][-1]['turn_ref'] = 'R02:A:S01:T001'
        changed['complete_dialogue_sha256'] = s.digest({k:v for k,v in changed.items() if k!='complete_dialogue_sha256'})
        tampered_row = {**row, 'complete_dialogue_sha256': changed['complete_dialogue_sha256']}
        with self.assertRaisesRegex(ValueError, 'non_R01'):
            s.validate_original_r01(changed, tampered_row)

    def test_frozen_feedback_is_complete_and_legacy_receipt_compatible(self):
        root = ROOTS[1]; data = s.feedback_data(root)
        with tempfile.TemporaryDirectory(prefix='v68-feedback-test-') as td:
            out = Path(td)/'feedback'
            _, files = s.freeze_feedback(root, out, data=data)
            self.assertEqual(s.read(out/root/'candidate_history.json'), data['candidate_history'])
            compiler = s._compiler()
            payload = {'messages':[{'role':'user','content':'Public task'}]}
            result, receipt = compiler.append_feedback_to_request(payload, out, root, 'writer', 'M_clean_policy')
            self.assertEqual(result['messages'][0]['content'], 'Public task'+data['writer_suffix'])
            self.assertEqual(receipt['source_lock_sha256'], s.sha(out/'source_lock.json'))
            self.assertEqual(receipt['suffix_sha256'], s.sha(out/root/'writer__M_clean_policy_feedback.txt'))
            self.assertEqual(payload['messages'][0]['content'], 'Public task')
            with self.assertRaisesRegex(ValueError, 'already_exists'):
                s.freeze_feedback(root, out, data=data)
            self.assertIn(out/'ADAPTER_RECEIPT.json', files)

if __name__ == '__main__': unittest.main()
