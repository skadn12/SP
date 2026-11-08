"""Offline checks for lossless public-prior rendering and hook isolation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parent.parent / 'engine/prior_rendering.py'
SPEC = importlib.util.spec_from_file_location('v90_rendering_under_test', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class LosslessPriorTests(unittest.TestCase):
    def rows(self):
        return [
            {'question_type': 'multi_select', 'facts': ['Exact A', 'Exact B'], 'dedup_action': 'warning_only'},
            {'question_type': 'ordering', 'facts': ['same text'], 'dedup_action': 'avoid_exact_set'},
            {'question_type': 'multi_select', 'facts': ['Exact A', 'Exact B'], 'dedup_action': 'avoid_exact_set'},
            {'question_type': 'multi_select', 'facts': ['Exact A', 'Exact B'], 'dedup_action': 'warning_only'},
        ]

    def context(self):
        return {'previous_candidates_dedup_only': self.rows(),
                'source_turns': [{'turn_ref': 'root:scene:T1', 'speaker': 'Alex', 'text': 'quote\n  preserved'}],
                'source_metadata': {'nested': {'sha': 'frozen'}},
                'used_six_statement_sets': [{'question_type': 'multi_select', 'prior_indices': [2]}]}

    def test_roundtrip_keeps_repeated_rows_types_actions_order_and_unicode(self):
        rows = self.rows() + [{'question_type': 'ordering', 'facts': ['引号 " 原样 \n'], 'dedup_action': 'warning_only'}]
        self.assertEqual(M.decode_prior(M.encode_prior(rows)), rows)

    def test_noncontiguous_original_indices_remain_identical(self):
        encoded = M.encode_prior(self.rows())
        self.assertEqual(encoded['groups'][0]['prior_indices'], [0, 3])
        self.assertEqual(encoded['groups'][0]['statement_sets'], [['Exact A', 'Exact B']] * 2)

    def test_context_copy_preserves_source_and_indices(self):
        original = self.context(); saved = deepcopy(original)
        transformed, receipt = M.grouped_context(original)
        self.assertEqual(original, saved)
        for key in ['source_turns', 'source_metadata', 'used_six_statement_sets']:
            self.assertEqual(original[key], transformed[key])
        transformed['source_turns'][0]['text'] = 'different'
        self.assertEqual(original, saved)
        self.assertTrue(receipt['cross_type_inventory_preserved'])

    def test_unknown_field_fails_without_silently_dropping_it(self):
        rows = self.rows(); rows[0]['history_status'] = 'distinct'
        with self.assertRaisesRegex(ValueError, 'unsupported_prior_projection_fields'):
            M.encode_prior(rows)

    def test_duplicate_or_missing_original_index_fails(self):
        encoded = M.encode_prior(self.rows())
        encoded['groups'][0]['prior_indices'][1] = 0
        with self.assertRaisesRegex(ValueError, 'prior_index_duplicate_or_invalid'):
            M.decode_prior(encoded)

    def test_missing_index_fails(self):
        encoded = M.encode_prior(self.rows())
        encoded['row_count'] += 1
        with self.assertRaisesRegex(ValueError, 'prior_index_missing'):
            M.decode_prior(encoded)

    def test_ordering_context_passthrough(self):
        context = {'previous_public_questions_dedup_only': {'question_index': []}, 'source_turns': ['full']}
        transformed, receipt = M.grouped_context(context)
        self.assertIs(transformed, context)
        self.assertIsNone(receipt)

    def test_hook_idempotence_original_closure_receipt_and_dependencies(self):
        calls = []
        def original(run, root, instructions, context, max_tokens):
            calls.append(context)
            return {'source': context.get('source_turns'), 'context': context}, {'suffix_sha256': 'unchanged'}
        target = SimpleNamespace(payload_with_feedback=original)
        assets = SimpleNamespace(legacy=SimpleNamespace(legacy=target), _dependencies=lambda: ('provider', {Path('/tmp/old')}))
        M.install(assets); first = target.payload_with_feedback; M.install(assets)
        self.assertIs(first, target.payload_with_feedback)
        payload, receipt = first('run', 'root', 'instructions', self.context(), 6144)
        self.assertEqual(receipt['suffix_sha256'], 'unchanged')
        self.assertEqual(M.decode_prior(payload['context']['previous_candidates_dedup_only']), self.rows())
        ordering = {'source_turns': ['full'], 'previous_public_questions_dedup_only': []}
        _, ordering_receipt = first('run', 'root', 'instructions', ordering, 6144)
        self.assertIs(calls[-1], ordering)
        self.assertEqual(ordering_receipt, {'suffix_sha256': 'unchanged'})
        provider, paths = assets._dependencies()
        self.assertEqual(provider, 'provider')
        self.assertEqual(paths, {Path('/tmp/old'), PATH.resolve()})


if __name__ == '__main__':
    unittest.main()
