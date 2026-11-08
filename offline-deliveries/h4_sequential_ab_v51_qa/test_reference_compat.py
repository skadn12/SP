"""Offline v51 citation-spelling compatibility; semantic gates stay unchanged."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import generate
import ordering_gate_v45 as ordering
import reference_compat


ROOT = 'cc-train-episode-138343'


def source():
    return {'root_id': ROOT, 'turns': [
        {'turn_ref': f'CANDIDATE:A:S01:T{i:03d}', 'speaker': 'Alex',
         'text': f'SYNTHETIC exact statement {i}.'} for i in range(1, 7)]}


class ReferenceCompatibilityTests(unittest.TestCase):
    def test_only_one_current_root_prefix_with_unique_exact_quote_and_matching_speaker(self):
        src = source()
        raw = {'turn_ref': ROOT + ':' + src['turns'][0]['turn_ref'],
               'speaker': 'Alex', 'quote': src['turns'][0]['text'], 'relation': 'supports'}
        before = deepcopy(raw)
        fixed, evidence = reference_compat.normalize_entry(raw, ROOT, src)
        self.assertEqual(raw, before)
        self.assertEqual(fixed, {**raw, 'turn_ref': src['turns'][0]['turn_ref']})
        self.assertEqual(len(evidence), 1)
        self.assertTrue(evidence[0]['exact_unique_quote'])
        self.assertFalse(evidence[0]['semantic_fields_changed'])
        self.assertTrue(evidence[0]['raw_response_unchanged'])
        self.assertEqual(reference_compat.normalize_entry(fixed, ROOT, src), (fixed, []))

        failures = [
            {**raw, 'turn_ref': 'cc-train-episode-135466:' + src['turns'][0]['turn_ref']},
            {**raw, 'turn_ref': ROOT + ':' + ROOT + ':' + src['turns'][0]['turn_ref']},
            {**raw, 'turn_ref': ROOT + ':CANDIDATE:A:S09:T001'},
            {**raw, 'quote': 'not in source'},
            {**raw, 'quote': ''},
            {**raw, 'speaker': 'Other'},
        ]
        repeated = deepcopy(src)
        repeated['turns'][0]['text'] = 'aaaa'
        failures.append(({**raw, 'quote': 'aa'}, repeated))
        duplicate = deepcopy(src)
        duplicate['turns'].append(deepcopy(duplicate['turns'][0]))
        failures.append((raw, duplicate))
        for entry in failures:
            entry, check_source = entry if isinstance(entry, tuple) else (entry, src)
            with self.subTest(entry=entry):
                self.assertEqual(reference_compat.normalize_entry(entry, ROOT, check_source), (entry, []))

    def test_multiselect_keeps_relations_gold_and_foreign_ref_diagnostic(self):
        src = source()
        with tempfile.TemporaryDirectory(prefix='v51-ref-SYNTHETIC-') as temporary:
            path = Path(temporary) / 'source.json'
            path.write_text(json.dumps(src))
            qid = 'synthetic-mc'
            row = {'candidate_id': qid, 'topic': 'SYNTHETIC', 'question': 'Which claims are supported?',
                   'options': [{'proposition': f'SYNTHETIC claim {i}.',
                                'relation': 'supports' if i < 4 else 'contradicts',
                                'turn_ref': ROOT + ':' + turn['turn_ref'], 'speaker': turn['speaker'],
                                'quote': turn['text'], 'rationale': 'SYNTHETIC fixture.'}
                               for i, turn in enumerate(src['turns'], 1)]}
            raw = {'root_id': ROOT, 'questions': [row], 'abstentions': []}
            accepted, rejected, omitted, envelope = generate.validate_draft(
                json.dumps(raw), [qid], ROOT, 'multi_select', src, path)
            self.assertEqual((rejected, omitted, envelope), ([], [], None))
            candidate = accepted[0]
            self.assertEqual(candidate['gold'], list('ABC'))
            self.assertEqual([candidate['option_evidence'][key]['relation'] for key in 'ABCDEF'],
                             ['supports'] * 3 + ['contradicts'] * 3)
            self.assertEqual(candidate['raw_model_candidate'], row)
            self.assertEqual(len(candidate['reference_normalizations']), 6)
            self.assertEqual(candidate['citation_diagnostics'], [])

            foreign = deepcopy(row)
            foreign['options'][0]['turn_ref'] = 'cc-train-episode-135466:' + src['turns'][0]['turn_ref']
            accepted, rejected, _, _ = generate.validate_draft(
                json.dumps({'root_id': ROOT, 'questions': [foreign]}), [qid], ROOT, 'multi_select', src, path)
            self.assertEqual(rejected, [])
            self.assertEqual(accepted[0]['gold'], list('ABC'))
            self.assertEqual(accepted[0]['option_evidence']['A']['turn_ref'], foreign['options'][0]['turn_ref'])
            self.assertIn({'label': 'A', 'code': 'turn_ref_unavailable'}, accepted[0]['citation_diagnostics'])
            self.assertEqual(len(accepted[0]['reference_normalizations']), 5)

    def test_ordering_recomputes_gold_only_after_exact_anchor_resolution(self):
        src = source()
        with tempfile.TemporaryDirectory(prefix='v51-order-SYNTHETIC-') as temporary:
            path = Path(temporary) / 'source.json'
            path.write_text(json.dumps(src))
            row = {'candidate_id': 'synthetic-order', 'root_id': ROOT, 'question_type': 'ordering',
                   'chronology_kind': 'conversation_order',
                   'question': 'Order as statements APPEAR IN THE SUPPLIED DIALOGUE.',
                   'items': {f'I{i}': turn['text'] for i, turn in enumerate(src['turns'], 1)},
                   'item_evidence': {f'I{i}': {'turn_ref': ROOT + ':' + turn['turn_ref'],
                       'speaker': turn['speaker'], 'quote': turn['text'], 'status': 'statement',
                       'reason': 'SYNTHETIC exact source.'} for i, turn in enumerate(src['turns'], 1)},
                   'gold': [f'I{i}' for i in range(6, 0, -1)]}
            errors, candidate = ordering.validate_generated_ordering(row, row['candidate_id'], ROOT, src, path)
            self.assertEqual(errors, [])
            self.assertEqual(candidate['gold'], [f'I{i}' for i in range(1, 7)])
            self.assertEqual(candidate['original_gold'], row['gold'])
            self.assertEqual(candidate['raw_model_candidate'], row)
            self.assertEqual(len(candidate['reference_normalizations']), 6)
            broken = deepcopy(row)
            broken['item_evidence']['I1']['speaker'] = 'Other'
            errors, candidate = ordering.validate_generated_ordering(broken, row['candidate_id'], ROOT, src, path)
            self.assertIn('I1:anchor_not_in_source', errors)
            self.assertIsNone(candidate)

    def test_nested_single_review_only_wraps_schema_and_keeps_duplicate_key_rejection(self):
        src = source()
        qid = 'synthetic-order'
        item = lambda turn: {'supported': True, 'order_anchor': {'turn_ref': turn['turn_ref'],
            'quote': turn['text']}, 'support_refs': [], 'alternative_matching_anchors': [],
            'actor_ok': True, 'modality_ok': True, 'state_ok': True, 'neutral_stem_ok': True}
        review = {'candidate_id': qid,
                  'items': {f'I{i}': item(turn) for i, turn in enumerate(src['turns'], 1)},
                  'coherent': False, 'unique_order': True, 'alternative_orders': [],
                  'neutral_stem_ok': True}
        records = []
        parsed = ordering._review_object(json.dumps({'reviews': review}), [qid], normalizations=records)
        self.assertEqual(parsed['reviews'], [review])
        self.assertEqual(records[0]['kind'], 'NESTED_SINGLE_REVIEW_OBJECT_WRAPPED')
        self.assertFalse(parsed['reviews'][0]['coherent'], 'Wrapper must not change reviewer verdict')
        for obj, ids in [({'reviews': {**review, 'candidate_id': 'wrong'}}, [qid]),
                         ({'reviews': {**review, 'extra': True}}, [qid]),
                         ({'reviews': review}, [qid, 'second'])]:
            with self.subTest(ids=ids, obj=obj), self.assertRaises(ValueError):
                ordering._review_object(json.dumps(obj), ids)
        with self.assertRaisesRegex(ValueError, 'review_json_invalid'):
            ordering._review_object('{"reviews":{"candidate_id":"synthetic-order",'
                                    '"candidate_id":"synthetic-order"}}', [qid])


if __name__ == '__main__':
    unittest.main()
