"""No-network regressions for layer order, denominator retention and metrics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ..core import atomic_json
from ..h4_qa_mining_probe_v1.intent_contract import canonicalize_intents
from ..h4_qa_mining_probe_v1.reasoning_contract import build_reasoning_contract
from ..h4_r02_miner_binding_v2.transfer import derive_all_intent_evidence
from .constants import ARMS, ROOTS, SCHEDULE, root_path
from . import reporting as r


def item(root=ROOTS[0], arm=ARMS[2], index=0, valid=True, state='PROPOSED_TRANSFER'):
    return {'root_id': root, 'arm_id': arm, 'intent_index': index, 'intent_id': 'intent_' + str(index),
        'planned_slot': index < 3, 'returned': True, 'structurally_valid': valid,
        'binding_state': state, 'binding_indices': [index], 'binding_errors': [], 'raw_bindings': [],
        'binding_mechanical_valid': state in {'PROPOSED_TRANSFER', 'NO_SUPPORTED_TRANSFER'},
        'proposed_transfer': state == 'PROPOSED_TRANSFER', 'valid_proposed_transfer': state == 'PROPOSED_TRANSFER',
        'model_declared_no_transfer': state == 'NO_SUPPORTED_TRANSFER', 'valid_no_transfer': state == 'NO_SUPPORTED_TRANSFER'}


def base_review(value, judgment='PASS'):
    return {**{f: value[f] for f in ('root_id', 'arm_id', 'intent_index')}, 'judgment': judgment,
        'reason': 'Authored fixture review of the exact utterances.', 'current_refs': ['R:S:T1'], 'reviewer': r.REVIEWER}


def transfer_review(value, judgment='PASS'):
    evidence = {'judgment': judgment, 'reason': 'Authored fixture comparison.',
        'current_refs': ['R:S:T1'], 'historical_observation_keys': ['observation_cards.json:M0.full_context']}
    return {**base_review(value, judgment), 'historical_observation_keys': ['observation_cards.json:M0.full_context'],
        'criteria': {c: deepcopy(evidence) for c in r.TRANSFER_CRITERIA}}


class ReportingTests(unittest.TestCase):
    def test_structural_missing_auto_state_keeps_denominator_and_no_semantic_fail(self):
        values = [item(index=i, valid=i == 0) for i in range(3)]
        values[2]['returned'] = False
        reviews, errors = r.assemble_intent_reviews(values, [base_review(values[0])])
        self.assertEqual(errors, [])
        self.assertEqual([x['judgment'] for x in reviews], ['PASS', r.NOT_EVALUATED, r.NOT_EVALUATED])
        summary = r.summarize_layers(values, reviews, [])['groups'][2]
        self.assertEqual(summary['planned_slots'], 3)
        self.assertEqual(summary['semantic_pass'], 1)
        self.assertEqual(summary['semantic_fail'], 0)
        self.assertEqual(summary['not_evaluated'], 2)
        self.assertEqual(summary['missing'], 1)
        self.assertEqual(summary['planned_pass_yield'], 1 / 3)
        self.assertEqual(summary['reviewable_pass_fraction'], 1)

    def test_missing_duplicate_unknown_reviews_are_errors_and_no_guessed_semantics(self):
        value = item()
        rows, errors = r.assemble_intent_reviews([value], [])
        self.assertEqual(rows, [])
        self.assertTrue(any('missing_semantic_review' in e for e in errors))
        review = base_review(value)
        _, errors = r.assemble_intent_reviews([value], [review, deepcopy(review), base_review(item(index=9))])
        self.assertTrue(any('duplicate_review_item' in e for e in errors))
        self.assertTrue(any('unknown_review_item' in e for e in errors))
        _, errors = r.assemble_intent_reviews([item(valid=False)], [review])
        self.assertTrue(any('structurally_unusable' in e for e in errors))

    def test_extra_output_retained_reviewed_separately_without_best_three_selection(self):
        values = [item(arm=ARMS[0], index=i) for i in range(4)]
        reviews = [base_review(v, 'FAIL' if i < 3 else 'PASS') for i, v in enumerate(values)]
        _, errors = r.assemble_intent_reviews(values, reviews[:3])
        self.assertTrue(any('missing_semantic_review' in e for e in errors))
        transfer, errors = r.assemble_transfer_reviews(values, reviews, [])
        self.assertEqual(errors, [])
        self.assertEqual({row['judgment'] for row in transfer}, {'NOT_APPLICABLE'})
        group = r.summarize_layers(values, reviews, transfer)['groups'][0]
        self.assertEqual(group['planned_pass_yield'], 0)
        self.assertEqual(group['extra_outputs'], 1)
        self.assertEqual(group['extra_output_reviews'][0]['judgment'], 'PASS')
        self.assertEqual(group['transfer'], {'status': 'NOT_APPLICABLE'})

    def test_transfer_base_gating_blocks_semantic_review_and_preserves_raw_binding(self):
        values = [item(index=i) for i in range(3)]
        values[0]['raw_bindings'] = [{'change_from_source': 'invalid legacy string'}]
        base = [base_review(v, j) for v, j in zip(values, ('FAIL', 'UNDETERMINED', r.NOT_EVALUATED))]
        rows, errors = r.assemble_transfer_reviews(values, base, [])
        self.assertEqual(errors, [])
        self.assertEqual({row['judgment'] for row in rows}, {'NOT_EVALUATED_INPUT_INTENT_NOT_PASS'})
        self.assertEqual(rows[0]['raw_bindings'], values[0]['raw_bindings'])
        _, errors = r.assemble_transfer_reviews(values, base, [transfer_review(values[0])])
        self.assertTrue(any('before_prerequisites' in e for e in errors))
        summary = r.summarize_layers(values, base, rows)['groups'][2]['transfer']
        self.assertEqual(summary['structurally_valid_proposed'], 3)
        self.assertEqual(summary['eligible_transfer_reviews'], 0)
        self.assertEqual(summary['transfer_pass'], 0)
        self.assertEqual(summary['not_evaluated_base'], 3)

    def test_transfer_invalid_decline_and_valid_proposal_have_distinct_states(self):
        values = [item(index=0, state='BINDING_INVALID'), item(index=1, state='NO_SUPPORTED_TRANSFER'), item(index=2)]
        base = [base_review(v) for v in values]
        rows, errors = r.assemble_transfer_reviews(values, base, [transfer_review(values[2])])
        self.assertEqual(errors, [])
        self.assertEqual([v['judgment'] for v in rows], ['NOT_EVALUATED_BINDING_INVALID', 'DECLINED_TRANSFER', 'PASS'])
        summary = r.summarize_layers(values, base, rows)['groups'][2]['transfer']
        self.assertEqual(summary['base_pass'], 3)
        self.assertEqual(summary['planned_slots'], 3)
        self.assertEqual(summary['eligible_transfer_reviews'], 1)
        self.assertEqual(summary['planned_transfer_pass_yield'], 1 / 3)
        self.assertEqual(summary['eligible_transfer_pass_fraction'], 1)
        self.assertEqual(summary['not_evaluated_binding'], 1)
        self.assertEqual(summary['declined_after_base_pass'], 1)
        authored = transfer_review(values[2]); authored['criteria']['scope']['judgment'] = 'FAIL'
        _, errors = r.assemble_transfer_reviews(values, base, [authored])
        self.assertTrue(any('without_all_criteria_pass' in e for e in errors))

    def test_orphan_binding_retained_without_slot_credit(self):
        value = item()
        orphan = {'root_id': ROOTS[0], 'arm_id': ARMS[2], 'binding_indices': [12], 'raw_binding': 123}
        rows, errors = r.assemble_transfer_reviews([value], [base_review(value)], [transfer_review(value)], [orphan])
        self.assertEqual(errors, [])
        self.assertIsNone(rows[-1]['intent_index'])
        self.assertFalse(rows[-1]['planned_slot'])
        self.assertEqual(rows[-1]['raw_binding'], 123)
        self.assertEqual(rows[-1]['judgment'], 'NOT_EVALUATED_BINDING_INVALID')
        self.assertEqual(r.summarize_layers([value], [base_review(value)], rows)['groups'][2]['transfer']['transfer_pass'], 1)

    def test_freeze_precedes_stage_two_and_detects_changed_stage_one(self):
        value = item(); mechanical = {'items': [value]}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            atomic_json(run / 'review_policy.json', {'immutable': True})
            r.write_jsonl(run / 'reports/intent_evidence_review.jsonl', [base_review(value)])
            locked = r.freeze_intent_review(run, mechanical)
            r.write_jsonl(run / 'reports/transfer_judgments.jsonl', [transfer_review(value)])
            self.assertEqual(r.freeze_intent_review(run, mechanical), locked)
            r.write_jsonl(run / 'reports/intent_evidence_review.jsonl', [base_review(value, 'FAIL')])
            with self.assertRaisesRegex(ValueError, 'CHANGED_AFTER_FREEZE'):
                r.freeze_intent_review(run, mechanical)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            r.write_jsonl(run / 'reports/transfer_judgments.jsonl', [transfer_review(value)])
            with self.assertRaisesRegex(ValueError, 'PRECEDES_INTENT_REVIEW_FREEZE'):
                r.freeze_intent_review(run, mechanical)

    def test_metrics_exact_span_scene_mapping_and_unknown_event(self):
        transcript = {'turns': [
            {'turn_ref': 'R:S1:T1', 'scene_ref': 'scene-a'},
            {'turn_ref': 'R:S1:T2', 'scene_ref': 'scene-a'},
            {'turn_ref': 'R:S2:T1', 'scene_ref': 'scene-b'},
            {'turn_ref': 'R:S2:T2', 'scene_ref': 'scene-b'}]}
        derived = {'usable': True, 'derived_current_evidence': [{'turn_ref': t['turn_ref']} for t in transcript['turns']]}
        metrics = r.evidence_metrics(derived, transcript)
        self.assertEqual(metrics['evidence_ref_count'], 4)
        self.assertEqual(metrics['distinct_scene_count'], 2)
        self.assertEqual(metrics['evidence_span'], {'min_position': 1, 'max_position': 4, 'width': 4})
        self.assertIsNone(metrics['distinct_event_count'])
        self.assertIsNone(metrics['admission_thresholds'])
        transcript['turns'] = [{'turn_ref': 'plain'}, {'turn_ref': 'unmapped'}]
        derived['derived_current_evidence'] = [{'turn_ref': 'plain'}, {'turn_ref': 'unmapped'}]
        self.assertIsNone(r.evidence_metrics(derived, transcript)['distinct_scene_count'])
        self.assertEqual(r.evidence_metrics({}, transcript)['evidence_span'], {'min_position': None, 'max_position': None, 'width': None})

    def test_ab_bad_envelope_retains_all_raw_items_with_structural_not_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); root = ROOTS[0]; arm = ARMS[0]
            folder = root_path(run, arm, root)
            atomic_json(run / 'inputs/shared' / root / 'expanded_dialogue_clean.json', {'turns': []})
            raw = {'intents': [{'intent_id': f'raw_{i}'} for i in range(4)], 'feedback_bindings': ['unexpected']}
            r.atomic_text(folder / 'raw/responses' / (root + '-intent-miner.json.txt'), json.dumps(raw))
            atomic_json(folder / 'intents.json', {'validation': {'valid': False, 'errors': ['intent_top_level_schema'], 'intents': []}})
            atomic_json(folder / 'reasoning_contracts.json', [])
            atomic_json(folder / 'derived_intent_evidence.json', {})
            group, values = r.collect_group(run, arm, root)
            self.assertEqual(group['returned'], 3)
            self.assertEqual(group['returned_total'], 4)
            self.assertEqual(group['extra_outputs'], 1)
            self.assertEqual(group['raw_bindings_envelope'], ['unexpected'])
            self.assertEqual([v['intent_id'] for v in values], ['raw_0', 'raw_1', 'raw_2', 'raw_3'])
            rows, errors = r.assemble_intent_reviews(values, [])
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 4)
            self.assertEqual({v['judgment'] for v in rows}, {r.NOT_EVALUATED})

    def test_all_arm_denominators_and_root_pairing_equal_weight(self):
        values, reviews = [], []
        for schedule in SCHEDULE:
            for index in range(3):
                value = item(root=schedule['root_id'], arm=schedule['arm_id'], index=index)
                values.append(value)
                # First root A/B/C PASS counts 1/2/3; second root 3/2/1.
                n = ARMS.index(value['arm_id']) + 1
                if value['root_id'] == ROOTS[1]: n = 4 - n
                reviews.append(base_review(value, 'PASS' if index < n else 'FAIL'))
        summary = r.summarize_layers(values, reviews, [])
        self.assertEqual([a['planned_slots'] for a in summary['arms']], [6, 6, 6])
        self.assertEqual([a['semantic_pass'] for a in summary['arms']], [4, 4, 4])
        self.assertEqual(summary['two_roots_equal_weight'], {'B_minus_A': 0.0, 'C_minus_B': 0.0, 'C_minus_A': 0.0})
        self.assertTrue(all(p['intent_ids_are_not_semantically_paired'] for p in summary['root_paired']))

    def test_accounting_scopes_same_request_id_by_arm_and_known_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); root = ROOTS[0]
            for arm, status in ((ARMS[0], 'completed'), (ARMS[1], 'reserved'), (ARMS[2], 'reserved')):
                path = root_path(run, arm, root) / 'ledger/requests' / (root + '-intent-miner.json')
                atomic_json(path, {'request_id': root + '-intent-miner', 'status': status,
                    'request_metadata': {'root_id': root, 'arm_id': arm},
                    'response_metadata': {'usage': {'prompt_tokens': 100, 'completion_tokens': 20}, 'finish_reason': 'stop'}})
                if arm == ARMS[1]:
                    atomic_json(path.parents[1] / 'observed_failures' / path.name, {'known': True})
            report = r.call_accounting(run)
            self.assertEqual(report['logical_calls'], 3)
            self.assertEqual(report['completed_calls'], 1)
            self.assertEqual(report['observed_failed_calls'], 1)
            self.assertEqual(report['unknown_reservations'], 1)
            self.assertEqual([g['logical_calls'] for g in report['groups'][:3]], [1, 1, 1])
            self.assertEqual(report['usage']['prompt_tokens'], 100)
            self.assertEqual(report['non_miner_request_paths'], [])


if __name__ == '__main__':
    unittest.main()
