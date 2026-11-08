"""Necessary synthetic checks; every artifact is confined to a temporary folder."""
from copy import deepcopy
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import reporting as r


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def dimensions(question_type='single_choice', change=None):
    result = {name: {'judgment': 'PASS' if name not in r.TYPE_SPECIFIC or r.TYPE_SPECIFIC[name] == question_type else 'NOT_APPLICABLE', 'reason': 'Synthetic applicability check.', 'quotes': []} for name in r.DIMENSIONS}
    if change:
        result[change[0]]['judgment'] = change[1]
    return result


def candidate(suffix='Q01', qtype='single_choice'):
    row = {'candidate_id': 'intent_1-C-' + suffix, 'question_type': qtype, 'question': 'Which synthetic fact?',
           'options': {'A': 'alpha', 'B': 'beta', 'C': 'gamma', 'D': 'delta'}, 'items': {}, 'correct_option': 'A', 'gold': ['A'],
           'evidence_refs': ['T1'], 'evidence_groups': {'fact': ['T1']}, 'discriminative_variable': 'synthetic fact',
           'schema_valid': True, 'contract_group_alignment': {'fact': False}, 'generation_source': 'contract_guided',
           'intent_reference': 'intent_1'}
    if qtype == 'multi_select':
        row.update(options={**row['options'], 'E': 'epsilon'}, correct_option=None, gold=['A', 'B'])
    if qtype == 'ordering':
        row.update(items={f'I{i}': f'event {i}' for i in range(1, 5)}, options={}, correct_option=None, gold=['I1', 'I2', 'I3', 'I4'])
    return row


def stage(arm=r.ARMS[0], root=r.ROOTS[0], slot='intent_1', which='primary', candidates=None, missing=None):
    candidates = candidates if candidates is not None else [candidate()]
    return {'arm_id': arm, 'root_id': root, 'slot_id': slot, 'stage': which,
            'request_id': f'{root}-contract_guided-{"generator" if which == "primary" else "repair"}-{slot}',
            'raw_parsed': {'candidates': deepcopy(candidates)}, 'validation': {'candidates': candidates, 'errors': [],
            'minimum_acceptance': {'accepted': not missing, 'missing_minimums': missing or {}}}}


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.run = Path(self.temp.name)
        for root in r.ROOTS:
            write(self.run / 'inputs/shared' / root / 'expanded_dialogue_clean.json', {'turns': [{'turn_ref': 'T1', 'text': 'Synthetic alpha fact.'}]})
            write(self.run / 'inputs/shared' / root / 'history_only.json', {'records': [{'stem': 'Synthetic prior stem', 'source_candidate_id': 'old'}]})

    def tearDown(self):
        self.temp.cleanup()

    def test_prepared_report_has_empty_formats_and_no_fake_qa_or_network(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
            result = r.report(self.run)
        self.assertEqual(result['status'], 'PREPARED_NOT_RUN')
        self.assertEqual(result['candidate_rows'], 0)
        self.assertEqual(result['accounting']['logical_calls'], 0)
        self.assertEqual(result['ACTUAL_EXTERNAL_LOGICAL_CALLS'], 0)
        self.assertEqual((self.run / 'reports/quality_review.jsonl').read_text(), '')
        self.assertEqual((self.run / 'reports/quality_review_views.jsonl').read_text(), '')
        self.assertFalse((self.run / 'audit/quality_review_freeze.json').exists())
        self.assertTrue((self.run / 'policy_snapshot/review_schemas/quality_review.schema.json').is_file())
        for group in result['summary']['groups']:
            self.assertEqual(group['planned_slots'], 3)
            self.assertIsNone(group['effective_sc_ms_yield_per_planned_slot'])
            self.assertEqual(group['token_cost_per_final_pass']['status'], 'NOT_RUN')
        self.assertEqual(result['history_pools'][0]['records_with_gold'], 0)

    def test_policy_idempotence_and_drift_rejection(self):
        r.prepare_review_assets(self.run)
        first = (self.run / 'review_policy.json').read_bytes()
        r.prepare_review_assets(self.run)
        self.assertEqual(first, (self.run / 'review_policy.json').read_bytes())
        write(self.run / 'review_policy.json', {'incorrect': True})
        with self.assertRaisesRegex(ValueError, 'existing_review_policy_differs'):
            r.prepare_review_assets(self.run)

    def test_quality_fail_uncertainty_and_applicability(self):
        self.assertEqual(r.aggregate_quality(dimensions(), 'single_choice'), 'PASS')
        self.assertEqual(r.aggregate_quality(dimensions(change=('GOLD_ITEM_SUPPORT', 'UNDETERMINED')), 'single_choice'), 'UNDETERMINED')
        row = dimensions(change=('GOLD_ITEM_SUPPORT', 'UNDETERMINED'))
        row['ANSWER_UNAMBIGUITY']['judgment'] = 'FAIL'
        self.assertEqual(r.aggregate_quality(row, 'single_choice'), 'FAIL')
        with self.assertRaisesRegex(ValueError, 'applicability'):
            r.aggregate_quality(dimensions(change=('SINGLE_CHOICE_UNIQUENESS', 'NOT_APPLICABLE')), 'single_choice')
        with self.assertRaisesRegex(ValueError, 'all_quality_dimensions_required'):
            r.aggregate_quality({}, 'single_choice')

    def test_every_primary_and_repair_row_retained_and_only_repair_final(self):
        primary = stage(missing={'ordering': 1})
        repair = stage(which='repair')
        r._write_jsonl(self.run / 'reports/runtime_stage_records.jsonl', [primary, repair])
        final = {**repair['validation']['candidates'][0], 'candidate_id': r.ROOTS[0] + ':intent_1-C-Q01', 'arm_id': r.ARMS[0], 'root_id': r.ROOTS[0], 'slot_id': 'intent_1', 'stage': 'repair'}
        r._write_jsonl(self.run / 'reports/runtime_final_candidates.jsonl', [final])
        items, requests, errors = r.collect_candidates(self.run)
        self.assertFalse(errors)
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0]['replaced_primary'])
        self.assertFalse(items[0]['final_validator_accepted'])
        self.assertTrue(items[1]['final_validator_accepted'])
        self.assertTrue(requests[0]['ordering_triggered_repair'])
        self.assertTrue(requests[0]['ordering_only_shortfall'])

    def test_identity_separates_arms_and_stages_and_hidden_view_fields(self):
        rows = [stage(arm=arm, which=which) for arm in r.ARMS for which in ('primary', 'repair')]
        r._write_jsonl(self.run / 'reports/runtime_stage_records.jsonl', rows)
        items, _, errors = r.collect_candidates(self.run)
        self.assertFalse(errors)
        self.assertEqual(len({i['view_id'] for i in items}), 4)
        for item in items:
            self.assertNotIn('intent_reference', item['candidate'])
            self.assertNotIn('generation_source', item['candidate'])
            self.assertNotIn('contract_group_alignment', item['candidate'])
            self.assertNotIn('candidate_id', item['candidate'])
            if item['arm_id'] == r.ARMS[1]:
                self.assertEqual(item['contract_group_alignment_analysis'], 'NOT_APPLICABLE')
                self.assertEqual(item['normalized_candidate']['contract_group_alignment'], {'fact': False})

    def test_missing_or_unparsed_calls_keep_request_denominator(self):
        broken = stage(candidates=[], missing={'single_choice': 2, 'multi_select': 1, 'ordering': 1})
        broken.update(raw_parsed=None, parse_error='synthetic parse failure')
        r._write_jsonl(self.run / 'reports/runtime_stage_records.jsonl', [broken])
        items, requests, errors = r.collect_candidates(self.run)
        self.assertFalse(items)
        self.assertFalse(errors)
        self.assertEqual(requests[0]['structure_status'], 'NOT_EVALUATED_STRUCTURE')
        self.assertEqual(requests[0]['parse_error'], 'synthetic parse failure')
        accounting = r.call_accounting(self.run)
        summary = r.summarize(items, requests, {}, [], accounting)
        self.assertEqual(len(summary['root_paired']), 2)
        self.assertTrue(all(g['planned_slots'] == 3 for g in summary['groups']))
        self.assertEqual(summary['groups'][0]['unparsed_or_empty_request_count'], 1)

    def test_dedup_requires_authored_complete_partition(self):
        r._write_jsonl(self.run / 'reports/runtime_stage_records.jsonl', [stage(candidates=[candidate(), candidate('Q02')])])
        items, _, _ = r.collect_candidates(self.run)
        for item in items:
            item['final_validator_accepted'] = True
        judgments = {i['view_id']: 'PASS' for i in items}
        absent = r.dedup_count(items, judgments, [], r.ARMS[0], r.ROOTS[0], 'final_sc_ms')
        self.assertIsNone(absent['count'])
        cluster = {'arm_id': r.ARMS[0], 'root_id': r.ROOTS[0], 'scope': 'final_sc_ms', 'cluster_id': 'manual-1',
                   'members': [i['view_id'] for i in items], 'reviewer': r.REVIEWER, 'reason': 'Synthetic same relation despite wording.'}
        complete = r.dedup_count(items, judgments, [cluster], r.ARMS[0], r.ROOTS[0], 'final_sc_ms')
        self.assertEqual(complete['count'], 1)
        duplicate = deepcopy(cluster)
        duplicate['members'].append(items[0]['view_id'])
        self.assertIsNone(r.dedup_count(items, judgments, [duplicate], r.ARMS[0], r.ROOTS[0], 'final_sc_ms')['count'])

    def test_cost_includes_replaced_primary_and_missing_usage_is_not_total(self):
        base = r.root_path(self.run, r.ARMS[0], r.ROOTS[0])
        for stage_name, total in [('generator', 100), ('repair', 140)]:
            request_id = f'{r.ROOTS[0]}-contract_guided-{stage_name}-intent_1'
            write(base / 'ledger/requests' / (request_id + '.json'), {'request_id': request_id, 'status': 'completed', 'backend_mode': 'live', 'response_metadata': {'usage': {'prompt_tokens': total - 20, 'completion_tokens': 20, 'total_tokens': total}, 'transport_attempt_count': 1, 'rate_limit_retry_count': 0}})
        accounting = r.call_accounting(self.run)
        group = accounting['groups'][0]
        self.assertEqual(group['token_usage_known_subtotal']['total_tokens'], 240)
        self.assertEqual(r.cost_per_yield(group, 2)['tokens_per_effective_question'], 120)
        self.assertEqual(r.cost_per_yield(group, 0)['status'], 'UNDEFINED_ZERO_YIELD')
        path = base / 'ledger/requests' / f'{r.ROOTS[0]}-contract_guided-repair-intent_1.json'
        state = json.loads(path.read_text())
        state['response_metadata'].pop('usage')
        write(path, state)
        group = r.call_accounting(self.run)['groups'][0]
        self.assertEqual(group['token_usage_known_subtotal']['total_tokens'], 100)
        self.assertEqual(r.cost_per_yield(group, 2)['status'], 'INCOMPLETE_USAGE')
        self.assertIsNone(r.cost_per_yield(group, 2)['tokens_per_effective_question'])

    def test_exact_quotes_and_complete_option_review(self):
        r._write_jsonl(self.run / 'reports/runtime_stage_records.jsonl', [stage()])
        items, _, _ = r.collect_candidates(self.run)
        quote = {'turn_ref': 'T1', 'quote': 'Synthetic alpha fact.'}
        row = {'view_id': items[0]['view_id'], 'reviewer': r.REVIEWER, 'judgment': 'PASS', 'reason': 'Synthetic valid review fixture.',
               'dimensions': dimensions(), 'quotes': [quote], 'option_checks': [{'label': label, 'selected_as_gold': label == 'A', 'support_status': 'SUPPORTED' if label == 'A' else 'NOT_ESTABLISHED', 'reason': 'Synthetic label analysis.', 'quotes': [quote]} for label in 'ABCD']}
        judgments, errors = r.validate_quality_reviews(self.run, items, [row])
        self.assertFalse(errors)
        self.assertEqual(judgments[items[0]['view_id']], 'PASS')
        changed = deepcopy(row)
        changed['quotes'][0]['quote'] = 'not the transcript'
        _, errors = r.validate_quality_reviews(self.run, items, [changed])
        self.assertTrue(any('nonexact_quote' in e for e in errors))
        changed = deepcopy(row)
        changed['option_checks'].pop()
        _, errors = r.validate_quality_reviews(self.run, items, [changed])
        self.assertTrue(any('each_option' in e for e in errors))

    def test_sc_ms_and_ordering_are_separate_and_fixed_root_denominators(self):
        rows = [candidate(), candidate('Q02', 'multi_select'), candidate('Q03', 'ordering')]
        r._write_jsonl(self.run / 'reports/runtime_stage_records.jsonl', [stage(candidates=rows)])
        items, requests, _ = r.collect_candidates(self.run)
        for item in items:
            item['final_validator_accepted'] = True
        judgments = {i['view_id']: 'PASS' for i in items}
        accounting = r.call_accounting(self.run)
        accounting['groups'][0]['logical_calls'] = 1
        summary = r.summarize(items, requests, judgments, [], accounting, quality_frozen=True)
        group = summary['groups'][0]
        self.assertEqual(group['final_sc_ms_pass_count'], 2)
        self.assertEqual(group['effective_sc_ms_yield_per_planned_slot'], 2 / 3)
        self.assertEqual(group['layers']['final']['ordering']['quality']['PASS'], 1)
        self.assertEqual(group['layers']['final']['sc_ms']['quality_pass_fraction'], 1)
        self.assertIsNone(summary['equal_root_weight_mean_delta'])

    def test_partial_call_preserved_without_inventing_normalization(self):
        partial = stage()
        partial.pop('validation')
        base = r.root_path(self.run, r.ARMS[0], r.ROOTS[0])
        write(base / 'stage_calls/intent_1/primary.json', partial)
        items, requests, errors = r.collect_candidates(self.run)
        self.assertFalse(errors)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]['content_reviewable'])
        self.assertFalse(items[0]['program_schema_valid'])
        self.assertFalse(items[0]['final_validator_accepted'])
        self.assertEqual(requests[0]['normalization_status'], 'NOT_COMPLETED_NOT_INFERRED')
        self.assertIsNone(requests[0]['whole_batch_not_accepted'])

    def test_synthetic_ledger_is_not_an_actual_experiment_result(self):
        row = stage()
        r._write_jsonl(self.run / 'reports/runtime_stage_records.jsonl', [row])
        base = r.root_path(self.run, r.ARMS[0], r.ROOTS[0])
        write(base / 'ledger/requests' / (row['request_id'] + '.json'), {'request_id': row['request_id'], 'status': 'completed', 'backend_mode': 'mock'})
        result = r.report(self.run)
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertIn('nonlive_or_synthetic_ledger_cannot_be_reported_as_real_QA', result['errors'])
        self.assertFalse(result['quality_frozen'])

    def test_quality_does_not_freeze_without_runtime_finalization(self):
        stages = []
        for arm in r.ARMS:
            for root in r.ROOTS:
                for number in (1, 2, 3):
                    row = stage(arm, root, 'intent_' + str(number), candidates=[], missing={'single_choice': 2, 'multi_select': 1, 'ordering': 1})
                    stages.append(row)
                    write(r.root_path(self.run, arm, root) / 'ledger/requests' / (row['request_id'] + '.json'), {'request_id': row['request_id'], 'status': 'completed', 'backend_mode': 'live', 'response_metadata': {'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11}, 'transport_attempt_count': 1, 'rate_limit_retry_count': 0}})
        r._write_jsonl(self.run / 'reports/runtime_stage_records.jsonl', stages)
        self.assertFalse(r.report(self.run)['quality_frozen'])
        write(self.run / 'reports/runtime_summary.json', {'status': 'COMPLETED', 'backend_mode': 'live'})
        r._write_jsonl(self.run / 'reports/runtime_final_candidates.jsonl', [])
        result = r.report(self.run)
        self.assertTrue(result['quality_frozen'])
        self.assertTrue(all(g['effective_sc_ms_yield_per_planned_slot'] == 0 for g in result['summary']['groups']))
        self.assertTrue(all(a['token_cost_per_final_pass']['status'] == 'UNDEFINED_ZERO_YIELD' for a in result['summary']['arms']))
        (self.run / 'reports/quality_review.jsonl').write_text('\n')
        changed = r.report(self.run)
        self.assertIn('frozen_quality_inputs_or_judgments_changed', changed['errors'])


if __name__ == '__main__':
    unittest.main()
