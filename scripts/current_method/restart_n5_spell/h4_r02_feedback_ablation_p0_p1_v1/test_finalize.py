"""Coverage and identity checks for the additive final report integration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..core import sha256_file
from .freeze import ARMS, ROOTS
from . import finalize


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def put_rows(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in value))


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.case = {'blind_id': 'Bfixture', 'root_id': ROOTS[0], 'candidate': {'question_type': 'single_choice'},
                     'transcript': {'turns': [{'turn_ref': 'T1', 'text': 'The option is supported.'}]}}
        self.review = {'blind_id': 'Bfixture', 'root_id': ROOTS[0], 'question_type': 'single_choice',
                       'full_transcript_read': True, 'gold_assessment': 'supported',
                       'item_checks': {'A': 'supported', 'B': 'not supported'}, 'ambiguity': 'none observed',
                       'duplicate_notes': 'not observed in reviewed material', 'feedback_relation_present': 'supported',
                       'feedback_relation_evidence_refs': ['T1'], 'recommendation': 'No label change in this experiment.'}

    def tearDown(self):
        self.tmp.cleanup()

    def test_review_requires_identity_full_read_and_real_refs_but_allows_uncertainty(self):
        finalize.validate_review(self.review, self.case)
        for key, value in [('full_transcript_read', False), ('root_id', ROOTS[1]),
                           ('feedback_relation_evidence_refs', ['MISSING']), ('item_checks', [])]:
            with self.assertRaises(RuntimeError):
                finalize.validate_review({**self.review, key: value}, self.case)
        uncertain = {**self.review, 'gold_assessment': 'underdetermined', 'feedback_relation_present': 'uncertain',
                     'feedback_relation_evidence_refs': []}
        finalize.validate_review(uncertain, self.case)

    def test_shared_case_preserves_two_arm_denominators_and_missing_review_blocks(self):
        mapping = [{'blind_id': 'Bfixture', 'arm': arm, 'root_id': ROOTS[0], 'candidate_id': 'fixture',
                    'question_type': 'single_choice', 'selection_hash': 'fixture', 'group_candidate_count': 1} for arm in ARMS]
        folder = self.run / 'diagnostics/blind_review'
        put_rows(folder / 'arm_mapping_do_not_show_reviewers.jsonl', mapping)
        put(folder / 'cases/Bfixture.json', self.case)
        with patch.object(finalize, 'expected_review_mapping', return_value=(mapping, {'Bfixture': self.case})):
            with self.assertRaisesRegex(RuntimeError, 'coverage_incomplete'):
                finalize.review_summary(self.run)
            put(folder / 'reviews/Bfixture.json', self.review)
            result = finalize.review_summary(self.run)
            self.assertEqual(result['unique_reviews_completed'], 1)
            self.assertEqual(result['selected_candidate_instances'], 2)
            for arm in ARMS:
                group = next(g for g in result['groups'] if g['arm'] == arm and g['root_id'] == 'all' and g['scope'] == 'main')
                self.assertEqual(group['reviewed_candidate_instances'], 1)
                self.assertEqual(group['gold_supported_rate_of_reviewed'], 1)
                self.assertIsNone(group['root_macro_gold_supported_rate'])
            put(folder / 'cases/Bfixture.json', {**self.case, 'arm': 'feedback_on'})
            with self.assertRaisesRegex(RuntimeError, 'blind_case_content_drift'):
                finalize.review_summary(self.run)

    def test_changed_selection_mapping_is_rejected(self):
        mapping = [{'blind_id': 'Bfixture', 'arm': ARMS[0], 'root_id': ROOTS[0], 'candidate_id': 'fixture',
                    'question_type': 'single_choice', 'selection_hash': 'fixture', 'group_candidate_count': 1}]
        put_rows(self.run / 'diagnostics/blind_review/arm_mapping_do_not_show_reviewers.jsonl', [{**mapping[0], 'candidate_id': 'different'}])
        with patch.object(finalize, 'expected_review_mapping', return_value=(mapping, {'Bfixture': self.case})):
            with self.assertRaisesRegex(RuntimeError, 'differs_from_frozen_selection'):
                finalize.review_summary(self.run)

    def test_old_52_missing_or_incomplete_diagnostics_block(self):
        with self.assertRaisesRegex(RuntimeError, 'diagnostics_missing'):
            finalize.old_ordering_summary(self.run)
        put(self.run / 'diagnostics/old_r02_ordering_review_summary.json', {'coverage': {'reviewed': 51, 'expected': 52, 'all_full_transcripts_read': True}})
        put_rows(self.run / 'diagnostics/old_r02_ordering_review.jsonl', [{}] * 51)
        (self.run / 'diagnostics/old_r02_ordering_review.md').write_text('partial')
        with self.assertRaisesRegex(RuntimeError, 'coverage_incomplete'):
            finalize.old_ordering_summary(self.run)

    def test_enrichment_cannot_emit_complete_before_required_review(self):
        put(self.run / 'audit/reporting_revision_v2.json', {'current_code': []})
        with patch.object(finalize, 'review_summary', side_effect=RuntimeError('review incomplete')):
            with self.assertRaisesRegex(RuntimeError, 'review incomplete'):
                finalize.enrich_report(self.run, {'status': 'COMPLETE'})
        self.assertFalse((self.run / 'reports/p1_comparison.json').exists())

    def qa_fixture(self):
        transport = []
        for arm in ARMS:
            manifest = {'roots': [{'root_id': root, 'shard': 'shard-00'} for root in ROOTS]}
            put(self.run / arm / 'qa/manifest.json', manifest)
            for root in ROOTS:
                rd = self.run / arm / 'qa/shards/shard-00/roots' / root
                response = rd / 'raw/backend_responses/miner.txt'
                response.parent.mkdir(parents=True, exist_ok=True)
                response.write_text('{"intents":[]}')
                state = {'request_id': root+'-intent-miner', 'status': 'completed',
                         'request_metadata': {'arm': arm, 'root_id': root, 'mode': 'intent_miner', 'attempt': 'primary'},
                         'raw_response_path': str(response), 'raw_response_sha256': sha256_file(response),
                         'response_metadata': {'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}}
                put(rd / 'ledger/requests/miner.json', state)
                for name in ('candidate_generation.json', 'candidate_pool.json', 'candidates_valid.json'):
                    put(rd / name, [])
                put(rd / 'qa_generation_report.json', {'candidate_count': 0, 'valid_candidate_count': 0, 'status': 'incomplete', 'valid_contract_count': 0})
                put(rd / 'qa_completion_checkpoint.json', {'files': {str(p.relative_to(rd)): sha256_file(p) for p in rd.rglob('*') if p.is_file()}})
                transport.extend([{'arm': arm, 'root_id': root, 'event': 'http_attempt_started'},
                                  {'arm': arm, 'root_id': root, 'event': 'logical_call_transport_outcome', 'status': 'completed', 'transport_attempt_count': 1}])
        summary = {'status': 'completed', 'completed_root_arms': 12, 'logical_calls': 12, 'miner_calls': 12,
                   'writer_primary_calls': 0, 'repair_calls': 0, 'completed_calls': 12, 'unknown_billing_status_calls': 0,
                   'http_attempts': 12, 'http_retries': 0, 'transport_failures': 0, 'timeout_calls': 0,
                   'completion_usage': {'prompt_tokens': 24, 'completion_tokens': 12, 'total_tokens': 36},
                   'arms': {arm: {'valid_candidate_count': 0} for arm in ARMS}}
        put(self.run / 'audit/qa_execution_summary.json', summary)
        put_rows(self.run / 'audit/qa_transport_events.jsonl', transport)
        return summary

    def test_per_arm_actual_ledger_tokens_and_global_reconciliation(self):
        summary = self.qa_fixture()
        result = finalize.generation_accounting(self.run)
        self.assertEqual(result['per_arm'][ARMS[0]]['miner_calls'], 6)
        self.assertEqual(result['per_arm'][ARMS[1]]['completion_usage']['total_tokens'], 18)
        self.assertIsNone(result['per_arm'][ARMS[0]]['monetary_cost'])
        put(self.run / 'audit/qa_execution_summary.json', {**summary, 'logical_calls': 13})
        with self.assertRaisesRegex(RuntimeError, 'global_accounting_drift'):
            finalize.generation_accounting(self.run)


if __name__ == '__main__':
    unittest.main()
