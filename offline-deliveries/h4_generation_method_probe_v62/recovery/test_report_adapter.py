"""Offline recovery report checks against bound interruption and intact requests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from copy import deepcopy
import unittest
from unittest.mock import patch
HERE = Path(__file__).resolve().parent

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

audit = load('recovery_audit_test_module', 'audit_provenance.py')
report = load('recovery_report_test_module', 'finalize_report.py')

class AdapterTests(unittest.TestCase):
    def setUp(self):
        blocker = patch('socket.socket', side_effect=AssertionError('No network in offline adapter tests'))
        blocker.start(); self.addCleanup(blocker.stop)
        self.plan, self.recovery = audit.recovery.verify()

    def row(self, chunk):
        return {'root_id': self.plan['root_id'], 'chunk': chunk['chunk_name'], 'arm': chunk['arm'],
            'history': self.plan['history'], 'prior': self.plan['historical_prior'],
            'generation_run': chunk['generation_run'], 'evaluation_run': chunk['evaluation_run']}

    def test_interrupted_audit_has_unknown_completion_not_provider_error(self):
        counts, request, root = audit.interrupted_generation_audit(self.row(self.plan['chunks'][0]))
        self.assertEqual(counts['http_requests'], 1)
        self.assertEqual(counts['http_errors'], 0)
        self.assertEqual(counts['interrupted_unobserved_slots'], 2)
        self.assertEqual(counts['initial_drafts'], 0)
        self.assertIsNone(request['completed_at_utc'])
        self.assertEqual(request['remote_outcome'], 'unknown')
        self.assertEqual(request['token_usage'], 'unknown')
        self.assertIsNone(request['emitted_drafts'])
        self.assertFalse(root['final_failures'])

    def test_interrupted_request_cost_unknown_and_zero_saved_drafts(self):
        run = Path(self.plan['chunks'][0]['generation_run'])
        drafts, requests, errors = report.request_details(run, self.plan['root_id'], 1)
        self.assertEqual(drafts, [])
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0]['interrupted_unobserved'])
        self.assertFalse(requests[0]['remote_error_observed'])
        self.assertIsNone(requests[0]['remote_completion_at_utc'])
        costs = report.cost_summary(requests, 0)
        self.assertEqual(costs['input_tokens'], 'unknown')
        self.assertEqual(costs['unknown_usage_http_attempts'], 1)

    def test_intact_completed_chunks_keep_all_ordinary_checks(self):
        chunks = [c for c in self.plan['chunks'][1:] if (Path(c['generation_run']) / 'REPORT.json').exists()]
        self.assertTrue(chunks, 'at least one intact completed chunk required for integration test')
        for chunk in chunks:
            result = audit.ordinary_generation_audit({'roots': [self.row(chunk)]})
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['counts']['http_requests'], json.loads((Path(chunk['generation_run']) / 'REPORT.json').read_text())['http_attempts'])
            self.assertFalse(result['counts'].get('interrupted_http_requests', 0))

    def test_missing_interruption_branch_cannot_pass_wrong_chunk(self):
        row = self.row(self.plan['chunks'][0]); row['chunk'] = 'direct_c02'
        with self.assertRaisesRegex(AssertionError, 'wrong interrupted chunk'):
            audit.interrupted_generation_audit(row)

    def test_empty_interrupted_chunk_does_not_count_quality_failure(self):
        run = Path(self.plan['chunks'][0]['generation_run'])
        with tempfile.TemporaryDirectory() as temp:
            evaluation = Path(temp)
            pool = evaluation / 'training_pool'; pool.mkdir()
            (pool / 'REPORT.json').write_text(json.dumps({'full_solver_calls': 0, 'new_unique_added': 0}))
            (pool / 'SELECTED_QUESTIONS.json').write_text(json.dumps({'questions': []}))
            solver = evaluation / 'solver'; solver.mkdir()
            (solver / 'SKIPPED.json').write_text(json.dumps({'status': 'EMPTY_POOL_NO_SOLVER', 'candidate_count': 0}))
            row, qm, qr, qs, drafts, req, scored = report.collect_chunk(self.plan['root_id'], 1, run, evaluation)
            self.assertEqual(row['slots'], 2)
            self.assertEqual(row['interrupted_unobserved_slots'], 2)
            self.assertEqual(row['interrupted_http_requests'], 1)
            self.assertEqual(row['remote_error_requests'], 0)
            self.assertEqual(row['other_mechanical_rejected_slots'], 0)
            self.assertEqual(row['initial_drafts'], 0)
            self.assertEqual(row['solver_calls'], 0)

    def test_matched_topics_keep_four_slots_per_arm_and_omit_lost_c01(self):
        chunks = [{**{key: 0 for key in report.TOTAL_KEYS}, 'chunk': c['index'], 'slots': 2,
                   'interrupted_unobserved_slots': 2 if c['chunk_name'] == 'direct_c01' else 0}
                  for c in self.plan['chunks']]
        value = report.matched_completed_method_comparison(self.plan, chunks, [], [], [], [], [], {}, [], [])
        self.assertEqual(value['topic_chunks'], [2, 3])
        for arm, row in value['arms'].items():
            self.assertEqual(row['planned_slots'], 4)
            self.assertEqual(row['execution_lost_slots'], 0)
            self.assertEqual(row['totals']['slots'], 4)
            self.assertEqual(row['content_usable_before_solver'], 0)

    def test_report_renders_explicit_interruption_and_matched_scope(self):
        baseline_path = report.HERE.parent / 'h4_multiselect_probe_v61/RESULT_SUMMARY.json'
        value = json.loads(baseline_path.read_text())
        chunks = [{**{key: 0 for key in report.TOTAL_KEYS}, 'chunk': c['index'], 'slots': 2,
                   'interrupted_unobserved_slots': 2 if c['chunk_name'] == 'direct_c01' else 0}
                  for c in self.plan['chunks']]
        value['arms'] = report.arm_summary(self.plan, chunks, [], [], [], [], [], {}, [])
        value['matched_completed_method_comparison'] = report.matched_completed_method_comparison(self.plan, chunks, [], [], [], [], [], {}, [], [])
        value.update(cross_arm_exact_overlap=[], new_usable_selected_before_exact_dedup=0, new_unique_increment=0)
        rendered = report.report_markdown(value)
        self.assertIn('两个名额为执行损失', rendered)
        self.assertIn('## 同时完成的主题 c02/c03', rendered)
        self.assertIn('远程完成时间仍未知', rendered)
        self.assertIn('| direct | 6 | 2 |', rendered)

if __name__ == '__main__': unittest.main()
