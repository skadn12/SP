"""Offline provenance and false-positive guards; these do not certify semantics."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import full_review_probe as p


class FakeCaller:
    def __init__(self, run):
        self.run = run
        self.request_counter = self.http_counter = 0

    def call(self, label, payload, proof):
        self.request_counter += 1
        self.http_counter += 1
        n = self.request_counter
        directory = self.run / 'requests' / f'{n:03d}-{label}'
        case = p.source_cases()[n - 1]
        raw = p.r.base.read(Path(case['source_request']) / 'RESPONSE.json')['text']
        if n == 2:
            value = json.loads(raw)
            row = value['checks'][2]
            row['realized'] = False
            row['actor_checks'][0].update(expected_action=p.EXECUTION_REQUIREMENT,
                satisfied=False, action='Only decides to eat; actual eating is absent.',
                reason='Execution is not reported.')
            raw = json.dumps(value)
        metadata = {'returned_model': p.r.MODEL, 'finish_reason': 'stop', 'http_status': 200,
            'offline_test_fake_transport': True,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
        minute = n - 1
        files = {'REQUEST.json': payload,
            'PROOF.json': {**proof, 'payload_sha256': p.r.base.digest(payload)},
            'RESERVATION.json': {'request_number': n, 'http_max': 6, 'max_attempts': 2,
                'retryable_statuses': [503, 504], 'status': 'RESERVED'},
            'RESPONSE.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESPONSE_01.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESULT_01.json': {'outcome': 'response', 'http_status': 200,
                'http_completed_at_utc': f'2026-10-10T13:0{minute}:01+00:00'},
            'HTTP_ATTEMPT.json': {'http_attempt_number': n, 'request_attempt': 1,
                'payload_sha256': p.r.base.digest(payload), 'actual_payload_equals_frozen': True,
                'requested_model': p.r.MODEL},
            'THROTTLE_01.json': {'minimum_gap_after_previous_response_seconds': 35,
                'waited_seconds': 0, 'http_started_at_utc': f'2026-10-10T13:0{minute}:00+00:00'}}
        for name, value in files.items():
            p.r.base.write(directory / name, value)
        return raw, directory


class FullReviewProbeV39Tests(unittest.TestCase):
    def test_b_pair_only_changes_execution_requirement(self):
        original, negative, _ = p.source_cases()
        changed = deepcopy(original['inputs'])
        changed['scene_plan']['developments'][2] = p.EXECUTION_REQUIREMENT
        self.assertEqual(negative['inputs'], changed)
        self.assertEqual(original['card'], negative['card'])
        self.assertEqual(original['inputs'], p._context(p.r.base.read(p.SOURCE_REQUEST / 'REQUEST.json')))
        self.assertEqual(p.r.base.digest(original['inputs']['branch_history']), p.HISTORY_SHA256)

    def test_positive_preserves_real_scene_and_full_original_history(self):
        positive = p.source_cases()[2]
        original = p._context(p.r.base.read(p.A_SOURCE_REQUEST / 'REQUEST.json'))
        self.assertEqual(positive['inputs'], original)
        self.assertEqual(len(original['branch_history']), 94)
        self.assertEqual(p.r.base.digest(original['branch_history']), p.A_HISTORY_SHA256)
        self.assertEqual(len(original['new_scene']['turns']), 14)
        self.assertEqual(positive['changed_paths'], [])
        self.assertIn('subsequent tests confirmed that stability', original['new_scene']['turns'][2]['text'])
        self.assertFalse(positive['diagnostic_fixture_only'])

    def test_requests_keep_each_source_schema_context_and_no_expected_answers(self):
        for case in p.source_cases():
            payload, proof = p.make_request(case)
            before = p.r.base.read(Path(case['source_request']) / 'REQUEST.json')['messages'][0]['content']
            after = payload['messages'][0]['content']
            def schema(prompt):
                return json.loads(prompt.split('\nOUTPUT SHAPE:\n')[1].split('\nREVIEW INPUT DATA:\n')[0])
            original_shape = schema(before)
            if case['name'] == p.CASE_NAMES[1]:
                original_shape['checks'][2]['actor_checks'][0]['expected_action'] = p.EXECUTION_REQUIREMENT
            self.assertEqual(original_shape, schema(after))
            self.assertEqual(p._context(payload), case['inputs'])
            self.assertEqual(proof['history_turns'], case['source_history_turns'])
            self.assertEqual(proof['history_sha256'], case['source_history_sha256'])
            self.assertNotIn('manual_semantic_checks', p._context(payload))
            self.assertNotIn('completion_positive_controls', p._context(payload))
            self.assertFalse(case['overall_pass_required'])

    def test_original_overclaim_is_not_program_certified(self):
        case = p.source_cases()[0]
        result = p.evaluate(case, p.r.base.read(p.SOURCE_REQUEST / 'RESPONSE.json')['text'])
        self.assertTrue(result['schema_valid'])
        self.assertTrue(result['observed']['card_s1_realized'])
        self.assertFalse(result['semantic_certified'])
        self.assertTrue(result['manual_semantic_review_required'])
        self.assertIn('consume', result['review']['raw_parsed']['route_fulfillment']['primary_event'])

    def test_wrong_execution_positive_is_not_silently_fixed(self):
        case = p.source_cases()[1]
        value = json.loads(p.r.base.read(p.SOURCE_REQUEST / 'RESPONSE.json')['text'])
        value['checks'][2]['actor_checks'][0]['expected_action'] = p.EXECUTION_REQUIREMENT
        result = p.evaluate(case, json.dumps(value))
        self.assertFalse(result['execution_fixture_rejected'])
        self.assertTrue(result['observed']['development_3_realized'])

    def test_a_completion_is_observed_without_b_actor_schema_or_certification(self):
        case = p.source_cases()[2]
        raw = p.r.base.read(p.A_SOURCE_REQUEST / 'RESPONSE.json')['text']
        result = p.evaluate(case, raw)
        self.assertTrue(result['schema_valid'], result['review']['schema_failures'])
        self.assertTrue(result['observed']['completion_development_1_realized'])
        self.assertEqual(result['observed']['completion_development_1_evidence'], [{'turn_ref': 'NEW:T003'}])
        self.assertIsNone(result['review']['actor_decision_provenance'])
        self.assertIsNone(result['execution_fixture_rejected'])
        self.assertFalse(result['semantic_certified'])

    def test_mock_three_case_run_audit_and_tamper(self):
        parent = p.RUNTIME / 'probes'
        parent.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='offline_full_review_', dir=parent))
        run.rmdir()
        try:
            self.assertEqual(p.prepare(run)['http_calls'], 0)
            self.assertEqual(p.audit(run)['status'], 'PASS')
            with patch.object(p.r, 'Caller', FakeCaller):
                report = p.live(run)
            self.assertEqual(report['logical_calls'], 3)
            self.assertEqual(report['http_attempts'], 3)
            self.assertFalse(report['production_admission'])
            audit = p.audit(run)
            self.assertEqual(audit['status'], 'PASS', audit['failures'])
            self.assertTrue(audit['offline_test_fake_transport'])
            third = run / 'requests/003-full-review-reported_test_completion_scene/EVALUATION.json'
            result = p.r.base.read(third)
            self.assertTrue(result['observed']['completion_development_1_realized'])
            with self.assertRaisesRegex(ValueError, 'probe_cannot_restart'):
                p.live(run)
            result['review']['raw_parsed']['reason'] = 'tampered'
            third.write_text(json.dumps(result))
            self.assertEqual(p.audit(run)['status'], 'FAIL')
        finally:
            shutil.rmtree(run)


if __name__ == '__main__':
    unittest.main()
