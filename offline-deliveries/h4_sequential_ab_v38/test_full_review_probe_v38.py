"""Offline-only receipts; never substitute these for semantic Qwen evidence."""
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
        self.request_counter += 1; self.http_counter += 1
        n = self.request_counter
        directory = self.run / 'requests' / f'{n:03d}-{label}'
        raw = p.r.base.read(p.SOURCE_REQUEST / 'RESPONSE.json')['text']
        if n == 2:
            value = json.loads(raw)
            row = value['checks'][2]
            row['realized'] = False
            row['actor_checks'][0].update(expected_action=p.EXECUTION_REQUIREMENT, satisfied=False,
                action='Only decides to eat; actual eating is absent.', reason='Execution is not reported.')
            raw = json.dumps(value)
        metadata = {'returned_model': p.r.MODEL, 'finish_reason': 'stop', 'http_status': 200,
            'offline_test_fake_transport': True,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
        minute = n - 1
        files = {'REQUEST.json': payload, 'PROOF.json': {**proof, 'payload_sha256': p.r.base.digest(payload)},
            'RESERVATION.json': {'request_number': n, 'http_max': 4, 'max_attempts': 2,
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


class FullReviewProbeTests(unittest.TestCase):
    def test_inputs_unchanged_except_required_action(self):
        original, negative = p.source_cases()
        changed = deepcopy(original['inputs'])
        changed['scene_plan']['developments'][2] = p.EXECUTION_REQUIREMENT
        self.assertEqual(negative['inputs'], changed)
        self.assertEqual(original['card'], negative['card'])
        for case in (original, negative):
            payload, proof = p.make_request(case)
            self.assertEqual(p._context(payload), case['inputs'])
            self.assertEqual(proof['history_turns'], 132)
            self.assertFalse(case['overall_pass_required'])
            self.assertNotIn('manual_semantic_checks', p._context(payload))

    def test_rules_do_not_add_schema_and_bind_both_scope_directions(self):
        case = p.source_cases()[0]
        before = p.r.base.read(p.SOURCE_REQUEST / 'REQUEST.json')['messages'][0]['content']
        after = p.make_request(case)[0]['messages'][0]['content']
        def schema(s):
            return json.loads(s.split('\nOUTPUT SHAPE:\n')[1].split('\nREVIEW INPUT DATA:\n')[0])
        self.assertEqual(schema(before), schema(after))
        self.assertIn('in EVERY output field', after)
        self.assertIn('does not establish execution', ' '.join(after.split()))
        self.assertIn('explicit new report can establish', after)
        self.assertIn('require execution evidence only when', after)

    def test_original_semantic_overclaim_is_not_program_certified(self):
        case = p.source_cases()[0]
        raw = p.r.base.read(p.SOURCE_REQUEST / 'RESPONSE.json')['text']
        result = p.evaluate(case, raw)
        self.assertTrue(result['schema_valid'])
        self.assertTrue(result['observed']['card_s1_realized'])
        self.assertFalse(result['semantic_certified'])
        self.assertTrue(result['manual_semantic_review_required'])
        self.assertIsNone(result['execution_fixture_rejected'])
        self.assertIn('consume', result['review']['raw_parsed']['route_fulfillment']['primary_event'])

    def test_wrong_execution_positive_not_silently_fixed(self):
        case = p.source_cases()[1]
        value = json.loads(p.r.base.read(p.SOURCE_REQUEST / 'RESPONSE.json')['text'])
        value['checks'][2]['actor_checks'][0]['expected_action'] = p.EXECUTION_REQUIREMENT
        result = p.evaluate(case, json.dumps(value))
        self.assertFalse(result['execution_fixture_rejected'])
        self.assertTrue(result['observed']['development_3_realized'])

    def test_mock_full_run_audit_and_tamper(self):
        parent = p.RUNTIME / 'probes'; parent.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='offline_full_review_', dir=parent)); run.rmdir()
        try:
            self.assertEqual(p.prepare(run)['http_calls'], 0)
            self.assertEqual(p.audit(run)['status'], 'PASS')
            with patch.object(p.r, 'Caller', FakeCaller):
                result = p.live(run)
            self.assertEqual(result['logical_calls'], 2)
            self.assertEqual(result['http_attempts'], 2)
            self.assertEqual(result['status'], 'COMPLETE_REQUIRES_INDEPENDENT_SEMANTIC_REVIEW')
            audited = p.audit(run)
            self.assertEqual(audited['status'], 'PASS', audited['failures'])
            self.assertTrue(audited['offline_test_fake_transport'])
            second = run / 'requests/002-full-review-execution_required_fixture/EVALUATION.json'
            outcome = p.r.base.read(second)
            self.assertTrue(outcome['execution_fixture_rejected'])
            self.assertTrue(outcome['observed']['card_s1_realized'])
            with self.assertRaisesRegex(ValueError, 'probe_cannot_restart'):
                p.live(run)
            outcome['review']['raw_parsed']['reason'] = 'tampered'
            second.write_text(json.dumps(outcome))
            self.assertEqual(p.audit(run)['status'], 'FAIL')
        finally:
            shutil.rmtree(run)


if __name__ == '__main__': unittest.main()
