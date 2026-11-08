"""Offline tests of immutable ledger provenance, bounded calls and audit replay."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import two_stage_probe as p


def ledger(case):
    c = case['inputs']
    return {'root_id': c['root_id'], 'scene_id': c['scene_id'],
        'scene_actions': [{'action_id': 'a1', 'actor': 'Emily', 'action': 'speaks in this scene',
            'state': 'completed', 'evidence': [{'turn_ref': 'NEW:T001'}]}],
        'historical_events': [{'event_id': 'h1', 'event': 'a prior conversation',
            'state': 'completed', 'evidence': [{'turn_ref': c['branch_history'][0]['turn_ref']}]}],
        'main_activity': {'description': 'A conversation', 'action_ids': ['a1']},
        'background_context': []}


class FakeCaller:
    invalid_B = False

    def __init__(self, run):
        self.run = run
        self.request_counter = self.http_counter = 0

    def call(self, label, payload, proof):
        self.request_counter += 1
        self.http_counter += 1
        n = self.request_counter
        directory = self.run / 'requests' / f'{n:03d}-{label}'
        cases = p.cases_by_name()
        if label.startswith('evidence-'):
            group = label.removeprefix('evidence-')
            case = cases[p.EVIDENCE_CASES[group]]
            raw = '{"bad":true}' if self.invalid_B and group == 'B' else json.dumps(ledger(case))
        else:
            name = label.removeprefix('full-review-')
            case = cases[name]
            raw = p.r.base.read(Path(case['source_request']) / 'RESPONSE.json')['text']
            if name == p.CASE_NAMES[1]:
                value = json.loads(raw)
                value['checks'][2]['actor_checks'][0]['expected_action'] = p.fixed.EXECUTION_REQUIREMENT
                raw = json.dumps(value)
        metadata = {'returned_model': p.r.MODEL, 'finish_reason': 'stop', 'http_status': 200,
            'offline_test_fake_transport': True,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
        begin = datetime(2026, 10, 10, 13, 0, tzinfo=timezone.utc) + timedelta(minutes=n-1)
        files = {'REQUEST.json': payload,
            'PROOF.json': {**proof, 'payload_sha256': p.r.base.digest(payload)},
            'RESERVATION.json': {'request_number': n, 'http_max': 10, 'max_attempts': 2,
                'retryable_statuses': [503, 504], 'status': 'RESERVED'},
            'RESPONSE.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESPONSE_01.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESULT_01.json': {'outcome': 'response', 'http_status': 200,
                'http_completed_at_utc': (begin + timedelta(seconds=1)).isoformat()},
            'HTTP_ATTEMPT.json': {'http_attempt_number': n, 'request_attempt': 1,
                'payload_sha256': p.r.base.digest(payload), 'actual_payload_equals_frozen': True,
                'requested_model': p.r.MODEL},
            'THROTTLE_01.json': {'minimum_gap_after_previous_response_seconds': 35,
                'waited_seconds': 0, 'http_started_at_utc': begin.isoformat()}}
        for name, value in files.items():
            p.r.base.write(directory / name, value)
        return raw, directory


class InvalidBLedgerCaller(FakeCaller):
    invalid_B = True


class TwoStageProbeTests(unittest.TestCase):
    def setUp(self):
        self.cases = p.cases_by_name()

    def new_run(self):
        parent = p.RUNTIME / 'probes'
        parent.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='offline_two_stage_', dir=parent))
        run.rmdir()
        self.addCleanup(lambda: shutil.rmtree(run) if run.exists() else None)
        p.prepare(run)
        return run

    def test_same_B_evidence_ignores_requirements_and_preserves_turns(self):
        original, negative = [self.cases[name] for name in p.CASE_NAMES[:2]]
        self.assertEqual(p.evidence.build_evidence_request(original), p.evidence.build_evidence_request(negative))
        context = p.evidence.evidence_context(original)
        self.assertEqual(set(context), {'root_id', 'scene_id', 'branch_history', 'new_scene'})
        self.assertEqual(context['branch_history'], original['inputs']['branch_history'])
        self.assertEqual(context['new_scene']['turns'], original['inputs']['new_scene']['turns'])
        self.assertEqual(len(context['branch_history']), 132)

    def test_full_review_preserves_schema_and_context_and_raw_ledger(self):
        for case in self.cases.values():
            raw = json.dumps(ledger(case), indent=2)
            baseline, _ = p.fixed.make_request(case)
            payload, proof = p.make_final_request(case, raw)
            before, after = [value['messages'][0]['content'] for value in (baseline, payload)]
            schema = lambda text: json.loads(text.split('\nOUTPUT SHAPE:\n', 1)[1].split('\nREVIEW INPUT DATA:\n', 1)[0])
            self.assertEqual(schema(before), schema(after))
            self.assertEqual(p.fixed._context(payload), case['inputs'])
            envelope = json.loads(after.split(p.EVIDENCE_MARKER, 1)[1].split('\nOUTPUT SHAPE:\n', 1)[0])
            self.assertEqual(envelope, {'raw_response_text': raw, 'raw_response_sha256': p.r.base.digest(raw)})
            self.assertEqual(proof['evidence_raw_response_sha256'], p.r.base.digest(raw))
            self.assertFalse(proof['evidence_semantic_certified'])
            self.assertNotIn('manual_semantic_checks', p.fixed._context(payload))

    def test_valid_but_wrong_ledger_is_not_silently_repaired_or_certified(self):
        case = self.cases[p.CASE_NAMES[0]]
        value = ledger(case)
        value['scene_actions'][0]['action'] = 'Wrong claim: already finished every action.'
        raw = json.dumps(value)
        result = p.evidence.validate_evidence(raw, case)
        self.assertTrue(result['schema_valid'])
        self.assertEqual(result['status'], 'VALID_STRUCTURE_ONLY')
        self.assertFalse(result['semantic_certified'])
        payload, _ = p.make_final_request(case, raw)
        envelope = json.loads(payload['messages'][0]['content'].split(p.EVIDENCE_MARKER)[1].split('\nOUTPUT SHAPE:\n')[0])
        self.assertEqual(envelope['raw_response_text'], raw)
        old_raw = p.r.base.read(p.fixed.SOURCE_REQUEST / 'RESPONSE.json')['text']
        final = p.fixed.evaluate(case, old_raw)
        self.assertIn('consume', final['review']['raw_parsed']['route_fulfillment']['primary_event'])
        self.assertFalse(final['semantic_certified'])

    def test_invalid_ledger_cannot_enter_final_request(self):
        with self.assertRaisesRegex(ValueError, 'evidence_structure_invalid_no_final_review'):
            p.make_final_request(self.cases[p.CASE_NAMES[0]], '{}')

    def test_five_call_replay_shared_ledger_and_restart_rejected(self):
        run = self.new_run()
        self.assertEqual(p.audit(run)['status'], 'PASS')
        with patch.object(p.r, 'Caller', FakeCaller):
            report = p.live(run)
        self.assertEqual((report['logical_calls'], report['http_attempts']), (5, 5))
        self.assertFalse(report['semantic_certified'])
        check = p.audit(run)
        self.assertEqual(check['status'], 'PASS', check['failures'])
        self.assertTrue(check['offline_test_fake_transport'])
        evidence_raw = p.r.base.read(run / 'requests/001-evidence-B/RESPONSE.json')['text']
        hashes = [p.r.base.read(path)['evidence_raw_response_sha256'] for path in (
            run / 'requests/002-full-review-original_decision_scene/PROOF.json',
            run / 'requests/003-full-review-execution_required_fixture/PROOF.json')]
        self.assertEqual(hashes, [p.r.base.digest(evidence_raw)] * 2)
        with self.assertRaisesRegex(ValueError, 'probe_cannot_restart'):
            p.live(run)

    def test_invalid_B_ledger_skips_both_B_reviews_and_continues_A(self):
        run = self.new_run()
        with patch.object(p.r, 'Caller', InvalidBLedgerCaller):
            report = p.live(run)
        self.assertEqual(report['logical_calls'], 3)
        self.assertEqual([row['status'] for row in report['stages']][1:3],
            ['SKIPPED_EVIDENCE_STRUCTURE_OR_TRANSPORT_FAILURE'] * 2)
        result = p.audit(run)
        self.assertEqual(result['status'], 'PASS', result['failures'])

    def test_tampered_evidence_raw_detected_even_with_new_response_hash(self):
        run = self.new_run()
        with patch.object(p.r, 'Caller', FakeCaller):
            p.live(run)
        path = run / 'requests/001-evidence-B/RESPONSE.json'
        response = p.r.base.read(path)
        response['text'] += ' '
        path.write_text(json.dumps(response))
        self.assertEqual(p.audit(run)['status'], 'FAIL')

    def test_tampered_final_ledger_payload_detected(self):
        run = self.new_run()
        with patch.object(p.r, 'Caller', FakeCaller):
            p.live(run)
        path = run / 'requests/003-full-review-execution_required_fixture/REQUEST.json'
        payload = p.r.base.read(path)
        payload['messages'][0]['content'] = payload['messages'][0]['content'].replace('A conversation', 'A changed story')
        path.write_text(json.dumps(payload))
        result = p.audit(run)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('request_replay_changed', result['failures'])

    def test_tampered_final_derived_evaluation_detected(self):
        run = self.new_run()
        with patch.object(p.r, 'Caller', FakeCaller):
            p.live(run)
        path = run / 'requests/005-full-review-reported_test_completion_scene/EVALUATION.json'
        result = p.r.base.read(path)
        result['semantic_certified'] = True
        path.write_text(json.dumps(result))
        self.assertEqual(p.audit(run)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
