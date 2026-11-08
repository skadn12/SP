"""Offline fixtures only; these patches are never submitted as Qwen output."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import state_citation_probe as p


def fixture(case):
    raw = json.loads(case['previous_raw'])
    history = case['original_context']['cumulative_history']
    rows = []
    for path in case['repair']['required_paths']:
        citation, leaf = p.r.state_contract._citation_leaf(raw, path)
        if leaf == 'turn_ref':
            candidates = [turn['turn_ref'] for turn in history if citation['quote'] in turn['text']]
            assert len(candidates) == 1
            value = candidates[0]
        else:
            value = next(turn['text'] for turn in history if turn['turn_ref'] == citation['turn_ref'])
        rows.append({'path': path, 'value': value})
    return {'citation_repairs': rows}


class FakeCaller:
    response_kind = 'patch'

    def __init__(self, run):
        self.run = run
        self.request_counter = self.http_counter = 0

    def call(self, label, payload, proof):
        self.request_counter += 1
        self.http_counter += 1
        directory = self.run / 'requests' / f'001-{label}'
        case = p.r.base.read(self.run / 'CASE.json')
        raw = json.dumps(fixture(case)) if self.response_kind == 'patch' else case['previous_raw']
        metadata = {'returned_model': p.r.MODEL, 'finish_reason': 'stop', 'http_status': 200,
            'offline_test_fake_transport': True,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
        files = {'REQUEST.json': payload, 'PROOF.json': {**proof, 'payload_sha256': p.r.base.digest(payload)},
            'RESERVATION.json': {'request_number': 1, 'http_max': 2, 'max_attempts': 2,
                'retryable_statuses': [503, 504], 'status': 'RESERVED'},
            'RESPONSE.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESPONSE_01.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESULT_01.json': {'outcome': 'response', 'http_status': 200,
                'http_completed_at_utc': '2026-10-10T13:00:01+00:00'},
            'HTTP_ATTEMPT.json': {'http_attempt_number': 1, 'request_attempt': 1,
                'payload_sha256': p.r.base.digest(payload), 'actual_payload_equals_frozen': True,
                'requested_model': p.r.MODEL},
            'THROTTLE_01.json': {'minimum_gap_after_previous_response_seconds': 35,
                'waited_seconds': 0, 'http_started_at_utc': '2026-10-10T13:00:00+00:00'}}
        for filename, value in files.items():
            p.r.base.write(directory / filename, value)
        return raw, directory


class StateCitationProbeTests(unittest.TestCase):
    def test_fixed_input_and_only_five_faulty_leaves(self):
        case = p.source_case()
        payload, proof = p.make_request(case)
        context = json.loads(payload['messages'][0]['content'].split('\nSTATE_INPUT_JSON\n')[1])
        self.assertEqual({k: v for k, v in context.items() if k != 'targeted_repair'}, case['original_context'])
        self.assertEqual(len(context['cumulative_history']), 132)
        self.assertEqual(proof['required_paths'], p.REQUIRED_PATHS)
        self.assertNotIn('Return a complete corrected snapshot', payload['messages'][0]['content'])

    def test_merge_preserves_all_unaffected_fields(self):
        case = p.source_case()
        before = json.loads(case['previous_raw'])
        synthetic = fixture(case)
        result = p.evaluate(case, json.dumps(synthetic))
        self.assertEqual(result['status'], 'CITATION_STRUCTURE_PASS')
        expected = deepcopy(before)
        for row in synthetic['citation_repairs']:
            citation, leaf = p.r.state_contract._citation_leaf(expected, row['path'])
            citation[leaf] = row['value']
        self.assertEqual(result['candidate'], expected)
        self.assertFalse(result['semantic_certified'])

    def test_full_rewrite_and_valid_sibling_edit_rejected(self):
        case = p.source_case()
        self.assertEqual(p.evaluate(case, case['previous_raw'])['status'], 'REJECT')
        altered = fixture(case)
        altered['citation_repairs'].append({'path': 'facts[0].evidence[0].turn_ref', 'value': 'R01:S03:T001'})
        self.assertEqual(p.evaluate(case, json.dumps(altered))['status'], 'REJECT')

    def test_offline_mock_call_audit_and_tampering(self):
        parent = p.RUNTIME / 'probes'
        parent.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='offline_state_citation_test_', dir=parent))
        run.rmdir()
        try:
            prepared = p.prepare(run)
            self.assertEqual(prepared['http_calls'], 0)
            self.assertEqual(p.audit(run)['status'], 'PASS')
            with patch.object(p.r, 'Caller', FakeCaller):
                report = p.live(run)
            self.assertEqual(report['logical_calls'], 1)
            self.assertEqual(report['http_attempts'], 1)
            self.assertEqual(report['status'], 'CITATION_STRUCTURE_PASS')
            audited = p.audit(run)
            self.assertEqual(audited['status'], 'PASS', audited['failures'])
            self.assertTrue(audited['offline_test_fake_transport'])
            with self.assertRaisesRegex(ValueError, 'probe_cannot_restart'):
                p.live(run)
            evaluation = next((run / 'requests').glob('*/EVALUATION.json'))
            saved = evaluation.read_bytes()
            doc = json.loads(saved)
            doc['candidate']['facts'][0]['event'] = 'tampered'
            evaluation.write_text(json.dumps(doc))
            self.assertEqual(p.audit(run)['status'], 'FAIL')
            evaluation.write_bytes(saved)
            preview = run / 'REQUEST_PREVIEW.json'
            doc = p.r.base.read(preview)
            doc['proof']['history_turns'] = 1
            preview.write_text(json.dumps(doc))
            self.assertEqual(p.audit(run)['status'], 'FAIL')
        finally:
            shutil.rmtree(run)

    def test_model_full_rewrite_is_bounded_one_call_failure(self):
        parent = p.RUNTIME / 'probes'
        parent.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='offline_state_citation_test_', dir=parent))
        run.rmdir()
        try:
            p.prepare(run)
            with patch.object(p.r, 'Caller', FakeCaller), patch.object(FakeCaller, 'response_kind', 'full'):
                report = p.live(run)
            self.assertEqual(report['status'], 'REJECT')
            self.assertEqual(report['logical_calls'], 1)
            self.assertEqual(p.audit(run)['status'], 'PASS')
        finally:
            shutil.rmtree(run)


if __name__ == '__main__': unittest.main()
