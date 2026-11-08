"""Offline tests only. Test-authored responses never become model or pool data."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import mapping_probe as p


def fixture(case):
    c = case['inputs']
    mappings = p.input_mapping(case)
    return {'root_id': c['root_id'], 'route': c['route'], 'scene_id': c['scene_plan']['scene_id'],
        'card_actor_bindings': [{'actor_id': actor_id, 'participant': 'OFFLINE TEST participant'}
                                for actor_id in mappings['required_actor_ids']],
        'card_plan_checks': [{'kind': row['kind'], 'id': row['id'], 'verdict': 'revise',
            'reason': 'OFFLINE TEST ONLY: deliberate negative fixture, no semantic judgment.',
            'correction': 'OFFLINE TEST ONLY: inspect the actual source and plan.',
            'evidence_development_indices': []} for row in [*mappings['steps'], *mappings['dependencies']]]}


class FakeCaller:
    def __init__(self, run):
        self.run = run
        self.request_counter = self.http_counter = 0
        self.cases = {p.r.base.read(path)['name']: p.r.base.read(path)
                      for path in p.r.base.read(run / 'MANIFEST.json')['cases']}

    def call(self, label, payload, proof):
        self.request_counter += 1; self.http_counter += 1
        directory = self.run / 'requests' / f'{self.request_counter:03d}-{label}'
        raw = json.dumps(fixture(self.cases[proof['case']]))
        metadata = {'returned_model': p.r.MODEL, 'finish_reason': 'stop', 'http_status': 200,
            'offline_test_fake_transport': True,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
        files = {'REQUEST.json': payload, 'PROOF.json': {**proof, 'payload_sha256': p.r.base.digest(payload)},
            'RESPONSE.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESPONSE_01.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESULT_01.json': {'outcome': 'response', 'http_status': 200},
            'HTTP_ATTEMPT.json': {'http_attempt_number': self.http_counter, 'request_attempt': 1,
                'payload_sha256': p.r.base.digest(payload), 'actual_payload_equals_frozen': True,
                'requested_model': p.r.MODEL},
            'THROTTLE_01.json': {'minimum_gap_after_previous_response_seconds': p.r.MIN_HTTP_GAP_SECONDS,
                'waited_seconds': 0, 'http_started_at_utc': p.r.now()}}
        for filename, value in files.items():
            p.r.base.write(directory / filename, value)
        return raw, directory


class MappingProbeTests(unittest.TestCase):
    def test_fixed_full_inputs_and_expected_answers_withheld(self):
        cases = p.source_cases()
        self.assertEqual([len(c['inputs']['history']) for c in cases], [132, 132, 120])
        for case in cases:
            payload, proof = p.make_request(case)
            context = json.loads(payload['messages'][0]['content'].split('\nMAPPING_INPUT_JSON\n')[1])
            self.assertEqual(context['scene_plan'], case['inputs']['scene_plan'])
            self.assertEqual(context['cumulative_history'], case['inputs']['history'])
            self.assertEqual(context['dataset_card'], case['inputs']['dataset_card'])
            self.assertNotIn('expected_checks', context)
            self.assertNotIn('expected_status', context)
            self.assertNotIn('card_actor_bindings', context['scene_plan'])
            self.assertTrue(proof['expected_answers_withheld'])
        self.assertEqual(cases[0]['inputs']['scene_plan']['card_development_links'][0]['development_indices'], [1])

    def test_unlinked_s02_evidence_is_rejected_without_changing_original(self):
        case = p.source_cases()[0]
        original = deepcopy(case['inputs']['scene_plan'])
        response = fixture(case)
        row = response['card_plan_checks'][0]
        row.update(verdict='compatible', correction='', evidence_development_indices=[2])
        checked = p.evaluate(case, json.dumps(response))
        self.assertEqual(checked['status'], 'FORMAT_ERROR')
        self.assertIn('card_plan_check_grounding_evidence_outside_mapping', [e['code'] for e in checked['errors']])
        self.assertEqual(case['inputs']['scene_plan'], original)
        self.assertNotIn('card_actor_bindings', original)

    def test_positive_dependency_cannot_borrow_future_endpoint_repair(self):
        case = p.source_cases()[1]
        response = fixture(case)
        row = next(row for row in response['card_plan_checks'] if row['kind'] == 'dependency' and row['id'] == 2)
        mappings = p.input_mapping(case)
        dep = next(dep for dep in mappings['dependencies'] if dep['id'] == 2)
        row.update(verdict='compatible', correction='', evidence_development_indices=sorted(set(
            dep['before_development_indices'] + dep['after_development_indices'])))
        checked = p.evaluate(case, json.dumps(response))
        self.assertEqual(checked['status'], 'FORMAT_ERROR')
        self.assertIn('card_dependency_positive_with_noncompatible_endpoint', [e['code'] for e in checked['errors']])

    def test_model_bindings_are_required(self):
        case = p.source_cases()[2]
        response = fixture(case)
        response['card_actor_bindings'] = []
        checked = p.evaluate(case, json.dumps(response))
        self.assertEqual(checked['status'], 'FORMAT_ERROR')
        self.assertIn('card_actor_binding_coverage_invalid', [e['code'] for e in checked['errors']])

    def test_prepare_fake_live_replay_cost_and_tamper_detection(self):
        parent = p.RUNTIME / 'probes'; parent.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='offline_mapping_test_', dir=parent)); run.rmdir()
        try:
            phantom = types.ModuleType('offline_dynamic_phantom'); phantom.__file__ = '_classes.py'
            with patch.dict(p.sys.modules, {'offline_dynamic_phantom': phantom}):
                prepared = p.prepare(run)
            self.assertEqual(prepared['http_calls'], 0)
            self.assertEqual(p.audit(run)['regression_status'], 'PREPARED_NO_LIVE_CALLS')
            with patch.object(p.r, 'Caller', FakeCaller):
                report = p.live(run)
            self.assertEqual(report['logical_calls'], 3)
            self.assertEqual(report['http_attempts'], 3)
            self.assertEqual(report['status'], 'REGRESSION_NOT_ESTABLISHED')
            self.assertEqual(report['cost']['provider_reported_tokens']['total_tokens'], 360)
            audit = p.audit(run)
            self.assertEqual(audit['status'], 'PASS', audit['failures'])
            self.assertTrue(audit['offline_test_fake_transport'])
            with self.assertRaisesRegex(ValueError, 'probe_cannot_restart'):
                p.live(run)
            evaluation_path = next((run / 'requests').glob('*/EVALUATION.json'))
            saved = evaluation_path.read_bytes()
            tampered = json.loads(saved); tampered['status'] = 'PASS'
            evaluation_path.write_text(json.dumps(tampered))
            self.assertEqual(p.audit(run)['status'], 'FAIL')
            evaluation_path.write_bytes(saved)
            case_path = run / 'cases/case_01.json'
            doc = p.r.base.read(case_path); doc['inputs']['scene_plan']['developments'][0] = 'tampered'
            case_path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                p.verify(run)
        finally:
            shutil.rmtree(run)


if __name__ == '__main__':
    unittest.main()
