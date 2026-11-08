"""Offline orchestration tests. Fake transport never invokes the model/API."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import b_contract_probe as p


class RecordedFakeCaller:
    """Reuse recorded responses; patch fixtures are test-only and never admitted."""
    def __init__(self, run):
        self.run = run
        self.request_counter = self.http_counter = 0
        self.cases = {p.r.base.read(path)['name']: p.r.base.read(path)
                      for path in p.r.base.read(run / 'MANIFEST.json')['cases']}

    @staticmethod
    def fresh_fixture(case):
        # Test-only structural fixture: remove the obsolete whole group, keep
        # all other recorded values. This never enters a real run or source.
        if case['name'] == 'fresh_v35_s03':
            doc = p.r.parse_object(case['previous_raw'])
            del doc['route_fulfillment']['parent_goal_checks']
            return json.dumps(doc)
        return case['previous_raw']

    def call(self, label, payload, proof):
        self.request_counter += 1
        self.http_counter += 1
        directory = self.run / 'requests' / f'{self.request_counter:03d}-{label}'
        case = self.cases[proof['regression_case']]
        if case['kind'] == 'legacy_repair' or proof.get('schema_repair_only'):
            spec = case['initial_repair'] if case['kind'] == 'legacy_repair' else p._format_repair(
                case, self.fresh_fixture(case), p._evaluate(case, self.fresh_fixture(case)))
            doc = spec['previous_raw']
            rows = []
            for auth in spec['authorized_repairs']:
                row = {'op': auth['op'], 'path': auth['path']}
                if auth['op'] == 'move':
                    row['from'] = auth['source']
                elif auth['op'] == 'replace':
                    row['value'] = auth['allowed'][0]
                else:
                    row['value'] = {'type': 'assigned_event_mismatch', 'historical_claim': '',
                                    'proposed_claim': 'OFFLINE TEST ONLY: original planned action',
                                    'correction': 'OFFLINE TEST ONLY: retain the recorded negative finding'}
                rows.append(row)
            raw = json.dumps({**{key: doc[key] for key in ('root_id', 'route', 'scene_id')},
                              'repair_patches': rows})
        else:
            # Deliberately replay old unmodified scans. These must not magically
            # become semantic/contract successes under the changed v35 contract.
            raw = self.fresh_fixture(case)
        metadata = {'returned_model': p.r.MODEL, 'finish_reason': 'stop',
                    'http_status': 200, 'offline_test_fake_transport': True}
        files = {
            'REQUEST.json': payload,
            'PROOF.json': {**proof, 'payload_sha256': p.r.base.digest(payload)},
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


class ProbeTests(unittest.TestCase):
    def test_frozen_input_identity_and_separate_dimension_expectations(self):
        cases = [case for case, _ in p._source_cases()]
        self.assertEqual(len(cases), 8)
        self.assertEqual(sum(case['logical_call_max'] for case in cases), 11)
        actual = [case for case in cases if case['name'].startswith('fresh_v35_')]
        self.assertEqual(len({p.r.base.digest(case['inputs']['history']) for case in actual}), 1)
        self.assertEqual(len(actual[0]['inputs']['history']), 132)
        for case in actual:
            payload, _ = p._make_request(case)
            sent = p._decode(payload)
            self.assertEqual(sent['cumulative_history'], case['inputs']['history'])
            self.assertNotIn('expected_dimensions', sent)
            self.assertNotIn(case['previous_raw'], payload['messages'][0]['content'])
        s03 = next(case for case in cases if case['name'] == 'fresh_v35_s03')
        self.assertEqual(s03['expected_status'], 'REJECT')
        self.assertFalse(s03['expected_dimensions']['old_goal_blocker'])
        self.assertEqual(s03['expected_dimensions']['card_compatible_steps'], ['s3', 's4'])

    def test_prepare_mock_live_replay_audit_and_tamper_detection(self):
        parent = p.RUNTIME / 'probes'
        parent.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='offline_probe_test_', dir=parent))
        run.rmdir()
        try:
            phantom = types.ModuleType('offline_dynamic_phantom')
            phantom.__file__ = '_classes.py'
            with patch.dict(p.sys.modules, {'offline_dynamic_phantom': phantom}):
                prepared = p.prepare(run)
            self.assertEqual(prepared['http_calls'], 0)
            self.assertEqual(p.audit(run)['regression_status'], 'PREPARED_NO_LIVE_CALLS')
            with patch.object(p.r, 'Caller', RecordedFakeCaller):
                report = p.live(run)
            self.assertEqual(report['status'], 'REGRESSION_NOT_ESTABLISHED')
            self.assertEqual(report['new_scenes'], 0)
            self.assertLessEqual(report['logical_calls'], 11)
            self.assertTrue(all(row['expected_match'] for row in report['cases'][:3]))
            s03 = next(row for row in report['cases'] if row['case'] == 'fresh_v35_s03')
            self.assertEqual(s03['logical_calls'], 2)
            self.assertTrue(s03['schema_valid'])
            self.assertEqual(s03['status'], 'REJECT')
            audited = p.audit(run)
            self.assertEqual(audited['status'], 'PASS', audited['failures'])
            self.assertEqual(audited['regression_status'], 'REGRESSION_NOT_ESTABLISHED')
            # An audit PASS means exact replay, even when regression has failed.
            response = sorted((run / 'requests').glob('*/RESPONSE.json'))[0]
            doc = p.r.base.read(response)
            doc['text'] = '{}'
            response.write_text(json.dumps(doc))
            with self.assertRaises((ValueError, FileNotFoundError)):
                result = p.audit(run)
                if result['status'] != 'PASS':
                    raise ValueError('expected audit failure on response tampering')
        finally:
            shutil.rmtree(run)


if __name__ == '__main__':
    unittest.main()
