"""New-run reference recovery uses raw-source binding and original flow replay."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import a3_state_recovery as x


class FakeTransportCaller:
    """Offline-only raw response fixture; every artifact lives in TemporaryDirectory."""
    invalid_patch = False
    def __init__(self, run):
        self.run = run
        self.request_counter = self.http_counter = 0

    def call(self, label, payload, proof):
        self.request_counter += 1; self.http_counter += 1
        directory = self.run / 'requests' / f'{self.request_counter:03d}-{label}'
        if proof['role'] == 'state_extractor':
            meta, repair, old, history = x._load(self.run)
            # Synthetic fixture selections test contracts; production never selects these IDs.
            response = {'reference_repairs': [{'path': row['path'], 'turn_ref': row['candidate_turn_refs'][0]}
                                             for row in repair['citation_reference_candidates']]}
            if self.invalid_patch:
                response['reference_repairs'].pop()
        else:
            assert proof['role'] == 'planner'
            response = {'root_id': x.a.ROOT, 'route': x.a.ROUTE, 'scenes': [],
                        'shortfall_reason': 'Offline fixture has no generated continuation.'}
        raw = json.dumps(response)
        metadata = {'returned_model': x.r.MODEL, 'finish_reason': 'stop', 'offline_test_fixture': True}
        for name, value in {
            'REQUEST.json': payload, 'PROOF.json': {**proof, 'payload_sha256': x.r.base.digest(payload)},
            'HTTP_ATTEMPT.json': {'http_attempt_number': self.http_counter, 'request_attempt': 1,
                'payload_sha256': x.r.base.digest(payload), 'requested_model': x.r.MODEL,
                'actual_payload_equals_frozen': True},
            'THROTTLE_01.json': {'minimum_gap_after_previous_response_seconds': x.r.MIN_HTTP_GAP_SECONDS},
            'HTTP_RESPONSE_01.json': {'text': raw, 'metadata': metadata},
            'HTTP_RESULT_01.json': {'outcome': 'response', 'http_status': 200},
            'RESPONSE.json': {'text': raw, 'metadata': metadata},
        }.items():
            x.r.base.write(directory / name, value)
        return raw, directory


class StateRecoveryExtensionTests(unittest.TestCase):
    def test_live_and_original_audit_replay_new_reference_call_then_only_new_planner(self):
        source = x._source()
        before = {path: path.read_bytes() for path in source['files']}
        with tempfile.TemporaryDirectory(prefix='state_recovery_offline_', dir=x.r.HERE / 'runs') as tmp:
            run = Path(tmp); run.rmdir()
            prepared = x.prepare(run)
            self.assertEqual(prepared['old_source_state_requests'], 2)
            self.assertEqual(prepared['new_reference_patch_logical_call_max'], 1)
            self.assertEqual(x.audit(run)['status'], 'PASS')
            with patch.object(x.r, 'Caller', FakeTransportCaller):
                result = x.live(run)
            self.assertEqual(result['logical_calls'], 2)
            self.assertEqual(result['new_state_reference_patch_requests'], 1)
            self.assertEqual(result['old_source_state_requests'], 2)
            self.assertEqual(result['total_lineage_logical_calls'], 4)
            self.assertEqual(result['status'], 'A3_SHORTFALL')
            audit = x.audit(run)
            self.assertEqual(audit['status'], 'PASS', audit['failures'])
            directories = sorted((run / 'requests').iterdir())
            self.assertIn('state-reference-recovery-1', directories[0].name)
            self.assertIn('planner-1', directories[1].name)
            merged = x.r.base.read(directories[0] / 'FORMAT_REPAIR_MERGE.json')['candidate']
            old = json.loads(source['raw'])
            for row in source['repair']['citation_reference_candidates']:
                path = row['path']
                import re
                fact, evidence = map(int, re.fullmatch(r'facts\[(\d+)\]\.evidence\[(\d+)\]\.turn_ref', path).groups())
                old['facts'][fact]['evidence'][evidence]['turn_ref'] = merged['facts'][fact]['evidence'][evidence]['turn_ref']
            self.assertEqual(old, merged)  # only eight declared reference leaves changed
            self.assertEqual(x.r.base.read(run / 'MANIFEST.json')['version'], x.a.VERSION)
            self.assertEqual(x.r.base.read(run / 'REPORT.json')['execution_entrypoint'], x.ENTRYPOINT)
        self.assertEqual(before, {path: path.read_bytes() for path in source['files']})

    def test_invalid_patch_stops_after_exactly_one_new_call_no_new_planner(self):
        with tempfile.TemporaryDirectory(prefix='state_recovery_reject_', dir=x.r.HERE / 'runs') as tmp:
            run = Path(tmp); run.rmdir(); x.prepare(run)
            with patch.object(x.r, 'Caller', FakeTransportCaller), patch.object(FakeTransportCaller, 'invalid_patch', True):
                result = x.live(run)
            self.assertEqual(result['logical_calls'], 1)
            self.assertEqual(result['new_state_reference_patch_requests'], 1)
            self.assertEqual(result['new_accepted_A_scenes'], 0)
            self.assertFalse((run / 'roots' / x.a.ROOT / 'A/S03.LATEST_STATE.json').exists())
            self.assertEqual(x.audit(run)['status'], 'PASS')

    def test_existing_patch_contract_rejects_semantic_or_unrequested_changes(self):
        source = x._source()
        patch_value = {'reference_repairs': [{'path': row['path'], 'turn_ref': row['candidate_turn_refs'][0]}
                                           for row in source['repair']['citation_reference_candidates']]}
        semantic = deepcopy(patch_value); semantic['facts'] = [{'statement': 'New fact'}]
        with self.assertRaisesRegex(ValueError, 'state_reference_patch_fields_invalid'):
            x.r.state_contract.merge_state_repair(source['raw'], semantic, source['repair'], source['history'])
        unknown = deepcopy(patch_value); unknown['reference_repairs'][0]['turn_ref'] = 'not-in-history'
        with self.assertRaisesRegex(ValueError, 'state_reference_patch_source_invalid'):
            x.r.state_contract.merge_state_repair(source['raw'], unknown, source['repair'], source['history'])


if __name__ == '__main__':
    unittest.main()
