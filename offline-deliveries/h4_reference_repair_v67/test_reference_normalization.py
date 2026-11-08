"""Offline safety and raw-replay checks for exact current-root reference aliases."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import asset_generation as a


class ReferenceNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.root = 'cc-train-episode-fixture'
        self.ref = 'R01:S01:T001'
        self.source = {'root_id': self.root, 'turns': [
            {'turn_ref': self.ref, 'speaker': 'Alex', 'text': 'The invitation is for Tuesday at 7 PM.'},
            {'turn_ref': self.root + ':P01:T001', 'speaker': 'Alex', 'text': 'I received the invitation.'}]}
        self.m = {'root_id': self.root, 'slot_id': 'fixture_s01', 'source_sha256': 'frozen-source-hash'}
        self.asset = {'root_id': self.root, 'slot_id': 'fixture_s01', 'public_stem': 'Select all that apply.',
                      'facts': [{'fact_id': 'F01', 'scope': {'actor': 'Alex', 'event': 'invitation',
                        'time': 'this report', 'condition': 'unspecified', 'action_state': 'reported'},
                        'anchor': {'turn_ref': self.root + ':' + self.ref, 'speaker': 'Alex',
                                   'quote': 'Tuesday at 7 PM'},
                        'supported': {'proposition': 'Alex reported Tuesday at 7 PM.', 'relation': 'supports',
                                      'rationale': 'Explicit invitation schedule.'},
                        'contradicted': {'proposition': 'Alex reported Wednesday at 7 PM.', 'relation': 'contradicts',
                                        'rationale': 'Same invitation specifies Tuesday.'},
                        'contradiction_unavailable_reason': None}]}
        self.net = patch('socket.socket', side_effect=AssertionError('No network in alias tests'))
        self.net.start(); self.addCleanup(self.net.stop)

    def run_asset(self, data=None, source=None):
        raw = json.dumps(self.asset if data is None else data)
        return raw, a.normalize_assets(raw, self.m, self.source if source is None else source)

    def test_alias_changes_only_ref_and_receipt_covers_match(self):
        original = deepcopy(self.asset)
        raw, (accepted, errors, receipt) = self.run_asset()
        expected = deepcopy(original); expected['facts'][0]['anchor']['turn_ref'] = self.ref
        self.assertEqual(accepted, expected); self.assertEqual(errors, [])
        self.assertEqual(self.asset, original); self.assertEqual(json.loads(raw), original)
        row = receipt['facts'][0]
        self.assertEqual(row['original_turn_ref'], self.root + ':' + self.ref)
        self.assertEqual(row['normalized_turn_ref'], self.ref)
        self.assertEqual(row['candidate_match_count'], 1)
        self.assertTrue(row['quote_matches'] and row['speaker_matches'] and row['normalization_applied'])
        self.assertEqual(receipt['normalized_ref_count'], 1)
        self.assertEqual(receipt['raw_text_sha256'], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(receipt['normalized_data_sha256'], a.base.digest(expected))

    def test_existing_formal_ref_is_never_stripped(self):
        data = deepcopy(self.asset)
        data['facts'][0]['anchor'] = {'turn_ref': self.root + ':P01:T001', 'speaker': 'Alex',
                                    'quote': 'I received the invitation.'}
        _, (accepted, errors, receipt) = self.run_asset(data)
        self.assertEqual(accepted, data); self.assertEqual(errors, [])
        self.assertEqual(receipt['normalized_ref_count'], 0)
        self.assertEqual(receipt['facts'][0]['resolution'], 'exact_ref_unchanged')

    def test_existing_ref_with_bad_quote_cannot_fall_back_to_alias(self):
        source = deepcopy(self.source)
        source['turns'].append({'turn_ref': self.root + ':' + self.ref, 'speaker': 'Alex', 'text': 'Different report.'})
        _, (accepted, errors, receipt) = self.run_asset(source=source)
        self.assertIsNone(accepted)
        self.assertEqual(errors, ['facts[0]:quote_not_exact_contiguous'])
        self.assertEqual(receipt['normalized_ref_count'], 0)

    def test_wrong_root_double_prefix_unknown_quote_and_speaker_are_rejected(self):
        for change in ('wrong_root', 'double_prefix', 'unknown', 'quote', 'speaker'):
            with self.subTest(change=change):
                data = deepcopy(self.asset); anchor = data['facts'][0]['anchor']
                if change == 'wrong_root': anchor['turn_ref'] = 'another-root:' + self.ref
                if change == 'double_prefix': anchor['turn_ref'] = self.root + ':' + anchor['turn_ref']
                if change == 'unknown': anchor['turn_ref'] = self.root + ':R01:S09:T999'
                if change == 'quote': anchor['quote'] = 'A different invitation.'
                if change == 'speaker': anchor['speaker'] = 'Blair'
                _, (accepted, errors, receipt) = self.run_asset(data)
                self.assertIsNone(accepted); self.assertEqual(errors, ['facts[0]:unknown_turn_ref:' + json.dumps(anchor['turn_ref']) + ':copy_exact_SOURCE_turn_ref'])
                self.assertEqual(receipt['normalized_ref_count'], 0)
                self.assertEqual(receipt['facts'][0]['normalized_turn_ref'], anchor['turn_ref'])

    def test_duplicate_source_ref_rejected_even_if_text_equal(self):
        source = deepcopy(self.source); source['turns'].append(deepcopy(source['turns'][0]))
        _, (accepted, errors, receipt) = self.run_asset(source=source)
        self.assertIsNone(accepted); self.assertEqual(errors, ['facts[0]:unknown_turn_ref:' + json.dumps(self.asset['facts'][0]['anchor']['turn_ref']) + ':copy_exact_SOURCE_turn_ref'])
        self.assertEqual(receipt['facts'][0]['candidate_match_count'], 2)
        self.assertEqual(receipt['normalized_ref_count'], 0)

    def test_source_root_mismatch_does_not_alias(self):
        source = deepcopy(self.source); source['root_id'] = 'other-root'
        _, (accepted, errors, receipt) = self.run_asset(source=source)
        self.assertIsNone(accepted); self.assertEqual(errors, ['facts[0]:unknown_turn_ref:' + json.dumps(self.asset['facts'][0]['anchor']['turn_ref']) + ':copy_exact_SOURCE_turn_ref'])
        self.assertEqual(receipt['normalized_ref_count'], 0)

    def test_invalid_schema_json_have_deterministic_receipts(self):
        accepted, errors, receipt = a.normalize_assets('not json', self.m, self.source)
        self.assertIsNone(accepted); self.assertEqual(errors, ['invalid_json'])
        self.assertEqual(receipt['facts'], []); self.assertIsNone(receipt['normalized_data_sha256'])
        data = deepcopy(self.asset); data['facts'].append({'fact_id': 'broken'})
        _, (accepted, errors, receipt) = self.run_asset(data)
        self.assertIsNone(accepted); self.assertIn('facts[1]:missing_keys:["anchor","contradicted","contradiction_unavailable_reason","scope","supported"]', errors)
        self.assertEqual(len(receipt['facts']), 2)

    def test_raw_receipt_replay_detects_tampering_without_rewriting_semantics(self):
        raw, (accepted, errors, receipt) = self.run_asset()
        with tempfile.TemporaryDirectory(prefix='v65-alias-replay-') as directory:
            folder = Path(directory); path = folder / 'REFERENCE_NORMALIZATION.json'
            a.write(path, receipt)
            self.assertEqual(a.replay_normalization(folder, raw, self.m, self.source), (accepted, errors))
            for changed_raw in (raw + ' ', raw.replace('Wednesday', 'Thursday')):
                with self.assertRaisesRegex(ValueError, 'asset_reference_normalization_replay_changed'):
                    a.replay_normalization(folder, changed_raw, self.m, self.source)
            changed = deepcopy(receipt); changed['facts'][0]['speaker_matches'] = False
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'asset_reference_normalization_replay_changed'):
                a.replay_normalization(folder, raw, self.m, self.source)

    def test_semantically_false_label_is_not_automatically_repaired(self):
        data = deepcopy(self.asset)
        data['facts'][0]['contradicted']['proposition'] = data['facts'][0]['supported']['proposition']
        _, (accepted, errors, _) = self.run_asset(data)
        self.assertEqual(errors, [])
        self.assertEqual(accepted['facts'][0]['contradicted'], data['facts'][0]['contradicted'])

    def test_generation_preserves_raw_normalizes_once_and_binds_receipt(self):
        # Reuse the established offline provider fixture, including its no-network guard.
        from test_asset_generation import AssetGenerationTests
        fixture = AssetGenerationTests(methodName='test_valid_assets_no_relation_leak_and_mapping_covers_all_versions')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        draft = fixture.asset()
        for fact in draft['facts']:
            fact['anchor']['turn_ref'] = fixture.root + ':' + fact['anchor']['turn_ref']
        original = deepcopy(draft)
        result, payloads = fixture.mock_generate([draft])
        self.assertEqual(len(payloads), 1)
        self.assertEqual(result['status'], 'ASSETS_READY')
        folder = next((fixture.run / 'requests').iterdir())
        response = a.read(folder / 'RESPONSE.json')
        self.assertEqual(json.loads(response['text']), original)
        normalized = a.read(fixture.run / 'ASSETS.json')
        expected = deepcopy(original)
        for fact in expected['facts']:
            fact['anchor']['turn_ref'] = fact['anchor']['turn_ref'][len(fixture.root) + 1:]
        self.assertEqual(normalized, expected)
        receipt_path = folder / 'REFERENCE_NORMALIZATION.json'
        self.assertEqual(a.read(receipt_path)['normalized_ref_count'], 8)
        self.assertIn(a.binding(receipt_path), result['files'])
        a.verify(fixture.run, complete=True)
        receipt = a.read(receipt_path)
        receipt['facts'][0]['normalized_turn_ref'] = 'different'
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'asset_reference_normalization_replay_changed'):
            a.verify(fixture.run, complete=True)

    def test_alias_normalization_and_delimiters_equal_v65(self):
        import ast
        old = Path(a.__file__).parent.parent / 'h4_more_roots_composition_v65' / 'asset_generation.py'
        tree = ast.parse(old.read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'TASK': constants['TASK'] = ast.literal_eval(node.value)
                    if isinstance(target, ast.Tuple) and [x.id for x in target.elts] == ['START', 'END']:
                        constants['START'], constants['END'] = ast.literal_eval(node.value)
        self.assertNotEqual(a.TASK.encode(), constants['TASK'].encode())
        self.assertEqual((a.START, a.END), (constants['START'], constants['END']))
        current = ast.parse(Path(a.__file__).read_text())
        for name in ('normalize_assets', 'validate_assets', 'replay_normalization'):
            previous_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            current_function = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(current_function), ast.dump(previous_function))

    def test_v66_failed_responses_produce_only_mechanical_hints(self):
        previous=Path(a.__file__).parent.parent/'h4_evidence_pairs_v66'
        for slot in ('root103866_s02','root109164_s01','root122711_s01','root122711_s02'):
            run=previous/'assets'/slot
            source=a.read(run/'SOURCE.json')
            m={'root_id':source['root_id'],'slot_id':slot,'source_sha256':a.sha(run/'SOURCE.json')}
            for path in sorted((run/'requests').glob('*/RESPONSE.json')):
                with self.subTest(slot=slot,response=path.parent.name):
                    before=path.read_bytes();raw=a.read(path)['text']
                    accepted,errors,receipt=a.normalize_assets(raw,m,source)
                    self.assertIsNone(accepted);self.assertTrue(errors)
                    repair=a.build_repair(raw,errors,m,source)
                    self.assertEqual(repair['previous_response'],raw)
                    if slot=='root103866_s02':
                        self.assertTrue(all(':missing_keys:["contradiction_unavailable_reason"]' in e for e in errors))
                        self.assertEqual(repair['reference_hints'],[])
                    else:
                        self.assertTrue(any(':unknown_turn_ref:' in e for e in errors))
                        self.assertGreater(len(repair['reference_hints']),0)
                    for hint in repair['reference_hints']:
                        matches=[t for t in source['turns'] if t['speaker']==hint['speaker'] and hint['quote'] in t['text']]
                        self.assertEqual(len(matches),1)
                        self.assertEqual(hint['candidate_turn_ref'],matches[0]['turn_ref'])
                        original=json.loads(raw)['facts'][hint['fact_index']]['anchor']['turn_ref']
                        self.assertEqual(original,hint['original_turn_ref'])
                    self.assertEqual(path.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
