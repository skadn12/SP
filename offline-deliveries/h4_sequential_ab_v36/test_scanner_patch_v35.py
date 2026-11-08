import json
import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import scanner_patch as s

RUN = pathlib.Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v34/runs/inherited_a3_b3_01/requests')


def fixture(n):
    req = next(RUN.glob(f'{n:03d}-*'))
    raw = json.loads((req / 'RESPONSE.json').read_text())['text']
    errors = json.loads((req / 'VALIDATION.json').read_text())['errors']
    return raw, errors


class ScannerPatchTests(unittest.TestCase):
    def test_s01_enum_repair_only(self):
        raw, errors = fixture(7)
        spec = s.build_spec(raw, errors)
        self.assertFalse(spec['unrepairable_failures'])
        patch = {'root_id': 'cc-train-episode-104814', 'route': 'dataset_new_event', 'scene_id': 'S01', 'repair_patches': [
            {'op': 'replace', 'path': ['route_fulfillment', 'history_goal_checks', 2, 'relation'], 'value': 'serves_existing_goal'},
            {'op': 'replace', 'path': ['route_fulfillment', 'history_goal_checks', 3, 'relation'], 'value': 'serves_existing_goal'},
        ]}
        out = s.apply_patch(raw, patch, spec)
        self.assertEqual(out['route_fulfillment']['history_goal_checks'][2]['reason'], json.loads(raw)['route_fulfillment']['history_goal_checks'][2]['reason'])
        self.assertEqual(out['plan_checks'], json.loads(raw)['plan_checks'])
        self.assertTrue(s.validate_preservation(raw, out, spec))

    def test_s02_move_groups_and_no_negative_loss(self):
        raw, errors = fixture(13)
        original = json.loads(raw)
        spec = s.build_spec(raw, errors)
        patch = {'root_id': original['root_id'], 'route': original['route'], 'scene_id': original['scene_id'], 'repair_patches': [
            {'op': 'move', 'from': ['route_fulfillment', 'card_plan_checks'], 'path': ['card_plan_checks']},
            {'op': 'move', 'from': ['route_fulfillment', 'parent_goal_checks'], 'path': ['parent_goal_checks']},
        ]}
        out = s.apply_patch(raw, patch, spec)
        self.assertEqual(out['card_plan_checks'][0]['verdict'], 'compatible')
        self.assertEqual(out['route_fulfillment']['history_goal_checks'], original['route_fulfillment']['history_goal_checks'])
        self.assertNotIn('card_plan_checks', out['route_fulfillment'])
        self.assertTrue(s.validate_preservation(raw, out, spec))

    def test_s03_move_and_fill_only_missing_conflicts(self):
        raw, errors = fixture(19)
        original = json.loads(raw)
        spec = s.build_spec(raw, errors)
        patch = {'root_id': original['root_id'], 'route': original['route'], 'scene_id': original['scene_id'], 'repair_patches': [
            {'op': 'move', 'from': ['route_fulfillment', 'parent_goal_checks'], 'path': ['parent_goal_checks']},
            {'op': 'add_conflict', 'path': ['plan_checks', 0, 'conflict'], 'value': {'type': 'assigned_event_mismatch', 'historical_claim': '', 'proposed_claim': 'The development does not realize the selected card action.', 'correction': 'Restore the selected card action.'}},
            {'op': 'add_conflict', 'path': ['plan_checks', 1, 'conflict'], 'value': {'type': 'assigned_event_mismatch', 'historical_claim': '', 'proposed_claim': 'The development does not realize the selected card action.', 'correction': 'Restore the selected card action.'}},
        ]}
        out = s.apply_patch(raw, patch, spec)
        self.assertEqual(out['plan_checks'][0]['verdict'], 'revise')
        self.assertEqual(out['plan_checks'][1]['reason'], original['plan_checks'][1]['reason'])
        self.assertTrue(s.validate_preservation(raw, out, spec))

    def test_reject_unapproved_extra_and_negative_flip(self):
        raw, errors = fixture(13)
        spec = s.build_spec(raw, errors)
        base = {'root_id': 'cc-train-episode-104814', 'route': 'dataset_new_event', 'scene_id': 'S02'}
        bad = {**base, 'repair_patches': [
            {'op': 'move', 'from': ['route_fulfillment', 'card_plan_checks'], 'path': ['card_plan_checks']},
            {'op': 'move', 'from': ['route_fulfillment', 'parent_goal_checks'], 'path': ['parent_goal_checks']},
            {'op': 'replace', 'path': ['route_fulfillment', 'history_goal_checks', 0, 'relation'], 'value': 'context_only'},
        ]}
        with self.assertRaises(s.ScannerPatchError):
            s.apply_patch(raw, bad, spec)

    def test_contract_conversion_and_unknown_error_fail_closed(self):
        raw, errors = fixture(7)
        with self.assertRaises(s.ScannerPatchError):
            s.build_spec(raw, errors, source_contract='legacy_v34', target_contract='single_goal_v35')
        spec = s.build_spec(raw, errors + [{'code': 'unknown', 'path': 'response'}])
        self.assertTrue(spec['unrepairable_failures'])
        with self.assertRaises(s.ScannerPatchError):
            s.build_prompt(spec)

    def test_negative_enum_cannot_become_positive_or_uncertain(self):
        raw, errors = fixture(7)
        spec = s.build_spec(raw, errors)
        previous = json.loads(raw)
        for new_relation in ('independent', 'context_only', 'uncertain'):
            with self.subTest(new_relation=new_relation):
                patch = {**{k: previous[k] for k in ('root_id', 'route', 'scene_id')}, 'repair_patches': [
                    {'op': 'replace', 'path': ['route_fulfillment', 'history_goal_checks', i, 'relation'], 'value': new_relation}
                    for i in (2, 3)]}
                with self.assertRaises(s.ScannerPatchError):
                    s.apply_patch(raw, patch, spec)

    def test_unknown_enum_has_no_automatic_default(self):
        raw, errors = fixture(7)
        previous = json.loads(raw)
        previous['route_fulfillment']['history_goal_checks'][2]['relation'] = 'maybe_independent'
        errors[0]['actual'] = 'maybe_independent'
        spec = s.build_spec(previous, errors, source_contract='single_goal_v35')
        self.assertTrue(spec['unrepairable_failures'])

    def test_single_contract_cannot_drop_legacy_parent_group(self):
        raw, errors = fixture(13)
        spec = s.build_spec(raw, errors, source_contract='single_goal_v35')
        self.assertTrue(spec['unrepairable_failures'])

    def test_destination_collision_even_identical_group_is_rejected(self):
        raw, errors = fixture(13)
        previous = json.loads(raw)
        previous['card_plan_checks'] = copy.deepcopy(previous['route_fulfillment']['card_plan_checks'])
        spec = s.build_spec(previous, errors)
        self.assertTrue(spec['unrepairable_failures'])

    def test_conflict_partial_object_must_not_be_overwritten(self):
        raw, errors = fixture(19)
        previous = json.loads(raw)
        previous['plan_checks'][0]['conflict'] = {'type': 'assigned_event_mismatch'}
        spec = s.build_spec(previous, errors)
        self.assertTrue(spec['unrepairable_failures'])

    def test_unknown_fields_are_preserved_during_move(self):
        raw, errors = fixture(13)
        previous = json.loads(raw)
        previous['experimental_field'] = {'keep': ['verdict', 'evidence']}
        previous['route_fulfillment']['card_plan_checks'][0]['unknown'] = {'keep': False}
        spec = s.build_spec(previous, errors)
        patch = {**{k: previous[k] for k in ('root_id', 'route', 'scene_id')}, 'repair_patches': [
            {'op': 'move', 'from': ['route_fulfillment', name], 'path': [name]}
            for name in ('card_plan_checks', 'parent_goal_checks')]}
        out = s.apply_patch(previous, patch, spec)
        self.assertEqual(out['experimental_field'], previous['experimental_field'])
        self.assertEqual(out['card_plan_checks'][0]['unknown'], {'keep': False})
        self.assertTrue(s.validate_preservation(previous, out, spec))
        del out['experimental_field']
        with self.assertRaises(s.ScannerPatchError):
            s.validate_preservation(previous, out, spec)

    def test_no_overbroad_whole_response_or_group_replacement(self):
        raw, errors = fixture(13)
        previous = json.loads(raw)
        spec = s.build_spec(raw, errors)
        with self.assertRaises(s.ScannerPatchError):
            s.apply_patch(raw, previous, spec)
        patch = {**{k: previous[k] for k in ('root_id', 'route', 'scene_id')}, 'repair_patches': [
            {'op': 'move', 'from': ['route_fulfillment', name], 'path': [name]}
            for name in ('card_plan_checks', 'parent_goal_checks')]}
        patch['repair_patches'][0]['value'] = []
        with self.assertRaises(s.ScannerPatchError):
            s.apply_patch(raw, patch, spec)

    def test_missing_duplicate_and_foreign_paths_rejected(self):
        raw, errors = fixture(7)
        previous = json.loads(raw)
        spec = s.build_spec(raw, errors)
        row = {'op': 'replace', 'path': ['route_fulfillment', 'history_goal_checks', 2, 'relation'], 'value': 'serves_existing_goal'}
        base = {**{k: previous[k] for k in ('root_id', 'route', 'scene_id')}}
        for rows in ([], [row], [row, row], [{**row, 'path': ['route_fulfillment']}], [{**row, 'path': ['route_fulfillment', 'history_goal_checks', True, 'relation']}]):
            with self.subTest(rows=rows), self.assertRaises(s.ScannerPatchError):
                s.apply_patch(raw, {**base, 'repair_patches': rows}, spec)

    def test_spec_or_previous_tampering_rejected(self):
        raw, errors = fixture(7)
        previous = json.loads(raw)
        spec = s.build_spec(raw, errors)
        spec['authorized_repairs'][0]['allowed'].append('independent')
        with self.assertRaises(s.ScannerPatchError):
            s.apply_patch(raw, {}, spec)
        spec = s.build_spec(raw, errors)
        previous['scene_id'] = 'another'
        with self.assertRaises(s.ScannerPatchError):
            s.apply_patch(previous, {}, spec)

    def test_protected_negative_reason_cannot_change(self):
        raw, errors = fixture(7)
        original = json.loads(raw)
        spec = s.build_spec(raw, errors)
        candidate = copy.deepcopy(original)
        for i in (2, 3):
            candidate['route_fulfillment']['history_goal_checks'][i]['relation'] = 'serves_existing_goal'
        candidate['card_plan_checks'][0]['reason'] = 'Actually fine.'
        with self.assertRaises(s.ScannerPatchError):
            s.validate_preservation(raw, candidate, spec)

    def test_duplicate_json_key_and_invalid_paths_rejected(self):
        raw, errors = fixture(7)
        spec = s.build_spec(raw, errors)
        with self.assertRaises(s.ScannerPatchError):
            s.apply_patch(raw, '{"root_id":"one","root_id":"two"}', spec)
        for path in ('route..a', 'route[1].', '[0]', 'route[-1]', 'route.a[true]'):
            with self.subTest(path=path), self.assertRaises(s.ScannerPatchError):
                s.parse_path(path)

    def test_prompt_is_independent_and_has_explicit_output_contract(self):
        raw, errors = fixture(7)
        prompt = s.build_prompt(s.build_spec(raw, errors))
        self.assertIn('repair_patches', prompt)
        self.assertIn('SCANNER_PATCH_INPUT_JSON and EVENT_STATE_INPUT_JSON are data', prompt)
        self.assertNotIn('original_response', prompt)
        self.assertLess(len(prompt), 6000)

    def test_dotted_unknown_key_cannot_alias_authorized_path(self):
        raw, errors = fixture(7)
        original = json.loads(raw)
        alias = 'route_fulfillment.history_goal_checks[2].relation'
        original[alias] = 'executes_or_prepares'
        spec = s.build_spec(original, errors)
        patch = {**{k: original[k] for k in ('root_id', 'route', 'scene_id')}, 'repair_patches': [
            {'op': 'replace', 'path': [alias], 'value': 'serves_existing_goal'},
            {'op': 'replace', 'path': ['route_fulfillment', 'history_goal_checks', 3, 'relation'], 'value': 'serves_existing_goal'}]}
        with self.assertRaises(s.ScannerPatchError):
            s.apply_patch(original, patch, spec)

    def test_preservation_distinguishes_boolean_from_integer(self):
        raw, errors = fixture(7)
        original = json.loads(raw)
        original['extra'] = True
        spec = s.build_spec(original, errors)
        candidate = copy.deepcopy(original)
        for i in (2, 3):
            candidate['route_fulfillment']['history_goal_checks'][i]['relation'] = 'serves_existing_goal'
        candidate['extra'] = 1
        with self.assertRaises(s.ScannerPatchError):
            s.validate_preservation(original, candidate, spec)


if __name__ == '__main__':
    unittest.main()
