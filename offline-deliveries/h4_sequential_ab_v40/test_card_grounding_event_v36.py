"""Offline structural checks; all added binding/evidence values are test fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import event_state as scan

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / 'h4_sequential_ab_v34/runs/inherited_a3_b3_01'


def case(slot):
    prefix = {'S02': '013', 'S03': '019'}[slot]
    directory = next((FROZEN / 'requests').glob(prefix + '-*'))
    payload = json.loads((directory / 'REQUEST.json').read_text())
    c = json.loads(payload['messages'][0]['content'].split('EVENT_STATE_INPUT_JSON\n', 1)[1])
    scene = deepcopy(c['scene_plan'])
    actor_ids = sorted({r['actor_id'] for r in c['dataset_cards'][0]['steps']})
    # Synthetic declarations only. Never used to generate a production scene.
    if slot == 'S02': actor_ids = ['actor_1']
    scene['card_actor_bindings'] = [{'actor_id': i, 'participant': name}
        for i, name in zip(actor_ids, ['Emily', 'James'])]
    return c, scene


class GroundedEventTests(unittest.TestCase):
    def test_strict_request_keeps_history_binding_and_single_six_field_contract(self):
        c, scene = case('S02')
        req = scan.build_request(c['root_id'], c['route'], scene, c['cumulative_history'],
            c['dataset_cards'], c['latest_state'], require_goal_checks=True,
            require_card_checks=True, require_card_grounding=True)
        task, data = req['messages'][0]['content'].split('\nEVENT_STATE_INPUT_JSON\n')
        d = json.loads(data)
        self.assertEqual(d['cumulative_history'], c['cumulative_history'])
        self.assertEqual(d['scene_plan']['card_actor_bindings'], scene['card_actor_bindings'])
        check = d['expected_card_plan_checks'][0]
        self.assertEqual(check['source']['bound_participant'], 'Emily')
        self.assertEqual(check['allowed_evidence_development_indices'], [1])
        self.assertEqual([x['development_index'] for x in check['linked_developments']], [1])
        self.assertEqual(set(d['card_plan_checks_template'][0]),
            {'kind', 'id', 'verdict', 'reason', 'correction', 'evidence_development_indices'})
        self.assertIn(scan.CARD_GROUNDING_CONTRACT.strip(), task)
        self.assertNotIn('V33 B SELECTED-CARD COMPARISON CONTRACT', task)

    def test_step_cannot_cite_an_unmapped_development(self):
        c, scene = case('S02')
        raw = [{'kind': 'step', 'id': 's1', 'verdict': 'compatible',
                'reason': 'SYNTHETIC semantic assertion, not a model result.', 'correction': '',
                'evidence_development_indices': [2]}]
        errors = scan.validate_card_plan_checks(raw, c['route'], scene, c['dataset_cards'],
            required=True, require_grounding=True)
        self.assertTrue(errors)
        self.assertTrue(any('evidence' in e['code'] for e in errors))
        raw[0].update(verdict='revise', correction='SYNTHETIC restore mapped actor action.',
                      evidence_development_indices=[])
        self.assertEqual(scan.validate_card_plan_checks(raw, c['route'], scene, c['dataset_cards'],
            required=True, require_grounding=True), [])

    def test_dependency_cannot_pass_after_merely_proposed_fix(self):
        c, scene = case('S03')
        rows = scan.expected_card_plan_checks(scene, c['dataset_cards'], require_grounding=True)
        raw = [{'kind': x['kind'], 'id': x['id'], 'verdict': 'compatible',
                'reason': 'SYNTHETIC evidence only.', 'correction': '',
                'evidence_development_indices': [1, 2, 3, 4]} for x in rows]
        raw[1].update(verdict='revise', reason='SYNTHETIC missing endpoint.', correction='Restore endpoint.',
                      evidence_development_indices=[])
        errors = scan.validate_card_plan_checks(raw, c['route'], scene, c['dataset_cards'],
            required=True, require_grounding=True)
        self.assertTrue(any('dependency' in e['code'] for e in errors))
        dep = next(x for x in rows if x['kind'] == 'dependency' and x['id'] == 2)
        self.assertEqual(dep['source']['before_step']['bound_participant'], 'James')
        self.assertEqual(dep['source']['after_step']['bound_participant'], 'Emily')
        self.assertEqual(len(dep['before_linked_developments']), 4)

    def test_missing_bindings_do_not_get_inferred_from_speakers(self):
        c, scene = case('S02'); scene.pop('card_actor_bindings')
        with self.assertRaises(scan.EventStateValidationError):
            scan.expected_card_plan_checks(scene, c['dataset_cards'], require_grounding=True)


if __name__ == '__main__':
    unittest.main()
