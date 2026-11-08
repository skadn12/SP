"""Contract regressions using frozen model responses; no production judgments."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import event_state as scan

BASE = Path(__file__).resolve().parent.parent
CONTROL = BASE / 'h4_sequential_ab_v34/probes/frozen_parent_goal_repair_01'


class SingleGoalV35Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = json.loads((CONTROL / 'cases/event_s02_card_replacement_old_goal.json').read_text())
        result = json.loads((CONTROL / 'results/event_s02_card_replacement_old_goal.json').read_text())
        directory = Path(result['requests'][-1])
        cls.raw = json.loads(json.loads((directory / 'RESPONSE.json').read_text())['text'])

    def request(self, repair=None):
        c = self.case['inputs']
        return scan.build_request(c['root_id'], c['route'], c['scene_plan'], c['history'],
            [c['dataset_card']], c['latest_state'], repair=repair, purpose_input=c['purpose_input'],
            require_goal_checks=True, require_card_checks=True)

    def validate(self, value):
        c = self.case['inputs']
        return scan.validate(value, c['root_id'], c['route'], c['scene_plan'], c['history'],
            require_goal_checks=True, dataset_cards=[c['dataset_card']], require_card_checks=True)

    def test_complete_contract_no_conflicting_root_inventory(self):
        content = self.request()['messages'][0]['content']
        trusted, encoded = content.split('\nEVENT_STATE_INPUT_JSON\n')
        data = json.loads(encoded)
        self.assertEqual(data['cumulative_history'], self.case['inputs']['history'])
        root = next(line for line in trusted.splitlines() if line.startswith('Root required fields:'))
        self.assertIn('card_plan_checks', root)
        self.assertNotIn('parent_goal_checks', root)
        nested = next(line for line in trusted.splitlines() if line.startswith('route_fulfillment fields:'))
        self.assertIn('history_goal_checks', nested)
        self.assertNotIn('card_plan_checks', nested)
        self.assertNotIn('parent_goal_checks_template', data)
        self.assertNotIn('V34 B CONCRETE PARENT-OBJECTIVE', trusted)
        self.assertNotIn('Return {root_id,route,scene_id,segments,timeline,plan_checks,route_fulfillment}', trusted)

    def test_duplicate_goal_group_is_not_accepted_in_single_contract(self):
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(self.raw)
        self.assertIn('single_goal_contract_duplicate_group', {r['code'] for r in caught.exception.errors})

    def test_existing_card_negatives_remain_binding_without_duplicate_group(self):
        value = deepcopy(self.raw)
        value.pop('parent_goal_checks')
        checked = self.validate(value)
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(len([r for r in scan.binding_plan_checks(checked) if r.get('card_plan_check')]), 9)
        self.assertEqual(value['card_plan_checks'], self.raw['card_plan_checks'])

    def test_unsupported_schema_error_cannot_trigger_full_rewrite(self):
        error = scan.EventStateValidationError([{'code': 'event_state_text_required',
            'path': 'timeline[0].statement', 'actual': ''}])
        spec = scan.build_local_repair(error, self.raw, self.case['inputs']['history'])
        self.assertTrue(spec['unrepairable_failures'])
        with self.assertRaisesRegex(ValueError, 'scanner_local_repair_unsupported'):
            self.request(repair=spec)


if __name__ == '__main__':
    unittest.main()
