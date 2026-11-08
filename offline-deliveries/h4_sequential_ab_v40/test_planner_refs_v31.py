"""Reference-only anchors must bind real history without correcting bad quotes."""
from copy import deepcopy
import unittest

import runner
import planner_contract
import test_planner_v19


class PlannerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_planner_v19.PlannerV19Tests()
        self.fixture.setUp()

    def test_reference_only_materializes_without_mutating_model_plan(self):
        f = self.fixture
        plan, state = f.fixture(runner.ROUTES[0])
        plan['scenes'][0]['source_anchors'] = [{'turn_ref': 'P:T001'}]
        original = deepcopy(plan)
        result = f.validate(plan, state)
        self.assertEqual(plan, original)
        self.assertEqual(result['scenes'][0]['source_anchors'], [
            {'turn_ref': 'P:T001', 'quote': f.history[0]['text']}])
        receipt = runner.evidence_refs.citation_provenance(original, f.history)
        self.assertEqual(receipt['dereferenced_citations'][0]['selection'], 'model_selected_reference')
        self.assertFalse(receipt['semantic_support_verified'])

    def test_unknown_id_is_not_searched_or_corrected(self):
        f = self.fixture
        plan, state = f.fixture(runner.ROUTES[0])
        plan['scenes'][0]['source_anchors'] = [{'turn_ref': 'T001'}]
        with self.assertRaisesRegex(ValueError, 'citation_reference_not_in_context'):
            f.validate(plan, state)

    def test_supplied_bad_quote_still_rejected(self):
        f = self.fixture
        plan, state = f.fixture(runner.ROUTES[0])
        plan['scenes'][0]['source_anchors'][0]['quote'] = 'I will attend an interview...'
        with self.assertRaisesRegex(ValueError, 'anchor_quote_not_exact'):
            f.validate(plan, state)

    def test_generated_shape_does_not_ask_model_to_copy_quote(self):
        shape = planner_contract.planner_shape_example(runner.ROUTES[0])
        self.assertEqual(set(shape['scenes'][0]['source_anchors'][0]), {'turn_ref'})


if __name__ == '__main__':
    unittest.main()
