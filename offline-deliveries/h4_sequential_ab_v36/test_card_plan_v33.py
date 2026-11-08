"""Selected-card omissions must reach the existing pre-realization replan gate."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import event_state as scan
import test_event_v19 as event_fixture


FROZEN = (Path(__file__).resolve().parent.parent / "h4_sequential_ab_v32" /
          "runs/inherited_a2_b3_01/requests/015-cc-train-episode-104814-B-S02-event-state-1-1")


class CardPlanV33Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        request = json.loads((FROZEN / "REQUEST.json").read_text())
        cls.context = json.loads(request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        cls.old_response = json.loads(json.loads((FROZEN / "RESPONSE.json").read_text())["text"])

    def setUp(self):
        self.context = deepcopy(type(self).context)
        self.raw = deepcopy(self.old_response)

    def validate(self, raw=None, **kwargs):
        c = self.context
        return scan.validate(self.raw if raw is None else raw, c["root_id"], c["route"],
            c["scene_plan"], c["cumulative_history"], dataset_cards=c["dataset_cards"],
            require_goal_checks=True, require_card_checks=True, **kwargs)

    def declared_checks(self):
        # Synthetic declarations exercise contracts, not a claim of semantic truth.
        expected = scan.expected_card_plan_checks(self.context["scene_plan"], self.context["dataset_cards"])
        return [{"kind": row["kind"], "id": row["id"], "verdict": "compatible",
                 "reason": "Synthetic schema-control declaration.", "correction": ""} for row in expected]

    def with_negative(self):
        self.raw["card_plan_checks"] = self.declared_checks()
        self.raw["card_plan_checks"][0].update(verdict="revise",
            reason="The selected refusal to return to school is absent from the automation visit.",
            correction="Plan an explicit refusal and peer response preserving the school-refusal premise.")
        return self.raw

    def test_frozen_pass_without_card_comparisons_cannot_pass_strict_gate(self):
        c = self.context
        legacy = scan.validate(self.raw, c["root_id"], c["route"], c["scene_plan"],
                               c["cumulative_history"], require_goal_checks=True)
        self.assertFalse(scan.requires_replan(legacy))
        before = deepcopy(self.raw)
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate()
        self.assertIn("card_plan_checks_required", {e["code"] for e in caught.exception.errors})
        self.assertEqual(self.raw, before)

    def test_request_binds_every_selected_source_and_keeps_full_history(self):
        c = self.context
        before = deepcopy(c)
        request = scan.build_request(c["root_id"], c["route"], c["scene_plan"],
            c["cumulative_history"], c["dataset_cards"], c["latest_state"], require_card_checks=True)
        content = request["messages"][0]["content"]
        output = json.loads(content.split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        self.assertEqual(output["cumulative_history"], c["cumulative_history"])
        rows = output["expected_card_plan_checks"]
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["source"], {**c["dataset_cards"][0]["steps"][0],
                                           "actor_role": "rebellious youth"})
        self.assertEqual(rows[0]["linked_developments"][0]["text"], c["scene_plan"]["developments"][0])
        self.assertEqual(rows[-3]["source"]["before_step"], rows[0]["source"])
        self.assertEqual(c, before)

    def test_missing_duplicate_and_wrong_typed_reference_never_pass(self):
        self.raw["card_plan_checks"] = self.declared_checks()
        variants = []
        missing = deepcopy(self.raw); missing["card_plan_checks"].pop()
        variants.append((missing, "card_plan_check_coverage_invalid"))
        duplicate = deepcopy(self.raw); duplicate["card_plan_checks"].append(deepcopy(duplicate["card_plan_checks"][0]))
        variants.append((duplicate, "card_plan_check_duplicate_identity"))
        wrong = deepcopy(self.raw); wrong["card_plan_checks"][-3]["id"] = True
        variants.append((wrong, "card_plan_check_identity_invalid"))
        for value, code in variants:
            with self.subTest(code=code), self.assertRaises(scan.EventStateValidationError) as caught:
                self.validate(value)
            self.assertIn(code, {e["code"] for e in caught.exception.errors})

    def test_card_negative_is_binding_despite_all_history_and_route_positives(self):
        self.with_negative()
        checked = self.validate()
        self.assertTrue(checked["route_fulfillment"]["satisfied"])
        self.assertTrue(scan.requires_replan(checked))
        binding = scan.binding_plan_checks(checked)
        self.assertEqual(len(binding), 1)
        self.assertTrue(binding[0]["card_plan_check"])
        self.assertEqual((binding[0]["kind"], binding[0]["id"]), ("step", "s1"))
        self.assertNotIn("development_index", binding[0])
        self.assertNotIn("constraint_ids", binding[0])

    def test_selected_seed_subset_does_not_require_unselected_steps(self):
        c = self.context
        c["scene_plan"]["card_development_links"] = [{"card_step_ids": ["s1"],
            "dependency_indices": [], "development_indices": [1]}]
        self.raw["card_plan_checks"] = self.declared_checks()
        self.assertEqual(len(self.raw["card_plan_checks"]), 1)
        # This demonstrates structural coverage only; a model must judge the core premise.
        self.assertFalse(scan.requires_replan(self.validate()))

    def test_negative_requires_actionable_correction_and_positive_requires_empty(self):
        self.with_negative()
        for value in ("", None):
            candidate = deepcopy(self.raw); candidate["card_plan_checks"][0]["correction"] = value
            with self.subTest(value=value), self.assertRaises(scan.EventStateValidationError) as caught:
                self.validate(candidate)
            self.assertIn("card_plan_check_correction_invalid", {e["code"] for e in caught.exception.errors})
        candidate = deepcopy(self.raw); candidate["card_plan_checks"][1]["correction"] = "Unrequested rewrite"
        with self.assertRaises(scan.EventStateValidationError):
            self.validate(candidate)

    def test_format_repair_can_fix_missing_field_but_cannot_erase_negative(self):
        self.with_negative()
        self.raw["card_plan_checks"][0]["correction"] = ""
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate()
        repair = scan.build_repair(caught.exception, self.raw, self.context["cumulative_history"])
        self.assertEqual(len(repair["original_negative_card_checks"]), 1)
        fixed = deepcopy(self.raw)
        fixed["card_plan_checks"][0]["correction"] = "Restore refusal and the peer's response."
        self.assertTrue(scan.validate_repair_preservation(self.raw, fixed, repair))
        self.validate(fixed)
        fixed["card_plan_checks"][0].update(verdict="compatible", correction="")
        with self.assertRaises(scan.EventStateValidationError) as rejected:
            scan.validate_repair_preservation(self.raw, fixed, repair)
        self.assertIn("event_state_repair_changed_negative_card_check",
                      {e["code"] for e in rejected.exception.errors})

    def test_timeline_patch_preserves_all_card_checks(self):
        self.with_negative()
        original_evidence = deepcopy(self.raw["timeline"][0]["evidence"])
        self.raw["timeline"][0]["evidence"] = []
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate()
        repair = scan.build_repair(caught.exception, self.raw, self.context["cumulative_history"])
        self.assertEqual(repair["repair_mode"], "timeline_entry_patch")
        patch = {"timeline_repairs": [{"constraint_id": "c1", "choice": "historical_claim",
                                       "evidence": original_evidence}], "plan_check_repairs": []}
        merged = scan.merge_event_repair(self.raw, patch, repair, self.context["cumulative_history"])
        self.assertEqual(merged["card_plan_checks"], self.raw["card_plan_checks"])
        self.validate(merged)
        merged["card_plan_checks"][0]["verdict"] = "compatible"
        with self.assertRaises(scan.EventStateValidationError):
            scan.validate_repair_preservation(self.raw, merged, repair)

    def test_strict_card_contract_is_B_only(self):
        f = event_fixture.EventV19Tests(); f.setUp()
        with self.assertRaisesRegex(ValueError, "card_plan_checks_B_only"):
            scan.build_request(f.root, f.route, f.scene, f.history, [], {}, require_card_checks=True)
        with self.assertRaises(scan.EventStateValidationError) as caught:
            scan.validate(f.result, f.root, f.route, f.scene, f.history, require_card_checks=True)
        self.assertIn("card_plan_checks_B_only", {e["code"] for e in caught.exception.errors})


if __name__ == "__main__":
    unittest.main()
