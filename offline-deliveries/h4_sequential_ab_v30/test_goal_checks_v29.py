"""v29 explicit per-timeline old-goal comparisons and B route gate."""
from copy import deepcopy
import unittest

import event_state as scan
import test_event_v19 as event_fixture


class HistoryGoalChecksV29Tests(unittest.TestCase):
    def setUp(self):
        f = event_fixture.EventV19Tests()
        f.setUp()
        self.f = f
        self.f.as_b(independent=True)
        self.f.result["route_fulfillment"].pop("satisfied")
        # Add a second historical state so coverage cannot pass by checking one
        # salient row and omitting another.
        self.f.history.append({"turn_ref": "R02:B:S01:T004",
                               "text": "I am planning the family dinner this weekend."})
        self.f.result["segments"].append({"segment_id": "R02:B:S01", "none_reason": ""})
        self.f.result["timeline"].append({"constraint_id": "c2", "status": "ongoing",
            "evidence_basis": "reported", "statement": "James plans a family dinner.",
            "evidence": [{"turn_ref": "R02:B:S01:T004"}],
            "implication": "A later dinner task may support this goal."})
        self.f.result["route_fulfillment"]["history_goal_checks"] = [
            {"constraint_id": "c1", "historical_goal": "The planned weekend trip.",
             "relation": "context_only", "reason": "Wallet recovery is independent of the trip's itinerary."},
            {"constraint_id": "c2", "historical_goal": "The family dinner plan.",
             "relation": "context_only", "reason": "The dinner is only background context for this scene."},
        ]

    def validate(self, result=None, *, strict=True):
        return scan.validate(self.f.result if result is None else result,
            self.f.root, self.f.route, self.f.scene, self.f.history,
            require_goal_checks=strict)

    def test_every_timeline_id_is_required(self):
        result = deepcopy(self.f.result)
        result["route_fulfillment"]["history_goal_checks"].pop()
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(result)
        self.assertIn("history_goal_check_coverage_invalid",
                      {error["code"] for error in caught.exception.errors})

    def test_independent_and_context_only_rows_pass(self):
        checked = self.validate()
        self.assertTrue(scan.route_fulfilled(checked["route_fulfillment"], "dataset_new_event",
                                             timeline=checked["timeline"], require_goal_checks=True))
        self.assertFalse(scan.requires_replan(checked))
        provenance = scan.route_decision_provenance(checked["route_fulfillment"], "dataset_new_event",
            timeline=checked["timeline"], require_goal_checks=True)
        self.assertTrue(provenance["history_goal_policy"]["global_satisfied"])
        self.assertEqual(provenance["history_goal_policy"]["blocking_checks"], [])

    def test_serves_old_goal_vetoes_global_distinct(self):
        result = deepcopy(self.f.result)
        result["route_fulfillment"]["history_goal_checks"][1]["relation"] = "serves_existing_goal"
        checked = self.validate(result)
        self.assertFalse(checked["route_fulfillment"]["satisfied"])
        self.assertTrue(scan.requires_replan(checked))
        diagnostics = scan.plan_diagnostics(checked)
        self.assertEqual(diagnostics[-1]["constraint_ids"], ["c2"])
        self.assertEqual(diagnostics[-1]["history_goal_checks"][0]["relation"], "serves_existing_goal")
        self.assertFalse(scan.route_fulfilled(checked["route_fulfillment"], "dataset_new_event",
                                              timeline=checked["timeline"], require_goal_checks=True))

    def test_uncertain_also_vetoes_and_is_not_inferred(self):
        result = deepcopy(self.f.result)
        result["route_fulfillment"]["history_goal_checks"][0]["relation"] = "uncertain"
        checked = self.validate(result)
        self.assertFalse(checked["route_fulfillment"]["satisfied"])
        provenance = scan.route_decision_provenance(checked["route_fulfillment"], "dataset_new_event",
            timeline=checked["timeline"], require_goal_checks=True)
        self.assertEqual([row["constraint_id"] for row in provenance["history_goal_policy"]["blocking_checks"]], ["c1"])

    def test_relation_values_and_duplicate_ids_are_explicit(self):
        result = deepcopy(self.f.result)
        result["route_fulfillment"]["history_goal_checks"][0]["relation"] = "not_a_goal"
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(result)
        self.assertIn("history_goal_check_relation_invalid",
                      {error["code"] for error in caught.exception.errors})
        result = deepcopy(self.f.result)
        result["route_fulfillment"]["history_goal_checks"][1]["constraint_id"] = "c1"
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(result)
        self.assertIn("history_goal_check_duplicate_id",
                      {error["code"] for error in caught.exception.errors})

    def test_legacy_non_strict_validation_remains_compatible(self):
        result = deepcopy(self.f.result)
        result["route_fulfillment"].pop("history_goal_checks")
        checked = self.validate(result, strict=False)
        self.assertTrue(checked["route_fulfillment"]["satisfied"])

    def test_missing_comparisons_are_not_filled_or_accepted(self):
        result = deepcopy(self.f.result)
        result["route_fulfillment"].pop("history_goal_checks")
        before = deepcopy(result)
        with self.assertRaises(scan.EventStateValidationError):
            self.validate(result)
        self.assertEqual(result, before)
        with self.assertRaisesRegex(ValueError, "history_goal_checks_timeline_required"):
            scan.compile_route_fulfillment(self.f.result["route_fulfillment"], self.f.route,
                                           require_goal_checks=True)

    def test_original_global_judgments_unchanged_by_veto(self):
        raw = deepcopy(self.f.result)
        raw["route_fulfillment"]["history_goal_checks"][1].update(
            relation="serves_existing_goal", reason="Wine substitution supports that dinner.")
        before = deepcopy(raw)
        checked = self.validate(raw)
        self.assertEqual(raw, before)
        for field in ("relation_to_history", "serves_existing_event", "reason", "correction"):
            self.assertEqual(checked["route_fulfillment"][field], raw["route_fulfillment"][field])
        self.assertFalse(checked["route_fulfillment"]["satisfied"])

    def test_format_repair_cannot_erase_serving_or_uncertain_judgment(self):
        for relation in ("serves_existing_goal", "uncertain"):
            raw = deepcopy(self.f.result)
            raw["route_fulfillment"]["history_goal_checks"][0]["relation"] = relation
            raw["route_fulfillment"]["history_goal_checks"].pop()
            with self.assertRaises(scan.EventStateValidationError) as caught:
                self.validate(raw)
            repair = scan.build_repair(caught.exception, raw, self.f.history)
            fixed = deepcopy(raw)
            fixed["route_fulfillment"]["history_goal_checks"].append(
                deepcopy(self.f.result["route_fulfillment"]["history_goal_checks"][1]))
            self.assertTrue(scan.validate_repair_preservation(raw, fixed, repair))
            self.assertTrue(scan.requires_replan(self.validate(fixed)))
            fixed["route_fulfillment"]["history_goal_checks"][0]["relation"] = "independent"
            with self.assertRaises(scan.EventStateValidationError) as rejected:
                scan.validate_repair_preservation(raw, fixed, repair)
            self.assertIn("event_state_repair_changed_negative_goal_comparison",
                          {error["code"] for error in rejected.exception.errors})

    def test_nonhistorical_patch_archives_and_remaps_comparisons_losslessly(self):
        raw = deepcopy(self.f.result)
        # The model originally treated a proposed fact as historical. The local
        # patch explicitly relocates it; comparisons must follow that same ID.
        raw["timeline"].insert(1, {"constraint_id": "c2", "status": "uncertain",
            "evidence_basis": "inferred", "statement": "The new wallet loss is proposed.",
            "evidence": [], "implication": "This is future scene work."})
        raw["timeline"][2]["constraint_id"] = "c3"
        goals = raw["route_fulfillment"]["history_goal_checks"]
        goals[1]["constraint_id"] = "c3"
        goals.insert(1, {"constraint_id": "c2", "historical_goal": "Proposed wallet recovery.",
                        "relation": "uncertain", "reason": "This has no historical support."})
        goals[2].update(relation="serves_existing_goal", reason="Current main work serves the dinner.")
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(raw)
        repair = scan.build_repair(caught.exception, raw, self.f.history)
        self.assertEqual(repair["repair_mode"], "timeline_entry_patch")
        patch = {"timeline_repairs": [{"constraint_id": "c2", "choice": "planned_development"}],
                 "plan_check_repairs": []}
        merged = scan.merge_event_repair(raw, patch, repair, self.f.history)
        checked = self.validate(merged)
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(merged["nonhistorical_goal_checks"][0]["entry"], goals[1])
        self.assertEqual(merged["route_fulfillment"]["history_goal_checks"][1],
                         {**goals[2], "constraint_id": "c2"})
        self.assertEqual(merged["route_fulfillment"]["relation_to_history"], "distinct_event")
        self.assertTrue(scan.validate_repair_preservation(raw, merged, repair))
        receipt = scan.event_repair_provenance(raw, patch, repair, merged)
        self.assertTrue(receipt["history_goal_checks"]["semantic_fields_unchanged"])
        broken = deepcopy(merged)
        broken["nonhistorical_goal_checks"][0]["entry"]["relation"] = "independent"
        with self.assertRaises(scan.EventStateValidationError):
            scan.validate_repair_preservation(raw, broken, repair)

    def test_new_B_prompt_requires_all_timeline_comparisons(self):
        template = scan.route_output_template(self.f.route, timeline=self.f.result["timeline"],
                                              require_goal_checks=True)
        self.assertEqual([row["constraint_id"] for row in template["history_goal_checks"]], ["c1", "c2"])
        self.assertIn("EVERY timeline c-ID", scan.HISTORY_GOAL_CONTRACT)
        self.assertNotIn("history_goal_checks", scan.route_output_template("old_event_continuation",
                                                                          require_goal_checks=True))


if __name__ == "__main__":
    unittest.main()
