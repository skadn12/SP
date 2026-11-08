"""Seed selection preserves original IDs without manufacturing Qwen judgments.

Synthetic semantic verdicts exercise admission contracts, not model accuracy.
No provider calls or historical artifact writes are performed.
"""
from copy import deepcopy
import json
import unittest

import event_state as scan
import realization_review as review
from planner_contract import validate_mapping


class SeedReviewV24Tests(unittest.TestCase):
    def setUp(self):
        self.root, self.route = "fixture", "dataset_new_event"
        self.card = {
            "card_id": "seed-recovery", "guidance_level": "seed",
            "source_metadata": {"quote": "SOURCE_ONLY_MUST_NOT_ENTER_ROOT_CONTEXT"},
            "actors": [{"actor_id": "a1", "role": "owner"}],
            "steps": [
                {"step_id": "s1", "actor_id": "a1", "action": "Report lost property", "state": "reported"},
                {"step_id": "s2", "actor_id": "a1", "action": "Arrange property search", "state": "planned"},
                {"step_id": "s3", "actor_id": "a1", "action": "Collect recovered property", "state": "completed"},
            ],
            "dependencies": [
                {"before": "s1", "after": "s2", "relation": "report before search"},
                {"before": "s2", "after": "s3", "relation": "search before recovery"},
            ],
        }
        self.scene = {
            "scene_id": "S01", "dataset_card_id": self.card["card_id"],
            "goal": "Start a lost-property recovery effort", "source_anchors": [],
            "developments": ["Report a newly lost wallet.", "Arrange to search after the old interview."],
            "card_development_links": [
                {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1]},
                {"card_step_ids": ["s2"], "dependency_indices": [1], "development_indices": [2]},
            ],
            "state_transition": {"basis_fact_ids": [], "basis_reason": "Plan explanation only.",
                                 "temporal_placement": "after the existing interview"},
        }
        self.history = [{"turn_ref": "P01:T001", "text": "I will attend the interview tomorrow."}]
        self.realization = {"root_id": self.root, "route": self.route, "scene_id": "S01", "status": "scene",
            "turns": [{"speaker": "Emily", "text": "I just lost my wallet. Let us search after the interview."}]}
        evidence = [{"turn_ref": "NEW:T001"}]
        self.raw = {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "checks": [{"development_index": i, "realized": True, "evidence": deepcopy(evidence)} for i in (1, 2)],
            "card_checks": [{"kind": k, "id": i, "realized": True, "evidence": deepcopy(evidence)}
                            for k, i in (("step", "s1"), ("step", "s2"), ("dependency", 1))],
            "new_event_or_progress": True, "consistent_with_history": True,
            "duplicates_existing_event": False, "reason": "SYNTHETIC: recovery has begun but not completed.",
            "route_fulfillment": {
                "role": self.route, "primary_event": "Lost wallet recovery", "primary_goal": "Recover property",
                "relation_to_history": "distinct_event", "serves_existing_event": False,
                "existing_event": "", "new_progress": True,
                "reason": "SYNTHETIC: lost property preserves the seed premise and has its own goal.",
                "correction": "", "history_evidence": [], "new_evidence": deepcopy(evidence),
            },
        }

    def validate(self, raw=None, card=None):
        return review.validate_review(self.raw if raw is None else raw, self.root, self.route,
            self.scene, self.realization, dataset_card=self.card if card is None else card,
            history=self.history)

    def test_seed_subset_passes_without_requiring_unselected_procedure(self):
        original_card, original_raw = deepcopy(self.card), deepcopy(self.raw)
        result = self.validate()
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["semantic_certified"])
        self.assertEqual(self.card, original_card)
        self.assertEqual(self.raw, original_raw)
        self.assertEqual(len(result["parsed"]["card_checks"]), 3)

    def test_scaffold_keeps_full_coverage_requirement(self):
        scaffold = deepcopy(self.card)
        scaffold["guidance_level"] = "scaffold"
        with self.assertRaisesRegex(ValueError, "card_steps_not_fully_mapped"):
            validate_mapping(self.scene, scaffold)
        result = self.validate(card=scaffold)
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(any("review_card_coverage_invalid" in item for item in result["schema_failures"]))
        complete = deepcopy(self.raw)
        for kind, identifier in (("step", "s3"), ("dependency", 2)):
            complete["card_checks"].append({"kind": kind, "id": identifier, "realized": True,
                                          "evidence": [{"turn_ref": "NEW:T001"}]})
        # Scripted full judgments test the reference contract only, not whether
        # the deliberately insufficient new dialogue actually supports them.
        self.assertTrue(self.validate(complete, scaffold)["schema_valid"])

    def test_seed_mapping_cannot_erase_or_invent_selected_refs(self):
        for mutation in ("empty", "unknown", "duplicate"):
            with self.subTest(mutation=mutation):
                scene = deepcopy(self.scene)
                if mutation == "empty":
                    scene["card_development_links"] = []
                elif mutation == "unknown":
                    scene["card_development_links"][0]["card_step_ids"] = ["s999"]
                else:
                    scene["card_development_links"][1]["card_step_ids"].append("s1")
                with self.assertRaises(ValueError):
                    review._card_references(self.card, scene)

    def test_missing_selected_card_judgment_is_not_filled(self):
        raw = deepcopy(self.raw)
        missing = raw["card_checks"].pop()
        result = self.validate(raw)
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["schema_valid"])
        repair = review.build_format_repair(result, raw, self.scene, self.history, self.realization,
                                          dataset_card=self.card)
        self.assertEqual(repair["repair_check_ids"]["card_checks"], [["dependency", 1]])
        patch = {"root_id": self.root, "route": self.route, "scene_id": "S01", "repair_updates": {}}
        merged = review.merge_review_repair(raw, patch, repair)
        self.assertEqual(self.validate(merged)["status"], "REJECT")
        patch["repair_updates"] = {"card_checks": [missing]}
        self.assertEqual(self.validate(review.merge_review_repair(raw, patch, repair))["status"], "PASS")

    def test_missing_route_semantics_never_inferred_from_seed_mapping(self):
        for field in ("primary_goal", "relation_to_history", "serves_existing_event", "new_progress", "reason"):
            raw = deepcopy(self.raw)
            raw["route_fulfillment"].pop(field)
            with self.subTest(field=field):
                result = self.validate(raw)
                self.assertEqual(result["status"], "REJECT")
                self.assertFalse(result["schema_valid"])
                self.assertIsNone(result["route_decision_provenance"])

    def test_core_premise_negative_blocks_positive_distinct_event_tuple(self):
        raw = deepcopy(self.raw)
        raw["card_checks"][0].update(realized=False, evidence=[],
            reason="SYNTHETIC: incidental contact exchange does not realize the seed's lost-property event.")
        result = self.validate(raw)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(result["parsed"]["route_fulfillment"]["satisfied"])
        self.assertTrue(any("card_step[s1]:not_realized" in item for item in result["failures"]))

    def test_old_logistics_is_rejected_despite_all_card_checks_passing(self):
        raw = deepcopy(self.raw)
        raw["route_fulfillment"].update(primary_event="Interview taxi", primary_goal="Reach the old interview",
            relation_to_history="old_event_substep", serves_existing_event=True, existing_event="Interview",
            reason="SYNTHETIC: taxi booking is old interview logistics.",
            correction="Give the new event its own main objective.",
            history_evidence=[{"turn_ref": "P01:T001"}])
        result = self.validate(raw)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["parsed"]["route_fulfillment"]["satisfied"])

    def test_prompts_select_original_ids_and_strip_source_evidence(self):
        original = deepcopy(self.card)
        prompt = review.build_review_prompt(self.root, self.route, self.scene, self.history,
            self.realization, dataset_card=self.card)
        payload = json.loads(prompt.split("\nREVIEW INPUT DATA:\n")[1])
        expected = [{"kind": "step", "id": "s1"}, {"kind": "step", "id": "s2"},
                    {"kind": "dependency", "id": 1}]
        self.assertEqual(payload["expected_check_ids"]["card_checks"], expected)
        self.assertEqual(payload["dataset_card_guidance"]["steps"], self.card["steps"])
        self.assertEqual(payload["dataset_card_guidance"]["dependencies"], self.card["dependencies"])
        self.assertFalse(payload["dataset_card_guidance"]["source_is_root_evidence"])
        self.assertNotIn("SOURCE_ONLY_MUST_NOT_ENTER_ROOT_CONTEXT", prompt)
        self.assertNotIn("basis_reason", payload["scene_plan"]["state_transition"])
        self.assertIn("return realized=false", prompt)
        request = scan.build_request(self.root, self.route, self.scene, self.history, [self.card], {})
        scan_prompt = request["messages"][0]["content"]
        self.assertIn("unselected source", scan_prompt)
        self.assertIn("Sharing people, theme or motivation is allowed", scan_prompt)
        self.assertIn("old_event_substep", scan_prompt)
        self.assertNotIn("SOURCE_ONLY_MUST_NOT_ENTER_ROOT_CONTEXT", scan_prompt)
        self.assertEqual(self.card, original)


if __name__ == "__main__":
    unittest.main()
