"""Offline review-contract regressions; authored judgments are NOT model runs.

Archive cases bind real v91 plans/cards to explicit reviewer-negative fixtures.
They test whether the new gate respects those findings, not whether a mechanical
validator can discover their semantic meaning without a substantive reader.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import event_state as event
from planner_contract import collect_mapping_errors


ARCHIVE = (Path(__file__).resolve().parents[2] / "h4_planner_cardinality_expansion_batch_v91"
           / "expansion_engine/runs/ten_roots_B_scene01/requests")


def review_fixture(scene, history, cards, root="r", route="dataset_new_event"):
    """Complete human-authored compatible envelope for contract mutation tests."""
    def quote(index):
        return {"development_index": index, "quote": scene["developments"][index - 1][:80]}

    output = {
        "root_id": root, "route": route, "scene_id": scene["scene_id"],
        "segments": [{"segment_id": item["segment_id"],
                      "none_reason": "No historical constraint selected in this authored contract fixture."}
                     for item in event.segment_index(history)],
        "timeline": [],
        "plan_checks": [{"development_index": index, "verdict": "compatible", "constraint_ids": [],
                         "reason": "This fixture declares a coherent new event.", "issue_type": "coherent_new_event",
                         "plan_evidence": [quote(index)]}
                        for index in range(1, len(scene["developments"]) + 1)],
        "card_checks": [],
        "consistency_checks": [{"category": category, "verdict": "compatible",
                                "reason": "This authored fixture declares no category defect.",
                                "plan_evidence": [quote(1)]}
                               for category in sorted(event.CONSISTENCY_CATEGORIES)],
    }
    for card in cards:
        def target_row(identifier, key, link_key):
            mapped = sorted({development for link in scene["card_development_links"]
                             if identifier in link[link_key] for development in link["development_indices"]})
            return {key: identifier, "mapped_development_indices": mapped, "verdict": "compatible",
                    "reason": "This fixture declares target semantics preserved.",
                    "plan_evidence": [quote(mapped[0])]}
        output["card_checks"].append({
            "card_id": card["card_id"],
            "function_check": {"verdict": "compatible", "reason": "Authored fixture declares the whole event function preserved.", "plan_evidence": [quote(1)]},
            "step_checks": [target_row(step["step_id"], "step_id", "card_step_ids") for step in card["steps"]],
            "dependency_checks": [target_row(index, "dependency_index", "dependency_indices")
                                  for index in range(1, len(card["dependencies"]) + 1)],
        })
    return output


class EventStateV92Tests(unittest.TestCase):
    def setUp(self):
        self.history = [{"turn_ref": "r:P01:T001", "text": "We can discuss a new lunch tomorrow."}]
        self.card = {
            "card_id": "restaurant-card", "guidance_level": "scaffold",
            "steps": [{"step_id": "s1", "actor_id": "a", "action": "request restaurant details", "state": "requested"},
                      {"step_id": "s2", "actor_id": "b", "action": "provide restaurant details", "state": "reported_action"}],
            "dependencies": [{"before": "s1", "after": "s2", "relation": "reply_to_request"}],
        }
        self.scene = {"scene_id": "S01", "dataset_card_id": self.card["card_id"],
                      "developments": ["Tomorrow Elena asks Sam for lunch restaurant details.",
                                       "Sam answers Elena's request with the restaurant address and cuisine."],
                      "card_development_links": [
                          {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1]},
                          {"card_step_ids": ["s2"], "dependency_indices": [1], "development_indices": [2]}]}
        self.response = review_fixture(self.scene, self.history, [self.card])

    def validate(self, response=None):
        return event.validate(response if response is not None else self.response,
                              "r", "dataset_new_event", self.scene, self.history, cards=[self.card])

    def assert_error(self, code, response=None):
        with self.assertRaises(event.EventStateValidationError) as caught:
            self.validate(response)
        self.assertIn(code, [error["code"] for error in caught.exception.errors])

    def test_complete_contract_is_unchanged_and_not_semantic_certification(self):
        before = deepcopy((self.response, self.scene, self.card))
        result = self.validate()
        self.assertEqual(result, self.response)
        self.assertEqual(before, (self.response, self.scene, self.card))
        self.assertFalse(event.requires_replan(result))
        self.assertFalse(event.derive_coverage(result, self.history)["semantic_certified"])

    def test_missing_or_duplicate_card_steps_and_dependencies_cannot_hide(self):
        for target, code in (("step_checks", "event_state_card_target_coverage_invalid"),
                             ("dependency_checks", "event_state_card_target_coverage_invalid")):
            with self.subTest(target=target, mutation="missing"):
                result = deepcopy(self.response)
                result["card_checks"][0][target].pop()
                self.assert_error(code, result)
            with self.subTest(target=target, mutation="duplicate"):
                result = deepcopy(self.response)
                result["card_checks"][0][target].append(deepcopy(result["card_checks"][0][target][0]))
                self.assert_error(code, result)

    def test_card_quote_must_be_from_its_actual_mapped_development(self):
        result = deepcopy(self.response)
        row = result["card_checks"][0]["step_checks"][0]
        row["mapped_development_indices"] = [2]
        row["plan_evidence"] = [{"development_index": 2, "quote": self.scene["developments"][1]}]
        self.assert_error("event_state_card_mapped_developments_mismatch", result)
        self.assert_error("event_state_plan_evidence_index_invalid", result)

    def test_history_quote_cannot_replace_plan_quote_and_own_check_must_be_cited(self):
        result = deepcopy(self.response)
        result["plan_checks"][0]["plan_evidence"][0]["quote"] = self.history[0]["text"]
        self.assert_error("event_state_plan_evidence_quote_invalid", result)
        result = deepcopy(self.response)
        result["plan_checks"][0]["plan_evidence"] = [{"development_index": 2, "quote": self.scene["developments"][1]}]
        self.assert_error("event_state_own_development_not_cited", result)

    def test_assigned_card_and_all_categories_are_required(self):
        with self.assertRaises(event.EventStateValidationError) as caught:
            event.validate(self.response, "r", "dataset_new_event", self.scene, self.history)
        self.assertIn("event_state_assigned_cards_required", [e["code"] for e in caught.exception.errors])
        result = deepcopy(self.response)
        result["card_checks"][0]["card_id"] = "unassigned-card"
        self.assert_error("event_state_card_identity_mismatch", result)
        for category in event.CONSISTENCY_CATEGORIES:
            with self.subTest(category=category):
                result = deepcopy(self.response)
                result["consistency_checks"] = [row for row in result["consistency_checks"] if row["category"] != category]
                self.assert_error("event_state_consistency_coverage_invalid", result)

    def test_card_and_internal_negatives_bind_when_all_historical_checks_pass(self):
        for section, kind in (("step_checks", "card_step"), ("dependency_checks", "card_dependency")):
            with self.subTest(section=section):
                result = deepcopy(self.response)
                result["card_checks"][0][section][0].update(verdict="revise", reason="Mapped plan substitutes the required action or relation.")
                result = self.validate(result)
                self.assertTrue(event.requires_replan(result))
                binding = event.binding_plan_checks(result)
                self.assertEqual([row["check_kind"] for row in binding], [kind])
                self.assertNotIn("development_index", binding[0])
                self.assertEqual(result["plan_checks"], self.response["plan_checks"])
        result = deepcopy(self.response)
        result["consistency_checks"][0].update(verdict="revise", reason="The named addressee is also presented as a different third person.")
        result = self.validate(result)
        self.assertEqual(event.binding_plan_checks(result)[0]["check_kind"], "plan_consistency")

    def test_soft_history_stays_diagnostic_but_independent_plan_conflict_binds(self):
        result = deepcopy(self.response)
        result["timeline"] = [{"constraint_id": "c1", "status": "uncertain", "evidence_basis": "inferred",
                               "statement": "A lunch could be planned.", "implication": "A tentative inference only.",
                               "evidence": [{"turn_ref": self.history[0]["turn_ref"], "quote": "discuss a new lunch"}]}]
        result["segments"][0]["none_reason"] = ""
        result["plan_checks"][0].update(verdict="revise", constraint_ids=["c1"], issue_type="historical_contradiction",
                                        reason="The reviewer records a tentative historical conflict.")
        parsed = self.validate(result)
        self.assertFalse(event.requires_replan(parsed))
        self.assertFalse(event.plan_diagnostics(parsed)[0]["requires_replan"])
        result["timeline"][0]["evidence_basis"] = "reported"
        self.assertTrue(event.requires_replan(self.validate(result)))
        result["timeline"][0]["evidence_basis"] = "inferred"
        result["plan_checks"][0]["issue_type"] = "internal_conflict"
        self.assertTrue(event.requires_replan(self.validate(result)))

    def test_v91_20432_absence_reject_is_contract_error_not_automatic_pass(self):
        reason = ("History contains no evidence of Emily stating intent to visit a college campus; "
                  "this is a new event not supported by prior turns.")
        self.assertTrue(event.absence_only_reason(reason))
        self.response["plan_checks"][0].update(verdict="revise", issue_type="unsupported_historical_assertion", reason=reason)
        with self.assertRaises(event.EventStateValidationError) as caught:
            self.validate()
        self.assertIn("event_state_history_absence_not_contradiction", [error["code"] for error in caught.exception.errors])
        repair = event.build_repair(caught.exception, self.response, self.history)
        self.assertEqual(repair["previous_raw"], self.response)
        self.assertIn("Never convert a contract error to an automatic pass", repair["instruction"])
        self.assertEqual(self.response["plan_checks"][0]["verdict"], "revise")

    def test_coherent_new_event_and_later_progress_do_not_need_prior_completion(self):
        self.response["plan_checks"][0]["issue_type"] = "later_progress"
        self.response["plan_checks"][0]["reason"] = "Tomorrow's explicitly later scene can report a new update."
        self.assertFalse(event.requires_replan(self.validate()))

    def test_material_missing_interval_can_still_require_replanning(self):
        self.response["plan_checks"][0].update(verdict="revise", issue_type="unclear_transition",
                                               reason="The plan reports a second test batch immediately after installation without an elapsed test interval.")
        self.assertFalse(event.absence_only_reason(self.response["plan_checks"][0]["reason"]))
        self.assertTrue(event.requires_replan(self.validate()))

    def test_old_event_route_has_empty_card_checks_but_consistency_is_required(self):
        response = review_fixture(self.scene, self.history, [], route="old_event_continuation")
        self.assertEqual(event.validate(response, "r", "old_event_continuation", self.scene, self.history)["card_checks"], [])
        response["consistency_checks"] = []
        with self.assertRaises(event.EventStateValidationError):
            event.validate(response, "r", "old_event_continuation", self.scene, self.history)

    def test_real_v91_substitution_plans_obey_mapping_shape_but_semantic_negatives_bind(self):
        cases = {
            "14304": ("s3", "Partnership metaphor does not instantiate distinctiveness-of-Christianity inquiry."),
            "19053": ("s4", "Successful schedule adjustment replaces the required no-results failure."),
            "19708": ("s2", "Barbecue contribution replaces the required no-matching-options report."),
            "194259": ("s1", "Old reservation confirmation and speech practice omit a new restaurant-details request."),
        }
        for root, (step_id, reason) in cases.items():
            with self.subTest(root=root):
                request_path = sorted(ARCHIVE.glob(f"*-cc-train-episode-{root}-B-S01-event-state-*-1/REQUEST.json"))[0]
                request = json.loads(request_path.read_text())
                context = json.loads(request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
                scene, history, cards = context["scene_plan"], context["cumulative_history"], context["dataset_cards"]
                self.assertEqual(collect_mapping_errors(scene, cards[0]), [])
                review = review_fixture(scene, history, cards, root=context["root_id"])
                row = next(row for row in review["card_checks"][0]["step_checks"] if row["step_id"] == step_id)
                row.update(verdict="revise", reason=reason)
                parsed = event.validate(review, context["root_id"], "dataset_new_event", scene, history, cards=cards)
                self.assertEqual(event.binding_plan_checks(parsed)[0]["step_id"], step_id)
                self.assertTrue(event.requires_replan(parsed))


if __name__ == "__main__":
    unittest.main()
