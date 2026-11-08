"""Final review receives source claims without previous semantic verdicts."""
from copy import deepcopy
import json
import unittest

import realization_review as review
from test_review_v19 import load_case, kwargs, refs_only, validate


class IndependentReviewInputTests(unittest.TestCase):
    def test_event_projection_keeps_exact_historical_claims_not_prior_reasoning(self):
        context, _ = load_case("024")
        original = context["event_state"]
        original["scan_observations"] = [{"observation_id": "o1",
            "kind": "not_found_in_supplied_history", "scope": "cumulative_history",
            "statement": "UNVERIFIED_ABSENCE_OBSERVATION", "limitation": "not_proof_of_absence"}]
        before = deepcopy(original)
        projected = review.review_event_context(original)
        self.assertEqual(original, before)
        self.assertEqual(set(projected), {"root_id", "route", "scene_id", "timeline"})
        self.assertEqual(len(projected["timeline"]), len(original["timeline"]))
        allowed = {"constraint_id", "status", "evidence_basis", "statement", "evidence", "uncertainty"}
        for old, new in zip(original["timeline"], projected["timeline"]):
            self.assertEqual(new, {key: value for key, value in old.items() if key in allowed})
            self.assertNotIn("implication", new)
        rendered = json.dumps(projected)
        self.assertNotIn(original["route_fulfillment"]["reason"], rendered)
        self.assertNotIn("plan_checks", rendered)
        self.assertNotIn("scan_observations", rendered)
        self.assertNotIn("UNVERIFIED_ABSENCE_OBSERVATION", rendered)
        projected["timeline"][0]["statement"] = "MODIFIED_VIEW_ONLY"
        self.assertEqual(original, before)

    def test_plan_projection_preserves_task_and_timing_without_self_labels(self):
        context, _ = load_case("024")
        plan = deepcopy(context["scene_plan"])
        plan["event_definition"] = {"trigger": "UNIQUE_TRIGGER_SELF_JUSTIFICATION",
                                    "main_goal": plan["goal"], "history_relation": "distinct_event"}
        plan["history_context_reason"] = "UNIQUE_PRIOR_JUDGMENT"
        plan["state_transition"]["new_event"] = "UNIQUE_PROMOTED_PLAN_NARRATIVE"
        before = deepcopy(plan)
        projected = review.review_plan_context(plan)
        self.assertEqual(plan, before)
        self.assertEqual(projected["goal"], plan["goal"])
        self.assertEqual(projected["developments"], plan["developments"])
        self.assertEqual(projected["source_anchors"], plan["source_anchors"])
        self.assertEqual(projected["card_development_links"], plan["card_development_links"])
        self.assertEqual(projected["state_transition"], {
            key: value for key, value in plan["state_transition"].items()
            if key in {"temporal_placement", "basis_fact_ids"}})
        rendered = json.dumps(projected)
        self.assertNotIn("UNIQUE_", rendered)
        self.assertNotIn("history_relation", rendered)

    def test_real_historical_payload_contains_no_prior_judgment_and_keeps_history(self):
        context, _ = load_case("024")
        options = kwargs(context); options.pop("history")
        prompt = review.build_review_prompt(context["root_id"], context["route"], context["scene_plan"],
            context["branch_history"], context["new_scene"], **options)
        payload = json.JSONDecoder().raw_decode(prompt.split("\nREVIEW INPUT DATA:\n")[1])[0]
        self.assertEqual(payload["branch_history"], context["branch_history"])
        self.assertEqual(payload["new_scene"], context["new_scene"])
        self.assertEqual(payload["latest_state"], context["latest_state"])
        self.assertEqual(payload["event_state"], review.review_event_context(context["event_state"]))
        self.assertEqual(payload["scene_plan"], review.review_plan_context(context["scene_plan"]))
        data_text = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(context["event_state"]["route_fulfillment"]["reason"], data_text)
        for item in context["event_state"]["timeline"]:
            self.assertNotIn(item["implication"], data_text)

    def test_projection_does_not_change_validation_or_evidence_provenance(self):
        context, raw = load_case("037")
        compact = refs_only(raw)
        full = validate(context, compact)
        projected = deepcopy(context)
        projected["scene_plan"] = review.review_plan_context(context["scene_plan"])
        projected["event_state"] = review.review_event_context(context["event_state"])
        view = validate(projected, compact)
        self.assertEqual(full["status"], view["status"])
        self.assertEqual(full["failures"], view["failures"])
        self.assertEqual(full["parsed"], view["parsed"])
        # Formal runner retains original snapshot provenance. A projected
        # validation view can have a different snapshot hash, never different
        # original cited text or semantic verdict.
        old_evidence = [item["evidence"] for item in full["citation_provenance"]["inherited_fact_evidence"]]
        new_evidence = [item["evidence"] for item in view["citation_provenance"]["inherited_fact_evidence"]]
        self.assertEqual(old_evidence, new_evidence)


if __name__ == "__main__":
    unittest.main()
