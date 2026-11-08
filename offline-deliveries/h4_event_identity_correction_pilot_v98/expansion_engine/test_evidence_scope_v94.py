"""Exact-reference and binding-negative regressions, not semantic certification.

Historical raw reviews are replayed only for their citation contract. Authored
negative fixtures exercise the gate without claiming a model found the defect.
"""
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import unittest

import event_state as event
from test_event_state_v92 import review_fixture


V93 = Path(__file__).resolve().parents[2] / "h4_review_reference_minfix_v93"
PROBE = V93 / "evidence/model_review_probe/requests"
FULL_RUN = V93 / "expansion_engine/runs/four_other_roots_B_scene01/requests"


def archived_review(folder, pattern):
    directory = sorted(folder.glob(pattern))[0]
    request = json.loads((directory / "REQUEST.json").read_text())
    input_text = request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1]
    context, _ = json.JSONDecoder().raw_decode(input_text)
    raw = json.loads((directory / "RESPONSE.json").read_text())["text"]
    return context, raw


def validate_archive(context, raw):
    return event.validate(raw, context["root_id"], context["route"], context["scene_plan"],
                          context["cumulative_history"], cards=context["dataset_cards"])


class EvidenceScopeV94Tests(unittest.TestCase):
    def setUp(self):
        self.history = [{"turn_ref": "r:P01:T001", "text": "We can discuss a new lunch tomorrow."}]
        self.card = {
            "card_id": "restaurant-card", "guidance_level": "scaffold",
            "steps": [{"step_id": "s1", "actor_id": "a", "action": "request restaurant details", "state": "requested"},
                      {"step_id": "s2", "actor_id": "b", "action": "provide restaurant details", "state": "reported_action"},
                      {"step_id": "s3", "actor_id": "a", "action": "thank the provider", "state": "reported_action"}],
            "dependencies": [{"before": "s1", "after": "s2", "relation": "reply_to_request"}],
        }
        self.scene = {
            "scene_id": "S01", "dataset_card_id": self.card["card_id"],
            "goal": "Elena asks Sam about a new lunch venue.", "actors": ["Elena", "Sam"],
            "state_transition": {"new_event": "Elena starts a fresh restaurant inquiry.",
                                 "temporal_placement": "Tomorrow, after the earlier discussion."},
            "developments": ["Elena asks Sam for lunch restaurant details.",
                             "Sam answers Elena's request with an address and cuisine.",
                             "Elena thanks Sam for the information."],
            "card_development_links": [
                {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1]},
                {"card_step_ids": ["s2"], "dependency_indices": [1], "development_indices": [2]},
                {"card_step_ids": ["s3"], "dependency_indices": [], "development_indices": [3]}],
        }
        self.response = review_fixture(self.scene, self.history, [self.card])

    def quote(self, index):
        return {"development_index": index, "quote": self.scene["developments"][index - 1]}

    def validate(self, response=None):
        return event.validate(self.response if response is None else response, "r", "dataset_new_event",
                              self.scene, self.history, cards=[self.card])

    def dependency(self, response=None):
        return (self.response if response is None else response)["card_checks"][0]["dependency_checks"][0]

    def assert_error(self, code, response=None):
        with self.assertRaises(event.EventStateValidationError) as caught:
            self.validate(response)
        self.assertIn(code, [row["code"] for row in caught.exception.errors])

    def test_dependency_endpoint_context_is_separate_from_unchanged_original_mapping(self):
        self.dependency()["plan_evidence"] = [self.quote(1), self.quote(2)]
        before = deepcopy((self.response, self.scene, self.card))
        self.assertEqual(self.validate(), self.response)
        table = event.review_reference_context(self.scene, [self.card])
        targets = table["actual_card_targets"][0]
        dependency = targets["dependency_targets"][0]
        self.assertEqual(dependency["mapped_development_indices"], [2])
        self.assertEqual(dependency["mapped_developments"], [table["numbered_developments"][1]])
        self.assertEqual(dependency["allowed_evidence_development_indices"], [1, 2])
        self.assertEqual(dependency["allowed_evidence_developments"], table["numbered_developments"][:2])
        self.assertEqual(targets["step_targets"][0]["allowed_evidence_development_indices"], [1])
        self.assertEqual(before, (self.response, self.scene, self.card))

    def test_original_dependency_mapping_requires_a_valid_exact_quote(self):
        for citation in (None,
                         {"development_index": 2, "quote": "invented quotation"},
                         {"development_index": 2, "quote": self.scene["developments"][1], "extra": "invalid"}):
            with self.subTest(citation=citation):
                response = deepcopy(self.response)
                self.dependency(response)["plan_evidence"] = [self.quote(1)] + ([citation] if citation else [])
                self.assert_error("event_state_dependency_original_mapping_not_cited", response)

    def test_dependency_cannot_cite_unrelated_development_or_change_mapping(self):
        self.dependency()["plan_evidence"] = [self.quote(2), self.quote(3)]
        self.assert_error("event_state_plan_evidence_index_invalid")
        self.dependency()["plan_evidence"] = [self.quote(1), self.quote(2)]
        self.dependency()["mapped_development_indices"] = [1, 2]
        self.assert_error("event_state_card_mapped_developments_mismatch")

    def test_step_evidence_remains_strictly_in_its_original_mapping(self):
        row = self.response["card_checks"][0]["step_checks"][0]
        row["plan_evidence"] = [self.quote(1), self.quote(2)]
        self.assert_error("event_state_plan_evidence_index_invalid")

    def test_expanded_dependency_evidence_preserves_binding_negative(self):
        self.dependency().update(verdict="revise", reason="The assigned relation is missing despite endpoint actions.",
                                 plan_evidence=[self.quote(1), self.quote(2)])
        raw = json.dumps(self.response, ensure_ascii=False)
        before = raw.encode("utf-8")
        result = event.validate(raw, "r", "dataset_new_event", self.scene, self.history, cards=[self.card])
        self.assertTrue(event.requires_replan(result))
        binding = event.binding_plan_checks(result)
        self.assertEqual(binding[0]["check_kind"], "card_dependency")
        self.assertEqual(binding[0]["mapped_development_indices"], [2])
        self.assertEqual(binding[0]["plan_evidence"], self.dependency()["plan_evidence"])
        self.assertEqual(raw.encode("utf-8"), before)

    def test_field_inventory_only_copies_allowlisted_original_strings_without_coercion(self):
        scene = deepcopy(self.scene)
        scene.update(goal={"text": "not a string"}, actors=["Elena’s … exact name", {"name": "Sam"}, 3, "", None])
        scene["state_transition"].update(new_event=False, temporal_placement=" literal ... time ",
                                          basis_fact_ids=["f1"], additional_field="not allowed")
        before = deepcopy(scene)
        self.assertEqual(event.plan_field_inventory(scene), [
            {"plan_field_id": "actors:1", "text": "Elena’s … exact name"},
            {"plan_field_id": "actors:4", "text": ""},
            {"plan_field_id": "state_transition:temporal_placement", "text": " literal ... time "}])
        self.assertEqual(event.review_reference_context(scene, [self.card])["plan_fields"],
                         event.plan_field_inventory(scene))
        self.assertEqual(scene, before)

    def test_consistency_accepts_mixed_development_and_original_field_evidence(self):
        row = self.response["consistency_checks"][0]
        row.update(verdict="revise", reason="This authored fixture identifies a participant mismatch in the goal.",
                   plan_evidence=[self.quote(1), {"plan_field_id": "goal", "quote": self.scene["goal"]}])
        before = deepcopy(self.response)
        result = self.validate()
        self.assertEqual(result, before)
        self.assertEqual(self.response, before)
        self.assertTrue(event.requires_replan(result))
        self.assertEqual(event.binding_plan_checks(result)[0]["plan_evidence"], row["plan_evidence"])

    def test_each_allowlisted_field_id_is_supported_for_consistency_only(self):
        for field in event.plan_field_inventory(self.scene):
            with self.subTest(field=field["plan_field_id"]):
                response = deepcopy(self.response)
                response["consistency_checks"][0]["plan_evidence"] = [
                    {"plan_field_id": field["plan_field_id"], "quote": field["text"]}]
                self.assertEqual(self.validate(response), response)

    def test_wrong_fields_nonstring_fields_and_invented_quotes_fail_closed(self):
        self.scene["state_transition"]["new_event"] = {"text": "not coerced"}
        for field_id in ("actors:0", "actors:3", "scene_plan.goal", "state_transition:basis_fact_ids",
                         "state_transition:new_event", "state_transition", 1):
            with self.subTest(field=field_id):
                response = deepcopy(self.response)
                response["consistency_checks"][0]["plan_evidence"] = [{"plan_field_id": field_id, "quote": "Elena"}]
                self.assert_error("event_state_plan_evidence_field_invalid", response)
        for quote in ("", "   ", "Elena ... lunch", "Elena asks Sam about an old lunch venue."):
            with self.subTest(quote=quote):
                response = deepcopy(self.response)
                response["consistency_checks"][0]["plan_evidence"] = [{"plan_field_id": "goal", "quote": quote}]
                self.assert_error("event_state_plan_evidence_quote_invalid", response)
        for development in (0, 4, True, "1"):
            with self.subTest(development=development):
                response = deepcopy(self.response)
                response["consistency_checks"][0]["plan_evidence"] = [
                    {"development_index": development, "quote": self.scene["developments"][0]}]
                self.assert_error("event_state_plan_evidence_index_invalid", response)

    def test_field_citation_shape_is_exclusive_and_cannot_bypass_other_checks(self):
        field_quote = {"plan_field_id": "actors:1", "quote": "Elena"}
        for citation in ({**field_quote, "development_index": 1}, {**field_quote, "extra": True}, {"quote": "Elena"}):
            with self.subTest(citation=citation):
                response = deepcopy(self.response)
                response["consistency_checks"][0]["plan_evidence"] = [citation]
                self.assert_error("event_state_fields_invalid", response)
        for target in ("development", "step", "dependency"):
            with self.subTest(target=target):
                response = deepcopy(self.response)
                row = (response["plan_checks"][0] if target == "development" else
                       response["card_checks"][0]["step_checks"][0] if target == "step" else self.dependency(response))
                row["plan_evidence"] = [field_quote]
                self.assert_error("event_state_fields_invalid", response)
        response = deepcopy(self.response)
        response["consistency_checks"][0]["plan_evidence"] = [field_quote, field_quote]
        self.assert_error("event_state_plan_evidence_duplicate", response)

    def test_request_tail_copies_fields_after_unchanged_raw_and_keeps_http_settings(self):
        repair = {"previous_raw": {"note": "OLD_RAW_SENTINEL", "invented_field": "scene_plan.goal"}}
        before = deepcopy((self.scene, self.card, self.history, repair))
        request = event.build_request("r", "dataset_new_event", self.scene, self.history, [self.card], {}, repair=repair)
        text = request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1]
        context, end = json.JSONDecoder().raw_decode(text)
        tail = text[end:]
        table = json.loads(tail.split("ACTUAL REVIEW REFERENCE TABLE\n", 1)[1])
        self.assertEqual(context["scene_plan"], self.scene)
        self.assertEqual(context["targeted_repair"], repair)
        self.assertEqual(table["plan_fields"], event.plan_field_inventory(self.scene))
        self.assertNotIn("OLD_RAW_SENTINEL", tail)
        self.assertIn("At least one valid exact quote MUST come from", tail)
        self.assertIn("ONLY consistency_checks", tail)
        self.assertIn("Read ALL", event.PROMPT)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(request["temperature"], .25)
        self.assertFalse(request["enable_thinking"])
        self.assertEqual(before, (self.scene, self.card, self.history, repair))

    def test_historical_false_field_development_quote_and_invented_ellipsis_stay_invalid(self):
        for root in ("19053", "20432"):
            for attempt in (1, 2):
                with self.subTest(root=root, attempt=attempt):
                    context, raw = archived_review(PROBE, f"*-cc-train-episode-{root}-plan-review-{attempt}")
                    before = raw.encode("utf-8")
                    with self.assertRaises(event.EventStateValidationError) as caught:
                        validate_archive(context, raw)
                    self.assertIn("event_state_plan_evidence_quote_invalid", [e["code"] for e in caught.exception.errors])
                    self.assertEqual(raw.encode("utf-8"), before)

    def test_historical_old_schema_cannot_bypass_new_required_function_judgment(self):
        for root in ("14304", "38601"):
            with self.subTest(root=root):
                context, raw = archived_review(FULL_RUN, f"*-cc-train-episode-{root}-B-S01-event-state-1-2")
                before = raw.encode("utf-8")
                with self.assertRaises(event.EventStateValidationError) as caught:
                    validate_archive(context, raw)
                self.assertTrue(any("function_check" in row.get("missing", []) or
                                    row.get("path", "").endswith("function_check")
                                    for row in caught.exception.errors))
                self.assertEqual(raw.encode("utf-8"), before)

    def test_historical_20432_empty_historical_evidence_is_still_rejected(self):
        context, raw = archived_review(FULL_RUN, "*-cc-train-episode-20432-B-S01-event-state-1-2")
        with self.assertRaises(event.EventStateValidationError) as caught:
            validate_archive(context, raw)
        self.assertIn("event_state_evidence_count_invalid", [e["code"] for e in caught.exception.errors])

    def test_historical_old_negative_is_not_silently_upgraded_to_new_contract(self):
        context, raw = archived_review(PROBE, "*-cc-train-episode-194259-plan-review-1")
        before = json.loads(raw)
        with self.assertRaises(event.EventStateValidationError):
            validate_archive(context, raw)
        self.assertEqual(json.loads(raw), before)
        self.assertTrue(any(row["verdict"] == "revise" for row in before["card_checks"][0]["step_checks"]))

    def test_original_transition_can_evidence_authored_relationship_negative_without_recertifying_old_positive(self):
        context, _ = archived_review(PROBE, "*-cc-train-episode-38601-plan-review-1")
        scene, history, cards = context["scene_plan"], context["cumulative_history"], context["dataset_cards"]
        transition = scene["state_transition"]["new_event"]
        self.assertIn("Elena and Sam", transition)
        response = review_fixture(scene, history, cards, root=context["root_id"])
        row = next(item for item in response["consistency_checks"] if item["category"] == "participant_identity")
        row.update(verdict="revise", reason="Authored negative: romantic agreement was Sam/Sidney, not Elena/Sam.",
                   plan_evidence=[{"plan_field_id": "state_transition:new_event", "quote": transition}])
        before = deepcopy(response)
        parsed = validate_archive(context, response)
        self.assertEqual(parsed, before)
        self.assertTrue(event.requires_replan(parsed))
        self.assertEqual(event.binding_plan_checks(parsed)[0]["category"], "participant_identity")


if __name__ == "__main__":
    unittest.main()
