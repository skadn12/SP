"""Ref-only review and targeted repairs, using immutable v18 live evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import realization_review as review

LIVE = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v18/runs/sequential_pilot_3roots_01/requests"


def load_case(prefix):
    directory = next(LIVE.glob(prefix + "-*"))
    payload = json.loads((directory / "REQUEST.json").read_text())
    context = json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split("\nREVIEW INPUT DATA:\n")[1])[0]
    raw = json.loads(json.loads((directory / "RESPONSE.json").read_text())["text"])
    role = raw["route_fulfillment"]
    is_a = raw["route"] == "old_event_continuation"
    role.update(primary_goal=role["primary_event"], serves_existing_event=is_a,
                existing_event=role["primary_event"] if is_a else "",
                new_progress=True, correction="")
    return context, raw


def refs_only(value):
    if isinstance(value, dict):
        if set(value) == {"turn_ref", "quote"}:
            return {"turn_ref": value["turn_ref"]}
        return {key: refs_only(item) for key, item in value.items()}
    if isinstance(value, list):
        return [refs_only(item) for item in value]
    return value


def kwargs(context):
    return {"dataset_card": context["dataset_card_guidance"],
            "latest_state": context["latest_state"], "event_state": context["event_state"],
            "history": context["branch_history"]}


def validate(context, raw):
    return review.validate_review(raw, context["root_id"], context["route"],
                                  context["scene_plan"], context["new_scene"], **kwargs(context))


def repair_spec(context, raw, result=None):
    options = kwargs(context)
    options.pop("history")
    return review.build_format_repair(result or validate(context, raw), raw,
        context["scene_plan"], context["branch_history"], context["new_scene"], **options)


def patch_object(context, updates):
    return {key: context[key] for key in ("root_id", "route", "scene_id")} | {"repair_updates": updates}


class ReferenceReviewTests(unittest.TestCase):
    def test_live_ellipsis_failure_resolved_by_ref_only_not_quote_rewriting(self):
        context, raw = load_case("036")
        old = validate(context, raw)
        self.assertFalse(old["schema_valid"])
        compact = refs_only(raw)
        original = deepcopy(compact)
        result = validate(context, compact)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(compact, original)
        self.assertEqual(result["raw_parsed"], original)
        receipt = result["citation_provenance"]
        self.assertTrue(receipt["dereferenced_citations"])
        self.assertTrue(receipt["inherited_fact_evidence"])
        self.assertEqual(receipt["inherited_fact_evidence"][0]["selection"],
                         "program_inherited_fact_binding_not_model_selected")
        self.assertFalse(result["semantic_certified"])

    def test_later_A_reference_valid_without_repeating_original_f2_reference(self):
        context, raw = load_case("025")
        result = validate(context, refs_only(raw))
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        f2 = next(item for item in result["parsed"]["state_checks"] if item["fact_id"] == "f2")
        self.assertTrue(all(item["turn_ref"].startswith("R02:A:") for item in f2["history_evidence"]))
        inherited = next(item for item in result["citation_provenance"]["inherited_fact_evidence"]
                         if item["check_namespace"] == "state_checks" and item["id"] == "f2")
        self.assertTrue(any(item["turn_ref"].endswith("P01:T017") for item in inherited["evidence"]))

    def test_empty_extra_history_keeps_fact_binding_and_uncited_diagnostic(self):
        context, raw = load_case("025")
        raw = refs_only(raw)
        f2 = next(item for item in raw["state_checks"] if item["fact_id"] == "f2")
        f2["history_evidence"] = []
        result = validate(context, raw)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertTrue(any(item.get("fact_id") == "f2" for item in result["empty_consistent_findings"]))
        self.assertFalse(result["semantic_certified"])

    def test_unknown_reference_rejected_not_fuzzily_mapped(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        raw["checks"][0]["evidence"] = [{"turn_ref": "NEW:T999"}]
        result = validate(context, raw)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("citation_reference_not_in_context" in item for item in result["schema_failures"]))

    def test_contradiction_requires_reason_and_new_evidence(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        item = raw["state_checks"][0]
        item.update(verdict="contradicted", new_evidence=[])
        item.pop("reason", None)
        result = validate(context, raw)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("contradiction_reason_required" in row for row in result["schema_failures"]))
        item.update(reason="The actor's new statement contradicts the cited old timing.",
                    new_evidence=[{"turn_ref": "NEW:T001"}])
        raw["consistent_with_history"] = False
        result = validate(context, raw)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")

    def test_missing_check_patch_adds_only_model_returned_row(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        removed = raw["checks"].pop()
        spec = repair_spec(context, raw)
        self.assertEqual(spec["repair_check_ids"]["checks"], [removed["development_index"]])
        empty = review.merge_review_repair(raw, patch_object(context, {}), spec)
        self.assertFalse(validate(context, empty)["schema_valid"])
        merged = review.merge_review_repair(raw, patch_object(context, {"checks": [removed]}), spec)
        self.assertEqual(merged["checks"][:-1], raw["checks"])
        self.assertTrue(validate(context, merged)["schema_valid"])

    def test_valid_negative_and_other_valid_rows_protected_in_partial_and_full(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        raw["checks"][0].update(realized=False, evidence=[], reason="Planned step not realized.")
        removed = raw["checks"].pop()
        spec = repair_spec(context, raw)
        first = deepcopy(raw["checks"][0]); first["realized"] = True
        with self.assertRaisesRegex(ValueError, "protected_check|negative_verdict"):
            review.merge_review_repair(raw, patch_object(context, {"checks": [first, removed]}), spec)
        complete = deepcopy(raw)
        complete["checks"].append(removed)
        complete["checks"][0] = first
        with self.assertRaisesRegex(ValueError, "protected_check|negative_verdict"):
            review.merge_review_repair(raw, complete, spec)
        good = review.merge_review_repair(raw, patch_object(context, {"checks": [removed]}), spec)
        result = validate(context, good)
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["status"], "REJECT")

    def test_invalid_negative_citation_can_repair_without_flipping_verdict(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        item = raw["state_checks"][0]
        item.update(verdict="contradicted", new_evidence=[{"turn_ref": "NEW:T999"}], reason="A timing conflict.")
        spec = repair_spec(context, raw)
        changed = deepcopy(item); changed.update(verdict="consistent", new_evidence=[])
        with self.assertRaisesRegex(ValueError, "negative_verdict_changed"):
            review.merge_review_repair(raw, patch_object(context, {"state_checks": [changed]}), spec)
        changed = deepcopy(item); changed["new_evidence"] = [{"turn_ref": "NEW:T001"}]
        merged = review.merge_review_repair(raw, patch_object(context, {"state_checks": [changed]}), spec)
        self.assertEqual(merged["state_checks"][0]["verdict"], "contradicted")

    def test_prompt_fixed_ids_short_references_and_new_progress_distinction(self):
        context, raw = load_case("037")
        options = kwargs(context); options.pop("history")
        prompt = review.build_review_prompt(context["root_id"], context["route"], context["scene_plan"],
            context["branch_history"], context["new_scene"], **options)
        instructions = prompt.split("\nREVIEW INPUT DATA:\n")[0]
        self.assertIn("Missing old evidence of that NEW action, occupation or participant is not a contradiction",
                      " ".join(instructions.split()))
        self.assertIn("DO NOT copy quote text", instructions)
        self.assertNotIn("plan_check", instructions)
        self.assertIn("requires 1..3 existing NEW:T refs", instructions)
        schema = json.JSONDecoder().raw_decode(instructions.split("\nOUTPUT SHAPE:\n")[1])[0]
        self.assertEqual(schema["checks"][0]["evidence"], [{"turn_ref": "<copy an existing NEW:T ID>"}])
        payload = json.JSONDecoder().raw_decode(prompt.split("\nREVIEW INPUT DATA:\n")[1])[0]
        self.assertEqual(payload["expected_check_ids"]["development_indices"],
                         list(range(1, len(context["scene_plan"]["developments"]) + 1)))

    def test_internally_inconsistent_role_false_can_be_corrected_by_model(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        raw["route_fulfillment"].update(satisfied=False, correction="Clarify the role verdict.")
        spec = repair_spec(context, raw)
        self.assertTrue(spec["allow_route_semantic_consistency_repair"])
        corrected = deepcopy(raw["route_fulfillment"])
        corrected.update(satisfied=True, correction="")
        merged = review.merge_review_repair(raw, patch_object(context, {"route_fulfillment": corrected}), spec)
        self.assertEqual(validate(context, merged)["status"], "PASS")

    def test_valid_negative_role_with_quote_error_cannot_flip_positive(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        role = raw["route_fulfillment"]
        role.update(satisfied=False, relation_to_history="old_event_substep", serves_existing_event=True,
                    existing_event="Existing travel", correction="Create a separate recovery goal.",
                    history_evidence=[{"turn_ref": context["branch_history"][0]["turn_ref"], "quote": "INVALID EXACT QUOTE"}])
        spec = repair_spec(context, raw)
        self.assertFalse(spec["allow_route_semantic_consistency_repair"])
        corrected = deepcopy(role)
        corrected.update(satisfied=True, relation_to_history="distinct_event", serves_existing_event=False,
                         existing_event="", correction="", history_evidence=[])
        with self.assertRaisesRegex(ValueError, "negative_verdict_changed"):
            review.merge_review_repair(raw, patch_object(context, {"route_fulfillment": corrected}), spec)

    def test_complete_corrected_response_removes_explicitly_omitted_extra_key(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        raw["unexpected_key"] = "invalid field"
        spec = repair_spec(context, raw)
        full = deepcopy(raw); full.pop("unexpected_key")
        merged = review.merge_review_repair(raw, full, spec)
        self.assertNotIn("unexpected_key", merged)
        self.assertTrue(validate(context, merged)["schema_valid"])

    def test_unknown_id_replaced_by_explicit_complete_group(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        expected_rows = deepcopy(raw["checks"])
        raw["checks"].append({"development_index": 999, "realized": True, "evidence": [{"turn_ref": "NEW:T001"}]})
        spec = repair_spec(context, raw)
        self.assertIn("checks", spec["replace_groups"])
        merged = review.merge_review_repair(raw, patch_object(context, {"checks": expected_rows}), spec)
        self.assertTrue(validate(context, merged)["schema_valid"])

    def test_unknown_negative_id_not_silently_discarded(self):
        context, raw = load_case("037")
        raw = refs_only(raw)
        expected_rows = deepcopy(raw["checks"])
        raw["checks"].append({"development_index": 999, "realized": False, "evidence": [], "reason": "Unrealized step"})
        spec = repair_spec(context, raw)
        with self.assertRaisesRegex(ValueError, "unbound_negative_removed"):
            review.merge_review_repair(raw, patch_object(context, {"checks": expected_rows}), spec)

    def test_invalid_json_full_model_response_restores_identity_and_checks(self):
        context, valid = load_case("037")
        valid = refs_only(valid)
        raw = "NOT JSON"
        spec = repair_spec(context, raw)
        merged = review.merge_review_repair(raw, valid, spec)
        self.assertTrue(validate(context, merged)["schema_valid"])


if __name__ == "__main__":
    unittest.main()
