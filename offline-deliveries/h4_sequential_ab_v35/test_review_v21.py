"""Reviewer outputs semantics; explicit fixed policy derives route admission."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import realization_review as review
from test_review_v19 import load_case, refs_only, validate, repair_spec, patch_object, kwargs


def valid_role(context, raw, *, negative=False):
    raw = refs_only(deepcopy(raw))
    role = raw["route_fulfillment"]
    role.pop("satisfied", None)
    if negative:
        role.update(relation_to_history="old_event_substep", serves_existing_event=True,
                    existing_event="The existing interview", new_progress=True,
                    primary_goal="Arrange travel for that existing interview",
                    reason="Travel serves the existing interview rather than a separate goal.",
                    correction="Use the guide to create a separate event with its own objective.")
    return raw


class DerivedReviewRouteTests(unittest.TestCase):
    def test_no_satisfied_output_derives_pass_without_modifying_raw(self):
        context, raw = load_case("037")
        raw = valid_role(context, raw)
        old = deepcopy(raw)
        result = validate(context, raw)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["parsed"]["route_fulfillment"]["satisfied"])
        self.assertNotIn("satisfied", result["raw_parsed"]["route_fulfillment"])
        self.assertEqual(raw, old)
        provenance = result["route_decision_provenance"]
        self.assertEqual(provenance["decision_source"], "program_policy_derived_from_explicit_model_fields")
        self.assertFalse(provenance["semantic_certified"])

    def test_old_step_semantic_fields_derive_reject_and_require_correction(self):
        context, raw = load_case("025")
        raw = valid_role(context, raw, negative=True)
        result = validate(context, raw)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertFalse(result["parsed"]["route_fulfillment"]["satisfied"])
        self.assertEqual(result["status"], "REJECT")
        raw["route_fulfillment"]["correction"] = ""
        invalid = validate(context, raw)
        self.assertFalse(invalid["schema_valid"])
        self.assertTrue(any("correction_required" in message for message in invalid["schema_failures"]))

    def test_new_format_simulation_of_v20_old_step_does_not_rewrite_old_output(self):
        folder = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v20/probes/historical_failures_01"
        path = folder / "requests/004-135466_B_old_step-1/RESPONSE.json"
        before = path.read_bytes()
        old = json.loads(json.loads(before)["text"])
        case = json.loads((folder / "cases/135466_B_old_step.json").read_text())["context"]
        simulated_new_output = deepcopy(old)
        simulated_new_output["route_fulfillment"].pop("satisfied")
        simulated_new_output["route_fulfillment"]["correction"] = "Generate another event whose objective is not transport for the existing interview."
        result = validate(case, simulated_new_output)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(path.read_bytes(), before)
        # This fixture is a contract simulation, not a new live Qwen response.
        self.assertTrue(old["route_fulfillment"]["satisfied"])

    def test_derived_negative_semantics_protected_during_citation_repair(self):
        context, raw = load_case("025")
        raw = valid_role(context, raw, negative=True)
        raw["route_fulfillment"]["history_evidence"] = [{
            "turn_ref": context["branch_history"][0]["turn_ref"], "quote": "WRONG QUOTE"}]
        spec = repair_spec(context, raw)
        self.assertFalse(spec["allow_route_semantic_consistency_repair"])
        correction = deepcopy(raw["route_fulfillment"])
        correction["history_evidence"] = [{"turn_ref": context["branch_history"][0]["turn_ref"]}]
        result = review.merge_review_repair(raw, patch_object(context, {"route_fulfillment": correction}), spec)
        self.assertEqual(validate(context, result)["status"], "REJECT")
        for field, replacement in (("primary_goal", "A completely different objective"),
                                    ("reason", "It is really independent"),
                                    ("existing_event", "Another event"),
                                    ("new_progress", False)):
            changed = deepcopy(correction); changed[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "negative_verdict_changed"):
                review.merge_review_repair(raw, patch_object(context, {"route_fulfillment": changed}), spec)

    def test_missing_or_inconsistent_semantics_not_inferred_or_given_pass(self):
        context, raw = load_case("037")
        raw = valid_role(context, raw)
        raw["route_fulfillment"].pop("new_progress")
        result = validate(context, raw)
        self.assertFalse(result["schema_valid"])
        self.assertIsNone(result["route_decision_provenance"])
        context, raw = load_case("037")
        raw = valid_role(context, raw)
        raw["route_fulfillment"].update(serves_existing_event=True, existing_event="Trip")
        result = validate(context, raw)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("relation_service_inconsistent" in message for message in result["schema_failures"]))

    def test_missing_correction_does_not_unlock_sound_negative_semantics(self):
        context, raw = load_case("025")
        raw = valid_role(context, raw, negative=True)
        raw["route_fulfillment"]["correction"] = ""
        spec = repair_spec(context, raw)
        self.assertFalse(spec["allow_route_semantic_consistency_repair"])
        fixed = deepcopy(raw["route_fulfillment"])
        fixed["correction"] = "Create a separate event."
        merged = review.merge_review_repair(raw, patch_object(context, {"route_fulfillment": fixed}), spec)
        self.assertEqual(validate(context, merged)["status"], "REJECT")
        changed = deepcopy(fixed)
        changed.update(relation_to_history="distinct_event", serves_existing_event=False,
                       existing_event="", correction="")
        with self.assertRaisesRegex(ValueError, "negative_verdict_changed"):
            review.merge_review_repair(raw, patch_object(context, {"route_fulfillment": changed}), spec)

    def test_prompt_requests_semantics_without_model_boolean(self):
        context, raw = load_case("037")
        options = kwargs(context); options.pop("history")
        prompt = review.build_review_prompt(context["root_id"], context["route"], context["scene_plan"],
            context["branch_history"], context["new_scene"], **options)
        shape = json.JSONDecoder().raw_decode(prompt.split("\nOUTPUT SHAPE:\n")[1])[0]
        self.assertNotIn("satisfied", shape["route_fulfillment"])
        self.assertIn("Do not output a separate satisfied boolean", prompt)


if __name__ == "__main__":
    unittest.main()
