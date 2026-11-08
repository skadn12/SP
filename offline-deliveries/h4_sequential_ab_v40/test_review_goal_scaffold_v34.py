"""Missing goal-check containers never supply or rewrite semantic judgments."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import realization_review as review
from test_review_leaf_repair_v33 import (recorded_case as negative_case,
    validate, synthetic_patch as negative_patch)


RECORDS = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v33/probes/frozen_b_repair_01/requests"


def omitted_case():
    directory = next(RECORDS.glob("002-*"))
    request = json.loads((directory / "REQUEST.json").read_text())
    context = json.JSONDecoder().raw_decode(
        request["messages"][0]["content"].split("\nREVIEW INPUT DATA:\n", 1)[1])[0]
    raw = json.loads((directory / "RESPONSE.json").read_text())["text"]
    return context, raw


def specification(context, raw, mode="exact_leaf_patch_v34"):
    return review.build_format_repair(validate(context, raw), raw,
        context["scene_plan"], context["branch_history"], context["new_scene"],
        dataset_card=context["dataset_card_guidance"], latest_state=context["latest_state"],
        event_state=context["event_state"], require_actor_checks=True, require_goal_checks=True, mode=mode)


def explicit_uncertain_patch(spec):
    """Deliberately rejecting synthetic judgments; no expected-PASS oracle."""
    values = {"historical_goal": "The concrete historical event described in the fixed c-ID.",
              "relation": "uncertain", "reason": "The synthetic reviewer leaves the relationship unresolved."}
    return {**spec["identity"], "repair_patches": [
        {"path": deepcopy(item["path"]), "value": values[item["path"][-1]]} for item in spec["targets"]]}


class MissingGoalScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.context, self.raw = omitted_case()
        self.before = json.loads(self.raw)
        self.spec = specification(self.context, self.raw)

    def test_recorded_s03_only_fixed_ids_are_scaffolded_and_all_fifteen_leaves_are_required(self):
        self.assertEqual(validate(self.context, self.raw)["schema_failures"],
            ["review_history_goal_checks_required:route_fulfillment.history_goal_checks"])
        self.assertEqual(self.spec["unrepairable_failures"], [])
        self.assertEqual(self.spec["scaffolds"][0]["value"],
                         [{"constraint_id": "c" + str(index)} for index in range(1, 6)])
        scaffolded = review._apply_v34_scaffolds(self.before, self.spec)
        for row in scaffolded["route_fulfillment"]["history_goal_checks"]:
            self.assertEqual(set(row), {"constraint_id"})
        self.assertFalse(validate(self.context, scaffolded)["schema_valid"])
        self.assertEqual({tuple(row["path"]) for row in self.spec["targets"]},
                         {("route_fulfillment", "history_goal_checks", index, field)
                          for index in range(5) for field in ("historical_goal", "relation", "reason")})
        self.assertTrue(all(row["previously_present"] is False and row["previous_value"] is None
                            for row in self.spec["targets"]))

    def test_explicit_model_owned_blockers_survive_merge_and_all_other_values_are_unchanged(self):
        merged = review.merge_review_repair(self.raw, explicit_uncertain_patch(self.spec), self.spec)
        checks = merged["route_fulfillment"]["history_goal_checks"]
        self.assertEqual([row["relation"] for row in checks], ["uncertain"] * 5)
        result = validate(self.context, merged)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["parsed"]["route_fulfillment"]["satisfied"])
        self.assertTrue(any("review_route_fulfillment_reject" in code for code in result["failures"]))
        restored = deepcopy(merged)
        restored["route_fulfillment"].pop("history_goal_checks")
        self.assertEqual(restored, self.before)
        self.assertEqual(json.loads(self.raw), self.before)

    def test_every_leaf_requires_an_explicit_value_no_missing_duplicate_or_unknown_paths(self):
        patch = explicit_uncertain_patch(self.spec)
        patch["repair_patches"].pop()
        with self.assertRaisesRegex(ValueError, "review_repair_missing_path"):
            review.merge_review_repair(self.raw, patch, self.spec)
        patch = explicit_uncertain_patch(self.spec)
        patch["repair_patches"].append(deepcopy(patch["repair_patches"][0]))
        with self.assertRaisesRegex(ValueError, "review_repair_duplicate_path"):
            review.merge_review_repair(self.raw, patch, self.spec)
        for path, value in [(["route_fulfillment", "history_goal_checks"], []),
                            (["route_fulfillment", "history_goal_checks", 0, "constraint_id"], "c9"),
                            (["route_fulfillment", "reason"], "Edited"),
                            (["consistent_with_history"], False)]:
            patch = explicit_uncertain_patch(self.spec)
            patch["repair_patches"].append({"path": path, "value": value})
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "review_repair_unrequested_path"):
                review.merge_review_repair(self.raw, patch, self.spec)

    def test_relation_enum_and_nonempty_text_are_enforced(self):
        for field, value in [("relation", "PASS"), ("relation", True),
                             ("historical_goal", ""), ("reason", " "), ("reason", {})]:
            patch = explicit_uncertain_patch(self.spec)
            next(row for row in patch["repair_patches"] if row["path"][-1] == field)["value"] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "review_repair_leaf_value_invalid"):
                review.merge_review_repair(self.raw, patch, self.spec)

    def test_existing_empty_null_or_partial_group_is_never_replaced(self):
        for existing in [None, [], [{"constraint_id": "c1", "historical_goal": "Existing automation goal",
                                    "relation": "serves_existing_goal", "reason": "Existing negative comparison"}]]:
            raw = deepcopy(self.before)
            raw["route_fulfillment"]["history_goal_checks"] = existing
            spec = specification(self.context, raw)
            self.assertEqual(spec["scaffolds"], [])
            self.assertTrue(spec["unrepairable_failures"])
            with self.subTest(existing=existing), self.assertRaisesRegex(ValueError, "review_repair_unrepairable_schema"):
                review.merge_review_repair(raw, {**spec["identity"], "repair_patches": []}, spec)

    def test_fixed_id_scaffold_and_source_context_cannot_be_changed(self):
        patch = explicit_uncertain_patch(self.spec)
        altered = deepcopy(self.spec)
        altered["scaffolds"][0]["value"][0]["relation"] = "independent"
        with self.assertRaisesRegex(ValueError, "review_repair_scaffold_invalid"):
            review.merge_review_repair(self.raw, patch, altered)
        altered = deepcopy(self.spec)
        altered["scaffolds"][0]["value"][0]["constraint_id"] = "c9"
        with self.assertRaisesRegex(ValueError, "review_repair_scaffold_invalid"):
            review.merge_review_repair(self.raw, patch, altered)
        altered = deepcopy(self.spec)
        altered["goal_comparison_context"]["timeline"][0]["statement"] += " changed"
        with self.assertRaisesRegex(ValueError, "review_repair_scaffold_source_changed"):
            review.merge_review_repair(self.raw, patch, altered)
        altered_context = deepcopy(self.context)
        altered_context["event_state"]["timeline"][0]["statement"] += " changed"
        with self.assertRaisesRegex(ValueError, "review_repair_goal_context_mismatch"):
            review.build_format_repair_prompt(self.spec, altered_context)

    def test_v33_mode_remains_fail_closed_for_the_same_missing_group(self):
        spec = specification(self.context, self.raw, mode="exact_leaf_patch_v33")
        self.assertNotIn("scaffolds", spec)
        self.assertTrue(spec["unrepairable_failures"])
        with self.assertRaisesRegex(ValueError, "review_repair_unrepairable_schema"):
            review.merge_review_repair(self.raw, {**spec["identity"], "repair_patches": []}, spec)

    def test_original_seven_leaf_negative_repair_still_works_in_v34(self):
        context, raw = negative_case()
        spec = specification(context, raw)
        self.assertEqual(spec["scaffolds"], [])
        self.assertEqual(len(spec["targets"]), 7)
        self.assertEqual(spec["unrepairable_failures"], [])
        merged = review.merge_review_repair(raw, negative_patch(spec), spec)
        result = validate(context, merged)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(sum(row["realized"] is False for row in merged["card_checks"]), 9)

    def test_prompt_separates_missing_judgments_from_program_owned_containers(self):
        prompt = review.build_format_repair_prompt(self.spec, self.context)
        self.assertIn("Do not default missing rows to a pass", prompt)
        self.assertNotIn("Review this new scene using the same Qwen model", prompt)
        self.assertEqual(json.loads(prompt.split("\nREVIEW_FORMAT_REPAIR:\n", 1)[1]), self.spec)
        output = self.spec["output_schema"]["properties"]["repair_patches"]
        self.assertEqual(output["minItems"], 15)
        self.assertEqual(output["maxItems"], 15)
        self.assertEqual(len(output["items"]["oneOf"]), 15)


if __name__ == "__main__":
    unittest.main()
