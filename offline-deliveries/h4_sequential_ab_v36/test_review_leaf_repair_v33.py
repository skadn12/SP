"""Freeze the real v32 rejection while repairing only its missing scalar text."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import realization_review as review


RECORDS = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v32/runs/inherited_a2_b3_01/requests"


def recorded_case():
    first = next(RECORDS.glob("017-*"))
    request = json.loads((first / "REQUEST.json").read_text())
    context = json.JSONDecoder().raw_decode(
        request["messages"][0]["content"].split("\nREVIEW INPUT DATA:\n", 1)[1])[0]
    raw = json.loads((first / "RESPONSE.json").read_text())["text"]
    return context, raw


def validate(context, raw):
    return review.validate_review(raw, context["root_id"], context["route"],
        context["scene_plan"], context["new_scene"],
        dataset_card=context["dataset_card_guidance"], latest_state=context["latest_state"],
        event_state=context["event_state"], history=context["branch_history"],
        require_actor_checks=True, require_goal_checks=True)


def specification(context, raw):
    return review.build_format_repair(validate(context, raw), raw,
        context["scene_plan"], context["branch_history"], context["new_scene"],
        dataset_card=context["dataset_card_guidance"], latest_state=context["latest_state"],
        event_state=context["event_state"], require_actor_checks=True, require_goal_checks=True,
        mode="exact_leaf_patch_v33")


def synthetic_patch(spec):
    return {**spec["identity"], "repair_patches": [
        {"path": item["path"], "value":
         "Create the assigned school-refusal conversation as an independent event; the current automation follow-up serves the old rollout."
         if item["path"][-1] == "correction" else
         "The expected school-related behavior is absent; this dialogue concerns the automation rollout."}
        for item in spec["targets"]]}


class ExactLeafReviewRepairTests(unittest.TestCase):
    def setUp(self):
        self.context, self.raw = recorded_case()
        self.spec = specification(self.context, self.raw)

    def test_real_017_repairs_seven_omissions_and_remains_hard_rejected(self):
        self.assertEqual(self.spec["unrepairable_failures"], [])
        self.assertEqual({tuple(row["path"]) for row in self.spec["targets"]},
            {("route_fulfillment", "correction"),
             *(("card_checks", i, "actor_checks", 0, "action") for i in range(6))})
        before = json.loads(self.raw)
        patch = synthetic_patch(self.spec)
        merged = review.merge_review_repair(self.raw, patch, self.spec)
        result = validate(self.context, merged)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(any("review_route_fulfillment_reject" in row for row in result["failures"]))
        self.assertEqual(sum(row["realized"] is False for row in merged["card_checks"]), 9)
        self.assertTrue(all(row["actor_checks"][0]["satisfied"] is False
                            for row in merged["card_checks"] if row["kind"] == "step"))
        # Undo the only authorized scalar additions: all other content is exact.
        restored = deepcopy(merged)
        for item in patch["repair_patches"]:
            parent = review._leaf_get(restored, item["path"][:-1])
            parent[item["path"][-1]] = review._leaf_get(before, item["path"])
        self.assertEqual(restored, before)
        self.assertEqual(json.loads(self.raw), before)
        self.assertFalse(result["semantic_certified"])

    def test_real_018_full_review_is_still_rejected_and_legacy_is_unchanged(self):
        second = next(RECORDS.glob("018-*"))
        repair = json.loads((second / "RESPONSE.json").read_text())["text"]
        with self.assertRaisesRegex(ValueError, "review_repair_envelope_invalid"):
            review.merge_review_repair(self.raw, repair, self.spec)
        first = next(RECORDS.glob("017-*"))
        legacy = json.loads((first / "FORMAT_REPAIR_SPEC.json").read_text())
        with self.assertRaisesRegex(ValueError, "review_repair_protected_field_changed:reason"):
            review.merge_review_repair(self.raw, repair, legacy)

    def test_unrequested_reason_negative_and_ancestor_cannot_be_patched(self):
        for path, value in [(["reason"], "Revised summary"),
                            (["card_checks", 0, "realized"], True),
                            (["card_checks", 0, "actor_checks", 0, "satisfied"], True),
                            (["route_fulfillment", "history_goal_checks", 0, "relation"], "independent"),
                            (["route_fulfillment"], {}),
                            (["card_checks", 0, "actor_checks", -1, "action"], "Missing"),
                            (["card_checks", True, "actor_checks", 0, "action"], "Missing")]:
            patch = synthetic_patch(self.spec)
            patch["repair_patches"].append({"path": path, "value": value})
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "review_repair_(unrequested_path|path_invalid)"):
                review.merge_review_repair(self.raw, patch, self.spec)

    def test_missing_duplicate_and_nonscalar_updates_are_rejected(self):
        patch = synthetic_patch(self.spec)
        patch["repair_patches"].pop()
        with self.assertRaisesRegex(ValueError, "review_repair_missing_path"):
            review.merge_review_repair(self.raw, patch, self.spec)
        patch = synthetic_patch(self.spec)
        patch["repair_patches"].append(deepcopy(patch["repair_patches"][0]))
        with self.assertRaisesRegex(ValueError, "review_repair_duplicate_path"):
            review.merge_review_repair(self.raw, patch, self.spec)
        for value in ("", "  ", {}, [], True, None):
            patch = synthetic_patch(self.spec)
            patch["repair_patches"][0]["value"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "review_repair_leaf_value_invalid"):
                review.merge_review_repair(self.raw, patch, self.spec)

    def test_changed_source_and_wrong_envelope_identity_are_rejected(self):
        patch = synthetic_patch(self.spec)
        patch["scene_id"] = "S03"
        with self.assertRaisesRegex(ValueError, "review_repair_identity_mismatch"):
            review.merge_review_repair(self.raw, patch, self.spec)
        raw = json.loads(self.raw)
        raw["reason"] += " altered"
        with self.assertRaisesRegex(ValueError, "review_repair_previous_response_changed"):
            review.merge_review_repair(raw, synthetic_patch(self.spec), self.spec)

    def test_citation_patch_selects_reference_without_rewriting_judgments(self):
        # First use the explicit synthetic repair only to make the frozen
        # rejection structurally complete, then isolate a citation typo.
        raw = review.merge_review_repair(self.raw, synthetic_patch(self.spec), self.spec)
        raw["checks"][0]["evidence"][0]["turn_ref"] = "NEW:T999"
        spec = specification(self.context, raw)
        self.assertEqual(spec["unrepairable_failures"], [])
        self.assertEqual([item["path"] for item in spec["targets"]],
                         [["checks", 0, "evidence", 0, "turn_ref"]])
        patch = {**spec["identity"], "repair_patches": [
            {"path": spec["targets"][0]["path"], "value": "NEW:T003"}]}
        merged = review.merge_review_repair(raw, patch, spec)
        result = validate(self.context, merged)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(merged["card_checks"], raw["card_checks"])
        patch["repair_patches"][0]["value"] = "NEW:T999"
        with self.assertRaisesRegex(ValueError, "review_repair_leaf_value_invalid"):
            review.merge_review_repair(raw, patch, spec)

    def test_ambiguous_structural_error_never_authorizes_parent_replacement(self):
        raw = json.loads(self.raw)
        raw["card_checks"][0].pop("actor_checks")
        spec = specification(self.context, raw)
        self.assertTrue(spec["unrepairable_failures"])
        self.assertFalse(any(item["path"] == ["card_checks", 0] for item in spec["targets"]))
        with self.assertRaisesRegex(ValueError, "review_repair_unrepairable_schema"):
            review.merge_review_repair(raw, synthetic_patch(spec), spec)

    def test_prompt_has_independent_patch_task_and_complete_local_schema(self):
        prompt = review.build_format_repair_prompt(self.spec, self.context)
        self.assertNotIn("Review this new scene using the same Qwen model", prompt)
        self.assertNotIn("\nOUTPUT SHAPE:\n", prompt)
        supplied = json.JSONDecoder().raw_decode(prompt.split("\nREVIEW INPUT DATA:\n", 1)[1])[0]
        self.assertEqual(supplied, self.context)
        self.assertEqual(json.loads(prompt.split("\nREVIEW_FORMAT_REPAIR:\n", 1)[1]), self.spec)
        schema = self.spec["output_schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["repair_patches"]["minItems"], 7)
        self.assertEqual(len(schema["properties"]["repair_patches"]["items"]["oneOf"]), 7)


if __name__ == "__main__":
    unittest.main()
