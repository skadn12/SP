"""Offline contract invariants; these checks do not certify model semantics."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

import realization_review as review
from test_actor_review_v29 import ROOT, ROUTE, strict_fixture


DELIVERIES = Path(__file__).resolve().parent.parent
ORIGINAL_REQUEST = (DELIVERIES / "h4_sequential_ab_v37/experiments/fixed_soda_a3_01/requests"
                    / "007-cc-train-episode-104814-B-S01-review-1-1")
V38_REQUESTS = DELIVERIES / "h4_sequential_ab_v38/probes/fixed_review_02/requests"


def old_review():
    spec = importlib.util.spec_from_file_location(
        "immutable_v38_realization_review", DELIVERIES / "h4_sequential_ab_v38/realization_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def sections(prompt):
    shape, context = prompt.split("\nOUTPUT SHAPE:\n", 1)[1].split("\nREVIEW INPUT DATA:\n", 1)
    return json.loads(shape), json.loads(context)


def source_context():
    context = sections(read(ORIGINAL_REQUEST / "REQUEST.json")["messages"][0]["content"])[1]
    card = read(ORIGINAL_REQUEST.parent.parent / "FROZEN_CARD.json")
    return context, card


def prompt(module, context, card):
    return module.build_review_prompt(context["root_id"], context["route"], context["scene_plan"],
        context["branch_history"], context["new_scene"], context["other_branch_added_turns_for_review_only"],
        dataset_card=card, latest_state=context["latest_state"], event_state=context["event_state"],
        purpose_input=context.get("purpose_input"), require_actor_checks=True, require_goal_checks=True)


def validate(module, raw, context, card):
    return module.validate_review(raw, context["root_id"], context["route"], context["scene_plan"],
        context["new_scene"], dataset_card=card, latest_state=context["latest_state"],
        event_state=context["event_state"], history=context["branch_history"],
        require_actor_checks=True, require_goal_checks=True)


class EvidenceFirstContractTests(unittest.TestCase):
    def test_full_history_context_and_output_fields_unchanged(self):
        context, card = source_context()
        before = deepcopy((context, card))
        old_shape, old_context = sections(prompt(old_review(), context, card))
        new_shape, new_context = sections(prompt(review, context, card))
        self.assertEqual(new_shape, old_shape)
        self.assertEqual(new_context, old_context)
        self.assertEqual(new_context, context)
        self.assertEqual(len(new_context["branch_history"]), 132)
        self.assertEqual((context, card), before)

    def test_evidence_precedes_requirements_and_verdicts(self):
        context, card = source_context()
        output, data = sections(prompt(review, context, card))
        keys = list(data)
        self.assertLess(keys.index("branch_history"), keys.index("new_scene"))
        self.assertLess(keys.index("new_scene"), keys.index("latest_state"))
        self.assertLess(keys.index("event_state"), keys.index("scene_plan"))
        self.assertLess(keys.index("scene_plan"), keys.index("dataset_card_guidance"))
        self.assertEqual([row["development_index"] for row in output["checks"]], [1, 2, 3, 4])
        for row in output["checks"]:
            self.assertEqual(list(row), ["development_index", "actor_checks", "evidence", "realized"])
        for row in output["checks"] + output["card_checks"]:
            self.assertLess(list(row).index("evidence"), list(row).index("realized"))
            for actor in row.get("actor_checks", []):
                self.assertEqual(list(actor), ["actor", "action", "evidence_mode", "evidence",
                    "expected_actor", "expected_action", "reason", "satisfied"])

    def test_optional_actor_audit_keeps_same_legacy_schema(self):
        scene, card, realization, _ = strict_fixture()
        old_shape, old_context = sections(old_review().build_review_prompt(
            ROOT, ROUTE, scene, [], realization, dataset_card=card))
        new_shape, new_context = sections(review.build_review_prompt(
            ROOT, ROUTE, scene, [], realization, dataset_card=card))
        self.assertEqual(new_shape, old_shape)
        self.assertEqual(new_context, old_context)
        for row in new_shape["checks"] + new_shape["card_checks"]:
            self.assertNotIn("actor_checks", row)
            self.assertLess(list(row).index("evidence"), list(row).index("realized"))

    def test_recorded_semantic_overclaim_is_not_automatically_changed(self):
        context, card = source_context()
        raw = read(ORIGINAL_REQUEST / "RESPONSE.json")["text"]
        old = validate(old_review(), raw, context, card)
        actual = validate(review, raw, context, card)
        old["version"] = review.REVIEW_VERSION
        self.assertEqual(actual, old)
        self.assertFalse(actual["semantic_certified"])
        self.assertIn("consume", actual["raw_parsed"]["route_fulfillment"]["primary_event"])

    def test_recorded_duplicate_key_rejection_is_not_repaired(self):
        context, card = source_context()
        context["scene_plan"]["developments"][2] = (
            "James admits he is hungry, then actually eats and finishes the found sandwich during this scene.")
        raw = read(V38_REQUESTS / "002-full-review-execution_required_fixture/RESPONSE.json")["text"]
        old = validate(old_review(), raw, context, card)
        actual = validate(review, raw, context, card)
        old["version"] = review.REVIEW_VERSION
        self.assertEqual(actual, old)
        self.assertFalse(actual["schema_valid"])
        self.assertIsNone(actual["raw_parsed"])
        self.assertTrue(any("duplicate_json_key:turn_ref" in item for item in actual["schema_failures"]))


if __name__ == "__main__":
    unittest.main()
