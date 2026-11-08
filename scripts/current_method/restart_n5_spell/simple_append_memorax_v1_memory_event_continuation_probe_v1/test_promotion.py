from __future__ import annotations

import copy
import unittest

from ..simple_append_v1.assembly import assemble_expanded_dialogue, validate_expanded_dialogue
from .promotion import build_promotable_child, validate_promotable_child


def _parent():
    return {
        "root_id": "root",
        "dialogue": {"dialogue_hash": "a" * 64, "turns": [
            {"turn_id": "root:P01:T001", "session_id": "root:P01",
             "speaker": "A", "text": "A discusses the old plan."},
            {"turn_id": "root:P01:T002", "session_id": "root:P01",
             "speaker": "B", "text": "B agrees to the old plan."},
        ]},
    }


def _plan():
    return {"scenes": [
        {"scene_ref": "R01:S01", "memory_event_id": "ME01"},
        {"scene_ref": "R01:S02", "memory_event_id": "ME02"},
        {"scene_ref": "R01:S03", "memory_event_id": "ME03"},
    ]}


def _child():
    return {
        "schema_version": "child_v1",
        "root_id": "root",
        "round_index": 1,
        "scenes": [{
            "scene_ref": f"R01:S{index:02d}",
            "session_ref": f"root:r01:s{index:02d}",
            "memory_event_id": f"ME{index:02d}",
            "dialogue_turns": [{
                "turn_ref": f"R01:S{index:02d}:T001",
                "speaker": "A",
                "text": f"Scene {index} completes a distinct concrete development.",
            }],
        } for index in range(1, 4)],
    }


def _audit(*, passed=("R01:S02",)):
    return {"scenes": [{
        "scene_ref": f"R01:S{index:02d}",
        "scene_verdict": "PASS" if f"R01:S{index:02d}" in passed else "FAIL",
    } for index in range(1, 4)]}


class PromotionContractTest(unittest.TestCase):
    def test_partial_promotion_keeps_only_passed_scene_verbatim(self):
        parent, plan, child = _parent(), _plan(), _child()
        result = build_promotable_child(
            parent=parent, plan=plan, child=child, adherence_audit=_audit())
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["audit"]["promotable_child_ready"])
        self.assertTrue(result["audit"]["partial_promotion"])
        promoted = result["promotable_child"]
        self.assertEqual([s["scene_ref"] for s in promoted["scenes"]],
                         ["R01:S02"])
        self.assertEqual(promoted["scenes"][0], child["scenes"][1])
        self.assertEqual(validate_promotable_child(
            promoted, source_child=child)["valid"], True)

    def test_failed_dependency_blocks_otherwise_passing_scene(self):
        parent, plan, child = _parent(), _plan(), _child()
        plan = copy.deepcopy(plan)
        plan["scenes"][1]["depends_on_scene_refs"] = ["R01:S01"]
        audit = _audit(passed=("R01:S02", "R01:S03"))
        result = build_promotable_child(
            parent=parent, plan=plan, child=child, adherence_audit=audit)
        self.assertEqual(result["audit"]["promoted_scene_refs"], ["R01:S03"])
        row = next(r for r in result["audit"]["scenes"]
                   if r["scene_ref"] == "R01:S02")
        self.assertIn("dependency_not_promotable:R01:S01",
                      row["mechanical_errors"])

    def test_parent_long_duplicate_is_not_promotable(self):
        parent, plan, child = _parent(), _plan(), _child()
        child = copy.deepcopy(child)
        child["scenes"][1]["dialogue_turns"][0]["text"] = \
            "A discusses the old plan."
        result = build_promotable_child(
            parent=parent, plan=plan, child=child,
            adherence_audit=_audit(passed=("R01:S02",)))
        self.assertEqual(result["audit"]["promoted_scene_refs"], [])
        row = next(r for r in result["audit"]["scenes"]
                   if r["scene_ref"] == "R01:S02")
        self.assertIn("exact_or_normalized_parent_duplicate",
                      row["mechanical_errors"])

    def test_all_pass_is_full_promotion(self):
        result = build_promotable_child(
            parent=_parent(), plan=_plan(), child=_child(),
            adherence_audit=_audit(passed=("R01:S01", "R01:S02", "R01:S03")))
        self.assertTrue(result["audit"]["full_round_ready"])
        self.assertFalse(result["audit"]["partial_promotion"])
        self.assertEqual(len(result["promotable_child"]["scenes"]), 3)

    def test_round_two_plan_without_scene_refs_uses_child_round(self):
        plan = {"scenes": [
            {"memory_event_id": f"ND{index:02d}"}
            for index in range(1, 4)
        ]}
        child = _child()
        child["round_index"] = 2
        for index, scene in enumerate(child["scenes"], 1):
            scene["scene_ref"] = f"R02:S{index:02d}"
            scene["session_ref"] = f"root:r02:s{index:02d}"
            scene["memory_event_id"] = f"ND{index:02d}"
            scene["dialogue_turns"][0]["turn_ref"] = \
                f"R02:S{index:02d}:T001"
        audit = {"scenes": [{
            "scene_ref": f"R02:S{index:02d}",
            "scene_verdict": "PASS" if index == 1 else "FAIL",
        } for index in range(1, 4)]}
        result = build_promotable_child(
            parent=_parent(), plan=plan, child=child,
            adherence_audit=audit)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["audit"]["promoted_scene_refs"],
                         ["R02:S01"])
        self.assertEqual(result["audit"]["rejected_scene_refs"],
                         ["R02:S02", "R02:S03"])

    def test_promoted_child_is_still_a_complete_parent_plus_child_view(self):
        result = build_promotable_child(
            parent=_parent(), plan=_plan(), child=_child(),
            adherence_audit=_audit())
        expanded = assemble_expanded_dialogue(
            parent=_parent(), child=result["promotable_child"])
        validation = validate_expanded_dialogue(expanded, parent=_parent())
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(validation["child_turn_count"], 1)
        self.assertEqual(expanded["child_scene_refs"], ["R01:S02"])
        self.assertEqual(expanded["turns"][0]["origin"], "source_grounded")
        self.assertEqual(expanded["turns"][-1]["origin"], "generated_extension")

    def test_promotable_child_cannot_change_scene_or_order(self):
        child = _child()
        value = {**child, "scenes": [child["scenes"][1], child["scenes"][0]]}
        result = validate_promotable_child(value, source_child=child)
        self.assertFalse(result["valid"])
        self.assertIn("promotable_scene_order_changed", result["errors"])


if __name__ == "__main__":
    unittest.main()
