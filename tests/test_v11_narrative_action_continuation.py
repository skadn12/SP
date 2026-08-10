from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tests.test_v11_shared_narrative_action_menu import migrated_fixture
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    build_scene_dependency_dag, compile_continuous_round_plan,
    validate_continuous_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_realizer_prompt_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_fresh_e2e_v11 import (
    run_mock_hybrid_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_continuation import (
    PROGRESSION_STAGES, action_slot_ref, validate_and_resolve_round_actions,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    validate_action_instance,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


PROBE_7 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_7")
PROBE_7_SHA = "d2353a677915d6496edd878a61825bc6c34bdb6b43e6acbbc3d65e034ff685b8"


def continuation(source: str, step: str, stage: str = "development"):
    return {"action_mode": "continuation", "continuation_of": source,
            "progression": {"stage": stage, "new_step": step}}


def validation_inputs(scene_count: int = 7):
    inputs, source, placements, plan = migrated_fixture(scene_count)
    refs = {row["entity_ref"] for row in inputs["planning_state"]["entities"]}
    return inputs, source, placements, plan, refs


def continuation_plan():
    inputs, source, placements, plan, refs = validation_inputs(7)
    entities = [row["entity_ref"] for row in inputs["planning_state"]["entities"]
                if row.get("origin") == "source_grounded"]
    if len(entities) < 2:
        entities = sorted(refs)
    plan["scenes"][0]["primary_action"] = {
        "action_mode": "new", "action_type": "develop_character_or_task_state",
        "arguments": {"target": entities[0], "dimension": "coping_confidence",
                      "direction": "increase", "trigger": "initial reassurance"}}
    plan["scenes"][1]["primary_action"] = continuation(
        "S01.primary_action", "A names one practical coping option.", "development")
    plan["scenes"][2]["primary_action"] = continuation(
        "S02.primary_action", "A commits to using that option after the next setback.",
        "consolidation")
    plan["scenes"][3]["secondary_action"] = {
        "action_mode": "new", "action_type": "develop_character_or_task_state",
        "arguments": {"target": entities[1], "dimension": "support_strategy",
                      "direction": "clarify", "trigger": "a personal example"}}
    plan["scenes"][4]["primary_action"] = continuation(
        "S04.secondary_action", "B converts the example into a concrete support plan.",
        "consolidation")
    plan["scenes"][5]["primary_action"] = {
        "action_mode": "new", "action_type": "develop_character_or_task_state",
        "arguments": {"target": entities[0], "dimension": "task_confidence",
                      "direction": "decrease", "trigger": "a new scheduling constraint"}}
    plan["scenes"][6]["primary_action"] = {
        "action_mode": "new", "action_type": "develop_character_or_task_state",
        "arguments": {"target": entities[1], "dimension": "task_confidence",
                      "direction": "increase", "trigger": "a distinct fallback plan"}}
    inputs["planner_output"] = plan
    return inputs, source, placements, plan, refs


def probe7_shadow_plan():
    plan = json.loads((PROBE_7 / "planner_response_parsed.json").read_text())
    plan["round_plan_version"] = "continuous_round_plan_v4"
    for scene in plan["scenes"]:
        for role in ("primary_action", "secondary_action"):
            action = scene.get(role)
            if isinstance(action, dict):
                action["action_mode"] = "new"
    source = json.loads((PROBE_7 / "source_anchor_registry.json").read_text())
    placements = json.loads((PROBE_7 / "placement_choice_registry.json").read_text())
    planning = json.loads((PROBE_7 / "planning_state_view.json").read_text())
    complete = json.loads((PROBE_7 / "complete_dialogue_view.json").read_text())
    refs = {row["entity_ref"] for row in planning["entities"]}
    return plan, source, placements, planning, complete, refs


class NarrativeActionContinuationTests(unittest.TestCase):
    def test_01_action_mode_enum(self):
        new = {"action_mode": "new", "action_type": "develop_character_or_task_state",
               "arguments": {"target": "p", "dimension": "d", "direction": "up",
                             "trigger": "t"}}
        self.assertTrue(validate_action_instance(new)["valid"])
        broken = deepcopy(new); broken["action_mode"] = "resume"
        self.assertFalse(validate_action_instance(broken)["valid"])

    def test_02_new_requires_type_and_arguments(self):
        self.assertFalse(validate_action_instance({"action_mode": "new"})["valid"])

    def test_03_new_forbids_continuation_fields(self):
        value = {"action_mode": "new", "action_type": "develop_character_or_task_state",
                 "arguments": {}, "continuation_of": "S01.primary_action"}
        self.assertFalse(validate_action_instance(value)["valid"])

    def test_04_continuation_requires_source_and_progression(self):
        self.assertTrue(validate_action_instance(continuation(
            "S01.primary_action", "A takes a new concrete step."))["valid"])
        self.assertFalse(validate_action_instance(
            {"action_mode": "continuation"})["valid"])

    def test_05_continuation_forbids_type_and_arguments(self):
        value = continuation("S01.primary_action", "A takes a new step.")
        value["action_type"] = "develop_character_or_task_state"
        value["arguments"] = {}
        self.assertFalse(validate_action_instance(value)["valid"])

    def test_06_action_slot_resolution_and_inheritance(self):
        _, _, _, plan, _ = continuation_plan()
        result = validate_and_resolve_round_actions(plan["scenes"])
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(action_slot_ref(5, "primary_action"), "S05.primary_action")
        resolved = result["resolved_action_slots"]["S05.primary_action"]
        self.assertEqual(resolved["root_action_ref"], "S04.secondary_action")
        self.assertEqual(resolved["resolved_action_type"],
                         "develop_character_or_task_state")
        self.assertEqual(resolved["action_contract"]["action_type"],
                         "develop_character_or_task_state")

    def test_07_invalid_slot_reference(self):
        _, _, _, plan, _ = continuation_plan()
        plan["scenes"][4]["primary_action"]["continuation_of"] = "S04.missing_action"
        result = validate_and_resolve_round_actions(plan["scenes"])
        self.assertTrue(any("continuation_slot_format" in error
                            for error in result["errors"]))

    def test_08_continuation_cycle(self):
        _, _, _, plan, _ = continuation_plan()
        plan["scenes"][0]["primary_action"] = continuation(
            "S02.primary_action", "First cyclic step.")
        plan["scenes"][1]["primary_action"] = continuation(
            "S01.primary_action", "Second cyclic step.")
        result = validate_and_resolve_round_actions(plan["scenes"])
        self.assertTrue(any("continuation_cycle" in error for error in result["errors"]))

    def test_09_dependency_ancestry_required(self):
        _, _, _, plan, _ = continuation_plan()
        scene = plan["scenes"][4]
        scene["predecessor_scene_aliases"] = ["s3"]
        scene["dependency_edges"] = [{"predecessor_scene_alias": "s3",
                                      "dependency_type": "causes"}]
        result = validate_and_resolve_round_actions(plan["scenes"])
        detail = next(row for row in result["error_details"] if row["error"] ==
                      "continuation_source_not_in_dependency_ancestry")
        self.assertEqual(detail["current_scene"], "s5")
        self.assertEqual(detail["referenced_scene"], "s4")
        self.assertEqual(detail["dependency_path"], [])

    def test_10_identical_new_actions_rejected_with_refs(self):
        _, _, _, plan, _ = continuation_plan()
        plan["scenes"][4]["primary_action"] = deepcopy(
            plan["scenes"][3]["secondary_action"])
        result = validate_and_resolve_round_actions(plan["scenes"])
        detail = next(row for row in result["error_details"]
                      if row["error"] == "redundant_independent_action")
        self.assertEqual(detail["current_action_ref"], "S05.primary_action")
        self.assertEqual(detail["first_action_ref"], "S04.secondary_action")
        self.assertIn("action_type", detail["normalized_signature"])

    def test_11_same_type_different_arguments_valid(self):
        _, _, _, plan, _ = continuation_plan()
        result = validate_and_resolve_round_actions(plan["scenes"])
        self.assertTrue(result["valid"], result["errors"])

    def test_12_progression_stage_whitelist(self):
        self.assertEqual(PROGRESSION_STAGES,
                         {"development", "escalation", "consolidation", "resolution"})
        value = continuation("S01.primary_action", "A takes a new step.", "initiation")
        self.assertFalse(validate_action_instance(value)["valid"])

    def test_13_empty_and_mechanical_progression_rejected(self):
        self.assertFalse(validate_action_instance(continuation(
            "S01.primary_action", ""))["valid"])
        _, _, _, plan, _ = continuation_plan()
        plan["scenes"][1]["primary_action"]["progression"]["new_step"] = (
            "continue the development")
        result = validate_and_resolve_round_actions(plan["scenes"])
        self.assertTrue(any("continuation_progression_not_new" in error
                            for error in result["errors"]))

    def test_14_duplicate_desired_outcome_rejected(self):
        _, _, _, plan, _ = continuation_plan()
        plan["scenes"][1]["desired_outcome"] = plan["scenes"][0]["desired_outcome"]
        result = validate_and_resolve_round_actions(plan["scenes"])
        self.assertTrue(any("continuation_desired_outcome_duplicate" in error
                            for error in result["errors"]))

    def test_15_realizer_receives_source_actual_dialogue(self):
        _, source, placements, plan, _ = continuation_plan()
        compiled = compile_continuous_round_plan(
            plan, source_anchor_registry=source,
            placement_choice_registry=placements, round_index=2)
        scene = compiled["scenes"][4]
        source_turn = {"turn_ref": "R02:S04:T001", "scene_ref": "R02:S04",
                       "speaker_ref": "p", "text": "B shares the source story."}
        context = {"p0_required_dialogue": [source_turn], "p1_dependency_dialogue": [],
                   "p1_planning_state": {}, "p2_optional_history": [],
                   "parent_true_tail_refs": [], "same_round_predecessor_refs": ["R02:S04"],
                   "predecessor_dialogue_sha256": {}}
        prompt, _ = build_realizer_prompt_v11(
            scene=scene, context_manifest=context,
            turn_budget={"minimum": 5, "target": 6, "maximum": 7},
            round_blueprint=compiled["round_blueprint"])
        self.assertIn("B shares the source story.", prompt)
        self.assertIn("actual_dialogue_generated_but_not_yet_round_reviewed", prompt)
        self.assertIn("do not replay the source action", prompt)

    def test_16_validator_distinguishes_progression_from_replay(self):
        inputs, _, _, plan, _ = continuation_plan()
        inputs["planner_output"] = plan
        with tempfile.TemporaryDirectory() as temp:
            accepted = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp) / "accepted", case="all_accept")
            repeated = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp) / "repeated",
                case="continuation_repeats_source")
        self.assertTrue(accepted["completed"])
        self.assertTrue(repeated["completed"])
        self.assertGreater(len(accepted["scene_decision_closure"][
            "accepted_scene_refs"]), len(repeated["scene_decision_closure"][
                "accepted_scene_refs"]))

    def test_17_source_rejection_blocks_continuation_successor(self):
        inputs, _, _, plan, _ = continuation_plan()
        inputs["planner_output"] = plan
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="continuation_source_rejected")
        self.assertTrue(result["completed"])
        rejected = result["scene_decision_closure"]["rejected_scene_refs"]
        self.assertIn("R02:S04", rejected)
        self.assertIn("R02:S05", rejected)

    def test_17b_rejected_source_is_a_closure_blocker_not_contract_failure(self):
        inputs, _, _, plan, _ = continuation_plan()
        inputs["planner_output"] = plan
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="all_accept")
        output = deepcopy(result["round_validator_output"])
        source_review = next(row for row in output["scene_reviews"]
                             if row["scene_alias"] == "s4")
        source_review["decision"] = "reject"
        source_review["historical_connection_status"] = "uncertain"
        source_review["continuity_status"] = "uncertain"
        source_review["new_development_status"] = "uncertain"
        source_review["reason_codes"] = ["uncertain_review"]
        validation = validate_round_validator_output(
            output, realized_scenes=result["realized_scenes"],
            review_world_view=result["review_world_view"],
            local_gate_results=result["local_scene_gate_results"])
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertTrue(any(
            row["reason"] == "source_scene_not_accepted"
            for row in validation["continuation_consistency_blockers"]))

    def test_18_continuation_does_not_directly_create_operation(self):
        _, source, placements, plan, _ = continuation_plan()
        compiled = compile_continuous_round_plan(
            plan, source_anchor_registry=source,
            placement_choice_registry=placements)
        self.assertTrue(all(scene["compiled_proposals"] == []
                            for scene in compiled["scenes"]))
        self.assertEqual(compiled["scenes"][4]["primary_action"]["action_mode"],
                         "continuation")

    def test_19_five_six_seven_scene_fixtures(self):
        for count in (5, 6, 7):
            inputs, source, placements, plan, refs = validation_inputs(count)
            result = validate_continuous_round_plan(
                plan, source_anchor_registry=source,
                placement_choice_registry=placements,
                authoritative_entity_refs=refs)
            self.assertTrue(result["valid"], (count, result["errors"]))

    def test_20_probe7_original_detects_redundant_independent_action(self):
        plan, source, placements, _, _, refs = probe7_shadow_plan()
        result = validate_continuous_round_plan(
            plan, source_anchor_registry=source,
            placement_choice_registry=placements,
            authoritative_entity_refs=refs)
        detail = next(row for row in result["action_continuation_validation"][
            "error_details"] if row["error"] == "redundant_independent_action")
        self.assertEqual(detail["current_action_ref"], "S05.primary_action")
        self.assertEqual(detail["first_action_ref"], "S04.secondary_action")

    def _assert_probe7_shadow_complete(self, plan):
        _, source, placements, planning, complete, refs = probe7_shadow_plan()
        result = validate_continuous_round_plan(
            plan, source_anchor_registry=source,
            placement_choice_registry=placements,
            authoritative_entity_refs=refs)
        self.assertTrue(result["valid"], result["errors"])
        compiled = compile_continuous_round_plan(
            plan, source_anchor_registry=source,
            placement_choice_registry=placements)
        dag = build_scene_dependency_dag(compiled)
        self.assertEqual(len(dag["nodes"]), 7)
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled, source_anchor_registry=source,
            placement_choice_registry=placements, planning_state=planning,
            complete_dialogue=complete)
        self.assertTrue(structural["valid"], structural["errors"])

    def test_21_probe7_continuation_shadow_complete(self):
        plan, *_ = probe7_shadow_plan()
        plan["scenes"][4]["primary_action"] = continuation(
            "S04.secondary_action",
            "A turns the earlier reassurance into a concrete coping decision.",
            "consolidation")
        self._assert_probe7_shadow_complete(plan)

    def test_22_probe7_distinct_new_shadow_complete(self):
        plan, *_ = probe7_shadow_plan()
        plan["scenes"][4]["primary_action"]["arguments"]["trigger"] = (
            "A independently chooses a concrete coping routine after a new setback")
        self._assert_probe7_shadow_complete(plan)

    def test_23_complete_seven_scene_mock_runtime(self):
        inputs, _, _, plan, _ = continuation_plan()
        inputs["planner_output"] = plan
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="all_accept")
        self.assertTrue(result["completed"], result.get("failure_stage"))
        self.assertEqual(len(result["realized_scenes"]), 7)
        self.assertTrue(any(scene["primary_action"]["action_mode"] == "continuation"
                            for scene in result["realized_scenes"]))
        self.assertGreaterEqual(len(result["durable_update_compile_result"][
            "accepted_updates"]), 1)
        self.assertGreaterEqual(sum(1 for row in result["transactions"]
                                    if row.get("status") == "COMMITTED"), 1)
        self.assertEqual(len(result["promotion_v2_result"][
            "promotable_child_scenes"]), 7)
        self.assertEqual(result["api_calls"], 0)

    def test_24_probe7_tree_remains_read_only(self):
        self.assertTrue(PROBE_7.is_dir())
        self.assertEqual(hash_tree(PROBE_7)["tree_sha256"], PROBE_7_SHA)


if __name__ == "__main__":
    unittest.main()
