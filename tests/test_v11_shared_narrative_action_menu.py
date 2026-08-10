from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tests.test_world_memory_slim_v11_hybrid_runtime import fixture_inputs
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    build_scene_dependency_dag, compile_continuous_round_plan,
    validate_continuous_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import build_realizer_prompt_v11
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_fresh_e2e_v11 import run_mock_hybrid_v11
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    NARRATIVE_ACTION_CONTRACTS, action_contract_registry,
    migrate_legacy_action_hints_to_narrative_actions, validate_action_instance,
    validate_action_contract_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.offline_preflight_v11 import build_round_blueprint_fixture
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.placement_choice_registry import (
    compile_placement_choice_registry, compile_source_anchor_registry,
    migrate_legacy_round_plan_source_placement,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


PROBE_6 = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_61555_v11_planner_probe_6")
PROBE_6_SHA = "9690e2dcb6cc0f9dd2afb074b9c413802daf0b265f6c13a5c8c70c84f3504ed0"


def migrated_fixture(scene_count: int = 6):
    inputs = fixture_inputs()
    source = compile_source_anchor_registry(inputs["anchor_registry"])
    placements = compile_placement_choice_registry(
        complete_dialogue=inputs["complete_dialogue"],
        anchor_registry=inputs["anchor_registry"])
    old = migrate_legacy_round_plan_source_placement(
        inputs["planner_output"], placement_choice_id="PC_TAIL_AFTER")
    plan = migrate_legacy_action_hints_to_narrative_actions(
        old, action_menu=inputs["action_menu"])
    if scene_count != len(plan["scenes"]):
        entities = [row["entity_ref"] for row in inputs["planning_state"]["entities"]]
        plan = build_round_blueprint_fixture(
            source_anchor_id=source["source_anchors"][0]["source_anchor_id"],
            placement_choice_id="PC_TAIL_AFTER", entity_refs=entities,
            single_action_selection={}, scene_count=scene_count)
    return inputs, source, placements, plan


class SharedNarrativeActionMenuTests(unittest.TestCase):
    def test_01_fourteen_action_types_are_complete(self):
        self.assertEqual(len(NARRATIVE_ACTION_CONTRACTS), 14)
        self.assertTrue(validate_action_contract_registry()["valid"])
        self.assertEqual({row["action_type"] for row in action_contract_registry()["actions"]},
                         set(NARRATIVE_ACTION_CONTRACTS))

    def test_02_every_argument_schema_is_enforced(self):
        for action_type, contract in NARRATIVE_ACTION_CONTRACTS.items():
            properties = contract["argument_schema"]["properties"]
            value = {}
            for key, schema in properties.items():
                kind = schema["type"]
                if kind == "object":
                    value[key] = {name: (sub.get("enum") or ["x"])[0]
                                  for name, sub in schema["properties"].items()}
                elif kind == "array":
                    value[key] = ["p1", "p2"][:max(1, schema.get("minItems", 1))]
                elif isinstance(kind, list):
                    value[key] = None
                else:
                    value[key] = (schema.get("enum") or ["x"])[0]
            action = {"action_mode": "new", "action_type": action_type,
                      "arguments": value}
            self.assertTrue(validate_action_instance(action)["valid"], action_type)
            missing = deepcopy(action)
            missing["arguments"].pop(next(iter(properties)))
            self.assertFalse(validate_action_instance(missing)["valid"], action_type)

    def test_03_primary_required_secondary_nullable(self):
        inputs, source, placements, plan = migrated_fixture()
        refs = {row["entity_ref"] for row in inputs["planning_state"]["entities"]}
        self.assertTrue(validate_continuous_round_plan(
            plan, source_anchor_registry=source, placement_choice_registry=placements,
            authoritative_entity_refs=refs)["valid"])
        broken = deepcopy(plan); broken["scenes"][0].pop("primary_action")
        self.assertFalse(validate_continuous_round_plan(
            broken, source_anchor_registry=source, placement_choice_registry=placements,
            authoritative_entity_refs=refs)["valid"])
        self.assertIsNone(plan["scenes"][0]["secondary_action"])

    def test_04_action_type_reuse_has_no_round_uniqueness_gate(self):
        inputs, source, placements, plan = migrated_fixture()
        refs = {row["entity_ref"] for row in inputs["planning_state"]["entities"]}
        shared = {"action_mode": "new", "action_type": "develop_character_or_task_state", "arguments": {
            "target": sorted(refs)[0], "dimension": "first", "direction": "up", "trigger": "a"}}
        plan["scenes"][0]["primary_action"] = deepcopy(shared)
        plan["scenes"][1]["primary_action"] = deepcopy(shared)
        plan["scenes"][1]["primary_action"]["arguments"]["dimension"] = "second"
        result = validate_continuous_round_plan(
            plan, source_anchor_registry=source, placement_choice_registry=placements,
            authoritative_entity_refs=refs)
        self.assertTrue(result["valid"], result["errors"])
        self.assertFalse(any("duplicate_action" in error for error in result["errors"]))

    def test_05_old_planner_allocation_fields_are_rejected(self):
        inputs, source, placements, plan = migrated_fixture()
        refs = {row["entity_ref"] for row in inputs["planning_state"]["entities"]}
        for field in ("action_id", "bundle_id", "candidate_update_hints"):
            broken = deepcopy(plan); broken["scenes"][0][field] = []
            result = validate_continuous_round_plan(
                broken, source_anchor_registry=source,
                placement_choice_registry=placements, authoritative_entity_refs=refs)
            self.assertFalse(result["valid"])

    def test_06_realizer_receives_action_and_same_contract(self):
        inputs, source, placements, plan = migrated_fixture()
        compiled = compile_continuous_round_plan(
            plan, source_anchor_registry=source, placement_choice_registry=placements)
        scene = compiled["scenes"][0]
        prompt, _ = build_realizer_prompt_v11(
            scene=scene, context_manifest={"p0_required_dialogue": [],
                "p1_dependency_dialogue": [], "p1_planning_state": {},
                "p2_optional_history": [], "parent_true_tail_refs": [],
                "same_round_predecessor_refs": [], "predecessor_dialogue_sha256": {}},
            turn_budget={"minimum": 5, "target": 6, "maximum": 7},
            round_blueprint=compiled["round_blueprint"])
        self.assertIn('"primary_action"', prompt)
        self.assertIn('"narrative_action_contracts"', prompt)
        self.assertNotIn('"candidate_update_hints"', prompt)

    def test_07_primary_not_realized_rejects_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(**fixture_inputs(), run_root=Path(temp),
                                         case="primary_action_not_realized")
        self.assertTrue(result["completed"])
        self.assertIn("R02:S02", result["scene_decision_closure"]["rejected_scene_refs"])

    def test_08_secondary_partial_can_retain_scene(self):
        inputs, _, _, plan = migrated_fixture()
        plan["scenes"][1]["secondary_action"] = {
            "action_mode": "new", "action_type": "reveal_or_transfer_information", "arguments": {
                "sender": plan["scenes"][1]["speaker_participants"][0]["participant_ref"],
                "receiver": plan["scenes"][1]["speaker_participants"][1]["participant_ref"],
                "information": "a partial scheduling detail", "transfer_mode": "conversation"}}
        inputs["planner_output"] = plan
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(**inputs, run_root=Path(temp),
                                         case="secondary_action_partial")
        self.assertTrue(result["completed"])
        self.assertIn("R02:S02", result["scene_decision_closure"]["accepted_scene_refs"])

    def test_09_accepted_scene_can_have_zero_durable_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(**fixture_inputs(), run_root=Path(temp),
                                         case="accept_no_updates")
        self.assertEqual(result["durable_update_compile_result"]["accepted_updates"], [])
        self.assertEqual(result["transactions"], [])
        self.assertEqual(len(result["promotion_v2_result"]["promotable_child_scenes"]), 6)

    def test_10_connected_person_stays_pre_review_narrative_only(self):
        inputs, source, placements, plan = migrated_fixture()
        existing = plan["scenes"][0]["speaker_participants"][0]["participant_ref"]
        plan["scenes"][0]["primary_action"] = {
            "action_mode": "new", "action_type": "introduce_connected_person", "arguments": {
                "participant_alias": "new_person_c", "display_name": "New person C",
                "entity_role": "reported project participant",
                "connection_type": "existing_person",
                "connection_target_ref": existing,
                "connection_description": "introduced through an established participant"}}
        disconnected = deepcopy(plan)
        disconnected["scenes"][0]["primary_action"]["arguments"][
            "connection_target_ref"] = "missing:world:ref"
        invalid = validate_continuous_round_plan(
            disconnected, source_anchor_registry=source,
            placement_choice_registry=placements,
            authoritative_entity_refs={row["entity_ref"]
                                       for row in inputs["planning_state"]["entities"]})
        self.assertFalse(invalid["valid"])
        self.assertTrue(any("unknown_world_ref" in error for error in invalid["errors"]))
        compiled = compile_continuous_round_plan(
            plan, source_anchor_registry=source, placement_choice_registry=placements)
        self.assertEqual(compiled["scenes"][0]["compiled_proposals"], [])
        self.assertEqual(compiled["scenes"][0]["primary_action"]["resolved_action_type"],
                         "introduce_connected_person")

    def test_11_state_development_without_verified_candidate_does_not_update(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(**fixture_inputs(), run_root=Path(temp),
                                         case="accept_no_updates")
        self.assertEqual(result["memory_final"], fixture_inputs()["authoritative_memory"])

    def test_12_probe6_shadow_removes_a0008_allocation_without_rewriting_history(self):
        plan = json.loads((PROBE_6 / "planner_response_parsed.json").read_text())
        menu = json.loads((PROBE_6 / "action_menu.json").read_text())
        source = json.loads((PROBE_6 / "source_anchor_registry.json").read_text())
        placements = json.loads((PROBE_6 / "placement_choice_registry.json").read_text())
        planning = json.loads((PROBE_6 / "planning_state_view.json").read_text())
        complete = json.loads((PROBE_6 / "complete_dialogue_view.json").read_text())
        migrated = migrate_legacy_action_hints_to_narrative_actions(plan, action_menu=menu)
        result = validate_continuous_round_plan(
            migrated, source_anchor_registry=source,
            placement_choice_registry=placements,
            authoritative_entity_refs={row["entity_ref"] for row in planning["entities"]})
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(migrated["scenes"][1]["primary_action"]["action_type"],
                         "develop_character_or_task_state")
        self.assertEqual(migrated["scenes"][3]["primary_action"]["action_type"],
                         "develop_character_or_task_state")
        self.assertNotEqual(migrated["scenes"][1]["primary_action"]["arguments"],
                            migrated["scenes"][3]["primary_action"]["arguments"])
        compiled = compile_continuous_round_plan(
            migrated, source_anchor_registry=source,
            placement_choice_registry=placements, round_index=1)
        dag = build_scene_dependency_dag(compiled)
        self.assertEqual(len(dag["nodes"]), len(migrated["scenes"]))
        self.assertTrue(dag["scene_dependency_dag_sha256"])
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled, source_anchor_registry=source,
            placement_choice_registry=placements, planning_state=planning,
            complete_dialogue=complete)
        self.assertTrue(structural["valid"], structural["errors"])
        self.assertEqual(hash_tree(PROBE_6)["tree_sha256"], PROBE_6_SHA)

    def test_13_complete_five_six_seven_scene_fixtures(self):
        for count in (5, 6, 7):
            inputs, source, placements, plan = migrated_fixture(count)
            result = validate_continuous_round_plan(
                plan, source_anchor_registry=source,
                placement_choice_registry=placements,
                authoritative_entity_refs={row["entity_ref"]
                                           for row in inputs["planning_state"]["entities"]})
            self.assertTrue(result["valid"], (count, result["errors"]))

    def test_14_complete_seven_scene_mock_runtime(self):
        inputs, _, _, plan = migrated_fixture(7)
        entities = [row["entity_ref"] for row in inputs["planning_state"]["entities"]]
        thread_ref, task_ref = "fixture:thread:1", "fixture:task:1"
        for anchor in inputs["anchor_registry"]["anchors"]:
            anchor.setdefault("linked_thread_refs", []).append(thread_ref)
            anchor.setdefault("linked_task_refs", []).append(task_ref)
        inputs["planning_state"].setdefault("open_threads", []).append(
            {"memory_ref": thread_ref, "subject_refs": entities})
        inputs["planning_state"].setdefault("tasks", []).append(
            {"memory_ref": task_ref, "subject_refs": entities})
        plan["scenes"][0]["primary_action"] = {"action_mode": "new", "action_type": "continue_open_thread",
            "arguments": {"thread_ref": thread_ref, "continuation_mode": "follow_up",
                          "new_progress": "they identify a concrete next question"}}
        plan["scenes"][1]["primary_action"] = {"action_mode": "new", "action_type": "advance_shared_task",
            "arguments": {"task_ref": task_ref, "progress_type": "new_obstacle",
                          "responsible_participants": entities,
                          "new_step_or_obstacle": "a schedule conflict must be handled"}}
        plan["scenes"][2]["primary_action"] = {"action_mode": "new", "action_type": "introduce_external_constraint",
            "arguments": {"constraint": "the budget is reduced", "affected_participants": entities,
                          "affected_plan_or_task": "the shared task"}}
        plan["scenes"][3]["primary_action"] = {"action_mode": "new", "action_type": "add_information_asymmetry",
            "arguments": {"informed_participant": entities[0],
                          "uninformed_participant": entities[1],
                          "hidden_information": "the deadline moved",
                          "information_source": "a private message"}}
        plan["scenes"][4]["primary_action"] = {"action_mode": "new", "action_type": "reveal_or_transfer_information",
            "arguments": {"sender": entities[0], "receiver": entities[1],
                          "information": "the deadline moved", "transfer_mode": "directly"}}
        plan["scenes"][5]["primary_action"] = {"action_mode": "new", "action_type": "create_misunderstanding",
            "arguments": {"mistaken_participant": entities[1],
                          "mistaken_belief": "the change was deliberate",
                          "misunderstanding_source": "an incomplete message",
                          "actual_situation": "the change was externally imposed"}}
        plan["scenes"][5]["secondary_action"] = {"action_mode": "new", "action_type": "create_or_escalate_conflict",
            "arguments": {"participants": entities, "conflict_subject": "responsibility",
                          "conflict_source": "the misunderstanding",
                          "escalation_or_creation": "create"}}
        plan["scenes"][6]["primary_action"] = {"action_mode": "new", "action_type": "develop_character_or_task_state",
            "arguments": {"target": entities[0], "dimension": "task_confidence",
                          "direction": "decrease", "trigger": "the conflict"}}
        inputs["planner_output"] = plan
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(**inputs, run_root=Path(temp), case="all_accept")
        self.assertTrue(result["completed"], result.get("failure_stage"))
        self.assertEqual(len(result["realized_scenes"]), 7)
        self.assertEqual(result["mock_calls"]["evidence_committer"], 0)
        self.assertTrue(result["durable_update_compile_result"][
            "post_review_internal_operation_compilation"])
        self.assertGreaterEqual(len(result["durable_update_compile_result"][
            "accepted_updates"]), 1)
        self.assertGreaterEqual(len(result["transactions"]), 1)
        self.assertTrue(all(row["status"] == "COMMITTED"
                            for row in result["transactions"]))


if __name__ == "__main__":
    unittest.main()
