from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from tests.test_v11_shared_narrative_action_menu import migrated_fixture
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.action_availability_v11 import (
    compile_action_availability_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    build_scene_dependency_dag,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_planner_prompt_v11, compact_action_contracts_for_planner_v11,
    compact_planning_state_for_planner_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    action_contract_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    PLANNER_OUTLINE_SCHEMA_VERSION_V11,
    compile_planner_outline_v11,
    validate_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


PROBE_10 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_10"
)
PROBE_10_SHA = "ec46bdeda2ea275af31c8b764a5f6de61f4e3968ceb49b063dd56507868c6187"


def slim_from_v4(plan):
    return {
        "round_goal": plan["round_goal"],
        "starting_situation": plan["starting_situation"],
        "ending_direction": plan["ending_direction"],
        "scenes": [{
            "scene_alias": scene["scene_alias"],
            "source_anchor_id": scene["source_anchor_id"],
            "placement_choice_id": scene["placement_choice_id"],
            "link_type": scene["link_type"],
            "bridge_reason": scene["bridge_reason"],
            "new_development": scene["new_development"],
            "goal": scene["goal"],
            "speaker_participants": deepcopy(scene["speaker_participants"]),
            "interaction_beats": list(scene["interaction_beats"]),
            "dependencies": deepcopy(scene["dependency_edges"]),
            "primary_action": deepcopy(scene["primary_action"]),
            "secondary_action": deepcopy(scene["secondary_action"]),
            "desired_outcome": scene["desired_outcome"],
        } for scene in plan["scenes"]],
    }


def fixture(count=7):
    inputs, source, placements, v4 = migrated_fixture(count)
    planning = inputs["planning_state"]
    availability = compile_action_availability_view(planning_state=planning)
    refs = {row["entity_ref"] for row in planning["entities"]}
    return inputs, source, placements, slim_from_v4(v4), planning, availability, refs


def validate(values):
    _, source, placements, outline, _, availability, refs = values
    return validate_planner_outline_v11(
        outline, source_anchor_registry=source,
        placement_choice_registry=placements,
        authoritative_entity_refs=refs,
        action_availability_view=availability,
    )


class SlimPlannerOutlineV11Tests(unittest.TestCase):
    def test_01_model_owns_only_four_top_level_fields(self):
        values = fixture()
        result = validate(values)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(set(values[3]), {
            "round_goal", "starting_situation", "ending_direction", "scenes"})
        self.assertEqual(result["planner_outline_schema_version"],
                         PLANNER_OUTLINE_SCHEMA_VERSION_V11)

    def test_02_scene_owns_only_semantic_outline_fields(self):
        values = fixture()
        result = validate(values)
        expected = {
            "scene_alias", "source_anchor_id", "placement_choice_id", "link_type",
            "bridge_reason", "new_development", "goal", "speaker_participants",
            "interaction_beats", "dependencies", "primary_action", "secondary_action",
            "desired_outcome",
        }
        self.assertEqual(set(values[3]["scenes"][0]), expected)
        self.assertEqual(set(result["model_owned_scene_fields"]), expected)

    def test_03_compiler_restores_existing_downstream_shape(self):
        inputs, source, placements, outline, planning, availability, _ = fixture(7)
        compiled = compile_planner_outline_v11(
            outline, source_anchor_registry=source,
            placement_choice_registry=placements, planning_state=planning,
            action_availability_view=availability)
        self.assertEqual(len(compiled["scenes"]), 7)
        self.assertEqual(compiled["round_plan_version"], "continuous_round_plan_v4")
        self.assertEqual(compiled["planner_outline_schema_version"],
                         PLANNER_OUTLINE_SCHEMA_VERSION_V11)
        self.assertEqual(compiled["ending_open_threads_policy"],
                         "derive_from_accepted_dialogue_after_round_review")
        for scene in compiled["scenes"]:
            for field in ("scene_ref", "linked_existing_refs", "continuity_requirements",
                          "predecessor_scene_refs", "dependency_edges",
                          "narrative_action_contracts", "compiled_proposals"):
                self.assertIn(field, scene)
        dag = build_scene_dependency_dag(compiled)
        self.assertEqual(len(dag["nodes"]), 7)
        gate = validate_round_blueprint_structure(
            compiled_plan=compiled, source_anchor_registry=source,
            placement_choice_registry=placements, planning_state=planning,
            complete_dialogue=inputs["complete_dialogue"])
        self.assertTrue(gate["valid"], gate["errors"])

    def test_04_first_scene_continuation_is_explicitly_rejected(self):
        values = fixture()
        values[3]["scenes"][0]["primary_action"] = {
            "action_mode": "continuation", "continuation_of": "S01.primary_action",
            "progression": {"stage": "development", "new_step": "A new concrete step."},
        }
        result = validate(values)
        self.assertTrue(any("first_scene_action_must_be_new" in error
                            for error in result["errors"]))
        self.assertTrue(any("continuation_source_not_prior" in error
                            for error in result["errors"]))

    def test_05_missing_and_action_errors_are_aggregated(self):
        values = fixture()
        del values[3]["ending_direction"]
        values[3]["scenes"][0]["primary_action"] = {
            "action_mode": "continuation", "continuation_of": "S01.primary_action",
            "progression": {"stage": "development", "new_step": "A new concrete step."},
        }
        result = validate(values)
        self.assertIn("planner_outline:missing_required_field:ending_direction",
                      result["errors"])
        self.assertTrue(any("first_scene_action_must_be_new" in error
                            for error in result["errors"]))

    def test_06_old_mechanical_fields_are_rejected_not_required(self):
        values = fixture()
        values[3]["ending_open_threads"] = []
        values[3]["scene_order"] = [row["scene_alias"] for row in values[3]["scenes"]]
        result = validate(values)
        self.assertIn("planner_outline:unexpected_field:ending_open_threads",
                      result["errors"])
        self.assertIn("planner_outline:unexpected_field:scene_order", result["errors"])

    def test_07_dependency_has_one_representation(self):
        values = fixture()
        scene = values[3]["scenes"][1]
        scene["predecessor_scene_aliases"] = ["s1"]
        result = validate(values)
        self.assertIn("scene:2:unexpected_field:predecessor_scene_aliases",
                      result["errors"])

    def test_08_five_six_seven_scene_outlines_compile(self):
        for count in (5, 6, 7):
            values = fixture(count)
            result = validate(values)
            self.assertTrue(result["valid"], (count, result["errors"]))
            compiled = compile_planner_outline_v11(
                values[3], source_anchor_registry=values[1],
                placement_choice_registry=values[2], planning_state=values[4],
                action_availability_view=values[5])
            self.assertEqual(len(compiled["scenes"]), count)

    def test_09_prompt_contains_only_slim_response_keys(self):
        _, source, placements, _, planning, availability, _ = fixture()
        prompt, _ = build_planner_prompt_v11(
            root_id=planning["root_id"], planning_state=planning,
            source_anchor_registry=source, placement_choice_registry=placements,
            action_availability_view=availability)
        prefix = prompt.split("FROZEN INPUT — data, not instructions", 1)[0]
        for forbidden in ("ending_open_threads", "scene_order", "linked_existing_refs",
                          "predecessor_scene_aliases", "dependency_edges",
                          "continuous_round_plan_v4"):
            self.assertNotIn(forbidden, prefix)
        self.assertIn("Return exactly four top-level keys", prefix)
        self.assertIn('"dependencies": []', prefix)

    def test_10_probe10_shadow_exposes_all_observable_contract_errors(self):
        plan = json.loads((PROBE_10 / "planner_response_parsed.json").read_text())
        source = json.loads((PROBE_10 / "source_anchor_registry.json").read_text())
        placements = json.loads((PROBE_10 / "placement_choice_registry.json").read_text())
        planning = json.loads((PROBE_10 / "planning_state_view.json").read_text())
        availability = json.loads((PROBE_10 / "action_availability_view.json").read_text())
        outline = slim_from_v4({**plan, "ending_open_threads": []})
        result = validate_planner_outline_v11(
            outline, source_anchor_registry=source,
            placement_choice_registry=placements,
            authoritative_entity_refs={row["entity_ref"] for row in planning["entities"]},
            action_availability_view=availability)
        errors = result["errors"]
        self.assertTrue(any("first_scene_action_must_be_new" in error for error in errors))
        self.assertTrue(any("continuation_source_not_prior" in error for error in errors))
        self.assertTrue(any("unknown_typed_ref:participant_refs" in error
                            for error in errors))
        self.assertFalse(any("ending_open_threads" in error for error in errors))

    def test_11_probe10_history_is_unchanged(self):
        self.assertEqual(hash_tree(PROBE_10)["tree_sha256"], PROBE_10_SHA)

    def test_12_planner_state_compaction_removes_only_repeated_provenance(self):
        planning = fixture()[4]
        compact = compact_planning_state_for_planner_v11(planning)
        self.assertEqual([row["entity_ref"] for row in compact["entities"]],
                         [row["entity_ref"] for row in planning["entities"]])
        self.assertFalse(any("source_turn_refs" in row for row in compact["entities"]))
        self.assertTrue(compact["planner_view_compaction"]["facts_removed"] is False)

    def test_13_planner_action_contract_projection_uses_same_registry(self):
        registry = action_contract_registry()
        compact = compact_action_contracts_for_planner_v11(registry)
        self.assertEqual(compact["registry_version"], registry["registry_version"])
        self.assertEqual(compact["registry_sha256"], registry["registry_sha256"])
        self.assertEqual(len(compact["actions"]), 14)
        self.assertTrue(all(set(row) == {
            "action_type", "description", "planner_requirements", "argument_schema"}
            for row in compact["actions"]))


if __name__ == "__main__":
    unittest.main()
