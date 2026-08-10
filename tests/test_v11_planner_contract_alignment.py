from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    compile_continuous_round_plan, validate_continuous_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    PLANNER_PROMPT_VERSION_V11, PLANNER_RETRY_PROMPT_VERSION_V11,
    build_planner_retry_prompt_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import DEFAULT_INPUT
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    migrate_legacy_action_hints_to_narrative_actions,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_only_probe_v11 import build_planner_probe_input
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.placement_choice_registry import migrate_legacy_round_plan_source_placement
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import validate_round_blueprint_structure


PROBE_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_61555_slim_continuous_expansion_v11_planner_transport_probe_1")
PROBE_TREE_SHA = "653514bfbd3b7325bab1315816207b66b01db6022cb9d1f515663df3c437a0aa"
PROBE_2_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_61555_v11_planner_probe_2")
PROBE_2_TREE_SHA = "43eef09b5ff1b5dde62e0946bed98d754f30bc2b8a27e65c3d55abdd9d974256"
PROBE_3_ROOT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_61555_v11_planner_probe_3")
PROBE_3_TREE_SHA = "77a2d96cd65cd845c2d3fe52e49937e7df24a96652e74965cbbf32887be25fb7"


class PlannerContractAlignmentV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = build_planner_probe_input(DEFAULT_INPUT)
        cls.old_plan = json.loads((PROBE_ROOT / "planner_response_parsed.json").read_text())
        cls.entity_refs = {row["entity_ref"] for row in cls.values["role_table"]}

    def validate(self, plan):
        return validate_continuous_round_plan(
            plan, source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            authoritative_entity_refs=self.entity_refs)

    def migrated(self):
        plan = migrate_legacy_round_plan_source_placement(
            self.old_plan, placement_choice_id="PC_TAIL_AFTER")
        for scene in plan["scenes"]:
            scene["dependency_edges"] = [{"predecessor_scene_alias": alias,
                "dependency_type": "continues"}
                for alias in scene["predecessor_scene_aliases"]]
            scene["linked_existing_refs"]["thread_refs"] = []
        return migrate_legacy_action_hints_to_narrative_actions(
            plan, action_menu=self.values["action_menu"])

    def test_prompt_versions_are_shared_action_contract(self):
        self.assertIn("shared_narrative_action", PLANNER_PROMPT_VERSION_V11)
        self.assertIn("shared_narrative_action", PLANNER_RETRY_PROMPT_VERSION_V11)
        self.assertIn("source_novelty", PLANNER_PROMPT_VERSION_V11)
        self.assertIn("already mentioned anywhere in the source dialogue",
                      self.values["prompt"])

    def test_prompt_exposes_shared_registry_not_concrete_menu(self):
        prompt = self.values["prompt"]
        self.assertIn('"shared_narrative_action_contracts"', prompt)
        self.assertIn('"primary_action"', prompt)
        self.assertIn('"secondary_action"', prompt)
        self.assertNotIn('"optional_persistent_actions"', prompt)
        self.assertNotIn('"allowed_atomic_bundles"', prompt)

    def test_live_schema_rejects_old_action_fields(self):
        old = migrate_legacy_round_plan_source_placement(
            self.old_plan, placement_choice_id="PC_TAIL_AFTER")
        result = self.validate(old)
        self.assertFalse(result["valid"])
        self.assertTrue(any("candidate_update_hints" in error for error in result["errors"]))

    def test_explicit_historical_migration_passes(self):
        plan = self.migrated()
        result = self.validate(plan)
        self.assertTrue(result["valid"], result["errors"])
        compiled = compile_continuous_round_plan(
            plan, source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"])
        gate = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            planning_state=self.values["planning_state"],
            complete_dialogue=self.values["complete_dialogue"])
        self.assertTrue(gate["valid"], gate["errors"])
        self.assertTrue(all(scene["compiled_proposals"] == [] for scene in compiled["scenes"]))

    def test_action_type_may_repeat_with_distinct_arguments(self):
        plan = self.migrated()
        a = deepcopy(plan["scenes"][0]["primary_action"])
        a["action_type"] = "develop_character_or_task_state"
        a["arguments"] = {"target": next(iter(self.entity_refs)), "dimension": "x",
                          "direction": "up", "trigger": "one"}
        plan["scenes"][0]["primary_action"] = a
        plan["scenes"][1]["primary_action"] = deepcopy(a)
        plan["scenes"][1]["primary_action"]["arguments"]["dimension"] = "y"
        self.assertTrue(self.validate(plan)["valid"])

    def test_action_id_and_bundle_id_are_rejected(self):
        for field in ("action_id", "bundle_id", "candidate_update_hints"):
            plan = self.migrated()
            plan["scenes"][0][field] = "A0001" if field != "candidate_update_hints" else []
            result = self.validate(plan)
            self.assertFalse(result["valid"])
            self.assertTrue(any(field in error for error in result["errors"]))

    def test_retry_uses_registry_guidance_without_old_response(self):
        retry = build_planner_retry_prompt_v11(
            base_prompt="FROZEN BASE", frozen_input_sha256="a" * 64,
            errors=["scene:1:primary_action.arguments:missing:target"])
        self.assertIn("shared narrative action registry", retry)
        self.assertIn("Never output action_id or bundle_id", retry)
        self.assertNotIn(self.old_plan["round_goal"], retry)

    def test_history_probe_tree_is_unchanged(self):
        self.assertEqual(hash_tree(PROBE_ROOT)["tree_sha256"], PROBE_TREE_SHA)
        self.assertEqual(hash_tree(PROBE_2_ROOT)["tree_sha256"], PROBE_2_TREE_SHA)
        self.assertEqual(hash_tree(PROBE_3_ROOT)["tree_sha256"], PROBE_3_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
