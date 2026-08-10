from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    compile_continuous_round_plan, validate_continuous_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_planner_retry_prompt_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    DEFAULT_INPUT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    migrate_legacy_action_hints_to_narrative_actions,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_only_probe_v11 import (
    build_planner_probe_input,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


PROBE_5 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_5")
PROBE_5_SHA = "14d6b65d34b48986fcd8a55a0ec32b2311ad9a87539c8b7197607f63619ee4d3"


class PlannerProbe5ShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = build_planner_probe_input(DEFAULT_INPUT)
        cls.plan = json.loads((PROBE_5 / "planner_response_parsed.json").read_text())
        cls.entities = {row["entity_ref"] for row in cls.values["role_table"]}

    def _validate(self, plan):
        return validate_continuous_round_plan(
            plan, source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            action_menu=self.values["action_menu"],
            authoritative_entity_refs=self.entities)

    @staticmethod
    def _add_shadow_outcomes(plan):
        value = deepcopy(plan)
        for index, scene in enumerate(value["scenes"], 1):
            scene["desired_outcome"] = (
                f"Shadow-only observable outcome for scene {index}.")
        return value

    def _migrate_actions(self, plan):
        return migrate_legacy_action_hints_to_narrative_actions(
            plan, action_menu=self.values["action_menu"])

    def test_probe5_was_not_truncated(self):
        attempt = json.loads((PROBE_5 /
            "attempts/planner_v11_probe_5_request_01.json").read_text())
        self.assertEqual(attempt["response_metadata"]["finish_reason"], "stop")
        self.assertEqual(attempt["parse_errors"], [])
        self.assertEqual(len(self.plan["scenes"]), 5)

    def test_validator_reports_each_missing_desired_outcome(self):
        result = self._validate(self.plan)
        self.assertFalse(result["valid"])
        for index in range(1, 6):
            self.assertIn(
                f"scene:{index}:missing_required_field:desired_outcome",
                result["errors"])

    def test_missing_field_no_longer_masks_independent_selection_errors(self):
        result = self._validate(self.plan)
        self.assertIn("scene:5:merge_target", result["errors"])
        self.assertTrue(any("candidate_update_hints" in error for error in result["errors"]))

    def test_adding_outcomes_alone_does_not_silently_pass(self):
        result = self._validate(self._add_shadow_outcomes(self.plan))
        self.assertFalse(result["valid"])
        self.assertIn("scene:5:merge_target", result["errors"])
        self.assertTrue(any("candidate_update_hints" in error for error in result["errors"]))

    def test_explicit_shadow_fixture_passes_full_local_chain(self):
        fixture = self._migrate_actions(self._add_shadow_outcomes(self.plan))
        fixture["scenes"][4]["merge_target_scene_alias"] = None
        validation = self._validate(fixture)
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_continuous_round_plan(
            fixture,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            round_index=1)
        gate = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            complete_dialogue=self.values["complete_dialogue"],
            planning_state=self.values["planning_state"])
        self.assertTrue(gate["valid"], gate["errors"])
        self.assertGreaterEqual(gate["longest_hard_dependency_chain"], 3)

    def test_live_prompt_repeats_desired_outcome_requirement(self):
        prompt = self.values["prompt"]
        self.assertGreaterEqual(prompt.count("desired_outcome"), 4)
        self.assertIn("Every scene, including the final scene", prompt)
        self.assertIn("Return exactly four top-level keys", prompt)
        self.assertNotIn("Every mainline scene uses", prompt)
        self.assertIn("same action_type may appear", prompt)

    def test_retry_guidance_is_compact_and_does_not_include_old_response(self):
        retry = build_planner_retry_prompt_v11(
            base_prompt="FROZEN BASE", frozen_input_sha256="a" * 64,
            errors=[
                "scene:1:missing_required_field:desired_outcome",
                "scene:5:merge_target",
                "scene:5:primary_action.arguments:missing:target",
            ])
        self.assertIn("Every scene object must include desired_outcome", retry)
        self.assertIn("Return exactly four top-level keys", retry)
        self.assertNotIn("mainline scene uses merge_target_scene_alias=null", retry)
        self.assertIn("Never output action_id or bundle_id", retry)
        self.assertNotIn(self.plan["round_goal"], retry)

    def test_shadow_does_not_modify_probe5(self):
        self.assertEqual(hash_tree(PROBE_5)["tree_sha256"], PROBE_5_SHA)


if __name__ == "__main__":
    unittest.main()
