import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
    dialogue_first_participant_planner_probe_v11 as probe,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    DEFAULT_INPUT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_namespace_repair_v11 import (
    apply_planner_namespace_repairs_v11,
    build_planner_namespace_only_retry_prompt_v11,
    compile_planner_namespace_repair_spec_v11,
    validate_planner_namespace_repair_response_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_25")
CANARY_SHA = "613f422ff2455157d380f89844f9b726611f985d51c4aef52b7ec483228d5038"


class Canary25LinkChoiceRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = probe._build_dialogue_first_probe_input(DEFAULT_INPUT)
        cls.outline = json.loads((CANARY / "attempts/planner_v11_attempt_01.json")
                                 .read_text())["parsed_response"]
        cls.errors = json.loads((CANARY / "planner/attempt_01_validation.json")
                                .read_text())["errors"]
        cls.spec = compile_planner_namespace_repair_spec_v11(
            outline=cls.outline, errors=cls.errors, validation={},
            action_contracts=cls.values["shared_narrative_action_contracts"],
            action_availability_view=cls.values["action_availability_view"])

    def response(self, link_type="relationship_followup"):
        choice = next(row["choice_id"] for row in self.spec[
            "link_type_choices"] if row["link_type"] == link_type)
        return {"action_repairs": [], "link_type_repairs": [{
            "scene_ref": "S06", "replacement_choice_id": choice}]}

    def test_real_failure_is_one_link_namespace_error(self):
        self.assertEqual(self.errors, ["scene:6:link_type:resolution"])
        self.assertEqual(self.spec["target_action_refs"], [])
        self.assertEqual(self.spec["target_link_scene_refs"], ["S06"])

    def test_link_only_prompt_uses_choice_ids_without_action_envelope(self):
        prompt = build_planner_namespace_only_retry_prompt_v11(self.spec)
        self.assertIn("LINK TYPE CHOICES", prompt)
        self.assertIn('"replacement_choice_id"', prompt)
        self.assertNotIn("COMPLETE REPLACEMENT ENVELOPE", prompt)
        self.assertNotIn("AVAILABLE ACTION CONTRACTS", prompt)
        self.assertNotIn("progression.stage", prompt)

    def test_historical_invalid_string_patch_stays_rejected(self):
        historical = json.loads((CANARY / "attempts/planner_v11_attempt_02.json")
                                .read_text())["parsed_response"]
        validation = validate_planner_namespace_repair_response_v11(
            historical, spec=self.spec)
        self.assertFalse(validation["valid"])
        self.assertIn("planner_namespace_repair:link_item:1:exact_keys",
                      validation["errors"])

    def test_choice_is_expanded_without_semantic_guessing_and_full_plan_passes(self):
        response = self.response()
        validation = validate_planner_namespace_repair_response_v11(
            response, spec=self.spec)
        self.assertTrue(validation["valid"], validation["errors"])
        applied = apply_planner_namespace_repairs_v11(
            frozen_outline=self.outline, response=response, spec=self.spec)
        self.assertTrue(applied["valid"], applied["errors"])
        repaired = applied["reconstructed_outline"]
        self.assertEqual(repaired["scenes"][5]["link_type"],
                         "relationship_followup")
        authoritative_refs = {row["entity_ref"] for row in self.values["role_table"]}
        full = probe._validate_dialogue_first_outline(
            repaired,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            authoritative_entity_refs=authoritative_refs,
            authoritative_entities=self.values["planning_state"]["entities"],
            action_availability_view=self.values["action_availability_view"])
        self.assertTrue(full["valid"], full["errors"])

    def test_history_unchanged(self):
        self.assertEqual(hash_tree(CANARY)["tree_sha256"], CANARY_SHA)


if __name__ == "__main__":
    unittest.main()
