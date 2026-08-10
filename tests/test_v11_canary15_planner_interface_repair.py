from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.action_availability_v11 import (
    compact_action_availability_for_planner,
    compile_action_availability_view,
    validate_action_availability_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_planner_prompt_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    DEFAULT_INPUT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_only_probe_v11 import (
    build_planner_probe_input,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    compile_planner_outline_v11,
    validate_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_speaker_repair_v11 import (
    apply_planner_speaker_repairs_v11,
    build_planner_speaker_only_retry_prompt_v11,
    compile_planner_speaker_repair_spec_v11,
    planner_speaker_only_retry_eligible,
    validate_planner_speaker_repair_response_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


CANARY15 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_15"
)
CANARY15_SHA = "fca44d7b6e822ba9d6e8cf547d604d86b34008c47643f5f93212e536f31fc4b0"


class Canary15PlannerInterfaceRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = build_planner_probe_input(DEFAULT_INPUT)
        cls.availability = compile_action_availability_view(
            planning_state=cls.values["planning_state"])
        cls.roles = cls.values["role_table"]
        cls.authoritative_refs = {row["entity_ref"] for row in cls.roles}
        record = json.loads((CANARY15 / (
            "attempts/planner_v11_attempt_02.json")).read_text())
        cls.attempt2 = record["parsed_response"]
        cls.attempt2_errors = json.loads((CANARY15 / (
            "planner/attempt_02_validation.json")).read_text())["errors"]

    def speaker_fixture(self, spec):
        refs = sorted(self.authoritative_refs)
        return {"speaker_repairs": [{
            "scene_ref": scene_ref,
            "speaker_participants": [{
                "participant_ref": refs[0], "scene_role": "primary",
                "intents": ["explain the new development"],
            }, {
                "participant_ref": refs[1], "scene_role": "supporting",
                "intents": ["question its consequence"],
            }],
        } for scene_ref in spec["target_scene_refs"]]}

    def current_slot_shadow(self, outline):
        result = deepcopy(outline)
        person_index = event_index = 0
        for scene in result["scenes"]:
            for role in ("primary_action", "secondary_action"):
                action = scene.get(role)
                if not isinstance(action, dict) or action.get(
                        "action_mode") != "new":
                    continue
                arguments = action.get("arguments") or {}
                if action.get("action_type") == "introduce_connected_person":
                    arguments.pop("temporary_entity_key", None)
                    arguments.pop("scene_presence", None)
                    arguments["participant_alias"] = f"shadow_person_{person_index + 1}"
                    person_index += 1
                elif action.get("action_type") == "introduce_connected_event":
                    arguments.pop("temporary_event_key", None)
                    arguments["event_alias"] = f"shadow_event_{event_index + 1}"
                    # Explicit shadow migration of the historical word
                    # "recently"; live validation never infers this enum from text.
                    arguments["narrative_time_relation"] = "past"
                    event_index += 1
        return result

    def test_01_planner_availability_exposes_no_creation_slots(self):
        self.assertTrue(validate_action_availability_view(
            self.availability)["valid"])
        self.assertNotIn("round_local_creation_slots", self.availability)

    def test_02_planner_projection_omits_unavailable_actions(self):
        projection = compact_action_availability_for_planner(self.availability)
        self.assertNotIn("advance_shared_task", projection["available_actions"])
        self.assertNotIn("actions", projection)
        self.assertTrue(projection["unavailable_actions_omitted"])

    def test_03_live_prompt_exposes_aliases_but_no_tmp_or_unavailable_task(self):
        prompt, _ = build_planner_prompt_v11(
            root_id=self.values["planning_state"]["root_id"],
            planning_state=self.values["planning_state"],
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            action_contracts=self.values["shared_narrative_action_contracts"],
            action_availability_view=self.availability,
        )
        self.assertIn('"participant_alias"', prompt)
        self.assertIn('"event_alias"', prompt)
        self.assertNotIn(":TMP:", prompt)
        self.assertNotIn('"available":false', prompt)
        self.assertNotIn('"action_type":"advance_shared_task"', prompt)
        self.assertNotIn("TMP", prompt)
        self.assertIn("post-review compiler owns all internal allocation identifiers", prompt)

    def test_04_canary15_attempt2_is_speaker_only_retry_eligible(self):
        self.assertTrue(planner_speaker_only_retry_eligible(
            outline=self.attempt2, errors=self.attempt2_errors))
        self.assertEqual(len(self.attempt2_errors), 3)

    def test_05_speaker_retry_prompt_freezes_outline_and_hides_invalid_tmp_ref(self):
        spec = compile_planner_speaker_repair_spec_v11(
            outline=self.attempt2, errors=self.attempt2_errors,
            authoritative_role_table=self.roles)
        prompt = build_planner_speaker_only_retry_prompt_v11(spec)
        self.assertEqual(spec["target_scene_refs"], ["S03", "S04", "S05"])
        self.assertNotIn("TMP:E:999999999999", prompt)
        self.assertNotIn(self.attempt2["round_goal"], prompt)
        self.assertTrue(prompt.rstrip().endswith("}"))

    def test_06_speaker_retry_accepts_only_exact_authoritative_refs(self):
        spec = compile_planner_speaker_repair_spec_v11(
            outline=self.attempt2, errors=self.attempt2_errors,
            authoritative_role_table=self.roles)
        response = self.speaker_fixture(spec)
        self.assertTrue(validate_planner_speaker_repair_response_v11(
            response, spec=spec)["valid"])
        broken = deepcopy(response)
        broken["speaker_repairs"][0]["speaker_participants"][0][
            "participant_ref"] = "cc-train-episode-61555:TMP:E:P01"
        self.assertFalse(validate_planner_speaker_repair_response_v11(
            broken, spec=spec)["valid"])

    def test_07_speaker_reconstruction_is_field_frozen(self):
        spec = compile_planner_speaker_repair_spec_v11(
            outline=self.attempt2, errors=self.attempt2_errors,
            authoritative_role_table=self.roles)
        result = apply_planner_speaker_repairs_v11(
            frozen_outline=self.attempt2,
            response=self.speaker_fixture(spec), spec=spec)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(all(result["freeze_audit"].values()))
        self.assertEqual(result["reconstructed_outline"]["round_goal"],
                         self.attempt2["round_goal"])

    def test_08_shadow_repair_passes_current_schema_compile_dag_and_gate(self):
        outline = self.current_slot_shadow(self.attempt2)
        spec = compile_planner_speaker_repair_spec_v11(
            outline=outline, errors=self.attempt2_errors,
            authoritative_role_table=self.roles)
        reconstructed = apply_planner_speaker_repairs_v11(
            frozen_outline=outline, response=self.speaker_fixture(spec),
            spec=spec)["reconstructed_outline"]
        validation = validate_planner_outline_v11(
            reconstructed,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            authoritative_entity_refs=self.authoritative_refs,
            action_availability_view=self.availability)
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_outline_v11(
            reconstructed,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            planning_state=self.values["planning_state"],
            action_availability_view=self.availability)
        gate = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            planning_state=self.values["planning_state"],
            complete_dialogue=self.values["complete_dialogue"])
        self.assertTrue(gate["valid"], gate["errors"])

    def test_09_reusing_a_narrative_alias_is_rejected(self):
        outline = self.current_slot_shadow(self.attempt2)
        source = next(
            action for scene in outline["scenes"]
            for action in (scene.get("primary_action"), scene.get("secondary_action"))
            if isinstance(action, dict) and action.get("action_type") ==
            "introduce_connected_person")
        outline["scenes"][-1]["secondary_action"] = {
            "action_mode": "new", "action_type": "introduce_connected_person",
            "arguments": {
                **deepcopy(source["arguments"]),
                "display_name": "A distinct second person",
                "connection_description": "A separate explicit connection to A.",
            },
        }
        validation = validate_planner_outline_v11(
            outline,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            authoritative_entity_refs=self.authoritative_refs,
            action_availability_view=self.availability)
        self.assertTrue(any("narrative_alias_reused" in error
                            for error in validation["errors"]))

    def test_10_canary15_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY15)["tree_sha256"], CANARY15_SHA)


if __name__ == "__main__":
    unittest.main()
