from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
    dialogue_first_participant_planner_probe_v11 as probe,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    build_scene_dependency_dag,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    DEFAULT_INPUT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_namespace_repair_v11 import (
    apply_planner_namespace_repairs_v11,
    build_planner_namespace_only_retry_prompt_v11,
    compile_planner_namespace_repair_spec_v11,
    planner_namespace_only_retry_eligible,
    validate_planner_namespace_repair_response_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    compile_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


PROBE_7 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_7"
)
PROBE_7_SHA = "2b1d2aea5cc3d82fbb738c9a444af1cd1193aba702f16634604b48f6ec6bd0c3"
PROBE_9 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_namespace_repair_probe_9"
)
PROBE_9_SHA = "9a05506a115d834748e0e777a87c13086e17cb633fe17db2dcbf138f5015e447"
PROBE_10 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_namespace_repair_probe_10"
)
PROBE_10_SHA = "b410438c30bfa3083ff2154f3a13c6d08904c58dff50f7e32a007022794e80d2"


class PlannerNamespaceRepairV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = probe._build_dialogue_first_probe_input(DEFAULT_INPUT)
        cls.outline = json.loads(
            (PROBE_7 / "planner_response_parsed.json").read_text())
        cls.authoritative_refs = {
            row["entity_ref"] for row in cls.values["role_table"]}
        cls.validation = probe._validate_dialogue_first_outline(
            cls.outline,
            source_anchor_registry=cls.values["source_anchor_registry"],
            placement_choice_registry=cls.values["placement_choice_registry"],
            authoritative_entity_refs=cls.authoritative_refs,
            authoritative_entities=cls.values["planning_state"]["entities"],
            action_availability_view=cls.values["action_availability_view"])
        cls.spec = compile_planner_namespace_repair_spec_v11(
            outline=cls.outline, errors=cls.validation["errors"],
            validation=cls.validation,
            action_contracts=cls.values["shared_narrative_action_contracts"],
            action_availability_view=cls.values["action_availability_view"])

    @classmethod
    def response(cls):
        original = cls.outline["scenes"][6]["primary_action"]["arguments"]
        link_choice = next(row["choice_id"] for row in cls.spec[
            "link_type_choices"] if row["link_type"] == "relationship_followup")
        return {
            "action_repairs": [{
                "action_ref": "S07.primary_action",
                "replacement": {
                    "action_mode": "new",
                    "action_type": "supersede_or_correct_information",
                    "arguments": deepcopy(original),
                },
            }],
            "link_type_repairs": [{
                "scene_ref": "S07",
                "replacement_choice_id": link_choice,
            }],
        }

    def test_probe7_history_and_real_errors_are_stable(self):
        self.assertEqual(hash_tree(PROBE_7)["tree_sha256"], PROBE_7_SHA)
        self.assertEqual(self.validation["errors"], [
            "scene:7:link_type:resolution",
            "action:S07.primary_action:unknown_action_type:"
            "resolve_or_correct_information",
        ])

    def test_mixed_action_and_link_errors_use_namespace_only_retry(self):
        self.assertTrue(planner_namespace_only_retry_eligible(
            outline=self.outline, errors=self.validation["errors"]))
        self.assertEqual(self.spec["target_action_refs"], ["S07.primary_action"])
        self.assertEqual(self.spec["target_link_scene_refs"], ["S07"])
        self.assertIn("relationship_followup", self.spec["allowed_link_types"])
        self.assertNotIn("resolution", self.spec["allowed_link_types"])
        self.assertEqual(len(self.spec["link_type_choices"]),
                         len(self.spec["allowed_link_types"]))
        self.assertEqual(self.spec["link_target_scene_contexts"][0]["scene_ref"],
                         "S07")
        self.assertNotIn("link_type", self.spec["link_target_scene_contexts"][0])

    def test_unrelated_or_parse_errors_are_not_namespace_only(self):
        self.assertFalse(planner_namespace_only_retry_eligible(
            outline=self.outline,
            errors=[*self.validation["errors"], "scene:2:speaker:3:schema"]))
        self.assertFalse(planner_namespace_only_retry_eligible(
            outline=self.outline, errors=self.validation["errors"],
            parse_errors=["json_parse_invalid"]))

    def test_prompt_separates_three_value_namespaces_and_freezes_outline(self):
        prompt = build_planner_namespace_only_retry_prompt_v11(self.spec)
        self.assertIn("LINK TYPE CHOICES", prompt)
        self.assertIn("AVAILABLE ACTION CONTRACTS", prompt)
        self.assertIn("FROZEN LINK TARGET CONTEXT", prompt)
        self.assertIn("progression.stage", prompt)
        self.assertIn("Never synthesize or rename action_type", prompt)
        self.assertNotIn('"replacement": {}', prompt)
        self.assertIn(
            "<COMPLETE_NEW_OR_CONTINUATION_ACTION_OBJECT_REQUIRED>", prompt)
        self.assertIn("COMPLETE REPLACEMENT ENVELOPE", prompt)
        self.assertIn("For ordinary actions, all contract argument fields", prompt)
        self.assertIn("PROGRAM CONNECTION CHOICES", prompt)
        self.assertIn("Display labels and shortened names", prompt)
        self.assertNotIn(self.outline["round_goal"], prompt)
        self.assertTrue(prompt.rstrip().endswith("}"))

    def test_response_contract_rejects_unknown_link_without_guessing(self):
        response = self.response()
        response["link_type_repairs"][0]["replacement_choice_id"] = "resolution"
        result = validate_planner_namespace_repair_response_v11(
            response, spec=self.spec)
        self.assertFalse(result["valid"])
        self.assertIn(
            "planner_namespace_repair:link_item:1:replacement_choice_id_not_allowed",
            result["errors"])
        self.assertEqual(
            self.outline["scenes"][6]["link_type"], "resolution")

    def test_exact_target_sets_and_top_level_are_required(self):
        missing = self.response()
        missing["link_type_repairs"] = []
        self.assertFalse(validate_planner_namespace_repair_response_v11(
            missing, spec=self.spec)["valid"])
        extra = self.response()
        extra["scenes"] = []
        self.assertFalse(validate_planner_namespace_repair_response_v11(
            extra, spec=self.spec)["valid"])

    def test_shadow_repair_freezes_non_targets_and_passes_full_chain(self):
        original = deepcopy(self.outline)
        result = apply_planner_namespace_repairs_v11(
            frozen_outline=self.outline, response=self.response(), spec=self.spec)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(all(result["freeze_audit"].values()))
        repaired = result["reconstructed_outline"]
        self.assertEqual(repaired["scenes"][6]["link_type"],
                         "relationship_followup")
        self.assertEqual(repaired["scenes"][6]["primary_action"]["action_type"],
                         "supersede_or_correct_information")
        for index in range(6):
            self.assertEqual(repaired["scenes"][index], original["scenes"][index])
        repaired_validation = probe._validate_dialogue_first_outline(
            repaired,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            authoritative_entity_refs=self.authoritative_refs,
            authoritative_entities=self.values["planning_state"]["entities"],
            action_availability_view=self.values["action_availability_view"])
        self.assertTrue(repaired_validation["valid"], repaired_validation["errors"])
        compiled = compile_planner_outline_v11(
            repaired,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            planning_state=self.values["planning_state"],
            action_availability_view=self.values["action_availability_view"])
        self.assertEqual(len(build_scene_dependency_dag(compiled)["nodes"]), 7)
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            planning_state=self.values["planning_state"],
            complete_dialogue=self.values["complete_dialogue"])
        self.assertTrue(structural["valid"], structural["errors"])
        self.assertEqual(self.outline, original)
        self.assertEqual(hash_tree(PROBE_7)["tree_sha256"], PROBE_7_SHA)

    def test_generic_unknown_tokens_are_classified_without_alias_table(self):
        generic = deepcopy(self.outline)
        generic["scenes"][6]["link_type"] = "invented_link"
        generic["scenes"][6]["primary_action"]["action_type"] = "invented_action"
        errors = [
            "scene:7:link_type:invented_link",
            "action:S07.primary_action:unknown_action_type:invented_action",
        ]
        self.assertTrue(planner_namespace_only_retry_eligible(
            outline=generic, errors=errors))
        self.assertEqual(generic["scenes"][6]["link_type"], "invented_link")
        self.assertEqual(generic["scenes"][6]["primary_action"]["action_type"],
                         "invented_action")

    def test_probe9_empty_patch_is_rejected_before_outline_reconstruction(self):
        self.assertEqual(hash_tree(PROBE_9)["tree_sha256"], PROBE_9_SHA)
        outline = json.loads((PROBE_9 / "frozen_source_outline.json").read_text())
        spec = json.loads((PROBE_9 / "namespace_repair_spec.json").read_text())
        response = json.loads((PROBE_9 / "planner_response_parsed.json").read_text())
        validation = validate_planner_namespace_repair_response_v11(
            response, spec=spec)
        self.assertFalse(validation["valid"])
        self.assertEqual(
            sum(error.endswith(":replacement_empty")
                for error in validation["errors"]),
            len(spec["target_action_refs"]),
        )
        result = apply_planner_namespace_repairs_v11(
            frozen_outline=outline, response=response, spec=spec)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["reconstructed_outline"])
        self.assertEqual(hash_tree(PROBE_9)["tree_sha256"], PROBE_9_SHA)

    def test_probe9_complete_patch_fixture_passes_full_local_chain(self):
        outline = json.loads((PROBE_9 / "frozen_source_outline.json").read_text())
        spec = json.loads((PROBE_9 / "namespace_repair_spec.json").read_text())
        display_to_ref = {
            row["display_name"]: row["entity_ref"]
            for row in self.values["planning_state"]["entities"]
        }
        display_to_ref["Friend C"] = "local_1"
        repairs = []
        for action_ref in spec["target_action_refs"]:
            scene_number = int(action_ref[1:3])
            slot = action_ref.split(".", 1)[1]
            replacement = deepcopy(outline["scenes"][scene_number - 1][slot])
            arguments = replacement.get("arguments") or {}
            for field in ("target", "sender", "receiver"):
                if arguments.get(field) in display_to_ref:
                    arguments[field] = display_to_ref[arguments[field]]
            repairs.append({"action_ref": action_ref, "replacement": replacement})
        response = {
            "action_repairs": repairs,
            "link_type_repairs": [{
                "scene_ref": "S07", "replacement": "consequence_followup",
            }],
        }
        result = apply_planner_namespace_repairs_v11(
            frozen_outline=outline, response=response, spec=spec)
        self.assertTrue(result["valid"], result["errors"])
        repaired = result["reconstructed_outline"]
        validation = probe._validate_dialogue_first_outline(
            repaired,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            authoritative_entity_refs=self.authoritative_refs,
            authoritative_entities=self.values["planning_state"]["entities"],
            action_availability_view=self.values["action_availability_view"])
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_outline_v11(
            repaired,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            planning_state=self.values["planning_state"],
            action_availability_view=self.values["action_availability_view"])
        self.assertEqual(len(build_scene_dependency_dag(compiled)["nodes"]), 7)
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=self.values["source_anchor_registry"],
            placement_choice_registry=self.values["placement_choice_registry"],
            planning_state=self.values["planning_state"],
            complete_dialogue=self.values["complete_dialogue"])
        self.assertTrue(structural["valid"], structural["errors"])
        self.assertEqual(hash_tree(PROBE_9)["tree_sha256"], PROBE_9_SHA)

    def test_probe10_flat_action_payloads_remain_strictly_rejected(self):
        self.assertEqual(hash_tree(PROBE_10)["tree_sha256"], PROBE_10_SHA)
        spec = json.loads((PROBE_10 / "namespace_repair_spec.json").read_text())
        response = json.loads((PROBE_10 / "planner_response_parsed.json").read_text())
        validation = validate_planner_namespace_repair_response_v11(
            response, spec=spec)
        self.assertFalse(validation["valid"])
        self.assertEqual(
            validation["errors"],
            [f"planner_action_repair:item:{index}:replacement:action_mode"
             for index in range(1, 9)],
        )
        for row in response["action_repairs"]:
            replacement = row["replacement"]
            self.assertNotIn("action_mode", replacement)
            self.assertNotIn("arguments", replacement)
        self.assertEqual(hash_tree(PROBE_10)["tree_sha256"], PROBE_10_SHA)


if __name__ == "__main__":
    unittest.main()
