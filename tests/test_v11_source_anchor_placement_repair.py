from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.context_compiler_v2 import (
    compile_scene_context_v2,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    PLANNER_OUTPUT_TOKEN_BUDGET_V11,
    audit_planner_output_size_v11,
    build_scene_dependency_dag,
    compile_continuous_round_plan,
    validate_continuous_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    DEFAULT_INPUT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.offline_preflight_v11 import (
    build_round_blueprint_fixture,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.placement_choice_registry import (
    compact_source_anchor_view,
    resolve_placement_choice,
    validate_placement_choice_registry,
    validate_source_anchor_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_only_probe_v11 import (
    build_planner_probe_input,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_shadow_audit_v11 import (
    analyze_probe4_source_placement_shadow,
    extract_complete_scene_objects_from_truncated_response,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_transport_v11 import (
    PLANNER_MAX_OUTPUT_TOKENS,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


PROBE_4_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_4")
PROBE_4_SHA = "5ddb7d818ff2736803a7079cbf0a640e7b0a5113b71dd21ad66e0dbddb649f5b"


class SourceAnchorPlacementRepairV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = build_planner_probe_input(DEFAULT_INPUT)
        cls.source_anchors = cls.values["source_anchor_registry"]
        cls.choices = cls.values["placement_choice_registry"]
        cls.entities = {row["entity_ref"] for row in cls.values["role_table"]}
        cls.source_pa4 = next(
            row for row in cls.source_anchors["source_anchors"]
            if row["source_anchor_id"] == "PA0004")
        cls.selection = {
            "action_id": "A0008",
            "proposed_content": {
                "state_category": "emotional_state",
                "state_dimension": "shared_responsibility_reflection",
                "initial_value": "newly_discussed",
            },
        }
        cls.raw_probe4 = (PROBE_4_ROOT / "raw/backend_responses/"
                          "planner_v11_probe_4_request_01.txt").read_text(
                              encoding="utf-8")

    def _plan(self, scene_count=7, choice="PC_TAIL_AFTER"):
        return build_round_blueprint_fixture(
            source_anchor_id="PA0004", placement_choice_id=choice,
            entity_refs=self.source_pa4["linked_entity_refs"],
            single_action_selection=self.selection, scene_count=scene_count)

    def _validate(self, plan):
        return validate_continuous_round_plan(
            plan, source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.choices,
            action_menu=self.values["action_menu"],
            authoritative_entity_refs=self.entities)

    def _full_chain(self, scene_count):
        plan = self._plan(scene_count)
        validation = self._validate(plan)
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_continuous_round_plan(
            plan, action_menu=self.values["action_menu"],
            source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.choices, round_index=1)
        dag = build_scene_dependency_dag(compiled)
        gate = validate_round_blueprint_structure(
            compiled_plan=compiled, source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.choices,
            complete_dialogue=self.values["complete_dialogue"],
            planning_state=self.values["planning_state"])
        self.assertEqual(len(dag["nodes"]), scene_count)
        self.assertTrue(gate["valid"], gate["errors"])
        self.assertGreaterEqual(gate["longest_hard_dependency_chain"], 3)
        size = audit_planner_output_size_v11(plan)
        self.assertTrue(size["valid"], size)
        return plan, compiled, gate, size

    def test_source_anchor_and_placement_choice_registries_are_independent(self):
        self.assertTrue(validate_source_anchor_registry(self.source_anchors)["valid"])
        self.assertTrue(validate_placement_choice_registry(
            self.choices, complete_dialogue=self.values["complete_dialogue"])["valid"])
        for row in self.source_anchors["source_anchors"]:
            self.assertFalse({"placement", "available_placements", "boundary"} & set(row))

    def test_middle_source_anchor_plus_tail_after_is_valid(self):
        result = self._validate(self._plan())
        self.assertTrue(result["valid"], result["errors"])
        _, compiled, _, _ = self._full_chain(7)
        self.assertEqual(compiled["scenes"][0]["source_anchor_id"], "PA0004")
        self.assertEqual(compiled["scenes"][0]["placement_choice_id"], "PC_TAIL_AFTER")
        self.assertEqual(compiled["scenes"][0]["placement"], "after")

    def test_unknown_source_anchor_is_rejected(self):
        plan = self._plan()
        plan["scenes"][0]["source_anchor_id"] = "PA9999"
        self.assertIn("scene:1:source_anchor_unknown:PA9999",
                      self._validate(plan)["errors"])

    def test_unknown_placement_choice_is_rejected(self):
        plan = self._plan()
        plan["scenes"][0]["placement_choice_id"] = "PC_FAKE"
        self.assertIn("scene:1:placement_choice_unknown:PC_FAKE",
                      self._validate(plan)["errors"])

    def test_planner_cannot_override_placement(self):
        plan = self._plan()
        plan["scenes"][0]["placement"] = "between"
        result = self._validate(plan)
        self.assertFalse(result["valid"])
        self.assertTrue(any("continuous_plan_program_owned" in value and
                            ".placement" in value for value in result["errors"]))

    def test_planner_cannot_override_turn_boundary(self):
        for field in ("after_turn_ref", "before_turn_ref"):
            plan = self._plan()
            plan["scenes"][0][field] = "invented"
            result = self._validate(plan)
            self.assertFalse(result["valid"])
            self.assertTrue(any(field in value for value in result["errors"]))

    def test_tail_after_choice_expands_program_owned_boundary(self):
        choice = resolve_placement_choice(self.choices, "PC_TAIL_AFTER")
        last_ref = self.values["complete_dialogue"]["turns"][-1]["turn_ref"]
        self.assertEqual(choice["placement"], "after")
        self.assertEqual(choice["after_turn_ref"], last_ref)
        self.assertIsNone(choice["before_turn_ref"])
        self.assertEqual(choice["preceding_known_turn_refs"][-1], last_ref)
        self.assertEqual(choice["following_constraint_turn_refs"], [])

    def test_gap_between_choice_expands_exact_adjacent_boundary(self):
        choice = resolve_placement_choice(self.choices, "PC_GAP_0001")
        refs = [row["turn_ref"] for row in self.values["complete_dialogue"]["turns"]]
        self.assertEqual(choice["placement"], "between")
        self.assertEqual(refs.index(choice["before_turn_ref"]),
                         refs.index(choice["after_turn_ref"]) + 1)
        self.assertEqual(choice["preceding_known_turn_refs"][-1],
                         choice["after_turn_ref"])
        self.assertEqual(choice["following_constraint_turn_refs"][0],
                         choice["before_turn_ref"])

    def test_parallel_choice_freezes_future_knowledge_boundary(self):
        choice = resolve_placement_choice(self.choices, "PC_PARALLEL_0001")
        self.assertEqual(choice["placement"], "parallel")
        self.assertFalse(choice["knowledge_boundary"][
            "following_context_is_character_knowledge"])
        self.assertEqual(choice["knowledge_boundary"]["following_context_role"],
                         "following_context_constraint_only")

    def test_future_context_is_constraint_not_character_knowledge(self):
        plan = self._plan(choice="PC_PARALLEL_0001")
        self.assertTrue(self._validate(plan)["valid"])
        compiled = compile_continuous_round_plan(
            plan, action_menu=self.values["action_menu"],
            source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.choices, round_index=1)
        context = compile_scene_context_v2(
            scene=compiled["scenes"][0],
            complete_dialogue=self.values["complete_dialogue"],
            source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.choices,
            planning_state=self.values["planning_state"], same_round_scenes={})
        self.assertTrue(context["future_knowledge_is_constraint_only"])
        self.assertTrue(context["following_context_constraint_refs"])
        roles = {row["turn_ref"]: row["knowledge_role"]
                 for row in context["p0_required_dialogue"]}
        for ref in context["following_context_constraint_refs"]:
            self.assertEqual(roles[ref], "following_context_constraint_only")

    def test_old_anchor_plus_placement_live_shape_is_rejected(self):
        plan = self._plan()
        scene = plan["scenes"][0]
        scene["historical_anchor_id"] = scene.pop("source_anchor_id")
        scene["placement"] = "after"
        scene.pop("placement_choice_id")
        result = self._validate(plan)
        self.assertFalse(result["valid"])
        self.assertIn("scene:1:schema", result["errors"])

    def test_planner_max_output_tokens_is_10240(self):
        self.assertEqual(PLANNER_MAX_OUTPUT_TOKENS, 10_240)
        self.assertEqual(PLANNER_OUTPUT_TOKEN_BUDGET_V11, 10_240)

    def test_complete_five_scene_fixture_passes(self):
        self._full_chain(5)

    def test_complete_six_scene_fixture_passes(self):
        self._full_chain(6)

    def test_complete_seven_scene_fixture_passes_with_margin(self):
        _, _, _, size = self._full_chain(7)
        self.assertGreaterEqual(size["estimated_safety_margin_tokens"], 1024)

    def test_compact_source_view_does_not_repeat_all_turn_refs(self):
        view = compact_source_anchor_view(self.source_anchors)
        for row in view["source_anchors"]:
            self.assertNotIn("source_turn_refs", row)
            self.assertIn("source_turn_ref_span", row)

    def test_probe4_extracts_only_six_complete_scenes(self):
        scenes = extract_complete_scene_objects_from_truncated_response(
            self.raw_probe4)
        self.assertEqual(len(scenes), 6)
        self.assertEqual([row["historical_anchor_id"] for row in scenes[2:6]],
                         ["PA0004", "PA0003", "PA0002", "PA0001"])

    def test_probe4_shadow_maps_middle_sources_to_tail_choice_without_success_claim(self):
        audit = analyze_probe4_source_placement_shadow(
            self.raw_probe4, placement_choice_registry=self.choices)
        self.assertTrue(audit["historical_response_remains_parse_invalid"])
        self.assertTrue(audit["shadow_does_not_claim_planner_success"])
        rows = audit["migrations"][2:6]
        self.assertEqual([row["new_contract_example"]["source_anchor_id"]
                          for row in rows],
                         ["PA0004", "PA0003", "PA0002", "PA0001"])
        self.assertTrue(all(row["new_contract_example"]["placement_choice_id"] ==
                            "PC_TAIL_AFTER" for row in rows))

    def test_probe4_history_tree_is_unchanged(self):
        self.assertEqual(hash_tree(PROBE_4_ROOT)["tree_sha256"], PROBE_4_SHA)


if __name__ == "__main__":
    unittest.main()
