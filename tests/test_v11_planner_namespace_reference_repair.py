from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from tests.test_v11_shared_narrative_action_menu import migrated_fixture
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.action_availability_v11 import (
    compile_action_availability_view, validate_action_against_availability,
    validate_action_availability_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    build_scene_dependency_dag, compile_continuous_round_plan,
    validate_continuous_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_planner_prompt_v11, planner_retry_guidance_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_continuation import (
    compile_action_slot_namespace, compile_allowed_continuation_slot_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    CANONICAL_ACTION_SLOT_PATTERN, NARRATIVE_ACTION_CONTRACTS,
    validate_action_instance,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


PROBE_8 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_8")
PROBE_8_SHA = "b101cdb4c82a927ccf8fa4f6cc7404b73f8abedc4c56f85c91d6201948d6d750"


def probe8_inputs():
    return {
        name: json.loads((PROBE_8 / filename).read_text())
        for name, filename in {
            "plan": "planner_response_parsed.json",
            "source": "source_anchor_registry.json",
            "placements": "placement_choice_registry.json",
            "planning": "planning_state_view.json",
            "complete": "complete_dialogue_view.json",
        }.items()
    }


def validate_probe8(plan):
    values = probe8_inputs()
    refs = {row["entity_ref"] for row in values["planning"]["entities"]}
    availability = compile_action_availability_view(
        planning_state=values["planning"])
    result = validate_continuous_round_plan(
        plan, source_anchor_registry=values["source"],
        placement_choice_registry=values["placements"],
        authoritative_entity_refs=refs,
        action_availability_view=availability)
    return values, refs, availability, result


def repaired_probe8_plan():
    values = probe8_inputs()
    plan = deepcopy(values["plan"])
    entities = sorted(row["entity_ref"] for row in values["planning"]["entities"])
    plan["scenes"][1]["primary_action"]["continuation_of"] = "S01.secondary_action"
    plan["scenes"][3]["primary_action"] = {
        "action_mode": "new", "action_type": "reveal_or_transfer_information",
        "arguments": {"sender": entities[0], "receiver": entities[1],
            "information": "A's past financial instability and its effect on current behavior",
            "transfer_mode": "direct conversation"}}
    plan["scenes"][4]["primary_action"] = {
        "action_mode": "new", "action_type": "develop_character_or_task_state",
        "arguments": {"target": entities[0], "dimension": "financial_planning_readiness",
            "direction": "increase",
            "trigger": "practical collaboration after the job-risk notice"}}
    plan["scenes"][5]["primary_action"] = {
        "action_mode": "new", "action_type": "develop_character_or_task_state",
        "arguments": {"target": entities[0], "dimension": "shared_values_alignment",
            "direction": "increase",
            "trigger": "a concrete discussion of responsibility and support"}}
    plan["scenes"][6]["primary_action"] = {
        "action_mode": "new", "action_type": "develop_character_or_task_state",
        "arguments": {"target": entities[0], "dimension": "emotional_processing",
            "direction": "increase",
            "trigger": "reflection on family tension and shared values"}}
    return plan


class PlannerNamespaceReferenceRepairTests(unittest.TestCase):
    def test_01_canonical_action_slot_regex(self):
        self.assertIsNotNone(CANONICAL_ACTION_SLOT_PATTERN.fullmatch("S01.primary_action"))
        for value in ("s1.primary_action", "S1.primary_action", "S08.primary_action",
                      "R01:S01.primary_action", "scene_1.primary_action"):
            self.assertIsNone(CANONICAL_ACTION_SLOT_PATTERN.fullmatch(value), value)

    def test_02_lowercase_scene_alias_slot_is_rejected(self):
        value = {"action_mode": "continuation", "continuation_of": "s1.secondary_action",
                 "progression": {"stage": "development", "new_step": "a concrete step"}}
        result = validate_action_instance(value)
        self.assertTrue(any("continuation_slot_format" in row for row in result["errors"]))

    def test_03_canonical_continuation_slot_is_schema_valid(self):
        value = {"action_mode": "continuation", "continuation_of": "S01.secondary_action",
                 "progression": {"stage": "development", "new_step": "a concrete step"}}
        self.assertTrue(validate_action_instance(value)["valid"])

    def test_04_allowed_slots_are_dependency_ancestors_only(self):
        action = {"action_mode": "new", "action_type": "develop_character_or_task_state",
                  "arguments": {"target": "p", "dimension": "d", "direction": "up",
                                "trigger": "t"}}
        scenes = [
            {"scene_alias": "s1", "predecessor_scene_aliases": [],
             "primary_action": action, "secondary_action": None},
            {"scene_alias": "s2", "predecessor_scene_aliases": [],
             "primary_action": action, "secondary_action": action},
            {"scene_alias": "s3", "predecessor_scene_aliases": ["s1"],
             "primary_action": action, "secondary_action": None},
        ]
        view = compile_allowed_continuation_slot_view(scenes)
        self.assertEqual(view["by_scene_ref"]["S03"], ["S01.primary_action"])
        self.assertNotIn("S02.primary_action", view["by_scene_ref"]["S03"])

    def test_05_null_secondary_slot_is_not_available(self):
        namespace = compile_action_slot_namespace(3)
        self.assertIn("S01.secondary_action", namespace["S02"])
        scenes = [{"scene_alias": "s1", "predecessor_scene_aliases": [],
                   "primary_action": {}, "secondary_action": None},
                  {"scene_alias": "s2", "predecessor_scene_aliases": ["s1"],
                   "primary_action": {}, "secondary_action": None}]
        self.assertNotIn("S01.secondary_action",
                         compile_allowed_continuation_slot_view(scenes)[
                             "by_scene_ref"]["S02"])

    def test_06_link_type_values_are_not_action_types(self):
        for value in ("backstory_expansion", "relationship_followup"):
            result = validate_action_instance(
                {"action_mode": "new", "action_type": value, "arguments": {}})
            self.assertTrue(any("unknown_action_type" in row for row in result["errors"]))
            self.assertNotIn(value, NARRATIVE_ACTION_CONTRACTS)

    def test_07_action_availability_view_is_not_concrete_menu(self):
        view = compile_action_availability_view(
            planning_state=probe8_inputs()["planning"])
        self.assertTrue(validate_action_availability_view(view)["valid"])
        encoded = json.dumps(view)
        for forbidden in ('"action_id"', '"bundle_id"', '"candidate_update_hints"'):
            self.assertNotIn(forbidden, encoded)

    def test_08_unavailable_action_is_rejected(self):
        values = probe8_inputs()
        view = compile_action_availability_view(planning_state=values["planning"])
        action = {"action_mode": "new", "action_type": "advance_shared_task",
                  "arguments": {"task_ref": "invented", "progress_type": "start",
                      "responsible_participants": [values["planning"]["entities"][0]["entity_ref"]],
                      "new_step_or_obstacle": "a step"}}
        errors = validate_action_against_availability(
            action, path="action:S01.primary_action", availability_view=view)
        self.assertTrue(any("action_unavailable" in row for row in errors))

    def test_09_no_task_refs_forbid_advance_shared_task(self):
        view = compile_action_availability_view(
            planning_state=probe8_inputs()["planning"])
        self.assertEqual(view["typed_reference_registry"]["task_refs"], [])
        self.assertFalse(view["actions"]["advance_shared_task"]["available"])

    def test_10_fabricated_task_and_cross_type_ref_are_rejected(self):
        values = probe8_inputs(); view = compile_action_availability_view(
            planning_state=values["planning"])
        participant = values["planning"]["entities"][0]["entity_ref"]
        for task_ref in ("fabricated:task", participant):
            action = {"action_mode": "new", "action_type": "advance_shared_task",
                "arguments": {"task_ref": task_ref, "progress_type": "start",
                    "responsible_participants": [participant],
                    "new_step_or_obstacle": "a step"}}
            errors = validate_action_against_availability(
                action, path="action:S01.primary_action", availability_view=view)
            self.assertTrue(any("unknown_typed_ref:task_refs" in row for row in errors))

    def test_11_participant_refs_are_typed(self):
        view = compile_action_availability_view(
            planning_state=probe8_inputs()["planning"])
        action = {"action_mode": "new", "action_type": "reveal_or_transfer_information",
                  "arguments": {"sender": "fabricated", "receiver": "also-fabricated",
                                "information": "x", "transfer_mode": "direct"}}
        errors = validate_action_against_availability(
            action, path="action:S01.primary_action", availability_view=view)
        self.assertEqual(sum("unknown_typed_ref:participant_refs" in row for row in errors), 2)

    def test_12_conflict_participants_minimum_and_distinct(self):
        base = {"action_mode": "new", "action_type": "resolve_or_deescalate_conflict",
                "arguments": {"conflict_ref_or_subject": "c", "resolution_mode": "talk",
                    "participants": ["p"], "new_working_arrangement": "arrangement"}}
        self.assertTrue(any("too_short" in row for row in
                            validate_action_instance(base)["errors"]))
        broken = deepcopy(base); broken["arguments"]["participants"] = ["p", "p"]
        self.assertTrue(any("duplicate_items" in row for row in
                            validate_action_instance(broken)["errors"]))

    def test_13_internal_state_action_is_valid_for_one_target(self):
        values = probe8_inputs(); view = compile_action_availability_view(
            planning_state=values["planning"])
        target = values["planning"]["entities"][0]["entity_ref"]
        action = {"action_mode": "new", "action_type": "develop_character_or_task_state",
                  "arguments": {"target": target, "dimension": "confidence",
                                "direction": "increase", "trigger": "reflection"}}
        self.assertEqual(validate_action_against_availability(
            action, path="action:S01.primary_action", availability_view=view), [])

    def test_14_prompt_has_namespace_availability_and_final_check(self):
        inputs, source, placements, _ = migrated_fixture(5)
        prompt, _ = build_planner_prompt_v11(
            root_id=inputs["planning_state"]["root_id"],
            planning_state=inputs["planning_state"], source_anchor_registry=source,
            placement_choice_registry=placements)
        for text in ("SCENE ALIAS", "ACTION SLOT REFERENCE", "LINK TYPES", "ACTION TYPES",
                     "ACTION AVAILABILITY AND TYPED REFERENCES", "FINAL ACTION CHECK",
                     "backstory_expansion and relationship_followup"):
            self.assertIn(text, prompt)

    def test_15_retry_guidance_is_exact_and_has_no_old_response(self):
        raw = probe8_inputs()["plan"]
        _, _, view, validation = validate_probe8(raw)
        guidance = planner_retry_guidance_v11(
            validation["errors"], validation=validation,
            action_availability_view=view)
        joined = "\n".join(guidance)
        self.assertIn("S01.secondary_action", joined)
        self.assertIn("task refs=[]", joined)
        self.assertIn("14 ACTION TYPES", joined)
        self.assertNotIn("potential job loss due to company restructuring", joined)

    def test_16_probe8_original_detects_five_root_causes(self):
        _, _, _, result = validate_probe8(probe8_inputs()["plan"])
        errors = result["errors"]
        checks = {
            "slot": any("S02.primary_action:continuation_slot_format" in row for row in errors),
            "backstory_namespace": any("S04.primary_action:unknown_action_type:backstory_expansion" in row for row in errors),
            "fabricated_task": any("S05.primary_action" in row and "task_ref" in row for row in errors),
            "relationship_namespace": any("S06.primary_action:unknown_action_type:relationship_followup" in row for row in errors),
            "conflict_participants": any("S07.primary_action.arguments.participants:too_short" in row for row in errors),
        }
        self.assertTrue(all(checks.values()), (checks, errors))

    def test_17_probe8_shadow_repair_passes_every_gate(self):
        plan = repaired_probe8_plan()
        values, _, _, result = validate_probe8(plan)
        self.assertTrue(result["valid"], result["errors"])
        compiled = compile_continuous_round_plan(
            plan, source_anchor_registry=values["source"],
            placement_choice_registry=values["placements"])
        self.assertEqual(len(build_scene_dependency_dag(compiled)["nodes"]), 7)
        gate = validate_round_blueprint_structure(
            compiled_plan=compiled, source_anchor_registry=values["source"],
            placement_choice_registry=values["placements"],
            planning_state=values["planning"], complete_dialogue=values["complete"])
        self.assertTrue(gate["valid"], gate["errors"])

    def test_18_five_six_seven_scene_fixtures_pass_explicit_availability(self):
        for count in (5, 6, 7):
            inputs, source, placements, plan = migrated_fixture(count)
            refs = {row["entity_ref"] for row in inputs["planning_state"]["entities"]}
            view = compile_action_availability_view(
                planning_state=inputs["planning_state"])
            result = validate_continuous_round_plan(
                plan, source_anchor_registry=source,
                placement_choice_registry=placements,
                authoritative_entity_refs=refs, action_availability_view=view)
            self.assertTrue(result["valid"], (count, result["errors"]))
            compiled = compile_continuous_round_plan(
                plan, source_anchor_registry=source,
                placement_choice_registry=placements)
            self.assertEqual(len(build_scene_dependency_dag(compiled)["nodes"]), count)
            gate = validate_round_blueprint_structure(
                compiled_plan=compiled, source_anchor_registry=source,
                placement_choice_registry=placements,
                planning_state=inputs["planning_state"],
                complete_dialogue=inputs["complete_dialogue"])
            self.assertTrue(gate["valid"], (count, gate["errors"]))

    def test_19_probe8_history_is_read_only(self):
        self.assertEqual(hash_tree(PROBE_8)["tree_sha256"], PROBE_8_SHA)


if __name__ == "__main__":
    unittest.main()
