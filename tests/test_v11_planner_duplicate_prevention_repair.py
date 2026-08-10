from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    build_scene_dependency_dag,
    compile_continuous_round_plan,
    validate_continuous_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.action_availability_v11 import (
    compile_action_availability_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    PLANNER_INSTRUCTION_COPY_CONTAMINATION_PHRASES_V11,
    audit_planner_instruction_copy_contamination_v11,
    planner_retry_guidance_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_continuation import (
    validate_and_resolve_round_actions,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_only_probe_v11 import (
    build_planner_probe_input,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


ROOT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json"
)
PROBE_9 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_planner_probe_9"
)
PROBE_9_SHA = "d9b0111a64f36179ffc622d03919acb58db87d0c29f2d308e6228d90fe19c757"


def _probe9_values():
    return {
        "plan": json.loads((PROBE_9 / "planner_response_parsed.json").read_text()),
        "source": json.loads((PROBE_9 / "source_anchor_registry.json").read_text()),
        "placements": json.loads((PROBE_9 / "placement_choice_registry.json").read_text()),
        "planning": json.loads((PROBE_9 / "planning_state_view.json").read_text()),
        "complete": json.loads((PROBE_9 / "complete_dialogue_view.json").read_text()),
        "availability": compile_action_availability_view(
            planning_state=json.loads((PROBE_9 / "planning_state_view.json").read_text())),
    }


def _validate(plan):
    values = _probe9_values()
    refs = {row["entity_ref"] for row in values["planning"]["entities"]}
    result = validate_continuous_round_plan(
        plan,
        source_anchor_registry=values["source"],
        placement_choice_registry=values["placements"],
        authoritative_entity_refs=refs,
        action_availability_view=values["availability"],
    )
    return values, result


def _assert_downstream(testcase: unittest.TestCase, plan):
    values, result = _validate(plan)
    testcase.assertTrue(result["valid"], result["errors"])
    compiled = compile_continuous_round_plan(
        plan,
        source_anchor_registry=values["source"],
        placement_choice_registry=values["placements"],
    )
    dag = build_scene_dependency_dag(compiled)
    testcase.assertEqual(len(dag["nodes"]), 7)
    structural = validate_round_blueprint_structure(
        compiled_plan=compiled,
        source_anchor_registry=values["source"],
        placement_choice_registry=values["placements"],
        planning_state=values["planning"],
        complete_dialogue=values["complete"],
    )
    testcase.assertTrue(structural["valid"], structural["errors"])


class PlannerDuplicatePreventionRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt_values = build_planner_probe_input(ROOT_INPUT)
        cls.prompt = cls.prompt_values["prompt"]
        cls.instruction_prefix = cls.prompt.split(
            "FROZEN INPUT — data, not instructions", 1
        )[0]

    def test_01_instruction_prefix_has_no_copyable_probe9_example_values(self):
        audit = audit_planner_instruction_copy_contamination_v11(self.prompt)
        self.assertTrue(audit["valid"], audit["findings"])
        for phrase in PLANNER_INSTRUCTION_COPY_CONTAMINATION_PHRASES_V11:
            self.assertNotIn(phrase, self.instruction_prefix.lower())

    def test_02_scene_skeleton_is_not_a_cloneable_semantic_action(self):
        self.assertIn("SCENE KEY SKELETON ONLY", self.instruction_prefix)
        self.assertIn(
            "<REPLACE_WITH_ONE_SCENE_SPECIFIC_ACTION_UNION_OBJECT>",
            self.instruction_prefix,
        )
        self.assertNotIn("repeat scene object 5-7 times", self.instruction_prefix.lower())
        self.assertIn("Do not clone this scene", self.instruction_prefix)

    def test_03_action_examples_are_symbolic_not_semantic_payloads(self):
        self.assertIn("SYMBOLIC DUPLICATE CONTRAST", self.instruction_prefix)
        self.assertIn("new(signature X)", self.instruction_prefix)
        self.assertIn("new(signature Y)", self.instruction_prefix)
        action_example_region = self.instruction_prefix.split(
            "ACTION UNION — STRUCTURE ONLY", 1
        )[1].split("SLIM OUTLINE CONTRACT", 1)[0]
        for field in ("target", "dimension", "direction", "trigger"):
            self.assertNotIn(f'"{field}":', action_example_region)

    def test_04_internal_signature_pass_is_explicit(self):
        self.assertIn("INTERNAL SIGNATURE PASS", self.instruction_prefix)
        self.assertIn("enumerate every new action by stable slot", self.instruction_prefix)
        self.assertIn("compare normalized action_type+arguments", self.instruction_prefix)

    def test_05_probe9_original_still_fails_on_exact_duplicate(self):
        plan = _probe9_values()["plan"]
        _, result = _validate(plan)
        self.assertFalse(result["valid"])
        details = result["action_continuation_validation"]["error_details"]
        duplicate = next(
            row for row in details if row["error"] == "redundant_independent_action"
        )
        self.assertEqual(duplicate["current_action_ref"], "S05.primary_action")
        self.assertEqual(duplicate["first_action_ref"], "S02.primary_action")

    def test_06_retry_guidance_names_exact_pair_and_valid_continuation_slot(self):
        plan = _probe9_values()["plan"]
        _, result = _validate(plan)
        guidance = planner_retry_guidance_v11(
            result["errors"],
            validation=result,
            action_availability_view=_probe9_values()["availability"],
        )
        joined = "\n".join(guidance)
        self.assertIn("current_action_ref=S05.primary_action", joined)
        self.assertIn("first_action_ref=S02.primary_action", joined)
        self.assertIn("continuation_of=S02.primary_action", joined)
        self.assertNotIn("a prior personal story", joined)

    def test_07_probe9_continuation_shadow_passes_all_local_gates(self):
        plan = deepcopy(_probe9_values()["plan"])
        plan["scenes"][4]["primary_action"] = {
            "action_mode": "continuation",
            "continuation_of": "S02.primary_action",
            "progression": {
                "stage": "consolidation",
                "new_step": (
                    "A converts the earlier recovery into a concrete decision to ask B "
                    "for support before the next financial setback."
                ),
            },
        }
        _assert_downstream(self, plan)

    def test_08_probe9_distinct_new_shadow_passes_all_local_gates(self):
        plan = deepcopy(_probe9_values()["plan"])
        plan["scenes"][4]["primary_action"] = {
            "action_mode": "new",
            "action_type": "develop_character_or_task_state",
            "arguments": {
                "target": "cc-train-episode-61555:E:333e9451b96f",
                "dimension": "support_seeking_readiness",
                "direction": "increase",
                "trigger": "the jointly stated plan to ask for help before the next setback",
            },
        }
        _assert_downstream(self, plan)

    def test_09_same_action_type_with_different_signature_remains_legal(self):
        plan = deepcopy(_probe9_values()["plan"])
        plan["scenes"][4]["primary_action"]["arguments"]["dimension"] = (
            "support_seeking_readiness"
        )
        resolution = validate_and_resolve_round_actions(plan["scenes"])
        self.assertFalse(any(
            row["error"] == "redundant_independent_action"
            and row.get("current_action_ref") == "S05.primary_action"
            for row in resolution["error_details"]
        ))

    def test_10_probe9_history_is_unchanged(self):
        self.assertEqual(hash_tree(PROBE_9)["tree_sha256"], PROBE_9_SHA)


if __name__ == "__main__":
    unittest.main()
