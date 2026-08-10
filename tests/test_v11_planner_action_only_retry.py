from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    DEFAULT_INPUT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_action_repair_v11 import (
    apply_planner_action_repairs_v11,
    build_planner_action_only_retry_prompt_v11,
    compile_planner_action_repair_spec_v11,
    planner_action_only_retry_eligible,
    validate_planner_action_repair_response_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_only_probe_v11 import (
    build_planner_probe_input,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    migrate_legacy_connected_person_contract_v2,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    compile_planner_outline_v11,
    validate_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


CANARY_4 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_4"
)
CANARY_4_SHA = "a07b0f9a09968141f0a72250fe2fb516227783fa495c78440ef6fbc6f9649c70"


class PlannerActionOnlyRetryV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = build_planner_probe_input(DEFAULT_INPUT)
        cls.entity_refs = {row["entity_ref"] for row in cls.values["role_table"]}
        cls.plans = []
        cls.validations = []
        cls.specs = []
        for attempt in range(1, 4):
            record = json.loads((CANARY_4 / (
                f"attempts/planner_v11_attempt_{attempt:02d}.json")).read_text())
            plan = migrate_legacy_connected_person_contract_v2(
                record["parsed_response"])
            validation = cls.validate_outline(plan)
            spec = compile_planner_action_repair_spec_v11(
                outline=plan, errors=validation["errors"], validation=validation,
                action_contracts=cls.values["shared_narrative_action_contracts"],
                action_availability_view=cls.values["action_availability_view"],
                authoritative_role_table=cls.values["role_table"],
            )
            cls.plans.append(plan)
            cls.validations.append(validation)
            cls.specs.append(spec)

    @classmethod
    def validate_outline(cls, plan):
        return validate_planner_outline_v11(
            plan,
            source_anchor_registry=cls.values["source_anchor_registry"],
            placement_choice_registry=cls.values["placement_choice_registry"],
            authoritative_entity_refs=cls.entity_refs,
            action_availability_view=cls.values["action_availability_view"],
        )

    @classmethod
    def assert_full_downstream(cls, testcase, plan):
        validation = cls.validate_outline(plan)
        testcase.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_outline_v11(
            plan,
            source_anchor_registry=cls.values["source_anchor_registry"],
            placement_choice_registry=cls.values["placement_choice_registry"],
            planning_state=cls.values["planning_state"],
            action_availability_view=cls.values["action_availability_view"],
        )
        gate = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=cls.values["source_anchor_registry"],
            placement_choice_registry=cls.values["placement_choice_registry"],
            planning_state=cls.values["planning_state"],
            complete_dialogue=cls.values["complete_dialogue"],
        )
        testcase.assertTrue(gate["valid"], gate["errors"])

    def fixture_responses(self):
        refs = sorted(self.entity_refs)
        return [
            {"action_repairs": [{
                "action_ref": "S01.primary_action",
                "replacement": {
                    "action_mode": "new",
                    "action_type": "introduce_external_constraint",
                    "arguments": {
                        "constraint": "The proposed outing must fit both coworkers' budgets.",
                        "affected_participants": refs,
                        "affected_plan_or_task": "the proposed group outing",
                    },
                },
            }]},
            {"action_repairs": [{
                "action_ref": "S04.secondary_action",
                "replacement": {
                    "action_mode": "new",
                    "action_type": "develop_character_or_task_state",
                    "arguments": {
                        "target": refs[0],
                        "dimension": "emotional_support",
                        "direction": "increase",
                        "trigger": (
                            "Discussion of the new constraint and its impact "
                            "on their relationship."
                        ),
                    },
                },
            }]},
            {"action_repairs": [{
                "action_ref": "S06.primary_action",
                "replacement": {
                    "action_mode": "continuation",
                    "continuation_of": "S05.secondary_action",
                    "progression": {
                        "stage": "resolution",
                        "new_step": (
                            "A converts the earlier emotional support into a concrete plan "
                            "for handling the workplace constraint."
                        ),
                    },
                },
            }]},
        ]

    def test_01_all_three_real_failures_are_action_only_eligible(self):
        for plan, validation in zip(self.plans, self.validations):
            self.assertTrue(planner_action_only_retry_eligible(
                outline=plan, errors=validation["errors"]))

    def test_02_cascading_continuations_target_only_invalid_root(self):
        self.assertEqual(self.specs[0]["target_action_refs"], ["S01.primary_action"])
        self.assertEqual(self.specs[1]["target_action_refs"], ["S04.secondary_action"])
        self.assertEqual(self.specs[2]["target_action_refs"], ["S06.primary_action"])

    def test_03_non_action_or_parse_failure_uses_full_outline_retry(self):
        self.assertFalse(planner_action_only_retry_eligible(
            outline=self.plans[0], errors=["scene:1:text:goal"]))
        self.assertFalse(planner_action_only_retry_eligible(
            outline=self.plans[0], errors=self.validations[0]["errors"],
            parse_errors=["json_parse_invalid"]))

    def test_04_prompt_is_compact_and_does_not_include_full_old_outline(self):
        for plan, spec in zip(self.plans, self.specs):
            prompt = build_planner_action_only_retry_prompt_v11(spec)
            self.assertNotIn(plan["round_goal"], prompt)
            self.assertNotIn('"scenes"', prompt)
            self.assertNotIn('"replacement": {}', prompt)
            self.assertIn(
                "<COMPLETE_NEW_OR_CONTINUATION_ACTION_OBJECT_REQUIRED>", prompt)
            self.assertIn("COMPLETE REPLACEMENT ENVELOPE", prompt)
            self.assertIn(
                "the only keys directly inside replacement are action_mode, action_type, "
                "and arguments", prompt)
            self.assertIn("Never place them beside action_type", prompt)
            self.assertLess(len(prompt), len(self.values["prompt"]) // 2)
            self.assertTrue(prompt.rstrip().endswith("}"))

    def test_05_prompt_exposes_only_available_action_contracts(self):
        prompt = build_planner_action_only_retry_prompt_v11(self.specs[0])
        self.assertNotIn('"available":false', prompt)
        self.assertIn("participant_alias", prompt)
        self.assertNotIn(":TMP:", prompt)
        self.assertIn("are not authoritative typed references", prompt)

    def test_06_output_cannot_rewrite_outline_fields(self):
        response = deepcopy(self.fixture_responses()[0])
        response["scenes"] = []
        result = validate_planner_action_repair_response_v11(
            response, spec=self.specs[0])
        self.assertFalse(result["valid"])
        self.assertIn("planner_action_repair:top_level_exact_keys", result["errors"])

    def test_07_target_set_is_exact_and_replacement_cannot_be_null(self):
        wrong = {"action_repairs": [{
            "action_ref": "S02.primary_action", "replacement": {}}]}
        self.assertFalse(validate_planner_action_repair_response_v11(
            wrong, spec=self.specs[0])["valid"])
        deleted = {"action_repairs": [{
            "action_ref": "S01.primary_action", "replacement": None}]}
        self.assertFalse(validate_planner_action_repair_response_v11(
            deleted, spec=self.specs[0])["valid"])

    def test_07b_empty_and_placeholder_replacements_fail_before_reconstruction(self):
        for replacement, expected in (
            ({}, "planner_action_repair:item:1:replacement_empty"),
            ("<COMPLETE_NEW_OR_CONTINUATION_ACTION_OBJECT_REQUIRED>",
             "planner_action_repair:item:1:replacement_not_object"),
        ):
            response = {"action_repairs": [{
                "action_ref": "S01.primary_action",
                "replacement": replacement,
            }]}
            validation = validate_planner_action_repair_response_v11(
                response, spec=self.specs[0])
            self.assertFalse(validation["valid"])
            self.assertIn(expected, validation["errors"])
            result = apply_planner_action_repairs_v11(
                frozen_outline=self.plans[0], response=response,
                spec=self.specs[0])
            self.assertFalse(result["valid"])
            self.assertIsNone(result["reconstructed_outline"])

    def test_07c_replacement_schema_availability_and_continuation_are_checked(self):
        unavailable = deepcopy(self.fixture_responses()[0])
        unavailable["action_repairs"][0]["replacement"] = {
            "action_mode": "new",
            "action_type": "advance_shared_task",
            "arguments": {
                "task_ref": "invented_task",
                "progress_type": "advance",
                "responsible_participants": sorted(self.entity_refs),
                "new_step_or_obstacle": "Take a next step.",
            },
        }
        result = validate_planner_action_repair_response_v11(
            unavailable, spec=self.specs[0])
        self.assertFalse(result["valid"])
        self.assertIn(
            "planner_action_repair:item:1:replacement_action_not_available:"
            "advance_shared_task", result["errors"])

        continuation = deepcopy(self.fixture_responses()[0])
        continuation["action_repairs"][0]["replacement"] = {
            "action_mode": "continuation",
            "continuation_of": "S02.primary_action",
            "progression": {
                "stage": "development",
                "new_step": "Advance the earlier action in a concrete way.",
            },
        }
        result = validate_planner_action_repair_response_v11(
            continuation, spec=self.specs[0])
        self.assertFalse(result["valid"])
        self.assertIn(
            "planner_action_repair:item:1:replacement_continuation_not_allowed:"
            "S02.primary_action", result["errors"])

    def test_08_reconstruction_freezes_non_action_and_untouched_actions(self):
        for plan, spec, response in zip(
                self.plans, self.specs, self.fixture_responses()):
            result = apply_planner_action_repairs_v11(
                frozen_outline=plan, response=response, spec=spec)
            self.assertTrue(result["valid"], result["errors"])
            self.assertTrue(all(result["freeze_audit"].values()))
            for key in ("round_goal", "starting_situation", "ending_direction"):
                self.assertEqual(result["reconstructed_outline"][key], plan[key])

    def test_09_three_shadow_repairs_pass_complete_local_chain(self):
        for plan, spec, response in zip(
                self.plans, self.specs, self.fixture_responses()):
            result = apply_planner_action_repairs_v11(
                frozen_outline=plan, response=response, spec=spec)
            self.assert_full_downstream(self, result["reconstructed_outline"])

    def test_10_invalid_action_is_not_locally_auto_repaired(self):
        plan = deepcopy(self.plans[0])
        compile_planner_action_repair_spec_v11(
            outline=plan, errors=self.validations[0]["errors"],
            validation=self.validations[0],
            action_contracts=self.values["shared_narrative_action_contracts"],
            action_availability_view=self.values["action_availability_view"],
            authoritative_role_table=self.values["role_table"],
        )
        self.assertEqual(plan, self.plans[0])
        self.assertEqual(
            plan["scenes"][0]["primary_action"]["action_type"],
            "introduce_connected_event",
        )

    def test_11_canary4_history_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_4)["tree_sha256"], CANARY_4_SHA)

    def test_12_current_argument_error_shape_is_action_only_eligible(self):
        errors = [
            "action:S06.secondary_action.arguments:missing:resolution_mode",
            "action:S06.secondary_action.arguments:unexpected:conflict_source",
            "action:S06.secondary_action:action_unavailable:"
            "resolve_or_deescalate_conflict",
        ]
        self.assertTrue(planner_action_only_retry_eligible(
            outline=self.plans[0], errors=errors))


if __name__ == "__main__":
    unittest.main()
