from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.prompts import (
    SLIM_PLANNER_PROMPT_VERSION,
    SLIM_PLANNER_RETRY_PROMPT_VERSION,
    build_slim_planner_prompt,
    build_slim_planner_retry_prompt,
    compact_planner_retry_errors,
    planner_action_selection_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.schemas import (
    validate_slim_planner_output,
)


HISTORICAL_V3_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v3")


class SlimPlannerRetryV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.menu = json.loads(
            (HISTORICAL_V3_RUN / "affordance_menu.json").read_text(encoding="utf-8"))
        cls.memory = json.loads(
            (HISTORICAL_V3_RUN / "world_memory_initial.json").read_text(encoding="utf-8"))
        cls.attempts = [json.loads(
            (HISTORICAL_V3_RUN / f"attempts/planner-a{index:02d}.json").read_text(
                encoding="utf-8")) for index in (1, 2, 3)]
        cls.validations = [json.loads(
            (HISTORICAL_V3_RUN /
             f"planner/validation_attempt_{index:02d}.json").read_text(
                encoding="utf-8")) for index in (1, 2, 3)]

    def test_selection_view_excludes_program_owned_operation_fields(self) -> None:
        view = planner_action_selection_view(self.menu)
        encoded = json.dumps(view, sort_keys=True)
        self.assertNotIn('"operation"', encoded)
        self.assertNotIn('"operation_class"', encoded)
        self.assertEqual(len(view["actions"]), len(self.menu["actions"]))
        self.assertTrue(view["frozen_action_menu_sha256"])

    def test_initial_prompt_places_exact_output_shape_last(self) -> None:
        prompt = build_slim_planner_prompt(
            parent_turns=[], action_menu=self.menu, role_table=self.memory["entities"])
        self.assertIn(SLIM_PLANNER_PROMPT_VERSION, prompt)
        self.assertIn("participants must be a nonempty subset", prompt)
        self.assertIn("selected_memory_threads must include", prompt)
        output_shape = json.loads(prompt.splitlines()[-1])
        self.assertEqual(set(output_shape), {
            "planner_schema_version", "selected_memory_threads",
            "scene_budget", "scene_goals",
        })

    def test_retry_prompt_has_no_competing_required_shape_wrapper(self) -> None:
        prompt = build_slim_planner_retry_prompt(
            action_menu=self.menu, role_table=self.memory["entities"],
            errors=self.validations[0]["errors"])
        self.assertIn(SLIM_PLANNER_RETRY_PROMPT_VERSION, prompt)
        self.assertIn("READ-ONLY FROZEN SELECTION REFERENCE", prompt)
        self.assertIn("OUTPUT ONLY ONE JSON OBJECT", prompt)
        self.assertNotIn('"required_shape"', prompt)
        self.assertNotIn('"operation":', prompt)
        self.assertNotIn('"operation_class":', prompt)
        output_shape = json.loads(prompt.splitlines()[-1])
        self.assertEqual(set(output_shape), {
            "planner_schema_version", "selected_memory_threads",
            "scene_budget", "scene_goals",
        })

    def test_echo_errors_are_compacted_for_next_retry(self) -> None:
        compact = compact_planner_retry_errors(self.validations[1]["errors"])
        self.assertEqual(compact[:3], [
            "retry_reference_object_was_echoed",
            "output_must_not_contain_actions_or_reference_wrapper_fields",
            "output_must_have_exact_four_top_level_fields",
        ])
        self.assertFalse(any("planner_program_field:$model.actions" in row
                             for row in compact))

    def test_historical_echo_outputs_remain_rejected_not_silently_repaired(self) -> None:
        for record in self.attempts[1:]:
            result = validate_slim_planner_output(
                record["parsed_response"], action_menu=self.menu)
            self.assertFalse(result["valid"])
            self.assertIn("planner_top_level_schema", result["errors"])

    def test_historical_attempt_one_has_a_valid_selection_only_repair(self) -> None:
        repaired = copy.deepcopy(self.attempts[0]["parsed_response"])
        repaired["scene_goals"][0]["participants"] = ["Co-workers B"]
        repaired["scene_goals"][3]["participants"] = ["Co-workers A"]
        repaired["selected_memory_threads"] = [
            "thread_002", "thread_005", "thread_006", "thread_003"]
        result = validate_slim_planner_output(repaired, action_menu=self.menu)
        self.assertTrue(result["valid"], result["errors"])

    def test_shadow_replay_does_not_modify_v3_history(self) -> None:
        before = hash_tree(HISTORICAL_V3_RUN)["tree_sha256"]
        _ = build_slim_planner_retry_prompt(
            action_menu=self.menu, role_table=self.memory["entities"],
            errors=self.validations[2]["errors"])
        self.assertEqual(before, hash_tree(HISTORICAL_V3_RUN)["tree_sha256"])
        self.assertEqual(before,
                         "311e509ff53577c61f8c2045e016516b609459e3b0fdcee657e02cdc674d4aae")


if __name__ == "__main__":
    unittest.main()
