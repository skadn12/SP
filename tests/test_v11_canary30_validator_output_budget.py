from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    ROUND_VALIDATOR_MAX_OUTPUT_TOKENS,
    audit_round_validator_prompt_budget,
    build_round_validator_input,
    build_round_validator_prompt,
    normalize_round_validator_output,
)


RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_30"
)
RUN_SHA = "1d70b7bddf6dd04fe94128f5857a742ff5ca5d656225328d348cb72336e580ef"


class Canary30ValidatorOutputBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realized = [json.loads(path.read_text()) for path in sorted(
            (RUN / "scenes").glob("*/realized_scene.json"))]
        cls.validator_input = build_round_validator_input(
            round_blueprint=json.loads(
                (RUN / "planner/compiled_round_plan.json").read_text()),
            review_world_view=json.loads((RUN / "review_world_view.json").read_text()),
            realized_scenes=cls.realized,
            scene_dependency_dag=json.loads((RUN / "scene_dependency_dag.json").read_text()),
            structural_gate_results=json.loads(
                (RUN / "local_scene_gate_results.json").read_text()),
        )

    def test_history_is_unchanged_and_failure_was_exact_output_limit(self):
        self.assertEqual(hash_tree(RUN)["tree_sha256"], RUN_SHA)
        for attempt in (1, 2):
            record = json.loads((
                RUN / f"attempts/round_validator_attempt_0{attempt}.json"
            ).read_text())
            self.assertEqual(record["response_metadata"]["finish_reason"], "length")
            self.assertEqual(record["provider_completion_tokens"], 8192)
            self.assertEqual(record["response_metadata"][
                "round_validator_max_output_tokens"], 8192)

    def test_truncated_historical_outputs_remain_strictly_invalid(self):
        for attempt in (1, 2):
            raw = (RUN / f"raw/backend_responses/round_validator_attempt_0{attempt}.txt").read_text()
            normalized = normalize_round_validator_output(raw)
            self.assertFalse(normalized["parse_valid"])

    def test_new_output_budget_and_compact_context_fit(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/root/autodl-tmp/Qwen3-4B", local_files_only=True)
        budget = audit_round_validator_prompt_budget(
            self.validator_input,
            token_counter=lambda text: len(tokenizer.encode(
                text, add_special_tokens=False)),
        )
        self.assertEqual(ROUND_VALIDATOR_MAX_OUTPUT_TOKENS, 12_288)
        self.assertEqual(budget["max_output_tokens"], 12_288)
        self.assertTrue(budget["valid"], budget)
        self.assertLess(budget["prompt_tokens"], 27_000)
        self.assertGreater(budget["remaining_tokens"], 1_000)

    def test_prompt_has_exact_optional_slots_and_single_close_rule(self):
        prompt = build_round_validator_prompt(self.validator_input)
        enums = json.loads(prompt.split(
            "ENUMS AND EXACT AUXILIARY ITEM SHAPES\n", 1
        )[1].split("\n\nFROZEN REVIEW INPUT", 1)[0])
        shapes = enums["durable_update_array_item_shapes"]
        self.assertEqual(len(shapes), 2)
        self.assertEqual([(row["source_scene_ref"], row["source_action_role"])
                          for row in shapes], [
            ("R01:S01", "primary"), ("R01:S01", "secondary"),
        ])
        self.assertTrue(all(row["verification_status"] is None for row in shapes))
        self.assertIn("Never restart or repeat the object", prompt)

    def test_known_context_projection_does_not_repeat_parent_ref_lists(self):
        self.assertTrue(all("preceding_known_context_refs" not in row
                            for row in self.validator_input["narrative_constraints"]))
        self.assertEqual(len(self.validator_input["scenes"]), 7)
        self.assertEqual(len(self.validator_input["historical_turns"]), 60)


if __name__ == "__main__":
    unittest.main()
