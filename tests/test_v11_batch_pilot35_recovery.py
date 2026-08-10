from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    materialize_dialogue_only_top_level_defaults,
    rehome_dialogue_only_trailing_empty_arrays,
    run_round_validator_with_retry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    build_shape_balanced_distractor_prompt,
)


PILOT35 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_35"
)
PILOT35_SHA = "8ccb932c0f89250b7e3dfacf21298bc9d7ae4ff0c08f629019bdbbea2e8f1a24"
EXPANSION = PILOT35 / "roots/cc-train-episode-216742/01_expansion"
MCQ_CANDIDATE = PILOT35 / (
    "roots/cc-train-episode-61555/qa_candidates/candidate_05"
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot35RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_validator = json.loads((
            EXPANSION / "raw/backend_responses/round_validator_attempt_01.txt"
        ).read_text(encoding="utf-8"))
        cls.realized = [load(path) for path in sorted(
            (EXPANSION / "scenes").glob("*/realized_scene.json"))]
        cls.validator_input = load(EXPANSION / "round_validator_input.json")
        cls.review = load(EXPANSION / "review_world_view.json")
        cls.gates = load(EXPANSION / "local_scene_gate_results.json")
        cls.gold = load(MCQ_CANDIDATE / "03_semantic_audit/qa/admitted_gold.json")
        cls.view = load(MCQ_CANDIDATE / "02_questioner/frozen/qa_source_view.json")
        cls.catalog = load(MCQ_CANDIDATE / "02_questioner/frozen/span_catalog.json")

    def test_historical_validator_response_has_one_exact_closure_row(self):
        self.assertEqual(set(self.raw_validator), {
            "round_decision", "round_reason_codes", "scene_reviews"})
        self.assertEqual(len(self.raw_validator["scene_reviews"]), 7)
        self.assertEqual(self.raw_validator["scene_reviews"][-1], {
            "durable_updates": [], "semantic_review_observations": []})
        strict = compile_round_validator_model_output(
            self.raw_validator, realized_scenes=self.realized)
        self.assertFalse(strict["valid"])
        self.assertEqual(strict["errors"], ["validator_model_top_level_schema"])

    def test_dialogue_only_defaults_are_narrow_and_deterministic(self):
        rehome = rehome_dialogue_only_trailing_empty_arrays(
            self.raw_validator, expected_scene_count=6)
        self.assertTrue(rehome["rehomed"])
        result = materialize_dialogue_only_top_level_defaults(rehome["output"])
        self.assertEqual(result["materialized_fields"], [])
        self.assertEqual(result["output"]["durable_updates"], [])
        self.assertEqual(result["output"]["semantic_review_observations"], [])
        self.assertEqual(result["output"]["scene_reviews"],
                         self.raw_validator["scene_reviews"][:-1])

    def test_unknown_top_level_field_is_not_hidden(self):
        invalid = dict(self.raw_validator)
        invalid["unexpected"] = []
        result = materialize_dialogue_only_top_level_defaults(invalid)
        self.assertEqual(result["materialized_fields"], [])
        self.assertEqual(result["output"], invalid)

    def test_pilot35_first_response_routes_to_compact_contract_retry(self):
        calls = []

        def backend(prompt, attempt):
            calls.append((prompt, attempt))
            return self.raw_validator if attempt == 1 else {}

        result = run_round_validator_with_retry(
            backend_call=backend, validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates, durable_update_policy="dialogue_only")
        self.assertFalse(result["completed"])
        self.assertEqual(result["physical_calls"], 2)
        self.assertEqual(result["attempts"][0][
            "dialogue_only_materialized_fields"], [])
        self.assertTrue(result["attempts"][0][
            "dialogue_only_top_level_rehomed"])
        self.assertEqual(result["attempts"][1]["request_mode"],
                         "contract_only_repair")
        self.assertLess(len(calls[1][0]), len(calls[0][0]) // 2)

    def test_strict_mode_does_not_materialize_dialogue_only_defaults(self):
        result = run_round_validator_with_retry(
            backend_call=lambda prompt, attempt: self.raw_validator,
            validator_input=self.validator_input,
            realized_scenes=self.realized, review_world_view=self.review,
            local_gate_results=self.gates, durable_update_policy="strict")
        self.assertFalse(result["completed"])
        self.assertEqual(result["attempts"][0][
            "dialogue_only_materialized_fields"], [])

    def test_distractor_prompt_requires_cross_composability(self):
        prompt = build_shape_balanced_distractor_prompt(
            frozen_dialogue=self.view["combined_dialogue"],
            span_catalog=self.catalog, base_qa=self.gold)
        self.assertIn("CROSS-COMPOSABILITY", prompt)
        self.assertIn("mentally compose all four combinations", prompt)
        self.assertIn("only coherent one", prompt)
        self.assertIn("do not call practical problem-solving 'emotional reflection'",
                      prompt)

    def test_pilot35_remains_read_only(self):
        self.assertEqual(PILOT35_SHA, hash_tree(PILOT35)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
