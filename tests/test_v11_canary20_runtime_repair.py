"""Offline regressions for the two Canary20 contract-boundary failures."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_realizer_retry_prompt_v11,
    normalize_realizer_dialogue_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    ROUND_VALIDATOR_INPUT_PROJECTION_VERSION,
    build_round_validator_input,
    build_round_validator_prompt,
    compile_round_validator_model_output,
)


CANARY20 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_20"
)
CANARY20_TREE_SHA = (
    "b492e1703e38b4150fffedf6b5456763e8cdee424cb86511f11bbae1251fba4e"
)


def load(relative: str):
    return json.loads((CANARY20 / relative).read_text(encoding="utf-8"))


class Canary20RuntimeRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiled = load("planner/compiled_round_plan.json")
        cls.review = load("review_world_view.json")
        cls.dag = load("scene_dependency_dag.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.realized = [
            load("scenes/R01_S01/realized_scene.json"),
            load("scenes/R01_S03/realized_scene.json"),
        ]
        cls.projected = build_round_validator_input(
            round_blueprint=cls.compiled,
            review_world_view=cls.review,
            realized_scenes=cls.realized,
            scene_dependency_dag=cls.dag,
            structural_gate_results=cls.gates,
        )

    def test_round_review_projection_contains_only_realized_scenes(self) -> None:
        expected = {"R01:S01", "R01:S03"}
        self.assertEqual(ROUND_VALIDATOR_INPUT_PROJECTION_VERSION,
                         "round_validator_realized_scene_projection_v4_compact_reference_scopes")
        self.assertEqual({row["scene_ref"] for row in self.projected["scenes"]},
                         expected)
        self.assertEqual({row["scene_ref"] for row in
                          self.projected["narrative_constraints"]}, expected)
        self.assertEqual(set(self.projected["structural_gate_results"]), expected)
        self.assertEqual(set(self.projected["narrative_order"]["scenes"][i]["scene_ref"]
                             for i in range(len(self.projected["narrative_order"]["scenes"]))),
                         expected)
        self.assertEqual(set(self.projected["scene_dependency_dag"]["nodes"]), expected)
        self.assertTrue(all(edge["from_scene_ref"] in expected and
                            edge["to_scene_ref"] in expected
                            for edge in self.projected["scene_dependency_dag"]["edges"]))

    def test_round_prompt_explicitly_forbids_reviews_of_absent_scenes(self) -> None:
        prompt = build_round_validator_prompt(self.projected)
        self.assertIn("Only review aliases in scenes", prompt)
        self.assertNotIn('"scene_alias":"s2"', prompt)
        self.assertNotIn('"scene_alias":"s4"', prompt)

    def test_old_overcomplete_review_remains_strictly_invalid(self) -> None:
        old = load("attempts/round_validator_attempt_01.json")["parsed_response"]
        result = compile_round_validator_model_output(old, realized_scenes=self.realized)
        self.assertFalse(result["valid"])
        self.assertIn("scene_review:2:unknown_scene_alias:s2", result["errors"])

    def test_s02_old_outputs_are_invalid_for_exact_speaker_registry(self) -> None:
        allowed = {"cc-train-episode-61555:E:333e9451b96f", "R01:P01"}
        for attempt in range(1, 4):
            output = load(
                f"attempts/realizer_R01_S02_attempt_{attempt:02d}.json"
            )["parsed_response"]
            result = normalize_realizer_dialogue_v11(
                output, scene_ref="R01:S02", allowed_speaker_refs=allowed)
            self.assertFalse(result["valid"])
            self.assertTrue(any(error.startswith("realizer_speaker_ref:")
                                for error in result["errors"]))

    def test_retry_exposes_exact_allowed_and_rejected_speaker_refs(self) -> None:
        allowed = ["cc-train-episode-61555:E:333e9451b96f", "R01:P01"]
        invalid = ["cc-train-episode-61555:E:e54518dcbafe"]
        prompt = build_realizer_retry_prompt_v11(
            base_prompt="BASE", frozen_input_sha256="sha",
            errors=["realizer_speaker_ref:2"], retry_attempt=2,
            turn_budget={"minimum": 5, "maximum": 7},
            allowed_speaker_refs=allowed, invalid_speaker_refs=invalid,
        )
        self.assertIn("EXACT ALLOWED speaker_ref VALUES", prompt)
        self.assertIn("INVALID speaker_ref VALUES", prompt)
        for ref in [*allowed, *invalid]:
            self.assertIn(ref, prompt)

    def test_retry_does_not_locally_rewrite_speaker_refs(self) -> None:
        old = load("attempts/realizer_R01_S02_attempt_01.json")["parsed_response"]
        invalid_before = old["dialogue_turns"][1]["speaker_ref"]
        build_realizer_retry_prompt_v11(
            base_prompt="BASE", frozen_input_sha256="sha",
            errors=["realizer_speaker_ref:2"], retry_attempt=2,
            allowed_speaker_refs=["cc-train-episode-61555:E:333e9451b96f", "R01:P01"],
            invalid_speaker_refs=[invalid_before],
        )
        self.assertEqual(old["dialogue_turns"][1]["speaker_ref"], invalid_before)

    def test_historical_canary_is_unchanged(self) -> None:
        self.assertEqual(hash_tree(CANARY20)["tree_sha256"], CANARY20_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
