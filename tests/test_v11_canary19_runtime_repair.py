"""Offline regression coverage for the two Canary19 live failures."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_realizer_retry_prompt_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    build_round_validator_prompt,
    compile_round_validator_model_output,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract import (
    ROUND_VALIDATOR_PROMPT_VERSION,
    ROUND_VALIDATOR_RETRY_PROMPT_VERSION,
)


CANARY19 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_19"
)
CANARY19_TREE_SHA = (
    "bce319c996f082151ffe988d048587e94be5ae65ad036dbf390d2f11fc58d201"
)


class Canary19RuntimeRepairTests(unittest.TestCase):
    def test_realizer_retry_is_attempt_specific_and_freezes_exact_copies(self) -> None:
        forbidden = ["A copied predecessor utterance."]
        second = build_realizer_retry_prompt_v11(
            base_prompt="BASE", frozen_input_sha256="abc", errors=[
                "previous_child_exact_copy"],
            turn_budget={"minimum": 5, "maximum": 7}, retry_attempt=2,
            forbidden_utterances=forbidden,
        )
        third = build_realizer_retry_prompt_v11(
            base_prompt="BASE", frozen_input_sha256="abc", errors=[
                "previous_child_exact_copy"],
            turn_budget={"minimum": 5, "maximum": 7}, retry_attempt=3,
            forbidden_utterances=forbidden,
        )
        self.assertNotEqual(second, third)
        self.assertIn("RETRY 2 OF 3", second)
        self.assertIn("RETRY 3 OF 3", third)
        self.assertIn(forbidden[0], second)
        self.assertIn("Do not retell, summarize, or replay", second)

    def test_round_prompt_ends_with_exact_typed_output_contract(self) -> None:
        value = json.loads((CANARY19 / "round_validator_input.json").read_text())
        prompt = build_round_validator_prompt(value)
        self.assertTrue(ROUND_VALIDATOR_PROMPT_VERSION.endswith(
            "v17_uniform_action_review"))
        self.assertTrue(ROUND_VALIDATOR_RETRY_PROMPT_VERSION.endswith(
            "v12_uniform_action_review"))
        self.assertIn('"round_decision":"accept"', prompt)
        self.assertIn('"future_knowledge_leak":false', prompt)
        self.assertIn('"scene_requires_update_commit":false', prompt)
        self.assertIn('Begin exactly {"round_decision":', prompt)
        self.assertIn("Candidate keys are unique", prompt)

    def test_canary19_response_remains_invalid_without_manual_repair(self) -> None:
        attempt = json.loads((CANARY19 / "attempts/round_validator_attempt_01.json").read_text())
        scenes = [json.loads((CANARY19 / f"scenes/R01_S0{i}/realized_scene.json").read_text())
                  for i in (1, 2, 3)]
        result = compile_round_validator_model_output(
            attempt["parsed_response"], realized_scenes=scenes)
        self.assertFalse(result["valid"])
        self.assertIn("validator_model_top_level_schema", result["errors"])

    def test_strict_corrected_fixture_reaches_full_validation(self) -> None:
        attempt = json.loads((CANARY19 / "attempts/round_validator_attempt_01.json").read_text())
        scenes = [json.loads((CANARY19 / f"scenes/R01_S0{i}/realized_scene.json").read_text())
                  for i in (1, 2, 3)]
        output = deepcopy(attempt["parsed_response"])
        output["round_decision"] = "partial_accept"
        output["round_reason_codes"] = ["partial_scene_rejection"]
        for index, review in enumerate(output["scene_reviews"]):
            review["future_knowledge_leak"] = False
            review["scene_requires_update_commit"] = index == 0
            for role in ("primary", "secondary"):
                action = scenes[index].get(f"{role}_action")
                realization = review.get(f"{role}_action_realization")
                if isinstance(action, dict) and action.get("action_mode") == "continuation":
                    realization["progression_evidence_turn_refs"] = list(
                        realization.get("evidence_turn_refs") or [])[:1]
        output["durable_updates"] = [
            {
                "candidate_update_key": "DU001", "source_scene_ref": "R01:S01",
                "source_action_role": "primary", "evidence_turn_refs": ["R01:S01:T005"],
                "verification_status": "confirmed", "atomic_group_key": None,
                "depends_on_candidate_keys": [],
                "verified_connection": {
                    "verification_status": "confirmed",
                    "connection_kind": "interaction_link", "relation_type": None,
                    "evidence_turn_refs": ["R01:S01:T005"],
                    "reason": "Dialogue explicitly connects Friend C to Co-workers A.",
                },
            },
            {
                "candidate_update_key": "DU002", "source_scene_ref": "R01:S01",
                "source_action_role": "secondary", "evidence_turn_refs": ["R01:S01:T001"],
                "verification_status": "confirmed", "atomic_group_key": None,
                "depends_on_candidate_keys": [],
            },
        ]
        output["semantic_review_observations"] = []
        compilation = compile_round_validator_model_output(output, realized_scenes=scenes)
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"], realized_scenes=scenes,
            review_world_view=json.loads((CANARY19 / "review_world_view.json").read_text()),
            local_gate_results=json.loads((CANARY19 / "local_scene_gate_results.json").read_text()),
        )
        self.assertTrue(validation["valid"], validation["errors"])

    def test_canary19_tree_is_unchanged(self) -> None:
        self.assertEqual(hash_tree(CANARY19)["tree_sha256"], CANARY19_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
