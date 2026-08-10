from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    build_round_validator_prompt,
    compile_round_validator_model_output,
    validate_round_validator_output,
)


RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_26"
)
RUN_SHA = "ff40065df06dbc91f8e329c182806320fb1d8b61cd7bf670d07b8650080b8de2"


class Canary26RoundValidatorCandidateScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator_input = json.loads((RUN / "round_validator_input.json").read_text())
        cls.scenes = cls.validator_input["scenes"]
        cls.first = json.loads(
            (RUN / "attempts/round_validator_attempt_01.json").read_text()
        )["parsed_response"]

    def test_history_is_unchanged(self):
        self.assertEqual(hash_tree(RUN)["tree_sha256"], RUN_SHA)

    def test_prompt_freezes_continuation_and_durable_candidate_slots(self):
        prompt = build_round_validator_prompt(self.validator_input)
        self.assertIn("only_slots_eligible_for_durable_update_candidates", prompt)
        self.assertIn("continuation_slots_requiring_progression_evidence", prompt)
        self.assertIn("Do not create one candidate per scene", prompt)
        self.assertIn("Never emit placeholder or one-per-scene rows", prompt)
        self.assertIn('"action_ref":"S01.primary_action"', prompt)
        self.assertIn('"action_ref":"S03.primary_action"', prompt)

    def test_actual_first_response_failure_is_contract_not_semantic_rejection(self):
        compiled = compile_round_validator_model_output(
            self.first, realized_scenes=self.scenes)
        self.assertFalse(compiled["valid"])
        self.assertEqual(compiled["errors"], [
            "scene_review:5:primary_action_realization:continuation_model_schema",
            "scene_review:6:primary_action_realization:continuation_model_schema",
        ])
        self.assertTrue(all(row["decision"] == "accept"
                            for row in self.first["scene_reviews"]))

    def test_scoped_shadow_fixture_passes_without_changing_semantics(self):
        output = deepcopy(self.first)
        scenes_by_alias = {row["scene_alias"]: row for row in self.scenes}
        anchor_refs = {row["source_anchor_id"]: row["source_refs"]
                       for row in self.validator_input["historical_anchor_index"]}
        for review in output["scene_reviews"]:
            source_anchor_id = scenes_by_alias[review["scene_alias"]]["source_anchor_id"]
            review["historical_supporting_turn_refs"] = anchor_refs[source_anchor_id][:1]
        for alias in ("s6", "s7"):
            review = next(row for row in output["scene_reviews"]
                          if row["scene_alias"] == alias)
            realization = review["primary_action_realization"]
            realization["progression_evidence_turn_refs"] = \
                realization["evidence_turn_refs"][:1]
        updates = []
        for source in output["durable_updates"]:
            if source["candidate_update_key"] == "DU001":
                item = deepcopy(source)
                item["verified_connection"]["evidence_turn_refs"] = \
                    item["evidence_turn_refs"][:1]
                updates.append(item)
            elif source["candidate_update_key"] == "DU003":
                item = deepcopy(source)
                item.pop("verified_connection", None)
                item["candidate_update_key"] = "DU002"
                item["depends_on_candidate_keys"] = ["DU001"]
                updates.append(item)
        output["durable_updates"] = updates
        output["semantic_review_observations"] = []
        compiled = compile_round_validator_model_output(
            output, realized_scenes=self.scenes)
        self.assertTrue(compiled["valid"], compiled["errors"])
        validation = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.scenes,
            review_world_view=json.loads((RUN / "review_world_view.json").read_text()),
            local_gate_results=json.loads(
                (RUN / "local_scene_gate_results.json").read_text()),
        )
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual([row["source_scene_ref"] for row in
                          compiled["compiled_output"]["durable_updates"]],
                         ["R01:S01", "R01:S03"])


if __name__ == "__main__":
    unittest.main()
