from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.durable_update_compiler import (
    durable_candidate_action_eligible,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    normalize_round_validator_output,
    run_round_validator_with_retry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    build_round_validator_contract_repair_plan,
    pack_round_validator_contract_repair_output,
)


RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_29"
)
RUN_SHA = "34113476751b7279ebe727df3a19b4abd6755a19e1975bf826d0e88b71651bae"


class Canary29TerminalValidatorRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator_input = json.loads((RUN / "round_validator_input.json").read_text())
        cls.realized = cls.validator_input["scenes"]
        cls.review = json.loads((RUN / "review_world_view.json").read_text())
        cls.gates = json.loads((RUN / "local_scene_gate_results.json").read_text())
        cls.raw_first = (RUN / "raw/backend_responses/round_validator_attempt_01.txt").read_text()
        cls.second = json.loads(
            (RUN / "attempts/round_validator_attempt_02.json").read_text()
        )["parsed_response"]
        compilation = compile_round_validator_model_output(
            cls.second, realized_scenes=cls.realized)
        eligible_refs = [
            str(action["action_ref"])
            for scene in cls.realized
            for action in (scene.get("primary_action"), scene.get("secondary_action"))
            if durable_candidate_action_eligible(action)
        ]
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.second,
            realized_scenes=cls.realized,
            compilation_errors=compilation["errors"],
            historical_anchor_refs={
                str(row["source_anchor_id"]): row["source_refs"]
                for row in cls.validator_input["historical_anchor_index"]
            },
            eligible_durable_action_refs=eligible_refs,
        )

    @classmethod
    def repair_fixture(cls):
        grouped = {
            "top_level_rehome": None,
            "scene_reference_repairs": [],
            "action_realization_repairs": [],
            "durable_update_evidence_repairs": [],
            "observation_reference_repairs": [],
            "unplanned_action_repairs": [],
            "future_knowledge_repairs": [],
            "durable_update_removals": [],
            "durable_dependency_repairs": [],
            "verified_connection_repairs": [],
            "verified_connection_removals": [],
            "missing_scene_reviews": [],
        }
        for row in cls.plan["scene_reference_targets"]:
            grouped["scene_reference_repairs"].append({
                "scene_alias": row["scene_alias"],
                "current_scene_supporting_turn_refs": (
                    row["allowed_current_scene_turn_refs"][:1]
                    if row["repair_current_scene_supporting_turn_refs"]
                    else row["original_current_scene_supporting_turn_refs"]),
                "dependency_context_turn_refs": (
                    row["allowed_dependency_turn_refs"][:1]
                    if row["repair_dependency_context_turn_refs"]
                    else row["original_dependency_context_turn_refs"]),
                "historical_supporting_turn_refs": (
                    row["allowed_historical_turn_refs"][:1]
                    if row["repair_historical_supporting_turn_refs"]
                    else row["original_historical_supporting_turn_refs"]),
            })
        for row in cls.plan["targets"]:
            grouped["action_realization_repairs"].append({
                "action_ref": row["action_ref"],
                "evidence_turn_refs": (
                    row["candidate_evidence_turn_refs"][:1]
                    if row["evidence_turn_refs_requires_repair"]
                    else row["original_evidence_turn_refs"]),
                "progression_evidence_turn_refs":
                    row["candidate_evidence_turn_refs"][:1],
            })
        grouped["durable_update_removals"] = [{
            "candidate_update_key": row["candidate_update_key"],
            "replacement": None,
        } for row in cls.plan["durable_removal_targets"]]
        grouped["durable_dependency_repairs"] = [{
            "candidate_update_key": row["candidate_update_key"],
            "depends_on_candidate_keys": [],
        } for row in cls.plan["durable_dependency_targets"]]
        grouped["verified_connection_removals"] = [{
            "candidate_update_key": row["candidate_update_key"],
            "replacement": None,
        } for row in cls.plan["verified_connection_removal_targets"]]
        return pack_round_validator_contract_repair_output(grouped)

    def test_historical_run_is_unchanged(self):
        self.assertEqual(hash_tree(RUN)["tree_sha256"], RUN_SHA)

    def test_first_raw_response_remains_rejected_for_duplicate_key(self):
        normalized = normalize_round_validator_output(self.raw_first)
        self.assertFalse(normalized["parse_valid"])
        self.assertIn("duplicate_json_object_key:scene_reviews", normalized["errors"][0])

    def test_second_response_has_only_continuation_contract_errors(self):
        compilation = compile_round_validator_model_output(
            self.second, realized_scenes=self.realized)
        self.assertFalse(compilation["valid"])
        self.assertEqual(len(compilation["errors"]), 7)
        self.assertTrue(all("continuation_model_schema" in error
                            for error in compilation["errors"]))
        self.assertTrue(self.plan["valid"], self.plan["errors"])
        self.assertEqual(len(self.plan["targets"]), 7)

    def test_terminal_third_request_is_contract_only_and_can_pass_fixture(self):
        responses = [self.raw_first, self.second, self.repair_fixture()]
        prompts = []

        def backend(prompt, attempt):
            prompts.append(prompt)
            return responses[attempt - 1]

        result = run_round_validator_with_retry(
            backend_call=backend,
            validator_input=self.validator_input,
            realized_scenes=self.realized,
            review_world_view=self.review,
            local_gate_results=self.gates,
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["physical_calls"], 3)
        self.assertEqual([row["request_mode"] for row in result["attempts"]], [
            "full_review", "full_contract_retry", "contract_only_repair",
        ])
        self.assertTrue(result["contract_only_repair_used"])
        self.assertLess(len(prompts[2]), len(prompts[1]))

    def test_no_fourth_request_after_invalid_terminal_repair(self):
        calls = []

        def backend(prompt, attempt):
            calls.append(attempt)
            return ([self.raw_first, self.second, {"repairs": []}][attempt - 1])

        result = run_round_validator_with_retry(
            backend_call=backend,
            validator_input=self.validator_input,
            realized_scenes=self.realized,
            review_world_view=self.review,
            local_gate_results=self.gates,
        )
        self.assertFalse(result["completed"])
        self.assertEqual(calls, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
