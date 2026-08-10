import hashlib
import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_canary_v11 import (
    execution_progress_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    ROUND_VALIDATOR_INPUT_VERSION,
    audit_round_validator_prompt_budget,
    build_round_validator_input,
)


CANARY_2 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_2"
)
CANARY_2_TREE_SHA = (
    "555d657cdf4233e80105cd2de762a9a54c30c777bec085494917557fabae1294"
)


def load(relative):
    return json.loads((CANARY_2 / relative).read_text(encoding="utf-8"))


class RoundValidatorInputCompactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = load("planner/compiled_round_plan.json")
        cls.review = load("review_world_view.json")
        cls.old_input = load("round_validator_input.json")
        cls.realized = cls.old_input["realized_scenes"]
        cls.dag = load("scene_dependency_dag.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.contexts = load("scene_contexts.json")
        cls.compact = build_round_validator_input(
            round_blueprint=cls.compiled,
            review_world_view=cls.review,
            realized_scenes=cls.realized,
            scene_dependency_dag=cls.dag,
            structural_gate_results=cls.gates,
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/root/autodl-tmp/Qwen3-4B", local_files_only=True
        )
        cls.budget = audit_round_validator_prompt_budget(
            cls.compact,
            token_counter=lambda text: len(tokenizer.encode(
                text, add_special_tokens=False)),
        )

    def test_old_duplicate_containers_are_removed(self):
        self.assertEqual(self.compact["round_validator_input_version"],
                         ROUND_VALIDATOR_INPUT_VERSION)
        self.assertNotIn("round_blueprint", self.compact)
        self.assertNotIn("review_world_view", self.compact)
        self.assertNotIn("realized_scenes", self.compact)
        self.assertEqual(len(self.compact["scenes"]), 7)

    def test_dialogue_is_preserved_exactly_once(self):
        expected = {row["scene_ref"]: row["dialogue_turns"] for row in self.realized}
        actual = {row["scene_ref"]: row["dialogue_turns"]
                  for row in self.compact["scenes"]}
        self.assertEqual(actual, expected)
        self.assertNotIn("realized_scenes", self.compact)

    def test_historical_turns_are_deduplicated_without_text_loss(self):
        expected = {}
        for anchor in self.review["historical_anchors"]:
            for turn in anchor["exact_excerpt"]:
                expected[turn["turn_ref"]] = {
                    key: turn[key] for key in (
                        "turn_ref", "speaker_ref", "speaker_label", "text")
                }
        actual = {row["turn_ref"]: row for row in self.compact["historical_turns"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 60)
        self.assertEqual(len(self.compact["historical_anchor_index"]), 6)

    def test_action_contracts_are_shared_by_type(self):
        contracts = self.compact["action_contracts_by_type"]
        self.assertTrue(contracts)
        self.assertTrue(all("validator_checks" in row for row in contracts.values()))
        for scene in self.compact["scenes"]:
            for action in (scene["primary_action"], scene["secondary_action"]):
                if action is not None:
                    self.assertNotIn("action_contract", action)

    def test_semantic_plan_and_knowledge_boundaries_remain(self):
        for compact_scene, original in zip(self.compact["scenes"], self.realized):
            for key in ("source_anchor_id", "placement_choice_id", "link_type",
                        "bridge_reason", "new_development", "goal",
                        "linked_existing_refs", "relationship_dynamic",
                        "interaction_beats", "branch_role",
                        "merge_target_scene_ref",
                        "desired_outcome"):
                self.assertEqual(compact_scene[key], original[key])
            for compact_participant, original_participant in zip(
                    compact_scene["speaker_participants"],
                    original["speaker_participants"]):
                self.assertEqual(compact_participant.get("scene_role"),
                                 original_participant.get("scene_role"))
        self.assertEqual(
            [row["scene_ref"] for row in self.compact["narrative_constraints"]],
            [row["scene_ref"] for row in self.review["narrative_constraints"]],
        )
        self.assertTrue(all("preceding_known_context_refs" not in row
                            for row in self.compact["narrative_constraints"]))
        self.assertTrue(all(row["following_context_role"] == "constraint_only"
                            for row in self.compact["narrative_constraints"]))

    def test_canary_2_shadow_prompt_fits_with_safety_margin(self):
        self.assertTrue(self.budget["valid"])
        self.assertLessEqual(self.budget["total_reserved_tokens"], 40_960)
        # This is free space after reserving the full 8,192-token response and
        # a separate 1,024-token safety margin.
        self.assertGreaterEqual(self.budget["remaining_tokens"], 4_000)
        self.assertLess(self.budget["prompt_tokens"], 32_768)

    def test_canary_2_payload_is_materially_smaller(self):
        old_bytes = len(json.dumps(
            self.old_input, ensure_ascii=False, separators=(",", ":")).encode())
        new_bytes = len(json.dumps(
            self.compact, ensure_ascii=False, separators=(",", ":")).encode())
        self.assertLess(new_bytes, old_bytes // 4)

    def test_frozen_input_sha_is_deterministic(self):
        rebuilt = build_round_validator_input(
            round_blueprint=self.compiled,
            review_world_view=self.review,
            realized_scenes=self.realized,
            scene_dependency_dag=self.dag,
            structural_gate_results=self.gates,
        )
        self.assertEqual(rebuilt, self.compact)
        value = dict(self.compact)
        actual = value.pop("frozen_round_validator_input_sha256")
        expected = hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(actual, expected)

    def test_failure_progress_retains_realized_artifacts(self):
        progress = execution_progress_v11({
            "compiled_round_plan": self.compiled,
            "realized": self.realized,
            "contexts": self.contexts,
        })
        self.assertEqual(progress["planned_scene_count"], 7)
        self.assertEqual(progress["realized_scene_count"], 7)
        self.assertEqual(progress["realized_turn_count"], 38)
        self.assertTrue(progress["parent_tail_retained"])
        self.assertTrue(progress["same_round_text_included"])

    def test_historical_canary_is_unchanged(self):
        self.assertEqual(hash_tree(CANARY_2)["tree_sha256"], CANARY_2_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
