from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
    production_batch_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.batch_cost_control_v11 import (
    BatchRequestBudget,
    PaidRequestBudgetExhausted,
    candidate_terminal_status,
    stable_error_signature,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_realizer_retry_prompt_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    classify_dialogue_only_evidence_alignment,
)


class BatchCostControlV11Test(unittest.TestCase):
    def test_root_budget_rejects_before_next_request(self):
        budget = BatchRequestBudget(
            root_ids=("r1", "r2"), per_root_paid_limit=2,
            batch_paid_limit=4)
        budget.reserve(root_id="r1", stage="planner", item_ref="round",
                       is_retry=False)
        budget.reserve(root_id="r1", stage="planner", item_ref="round",
                       is_retry=True)
        with self.assertRaisesRegex(
                PaidRequestBudgetExhausted,
                "budget_exhausted_before_realizer:root:r1"):
            budget.reserve(root_id="r1", stage="realizer", item_ref="s1")
        self.assertEqual(2, budget.paid_request_count)
        self.assertEqual(1, budget.retry_request_count)

    def test_batch_budget_does_not_let_one_more_request_escape(self):
        budget = BatchRequestBudget(
            root_ids=("r1", "r2"), per_root_paid_limit=3,
            batch_paid_limit=2)
        budget.reserve(root_id="r1", stage="planner", item_ref="round")
        budget.reserve(root_id="r2", stage="planner", item_ref="round")
        with self.assertRaisesRegex(
                PaidRequestBudgetExhausted,
                "budget_exhausted_before_realizer:batch:r2"):
            budget.reserve(root_id="r2", stage="realizer", item_ref="s1")

    def test_stable_error_signature_removes_volatile_refs(self):
        left = stable_error_signature(
            stage="realizer", item_ref="scene",
            errors=("R01:S02:T009:turn_count_outside_frozen_budget",))
        right = stable_error_signature(
            stage="realizer", item_ref="scene",
            errors=("R01:S02:T010:turn_count_outside_frozen_budget",))
        self.assertEqual(left, right)

    def test_candidate_limit_and_early_stop_are_frozen(self):
        self.assertEqual(3, production_batch_v11.QA_CANDIDATE_LIMIT)
        self.assertTrue(production_batch_v11.STOP_ON_FIRST_ADMITTED)
        self.assertTrue(production_batch_v11.candidate_generation_should_stop(
            {"status": "admitted"}))
        self.assertFalse(production_batch_v11.candidate_generation_should_stop(
            {"status": "mcq_materialized"}))

    def test_candidate_terminal_states_are_explicit(self):
        self.assertEqual("admitted", candidate_terminal_status(
            stage="spell_filter", admitted=True))
        self.assertEqual("position_sensitive", candidate_terminal_status(
            stage="spell_filter", final_decision="no_context_position_bias"))
        self.assertEqual("solver_too_easy", candidate_terminal_status(
            stage="spell_filter", final_decision="bad_ground_semantic_leakage"))
        self.assertEqual("rejected_semantic", candidate_terminal_status(
            stage="semantic_audit", final_decision="reject"))

    def test_compact_realizer_retry_omits_full_base_prompt(self):
        prompt = build_realizer_retry_prompt_v11(
            base_prompt="FULL_BASE_PROMPT_MUST_NOT_APPEAR",
            frozen_input_sha256="a" * 64,
            errors=("turn_count_outside_frozen_budget",),
            turn_budget={"minimum": 5, "maximum": 6},
            retry_attempt=2, retry_limit=2,
            forbidden_utterances=("copied predecessor line",),
            allowed_speaker_refs=("E1", "E2"),
            compact_retry_context={
                "scene_plan": {"scene_ref": "R01:S02"},
                "predecessor_dialogue_read_only": ["context"],
            })
        self.assertNotIn("FULL_BASE_PROMPT_MUST_NOT_APPEAR", prompt)
        self.assertIn("read-only context", prompt)
        self.assertIn("copied predecessor line", prompt)
        self.assertIn(
            '{"dialogue_turns":[{"speaker_ref":"one exact allowed speaker ref",'
            '"text":"non-empty natural utterance"}]}', prompt)
        self.assertIn("contain exactly speaker_ref and text", prompt)
        self.assertIn("Do not output turn_ref, scene_ref", prompt)

    def test_dialogue_only_subset_mismatch_can_be_degraded(self):
        audit = classify_dialogue_only_evidence_alignment({
            "valid": False,
            "errors": [
                "scene_review:1:primary_action_realization:"
                "progression_evidence_not_action_evidence",
                "scene_review:2:dependency_context_source_not_declared",
            ],
        })
        self.assertTrue(audit["admissible"])
        self.assertEqual("degraded_evidence_alignment",
                         audit["dialogue_validation_status"])
        self.assertEqual("skipped_evidence_mismatch",
                         audit["world_update_status"])

    def test_empty_action_evidence_remains_hard_failure(self):
        audit = classify_dialogue_only_evidence_alignment({
            "valid": False,
            "errors": [
                "scene_review:1:secondary_action_realization:evidence_turn_refs",
                "scene_review:1:secondary_action_realization:evidence_required",
                "scene_review:2:primary_action_realization:"
                "continuation_progression_evidence_required",
            ],
        })
        self.assertFalse(audit["admissible"])
        self.assertEqual("rejected", audit["dialogue_validation_status"])

    def test_wrong_scene_or_unknown_refs_remain_hard_failure(self):
        for error in (
            "scene_review:1:current_scene_supporting_turn_refs",
            "scene_review:1:unknown_scene_alias:s99",
            "validator_scene_review_coverage",
            "scene_review:2:uses_scene_refs",
        ):
            with self.subTest(error=error):
                self.assertFalse(classify_dialogue_only_evidence_alignment({
                    "valid": False, "errors": [error]})["admissible"])


if __name__ == "__main__":
    unittest.main()
