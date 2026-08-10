from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.frontier_k5_replay import (
    merge_incremental_k5_result,
)


def _record(reward: int, slot: int) -> dict[str, object]:
    return {"request_id": f"r{slot}", "local_reward": reward,
            "parsed_letter": "A", "sampling_seed": slot}


def _existing(*rewards: int) -> dict[str, object]:
    return {
        "qa_id": "qa-1",
        "solver_ready": True,
        "grounding": {"decision": "pass_grounding"},
        "full_context_rollouts": [
            _record(reward, slot) for slot, reward in enumerate(rewards)],
        "evidence_ready": True,
        "semantic_evidence_audit_executed": False,
        "admitted": False,
    }


class MemoraxFrontierK5ReplayTest(unittest.TestCase):
    def test_four_of_five_is_training_frontier(self) -> None:
        result = merge_incremental_k5_result(
            _existing(1, 1, 1), [_record(1, 3), _record(0, 4)])
        self.assertEqual("4/5", result["full_context_bucket"])
        self.assertEqual("admit_frontier_k5", result["decision"])
        self.assertTrue(result["admitted"])
        self.assertTrue(result["training_ready"])

    def test_zero_and_five_of_five_are_rejected(self) -> None:
        cases = [
            ((_existing(0, 0, 0), [_record(0, 3), _record(0, 4)]),
             "0/5", "reject_all_wrong"),
            ((_existing(1, 1, 1), [_record(1, 3), _record(1, 4)]),
             "5/5", "reject_easy"),
        ]
        for (existing, new), bucket, decision in cases:
            with self.subTest(bucket=bucket):
                result = merge_incremental_k5_result(existing, new)
                self.assertEqual(bucket, result["full_context_bucket"])
                self.assertEqual(decision, result["decision"])
                self.assertFalse(result["training_ready"])

    def test_no_context_rejection_never_runs_full_context(self) -> None:
        existing = {
            "qa_id": "qa-leak",
            "grounding": {"decision": "bad_ground_semantic_leakage"},
            "admitted": False,
        }
        result = merge_incremental_k5_result(existing, [])
        self.assertEqual(
            "bad_ground_semantic_leakage", result["frontier_replay_decision"])
        self.assertEqual(0, result["new_full_context_request_count"])
        self.assertFalse(result["training_ready"])

    def test_no_context_rejection_cannot_receive_new_rollout(self) -> None:
        with self.assertRaisesRegex(
                ValueError, "new_full_context_rollout_for_grounding_rejection"):
            merge_incremental_k5_result({
                "grounding": {"decision": "bad_ground_semantic_leakage"}},
                [_record(0, 3)])

    def test_evidence_mismatch_is_marked_not_blocking(self) -> None:
        existing = _existing(1, 1, 0)
        existing["evidence_ready"] = False
        result = merge_incremental_k5_result(
            existing, [_record(0, 3), _record(0, 4)])
        self.assertEqual("2/5", result["full_context_bucket"])
        self.assertTrue(result["training_ready"])
        self.assertEqual(
            "evidence_mismatch_marked", result["evidence_semantic_status"])


if __name__ == "__main__":
    unittest.main()
