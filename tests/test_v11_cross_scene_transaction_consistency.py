import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.append_only_memory import (
    evaluate_scene_commit,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_canary_v11 import (
    _apply_durable_updates,
    _audit_transaction_memory_consistency,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY_13 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_13"
)
CANARY_13_TREE_SHA = (
    "a60fba7a5e6c9e99b6857f19912d863e867bcc1c060e5bd910bcdb63985ea63c"
)


def load(relative):
    return json.loads((CANARY_13 / relative).read_text(encoding="utf-8"))


class CrossSceneTransactionConsistencyTests(unittest.TestCase):
    def test_scene_commit_gate_accepts_already_committed_dependency(self):
        proposal = {
            "proposal_ref": "R01:S02:P01",
            "proposal_dependencies": ["R01:S01:P01"],
        }
        claims = [{"proposal_ref": "R01:S02:P01",
                   "realization_status": "realized"}]
        review = {
            "proposal_reviews": [{
                "proposal_ref": "R01:S02:P01", "review_decision": "confirmed",
                "verification_status": "semantically_verified"}],
            "scope_status": "clean", "world_consistency_status": "clean",
        }
        blocked = evaluate_scene_commit(
            compiled_proposals=[proposal], proposal_claims=claims,
            committer_output=review)
        accepted = evaluate_scene_commit(
            compiled_proposals=[proposal], proposal_claims=claims,
            committer_output=review,
            previously_committed_proposal_refs=["R01:S01:P01"])
        self.assertFalse(blocked["commit_ready"])
        self.assertTrue(accepted["commit_ready"])

    def test_canary13_two_scene_dependency_materializes_both_allocations(self):
        validator_input = load("round_validator_input.json")
        generated = {row["scene_ref"]: row for row in validator_input["scenes"]}
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            memory, transactions = _apply_durable_updates(
                run_root=run_root, memory=load("world_memory_initial.json"),
                compiled_plan=load("planner/compiled_round_plan.json"),
                generated=generated,
                accepted_updates=load("durable_update_compile_result.json")[
                    "accepted_updates"])
            journal = [json.loads(line) for line in
                       (run_root / "run_commit_journal.jsonl").read_text().splitlines()]
            audit = _audit_transaction_memory_consistency(
                memory=memory, transactions=transactions, journal_rows=journal)
            generated_records = [row for row in memory["authoritative_records"]
                                 if row.get("origin") == "generated_extension"]
            self.assertEqual(2, len(transactions))
            self.assertEqual(2, len(generated_records))
            self.assertEqual(62, memory["high_water_marks"]["memory"])
            self.assertEqual(
                {"R01:S01:PDU01", "R01:S02:PDU02"},
                {row["proposal_ref"] for row in generated_records})
            self.assertTrue(audit["valid"], audit)
            self.assertEqual(2, audit["prepared_count"])
            self.assertEqual(2, audit["committed_count"])

    def test_canary13_history_is_unchanged(self):
        self.assertEqual(CANARY_13_TREE_SHA, hash_tree(CANARY_13)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
