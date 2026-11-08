"""No-network regressions for explicit ordering abstention across one repair."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_repair_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import (
    FakeClient, FakeLedger, batch_fixture, expanded_fixture,
)
from .admission import GROUNDED_VERSION, normalize_stage, admit_stage, repair_required


ORDERING_POLICY = {"version": "ordering_grounded_options_v2", "node_count": 4}
CONTRACT = {"intent_id": "intent_1", "required_evidence_groups": {}}
ABSTENTION = {"question_type": "ordering", "status": "INSUFFICIENT_EVIDENCE",
              "reason": "The text does not determine four distinct events in one strict order."}


class GroundedAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expanded = expanded_fixture()
        self.context = {"run_id": "unit", "arm_id": "grounded"}
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("NETWORK_FORBIDDEN"))
        guard.start()
        self.addCleanup(guard.stop)

    def batch(self, *, abstain=True):
        raw = batch_fixture(counts=(2, 1, 0))
        if abstain:
            raw["abstentions"] = [copy.deepcopy(ABSTENTION)]
        return raw

    def execute(self, replies, **kwargs):
        self.client, self.ledger = FakeClient(replies), FakeLedger()
        return generation.execute_writer_batch(expanded=self.expanded, contract=CONTRACT,
            root_dir=self.root, root_id="test-root", ledger=self.ledger, client=self.client,
            ordering_policy=ORDERING_POLICY, ordering_context=self.context,
            admission_policy=GROUNDED_VERSION, **kwargs)

    def finalize(self, result):
        return generation.finalize_writer_candidates(root_dir=self.root, root_id="test-root",
            candidates=result["candidates"], generation_records=result["generation_records"],
            expanded=self.expanded, ordering_policy=ORDERING_POLICY, ordering_context=self.context,
            admission_policy=GROUNDED_VERSION)

    def test_legal_ordering_abstention_retains_scms_without_repair(self):
        result = self.execute([self.batch()])
        self.assertEqual(len(self.client.calls), 1)
        self.assertFalse(result["repair_requested"])
        self.assertEqual(len(result["candidates"]), 3)
        audit = result["admission_summary"]
        self.assertTrue(audit["accepted"])
        self.assertFalse(audit["full_quota_met"])
        self.assertEqual(audit["valid_counts"].get("ordering", 0), 0)
        self.assertEqual(audit["abstained_minimums"], {"ordering": 1})
        self.assertEqual(audit["unresolved_minimums"], {})
        self.assertEqual(audit["semantic_support_status"], "NOT_ASSESSED")
        self.assertEqual(self.finalize(result)["valid_candidate_count"], 3)

    def test_omission_is_not_abstention(self):
        result = self.execute([self.batch(abstain=False), self.batch(abstain=False)])
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(result["admission_summary"]["unresolved_minimums"], {"ordering": 1})
        self.assertEqual(result["admission_summary"]["effective_abstentions"], [])
        self.assertEqual(len(result["candidates"]), 3)

    def test_empty_response_does_not_waive_scms_or_ordering(self):
        result = self.execute([{"candidates": []}, {"candidates": []}])
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(result["admission_summary"]["unresolved_minimums"],
                         {"single_choice": 2, "multi_select": 1, "ordering": 1})

    def test_abstention_does_not_hide_emitted_invalid_ordering(self):
        primary = self.batch()
        primary["candidates"].append({"question_type": "ordering", "event_nodes": []})
        result = self.execute([primary, self.batch()])
        self.assertEqual(len(self.client.calls), 2)
        audit = result["generation_records"][0]["validation"]
        self.assertEqual(audit["effective_abstentions"], [])
        self.assertEqual(audit["blocked_abstentions"], [ABSTENTION])
        self.assertEqual(audit["row_checks"][-1]["decision"], "REJECT")
        self.assertTrue(all(row["admission_stage"] == "primary" for row in result["candidates"]))
        self.assertEqual(result["admission_summary"]["effective_abstentions"], [ABSTENTION])

    def test_scms_repair_preserves_abstention_and_rejects_new_ordering(self):
        primary, repair = self.batch(), self.batch(abstain=False)
        primary["candidates"][2]["gold"] = ["A"]  # Existing validator rejects one-answer MS.
        repair["candidates"].append({"question_type": "ordering", "event_nodes": []})
        result = self.execute([primary, repair])
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual([row["admission_stage"] for row in result["candidates"]],
                         ["primary", "primary", "repair"])
        expected = build_repair_prompt(self.expanded, CONTRACT, primary, "contract_guided",
            ordering_policy=ORDERING_POLICY, primary_abstentions=[ABSTENTION])
        self.assertEqual(self.client.calls[1]["prompt"], expected["prompt"])
        repaired = result["generation_records"][1]["validation"]
        self.assertEqual(repaired["effective_abstentions"], [ABSTENTION])
        self.assertIn("candidate_conflicts_with_primary_abstention", str(repaired["row_checks"][-1]))
        self.assertEqual(self.finalize(result)["valid_candidate_count"], 3)

    def test_primary_abstention_survives_malformed_repair(self):
        primary = self.batch()
        primary["candidates"] = primary["candidates"][:2]
        result = self.execute([primary, "{broken JSON"])
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["admission_summary"]["effective_abstentions"], [ABSTENTION])
        self.assertEqual(result["admission_summary"]["unresolved_minimums"], {"multi_select": 1})

    def test_malformed_json_is_not_salvaged_as_abstention(self):
        primary = json.dumps(self.batch())[:-1] + ', "candidates": []}'
        result = self.execute([primary, {"candidates": []}])
        audit = result["generation_records"][0]["validation"]
        self.assertTrue(audit["response_failure"])
        self.assertEqual(audit["effective_abstentions"], [])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(len(self.client.calls), 2)

    def test_invalid_abstention_is_not_forwarded_as_primary_permission(self):
        for changes in ({"question_type": "single_choice"}, {"reason": " "},
                        {"status": "UNKNOWN"}, {"extra": True}):
            with self.subTest(changes=changes):
                self.root = Path(self.temp.name) / str(len(list(Path(self.temp.name).iterdir())))
                primary = self.batch()
                primary["abstentions"][0].update(changes)
                result = self.execute([primary, {"candidates": []}])
                audit = result["generation_records"][0]["validation"]
                self.assertTrue(audit["response_failure"])
                self.assertEqual(audit["effective_abstentions"], [])
                self.assertEqual(result["admission_summary"]["effective_abstentions"], [])

    def test_malformed_row_still_triggers_repair_alongside_legal_abstention(self):
        primary = self.batch()
        primary["candidates"].append("broken row")
        result = self.execute([primary, {"candidates": []}])
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(result["admission_summary"]["effective_abstentions"], [ABSTENTION])

    def test_abstention_reason_is_frozen_from_primary(self):
        primary, repair = self.batch(), self.batch()
        primary["candidates"] = primary["candidates"][:2]
        repair["abstentions"][0]["reason"] = "A changed explanation."
        result = self.execute([primary, repair])
        self.assertEqual(result["admission_summary"]["effective_abstentions"], [ABSTENTION])

    def test_policy_mismatch_stops_before_any_request(self):
        client, ledger = FakeClient([]), FakeLedger()
        with self.assertRaisesRegex(ValueError, "POLICY_MISMATCH"):
            generation.execute_writer_batch(expanded=self.expanded, contract=CONTRACT,
                root_dir=self.root, root_id="test-root", ledger=ledger, client=client,
                ordering_policy=ORDERING_POLICY, ordering_context=self.context,
                admission_policy="partial_v1")
        self.assertEqual(client.calls, [])
        self.assertEqual(ledger.calls, [])

    def test_v1_transcript_abstention_behavior_unchanged(self):
        normalized = normalize_stage(self.batch(), intent_id="intent_1", source="contract_guided",
            contract=CONTRACT, expanded=self.expanded, root_id="test-root", stage="primary",
            request_id="unit")
        self.assertTrue(normalized["response_failure"])
        self.assertIn("abstentions_not_allowed_or_invalid", normalized["errors"])
        _, acceptance = admit_stage(normalized)
        self.assertTrue(repair_required(self.batch(), normalized, acceptance, expanded=self.expanded))

    def test_legacy_miner_free_route_rejects_grounded_policy_before_mining(self):
        client, ledger = FakeClient([]), FakeLedger()
        with self.assertRaisesRegex(ValueError, "EXPLICIT_PARTIAL_WRITER_ENTRY"):
            generation.generate_root_qa(expanded=self.expanded, root_dir=self.root,
                root_id="test-root", api_config=Path("/nonexistent"), ledger=ledger, client=client,
                ordering_policy=ORDERING_POLICY, ordering_context=self.context)
        self.assertEqual(client.calls, [])
        self.assertEqual(ledger.calls, [])


if __name__ == "__main__":
    unittest.main()
