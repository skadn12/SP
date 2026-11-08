"""Meaningful no-network regressions for retained primaries and one repair."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt, build_repair_prompt
from ..h4_r02_feedback_transfer_p2_2_v1.test_generation_hooks import (
    FakeClient, FakeLedger, batch_fixture, expanded_fixture,
)
from ..h4_r02_ordering_temporal_contract_v1.test_runtime import synthetic_candidates
from .admission import admit_stage, normalize_stage, repair_required, verify_retained_candidate
from .sources import memory_only_envelope, projection_sha256

POLICY = {"version": "ordering_temporal_contract_v1", "node_count": 4}
CONTRACT = {"intent_id": "intent_1", "required_evidence_groups": {}}


class PartialAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expanded = expanded_fixture()
        self.context = {"run_id": "unit", "arm_id": "test"}
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("NETWORK_FORBIDDEN"))
        guard.start()
        self.addCleanup(guard.stop)

    def batch(self):
        return synthetic_candidates(build_prompt(self.expanded, CONTRACT, ordering_policy=POLICY)["prompt"])

    def execute(self, replies, policy="partial_v1"):
        self.client, self.ledger = FakeClient(replies), FakeLedger()
        return generation.execute_writer_batch(expanded=self.expanded, contract=CONTRACT,
            root_dir=self.root, root_id="test-root", ledger=self.ledger, client=self.client,
            ordering_policy=POLICY, ordering_context=self.context, admission_policy=policy)

    def finalize(self, result):
        return generation.finalize_writer_candidates(root_dir=self.root, root_id="test-root",
            candidates=result["candidates"], generation_records=result["generation_records"],
            expanded=self.expanded, ordering_policy=POLICY, ordering_context=self.context,
            admission_policy="partial_v1")

    def test_bad_ordering_retains_independent_scms_even_when_repair_fails(self):
        primary = self.batch()
        primary["candidates"][-1]["temporal_relations"] = []
        result = self.execute([primary, {"candidates": []}])
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual([row["question_type"] for row in result["candidates"]],
                         ["single_choice", "single_choice", "multi_select"])
        self.assertTrue(all(row["admission_stage"] == "primary" for row in result["candidates"]))
        self.assertEqual(self.finalize(result)["valid_candidate_count"], 3)
        expected = build_repair_prompt(self.expanded, CONTRACT, primary, "contract_guided", ordering_policy=POLICY)
        self.assertEqual(self.client.calls[1]["prompt"], expected["prompt"])

    def test_repair_fills_only_ordering_and_preserves_primary(self):
        primary, repair = self.batch(), self.batch()
        primary["candidates"][-1]["temporal_relations"] = []
        for row in repair["candidates"]:
            row["question"] = "repair " + row["question"]
        result = self.execute([primary, repair])
        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual([row["admission_stage"] for row in result["candidates"]],
                         ["primary"] * 3 + ["repair"])
        self.assertEqual(self.finalize(result)["valid_candidate_count"], 4)
        altered = {**result["candidates"][-1], "admission_stage": "primary"}
        with self.assertRaisesRegex(ValueError, "IDENTITY"):
            verify_retained_candidate(altered, expanded=self.expanded, policy=POLICY,
                context={**self.context, "root_dir": self.root, "root_id": "test-root"})

    def test_final_validator_rejection_triggers_repair_before_counting(self):
        primary, repair = self.batch(), self.batch()
        primary["candidates"][2]["gold"] = ["A"]
        result = self.execute([primary, repair])
        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual(next(r for r in result["candidates"] if r["question_type"] == "multi_select")["admission_stage"], "repair")

    def test_stage_alias_keeps_same_raw_ordinal_independent(self):
        primary, repair = self.batch(), self.batch()
        primary["candidates"][-1]["temporal_relations"] = []
        repair["candidates"] = [repair["candidates"][-1]]
        result = self.execute([primary, repair])
        self.assertEqual(result["candidates"][0]["raw_candidate_id"], result["candidates"][-1]["raw_candidate_id"])
        self.assertNotEqual(result["candidates"][0]["candidate_id"], result["candidates"][-1]["candidate_id"])
        self.assertEqual(self.finalize(result)["valid_candidate_count"], 4)

    def test_local_malformed_values_do_not_delete_other_rows(self):
        for field, bad in (("options", [1, 2]), ("gold", 123), ("evidence_refs", 42)):
            with self.subTest(field=field):
                self.root = Path(self.temp.name) / field
                primary = self.batch()
                primary["candidates"][2][field] = bad
                result = self.execute([primary, {"candidates": []}])
                self.assertEqual(len(result["candidates"]), 3)
                self.assertEqual({r["question_type"] for r in result["candidates"]}, {"single_choice", "ordering"})

    def test_strict_duplicate_json_keys_are_response_failure(self):
        raw = json.dumps(self.batch())[:-1] + ', "candidates": []}'
        result = self.execute([raw, {"candidates": []}])
        self.assertEqual(result["candidates"], [])
        self.assertTrue(result["generation_records"][0]["validation"]["response_failure"])

    def test_root_identity_failure_is_response_level(self):
        primary = {**self.batch(), "root_id": "other-root"}
        result = self.execute([primary, {"candidates": []}])
        self.assertEqual(result["candidates"], [])
        self.assertIn("response_root_identity_mismatch", result["generation_records"][0]["validation"]["errors"])

    def test_duplicates_do_not_fill_quota_and_extra_rows_are_capped(self):
        primary, repair = self.batch(), self.batch()
        primary["candidates"][1] = copy.deepcopy(primary["candidates"][0])
        repair["candidates"].append(copy.deepcopy(repair["candidates"][0]))
        result = self.execute([primary, repair])
        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual(sum(r["admission_stage"] == "repair" for r in result["candidates"]), 1)
        self.assertEqual(len({r["candidate_id"] for r in result["candidates"]}), 4)

    def test_legacy_policy_still_discards_failed_batch(self):
        primary = self.batch()
        primary["candidates"][-1]["temporal_relations"] = []
        result = self.execute([primary, {"candidates": []}], policy=None)
        self.assertEqual(result["candidates"], [])

    def test_memory_legal_shortage_abstains_but_malformed_row_repairs(self):
        items = [{"memory_ref": "MEM_ITEM_001", "bucket_type": "event", "key": "item", "content": "A fact."}]
        expanded = memory_only_envelope("test-root", items, snapshot_sha256="a"*64,
                                       projection_sha256=projection_sha256(items))
        raw = {"candidates": [], "abstentions": [{"question_type": kind,
               "status": "INSUFFICIENT_EVIDENCE", "reason": "No temporal/answer basis."}
               for kind in ("single_choice", "multi_select", "ordering")]}
        normalized = normalize_stage(raw, intent_id="memory_only_mixed", source="contract_guided",
            contract={}, expanded=expanded, root_id="test-root", stage="primary", request_id="test")
        _, acceptance = admit_stage(normalized)
        self.assertFalse(repair_required(raw, normalized, acceptance, expanded=expanded))
        self.assertFalse(repair_required({"candidates": []}, normalized, acceptance, expanded=expanded))
        self.assertEqual(acceptance["abstention_reason"], "UNSPECIFIED_EMPTY_RESPONSE")
        raw["candidates"] = ["not a candidate"]
        normalized = normalize_stage(raw, intent_id="memory_only_mixed", source="contract_guided",
            contract={}, expanded=expanded, root_id="test-root", stage="primary", request_id="test")
        self.assertTrue(repair_required(raw, normalized, acceptance, expanded=expanded))


if __name__ == "__main__":
    unittest.main()
