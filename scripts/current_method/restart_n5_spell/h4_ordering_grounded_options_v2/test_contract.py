"""No-network checks of evidence authenticity, one direction, and provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ..core import atomic_text, canonical_json, sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt, build_repair_prompt
from ..h4_r02_ordering_temporal_contract_v1.contract import (
    derive_ordering, policy_config, register_runtime_context, verify_candidate,
)
from .contract import VERSION
from .guidance import synthetic_example

POLICY = {"version": VERSION, "node_count": 4}
IDENTITY = {"run_id": "run", "arm_id": "arm", "root_id": "root",
            "intent_id": "intent_1", "source": "contract_guided", "stage": "primary",
            "request_id": "req", "raw_row_index": 1,
            "candidate_id": "root:intent_1-C-Q01", "raw_sha256": "raw"}


class GroundedContractTests(unittest.TestCase):
    def setUp(self):
        fixture = synthetic_example()
        self.expanded = fixture["expanded"]
        self.row = fixture["raw"]["candidates"][0]
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("NETWORK_FORBIDDEN"))
        guard.start()
        self.addCleanup(guard.stop)

    def derive(self, row=None, expanded=None):
        return derive_ordering(self.row if row is None else row,
            expanded=self.expanded if expanded is None else expanded, policy=POLICY, identity=IDENTITY)

    def test_only_one_direction_and_program_gold(self):
        result = self.derive()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["projected_row"]["gold"], ["I2", "I4", "I1", "I3"])
        self.assertEqual(result["record"]["raw_row"], self.row)
        first = result["record"]["program_temporal_relations"][0]
        self.assertEqual((first["before"], first["after"]), ("I2", "I4"))
        self.assertIn("I2 precedes I4", first["temporal_basis"])
        self.assertEqual(result["record"]["semantic_support_status"], "NOT_ASSESSED")

    def test_legacy_fields_cannot_override_pair(self):
        for field, value in (("before", "I4"), ("after", "I2"), ("temporal_basis", "reverse")):
            row = copy.deepcopy(self.row)
            row["temporal_relations"][0][field] = value
            self.assertIn("ORDERING_SCHEMA_INVALID", self.derive(row)["errors"])
        row = copy.deepcopy(self.row)
        row["ordering_contract_version"] = "ordering_temporal_contract_v1"
        self.assertIn("ORDERING_CONTRACT_VERSION_MISMATCH", self.derive(row)["errors"])
        for count in (3, 5, 6):
            with self.assertRaises(ValueError):
                policy_config({"version": VERSION, "node_count": count})

    def test_quote_authenticity_and_complete_reference_coverage(self):
        for target in ("event_nodes", "temporal_relations"):
            row = copy.deepcopy(self.row)
            row[target][0]["evidence_spans"][0]["quote"] = "I have already submitted the revised proposal."
            self.assertIn("ORDERING_QUOTE_NOT_EXACT_SOURCE_SUBSTRING", self.derive(row)["errors"])
            row = copy.deepcopy(self.row)
            row[target][0]["evidence_spans"][0]["turn_ref"] = "other-root:turn"
            result = self.derive(row)
            self.assertIn("ORDERING_SPAN_REF_NOT_DECLARED", result["errors"])
            self.assertIn("ORDERING_QUOTE_REFERENCE_COVERAGE_MISMATCH", result["errors"])
        row = copy.deepcopy(self.row)
        row["event_nodes"][0]["evidence_spans"] = []
        self.assertIn("ORDERING_SCHEMA_INVALID", self.derive(row)["errors"])

    def test_multiple_spans_for_context_and_no_semantic_certification(self):
        row = copy.deepcopy(self.row)
        node = row["event_nodes"][0]
        other = self.expanded["turns"][0]
        node["evidence_refs"].append(other["turn_ref"])
        node["evidence_spans"].append({"turn_ref": other["turn_ref"], "quote": other["text"]})
        self.assertEqual(self.derive(row)["status"], "PASS")
        # Exact quotations alone cannot determine actor/state entailment. Do
        # not add a fake semantic PASS or a keyword heuristic for these cases.
        node.update(actor="Unsupported actor", event_state="planned",
                    item_text="An unsupported event claim")
        result = self.derive(row)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["record"]["semantic_support_status"], "NOT_ASSESSED")

    def test_cycles_and_nonunique_order_are_still_rejected(self):
        row = copy.deepcopy(self.row)
        row["temporal_relations"] = row["temporal_relations"][:1]
        self.assertIn("ORDER_NOT_DETERMINED_BY_CONSTRAINTS", self.derive(row)["errors"])
        row = copy.deepcopy(self.row)
        reverse = copy.deepcopy(row["temporal_relations"][0])
        reverse["ordered_pair"].reverse()
        row["temporal_relations"].append(reverse)
        self.assertIn("ORDERING_GRAPH_CYCLE", self.derive(row)["errors"])

    def test_date_anchor_must_be_in_quoted_segment(self):
        expanded = copy.deepcopy(self.expanded)
        expanded["conversation_date_segments"] = [
            {"segment_id": f"S{i}", "conversation_date": f"2026-06-0{i}",
             "first_turn_index_1_based": i, "last_turn_index_1_based": i,
             "first_turn_ref": turn["turn_ref"], "last_turn_ref": turn["turn_ref"]}
            for i, turn in enumerate(expanded["turns"], 1)]
        row = copy.deepcopy(self.row)
        # I1 cites T3; a date merely occurring elsewhere is insufficient.
        row["event_nodes"][0]["date_anchors"] = [
            {"segment_id": "S1", "conversation_date": "2026-06-01"}]
        self.assertIn("ORDERING_DATE_ANCHOR_OUTSIDE_EVIDENCE", self.derive(row, expanded)["errors"])
        row["event_nodes"][0]["date_anchors"] = [
            {"segment_id": "S3", "conversation_date": "2026-06-03"}]
        self.assertEqual(self.derive(row, expanded)["status"], "PASS")
        row["event_nodes"][0]["date_anchors"][0]["conversation_date"] = "2026-07-03"
        self.assertIn("ORDERING_DATE_ANCHOR_UNKNOWN", self.derive(row, expanded)["errors"])

    def test_frozen_raw_and_projection_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = {"candidates": [self.row]}
            text = canonical_json(raw)
            context = {**IDENTITY, "root_dir": root, "raw_sha256": sha256_bytes(text.encode())}
            atomic_text(root / "raw/responses/req.json.txt", text)
            register_runtime_context(root, POLICY, {"run_id": "run", "arm_id": "arm", "root_id": "root",
                "requests": {"req": {key: context[key] for key in ("intent_id", "source", "stage", "raw_sha256")}}})
            row = normalize_batch(raw, intent_id="intent_1", source="contract_guided", contract={},
                expanded=self.expanded, ordering_policy=POLICY, ordering_context=context)["candidates"][0]
            self.assertTrue(row["schema_valid"], row["schema_errors"])
            row.update(candidate_id="root:" + row["candidate_id"], root_id="root")
            self.assertEqual(verify_candidate(row, expanded=self.expanded, policy=POLICY, context=context)["status"], "PASS")
            sidecar = json.loads((root / row["ordering_derivation"]["path"]).read_text())
            self.assertEqual(sidecar["raw_row"], self.row)
            for field, value in (("semantic_support_status", "PASS"),
                                 ("program_temporal_relations", []), ("event_nodes", [])):
                changed = {**row, field: value}
                with self.subTest(field=field), self.assertRaises(ValueError):
                    verify_candidate(changed, expanded=self.expanded, policy=POLICY, context=context)
            row["gold"].reverse()
            with self.assertRaisesRegex(ValueError, "PROJECTION_MISMATCH"):
                verify_candidate(row, expanded=self.expanded, policy=POLICY, context=context)

    def test_full_transcript_abstention_and_frozen_repair(self):
        contract = {"intent_id": "intent_1", "required_evidence_groups": {}}
        prompt = build_prompt(self.expanded, contract, ordering_policy=POLICY)
        self.assertIn("Only ordering may explicitly abstain", prompt["prompt"])
        abstentions = [{"question_type": "ordering", "status": "INSUFFICIENT_EVIDENCE",
                        "reason": "No supported complete chronology."}]
        repaired = build_repair_prompt(self.expanded, contract, {"candidates": []}, "contract_guided",
            ordering_policy=POLICY, primary_abstentions=abstentions)
        self.assertIn("Do not emit or repair an ordering candidate", repaired["prompt"])
        self.assertIn(json.dumps(abstentions, ensure_ascii=False), repaired["prompt"])
        self.assertTrue(repaired["prompt_version"].endswith("_repair_v2"))

    def test_nonordering_cannot_claim_ordering_authority(self):
        sc = {"question_type": "single_choice", "question": "Who registered?",
              "options": dict(zip("ABCD", ("Ava", "Ben", "Cara", "Dan"))), "correct_option": "A",
              "evidence_refs": ["EXAMPLE_ONLY_T1"],
              "evidence_groups": {"person": ["EXAMPLE_ONLY_T1"]}, "discriminative_variable": "actor"}
        for field in ("ordering_contract_version", "ordering_derivation", "semantic_support_status",
                      "derived_gold", "program_temporal_relations"):
            row = {**sc, field: "self-certified"}
            result = normalize_batch({"candidates": [row]}, intent_id="intent_1",
                source="contract_guided", contract={}, expanded=self.expanded, ordering_policy=POLICY)
            self.assertIn("WRITER_ORDERING_AUTHORITY_ON_NONORDERING", result["candidates"][0]["schema_errors"])

    def test_free_v2_is_rejected_before_any_call(self):
        from ..h4_qa_mining_hybrid_generation_v1.free_exploration_generator import build_prompt as free
        with self.assertRaisesRegex(ValueError, "REQUIRES_CONTRACT_GUIDED"):
            free(self.expanded, {"intent_id": "intent_1"}, ordering_policy=POLICY)


if __name__ == "__main__":
    unittest.main()
