"""Offline graph, schema, normalization and integrity regression fixtures."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from .contract import (VERSION, derive_ordering, ordering_schema, policy_config,
                       persist_derivation, register_runtime_context, resolve_runtime_context,
                       verify_candidate)

POLICY = {"version": VERSION, "node_count": 4}
EXPANDED = {"turns": [
    {"turn_ref": f"t{i}", "speaker": "S", "text": f"event {i}"}
    for i in range(1, 7)
]}
IDENTITY = {"run_id": "run", "arm_id": "arm", "root_id": "root",
            "intent_id": "intent_1", "source": "contract_guided", "stage": "primary",
            "request_id": "req", "raw_row_index": 1,
            "candidate_id": "root:intent_1-C-Q01", "raw_sha256": "raw"}


def edge(before, after):
    return {"before": before, "after": after, "evidence_refs": ["t1"], "temporal_basis": "date"}


def fixture_row(n=4, edges=None):
    return {"question_type": "ordering", "ordering_contract_version": VERSION,
            "question": "Order the completed events", "discriminative_variable": "time",
            "event_nodes": [{"item_id": f"I{i}", "item_text": f"completed event {i}",
                             "actor": "S", "event_state": "completed", "evidence_refs": [f"t{i}"]}
                            for i in range(1, n + 1)],
            "temporal_relations": edges if edges is not None else [
                edge(f"I{i}", f"I{i+1}") for i in range(1, n)],
            "evidence_groups": {"time": ["t1", "t2"]}, "gold": None}


def derive(row, n=4):
    return derive_ordering(row, expanded=EXPANDED,
                           policy={"version": VERSION, "node_count": n}, identity=IDENTITY)


class GraphTests(unittest.TestCase):
    def test_policy_validation_and_schema(self):
        self.assertEqual(policy_config(None), {"version": "legacy", "node_count": 4})
        for invalid in ({"version": VERSION, "node_count": 7}, {"version": "legacy", "node_count": 5},
                        {"version": VERSION, "node_count": True}, {"version": VERSION},
                        {"version": "other", "node_count": 4}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                policy_config(invalid)
        self.assertEqual(ordering_schema(5)["properties"]["event_nodes"]["minItems"], 5)

    def test_unique_4_5_6_without_id_tiebreaking(self):
        for n in (4, 5, 6):
            ids = [f"I{x}" for x in range(n, 0, -1)]
            result = derive(fixture_row(n, [edge(a, b) for a, b in zip(ids, ids[1:])]), n)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["projected_row"]["gold"], ids)
            self.assertEqual(result["record"]["semantic_support_status"], "NOT_ASSESSED")

    def test_sparse_unique_edges_and_duplicates(self):
        result = derive(fixture_row(edges=[edge("I2", "I1"), edge("I1", "I4"), edge("I4", "I3"), edge("I2", "I1")]))
        self.assertEqual(result["projected_row"]["gold"], ["I2", "I1", "I4", "I3"])
        self.assertEqual(len(result["record"]["temporal_relations"]), 4)
        self.assertEqual(len(result["record"]["unique_edges"]), 3)

    def test_insufficient_isolated_and_no_edges(self):
        for edges in ([], [edge("I1", "I2")], [edge("I1", "I2"), edge("I2", "I3")]):
            result = derive(fixture_row(edges=edges))
            self.assertIn("ORDER_NOT_DETERMINED_BY_CONSTRAINTS", result["errors"])
            self.assertEqual(result["projected_row"]["gold"], [])

    def test_cycle_checked_before_ties(self):
        for edges in ([edge("I1", "I1")],
                      [edge("I1", "I2"), edge("I2", "I1")],
                      [edge("I1", "I2"), edge("I2", "I1"), edge("I3", "I4")]):
            result = derive(fixture_row(edges=edges))
            self.assertIn("ORDERING_GRAPH_CYCLE", result["errors"])
            self.assertNotIn("ORDER_NOT_DETERMINED_BY_CONSTRAINTS", result["errors"])
        self.assertIn("ORDERING_SELF_LOOP", derive(fixture_row(edges=[edge("I1", "I1")]))["errors"])

    def test_unknown_missing_duplicate_and_count(self):
        bad = fixture_row(); bad["event_nodes"][0]["item_id"] = "I9"
        self.assertIn("ORDERING_NODE_ID_SET_INVALID", derive(bad)["errors"])
        bad = fixture_row(); bad["event_nodes"].pop()
        self.assertIn("ORDERING_NODE_COUNT_INVALID", derive(bad)["errors"])
        bad = fixture_row(); bad["event_nodes"][1]["item_id"] = "I1"
        self.assertIn("ORDERING_NODE_ID_INVALID_OR_DUPLICATE", derive(bad)["errors"])
        bad = fixture_row(); bad["temporal_relations"][0]["before"] = "I9"
        self.assertIn("ORDERING_EDGE_ENDPOINT_UNKNOWN", derive(bad)["errors"])
        self.assertIn("ORDERING_NODE_COUNT_INVALID", derive(fixture_row(5))["errors"])

    def test_refs_missing_unknown_cross_transcript_and_exact_excerpts(self):
        for ref in ("other-root:t1", "t999", 1, {"turn_ref": "t1"}):
            bad = fixture_row(); bad["event_nodes"][0]["evidence_refs"] = [ref]
            self.assertIn("ORDERING_EVIDENCE_REF_UNKNOWN", derive(bad)["errors"])
        for field in ("event_nodes", "temporal_relations"):
            bad = fixture_row(); bad[field][0]["evidence_refs"] = []
            self.assertIn("ORDERING_EVIDENCE_MISSING", derive(bad)["errors"])
        result = derive(fixture_row())
        self.assertEqual(result["projected_row"]["evidence_refs"], ["t1", "t2", "t3", "t4"])
        self.assertEqual(result["record"]["evidence_projection"]["exact_excerpts"][0],
                         {"kind": "event_node", "index": 0, "turn_ref": "t1", "speaker": "S", "text": "event 1"})

    def test_writer_gold_and_program_fields_rejected(self):
        for gold in ([], "", ["I1", "I2", "I3", "I4"], "I1,I2,I3,I4", False):
            bad = fixture_row(); bad["gold"] = gold
            self.assertIn("WRITER_GOLD_FORBIDDEN", derive(bad)["errors"])
        for field in ("final_order", "canonical_gold", "derived_gold", "gold_derivation", "items", "evidence_refs"):
            bad = fixture_row(); bad[field] = []
            self.assertIn("WRITER_PROGRAM_FIELDS_FORBIDDEN", derive(bad)["errors"])
        good = fixture_row(); del good["gold"]
        self.assertEqual(derive(good)["status"], "PASS")

    def test_unknown_fields_metadata_and_missing_version_rejected(self):
        bad = fixture_row(); bad["self_reported_valid"] = True
        self.assertIn("ORDERING_SCHEMA_INVALID", derive(bad)["errors"])
        bad = fixture_row(); bad["metadata"] = {"gold": "I1"}
        self.assertIn("ORDERING_SCHEMA_INVALID", derive(bad)["errors"])
        bad = fixture_row(); del bad["ordering_contract_version"]
        self.assertIn("ORDERING_CONTRACT_VERSION_MISMATCH", derive(bad)["errors"])


class NormalizationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = {**IDENTITY, "root_dir": self.root}
        self.contract = {"intent_id": "intent_1", "required_evidence_groups": {"time": ["t1"]}}

    def normalize(self, rows, policy=POLICY, stage="primary"):
        raw = {"candidates": rows}
        text = canonical_json(raw)
        context = {**self.context, "stage": stage, "request_id": "req-" + stage,
                   "raw_sha256": sha256_bytes(text.encode()), "root_dir": self.root}
        atomic_text(self.root / "raw" / "responses" / f"{context['request_id']}.json.txt", text)
        registry_path = self.root / "ordering_runtime_context.json"
        requests = (json.loads(registry_path.read_text())["requests"] if registry_path.exists() else {})
        requests[context["request_id"]] = {key: context[key] for key in
                                           ("intent_id", "source", "stage", "raw_sha256")}
        register_runtime_context(self.root, policy, {
            "run_id": "run", "arm_id": "arm", "root_id": "root",
            "requests": requests})
        return normalize_batch(raw, intent_id="intent_1", source="contract_guided",
                               contract=self.contract, expanded=EXPANDED,
                               ordering_policy=policy, ordering_context=context)

    def candidate(self):
        candidate = self.normalize([fixture_row()])["candidates"][0]
        candidate["candidate_id"] = "root:" + candidate["candidate_id"]
        candidate["root_id"] = "root"
        return candidate

    def verify(self, candidate, **kwargs):
        return verify_candidate(candidate, expanded=kwargs.get("expanded", EXPANDED),
                                policy=kwargs.get("policy", POLICY),
                                context={"root_dir": self.root, "root_id": "root", "run_id": "run", "arm_id": "arm"})

    def test_normalized_failure_keeps_row_index_and_scms_identity(self):
        sc = {"question_type": "single_choice", "question": "Q?",
              "options": dict(zip("ABCD", ("a", "b", "c", "d"))), "correct_option": "B",
              "evidence_refs": ["t1"], "evidence_groups": {"time": ["t1"]}, "discriminative_variable": "x"}
        ms = {**sc, "question_type": "multi_select", "options": {**sc["options"], "E": "e"}, "gold": ["A", "C"]}
        rows = [sc, fixture_row(edges=[]), ms, sc]
        result = self.normalize(rows)
        baseline = normalize_batch({"candidates": rows}, intent_id="intent_1", source="contract_guided",
                                   contract=self.contract, expanded=EXPANDED)
        for index in (0, 2, 3):
            self.assertEqual(result["candidates"][index], baseline["candidates"][index])
        self.assertFalse(result["candidates"][1]["schema_valid"])
        self.assertEqual([row["source_row_index"] for row in result["candidates"]], [1, 2, 3, 4])
        self.assertEqual(result["candidates"][2]["candidate_id"], "intent_1-C-Q03")
        self.assertTrue((self.root / result["candidates"][1]["ordering_derivation"]["path"]).exists())

    def test_complete_integrity_and_4_5_6_projection(self):
        for n in (4, 5, 6):
            self.root = Path(self.temp.name) / str(n)
            policy = {"version": VERSION, "node_count": n}
            candidate = self.normalize([fixture_row(n)], policy=policy)["candidates"][0]
            candidate["candidate_id"] = "root:" + candidate["candidate_id"]
            candidate["root_id"] = "root"
            self.assertEqual(self.verify(candidate, policy=policy)["status"], "PASS")

    def test_version_downgrade_and_missing_sidecar_rejected(self):
        original = self.candidate()
        for key in ("ordering_contract_version", "ordering_derivation"):
            candidate = copy.deepcopy(original); candidate.pop(key)
            with self.assertRaises(ValueError): self.verify(candidate)
        candidate = copy.deepcopy(original)
        candidate.pop("ordering_contract_version"); candidate.pop("ordering_derivation")
        with self.assertRaisesRegex(ValueError, "RUNTIME_POLICY_MISMATCH"):
            self.verify(candidate, policy={"version": "legacy", "node_count": 4})

    def test_gold_text_identity_transcript_and_sidecar_tampering_rejected(self):
        original = self.candidate()
        for key, value in (("gold", list(reversed(original["gold"]))), ("items", {**original["items"], "I1": "changed"}),
                           ("candidate_id", "root:intent_2-C-Q01"), ("intent_id", "intent_2"),
                           ("source_row_index", 2)):
            candidate = copy.deepcopy(original); candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.verify(candidate)
        changed = copy.deepcopy(EXPANDED); changed["turns"][0]["text"] = "changed"
        with self.assertRaisesRegex(ValueError, "REDERIVATION"): self.verify(original, expanded=changed)
        path = self.root / original["ordering_derivation"]["path"]
        record = json.loads(path.read_text()); record["derived_gold"].reverse(); atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, "HASH_MISMATCH"): self.verify(original)

    def test_request_inventory_rejected(self):
        candidate = self.candidate()
        registry = json.loads((self.root / "ordering_runtime_context.json").read_text())
        registry["requests"]["req-primary"]["stage"] = "repair"
        atomic_json(self.root / "ordering_runtime_context.json", registry)
        with self.assertRaisesRegex(ValueError, "REGISTERED_REQUEST_MISMATCH"): self.verify(candidate)

    def test_raw_missing_or_modified_and_registry_missing_rejected(self):
        candidate = self.candidate()
        raw_path = self.root / "raw/responses/req-primary.json.txt"
        original = raw_path.read_text()
        raw_path.unlink()
        with self.assertRaisesRegex(ValueError, "RAW_RESPONSE_MISSING"): self.verify(candidate)
        atomic_text(raw_path, original + " ")
        with self.assertRaisesRegex(ValueError, "RAW_RESPONSE_HASH_MISMATCH"): self.verify(candidate)
        atomic_text(raw_path, original)
        (self.root / "ordering_runtime_context.json").unlink()
        with self.assertRaisesRegex(ValueError, "RUNTIME_CONTEXT_MISSING"): self.verify(candidate)

    def test_registry_missing_policy_cannot_downgrade(self):
        candidate = self.candidate()
        path = self.root / "ordering_runtime_context.json"
        registry = json.loads(path.read_text()); registry.pop("ordering_policy")
        atomic_json(path, registry)
        with self.assertRaisesRegex(ValueError, "RUNTIME_POLICY_MISSING"): self.verify(candidate)
        with self.assertRaisesRegex(ValueError, "RUNTIME_POLICY_MISSING"):
            resolve_runtime_context(self.root)

    def test_runtime_context_resolves_trusted_policy_and_rejects_conflicts(self):
        self.candidate()
        policy, context = resolve_runtime_context(self.root)
        self.assertEqual(policy, POLICY)
        self.assertEqual(context["run_id"], "run")
        with self.assertRaisesRegex(ValueError, "RUNTIME_POLICY_MISMATCH"):
            resolve_runtime_context(self.root, {"version": "legacy", "node_count": 4})
        with self.assertRaisesRegex(ValueError, "RUNTIME_IDENTITY_MISMATCH"):
            resolve_runtime_context(self.root, ordering_context={"arm_id": "other"})

    def test_primary_repair_sidecars_distinct_and_immutable(self):
        primary = self.normalize([fixture_row()])["candidates"][0]["ordering_derivation"]
        repair = self.normalize([fixture_row()], stage="repair")["candidates"][0]["ordering_derivation"]
        self.assertNotEqual(primary["path"], repair["path"])
        record = json.loads((self.root / primary["path"]).read_text()); record["errors"] = ["changed"]
        with self.assertRaisesRegex(ValueError, "DERIVATION_DRIFT"):
            persist_derivation(record, root_dir=self.root)


if __name__ == "__main__":
    unittest.main()
