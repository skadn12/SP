"""Offline source isolation and shared ordering derivation checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ..core import atomic_text, canonical_json, sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt, build_repair_prompt
from ..h4_qa_multiroot_k3_t08_probe_v1.validator import validate_candidate
from ..h4_r02_ordering_temporal_contract_v1.contract import (
    VERSION, derive_ordering, register_runtime_context, verify_candidate,
)
from ..h4_r02_ordering_temporal_contract_v1.guidance import synthetic_example
from .sources import (
    MEMORY_ONLY, TRANSCRIPT_ONLY, TRANSCRIPT_PLUS_MEMORY, evidence_catalog,
    generic_memory_contract, memory_navigation_block, memory_only_envelope,
    projection_sha256, source_envelope,
)

POLICY = {"version": VERSION, "node_count": 4}


def memory_fixture(n=4):
    example = synthetic_example(n)
    items = [{"memory_ref": f"MEM_ITEM_{index:03d}", "bucket_type": "Event", "key": f"example-{index}",
              "content": turn["text"]} for index, turn in enumerate(example["expanded"]["turns"], 1)]
    refs = {turn["turn_ref"]: item["memory_ref"]
            for turn, item in zip(example["expanded"]["turns"], items)}
    raw = json.loads(json.dumps(example["raw"]))
    row = raw["candidates"][0]
    row["intent_reference"] = "memory_only_mixed"
    for entity in row["event_nodes"] + row["temporal_relations"]:
        entity["evidence_refs"] = [refs[ref] for ref in entity["evidence_refs"]]
    row["evidence_groups"] = {key: [refs[ref] for ref in group]
                              for key, group in row["evidence_groups"].items()}
    expanded = memory_only_envelope("root", items, snapshot_sha256="a" * 64,
                                    projection_sha256=projection_sha256(items))
    return expanded, raw


class SourceIsolationTests(unittest.TestCase):
    def test_transcript_prompt_unchanged_and_navigation_is_only_suffix(self):
        original = {"root_id": "root", "turns": [{"turn_ref": "t1", "speaker": "S", "text": "VISIBLE_TRANSCRIPT"}]}
        contract = {"intent_id": "intent_1", "intent_type": "temporal", "discriminative_variable": "VISIBLE_INTENT"}
        baseline = build_prompt(original, contract, ordering_policy=POLICY)
        explicit = source_envelope(original, TRANSCRIPT_ONLY)
        self.assertEqual(baseline, build_prompt(explicit, contract, ordering_policy=POLICY))
        memory, _ = memory_fixture()
        plus = source_envelope(original, TRANSCRIPT_PLUS_MEMORY, memory_items=memory["memory_items"],
                               snapshot_sha256=memory["memory_snapshot_sha256"],
                               projection_sha256_value=memory["memory_projection_sha256"])
        prompt = build_prompt(plus, contract, ordering_policy=POLICY)["prompt"]
        block = memory_navigation_block(plus)
        self.assertEqual(prompt, baseline["prompt"] + "\n" + block)
        repair = build_repair_prompt(plus, contract, {"candidates": []}, "contract_guided", ordering_policy=POLICY)["prompt"]
        self.assertEqual(repair.count(block), 1)
        self.assertEqual(set(evidence_catalog(plus)), {"t1"})
        self.assertNotIn(memory["memory_snapshot_sha256"], prompt)
        self.assertNotIn(memory["memory_projection_sha256"], prompt)

    def test_memory_only_discarded_transcript_and_rejects_intent_and_hidden_fields(self):
        memory, _ = memory_fixture()
        original = {"root_id": "root", "turns": [{"turn_ref": "PRIVATE_REF", "text": "PRIVATE_TRANSCRIPT"}],
                    "reasoning_chain": "PRIVATE_CHAIN", "qa": "PRIVATE_QA"}
        envelope = source_envelope(original, MEMORY_ONLY, memory_items=memory["memory_items"],
                                  snapshot_sha256=memory["memory_snapshot_sha256"],
                                  projection_sha256_value=memory["memory_projection_sha256"])
        contract = generic_memory_contract()
        for spec in (build_prompt(envelope, contract, ordering_policy=POLICY),
                     build_repair_prompt(envelope, contract, {"candidates": []}, "contract_guided", ordering_policy=POLICY)):
            prompt = spec["prompt"]
            for forbidden in ("PRIVATE_REF", "PRIVATE_TRANSCRIPT", "PRIVATE_CHAIN", "PRIVATE_QA",
                              "EXAMPLE_ONLY_T", "CONTRACT GUIDANCE:", "DIALOGUE:\n", "transcript_sha256", "a" * 64):
                self.assertNotIn(forbidden, prompt)
            self.assertIn("2 single_choice, 1 multi_select, and 1 ordering", prompt)
            self.assertIn("INSUFFICIENT_EVIDENCE", prompt)
            self.assertIn("MEM_ITEM_001", prompt)
            self.assertEqual(spec["expected_counts"], {"single_choice": 2, "multi_select": 1, "ordering": 1})
        with self.assertRaisesRegex(ValueError, "INTENT_LEAK"):
            build_prompt(envelope, {**contract, "reasoning_chain": "SECRET"}, ordering_policy=POLICY)
        for field in ("transcript", "reasoning_chain", "old_gold"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "INPUT_LEAK"):
                build_prompt({**envelope, field: "SECRET"}, contract, ordering_policy=POLICY)
        with self.assertRaisesRegex(ValueError, "INPUT_LEAK"):
            build_prompt({**envelope, "turns": original["turns"]}, contract, ordering_policy=POLICY)

    def test_memory_items_do_not_imitate_turns_and_projection_drift_rejected(self):
        expanded, _ = memory_fixture()
        catalog = evidence_catalog(expanded)
        self.assertEqual(set(catalog), {f"MEM_ITEM_{x:03d}" for x in range(1, 5)})
        self.assertTrue(all("turn_ref" not in row for row in catalog.values()))
        drift = deepcopy(expanded)
        drift["memory_items"][0]["content"] = "Changed fact"
        with self.assertRaisesRegex(ValueError, "PROJECTION_DRIFT"):
            evidence_catalog(drift)
        drift = deepcopy(expanded)
        drift["memory_items"][0]["turn_ref"] = "t1"
        with self.assertRaisesRegex(ValueError, "ITEM_INVALID"):
            evidence_catalog(drift)


class MemoryDerivationTests(unittest.TestCase):
    def test_same_dag_algorithm_4_5_6_and_memory_source_metadata(self):
        for n in (4, 5, 6):
            expanded, raw = memory_fixture(n)
            result = derive_ordering(raw["candidates"][0], expanded=expanded,
                                    policy={"version": VERSION, "node_count": n}, identity={})
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["projected_row"]["gold"], ["I2", "I1", "I4", "I3", "I5", "I6"][:n])
            record = result["record"]
            self.assertNotIn("transcript_sha256", record)
            self.assertEqual(record["evidence_source"]["memory_snapshot_sha256"], "a" * 64)
            self.assertEqual(record["evidence_projection"]["support_scope"], "MEMORY_REFERENCE_EXISTENCE_ONLY")
            self.assertTrue(all("memory_ref" in x and "turn_ref" not in x for x in record["evidence_projection"]["exact_excerpts"]))
            bad = deepcopy(raw["candidates"][0])
            bad["event_nodes"][0]["evidence_refs"] = ["EXAMPLE_ONLY_T1"]
            self.assertIn("ORDERING_EVIDENCE_REF_UNKNOWN", derive_ordering(bad, expanded=expanded,
                          policy={"version": VERSION, "node_count": n}, identity={})["errors"])
        transcript = {"turns": [{"turn_ref": "t1", "text": "Original"}]}
        self.assertIn("ORDERING_EVIDENCE_REF_UNKNOWN", derive_ordering(raw["candidates"][0], expanded=transcript,
                      policy={"version": VERSION, "node_count": n}, identity={})["errors"])

    def test_sc_ms_use_memory_refs_and_keep_original_answer_contract(self):
        expanded, _ = memory_fixture()
        sc = {"question_type": "single_choice", "question": "Which event began?",
              "options": {"A": "Inspection", "B": "Filing", "C": "Review", "D": "Cleanup"},
              "correct_option": "A", "evidence_refs": ["MEM_ITEM_001"],
              "evidence_groups": {"support": ["MEM_ITEM_001"]}, "discriminative_variable": "event"}
        ms = {**sc, "question_type": "multi_select", "options": {**sc["options"], "E": "Packing"}, "gold": ["A", "B"]}
        rows = normalize_batch({"candidates": [sc, ms]}, intent_id="memory_only_mixed", source="contract_guided",
                               contract=generic_memory_contract(), expanded=expanded, ordering_policy=POLICY)["candidates"]
        self.assertEqual(rows[0]["gold"], ["A"])
        self.assertEqual(rows[1]["gold"], ["A", "B"])
        for row in rows:
            self.assertTrue(row["schema_valid"])
            self.assertEqual(row["evidence_validation_scope"], "MEMORY_REFERENCE_EXISTENCE_ONLY")
            self.assertEqual(validate_candidate(row, valid_turn_refs=set(evidence_catalog(expanded)))["decision"], "pass")
        sc["evidence_refs"] = ["t1"]
        result = normalize_batch({"candidates": [sc]}, intent_id="memory_only_mixed", source="contract_guided",
                                 contract=generic_memory_contract(), expanded=expanded, ordering_policy=POLICY)
        self.assertIn("evidence_ref_unknown", result["candidates"][0]["schema_errors"])

    def test_persisted_derivation_binds_snapshot_and_projection(self):
        expanded, raw = memory_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = canonical_json(raw)
            context = {"root_dir": root, "run_id": "run", "arm_id": "MEMORY_ONLY", "root_id": "root",
                       "intent_id": "memory_only_mixed", "source": "contract_guided", "stage": "primary",
                       "request_id": "request", "raw_sha256": sha256_bytes(text.encode())}
            atomic_text(root / "raw/responses/request.json.txt", text)
            register_runtime_context(root, POLICY, {**{x: context[x] for x in ("root_id", "run_id", "arm_id")},
                                     "requests": {"request": {x: context[x] for x in ("intent_id", "source", "stage", "raw_sha256")}}})
            candidate = normalize_batch(raw, intent_id="memory_only_mixed", source="contract_guided",
                                        contract=generic_memory_contract(), expanded=expanded,
                                        ordering_policy=POLICY, ordering_context=context)["candidates"][0]
            candidate.update(candidate_id="root:" + candidate["candidate_id"], root_id="root")
            self.assertEqual(verify_candidate(candidate, expanded=expanded, policy=POLICY, context={"root_dir": root})["status"], "PASS")
            changed = deepcopy(expanded)
            changed["memory_snapshot_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "REDERIVATION"):
                verify_candidate(candidate, expanded=changed, policy=POLICY, context={"root_dir": root})
            changed = deepcopy(expanded)
            changed["root_id"] = "other-root"
            with self.assertRaisesRegex(ValueError, "REDERIVATION"):
                verify_candidate(candidate, expanded=changed, policy=POLICY, context={"root_dir": root})
            changed = deepcopy(expanded)
            changed["memory_items"][0]["content"] += " Updated"
            changed["memory_projection_sha256"] = projection_sha256(changed["memory_items"])
            with self.assertRaisesRegex(ValueError, "REDERIVATION"):
                verify_candidate(candidate, expanded=changed, policy=POLICY, context={"root_dir": root})


if __name__ == "__main__":
    unittest.main()
