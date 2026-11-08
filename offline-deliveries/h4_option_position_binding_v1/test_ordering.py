"""Focused checks for source-bound ordering option-position binding."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ordering import LABELS, bind, content_pairs, run, sha, source_order, verify


ROOT = "cc-train-episode-123456"
SOURCE = {
    "root_id": ROOT,
    "turns": [
        {"turn_ref": f"R01:S01:T00{i}", "speaker": "Sam", "text": f"Sam states event {i}."}
        for i in range(1, 7)
    ],
}
RAW = {
    "candidate_id": f"{ROOT}-order-v1-001",
    "root_id": ROOT,
    "question_type": "ordering",
    "chronology_kind": "conversation_order",
    "topic": "Sam's sequence",
    "question": "In what order do these statements APPEAR IN THE SUPPLIED DIALOGUE?",
    "items": {f"I{i}": f"Sam states event {i}." for i in range(1, 7)},
    "item_evidence": {
        f"I{i}": {"turn_ref": f"R01:S01:T00{i}", "speaker": "Sam",
                  "quote": f"Sam states event {i}.", "status": "statement",
                  "reason": f"Sam names event {i}."}
        for i in range(1, 7)
    },
    "gold": list(LABELS),
    "order_reason": "Source turn order.",
}


class OrderingPositionTest(unittest.TestCase):
    def test_rebind_is_deterministic_and_preserves_item_evidence_pairs(self):
        original = deepcopy(RAW)
        rebound, audit = bind(RAW, SOURCE, "a" * 64)
        self.assertEqual(bind(RAW, SOURCE, "a" * 64), (rebound, audit))
        self.assertEqual(RAW, original)
        self.assertEqual(source_order(rebound, SOURCE), rebound["gold"])
        self.assertNotEqual(rebound["gold"], list(LABELS))
        self.assertNotEqual(rebound["gold"][0], "I1")
        self.assertNotEqual(rebound["gold"][-1], "I6")
        self.assertGreaterEqual(sum(a != b for a, b in zip(rebound["gold"], LABELS)), 4)
        self.assertEqual(content_pairs(RAW), content_pairs(rebound))
        self.assertEqual(set(audit["old_to_new_label"]), set(LABELS))
        self.assertEqual(audit["old_item_evidence_multiset_sha256"],
                         audit["new_item_evidence_multiset_sha256"])

    def test_raw_gold_must_match_source_before_rebinding(self):
        row = deepcopy(RAW)
        row["gold"] = ["I2", "I1", "I3", "I4", "I5", "I6"]
        with self.assertRaisesRegex(ValueError, "gold_conflicts_with_source_order"):
            bind(row, SOURCE, "a" * 64)

    def test_wrong_quote_and_label_reference_are_rejected(self):
        row = deepcopy(RAW)
        row["item_evidence"]["I1"]["quote"] = "not in source"
        with self.assertRaisesRegex(ValueError, "quote_or_speaker"):
            bind(row, SOURCE, "a" * 64)
        row = deepcopy(RAW)
        row["item_evidence"]["I1"]["reason"] = "I1 happened first."
        with self.assertRaisesRegex(ValueError, "evidence_reason_contains_label_reference"):
            bind(row, SOURCE, "a" * 64)

    def test_cli_rejects_hash_mismatch_without_silent_repair(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            source_dir = base / "source" / ROOT
            source_dir.mkdir(parents=True)
            source_raw = json.dumps(SOURCE).encode()
            (source_dir / "source.json").write_bytes(source_raw)
            good = deepcopy(RAW)
            good["source_sha256"] = sha(source_raw)
            bad = deepcopy(RAW)
            bad["candidate_id"] = f"{ROOT}-order-v1-002"
            bad["source_sha256"] = "0" * 64
            input_path = base / "input.json"
            input_path.write_text(json.dumps({"questions": [good, bad]}))
            report = run(input_path, base / "source", base / "output")
            output = json.loads((base / "output" / "POOL.json").read_text())
            self.assertEqual(report["bound_count"], 1)
            self.assertEqual(report["rejected_count"], 1)
            self.assertIn("source_hash_mismatch", report["rejects"][0]["reason"])
            self.assertEqual(output["questions"][0]["candidate_id"], good["candidate_id"])
            self.assertEqual(output["questions"][0]["source_sha256"], sha(source_raw))
            self.assertEqual(output["questions"][0]["source_path"], str((source_dir / "source.json").resolve()))
            self.assertEqual(verify(input_path, base / "source", base / "output")["status"], "VERIFIED")
            output["questions"][0]["gold"] = list(LABELS)
            (base / "output" / "POOL.json").write_text(json.dumps(output))
            with self.assertRaisesRegex(ValueError, "saved_pool_changed"):
                verify(input_path, base / "source", base / "output")

    def test_cli_rejects_duplicate_event_set_after_position_binding(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            source_dir = base / "source" / ROOT
            source_dir.mkdir(parents=True)
            (source_dir / "source.json").write_text(json.dumps(SOURCE))
            duplicate = deepcopy(RAW)
            duplicate["candidate_id"] = f"{ROOT}-order-v1-002"
            input_path = base / "input.json"
            input_path.write_text(json.dumps([RAW, duplicate]))
            report = run(input_path, base / "source", base / "output")
            self.assertEqual(report["bound_count"], 1)
            self.assertIn("duplicate_event_set", report["rejects"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
