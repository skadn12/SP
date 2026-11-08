"""Regression checks for source-bound multiselect position rebinding."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("position_multiselect", HERE / "multiselect.py")
binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding)
INPUT = HERE.parent / "h4_crossroot_multiselect_batch_v32_raw_newroots/solver_pool/POOL.json"


class PositionBindingTests(unittest.TestCase):
    def test_frozen_qwen_pool_round_trip_and_diversity(self):
        original = json.loads(INPUT.read_text(encoding="utf-8"))
        pool, report = binding.build(INPUT)
        self.assertEqual((report["candidate_count"], report["non_six_nonprefix_count"],
                          report["six_true_count"]), (15, 12, 3))
        self.assertEqual(binding.build(INPUT), (pool, report))
        self.assertEqual(pool["provenance"]["input_pool_sha256"], binding.load_protocol().sha(INPUT))
        for before, after, audit in zip(original["questions"], pool["questions"], report["bindings"]):
            self.assertEqual(set(audit["display_to_original"].values()), set(binding.LABELS))
            self.assertEqual(before["question"], after["question"])
            for shown, source in audit["display_to_original"].items():
                self.assertEqual(after["options"][shown], before["options"][source])
                self.assertEqual(after["option_evidence"][shown], before["option_evidence"][source])
                self.assertEqual(shown in after["gold"], source in before["gold"])
            if len(before["gold"]) < 6:
                self.assertNotEqual(set(after["gold"]), set(binding.LABELS[:len(before["gold"])]))
            else:
                self.assertEqual(after["gold"], list(binding.LABELS))

    def test_replay_rejects_tampered_mapping(self):
        pool, report = binding.build(INPUT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "POOL.json"
            audit = Path(directory) / "POSITION_BINDING.json"
            output.write_text(json.dumps(pool), encoding="utf-8")
            audit.write_text(json.dumps(report), encoding="utf-8")
            binding.verify(INPUT, output, audit)
            pool["questions"][0]["gold"] = ["A", "B"]
            output.write_text(json.dumps(pool), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "POSITION_BINDING_REPLAY_MISMATCH"):
                binding.verify(INPUT, output, audit)

    def test_requires_unrebound_input(self):
        pool, _ = binding.build(INPUT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "POOL.json"
            path.write_text(json.dumps(pool), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ORIGINAL_QWEN_RAW_POOL_REQUIRED"):
                binding.build(path)


if __name__ == "__main__":
    unittest.main()
