"""Focused checks for the raw Qwen mechanical gate and Solver contract."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import bridge


PROTOCOL_PATH = (Path(__file__).resolve().parent.parent /
                 "h4_crossroot_multiselect_solver_selection_v2" / "protocol.py")
spec = importlib.util.spec_from_file_location("v2_protocol", PROTOCOL_PATH)
assert spec and spec.loader
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class RawBridgeTests(unittest.TestCase):
    def setUp(self):
        root = "cc-train-episode-231899"
        directory = bridge.BATCHES / root / "batch-01"
        self.row = bridge.read(directory / "MODEL_DRAFT.json")["questions"][0]
        source = bridge.read(directory / "source.json")
        self.turns = {turn["turn_ref"]: turn for turn in source["turns"]}
        self.root = root
        self.path = (directory / "source.json").resolve()
        self.hash = bridge.sha(self.path)

    def check(self, row):
        return bridge.validate_candidate(row, self.row["candidate_id"], self.turns,
                                         self.root, self.path, self.hash)

    def test_prior_six_distinct_turn_and_r01_rejections_do_not_apply(self):
        self.assertLess(len({x["turn_ref"] for x in self.row["options"]}), 6)
        self.assertFalse(any(x["turn_ref"].startswith("R01:") for x in self.row["options"]))
        errors, normalized = self.check(self.row)
        self.assertEqual(errors, [])
        self.assertEqual(normalized["options"]["A"], self.row["options"][0]["proposition"])
        self.assertEqual(normalized["gold"], [label for label, option in zip(bridge.LABELS, self.row["options"])
                                              if option["relation"] == "supports"])

    def test_reject_source_and_gold_contract_errors(self):
        cases = (("quote_not_verbatim", "option_1_quote_not_verbatim", lambda r: r["options"][0].update(quote="made up")),
                 ("speaker_mismatch", "option_1_speaker_mismatch", lambda r: r["options"][0].update(speaker="Someone else")),
                 ("duplicate_option", "options_not_unique", lambda r: r["options"][1].update(proposition=r["options"][0]["proposition"])),
                 ("bad_relation", "option_1_relation_invalid", lambda r: r["options"][0].update(relation="maybe")))
        for label, expected, mutation in cases:
            with self.subTest(label=label):
                row = deepcopy(self.row)
                mutation(row)
                errors, normalized = self.check(row)
                self.assertIn(expected, errors)
                self.assertIsNone(normalized)

    def test_all_six_supported_is_allowed_but_one_is_rejected(self):
        row = deepcopy(self.row)
        for option in row["options"]:
            option["relation"] = "supports"
        errors, normalized = self.check(row)
        self.assertEqual(errors, [])
        self.assertEqual(normalized["gold"], list(bridge.LABELS))
        for option in row["options"][1:]:
            option["relation"] = "contradicts"
        errors, normalized = self.check(row)
        self.assertIn("gold_cardinality_invalid", errors)
        self.assertIsNone(normalized)

    def test_frozen_16_and_solver_v2_pool_contract(self):
        pool, report = bridge.build()
        self.assertEqual((report["draft_count"], report["mechanical_pass_count"]), (16, 16))
        self.assertEqual(report["mechanical_reject_count"], 0)
        self.assertEqual(len(report["verdicts"]), 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "POOL.json"
            path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
            _, questions, sources = protocol.validate_pool(path)
        self.assertEqual(len(questions), 16)
        self.assertEqual(set(sources), set(bridge.ROOTS))


if __name__ == "__main__":
    unittest.main()
