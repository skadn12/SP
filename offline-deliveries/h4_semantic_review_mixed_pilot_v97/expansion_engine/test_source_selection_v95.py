"""Controlled v95 selection uses exact unused SODA events before any request."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import source_pool


class ControlledSourceSelectionV95Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.soda_rows = [
            {"row_idx": index, "row": {
                "original_index": item_id, "split": "train",
                "narrative": f"Complete event {item_id}: a visit is arranged.",
                "dialogue": ["Let us visit on Tuesday.", "I can come after lunch."],
                "speakers": ["Elena", "Mo"],
            }}
            for index, item_id in enumerate((10, 20, 30))
        ]
        soda_raw = self._write("soda_train_sample.json", {"rows": self.soda_rows})
        self._write("soda_train_sample_source.json", {
            "url": "https://fixture.invalid/soda",
            "sha256": hashlib.sha256(soda_raw).hexdigest(),
        })
        multiwoz_raw = self._write("multiwoz_train_sample.json", [{
            "dialogue_id": "fixture-train-1", "turns": [
                {"turn_id": 0, "speaker": "USER", "utterance": "Find a restaurant."},
                {"turn_id": 1, "speaker": "SYSTEM", "utterance": "Which area?"},
            ],
        }])
        self._write("multiwoz_train_source.json", {
            "url": "https://fixture.invalid/multiwoz",
            "download_sha256": hashlib.sha256(multiwoz_raw).hexdigest(),
            "download_bytes": len(multiwoz_raw),
        })

    def _write(self, name, value):
        raw = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        (self.directory / name).write_bytes(raw)
        return raw

    def _freeze(self, source_ids, consumed=frozenset()):
        return source_pool.freeze_unused_source_subset(
            consumed, len(source_ids), self.directory,
            selected_source_ids=source_ids,
        )

    def test_selected_order_and_complete_source_lineage_survive_freezing(self):
        original_bytes = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        selected = ["soda:train:20", "soda:train:10"]
        manifest = self._freeze(selected)
        self.assertEqual([item["source_id"] for item in manifest["items"]], selected)
        self.assertEqual(manifest["preselected_source_ids"], selected)
        self.assertFalse(manifest["selection_is_complexity_or_difficulty_certificate"])
        for item, row_index in zip(manifest["items"], (1, 0)):
            row = self.soda_rows[row_index]["row"]
            self.assertEqual([unit["text"] for unit in item["units"]],
                             [row["narrative"], *row["dialogue"]])
            canonical_row = json.dumps(row, ensure_ascii=True, sort_keys=True,
                                       separators=(",", ":")).encode()
            self.assertEqual(item["source_item_sha256"], hashlib.sha256(canonical_row).hexdigest())
            self.assertEqual(item["source_file_sha256"],
                             hashlib.sha256(original_bytes["soda_train_sample.json"]).hexdigest())
            self.assertEqual(item["source_span"], {
                "narrative": "full", "dialogue_start": 0, "dialogue_end": 1})
        self.assertEqual({p.name: p.read_bytes() for p in self.directory.iterdir()}, original_bytes)

    def test_explicit_selection_cannot_reuse_consumed_event(self):
        consumed = {("allenai/soda", "train", "20")}
        with self.assertRaisesRegex(ValueError, "selected_source_unavailable_or_consumed"):
            self._freeze(["soda:train:20"], consumed)
        available = self._freeze(["soda:train:30", "soda:train:10"], consumed)
        self.assertEqual([item["item_id"] for item in available["items"]], ["30", "10"])

    def test_controlled_batch_rejects_existing_non_soda_source(self):
        with self.assertRaisesRegex(ValueError, "controlled_batch_requires_preselected_soda_sources"):
            self._freeze(["multiwoz22:train:fixture-train-1"])

    def test_duplicate_unknown_and_count_mismatched_selection_rejected(self):
        with self.assertRaisesRegex(ValueError, "source_selection_invalid"):
            self._freeze(["soda:train:10", "soda:train:10"])
        with self.assertRaisesRegex(ValueError, "selected_source_unavailable_or_consumed"):
            self._freeze(["soda:train:999"])
        with self.assertRaisesRegex(ValueError, "source_selection_invalid"):
            source_pool.freeze_unused_source_subset(
                set(), 2, self.directory, selected_source_ids=["soda:train:10"])


if __name__ == "__main__":
    unittest.main()
