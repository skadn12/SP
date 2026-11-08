"""Offline regression checks against the actual frozen local source pool."""

import copy
import hashlib
import json
import unittest

from source_pool import SOURCE_DIR, freeze_unused_source_subset


HISTORICAL_SIX = {
    ("allenai/soda", "train", "6027743"),
    ("allenai/soda", "train", "6027790"),
    ("allenai/soda", "train", "6027845"),
    ("MultiWOZ_2.2", "train", "SNG1013.json"),
    ("MultiWOZ_2.2", "train", "SSNG0142.json"),
    ("MultiWOZ_2.2", "train", "PMUL0843.json"),
}


def event_key(item):
    return item["dataset"], item["split"], item["item_id"]


def sha(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(data).hexdigest()


class SourcePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.soda = json.loads((SOURCE_DIR / "soda_train_sample.json").read_text())
        cls.multiwoz = json.loads((SOURCE_DIR / "multiwoz_train_sample.json").read_text())
        cls.raw_items = {
            ("allenai/soda", "train", str(entry["row"]["original_index"])): entry["row"]
            for entry in cls.soda["rows"]
        }
        cls.raw_items.update({
            ("MultiWOZ_2.2", "train", entry["dialogue_id"]): entry for entry in cls.multiwoz
        })

    def test_next_nine_are_real_unused_events_and_match_existing_selection(self):
        result = freeze_unused_source_subset(HISTORICAL_SIX, count=9)
        keys = {event_key(item) for item in result["items"]}
        self.assertEqual(len(result["items"]), 9)
        self.assertEqual(len(keys), 9)
        self.assertFalse(keys & HISTORICAL_SIX)
        self.assertTrue(keys <= self.raw_items.keys())
        self.assertEqual([item["item_id"] for item in result["items"]], [
            "MUL0358.json", "MUL0928.json", "MUL1681.json", "MUL1914.json",
            "MUL2293.json", "MUL2431.json", "PMUL0121.json", "PMUL0172.json",
            "PMUL0949.json",
        ])

    def test_selection_and_manifest_are_deterministic(self):
        first = freeze_unused_source_subset(HISTORICAL_SIX, count=9)
        second = freeze_unused_source_subset(set(reversed(sorted(HISTORICAL_SIX))), count=9)
        self.assertEqual(first, second)
        unsigned = {key: value for key, value in first.items() if key != "manifest_sha256"}
        self.assertEqual(first["manifest_sha256"], sha(unsigned))

    def test_full_raw_items_excerpts_and_files_have_real_hashes(self):
        result = freeze_unused_source_subset(HISTORICAL_SIX, count=1000)
        receipts = {row["dataset"]: row for row in result["files"]}
        self.assertEqual({item["dataset"] for item in result["items"]}, {"allenai/soda", "MultiWOZ_2.2"})
        for receipt in receipts.values():
            from pathlib import Path
            self.assertEqual(receipt["local_sha256"], sha(Path(receipt["local_path"]).read_bytes()))
            self.assertEqual(receipt["source_receipt_sha256"], sha(Path(receipt["source_receipt_path"]).read_bytes()))
        for item in result["items"]:
            with self.subTest(source_id=item["source_id"]):
                original = self.raw_items[event_key(item)]
                self.assertEqual(item["source_item_sha256"], sha(original))
                self.assertEqual(item["excerpt_sha256"], sha(item["units"]))
                self.assertEqual(item["source_file_sha256"], receipts[item["dataset"]]["local_sha256"])
                if item["dataset"] == "MultiWOZ_2.2":
                    selected = original["turns"][:12]
                    self.assertEqual(item["source_span"], {"turn_start": 0, "turn_end": len(selected) - 1})
                    self.assertEqual([unit["text"] for unit in item["units"]],
                                     [turn["utterance"] for turn in selected])
                else:
                    self.assertEqual(item["units"][0]["text"], original["narrative"])
                    self.assertEqual([unit["text"] for unit in item["units"][1:]], original["dialogue"])

    def test_changing_card_or_span_cannot_reintroduce_consumed_event(self):
        first = freeze_unused_source_subset(HISTORICAL_SIX, count=1)["items"][0]
        changed = copy.deepcopy(first)
        changed.update(card_id="different-card", source_id="different-alias",
                       source_span={"turn_start": 4, "turn_end": 7}, units=[])
        consumed = HISTORICAL_SIX | {event_key(changed)}
        result = freeze_unused_source_subset(consumed, count=1000)
        self.assertNotIn(event_key(first), {event_key(item) for item in result["items"]})

    def test_shortage_returns_only_remaining_items_and_empty_when_exhausted(self):
        keep = {("allenai/soda", "train", "6027731"),
                ("MultiWOZ_2.2", "train", "MUL0358.json")}
        consumed = set(self.raw_items) - keep
        result = freeze_unused_source_subset(consumed, count=9)
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual({event_key(item) for item in result["items"]}, keep)
        self.assertEqual(freeze_unused_source_subset(set(self.raw_items), count=9)["items"], [])

    def test_invalid_count_or_consumed_identity_rejected(self):
        for count in (0, -1, True, 1.5, "9"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "source_count_invalid"):
                freeze_unused_source_subset(HISTORICAL_SIX, count=count)
        for keys in ({("dataset", "train")}, {("dataset", "train", 1)}, ["not-a-key"]):
            with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, "consumed_event_keys_invalid"):
                freeze_unused_source_subset(keys, count=1)


if __name__ == "__main__":
    unittest.main()
