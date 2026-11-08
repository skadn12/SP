"""Offline source-diversity and no-reuse checks; no provider or real reservation."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def local_module(name):
    path = Path(__file__).resolve().parent / (name + ".py")
    spec = importlib.util.spec_from_file_location("v24_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pool = local_module("source_pool")
inventory = local_module("source_inventory")


class SourceSelectionV24Tests(unittest.TestCase):
    def setUp(self):
        self.soda = {"rows": [
            {"row_idx": i, "truncated_cells": [], "row": {
                "original_index": 100 + i, "split": "train", "relation": relation,
                "narrative": "An event.", "speakers": ["A", "B"],
                "dialogue": ["A request.", "An answer."],
            }} for i, relation in enumerate(["xNeed", "xWant", "xReact"] * 3)
        ]}
        self.multiwoz = [
            {"dialogue_id": str(i), "services": [service], "turns": [
                {"turn_id": str(j), "speaker": "USER" if j % 2 == 0 else "SYSTEM",
                 "utterance": "Request or response."} for j in range(14)
            ]} for i, service in enumerate(["hotel", "restaurant", "taxi"] * 3)
        ]
        self.receipt = {"local_sha256": "a" * 64}

    def freeze(self, count=9, consumed=None):
        def read(_path, _receipt, dataset):
            return (deepcopy(self.soda) if dataset == "allenai/soda" else deepcopy(self.multiwoz),
                    deepcopy(self.receipt))
        with patch.object(pool, "_read_source", side_effect=read):
            return pool.freeze_unused_source_subset(set() if consumed is None else consumed, count=count)

    def test_balances_datasets_and_native_strata_and_is_prefix_stable(self):
        result = self.freeze()
        counts = Counter(item["dataset"] for item in result["items"])
        self.assertEqual(sorted(counts.values()), [4, 5])
        selected = result["selection_provenance"]["selected_order"]
        for dataset in counts:
            strata = Counter(tuple(item["metadata_values"]) for item in selected if item["dataset"] == dataset)
            self.assertEqual(len(strata), 3)
            self.assertLessEqual(max(strata.values()) - min(strata.values()), 1)
        self.assertEqual(self.freeze(count=6)["items"], result["items"][:6])
        self.assertEqual(result, self.freeze())
        self.soda["rows"].reverse()
        self.multiwoz.reverse()
        self.assertEqual(result, self.freeze())

    def test_consumed_whole_events_are_excluded_and_exhaustion_does_not_reuse(self):
        consumed = {("MultiWOZ_2.2", "train", str(i)) for i in range(8)}
        consumed.update(("allenai/soda", "train", str(100 + i)) for i in range(8))
        result = self.freeze(count=9, consumed=consumed)
        self.assertEqual(len(result["items"]), 2)
        selected_keys = {inventory.event_key(item) for item in result["items"]}
        self.assertFalse(selected_keys & consumed)
        self.assertEqual(self.freeze(count=9, consumed=consumed | selected_keys)["items"], [])

    def test_missing_metadata_is_empty_and_real_labels_are_not_replaced(self):
        del self.soda["rows"][0]["row"]["relation"]
        self.multiwoz[0]["services"] = ["taxi", "hotel", "taxi"]
        result = self.freeze(count=99)
        records = {row["source_id"]: row for row in result["selection_provenance"]["selected_order"]}
        self.assertEqual(records["soda:train:100"]["metadata_values"], [])
        self.assertEqual(records["multiwoz22:train:0"]["metadata_values"], ["hotel", "taxi"])

    def test_manifest_and_raw_item_and_excerpt_hashes_remain_bound(self):
        result = self.freeze()
        without_hash = {k: v for k, v in result.items() if k != "manifest_sha256"}
        self.assertEqual(result["manifest_sha256"], pool._sha(pool._canonical(without_hash)))
        for item in result["items"]:
            if item["dataset"] == "allenai/soda":
                entry = next(e for e in self.soda["rows"] if str(e["row"]["original_index"]) == item["item_id"])
                original = entry["row"]
                expected = pool._soda_item(entry, self.receipt)
            else:
                original = next(e for e in self.multiwoz if e["dialogue_id"] == item["item_id"])
                expected = pool._multiwoz_item(original, self.receipt)
            self.assertEqual(item, expected)
            self.assertEqual(item["source_item_sha256"], pool._sha(pool._canonical(original)))
            self.assertEqual(item["excerpt_sha256"], pool._sha(pool._canonical(item["units"])))
        allocation = inventory.legacy.allocate(result, ["r1", "r2", "r3"],
                                              {"accepted_uses": [], "prior_evidence": []}, 3)
        self.assertEqual(allocation["assigned_slots"], 9)
        for rows in allocation["assignments"].values():
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["dataset"] for row in rows}), 2)


class ReservationV24Tests(unittest.TestCase):
    def test_failed_v25_and_current_v34_reservations_remain_consumed(self):
        generations = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = {f"v{n}_runs": root / f"v{n}" for n in generations}
            current = root / "v38"
            item = {"dataset": "fixture", "split": "train", "item_id": "unused",
                    "source_id": "fixture:unused", "source_item_sha256": "a" * 64}
            failed = args["v25_runs"] / "failed_before_report"
            inventory.reserve(failed, {"r1": [item]}, inventory.collect_prior(current, **args))
            prior = inventory.collect_prior(current, **args)
            self.assertEqual(prior["accepted_uses"][0]["origin"], "v25_reservation")
            with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                inventory.reserve(current / "new_owner", {"r2": [item]}, prior)
            inventory.reserve(failed, {"r1": [item]}, prior)  # Same-slot retry only.
            second = dict(item, item_id="second", source_id="fixture:second")
            inventory.reserve(current / "new", {"r2": [second]}, prior)
            all_used = inventory.collect_prior(current, **args)
            self.assertEqual(len(all_used["accepted_uses"]), 2)
            self.assertEqual({row["origin"] for row in all_used["accepted_uses"]},
                             {"v25_reservation", "v38_reservation"})
            changed = dict(item, source_item_sha256="b" * 64)
            with self.assertRaisesRegex(ValueError, "source_event_hash_changed"):
                inventory.validate_available({"r2": [changed]}, all_used)


if __name__ == "__main__":
    unittest.main()
