"""Offline cross-version source accounting; no production writes or API calls."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import source_inventory as inventory


class FreshInventoryV43Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="v43_inventory_MOCK_ONLY_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.current = self.root / "v43" / "runs"
        self.paths = {
            name: self.root / name.removesuffix("_runs") / "runs"
            for name in inspect.signature(inventory.collect_prior).parameters
            if name != "current_runs"
        }
        sockets = patch("socket.socket", side_effect=AssertionError("offline_inventory_only"))
        sockets.start()
        self.addCleanup(sockets.stop)

    def collect(self):
        return inventory.collect_prior(self.current, **self.paths)

    @staticmethod
    def item(identity):
        return {
            "dataset": "MOCK_ONLY", "split": "train", "item_id": identity,
            "source_id": "MOCK_ONLY:train:" + identity,
            "source_item_sha256": "a" * 64,
        }

    @staticmethod
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def test_failed_v41_two_source_collection_stays_consumed(self):
        owner = self.paths["v41_runs"] / "failed_b2"
        sources = [self.item("failed-one"), self.item("failed-two")]
        inventory.reserve(owner, {"root": sources}, self.collect())
        reservation_bytes = (owner / "SOURCE_RESERVATIONS.json").read_bytes()
        self.write(owner / "LIVE_STARTED.json", {})
        self.write(owner / "REPORT.json", {
            "status": "COLLECTION_SHORTFALL", "accepted_uses": [],
            "retained_candidates": 0, "training_pool_eligible": False,
        })
        prior = self.collect()
        self.assertEqual(len(prior["accepted_uses"]), 2)
        self.assertEqual(len(prior["reserved_uses"]), 2)
        self.assertEqual({row["origin"] for row in prior["accepted_uses"]},
                         {"v41_reservation"})
        for source in sources:
            with self.subTest(source=source["source_id"]):
                with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                    inventory.reserve(self.current / "new_b2", {"root": [source]}, prior)
        self.assertEqual((owner / "SOURCE_RESERVATIONS.json").read_bytes(), reservation_bytes)

    def test_v42_v43_reservations_are_scanned_without_training_promotion(self):
        for version, directory in ((42, self.paths["v42_runs"]), (43, self.current)):
            owner = directory / "pending_candidate"
            inventory.reserve(owner, {"root": [self.item(f"v{version}")]}, self.collect())
            self.write(owner / "REPORT.json", {
                "accepted_uses": [], "retained_candidates": 1,
                "training_pool_eligible": False,
            })
        prior = self.collect()
        self.assertEqual({row["origin"] for row in prior["accepted_uses"]},
                         {"v42_reservation", "v43_reservation"})
        self.assertTrue(all(row["status"] == "reserved" for row in prior["accepted_uses"]))

    def test_diagnostic_reuse_does_not_consume_twice_or_admit_fresh_data(self):
        owner = self.paths["v41_runs"] / "failed_b2"
        payload = inventory.reserve(owner, {"root": [self.item("consumed")]}, self.collect())
        before = self.collect()
        diagnostic = self.paths["v42_runs"].parent / "diagnostics" / "reused_source"
        self.write(diagnostic / "LIVE_STARTED.json", {})
        # Even an accidental report with an accepted_uses entry outside runs
        # must not promote this diagnostic reuse into fresh source accounting.
        self.write(diagnostic / "REPORT.json", {
            "accepted_uses": payload["reservations"], "retained_candidates": 1,
            "training_pool_eligible": False, "production_admission": "diagnostic_only",
        })
        after = self.collect()
        self.assertEqual(after, before)
        self.assertEqual(after["accepted_uses"][0]["status"], "reserved")

    def test_live_v42_or_v43_run_requires_reservation(self):
        for version, directory in ((42, self.paths["v42_runs"]), (43, self.current)):
            owner = directory / "missing_reservation"
            self.write(owner / "LIVE_STARTED.json", {})
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "live_run_missing_source_reservations"):
                    self.collect()
            (owner / "LIVE_STARTED.json").unlink()

    def test_prepared_v41_v42_v43_sources_remain_available(self):
        for version, directory in ((41, self.paths["v41_runs"]),
                                   (42, self.paths["v42_runs"]), (43, self.current)):
            owner = directory / "prepared"
            source = self.item(f"prepared-v{version}")
            self.write(owner / "MANIFEST.json", {"version": version})
            self.write(owner / "dataset_sources.json", {"items": [source]})
            self.write(owner / "UNIQUE_EVENT_ALLOCATION.json", {
                "assignments": {"root": [source]},
            })
            inventory.validate_available({"root": [source]}, self.collect())
        self.assertEqual(self.collect()["accepted_uses"], [])
        self.assertEqual(self.collect()["prior_evidence"], [])

    def test_new_current_owner_cannot_hide_duplicate_v41_source(self):
        owner = self.paths["v41_runs"] / "failed"
        payload = inventory.reserve(owner, {"root": [self.item("consumed")]}, self.collect())
        forged = deepcopy(payload)
        duplicate = self.current / "corrupt_duplicate"
        forged["run"] = str(duplicate.resolve())
        forged["reservations"][0]["run"] = str(duplicate.resolve())
        self.write(duplicate / "SOURCE_RESERVATIONS.json", forged)
        with self.assertRaisesRegex(ValueError, "source_event_consumed_by_multiple_slots"):
            self.collect()


if __name__ == "__main__":
    unittest.main()
