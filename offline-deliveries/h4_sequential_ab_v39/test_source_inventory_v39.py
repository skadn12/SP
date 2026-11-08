"""Offline inventory upgrade checks; all reservations are temporary fixtures."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import source_inventory as inventory


class V39ReservationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kwargs = {
            name: self.root / name
            for name in inspect.signature(inventory.collect_prior).parameters
            if name != "current_runs"
        }
        self.current = self.root / "v39" / "runs"
        self.item = {
            "dataset": "fixture", "split": "train", "item_id": "reserved-v38",
            "source_id": "fixture:reserved-v38", "source_item_sha256": "a" * 64,
        }

    def collect(self):
        return inventory.collect_prior(self.current, **self.kwargs)

    def test_failed_v38_remains_consumed_and_same_owner_can_retry(self):
        owner = self.kwargs["v38_runs"] / "failed"
        original = inventory.reserve(owner, {"root": [self.item]}, self.collect())
        (owner / "LIVE_STARTED.json").write_text("{}\n")
        (owner / "REPORT.json").write_text(json.dumps({
            "status": "B_SHORTFALL", "accepted_uses": [],
        }))
        prior = self.collect()
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(prior["accepted_uses"][0]["origin"], "v38_reservation")
        self.assertEqual(prior["accepted_uses"][0]["status"], "reserved")
        self.assertEqual(inventory.reserve(owner, {"root": [self.item]}, prior), original)
        with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
            inventory.reserve(self.current / "different_run", {"root": [self.item]}, prior)
        new_item = dict(self.item, item_id="fresh-v39", source_id="fixture:fresh-v39")
        inventory.reserve(self.current / "fresh", {"root": [new_item]}, prior)
        self.assertEqual({row["origin"] for row in self.collect()["accepted_uses"]},
                         {"v38_reservation", "v39_reservation"})

    def test_duplicate_historical_and_current_owner_is_rejected(self):
        old_owner = self.kwargs["v38_runs"] / "old"
        payload = inventory.reserve(old_owner, {"root": [self.item]}, self.collect())
        new_owner = self.current / "other"
        new_owner.mkdir(parents=True)
        # Simulate an externally corrupted reservation; never call reserve() to bypass uniqueness.
        payload["run"] = str(new_owner.resolve())
        payload["reservations"][0]["run"] = str(new_owner.resolve())
        (new_owner / "SOURCE_RESERVATIONS.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "source_event_consumed_by_multiple_slots"):
            self.collect()

    def test_v38_live_without_reservation_is_rejected(self):
        owner = self.kwargs["v38_runs"] / "interrupted"
        owner.mkdir(parents=True)
        (owner / "LIVE_STARTED.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "live_run_missing_source_reservations"):
            self.collect()

    def test_v38_prepare_only_does_not_consume_assigned_source(self):
        owner = self.kwargs["v38_runs"] / "prepare_only"
        owner.mkdir(parents=True)
        (owner / "MANIFEST.json").write_text(json.dumps({"version": "h4_sequential_ab_v38"}))
        (owner / "dataset_sources.json").write_text(json.dumps({"items": [self.item]}))
        prior = self.collect()
        self.assertEqual(prior["accepted_uses"], [])
        self.assertEqual(prior["reserved_uses"], [])
        self.assertNotIn(str(owner.resolve()), prior["prior_runs"])

    def test_duplicate_directory_alias_counts_one_owner(self):
        owner = self.kwargs["v38_runs"] / "reserved"
        inventory.reserve(owner, {"root": [self.item]}, self.collect())
        alias = self.root / "v38_alias"
        alias.symlink_to(self.kwargs["v38_runs"], target_is_directory=True)
        prior = inventory.collect_prior(alias, **self.kwargs)
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(len(prior["reserved_uses"]), 1)
        self.assertEqual(prior["accepted_uses"][0]["origin"], "v38_reservation")

    def test_experiment_sibling_is_not_production_inventory(self):
        # A fixed-source diagnostic may replay an occupied item under experiments.
        # Its separate output scope must not create a second production owner.
        owner = self.kwargs["v38_runs"] / "reserved"
        payload = inventory.reserve(owner, {"root": [self.item]}, self.collect())
        experimental = self.current.parent / "experiments" / "fixed_source"
        experimental.mkdir(parents=True)
        payload["run"] = str(experimental.resolve())
        payload["reservations"][0]["run"] = str(experimental.resolve())
        (experimental / "SOURCE_RESERVATIONS.json").write_text(json.dumps(payload))
        prior = self.collect()
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertNotIn(str(experimental.resolve()), prior["prior_runs"])


if __name__ == "__main__":
    unittest.main()
