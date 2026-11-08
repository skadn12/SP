"""Candidate collection preserves global source consumption across versions.

Every inventory directory and reservation is isolated under TemporaryDirectory.
These fixtures never inspect or reserve actual production sources, and cannot
call a model. Candidate retention or training admission does not free a source.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import source_inventory as inventory


class CandidateInventoryV41Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="v41_inventory_MOCK_ONLY_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.current = self.root / "v41" / "runs"
        self.prior_paths = {
            name: self.root / name
            for name in inspect.signature(inventory.collect_prior).parameters
            if name != "current_runs"
        }
        self.versions = {
            39: self.prior_paths["v39_runs"],
            40: self.prior_paths["v40_runs"],
            41: self.current,
        }
        sockets = patch("socket.socket", side_effect=AssertionError("offline_inventory_only"))
        sockets.start()
        self.addCleanup(sockets.stop)

    def collect(self, *, current=None, prior_paths=None):
        return inventory.collect_prior(
            current if current is not None else self.current,
            **(self.prior_paths if prior_paths is None else prior_paths),
        )

    @staticmethod
    def item(identity):
        return {
            "dataset": "MOCK_ONLY", "split": "train", "item_id": identity,
            "source_id": "MOCK_ONLY:train:" + identity,
            "source_item_sha256": "a" * 64,
        }

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def test_failed_v39_v40_v41_reservations_stay_consumed(self):
        originals = {}
        for version, directory in self.versions.items():
            owner = directory / "failed_candidate"
            item = self.item(f"failed-v{version}")
            payload = inventory.reserve(owner, {"root": [item]}, self.collect())
            self.write(owner / "LIVE_STARTED.json", {})
            self.write(owner / "REPORT.json", {
                "status": "COLLECTION_SHORTFALL", "accepted_uses": [],
                "retained_candidates": 0, "training_pool_eligible": False,
            })
            originals[version] = (owner, item, payload,
                                  (owner / "SOURCE_RESERVATIONS.json").read_bytes())

        prior = self.collect()
        self.assertEqual(len(prior["accepted_uses"]), 3)
        self.assertEqual(len(prior["reserved_uses"]), 3)
        self.assertEqual({row["origin"] for row in prior["accepted_uses"]},
                         {"v39_reservation", "v40_reservation", "v41_reservation"})
        self.assertTrue(all(row["status"] == "reserved" for row in prior["accepted_uses"]))
        for version, (owner, item, payload, before) in originals.items():
            with self.subTest(version=version):
                self.assertEqual(inventory.reserve(owner, {"root": [item]}, prior), payload)
                self.assertEqual((owner / "SOURCE_RESERVATIONS.json").read_bytes(), before)
                with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                    inventory.reserve(self.current / f"replacement_for_v{version}",
                                      {"other_root": [item]}, prior)

    def test_prepare_only_v39_v40_v41_does_not_consume_source(self):
        for version, directory in self.versions.items():
            owner = directory / "prepared_only"
            item = self.item(f"prepared-v{version}")
            self.write(owner / "MANIFEST.json", {
                "version": f"h4_sequential_ab_v{version}",
                "training_pool_eligible": False,
            })
            self.write(owner / "dataset_sources.json", {"items": [item]})
            self.write(owner / "UNIQUE_EVENT_ALLOCATION.json", {
                "assignments": {"root": [item]},
            })
        prior = self.collect()
        self.assertEqual(prior["accepted_uses"], [])
        self.assertEqual(prior["reserved_uses"], [])
        self.assertEqual(prior["prior_runs"], [])
        # A prepared allocation is still available for a subsequently started run.
        inventory.reserve(self.current / "started", {"root": [self.item("prepared-v39")]}, prior)
        self.assertEqual(len(self.collect()["accepted_uses"]), 1)

    def test_reservation_and_two_historical_reports_count_one_source(self):
        owner = self.versions[39] / "accepted_historical"
        payload = inventory.reserve(owner, {"root": [self.item("historical")]}, self.collect())
        accepted = {**payload["reservations"][0], "card_id": "MOCK_ONLY_CARD"}
        for filename in ("REPORT.json", "UNIQUE_EVENT_REPORT.json"):
            self.write(owner / filename, {"accepted_uses": [accepted]})
        prior = self.collect()
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(len(prior["reserved_uses"]), 1)
        self.assertEqual(prior["accepted_uses"][0]["status"], "accepted")
        self.assertEqual(prior["accepted_uses"][0]["origin"], "v39_accepted")
        self.assertEqual(len(prior["prior_evidence"]), 3)
        with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
            inventory.reserve(self.current / "candidate_run", {"root": [self.item("historical")]}, prior)

    def test_directory_aliases_do_not_double_count_historical_reservations(self):
        owner = self.versions[39] / "reserved"
        inventory.reserve(owner, {"root": [self.item("alias")]}, self.collect())
        alias = self.root / "v39_alias"
        alias.symlink_to(self.versions[39], target_is_directory=True)
        paths = {**self.prior_paths, "v40_runs": alias}
        prior = self.collect(current=alias, prior_paths=paths)
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(len(prior["reserved_uses"]), 1)
        self.assertEqual(prior["prior_runs"], [str(owner.resolve())])

    def test_conflicting_v41_owner_cannot_hide_behind_historical_reservation(self):
        old_owner = self.versions[40] / "reserved"
        payload = inventory.reserve(old_owner, {"root": [self.item("duplicate")]}, self.collect())
        new_owner = self.current / "corrupted_duplicate"
        invalid = deepcopy(payload)
        invalid["run"] = str(new_owner.resolve())
        invalid["reservations"][0]["run"] = str(new_owner.resolve())
        self.write(new_owner / "SOURCE_RESERVATIONS.json", invalid)
        with self.assertRaisesRegex(ValueError, "source_event_consumed_by_multiple_slots"):
            self.collect()

    def test_live_without_reservation_is_rejected_in_each_generation(self):
        for version, directory in self.versions.items():
            owner = directory / "missing_reservation"
            self.write(owner / "LIVE_STARTED.json", {})
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "live_run_missing_source_reservations"):
                    self.collect()
            (owner / "LIVE_STARTED.json").unlink()

    def test_retained_candidate_report_never_promotes_reserved_source_to_accepted(self):
        owner = self.current / "pending_candidate"
        inventory.reserve(owner, {"root": [self.item("pending")]}, self.collect())
        self.write(owner / "REPORT.json", {
            "status": "CANDIDATE_COLLECTION_COMPLETE", "accepted_uses": [],
            "retained_candidates": 1, "training_pool_eligible": False,
            "candidate_metadata": [{"scene_id": "S01", "status": "PENDING_REVIEW"}],
        })
        prior = self.collect()
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(prior["accepted_uses"][0]["status"], "reserved")
        with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
            inventory.reserve(self.current / "later", {"root": [self.item("pending")]}, prior)


if __name__ == "__main__":
    unittest.main()
