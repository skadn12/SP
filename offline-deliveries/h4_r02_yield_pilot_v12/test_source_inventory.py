"""Source consumption checks; no model calls and no writes to prior runs."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import source_inventory as inventory


def source(item_id="one"):
    return {"dataset": "fixture/events", "split": "train", "item_id": item_id,
            "source_id": "fixture:train:" + item_id, "source_item_sha256": "a" * 64,
            "source_file_sha256": "b" * 64}


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.runs = self.base / "runs"

    def prior(self):
        return inventory.collect_prior(self.runs, v1_runs=self.base / "v1",
                                       v2_runs=self.base / "v2", v4_runs=self.base / "v4",
                                       v5_runs=self.base / "v5", v6_runs=self.base / "v6",
                                       v7_runs=self.base / "v7", v8_runs=self.base / "v8",
                                       v9_runs=self.base / "v9", v11_runs=self.base / "v11")

    def reserve(self, name="run1", root="root1", item=None):
        return inventory.reserve(self.runs / name, {root: [item or source()]}, self.prior())

    def test_new_event_reserves_before_any_success(self):
        self.assertEqual(self.prior()["accepted_uses"], [])
        payload = self.reserve()
        self.assertEqual(payload["reservations"][0]["scene_id"], "S01")
        self.assertFalse((self.runs / "run1" / "REPORT.json").exists())
        prior = self.prior()
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(prior["accepted_uses"][0]["status"], "reserved")
        inventory.validate_available({"root1": [source("two")]}, prior)

    def test_same_source_cannot_be_allocated_across_roots_in_one_batch(self):
        with self.assertRaisesRegex(ValueError, "source_event_assigned_twice"):
            inventory.reserve(self.runs / "bad", {"root1": [source()], "root2": [source()]}, self.prior())
        self.assertFalse((self.runs / "bad" / "SOURCE_RESERVATIONS.json").exists())

    def test_failure_does_not_release_for_another_run_or_root(self):
        self.reserve()
        (self.runs / "run1" / "REPORT.json").write_text(json.dumps({"status": "FAIL", "accepted_uses": []}))
        for name, root in (("run2", "root1"), ("run2", "root2")):
            with self.subTest(name=name, root=root):
                with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                    self.reserve(name, root)
                self.assertFalse((self.runs / name / "SOURCE_RESERVATIONS.json").exists())

    def test_retry_same_slot_reuses_identical_file_without_rewriting(self):
        initial = self.reserve()
        path = self.runs / "run1" / "SOURCE_RESERVATIONS.json"
        before = path.stat().st_mtime_ns, path.read_bytes()
        self.assertEqual(self.reserve(), initial)
        self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)
        self.assertEqual(len(self.prior()["accepted_uses"]), 1)
        with self.assertRaisesRegex(ValueError, "existing_reservations_changed"):
            self.reserve(root="different-root")
        changed = source()
        changed["scene_id"] = "S02"
        with self.assertRaisesRegex(ValueError, "existing_reservations_changed"):
            self.reserve(item=changed)

    def test_card_id_and_span_do_not_make_used_event_new(self):
        self.reserve()
        card = {"card_id": "new-card", "source_id": source()["source_id"],
                "lineage": {**source(), "source_span": {"turn_start": 3, "turn_end": 8}}}
        self.assertEqual(inventory.event_key(card), inventory.event_key(source()))
        with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
            inventory.validate_available({"root2": [card]}, self.prior())

    def test_source_hash_change_is_not_a_new_event(self):
        self.reserve()
        changed = source()
        changed["source_item_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "source_event_hash_changed"):
            inventory.validate_available({"root2": [changed]}, self.prior())
        card = {"card_id": "card", "source_id": changed["source_id"], "lineage": changed}
        with self.assertRaisesRegex(ValueError, "allocated_card_source_hash_changed"):
            inventory.verify_cards_for_root([card], {"assignments": {"root1": [source()]}}, "root1")

    def test_accepted_and_reserved_records_count_once(self):
        self.reserve()
        report = {"accepted_uses": [{"dataset": source()["dataset"], "split": "train", "item_id": "one",
                                    "source_id": source()["source_id"], "root_id": "root1", "scene_id": "S01",
                                    "card_id": "c1"}]}
        for filename in ("REPORT.json", "UNIQUE_EVENT_REPORT.json"):
            (self.runs / "run1" / filename).write_text(json.dumps(report))
        prior = self.prior()
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(len(prior["reserved_uses"]), 1)
        self.assertEqual(prior["accepted_uses"][0]["status"], "accepted")
        with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
            self.reserve("later")

    def test_report_cannot_move_accepted_event_to_other_scene(self):
        self.reserve()
        item = {**source(), "root_id": "root1", "scene_id": "S02"}
        (self.runs / "run1" / "REPORT.json").write_text(json.dumps({"accepted_uses": [item]}))
        with self.assertRaisesRegex(ValueError, "source_event_consumed_by_multiple_slots"):
            self.prior()

    def test_legacy_allocator_excludes_reserved_event(self):
        self.reserve()
        manifest = {"items": [source(), source("two")], "manifest_sha256": "d" * 64}
        allocation = inventory.legacy.allocate(manifest, ["root2"], self.prior())
        self.assertEqual(allocation["assigned_slots"], 1)
        self.assertEqual(allocation["assignments"]["root2"][0]["item_id"], "two")

    def test_duplicate_reservation_files_and_changed_evidence_rejected(self):
        self.reserve()
        prior = self.prior()
        path = self.runs / "run1" / "SOURCE_RESERVATIONS.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "historical_usage_evidence_changed"):
            inventory.reserve(self.runs / "run2", {"root2": [source("two")]}, prior)
        payload = json.loads(path.read_text())
        second = self.runs / "run2"
        second.mkdir()
        payload["run"] = str(second)
        payload["reservations"][0]["run"] = str(second)
        (second / "SOURCE_RESERVATIONS.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "source_event_consumed_by_multiple_slots"):
            self.prior()

    def test_live_without_reservations_is_not_silently_available(self):
        run = self.runs / "interrupted"
        run.mkdir(parents=True)
        (run / "LIVE_STARTED.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "live_run_missing_source_reservations"):
            self.prior()

    def test_historical_recovery_deltas_still_consume_all_six_sources(self):
        # Read-only regression against the actual accepted historical inventory.
        prior = inventory.collect_prior(self.runs, v4_runs=self.base / "v4",
                                        v5_runs=self.base / "v5", v6_runs=self.base / "v6",
                                        v7_runs=self.base / "v7", v8_runs=self.base / "v8",
                                        v9_runs=self.base / "v9", v11_runs=self.base / "v11")
        self.assertEqual(len(prior["accepted_uses"]), 6)
        self.assertEqual({row["item_id"] for row in prior["accepted_uses"]}, {
            "PMUL0843.json", "SSNG0142.json", "SNG1013.json", "6027743", "6027790", "6027845"})
        self.assertEqual(len(prior["prior_runs"]), 5)
        historic = deepcopy(prior["accepted_uses"][0])
        historic.pop("run")
        historic.pop("root_id")
        with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
            inventory.validate_available({"new-root": [historic]}, prior)

    def test_current_and_previous_version_reservations_are_both_consumed(self):
        previous_runs = self.base / "v4"
        previous = previous_runs / "previous_failed"
        inventory.reserve(previous, {"old_root": [source("old")]}, self.prior())
        (previous / "REPORT.json").write_text(json.dumps({"status": "FAIL", "accepted_uses": []}))
        prior = self.prior()
        self.assertEqual({row["item_id"] for row in prior["accepted_uses"]}, {"old"})
        self.assertEqual(prior["accepted_uses"][0]["origin"], "v4_reservation")
        with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
            inventory.reserve(self.runs / "new_bad", {"new_root": [source("old")]}, prior)
        inventory.reserve(self.runs / "new_good", {"new_root": [source("new")]}, prior)
        combined = self.prior()
        self.assertEqual({row["item_id"] for row in combined["accepted_uses"]}, {"old", "new"})
        self.assertEqual({row["origin"] for row in combined["accepted_uses"]},
                         {"v4_reservation", "v12_reservation"})
        self.assertEqual(len(combined["reserved_uses"]), 2)

    def test_same_runs_directory_is_not_counted_twice(self):
        self.reserve()
        prior = inventory.collect_prior(self.runs, v1_runs=self.base / "v1",
                                        v2_runs=self.base / "v2", v4_runs=self.runs,
                                        v5_runs=self.runs, v6_runs=self.runs,
                                        v7_runs=self.runs, v8_runs=self.runs, v9_runs=self.runs, v11_runs=self.runs)
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(len(prior["reserved_uses"]), 1)
        self.assertEqual(len(prior["prior_evidence"]), 1)

    def test_real_v4_reservations_and_acceptance_remain_excluded(self):
        # The historical fixture is read only. All nine reserved v4 events,
        # including eight without accepted scenes, must remain unavailable.
        prior = inventory.collect_prior(self.runs, v5_runs=self.base / "v5", v6_runs=self.base / "v6",
                                        v7_runs=self.base / "v7", v8_runs=self.base / "v8",
                                        v9_runs=self.base / "v9", v11_runs=self.base / "v11")
        self.assertEqual(len(prior["accepted_uses"]), 15)
        self.assertEqual(len(prior["reserved_uses"]), 9)
        self.assertEqual(len(prior["prior_runs"]), 6)
        v4_uses = [row for row in prior["accepted_uses"] if row["origin"].startswith("v4_")]
        self.assertEqual(len(v4_uses), 9)
        self.assertEqual(sum(row["status"] == "accepted" for row in v4_uses), 1)
        self.assertEqual(sum(row["status"] == "reserved" for row in v4_uses), 8)
        self.assertEqual({row["item_id"] for row in v4_uses}, {
            "MUL0358.json", "MUL0928.json", "MUL1681.json", "MUL1914.json",
            "MUL2293.json", "MUL2431.json", "PMUL0121.json", "PMUL0172.json", "PMUL0949.json",
        })
        v4_evidence = [row["path"] for row in prior["prior_evidence"]
                       if str(inventory.V4_RUNS) in row["path"]]
        self.assertEqual(len(v4_evidence), 2)
        self.assertTrue(any(path.endswith("SOURCE_RESERVATIONS.json") for path in v4_evidence))
        self.assertTrue(any(path.endswith("REPORT.json") for path in v4_evidence))
        for row in v4_uses:
            with self.subTest(item=row["item_id"]):
                candidate = {key: value for key, value in row.items() if key not in {"run", "root_id"}}
                with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                    inventory.validate_available({"new-root": [candidate]}, prior)
        actual_manifest = inventory.legacy.read(
            inventory.V4_RUNS / "r02_independent_routes_v4_01" / "dataset_sources.json")
        allocation = inventory.legacy.allocate(actual_manifest, ["new-root"], prior)
        self.assertEqual(allocation["assigned_slots"], 0)

    def test_failed_v5_reservations_remain_consumed_after_version_change(self):
        old_run = self.base / "v5" / "failed_run"
        inventory.reserve(old_run, {"old-root": [source("v5-failed")]}, self.prior())
        (old_run / "REPORT.json").write_text(json.dumps({"status": "FAIL", "accepted_uses": []}))
        prior = self.prior()
        self.assertEqual(len(prior["accepted_uses"]), 1)
        self.assertEqual(prior["accepted_uses"][0]["origin"], "v5_reservation")
        for root, scene in (("old-root", "S01"), ("new-root", "S02")):
            with self.subTest(root=root, scene=scene):
                candidate = {**source("v5-failed"), "scene_id": scene,
                             "card_id": "replacement", "source_span": {"start": 8, "end": 12}}
                with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                    inventory.reserve(self.runs / "v6-failed", {root: [candidate]}, prior)
        self.assertFalse((self.runs / "v6-failed" / "SOURCE_RESERVATIONS.json").exists())
        inventory.reserve(self.runs / "v6-good", {"new-root": [source("v6-new")]}, prior)
        self.assertEqual({row["origin"] for row in self.prior()["accepted_uses"]},
                         {"v5_reservation", "v12_reservation"})

    def test_cross_version_duplicate_reservation_files_fail_closed(self):
        old_run = self.base / "v5" / "old_run"
        payload = inventory.reserve(old_run, {"root": [source()]}, self.prior())
        new_run = self.runs / "duplicate"
        new_run.mkdir(parents=True)
        payload["run"] = str(new_run)
        payload["reservations"][0]["run"] = str(new_run)
        (new_run / "SOURCE_RESERVATIONS.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "source_event_consumed_by_multiple_slots"):
            self.prior()

    def test_failed_v6_and_current_v12_sources_remain_permanently_consumed(self):
        previous = self.base / "v6" / "failed"
        inventory.reserve(previous, {"old-root": [source("v6-failed")]}, self.prior())
        (previous / "REPORT.json").write_text(json.dumps({"status": "FAIL", "accepted_uses": []}))
        inventory.reserve(self.runs / "current", {"new-root": [source("v7-new")]}, self.prior())
        prior = self.prior()
        self.assertEqual({row["origin"] for row in prior["accepted_uses"]},
                         {"v6_reservation", "v12_reservation"})
        for item_id in ("v6-failed", "v7-new"):
            with self.subTest(item_id=item_id), self.assertRaisesRegex(
                    ValueError, "source_event_already_consumed"):
                candidate = {**source(item_id), "card_id": "changed-card", "scene_id": "S02",
                             "source_span": {"start": 2, "end": 4}}
                inventory.reserve(self.runs / "reuse", {"another-root": [candidate]}, prior)

    def test_every_reservation_generation_is_scanned_in_isolated_directories(self):
        for version in ("v4", "v5", "v6", "v7", "v8", "v9", "v11", "v12"):
            runs = self.runs if version == "v12" else self.base / version
            inventory.reserve(runs / "reserved", {"root": [source(version)]}, self.prior())
        prior = self.prior()
        self.assertEqual({row["item_id"] for row in prior["accepted_uses"]},
                         {"v4", "v5", "v6", "v7", "v8", "v9", "v11", "v12"})
        self.assertEqual({row["origin"] for row in prior["accepted_uses"]},
                         {version + "_reservation" for version in ("v4", "v5", "v6", "v7", "v8", "v9", "v11", "v12")})

    def test_failed_v9_reservation_remains_consumed_in_v12(self):
        old_run = self.base / "v9" / "failed"
        inventory.reserve(old_run, {"old-root": [source("v9-failed")]}, self.prior())
        (old_run / "REPORT.json").write_text(json.dumps({"status": "FAIL", "accepted_uses": []}))
        prior = self.prior()
        self.assertEqual(prior["accepted_uses"][0]["origin"], "v9_reservation")
        with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
            inventory.reserve(self.runs / "reuse", {"new-root": [source("v9-failed")]}, prior)
        self.assertFalse((self.runs / "reuse" / "SOURCE_RESERVATIONS.json").exists())

    def test_real_inventory_through_v11_excludes_all_33_used_sources(self):
        # Only the current generation is isolated. Historical files are read
        # without modification, including all nine unsuccessful v5 B slots and the v6/v7/v8 B sources.
        from source_pool import freeze_unused_source_subset

        prior = inventory.collect_prior(self.runs)
        self.assertEqual(len(prior["accepted_uses"]), 33)
        self.assertEqual(len(prior["reserved_uses"]), 27)
        self.assertEqual(len(prior["prior_runs"]), 13)  # v11 expansion and empty QA report
        v11_uses = [row for row in prior["accepted_uses"] if row["origin"] == "v11_reservation"]
        self.assertEqual(len(v11_uses), 5)
        self.assertTrue(all(row["status"] == "reserved" for row in v11_uses))
        v9_uses = [row for row in prior["accepted_uses"] if row["origin"].startswith("v9_")]
        self.assertEqual(len(v9_uses), 1)
        self.assertEqual(v9_uses[0]["status"], "accepted")
        self.assertEqual(v9_uses[0]["item_id"], "PMUL3484.json")
        v6_uses = [row for row in prior["accepted_uses"] if row["origin"].startswith("v6_")]
        self.assertEqual(len(v6_uses), 1)
        self.assertEqual(v6_uses[0]["status"], "accepted")
        v7_uses = [row for row in prior["accepted_uses"] if row["origin"].startswith("v7_")]
        self.assertEqual(len(v7_uses), 1)
        v8_uses = [row for row in prior["accepted_uses"] if row["origin"].startswith("v8_")]
        self.assertEqual(len(v8_uses), 1)
        self.assertEqual(v8_uses[0]["status"], "reserved")
        v5_uses = [row for row in prior["accepted_uses"] if row["origin"].startswith("v5_")]
        self.assertEqual(len(v5_uses), 9)
        self.assertTrue(all(row["status"] == "reserved" for row in v5_uses))
        self.assertEqual({row["item_id"] for row in v5_uses}, {
            "PMUL1031.json", "PMUL1146.json", "PMUL1251.json", "PMUL1586.json",
            "PMUL1842.json", "PMUL2097.json", "PMUL2460.json", "PMUL2583.json", "PMUL2710.json",
        })
        v5_evidence = [row["path"] for row in prior["prior_evidence"]
                       if str(inventory.V5_RUNS) in row["path"]]
        self.assertEqual(len(v5_evidence), 2)
        self.assertTrue(any(path.endswith("SOURCE_RESERVATIONS.json") for path in v5_evidence))
        self.assertTrue(any(path.endswith("REPORT.json") for path in v5_evidence))
        for row in prior["accepted_uses"]:
            with self.subTest(item=row["item_id"]):
                candidate = {key: value for key, value in row.items() if key not in {"run", "root_id"}}
                with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                    inventory.reserve(self.runs / "forbidden", {"new-root": [candidate]}, prior)
        self.assertFalse((self.runs / "forbidden" / "SOURCE_RESERVATIONS.json").exists())
        consumed = {inventory.event_key(row) for row in prior["accepted_uses"]}
        available = freeze_unused_source_subset(consumed, count=1000)
        self.assertEqual(len(available["items"]), 37)
        self.assertFalse(consumed & {inventory.event_key(item) for item in available["items"]})
        entire_pool = freeze_unused_source_subset(set(), count=1000)
        self.assertEqual(len(entire_pool["items"]), 70)
        allocation = inventory.legacy.allocate(entire_pool, ["new-root"], prior, slots_per_root=1000)
        self.assertEqual(allocation["assigned_slots"], 37)
        self.assertEqual(len(allocation["excluded_prior_source_ids"]), 33)
        self.assertEqual({inventory.event_key(item) for item in allocation["assignments"]["new-root"]},
                         {inventory.event_key(item) for item in available["items"]})


if __name__ == "__main__":
    unittest.main()
