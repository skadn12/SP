import inspect
import json
from pathlib import Path
import tempfile
import unittest

import source_inventory as inventory


class V36ReservationHistoryTests(unittest.TestCase):
    def test_failed_v36_and_current_v38_reservations_remain_consumed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kwargs = {
                name: root / name
                for name in inspect.signature(inventory.collect_prior).parameters
                if name != "current_runs"
            }
            current = root / "v38"
            failed_owner = kwargs["v36_runs"] / "failed_before_acceptance"
            item = {"dataset": "fixture", "split": "train", "item_id": "failed-v36",
                    "source_id": "fixture:failed-v36", "source_item_sha256": "a" * 64}
            inventory.reserve(failed_owner, {"r": [item]},
                              inventory.collect_prior(current, **kwargs))
            (failed_owner / "LIVE_STARTED.json").write_text("{}\n")
            (failed_owner / "REPORT.json").write_text(json.dumps({
                "status": "B_SHORTFALL", "accepted_uses": [],
            }))
            prior = inventory.collect_prior(current, **kwargs)
            self.assertEqual(len(prior["accepted_uses"]), 1)
            self.assertEqual(prior["accepted_uses"][0]["origin"], "v36_reservation")
            self.assertEqual(prior["accepted_uses"][0]["status"], "reserved")
            with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                inventory.reserve(current / "new_owner", {"r": [item]}, prior)
            inventory.reserve(failed_owner, {"r": [item]}, prior)
            new_item = dict(item, item_id="new-v38", source_id="fixture:new-v38")
            inventory.reserve(current / "current_owner", {"r": [new_item]}, prior)
            occupied = inventory.collect_prior(current, **kwargs)
            self.assertEqual(len(occupied["accepted_uses"]), 2)
            self.assertEqual({row["origin"] for row in occupied["accepted_uses"]},
                             {"v36_reservation", "v39_reservation"})

    def test_v36_live_run_without_reservation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kwargs = {
                name: root / name
                for name in inspect.signature(inventory.collect_prior).parameters
                if name != "current_runs"
            }
            incomplete = kwargs["v36_runs"] / "interrupted"
            incomplete.mkdir(parents=True)
            (incomplete / "LIVE_STARTED.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "live_run_missing_source_reservations"):
                inventory.collect_prior(root / "v38", **kwargs)


if __name__ == "__main__":
    unittest.main()
