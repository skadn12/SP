import inspect
from pathlib import Path
import tempfile
import unittest

import source_inventory as inventory


class RecentReservationTests(unittest.TestCase):
    def test_v34_v35_failed_reservations_survive_version_upgrade(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kwargs = {name: root / name for name in inspect.signature(inventory.collect_prior).parameters if name != "current_runs"}
            current = root / "v37"
            items = []
            for version in (34, 35):
                item = {"dataset": "fixture", "split": "train", "item_id": f"failed-{version}",
                        "source_id": f"fixture:failed-{version}", "source_item_sha256": str(version // 10) * 64}
                owner = kwargs[f"v{version}_runs"] / "failed_before_report"
                inventory.reserve(owner, {"r": [item]}, inventory.collect_prior(current, **kwargs))
                items.append(item)
            prior = inventory.collect_prior(current, **kwargs)
            self.assertEqual({x["origin"] for x in prior["accepted_uses"]}, {"v34_reservation", "v35_reservation"})
            for item in items:
                with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                    inventory.reserve(current / "new_owner", {"r": [item]}, prior)


if __name__ == "__main__":
    unittest.main()
