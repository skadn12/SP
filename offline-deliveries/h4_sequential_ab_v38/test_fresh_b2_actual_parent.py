"""Read-only actual A3 admission plus temporary preparation; never live or reserve."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoTokenizer
import fresh_b2 as b

r = b.r


class ActualHistoricalA3Tests(unittest.TestCase):
    def test_exact_audited_v34_A3_is_admitted_without_changes(self):
        parent = r.AUDITED_V34_A3_RUN
        before = r.base.bindings([p for p in parent.rglob("*") if p.is_file()])
        with patch("socket.socket", side_effect=AssertionError("network_forbidden")):
            admitted = r.checked_parent_a3(parent, r.DEFAULT_ROOTS[0])
        history = r.base.read(admitted["cumulative_path"])["turns"]
        self.assertEqual(len(history), 132)
        self.assertEqual(r.base.digest(history), r.AUDITED_V34_A3_HISTORY_SHA256)
        self.assertEqual(admitted["result"]["accepted_scenes"], 3)
        r.base.verify_bindings(before)

    def test_other_historical_directory_is_not_admitted(self):
        with self.assertRaisesRegex(ValueError, "parent_A3_outside_current_delivery"):
            r.checked_parent_a3(r.AUDITED_V34_A3_RUN.parent / "another_A3", r.DEFAULT_ROOTS[0])

    def test_actual_A3_and_sources_prepare_only_in_temporary_delivery(self):
        original_here = r.HERE
        source_dir = original_here.parent / "h4_external_event_source_refresh_v1"
        prior = r.source_inventory.collect_prior()
        with tempfile.TemporaryDirectory(prefix="v38_actual_A3_prepare_only_") as folder:
            delivery = Path(folder) / "offline-deliveries/h4_sequential_ab_v38"
            delivery.mkdir(parents=True)
            for path in original_here.glob("*.py"):
                shutil.copy2(path, delivery / path.name)
            shutil.copy2(original_here / "POLICY.md", delivery / "POLICY.md")
            unique = delivery.parent / "h4_r02_source_unique_v2"
            unique.mkdir()
            shutil.copy2(original_here.parent / "h4_r02_source_unique_v2/unique_events.py", unique / "unique_events.py")
            run = delivery / "runs/PREPARE_ONLY"
            with patch.object(r, "HERE", delivery), \
                 patch.object(r.Caller, "call", side_effect=AssertionError("no_model_calls")), \
                 patch("socket.socket", side_effect=AssertionError("network_forbidden")):
                prepared = b.prepare(run, r.AUDITED_V34_A3_RUN, source_dir)
                manifest = b.verify(run)
            self.assertEqual(prepared["assigned_B_slots"], 2)
            self.assertEqual(manifest["logical_max"], 34)
            self.assertEqual(manifest["http_max"], 68)
            self.assertEqual(manifest["inherited_A"]["history_sha256"], r.AUDITED_V34_A3_HISTORY_SHA256)
            self.assertFalse((run / "SOURCE_RESERVATIONS.json").exists())
            self.assertFalse((run / "LIVE_STARTED.json").exists())
            self.assertFalse((run / "requests").exists())
            self.assertEqual([row["source_id"] for row in r.base.read(run / "dataset_sources.json")["items"]],
                             ["multiwoz22:train:PMUL3559.json", "soda:train:6028248"])
        self.assertEqual(r.source_inventory.collect_prior(), prior)


if __name__ == "__main__":
    unittest.main()
