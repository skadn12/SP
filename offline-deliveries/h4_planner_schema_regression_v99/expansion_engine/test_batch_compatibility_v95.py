"""Offline v95 subset capacity and permanent v94 reservation regressions.

The v94 generation has diagnostic requests only, so a synthetic failed v94 run
checks collection and ownership without writing to any prior delivery. Capacity
uses a fake tokenizer/native probe and never contacts a provider.
"""
import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import capacity_probe as cp
import runner as r


class BatchCompatibilityV95Tests(unittest.TestCase):
    def test_capacity_estimates_only_frozen_selected_roots(self):
        selected = "fixture-selected-root"
        absent = "fixture-unselected-root"
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            r.base.write(run / "sources" / f"{selected}.json", {"turns": [
                {"turn_ref": "P:S01:T001", "speaker": "Elena", "text": "Complete selected source."}]})
            tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: list(range(50)))
            native = SimpleNamespace(load_probe=lambda *args: SimpleNamespace(contract=SimpleNamespace(
                dialogue_rows=lambda source: source["turns"],
                build_prompt=lambda *args, **kwargs: "Complete placeholder prompt.")))
            loader = SimpleNamespace(exec_module=lambda module: None)
            transport = (lambda: {}, None, lambda value: value,
                         lambda *args: {"status": "PASS"},
                         SimpleNamespace(tokenizer_from=lambda path: tokenizer))
            with patch.object(cp.r, "verify", return_value=({"roots": [selected]}, {})), \
                 patch.object(cp.r, "ROOTS", (selected, absent)), \
                 patch.object(cp.r.base, "transport", return_value=transport), \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 patch.object(cp.importlib.util, "spec_from_file_location", return_value=SimpleNamespace(loader=loader)), \
                 patch.object(cp.importlib.util, "module_from_spec", return_value=native):
                result = cp.check(run)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual({row["root_id"] for row in result["solver_estimates"]}, {selected})
            self.assertEqual({row["question_type"] for row in result["solver_estimates"]}, {"multi_select", "ordering"})
            self.assertEqual(len(result["solver_estimates"]), 2)
            self.assertEqual(result["network_requests"], 0)
            self.assertFalse(result["historical_prefix_truncation"])
            self.assertTrue((run / "OFFLINE_CAPACITY_CHECKS.json").is_file())
            self.assertFalse((run / "sources" / f"{absent}.json").exists())

    def test_failed_v94_reservations_cannot_be_reassigned_in_v95(self):
        inventory = r.source_inventory
        self.assertEqual(inventory.V94_RUNS, r.DELIVERIES / "h4_dependency_plan_field_evidence_minfix_v94/expansion_engine/runs")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            empty, v94_runs, v95_runs = directory / "empty", directory / "v94", directory / "v95"
            empty.mkdir()
            parameters = inspect.signature(inventory.collect_prior).parameters
            kwargs = {name: empty for name in parameters if name.endswith("_runs")}
            kwargs.update(current_runs=v95_runs, v94_runs=v94_runs)
            old_run = v94_runs / "failed_root"
            item = {"dataset": "fixture_dataset", "split": "train", "item_id": "fixture_event",
                    "source_id": "fixture:train:event", "source_item_sha256": "a" * 64}
            prior = inventory.collect_prior(**kwargs)
            self.assertEqual(prior["accepted_uses"], [])
            reserved = inventory.reserve(old_run, {"fixture-root": [item]}, prior)
            # Simulate interruption before REPORT.json or accepted output.
            self.assertFalse((old_run / "REPORT.json").exists())
            occupied = inventory.collect_prior(**kwargs)
            self.assertEqual(len(occupied["accepted_uses"]), 1)
            self.assertEqual(occupied["accepted_uses"][0]["origin"], "v94_reservation")
            self.assertEqual(occupied["accepted_uses"][0]["status"], "reserved")
            self.assertEqual(occupied["accepted_uses"][0]["run"], str(old_run.resolve()))
            with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                inventory.validate_available({"another-root": [item]}, occupied)
            self.assertEqual(inventory.reserve(old_run, {"fixture-root": [item]}, occupied), reserved)
            # Two collector arguments pointing at the same historical directory
            # must not double-count or free the original reservation.
            kwargs["current_runs"] = v94_runs
            duplicated = inventory.collect_prior(**kwargs)
            self.assertEqual(len(duplicated["accepted_uses"]), 1)
            self.assertEqual(len(duplicated["reserved_uses"]), 1)


if __name__ == "__main__":
    unittest.main()
