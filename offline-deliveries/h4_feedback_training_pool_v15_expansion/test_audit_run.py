"""Audit runtime isolation checks; no model calls or historical writes."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import audit_run as audit


class AuditRuntimeTests(unittest.TestCase):
    def test_runtime_is_loaded_from_current_delivery_despite_cached_runner(self):
        old_runner = SimpleNamespace(__file__="/old-delivery/runner.py")
        spec = importlib.util.spec_from_file_location("isolated_v15_audit_test", Path(audit.__file__))
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"runner": old_runner}):
            spec.loader.exec_module(module)
        self.assertEqual(Path(module.r.__file__).resolve(), audit.HERE / "runner.py")
        self.assertEqual(module.r.source_inventory.HERE, audit.HERE)
        self.assertEqual(module.r.LOCK_PATH,
                         module.r.source_inventory.V2_RUNS / ".unique_event_live.lock")

    def test_cached_historical_validation_module_is_rejected(self):
        spec = importlib.util.spec_from_file_location("old_dependency_v15_audit_test", Path(audit.__file__))
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"source_inventory": SimpleNamespace(
                __file__="/old-delivery/source_inventory.py")}):
            with self.assertRaisesRegex(RuntimeError, "audit_dependency_outside_current_delivery"):
                spec.loader.exec_module(module)

    def test_historical_and_other_delivery_paths_are_rejected_before_reading(self):
        candidates = [audit.HERE.parent / "h4_r02_reference_contract_v9/runs/history",
                      audit.HERE / "other_runs/test", audit.HERE / "runs/nested/test"]
        with patch.object(audit.r.base, "read", side_effect=AssertionError("unexpected read")):
            for path in candidates:
                with self.subTest(path=path), self.assertRaisesRegex(
                        ValueError, "audit_run_must_belong_to_current_v15_runs"):
                    audit.audit_run(path)

    def test_symlink_to_historical_run_is_not_a_current_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "current/runs").mkdir(parents=True)
            (root / "history/runs/old").mkdir(parents=True)
            (root / "current/runs/alias").symlink_to(root / "history/runs/old", target_is_directory=True)
            with patch.object(audit, "HERE", root / "current"):
                with self.assertRaisesRegex(ValueError, "audit_run_must_belong_to_current_v15_runs"):
                    audit.audit_run(root / "current/runs/alias")


if __name__ == "__main__":
    unittest.main()
