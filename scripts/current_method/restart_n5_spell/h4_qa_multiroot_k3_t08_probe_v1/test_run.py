"""Zero-API tests for the inventory and root-preparation stages."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from .run import _prepare_root, _selected_entries, inventory_phase


class RunPlanningTests(unittest.TestCase):
    def _expanded(self, root_id: str = "test-root") -> dict[str, object]:
        return {
            "root_id": root_id,
            "turns": [
                {"turn_ref": "D001", "speaker": "Maya", "text": "I am considering a new path."},
                {"turn_ref": "D002", "speaker": "Jordan", "text": "What would you like to test?"},
                {"turn_ref": "D003", "speaker": "Maya", "text": "I can try a bakery shift."},
            ],
        }

    def _inventory_args(self, run_root: Path, source: Path, *, max_roots: int = 1) -> Namespace:
        return Namespace(
            run_root=run_root,
            root_source=[source],
            max_roots=max_roots,
            _argv=["--phase", "inventory"],
        )

    def test_inventory_is_offline_and_freezes_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "expanded.json"
            source.write_text(json.dumps(self._expanded()), encoding="utf-8")
            output = root / "inventory"
            result = inventory_phase(self._inventory_args(output, source))
            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(result["solver_calls"], 0)
            self.assertEqual(result["root_count"], 1)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_root_ids"], ["test-root"])
            self.assertEqual(_selected_entries(manifest)[0]["root_id"], "test-root")

    def test_prepare_root_writes_clean_view_and_checkpoint_without_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "expanded.json"
            source.write_text(json.dumps(self._expanded()), encoding="utf-8")
            inventory_dir = root / "inventory"
            inventory_phase(self._inventory_args(inventory_dir, source))
            manifest = json.loads((inventory_dir / "manifest.json").read_text(encoding="utf-8"))
            entry = _selected_entries(manifest)[0]
            run_root = root / "run"
            root_dir, cleaned = _prepare_root(
                run_root, entry, manifest_sha=manifest["manifest_sha256"]
            )
            self.assertEqual(cleaned["root_id"], "test-root")
            self.assertTrue((root_dir / "expanded_dialogue_clean.json").is_file())
            checkpoint = json.loads((root_dir / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["stage"], "EXPANSION_READY")

    def test_prepare_root_rejects_source_drift_after_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "expanded.json"
            source.write_text(json.dumps(self._expanded()), encoding="utf-8")
            inventory_dir = root / "inventory"
            inventory_phase(self._inventory_args(inventory_dir, source))
            manifest = json.loads((inventory_dir / "manifest.json").read_text(encoding="utf-8"))
            entry = _selected_entries(manifest)[0]
            source.write_text(json.dumps(self._expanded() | {"extra": True}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source_hash_drift"):
                _prepare_root(root / "run", entry, manifest_sha=manifest["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
