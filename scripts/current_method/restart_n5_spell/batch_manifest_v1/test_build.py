from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ..input_adapter import dialogue_content_hash
from ..unified_dialogue_root import validate_unified_dialogue_root
from .build import build_selection_manifest, write_selection_manifest


def _root(root_id: str, source_path: Path) -> dict:
    turns = [{
        "turn_id": f"{root_id}:T001", "session_id": f"{root_id}:S01",
        "speaker": "Alex", "text": "The plan remains open.",
        "partition": "parent", "source_id": root_id,
        "source_session_index": 1, "source_turn_index": 1,
    }]
    return {
        "schema_version": "unified_dialogue_root_v1", "root_id": root_id,
        "sample_id": root_id, "lineage_group": root_id,
        "source": {
            "dataset": "fixture", "split": "test", "source_id": root_id,
            "source_row_index": 0, "source_path": str(source_path.resolve()),
            "source_file_sha256": "0" * 64, "source_record_sha256": "1" * 64,
            "source_record_sha256_scope": "fixture",
        },
        "dialogue": {
            "dialogue_hash": dialogue_content_hash(turns),
            "session_count": 1, "turn_count": 1, "turns": turns,
        },
        "session_metadata": [{
            "session_id": turns[0]["session_id"], "source_session_index": 1,
            "time_interval": "day 1", "summary": "A short exchange.",
            "turn_ids": [turns[0]["turn_id"]],
        }],
        "source_metadata": {"fixture": True},
    }


def _clean_manifest(base: Path, root: dict) -> Path:
    root_path = base / "roots" / root["root_id"] / "unified_root.json"
    root_path.parent.mkdir(parents=True)
    root_path.write_text(json.dumps(root), encoding="utf-8")
    payload = {
        "manifest_version": "named_input_cleaning_batch_manifest_v1",
        "status": "passed", "expected_count": 1, "record_count": 1,
        "roots_directory": str(root_path.parent.parent),
        "global_checks": {
            "all_records_valid": True, "all_repeatable": True,
            "expected_count": True, "mapping_coverage_exact": True,
            "names_unique_within_each_root": True, "root_ids_unique": True,
            "source_unchanged": True,
        },
        "records": [{
            "root_id": root["root_id"], "output_path": str(root_path),
            "output_sha256": hashlib.sha256(root_path.read_bytes()).hexdigest(),
        }],
    }
    path = base / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class SelectionManifestTests(unittest.TestCase):
    def test_collapses_identical_cumulative_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.jsonl"
            source.write_text("fixture\n", encoding="utf-8")
            root = _root("cc-train-episode-1", source)
            first = _clean_manifest(base / "first", root)
            selection = build_selection_manifest([first], expected_count=1)
            selection_path = write_selection_manifest(base / "selection.json", selection)
            manifest = build_selection_manifest(
                [selection_path, first], expected_count=1)
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual(len(manifest["duplicate_sources_collapsed"]), 1)
            self.assertTrue(manifest["global_checks"]["no_data_copied"])
            self.assertTrue(validate_unified_dialogue_root(root)["valid"])

    def test_writes_only_manifest_and_rejects_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.jsonl"
            source.write_text("fixture\n", encoding="utf-8")
            manifest = build_selection_manifest(
                [_clean_manifest(base / "source", _root("r1", source))],
                expected_count=1,
            )
            output = write_selection_manifest(base / "selection.json", manifest)
            self.assertTrue(output.is_file())
            self.assertEqual(len(list(base.rglob("unified_root.json"))), 1)
            with self.assertRaises(FileExistsError):
                write_selection_manifest(output, manifest)

    def test_can_select_a_small_root_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.jsonl"
            source.write_text("fixture\n", encoding="utf-8")
            first = _root("r1", source)
            second = _root("r2", source)
            manifest_path = _clean_manifest(base / "source", first)
            # Add a second record to the fixture manifest to exercise filtering.
            second_path = base / "source" / "roots" / "r2" / "unified_root.json"
            second_path.parent.mkdir(parents=True)
            second_path.write_text(json.dumps(second), encoding="utf-8")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["expected_count"] = payload["record_count"] = 2
            payload["records"].append({
                "root_id": "r2", "output_path": str(second_path),
                "output_sha256": hashlib.sha256(second_path.read_bytes()).hexdigest(),
            })
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            selected = build_selection_manifest(
                [manifest_path], expected_count=1, root_ids=["r2"])
            self.assertEqual(selected["record_count"], 1)
            self.assertEqual(selected["selection_root_ids"], ["r2"])


if __name__ == "__main__":
    unittest.main()
