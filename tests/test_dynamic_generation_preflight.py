from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.current_method.restart_n5_spell import dynamic_generation_preflight as dgp
from scripts.current_method.restart_n5_spell.core import atomic_json
from scripts.current_method.restart_n5_spell.input_adapter import dialogue_content_hash
from scripts.current_method.restart_n5_spell.stateful_outline_planner import (
    build_dynamic_stateful_planner_input,
    build_stateful_outline_overview_prompt,
)
from scripts.current_method.restart_n5_spell.unified_dialogue_root import (
    UNIFIED_DIALOGUE_ROOT_VERSION, validate_unified_dialogue_root,
)


def _root(index: int, *, sessions: int = 5,
          dataset: str = "conversation_chronicles") -> dict:
    root_id = f"test-root-{index}"
    source_id = f"source-{index}"
    turns, metadata = [], []
    for session_index in range(1, sessions + 1):
        session_id = f"{root_id}:parent_session_{session_index:02d}"
        turn_ids = []
        for turn_index in range(1, 3):
            turn_id = f"{root_id}:P{session_index:02d}:T{turn_index:03d}"
            turn_ids.append(turn_id)
            turns.append({
                "turn_id": turn_id, "session_id": session_id,
                "speaker": "Alex" if turn_index == 1 else "Blair",
                "text": f"Dialogue {index}-{session_index}-{turn_index}.",
                "partition": "parent", "source_id": source_id,
                "source_session_index": session_index,
                "source_turn_index": turn_index,
            })
        metadata.append({
            "session_id": session_id,
            "source_session_index": session_index,
            "time_interval": f"SENTINEL_TIME_{session_index}",
            "summary": f"SENTINEL_SUMMARY_{session_index}",
            "turn_ids": turn_ids,
        })
    return {
        "schema_version": UNIFIED_DIALOGUE_ROOT_VERSION,
        "root_id": root_id, "sample_id": root_id, "lineage_group": root_id,
        "source": {
            "dataset": dataset, "split": "train", "source_id": source_id,
            "source_row_index": index, "source_path": f"/tmp/source-{index}.json",
            "source_file_sha256": "1" * 64,
            "source_record_sha256": "2" * 64,
            "source_record_sha256_scope": "test_fixture_bytes",
        },
        "dialogue": {
            "dialogue_hash": dialogue_content_hash(turns),
            "session_count": sessions, "turn_count": len(turns), "turns": turns,
        },
        "session_metadata": metadata,
        "source_metadata": {"relationship": "test peers"},
    }


def _write_locomo_source(path: Path) -> None:
    sessions = ["session_1", "session_2", "session_3"]
    turns = [{
        "turn_id": f"D{session_index}:{turn_index}",
        "session_id": session_id,
        "speaker": "John" if turn_index == 1 else "Maria",
        "text": f"LoCoMo turn {session_index}-{turn_index}.",
        "partition": "parent", "session_date_time": f"day {session_index}",
    } for session_index, session_id in enumerate(sessions, 1)
      for turn_index in range(1, 3)]
    atomic_json(path, {
        "sample_id": "conv-test", "source_session_ids": sessions,
        "source_session_count": len(sessions), "source_turn_count": len(turns),
        "source_child_turn_count": 0, "turns": turns,
        "dialogue_sha256": dialogue_content_hash(turns),
    })


class DynamicGenerationPreflightTests(unittest.TestCase):
    def test_unified_validator_supports_dynamic_session_counts(self) -> None:
        root = _root(1, sessions=9, dataset="locomo")
        self.assertTrue(validate_unified_dialogue_root(root)["valid"])
        result = validate_unified_dialogue_root(root, expected_session_count=5)
        self.assertFalse(result["valid"])
        self.assertIn("expected_session_count", result["errors"])

    def test_dynamic_planner_builder_preserves_existing_prompt(self) -> None:
        root = _root(2)
        turns = [{key: turn[key] for key in (
            "turn_id", "session_id", "speaker", "text", "partition")}
            for turn in root["dialogue"]["turns"]]
        result = build_dynamic_stateful_planner_input(
            parent_turns=turns, root_id=root["root_id"],
            sample_id=root["sample_id"], lineage_id="dynamic-lineage",
            run_id="dynamic-run", dialogue_hash=root["dialogue"]["dialogue_hash"],
            output_directory="/tmp/dynamic-run")
        self.assertEqual(build_stateful_outline_overview_prompt(parent_turns=turns),
                         result["overview_prompt"])
        self.assertEqual((10, 5), (result["parent_turn_count"],
                                  result["parent_session_count"]))

    def test_single_root_preflight_stops_before_api_and_excludes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root_path, output = base / "unified_root.json", base / "preflight"
            atomic_json(root_path, _root(3))
            result = dgp.preflight_unified_root(
                root_path, output_directory=output,
                token_counter=lambda value: len(value.split()))
            self.assertTrue(result["preflight_pass"])
            self.assertEqual(0, result["api_calls"])
            prompt_text = (output / "prompts/planner_overview.txt").read_text()
            self.assertNotIn("SENTINEL_TIME_", prompt_text)
            self.assertNotIn("SENTINEL_SUMMARY_", prompt_text)
            with self.assertRaises(FileExistsError):
                dgp.preflight_unified_root(
                    root_path, output_directory=output,
                    token_counter=lambda value: len(value.split()))

    def test_locomo_compat_root_uses_same_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "parent_dialogue.json"
            _write_locomo_source(source)
            root = dgp.build_locomo_compat_unified_root(source)
            self.assertEqual(3, root["dialogue"]["session_count"])
            root_path = base / "locomo_unified_root.json"
            atomic_json(root_path, root)
            result = dgp.preflight_unified_root(
                root_path, output_directory=base / "locomo_preflight",
                token_counter=lambda value: len(value.split()))
            self.assertTrue(result["preflight_pass"])
            self.assertEqual("locomo", result["dataset"])

    def test_ten_cc_plus_one_locomo_batch_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = []
            for index in range(10):
                path = base / f"root-{index}.json"
                atomic_json(path, _root(index))
                paths.append(path)
            locomo = base / "locomo_parent.json"
            _write_locomo_source(locomo)
            snapshot_manifest = base / "snapshot_manifest.json"
            atomic_json(snapshot_manifest, {"test": True})
            snapshot = {
                "source_snapshot_root": str(base),
                "source_snapshot_manifest_path": str(snapshot_manifest),
                "source_snapshot_manifest_sha256": "3" * 64,
                "source_snapshot_file_count": 1,
            }
            with patch.object(dgp, "create_source_snapshot", return_value=snapshot), \
                    patch.object(dgp, "project_diff_sha256", return_value="4" * 64), \
                    patch.object(dgp, "git_value", return_value="head"):
                report = dgp.run_generation_preflight_batch(
                    cc_root_paths=paths, locomo_parent_source=locomo,
                    output_root=base / "batch",
                    token_counter=lambda value: len(value.split()))
            self.assertTrue(report["generation_preflight_ready"])
            self.assertFalse(report["generation_ready"])
            self.assertEqual((10, 1, 11), (report["cc_root_count"],
                                           report["locomo_root_count"],
                                           report["root_count"]))
            self.assertEqual(0, report["api_calls"])


if __name__ == "__main__":
    unittest.main()
