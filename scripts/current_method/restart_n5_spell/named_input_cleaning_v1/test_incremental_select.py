from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .incremental_select import select_next, write_mapping_template, write_selection


def _record(index: int) -> dict:
    record = {
        "dataID": f"episode-{index}",
        "relationship": "Classmates",
        "time_interval": ["t1", "t2", "t3", "t4", "t5"],
        "summary": ["A short exchange."] * 5,
    }
    for name in ("first", "second", "third", "fourth", "fifth"):
        record[f"{name}_session_speakers"] = ["Classmates A", "Classmates B"]
        record[f"{name}_session_dialogue"] = ["A said hello.", "B replied."]
    return record


class IncrementalSelectionTests(unittest.TestCase):
    def test_selects_unseen_rows_in_source_order_and_writes_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "train.jsonl"
            source.write_text(
                "".join(json.dumps(_record(i)) + "\n" for i in range(1, 7)),
                encoding="utf-8",
            )
            excluded = base / "old.json"
            excluded.write_text(json.dumps({
                "records": [{"root_id": "cc-train-episode-1"},
                            {"root_id": "cc-train-episode-2"}],
            }), encoding="utf-8")
            payload = select_next(
                input_path=source, count=3,
                exclude_manifests=[excluded],
            )
            self.assertEqual(
                [row["source_id"] for row in payload["records"]],
                ["episode-3", "episode-4", "episode-5"],
            )
            self.assertEqual(payload["scan_start_row_index"], 0)
            self.assertEqual(payload["next_cursor_row_index"], 5)
            self.assertEqual(payload["skip_counts"]["excluded_root"], 2)
            template = write_mapping_template(base / "mapping.json", payload)
            data = json.loads(template.read_text(encoding="utf-8"))
            self.assertEqual(sorted(data["mappings"]), [
                "episode-3", "episode-4", "episode-5",
            ])
            self.assertEqual(
                data["mappings"]["episode-3"]["speaker_map"],
                {"Classmates A": "", "Classmates B": ""},
            )
            selection = write_selection(base / "selection.json", payload)
            self.assertTrue(selection.is_file())

    def test_cursor_requires_same_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "train.jsonl"
            source.write_text(json.dumps(_record(1)) + "\n", encoding="utf-8")
            first = select_next(input_path=source, count=1)
            cursor = base / "cursor.json"
            write_selection(cursor, first)
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                select_next(input_path=source, count=1, cursor_manifest=cursor)


if __name__ == "__main__":
    unittest.main()
