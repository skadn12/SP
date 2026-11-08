"""Tests for promoting a complete dialogue into the next frozen Parent."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .cumulative_parent import build_cumulative_parent


class CumulativeParentTests(unittest.TestCase):
    def test_regenerates_parent_ids_and_preserves_natural_boundaries(self):
        expanded = {
            "root_id": "cc-train-episode-source",
            "turn_count": 4,
            "complete_dialogue_sha256": "a" * 64,
            "turns": [
                {"turn_ref": "old:P01:T001", "session_ref": "old:parent_01",
                 "scene_ref": None, "speaker": "A", "text": "parent one"},
                {"turn_ref": "old:P01:T002", "session_ref": "old:parent_01",
                 "scene_ref": None, "speaker": "B", "text": "parent two"},
                {"turn_ref": "R01:S01:T001", "session_ref": "old:r01:s01",
                 "scene_ref": "R01:S01", "speaker": "A", "text": "child one"},
                {"turn_ref": "R01:S01:T002", "session_ref": "old:r01:s01",
                 "scene_ref": "R01:S01", "speaker": "B", "text": "child two"},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "expanded.json"
            source_path.write_text(json.dumps(expanded), encoding="utf-8")
            result = build_cumulative_parent(
                expanded, source_path=source_path,
                root_id="cc-train-episode-cumulative")

        turns = result["dialogue"]["turns"]
        self.assertEqual(result["dialogue"]["turn_count"], 4)
        self.assertEqual(result["dialogue"]["session_count"], 2)
        self.assertEqual([row["text"] for row in turns],
                         ["parent one", "parent two", "child one", "child two"])
        self.assertEqual([row["speaker"] for row in turns], ["A", "B", "A", "B"])
        self.assertEqual([row["turn_id"] for row in turns], [
            "cc-train-episode-cumulative:P01:T001",
            "cc-train-episode-cumulative:P01:T002",
            "cc-train-episode-cumulative:P02:T001",
            "cc-train-episode-cumulative:P02:T002",
        ])
        self.assertEqual({row["partition"] for row in turns}, {"parent"})


if __name__ == "__main__":
    unittest.main()
