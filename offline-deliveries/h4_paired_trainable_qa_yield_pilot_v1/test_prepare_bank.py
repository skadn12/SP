"""Scope checks for the frozen historical question bank."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import prepare


class QuestionBankTests(unittest.TestCase):
    def test_includes_target_questions_and_excludes_unrelated_artifacts(self) -> None:
        with TemporaryDirectory() as temporary:
            repo = Path(temporary)
            root = "cc-train-episode-136202"
            backup = "cc-train-episode-197635"

            def write(relative: str, value: object) -> None:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

            write(f"restart-n5-runs/run/roots/{root}/candidate_pool.json", [{"question": "Old question"}])
            write("restart-n5-runs/run/hard_candidates.json", {"candidates": [
                {"root_id": backup, "question": "Backup question"},
                {"root_id": "cc-train-episode-259192", "question": "Other question"},
            ]})
            write("offline-deliveries/run/questions.json", [
                {"candidate_id": root + ":Q1", "question": "Target question"},
            ])
            write("offline-deliveries/unrelated/candidates.json", [
                {"root_id": "cc-train-episode-259192", "question": "Other question"},
            ])
            write("offline-deliveries/metadata_only/candidates.json", {
                "source": {"root_id": root},
                "candidates": [{"root_id": "cc-train-episode-259192", "question": "Other question"}],
            })
            write(f"restart-n5-runs/run/roots/{root}/memory/event_candidates.json", [
                {"root_id": root, "question": "Not a question-bank artifact"},
            ])
            write(f"restart-n5-runs/run/roots/{root}-other/candidates_valid.json", [
                {"question": "Prefix collision"},
            ])

            with patch.object(prepare, "REPO", repo):
                bank = prepare.question_bank()

            self.assertEqual([Path(row["path"]).name for row in bank],
                             ["questions.json", "hard_candidates.json", "candidate_pool.json"])
            self.assertEqual([row["root_ids"] for row in bank], [[root], [backup], [root]])
            self.assertTrue(all(len(row["sha256"]) == 64 for row in bank))


if __name__ == "__main__":
    unittest.main()
