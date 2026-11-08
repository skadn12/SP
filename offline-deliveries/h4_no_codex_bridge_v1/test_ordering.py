"""Focused replay and fail-closed tests for the ordering bridge."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ordering import build


DELIVERIES = Path(__file__).resolve().parent.parent
BATCH = DELIVERIES / "h4_crossroot_ordering_batch_v5_fresh_roots"
PILOT = DELIVERIES / "h4_crossroot_ordering_no_codex_pilot_v1"


def run(batch: Path = BATCH, pilot: Path = PILOT):
    return build(batch, pilot, pilot / "semantic_review_deepseek_v2",
                 pilot / "semantic_challenge_deepseek_v3", pilot / "semantic_review_glm_v4")


class OrderingBridgeTest(unittest.TestCase):
    def test_frozen_replay(self):
        pool, audit = run()
        self.assertEqual(pool["candidate_count"], 1)
        self.assertEqual(len(audit["candidates"]), 10)
        self.assertEqual(pool["questions"][0]["candidate_id"],
                         "cc-train-episode-194259-order-v5-003")
        self.assertFalse(pool["training_started"])

    def copy_pilot(self, target: Path) -> Path:
        pilot = target / PILOT.name
        shutil.copytree(PILOT, pilot, ignore=shutil.ignore_patterns("__pycache__", "semantic_review"))
        return pilot

    def test_changed_candidate_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pilot = self.copy_pilot(Path(temp))
            path = pilot / "LABEL_REBOUND_UNREVIEWED.json"
            rows = json.loads(path.read_text())
            rows[0]["items"]["I1"] += " altered"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "input_hashes"):
                run(pilot=pilot)

    def test_forged_pass_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pilot = self.copy_pilot(Path(temp))
            path = pilot / "semantic_review_glm_v4" / "SEMANTIC_REVIEW.json"
            review = json.loads(path.read_text())
            review["roots"][0]["checks"][0]["semantic_pass"] = True
            path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, "pass_changed"):
                run(pilot=pilot)

    def test_missing_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pilot = self.copy_pilot(Path(temp))
            response = (pilot / "semantic_review_glm_v4" /
                        "cc-train-episode-210046" / "RESPONSE.json")
            response.unlink()
            with self.assertRaises(FileNotFoundError):
                run(pilot=pilot)


if __name__ == "__main__":
    unittest.main()
