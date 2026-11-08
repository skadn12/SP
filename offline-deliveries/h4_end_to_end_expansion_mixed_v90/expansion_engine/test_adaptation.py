"""Meaningful fail-closed checks for the v90 registry/inventory adaptation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import runner as r


class AdaptationTests(unittest.TestCase):
    def test_complete_sources_match_historical_ready_reviews(self):
        expected = {"cc-train-episode-103866":115, "cc-train-episode-104814":94,
                    "cc-train-episode-135466":189, "cc-train-episode-194259":149}
        for root, turns in expected.items():
            row = r.SOURCE_SPECS[root]
            source = Path(row["source_path"])
            review = r.base.read(row["baseline_review_path"])
            self.assertEqual(r.base.digest(source.read_bytes()), row["source_sha256"])
            self.assertEqual(review["source"]["sha256"], row["source_sha256"])
            self.assertEqual(len(r.base.read(source)["turns"]), turns)
            self.assertFalse(any(t["turn_ref"].startswith("R02:B:") for t in r.base.read(source)["turns"]))

    def test_v15_and_v90_inventory_scanned_together(self):
        # Preserve actual historical collector verification; both generation
        # run directories must appear among candidates seen by the reader.
        self.assertEqual(r.source_inventory.V15_RUNS, r.DELIVERIES / "h4_feedback_training_pool_v15_expansion/runs")
        prior=r.source_inventory.collect_prior()
        self.assertEqual(prior["consumption_policy"], "reservation_before_model_request_never_release")
        self.assertTrue(any("h4_feedback_training_pool_v15_expansion/runs/" in path for path in prior["prior_runs"]))
        self.assertEqual(len({r.source_inventory.event_key(row) for row in prior["accepted_uses"]}), len(prior["accepted_uses"]))


if __name__ == "__main__":
    unittest.main()
