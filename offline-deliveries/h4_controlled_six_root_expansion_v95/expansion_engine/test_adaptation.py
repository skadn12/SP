"""Complete source and global consumption checks for the v95 derived engine."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import runner as r


class AdaptationTests(unittest.TestCase):
    def test_complete_sources_match_historical_ready_reviews(self):
        self.assertEqual(len(r.SOURCE_SPECS), 10)
        for root in r.SOURCE_SPECS:
            row = r.SOURCE_SPECS[root]
            turns = row["source_turns"]
            source = Path(row["source_path"])
            review = r.base.read(row["baseline_review_path"])
            self.assertEqual(r.base.digest(source.read_bytes()), row["source_sha256"])
            self.assertEqual(review["source"]["sha256"], row["source_sha256"])
            self.assertEqual(len(r.base.read(source)["turns"]), turns)
            self.assertFalse(any(t["turn_ref"].startswith("R02:B:") for t in r.base.read(source)["turns"]))

    def test_v15_and_sealed_v90_inventory_scanned_together(self):
        # Preserve actual historical collector verification; both generation
        # run directories must appear among candidates seen by the reader.
        self.assertEqual(r.source_inventory.V15_RUNS, r.DELIVERIES / "h4_feedback_training_pool_v15_expansion/runs")
        prior=r.source_inventory.collect_prior()
        self.assertEqual(prior["consumption_policy"], "reservation_before_model_request_never_release")
        self.assertTrue(any("h4_feedback_training_pool_v15_expansion/runs/" in path for path in prior["prior_runs"]))
        self.assertTrue(any("h4_end_to_end_expansion_mixed_v90/expansion_engine/runs/" in path for path in prior["prior_runs"]))
        self.assertEqual(len({r.source_inventory.event_key(row) for row in prior["accepted_uses"]}), len(prior["accepted_uses"]))
        v90 = r.base.read(r.source_inventory.V90_RUNS / "four_roots_B_scene01/SOURCE_RESERVATIONS.json")
        occupied = {r.source_inventory.event_key(row) for row in prior["accepted_uses"]}
        self.assertTrue(all(r.source_inventory.event_key(row) in occupied for row in v90["reservations"]))
        for row in v90["reservations"]:
            with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                r.source_inventory.validate_available({row["root_id"]: [row]}, prior)

    def test_cohorts_are_separate_and_have_true_root_feedback(self):
        batch = {root for root,row in r.SOURCE_SPECS.items() if row["cohort"] == "batch_other_roots"}
        regression = {root for root,row in r.SOURCE_SPECS.items() if row["cohort"] == "prior_planner_failure_regression"}
        self.assertEqual(len(batch), 8)
        self.assertEqual(regression, {"cc-train-episode-104814", "cc-train-episode-194259"})
        self.assertFalse(batch & {"cc-train-episode-"+str(x) for x in (103866,104814,135466,194259)})
        self.assertTrue(all(row["first_round_feedback"]["has_root_specific_feedback"] for row in r.SOURCE_SPECS.values()))

    def test_sealed_v91_reservations_remain_consumed(self):
        prior = r.source_inventory.collect_prior()
        old = r.base.read(r.source_inventory.V91_RUNS / "ten_roots_B_scene01/SOURCE_RESERVATIONS.json")
        self.assertEqual(len(old["reservations"]), 10)
        self.assertTrue(any("h4_planner_cardinality_expansion_batch_v91/expansion_engine/runs/" in p
                            for p in prior["prior_runs"]))
        by_key = {r.source_inventory.event_key(row): row for row in prior["accepted_uses"]}
        for row in old["reservations"]:
            self.assertEqual(by_key[r.source_inventory.event_key(row)]["run"], row["run"])
            with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                r.source_inventory.validate_available({row["root_id"]: [row]}, prior)

    def test_failed_v93_reservations_cannot_be_reallocated(self):
        prior = r.source_inventory.collect_prior()
        old = r.base.read(r.source_inventory.V93_RUNS / "four_other_roots_B_scene01/SOURCE_RESERVATIONS.json")
        self.assertEqual(len(old["reservations"]), 4)
        by_key = {r.source_inventory.event_key(row): row for row in prior["accepted_uses"]}
        for row in old["reservations"]:
            self.assertEqual(by_key[r.source_inventory.event_key(row)]["run"], row["run"])
            with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                r.source_inventory.validate_available({row["root_id"]: [row]}, prior)


if __name__ == "__main__":
    unittest.main()
