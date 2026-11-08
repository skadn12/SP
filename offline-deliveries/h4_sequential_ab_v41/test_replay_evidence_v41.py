"""Real frozen-response replay, always offline and never a new source owner."""
import json
from pathlib import Path
import tempfile
import unittest

import replay_evidence


class ActualHistoricalReplayTests(unittest.TestCase):
    def test_actual_missing_field_retains_dialogue_without_rewriting_reject(self):
        artifacts = replay_evidence.build_replay()
        report = artifacts["REPORT.json"]
        incomplete = next(row for row in report["scenarios"]
                          if row["name"] == "v40_missing_history_goal_checks")
        self.assertEqual(incomplete["historical_validator_status"], "REJECT")
        self.assertFalse(incomplete["historical_schema_valid"])
        self.assertEqual(incomplete["disposition"], "retained_pending_review")
        self.assertIsNone(incomplete["raw_model_aggregate_verdict"])
        self.assertEqual(incomplete["audit_issues"][0]["code"], "legacy_review_schema_incomplete")
        self.assertEqual(report["unique_diagnostic_body"], {"turns": 16, "body_tokens": 504})
        self.assertEqual(report["formal_accepted_cumulative"], {"turns": 132, "body_tokens": 3672})
        self.assertEqual(report["fresh_A_body_tokens"], 0)
        self.assertEqual(report["fresh_B_body_tokens"], 0)
        self.assertEqual(report["training_admitted_body_tokens"], 0)

    def test_generated_replay_recomputes_and_catches_tampered_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "offline_replay"
            audit = replay_evidence.replay(output)
            self.assertEqual(audit["status"], "ARTIFACTS_CONSISTENT")
            self.assertFalse((output / "SOURCE_RESERVATIONS.json").exists())
            self.assertFalse((output / "requests").exists())
            path = output / "scenarios/v40_missing_history_goal_checks/DECISION.json"
            modified = json.loads(path.read_text())
            modified["training_pool_eligible"] = True
            path.write_text(json.dumps(modified))
            with self.assertRaisesRegex(ValueError, "candidate_artifact_changed"):
                replay_evidence.verify(output)

    def test_replay_does_not_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "replay_output_already_exists"):
                replay_evidence.replay(Path(tmp))


if __name__ == "__main__":
    unittest.main()
