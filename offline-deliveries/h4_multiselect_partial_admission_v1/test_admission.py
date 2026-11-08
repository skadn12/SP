import unittest

from admission import candidate_decision, numeric_reason


class AdmissionTests(unittest.TestCase):
    def test_full_zero_partial_gain_is_numeric_candidate(self):
        metrics = dict(full_exact_rate=0, no_exact_rate=0,
                       full_partial_mean=1 / 3, no_partial_mean=0)
        result = candidate_decision(metrics, quality_check_status="PASS",
                                    permutation_check_status="PASS",
                                    pairing_check_status="PASS", split_check_status="PASS")
        self.assertEqual(result["new_policy_reason"], "FULL_ZERO_PARTIAL_GAIN")
        self.assertEqual(result["new_policy_candidate_status"], "TRAINING_CANDIDATE")

    def test_checks_are_fail_closed(self):
        metrics = dict(full_exact_rate=0, no_exact_rate=0,
                       full_partial_mean=1 / 3, no_partial_mean=0)
        result = candidate_decision(metrics, quality_check_status="UNVERIFIED",
                                    permutation_check_status="PENDING",
                                    pairing_check_status="PASS", split_check_status="PASS")
        self.assertEqual(result["new_policy_candidate_status"], "DIAGNOSTIC_PENDING")
        self.assertTrue(result["blockers"])

    def test_incomplete_pairing_keeps_pattern_provisional(self):
        metrics = dict(full_exact_rate=0, no_exact_rate=0,
                       full_partial_mean=1 / 3, no_partial_mean=0)
        result = candidate_decision(metrics, quality_check_status="PASS",
                                    permutation_check_status="PASS",
                                    pairing_check_status="INCOMPLETE", split_check_status="PASS")
        self.assertTrue(result["observed_numeric_pattern"])
        self.assertFalse(result["numeric_candidate"])
        self.assertEqual(result["new_policy_candidate_status"], "DIAGNOSTIC_PENDING")

    def test_direction_conflict_and_resolved(self):
        self.assertEqual(numeric_reason(dict(full_exact_rate=0.2, no_exact_rate=0,
                                             full_partial_mean=0.1, no_partial_mean=0.2)),
                         "METRIC_DIRECTION_CONFLICT")
        self.assertEqual(numeric_reason(dict(full_exact_rate=1, no_exact_rate=0,
                                             full_partial_mean=1, no_partial_mean=0)),
                         "OBSERVED_FULL_RESOLVED")

    def test_missing_pairing(self):
        self.assertEqual(numeric_reason(dict(full_exact_rate=None, no_exact_rate=0,
                                             full_partial_mean=None, no_partial_mean=0)),
                         "PAIRING_INCOMPLETE")

    def test_confirmed_quarantine_blocks(self):
        metrics = dict(full_exact_rate=0, no_exact_rate=0,
                       full_partial_mean=1 / 3, no_partial_mean=0)
        result = candidate_decision(metrics, quality_check_status="PASS",
                                    permutation_check_status="PASS",
                                    pairing_check_status="PASS", split_check_status="PASS",
                                    confirmed_quarantine=True)
        self.assertEqual(result["new_policy_candidate_status"], "DIAGNOSTIC_PENDING")
        self.assertIn("CONFIRMED_QUARANTINE", result["blockers"])


if __name__ == "__main__":
    unittest.main()
