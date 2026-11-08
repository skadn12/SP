"""Exercise diagnostic orchestration without provider calls or source use."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_probe as probe
from test_probe_contract import CLAIMS, HISTORY, ROOT, output


class ProbeRunnerTests(unittest.TestCase):
    def execute(self, failures=None, invalid=None):
        failures, invalid = set(failures or []), set(invalid or [])
        inputs = {"root_id": ROOT, "cumulative_history": deepcopy(HISTORY),
                  "claims": deepcopy(CLAIMS), "prior_summary": {"facts": ["Candidate summary."]}}
        inventory = {"accepted_uses": [], "reserved_uses": []}
        manifest = {"inventory_before": inventory, "schedule": [
            {"condition": condition, "replica": replica} for condition, replica in probe.SCHEDULE]}
        captured = []

        class FakeCaller:
            def __init__(self, run):
                self.run, self.http_counter, self.request_counter = run, 0, 0

            def call(self, label, payload, proof):
                self.request_counter += 1
                self.http_counter += 1
                captured.append((label, deepcopy(payload), deepcopy(proof)))
                directory = self.run / "requests" / f"{self.request_counter:03d}-{label}"
                # Simulate a caller mutating its argument; later calls must stay frozen.
                payload["messages"].append({"role": "assistant", "content": "Must not leak."})
                if self.request_counter in failures:
                    raise RuntimeError("fixture_transport_failure")
                raw = {} if self.request_counter in invalid else output("not_established")
                return json.dumps(raw), directory

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(probe, "verify", return_value=(manifest, inputs)), \
                    patch.object(probe.runtime.source_inventory, "collect_prior", return_value=inventory), \
                    patch.object(probe.runtime.source_inventory, "reserve", side_effect=AssertionError("source_use")), \
                    patch.object(probe.runtime, "Caller", FakeCaller):
                report = probe.live(run)
                saved = probe.runtime.base.read(run / "REPORT.json")
                with self.assertRaisesRegex(ValueError, "diagnostic_already_started"):
                    probe.live(run)
        self.assertEqual(report, {key: value for key, value in saved.items() if key != "rows"})
        return saved, captured

    def test_six_independent_calls_preserve_paired_context_and_never_reserve(self):
        report, captured = self.execute()
        self.assertEqual(len(captured), 6)
        self.assertEqual(report["logical_requests"], 6)
        self.assertEqual(report["http_attempts"], 6)
        self.assertTrue(report["inventory_unchanged"])
        self.assertFalse(report["source_reservations_created"])
        requests = {}
        for (condition, replica), (label, payload, proof) in zip(probe.SCHEDULE, captured):
            self.assertEqual(label, f"{condition}-{replica}")
            self.assertEqual(len(payload["messages"]), 1)
            self.assertFalse(proof["prior_answers_supplied"])
            self.assertFalse(proof["expected_answers_supplied"])
            self.assertEqual(payload, requests.setdefault(condition, payload))
        marker = probe.contract.INPUT_MARKER
        contexts = {key: json.loads(value["messages"][0]["content"].split(marker, 1)[1])
                    for key, value in requests.items()}
        self.assertIsNone(contexts["raw_history_only"].pop("prior_summary"))
        self.assertIsNotNone(contexts["with_prior_summary"].pop("prior_summary"))
        self.assertEqual(contexts["raw_history_only"], contexts["with_prior_summary"])

    def test_schema_rejections_do_not_trigger_repair_or_fill_missing_verdicts(self):
        report, captured = self.execute(invalid=[1, 4, 5])
        self.assertEqual(len(captured), 6)
        raw = report["summary"]["raw_history_only"]
        self.assertEqual(raw["request_statuses"], {"SCHEMA_REJECT": 3})
        self.assertTrue(all(value == {} for value in raw["claim_verdict_counts"].values()))
        self.assertEqual(report["summary"]["with_prior_summary"]["request_statuses"], {"SCHEMA_PASS": 3})

    def test_transport_failure_is_separate_and_remaining_calls_stay_independent(self):
        report, captured = self.execute(failures=[2], invalid=[3])
        self.assertEqual(len(captured), 6)
        statuses = report["summary"]["with_prior_summary"]["request_statuses"]
        self.assertEqual(statuses, {"TRANSPORT_FAILURE": 1, "SCHEMA_REJECT": 1, "SCHEMA_PASS": 1})
        for counts in report["summary"]["with_prior_summary"]["claim_verdict_counts"].values():
            self.assertEqual(counts, {"not_established": 1})


if __name__ == "__main__":
    unittest.main()
