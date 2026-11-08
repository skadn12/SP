from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import retry_429 as recovery


class RecoveryTests(unittest.TestCase):
    def test_actual_parent_scope_is_exactly_five_failed_payloads_not_successes(self):
        parent = recovery.HERE / "runs/end_to_end_5roots_01"
        scope = recovery.scope_from_parent(parent)
        self.assertEqual(len(scope["requests"]), 5)
        self.assertEqual(sum(len(row["candidate_ids"]) for row in scope["requests"]), 11)
        self.assertEqual(sum(recovery.scope_counts(scope, "A").values()), 2)
        self.assertEqual(sum(recovery.scope_counts(scope, "B").values()), 9)
        for row in scope["requests"]:
            original = recovery.read(Path(row["parent_request_dir"]) / "REQUEST.json")
            self.assertEqual(row["payload"], original)
            proof = recovery.retry_proof(row)
            self.assertEqual(proof["retry_number"], 1)
            self.assertEqual(proof["feedback_receipt"], row["parent_proof"]["feedback_receipt"])
            self.assertEqual(recovery.qa.base.digest(original), row["payload_sha256"])

    def test_global_pacing_preserves_payload_and_records_at_least_35_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            clock, sleeps = [0.0], []
            def sleep(seconds):
                self.assertLessEqual(seconds, 35)
                sleeps.append(seconds)
                clock[0] += seconds
            class Caller:
                def __init__(self):
                    self.calls = []
                def call(self, label, payload, proof):
                    self.calls.append(deepcopy(payload))
                    path = run / label
                    path.mkdir()
                    (path / "HTTP_ATTEMPT.json").write_text('{}')
                    clock[0] += 2
                    return '{}', path, {}
            caller = Caller()
            paced = recovery.PacedCalls(run, monotonic=lambda: clock[0], sleep=sleep)
            payload = {"messages": [{"role": "user", "content": "Exact\nrequest"}]}
            paced.call(caller, "one", payload, {})
            paced.call(caller, "two", payload, {})
            self.assertEqual(caller.calls, [payload, payload])
            self.assertEqual(sleeps, [35])
            self.assertEqual(paced.rows[1]["interval_since_previous_call_completion_seconds"], 35)

    def test_parent_success_same_statement_set_never_enters_recovery_solver(self):
        original = {"candidate_id": "old", "root_id": "r", "question_type": "multi_select",
                    "options": {"A":"one", "B":"two"}}
        copied = {**original, "candidate_id":"new", "options":{"A":"two", "B":"one"}}
        different = {**original, "candidate_id":"fresh", "options":{"A":"three", "B":"four"}}
        accepted, rejected = recovery.reject_parent_success([copied,different],[original])
        self.assertEqual([q['candidate_id'] for q in accepted], ['fresh'])
        self.assertEqual(rejected[0]['errors'], ['duplicate_parent_success_statement_set'])


if __name__ == "__main__":
    unittest.main()
