"""Gate compatibility resume uses private journals and never a transport."""
from copy import deepcopy
import hashlib
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .budget import BudgetLedger
from .capacity import proposed_budget
from .test_experiment_ledger import mock_scope


def competing_gate_claim(registry, scope, identity, source, barrier, results):
    try:
        ledger = BudgetLedger(registry, root_id="MOCK-root", round_index=3,
                              budget=proposed_budget(), execution_scope=scope)
        barrier.wait(timeout=10)
        slots = ledger.claim_gate_result_resume(identity, **source)
        results.put(("MOCK_CLAIMED", slots))
    except Exception as exc:
        results.put((type(exc).__name__, str(exc)))


class GateReuseLedgerTests(unittest.TestCase):
    def setUp(self):
        for target in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(target, side_effect=AssertionError("network_forbidden_in_ledger_test"))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory(prefix="h4-gate-reuse-ledger-MOCK-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.root / "MOCK_registry"
        self.scope = mock_scope(self.root)
        self.budget = proposed_budget()
        self.original = {"branch_id": "MOCK-original", "experiment_id": self.scope["experiment_id"]}
        self.identity = {"identity": self.original, "compatibility_manifest_sha256": "c" * 64}
        self.slots = ["batch_2", "batch_4", "batch_6", "batch_8"]

    def ledger(self):
        return BudgetLedger(self.registry, root_id="MOCK-root", round_index=3,
                            budget=self.budget, execution_scope=self.scope)

    @staticmethod
    def source(ledger):
        return {"source_ledger_header": ledger.snapshot()[0],
                "source_ledger_sha256": hashlib.sha256(ledger.path.read_bytes()).hexdigest()}

    def completed_gates(self, *, last_status="RESPONSE_RECEIVED"):
        ledger = self.ledger()
        ledger.claim_execution(self.original)
        self.assertTrue(ledger.reserve("closure", {"closure": 1}, cumulative_tokens=1))
        self.assertTrue(ledger.reserve("supplement", {"supplement": 1}, cumulative_tokens=1))
        call = ledger.begin("supplement", "supplement", input_tokens=1)
        ledger.finish(call, status="RESPONSE_RECEIVED")
        ledger.claim_resume({"identity": self.original}, **self.source(ledger))
        for slot in self.slots:
            self.assertTrue(ledger.reserve(slot, {"gate": 2, "realizer": 1, "adherence": 1},
                                           cumulative_tokens=1, added_scene_count=2))
            for _ in range(2):
                call = ledger.begin(slot, "gate", input_tokens=1)
                if call != 9 or last_status is not None:
                    ledger.finish(call, status=last_status if call == 9 else "RESPONSE_RECEIVED")
        return ledger

    def claim(self, ledger):
        return ledger.claim_gate_result_resume(self.identity, **self.source(ledger))

    def test_exact_consumed_state_reuses_stable_slots_without_reset(self):
        ledger = self.completed_gates()
        before = ledger.path.read_bytes()
        reservations = BudgetLedger._active(ledger.snapshot())
        self.assertEqual(self.claim(ledger), self.slots)
        self.assertTrue(ledger.path.read_bytes().startswith(before))
        self.assertEqual(BudgetLedger._active(ledger.snapshot()), reservations)
        rows = ledger.snapshot()
        self.assertEqual(sum(row["kind"] == "attempt" for row in rows), 9)
        self.assertEqual(rows[-1]["kind"], "gate_result_resume")
        self.assertEqual(rows[-1]["generation_reservation_ids"], self.slots)
        with self.assertRaisesRegex(ValueError, "reserved_role_exhausted"):
            ledger.begin("supplement", "supplement", input_tokens=1)
        with self.assertRaisesRegex(ValueError, "reserved_role_exhausted"):
            ledger.begin(self.slots[0], "gate", input_tokens=1)

    def test_same_claim_restart_or_changed_directory_cannot_claim_again(self):
        ledger = self.completed_gates()
        source = self.source(ledger)
        self.claim(ledger)
        before = ledger.path.read_bytes()
        restarted = self.ledger()
        with self.assertRaisesRegex(ValueError, "changed_or_already_claimed"):
            restarted.claim_gate_result_resume(self.identity, **source)
        changed = {**self.identity, "new_output_directory": str(self.root / "MOCK-other-output")}
        with self.assertRaisesRegex(ValueError, "already_claimed_no_outer_attempt"):
            restarted.claim_gate_result_resume(changed, **self.source(restarted))
        self.assertEqual(restarted.path.read_bytes(), before)

    def test_wrong_header_sha_or_original_identity_do_not_append(self):
        ledger = self.completed_gates()
        before = ledger.path.read_bytes()
        source = self.source(ledger)
        wrong_header = deepcopy(source)
        wrong_header["source_ledger_header"]["round_index"] = 4
        for kwargs, identity, expected in (
                (wrong_header, self.identity, "ledger_identity_mismatch"),
                ({**source, "source_ledger_sha256": "0" * 64}, self.identity, "ledger_changed"),
                (source, {"identity": {"branch_id": "MOCK-another"}}, "original_execution_and_resume")):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                ledger.claim_gate_result_resume(identity, **kwargs)
            self.assertEqual(ledger.path.read_bytes(), before)

    def test_unknown_or_failed_source_gate_blocks_claim(self):
        for status in (None, "TIMEOUT", "HTTP_REJECTED", "RESPONSE_UNKNOWN"):
            with self.subTest(status=status):
                self.scope = {**self.scope, "experiment_id": "MOCK_" + str(status)}
                ledger = self.completed_gates(last_status=status)
                before = ledger.path.read_bytes()
                with self.assertRaisesRegex(ValueError, "responses_must_be_confirmed"):
                    self.claim(ledger)
                self.assertEqual(ledger.path.read_bytes(), before)

    def test_existing_downstream_attempt_cannot_be_claimed_as_unused(self):
        for status in (None, "TIMEOUT", "HTTP_REJECTED", "RESPONSE_RECEIVED"):
            with self.subTest(status=status):
                self.scope = {**self.scope, "experiment_id": "MOCK_prior_" + str(status)}
                ledger = self.completed_gates()
                call = ledger.begin(self.slots[0], "realizer", input_tokens=1)
                if status is not None:
                    ledger.finish(call, status=status)
                before = ledger.path.read_bytes()
                with self.assertRaisesRegex(ValueError, "no_downstream_attempts"):
                    self.claim(ledger)
                self.assertEqual(ledger.path.read_bytes(), before)

    def test_slot_checks_require_claim_and_leave_bytes_unchanged(self):
        ledger = self.completed_gates()
        before = ledger.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "slot_not_claimed"):
            ledger.check_reused_generation_slot(self.slots[0], cumulative_tokens=143, added_scene_count=2)
        self.assertEqual(ledger.path.read_bytes(), before)
        self.claim(ledger)
        before = ledger.path.read_bytes()
        for slot in self.slots:
            self.assertTrue(ledger.check_reused_generation_slot(slot, cumulative_tokens=143, added_scene_count=2))
        self.assertFalse(ledger.check_reused_generation_slot(self.slots[0],
                         cumulative_tokens=self.budget["max_cumulative_tokens"], added_scene_count=1))
        with self.assertRaisesRegex(ValueError, "slot_not_claimed"):
            ledger.check_reused_generation_slot("new_slot", cumulative_tokens=143, added_scene_count=1)
        self.assertEqual(ledger.path.read_bytes(), before)

    def test_invalid_capacity_does_not_consume_slot(self):
        ledger = self.completed_gates()
        self.claim(ledger)
        before = ledger.path.read_bytes()
        for cumulative, scenes in ((-1, 1), (True, 1), (1, 3), (1, -1), (1, True)):
            with self.subTest(cumulative=cumulative, scenes=scenes), self.assertRaisesRegex(ValueError, "invalid_reused_generation_capacity"):
                ledger.check_reused_generation_slot(self.slots[0], cumulative_tokens=cumulative, added_scene_count=scenes)
        self.assertEqual(ledger.path.read_bytes(), before)
        self.assertEqual(ledger.begin(self.slots[0], "realizer", input_tokens=1), 10)

    def test_role_total_and_output_limits_are_still_checked(self):
        ledger = self.completed_gates()
        self.claim(ledger)
        before = ledger.path.read_bytes()
        for name in ("role", "total", "output"):
            ledger.budget = deepcopy(self.budget)
            if name == "role":
                ledger.budget["role_requests"]["adherence"] -= 1
            elif name == "total":
                ledger.budget["total_requests"] -= 1
            else:
                ledger.budget["max_generated_tokens"] -= 1
            with self.subTest(limit=name):
                self.assertFalse(ledger.check_reused_generation_slot(self.slots[0], cumulative_tokens=1, added_scene_count=2))
                self.assertEqual(ledger.path.read_bytes(), before)

    def test_unknown_and_failed_downstream_slots_remain_consumed(self):
        for status in (None, "TIMEOUT", "HTTP_REJECTED", "RESPONSE_UNKNOWN", "RESPONSE_RECEIVED"):
            with self.subTest(status=status):
                self.scope = {**self.scope, "experiment_id": "MOCK_downstream_" + str(status)}
                ledger = self.completed_gates()
                self.claim(ledger)
                call = ledger.begin(self.slots[0], "realizer", input_tokens=1)
                if status is not None:
                    ledger.finish(call, status=status)
                before = ledger.path.read_bytes()
                restarted = self.ledger()
                with self.assertRaisesRegex(ValueError, "already_attempted_no_replay"):
                    restarted.check_reused_generation_slot(self.slots[0], cumulative_tokens=1, added_scene_count=2)
                with self.assertRaisesRegex(ValueError, "reserved_role_exhausted"):
                    restarted.begin(self.slots[0], "realizer", input_tokens=1)
                self.assertEqual(ledger.path.read_bytes(), before)

    def test_all_eight_downstream_slots_share_existing_budget(self):
        ledger = self.completed_gates()
        self.claim(ledger)
        for slot in self.slots:
            self.assertTrue(ledger.check_reused_generation_slot(slot, cumulative_tokens=143, added_scene_count=2))
            for role in ("realizer", "adherence"):
                call = ledger.begin(slot, role, input_tokens=1)
                ledger.finish(call, status="RESPONSE_RECEIVED")
        attempts = [row for row in ledger.snapshot() if row["kind"] == "attempt"]
        self.assertEqual(len(attempts), 17)
        self.assertEqual({role: sum(row["role"] == role for row in attempts)
                          for role in ("supplement", "gate", "realizer", "adherence")},
                         {"supplement": 1, "gate": 8, "realizer": 4, "adherence": 4})
        self.assertFalse(ledger.reserve("new_gate", {"gate": 1}, cumulative_tokens=1))

    def test_two_processes_can_only_claim_once(self):
        ledger = self.completed_gates()
        source = self.source(ledger)
        before = ledger.path.read_bytes()
        ctx = multiprocessing.get_context("fork")
        barrier, results = ctx.Barrier(2), ctx.Queue()
        workers = [ctx.Process(target=competing_gate_claim,
                   args=(self.registry, self.scope, self.identity, source, barrier, results)) for _ in range(2)]
        for worker in workers:
            worker.start()
        outcomes = [results.get(timeout=20) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
                self.fail("concurrent mock claim did not finish")
            self.assertEqual(worker.exitcode, 0)
        results.close()
        self.assertEqual(sum(row[0] == "MOCK_CLAIMED" for row in outcomes), 1, outcomes)
        self.assertEqual(sum("already_claimed" in str(row[1]) for row in outcomes), 1, outcomes)
        self.assertTrue(ledger.path.read_bytes().startswith(before))
        self.assertEqual(sum(row["kind"] == "gate_result_resume" for row in ledger.snapshot()), 1)
        self.assertEqual(sum(row["kind"] == "attempt" for row in ledger.snapshot()), 9)

    def test_no_original_resume_or_changed_reservation_shape_is_rejected(self):
        ledger = self.completed_gates()
        for kind, mutation, expected in (
                ("resume_execution", lambda row: {**row, "kind": "MOCK_removed_resume"}, "original_execution_and_resume"),
                ("reserve", lambda row: {**row, "requests": {**row["requests"], "realizer": 2}}, "reserved_generation_slots")):
            # Private journal corruption tests do not mutate any actual run.
            rows = ledger.snapshot()
            match = next(index for index, row in enumerate(rows)
                         if row["kind"] == kind and (kind != "reserve" or row["reservation_id"] == self.slots[0]))
            rows[match] = mutation(rows[match])
            source = self.source(ledger)
            with patch.object(ledger, "_locked") as locked:
                locked.return_value.__enter__.return_value = rows
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, expected):
                    ledger.claim_gate_result_resume(self.identity, **source)


if __name__ == "__main__":
    unittest.main()
