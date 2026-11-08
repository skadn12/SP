"""Experiment scopes on the existing ledger; only temporary local journals."""
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from .budget import BudgetLedger
from .capacity import proposed_budget


def mock_scope(root):
    return {"mode": "extra_experiment", "experiment_id": "MOCK_EXTRA_VAL_01",
            "grant_sha256": "a" * 64, "source_run": str(root / "MOCK_old_run"),
            "production_ledger_path": str(root / "MOCK_production.jsonl"),
            "production_ledger_sha256": "b" * 64}


def competing_claim(registry, scope, barrier, results):
    """Two processes use the same stable inode, then attempt the same claim."""
    try:
        ledger = BudgetLedger(registry, root_id="MOCK-root", round_index=3,
                              budget=proposed_budget(), execution_scope=scope)
        barrier.wait(timeout=10)
        ledger.claim_execution({"branch_id": "MOCK-competing-process"})
        if not ledger.reserve("supplement", {"supplement": 1}, cumulative_tokens=1):
            raise AssertionError("mock reservation unexpectedly rejected")
        ledger.begin("supplement", "supplement", input_tokens=1)
        results.put("MOCK_SEND_QUALIFIED")
    except Exception as exc:
        results.put(type(exc).__name__ + ":" + str(exc))


class ExperimentLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h4-experiment-ledger-MOCK-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.root / "MOCK_registry"
        self.scope = mock_scope(self.root)
        self.budget = proposed_budget()

    def ledger(self, *, scope=None, budget=None, root_id="MOCK-root", round_index=3):
        return BudgetLedger(self.registry, root_id=root_id, round_index=round_index,
            budget=budget or self.budget, execution_scope=self.scope if scope is None else scope)

    def test_production_key_header_and_consumed_opportunity_are_unchanged(self):
        ledger = BudgetLedger(self.registry, root_id="MOCK-root", round_index=3, budget=self.budget)
        old_identity = {"root_id": "MOCK-root", "round_index": 3}
        old_key = hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()
        self.assertEqual(ledger.path.name, old_key + ".jsonl")
        self.assertEqual(ledger.snapshot(), [{"kind": "identity", **old_identity, "budget": self.budget}])
        ledger.claim_execution({"branch_id": "MOCK-old"})
        ledger.reserve("supplement", {"supplement": 1}, cumulative_tokens=1)
        ledger.begin("supplement", "supplement", input_tokens=1)
        before = ledger.path.read_bytes()
        restarted = BudgetLedger(self.registry, root_id="MOCK-root", round_index=3, budget=self.budget)
        with self.assertRaisesRegex(ValueError, "already_claimed_no_outer_attempt"):
            restarted.claim_execution({"branch_id": "MOCK-new-directory"})
        self.assertEqual(ledger.path.read_bytes(), before)
        self.assertEqual(len(list(self.registry.glob("*.jsonl"))), 1)

    def test_extra_scope_does_not_modify_production_journal(self):
        production = BudgetLedger(self.registry, root_id="MOCK-root", round_index=3, budget=self.budget)
        production.claim_execution({"branch_id": "MOCK-old"})
        before = hashlib.sha256(production.path.read_bytes()).hexdigest()
        experiment = self.ledger()
        experiment.claim_execution({"branch_id": "MOCK-extra"})
        experiment.reserve("supplement", {"supplement": 1}, cumulative_tokens=1)
        experiment.begin("supplement", "supplement", input_tokens=1)
        self.assertNotEqual(production.path, experiment.path)
        self.assertEqual(hashlib.sha256(production.path.read_bytes()).hexdigest(), before)
        self.assertEqual(experiment.snapshot()[0]["execution_scope"], self.scope)

    def test_fixed_id_key_ignores_run_directory(self):
        first = self.ledger()
        key = hashlib.sha256(json.dumps({"experiment_id": self.scope["experiment_id"]}, sort_keys=True).encode()).hexdigest()
        self.assertEqual(first.path.name, key + ".jsonl")
        first.claim_execution({"branch_id": "MOCK-output-A"})
        with self.assertRaisesRegex(ValueError, "already_claimed_no_outer_attempt"):
            self.ledger().claim_execution({"branch_id": "MOCK-output-B"})
        self.assertEqual(len(list(self.registry.glob("*.jsonl"))), 1)

    def test_same_authorization_changed_binding_rejected_without_new_ledger(self):
        ledger = self.ledger()
        before = ledger.path.read_bytes()
        for field in ("grant_sha256", "production_ledger_sha256", "source_run", "production_ledger_path"):
            changed = deepcopy(self.scope)
            changed[field] = "c" * 64 if field.endswith("sha256") else changed[field] + "-changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "persistent_budget_identity_mismatch"):
                self.ledger(scope=changed)
        for changes in ({"root_id": "MOCK-other-root"}, {"round_index": 4}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "persistent_budget_identity_mismatch"):
                self.ledger(**changes)
        changed_budget = deepcopy(self.budget)
        changed_budget["role_requests"]["gate"] -= 1
        with self.assertRaisesRegex(ValueError, "persistent_budget_identity_mismatch"):
            self.ledger(budget=changed_budget)
        self.assertEqual(ledger.path.read_bytes(), before)
        self.assertEqual(len(list(self.registry.glob("*.jsonl"))), 1)

    def test_scope_is_a_detached_immutable_header_value(self):
        scope = deepcopy(self.scope)
        ledger = self.ledger(scope=scope)
        scope["source_run"] += "-mutated"
        self.assertEqual(ledger.identity["execution_scope"], self.scope)
        self.assertEqual(ledger.snapshot()[0]["execution_scope"], self.scope)

    def test_incomplete_or_malformed_scope_fails_before_registry(self):
        samples = [False, {}, {**self.scope, "mode": "production"},
                   {**self.scope, "experiment_id": "../new-budget"},
                   {**self.scope, "grant_sha256": "not-sha"},
                   {**self.scope, "production_ledger_sha256": ""},
                   {**self.scope, "extra_field": True}, {**self.scope, "source_run": None}]
        for scope in samples:
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "complete_explicit_experiment_scope"):
                self.ledger(scope=scope)
        self.assertFalse(self.registry.exists())

    def test_timeout_http_rejection_and_unknown_each_remain_consumed(self):
        for status in ("HTTP_REJECTED", "TIMEOUT", "RESPONSE_UNKNOWN", "INVALID_PATCH"):
            with self.subTest(status=status):
                scope = {**self.scope, "experiment_id": "MOCK_" + status}
                ledger = self.ledger(scope=scope)
                ledger.claim_execution({"branch_id": "MOCK-first"})
                self.assertTrue(ledger.reserve("supplement", {"supplement": 1}, cumulative_tokens=1))
                call = ledger.begin("supplement", "supplement", input_tokens=1)
                ledger.finish(call, status=status)
                restarted = self.ledger(scope=scope)
                with self.assertRaisesRegex(ValueError, "reserved_role_exhausted"):
                    restarted.begin("supplement", "supplement", input_tokens=1)
                with self.assertRaisesRegex(ValueError, "already_claimed_no_outer_attempt"):
                    restarted.claim_execution({"branch_id": "MOCK-second"})
                self.assertEqual(sum(row["kind"] == "attempt" for row in restarted.snapshot()), 1)

    def test_crash_between_claim_and_send_is_not_replayed(self):
        ledger = self.ledger()
        ledger.claim_execution({"branch_id": "MOCK-before-send-crash"})
        self.assertFalse(any(row["kind"] == "attempt" for row in ledger.snapshot()))
        with self.assertRaisesRegex(ValueError, "already_claimed_no_outer_attempt"):
            self.ledger().claim_execution({"branch_id": "MOCK-restart"})

    def test_crash_after_begin_without_result_is_not_replayed(self):
        ledger = self.ledger()
        ledger.reserve("supplement", {"supplement": 1}, cumulative_tokens=1)
        ledger.begin("supplement", "supplement", input_tokens=1)
        with self.assertRaisesRegex(ValueError, "reserved_role_exhausted"):
            self.ledger().begin("supplement", "supplement", input_tokens=1)
        self.assertFalse(any(row["kind"] == "result" for row in ledger.snapshot()))

    def test_two_processes_only_one_can_claim_and_begin(self):
        ctx = multiprocessing.get_context("fork")
        barrier, results = ctx.Barrier(2), ctx.Queue()
        processes = [ctx.Process(target=competing_claim,
                     args=(self.registry, self.scope, barrier, results)) for _ in range(2)]
        for proc in processes:
            proc.start()
        outcomes = [results.get(timeout=20) for _ in range(2)]
        for proc in processes:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                self.fail("concurrent ledger child did not finish")
            self.assertEqual(proc.exitcode, 0)
        results.close()
        self.assertEqual(outcomes.count("MOCK_SEND_QUALIFIED"), 1, outcomes)
        self.assertEqual(sum("already_claimed_no_outer_attempt" in value for value in outcomes), 1, outcomes)
        journal = self.ledger().snapshot()
        self.assertEqual(sum(row["kind"] == "execution" for row in journal), 1)
        self.assertEqual(sum(row["kind"] == "attempt" for row in journal), 1)

    def test_all_roles_share_authorization_and_no_budget_can_be_borrowed(self):
        ledger = self.ledger()
        self.assertTrue(ledger.reserve("all_authorized_slots", self.budget["role_requests"], cumulative_tokens=1))
        for role, count in self.budget["role_requests"].items():
            if role == "closure":
                continue
            for _ in range(count):
                ledger.begin("all_authorized_slots", role, input_tokens=1)
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "reserved_role_exhausted"):
                ledger.begin("all_authorized_slots", role, input_tokens=1)
        journal = ledger.snapshot()
        attempts = [row for row in journal if row["kind"] == "attempt"]
        self.assertEqual(len(attempts), 17)
        self.assertEqual({role: sum(row["role"] == role for row in attempts)
                          for role in ("planner", "supplement", "gate", "realizer", "adherence")},
                         {"planner": 0, "supplement": 1, "gate": 8, "realizer": 4, "adherence": 4})
        self.assertFalse(ledger.reserve("borrow_old_balance", {"gate": 1}, cumulative_tokens=1))
        self.assertEqual(journal[0]["execution_scope"], self.scope)

    def test_role_and_total_reservation_limits_are_independent(self):
        for role in ("planner", "supplement", "gate", "realizer", "adherence", "closure"):
            with self.subTest(role=role):
                ledger = self.ledger(scope={**self.scope, "experiment_id": "MOCK_ROLE_" + role})
                self.assertFalse(ledger.reserve("too_many", {role: self.budget["role_requests"][role] + 1}, cumulative_tokens=1))
                self.assertFalse(any(row["kind"] == "attempt" for row in ledger.snapshot()))
        budget = deepcopy(self.budget)
        budget["total_requests"] = 2
        ledger = self.ledger(scope={**self.scope, "experiment_id": "MOCK_TOTAL"}, budget=budget)
        self.assertTrue(ledger.reserve("one", {"supplement": 1}, cumulative_tokens=1))
        self.assertTrue(ledger.reserve("two", {"gate": 1}, cumulative_tokens=1))
        self.assertFalse(ledger.reserve("three", {"adherence": 1}, cumulative_tokens=1))

    def test_invalid_local_budget_does_not_create_registry(self):
        budget = deepcopy(self.budget)
        budget["role_requests"]["supplement"] = 2
        with self.assertRaisesRegex(ValueError, "one_initial_and_one_supplement_maximum"):
            self.ledger(budget=budget)
        self.assertFalse(self.registry.exists())

    def test_invalid_final_input_does_not_consume_reserved_request(self):
        ledger = self.ledger()
        ledger.reserve("supplement", {"supplement": 1}, cumulative_tokens=1)
        before = ledger.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "input_budget_exceeded_no_truncation"):
            ledger.begin("supplement", "supplement", input_tokens=self.budget["role_capacity"]["supplement"]["max_input_tokens"] + 1)
        self.assertEqual(ledger.path.read_bytes(), before)
        self.assertEqual(ledger.begin("supplement", "supplement", input_tokens=1), 1)

    def test_reading_existing_results_does_not_send_or_modify_journal(self):
        ledger = self.ledger()
        ledger.reserve("supplement", {"supplement": 1}, cumulative_tokens=1)
        call = ledger.begin("supplement", "supplement", input_tokens=1)
        ledger.finish(call, status="RESPONSE_RECEIVED")
        before = ledger.path.read_bytes()
        restarted = self.ledger()
        self.assertEqual(restarted.snapshot(), ledger.snapshot())
        self.assertEqual(ledger.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
