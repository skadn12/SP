"""No-network tests of the independent three-call Miner-only scope."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from ..backends import BackendResponse
from ..core import atomic_json, sha256_file
from ..live_backends import ChallengerAPIConfig, OpenAICompatibleChallengerClient
from . import runtime as r


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.root_id = r.FIRST_ROOTS[0]
        self.root = r.root_path(self.run, self.root_id)
        self.ledger = self.new_ledger(self.root_id)
        self.calls = 0
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def new_ledger(self, root_id, **kwargs):
        return r.BoundedLedger(r.root_path(self.run, root_id), "freeze", self.run,
                               root_id, backend_mode="mock", **kwargs)

    def call(self):
        self.calls += 1
        return BackendResponse('{"intents":[]}', "fixture", {"transport_attempt_count": 1})

    def kwargs(self, root_id=None):
        root_id = root_id or self.root_id
        request_id = f"{root_id}-intent-miner"
        return dict(request_id=request_id, intent="h4_k3_intent_miner", attempt=1,
                    prompt_sha256="prompt", call=self.call,
                    budget_class="fresh_qa_generation", budget_limit=10000,
                    total_budget_limit=10000,
                    request_metadata={"max_tokens": 4096, "temperature": 0.1,
                                      "request_id": request_id, "probe_version": "fixture"},
                    backend_identity={"model": "fixture"})

    def test_completed_cache_exact_and_drift_guard(self):
        a = self.ledger.invoke(**self.kwargs())
        b = self.ledger.invoke(**self.kwargs())
        self.assertEqual(a.raw_text, b.raw_text)
        self.assertTrue(b.metadata["reused"])
        self.assertEqual(self.calls, 1)
        for field, value in (("prompt_sha256", "different"),
                             ("backend_identity", {"model": "other"}),
                             ("budget_limit", 22)):
            with self.subTest(field=field):
                args = self.kwargs(); args[field] = value
                with self.assertRaises(r.GuardrailStop):
                    self.ledger.invoke(**args)
        self.ledger.delegate.manifest_sha256 = "changed-input-config-or-code-identity"
        with self.assertRaisesRegex(r.GuardrailStop, "cached_request_contract_drift"):
            self.ledger.invoke(**self.kwargs())
        self.ledger.delegate.manifest_sha256 = "freeze"
        state = self.ledger.delegate.read(self.kwargs()["request_id"])
        Path(state["raw_response_path"]).write_text("tampered")
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**self.kwargs())

    def test_failed_logical_call_counted_and_not_resent(self):
        def fail():
            self.calls += 1
            raise TimeoutError("fixture timeout")
        args = {**self.kwargs(), "call": fail}
        for _ in range(2):
            with self.assertRaisesRegex(Exception, "fixture timeout"):
                self.ledger.invoke(**args)
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(r.audit_reservations(self.run)), 1)
        failure = self.root / "ledger/observed_failures" / (args["request_id"] + ".json")
        self.assertFalse(r.read(failure)["automatic_resend"])
        self.assertEqual(r.read(failure)["billing_status"], "unknown_no_completion_recorded")
        with self.assertRaisesRegex(r.GuardrailStop, "cached_request_contract_drift"):
            self.ledger.invoke(**{**args, "prompt_sha256": "changed"})
        failure.unlink()
        with self.assertRaisesRegex(r.GuardrailStop, "unknown_paid_status"):
            self.ledger.invoke(**args)

    def test_known_failure_does_not_block_other_scheduled_roots(self):
        def fail():
            self.calls += 1
            raise ValueError("ordinary observed failure")
        with self.assertRaises(ValueError):
            self.ledger.invoke(**{**self.kwargs(), "call": fail})
        for root_id in r.FIRST_ROOTS[1:]:
            self.new_ledger(root_id).invoke(**self.kwargs(root_id))
        self.assertEqual(self.calls, 3)
        states = [r.read(p) for p in r.audit_reservations(self.run)]
        self.assertEqual(len(states), 3)
        self.assertEqual(sum(s["status"] == "completed" for s in states), 2)
        self.assertTrue(all(s["budget_limit"] == 1 and s["total_budget_limit"] == 1 for s in states))

    def test_orphan_any_root_blocks_new_call_without_resend(self):
        class Interrupted(BaseException):
            pass
        def interrupt():
            self.calls += 1
            raise Interrupted("process interrupted after reservation")
        with self.assertRaises(Interrupted):
            self.ledger.invoke(**{**self.kwargs(), "call": interrupt})
        other_id = r.FIRST_ROOTS[1]
        with self.assertRaisesRegex(r.GuardrailStop, "unknown_paid_status"):
            self.new_ledger(other_id).invoke(**self.kwargs(other_id))
        with self.assertRaisesRegex(r.GuardrailStop, "unknown_paid_status"):
            self.ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(r.state_paths(self.run)), 1)

    def test_observed_failure_reservation_drift(self):
        def fail():
            raise TimeoutError("failed")
        with self.assertRaises(TimeoutError):
            self.ledger.invoke(**{**self.kwargs(), "call": fail})
        path = self.ledger.delegate._path(self.kwargs()["request_id"])
        state = r.read(path); state["created_at"] = "changed"
        atomic_json(path, state)
        with self.assertRaisesRegex(r.GuardrailStop, "observed_failure_reservation_drift"):
            r.audit_reservations(self.run)

    def test_token_guard_before_reservation(self):
        def preflight(*args):
            raise r.GuardrailStop("fixture token overflow")
        self.ledger.before_request = preflight
        with self.assertRaisesRegex(r.GuardrailStop, "token overflow"):
            self.ledger.invoke(**self.kwargs())
        self.assertEqual(len(r.state_paths(self.run)), 0)
        self.assertEqual(self.calls, 0)

    def test_input_config_code_sha_callback_before_reservation(self):
        for name in ("input.json", "config.json", "implementation.py"):
            with self.subTest(identity=name):
                frozen = self.run / name
                frozen.write_text("original")
                expected = sha256_file(frozen)
                frozen.write_text("changed")
                def verify(*args):
                    if sha256_file(frozen) != expected:
                        raise r.GuardrailStop("frozen_identity_drift")
                self.ledger.before_request = verify
                with self.assertRaisesRegex(r.GuardrailStop, "frozen_identity_drift"):
                    self.ledger.invoke(**self.kwargs())
                self.assertEqual(len(r.state_paths(self.run)), 0)
        self.assertEqual(self.calls, 0)

    def test_callback_order_and_success_is_saved_before_usage_guard(self):
        sequence = []
        request_id = self.kwargs()["request_id"]
        def before(got_id, metadata):
            self.assertEqual(got_id, request_id)
            self.assertEqual(metadata["assigned_mode"], r.MODE)
            self.assertIsNone(self.ledger.delegate.read(request_id))
            sequence.append("verified")
        def send():
            self.assertEqual(self.ledger.delegate.read(request_id)["status"], "reserved")
            sequence.append("sent")
            return self.call()
        def after(got_id, response):
            state = self.ledger.delegate.read(got_id)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(Path(state["raw_response_path"]).read_text(), response.raw_text)
            sequence.append("saved_before_usage_check")
            raise r.GuardrailStop("fixture service count drift")
        self.ledger.before_request, self.ledger.after_response = before, after
        with self.assertRaisesRegex(r.GuardrailStop, "service count drift"):
            self.ledger.invoke(**{**self.kwargs(), "call": send})
        self.assertEqual(sequence, ["verified", "sent", "saved_before_usage_check"])
        self.assertEqual(self.calls, 1)
        self.assertFalse((self.root / "ledger/observed_failures" / f"{request_id}.json").exists())
        with self.assertRaisesRegex(r.GuardrailStop, "service count drift"):
            self.ledger.invoke(**{**self.kwargs(), "call": send})
        self.assertEqual(self.calls, 1)

    def test_live_mode_requires_identity_token_and_usage_callbacks(self):
        with self.assertRaisesRegex(r.GuardrailStop, "requires_freeze_token_and_usage_guards"):
            r.BoundedLedger(self.root, "freeze", self.run, self.root_id)
        with self.assertRaisesRegex(r.GuardrailStop, "requires_freeze_token_and_usage_guards"):
            r.BoundedLedger(self.root, "freeze", self.run, self.root_id,
                            before_request=lambda *args: None)

    def test_root_and_total_budget_limits_cannot_borrow_v1_budget(self):
        self.assertEqual((r.PLAN_TOTAL, r.LIMIT_TOTAL, r.LIMIT_PER_ROOT), (3, 3, 1))
        with patch.object(r, "LIMIT_PER_ROOT", 0):
            with self.assertRaisesRegex(r.GuardrailStop, "root_logical_budget_exhausted"):
                self.ledger.invoke(**self.kwargs())
        with patch.object(r, "LIMIT_TOTAL", 0):
            with self.assertRaisesRegex(r.GuardrailStop, "experiment_logical_budget_exhausted"):
                self.ledger.invoke(**self.kwargs())
        for root_id in r.FIRST_ROOTS:
            self.new_ledger(root_id).invoke(**self.kwargs(root_id))
        self.assertEqual(len(r.audit_reservations(self.run)), 3)
        self.assertEqual(self.calls, 3)
        with self.assertRaises(r.GuardrailStop):
            self.new_ledger("cc-train-episode-fourth")
        args = self.kwargs(); args["request_id"] += "-second"
        with self.assertRaises(r.GuardrailStop):
            self.ledger.invoke(**args)
        self.assertEqual(self.calls, 3)

    def test_forbidden_writer_repair_solver_and_second_miner_rejected_before_send(self):
        atomic_json(self.root / "reasoning_contracts.json", [{"intent_id": "I001"}])
        names = ("contract_guided-generator-I001", "free_exploration-generator-I001",
                 "contract_guided-repair-I001", "free_exploration-repair-I001",
                 "solver", "planner", "realizer", "memory", "training",
                 "intent-miner-repair", "intent-miner-2")
        for name in names:
            with self.subTest(role=name):
                args = self.kwargs(); args["request_id"] = f"{self.root_id}-{name}"
                with self.assertRaises(r.GuardrailStop):
                    self.ledger.invoke(**args)
        self.assertEqual(self.calls, 0)
        self.assertEqual(r.state_paths(self.run), [])

    def test_original_miner_request_fields_required(self):
        cases = (("intent", "h4_k3_candidate_generator"), ("intent", "miner"),
                 ("attempt", 2), ("attempt", True), ("budget_class", "solver"))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                args = self.kwargs(); args[field] = value
                with self.assertRaises(r.GuardrailStop):
                    self.ledger.invoke(**args)
        for field, value in (("max_tokens", 8192), ("max_tokens", 4096.0),
                             ("temperature", 0.8), ("request_id", "other"), ("seed", 0)):
            with self.subTest(metadata=field):
                args = self.kwargs(); args["request_metadata"][field] = value
                with self.assertRaises(r.GuardrailStop):
                    self.ledger.invoke(**args)
        self.assertEqual(self.calls, 0)

    def test_root_directory_and_reservation_inventory_scope(self):
        with self.assertRaises(r.GuardrailStop):
            r.BoundedLedger(self.run / "elsewhere", "freeze", self.run, self.root_id)
        self.ledger.invoke(**self.kwargs())
        state = self.ledger.delegate._path(self.kwargs()["request_id"])
        unexpected = self.run / "qa/shards/shard-01/roots" / self.root_id / "ledger/requests" / state.name
        unexpected.parent.mkdir(parents=True)
        state.replace(unexpected)
        with self.assertRaisesRegex(r.GuardrailStop, "root_scope_or_budget_violation"):
            r.audit_reservations(self.run)

    def test_unique_lock_and_guardrail_bypasses_generator_exception_handler(self):
        self.assertTrue(issubclass(r.GuardrailStop, BaseException))
        self.assertFalse(issubclass(r.GuardrailStop, Exception))
        with r.exclusive_run(self.run):
            with self.assertRaisesRegex(RuntimeError, "already_running"):
                with r.exclusive_run(self.run):
                    pass
        with r.exclusive_run(self.run):
            pass

    def test_completed_checkpoint_hash_and_added_file_drift(self):
        self.assertFalse(r.verify_checkpoint(self.root))
        (self.root / "intents.json").write_text('{"fixture":true}')
        r.checkpoint_root(self.root)
        self.assertTrue(r.verify_checkpoint(self.root))
        r.checkpoint_root(self.root)
        (self.root / "intents.json").write_text("changed")
        with self.assertRaisesRegex(r.GuardrailStop, "completed_artifact_drift"):
            r.verify_checkpoint(self.root)
        (self.root / "intents.json").write_text('{"fixture":true}')
        (self.root / "candidate_pool.json").write_text("[]")
        with self.assertRaisesRegex(r.GuardrailStop, "completed_artifact_inventory_drift"):
            r.verify_checkpoint(self.root)

    def test_original_client_429_retries_are_one_logical_reservation(self):
        config = ChallengerAPIConfig("https://fixture.invalid/v1", "not-a-real-key",
                                     "Qwen/Qwen3-14B", temperature=0.1,
                                     enable_thinking=False, timeout_seconds=180)
        body = {"choices": [{"message": {"content": '{"intents":[]}'},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
        response = io.BytesIO(json.dumps(body).encode())
        response.status = 200
        failures = [urllib.error.HTTPError("https://fixture.invalid", 429, "limited", {}, None)
                    for _ in range(2)]
        client = r.ObservedClient(OpenAICompatibleChallengerClient(config),
                                  self.run / "audit/transport.jsonl", r.MODE, self.root_id)
        with (patch("urllib.request.urlopen", side_effect=[*failures, response]) as mocked,
              patch("scripts.current_method.restart_n5_spell.live_backends.time.sleep")):
            result = self.ledger.invoke(**{**self.kwargs(),
                "call": lambda: client.complete(prompt="fixture", max_tokens=4096, json_object=True)})
        self.assertEqual(len(r.state_paths(self.run)), 1)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(result.metadata["transport_attempt_count"], 3)
        self.assertEqual(result.metadata["rate_limit_retry_count"], 2)
        for call in mocked.call_args_list:
            payload = json.loads(call.args[0].data)
            self.assertNotIn("seed", payload)
            self.assertEqual(payload["temperature"], 0.1)
            self.assertIs(payload["enable_thinking"], False)
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertEqual(call.kwargs["timeout"], 180)
        events = [json.loads(line) for line in (self.run / "audit/transport.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["event"] == "http_attempt_started" for row in events), 3)


if __name__ == "__main__":
    unittest.main()
