"""No-network tests for paired identity, order, budgets and crash recovery."""
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
        # Deliberately identical per-root prompt hashes across arms prove that
        # the arm dimension itself, rather than different prompts, isolates cache.
        self.pairing = [{**row, "prompt_sha256": "prompt-" + row["root_id"],
                         "max_output_tokens": 4096, "logical_call_limit": 1}
                        for row in r.SCHEDULE]
        self.write_pairing()
        atomic_json(self.run / "schedule.json", {"version": r.VERSION, "schedule": list(r.SCHEDULE)})
        self.calls = 0
        for name in ("urllib.request.urlopen", "socket.create_connection"):
            blocker = patch(name, side_effect=AssertionError("network forbidden"))
            blocker.start(); self.addCleanup(blocker.stop)

    def write_pairing(self):
        (self.run / "pairing_manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.pairing))

    def new_ledger(self, index=0, **kwargs):
        row = self.pairing[index]
        return r.BoundedLedger(r.root_path(self.run, row["arm_id"], row["root_id"]),
            "freeze", self.run, row["arm_id"], row["root_id"], backend_mode="mock", **kwargs)

    def call(self):
        self.calls += 1
        return BackendResponse('{"intents":[]}', "fixture", {"transport_attempt_count": 1})

    def kwargs(self, index=0):
        row = self.pairing[index]
        return dict(request_id=row["request_id"], intent="h4_k3_intent_miner", attempt=1,
                    prompt_sha256=row["prompt_sha256"], call=self.call,
                    budget_class="fresh_qa_generation", budget_limit=10000,
                    total_budget_limit=10000,
                    request_metadata={"max_tokens": 4096, "temperature": 0.1,
                                      "request_id": row["request_id"], "probe_version": "fixture"},
                    backend_identity={"model": "fixture"})

    def test_same_request_id_and_prompt_never_cross_arm_cache(self):
        for index, row in enumerate(self.pairing):
            ledger = self.new_ledger(index)
            self.assertNotIn("reused", ledger.invoke(**self.kwargs(index)).metadata)
            cached = ledger.invoke(**self.kwargs(index))
            self.assertTrue(cached.metadata["reused"])
            state = ledger.delegate.read(row["request_id"])
            self.assertEqual(state["request_metadata"]["arm_id"], row["arm_id"])
            self.assertEqual(state["request_metadata"]["global_request_id"], row["global_request_id"])
            self.assertEqual(state["budget_limit"], 1)
            self.assertEqual(state["total_budget_limit"], 1)
        self.assertEqual(self.calls, 6)
        self.assertEqual(len(r.audit_reservations(self.run)), 6)
        events = r._read_rows(self.run / "audit/call_events.jsonl")
        sent = [row for row in events if row["event"] == "new_logical_call"]
        self.assertEqual([r._identity(row) for row in sent], list(r.SCHEDULE))
        self.assertEqual([row["experiment_call_index"] for row in sent], list(range(1, 7)))
        self.assertTrue(all(set(r.IDENTITY_KEYS) <= row.keys() for row in events))

    def test_fixed_order_cannot_skip_or_reorder_pairs(self):
        for index in (1, 2, 3, 4, 5):
            with self.subTest(index=index):
                with self.assertRaisesRegex(r.GuardrailStop, "out_of_frozen_order"):
                    self.new_ledger(index).invoke(**self.kwargs(index))
        self.assertEqual(self.calls, 0)
        self.assertEqual(r.state_paths(self.run), [])
        self.new_ledger(0).invoke(**self.kwargs(0))
        with self.assertRaisesRegex(r.GuardrailStop, "out_of_frozen_order"):
            self.new_ledger(2).invoke(**self.kwargs(2))
        self.new_ledger(1).invoke(**self.kwargs(1))
        self.assertEqual(self.calls, 2)

    def test_scope_caps_forbid_third_root_other_arm_and_second_sampling(self):
        self.assertEqual((r.PLAN_TOTAL, r.LIMIT_TOTAL, r.LIMIT_PER_ROOT,
                          r.LIMIT_PER_ARM, r.LIMIT_PER_PAIR), (6, 6, 3, 2, 1))
        row = self.pairing[0]
        for arm_id, root_id in (("A_NEW", row["root_id"]),
                                (row["arm_id"], "cc-train-episode-197635")):
            with self.assertRaises(r.GuardrailStop):
                r.BoundedLedger(r.root_path(self.run, arm_id, root_id), "freeze",
                                self.run, arm_id, root_id, backend_mode="mock")
        ledger = self.new_ledger()
        for suffix in ("contract_guided-generator-I001", "free_exploration-generator-I001",
                       "contract_guided-repair-I001", "solver", "planner", "realizer",
                       "memory", "training", "intent-miner-2", "intent-miner-repair"):
            args = self.kwargs(); args["request_id"] = row["root_id"] + "-" + suffix
            with self.assertRaises(r.GuardrailStop):
                ledger.invoke(**args)
        for field, value in (("attempt", 2), ("attempt", True),
                             ("intent", "h4_k3_candidate_generator"), ("budget_class", "solver")):
            with self.assertRaises(r.GuardrailStop):
                ledger.invoke(**{**self.kwargs(), field: value})
        self.assertEqual(self.calls, 0)

    def test_six_exact_pairing_rows_and_schedule_required(self):
        original = list(self.pairing)
        for rows in (original[:-1], original + [original[0]],
                     [original[1], original[0], *original[2:]]):
            self.pairing = rows; self.write_pairing()
            with self.assertRaisesRegex(r.GuardrailStop, "fixed_pairing_or_schedule_drift"):
                self.new_ledger()
        self.pairing = original; self.write_pairing()
        atomic_json(self.run / "schedule.json", {"schedule": list(reversed(r.SCHEDULE))})
        with self.assertRaisesRegex(r.GuardrailStop, "fixed_pairing_or_schedule_drift"):
            self.new_ledger()
        (self.run / "schedule.json").unlink()
        with self.assertRaisesRegex(r.GuardrailStop, "pairing_and_schedule_required"):
            self.new_ledger()

    def test_pairing_and_prompt_drift_block_before_reservation(self):
        ledger = self.new_ledger()
        with self.assertRaisesRegex(r.GuardrailStop, "prompt_outside_frozen_pairing"):
            ledger.invoke(**{**self.kwargs(), "prompt_sha256": "other"})
        self.pairing[0]["unused_frozen_sidecar_field"] = "changed"
        self.write_pairing()
        with self.assertRaisesRegex(r.GuardrailStop, "changed_after_ledger_creation"):
            ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 0)
        self.assertEqual(r.state_paths(self.run), [])

    def test_frozen_manifest_change_after_a_call_blocks_next_pair(self):
        self.new_ledger().invoke(**self.kwargs())
        self.pairing[0]["unused_frozen_sidecar_field"] = "changed"
        self.write_pairing()
        with self.assertRaisesRegex(r.GuardrailStop, "reservation_frozen_contract_drift"):
            self.new_ledger(1).invoke(**self.kwargs(1))
        self.assertEqual(self.calls, 1)

    def test_completed_cache_verifies_raw_backend_and_manifest(self):
        ledger = self.new_ledger()
        ledger.invoke(**self.kwargs())
        with self.assertRaisesRegex(r.GuardrailStop, "cached_request_contract_drift"):
            ledger.invoke(**{**self.kwargs(), "backend_identity": {"model": "other"}})
        ledger.delegate.manifest_sha256 = "changed-code-config-or-input"
        with self.assertRaisesRegex(r.GuardrailStop, "cached_request_contract_drift"):
            ledger.invoke(**self.kwargs())
        ledger.delegate.manifest_sha256 = "freeze"
        state = ledger.delegate.read(self.pairing[0]["request_id"])
        Path(state["raw_response_path"]).write_text("changed")
        with self.assertRaisesRegex(r.GuardrailStop, "completed_cache_verification_failed"):
            ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 1)

    def test_failure_counts_and_replays_while_remaining_schedule_continues(self):
        def failed():
            self.calls += 1
            raise TimeoutError("ordinary fixture timeout")
        ledger = self.new_ledger()
        for _ in range(2):
            with self.assertRaisesRegex(Exception, "ordinary fixture timeout"):
                ledger.invoke(**{**self.kwargs(), "call": failed})
        self.assertEqual(self.calls, 1)
        for index in range(1, 6):
            self.new_ledger(index).invoke(**self.kwargs(index))
        states = [r.read(path) for path in r.audit_reservations(self.run)]
        self.assertEqual(self.calls, 6)
        self.assertEqual(len(states), 6)
        self.assertEqual(sum(row["status"] == "completed" for row in states), 5)
        failure = r.read(ledger.root_dir / "ledger/observed_failures" /
                         (self.pairing[0]["request_id"] + ".json"))
        self.assertEqual(failure["global_request_id"], self.pairing[0]["global_request_id"])
        self.assertFalse(failure["automatic_resend"])

    def test_orphan_blocks_every_arm_and_never_resends(self):
        class Interrupted(BaseException):
            pass
        def interrupted():
            self.calls += 1
            raise Interrupted("interrupted after reservation")
        ledger = self.new_ledger()
        with self.assertRaises(Interrupted):
            ledger.invoke(**{**self.kwargs(), "call": interrupted})
        for index in range(6):
            with self.assertRaisesRegex(r.GuardrailStop, "unknown_paid_status"):
                self.new_ledger(index).invoke(**self.kwargs(index))
        self.assertEqual(self.calls, 1)

    def test_moving_or_removing_reservations_cannot_restore_budget(self):
        ledger = self.new_ledger()
        ledger.invoke(**self.kwargs())
        original = ledger.delegate._path(self.pairing[0]["request_id"])
        moved = self.run / "backup/ledger/requests" / original.name
        moved.parent.mkdir(parents=True)
        original.replace(moved)
        self.assertIn(moved, r.state_paths(self.run))
        with self.assertRaisesRegex(r.GuardrailStop, "reservation_directory_scope_drift"):
            self.new_ledger(1).invoke(**self.kwargs(1))
        moved.rename(moved.with_suffix(".archived"))
        self.assertEqual(r.state_paths(self.run), [])
        with self.assertRaisesRegex(r.GuardrailStop, "event_reservation_inventory_mismatch"):
            self.new_ledger(0).invoke(**self.kwargs())
        self.assertEqual(self.calls, 1)

    def test_copying_completed_reservation_across_arms_is_rejected(self):
        ledger = self.new_ledger()
        ledger.invoke(**self.kwargs())
        state = ledger.delegate.read(self.pairing[0]["request_id"])
        target = self.new_ledger(1).delegate._path(self.pairing[1]["request_id"])
        atomic_json(target, state)
        with self.assertRaisesRegex(r.GuardrailStop, "reservation_directory_scope_drift"):
            r.audit_reservations(self.run)
        self.assertEqual(self.calls, 1)

    def test_global_event_without_request_file_is_guarded(self):
        r.append(self.run / "audit/call_events.jsonl", {
            **r.SCHEDULE[0], "event": "new_logical_call", "experiment_call_index": 1,
        })
        with self.assertRaisesRegex(r.GuardrailStop, "event_reservation_inventory_mismatch"):
            self.new_ledger().invoke(**self.kwargs())
        self.assertEqual(self.calls, 0)

    def test_sha_and_token_callbacks_run_before_any_reservation(self):
        for label in ("input", "configuration", "implementation", "token"):
            with self.subTest(label=label):
                path = self.run / (label + ".txt")
                path.write_text("frozen"); expected = sha256_file(path)
                path.write_text("changed")
                def verify(request_id, metadata):
                    self.assertEqual(metadata["global_request_id"], self.pairing[0]["global_request_id"])
                    self.assertEqual(metadata["arm_id"], self.pairing[0]["arm_id"])
                    if sha256_file(path) != expected:
                        raise r.GuardrailStop(label + " preflight drift")
                with self.assertRaisesRegex(r.GuardrailStop, "preflight drift"):
                    self.new_ledger(before_request=verify).invoke(**self.kwargs())
        self.assertEqual(self.calls, 0)
        self.assertEqual(r.state_paths(self.run), [])
        self.assertFalse((self.run / "audit/call_events.jsonl").exists())

    def test_completed_response_saved_before_usage_block_and_cache_rechecks(self):
        def after(request_id, response):
            state = ledger.delegate.read(request_id)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(Path(state["raw_response_path"]).read_text(), response.raw_text)
            raise r.GuardrailStop("usage mismatch")
        ledger = self.new_ledger(after_response=after)
        for _ in range(2):
            with self.assertRaisesRegex(r.GuardrailStop, "usage mismatch"):
                ledger.invoke(**self.kwargs())
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(r.audit_reservations(self.run)), 1)

    def test_original_miner_config_and_live_callbacks_required(self):
        ledger = self.new_ledger()
        for field, value in (("max_tokens", 8192), ("max_tokens", 4096.0),
                             ("temperature", 0.8), ("request_id", "other"), ("seed", 0)):
            args = self.kwargs(); args["request_metadata"][field] = value
            with self.assertRaises(r.GuardrailStop):
                ledger.invoke(**args)
        row = self.pairing[0]
        with self.assertRaisesRegex(r.GuardrailStop, "requires_freeze_token_and_usage_guards"):
            r.BoundedLedger(ledger.root_dir, "freeze", self.run, row["arm_id"], row["root_id"])
        self.assertEqual(self.calls, 0)

    def test_run_lock_and_completed_checkpoint_drift(self):
        self.assertFalse(issubclass(r.GuardrailStop, Exception))
        with r.exclusive_run(self.run):
            with self.assertRaisesRegex(RuntimeError, "already_running"):
                with r.exclusive_run(self.run):
                    pass
        ledger = self.new_ledger()
        self.assertFalse(r.verify_checkpoint(ledger.root_dir))
        artifact = ledger.root_dir / "intents.json"
        artifact.write_text("fixture")
        r.checkpoint_root(ledger.root_dir)
        self.assertTrue(r.verify_checkpoint(ledger.root_dir))
        artifact.write_text("changed")
        with self.assertRaisesRegex(r.GuardrailStop, "completed_artifact_drift"):
            r.verify_checkpoint(ledger.root_dir)
        artifact.write_text("fixture")
        (ledger.root_dir / "candidate_pool.json").write_text("[]")
        with self.assertRaisesRegex(r.GuardrailStop, "inventory_drift"):
            r.verify_checkpoint(ledger.root_dir)

    def test_original_http_429_retries_remain_one_logical_call(self):
        config = ChallengerAPIConfig("https://fixture.invalid/v1", "not-a-real-key",
                                     "Qwen/Qwen3-14B", temperature=0.1,
                                     enable_thinking=False, timeout_seconds=180)
        body = {"choices": [{"message": {"content": '{"intents":[]}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
        response = io.BytesIO(json.dumps(body).encode()); response.status = 200
        errors = [urllib.error.HTTPError("https://fixture.invalid", 429, "limited", {}, None)
                  for _ in range(2)]
        row = self.pairing[0]
        client = r.ObservedClient(OpenAICompatibleChallengerClient(config),
            self.run / "audit/transport.jsonl", row["arm_id"], row["root_id"])
        with (patch("urllib.request.urlopen", side_effect=[*errors, response]) as mocked,
              patch("scripts.current_method.restart_n5_spell.live_backends.time.sleep")):
            result = self.new_ledger().invoke(**{**self.kwargs(),
                "call": lambda: client.complete(prompt="fixture", max_tokens=4096, json_object=True)})
        self.assertEqual(len(r.audit_reservations(self.run)), 1)
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


if __name__ == "__main__":
    unittest.main()
