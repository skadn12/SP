"""Offline control-flow regression: real runner + recorded mock transport.

Fixtures exercise contracts, not Qwen semantic quality. Socket creation is
forbidden and all run/inventory writes stay in temporary directories.
"""
from contextlib import ExitStack
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import runner
import audit_sequential

_spec = importlib.util.spec_from_file_location("v15_scripted_transport_fixture",
    runner.HERE.parent / "h4_feedback_training_pool_v15_expansion/test_end_to_end.py")
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
FROZEN = runner.HERE.parent / "h4_feedback_training_pool_v15_expansion/runs/end_to_end_5roots_01"
ROOT = "cc-train-episode-104814"


class SequentialProvider(fixtures.ScriptedProvider):
    def role_verdict(self, ctx, *, realized=False, negative=False):
        history = ctx.get("cumulative_history", ctx.get("branch_history"))
        anchor = history[-1]
        is_b = ctx["route"] == runner.ROUTES[1]
        value = {"role": ctx["route"], "satisfied": not negative,
                 "primary_event": "MOCK ONLY: independent appointment" if is_b else "MOCK ONLY: old-event progress",
                 "relation_to_history": "old_event_substep" if negative else "distinct_event" if is_b else "continuation",
                 "reason": "MOCK ONLY: scripted route role judgment, not semantic evidence.",
                 "history_evidence": [{"turn_ref": anchor["turn_ref"], "quote": anchor["text"]}]}
        if realized:
            value["new_evidence"] = [{"turn_ref": "NEW:T001", "quote": ctx["new_scene"]["turns"][0]["text"]}]
        return value

    def state_extractor(self, ctx):
        if self.mode == "A_rejected" and ctx["route"] == runner.ROUTES[0]:
            return {"MOCK_invalid_state": True}
        return super().state_extractor(ctx)

    def event_state_extractor(self, ctx):
        value = super().event_state_extractor(ctx)
        negative = self.mode == "B_role_replan" and ctx["route"] == runner.ROUTES[1] and "B_role_replan" not in self.used_injections
        if negative:
            self.used_injections.add("B_role_replan")
        value["route_fulfillment"] = self.role_verdict(ctx, negative=negative)
        return value

    def realizer(self, ctx):
        value = super().realizer(ctx)
        if ctx["route"] == runner.ROUTES[1] and self.mode == "B_schema_repair" and self.mode not in self.used_injections:
            self.used_injections.add(self.mode)
            value["turns"][2]["speaker"] = "MOCK_INVALID_SPEAKER"
        if ctx["route"] == runner.ROUTES[1] and self.mode == "B_copy_A" and self.mode not in self.used_injections:
            self.used_injections.add(self.mode)
            value["turns"] = [{"speaker": turn["speaker"], "text": turn["text"]}
                              for turn in ctx["cumulative_history"][-12:]]
        return value

    def scene_reviewer(self, ctx):
        value = super().scene_reviewer(ctx)
        negative = self.mode == "B_role_repair" and ctx["route"] == runner.ROUTES[1] and self.mode not in self.used_injections
        if negative:
            self.used_injections.add(self.mode)
        value["route_fulfillment"] = self.role_verdict(ctx, realized=True, negative=negative)
        return value


def initialize_mock(caller, run, fake):
    caller.run, caller.limit = run, 200
    caller.profile, caller.factory = object(), fake.factory
    caller.capacity = lambda *_args: {"status": "PASS", "mock_capacity_only": True}
    caller.annotate, caller.tokenizer = lambda value: value, object()
    caller.http_counter = caller.request_counter = 0
    caller.last_http_completed = None


class SequentialFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v18-sequential-mock-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        for name in ("sources", "feedback"):
            shutil.copytree(FROZEN / name, self.run / name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("socket.socket", side_effect=AssertionError("network_forbidden_in_mock")))
        self.stack.enter_context(patch.object(runner, "MIN_HTTP_GAP_SECONDS", 0))

    def exercise(self, mode=None):
        fake = SequentialProvider(mode)
        caller = runner.Caller.__new__(runner.Caller)
        initialize_mock(caller, self.run, fake)
        results = runner.run_root(caller, self.run, ROOT, fixtures.fixture_cards(1), target_scenes=1)
        frozen = runner.base.read(self.run / "sources" / f"{ROOT}.json")["turns"]
        accepted_a = runner.base.read(self.run / "roots" / ROOT / "A/cumulative_dialogue.json")["turns"]
        expected_a = 0 if mode == "A_rejected" else 1
        self.assertEqual([item["accepted_scenes"] for item in results], [expected_a, 1], results)
        self.assertEqual(accepted_a[:len(frozen)], frozen)
        for observation in fake.observed:
            ctx = observation["context"]
            history = ctx.get("cumulative_history", ctx.get("branch_history"))
            self.assertEqual(history, frozen if ctx["route"] == runner.ROUTES[0] else accepted_a,
                             observation["role"])
        actual = runner.base.read(self.run / "sources_merged" / f"{ROOT}.json")["turns"]
        self.assertEqual(actual[:len(accepted_a)], accepted_a)
        self.assertEqual(len(actual), len(frozen) + 12 * (expected_a + 1))
        self.assertEqual(len(set(turn["turn_ref"] for turn in actual)), len(actual))
        b_input = runner.base.read(self.run / "roots" / ROOT / "B/INPUT.json")
        self.assertEqual(b_input["turns"], accepted_a)
        for directory in (self.run / "requests").iterdir():
            payload = runner.base.read(directory / "REQUEST.json")
            proof = runner.base.read(directory / "PROOF.json")
            attempt = runner.base.read(directory / "HTTP_ATTEMPT.json")
            self.assertEqual(attempt["payload_sha256"], runner.base.digest(payload))
            self.assertEqual(proof["payload_sha256"], runner.base.digest(payload))
            self.assertTrue(attempt["actual_payload_equals_frozen"])
            if proof["role"] in ("planner", "realizer"):
                receipt = proof["feedback_receipt"]
                self.assertTrue(receipt["feedback_appended"])
                self.assertFalse(receipt["current_generation_feedback_used"])
                self.assertEqual(receipt["request_payload_sha256"], runner.base.digest(payload))
        return fake, results

    def test_B_all_stages_receive_exact_accepted_A_and_frozen_feedback(self):
        fake, _ = self.exercise()
        roles = {item["role"] for item in fake.observed if item["context"]["route"] == runner.ROUTES[1]}
        self.assertEqual(roles, {"state_extractor", "planner", "event_state_extractor", "realizer", "scene_reviewer"})

    def test_failed_A_is_never_injected_into_B(self):
        self.exercise("A_rejected")

    def test_B_schema_repair_keeps_the_same_complete_history(self):
        fake, _ = self.exercise("B_schema_repair")
        calls = [row for row in fake.observed if row["role"] == "realizer" and row["context"]["route"] == runner.ROUTES[1]]
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[1]["context"]["targeted_repair"])

    def test_B_old_event_substep_replans_before_realization(self):
        fake, _ = self.exercise("B_role_replan")
        calls = [row for row in fake.observed if row["context"]["route"] == runner.ROUTES[1]]
        self.assertEqual([row["role"] for row in calls],
            ["state_extractor", "planner", "event_state_extractor", "planner", "event_state_extractor", "realizer", "scene_reviewer"])
        self.assertTrue(calls[3]["context"]["targeted_repair"]["plan_checks"][0]["route_role_check"])

    def test_B_wrong_role_in_realization_triggers_bounded_repair(self):
        fake, _ = self.exercise("B_role_repair")
        calls = [row for row in fake.observed if row["role"] == "realizer" and row["context"]["route"] == runner.ROUTES[1]]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["context"]["targeted_repair"]["route_fulfillment"]["relation_to_history"], "old_event_substep")

    def test_B_verbatim_A_copy_rejected_before_reviewer_then_repaired(self):
        fake, _ = self.exercise("B_copy_A")
        calls = [row["role"] for row in fake.observed if row["context"]["route"] == runner.ROUTES[1]]
        self.assertEqual(calls[-3:], ["realizer", "realizer", "scene_reviewer"])
        failures = [runner.base.read(path) for path in self.run.glob("requests/*/ACCEPTANCE.json")]
        self.assertTrue(any(item.get("stage") == "duplicate" and item["status"] == "REJECT" for item in failures))

    def test_audit_does_not_mark_incomplete_run_passed(self):
        delivery = self.run / "audit_only_delivery"
        incomplete = delivery / "runs/incomplete"
        incomplete.mkdir(parents=True)
        (incomplete / "MANIFEST.json").write_text("{}")
        with patch.object(audit_sequential, "HERE", delivery):
            with self.assertRaisesRegex(ValueError, "incomplete_run_missing:REPORT.json"):
                audit_sequential.audit_run(incomplete)
        self.assertFalse((incomplete / "FINAL_AUDIT.json").exists())


class FullMockLiveAuditTests(unittest.TestCase):
    def test_real_prepare_live_reservation_and_independent_audit(self):
        with tempfile.TemporaryDirectory(prefix="v18-complete-mock-") as temporary, ExitStack() as stack:
            temp = Path(temporary)
            delivery = temp / "offline-deliveries/h4_sequential_ab_v18"
            delivery.mkdir(parents=True)
            for path in runner.HERE.glob("*.py"):
                shutil.copy2(path, delivery / path.name)
            (delivery / "POLICY.md").write_text("MOCK ONLY: artifact binding fixture.\n")
            unique = delivery.parent / "h4_r02_source_unique_v2"
            unique.mkdir()
            shutil.copy2(runner.HERE.parent / "h4_r02_source_unique_v2/unique_events.py", unique / "unique_events.py")
            run = delivery / "runs/MOCK_ONLY"
            fake = SequentialProvider()
            def before_request():
                self.assertTrue((run / "SOURCE_RESERVATIONS.json").exists())
                self.assertTrue((run / "LIVE_STARTED.json").exists())
            fake.before_request = before_request
            original_collect = runner.source_inventory.collect_prior
            prior_paths = {key: temp / key for key in inspect.signature(original_collect).parameters if key != "current_runs"}
            def isolated_inventory():
                return original_collect(delivery / "runs", **prior_paths)
            stack.enter_context(patch.object(runner, "HERE", delivery))
            stack.enter_context(patch.object(audit_sequential, "HERE", delivery))
            stack.enter_context(patch.object(audit_sequential, "r", runner))
            stack.enter_context(patch.object(runner, "LOCK_PATH", temp / "inventory.lock"))
            stack.enter_context(patch.object(runner, "MIN_HTTP_GAP_SECONDS", 0))
            stack.enter_context(patch.object(runner.source_inventory, "collect_prior", side_effect=isolated_inventory))
            stack.enter_context(patch.object(runner.Caller, "__init__", lambda caller, run_path: initialize_mock(caller, run_path, fake)))
            stack.enter_context(patch("socket.socket", side_effect=AssertionError("network_forbidden_in_mock")))
            runner.prepare(run, scenes=1, roots=[ROOT])
            output = runner.live(run)
            self.assertEqual(output["accepted_scenes"], 2, output)
            audit = audit_sequential.audit_run(run)
            self.assertEqual(audit["status"], "PASS", audit["failures"])
            self.assertEqual(audit["merged_roots_verified"], 1)
            self.assertEqual(audit["source_overlap_count"], 0)
            # A prompt tamper cannot survive independent request revalidation.
            directory = next(path for path in (run / "requests").iterdir()
                if runner.base.read(path / "PROOF.json").get("route") == runner.ROUTES[1])
            payload = runner.base.read(directory / "REQUEST.json")
            payload["messages"][0]["content"] += "MOCK_TAMPER"
            (directory / "REQUEST.json").write_text(json.dumps(payload))
            for name in ("FINAL_AUDIT.json", "SEQUENTIAL_SUMMARY.json", "SEQUENTIAL_SUMMARY.md"):
                (run / name).unlink()
            negative = audit_sequential.audit_run(run)
            self.assertEqual(negative["status"], "FAIL")
            self.assertTrue(any("request_payload_hash_mismatch" in item["error"] for item in negative["failures"]))


if __name__ == "__main__":
    unittest.main()
