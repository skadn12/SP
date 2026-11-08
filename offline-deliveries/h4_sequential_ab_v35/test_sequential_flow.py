"""Offline control-flow regression: real runner + recorded mock transport.

Fixtures exercise contracts, not Qwen semantic quality. Socket creation is
forbidden and all run/inventory writes stay in temporary directories.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
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
    def factory(self, profile, *, on_http_attempt):
        old = super().factory(profile, on_http_attempt=on_http_attempt)
        owner = self
        class Client:
            def complete(self, *, prompt, max_tokens, json_object):
                leaf_repair = "\nREVIEW_FORMAT_REPAIR:\n" in prompt and any(mode in prompt for mode in ("exact_leaf_patch_v33", "exact_leaf_patch_v34"))
                scanner_patch = "\nSCANNER_PATCH_INPUT_JSON\n" in prompt
                if "\nPURPOSE_INPUT_JSON\n" not in prompt and not leaf_repair and not scanner_patch:
                    return old.complete(prompt=prompt, max_tokens=max_tokens, json_object=json_object)
                actual = {"model": runner.MODEL, "temperature": .25, "max_tokens": max_tokens,
                          "enable_thinking": False, "response_format": {"type": "json_object"},
                          "messages": [{"role": "user", "content": prompt}]}
                owner.before_request()
                on_http_attempt(actual)
                if scanner_patch:
                    ctx = audit_sequential.decode_context(actual, "\nEVENT_STATE_INPUT_JSON\n")
                    spec = json.JSONDecoder().raw_decode(prompt.split("\nSCANNER_PATCH_INPUT_JSON\n", 1)[1])[0]
                    owner.role_counts["event_state_extractor"] += 1
                    owner.observed.append({"role": "event_state_extractor", "context": ctx, "payload": actual})
                    operations = []
                    for auth in spec["authorized_repairs"]:
                        assert auth["op"] == "add_conflict", "unexpected_offline_patch_operation"
                        operations.append({"op": auth["op"], "path": auth["path"],
                            "value": deepcopy(owner.preserved_event_negative["conflict"])})
                    value = {**spec["identity"], "repair_patches": operations}
                    return SimpleNamespace(raw_text=json.dumps(value), metadata={
                        "returned_model": runner.MODEL, "finish_reason": "stop", "http_status": 200})
                if leaf_repair:
                    ctx = fixtures.input_context(prompt)
                    repair = json.JSONDecoder().raw_decode(prompt.split("\nREVIEW_FORMAT_REPAIR:\n", 1)[1])[0]
                    owner.role_counts["scene_reviewer"] += 1
                    owner.observed.append({"role": "scene_reviewer", "context": ctx, "payload": actual})
                    assert not repair["unrepairable_failures"], "unsupported_mock_repair_requested"
                    value = {**repair["identity"], "repair_patches": [
                        {"path": target["path"], "value": deepcopy(owner.omitted_scalar)}
                        for target in repair["targets"]]}
                    return SimpleNamespace(raw_text=json.dumps(value), metadata={
                        "returned_model": runner.MODEL, "finish_reason": "stop", "http_status": 200})
                ctx = json.JSONDecoder().raw_decode(prompt.split("\nPURPOSE_INPUT_JSON\n", 1)[1])[0]
                owner.role_counts["purpose_proposer"] += 1
                owner.observed.append({"role": "purpose_proposer", "context": ctx, "payload": actual})
                value = {"root_id": ctx["root_id"], "route": ctx["route"], "scene_id": ctx["scene_id"],
                         "dataset_card_id": ctx["dataset_card"]["card_id"],
                         "trigger": "MOCK ONLY: a new hotel event.", "goal": "MOCK ONLY: arrange a new hotel stay."}
                return SimpleNamespace(raw_text=json.dumps(value), metadata={
                    "returned_model": runner.MODEL, "finish_reason": "stop", "http_status": 200})
        return Client()

    def planner(self, ctx):
        value = super().planner(ctx)
        scene = value["scenes"][0]
        is_b = ctx["route"] == runner.ROUTES[1]
        scene["event_definition"] = {"trigger": "MOCK ONLY: a new request.",
            "history_relation": "distinct_event" if is_b else "continuation"}
        if is_b:
            scene["source_anchors"] = []
            scene["state_transition"]["basis_fact_ids"] = []
            scene["state_transition"]["basis_reason"] = "MOCK ONLY: card determines the goal; history remains context."
        if is_b and self.mode == "B_card_drift_replan":
            scene["goal"] = ("MOCK ONLY: prepare for the old interview." if not ctx.get("targeted_repair")
                             else "MOCK ONLY: arrange the assigned hotel stay.")
        return value

    def role_verdict(self, ctx, *, realized=False, negative=False):
        history = ctx.get("cumulative_history", ctx.get("branch_history"))
        anchor = history[-1]
        is_b = ctx["route"] == runner.ROUTES[1]
        value = {"role": ctx["route"], "satisfied": not negative,
                 "primary_event": "MOCK ONLY: independent appointment" if is_b else "MOCK ONLY: old-event progress",
                 "relation_to_history": "old_event_substep" if negative else "distinct_event" if is_b else "continuation",
                 "reason": "MOCK ONLY: scripted route role judgment, not semantic evidence.",
                 "history_evidence": [{"turn_ref": anchor["turn_ref"]}],
                 "primary_goal": "MOCK ONLY: independent new appointment" if is_b else "MOCK ONLY: advance prior event",
                 "serves_existing_event": negative or not is_b,
                 "existing_event": "MOCK ONLY: the prior appointment" if negative or not is_b else "",
                 "new_progress": True,
                 "correction": "MOCK ONLY: create another activity with its own goal." if negative else ""}
        if realized:
            value["new_evidence"] = [{"turn_ref": "NEW:T001"}]
        if self.mode == "roles_without_satisfied":
            value.pop("satisfied")
        return value

    def state_extractor(self, ctx):
        if self.mode == "B_local_patches" and ctx["route"] == runner.ROUTES[1] and ctx["scene_id"] == "S01":
            if ctx.get("targeted_repair", {}).get("repair_mode") == "reference_patch":
                return {"reference_repairs": [{"path": row["path"], "turn_ref": row["candidate_turn_refs"][0]}
                    for row in ctx["targeted_repair"]["citation_reference_candidates"]]}
            value = super().state_extractor(ctx)
            value["facts"][0]["evidence"][0]["turn_ref"] = "MOCK_INVALID_ID"
            return value
        if self.mode == "A_rejected" and ctx["route"] == runner.ROUTES[0]:
            return {"MOCK_invalid_state": True}
        if (self.mode == "A_middle_rejected" and ctx["route"] == runner.ROUTES[0]
                and ctx["scene_id"] == "S02"):
            return {"MOCK_invalid_state": True}
        return super().state_extractor(ctx)

    def event_state_extractor(self, ctx):
        if (self.mode == "B_local_patches" and ctx.get("targeted_repair", {}).get("repair_mode")
                == "timeline_entry_patch"):
            return {"timeline_repairs": [{"constraint_id": row["constraint_id"], "choice": "planned_development"}
                    for row in ctx["targeted_repair"]["faulty_timeline_entries"]], "plan_check_repairs": []}
        value = super().event_state_extractor(ctx)
        negative = self.mode == "B_role_replan" and ctx["route"] == runner.ROUTES[1] and "B_role_replan" not in self.used_injections
        if negative:
            self.used_injections.add("B_role_replan")
        value["route_fulfillment"] = self.role_verdict(ctx, negative=negative)
        for check in value["plan_checks"]:
            check["comparison_kind"] = "new_development"
        if self.mode == "B_local_patches" and ctx["route"] == runner.ROUTES[1] and ctx["scene_id"] == "S01":
            value["timeline"].append({"constraint_id": f"c{len(value['timeline']) + 1}",
                "status": "uncertain", "evidence_basis": "inferred", "statement": "MOCK ONLY: proposed new activity.",
                "evidence": [], "implication": "MOCK ONLY: this belongs to the new plan."})
        if (ctx["route"] == runner.ROUTES[1] and self.mode == "B_card_drift_replan"
                and "old interview" in ctx["scene_plan"]["goal"]):
            value["plan_checks"][0].update(verdict="revise", constraint_ids=[],
                reason="MOCK ONLY: the plan abandoned the assigned hotel purpose.",
                conflict={"type": "assigned_event_mismatch", "historical_claim": "",
                          "proposed_claim": "MOCK ONLY: old interview preparation replaces the hotel event.",
                          "correction": "MOCK ONLY: instantiate the assigned hotel event."})
        if ctx["route"] == runner.ROUTES[1] and self.mode == "B_event_preservation_repair":
            if ctx.get("targeted_repair") is not None:
                value["plan_checks"][0] = deepcopy(self.preserved_event_negative)
            elif self.mode not in self.used_injections:
                self.used_injections.add(self.mode)
                first = value["plan_checks"][0]
                first.update(verdict="revise", comparison_kind="historical_assertion",
                    reason="MOCK ONLY: an old premise conflicts with the proposed claim.",
                    conflict={"type": "historical_contradiction", "historical_claim": "MOCK ONLY: old appointment remains unconfirmed.",
                              "proposed_claim": "MOCK ONLY: appointment was already confirmed.",
                              "correction": "MOCK ONLY: keep the old appointment unconfirmed and propose new progress."})
                self.preserved_event_negative = deepcopy(first)
                # v35 supports completing an explicit negative's missing conflict;
                # a missing semantic judgment row is deliberately unrepairable.
                first.pop("conflict")
        if ctx["route"] == runner.ROUTES[1]:
            if "parent_goal_checks_template" in ctx:
                value["parent_goal_checks"] = [
                    {"constraint_id": row["constraint_id"], "parent_goal": None,
                     "functional_relation": "context_only", "reason": "MOCK ONLY: fixed context comparison."}
                    for row in value["timeline"]]
            value["card_plan_checks"] = [
                {"kind": row["kind"], "id": row["id"], "verdict": "compatible",
                 "reason": "MOCK ONLY: explicit selected-card comparison; no semantic certification.",
                 "correction": ""}
                for row in ctx["expected_card_plan_checks"]]
            value["route_fulfillment"]["history_goal_checks"] = [
                {"constraint_id": row["constraint_id"], "historical_goal": row["statement"],
                 "relation": "serves_existing_goal" if negative else "context_only",
                 "reason": "MOCK ONLY: explicit row comparison, no semantic certification."}
                for row in value["timeline"]]
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
        if ctx["route"] == runner.ROUTES[1]:
            value["route_fulfillment"]["history_goal_checks"] = [
                {"constraint_id": row["constraint_id"], "historical_goal": row["statement"],
                 "relation": "serves_existing_goal" if negative else "context_only",
                 "reason": "MOCK ONLY: explicit independent reviewer comparison."}
                for row in ctx["event_state"]["timeline"]]
            by_ref = {row["turn_ref"]: row for row in ctx["new_scene"]["turns"]}
            def actor_check(action, evidence, expected_actor=None):
                ref = evidence[0]
                ref = ref if isinstance(ref, str) else ref["turn_ref"]
                actor = by_ref[ref]["speaker"]
                return {"expected_actor": expected_actor or actor, "expected_action": action,
                        "actor": actor, "action": "MOCK ONLY: fixture action.",
                        "evidence_mode": "direct_speech", "evidence": deepcopy(evidence),
                        "satisfied": True, "reason": "MOCK ONLY: scripted attributed action."}
            for row in value["checks"]:
                row["actor_checks"] = [actor_check(
                    ctx["scene_plan"]["developments"][row["development_index"] - 1], row["evidence"])]
            card = ctx["dataset_card_guidance"]
            steps = {step["step_id"]: step for step in card["steps"]}
            bindings = {}
            for row in value.get("card_checks", []):
                if row["kind"] != "step":
                    continue
                step = steps[row["id"]]
                check = actor_check(step["action"], row["evidence"])
                bindings.setdefault(step["actor_id"], check["actor"])
                check["expected_actor"] = bindings[step["actor_id"]]
                row["actor_checks"] = [check]
            value["card_actor_bindings"] = [
                {"card_actor_id": key, "actor": actor} for key, actor in bindings.items()]
        if ctx["route"] == runner.ROUTES[1] and ctx["scene_id"] == "S01" and self.mode == "B_partial_review_repair":
            if self.mode not in self.used_injections:
                self.used_injections.add(self.mode)
                # v33 fixes scalar omissions only. Missing whole checks remain
                # deliberately incomplete, covered by B_review_incomplete.
                self.omitted_scalar = value["checks"][-1]["actor_checks"][0]["action"]
                value["checks"][-1]["actor_checks"][0]["action"] = ""
        if ctx["route"] == runner.ROUTES[1] and self.mode == "B_review_incomplete":
            value["checks"].pop()
        if ctx["route"] == runner.ROUTES[1] and ctx["scene_id"] == "S01" and self.mode == "B_review_role_consistency_repair":
            if self.mode not in self.used_injections:
                self.used_injections.add(self.mode)
                value["route_fulfillment"].update(satisfied=False, correction="MOCK ONLY: clarify the inconsistent role verdict.")
        return value


def initialize_mock(caller, run, fake):
    caller.run, caller.limit = run, 200
    caller.profile, caller.factory = runner.base.transport()[0](), fake.factory
    caller.capacity = lambda *_args: {"status": "PASS", "mock_capacity_only": True}
    caller.annotate, caller.tokenizer = lambda value: value, object()
    caller.http_counter = caller.request_counter = 0
    caller.last_http_completed = None


class SequentialFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v23-sequential-mock-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        for name in ("sources", "feedback"):
            shutil.copytree(FROZEN / name, self.run / name)
        runner.base.write(self.run / "PURPOSE_FEEDBACK_PROJECTION.json",
            runner.purpose_contract.project_feedback(self.run / "feedback", ROOT, runner.base.feedback))
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
        expected_b = 0 if mode in ("B_review_incomplete", "B_role_repair") else 1
        self.assertEqual([item["accepted_scenes"] for item in results], [expected_a, expected_b], results)
        self.assertEqual(accepted_a[:len(frozen)], frozen)
        for observation in fake.observed:
            ctx = observation["context"]
            if observation["role"] == "purpose_proposer":
                self.assertNotIn("cumulative_history", ctx)
                continue
            history = ctx.get("cumulative_history", ctx.get("branch_history"))
            self.assertEqual(history, frozen if ctx["route"] == runner.ROUTES[0] else accepted_a,
                             observation["role"])
        actual = runner.base.read(self.run / "sources_merged" / f"{ROOT}.json")["turns"]
        self.assertEqual(actual[:len(accepted_a)], accepted_a)
        self.assertEqual(len(actual), len(frozen) + 12 * (expected_a + expected_b))
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
        self.assertEqual(roles, {"purpose_proposer", "state_extractor", "planner", "event_state_extractor", "realizer", "scene_reviewer"})

    def test_three_scene_routes_accumulate_only_accepted_predecessors(self):
        fake = SequentialProvider("A_middle_rejected")
        caller = runner.Caller.__new__(runner.Caller)
        initialize_mock(caller, self.run, fake)
        result = runner.run_root(caller, self.run, ROOT, fixtures.fixture_cards(3), target_scenes=3)
        self.assertEqual([row["target_scenes"] for row in result], [3, 3])
        self.assertEqual([row["accepted_scenes"] for row in result], [2, 3], result)
        frozen = runner.base.read(self.run / "sources" / f"{ROOT}.json")["turns"]
        accepted_a = runner.base.read(self.run / "roots" / ROOT / "A/cumulative_dialogue.json")["turns"]
        self.assertFalse(any(turn["turn_ref"].startswith("R02:A:S02:") for turn in accepted_a))
        expected = {}
        history = list(frozen)
        for letter, route in zip(("A", "B"), runner.ROUTES):
            for number in range(1, 4):
                scene_id = f"S{number:02d}"
                expected[(route, scene_id)] = list(history)
                path = self.run / "roots" / ROOT / letter / f"{scene_id}.json"
                if path.exists():
                    history = runner.base.append_scene(history, ROOT, route, runner.base.read(path))
        for observation in fake.observed:
            ctx = observation["context"]
            scene_id = ctx.get("target_scene_id", ctx.get("scene_id"))
            if observation["role"] == "purpose_proposer":
                self.assertNotIn("cumulative_history", ctx)
                continue
            actual = ctx.get("cumulative_history", ctx.get("branch_history"))
            self.assertEqual(actual, expected[(ctx["route"], scene_id)], observation["role"])
            if ctx["route"] == runner.ROUTES[1]:
                self.assertEqual(actual[:len(accepted_a)], accepted_a)
        final = runner.base.read(self.run / "sources_merged" / f"{ROOT}.json")["turns"]
        self.assertEqual(final, history)
        self.assertEqual(len(final), len(frozen) + 5 * 12)
        self.assertEqual(len({turn["turn_ref"] for turn in final}), len(final))

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
            ["purpose_proposer", "state_extractor", "planner", "event_state_extractor", "planner", "event_state_extractor", "realizer", "scene_reviewer"])
        self.assertTrue(calls[4]["context"]["targeted_repair"]["plan_checks"][0]["route_role_check"])

    def test_card_purpose_drift_returns_to_planner_before_any_realizer(self):
        fake, _ = self.exercise("B_card_drift_replan")
        calls = [row for row in fake.observed if row["context"]["route"] == runner.ROUTES[1]]
        self.assertEqual([row["role"] for row in calls], [
            "purpose_proposer", "state_extractor", "planner", "event_state_extractor",
            "planner", "event_state_extractor", "realizer", "scene_reviewer"])
        self.assertIn("old interview", calls[3]["context"]["scene_plan"]["goal"])
        self.assertEqual(calls[4]["context"]["targeted_repair"]["plan_checks"][0]["conflict"]["type"],
                         "assigned_event_mismatch")
        self.assertIn("assigned hotel", calls[6]["context"]["scene_plan"]["goal"])
        self.assertTrue(all(row["context"]["purpose_input"]["goal"] == "MOCK ONLY: arrange a new hotel stay."
                            for row in calls[1:]))

    def test_unknown_transport_failure_is_local_to_slot_and_remaining_slots_continue(self):
        fake = SequentialProvider()
        caller = runner.Caller.__new__(runner.Caller)
        initialize_mock(caller, self.run, fake)
        original = caller.call
        def fail_first(label, payload, proof):
            if proof["role"] == "purpose_proposer" and proof["scene_id"] == "S01":
                raise RuntimeError("provider_request_failed_after_retry_policy")
            return original(label, payload, proof)
        with patch.object(caller, "call", side_effect=fail_first):
            result = runner.run_root(caller, self.run, ROOT, fixtures.fixture_cards(3), target_scenes=3,
                                     routes=(runner.ROUTES[1],))
        self.assertEqual([item["scene_id"] for item in result[0]["scene_metadata"]], ["S02", "S03"])
        self.assertEqual(result[0]["failures"][0]["failure_category"], "transport_failure")
        self.assertTrue(result[0]["failures"][0]["remaining_slots_continue"])

    def test_programming_failure_is_not_swallowed_as_transport_shortfall(self):
        fake = SequentialProvider()
        caller = runner.Caller.__new__(runner.Caller)
        initialize_mock(caller, self.run, fake)
        with patch.object(caller, "call", side_effect=RuntimeError("bad_runtime_configuration")):
            with self.assertRaisesRegex(RuntimeError, "bad_runtime_configuration"):
                runner.run_root(caller, self.run, ROOT, fixtures.fixture_cards(3), target_scenes=3,
                                routes=(runner.ROUTES[1],))

    def test_purpose_projection_excludes_history_examples_and_preserves_guidance_verbatim(self):
        projection = runner.base.read(self.run / "PURPOSE_FEEDBACK_PROJECTION.json")
        source = Path(projection["source_path"]).read_text()
        original = json.loads(source.splitlines()[2])
        self.assertEqual(projection["projected_feedback"]["guidance"], original["guidance"])
        self.assertEqual(set(projection["projected_feedback"]), set(runner.purpose_contract.FEEDBACK_FIELDS))
        self.assertNotIn("historical_examples", runner.purpose_contract.feedback_suffix(projection))
        fake, _ = self.exercise()
        purpose = next(row for row in fake.observed if row["role"] == "purpose_proposer")
        self.assertEqual(set(purpose["context"]), {"root_id", "route", "scene_id", "allowed_speakers", "dataset_card"})
        self.assertEqual(purpose["payload"]["max_tokens"], 4096)
        text = purpose["payload"]["messages"][0]["content"]
        self.assertNotIn("source_turn_excerpts", text)
        self.assertNotIn("historical_stem_unverified", text)

    def test_B_wrong_role_in_realization_stops_wrong_plan_repetition(self):
        fake, _ = self.exercise("B_role_repair")
        calls = [row for row in fake.observed if row["role"] == "realizer" and row["context"]["route"] == runner.ROUTES[1]]
        self.assertEqual(len(calls), 1)

    def test_B_verbatim_A_copy_rejected_before_reviewer_then_repaired(self):
        fake, _ = self.exercise("B_copy_A")
        calls = [row["role"] for row in fake.observed if row["context"]["route"] == runner.ROUTES[1]]
        self.assertEqual(calls[-3:], ["realizer", "realizer", "scene_reviewer"])
        failures = [runner.base.read(path) for path in self.run.glob("requests/*/ACCEPTANCE.json")]
        self.assertTrue(any(item.get("stage") == "duplicate" and item["status"] == "REJECT" for item in failures))

    def test_partial_review_retry_fills_only_missing_scalar_without_rewriting_scene(self):
        fake, _ = self.exercise("B_partial_review_repair")
        calls = [row["role"] for row in fake.observed if row["context"]["route"] == runner.ROUTES[1]]
        self.assertEqual(calls.count("realizer"), 1)
        self.assertEqual(calls.count("scene_reviewer"), 2)
        merged = [runner.base.read(path) for path in self.run.glob("requests/*/FORMAT_REPAIR_MERGE.json")]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["status"], "PASS")
        self.assertEqual(len(merged[0]["candidate"]["checks"]), 2)
        self.assertEqual(merged[0]["candidate"]["checks"][-1]["actor_checks"][0]["action"], fake.omitted_scalar)
        specs = [runner.base.read(path) for path in self.run.glob("requests/*/FORMAT_REPAIR_SPEC.json")]
        self.assertEqual(len(specs), 1)
        self.assertEqual([row["path"] for row in specs[0]["targets"]], [["checks", 1, "actor_checks", 0, "action"]])

    def test_exhausted_review_format_is_audit_incomplete_not_content_failure(self):
        fake, results = self.exercise("B_review_incomplete")
        failure = results[1]["failures"][0]
        self.assertEqual(failure["reason"], "audit_incomplete")
        self.assertFalse(failure["scene_content_failure_established"])
        self.assertEqual(sum(row["role"] == "realizer" and row["context"]["route"] == runner.ROUTES[1]
                             for row in fake.observed), 1)

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
    def test_local_reference_and_timeline_patches_survive_full_live_audit(self):
        self.exercise_live("B_local_patches")

    def exercise_live(self, mode=None, *, tamper=False):
        with tempfile.TemporaryDirectory(prefix="v23-complete-mock-") as temporary, ExitStack() as stack:
            temp = Path(temporary)
            delivery = temp / "offline-deliveries/h4_sequential_ab_v35"
            delivery.mkdir(parents=True)
            for path in runner.HERE.glob("*.py"):
                shutil.copy2(path, delivery / path.name)
            (delivery / "POLICY.md").write_text("MOCK ONLY: artifact binding fixture.\n")
            unique = delivery.parent / "h4_r02_source_unique_v2"
            unique.mkdir()
            shutil.copy2(runner.HERE.parent / "h4_r02_source_unique_v2/unique_events.py", unique / "unique_events.py")
            run = delivery / "runs/MOCK_ONLY"
            fake = SequentialProvider(mode)
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
            runner.prepare(run, scenes=3, roots=[ROOT], routes=(runner.ROUTES[1],))
            output = runner.live(run)
            expected_scenes = 2 if mode == "B_review_role_consistency_repair" else 3
            self.assertEqual(output["accepted_scenes"], expected_scenes, output)
            self.assertEqual(output["inherited_A_scenes"], 2)
            self.assertEqual(output["new_A_calls"], 0)
            audit = audit_sequential.audit_run(run)
            self.assertEqual(audit["status"], "PASS", audit["failures"])
            self.assertEqual(audit["merged_roots_verified"], 1)
            self.assertEqual(audit["source_overlap_count"], 0)
            if mode == "B_local_patches":
                self.assertEqual(audit["local_patch_merge_replays"], 2)
                merges = [runner.base.read(path) for path in run.glob("requests/*/FORMAT_REPAIR_MERGE.json")]
                self.assertEqual(len(merges), 2)
                self.assertTrue(all(item["status"] == "PASS" for item in merges))
            if mode == "roles_without_satisfied":
                decisions = [runner.base.read(path) for path in run.glob("requests/*/ROUTE_DECISION_PROVENANCE.json")]
                self.assertEqual(len(decisions), 3)
                for decision in decisions:
                    self.assertEqual(decision["decision_source"], "program_policy_derived_from_explicit_model_fields")
                    self.assertTrue(decision["output"]["satisfied"])
                for path in run.glob("requests/*/RESPONSE.json"):
                    raw = json.loads(runner.base.read(path)["text"])
                    if "route_fulfillment" in raw:
                        self.assertNotIn("satisfied", raw["route_fulfillment"])
            b_calls = [row["role"] for row in fake.observed if row["context"].get("route") == runner.ROUTES[1]]
            if mode == "B_partial_review_repair":
                self.assertEqual(b_calls.count("realizer"), 3)
                self.assertEqual(b_calls.count("scene_reviewer"), 4)
                merges = [runner.base.read(path) for path in run.glob("requests/*/FORMAT_REPAIR_MERGE.json")]
                self.assertEqual(len(merges), 1)
                self.assertEqual(merges[0]["status"], "PASS")
            if mode == "B_review_role_consistency_repair":
                # A syntactically valid false cannot be changed by a format
                # patch, even when it contradicts other route fields.
                self.assertEqual(b_calls.count("realizer"), 3)
                self.assertEqual(b_calls.count("scene_reviewer"), 3)
                self.assertFalse(list(run.glob("requests/*/FORMAT_REPAIR_MERGE.json")))
                specs = [runner.base.read(path) for path in run.glob("requests/*/FORMAT_REPAIR_SPEC.json")]
                self.assertEqual(len(specs), 1)
                self.assertTrue(specs[0]["unrepairable_failures"])
                self.assertFalse(specs[0]["previous_response"]["route_fulfillment"]["satisfied"])
                raw_reviews = [runner.base.read(path)["text"] for path in run.glob("requests/*S01-review-*/RESPONSE.json")]
                self.assertEqual(len(raw_reviews), 1)
                original_review = json.loads(raw_reviews[0])
                self.assertFalse(original_review["route_fulfillment"]["satisfied"])
                self.assertEqual(specs[0]["previous_response"], original_review)
                failure = output["routes"][0]["failures"][0]
                self.assertEqual(failure["scene_id"], "S01")
                self.assertEqual(failure["reason"], "audit_incomplete")
                self.assertFalse(failure["scene_content_failure_established"])
            if mode == "B_event_preservation_repair":
                self.assertEqual(b_calls.count("event_state_extractor"), 5)
                self.assertEqual(b_calls.count("planner"), 4)
                preserved = [runner.base.read(path) for path in run.glob("requests/*/FORMAT_REPAIR_PRESERVATION.json")]
                self.assertEqual(len(preserved), 1)
                self.assertTrue(preserved[0]["unchanged_negative_findings_required"])
                event_calls = [row for row in fake.observed if row["context"].get("route") == runner.ROUTES[1]
                               and row["role"] == "event_state_extractor"]
                spec = next(runner.base.read(path) for path in run.glob("requests/*B*S01-event-state-*/FORMAT_REPAIR_SPEC.json"))
                repaired = spec["previous_raw"]
                expected = deepcopy(fake.preserved_event_negative); expected.pop("conflict")
                self.assertEqual(repaired["plan_checks"][0], expected)
                self.assertEqual(audit["event_local_patch_request_replays"], 1)
                self.assertEqual(audit["local_patch_merge_replays"], 1)
                decision = next(runner.base.read(path) for path in run.glob("requests/*B*S01-planner-1/EVENT_STATE_DECISION.json"))
                self.assertEqual(decision["status"], "REPLAN")
            if not tamper:
                return
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

    def test_real_prepare_live_reservation_and_independent_audit(self):
        self.exercise_live(tamper=True)

    def test_full_audit_replays_partial_review_merge(self):
        self.exercise_live("B_partial_review_repair")

    def test_full_audit_replays_preserved_negative_event_repair(self):
        self.exercise_live("B_event_preservation_repair")

    def test_full_audit_preserves_inconsistent_negative_role_without_rejudgment(self):
        self.exercise_live("B_review_role_consistency_repair")

    def test_full_audit_derives_route_policy_without_repeated_model_boolean(self):
        self.exercise_live("roles_without_satisfied")


if __name__ == "__main__":
    unittest.main()
