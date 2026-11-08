"""Affected offline regressions; all sockets blocked for the entire suite."""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .budget import BudgetLedger, validate_budget
from .fixtures import failed_bundle, fixture_patch, example_budget, FixtureBackend, FAILED
from .planning import prepare_original, merge_patch, sha
from .prompts import render_realizer, render_supplement, TARGET_RULES
from .retention import assess_retention, dependencies_satisfied, local_fact_input
from .workflow import execute_policy
from .run import main, validate_frozen_sources, offline_network_guard
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import normalize_realizer_output


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = failed_bundle()
        cls.network = offline_network_guard()
        cls.network.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.network.__exit__(None, None, None)

    def prepared(self):
        b = self.bundle
        return prepare_original(b["original_plan"], parent=b["parent"], candidates=b["candidates"], verified_tracks=b["verified_tracks"])

    def run_fixture(self, budget=None, backend=None):
        with tempfile.TemporaryDirectory(prefix="h4-incremental-test-") as temp:
            return execute_policy(bundle=deepcopy(self.bundle), budget=budget or example_budget(),
                backend=backend or FixtureBackend(self.bundle), state_registry=Path(temp)/"registry", artifact_dir=Path(temp)/"run")

    def test_01_old_failure_unchanged(self):
        p = self.prepared()
        original = json.loads((FAILED / "planner/validation.json").read_text())
        self.assertFalse(p["legacy_validation"]["valid"])
        self.assertEqual(original["errors"], p["legacy_validation"]["errors"])
        self.assertIn("continuation_2_future_only_after_state", p["goal_diagnostics"])

    def test_02_hard_identity_not_waived(self):
        b = deepcopy(self.bundle)
        b["original_plan"]["selected_continuations"][0]["verified_track_ref"] = "VT999"
        p = prepare_original(b["original_plan"], parent=b["parent"], candidates=b["candidates"], verified_tracks=b["verified_tracks"])
        self.assertFalse(p["valid_for_new_policy"])
        self.assertTrue(any("unknown_verified_track" in e for e in p["hard_errors"]))

    def test_03_eight_scenes_four_batches_no_second_plan(self):
        report = self.run_fixture()
        self.assertEqual(report["status"], "EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY", report)
        self.assertEqual(report["retained_scene_count"], 8, report)
        self.assertEqual(sum(c["role"] == "realizer" for c in report["new_calls"]), 4)
        self.assertEqual(sum(c["role"] == "supplement" for c in report["new_calls"]), 1)
        self.assertEqual(report["external_call_count"], 0)
        self.assertFalse(report["all_goals_complete"])

    def test_04_supplement_consumed_on_timeout_and_restart(self):
        with tempfile.TemporaryDirectory(prefix="h4-budget-test-") as temp:
            ledger = BudgetLedger(temp, root_id="mock", round_index=3, budget=example_budget())
            self.assertTrue(ledger.reserve("s", {"supplement": 1}, cumulative_tokens=10))
            ledger.begin("s", "supplement", input_tokens=100)
            restarted = BudgetLedger(temp, root_id="mock", round_index=3, budget=example_budget())
            with self.assertRaises(ValueError):
                restarted.begin("s", "supplement", input_tokens=100)
        report = self.run_fixture(backend=FixtureBackend(self.bundle, fail_role="supplement"))
        self.assertEqual(report["status"], "SUPPLEMENT_FAILED_STOP_NO_RETRY")
        self.assertEqual(len(report["new_calls"]), 1)

    def test_05_no_supplement_stops(self):
        report = self.run_fixture(backend=FixtureBackend(self.bundle, no_supplement=True))
        self.assertEqual(report["status"], "NO_SUPPLEMENT_STOP")
        self.assertEqual(len(report["new_calls"]), 1)

    def test_06_goals_immutable_and_multiple_same_event_phases(self):
        p = self.prepared(); old = sha(p["original"])
        merged = merge_patch(p, fixture_patch(self.bundle))
        self.assertTrue(merged["valid"], merged)
        self.assertEqual(sha(merged["merged"]["original_plan"]), old)
        self.assertEqual(len(merged["merged"]["added_phases"]), 2)
        self.assertEqual(len({r["scene"]["memory_event_id"] for r in merged["merged"]["added_phases"]}), 1)

    def test_07_reject_goal_rewrite_and_unrelated_source(self):
        p = fixture_patch(self.bundle); p["phases"][0]["original_goal"] = "lowered goal"
        self.assertFalse(merge_patch(self.prepared(), p)["valid"])
        p = fixture_patch(self.bundle); p["phases"][0]["source_candidate_id"] = "ND01:C01"
        self.assertFalse(merge_patch(self.prepared(), p)["valid"])

    def test_08_binding_history_or_no_progress_reject(self):
        for field in ("participant_adherence", "history_consistency", "novel_progress"):
            report = self.run_fixture(backend=FixtureBackend(self.bundle, bad_audit={field: "fail"}))
            self.assertEqual(report["retained_scene_count"], 0, field)

    def test_09_unknown_not_retained(self):
        report = self.run_fixture(backend=FixtureBackend(self.bundle, bad_audit={"goal_status": "UNKNOWN"}))
        self.assertEqual(report["retained_scene_count"], 0)

    def test_10_dependencies_do_not_assume_goal_complete(self):
        report = self.run_fixture(backend=FixtureBackend(self.bundle, establish=False))
        self.assertEqual(report["retained_scene_count"], 6)
        self.assertEqual(sum(s.get("status") == "DEFERRED_DEPENDENCY" for s in report["scene_decisions"]), 2)

    def test_11_budget_reserves_audit_and_defers(self):
        b = example_budget(); b["role_requests"]["adherence"] = 1
        report = self.run_fixture(budget=b)
        self.assertEqual(report["retained_scene_count"], 2)
        self.assertIn("DEFERRED_BUDGET", report["status"])
        self.assertEqual(sum(c["role"] == "realizer" for c in report["new_calls"]), 1)

    def test_12_missing_budget_blocked(self):
        with self.assertRaises(ValueError):
            validate_budget({"total_requests": 16})

    def test_13_facts_only_actual_dialogue(self):
        scenes = [{"after_state": "completed", "dialogue_turns": [{"turn_ref": "R03:S01:T001", "speaker": "A", "text": "I intend to invite them later."}]}]
        out = local_fact_input(scenes)
        self.assertEqual(out[0]["text"], "I intend to invite them later.")
        self.assertNotIn("completed", json.dumps(out))

    def test_14_exact_copy_validator_unchanged(self):
        parent = self.bundle["parent"]
        scene = next(iter(self.prepared()["candidate_registry"].values()))["scene"]
        turns = [{"speaker": scene["required_participants"][i % 2], "text": f"mock fresh text {i}"} for i in range(14)]
        turns[0]["text"] = parent["dialogue"]["turns"][0]["text"]
        result = normalize_realizer_output({"scenes": [{"dialogue_turns": turns}]}, plan={"scenes": [scene]}, parent=parent, round_index=3)
        self.assertFalse(result["valid"])
        self.assertTrue(any("parent_exact_copy" in e for e in result["errors"]))

    def test_15_text_free_and_full_input_boundary(self):
        b = self.bundle; scene = next(iter(self.prepared()["candidate_registry"].values()))["scene"]
        text = b["parent"]["dialogue"]["turns"][0]["text"]
        free = render_realizer(parent=b["parent"], plan={"scenes": [scene]}, mode="text_free")
        full = render_realizer(parent=b["parent"], plan={"scenes": [scene]}, mode="full_parent")
        self.assertNotIn(text, free["prompt"])
        self.assertIn(text, full["prompt"])
        self.assertNotIn("must already be true", free["prompt"])
        with self.assertRaises(ValueError):
            render_realizer(parent=b["parent"], plan={"scenes": [scene]}, continuity_shape={"text": "leak"})

    def test_16_no_real_backend_offline(self):
        with tempfile.TemporaryDirectory(prefix="h4-offline-test-") as temp:
            with self.assertRaisesRegex(ValueError, "offline_entry_requires_mock"):
                execute_policy(bundle=self.bundle, budget=example_budget(), backend=object(), state_registry=temp, artifact_dir=Path(temp)/"run")

    def test_17_outer_attempt_not_restarted(self):
        with tempfile.TemporaryDirectory(prefix="h4-claim-test-") as temp:
            l = BudgetLedger(temp, root_id="mock", round_index=3, budget=example_budget())
            l.claim_execution({"branch_id": "a"})
            with self.assertRaises(ValueError):
                BudgetLedger(temp, root_id="mock", round_index=3, budget=example_budget()).claim_execution({"branch_id": "b"})

    def test_18_invalid_patch_stops_without_other_requests(self):
        class BadPatch(FixtureBackend):
            def request(self, request):
                result = super().request(request)
                if request["role"] == "supplement":
                    result["phases"][0]["source_candidate_id"] = ["bad"]
                return result
        report = self.run_fixture(backend=BadPatch(self.bundle))
        self.assertEqual(report["status"], "SUPPLEMENT_INVALID_STOP")
        self.assertEqual(len(report["new_calls"]), 1)

    def test_19_bare_fail_cannot_be_reinterpreted(self):
        class BareFail(FixtureBackend):
            def request(self, request):
                result = super().request(request)
                return {"verdict": "FAIL"} if request["role"] == "adherence" else result
        self.assertEqual(self.run_fixture(backend=BareFail(self.bundle))["retained_scene_count"], 0)

    def test_20_copy_cannot_pass_with_partial_goal(self):
        parent_text = self.bundle["parent"]["dialogue"]["turns"][0]["text"]
        class Copier(FixtureBackend):
            def request(self, request):
                result = super().request(request)
                if request["role"] == "realizer":
                    for scene in result["scenes"]:
                        scene["dialogue_turns"][0]["text"] = parent_text
                return result
        report = self.run_fixture(backend=Copier(self.bundle))
        self.assertEqual(report["retained_scene_count"], 0)
        self.assertTrue(any(s.get("status") == "STRUCTURE_COPY_OR_BINDING_REJECT" for s in report["scene_decisions"]))

    def test_21_not_realized_with_new_progress_can_retain(self):
        report = self.run_fixture(backend=FixtureBackend(self.bundle, bad_audit={"goal_status": "not_realized"}))
        self.assertEqual(report["retained_scene_count"], 8)
        self.assertTrue(all(v == "not_realized" for v in report["goal_statuses"].values()))

    def test_22_missing_real_budget_cli_blocks_before_backend(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                main(["--execute", "--run-dir", "/tmp/not-created-h4-test"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(Path("/tmp/not-created-h4-test").exists())

    def test_23_source_identity_matches_actual_frozen_files(self):
        validate_frozen_sources(self.bundle)
        b = deepcopy(self.bundle); b["parent"]["dialogue"]["turns"][0]["text"] = "tampered"
        with self.assertRaises(ValueError):
            validate_frozen_sources(b)

    def test_24_three_scene_batch_is_rejected(self):
        scene = next(iter(self.prepared()["candidate_registry"].values()))["scene"]
        with self.assertRaises(ValueError):
            render_realizer(parent=self.bundle["parent"], plan={"scenes": [scene, scene, scene]})

    def test_25_forward_or_fabricated_dependency_rejected(self):
        p = fixture_patch(self.bundle)
        p["phases"][0]["dependencies"][0]["phase_id"] = "phase:unknown_future"
        self.assertFalse(merge_patch(self.prepared(), p)["valid"])


if __name__ == "__main__":
    unittest.main()
