"""Root-scope CLI regression with private grants, temporary ledgers and HTTP mocks."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from . import experiment_authorization as approval
from .input_views import reconstruct_supplement_inputs
from .root_scope import ROOT_POLICY_ID, ROOT_SCOPE, ROOT_PROMPT_VERSION, prepare_bundle
from .test_experiment_entry import (
    ExperimentCliHarness, EXPERIMENT_ID, MOCK_SECRET, snapshot,
)
from .test_readiness import Response
from .test_root_planning import existing_phase_patch, independent_new_patch
from .test_supplement_alignment import read_json


def mixed_root_patch(bundle):
    patch = existing_phase_patch(bundle, ("ME03:C01",))
    patch["new_developments"] = independent_new_patch(bundle)["new_developments"]
    return patch


class RootCliHarness(ExperimentCliHarness):
    """The production index never changes; this grant exists only in a mock."""
    def __enter__(self):
        super().__enter__()
        self.legacy_record = deepcopy(self.record)
        self.bundle.update(policy_id=ROOT_POLICY_ID, incremental_scope=ROOT_SCOPE,
                           supplement_prompt_version=ROOT_PROMPT_VERSION)
        self.bundle["identity"]["branch_id"] = "MOCK-ROOT-INCREMENT-NOT-AUTHORIZED-FOR-LIVE"
        self.pin_fixture_approval()
        return self

    def invoke(self, *, case="mixed", authorization_mutation=None, **kwargs):
        def root_authorization(value):
            value["policy_id"] = ROOT_POLICY_ID
            if authorization_mutation:
                authorization_mutation(value)
        return super().invoke(case=case, authorization_mutation=root_authorization, **kwargs)

    def http_open(self, request, **kwargs):
        payload = json.loads(request.data)
        self.http_payloads.append(payload)
        actual = self.requests[-1]
        assert request.full_url == self.model["endpoint"]
        assert payload["messages"] == actual["messages"]
        if self.case == "timeout":
            raise TimeoutError("MOCK root response status unknown")
        if actual["role"] == "supplement":
            content = mixed_root_patch(self.bundle)
            if self.case == "no_supplement":
                content.update(status="NO_SUPPLEMENT", phases=[], new_developments=[])
            elif self.case == "invalid_source":
                content["phases"][0]["source_candidate_id"] = "ME999:C01"
            elif self.case == "independent_only":
                content["phases"] = []
            elif self.case == "new_people_cross_batch":
                content = independent_new_patch(self.bundle, count=3, new_people=True)
        else:
            content = self.fixture.request(actual)
            if self.case == "reject_me07" and actual["role"] == "gate":
                for row in content["candidate_reviews"]:
                    if row["candidate_ref"] == "ME07:C01":
                        row["decision"] = "fail"
                        row["reason"] = "OFFLINE MOCK rejected event, not a semantic judgment."
            if self.case in {"history_conflict", "binding_failure"} and actual["role"] == "adherence":
                field = "history_consistency" if self.case == "history_conflict" else "participant_adherence"
                for row in content["scene_audits"]:
                    row[field] = "fail"
        return Response(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(content)}}]}).encode())


def exercise_root_cli(output=None, *, case="mixed"):
    """Export request/contract evidence, not private grants or mock R03 products."""
    with RootCliHarness() as harness:
        report = harness.invoke(case=case)
        run_dir = harness.root / "MOCK_run"
        exported = {
            "result_MOCK_ONLY.json": {
                "case": case, "status": report["status"], "real_external_calls": 0,
                "mock_http_attempt_count": len(harness.http_payloads),
                "production_ledger_unchanged": True, "real_experiment_ledger_unchanged": True,
                "mock_grant_not_registered_in_production_or_exported": True, "real_credentials_read": False,
                "workflow_live_counter_counts_mock_transport_only": True,
                "mock_dialogue_not_exported_or_admitted": True},
            "mock_final_report.json": report,
            "mock_requests.json": harness.requests,
            "mock_http_payloads.json": harness.http_payloads,
            "mock_ledger_snapshot.json": read_json(run_dir / "ledger_snapshot.json"),
            "final_messages_MOCK_CAPTURE.json": harness.http_payloads[0]["messages"],
            "supplement_contract_identity.json": read_json(run_dir / "supplement_contract_identity.json"),
            "source_identity_MOCK_ONLY.json": {
                "root_id": harness.bundle["identity"]["root_id"],
                "round_index": harness.bundle["identity"]["round_index"],
                "history_sha256": harness.bundle["history_sha256"],
                "original_plan_sha256": harness.bundle["original_plan_sha256"],
                "policy_id": ROOT_POLICY_ID, "incremental_scope": ROOT_SCOPE,
                "supplement_prompt_version": ROOT_PROMPT_VERSION},
            "unchanged_actual_state_sha256.json": {
                "production_before": harness.real_before[0],
                "production_after": snapshot(harness.real_production),
                "experiment_before": harness.real_before[1],
                "experiment_after": snapshot(harness.real_experiments)},
        }
        for name in ("increment_patch_and_diff.json", "supplement_input_projection_mapping.json"):
            if (run_dir / name).exists():
                exported[name] = read_json(run_dir / name)
        assert MOCK_SECRET not in json.dumps(exported)
        if output is not None:
            from .readiness import save
            for name, value in exported.items():
                save(output, name, value)
        return exported


class RootEntryTests(unittest.TestCase):
    def test_01_root_final_http_messages_have_exact_sources_history_feedback_and_memory(self):
        with RootCliHarness() as harness:
            report = harness.invoke(case="invalid_source")
            self.assertEqual(report["status"], "SUPPLEMENT_INVALID_STOP")
            self.assertEqual(len(harness.http_payloads), 1)
            text = harness.http_payloads[0]["messages"][0]["content"]
            data = json.loads(text.rsplit("\n\nINPUT DATA\n", 1)[1])
            mapping = read_json(harness.root / "MOCK_run/supplement_input_projection_mapping.json")
            restored = reconstruct_supplement_inputs(data, mapping)
            self.assertEqual(restored["authoritative_history"], harness.bundle["parent"]["dialogue"]["turns"])
            self.assertEqual(len(restored["authoritative_history"]), 143)
            self.assertEqual(restored["original_plan"], harness.bundle["original_plan"])
            self.assertTrue(restored["shared_feedback_view"])
            self.assertEqual(restored["shared_feedback_view"], harness.bundle["shared_feedback_view"])
            self.assertEqual(restored["memory_navigation"], harness.bundle["candidates"])
            self.assertTrue(mapping["reconstruction_exact"])
            self.assertTrue(any(row["original_field"].startswith("/memory_navigation/")
                                for row in mapping["reference_replacements"]))
            prepared = prepare_bundle(harness.bundle)
            identity = harness.requests[0]["supplement_contract_identity"]
            rows = identity["allowed_source_view"]
            self.assertEqual(len(rows), 11)
            self.assertEqual({row["source_candidate_id"] for row in rows},
                             set(prepared["candidate_registry"]))
            for row in rows:
                self.assertEqual(row["original_goal"], prepared["candidate_registry"][row["source_candidate_id"]]["scene"]["goal"])
            self.assertEqual(identity["prompt_version"], ROOT_PROMPT_VERSION)
            self.assertEqual(identity["scope_mode"], ROOT_SCOPE)
            self.assertIn("phase:[A-Za-z0-9_-]+", text)
            self.assertIn("phase:new-event_2", text)
            self.assertIn("not a checklist", text)
            self.assertNotIn("affected events only", text)
            self.assertNotIn("frozen_sources", data["source_scope"])

    def test_02_complete_cli_merges_both_branches_and_retains_partial_progress_within_budget(self):
        with RootCliHarness() as harness:
            report = harness.invoke()
            self.assertEqual(report["status"], "EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY")
            self.assertEqual(report["retained_scene_count"], 8)
            self.assertEqual(report["retained_turn_count"], 112)
            self.assertEqual(report["attempted_request_counts"], {
                "planner": 0, "supplement": 1, "gate": 8, "realizer": 4,
                "adherence": 4, "closure": 0})
            self.assertEqual(len(harness.http_payloads), 17)
            self.assertFalse(report["all_goals_complete"])
            self.assertEqual(report["goal_statuses"]["initial:ME07:C01"], "partial")
            goals = {row["phase_id"]: row for row in report["original_goals_and_phase_statuses"]}
            self.assertTrue(goals["phase:root_1"]["retained"])
            self.assertTrue(goals["phase:independent_1"]["retained"])
            self.assertIn("new_goal", goals["phase:independent_1"])
            self.assertNotIn("original_goal", goals["phase:independent_1"])
            for request in harness.requests:
                self.assertEqual(request["execution_scope"], harness.scope)
            self.assertEqual(report["remote_memory_calls"], 0)
            self.assertEqual(report["other_model_calls"], 0)
            self.assertFalse(report["automatic_training_admission"])

    def test_03_rejected_diagnostic_event_does_not_make_all_other_progress_invalid(self):
        with RootCliHarness() as harness:
            report = harness.invoke(case="reject_me07")
            rejected = [row for row in report["scene_decisions"]
                        if row["phase_id"] == "initial:ME07:C01"]
            self.assertEqual(rejected[0]["status"], "GATE_REJECT_OR_INVALID")
            self.assertGreater(report["retained_scene_count"], 0)
            self.assertTrue(any(row.get("retained") and row["phase_id"] == "initial:ME03:C01"
                                for row in report["scene_decisions"]))
            self.assertLessEqual(len(harness.http_payloads), 17)

    def test_04_history_conflict_and_binding_failure_cannot_retain_content(self):
        for case in ("history_conflict", "binding_failure"):
            with self.subTest(case=case), RootCliHarness() as harness:
                report = harness.invoke(case=case)
                self.assertEqual(report["retained_scene_count"], 0)
                self.assertEqual(report["retained_turn_count"], 0)
                self.assertFalse((harness.root / "MOCK_run/local_cumulative_dialogue.json").exists())
                self.assertTrue(any(row.get("errors") for row in report["scene_decisions"]))
                for request in harness.requests:
                    if request["role"] == "realizer":
                        data = json.loads(request["messages"][0]["content"].rsplit("\n\nINPUT DATA\n", 1)[1])
                        self.assertEqual(data["retained_actual_prefix"], [])

    def test_05_invalid_source_stops_without_downstream_requests(self):
        with RootCliHarness() as harness:
            report = harness.invoke(case="invalid_source")
            self.assertEqual(report["status"], "SUPPLEMENT_INVALID_STOP")
            self.assertEqual([request["role"] for request in harness.requests], ["supplement"])
            result = read_json(harness.root / "MOCK_run/increment_patch_and_diff.json")
            self.assertIn("patch_phase_1_unknown_source", result["errors"])

    def test_06_root_no_supplement_does_not_delete_original_initial_plan(self):
        with RootCliHarness() as harness:
            report = harness.invoke(case="no_supplement")
            self.assertEqual(report["retained_scene_count"], 6)
            self.assertEqual(report["supplement_phase_count"], 0)
            self.assertEqual(report["attempted_request_counts"]["supplement"], 1)
            self.assertEqual(report["attempted_request_counts"]["realizer"], 3)

    def test_07_text_free_and_later_facts_contain_actual_retained_text_only(self):
        with RootCliHarness() as harness:
            harness.invoke()
            requests = [request for request in harness.requests if request["role"] == "realizer"]
            for index, request in enumerate(requests):
                self.assertTrue(request["prompt_version"].endswith("text_free"))
                data = json.loads(request["messages"][0]["content"].rsplit("\n\nINPUT DATA\n", 1)[1])
                self.assertNotIn("authoritative_history", data)
                self.assertLessEqual(len(data["planned_phases_not_facts"]), 2)
                self.assertEqual(len(data["retained_actual_prefix"]), index * 28)
                for turn in data["retained_actual_prefix"]:
                    self.assertEqual(set(turn), {"turn_ref", "speaker", "text"})
                    self.assertTrue(turn["text"].startswith("OFFLINE MOCK "))
                    self.assertNotIn("initial:ND08:C01", turn["text"])
            goals = read_json(harness.root / "MOCK_run/final_report.json")["original_goals_and_phase_statuses"]
            self.assertTrue(any(not row["retained"] and row["phase_id"] == "initial:ND08:C01" for row in goals))

    def test_08_real_old_scope_grant_is_not_repurposed_for_new_policy(self):
        with RootCliHarness() as harness:
            real_path = Path(approval.__file__).parent / "experiment_authorizations" / (EXPERIMENT_ID + ".json")
            original_bytes = real_path.read_bytes()
            record = json.loads(original_bytes)
            self.assertEqual(record["binding"]["allowed_source_ids"], ["ME07:C01"])
            self.assertNotEqual(record["binding"]["policy_id"], ROOT_POLICY_ID)
            harness.approval_path.write_bytes(original_bytes)
            harness.repin_tampered_private_record()
            with self.assertRaisesRegex(ValueError, "experiment_binding_mismatch"):
                harness.invoke()
            harness.assert_unclaimed()
            self.assertEqual(real_path.read_bytes(), original_bytes)

    def test_09_consumed_production_does_not_switch_to_root_experiment(self):
        with RootCliHarness() as harness:
            with self.assertRaisesRegex(ValueError, "already_claimed"):
                harness.invoke(mode=None, experiment_id=None)
            harness.assert_unclaimed()

    def test_10_explicit_scope_and_template_cannot_mismatch_or_default(self):
        for field, value in (("incremental_scope", None), ("supplement_prompt_version", None),
                             ("supplement_prompt_version", "incremental_retention_v1.supplement.readiness_v2")):
            with self.subTest(field=field, value=value), RootCliHarness() as harness:
                if value is None:
                    harness.bundle.pop(field)
                else:
                    harness.bundle[field] = value
                with self.assertRaisesRegex(ValueError, "explicit_root_scope_and_template_required"):
                    harness.invoke()
                harness.assert_unclaimed()

    def test_11_same_grant_cannot_resend_after_failure_or_new_output_directory(self):
        for case in ("invalid_source", "timeout"):
            with self.subTest(case=case), RootCliHarness() as harness:
                harness.invoke(case=case)
                self.assertEqual(len(harness.http_payloads), 1)
                with self.assertRaisesRegex(ValueError, "already_claimed"):
                    harness.invoke(directory="MOCK_restart")
                self.assertEqual(len(harness.http_payloads), 1)

    def test_12_export_is_mock_only_without_grant_secret_or_dialogue_product(self):
        exported = exercise_root_cli(case="invalid_source")
        self.assertEqual(exported["result_MOCK_ONLY.json"]["real_external_calls"], 0)
        self.assertNotIn(MOCK_SECRET, json.dumps(exported))
        self.assertFalse(any("cumulative" in key or "authorization" in key for key in exported))

    def test_13_unestablished_dependency_defers_only_dependent_phase(self):
        with RootCliHarness() as harness:
            harness.fixture.establish = False
            report = harness.invoke()
            decisions = {row["phase_id"]: row for row in report["scene_decisions"]}
            self.assertEqual(decisions["phase:root_1"]["status"], "DEFERRED_DEPENDENCY")
            self.assertTrue(decisions["phase:independent_1"]["retained"])
            self.assertEqual(report["retained_scene_count"], 7)
            self.assertEqual(report["goal_statuses"]["initial:ME03:C01"], "partial")

    def test_14_extra_phases_defer_when_purpose_budget_cannot_reserve_checks(self):
        with RootCliHarness() as harness:
            harness.budget["role_requests"]["realizer"] = 3
            harness.budget["role_requests"]["adherence"] = 3
            harness.pin_fixture_approval()
            report = harness.invoke()
            self.assertEqual(report["status"], "EXECUTION_ENDED_WITH_DEFERRED_BUDGET")
            self.assertEqual(report["retained_scene_count"], 6)
            self.assertEqual(report["attempted_request_counts"]["realizer"], 3)
            self.assertEqual(report["attempted_request_counts"]["adherence"], 3)
            self.assertTrue(any(row["phase_id"] == "phase:independent_1" and
                                row["status"] == "DEFERRED_BUDGET" for row in report["scene_decisions"]))

    def test_15_new_people_in_later_batch_come_only_from_retained_dialogue(self):
        with RootCliHarness() as harness:
            # Explicit private test budget, not a grant or proposed live increase.
            harness.budget["role_requests"].update(gate=9, realizer=5, adherence=5)
            harness.budget["total_requests"] = 21
            harness.budget["max_generated_tokens"] = sum(
                count * harness.budget["max_output_tokens"][role]
                for role, count in harness.budget["role_requests"].items())
            harness.pin_fixture_approval()
            report = harness.invoke(case="new_people_cross_batch")
            self.assertEqual(report["status"], "EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY")
            self.assertEqual(report["retained_scene_count"], 9)
            requests = [row for row in harness.requests if row["role"] == "realizer"]
            self.assertEqual(len(requests), 5)
            data = json.loads(requests[-1]["messages"][0]["content"].rsplit("\n\nINPUT DATA\n", 1)[1])
            self.assertIn("MockVisitor1", data["text_free_continuity_shape"]["speakers"])
            self.assertIn("MockVisitor2", data["text_free_continuity_shape"]["speakers"])
            self.assertNotIn("MockVisitor3", data["text_free_continuity_shape"]["speakers"])
            self.assertNotIn("authoritative_history", data)


if __name__ == "__main__":
    unittest.main()
