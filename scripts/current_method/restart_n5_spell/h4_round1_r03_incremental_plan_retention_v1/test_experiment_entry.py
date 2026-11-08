"""Explicit experiment CLI regression; private approvals, ledgers and HTTP mocks."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

from . import POLICY_ID, experiment_authorization as approval, run
from .fixtures import FixtureBackend
from .input_views import reconstruct_supplement_inputs
from .planning import sha
from .supplement_contract_alignment import ALIGNED_VERSION, LEGACY_VERSION
from .test_readiness import Response
from .test_supplement_alignment import ACTUAL, EXPECTED_ERRORS, read_json


EXPERIMENT_ID = "EXTRA_VAL_122711_CONTRACT_ALIGN_V1_01"
MOCK_SECRET = "mock-experiment-secret-never-export"


def snapshot(directory):
    directory = Path(directory)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()} if directory.exists() else {}


class ExperimentCliHarness:
    """Private pinned grant plus the real CLI, workflow and single-shot HTTP path.

    The approval index is isolated inside this socket-blocked fixture.
    Neither a reusable live authorization nor the fake credential is exported.
    """
    def __enter__(self):
        self.stack = ExitStack()
        self.stack.enter_context(run.offline_network_guard())
        self.real_production = run.production_registry()
        self.real_experiments = approval.experiment_registry()
        self.real_before = (snapshot(self.real_production), snapshot(self.real_experiments))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="h4-experiment-MOCK-")))
        self.bundle = read_json(ACTUAL / "inputs/bundle.json")
        self.bundle["identity"]["branch_id"] = "MOCK-EXTRA-EXPERIMENT-NOT-PRODUCTION"
        self.bundle["supplement_prompt_version"] = ALIGNED_VERSION
        self.budget = read_json(ACTUAL / "inputs/budget.json")
        self.model = read_json(ACTUAL / "inputs/model_config.json")
        identity = {key: self.bundle["identity"][key] for key in ("root_id", "round_index")}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.production = self.root / "MOCK_production"
        self.production.mkdir()
        self.prior_ledger = self.production / (key + ".jsonl")
        self.prior_ledger.write_bytes((self.real_production / self.prior_ledger.name).read_bytes())
        self.prior_bytes = self.prior_ledger.read_bytes()
        self.experiments = self.root / "MOCK_experiments"
        self.approval_generation = 0
        self.approvals = self.root / "MOCK_approvals_0"
        self.stack.enter_context(patch.object(approval, "approval_directory", side_effect=lambda: self.approvals))
        self.stack.enter_context(patch.object(run, "production_registry", return_value=self.production))
        self.stack.enter_context(patch.object(run, "experiment_registry", return_value=self.experiments))
        self.environment = self.stack.enter_context(patch.dict("os.environ", {self.model["api_key_env"]: MOCK_SECRET}))
        self.http_payloads = []
        self.requests = []
        self.case = "no_supplement"
        self.fixture = FixtureBackend(self.bundle)
        original_request = run.SingleShotTransport.request

        def observed_request(transport, request):
            self.requests.append(deepcopy(request))
            return original_request(transport, request)

        self.stack.enter_context(patch.object(run.SingleShotTransport, "request", observed_request))
        self.stack.enter_context(patch("urllib.request.build_opener", return_value=Mock(open=self.http_open)))
        self.stack.enter_context(patch("sys.stdout", new=io.StringIO()))
        self.stack.enter_context(patch("sys.stderr", new=io.StringIO()))
        self.pin_fixture_approval()
        return self

    def __exit__(self, *args):
        try:
            assert self.prior_ledger.read_bytes() == self.prior_bytes
            assert (snapshot(self.real_production), snapshot(self.real_experiments)) == self.real_before
        finally:
            self.stack.close()

    def pin_fixture_approval(self):
        """Explicitly register into a fresh private registry for each test setup.

        Setup changes do not overwrite an earlier grant: each fixture generation
        uses an independent temporary directory, never a production registry.
        """
        self.approval_generation += 1
        self.approvals = self.root / ("MOCK_approvals_" + str(self.approval_generation))
        self.approvals.mkdir()
        self.approval_path = self.approvals / (EXPERIMENT_ID + ".json")
        evidence = ACTUAL / "execution/final_report.json"
        self.record = {"version": approval.VERSION, "experiment_id": EXPERIMENT_ID,
            "explicit_extra_validation_authorized": True,
            "binding": approval.binding_for(self.bundle, self.budget, self.model),
            "prior_production": {"run_path": str(ACTUAL), "ledger_path": str(self.prior_ledger.resolve()),
                "ledger_sha256": hashlib.sha256(self.prior_bytes).hexdigest(),
                "evidence_files": [{"path": str(evidence), "sha256": approval.file_sha(evidence)}]}}
        source = self.root / ("MOCK_registration_input_" + str(self.approval_generation) + ".json")
        source.write_text(json.dumps(self.record, ensure_ascii=False, indent=2))
        approval.register_experiment_record(source, expected_sha256=approval.file_sha(source),
            explicitly_authorized=True, bundle=self.bundle, budget=self.budget,
            model_config=self.model, production_registry=self.production)
        self.scope = approval.load_experiment_scope(EXPERIMENT_ID, bundle=self.bundle,
            budget=self.budget, model_config=self.model, production_registry=self.production)

    def repin_tampered_private_record(self):
        """Simulate corrupted registry data to test downstream binding rejection."""
        approval.authorization_index_path().write_text(json.dumps({
            "experiment_id": EXPERIMENT_ID,
            "record_sha256": approval.file_sha(self.approval_path)}) + "\n")

    def http_open(self, request, **kwargs):
        payload = json.loads(request.data)
        self.http_payloads.append(payload)
        assert request.full_url == self.model["endpoint"]
        actual = self.requests[-1]
        assert payload["messages"] == actual["messages"]
        if self.case == "http_400":
            raise urllib.error.HTTPError(request.full_url, 400, "MOCK rejection",
                {"content-type": "application/json"}, io.BytesIO(b'{"error":"MOCK complete rejection"}'))
        if self.case == "timeout":
            raise TimeoutError("MOCK ambiguous response")
        if actual["role"] == "supplement" and self.case == "invalid_patch":
            content = read_json(ACTUAL / "execution/responses/001_supplement.json")
        elif actual["role"] == "supplement" and self.case == "no_supplement":
            content = json.dumps({"status": "NO_SUPPLEMENT", "original_plan_sha256": self.bundle["original_plan_sha256"],
                "phases": [], "reason": "OFFLINE MOCK; no real model generation."})
        else:
            content = json.dumps(self.fixture.request(actual))
        return Response(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode())

    def invoke(self, *, case="no_supplement", directory="MOCK_run", mode="extra-experiment",
               experiment_id=EXPERIMENT_ID, extra_args=(), authorization_mutation=None):
        self.case = case
        authorization = {"explicitly_authorized": True, "policy_id": POLICY_ID,
            "identity": self.bundle["identity"], "bundle_sha256": sha(self.bundle),
            "budget_sha256": sha(self.budget), "model_config_sha256": sha(self.model),
            "implementation_manifest": run.implementation_manifest()}
        if mode == "extra-experiment":
            authorization["experiment_scope"] = deepcopy(self.scope)
        if authorization_mutation:
            authorization_mutation(authorization)
        for name, value in (("bundle", self.bundle), ("budget", self.budget), ("model", self.model),
                            ("authorization", authorization)):
            (self.root / (name + ".json")).write_text(json.dumps(value))
        args = ["--execute", "--run-dir", str(self.root / directory),
            "--bundle", str(self.root / "bundle.json"), "--budget", str(self.root / "budget.json"),
            "--model-config", str(self.root / "model.json"), "--authorization", str(self.root / "authorization.json")]
        if mode is not None:
            args += ["--execution-mode", mode]
        if experiment_id is not None:
            args += ["--experiment-id", experiment_id]
        run.main(args + list(extra_args))
        return read_json(self.root / directory / "final_report.json")

    def assert_unclaimed(self):
        assert not self.experiments.exists(), "local rejection created an experiment ledger"
        assert self.requests == self.http_payloads == []


def exercise_experiment_cli(output=None, *, case="no_supplement"):
    """Export clearly labelled mock evidence, never fixture grant/credentials."""
    with ExperimentCliHarness() as harness:
        report = harness.invoke(case=case)
        run_dir = harness.root / "MOCK_run"
        exported = {"result_MOCK_ONLY.json": {"case": case, "status": report["status"],
            "real_external_calls": 0, "mock_http_attempt_count": len(harness.http_payloads),
            "production_ledger_unchanged": True, "real_experiment_ledger_unchanged": True,
            "no_real_credentials_read": True, "mock_authorization_exported": False,
            "workflow_live_counter_counts_mock_transport_only": True},
            "mock_final_report.json": report, "mock_requests.json": harness.requests,
            "mock_http_payloads.json": harness.http_payloads,
            "unchanged_actual_state_sha256.json": {
                "production_before": harness.real_before[0], "production_after": snapshot(harness.real_production),
                "experiment_before": harness.real_before[1], "experiment_after": snapshot(harness.real_experiments)},
            "mock_ledger_snapshot.json": read_json(run_dir / "ledger_snapshot.json"),
            "final_messages_MOCK_CAPTURE.json": harness.http_payloads[0]["messages"],
            "supplement_contract_identity.json": read_json(run_dir / "supplement_contract_identity.json")}
        for name in ("increment_patch_and_diff.json", "supplement_input_projection_mapping.json"):
            if (run_dir / name).exists():
                exported[name] = read_json(run_dir / name)
        assert MOCK_SECRET not in json.dumps(exported)
        if output is not None:
            from .readiness import save
            for name, value in exported.items():
                save(output, name, value)
        return exported


class ExperimentEntryTests(unittest.TestCase):
    def test_default_production_consumed_still_refuses_without_experiment_fallback(self):
        with ExperimentCliHarness() as harness:
            with self.assertRaisesRegex(ValueError, "already_claimed"):
                harness.invoke(mode=None, experiment_id=None)
            harness.assert_unclaimed()

    def test_explicit_experiment_sends_in_private_ledger_while_production_is_consumed(self):
        with ExperimentCliHarness() as harness:
            report = harness.invoke()
            self.assertEqual(report["status"], "NO_SUPPLEMENT_STOP")
            self.assertEqual(len(harness.http_payloads), 1)
            self.assertEqual(report["execution_scope"], harness.scope)
            self.assertTrue(report["extra_version_validation"])
            self.assertFalse(report["first_generation_result"])
            self.assertFalse(report["automatic_training_admission"])
            self.assertEqual(len(list(harness.experiments.glob("*.jsonl"))), 1)

    def test_missing_experiment_id_and_implicit_mode_rejected(self):
        for mode, experiment_id in (("extra-experiment", None), (None, EXPERIMENT_ID)):
            with self.subTest(mode=mode), ExperimentCliHarness() as harness:
                with self.assertRaises(SystemExit):
                    harness.invoke(mode=mode, experiment_id=experiment_id)
                harness.assert_unclaimed()

    def test_unknown_id_does_not_register_new_authorization(self):
        with ExperimentCliHarness() as harness:
            with self.assertRaisesRegex(ValueError, "unknown_extra_experiment"):
                harness.invoke(experiment_id="UNAUTHORIZED_NEW_ID")
            harness.assert_unclaimed()

    def test_changed_pinned_record_rejected_before_send(self):
        with ExperimentCliHarness() as harness:
            harness.approval_path.write_text(harness.approval_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "record_sha_mismatch"):
                harness.invoke()
            harness.assert_unclaimed()

    def test_execution_authorization_requires_exact_experiment_scope(self):
        with ExperimentCliHarness() as harness:
            with self.assertRaisesRegex(ValueError, "authorization_experiment_scope_mismatch"):
                harness.invoke(authorization_mutation=lambda value: value.pop("experiment_scope"))
            harness.assert_unclaimed()

    def test_pinned_scope_binding_cannot_change_allowed_sources_or_template_sha(self):
        for field, value in (("allowed_source_ids", ["ME08:C01"]), ("template_sha256", "0" * 64)):
            with self.subTest(field=field), ExperimentCliHarness() as harness:
                if field == "template_sha256":
                    harness.record["binding"]["template"][field] = value
                else:
                    harness.record["binding"][field] = value
                harness.approval_path.write_text(json.dumps(harness.record))
                harness.repin_tampered_private_record()
                with self.assertRaisesRegex(ValueError, "experiment_binding_mismatch"):
                    harness.invoke()
                harness.assert_unclaimed()

    def test_input_branch_source_sha_model_and_budget_changes_do_not_gain_scope(self):
        changes = {
            "branch": lambda h: h.bundle["identity"].update(branch_id="MOCK-CHANGED-BRANCH"),
            "history_sha": lambda h: h.bundle.update(history_sha256="0" * 64),
            "plan_sha": lambda h: h.bundle.update(original_plan_sha256="0" * 64),
            "source_sha": lambda h: h.bundle["frozen_sources"]["plan"].update(sha256="0" * 64),
            "model_parameter": lambda h: h.model.update(temperature=0.3),
            "budget": lambda h: h.budget["role_requests"].update(gate=7),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name), ExperimentCliHarness() as harness:
                mutate(harness)
                with self.assertRaises(ValueError):
                    harness.invoke()
                harness.assert_unclaimed()

    def test_legacy_template_cannot_use_aligned_experiment_grant(self):
        with ExperimentCliHarness() as harness:
            harness.bundle["supplement_prompt_version"] = LEGACY_VERSION
            with self.assertRaisesRegex(ValueError, "aligned_template_required"):
                harness.invoke()
            harness.assert_unclaimed()

    def test_url_credentials_invalid_input_and_budget_fail_without_claim(self):
        changes = {
            "url": lambda h: h.model.update(endpoint="https://offline.invalid/v1"),
            "credential": lambda h: h.environment.update({h.model["api_key_env"]: ""}),
            "input": lambda h: h.bundle["parent"]["dialogue"]["turns"][0].update(text="MOCK altered history"),
            "budget": lambda h: h.budget["role_requests"].update(supplement=2),
            "tokenizer": lambda h: h.model.update(tokenizer_json_sha256="0" * 64),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name), ExperimentCliHarness() as harness:
                mutate(harness)
                if name == "tokenizer":
                    harness.pin_fixture_approval()
                with self.assertRaises(ValueError):
                    harness.invoke()
                harness.assert_unclaimed()

    def test_known_capacity_and_first_batch_budget_blocks_precede_claim(self):
        for kind in ("capacity", "first_batch"):
            with self.subTest(kind=kind), ExperimentCliHarness() as harness:
                if kind == "capacity":
                    harness.budget["role_capacity"]["supplement"]["max_input_tokens"] = 1
                else:
                    harness.budget["role_requests"]["gate"] = 1
                harness.pin_fixture_approval()
                with self.assertRaisesRegex(ValueError, "known_.*blocked_before_attempt"):
                    harness.invoke()
                harness.assert_unclaimed()

    def test_repeated_command_and_changed_output_directory_do_not_reset(self):
        with ExperimentCliHarness() as harness:
            harness.invoke()
            with self.assertRaises((ValueError, FileExistsError)):
                harness.invoke()
            with self.assertRaisesRegex(ValueError, "already_claimed"):
                harness.invoke(directory="MOCK_second_output")
            self.assertEqual(len(harness.http_payloads), 1)

    def test_rejection_timeout_and_invalid_patch_remain_consumed_on_restart(self):
        for case in ("http_400", "timeout", "invalid_patch"):
            with self.subTest(case=case), ExperimentCliHarness() as harness:
                report = harness.invoke(case=case)
                expected = "SUPPLEMENT_INVALID_STOP" if case == "invalid_patch" else "SUPPLEMENT_FAILED_STOP_NO_RETRY"
                self.assertEqual(report["status"], expected)
                self.assertEqual(report["attempted_request_counts"]["supplement"], 1)
                for role in ("gate", "realizer", "adherence"):
                    self.assertEqual(report["attempted_request_counts"][role], 0)
                with self.assertRaisesRegex(ValueError, "already_claimed"):
                    harness.invoke(directory="MOCK_restart")
                self.assertEqual(len(harness.http_payloads), 1)
                if case in ("http_400", "timeout"):
                    evidence = read_json(harness.root / "MOCK_run/failures/001_http_evidence.json")
                    self.assertEqual(evidence["request_attempt_count"], 1)
                    self.assertEqual(evidence["retry_count"], 0)
                    self.assertEqual(evidence["fallback_count"], 0)
                    if case == "http_400":
                        self.assertEqual(evidence["http_status"], 400)
                        self.assertEqual(json.loads(evidence["response_body"]), {"error": "MOCK complete rejection"})
                    else:
                        self.assertTrue(evidence["response_unavailable"])

    def test_original_patch_reproduces_eight_errors_without_downstream_requests(self):
        with ExperimentCliHarness() as harness:
            report = harness.invoke(case="invalid_patch")
            merged = read_json(harness.root / "MOCK_run/increment_patch_and_diff.json")
            self.assertEqual(merged["errors"], EXPECTED_ERRORS)
            self.assertEqual(report["status"], "SUPPLEMENT_INVALID_STOP")
            self.assertEqual([r["role"] for r in harness.requests], ["supplement"])

    def test_valid_patch_continues_every_role_with_same_scope_and_finite_budget(self):
        with ExperimentCliHarness() as harness:
            report = harness.invoke(case="valid_patch")
            self.assertEqual(set(r["role"] for r in harness.requests), {"supplement", "gate", "realizer", "adherence"})
            for request in harness.requests:
                self.assertEqual(request["execution_scope"], harness.scope)
            for role, count in report["attempted_request_counts"].items():
                self.assertLessEqual(count, harness.budget["role_requests"][role])
            self.assertLessEqual(len(harness.http_payloads), 17)
            self.assertEqual(report["attempted_request_counts"]["supplement"], 1)
            ledger = read_json(harness.root / "MOCK_run/ledger_snapshot.json")
            self.assertEqual(sum(r["kind"] == "attempt" for r in ledger), len(harness.http_payloads))
            self.assertEqual(len(list(harness.experiments.glob("*.jsonl"))), 1)
            self.assertGreater(report["retained_scene_count"], 0)
            self.assertTrue(all(status != "realized" for status in report["goal_statuses"].values()))

    def test_final_http_messages_preserve_aligned_contract_and_original_input(self):
        with ExperimentCliHarness() as harness:
            harness.invoke()
            sent = harness.http_payloads[0]["messages"]
            content = sent[0]["content"]
            request = harness.requests[0]
            self.assertEqual(request["prompt_version"], ALIGNED_VERSION)
            mapping = read_json(harness.root / "MOCK_run/supplement_input_projection_mapping.json")
            restored = reconstruct_supplement_inputs(json.loads(content.rsplit("\n\nINPUT DATA\n", 1)[1]), mapping)
            self.assertEqual(restored["shared_feedback_view"], harness.bundle["shared_feedback_view"])
            self.assertEqual(restored["authoritative_history"], harness.bundle["parent"]["dialogue"]["turns"])
            self.assertEqual(len(restored["authoritative_history"]), 143)
            self.assertEqual(restored["original_plan"], harness.bundle["original_plan"])
            self.assertEqual(restored["affected_candidate_ids"], ["ME07:C01"])
            self.assertNotIn(EXPERIMENT_ID, content)
            self.assertNotIn("authorization_sha256", content)
            self.assertEqual(len(harness.http_payloads), 1)


if __name__ == "__main__":
    unittest.main()
