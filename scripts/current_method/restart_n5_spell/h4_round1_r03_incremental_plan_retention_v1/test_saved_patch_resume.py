"""Saved-patch resume checks use private ledgers and socket-blocked HTTP mocks."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import resume, run
from .budget import BudgetLedger
from .planning import sha
from .plan_normalization import VERSION
from .test_root_entry import RootCliHarness
from .test_root_planning import independent_new_patch
from .test_experiment_entry import EXPERIMENT_ID, snapshot
from .test_readiness import Response

ACTUAL = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_authorization_registry_v1_execution_live_1")


class DuplicateMockHarness(RootCliHarness):
    def http_open(self, request, **kwargs):
        actual = self.requests[-1]
        if actual["role"] != "supplement":
            return super().http_open(request, **kwargs)
        payload = json.loads(request.data)
        self.http_payloads.append(payload)
        content = independent_new_patch(self.bundle)
        duplicate = deepcopy(content["new_developments"][0])
        duplicate["phase_id"] = "phase:independent_alias"
        content["new_developments"].append(duplicate)
        return Response(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(content)}}]}).encode())


def freeze_mock_resume(harness):
    report = harness.invoke()
    if report["status"] != "MERGED_OUTLINE_REJECT_STOP":
        raise AssertionError(report["status"])
    manifest = resume.build_resume_manifest(harness.root / "MOCK_run",
                                           source_authorization=harness.root / "authorization.json")
    manifest_path = harness.root / "resume_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    authorization = resume.read(harness.root / "authorization.json")
    authorization.update(implementation_manifest=run.implementation_manifest(),
        resume_source_manifest_sha256=sha(manifest), normalization_version=VERSION,
        resume_explicitly_authorized=True)
    auth_path = harness.root / "resume_authorization.json"
    auth_path.write_text(json.dumps(authorization))
    harness.stack.enter_context(patch.object(resume, "production_registry", return_value=harness.production))
    harness.stack.enter_context(patch.object(resume, "experiment_registry", return_value=harness.experiments))
    return manifest, ["--source-manifest", str(manifest_path), "--run-dir", str(harness.root / "MOCK_resume"),
        "--authorization", str(auth_path), "--experiment-id", EXPERIMENT_ID,
        "--normalization-version", VERSION, "--execute"]


class SavedPatchResumeTests(unittest.TestCase):
    def test_real_source_reproduces_collision_and_blocks_before_credentials_or_ledger(self):
        with run.offline_network_guard(), tempfile.TemporaryDirectory(prefix="h4-resume-readonly-") as directory:
            manifest = resume.build_resume_manifest(ACTUAL / "experiment/execution",
                source_authorization=ACTUAL / "experiment/inputs/authorization.json")
            source = resume.validate_saved_source(manifest)
            before = Path(manifest["ledger_path"]).read_bytes()
            path = Path(directory) / "source.json"
            path.write_text(json.dumps(manifest))
            with patch.object(resume, "credential_environment", side_effect=AssertionError("credentials forbidden")), \
                    patch.object(resume, "SingleShotTransport", side_effect=AssertionError("transport forbidden")):
                result = resume.main(["--source-manifest", str(path), "--run-dir", str(Path(directory) / "blocked"),
                    "--experiment-id", manifest["execution_scope"]["experiment_id"],
                    "--normalization-version", VERSION, "--execute"])
            self.assertEqual(result["status"], "PLAN_NORMALIZATION_CONFLICT_STOP")
            self.assertFalse(result["resume_claimed"])
            self.assertEqual(result["external_call_count"], 0)
            self.assertEqual(before, Path(manifest["ledger_path"]).read_bytes())
            self.assertEqual(source["saved_patch"]["status"], "PATCH")

    def test_saved_true_duplicate_runs_full_mock_chain_on_same_ledger_without_supplement(self):
        with DuplicateMockHarness() as harness:
            manifest, args = freeze_mock_resume(harness)
            before = Path(manifest["ledger_path"]).read_bytes()
            result = resume.main(args)
            self.assertEqual(result["status"], "EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY")
            self.assertEqual(result["attempted_request_counts"]["supplement"], 0)
            self.assertEqual(result["experiment_total_attempted_counts"]["supplement"], 1)
            self.assertEqual(result["retained_scene_count"], 7)
            self.assertTrue(Path(manifest["ledger_path"]).read_bytes().startswith(before))
            roles = [request["role"] for request in harness.requests]
            self.assertEqual(roles.count("supplement"), 1)
            self.assertEqual(roles.count("gate"), 7)
            self.assertEqual(roles.count("realizer"), 4)
            self.assertEqual(roles.count("adherence"), 4)
            self.assertTrue(all(request["execution_scope"] == harness.scope for request in harness.requests))
            normalization = resume.read(harness.root / "MOCK_resume/plan_normalization.json")
            self.assertEqual(len(normalization["aliases"]), 1)
            self.assertEqual(normalization["candidate_aliases"], {"ND10:C01": "ND09:C01"})
            self.assertLessEqual(len(roles), 17)

    def test_same_command_or_new_output_directory_cannot_repeat_downstream(self):
        with DuplicateMockHarness() as harness:
            _, args = freeze_mock_resume(harness)
            resume.main(args)
            before = len(harness.requests)
            args[args.index("--run-dir") + 1] = str(harness.root / "different_output")
            with self.assertRaisesRegex(ValueError, "ledger_changed"):
                resume.main(args)
            self.assertEqual(len(harness.requests), before)

    def test_timeout_consumes_downstream_attempt_and_cannot_resume_again(self):
        with DuplicateMockHarness() as harness:
            manifest, args = freeze_mock_resume(harness)
            harness.case = "timeout"
            result = resume.main(args)
            self.assertEqual(result["status"], "STAGE_FAILURE_STOP_NO_RETRY")
            self.assertEqual(result["attempted_request_counts"]["gate"], 1)
            self.assertEqual(result["experiment_total_attempted_counts"]["supplement"], 1)
            self.assertEqual(result["unused_reserved_request_counts"]["supplement"], 0)
            before = Path(manifest["ledger_path"]).read_bytes()
            args[args.index("--run-dir") + 1] = str(harness.root / "restart_after_timeout")
            with self.assertRaisesRegex(ValueError, "ledger_changed"):
                resume.main(args)
            self.assertEqual(Path(manifest["ledger_path"]).read_bytes(), before)
            self.assertEqual(len(harness.requests), 2)

    def test_changed_source_bytes_or_manifest_identity_refuses(self):
        with DuplicateMockHarness() as harness:
            manifest, _ = freeze_mock_resume(harness)
            for mutation in (lambda m: m.update(identity={"root_id": "foreign"}),
                             lambda m: m.update(ledger_sha256="0" * 64),
                             lambda m: m["files"].update({"input_freeze.json": "0" * 64})):
                with self.subTest(mutation=mutation):
                    changed = deepcopy(manifest)
                    mutation(changed)
                    with self.assertRaises(ValueError):
                        resume.validate_saved_source(changed)
            self.assertEqual(len(harness.requests), 1)

    def test_current_authorization_or_credentials_failure_does_not_consume_resume(self):
        for case in ("authorization", "credentials"):
            with self.subTest(case=case), DuplicateMockHarness() as harness:
                manifest, args = freeze_mock_resume(harness)
                before = Path(manifest["ledger_path"]).read_bytes()
                if case == "authorization":
                    auth_path = Path(args[args.index("--authorization") + 1])
                    auth = resume.read(auth_path)
                    auth["resume_source_manifest_sha256"] = "0" * 64
                    auth_path.write_text(json.dumps(auth))
                else:
                    harness.environment.update({harness.model["api_key_env"]: ""})
                with self.assertRaises(ValueError):
                    resume.main(args)
                self.assertEqual(len(harness.requests), 1)
                self.assertEqual(Path(manifest["ledger_path"]).read_bytes(), before)

    def test_atomic_resume_claim_allows_one_winner(self):
        with DuplicateMockHarness() as harness:
            manifest, _ = freeze_mock_resume(harness)
            source = resume.validate_saved_source(manifest)
            ledger = BudgetLedger(harness.experiments, root_id=harness.bundle["identity"]["root_id"],
                round_index=3, budget=harness.budget, execution_scope=harness.scope)
            def claim(_):
                try:
                    ledger.claim_resume({"identity": harness.bundle["identity"]},
                        source_ledger_sha256=manifest["ledger_sha256"],
                        source_ledger_header=source["resume_context"]["source_ledger_header"])
                    return "won"
                except ValueError:
                    return "refused"
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(claim, range(2)))
            self.assertEqual(sorted(results), ["refused", "won"])
            self.assertEqual(len([row for row in ledger.snapshot() if row["kind"] == "resume_execution"]), 1)
            self.assertEqual(len(harness.requests), 1)


if __name__ == "__main__":
    unittest.main()
