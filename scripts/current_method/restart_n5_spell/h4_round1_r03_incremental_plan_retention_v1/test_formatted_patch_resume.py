"""Saved format failures resume on private ledgers with socket-blocked mocks."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from . import formatted_resume_source as source_loader, resume, run
from .planning import sha
from .root_scope import with_supplement_formatting
from .supplement_format import VERSION as FORMAT_VERSION
from .task_selection import VERSION as SELECTION_VERSION
from .workflow import prepare_saved_resume
from .test_experiment_entry import EXPERIMENT_ID, snapshot
from .test_supplement_format_entry import FormattingRootCliHarness

ACTUAL = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_aligned_prompts_single_root_v1_live_1")


def freeze_formatted_mock(harness, *, case="mixed"):
    report = harness.invoke(case=case)
    if report["status"] != "SUPPLEMENT_INVALID_STOP":
        raise AssertionError(report["status"])
    planning = Path(source_loader.__file__).resolve().with_name("root_planning.py")
    target = harness.root / "method_snapshot" / planning.relative_to(planning.parents[4])
    target.parent.mkdir(parents=True)
    shutil.copyfile(planning, target)
    manifest = source_loader.build_formatted_resume_manifest(harness.root / "MOCK_run",
        source_authorization=harness.root / "authorization.json")
    manifest_path = harness.root / "formatted_resume_source.json"
    manifest_path.write_text(json.dumps(manifest))
    authorization = resume.read(harness.root / "authorization.json")
    authorization.update(implementation_manifest=run.implementation_manifest(),
        bundle_sha256=sha(with_supplement_formatting(harness.bundle)),
        resume_source_manifest_sha256=sha(manifest), normalization_version=SELECTION_VERSION,
        resume_explicitly_authorized=True, formatted_supplement_resume_explicitly_authorized=True,
        supplement_format_version=FORMAT_VERSION)
    auth_path = harness.root / "formatted_resume_authorization.json"
    auth_path.write_text(json.dumps(authorization))
    harness.stack.enter_context(patch.object(resume, "production_registry", return_value=harness.production))
    harness.stack.enter_context(patch.object(resume, "experiment_registry", return_value=harness.experiments))
    args = ["--source-manifest", str(manifest_path), "--run-dir", str(harness.root / "MOCK_resume"),
        "--authorization", str(auth_path), "--experiment-id", EXPERIMENT_ID,
        "--normalization-version", SELECTION_VERSION, "--execute"]
    return manifest, args


class FormattedPatchResumeTests(unittest.TestCase):
    def test_actual_failure_replays_with_twelve_candidates_and_unchanged_source(self):
        with run.offline_network_guard():
            before = snapshot(ACTUAL)
            manifest = source_loader.build_formatted_resume_manifest(ACTUAL / "execution",
                source_authorization=ACTUAL / "inputs/authorization.json")
            ledger_before = Path(manifest["ledger_path"]).read_bytes()
            saved_journal = resume.read(ACTUAL / "execution/ledger_snapshot.json")
            journal = [json.loads(line) for line in ledger_before.splitlines() if line.strip()]
            if journal == saved_journal:
                source = source_loader.validate_formatted_source(manifest)
                self.assertEqual(source["original_failure"]["errors"],
                                 ["patch_phase_1_identity", "root_new_development_1_shape"])
                preflight = source["format_preflight"]
                self.assertIsInstance(source["saved_patch"], str)
                self.assertEqual(source["saved_patch"], resume.read(ACTUAL / "execution/responses/001_supplement.json"))
                self.assertEqual(source["bundle"], with_supplement_formatting(source["original_bundle"]))
            else:
                # A real authorized continuation may have consumed this source
                # after the test was added. Its old raw response remains replayable.
                with self.assertRaisesRegex(ValueError, "original_ledger_snapshot_mismatch"):
                    source_loader.validate_formatted_source(manifest)
                bundle = with_supplement_formatting(resume.read(ACTUAL / "execution/input_freeze.json"))
                raw = resume.read(ACTUAL / "execution/responses/001_supplement.json")
                preflight = prepare_saved_resume(bundle, raw, normalization_version=SELECTION_VERSION)
            self.assertTrue(preflight["valid"])
            self.assertEqual(len(preflight["normalization"]["queue"]), 12)
            self.assertEqual(snapshot(ACTUAL), before)
            self.assertEqual(Path(manifest["ledger_path"]).read_bytes(), ledger_before)

    def test_format_failure_resumes_same_ledger_without_repeating_supplement(self):
        with FormattingRootCliHarness(use_formatter=False) as harness:
            manifest, args = freeze_formatted_mock(harness)
            old_source = snapshot(harness.root / "MOCK_run")
            ledger_before = Path(manifest["ledger_path"]).read_bytes()
            report = resume.main(args)
            self.assertEqual(report["status"], "EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY")
            self.assertEqual(report["attempted_request_counts"], {
                "planner": 0, "supplement": 0, "gate": 8, "realizer": 4,
                "adherence": 4, "closure": 0})
            self.assertEqual(report["experiment_total_attempted_counts"]["supplement"], 1)
            self.assertEqual(report["retained_scene_count"], 8)
            self.assertEqual([row["role"] for row in harness.requests].count("supplement"), 1)
            self.assertTrue(Path(manifest["ledger_path"]).read_bytes().startswith(ledger_before))
            self.assertEqual(snapshot(harness.root / "MOCK_run"), old_source)
            formatted = resume.read(harness.root / "MOCK_resume/supplement_formatting.json")
            self.assertEqual(formatted["raw_output"], harness.raw_supplement)
            self.assertEqual(formatted["formatter_version"], FORMAT_VERSION)
            self.assertEqual(report["execution_scope"], harness.scope)
            args[args.index("--run-dir") + 1] = str(harness.root / "repeat_attempt")
            with self.assertRaisesRegex(ValueError, "ledger_changed"):
                resume.main(args)
            self.assertEqual(len(harness.requests), 17)

    def test_tampered_manifest_raw_and_ledger_are_rejected_before_credentials(self):
        with FormattingRootCliHarness(use_formatter=False) as harness:
            manifest, _ = freeze_formatted_mock(harness)
            for mutate in (
                lambda value: value.update(identity={"root_id": "foreign"}),
                lambda value: value.update(ledger_sha256="0" * 64),
                lambda value: value.update(supplement_format_version="unknown"),
                lambda value: value["files"].update({"responses/001_supplement.json": "0" * 64}),
                lambda value: value["original_planning"].update(sha256="0" * 64),
            ):
                changed = deepcopy(manifest)
                mutate(changed)
                with self.subTest(change=changed), self.assertRaises(ValueError):
                    source_loader.validate_formatted_source(changed)
            self.assertEqual(len(harness.requests), 1)

    def test_repinning_changed_raw_does_not_bypass_http_evidence(self):
        with FormattingRootCliHarness(use_formatter=False) as harness:
            _, _ = freeze_formatted_mock(harness)
            raw = harness.root / "MOCK_run/responses/001_supplement.json"
            raw.write_text(json.dumps(harness.raw_supplement + " "))
            changed = source_loader.build_formatted_resume_manifest(harness.root / "MOCK_run",
                source_authorization=harness.root / "authorization.json")
            with self.assertRaisesRegex(ValueError, "model_response_mismatch"):
                source_loader.validate_formatted_source(changed)
            self.assertEqual(len(harness.requests), 1)

    def test_duplicate_keys_in_saved_raw_are_not_lost_by_old_parser(self):
        with FormattingRootCliHarness(use_formatter=False) as harness:
            manifest, _ = freeze_formatted_mock(harness, case="duplicate_key")
            with self.assertRaisesRegex(ValueError, "formatted_patch_or_selection_invalid"):
                source_loader.validate_formatted_source(manifest)
            self.assertEqual(len(harness.requests), 1)

    def test_current_authorization_must_bind_formatter_and_derived_bundle(self):
        for field in ("bundle_sha256", "formatted_supplement_resume_explicitly_authorized",
                      "supplement_format_version", "resume_source_manifest_sha256"):
            with self.subTest(field=field), FormattingRootCliHarness(use_formatter=False) as harness:
                manifest, args = freeze_formatted_mock(harness)
                before = Path(manifest["ledger_path"]).read_bytes()
                auth_path = Path(args[args.index("--authorization") + 1])
                authorization = resume.read(auth_path)
                authorization.pop(field)
                auth_path.write_text(json.dumps(authorization))
                with patch.object(resume, "credential_environment", side_effect=AssertionError("credentials forbidden")):
                    with self.assertRaises(ValueError):
                        resume.main(args)
                self.assertEqual(Path(manifest["ledger_path"]).read_bytes(), before)
                self.assertEqual(len(harness.requests), 1)

    def test_legacy_manifest_does_not_select_formatter(self):
        with FormattingRootCliHarness(use_formatter=False) as harness:
            _, _ = freeze_formatted_mock(harness)
            legacy = resume.build_resume_manifest(harness.root / "MOCK_run",
                source_authorization=harness.root / "authorization.json")
            with self.assertRaises(ValueError):
                resume.validate_saved_source(legacy)
            with self.assertRaisesRegex(ValueError, "manifest_version_mismatch"):
                source_loader.validate_formatted_source(legacy)


if __name__ == "__main__":
    unittest.main()
