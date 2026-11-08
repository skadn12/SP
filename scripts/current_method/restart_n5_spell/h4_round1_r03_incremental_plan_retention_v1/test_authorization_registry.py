"""Data registration and actual entry tests with private state and HTTP mocks."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from unittest.mock import patch
import unittest

from . import experiment_authorization as approval, run
from .test_experiment_entry import EXPERIMENT_ID, ExperimentCliHarness
from .test_root_entry import RootCliHarness


def fixture_record(harness, experiment_id="MOCK_EXPLICITLY_APPROVED_OTHER_01", *, bundle=None):
    record = deepcopy(harness.record)
    record["experiment_id"] = experiment_id
    if bundle is not None:
        record["binding"] = approval.binding_for(bundle, harness.budget, harness.model)
    source = harness.root / (experiment_id + "_registration_input.json")
    source.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return source


def register(harness, source, *, explicitly_authorized=True, bundle=None, expected_sha=None):
    return approval.register_experiment_record(source,
        expected_sha256=expected_sha or approval.file_sha(source),
        explicitly_authorized=explicitly_authorized,
        bundle=harness.bundle if bundle is None else bundle,
        budget=harness.budget, model_config=harness.model,
        production_registry=harness.production)


class AuthorizationRegistryTests(unittest.TestCase):
    def test_01_registration_is_explicit_and_unknown_id_is_not_auto_registered(self):
        with ExperimentCliHarness() as harness:
            source = fixture_record(harness)
            before = approval.authorization_index_path().read_bytes()
            with self.assertRaises(ValueError):
                register(harness, source, explicitly_authorized=False)
            with self.assertRaisesRegex(ValueError, "unknown_extra_experiment"):
                approval.load_experiment_scope("MOCK_EXPLICITLY_APPROVED_OTHER_01",
                    bundle=harness.bundle, budget=harness.budget, model_config=harness.model,
                    production_registry=harness.production)
            self.assertEqual(approval.authorization_index_path().read_bytes(), before)
            harness.assert_unclaimed()

    def test_02_same_record_registration_is_idempotent_and_does_not_claim(self):
        with ExperimentCliHarness() as harness:
            index_before = approval.authorization_index_path().read_bytes()
            record_before = harness.approval_path.read_bytes()
            register(harness, harness.approval_path)
            register(harness, harness.approval_path)
            self.assertEqual(approval.authorization_index_path().read_bytes(), index_before)
            self.assertEqual(harness.approval_path.read_bytes(), record_before)
            self.assertEqual(approval.approved_records()[EXPERIMENT_ID], approval.file_sha(harness.approval_path))
            harness.assert_unclaimed()

    def test_03_same_id_with_different_valid_binding_rejects_without_overwrite(self):
        with ExperimentCliHarness() as harness:
            changed = deepcopy(harness.bundle)
            changed["identity"]["branch_id"] = "MOCK_DIFFERENT_APPROVED_INPUT"
            source = fixture_record(harness, EXPERIMENT_ID, bundle=changed)
            index_before = approval.authorization_index_path().read_bytes()
            record_before = harness.approval_path.read_bytes()
            with self.assertRaises(ValueError):
                register(harness, source, bundle=changed)
            self.assertEqual(approval.authorization_index_path().read_bytes(), index_before)
            self.assertEqual(harness.approval_path.read_bytes(), record_before)
            harness.assert_unclaimed()

    def test_04_hash_and_binding_mismatches_reject_before_registration(self):
        for case in ("sha", "input", "scope", "budget", "template"):
            with self.subTest(case=case), RootCliHarness() as harness:
                source = fixture_record(harness)
                before = approval.authorization_index_path().read_bytes()
                if case != "sha":
                    record = json.loads(source.read_text())
                    if case == "input":
                        record["binding"]["history_sha256"] = "0" * 64
                    elif case == "scope":
                        record["binding"]["incremental_scope"] = "unapproved_scope"
                    elif case == "budget":
                        record["binding"]["real_request_limit"] = 18
                    else:
                        record["binding"]["template"]["template_sha256"] = "0" * 64
                    source.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    register(harness, source, expected_sha="0" * 64 if case == "sha" else None)
                self.assertEqual(approval.authorization_index_path().read_bytes(), before)
                harness.assert_unclaimed()

    def test_05_concurrent_identical_registration_adds_one_immutable_record(self):
        with ExperimentCliHarness() as harness:
            source = fixture_record(harness)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(register, harness, source) for _ in range(2)]
                for future in futures:
                    future.result()
            rows = [json.loads(line) for line in approval.authorization_index_path().read_text().splitlines()]
            matching = [row for row in rows if row["experiment_id"] == "MOCK_EXPLICITLY_APPROVED_OTHER_01"]
            self.assertEqual(len(matching), 1)
            self.assertEqual(approval.approved_records()["MOCK_EXPLICITLY_APPROVED_OTHER_01"], approval.file_sha(source))
            harness.assert_unclaimed()

    def test_06_unrelated_registration_does_not_change_code_or_existing_scope(self):
        with RootCliHarness() as harness:
            manifest_before = run.implementation_manifest()
            self.assertFalse(any("experiment_authorizations" in path for path in manifest_before))
            scope_before = deepcopy(harness.scope)
            record_before = harness.approval_path.read_bytes()
            register(harness, fixture_record(harness))
            self.assertEqual(run.implementation_manifest(), manifest_before)
            self.assertEqual(harness.approval_path.read_bytes(), record_before)
            self.assertEqual(approval.load_experiment_scope(EXPERIMENT_ID,
                bundle=harness.bundle, budget=harness.budget, model_config=harness.model,
                production_registry=harness.production), scope_before)
            report = harness.invoke(case="invalid_source")
            self.assertEqual(report["attempted_request_counts"]["supplement"], 1)
            self.assertEqual(report["execution_scope"], scope_before)

    def test_07_reregistering_consumed_record_does_not_reset_ledger(self):
        with RootCliHarness() as harness:
            harness.invoke(case="invalid_source")
            ledger = next(harness.experiments.glob("*.jsonl"))
            before = ledger.read_bytes()
            register(harness, harness.approval_path)
            self.assertEqual(ledger.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "already_claimed"):
                harness.invoke(directory="MOCK_other_directory_after_reregister")
            self.assertEqual(len(harness.http_payloads), 1)
            self.assertEqual(ledger.read_bytes(), before)

    def test_08_registration_cli_to_real_entry_and_single_mock_send(self):
        from . import register_experiment
        with RootCliHarness() as harness:
            source = fixture_record(harness)
            for name, value in (("bundle", harness.bundle), ("budget", harness.budget), ("model", harness.model)):
                (harness.root / ("register_" + name + ".json")).write_text(json.dumps(value))
            args = ["--record", str(source), "--expected-sha256", approval.file_sha(source),
                "--explicitly-authorized", "--bundle", str(harness.root / "register_bundle.json"),
                "--budget", str(harness.root / "register_budget.json"),
                "--model-config", str(harness.root / "register_model.json")]
            with patch.object(register_experiment, "production_registry", return_value=harness.production):
                register_experiment.main(args)
            harness.assert_unclaimed()
            experiment_id = "MOCK_EXPLICITLY_APPROVED_OTHER_01"
            harness.scope = approval.load_experiment_scope(experiment_id,
                bundle=harness.bundle, budget=harness.budget, model_config=harness.model,
                production_registry=harness.production)
            report = harness.invoke(case="invalid_source", experiment_id=experiment_id)
            self.assertEqual(report["execution_scope"]["experiment_id"], experiment_id)
            self.assertEqual(report["attempted_request_counts"]["supplement"], 1)
            self.assertEqual(len(harness.http_payloads), 1)

    def test_09_explicit_current_code_record_starts_but_old_code_identity_cannot(self):
        with RootCliHarness() as harness:
            manifest = run.implementation_manifest()
            self.assertIn(str(run.__file__), manifest)
            with self.assertRaisesRegex(ValueError, "authorization_frozen_identity_mismatch"):
                harness.invoke(authorization_mutation=lambda value: value["implementation_manifest"].update({str(run.__file__): "0" * 64}))
            harness.assert_unclaimed()
            report = harness.invoke(case="invalid_source", directory="MOCK_current_explicit_code")
            self.assertEqual(report["status"], "SUPPLEMENT_INVALID_STOP")
            self.assertEqual(len(harness.http_payloads), 1)

    def test_10_equivalent_reregistration_preserves_original_record_bytes_and_sha(self):
        with ExperimentCliHarness() as harness:
            original = harness.approval_path.read_bytes()
            index_before = approval.authorization_index_path().read_bytes()
            source = harness.root / "MOCK_equivalent_json_different_serialization.json"
            source.write_text(json.dumps(harness.record, sort_keys=True, separators=(",", ":")))
            self.assertNotEqual(source.read_bytes(), original)
            result = register(harness, source)
            self.assertEqual(result["record_sha256"], approval.file_sha(harness.approval_path))
            self.assertEqual(harness.approval_path.read_bytes(), original)
            self.assertEqual(approval.authorization_index_path().read_bytes(), index_before)
            harness.assert_unclaimed()

    def test_11_concurrent_conflicting_registration_cannot_overwrite_winner(self):
        with ExperimentCliHarness() as harness:
            experiment_id = "MOCK_CONCURRENT_CONFLICT_01"
            source_a = fixture_record(harness, experiment_id)
            bundle_b = deepcopy(harness.bundle)
            bundle_b["identity"]["branch_id"] = "MOCK_SECOND_CONFLICTING_BINDING"
            record_b = deepcopy(harness.record)
            record_b["experiment_id"] = experiment_id
            record_b["binding"] = approval.binding_for(bundle_b, harness.budget, harness.model)
            source_b = harness.root / "MOCK_conflicting_second_record.json"
            source_b.write_text(json.dumps(record_b))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(register, harness, source_a),
                           pool.submit(register, harness, source_b, bundle=bundle_b)]
                results, failures = [], []
                for future in futures:
                    try:
                        results.append(future.result())
                    except ValueError as exc:
                        failures.append(str(exc))
            self.assertEqual(len(results), 1)
            self.assertEqual(failures, ["experiment_registration_conflict"])
            target = harness.approvals / (experiment_id + ".json")
            self.assertEqual(approval.approved_records()[experiment_id], approval.file_sha(target))
            self.assertIn(target.read_bytes(), (source_a.read_bytes(), source_b.read_bytes()))
            rows = [json.loads(line) for line in approval.authorization_index_path().read_text().splitlines()]
            self.assertEqual(sum(row["experiment_id"] == experiment_id for row in rows), 1)
            harness.assert_unclaimed()

    def test_12_duplicate_index_entries_fail_closed_without_send(self):
        with ExperimentCliHarness() as harness:
            index = approval.authorization_index_path()
            original = index.read_bytes()
            index.write_bytes(original + original)
            with self.assertRaisesRegex(ValueError, "invalid_experiment_approval_index"):
                harness.invoke()
            harness.assert_unclaimed()


if __name__ == "__main__":
    unittest.main()
