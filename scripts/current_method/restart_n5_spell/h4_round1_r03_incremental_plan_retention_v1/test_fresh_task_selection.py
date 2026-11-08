"""Fresh same-task routing with private grants, ledgers and mocked HTTP only."""
import json
import unittest
from unittest.mock import patch

from . import run
from .task_selection import VERSION
from .test_experiment_entry import ExperimentCliHarness
from .test_root_entry import RootCliHarness
from .test_task_selection_entry import ActualPatchHarness
from .workflow import execute_policy


ARGS = ("--fresh-normalization-version", VERSION)


def authorize_selection(value):
    value["fresh_normalization_version"] = VERSION


class FreshTaskSelectionTests(unittest.TestCase):
    def test_explicit_fresh_selection_uses_existing_rules_without_an_extra_request(self):
        with ActualPatchHarness() as harness:
            report = harness.invoke(extra_args=ARGS, authorization_mutation=authorize_selection)
            self.assertNotEqual(report["status"], "MERGED_OUTLINE_REJECT_STOP")
            self.assertGreater(report["retained_scene_count"], 0)
            self.assertEqual(report["fresh_normalization_version"], VERSION)
            self.assertNotIn("resume_claimed", report)
            normalization = json.loads((harness.root / "MOCK_run/plan_normalization.json").read_text())
            self.assertEqual((normalization["before_count"], normalization["after_count"]), (13, 12))
            self.assertEqual(normalization["unselected_variants"][0]["candidate_id"], "ND09:C01")
            self.assertEqual(normalization["unselected_variants"][0]["preferred_candidate_id"], "ND02:C01")
            self.assertEqual(report["selection_summary"]["not_selected"], normalization["not_selected"])
            for request in harness.requests:
                self.assertEqual(request["plan_normalization_version"], VERSION)
                if request["role"] != "supplement":
                    text = json.dumps(request["messages"])
                    self.assertNotIn("ND09", text)
                    self.assertNotIn("phase:new-event_2", text)
            self.assertEqual(report["attempted_request_counts"]["supplement"], 1)
            self.assertEqual(report["attempted_request_counts"]["planner"], 0)
            self.assertLessEqual(len(harness.requests), 17)

    def test_default_fresh_route_retains_original_outline_rejection(self):
        with ActualPatchHarness() as harness:
            report = harness.invoke()
            self.assertEqual(report["status"], "MERGED_OUTLINE_REJECT_STOP")
            self.assertEqual([request["role"] for request in harness.requests], ["supplement"])
            self.assertNotIn("fresh_normalization_version", report)
            self.assertFalse((harness.root / "MOCK_run/plan_normalization.json").exists())

    def test_cli_and_authorization_must_select_the_same_version_before_credentials(self):
        for args, mutation in ((ARGS, None), ((), authorize_selection)):
            with self.subTest(args=args), RootCliHarness() as harness:
                with patch.object(run, "credential_environment", side_effect=AssertionError("no credential access")):
                    with self.assertRaisesRegex(ValueError, "fresh_selection_authorization_mismatch"):
                        harness.invoke(extra_args=args, authorization_mutation=mutation)
                harness.assert_unclaimed()

    def test_non_root_and_saved_resume_cannot_use_fresh_selection(self):
        with ExperimentCliHarness() as harness:
            with self.assertRaisesRegex(ValueError, "fresh_selection_requires_explicit_fresh_root_entry"):
                harness.invoke(extra_args=ARGS, authorization_mutation=authorize_selection)
            harness.assert_unclaimed()
        with RootCliHarness() as harness:
            for extra in ({"saved_patch": {}}, {"normalization_version": VERSION}):
                with self.subTest(extra=extra):
                    with self.assertRaisesRegex(ValueError, "fresh_selection_requires_explicit_fresh_root_entry"):
                        execute_policy(bundle=harness.bundle, budget=harness.budget,
                            backend=harness.fixture, state_registry=harness.experiments,
                            artifact_dir=harness.root / "MOCK_direct", fresh_normalization_version=VERSION,
                            **extra)
            harness.assert_unclaimed()
            self.assertFalse((harness.root / "MOCK_direct").exists())

    def test_no_supplement_still_selects_and_executes_original_queue(self):
        with RootCliHarness() as harness:
            report = harness.invoke(case="no_supplement", extra_args=ARGS,
                                    authorization_mutation=authorize_selection)
            self.assertGreater(report["retained_scene_count"], 0)
            self.assertEqual(report["attempted_request_counts"]["supplement"], 1)
            self.assertTrue((harness.root / "MOCK_run/normalized_execution_queue.json").is_file())
            self.assertEqual(report["normalization_version"], VERSION)
