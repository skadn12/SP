"""Production feedback source and fail-closed input tests; no network/state."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import bundle as module
from .bundle import build_failed_bundle, validate_bundle_feedback, validate_shared_feedback_view
from .fixtures import failed_bundle, FixtureBackend, example_budget


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = patch("socket.socket.connect", side_effect=AssertionError("OFFLINE_NETWORK_BLOCKED"))
        cls.network.start()
        cls.bundle = build_failed_bundle(branch_id="offline-bundle-regression")

    @classmethod
    def tearDownClass(cls):
        cls.network.stop()

    def test_01_production_builder_matches_original_sent_content(self):
        bundle = self.bundle
        sources = bundle["shared_feedback_lineage"]["sources"]
        actual = module.feedback_from_actual_prompt(Path(sources["original_prompt"]["path"]).read_text())
        self.assertEqual(bundle["shared_feedback_view"], actual)
        self.assertEqual(actual["historical_qa"]["canonical_gold"], ["I2", "I1", "I3", "I4"])
        self.assertEqual(len(actual["historical_qa"]["options_or_items"]), 4)
        self.assertEqual(len(actual["historical_qa"]["evidence_refs"]), 4)
        self.assertEqual(actual["solver_observations"]["M_clean_policy"]["full_context"]["correct_count"], 10)
        self.assertEqual(actual["solver_observations"]["M_inclusive_policy"]["no_context"]["correct_count"], 1)
        self.assertTrue(validate_bundle_feedback(bundle)["original_sent_content_match"])

    def test_02_production_builder_has_no_fixture_import_or_notice(self):
        source = Path(module.__file__).read_text()
        self.assertNotIn("from .fixtures", source)
        self.assertNotIn("fixture_notice", self.bundle)

    def test_03_fixture_uses_same_production_bundle_data(self):
        fixture = failed_bundle()
        for key in ("shared_feedback_view", "shared_feedback_lineage", "original_plan", "parent", "frozen_sources"):
            self.assertEqual(fixture[key], self.bundle[key], key)
        self.assertIn("fixture_notice", fixture)

    def test_04_source_plan_history_intact(self):
        self.assertEqual(len(self.bundle["parent"]["dialogue"]["turns"]), 143)
        self.assertEqual(len(self.bundle["original_plan"]["selected_continuations"]), 3)
        self.assertEqual(len(self.bundle["original_plan"]["new_development_candidates"]), 8)
        plan_file = json.loads(Path(self.bundle["frozen_sources"]["plan"]["path"]).read_text())
        self.assertEqual(plan_file["value"], self.bundle["original_plan"])
        self.assertEqual(self.bundle["retained_r03_turns"], [])

    def test_05_missing_null_empty_feedback(self):
        for value in ("missing", None, {}):
            item = deepcopy(self.bundle)
            if value == "missing":
                item.pop("shared_feedback_view")
            else:
                item["shared_feedback_view"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "shared_feedback_view_required"):
                validate_bundle_feedback(item)

    def test_06_qa_items_question_gold_required(self):
        for key in ("options_or_items", "question", "canonical_gold"):
            item = deepcopy(self.bundle)
            item["shared_feedback_view"]["historical_qa"].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_bundle_feedback(item)

    def test_07_root_source_snapshot_content_tamper_rejected(self):
        for key in ("root_id", "source_candidate_id", "source_cluster_id", "observation_snapshot_id"):
            item = deepcopy(self.bundle)
            view = item["shared_feedback_view"]
            if key == "observation_snapshot_id":
                view["solver_observations"]["M_clean_policy"][key] = "different-snapshot"
            else:
                view[key] = "different-source"
            # Updating the asserted lineage cannot replace original sent evidence.
            item["shared_feedback_lineage"].update(module._feedback_identity(view))
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_bundle_feedback(item)

    def test_08_missing_lineage_source_or_changed_source_hash_rejected(self):
        for mutation in ("lineage", "source", "sha"):
            item = deepcopy(self.bundle)
            if mutation == "lineage":
                item.pop("shared_feedback_lineage")
            elif mutation == "source":
                item["shared_feedback_lineage"]["sources"].pop("question_bank")
            else:
                item["shared_feedback_lineage"]["sources"]["original_prompt"]["sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_bundle_feedback(item)

    def test_09_absent_optional_evidence_and_counts_not_invented(self):
        item = deepcopy(self.bundle)
        view = item["shared_feedback_view"]
        view["historical_qa"].pop("evidence_refs")
        view["historical_qa"].pop("source_data_status")
        for row in view["solver_observations"].values():
            row["full_context"] = {"accuracy": 0.75, "sample_size": "UNKNOWN"}
            row["no_context"] = {"accuracy": "UNKNOWN", "sample_size": "UNKNOWN"}
        before = deepcopy(view)
        validate_shared_feedback_view(view)
        item["shared_feedback_lineage"].update(module._feedback_identity(view))
        # This local source mock represents genuinely absent fields, not a
        # deletion from the real frozen source, which equality checks reject.
        with patch.object(module, "_recover_feedback", return_value=deepcopy(view)):
            self.assertTrue(validate_bundle_feedback(item)["valid"])
        self.assertEqual(view, before)
        self.assertNotIn("M0", view["solver_observations"])

    def test_10_plan_identity_not_relabelled(self):
        item = deepcopy(self.bundle)
        item["original_plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "original_plan"):
            validate_bundle_feedback(item)

    def test_11_feedback_change_not_merely_serialization_difference(self):
        item = deepcopy(self.bundle)
        item["shared_feedback_view"] = json.loads(json.dumps(item["shared_feedback_view"], sort_keys=True, indent=4))
        self.assertTrue(validate_bundle_feedback(item)["valid"])
        item["shared_feedback_view"]["historical_qa"]["question"] += " Changed target."
        with self.assertRaisesRegex(ValueError, "differs_from_original_source"):
            validate_bundle_feedback(item)

    def test_12_invalid_feedback_stops_actual_workflow_before_state_or_send(self):
        from .workflow import execute_policy
        for case in ("missing", "null", "empty", "items", "root", "source", "cluster", "snapshot"):
            item = deepcopy(self.bundle)
            if case == "missing":
                item.pop("shared_feedback_view")
            elif case == "null":
                item["shared_feedback_view"] = None
            elif case == "empty":
                item["shared_feedback_view"] = {}
            elif case == "items":
                item["shared_feedback_view"]["historical_qa"].pop("options_or_items")
            elif case == "snapshot":
                item["shared_feedback_view"]["solver_observations"]["M_clean_policy"]["observation_snapshot_id"] = "wrong"
            else:
                key = {"root": "root_id", "source": "source_candidate_id", "cluster": "source_cluster_id"}[case]
                item["shared_feedback_view"][key] = "wrong"
            backend = FixtureBackend(item)
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="h4-feedback-negative-") as temp:
                state = Path(temp) / "mock-state"
                with self.assertRaises(ValueError):
                    execute_policy(bundle=item, budget=example_budget(), backend=backend,
                                   state_registry=state, artifact_dir=Path(temp) / "mock-run")
                self.assertEqual(backend.requests, [])
                self.assertFalse(state.exists(), "Preflight failure must not consume any execution/supplement state")


if __name__ == "__main__":
    unittest.main()
