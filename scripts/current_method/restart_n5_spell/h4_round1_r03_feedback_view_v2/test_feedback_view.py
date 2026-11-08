"""Offline contract tests for the v2 shared feedback view."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from . import EXTERNAL_CALLS, HIDDEN_BACKEND_KEYS
from .compiler import expansion_view_v3, hidden_keys, load_frozen_inputs, qa_fresh_view_v3
from .preflight import run
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.profiles import (
    EXPANDED_PROFILE,
    EXPANDED_SINGLE_PLAN_PROFILE,
    LEGACY_PROFILE,
)


class SharedFeedbackViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Tests must never rewrite the immutable, versioned preflight artifact.
        # The production preflight caller supplies an explicit output path.
        cls._output_dir = TemporaryDirectory(prefix="h4_feedback_view_test_")
        cls.report = run(Path(cls._output_dir.name))
        cls.inputs = load_frozen_inputs()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._output_dir.cleanup()

    def test_preflight_pass_and_all_calls_zero(self) -> None:
        self.assertEqual(self.report["status"], "EXPANDED_SINGLE_PLAN_OFFLINE_PREFLIGHT_PASS")
        self.assertTrue(self.report["external_calls_zero"])
        self.assertTrue(all(value == 0 for value in EXTERNAL_CALLS.values()))

    def test_authoritative_scope_is_parent_r01_r02(self) -> None:
        self.assertEqual(len(self.inputs["dialogues"]), 8)
        for row in self.inputs["dialogues"].values():
            self.assertEqual(row["source_prefix"], "Parent+R01+R02-feedback")
            self.assertTrue(row["checks"]["r03_turns_absent"])
            self.assertTrue(row["checks"]["latest_round_is_r02"])
            self.assertEqual(len(row["turns"]), row["turn_count"])

    def test_shared_backend_has_complete_historical_qa(self) -> None:
        for record in self.inputs["records"]:
            qa = record["historical_qa"]
            self.assertIn(qa["question_type"], {"single_choice", "multi_select", "ordering"})
            self.assertTrue(qa["question"])
            self.assertTrue(qa["options_or_items"])
            self.assertIsInstance(qa["canonical_gold"], list)

    def test_expansion_view_does_not_leak_backend_management(self) -> None:
        view = expansion_view_v3(self.inputs["records"][0])
        self.assertFalse(hidden_keys(view))
        self.assertIn("canonical_gold", view["historical_qa"])
        self.assertTrue(view["historical_qa"]["options_or_items"])
        self.assertNotIn("position_mechanism", view["solver_observations"]["M_clean_policy"])
        self.assertTrue(HIDDEN_BACKEND_KEYS)

    def test_fresh_view_is_not_bound_to_old_question(self) -> None:
        view = qa_fresh_view_v3(self.inputs["records"][0], [])
        self.assertNotIn("historical_qa", view)
        self.assertNotIn("source_candidate_id", view)
        self.assertIn("source_cluster_id", view)
        self.assertFalse(hidden_keys(view))

    def test_templates_and_prompt_renderers_are_connected(self) -> None:
        checks = self.report["checks"]
        self.assertTrue(checks["unresolved_placeholders_absent"])
        self.assertTrue(self.report["planner_realizer_existing_builders_invoked_offline"])
        self.assertTrue(self.report["qa_existing_revision_1_builders_invoked_offline"])
        self.assertTrue(self.report["qa_output_contract_revision_1_unchanged"])

    def test_feedback_route_is_explicitly_profile_scoped(self) -> None:
        self.assertIsNone(LEGACY_PROFILE.feedback_view_version)
        self.assertEqual(EXPANDED_PROFILE.feedback_view_version, "h4_round1_r03_feedback_view_v2")
        self.assertTrue(EXPANDED_PROFILE.event_bucket_only)
        self.assertNotEqual(LEGACY_PROFILE.realizer_input_mode, EXPANDED_PROFILE.realizer_input_mode)

    def test_current_expanded_route_is_single_plan_and_historical_route_is_preserved(self) -> None:
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.name, "expanded_single_plan_v1")
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.max_plan_variants, 1)
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.max_new_development_candidates, 8)
        self.assertEqual(EXPANDED_SINGLE_PLAN_PROFILE.total_scene_limit, 6)
        self.assertEqual(EXPANDED_PROFILE.name, "expanded_multi_plan_v1")
        self.assertEqual(EXPANDED_PROFILE.max_plan_variants, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
