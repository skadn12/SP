"""Regression tests for the append-only Planner feedback adapter."""

from __future__ import annotations

import unittest

from .prompts import build_planner_prompt


def _parent():
    return {"root_id": "fixture", "dialogue": {"turns": [
        {"turn_id": "T01", "speaker": "Taylor", "text": "We met yesterday."},
        {"turn_id": "T02", "speaker": "Casey", "text": "I remember."},
    ]}}


def _kwargs():
    return dict(
        parent=_parent(), memory_overview={}, candidates=[], verified_tracks=[],
        memory_only_fast=True, event_bucket_only=True,
        memory_source_mode="event_bucket_only",
        speaker_registry={"speakers": [
            {"speaker_ref": "SP01", "canonical_name": "Taylor"},
            {"speaker_ref": "SP02", "canonical_name": "Casey"},
        ]}, strict_speaker_refs=True, include_authoritative_transcript=True,
    )


class PlannerFeedbackAdapterTest(unittest.TestCase):
    def test_none_and_empty_feedback_are_historical_defaults(self):
        baseline = build_planner_prompt(**_kwargs())
        self.assertEqual(baseline, build_planner_prompt(**_kwargs(),
                                                        feedback_guidance=None))
        self.assertEqual(baseline, build_planner_prompt(**_kwargs(),
                                                        feedback_guidance=""))

    def test_feedback_is_appended_without_changing_the_base_contract(self):
        baseline = build_planner_prompt(**_kwargs())
        guidance = "FROZEN FEEDBACK GUIDANCE\nPrefer a concrete state change."
        with_feedback = build_planner_prompt(**_kwargs(),
                                              feedback_guidance=guidance)
        self.assertEqual(with_feedback["prompt"],
                         baseline["prompt"] + "\n\n" + guidance)
        for key in ("prompt_version", "candidate_count", "scene_quota",
                    "max_continuations", "max_new_development_candidates",
                    "max_plan_variants"):
            self.assertEqual(with_feedback[key], baseline[key])


if __name__ == "__main__":
    unittest.main()
