"""Contract checks for the append-only Realizer feedback adapter."""

from __future__ import annotations

import unittest

from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
    build_realizer_prompt,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.profiles import (
    R1_ALIGNED_BASELINE_PROFILE,
    R1_ALIGNED_FEEDBACK_PROFILE,
)
from .renderers import normalize_parent, render_expansion_realizer


def _dialogue() -> dict:
    return {
        "root_id": "cc-train-episode-test",
        "turns": [
            {"turn_ref": "R01:S01:T001", "speaker": "A", "text": "Hello."},
            {"turn_ref": "R01:S01:T002", "speaker": "B", "text": "Hi."},
        ],
    }


def _record() -> dict:
    return {
        "root_id": "cc-train-episode-test",
        "source_cluster_id": "cluster-test",
        "source_candidate_id": "candidate-test",
        "historical_qa": {
            "question_type": "single_choice",
            "question": "What happened?",
            "options_or_items": {"A": "A", "B": "B"},
            "canonical_gold": ["A"],
            "evidence_refs": ["R01:S01:T001"],
            "source_data_status": {},
        },
        "performance": {"M_clean_policy": {"full": 1, "total": 1}},
    }


def _plan() -> dict:
    return {"scenes": [{
        "memory_event_id": "ME01", "development_mode": "continuation",
        "source_basis": [], "new_people": [],
        "verified_current_state": "A and B met.",
        "goal": "A and B continue their discussion.",
        "required_participants": ["A", "B"],
        "transition_from_previous": "They exchange another detail.",
        "before_state": "They met.", "required_transition": "They exchange a detail.",
        "after_state": "The detail is exchanged.", "actor_bindings": [],
        "interaction_beats": ["They compare notes."],
    }]}


class RealizerFeedbackAdapterTest(unittest.TestCase):
    def test_r1_only_continuation_profiles_omit_historical_dialogue(self) -> None:
        for profile in (R1_ALIGNED_BASELINE_PROFILE, R1_ALIGNED_FEEDBACK_PROFILE):
            with self.subTest(profile=profile.name):
                if profile.feedback_view_version:
                    rendered = render_expansion_realizer(
                        record=_record(), dialogue=_dialogue(), plan=_plan(),
                        style_profile={"register": "informal"},
                        continuity_shape={"turn_count": 2},
                        input_mode=profile.realizer_input_mode)
                else:
                    rendered = build_realizer_prompt(
                        parent=normalize_parent(_dialogue()), plan=_plan(),
                        style_profile={"register": "informal"},
                        continuity_shape={"turn_count": 2},
                        include_full_parent=profile.realizer_input_mode == "full_parent",
                        request_purpose="normal_generation")
                self.assertFalse(rendered["full_parent_included"])
                self.assertNotIn("Hello.", rendered["prompt"])
                self.assertNotIn("Hi.", rendered["prompt"])
                self.assertNotIn("FROZEN PARENT TRANSCRIPT", rendered["prompt"])
                self.assertIn("CONTINUITY SHAPE ONLY", rendered["prompt"])
                self.assertIn("TEXT-FREE STYLE PROFILE", rendered["prompt"])

    def test_feedback_is_append_only_after_historical_builder(self) -> None:
        dialogue = _dialogue()
        parent = normalize_parent(dialogue)
        plan = _plan()
        base = build_realizer_prompt(
            parent=parent, plan=plan, style_profile={"register": "informal"},
            continuity_shape={"turn_count": 2}, include_full_parent=False,
            request_purpose="normal_generation")
        adapted = render_expansion_realizer(
            record=_record(), dialogue=dialogue, plan=plan,
            style_profile={"register": "informal"},
            continuity_shape={"turn_count": 2}, input_mode="text_free")
        self.assertTrue(adapted["prompt"].startswith(base["prompt"] + "\n\n"))
        self.assertIn("【当前角色可见的共享反馈】", adapted["prompt"])
        self.assertIn("【反馈边界】", adapted["prompt"])
        self.assertEqual(adapted["full_parent_included"], False)

    def test_feedback_appendix_does_not_duplicate_parent_text(self) -> None:
        adapted = render_expansion_realizer(
            record=_record(), dialogue=_dialogue(), plan=_plan(),
            style_profile={}, continuity_shape={}, input_mode="text_free")
        self.assertEqual(adapted["prompt"].count("Hello."), 0)
        self.assertEqual(adapted["prompt"].count("FROZEN SCENE PLANS"), 1)

    def test_full_parent_mode_keeps_historical_transcript_boundary(self) -> None:
        dialogue = _dialogue()
        adapted = render_expansion_realizer(
            record=_record(), dialogue=dialogue, plan=_plan(),
            input_mode="full_parent")
        self.assertIn("FROZEN PARENT TRANSCRIPT", adapted["prompt"])
        self.assertIn("A | Hello.", adapted["prompt"])
        self.assertTrue(adapted["full_parent_included"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
