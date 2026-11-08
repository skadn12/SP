from __future__ import annotations

import unittest

from .scene_continuation import (
    build_scene_continuation_prompt,
    build_scene_continuation_request,
    materialize_scene_continuation_output,
    merge_scene_continuations,
)


def _fixture():
    plan = {"scenes": [
        {"memory_event_id": "ND01", "participants": ["A", "B"],
         "before_state": "before", "required_transition": "do X",
         "after_state": "after", "new_development": "do X",
         "durable_outcome": "after", "interaction_beats": ["beat"]},
        {"memory_event_id": "ND02", "participants": ["A", "B"],
         "before_state": "before", "required_transition": "do Y",
         "after_state": "after", "new_development": "do Y",
         "durable_outcome": "after", "interaction_beats": ["beat"]},
        {"memory_event_id": "ND03", "participants": ["A", "B"],
         "before_state": "before", "required_transition": "do Z",
         "after_state": "after", "new_development": "do Z",
         "durable_outcome": "after", "interaction_beats": ["beat"]},
    ]}
    child = {"round_index": 2, "scenes": [{
        "scene_ref": f"R02:S{index:02d}", "memory_event_id": f"ND{index:02d}",
        "dialogue_turns": [{"turn_ref": f"R02:S{index:02d}:T{n:03d}",
                            "speaker": "A", "text": f"old {n}"}
                           for n in range(1, 3)],
    } for index in range(1, 4)]}
    audit = {"scenes": [
        {"participant_adherence": "pass", "development_status": "realized",
         "durable_outcome_status": "realized", "reason": "done"},
        {"participant_adherence": "pass", "development_status": "realized",
         "durable_outcome_status": "partial", "reason": "needs confirmation"},
        {"participant_adherence": "fail", "development_status": "not_realized",
         "durable_outcome_status": "not_realized", "reason": "drift"},
    ]}
    return plan, child, audit


class SceneContinuationTest(unittest.TestCase):
    def test_only_partial_participant_safe_scene_is_eligible(self):
        request = build_scene_continuation_request(
            plan=_fixture()[0], child=_fixture()[1], adherence_audit=_fixture()[2])
        self.assertEqual(request["eligible_scene_count"], 1)
        self.assertEqual(request["eligible_scenes"][0]["scene_ref"], "R02:S02")
        self.assertEqual(
            {row["scene_ref"] for row in request["excluded_scenes"]},
            {"R02:S01", "R02:S03"})

    def test_materialization_generates_refs_and_rejects_model_ids(self):
        plan, child, audit = _fixture()
        request = build_scene_continuation_request(
            plan=plan, child=child, adherence_audit=audit)
        raw = {"continuations": [{"dialogue_turns": [
            {"speaker": "A", "text": "A makes the follow-up call."},
            {"speaker": "B", "text": "B confirms the result."},
        ]}]}
        materialized = materialize_scene_continuation_output(
            raw, request=request, child=child)
        self.assertTrue(materialized["valid"], materialized["errors"])
        self.assertEqual(materialized["continuations"][0]["scene_ref"], "R02:S02")
        self.assertEqual(
            [row["turn_ref"] for row in materialized["continuations"][0]["dialogue_turns"]],
            ["R02:S02:T003", "R02:S02:T004"])

    def test_prompt_contains_only_eligible_scenes_and_no_parent_or_memory(self):
        plan, child, audit = _fixture()
        request = build_scene_continuation_request(
            plan=plan, child=child, adherence_audit=audit)
        prompt = build_scene_continuation_prompt(request)
        self.assertEqual(prompt["eligible_scene_count"], 1)
        self.assertFalse(prompt["full_parent_included"])
        self.assertFalse(prompt["memory_included"])
        self.assertIn("do Y", prompt["prompt"])
        self.assertNotIn("do X", prompt["prompt"])
        self.assertNotIn("do Z", prompt["prompt"])

    def test_merge_preserves_original_and_appends_only_selected_scene(self):
        plan, child, audit = _fixture()
        request = build_scene_continuation_request(
            plan=plan, child=child, adherence_audit=audit)
        materialized = materialize_scene_continuation_output(
            {"continuations": [{"dialogue_turns": [
                {"speaker": "A", "text": "A makes the follow-up call."},
                {"speaker": "B", "text": "B confirms the follow-up."},
            ]}]}, request=request, child=child)
        merged = merge_scene_continuations(child, materialized)
        self.assertEqual(len(merged["scenes"][0]["dialogue_turns"]), 2)
        self.assertEqual(len(merged["scenes"][1]["dialogue_turns"]), 4)
        self.assertEqual(len(merged["scenes"][2]["dialogue_turns"]), 2)
        self.assertEqual(merged["scenes"][1]["dialogue_turns"][-1]["turn_ref"],
                         "R02:S02:T004")

    def test_exact_repeated_output_block_is_collapsed_mechanically(self):
        plan, child, audit = _fixture()
        audit["scenes"][2] = {
            "participant_adherence": "pass",
            "development_status": "partial",
            "durable_outcome_status": "partial",
            "reason": "needs completion",
        }
        request = build_scene_continuation_request(
            plan=plan, child=child, adherence_audit=audit)
        first = {"dialogue_turns": [
            {"speaker": "A", "text": "A completes Y."},
            {"speaker": "B", "text": "B confirms Y."}]}
        second = {"dialogue_turns": [
            {"speaker": "B", "text": "B completes Z."}]}
        materialized = materialize_scene_continuation_output(
            {"continuations": [first, second, first, second]},
            request=request, child=child)
        self.assertTrue(materialized["valid"], materialized["errors"])
        self.assertEqual(len(materialized["continuations"]), 2)
        self.assertEqual(materialized["normalizations"],
                         ["collapsed_exact_repeated_continuation_block"])

    def test_extra_scene_local_speaker_is_allowed(self):
        plan, child, audit = _fixture()
        request = build_scene_continuation_request(
            plan=plan, child=child, adherence_audit=audit)
        materialized = materialize_scene_continuation_output(
            {"continuations": [{"dialogue_turns": [
                {"speaker": "B", "text": "B completes the follow-up."},
                {"speaker": "Clerk", "text": "The clerk supplies the receipt."},
            ]}]}, request=request, child=child)
        self.assertTrue(materialized["valid"], materialized["errors"])

    def test_missing_required_participant_is_rejected(self):
        plan, child, audit = _fixture()
        request = build_scene_continuation_request(
            plan=plan, child=child, adherence_audit=audit)
        materialized = materialize_scene_continuation_output(
            {"continuations": [{"dialogue_turns": [
                {"speaker": "Passerby", "text": "A passerby supplies context."},
            ]}]}, request=request, child=child)
        self.assertFalse(materialized["valid"])
        self.assertIn(
            "continuation_1_required_participant_missing:B",
            materialized["errors"])


if __name__ == "__main__":
    unittest.main()
