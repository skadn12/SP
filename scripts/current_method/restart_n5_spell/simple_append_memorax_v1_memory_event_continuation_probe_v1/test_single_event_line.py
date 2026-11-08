from __future__ import annotations

import copy
import unittest

from .single_event_line import (
    build_gate_compatible_plan,
    build_single_event_line_prompt,
    materialize_single_event_line_plan,
    validate_materialized_single_event_line,
    validate_single_event_gate_plan,
    validate_single_event_line_plan,
)
from .contracts import (
    build_semantic_gate_context,
    materialize_planner_output,
    validate_realizer_plan_isolation,
    validate_semantic_gate_output,
    normalize_realizer_output,
)
from .promotion import build_promotable_child


def _track():
    return {
        "verified_track_ref": "VT01",
        "memory_event_ref": "ME01",
        "memory_event_key": "soccer practice",
        "memory_event_display": "Soccer practice",
        "verified_current_state": "A has decided to resume practice tomorrow.",
        "memory_event_content": ["A decided to resume practice."],
        "evidence_turn_refs": ["root:P01:T003"],
    }


def _parent():
    return {"root_id": "root", "dialogue": {
        "dialogue_hash": "a" * 64,
        "turns": [{
            "turn_id": "root:P01:T001", "session_id": "root:P01",
            "speaker": "A", "text": "A decided to resume practice tomorrow.",
        }, {
            "turn_id": "root:P01:T002", "session_id": "root:P01",
            "speaker": "B", "text": "B acknowledged the decision.",
        }],
    }}


def _step(index: int):
    return {
        "goal": f"Advance the soccer thread at step {index}.",
        "required_participants": ["A", "B"],
        "transition_from_previous": (
            "A turns the earlier decision into a concrete next action."
            if index == 1 else
            f"A responds to the concrete result of step {index - 1}."),
        "required_transition": f"A completes a new soccer development {index}.",
        "after_state": f"A has reached the new soccer state {index}.",
        "actor_bindings": [
            {"actor_ref": "A", "binding": "A carries out the planned step."},
            {"actor_ref": "B", "binding": "B responds to A's step."},
        ],
        "interaction_beats": [
            f"A describes the concrete soccer step {index}.",
            f"B responds to the result of soccer step {index}.",
        ],
        "new_people": [],
    }


class SingleEventLineContractTest(unittest.TestCase):
    def test_prompt_contains_only_one_target_event(self):
        prompt = build_single_event_line_prompt(
            target_track=_track(),
            speaker_registry={"speakers": [{"speaker_ref": "SP01",
                                             "canonical_name": "A"}]})
        self.assertEqual(prompt["target_event_ref"], "ME01")
        self.assertIn("one ordered continuation", prompt["prompt"])
        self.assertIn("required refs",
                      prompt["prompt"])
        self.assertEqual(prompt["prompt_version"],
                         "single_event_line_planner_prompt_v3_required_participants")
        self.assertNotIn("selected_continuations", prompt["prompt"])

    def test_valid_three_step_plan(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        result = validate_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        self.assertTrue(result["valid"], result["errors"])

    def test_plan_requires_exactly_three_steps(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2)]}
        result = validate_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        self.assertFalse(result["valid"])
        self.assertIn("step_count:2/3", result["errors"])

    def test_materializer_creates_one_event_chain(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        plan = materialize_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        result = validate_materialized_single_event_line(plan)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual([scene["memory_event_ref"] for scene in plan["scenes"]],
                         ["ME01", "ME01", "ME01"])
        self.assertEqual(plan["scenes"][1]["depends_on_scene_refs"],
                         ["R01:S01"])
        self.assertEqual(plan["scenes"][2]["depends_on_scene_refs"],
                         ["R01:S02"])
        self.assertEqual(plan["scenes"][1]["before_state"],
                         plan["scenes"][0]["after_state"])
        self.assertEqual(plan["scenes"][0]["actor_bindings"][0]["actor"], "A")
        self.assertNotIn("actor_ref", plan["scenes"][0]["actor_bindings"][0])

    def test_extra_speaker_is_allowed_when_required_people_appear(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        plan = materialize_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        raw = {"scenes": [{"dialogue_turns": [
            {"speaker": ["A", "B", "Passerby"][index % 3],
             "text": f"scene {scene} turn {index}"}
            for index in range(9)]} for scene in range(1, 4)]}
        result = normalize_realizer_output(raw, plan=plan, parent=_parent())
        self.assertTrue(result["valid"], result["errors"])
        self.assertIn("Passerby",
                      result["speaker_coverage"][0]["additional_speakers"])

    def test_missing_required_speaker_is_still_rejected(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        plan = materialize_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        plan["scenes"][0]["required_participants"] = ["A", "B"]
        raw = {"scenes": [{"dialogue_turns": [
            {"speaker": "A", "text": f"only A scene {scene} turn {index}"}
            for index in range(9)]} for scene in range(1, 4)]}
        result = normalize_realizer_output(raw, plan=plan, parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("R01:S01:required_participant_missing:B",
                      result["errors"])

    def test_gate_adapter_uses_unique_step_ids_and_one_event_ref(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        ordered = materialize_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        plan = build_gate_compatible_plan(ordered)
        validation = validate_single_event_gate_plan(plan)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual([row["memory_event_id"] for row in plan["scenes"]],
                         ["ME01:STEP01", "ME01:STEP02", "ME01:STEP03"])
        self.assertEqual([row["memory_event_ref"] for row in plan["scenes"]],
                         ["ME01", "ME01", "ME01"])

    def test_zero_api_gate_realizer_promotion_chain(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        ordered = materialize_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        plan = build_gate_compatible_plan(ordered)
        candidates = [{
            "memory_event_id": "ME01",
            "memory_event_key": "soccer practice",
            "memory_event_display": "Soccer practice",
            "memory_event_content": ["A decided to resume practice."],
        }]
        context = build_semantic_gate_context(
            parent=_parent(), candidates=candidates, plan=plan,
            memory_only_fast=False)
        refs = context["expected_candidate_refs"]
        self.assertEqual(refs, [
            "ME01:STEP01:C01", "ME01:STEP02:C01", "ME01:STEP03:C01"])
        raw_gate = {"candidate_reviews": [{
            "candidate_ref": ref,
            "decision": "pass",
            "duplicate_with_parent": False,
            "evidence_turn_refs": [],
            "parent_relation": "NONE",
        } for ref in refs]}
        gate = validate_semantic_gate_output(
            raw_gate, plan=plan, parent=_parent())
        self.assertTrue(gate["valid"], gate["errors"])
        materialized = materialize_planner_output(
            plan, passed_candidate_refs=gate["passed_candidate_refs"])
        isolation = validate_realizer_plan_isolation(
            raw_plan=plan, materialized_plan=materialized,
            passed_candidate_refs=gate["passed_candidate_refs"])
        self.assertTrue(isolation["valid"], isolation["errors"])

        child = {"schema_version": "child_v1", "root_id": "root",
                 "round_index": 1, "scenes": [{
            "scene_ref": f"R01:S{index:02d}",
            "session_ref": f"root:r01:s{index:02d}",
            "memory_event_id": f"ME01:STEP{index:02d}",
            "dialogue_turns": [{
                "turn_ref": f"R01:S{index:02d}:T001", "speaker": "A",
                "text": f"A completes a concrete soccer development {index} now.",
            }],
        } for index in range(1, 4)]}
        audit = {"scenes": [{
            "scene_ref": f"R01:S{index:02d}",
            "scene_verdict": "FAIL" if index == 1 else "PASS",
        } for index in range(1, 4)]}
        promotion = build_promotable_child(
            parent=_parent(), plan=materialized, child=child,
            adherence_audit=audit)
        self.assertTrue(promotion["valid"], promotion["errors"])
        self.assertEqual(promotion["audit"]["promoted_scene_refs"], [])

    def test_materialized_plan_rejects_parallel_edges(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        plan = materialize_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        bad = copy.deepcopy(plan)
        bad["scenes"][2]["depends_on_scene_refs"] = []
        result = validate_materialized_single_event_line(bad)
        self.assertFalse(result["valid"])
        self.assertIn("scene_3_dependency_chain", result["errors"])

    def test_unknown_speaker_is_rejected(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        value["steps"][1]["required_participants"] = ["A", "C"]
        result = validate_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        self.assertFalse(result["valid"])
        self.assertTrue(any("unknown_participants" in error
                            for error in result["errors"]))

    def test_goal_may_mention_auxiliary_known_participant(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        value["steps"][0]["required_participants"] = ["A"]
        value["steps"][0]["actor_bindings"] = [
            {"actor_ref": "A", "binding": "A shares the update."}]
        value["steps"][0]["goal"] = "A shares the result with B."
        value["steps"][0]["interaction_beats"] = [
            "A explains the completed practice step."]
        result = validate_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        self.assertTrue(result["valid"], result["errors"])

    def test_beat_may_mention_auxiliary_known_participant(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        value["steps"][0]["required_participants"] = ["A"]
        value["steps"][0]["actor_bindings"] = [
            {"actor_ref": "A", "binding": "A shares the update."}]
        value["steps"][0]["goal"] = "A describes the practice result."
        value["steps"][0]["interaction_beats"] = [
            "A explains the result.", "B reacts to A's explanation."]
        result = validate_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        self.assertTrue(result["valid"], result["errors"])

    def test_short_canonical_name_does_not_match_inside_other_words(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        value["steps"][0]["required_participants"] = ["A"]
        value["steps"][0]["actor_bindings"] = [
            {"actor_ref": "A", "binding": "A owns the action."}]
        value["steps"][0]["goal"] = "A practices balance outdoors."
        value["steps"][0]["interaction_beats"] = [
            "A describes balance and stability."]
        result = validate_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        self.assertTrue(result["valid"], result["errors"])

    def test_duplicate_transition_is_rejected(self):
        value = {"round_goal": "Continue soccer practice.",
                 "steps": [_step(1), _step(2), _step(3)]}
        value["steps"][2]["required_transition"] = \
            value["steps"][1]["required_transition"]
        result = validate_single_event_line_plan(
            value, target_track=_track(), parent_speakers=["A", "B"])
        self.assertFalse(result["valid"])
        self.assertIn("step_3_duplicate_transition", result["errors"])


if __name__ == "__main__":
    unittest.main()
