import unittest

from .scene_replacement import (
    build_scene_replacement_prompt,
    build_scene_replacement_request,
    materialize_replacement_plan,
    materialize_scene_replacement_output,
    merge_scene_replacement,
)


def _candidate(event_id, *, goal="replacement goal"):
    return {
        "memory_event_id": event_id,
        "memory_event_ref": event_id,
        "memory_event_key": event_id,
        "development_mode": "continuation",
        "participants": ["A", "B"],
        "goal": goal,
        "before_state": "old state",
        "required_transition": "new development",
        "after_state": "durable outcome",
        "durable_outcome": "durable outcome",
        "actor_bindings": [{"actor": "A", "binding": "A acts"}],
        "interaction_beats": ["A proposes the new step."],
    }


def _raw_scene(event_id, candidates):
    return {
        "memory_event_id": event_id,
        "memory_event_ref": event_id,
        "development_mode": "continuation",
        "participants": ["A", "B"],
        "next_step_candidates": candidates,
    }


class SceneReplacementTests(unittest.TestCase):
    def test_selects_first_unused_gate_passed_candidate_from_other_event(self):
        raw_plan = {"candidate_pool": [
            _raw_scene("ME01", [_candidate("ME01")]),
            _raw_scene("ME02", [_candidate("ME02")]),
            _raw_scene("ND01", [_candidate("ND01")]),
        ]}
        plan = {"scenes": [
            {"scene_ref": "R01:S01", "memory_event_id": "ME01",
             "selected_candidate_ref": "ME01:C01"},
            {"scene_ref": "R01:S02", "memory_event_id": "ME02",
             "selected_candidate_ref": "ME02:C01"},
            {"scene_ref": "R01:S03", "memory_event_id": "OLD",
             "selected_candidate_ref": "OLD:C01"},
        ]}
        child = {"round_index": 1, "scenes": [
            {"scene_ref": "R01:S01", "memory_event_id": "ME01"},
            {"scene_ref": "R01:S02", "memory_event_id": "ME02"},
            {"scene_ref": "R01:S03", "memory_event_id": "OLD"},
        ]}
        audit = {"scenes": [
            {"scene_ref": "R01:S01", "scene_verdict": "PASS"},
            {"scene_ref": "R01:S02", "scene_verdict": "PASS"},
            {"scene_ref": "R01:S03", "scene_verdict": "FAIL"},
        ]}
        request = build_scene_replacement_request(
            raw_plan=raw_plan, materialized_plan=plan, child=child,
            adherence_audit=audit,
            passed_candidate_refs=["ME01:C01", "ME02:C01", "ND01:C01"])
        self.assertEqual(request["failed_scene_refs"], ["R01:S03"])
        self.assertEqual(request["selected_replacement"]["candidate_ref"],
                         "ND01:C01")
        self.assertEqual(request["selected_replacement"]["scene"]["scene_ref"],
                         "R01:S03")

    def test_no_failed_scene_does_not_schedule_replacement(self):
        plan = {"scenes": [{"selected_candidate_ref": "ME01:C01",
                             "memory_event_id": "ME01"}]}
        child = {"round_index": 1, "scenes": [{"scene_ref": "R01:S01",
                                                  "memory_event_id": "ME01"}]}
        request = build_scene_replacement_request(
            raw_plan={"candidate_pool": []}, materialized_plan=plan,
            child=child, adherence_audit={"scenes": [{"scene_verdict": "PASS"}]},
            passed_candidate_refs=["ME01:C01"])
        self.assertIsNone(request["selected_replacement"])

    def test_outline_conflict_blocks_replacement_candidate(self):
        raw_plan = {"candidate_pool": [
            _raw_scene("ME01", [_candidate("ME01")]),
            _raw_scene("ND01", [_candidate("ND01")]),
        ]}
        plan = {"scenes": [
            {"scene_ref": "R01:S01", "memory_event_id": "ME01",
             "selected_candidate_ref": "ME01:C01"},
            {"scene_ref": "R01:S02", "memory_event_id": "OLD",
             "selected_candidate_ref": "OLD:C01"},
        ]}
        child = {"round_index": 1, "scenes": [
            {"scene_ref": "R01:S01", "memory_event_id": "ME01"},
            {"scene_ref": "R01:S02", "memory_event_id": "OLD"},
        ]}
        audit = {"scenes": [
            {"scene_ref": "R01:S01", "scene_verdict": "PASS"},
            {"scene_ref": "R01:S02", "scene_verdict": "FAIL"},
        ]}
        request = build_scene_replacement_request(
            raw_plan=raw_plan, materialized_plan=plan, child=child,
            adherence_audit=audit, passed_candidate_refs=["ME01:C01", "ND01:C01"],
            outline_conflict_edges=[["ME01:C01", "ND01:C01"]])
        self.assertIsNone(request["selected_replacement"])

    def test_materialize_and_merge_replacement_replaces_failed_slot(self):
        request = {
            "selected_replacement": {
                "candidate_ref": "ND01:C01",
                "memory_event_id": "ND01",
                "scene": {"scene_ref": "R01:S03", "participants": ["A", "B"]},
            }
        }
        prompt = build_scene_replacement_prompt(request)
        self.assertIn("exactly one", prompt["prompt"])
        materialized = materialize_scene_replacement_output(
            {"scenes": [{"dialogue_turns": [
                {"speaker": "A", "text": f"A starts the new event {i}."}
                if i % 2 else {"speaker": "B", "text": f"B confirms the result {i}."}
                for i in range(1, 10)
            ]}]}, request=request)
        self.assertTrue(materialized["valid"])
        child = {"round_index": 1, "scenes": [
            {"scene_ref": "R01:S03", "memory_event_id": "OLD",
             "dialogue_turns": [{"turn_ref": "R01:S03:T001",
                                  "speaker": "A", "text": "old rejected"}]}
        ]}
        merged = merge_scene_replacement(child, materialized)
        self.assertEqual(merged["scenes"][0]["memory_event_id"], "ND01")
        self.assertEqual(len(merged["scenes"][0]["dialogue_turns"]), 9)
        self.assertEqual(merged["scenes"][0]["dialogue_turns"][0]["turn_ref"],
                         "R01:S03:T001")
        self.assertNotIn("old rejected", str(merged))

    def test_replacement_plan_updates_only_failed_slot(self):
        plan = {"scenes": [
            {"scene_ref": "R01:S01", "memory_event_id": "ME01"},
            {"scene_ref": "R01:S02", "memory_event_id": "ME02"},
            {"scene_ref": "R01:S03", "memory_event_id": "OLD"},
        ]}
        request = {"selected_replacement": {
            "candidate_ref": "ND01:C01", "memory_event_id": "ND01",
            "scene": {"scene_ref": "R01:S03", "memory_event_id": "ND01",
                      "required_transition": "new"},
        }}
        updated = materialize_replacement_plan(plan, request)
        self.assertEqual([row["memory_event_id"] for row in updated["scenes"]],
                         ["ME01", "ME02", "ND01"])
        self.assertEqual(updated["scenes"][2]["required_transition"], "new")

    def test_replacement_allows_auxiliary_speaker(self):
        request = {"selected_replacement": {
            "candidate_ref": "ND01:C01", "memory_event_id": "ND01",
            "scene": {"scene_ref": "R01:S03",
                      "required_participants": ["A", "B"]},
        }}
        turns = [
            {"speaker": "A" if index % 2 else "B", "text": f"core {index}"}
            for index in range(1, 9)
        ] + [{"speaker": "Clerk", "text": "The clerk supplies a receipt."}]
        materialized = materialize_scene_replacement_output(
            {"scenes": [{"dialogue_turns": turns}]}, request=request)
        self.assertTrue(materialized["valid"], materialized["errors"])

    def test_replacement_plan_uses_target_round_when_plan_has_no_round_field(self):
        plan = {"scenes": [
            {"memory_event_id": "ME01"},
            {"memory_event_id": "ME02"},
            {"memory_event_id": "OLD"},
        ]}
        request = {"selected_replacement": {
            "candidate_ref": "ND01:C01", "memory_event_id": "ND01",
            "scene": {"scene_ref": "R02:S03", "memory_event_id": "ND01",
                      "required_transition": "new"},
        }}
        updated = materialize_replacement_plan(plan, request)
        self.assertEqual(updated["scenes"][2]["memory_event_id"], "ND01")

    def test_replacement_rejects_missing_required_participant(self):
        request = {"selected_replacement": {
            "candidate_ref": "ND01:C01", "memory_event_id": "ND01",
            "scene": {"scene_ref": "R01:S03",
                      "required_participants": ["A", "B"]},
        }}
        turns = [
            {"speaker": "A", "text": f"A continues alone {index}."}
            for index in range(1, 10)
        ]
        materialized = materialize_scene_replacement_output(
            {"scenes": [{"dialogue_turns": turns}]}, request=request)
        self.assertFalse(materialized["valid"])
        self.assertIn(
            "replacement_required_participant_missing:B",
            materialized["errors"])


if __name__ == "__main__":
    unittest.main()
