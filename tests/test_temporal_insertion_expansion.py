import copy
import unittest

from scripts.current_method.restart_n5_spell.temporal_insertion_expansion import (
    TEMPORAL_GRAPH_CONTRACT_VERSION,
    build_temporal_outline_prompt,
    build_temporal_realizer_prompt,
    compile_temporal_scene_contracts,
    merge_temporal_insertions,
    normalize_temporal_outline_plan,
    validate_temporal_outline_plan,
)


class TemporalInsertionExpansionTests(unittest.TestCase):
    def setUp(self):
        self.turns = [
            {"turn_id": f"D{i}:1", "session_id": f"D{i}",
             "speaker": "John" if i % 2 else "Maria",
             "text": f"Original fact {i}."}
            for i in range(1, 5)
        ]
        assignments = [
            (1, "D1", "D2", "T0001"),
            (1, "D1", "D2", "T0001"),
            (2, "D2", "D3", "T0002"),
            (2, "D2", "D3", "T0002"),
            (2, "D2", "D3", "T0002"),
            (3, "D3", "D4", "T0003"),
            (3, "D3", "D4", "T0003"),
            (4, "D4", "END", "T0004"),
            (4, "D4", "END", "T0004"),
            (4, "D4", "END", "T0004"),
        ]
        self.raw = {
            "title": "Inserted episodes",
            "expansion_objective": "Add missing episodes without changing originals.",
            "allowed_new_characters": [],
            "event_arcs": [
                {"summary": f"Arc {arc}",
                 "stages": [f"Arc {arc} new stage {stage}"
                            for stage in range(1, 5)]}
                for arc in range(1, 5)
            ],
            "scenes": [
                {"purpose": f"Scene {index}",
                 "participants": ["John", "Maria"],
                 "location": "library",
                 "arc_index": arc,
                 "anchor_aliases": [alias],
                 "handoff_requirement": "Reach the next concrete step."}
                for index, (arc, after, before, alias) in
                enumerate(assignments, 1)
            ],
        }

    def test_valid_plan_accepts_and_normalizes(self):
        result = validate_temporal_outline_plan(
            self.raw, original_turns=self.turns)
        self.assertEqual("temporal_outline_accept", result["decision"])
        outline = normalize_temporal_outline_plan(
            self.raw, original_turns=self.turns, round_index=1)
        self.assertEqual(10, len(outline["scene_plan"]))
        self.assertFalse(outline["temporal_contract"][
            "fixed_positional_stage_schedule_used"])

    def test_unknown_anchor_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["scenes"][0]["anchor_aliases"] = ["T9999"]
        result = validate_temporal_outline_plan(
            raw, original_turns=self.turns)
        self.assertIn("scene_anchors", result["reasons"])

    def test_insertion_gap_is_derived_from_latest_anchor(self):
        raw = copy.deepcopy(self.raw)
        raw["scenes"][0]["anchor_aliases"] = ["T0001", "T0003"]
        outline = normalize_temporal_outline_plan(
            raw, original_turns=self.turns, round_index=1)
        selected = next(scene for scene in outline["scene_plan"]
                        if scene["api_scene_index"] == 1)
        self.assertEqual("D3", selected["insertion_after_session_id"])
        self.assertEqual("D4", selected["insertion_before_session_id"])

    def test_arc_with_invalid_scene_count_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        for scene in raw["scenes"]:
            if scene["arc_index"] == 4:
                scene["arc_index"] = 3
        result = validate_temporal_outline_plan(
            raw, original_turns=self.turns)
        self.assertIn("arc_scene_count_invalid", result["reasons"])

    def test_scene_array_is_stably_sorted_by_anchor_time(self):
        raw = copy.deepcopy(self.raw)
        raw["scenes"][0], raw["scenes"][-1] = (
            raw["scenes"][-1], raw["scenes"][0])
        result = validate_temporal_outline_plan(raw, original_turns=self.turns)
        self.assertEqual("temporal_outline_accept", result["decision"])
        outline = normalize_temporal_outline_plan(
            raw, original_turns=self.turns, round_index=1)
        self.assertEqual("D1", outline["scene_plan"][0][
            "insertion_after_session_id"])

    def test_contract_maps_stages_within_api_selected_arc(self):
        outline = normalize_temporal_outline_plan(
            self.raw, original_turns=self.turns, round_index=1)
        contracts = compile_temporal_scene_contracts(
            outline, original_turns=self.turns)
        self.assertEqual(TEMPORAL_GRAPH_CONTRACT_VERSION,
                         contracts[0]["contract_version"])
        self.assertEqual(["E1:stage_01", "E1:stage_02"],
                         contracts[0]["assigned_event_stage_ids"])
        self.assertEqual(["E2:stage_01"],
                         contracts[2]["assigned_event_stage_ids"])
        self.assertEqual("D1", contracts[0][
            "insertion_after_session_id"])
        self.assertEqual("D2", contracts[0][
            "insertion_before_session_id"])

    def test_new_character_introduction_uses_sorted_scene_position(self):
        raw = copy.deepcopy(self.raw)
        raw["allowed_new_characters"] = [{"name": "Jean", "role": "advisor"}]
        raw["scenes"][-1]["participants"] = ["John", "Jean"]
        raw["scenes"][-1]["anchor_aliases"] = ["T0001"]
        outline = normalize_temporal_outline_plan(
            raw, original_turns=self.turns, round_index=1)
        declared = outline["allowed_new_characters"][0]
        self.assertEqual("Jean", declared["name"])
        self.assertEqual("S3", declared["introduction_scene"])

    def test_merge_preserves_originals_and_inserts_at_each_gap(self):
        outline = normalize_temporal_outline_plan(
            self.raw, original_turns=self.turns, round_index=1)
        contracts = compile_temporal_scene_contracts(
            outline, original_turns=self.turns)
        sessions = [{
            "graph_scene_id": contract["scene_id"],
            "dialogue_turns": [{"turn_id": f"C{index:04d}",
                                "session_id": f"session_{index:02d}",
                                "speaker": "John",
                                "text": f"Inserted scene {index}."}],
        } for index, contract in enumerate(contracts, 1)]
        merged = merge_temporal_insertions(
            original_turns=self.turns, sessions=sessions,
            contracts=contracts, round_index=1)
        ids = [turn["turn_id"] for turn in merged["turns"]]
        self.assertEqual(["D1:1", "I01C0001", "I01C0002", "D2:1"],
                         ids[:4])
        self.assertEqual("D4:1", ids[-4])
        self.assertEqual(["I01C0008", "I01C0009", "I01C0010"], ids[-3:])
        self.assertTrue(merged["original_order_preserved"])
        self.assertEqual(14, merged["combined_turn_count"])

    def test_prompt_requires_explicit_temporal_gaps(self):
        prompt = build_temporal_outline_prompt(
            original_turns=self.turns, round_index=1)
        self.assertIn("after D1 / before D2", prompt)
        self.assertIn("after D4 / before END", prompt)
        self.assertIn("immediately after the original session containing its latest anchor", prompt)
        self.assertIn("following original session are future consistency", prompt)
        self.assertIn("Never use undeclared groups", prompt)
        self.assertIn("allowed_new_characters", prompt)

    def test_realizer_prompt_marks_following_context_as_future_only(self):
        outline = normalize_temporal_outline_plan(
            self.raw, original_turns=self.turns, round_index=1)
        contract = compile_temporal_scene_contracts(
            outline, original_turns=self.turns)[0]
        prompt = build_temporal_realizer_prompt(
            graph_scene_contract=contract,
            session_index=1,
            narrative_beats=[{"event_id": "E1"}],
            recent_sessions=[],
            continuity_summary={},
            active_plans=[],
            pending_decisions=[],
            superseded_information=[],
            character_knowledge={},
            visibility_boundaries=[],
            unresolved_hooks=[],
            allowed_people=["John", "Maria"],
            session_location="library",
            allowed_event_arc_ids=["E1"],
            forbidden_conflicts=[],
            parent_anchors=contract["parent_anchor_evidence"],
            allowed_durable_subjects=["John", "Maria"],
            durable_predicate_whitelist=["plans"],
        )
        self.assertIn("following_original_context_constraint_only", prompt)
        self.assertIn("future text that must remain consistent", prompt)


if __name__ == "__main__":
    unittest.main()
