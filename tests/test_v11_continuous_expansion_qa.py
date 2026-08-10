from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    adapt_expanded_dialogue,
    build_cross_boundary_lineage_pair_registry,
    build_cross_boundary_questioner_prompt,
    build_qa_source_view,
    select_frozen_lineage_pair,
    validate_cross_boundary_core_object,
)


SOURCE = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_34/"
    "expanded_dialogue_round_1.json"
)
PILOT13_EXPANSION = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_13/roots/"
    "cc-train-episode-41625/01_expansion"
)
PILOT13_QA = PILOT13_EXPANSION.parent / "02_questioner"


class ContinuousExpansionQATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expanded = json.loads(SOURCE.read_text(encoding="utf-8"))
        cls.view = build_qa_source_view(cls.expanded)

    def same_speaker_pair(self):
        parent = next(turn for turn in self.view["combined_dialogue"]
                      if self.view["partition_by_turn"][turn["turn_id"]] ==
                      "original_parent")
        child = next(turn for turn in self.view["combined_dialogue"]
                     if self.view["partition_by_turn"][turn["turn_id"]] ==
                     "promoted_child" and turn["speaker"] == parent["speaker"])
        return parent, child

    def test_adapter_preserves_complete_parent_child_dialogue(self) -> None:
        turns, partitions = adapt_expanded_dialogue(self.expanded)
        self.assertEqual(92, len(turns))
        self.assertEqual(60, list(partitions.values()).count("original_parent"))
        self.assertEqual(32, list(partitions.values()).count("promoted_child"))
        self.assertEqual(self.expanded["canonical_turn_refs"],
                         [row["turn_id"] for row in turns])

    def test_prompt_contains_full_dialogue_and_no_world_memory(self) -> None:
        prompt = build_cross_boundary_questioner_prompt(
            root_id=self.expanded["root_id"],
            span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"])
        self.assertEqual(92, prompt.count('"span_id"'))
        self.assertIn('"source_partition":"original_parent"', prompt)
        self.assertIn('"source_partition":"promoted_child"', prompt)
        self.assertIn("Do not use World Memory", prompt)
        self.assertIn("Every factual clause", prompt)
        self.assertIn("submitted span must explicitly name that speaker", prompt)
        self.assertIn("Preserve modality exactly", prompt)
        self.assertIn("exact speaker shown", prompt)
        self.assertIn("Evidence speakers may differ", prompt)
        self.assertNotIn("continued tail session", prompt)
        self.assertIn("do not concatenate unrelated facts", prompt)
        self.assertIn("program will not synthesize or overwrite it", prompt)
        self.assertIn('"correct_answer_text"', prompt)
        self.assertIn("2 to 6 exact evidence spans", prompt)
        self.assertIn("program will not choose evidence for you", prompt)
        self.assertNotIn("ALLOWED LINEAGE PAIRS", prompt)
        self.assertNotIn("FROZEN SUPPORT PAIR", prompt)

    def test_cross_boundary_fixture_passes(self) -> None:
        parent, child = self.same_speaker_pair()
        raw = {"question": "What changed across the two parts?",
               "correct_answer_text": (
                   f"Initially: {parent['text'].strip()} "
                   f"Later: {child['text'].strip()}"),
               "answer_support_span_ids": [f"{parent['turn_id']}#S0001",
                                           f"{child['turn_id']}#S0001"]}
        result = validate_cross_boundary_core_object(
            raw, span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"])
        self.assertEqual("cross_boundary_qa_accept", result["decision"])

    def test_parent_only_support_is_valid_and_not_a_quality_failure(self) -> None:
        parent = [turn for turn in self.view["combined_dialogue"]
                  if self.view["partition_by_turn"][turn["turn_id"]] ==
                  "original_parent"][:2]
        raw = {"question": "What happened?", "correct_answer_text": "A fact.",
               "answer_support_span_ids": [f"{row['turn_id']}#S0001"
                                           for row in parent]}
        result = validate_cross_boundary_core_object(
            raw, span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"])
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertEqual("parent_only", result["evidence_scope"])
        self.assertFalse(result["scope_is_quality_gate"])

    def test_child_only_and_multi_child_scopes_are_valid(self) -> None:
        child = [turn for turn in self.view["combined_dialogue"]
                 if self.view["partition_by_turn"][turn["turn_id"]] ==
                 "promoted_child"]
        by_session = {}
        for row in child:
            by_session.setdefault(row["session_id"], []).append(row)
        same_session = next(rows for rows in by_session.values()
                            if len(rows) >= 2)[:2]
        child_only = validate_cross_boundary_core_object(
            {"question": "What was decided?", "correct_answer_text": "A fact.",
             "answer_support_span_ids": [f"{row['turn_id']}#S0001"
                                         for row in same_session]},
            span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"])
        self.assertEqual("cross_boundary_qa_accept", child_only["decision"])
        self.assertEqual("child_only", child_only["evidence_scope"])
        different_sessions = [rows[0] for rows in by_session.values()][:2]
        multi_child = validate_cross_boundary_core_object(
            {"question": "What developments occurred?",
             "correct_answer_text": "Two connected developments occurred.",
             "answer_support_span_ids": [f"{row['turn_id']}#S0001"
                                         for row in different_sessions]},
            span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"])
        self.assertEqual("cross_boundary_qa_accept", multi_child["decision"])
        self.assertEqual("multi_child", multi_child["evidence_scope"])

    def test_non_tail_parent_plus_child_is_allowed(self) -> None:
        parent = next(turn for turn in self.view["combined_dialogue"]
                      if self.view["partition_by_turn"][turn["turn_id"]] ==
                      "original_parent")
        child = next(turn for turn in self.view["combined_dialogue"]
                     if self.view["partition_by_turn"][turn["turn_id"]] ==
                     "promoted_child")
        raw = {"question": "What changed?",
               "correct_answer_text": "The situation changed over time.",
               "answer_support_span_ids": [f"{parent['turn_id']}#S0001",
                                           f"{child['turn_id']}#S0001"]}
        result = validate_cross_boundary_core_object(
            raw, span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"])
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertFalse(result["gold_materialized_locally"])
        self.assertTrue(result["questioner_selected_evidence"])

    def test_model_answer_is_preserved_for_semantic_audit(self) -> None:
        parent, child = self.same_speaker_pair()
        raw = {"question": "What changed?",
               "correct_answer_text": "The speaker felt better later.",
               "answer_support_span_ids": [f"{parent['turn_id']}#S0001",
                                           f"{child['turn_id']}#S0001"]}
        result = validate_cross_boundary_core_object(
            raw, span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"])
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertEqual("The speaker felt better later.",
                         result["core_qa"]["correct_answer_text"])
        self.assertFalse(result["gold_materialized_locally"])

    def test_unknown_support_ref_is_schema_invalid(self) -> None:
        raw = {"question": "What happened?", "correct_answer_text": "A fact.",
               "answer_support_span_ids": ["unknown#S0001", "also_unknown#S0001"]}
        result = validate_cross_boundary_core_object(
            raw, span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"])
        self.assertEqual("cross_boundary_qa_schema_invalid", result["decision"])

    def test_excluded_prior_question_is_contract_rejected(self) -> None:
        parent = next(turn for turn in self.view["combined_dialogue"]
                      if self.view["partition_by_turn"][turn["turn_id"]] ==
                      "original_parent")
        child = next(turn for turn in self.view["combined_dialogue"]
                     if self.view["partition_by_turn"][turn["turn_id"]] ==
                     "promoted_child")
        question = "How did the earlier event affect the later decision?"
        result = validate_cross_boundary_core_object(
            {"question": question, "correct_answer_text": "A changed.",
             "answer_support_span_ids": [
                f"{parent['turn_id']}#S0001", f"{child['turn_id']}#S0001"]},
            span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"],
            excluded_questions=(question,),
        )
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        self.assertIn("cross_boundary_qa_repeats_excluded_question",
                      result["reasons"])

    def test_lineage_pair_registry_rejects_pilot13_unanchored_pair(self) -> None:
        expanded = json.loads((
            PILOT13_EXPANSION / "expanded_dialogue_round_1.json"
        ).read_text(encoding="utf-8"))
        view = build_qa_source_view(expanded)
        registry = build_cross_boundary_lineage_pair_registry(
            round_validator_input=json.loads((
                PILOT13_EXPANSION / "round_validator_input.json"
            ).read_text(encoding="utf-8")),
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
        )
        record = next((PILOT13_QA / "raw/question_answer_writer").glob("*.json"))
        raw = json.loads(json.loads(record.read_text(encoding="utf-8"))[
            "raw_response"])
        result = validate_cross_boundary_core_object(
            raw,
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=registry,
        )
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        self.assertIn("cross_boundary_qa_pair_not_anchor_linked",
                      result["reasons"])

    def test_mark_only_policy_accepts_and_labels_unanchored_pair(self) -> None:
        expanded = json.loads((
            PILOT13_EXPANSION / "expanded_dialogue_round_1.json"
        ).read_text(encoding="utf-8"))
        view = build_qa_source_view(expanded)
        registry = build_cross_boundary_lineage_pair_registry(
            round_validator_input=json.loads((
                PILOT13_EXPANSION / "round_validator_input.json"
            ).read_text(encoding="utf-8")),
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
        )
        record = next((PILOT13_QA / "raw/question_answer_writer").glob("*.json"))
        raw = json.loads(json.loads(record.read_text(encoding="utf-8"))[
            "raw_response"])
        result = validate_cross_boundary_core_object(
            raw,
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=registry,
            evidence_mismatch_policy="mark_only",
        )
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertTrue(result["evidence_mismatch"])
        self.assertEqual("mismatch_marked", result["evidence_alignment_status"])
        self.assertIn("cross_boundary_qa_pair_not_anchor_linked",
                      result["evidence_mismatch_reasons"])

    def test_mark_only_does_not_accept_unknown_evidence_refs(self) -> None:
        result = validate_cross_boundary_core_object(
            {"question": "What changed?", "answer_support_span_ids": [
                "unknown-parent#S0001", "unknown-child#S0001"]},
            span_catalog=self.view["span_catalog"],
            partition_by_turn=self.view["partition_by_turn"],
            evidence_mismatch_policy="mark_only",
        )
        self.assertEqual("cross_boundary_qa_schema_invalid", result["decision"])

    def test_lineage_pair_registry_allows_declared_anchor_pair(self) -> None:
        expanded = json.loads((
            PILOT13_EXPANSION / "expanded_dialogue_round_1.json"
        ).read_text(encoding="utf-8"))
        view = build_qa_source_view(expanded)
        registry = build_cross_boundary_lineage_pair_registry(
            round_validator_input=json.loads((
                PILOT13_EXPANSION / "round_validator_input.json"
            ).read_text(encoding="utf-8")),
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
        )
        self.assertEqual(5, len(registry["rows"]))
        row = registry["rows"][0]
        pair = row["eligible_same_speaker_pairs"][0]
        parent_id = pair["parent_span_id"]
        child_id = pair["child_span_id"]
        raw = {
            "question": "What was said initially, and what was said later?",
            "answer_support_span_ids": [parent_id, child_id],
        }
        result = validate_cross_boundary_core_object(
            raw,
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=registry,
        )
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        prompt = build_cross_boundary_questioner_prompt(
            root_id=expanded["root_id"],
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=registry,
        )
        self.assertIn("ALLOWED LINEAGE PAIRS", prompt)
        self.assertIn(row["source_anchor_id"], prompt)

    def test_candidate_pair_selection_rotates_across_anchor_scenes(self) -> None:
        expanded = json.loads((
            PILOT13_EXPANSION / "expanded_dialogue_round_1.json"
        ).read_text(encoding="utf-8"))
        view = build_qa_source_view(expanded)
        registry = build_cross_boundary_lineage_pair_registry(
            round_validator_input=json.loads((
                PILOT13_EXPANSION / "round_validator_input.json"
            ).read_text(encoding="utf-8")),
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
        )
        first = select_frozen_lineage_pair(registry, candidate_index=1)
        second = select_frozen_lineage_pair(registry, candidate_index=2)
        self.assertNotEqual(first["scene_ref"], second["scene_ref"])
        self.assertNotEqual(
            (first["parent_span_id"], first["child_span_id"]),
            (second["parent_span_id"], second["child_span_id"]),
        )

    def test_frozen_pair_is_a_hard_questioner_boundary(self) -> None:
        expanded = json.loads((
            PILOT13_EXPANSION / "expanded_dialogue_round_1.json"
        ).read_text(encoding="utf-8"))
        view = build_qa_source_view(expanded)
        registry = build_cross_boundary_lineage_pair_registry(
            round_validator_input=json.loads((
                PILOT13_EXPANSION / "round_validator_input.json"
            ).read_text(encoding="utf-8")),
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
        )
        frozen = select_frozen_lineage_pair(registry, candidate_index=1)
        other = select_frozen_lineage_pair(registry, candidate_index=2)
        result = validate_cross_boundary_core_object(
            {"question": "What changed across the two moments?",
             "answer_support_span_ids": [other["parent_span_id"],
                                         other["child_span_id"]]},
            span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=registry,
            frozen_support_pair=frozen,
            evidence_mismatch_policy="mark_only",
        )
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        self.assertIn("cross_boundary_qa_frozen_support_pair_mismatch",
                      result["reasons"])
        prompt = build_cross_boundary_questioner_prompt(
            root_id=expanded["root_id"], span_catalog=view["span_catalog"],
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=registry, frozen_support_pair=frozen)
        self.assertIn("FROZEN SUPPORT PAIR", prompt)
        self.assertIn(frozen["parent_span_id"], prompt)


if __name__ == "__main__":
    unittest.main()
