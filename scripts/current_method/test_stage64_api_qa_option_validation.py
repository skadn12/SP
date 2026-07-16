from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from scripts.current_method import native_api_graph_dialogue_expansion as subject


class Stage64ApiQaOptionValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        event_ids = ["event-1", "event-2", "event-3", "event-4"]
        self.current_event_plan = {
            "events": [
                {
                    "event_instance_id": event_id,
                    "entity": "Noah" if index == 2 else "Maya",
                    "slot": "item_beside_lamp",
                    "status": "active" if index == 0 else "outdated",
                }
                for index, event_id in enumerate(event_ids)
            ],
        }
        self.current_alignment = {
            event_id: {"turn_ids": [f"D1:{index}"]}
            for index, event_id in enumerate(event_ids, start=1)
        }
        self.hashes = {
            "graph_snapshot_hash": "graph-hash",
            "dialogue_hash": "dialogue-hash",
            "alignment_hash": "alignment-hash",
        }

    def _proposal(self, texts: list[str], *, option_ids: list[str] | None = None, gold: str = "A") -> dict:
        option_ids = option_ids or list("ABCD")
        options = [
            {
                "id": option_ids[index],
                "text": text,
                "answer_type": "ObjectOrCommonNoun",
                "source_event_instance_id": f"event-{index + 1}",
            }
            for index, text in enumerate(texts)
        ]
        return {
            "question_type": "single_choice",
            "question": "What item did Maya leave beside the lamp?",
            "question_text": "What item did Maya leave beside the lamp?",
            "qa_operation": "relation_lookup",
            "qa_subtype": "object_location",
            "answer_type": "ObjectOrCommonNoun",
            "answer_format": "single_choice_mcq",
            "options": options,
            "gold": [gold],
            "query_plan": {
                "answer_source": "graph_anchor",
                "qa_operation": "relation_lookup",
                "qa_subtype": "object_location",
                "answer_type": "ObjectOrCommonNoun",
                "reasoning_anchor_path": ["event-1", "event-2", "event-3"],
                "required_anchor_refs": ["event-1", "event-2", "event-3"],
                "supporting_anchor_refs": ["event-1", "event-2", "event-3"],
                "gold_anchor_refs": ["event-1"],
                "distractor_anchor_refs": ["event-2", "event-3", "event-4"],
                "gold_answer": texts[option_ids.index(gold)] if gold in option_ids else gold,
                "normalized_gold_answer": (
                    texts[option_ids.index(gold)] if gold in option_ids else gold
                ).casefold(),
            },
        }

    def _build(self, proposal: dict) -> dict:
        return subject._build_stage64_api_qa_from_proposal(
            current_event_plan=self.current_event_plan,
            graph_update={"events": copy.deepcopy(self.current_event_plan["events"])},
            current_alignment=self.current_alignment,
            qa_proposal=proposal,
            hashes=self.hashes,
        )

    def test_rejects_exact_duplicate_option_text(self) -> None:
        proposal = self._proposal(
            ["Red leather notebook", "red leather notebook!", "Ceramic mug", "Brass key"]
        )

        with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "exact duplicate option text"):
            self._build(proposal)

    def test_accepts_semantically_distinct_near_surface_options(self) -> None:
        proposal = self._proposal(
            [
                "Red leather notebook",
                "Red linen notebook",
                "Ceramic travel mug",
                "Small brass key",
            ]
        )

        qa = self._build(proposal)
        self.assertEqual(len(qa["options"]), 4)

    def test_rejects_gold_text_resolving_to_multiple_labels(self) -> None:
        proposal = self._proposal(
            ["Blue binder", "Red notebook", "Ceramic mug", "Brass key"],
            option_ids=["Red notebook", "B", "C", "D"],
            gold="Red notebook",
        )

        with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "gold text is ambiguous across option labels"):
            self._build(proposal)

    def test_accepts_distinct_options_with_unambiguous_gold(self) -> None:
        proposal = self._proposal(["Red notebook", "Blue binder", "Ceramic mug", "Brass key"])

        qa = self._build(copy.deepcopy(proposal))

        self.assertEqual(qa["gold"], ["A"])
        self.assertEqual([option["id"] for option in qa["options"]], list("ABCD"))
        self.assertEqual(
            qa["query_plan"]["resolved_reasoning_event_instance_ids"],
            ["event-1", "event-2", "event-3"],
        )
        self.assertEqual(
            qa["query_plan"]["resolved_distractor_event_instance_ids"],
            ["event-2", "event-3", "event-4"],
        )

    def test_raw_graph_slot_failure_does_not_waste_qa_only_repair(self) -> None:
        error = subject.NativeGraphDialogueReject(
            "stage64 API QA semantic quality failed: ['generic_raw_qa_slot']"
        )
        self.assertFalse(subject._is_stage64_qa_repairable_error(error))

    def test_missing_compiled_operation_fields_do_not_waste_qa_only_repair(self) -> None:
        for reason in (
            "bridge_missing_two_hop_dependency",
            "visibility_missing_scope",
            "count_missing_candidate_set",
            "cause_missing_ordered_dependency",
        ):
            with self.subTest(reason=reason):
                error = subject.NativeGraphDialogueReject(
                    f"stage64 API QA semantic quality failed: ['{reason}']"
                )
                self.assertFalse(subject._is_stage64_qa_repairable_error(error))

    def test_qa_schema_failure_remains_repairable(self) -> None:
        error = subject.NativeGraphDialogueReject("stage64 API QA exact duplicate option text")
        self.assertTrue(subject._is_stage64_qa_repairable_error(error))

    def test_graph_cache_key_is_operation_and_answer_type_specific(self) -> None:
        args = SimpleNamespace(min_length_delta_tokens=512, query_conditioned_graph_expansion=False)
        seed = {"sample_id": "seed-1"}
        common = {
            "seed_row": seed,
            "planned_cell": {"question_type": "single_choice", "attack_skill": "conditional_plan"},
            "api_args": args,
            "sample_index": 3,
            "round_id": "stage64",
        }

        temporal_key = subject.graph_cache_key_for_attempt(
            policy={
                "stage64_target": {
                    "qa_operation": "temporal_state_update",
                    "answer_type": "DateOrTime",
                }
            },
            **common,
        )
        count_key = subject.graph_cache_key_for_attempt(
            policy={"stage64_target": {"qa_operation": "count", "answer_type": "Number"}},
            **common,
        )

        self.assertNotEqual(temporal_key, count_key)

    def test_stage64_graph_quality_failure_has_stable_reason(self) -> None:
        error = subject.NativeGraphDialogueReject(
            "stage64 graph semantic quality failed: ['generic_graph_slot']"
        )
        self.assertEqual(subject._failure_reason_from_exception(error), "stage64_graph_quality")


if __name__ == "__main__":
    unittest.main()
