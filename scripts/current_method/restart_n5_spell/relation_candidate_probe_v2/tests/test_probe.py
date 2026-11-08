from __future__ import annotations

import unittest

from ..contracts import Fact, Turn
from ..fact_extractor import extract_facts
from ..paired_qa_planner import build_paired_qa_target
from ..relation_miner import mine_relations
from ..validator import validate_relation


def _turn(index: int, text: str, speaker: str = "Classmates A",
          origin: str = "generated_extension") -> Turn:
    return Turn(index, f"T{index}", speaker, text, origin, "root-1")


class FactExtractionTest(unittest.TestCase):
    def test_extracts_specific_proposal_and_outcome(self) -> None:
        turns = [
            _turn(1, "I was thinking a music festival. It could be fun."),
            _turn(7, "I think I'm going to go for it and start planning the festival."),
        ]
        proposals, outcomes = extract_facts(turns)
        self.assertTrue(any(f.event_key == "event_format:music_festival" for f in proposals))
        self.assertTrue(any(f.event_key == "event_format:music_festival" for f in outcomes))

    def test_broad_topic_is_not_an_event_fact(self) -> None:
        turns = [_turn(1, "We should focus on environmental causes."),
                 _turn(8, "We agreed the event should help people.")]
        proposals, outcomes = extract_facts(turns)
        self.assertEqual(proposals, [])
        self.assertEqual(outcomes, [])

    def test_broad_topic_fact_is_rejected(self) -> None:
        proposal = Fact("PF1", "root-1", "topic:environment", "environment topic",
                        "A", ["A"], "proposal", "environment", "T1", 1, "A",
                        "We should focus on environmental causes.", "source_grounded", "suggest")
        outcome = Fact("OF1", "root-1", "topic:environment", "environment topic",
                       "B", ["A", "B"], "outcome", "environment", "T8", 8, "B",
                       "We agreed to focus on environmental causes.", "source_grounded", "agreed")
        self.assertIn("broad_topic_only", validate_relation(proposal, outcome)["errors"])


class RelationValidationTest(unittest.TestCase):
    def test_same_event_same_value_is_eligible(self) -> None:
        proposal = Fact("PF1", "root-1", "event_type:charity_event", "charity event decision",
                        "Classmates B", ["Classmates B"], "proposal", "charity event",
                        "T1", 1, "Classmates B", "Maybe we could have a charity event next?",
                        "source_grounded", "idea")
        outcome = Fact("OF1", "root-1", "event_type:charity_event", "charity event decision",
                       "Classmates A", ["Classmates A", "Classmates B"], "outcome", "charity event",
                       "T8", 8, "Classmates A", "Let's start planning the charity event soon.",
                       "source_grounded", "let_us")
        result = validate_relation(proposal, outcome)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["value_relation"], "same")

    def test_reversed_order_rejected(self) -> None:
        proposal = Fact("PF1", "root-1", "event_format:music_festival", "music festival format",
                        "A", ["A"], "proposal", "music festival", "T8", 8, "A", "thinking festival", "source_grounded", "thinking")
        outcome = Fact("OF1", "root-1", "event_format:music_festival", "music festival format",
                       "B", ["B"], "outcome", "music festival", "T2", 2, "B", "agreed festival", "source_grounded", "agreed")
        self.assertIn("proposal_not_before_outcome", validate_relation(proposal, outcome)["errors"])

    def test_different_event_slot_rejected(self) -> None:
        proposal = Fact("PF1", "root-1", "venue:green_venue", "green venue selection", "A", ["A"], "proposal", "green venue", "T1", 1, "A", "suggest green venue", "generated_extension", "suggest")
        outcome = Fact("OF1", "root-1", "school_partnership:recycling_drive", "school recycling partnership", "B", ["B"], "outcome", "school partnership", "T8", 8, "B", "agreed school", "generated_extension", "agreed")
        self.assertIn("event_slot_mismatch", validate_relation(proposal, outcome)["errors"])

    def test_same_actor_different_event_slot_rejected(self) -> None:
        proposal = Fact("PF1", "root-1", "venue:green_venue", "green venue selection",
                        "A", ["A"], "proposal", "green venue", "T1", 1, "A",
                        "I suggest a green venue.", "source_grounded", "suggest")
        outcome = Fact("OF1", "root-1", "planning_tool:shared_google_doc",
                       "shared planning document", "A", ["A"], "outcome",
                       "shared Google Doc for planning", "T8", 8, "A",
                       "We agreed to use a shared Google Doc.", "source_grounded", "agreed")
        self.assertIn("event_slot_mismatch", validate_relation(proposal, outcome)["errors"])

    def test_changed_values_are_recorded(self) -> None:
        proposal = Fact("PF1", "root-1", "event_format:music_festival", "music festival format",
                        "A", ["A"], "proposal", "Saturday", "T1", 1, "A",
                        "I propose Saturday for the music festival.", "source_grounded", "propose")
        outcome = Fact("OF1", "root-1", "event_format:music_festival", "music festival format",
                       "A", ["A", "B"], "outcome", "Sunday", "T8", 8, "A",
                       "We agreed on Sunday for the music festival.", "source_grounded", "agreed")
        relations, rejections = mine_relations([proposal], [outcome])
        self.assertEqual(rejections, [])
        self.assertEqual(relations[0].value_relation, "changed")
        self.assertEqual(relations[0].status, "eligible")

    def test_vague_positive_response_is_uncertain(self) -> None:
        proposal = Fact("PF1", "root-1", "event_format:music_festival", "music festival format",
                        "A", ["A"], "proposal", "music festival", "T1", 1, "A",
                        "I propose a music festival.", "source_grounded", "propose")
        outcome = Fact("OF1", "root-1", "event_format:music_festival", "music festival format",
                       "B", ["A", "B"], "outcome", "music festival", "T8", 8, "B",
                       "That sounds good.", "source_grounded", "interest", strength="medium")
        relations, rejections = mine_relations([proposal], [outcome])
        self.assertEqual(rejections, [])
        self.assertEqual(relations[0].status, "uncertain")

    def test_same_turn_is_rejected(self) -> None:
        proposal = Fact("PF1", "root-1", "event_type:charity_event", "charity event decision",
                        "A", ["A"], "proposal", "charity event", "T1", 1, "A",
                        "Maybe a charity event?", "source_grounded", "idea")
        outcome = Fact("OF1", "root-1", "event_type:charity_event", "charity event decision",
                       "A", ["A"], "outcome", "charity event", "T1", 1, "A",
                       "Let's start the charity event.", "source_grounded", "let_us")
        result = validate_relation(proposal, outcome)
        self.assertFalse(result["valid"])
        self.assertIn("same_turn_evidence", result["errors"])

    def test_missing_evidence_is_rejected(self) -> None:
        proposal = Fact("PF1", "root-1", "event_type:charity_event", "charity event decision",
                        "A", ["A"], "proposal", "charity event", "T1", 1, "A", "",
                        "source_grounded", "idea")
        outcome = Fact("OF1", "root-1", "event_type:charity_event", "charity event decision",
                       "A", ["A"], "outcome", "charity event", "T8", 8, "A",
                       "We agreed to start the charity event.", "source_grounded", "agreed")
        self.assertIn("direct_evidence_missing", validate_relation(proposal, outcome)["errors"])

    def test_mining_order_is_deterministic(self) -> None:
        proposals = [
            Fact("PF1", "root-1", "event_type:charity_event", "charity event decision",
                 "A", ["A"], "proposal", "charity event", "T1", 1, "A",
                 "I propose a charity event.", "source_grounded", "propose"),
            Fact("PF2", "root-1", "event_format:music_festival", "music festival format",
                 "A", ["A"], "proposal", "music festival", "T2", 2, "A",
                 "I propose a music festival.", "source_grounded", "propose"),
        ]
        outcomes = [
            Fact("OF1", "root-1", "event_type:charity_event", "charity event decision",
                 "B", ["A", "B"], "outcome", "charity event", "T8", 8, "B",
                 "We agreed on the charity event.", "source_grounded", "agreed"),
            Fact("OF2", "root-1", "event_format:music_festival", "music festival format",
                 "B", ["A", "B"], "outcome", "music festival", "T9", 9, "B",
                 "We agreed on the music festival.", "source_grounded", "agreed"),
        ]
        first, _ = mine_relations(proposals, outcomes)
        second, _ = mine_relations(proposals, outcomes)
        self.assertEqual([r.to_dict() for r in first], [r.to_dict() for r in second])

    def test_cross_parent_child_is_recorded(self) -> None:
        proposal = Fact("PF1", "root-1", "event_format:music_festival", "music festival format", "A", ["A"], "proposal", "music festival", "P:T1", 1, "A", "thinking music festival", "source_grounded", "thinking")
        outcome = Fact("OF1", "root-1", "event_format:music_festival", "music festival format", "B", ["A", "B"], "outcome", "music festival", "R:T3", 3, "B", "agreed music festival", "generated_extension", "agreed")
        relations, rejections = mine_relations([proposal], [outcome])
        self.assertEqual(len(rejections), 0)
        self.assertTrue(relations[0].cross_parent_child)
        self.assertEqual(relations[0].scope, "parent_child")


class PairedPlannerTest(unittest.TestCase):
    def test_same_and_changed_intents(self) -> None:
        base = {
            "relation_id": "REL0001", "status": "eligible",
            "proposal": {"turn_refs": ["T1"], "value": "Saturday"},
            "outcome": {"turn_refs": ["T8"], "value": "Sunday"},
            "value_relation": "changed",
        }
        target = build_paired_qa_target(base)
        self.assertIn("compare", target["composite_intent"])
        base["value_relation"] = "same"
        self.assertIn("whether", build_paired_qa_target(base)["composite_intent"])


if __name__ == "__main__":
    unittest.main()
