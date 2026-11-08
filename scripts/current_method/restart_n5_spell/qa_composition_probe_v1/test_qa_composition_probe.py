from __future__ import annotations

import unittest

from .contracts import AtomicQA, extract_entities, extract_topic_terms
from .input_adapter import build_atomic_ir
from .pair_compiler import compile_pairs
from .static_audit import audit_composite, is_mechanical_conjunction


def _qa(
    qa_id: str, *, root: str = "root-1", question: str,
    gold: str, refs: list[str], question_type: str = "causal",
    evidence: list[str] | None = None,
) -> AtomicQA:
    evidence = evidence or [f"Evidence for {ref}" for ref in refs]
    values = (question, gold, *evidence)
    return AtomicQA(
        qa_id=qa_id,
        root_id=root,
        question=question,
        gold_text=gold,
        correct_option="A",
        options={"A": gold, "B": "Wrong B", "C": "Wrong C", "D": "Wrong D"},
        evidence_turn_refs=refs,
        evidence_texts=evidence,
        source_scope="parent_child",
        entities=extract_entities(*values),
        event_terms=extract_topic_terms(*values),
        turn_indices=[int(ref[1:]) for ref in refs],
        question_type=question_type,
    )


class PairCompilerTest(unittest.TestCase):
    def test_different_root_rejected(self) -> None:
        left = _qa("QA01", question="Why did Classmates A support music?",
                   gold="For the festival", refs=["T1"])
        right = _qa("QA02", root="root-2",
                    question="What music outcome did Classmates A choose?",
                    gold="A festival", refs=["T3"])
        candidates, rejections = compile_pairs([left, right])
        self.assertEqual(candidates, [])
        self.assertEqual(rejections[0]["reason"], "different_root")

    def test_identical_evidence_rejected(self) -> None:
        left = _qa("QA01", question="Why did Classmates A support music?",
                   gold="For the festival", refs=["T1"])
        right = _qa("QA02", question="What music did Classmates A choose?",
                    gold="A concert", refs=["T1"])
        candidates, rejections = compile_pairs([left, right])
        self.assertEqual(candidates, [])
        self.assertEqual(rejections[0]["reason"], "same_evidence")

    def test_paraphrase_fact_rejected(self) -> None:
        left = _qa(
            "QA01", question="What music event did Classmates A choose?",
            gold="They chose a community music festival", refs=["T1"])
        right = _qa(
            "QA02", question="Which music event was chosen by Classmates A?",
            gold="They chose a community music festival", refs=["T2"])
        candidates, rejections = compile_pairs([left, right])
        self.assertEqual(candidates, [])
        self.assertEqual(rejections[0]["reason"], "duplicate_fact")

    def test_state_change_candidate(self) -> None:
        left = _qa(
            "QA01", question="When did Classmates A originally plan music practice?",
            gold="Next week", refs=["T1"], question_type="information_revision")
        right = _qa(
            "QA02", question="When did Classmates A finally plan music practice?",
            gold="Tomorrow", refs=["T9"], question_type="information_revision")
        candidates, _ = compile_pairs([left, right])
        self.assertEqual(candidates[0]["candidate_type"], "state_change")

    def test_proposal_vs_outcome_candidate(self) -> None:
        left = _qa(
            "QA01", question="What day did Classmates B suggest for the meeting?",
            gold="Saturday", refs=["T2"], question_type="relationship_or_decision")
        right = _qa(
            "QA02", question="What day was finally agreed on for the meeting?",
            gold="Sunday", refs=["T8"], question_type="relationship_or_decision")
        candidates, _ = compile_pairs([left, right])
        self.assertEqual(candidates[0]["candidate_type"], "proposal_vs_outcome")

    def test_actor_outcome_binding_candidate(self) -> None:
        left = _qa(
            "QA01", question="Who preferred Sunday for the meeting?",
            gold="Classmates B", refs=["T2"], question_type="information_visibility")
        right = _qa(
            "QA02", question="What day was finally agreed on for the meeting?",
            gold="Sunday", refs=["T8"], question_type="relationship_or_decision")
        candidates, _ = compile_pairs([left, right])
        self.assertEqual(candidates[0]["candidate_type"], "actor_outcome_binding")

    def test_pair_order_is_stable(self) -> None:
        qas = [
            _qa("QA03", question="What day was finally agreed on for the meeting?",
                gold="Sunday", refs=["T8"], question_type="relationship_or_decision"),
            _qa("QA01", question="What day did Classmates B suggest for the meeting?",
                gold="Saturday", refs=["T2"], question_type="relationship_or_decision"),
            _qa("QA02", question="Who preferred Sunday for the meeting?",
                gold="Classmates B", refs=["T4"], question_type="information_visibility"),
        ]
        first, _ = compile_pairs(qas)
        second, _ = compile_pairs(reversed(qas))
        self.assertEqual(first, second)


class InputAndAuditTest(unittest.TestCase):
    def test_full_context_failure_is_not_an_atomic_building_block(self) -> None:
        transcript = {
            "root_id": "root-1",
            "turns": [{"turn_ref": "T1", "text": "Saturday was proposed."}],
        }
        row = {
            "qa_id": "QA01",
            "question": "What day was proposed?",
            "question_type": "relationship_or_decision",
            "options": {"A": "Saturday", "B": "Sunday", "C": "Monday", "D": "Friday"},
            "reference_answer_letter": "A",
            "canonical_answer_text": "Saturday",
            "evidence_turn_ids": ["T1"],
        }
        evaluation = {
            "grounding": {"decision": "pass_grounding"},
            "full_context_correct": 0,
            "full_context_total": 5,
        }
        qas, records = build_atomic_ir(
            [row], transcript=transcript, evaluations={"QA01": evaluation})
        self.assertEqual(qas, [])
        self.assertIn("full_context_not_stably_correct",
                      records[0]["ineligibility_reasons"])

        evaluation["full_context_correct"] = 5
        qas, _ = build_atomic_ir(
            [row], transcript=transcript, evaluations={"QA01": evaluation})
        self.assertEqual(len(qas), 1)

    def test_missing_gold_or_evidence_is_ineligible(self) -> None:
        transcript = {
            "root_id": "root-1",
            "turns": [{"turn_ref": "T1", "text": "Saturday was proposed."}],
        }
        base = {
            "qa_id": "QA01",
            "question": "What day was proposed?",
            "question_type": "relationship_or_decision",
            "options": {"A": "Saturday", "B": "Sunday", "C": "Monday", "D": "Friday"},
            "reference_answer_letter": "A",
            "canonical_answer_text": "",
            "evidence_turn_ids": [],
        }
        qas, records = build_atomic_ir(
            [base], transcript=transcript,
            evaluations={"QA01": {"grounding": {"decision": "pass_grounding"}}},
        )
        self.assertEqual(qas, [])
        self.assertIn("missing_gold", records[0]["ineligibility_reasons"])
        self.assertIn("missing_or_duplicate_evidence", records[0]["ineligibility_reasons"])

    def test_mechanical_conjunction_detection(self) -> None:
        self.assertTrue(is_mechanical_conjunction(
            "What did Classmates A propose, and what did Classmates B choose?"))
        self.assertFalse(is_mechanical_conjunction(
            "How did the final choice differ from the original proposal?"))

    def test_static_audit_checks_both_evidence_groups(self) -> None:
        qa1 = _qa(
            "QA01", question="What day did Classmates B suggest for the meeting?",
            gold="Saturday", refs=["T2"], question_type="relationship_or_decision")
        qa2 = _qa(
            "QA02", question="What day was finally agreed on for the meeting?",
            gold="Sunday", refs=["T8"], question_type="relationship_or_decision")
        composite = {
            "required_qa_ids": ["QA01", "QA02"],
            "required_evidence_groups": [
                {"source_qa_id": "QA01", "turn_refs": ["T2"]},
                {"source_qa_id": "QA02", "turn_refs": ["T8"]},
            ],
            "required_evidence_turn_refs": ["T2", "T8"],
            "question": "How did the final meeting day differ from the proposed day?",
            "answer_dimension": "transition",
            "gold_text": "It changed from Saturday to Sunday.",
            "options": {
                "A": "It changed from Saturday to Sunday.",
                "B": "It changed from Sunday to Saturday.",
                "C": "It changed from Saturday to Saturday.",
                "D": "It changed from Sunday to Sunday.",
            },
            "correct_option": "A",
            "relation_summary": "The final day differs from the earlier proposal.",
            "is_mechanical_conjunction": False,
        }
        audit = audit_composite(composite, atomic_by_id={"QA01": qa1, "QA02": qa2})
        self.assertTrue(audit["valid"], audit)

        composite["required_evidence_groups"][1]["turn_refs"] = ["T2"]
        bad = audit_composite(composite, atomic_by_id={"QA01": qa1, "QA02": qa2})
        self.assertIn("evidence_group_outside_source:QA02", bad["errors"])


if __name__ == "__main__":
    unittest.main()
