from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import stage3_claim_semantic_validator as semantic  # noqa: E402


def claim(
    claim_id: str,
    entity: str,
    predicate: str,
    value: object,
    *,
    status: str = "active",
    time_index: int = 1,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim_type": "node",
        "claim_kind": "node",
        "entity": entity,
        "slot": predicate,
        "predicate": predicate,
        "value": value,
        "status": status,
        "global_time_index": time_index,
        "canonical_support_claim": {
            "subject": entity,
            "predicate": predicate,
            "objects": [value],
            "status": status,
            "time": {"global_time_index": time_index},
        },
    }


def qa(
    qa_id: str,
    question: str,
    target_claim_id: str,
    gold_text: str,
    *,
    rule: str = "read_named_entity_attribute",
    status: str | None = None,
    candidates: list[str] | None = None,
    binder_claim_ids: list[str] | None = None,
) -> dict[str, object]:
    derivation: dict[str, object] = {
        "rule": rule,
        "target_claim_ids": [target_claim_id],
        "answer_claim_ids": [target_claim_id],
    }
    if status is not None:
        derivation["status"] = status
    if candidates is not None:
        derivation["candidate_claim_ids"] = candidates
    if binder_claim_ids is not None:
        derivation["binder_claim_ids"] = binder_claim_ids
    return {
        "qa_id": qa_id,
        "question_text_draft": question,
        "gold": ["A"],
        "gold_answer_text": gold_text,
        "options": [
            {
                "id": "A",
                "text": gold_text,
                "truth_value": True,
                "source_claim_id": target_claim_id,
            }
        ],
        "proof": {
            "proof_id": f"proof-{qa_id}",
            "required_claim_ids": list(
                dict.fromkeys([*(binder_claim_ids or []), target_claim_id])
            ),
            "derivation": derivation,
        },
    }


def codes(report: dict[str, object]) -> set[str]:
    return {item["code"] for item in report["issues"]}  # type: ignore[index]


class SemanticValidatorGoldenTest(unittest.TestCase):
    def test_atomic_typed_location_passes(self) -> None:
        location = claim("C1", "Alice", "destination", "Paris")
        question = qa("Q1", "What is Alice's destination?", "C1", "Paris")

        report = semantic.validate_qa_semantics({"claims": [location]}, question)

        self.assertTrue(report["pass"], report)

    def test_location_and_possession_use_general_type_families(self) -> None:
        destination = claim(
            "C1",
            "Amina",
            "travel_destination",
            "plans to organize a workshop at the Central Library",
        )
        possession = claim(
            "C2", "Comet", "owned_item", "was a companion animal of Priya"
        )
        destination_qa = qa(
            "Q1",
            "What is Amina's travel destination?",
            "C1",
            str(destination["value"]),
        )
        possession_qa = qa(
            "Q2", "What is Comet's owned item?", "C2", str(possession["value"])
        )

        first = semantic.validate_qa_semantics(
            {"claims": [destination]}, destination_qa
        )
        second = semantic.validate_qa_semantics(
            {"claims": [possession]}, possession_qa
        )

        self.assertIn("predicate_value_type_mismatch", codes(first))
        self.assertIn("answer_value_wrong_granularity", codes(first))
        self.assertIn("predicate_value_type_mismatch", codes(second))
        self.assertIn("semantic_role_direction_mismatch", codes(second))

    def test_reported_binder_must_not_silently_become_fact(self) -> None:
        binder = claim(
            "C1", "Alice", "travel", "a trip to Oslo", status="report"
        )
        target = claim("C2", "Alice", "occupation", "architect")
        unqualified = qa(
            "Q1",
            "What is the occupation of the person whose travel is a trip to Oslo?",
            "C2",
            "architect",
            rule="bind_unique_entity_then_read_attribute",
            binder_claim_ids=["C1"],
        )
        qualified = copy.deepcopy(unqualified)
        qualified["qa_id"] = "Q2"
        qualified["question_text_draft"] = (
            "What is the occupation of the person whose reported, unconfirmed travel was a trip to Oslo?"
        )

        bad = semantic.validate_qa_semantics(
            {"claims": [binder, target]}, unqualified
        )
        good = semantic.validate_qa_semantics(
            {"claims": [binder, target]}, qualified
        )

        self.assertIn("binder_status_qualification_missing", codes(bad))
        self.assertTrue(good["pass"], good)

    def test_expired_direct_answer_is_not_current_by_default(self) -> None:
        old = claim("C1", "Alice", "vehicle", "a blue bicycle", status="expired")
        unqualified = qa("Q1", "What is Alice's vehicle?", "C1", "a blue bicycle")
        qualified = qa(
            "Q2",
            "What was Alice's expired vehicle record?",
            "C1",
            "a blue bicycle",
        )

        bad = semantic.validate_qa_semantics({"claims": [old]}, unqualified)
        good = semantic.validate_qa_semantics({"claims": [old]}, qualified)

        self.assertIn("answer_status_qualification_missing", codes(bad))
        self.assertTrue(good["pass"], good)

    def test_status_conditioned_nonactive_answers_are_valid(self) -> None:
        for index, status in enumerate(("belief", "outdated", "canceled"), start=1):
            with self.subTest(status=status):
                target = claim(f"C{index}", "Alice", "access_scope", f"scope-{index}", status=status)
                question = qa(
                    f"Q{index}",
                    f"Which access scope is recorded as {status} for Alice?",
                    f"C{index}",
                    f"scope-{index}",
                    rule="select_unique_status_in_entity_slot",
                    status=status,
                    candidates=[f"C{index}"],
                )
                report = semantic.validate_qa_semantics(
                    {"claims": [target]}, question
                )
                self.assertTrue(report["pass"], report)

    def test_current_primary_can_explicitly_disregard_canceled_candidate(self) -> None:
        active = claim("C-active", "John", "access_scope", "community center", time_index=2)
        canceled = claim(
            "C-canceled",
            "John",
            "access_scope",
            "food bank",
            status="canceled",
            time_index=4,
        )
        primary = qa(
            "Q-primary",
            "After disregarding the canceled proposal, what is John's current access scope?",
            "C-active",
            "community center",
            rule="select_current_active_after_status_filter",
            candidates=["C-active", "C-canceled"],
        )

        report = semantic.validate_qa_semantics(
            {"claims": [active, canceled]}, primary
        )

        self.assertTrue(report["pass"], report)

    def test_direct_lookup_requires_status_or_time_when_claims_conflict(self) -> None:
        old = claim("C1", "Alice", "destination", "Rome", status="outdated", time_index=1)
        current = claim("C2", "Alice", "destination", "Paris", time_index=2)
        vague = qa("Q1", "What is Alice's destination?", "C2", "Paris")
        resolved = qa(
            "Q2", "What is Alice's current destination?", "C2", "Paris"
        )

        bad = semantic.validate_qa_semantics({"claims": [old, current]}, vague)
        good = semantic.validate_qa_semantics(
            {"claims": [old, current]}, resolved
        )

        self.assertIn("dialogue_value_ambiguity", codes(bad))
        self.assertTrue(good["pass"], good)

    def test_full_dialogue_location_conflict_is_detected(self) -> None:
        current = claim("C1", "Alice", "destination", "Berlin")
        question = qa("Q1", "What is Alice's destination?", "C1", "Berlin")
        dialogue = {
            "turns": [
                {
                    "turn_id": "D1:1",
                    "speaker": "Alice",
                    "text": "Last year I took a trip and relaxed in Rome.",
                }
            ]
        }

        report = semantic.validate_qa_semantics(
            {"claims": [current]}, question, dialogue
        )

        self.assertIn("dialogue_value_ambiguity", codes(report))
        conflict = next(
            item for item in report["issues"] if item["code"] == "dialogue_value_ambiguity"
        )
        self.assertEqual(
            conflict["details"]["conflicting_candidates"][0]["turn_id"], "D1:1"
        )

    def test_binder_and_target_subject_direction_must_match(self) -> None:
        binder = claim("C1", "Alice", "hobby", "chess")
        target = claim("C2", "Bob", "occupation", "teacher")
        question = qa(
            "Q1",
            "What is the occupation of the person whose hobby is chess?",
            "C2",
            "teacher",
            rule="bind_unique_entity_then_read_attribute",
            binder_claim_ids=["C1"],
        )

        report = semantic.validate_qa_semantics(
            {"claims": [binder, target]}, question
        )

        self.assertIn("binder_target_subject_mismatch", codes(report))


class SemanticValidatorBatchAndMutationTest(unittest.TestCase):
    def test_batch_reports_each_sibling_without_mutating_or_filtering(self) -> None:
        good_claim = claim("C1", "Alice", "destination", "Paris")
        bad_claim = claim(
            "C2", "Bob", "destination", "plans to give a lecture at a library"
        )
        good_qa = qa("Q-good", "What is Alice's destination?", "C1", "Paris")
        bad_qa = qa(
            "Q-bad", "What is Bob's destination?", "C2", str(bad_claim["value"])
        )
        claim_ir = {"claims": [good_claim, bad_claim]}
        siblings = [good_qa, bad_qa]
        before = copy.deepcopy((claim_ir, siblings))

        report = semantic.validate_multiqa_semantics(claim_ir, siblings)

        self.assertEqual(report["accepted_qa_ids"], ["Q-good"])
        self.assertEqual(report["rejected_qa_ids"], ["Q-bad"])
        self.assertEqual(report["qa_count"], 2)
        self.assertEqual((claim_ir, siblings), before)

    def test_mutations_of_type_status_and_subject_are_killed(self) -> None:
        base_claim = claim("C1", "Alice", "destination", "Paris")
        base_qa = qa("Q1", "What is Alice's destination?", "C1", "Paris")
        self.assertTrue(
            semantic.validate_qa_semantics({"claims": [base_claim]}, base_qa)["pass"]
        )

        value_mutant = copy.deepcopy(base_claim)
        value_mutant["value"] = "plans to attend a conference in Paris"
        value_mutant["canonical_support_claim"]["objects"] = [value_mutant["value"]]
        status_mutant = copy.deepcopy(base_claim)
        status_mutant["status"] = "unconfirmed"
        status_mutant["canonical_support_claim"]["status"] = "unconfirmed"
        subject_mutant = copy.deepcopy(base_claim)
        subject_mutant["entity"] = "Bob"
        subject_mutant["canonical_support_claim"]["subject"] = "Bob"

        self.assertIn(
            "predicate_value_type_mismatch",
            codes(semantic.validate_qa_semantics({"claims": [value_mutant]}, base_qa)),
        )
        self.assertIn(
            "answer_status_qualification_missing",
            codes(semantic.validate_qa_semantics({"claims": [status_mutant]}, base_qa)),
        )
        self.assertIn(
            "question_subject_mismatch",
            codes(semantic.validate_qa_semantics({"claims": [subject_mutant]}, base_qa)),
        )

    def test_module_is_independent_from_compiler_implementation(self) -> None:
        source = Path(semantic.__file__).read_text(encoding="utf-8")
        self.assertNotIn("stage3_claim_proof_compiler", source)
        self.assertNotIn("stage3_claim_ir import", source)


class Run224SemanticReplayTest(unittest.TestCase):
    RUN224 = ROOT / (
        "base_restart_current_method_validation/224_stage3_claim_first_smoke_n1/"
        "generation/clean_samples.jsonl"
    )

    def test_known_three_relation_failures_are_isolated_from_three_status_qas(self) -> None:
        if not self.RUN224.exists():
            self.skipTest("run224 artifact is not available")
        sample = json.loads(self.RUN224.read_text(encoding="utf-8").splitlines()[0])
        before = copy.deepcopy(sample)

        report = semantic.validate_sample_semantics(sample)

        self.assertEqual(report["qa_count"], 6)
        self.assertEqual(report["accepted_count"], 3)
        self.assertEqual(report["rejected_count"], 3)
        self.assertEqual(
            report["accepted_qa_ids"],
            [
                "qa_234b7cfcb558820c49b91301",
                "qa_450a8cc5cea35be763433249",
                "qa_456305c79b7e42bbcc6a4552",
            ],
        )
        by_id = {item["qa_id"]: item for item in report["reports"]}
        self.assertIn(
            "binder_status_qualification_missing",
            codes(by_id["qa_d6ecc8f0203ad3adaa334abf"]),
        )
        self.assertIn(
            "semantic_role_direction_mismatch",
            codes(by_id["qa_402eb7934f9c557ea423803d"]),
        )
        self.assertIn(
            "predicate_value_type_mismatch",
            codes(by_id["qa_94e911fd33573db431141053"]),
        )
        self.assertIn(
            "dialogue_value_ambiguity",
            codes(by_id["qa_94e911fd33573db431141053"]),
        )
        self.assertEqual(sample, before)


if __name__ == "__main__":
    unittest.main()
