"""Focused checks for the paired pilot's human review boundary."""

from __future__ import annotations

import json
import unittest

import review_gate


BANK_SHA = "b" * 64
SOURCE_SHA = "a" * 64


def surface() -> dict:
    turns = [{"turn_ref": f"T{i}", "speaker": "Speaker", "text": f"Evidence {i}: status was recorded."}
             for i in range(1, 9)]
    turns.extend({"turn_ref": f"C{i}", "speaker": "Speaker",
                  "text": f"Event {i} occurred before event {i + 1}."} for i in range(1, 4))
    return {"root_id": "sample-root", "source_sha256": SOURCE_SHA, "turns": turns}


def multi_candidate() -> dict:
    choices = {label: f"Claim about relation {index}." for index, label in enumerate("ABCD", 1)}
    evidence = {}
    for index, label in enumerate("ABCD", 1):
        refs = (f"T{2 * index - 1}", f"T{2 * index}")
        evidence[label] = {"checkpoints": [{"ref": ref, "quote": f"Evidence {2 * index - 1 + offset}"}
                                               for offset, ref in enumerate(refs)]}
    return {"candidate_id": "candidate-1", "question_type": "multi_select",
            "question": "Which claims hold?", "options": choices, "gold": ["A", "C"],
            "option_evidence": evidence, "status": "UNREVIEWED_CANDIDATE"}


def artifact(candidate: dict) -> bytes:
    return json.dumps({"questions": [candidate]}, ensure_ascii=False).encode("utf-8")


def completed_review(packet: dict, gold: list[str]) -> dict:
    review = review_gate.review_template(packet)
    review["reviewer"] = "independent-reviewer"
    review["review_minutes"] = 3.5
    for index, row in enumerate(review["option_reviews"], 1):
        row["verdict"] = "supported" if row["label"] in gold else "contradicted"
        row["same_scope"] = True
        row["direct_evidence"] = True
        row["reason"] = "Directly checked both source statements and scope."
        row["checked_facts"] = [
            {"relation_key": f"relation-{index}", "ref": f"T{2 * index - 1}",
             "quote": f"Evidence {2 * index - 1}"},
            {"relation_key": f"relation-{index}", "ref": f"T{2 * index}",
             "quote": f"Evidence {2 * index}"},
        ]
    review["checks"] = dict.fromkeys(review_gate.CHECKS, True)
    review["dedup"] = {"tested_relation_keys": [f"relation-{index}" for index in range(1, 5)],
                       "temporal_chain_key": "", "prior_bank_result": "unique",
                       "within_root_result": "unique", "matched_refs": [],
                       "reason": "Compared these four relations with the frozen prior bank and other root candidates."}
    review["decision"] = "pass_to_m0"
    return review


class ReviewGateTests(unittest.TestCase):
    def test_multi_blinds_gold_and_binds_original_candidate(self) -> None:
        candidate = multi_candidate()
        raw = artifact(candidate)
        packet = review_gate.prepare_review_inputs(raw, surface(), BANK_SHA)["packets"][0]
        self.assertNotIn("gold", packet["candidate"])
        self.assertNotIn("candidate_id", packet["candidate"])
        self.assertEqual(packet["candidate"]["options"], candidate["options"])
        review = completed_review(packet, candidate["gold"])
        result = review_gate.validate_review(packet, review, raw, surface(), BANK_SHA)
        self.assertEqual(result["decision"], "pass_to_m0")
        self.assertEqual(result["option_decision_counts"],
                         {"supported": 2, "contradicted": 2, "ambiguous": 0})
        self.assertEqual(result["question_sha256"], review_gate._digest(candidate))
        self.assertEqual(result["review_minutes"], 3.5)
        aggregate = review_gate.aggregate_reviews([{"packet": packet, "review": review,
                                                    "result": result}], [])
        self.assertEqual(aggregate["summary"]["counts"]["m0_eligible"], 1)
        self.assertEqual(aggregate["training_promoted"], 0)

    def test_wrong_gold_and_missing_review_cannot_pass(self) -> None:
        candidate = multi_candidate()
        raw = artifact(candidate)
        packet = review_gate.prepare_review_inputs(raw, surface(), BANK_SHA)["packets"][0]
        review = completed_review(packet, ["A", "B"])
        with self.assertRaisesRegex(ValueError, "option_verdict_pass_inconsistent"):
            review_gate.validate_review(packet, review, raw, surface(), BANK_SHA)
        review = completed_review(packet, candidate["gold"])
        review["option_reviews"].pop()
        with self.assertRaisesRegex(ValueError, "option_review_labels_invalid"):
            review_gate.validate_review(packet, review, raw, surface(), BANK_SHA)

    def test_tampered_source_and_candidate_are_rejected(self) -> None:
        candidate = multi_candidate()
        raw = artifact(candidate)
        packet = review_gate.prepare_review_inputs(raw, surface(), BANK_SHA)["packets"][0]
        review = completed_review(packet, candidate["gold"])
        changed = multi_candidate()
        changed["options"]["A"] = "Changed wording."
        with self.assertRaisesRegex(ValueError, "packet_not_bound_to_raw_candidate"):
            review_gate.validate_review(packet, review, artifact(changed), surface(), BANK_SHA)
        changed_surface = surface()
        changed_surface["turns"][0]["text"] = "Other content"
        with self.assertRaisesRegex(ValueError, "packet_not_bound_to_raw_candidate"):
            review_gate.validate_review(packet, review, raw, changed_surface, BANK_SHA)

    def test_shape_failure_is_not_semantic_pass(self) -> None:
        raw = artifact(multi_candidate())
        audit = {"source_sha256": SOURCE_SHA, "errors": [],
                 "candidates": [{"status": "FAIL_SHAPE", "errors": ["quote_not_found"]}]}
        prepared = review_gate.prepare_review_inputs(raw, surface(), BANK_SHA, audit)
        self.assertEqual(prepared["packets"], [])
        self.assertEqual(prepared["skipped"][0]["reason"], "shape_failed")

    def test_invalid_response_is_not_counted_as_emitted_candidate(self) -> None:
        prepared = review_gate.prepare_review_inputs(b"{}", surface(), BANK_SHA,
                                                     {"source_sha256": SOURCE_SHA,
                                                      "errors": ["questions_array_missing"],
                                                      "candidates": []})
        summary = review_gate.summarize_reviews([], prepared["skipped"])
        self.assertEqual(summary["counts"]["emitted"], 0)
        self.assertEqual(summary["counts"]["response_shape_failures"], 1)

    def test_missing_questions_array_is_recorded_without_stopping_review(self) -> None:
        audit = {"source_sha256": SOURCE_SHA, "errors": ["questions_array_missing"],
                 "candidates": []}
        for raw in (b"{}", b'{"questions":"not an array"}', b"[]"):
            prepared = review_gate.prepare_review_inputs(raw, surface(), BANK_SHA, audit)
            self.assertEqual(prepared, {"packets": [], "skipped": [{
                "index": None, "reason": "shape_failed", "detail": ["questions_array_missing"]}]})
            aggregate = review_gate.aggregate_reviews([], prepared["skipped"])
            self.assertEqual(aggregate["summary"]["counts"]["m0_eligible"], 0)

    def test_rejected_candidate_keeps_complete_option_audit(self) -> None:
        candidate = multi_candidate()
        raw = artifact(candidate)
        packet = review_gate.prepare_review_inputs(raw, surface(), BANK_SHA)["packets"][0]
        review = completed_review(packet, candidate["gold"])
        review["option_reviews"][1]["verdict"] = "ambiguous"
        review["option_reviews"][1]["direct_evidence"] = False
        review["checks"]["direct_false_contradiction"] = False
        review["decision"] = "reject"
        review["rejection_reasons"] = ["unsupported_false"]
        result = review_gate.validate_review(packet, review, raw, surface(), BANK_SHA)
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["option_decision_counts"]["ambiguous"], 1)
        summary = review_gate.summarize_reviews([result], [])
        self.assertEqual(summary["counts"]["semantic_pass"], 0)
        self.assertEqual(summary["counts"]["m0_eligible"], 0)

    def test_ordering_requires_each_visible_edge(self) -> None:
        order = ["I2", "I1", "I4", "I3"]
        candidate = {"candidate_id": "candidate-order", "question_type": "ordering",
                     "question": "Order the observed events.",
                     "items": {f"I{i}": f"Observed event {i}." for i in range(1, 5)},
                     "gold": order,
                     "event_evidence": {f"I{i}": {"evidence": {"ref": f"T{i}",
                                                                    "quote": f"Evidence {i}"}}
                                        for i in range(1, 5)},
                     "adjacent_edge_evidence": [
                         {"before_ref": f"T{int(before[1:])}",
                          "after_ref": f"T{int(after[1:])}",
                          "temporal_evidence": {"ref": f"C{index}",
                                                "quote": f"Event {index} occurred before event {index + 1}."}}
                         for index, (before, after) in enumerate(zip(order, order[1:]), 1)]}
        raw = artifact(candidate)
        packet = review_gate.prepare_review_inputs(raw, surface(), BANK_SHA)["packets"][0]
        self.assertEqual(packet["candidate"]["proposed_order"], order)
        self.assertEqual(packet["cited_evidence"]["choices"]["I1"][0]["quote"], "Evidence 1")
        review = review_gate.review_template(packet)
        review["reviewer"] = "independent-reviewer"
        review["review_minutes"] = 4
        for item in review["option_reviews"]:
            label = item["label"]
            index = int(label[1:])
            item.update(verdict="observed", same_scope=True, direct_evidence=True,
                        reason="The source explicitly reports the occurrence.",
                        checked_facts=[{"relation_key": f"event-{index}", "ref": f"T{index}",
                                        "quote": f"Evidence {index}"}])
        for index, item in enumerate(review["edge_reviews"], 1):
            item.update(explicit_order=True, before_ref=f"T{int(item['before_label'][1:])}",
                        after_ref=f"T{int(item['after_label'][1:])}", temporal_ref=f"C{index}",
                        temporal_quote=f"Event {index} occurred before event {index + 1}.",
                        reason="Explicit temporal language connects these occurrences.")
        review["checks"] = dict.fromkeys(review_gate.CHECKS, True)
        review["dedup"] = {"tested_relation_keys": ["event-1", "event-2", "event-3", "event-4"],
                           "temporal_chain_key": "event-2>event-1>event-4>event-3",
                           "prior_bank_result": "unique", "within_root_result": "unique",
                           "matched_refs": [], "reason": "Unique temporal chain in frozen bank."}
        review["decision"] = "pass_to_m0"
        result = review_gate.validate_review(packet, review, raw, surface(), BANK_SHA)
        self.assertEqual(result["edge_decision_counts"], {"explicit": 3, "unproved": 0})
        review["edge_reviews"][1]["explicit_order"] = False
        with self.assertRaisesRegex(ValueError, "ordering_pass_inconsistent"):
            review_gate.validate_review(packet, review, raw, surface(), BANK_SHA)


if __name__ == "__main__":
    unittest.main()
