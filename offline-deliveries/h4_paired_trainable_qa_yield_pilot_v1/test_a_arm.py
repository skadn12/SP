"""Offline contract checks for the paired pilot's current-Writer arm."""

from __future__ import annotations

import copy
import json
import unittest

import a_arm


def surface() -> dict:
    turns = [
        ("example:P01:T001", "Emily", "I submitted the blue form."),
        ("example:P01:T002", "James", "I kept the green file."),
        ("R02:S01:T001", "Emily", "I sent the blue form after breakfast."),
        ("R02:S01:T002", "James", "I opened the green file before lunch."),
        ("R02:S02:T001", "Emily", "I signed the blue form after lunch."),
        ("R02:S02:T002", "James", "I closed the green file before dinner."),
    ]
    return {"root_id": "example", "source_sha256": "a" * 64,
            "turns": [{"turn_ref": ref, "speaker": speaker, "text": text} for ref, speaker, text in turns]}


def multi_candidate(candidate_id: str) -> dict:
    texts = surface()["turns"]
    evidence = {}
    for index, label in enumerate("ABCDE"):
        true = index < 2
        first, second = texts[index % 2], texts[2 + index % 4]
        evidence[label] = {"verdict": "TRUE" if true else "FALSE", "clauses": [
            {"claim": "first fact", "verdict": "TRUE", "refs": [first["turn_ref"]], "quote": first["text"][:8], "reason": "direct statement"},
            {"claim": "second fact", "verdict": "TRUE" if true else "FALSE", "refs": [second["turn_ref"]], "quote": second["text"][:8], "reason": "direct statement or contradiction"},
        ]}
    return {"candidate_id": candidate_id, "question_type": "multi_select", "intent": "two distinct facts", "question": "Select all that apply.",
            "options": {label: f"claim {label} one and claim {label} two" for label in "ABCDE"},
            "gold": ["A", "B"], "option_evidence": evidence}


def ordering_candidate() -> dict:
    refs = ["R02:S01:T001", "R02:S01:T002", "R02:S02:T001", "R02:S02:T002"]
    texts = {turn["turn_ref"]: turn["text"] for turn in surface()["turns"]}
    labels = ["I1", "I2", "I3", "I4"]
    # Reverse the display order deliberately: structure must not infer occurrence order.
    gold = list(reversed(labels))
    evidence = {label: {"track": "form" if i % 2 == 0 else "file", "refs": [refs[i]],
                        "quote": texts[refs[i]][:8], "reason": "reported action"}
                for i, label in enumerate(labels)}
    edges = [{"before": before, "after": after, "before_refs": evidence[before]["refs"],
              "after_refs": evidence[after]["refs"], "order_refs": evidence[before]["refs"],
              "reason": "claimed spoken time cue"} for before, after in zip(gold, gold[1:])]
    return {"candidate_id": "order-1", "question_type": "ordering", "intent": "action sequence",
            "question": "Put these four actions in actual occurrence order.",
            "items": {label: f"action {label}" for label in labels}, "gold": gold,
            "item_evidence": evidence, "adjacent_edges": edges}


class CurrentWriterArmTests(unittest.TestCase):
    def test_prompt_has_one_type_and_only_visible_turn_fields(self) -> None:
        example = surface()
        example["turns"][0]["hidden_gold"] = "secret"
        for question_type, cap in (("multi_select", 2), ("ordering", 1)):
            rendered = a_arm.prompt(example, question_type)
            self.assertIn(f"Submit only {question_type} questions, at most {cap}", rendered)
            self.assertNotIn("source_sha256", rendered)
            self.assertNotIn("hidden_gold", rendered)
            self.assertIn("R02:S02:T002", rendered)

    def test_two_multi_candidates_audited_independently(self) -> None:
        first, second = multi_candidate("one"), multi_candidate("two")
        second["option_evidence"]["E"]["clauses"][0]["quote"] = "not quoted"
        raw = json.dumps({"questions": [first, second], "abstentions": []})
        result = a_arm.validate(raw, surface(), "multi_select")
        self.assertEqual(result["status"], "FAIL_SHAPE")
        self.assertEqual(result["candidates"][0]["status"], "PASS_SHAPE_ONLY_NOT_GOLD")
        self.assertEqual(result["candidates"][1]["errors"], ["quote_not_found_in_cited_speech"])

    def test_order_shape_does_not_certify_chronology(self) -> None:
        raw = json.dumps({"questions": [ordering_candidate()], "abstentions": []})
        result = a_arm.validate(raw, surface(), "ordering")
        self.assertEqual(result["status"], "PASS_SHAPE_ONLY_NOT_GOLD")
        self.assertFalse(result["chronology_verified"])
        self.assertFalse(result["candidates"][0]["chronology_verified"])

    def test_abstention_and_cap(self) -> None:
        raw = json.dumps({"questions": [], "abstentions": [{"question_type": "ordering", "reason": "No explicit occurrence order."}]})
        self.assertEqual(a_arm.validate(raw, surface(), "ordering")["status"], "ABSTAINED")
        candidate = ordering_candidate()
        duplicate = copy.deepcopy(candidate)
        duplicate["candidate_id"] = "order-2"
        result = a_arm.validate(json.dumps({"questions": [candidate, duplicate], "abstentions": []}), surface(), "ordering")
        self.assertIn("candidate_cap_exceeded", result["errors"])

    def test_invalid_json_has_no_candidates(self) -> None:
        result = a_arm.validate("{", surface(), "multi_select")
        self.assertEqual(result["errors"], ["invalid_json"])
        self.assertEqual(result["candidates"], [])


if __name__ == "__main__":
    unittest.main()
