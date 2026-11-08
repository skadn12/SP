from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ordering_gate import (LABELS, MODEL, ORDERING_GENERATION_INSTRUCTIONS,
                           build_review_prompt, resolve_ordering_anchors,
                           review_response, validate_generated_ordering)


class OrderingSpanContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.source = {"root_id": "root", "turns": [
            {"turn_ref": "T1", "speaker": "James", "text":
             "The total is $350. The amount is confirmed. Payment is planned. Use the shared account."},
            {"turn_ref": "T2", "speaker": "Emily", "text":
             "I’m looking forward to this. We will laugh and relax."},
            {"turn_ref": "T3", "speaker": "James", "text":
             "Me too. Meet at the station on Saturday."}]}
        self.source_path = self.directory / "source.json"
        self.source_path.write_text(json.dumps(self.source))
        definitions = {
            "I1": (0, "Use the shared account."),
            "I2": (0, "The total is $350."),
            "I3": (1, "We will laugh and relax."),
            "I4": (1, "I’m looking forward to this."),
            "I5": (2, "Meet at the station on Saturday."),
            "I6": (2, "Me too."),
        }
        self.expected = ["I2", "I1", "I4", "I3", "I6", "I5"]
        self.row = {
            "candidate_id": "q1", "root_id": "root", "question_type": "ordering",
            "chronology_kind": "conversation_order",
            "question": "Order the statements as they APPEAR IN THE SUPPLIED DIALOGUE, using left-to-right clause order within a turn.",
            "items": {label: quote for label, (_, quote) in definitions.items()},
            "gold": list(LABELS),
            "item_evidence": {
                label: {"turn_ref": self.source["turns"][index]["turn_ref"],
                        "speaker": self.source["turns"][index]["speaker"], "quote": quote,
                        "status": "statement", "reason": "This is the primary clause."}
                for label, (index, quote) in definitions.items()},
        }
        errors, self.candidate = validate_generated_ordering(
            self.row, "q1", "root", self.source, self.source_path)
        self.assertEqual(errors, [])
        self.verdict = {
            "candidate_id": "q1", "items": {
                label: {"supported": True, "order_anchor": {
                    "turn_ref": self.row["item_evidence"][label]["turn_ref"],
                    "quote": self.row["item_evidence"][label]["quote"], "reason": "Primary clause."},
                    "support_refs": [], "alternative_matching_anchors": [], "actor_ok": True,
                    "modality_ok": True, "state_ok": True, "neutral_stem_ok": True, "reason": "Supported."}
                for label in LABELS},
            "coherent": True, "unique_order": True, "alternative_orders": [],
            "neutral_stem_ok": True, "reason": "A unique textual order.",
        }

    def review(self, verdict=None, suffix="", model=MODEL, mutate_prompt=None):
        raw = json.dumps({"reviews": [self.verdict if verdict is None else verdict]})
        metadata = {"requested_model": MODEL, "returned_model": model,
                    "finish_reason": "stop", "http_status": 200}
        prompt = build_review_prompt(self.candidate, self.source) + suffix
        if mutate_prompt:
            prompt = mutate_prompt(prompt)
        request = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
        request_path, response_path = self.directory / "request.json", self.directory / "response.json"
        request_path.write_text(json.dumps(request))
        response_path.write_text(json.dumps({"text": raw, "metadata": metadata}))
        return review_response(raw, [self.candidate], self.source, response_path, request_path, metadata)

    @staticmethod
    def alternative(quote, turn_ref="T1"):
        return {"turn_ref": turn_ref, "quote": quote, "reason": "Another plausible occurrence."}

    def test_three_turns_six_items_have_unique_clause_order_without_text_edits(self):
        self.assertEqual(self.candidate["gold"], self.expected)
        self.assertEqual(self.candidate["original_gold"], list(LABELS))
        self.assertEqual(self.candidate["items"], self.row["items"])
        self.assertEqual(self.candidate["item_evidence"], self.row["item_evidence"])
        self.assertEqual(self.row["gold"], list(LABELS))
        self.assertEqual(self.candidate["generated_order_anchor_spans"]["I3"]["char_start"], 29)

    def test_unicode_offsets_are_source_string_slices_and_revalidate(self):
        spans = self.candidate["generated_order_anchor_spans"]
        for span in spans.values():
            text = self.source["turns"][span["turn_index"]]["text"]
            self.assertEqual(text[span["char_start"]:span["char_end"]], span["quote"])
        errors, resolved, gold = resolve_ordering_anchors(spans, self.source)
        self.assertEqual((errors, resolved, gold), ([], spans, self.expected))

    def test_whole_turn_reused_twice_rejects_without_label_tiebreak(self):
        evidence = deepcopy(self.row["item_evidence"])
        for label in ("I1", "I2"):
            evidence[label]["quote"] = self.source["turns"][0]["text"]
        errors, _, gold = resolve_ordering_anchors(evidence, self.source)
        self.assertTrue(any("overlap_or_tie" in error for error in errors))
        self.assertEqual(gold, [])

    def test_nested_quotes_reject_overlap(self):
        evidence = deepcopy(self.row["item_evidence"])
        evidence["I1"]["quote"] = "The total is $350. The amount is confirmed."
        errors, _, gold = resolve_ordering_anchors(evidence, self.source)
        self.assertTrue(any("overlap_or_tie" in error for error in errors))
        self.assertEqual(gold, [])

    def test_repeated_and_overlapping_substring_occurrences_are_ambiguous(self):
        source = deepcopy(self.source)
        source["turns"][0]["text"] += " The total is $350."
        errors, _, gold = resolve_ordering_anchors(self.row["item_evidence"], source)
        self.assertIn("I2:quote_not_unique_in_turn", errors)
        self.assertEqual(gold, [])
        source["turns"][0]["text"] = "ababa " + self.source["turns"][0]["text"]
        evidence = deepcopy(self.row["item_evidence"])
        evidence["I2"]["quote"] = "aba"
        self.assertIn("I2:quote_not_unique_in_turn", resolve_ordering_anchors(evidence, source)[0])

    def test_invalid_quote_speaker_or_supplied_offset_never_get_repaired(self):
        evidence = deepcopy(self.candidate["generated_order_anchor_spans"])
        evidence["I1"].update(char_start=999, speaker="Emily")
        errors, _, gold = resolve_ordering_anchors(evidence, self.source)
        self.assertIn("I1:anchor_char_start_mismatch", errors)
        self.assertIn("I1:speaker_mismatch", errors)
        self.assertEqual(gold, [])
        evidence["I1"]["quote"] = "A fabricated quotation."
        self.assertIn("I1:quote_not_exact", resolve_ordering_anchors(evidence, self.source)[0])

    def test_independent_qwen_review_accepts_same_turn_and_keeps_raw_claims(self):
        result = self.review()
        self.assertEqual(result["rejected"], [])
        accepted = result["accepted"][0]
        self.assertEqual(accepted["gold"], self.expected)
        self.assertEqual(accepted["items"], self.row["items"])
        self.assertEqual(accepted["item_evidence"], self.row["item_evidence"])
        self.assertEqual(accepted["order_anchor"]["I1"], accepted["order_anchor"]["I2"])
        self.assertNotEqual(accepted["order_anchor_spans"]["I1"], accepted["order_anchor_spans"]["I2"])
        self.assertEqual(accepted["qwen_review"]["returned_model"], MODEL)
        self.assertFalse(accepted["provenance"]["codex_semantic_review_used"])

    def test_review_primary_span_overlap_rejects_even_when_reviewer_says_unique(self):
        self.verdict["items"]["I1"]["order_anchor"] = deepcopy(self.verdict["items"]["I2"]["order_anchor"])
        result = self.review()
        self.assertEqual(result["accepted"], [])
        self.assertTrue(any("overlap_or_tie" in r for r in result["rejected"][0]["reasons"]))

    def test_safe_same_turn_alternative_is_allowed(self):
        self.verdict["items"]["I2"]["alternative_matching_anchors"] = [self.alternative("The amount is confirmed.")]
        self.assertEqual(self.review()["accepted"][0]["gold"], self.expected)

    def test_same_turn_alternative_crossing_changes_gold_and_rejects(self):
        self.verdict["items"]["I1"]["alternative_matching_anchors"] = [self.alternative("The total is $350.")]
        reasons = self.review()["rejected"][0]["reasons"]
        self.assertIn("alternative_anchor_changes_or_ties_order:I2:I1", reasons)

    def test_joint_alternatives_cross_even_when_each_alone_is_safe(self):
        self.verdict["items"]["I2"]["alternative_matching_anchors"] = [self.alternative("Payment is planned.")]
        self.assertEqual(len(self.review()["accepted"]), 1)
        self.verdict["items"]["I1"]["alternative_matching_anchors"] = [self.alternative("The amount is confirmed.")]
        result = self.review()
        self.assertIn("alternative_anchor_changes_or_ties_order:I2:I1", result["rejected"][0]["reasons"])

    def test_alternative_quote_must_be_exact_and_unique(self):
        self.verdict["items"]["I2"]["alternative_matching_anchors"] = [self.alternative("Missing clause.")]
        self.assertIn("I2:alternative_0:quote_not_exact", self.review()["rejected"][0]["reasons"])

    def test_support_refs_never_determine_gold(self):
        self.verdict["items"]["I2"]["support_refs"] = ["T3"]
        self.assertEqual(self.review()["accepted"][0]["gold"], self.expected)

    def test_old_turn_only_review_cannot_claim_clause_verification(self):
        del self.verdict["items"]["I1"]["order_anchor"]["quote"]
        result = self.review()
        self.assertEqual(result["accepted"], [])
        self.assertIn("review_anchor_schema", result["artifact"]["errors"])

    def test_frozen_previous_round_feedback_preserved_and_wrong_model_rejected(self):
        suffix = '\n[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]\n{"guidance":"previous round"}\n[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]\n'
        self.assertEqual(len(self.review(suffix=suffix)["accepted"]), 1)
        self.assertIn("review_response_model_mismatch", self.review(model="Other/model")["artifact"]["errors"])

    def test_source_or_review_prompt_tampering_fails_binding(self):
        result = self.review(mutate_prompt=lambda p: p.replace("The total is $350.", "The total is $999."))
        self.assertIn("review_prompt_source_or_candidate_mismatch", result["artifact"]["errors"])
        self.source_path.write_text('{}')
        self.assertIn("source_file_binding_mismatch", self.review()["rejected"][0]["reasons"])

    def test_generation_prompt_explicitly_allows_same_turn_textual_order(self):
        self.assertIn("including left-to-right clause", ORDERING_GENERATION_INSTRUCTIONS)
        self.assertIn("need not be in six different turns", ORDERING_GENERATION_INSTRUCTIONS)
        self.assertIn("Make this within-turn rule clear", ORDERING_GENERATION_INSTRUCTIONS)

    def test_actual_review_payload_excludes_writer_gold_evidence_and_anchors(self):
        self.review()
        payload = json.loads((self.directory / "request.json").read_text())
        prompt = payload["messages"][0]["content"]
        data = json.loads(prompt.split("\nDATA:\n", 1)[1])
        public = data["candidates"][0]
        self.assertEqual(set(public), {"candidate_id", "root_id", "question_type",
                                      "chronology_kind", "question", "items"})
        original_prompt = build_review_prompt(self.candidate, self.source)
        perturbed = deepcopy(self.candidate)
        perturbed.update(gold=list(reversed(self.expected)), original_gold=["WRITER_GOLD_SENTINEL"],
                         item_evidence={"WRITER_EVIDENCE_SENTINEL": "hidden"},
                         generated_order_anchor_spans={"WRITER_ANCHOR_SENTINEL": "hidden"},
                         order_anchor_spans={"REVIEW_ANCHOR_SENTINEL": "hidden"},
                         qwen_review={"PRIOR_REVIEW_SENTINEL": "hidden"})
        self.assertEqual(build_review_prompt(perturbed, self.source), original_prompt)

    def test_review_proof_rejects_injected_writer_gold_in_candidate_payload(self):
        def inject(prompt):
            prefix, data_text = prompt.split("\nDATA:\n", 1)
            data = json.loads(data_text)
            data["candidates"][0]["gold"] = list(self.expected)
            return prefix + "\nDATA:\n" + json.dumps(data)
        result = self.review(mutate_prompt=inject)
        self.assertEqual(result["accepted"], [])
        self.assertIn("review_prompt_source_or_candidate_mismatch", result["artifact"]["errors"])


if __name__ == "__main__":
    unittest.main()
