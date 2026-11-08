from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ordering_gate import LABELS, MODEL, build_review_prompt, review_response, validate_generated_ordering


class OrderingGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.source = {"root_id": "root", "turns": [
            {"turn_ref": f"R00:T{i:03d}", "speaker": "Emily", "text": f"I report event number {i}."}
            for i in range(30)]}
        self.source_path = self.directory / "source.json"
        self.source_path.write_text(json.dumps(self.source))
        positions = [3, 7, 11, 15, 19, 23]
        self.row = {"candidate_id": "q1", "root_id": "root", "question_type": "ordering",
                    "chronology_kind": "conversation_order", "question": "In what order do the statements appear in the supplied dialogue?",
                    "items": {label: f"Emily reports event number {pos}." for label, pos in zip(LABELS, positions)},
                    "gold": list(reversed(LABELS)), "item_evidence": {
                        label: {"turn_ref": self.source["turns"][pos]["turn_ref"], "speaker": "Emily",
                                "quote": self.source["turns"][pos]["text"], "status": "report",
                                "reason": "A source utterance about this event."} for label, pos in zip(LABELS, positions)}}
        errors, self.candidate = validate_generated_ordering(self.row, "q1", "root", self.source, self.source_path)
        self.assertEqual(errors, [])
        self.verdict = {"candidate_id": "q1", "items": {
            label: {"supported": True, "order_anchor": {"turn_ref": self.row["item_evidence"][label]["turn_ref"], "reason": "Primary utterance."},
                    "support_refs": [], "alternative_matching_turn_refs": [], "actor_ok": True,
                    "modality_ok": True, "state_ok": True, "neutral_stem_ok": True, "reason": "Supported."} for label in LABELS},
            "coherent": True, "unique_order": True, "alternative_orders": [], "neutral_stem_ok": True, "reason": "One order."}

    def review(self, verdict=None, metadata_patch=None, prompt_suffix="", request_mutator=None):
        raw = json.dumps({"reviews": [self.verdict if verdict is None else verdict]})
        metadata = {"requested_model": MODEL, "returned_model": MODEL, "finish_reason": "stop", "http_status": 200,
                    "raw_response": {"id": "response", "model": MODEL, "choices": [{"finish_reason": "stop", "message": {"content": raw}}]}}
        metadata.update(metadata_patch or {})
        payload = {"model": MODEL, "messages": [{"role": "user", "content": build_review_prompt(self.candidate, self.source) + prompt_suffix}]}
        if request_mutator:
            request_mutator(payload)
        request_path, response_path = self.directory / "request.json", self.directory / "response.json"
        request_path.write_text(json.dumps(payload))
        response_path.write_text(json.dumps({"text": raw, "metadata": metadata}))
        return review_response(raw, [self.candidate], self.source, response_path, request_path, metadata)

    def test_generation_normalizes_only_gold_without_round_or_position_filter(self):
        self.assertEqual(self.candidate["gold"], list(LABELS))
        self.assertEqual(self.candidate["original_gold"], list(reversed(LABELS)))
        self.assertEqual(self.candidate["items"], self.row["items"])
        self.assertEqual(self.row["gold"], list(reversed(LABELS)))
        self.assertTrue(all(e["turn_ref"].startswith("R00:") for e in self.candidate["item_evidence"].values()))

    def test_invalid_quote_and_speaker_reject_without_rewriting(self):
        row = deepcopy(self.row)
        row["item_evidence"]["I1"].update(quote="Not in the source", speaker="Someone else")
        errors, candidate = validate_generated_ordering(row, "q1", "root", self.source, self.source_path)
        self.assertEqual(errors, ["I1:quote_not_exact", "I1:speaker_mismatch"])
        self.assertIsNone(candidate)

    def test_missing_reason_bad_status_and_wrong_scope_rejected(self):
        row = deepcopy(self.row)
        row["question"] = "What is the real-world chronology?"
        row["item_evidence"]["I1"].update(reason=" ", status="unsupported-status")
        errors, _ = validate_generated_ordering(row, "q1", "root", self.source, self.source_path)
        self.assertIn("I1:evidence_reason_missing", errors)
        self.assertIn("I1:evidence_status_invalid", errors)
        self.assertIn("question_conversation_scope_missing", errors)

    def test_json_label_key_order_normalized_without_content_rebinding(self):
        row = deepcopy(self.row)
        row["items"] = dict(reversed(list(row["items"].items())))
        errors, candidate = validate_generated_ordering(row, "q1", "root", self.source, self.source_path)
        self.assertEqual(errors, [])
        self.assertEqual(tuple(candidate["items"]), LABELS)
        self.assertEqual(candidate["items"], row["items"])

    def test_duplicate_items_anchors_and_label_hint_reject(self):
        row = deepcopy(self.row)
        row["items"]["I2"] = row["items"]["I1"]
        row["item_evidence"]["I2"] = deepcopy(row["item_evidence"]["I1"])
        row["question"] += " I1 then I2."
        errors, _ = validate_generated_ordering(row, "q1", "root", self.source, self.source_path)
        self.assertIn("duplicate_item_text", errors)
        self.assertIn("order_anchors_not_distinct", errors)
        self.assertIn("question_contains_label_or_source_position_hint", errors)

    def test_pass_has_actual_proof_and_preserves_text_and_original_gold(self):
        result = self.review()
        self.assertEqual(result["rejected"], [])
        row = result["accepted"][0]
        self.assertEqual(row["items"], self.row["items"])
        self.assertEqual(row["item_evidence"], self.row["item_evidence"])
        self.assertEqual(row["original_gold"], self.row["gold"])
        self.assertEqual(row["gold"], list(LABELS))
        self.assertEqual(row["qwen_review"]["returned_model"], MODEL)
        self.assertEqual(len(row["qwen_review"]["response_sha256"]), 64)

    def test_safe_alternative_preserving_order_is_allowed(self):
        self.verdict["items"]["I1"]["alternative_matching_turn_refs"] = ["R00:T004"]
        self.verdict["alternative_orders"] = [list(LABELS)]
        self.assertEqual(len(self.review()["accepted"]), 1)

    def test_alternative_changes_order_rejected_even_if_reviewer_says_unique(self):
        self.verdict["items"]["I1"]["alternative_matching_turn_refs"] = ["R00:T008"]
        result = self.review()
        self.assertEqual(result["accepted"], [])
        self.assertIn("alternative_anchor_changes_or_ties_order:I1:I2", result["rejected"][0]["reasons"])

    def test_simultaneous_alternatives_crossing_rejected(self):
        self.verdict["items"]["I1"]["alternative_matching_turn_refs"] = ["R00:T006"]
        self.verdict["items"]["I2"]["alternative_matching_turn_refs"] = ["R00:T004"]
        self.assertIn("alternative_anchor_changes_or_ties_order:I1:I2", self.review()["rejected"][0]["reasons"])

    def test_contextual_support_never_changes_order(self):
        self.verdict["items"]["I1"]["support_refs"] = ["R00:T029"]
        self.assertEqual(self.review()["accepted"][0]["gold"], list(LABELS))

    def test_verified_anchor_can_recompute_gold_but_never_rewrite_claim(self):
        self.verdict["items"]["I1"]["order_anchor"]["turn_ref"] = "R00:T009"
        result = self.review()
        self.assertEqual(result["accepted"][0]["gold"], ["I2", "I1", "I3", "I4", "I5", "I6"])
        self.assertEqual(result["accepted"][0]["items"], self.row["items"])

    def test_wrong_model_and_truncated_envelope_reject(self):
        wrong = self.review(metadata_patch={"returned_model": "Different/model"})
        self.assertIn("review_response_model_mismatch", wrong["artifact"]["errors"])
        truncated = self.review(metadata_patch={"raw_response": {"model": MODEL, "choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}})
        self.assertIn("review_incomplete_response", truncated["artifact"]["errors"])
        self.assertEqual(truncated["accepted"], [])

    def test_malformed_review_fails_closed(self):
        verdict = deepcopy(self.verdict)
        verdict["items"] = 42
        result = self.review(verdict=verdict)
        self.assertEqual(result["accepted"], [])
        self.assertIn("review_items_schema", result["artifact"]["errors"])

    def test_previous_round_feedback_suffix_allowed_but_unknown_suffix_rejected(self):
        suffix = '\n[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]\n{"guidance": "previous round"}\n[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]\n'
        self.assertEqual(len(self.review(prompt_suffix=suffix)["accepted"]), 1)
        result = self.review(prompt_suffix="\nUntracked instruction")
        self.assertIn("review_unrecognized_prompt_suffix", result["artifact"]["errors"])

    def test_request_source_tampering_fails_closed(self):
        def mutate(payload):
            payload["messages"][0]["content"] = payload["messages"][0]["content"].replace('I report event number 0.', 'Altered transcript.')
        result = self.review(request_mutator=mutate)
        self.assertIn("review_prompt_source_or_candidate_mismatch", result["artifact"]["errors"])

    def test_source_changed_after_generation_rejects(self):
        self.source_path.write_text('{}')
        result = self.review()
        self.assertEqual(result["accepted"], [])
        self.assertIn("source_file_binding_mismatch", result["rejected"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
