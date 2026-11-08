from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import card_selection as cs


V4_RUN = (Path(__file__).resolve().parent.parent / "h4_r02_independent_routes_v4" /
          "runs" / "r02_independent_routes_v4_01")


class CardSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = cs._legacy.freeze_source_subset()
        cls.item = cls.manifest["items"][0]
        cls.subset = cs._legacy.select_source_subset(cls.manifest, [cls.item["source_id"]])

    def seed(self, source_id=None):
        item = next(item for item in self.subset["items"] if item["source_id"] == source_id) if source_id else self.item
        unit = next(unit for unit in item["units"] if unit["kind"] == "dialogue")
        return {
            "card_id": "pilot_seed", "source_id": item["source_id"], "guidance_level": "seed",
            "actors": [{"actor_id": "actor_1", "source_speaker": unit["speaker"], "role": "service provider"}],
            "steps": [{"step_id": "s1", "actor_id": "actor_1", "action": "actor_1 offers a service",
                       "state": "proposed", "evidence": [{"unit_id": unit["unit_id"], "quote": unit["text"]}]}],
            "dependencies": [], "confusion_axes": [], "open_ends": ["Service delivery has not occurred."],
            "downgrade_reason": "Only a service offer is retained; later developments need realization.",
        }

    def response(self, cards):
        return {"schema_version": cs.SCHEMA_VERSION, "cards": cards}

    def test_request_is_single_source_and_keeps_qwen_config(self):
        request = cs.build_extraction_request(self.subset)
        self.assertEqual(request["model"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(request["temperature"], 0.25)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertFalse(request["enable_thinking"])
        self.assertEqual(request["response_format"], {"type": "json_object"})
        prompt = request["messages"][0]["content"]
        self.assertIn("at most ONE event", prompt)
        self.assertIn("zero or one card only", prompt)
        with self.assertRaisesRegex(ValueError, "single_source_required"):
            cs.build_extraction_request(self.manifest)

    def test_repair_feedback_is_bound_to_same_source(self):
        feedback = {"source_id": self.item["source_id"], "candidate_errors": [
            {"candidate_index": 0, "error_code": "EVIDENCE_INVALID", "reason": "quote is not exact"}
        ]}
        prompt = cs.build_extraction_request(self.subset, repair=feedback)["messages"][0]["content"]
        self.assertIn("REPAIR_FEEDBACK_JSON", prompt)
        self.assertIn("quote is not exact", prompt)
        self.assertIn(self.item["source_id"], prompt)

    def test_normal_single_card_is_accepted_without_retry(self):
        result = cs.select_cards(self.response([self.seed()]), self.subset)
        self.assertEqual(len(result["accepted_cards"]), 1)
        self.assertFalse(result["retryable"])
        self.assertEqual(result["missing_source_ids"], [])
        self.assertEqual(result["candidate_results"][0]["status"], "selected")

    def test_bad_quote_candidate_does_not_block_later_valid_candidate(self):
        invalid = self.seed()
        invalid["card_id"] = "bad_quote"
        invalid["steps"][0]["evidence"][0]["quote"] = "not in the source"
        valid = self.seed()
        valid["card_id"] = "later_valid"
        result = cs.select_cards(self.response([invalid, valid]), self.subset)
        self.assertEqual([card["card_id"] for card in result["accepted_cards"]], ["later_valid"])
        self.assertEqual(result["candidate_results"][0]["status"], "invalid")
        self.assertEqual(result["candidate_results"][1]["status"], "selected")
        self.assertFalse(result["retryable"])
        self.assertEqual(len(result["repair_feedback"]["candidate_errors"]), 1)

    def test_first_valid_candidate_wins_and_later_valid_is_not_reusable(self):
        first = self.seed()
        second = self.seed()
        second["card_id"] = "second"
        result = cs.select_cards(self.response([first, second]), self.subset)
        self.assertEqual([card["card_id"] for card in result["accepted_cards"]], ["pilot_seed"])
        self.assertEqual(result["candidate_results"][1]["status"], "valid_unselected_never_reuse_source")
        self.assertEqual(result["rejected_records"][0]["reason"], "valid_unselected_never_reuse_source")

    def test_unknown_source_is_rejected_and_retryable(self):
        card = self.seed()
        card["source_id"] = "multiwoz22:train:invented.json"
        result = cs.select_cards(self.response([card]), self.subset)
        self.assertEqual(result["accepted_cards"], [])
        self.assertTrue(result["retryable"])
        self.assertEqual(result["rejected_records"][0]["error_code"], "UNKNOWN_SOURCE")

    def test_empty_cards_is_intentional_omission_not_retryable(self):
        result = cs.select_cards(self.response([]), self.subset)
        self.assertEqual(result["accepted_cards"], [])
        self.assertEqual(result["missing_source_ids"], [self.item["source_id"]])
        self.assertFalse(result["retryable"])
        self.assertEqual(result["repair_feedback"]["candidate_errors"], [])

    def test_malformed_and_duplicate_json_keys_fail_closed(self):
        malformed = cs.select_cards("{not json", self.subset)
        self.assertEqual(malformed["accepted_cards"], [])
        self.assertTrue(malformed["retryable"])
        duplicated = cs.select_cards(
            '{"schema_version":"h4.r02.dataset_cards.v1","schema_version":"bad","cards":[]}',
            self.subset,
        )
        self.assertEqual(duplicated["accepted_cards"], [])
        self.assertIn("duplicate_json_key", duplicated["rejected_records"][0]["reason"])
        self.assertTrue(duplicated["retryable"])

    def test_wrong_top_level_schema_and_non_list_cards_rejected(self):
        wrong_schema = cs.select_cards({"schema_version": "wrong", "cards": []}, self.subset)
        self.assertTrue(wrong_schema["retryable"])
        non_list = cs.select_cards({"schema_version": cs.SCHEMA_VERSION, "cards": {}}, self.subset)
        self.assertEqual(non_list["accepted_cards"], [])
        self.assertTrue(non_list["retryable"])

    def test_real_v4_nine_extraction_responses_offline_replay(self):
        with (V4_RUN / "dataset_sources.json").open(encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        response_paths = sorted((V4_RUN / "requests").glob("*-extract-*/RESPONSE.json"))
        self.assertEqual(len(response_paths), 9)
        old_accepted = 0
        old_rejection_reasons = Counter()
        selected = 0
        multi_sources = 0
        multi_cards_seen = 0
        multi_selected = 0
        multi_valid_candidates = 0
        failure_reasons = Counter()
        for response_path in response_paths:
            old_validation_path = response_path.with_name("VALIDATION.json")
            old_validation = json.loads(old_validation_path.read_text(encoding="utf-8"))
            old_accepted += len(old_validation.get("accepted_cards", []))
            for record in old_validation.get("rejected_records", []):
                old_rejection_reasons[record.get("reason", "unknown")] += 1
            with response_path.open(encoding="utf-8") as handle:
                raw_response = json.load(handle)["text"]
            parsed = json.loads(raw_response)
            cards = parsed["cards"]
            if len(cards) > 1:
                multi_sources += 1
                multi_cards_seen += len(cards)
            source_id = cards[0]["source_id"] if cards else None
            if source_id is None:
                request_path = response_path.with_name("REQUEST.json")
                request = json.loads(request_path.read_text(encoding="utf-8"))
                marker = "SOURCE_ITEMS_JSON:\n"
                source_payload = json.loads(request["messages"][0]["content"].split(marker, 1)[1])
                source_id = source_payload["source_items"][0]["source_id"]
            subset = cs._legacy.select_source_subset(source_manifest, [source_id])
            result = cs.select_cards(raw_response, subset)
            selected += len(result["accepted_cards"])
            if len(cards) > 1:
                multi_selected += len(result["accepted_cards"])
                multi_valid_candidates += sum(
                    candidate["status"] in {"selected", "valid_unselected_never_reuse_source"}
                    for candidate in result["candidate_results"]
                )
            for record in result["rejected_records"]:
                failure_reasons[record.get("reason", "unknown")] += 1
        print("V4_CARD_REPLAY " + json.dumps({
            "responses": len(response_paths), "old_v1_accepted_sources": old_accepted,
            "old_v1_rejection_reasons": dict(old_rejection_reasons),
            "new_accepted_sources": selected,
            "multi_candidate_sources": multi_sources, "multi_candidate_cards_seen": multi_cards_seen,
            "multi_candidate_sources_recovered": multi_selected,
            "valid_cards_among_multi_candidates": multi_valid_candidates,
            "remaining_rejection_reasons": dict(failure_reasons),
        }, ensure_ascii=True, sort_keys=True))
        self.assertEqual(multi_sources, 4)
        self.assertEqual(multi_cards_seen, 8)
        self.assertEqual(multi_selected, 4)
        self.assertEqual(multi_valid_candidates, 8)
        self.assertEqual(selected, 9)
        self.assertEqual(old_accepted, 5)
        self.assertEqual(old_rejection_reasons, {"card: duplicate source_id": 8})
        self.assertEqual(failure_reasons, {"valid_unselected_never_reuse_source": 4})


if __name__ == "__main__":
    unittest.main()
