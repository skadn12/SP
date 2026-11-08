import copy
import json
import unittest

import dataset_cards as dc


class DatasetCardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = dc.freeze_source_subset()

    def seed(self):
        item = self.manifest["items"][0]
        unit = next(unit for unit in item["units"] if unit["kind"] == "dialogue")
        return {
            "card_id": "pilot_seed", "source_id": item["source_id"], "guidance_level": "seed",
            "actors": [{"actor_id": "actor_1", "source_speaker": unit["speaker"], "role": "service provider"}],
            "steps": [{"step_id": "s1", "actor_id": "actor_1", "action": "actor_1 offers a service",
                       "state": "proposed", "evidence": [{"unit_id": unit["unit_id"], "quote": unit["text"]}]}],
            "dependencies": [], "confusion_axes": [], "open_ends": ["Service delivery has not occurred."],
            "downgrade_reason": "Only a service offer is retained; subsequent developments need realization.",
        }

    def validate(self, card):
        return dc.validate_extracted_cards({"schema_version": dc.SCHEMA_VERSION, "cards": [card]}, self.manifest)

    def test_frozen_six_train_inputs_and_request(self):
        self.assertEqual(len(self.manifest["items"]), 6)
        self.assertEqual([item["dataset"] for item in self.manifest["items"]].count("allenai/soda"), 3)
        self.assertTrue(all(item["split"] == "train" for item in self.manifest["items"]))
        self.assertTrue(all(not source["revision_pinned"] for source in self.manifest["files"]))
        self.assertEqual(self.manifest, dc.freeze_source_subset())
        request = dc.build_extraction_request(self.manifest)
        self.assertEqual(request["model"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(len(request["messages"]), 1)
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertIn(self.manifest["manifest_sha256"], request["messages"][0]["content"])

    def test_seed_downgrade_is_valid(self):
        result = self.validate(self.seed())
        self.assertEqual(result["rejected_records"], [])
        self.assertEqual(len(result["accepted_cards"]), 1)
        card = result["accepted_cards"][0]
        ref = card["steps"][0]["evidence"][0]
        self.assertEqual(ref["kind"], "dialogue")
        self.assertEqual(ref["char_start"], 0)
        guidance = dc.build_realization_guidance(card)
        self.assertNotIn("evidence", guidance["steps"][0])
        self.assertNotIn("source_speaker", guidance["actors"][0])
        self.assertFalse(guidance["source_is_root_evidence"])

    def test_batches_bound_to_specific_sources(self):
        source_ids = [item["source_id"] for item in self.manifest["items"][:3]]
        batch = dc.select_source_subset(self.manifest, source_ids)
        request = dc.build_extraction_request(self.manifest, source_ids)
        content = json.loads(request["messages"][0]["content"].split("SOURCE_ITEMS_JSON:\n", 1)[1])
        self.assertEqual(content["source_manifest_sha256"], batch["manifest_sha256"])
        self.assertEqual([item["source_id"] for item in content["source_items"]], source_ids)
        result = dc.validate_extracted_cards({"schema_version": dc.SCHEMA_VERSION, "cards": [self.seed()]}, self.manifest, source_ids)
        self.assertEqual(result["source_manifest_sha256"], batch["manifest_sha256"])
        self.assertEqual(len(result["missing_source_ids"]), 2)
        other_source_ids = [item["source_id"] for item in self.manifest["items"][3:]]
        wrong_batch = dc.validate_extracted_cards({"schema_version": dc.SCHEMA_VERSION, "cards": [self.seed()]}, self.manifest, other_source_ids)
        self.assertEqual(wrong_batch["accepted_cards"], [])

    def test_hallucinated_quote_is_rejected(self):
        card = self.seed()
        card["steps"][0]["evidence"][0]["quote"] = "The pet care was completed successfully yesterday."
        result = self.validate(card)
        self.assertEqual(result["accepted_cards"], [])
        self.assertIn("not exact", result["rejected_records"][0]["reason"])

    def test_out_of_span_reference_is_rejected(self):
        card = self.seed()
        card["steps"][0]["evidence"][0]["unit_id"] = "dialogue:99"
        result = self.validate(card)
        self.assertEqual(result["accepted_cards"], [])
        self.assertIn("unknown source unit", result["rejected_records"][0]["reason"])

    def test_unknown_source_is_rejected(self):
        card = self.seed()
        card["source_id"] = "soda:test:invented"
        self.assertEqual(self.validate(card)["accepted_cards"], [])

    def test_single_stage_scaffold_is_not_silently_downgraded(self):
        card = self.seed()
        card["guidance_level"] = "scaffold"
        self.assertEqual(self.validate(card)["accepted_cards"], [])

    def test_unknown_dependency_is_rejected(self):
        card = self.seed()
        card["dependencies"] = [{"before": "s1", "after": "s7", "relation": "retry", "evidence": copy.deepcopy(card["steps"][0]["evidence"])}]
        self.assertEqual(self.validate(card)["accepted_cards"], [])

    def test_duplicate_source_rejects_both(self):
        first = self.seed()
        second = copy.deepcopy(first)
        second["card_id"] = "duplicate_seed"
        result = dc.validate_extracted_cards({"schema_version": dc.SCHEMA_VERSION, "cards": [first, second]}, self.manifest)
        self.assertEqual(result["accepted_cards"], [])
        self.assertEqual(len(result["rejected_records"]), 2)

    def test_invalid_json_and_extra_fields_rejected(self):
        self.assertEqual(dc.validate_extracted_cards("```json\n{}\n```", self.manifest)["accepted_cards"], [])
        card = self.seed()
        card["hallucinated_source_fact"] = "result"
        self.assertEqual(self.validate(card)["accepted_cards"], [])

    def test_nonhashable_fields_return_rejection(self):
        for field in ("source_id", "card_id", "guidance_level"):
            with self.subTest(field=field):
                card = self.seed()
                card[field] = ["bad"]
                self.assertEqual(self.validate(card)["accepted_cards"], [])
        for field in ("step_id", "actor_id", "state"):
            with self.subTest(step_field=field):
                card = self.seed()
                card["steps"][0][field] = {"bad": "bad"}
                self.assertEqual(self.validate(card)["accepted_cards"], [])
        card = self.seed()
        card["steps"][0]["evidence"][0]["unit_id"] = []
        self.assertEqual(self.validate(card)["accepted_cards"], [])


if __name__ == "__main__":
    unittest.main()
