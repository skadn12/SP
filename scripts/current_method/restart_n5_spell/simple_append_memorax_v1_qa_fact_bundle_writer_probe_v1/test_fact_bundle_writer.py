import unittest

from .fact_bundle_writer_contract import (
    SCHEMA_VERSION,
    materialize_batch,
    materialize_mcq,
    validate_writer_output,
)


def facts():
    return [
        {"fact_id": "F0001", "fact_text": "A planned the visit.",
         "source_turn_ids": ["root:P01:T01"]},
        {"fact_id": "F0002", "fact_text": "A chose Saturday morning.",
         "source_turn_ids": ["root:P01:T02"]},
        {"fact_id": "F0003", "fact_text": "B brought a notebook.",
         "source_turn_ids": ["root:C01:T01"]},
        {"fact_id": "F0004", "fact_text": "They tested a glaze.",
         "source_turn_ids": ["root:C01:T02"]},
    ]


def bundles():
    return [{
        "bundle_id": "B0001", "fact_ids": ["F0001", "F0002", "F0003", "F0004"],
        "bundle_class": "parent_child",
    }]


def row():
    return {
        "bundle_id": "B0001", "question_type": "temporal_sequence",
        "question": "What happened after the visit was planned?",
        "answer_dimension": "order", "gold_fact_id": "F0003",
        "distractor_fact_ids": ["F0001", "F0002", "F0004"],
    }


class FactBundleWriterTests(unittest.TestCase):
    def test_valid_reference_only_row(self):
        value = {"fact_bundle_writer_schema_version": SCHEMA_VERSION, "questions": [row()]}
        result = validate_writer_output(value, facts=facts(), bundles=bundles())
        self.assertTrue(result["valid"], result)
        item = materialize_mcq(row(), facts=facts(), bundles=bundles())
        self.assertEqual(set(item["options"].values()), {
            "A planned the visit.", "A chose Saturday morning.",
            "B brought a notebook.", "They tested a glaze.",
        })
        self.assertIn("root:C01:T01", item["evidence_turn_refs"])
        self.assertNotIn("gold_fact_id", item)
        self.assertNotIn("fact_bundle_id", item)
        self.assertNotIn("option_fact_ids", item)

    def test_unknown_or_outside_bundle_fact_is_rejected(self):
        bad = {**row(), "gold_fact_id": "F9999"}
        result = validate_writer_output({
            "fact_bundle_writer_schema_version": SCHEMA_VERSION,
            "questions": [bad],
        }, facts=facts(), bundles=bundles())
        self.assertFalse(result["valid"])
        self.assertIn("gold_fact_not_in_bundle", result["items"][0]["errors"])

    def test_duplicate_gold_distractor_is_rejected(self):
        bad = {**row(), "distractor_fact_ids": ["F0003", "F0002", "F0004"]}
        result = validate_writer_output({
            "fact_bundle_writer_schema_version": SCHEMA_VERSION,
            "questions": [bad],
        }, facts=facts(), bundles=bundles())
        self.assertFalse(result["valid"])
        self.assertIn("gold_repeated_as_distractor", result["items"][0]["errors"])

    def test_empty_and_variable_size_batches(self):
        for count in (0, 1, 3):
            value = {"fact_bundle_writer_schema_version": SCHEMA_VERSION,
                     "questions": [row()] * count}
            result = validate_writer_output(value, facts=facts(), bundles=bundles())
            self.assertTrue(result["valid"], (count, result))

    def test_model_cannot_supply_option_text_or_positions(self):
        bad = {**row(), "options": {"A": "invented"}}
        result = validate_writer_output({
            "fact_bundle_writer_schema_version": SCHEMA_VERSION,
            "questions": [bad],
        }, facts=facts(), bundles=bundles())
        self.assertFalse(result["valid"])
        self.assertIn("question_schema", result["items"][0]["errors"])


if __name__ == "__main__":
    unittest.main()
