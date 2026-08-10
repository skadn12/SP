import copy
import unittest

from scripts.current_method.restart_n5_spell.reference_exemplar_relation_first import (
    RELATION_PROMPT_VERSION,
    build_relation_miner_prompt,
    normalize_relation_batch,
    relation_prompt_sha256,
    relation_schema_sha256,
    validate_relation_batch_schema,
)
from scripts.current_method.restart_n5_spell.run_reference_exemplar_relation_first_canary import (
    DEFAULT_EXEMPLARS,
)
from scripts.current_method.restart_n5_spell.run_questioner_joint_reference_ab_canary import (
    DEFAULT_REFERENCE,
)
from scripts.current_method.restart_n5_spell.run_reference_guided_relation_first_canary import (
    DEFAULT_ALIAS_MAP,
    DEFAULT_DIALOGUE,
    _source_inputs,
)


class ReferenceExemplarRelationFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _source_inputs(DEFAULT_DIALOGUE, DEFAULT_ALIAS_MAP, DEFAULT_REFERENCE)
        cls.aliases = cls.source["aliases"]
        cls.exemplars = DEFAULT_EXEMPLARS.read_text(encoding="utf-8")

    def valid_batch(self):
        relations = []
        for index in range(1, 5):
            relations.append({
                "relation_id": f"R{index}",
                "status": "valid",
                "reasoning_type": "background_requirement_matching",
                "parent_turn_aliases": [f"P{index:03d}"],
                "child_turn_aliases": [f"C{index:03d}"],
                "parent_unique_fact": f"The parent establishes condition {index}.",
                "child_unique_fact": f"The child establishes requirement {index}.",
                "combined_conclusion": f"Person {index} satisfies requirement {index}.",
                "missing_from_parent_only": f"Requirement {index} is absent.",
                "missing_from_child_only": f"Condition {index} is absent.",
            })
        return {"relations": relations}

    def test_schema_requires_exactly_four_slots(self):
        payload = self.valid_batch()
        self.assertEqual(validate_relation_batch_schema(payload)["decision"],
                         "relation_batch_accept")
        payload["relations"].pop()
        self.assertEqual(validate_relation_batch_schema(payload)["decision"],
                         "relation_batch_schema_invalid")

    def test_no_valid_relation_is_allowed(self):
        payload = self.valid_batch()
        payload["relations"][3] = {
            "relation_id": "R4",
            "status": "no_valid_relation",
            "reason": "No fourth nonredundant joint relation was found.",
        }
        self.assertEqual(validate_relation_batch_schema(payload)["decision"],
                         "relation_batch_accept")
        normalized = normalize_relation_batch(payload, self.aliases)
        self.assertEqual(normalized["empty_relation_count"], 1)

    def test_model_quote_fields_are_forbidden(self):
        payload = self.valid_batch()
        payload["relations"][0]["parent_quotes"] = ["model supplied quote"]
        self.assertEqual(validate_relation_batch_schema(payload)["decision"],
                         "relation_batch_schema_invalid")

    def test_aliases_resolve_to_complete_frozen_text(self):
        normalized = normalize_relation_batch(self.valid_batch(), self.aliases)
        row = normalized["relations"][0]
        self.assertEqual(row["parent_quotes"], [self.aliases["P001"]["text"]])
        self.assertEqual(row["child_quotes"], [self.aliases["C001"]["text"]])
        self.assertEqual(row["local_validation"], "accept")

    def test_wrong_partition_is_rejected(self):
        payload = self.valid_batch()
        payload["relations"][0]["parent_turn_aliases"] = ["C001"]
        normalized = normalize_relation_batch(payload, self.aliases)
        self.assertIn("wrong_partition:C001",
                      normalized["relations"][0]["local_validation_errors"])

    def test_duplicate_support_bundle_is_rejected(self):
        payload = self.valid_batch()
        payload["relations"][1]["parent_turn_aliases"] = ["P001"]
        payload["relations"][1]["child_turn_aliases"] = ["C001"]
        normalized = normalize_relation_batch(payload, self.aliases)
        self.assertEqual(normalized["relations"][0]["local_validation"], "accept")
        self.assertIn("duplicate_relation_support",
                      normalized["relations"][1]["local_validation_errors"])

    def test_vague_conclusion_blacklist_is_deterministic(self):
        for phrase in (
            "The parent aligns with the child.",
            "The parent provides a foundation for the later work.",
            "Both emphasize community service.",
        ):
            payload = self.valid_batch()
            payload["relations"][0]["combined_conclusion"] = phrase
            normalized = normalize_relation_batch(payload, self.aliases)
            self.assertIn("vague_combined_conclusion",
                          normalized["relations"][0]["local_validation_errors"])

    def test_concrete_conclusion_is_not_rejected(self):
        payload = self.valid_batch()
        payload["relations"][0]["combined_conclusion"] = (
            "John satisfies the position's three-year community-service requirement."
        )
        normalized = normalize_relation_batch(payload, self.aliases)
        self.assertEqual(normalized["relations"][0]["local_validation"], "accept")

    def test_prompt_contains_real_demonstrations_and_alias_only_contract(self):
        prompt = build_relation_miner_prompt(self.source["turns"], self.exemplars)
        self.assertIn(RELATION_PROMPT_VERSION, prompt)
        self.assertIn("Positive 1: background_requirement_matching", prompt)
        self.assertIn("Positive 2: prior_evidence_later_choice", prompt)
        self.assertIn("Positive 3: source_verification_synthesis", prompt)
        self.assertIn("Negative: child_already_contains_complete_conclusion", prompt)
        self.assertIn("Return aliases only", prompt)
        self.assertIn("Return exactly four relation slots R1, R2, R3, R4", prompt)
        self.assertIn("[P001]", prompt)
        self.assertIn("[C134]", prompt)
        self.assertNotIn('"parent_quotes"', prompt.split("OUTPUT JSON SCHEMA", 1)[1])

    def test_contract_shas_are_stable(self):
        self.assertEqual(relation_prompt_sha256(), relation_prompt_sha256())
        self.assertEqual(relation_schema_sha256(), relation_schema_sha256())
        self.assertEqual(len(relation_prompt_sha256()), 64)
        self.assertEqual(len(relation_schema_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
