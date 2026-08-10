import copy
import json
import unittest

from scripts.current_method.restart_n5_spell.reference_guided_relation_first import (
    QA_AUDIT_VERSION,
    RELATION_AUDIT_VERSION,
    build_qa_writer_prompt,
    derive_qa_audit_decision,
    derive_relation_audit_decision,
    extract_relation_reference,
    freeze_relation,
    normalize_relation_batch,
    route_conclusion,
    select_stage2_relations,
    validate_qa_writer_object,
    validate_relation_batch_schema,
    validate_stage1_audit,
    validate_stage2_audit,
)
from scripts.current_method.restart_n5_spell.run_questioner_joint_reference_ab_canary import (
    DEFAULT_REFERENCE,
)
from scripts.current_method.restart_n5_spell.run_reference_guided_relation_first_canary import (
    DEFAULT_ALIAS_MAP,
    DEFAULT_DIALOGUE,
    _source_inputs,
)
from scripts.current_method.restart_n5_spell.core import canonical_json, sha256_bytes


class ReferenceGuidedRelationFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _source_inputs(DEFAULT_DIALOGUE, DEFAULT_ALIAS_MAP, DEFAULT_REFERENCE)
        cls.aliases = cls.source["aliases"]

    def valid_batch(self):
        relations = []
        for index in range(1, 9):
            pa = f"P{index:03d}"
            ca = f"C{index:03d}"
            relations.append({
                "relation_id": f"R{index}", "status": "valid",
                "reasoning_type": "test_relation",
                "parent_turn_aliases": [pa],
                "parent_quotes": [self.aliases[pa]["text"]],
                "child_turn_aliases": [ca],
                "child_quotes": [self.aliases[ca]["text"]],
                "parent_unique_fact": f"parent fact {index}",
                "child_unique_fact": f"child fact {index}",
                "combined_conclusion": f"combined {index}",
                "missing_from_parent_only": f"child fact {index}",
                "missing_from_child_only": f"parent fact {index}",
            })
        return {"relations": relations}

    def test_relation_reference_excludes_question_writing_sections(self):
        text = extract_relation_reference(DEFAULT_REFERENCE.read_text(encoding="utf-8"))
        self.assertIn("SINGLE-SIDE ABLATION", text)
        self.assertIn("RELATION-TYPE GUIDANCE", text)
        self.assertNotIn("### Step 4: Write the question", text)
        self.assertNotIn("## 3. Positive examples", text)
        self.assertNotIn('"question":', text)

    def test_relation_schema_requires_exactly_eight_slots(self):
        valid = self.valid_batch()
        self.assertEqual(validate_relation_batch_schema(valid)["decision"],
                         "relation_batch_accept")
        valid["relations"].pop()
        self.assertEqual(validate_relation_batch_schema(valid)["decision"],
                         "relation_batch_schema_invalid")

    def test_no_valid_relation_slot_is_allowed(self):
        payload = self.valid_batch()
        payload["relations"][7] = {
            "relation_id": "R8", "status": "no_valid_relation",
            "reason": "No eighth relation survived ablation.",
        }
        self.assertEqual(validate_relation_batch_schema(payload)["decision"],
                         "relation_batch_accept")
        normalized = normalize_relation_batch(payload, self.aliases)
        self.assertEqual(normalized["empty_relation_count"], 1)

    def test_alias_partition_and_exact_quote_validation(self):
        payload = self.valid_batch()
        payload["relations"][0]["parent_quotes"] = ["not the frozen text"]
        normalized = normalize_relation_batch(payload, self.aliases)
        self.assertEqual(normalized["relations"][0]["local_validation"], "reject")
        self.assertIn("quote_mismatch:P001",
                      normalized["relations"][0]["local_validation_errors"])
        payload = self.valid_batch()
        payload["relations"][0]["parent_turn_aliases"] = ["C001"]
        payload["relations"][0]["parent_quotes"] = [self.aliases["C001"]["text"]]
        normalized = normalize_relation_batch(payload, self.aliases)
        self.assertIn("wrong_partition:C001",
                      normalized["relations"][0]["local_validation_errors"])

    def test_duplicate_support_is_rejected_without_local_rewrite(self):
        payload = self.valid_batch()
        payload["relations"][1]["parent_turn_aliases"] = ["P001"]
        payload["relations"][1]["parent_quotes"] = [self.aliases["P001"]["text"]]
        payload["relations"][1]["child_turn_aliases"] = ["C001"]
        payload["relations"][1]["child_quotes"] = [self.aliases["C001"]["text"]]
        normalized = normalize_relation_batch(payload, self.aliases)
        self.assertIn("duplicate_relation_support",
                      normalized["relations"][1]["local_validation_errors"])

    def test_stage1_audit_and_first_four_selection(self):
        normalized = normalize_relation_batch(self.valid_batch(), self.aliases)
        source_sha = sha256_bytes(canonical_json(normalized).encode())
        items = []
        for row in normalized["relations"]:
            items.append({
                "relation_id": row["relation_id"],
                "parent_only_can_derive_combined_conclusion": False,
                "child_only_can_derive_combined_conclusion": False,
                "combined_can_derive_combined_conclusion": True,
                "child_fully_restates_parent": False,
                "evidence_redundant": False,
                "natural_relation": True,
                "unsupported_causal_claim": False,
                "notes": "",
            })
        audit = {"audit_version": RELATION_AUDIT_VERSION,
                 "source_normalized_relations_sha256": source_sha, "items": items}
        validated = validate_stage1_audit(audit, normalized, source_sha)
        self.assertEqual(validated["stage1_joint_relations"], 8)
        selected = select_stage2_relations(normalized, validated)
        self.assertEqual([row["relation_id"] for row in selected],
                         ["R1", "R2", "R3", "R4"])

    def test_relation_audit_decision_priority(self):
        row = {
            "parent_only_can_derive_combined_conclusion": True,
            "child_only_can_derive_combined_conclusion": False,
            "combined_can_derive_combined_conclusion": True,
            "child_fully_restates_parent": False, "evidence_redundant": False,
            "natural_relation": True, "unsupported_causal_claim": False,
        }
        self.assertEqual(derive_relation_audit_decision(row),
                         "parent_only_sufficient")

    def test_qa_writer_sees_only_frozen_relation_and_cannot_drift(self):
        normalized = normalize_relation_batch(self.valid_batch(), self.aliases)
        relation = freeze_relation(normalized["relations"][0], "a" * 64)
        prompt = build_qa_writer_prompt(relation)
        self.assertIn(self.aliases["P001"]["text"], prompt)
        self.assertIn(self.aliases["C001"]["text"], prompt)
        self.assertNotIn(self.aliases["P002"]["text"], prompt)
        raw = {
            "relation_id": "R1", "fixed_parent_turn_aliases": ["P001"],
            "fixed_child_turn_aliases": ["C001"], "question": "Question?",
            "answer": "Answer", "relation_preserved": True,
        }
        self.assertEqual(validate_qa_writer_object(raw, relation)["decision"],
                         "qa_accept")
        drift = copy.deepcopy(raw)
        drift["fixed_child_turn_aliases"] = ["C002"]
        self.assertEqual(validate_qa_writer_object(drift, relation)["decision"],
                         "qa_relation_drift")

    def test_stage2_audit_and_route_threshold(self):
        qa = [{"relation_id": "R1"}, {"relation_id": "R2"}]
        source_sha = "b" * 64
        rows = []
        for relation_id in ("R1", "R2"):
            rows.append({
                "relation_id": relation_id,
                "parent_only_can_answer_uniquely": False,
                "child_only_can_answer_uniquely": False,
                "combined_can_answer_uniquely": True,
                "single_turn_shortcut": False,
                "question_leaks_intermediate_relation": False,
                "answer_supported_by_frozen_relation": True,
                "relation_preserved_semantically": True,
                "notes": "",
            })
        audit = {"audit_version": QA_AUDIT_VERSION,
                 "source_qa_summary_sha256": source_sha, "items": rows}
        validated = validate_stage2_audit(audit, qa, source_sha)
        self.assertEqual(validated["qa_joint_preserved"], 2)
        self.assertEqual(route_conclusion(2, 2, 2),
                         "reference_guided_relation_first_supported_for_distractor_stage")
        bad = dict(rows[0])
        bad["child_only_can_answer_uniquely"] = True
        self.assertEqual(derive_qa_audit_decision(bad), "child_only_sufficient")


if __name__ == "__main__":
    unittest.main()
