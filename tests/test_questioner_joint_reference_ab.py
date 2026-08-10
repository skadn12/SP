import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.questioner_joint_reference_ab import (
    BLIND_SEED,
    FORMAT_ONLY_RETRY_SUFFIX,
    PaidRequestBudget,
    JointReferenceCanaryError,
    alias_lookup,
    build_blind_review,
    build_group_prompts,
    build_turn_aliases,
    format_retry_suffix_sha256,
    load_full_dialogue,
    normalize_group_result,
    request_bodies_differ_only_by_reference,
    schema_sha256,
    validate_response,
)
from scripts.current_method.restart_n5_spell.run_questioner_joint_reference_ab_canary import (
    DEFAULT_DIALOGUE,
    DEFAULT_REFERENCE,
    _assert_inputs,
    _request_body,
)
from scripts.current_method.restart_n5_spell.live_backends import ChallengerAPIConfig


class QuestionerJointReferenceABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.turns = load_full_dialogue(DEFAULT_DIALOGUE)
        cls.alias_map, cls.aliased = build_turn_aliases(cls.turns)
        cls.aliases = alias_lookup(cls.alias_map, cls.aliased)
        cls.reference = DEFAULT_REFERENCE.read_text(encoding="utf-8")

    def valid_payload(self):
        candidates = []
        for index in range(1, 4):
            candidates.append({
                "candidate_id": f"Q{index}",
                "status": "valid",
                "reasoning_type": "test_relation",
                "parent_turn_aliases": [f"P{index:03d}"],
                "child_turn_aliases": [f"C{index:03d}"],
                "question": f"Question {index}?",
                "answer": f"Answer {index}",
                "parent_contribution": "Parent contribution",
                "child_contribution": "Child contribution",
                "why_parent_only_is_insufficient": "Parent is insufficient",
                "why_child_only_is_insufficient": "Child is insufficient",
                "why_combined_is_sufficient": "Together they are sufficient",
                "single_turn_shortcut_check": "No single turn shortcut",
                "repeated_evidence_check": "Evidence is not repeated",
            })
        candidates.append({
            "candidate_id": "Q4",
            "status": "no_valid_candidate",
            "reason": "No fourth relationship survived both ablations.",
        })
        return {"candidates": candidates}

    def test_authoritative_dialogue_counts_and_alias_mapping(self):
        self.assertEqual(len(self.turns), 371)
        self.assertEqual(self.alias_map["parent_turn_count"], 237)
        self.assertEqual(self.alias_map["child_turn_count"], 134)
        self.assertEqual(self.alias_map["entries"][0]["alias"], "P001")
        self.assertEqual(self.alias_map["entries"][236]["alias"], "P237")
        self.assertEqual(self.alias_map["entries"][237]["alias"], "C001")
        self.assertEqual(self.alias_map["entries"][-1]["alias"], "C134")
        for original, aliased in zip(self.turns, self.aliased["combined_dialogue"]):
            for field in ("turn_id", "session_id", "speaker", "text"):
                self.assertEqual(original[field], aliased[field])

    def test_reference_exists_and_hashes_are_stable(self):
        self.assertTrue(DEFAULT_REFERENCE.is_file())
        self.assertEqual(len(schema_sha256()), 64)
        self.assertEqual(len(format_retry_suffix_sha256()), 64)
        self.assertNotIn("previous response content", FORMAT_ONLY_RETRY_SUFFIX)

    def test_ab_prompt_and_request_only_differ_by_reference(self):
        group_a, group_b, proof = build_group_prompts(self.aliased, self.reference)
        self.assertTrue(proof["group_b_without_reference_equals_group_a"])
        api = ChallengerAPIConfig(
            base_url="https://example.invalid/v1", api_key="x",
            model="Qwen/Qwen3-14B", temperature=0.25,
        )
        body_a = _request_body(api, group_a)
        body_b = _request_body(api, group_b)
        self.assertTrue(request_bodies_differ_only_by_reference(
            body_a, body_b, self.reference
        ))

    def test_schema_parsing_and_no_valid_candidate(self):
        result = validate_response(json.dumps(self.valid_payload()), self.aliases)
        self.assertEqual(result["parse_status"], "valid")
        self.assertFalse(result["retry_eligible"])
        normalized = normalize_group_result("A", result, self.aliases, "primary")
        self.assertEqual(normalized["candidates"][3]["status"], "no_valid_candidate")
        self.assertEqual(normalized["candidates"][0]["resolved_parent_turns"][0]["alias"], "P001")

    def test_structural_errors_retry_but_alias_errors_do_not(self):
        malformed = self.valid_payload()
        malformed["candidates"].pop()
        result = validate_response(json.dumps(malformed), self.aliases)
        self.assertEqual(result["parse_status"], "schema_invalid")
        self.assertTrue(result["retry_eligible"])
        bad_alias = self.valid_payload()
        bad_alias["candidates"][0]["parent_turn_aliases"] = ["C001"]
        result = validate_response(json.dumps(bad_alias), self.aliases)
        self.assertEqual(result["parse_status"], "alias_invalid")
        self.assertFalse(result["retry_eligible"])

    def test_paid_api_budget_limit(self):
        budget = PaidRequestBudget(4)
        self.assertEqual([budget.reserve() for _ in range(4)], [1, 2, 3, 4])
        with self.assertRaises(JointReferenceCanaryError):
            budget.reserve()

    def test_output_directory_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "run"
            _assert_inputs(DEFAULT_DIALOGUE, DEFAULT_REFERENCE, run_root)
            run_root.mkdir()
            with self.assertRaises(FileExistsError):
                _assert_inputs(DEFAULT_DIALOGUE, DEFAULT_REFERENCE, run_root)

    def test_blind_shuffle_is_reproducible_and_hides_groups(self):
        validated = validate_response(json.dumps(self.valid_payload()), self.aliases)
        group_a = normalize_group_result("A", validated, self.aliases, "primary")
        group_b = normalize_group_result("B", validated, self.aliases, "primary")
        packet1, key1 = build_blind_review([group_a, group_b], BLIND_SEED)
        packet2, key2 = build_blind_review([group_a, group_b], BLIND_SEED)
        self.assertEqual(packet1, packet2)
        self.assertEqual(key1, key2)
        rendered = json.dumps(packet1)
        self.assertNotIn('"group"', rendered)
        self.assertNotIn("reference_used", rendered)
        self.assertTrue(all(
            item.get("human_audit", {}).get("final_joint_judgment") is None
            for item in packet1["items"] if item["slot_status"] == "valid"
        ))


if __name__ == "__main__":
    unittest.main()
