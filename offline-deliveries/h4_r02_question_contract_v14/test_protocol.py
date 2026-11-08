"""Cross-stage checks for relaxed citations and same-turn ordering anchors."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol = load("v14_protocol_tests", HERE / "evaluation/protocol.py")
multi = load("v14_multi_tests", HERE / "multiselect_gate.py")
MODEL = "Qwen/Qwen3.6-35B-A3B"


class QuestionContractTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.source = {"root_id": "root", "turns": [
            {"turn_ref": "R01:T01", "speaker": "A", "text": "Alpha fact. Beta fact."},
            {"turn_ref": "R01:T02", "speaker": "B", "text": "Gamma fact. Delta fact."},
            {"turn_ref": "R01:T03", "speaker": "A", "text": "Epsilon fact. Zeta fact."}]}
        self.path = self.folder / "source.json"
        self.path.write_text(json.dumps(self.source))
        self.turns = {turn["turn_ref"]: turn for turn in self.source["turns"]}
        self.raw = {"candidate_id": "root-q", "question": "Select all that apply.",
                    "options": [{"proposition": "Statement " + label,
                                 "relation": "supports" if label in "AC" else "contradicts",
                                 "turn_ref": "wrong_ref", "speaker": "wrong_speaker",
                                 "quote": "citation text with ...", "rationale": ""}
                                for label in "ABCDEF"]}

    def normalize(self, row=None):
        return multi.validate_candidate(self.raw if row is None else row, "root-q", self.turns,
                                        "root", self.path, protocol.sha(self.path))

    def validate(self, question, *, reviewed=False):
        pool = {"version": "h4_r02_question_contract_v14", "questions": [question],
                "provenance": {"requested_model": MODEL, "returned_model": MODEL}}
        path = self.folder / "pool.json"
        path.write_text(json.dumps(pool))
        return protocol.validate_pool(path, require_ordering_review=reviewed)

    def ordering(self):
        quotes = ["Alpha fact.", "Beta fact.", "Gamma fact.", "Delta fact.", "Epsilon fact.", "Zeta fact."]
        labels = list(reversed(["I" + str(i) for i in range(1, 7)]))
        evidence = {label: {"turn_ref": self.source["turns"][i // 2]["turn_ref"],
                            "speaker": self.source["turns"][i // 2]["speaker"], "quote": quote,
                            "reason": "Source states this.", "status": "statement"}
                    for i, (label, quote) in enumerate(zip(labels, quotes))}
        return {"candidate_id": "root-order", "root_id": "root", "question_type": "ordering",
                "chronology_kind": "conversation_order", "question": "How do the claims APPEAR IN THE SUPPLIED DIALOGUE?",
                "items": {"I" + str(i): "Content " + str(i) for i in range(1, 7)},
                "gold": labels, "item_evidence": evidence, "order_anchor_spans": deepcopy(evidence),
                "source_path": str(self.path), "source_sha256": protocol.sha(self.path),
                "qwen_review": {"status": "PASS", "returned_model": MODEL}}

    def test_bad_citations_and_missing_topic_are_diagnostic_across_both_stages(self):
        raw = deepcopy(self.raw)
        errors, question = self.normalize()
        self.assertEqual(errors, [])
        self.assertTrue(question["citation_diagnostics"])
        self.assertEqual(question["citation_gate"], "DIAGNOSTIC_ONLY")
        self.assertEqual(question["raw_model_candidate"], raw)
        self.assertEqual(self.raw, raw)
        _, loaded, _ = self.validate(question)
        self.assertEqual(loaded[0], question)

    def test_missing_or_nontext_citation_fields_do_not_reject(self):
        for option in self.raw["options"]:
            option.pop("rationale")
            option["turn_ref"] = ["malformed"]
            option["quote"] = None
            option["speaker"] = 12
        errors, question = self.normalize()
        self.assertEqual(errors, [])
        self.validate(question)
        self.assertEqual(question["option_evidence"]["A"]["turn_ref"], ["malformed"])

    def test_gold_zero_and_one_are_still_rejected(self):
        for count in (0, 1):
            for index, option in enumerate(self.raw["options"]):
                option["relation"] = "supports" if index < count else "contradicts"
            errors, candidate = self.normalize()
            self.assertIn("gold_cardinality_invalid", errors)
            self.assertIsNone(candidate)

    def test_gold_all_six_is_permitted(self):
        for option in self.raw["options"]:
            option["relation"] = "supports"
        errors, question = self.normalize()
        self.assertEqual(errors, [])
        self.assertEqual(question["gold"], list("ABCDEF"))
        self.validate(question)

    def test_relations_and_explicit_raw_gold_cannot_be_relabelled(self):
        self.raw["gold"] = ["A", "B"]
        self.assertIn("raw_gold_relation_mismatch", self.normalize()[0])
        del self.raw["gold"]
        _, question = self.normalize()
        question["gold"] = ["B", "C"]
        with self.assertRaisesRegex(ValueError, "relation_gold_mismatch"):
            self.validate(question)

    def test_unknown_or_missing_relation_rejects(self):
        self.raw["options"][1]["relation"] = "unknown"
        self.assertIn("B:relation_invalid", self.normalize()[0])
        self.raw["options"][1].pop("relation")
        self.assertIn("B:relation_invalid", self.normalize()[0])

    def test_whitespace_duplicates_reject_in_both_stages(self):
        _, question = self.normalize()
        question["options"]["B"] = "  statement   A  "
        with self.assertRaisesRegex(ValueError, "SIX_UNIQUE_OPTIONS"):
            self.validate(question)
        self.raw["options"][1]["proposition"] = "  statement   A  "
        self.assertIn("options_not_unique", self.normalize()[0])

    def test_source_hash_and_root_binding_remain_hard(self):
        _, question = self.normalize()
        question["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SOURCE_PATH_OR_SHA_INVALID"):
            self.validate(question)
        errors, candidate = multi.validate_candidate(self.raw, "root-q", self.turns, "root", self.path, "0" * 64)
        self.assertIn("source_sha_mismatch", errors)
        self.assertIsNone(candidate)
        errors, _ = multi.validate_candidate(self.raw, "root-q", self.turns, "other", self.path, protocol.sha(self.path))
        self.assertIn("source_root_mismatch", errors)

    def test_new_branch_evidence_is_not_a_hidden_admission_rule(self):
        for option in self.raw["options"]:
            option.update(turn_ref="R01:T01", speaker="A", quote="Alpha fact.")
        errors, question = self.normalize()
        self.assertEqual(errors, [])
        self.validate(question)

    def test_gold_and_citation_metadata_do_not_enter_prompt_or_permutations(self):
        _, question = self.normalize()
        changed = deepcopy(question)
        changed["gold"] = ["B", "D", "F"]
        changed["option_evidence"] = {"A": {"quote": "secret diagnostic marker"}}
        variants = protocol.make_variants(question, seed=19, count=6)
        self.assertEqual(variants, protocol.make_variants(changed, seed=19, count=6))
        self.assertEqual(len({tuple(row["display_order_content_ids"]) for row in variants}), 6)
        self.assertEqual(protocol.prompt(question, variants[0], self.source),
                         protocol.prompt(changed, variants[0], self.source))
        self.assertNotIn("citation text with", protocol.prompt(question, variants[0], self.source))
        self.assertNotIn("secret diagnostic marker", protocol.prompt(changed, variants[0], self.source))

    def test_three_turns_six_distinct_clauses_pass(self):
        question = self.ordering()
        _, questions, _ = self.validate(question, reviewed=True)
        self.assertEqual(questions[0]["gold"], list(reversed(["I" + str(i) for i in range(1, 7)])))

    def test_same_turn_overlapping_clauses_reject(self):
        question = self.ordering()
        question["order_anchor_spans"]["I5"]["quote"] = "Alpha fact. Beta fact."
        with self.assertRaisesRegex(ValueError, "ORDERING_ANCHOR_SPANS_INVALID"):
            self.validate(question, reviewed=True)

    def test_same_turn_duplicate_anchor_rejects(self):
        question = self.ordering()
        question["order_anchor_spans"]["I5"] = deepcopy(question["order_anchor_spans"]["I6"])
        with self.assertRaisesRegex(ValueError, "ORDERING_ANCHOR_SPANS_INVALID"):
            self.validate(question, reviewed=True)

    def test_stale_offsets_and_wrong_gold_reject(self):
        question = self.ordering()
        question["order_anchor_spans"]["I6"]["char_start"] = 7
        with self.assertRaisesRegex(ValueError, "ORDERING_ANCHOR_SPANS_INVALID"):
            self.validate(question, reviewed=True)
        question = self.ordering()
        question["gold"] = list(reversed(question["gold"]))
        with self.assertRaisesRegex(ValueError, "ORDERING_ANCHOR_GOLD_MISMATCH"):
            self.validate(question, reviewed=True)

    def test_reviewed_ordering_cannot_fallback_to_generated_anchors(self):
        question = self.ordering()
        question["generated_order_anchor_spans"] = question.pop("order_anchor_spans")
        with self.assertRaisesRegex(ValueError, "ORDERING_FINAL_ANCHOR_SPANS_REQUIRED"):
            self.validate(question, reviewed=True)

    def test_final_reviewed_spans_are_authoritative(self):
        question = self.ordering()
        # The generator's citation may be corrected by the independent Qwen
        # anchor review; evaluator must check final reviewed spans and Gold.
        question["item_evidence"]["I5"]["turn_ref"] = "obsolete_generated_ref"
        self.validate(question, reviewed=True)

    def test_wrong_reviewer_identity_rejects(self):
        question = self.ordering()
        question["qwen_review"]["returned_model"] = "other_model"
        with self.assertRaisesRegex(ValueError, "ORDERING_QWEN_REVIEW_REQUIRED"):
            self.validate(question, reviewed=True)


if __name__ == "__main__":
    unittest.main()
