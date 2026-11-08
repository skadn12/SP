"""Meaningful regressions for source drift, evidence, graph and review admission."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.current_method.production_question_gate import (
    audit_candidate, bind_candidates, build_review_packet, file_binding,
    fingerprint, validate_review,
)


class ProductionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.source = self.directory / "source.json"
        self.turns = [
            {"turn_ref": "T001", "speaker": "Alice", "text": "The train departs at 15:00. I need a booking reference."},
            {"turn_ref": "T002", "speaker": "Bob", "text": "The hotel check-in is at 13:00. No room is reserved."},
            {"turn_ref": "T003", "speaker": "Alice", "text": "The gallery visit starts at 10:00. The tickets are confirmed."},
        ]
        self.source.write_text(json.dumps({"root_id": "root-a", "turns": self.turns}))
        self.selection = self.directory / "selection.json"
        self.selection.write_text(json.dumps({"roots": [dict(root_id="root-a", **file_binding(self.source))]}))

    def tearDown(self):
        self.temp.cleanup()

    def ev(self, n):
        turn = self.turns[n - 1]
        return {key: turn[key] for key in ("turn_ref", "speaker")} | {"quote": turn["text"]}

    def order(self):
        return {"candidate_id": "order-a", "root_id": "root-a", "author_id": "author",
            "question_type": "ordering", "task_operation": "ORDER_EXPLICIT_TIME",
            "question": "Order the events by their clock times.",
            "items": {"A": "Train departure", "B": "Hotel check-in", "C": "Gallery visit"},
            "gold": ["C", "B", "A"],
            "item_analysis": {key: {"evidence": [self.ev(n)]} for key, n in zip("ABC", [1, 2, 3])},
            "ordering_analysis": {"operation_class": "clock_value", "edges": [
                {"before": "C", "after": "B", "evidence": [self.ev(3), self.ev(2)], "rationale": "10:00 < 13:00"},
                {"before": "B", "after": "A", "evidence": [self.ev(2), self.ev(1)], "rationale": "13:00 < 15:00"}]}}

    def bound(self, candidate=None):
        return bind_candidates([candidate or self.order()], self.selection)[0][0]

    def reviewed(self, candidate):
        review = build_review_packet(candidate, self.selection)["review_template"]
        review["reviewer_id"] = "independent-reviewer"
        review["decision"] = "PASS"
        review["checks"] = {key: {"decision": "PASS", "rationale": "Specific source-backed reviewer explanation for " + key}
                            for key in review["checks"]}
        return review

    def test_binding_derived_without_mutating_draft_or_trusting_relative_path(self):
        candidate = self.order()
        candidate["source_binding"] = {"path": "../wrong-relative.json", "sha256": file_binding(self.source)["sha256"]}
        before = copy.deepcopy(candidate)
        bound, report = bind_candidates([candidate], self.selection)
        self.assertEqual(candidate, before)
        self.assertEqual(bound[0]["source_binding"], file_binding(self.source))
        self.assertEqual(len(report["binding_changes"]), 1)
        self.assertTrue(audit_candidate(bound[0], self.selection)["mechanical_pass"])
        self.assertFalse(report["semantically_approved"])

    def test_selection_detects_source_drift_and_wrong_root(self):
        self.source.write_text(self.source.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash drift"):
            self.bound()
        self.selection.write_text(json.dumps({"roots": [dict(root_id="other-root", **file_binding(self.source))]}))
        with self.assertRaisesRegex(ValueError, "root mismatch"):
            self.bound()

    def test_binding_rejects_hash_substitution_unselected_root_and_duplicates(self):
        candidate = self.order(); candidate["source_binding"] = {"sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.bound(candidate)
        candidate = self.order(); candidate["root_id"] = "not-selected"
        with self.assertRaisesRegex(ValueError, "not in frozen selection"):
            self.bound(candidate)
        with self.assertRaisesRegex(ValueError, "unique"):
            bind_candidates([self.order(), self.order()], self.selection)

    def test_quote_and_speaker_errors_never_become_semantic_pass(self):
        candidate = self.bound()
        candidate["item_analysis"]["A"]["evidence"][0]["quote"] = "The train departs at 12:00."
        candidate["item_analysis"]["B"]["evidence"][0]["speaker"] = "Invented"
        audit = audit_candidate(candidate, self.selection)
        self.assertFalse(audit["mechanical_pass"])
        errors = [entry["error"] for entry in audit["mechanical_errors"]]
        self.assertIn("speaker mismatch", errors)
        self.assertTrue(any("exact" in error for error in errors))
        review = self.reviewed(candidate)
        self.assertFalse(validate_review(candidate, review, self.selection)["admitted"])

    def test_graph_must_be_unique_acyclic_and_equal_gold(self):
        candidate = self.bound()
        del candidate["ordering_analysis"]["edges"][1]
        self.assertFalse(audit_candidate(candidate, self.selection)["mechanical_pass"])
        candidate = self.bound()
        candidate["ordering_analysis"]["edges"].append({"before": "A", "after": "C", "evidence": [self.ev(1)]})
        audit = audit_candidate(candidate, self.selection)
        self.assertFalse(audit["ordering_checks"]["graph"]["acyclic"])
        candidate = self.bound(); candidate["gold"] = ["A", "B", "C"]
        self.assertFalse(audit_candidate(candidate, self.selection)["mechanical_pass"])

    def test_discourse_mention_overlap_is_a_review_issue_not_rejection(self):
        candidate = self.order()
        candidate["task_operation"] = "ORDER_DISCOURSE_RECONSTRUCTION"
        candidate["ordering_analysis"]["operation_class"] = "discourse_reconstruction"
        candidate["gold"] = list("ABC")
        candidate["ordering_analysis"]["edges"] = [
            {"before": "A", "after": "B", "evidence": [self.ev(1), self.ev(2)]},
            {"before": "B", "after": "C", "evidence": [self.ev(2), self.ev(3)]}]
        candidate = self.bound(candidate)
        audit = audit_candidate(candidate, self.selection)
        self.assertTrue(audit["mechanical_pass"])
        self.assertTrue(audit["ordering_checks"]["gold_equals_strict_earliest_cited_turn_order"])
        self.assertFalse(audit["ordering_checks"]["first_relevant_mentions_verified"])
        self.assertIn("intended", audit["ordering_checks"]["mention_overlap_policy"])
        self.assertTrue(validate_review(candidate, self.reviewed(candidate), self.selection)["admitted"])

    def test_shared_turn_citations_do_not_invent_a_strict_first_mention_order(self):
        candidate = self.bound()
        candidate["item_analysis"] = {key: {"evidence": [self.ev(1)]} for key in "ABC"}
        audit = audit_candidate(candidate, self.selection)
        self.assertTrue(audit["ordering_checks"]["cited_position_ties"])
        self.assertFalse(audit["ordering_checks"]["gold_equals_strict_earliest_cited_turn_order"])

    def test_review_requires_specific_complete_checks_and_independent_identity(self):
        candidate = self.bound()
        review = self.reviewed(candidate)
        self.assertTrue(validate_review(candidate, review, self.selection)["admitted"])
        review["reviewer_id"] = "author"
        self.assertFalse(validate_review(candidate, review, self.selection)["admitted"])
        review = self.reviewed(candidate); del review["checks"]["source_dependence"]
        self.assertFalse(validate_review(candidate, review, self.selection)["admitted"])
        review = self.reviewed(candidate); review["checks"]["common_sense"]["decision"] = "HOLD"
        self.assertFalse(validate_review(candidate, review, self.selection)["admitted"])
        review = self.reviewed(candidate); review["checks"]["mention_order"]["rationale"] = " "
        self.assertFalse(validate_review(candidate, review, self.selection)["admitted"])

    def test_review_binding_drift_and_postreview_rewrite_block_admission(self):
        candidate = self.bound(); review = self.reviewed(candidate)
        candidate["question"] += " Changed."
        self.assertNotEqual(fingerprint(candidate), review["candidate_sha256"])
        self.assertFalse(validate_review(candidate, review, self.selection)["admitted"])
        candidate = self.bound(); review = self.reviewed(candidate)
        review["source_binding"]["path"] = "relative.json"
        self.assertFalse(validate_review(candidate, review, self.selection)["admitted"])

    def test_multi_select_truth_boolean_and_six_gold_control_are_explicit(self):
        candidate = self.order()
        candidate.pop("items"); candidate.pop("ordering_analysis")
        candidate["question_type"] = "multi_select"; candidate["task_operation"] = "MC_EVIDENCE_SELECTION"
        candidate["options"] = {key: "Statement " + key for key in "ABCDEF"}
        candidate["gold"] = list("ABCDEF")
        candidate["item_analysis"] = {key: {"evidence": [self.ev(1)], "should_select": True} for key in "ABCDEF"}
        candidate = self.bound(candidate)
        self.assertIn("six_gold_control", build_review_packet(candidate, self.selection)["required_checks"])
        candidate["item_analysis"]["A"]["should_select"] = 1
        self.assertFalse(audit_candidate(candidate, self.selection)["mechanical_pass"])

    def test_editorial_cues_flagged_but_not_automatically_deleted(self):
        candidate = self.order(); candidate["items"]["A"] = "Finally, the train departure"
        candidate = self.bound(candidate)
        before = copy.deepcopy(candidate)
        packet = build_review_packet(candidate, self.selection)
        self.assertEqual(candidate, before)
        self.assertTrue(packet["mechanical_audit"]["presentation_lint"]["flags"])
        self.assertTrue(packet["semantic_review_required"])
        self.assertFalse(packet["semantically_approved"])

    def test_cli_keeps_private_review_packets_and_does_not_overwrite(self):
        inp = self.directory / "input.json"; inp.write_text(json.dumps([self.order()]))
        out = self.directory / "output"
        command = [sys.executable, str(Path(__file__).with_name("production_question_gate.py")),
            "--input", str(inp), "--selection", str(self.selection), "--output-dir", str(out)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        audit = json.loads((out / "PRODUCTION_GATE_AUDIT.json").read_text())
        self.assertEqual(audit["admitted_count"], 0)
        self.assertTrue(audit["semantic_review_required"])
        self.assertEqual((out / "CANDIDATES_BOUND_PRIVATE.json").stat().st_mode & 0o777, 0o600)
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
