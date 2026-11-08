"""Regression checks for full-source/question-scoped evidence handling.

These tests are offline and do not run Reviewer, Solver, network APIs, or
training.  They exercise the existing admitted artifacts and a copied candidate.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import source_workflow as workflow


BASE = PROJECT / "research-protocols/selfplay_week_20261019/base_qa_v2_2_1_acceptance_20261102"


def load_candidate():
    root = BASE / "outputs/additional_104814/candidate_02_actor_outcome_multiselect"
    return (json.loads((root / "CANDIDATE.json").read_text()),
            json.loads((root / "TARGET.json").read_text()))


class FullSourceQuestionScopeTests(unittest.TestCase):
    def test_complete_sources_are_frozen_and_not_truncated(self):
        expected = {"cc-train-episode-103866": 197, "cc-train-episode-104814": 178}
        for root, count in expected.items():
            source, scope = workflow.load_full_source(HERE / "inputs" / root)
            self.assertEqual(len(source["turns"]), count)
            self.assertEqual(len(scope["turns"]), count)
            self.assertFalse(scope["formal_qa_eligible"])

    def test_unrelated_context_only_turn_does_not_veto_allowed_question_evidence(self):
        source, scope = workflow.load_full_source(HERE / "inputs/cc-train-episode-104814")
        candidate, target = load_candidate()
        result = workflow.question_evidence_audit(source, scope, candidate, target)
        self.assertTrue(result["evidence_eligible"], result)
        self.assertEqual(result["full_source_turn_count"], 178)
        self.assertEqual(result["full_source_scan_coverage"], 178)
        self.assertFalse(result["all_turns_formally_allowed_informational_only"])
        self.assertFalse(result["difficulty_is_validity_veto"])

    def test_cited_restricted_turn_blocks_with_specific_reference(self):
        source, scope = workflow.load_full_source(HERE / "inputs/cc-train-episode-104814")
        candidate, target = load_candidate()
        candidate = copy.deepcopy(candidate)
        candidate["item_analysis"]["james_italian_address_only"]["evidence"].append({
            "turn_ref": "cc-train-episode-104814:P01:T001",
            "speaker": "Emily",
            "quote": "Hey there, James. How's it going?",
        })
        result = workflow.question_evidence_audit(source, scope, candidate, target)
        self.assertFalse(result["evidence_eligible"])
        self.assertIn("cc-train-episode-104814:P01:T001", result["ineligible_turn_refs"])

    def test_full_source_scan_is_required_for_alternative_path_review(self):
        source, scope = workflow.load_full_source(HERE / "inputs/cc-train-episode-104814")
        candidate, target = load_candidate()
        candidate = copy.deepcopy(candidate)
        candidate["query_path_audit"]["source_scan"] = []
        result = workflow.question_evidence_audit(source, scope, candidate, target)
        self.assertFalse(result["evidence_eligible"])
        self.assertIn("FULL_SOURCE_COUNTERPATH_SCAN_INCOMPLETE", result["errors"])


if __name__ == "__main__":
    unittest.main()
