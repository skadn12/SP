"""CPU contract checks for source-free answers, pairing and complete denominators."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("quality_pilot_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def candidate(cid, kind="multi_select", gold=None):
    field = "options" if kind == "multi_select" else "items"
    keys = list("ABCDEF") if kind == "multi_select" else ["I1", "I2", "I3"]
    return {"candidate_id": cid, "root_id": "train-root", "question_type": kind,
        "task_operation": "MC_CONSTRAINT_CONJUNCTION" if kind == "multi_select" else "ORDER_EXPLICIT_TIME",
        "question": "Select supported statements." if kind == "multi_select" else "Order the events.",
        field: {key: "Event " + key for key in keys}, "gold": gold if gold is not None else keys[:2],
        "source_binding": {"path": "/unused/train-source.json", "sha256": "0" * 64, "bytes": 1},
        "author_id": "author"}


class Contracts(unittest.TestCase):
    def test_source_free_supported_and_full_citation_requirement(self):
        row = {"condition": "QUESTION_ONLY", "labels": ["A", "B"], "question_type": "ordering"}
        output = {"item_evidence": {label: {"judgment": "supported", "turn_refs": [], "reason": "A wording-based prediction."}
            for label in row["labels"]}, "answer_reason": "Best prediction without source.", "final_answer": "<box>A B</box>"}
        schema = runner.question_only_schema(row["labels"], row["question_type"])
        self.assertEqual(runner.validate_output(json.dumps(output), row, schema)[1], ["A", "B"])
        output["item_evidence"]["A"]["turn_refs"] = ["hidden-turn"]
        with self.assertRaises(Exception):
            runner.validate_output(json.dumps(output), row, schema)
        output["item_evidence"]["A"]["turn_refs"] = []
        row["condition"] = "FULL"
        full_schema = runner.diagnostic.repair.schema(row["labels"], ["visible-turn"], "ordering")
        with self.assertRaisesRegex(AssertionError, "SUPPORTED_WITHOUT_CITATION"):
            runner.validate_output(json.dumps(output), row, full_schema)
        for value in output["item_evidence"].values():
            value["turn_refs"] = ["visible-turn"]
        self.assertEqual(runner.validate_output(json.dumps(output), row, full_schema)[1], ["A", "B"])

    def test_paired_mapping_preserves_parent_content_and_own_gold(self):
        parent = candidate("parent", gold=["A", "B", "D"])
        child = candidate("child", gold=["B", "C"])
        child.update(parent_candidate_id="parent", paired_item_identity_preserved=True)
        child["options"] = {key: "Revised " + text for key, text in child["options"].items()}
        untouched = copy.deepcopy(parent)
        child_public, child_private = runner.gate.prepare_display(child, runner.DISPLAY_SEED)
        parent_public, parent_private = runner.parent_display(parent, child_public, child_private)
        self.assertEqual(parent_private["display_to_canonical"], child_private["display_to_canonical"])
        self.assertEqual(parent_private["canonical_gold"], ["A", "B", "D"])
        self.assertNotEqual(parent_private["display_gold"], child_private["display_gold"])
        for label, original in parent_private["display_to_canonical"].items():
            self.assertEqual(parent_public["options"][label], parent["options"][original])
        self.assertEqual(runner.sampling_seed(parent), runner.sampling_seed(child))
        self.assertEqual(parent, untouched)

    def test_invalid_answers_keep_exact_and_order_pair_denominators(self):
        scores = []
        for index, (question, valid) in enumerate([
                (candidate("mc4", gold=list("ABCD")), False),
                (candidate("mc6", gold=list("ABCDEF")), True),
                (candidate("ord", "ordering", ["I1", "I2", "I3"]), False)]):
            public, private = runner.gate.prepare_display(question, runner.DISPLAY_SEED)
            row = {**public, "request_id": str(index), "condition": "FULL", "cohort": "new"}
            scores.append(runner.score_record(row, private, private["display_gold"] if valid else [], valid))
        result = runner.aggregate(scores)
        self.assertEqual((result["frozen_requests"], result["valid_requests"], result["correct"]), (3, 1, 1))
        self.assertEqual(result["exact_accuracy"], 1 / 3)
        self.assertEqual((result["ordering_pairwise_correct"], result["ordering_pairwise_total"]), (0, 3))
        self.assertEqual(result["mc_false_negative"], 4)
        self.assertEqual((result["mc_allselect_correct"], result["mc_allselect_accuracy"]), (1, .5))

    def test_review_requires_exact_draft_and_independent_reviewer(self):
        old = {"parent": candidate("parent")}
        new = candidate("new")
        new["parent_candidate_id"] = "parent"
        review = {"candidate_id": "new", "candidate_sha256": runner.fingerprint(new), "decision": "PASS",
            "author_id": "author", "reviewer_id": "reviewer", "parent_baseline_gold_valid": False}
        admitted, _, valid = runner.reviewed_candidates([new], {"reviews": [review]}, old)
        self.assertEqual(admitted, [new])
        self.assertFalse(valid["parent"])
        review["reviewer_id"] = "author"
        with self.assertRaisesRegex(AssertionError, "INDEPENDENT_REVIEW_REQUIRED"):
            runner.reviewed_candidates([new], {"reviews": [review]}, old)
        review["reviewer_id"] = "reviewer"
        new["question"] += " Changed after review."
        with self.assertRaisesRegex(AssertionError, "REVIEWED_CANDIDATE_DRIFT"):
            runner.reviewed_candidates([new], {"reviews": [review]}, old)


if __name__ == "__main__":
    unittest.main(verbosity=2)
