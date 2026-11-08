"""Tests for the frozen ID-only Gold audit sample."""

import copy
import unittest

from make_sample import ROOTS, build_manifest, digest, encoded


def fixture(selected_ids):
    questions = {}
    for root in sorted(ROOTS):
        for number in range(4):
            qid = root.removeprefix("cc-train-episode-") + f"-q{number}"
            question = {"candidate_id": qid, "root_id": root,
                        "question": "opaque text" + str(number)}
            questions[qid] = {"canonical_question": question, "root_id": root,
                              "selected_by_policy": qid in selected_ids,
                              "full_partial_mean": {"fraction": "1/3"}}
    chosen = [copy.deepcopy(row["canonical_question"]) for qid, row in questions.items()
              if qid in selected_ids]
    results = {
        "report_version": "h4_crossroot_multiselect_solver_selection_v2",
        "evaluation_data_status": "VALID",
        "generator_provenance": {"semantic_review_performed": False,
                                 "codex_question_edits": False,
                                 "mechanical_gate_version": "raw_source_and_gold_v1"},
        "counts": {"canonical_questions": 16, "selected_questions": len(chosen),
                   "unresolved_questions": 0},
        "questions": questions,
    }
    return results, {"questions": chosen, "count": len(chosen)}


class SampleTests(unittest.TestCase):
    def test_full_set_covers_each_root_and_is_score_independent(self):
        all_ids = {root.removeprefix("cc-train-episode-") + f"-q{number}"
                   for root in ROOTS for number in range(4)}
        results, selected = fixture(all_ids)
        first = build_manifest(results, selected, "a" * 64, "b" * 64)
        self.assertEqual(first["sample_count"], 6)
        self.assertEqual({row["root_id"] for row in first["sampled_ids_and_hashes"]}, ROOTS)
        reversed_results = copy.deepcopy(results)
        reversed_results["questions"] = dict(reversed(list(reversed_results["questions"].items())))
        for row in reversed_results["questions"].values():
            row["full_partial_mean"] = {"fraction": "2/3"}
        second = build_manifest(reversed_results, selected, "a" * 64, "b" * 64)
        self.assertEqual(first["sampled_ids_and_hashes"], second["sampled_ids_and_hashes"])
        self.assertEqual(digest(encoded(first)), digest(encoded(build_manifest(
            results, selected, "a" * 64, "b" * 64))))

    def test_four_or_fewer_selected_are_all_sampled(self):
        selected_ids = {root.removeprefix("cc-train-episode-") + "-q0"
                        for root in sorted(ROOTS)[:3]}
        results, selected = fixture(selected_ids)
        manifest = build_manifest(results, selected, "a" * 64, "b" * 64)
        self.assertEqual(manifest["sample_count"], 3)
        self.assertEqual({row["candidate_id"] for row in manifest["sampled_ids_and_hashes"]},
                         selected_ids)

    def test_rejects_mismatched_selected_file(self):
        results, selected = fixture({"231899-q0"})
        selected["questions"][0]["question"] = "changed text"
        with self.assertRaisesRegex(ValueError, "SELECTED_LIST_OR_COUNTS_MISMATCH"):
            build_manifest(results, selected, "a" * 64, "b" * 64)

    def test_rejects_semantically_reviewed_pool(self):
        results, selected = fixture(set())
        results["generator_provenance"]["semantic_review_performed"] = True
        with self.assertRaisesRegex(ValueError, "COMPLETE_RAW_SOLVER_REPORT_REQUIRED"):
            build_manifest(results, selected, "a" * 64, "b" * 64)


if __name__ == "__main__":
    unittest.main()
