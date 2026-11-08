"""Regression checks for public/private separation and answer-preserving display."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.question_presentation_gate import (
    coverage_report, is_correct_display_answer, lint_candidate, prepare_display,
    restore_prediction,
)


def mc(candidate_id="mc-1", count=2):
    return {"candidate_id": candidate_id, "root_id": "root-a", "question_type": "multi_select",
            "task_operation": "MC_EVIDENCE_SELECTION", "question": "Select all supported statements.",
            "options": {"A": "Alice accepted the meeting.", "B": "Bob declined it.",
                        "C": "Carol entered it in her calendar.", "D": "Dan completed the meeting.",
                        "E": "Eve attended it.", "F": "Fran cancelled it."},
            "gold": ["A", "C"] if count == 2 else list("ABCDEF"[:count]),
            "source_binding": {"sha256": "1" * 64, "path": "/source.json", "bytes": 100}}


def ordering():
    return {"id": "order-1", "root": "root-a", "type": "ordering", "operation": "ORDER_EXPLICIT_TIME",
            "question": "Order the activities by their agreed start times.",
            "items": {"I1": "Team meeting", "I2": "Supplier phone call", "I3": "Presentation review"},
            "gold": ["I3", "I1", "I2"], "source_binding": {"sha256": "2" * 64}}


class PresentationGateTests(unittest.TestCase):
    def test_mc_answers_follow_content_after_shuffle_and_use_set_comparison(self):
        public, private = prepare_display(mc(), 72)
        selected = [label for label, text in public["options"].items()
                    if text in {"Alice accepted the meeting.", "Carol entered it in her calendar."}]
        self.assertEqual(set(restore_prediction(private, selected)), {"A", "C"})
        self.assertTrue(is_correct_display_answer(private, list(reversed(selected))))
        wrong = [label for label, text in public["options"].items()
                 if text == "Dan completed the meeting."]
        self.assertFalse(is_correct_display_answer(private, selected + wrong))

    def test_ordering_restores_content_sequence_and_rejects_set_equivalence(self):
        public, private = prepare_display(ordering(), "frozen-seed")
        by_text = {text: label for label, text in public["items"].items()}
        prediction = [by_text[t] for t in ["Presentation review", "Team meeting", "Supplier phone call"]]
        self.assertEqual(restore_prediction(private, prediction), ["I3", "I1", "I2"])
        self.assertTrue(is_correct_display_answer(private, prediction))
        self.assertFalse(is_correct_display_answer(private, list(reversed(prediction))))

    def test_six_gold_is_preserved_and_all_selected_scores_correct(self):
        public, private = prepare_display(mc(count=6), 9)
        self.assertEqual(private["display_gold"], list("ABCDEF"))
        self.assertTrue(is_correct_display_answer(private, list(reversed(public["options"]))))
        self.assertFalse(is_correct_display_answer(private, list("ABCDE")))

    def test_reproducible_nonidentity_independent_of_gold_and_no_mutation(self):
        candidate = mc()
        before = copy.deepcopy(candidate)
        public, private = prepare_display(candidate, 11)
        self.assertEqual((public, private), prepare_display(candidate, 11))
        self.assertEqual(candidate, before)
        changed = copy.deepcopy(candidate)
        changed["gold"] = list("ABCDEF")
        new_public, new_private = prepare_display(changed, 11)
        self.assertEqual(public, new_public)
        self.assertEqual(private["display_to_canonical"], new_private["display_to_canonical"])
        self.assertNotEqual(list(private["display_to_canonical"].values()), list("ABCDEF"))
        self.assertFalse(private["historical_scores_reusable"])

    def test_nonidentity_changes_supplied_order_even_when_ids_are_not_sorted(self):
        candidate = ordering()
        candidate["items"] = {"I3": "Presentation review", "I1": "Team meeting", "I2": "Supplier phone call"}
        supplied_order = list(candidate["items"].values())
        for seed in range(20):
            public, private = prepare_display(candidate, seed)
            self.assertNotEqual(list(public["items"].values()), supplied_order)
            self.assertTrue(private["nonidentity_permutation"])
            self.assertTrue(is_correct_display_answer(private, private["display_gold"]))

    def test_public_whitelist_blocks_private_top_level_and_nested_metadata(self):
        candidate = mc()
        for key in ["private", "evidence", "trigger", "author", "author_id", "private_option_notes"]:
            candidate[key] = "SECRET_" + key
        candidate["source_binding"]["gold"] = "SECRET_nested_gold"
        public, private = prepare_display(candidate, 4)
        serialized = json.dumps(public)
        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("gold", public)
        self.assertNotIn("display_to_canonical", public)
        self.assertEqual(set(public), {"schema_version", "candidate_id", "root_id", "question_type",
                                      "task_operation", "question", "options", "source_binding"})
        self.assertEqual(private["status"], "mechanical_only")

    def test_coverage_exposes_missing_four_and_six_without_mutating_gold(self):
        candidates = [mc("two", 2), mc("three", 3), mc("five", 5), ordering()]
        before = copy.deepcopy(candidates)
        report = coverage_report(candidates)
        self.assertEqual(report["missing_gold_counts"], [4, 6])
        self.assertEqual(report["deficits"], {"2": 0, "3": 0, "4": 1, "5": 0, "6": 1})
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(candidates, before)
        complete = coverage_report([mc(str(k), k) for k in range(2, 7)])
        self.assertTrue(complete["coverage_complete"])
        self.assertFalse(complete["semantically_approved"])

    def test_duplicate_questions_do_not_fill_coverage(self):
        report = coverage_report([mc("same", 2), mc("same", 2)], minimum_per_count=2)
        self.assertEqual(report["gold_count_histogram"]["2"], 1)
        self.assertEqual(report["deficits"]["2"], 1)
        self.assertEqual(len(report["invalid_candidates"]), 1)

    def test_rejects_duplicate_unknown_incomplete_and_invalid_cardinality(self):
        bad = []
        a = mc(); a["gold"] = ["A", "A"]; bad.append(a)
        a = mc(); a["gold"] = ["A", "Z"]; bad.append(a)
        a = mc(); a["gold"] = ["A"]; bad.append(a)
        a = mc(); del a["options"]["F"]; bad.append(a)
        a = ordering(); a["gold"] = ["I3", "I1"]; bad.append(a)
        a = ordering(); a["gold"] = ["I3", "I1", "I1"]; bad.append(a)
        for candidate in bad:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    prepare_display(candidate, 1)

    def test_prediction_rejects_duplicate_and_unknown_labels(self):
        _, private = prepare_display(mc(), 3)
        for prediction in [["A", "A"], ["Z"], "A C"]:
            with self.assertRaises(ValueError):
                restore_prediction(private, prediction)

    def test_lint_flags_cues_without_deleting_or_semantically_approving(self):
        candidate = ordering()
        candidate["items"] = {"I1": "Initially, meeting at 10:00", "I2": "Later, call at 11:30",
                              "I3": "最初复核 at 09:15; 最后; lastly"}
        before = copy.deepcopy(candidate)
        report = lint_candidate(candidate)
        codes = [flag["code"] for flag in report["flags"]]
        self.assertEqual(codes.count("ORDERING_DISCOURSE_CUE"), 3)
        self.assertEqual(report["flags"][2]["matches"], ["最初", "最后", "lastly"])
        self.assertIn("ALL_ORDERING_ITEMS_EXPOSE_CLOCK_TIME", codes)
        self.assertFalse(report["semantically_approved"])
        self.assertEqual(candidate, before)
        candidate = mc(); candidate["question"] = "Select exactly five supported statements."
        self.assertEqual(lint_candidate(candidate)["flags"][0]["code"], "QUESTION_REVEALS_SELECTION_COUNT")

    def test_cli_missing_coverage_keeps_audit_but_never_declares_solver_ready(self):
        script = Path(__file__).with_name("question_presentation_gate.py")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inp = root / "input.json"
            inp.write_text(json.dumps([mc()]))
            out = root / "prepared"
            result = subprocess.run([sys.executable, str(script), "--input", str(inp),
                                     "--output-dir", str(out), "--seed", "test",
                                     "--require-coverage"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            audit = json.loads((out / "PREPARATION_AUDIT.json").read_text())
            self.assertFalse(audit["solver_ready_declared"])
            self.assertFalse(audit["coverage"]["coverage_complete"])
            self.assertTrue((out / "PUBLIC_DISPLAYS.json").exists())
            self.assertTrue((out / "PRIVATE_SCORING.json").exists())
            self.assertEqual((out / "PRIVATE_SCORING.json").stat().st_mode & 0o777, 0o600)
            self.assertFalse(list(out.glob("*SOLVER_READY*")))


if __name__ == "__main__":
    unittest.main()
