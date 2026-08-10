from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.mcq_option_diagnostics import (
    diagnose_option_shape,
)


class MCQOptionDiagnosticsTest(unittest.TestCase):
    def test_detects_gold_plus_three_single_mutations(self) -> None:
        report = diagnose_option_shape(options={
            "A":"I saw dad talking; therapy today helped me discuss dad.",
            "B":"I saw dad kissing; therapy yesterday helped me discuss dad.",
            "C":"I saw dad kissing; therapy today helped me discuss dad.",
            "D":"I saw dad kissing; therapy today helped me discuss mom.",
        }, reference_answer_letter="C")
        self.assertTrue(report["single_mutation_star"])
        self.assertTrue(report["correct_unique_medoid"])
        self.assertTrue(report["answer_shape_leakage_risk"])

    def test_factorial_options_are_not_single_mutation_star(self) -> None:
        report = diagnose_option_shape(options={
            "A":"They witnessed a betrayal and later found therapy helpful.",
            "B":"They witnessed a betrayal and later confronted their father.",
            "C":"They heard an argument and later found therapy helpful.",
            "D":"They heard an argument and later confronted their father.",
        }, reference_answer_letter="A")
        self.assertFalse(report["single_mutation_star"])
        self.assertFalse(report["answer_shape_leakage_risk"])

    def test_tiny_medoid_advantage_is_not_shape_leakage(self) -> None:
        initial_gold = (
            "Initially: I was thinking a music festival. I have a lot of musician "
            "friends who would love to perform, and I think it could be really fun "
            "and successful.")
        initial_false = (
            "Initially: I was planning a community theater production. I know "
            "several actor friends who would enjoy rehearsing, and I thought it "
            "could be engaging and popular.")
        later_gold = (
            "Later: That's fantastic, Classmates B. I'm confident we've covered "
            "all the bases. Musician C, let's make sure we're all aligned on the "
            "schedule and venue details.")
        later_false = (
            "Later: That's disappointing, Classmates B. I'm worried we've uncovered "
            "several major problems. Organizer D, let's make sure we revise the "
            "budget and performer list.")
        report = diagnose_option_shape(options={
            "A": initial_false + " " + later_false,
            "B": initial_gold + " " + later_gold,
            "C": initial_false + " " + later_gold,
            "D": initial_gold + " " + later_false,
        }, reference_answer_letter="B")
        self.assertTrue(report["correct_unique_medoid"])
        self.assertFalse(report["correct_unique_medoid_leakage"])
        self.assertFalse(report["answer_shape_leakage_risk"])

    def test_question_content_that_only_matches_gold_is_leakage(self) -> None:
        report = diagnose_option_shape(
            question=("What did A initially plan for the music festival, and "
                      "what changed after the sustainability requirement?"),
            options={
                "A": "Initially an art show; later a volunteer program.",
                "B": "Initially a music festival; later a sustainability initiative.",
                "C": "Initially a food fair; later a technology upgrade.",
                "D": "Initially a book fair; later a safety initiative.",
            },
            reference_answer_letter="B",
        )
        self.assertTrue(report["question_cue_leakage_risk"])
        self.assertTrue(report["answer_shape_leakage_risk"])

    def test_shared_question_domain_across_options_is_not_cue_leakage(self) -> None:
        report = diagnose_option_shape(
            question=("How did A's music festival plan change after the sponsor's "
                      "sustainability requirement?"),
            options={
                "A": "The music festival was cancelled because the sustainability requirement was rejected.",
                "B": "The music festival added a recycling plan for the sustainability requirement.",
                "C": "The music festival changed venues to meet the sustainability requirement.",
                "D": "The music festival delayed booking while reviewing the sustainability requirement.",
            },
            reference_answer_letter="B",
        )
        self.assertFalse(report["question_cue_leakage_risk"])

    def test_two_gold_neighbour_mutations_are_shape_leakage(self) -> None:
        report = diagnose_option_shape(options={
            "A": ("Classmates A's past procrastination in writing led to negative "
                  "consequences, but their eventual success as a published author "
                  "influenced them to adopt a more responsible approach to current "
                  "writing tasks."),
            "B": ("Classmates A's past procrastination in writing led to negative "
                  "consequences, but their eventual success as a published author "
                  "influenced them to adopt a more responsible approach to managing "
                  "their time for hobbies."),
            "C": ("Classmates A's past procrastination in writing led to negative "
                  "consequences, but their eventual success as a published author "
                  "influenced them to adopt a more responsible approach to current "
                  "life tasks."),
            "D": ("Classmates A's past procrastination in writing led to negative "
                  "consequences, but their eventual success as a published author "
                  "influenced them to adopt a more responsible approach to handling "
                  "their car's maintenance."),
        }, reference_answer_letter="C")
        self.assertEqual(2, report["small_local_mutation_distractor_count"])
        self.assertTrue(report["single_mutation_star"])
        self.assertTrue(report["answer_shape_leakage_risk"])

    def test_shared_false_component_across_all_distractors_is_leakage(self) -> None:
        report = diagnose_option_shape(options={
            "A": ("A scheduled the mechanic and became more careless, while "
                  "claiming improved responsibility."),
            "B": ("A scheduled the mechanic and became more careless, showing "
                  "reduced responsibility."),
            "C": ("A scheduled the mechanic and became more careless, showing "
                  "no meaningful change."),
            "D": ("A scheduled the mechanic and became more careful, showing "
                  "improved responsibility."),
        }, reference_answer_letter="D")
        self.assertIn("more careless",
                      report["shared_distractor_only_bigrams"])
        self.assertTrue(report["shared_false_trajectory_risk"])
        self.assertTrue(report["answer_shape_leakage_risk"])


if __name__ == "__main__":
    unittest.main()
