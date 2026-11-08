from fractions import Fraction
import unittest

from selection import ground_decision, select_question


VARIANTS = tuple(f"perm_{index:02d}" for index in range(6))


def uniform_scores(value):
    return {variant: [value] * 3 for variant in VARIANTS}


def passing_ground():
    return ground_decision([False, False, False])


class GroundDecisionTests(unittest.TestCase):
    def test_initial_zero_or_one_correct_passes(self):
        for initial in ([False, False, False], [True, False, False],
                        [False, True, False], [False, False, True]):
            with self.subTest(initial=initial):
                result = ground_decision(initial)
                self.assertIs(result["ground_pass"], True)
                self.assertEqual(result["status"], "PASS")

    def test_initial_three_correct_filters(self):
        result = ground_decision([True, True, True])
        self.assertIs(result["ground_pass"], False)
        self.assertEqual(result["status"], "FILTERED")

    def test_initial_two_correct_needs_extension(self):
        for initial in ([False, True, True], [True, False, True],
                        [True, True, False]):
            with self.subTest(initial=initial):
                result = ground_decision(initial)
                self.assertIsNone(result["ground_pass"])
                self.assertEqual(result["status"], "NEEDS_EXTENSION")

    def test_extension_with_zero_correct_passes(self):
        result = ground_decision([True, True, False], [False, False])
        self.assertIs(result["ground_pass"], True)
        self.assertEqual(result["status"], "PASS")

    def test_extension_with_one_or_two_correct_filters(self):
        for extra in ([True, False], [False, True], [True, True]):
            with self.subTest(extra=extra):
                result = ground_decision([True, True, False], extra)
                self.assertIs(result["ground_pass"], False)
                self.assertEqual(result["status"], "FILTERED")

    def test_incomplete_initial_is_not_a_ground_pass_or_filter(self):
        for initial in ([], [False], [False, False], [False] * 4):
            with self.subTest(initial=initial):
                result = ground_decision(initial)
                self.assertIsNone(result["ground_pass"])
                self.assertEqual(result["status"], "EVALUATION_INCOMPLETE")

    def test_incomplete_extension_is_not_a_ground_pass_or_filter(self):
        for extra in ([], [False], [False] * 3):
            with self.subTest(extra=extra):
                result = ground_decision([True, True, False], extra)
                self.assertIsNone(result["ground_pass"])
                self.assertEqual(result["status"], "EVALUATION_INCOMPLETE")


class SelectionTests(unittest.TestCase):
    def assert_fraction(self, result, expected):
        self.assertEqual(result["full_partial_mean"]["fraction"], str(expected))
        self.assertAlmostEqual(result["full_partial_mean"]["value"], float(expected))

    def test_first_question_is_selected_despite_original_zero_and_no_exact_success(self):
        scores = {
            "perm_00": [Fraction(0)] * 3,
            "perm_01": [Fraction(1, 3)] * 3,
            "perm_02": [Fraction(1, 3), Fraction(0), Fraction(0)],
            "perm_03": [Fraction(0)] * 3,
            "perm_04": [Fraction(2, 3)] * 3,
            "perm_05": [Fraction(0)] * 3,
        }
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_fraction(result, Fraction(5, 27))
        self.assertEqual(result["permutation_means"]["perm_02"]["fraction"], "1/9")

    def test_second_question_is_selected_without_gain_or_position_gate(self):
        means = (Fraction(1, 3), Fraction(0), Fraction(0),
                 Fraction(0), Fraction(1, 3), Fraction(1))
        scores = {variant: [mean] * 3 for variant, mean in zip(VARIANTS, means)}
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_fraction(result, Fraction(5, 18))

    def test_original_full_zero_does_not_stop_other_permutations(self):
        scores = uniform_scores(Fraction(1, 3))
        scores["perm_00"] = [Fraction(0)] * 3
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_fraction(result, Fraction(5, 18))

    def test_negative_contributions_are_not_clipped(self):
        scores = uniform_scores(Fraction(1, 3))
        scores["perm_00"] = [Fraction(-1, 3)] * 3
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_fraction(result, Fraction(2, 9))
        self.assertEqual(result["permutation_means"]["perm_00"]["fraction"], "-1/3")

    def test_signed_cancellation_at_zero_does_not_select(self):
        scores = uniform_scores(Fraction(1, 5))
        scores["perm_00"] = [Fraction(-1)] * 3
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], False)
        self.assert_fraction(result, Fraction(0))

    def test_zero_or_negative_mean_does_not_select(self):
        for value in (Fraction(0), Fraction(-1, 3), Fraction(-1)):
            with self.subTest(value=value):
                result = select_question(passing_ground(), uniform_scores(value))
                self.assertIs(result["selected_by_policy"], False)
                self.assert_fraction(result, value)

    def test_perfect_mean_does_not_select(self):
        result = select_question(passing_ground(), uniform_scores(Fraction(1)))
        self.assertIs(result["selected_by_policy"], False)
        self.assert_fraction(result, Fraction(1))

    def test_strictly_positive_mean_has_no_extra_lower_threshold(self):
        scores = uniform_scores(Fraction(0))
        scores["perm_05"][2] = Fraction(1, 3)
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_fraction(result, Fraction(1, 54))

    def test_mean_below_one_selects_even_with_many_perfect_answers(self):
        scores = uniform_scores(Fraction(1))
        scores["perm_00"][0] = Fraction(2, 3)
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_fraction(result, Fraction(53, 54))

    def test_repeat_variation_does_not_add_an_admission_gate(self):
        scores = uniform_scores(Fraction(1, 3))
        scores["perm_00"] = [Fraction(-1), Fraction(0), Fraction(1)]
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_fraction(result, Fraction(5, 18))

    def test_ground_filtered_does_not_select(self):
        ground = ground_decision([True, True, True])
        result = select_question(ground, uniform_scores(Fraction(1, 3)))
        self.assertIs(result["selected_by_policy"], False)
        self.assertEqual(result["selection_reason"], "GROUND_FILTERED")

    def test_ground_needs_extension_is_incomplete(self):
        ground = ground_decision([True, True, False])
        result = select_question(ground, uniform_scores(Fraction(1, 3)))
        self.assertIsNone(result["selected_by_policy"])
        self.assertEqual(result["selection_reason"], "GROUND_EVALUATION_INCOMPLETE")

    def test_ground_incomplete_is_incomplete(self):
        result = select_question(ground_decision([False]), uniform_scores(Fraction(1, 3)))
        self.assertIsNone(result["selected_by_policy"])
        self.assertEqual(result["selection_reason"], "GROUND_EVALUATION_INCOMPLETE")

    def test_missing_or_extra_permutation_is_incomplete(self):
        missing = uniform_scores(Fraction(1, 3))
        del missing["perm_05"]
        extra = uniform_scores(Fraction(1, 3))
        extra["perm_06"] = [Fraction(1, 3)] * 3
        for scores in (missing, extra):
            with self.subTest(variants=list(scores)):
                result = select_question(passing_ground(), scores)
                self.assertIsNone(result["selected_by_policy"])
                self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")

    def test_missing_or_extra_repeat_is_incomplete(self):
        for count in (0, 1, 2, 4):
            with self.subTest(count=count):
                scores = uniform_scores(Fraction(1, 3))
                scores["perm_00"] = [Fraction(1, 3)] * count
                result = select_question(passing_ground(), scores)
                self.assertIsNone(result["selected_by_policy"])
                self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")

    def test_null_score_is_incomplete(self):
        scores = uniform_scores(Fraction(1, 3))
        scores["perm_03"][1] = None
        result = select_question(passing_ground(), scores)
        self.assertIsNone(result["selected_by_policy"])
        self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")

    def test_data_errors_prevent_selection(self):
        result = select_question(passing_ground(), uniform_scores(Fraction(1, 3)),
                                 data_errors=["source_response_hash_mismatch"])
        self.assertIsNone(result["selected_by_policy"])
        self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")

    def test_missing_full_data_is_reported_before_ground_filter(self):
        result = select_question(ground_decision([True, True, True]), {})
        self.assertIsNone(result["selected_by_policy"])
        self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")


if __name__ == "__main__":
    unittest.main()
