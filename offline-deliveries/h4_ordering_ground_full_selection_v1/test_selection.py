from copy import deepcopy
from fractions import Fraction
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import select_offline

from selection import (
    FULL_ROLLOUTS,
    GROUND_EXTENSION,
    GROUND_VARIANTS,
    GROUND_VERSION,
    VARIANTS,
    VERSION,
    ground_decision,
    ordering_metrics,
    select_question,
)


def uniform_scores(value):
    return {variant: [value] for variant in VARIANTS}


def passing_ground():
    return ground_decision([False, False, False])


def load_evidence():
    here = Path(__file__).resolve().parent
    plan = json.loads((here / "SOURCE_PLAN.json").read_text(encoding="utf-8"))
    paths = plan["paths"]
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    frozen = json.loads(Path(paths["questions"]).read_text(encoding="utf-8"))
    gold = json.loads(Path(paths["gold"]).read_text(encoding="utf-8"))
    question = next(q for q in frozen["candidates"] if q["candidate_id"] == plan["question_id"])
    question = {**question, "gold": gold[plan["question_id"]]}
    identity = json.loads(Path(paths["model_identity"]).read_text(encoding="utf-8"))
    requests = {request["request_id"]: request for request in manifest["requests"]
                if request["candidate_id"] == question["candidate_id"]}
    responses = {request_id: json.loads((Path(paths["baseline_run"]) / "responses" /
                                        f"{request_id}.json").read_text(encoding="utf-8"))
                 for request_id in requests}
    spec = importlib.util.spec_from_file_location("ordering_test_box_protocol", paths["boxed_protocol"])
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    return {"plan": plan, "manifest": manifest, "question": question,
            "identity": identity, "requests": requests, "responses": responses,
            "protocol": protocol}


class EvidenceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = load_evidence()

    def setUp(self):
        for key in ("plan", "manifest", "question", "identity", "requests", "responses"):
            setattr(self, key, deepcopy(self.evidence[key]))
        self.protocol = self.evidence["protocol"]

    def request_id(self, variant, mode="full_context"):
        return f"{self.question['candidate_id']}__{variant}__{mode}__s00"

    def canonical_from_raw(self, request, response):
        variant = request["variant"]
        parsed = self.protocol.parse_structured_box(
            response["raw_output"], question_type="ordering", allowed_item_ids=variant["items"])
        self.assertEqual(parsed["parse_status"], "success")
        return [variant["display_to_canonical"][item] for item in parsed["parsed_answer"]]


class OrderingMetricsTests(unittest.TestCase):
    def setUp(self):
        self.gold = ["I3", "I4", "I2", "I6", "I1", "I5"]

    def test_complete_gold_is_perfect(self):
        result = ordering_metrics(self.gold, self.gold)
        self.assertIs(result["exact_match"], True)
        self.assertEqual(result["pair_count"], 15)
        self.assertEqual(result["correct_pairs"], 15)
        self.assertEqual(result["incorrect_pairs"], 0)
        self.assertEqual(result["pairwise_accuracy"]["fraction"], "1")
        self.assertEqual(result["partial_score"]["fraction"], "1")
        self.assertEqual(result["inverted_pairs"], [])

    def test_reverse_gold_preserves_negative_one(self):
        result = ordering_metrics(list(reversed(self.gold)), self.gold)
        self.assertIs(result["exact_match"], False)
        self.assertEqual(result["correct_pairs"], 0)
        self.assertEqual(result["incorrect_pairs"], 15)
        self.assertEqual(result["pairwise_accuracy"]["fraction"], "0")
        self.assertEqual(result["partial_score"]["fraction"], "-1")
        self.assertEqual(len(result["inverted_pairs"]), 15)

    def test_single_adjacent_swap_counts_one_inverted_pair(self):
        prediction = self.gold.copy()
        prediction[3], prediction[4] = prediction[4], prediction[3]
        result = ordering_metrics(prediction, self.gold)
        self.assertIs(result["exact_match"], False)
        self.assertEqual(result["correct_pairs"], 14)
        self.assertEqual(result["incorrect_pairs"], 1)
        self.assertEqual(result["pairwise_accuracy"]["fraction"], "14/15")
        self.assertEqual(result["partial_score"]["fraction"], "13/15")
        self.assertEqual(result["inverted_pairs"], [{
            "gold_before": "I6", "gold_after": "I1",
            "predicted_before": "I1", "predicted_after": "I6",
        }])

    def test_two_item_order_has_one_pair(self):
        result = ordering_metrics(["later", "earlier"], ["earlier", "later"])
        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["partial_score"]["fraction"], "-1")

    def test_uniform_random_order_has_zero_expected_net_score(self):
        gold = ["content-z", "content-a", "content-m", "content-c"]
        scores = [ordering_metrics(list(order), gold) for order in permutations(gold)]
        mean = sum((Fraction(score["partial_score"]["fraction"])
                    for score in scores), Fraction()) / len(scores)
        accuracy = sum((Fraction(score["pairwise_accuracy"]["fraction"])
                       for score in scores), Fraction()) / len(scores)
        self.assertEqual(mean, Fraction(0))
        self.assertEqual(accuracy, Fraction(1, 2))

    def test_missing_extra_or_unknown_prediction_ids_are_invalid(self):
        for prediction in (self.gold[:-1], self.gold + ["other"],
                           self.gold[:-1] + ["other"]):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):
                    ordering_metrics(prediction, self.gold)

    def test_duplicate_ids_are_invalid_in_prediction_and_gold(self):
        duplicate = self.gold[:-1] + [self.gold[0]]
        for prediction, gold in ((duplicate, self.gold), (self.gold, duplicate),
                                 (duplicate, duplicate)):
            with self.subTest(prediction=prediction, gold=gold):
                with self.assertRaises(ValueError):
                    ordering_metrics(prediction, gold)

    def test_empty_or_single_event_order_is_invalid(self):
        for order in ([], ["I1"]):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    ordering_metrics(order, order)


class GroundDecisionTests(unittest.TestCase):
    def test_policy_pins_distinct_existing_displays(self):
        self.assertEqual(VERSION, "ordering_ground_full_selection_v1")
        self.assertEqual(GROUND_VERSION, "exact_k3_adaptive_k5_fixed_displays_v1")
        self.assertEqual(VARIANTS, tuple(f"perm_{i:02d}" for i in range(12)))
        self.assertEqual(GROUND_VARIANTS, ("perm_00", "perm_01", "perm_02"))
        self.assertEqual(GROUND_EXTENSION, ("perm_03", "perm_04"))
        self.assertEqual(FULL_ROLLOUTS, 1)

    def test_zero_or_one_initial_exact_answer_passes(self):
        for initial in ([False, False, False], [True, False, False],
                        [False, True, False], [False, False, True]):
            with self.subTest(initial=initial):
                result = ground_decision(initial)
                self.assertIs(result["ground_pass"], True)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["evaluated_count"], 3)
                self.assertEqual(result["exact_correct"], sum(initial))

    def test_three_initial_exact_answers_filter(self):
        result = ground_decision([True, True, True])
        self.assertIs(result["ground_pass"], False)
        self.assertEqual(result["status"], "FILTERED")
        self.assertEqual(result["exact_correct"], 3)
        self.assertEqual(result["evaluated_count"], 3)

    def test_two_initial_exact_answers_need_extension(self):
        for initial in ([False, True, True], [True, False, True],
                        [True, True, False]):
            with self.subTest(initial=initial):
                result = ground_decision(initial)
                self.assertIsNone(result["ground_pass"])
                self.assertEqual(result["status"], "NEEDS_EXTENSION")
                self.assertEqual(result["exact_correct"], 2)
                self.assertEqual(result["evaluated_count"], 3)

    def test_two_of_five_exact_answers_pass(self):
        result = ground_decision([True, True, False], [False, False])
        self.assertIs(result["ground_pass"], True)
        self.assertEqual(result["exact_correct"], 2)
        self.assertEqual(result["evaluated_count"], 5)

    def test_three_or_four_of_five_exact_answers_filter(self):
        for extra in ([True, False], [False, True], [True, True]):
            with self.subTest(extra=extra):
                result = ground_decision([True, True, False], extra)
                self.assertIs(result["ground_pass"], False)
                self.assertEqual(result["exact_correct"], 2 + sum(extra))
                self.assertEqual(result["evaluated_count"], 5)

    def test_missing_or_non_boolean_initial_answers_are_unresolved(self):
        for initial in ([], [False], [False, False], [False] * 4,
                        [False, None, False], [False, 0, False],
                        [False, "false", False]):
            with self.subTest(initial=initial):
                result = ground_decision(initial)
                self.assertIsNone(result["ground_pass"])
                self.assertEqual(result["status"], "EVALUATION_INCOMPLETE")

    def test_missing_or_non_boolean_extension_answers_are_unresolved(self):
        for extra in ([], [False], [False] * 3, [False, None], [False, 0]):
            with self.subTest(extra=extra):
                result = ground_decision([True, True, False], extra)
                self.assertIsNone(result["ground_pass"])
                self.assertEqual(result["status"], "EVALUATION_INCOMPLETE")


class SelectionTests(unittest.TestCase):
    def assert_mean(self, result, expected):
        self.assertEqual(result["full_partial_mean"]["fraction"], str(expected))
        self.assertAlmostEqual(result["full_partial_mean"]["value"], float(expected))

    def test_no_full_exact_success_is_required(self):
        result = select_question(passing_ground(), uniform_scores(Fraction(13, 15)))
        self.assertIs(result["selected_by_policy"], True)
        self.assert_mean(result, Fraction(13, 15))
        self.assertEqual(result["selection_reason"],
                         "GROUND_PASS_AND_FULL_PARTIAL_IN_RANGE")

    def test_negative_display_score_is_retained_and_does_not_veto(self):
        scores = uniform_scores(Fraction(1, 3))
        scores["perm_00"] = [Fraction(-1)]
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_mean(result, Fraction(2, 9))
        self.assertEqual(result["permutation_means"]["perm_00"]["fraction"], "-1")
        self.assertEqual(result["permutation_score_range"]["fraction"], "4/3")

    def test_zero_first_display_does_not_stop_complete_evaluation(self):
        scores = uniform_scores(Fraction(1, 3))
        scores["perm_00"] = [Fraction(0)]
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_mean(result, Fraction(11, 36))

    def test_signed_cancellation_at_zero_is_not_selected(self):
        scores = uniform_scores(Fraction(1, 11))
        scores["perm_00"] = [Fraction(-1)]
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], False)
        self.assert_mean(result, Fraction(0))
        self.assertEqual(result["selection_reason"], "FULL_PARTIAL_NONPOSITIVE")

    def test_zero_and_negative_means_are_not_selected(self):
        for value in (Fraction(0), Fraction(-1, 15), Fraction(-1)):
            with self.subTest(value=value):
                result = select_question(passing_ground(), uniform_scores(value))
                self.assertIs(result["selected_by_policy"], False)
                self.assert_mean(result, value)
                self.assertEqual(result["selection_reason"], "FULL_PARTIAL_NONPOSITIVE")

    def test_perfect_mean_is_not_a_new_hard_item(self):
        result = select_question(passing_ground(), uniform_scores(Fraction(1)))
        self.assertIs(result["selected_by_policy"], False)
        self.assert_mean(result, Fraction(1))
        self.assertEqual(result["selection_reason"], "FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM")

    def test_strictly_positive_mean_has_no_extra_minimum(self):
        scores = uniform_scores(Fraction(0))
        scores["perm_11"] = [Fraction(1, 15)]
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_mean(result, Fraction(1, 180))

    def test_mean_below_one_can_select_many_perfect_answers(self):
        scores = uniform_scores(Fraction(1))
        scores["perm_00"] = [Fraction(13, 15)]
        result = select_question(passing_ground(), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assert_mean(result, Fraction(89, 90))

    def test_ground_filter_prevents_selection(self):
        result = select_question(ground_decision([True, True, True]),
                                 uniform_scores(Fraction(2, 3)))
        self.assertIs(result["selected_by_policy"], False)
        self.assertEqual(result["selection_reason"], "GROUND_FILTERED")

    def test_unresolved_ground_stays_unresolved(self):
        for ground in (ground_decision([True, True, False]), ground_decision([False])):
            with self.subTest(ground=ground):
                result = select_question(ground, uniform_scores(Fraction(2, 3)))
                self.assertIsNone(result["selected_by_policy"])
                self.assertEqual(result["selection_reason"], "GROUND_EVALUATION_INCOMPLETE")

    def test_missing_or_extra_display_is_unresolved(self):
        missing = uniform_scores(Fraction(2, 3))
        del missing["perm_11"]
        extra = uniform_scores(Fraction(2, 3))
        extra["perm_12"] = [Fraction(2, 3)]
        for scores in (missing, extra):
            with self.subTest(variants=list(scores)):
                result = select_question(passing_ground(), scores)
                self.assertIsNone(result["selected_by_policy"])
                self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")
                self.assertIsNone(result["full_partial_mean"])

    def test_missing_or_extra_response_is_unresolved(self):
        for count in (0, 2, 3):
            with self.subTest(count=count):
                scores = uniform_scores(Fraction(2, 3))
                scores["perm_00"] = [Fraction(2, 3)] * count
                result = select_question(passing_ground(), scores)
                self.assertIsNone(result["selected_by_policy"])
                self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")

    def test_missing_invalid_or_out_of_range_score_is_unresolved(self):
        for score in (None, 0.5, True, 1, Fraction(2), Fraction(-2)):
            with self.subTest(score=score):
                scores = uniform_scores(Fraction(2, 3))
                scores["perm_05"] = [score]
                result = select_question(passing_ground(), scores)
                self.assertIsNone(result["selected_by_policy"])
                self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")
                self.assertIsNone(result["full_partial_mean"])

    def test_mapping_or_provenance_error_is_unresolved(self):
        result = select_question(passing_ground(), uniform_scores(Fraction(2, 3)),
                                 data_errors=["source_response_hash_mismatch"])
        self.assertIsNone(result["selected_by_policy"])
        self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")

    def test_missing_full_data_is_not_hidden_by_ground_filter(self):
        result = select_question(ground_decision([True, True, True]), {})
        self.assertIsNone(result["selected_by_policy"])
        self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")


class ExistingOrderingEvidenceTests(EvidenceTestCase):
    def test_current_twelve_full_answers_have_two_thirds_net_mean(self):
        gold = self.question["gold"]
        scores, exact_variants = {}, []
        pairwise_sum = Fraction()
        for variant in VARIANTS:
            request_id = self.request_id(variant)
            prediction = self.canonical_from_raw(self.requests[request_id], self.responses[request_id])
            metrics = ordering_metrics(prediction, gold)
            scores[variant] = [Fraction(metrics["partial_score"]["fraction"])]
            pairwise_sum += Fraction(metrics["pairwise_accuracy"]["fraction"])
            if metrics["exact_match"]:
                exact_variants.append(variant)
        self.assertEqual(exact_variants, ["perm_02", "perm_11"])
        self.assertEqual(pairwise_sum / len(VARIANTS), Fraction(5, 6))
        ground_answers = []
        for variant in GROUND_VARIANTS:
            request_id = self.request_id(variant, "no_context")
            prediction = self.canonical_from_raw(self.requests[request_id], self.responses[request_id])
            ground_answers.append(ordering_metrics(prediction, gold)["exact_match"])
        self.assertEqual(ground_answers, [False, False, False])
        result = select_question(ground_decision(ground_answers), scores)
        self.assertIs(result["selected_by_policy"], True)
        self.assertEqual(result["full_partial_mean"]["fraction"], "2/3")
        self.assertEqual(result["permutation_score_min"]["fraction"], "1/5")
        self.assertEqual(result["permutation_score_max"]["fraction"], "1")
        self.assertEqual(result["permutation_score_range"]["fraction"], "4/5")


class ScoreResponseTests(EvidenceTestCase):
    def setUp(self):
        super().setUp()
        request_id = self.request_id("perm_00")
        self.request = self.requests[request_id]
        self.response = self.responses[request_id]

    def score(self):
        return select_offline.score_response(
            self.question, self.request, self.response, self.manifest, self.protocol)

    def synchronize_displayed_gold(self):
        inverse = {canonical: display for display, canonical in
                   self.request["variant"]["display_to_canonical"].items()}
        gold = [inverse[item] for item in self.question["gold"]]
        self.request["variant"]["displayed_gold"] = gold
        self.response["displayed_gold"] = gold.copy()

    def test_valid_raw_is_parsed_mapped_and_scored(self):
        result = self.score()
        self.assertIs(result["evaluation_valid"], True)
        self.assertIs(result["exact_match"], False)
        self.assertEqual(result["canonical_prediction"], ["I3", "I4", "I2", "I1", "I6", "I5"])
        self.assertEqual(result["partial_score"]["fraction"], "13/15")
        self.assertEqual(result["inverted_pair_contents"][0]["gold_before_text"], self.question["items"]["I6"])

    def test_stored_score_can_be_missing_or_wrong_without_changing_raw_score(self):
        self.response.pop("score")
        self.assertEqual(self.score()["partial_score"]["fraction"], "13/15")
        self.response["score"] = {"correct": True, "canonical_prediction": self.question["gold"],
                                  "metrics": {"pairwise_accuracy": 1}}
        self.assertIs(self.score()["exact_match"], False)
        self.assertEqual(self.score()["partial_score"]["fraction"], "13/15")

    def test_changed_valid_raw_controls_perfect_and_negative_score(self):
        self.response["raw_output"] = "<box>" + ",".join(self.response["displayed_gold"]) + "</box>"
        self.assertIs(self.score()["exact_match"], True)
        self.assertEqual(self.score()["partial_score"]["fraction"], "1")
        self.response["raw_output"] = "<box>" + ",".join(reversed(self.response["displayed_gold"])) + "</box>"
        self.assertEqual(self.score()["partial_score"]["fraction"], "-1")

    def test_bijective_mapping_with_changed_content_is_invalid(self):
        self.request["variant"]["items"]["I1"] = "Changed event content"
        with self.assertRaisesRegex(ValueError, "DISPLAY_CONTENT_CHANGED"):
            self.score()

    def test_non_bijective_mapping_is_invalid(self):
        self.request["variant"]["display_to_canonical"]["I1"] = "I1"
        with self.assertRaisesRegex(ValueError, "DISPLAY_MAPPING_NOT_BIJECTIVE"):
            self.score()

    def test_valid_content_mapping_outside_fixed_display_plan_is_invalid(self):
        variant = self.request["variant"]
        mapping = variant["display_to_canonical"]
        mapping["I1"], mapping["I2"] = mapping["I2"], mapping["I1"]
        variant["items"] = {display: self.question["items"][canonical]
                            for display, canonical in mapping.items()}
        self.synchronize_displayed_gold()
        with self.assertRaisesRegex(ValueError, "FIXED_DISPLAY_ORDER_MISMATCH"):
            self.score()

    def test_relabeling_mapping_keys_cannot_bypass_fixed_display_plan(self):
        variant = self.request["variant"]
        mapping = variant["display_to_canonical"]
        variant["display_to_canonical"] = {
            {"I1": "I2", "I2": "I1"}.get(display, display): canonical
            for display, canonical in mapping.items()
        }
        variant["items"] = {display: self.question["items"][canonical]
                            for display, canonical in variant["display_to_canonical"].items()}
        self.synchronize_displayed_gold()
        with self.assertRaisesRegex(ValueError, "FIXED_DISPLAY_ORDER_MISMATCH"):
            self.score()

    def test_request_or_response_displayed_gold_mismatch_is_invalid(self):
        for location in ("request", "response"):
            with self.subTest(location=location):
                original = deepcopy(self.request), deepcopy(self.response)
                target = self.request["variant"] if location == "request" else self.response
                target["displayed_gold"] = list(reversed(target["displayed_gold"]))
                with self.assertRaisesRegex(ValueError, "DISPLAYED_GOLD_MISMATCH"):
                    self.score()
                self.request, self.response = original

    def test_invalid_canonical_gold_is_rejected(self):
        self.question["gold"][-1] = "unknown"
        with self.assertRaisesRegex(ValueError, "GOLD_ITEM_SET_MISMATCH"):
            self.score()

    def test_duplicate_incomplete_unknown_and_unboxed_raw_are_invalid(self):
        for raw in ("<box>I1,I2,I3,I4,I5,I5</box>", "<box>I1,I2,I3,I4,I5</box>",
                    "<box>I1,I2,I3,I4,I5,I7</box>", "I1,I2,I3,I4,I5,I6",
                    "text <box>I1,I2,I3,I4,I5,I6</box>",
                    "<box>I1,I2,I3,I4,I5,I6</box><box>I1,I2,I3,I4,I5,I6</box>"):
            with self.subTest(raw=raw):
                self.response["raw_output"] = raw
                with self.assertRaisesRegex(ValueError, "PARSE_FAILURE"):
                    self.score()

    def test_missing_or_non_text_raw_is_invalid(self):
        for raw in (None, 123, []):
            with self.subTest(raw=raw):
                self.response["raw_output"] = raw
                with self.assertRaisesRegex(ValueError, "RAW_OUTPUT_MISSING_OR_NOT_TEXT"):
                    self.score()

    def test_wrong_request_response_identity_is_rejected(self):
        for field, value in (("request_id", "wrong-id"), ("seed", 0),
                             ("prompt_sha256", "0" * 64), ("mode", "no_context"),
                             ("variant_id", "perm_01")):
            with self.subTest(field=field):
                original = self.response[field]
                self.response[field] = value
                with self.assertRaises(ValueError):
                    self.score()
                self.response[field] = original

    def test_invalid_prompt_hash_and_seed_are_rejected_even_when_matched(self):
        for field, value in (("prompt_sha256", "not-a-hash"), ("seed", True)):
            with self.subTest(field=field):
                request_value, response_value = self.request[field], self.response[field]
                self.request[field] = self.response[field] = value
                with self.assertRaisesRegex(ValueError, "REQUEST_SEED_OR_PROMPT_HASH_INVALID"):
                    self.score()
                self.request[field], self.response[field] = request_value, response_value

    def test_wrong_model_is_rejected(self):
        self.response["model_identifier"] = "/some/other/model"
        with self.assertRaisesRegex(ValueError, "SOLVER_MODEL_MISMATCH"):
            self.score()

    def test_wrong_decode_or_sampling_configuration_is_rejected(self):
        for field, value in (("temperature", 0.0), ("top_p", 0.5), ("max_tokens", 64),
                             ("structured_output_api", "other-api"),
                             ("structured_output_version", "other-version"), ("sampling_seed", 0)):
            with self.subTest(field=field):
                original = self.response["metadata"][field]
                self.response["metadata"][field] = value
                with self.assertRaises(ValueError):
                    self.score()
                self.response["metadata"][field] = original

    def test_unfinished_generation_is_rejected_even_with_complete_raw(self):
        for reason in (None, "length", "abort", "error"):
            with self.subTest(reason=reason):
                self.response["metadata"]["finish_reason"] = reason
                with self.assertRaisesRegex(ValueError, "INCOMPLETE_GENERATION"):
                    self.score()

    def test_eos_finish_is_valid(self):
        self.response["metadata"]["finish_reason"] = "eos"
        self.assertIs(self.score()["evaluation_valid"], True)

    def test_invalid_object_or_nested_container_schema_is_value_error(self):
        arguments = {"question": self.question, "request": self.request,
                     "response": self.response, "manifest": self.manifest}
        for argument in arguments:
            for invalid in (None, []):
                with self.subTest(argument=argument, invalid=invalid):
                    changed = deepcopy(arguments)
                    changed[argument] = invalid
                    with self.assertRaises(ValueError):
                        select_offline.score_response(**changed, protocol=self.protocol)
        for argument, keys, invalid in (
            ("question", ("items",), []),
            ("question", ("gold",), "I1,I2,I3,I4,I5,I6"),
            ("request", ("variant",), []),
            ("request", ("variant", "display_to_canonical"), []),
            ("request", ("variant", "items"), []),
            ("response", ("metadata",), []),
            ("response", ("metadata",), None),
        ):
            with self.subTest(argument=argument, keys=keys, invalid=invalid):
                changed = deepcopy(arguments)
                target = changed[argument]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = invalid
                with self.assertRaises(ValueError):
                    select_offline.score_response(**changed, protocol=self.protocol)


class EvaluateQuestionTests(EvidenceTestCase):
    def evaluate(self):
        sources = {self.plan["paths"]["manifest"]:
                   self.plan["hashes"][self.plan["paths"]["manifest"]]}

        def in_memory_load(path, expected, source_hashes):
            request_id = path.stem
            if request_id not in self.responses:
                raise FileNotFoundError(f"MISSING_TEST_RESPONSE:{request_id}")
            source_hashes[str(path)] = expected
            return deepcopy(self.responses[request_id])

        with patch.object(select_offline, "verified_load", side_effect=in_memory_load):
            return select_offline.evaluate_question(
                self.question, self.manifest, self.identity, self.plan, self.protocol, sources)

    def mark_ground_exact(self, variant):
        response = self.responses[self.request_id(variant, "no_context")]
        response["raw_output"] = "<box>" + ",".join(response["displayed_gold"]) + "</box>"

    def test_missing_required_full_is_unresolved_but_ground_cache_is_reusable(self):
        del self.responses[self.request_id("perm_11")]
        result = self.evaluate()
        self.assertIsNone(result["selected_by_policy"])
        self.assertIsNone(result["full_partial_mean"])
        self.assertIsNone(result["full_pairwise_accuracy_mean"])
        self.assertEqual(result["full_evaluated_count"], 11)
        self.assertIs(result["ground_pass"], True)
        self.assertIs(result["ground_cache"]["reusable"], True)
        self.assertEqual(len(result["ground_cache"]["source_responses"]), 3)
        self.assertTrue(any("MISSING_TEST_RESPONSE" in error for error in result["data_errors"]))

    def test_missing_unused_no_context_diagnostic_does_not_block_selection(self):
        del self.responses[self.request_id("perm_11", "no_context")]
        result = self.evaluate()
        self.assertIs(result["selected_by_policy"], True)
        self.assertEqual(result["full_partial_mean"]["fraction"], "2/3")
        self.assertEqual(result["data_errors"], [])
        self.assertIs(result["ground_cache"]["reusable"], True)
        diagnostics = result["diagnostics_not_used_for_selection"]
        self.assertIsNone(diagnostics["no_context_pairwise_accuracy_mean"])
        self.assertIsNone(diagnostics["pairwise_context_gain"])
        self.assertFalse(diagnostics["no_context_responses"][-1]["evaluation_valid"])

    def test_missing_required_ground_prevents_cache_reuse_and_selection(self):
        del self.responses[self.request_id("perm_02", "no_context")]
        result = self.evaluate()
        self.assertIsNone(result["selected_by_policy"])
        self.assertIsNone(result["ground_pass"])
        self.assertIs(result["ground_cache"]["reusable"], False)
        self.assertTrue(result["ground_cache"]["data_errors"])

    def test_k5_extension_uses_fixed_two_displays_and_caches_five_answers(self):
        self.mark_ground_exact("perm_01")
        self.mark_ground_exact("perm_02")
        result = self.evaluate()
        self.assertIs(result["selected_by_policy"], True)
        self.assertEqual(result["ground"]["exact_correct"], 2)
        self.assertEqual(result["ground"]["evaluated_count"], 5)
        cache = result["ground_cache"]
        self.assertIs(cache["reusable"], True)
        self.assertEqual([row["variant_id"] for row in cache["source_responses"]],
                         ["perm_00", "perm_01", "perm_02", "perm_03", "perm_04"])
        self.assertEqual(len(cache["bindings"]["required_evaluations"]), 5)
        self.assertEqual(cache["bindings"]["ground_sampling"], "different_displays_one_response_each")

    def test_k5_three_exact_filters_without_new_calls(self):
        for variant in ("perm_01", "perm_02", "perm_03"):
            self.mark_ground_exact(variant)
        result = self.evaluate()
        self.assertIs(result["selected_by_policy"], False)
        self.assertEqual(result["selection_reason"], "GROUND_FILTERED")
        self.assertEqual(result["ground"]["exact_correct"], 3)
        self.assertEqual(result["ground"]["evaluated_count"], 5)
        self.assertEqual(len(result["ground_cache"]["source_responses"]), 5)

    def test_missing_k5_extension_is_unresolved_and_not_filled_with_zero(self):
        self.mark_ground_exact("perm_01")
        self.mark_ground_exact("perm_02")
        del self.responses[self.request_id("perm_04", "no_context")]
        result = self.evaluate()
        self.assertIsNone(result["selected_by_policy"])
        self.assertIsNone(result["ground_pass"])
        self.assertIs(result["ground_cache"]["reusable"], False)
        self.assertEqual(len(result["ground_cache"]["source_responses"]), 5)
        self.assertIsNone(result["ground_cache"]["source_responses"][-1]["exact_match"])

    def test_missing_full_report_does_not_claim_selected_question(self):
        del self.responses[self.request_id("perm_11")]
        question = self.evaluate()
        result = {
            "questions": {self.question["candidate_id"]: question},
            "counts": {"unique_existing_responses_read": 23,
                       "reused_ground_responses": 3, "required_ground_responses": 3,
                       "selected_questions": 0},
        }
        report = select_offline.report(result)
        self.assertIn("Selected canonical questions: 0", report)
        self.assertNotIn("Selected canonical questions: 1", report)
        self.assertIn("EVALUATION_INCOMPLETE_OR_INVALID", report)
        self.assertIn("unresolved", report)
        self.assertIn("MISSING_TEST_RESPONSE", report)


if __name__ == "__main__":
    unittest.main()
