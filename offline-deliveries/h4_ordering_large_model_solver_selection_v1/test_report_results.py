from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import report_results as reports


class AdaptiveReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "prompts").mkdir()
        (self.run / "responses").mkdir()
        pool = json.loads((reports.GENERATOR / "READY_FOR_EVALUATION.json").read_text())
        self.question = deepcopy(pool["questions"][0])
        self.manifest = {
            "model": "/root/autodl-tmp/Qwen3-4B", "temperature": 0.8,
            "top_p": 0.9, "max_tokens": 32, "enable_thinking": False,
            "input_sha256": {"source.json": "a" * 64},
            "solver_identity": {"model_path": "/root/autodl-tmp/Qwen3-4B", "fingerprint": "b" * 64},
            "code_sha256": {}, "requests": [],
        }
        for mode, indexes in (("no_context", range(5)), ("full_context", range(12))):
            for index in indexes:
                variant_id = f"perm_{index:02d}"
                mapping = {f"I{offset + 1}": f"I{number}" for offset, number in
                           enumerate(reports.scorer.DISPLAY_ORDERS[index])}
                inverse = {canonical: label for label, canonical in mapping.items()}
                rid = f"{self.question['candidate_id']}__{variant_id}__{mode}__s00"
                prompt = "fixture prompt " + rid
                (self.run / "prompts" / (rid + ".txt")).write_text(prompt)
                self.manifest["requests"].append({
                    "candidate_id": self.question["candidate_id"], "question_type": "ordering",
                    "request_id": rid, "mode": mode, "slot": 0, "seed": index,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_tokens": 10,
                    "variant": {"variant_id": variant_id, "display_to_canonical": mapping,
                                "items": {label: self.question["items"][canonical] for label, canonical in mapping.items()},
                                "displayed_gold": [inverse[item] for item in self.question["gold"]]},
                })

    def request(self, mode, index):
        return next(row for row in self.manifest["requests"] if row["mode"] == mode
                    and row["variant"]["variant_id"] == f"perm_{index:02d}")

    def answer(self, mode, index, canonical_prediction, *, raw=None):
        request = self.request(mode, index)
        inverse = {canonical: label for label, canonical in request["variant"]["display_to_canonical"].items()}
        response = {
            "request_id": request["request_id"], "seed": request["seed"],
            "prompt_sha256": request["prompt_sha256"], "mode": mode,
            "variant_id": request["variant"]["variant_id"],
            "displayed_gold": request["variant"]["displayed_gold"],
            "model_identifier": self.manifest["model"],
            "raw_output": raw if raw is not None else "<box>" + ",".join(inverse[item] for item in canonical_prediction) + "</box>",
            "metadata": {"temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
                         "enable_thinking": False, "sampling_seed": request["seed"], "finish_reason": "stop",
                         "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                         "structured_output_version": "finite_choice_set_v1"},
        }
        path = self.run / "responses" / (request["request_id"] + ".json")
        path.write_text(json.dumps(response))
        return request["request_id"]

    def build_run(self, initial, extra=None, *, full="swap"):
        gold = self.question["gold"]
        reverse = list(reversed(gold))
        executed = [self.answer("no_context", index, gold if exact else reverse)
                    for index, exact in enumerate(initial)]
        ground = reports.policy.ground_decision(initial, extra)
        if extra is not None:
            executed += [self.answer("no_context", index + 3, gold if exact else reverse)
                         for index, exact in enumerate(extra)]
        if ground["ground_pass"] is True:
            prediction = gold.copy()
            if full == "swap":
                prediction[0], prediction[1] = prediction[1], prediction[0]
            elif full == "reverse":
                prediction = reverse
            executed += [self.answer("full_context", index, prediction) for index in range(12)]
        status = {
            "candidate_id": self.question["candidate_id"], "ground": ground,
            "status": ("FULL_DONE" if ground["ground_pass"] else "GROUND_FILTERED_FULL_SKIPPED"),
            "executed_request_ids": executed.copy(),
            "full_skipped_reason": None if ground["ground_pass"] else "GROUND_FILTERED",
        }
        return executed, status

    def evaluate(self, executed, status):
        return reports.evaluate_question(self.question, self.manifest, status, self.run, {}, executed)

    def test_ground_filtered_has_no_missing_full_errors_or_fabricated_scores(self):
        executed, status = self.build_run([True, True, True])
        result = self.evaluate(executed, status)
        self.assertIs(result["selected_by_policy"], False)
        self.assertEqual(result["selection_reason"], "GROUND_FILTERED_FULL_SKIPPED")
        self.assertEqual(result["data_errors"], [])
        self.assertIsNone(result["full_partial_mean"])
        self.assertEqual(result["full_responses"], [])
        self.assertEqual(result["full_expected_count"], 0)

    def test_partial_positive_without_exact_is_selected_and_cache_only_has_ground(self):
        executed, status = self.build_run([False, False, False])
        result = self.evaluate(executed, status)
        self.assertIs(result["selected_by_policy"], True)
        self.assertEqual(result["full_partial_mean"]["fraction"], "13/15")
        self.assertEqual(result["full_exact_correct"], 0)
        self.assertEqual(len(result["full_responses"]), 12)
        self.assertEqual(len(result["ground_cache"]["source_responses"]), 3)
        self.assertEqual(len(result["ground_cache"]["bindings"]["planned_evaluations"]), 17)
        for row in result["full_responses"]:
            self.assertEqual(row["inverted_pair_contents"][0]["gold_before_text"], self.question["items"]["I1"])

    def test_negative_full_scores_remain_negative(self):
        executed, status = self.build_run([False, False, False], full="reverse")
        result = self.evaluate(executed, status)
        self.assertIs(result["selected_by_policy"], False)
        self.assertEqual(result["full_partial_mean"]["fraction"], "-1")
        self.assertEqual(result["selection_reason"], "FULL_PARTIAL_NONPOSITIVE")

    def test_large_permutation_range_and_zero_exact_do_not_block_positive_mean(self):
        executed, status = self.build_run([False, False, False])
        self.answer("full_context", 0, list(reversed(self.question["gold"])))
        result = self.evaluate(executed, status)
        self.assertIs(result["selected_by_policy"], True)
        self.assertEqual(result["full_partial_mean"]["fraction"], "32/45")
        self.assertEqual(result["permutation_score_range"]["fraction"], "28/15")
        self.assertEqual(result["full_exact_correct"], 0)

    def test_perfect_full_is_not_new_training_selection(self):
        executed, status = self.build_run([False, False, False], full="gold")
        result = self.evaluate(executed, status)
        self.assertIs(result["selected_by_policy"], False)
        self.assertEqual(result["full_exact_rate"]["fraction"], "1")

    def test_adaptive_extension_is_scored_only_when_required(self):
        for extra, passes in (([False, False], True), ([True, False], False)):
            executed, status = self.build_run([True, True, False], extra)
            result = self.evaluate(executed, status)
            self.assertIs(result["ground"]["ground_pass"], passes)
            self.assertEqual(result["ground"]["evaluated_count"], 5)
            self.assertEqual(len(result["ground_cache"]["source_responses"]), 5)

    def test_missing_full_response_is_unresolved_instead_of_zero(self):
        executed, status = self.build_run([False, False, False])
        (self.run / "responses" / (executed[-1] + ".json")).unlink()
        result = self.evaluate(executed, status)
        self.assertIsNone(result["selected_by_policy"])
        self.assertIsNone(result["full_partial_mean"])
        self.assertIsNone(result["full_responses"][-1]["partial_score"])
        self.assertEqual(result["full_evaluated_count"], 11)

    def test_invalid_ground_response_is_unresolved_and_skips_full(self):
        executed = [self.answer("no_context", index, self.question["gold"], raw="bad") for index in range(3)]
        status = {"candidate_id": self.question["candidate_id"],
                  "ground": reports.policy.ground_decision([None, None, None]),
                  "status": "EVALUATION_INCOMPLETE_FULL_SKIPPED",
                  "executed_request_ids": executed, "full_skipped_reason": "GROUND_INVALID"}
        result = self.evaluate(executed, status)
        self.assertIsNone(result["selected_by_policy"])
        self.assertIsNone(result["ground"]["ground_pass"])
        self.assertFalse(result["ground_cache"]["reusable"])
        self.assertEqual(result["full_responses"], [])

    def test_extra_ground_call_violates_adaptive_execution(self):
        executed, status = self.build_run([False, False, False])
        executed.append(self.answer("no_context", 3, self.question["gold"]))
        status["executed_request_ids"] = executed
        result = self.evaluate(executed, status)
        self.assertIsNone(result["selected_by_policy"])
        self.assertIn("UNREQUIRED_SOLVER_CALL_EXECUTED", result["data_errors"])

    def test_corrupted_planned_mapping_is_rejected_even_if_slot_is_skipped(self):
        executed, status = self.build_run([True, True, True])
        self.request("full_context", 11)["variant"]["items"]["I1"] = "changed content"
        with self.assertRaisesRegex(ValueError, "PLANNED_DISPLAY_MAPPING_OR_GOLD_INVALID"):
            self.evaluate(executed, status)

    def test_cache_binds_unused_ground_extension_and_checkpoint(self):
        executed, status = self.build_run([False, False, False])
        first = self.evaluate(executed, status)["ground_cache"]["cache_key"]
        self.request("no_context", 4)["seed"] += 100
        second = self.evaluate(executed, status)["ground_cache"]["cache_key"]
        self.assertNotEqual(first, second)
        self.manifest["solver_identity"]["fingerprint"] = "c" * 64
        self.assertNotEqual(second, self.evaluate(executed, status)["ground_cache"]["cache_key"])

    def test_artifacts_allow_identical_rerun_and_refuse_any_changed_output(self):
        reports.write_artifacts(self.run, {"a.json": {"a": 1}, "b.txt": "hello"})
        reports.write_artifacts(self.run, {"a.json": {"a": 1}, "b.txt": "hello"})
        with self.assertRaises(FileExistsError):
            reports.write_artifacts(self.run, {"a.json": {"a": 2}, "new.txt": "must not be written"})
        self.assertFalse((self.run / "new.txt").exists())


if __name__ == "__main__":
    unittest.main()
