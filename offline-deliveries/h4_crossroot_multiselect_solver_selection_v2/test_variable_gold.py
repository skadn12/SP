"""Offline checks for variable-size Gold and the new all-correct score."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import VARIANTS, display, request_plan, sha, validate_pool
import report_results as report


GOLDS = {
    2: ["B", "E"],
    3: ["A", "C", "F"],
    4: ["A", "C", "D", "F"],
    5: ["A", "B", "D", "E", "F"],
    6: list("ABCDEF"),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(folder: Path, gold: list[str]):
    source_path = folder / "source.json"
    write_json(source_path, {"root_id": "cc-train-episode-test", "turns": [
        {"turn_ref": "R02:S01:T001", "text": "source quote"}]})
    question = {
        "candidate_id": "test-multi-01", "root_id": "cc-train-episode-test",
        "question_type": "multi_select", "question": "Select all that apply.",
        "options": {label: f"claim {label}" for label in "ABCDEF"},
        "gold": gold,
        "option_evidence": {label: {"turn_ref": "R02:S01:T001", "quote": "source quote",
                                    "relation": "supports" if label in gold else "contradicts",
                                    "reason": "Fixture evidence."}
                            for label in "ABCDEF"},
        "source_path": str(source_path), "source_sha256": sha(source_path),
    }
    pool_path = folder / "READY_FOR_EVALUATION.json"
    write_json(pool_path, {"status": "CONTENT_AUDITED_READY_FOR_SOLVER",
                           "candidate_count": 1,
                           "provenance": {"requested_model": "Qwen/Qwen3.6-35B-A3B",
                                          "returned_model": "Qwen/Qwen3.6-35B-A3B"},
                           "questions": [question]})
    return pool_path, question


def boxed(labels: list[str]) -> str:
    return "<box>" + ",".join(labels) + "</box>"


class VariableGoldTests(unittest.TestCase):
    def test_pool_and_permutations_accept_two_through_six_gold(self):
        for g, gold in GOLDS.items():
            with self.subTest(g=g), TemporaryDirectory() as name:
                pool_path, question = fixture(Path(name), gold)
                _, questions, sources = validate_pool(pool_path)
                self.assertEqual(questions, [question])
                requests, prompts = request_plan(questions, sources, lambda *args: "prompt")
                self.assertEqual(len(requests), 23)
                self.assertEqual(len(prompts), 23)
                displays = [display(question, shift) for shift in range(6)]
                for variant in displays:
                    mapping = variant["display_to_canonical"]
                    self.assertEqual({mapping[label] for label in variant["displayed_gold"]},
                                     set(gold))
                distinct_gold_sets = {frozenset(variant["displayed_gold"]) for variant in displays}
                self.assertEqual(len(distinct_gold_sets) == 1, g == 6)

    def test_zero_and_one_gold_are_rejected(self):
        for gold in ([], ["A"]):
            with self.subTest(gold=gold), TemporaryDirectory() as name:
                pool_path, _ = fixture(Path(name), gold)
                with self.assertRaisesRegex(ValueError, "SIX_OPTION_TWO_TO_SIX_GOLD"):
                    validate_pool(pool_path)

    def test_score_uses_negative_false_positive_only_when_false_options_exist(self):
        policy = report.policy
        self.assertEqual(str(policy.partial_score(1, 1, 2, 6)), "1/4")
        self.assertEqual(str(policy.partial_score(3, 2, 4, 6)), "-1/4")
        self.assertEqual(str(policy.partial_score(4, 1, 5, 6)), "-1/5")
        self.assertEqual(str(policy.partial_score(5, 0, 6, 6)), "5/6")
        with self.assertRaises(ValueError):
            policy.partial_score(1, 0, 1, 6)

    def test_report_recomputes_full_scores_for_each_gold_size(self):
        for g, gold in GOLDS.items():
            with self.subTest(g=g), TemporaryDirectory() as name:
                folder = Path(name)
                pool_path, question = fixture(folder, gold)
                _, questions, sources = validate_pool(pool_path)
                planned, _ = request_plan(questions, sources, lambda *args: "prompt")
                qid = question["candidate_id"]
                requests = {(qid, row["mode"], row["variant"]["variant_id"], row["slot"]): row
                            for row in planned}
                manifest = {
                    "model": "/root/autodl-tmp/Qwen3-4B",
                    "solver_identity": {"model_path": "/root/autodl-tmp/Qwen3-4B"},
                    "generation_config": {"temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
                                          "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                                          "structured_output_version": "finite_choice_set_v1",
                                          "enable_thinking": False},
                    "code_sha256": {str(report.POLICY_PATH): sha(report.POLICY_PATH)},
                }
                run = folder / "run"
                executed = []

                def add(row, canonical_prediction):
                    mapping = row["variant"]["display_to_canonical"]
                    shown = [label for label, canonical in mapping.items()
                             if canonical in canonical_prediction]
                    raw = boxed(shown)
                    score = report.scorer.score_response(question, row,
                                                         {"request_id": row["request_id"],
                                                          "raw_output": raw})
                    self.assertEqual(score["scoring_status"], "SCORED")
                    j = report.policy.partial_score(score["tp"], score["fp"], g, 6)
                    response = {
                        "request_id": row["request_id"], "candidate_id": qid,
                        "root_id": row["root_id"], "source_sha256": row["source_sha256"],
                        "mode": row["mode"], "variant_id": row["variant"]["variant_id"],
                        "slot": row["slot"], "seed": row["seed"],
                        "prompt_sha256": row["prompt_sha256"], "raw_output": raw,
                        "model_identifier": manifest["model"],
                        "metadata": {**manifest["generation_config"], "finish_reason": "stop"},
                        "displayed_gold": row["variant"]["displayed_gold"],
                        "partial_score_fraction": str(j), "evaluation_valid": True,
                        "data_errors": [],
                    }
                    write_json(run / "responses" / (row["request_id"] + ".json"), response)
                    executed.append(row["request_id"])

                for slot in range(3):
                    add(requests[qid, "no_context", "perm_00", slot], ["A"])
                for variant in VARIANTS:
                    for slot in range(3):
                        add(requests[qid, "full_context", variant, slot], gold[:-1])
                ground = report.policy.ground_decision([False, False, False])
                status = {"candidate_id": qid, "root_id": question["root_id"],
                          "ground": ground, "status": "GROUND_PASS_ALL_FULL_EVALUATED",
                          "executed_request_ids": executed, "full_skipped_reason": None}
                result = report.evaluate_question(question, manifest, status, requests,
                                                  executed, run)
                self.assertEqual(result["data_errors"], [])
                self.assertTrue(result["selected_by_policy"])
                self.assertEqual(result["full_partial_mean"]["fraction"], str(
                    report.policy.partial_score(g - 1, 0, g, 6)))
                self.assertEqual(result["full_exact_correct"], 0)
                self.assertEqual(len(result["full_responses"]), 18)


if __name__ == "__main__":
    unittest.main()
