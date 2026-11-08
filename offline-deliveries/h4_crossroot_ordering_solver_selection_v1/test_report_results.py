"""Exercise cross-root report bindings with synthetic structured responses."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import report_results as reports


class CrossRootReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        shutil.copytree(reports.HERE / "run", self.run,
                        ignore=shutil.ignore_patterns("responses", "question_status",
                                                       "STARTED.json", "COMPLETE.json",
                                                       "QUESTION_STATUS.json", "FAILED.json"))
        (self.run / "responses").mkdir()
        self.manifest = json.loads((self.run / "manifest.json").read_text())
        self.questions = json.loads((self.run / "questions.json").read_text())["candidates"]

    def answer(self, row, question, canonical_prediction):
        inverse = {canonical: display for display, canonical in
                   row["variant"]["display_to_canonical"].items()}
        response = {
            "request_id": row["request_id"], "candidate_id": row["candidate_id"],
            "root_id": row["root_id"], "source_sha256": row["source_sha256"],
            "mode": row["mode"], "variant_id": row["variant"]["variant_id"],
            "seed": row["seed"], "prompt_sha256": row["prompt_sha256"],
            "raw_output": "<box>" + ",".join(inverse[item] for item in canonical_prediction) + "</box>",
            "model_identifier": self.manifest["model"],
            "metadata": {
                "temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
                "enable_thinking": False, "sampling_seed": row["seed"],
                "finish_reason": "stop",
                "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                "structured_output_version": "finite_choice_set_v1",
            },
            "displayed_gold": row["variant"]["displayed_gold"],
        }
        (self.run / "responses" / (row["request_id"] + ".json")).write_text(
            json.dumps(response) + "\n", encoding="utf-8")
        return row["request_id"]

    def prepare(self):
        executed, statuses = [], []
        for index, question in enumerate(self.questions):
            qid = question["candidate_id"]
            rows = [row for row in self.manifest["requests"] if row["candidate_id"] == qid]
            start = len(executed)
            ground = [row for row in rows if row["mode"] == "no_context"][:3]
            for row in ground:
                prediction = list(reversed(question["gold"])) if index == 0 else question["gold"]
                executed.append(self.answer(row, question, prediction))
            decision = reports.policy.ground_decision([False] * 3 if index == 0 else [True] * 3)
            if index == 0:
                for row in [row for row in rows if row["mode"] == "full_context"]:
                    prediction = deepcopy(question["gold"])
                    prediction[:2] = reversed(prediction[:2])
                    executed.append(self.answer(row, question, prediction))
            statuses.append({
                "candidate_id": qid, "root_id": question["root_id"],
                "ground": decision,
                "status": "GROUND_PASS_ALL_FULL_EVALUATED" if index == 0 else "GROUND_FILTERED_FULL_SKIPPED",
                "executed_request_ids": executed[start:],
                "full_skipped_reason": None if index == 0 else "GROUND_FILTERED",
            })
        (self.run / "COMPLETE.json").write_text(json.dumps({
            "executed_request_ids": executed, "actual_call_count": len(executed)}), encoding="utf-8")
        (self.run / "QUESTION_STATUS.json").write_text(json.dumps({"questions": statuses}), encoding="utf-8")

    def test_two_roots_distinct_gold_and_adaptive_calls(self):
        self.prepare()
        result = reports.analyze(self.run)
        self.assertEqual(result["counts"]["new_solver_calls"], 18)
        self.assertEqual(result["counts"]["selected_questions"], 1)
        self.assertEqual(result["counts"]["ground_filtered_questions"], 1)
        selected = result["questions"][self.questions[0]["candidate_id"]]
        self.assertEqual(selected["full_partial_mean"]["fraction"], "13/15")
        self.assertEqual(selected["canonical_question"]["gold"], self.questions[0]["gold"])
        self.assertEqual(selected["ground_cache"]["bindings"]["source_sha256"],
                         result["source_hashes_by_root"][self.questions[0]["root_id"]])

    def test_wrong_response_root_is_rejected(self):
        self.prepare()
        path = next((self.run / "responses").glob("*.json"))
        response = json.loads(path.read_text())
        response["root_id"] = self.questions[1]["root_id"]
        path.write_text(json.dumps(response), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "RESPONSE_SOURCE_BINDING_MISMATCH"):
            reports.analyze(self.run)


if __name__ == "__main__":
    unittest.main()
