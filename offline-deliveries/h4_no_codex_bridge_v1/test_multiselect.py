"""Regression checks for the frozen no-Codex multi-select review bridge."""

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import multiselect


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
BATCH = DELIVERIES / "h4_crossroot_multiselect_batch_v28_newroots_quota4to8"
REVIEW = DELIVERIES / "h4_crossroot_multiselect_auto_review_v2_deepseek"
PROTOCOL_PATH = DELIVERIES / "h4_crossroot_multiselect_solver_selection_v2" / "protocol.py"
spec = importlib.util.spec_from_file_location("multiselect_solver_protocol_v2", PROTOCOL_PATH)
assert spec and spec.loader
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def paths(root: str) -> tuple[Path, Path]:
    name = "cc-train-episode-" + root
    return BATCH / name / "batch-01", REVIEW / name


class MultiSelectBridgeTest(unittest.TestCase):
    def test_frozen_batch_is_closed_without_codex(self):
        for root in ("194259", "210046", "231899", "96544"):
            with self.subTest(root=root):
                result = multiselect.build(*paths(root))
                self.assertEqual(result["status"], "NO_APPROVED_CANDIDATES")
                self.assertEqual(result["questions"], [])
                self.assertTrue(result["provenance"]["no_codex_content_review"])

    def test_cannot_promote_rejected_candidate_by_editing_decision(self):
        batch, review = paths("210046")
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory)
            for filename in ("PREPARED.json", "INPUT.json", "REQUEST.json",
                             "PROMPT.txt", "RESPONSE.json", "REVIEW_RESULT_POLICY_V3.json"):
                (altered / filename).write_bytes((review / filename).read_bytes())
            result_path = altered / "REVIEW_RESULT_POLICY_V3.json"
            result = json.loads(result_path.read_text())
            qid = result["rejected_ids"].pop(0)
            result["approved_ids"].append(qid)
            result["candidate_decisions"][qid]["approved"] = True
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "review_decision_differs_from_model_or_policy"):
                multiselect.build(batch, altered)

    def test_changed_model_response_fails_closed(self):
        batch, review = paths("210046")
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory)
            for filename in ("PREPARED.json", "INPUT.json", "REQUEST.json",
                             "PROMPT.txt", "RESPONSE.json", "REVIEW_RESULT_POLICY_V3.json"):
                (altered / filename).write_bytes((review / filename).read_bytes())
            path = altered / "RESPONSE.json"
            response = json.loads(path.read_text())
            body = json.loads(response["raw_body"])
            body["model"] = "different-checkpoint"
            response["raw_body"] = json.dumps(body)
            path.write_text(json.dumps(response))
            with self.assertRaisesRegex(ValueError, "review_result_binding_invalid"):
                multiselect.build(batch, altered)

    def test_normalize_allows_two_through_six_gold_options(self):
        for count in range(2, 7):
            with self.subTest(gold=count):
                row = {"candidate_id": f"test-{count}", "topic": "topic",
                       "question": "Which statements are correct?", "options": [
                           {"proposition": str(index), "relation": "supports" if index < count
                            else "contradicts", "turn_ref": str(index), "quote": str(index),
                            "rationale": str(index)} for index in range(6)]}
                result = multiselect.normalize(row, "root", 0)
                self.assertEqual(len(result["gold"]), count)
                self.assertEqual({label for label, evidence in result["option_evidence"].items()
                                  if evidence["relation"] == "supports"}, set(result["gold"]))

    def test_normalized_question_matches_solver_v2_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.json"
            source.write_text(json.dumps({"root_id": "cc-train-episode-test", "turns": [
                {"turn_ref": str(index), "speaker": "speaker", "text": f"quote {index}"}
                for index in range(6)]}))
            row = {"candidate_id": "test-multi-01", "topic": "topic",
                   "question": "Which statements are correct?", "options": [
                       {"proposition": f"claim {index}", "relation": "supports" if index < 3
                        else "contradicts", "turn_ref": str(index), "quote": f"quote {index}",
                        "rationale": "evidence"} for index in range(6)]}
            question = {**multiselect.normalize(row, "cc-train-episode-test", 0),
                        "source_path": str(source), "source_sha256": protocol.sha(source)}
            pool = folder / "pool.json"
            pool.write_text(json.dumps({"status": "CONTENT_AUDITED_READY_FOR_SOLVER",
                "candidate_count": 1, "provenance": {"requested_model": "Qwen/Qwen3.6-35B-A3B",
                    "returned_model": "Qwen/Qwen3.6-35B-A3B",
                    "no_codex_content_review": True}, "questions": [question]}))
            _, questions, _ = protocol.validate_pool(pool)
            self.assertEqual(questions, [question])


if __name__ == "__main__":
    unittest.main()
