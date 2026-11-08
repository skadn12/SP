"""Small CPU-only regressions; all writes stay in temporary directories."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("question_batch_under_test", Path(__file__).with_name("question_batch.py"))
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def result(prompt="prompt", ids=(1, 2), completions=1):
    return SimpleNamespace(prompt=prompt, prompt_token_ids=list(ids), outputs=[
        SimpleNamespace(text="unchanged model text", finish_reason="stop", stop_reason=None, token_ids=[3])
        for _ in range(completions)])


@contextlib.contextmanager
def fixture(raw_errors=False):
    with tempfile.TemporaryDirectory(prefix="question-batch-cpu-") as directory:
        root = Path(directory)
        out = root / "output"
        out.mkdir()
        with patch.object(batch, "OUT", out):
            freeze = root / "execution_freeze.json"
            batch.save(freeze, {})
            source = root / "source.json"
            batch.save(source, {})
            ready = root / "ready.json"
            batch.save(ready, {})
            rows, private_rows, reviews, bindings = [], [], [], [batch.bind(freeze)]
            for number in range(2):
                cid, rid = "candidate" + str(number), "request" + str(number)
                candidate = root / (cid + ".json")
                batch.save(candidate, {"gold": ["A"]})
                prompt, schema = out / (rid + ".txt"), out / (rid + "_schema.json")
                prompt.write_text("prompt")
                batch.save(schema, {})
                row = {"candidate_id": cid, "request_id": rid, "root_id": "root" + str(number),
                    "question_type": "multi_select", "task_operation": "MC_EVIDENCE_SELECTION",
                    "source_path": str(source), "source_binding": batch.bind(source),
                    "question_ready_binding": batch.bind(ready), "sampling_seed": number,
                    "prompt_path": str(prompt), "schema_path": str(schema), "prompt_tokens": 2,
                    "display_to_canonical": {"A": "A"}}
                rows.append(row)
                private_rows.append({key: row[key] for key in (
                    "request_id", "candidate_id", "source_binding", "question_ready_binding")})
                private_rows[-1]["candidate_binding"] = batch.bind(candidate)
                reviews.append({"candidate_id": cid, "candidate_binding": batch.bind(candidate)})
                bindings.extend((batch.bind(prompt), batch.bind(schema)))
                raw = batch.returned_record(row, [result()], "prompt", [1, 2], {"model_directory": "unused"})
                if raw_errors and number == 0:
                    raw["output_validation_errors"] = ["PROMPT_TOKEN_IDS_DRIFT"]
                batch.save(out / "raw" / (rid + ".json"), raw)
            batch.save(out / "PUBLIC_REQUESTS.json", {"requests": rows})
            batch.save(out / "PROMPT_TOKEN_IDS.json", {row["request_id"]: [1, 2] for row in rows})
            review = root / "review.json"
            batch.save(review, {"reviews": reviews, "question_execution_freeze_binding": batch.bind(freeze)})
            batch.save(out / "PRIVATE_SCORING_BINDINGS.json", {"requests": private_rows,
                "content_review_binding": batch.bind(review), "execution_freeze_binding": batch.bind(freeze),
                "public_requests_binding": batch.bind(out / "PUBLIC_REQUESTS.json"),
                "prompt_token_ids_binding": batch.bind(out / "PROMPT_TOKEN_IDS.json")})
            bindings.extend(batch.bind(out / name) for name in ("PUBLIC_REQUESTS.json", "PROMPT_TOKEN_IDS.json"))
            private_binding = batch.bind(out / "PRIVATE_SCORING_BINDINGS.json")
            batch.save(out / "PREPARATION.json", {"requests": 2, "bindings": bindings,
                "private_scoring_binding": private_binding, "content_review_binding": batch.bind(review)})
            batch.save(out / "STARTED.json", {"requests": 2, "request_ids": [row["request_id"] for row in rows],
                "preparation": batch.bind(out / "PREPARATION.json"), "private_scoring_binding": private_binding})
            batch.save(out / "RAW_RESPONSE_FREEZE.json", {"requests": 2, "returned_requests": 2,
                "request_ids": [row["request_id"] for row in rows], "started_binding": batch.bind(out / "STARTED.json"),
                "bindings": [batch.bind(out / "raw" / (row["request_id"] + ".json")) for row in rows]})
            yield root, out


class QuestionBatchTests(unittest.TestCase):
    def test_return_validation_preserves_all_text_and_detects_same_length_drift(self):
        row = {"request_id": "request", "sampling_seed": 1}
        for results, expected in [([result()], []), ([result(ids=(2, 1))], ["PROMPT_TOKEN_IDS_DRIFT"]),
                ([result(prompt="other")], ["RETURNED_PROMPT_DRIFT"]), ([], ["RETURNED_RESULT_COUNT_MISMATCH"]),
                ([result(), result()], ["RETURNED_RESULT_COUNT_MISMATCH"]),
                ([result(completions=2)], ["COMPLETION_COUNT_MISMATCH"])]:
            with self.subTest(expected=expected):
                raw = batch.returned_record(row, results, "prompt", [1, 2], {"model_directory": "unused"})
                self.assertEqual(raw["output_validation_errors"], expected)
                self.assertEqual(len(raw["returned_results"]), len(results))
                self.assertTrue(all(comp["raw_output"] == "unchanged model text"
                    for returned in raw["returned_results"] for comp in returned["outputs"]))

    def test_worker_preparation_does_not_open_private_files(self):
        with fixture() as (root, out):
            for path in (root / "candidate0.json", root / "candidate1.json", root / "review.json", out / "PRIVATE_SCORING_BINDINGS.json"):
                path.unlink()
            self.assertEqual(len(batch.public_preparation()[1]), 2)

    def test_changed_candidate_is_rejected_before_scoring(self):
        with fixture() as (root, out):
            (root / "candidate0.json").write_text('{"gold":["B"]}')
            with self.assertRaisesRegex(AssertionError, "BINDING_CHANGED"):
                batch.analyze()
            self.assertFalse((out / "SCORES.json").exists())

    def test_validation_failure_is_invalid_and_keeps_two_request_denominator(self):
        with fixture(raw_errors=True) as (_, out), patch.object(batch.diagnostic.repair, "validate", return_value=({}, ["A"])), patch.object(batch.diagnostic, "mechanical_consistency", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            batch.analyze()
            scores, packet = batch.read(out / "SCORES.json"), batch.read(out / "REASON_REVIEW_PACKET.json")
            self.assertEqual((scores["requests"], scores["correct"]), (2, 1))
            self.assertEqual(packet["requests"][0]["status"], "INVALID")
            self.assertEqual(packet["requests"][0]["error"], "OUTPUT_VALIDATION_ERRORS")

    def test_unsealed_raw_file_and_changed_start_are_rejected(self):
        with fixture() as (_, out):
            batch.save(out / "raw" / "unexpected.json", {})
            with self.assertRaisesRegex(AssertionError, "RAW_DIRECTORY_SET_DRIFT"):
                batch.analyze()
        with fixture() as (_, out):
            (out / "STARTED.json").write_text('{}')
            with self.assertRaisesRegex(AssertionError, "STARTED_BINDING_DRIFT"):
                batch.analyze()


if __name__ == "__main__":
    unittest.main()
