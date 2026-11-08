"""Offline contract and independent scoring checks; no Solver imports or calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import VARIANTS, display, request_plan, sha, validate_pool


def load_report():
    spec = importlib.util.spec_from_file_location("test_crossroot_multiselect_report",
                                                  HERE / "report_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_report()


def question(source_path, source_hash):
    return {"candidate_id": "test-multiselect-01", "root_id": "cc-train-episode-test",
            "question_type": "multi_select", "question": "Select supported claims.",
            "options": dict(zip("ABCDEF", ("claim A", "claim B", "claim C",
                                            "claim D", "claim E", "claim F"))),
            "gold": ["A", "C", "E"],
            "option_evidence": {label: {"turn_ref": "R01:S01:T001", "quote": "source quote",
                                        "relation": "supports" if label in "ACE" else "contradicts",
                                        "reason": "Test evidence."}
                                for label in "ABCDEF"},
            "source_path": str(source_path), "source_sha256": source_hash}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def source_and_pool(tmp_path):
    source_path = tmp_path / "source.json"
    write_json(source_path, {"root_id": "cc-train-episode-test", "turns": [
        {"turn_ref": "R01:S01:T001", "text": "source quote"}]})
    q = question(source_path, sha(source_path))
    pool_path = tmp_path / "READY_FOR_EVALUATION.json"
    pool = {"status": "CONTENT_AUDITED_READY_FOR_SOLVER", "candidate_count": 1,
            "provenance": {"requested_model": "Qwen/Qwen3.6-35B-A3B",
                           "returned_model": "Qwen/Qwen3.6-35B-A3B"},
            "questions": [q]}
    write_json(pool_path, pool)
    return source_path, pool_path, q


def check_pool_source_binding_and_cyclic_mapping(tmp_path):
    source_path, pool_path, q = source_and_pool(tmp_path)
    _, questions, sources = validate_pool(pool_path)
    assert questions == [q]
    assert sources[q["root_id"]]["sha256"] == sha(source_path)
    assert display(q, 0)["display_to_canonical"] == dict(zip("ABCDEF", "ABCDEF"))
    assert display(q, 1)["display_to_canonical"] == dict(zip("ABCDEF", "BCDEFA"))
    for shift in range(6):
        variant = display(q, shift)
        assert {variant["display_to_canonical"][label] for label in variant["displayed_gold"]} == set(q["gold"])
    requests, prompts = request_plan(questions, sources, lambda *args: "prompt")
    assert len(requests) == len(prompts) == 23
    assert sum(row["mode"] == "no_context" for row in requests) == 5
    assert sum(row["mode"] == "full_context" for row in requests) == 18
    assert {row["variant"]["variant_id"] for row in requests if row["mode"] == "full_context"} == set(VARIANTS)


def check_pool_rejects_wrong_source_and_incomplete_evidence(tmp_path):
    source_path, pool_path, q = source_and_pool(tmp_path)
    q["source_sha256"] = "0" * 64
    write_json(pool_path, {"status": "CONTENT_AUDITED_READY_FOR_SOLVER", "candidate_count": 1,
                           "provenance": {"requested_model": "Qwen/Qwen3.6-35B-A3B",
                                          "returned_model": "Qwen/Qwen3.6-35B-A3B"},
                           "questions": [q]})
    try:
        validate_pool(pool_path)
    except ValueError as error:
        assert "SOURCE_PATH_OR_SHA_INVALID" in str(error)
    else:
        raise AssertionError("wrong source hash accepted")
    q["source_sha256"] = sha(source_path)
    del q["option_evidence"]["F"]
    write_json(pool_path, {"status": "CONTENT_AUDITED_READY_FOR_SOLVER", "candidate_count": 1,
                           "provenance": {"requested_model": "Qwen/Qwen3.6-35B-A3B",
                                          "returned_model": "Qwen/Qwen3.6-35B-A3B"},
                           "questions": [q]})
    try:
        validate_pool(pool_path)
    except ValueError as error:
        assert "COMPLETE_OPTION_EVIDENCE_REQUIRED" in str(error)
    else:
        raise AssertionError("incomplete evidence accepted")


def check_report_recomputes_ground_and_full_from_raw_responses(tmp_path, ground_output, passed):
    _, pool_path, q = source_and_pool(tmp_path)
    _, questions, sources = validate_pool(pool_path)
    requests, _ = request_plan(questions, sources, lambda *args: "prompt")
    by_key = {(row["candidate_id"], row["mode"], row["variant"]["variant_id"], row["slot"]): row
              for row in requests}
    manifest = {"model": "/root/autodl-tmp/Qwen3-4B",
                "solver_identity": {"model_path": "/root/autodl-tmp/Qwen3-4B"},
                "generation_config": {"temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
                                      "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                                      "structured_output_version": "finite_choice_set_v1",
                                      "enable_thinking": False},
                "code_sha256": {str(report.POLICY_PATH): sha(report.POLICY_PATH)}}
    run = tmp_path / "run"
    executed = []

    def add(row, output):
        response = {"request_id": row["request_id"], "candidate_id": row["candidate_id"],
                    "root_id": row["root_id"], "source_sha256": row["source_sha256"],
                    "mode": row["mode"], "variant_id": row["variant"]["variant_id"],
                    "slot": row["slot"], "seed": row["seed"],
                    "prompt_sha256": row["prompt_sha256"], "raw_output": output,
                    "model_identifier": manifest["model"],
                    "metadata": {**manifest["generation_config"], "finish_reason": "stop"},
                    "displayed_gold": row["variant"]["displayed_gold"],
                    "partial_score_fraction": "-1" if not passed else "2/3",
                    "evaluation_valid": True, "data_errors": []}
        write_json(run / "responses" / (row["request_id"] + ".json"), response)
        executed.append(row["request_id"])

    qid = q["candidate_id"]
    for slot in range(3):
        row = by_key[qid, "no_context", "perm_00", slot]
        if not passed:
            row_output = ground_output
            j = "1"
        else:
            row_output = ground_output
            j = "-1"
        add(row, row_output)
        response_path = run / "responses" / (row["request_id"] + ".json")
        response = json.loads(response_path.read_text())
        response["partial_score_fraction"] = j
        write_json(response_path, response)
    if passed:
        for variant in VARIANTS:
            for slot in range(3):
                row = by_key[qid, "full_context", variant, slot]
                mapping = row["variant"]["display_to_canonical"]
                labels = [label for label, canonical in mapping.items() if canonical in {"A", "C"}]
                add(row, "<box>" + ",".join(labels) + "</box>")
    ground = report.policy.ground_decision([not passed] * 3)
    status = {"candidate_id": qid, "root_id": q["root_id"], "ground": ground,
              "status": "GROUND_PASS_ALL_FULL_EVALUATED" if passed else "GROUND_FILTERED_FULL_SKIPPED",
              "executed_request_ids": executed,
              "full_skipped_reason": None if passed else "GROUND_FILTERED"}
    result = report.evaluate_question(q, manifest, status, by_key, executed, run)
    assert result["data_errors"] == []
    assert result["selected_by_policy"] is passed
    if passed:
        assert result["full_partial_mean"]["fraction"] == "2/3"
        assert result["full_exact_correct"] == 0
        assert len(result["full_responses"]) == 18
    else:
        assert result["full_responses"] == []


class SelectionRunnerTests(unittest.TestCase):
    def test_pool_source_binding_and_cyclic_mapping(self):
        with TemporaryDirectory() as folder:
            check_pool_source_binding_and_cyclic_mapping(Path(folder))

    def test_pool_rejects_wrong_source_and_incomplete_evidence(self):
        with TemporaryDirectory() as folder:
            check_pool_rejects_wrong_source_and_incomplete_evidence(Path(folder))

    def test_report_selects_partial_full_performance(self):
        with TemporaryDirectory() as folder:
            check_report_recomputes_ground_and_full_from_raw_responses(
                Path(folder), "<box>B,D,F</box>", True)

    def test_report_filters_easy_ground(self):
        with TemporaryDirectory() as folder:
            check_report_recomputes_ground_and_full_from_raw_responses(
                Path(folder), "<box>A,C,E</box>", False)


if __name__ == "__main__":
    unittest.main()
