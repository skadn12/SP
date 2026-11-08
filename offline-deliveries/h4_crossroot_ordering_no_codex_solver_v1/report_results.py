"""Re-score a completed no-Codex ordering Solver run from frozen inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from build import HERE, BATCH, make_pool, read, require, sha


POOL = HERE / "READY_FOR_EVALUATION.json"
AUDIT = HERE / "AUTOMATIC_REVIEW_AUDIT.json"
RUN = HERE / "run"
BASE_REPORT = HERE.parent / "h4_crossroot_ordering_solver_selection_v1" / "report_results.py"
POLICY_PATH = HERE.parent / "h4_ordering_ground_full_selection_v1" / "selection.py"


def load_base():
    spec = importlib.util.spec_from_file_location("auto_review_ordering_base_report", BASE_REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()


def verified_run(run: Path) -> tuple[dict, list[dict], list[str], dict]:
    pool, audit = read(POOL), read(AUDIT)
    require((pool, audit) == make_pool(), "automatic_review_pool_or_audit_changed")
    manifest = read(run / "manifest.json")
    require(manifest.get("execution") == "ground_first_adaptive_v1"
            and manifest.get("pool_sha256") == sha(POOL)
            and manifest.get("label_mapping_audit_sha256") == sha(AUDIT)
            and manifest.get("candidate_count") == len(pool["questions"])
            and manifest.get("generator_provenance") == pool["provenance"]
            and manifest.get("policy_version") == base.policy.VERSION
            and manifest.get("ground_rule_version") == base.policy.GROUND_VERSION
            and manifest.get("temperature") == 0.8
            and manifest.get("top_p") == 0.9
            and manifest.get("max_tokens") == 32
            and manifest.get("enable_thinking") is False
            and manifest.get("rollouts_per_position") == 1,
            "run_policy_or_pool_mismatch")
    for mapping in (manifest["code_sha256"], manifest["historical_snapshot"]):
        for path, digest in mapping.items():
            require(sha(Path(path)) == digest, "frozen_code_or_input_changed:" + path)
    for name, digest in manifest["input_sha256"].items():
        require(sha(run / name) == digest, "run_input_changed:" + name)
    require(read(run / "questions.json")["candidates"] == pool["questions"]
            and read(run / "gold.json")
            == {q["candidate_id"]: q["gold"] for q in pool["questions"]},
            "run_questions_or_gold_changed")
    sources = {}
    for question in pool["questions"]:
        root = question["root_id"]
        source_path = BATCH / root / "source.json"
        copy_path = run / "sources" / (root + ".json")
        require(question["source_sha256"] == sha(source_path) == sha(copy_path),
                "run_source_changed:" + root)
        sources[str(copy_path)] = sha(copy_path)
    complete = read(run / "COMPLETE.json")
    executed = complete["executed_request_ids"]
    planned = {request["request_id"]: request for request in manifest["requests"]}
    require(len(executed) == len(set(executed)) == complete["actual_call_count"]
            and set(executed) <= set(planned)
            and {path.stem for path in (run / "responses").glob("*.json")} == set(executed),
            "run_response_set_invalid")
    for request_id in executed:
        request, response = planned[request_id], read(run / "responses" / (request_id + ".json"))
        require(response.get("request_id") == request_id
                and response.get("candidate_id") == request["candidate_id"]
                and response.get("root_id") == request["root_id"]
                and response.get("source_sha256") == request["source_sha256"],
                "run_response_binding_invalid:" + request_id)
    statuses = read(run / "QUESTION_STATUS.json")["questions"]
    require({row["candidate_id"] for row in statuses}
            == {question["candidate_id"] for question in pool["questions"]}
            and len(statuses) == len(pool["questions"]), "run_status_set_invalid")
    return manifest, pool["questions"], executed, {row["candidate_id"]: row for row in statuses}


def analyze(run: Path) -> dict:
    manifest, questions, executed, statuses = verified_run(run)
    source_hashes = {question["root_id"]: question["source_sha256"] for question in questions}
    sources = {str(run / "manifest.json"): sha(run / "manifest.json"),
               str(POOL): sha(POOL), str(AUDIT): sha(AUDIT),
               str(run / "questions.json"): sha(run / "questions.json"),
               str(run / "gold.json"): sha(run / "gold.json")}
    results = {}
    for question in questions:
        root = question["root_id"]
        source_copy = run / "sources" / (root + ".json")
        sources[str(source_copy)] = sha(source_copy)
        per_question_manifest = {
            **manifest,
            "input_sha256": {**manifest["input_sha256"], "source.json": source_hashes[root]},
        }
        results[question["candidate_id"]] = base.prior.evaluate_question(
            question, per_question_manifest, statuses[question["candidate_id"]],
            run, sources, executed)
    unresolved = sum(row["selected_by_policy"] is None for row in results.values())
    counts = {"canonical_questions": len(questions),
              "selected_questions": sum(row["selected_by_policy"] is True for row in results.values()),
              "not_selected_questions": sum(row["selected_by_policy"] is False for row in results.values()),
              "unresolved_questions": unresolved,
              "ground_filtered_questions": sum(row["ground"]["ground_pass"] is False
                                               for row in results.values()),
              "executed_ground_responses": sum("__no_context__" in rid for rid in executed),
              "executed_full_responses": sum("__full_context__" in rid for rid in executed),
              "new_solver_calls": len(executed), "actual_trained_questions": 0}
    return {"report_version": HERE.name, "policy_version": base.policy.VERSION,
            "execution": "ground_first_adaptive_v1",
            "evaluation_data_status": "VALID" if not unresolved else "INCOMPLETE_OR_INVALID",
            "selection_basis": "ground_pass AND 0 < equal_permutation_full_net_pairwise_mean < 1",
            "generator_provenance": manifest["generator_provenance"],
            "generator_model": "Qwen/Qwen3.6-35B-A3B",
            "solver_model": "Qwen3-4B (base)",
            "solver_identity": manifest["solver_identity"],
            "run": str(run), "source_hashes_by_root": source_hashes,
            "source_hashes": sources, "counts": counts,
            "sft_export_executed": False, "training_executed": False,
            "questions": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    result = analyze(args.run)
    selected = [row["canonical_question"] for row in result["questions"].values()
                if row["selected_by_policy"] is True]
    caches = {qid: row["ground_cache"] for qid, row in result["questions"].items()}
    base.prior.write_artifacts(args.output, {
        "RESULTS.json": result,
        "SELECTED_QUESTIONS.json": {"questions": selected, "count": len(selected),
                                    "source_result": str(args.output / "RESULTS.json"),
                                    "training_started": False},
        "GROUND_CACHE.json": caches,
        "REPORT.md": "\n".join(["# No-Codex Ordering Solver Selection", "",
                                "Generator: Qwen/Qwen3.6-35B-A3B. Content review: independent DeepSeek and GLM model gates.",
                                "Solver: Qwen3-4B (base). No training or export occurred.", "",
                                f"Candidates: {result['counts']['canonical_questions']}; selected: {len(selected)}; calls: {result['counts']['new_solver_calls']}.",
                                ""]),
    })
    print(json.dumps({"status": result["evaluation_data_status"],
                      "selected": len(selected), "calls": result["counts"]["new_solver_calls"]}))


if __name__ == "__main__":
    main()
