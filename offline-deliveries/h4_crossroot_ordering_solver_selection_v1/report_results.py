"""Independently score a completed cross-root ordering run without Solver calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
PRIOR_REPORT = DELIVERIES / "h4_ordering_large_model_solver_selection_v1/report_results.py"
POOL = DELIVERIES / "h4_crossroot_ordering_label_rebind_v1/READY_FOR_EVALUATION.json"
AUDIT = POOL.parent / "LABEL_MAPPING_AUDIT.json"
POLICY = DELIVERIES / "h4_ordering_ground_full_selection_v1/selection.py"
REPORT_VERSION = "h4_crossroot_ordering_solver_selection_v1"
GENERATOR_MODEL = "Qwen/Qwen3.6-35B-A3B"


def module(name, path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prior = module("crossroot_prior_report", PRIOR_REPORT)
policy = prior.policy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError("FROZEN_FILE_CHANGED:" + str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze(run):
    manifest = read(run / "manifest.json")
    if (manifest.get("execution") != "ground_first_adaptive_v1"
            or manifest.get("temperature") != 0.8 or manifest.get("top_p") != 0.9
            or manifest.get("max_tokens") != 32 or manifest.get("enable_thinking") is not False
            or manifest.get("rollouts_per_position") != 1
            or manifest.get("policy_version") != policy.VERSION
            or manifest.get("ground_rule_version") != policy.GROUND_VERSION):
        raise ValueError("EVALUATION_POLICY_OR_DECODING_MISMATCH")
    for key in ("code_sha256", "historical_snapshot"):
        if not manifest.get(key):
            raise ValueError("PINNED_HASHES_MISSING:" + key)
        for path, digest in manifest[key].items():
            if sha(path) != digest:
                raise ValueError("PINNED_FILE_CHANGED:" + path)
    pool = read(POOL, manifest["pool_sha256"])
    audit = read(AUDIT, manifest["label_mapping_audit_sha256"])
    if (pool["provenance"] != manifest["generator_provenance"]
            or pool["provenance"]["returned_model"] != GENERATOR_MODEL
            or pool["provenance"]["codex_role"] != "DETERMINISTIC_LABEL_REBIND_ONLY"):
        raise ValueError("GENERATOR_PROVENANCE_MISMATCH")
    questions = read(run / "questions.json", manifest["input_sha256"]["questions.json"])["candidates"]
    gold = read(run / "gold.json", manifest["input_sha256"]["gold.json"])
    ids = [q["candidate_id"] for q in questions]
    audit_rows = {row["derived_candidate_id"]: row for row in audit["candidates"]}
    if (len(ids) != len(set(ids)) or len(ids) != manifest["candidate_count"]
            or questions != pool["questions"] or set(ids) != set(audit_rows)
            or gold != {q["candidate_id"]: q["gold"] for q in questions}):
        raise ValueError("QUESTION_POOL_OR_GOLD_MISMATCH")
    sources = {str(run / "manifest.json"): sha(run / "manifest.json"),
               str(POOL): sha(POOL), str(AUDIT): sha(AUDIT),
               str(run / "questions.json"): sha(run / "questions.json"),
               str(run / "gold.json"): sha(run / "gold.json")}
    source_hashes = {}
    for question in questions:
        qid, root_id = question["candidate_id"], question["root_id"]
        key = str(Path("sources") / (root_id + ".json"))
        path = run / key
        digest = manifest["input_sha256"][key]
        source = read(path, digest)
        sources[str(path)] = digest
        if (source["root_id"] != root_id or len(source["turns"]) != manifest["source_turn_count_by_root"][root_id]
                or digest != audit_rows[qid]["source_sha256"]
                or digest != sha(manifest["source_paths_by_root"][root_id])
                or set(question["items"]) != {f"I{n}" for n in range(1, 7)}
                or set(question["gold"]) != set(question["items"])):
            raise ValueError("QUESTION_SOURCE_BINDING_MISMATCH:" + qid)
        policy.ordering_metrics(question["gold"], question["gold"])
        source_hashes[root_id] = digest
    identity = manifest["solver_identity"]
    if (identity["model_path"] != manifest["model"]
            or not re.fullmatch(r"[0-9a-f]{64}", identity["model_and_tokenizer_fingerprint_sha256"])):
        raise ValueError("SOLVER_IDENTITY_MISMATCH")
    planned = manifest["requests"]
    if len(planned) != len(questions) * 17 or len({row["request_id"] for row in planned}) != len(planned):
        raise ValueError("PLANNED_REQUEST_SET_INVALID")
    for row in planned:
        question = next((q for q in questions if q["candidate_id"] == row["candidate_id"]), None)
        if question is None or row.get("root_id") != question["root_id"] or row.get("source_sha256") != source_hashes[question["root_id"]]:
            raise ValueError("REQUEST_SOURCE_BINDING_MISMATCH:" + row["request_id"])
    complete = read(run / "COMPLETE.json")
    executed = complete["executed_request_ids"]
    if (len(executed) != len(set(executed)) or set(executed) - {row["request_id"] for row in planned}
            or complete["actual_call_count"] != len(executed)):
        raise ValueError("COMPLETE_EXECUTION_SET_INVALID")
    response_ids = {path.stem for path in (run / "responses").glob("*.json")}
    if response_ids - set(executed):
        raise ValueError("UNRECORDED_RESPONSE_FILE")
    for request_id in executed:
        path = run / "responses" / (request_id + ".json")
        if not path.exists():
            continue  # The per-question scorer reports this as unresolved.
        response = read(path)
        request = next(row for row in planned if row["request_id"] == request_id)
        if (response.get("root_id") != request["root_id"]
                or response.get("source_sha256") != request["source_sha256"]):
            raise ValueError("RESPONSE_SOURCE_BINDING_MISMATCH:" + request_id)
    statuses = read(run / "QUESTION_STATUS.json")["questions"]
    status_by_id = {row["candidate_id"]: row for row in statuses}
    if len(status_by_id) != len(statuses) or set(status_by_id) != set(ids):
        raise ValueError("QUESTION_STATUS_SET_MISMATCH")
    results = {}
    for question in questions:
        root_id = question["root_id"]
        per_question_manifest = {
            **manifest,
            "input_sha256": {**manifest["input_sha256"], "source.json": source_hashes[root_id]},
        }
        results[question["candidate_id"]] = prior.evaluate_question(
            question, per_question_manifest, status_by_id[question["candidate_id"]],
            run, sources, executed)
    counts = {
        "canonical_questions": len(questions),
        "selected_questions": sum(q["selected_by_policy"] is True for q in results.values()),
        "not_selected_questions": sum(q["selected_by_policy"] is False for q in results.values()),
        "unresolved_questions": sum(q["selected_by_policy"] is None for q in results.values()),
        "ground_filtered_questions": sum(q["ground"]["ground_pass"] is False for q in results.values()),
        "executed_ground_responses": sum("__no_context__" in rid for rid in executed),
        "executed_full_responses": sum("__full_context__" in rid for rid in executed),
        "new_solver_calls": len(executed), "actual_trained_questions": 0,
    }
    return {
        "report_version": REPORT_VERSION, "policy_version": policy.VERSION,
        "execution": "ground_first_adaptive_v1",
        "evaluation_data_status": "VALID" if not counts["unresolved_questions"] else "INCOMPLETE_OR_INVALID",
        "selection_basis": "ground_pass AND 0 < equal_permutation_full_net_pairwise_mean < 1",
        "ground_sampling": "different_displays_one_response_each",
        "full_variant_ids": list(policy.VARIANTS), "full_rollouts_per_variant": 1,
        "full_score_formula": "(correct_pairs - incorrect_pairs) / 15",
        "permutation_diagnostics_used_as_gate": False,
        "context_gain_used_as_gate_or_priority": False,
        "full_exact_success_required": False,
        "generator_provenance": pool["provenance"], "generator_model": GENERATOR_MODEL,
        "solver_model": "Qwen3-4B (base)", "solver_identity": identity,
        "run": str(run), "source_hashes_by_root": source_hashes,
        "source_hashes": sources, "historical_files_unchanged": True,
        "historical_protected_file_count": len(manifest["historical_snapshot"]),
        "counts": counts, "production_pool_modified": False,
        "sft_export_executed": False, "training_executed": False,
        "questions": results,
    }


def report(result):
    counts = result["counts"]
    lines = ["# Cross-Root Ordering Solver Selection", "",
             "Generator: `Qwen/Qwen3.6-35B-A3B` (SiliconFlow); Codex only rebound labels.",
             f"Solver: `Qwen3-4B (base)` at `{result['solver_identity']['model_path']}`.", "",
             "Ground: three different fixed displays, with two more after exactly 2/3 exact.",
             "After ground pass, all 12 full displays receive one response each.",
             "Select iff ground passes and 0 < mean net pairwise J < 1; negative scores are retained.",
             "Exact success, context gain and permutation range are diagnostics, not gates.", "",
             "| Root | Question | Ground exact | Full J mean | Full exact | J range | Selected | Reason |",
             "|---|---|---|---|---|---|---|---|"]
    for qid, row in result["questions"].items():
        ground = row["ground"]
        show = lambda value: value["fraction"] if value is not None else "not evaluated"
        exact = f"{row['full_exact_correct']}/12" if row["full_exact_correct"] is not None else "not evaluated"
        lines.append(f"| {row['root_id']} | {qid} | {ground.get('exact_correct', '?')}/{ground.get('evaluated_count', '?')} "
                     f"| {show(row['full_partial_mean'])} | {exact} | {show(row['permutation_score_range'])} "
                     f"| {row['selected_by_policy']} | {row['selection_reason']} |")
        if row["data_errors"]:
            lines.append(f"\nData errors for `{qid}`: " + "; ".join(row["data_errors"]))
    lines += ["", f"{counts['canonical_questions']} canonical questions; {counts['selected_questions']} selected; "
              f"{counts['ground_filtered_questions']} ground-filtered; {counts['unresolved_questions']} unresolved.",
              f"Actual Solver calls: {counts['new_solver_calls']} = {counts['executed_ground_responses']} ground + "
              f"{counts['executed_full_responses']} full.",
              "The `254852` question uses R01 evidence and does not count as R02 content yield.",
              "No SFT export, production pool change, or training occurred.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "run")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    result = analyze(args.run)
    selected = [q["canonical_question"] for q in result["questions"].values()
                if q["selected_by_policy"] is True]
    caches = {qid: q["ground_cache"] for qid, q in result["questions"].items()}
    prior.write_artifacts(args.output, {
        "RESULTS.json": result,
        "SELECTED_QUESTIONS.json": {"questions": selected, "count": len(selected),
                                    "source_result": str(args.output / "RESULTS.json"),
                                    "training_started": False},
        "GROUND_CACHE.json": caches,
        "REPORT.md": report(result),
    })
    print(json.dumps({"status": result["evaluation_data_status"],
                      "selected": len(selected), "calls": result["counts"]["new_solver_calls"]}), flush=True)


if __name__ == "__main__":
    main()
