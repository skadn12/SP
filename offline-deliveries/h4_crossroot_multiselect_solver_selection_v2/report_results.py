"""Independently re-score a completed cross-root multiselect Solver run."""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
POLICY_PATH = DELIVERIES / "h4_multiselect_ground_full_selection_v2/selection.py"
SCORER_PATH = HERE / "scoring.py"
REPORT_VERSION = "h4_crossroot_multiselect_solver_selection_v2"
EXECUTION = "multiselect_ground_k3_adaptive_k5_full_6x3_g2to6_v2"

from protocol import FULL_SLOTS, GROUND_SLOTS, VARIANTS, display, object_sha, read, sha, validate_pool


def module(name, path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


policy = module("crossroot_multiselect_report_policy", POLICY_PATH)
scorer = module("crossroot_multiselect_report_scorer", SCORER_PATH)


def checked_read(path, expected):
    if sha(path) != expected:
        raise ValueError("FROZEN_FILE_CHANGED:" + str(path))
    return read(path)


def expected_keys():
    return ([('no_context', 'perm_00', slot) for slot in GROUND_SLOTS]
            + [('full_context', variant, slot) for variant in VARIANTS for slot in FULL_SLOTS])


def validate_request(question, row, run):
    qid = question["candidate_id"]
    mode, variant_id, slot = row["mode"], row["variant"]["variant_id"], row["slot"]
    key = mode, variant_id, slot
    if key not in expected_keys():
        raise ValueError("UNEXPECTED_REQUEST_SLOT:" + qid)
    request_id = f"{qid}__{variant_id}__{mode}__s{slot:02d}"
    if (row.get("request_id") != request_id or row.get("candidate_id") != qid
            or row.get("root_id") != question["root_id"]
            or row.get("source_sha256") != question["source_sha256"]
            or row.get("question_type") != "multi_select"
            or type(row.get("seed")) is not int
            or type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", row["prompt_sha256"])):
        raise ValueError("REQUEST_ID_OR_SOURCE_BINDING_MISMATCH:" + qid)
    shift = int(variant_id[-2:])
    if row["variant"] != display(question, shift):
        raise ValueError("DISPLAY_MAPPING_OR_GOLD_INVALID:" + request_id)
    if sha(run / "prompts" / (request_id + ".txt")) != row["prompt_sha256"]:
        raise ValueError("PROMPT_HASH_MISMATCH:" + request_id)
    return key


def evaluate_question(question, manifest, status, requests, executed, run):
    qid = question["candidate_id"]
    actual = [request_id for request_id in executed if request_id.startswith(qid + "__")]
    errors, rows = [], {}
    if (status.get("candidate_id") != qid or status.get("root_id") != question["root_id"]
            or status.get("executed_request_ids") != actual):
        errors.append("QUESTION_STATUS_EXECUTED_IDS_MISMATCH")

    def score(mode, variant, slot):
        request = requests[qid, mode, variant, slot]
        request_id = request["request_id"]
        result = {"request_id": request_id, "mode": mode, "variant_id": variant,
                  "slot": slot, "evaluation_valid": False, "exact_match": None,
                  "partial_score": None, "data_errors": []}
        try:
            if request_id not in actual:
                raise ValueError("REQUIRED_EVALUATION_NOT_EXECUTED")
            response_path = run / "responses" / (request_id + ".json")
            response = read(response_path)
            for field in ("request_id", "candidate_id", "root_id", "source_sha256", "mode",
                          "variant_id", "slot", "seed", "prompt_sha256"):
                expected = (request[field] if field in request else
                            request["variant"]["variant_id"] if field == "variant_id" else None)
                if response.get(field) != expected:
                    raise ValueError("REQUEST_RESPONSE_MISMATCH:" + field)
            if response.get("displayed_gold") != request["variant"]["displayed_gold"]:
                raise ValueError("DISPLAYED_GOLD_RESPONSE_MISMATCH")
            if response.get("model_identifier") != manifest["model"]:
                raise ValueError("SOLVER_MODEL_MISMATCH")
            metadata = response.get("metadata", {})
            if metadata.get("finish_reason") not in ("stop", "eos"):
                raise ValueError("INCOMPLETE_GENERATION")
            for field in ("temperature", "top_p", "max_tokens", "structured_output_api",
                          "structured_output_version", "enable_thinking"):
                if metadata.get(field) != manifest["generation_config"].get(field):
                    raise ValueError("DECODING_MISMATCH:" + field)
            parsed = scorer.score_response(question, request, response)
            if parsed["scoring_status"] != "SCORED":
                raise ValueError("SCORING:" + parsed["scoring_status"] + ":" + str(parsed["parse_reason"]))
            j = policy.partial_score(parsed["tp"], parsed["fp"], parsed["g"], parsed["n"])
            if response.get("partial_score_fraction") != str(j):
                raise ValueError("STORED_PARTIAL_SCORE_MISMATCH")
            if response.get("evaluation_valid") is not True or response.get("data_errors"):
                raise ValueError("RUNNER_VALIDITY_MISMATCH")
            prediction = set(parsed["canonical_prediction"])
            gold = set(question["gold"])
            result.update({"evaluation_valid": True, "exact_match": parsed["exact_match"],
                           "partial_score": policy.fraction(j),
                           "canonical_prediction": parsed["canonical_prediction"],
                           "display_prediction": parsed["display_prediction"],
                           "tp": parsed["tp"], "fp": parsed["fp"], "fn": parsed["fn"],
                           "false_positive_contents": {key: question["options"][key]
                                                       for key in sorted(prediction - gold)},
                           "false_negative_contents": {key: question["options"][key]
                                                       for key in sorted(gold - prediction)},
                           "raw_output": response["raw_output"],
                           "source_path": str(response_path), "source_sha256": sha(response_path),
                           "seed": request["seed"], "prompt_sha256": request["prompt_sha256"],
                           "variant": request["variant"]})
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            result["data_errors"].append(str(exc))
        rows[mode, variant, slot] = result
        return result

    ground_rows = [score("no_context", "perm_00", slot) for slot in range(3)]
    initial = [row["exact_match"] for row in ground_rows]
    ground = policy.ground_decision(initial)
    if ground["status"] == "NEEDS_EXTENSION":
        extension = [score("no_context", "perm_00", slot) for slot in (3, 4)]
        ground_rows += extension
        ground = policy.ground_decision(initial, [row["exact_match"] for row in extension])
    ground_errors = [f"{row['request_id']}:{error}" for row in ground_rows
                     for error in row["data_errors"]]
    errors += ground_errors
    if status.get("ground") != ground:
        errors.append("QUESTION_STATUS_GROUND_DECISION_MISMATCH")
    full_rows = []
    if ground["ground_pass"] is True:
        full_rows = [score("full_context", variant, slot) for variant in VARIANTS
                     for slot in FULL_SLOTS]
        if status.get("status") != "GROUND_PASS_ALL_FULL_EVALUATED" or status.get("full_skipped_reason") is not None:
            errors.append("QUESTION_STATUS_FULL_EVALUATION_MISMATCH")
    else:
        expected_status = ("GROUND_FILTERED_FULL_SKIPPED" if ground["ground_pass"] is False
                           else "EVALUATION_INCOMPLETE_FULL_SKIPPED")
        if status.get("status") != expected_status or not status.get("full_skipped_reason"):
            errors.append("QUESTION_STATUS_FULL_SKIP_MISMATCH")
    required = [row["request_id"] for row in ground_rows + full_rows]
    if actual != required:
        errors.append("ADAPTIVE_EXECUTION_ORDER_OR_REQUIRED_CALL_SET_MISMATCH")
    errors += [f"{row['request_id']}:{error}" for row in full_rows for error in row["data_errors"]]
    if full_rows:
        scores = {variant: [Fraction(rows["full_context", variant, slot]["partial_score"]["fraction"])
                            if rows["full_context", variant, slot]["evaluation_valid"] else None
                            for slot in FULL_SLOTS] for variant in VARIANTS}
        decision = policy.select_question(ground, scores, data_errors=errors)
    else:
        decision = {"policy_version": policy.VERSION, "ground_pass": ground["ground_pass"],
                    "selected_by_policy": False if ground["ground_pass"] is False and not errors else None,
                    "selection_reason": ("GROUND_FILTERED_FULL_SKIPPED" if ground["ground_pass"] is False
                                         and not errors else "EVALUATION_INCOMPLETE_OR_INVALID"),
                    "full_partial_mean": None, "permutation_means": {},
                    "permutation_score_min": None, "permutation_score_max": None,
                    "permutation_score_range": None, "data_errors": sorted(set(errors))}
    complete = len(full_rows) == 18 and all(row["evaluation_valid"] for row in full_rows)
    exact = sum(row["exact_match"] is True for row in full_rows) if complete else None
    bindings = {"question_sha256": object_sha(question), "gold_sha256": object_sha(question["gold"]),
                "source_sha256": question["source_sha256"], "evaluation_display": display(question, 0),
                "prompt_sha256": requests[qid, "no_context", "perm_00", 0]["prompt_sha256"],
                "ground_request_ids": [row["request_id"] for row in ground_rows],
                "solver_checkpoint": manifest["solver_identity"],
                "decoding": manifest["generation_config"],
                "ground_rule_version": policy.GROUND_VERSION,
                "selection_policy_version": policy.VERSION,
                "report_version": REPORT_VERSION,
                "policy_code_sha256": manifest["code_sha256"][str(POLICY_PATH)]}
    cache = {"cache_key": object_sha(bindings), "bindings": bindings, "decision": ground,
             "reusable": not ground_errors and ground["ground_pass"] is not None,
             "data_errors": ground_errors, "source_responses": ground_rows}
    return {**decision, "canonical_question": question, "root_id": question["root_id"],
            "ground": ground, "ground_responses": ground_rows, "ground_cache": cache,
            "execution_status": status.get("status"), "executed_request_ids": actual,
            "full_expected_count": 18 if ground["ground_pass"] is True else 0,
            "full_evaluated_count": sum(row["evaluation_valid"] for row in full_rows),
            "full_exact_correct": exact,
            "full_exact_rate": policy.fraction(Fraction(exact, 18)) if exact is not None else None,
            "full_responses": full_rows,
            "diagnostics_not_used_for_selection": {"context_gain_used_as_gate_or_priority": False,
                                                   "permutation_range_used_as_gate": False,
                                                   "full_exact_success_required": False}}


def analyze(run):
    manifest = read(run / "manifest.json")
    if (manifest.get("execution") != EXECUTION or manifest.get("policy_version") != policy.VERSION
            or manifest.get("ground_rule_version") != policy.GROUND_VERSION
            or manifest.get("temperature") != 0.8 or manifest.get("top_p") != 0.9
            or manifest.get("max_tokens") != 32 or manifest.get("enable_thinking") is not False
            or manifest.get("ground_display") != "perm_00"
            or manifest.get("full_variant_ids") != list(VARIANTS)
            or manifest.get("full_rollouts_per_variant") != 3):
        raise ValueError("EVALUATION_POLICY_OR_DECODING_MISMATCH")
    for field in ("code_sha256", "historical_snapshot"):
        if not manifest.get(field):
            raise ValueError("PINNED_HASHES_MISSING:" + field)
        for path, digest in manifest[field].items():
            if sha(path) != digest:
                raise ValueError("PINNED_FILE_CHANGED:" + path)
    pool_path = Path(manifest["pool_path"])
    checked_read(pool_path, manifest["pool_sha256"])
    pool, pool_questions, sources = validate_pool(pool_path)
    questions = checked_read(run / "questions.json", manifest["input_sha256"]["questions.json"])["candidates"]
    gold = checked_read(run / "gold.json", manifest["input_sha256"]["gold.json"])
    if (questions != pool_questions or pool["provenance"] != manifest["generator_provenance"]
            or gold != {q["candidate_id"]: q["gold"] for q in questions}
            or len(questions) != manifest["candidate_count"]):
        raise ValueError("FROZEN_POOL_QUESTIONS_OR_GOLD_MISMATCH")
    for root, source in sources.items():
        key = str(Path("sources") / (root + ".json"))
        if (source["path"] != manifest["source_paths_by_root"][root]
                or source["turn_count"] != manifest["source_turn_count_by_root"][root]
                or source["sha256"] != manifest["input_sha256"][key]
                or sha(run / key) != source["sha256"]):
            raise ValueError("FROZEN_SOURCE_BINDING_MISMATCH:" + root)
    identity = manifest["solver_identity"]
    if (identity["model_path"] != manifest["model"]
            or not re.fullmatch(r"[0-9a-f]{64}", identity["model_and_tokenizer_fingerprint_sha256"])):
        raise ValueError("SOLVER_IDENTITY_MISMATCH")
    planned = manifest["requests"]
    if len(planned) != len(questions) * 23:
        raise ValueError("PREPLANNED_REQUEST_COUNT_MISMATCH")
    requests = {}
    questions_by_id = {q["candidate_id"]: q for q in questions}
    for row in planned:
        qid = row["candidate_id"]
        if qid not in questions_by_id:
            raise ValueError("PLANNED_UNKNOWN_QUESTION:" + qid)
        key = validate_request(questions_by_id[qid], row, run)
        request_key = (qid, *key)
        if request_key in requests:
            raise ValueError("DUPLICATE_PREPLANNED_REQUEST")
        requests[request_key] = row
    expected = {(q["candidate_id"], *key) for q in questions for key in expected_keys()}
    if set(requests) != expected:
        raise ValueError("PREPLANNED_REQUEST_SET_MISMATCH")
    complete = read(run / "COMPLETE.json")
    executed = complete["executed_request_ids"]
    planned_ids = {row["request_id"] for row in planned}
    if (len(executed) != len(set(executed)) or set(executed) - planned_ids
            or complete.get("actual_call_count") != len(executed)):
        raise ValueError("COMPLETE_EXECUTION_SET_INVALID")
    existing = {path.stem for path in (run / "responses").glob("*.json")}
    if existing - set(executed):
        raise ValueError("UNRECORDED_RESPONSE_FILE")
    statuses = read(run / "QUESTION_STATUS.json")["questions"]
    status_by_id = {row["candidate_id"]: row for row in statuses}
    if len(status_by_id) != len(statuses) or set(status_by_id) != set(questions_by_id):
        raise ValueError("QUESTION_STATUS_SET_MISMATCH")
    results = {qid: evaluate_question(question, manifest, status_by_id[qid], requests, executed, run)
               for qid, question in questions_by_id.items()}
    counts = {"canonical_questions": len(questions),
              "selected_questions": sum(q["selected_by_policy"] is True for q in results.values()),
              "not_selected_questions": sum(q["selected_by_policy"] is False for q in results.values()),
              "unresolved_questions": sum(q["selected_by_policy"] is None for q in results.values()),
              "ground_filtered_questions": sum(q["ground"]["ground_pass"] is False
                                               for q in results.values()),
              "executed_ground_responses": sum("__no_context__" in rid for rid in executed),
              "executed_full_responses": sum("__full_context__" in rid for rid in executed),
              "new_solver_calls": len(executed), "actual_trained_questions": 0}
    return {"report_version": REPORT_VERSION, "policy_version": policy.VERSION,
            "evaluation_data_status": "VALID" if not counts["unresolved_questions"] else "INCOMPLETE_OR_INVALID",
            "selection_basis": "ground_pass AND 0 < equal_permutation_full_partial_mean < 1",
            "ground_sampling": "original_display_three_responses_adaptive_two_more",
            "full_variant_ids": list(VARIANTS), "full_rollouts_per_variant": 3,
            "full_score_formula": "TP/g - FP/(n-g) for 2<=g<6; TP/6 for g=6",
            "all_correct_gold_position_note": "For g=6, displayed Gold is ABCDEF under every permutation; content positions change, Gold label positions cannot.",
            "permutation_diagnostics_used_as_gate": False,
            "context_gain_used_as_gate_or_priority": False,
            "full_exact_success_required": False,
            "generator_provenance": pool["provenance"],
            "solver_model": "Qwen3-4B (base)", "solver_identity": identity,
            "run": str(run), "source_hashes_by_root": {root: row["sha256"] for root, row in sources.items()},
            "historical_files_unchanged": True, "counts": counts,
            "production_pool_modified": False, "sft_export_executed": False,
            "training_executed": False, "questions": results}


def report(result):
    counts = result["counts"]
    provenance = result["generator_provenance"]
    review_note = ("Qwen originals passed source/Gold mechanical checks; no Codex semantic review or question edits."
                   if provenance.get("mechanical_gate_version") == "raw_source_and_gold_v1"
                   else "Content-audited questions were evaluated.")
    lines = ["# Cross-Root Multiselect Solver Selection", "",
             "Generator: `Qwen/Qwen3.6-35B-A3B` (SiliconFlow). " + review_note,
             "Solver: `Qwen3-4B (base)`. Ground is original display K3, adaptive K5.",
             "Ground-passing questions receive six forward cyclic full displays, three responses each.",
             "Select iff ground passes and 0 < equal-display mean J < 1; J = TP/g - FP/(6-g) for g<6, TP/6 for g=6.",
             "For six-correct items, Gold is ABCDEF in every display; permutations change content positions but cannot vary Gold labels.",
             "Exact success, context gain and permutation range are diagnostics, not gates.", "",
             "| Root | Question | Ground exact | Full mean J | Full exact | Display range | Selected | Reason |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for qid, row in result["questions"].items():
        ground = row["ground"]
        show = lambda value: value["fraction"] if value is not None else "not evaluated"
        exact = (f"{row['full_exact_correct']}/18" if row["full_exact_correct"] is not None
                 else "not evaluated")
        lines.append(f"| {row['root_id']} | {qid} | "
                     f"{ground.get('exact_correct', '?')}/{ground.get('evaluated_count', '?')} "
                     f"| {show(row['full_partial_mean'])} | {exact} "
                     f"| {show(row['permutation_score_range'])} | {row['selected_by_policy']} "
                     f"| {row['selection_reason']} |")
        if row["data_errors"]:
            lines.append("\nData errors for `" + qid + "`: " + "; ".join(row["data_errors"]))
    lines += ["", f"{counts['canonical_questions']} canonical questions; {counts['selected_questions']} selected; "
              f"{counts['ground_filtered_questions']} ground-filtered; {counts['unresolved_questions']} unresolved.",
              f"Solver calls: {counts['new_solver_calls']} = {counts['executed_ground_responses']} ground + "
              f"{counts['executed_full_responses']} full.",
              "No SFT export, production pool change, or training occurred.", ""]
    for qid, row in result["questions"].items():
        if not row["full_responses"]:
            continue
        lines += [f"## {qid}", "", "| Display | J slot 0 | J slot 1 | J slot 2 | Mean J | Exact |",
                  "|---|---:|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            cells = [cell for cell in row["full_responses"] if cell["variant_id"] == variant]
            scores = [cell["partial_score"]["fraction"] if cell["partial_score"] else "invalid"
                      for cell in cells]
            mean = row["permutation_means"].get(variant, {}).get("fraction", "unresolved")
            exact = sum(cell["exact_match"] is True for cell in cells)
            lines.append(f"| {variant} | {' | '.join(scores)} | {mean} | {exact}/3 |")
        lines.append("")
    return "\n".join(lines)


def write_artifacts(out, artifacts):
    out.mkdir(parents=True, exist_ok=True)
    contents = {name: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
                for name, value in artifacts.items()}
    for name, content in contents.items():
        path = out / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError("VERSIONED_OUTPUT_ALREADY_EXISTS:" + str(path))
    for name, content in contents.items():
        path = out / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "run")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    result = analyze(args.run)
    selected = [row["canonical_question"] for row in result["questions"].values()
                if row["selected_by_policy"] is True]
    write_artifacts(args.output, {
        "RESULTS.json": result,
        "SELECTED_QUESTIONS.json": {"questions": selected, "count": len(selected),
                                    "source_result": str(args.output / "RESULTS.json"),
                                    "training_started": False},
        "GROUND_CACHE.json": {qid: row["ground_cache"] for qid, row in result["questions"].items()},
        "REPORT.md": report(result),
    })
    print(json.dumps({"status": result["evaluation_data_status"],
                      "selected": len(selected), "calls": result["counts"]["new_solver_calls"]}), flush=True)


if __name__ == "__main__":
    main()
