"""Score a completed adaptive Solver run without loading or calling the Solver."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
OLD = HERE.parent / "h4_ordering_ground_full_selection_v1"
GENERATOR = HERE.parent / "h4_ordering_large_model_generation_v1"
PROTOCOL_PATH = PROJECT / "current_method_github/scripts/current_method/restart_n5_spell/structured_qa_box_protocol.py"
REPORT_VERSION = "ordering_large_model_solver_selection_v1"
EXECUTION = "ground_first_adaptive_v1"
MODEL_NAME = "Qwen3-4B (base)"
GENERATOR_MODEL = "Qwen/Qwen3.6-35B-A3B"


def load_module(name: str, path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load_module("large_model_ordering_policy", OLD / "selection.py")
previous_selection = sys.modules.get("selection")
sys.modules["selection"] = policy
try:
    scorer = load_module("large_model_ordering_scorer", OLD / "select_offline.py")
finally:
    if previous_selection is None:
        del sys.modules["selection"]
    else:
        sys.modules["selection"] = previous_selection
protocol = load_module("large_model_ordering_protocol", PROTOCOL_PATH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path: Path, sources: dict, expected: str | None = None) -> object:
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"SOURCE_HASH_MISMATCH:{path}")
    sources[str(path)] = digest
    return json.loads(path.read_text(encoding="utf-8"))


def validate_request(question: dict, request: dict, run: Path, sources: dict) -> tuple:
    qid = question["candidate_id"]
    variant = request["variant"]
    variant_id, mode, slot = variant["variant_id"], request["mode"], request["slot"]
    ground_variants = policy.GROUND_VARIANTS + policy.GROUND_EXTENSION
    allowed = ground_variants if mode == "no_context" else policy.VARIANTS
    if (request.get("candidate_id") != qid or request.get("question_type") != "ordering"
            or mode not in {"no_context", "full_context"} or variant_id not in allowed
            or type(slot) is not int or slot != 0):
        raise ValueError("REQUEST_SLOT_IDENTITY_INVALID")
    expected_id = f"{qid}__{variant_id}__{mode}__s00"
    if request["request_id"] != expected_id:
        raise ValueError("REQUEST_ID_SLOT_MISMATCH")
    order = scorer.DISPLAY_ORDERS[policy.VARIANTS.index(variant_id)]
    mapping = {f"I{index + 1}": f"I{number}" for index, number in enumerate(order)}
    displayed = {label: question["items"][canonical] for label, canonical in mapping.items()}
    inverse = {canonical: label for label, canonical in mapping.items()}
    if (variant["display_to_canonical"] != mapping or variant["items"] != displayed
            or list(variant["items"]) != list(mapping)
            or variant["displayed_gold"] != [inverse[item] for item in question["gold"]]):
        raise ValueError("PLANNED_DISPLAY_MAPPING_OR_GOLD_INVALID")
    if (type(request["seed"]) is not int or type(request["prompt_tokens"]) is not int
            or request["prompt_tokens"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", request["prompt_sha256"])):
        raise ValueError("PLANNED_SEED_OR_PROMPT_METADATA_INVALID")
    prompt = run / "prompts" / (expected_id + ".txt")
    digest = sha256(prompt)
    if digest != request["prompt_sha256"]:
        raise ValueError(f"PROMPT_HASH_MISMATCH:{expected_id}")
    sources[str(prompt)] = digest
    return mode, variant_id, slot


def evaluate_question(question: dict, manifest: dict, status: dict, run: Path,
                      sources: dict, complete_ids: list[str]) -> dict:
    requests = {}
    for request in manifest["requests"]:
        if request.get("candidate_id") != question["candidate_id"]:
            continue
        key = validate_request(question, request, run, sources)
        if key in requests:
            raise ValueError("DUPLICATE_PLANNED_EVALUATION_SLOT")
        requests[key] = request
    expected = {("no_context", variant, 0) for variant in policy.GROUND_VARIANTS + policy.GROUND_EXTENSION}
    expected |= {("full_context", variant, 0) for variant in policy.VARIANTS}
    if set(requests) != expected:
        raise ValueError("PREPLANNED_SEVENTEEN_SLOT_SET_MISMATCH")
    errors, rows = [], {}
    actual_ids = [rid for rid in complete_ids if rid.startswith(question["candidate_id"] + "__")]
    status_ids = status.get("executed_request_ids")
    if (not isinstance(status_ids, list) or len(status_ids) != len(set(status_ids))
            or set(status_ids) != set(actual_ids)):
        errors.append("QUESTION_STATUS_EXECUTED_IDS_MISMATCH")

    def score(mode: str, variant: str) -> dict:
        key = mode, variant, 0
        request = requests[key]
        row = {"request_id": request["request_id"], "variant_id": variant, "mode": mode,
               "slot": 0, "evaluation_valid": False, "exact_match": None,
               "partial_score": None, "data_errors": []}
        path = run / "responses" / (request["request_id"] + ".json")
        try:
            if request["request_id"] not in actual_ids:
                raise ValueError("REQUIRED_EVALUATION_NOT_EXECUTED")
            response = read(path, sources)
            row.update(source_path=str(path), source_sha256=sources[str(path)])
            row.update(scorer.score_response(question, request, response, manifest, protocol))
            if response.get("metadata", {}).get("enable_thinking", False) is not False:
                raise ValueError("DECODING_MISMATCH:enable_thinking")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(evaluation_valid=False, exact_match=None, partial_score=None,
                       data_errors=[str(exc)])
        rows[key] = row
        return row

    ground_rows = [score("no_context", variant) for variant in policy.GROUND_VARIANTS]
    initial = [row["exact_match"] for row in ground_rows]
    ground = policy.ground_decision(initial)
    if ground["status"] == "NEEDS_EXTENSION":
        extension = [score("no_context", variant) for variant in policy.GROUND_EXTENSION]
        ground_rows += extension
        ground = policy.ground_decision(initial, [row["exact_match"] for row in extension])
    ground_errors = [f"{row['variant_id']}:{error}" for row in ground_rows for error in row["data_errors"]]
    errors += ground_errors
    reported_ground = status.get("ground", {})
    if not isinstance(reported_ground, dict) or any(reported_ground.get(key) != value for key, value in ground.items()):
        errors.append("QUESTION_STATUS_GROUND_DECISION_MISMATCH")
    required_ids = {row["request_id"] for row in ground_rows}
    full_rows = []
    if ground["ground_pass"] is True:
        full_rows = [score("full_context", variant) for variant in policy.VARIANTS]
        required_ids |= {row["request_id"] for row in full_rows}
        if status.get("full_skipped_reason") is not None:
            errors.append("PASSED_GROUND_FULL_MARKED_SKIPPED")
    else:
        expected_status = ("GROUND_FILTERED_FULL_SKIPPED" if ground["ground_pass"] is False
                           else "EVALUATION_INCOMPLETE_FULL_SKIPPED")
        if status.get("status") != expected_status or not status.get("full_skipped_reason"):
            errors.append("QUESTION_STATUS_FULL_SKIP_MISMATCH")
    if set(actual_ids) - required_ids:
        errors.append("UNREQUIRED_SOLVER_CALL_EXECUTED")
    if actual_ids != [row["request_id"] for row in ground_rows + full_rows]:
        errors.append("ADAPTIVE_EXECUTION_ORDER_OR_REQUIRED_CALL_SET_MISMATCH")
    errors += [f"{row['variant_id']}:{error}" for row in full_rows for error in row["data_errors"]]
    if full_rows:
        full_scores = {row["variant_id"]: [Fraction(row["partial_score"]["fraction"])
                                         if row["evaluation_valid"] else None] for row in full_rows}
        decision = policy.select_question(ground, full_scores, data_errors=errors)
    else:
        decision = {
            "policy_version": policy.VERSION, "ground_pass": ground["ground_pass"],
            "selected_by_policy": False if ground["ground_pass"] is False and not errors else None,
            "selection_reason": ("GROUND_FILTERED_FULL_SKIPPED" if ground["ground_pass"] is False and not errors
                                 else "EVALUATION_INCOMPLETE_OR_INVALID"),
            "full_partial_mean": None, "permutation_means": {}, "permutation_score_min": None,
            "permutation_score_max": None, "permutation_score_range": None,
            "data_errors": sorted(set(errors)),
        }
    complete = len(full_rows) == 12 and all(row["evaluation_valid"] for row in full_rows)
    mean_pairwise = (sum((Fraction(row["pairwise_accuracy"]["fraction"]) for row in full_rows), Fraction()) / 12
                     if complete else None)
    exact = sum(row["exact_match"] is True for row in full_rows) if complete else None
    planned = [requests["no_context", variant, 0] for variant in policy.GROUND_VARIANTS + policy.GROUND_EXTENSION]
    planned += [requests["full_context", variant, 0] for variant in policy.VARIANTS]
    bindings = {
        "question_sha256": object_sha256(question), "gold": question["gold"],
        "gold_sha256": object_sha256(question["gold"]),
        "source_sha256": manifest["input_sha256"]["source.json"],
        "ground_sampling": "different_displays_one_response_each",
        "fixed_initial_variant_ids": list(policy.GROUND_VARIANTS),
        "fixed_extension_variant_ids": list(policy.GROUND_EXTENSION),
        "planned_evaluations": planned,
        "required_ground_request_ids": [row["request_id"] for row in ground_rows],
        "solver_checkpoint": manifest["solver_identity"],
        "solver_identity_sha256": object_sha256(manifest["solver_identity"]),
        "decoding": {**{key: manifest[key] for key in ("temperature", "top_p", "max_tokens", "enable_thinking")},
                     "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                     "structured_output_version": "finite_choice_set_v1"},
        "ground_rule_version": policy.GROUND_VERSION, "selection_policy_version": policy.VERSION,
        "execution_version": EXECUTION, "report_version": REPORT_VERSION,
        "prompt_manifest_sha256": sources.get(str(run / "manifest.json"), object_sha256(manifest)),
        "implementation_sha256": {str(path): sha256(path) for path in
                                  (OLD / "selection.py", OLD / "select_offline.py", PROTOCOL_PATH, Path(__file__))},
        "runner_code_sha256": manifest.get("code_sha256", {}),
    }
    cache_errors = ground_errors + [error for error in errors if error in {
        "QUESTION_STATUS_EXECUTED_IDS_MISMATCH", "QUESTION_STATUS_GROUND_DECISION_MISMATCH", "UNREQUIRED_SOLVER_CALL_EXECUTED"}]
    cache = {"cache_key": object_sha256(bindings), "bindings": bindings, "decision": ground,
             "reusable": not cache_errors and ground["ground_pass"] is not None,
             "data_errors": sorted(set(cache_errors)),
             "source_responses": [row for row in ground_rows if "source_sha256" in row]}
    frequencies = Counter((pair["gold_before"], pair["gold_after"])
                          for row in full_rows if row["evaluation_valid"] for pair in row["inverted_pairs"])
    return {
        **decision, "canonical_question": question, "root_id": question["root_id"],
        "ground": ground, "ground_responses": ground_rows, "ground_cache": cache,
        "execution_status": status.get("status"), "executed_request_ids": actual_ids,
        "full_skipped_reason": status.get("full_skipped_reason"), "full_planned_count": 12,
        "full_expected_count": 12 if ground["ground_pass"] is True else 0,
        "full_executed_count": sum(rid in actual_ids for rid in
                                   (request["request_id"] for request in planned if request["mode"] == "full_context")),
        "full_evaluated_count": sum(row["evaluation_valid"] for row in full_rows),
        "full_pairwise_accuracy_mean": policy.fraction(mean_pairwise) if mean_pairwise is not None else None,
        "full_exact_correct": exact,
        "full_exact_rate": policy.fraction(Fraction(exact, 12)) if exact is not None else None,
        "full_responses": full_rows,
        "diagnostics_not_used_for_selection": {
            "full_inverted_pair_frequencies": [{"gold_before": before, "gold_after": after,
                                                "gold_before_text": question["items"][before],
                                                "gold_after_text": question["items"][after], "count": count}
                                               for (before, after), count in sorted(frequencies.items())] if complete else None,
            "context_gain_used_as_gate_or_priority": False, "permutation_range_used_as_gate": False,
            "full_exact_success_required": False,
        },
    }


def analyze(run: Path) -> dict:
    sources = {}
    manifest = read(run / "manifest.json", sources)
    if (manifest.get("execution") != EXECUTION or manifest.get("enable_thinking") is not False
            or manifest.get("temperature") != 0.8 or manifest.get("top_p") != 0.9
            or manifest.get("max_tokens") != 32 or manifest.get("rollouts_per_position") != 1
            or manifest.get("policy_version") != policy.VERSION
            or manifest.get("ground_rule_version") != policy.GROUND_VERSION):
        raise ValueError("ADAPTIVE_RUN_DECODING_CONFIG_MISMATCH")
    identity = manifest["solver_identity"]
    if (identity["model_path"] != manifest["model"]
            or not re.fullmatch(r"[0-9a-f]{64}", identity["model_and_tokenizer_fingerprint_sha256"])
            or not identity.get("evidence_strength") or "weight_revision" not in identity):
        raise ValueError("SOLVER_IDENTITY_EVIDENCE_MISMATCH")
    for key in ("code_sha256", "historical_snapshot"):
        if not isinstance(manifest.get(key), dict) or not manifest[key]:
            raise ValueError(f"REQUIRED_PINNED_HASHES_MISSING:{key}")
        for path, expected in manifest[key].items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"PINNED_FILE_CHANGED:{path}")
    frozen = read(run / "questions.json", sources, manifest["input_sha256"]["questions.json"])
    source = read(run / "source.json", sources, manifest["input_sha256"]["source.json"])
    if source.get("root_id") != "cc-train-episode-259192" or len(source["turns"]) != 188:
        raise ValueError("FROZEN_ROOT_OR_SOURCE_TURN_COUNT_MISMATCH")
    ready = read(GENERATOR / "READY_FOR_EVALUATION.json", sources)
    if manifest.get("pool_sha256") != sources[str(GENERATOR / "READY_FOR_EVALUATION.json")]:
        raise ValueError("FROZEN_READY_POOL_HASH_MISMATCH")
    expected_questions = {q["candidate_id"]: q for q in ready["questions"]}
    questions = frozen["candidates"]
    ids = [question["candidate_id"] for question in questions]
    if (len(ids) != len(set(ids)) or set(ids) != set(expected_questions)
            or any(question != expected_questions[question["candidate_id"]] for question in questions)):
        raise ValueError("SOURCE_SUPPORTED_GENERATOR_CANDIDATE_POOL_MISMATCH")
    provenance = manifest["generator_provenance"]
    if (provenance.get("requested_model") != GENERATOR_MODEL or provenance.get("returned_model") != GENERATOR_MODEL
            or provenance.get("source_sha256") != manifest["input_sha256"]["source.json"]
            or provenance != ready["provenance"]):
        raise ValueError("GENERATOR_PROVENANCE_MISMATCH")
    if manifest.get("candidate_count") != len(questions) or manifest.get("request_count") != len(questions) * 17:
        raise ValueError("PLANNED_CANDIDATE_OR_REQUEST_COUNT_MISMATCH")
    if len(manifest["requests"]) != len(questions) * 17:
        raise ValueError("REQUEST_ARRAY_COUNT_MISMATCH")
    planned_ids = [request["request_id"] for request in manifest["requests"]]
    if len(planned_ids) != len(set(planned_ids)):
        raise ValueError("DUPLICATE_PLANNED_REQUEST_ID")
    complete = read(run / "COMPLETE.json", sources)
    complete_ids = complete["executed_request_ids"]
    if (not isinstance(complete_ids, list) or len(complete_ids) != len(set(complete_ids))
            or set(complete_ids) - set(planned_ids)):
        raise ValueError("COMPLETE_EXECUTED_REQUEST_ID_SET_INVALID")
    if "actual_call_count" in complete and complete["actual_call_count"] != len(complete_ids):
        raise ValueError("COMPLETE_ACTUAL_CALL_COUNT_MISMATCH")
    existing_ids = {path.stem for path in (run / "responses").glob("*.json")}
    if existing_ids - set(complete_ids):
        raise ValueError("UNRECORDED_RESPONSE_FILE")
    status = read(run / "QUESTION_STATUS.json", sources)
    statuses = {row["candidate_id"]: row for row in status["questions"]}
    if len(statuses) != len(status["questions"]) or set(statuses) != set(ids):
        raise ValueError("QUESTION_STATUS_SET_MISMATCH")
    results = {}
    for question in questions:
        if (question["question_type"] != "ordering" or question["root_id"] != source["root_id"]
                or set(question["items"]) != {f"I{number}" for number in range(1, 7)}):
            raise ValueError("CANONICAL_SIX_ITEM_ORDERING_REQUIRED")
        policy.ordering_metrics(question["gold"], list(question["items"]))
        results[question["candidate_id"]] = evaluate_question(
            question, manifest, statuses[question["candidate_id"]], run, sources, complete_ids)
    counts = {
        "canonical_questions": len(questions),
        "selected_questions": sum(q["selected_by_policy"] is True for q in results.values()),
        "not_selected_questions": sum(q["selected_by_policy"] is False for q in results.values()),
        "unresolved_questions": sum(q["selected_by_policy"] is None for q in results.values()),
        "ground_filtered_questions": sum(q["ground"]["ground_pass"] is False for q in results.values()),
        "executed_ground_responses": sum("__no_context__" in rid for rid in complete_ids),
        "executed_full_responses": sum("__full_context__" in rid for rid in complete_ids),
        "new_solver_calls": len(complete_ids), "actual_trained_questions": 0,
    }
    return {
        "report_version": REPORT_VERSION, "policy_version": policy.VERSION, "execution": EXECUTION,
        "evaluation_data_status": "VALID" if not counts["unresolved_questions"] else "INCOMPLETE_OR_INVALID",
        "selection_basis": "ground_pass AND 0 < equal_permutation_full_net_pairwise_mean < 1",
        "ground_sampling": "different_displays_one_response_each", "ground_sampling_is_same_display_k3": False,
        "full_variant_ids": list(policy.VARIANTS), "full_rollouts_per_variant": 1,
        "full_score_formula": "(correct_pairs - incorrect_pairs) / 15 = 2 * pairwise_accuracy - 1",
        "permutation_diagnostics_used_as_gate": False, "context_gain_used_as_gate_or_priority": False,
        "full_exact_success_required": False, "generator_provenance": provenance,
        "generator_model": GENERATOR_MODEL, "solver_model": MODEL_NAME, "solver_identity": identity,
        "run": str(run), "source_hashes": sources, "historical_files_unchanged": True,
        "historical_protected_file_count": len(manifest["historical_snapshot"]), "counts": counts,
        "production_pool_modified": False, "sft_export_executed": False, "training_executed": False,
        "questions": results,
    }


def report(result: dict) -> str:
    value = lambda obj: obj["fraction"] if obj is not None else "not evaluated"
    lines = ["# Large-Model Ordering Solver Selection", "",
             f"Generator: `{result['generator_model']}` (SiliconFlow).",
             f"Solver: **{result['solver_model']}**, checkpoint `{result['solver_identity']['model_path']}`.",
             "Root: `cc-train-episode-259192`; 188 complete dialogue turns through R03 (134 prior + 54 R03).", "",
             "Select when ground passes and the mean full net pairwise score is strictly between 0 and 1.",
             "Ground uses three different fixed displays, with two more only after exactly two correct initial answers.",
             "Ground-filtered questions skip full. Passed questions receive all 12 fixed full displays, one answer each.",
             "Negative scores are retained. Exact success and permutation range are diagnostics only.", "",
             "| Question | Ground exact | Ground | Full J mean | Full exact | J range | Selected |",
             "|---|---|---|---|---|---|---|"]
    for qid, q in result["questions"].items():
        ground = q["ground"]
        exact = f"{q['full_exact_correct']}/12" if q["full_exact_correct"] is not None else "not evaluated"
        lines.append(f"| {qid} | {ground.get('exact_correct', '?')}/{ground.get('evaluated_count', '?')} | "
                     f"{ground['status']} | {value(q['full_partial_mean'])} | {exact} | "
                     f"{value(q['permutation_score_range'])} | {q['selected_by_policy']} |")
    for qid, q in result["questions"].items():
        lines += ["", f"## {qid}", "", q["canonical_question"]["question"], "",
                  f"Reason: `{q['selection_reason']}`.",
                  "Complete canonical Gold: `" + ",".join(q["canonical_question"]["gold"]) + "`."]
        if q["full_skipped_reason"]:
            lines += ["", "Full skipped: `" + str(q["full_skipped_reason"]) + "`."]
        if q["full_responses"]:
            lines += ["", f"Full J minimum/maximum: {value(q['permutation_score_min'])} / "
                      f"{value(q['permutation_score_max'])}; range {value(q['permutation_score_range'])}.", "",
                      "| Display | Correct pairs / 15 | J | Exact | Canonical prediction |",
                      "|---|---|---|---|---|"]
            for row in q["full_responses"]:
                lines.append(f"| {row['variant_id']} | {row.get('correct_pairs', '?')}/15 | "
                             f"{value(row['partial_score'])} | {row['exact_match']} | "
                             f"{','.join(row.get('canonical_prediction', []))} |")
        if q["data_errors"]:
            lines += ["", "Evaluation errors: " + "; ".join(q["data_errors"])]
    counts = result["counts"]
    lines += ["", "## Scope", "",
              f"{counts['canonical_questions']} canonical candidates; {counts['selected_questions']} selected; "
              f"{counts['ground_filtered_questions']} ground-filtered; {counts['unresolved_questions']} unresolved.",
              f"Actual Solver calls: {counts['new_solver_calls']} = {counts['executed_ground_responses']} ground + "
              f"{counts['executed_full_responses']} full.",
              "Selected means passed this training selection policy; permutation behavior is recorded separately.",
              "Questions retain the full Gold as their target. Displays are not independent questions.",
              "No SFT export, production pool changes, or training was executed.",
              "Historical artifacts were hash-checked unchanged. Ground cache binds all 17 planned slots and only actual ground responses.",
              "RESULTS.json retains raw answers, mappings, scores, and inverted-pair content.", ""]
    return "\n".join(lines)


def write_artifacts(output: Path, artifacts: dict[str, object]) -> None:
    contents = {name: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
                for name, value in artifacts.items()}
    for name, content in contents.items():
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"VERSIONED_OUTPUT_ALREADY_EXISTS:{path}")
    for name, content in contents.items():
        path = output / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "run")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    result = analyze(args.run.resolve())
    selected = [{**q["canonical_question"], "selected_by_policy": True,
                 "selection_reason": q["selection_reason"], "full_partial_mean": q["full_partial_mean"],
                 "full_pairwise_accuracy_mean": q["full_pairwise_accuracy_mean"]}
                for q in result["questions"].values() if q["selected_by_policy"] is True]
    write_artifacts(args.output, {
        "RESULTS.json": result,
        "SELECTED_QUESTIONS.json": {"report_version": REPORT_VERSION, "policy_version": policy.VERSION,
                                    "canonical_question_count": len(selected), "questions": selected,
                                    "order_is_not_training_priority": True,
                                    "sft_export_executed": False, "training_executed": False},
        "GROUND_CACHE.json": {"report_version": REPORT_VERSION, "policy_version": policy.VERSION,
                              "entries": {qid: q["ground_cache"] for qid, q in result["questions"].items()}},
        "REPORT.md": report(result),
    })
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
