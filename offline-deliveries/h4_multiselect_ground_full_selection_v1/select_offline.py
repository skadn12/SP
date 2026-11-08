"""Re-score accepted responses and write independent, immutable selection artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

from selection import GROUND_VERSION, VARIANTS, VERSION, fraction, ground_decision, select_question


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "h4_multiselect_partial_admission_v1"
BASELINE = HERE.parent / "h4_259192_manual_qa_v2" / "probe_v3" / "run"
ACCEPTED = PREVIOUS / "followup_k3_revision_2" / "analysis.json"
ACCEPTED_SHA256 = "86632bb4abe42c1271ad47c6c6ab0aa561aaecd9ba27ebee8b2979a93d98e7bb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_load(path: Path, expected: str | None, sources: dict) -> object:
    if not expected or sha256(path) != expected:
        raise ValueError(f"SOURCE_HASH_MISMATCH:{path}")
    sources[str(path)] = expected
    return load(path)


def historical_snapshot() -> dict:
    return {str(path): sha256(path)
            for root in (PREVIOUS, BASELINE.parent.parent)
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts}


def score_row(question: dict, source_row: dict, accepted: dict, execution: dict,
              score_response, sources: dict) -> dict:
    request = source_row["request"]
    path = Path(source_row["response_path"])
    response = verified_load(path, accepted["hashes"].get(str(path)), sources)
    errors = []
    for field in ("request_id", "prompt_sha256", "seed"):
        if request.get(field) is None or response.get(field) != request[field]:
            errors.append(f"REQUEST_RESPONSE_MISMATCH:{field}")
    if response.get("mode") != request["mode"] or response.get("variant_id") != request["variant"]["variant_id"]:
        errors.append("REQUEST_RESPONSE_IDENTITY_MISMATCH")
    if response.get("model_identifier") != execution["model"]:
        errors.append("SOLVER_MODEL_MISMATCH")
    for field in ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version"):
        if response.get("metadata", {}).get(field) != execution["generation_config"].get(field):
            errors.append(f"DECODING_MISMATCH:{field}")
    if response.get("metadata", {}).get("finish_reason") not in {"stop", "eos"}:
        errors.append("INCOMPLETE_GENERATION")
    score = score_response(question, request, response)
    if score["scoring_status"] != "SCORED":
        errors.append(f"SCORING:{score['scoring_status']}:{score['parse_reason']}")
    valid = not errors
    j = Fraction(score["tp"], score["g"]) - Fraction(score["fp"], score["n"] - score["g"]) if valid else None
    prediction = set(score["canonical_prediction"] or [])
    gold = set(question["gold"])
    texts = lambda ids: {key: question["options"][key] for key in sorted(ids)}
    return {
        "request_id": request["request_id"], "mode": request["mode"],
        "variant_id": request["variant"]["variant_id"], "slot": request["slot"],
        "source_path": str(path), "source_sha256": sources[str(path)],
        "seed": request["seed"], "prompt_sha256": request["prompt_sha256"],
        "variant": request["variant"], "raw_output": response.get("raw_output"),
        "canonical_prediction": score["canonical_prediction"],
        "exact_match": score["exact_match"] if valid else None,
        "tp": score["tp"], "fp": score["fp"], "fn": score["fn"],
        "partial_score": fraction(j) if j is not None else None,
        "false_positive_contents": texts(prediction - gold) if valid else None,
        "false_negative_contents": texts(gold - prediction) if valid else None,
        "evaluation_valid": valid, "data_errors": errors,
    }


def evaluate_question(question: dict, old: dict, accepted: dict, execution: dict,
                      identity: dict, score_response, sources: dict) -> dict:
    errors, rows, seen = [], {}, set()
    if object_sha256(question) != old["question_sha256"]:
        errors.append("QUESTION_VERSION_MISMATCH")
    if question["gold"] != old["gold"]:
        errors.append("GOLD_VERSION_MISMATCH")
    ground_errors = list(errors)
    for source_row in old["responses"]:
        request = source_row["request"]
        mode, variant, slot = request["mode"], request["variant"]["variant_id"], request["slot"]
        needed = (mode == "full_context" and variant in VARIANTS) or (mode == "no_context" and variant == "perm_00")
        if not needed:
            continue
        key = mode, variant, slot
        if key in seen:
            issue = f"DUPLICATE_EVALUATION_SLOT:{key}"
            errors.append(issue)
            if mode == "no_context":
                ground_errors.append(issue)
            continue
        seen.add(key)
        try:
            rows[key] = score_row(question, source_row, accepted, execution, score_response, sources)
            issues = [f"{request['request_id']}:{error}" for error in rows[key]["data_errors"]]
            errors.extend(issues)
            if mode == "no_context":
                ground_errors.extend(issues)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue = f"{request.get('request_id')}:{exc}"
            errors.append(issue)
            if mode == "no_context":
                ground_errors.append(issue)
    expected = {("full_context", variant, slot) for variant in VARIANTS for slot in range(3)}
    expected |= {("no_context", "perm_00", slot) for slot in range(3)}
    if seen != expected:
        errors.append("REQUIRED_EVALUATION_SLOT_SET_MISMATCH")
    if {key for key in seen if key[0] == "no_context"} != {key for key in expected if key[0] == "no_context"}:
        ground_errors.append("GROUND_EVALUATION_SLOT_SET_MISMATCH")
    for mode, variant in sorted({(mode, variant) for mode, variant, _ in expected}):
        group = [rows[key] for key in sorted(rows) if key[:2] == (mode, variant)]
        if len({object_sha256(row["variant"]) for row in group}) > 1:
            issue = f"DISPLAY_CHANGED_WITHIN_REPEATS:{mode}:{variant}"
            errors.append(issue)
            if mode == "no_context":
                ground_errors.append(issue)
        if len({row["prompt_sha256"] for row in group}) > 1:
            issue = f"PROMPT_CHANGED_WITHIN_REPEATS:{mode}:{variant}"
            errors.append(issue)
            if mode == "no_context":
                ground_errors.append(issue)
    original = [rows.get(("no_context", "perm_00", slot)) for slot in range(3)]
    ground = ground_decision([row["exact_match"] if row else None for row in original])
    # This delivery only has K3 ground evidence. Boundary cases stay unresolved;
    # an offline selector must never schedule inference to obtain an extension.
    full_scores = {variant: [Fraction(row["partial_score"]["fraction"]) if row and row["partial_score"] else None
                            for slot in range(3)
                            for row in [rows.get(("full_context", variant, slot))]] for variant in VARIANTS}
    decision = select_question(ground, full_scores, data_errors=errors)
    full_rows = [rows[key] for key in sorted(rows) if key[0] == "full_context"]
    full_complete = len(full_rows) == 18 and all(row["evaluation_valid"] for row in full_rows) and not errors
    bindings = {
        "question_sha256": object_sha256(question), "gold": question["gold"],
        "gold_sha256": object_sha256(question["gold"]),
        "evaluation_variant": original[0]["variant"] if original[0] else None,
        "solver_checkpoint": {"model": execution["model"],
                              "fingerprint_sha256": identity["model_and_tokenizer_fingerprint_sha256"],
                              "weight_revision": identity["weight_revision"],
                              "identity_evidence_sha256": execution["model_identity_evidence"]["sha256"],
                              "evidence_scope": execution["model_identity_scope"]},
        "prompt_sha256": original[0]["prompt_sha256"] if original[0] else None,
        "decoding": execution["generation_config"], "ground_rule_version": GROUND_VERSION,
        "selection_policy_version": VERSION, "selection_code_sha256": sha256(HERE / "selection.py"),
    }
    cache = {"cache_key": object_sha256(bindings), "bindings": bindings, "decision": ground,
             "reusable": not ground_errors and ground["ground_pass"] is not None,
             "data_errors": ground_errors,
             "source_responses": [row for row in original if row is not None]}
    return {
        **decision, "root_id": question["root_id"], "canonical_question": question,
        "ground": ground, "ground_cache": cache,
        "full_expected_count": 18, "full_evaluated_count": len(full_rows),
        "full_exact_correct": sum(row["exact_match"] is True for row in full_rows) if full_complete else None,
        "full_exact_rate": sum(row["exact_match"] is True for row in full_rows) / 18 if full_complete else None,
        "full_exact_rate_fraction": str(Fraction(sum(row["exact_match"] is True for row in full_rows), 18)) if full_complete else None,
        "full_responses": full_rows,
        "legacy_policy_decision": {key: old[key] for key in (
            "policy_version", "new_policy_candidate_status", "new_policy_reason", "numeric_candidate", "blockers")},
        "diagnostics_not_used_for_selection": {
            "no_context_all_six": old["combined_k3"]["modes"]["no_context"],
            "partial_context_gain": old["combined_k3"]["partial_context_gain"],
            "exact_context_gain": old["combined_k3"]["exact_context_gain"],
            "position_mechanism": old["position_diagnostics"]["full_context"]["position_mechanism"],
        },
    }


def analyze() -> dict:
    sources = {}
    accepted = verified_load(ACCEPTED, ACCEPTED_SHA256, sources)
    hashes = accepted["hashes"]
    execution_path = ACCEPTED.parent / "execution_manifest.json"
    execution = verified_load(execution_path, hashes.get(str(execution_path)), sources)
    identity_path = Path(execution["model_identity_evidence"]["path"])
    identity = verified_load(identity_path, execution["model_identity_evidence"]["sha256"], sources)
    frozen = verified_load(BASELINE / "questions.json", hashes.get(str(BASELINE / "questions.json")), sources)
    gold = verified_load(BASELINE / "gold.json", hashes.get(str(BASELINE / "gold.json")), sources)
    scoring_path = PREVIOUS / "scoring.py"
    if sha256(scoring_path) != hashes.get(str(scoring_path)):
        raise ValueError("SCORING_CODE_HASH_MISMATCH")
    sources[str(scoring_path)] = hashes[str(scoring_path)]
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("frozen_multiselect_scoring", scoring_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    questions = {q["candidate_id"]: {**q, "gold": gold[q["candidate_id"]]}
                 for q in frozen["candidates"] if q["question_type"] == "multi_select"}
    if set(questions) != set(accepted["questions"]):
        raise ValueError("CANONICAL_QUESTION_SET_MISMATCH")
    results = {qid: evaluate_question(question, accepted["questions"][qid], accepted,
                                      execution, identity, scorer.score_response, sources)
               for qid, question in sorted(questions.items())}
    return {
        "policy_version": VERSION, "evaluation_data_status": "VALID" if not any(
            q["data_errors"] or q["selected_by_policy"] is None for q in results.values()) else "INCOMPLETE_OR_INVALID",
        "selection_basis": "ground_pass AND 0 < equal_permutation_full_partial_mean < 1",
        "eligibility_scope": "Question, complete Gold, and training split already confirmed; no new quality review.",
        "permutation_diagnostics_used_as_gate": False, "context_gain_used_as_gate_or_priority": False,
        "full_exact_success_required": False, "full_variant_ids": list(VARIANTS), "full_rollouts_per_variant": 3,
        "source_analysis": str(ACCEPTED), "source_hashes": sources,
        "implementation_hashes": {str(HERE / name): sha256(HERE / name)
                                  for name in ("selection.py", "select_offline.py", "POLICY.md")},
        "excluded_evaluations": accepted["excluded_runs"],
        "counts": {"canonical_questions": len(results),
                   "selected_questions": sum(q["selected_by_policy"] is True for q in results.values()),
                   "not_selected_questions": sum(q["selected_by_policy"] is False for q in results.values()),
                   "unresolved_questions": sum(q["selected_by_policy"] is None for q in results.values()),
                   "reused_ground_responses": sum(len(q["ground_cache"]["source_responses"]) for q in results.values()),
                   "reused_full_responses": sum(len(q["full_responses"]) for q in results.values()),
                   "new_solver_calls": 0, "new_generated_questions": 0, "actual_trained_questions": 0},
        "production_pool_modified": False, "sft_export_executed": False, "training_executed": False,
        "questions": results,
    }


def report(result: dict) -> str:
    lines = ["# Ground + Full Partial Selection V1", "",
             "Independent offline trial selection. Existing eligibility is assumed confirmed.",
             "Selected by this policy; permutation behavior is recorded separately, not certified stable.", "",
             "Rule: ground passes AND 0 < full partial mean < 1. No full > no requirement.",
             "Six forward displays, three full responses per display; exact fraction arithmetic and equal display weights.", "",
             "| Question | Ground exact | Full mean J | Full exact | Display mean minimum | Maximum | Range | Selected | Legacy status |",
             "|---|---|---|---|---|---|---|---|---|"]
    for qid, q in result["questions"].items():
        val = lambda key: q[key]["fraction"] if q[key] is not None else "unresolved"
        ground = q["ground"]
        lines.append(f"| {qid} | {ground.get('exact_correct', 'unresolved')}/{ground.get('evaluated_count', '?')} ({ground['status']}) "
                     f"| {val('full_partial_mean')} | {q['full_exact_correct']}/{q['full_expected_count']} "
                     f"| {val('permutation_score_min')} | {val('permutation_score_max')} | {val('permutation_score_range')} "
                     f"| {q['selected_by_policy']} | {q['legacy_policy_decision']['new_policy_candidate_status']} |")
    for qid, q in result["questions"].items():
        lines += ["", f"## {qid}", "", f"Selection reason: `{q['selection_reason']}`.", "",
                  "| Display | J slot 0 | J slot 1 | J slot 2 | Mean J | Exact correct |", "|---|---|---|---|---|---|"]
        for variant in VARIANTS:
            rows = sorted((row for row in q["full_responses"] if row["variant_id"] == variant), key=lambda row: row["slot"])
            scores = {row["slot"]: row["partial_score"]["fraction"] if row["partial_score"] else "invalid" for row in rows}
            mean = q["permutation_means"].get(variant, {}).get("fraction", "unresolved")
            exact = sum(row["exact_match"] is True for row in rows) if len(rows) == 3 and all(row["evaluation_valid"] for row in rows) else "unresolved"
            lines.append(f"| {variant} | {scores.get(0, 'missing')} | {scores.get(1, 'missing')} | {scores.get(2, 'missing')} | {mean} | {exact}/3 |")
        lines += ["", "Legacy gains and position mechanisms are diagnostics only; they do not reject or rank a selected item."]
        if q["data_errors"]:
            lines += ["", "Evaluation data errors: " + "; ".join(q["data_errors"])]
    lines += ["", "## Execution Scope", "",
              f"Selected canonical questions: {result['counts']['selected_questions']}. Actual training: 0.",
              f"Reused ground responses: {result['counts']['reused_ground_responses']}. "
              f"Reused full responses: {result['counts']['reused_full_responses']}. New Solver calls: 0.",
              "No generated questions, SFT export, production-pool update, or training-code changes.",
              "RESULTS.json retains raw full responses, mappings, scores, FP/FN content, and legacy decisions.",
              "GROUND_CACHE.json binds ground decisions to question/Gold/display/model/prompt/decode/rule versions.",
              "SELECTED_QUESTIONS.json lists canonical questions with complete Gold; it is not an SFT dataset.",
              "SOURCE_INTEGRITY.json verifies that historical source files were unchanged.", ""]
    return "\n".join(lines)


def write_artifacts(artifacts: dict[str, object]) -> None:
    contents = {name: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
                for name, value in artifacts.items()}
    for name, content in contents.items():
        path = HERE / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"VERSIONED_OUTPUT_ALREADY_EXISTS:{path}")
    for name, content in contents.items():
        path = HERE / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)


def main() -> None:
    before = historical_snapshot()
    try:
        result = analyze()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        write_artifacts({"EVALUATION_ERROR.json": {"policy_version": VERSION,
                        "evaluation_data_status": "INCOMPLETE_OR_INVALID", "selected_by_policy": None,
                        "error": str(exc), "new_solver_calls": 0, "training_executed": False}})
        raise SystemExit(f"Offline evaluation error; no selection fabricated: {exc}")
    after = historical_snapshot()
    if before != after:
        raise RuntimeError("HISTORICAL_SOURCE_CHANGED_DURING_OFFLINE_ANALYSIS")
    selected = [{"candidate_id": qid, **q["canonical_question"],
                 "selected_by_policy": True, "selection_reason": q["selection_reason"],
                 "full_partial_mean": q["full_partial_mean"]}
                for qid, q in result["questions"].items() if q["selected_by_policy"] is True]
    write_artifacts({
        "RESULTS.json": result,
        "SELECTED_QUESTIONS.json": {"policy_version": VERSION, "order_is_not_training_priority": True,
                                    "canonical_question_count": len(selected), "questions": selected,
                                    "sft_export_executed": False, "training_executed": False},
        "GROUND_CACHE.json": {"policy_version": VERSION, "entries": {
            qid: q["ground_cache"] for qid, q in result["questions"].items()}},
        "REPORT.md": report(result),
        "SOURCE_INTEGRITY.json": {"status": "PASS", "historical_files_unchanged": True,
                                  "protected_file_count": len(before), "protected_hashes": before},
    })
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
