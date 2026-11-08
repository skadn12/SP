"""Re-score frozen ordering answers; write a separate immutable offline selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from selection import (
    GROUND_EXTENSION, GROUND_VARIANTS, GROUND_VERSION, VARIANTS, VERSION,
    fraction, ground_decision, ordering_metrics, select_question,
)


HERE = Path(__file__).resolve().parent
PLAN = HERE / "SOURCE_PLAN.json"
PLAN_SHA256 = "e0d59085873b42eaeeda1b62bbaa75d778ff042984eca67aa74911c7aea61562"
BASE_ORDERS = ("412356", "345162", "256431", "624513", "531624", "163245")
DISPLAY_ORDERS = BASE_ORDERS + tuple(order[::-1] for order in BASE_ORDERS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def verified_load(path: Path, expected: str | None, sources: dict) -> object:
    if not expected or sha256(path) != expected:
        raise ValueError(f"SOURCE_HASH_MISMATCH:{path}")
    sources[str(path)] = expected
    return json.loads(path.read_text(encoding="utf-8"))


def historical_snapshot() -> dict:
    roots = (HERE.parent / "h4_259192_manual_qa_v2",
             HERE.parent / "h4_multiselect_partial_admission_v1",
             HERE.parent / "h4_multiselect_ground_full_selection_v1")
    snapshot = {str(path): sha256(path) for root in roots for path in sorted(root.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts}
    protocol = HERE.parent.parent / "current_method_github/scripts/current_method/restart_n5_spell/structured_qa_box_protocol.py"
    snapshot[str(protocol)] = sha256(protocol)
    return snapshot


def score_response(question: dict, request: dict, response: dict, manifest: dict, protocol) -> dict:
    if any(not isinstance(value, dict) for value in (question, request, response, manifest)):
        raise ValueError("QUESTION_REQUEST_RESPONSE_MANIFEST_OBJECTS_REQUIRED")
    if (question.get("question_type") != "ordering" or request.get("question_type") != "ordering"
            or request.get("candidate_id") != question["candidate_id"]):
        raise ValueError("ORDERING_QUESTION_REQUEST_MISMATCH")
    items, gold, variant = question["items"], question["gold"], request["variant"]
    if not isinstance(items, dict) or not isinstance(gold, list) or not isinstance(variant, dict):
        raise ValueError("ITEMS_GOLD_VARIANT_SCHEMA_INVALID")
    if set(gold) != set(items):
        raise ValueError("GOLD_ITEM_SET_MISMATCH")
    ordering_metrics(gold, gold)
    mapping, displayed = variant["display_to_canonical"], variant["items"]
    if not isinstance(mapping, dict) or not isinstance(displayed, dict):
        raise ValueError("MAPPING_DISPLAY_SCHEMA_INVALID")
    if set(mapping) != set(items) or set(mapping.values()) != set(items) or set(displayed) != set(items):
        raise ValueError("DISPLAY_MAPPING_NOT_BIJECTIVE")
    if any(displayed[key] != items[value] for key, value in mapping.items()):
        raise ValueError("DISPLAY_CONTENT_CHANGED")
    variant_id, mode, slot = variant["variant_id"], request["mode"], request["slot"]
    if variant_id not in VARIANTS or mode not in {"no_context", "full_context"} or type(slot) is not int or slot != 0:
        raise ValueError("EVALUATION_SLOT_INVALID")
    order = DISPLAY_ORDERS[VARIANTS.index(variant_id)]
    expected_mapping = {f"I{index + 1}": f"I{number}" for index, number in enumerate(order)}
    if mapping != expected_mapping or list(displayed) != list(expected_mapping):
        raise ValueError("FIXED_DISPLAY_ORDER_MISMATCH")
    expected_id = f"{question['candidate_id']}__{variant_id}__{mode}__s00"
    if request["request_id"] != expected_id:
        raise ValueError("REQUEST_ID_SLOT_MISMATCH")
    inverse = {canonical: display for display, canonical in mapping.items()}
    displayed_gold = [inverse[item] for item in gold]
    if variant["displayed_gold"] != displayed_gold or response.get("displayed_gold") != displayed_gold:
        raise ValueError("DISPLAYED_GOLD_MISMATCH")
    for field in ("request_id", "seed", "prompt_sha256"):
        if request.get(field) is None or response.get(field) != request[field]:
            raise ValueError(f"REQUEST_RESPONSE_MISMATCH:{field}")
    if type(request["seed"]) is not int or not re.fullmatch(r"[0-9a-f]{64}", request["prompt_sha256"]):
        raise ValueError("REQUEST_SEED_OR_PROMPT_HASH_INVALID")
    if response.get("mode") != mode or response.get("variant_id") != variant_id:
        raise ValueError("RESPONSE_EVALUATION_IDENTITY_MISMATCH")
    if response.get("model_identifier") != manifest["model"]:
        raise ValueError("SOLVER_MODEL_MISMATCH")
    metadata = response.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("RESPONSE_METADATA_OBJECT_REQUIRED")
    for field in ("temperature", "top_p", "max_tokens"):
        if metadata.get(field) != manifest[field]:
            raise ValueError(f"DECODING_MISMATCH:{field}")
    if (metadata.get("structured_output_api") != "structured_outputs=StructuredOutputsParams(choice)"
            or metadata.get("structured_output_version") != "finite_choice_set_v1"):
        raise ValueError("STRUCTURED_DECODING_MISMATCH")
    if metadata.get("sampling_seed") != request["seed"]:
        raise ValueError("SAMPLING_SEED_MISMATCH")
    if metadata.get("finish_reason") not in {"stop", "eos"}:
        raise ValueError("INCOMPLETE_GENERATION")
    raw = response.get("raw_output")
    if not isinstance(raw, str):
        raise ValueError("RAW_OUTPUT_MISSING_OR_NOT_TEXT")
    parsed = protocol.parse_structured_box(raw, question_type="ordering", allowed_item_ids=displayed)
    if parsed["parse_status"] != "success":
        raise ValueError(f"PARSE_FAILURE:{parsed.get('parse_reason')}")
    prediction = [mapping[item] for item in parsed["parsed_answer"]]
    metrics = ordering_metrics(prediction, gold)
    existing_metric = protocol.evaluate_ordering(prediction, gold)
    if (existing_metric["exact_order_match"] != metrics["exact_match"]
            or existing_metric["pairwise_accuracy"] != metrics["pairwise_accuracy"]["value"]):
        raise ValueError("EXISTING_ORDERING_METRIC_MISMATCH")
    inverted_contents = [{**pair, "gold_before_text": items[pair["gold_before"]],
                          "gold_after_text": items[pair["gold_after"]]}
                         for pair in metrics["inverted_pairs"]]
    return {
        "request_id": request["request_id"], "variant_id": variant_id, "mode": mode, "slot": slot,
        "seed": request["seed"], "prompt_sha256": request["prompt_sha256"], "variant": variant,
        "raw_output": raw, "display_prediction": parsed["parsed_answer"],
        "canonical_prediction": prediction, **metrics, "inverted_pair_contents": inverted_contents,
        "evaluation_valid": True, "data_errors": [],
    }


def evaluate_question(question: dict, manifest: dict, identity: dict, plan: dict,
                      protocol, sources: dict) -> dict:
    requests, duplicate_slots = {}, set()
    for request in manifest["requests"]:
        if request.get("candidate_id") != question["candidate_id"]:
            continue
        key = request["mode"], request["variant"]["variant_id"], request["slot"]
        if key in requests:
            duplicate_slots.add(key)
        requests[key] = request
    rows = {}
    for mode in ("no_context", "full_context"):
        for variant in VARIANTS:
            key = mode, variant, 0
            request = requests.get(key)
            row = {"request_id": None, "variant_id": variant, "mode": mode, "slot": 0,
                   "evaluation_valid": False, "exact_match": None, "partial_score": None,
                   "data_errors": []}
            try:
                if key in duplicate_slots:
                    raise ValueError("DUPLICATE_EVALUATION_SLOT")
                if request is None:
                    raise ValueError("REQUIRED_RESPONSE_REQUEST_MISSING")
                path = Path(plan["paths"]["baseline_run"]) / "responses" / (request["request_id"] + ".json")
                row["request_id"], row["source_path"] = request["request_id"], str(path)
                response = verified_load(path, plan["hashes"].get(str(path)), sources)
                row["source_sha256"] = sources[str(path)]
                row.update(score_response(question, request, response, manifest, protocol))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["data_errors"] = [str(exc)]
            rows[key] = row
    initial = [rows["no_context", variant, 0] for variant in GROUND_VARIANTS]
    ground = ground_decision([row["exact_match"] for row in initial])
    ground_rows = list(initial)
    if ground["status"] == "NEEDS_EXTENSION":
        extension = [rows["no_context", variant, 0] for variant in GROUND_EXTENSION]
        ground_rows += extension
        ground = ground_decision([row["exact_match"] for row in initial], [row["exact_match"] for row in extension])
    ground_errors = [f"{row['variant_id']}:{error}" for row in ground_rows for error in row["data_errors"]]
    full_rows = [rows["full_context", variant, 0] for variant in VARIANTS]
    errors = ground_errors + [f"{row['variant_id']}:{error}" for row in full_rows for error in row["data_errors"]]
    expected_full = {("full_context", variant, 0) for variant in VARIANTS}
    if {key for key in requests if key[0] == "full_context"} != expected_full:
        errors.append("FULL_REQUEST_SLOT_SET_MISMATCH")
    full_scores = {row["variant_id"]: [Fraction(row["partial_score"]["fraction"]) if row["partial_score"] else None]
                   for row in full_rows}
    decision = select_question(ground, full_scores, data_errors=errors)
    complete = all(row["evaluation_valid"] for row in full_rows) and not any(
        error.startswith("FULL_REQUEST") for error in errors)
    full_mean_a = sum((Fraction(row["pairwise_accuracy"]["fraction"]) for row in full_rows), Fraction()) / 12 if complete else None
    exact = sum(row["exact_match"] is True for row in full_rows) if complete else None
    bindings = {
        "question_sha256": object_sha256(question), "gold": question["gold"],
        "gold_sha256": object_sha256(question["gold"]),
        "ground_sampling": "different_displays_one_response_each",
        "fixed_initial_variant_ids": list(GROUND_VARIANTS), "fixed_extension_variant_ids": list(GROUND_EXTENSION),
        "required_evaluations": [{key: row.get(key) for key in
                                  ("variant_id", "slot", "variant", "request_id", "prompt_sha256", "seed")}
                                 for row in ground_rows],
        "solver_checkpoint": {"model": manifest["model"],
                              "fingerprint_sha256": identity["model_and_tokenizer_fingerprint_sha256"],
                              "weight_revision": identity["weight_revision"],
                              "identity_evidence_sha256": plan["hashes"][plan["paths"]["model_identity"]],
                              "evidence_scope": identity["evidence_strength"]},
        "decoding": {**{key: manifest[key] for key in ("temperature", "top_p", "max_tokens", "enable_thinking")},
                     "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                     "structured_output_version": "finite_choice_set_v1"},
        "ground_rule_version": GROUND_VERSION, "selection_policy_version": VERSION,
        "selection_code_sha256": sha256(HERE / "selection.py"),
        "prompt_manifest_sha256": sources[plan["paths"]["manifest"]],
    }
    cache = {"cache_key": object_sha256(bindings), "bindings": bindings, "decision": ground,
             "reusable": not ground_errors and ground["ground_pass"] is not None,
             "data_errors": ground_errors, "source_responses": ground_rows}
    diagnostic_no = [rows["no_context", variant, 0] for variant in VARIANTS]
    no_complete = all(row["evaluation_valid"] for row in diagnostic_no)
    no_mean = sum((Fraction(row["pairwise_accuracy"]["fraction"]) for row in diagnostic_no), Fraction()) / 12 if no_complete else None
    frequencies = Counter((pair["gold_before"], pair["gold_after"])
                          for row in full_rows if row["evaluation_valid"] for pair in row["inverted_pairs"])
    return {
        **decision, "root_id": question["root_id"], "canonical_question": question,
        "ground": ground, "ground_cache": cache,
        "full_expected_count": 12, "full_evaluated_count": sum(row["evaluation_valid"] for row in full_rows),
        "full_pairwise_accuracy_mean": fraction(full_mean_a) if full_mean_a is not None else None,
        "full_exact_correct": exact, "full_exact_rate": fraction(Fraction(exact, 12)) if exact is not None else None,
        "full_responses": full_rows,
        "diagnostics_not_used_for_selection": {
            "no_context_exact_correct": sum(row["exact_match"] is True for row in diagnostic_no) if no_complete else None,
            "no_context_response_count": 12, "no_context_pairwise_accuracy_mean": fraction(no_mean) if no_mean is not None else None,
            "pairwise_context_gain": fraction(full_mean_a - no_mean) if full_mean_a is not None and no_mean is not None else None,
            "no_context_responses": diagnostic_no,
            "full_inverted_pair_frequencies": [{"gold_before": before, "gold_after": after,
                                                "gold_before_text": question["items"][before],
                                                "gold_after_text": question["items"][after], "count": count}
                                               for (before, after), count in sorted(frequencies.items())] if complete else None,
            "historical_response_score_used_for_selection": False,
            "historical_status": "Probe-only evidence; no previous formal ordering selection completed.",
        },
    }


def analyze() -> dict:
    sources = {}
    plan = verified_load(PLAN, PLAN_SHA256, sources)
    if plan["policy_version"] != VERSION or plan["ground_rule"]["cache_rule_version"] != GROUND_VERSION:
        raise ValueError("SOURCE_PLAN_RULE_VERSION_MISMATCH")
    paths, hashes = plan["paths"], plan["hashes"]
    manifest = verified_load(Path(paths["manifest"]), hashes[paths["manifest"]], sources)
    frozen = verified_load(Path(paths["questions"]), hashes[paths["questions"]], sources)
    gold = verified_load(Path(paths["gold"]), hashes[paths["gold"]], sources)
    source = verified_load(Path(paths["source"]), hashes[paths["source"]], sources)
    identity = verified_load(Path(paths["model_identity"]), hashes[paths["model_identity"]], sources)
    started_path = Path(paths["baseline_run"]) / "STARTED.json"
    verified_load(started_path, hashes[str(started_path)], sources)
    for name in ("questions.json", "gold.json", "source.json"):
        path = str(Path(paths["baseline_run"]) / name)
        if manifest["input_sha256"][name] != sources[path]:
            raise ValueError(f"MANIFEST_INPUT_HASH_MISMATCH:{name}")
    if (identity["model_path"] != manifest["model"] or manifest["enable_thinking"] is not False
            or manifest["rollouts_per_position"] != 1):
        raise ValueError("MODEL_OR_EVALUATION_CONFIG_MISMATCH")
    matches = [q for q in frozen["candidates"] if q["candidate_id"] == plan["question_id"]]
    if len(matches) != 1:
        raise ValueError("CANONICAL_QUESTION_ID_MISSING_OR_DUPLICATE")
    question = {**matches[0], "gold": gold[plan["question_id"]]}
    if question["root_id"] != plan["root_id"] or source["root_id"] != plan["root_id"]:
        raise ValueError("ROOT_ID_MISMATCH")
    for key in ("boxed_protocol", "display_generator"):
        path = Path(paths[key])
        if sha256(path) != hashes[str(path)]:
            raise ValueError(f"FROZEN_HELPER_HASH_MISMATCH:{path}")
        sources[str(path)] = hashes[str(path)]
    # Import only the existing pure parser/metric, never probe or Solver modules.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("frozen_ordering_box_protocol", paths["boxed_protocol"])
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    q = evaluate_question(question, manifest, identity, plan, protocol, sources)
    return {
        "policy_version": VERSION,
        "evaluation_data_status": "VALID" if q["selected_by_policy"] is not None and not q["data_errors"] else "INCOMPLETE_OR_INVALID",
        "selection_basis": "ground_pass AND 0 < equal_permutation_full_net_pairwise_mean < 1",
        "eligibility_scope": "Question, complete Gold, and train eligibility already confirmed; no new quality review.",
        "ground_sampling": "different_displays_one_response_each",
        "ground_sampling_is_same_display_k3": False, "full_variant_ids": list(VARIANTS), "full_rollouts_per_variant": 1,
        "full_score_formula": "(correct_pairs - incorrect_pairs) / pair_count = 2 * pairwise_accuracy - 1",
        "permutation_diagnostics_used_as_gate": False, "context_gain_used_as_gate_or_priority": False,
        "full_exact_success_required": False, "source_plan": str(PLAN), "source_hashes": sources,
        "implementation_hashes": {str(HERE / name): sha256(HERE / name)
                                  for name in ("selection.py", "select_offline.py", "POLICY.md")},
        "counts": {"canonical_questions": 1, "selected_questions": int(q["selected_by_policy"] is True),
                   "not_selected_questions": int(q["selected_by_policy"] is False), "unresolved_questions": int(q["selected_by_policy"] is None),
                   "required_ground_responses": len(q["ground_cache"]["source_responses"]), "required_full_responses": 12,
                   "reused_ground_responses": sum("source_sha256" in row for row in q["ground_cache"]["source_responses"]),
                   "reused_full_responses": sum("source_sha256" in row for row in q["full_responses"]),
                   "existing_no_context_diagnostic_responses": sum("source_sha256" in row for row in
                       q["diagnostics_not_used_for_selection"]["no_context_responses"]),
                   "unique_existing_responses_read": sum("source_sha256" in row for row in
                       q["full_responses"] + q["diagnostics_not_used_for_selection"]["no_context_responses"]),
                   "new_solver_calls": 0, "new_generated_questions": 0, "actual_trained_questions": 0},
        "production_pool_modified": False, "sft_export_executed": False, "training_executed": False,
        "questions": {question["candidate_id"]: q},
    }


def report(result: dict) -> str:
    qid, q = next(iter(result["questions"].items()))
    value = lambda obj: obj["fraction"] if obj is not None else "unresolved"
    lines = ["# Ordering Ground + Full Selection V1", "",
             "Independent offline trial selection; permutation behavior recorded separately.", "",
             "Rule: ground passes AND 0 < full net pairwise mean < 1.",
             "Ground uses the fixed first three different displays, one answer each (not original-display K3).",
             "Full uses all twelve existing displays, one answer each (not multiselect six-display K3).", "",
             "| Question | Root | Ground | Mean net J | Mean pairwise accuracy | Full exact | Selected |",
             "|---|---|---|---|---|---|---|",
             f"| {qid} | {q['root_id']} | {q['ground'].get('exact_correct', '?')}/{q['ground'].get('evaluated_count', '?')} "
             f"({q['ground']['status']}) | {value(q['full_partial_mean'])} | {value(q['full_pairwise_accuracy_mean'])} "
             f"| {q['full_exact_correct']}/12 | {q['selected_by_policy']} |", "",
             f"Selection reason: `{q['selection_reason']}`.",
             f"Complete canonical Gold: `{','.join(q['canonical_question']['gold'])}`.",
             f"Net display range: {value(q['permutation_score_min'])} to {value(q['permutation_score_max'])}; "
             f"spread {value(q['permutation_score_range'])} (diagnostic only).", "",
             "| Display | Correct pairs / 15 | Pairwise accuracy | Net J | Exact | Canonical prediction |",
             "|---|---|---|---|---|---|"]
    for row in q["full_responses"]:
        lines.append(f"| {row['variant_id']} | {row.get('correct_pairs', '?')}/15 | {value(row.get('pairwise_accuracy'))} "
                     f"| {value(row['partial_score'])} | {row['exact_match']} | {','.join(row.get('canonical_prediction', []))} |")
    lines += ["", "## Errors Recorded", "",
              "| Gold direction | Full predictions inverting this pair |", "|---|---|"]
    for pair in q["diagnostics_not_used_for_selection"]["full_inverted_pair_frequencies"] or []:
        lines.append(f"| {pair['gold_before']} before {pair['gold_after']} | {pair['count']}/12 |")
    if q["data_errors"]:
        lines += ["", "Evaluation data errors: " + "; ".join(q["data_errors"])]
    counts = result["counts"]
    lines += ["", "## Scope", "",
              f"{counts['unique_existing_responses_read']} existing answers checked: "
              f"{counts['reused_ground_responses']} ground + 12 full evaluation slots; "
              f"{12 - counts['required_ground_responses']} additional no-context slots are diagnostics only.",
              "No new questions, Solver calls, SFT export, production-pool changes, or training.",
              f"Selected canonical questions: {counts['selected_questions']}; targets retain the complete Gold.",
              "Displays of one canonical question do not form independent questions.",
              "Existing multiselect selection remains separate and unchanged (two selected canonical questions).",
              "GROUND_CACHE.json binds every used ground display, prompt/seed, checkpoint evidence, decode and rules.",
              "RESULTS.json retains raw answers, mappings, exact/net/pairwise scores and inverted-pair content.",
              "SOURCE_PLAN.json pins source evidence; SOURCE_INTEGRITY.json verifies historical files unchanged.", ""]
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
    if before != historical_snapshot():
        raise RuntimeError("HISTORICAL_SOURCE_CHANGED_DURING_OFFLINE_ANALYSIS")
    selected = [{**q["canonical_question"], "selected_by_policy": True, "selection_reason": q["selection_reason"],
                 "full_partial_mean": q["full_partial_mean"], "full_pairwise_accuracy_mean": q["full_pairwise_accuracy_mean"]}
                for q in result["questions"].values() if q["selected_by_policy"] is True]
    write_artifacts({
        "RESULTS.json": result,
        "SELECTED_QUESTIONS.json": {"policy_version": VERSION, "canonical_question_count": len(selected),
                                    "order_is_not_training_priority": True, "questions": selected,
                                    "sft_export_executed": False, "training_executed": False},
        "GROUND_CACHE.json": {"policy_version": VERSION, "entries": {
            qid: q["ground_cache"] for qid, q in result["questions"].items()}},
        "REPORT.md": report(result),
        "SOURCE_INTEGRITY.json": {"status": "PASS", "historical_files_unchanged": True,
                                  "protected_file_count": len(before), "protected_hashes": before},
    })
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
