"""Versioned R02 reported-statement generation with source-scoped explanations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "h4_crossroot_multiselect_batch_v4/generate.py"
V4_SHA256 = "a1e65ddacfe877db12d40e131208f7d65bc4bfaaefacbd01a5b84f18f6a94e5e"
LABELS = tuple("ABCDEF")
FIELDS = {
    "candidate_id", "root_id", "question_type", "topic", "question",
    "options", "gold", "option_evidence", "answer_reason",
}


def valid_gold(gold: object) -> bool:
    return (isinstance(gold, list) and 2 <= len(gold) <= 6
            and all(isinstance(label, str) for label in gold)
            and len(set(gold)) == len(gold) and set(gold) <= set(LABELS))


def check_candidate(row: object, root_id: str, turns: list[dict]) -> list[str]:
    if not isinstance(row, dict):
        return ["candidate_not_object"]
    errors = []
    if set(row) != FIELDS:
        errors.append("candidate_schema_fields_mismatch")
    if row.get("candidate_id") != f"{root_id.rsplit('-', 1)[-1]}-qwen36-multi-01":
        errors.append("candidate_id_invalid")
    if row.get("root_id") != root_id or row.get("question_type") != "multi_select":
        errors.append("root_or_type_mismatch")
    for field in ("topic", "question", "answer_reason"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{field}_missing")
    question = row.get("question")
    if isinstance(question, str):
        if "select all that apply" not in question.lower():
            errors.append("select_all_instruction_missing")
        if "r02" not in question.lower():
            errors.append("question_not_r02_scoped")
    options = row.get("options")
    if (not isinstance(options, dict) or set(options) != set(LABELS) or
            any(not isinstance(value, str) or not value.strip() for value in options.values())):
        errors.append("six_options_required")
    elif len({value.strip().casefold() for value in options.values()}) != 6:
        errors.append("duplicate_option_text")
    gold = row.get("gold")
    if not valid_gold(gold):
        errors.append("gold_must_have_2_to_6_distinct_labels")
    evidence = row.get("option_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(LABELS):
        return errors + ["six_evidence_entries_required"]
    by_ref = {turn["turn_ref"]: turn for turn in turns}
    gold_refs = set()
    for label in LABELS:
        item = evidence[label]
        if not isinstance(item, dict):
            errors.append(f"{label}:evidence_not_object")
            continue
        if set(item) != {"turn_ref", "quote", "relation", "reason"}:
            errors.append(f"{label}:evidence_schema_fields_mismatch")
        ref, quote = item.get("turn_ref"), item.get("quote")
        if not isinstance(ref, str) or not ref.startswith("R02:") or ref not in by_ref:
            errors.append(f"{label}:invalid_r02_turn_ref")
        elif not isinstance(quote, str) or not quote.strip() or quote not in by_ref[ref]["text"]:
            errors.append(f"{label}:quote_not_verbatim")
        if isinstance(quote, str) and len(quote) > 240:
            errors.append(f"{label}:quote_too_long")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            errors.append(f"{label}:reason_missing")
        if valid_gold(gold):
            expected = "supports" if label in gold else "contradicts"
            if item.get("relation") != expected:
                errors.append(f"{label}:relation_mismatch")
            if label in gold and isinstance(ref, str) and ref.startswith("R02:"):
                gold_refs.add(ref)
    if valid_gold(gold) and len(gold_refs) < 2:
        errors.append("fewer_than_two_distinct_r02_gold_anchors")
    return errors


def target_gold(root_id: str, count: int) -> tuple[str, ...]:
    """Choose distinct non-prefix patterns across the three fixed roots."""
    patterns = {
        2: ("AD", "BE", "CF"),
        3: ("ACE", "BDF", "ADF"),
        4: ("ABDF", "ACDF", "BCEF"),
        5: ("BCDEF", "ABDEF", "ABCDF"),
        6: ("ABCDEF", "ABCDEF", "ABCDEF"),
    }
    suffixes = ("136202", "197635", "254852")
    suffix = root_id.rsplit("-", 1)[-1]
    if suffix not in suffixes or count not in patterns:
        raise ValueError("unsupported_root_or_gold_count")
    return tuple(patterns[count][suffixes.index(suffix)])


def relabel(row: dict) -> tuple[dict, dict]:
    """Only A-F labels move; source-label explanations are retained verbatim."""
    original_gold = set(row["gold"])
    target = set(target_gold(row["root_id"], len(original_gold)))
    old_true = [label for label in LABELS if label in original_gold]
    old_false = [label for label in LABELS if label not in original_gold]
    new_true = [label for label in LABELS if label in target]
    new_false = [label for label in LABELS if label not in target]
    old_to_new = dict(zip(old_true + old_false, new_true + new_false))
    new_to_old = {new: old for old, new in old_to_new.items()}
    derived = dict(row)
    derived["options"] = {new: row["options"][new_to_old[new]] for new in LABELS}
    derived["gold"] = [label for label in LABELS if label in target]
    derived["option_evidence"] = {
        new: row["option_evidence"][new_to_old[new]] for new in LABELS}
    meta = {
        "candidate_id": row["candidate_id"],
        "source_candidate_sha256": hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "source_gold": [label for label in LABELS if label in original_gold],
        "derived_gold": derived["gold"],
        "source_to_derived_label": old_to_new,
        "derived_to_source_label": new_to_old,
        "six_correct_position_exception": len(original_gold) == 6,
        "relabel_rule": "fixed_root_cardinality_patterns_v1",
        "explanation_label_scope": "source_labels_only",
        "ready_for_evaluation": False,
        "admission_blocker": "DERIVED_EXPLANATIONS_USE_SOURCE_LABELS",
    }
    meta["derived_candidate_sha256"] = hashlib.sha256(
        json.dumps(derived, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return derived, meta


def runner():
    if hashlib.sha256(V4.read_bytes()).hexdigest() != V4_SHA256:
        raise ValueError("v4_runner_dependency_changed")
    spec = importlib.util.spec_from_file_location("h4_crossroot_multiselect_batch_v4_runner", V4)
    if spec is None or spec.loader is None:
        raise ImportError("v4_runner_unavailable")
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    v4.HERE = HERE
    v4.__file__ = __file__
    prior_generate = sys.modules.pop("generate", None)
    try:
        batch = v4.runner()
    finally:
        sys.modules.pop("generate", None)
        if prior_generate is not None:
            sys.modules["generate"] = prior_generate

    def validate(root_id: str) -> dict:
        entry, source, _, _ = batch.frozen_inputs(root_id)
        directory = HERE / root_id
        response = batch.read_json(directory / "RESPONSE.json")
        if (response.get("source_sha256") != entry["source_sha256"] or
                response.get("payload_sha256") != entry["payload_sha256"]):
            raise ValueError("response_binding_changed")
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("response_metadata_invalid")
        try:
            parsed = json.loads(response["raw_text"])
        except (TypeError, ValueError):
            parsed = None
        questions = parsed.get("questions") if isinstance(parsed, dict) else None
        abstentions = parsed.get("abstentions") if isinstance(parsed, dict) else None
        global_errors = []
        if not isinstance(parsed, dict) or set(parsed) != {"questions", "abstentions"}:
            global_errors.append("top_level_schema_mismatch")
        if not isinstance(questions, list):
            global_errors.append("questions_array_missing")
            questions = []
        if len(questions) > 1:
            global_errors.append("candidate_count_exceeds_one")
        if not isinstance(abstentions, list) or any(not isinstance(value, str) for value in abstentions):
            global_errors.append("abstentions_array_invalid")
        elif not questions and not abstentions:
            global_errors.append("missing_abstention_reason")
        if metadata.get("returned_model") != batch.MODEL:
            global_errors.append("returned_model_mismatch")
        if metadata.get("finish_reason") == "length":
            global_errors.append("response_truncated")
        checks, valid = [], []
        for index, question in enumerate(questions, 1):
            errors = check_candidate(question, root_id, source["turns"])
            candidate_id = question.get("candidate_id") if isinstance(question, dict) else None
            checks.append({"index": index, "candidate_id": candidate_id, "errors": errors})
            if not errors:
                valid.append(question)
        if global_errors:
            valid = []
        derived = [relabel(question) for question in valid]
        provenance = {
            "author": "REMOTE_LARGE_MODEL", "provider": "siliconflow",
            "requested_model": batch.MODEL, "returned_model": metadata.get("returned_model"),
            "response_id": (metadata.get("raw_response") or {}).get("id"),
            "root_id": root_id, "source_sha256": entry["source_sha256"],
            "generation_view_type": v4.VIEW_TYPE,
            "generation_view_sha256": entry["generation_view_sha256"],
            "prompt_sha256": entry["prompt_sha256"], "payload_sha256": entry["payload_sha256"],
            "model_questions_modified_by_codex": False,
            "derived_candidates_relabelled": True,
            "derived_explanation_label_scope": "source_labels_only",
            "derived_ready_for_evaluation": False,
            "solver_evaluated": False, "selected_for_training": False,
            "training_started": False,
        }
        status = ("MECHANICAL_REJECTED" if global_errors or (questions and not valid) else
                  "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if questions else
                  "NO_CANDIDATE_ABSTAINED")
        result = {
            "status": status, "root_id": root_id,
            "candidate_count": len(questions), "mechanically_valid_count": len(valid),
            "global_errors": global_errors, "checks": checks,
            "semantic_gold_verified": False, "provenance": provenance,
            "derived_explanation_label_scope": "source_labels_only",
            "derived_ready_for_evaluation": False,
            "derived_admission_blocker": "DERIVED_EXPLANATIONS_USE_SOURCE_LABELS",
            "derived_gold_patterns": ["".join(item[1]["derived_gold"]) for item in derived],
            "six_correct_position_exceptions": sum(item[1]["six_correct_position_exception"]
                                                   for item in derived),
        }
        batch.write_new(directory / "PARSED_RESPONSE.json", parsed)
        batch.write_new(directory / "VALIDATED_CANDIDATES.json", {
            "status": status, "provenance": provenance, "questions": valid,
        })
        batch.write_new(directory / "DERIVED_CANDIDATES.json", {
            "status": status, "content_audited": False,
            "relabel_only": True, "position_rule": "fixed_root_cardinality_patterns_v1",
            "explanation_label_scope": "source_labels_only",
            "ready_for_evaluation": False,
            "admission_blocker": "DERIVED_EXPLANATIONS_USE_SOURCE_LABELS",
            "questions": [item[0] for item in derived],
            "mappings": [item[1] for item in derived],
        })
        batch.write_new(directory / "VALIDATION.json", result)
        return result

    batch.check_candidate = check_candidate
    batch.validate = validate
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--execute", choices=(
        "cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852"))
    actions.add_argument("--validate", choices=(
        "cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852"))
    actions.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    batch = runner()
    if args.preflight:
        batch.preflight()
    elif args.execute:
        batch.execute(args.execute)
    elif args.validate:
        print(json.dumps(batch.validate(args.validate), ensure_ascii=False))
    elif args.summary:
        batch.summary()
    else:
        batch.prepare()
