"""Independent, fault-isolated validation of immutable v16 model responses."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V16 = HERE.parent / "h4_crossroot_multiselect_batch_v16/generate.py"
V16_SHA256 = "035a3aef0682dd1426bceb61847bac0d1e7c8aec9bb99350a6c86a7043c0b164"
ROOTS = ("cc-train-episode-136202", "cc-train-episode-197635",
         "cc-train-episode-254852")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_v16():
    if sha(V16.read_bytes()) != V16_SHA256:
        raise ValueError("v16_generator_changed")
    spec = importlib.util.spec_from_file_location("v16_frozen_revalidation", V16)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_refs(module, row: object) -> set[str]:
    if not isinstance(row, dict):
        return set()
    return {item["turn_ref"] for item in module.evidence_items(row)
            if isinstance(item.get("turn_ref"), str)}


def inspect_questions(module, v8, root_id: str, source: dict, context: dict,
                      prior: list[dict], questions: list) -> tuple[list[dict], list[dict]]:
    checks, valid = [], []
    accepted_refs: set[str] = set()
    for index, row in enumerate(questions, 1):
        try:
            errors, diagnostics = module.check_candidate(
                row, root_id, index, source, context, prior + valid, v8)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            errors, diagnostics = [f"candidate_checker_exception:{type(error).__name__}"], []
        if index > 3:
            errors.append("candidate_count_exceeds_three")
        refs = safe_refs(module, row)
        if refs & accepted_refs:
            errors.append("overlap_with_accepted_candidate")
        if not errors:
            valid.append(row)
            accepted_refs.update(refs)
        checks.append({"index": index,
                       "candidate_id": row.get("candidate_id") if isinstance(row, dict) else None,
                       "errors": errors, "diagnostics": diagnostics,
                       "mechanically_valid": not errors})
    return checks, valid


def revalidate(root_id: str) -> dict:
    module = load_v16()
    _, v8, batch, directory, manifest, source, context, prior, _, _, _ = module.inputs(root_id)
    ordering = batch.ordering
    response_path = directory / "RESPONSE.json"
    response = ordering.read_json(response_path)
    if (response.get("source_sha256") != manifest["source_sha256"] or
            response.get("payload_sha256") != manifest["payload_sha256"]):
        raise ValueError("response_binding_changed")
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata_missing")
    try:
        parsed = json.loads(response["raw_text"])
    except (TypeError, ValueError):
        parsed = None
    errors, diagnostics = [], []
    if not isinstance(parsed, dict):
        errors.append("response_not_json_object")
        questions = []
        abstentions = None
    else:
        if set(parsed) != {"questions", "abstentions"}:
            diagnostics.append("top_level_schema_mismatch")
        questions = parsed.get("questions")
        abstentions = parsed.get("abstentions")
        if not isinstance(questions, list):
            errors.append("questions_array_missing")
            questions = []
    if not isinstance(abstentions, list) or any(not isinstance(x, str) for x in abstentions):
        diagnostics.append("abstentions_array_invalid")
    elif len(questions) < 3 and not abstentions:
        diagnostics.append("missing_abstention_reason")
    if metadata.get("returned_model") != batch.MODEL:
        errors.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        errors.append("response_truncated")
    checks, valid = inspect_questions(module, v8, root_id, source, context, prior, questions)
    if errors:
        valid = []
    status = ("GLOBAL_RESPONSE_INVALID" if errors else
              "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if valid else
              "MECHANICAL_REJECTED" if questions else "NO_CANDIDATE_ABSTAINED")
    result = {"version": HERE.name, "root_id": root_id, "status": status,
              "response_path": str(response_path), "response_sha256": sha(response_path.read_bytes()),
              "requested_model": batch.MODEL, "returned_model": metadata.get("returned_model"),
              "candidate_count": len(questions), "mechanically_valid_count": len(valid),
              "fatal_response_errors": errors, "response_diagnostics": diagnostics,
              "checks": checks, "validated_candidate_ids": [q["candidate_id"] for q in valid],
              "semantic_gold_verified": False, "solver_evaluated": False,
              "selected_for_training": False, "training_started": False}
    out = HERE / root_id
    ordering.write_new(out / "REVALIDATION.json", result)
    ordering.write_new(out / "MECHANICAL_CANDIDATES.json", {
        "status": status, "response_sha256": result["response_sha256"],
        "questions": valid, "content_audit_pending": True})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_id", choices=ROOTS)
    args = parser.parse_args()
    print(json.dumps(revalidate(args.root_id), ensure_ascii=False))
