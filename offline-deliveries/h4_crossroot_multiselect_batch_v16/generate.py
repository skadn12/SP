"""One frozen two-question request per root, with per-candidate validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
V12_PATH = PARENT / "h4_crossroot_multiselect_batch_v12/generate.py"
V12_SHA256 = "ef450533d4ef8356e54233accc53a58c148cae8f0391122feaa3f3cc8afb6025"
OUTPUT_LIMIT = 12288
LABELS = tuple("ABCDEF")
PRIOR_PATHS = (
    "h4_crossroot_multiselect_solver_report_two_roots_v1/SELECTED_QUESTIONS.json",
    "h4_crossroot_multiselect_codex_solver_v15_report/SELECTED_QUESTIONS.json",
    "h4_crossroot_multiselect_batch_v8/{root}/PARSED_RESPONSE.json",
    "h4_crossroot_multiselect_batch_v12/{root}/slot-02/PARSED_RESPONSE.json",
    "h4_crossroot_multiselect_batch_v12/{root}/slot-03/PARSED_RESPONSE.json",
    "h4_crossroot_multiselect_batch_v13/{root}/slot-04/PARSED_RESPONSE.json",
    "h4_crossroot_multiselect_targeted_revision_v14/{root}/slot-04/PARSED_RESPONSE.json",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def base():
    if sha(V12_PATH.read_bytes()) != V12_SHA256:
        raise ValueError("v12_dependency_changed")
    spec = importlib.util.spec_from_file_location("v16_frozen_v12", V12_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v8, batch = module.dependency()
    return module, v8, batch


def candidate_id(root_id: str, index: int) -> str:
    return f"{root_id.rsplit('-', 1)[-1]}-qwen36-multi-{index + 5:02d}"


def evidence_items(question: dict) -> list[dict]:
    evidence = question.get("option_evidence")
    if isinstance(evidence, dict):
        entries = evidence.values()
    elif isinstance(evidence, list):
        entries = evidence
    else:
        entries = ()
    return [item for item in entries if isinstance(item, dict)]


def prior_context(root_id: str) -> tuple[dict, list[dict]]:
    records, questions = [], []
    for template in PRIOR_PATHS:
        path = PARENT / template.format(root=root_id)
        raw = path.read_bytes()
        parsed = json.loads(raw)
        rows = parsed if isinstance(parsed, list) else parsed.get("questions", [])
        if not isinstance(rows, list):
            raise ValueError(f"prior_questions_invalid:{path}")
        owned = [q for q in rows if isinstance(q, dict) and q.get("root_id") == root_id]
        records.append({"path": str(path), "sha256": sha(raw), "question_count": len(owned)})
        questions.extend(owned)
    refs = sorted({item["turn_ref"] for q in questions for item in evidence_items(q)
                   if isinstance(item.get("turn_ref"), str)})
    selected = questions[:sum(record["question_count"] for record in records[:2])]
    patterns = [{"candidate_id": q.get("candidate_id"), "gold": "".join(sorted(q["gold"]))}
                for q in selected if isinstance(q.get("gold"), list)]
    themes = sorted({str(q.get("topic", "")).strip() for q in questions if q.get("topic")})
    return {"source_records": records, "forbidden_turn_refs": refs,
            "prior_selected_gold_patterns": patterns, "forbidden_themes": themes}, questions


def generation_view(source: dict, context: dict) -> dict:
    forbidden = set(context["forbidden_turn_refs"])
    turns = [{"turn_ref": t["turn_ref"], "speaker": t["speaker"], "text": t["text"]}
             for t in source["turns"] if t["turn_ref"].startswith("R02:")
             and t["turn_ref"] not in forbidden]
    if len(turns) < 18:
        raise ValueError("fewer_than_eighteen_unused_r02_turns")
    return {"generation_view_type": "unused_r02_turns_compact_prior_v16",
            "root_id": source["root_id"], "turn_count": len(turns), "turns": turns}


def profile(batch) -> dict:
    result = batch.profile()
    result["profile_id"] = HERE.name
    result["output_limit"] = OUTPUT_LIMIT
    result["request_parameters"]["max_tokens"] = OUTPUT_LIMIT
    return result


def prompt_for(root_id: str, context: dict, view: dict) -> str:
    instructions = (HERE / "instructions.txt").read_text()
    instructions = instructions.replace("{ROOT_ID}", root_id)
    instructions = instructions.replace("{ID_ONE}", candidate_id(root_id, 1))
    instructions = instructions.replace("{ID_TWO}", candidate_id(root_id, 2))
    instructions = instructions.replace("{ID_THREE}", candidate_id(root_id, 3))
    compact = {key: context[key] for key in
               ("forbidden_turn_refs", "prior_selected_gold_patterns", "forbidden_themes")}
    return instructions + json.dumps({"prior_questions_avoid_repeating": compact,
                                      "source": view}, ensure_ascii=False, indent=2) + "\n"


def prepare(root_id: str) -> dict:
    module, _, batch = base()
    if root_id not in batch.ROOTS:
        raise ValueError("unknown_root")
    directory = HERE / root_id
    if directory.exists():
        raise FileExistsError(f"root_exists_no_overwrite:{directory}")
    ordering = batch.ordering
    raw, source = ordering.checked_source(root_id, ordering.source_manifest()[root_id], frozen=False)
    context, _ = prior_context(root_id)
    view = generation_view(source, context)
    prompt = prompt_for(root_id, context, view)
    request = ordering.build_payload(profile(batch), prompt)
    manifest = {"version": HERE.name, "root_id": root_id, "model": batch.MODEL,
                "source_sha256": sha(raw), "prior_context_sha256": sha(ordering.canonical(context)),
                "instructions_sha256": sha((HERE / "instructions.txt").read_bytes()),
                "script_sha256": sha(Path(__file__).read_bytes()),
                "view_sha256": sha(ordering.canonical(view)),
                "prompt_sha256": sha(prompt.encode()), "payload_sha256": sha(ordering.canonical(request)),
                "maximum_candidates": 3, "maximum_http_attempts": 1,
                "status": "PREPARED_NO_API_CALL", "solver_calls": 0, "training_started": False}
    directory.mkdir(parents=True)
    ordering.write_new(directory / "source.json", raw.decode(), raw=True)
    ordering.write_new(directory / "prior_context.json", context)
    ordering.write_new(directory / "generation_view.json", view)
    ordering.write_new(directory / "prompt.txt", prompt, raw=True)
    ordering.write_new(directory / "request.json", request)
    ordering.write_new(directory / "PREPARED.json", manifest)
    return {"root_id": root_id, "status": manifest["status"], "source_turns": len(view["turns"]),
            "prompt_bytes": len(prompt.encode()), "payload_sha256": manifest["payload_sha256"]}


def inputs(root_id: str):
    module, v8, batch = base()
    ordering = batch.ordering
    directory = HERE / root_id
    manifest = ordering.read_json(directory / "PREPARED.json")
    if (manifest.get("script_sha256") != sha(Path(__file__).read_bytes()) or
            manifest.get("instructions_sha256") != sha((HERE / "instructions.txt").read_bytes())):
        raise ValueError("generator_freeze_changed")
    raw, source = ordering.checked_source(root_id, ordering.source_manifest()[root_id], frozen=False)
    context, questions = prior_context(root_id)
    view = generation_view(source, context)
    prompt = prompt_for(root_id, context, view)
    request = ordering.build_payload(profile(batch), prompt)
    if (raw != (directory / "source.json").read_bytes() or
            context != ordering.read_json(directory / "prior_context.json") or
            view != ordering.read_json(directory / "generation_view.json") or
            prompt != (directory / "prompt.txt").read_text() or
            request != ordering.read_json(directory / "request.json") or
            manifest.get("source_sha256") != sha(raw) or
            manifest.get("prior_context_sha256") != sha(ordering.canonical(context)) or
            manifest.get("view_sha256") != sha(ordering.canonical(view)) or
            manifest.get("prompt_sha256") != sha(prompt.encode()) or
            manifest.get("payload_sha256") != sha(ordering.canonical(request))):
        raise ValueError("frozen_inputs_changed")
    return module, v8, batch, directory, manifest, source, context, questions, view, prompt, request


def check_candidate(row: object, root_id: str, index: int, source: dict,
                    context: dict, prior: list[dict], v8) -> tuple[list[str], list[str]]:
    if not isinstance(row, dict):
        return ["candidate_not_object"], []
    old_id = f"{root_id.rsplit('-', 1)[-1]}-qwen36-multi-01"
    errors = v8.check_candidate(dict(row, candidate_id=old_id), root_id, source["turns"])
    diagnostics = []
    if row.get("candidate_id") != candidate_id(root_id, index):
        errors.append("candidate_id_invalid_for_position")
    gold = row.get("gold")
    if v8.valid_gold(gold):
        pattern = "".join(sorted(gold))
        if len(gold) < 6 and pattern == "ABCDEF"[:len(gold)]:
            errors.append("gold_prefix_forbidden")
        if any(len(item["gold"]) == len(gold) and item["gold"] == pattern
               for item in context["prior_selected_gold_patterns"]):
            diagnostics.append("prior_selected_gold_pattern_repeated_same_cardinality")
    evidence = row.get("option_evidence")
    if isinstance(evidence, dict) and set(evidence) == set(LABELS):
        refs = [item.get("turn_ref") if isinstance(item, dict) else None
                for item in evidence.values()]
        allowed = {turn["turn_ref"] for turn in generation_view(source, context)["turns"]}
        if any(ref not in allowed for ref in refs):
            errors.append("anchor_not_in_unused_r02_view")
        if len(set(refs)) != 6:
            errors.append("six_distinct_evidence_turns_required")
    options = row.get("options")
    if isinstance(options, dict) and set(options) == set(LABELS):
        normalized = {value.strip().casefold() for value in options.values() if isinstance(value, str)}
        prior_text = {value.strip().casefold() for q in prior
                      for value in (q.get("options") or {}).values() if isinstance(value, str)}
        if normalized & prior_text:
            errors.append("prior_option_text_reused")
    return errors, diagnostics


def validate(root_id: str) -> dict:
    _, v8, batch, directory, manifest, source, context, prior, _, _, _ = inputs(root_id)
    ordering = batch.ordering
    response = ordering.read_json(directory / "RESPONSE.json")
    if (response.get("source_sha256") != manifest["source_sha256"] or
            response.get("payload_sha256") != manifest["payload_sha256"]):
        raise ValueError("response_binding_changed")
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata_invalid")
    try:
        parsed = json.loads(response["raw_text"])
    except (TypeError, ValueError):
        parsed = None
    global_errors = []
    if not isinstance(parsed, dict) or set(parsed) != {"questions", "abstentions"}:
        global_errors.append("top_level_schema_mismatch")
    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    abstentions = parsed.get("abstentions") if isinstance(parsed, dict) else None
    if not isinstance(questions, list):
        questions = []
        global_errors.append("questions_array_missing")
    if not isinstance(abstentions, list) or any(not isinstance(x, str) for x in abstentions):
        global_errors.append("abstentions_array_invalid")
    elif len(questions) < 3 and not abstentions:
        global_errors.append("missing_abstention_reason")
    if metadata.get("returned_model") != batch.MODEL:
        global_errors.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        global_errors.append("response_truncated")
    checks, valid, prior_new = [], [], list(prior)
    earlier_refs = set()
    for index, row in enumerate(questions, 1):
        errors, diagnostics = check_candidate(row, root_id, index, source, context, prior_new, v8)
        if index > 3:
            errors.append("candidate_count_exceeds_three")
        if isinstance(row, dict):
            refs = {item.get("turn_ref") for item in evidence_items(row)}
            if refs & earlier_refs:
                errors.append("cross_question_anchor_overlap")
            earlier_refs.update(refs)
        if isinstance(row, dict):
            prior_new.append(row)
        checks.append({"index": index, "candidate_id": row.get("candidate_id") if isinstance(row, dict) else None,
                       "errors": errors, "diagnostics": diagnostics})
        if not errors and not global_errors:
            valid.append(row)
    status = ("GLOBAL_RESPONSE_INVALID" if global_errors else
              "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if valid else
              "MECHANICAL_REJECTED" if questions else "NO_CANDIDATE_ABSTAINED")
    provenance = {"author": "REMOTE_LARGE_MODEL", "provider": "siliconflow",
                  "requested_model": batch.MODEL, "returned_model": metadata.get("returned_model"),
                  "response_id": (metadata.get("raw_response") or {}).get("id"),
                  "root_id": root_id, "source_sha256": manifest["source_sha256"],
                  "prompt_sha256": manifest["prompt_sha256"],
                  "payload_sha256": manifest["payload_sha256"],
                  "model_questions_modified_by_codex": False, "semantic_gold_verified": False,
                  "solver_evaluated": False, "selected_for_training": False, "training_started": False}
    result = {"status": status, "root_id": root_id, "candidate_count": len(questions),
              "mechanically_valid_count": len(valid), "global_errors": global_errors,
              "checks": checks, "provenance": provenance}
    ordering.write_new(directory / "PARSED_RESPONSE.json", parsed)
    ordering.write_new(directory / "VALIDATED_CANDIDATES.json", {
        "status": status, "provenance": provenance, "questions": valid})
    ordering.write_new(directory / "VALIDATION.json", result)
    return result


def execute(root_id: str) -> dict:
    _, _, batch, directory, manifest, _, _, _, _, prompt, request = inputs(root_id)
    ordering = batch.ordering
    ordering.write_new(directory / "ATTEMPT_RESERVED.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": batch.MODEL,
        "payload_sha256": manifest["payload_sha256"], "maximum_http_attempts": 1})

    def on_attempt(actual: dict) -> None:
        if actual != request:
            raise ValueError("actual_payload_changed")
        ordering.write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "payload_sha256": sha(ordering.canonical(actual)),
            "model": actual["model"], "max_tokens": actual["max_tokens"]})

    try:
        response = ordering.credential_client(profile(batch), on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=OUTPUT_LIMIT, json_object=True)
    except Exception as error:
        result = {"status": "ERROR_NO_RETRY", "root_id": root_id,
                  "error_type": type(error).__name__, "metadata": getattr(error, "metadata", None)}
        ordering.write_new(directory / "ERROR.json", result)
        return result
    ordering.write_new(directory / "RESPONSE.json", {
        "source_sha256": manifest["source_sha256"],
        "payload_sha256": manifest["payload_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata})
    return validate(root_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "validate"))
    parser.add_argument("root_id", choices=(
        "cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852"))
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.root_id)
    elif args.action == "preflight":
        _, _, _, _, manifest, _, _, _, view, prompt, _ = inputs(args.root_id)
        result = {"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL", "root_id": args.root_id,
                  "available_turns": len(view["turns"]), "prompt_bytes": len(prompt.encode()),
                  "payload_sha256": manifest["payload_sha256"]}
    elif args.action == "execute":
        result = execute(args.root_id)
    else:
        result = validate(args.root_id)
    print(json.dumps(result, ensure_ascii=False))
